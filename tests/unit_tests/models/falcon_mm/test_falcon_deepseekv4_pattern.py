# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for DSv4 hybrid layer-pattern generation."""

import json
from pathlib import Path

import pytest

from megatron.bridge.models.falcon_mm.falcon_deepseekv4_pattern import falcon_deepseekv4_layer_pattern


pytestmark = pytest.mark.unit

_DSV4_FLASH_CONFIG = Path("FALCON_MM_AGENT_NOTES/assets/DeepSeek-V4-Flash/config.json")


def test_letter_mapping_pp1_example() -> None:
    assert falcon_deepseekv4_layer_pattern([0, 4, 128, 4]) == "WECEHECE"


@pytest.mark.parametrize(
    ("compress_ratio", "expected_letter"),
    [
        (0, "W"),
        (4, "C"),
        (128, "H"),
    ],
)
def test_letter_mapping_single_layer(compress_ratio: int, expected_letter: str) -> None:
    pattern = falcon_deepseekv4_layer_pattern([compress_ratio])
    assert pattern == f"{expected_letter}E"


def test_unknown_compress_ratio_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported compress_ratio 2 at layer index 0"):
        falcon_deepseekv4_layer_pattern([2])


def test_empty_compress_ratios_raises() -> None:
    with pytest.raises(ValueError, match="compress_ratios must not be empty"):
        falcon_deepseekv4_layer_pattern([])


def test_pp2_even_split() -> None:
    assert falcon_deepseekv4_layer_pattern([0, 0, 4, 128], pipeline_model_parallel_size=2) == "WEWE|CEHE"


def test_pp2_odd_split_stage_lengths() -> None:
    config = json.loads(_DSV4_FLASH_CONFIG.read_text(encoding="utf-8"))
    compress_ratios = config["compress_ratios"][: config["num_hidden_layers"]]
    assert len(compress_ratios) == 43

    pattern = falcon_deepseekv4_layer_pattern(compress_ratios, pipeline_model_parallel_size=2)
    stage_segments = pattern.split("|")
    assert len(stage_segments) == 2
    assert len(stage_segments[0]) // 2 == 22
    assert len(stage_segments[1]) // 2 == 21


def test_pp4_split_stage_lengths() -> None:
    config = json.loads(_DSV4_FLASH_CONFIG.read_text(encoding="utf-8"))
    compress_ratios = config["compress_ratios"][: config["num_hidden_layers"]]

    pattern = falcon_deepseekv4_layer_pattern(compress_ratios, pipeline_model_parallel_size=4)
    stage_segments = pattern.split("|")
    assert len(stage_segments) == 4
    stage_layer_counts = [len(segment) // 2 for segment in stage_segments]
    assert stage_layer_counts == [11, 11, 11, 10]


def test_hash_constraint_raises_when_stage0_too_small() -> None:
    config = json.loads(_DSV4_FLASH_CONFIG.read_text(encoding="utf-8"))
    compress_ratios = config["compress_ratios"][: config["num_hidden_layers"]]

    with pytest.raises(
        ValueError,
        match=r"Pipeline stage 0 has 2 decoder layers but num_hash_layers=3",
    ):
        falcon_deepseekv4_layer_pattern(
            compress_ratios,
            pipeline_model_parallel_size=22,
            num_hash_layers=3,
        )


def test_mtp_num_layers_1() -> None:
    assert falcon_deepseekv4_layer_pattern([0], mtp_num_layers=1) == "WE/WE"


def test_mtp_num_layers_2() -> None:
    assert falcon_deepseekv4_layer_pattern([0], mtp_num_layers=2) == "WE/WE/WE"


def test_dsv4_flash_shape_pp1() -> None:
    config = json.loads(_DSV4_FLASH_CONFIG.read_text(encoding="utf-8"))
    compress_ratios = config["compress_ratios"][: config["num_hidden_layers"]]
    assert len(compress_ratios) == 43

    pattern = falcon_deepseekv4_layer_pattern(compress_ratios)
    assert len(pattern) == 86
    assert set(pattern) <= {"W", "C", "H", "E"}


def test_invalid_pipeline_model_parallel_size_raises() -> None:
    with pytest.raises(ValueError, match="pipeline_model_parallel_size must be >= 1"):
        falcon_deepseekv4_layer_pattern([0], pipeline_model_parallel_size=0)


def test_invalid_mtp_num_layers_raises() -> None:
    with pytest.raises(ValueError, match="mtp_num_layers must be >= 0"):
        falcon_deepseekv4_layer_pattern([0], mtp_num_layers=-1)
