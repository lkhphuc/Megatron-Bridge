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

"""Unit tests for Falcon DSv4 explicit hybrid patterns."""

import pytest

from megatron.bridge.models.falcon_mm.falcon_deepseekv4_pattern import (
    falcon_deepseekv4_pattern_from_compress_ratios,
    falcon_deepseekv4_pattern_metadata,
    falcon_deepseekv4_pipeline_pattern,
)


pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("pattern", "expected_ratios", "expected_moe_freq"),
    [
        ("W-C-H-", [0, 4, 128], [0, 0, 0]),
        ("WECEHE", [0, 4, 128], [1, 1, 1]),
        ("W-C-|H-", [0, 4, 128], [0, 0, 0]),
        ("W-CEHE", [0, 4, 128], [0, 1, 1]),
    ],
)
def test_pattern_metadata(pattern: str, expected_ratios: list[int], expected_moe_freq: list[int]) -> None:
    assert falcon_deepseekv4_pattern_metadata(pattern) == (expected_ratios, expected_moe_freq)


@pytest.mark.parametrize("pattern", ["", "W", "W*", "WC"])
def test_pattern_metadata_rejects_malformed_patterns(pattern: str) -> None:
    with pytest.raises(ValueError):
        falcon_deepseekv4_pattern_metadata(pattern)


def test_pattern_from_compress_ratios() -> None:
    assert falcon_deepseekv4_pattern_from_compress_ratios([0, 4, 128]) == "WECEHE"
    assert falcon_deepseekv4_pattern_from_compress_ratios([0, 4], ffn_symbol="-") == "W-C-"


def test_pipeline_pattern_preserves_decoder_pairs() -> None:
    pattern = falcon_deepseekv4_pipeline_pattern(
        "W-W-C-H-C-H-C-W-",
        pipeline_model_parallel_size=4,
    )
    assert pattern == "W-W-|C-H-|C-H-|C-W-"


def test_hash_constraint_raises_when_stage0_too_small() -> None:
    with pytest.raises(
        ValueError,
        match=r"Pipeline stage 0 has 2 decoder layers but num_hash_layers=3",
    ):
        falcon_deepseekv4_pipeline_pattern(
            "WECEHECEWECEHECE",
            pipeline_model_parallel_size=4,
            num_hash_layers=3,
        )


def test_hash_constraint_requires_leading_moe_symbols() -> None:
    with pytest.raises(ValueError, match="first 2 decoder layers to use MoE"):
        falcon_deepseekv4_pipeline_pattern(
            "W-CEHECE",
            pipeline_model_parallel_size=1,
            num_hash_layers=2,
        )
