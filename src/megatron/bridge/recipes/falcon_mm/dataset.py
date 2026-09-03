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

"""Shared Energon dataset helpers for Falcon-MM recipes."""

from __future__ import annotations

from typing import Any

from megatron.bridge.data.builders.energon import EnergonDatasetConfig
from megatron.bridge.models.falcon_mm.data import (
    FalconDeepseekv4EnergonTaskEncoderConfig,
    FalconEnergonTaskEncoderConfig,
)


def make_falcon_energon_dataset(
    task_encoder: FalconEnergonTaskEncoderConfig | FalconDeepseekv4EnergonTaskEncoderConfig,
    *,
    seq_length: int,
    micro_batch_size: int = 1,
    packing_buffer_size: int = 16,
    num_workers: int = 2,
    shuffle_buffer_size: int = 100,
    max_samples_per_sequence: int = 100,
    **dataset_overrides: Any,
) -> EnergonDatasetConfig:
    """Build the Falcon-MM Energon dataset config (THD Energon-native packing).

    ``path`` is intentionally ``None`` — point it at an indexed Energon dataset
    root or a MetadatasetV2 YAML via CLI: ``dataset.path=<path>``.

    Energon requires ``shuffle_buffer_size`` and ``max_samples_per_sequence``
    to be set explicitly (no API defaults): they trade shuffle randomness
    against sequential read locality on webdataset shards. ``100`` / ``100``
    is Energon's recommended starting point for image-scale samples.

    Pack-then-decode keeps the shuffle buffer cheap (header estimates, not
    ``pixel_values``), so these are shuffle-quality knobs rather than RAM
    panic valves. Override via kwargs or CLI when samples are much larger /
    smaller than typical MINT pages.
    """
    kwargs: dict[str, Any] = {
        "path": None,
        "seq_length": seq_length,
        "micro_batch_size": micro_batch_size,
        "num_workers": num_workers,
        "packing_buffer_size": packing_buffer_size,
        "shuffle_buffer_size": shuffle_buffer_size,
        "max_samples_per_sequence": max_samples_per_sequence,
        "task_encoder": task_encoder,
    }
    kwargs.update(dataset_overrides)
    return EnergonDatasetConfig(**kwargs)
