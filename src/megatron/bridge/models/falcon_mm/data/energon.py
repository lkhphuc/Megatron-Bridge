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

"""Falcon-MM Energon task encoder (THD Energon-native packing only).

The Falcon-MM data plane is Energon-native packing end to end (see
``energon-torchtitan-bridge-port.md``): per-sample work happens in the Energon
worker, and each emitted sample is one complete THD pack. The implementation
currently inherits the Qwen-VL pipeline wholesale — processor placeholder
expansion, assistant loss masking, first-fit-decreasing packing, and the THD
batch builder — which is exactly the contract the encoder-free model consumes.
Per-task (VLM/perception/OCR) sample processing is the planned extension point
(feasibility doc, work item 2) and belongs in ``FalconTaskEncoder`` overrides.

Note on wiring: ``FalconEnergonTaskEncoderConfig`` subclasses
``QwenVLEnergonTaskEncoderConfig`` deliberately. The packing gate
(``data/builders/energon.py``) and the builder's isinstance branch both match
the Qwen config, so the Falcon config is accepted with no shared-code edits;
while ``FalconTaskEncoder`` is a pass-through, the builder constructing a
``QwenVLTaskEncoder`` is functionally identical. When the encoder gains real
behavior, add one isinstance branch for the Falcon config in
``build_energon_task_encoder`` (before the Qwen branch).
"""

from dataclasses import dataclass

from megatron.bridge.data.builders.energon import QwenVLEnergonTaskEncoderConfig
from megatron.bridge.models.qwen_vl.data.energon import QwenVLTaskEncoder


@dataclass(kw_only=True)
class FalconEnergonTaskEncoderConfig(QwenVLEnergonTaskEncoderConfig):
    """Serializable settings for the Falcon-MM Energon task encoder.

    Currently adds no fields over the Qwen-VL config; it exists as the stable
    namespace for Falcon-specific options (per-task caps, two-constraint
    packing, augmentation configs) as they are ported.
    """


class FalconTaskEncoder(QwenVLTaskEncoder):
    """Falcon-MM task encoder. Pass-through for now; override ``encode_sample``
    for per-task processing as it is ported from the torchtitan pipeline."""
