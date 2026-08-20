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

"""Encoder-free Falcon multimodal model.

The model reuses :class:`Qwen3VLModel` wholesale — placeholder-span scatter,
MRoPE x THD x CP, per-stage packed metadata handling — and replaces only the
vision tower with an encoder-free patch embed (Conv3d patchify + hMLP merger).
See ``natively-multimodal-300b-feasibility.md`` (D1) for the design.
"""

from megatron.bridge.models.falcon_mm.modelling_falcon_mm.model import (
    FalconMMModel,
    FalconPatchEmbedVisionModel,
)


__all__ = ["FalconMMModel", "FalconPatchEmbedVisionModel"]
