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

"""Encoder-free Falcon multimodal models.

Two trunk variants share the encoder-free patch embed (Conv3d patchify +
hMLP merger) and the placeholder-span scatter contract:

- :class:`FalconQwen35Model` reuses :class:`Qwen3VLModel` wholesale —
  placeholder-span scatter, MRoPE x THD x CP, per-stage packed metadata
  handling — and replaces only the vision tower with the patch embed.
- :class:`FalconDeepseekv4Model` composes the patch embed with a
  Megatron-Core ``HybridModel`` DSv4 trunk, following the Nemotron Omni
  wrapper pattern.
"""

from megatron.bridge.models.falcon_mm.modelling_falcon_mm.model import (
    FalconPatchEmbedVisionModel,
    FalconQwen35Model,
)
from megatron.bridge.models.falcon_mm.modelling_falcon_mm.model_deepseekv4 import (
    FalconDeepseekv4Model,
)


__all__ = ["FalconDeepseekv4Model", "FalconPatchEmbedVisionModel", "FalconQwen35Model"]
