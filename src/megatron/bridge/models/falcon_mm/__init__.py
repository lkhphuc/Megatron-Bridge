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

"""Falcon multimodal (encoder-free) model family."""

from megatron.bridge.models.falcon_mm.falcon_mm_provider import FalconMMModelProvider
from megatron.bridge.models.falcon_mm.modelling_falcon_mm import (
    FalconMMModel,
    FalconPatchEmbedVisionModel,
)


__all__ = [
    "FalconMMModel",
    "FalconMMModelProvider",
    "FalconPatchEmbedVisionModel",
]
