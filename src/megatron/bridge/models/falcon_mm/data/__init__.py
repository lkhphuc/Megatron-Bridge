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

from megatron.bridge.models.falcon_mm.data.cookers import (
    FALCON_CRUDE_TYPE_CAPTION,
    FALCON_CRUDE_TYPE_KEY,
    FALCON_CRUDE_TYPE_MINT,
    FALCON_CRUDE_TYPE_TEXT,
    FALCON_DEFAULT_MIN_PIXELS,
    FALCON_DEFAULT_PATCH_SIZE,
    FALCON_DEFAULT_SPATIAL_MERGE_SIZE,
    FALCON_DEFAULT_TEMPORAL_PATCH_SIZE,
    FalconDocumentSample,
    FalconPendingSample,
    cook_caption_pair,
    cook_mint_interleaved,
    cook_text_document,
)
from megatron.bridge.models.falcon_mm.data.energon import (
    FalconDeepseekv4EnergonTaskEncoderConfig,
    FalconEnergonTaskEncoderConfig,
    FalconTaskEncoder,
)
from megatron.bridge.models.falcon_mm.data.visual_markers import (
    FALCON_IMAGE_MARKER,
    IMAGE_MARKER_OVERHEAD_TOKENS,
    VisualMarkerIds,
)


__all__ = [
    "FALCON_CRUDE_TYPE_CAPTION",
    "FALCON_CRUDE_TYPE_KEY",
    "FALCON_CRUDE_TYPE_MINT",
    "FALCON_CRUDE_TYPE_TEXT",
    "FALCON_DEFAULT_MIN_PIXELS",
    "FALCON_DEFAULT_PATCH_SIZE",
    "FALCON_DEFAULT_SPATIAL_MERGE_SIZE",
    "FALCON_DEFAULT_TEMPORAL_PATCH_SIZE",
    "FALCON_IMAGE_MARKER",
    "IMAGE_MARKER_OVERHEAD_TOKENS",
    "FalconDeepseekv4EnergonTaskEncoderConfig",
    "FalconDocumentSample",
    "FalconEnergonTaskEncoderConfig",
    "FalconPendingSample",
    "FalconTaskEncoder",
    "VisualMarkerIds",
    "cook_caption_pair",
    "cook_mint_interleaved",
    "cook_text_document",
]
