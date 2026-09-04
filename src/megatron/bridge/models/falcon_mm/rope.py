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

"""Qwen-compatible T/H/W mRoPE helpers for the Falcon-MM DSv4 trunk.

Reuses :func:`megatron.bridge.models.qwen_vl.modelling_qwen3_vl.rope.get_rope_index`
with Falcon DSv4 visual marker ids. Multimodal mode is gated by
``mrope_section is not None`` on the MCore config; ``position_embedding_type``
stays ``\"rope\"``.
"""

from __future__ import annotations

from typing import Optional

import torch
from megatron.core.packed_seq_params import PackedSeqParams

from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.rope import get_rope_index


# Provisional Qwen3.5-compatible section for R=64 (F=32 frequency pairs).
FALCON_DSV4_DEFAULT_MROPE_SECTION: list[int] = [11, 11, 10]

# DeepSeek-V4 reserved placeholders used as Falcon visual markers.
FALCON_DSV4_IMAGE_TOKEN_ID = 128000
FALCON_DSV4_VISION_START_TOKEN_ID = 128001
# Video is rejected by FalconDeepseekv4Model; use an id that never appears.
FALCON_DSV4_UNUSED_VIDEO_TOKEN_ID = -1


def is_mrope_position_ids(position_ids: torch.Tensor | None) -> bool:
    """Return whether ``position_ids`` is explicit T/H/W metadata ``[3, B, S]``."""
    return isinstance(position_ids, torch.Tensor) and position_ids.dim() == 3 and position_ids.size(0) == 3


def expand_scalar_position_ids(position_ids: torch.Tensor) -> torch.Tensor:
    """Expand ``[B, S]`` scalar positions to equal-axis ``[3, B, S]``."""
    if position_ids.dim() != 2:
        raise ValueError(f"Scalar position_ids must be [B, S], got shape {tuple(position_ids.shape)}")
    return position_ids.unsqueeze(0).expand(3, -1, -1).contiguous()


def interleaved_mrope_section_for_rotary_dim(rotary_dim: int) -> list[int]:
    """Return the stride-3 interleaved T/H/W section for ``rotary_dim`` channels.

    Matches MCore's interleaved validation:
    ``((F + 2) // 3, (F + 1) // 3, F // 3)`` where ``F = rotary_dim // 2``.
    """
    if rotary_dim <= 0 or rotary_dim % 2 != 0:
        raise ValueError(f"rotary_dim must be a positive even integer, got {rotary_dim}")
    pairs = rotary_dim // 2
    return [(pairs + 2) // 3, (pairs + 1) // 3, pairs // 3]


def build_falcon_dsv4_mrope_position_ids(
    input_ids: torch.Tensor,
    *,
    spatial_merge_size: int,
    image_token_id: int = FALCON_DSV4_IMAGE_TOKEN_ID,
    vision_start_token_id: int = FALCON_DSV4_VISION_START_TOKEN_ID,
    video_token_id: int = FALCON_DSV4_UNUSED_VIDEO_TOKEN_ID,
    image_grid_thw: Optional[torch.Tensor] = None,
    video_grid_thw: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    packed_seq_params: Optional[PackedSeqParams] = None,
) -> torch.Tensor:
    """Build Qwen-compatible ``[3, B, S]`` T/H/W positions for Falcon DSv4.

    Text-only sequences (no grids) produce equal T/H/W axes. Packed THD packs
    reset coordinates at each logical sample boundary; padding rows stay at
    the safe zero coordinate filled by :func:`get_rope_index`.
    """
    position_ids, _ = get_rope_index(
        spatial_merge_size,
        image_token_id,
        video_token_id,
        vision_start_token_id,
        input_ids=input_ids,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        attention_mask=attention_mask,
        packed_seq_params=packed_seq_params,
    )
    return position_ids


def resolve_falcon_dsv4_mrope_position_ids(
    position_ids: torch.Tensor | None,
    input_ids: torch.Tensor | None,
    *,
    spatial_merge_size: int,
    image_token_id: int,
    vision_start_token_id: int,
    video_token_id: int = FALCON_DSV4_UNUSED_VIDEO_TOKEN_ID,
    image_grid_thw: Optional[torch.Tensor] = None,
    video_grid_thw: Optional[torch.Tensor] = None,
    packed_seq_params: Optional[PackedSeqParams] = None,
    require_global_positions: bool = False,
) -> torch.Tensor | None:
    """Normalize Falcon DSv4 positions for multimodal RoPE.

    Preference order:
    1. Explicit ``[3, B, S]`` metadata (reused as-is).
    2. Recompute from ``input_ids`` + visual grids (ignores ordinary 2D
       dataloader arange rows, matching the Qwen VL contract).
    3. Expand 2D scalar positions to equal axes when recomputation is
       impossible (no ``input_ids``) and global positions are not required.

    Args:
        require_global_positions: When True (contiguous CP with rank-local
            tokens), explicit global ``[3, B, S]`` positions are mandatory.
    """
    if is_mrope_position_ids(position_ids):
        return position_ids

    if require_global_positions:
        raise ValueError(
            "Contiguous CP with Falcon DSv4 mRoPE requires global [3, B, S] "
            "position_ids before token partitioning. Compute them in the VLM "
            "step from the full packed sequence."
        )

    if input_ids is not None:
        return build_falcon_dsv4_mrope_position_ids(
            input_ids,
            spatial_merge_size=spatial_merge_size,
            image_token_id=image_token_id,
            vision_start_token_id=vision_start_token_id,
            video_token_id=video_token_id,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            packed_seq_params=packed_seq_params,
        )

    if isinstance(position_ids, torch.Tensor) and position_ids.dim() == 2:
        return expand_scalar_position_ids(position_ids)

    return None
