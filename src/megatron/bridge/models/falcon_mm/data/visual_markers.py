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

"""Trunk-agnostic visual marker contract for Falcon-MM.

Every Falcon-MM trunk consumes the same token-stream layout per image::

    [vision_start] [image_pad x N] [vision_end]

where ``N = prod(image_grid_thw) // spatial_merge_size**2`` merged visual
tokens. Each marker is a single special token; only ``image_pad`` is
expanded. The Qwen trunk emits this natively through the HF processor; the
DeepSeek-V4 and Llama3 trunks map the markers onto reserved tokenizer
placeholders.

This module is the single source of truth for the marker literal used in
assembled documents and for resolving the three marker token ids per trunk.
The model-side scatter target is always ``image_pad_id``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any


logger = logging.getLogger(__name__)

# Literal marker inserted per image during interleaved document assembly.
# Processors split text on this literal and expand it to the marker triple;
# it is never tokenized verbatim.
FALCON_IMAGE_MARKER = "<|vision_start|><|image_pad|><|vision_end|>"

# Tokens remaining around each visual span after ``image_pad`` expansion:
# ``vision_start`` and ``vision_end`` (one each). Uniform across trunks.
IMAGE_MARKER_OVERHEAD_TOKENS = 2

QWEN_VISION_START_TOKEN = "<|vision_start|>"
QWEN_IMAGE_PAD_TOKEN = "<|image_pad|>"
QWEN_VISION_END_TOKEN = "<|vision_end|>"

# DeepSeek-V4 reserves 800 ``place▁holder▁no▁N`` tokens (ids 128000-128799)
# that no released chat template or config references. The official V4 vision
# release expands images into out-of-vocabulary sentinel ids instead, so
# these in-vocab placeholders are unclaimed; Falcon-MM assigns the first
# three as the visual markers.
DSV4_IMAGE_PAD_TOKEN = "<｜place▁holder▁no▁0｜>"
DSV4_VISION_START_TOKEN = "<｜place▁holder▁no▁1｜>"
DSV4_VISION_END_TOKEN = "<｜place▁holder▁no▁2｜>"

# Llama3 ships unused ``<|reserved_special_token_N|>`` ids (128002+). Falcon-MM
# assigns the first three as visual markers (pad / start / end).
LLAMA3_IMAGE_PAD_TOKEN = "<|reserved_special_token_0|>"
LLAMA3_VISION_START_TOKEN = "<|reserved_special_token_1|>"
LLAMA3_VISION_END_TOKEN = "<|reserved_special_token_2|>"


@dataclass(frozen=True)
class VisualMarkerIds:
    """Resolved token ids for the visual marker triple of one trunk."""

    vision_start_id: int
    image_pad_id: int
    vision_end_id: int

    def as_tensor_ids(self) -> list[int]:
        """Return all three marker ids as a plain list (for label masking)."""
        return [self.vision_start_id, self.image_pad_id, self.vision_end_id]


def resolve_marker_ids(
    tokenizer: Any,
    *,
    vision_start_token: str,
    image_pad_token: str,
    vision_end_token: str,
) -> VisualMarkerIds:
    """Resolve and validate the three marker token ids from a tokenizer.

    Raises:
        ValueError: If any marker token is missing, resolves to ``unk``, or
            collides with another marker.
    """
    unk_token_id = getattr(tokenizer, "unk_token_id", None)
    resolved: dict[str, int] = {}
    for role, token in (
        ("vision_start", vision_start_token),
        ("image_pad", image_pad_token),
        ("vision_end", vision_end_token),
    ):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None or (unk_token_id is not None and token_id == unk_token_id):
            raise ValueError(f"Tokenizer must define reserved visual marker {token!r}; got token id {token_id!r}.")
        resolved[role] = int(token_id)
    ids = VisualMarkerIds(
        vision_start_id=resolved["vision_start"],
        image_pad_id=resolved["image_pad"],
        vision_end_id=resolved["vision_end"],
    )
    if len(set(ids.as_tensor_ids())) != 3:
        raise ValueError(f"Visual marker token ids must be distinct; got {ids}.")
    logger.info(
        "Resolved visual marker ids: vision_start=%d image_pad=%d vision_end=%d",
        ids.vision_start_id,
        ids.image_pad_id,
        ids.vision_end_id,
    )
    return ids


def resolve_qwen_marker_ids(tokenizer: Any) -> VisualMarkerIds:
    """Resolve marker ids for the Qwen trunk (native special tokens)."""
    return resolve_marker_ids(
        tokenizer,
        vision_start_token=QWEN_VISION_START_TOKEN,
        image_pad_token=QWEN_IMAGE_PAD_TOKEN,
        vision_end_token=QWEN_VISION_END_TOKEN,
    )


def resolve_dsv4_marker_ids(tokenizer: Any) -> VisualMarkerIds:
    """Resolve marker ids for the DSv4 trunk (reserved placeholders)."""
    return resolve_marker_ids(
        tokenizer,
        vision_start_token=DSV4_VISION_START_TOKEN,
        image_pad_token=DSV4_IMAGE_PAD_TOKEN,
        vision_end_token=DSV4_VISION_END_TOKEN,
    )


def resolve_llama3_marker_ids(tokenizer: Any) -> VisualMarkerIds:
    """Resolve marker ids for the Llama3 trunk (reserved special tokens)."""
    return resolve_marker_ids(
        tokenizer,
        vision_start_token=LLAMA3_VISION_START_TOKEN,
        image_pad_token=LLAMA3_IMAGE_PAD_TOKEN,
        vision_end_token=LLAMA3_VISION_END_TOKEN,
    )
