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

"""Crude Energon cookers and pending-sample types for Falcon-MM."""

from __future__ import annotations

import json
from typing import Any

from megatron.energon import Sample, basic_sample_keys, stateless
from megatron.energon.edataclass import edataclass

from megatron.bridge.data.energon.task_encoder_utils import ChatMLSample
from megatron.bridge.models.falcon_mm.data.images import (
    ImageDocument,
    ImagePageRef,
    assemble_mint_sequence,
)


FALCON_CRUDE_TYPE_KEY = "crude_type"
FALCON_CRUDE_TYPE_MINT = "mint"
FALCON_CRUDE_TYPE_TEXT = "text"
FALCON_CRUDE_TYPE_CAPTION = "caption"

# Prefer LAION / CC12M ``jpg``; fall back to other single-image WDS members.
_CAPTION_IMAGE_MEMBER_KEYS = ("jpg", "jpeg", "png", "webp")

SUBFLAVOR_MAX_PIXELS = "max_pixels"
SUBFLAVOR_MIN_PIXELS = "min_pixels"
SUBFLAVOR_MAX_NUM_IMAGES = "max_num_images"

# Early-fusion Falcon patchify: ``patch_size=32``, ``merge_size=1``,
# ``temporal_patch_size=1``. ``min_pixels = patch_size**2 * merge_size``
# (1024) is a processor upscale floor, not a drop filter — per-source drop
# policy stays on subflavors.
FALCON_DEFAULT_PATCH_SIZE = 32
FALCON_DEFAULT_SPATIAL_MERGE_SIZE = 1
FALCON_DEFAULT_TEMPORAL_PATCH_SIZE = 1
FALCON_DEFAULT_MIN_PIXELS = FALCON_DEFAULT_PATCH_SIZE**2 * FALCON_DEFAULT_SPATIAL_MERGE_SIZE


@edataclass
class FalconDocumentSample(Sample):
    """Cooked Falcon document before packing-length computation.

    ``sequence`` is text strings plus :class:`ImagePageRef` placeholders.
    Pixels are not materialized here.
    """

    sequence: list[Any]

    def __repr__(self) -> str:
        # Energon restore / error dumps call ``repr(sample)``; keep text short.
        n_text = sum(1 for part in self.sequence if isinstance(part, str))
        n_images = len(self.sequence) - n_text
        preview = next((part[:80] for part in self.sequence if isinstance(part, str) and part), "")
        if preview and len(preview) >= 80:
            preview = preview[:77] + "..."
        return (
            f"FalconDocumentSample(__key__={self.__key__!r}, parts={len(self.sequence)}, "
            f"text_spans={n_text}, images={n_images}, preview={preview!r})"
        )


@edataclass
class FalconPendingSample(Sample):
    """Pre-pack Falcon sample: exact packing length plus deferred media.

    ``preencode_sample`` sets ``packing_length`` from tokenizer + image-header
    geometry (no pixel decode), and ``image_target_sizes`` to the processor's
    smart-resize target per :class:`ImagePageRef`.
    ``postencode_sample`` decodes image refs to those targets, making the processor
    resize a no-op, so the resulting sequence length matches ``packing_length``.

    ``drop_reason`` marks samples that must not train (overlong, too many
    images, …). Preencode returns these instead of raising ``SkipSample`` so
    Energon buffer restore does not die with "Unexpected skip sample".
    Postencode converts ``drop_reason`` into ``SkipSample``.
    """

    packing_length: int
    text: str
    images: list[Any]
    min_pixels: int
    max_pixels: int
    chatml: ChatMLSample | None = None
    image_target_sizes: list[tuple[int, int]] | None = None
    drop_reason: str | None = None

    def __repr__(self) -> str:
        text_preview = self.text[:80] + ("..." if len(self.text) > 80 else "")
        return (
            f"FalconPendingSample(__key__={self.__key__!r}, packing_length={self.packing_length}, "
            f"images={len(self.images)}, drop_reason={self.drop_reason!r}, text={text_preview!r})"
        )


def _parse_json_member(payload: Any) -> Any:
    """Decode a crude ``json`` member that may still be raw bytes."""
    if isinstance(payload, (bytes, bytearray)):
        return json.loads(payload)
    if isinstance(payload, str):
        return json.loads(payload)
    return payload


def _parse_image_member(payload: Any) -> ImageDocument:
    """Accept raw image bytes or an :class:`ImageDocument`."""
    if isinstance(payload, ImageDocument):
        return payload
    if isinstance(payload, (bytes, bytearray)):
        return ImageDocument.from_bytes(bytes(payload))
    raise TypeError(f"Image payload must be bytes or ImageDocument; got {type(payload).__name__}")


@stateless
def cook_mint_interleaved(sample: dict[str, Any]) -> FalconDocumentSample:
    """Cook a crude MINT ``json`` + ``tiff`` sample into a document sequence."""
    meta = _parse_json_member(sample["json"])
    if not isinstance(meta, dict):
        raise ValueError(
            f"MINT json member must be a dict; got {type(meta).__name__} for key {sample.get('__key__')!r}."
        )
    sequence = assemble_mint_sequence(
        texts=meta.get("texts") or [],
        images=meta.get("images") or [],
        captions=meta.get("captions") or [],
        tiff_payload=_parse_image_member(sample["tiff"]),
        sample_key=sample.get("__key__"),
    )
    return FalconDocumentSample(
        **basic_sample_keys(sample),
        sequence=sequence,
    )


@stateless
def cook_text_document(sample: dict[str, Any]) -> FalconDocumentSample:
    """Cook a JSONL crude sample ``{"text": ...}`` into a text-only document."""
    payload = _parse_json_member(sample.get("json"))
    if not isinstance(payload, dict) or not isinstance(payload.get("text"), str):
        raise ValueError(
            "JSONL text sample must be a dict with a string 'text' key; "
            f"got {type(payload).__name__} for key {sample.get('__key__')!r}."
        )
    return FalconDocumentSample(
        **basic_sample_keys(sample),
        sequence=[payload["text"]],
    )


def _parse_caption_text(payload: Any) -> str:
    """Decode a crude ``txt`` member to a stripped UTF-8 caption string."""
    if isinstance(payload, (bytes, bytearray)):
        text = bytes(payload).decode("utf-8")
    elif isinstance(payload, str):
        text = payload
    else:
        raise TypeError(f"Caption txt member must be bytes or str; got {type(payload).__name__}")
    return text.strip()


def _extract_caption_image_payload(sample: dict[str, Any]) -> Any:
    """Return the first present image member among common WebDataset suffixes."""
    for key in _CAPTION_IMAGE_MEMBER_KEYS:
        if key in sample and sample[key] is not None:
            return sample[key]
    raise ValueError(
        "Caption sample must include one of "
        f"{_CAPTION_IMAGE_MEMBER_KEYS}; got keys {sorted(sample)!r} for "
        f"{sample.get('__key__')!r}."
    )


@stateless
def cook_caption_pair(sample: dict[str, Any]) -> FalconDocumentSample:
    """Cook a crude image+caption WebDataset sample (CC12M: ``jpg`` + ``txt``).

    Builds ``[ImagePageRef, caption]`` so the encoder emits image markers then
    the caption text. Optional ``json`` metadata is ignored.
    """
    if "txt" not in sample or sample["txt"] is None:
        raise ValueError(f"Caption sample missing 'txt' member for key {sample.get('__key__')!r}.")
    caption = _parse_caption_text(sample["txt"])
    if not caption:
        raise ValueError(f"Caption sample has empty 'txt' for key {sample.get('__key__')!r}.")
    document = _parse_image_member(_extract_caption_image_payload(sample))
    if len(document.page_sizes) != 1:
        raise ValueError(
            f"Caption image must be a single page; got {len(document.page_sizes)} pages "
            f"for key {sample.get('__key__')!r}."
        )
    return FalconDocumentSample(
        **basic_sample_keys(sample),
        sequence=[ImagePageRef(document, 0), caption],
    )


def resolve_subflavor_int(
    subflavors: dict[str, Any] | None,
    key: str,
    default: int | None,
) -> int | None:
    """Return an integer subflavor override, or ``default`` when unset."""
    if not subflavors or key not in subflavors or subflavors[key] is None:
        return default
    return int(subflavors[key])
