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

Per-sample work is split so packing can group samples without decoding pixels:

- Crude cookers assemble MINT / JSONL documents without pixel decode.
- ``preencode_sample`` (Energon pre-pack map) computes an **exact** packing length
  from the tokenizer plus image-header geometry / ``min_pixels`` /
  ``max_pixels`` / patch-merge config (and optional ``max_num_images``), and
  records the processor's smart-resize target per image page.
- ``postencode_sample`` decodes pages to exactly those targets (making the
  processor's own resize a no-op) and runs the processor only for samples
  selected into a pack; the prepared sequence length must match the preencode
  packing length.

Implementation inherits the Qwen-VL first-fit-decreasing packer and THD batch
builder, and adds a PRETRAINING path: plain interleaved text-images (no ChatML
template, no assistant loss masking), with full LM loss on every token except
visual/special tokens.

``ChatMLSample`` is accepted for the mock path: packing length in
``preencode_sample``, then the Qwen encoder in ``postencode_sample``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

import torch
from megatron.energon import Cooker, SkipSample, TaskEncoder, stateless
from PIL import Image

from megatron.bridge.data.builders.energon import QwenVLEnergonTaskEncoderConfig
from megatron.bridge.data.collators.sequence_padding import use_processor_right_padding
from megatron.bridge.data.datasets.utils import IGNORE_INDEX
from megatron.bridge.data.energon.task_encoder_utils import ChatMLSample
from megatron.bridge.data.packing.algorithms import first_fit_decreasing
from megatron.bridge.data.token_utils import extract_skipped_token_ids
from megatron.bridge.models.falcon_mm.data.cookers import (
    FALCON_CRUDE_TYPE_KEY,
    FALCON_CRUDE_TYPE_MINT,
    FALCON_CRUDE_TYPE_TEXT,
    FALCON_DEFAULT_MIN_PIXELS,
    FALCON_DEFAULT_PATCH_SIZE,
    FALCON_DEFAULT_SPATIAL_MERGE_SIZE,
    FALCON_DEFAULT_TEMPORAL_PATCH_SIZE,
    SUBFLAVOR_MAX_NUM_IMAGES,
    SUBFLAVOR_MAX_PIXELS,
    SUBFLAVOR_MIN_PIXELS,
    FalconDocumentSample,
    FalconPendingSample,
    cook_mint_interleaved,
    cook_text_document,
    resolve_subflavor_int,
)
from megatron.bridge.models.falcon_mm.data.deepseek_v4_processor import FalconDeepseekv4Processor
from megatron.bridge.models.falcon_mm.data.images import ImagePageRef, smart_resize
from megatron.bridge.models.falcon_mm.data.visual_markers import (
    FALCON_IMAGE_MARKER,
    IMAGE_MARKER_OVERHEAD_TOKENS,
    VisualMarkerIds,
    resolve_qwen_marker_ids,
)
from megatron.bridge.models.qwen_vl.data.collate_fn import QWEN_VISUAL_KEYS, QwenVLPreparedSequence
from megatron.bridge.models.qwen_vl.data.energon import (
    QwenVLPackedTaskSample,
    QwenVLTaskBatch,
    QwenVLTaskEncoder,
    QwenVLTaskSample,
    _aligned_sequence_length,
    _stateless_with_failure_tolerance,
    _visual_token_count,
)


try:
    from transformers import Qwen3VLImageProcessor as _QwenVLImageProcessor
except ImportError:  # pragma: no cover - pinned transformers may only expose Qwen2VL
    from transformers import Qwen2VLImageProcessor as _QwenVLImageProcessor


if TYPE_CHECKING:
    from megatron.bridge.data.builders.energon import EnergonDatasetConfig


logger = logging.getLogger(__name__)

# Caps skipped-sample warnings per encoder-instance so a stream of overlong
# or oversized documents does not flood the logs (each Energon worker process
# holds its own task-encoder instance). After the cap, every
# ``_SKIP_WARNING_REPORT_EVERY``-th skip is still reported so the data-loss
# rate stays observable.
_SKIP_WARNING_LOG_LIMIT = 32
_SKIP_WARNING_REPORT_EVERY = 1000

# Stable reason keys piggybacked onto successful packs for WandB ``data/skipped-*``.
SKIP_REASON_OVERLONG = "overlong"
SKIP_REASON_MAX_NUM_IMAGES = "max_num_images"
SKIP_REASON_MAX_VISUAL_TOKENS = "max_visual_tokens"
SKIP_REASON_GEOMETRY = "geometry"
SKIP_REASON_DECOMPRESSION_BOMB = "decompression_bomb"
SKIP_REASON_CHATML_OVERLENGTH = "chatml_overlength"
SKIP_REASON_LENGTH_MISMATCH = "length_mismatch"
SKIP_REASON_OVERSIZE_PACK = "oversize_pack"


def visual_tokens_for_resized(
    width: int,
    height: int,
    *,
    patch_size: int,
    merge_size: int,
    temporal_patch_size: int,
) -> int:
    """Merged visual tokens for an already smart-resized ``(width, height)``."""
    return (temporal_patch_size * (height // patch_size) * (width // patch_size)) // (merge_size**2)


def estimate_visual_tokens(
    width: int,
    height: int,
    *,
    patch_size: int,
    merge_size: int,
    temporal_patch_size: int,
    min_pixels: int,
    max_pixels: int,
) -> int:
    """Estimate merged visual tokens for one image after Qwen-style resize.

    ``width`` / ``height`` are PIL-order (same as :func:`smart_resize`).
    """
    resized_width, resized_height = smart_resize(
        width,
        height,
        factor=patch_size * merge_size,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    return visual_tokens_for_resized(
        resized_width,
        resized_height,
        patch_size=patch_size,
        merge_size=merge_size,
        temporal_patch_size=temporal_patch_size,
    )


def _pil_image_size(image: Any) -> tuple[int, int]:
    """Return ``(width, height)`` from a PIL image; reject tensors / other types.

    Torch CHW tensors use ``shape[-2:] == (height, width)`` — the opposite of
    PIL ``Image.size``. Callers must convert to PIL before size-based packing.
    """
    if not isinstance(image, Image.Image):
        raise TypeError(
            f"Expected a PIL Image for size-based packing; got {type(image).__name__}. "
            "Torch tensors are (C, H, W) — convert to PIL first (do not use .size / .shape[-2:])."
        )
    return image.size


def _require_energon_packing(enable_energon_packing: bool) -> None:
    """Reject unpacked configurations: the Falcon encoder is packing-only."""
    if not enable_energon_packing:
        raise ValueError(
            "Falcon-MM Energon task encoder supports only THD Energon-native packing; "
            "set dataset.packing_buffer_size (micro_batch_size=1)."
        )


@dataclass(kw_only=True)
class FalconEnergonTaskEncoderConfig(QwenVLEnergonTaskEncoderConfig):
    """Serializable settings for the Falcon-MM Energon task encoder.

    Early-fusion Falcon defaults: ``patch_size=32``, ``spatial_merge_size=1``
    (no ViT merge), ``temporal_patch_size=1``, and
    ``min_pixels = patch_size**2 * merge_size`` (1024). Per-source pixel /
    image-count overrides still belong on Energon ``subflavors``.
    """

    patch_size: int = FALCON_DEFAULT_PATCH_SIZE
    spatial_merge_size: int = FALCON_DEFAULT_SPATIAL_MERGE_SIZE
    temporal_patch_size: int = FALCON_DEFAULT_TEMPORAL_PATCH_SIZE
    min_pixels: int = FALCON_DEFAULT_MIN_PIXELS

    def _apply_patch_geometry(self, vision: Any) -> None:
        """Write Falcon patch / merge knobs onto an HF vision image processor."""
        vision.patch_size = self.patch_size
        vision.temporal_patch_size = self.temporal_patch_size
        vision.merge_size = self.spatial_merge_size

    def build_task_encoder(
        self,
        config: "EnergonDatasetConfig",
        *,
        effective_packing: bool,
        enable_energon_packing: bool,
        trust_remote_code: bool,
    ) -> Any:
        """Construct the Falcon-MM task encoder (Qwen family + document path)."""
        _require_energon_packing(enable_energon_packing)
        tokenizer, image_processor = self._load_hf_processors(trust_remote_code)
        vision = getattr(image_processor, "image_processor", image_processor)
        self._apply_patch_geometry(vision)
        return FalconTaskEncoder(
            tokenizer=tokenizer,
            image_processor=image_processor,
            **self._task_encoder_kwargs(
                config,
                effective_packing=effective_packing,
                enable_energon_packing=enable_energon_packing,
            ),
        )


@dataclass(kw_only=True)
class FalconDeepseekv4EnergonTaskEncoderConfig(FalconEnergonTaskEncoderConfig):
    """Serializable settings for the Falcon-MM DSv4 Energon task encoder.

    Patch geometry defaults match the Falcon patch embed (``patch_size=32``,
    ``spatial_merge_size=1``, ``temporal_patch_size=1``). Keep the encoder
    and model values aligned when overriding any of them.
    """

    hf_processor_path: str = "deepseek-ai/DeepSeek-V4-Flash"
    ds_tokenizer_path: str = ""
    image_processor_path: str | None = None

    def build_task_encoder(
        self,
        config: "EnergonDatasetConfig",
        *,
        effective_packing: bool,
        enable_energon_packing: bool,
        trust_remote_code: bool,
    ) -> Any:
        """Construct the Falcon-MM DSv4 task encoder at runtime."""
        from transformers import AutoTokenizer

        _require_energon_packing(enable_energon_packing)
        if not self.ds_tokenizer_path.strip():
            raise ValueError("ds_tokenizer_path must be a non-empty string.")

        tokenizer = AutoTokenizer.from_pretrained(
            self.ds_tokenizer_path,
            trust_remote_code=trust_remote_code,
        )
        if self.image_processor_path:
            image_processor = _QwenVLImageProcessor.from_pretrained(self.image_processor_path)
        else:
            image_processor = _QwenVLImageProcessor()
        self._apply_patch_geometry(image_processor)
        processor = FalconDeepseekv4Processor(
            tokenizer,
            image_processor,
            spatial_merge_size=self.spatial_merge_size,
        )
        return FalconTaskEncoder(
            tokenizer=tokenizer,
            image_processor=processor,
            **self._task_encoder_kwargs(
                config,
                effective_packing=effective_packing,
                enable_energon_packing=enable_energon_packing,
            ),
        )


def _assemble_interleaved_document(sequence: list[Any]) -> tuple[str, list[ImagePageRef]]:
    """Assemble interleaved web-document text, keeping the document whole.

    Walks text parts (``str``) verbatim; :class:`ImagePageRef` parts become the
    literal image marker. Blocks are separated by a single newline.
    """
    blocks: list[str] = []
    images: list[ImagePageRef] = []
    for part in sequence:
        if isinstance(part, str):
            if part:
                blocks.append(part)
        elif isinstance(part, ImagePageRef):
            blocks.append(FALCON_IMAGE_MARKER)
            images.append(part)
        else:
            raise TypeError(f"Unsupported document part {type(part).__name__}; expected str or ImagePageRef")
    return "\n".join(blocks), images


def _materialize_pending_images(
    images: list[ImagePageRef],
    *,
    target_sizes: list[tuple[int, int]],
) -> list[Image.Image]:
    """Decode :class:`ImagePageRef` parts to RGB PIL at preencode resize targets.

    Shared :class:`ImageDocument` instances decode once. Targets make the HF
    processor's resize a no-op so the preencode packing length stays exact.
    """
    if len(target_sizes) != len(images):
        raise ValueError(f"Got {len(target_sizes)} target sizes for {len(images)} images")

    page_targets_by_doc: dict[int, list[tuple[int, int] | None]] = {}
    for index, image in enumerate(images):
        doc_id = id(image.document)
        if doc_id not in page_targets_by_doc:
            page_targets_by_doc[doc_id] = [None] * len(image.document.page_sizes)
        page_targets_by_doc[doc_id][image.page_index] = target_sizes[index]

    decoded_documents: dict[int, list[Image.Image]] = {}
    materialized: list[Image.Image] = []
    for image in images:
        doc_id = id(image.document)
        pages = decoded_documents.get(doc_id)
        if pages is None:
            pages = image.document.decode_rgb_pages(target_sizes=page_targets_by_doc[doc_id])
            decoded_documents[doc_id] = pages
        materialized.append(pages[image.page_index])
    return materialized


class FalconTaskEncoder(QwenVLTaskEncoder):
    """Falcon-MM task encoder.

    Extends the Qwen-VL encoder with a document-style pretraining blend.
    Packing and batching are inherited; ``preencode_sample`` only estimates
    length so the packing buffer does not hold ``pixel_values``. Decode and
    the HF processor run in ``postencode_sample``.
    """

    # Leave crude members as bytes/json so TIFF/JPEG are not decoded at load.
    decoder = None

    # Energon documents ``encode_sample`` and ``postencode_sample`` as mutually
    # exclusive and names ``preencode_sample`` as the pre-pack map. Rebind the
    # Qwen-inherited ``encode_sample`` back to the Energon base implementation
    # so ``TaskEncoder._is_overridden`` treats it as absent; the ChatML
    # postencode path still reaches the Qwen implementation via ``super()``.
    encode_sample = TaskEncoder.encode_sample

    cookers: Sequence[Cooker[FalconDocumentSample | ChatMLSample]] = [
        Cooker(cook_mint_interleaved, has_subflavors={FALCON_CRUDE_TYPE_KEY: FALCON_CRUDE_TYPE_MINT}),
        Cooker(cook_text_document, has_subflavors={FALCON_CRUDE_TYPE_KEY: FALCON_CRUDE_TYPE_TEXT}),
    ]

    def _log_skip_warning(self, message: str, *args: Any) -> None:
        """Log a skipped-sample warning, capped per encoder instance."""
        count = getattr(self, "_skip_warning_count", 0) + 1
        self._skip_warning_count = count
        if count <= _SKIP_WARNING_LOG_LIMIT:
            logger.warning(message, *args)
            if count == _SKIP_WARNING_LOG_LIMIT:
                logger.warning(
                    "Suppressing further Falcon-MM skipped-sample warnings for this process "
                    "(reporting every %d-th skip).",
                    _SKIP_WARNING_REPORT_EVERY,
                )
        elif count % _SKIP_WARNING_REPORT_EVERY == 0:
            logger.warning("Falcon-MM skipped %d samples so far in this process.", count)

    def _record_skip(self, reason: str, *, count: int = 1) -> None:
        """Accumulate skipped-sample counts for WandB ``data/skipped-*`` metrics."""
        if count <= 0:
            return
        counts = getattr(self, "_pack_skip_counts", None)
        if counts is None:
            counts = {}
            self._pack_skip_counts = counts
        counts[reason] = counts.get(reason, 0) + count

    def _consume_skip_stats(self) -> dict[str, int]:
        """Return and clear skip deltas since the last successful pack."""
        counts = getattr(self, "_pack_skip_counts", None)
        self._pack_skip_counts = {}
        return dict(counts) if counts else {}

    def encode_batch(self, batch: QwenVLTaskBatch) -> dict:
        """Encode batch dict and attach per-worker skip deltas for pack-stats logging."""
        raw = super().encode_batch(batch)
        raw["pack_skip_stats"] = self._consume_skip_stats()
        return raw

    def _too_many_images_reason(self, sample_key: Any, num_images: int, max_num_images: int | None) -> str | None:
        """Return a drop reason when a sample exceeds the image-count budget."""
        if max_num_images is not None and num_images > max_num_images:
            self._record_skip(SKIP_REASON_MAX_NUM_IMAGES)
            self._log_skip_warning(
                "Defer-dropping sample %s: %d images exceeds max_num_images=%d (pre-decode)",
                sample_key,
                num_images,
                max_num_images,
            )
            return f"max_num_images:{num_images}>{max_num_images}"
        return None

    def _overlong_packing_reason(self, sample_key: Any, packing_length: int) -> str | None:
        """Return a drop reason when packing length exceeds ``seq_length``.

        Prefer a deferred drop over ``SkipSample`` in preencode: Energon buffer
        restore cannot swallow ``SkipSample`` (``Unexpected skip sample``).
        """
        if packing_length > self.seq_length:
            self._record_skip(SKIP_REASON_OVERLONG)
            self._log_skip_warning(
                "Defer-dropping sample %s: packing length %d exceeds seq_length=%d",
                sample_key,
                packing_length,
                self.seq_length,
            )
            return f"overlong:{packing_length}>{self.seq_length}"
        return None

    def _marker_token_ids(self) -> VisualMarkerIds | None:
        """Resolve the trunk's visual marker ids (cached, best effort).

        The DSv4 shim resolves them from its tokenizer at construction. The
        Qwen trunk resolves its native special tokens. Resolution failure
        (e.g. mock tokenizers in tests) degrades to ``None`` — the Qwen
        markers are already covered by ``extract_skipped_token_ids``.
        """
        marker_ids = getattr(self, "_marker_token_ids_cache", None)
        if marker_ids is None:
            candidate = getattr(self.image_processor, "marker_ids", None)
            if isinstance(candidate, VisualMarkerIds):
                marker_ids = candidate
            else:
                try:
                    marker_ids = resolve_qwen_marker_ids(self.hf_tokenizer)
                except (AttributeError, TypeError, ValueError):
                    marker_ids = None
            self._marker_token_ids_cache = marker_ids
        return marker_ids

    @_stateless_with_failure_tolerance(100)
    def preencode_sample(
        self,
        sample: ChatMLSample | FalconDocumentSample | FalconPendingSample | dict[str, Any],
    ) -> FalconPendingSample:
        """Compute exact packing length without decoding pixels.

        Subflavor keys ``max_pixels``, ``min_pixels``, and ``max_num_images``
        override the encoder defaults for that sample.
        """
        if isinstance(sample, FalconPendingSample):
            return sample
        if isinstance(sample, ChatMLSample):
            return self._pending_from_chatml(sample)
        if isinstance(sample, dict):
            sample = cook_text_document(sample)
        if not isinstance(sample, FalconDocumentSample):
            raise TypeError(f"Unsupported Falcon sample type {type(sample).__name__}")
        return self._pending_from_document(sample)

    @_stateless_with_failure_tolerance(100)
    def postencode_sample(self, sample: FalconPendingSample) -> QwenVLTaskSample:
        """Decode selected images and run the document or ChatML processor path."""
        if sample.drop_reason:
            # Counted once when drop_reason was set in preencode.
            raise SkipSample()
        if sample.chatml is not None:
            return self._postencode_chatml(sample)
        try:
            if sample.image_target_sizes is None:
                raise ValueError("Document postencode requires image_target_sizes from preencode.")
            images = _materialize_pending_images(sample.images, target_sizes=sample.image_target_sizes)
        except Image.DecompressionBombError:
            self._record_skip(SKIP_REASON_DECOMPRESSION_BOMB)
            self._log_skip_warning(
                "Skipping sample %s: Pillow DecompressionBombError",
                sample.__key__,
            )
            raise SkipSample() from None
        return self._encode_document(sample, images)

    def _postencode_chatml(self, sample: FalconPendingSample) -> QwenVLTaskSample:
        """Run the Qwen chat-template path, mapping overlength to ``SkipSample``.

        The ChatML preencode length is a lower bound (template tokens are only
        added by the Qwen encoder here), so both overlength outcomes are
        converted to skips: prepared sequences the Qwen encoder rejects with
        ``ValueError``, and prepared sequences longer than the promised
        packing length (which would break the ``seq_length`` bound of the
        pack). A ``ValueError`` inside the packing wrapper would burn the
        consecutive-failure budget instead of skipping for free.
        """
        try:
            encoded = super().encode_sample(sample.chatml)
        except ValueError as error:
            if "exceeds seq_length" in str(error):
                self._record_skip(SKIP_REASON_CHATML_OVERLENGTH)
                self._log_skip_warning("Skipping sample %s: %s", sample.__key__, error)
                raise SkipSample() from error
            raise
        if encoded.prepared_sequence is not None:
            actual_length = _aligned_sequence_length(
                encoded.prepared_sequence, self.in_batch_packing_pad_to_multiple_of
            )
            if actual_length > sample.packing_length:
                self._record_skip(SKIP_REASON_CHATML_OVERLENGTH)
                self._log_skip_warning(
                    "Skipping sample %s: ChatML postencode length %d exceeds preencode packing_length %d",
                    sample.__key__,
                    actual_length,
                    sample.packing_length,
                )
                raise SkipSample()
        return encoded

    def select_samples_to_pack(self, samples: list[FalconPendingSample]) -> list[list[FalconPendingSample]]:
        """Partition one Energon buffer using exact preencode packing lengths."""
        if not self.enable_energon_packing:
            raise RuntimeError("Energon packing hooks require enable_energon_packing=True.")
        if not samples:
            return []
        if any(not isinstance(sample, FalconPendingSample) for sample in samples):
            raise TypeError("Falcon packing requires FalconPendingSample packing lengths before selection.")
        keep = [sample for sample in samples if not sample.drop_reason]
        drops = [sample for sample in samples if sample.drop_reason]
        packs = (
            first_fit_decreasing(
                keep,
                self.seq_length,
                item_lengths=[sample.packing_length for sample in keep],
            )
            if keep
            else []
        )
        # Singleton packs so postencode can SkipSample each drop without mixing
        # them into real packs.
        packs.extend([[drop] for drop in drops])
        return packs

    @stateless
    def pack_selected_samples(self, samples: list[QwenVLTaskSample]) -> QwenVLPackedTaskSample:
        """Wrap one postencoded group.

        Energon may deliver an empty list when every member raised ``SkipSample``
        during postencode (those are swallowed by the packing wrapper). Skip the
        pack instead of raising ``ValueError``: ``SkipSample`` is swallowed for
        free by the packing failure handler, while a ``ValueError`` would burn
        the consecutive-failure budget (100 consecutive failures kill the
        worker).

        Also re-check prepared lengths: if preencode packing lengths under-shot
        the true postencode lengths, refuse the oversize group with SkipSample
        rather than letting ``batch`` raise.
        """
        prepared = [sample for sample in samples if sample.prepared_sequence is not None]
        if not prepared:
            # Members already recorded their own skips during postencode.
            raise SkipSample()
        packed_length = sum(
            _aligned_sequence_length(sample.prepared_sequence, self.in_batch_packing_pad_to_multiple_of)
            for sample in prepared
            if sample.prepared_sequence is not None
        )
        if packed_length > self.seq_length:
            self._record_skip(SKIP_REASON_OVERSIZE_PACK, count=len(prepared))
            self._log_skip_warning(
                "Skipping oversize pack keys=%s: prepared length %d exceeds seq_length=%d",
                [sample.__key__ for sample in prepared],
                packed_length,
                self.seq_length,
            )
            raise SkipSample()
        return QwenVLPackedTaskSample(
            __key__=",".join(sample.__key__ for sample in prepared),
            __restore_key__=(),
            __subflavors__={},
            samples=prepared,
        )

    def _document_token_count(self, text: str, visual_token_counts: list[int]) -> int:
        """Exact token count matching document assembly + processor expansion.

        Assembled documents are ``text_span`` / marker / ``text_span`` / ... with
        ``FALCON_IMAGE_MARKER`` between spans. Preencode tokenizes each text span
        (same as the processor) and adds header-derived visual tokens plus the
        marker overhead (``vision_start`` / ``vision_end``, uniform across
        trunks — see ``visual_markers``).
        """
        parts = text.split(FALCON_IMAGE_MARKER)
        if len(parts) != len(visual_token_counts) + 1:
            raise ValueError(
                f"Document marker split produced {len(parts)} text parts for {len(visual_token_counts)} images."
            )
        return (
            sum(self._text_token_count(part) for part in parts)
            + sum(visual_token_counts)
            + IMAGE_MARKER_OVERHEAD_TOKENS * len(visual_token_counts)
        )

    def _resolved_image_knobs(self, subflavors: dict[str, Any] | None) -> tuple[int, int, int | None]:
        """Resolve min/max pixels and max_num_images from encoder defaults + subflavors."""
        min_pixels = resolve_subflavor_int(subflavors, SUBFLAVOR_MIN_PIXELS, self.min_pixels)
        max_pixels = resolve_subflavor_int(subflavors, SUBFLAVOR_MAX_PIXELS, self.max_pixels)
        max_num_images = resolve_subflavor_int(subflavors, SUBFLAVOR_MAX_NUM_IMAGES, self.max_num_images)
        if min_pixels is None or max_pixels is None:
            raise ValueError("min_pixels and max_pixels must be set on the encoder or subflavors.")
        if min_pixels <= 0 or max_pixels <= 0:
            raise ValueError("min_pixels and max_pixels must be greater than 0.")
        if max_num_images is not None and max_num_images <= 0:
            raise ValueError("max_num_images must be greater than 0 when set.")
        return min_pixels, max_pixels, max_num_images

    def _text_token_count(self, text: str) -> int:
        """Return tokenizer length for one text span without special tokens."""
        if not text:
            return 0
        encoded = self.hf_tokenizer(text, add_special_tokens=False)
        # BatchEncoding is Mapping-like but not always a ``dict`` subclass; never
        # use ``len(encoded)`` (that is the number of keys, typically 2).
        input_ids = encoded["input_ids"]
        if input_ids and isinstance(input_ids[0], (list, tuple)):
            input_ids = input_ids[0]
        return len(input_ids)

    def _align_packing_length(self, length: int) -> int:
        """Align a token count the same way packed segments are aligned."""
        multiple = self.in_batch_packing_pad_to_multiple_of
        if multiple <= 1:
            return length
        return ((length + multiple - 1) // multiple) * multiple

    def _dropped_pending(
        self,
        sample: ChatMLSample | FalconDocumentSample,
        *,
        drop_reason: str,
        min_pixels: int,
        max_pixels: int,
        packing_length: int = 0,
        text: str = "",
        images: list[Any] | None = None,
        chatml: ChatMLSample | None = None,
        image_target_sizes: list[tuple[int, int]] | None = None,
    ) -> FalconPendingSample:
        """Build a pending sample marked for postencode SkipSample (restore-safe)."""
        return FalconPendingSample.derive_from(
            sample,
            packing_length=packing_length,
            text=text,
            images=[] if images is None else images,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            chatml=chatml,
            image_target_sizes=image_target_sizes,
            drop_reason=drop_reason,
        )

    def _pending_from_document(self, sample: FalconDocumentSample) -> FalconPendingSample:
        """Build a packable pending sample from a cooked document.

        Overlong / too-many-images cases return a pending sample with
        ``drop_reason`` set instead of raising ``SkipSample``, so Energon
        reading-buffer restore does not fail with "Unexpected skip sample".
        """
        sample_key = sample.__key__
        min_pixels, max_pixels, max_num_images = self._resolved_image_knobs(sample.__subflavors__)
        text, images = _assemble_interleaved_document(sample.sequence)
        drop_reason = self._too_many_images_reason(sample_key, len(images), max_num_images)
        if drop_reason is not None:
            return self._dropped_pending(
                sample,
                drop_reason=drop_reason,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
                text=text,
                images=images,
            )

        try:
            factor = self.patch_size * self.merge_size
            image_target_sizes: list[tuple[int, int]] = []
            visual_token_counts: list[int] = []
            for image in images:
                # One smart_resize; decode uses the same target so processor resize is a no-op.
                target = smart_resize(
                    *image.size,
                    factor=factor,
                    min_pixels=min_pixels,
                    max_pixels=max_pixels,
                )
                image_target_sizes.append(target)
                visual_token_counts.append(
                    visual_tokens_for_resized(
                        *target,
                        patch_size=self.patch_size,
                        merge_size=self.merge_size,
                        temporal_patch_size=self.temporal_patch_size,
                    )
                )
            packing_length = self._align_packing_length(self._document_token_count(text, visual_token_counts))
        except (TypeError, ValueError) as error:
            self._record_skip(SKIP_REASON_GEOMETRY)
            self._log_skip_warning("Defer-dropping sample %s: %s", sample_key, error)
            return self._dropped_pending(
                sample,
                drop_reason=f"geometry:{error}",
                min_pixels=min_pixels,
                max_pixels=max_pixels,
                text=text,
                images=images,
            )

        drop_reason = self._overlong_packing_reason(sample_key, packing_length)
        return FalconPendingSample.derive_from(
            sample,
            packing_length=packing_length,
            text=text,
            images=images,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            image_target_sizes=image_target_sizes,
            drop_reason=drop_reason,
        )

    def _pending_from_chatml(self, sample: ChatMLSample) -> FalconPendingSample:
        """Packing length for ChatML from conversation text + media headers.

        ChatML postencode still runs the full Qwen chat-template path, so this
        length is a lower bound for that path (template tokens are added
        later). Samples whose prepared length exceeds the bound are skipped at
        postencode (see ``_postencode_chatml``).

        Overlong / too-many-images use ``drop_reason`` (restore-safe), not
        ``SkipSample`` in preencode.
        """
        min_pixels, max_pixels, max_num_images = self._resolved_image_knobs(sample.__subflavors__)
        images = list(sample.imgs or [])
        drop_reason = self._too_many_images_reason(sample.__key__, len(images), max_num_images)
        if drop_reason is not None:
            return self._dropped_pending(
                sample,
                drop_reason=drop_reason,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
                chatml=sample,
            )
        conversation = sample.conversation
        if not isinstance(conversation, str):
            conversation = json.dumps(conversation)
        try:
            visual_tokens = sum(
                estimate_visual_tokens(
                    *_pil_image_size(image),
                    patch_size=self.patch_size,
                    merge_size=self.merge_size,
                    temporal_patch_size=self.temporal_patch_size,
                    min_pixels=min_pixels,
                    max_pixels=max_pixels,
                )
                for image in images
            )
        except (TypeError, ValueError, AttributeError) as error:
            self._record_skip(SKIP_REASON_GEOMETRY)
            self._log_skip_warning("Defer-dropping sample %s: %s", sample.__key__, error)
            return self._dropped_pending(
                sample,
                drop_reason=f"geometry:{error}",
                min_pixels=min_pixels,
                max_pixels=max_pixels,
                chatml=sample,
            )
        packing_length = self._align_packing_length(self._text_token_count(conversation) + visual_tokens)
        drop_reason = self._overlong_packing_reason(sample.__key__, packing_length)
        return FalconPendingSample.derive_from(
            sample,
            packing_length=packing_length,
            text="",
            images=[],
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            chatml=sample,
            drop_reason=drop_reason,
        )

    def _encode_document(self, sample: FalconPendingSample, images: list[Image.Image]) -> QwenVLTaskSample:
        """Run the document pretraining path for one pending sample."""
        prepared_sequence = self._prepare_falcon_document_sequence(
            sample.text,
            images,
            min_pixels=sample.min_pixels,
            max_pixels=sample.max_pixels,
        )

        if self.max_visual_tokens is not None:
            total_visual_tokens = _visual_token_count(
                prepared_sequence.visual_values.get("image_grid_thw"), self.merge_size
            )
            if total_visual_tokens > self.max_visual_tokens:
                self._record_skip(SKIP_REASON_MAX_VISUAL_TOKENS)
                self._log_skip_warning(
                    "Skipping sample %s: %d visual tokens exceeds max_visual_tokens=%d",
                    sample.__key__,
                    total_visual_tokens,
                    self.max_visual_tokens,
                )
                raise SkipSample()

        # Pure guard: page refs decode to the processor's smart-resize target,
        # so lengths match by construction. A mismatch means the geometry
        # contract broke (e.g. untracked processor behavior); longer than
        # promised breaks packing and must skip, shorter is conservative waste.
        actual_length = _aligned_sequence_length(prepared_sequence, self.in_batch_packing_pad_to_multiple_of)
        if actual_length != sample.packing_length:
            self._log_skip_warning(
                "Sample %s: postencode length %d != preencode packing_length %d "
                "(patch=%d merge=%d temporal=%d min_pixels=%d max_pixels=%d)",
                sample.__key__,
                actual_length,
                sample.packing_length,
                self.patch_size,
                self.merge_size,
                self.temporal_patch_size,
                sample.min_pixels,
                sample.max_pixels,
            )
            if actual_length > sample.packing_length:
                self._record_skip(SKIP_REASON_LENGTH_MISMATCH)
                raise SkipSample()
        if actual_length > self.seq_length:
            self._record_skip(SKIP_REASON_OVERLONG)
            self._log_skip_warning(
                "Skipping sample %s: aligned sequence length %d exceeds seq_length=%d",
                sample.__key__,
                actual_length,
                self.seq_length,
            )
            raise SkipSample()

        return QwenVLTaskSample(
            __key__=sample.__key__,
            __subflavors__=sample.__subflavors__,
            example=None,
            __restore_key__=sample.__restore_key__,
            prepared_sequence=prepared_sequence,
        )

    def _skipped_token_ids(self, processor: Any) -> torch.Tensor:
        """Return label-mask token ids: processor pad tokens plus all three visual markers.

        Uniform across trunks: on the Qwen trunk the markers are already in the
        pad-token set (the union is a no-op); on the DSv4 trunk the reserved
        placeholders are added here.
        """
        base = extract_skipped_token_ids(processor)
        marker_ids = self._marker_token_ids()
        extra = (
            torch.tensor(marker_ids.as_tensor_ids(), dtype=torch.int)
            if marker_ids is not None
            else torch.IntTensor([])
        )
        if base.numel() == 0:
            return extra
        if extra.numel() == 0:
            return base
        return torch.unique(torch.cat([base, extra]))

    def _prepare_falcon_document_sequence(
        self,
        text: str,
        images: list[Any],
        *,
        min_pixels: int,
        max_pixels: int,
    ) -> QwenVLPreparedSequence:
        """Process one document into the exact row consumed by THD packing.

        Mirrors ``_prepare_qwen_vl_sequence`` with the pretraining difference:
        there is no assistant-span mask — every position whose shifted target
        is a real token carries loss.
        """
        processor = self.image_processor
        skipped_tokens = self._skipped_token_ids(processor)
        processor_kwargs: dict[str, Any] = {
            "text": [text],
            "padding": False,
            "return_tensors": "pt",
            "min_pixels": min_pixels,
            "max_pixels": max_pixels,
        }
        if images:
            processor_kwargs["images"] = images

        with use_processor_right_padding(processor):
            sample_batch = {
                key: value.contiguous() if isinstance(value, torch.Tensor) else value
                for key, value in processor(**processor_kwargs).items()
            }

        input_ids = sample_batch["input_ids"][0]
        position_ids = torch.arange(input_ids.numel(), device=input_ids.device, dtype=torch.long)

        labels = torch.cat([input_ids[1:], input_ids.new_full((1,), IGNORE_INDEX)])
        if skipped_tokens.numel() > 0:
            labels = labels.masked_fill(torch.isin(labels, skipped_tokens.to(device=labels.device)), IGNORE_INDEX)
        loss_mask = (labels != IGNORE_INDEX).to(torch.float32)

        visual_values = {}
        for key in QWEN_VISUAL_KEYS:
            value = sample_batch.get(key)
            if not isinstance(value, torch.Tensor):
                continue
            if key in {"pixel_values", "pixel_values_videos"} and value.dim() == 5:
                value = value.flatten(0, 1)
            elif key in {"image_grid_thw", "video_grid_thw"} and value.dim() == 3:
                value = value.flatten(0, 1)
            visual_values[key] = value

        return QwenVLPreparedSequence(
            row={
                "input_ids": input_ids,
                "position_ids": position_ids,
                "labels": labels,
                "loss_mask": loss_mask,
            },
            visual_values=visual_values,
        )
