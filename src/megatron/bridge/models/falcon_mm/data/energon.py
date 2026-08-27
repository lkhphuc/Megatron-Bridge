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
inherits the Qwen-VL packing pipeline — first-fit-decreasing packing and the
THD batch builder — while replacing sample preparation with a document-style
PRETRAINING path: plain interleaved text (no ChatML template, no assistant
loss masking), with full LM loss on every token except visual/special tokens.

Three input sample types are accepted by ``encode_sample``:

- ``InterleavedSample`` (MINT-1T-ArXiv webdataset path): Energon's standard
  interleaved sample type — a ``sequence`` of ``str`` text parts (segments
  and captions) and image tensors in document order, assembled by
  ``mint_webdataset.MintInterleavedWebdataset``. Text parts are kept verbatim
  and each image part becomes the literal marker the Qwen chat template emits
  per image (``<|vision_start|><|image_pad|><|vision_end|>``) so the processor
  expands ``<|image_pad|>`` from the image grid.
- ``dict`` (Energon JSONL ``CrudeSample``, text-only blend path): the
  ``sample["json"]`` payload must be a dict with a string ``"text"`` key,
  which is encoded as a plain document with no images. Energon requires a
  registered cooker for crude datasets, so ``FalconTaskEncoder`` registers a
  pass-through cooker.
- ``ChatMLSample``: passed through to the Qwen-VL encoder unchanged (keeps
  the mock/chatml path working).

Note on wiring: ``FalconEnergonTaskEncoderConfig`` subclasses
``QwenVLEnergonTaskEncoderConfig`` deliberately so the packing gate
(``data/builders/energon.py``) matches with no shared-code edits. Encoder
construction is polymorphic: the builder calls
``task_config.build_task_encoder(...)`` for the whole Qwen family, and this
module's config override returns a ``FalconTaskEncoder`` while every other
Qwen-config value still constructs a ``QwenVLTaskEncoder``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from megatron.energon import SkipSample, stateless
from megatron.energon.flavors.interleaved import InterleavedSample
from megatron.energon.task_encoder.cooking import Cooker

from megatron.bridge.data.builders.energon import QwenVLEnergonTaskEncoderConfig
from megatron.bridge.models.falcon_mm.data.deepseek_v4_processor import FalconDeepseekv4Processor


try:
    from transformers import Qwen3VLImageProcessor as _QwenVLImageProcessor
except ImportError:  # pragma: no cover - pinned transformers may only expose Qwen2VL
    from transformers import Qwen2VLImageProcessor as _QwenVLImageProcessor


if TYPE_CHECKING:
    from megatron.bridge.data.builders.energon import EnergonDatasetConfig
from megatron.bridge.data.collators.sequence_padding import use_processor_right_padding
from megatron.bridge.data.datasets.utils import IGNORE_INDEX
from megatron.bridge.data.energon.task_encoder_utils import ChatMLSample, _images_to_pil
from megatron.bridge.data.token_utils import extract_skipped_token_ids
from megatron.bridge.models.qwen_vl.data.collate_fn import (
    QWEN_VISUAL_KEYS,
    QwenVLPreparedSequence,
)
from megatron.bridge.models.qwen_vl.data.energon import (
    QwenVLTaskEncoder,
    QwenVLTaskSample,
    _aligned_sequence_length,
    _stateless_with_failure_tolerance,
    _visual_token_count,
)


logger = logging.getLogger(__name__)

# Literal marker the Qwen chat template emits per image; the processor expands
# ``<|image_pad|>`` into one token per merged visual token from the grid.
FALCON_IMAGE_MARKER = "<|vision_start|><|image_pad|><|vision_end|>"

# Cap on per-process skipped-sample warnings so a stream of overlong or
# oversized documents does not flood the logs (each Energon worker process
# logs its own share).
_SKIP_WARNING_LOG_LIMIT = 32
_skip_warning_count = 0


def _log_skip_warning(message: str, *args: Any) -> None:
    """Log a skipped-sample warning, capped per process at ``_SKIP_WARNING_LOG_LIMIT``."""
    global _skip_warning_count
    if _skip_warning_count >= _SKIP_WARNING_LOG_LIMIT:
        return
    _skip_warning_count += 1
    logger.warning(message, *args)
    if _skip_warning_count == _SKIP_WARNING_LOG_LIMIT:
        logger.warning("Suppressing further Falcon-MM skipped-sample warnings for this process.")


@dataclass(kw_only=True)
class FalconEnergonTaskEncoderConfig(QwenVLEnergonTaskEncoderConfig):
    """Serializable settings for the Falcon-MM Energon task encoder.

    Adds no fields over the Qwen-VL config; the subclass exists so the
    builder dispatches polymorphically to :class:`FalconTaskEncoder`
    (the document pretraining path) while the packing gate
    (``data/builders/energon.py``) matches the whole Qwen family.
    """

    def build_task_encoder(
        self,
        config: "EnergonDatasetConfig",
        *,
        effective_packing: bool,
        enable_energon_packing: bool,
        trust_remote_code: bool,
    ) -> Any:
        """Construct the Falcon-MM task encoder (Qwen family + document path)."""
        tokenizer, image_processor = self._load_hf_processors(trust_remote_code)
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

    The patch geometry fields must match the model provider's patch embed
    (defaults mirror ``FalconDeepseekv4ModelProvider``). A mismatch between
    the image processor's ``patch_size`` and the Conv3d patchify kernel
    changes the visual token count per image and breaks the placeholder-span
    contract.
    """

    # ``hf_processor_path`` is required non-empty by the parent config but
    # unused by the DSv4 builder (the DSv4 path loads its own tokenizer and
    # image processor). Default it to the DSv4 HF repo id.
    hf_processor_path: str = "deepseek-ai/DeepSeek-V4-Flash"
    ds_tokenizer_path: str = ""
    image_processor_path: str | None = None
    spatial_merge_size: int = 2
    patch_size: int = 16
    temporal_patch_size: int = 2

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
        # Force the patch geometry to match the model's patch embed. These
        # are plain attributes consumed by the image processor's preprocess.
        image_processor.patch_size = self.patch_size
        image_processor.temporal_patch_size = self.temporal_patch_size
        image_processor.merge_size = self.spatial_merge_size
        processor = FalconDeepseekv4Processor(
            tokenizer,
            image_processor,
            spatial_merge_size=self.spatial_merge_size,
        )
        return FalconDeepseekv4TaskEncoder(
            tokenizer=tokenizer,
            image_processor=processor,
            **self._task_encoder_kwargs(
                config,
                effective_packing=effective_packing,
                enable_energon_packing=enable_energon_packing,
            ),
        )


def _assemble_interleaved_document(sequence: list[Any]) -> tuple[str, list[Any]]:
    """Assemble interleaved web-document text, keeping the document whole.

    Walks an Energon ``InterleavedSample.sequence``: text parts (``str``) are
    appended verbatim and image parts (tensors) become the literal image
    marker. Blocks are separated by a single newline. Documents are never
    truncated here — length control happens after processor tokenization,
    when each marker has been expanded to its exact visual token count from
    the image grid; overlong documents are dropped in ``_encode_document``.

    Returns the assembled text and the image parts referenced by the markers,
    in document order.
    """
    blocks: list[str] = []
    images: list[Any] = []
    for part in sequence:
        if isinstance(part, str):
            if part:
                blocks.append(part)
        else:
            blocks.append(FALCON_IMAGE_MARKER)
            images.append(part)
    return "\n".join(blocks), images


def _text_from_crude(sample: dict[str, Any]) -> str:
    """Extract the document text from a JSONL ``CrudeSample`` dict.

    The JSONL line payload (``sample["json"]``) must be a dict holding the
    document under a string ``"text"`` key.
    """
    payload = sample.get("json")
    if not isinstance(payload, dict) or not isinstance(payload.get("text"), str):
        raise ValueError(
            "JSONL text sample must be a dict with a string 'text' key; "
            f"got {type(payload).__name__} for key {sample.get('__key__')!r}."
        )
    return payload["text"]


@stateless
def _cook_crude_sample(sample: dict[str, Any]) -> dict[str, Any]:
    """Pass a JSONL ``CrudeSample`` dict through to ``encode_sample`` unchanged."""
    return sample


class FalconTaskEncoder(QwenVLTaskEncoder):
    """Falcon-MM task encoder.

    Extends the Qwen-VL encoder with a document-style pretraining path for
    ``InterleavedSample`` (interleaved web documents) and Energon JSONL
    ``CrudeSample`` dicts (text-only ``{"text": ...}`` blends). Packing,
    batching, and ``ChatMLSample`` handling are inherited unchanged.

    Documents are encoded whole — never truncated. Length control happens
    after processor tokenization (exact visual-token expansion), when
    ``_encode_document`` drops overlong documents with ``SkipSample``.
    """

    # Energon rejects crude (JSONL) datasets unless the task encoder registers
    # a cooker. The dict-to-document conversion lives in ``encode_sample``,
    # so the cooker is a pass-through.
    cookers = [Cooker(_cook_crude_sample)]

    @_stateless_with_failure_tolerance(100)
    def encode_sample(self, sample: ChatMLSample | InterleavedSample | dict[str, Any]) -> QwenVLTaskSample:
        """Normalize one Energon sample into a prepared THD-packable sequence.

        Accepts an ``InterleavedSample`` (MINT webdataset path), a
        ``CrudeSample`` dict from Energon's JSONL crude dataset (text-only
        blend path), or a ``ChatMLSample`` (delegated to the Qwen-VL encoder).
        Document inputs always produce a prepared sequence with full LM loss
        (``example=None``), ready for Energon native packing.
        """
        if isinstance(sample, InterleavedSample):
            text, images = _assemble_interleaved_document(sample.sequence)
            return self._encode_document(sample, text, images)
        if isinstance(sample, dict):
            text = _text_from_crude(sample)
            return self._encode_document(sample, text, [])
        return super().encode_sample(sample)

    def _encode_document(self, sample: Any, text: str, images: list[Any]) -> QwenVLTaskSample:
        """Run the document pretraining path for one normalized sample."""
        images = _images_to_pil(images)
        if self.max_num_images is not None and len(images) > self.max_num_images:
            _log_skip_warning(
                "Skipping sample %s: %d images exceeds max_num_images=%d",
                getattr(sample, "__key__", "<unknown>"),
                len(images),
                self.max_num_images,
            )
            raise SkipSample()

        prepared_sequence = self._prepare_falcon_document_sequence(text, images)

        if self.max_visual_tokens is not None:
            total_visual_tokens = _visual_token_count(
                prepared_sequence.visual_values.get("image_grid_thw"), self.merge_size
            )
            if total_visual_tokens > self.max_visual_tokens:
                _log_skip_warning(
                    "Skipping sample %s: %d visual tokens exceeds max_visual_tokens=%d",
                    getattr(sample, "__key__", "<unknown>"),
                    total_visual_tokens,
                    self.max_visual_tokens,
                )
                raise SkipSample()

        packing_length = _aligned_sequence_length(prepared_sequence, self.in_batch_packing_pad_to_multiple_of)
        if packing_length > self.seq_length:
            _log_skip_warning(
                "Skipping sample %s: aligned sequence length %d exceeds seq_length=%d",
                getattr(sample, "__key__", "<unknown>"),
                packing_length,
                self.seq_length,
            )
            raise SkipSample()

        return QwenVLTaskSample(
            __key__=sample["__key__"] if isinstance(sample, dict) else sample.__key__,
            __subflavors__=sample.get("__subflavors__") if isinstance(sample, dict) else sample.__subflavors__,
            example=None,
            __restore_key__=sample.get("__restore_key__", ()) if isinstance(sample, dict) else sample.__restore_key__,
            prepared_sequence=prepared_sequence,
        )

    def _skipped_token_ids(self, processor: Any) -> torch.Tensor:
        """Return label-mask token ids for the given processor."""
        return extract_skipped_token_ids(processor)

    def _prepare_falcon_document_sequence(self, text: str, images: list[Any]) -> QwenVLPreparedSequence:
        """Process one document into the exact row consumed by THD packing.

        Mirrors ``_prepare_qwen_vl_sequence`` with the pretraining difference:
        there is no assistant-span mask — every position whose shifted target
        is a real token carries loss. Labels are shifted left and masked to
        ``IGNORE_INDEX`` for visual/special tokens from
        ``extract_skipped_token_ids``; the loss mask is ``labels != IGNORE_INDEX``
        so ignored positions contribute neither to the loss numerator nor to
        the per-token ``num_tokens`` denominator.
        """
        processor = self.image_processor
        skipped_tokens = self._skipped_token_ids(processor)
        processor_kwargs: dict[str, Any] = {
            "text": [text],
            "padding": False,
            "return_tensors": "pt",
        }
        if self.min_pixels is not None:
            processor_kwargs["min_pixels"] = self.min_pixels
        if self.max_pixels is not None:
            processor_kwargs["max_pixels"] = self.max_pixels
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
        # The loss mask follows the labels: any position whose shifted target
        # is IGNORE_INDEX must contribute neither to the loss numerator nor to
        # the per-token num_tokens denominator. MCore's cross entropy zeroes
        # the gradient at IGNORE_INDEX positions but reports logsumexp(logits)
        # there, so leaving them in the mask would pollute the reported loss
        # with garbage values (one per visual/special token). This also zeroes
        # the final position, whose label is IGNORE_INDEX by construction.
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

        # No "attention_mask" row entry, by design: this path is THD-only, and
        # a pack carries its per-segment causal structure in cu_seqlens (the
        # packer emits attention_mask=None). An all-ones mask would be dropped
        # by build_mcore_thd_sequence_batch_from_rows and a non-trivial one is
        # rejected there, so omitting the key keeps the THD contract explicit
        # instead of leaving a misleading all-ones tensor for downstream code.
        return QwenVLPreparedSequence(
            row={
                "input_ids": input_ids,
                "position_ids": position_ids,
                "labels": labels,
                "loss_mask": loss_mask,
            },
            visual_values=visual_values,
        )


class FalconDeepseekv4TaskEncoder(FalconTaskEncoder):
    """Falcon-MM task encoder wired to the DeepSeek-V4 processor shim."""

    def _skipped_token_ids(self, processor: Any) -> torch.Tensor:
        """Mask DSv4 placeholder ids in addition to the base skipped tokens."""
        base = super()._skipped_token_ids(processor)
        placeholder = torch.tensor([processor.image_token_id], dtype=torch.int)
        if base.numel() == 0:
            return placeholder
        return torch.unique(torch.cat([base, placeholder]))
