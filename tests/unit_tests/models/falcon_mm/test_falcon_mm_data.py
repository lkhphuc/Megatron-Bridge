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

"""Unit tests for Falcon-MM Energon data plane (cook / pack-then-decode / DSv4)."""

import io
import json
from unittest.mock import MagicMock

import pytest
import torch
from megatron.energon import DefaultTaskEncoder, SkipSample, TaskEncoder
from PIL import Image

from megatron.bridge.data.builders.energon import QwenVLEnergonTaskEncoderConfig
from megatron.bridge.data.datasets.utils import IGNORE_INDEX
from megatron.bridge.models.falcon_mm.data import (
    FalconEnergonTaskEncoderConfig,
    FalconTaskEncoder,
)
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
from megatron.bridge.models.falcon_mm.data.deepseek_v4_processor import FalconDeepseekv4Processor
from megatron.bridge.models.falcon_mm.data.images import (
    ImageDocument,
    ImagePageRef,
    assemble_mint_sequence,
    smart_resize,
)
from megatron.bridge.models.falcon_mm.data.llama3_processor import FalconLlama3Processor
from megatron.bridge.models.falcon_mm.data.visual_markers import (
    DSV4_IMAGE_PAD_TOKEN,
    DSV4_VISION_END_TOKEN,
    DSV4_VISION_START_TOKEN,
    FALCON_IMAGE_MARKER,
    LLAMA3_IMAGE_PAD_TOKEN,
    LLAMA3_VISION_END_TOKEN,
    LLAMA3_VISION_START_TOKEN,
)
from megatron.bridge.models.qwen_vl.data.energon import QwenVLTaskEncoder, QwenVLTaskSample


def _document_sample(sequence, *, key="shard/0", subflavors=None):
    return FalconDocumentSample(
        __key__=key,
        __restore_key__=("shard", 0),
        __subflavors__=subflavors or {},
        sequence=sequence,
    )


def _make_tokenizer():
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 1
    tokenizer.image_token_id = 151655
    tokenizer.video_token_id = 151656

    def _encode(text, add_special_tokens=False, **kwargs):
        del add_special_tokens, kwargs
        if not text:
            return {"input_ids": []}
        return {"input_ids": list(range(len(text.split())))}

    tokenizer.side_effect = _encode
    return tokenizer


def _encode_full(encoder, sample):
    return encoder.postencode_sample(encoder.preencode_sample(sample))


def _make_document_processor(input_ids=None, *, with_visuals=False, skipped_tokens=None):
    """Mock HF processor. Default length tracks the text kwarg so packing stays consistent."""
    processor = MagicMock()
    processor.added_tokens_decoder = skipped_tokens or {}

    def _call(*args, **kwargs):
        del args
        if input_ids is not None:
            ids_tuple = tuple(input_ids)
        else:
            text = kwargs.get("text") or [""]
            doc = text[0] if text else ""
            # Mirror the mock tokenizer: one id per whitespace-separated token.
            n = len(doc.split()) if doc.strip() else 1
            ids_tuple = tuple(range(5, 5 + n))
        ids = torch.tensor([list(ids_tuple)], dtype=torch.long)
        output = {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
        }
        if with_visuals:
            output["pixel_values"] = torch.randn(4, 8)
            output["image_grid_thw"] = torch.tensor([[1, 2, 2], [1, 2, 2]])
        return output

    processor.side_effect = _call
    return processor


def _tiff_bytes(*sizes: tuple[int, int]) -> bytes:
    buffer = io.BytesIO()
    pages = [Image.new("RGB", size, "red") for size in sizes]
    pages[0].save(buffer, format="TIFF", save_all=True, append_images=pages[1:])
    return buffer.getvalue()


def _jpeg_bytes(size: tuple[int, int] = (8, 8)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, "blue").save(buffer, format="JPEG")
    return buffer.getvalue()


def _crude_sample(payload, *, subflavors=None):
    return {
        "json": payload,
        "__key__": "train.jsonl/0",
        "__shard__": "train.jsonl",
        "__restore_key__": ("train.jsonl", 0),
        "__sources__": (),
        "__subflavors__": subflavors or {},
    }


class TestFalconEnergonDefaults:
    def test_packing_gate_and_early_fusion_defaults(self):
        config = FalconEnergonTaskEncoderConfig(hf_processor_path="Qwen/Qwen3.5-2B")
        assert isinstance(config, QwenVLEnergonTaskEncoderConfig)
        assert config.patch_size == FALCON_DEFAULT_PATCH_SIZE == 32
        assert config.spatial_merge_size == FALCON_DEFAULT_SPATIAL_MERGE_SIZE == 1
        assert config.temporal_patch_size == FALCON_DEFAULT_TEMPORAL_PATCH_SIZE == 1
        assert config.min_pixels == FALCON_DEFAULT_MIN_PIXELS == 1024
        assert config.max_sample_length is None

    def test_global_sample_limit_reaches_task_encoder_kwargs(self):
        config = FalconEnergonTaskEncoderConfig(hf_processor_path="Qwen/Qwen3.5-2B", max_sample_length=8192)
        kwargs = config._task_encoder_kwargs(
            MagicMock(seq_length=16384),
            effective_packing=True,
            enable_energon_packing=True,
        )
        assert kwargs["max_sample_length"] == 8192
        assert kwargs["max_padding_length"] == 16384

    def test_build_requires_energon_packing(self):
        config = FalconEnergonTaskEncoderConfig(hf_processor_path="Qwen/Qwen3.5-2B")
        with pytest.raises(ValueError, match="Energon-native packing"):
            config.build_task_encoder(
                MagicMock(),
                effective_packing=False,
                enable_energon_packing=False,
                trust_remote_code=False,
            )


class TestMintAndImages:
    def test_assemble_and_cook_mint(self):
        document = ImageDocument.from_bytes(_tiff_bytes((4, 4), (4, 4)))
        assert assemble_mint_sequence(
            texts=["t0", "t1", "t2"],
            images=["x", None, "y"],
            captions=["c0", "c1"],
            tiff_payload=document,
            sample_key="k",
        ) == [
            "t0",
            ImagePageRef(document, 0),
            "c0",
            "t1",
            "t2",
            ImagePageRef(document, 1),
            "c1",
        ]

        with pytest.raises(ValueError, match="image references than image pages"):
            assemble_mint_sequence(
                texts=["a", "b"],
                images=["x", "y"],
                captions=[],
                tiff_payload=ImageDocument.from_bytes(_tiff_bytes((4, 4))),
                sample_key="k",
            )

        cooked = cook_mint_interleaved(
            {
                "__key__": "k",
                "__restore_key__": ("shard", 0),
                "__subflavors__": {FALCON_CRUDE_TYPE_KEY: FALCON_CRUDE_TYPE_MINT},
                "json": {
                    "texts": ["hello", "more"],
                    "images": ["x", "y"],
                    "captions": ["cap", "cap2"],
                },
                "tiff": _tiff_bytes((4, 4), (8, 2)),
            }
        )
        assert isinstance(cooked, FalconDocumentSample)
        assert cooked.sequence[0] == "hello"
        assert isinstance(cooked.sequence[1], ImagePageRef)
        assert cooked.sequence[1].size == (4, 4)
        assert cooked.sequence[2] == "cap"
        assert cooked.sequence[3] == "more"
        assert isinstance(cooked.sequence[4], ImagePageRef)
        assert cooked.sequence[4].size == (8, 2)
        assert cooked.sequence[5] == "cap2"

    def test_cook_caption_pair_jpg_txt(self):
        cooked = cook_caption_pair(
            {
                "__key__": "00000000",
                "__restore_key__": ("shard", 0),
                "__subflavors__": {FALCON_CRUDE_TYPE_KEY: FALCON_CRUDE_TYPE_CAPTION},
                "jpg": _jpeg_bytes((16, 12)),
                "txt": b"a blue square\n",
                "json": {"width": 16, "height": 12},
            }
        )
        assert isinstance(cooked, FalconDocumentSample)
        assert len(cooked.sequence) == 2
        assert isinstance(cooked.sequence[0], ImagePageRef)
        assert cooked.sequence[0].size == (16, 12)
        assert cooked.sequence[1] == "a blue square"

        with pytest.raises(ValueError, match="empty 'txt'"):
            cook_caption_pair(
                {
                    "__key__": "empty",
                    "__restore_key__": ("shard", 1),
                    "__subflavors__": {FALCON_CRUDE_TYPE_KEY: FALCON_CRUDE_TYPE_CAPTION},
                    "jpg": _jpeg_bytes(),
                    "txt": "   ",
                }
            )

        with pytest.raises(ValueError, match="one of"):
            cook_caption_pair(
                {
                    "__key__": "noimg",
                    "__restore_key__": ("shard", 2),
                    "__subflavors__": {FALCON_CRUDE_TYPE_KEY: FALCON_CRUDE_TYPE_CAPTION},
                    "txt": "caption only",
                }
            )

    def test_image_document_headers_and_target_decode(self):
        document = ImageDocument.from_bytes(_tiff_bytes((200, 100)))
        assert document.page_sizes == ((200, 100),)
        pages = document.decode_rgb_pages(target_sizes=[(100, 50)])
        assert len(pages) == 1
        assert isinstance(pages[0], Image.Image)
        assert pages[0].size == (100, 50)
        assert pages[0].mode == "RGB"

    def test_smart_resize_idempotent(self):
        """Decode targets must be fixed points of the processor's smart_resize."""
        for width, height in [(200, 100), (4000, 3000), (16, 16), (1731, 2241)]:
            target = smart_resize(width, height, factor=32, min_pixels=1024, max_pixels=1048576)
            assert smart_resize(*target, factor=32, min_pixels=1024, max_pixels=1048576) == target
            assert target[0] % 32 == 0 and target[1] % 32 == 0
            assert 1024 <= target[0] * target[1] <= 1048576


class TestFalconTaskEncoder:
    def _make_encoder(self, processor=None, **kwargs):
        kwargs.setdefault("max_padding_length", 512)
        kwargs.setdefault("patch_size", FALCON_DEFAULT_PATCH_SIZE)
        kwargs.setdefault("spatial_merge_size", FALCON_DEFAULT_SPATIAL_MERGE_SIZE)
        kwargs.setdefault("min_pixels", FALCON_DEFAULT_MIN_PIXELS)
        kwargs.setdefault("max_pixels", 1003520)
        return FalconTaskEncoder(
            tokenizer=_make_tokenizer(),
            image_processor=processor or _make_document_processor(),
            **kwargs,
        )

    def test_encode_sample_shadowed_for_energon_contract(self):
        """Energon requires preencode/postencode without an overridden encode_sample."""
        encoder = self._make_encoder(enable_energon_packing=True)
        assert encoder._is_overridden(encoder.preencode_sample)
        assert not encoder._is_overridden(encoder.encode_sample, bases=(TaskEncoder, DefaultTaskEncoder))

    def test_cookers_and_pack_then_decode(self):
        processor = _make_document_processor()
        encoder = self._make_encoder(processor=processor, enable_energon_packing=True)
        assert len(encoder.cookers) == 3
        assert encoder.cookers[0].has_subflavors == {FALCON_CRUDE_TYPE_KEY: FALCON_CRUDE_TYPE_MINT}
        assert encoder.cookers[0].cook is cook_mint_interleaved
        assert encoder.cookers[1].has_subflavors == {FALCON_CRUDE_TYPE_KEY: FALCON_CRUDE_TYPE_CAPTION}
        assert encoder.cookers[1].cook is cook_caption_pair
        assert encoder.cookers[2].has_subflavors == {FALCON_CRUDE_TYPE_KEY: FALCON_CRUDE_TYPE_TEXT}
        assert encoder.cookers[2].cook is cook_text_document

        sample = _crude_sample({"text": "hello world"}, subflavors={"max_pixels": 262144})
        pending = encoder.preencode_sample(sample)
        assert isinstance(pending, FalconPendingSample)
        assert pending.max_pixels == 262144
        assert pending.packing_length > 0
        processor.assert_not_called()
        packs = encoder.select_samples_to_pack([pending, pending])
        assert packs and all(isinstance(group[0], FalconPendingSample) for group in packs)

        encoded = encoder.postencode_sample(pending)
        assert isinstance(encoded, QwenVLTaskSample)
        processor.assert_called_once()

    def test_pack_selected_samples_skips_empty(self):
        encoder = self._make_encoder(enable_energon_packing=True)
        with pytest.raises(SkipSample):
            encoder.pack_selected_samples([])

    def test_preencode_overlong_sets_drop_reason_not_skip(self):
        """Buffer restore cannot SkipSample; overlong docs must defer via drop_reason."""
        encoder = self._make_encoder(enable_energon_packing=True, max_padding_length=8)
        # Force a large packing estimate without going through the processor.
        pending = encoder.preencode_sample(_crude_sample({"text": "word " * 200}))
        assert isinstance(pending, FalconPendingSample)
        assert pending.drop_reason is not None
        assert pending.drop_reason.startswith("overlong:")
        assert pending.packing_length > encoder.seq_length
        assert encoder._pack_skip_counts.get("overlong") == 1

        packs = encoder.select_samples_to_pack([pending])
        assert packs == [[pending]]
        with pytest.raises(SkipSample):
            encoder.postencode_sample(pending)
        # Deferred drop must not double-count at postencode.
        assert encoder._pack_skip_counts.get("overlong") == 1

    def test_max_sample_length_filters_before_packing_without_shrinking_pack(self):
        encoder = self._make_encoder(enable_energon_packing=True, max_padding_length=16)
        subflavors = {"max_sample_length": 8, "max_pixels": 262144}
        first = encoder.preencode_sample(_document_sample(["word " * 8], key="first", subflavors=subflavors))
        second = encoder.preencode_sample(_document_sample(["word " * 8], key="second", subflavors=subflavors))
        rejected = encoder.preencode_sample(_document_sample(["word " * 9], key="rejected", subflavors=subflavors))

        assert first.packing_length == second.packing_length == 8
        assert first.max_pixels == 262144
        assert rejected.packing_length == 9
        assert rejected.drop_reason == "max_sample_length:9>8"
        assert encoder._pack_skip_counts == {"max_sample_length": 1}
        packs = encoder.select_samples_to_pack([first, second, rejected])
        assert any(set(sample.__key__ for sample in pack) == {"first", "second"} for pack in packs)
        assert [rejected] in packs
        with pytest.raises(SkipSample):
            encoder.postencode_sample(rejected)
        assert encoder._pack_skip_counts == {"max_sample_length": 1}

    def test_max_sample_length_is_per_source_and_must_be_positive(self):
        encoder = self._make_encoder(enable_energon_packing=True, max_padding_length=16)
        uncapped = encoder.preencode_sample(_document_sample(["word " * 9]))
        assert uncapped.drop_reason is None
        with pytest.raises(ValueError, match="max_sample_length must be greater than 0"):
            encoder.preencode_sample(_document_sample(["word"], subflavors={"max_sample_length": 0}))

    def test_global_max_sample_length_counts_visual_tokens_and_uses_tighter_source_cap(self):
        processor = _make_document_processor()
        encoder = self._make_encoder(
            processor=processor,
            enable_energon_packing=True,
            max_padding_length=64,
            max_sample_length=10,
        )
        document = ImageDocument.from_bytes(_tiff_bytes((128, 128)))
        image = ImagePageRef(document, 0)
        visual = encoder.preencode_sample(_document_sample(["short", image], key="visual"))
        assert visual.packing_length > 10
        assert visual.drop_reason == f"max_sample_length:{visual.packing_length}>10"
        processor.assert_not_called()

        tighter = encoder.preencode_sample(
            _document_sample(["word " * 9], key="tighter", subflavors={"max_sample_length": 8})
        )
        assert tighter.drop_reason == "max_sample_length:9>8"
        looser = encoder.preencode_sample(
            _document_sample(["word " * 11], key="looser", subflavors={"max_sample_length": 12})
        )
        assert looser.drop_reason == "max_sample_length:11>10"
        with pytest.raises(ValueError, match="max_sample_length must be greater than 0"):
            self._make_encoder(max_sample_length=0)

    def test_encode_batch_attaches_and_clears_skip_stats(self, monkeypatch):
        encoder = self._make_encoder(enable_energon_packing=True, max_padding_length=8)
        encoder.preencode_sample(_crude_sample({"text": "word " * 200}))
        monkeypatch.setattr(QwenVLTaskEncoder, "encode_batch", lambda self, batch: {"input_ids": None})
        out = encoder.encode_batch(None)
        assert out["pack_skip_stats"] == {"overlong": 1}
        assert encoder._consume_skip_stats() == {}

    def test_max_num_images_records_skip(self):
        encoder = self._make_encoder(enable_energon_packing=True, max_num_images=1)
        document = ImageDocument.from_bytes(_tiff_bytes((8, 8), (8, 8)))
        pending = encoder.preencode_sample(
            _document_sample(["a", ImagePageRef(document, 0), "b", ImagePageRef(document, 1), "c"])
        )
        assert pending.drop_reason is not None
        assert pending.drop_reason.startswith("max_num_images:")
        assert encoder._pack_skip_counts.get("max_num_images") == 1

    def test_pack_selected_samples_skips_oversize_prepared_group(self):
        from megatron.bridge.models.qwen_vl.data.collate_fn import QwenVLPreparedSequence

        encoder = self._make_encoder(enable_energon_packing=True, max_padding_length=8)
        prepared = QwenVLPreparedSequence(
            row={
                "input_ids": torch.arange(10, dtype=torch.long),
                "labels": torch.arange(10, dtype=torch.long),
                "loss_mask": torch.ones(10),
                "position_ids": torch.arange(10, dtype=torch.long),
            },
            visual_values={},
        )
        sample = QwenVLTaskSample(
            __key__="long",
            __restore_key__=(),
            __subflavors__={},
            example=None,
            prepared_sequence=prepared,
        )
        with pytest.raises(SkipSample):
            encoder.pack_selected_samples([sample])
        assert encoder._pack_skip_counts.get("oversize_pack") == 1

    def test_select_isolates_dropped_from_keep(self):
        encoder = self._make_encoder(enable_energon_packing=True)
        keep = encoder.preencode_sample(_crude_sample({"text": "hello"}))
        drop = encoder.preencode_sample(_crude_sample({"text": "hello"}))
        drop.drop_reason = "overlong:999>8"
        packs = encoder.select_samples_to_pack([keep, drop])
        assert [keep] in packs
        assert [drop] in packs
        assert not any(keep in group and drop in group for group in packs)

    def test_preencode_packing_length_matches_postencode(self):
        """Header geometry + tokenizer must match the processor row length."""
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import PreTrainedTokenizerFast, Qwen2VLImageProcessor

        from megatron.bridge.models.qwen_vl.data.energon import _aligned_sequence_length

        vocab = {
            "<unk>": 0,
            "Hello": 1,
            "cap": 2,
            DSV4_IMAGE_PAD_TOKEN: 3,
            DSV4_VISION_START_TOKEN: 4,
            DSV4_VISION_END_TOKEN: 5,
        }
        backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
        backend.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend,
            unk_token="<unk>",
            pad_token="<unk>",
            eos_token="<unk>",
        )
        image_processor = Qwen2VLImageProcessor(
            size={"shortest_edge": 32, "longest_edge": 256},
            patch_size=FALCON_DEFAULT_PATCH_SIZE,
            temporal_patch_size=FALCON_DEFAULT_TEMPORAL_PATCH_SIZE,
            merge_size=FALCON_DEFAULT_SPATIAL_MERGE_SIZE,
        )
        processor = FalconDeepseekv4Processor(
            tokenizer, image_processor, spatial_merge_size=FALCON_DEFAULT_SPATIAL_MERGE_SIZE
        )
        encoder = FalconTaskEncoder(
            tokenizer=tokenizer,
            image_processor=processor,
            max_padding_length=2048,
            patch_size=FALCON_DEFAULT_PATCH_SIZE,
            spatial_merge_size=FALCON_DEFAULT_SPATIAL_MERGE_SIZE,
            temporal_patch_size=FALCON_DEFAULT_TEMPORAL_PATCH_SIZE,
            min_pixels=FALCON_DEFAULT_MIN_PIXELS,
            max_pixels=1003520,
            enable_energon_packing=True,
        )
        document = ImageDocument.from_bytes(_tiff_bytes((64, 64), (128, 64)))
        sample = _document_sample(["Hello", ImagePageRef(document, 0), "cap", ImagePageRef(document, 1), "cap"])
        pending = encoder.preencode_sample(sample)
        encoded = encoder.postencode_sample(pending)
        actual = _aligned_sequence_length(encoded.prepared_sequence, encoder.in_batch_packing_pad_to_multiple_of)
        assert actual == pending.packing_length

    def test_document_markers_and_loss_mask(self):
        processor = _make_document_processor(with_visuals=True)
        encoder = self._make_encoder(processor=processor)
        document = ImageDocument.from_bytes(_tiff_bytes((8, 8), (8, 8)))
        encoded = _encode_full(
            encoder,
            _document_sample(
                ["Hello", ImagePageRef(document, 0), "cap0", "world", "tail", ImagePageRef(document, 1), "cap1"]
            ),
        )
        prepared = encoded.prepared_sequence
        assert prepared is not None and prepared.visual_values
        loss_mask = prepared.row["loss_mask"]
        assert torch.all(loss_mask[:-1] == 1) and loss_mask[-1] == 0
        text = processor.call_args.kwargs["text"][0]
        assert text == f"Hello\n{FALCON_IMAGE_MARKER}\ncap0\nworld\ntail\n{FALCON_IMAGE_MARKER}\ncap1"
        assert len(processor.call_args.kwargs["images"]) == 2
        assert all(isinstance(image, Image.Image) for image in processor.call_args.kwargs["images"])

    def test_page_ref_decode_in_postencode(self):
        document = ImageDocument.from_bytes(_tiff_bytes((8, 8)))
        processor = _make_document_processor(with_visuals=True)
        encoder = self._make_encoder(processor=processor)
        _encode_full(encoder, _document_sample(["Hello", ImagePageRef(document, 0), "cap"]))
        images = processor.call_args.kwargs["images"]
        # Decoded to the smart-resize target (min_pixels upscale), not native size.
        assert len(images) == 1 and images[0].size == (32, 32)

    def test_overlong_skipped_and_text_labels(self):
        long_processor = _make_document_processor(input_ids=tuple(range(10)))
        encoder = self._make_encoder(processor=long_processor, max_padding_length=8)
        pending = encoder.preencode_sample(_crude_sample({"text": "a plain document"}))
        with pytest.raises(SkipSample):
            encoder.postencode_sample(pending)

        processor = _make_document_processor(input_ids=(5, 6, 7, 8), skipped_tokens={7: "<|image_pad|>"})
        encoded = _encode_full(
            self._make_encoder(processor=processor),
            _crude_sample({"text": "a plain document text"}),
        )
        row = encoded.prepared_sequence.row
        assert row["loss_mask"].tolist() == [1.0, 0.0, 1.0, 0.0]
        assert row["labels"].tolist() == [6, IGNORE_INDEX, 8, IGNORE_INDEX]

    @pytest.mark.parametrize(
        "payload",
        [
            {"messages": [{"role": "user", "content": "Hi"}]},
            [{"role": "user", "content": "Hi"}],
            {"text": 123},
        ],
    )
    def test_text_cooker_rejects_non_document_payloads(self, payload):
        with pytest.raises(ValueError, match="JSONL text sample"):
            cook_text_document(_crude_sample(payload))

    def _make_chatml_sample(self):
        from megatron.bridge.data.energon.metadata import sample_metadata_kwargs
        from megatron.bridge.data.energon.task_encoder_utils import ChatMLSample

        return ChatMLSample(
            **sample_metadata_kwargs(key="key", restore_key="restore_key", subflavors={}),
            imgs=None,
            videos=None,
            conversation=json.dumps(
                [
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello"},
                ]
            ),
        )

    def test_chatml_deferred_as_pending(self):
        encoder = self._make_encoder()
        sample = self._make_chatml_sample()
        pending = encoder.preencode_sample(sample)
        assert isinstance(pending, FalconPendingSample)
        assert pending.chatml is sample

    def test_chatml_postencode_overlength_skips(self, monkeypatch):
        """Qwen-path overlength ValueError must become a free SkipSample."""
        encoder = self._make_encoder()
        pending = encoder.preencode_sample(self._make_chatml_sample())

        def _overlong_encode(self, sample):
            raise ValueError(f"Sample {sample.__key__} aligned sequence length 999 exceeds seq_length=512.")

        monkeypatch.setattr(QwenVLTaskEncoder, "encode_sample", _overlong_encode)
        with pytest.raises(SkipSample):
            encoder.postencode_sample(pending)

    def test_chatml_postencode_longer_than_promised_skips(self, monkeypatch):
        """ChatML preencode is a lower bound; longer prepared rows break packs."""
        encoder = self._make_encoder()
        pending = encoder.preencode_sample(self._make_chatml_sample())

        prepared = MagicMock()
        prepared.sequence_length = pending.packing_length + 1

        def _longer_encode(self, sample):
            return QwenVLTaskSample(
                __key__=sample.__key__,
                __subflavors__=sample.__subflavors__,
                example=None,
                __restore_key__=sample.__restore_key__,
                prepared_sequence=prepared,
            )

        monkeypatch.setattr(QwenVLTaskEncoder, "encode_sample", _longer_encode)
        with pytest.raises(SkipSample):
            encoder.postencode_sample(pending)

    def test_skip_warning_counter_per_instance(self):
        first = self._make_encoder()
        second = self._make_encoder()
        first._log_skip_warning("skip %s", "a")
        first._log_skip_warning("skip %s", "b")
        assert first._skip_warning_count == 2
        assert not hasattr(second, "_skip_warning_count")


class TestFalconDeepseekv4Processor:
    def _processor(self, *, with_markers: bool = True):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import PreTrainedTokenizerFast, Qwen2VLImageProcessor

        vocab = {"<unk>": 0, "hello": 1, "world": 2, "cap": 3, "tail": 4}
        if with_markers:
            vocab[DSV4_IMAGE_PAD_TOKEN] = 5
            vocab[DSV4_VISION_START_TOKEN] = 6
            vocab[DSV4_VISION_END_TOKEN] = 7
        backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
        backend.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend,
            unk_token="<unk>",
            pad_token="<unk>",
            eos_token="<unk>",
        )
        image_processor = Qwen2VLImageProcessor(
            size={"shortest_edge": 32, "longest_edge": 256},
            patch_size=32,
            temporal_patch_size=FALCON_DEFAULT_TEMPORAL_PATCH_SIZE,
            merge_size=1,
        )
        return (
            FalconDeepseekv4Processor(tokenizer, image_processor, spatial_merge_size=1),
            tokenizer,
            image_processor,
        )

    def test_placeholder_expansion_and_order(self):
        """Each marker expands to [vision_start] + [image_pad x N] + [vision_end]."""
        processor, tokenizer, image_processor = self._processor()
        images = [Image.new("RGB", (64, 64), "red"), Image.new("RGB", (128, 64), "blue")]
        document = f"hello\n{FALCON_IMAGE_MARKER}\ncap\n{FALCON_IMAGE_MARKER}\ntail"
        result = processor(text=[document], images=images, return_tensors="pt")
        grids = image_processor(images=images, return_tensors="pt")["image_grid_thw"]
        spans = [int(grids[i].prod().item()) for i in range(2)]
        pad_id = tokenizer.convert_tokens_to_ids(DSV4_IMAGE_PAD_TOKEN)
        start_id = tokenizer.convert_tokens_to_ids(DSV4_VISION_START_TOKEN)
        end_id = tokenizer.convert_tokens_to_ids(DSV4_VISION_END_TOKEN)
        assert processor.marker_ids.vision_start_id == start_id
        assert processor.marker_ids.image_pad_id == pad_id
        assert processor.marker_ids.vision_end_id == end_id
        assert processor.image_token_id == pad_id
        expected = (
            [tokenizer.convert_tokens_to_ids("hello")]
            + [start_id]
            + [pad_id] * spans[0]
            + [end_id]
            + [tokenizer.convert_tokens_to_ids("cap")]
            + [start_id]
            + [pad_id] * spans[1]
            + [end_id]
            + [tokenizer.convert_tokens_to_ids("tail")]
        )
        assert result["input_ids"][0].tolist() == expected

    def test_marker_mismatch_and_missing_markers(self):
        processor, _, _ = self._processor()
        with pytest.raises(ValueError, match="image marker"):
            processor(text=[f"hello\n{FALCON_IMAGE_MARKER}\nworld"], images=[])
        with pytest.raises(ValueError, match="reserved visual marker"):
            self._processor(with_markers=False)


class TestFalconDeepseekv4Encoder:
    def test_markers_skipped_and_document_encode(self):
        processor, tokenizer, _ = TestFalconDeepseekv4Processor()._processor()
        encoder = FalconTaskEncoder(
            tokenizer=tokenizer,
            image_processor=processor,
            max_padding_length=2048,
            patch_size=FALCON_DEFAULT_PATCH_SIZE,
            spatial_merge_size=FALCON_DEFAULT_SPATIAL_MERGE_SIZE,
            min_pixels=FALCON_DEFAULT_MIN_PIXELS,
            max_pixels=1003520,
        )
        skipped = encoder._skipped_token_ids(processor).tolist()
        assert processor.marker_ids.vision_start_id in skipped
        assert processor.marker_ids.image_pad_id in skipped
        assert processor.marker_ids.vision_end_id in skipped

        document = ImageDocument.from_bytes(_tiff_bytes((64, 64), (64, 64)))
        encoded = _encode_full(
            encoder,
            _document_sample(
                ["Hello", ImagePageRef(document, 0), "cap0", "world", "tail", ImagePageRef(document, 1), "cap1"]
            ),
        )
        prepared = encoded.prepared_sequence
        assert prepared is not None and prepared.visual_values
        labels = prepared.row["labels"].tolist()
        loss_mask = prepared.row["loss_mask"]
        assert labels.count(IGNORE_INDEX) >= 2
        assert all(mask == 0.0 for label, mask in zip(labels, loss_mask.tolist()) if label == IGNORE_INDEX)
        assert loss_mask[-1] == 0


class TestFalconLlama3Processor:
    def _processor(self, *, with_markers: bool = True):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import PreTrainedTokenizerFast, Qwen2VLImageProcessor

        vocab = {"<unk>": 0, "hello": 1, "world": 2, "cap": 3, "tail": 4}
        if with_markers:
            vocab[LLAMA3_IMAGE_PAD_TOKEN] = 5
            vocab[LLAMA3_VISION_START_TOKEN] = 6
            vocab[LLAMA3_VISION_END_TOKEN] = 7
        backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
        backend.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend,
            unk_token="<unk>",
            pad_token="<unk>",
            eos_token="<unk>",
        )
        image_processor = Qwen2VLImageProcessor(
            size={"shortest_edge": 32, "longest_edge": 256},
            patch_size=32,
            temporal_patch_size=FALCON_DEFAULT_TEMPORAL_PATCH_SIZE,
            merge_size=1,
        )
        return (
            FalconLlama3Processor(tokenizer, image_processor, spatial_merge_size=1),
            tokenizer,
            image_processor,
        )

    def test_placeholder_expansion_and_order(self):
        """Each marker expands to [vision_start] + [image_pad x N] + [vision_end]."""
        processor, tokenizer, image_processor = self._processor()
        images = [Image.new("RGB", (64, 64), "red"), Image.new("RGB", (128, 64), "blue")]
        document = f"hello\n{FALCON_IMAGE_MARKER}\ncap\n{FALCON_IMAGE_MARKER}\ntail"
        result = processor(text=[document], images=images, return_tensors="pt")
        grids = image_processor(images=images, return_tensors="pt")["image_grid_thw"]
        spans = [int(grids[i].prod().item()) for i in range(2)]
        pad_id = tokenizer.convert_tokens_to_ids(LLAMA3_IMAGE_PAD_TOKEN)
        start_id = tokenizer.convert_tokens_to_ids(LLAMA3_VISION_START_TOKEN)
        end_id = tokenizer.convert_tokens_to_ids(LLAMA3_VISION_END_TOKEN)
        assert processor.marker_ids.vision_start_id == start_id
        assert processor.marker_ids.image_pad_id == pad_id
        assert processor.marker_ids.vision_end_id == end_id
        assert processor.image_token_id == pad_id
        expected = (
            [tokenizer.convert_tokens_to_ids("hello")]
            + [start_id]
            + [pad_id] * spans[0]
            + [end_id]
            + [tokenizer.convert_tokens_to_ids("cap")]
            + [start_id]
            + [pad_id] * spans[1]
            + [end_id]
            + [tokenizer.convert_tokens_to_ids("tail")]
        )
        assert result["input_ids"][0].tolist() == expected

    def test_marker_mismatch_and_missing_markers(self):
        processor, _, _ = self._processor()
        with pytest.raises(ValueError, match="image marker"):
            processor(text=[f"hello\n{FALCON_IMAGE_MARKER}\nworld"], images=[])
        with pytest.raises(ValueError, match="reserved visual marker"):
            self._processor(with_markers=False)
