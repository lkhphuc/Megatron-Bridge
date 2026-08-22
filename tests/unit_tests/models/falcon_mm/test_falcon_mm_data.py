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

"""Unit tests for the Falcon-MM Energon task-encoder config and document path."""

import json
from unittest.mock import MagicMock

import pytest
import torch
from megatron.energon import SkipSample
from megatron.energon.flavors.interleaved import InterleavedSample
from PIL import Image

from megatron.bridge.data.builders.energon import QwenVLEnergonTaskEncoderConfig
from megatron.bridge.data.datasets.utils import IGNORE_INDEX
from megatron.bridge.models.falcon_mm.data import (
    FalconEnergonTaskEncoderConfig,
    FalconTaskEncoder,
)
from megatron.bridge.models.falcon_mm.data.energon import FALCON_IMAGE_MARKER
from megatron.bridge.models.falcon_mm.data.mint_webdataset import (
    mint_interleaved_sample_loader,
    tiffhandler,
)
from megatron.bridge.models.qwen_vl.data.energon import QwenVLTaskEncoder, QwenVLTaskSample


def _make_tokenizer():
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 1
    tokenizer.image_token_id = 151655
    tokenizer.video_token_id = 151656
    return tokenizer


def _make_document_processor(input_ids=(5, 6, 7, 8), *, with_visuals=False, skipped_tokens=None):
    """Fake HF processor for the document path.

    ``__call__`` returns fixed ``input_ids``/``attention_mask`` tensors (plus
    visual payloads when ``with_visuals``); ``added_tokens_decoder`` backs
    ``extract_skipped_token_ids``.
    """
    processor = MagicMock()
    processor.added_tokens_decoder = skipped_tokens or {}
    ids = torch.tensor([list(input_ids)], dtype=torch.long)
    output = {
        "input_ids": ids,
        "attention_mask": torch.ones_like(ids),
    }
    if with_visuals:
        output["pixel_values"] = torch.randn(4, 8)
        output["image_grid_thw"] = torch.tensor([[1, 2, 2], [1, 2, 2]])
    processor.return_value = output
    return processor


class TestFalconEnergonTaskEncoderConfig:
    def test_passes_qwen_isinstance_gate(self):
        """The packing gate in data/builders/energon.py matches the Qwen config.

        The Falcon config must remain a subclass so Energon-native packing is
        accepted with no shared-code edits.
        """
        config = FalconEnergonTaskEncoderConfig(hf_processor_path="Qwen/Qwen3.5-2B")
        assert isinstance(config, QwenVLEnergonTaskEncoderConfig)

    def test_inherits_qwen_defaults(self):
        config = FalconEnergonTaskEncoderConfig(hf_processor_path="Qwen/Qwen3.5-2B")
        assert config.spatial_merge_size == 2
        assert config.max_num_images == 10
        assert config.max_visual_tokens == 16384

    def test_falcon_task_encoder_is_qwen_task_encoder(self):
        assert issubclass(FalconTaskEncoder, QwenVLTaskEncoder)


class TestMintInterleavedSampleLoader:
    """MINT json+tiff assembly into Energon's standard InterleavedSample."""

    def test_assembles_sequence_in_document_order(self):
        pages = [torch.zeros(3, 4, 4), torch.ones(3, 4, 4)]
        result = mint_interleaved_sample_loader(
            {
                "__key__": "k",
                "json": {
                    "texts": ["t0", "t1", "t2"],
                    "images": ["x", None, "y"],
                    "captions": ["c0", "c1"],
                },
                "tiff": pages,
            }
        )
        assert result["sequence"] == ["t0", pages[0], "c0", "t1", "t2", pages[1], "c1"]

    def test_rejects_page_count_mismatch(self):
        with pytest.raises(ValueError, match="tiff pages"):
            mint_interleaved_sample_loader(
                {"__key__": "k", "json": {"texts": ["a"], "images": ["x"], "captions": []}, "tiff": []}
            )

    def test_tiffhandler_decodes_all_pages(self):
        import io

        buffer = io.BytesIO()
        Image.new("RGB", (4, 4), "red").save(
            buffer, format="TIFF", save_all=True, append_images=[Image.new("RGB", (4, 4), "blue")]
        )
        handler = tiffhandler()
        pages = handler("sample.tiff", buffer.getvalue())
        assert len(pages) == 2
        assert pages[0].shape == (3, 4, 4)
        assert pages[0].dtype == torch.float32
        assert handler("sample.json", b"{}") is None


class TestFalconTaskEncoderDocumentPath:
    """Document-style pretraining encoding (InterleavedSample + JSONL dicts)."""

    def _make_encoder(self, processor=None, **kwargs):
        kwargs.setdefault("max_padding_length", 512)
        kwargs.setdefault("patch_size", 14)
        kwargs.setdefault("spatial_merge_size", 2)
        return FalconTaskEncoder(
            tokenizer=_make_tokenizer(),
            image_processor=processor or _make_document_processor(),
            **kwargs,
        )

    @staticmethod
    def _crude_sample(payload):
        return {
            "json": payload,
            "__key__": "train.jsonl/0",
            "__shard__": "train.jsonl",
            "__restore_key__": ("train.jsonl", 0),
            "__sources__": (),
            "__subflavors__": {},
        }

    def test_cooker_passes_crude_sample_through(self):
        """Energon requires a registered cooker for crude (JSONL) datasets."""
        encoder = self._make_encoder()
        sample = self._crude_sample({"text": "hello"})
        assert len(encoder.cookers) == 1
        assert encoder.cookers[0].cook(sample) is sample

    def test_encode_interleaved_sample(self):
        processor = _make_document_processor(with_visuals=True)
        encoder = self._make_encoder(processor=processor)
        image_tensors = [torch.rand(3, 8, 8), torch.rand(3, 8, 8)]
        sample = InterleavedSample(
            __key__="shard/0",
            __restore_key__=("shard", 0),
            __subflavors__={},
            sequence=[
                "Hello",
                image_tensors[0],
                "cap0",
                "world",
                "tail",
                image_tensors[1],
                "cap1",
            ],
        )

        encoded = encoder.encode_sample(sample)

        assert isinstance(encoded, QwenVLTaskSample)
        assert encoded.example is None
        prepared = encoded.prepared_sequence
        assert prepared is not None
        assert prepared.visual_values
        loss_mask = prepared.row["loss_mask"]
        assert loss_mask.dtype == torch.float32
        assert torch.all(loss_mask[:-1] == 1)
        assert loss_mask[-1] == 0

        # Document assembly: markers replace image parts in sequence order.
        text = processor.call_args.kwargs["text"][0]
        assert text.count(FALCON_IMAGE_MARKER) == 2
        assert text == f"Hello\n{FALCON_IMAGE_MARKER}\ncap0\nworld\ntail\n{FALCON_IMAGE_MARKER}\ncap1"
        images = processor.call_args.kwargs["images"]
        assert len(images) == 2
        assert all(isinstance(image, Image.Image) for image in images)

    def test_encode_crude_text_dict(self):
        processor = _make_document_processor(input_ids=(5, 6, 7, 8), skipped_tokens={7: "<|image_pad|>"})
        encoder = self._make_encoder(processor=processor)
        sample = self._crude_sample({"text": "a plain document"})

        encoded = encoder.encode_sample(sample)

        assert isinstance(encoded, QwenVLTaskSample)
        assert encoded.example is None
        assert encoded.__key__ == "train.jsonl/0"
        assert encoded.__restore_key__ == ("train.jsonl", 0)
        prepared = encoded.prepared_sequence
        assert prepared is not None
        assert prepared.visual_values == {}
        row = prepared.row
        # Loss mask follows the masked labels: the position predicting the
        # skipped visual token and the final position are excluded.
        assert row["loss_mask"].tolist() == [1.0, 0.0, 1.0, 0.0]
        # Labels shift left; skipped (visual/special) label tokens are masked.
        assert row["labels"].tolist() == [6, IGNORE_INDEX, 8, IGNORE_INDEX]

    def test_document_text_is_kept_whole(self):
        """Documents are never char-truncated; length control is post-tokenization."""
        processor = _make_document_processor()
        encoder = self._make_encoder(processor=processor)
        sample = self._crude_sample({"text": "x" * 100})

        encoder.encode_sample(sample)

        text = processor.call_args.kwargs["text"][0]
        assert text == "x" * 100

    def test_overlong_document_is_dropped_after_expansion(self):
        """A document whose expanded token count exceeds seq_length is skipped."""
        processor = _make_document_processor(input_ids=tuple(range(10)))
        encoder = self._make_encoder(processor=processor, max_padding_length=8)
        sample = self._crude_sample({"text": "a plain document"})

        with pytest.raises(SkipSample):
            encoder.encode_sample(sample)

    def test_encode_crude_sample_rejects_messages_payload(self):
        encoder = self._make_encoder()
        sample = self._crude_sample({"messages": [{"role": "user", "content": "Hi"}]})

        with pytest.raises(ValueError, match="JSONL text sample"):
            encoder.encode_sample(sample)

    def test_encode_crude_sample_rejects_bare_list_payload(self):
        encoder = self._make_encoder()
        sample = self._crude_sample([{"role": "user", "content": "Hi"}])

        with pytest.raises(ValueError, match="JSONL text sample"):
            encoder.encode_sample(sample)

    def test_encode_crude_sample_rejects_non_string_text(self):
        encoder = self._make_encoder()
        sample = self._crude_sample({"text": 123})

        with pytest.raises(ValueError, match="JSONL text sample"):
            encoder.encode_sample(sample)

    def test_encode_chatml_sample_still_works(self):
        from megatron.bridge.data.energon.metadata import sample_metadata_kwargs
        from megatron.bridge.data.energon.task_encoder_utils import ChatMLSample

        encoder = self._make_encoder()
        sample = ChatMLSample(
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

        encoded = encoder.encode_sample(sample)

        assert isinstance(encoded, QwenVLTaskSample)
