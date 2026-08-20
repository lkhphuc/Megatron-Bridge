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

"""Unit tests for the Falcon-MM provider (encoder-free Qwen3.5 trunk)."""

from megatron.bridge.models.falcon_mm import (
    FalconMMModel,
    FalconMMModelProvider,
    FalconPatchEmbedVisionModel,
)
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model import Qwen3VLModel
from megatron.bridge.models.qwen_vl.qwen35_vl_provider import Qwen35VLModelProvider


def _minimal_qwen35_vl_provider() -> Qwen35VLModelProvider:
    """A tiny Qwen3.5-VL provider; architecture values are irrelevant here."""
    return Qwen35VLModelProvider(
        num_layers=2,
        hidden_size=64,
        ffn_hidden_size=128,
        num_attention_heads=4,
        kv_channels=32,
        seq_length=64,
        vocab_size=128,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
    )


class TestFalconMMModelProvider:
    def test_model_class_is_falcon(self):
        provider = _minimal_qwen35_vl_provider()
        assert provider.MODEL_CLASS is None  # base resolves to Qwen3VLModel in provide()
        falcon = FalconMMModelProvider.from_qwen35_vl_provider(provider)
        assert falcon.MODEL_CLASS is FalconMMModel
        assert issubclass(FalconMMModel, Qwen3VLModel)

    def test_from_qwen35_vl_provider_copies_fields(self):
        source = _minimal_qwen35_vl_provider()
        falcon = FalconMMModelProvider.from_qwen35_vl_provider(source)
        assert isinstance(falcon, FalconMMModelProvider)
        for field_name in (
            "num_layers",
            "hidden_size",
            "ffn_hidden_size",
            "num_attention_heads",
            "seq_length",
            "vocab_size",
            "experimental_attention_variant",
            "linear_attention_freq",
            "mrope_section",
            "image_token_id",
            "spatial_merge_size",
            "patch_size",
        ):
            assert getattr(falcon, field_name) == getattr(source, field_name), field_name
        # vision_config carries the patch geometry for the patch embed; shared by reference.
        assert falcon.vision_config is source.vision_config
        # GDN hybrid defaults flow through unchanged.
        assert falcon.experimental_attention_variant == "gated_delta_net"
        assert falcon.linear_attention_freq == 4

    def test_patch_embed_module_exists(self):
        assert FalconPatchEmbedVisionModel is not None
