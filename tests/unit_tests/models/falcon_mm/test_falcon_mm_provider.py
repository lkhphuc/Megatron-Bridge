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
    FalconPatchEmbedVisionModel,
    FalconQwen35Model,
    FalconQwen35ModelProvider,
    FalconQwen35MoEModelProvider,
    falcon_qwen35_provider_from_qwen35_vl,
)
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model import Qwen3VLModel
from megatron.bridge.models.qwen_vl.qwen35_vl_provider import Qwen35VLModelProvider, Qwen35VLMoEModelProvider


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


def _minimal_qwen35_vl_moe_provider() -> Qwen35VLMoEModelProvider:
    return Qwen35VLMoEModelProvider(
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
        num_moe_experts=8,
        moe_router_topk=2,
        moe_ffn_hidden_size=32,
        moe_shared_expert_intermediate_size=32,
    )


class TestFalconQwen35ModelProvider:
    def test_direct_provider_keeps_custom_hmlp_geometry(self):
        provider = FalconQwen35ModelProvider(
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
            patch_size=16,
            patch_embed_num_layers=2,
            patch_embed_hidden_size=11,
        )
        assert provider.patch_size == provider.vision_config.patch_size == 16
        assert provider.vision_config.hmlp_num_layers == 2
        assert provider.vision_config.hmlp_intermediate_size == 11
        assert provider.vision_config.out_hidden_size == provider.hidden_size
        assert provider.spatial_merge_size == provider.vision_config.spatial_merge_size == 1
        assert provider.temporal_patch_size == provider.vision_config.temporal_patch_size == 1

    def test_model_class_is_falcon(self):
        provider = _minimal_qwen35_vl_provider()
        assert provider.MODEL_CLASS is None  # base resolves to Qwen3VLModel in provide()
        falcon = falcon_qwen35_provider_from_qwen35_vl(provider)
        assert falcon.MODEL_CLASS is FalconQwen35Model
        assert issubclass(FalconQwen35Model, Qwen3VLModel)

    def test_falcon_qwen35_provider_from_qwen35_vl_copies_fields(self):
        source = _minimal_qwen35_vl_provider()
        falcon = falcon_qwen35_provider_from_qwen35_vl(source)
        assert isinstance(falcon, FalconQwen35ModelProvider)
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
        ):
            assert getattr(falcon, field_name) == getattr(source, field_name), field_name
        # Falcon early-fusion geometry (not HF ViT 16/2/2).
        assert source.patch_size == 16
        assert source.spatial_merge_size == 2
        assert source.temporal_patch_size == 2
        assert falcon.patch_size == 32
        assert falcon.spatial_merge_size == 1
        assert falcon.temporal_patch_size == 1
        assert falcon.patch_embed_num_layers == 3
        assert falcon.vision_config.hmlp_num_layers == 3
        assert falcon.vision_config is source.vision_config
        assert falcon.vision_config.patch_size == falcon.patch_size
        assert falcon.vision_config.spatial_merge_size == falcon.spatial_merge_size
        assert falcon.vision_config.temporal_patch_size == falcon.temporal_patch_size
        assert falcon.vision_config.out_hidden_size == falcon.hidden_size
        # GDN hybrid defaults flow through unchanged (alias may be "gdn").
        assert falcon.experimental_attention_variant == source.experimental_attention_variant
        assert falcon.linear_attention_freq == 4

    def test_falcon_qwen35_provider_from_qwen35_vl_moe_returns_moe_falcon(self):
        source = _minimal_qwen35_vl_moe_provider()
        falcon = falcon_qwen35_provider_from_qwen35_vl(source)
        assert isinstance(falcon, FalconQwen35MoEModelProvider)
        assert falcon.MODEL_CLASS is FalconQwen35Model
        assert falcon.num_moe_experts == source.num_moe_experts
        assert falcon.moe_router_topk == source.moe_router_topk
        assert falcon.moe_ffn_hidden_size == source.moe_ffn_hidden_size

    def test_patch_embed_module_exists(self):
        assert FalconPatchEmbedVisionModel is not None
