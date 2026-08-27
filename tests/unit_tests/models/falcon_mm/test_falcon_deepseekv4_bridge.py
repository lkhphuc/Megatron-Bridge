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

"""Unit tests for the Falcon DeepSeek-V4 HybridModel checkpoint bridge."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import ReplicatedMapping
from megatron.bridge.models.deepseek.deepseek_v4_bridge import _HCAlphaMapping
from megatron.bridge.models.falcon_mm.falcon_deepseekv4_bridge import FalconDeepseekv4Bridge
from megatron.bridge.models.falcon_mm.falcon_deepseekv4_provider import (
    FalconDeepseekv4ModelProvider,
)


def _by_megatron(registry):
    """Index mappings by megatron_param for quick lookup."""
    return {m.megatron_param: m for m in registry.mappings}


def _falcon_dsv4_hf_config(
    *,
    num_hidden_layers: int = 2,
    num_hash_layers: int = 1,
    compress_ratios: list[int] | None = None,
):
    """Minimal DSv4 HF config for hybrid-bridge unit tests."""
    if compress_ratios is None:
        # Layer 0: sliding (no compressor); layer 1: CSA (compressor + indexer).
        compress_ratios = [0, 4]
    return SimpleNamespace(
        head_dim=512,
        qk_rope_head_dim=64,
        q_lora_rank=1024,
        o_groups=8,
        o_lora_rank=1024,
        rope_theta=10000,
        compress_rope_theta=160000,
        rope_scaling={"factor": 16, "original_max_position_embeddings": 65536, "rope_theta": 10000},
        num_hidden_layers=num_hidden_layers,
        num_nextn_predict_layers=1,
        num_hash_layers=num_hash_layers,
        compress_ratios=compress_ratios,
        sliding_window=128,
        index_n_heads=64,
        index_head_dim=128,
        index_topk=512,
        hc_mult=4,
        hc_sinkhorn_iters=20,
        scoring_func="sqrtsoftplus",
        num_experts_per_tok=6,
        norm_topk_prob=True,
        routed_scaling_factor=1.5,
        vocab_size=129280,
        swiglu_limit=10.0,
        moe_intermediate_size=1024,
        n_shared_experts=1,
        tie_word_embeddings=False,
        # Generic CONFIG_MAPPING fields used by MegatronModelBridge.provider_bridge
        hidden_size=256,
        intermediate_size=1024,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_local_experts=4,
        rms_norm_eps=1e-6,
        max_position_embeddings=4096,
    )


@pytest.fixture
def bridge_two_layers():
    """Bridge stubbed with a 2-layer HF config (ratios [0, 4], 1 hash layer)."""
    bridge = FalconDeepseekv4Bridge()
    bridge.hf_config = _falcon_dsv4_hf_config()
    return bridge


class TestGlobalMappings:
    """Global embed / head / norm / HC-head mappings under language_model."""

    @pytest.mark.parametrize(
        ("megatron_param", "hf_param"),
        [
            ("language_model.embedding.word_embeddings.weight", "embed.weight"),
            ("language_model.output_layer.weight", "head.weight"),
            ("language_model.decoder.final_norm.weight", "norm.weight"),
            ("language_model.decoder.hc_head_fn", "hc_head_fn"),
            ("language_model.decoder.hc_head_base", "hc_head_base"),
            ("language_model.decoder.hc_head_scale", "hc_head_scale"),
        ],
    )
    def test_global_mapping_present(self, bridge_two_layers, megatron_param, hf_param):
        mapping = _by_megatron(bridge_two_layers.mapping_registry()).get(megatron_param)
        assert mapping is not None, f"missing global mapping: {megatron_param}"
        assert mapping.hf_param == hf_param


class TestPatchEmbedMappings:
    """All eight patch-embed / merger mappings must be present."""

    @pytest.mark.parametrize(
        ("megatron_param", "hf_param"),
        [
            ("vision_model.patch_embed.proj.weight", "patch_embed.proj.weight"),
            ("vision_model.patch_embed.proj.bias", "patch_embed.proj.bias"),
            ("vision_model.merger.patch_norm.weight", "patch_embed.merger.patch_norm.weight"),
            ("vision_model.merger.patch_norm.bias", "patch_embed.merger.patch_norm.bias"),
            ("vision_model.merger.linear_fc1.weight", "patch_embed.merger.linear_fc1.weight"),
            ("vision_model.merger.linear_fc1.bias", "patch_embed.merger.linear_fc1.bias"),
            ("vision_model.merger.linear_fc2.weight", "patch_embed.merger.linear_fc2.weight"),
            ("vision_model.merger.linear_fc2.bias", "patch_embed.merger.linear_fc2.bias"),
        ],
    )
    def test_patch_embed_mapping_present(self, bridge_two_layers, megatron_param, hf_param):
        mapping = _by_megatron(bridge_two_layers.mapping_registry()).get(megatron_param)
        assert mapping is not None, f"missing patch-embed mapping: {megatron_param}"
        assert mapping.hf_param == hf_param


class TestLayerSplit:
    """HF layer n → hybrid attention 2n + MoE 2n+1."""

    def test_hf_layer_0_at_hybrid_0_and_1(self, bridge_two_layers):
        params = _by_megatron(bridge_two_layers.mapping_registry())
        assert "language_model.decoder.layers.0.inner_layer.input_layernorm.weight" in params
        assert params["language_model.decoder.layers.0.inner_layer.input_layernorm.weight"].hf_param == (
            "layers.0.attn_norm.weight"
        )
        assert "language_model.decoder.layers.1.inner_layer.pre_mlp_layernorm.weight" in params
        assert params["language_model.decoder.layers.1.inner_layer.pre_mlp_layernorm.weight"].hf_param == (
            "layers.0.ffn_norm.weight"
        )

    def test_hf_layer_1_at_hybrid_2_and_3(self, bridge_two_layers):
        params = _by_megatron(bridge_two_layers.mapping_registry())
        assert "language_model.decoder.layers.2.inner_layer.input_layernorm.weight" in params
        assert params["language_model.decoder.layers.2.inner_layer.input_layernorm.weight"].hf_param == (
            "layers.1.attn_norm.weight"
        )
        assert "language_model.decoder.layers.3.inner_layer.pre_mlp_layernorm.weight" in params
        assert params["language_model.decoder.layers.3.inner_layer.pre_mlp_layernorm.weight"].hf_param == (
            "layers.1.ffn_norm.weight"
        )


class TestCompressorAndIndexer:
    """Compressor for ratio > 0; indexer only for ratio == 4."""

    def test_ratio_0_has_no_compressor_or_indexer(self, bridge_two_layers):
        params = _by_megatron(bridge_two_layers.mapping_registry())
        # HF layer 0 → hybrid attn index 0, ratio 0
        assert (
            "language_model.decoder.layers.0.inner_layer.self_attention.core_attention.compressor.linear_wkv.weight"
            not in params
        )
        assert (
            "language_model.decoder.layers.0.inner_layer.self_attention.core_attention.indexer.linear_wq_b.weight"
            not in params
        )

    def test_ratio_4_has_compressor_and_indexer(self, bridge_two_layers):
        params = _by_megatron(bridge_two_layers.mapping_registry())
        # HF layer 1 → hybrid attn index 2, ratio 4
        assert (
            "language_model.decoder.layers.2.inner_layer.self_attention.core_attention.compressor.linear_wkv.weight"
            in params
        )
        assert (
            "language_model.decoder.layers.2.inner_layer.self_attention.core_attention.indexer.linear_wq_b.weight"
            in params
        )
        assert (
            "language_model.decoder.layers.2.inner_layer.self_attention."
            "core_attention.indexer.compressor.linear_wkv.weight" in params
        )


class TestHashRouting:
    """Hash layers get tid2eid; non-hash layers get expert_bias."""

    def test_hash_layer_0_has_tid2eid_not_bias(self, bridge_two_layers):
        params = _by_megatron(bridge_two_layers.mapping_registry())
        # HF layer 0 MoE → hybrid index 1
        assert "language_model.decoder.layers.1.inner_layer.mlp.router.tid2eid" in params
        assert "language_model.decoder.layers.1.inner_layer.mlp.router.expert_bias" not in params

    def test_non_hash_layer_1_has_bias_not_tid2eid(self, bridge_two_layers):
        params = _by_megatron(bridge_two_layers.mapping_registry())
        # HF layer 1 MoE → hybrid index 3
        assert "language_model.decoder.layers.3.inner_layer.mlp.router.expert_bias" in params
        assert "language_model.decoder.layers.3.inner_layer.mlp.router.tid2eid" not in params


class TestHCAlphaMappings:
    """HC alpha primary mappings exist for both attn and ffn hybrid layers."""

    def test_attn_and_ffn_alpha_for_hf_layer_0(self, bridge_two_layers):
        registry = bridge_two_layers.mapping_registry()
        params = _by_megatron(registry)

        attn_pre = "language_model.decoder.layers.0.hyper_connection.alpha_pre"
        ffn_pre = "language_model.decoder.layers.1.hyper_connection.alpha_pre"
        assert attn_pre in params
        assert ffn_pre in params
        assert isinstance(params[attn_pre], _HCAlphaMapping)
        assert isinstance(params[ffn_pre], _HCAlphaMapping)
        assert params[attn_pre].hf_param == "layers.0.hc_attn_scale"
        assert params[ffn_pre].hf_param == "layers.0.hc_ffn_scale"

        # Secondary post/res registered to suppress export warnings
        assert "language_model.decoder.layers.0.hyper_connection.alpha_post" in params
        assert "language_model.decoder.layers.1.hyper_connection.alpha_res" in params

    def test_hc_fn_and_base_are_replicated(self, bridge_two_layers):
        params = _by_megatron(bridge_two_layers.mapping_registry())
        attn_fn = params["language_model.decoder.layers.0.hyper_connection.mapping_proj.weight"]
        ffn_fn = params["language_model.decoder.layers.1.hyper_connection.mapping_proj.weight"]
        assert isinstance(attn_fn, ReplicatedMapping)
        assert isinstance(ffn_fn, ReplicatedMapping)
        assert attn_fn.hf_param == "layers.0.hc_attn_fn"
        assert ffn_fn.hf_param == "layers.0.hc_ffn_fn"


class TestProviderBridge:
    """provider_bridge maps key DSv4 fields onto FalconDeepseekv4ModelProvider."""

    def test_key_fields(self):
        hf_pretrained = MagicMock()
        hf_pretrained.config = _falcon_dsv4_hf_config(
            num_hidden_layers=2,
            num_hash_layers=1,
            compress_ratios=[0, 4, 0],  # extra MTP ratio must be sliced off
        )
        # Seed provider with dataclass-valid defaults; provider_bridge overwrites.
        provider = FalconDeepseekv4ModelProvider(
            num_layers=1,
            hidden_size=256,
            num_attention_heads=4,
            ffn_hidden_size=1024,
            num_moe_experts=4,
            vocab_size=129280,
        )

        bridge = FalconDeepseekv4Bridge.__new__(FalconDeepseekv4Bridge)
        with patch.object(MegatronModelBridge, "provider_bridge", return_value=provider):
            out = bridge.provider_bridge(hf_pretrained)

        assert out is provider
        assert isinstance(out, FalconDeepseekv4ModelProvider)
        assert out.num_layers == 2
        assert out.v_head_dim == 512
        assert out.qk_pos_emb_head_dim == 64
        assert out.q_lora_rank == 1024
        assert out.o_groups == 8
        assert out.o_lora_rank == 1024
        assert out.rotary_percent == 1.0
        assert out.rope_type == "yarn"
        assert out.moe_n_hash_layers == 1
        assert out.mtp_num_layers == 0
        assert out.csa_compress_ratios == [0, 4]
        assert len(out.csa_compress_ratios) == out.num_layers
        assert out.make_vocab_size_divisible_by == 1280
        assert out.seq_length == 4096
        assert out.enable_hyper_connections is True
        assert out.num_residual_streams == 4
        assert out.dsa_kernel_backend in ("cudnn", "none")
