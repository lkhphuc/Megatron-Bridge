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

"""Unit tests for the Falcon-MM DeepSeek-V4 provider (encoder-free DSv4 trunk)."""

import pytest
import torch
from megatron.core.utils import get_attr_wrapped_model

from megatron.bridge.models.falcon_mm import (
    FalconDeepseekv4Model,
    FalconDeepseekv4ModelProvider,
    falcon_deepseekv4_layer_pattern,
    falcon_dsv4_stack_spec,
)


def _minimal_dsv4_provider() -> FalconDeepseekv4ModelProvider:
    """A tiny DSv4 provider; architecture values are irrelevant here."""
    return FalconDeepseekv4ModelProvider(
        num_layers=4,
        hidden_size=256,
        num_attention_heads=4,
        seq_length=64,
        vocab_size=129280,
        v_head_dim=64,
        qk_pos_emb_head_dim=16,
        q_lora_rank=64,
        o_groups=2,
        o_lora_rank=64,
        csa_compress_ratios=[0, 4, 128, 4],
        csa_window_size=32,
        dsa_indexer_n_heads=4,
        dsa_indexer_head_dim=32,
        dsa_indexer_topk=16,
        num_moe_experts=8,
        moe_router_topk=2,
        moe_ffn_hidden_size=64,
        moe_shared_expert_intermediate_size=64,
        moe_layer_freq=[1, 1, 1, 1],
        moe_n_hash_layers=2,
        actual_vocab_size=129280,
    )


class TestFalconDeepseekv4ModelProvider:
    def test_model_class_is_falcon_deepseekv4(self):
        provider = _minimal_dsv4_provider()
        assert provider.MODEL_CLASS is FalconDeepseekv4Model

    def test_dsv4_architecture_defaults(self):
        provider = _minimal_dsv4_provider()
        assert provider.experimental_attention_variant == "dsv4_hybrid"
        assert provider.multi_latent_attention is True
        assert provider.qk_layernorm is True
        assert provider.gated_linear_unit is True
        assert provider.position_embedding_type == "rope"
        assert provider.mrope_section == [11, 11, 10]
        assert provider.mrope_interleaved is True
        assert provider.hybrid_stack_spec is falcon_dsv4_stack_spec
        assert provider.moe_vl_routing is False

    def test_falcon_surface_defaults(self):
        provider = _minimal_dsv4_provider()
        assert provider.image_token_id == 128000
        assert provider.vision_start_token_id == 128001
        assert provider.patch_size == 32
        assert provider.temporal_patch_size == 1
        assert provider.spatial_merge_size == 1
        assert provider.in_channels == 3
        assert provider.patch_embed_hidden_size == 1024

    def test_patch_embed_config_fields(self):
        provider = _minimal_dsv4_provider()
        config = provider._build_patch_embed_config()
        assert config.hidden_size == provider.patch_embed_hidden_size
        assert config.out_hidden_size == provider.hidden_size
        assert config.patch_size == provider.patch_size
        assert config.temporal_patch_size == provider.temporal_patch_size
        assert config.in_channels == provider.in_channels
        assert config.spatial_merge_size == provider.spatial_merge_size
        assert config.pipeline_model_parallel_size == 1

    def test_pattern_generation_inputs(self):
        # The provider generates its pattern from csa_compress_ratios; the
        # helper itself is covered by test_falcon_deepseekv4_pattern.py.
        provider = _minimal_dsv4_provider()
        pattern = falcon_deepseekv4_layer_pattern(
            provider.csa_compress_ratios[: provider.num_layers],
            pipeline_model_parallel_size=provider.pipeline_model_parallel_size or 1,
            mtp_num_layers=provider.mtp_num_layers or 0,
            num_hash_layers=provider.moe_n_hash_layers or 0,
        )
        assert pattern == "WECEHECE"

    def test_pattern_ratio_shortfall_raises(self):
        provider = _minimal_dsv4_provider()
        provider.num_layers = 8  # more layers than ratios provided
        provider.vocab_size = 129280
        with pytest.raises(ValueError, match="csa_compress_ratios"):
            provider.provide(pre_process=True, post_process=True)

    def test_pp_wrapper_surfaces_embedding_gradient_state(self):
        class _LanguageModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.post_process = True
                self.embedding_activation_buffer = [torch.ones(1)]
                self.grad_output_buffer = [torch.ones(1)]
                self.output_layer = torch.nn.Linear(1, 1, bias=False)

            def shared_embedding_or_output_weight(self) -> torch.nn.Parameter:
                return self.output_layer.weight

        model = FalconDeepseekv4Model.__new__(FalconDeepseekv4Model)
        torch.nn.Module.__init__(model)
        model.pre_process = False
        model.post_process = True
        model.share_embeddings_and_output_weights = True
        model.language_model = _LanguageModel()

        # MCore's attribute resolver stops at this multimodal wrapper for both
        # PP boundary flags, so every property consumed by embedding-gradient
        # finalization must be available here.
        assert get_attr_wrapped_model(model, "pre_process", return_model_obj=True) is model
        assert get_attr_wrapped_model(model, "post_process", return_model_obj=True) is model
        assert model.shared_embedding_or_output_weight() is model.language_model.output_layer.weight
        assert model.embedding_activation_buffer is model.language_model.embedding_activation_buffer
        assert model.grad_output_buffer is model.language_model.grad_output_buffer
        assert model.output_layer is model.language_model.output_layer
