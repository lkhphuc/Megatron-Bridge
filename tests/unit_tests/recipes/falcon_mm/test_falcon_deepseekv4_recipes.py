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

"""Unit tests for Falcon-MM DeepSeek-V4 scaling recipes."""

from megatron.bridge.recipes.falcon_mm.dataset import FALCON_ENERGON_MICRO_BATCH_SIZE
from megatron.bridge.recipes.falcon_mm.falcon_deepseekv4_common import (
    FALCON_DSV4_DENSE_1B_PATTERN,
    FALCON_DSV4_DENSE_3B_PATTERN,
    FALCON_DSV4_DENSE_10B_PATTERN,
    FALCON_DSV4_MOE_35B_A3B_PATTERN,
    falcon_deepseekv4_1b_provider,
    falcon_deepseekv4_3b_provider,
    falcon_deepseekv4_10b_provider,
    falcon_deepseekv4_35b_a3b_provider,
)


class TestFalconDeepseekv4ScalingProviders:
    def test_dense_scale_geometry(self):
        expected = [
            (falcon_deepseekv4_1b_provider, 16, 1280, 3584, 256),
            (falcon_deepseekv4_3b_provider, 24, 2048, 5632, 512),
            (falcon_deepseekv4_10b_provider, 32, 4096, 11264, 1024),
        ]
        for factory, num_decoder_layers, hidden_size, ffn_hidden_size, q_lora_rank in expected:
            provider = factory()
            provider.finalize()
            assert provider.num_layers == 2 * num_decoder_layers
            assert provider.hidden_size == hidden_size
            assert provider.ffn_hidden_size == ffn_hidden_size
            assert provider.q_lora_rank == q_lora_rank
            assert provider.o_lora_rank == q_lora_rank
            assert provider.num_moe_experts is None
            assert provider.moe_n_hash_layers == 0
            assert provider.moe_layer_freq == [0, 0] * num_decoder_layers
            assert provider.moe_grouped_gemm is False
            assert len(provider.csa_compress_ratios) == provider.num_layers

        assert falcon_deepseekv4_1b_provider().hybrid_layer_pattern == FALCON_DSV4_DENSE_1B_PATTERN
        assert falcon_deepseekv4_3b_provider().hybrid_layer_pattern == FALCON_DSV4_DENSE_3B_PATTERN
        assert falcon_deepseekv4_10b_provider().hybrid_layer_pattern == FALCON_DSV4_DENSE_10B_PATTERN

    def test_moe_35b_a3b_geometry(self):
        provider = falcon_deepseekv4_35b_a3b_provider()
        provider.finalize()
        assert provider.hybrid_layer_pattern == FALCON_DSV4_MOE_35B_A3B_PATTERN
        assert provider.num_layers == 80
        assert provider.hidden_size == 2048
        assert provider.num_moe_experts == 256
        assert provider.moe_router_topk == 8
        assert provider.moe_ffn_hidden_size == 512
        assert provider.moe_shared_expert_intermediate_size == 512
        assert provider.moe_n_hash_layers == 3
        assert provider.moe_layer_freq == [0, 1] * 40
        assert provider.moe_vl_routing is True
        assert len(provider.csa_compress_ratios) == 80

    def test_dense_1b_recipe_is_config_container(self):
        from megatron.bridge.recipes.falcon_mm.b200.falcon_deepseekv4 import (
            falcon_deepseekv4_1b_pretrain_8gpu_b200_bf16_config,
        )
        from megatron.bridge.training.config import ConfigContainer

        cfg = falcon_deepseekv4_1b_pretrain_8gpu_b200_bf16_config()
        assert isinstance(cfg, ConfigContainer)
        assert cfg.model.hybrid_layer_pattern == FALCON_DSV4_DENSE_1B_PATTERN
        assert cfg.model.expert_model_parallel_size == 1
        assert cfg.train.micro_batch_size == FALCON_ENERGON_MICRO_BATCH_SIZE
        # Short alias points at the B200 recipe (shared fused path).
        assert cfg.model.dsa_kernel_backend == "cudnn"
        assert cfg.model.apply_rope_fusion is True
        assert cfg.dist.enable_megatron_core_experimental is True
        assert cfg.env_vars["CUDA_DEVICE_MAX_CONNECTIONS"] == 32
        assert cfg.env_vars["NVLINK_DOMAIN_SIZE"] == 8
        # Dense: no HybridEP / GroupedGEMM knobs.
        assert cfg.model.moe_token_dispatcher_type != "flex" or cfg.model.num_moe_experts is None
        assert cfg.model.moe_grouped_gemm is False

    def test_b200_dense_and_moe_scaling_share_blackwell_flags(self):
        from megatron.bridge.recipes.falcon_mm.b200.falcon_deepseekv4 import (
            falcon_deepseekv4_1b_pretrain_8gpu_b200_bf16_config,
            falcon_deepseekv4_3b_pretrain_8gpu_b200_bf16_config,
            falcon_deepseekv4_10b_pretrain_8gpu_b200_bf16_config,
            falcon_deepseekv4_35b_a3b_pretrain_8gpu_b200_bf16_config,
        )

        dense_cfgs = [
            falcon_deepseekv4_1b_pretrain_8gpu_b200_bf16_config(),
            falcon_deepseekv4_3b_pretrain_8gpu_b200_bf16_config(),
            falcon_deepseekv4_10b_pretrain_8gpu_b200_bf16_config(),
        ]
        for cfg in dense_cfgs:
            assert cfg.model.dsa_kernel_backend == "cudnn"
            assert cfg.model.apply_rope_fusion is True
            assert cfg.model.dsa_indexer_loss_coeff == 0.01
            assert cfg.model.dsa_indexer_use_sparse_loss is True
            assert cfg.dist.enable_megatron_core_experimental is True
            assert cfg.env_vars["CUDA_DEVICE_MAX_CONNECTIONS"] == 32
            assert cfg.model.expert_model_parallel_size == 1
            assert cfg.model.moe_grouped_gemm is False
            assert cfg.model.moe_token_dispatcher_type != "flex"

        moe_cfg = falcon_deepseekv4_35b_a3b_pretrain_8gpu_b200_bf16_config()
        assert moe_cfg.model.expert_model_parallel_size == 8
        assert moe_cfg.model.moe_token_dispatcher_type == "flex"
        assert moe_cfg.model.moe_flex_dispatcher_backend == "hybridep"
        assert moe_cfg.model.moe_grouped_gemm is True
        assert moe_cfg.env_vars["NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN"] == 8
        assert moe_cfg.env_vars["CUDA_DEVICE_MAX_CONNECTIONS"] == 32

    def test_b200_moe_35b_a3b_mxfp8_qwen30b_throughput_shape(self):
        from megatron.bridge.recipes.falcon_mm.b200.falcon_deepseekv4 import (
            falcon_deepseekv4_35b_a3b_pretrain_8gpu_b200_fp8mx_config,
        )
        from megatron.bridge.recipes.falcon_mm.dataset import FALCON_ENERGON_MICRO_BATCH_SIZE

        cfg = falcon_deepseekv4_35b_a3b_pretrain_8gpu_b200_fp8mx_config()
        assert cfg.model.seq_length == 4096
        assert cfg.dataset.seq_length == 4096
        assert cfg.train.global_batch_size == 512
        assert cfg.train.micro_batch_size == FALCON_ENERGON_MICRO_BATCH_SIZE
        assert cfg.train.micro_batch_size == 1
        assert cfg.mixed_precision.fp8_recipe == "mxfp8"
        assert cfg.model.moe_router_padding_for_fp8 is True
        assert cfg.model.dsa_indexer_loss_coeff == 0.01
        assert cfg.model.dsa_indexer_use_sparse_loss is True
        assert cfg.model.recompute_granularity is None
        assert cfg.model.recompute_modules is None
        assert cfg.model.cuda_graph_impl == "transformer_engine"
        assert cfg.dataset.pad_to_max_length is True
        assert cfg.model.moe_flex_dispatcher_num_sms == 64
        assert cfg.model.moe_router_force_load_balancing is False
        assert cfg.comm_overlap.overlap_moe_expert_parallel_comm is False
        assert cfg.comm_overlap.delay_wgrad_compute is False
        assert cfg.ddp.average_in_collective is False
        assert cfg.env_vars["NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN"] == 8
        assert cfg.env_vars["NVTE_CUTEDSL_FUSED_GROUPED_MLP"] == 1
