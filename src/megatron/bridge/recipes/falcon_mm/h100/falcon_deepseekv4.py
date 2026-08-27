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

"""Falcon-MM DeepSeek-V4 (encoder-free, HybridModel trunk) recipes for H100.

Launch with the VLM step (CP=1 at this time; the DSv4 contiguous-CP step
lands with plan item D4):

    scripts/training/run_recipe.py \
        --recipe falcon_deepseekv4_mock_2b6_pretrain_2gpu_h100_bf16_config \
        --step_func vlm_step \
        dataset.path=<energon dataset or MetadatasetV2 yaml>

The mock is from-scratch only: pass ``checkpoint.pretrained_checkpoint=null``.
Text is tokenized with the real DeepSeek-V4 tokenizer
(``_DSV4_TOKENIZER_PATH``); override with
``dataset.task_encoder.ds_tokenizer_path=<path>``.
"""

import torch

from megatron.bridge.data.builders.energon import EnergonDatasetConfig
from megatron.bridge.models.falcon_mm import FalconDeepseekv4ModelProvider
from megatron.bridge.models.falcon_mm.data import FalconDeepseekv4EnergonTaskEncoderConfig
from megatron.bridge.recipes.common import _pretrain_common
from megatron.bridge.recipes.utils.environment_utils import COMMON_RECIPE_ENV_VARS
from megatron.bridge.recipes.utils.optimizer_utils import distributed_fused_adam_with_cosine_annealing
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import ConfigContainer


_DSV4_TOKENIZER_PATH = "FALCON_MM_AGENT_NOTES/assets/DeepSeek-V4-Flash"

# 12-layer mock: window, window, then alternating CSA(4) / HCA(128).
_MOCK_COMPRESS_RATIOS = [0, 0, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128]

__all__ = [
    "falcon_deepseekv4_mock_2b6_pretrain_2gpu_h100_bf16_config",
]


def _make_falcon_dsv4_energon_dataset(seq_length: int, micro_batch_size: int) -> EnergonDatasetConfig:
    """Create the Energon dataset config for Falcon DSv4 (THD Energon-native packing only).

    ``path`` is intentionally None — point it at an indexed Energon dataset
    root or a MetadatasetV2 YAML mixture via CLI: ``dataset.path=<path>``.
    """
    return EnergonDatasetConfig(
        path=None,
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        num_workers=2,
        packing_buffer_size=16,
        task_encoder=FalconDeepseekv4EnergonTaskEncoderConfig(ds_tokenizer_path=_DSV4_TOKENIZER_PATH),
    )


def _falcon_deepseekv4_mock_2b6_provider() -> FalconDeepseekv4ModelProvider:
    """Build the ~2.6B mock DSv4 provider (D1-spike-verified sizing).

    Keeps the DSv4 Flash per-head attention geometry (v_head_dim=512,
    qk_rope=64) and shrinks the counts: hidden 2048, 12 layers, 32 experts.
    """
    return FalconDeepseekv4ModelProvider(
        num_layers=len(_MOCK_COMPRESS_RATIOS),
        hidden_size=2048,
        num_attention_heads=16,
        vocab_size=129280,
        seq_length=1024,
        # ---- MLA geometry ----
        v_head_dim=512,
        qk_pos_emb_head_dim=64,
        q_lora_rank=512,
        o_groups=4,
        o_lora_rank=512,
        # ---- CSA / indexer ----
        csa_compress_ratios=_MOCK_COMPRESS_RATIOS,
        csa_window_size=128,
        csa_compress_rotary_base=160000.0,
        dsa_indexer_n_heads=16,
        dsa_indexer_head_dim=128,
        dsa_indexer_topk=64,
        dsa_kernel_backend="none",
        dsa_indexer_loss_coeff=0.01,
        dsa_indexer_use_sparse_loss=True,
        # ---- RoPE (YaRN) ----
        rotary_base=10000,
        rope_type="yarn",
        rotary_scaling_factor=16.0,
        original_max_position_embeddings=65536,
        beta_fast=32.0,
        beta_slow=1.0,
        mscale=1.0,
        mscale_all_dim=1.0,
        # ---- Hyper-connections ----
        enable_hyper_connections=True,
        use_fused_mhc=False,
        num_residual_streams=4,
        mhc_sinkhorn_iterations=20,
        # ---- MoE ----
        num_moe_experts=32,
        moe_router_topk=4,
        moe_ffn_hidden_size=768,
        moe_shared_expert_intermediate_size=768,
        moe_layer_freq=[1] * len(_MOCK_COMPRESS_RATIOS),
        moe_n_hash_layers=2,
        actual_vocab_size=129280,
        moe_router_score_function="sqrtsoftplus",
        moe_router_enable_expert_bias=True,
        moe_router_dtype="fp32",
        moe_router_topk_scaling_factor=1.5,
        moe_router_load_balancing_type="noaux_tc",
        moe_aux_loss_coeff=0.0,
        moe_token_dispatcher_type="alltoall",
        moe_permute_fusion=True,
        moe_shared_expert_overlap=False,
        activation_func_clamp_value=10.0,
        # ---- Parallelism / runtime ----
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_tensor_parallel_size=1,
        sequence_parallel=False,
        mtp_num_layers=0,
        share_embeddings_and_output_weights=False,
        make_vocab_size_divisible_by=1280,
        layernorm_epsilon=1e-6,
        gradient_accumulation_fusion=False,
        cross_entropy_loss_fusion=False,
        calculate_per_token_loss=True,
        # "auto" satisfies the THD packing backend guard; the DSv4 attention
        # modules select their own kernels via dsa_kernel_backend.
        attention_backend="auto",
    )


def _apply_falcon_dsv4_energon_runtime(cfg: ConfigContainer, *, max_lr: float, min_lr: float) -> None:
    """Apply shared Energon-packing pretrain runtime knobs for Falcon DSv4."""
    cfg.checkpoint.save_interval = 100

    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.cuda_graph_impl = "none"
    cfg.model.virtual_pipeline_model_parallel_size = None

    # Memory saving (off for the mock)
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None

    # Training config
    cfg.train.train_iters = 1000
    cfg.train.global_batch_size = 8
    cfg.train.micro_batch_size = 1  # required by Energon native packing
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100
    cfg.train.manual_gc_eval = 100
    cfg.validation.eval_interval = 500
    cfg.validation.eval_iters = 8

    # Optimizer
    opt_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=20,
        lr_decay_iters=1000,
        max_lr=max_lr,
        min_lr=min_lr,
    )
    cfg.optimizer = opt_cfg
    cfg.scheduler = scheduler_cfg
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # Tokenizer for the training loop (checkpoint metadata, eval splits).
    # The dataset plane tokenizes inside the Energon workers with the same
    # DSv4 tokenizer (see the task encoder config).
    cfg.tokenizer.tokenizer_model = _DSV4_TOKENIZER_PATH

    # Dataset config: THD Energon-native packing only
    cfg.dataset = _make_falcon_dsv4_energon_dataset(cfg.model.seq_length, cfg.train.micro_batch_size)

    # DDP config
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.average_in_collective = False  # required by Energon native packing
    cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"

    # Keep MoE EP overlap off until the packed schedule-plan threading lands (C4).
    cfg.comm_overlap = CommOverlapConfig(
        tp_comm_overlap=False,
        overlap_grad_reduce=False,
        overlap_param_gather=False,
        overlap_param_gather_with_optimizer_step=False,
        overlap_moe_expert_parallel_comm=False,
        delay_wgrad_compute=False,
    )

    cfg.mixed_precision = "bf16_mixed"
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 1,
    }


def falcon_deepseekv4_mock_2b6_pretrain_2gpu_h100_bf16_config() -> ConfigContainer:
    """Return a from-scratch pretrain smoke config for Falcon DSv4 (~2.6B mock).

    Default configuration: 2 GPU
    - TP=1, PP=1, CP=1, EP=2 (D1-verified topology)
    - Energon-native THD packing (packing_buffer_size=16), MBS=1
    - Real DeepSeek-V4 tokenizer; image placeholder = reserved token 128000
    - LR=3e-4, cosine to 3e-5, 1000 iters
    - Sequence length: 1024

    This is the current working config from the D1 spike. From-scratch only:
    pass ``checkpoint.pretrained_checkpoint=null`` to the launcher.
    """
    cfg = _pretrain_common()
    cfg.model = _falcon_deepseekv4_mock_2b6_provider()
    cfg.model.expert_model_parallel_size = 2
    _apply_falcon_dsv4_energon_runtime(cfg, max_lr=3e-4, min_lr=3e-5)
    return cfg
