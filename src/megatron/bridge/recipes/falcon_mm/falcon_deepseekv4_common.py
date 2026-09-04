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

"""Shared Falcon-MM DeepSeek-V4 providers and Energon runtime helpers."""

from __future__ import annotations

from typing import Any

import torch

from megatron.bridge.models.deepseek.deepseek_v4_bridge import (
    deepseek_v4_supports_blackwell_fused_kernels,
)
from megatron.bridge.models.falcon_mm import FalconDeepseekv4ModelProvider
from megatron.bridge.models.falcon_mm.data import FalconDeepseekv4EnergonTaskEncoderConfig
from megatron.bridge.recipes.deepseek.h100.deepseek_v4 import _deepseek_v4_mxfp8_quant_recipe
from megatron.bridge.recipes.falcon_mm.dataset import (
    FALCON_ENERGON_MICRO_BATCH_SIZE,
    make_falcon_energon_dataset,
)
from megatron.bridge.recipes.utils.environment_utils import COMMON_RECIPE_ENV_VARS
from megatron.bridge.recipes.utils.optimizer_utils import (
    distributed_fused_adam_with_cosine_annealing,
)
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.mixed_precision import bf16_with_mxfp8_mixed
from megatron.bridge.utils.cuda_graph import set_cuda_graph_modules


DSV4_TOKENIZER_PATH = "deepseek-ai/DeepSeek-V4-Flash-Base"

# Canonical from-scratch architecture declarations. Every logical decoder
# layer is an attention component (W=window, C=CSA, H=HCA) followed by its FFN
# component (-=dense MLP, E=MoE). Strings are split into eight-layer chunks so
# the complete ordering remains easy to count and visually review.
# Last decoder attention is always C (CSA), matching DeepSeek-V4 Flash/Pro
# (HF ``compress_ratios`` decoder tail); trailing window-0 is MTP-only upstream.

# 25 decoder layers: W=2, C=8, H=6; dense MLP after every attention component.
FALCON_DSV4_DENSE_1B_PATTERN = "".join(
    (
        "W-W-C-H-C-H-C-H-",  # Decoder layers 1-8.
        "C-H-C-H-C-H-C-H-",  # Decoder layers 9-16.
        "C-H-C-H-C-H-C-H-C-",  # Decoder layers 17-25 (last attention = C).
    )
)

# 33 decoder layers: W=2, C=12, H=10; dense MLP after every attention component.
FALCON_DSV4_DENSE_3B_PATTERN = "".join(
    (
        "W-W-C-H-C-H-C-H-",  # Decoder layers 1-8.
        "C-H-C-H-C-H-C-H-",  # Decoder layers 9-16.
        "C-H-C-H-C-H-C-H-",  # Decoder layers 17-24.
        "C-H-C-H-C-H-C-H-C-",  # Decoder layers 25-33 (last attention = C).
    )
)

# 32 decoder layers: W=2, C=16, H=14; dense MLP after every attention component.
FALCON_DSV4_DENSE_10B_PATTERN = "".join(
    (
        "W-W-C-H-C-H-C-H-",  # Decoder layers 1-8.
        "C-H-C-H-C-H-C-H-",  # Decoder layers 9-16.
        "C-H-C-H-C-H-C-H-",  # Decoder layers 17-24.
        "C-H-C-H-C-H-C-H-C-",  # Decoder layers 25-33 (last attention = C).
    )
)

# 41 decoder layers: W=2, C=20, H=18; MoE after every attention component.
FALCON_DSV4_MOE_35B_A3B_PATTERN = "".join(
    (
        "WEWECEHECEHECEHE",  # Decoder layers 1-8.
        "CEHECEHECEHECEHE",  # Decoder layers 9-16.
        "CEHECEHECEHECEHE",  # Decoder layers 17-24.
        "CEHECEHECEHECEHE",  # Decoder layers 25-32.
        "CEHECEHECEHECEHECE",  # Decoder layers 33-41 (last attention = C).
    )
)

# Shared B200 process env for Falcon DSv4 (dense/MoE scaling recipes).
# H100 recipes intentionally keep CUDA_DEVICE_MAX_CONNECTIONS=1.
FALCON_DSV4_B200_ENV_VARS: dict[str, Any] = {
    "CUDA_DEVICE_MAX_CONNECTIONS": 32,
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "NVLINK_DOMAIN_SIZE": 8,
    "USE_MNNVL": 0,
    "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
}


def _falcon_deepseekv4_scale_provider(
    *,
    hybrid_layer_pattern: str,
    hidden_size: int,
    q_lora_rank: int,
    ffn_hidden_size: int | None = None,
    num_moe_experts: int | None = None,
    moe_router_topk: int = 6,
    moe_ffn_hidden_size: int | None = None,
    moe_shared_expert_intermediate_size: int | None = None,
) -> FalconDeepseekv4ModelProvider:
    """Build a member of the Falcon DSv4 dense/MoE scaling family."""
    configured_uses_moe = num_moe_experts is not None

    provider_kwargs: dict[str, Any] = {
        "hybrid_layer_pattern": hybrid_layer_pattern,
        "hidden_size": hidden_size,
        "num_attention_heads": 64,
        "vocab_size": 129280,
        "seq_length": 8192,
        "v_head_dim": 512,
        "qk_pos_emb_head_dim": 64,
        "q_lora_rank": q_lora_rank,
        "o_groups": 4,
        "o_lora_rank": q_lora_rank,
        "csa_window_size": 128,
        "csa_compress_rotary_base": 160000.0,
        "dsa_indexer_n_heads": 64,
        "dsa_indexer_head_dim": 128,
        "dsa_indexer_topk": 128,
        "dsa_kernel_backend": "cudnn",
        "dsa_indexer_loss_coeff": 0.01,
        "dsa_indexer_use_sparse_loss": True,
        "rotary_base": 10000,
        "rope_type": "yarn",
        "rotary_scaling_factor": 16.0,
        "original_max_position_embeddings": 65536,
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "mscale": 1.0,
        "mscale_all_dim": 1.0,
        "enable_hyper_connections": True,
        "use_fused_mhc": False,
        "num_residual_streams": 4,
        "mhc_sinkhorn_iterations": 20,
        "actual_vocab_size": 129280,
        "activation_func_clamp_value": 10.0,
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "context_parallel_size": 1,
        "expert_tensor_parallel_size": 1,
        "sequence_parallel": False,
        "mtp_num_layers": 0,
        "share_embeddings_and_output_weights": False,
        "make_vocab_size_divisible_by": 1280,
        "layernorm_epsilon": 1e-6,
        "gradient_accumulation_fusion": False,
        "cross_entropy_loss_fusion": True,
        "cross_entropy_fusion_impl": "native",
        "calculate_per_token_loss": True,
        "attention_backend": "auto",
    }
    if configured_uses_moe:
        provider_kwargs.update(
            num_moe_experts=num_moe_experts,
            moe_router_topk=moe_router_topk,
            moe_ffn_hidden_size=moe_ffn_hidden_size,
            moe_shared_expert_intermediate_size=moe_shared_expert_intermediate_size,
            moe_n_hash_layers=0,
            moe_router_score_function="sqrtsoftplus",
            moe_router_enable_expert_bias=True,
            moe_router_dtype="fp32",
            moe_router_topk_scaling_factor=1.5,
            moe_router_load_balancing_type="noaux_tc",
            moe_aux_loss_coeff=0.0,
            moe_token_dispatcher_type="alltoall",
            moe_permute_fusion=True,
            moe_shared_expert_overlap=True,
            moe_vl_routing=True,
        )
    else:
        provider_kwargs.update(
            ffn_hidden_size=ffn_hidden_size,
            moe_n_hash_layers=0,
            moe_grouped_gemm=False,
            moe_permute_fusion=False,
            moe_vl_routing=False,
        )
    return FalconDeepseekv4ModelProvider(**provider_kwargs)


def falcon_deepseekv4_1b_provider() -> FalconDeepseekv4ModelProvider:
    """Build the 1.08B dense Falcon DSv4 scaling provider."""
    return _falcon_deepseekv4_scale_provider(
        hybrid_layer_pattern=FALCON_DSV4_DENSE_1B_PATTERN,
        hidden_size=1024,
        q_lora_rank=256,
        ffn_hidden_size=3584,
    )


def falcon_deepseekv4_3b_provider() -> FalconDeepseekv4ModelProvider:
    """Build the 3.1B dense Falcon DSv4 scaling provider."""
    return _falcon_deepseekv4_scale_provider(
        hybrid_layer_pattern=FALCON_DSV4_DENSE_3B_PATTERN,
        hidden_size=2048,
        q_lora_rank=512,
        ffn_hidden_size=5632,
    )


def falcon_deepseekv4_10b_provider() -> FalconDeepseekv4ModelProvider:
    """Build the 9.4B dense Falcon DSv4 scaling provider."""
    return _falcon_deepseekv4_scale_provider(
        hybrid_layer_pattern=FALCON_DSV4_DENSE_10B_PATTERN,
        hidden_size=4096,
        q_lora_rank=1024,
        ffn_hidden_size=12288,
    )


def falcon_deepseekv4_35b_a3b_provider() -> FalconDeepseekv4ModelProvider:
    """Build the approximately 35B-total/3B-active Falcon DSv4 MoE provider."""
    return _falcon_deepseekv4_scale_provider(
        hybrid_layer_pattern=FALCON_DSV4_MOE_35B_A3B_PATTERN,
        hidden_size=2048,
        q_lora_rank=512,
        num_moe_experts=256,
        moe_router_topk=8,
        moe_ffn_hidden_size=512,
        moe_shared_expert_intermediate_size=512,
    )


def make_falcon_dsv4_energon_dataset(seq_length: int):
    """DSv4 Falcon Energon dataset (shared packing defaults; MBS fixed at 1)."""
    dataset = make_falcon_energon_dataset(
        FalconDeepseekv4EnergonTaskEncoderConfig(ds_tokenizer_path=DSV4_TOKENIZER_PATH),
        seq_length=seq_length,
    )
    # The DSv4 fused packed kernels require stable physical width.
    dataset.pad_to_max_length = True
    return dataset


def apply_falcon_dsv4_energon_runtime(
    cfg: ConfigContainer,
    *,
    max_lr: float,
    min_lr: float,
    env_vars: dict[str, Any] | None = None,
    enable_blackwell_kernels: bool = False,
    save_interval: int = 5000,
    train_iters: int = 1000,
    lr_warmup_iters: int = 100,
    lr_decay_iters: int | None = None,
    manual_gc_interval: int = 100,
    eval_interval: int = 500,
    eval_iters: int = 8,
    enable_megatron_core_experimental: bool = False,
) -> None:
    """Apply shared Energon-packing pretrain runtime knobs for Falcon DSv4.

    Enables Flash-safe selective recompute (``mla_up_proj``, ``mhc``). Full-layer
    recompute stays off: re-running layers drops ``input_ids``, which breaks
    hash-routed MoE. ``core_attn`` recompute is unsupported in DSv4 hybrid
    attention.
    """
    cfg.checkpoint.save_interval = save_interval
    cfg.checkpoint.async_save = True

    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.cuda_graph_impl = "none"
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.apply_rope_fusion = True

    if enable_blackwell_kernels and deepseek_v4_supports_blackwell_fused_kernels():
        cfg.model.use_fused_mhc = True

    # Fused CSA/DSA — unfused path dominated the seq-8k memory peak.
    cfg.model.dsa_kernel_backend = "cudnn"

    # Flash-safe selective recompute (not core_attn; not full-layer).
    cfg.model.recompute_granularity = "selective"
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.recompute_modules = ["mla_up_proj", "mhc"]

    cfg.train.train_iters = train_iters
    cfg.train.global_batch_size = 8
    cfg.train.micro_batch_size = FALCON_ENERGON_MICRO_BATCH_SIZE
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = manual_gc_interval
    cfg.train.manual_gc_eval = manual_gc_interval
    cfg.validation.eval_interval = eval_interval
    cfg.validation.eval_iters = eval_iters
    cfg.validation.eval_micro_batch_size = FALCON_ENERGON_MICRO_BATCH_SIZE

    cfg.logger.log_throughput_to_tensorboard = True
    cfg.logger.throughput_window_size = 10
    cfg.logger.console_color = "auto"

    opt_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=lr_warmup_iters,
        lr_decay_iters=lr_decay_iters,
        max_lr=max_lr,
        min_lr=min_lr,
    )
    cfg.optimizer = opt_cfg
    cfg.scheduler = scheduler_cfg
    # Precision-aware Adam moments (Flash H100/MXFP8 hygiene) while training BF16.
    cfg.optimizer.use_precision_aware_optimizer = True
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.bfloat16
    cfg.optimizer.exp_avg_sq_dtype = torch.bfloat16

    cfg.tokenizer.tokenizer_model = DSV4_TOKENIZER_PATH
    cfg.dataset = make_falcon_dsv4_energon_dataset(cfg.model.seq_length)

    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.average_in_collective = False  # required by Energon native packing
    cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"

    # EP overlap stays off: Energon VLM schedule plans do not forward visual /
    # packed-sequence metadata (see ConfigContainer._disable_native_energon...).
    cfg.comm_overlap = CommOverlapConfig(
        tp_comm_overlap=False,
        overlap_grad_reduce=False,
        overlap_param_gather=False,
        overlap_param_gather_with_optimizer_step=False,
        overlap_moe_expert_parallel_comm=False,
        delay_wgrad_compute=False,
    )

    if enable_megatron_core_experimental:
        cfg.dist.enable_megatron_core_experimental = True

    cfg.mixed_precision = "bf16_mixed"
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
        # Varlen packs fragment the allocator; expandable segments avoid the
        # reserved-but-unallocated growth seen at longer sequences.
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        **(env_vars or {}),
    }


def apply_falcon_dsv4_blackwell_features(cfg: ConfigContainer) -> None:
    """Apply Falcon-safe Blackwell feature knobs for DSv4 B200 recipes.

    Always enables fused RoPE/mHC/DSA, indexer loss, and experimental Core.
    MoE-only knobs (GroupedGEMM, HybridEP, router fusion) apply when
    ``num_moe_experts`` is set. Leaves EP overlap and CUDA graphs untouched.
    """
    cfg.model.apply_rope_fusion = True
    if deepseek_v4_supports_blackwell_fused_kernels():
        cfg.model.use_fused_mhc = True
    cfg.model.dsa_kernel_backend = "cudnn"
    cfg.model.dsa_indexer_loss_coeff = 0.01
    cfg.model.dsa_indexer_use_sparse_loss = True
    cfg.dist.enable_megatron_core_experimental = True

    if cfg.model.num_moe_experts is None:
        return

    cfg.model.moe_grouped_gemm = True
    cfg.model.moe_permute_fusion = True
    if hasattr(cfg.model, "moe_router_fusion"):
        cfg.model.moe_router_fusion = True
    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_flex_dispatcher_num_sms = 16
    cfg.model.moe_shared_expert_overlap = False
    cfg.model.moe_permute_fusion_into_hybridep = False


def apply_falcon_dsv4_mxfp8(cfg: ConfigContainer) -> None:
    """Enable MXFP8 while preserving the Falcon DSv4 training objective."""
    cfg.model.moe_router_padding_for_fp8 = True
    cfg.model.mtp_eval_in_bf16 = True
    cfg.model.quant_recipe = _deepseek_v4_mxfp8_quant_recipe()

    cfg.ddp.use_distributed_optimizer = True
    # Energon native packing still requires average_in_collective=False.
    cfg.ddp.average_in_collective = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"

    cfg.mixed_precision = bf16_with_mxfp8_mixed()
    cfg.mixed_precision.fp8_param_gather = False
    cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = False


def apply_falcon_dsv4_te_cuda_graphs(cfg: ConfigContainer) -> None:
    """Enable TE-scoped CUDA graphs for Falcon DSv4 MoE.

    Requires fixed pack width because HybridEP auto uneven-padding is skipped
    when CUDA graphs are enabled. EP overlap stays off.
    """
    cfg.model.cuda_graph_impl = "transformer_engine"
    set_cuda_graph_modules(cfg.model, ["attn", "moe_router", "moe_preprocess"])
    cfg.model.cuda_graph_warmup_steps = 3
    cfg.model.use_te_rng_tracker = True
    cfg.rng.te_rng_tracker = True
    cfg.dataset.pad_to_max_length = True
    if cfg.comm_overlap is not None:
        cfg.comm_overlap.overlap_moe_expert_parallel_comm = False
        cfg.comm_overlap.delay_wgrad_compute = False


# B200 process env mirrored from ``qwen3_30b_a3b_pretrain_8gpu_b200_fp8mx_config``.
# EP overlap / delay-wgrad are intentionally omitted: ConfigContainer disables
# them for Energon-native packing (VLM schedule-plan gap).
FALCON_DSV4_B200_QWEN30B_MXFP8_ENV_VARS: dict[str, Any] = {
    **FALCON_DSV4_B200_ENV_VARS,
    "NCCL_GRAPH_REGISTER": 0,
    "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
    "NCCL_NVLS_ENABLE": 0,
    "NVTE_NORM_FWD_USE_CUDNN": 1,
    "NVTE_NORM_BWD_USE_CUDNN": 1,
    "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
    "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
    "NVTE_CUTEDSL_FUSED_GROUPED_MLP": 1,
}


def apply_falcon_dsv4_qwen30b_style_mxfp8_throughput(
    cfg: ConfigContainer,
    *,
    seq_length: int = 4096,
    global_batch_size: int = 512,
) -> None:
    """Map Qwen3-30B-A3B 8-GPU MXFP8 throughput knobs onto Falcon DSv4 + Energon.

    Upstream reference (performance-summary / B200 perf recipe)::

        TP1 PP1 CP1 EP8, seq=4096, GBS=512, MBS=4, MXFP8, HybridEP,
        TE-scoped CUDA graphs, no recompute, EP overlap + delay wgrad.

    Energon-native THD packing forces physical ``MBS=1`` and keeps EP overlap
    off. Token slots per step stay matched via ``GBS=512`` packs of width
    ``seq_length``. Fixed pack width (``pad_to_max_length``) is required for
    HybridEP under CUDA graphs.

    Natural MoE routing is kept (not force-balanced). For a pure throughput
    ceiling on the same mesh, override
    ``model.moe_router_force_load_balancing=true`` — that is a new experiment,
    not a drop-in for mixture training.
    """
    apply_falcon_dsv4_mxfp8(cfg)
    apply_falcon_dsv4_te_cuda_graphs(cfg)

    cfg.model.seq_length = seq_length
    if cfg.dataset is not None and hasattr(cfg.dataset, "seq_length"):
        cfg.dataset.seq_length = seq_length

    cfg.train.global_batch_size = global_batch_size
    cfg.train.micro_batch_size = FALCON_ENERGON_MICRO_BATCH_SIZE
    cfg.validation.eval_micro_batch_size = FALCON_ENERGON_MICRO_BATCH_SIZE

    # Match the B200 Qwen3-30B MXFP8 recipe: no activation recompute when fit.
    # If this OOMs under real vision packs, restore selective
    # ``["mla_up_proj", "mhc"]`` via CLI.
    cfg.model.recompute_granularity = None
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.recompute_modules = None

    cfg.model.moe_router_force_load_balancing = False
    cfg.model.moe_flex_dispatcher_num_sms = 64
    if hasattr(cfg.model, "moe_hybridep_num_sms"):
        cfg.model.moe_hybridep_num_sms = 64
    if hasattr(cfg.model, "moe_hybridep_num_sms_preprocessing"):
        cfg.model.moe_hybridep_num_sms_preprocessing = 32
    if hasattr(cfg.model, "high_priority_a2a_comm_stream"):
        cfg.model.high_priority_a2a_comm_stream = True
    if hasattr(cfg.model, "moe_mlp_glu_interleave_size"):
        cfg.model.moe_mlp_glu_interleave_size = 32
    if hasattr(cfg.model, "moe_pad_experts_for_cuda_graph_inference"):
        cfg.model.moe_pad_experts_for_cuda_graph_inference = True

    # EP overlap stays explicitly false — validate() would clear it anyway.
    cfg.comm_overlap = CommOverlapConfig(
        tp_comm_overlap=False,
        overlap_grad_reduce=False,
        overlap_param_gather=False,
        overlap_param_gather_with_optimizer_step=False,
        overlap_moe_expert_parallel_comm=False,
        delay_wgrad_compute=False,
    )

    cfg.env_vars = {
        **cfg.env_vars,
        **FALCON_DSV4_B200_QWEN30B_MXFP8_ENV_VARS,
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": cfg.model.expert_model_parallel_size,
    }
