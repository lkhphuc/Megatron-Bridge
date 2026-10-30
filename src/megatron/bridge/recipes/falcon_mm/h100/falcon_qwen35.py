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

"""Falcon-MM (encoder-free) recipes for H100, Energon-native THD packing only."""

from dataclasses import fields

import torch

from megatron.bridge import AutoBridge
from megatron.bridge.models.falcon_mm import (
    FalconQwen35ModelProvider,
    FalconQwen35MoEModelProvider,
    falcon_qwen35_provider_from_qwen35_vl,
)
from megatron.bridge.models.falcon_mm.data import FalconEnergonTaskEncoderConfig
from megatron.bridge.recipes.common import _sft_common_vlm
from megatron.bridge.recipes.falcon_mm.dataset import (
    FALCON_ENERGON_MICRO_BATCH_SIZE,
    make_falcon_energon_dataset,
)
from megatron.bridge.recipes.utils.environment_utils import COMMON_RECIPE_ENV_VARS
from megatron.bridge.recipes.utils.optimizer_utils import distributed_fused_adam_with_cosine_annealing
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.mixed_precision import bf16_mixed


_HF_2B = "Qwen/Qwen3.5-2B"
_HF_35B_A3B = "Qwen/Qwen3.5-35B-A3B"

# Pretrain LR schedule shared with Qwen3.5-VL 35B-A3B library recipes.
_FALCON_QWEN35_PRETRAIN_MAX_LR = 3e-4
_FALCON_QWEN35_PRETRAIN_MIN_LR = 3e-5

__all__ = [
    "falcon_qwen35_2b_sft_1gpu_h100_bf16_config",
    "falcon_qwen35_2b_sft_2gpu_h100_bf16_config",
    "falcon_qwen35_2b_pretrain_4gpu_h100_bf16_config",
    "falcon_qwen35_2b_pretrain_8gpu_h100_bf16_config",
    "falcon_qwen35_35b_a3b_sft_8gpu_h100_bf16_config",
    "falcon_qwen35_35b_a3b_sft_16gpu_h100_bf16_config",
    "falcon_qwen35_35b_a3b_pretrain_8gpu_h100_bf16_config",
    "falcon_qwen35_35b_a3b_pretrain_16gpu_h100_bf16_config",
]


def _make_falcon_energon_dataset(hf_path: str, seq_length: int):
    """Qwen-trunk Falcon Energon dataset (shared packing defaults; MBS fixed at 1)."""
    return make_falcon_energon_dataset(
        FalconEnergonTaskEncoderConfig(hf_processor_path=hf_path),
        seq_length=seq_length,
    )


def _apply_falcon_energon_sft_runtime(
    cfg: ConfigContainer,
    *,
    hf_path: str,
    max_lr: float,
    min_lr: float,
    global_batch_size: int,
) -> None:
    """Apply shared Energon-packing SFT runtime knobs (dense or MoE)."""
    cfg.model.seq_length = 4096
    cfg.checkpoint.save_interval = 100
    # Overlap checkpoint I/O with training (torch_dist + persistent worker).
    cfg.checkpoint.async_save = True

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False

    # VLM-specific settings
    cfg.model.freeze_language_model = False
    cfg.model.freeze_vision_model = False
    cfg.model.freeze_vision_projection = False

    # MTP (Energon native packing rejects MTP)
    cfg.model.mtp_num_layers = 0
    cfg.model.mtp_loss_scaling_factor = 0.1

    # TE and kernels (Energon native packing rejects CUDA graphs)
    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3
    cfg.model.attention_backend = "auto"
    cfg.model.gradient_accumulation_fusion = True
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"

    # Memory saving
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # Required by Energon native packing
    cfg.model.calculate_per_token_loss = True

    # Training config
    cfg.train.train_iters = 300000
    cfg.train.global_batch_size = global_batch_size
    cfg.train.micro_batch_size = FALCON_ENERGON_MICRO_BATCH_SIZE
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100
    cfg.train.manual_gc_eval = 100
    cfg.validation.eval_interval = 500
    cfg.validation.eval_iters = 32
    cfg.validation.eval_micro_batch_size = FALCON_ENERGON_MICRO_BATCH_SIZE

    cfg.logger.log_throughput_to_tensorboard = True
    cfg.logger.throughput_window_size = 10
    cfg.logger.console_color = "auto"

    # Optimizer
    opt_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=200,
        lr_decay_iters=300000,
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

    # Dataset config: THD Energon-native packing only
    cfg.dataset = _make_falcon_energon_dataset(hf_path, cfg.model.seq_length)

    # DDP config
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.average_in_collective = False  # required by Energon native packing
    cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"

    cfg.mixed_precision = "bf16_mixed"
    # Keep the complete process environment visible on the recipe.
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
    }


def _enable_gdn_conv_fusion(model) -> None:
    """Enable GDN pre-gated-delta-rule fusion when Megatron-Core exposes it.

    Same guard as the Qwen3.5-VL recipes: assigning an unknown field would
    silently create a phantom attribute on older Core builds.
    """
    if hasattr(type(model), "gdn_pre_gated_delta_rule_fusion") or hasattr(model, "gdn_pre_gated_delta_rule_fusion"):
        model.gdn_pre_gated_delta_rule_fusion = True


def _apply_falcon_moe_runtime(cfg: ConfigContainer) -> None:
    """MoE dispatcher / EP defaults shared by Falcon-MM MoE recipes."""
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.bias_activation_fusion = True
    cfg.model.cross_entropy_fusion_impl = "te"

    cfg.model.moe_token_dispatcher_type = "alltoall"
    cfg.model.moe_flex_dispatcher_backend = None
    cfg.model.moe_router_fusion = True
    cfg.model.moe_permute_fusion = True
    cfg.model.moe_grouped_gemm = True
    cfg.model.moe_shared_expert_overlap = False
    cfg.model.moe_router_force_load_balancing = False
    cfg.model.moe_router_padding_for_fp8 = False

    cfg.optimizer.overlap_param_gather_with_optimizer_step = False

    # Keep MoE EP overlap off until Falcon-MM packing + PP path is validated.
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
        "CUDA_DEVICE_MAX_CONNECTIONS": 1,
    }


def _apply_falcon_qwen35_moe_hybridep(cfg: ConfigContainer, *, expert_parallel_size: int) -> None:
    """Map Qwen3.5-VL 35B-A3B HybridEP knobs onto Falcon-MM Energon packing.

    Upstream references:
    - library ``qwen35_vl_35b_a3b_pretrain_config`` / H100 execution policy
    - B200 perf ``qwen35_vl_35b_a3b_pretrain_8gpu_b200_bf16_config``

    CUDA graphs stay off (Energon-native packing). EP A2A overlap stays off
    (VLM schedule-plan gap, same as Falcon DSv4). Fixed pack width helps
    HybridEP under variable-shape vision tokens.

    ``moe_permute_fusion_into_hybridep`` stays False (Falcon DSv4 Energon
    path). Upstream Qwen3.5-VL H100 sets it True, but fused permute+dispatch
    fails HybridEP ``config.is_valid`` on Falcon THD packs
    (``AssertionError: The config is not valid.``).
    """
    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_hybridep_num_sms = None
    cfg.model.moe_flex_dispatcher_num_sms = 16
    cfg.model.moe_hybridep_num_sms_preprocessing = 16
    cfg.model.moe_permute_fusion = True
    # Keep unfused permute ↔ HybridEP (see docstring); matches Falcon DSv4.
    cfg.model.moe_permute_fusion_into_hybridep = False
    cfg.model.moe_shared_expert_overlap = False
    cfg.model.moe_router_force_load_balancing = False
    cfg.model.apply_rope_fusion = False
    cfg.model.batch_p2p_sync = False
    if hasattr(cfg.model, "high_priority_a2a_comm_stream"):
        cfg.model.high_priority_a2a_comm_stream = True
    if hasattr(cfg.model, "overlap_dispatch_backward_with_experts_wgrad"):
        # Safe without CUDA graphs / EP overlap; mirrors the H100 execution policy.
        cfg.model.overlap_dispatch_backward_with_experts_wgrad = True

    cfg.dataset.pad_to_max_length = True

    cfg.env_vars = {
        **cfg.env_vars,
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": expert_parallel_size,
        "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 64,
        "NUM_OF_TOKENS_PER_CHUNK_DISPATCH_API": 64,
        "NUM_OF_TOKENS_PER_CHUNK_PREPROCESSING_API": 64,
        "NVLINK_DOMAIN_SIZE": 8,
        "USE_MNNVL": 0,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 0,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 0,
        "NVTE_NORM_BWD_USE_CUDNN": 1,
        "NVTE_NORM_FWD_USE_CUDNN": 1,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }


def _falcon_qwen35_2b_sft_base() -> ConfigContainer:
    """Shared base for Falcon-MM 2B dense (Qwen3.5-2B trunk) SFT recipes."""
    cfg = _sft_common_vlm()
    cfg.model = falcon_qwen35_provider_from_qwen35_vl(
        AutoBridge.from_hf_pretrained(_HF_2B).to_megatron_provider(load_weights=False)
    )
    _apply_falcon_energon_sft_runtime(
        cfg,
        hf_path=_HF_2B,
        max_lr=5e-6,
        min_lr=5e-7,
        global_batch_size=8,
    )
    return cfg


def falcon_qwen35_2b_sft_1gpu_h100_bf16_config() -> ConfigContainer:
    """Return an SFT dev config for Falcon-MM 2B (encoder-free, Qwen3.5-2B trunk).

    Default configuration: 1 GPU
    - TP=1, PP=1, CP=1
    - Energon-native THD packing (packing_buffer_size=16), MBS=1
    - LR=5e-6 (full SFT)
    - Sequence length: 4096

    This is the PP=1 baseline for the PP loss-parity sentinel
    (energon-packing-pp-enablement.md, step 2).
    """
    return _falcon_qwen35_2b_sft_base()


def falcon_qwen35_2b_sft_2gpu_h100_bf16_config() -> ConfigContainer:
    """Return an SFT dev config for Falcon-MM 2B (encoder-free, Qwen3.5-2B trunk).

    Default configuration: 2 GPU
    - TP=1, PP=2, CP=1
    - Energon-native THD packing (packing_buffer_size=16), MBS=1
    - LR=5e-6 (full SFT)
    - Sequence length: 4096

    With PP>1, packs become fixed-width (pad_to_max_length auto-flip) and VPP
    must stay unset (Energon loader-state resume gap).
    """
    cfg = _falcon_qwen35_2b_sft_base()
    cfg.model.pipeline_model_parallel_size = 2
    cfg.model.moe_token_dispatcher_type = "alltoall"
    return cfg


def _falcon_qwen35_2b_moe_pretrain_base() -> ConfigContainer:
    """2B toy MoE for single-node MoE+PP debug (from-scratch only).

    Starts from the dense 2B recipe, then swaps in a Falcon MoE provider that
    reuses the dense trunk dims with a small expert set (8 experts / top-2,
    expert FFN = dense ``ffn_hidden_size // 4``). Override the launcher's
    pretrained path: ``checkpoint.pretrained_checkpoint=null``.
    """
    cfg = _falcon_qwen35_2b_sft_base()
    dense = cfg.model
    assert isinstance(dense, FalconQwen35ModelProvider)
    expert_ffn = max(dense.ffn_hidden_size // 4, 128)
    init_kwargs = {
        field.name: getattr(dense, field.name)
        for field in fields(FalconQwen35MoEModelProvider)
        if field.init and hasattr(dense, field.name)
    }
    init_kwargs.update(
        {
            "num_moe_experts": 8,
            "moe_router_topk": 2,
            "moe_ffn_hidden_size": expert_ffn,
            "moe_shared_expert_intermediate_size": expert_ffn,
            "moe_shared_expert_gate": True,
            "moe_grouped_gemm": True,
            "moe_token_dispatcher_type": "alltoall",
            "moe_permute_fusion": True,
            "moe_router_load_balancing_type": "global_aux_loss",
            "moe_router_pre_softmax": False,
            "moe_aux_loss_coeff": 1e-3,
        }
    )
    cfg.model = FalconQwen35MoEModelProvider(**init_kwargs)
    _apply_falcon_moe_runtime(cfg)
    cfg.model.pipeline_model_parallel_size = 2
    cfg.model.expert_model_parallel_size = 2
    cfg.checkpoint.pretrained_checkpoint = None
    return cfg


def falcon_qwen35_2b_pretrain_4gpu_h100_bf16_config() -> ConfigContainer:
    """Return a from-scratch MoE pretrain config for Falcon-MM 2B (toy MoE).

    Default configuration: 4 GPUs
    - TP=1, PP=2, EP=2, CP=1
    - 8 experts / top-2 (built from dense Qwen3.5-2B dims)
    - Energon-native THD packing, MBS=1
    - No pretrained checkpoint (random init)

    Launch example::

        RECIPE=falcon_qwen35_2b_moe_pretrain_4gpu_h100_bf16_config NUM_GPUS=4 PP=2 \\
          bash examples/models/falcon_mm/sft_energon.sh \\
          checkpoint.pretrained_checkpoint=null
    """
    return _falcon_qwen35_2b_moe_pretrain_base()


def falcon_qwen35_2b_pretrain_8gpu_h100_bf16_config() -> ConfigContainer:
    """Return a from-scratch MoE pretrain config for Falcon-MM 2B on 8 GPUs.

    Default configuration: 8 GPUs
    - TP=1, PP=2, EP=4, CP=1
    - Same toy MoE as the 4-GPU recipe
    """
    cfg = _falcon_qwen35_2b_moe_pretrain_base()
    cfg.model.expert_model_parallel_size = 4
    cfg.train.global_batch_size = 16
    return cfg


def _falcon_qwen35_35b_a3b_sft_base() -> ConfigContainer:
    """Falcon-MM 35B-A3B MoE SFT: shared runtime + HF MoE architecture.

    Reuses the same Energon/MoE runtime helpers as the 2B MoE recipes. Model
    dims come from ``Qwen/Qwen3.5-35B-A3B`` (cannot derive from the 2B dense
    trunk — different hidden/expert geometry).
    """
    cfg = _sft_common_vlm()
    cfg.model = falcon_qwen35_provider_from_qwen35_vl(
        AutoBridge.from_hf_pretrained(_HF_35B_A3B).to_megatron_provider(load_weights=False)
    )
    _apply_falcon_energon_sft_runtime(
        cfg,
        hf_path=_HF_35B_A3B,
        max_lr=2e-5,
        min_lr=2e-6,
        global_batch_size=16,
    )
    _apply_falcon_moe_runtime(cfg)

    # Memory saving for 35B MoE full SFT on H100
    cfg.model.recompute_granularity = "full"
    cfg.model.recompute_method = "uniform"
    cfg.model.recompute_num_layers = 1

    cfg.model.pipeline_model_parallel_size = 2
    cfg.model.expert_model_parallel_size = 4
    return cfg


def falcon_qwen35_35b_a3b_sft_8gpu_h100_bf16_config() -> ConfigContainer:
    """Return an SFT config for Falcon-MM 35B-A3B MoE (Qwen3.5-35B-A3B trunk).

    Default configuration: 8 GPUs
    - TP=1, PP=2, EP=4, CP=1  (MoE + pipeline combination)
    - Energon-native THD packing (packing_buffer_size=16), MBS=1
    - alltoall MoE dispatcher (correctness-oriented)
    - Sequence length: 4096
    """
    return _falcon_qwen35_35b_a3b_sft_base()


def falcon_qwen35_35b_a3b_sft_16gpu_h100_bf16_config() -> ConfigContainer:
    """Return an SFT config for Falcon-MM 35B-A3B MoE (Qwen3.5-35B-A3B trunk).

    Default configuration: 16 GPUs
    - TP=1, PP=2, EP=8, CP=1  (MoE + pipeline combination)
    - Energon-native THD packing (packing_buffer_size=16), MBS=1
    - alltoall MoE dispatcher (correctness-oriented)
    - Sequence length: 4096
    """
    cfg = _falcon_qwen35_35b_a3b_sft_base()
    cfg.model.expert_model_parallel_size = 8
    cfg.train.global_batch_size = 32
    return cfg


def _falcon_qwen35_35b_a3b_pretrain_base() -> ConfigContainer:
    """From-scratch Falcon-MM 35B-A3B MoE pretrain with Qwen3.5-VL MoE knobs.

    Starts from the Energon SFT runtime (THD packing, MBS=1, MTP off, CUDA
    graphs off), then remaps convergence / execution settings from the
    upstream Qwen3.5-VL 35B-A3B pretrain path:

    - LR 3e-4 / 3e-5, warmup 500
    - HybridEP flex dispatcher (H100/B200 perf + library pretrain)
    - Precision-aware Adam moments (bf16)
    - Selective recompute ``core_attn`` / ``gdn_norm_out`` / ``moe_act``
    - GDN pre-gated-delta-rule fusion when Core exposes it
    - No pretrained checkpoint (random init + Falcon patch embed)
    """
    cfg = _falcon_qwen35_35b_a3b_sft_base()
    _enable_gdn_conv_fusion(cfg.model)

    # Pretrain schedule (matches qwen35_vl_35b_a3b_pretrain_*).
    opt_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=500,
        lr_decay_iters=300000,
        max_lr=_FALCON_QWEN35_PRETRAIN_MAX_LR,
        min_lr=_FALCON_QWEN35_PRETRAIN_MIN_LR,
    )
    cfg.optimizer = opt_cfg
    cfg.scheduler = scheduler_cfg
    cfg.optimizer.use_precision_aware_optimizer = True
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.bfloat16
    cfg.optimizer.exp_avg_sq_dtype = torch.bfloat16
    cfg.optimizer.overlap_param_gather_with_optimizer_step = False

    cfg.mixed_precision = bf16_mixed()
    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False

    # Preserve architecture vocabulary (multimodal tokens) for from-scratch runs.
    cfg.tokenizer.use_tokenizer_vocab_size = False
    cfg.tokenizer.tokenizer_model = _HF_35B_A3B

    # Activation memory: library pretrain selective set (not full-layer SFT).
    cfg.model.recompute_granularity = "selective"
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.recompute_modules = ["core_attn", "gdn_norm_out", "moe_act"]

    cfg.model.freeze_language_model = False
    cfg.model.freeze_vision_model = False
    cfg.model.freeze_vision_projection = False
    cfg.checkpoint.pretrained_checkpoint = None

    _apply_falcon_qwen35_moe_hybridep(cfg, expert_parallel_size=cfg.model.expert_model_parallel_size)
    return cfg


def falcon_qwen35_35b_a3b_pretrain_8gpu_h100_bf16_config() -> ConfigContainer:
    """Return a from-scratch pretrain config for Falcon-MM 35B-A3B on 8 H100s.

    Default configuration: 8 GPUs
    - TP=1, PP=2, EP=4, CP=1
    - HybridEP (from Qwen3.5-VL MoE pretrain/perf), EP overlap off
    - Energon-native THD packing, MBS=1, GBS=16
    - Selective recompute; LR=3e-4
    - Sequence length: 4096

    Launch with ``checkpoint.pretrained_checkpoint=null`` (already the default)
    and ``--mode pretrain --step_func vlm_step``.
    """
    return _falcon_qwen35_35b_a3b_pretrain_base()


def falcon_qwen35_35b_a3b_pretrain_16gpu_h100_bf16_config() -> ConfigContainer:
    """Return a from-scratch pretrain config for Falcon-MM 35B-A3B on 16 H100s.

    Execution resize toward ``qwen35_vl_35b_a3b_pretrain_config`` /
    ``qwen35_vl_35b_a3b_pretrain_16gpu_h100_bf16_config``:

    Default configuration: 16 GPUs
    - TP=1, PP=2, EP=8, CP=1
    - HybridEP, selective recompute, GBS=32 (raise toward 512 for throughput)
    - CUDA graphs stay off (Energon packing)
    - Sequence length: 4096
    """
    cfg = _falcon_qwen35_35b_a3b_pretrain_base()
    cfg.model.expert_model_parallel_size = 8
    cfg.train.global_batch_size = 32
    _apply_falcon_qwen35_moe_hybridep(cfg, expert_parallel_size=8)
    return cfg
