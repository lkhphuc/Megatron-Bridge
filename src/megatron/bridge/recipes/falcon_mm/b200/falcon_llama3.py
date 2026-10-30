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

"""Falcon-MM Llama3 dense recipes for eight B200 GPUs.

From-scratch early-fusion pretrain with a dense Llama3-8B GQA trunk and plain
1D RoPE. Launch via ``examples/models/falcon_mm/thd_energon.sh`` with
``--step_func vlm_step`` (the shell sets that) and
``checkpoint.pretrained_checkpoint=null``.

Tokenizer path (gated Hub id needs cache or a local snapshot)::

    export LLAMA3_TOKENIZER_PATH=/path/to/Meta-Llama-3-8B-tokenizer

``thd_energon.sh`` auto-picks ``${WORKSPACE}/hf/Meta-Llama-3-8B-tokenizer``
when that directory exists.
"""

from __future__ import annotations

import os

import torch

from megatron.bridge.models.falcon_mm import FalconLlama3ModelProvider
from megatron.bridge.models.falcon_mm.data import FalconLlama3EnergonTaskEncoderConfig
from megatron.bridge.recipes.common import _pretrain_common
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
from megatron.bridge.training.mixed_precision import get_mixed_precision_config


# Hub id requires HF auth + network (or a warm cache). Prefer overriding via
# LLAMA3_TOKENIZER_PATH to a local tokenizer snapshot with the Llama3 reserved
# special tokens (see resolve_llama3_tokenizer_path).
_LLAMA3_TOKENIZER_HUB_ID = "meta-llama/Meta-Llama-3-8B"

# Low-precision recipes mirrored from library ``llama3_8b_pretrain_2gpu_h100_*``.
_LLAMA3_LOW_PRECISION_RECIPES = (
    "bf16_with_mxfp8_mixed",
    "bf16_with_fp8_current_scaling_mixed",
    "bf16_with_nvfp4_mixed",
)


def resolve_llama3_tokenizer_path() -> str:
    """Resolve the Llama3 tokenizer path for model + Energon data plane.

    Order: ``LLAMA3_TOKENIZER_PATH`` env, else the Hub id
    ``meta-llama/Meta-Llama-3-8B``.
    """
    return os.environ.get("LLAMA3_TOKENIZER_PATH", "").strip() or _LLAMA3_TOKENIZER_HUB_ID


LLAMA3_TOKENIZER_PATH = _LLAMA3_TOKENIZER_HUB_ID  # backward-compatible alias

_B200_ENV_VARS: dict[str, int | str] = {
    "CUDA_DEVICE_MAX_CONNECTIONS": 32,
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "NCCL_GRAPH_REGISTER": 0,
    "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
    "NCCL_NVLS_ENABLE": 0,
    "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
    "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
    "NVTE_NORM_BWD_USE_CUDNN": 1,
    "NVTE_NORM_FWD_USE_CUDNN": 1,
}

_FALCON_MAX_LR = 3e-4
_FALCON_MIN_LR = 3e-5

__all__ = [
    "falcon_llama3_8b_pretrain_8gpu_b200_bf16_config",
    "falcon_llama3_8b_pretrain_8gpu_b200_fp8cs_config",
    "falcon_llama3_8b_pretrain_8gpu_b200_fp8mx_config",
    "falcon_llama3_8b_pretrain_8gpu_b200_nvfp4_config",
    "resolve_llama3_tokenizer_path",
]


def falcon_llama3_8b_provider() -> FalconLlama3ModelProvider:
    """Build the from-scratch Falcon Llama3-8B dense GQA provider."""
    return FalconLlama3ModelProvider(
        num_layers=32,
        hidden_size=4096,
        ffn_hidden_size=14336,
        num_attention_heads=32,
        num_query_groups=8,
        vocab_size=128256,
        seq_length=4096,
        rotary_base=500000,
        position_embedding_type="rope",
        normalization="RMSNorm",
        gated_linear_unit=True,
        add_bias_linear=False,
        share_embeddings_and_output_weights=False,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        sequence_parallel=False,
        mtp_num_layers=0,
        calculate_per_token_loss=True,
        cross_entropy_loss_fusion=True,
        cross_entropy_fusion_impl="native",
        attention_backend="auto",
        gradient_accumulation_fusion=True,
        apply_rope_fusion=True,
    )


def _make_falcon_llama3_energon_dataset(seq_length: int):
    """Llama3 Falcon Energon dataset (shared packing defaults; MBS fixed at 1)."""
    tokenizer_path = resolve_llama3_tokenizer_path()
    return make_falcon_energon_dataset(
        FalconLlama3EnergonTaskEncoderConfig(llama_tokenizer_path=tokenizer_path),
        seq_length=seq_length,
    )


def _apply_falcon_llama3_energon_runtime(cfg: ConfigContainer) -> None:
    """Apply Energon-packing pretrain runtime knobs for Falcon Llama3."""
    cfg.checkpoint.save_interval = 5000
    cfg.checkpoint.async_save = True
    cfg.checkpoint.pretrained_checkpoint = None

    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.cuda_graph_impl = "none"
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.mtp_num_layers = 0
    cfg.model.apply_rope_fusion = True
    cfg.model.calculate_per_token_loss = True

    # Selective attention recompute for 8B x 4k headroom on B200.
    cfg.model.recompute_granularity = "selective"
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.recompute_modules = ["core_attn"]

    cfg.train.train_iters = 1000
    cfg.train.global_batch_size = 8
    cfg.train.micro_batch_size = FALCON_ENERGON_MICRO_BATCH_SIZE
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100
    cfg.train.manual_gc_eval = 100
    cfg.validation.eval_interval = 500
    cfg.validation.eval_iters = 8
    cfg.validation.eval_micro_batch_size = FALCON_ENERGON_MICRO_BATCH_SIZE

    cfg.logger.log_throughput_to_tensorboard = True
    cfg.logger.throughput_window_size = 10
    cfg.logger.console_color = "auto"

    opt_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=100,
        lr_decay_iters=1000,
        max_lr=_FALCON_MAX_LR,
        min_lr=_FALCON_MIN_LR,
    )
    cfg.optimizer = opt_cfg
    cfg.scheduler = scheduler_cfg
    cfg.optimizer.use_precision_aware_optimizer = True
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.bfloat16
    cfg.optimizer.exp_avg_sq_dtype = torch.bfloat16

    cfg.tokenizer.tokenizer_model = resolve_llama3_tokenizer_path()
    cfg.tokenizer.use_tokenizer_vocab_size = False
    cfg.dataset = _make_falcon_llama3_energon_dataset(cfg.model.seq_length)

    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.average_in_collective = False  # required by Energon native packing
    cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"

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
        **_B200_ENV_VARS,
    }


def falcon_llama3_8b_pretrain_8gpu_b200_bf16_config() -> ConfigContainer:
    """Return an 8× B200 from-scratch pretrain config for Falcon Llama3-8B.

    Default configuration: 8 GPUs
    - TP=1, PP=1, CP=1 (DP absorbs world size)
    - Dense GQA Llama3-8B trunk + Falcon early-fusion patch embed
    - Plain 1D RoPE (no mRoPE)
    - Energon-native THD packing, MBS=1, GBS=8, seq=4096
    - Selective ``core_attn`` recompute; LR=3e-4
    - No pretrained checkpoint (random init)

    Launch example::

        RECIPE=falcon_llama3_8b_pretrain_8gpu_b200_bf16_config \\
          NUM_GPUS=8 SEQ_LENGTH=4096 \\
          ./examples/models/falcon_mm/thd_energon.sh
    """
    cfg = _pretrain_common()
    cfg.model = falcon_llama3_8b_provider()
    _apply_falcon_llama3_energon_runtime(cfg)
    return cfg


def _apply_falcon_llama3_low_precision(cfg: ConfigContainer, mixed_precision_recipe: str) -> None:
    """Map upstream Llama3-8B low-precision knobs onto Falcon Energon packing.

    Mirrors ``_llama3_8b_pretrain_2gpu_h100_low_precision`` recipe names and the
    NVFP4 last-4-layers BF16 carve-out. Param-gather stays off (Energon path;
    same as Falcon DSv4 MXFP8).
    """
    if mixed_precision_recipe not in _LLAMA3_LOW_PRECISION_RECIPES:
        raise ValueError(
            f"Unsupported Falcon Llama3 low-precision recipe {mixed_precision_recipe!r}; "
            f"expected one of {_LLAMA3_LOW_PRECISION_RECIPES}."
        )

    precision = get_mixed_precision_config(mixed_precision_recipe)
    if mixed_precision_recipe == "bf16_with_nvfp4_mixed":
        # Match library Llama3-8B NVFP4: keep the last 4 layers in BF16.
        precision.first_last_layers_bf16 = True
        precision.num_layers_at_start_in_bf16 = 0
        precision.num_layers_at_end_in_bf16 = 4

    # Energon-native packing: avoid FP8/FP4 param-gather + DDP AG overlap.
    precision.fp8_param_gather = False
    if hasattr(precision, "reuse_grad_buf_for_mxfp8_param_ag"):
        precision.reuse_grad_buf_for_mxfp8_param_ag = False
    if hasattr(precision, "fp4_param_gather"):
        precision.fp4_param_gather = False

    cfg.mixed_precision = precision

    # TE FP8/FP4 kernels require the token count (THD leading dim) divisible by
    # 8. Energon packs without pad_to_max_length have variable width (e.g.
    # seq_length=16384 can yield 16382), which trips
    # assert_dim_for_fp8_exec. Fixed pack width matches Falcon DSv4 MXFP8.
    cfg.dataset.pad_to_max_length = True

    # Upstream Llama3 low-precision recipes use FP32 Adam moments.
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32


def _falcon_llama3_8b_pretrain_8gpu_b200_low_precision(mixed_precision_recipe: str) -> ConfigContainer:
    """Shared 8× B200 Falcon Llama3-8B base with a low-precision recipe applied."""
    cfg = falcon_llama3_8b_pretrain_8gpu_b200_bf16_config()
    _apply_falcon_llama3_low_precision(cfg, mixed_precision_recipe)
    return cfg


def falcon_llama3_8b_pretrain_8gpu_b200_fp8mx_config() -> ConfigContainer:
    """Return an 8× B200 MXFP8 pretrain config for Falcon Llama3-8B.

    Same Energon packing / mesh as the BF16 recipe; mixed precision from
    ``bf16_with_mxfp8_mixed`` (library Llama3-8B / B200 perf ``fp8mx``).
    """
    return _falcon_llama3_8b_pretrain_8gpu_b200_low_precision("bf16_with_mxfp8_mixed")


def falcon_llama3_8b_pretrain_8gpu_b200_fp8cs_config() -> ConfigContainer:
    """Return an 8× B200 FP8 current-scaling pretrain config for Falcon Llama3-8B.

    Same Energon packing / mesh as the BF16 recipe; mixed precision from
    ``bf16_with_fp8_current_scaling_mixed`` (library Llama3-8B ``fp8cs``).
    """
    return _falcon_llama3_8b_pretrain_8gpu_b200_low_precision("bf16_with_fp8_current_scaling_mixed")


def falcon_llama3_8b_pretrain_8gpu_b200_nvfp4_config() -> ConfigContainer:
    """Return an 8× B200 NVFP4 pretrain config for Falcon Llama3-8B.

    Same Energon packing / mesh as the BF16 recipe; mixed precision from
    ``bf16_with_nvfp4_mixed`` with the last 4 layers kept in BF16 (library
    Llama3-8B NVFP4 carve-out).
    """
    return _falcon_llama3_8b_pretrain_8gpu_b200_low_precision("bf16_with_nvfp4_mixed")
