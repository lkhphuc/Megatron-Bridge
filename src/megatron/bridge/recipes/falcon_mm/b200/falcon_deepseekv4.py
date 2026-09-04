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

"""Falcon-MM DeepSeek-V4 recipes for eight B200 GPUs.

Dense 1B/3B/10B and MoE 35B-A3B providers share the Falcon Energon runtime.
The MoE MXFP8 variant preserves the BF16 indexer loss and routing semantics.
The full Flash recipe uses HF architecture metadata for weight-import bring-up.
"""

from collections.abc import Callable

import torch

from megatron.bridge.models.deepseek.deepseek_v4_bridge import (
    deepseek_v4_supports_blackwell_fused_kernels,
)
from megatron.bridge.models.falcon_mm import (
    FalconDeepseekv4Bridge,
    FalconDeepseekv4ModelProvider,
)
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.recipes.common import _pretrain_common
from megatron.bridge.recipes.falcon_mm.falcon_deepseekv4_common import (
    FALCON_DSV4_B200_ENV_VARS,
    apply_falcon_dsv4_blackwell_features,
    apply_falcon_dsv4_energon_runtime,
    apply_falcon_dsv4_qwen30b_style_mxfp8_throughput,
    falcon_deepseekv4_1b_provider,
    falcon_deepseekv4_3b_provider,
    falcon_deepseekv4_10b_provider,
    falcon_deepseekv4_35b_a3b_provider,
)
from megatron.bridge.training.config import ConfigContainer


DEEPSEEK_V4_FLASH_HF_PATH = "deepseek-ai/DeepSeek-V4-Flash"

_B200_ENV_VARS = FALCON_DSV4_B200_ENV_VARS

_FALCON_MAX_LR = 3e-4
_FALCON_MIN_LR = 3e-5

__all__ = [
    "falcon_deepseekv4_1b_pretrain_8gpu_b200_bf16_config",
    "falcon_deepseekv4_3b_pretrain_8gpu_b200_bf16_config",
    "falcon_deepseekv4_10b_pretrain_8gpu_b200_bf16_config",
    "falcon_deepseekv4_35b_a3b_pretrain_8gpu_b200_bf16_config",
    "falcon_deepseekv4_35b_a3b_pretrain_8gpu_b200_fp8mx_config",
    "falcon_deepseekv4_flash_pretrain_8gpu_b200_bf16_config",
]


def _falcon_deepseekv4_flash_provider() -> FalconDeepseekv4ModelProvider:
    """Build an exact DeepSeek-V4-Flash provider on the Falcon HybridModel trunk.

    Uses :class:`FalconDeepseekv4Bridge` explicitly (not AutoBridge; the Falcon
    bridge is not registered under ``deepseek_v4``). Only the HF config is
    required for architecture; weights are not loaded here.
    """
    hf_pretrained = PreTrainedCausalLM.from_pretrained(
        DEEPSEEK_V4_FLASH_HF_PATH,
        trust_remote_code=True,
    )
    provider = FalconDeepseekv4Bridge().provider_bridge(hf_pretrained)
    assert isinstance(provider, FalconDeepseekv4ModelProvider)

    # Bridge defaults to flex/hybridep; keep HybridEP for B200 Flash bring-up.
    provider.moe_token_dispatcher_type = "flex"
    provider.moe_flex_dispatcher_backend = "hybridep"
    provider.moe_flex_dispatcher_num_sms = 16
    provider.moe_shared_expert_overlap = False
    provider.moe_grouped_gemm = True
    provider.moe_permute_fusion = True

    # Hash image pads bypass tid2eid via shared expert_bias (text keeps tid2eid).
    provider.moe_vl_routing = True

    # Fused CE is required at longer sequences; native impl matches the Falcon DSv4 recipes.
    provider.cross_entropy_loss_fusion = True
    provider.cross_entropy_fusion_impl = "native"
    provider.calculate_per_token_loss = True
    provider.attention_backend = "auto"
    provider.gradient_accumulation_fusion = False
    provider.apply_rope_fusion = True
    if deepseek_v4_supports_blackwell_fused_kernels():
        provider.use_fused_mhc = True
    # Force fused CSA/DSA on B200 — unfused path dominates the 8k memory peak.
    provider.dsa_kernel_backend = "cudnn"

    # Devenv-first sequence; override to 4096 via CLI when memory allows.
    provider.seq_length = 1024
    return provider


def _apply_falcon_dsv4_flash_energon_runtime(cfg: ConfigContainer, *, max_lr: float, min_lr: float) -> None:
    """Apply the shared Falcon runtime with the short Flash bring-up schedule."""
    apply_falcon_dsv4_energon_runtime(
        cfg,
        max_lr=max_lr,
        min_lr=min_lr,
        env_vars=_B200_ENV_VARS,
        enable_blackwell_kernels=True,
        save_interval=100,
        train_iters=100,
        lr_warmup_iters=10,
        lr_decay_iters=100,
        manual_gc_interval=50,
        eval_interval=50,
        eval_iters=4,
        enable_megatron_core_experimental=True,
    )
    apply_falcon_dsv4_blackwell_features(cfg)


def _scale_b200_base(
    provider_factory: Callable[[], FalconDeepseekv4ModelProvider],
    *,
    expert_parallel_size: int,
) -> ConfigContainer:
    """Shared 8x B200 dense/MoE base with explicit expert parallelism."""
    cfg = _pretrain_common()
    cfg.model = provider_factory()
    apply_falcon_dsv4_energon_runtime(
        cfg,
        max_lr=_FALCON_MAX_LR,
        min_lr=_FALCON_MIN_LR,
        env_vars=_B200_ENV_VARS,
        enable_blackwell_kernels=True,
        enable_megatron_core_experimental=True,
    )
    apply_falcon_dsv4_blackwell_features(cfg)
    cfg.model.expert_model_parallel_size = expert_parallel_size
    if cfg.model.num_moe_experts is not None:
        cfg.env_vars = {
            **cfg.env_vars,
            "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": expert_parallel_size,
        }
    return cfg


def falcon_deepseekv4_1b_pretrain_8gpu_b200_bf16_config() -> ConfigContainer:
    """Return an 8x B200 from-scratch pretrain config for Falcon DSv4 dense ~1B.

    Shared Blackwell fused path (fused mHC/DSA, experimental Core,
    B200 env). Dense MLP: TP=1, PP=1, CP=1, EP=1 (DP absorbs world size).
    """
    return _scale_b200_base(falcon_deepseekv4_1b_provider, expert_parallel_size=1)


def falcon_deepseekv4_3b_pretrain_8gpu_b200_bf16_config() -> ConfigContainer:
    """Return an 8x B200 from-scratch pretrain config for Falcon DSv4 dense ~3B."""
    return _scale_b200_base(falcon_deepseekv4_3b_provider, expert_parallel_size=1)


def falcon_deepseekv4_10b_pretrain_8gpu_b200_bf16_config() -> ConfigContainer:
    """Return an 8x B200 from-scratch pretrain config for Falcon DSv4 dense ~10B."""
    return _scale_b200_base(falcon_deepseekv4_10b_provider, expert_parallel_size=1)


def falcon_deepseekv4_35b_a3b_pretrain_8gpu_b200_bf16_config() -> ConfigContainer:
    """Return an 8x B200 from-scratch pretrain config for Falcon DSv4 ~35B-A3B MoE.

    Shared Blackwell + HybridEP path. Mesh: TP=1, PP=1, CP=1, EP=8.
    Defaults stay devenv-sized (GBS=8, selective recompute). For the Qwen3-30B
    MXFP8 throughput shape on the same Energon mixture, use
    :func:`falcon_deepseekv4_moe_35b_a3b_pretrain_8gpu_b200_fp8mx_config`.
    """
    return _scale_b200_base(falcon_deepseekv4_35b_a3b_provider, expert_parallel_size=8)


def falcon_deepseekv4_35b_a3b_pretrain_8gpu_b200_fp8mx_config() -> ConfigContainer:
    """Return an 8x B200 MXFP8 throughput config for Falcon DSv4 ~35B-A3B MoE.

    Adapts the upstream Qwen3-30B-A3B 8-GPU MXFP8 recipe
    (``qwen3_30b_a3b_pretrain_8gpu_b200_fp8mx_config`` / GB200 40,960 tok/s/GPU
    row) onto Falcon-MM Energon-native THD packing:

    - TP=1, PP=1, CP=1, EP=8, HybridEP
    - seq_length / pack width 4096, GBS=512, physical MBS=1 (packing)
    - MXFP8 + TE CUDA graphs (``attn``, ``moe_router``, ``moe_preprocess``)
    - ``pad_to_max_length=True`` for HybridEP under CUDA graphs
    - no activation recompute (restore ``mla_up_proj``+``mhc`` on OOM)
    - EP overlap stays off (Energon VLM schedule-plan constraint)
    - natural MoE routing (not force-balanced)

    Launch with ``--step_func vlm_step`` and point ``dataset.path`` at the
    Energon mixture. Expect lower absolute TPS than the text-only Qwen3
    benchmark: vision decode, MBS=1 vs MBS=4, and disabled EP overlap all
    move the ceiling.
    """
    cfg = falcon_deepseekv4_35b_a3b_pretrain_8gpu_b200_bf16_config()
    apply_falcon_dsv4_qwen30b_style_mxfp8_throughput(cfg)
    return cfg


def falcon_deepseekv4_flash_pretrain_8gpu_b200_bf16_config() -> ConfigContainer:
    """Return an 8x B200 from-scratch Flash pretrain config for Falcon DSv4.

    Architecture matches ``deepseek-ai/DeepSeek-V4-Flash`` (via
    :class:`FalconDeepseekv4Bridge`) so language weights can be imported later.
    This is an execution resize for devenv parallelism bring-up — not the
    64-GPU GB200 MXFP8 perf contract.

    Default configuration: 8 GPU
    - TP=1, PP=4, CP=1, EP=2, ETP=1, VPP=None (world = PP x EP = 8)
    - HybridEP dispatcher, fused RoPE/mHC/DSA
    - Energon-native THD packing (packing_buffer_size=16), MBS=1, GBS=8
    - BF16, selective recompute ``mla_up_proj``+``mhc``
    - Sequence length: 1024 (override to 4096 when memory allows)
    - MTP disabled (Energon packing); patch embed random for text CKPTs

    Launch with ``--step_func vlm_step``. From-scratch smoke: pass
    ``checkpoint.pretrained_checkpoint=null``. Point ``dataset.path`` at the
    Energon mixture. If this OOMs, try TP1/PP2/EP4 before scaling GPUs.
    """
    cfg = _pretrain_common()
    cfg.model = _falcon_deepseekv4_flash_provider()

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 2
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.model.params_dtype = torch.bfloat16

    _apply_falcon_dsv4_flash_energon_runtime(cfg, max_lr=3e-4, min_lr=3e-5)
    # Flash 8-GPU mesh is PP4/EP2 — HybridEP domain must match EP group size.
    cfg.env_vars = {
        **cfg.env_vars,
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": cfg.model.expert_model_parallel_size,
    }
    return cfg
