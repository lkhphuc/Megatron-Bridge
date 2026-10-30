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

"""Falcon-MM (encoder-free) Qwen3.5 recipes for eight B200 GPUs.

Execution resize of the H100 Energon THD recipes: same packing / loss /
optimizer contract, with a B200-friendly mesh (PP=1, full-node EP) and process
env. CUDA graphs stay off (Energon-native packing).
"""

from __future__ import annotations

from megatron.bridge.recipes.falcon_mm.h100 import falcon_qwen35 as h100
from megatron.bridge.recipes.utils.environment_utils import COMMON_RECIPE_ENV_VARS
from megatron.bridge.training.config import ConfigContainer


# B200 process env shared with Falcon DSv4 B200 / Qwen3.5-VL B200 perf recipes.
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

__all__ = [
    "falcon_qwen35_2b_sft_8gpu_b200_bf16_config",
    "falcon_qwen35_2b_pretrain_8gpu_b200_bf16_config",
    "falcon_qwen35_35b_a3b_sft_8gpu_b200_bf16_config",
    "falcon_qwen35_35b_a3b_pretrain_8gpu_b200_bf16_config",
]


def _apply_b200_env(cfg: ConfigContainer) -> None:
    """Merge B200 process env on top of the recipe's HybridEP / MoE env."""
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
        **cfg.env_vars,
        **_B200_ENV_VARS,
    }


def falcon_qwen35_2b_sft_8gpu_b200_bf16_config() -> ConfigContainer:
    """Return an 8× B200 SFT config for Falcon-MM 2B (Qwen3.5-2B trunk).

    Default configuration: 8 GPUs
    - TP=1, PP=1, CP=1 (DP absorbs world size)
    - Energon-native THD packing (packing_buffer_size=16), MBS=1, GBS=8
    - LR=5e-6 (full SFT), sequence length 4096
    """
    cfg = h100._falcon_qwen35_2b_sft_base()
    cfg.train.global_batch_size = 8
    _apply_b200_env(cfg)
    return cfg


def falcon_qwen35_2b_pretrain_8gpu_b200_bf16_config() -> ConfigContainer:
    """Return an 8× B200 from-scratch toy-MoE pretrain config for Falcon-MM 2B.

    Default configuration: 8 GPUs
    - TP=1, PP=1, EP=8, CP=1
    - Same 8-expert / top-2 toy MoE as the H100 recipes
    - No pretrained checkpoint (random init)

    Launch with ``checkpoint.pretrained_checkpoint=null``.
    """
    cfg = h100._falcon_qwen35_2b_moe_pretrain_base()
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.expert_model_parallel_size = 8
    cfg.train.global_batch_size = 16
    _apply_b200_env(cfg)
    return cfg


def falcon_qwen35_35b_a3b_sft_8gpu_b200_bf16_config() -> ConfigContainer:
    """Return an 8× B200 SFT config for Falcon-MM 35B-A3B MoE (Qwen3.5-35B-A3B).

    Execution resize of ``falcon_qwen35_35b_a3b_sft_8gpu_h100_bf16_config``:
    B200 HBM lets us drop the H100 PP=2 / full-recompute layout.

    Default configuration: 8 GPUs
    - TP=1, PP=1, EP=8, CP=1
    - Energon-native THD packing (packing_buffer_size=16), MBS=1, GBS=16
    - alltoall MoE dispatcher (correctness-oriented; EP overlap off)
    - No activation recompute (restore full uniform recompute on OOM)
    - Sequence length: 4096
    """
    cfg = h100._falcon_qwen35_35b_a3b_sft_base()

    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.expert_model_parallel_size = 8

    # H100 8-GPU needed full recompute; B200 + EP=8 should fit without it.
    cfg.model.recompute_granularity = None
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.recompute_modules = None

    _apply_b200_env(cfg)
    return cfg


def falcon_qwen35_35b_a3b_pretrain_8gpu_b200_bf16_config() -> ConfigContainer:
    """Return an 8× B200 from-scratch pretrain config for Falcon-MM 35B-A3B.

    Adapts ``qwen35_vl_35b_a3b_pretrain_8gpu_b200_bf16_config`` onto Falcon-MM
    Energon-native THD packing:

    - TP=1, PP=1, CP=1, EP=8, HybridEP
    - seq_length 4096, MBS=1, GBS=16 (raise toward 512 for throughput parity)
    - No activation recompute (B200 HBM; restore selective on OOM)
    - Pretrain LR 3e-4 / 3e-5, precision-aware Adam
    - CUDA graphs stay off (Energon packing); EP overlap off
    - Natural MoE routing (not force-balanced)

    Launch with ``--mode pretrain --step_func vlm_step`` and
    ``checkpoint.pretrained_checkpoint=null``.
    """
    cfg = h100._falcon_qwen35_35b_a3b_pretrain_base()

    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_dtype = None
    cfg.model.expert_model_parallel_size = 8
    cfg.train.global_batch_size = 16

    # B200 perf recipe drops recompute; Falcon has no ViT so headroom is larger.
    cfg.model.recompute_granularity = None
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.recompute_modules = None

    h100._apply_falcon_qwen35_moe_hybridep(cfg, expert_parallel_size=8)
    # B200 perf uses SM margin 20; HybridEP helper sets 0 for the H100 path.
    _apply_b200_env(cfg)
    cfg.env_vars["NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN"] = 8
    return cfg
