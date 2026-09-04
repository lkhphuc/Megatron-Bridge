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

"""Falcon-MM DeepSeek-V4 dense and MoE recipes for H100.

Launch with ``--step_func vlm_step`` and an Energon dataset path. The scaling
recipes train from scratch; set ``checkpoint.pretrained_checkpoint=null``.
"""

from collections.abc import Callable

from megatron.bridge.models.falcon_mm import FalconDeepseekv4ModelProvider
from megatron.bridge.recipes.common import _pretrain_common
from megatron.bridge.recipes.falcon_mm.falcon_deepseekv4_common import (
    apply_falcon_dsv4_energon_runtime,
    falcon_deepseekv4_1b_provider,
    falcon_deepseekv4_3b_provider,
    falcon_deepseekv4_10b_provider,
    falcon_deepseekv4_35b_a3b_provider,
)
from megatron.bridge.training.config import ConfigContainer


__all__ = [
    "falcon_deepseekv4_1b_pretrain_2gpu_h100_bf16_config",
    "falcon_deepseekv4_3b_pretrain_2gpu_h100_bf16_config",
    "falcon_deepseekv4_10b_pretrain_2gpu_h100_bf16_config",
    "falcon_deepseekv4_35b_a3b_pretrain_8gpu_h100_bf16_config",
]


def _falcon_deepseekv4_pretrain_base(
    provider_factory: Callable[[], FalconDeepseekv4ModelProvider],
    *,
    expert_parallel_size: int,
) -> ConfigContainer:
    """Shared H100 base for dense and MoE Falcon DSv4 recipes."""
    cfg = _pretrain_common()
    cfg.model = provider_factory()
    apply_falcon_dsv4_energon_runtime(
        cfg,
        max_lr=3e-4,
        min_lr=3e-5,
        env_vars={"CUDA_DEVICE_MAX_CONNECTIONS": 1},
    )
    cfg.model.expert_model_parallel_size = expert_parallel_size
    return cfg


def falcon_deepseekv4_1b_pretrain_2gpu_h100_bf16_config() -> ConfigContainer:
    """Return a 2-GPU H100 from-scratch pretrain config for Falcon DSv4 dense ~1B.

    Default configuration: 2 GPU
    - TP=1, PP=1, CP=1, EP=1 (dense MLP; DP absorbs world size)
    - Energon-native THD packing, MBS=1, seq=1024
    - Selective recompute: ``mla_up_proj``, ``mhc``

    Launch with ``--step_func vlm_step``. From-scratch smoke: pass
    ``checkpoint.pretrained_checkpoint=null``.
    """
    return _falcon_deepseekv4_pretrain_base(falcon_deepseekv4_1b_provider, expert_parallel_size=1)


def falcon_deepseekv4_3b_pretrain_2gpu_h100_bf16_config() -> ConfigContainer:
    """Return a 2-GPU H100 from-scratch pretrain config for Falcon DSv4 dense ~3B."""
    return _falcon_deepseekv4_pretrain_base(falcon_deepseekv4_3b_provider, expert_parallel_size=1)


def falcon_deepseekv4_10b_pretrain_2gpu_h100_bf16_config() -> ConfigContainer:
    """Return a 2-GPU H100 from-scratch pretrain config for Falcon DSv4 dense ~10B."""
    return _falcon_deepseekv4_pretrain_base(falcon_deepseekv4_10b_provider, expert_parallel_size=1)


def falcon_deepseekv4_35b_a3b_pretrain_8gpu_h100_bf16_config() -> ConfigContainer:
    """Return an 8-GPU H100 from-scratch pretrain config for Falcon DSv4 ~35B-A3B MoE.

    Default configuration: 8 GPU
    - TP=1, PP=1, CP=1, EP=8
    - Shared Falcon Energon runtime
    """
    return _falcon_deepseekv4_pretrain_base(falcon_deepseekv4_35b_a3b_provider, expert_parallel_size=8)
