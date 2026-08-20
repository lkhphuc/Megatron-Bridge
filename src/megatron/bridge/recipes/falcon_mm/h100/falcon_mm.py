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

import torch

from megatron.bridge import AutoBridge
from megatron.bridge.data.builders.energon import EnergonDatasetConfig
from megatron.bridge.models.falcon_mm import FalconMMModelProvider
from megatron.bridge.models.falcon_mm.data import FalconEnergonTaskEncoderConfig
from megatron.bridge.recipes.common import _sft_common_vlm
from megatron.bridge.recipes.utils.environment_utils import COMMON_RECIPE_ENV_VARS
from megatron.bridge.recipes.utils.optimizer_utils import distributed_fused_adam_with_cosine_annealing
from megatron.bridge.training.config import ConfigContainer


__all__ = [
    "falcon_mm_2b_sft_1gpu_h100_bf16_config",
    "falcon_mm_2b_sft_2gpu_h100_bf16_config",
]


def _make_falcon_energon_dataset(hf_path: str, seq_length: int, micro_batch_size: int) -> EnergonDatasetConfig:
    """Create the Energon dataset config for Falcon-MM (THD Energon-native packing only).

    ``path`` is intentionally None — point it at an indexed Energon dataset root
    or a MetadatasetV2 YAML mixture via CLI: ``dataset.path=<path>``.
    """
    return EnergonDatasetConfig(
        path=None,
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        num_workers=2,
        packing_buffer_size=16,
        task_encoder=FalconEnergonTaskEncoderConfig(hf_processor_path=hf_path),
    )


def _falcon_mm_2b_sft_base() -> ConfigContainer:
    """Shared base for the Falcon-MM 2B (Qwen3.5-2B trunk) SFT dev recipes."""
    cfg = _sft_common_vlm()

    # Model config: Qwen3.5-2B LM trunk + encoder-free patch embed.
    hf_path = "Qwen/Qwen3.5-2B"
    cfg.model = FalconMMModelProvider.from_qwen35_vl_provider(
        AutoBridge.from_hf_pretrained(hf_path).to_megatron_provider(load_weights=False)
    )
    cfg.model.seq_length = 4096

    # Parallelism settings
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
    cfg.train.global_batch_size = 8
    cfg.train.micro_batch_size = 1  # required by Energon native packing
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100
    cfg.train.manual_gc_eval = 100
    cfg.validation.eval_interval = 500
    cfg.validation.eval_iters = 32

    # Optimizer
    opt_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=200,
        lr_decay_iters=300000,
        max_lr=5e-6,
        min_lr=5e-7,
    )
    cfg.optimizer = opt_cfg
    cfg.scheduler = scheduler_cfg
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # Dataset config: THD Energon-native packing only
    cfg.dataset = _make_falcon_energon_dataset(hf_path, cfg.model.seq_length, cfg.train.micro_batch_size)

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
    return cfg


def falcon_mm_2b_sft_1gpu_h100_bf16_config() -> ConfigContainer:
    """Return an SFT dev config for Falcon-MM 2B (encoder-free, Qwen3.5-2B trunk).

    Default configuration: 1 GPU
    - TP=1, PP=1, CP=1
    - Energon-native THD packing (packing_buffer_size=16), MBS=1
    - LR=5e-6 (full SFT)
    - Sequence length: 4096

    This is the PP=1 baseline for the PP loss-parity sentinel
    (energon-packing-pp-enablement.md, step 2).
    """
    return _falcon_mm_2b_sft_base()


def falcon_mm_2b_sft_2gpu_h100_bf16_config() -> ConfigContainer:
    """Return an SFT dev config for Falcon-MM 2B (encoder-free, Qwen3.5-2B trunk).

    Default configuration: 2 GPU
    - TP=1, PP=2, CP=1
    - Energon-native THD packing (packing_buffer_size=16), MBS=1
    - LR=5e-6 (full SFT)
    - Sequence length: 4096

    Requires the Energon-packing PP guard to be removed
    (src/megatron/bridge/training/config.py — see energon-packing-pp-enablement.md).
    With PP>1, packs become fixed-width (pad_to_max_length auto-flip) and VPP
    must stay unset (Energon loader-state resume gap).
    """
    cfg = _falcon_mm_2b_sft_base()
    cfg.model.pipeline_model_parallel_size = 2
    cfg.model.moe_token_dispatcher_type = "all2all"
    return cfg
