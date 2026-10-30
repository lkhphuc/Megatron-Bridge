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

"""B200 Qwen3-VL recipes with Energon-native packed data."""

from megatron.bridge.data.builders import EnergonDatasetConfig, QwenVLEnergonTaskEncoderConfig
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.utils.cuda_graph import clear_cuda_graph_modules


def qwen3_vl_30b_a3b_pretrain_8gpu_b200_fp8mx_energon_config() -> ConfigContainer:
    """Benchmark full Qwen3-VL 30B-A3B training on 8 B200s with 4096-token THD packs.

    Set ``dataset.path`` to an indexed Qwen ChatMLWebdataset root before
    launching. Raw MINT, caption, and FineWeb mixtures need a Qwen-compatible
    sample adapter.
    This uses the performance recipe's benchmark settings, including forced
    expert balancing, so it is intended for throughput comparison rather than
    convergence training.
    """
    # Import on invocation to keep the library and performance recipe packages
    # free of an import cycle during recipe discovery.
    from megatron.bridge.perf_recipes.qwen_vl.b200.qwen3_vl import (
        qwen3_vl_30b_a3b_pretrain_8gpu_b200_fp8mx_config,
    )

    cfg = qwen3_vl_30b_a3b_pretrain_8gpu_b200_fp8mx_config()
    hf_processor_path = cfg.dataset.hf_processor_path
    vision_config = getattr(cfg.model, "vision_config", None)
    cfg.dataset = EnergonDatasetConfig(
        path=None,
        seq_length=4096,
        micro_batch_size=1,
        num_workers=2,
        do_validation=False,
        packing_buffer_size=16,
        pad_to_max_length=True,
        dataset_kwargs={"repeat": True},
        task_encoder=QwenVLEnergonTaskEncoderConfig(
            hf_processor_path=hf_processor_path,
            patch_size=getattr(vision_config, "patch_size", getattr(cfg.model, "patch_size", 16)),
            temporal_patch_size=getattr(vision_config, "temporal_patch_size", 2),
            spatial_merge_size=getattr(vision_config, "spatial_merge_size", 2),
        ),
    )

    cfg.train.micro_batch_size = 1
    cfg.train.global_batch_size = 64
    cfg.model.seq_length = cfg.dataset.seq_length
    cfg.model.freeze_language_model = False
    cfg.model.freeze_vision_model = False
    cfg.model.freeze_vision_projection = False
    cfg.model.calculate_per_token_loss = True
    cfg.ddp.average_in_collective = False

    # Energon-native packs carry variable THD boundaries that the benchmark's
    # TE-scoped CUDA graphs cannot capture.
    cfg.model.cuda_graph_impl = "none"
    clear_cuda_graph_modules(cfg.model)
    cfg.model.use_te_rng_tracker = False
    cfg.rng.te_rng_tracker = False
    return cfg
