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

"""Unit tests for Falcon-MM Llama3 recipes."""

from megatron.bridge.data.builders.energon import EnergonDatasetConfig
from megatron.bridge.models.falcon_mm import FalconLlama3ModelProvider
from megatron.bridge.models.falcon_mm.data import FalconLlama3EnergonTaskEncoderConfig
from megatron.bridge.recipes.falcon_mm.b200.falcon_llama3 import (
    falcon_llama3_8b_pretrain_8gpu_b200_bf16_config,
    falcon_llama3_8b_pretrain_8gpu_b200_fp8cs_config,
    falcon_llama3_8b_pretrain_8gpu_b200_fp8mx_config,
    falcon_llama3_8b_pretrain_8gpu_b200_nvfp4_config,
    falcon_llama3_8b_provider,
)
from megatron.bridge.recipes.falcon_mm.dataset import FALCON_ENERGON_MICRO_BATCH_SIZE
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.mixed_precision import MixedPrecisionConfig


class TestFalconLlama3Recipes:
    def test_provider_geometry(self):
        provider = falcon_llama3_8b_provider()
        assert isinstance(provider, FalconLlama3ModelProvider)
        assert provider.num_layers == 32
        assert provider.hidden_size == 4096
        assert provider.num_query_groups == 8
        assert provider.seq_length == 4096
        assert provider.calculate_per_token_loss is True

    def test_8gpu_b200_recipe(self):
        cfg = falcon_llama3_8b_pretrain_8gpu_b200_bf16_config()
        assert isinstance(cfg, ConfigContainer)
        assert isinstance(cfg.model, FalconLlama3ModelProvider)
        assert cfg.model.tensor_model_parallel_size == 1
        assert cfg.model.pipeline_model_parallel_size == 1
        assert cfg.model.context_parallel_size == 1
        assert cfg.model.seq_length == 4096
        assert cfg.model.recompute_modules == ["core_attn"]
        assert cfg.model.cuda_graph_impl == "none"
        assert cfg.train.micro_batch_size == FALCON_ENERGON_MICRO_BATCH_SIZE
        assert cfg.train.global_batch_size == 8
        assert cfg.checkpoint.pretrained_checkpoint is None
        assert cfg.ddp.average_in_collective is False
        assert isinstance(cfg.dataset, EnergonDatasetConfig)
        assert isinstance(cfg.dataset.task_encoder, FalconLlama3EnergonTaskEncoderConfig)
        assert cfg.dataset.task_encoder.llama_tokenizer_path == "meta-llama/Meta-Llama-3-8B"
        assert cfg.tokenizer.tokenizer_model == "meta-llama/Meta-Llama-3-8B"
        assert cfg.env_vars["CUDA_DEVICE_MAX_CONNECTIONS"] == 32

    def test_tokenizer_path_env_override(self, monkeypatch):
        from megatron.bridge.recipes.falcon_mm.b200 import falcon_llama3 as recipe_module

        monkeypatch.setenv("LLAMA3_TOKENIZER_PATH", "/tmp/local-llama3-tok")
        assert recipe_module.resolve_llama3_tokenizer_path() == "/tmp/local-llama3-tok"
        cfg = falcon_llama3_8b_pretrain_8gpu_b200_bf16_config()
        assert cfg.tokenizer.tokenizer_model == "/tmp/local-llama3-tok"
        assert cfg.dataset.task_encoder.llama_tokenizer_path == "/tmp/local-llama3-tok"

    def test_low_precision_recipes(self):
        mx = falcon_llama3_8b_pretrain_8gpu_b200_fp8mx_config()
        assert isinstance(mx.mixed_precision, MixedPrecisionConfig)
        assert mx.mixed_precision.fp8_recipe == "mxfp8"
        assert mx.mixed_precision.fp8_param_gather is False
        assert mx.ddp.average_in_collective is False
        assert mx.dataset.pad_to_max_length is True

        cs = falcon_llama3_8b_pretrain_8gpu_b200_fp8cs_config()
        assert cs.mixed_precision.fp8_recipe == "tensorwise"
        assert cs.mixed_precision.fp8_param_gather is False
        assert cs.dataset.pad_to_max_length is True

        nv = falcon_llama3_8b_pretrain_8gpu_b200_nvfp4_config()
        assert nv.mixed_precision.fp4_recipe == "nvfp4"
        assert nv.mixed_precision.first_last_layers_bf16 is True
        assert nv.mixed_precision.num_layers_at_end_in_bf16 == 4
        assert nv.mixed_precision.fp4_param_gather is False
        assert nv.optimizer.use_precision_aware_optimizer is False
        assert nv.dataset.pad_to_max_length is True
