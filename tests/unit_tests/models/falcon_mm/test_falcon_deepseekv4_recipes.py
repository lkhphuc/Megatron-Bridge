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

"""Unit tests for the Falcon-MM DeepSeek-V4 H100 mock recipe (Energon packing)."""

import pytest
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_dsv4_stack_spec

from megatron.bridge.data.builders.energon import EnergonDatasetConfig
from megatron.bridge.models.falcon_mm import FalconDeepseekv4ModelProvider, falcon_deepseekv4_layer_pattern
from megatron.bridge.models.falcon_mm.data import FalconDeepseekv4EnergonTaskEncoderConfig
from megatron.bridge.recipes.falcon_mm.h100 import falcon_deepseekv4 as recipe_module
from megatron.bridge.training.config import ConfigContainer


def _apply_override(obj: object, key: str, value: object) -> None:
    """Set a recipe field only when it already exists on the config object."""
    if not hasattr(obj, key):
        raise ValueError(f"Config has no field '{key}'")
    setattr(obj, key, value)


def _assert_energon_packing_contract(cfg: ConfigContainer) -> None:
    """The invariants Energon-native packing requires (config.py validation)."""
    assert isinstance(cfg, ConfigContainer)
    assert isinstance(cfg.model, FalconDeepseekv4ModelProvider)
    assert isinstance(cfg.dataset, EnergonDatasetConfig)
    assert isinstance(cfg.dataset.task_encoder, FalconDeepseekv4EnergonTaskEncoderConfig)
    assert cfg.dataset.packing_buffer_size == 16
    assert cfg.dataset.micro_batch_size == 1
    assert cfg.train.micro_batch_size == 1
    assert cfg.model.calculate_per_token_loss is True
    assert cfg.ddp.average_in_collective is False
    assert (cfg.model.mtp_num_layers or 0) == 0
    assert cfg.model.cuda_graph_impl == "none"
    assert cfg.model.seq_length == 1024


class TestFalconDeepseekv4Mock2B6PretrainRecipes:
    def test_2gpu_recipe_provider_and_parallelism(self):
        cfg = recipe_module.falcon_deepseekv4_mock_2b6_pretrain_2gpu_h100_bf16_config()
        _assert_energon_packing_contract(cfg)

        assert isinstance(cfg.model, FalconDeepseekv4ModelProvider)
        assert cfg.model.expert_model_parallel_size == 2
        assert cfg.model.tensor_model_parallel_size == 1
        assert cfg.model.pipeline_model_parallel_size == 1
        assert cfg.model.context_parallel_size == 1
        assert cfg.model.sequence_parallel is False

    def test_2gpu_recipe_mock_sizing(self):
        cfg = recipe_module.falcon_deepseekv4_mock_2b6_pretrain_2gpu_h100_bf16_config()
        model = cfg.model
        assert model.num_layers == 12
        assert model.hidden_size == 2048
        assert model.num_moe_experts == 32
        assert model.moe_n_hash_layers == 2
        assert model.vocab_size == 129280
        assert model.image_token_id == 128000

    def test_2gpu_recipe_stack_spec_and_pattern(self):
        cfg = recipe_module.falcon_deepseekv4_mock_2b6_pretrain_2gpu_h100_bf16_config()
        assert cfg.model.hybrid_stack_spec is hybrid_dsv4_stack_spec
        pattern = falcon_deepseekv4_layer_pattern(cfg.model.csa_compress_ratios)
        assert pattern == "WEWE" + "CEHE" * 5

    def test_2gpu_recipe_dataset_encoder(self):
        cfg = recipe_module.falcon_deepseekv4_mock_2b6_pretrain_2gpu_h100_bf16_config()
        assert isinstance(cfg.dataset, EnergonDatasetConfig)
        encoder = cfg.dataset.task_encoder
        assert isinstance(encoder, FalconDeepseekv4EnergonTaskEncoderConfig)
        assert encoder.ds_tokenizer_path
        assert encoder.patch_size == 16
        assert encoder.temporal_patch_size == 2
        assert encoder.spatial_merge_size == 2

    def test_override_guard_rejects_unknown_field(self):
        cfg = recipe_module.falcon_deepseekv4_mock_2b6_pretrain_2gpu_h100_bf16_config()
        with pytest.raises(ValueError, match="Config has no field"):
            _apply_override(cfg.model, "not_a_real_field", 1)

    def test_override_guard_allows_existing_field(self):
        cfg = recipe_module.falcon_deepseekv4_mock_2b6_pretrain_2gpu_h100_bf16_config()
        _apply_override(cfg.train, "global_batch_size", 4)
        assert cfg.train.global_batch_size == 4
