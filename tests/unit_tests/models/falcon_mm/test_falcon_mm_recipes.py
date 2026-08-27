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

"""Unit tests for the Falcon-MM H100 recipes (Energon-native packing)."""

import pytest

from megatron.bridge.data.builders.energon import EnergonDatasetConfig
from megatron.bridge.models.falcon_mm import FalconQwen35Model, FalconQwen35ModelProvider, FalconQwen35MoEModelProvider
from megatron.bridge.models.falcon_mm.data import FalconEnergonTaskEncoderConfig
from megatron.bridge.models.qwen_vl.qwen35_vl_provider import Qwen35VLModelProvider, Qwen35VLMoEModelProvider
from megatron.bridge.recipes.falcon_mm.h100 import falcon_qwen35 as recipe_module
from megatron.bridge.training.config import ConfigContainer


def _minimal_qwen35_vl_provider() -> Qwen35VLModelProvider:
    return Qwen35VLModelProvider(
        num_layers=2,
        hidden_size=64,
        ffn_hidden_size=128,
        num_attention_heads=4,
        kv_channels=32,
        seq_length=64,
        vocab_size=128,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
    )


def _minimal_qwen35_vl_moe_provider() -> Qwen35VLMoEModelProvider:
    return Qwen35VLMoEModelProvider(
        num_layers=2,
        hidden_size=64,
        ffn_hidden_size=128,
        num_attention_heads=4,
        kv_channels=32,
        seq_length=64,
        vocab_size=128,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        num_moe_experts=8,
        moe_router_topk=2,
        moe_ffn_hidden_size=32,
        moe_shared_expert_intermediate_size=32,
    )


class _FakeAutoBridge:
    """Returns a real (tiny) Qwen3.5-VL provider without any I/O."""

    _hf_path: str = "Qwen/Qwen3.5-2B"

    @staticmethod
    def from_hf_pretrained(hf_path: str):
        bridge = _FakeAutoBridge()
        bridge._hf_path = hf_path
        return bridge

    def to_megatron_provider(self, load_weights: bool = False):
        if "35B-A3B" in self._hf_path or "A3B" in self._hf_path:
            return _minimal_qwen35_vl_moe_provider()
        return _minimal_qwen35_vl_provider()


@pytest.fixture(autouse=True)
def _patch_autobridge(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(recipe_module, "AutoBridge", _FakeAutoBridge)


def _assert_energon_packing_contract(cfg: ConfigContainer) -> None:
    """The invariants Energon-native packing requires (config.py validation)."""
    assert isinstance(cfg, ConfigContainer)
    assert isinstance(cfg.model, (FalconQwen35ModelProvider, FalconQwen35MoEModelProvider))
    assert cfg.model.MODEL_CLASS is FalconQwen35Model
    assert isinstance(cfg.dataset, EnergonDatasetConfig)
    assert isinstance(cfg.dataset.task_encoder, FalconEnergonTaskEncoderConfig)
    assert cfg.dataset.packing_buffer_size == 16
    assert cfg.dataset.micro_batch_size == 1
    assert cfg.train.micro_batch_size == 1
    assert cfg.model.calculate_per_token_loss is True
    assert cfg.ddp.average_in_collective is False
    assert (cfg.model.mtp_num_layers or 0) == 0
    assert cfg.model.cuda_graph_impl == "none"


class TestFalconQwen352BSftRecipes:
    def test_1gpu_recipe_is_pp1_baseline(self):
        cfg = recipe_module.falcon_qwen35_2b_sft_1gpu_h100_bf16_config()
        _assert_energon_packing_contract(cfg)
        assert isinstance(cfg.model, FalconQwen35ModelProvider)
        assert cfg.model.pipeline_model_parallel_size == 1
        assert cfg.model.virtual_pipeline_model_parallel_size is None

    def test_2gpu_recipe_enables_pp2(self):
        cfg = recipe_module.falcon_qwen35_2b_sft_2gpu_h100_bf16_config()
        _assert_energon_packing_contract(cfg)
        assert cfg.model.pipeline_model_parallel_size == 2
        # VPP must stay unset for Energon packing (loader-state resume gap).
        assert cfg.model.virtual_pipeline_model_parallel_size is None

    def test_recipes_share_base(self):
        cfg1 = recipe_module.falcon_qwen35_2b_sft_1gpu_h100_bf16_config()
        cfg2 = recipe_module.falcon_qwen35_2b_sft_2gpu_h100_bf16_config()
        assert cfg1.model.seq_length == cfg2.model.seq_length == 4096
        assert cfg1.train.global_batch_size == cfg2.train.global_batch_size
        assert cfg1.dataset.seq_length == cfg2.dataset.seq_length


class TestFalconQwen352BMoEPretrainRecipes:
    def test_4gpu_recipe_builds_moe_from_dense(self):
        cfg = recipe_module.falcon_qwen35_2b_moe_pretrain_4gpu_h100_bf16_config()
        _assert_energon_packing_contract(cfg)
        assert isinstance(cfg.model, FalconQwen35MoEModelProvider)
        assert cfg.model.num_moe_experts == 8
        assert cfg.model.moe_router_topk == 2
        assert cfg.model.moe_ffn_hidden_size == max(128 // 4, 128)  # dense ffn in fake provider is 128
        assert cfg.model.pipeline_model_parallel_size == 2
        assert cfg.model.expert_model_parallel_size == 2
        assert cfg.model.moe_token_dispatcher_type == "alltoall"
        assert cfg.checkpoint.pretrained_checkpoint is None

    def test_8gpu_recipe_scales_ep(self):
        cfg = recipe_module.falcon_qwen35_2b_moe_pretrain_8gpu_h100_bf16_config()
        _assert_energon_packing_contract(cfg)
        assert isinstance(cfg.model, FalconQwen35MoEModelProvider)
        assert cfg.model.expert_model_parallel_size == 4
        assert cfg.train.global_batch_size == 16
        assert cfg.checkpoint.pretrained_checkpoint is None


class TestFalconQwen3535BA3BSftRecipes:
    def test_8gpu_recipe_enables_moe_and_pp(self):
        cfg = recipe_module.falcon_qwen35_35b_a3b_sft_8gpu_h100_bf16_config()
        _assert_energon_packing_contract(cfg)
        assert isinstance(cfg.model, FalconQwen35MoEModelProvider)
        assert cfg.model.pipeline_model_parallel_size == 2
        assert cfg.model.expert_model_parallel_size == 4
        assert cfg.model.tensor_model_parallel_size == 1
        assert cfg.model.moe_token_dispatcher_type == "alltoall"
        assert cfg.model.virtual_pipeline_model_parallel_size is None

    def test_16gpu_recipe_scales_ep(self):
        cfg = recipe_module.falcon_qwen35_35b_a3b_sft_16gpu_h100_bf16_config()
        _assert_energon_packing_contract(cfg)
        assert isinstance(cfg.model, FalconQwen35MoEModelProvider)
        assert cfg.model.pipeline_model_parallel_size == 2
        assert cfg.model.expert_model_parallel_size == 8
        assert cfg.train.global_batch_size == 32
        assert cfg.model.moe_token_dispatcher_type == "alltoall"
