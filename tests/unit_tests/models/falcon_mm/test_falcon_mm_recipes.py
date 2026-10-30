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

"""Smoke tests: Falcon-MM Qwen recipes wire the Falcon Energon task encoder."""

import pytest

from megatron.bridge.data.builders.energon import EnergonDatasetConfig
from megatron.bridge.models.falcon_mm.data import FalconEnergonTaskEncoderConfig
from megatron.bridge.models.qwen_vl.qwen35_vl_provider import Qwen35VLModelProvider, Qwen35VLMoEModelProvider
from megatron.bridge.recipes.falcon_mm.b200 import falcon_qwen35 as b200_recipe_module
from megatron.bridge.recipes.falcon_mm.h100 import falcon_qwen35 as h100_recipe_module


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
    _hf_path: str = "Qwen/Qwen3.5-2B"

    @staticmethod
    def from_hf_pretrained(hf_path: str):
        bridge = _FakeAutoBridge()
        bridge._hf_path = hf_path
        return bridge

    def to_megatron_provider(self, load_weights: bool = False):
        if "A3B" in self._hf_path:
            return _minimal_qwen35_vl_moe_provider()
        return _minimal_qwen35_vl_provider()


@pytest.fixture(autouse=True)
def _patch_autobridge(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(h100_recipe_module, "AutoBridge", _FakeAutoBridge)


@pytest.mark.parametrize(
    "recipe_fn",
    [
        h100_recipe_module.falcon_qwen35_2b_sft_1gpu_h100_bf16_config,
        h100_recipe_module.falcon_qwen35_2b_sft_2gpu_h100_bf16_config,
        h100_recipe_module.falcon_qwen35_2b_pretrain_4gpu_h100_bf16_config,
        h100_recipe_module.falcon_qwen35_2b_pretrain_8gpu_h100_bf16_config,
        h100_recipe_module.falcon_qwen35_35b_a3b_sft_8gpu_h100_bf16_config,
        h100_recipe_module.falcon_qwen35_35b_a3b_sft_16gpu_h100_bf16_config,
        h100_recipe_module.falcon_qwen35_35b_a3b_pretrain_8gpu_h100_bf16_config,
        h100_recipe_module.falcon_qwen35_35b_a3b_pretrain_16gpu_h100_bf16_config,
        b200_recipe_module.falcon_qwen35_2b_sft_8gpu_b200_bf16_config,
        b200_recipe_module.falcon_qwen35_2b_pretrain_8gpu_b200_bf16_config,
        b200_recipe_module.falcon_qwen35_35b_a3b_sft_8gpu_b200_bf16_config,
        b200_recipe_module.falcon_qwen35_35b_a3b_pretrain_8gpu_b200_bf16_config,
    ],
    ids=lambda fn: fn.__name__,
)
def test_recipe_uses_falcon_energon_encoder(recipe_fn):
    cfg = recipe_fn()
    assert isinstance(cfg.dataset, EnergonDatasetConfig)
    assert isinstance(cfg.dataset.task_encoder, FalconEnergonTaskEncoderConfig)


def test_b200_35b_sft_uses_pp1_ep8_without_full_recompute():
    cfg = b200_recipe_module.falcon_qwen35_35b_a3b_sft_8gpu_b200_bf16_config()
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.expert_model_parallel_size == 8
    assert cfg.model.recompute_granularity is None
    assert cfg.env_vars["CUDA_DEVICE_MAX_CONNECTIONS"] == 32


def test_h100_35b_pretrain_uses_hybridep_and_pretrain_lr():
    cfg = h100_recipe_module.falcon_qwen35_35b_a3b_pretrain_8gpu_h100_bf16_config()
    assert cfg.checkpoint.pretrained_checkpoint is None
    assert cfg.model.moe_token_dispatcher_type == "flex"
    assert cfg.model.moe_flex_dispatcher_backend == "hybridep"
    assert cfg.model.moe_permute_fusion_into_hybridep is False
    assert cfg.model.pipeline_model_parallel_size == 2
    assert cfg.model.expert_model_parallel_size == 4
    assert cfg.model.recompute_granularity == "selective"
    assert cfg.model.recompute_modules == ["core_attn", "gdn_norm_out", "moe_act"]
    assert cfg.optimizer.lr == pytest.approx(3e-4)
    assert cfg.optimizer.use_precision_aware_optimizer is True
    assert cfg.dataset.pad_to_max_length is True
    assert cfg.env_vars["NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN"] == 4


def test_b200_35b_pretrain_uses_pp1_ep8_hybridep_without_recompute():
    cfg = b200_recipe_module.falcon_qwen35_35b_a3b_pretrain_8gpu_b200_bf16_config()
    assert cfg.checkpoint.pretrained_checkpoint is None
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.expert_model_parallel_size == 8
    assert cfg.model.moe_token_dispatcher_type == "flex"
    assert cfg.model.moe_flex_dispatcher_backend == "hybridep"
    assert cfg.model.recompute_granularity is None
    assert cfg.optimizer.lr == pytest.approx(3e-4)
    assert cfg.env_vars["CUDA_DEVICE_MAX_CONNECTIONS"] == 32
    assert cfg.env_vars["NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN"] == 8
