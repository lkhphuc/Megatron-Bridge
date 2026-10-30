"""Offline coverage for the 8-B200 Qwen3-VL 30B native-packing recipe."""

import pytest
from scripts.training.recipe_metadata import recipe_step, resolved_benchmark_recipe_metadata

from megatron.bridge.data.builders import EnergonDatasetConfig, QwenVLEnergonTaskEncoderConfig
from megatron.bridge.recipes.qwen_vl import qwen3_vl_30b_a3b_pretrain_8gpu_b200_fp8mx_energon_config
from tests.unit_tests.recipes.recipe_test_utils import patch_recipe_construction_dependencies


@pytest.mark.unit
def test_qwen3_vl_30b_b200_energon_recipe_keeps_full_training_and_native_thd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The packed entry point stays separate from the mock benchmark and trains all modules."""
    patch_recipe_construction_dependencies(monkeypatch)
    recipe_name = "qwen3_vl_30b_a3b_pretrain_8gpu_b200_fp8mx_energon_config"

    assert resolved_benchmark_recipe_metadata(recipe_name) is None
    assert recipe_step(recipe_name, native_energon_packing=True) == "vlm_step"
    cfg = qwen3_vl_30b_a3b_pretrain_8gpu_b200_fp8mx_energon_config()

    assert isinstance(cfg.dataset, EnergonDatasetConfig)
    assert isinstance(cfg.dataset.task_encoder, QwenVLEnergonTaskEncoderConfig)
    assert cfg.dataset.seq_length == cfg.model.seq_length == 4096
    assert cfg.dataset.micro_batch_size == cfg.train.micro_batch_size == 1
    assert cfg.dataset.packing_buffer_size == 16
    assert cfg.dataset.dataset_kwargs["repeat"] is True
    assert cfg.dataset.pad_to_max_length is True
    assert cfg.train.global_batch_size == 64
    assert cfg.peft is None
    assert cfg.model.freeze_language_model is False
    assert cfg.model.freeze_vision_model is False
    assert cfg.model.freeze_vision_projection is False
    assert cfg.model.calculate_per_token_loss is True
    assert cfg.ddp.average_in_collective is False
    assert cfg.model.cuda_graph_impl == "none"

    cfg.dataset.path = "/tmp/indexed-energon"
    cfg.dataset.validate()
