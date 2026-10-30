"""Unit tests for the Falcon early-fusion hMLP patch embedder."""

import pytest
import torch
from torch import nn

from megatron.bridge.models.falcon_mm.modelling_falcon_mm.model import FalconPatchEmbedVisionModel
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.transformer_config import Qwen3VLTransformerConfig


def _config(
    *,
    patch_size: int = 32,
    out_hidden_size: int = 64,
    temporal_patch_size: int = 1,
    spatial_merge_size: int = 1,
):
    return Qwen3VLTransformerConfig(
        num_layers=1,
        num_attention_heads=4,
        hidden_size=16,
        out_hidden_size=out_hidden_size,
        in_channels=3,
        patch_size=patch_size,
        temporal_patch_size=temporal_patch_size,
        spatial_merge_size=spatial_merge_size,
    )


@pytest.mark.unit
def test_hmlp_has_three_linear_stages_for_32_pixel_patches():
    model = FalconPatchEmbedVisionModel(_config(), num_layers=3)

    assert [tuple(layer.weight.shape) for layer in model.modules() if isinstance(layer, nn.Linear)] == [
        (16, 3 * 8 * 8),
        (16, 4 * 16),
        (64, 4 * 16),
    ]
    assert not any(isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d)) for layer in model.modules())


@pytest.mark.unit
def test_hmlp_embeds_each_patch_independently_and_preserves_pack_order():
    torch.manual_seed(13)
    model = FalconPatchEmbedVisionModel(_config(), num_layers=3)
    # n (c t h w) -> n d, with t=1 and h=w=32.
    patches = torch.randn(3, 3 * 32 * 32)
    grid_thw = torch.tensor([[1, 1, 2], [1, 1, 1]])

    actual, deepstack = model(hidden_states=patches, grid_thw=grid_thw)
    separately = torch.cat(
        [model(hidden_states=patches[index : index + 1], grid_thw=torch.tensor([[1, 1, 1]]))[0] for index in range(3)],
        dim=0,
    )

    assert actual.shape == (3, 64)
    torch.testing.assert_close(actual, separately)
    assert deepstack == []


@pytest.mark.unit
def test_hmlp_uses_spatially_adjacent_cells_for_each_merge():
    model = FalconPatchEmbedVisionModel(_config(), num_layers=3)
    for stage_index, stage in enumerate(model.stages):
        stage[1] = nn.Identity()
        if len(stage) == 3:
            stage[2] = nn.Identity()
        with torch.no_grad():
            stage[0].weight.zero_()
            stage[0].bias.zero_()
            if stage_index < 2:
                stage[0].weight[0, 0] = 1
            else:
                for quadrant in range(4):
                    stage[0].weight[quadrant, quadrant * 16] = 1

    # n (c h w) -> n d: the first four features retain each quadrant's top-left pixel.
    channel = torch.arange(32 * 32, dtype=torch.float32)
    patch = torch.stack((channel, torch.zeros_like(channel), torch.zeros_like(channel))).reshape(1, -1)
    embedded, _ = model(patch, torch.tensor([[1, 1, 1]]))

    expected = torch.zeros(64)
    expected[:4] = torch.tensor([0.0, 16.0, 512.0, 528.0])
    torch.testing.assert_close(embedded[0], expected)


@pytest.mark.unit
@pytest.mark.parametrize("patch_size,num_layers,out_hidden_size", [(16, 2, 64), (28, 2, 65), (32, 3, 65)])
def test_hmlp_supports_matching_patch_size_and_merge_depth(patch_size: int, num_layers: int, out_hidden_size: int):
    model = FalconPatchEmbedVisionModel(
        _config(patch_size=patch_size, out_hidden_size=out_hidden_size), num_layers=num_layers
    )
    patches = torch.randn(2, 3 * patch_size * patch_size)

    actual, _ = model(patches, torch.tensor([[1, 1, 2]]))

    assert actual.shape == (2, out_hidden_size)


@pytest.mark.unit
def test_hmlp_28_pixel_patch_starts_from_7_pixel_cells():
    model = FalconPatchEmbedVisionModel(_config(patch_size=28, out_hidden_size=65), num_layers=2)

    assert [tuple(layer.weight.shape) for layer in model.modules() if isinstance(layer, nn.Linear)] == [
        (16, 3 * 14 * 14),
        (65, 4 * 16),
    ]


@pytest.mark.unit
def test_hmlp_intermediate_width_is_configurable_independently_of_lm_width():
    model = FalconPatchEmbedVisionModel(_config(patch_size=28, out_hidden_size=65), num_layers=2, intermediate_size=11)

    assert [tuple(layer.weight.shape) for layer in model.modules() if isinstance(layer, nn.Linear)] == [
        (11, 3 * 14 * 14),
        (65, 4 * 11),
    ]
    output, _ = model(torch.randn(1, 3 * 28 * 28), torch.tensor([[1, 1, 1]]))
    assert output.shape == (1, 65)


@pytest.mark.unit
@pytest.mark.parametrize(
    "config,num_layers,intermediate_size",
    [
        (_config(temporal_patch_size=2), 3, None),
        (_config(spatial_merge_size=2), 3, None),
        (_config(patch_size=28), 3, None),
        (_config(), 0, None),
        (_config(), 3, 0),
    ],
)
def test_hmlp_rejects_incompatible_geometry(
    config: Qwen3VLTransformerConfig, num_layers: int, intermediate_size: int | None
):
    with pytest.raises(ValueError):
        FalconPatchEmbedVisionModel(config, num_layers=num_layers, intermediate_size=intermediate_size)
