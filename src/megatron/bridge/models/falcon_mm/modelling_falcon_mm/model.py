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

"""Encoder-free Falcon multimodal model.

Design (see ``natively-multimodal-300b-feasibility.md``, D1/R6): patches are
embedded by a hierarchical MLP stem and
scattered into the LM trunk at the processor-expanded placeholder spans — the
same length-preserving contract every Bridge VLM uses, so the THD/Energon data
plane, MRoPE x THD x CP machinery, and per-stage packed-metadata handling are
inherited unchanged from :class:`Qwen3VLModel`.

The patch embed satisfies the exact ``vision_model`` call contract of
``Qwen3VLModel`` — ``vision_model(hidden_states=pixel_patches,
grid_thw=grid_thw) -> (merged_embeddings, deepstack_feature_lists)`` — so no
forward-path changes are needed. Because the embed is per-token independent,
no cross-CP-rank vision all-gather is required (unlike the ViT path); the
existing ``vision_dp_when_cp`` machinery is simply never needed.
"""

import torch
from einops import rearrange
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from torch import nn

from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model import Qwen3VLModel
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.transformer_config import (
    get_vision_model_config,
)


class FalconPatchEmbedVisionModel(MegatronModule):
    """Independent per-patch hMLP stem with the Qwen3-VL vision call contract.

    The starting cell edge is ``patch_size // 2**num_layers``. Each linear
    stage merges 2×2 neighboring cells, so 32×32 with three stages grows
    4→8→16→32, while 28×28 with two stages grows 7→14→28. The last
    linear stage produces the exact LM width. LayerNorm keeps each patch
    independent even in a packed batch. There is no transformer, positional
    embedding, or deepstack.

    Args:
        transformer_config: Vision geometry and LM output width.
        num_layers: Number of linear 2×2 merge stages.
        intermediate_size: Width of each non-final linear stage. Defaults to
            one quarter of the LM width, rounded down.
        pg_collection: Kept for the common Falcon model construction contract.
    """

    def __init__(
        self,
        transformer_config: TransformerConfig,
        *,
        num_layers: int = 3,
        intermediate_size: int | None = None,
        pg_collection: ProcessGroupCollection | None = None,
    ) -> None:
        super().__init__(config=transformer_config)
        del pg_collection
        self.patch_size = transformer_config.patch_size
        self.in_channels = transformer_config.in_channels
        self.out_hidden_size = transformer_config.out_hidden_size
        self.num_layers = num_layers
        if transformer_config.temporal_patch_size != 1 or transformer_config.spatial_merge_size != 1:
            raise ValueError("Falcon hMLP requires temporal_patch_size=1 and spatial_merge_size=1")
        if num_layers < 1 or self.patch_size < 1 or self.patch_size % (2**num_layers):
            raise ValueError("Falcon hMLP patch_size must be divisible by 2**num_layers")
        if self.out_hidden_size < 1:
            raise ValueError("Falcon hMLP out_hidden_size must be positive")
        if intermediate_size is None:
            intermediate_size = max(1, self.out_hidden_size // 4)
        if intermediate_size < 1:
            raise ValueError("Falcon hMLP intermediate_size must be positive")

        self.start_patch_size = self.patch_size // 2**num_layers
        self.intermediate_size = intermediate_size
        widths = [intermediate_size] * (num_layers - 1) + [self.out_hidden_size]
        input_width = self.in_channels * (2 * self.start_patch_size) ** 2
        self.stages = nn.ModuleList()
        for stage_index, output_width in enumerate(widths):
            stage = nn.Sequential(
                nn.Linear(input_width, output_width),
                nn.LayerNorm(output_width, eps=transformer_config.layernorm_epsilon),
            )
            if stage_index < num_layers - 1:
                stage.append(nn.GELU())
            self.stages.append(stage)
            input_width = 4 * output_width

    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, list]:
        """Embed pixel patches into LM-width token embeddings.

        Args:
            hidden_states: Flattened channel-first pixel patches,
                ``[n, channels * 1 * patch_size ** 2]``.
            grid_thw: Per-image grid sizes ``[n_grids, 3]``.

        Returns:
            ``(embeddings, [])`` with one LM-width embedding per input patch.
        """
        del kwargs
        if hidden_states.ndim != 2 or hidden_states.shape[1] != self.in_channels * self.patch_size**2:
            raise ValueError("Falcon hMLP expects flattened channel-first patches of the configured size")
        if grid_thw.ndim != 2 or grid_thw.shape[1] != 3 or int(grid_thw.prod(-1).sum()) != hidden_states.shape[0]:
            raise ValueError("grid_thw must describe exactly one entry per input patch")
        if bool(torch.any(grid_thw[:, 0] != 1)):
            raise ValueError("Falcon hMLP requires T=1 for every image grid")

        # n (c h w) -> n h w c; the Qwen-compatible payload has T=1 and C,H,W flattening.
        patches = rearrange(hidden_states, "n (c h w) -> n h w c", c=self.in_channels, h=self.patch_size)
        patches = patches.to(dtype=self.stages[0][0].weight.dtype)
        first_merge_size = 2 * self.start_patch_size
        # n (h ph) (w pw) c -> (n h w) (ph pw c), with ph=pw=2*start_patch_size.
        cells = rearrange(patches, "n (h ph) (w pw) c -> (n h w) (ph pw c)", ph=first_merge_size, pw=first_merge_size)
        cells = self.stages[0](cells)
        cell_grid = self.patch_size // first_merge_size
        # (n h w) d -> n h w d, restoring the within-patch cell grid.
        cells = rearrange(cells, "(n h w) d -> n h w d", h=cell_grid, w=cell_grid)
        for stage_index in range(1, len(self.stages)):
            # n (h ph) (w pw) d -> (n h w) (ph pw d), with ph=pw=2.
            cells = rearrange(cells, "n (h ph) (w pw) d -> (n h w) (ph pw d)", ph=2, pw=2)
            cells = self.stages[stage_index](cells)
            cell_grid //= 2
            # (n h w) d -> n h w d, preserving each local cell grid.
            cells = rearrange(cells, "(n h w) d -> n h w d", h=cell_grid, w=cell_grid)
        # n 1 1 d -> n d, preserving the input patch order across image grids.
        return rearrange(cells, "n 1 1 d -> n d"), []


class FalconQwen35Model(Qwen3VLModel):
    """Encoder-free multimodal model: Qwen3-VL LM trunk + patch embed.

    Constructed exactly like :class:`Qwen3VLModel` except the ViT vision tower
    is replaced by :class:`FalconPatchEmbedVisionModel`. All multimodal
    machinery (placeholder-span scatter, MRoPE recompute per PP stage, packed
    THD handling, CP slicing) is inherited unchanged.
    """

    def __init__(self, *args, **kwargs) -> None:
        # Skip the ViT construction, then install the patch embed in its place.
        kwargs["add_encoder"] = False
        super().__init__(*args, **kwargs)

        if self.pre_process:
            megatron_vision_config = get_vision_model_config(
                self.vision_transformer_config, megatron_config=self.config
            )
            megatron_vision_config.pipeline_model_parallel_size = 1
            megatron_vision_config.first_pipeline_num_layers = None
            self.vision_model = FalconPatchEmbedVisionModel(
                megatron_vision_config,
                num_layers=getattr(self.vision_transformer_config, "hmlp_num_layers", 3),
                intermediate_size=getattr(self.vision_transformer_config, "hmlp_intermediate_size", None),
                pg_collection=self.pg_collection,
            )
        # Forward resolves the vision tower through ``self.vision_model``; the
        # flag only gates ViT construction in the base class.
        self.add_encoder = True

    def freeze(
        self,
        freeze_language_model: bool,
        freeze_vision_model: bool,
        freeze_vision_projection: bool,
    ) -> None:
        """Freeze model modules.

        The hMLP patch embed is a single module, so
        ``freeze_vision_projection`` implies ``freeze_vision_model`` here.
        """
        modules = []
        if freeze_language_model and self.language_model is not None:
            modules.append(self.language_model)
        if (freeze_vision_model or freeze_vision_projection) and self.vision_model is not None:
            modules.append(self.vision_model)
        for module in modules:
            for param in module.parameters():
                param.requires_grad = False
