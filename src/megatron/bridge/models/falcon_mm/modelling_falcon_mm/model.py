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
embedded by a lightweight patch embed (Conv3d patchify + hMLP merger) and
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

from typing import Optional

import torch
from megatron.core.extensions.transformer_engine import (
    TEColumnParallelLinear,
    TENorm,
    TERowParallelLinear,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig

from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model import Qwen3VLModel
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.transformer_config import (
    get_vision_model_config,
)
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.utils import (
    PatchMergerSubmodules,
    Qwen3VLVisionPatchEmbed,
    Qwen3VLVisionPatchMerger,
)


class FalconPatchEmbedVisionModel(MegatronModule):
    """Encoder-free patch embed with the Qwen3-VL vision-tower call contract.

    Pipeline: Conv3d patchify (per-patch projection to the vision hidden size)
    followed by the Qwen3-VL hMLP patch merger (norm -> merge² grouping -> fc1
    -> GELU -> fc2 to the LM hidden size). There is no transformer, no position
    embedding table, and no deepstack — image position information is carried
    exclusively by the LM-side MRoPE.

    Args:
        transformer_config (Qwen3VLTransformerConfig): Megatron vision config
            produced by ``get_vision_model_config`` (carries ``patch_size``,
            ``temporal_patch_size``, ``in_channels``, ``hidden_size``,
            ``spatial_merge_size`` and ``out_hidden_size`` = LM hidden size).
        patch_merger_spec (PatchMergerSubmodules): Spec for the merger's norm
            and linear layers (TE column/row parallel, TP-safe).
        pg_collection (ProcessGroupCollection): Process groups; the merger is
            TP-sharded through ``pg_collection.tp``.
    """

    def __init__(
        self,
        transformer_config: TransformerConfig,
        patch_merger_spec: ModuleSpec,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ) -> None:
        super().__init__(config=transformer_config)
        self.spatial_merge_size = transformer_config.spatial_merge_size
        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        self.pg_collection = pg_collection

        self.patch_embed = Qwen3VLVisionPatchEmbed(transformer_config)
        self.merger = Qwen3VLVisionPatchMerger(
            transformer_config,
            patch_merger_spec,
            use_postshuffle_norm=False,
            tp_group=self.pg_collection.tp,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, list]:
        """Embed pixel patches into LM-width token embeddings.

        Args:
            hidden_states (torch.Tensor): Flattened pixel patches,
                ``[total_patches, in_channels * temporal_patch_size * patch_size ** 2]``.
            grid_thw (torch.Tensor): Per-image/frame grid sizes ``[n_grids, 3]``.

        Returns:
            tuple: ``(embeddings, [])`` where embeddings is
            ``[sum(t*h*w / merge**2), lm_hidden_size]`` in pack order. The
            empty list keeps the deepstack contract (no deepstack features).
        """
        hidden_states = self.patch_embed(hidden_states)
        hidden_states = self.merger(hidden_states)
        # Mirror Qwen3VLVisionModel: regroup per grid and concatenate (the
        # merger output is already contiguous per image; this guards ordering).
        split_sizes = (grid_thw.prod(-1) // self.spatial_merge_size**2).tolist()
        hidden_states = torch.cat(torch.split(hidden_states, split_sizes), dim=0)
        return hidden_states, []


class FalconMMModel(Qwen3VLModel):
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
            patch_merger_spec = PatchMergerSubmodules(
                patch_norm=TENorm,
                linear_fc1=TEColumnParallelLinear,
                linear_fc2=TERowParallelLinear,
            )
            self.vision_model = FalconPatchEmbedVisionModel(
                megatron_vision_config,
                patch_merger_spec,
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

        The patch embed and its merger are a single module, so
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
