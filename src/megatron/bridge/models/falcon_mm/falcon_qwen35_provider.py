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

"""Model provider for the encoder-free Falcon multimodal model (Qwen3.5 trunk)."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import ClassVar

from megatron.bridge.models.falcon_mm.data.cookers import (
    FALCON_DEFAULT_PATCH_SIZE,
    FALCON_DEFAULT_SPATIAL_MERGE_SIZE,
    FALCON_DEFAULT_TEMPORAL_PATCH_SIZE,
)
from megatron.bridge.models.falcon_mm.modelling_falcon_mm import FalconQwen35Model
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model import Qwen3VLModel
from megatron.bridge.models.qwen_vl.qwen35_vl_provider import (
    Qwen35VLModelProvider,
    Qwen35VLMoEModelProvider,
)


@dataclass
class FalconQwen35ModelProvider(Qwen35VLModelProvider):
    """Provider for the encoder-free Falcon multimodal model (dense Qwen3.5 trunk).

    Identical architecture and config surface to :class:`Qwen35VLModelProvider`
    (hybrid GDN + gated-attention LM trunk, MRoPE); the only difference is that
    ``provide()`` constructs :class:`FalconQwen35Model`, which replaces the ViT
    vision tower with an encoder-free patch embed. The ``vision_config`` field
    is still required: it carries the patch geometry (patch size, temporal
    patch size, channels, spatial merge size) and ``out_hidden_size`` for the
    patch embed.

    Patch geometry defaults match the Falcon Energon data plane
    (``patch_size=32``, ``spatial_merge_size=1``, ``temporal_patch_size=1``),
    not HF Qwen3.5-VL's ViT geometry.
    """

    MODEL_CLASS: ClassVar[type[Qwen3VLModel] | None] = FalconQwen35Model
    patch_size: int = FALCON_DEFAULT_PATCH_SIZE
    spatial_merge_size: int = FALCON_DEFAULT_SPATIAL_MERGE_SIZE
    temporal_patch_size: int = FALCON_DEFAULT_TEMPORAL_PATCH_SIZE
    patch_embed_num_layers: int = 3
    patch_embed_hidden_size: int | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        _align_falcon_patch_geometry(self)


@dataclass
class FalconQwen35MoEModelProvider(Qwen35VLMoEModelProvider):
    """Provider for encoder-free Falcon multimodal MoE (Qwen3.5 MoE trunk)."""

    MODEL_CLASS: ClassVar[type[Qwen3VLModel] | None] = FalconQwen35Model
    patch_size: int = FALCON_DEFAULT_PATCH_SIZE
    spatial_merge_size: int = FALCON_DEFAULT_SPATIAL_MERGE_SIZE
    temporal_patch_size: int = FALCON_DEFAULT_TEMPORAL_PATCH_SIZE
    patch_embed_num_layers: int = 3
    patch_embed_hidden_size: int | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        _align_falcon_patch_geometry(self)


def _align_falcon_patch_geometry(provider: FalconQwen35ModelProvider | FalconQwen35MoEModelProvider) -> None:
    """Align the vision config to the Falcon provider's patch geometry.

    HF Qwen3.5-VL defaults (``patch_size=16``, ``spatial_merge_size=2``,
    ``temporal_patch_size=2``) must not leak into the Falcon patch embed: the
    Energon task encoder materializes ``32×32`` patches with ``merge_size=1``.
    Those values would mismatch the flattened Falcon patches and fail
    geometry validation in :class:`FalconPatchEmbedVisionModel`.
    """
    vision = provider.vision_config
    if vision is None:
        return
    vision.patch_size = provider.patch_size
    vision.spatial_merge_size = provider.spatial_merge_size
    vision.temporal_patch_size = provider.temporal_patch_size
    vision.out_hidden_size = provider.hidden_size
    vision.hmlp_num_layers = provider.patch_embed_num_layers
    vision.hmlp_intermediate_size = provider.patch_embed_hidden_size


def falcon_qwen35_provider_from_qwen35_vl(
    provider: Qwen35VLModelProvider | Qwen35VLMoEModelProvider,
) -> FalconQwen35ModelProvider | FalconQwen35MoEModelProvider:
    """Build a Falcon-MM provider from a Qwen3.5-VL dense or MoE provider.

    Intended for recipes that derive architecture fields from a Hugging Face
    config via ``AutoBridge.from_hf_pretrained(...).to_megatron_provider(
    load_weights=False)``: the returned Qwen3.5-VL provider carries all
    LM/vision architecture fields, which are copied by reference into an
    equivalent Falcon-MM provider. MoE trunks return
    :class:`FalconQwen35MoEModelProvider`.

    Patch geometry (``patch_size``, ``spatial_merge_size``,
    ``temporal_patch_size``) is always reset to Falcon early-fusion defaults
    and written through to ``vision_config``, so the patch embed matches the
    Energon dataloader rather than the HF ViT tower.
    """
    target_cls: type[FalconQwen35ModelProvider | FalconQwen35MoEModelProvider]
    if isinstance(provider, Qwen35VLMoEModelProvider):
        target_cls = FalconQwen35MoEModelProvider
    else:
        target_cls = FalconQwen35ModelProvider
    init_kwargs = {
        field.name: getattr(provider, field.name)
        for field in fields(target_cls)
        if field.init and hasattr(provider, field.name)
    }
    # Drop HF ViT geometry; Falcon dataclass defaults / _align apply instead.
    init_kwargs.pop("patch_size", None)
    init_kwargs.pop("spatial_merge_size", None)
    init_kwargs.pop("temporal_patch_size", None)
    falcon = target_cls(**init_kwargs)
    _align_falcon_patch_geometry(falcon)
    return falcon
