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

"""Model provider for the encoder-free Falcon multimodal model."""

from dataclasses import dataclass, fields
from typing import ClassVar

from megatron.bridge.models.falcon_mm.modelling_falcon_mm import FalconMMModel
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model import Qwen3VLModel
from megatron.bridge.models.qwen_vl.qwen35_vl_provider import Qwen35VLModelProvider


@dataclass
class FalconMMModelProvider(Qwen35VLModelProvider):
    """Provider for the encoder-free Falcon multimodal model.

    Identical architecture and config surface to :class:`Qwen35VLModelProvider`
    (hybrid GDN + gated-attention LM trunk, MRoPE); the only difference is that
    ``provide()`` constructs :class:`FalconMMModel`, which replaces the ViT
    vision tower with an encoder-free patch embed. The ``vision_config`` field
    is still required: it carries the patch geometry (patch size, temporal
    patch size, channels, spatial merge size) and ``out_hidden_size`` for the
    patch embed.
    """

    MODEL_CLASS: ClassVar[type[Qwen3VLModel] | None] = FalconMMModel

    @classmethod
    def from_qwen35_vl_provider(cls, provider: Qwen35VLModelProvider) -> "FalconMMModelProvider":
        """Build a Falcon-MM provider from a Qwen3.5-VL provider's fields.

        Intended for recipes that derive architecture fields from a Hugging
        Face config via ``AutoBridge.from_hf_pretrained(...).to_megatron_provider(
        load_weights=False)``: the returned Qwen3.5-VL provider carries all
        LM/vision architecture fields, which are copied by reference into an
        equivalent Falcon-MM provider.
        """
        init_kwargs = {
            field.name: getattr(provider, field.name)
            for field in fields(cls)
            if field.init and hasattr(provider, field.name)
        }
        return cls(**init_kwargs)
