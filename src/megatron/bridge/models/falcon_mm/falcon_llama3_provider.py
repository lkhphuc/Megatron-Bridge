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

"""Model provider for the encoder-free Falcon multimodal model on a Llama3 trunk."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import ClassVar

from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.pipeline_parallel.utils import is_pp_first_stage, is_pp_last_stage
from megatron.core.transformer.spec_utils import ModuleSpec

from megatron.bridge.models.falcon_mm.modelling_falcon_mm.model_llama3 import (
    FalconLlama3Model,
)
from megatron.bridge.models.falcon_mm.rope import FALCON_LLAMA3_IMAGE_TOKEN_ID
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.transformer_config import (
    Qwen3VLTransformerConfig,
)
from megatron.bridge.utils.vocab_utils import calculate_padded_vocab_size


@dataclass
class FalconLlama3ModelProvider(GPTModelProvider):
    """Provider for the encoder-free Falcon multimodal model (dense Llama3 trunk).

    Dense GQA GPT trunk with plain 1D RoPE plus the Falcon early-fusion patch
    embed. Architecture size fields come from the recipe (Llama3-8B defaults
    below); the provider adds the multimodal surface (placeholder token id and
    patch geometry).

    Attributes:
        image_token_id: Placeholder token id whose positions receive the
            patch embeddings (Llama3 ``<|reserved_special_token_0|>``).
        patch_size: Spatial patch edge for the hMLP stem.
        temporal_patch_size: Must be 1; no frame duplication.
        in_channels: Image channels.
        spatial_merge_size: Must be 1; hMLP does its own local merging.
        patch_embed_num_layers: Number of hMLP linear stages.
        patch_embed_hidden_size: Width of each non-final hMLP stage. ``None``
            uses one quarter of the LM width.
    """

    MODEL_CLASS: ClassVar[type[GPTModel] | None] = FalconLlama3Model

    # Falcon multimodal surface
    image_token_id: int = FALCON_LLAMA3_IMAGE_TOKEN_ID
    patch_size: int = 32
    temporal_patch_size: int = 1
    in_channels: int = 3
    spatial_merge_size: int = 1
    patch_embed_num_layers: int = 3
    patch_embed_hidden_size: int | None = None

    # Llama3 dense GQA defaults (8B); recipes may override size fields.
    num_layers: int = 32
    hidden_size: int = 4096
    ffn_hidden_size: int = 14336
    num_attention_heads: int = 32
    num_query_groups: int = 8
    vocab_size: int | None = 128256
    seq_length: int = 4096
    rotary_base: int = 500000
    rotary_percent: float = 1.0
    position_embedding_type: str = "rope"
    normalization: str = "RMSNorm"
    gated_linear_unit: bool = True
    add_bias_linear: bool = False
    add_qkv_bias: bool = False
    share_embeddings_and_output_weights: bool = False
    layernorm_epsilon: float = 1e-5
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    apply_rope_fusion: bool = True
    bias_activation_fusion: bool = True
    masked_softmax_fusion: bool = True
    persist_layer_norm: bool = True
    bias_dropout_fusion: bool = True
    should_pad_vocab: bool = False

    def _build_patch_embed_config(self) -> Qwen3VLTransformerConfig:
        """Build the vision config consumed by the Falcon patch embed."""
        config = Qwen3VLTransformerConfig(
            hidden_size=self.patch_embed_hidden_size
            if self.patch_embed_hidden_size is not None
            else self.hidden_size // 4,
            out_hidden_size=self.hidden_size,
            patch_size=self.patch_size,
            temporal_patch_size=self.temporal_patch_size,
            in_channels=self.in_channels,
            spatial_merge_size=self.spatial_merge_size,
            # The patch embed builds no attention and no transformer layers,
            # but TransformerConfig validation requires positive head/layer
            # counts (kv_channels and init-method scaling divide by them).
            num_attention_heads=1,
            num_layers=1,
        )
        config.tensor_model_parallel_size = self.tensor_model_parallel_size
        config.pipeline_model_parallel_size = 1
        config.first_pipeline_num_layers = None
        config.layernorm_epsilon = 1e-6
        params_dtype = getattr(self, "params_dtype", None)
        if params_dtype is not None:
            config.params_dtype = params_dtype
        return config

    def provide(self, pre_process=None, post_process=None, vp_stage=None) -> FalconLlama3Model:
        """Configure and instantiate the Falcon Llama3 multimodal model.

        Args:
            pre_process: Include embedding + patch embed (first pipeline stage).
            post_process: Include output layer (last pipeline stage).
            vp_stage: Virtual pipeline stage index.

        Returns:
            The composed :class:`FalconLlama3Model`.
        """
        assert self.vocab_size is not None, "vocab_size must be configured before calling provide()"

        transformer_layer_spec = self.transformer_layer_spec
        if not isinstance(transformer_layer_spec, ModuleSpec):
            if "vp_stage" in inspect.signature(transformer_layer_spec).parameters:
                transformer_layer_spec = transformer_layer_spec(self, vp_stage=vp_stage)
            else:
                transformer_layer_spec = transformer_layer_spec(self)

        if self.should_pad_vocab:
            padded_vocab_size = calculate_padded_vocab_size(
                self.vocab_size, self.make_vocab_size_divisible_by, self.tensor_model_parallel_size
            )
        else:
            padded_vocab_size = self.vocab_size

        pre_process = pre_process if pre_process is not None else is_pp_first_stage(self._pg_collection.pp)
        post_process = post_process if post_process is not None else is_pp_last_stage(self._pg_collection.pp)

        return FalconLlama3Model(
            language_transformer_config=self,
            language_transformer_layer_spec=transformer_layer_spec,
            language_vocab_size=padded_vocab_size,
            language_max_sequence_length=self.seq_length,
            patch_embed_config=self._build_patch_embed_config(),
            patch_embed_num_layers=self.patch_embed_num_layers,
            patch_embed_hidden_size=self.patch_embed_hidden_size,
            image_token_id=self.image_token_id,
            parallel_output=self.parallel_output,
            share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
            pre_process=pre_process,
            post_process=post_process,
            pg_collection=self._pg_collection,
            vp_stage=vp_stage,
            rotary_percent=self.rotary_percent,
            rotary_base=self.rotary_base,
            rope_scaling=self.rope_scaling,
            rope_scaling_factor=self.rope_scaling_factor,
            seq_len_interpolation_factor=self.seq_len_interpolation_factor,
        )
