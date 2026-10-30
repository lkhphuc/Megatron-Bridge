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

"""Encoder-free Falcon multimodal model on a dense Llama3 GPT trunk.

Composes the trunk-agnostic patch embed (:class:`FalconPatchEmbedVisionModel`)
with a Megatron-Core ``GPTModel`` (dense GQA, plain 1D RoPE). The forward pass
embeds tokens, scatters patch embeddings at the placeholder spans, and calls
the language model with ``decoder_input`` — the same composition pattern as
:class:`FalconDeepseekv4Model`, without mRoPE or hybrid/MoE routing.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from megatron.core import tensor_parallel
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig

from megatron.bridge.models.falcon_mm.modelling_falcon_mm.model import (
    FalconPatchEmbedVisionModel,
)
from megatron.bridge.models.falcon_mm.rope import FALCON_LLAMA3_IMAGE_TOKEN_ID


logger = logging.getLogger(__name__)


def _ignore_transformer_engine_extra_state(module: torch.nn.Module, incompatible_keys) -> None:
    """Drop TE ``_extra_state`` entries from incompatible checkpoint keys."""
    del module
    for keys in incompatible_keys._asdict().values():
        for key in keys[::-1]:
            if "extra_state" in key:
                logger.warning("Ignoring Transformer Engine checkpoint key %s", key)
                keys.remove(key)


class FalconLlama3Model(MegatronModule):
    """Encoder-free multimodal model: dense Llama3 GPT trunk + patch embed.

    The language model is a :class:`GPTModel` built with
    ``scatter_embedding_sequence_parallel`` disabled so this wrapper owns the
    sequence-parallel scatter after the vision merge. The patch embed lives on
    the first pipeline stage only. Position encoding stays plain 1D RoPE.

    Args:
        language_transformer_config: Llama trunk config (the provider instance).
        language_transformer_layer_spec: Resolved GPT decoder layer spec.
        language_vocab_size: Padded vocabulary size.
        language_max_sequence_length: Maximum sequence length.
        patch_embed_config: Vision config for the patch embed.
        patch_embed_num_layers: Number of 2× hMLP merge stages.
        patch_embed_hidden_size: Width of the non-final hMLP stages.
        image_token_id: Placeholder token id whose positions receive patch
            embeddings (Llama3 reserved special token 128002 by default).
        parallel_output: Keep output logits split across TP ranks.
        share_embeddings_and_output_weights: Tie input/output embeddings.
        pre_process: Include embedding + patch embed (first pipeline stage).
        post_process: Include output layer (last pipeline stage).
        pg_collection: Process groups.
        vp_stage: Virtual pipeline stage index.
        rotary_percent: Fraction of hidden dims used for RoPE.
        rotary_base: RoPE base frequency (Llama3 uses 500000).
        rope_scaling: Whether Llama3-style RoPE scaling is enabled.
        rope_scaling_factor: RoPE scaling factor when enabled.
        seq_len_interpolation_factor: Optional linear RoPE interpolation.
    """

    model_owns_packing = False
    model_owns_mtp_loss_mask_packing = False
    model_slices_context_parallel_inputs = False

    def __init__(
        self,
        *,
        language_transformer_config: TransformerConfig,
        language_transformer_layer_spec: ModuleSpec,
        language_vocab_size: int,
        language_max_sequence_length: int,
        patch_embed_config: TransformerConfig,
        patch_embed_num_layers: int = 3,
        patch_embed_hidden_size: int | None = None,
        image_token_id: int = FALCON_LLAMA3_IMAGE_TOKEN_ID,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
        pre_process: bool = True,
        post_process: bool = True,
        pg_collection: Optional[ProcessGroupCollection] = None,
        vp_stage: Optional[int] = None,
        rotary_percent: float = 1.0,
        rotary_base: int = 500000,
        rope_scaling: bool = False,
        rope_scaling_factor: float = 1.0,
        seq_len_interpolation_factor: Optional[float] = None,
    ) -> None:
        super().__init__(config=language_transformer_config)
        self.pre_process = pre_process
        self.post_process = post_process
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights
        self.image_token_id = image_token_id
        self.sequence_parallel_lm = language_transformer_config.sequence_parallel

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        self.pg_collection = pg_collection

        self.language_model = GPTModel(
            config=language_transformer_config,
            transformer_layer_spec=language_transformer_layer_spec,
            vocab_size=language_vocab_size,
            max_sequence_length=language_max_sequence_length,
            parallel_output=parallel_output,
            share_embeddings_and_output_weights=share_embeddings_and_output_weights,
            position_embedding_type="rope",
            rotary_percent=rotary_percent,
            rotary_base=rotary_base,
            rope_scaling=rope_scaling,
            rope_scaling_factor=rope_scaling_factor,
            seq_len_interpolation_factor=seq_len_interpolation_factor,
            scatter_embedding_sequence_parallel=False,
            pre_process=pre_process,
            post_process=post_process,
            pg_collection=pg_collection,
            vp_stage=vp_stage,
        )

        self.vision_model = None
        if pre_process:
            self.vision_model = FalconPatchEmbedVisionModel(
                patch_embed_config,
                num_layers=patch_embed_num_layers,
                intermediate_size=patch_embed_hidden_size,
                pg_collection=self.pg_collection,
            )
            self.vision_model.register_load_state_dict_post_hook(_ignore_transformer_engine_extra_state)
        self.language_model.register_load_state_dict_post_hook(_ignore_transformer_engine_extra_state)

    def shared_embedding_or_output_weight(self) -> torch.nn.Parameter | None:
        """Return the stage-local shared embedding or output weight."""
        return self.language_model.shared_embedding_or_output_weight()

    @property
    def embedding_activation_buffer(self) -> list[torch.Tensor]:
        """Expose the inner output-layer activation buffer to PP schedules."""
        return self.language_model.embedding_activation_buffer

    @property
    def grad_output_buffer(self) -> list[torch.Tensor]:
        """Expose the inner output-layer gradient buffer to PP schedules."""
        return self.language_model.grad_output_buffer

    @property
    def output_layer(self) -> torch.nn.Module:
        """Expose the inner output layer used by deferred embedding wgrad."""
        return self.language_model.output_layer

    def set_input_tensor(self, input_tensor) -> None:
        """Set input tensor for pipeline parallelism (non-first stages)."""
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        assert len(input_tensor) == 1, "input_tensor should only be length 1 for FalconLlama3"
        if not self.pre_process:
            self.language_model.set_input_tensor(input_tensor[0])

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

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
        inference_context: Optional[object] = None,
        runtime_gather_output: Optional[bool] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward: embed, scatter patch embeds, run the Llama3 GPT trunk.

        On the first pipeline stage: look up token embeddings, embed the pixel
        patches, and scatter the patch embeddings into the placeholder spans
        (``input_ids == image_token_id``). Then run the GPT trunk with the
        merged embeddings as ``decoder_input``. Uses plain 1D RoPE only.
        """
        del kwargs
        if pixel_values_videos is not None or video_grid_thw is not None:
            raise NotImplementedError("FalconLlama3Model does not support video inputs yet.")

        combined_embeddings = None
        if self.pre_process:
            if input_ids is None:
                raise ValueError("The first FalconLlama3 pipeline stage requires input_ids.")

            vision_embeds = None
            if pixel_values is not None and pixel_values.numel() > 0:
                if image_grid_thw is None:
                    raise ValueError("image_grid_thw is required when pixel_values are provided.")
                vision_embeds, _ = self.vision_model(hidden_states=pixel_values, grid_thw=image_grid_thw)

            # Mask placeholder ids out of the embedding lookup; their rows are
            # overwritten by the scatter below.
            text_ids = input_ids.masked_fill(input_ids == self.image_token_id, 0)
            combined_embeddings = self.language_model.embedding(
                input_ids=text_ids,
                position_ids=None,
            ).clone()  # [s, b, h]

            combined_embeddings = combined_embeddings.transpose(0, 1).contiguous()  # [b, s, h]
            if vision_embeds is not None:
                vision_mask = input_ids == self.image_token_id
                combined_embeddings[vision_mask] = vision_embeds
            combined_embeddings = combined_embeddings.transpose(0, 1).contiguous()  # [s, b, h]

            if self.sequence_parallel_lm:
                combined_embeddings = tensor_parallel.scatter_to_sequence_parallel_region(
                    combined_embeddings,
                    group=self.pg_collection.tp,
                ).contiguous()

        if packed_seq_params is not None:
            # THD packs carry their causal structure in cu_seqlens.
            attention_mask = None

        return self.language_model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            decoder_input=combined_embeddings,
            labels=labels,
            loss_mask=loss_mask,
            inference_context=inference_context,
            runtime_gather_output=runtime_gather_output,
            packed_seq_params=packed_seq_params,
        )
