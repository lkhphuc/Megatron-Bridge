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

"""Encoder-free Falcon multimodal model on a DeepSeek-V4 HybridModel trunk.

Design (see ``FALCON_MM_AGENT_NOTES/5-deepseek-v4-backbone-plan.md``, A1):
the model composes the trunk-agnostic patch embed
(:class:`FalconPatchEmbedVisionModel`) with a Megatron-Core ``HybridModel``
that runs the DSv4 stack spec (``hybrid_dsv4_stack_spec``). The forward pass
embeds the tokens, scatters the patch embeddings at the placeholder spans,
and calls the language model with ``decoder_input`` — the same composition
pattern as ``modeling_nemotron_omni.NemotronOmniModel``.

Two DSv4-specific notes:

- ``input_ids`` is always forwarded to the language model alongside
  ``decoder_input``: hash-routed MoE layers read token ids for routing
  (``HybridModel.forward`` passes them to the decoder when
  ``moe_n_hash_layers > 0``). The embedding lookup itself is skipped because
  ``decoder_input`` is provided.
- Context parallelism uses the contiguous partition mode (the CSA
  compressor exchanges boundary hidden states between adjacent CP ranks).
  The step function slices this rank's token window and visual payload
  batch-side (see ``vlm_step.get_batch``), so the model consumes rank-local
  tensors and the scatter stays naturally rank-local.
"""

import logging
from typing import Optional

import torch
from megatron.core import tensor_parallel
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig

from megatron.bridge.models.falcon_mm.modelling_falcon_mm.model import (
    FalconPatchEmbedVisionModel,
)
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.utils import (
    PatchMergerSubmodules,
)


logger = logging.getLogger(__name__)


def _ignore_transformer_engine_extra_state(module: torch.nn.Module, incompatible_keys) -> None:
    """Drop TE ``_extra_state`` entries from incompatible checkpoint keys.

    TE modules write an ``_extra_state`` entry into the state dict that older
    checkpoints (and the HF-side export format) do not carry. Same helper as
    ``modeling_nemotron_omni._ignore_transformer_engine_extra_state``.
    """
    del module
    for keys in incompatible_keys._asdict().values():
        for key in keys[::-1]:
            if "extra_state" in key:
                logger.warning("Ignoring Transformer Engine checkpoint key %s", key)
                keys.remove(key)


class FalconDeepseekv4Model(MegatronModule):
    """Encoder-free multimodal model: DSv4 HybridModel trunk + patch embed.

    Constructed like the Nemotron Omni wrapper: the language model is a
    :class:`HybridModel` built with ``scatter_embedding_sequence_parallel``
    disabled so the wrapper owns the sequence-parallel scatter after the
    vision merge. The patch embed lives on the first pipeline stage only.

    Args:
        language_transformer_config (TransformerConfig): DSv4 trunk config
            (the provider instance; carries MLA/CSA/MoE/mHC fields).
        language_transformer_layer_spec (ModuleSpec): Resolved
            ``hybrid_dsv4_stack_spec`` module spec.
        language_vocab_size (int): Padded vocabulary size.
        language_max_sequence_length (int): Maximum sequence length.
        patch_embed_config (TransformerConfig): Vision config for the patch
            embed (patch geometry, vision hidden size, ``out_hidden_size``
            equal to the LM hidden size).
        patch_merger_spec (PatchMergerSubmodules): Spec for the merger norm
            and linear layers (TP-safe TE modules).
        image_token_id (int): Placeholder token id whose embedding positions
            receive the patch embeddings (DSv4 reserved token 128000).
        parallel_output (bool): Keep output logits split across TP ranks.
        share_embeddings_and_output_weights (bool): Tie input/output
            embeddings.
        pre_process (bool): Include embedding + patch embed (first pipeline
            stage).
        post_process (bool): Include output layer (last pipeline stage).
        hybrid_layer_pattern (str | None): DSv4 hybrid layer pattern, with
            ``|`` pipeline stage boundaries when PP > 1.
        pg_collection (ProcessGroupCollection | None): Process groups.
        vp_stage (int | None): Virtual pipeline stage index.
    """

    model_owns_packing = False
    model_owns_mtp_loss_mask_packing = False
    # The step slices contiguous CP inputs batch-side (informational flag;
    # nothing in Bridge reads it at this time).
    model_slices_context_parallel_inputs = False

    def __init__(
        self,
        *,
        language_transformer_config: TransformerConfig,
        language_transformer_layer_spec: ModuleSpec,
        language_vocab_size: int,
        language_max_sequence_length: int,
        patch_embed_config: TransformerConfig,
        patch_merger_spec: PatchMergerSubmodules,
        image_token_id: int,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
        pre_process: bool = True,
        post_process: bool = True,
        hybrid_layer_pattern: Optional[str] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        vp_stage: Optional[int] = None,
    ) -> None:
        super().__init__(config=language_transformer_config)
        self.pre_process = pre_process
        self.post_process = post_process
        self.image_token_id = image_token_id
        self.sequence_parallel_lm = language_transformer_config.sequence_parallel

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        self.pg_collection = pg_collection

        self.language_model = HybridModel(
            config=language_transformer_config,
            hybrid_stack_spec=language_transformer_layer_spec,
            vocab_size=language_vocab_size,
            max_sequence_length=language_max_sequence_length,
            hybrid_layer_pattern=hybrid_layer_pattern,
            parallel_output=parallel_output,
            position_embedding_type="rope",
            scatter_embedding_sequence_parallel=False,
            share_embeddings_and_output_weights=share_embeddings_and_output_weights,
            pre_process=pre_process,
            post_process=post_process,
            pg_collection=pg_collection,
            vp_stage=vp_stage,
        )

        self.vision_model = None
        if pre_process:
            self.vision_model = FalconPatchEmbedVisionModel(
                patch_embed_config,
                patch_merger_spec,
                pg_collection=self.pg_collection,
            )
            self.vision_model.register_load_state_dict_post_hook(_ignore_transformer_engine_extra_state)
        self.language_model.register_load_state_dict_post_hook(_ignore_transformer_engine_extra_state)

    def set_input_tensor(self, input_tensor) -> None:
        """Set input tensor for pipeline parallelism (non-first stages)."""
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        assert len(input_tensor) == 1, "input_tensor should only be length 1 for FalconDeepseekv4"
        if not self.pre_process:
            self.language_model.set_input_tensor(input_tensor[0])

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
        """Forward function of the Falcon DeepSeek-V4 model.

        On the first pipeline stage: look up token embeddings, embed the
        pixel patches, and scatter the patch embeddings into the placeholder
        spans (``input_ids == image_token_id``). Then run the DSv4 trunk with
        the merged embeddings as ``decoder_input``. Positions are ordinary
        1D rope positions handled inside the DSv4 attention modules from the
        packed metadata — there is no MRoPE on this trunk.

        Returns the per-token loss when ``labels`` are given, otherwise the
        final hidden states or logits, matching the ``HybridModel`` contract.
        """
        if pixel_values_videos is not None or video_grid_thw is not None:
            raise NotImplementedError("FalconDeepseekv4Model does not support video inputs yet.")

        combined_embeddings = None
        if self.pre_process:
            if input_ids is None:
                raise ValueError("The first FalconDeepseekv4 pipeline stage requires input_ids.")

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

        # input_ids rides along for hash-routed MoE layers, which route by
        # token id. The embedding lookup is skipped because decoder_input is
        # provided (see HybridModel.forward).
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
