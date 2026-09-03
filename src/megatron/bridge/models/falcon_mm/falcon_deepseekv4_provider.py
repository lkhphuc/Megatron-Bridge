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

"""Model provider for the encoder-free Falcon multimodal model on a DSv4 trunk."""

from dataclasses import dataclass
from typing import Callable, ClassVar, Optional, Union

import torch
from megatron.core.extensions.transformer_engine import (
    TEColumnParallelLinear,
    TENorm,
    TERowParallelLinear,
)
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_dsv4_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.pipeline_parallel.utils import is_pp_first_stage, is_pp_last_stage
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import MLATransformerConfig

from megatron.bridge.models.falcon_mm.falcon_deepseekv4_pattern import (
    falcon_deepseekv4_layer_pattern,
)
from megatron.bridge.models.falcon_mm.modelling_falcon_mm.model_deepseekv4 import (
    FalconDeepseekv4Model,
)
from megatron.bridge.models.hybrid.hybrid_provider import HybridModelProvider
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.transformer_config import (
    Qwen3VLTransformerConfig,
)
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.utils import (
    PatchMergerSubmodules,
)
from megatron.bridge.utils.vocab_utils import calculate_padded_vocab_size


@dataclass
class FalconDeepseekv4ModelProvider(MLATransformerConfig, HybridModelProvider):
    """Provider for the encoder-free Falcon multimodal model (DSv4 trunk).

    Combines :class:`MLATransformerConfig` (the MLA fields ``q_lora_rank``,
    ``v_head_dim``, ``qk_pos_emb_head_dim``, ``o_groups``, ``o_lora_rank``,
    plus the DSv4 ``qk_head_dim`` / ``kv_lora_rank`` derivation) with
    :class:`HybridModelProvider` (HybridModel construction), mirroring how
    :class:`MLAModelProvider` combines MLA fields with the GPT provider.

    The trunk is a Megatron-Core ``HybridModel`` built with
    ``hybrid_dsv4_stack_spec`` (MLA + CSA/HCA attention, hash-routed MoE,
    hyper-connections). The provider adds the Falcon-specific surface on
    top: the placeholder token id and the patch-embed geometry. DSv4
    architecture constants (attention variant, MLA, gated MoE) are defaulted
    here; size fields (hidden size, layer count, experts) come from the
    recipe or, later, from the HF config through the DSv4 bridge.

    The placeholder defaults to the DSv4 reserved special token
    ``<｜place▁holder▁no▁0｜>`` (id 128000), which exists in the released
    129280-token vocabulary, so no vocab extension is needed. The data plane
    brackets each visual span with ``<｜place▁holder▁no▁1｜>`` /
    ``<｜place▁holder▁no▁2｜>`` as vision start/end markers — see
    ``data/visual_markers.py`` for the shared marker contract; this id is the
    scatter target (``image_pad``) and must stay aligned with the resolved
    ``image_pad_id`` there.

    Attributes:
        image_token_id: Placeholder token id whose positions receive the
            patch embeddings.
        patch_size: Spatial patch edge for the Conv3d patchify.
        temporal_patch_size: Temporal patch size for Conv3d patchify
            (Falcon early-fusion default is 1; no frame duplication).
        in_channels: Image channels.
        spatial_merge_size: Spatial merge factor of the patch merger; one
            visual token covers merge² patches.
        patch_embed_hidden_size: Width of the patch embed before the merger.
    """

    MODEL_CLASS: ClassVar[type[HybridModel] | None] = FalconDeepseekv4Model

    # Falcon multimodal surface
    image_token_id: int = 128000
    patch_size: int = 32
    temporal_patch_size: int = 1
    in_channels: int = 3
    spatial_merge_size: int = 1
    patch_embed_hidden_size: int = 1024

    # DSv4 architecture constants (mirror deepseek_v4_bridge.provider_bridge)
    experimental_attention_variant: Optional[str] = "dsv4_hybrid"
    multi_latent_attention: bool = True
    qk_layernorm: bool = True
    normalization: str = "RMSNorm"
    add_bias_linear: bool = False
    rotary_percent: float = 1.0
    position_embedding_type: str = "rope"
    gated_linear_unit: bool = True
    moe_grouped_gemm: bool = True
    moe_router_pre_softmax: bool = False

    # DSv4 stack spec (config-aware callable; resolved with this provider)
    hybrid_stack_spec: Union[ModuleSpec, Callable, None] = hybrid_dsv4_stack_spec

    def _build_patch_embed_config(self) -> Qwen3VLTransformerConfig:
        """Build the vision config consumed by the Falcon patch embed."""
        config = Qwen3VLTransformerConfig(
            hidden_size=self.patch_embed_hidden_size,
            out_hidden_size=self.hidden_size,
            patch_size=self.patch_size,
            temporal_patch_size=self.temporal_patch_size,
            in_channels=self.in_channels,
            spatial_merge_size=self.spatial_merge_size,
            # The patch embed builds no attention and no transformer layers,
            # but TransformerConfig validation requires a positive head count
            # (kv_channels derivation divides by it) and a positive layer
            # count (init-method scaling divides by it). Both default to 0.
            num_attention_heads=16,
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

    def _fix_hash_layer_numbering(self, model: FalconDeepseekv4Model) -> None:
        """Re-mark hash-routed MoE layers per HF decoder-layer index.

        HybridStack assigns one ``layer_number`` per pattern symbol, so the
        MoE sublayer of HF decoder layer ``n`` sits at hybrid index
        ``2n + 1`` with ``layer_number = 2n + 2``. Core's hash check
        (``layer_number <= moe_n_hash_layers``) then marks the wrong layers:
        the split attention/MLP pattern shifts every MoE layer out of the
        hash range. The hybrid DSv4 path has no upstream hash coverage (no
        hybrid DSv4 config sets hash layers at this time), so correct the
        flags here: enable hash routing with a ``tid2eid`` buffer on the
        first ``moe_n_hash_layers`` decoder layers, and disable the expert
        bias there (matching core's hash-layer router shape). The buffer is
        initialized round-robin as a placeholder; the bridge overwrites it
        with the checkpoint table on import. Tracked in
        ``FALCON_MM_AGENT_NOTES/5-deepseek-v4-backbone-plan.md``.
        """
        num_hash_layers = self.moe_n_hash_layers or 0
        if num_hash_layers <= 0:
            return
        for hf_index in range(num_hash_layers):
            hybrid_index = 2 * hf_index + 1
            router = model.language_model.decoder.layers[hybrid_index].inner_layer.mlp.router
            if router.is_hash_layer and router.tid2eid is not None:
                continue
            router.is_hash_layer = True
            router.enable_expert_bias = False
            router.expert_bias = None
            if router.tid2eid is None:
                # The non-hash branch sets a plain ``tid2eid = None`` attribute,
                # which blocks register_buffer under the same name.
                if "tid2eid" in router.__dict__:
                    del router.tid2eid
                ids = torch.arange(self.actual_vocab_size, device=router.weight.device)
                tid2eid = torch.stack(
                    [(ids + k) % self.num_moe_experts for k in range(self.moe_router_topk)], dim=1
                ).to(torch.int32)
                router.register_buffer("tid2eid", tid2eid)

    def provide(self, pre_process=None, post_process=None, vp_stage=None) -> FalconDeepseekv4Model:
        """Configure and instantiate the Falcon DSv4 multimodal model.

        Args:
            pre_process: Include embedding + patch embed (first pipeline stage).
            post_process: Include output layer (last pipeline stage).
            vp_stage: Virtual pipeline stage. Must be None: HybridModel does
                not support virtual pipelining in Bridge at this time.

        Returns:
            The composed :class:`FalconDeepseekv4Model`.
        """
        assert getattr(self, "virtual_pipeline_model_parallel_size", None) is None and vp_stage is None, (
            "Virtual pipeline model parallelism is unsupported for FalconDeepseekv4 "
            "(upstream MCore HybridModel API dependency)."
        )
        assert self.vocab_size is not None, "vocab_size must be configured before calling provide()"

        hybrid_stack_spec = self._resolve_hybrid_stack_spec()

        if self.hybrid_layer_pattern is None:
            ratios = list(self.csa_compress_ratios or [])
            num_layers = self.num_layers or len(ratios)
            if len(ratios) < num_layers:
                raise ValueError(
                    f"csa_compress_ratios has {len(ratios)} entries but num_layers is {num_layers}. "
                    "DSv4 needs one compress ratio per decoder layer."
                )
            # Assign back so checkpointing and layer-count derivation see the
            # same pattern the model was built with.
            self.hybrid_layer_pattern = falcon_deepseekv4_layer_pattern(
                ratios[:num_layers],
                pipeline_model_parallel_size=self.pipeline_model_parallel_size or 1,
                mtp_num_layers=self.mtp_num_layers or 0,
                num_hash_layers=self.moe_n_hash_layers or 0,
            )

        if self.should_pad_vocab:
            padded_vocab_size = calculate_padded_vocab_size(
                self.vocab_size, self.make_vocab_size_divisible_by, self.tensor_model_parallel_size
            )
        else:
            padded_vocab_size = self.vocab_size

        pre_process = pre_process if pre_process is not None else is_pp_first_stage(self._pg_collection.pp)
        post_process = post_process if post_process is not None else is_pp_last_stage(self._pg_collection.pp)

        patch_merger_spec = PatchMergerSubmodules(
            patch_norm=TENorm,
            linear_fc1=TEColumnParallelLinear,
            linear_fc2=TERowParallelLinear,
        )

        model = FalconDeepseekv4Model(
            language_transformer_config=self,
            language_transformer_layer_spec=hybrid_stack_spec,
            language_vocab_size=padded_vocab_size,
            language_max_sequence_length=self.seq_length,
            patch_embed_config=self._build_patch_embed_config(),
            patch_merger_spec=patch_merger_spec,
            image_token_id=self.image_token_id,
            parallel_output=self.parallel_output,
            share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
            pre_process=pre_process,
            post_process=post_process,
            hybrid_layer_pattern=self.hybrid_layer_pattern,
            pg_collection=self._pg_collection,
            vp_stage=vp_stage,
        )
        self._fix_hash_layer_numbering(model)
        return model
