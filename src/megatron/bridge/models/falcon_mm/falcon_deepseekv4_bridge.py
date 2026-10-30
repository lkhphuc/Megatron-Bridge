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

"""Bridge for DeepSeek-V4 HF checkpoints onto the Falcon HybridModel trunk.

Maps the same DSv4 custom checkpoint layout used by
:class:`~megatron.bridge.models.deepseek.deepseek_v4_bridge.DeepSeekV4Bridge`
onto :class:`~megatron.bridge.models.falcon_mm.FalconDeepseekv4ModelProvider`
(Megatron-Core ``HybridModel``). Each HF decoder layer becomes two hybrid
layers (attention at ``2n``, MoE MLP at ``2n+1``) under
``language_model.decoder.layers.*.inner_layer``.

This bridge is used explicitly — it is not registered under the
``deepseek_v4`` model type (that remains owned by the GPT-trunk bridge).
"""

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    ColumnParallelMapping,
    GatedMLPMapping,
    ReplicatedMapping,
)
from megatron.bridge.models.deepseek.deepseek_v4_bridge import (
    DeepSeekV4Bridge,
    _dsv4_compress_ratios,
    _dsv4_num_hash_layers,
    _HCAlphaMapping,
    _HCAlphaSecondaryMapping,
    _ReplicatedOptional,
    deepseek_v4_supports_blackwell_fused_kernels,
    deepseek_v4_supports_fused_dsa_kernels,
)
from megatron.bridge.models.falcon_mm.falcon_deepseekv4_pattern import (
    falcon_deepseekv4_pattern_from_compress_ratios,
)
from megatron.bridge.models.falcon_mm.falcon_deepseekv4_provider import (
    FalconDeepseekv4ModelProvider,
)
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM


class _AutoOptional(AutoMapping):
    """AutoMapping that tolerates absent HF keys.

    Used for the patch-embed weights: text-only DSv4 checkpoints do not carry
    ``patch_embed.*`` keys, so the strict import/export HF-key validation
    must skip them (same pattern as ``_ReplicatedOptional`` in the GPT-trunk
    bridge). The patch embed keeps its random init on import: the bridge's
    ``maybe_modify_loaded_hf_weight`` returns None for absent keys and the
    conversion loop skips params whose mapping returns None.
    """

    def __init__(self, megatron_param: str, hf_param: str) -> None:
        super().__init__(megatron_param, hf_param)
        self.allow_hf_name_mismatch = True

    def hf_to_megatron(self, hf_weights, megatron_module):
        """Return None for absent HF weights so the import skips the param."""
        if hf_weights is None:
            return None
        return super().hf_to_megatron(hf_weights, megatron_module)


class FalconDeepseekv4Bridge(DeepSeekV4Bridge):
    """Bridge for DSv4 checkpoints onto the Falcon DeepSeek-V4 (HybridModel) trunk."""

    PROVIDER_CLASS = FalconDeepseekv4ModelProvider

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> FalconDeepseekv4ModelProvider:
        """Build a :class:`FalconDeepseekv4ModelProvider` from a DSv4 HF config.

        Uses the generic CONFIG_MAPPING construction with
        ``PROVIDER_CLASS = FalconDeepseekv4ModelProvider``, then applies the
        same DSv4 architecture fields as the GPT-trunk bridge, adapted for
        HybridModel (no ``transformer_layer_spec``; CSA fusion via
        ``dsa_kernel_backend``; ``mtp_num_layers = 0``; compress ratios sliced
        to ``num_hidden_layers`` only).

        Args:
            hf_pretrained: HuggingFace pretrained wrapper carrying the DSv4 config.

        Returns:
            A configured Falcon DeepSeek-V4 hybrid multimodal provider.
        """
        # Skip DeepSeekV4Bridge.provider_bridge (builds MLAModelProvider + GPT
        # layer spec). Use the base construction with our PROVIDER_CLASS.
        provider = MegatronModelBridge.provider_bridge(self, hf_pretrained)
        assert isinstance(provider, FalconDeepseekv4ModelProvider)
        hf_config = hf_pretrained.config
        provider.patch_embed_num_layers = int(getattr(hf_config, "patch_embed_num_layers", 3))
        provider.patch_embed_hidden_size = getattr(hf_config, "patch_embed_hidden_size", None)

        use_blackwell_fused_kernels = deepseek_v4_supports_blackwell_fused_kernels()
        use_dsa_kernel_fusion = use_blackwell_fused_kernels and deepseek_v4_supports_fused_dsa_kernels()

        # Hybrid / MLA providers keep position_embedding_type="rope"; YaRN is
        # expressed via rope_type + rotary scaling fields (not GPT yarn_*).
        provider.position_embedding_type = "rope"

        # ---- Attention / MLA geometry ----
        provider.v_head_dim = hf_config.head_dim
        provider.qk_pos_emb_head_dim = hf_config.qk_rope_head_dim
        provider.rotary_percent = 1.0
        provider.q_lora_rank = hf_config.q_lora_rank
        provider.o_groups = hf_config.o_groups
        provider.o_lora_rank = hf_config.o_lora_rank

        # ---- Rotary embeddings (YaRN) ----
        provider.apply_rope_fusion = True
        provider.rope_type = "yarn"
        rope_params = getattr(hf_config, "rope_scaling", None) or getattr(hf_config, "rope_parameters", None) or {}
        if "compress" in rope_params:
            main_rope_params = rope_params.get("main", {})
            compress_rope_params = rope_params["compress"]
        else:
            main_rope_params = rope_params
            compress_rope_params = rope_params
        provider.rotary_base = float(main_rope_params.get("rope_theta", hf_config.rope_theta))
        provider.csa_compress_rotary_base = float(
            getattr(hf_config, "compress_rope_theta", compress_rope_params.get("rope_theta", provider.rotary_base))
        )
        provider.rotary_scaling_factor = float(compress_rope_params["factor"])
        provider.original_max_position_embeddings = int(compress_rope_params["original_max_position_embeddings"])
        provider.beta_fast = float(compress_rope_params.get("beta_fast", 32))
        provider.beta_slow = float(compress_rope_params.get("beta_slow", 1))
        provider.mscale = 1.0
        provider.mscale_all_dim = 1.0
        # Multimodal RoPE: keep position_embedding_type="rope"; DSv4 layer-local
        # rotary modules consume mrope_section independently of model-level type.
        provider.mrope_section = (
            rope_params.get("mrope_section") or main_rope_params.get("mrope_section") or [11, 11, 10]
        )
        provider.mrope_interleaved = bool(
            rope_params.get("mrope_interleaved", main_rope_params.get("mrope_interleaved", True))
        )

        # ---- CSA / hybrid pattern ----
        # Phase 1: no MTP — slice compress ratios to num_hidden_layers only.
        # num_layers is derived from hybrid_layer_pattern in finalize().
        num_hidden_layers = int(hf_config.num_hidden_layers)
        compress_ratios = _dsv4_compress_ratios(hf_config)[:num_hidden_layers]
        provider.num_layers = None
        provider.csa_compress_ratios = compress_ratios
        provider.hybrid_layer_pattern = falcon_deepseekv4_pattern_from_compress_ratios(compress_ratios)
        provider.csa_window_size = hf_config.sliding_window
        provider.dsa_indexer_n_heads = hf_config.index_n_heads
        provider.dsa_indexer_head_dim = hf_config.index_head_dim
        provider.dsa_indexer_topk = hf_config.index_topk
        provider.dsa_kernel_backend = "cudnn" if use_dsa_kernel_fusion else "none"

        # ---- Hyper-Connections (mHC) ----
        provider.enable_hyper_connections = True
        provider.use_fused_mhc = use_blackwell_fused_kernels
        provider.num_residual_streams = hf_config.hc_mult
        provider.mhc_sinkhorn_iterations = hf_config.hc_sinkhorn_iters

        # ---- MoE ----
        provider.gated_linear_unit = True
        provider.moe_grouped_gemm = True
        provider.moe_router_pre_softmax = False
        provider.moe_token_dispatcher_type = "flex"
        provider.moe_flex_dispatcher_backend = "hybridep"
        provider.moe_flex_dispatcher_num_sms = 16
        provider.moe_permute_fusion_into_hybridep = False
        provider.moe_router_load_balancing_type = "noaux_tc"
        provider.moe_shared_expert_overlap = True
        provider.moe_router_score_function = hf_config.scoring_func
        provider.moe_router_enable_expert_bias = True
        provider.moe_router_dtype = "fp32"
        provider.moe_permute_fusion = True
        provider.moe_aux_loss_coeff = 0.0
        provider.moe_router_topk = hf_config.num_experts_per_tok
        provider.norm_topk_prob = hf_config.norm_topk_prob
        provider.moe_router_topk_scaling_factor = hf_config.routed_scaling_factor
        provider.moe_n_hash_layers = _dsv4_num_hash_layers(hf_config)
        provider.actual_vocab_size = hf_config.vocab_size
        provider.activation_func_clamp_value = hf_config.swiglu_limit
        # moe_layer_freq is derived from hybrid_layer_pattern in finalize().
        provider.moe_shared_expert_intermediate_size = hf_config.moe_intermediate_size * hf_config.n_shared_experts

        # ---- MTP (phase 1: disabled — Energon packing rejects MTP) ----
        provider.mtp_num_layers = 0

        # ---- Misc ----
        provider.share_embeddings_and_output_weights = bool(hf_config.tie_word_embeddings)
        provider.make_vocab_size_divisible_by = 1280
        provider.seq_length = 4096

        return provider

    def maybe_modify_loaded_hf_weight(self, hf_param, hf_state_dict):
        """Skip absent patch-embed keys; dequant everything else.

        Text-only DSv4 checkpoints carry no ``patch_embed.*`` keys. Returning
        None for those keys lets the conversion loop skip the patch embed,
        which keeps its random init. All other keys go through the GPT-trunk
        hook (FP8 / MXFP4 dequantisation).
        """
        if isinstance(hf_param, str) and hf_param.startswith("patch_embed.") and hf_param not in hf_state_dict:
            return None
        return super().maybe_modify_loaded_hf_weight(hf_param, hf_state_dict)

    def mapping_registry(self) -> MegatronMappingRegistry:
        """Return HybridModel weight mappings for DSv4 HF checkpoints.

        Builds explicit per-HF-layer mappings because each HF decoder layer
        splits into hybrid indices ``2n`` (attention) and ``2n+1`` (MoE).
        Compressor / indexer / hash-router keys are gated on
        ``csa_compress_ratios`` and ``moe_n_hash_layers``.

        Returns:
            Registry of Megatron ↔ HF parameter mappings.
        """
        hf_config = self.hf_config
        num_hidden_layers = int(hf_config.num_hidden_layers)
        compress_ratios = _dsv4_compress_ratios(hf_config)[:num_hidden_layers]
        num_hash_layers = _dsv4_num_hash_layers(hf_config)

        mappings: list = [
            # ------ Global: embed / head / final norm / HC head ------
            AutoMapping("language_model.embedding.word_embeddings.weight", "embed.weight"),
            AutoMapping("language_model.output_layer.weight", "head.weight"),
            AutoMapping("language_model.decoder.final_norm.weight", "norm.weight"),
            ReplicatedMapping("language_model.decoder.hc_head_fn", "hc_head_fn"),
            ReplicatedMapping("language_model.decoder.hc_head_base", "hc_head_base"),
            ReplicatedMapping("language_model.decoder.hc_head_scale", "hc_head_scale"),
        ]

        # Falcon-local hMLP keys are absent from text-only DSv4 checkpoints.
        # Keep this mapping isolated so it can be cherry-picked or dropped independently upstream.
        patch_embed_num_layers = int(getattr(hf_config, "patch_embed_num_layers", 3))
        for stage in range(patch_embed_num_layers):
            for module in range(2):
                for param in ("weight", "bias"):
                    name = f"stages.{stage}.{module}.{param}"
                    mappings.append(_AutoOptional(f"vision_model.{name}", f"patch_embed.{name}"))

        for n in range(num_hidden_layers):
            a = 2 * n  # attention hybrid layer
            m = 2 * n + 1  # MoE hybrid layer
            attn_pfx = f"language_model.decoder.layers.{a}.inner_layer"
            moe_pfx = f"language_model.decoder.layers.{m}.inner_layer"
            hc_attn = f"language_model.decoder.layers.{a}.hyper_connection"
            hc_ffn = f"language_model.decoder.layers.{m}.hyper_connection"

            # ------ Attention hybrid layer (index 2n) ------
            mappings += [
                AutoMapping(f"{attn_pfx}.input_layernorm.weight", f"layers.{n}.attn_norm.weight"),
                AutoMapping(
                    f"{attn_pfx}.self_attention.linear_q_down_proj.weight",
                    f"layers.{n}.attn.wq_a.weight",
                ),
                AutoMapping(
                    f"{attn_pfx}.self_attention.q_layernorm.weight",
                    f"layers.{n}.attn.q_norm.weight",
                ),
                AutoMapping(
                    f"{attn_pfx}.self_attention.linear_q_up_proj.weight",
                    f"layers.{n}.attn.wq_b.weight",
                ),
                AutoMapping(
                    f"{attn_pfx}.self_attention.linear_kv_proj.weight",
                    f"layers.{n}.attn.wkv.weight",
                ),
                AutoMapping(
                    f"{attn_pfx}.self_attention.kv_layernorm.weight",
                    f"layers.{n}.attn.kv_norm.weight",
                ),
                ReplicatedMapping(
                    f"{attn_pfx}.self_attention.linear_o_group_proj",
                    f"layers.{n}.attn.wo_a.weight",
                ),
                AutoMapping(
                    f"{attn_pfx}.self_attention.linear_proj.weight",
                    f"layers.{n}.attn.wo_b.weight",
                ),
                ColumnParallelMapping(
                    f"{attn_pfx}.self_attention.core_attention.attn_sink",
                    f"layers.{n}.attn.attn_sink",
                ),
                # HC (attn)
                ReplicatedMapping(f"{hc_attn}.mapping_proj.weight", f"layers.{n}.hc_attn_fn"),
                ReplicatedMapping(f"{hc_attn}.bias", f"layers.{n}.hc_attn_base"),
                _HCAlphaMapping(
                    megatron_pre=f"{hc_attn}.alpha_pre",
                    megatron_post=f"{hc_attn}.alpha_post",
                    megatron_res=f"{hc_attn}.alpha_res",
                    hf_param=f"layers.{n}.hc_attn_scale",
                ),
                _HCAlphaSecondaryMapping(f"{hc_attn}.alpha_post", f"layers.{n}.hc_attn_scale", 1),
                _HCAlphaSecondaryMapping(f"{hc_attn}.alpha_res", f"layers.{n}.hc_attn_scale", 2),
            ]

            if compress_ratios[n] > 0:
                mappings += [
                    _ReplicatedOptional(
                        f"{attn_pfx}.self_attention.core_attention.compressor.linear_wkv.weight",
                        f"layers.{n}.attn.compressor.wkv.weight",
                    ),
                    _ReplicatedOptional(
                        f"{attn_pfx}.self_attention.core_attention.compressor.linear_wgate.weight",
                        f"layers.{n}.attn.compressor.wgate.weight",
                    ),
                    _ReplicatedOptional(
                        f"{attn_pfx}.self_attention.core_attention.compressor.ape",
                        f"layers.{n}.attn.compressor.ape",
                    ),
                    _ReplicatedOptional(
                        f"{attn_pfx}.self_attention.core_attention.compressor.norm.weight",
                        f"layers.{n}.attn.compressor.norm.weight",
                    ),
                ]

            if compress_ratios[n] == 4:
                mappings += [
                    _ReplicatedOptional(
                        f"{attn_pfx}.self_attention.core_attention.indexer.linear_wq_b.weight",
                        f"layers.{n}.attn.indexer.wq_b.weight",
                    ),
                    _ReplicatedOptional(
                        f"{attn_pfx}.self_attention.core_attention.indexer.linear_weights_proj.weight",
                        f"layers.{n}.attn.indexer.scorer.weights_proj.weight",
                    ),
                    _ReplicatedOptional(
                        f"{attn_pfx}.self_attention.core_attention.indexer.compressor.linear_wkv.weight",
                        f"layers.{n}.attn.indexer.compressor.wkv.weight",
                    ),
                    _ReplicatedOptional(
                        f"{attn_pfx}.self_attention.core_attention.indexer.compressor.linear_wgate.weight",
                        f"layers.{n}.attn.indexer.compressor.wgate.weight",
                    ),
                    _ReplicatedOptional(
                        f"{attn_pfx}.self_attention.core_attention.indexer.compressor.ape",
                        f"layers.{n}.attn.indexer.compressor.ape",
                    ),
                    _ReplicatedOptional(
                        f"{attn_pfx}.self_attention.core_attention.indexer.compressor.norm.weight",
                        f"layers.{n}.attn.indexer.compressor.norm.weight",
                    ),
                ]

            # ------ MoE hybrid layer (index 2n+1) ------
            mappings += [
                AutoMapping(f"{moe_pfx}.pre_mlp_layernorm.weight", f"layers.{n}.ffn_norm.weight"),
                AutoMapping(f"{moe_pfx}.mlp.router.weight", f"layers.{n}.ffn.gate.weight"),
            ]
            if n >= num_hash_layers:
                mappings.append(
                    AutoMapping(f"{moe_pfx}.mlp.router.expert_bias", f"layers.{n}.ffn.gate.bias"),
                )
            if n < num_hash_layers:
                mappings.append(
                    AutoMapping(f"{moe_pfx}.mlp.router.tid2eid", f"layers.{n}.ffn.gate.tid2eid"),
                )
            mappings += [
                GatedMLPMapping(
                    megatron_param=f"{moe_pfx}.mlp.experts.linear_fc1.weight*",
                    gate=f"layers.{n}.ffn.experts.*.w1.weight",
                    up=f"layers.{n}.ffn.experts.*.w3.weight",
                ),
                AutoMapping(
                    f"{moe_pfx}.mlp.experts.linear_fc2.weight*",
                    f"layers.{n}.ffn.experts.*.w2.weight",
                ),
                GatedMLPMapping(
                    megatron_param=f"{moe_pfx}.mlp.shared_experts.linear_fc1.weight",
                    gate=f"layers.{n}.ffn.shared_experts.w1.weight",
                    up=f"layers.{n}.ffn.shared_experts.w3.weight",
                ),
                AutoMapping(
                    f"{moe_pfx}.mlp.shared_experts.linear_fc2.weight",
                    f"layers.{n}.ffn.shared_experts.w2.weight",
                ),
                # HC (ffn)
                ReplicatedMapping(f"{hc_ffn}.mapping_proj.weight", f"layers.{n}.hc_ffn_fn"),
                ReplicatedMapping(f"{hc_ffn}.bias", f"layers.{n}.hc_ffn_base"),
                _HCAlphaMapping(
                    megatron_pre=f"{hc_ffn}.alpha_pre",
                    megatron_post=f"{hc_ffn}.alpha_post",
                    megatron_res=f"{hc_ffn}.alpha_res",
                    hf_param=f"layers.{n}.hc_ffn_scale",
                ),
                _HCAlphaSecondaryMapping(f"{hc_ffn}.alpha_post", f"layers.{n}.hc_ffn_scale", 1),
                _HCAlphaSecondaryMapping(f"{hc_ffn}.alpha_res", f"layers.{n}.hc_ffn_scale", 2),
            ]

        return MegatronMappingRegistry(*mappings)
