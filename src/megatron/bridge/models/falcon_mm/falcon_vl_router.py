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

"""Falcon-MM VL hash-layer router: image pads bypass tid2eid with shared expert_bias."""

from __future__ import annotations

from typing import Optional

import torch
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.transformer_config import TransformerConfig

from megatron.bridge.models.conversion.param_mapping import AutoMapping


class FalconVLTopKRouter(TopKRouter):
    """Top-k router with hash-layer image_pad bypass for Falcon multimodal.

    When ``moe_vl_routing`` is enabled on a hash MoE layer:

    - Text tokens keep frozen ``tid2eid[input_ids]`` expert selection.
    - Image pad tokens (``input_ids == image_token_id``) select via
      ``topk(scores + expert_bias)`` using the same shared ``expert_bias`` as
      non-hash layers. There is no separate ``bias_vl``.
    - Routing weights always come from unbiased scores.

    When ``moe_vl_routing`` is false, behavior matches :class:`TopKRouter`.
    Non-hash layers always delegate to the parent implementation.
    """

    def __init__(
        self,
        config: TransformerConfig,
        pg_collection: Optional[ProcessGroupCollection] = None,
        is_mtp_layer: bool = False,
        layer_number: Optional[int] = None,
        hash_moe_layer_threshold: int | None = None,
    ) -> None:
        super().__init__(
            config=config,
            pg_collection=pg_collection,
            is_mtp_layer=is_mtp_layer,
            layer_number=layer_number,
            hash_moe_layer_threshold=hash_moe_layer_threshold,
        )
        self.moe_vl_routing = bool(getattr(config, "moe_vl_routing", False))
        self.image_token_id = getattr(config, "image_token_id", None)

        # Core disables expert_bias on hash layers; VL needs the shared bias for
        # image-pad top-k while keeping tid2eid for text.
        if self.moe_vl_routing and self.is_hash_layer:
            self._ensure_shared_expert_bias()

    def _ensure_shared_expert_bias(self) -> None:
        """Register float32 ``expert_bias`` (and token counters) if missing."""
        num_experts = self.config.num_moe_experts
        device = self.weight.device
        self.enable_expert_bias = True
        if self.local_tokens_per_expert is None:
            if "local_tokens_per_expert" in self.__dict__:
                del self.local_tokens_per_expert
            self.register_buffer(
                "local_tokens_per_expert",
                torch.zeros(num_experts, dtype=torch.float32, device=device),
                persistent=False,
            )
        if self.expert_bias is None:
            if "expert_bias" in self.__dict__:
                del self.expert_bias
            self.register_buffer(
                "expert_bias",
                torch.zeros(num_experts, dtype=torch.float32, device=device),
            )

    def _hash_routing(self, logits: torch.Tensor, input_ids: torch.Tensor):
        """Hash routing with optional image_pad top-k bypass.

        Args:
            logits: Gating logits, shape ``[num_tokens, num_experts]``.
            input_ids: Token IDs, shape ``[batch, seq]`` (transposed to match
                ``[s*b]`` flattening of hidden states).

        Returns:
            Tuple of ``(routing_probs, routing_map)``.
        """
        if not (self.moe_vl_routing and self.is_hash_layer):
            return super()._hash_routing(logits, input_ids)

        if self.image_token_id is None:
            raise ValueError(
                "moe_vl_routing requires image_token_id on the provider/config so "
                "hash-layer image pads can bypass tid2eid."
            )

        if self.score_function == "softmax":
            scores = torch.softmax(logits, dim=-1, dtype=torch.float32).type_as(logits)
        elif self.score_function == "sigmoid":
            scores = torch.sigmoid(logits.float()).type_as(logits)
        elif self.score_function == "sqrtsoftplus":
            scores = torch.nn.functional.softplus(logits.float()).sqrt().type_as(logits)
        else:
            raise ValueError(f"Invalid score_function: {self.score_function}")

        # input_ids is [b, s] from the model; hidden states flatten as [s*b, h].
        flat_ids = input_ids.T.reshape(-1)
        if flat_ids.numel() != logits.shape[0]:
            raise ValueError(f"input_ids flat length {flat_ids.numel()} != num_tokens {logits.shape[0]}")

        top_indices = self.tid2eid[flat_ids].long()
        image_mask = flat_ids == int(self.image_token_id)

        if self.config.moe_router_force_load_balancing or self.config.moe_router_force_biased is not None:
            _, top_indices = torch.topk(logits, k=self.topk, dim=1)
        elif image_mask.any():
            scores_for_routing = scores.float()
            if self.expert_bias is not None:
                scores_for_routing = scores_for_routing + self.expert_bias.float()
            _, image_indices = torch.topk(
                scores_for_routing[image_mask],
                k=self.topk,
                dim=1,
                sorted=torch.is_grad_enabled(),
            )
            top_indices = top_indices.clone()
            top_indices[image_mask] = image_indices

        # Weights always from unbiased scores.
        probs = scores.gather(1, top_indices)
        if self.score_function != "softmax":
            probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-20)

        if self.config.moe_router_topk_scaling_factor:
            probs = probs * self.config.moe_router_topk_scaling_factor

        routing_probs = torch.zeros_like(logits).scatter(1, top_indices, probs)
        routing_map = torch.zeros_like(logits).int().scatter(1, top_indices, 1).bool()
        return routing_probs, routing_map


AutoMapping.register_module_type("FalconVLTopKRouter", "replicated")
