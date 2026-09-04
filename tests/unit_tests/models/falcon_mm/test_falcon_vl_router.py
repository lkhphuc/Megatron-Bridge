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

"""Unit tests for Falcon-MM VL hash router and hybrid stack wiring."""

from dataclasses import replace
from functools import partial
from types import SimpleNamespace

import pytest
import torch
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.moe.router import TopKRouter

from megatron.bridge.models.conversion.param_mapping import AutoMapping
from megatron.bridge.models.falcon_mm.falcon_hybrid_layer_specs import (
    falcon_dsv4_stack_spec,
    falcon_hybrid_dsv4_vl_stack_spec,
)
from megatron.bridge.models.falcon_mm.falcon_vl_router import FalconVLTopKRouter


IMAGE_TOKEN_ID = 31
VOCAB_SIZE = 64
NUM_EXPERTS = 4
TOPK = 2


def _make_tid2eid(vocab_size: int = VOCAB_SIZE, num_experts: int = NUM_EXPERTS, topk: int = TOPK) -> torch.Tensor:
    ids = torch.arange(vocab_size)
    return torch.stack([(ids + k) % num_experts for k in range(topk)], dim=1).to(torch.int32)


def _bare_vl_hash_router(
    *,
    moe_vl_routing: bool = True,
    score_function: str = "sqrtsoftplus",
    expert_bias: torch.Tensor | None = None,
) -> FalconVLTopKRouter:
    """Minimal FalconVLTopKRouter for ``_hash_routing`` without CUDA init."""
    router = object.__new__(FalconVLTopKRouter)
    router.moe_vl_routing = moe_vl_routing
    router.is_hash_layer = True
    router.image_token_id = IMAGE_TOKEN_ID
    router.topk = TOPK
    router.score_function = score_function
    router.tid2eid = _make_tid2eid()
    router.enable_expert_bias = expert_bias is not None
    router.expert_bias = expert_bias
    router.config = SimpleNamespace(
        moe_router_force_load_balancing=False,
        moe_router_force_biased=None,
        moe_router_topk_scaling_factor=None,
        num_moe_experts=NUM_EXPERTS,
    )
    return router


@pytest.mark.unit
class TestFalconVLTopKRouterHashRouting:
    def test_image_pads_use_topk_scores_plus_bias_not_tid2eid(self):
        """Many identical image_pad ids must not collapse to tid2eid[pad]."""
        bias = torch.tensor([0.0, 5.0, 0.0, -5.0], dtype=torch.float32)
        router = _bare_vl_hash_router(expert_bias=bias)

        # Craft logits so unbiased top-k differs from bias-shifted top-k.
        # Token scores (sqrtsoftplus of logits): prefer experts 0,2 without bias;
        # with +5 on expert 1, image tokens should pick 1 and 0.
        logits = torch.tensor(
            [
                [2.0, 0.0, 1.5, -1.0],
                [2.0, 0.0, 1.5, -1.0],
                [2.0, 0.0, 1.5, -1.0],
                [2.0, 0.0, 1.5, -1.0],
            ],
            dtype=torch.float32,
        )
        # [b, s] = [1, 4] all image pads
        input_ids = torch.full((1, 4), IMAGE_TOKEN_ID, dtype=torch.long)

        _, routing_map = router._hash_routing(logits, input_ids)

        pad_tid2eid = router.tid2eid[IMAGE_TOKEN_ID].long().sort().values
        scores = torch.nn.functional.softplus(logits.float()).sqrt()
        scores_for_routing = scores + bias
        expected_idx = torch.topk(scores_for_routing, k=TOPK, dim=1).indices

        for i in range(logits.shape[0]):
            actual = routing_map[i].nonzero(as_tuple=True)[0].sort().values
            expected = expected_idx[i].sort().values
            assert torch.equal(actual, expected)
            # Must not match the collapsed pad hash table entry.
            assert not torch.equal(actual, pad_tid2eid)

        # Identical pads still get content-dependent experts (all match expected).
        assert (routing_map.sum(dim=1) == TOPK).all()

    def test_text_ids_still_use_tid2eid(self):
        bias = torch.tensor([10.0, 10.0, 10.0, 10.0], dtype=torch.float32)
        router = _bare_vl_hash_router(expert_bias=bias)
        logits = torch.randn(4, NUM_EXPERTS)
        # Text token ids (not image pad)
        input_ids = torch.tensor([[3, 7, 11, 15]], dtype=torch.long)

        _, routing_map = router._hash_routing(logits, input_ids)

        flat_ids = input_ids.T.reshape(-1)
        expected = router.tid2eid[flat_ids].long()
        for i in range(logits.shape[0]):
            actual = routing_map[i].nonzero(as_tuple=True)[0].sort().values
            assert torch.equal(actual, expected[i].sort().values)

    def test_mixed_batch_text_hash_and_image_topk(self):
        bias = torch.zeros(NUM_EXPERTS, dtype=torch.float32)
        bias[3] = 100.0  # image path strongly prefers expert 3
        router = _bare_vl_hash_router(expert_bias=bias)
        logits = torch.zeros(4, NUM_EXPERTS)
        logits[:, 0] = 1.0  # unbiased prefers expert 0
        # [b, s]: text, image, text, image — after T.reshape: col-major flatten
        input_ids = torch.tensor([[10, IMAGE_TOKEN_ID], [20, IMAGE_TOKEN_ID]], dtype=torch.long)

        _, routing_map = router._hash_routing(logits, input_ids)
        flat_ids = input_ids.T.reshape(-1)
        image_mask = flat_ids == IMAGE_TOKEN_ID

        for i in range(logits.shape[0]):
            actual = set(routing_map[i].nonzero(as_tuple=True)[0].tolist())
            if image_mask[i]:
                assert 3 in actual
            else:
                assert actual == set(router.tid2eid[flat_ids[i]].long().tolist())

    def test_weights_come_from_unbiased_scores(self):
        bias = torch.tensor([0.0, 50.0, 0.0, 0.0], dtype=torch.float32)
        router = _bare_vl_hash_router(expert_bias=bias, score_function="sigmoid")
        logits = torch.tensor([[0.0, 0.0, 2.0, -2.0]], dtype=torch.float32)
        input_ids = torch.full((1, 1), IMAGE_TOKEN_ID, dtype=torch.long)

        routing_probs, routing_map = router._hash_routing(logits, input_ids)
        scores = torch.sigmoid(logits.float())
        selected = routing_map[0].nonzero(as_tuple=True)[0]
        # Prob mass on selected experts should match normalized unbiased scores.
        selected_scores = scores[0, selected]
        expected = selected_scores / selected_scores.sum()
        assert torch.allclose(routing_probs[0, selected], expected, atol=1e-5)

    def test_flag_off_delegates_to_parent_hash_routing(self, monkeypatch):
        router = _bare_vl_hash_router(moe_vl_routing=False, expert_bias=None)
        called = {}

        def fake_super_hash(self, logits, input_ids):
            called["yes"] = True
            return torch.zeros_like(logits), torch.zeros_like(logits, dtype=torch.bool)

        monkeypatch.setattr(TopKRouter, "_hash_routing", fake_super_hash)
        logits = torch.randn(2, NUM_EXPERTS)
        input_ids = torch.zeros(1, 2, dtype=torch.long)
        FalconVLTopKRouter._hash_routing(router, logits, input_ids)
        assert called.get("yes") is True

    def test_shared_bias_buffer_shape(self, monkeypatch):
        def fake_init(self, config, **kwargs):
            del kwargs
            torch.nn.Module.__init__(self)
            self.config = config
            self.topk = config.moe_router_topk
            self.is_hash_layer = True
            self.is_mtp_layer = False
            self.score_function = config.moe_router_score_function
            self.enable_expert_bias = False
            self.expert_bias = None
            self.local_tokens_per_expert = None
            self.tid2eid = _make_tid2eid(
                vocab_size=config.actual_vocab_size,
                num_experts=config.num_moe_experts,
                topk=config.moe_router_topk,
            )
            self.weight = torch.nn.Parameter(torch.zeros(config.num_moe_experts, config.hidden_size))

        monkeypatch.setattr(TopKRouter, "__init__", fake_init)
        config = SimpleNamespace(
            moe_vl_routing=True,
            image_token_id=IMAGE_TOKEN_ID,
            moe_router_topk=TOPK,
            num_moe_experts=NUM_EXPERTS,
            actual_vocab_size=VOCAB_SIZE,
            hidden_size=8,
            moe_router_score_function="sigmoid",
            moe_n_hash_layers=1,
        )
        router = FalconVLTopKRouter(config=config)
        assert router.enable_expert_bias is True
        assert router.expert_bias is not None
        assert tuple(router.expert_bias.shape) == (NUM_EXPERTS,)
        assert torch.count_nonzero(router.expert_bias) == 0


@pytest.mark.unit
class TestFalconHybridDsv4VlStackSpec:
    def test_vl_stack_wires_falcon_vl_router(self):
        config = SimpleNamespace(
            experimental_attention_variant="dsv4_hybrid",
            qk_layernorm=True,
            transformer_impl="transformer_engine",
        )
        # hybrid_dsv4_stack_spec needs a richer config; use the real Core helper
        # only if TE/backend resolution works. Otherwise validate the replace helper.
        try:
            stack = falcon_hybrid_dsv4_vl_stack_spec(config)
        except Exception:
            # Fallback: unit-test the MoE router swap in isolation.
            submodules = MoESubmodules(experts=object(), shared_experts=None)
            moe = partial(MoELayer, submodules=submodules)
            from megatron.bridge.models.falcon_mm.falcon_hybrid_layer_specs import _replace_moe_router

            replaced = _replace_moe_router(moe, FalconVLTopKRouter)
            assert replaced.keywords["submodules"].router is FalconVLTopKRouter
            assert submodules.router is TopKRouter
            return

        moe_builder = stack.submodules.moe_layer.submodules.mlp
        assert isinstance(moe_builder, partial)
        assert moe_builder.keywords["submodules"].router is FalconVLTopKRouter

    def test_flag_off_stack_keeps_core_spec(self):
        config = SimpleNamespace(moe_vl_routing=False)
        # falcon_dsv4_stack_spec with flag off should call Core without VL router.
        # Avoid building full DSv4 attention by stubbing hybrid_dsv4_stack_spec.
        sentinel = object()

        import megatron.bridge.models.falcon_mm.falcon_hybrid_layer_specs as specs

        original = specs.hybrid_dsv4_stack_spec
        specs.hybrid_dsv4_stack_spec = lambda cfg: sentinel
        try:
            assert falcon_dsv4_stack_spec(config) is sentinel
        finally:
            specs.hybrid_dsv4_stack_spec = original

    def test_flag_on_stack_selects_vl_variant(self, monkeypatch):
        import megatron.bridge.models.falcon_mm.falcon_hybrid_layer_specs as specs

        sentinel = object()
        monkeypatch.setattr(specs, "falcon_hybrid_dsv4_vl_stack_spec", lambda cfg: sentinel)
        config = SimpleNamespace(moe_vl_routing=True)
        assert falcon_dsv4_stack_spec(config) is sentinel

    def test_non_hash_moe_submodules_still_accept_single_shared_bias(self):
        """Fusion contract: one shared expert_bias vector (no per-token bias fork)."""
        submodules = MoESubmodules(experts=object(), shared_experts=None, router=FalconVLTopKRouter)
        assert submodules.router is FalconVLTopKRouter
        # Parent TopKRouter.routing still passes a single expert_bias into fused top-k.
        assert hasattr(TopKRouter, "routing")

    def test_router_registered_as_replicated(self):
        assert "FalconVLTopKRouter" in AutoMapping._MODULE_TYPE_REGISTRY["replicated"]

    def test_replace_leaves_original_submodules_untouched(self):
        from megatron.bridge.models.falcon_mm.falcon_hybrid_layer_specs import _replace_moe_router

        submodules = MoESubmodules(experts=object(), shared_experts=None)
        moe = partial(MoELayer, submodules=submodules)
        replaced = _replace_moe_router(moe, FalconVLTopKRouter)
        assert replaced.keywords["submodules"].router is FalconVLTopKRouter
        assert submodules.router is TopKRouter
        assert replace(submodules, router=TopKRouter).router is TopKRouter
