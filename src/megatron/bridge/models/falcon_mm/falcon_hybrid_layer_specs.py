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

"""Falcon-MM hybrid stack specs (DSv4 + optional VL hash router)."""

from __future__ import annotations

import copy
from dataclasses import replace
from functools import partial

from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_dsv4_stack_spec
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig

from megatron.bridge.models.falcon_mm.falcon_vl_router import FalconVLTopKRouter


def _replace_moe_router(moe_builder, router_cls: type) -> object:
    """Return a MoELayer builder with ``submodules.router`` set to ``router_cls``."""
    if not isinstance(moe_builder, partial):
        return moe_builder
    if not (isinstance(moe_builder.func, type) and issubclass(moe_builder.func, MoELayer)):
        return moe_builder
    kwargs = dict(moe_builder.keywords or {})
    submodules = kwargs.get("submodules")
    if submodules is None or getattr(submodules, "router", None) is not TopKRouter:
        return moe_builder
    kwargs["submodules"] = replace(submodules, router=router_cls)
    return partial(moe_builder.func, *moe_builder.args, **kwargs)


def falcon_hybrid_dsv4_vl_stack_spec(config: TransformerConfig) -> ModuleSpec:
    """Core ``hybrid_dsv4_stack_spec`` with :class:`FalconVLTopKRouter` on MoE layers.

    Non-hash layers keep Core fused top-k with a single shared ``expert_bias``.
    Hash layers use the VL image-pad bypass implemented by the custom router.
    """
    base = hybrid_dsv4_stack_spec(config)
    stack = copy.deepcopy(base)
    moe_layer = stack.submodules.moe_layer
    moe_layer.submodules.mlp = _replace_moe_router(moe_layer.submodules.mlp, FalconVLTopKRouter)
    return stack


def falcon_dsv4_stack_spec(config: TransformerConfig) -> ModuleSpec:
    """Select Core DSv4 stack or the VL hash-router variant from ``moe_vl_routing``."""
    if getattr(config, "moe_vl_routing", False):
        return falcon_hybrid_dsv4_vl_stack_spec(config)
    return hybrid_dsv4_stack_spec(config)
