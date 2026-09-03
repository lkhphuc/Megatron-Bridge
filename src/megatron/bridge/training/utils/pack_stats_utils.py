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

"""Accumulate and resolve packed multimodal dataloader statistics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from megatron.bridge.training.utils.flop_utils import (
    _accumulator_to_int,
    _add_flops_accumulator,
    _real_subseq_lengths,
    _scalar_sum_for_accumulator,
    vision_patch_stats_from_grid_thw,
)


_PACK_ACCUMULATOR_NAMES = (
    "_pack_physical_tokens",
    "_pack_real_tokens",
    "_pack_docs",
    "_pack_images",
    "_pack_image_tokens",
    "_pack_loss_tokens",
    "_pack_microbatches",
)

# Fixed order for DP all-reduce of Falcon-MM / Energon packing skip reasons.
PACK_SKIP_REASONS = (
    "overlong",
    "max_num_images",
    "max_visual_tokens",
    "geometry",
    "decompression_bomb",
    "chatml_overlength",
    "length_mismatch",
    "oversize_pack",
)


def _empty_skip_counts() -> dict[str, int]:
    return {reason: 0 for reason in PACK_SKIP_REASONS}


@dataclass(frozen=True)
class PackRuntimeStats:
    """Data-parallel-global pack stats for one training step."""

    physical_tokens: int = 0
    real_tokens: int = 0
    docs: int = 0
    images: int = 0
    image_tokens: int = 0
    loss_tokens: int = 0
    microbatches: int = 0
    skipped_by_reason: dict[str, int] = field(default_factory=_empty_skip_counts)

    @property
    def padding_fraction(self) -> float:
        """Fraction of physical pack tokens that are padding."""
        if self.physical_tokens <= 0:
            return 0.0
        return max(self.physical_tokens - self.real_tokens, 0) / float(self.physical_tokens)

    @property
    def docs_per_pack(self) -> float:
        """Mean source documents per physical pack microbatch."""
        if self.microbatches <= 0:
            return 0.0
        return self.docs / float(self.microbatches)

    @property
    def skipped_samples(self) -> int:
        """Total samples skipped during packing for this step."""
        return sum(int(count) for count in self.skipped_by_reason.values())


def reset_pack_stats_accumulators(state: Any) -> None:
    """Clear per-step pack-stat accumulators on ``state``."""
    for name in _PACK_ACCUMULATOR_NAMES:
        setattr(state, name, 0)
    setattr(state, "_pack_skip_pending", _empty_skip_counts())
    setattr(state, "_pack_requires_global_reduce", False)


def _cu_last(cu: torch.Tensor | None) -> int | torch.Tensor | None:
    if cu is None:
        return None
    squeezed = cu.squeeze()
    if squeezed.numel() == 0:
        return None
    return squeezed[-1]


def _count_images(visual_inputs: Any) -> int | torch.Tensor:
    if visual_inputs is None:
        return 0
    total: int | torch.Tensor = 0
    for attr in ("image_grid_thw", "video_grid_thw"):
        grid = getattr(visual_inputs, attr, None)
        if grid is None or grid.numel() == 0:
            continue
        num = grid.reshape(-1, grid.size(-1)).size(0)
        total = total + num
    return total


def _count_image_tokens(visual_inputs: Any, *, spatial_merge_size: int) -> int | torch.Tensor:
    if visual_inputs is None:
        return 0
    total: int | torch.Tensor = 0
    for attr in ("image_grid_thw", "video_grid_thw"):
        grid = getattr(visual_inputs, attr, None)
        if grid is None or grid.numel() == 0:
            continue
        _, _, merged = vision_patch_stats_from_grid_thw(grid, spatial_merge_size=spatial_merge_size)
        total = total + merged
    return total


def accumulate_pack_stats(
    state: Any,
    tokens: torch.Tensor | None,
    *,
    packed_seq_params: dict[str, Any] | None = None,
    visual_inputs: Any = None,
    loss_mask: torch.Tensor | None = None,
    spatial_merge_size: int = 1,
    vp_stage: int | None = None,
) -> None:
    """Accumulate per-microbatch packed-dataloader stats onto ``state``.

    Only virtual-pipeline stage 0 (or an unchunked model) contributes, matching
    :func:`~megatron.bridge.training.utils.flop_utils.accumulate_flops_metadata`.

    Writes ephemeral ``_pack_*`` accumulators consumed by
    :func:`resolve_global_pack_stats` at the end of the training step.
    """
    if vp_stage not in (None, 0) or tokens is None:
        return
    if packed_seq_params is None or packed_seq_params.get("cu_seqlens_q") is None:
        return

    cu_seqlens_q = packed_seq_params.get("cu_seqlens_q")
    cu_seqlens_q_padded = packed_seq_params.get("cu_seqlens_q_padded")
    # Prefer unpadded boundaries for real tokens; padded cu for physical width.
    real_lens = _real_subseq_lengths(
        cu_seqlens_q_padded if cu_seqlens_q_padded is not None else cu_seqlens_q,
        cu_seqlens_unpadded=cu_seqlens_q if cu_seqlens_q_padded is not None else None,
    )
    if real_lens is None or real_lens.numel() == 0:
        return

    physical_end = _cu_last(cu_seqlens_q_padded if cu_seqlens_q_padded is not None else cu_seqlens_q)
    if physical_end is None:
        physical_tokens: int | torch.Tensor = tokens.size(0) * tokens.size(1)
    else:
        physical_tokens = physical_end

    real_tokens = _scalar_sum_for_accumulator(real_lens)
    # Count every cu_seqlens segment as one packed document (same as len(cu)-1).
    docs = real_lens.numel()
    images = _count_images(visual_inputs)
    image_tokens = _count_image_tokens(visual_inputs, spatial_merge_size=max(int(spatial_merge_size), 1))
    loss_tokens: int | torch.Tensor = 0
    if loss_mask is not None:
        loss_tokens = _scalar_sum_for_accumulator(loss_mask.detach())

    _add_flops_accumulator(state, "_pack_physical_tokens", physical_tokens)
    _add_flops_accumulator(state, "_pack_real_tokens", real_tokens)
    _add_flops_accumulator(state, "_pack_docs", docs)
    _add_flops_accumulator(state, "_pack_images", images)
    _add_flops_accumulator(state, "_pack_image_tokens", image_tokens)
    _add_flops_accumulator(state, "_pack_loss_tokens", loss_tokens)
    _add_flops_accumulator(state, "_pack_microbatches", tokens.size(0))
    setattr(state, "_pack_requires_global_reduce", True)


def accumulate_pack_skip_stats(
    state: Any,
    skip_stats: dict[str, int] | None,
    *,
    vp_stage: int | None = None,
) -> None:
    """Accumulate packing skip deltas piggybacked on a successful microbatch."""
    if vp_stage not in (None, 0) or not skip_stats:
        return
    pending = getattr(state, "_pack_skip_pending", None)
    if pending is None:
        pending = _empty_skip_counts()
        setattr(state, "_pack_skip_pending", pending)
    for reason, count in skip_stats.items():
        if reason not in pending:
            continue
        pending[reason] = int(pending.get(reason, 0)) + int(count)
    setattr(state, "_pack_requires_global_reduce", True)


def _local_skip_counts(state: Any) -> list[int]:
    pending = getattr(state, "_pack_skip_pending", None) or {}
    return [int(pending.get(reason, 0)) for reason in PACK_SKIP_REASONS]


def resolve_global_pack_stats(
    state: Any,
    *,
    data_parallel_size: int,
    dp_group: Any = None,
) -> PackRuntimeStats:
    """Reduce per-rank pack accumulators to data-parallel-global totals."""
    local_physical = _accumulator_to_int(getattr(state, "_pack_physical_tokens", 0))
    local_real = _accumulator_to_int(getattr(state, "_pack_real_tokens", 0))
    local_docs = _accumulator_to_int(getattr(state, "_pack_docs", 0))
    local_images = _accumulator_to_int(getattr(state, "_pack_images", 0))
    local_image_tokens = _accumulator_to_int(getattr(state, "_pack_image_tokens", 0))
    local_loss_tokens = _accumulator_to_int(getattr(state, "_pack_loss_tokens", 0))
    local_microbatches = _accumulator_to_int(getattr(state, "_pack_microbatches", 0))
    local_skips = _local_skip_counts(state)

    use_all_reduce = (
        bool(getattr(state, "_pack_requires_global_reduce", False))
        and dp_group is not None
        and data_parallel_size > 1
        and torch.distributed.is_available()
        and torch.distributed.is_initialized()
    )
    if use_all_reduce:
        device = torch.cuda.current_device() if torch.cuda.is_available() else None
        values = torch.tensor(
            [
                local_physical,
                local_real,
                local_docs,
                local_images,
                local_image_tokens,
                local_loss_tokens,
                local_microbatches,
                *local_skips,
            ],
            dtype=torch.long,
            device=device,
        )
        torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.SUM, group=dp_group)
        reduced = [int(x) for x in values.tolist()]
        physical_tokens, real_tokens, docs, images, image_tokens, loss_tokens, microbatches = reduced[:7]
        skip_counts = reduced[7:]
    else:
        # Dense / no collective: extrapolate from the local rank.
        physical_tokens = local_physical * data_parallel_size
        real_tokens = local_real * data_parallel_size
        docs = local_docs * data_parallel_size
        images = local_images * data_parallel_size
        image_tokens = local_image_tokens * data_parallel_size
        loss_tokens = local_loss_tokens * data_parallel_size
        microbatches = local_microbatches * data_parallel_size
        skip_counts = [count * data_parallel_size for count in local_skips]

    skipped_by_reason = {
        reason: max(int(count), 0) for reason, count in zip(PACK_SKIP_REASONS, skip_counts, strict=True)
    }
    return PackRuntimeStats(
        physical_tokens=max(physical_tokens, 0),
        real_tokens=max(real_tokens, 0),
        docs=max(docs, 0),
        images=max(images, 0),
        image_tokens=max(image_tokens, 0),
        loss_tokens=max(loss_tokens, 0),
        microbatches=max(microbatches, 0),
        skipped_by_reason=skipped_by_reason,
    )


def pack_stats_to_log_metrics(stats: PackRuntimeStats, train_state: Any | None = None) -> dict[str, float]:
    """Convert step pack stats into ``data/*`` logger metrics."""
    if stats.microbatches <= 0 and stats.physical_tokens <= 0 and stats.skipped_samples <= 0:
        return {}
    metrics: dict[str, float] = {}
    if stats.microbatches > 0 or stats.physical_tokens > 0:
        metrics.update(
            {
                "data/padding-fraction": float(stats.padding_fraction),
                "data/docs-per-pack": float(stats.docs_per_pack),
                "data/images": float(stats.images),
                "data/image-tokens": float(stats.image_tokens),
                "data/real-tokens": float(stats.real_tokens),
            }
        )
        if stats.loss_tokens > 0:
            metrics["data/loss-tokens"] = float(stats.loss_tokens)
    # Always log skip counts (including zeros) whenever we emit a data/* report so
    # WandB does not linearly interpolate across missing log_interval points.
    metrics["data/skipped-samples"] = float(stats.skipped_samples)
    for reason in PACK_SKIP_REASONS:
        metrics[f"data/skipped-{reason}"] = float(stats.skipped_by_reason.get(reason, 0))
    if train_state is not None:
        metrics["data/images-seen"] = float(getattr(train_state, "consumed_images", 0))
        metrics["data/image-tokens-seen"] = float(getattr(train_state, "consumed_image_tokens", 0))
        metrics["data/docs-seen"] = float(getattr(train_state, "consumed_pack_docs", 0))
        metrics["data/real-tokens-seen"] = float(getattr(train_state, "consumed_real_tokens", 0))
        metrics["data/skipped-samples-seen"] = float(getattr(train_state, "consumed_pack_skips", 0))
    return metrics


def fold_pack_stats_into_train_state(train_state: Any, stats: PackRuntimeStats) -> None:
    """Add step totals into checkpointed cumulative counters on ``train_state``."""
    train_state.consumed_pack_docs += int(stats.docs)
    train_state.consumed_images += int(stats.images)
    train_state.consumed_image_tokens += int(stats.image_tokens)
    train_state.consumed_real_tokens += int(stats.real_tokens)
    train_state.consumed_pack_skips += int(stats.skipped_samples)


def accumulate_pack_stats_for_log_interval(state: Any, stats: PackRuntimeStats) -> None:
    """Fold one step's pack stats into log-interval totals on ``state``."""
    if stats.microbatches <= 0 and stats.physical_tokens <= 0 and stats.skipped_samples <= 0:
        return
    _add_flops_accumulator(state, "_pack_interval_physical_tokens", stats.physical_tokens)
    _add_flops_accumulator(state, "_pack_interval_real_tokens", stats.real_tokens)
    _add_flops_accumulator(state, "_pack_interval_docs", stats.docs)
    _add_flops_accumulator(state, "_pack_interval_images", stats.images)
    _add_flops_accumulator(state, "_pack_interval_image_tokens", stats.image_tokens)
    _add_flops_accumulator(state, "_pack_interval_loss_tokens", stats.loss_tokens)
    _add_flops_accumulator(state, "_pack_interval_microbatches", stats.microbatches)
    _add_flops_accumulator(state, "_pack_interval_steps", 1)
    interval_skips = getattr(state, "_pack_interval_skips", None)
    if interval_skips is None:
        interval_skips = _empty_skip_counts()
        setattr(state, "_pack_interval_skips", interval_skips)
    for reason, count in stats.skipped_by_reason.items():
        if reason not in interval_skips:
            continue
        interval_skips[reason] = int(interval_skips.get(reason, 0)) + int(count)


def reset_pack_stats_log_interval(state: Any) -> None:
    """Clear log-interval pack totals after they have been emitted."""
    for name in (
        "_pack_interval_physical_tokens",
        "_pack_interval_real_tokens",
        "_pack_interval_docs",
        "_pack_interval_images",
        "_pack_interval_image_tokens",
        "_pack_interval_loss_tokens",
        "_pack_interval_microbatches",
        "_pack_interval_steps",
    ):
        setattr(state, name, 0)
    setattr(state, "_pack_interval_skips", _empty_skip_counts())


def consume_pack_stats_log_metrics(state: Any, train_state: Any | None = None) -> dict[str, float]:
    """Build ``data/*`` metrics from log-interval totals and reset the interval."""
    interval_skips = getattr(state, "_pack_interval_skips", None) or _empty_skip_counts()
    interval = PackRuntimeStats(
        physical_tokens=_accumulator_to_int(getattr(state, "_pack_interval_physical_tokens", 0)),
        real_tokens=_accumulator_to_int(getattr(state, "_pack_interval_real_tokens", 0)),
        docs=_accumulator_to_int(getattr(state, "_pack_interval_docs", 0)),
        images=_accumulator_to_int(getattr(state, "_pack_interval_images", 0)),
        image_tokens=_accumulator_to_int(getattr(state, "_pack_interval_image_tokens", 0)),
        loss_tokens=_accumulator_to_int(getattr(state, "_pack_interval_loss_tokens", 0)),
        microbatches=_accumulator_to_int(getattr(state, "_pack_interval_microbatches", 0)),
        skipped_by_reason={reason: int(interval_skips.get(reason, 0)) for reason in PACK_SKIP_REASONS},
    )
    metrics = pack_stats_to_log_metrics(interval, train_state)
    reset_pack_stats_log_interval(state)
    return metrics
