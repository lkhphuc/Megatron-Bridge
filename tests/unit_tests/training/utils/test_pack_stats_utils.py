from types import SimpleNamespace

import torch

from megatron.bridge.training.state import TrainState
from megatron.bridge.training.utils.pack_stats_utils import (
    PackRuntimeStats,
    accumulate_pack_skip_stats,
    accumulate_pack_stats,
    accumulate_pack_stats_for_log_interval,
    consume_pack_stats_log_metrics,
    fold_pack_stats_into_train_state,
    pack_stats_to_log_metrics,
    reset_pack_stats_accumulators,
    resolve_global_pack_stats,
)


class _FakeVisualInputs:
    def __init__(self, image_grid_thw=None, video_grid_thw=None):
        self.image_grid_thw = image_grid_thw
        self.video_grid_thw = video_grid_thw


class TestAccumulatePackStats:
    def test_zero_padding_pack(self):
        state = SimpleNamespace()
        reset_pack_stats_accumulators(state)
        tokens = torch.zeros(1, 10, dtype=torch.long)
        packed = {"cu_seqlens_q": torch.tensor([0, 4, 10], dtype=torch.int32)}
        loss_mask = torch.ones(1, 10)

        accumulate_pack_stats(
            state,
            tokens,
            packed_seq_params=packed,
            visual_inputs=None,
            loss_mask=loss_mask,
            spatial_merge_size=1,
            vp_stage=0,
        )
        stats = resolve_global_pack_stats(state, data_parallel_size=1, dp_group=None)

        assert stats.physical_tokens == 10
        assert stats.real_tokens == 10
        assert stats.docs == 2
        assert stats.padding_fraction == 0.0
        assert stats.docs_per_pack == 2.0
        assert stats.loss_tokens == 10
        assert stats.microbatches == 1

    def test_alignment_and_trailing_padding(self):
        state = SimpleNamespace()
        reset_pack_stats_accumulators(state)
        # Two docs of length 3 and 5, aligned/padded to physical width 16.
        tokens = torch.zeros(1, 16, dtype=torch.long)
        packed = {
            "cu_seqlens_q": torch.tensor([0, 3, 8], dtype=torch.int32),
            "cu_seqlens_q_padded": torch.tensor([0, 4, 16], dtype=torch.int32),
        }

        accumulate_pack_stats(state, tokens, packed_seq_params=packed, vp_stage=None)
        stats = resolve_global_pack_stats(state, data_parallel_size=1)

        assert stats.physical_tokens == 16
        assert stats.real_tokens == 8
        assert stats.docs == 2
        assert stats.padding_fraction == 0.5

    def test_images_and_image_tokens(self):
        state = SimpleNamespace()
        reset_pack_stats_accumulators(state)
        tokens = torch.zeros(1, 12, dtype=torch.long)
        packed = {"cu_seqlens_q": torch.tensor([0, 12], dtype=torch.int32)}
        # Two images: 1x2x2 and 1x4x4 patches; merge_size=2 → 1 + 4 = 5 image tokens.
        visual = _FakeVisualInputs(
            image_grid_thw=torch.tensor([[1, 2, 2], [1, 4, 4]], dtype=torch.int64),
        )

        accumulate_pack_stats(
            state,
            tokens,
            packed_seq_params=packed,
            visual_inputs=visual,
            spatial_merge_size=2,
        )
        stats = resolve_global_pack_stats(state, data_parallel_size=1)

        assert stats.images == 2
        assert stats.image_tokens == 5

    def test_skips_non_packed_and_non_primary_vp_stage(self):
        state = SimpleNamespace()
        reset_pack_stats_accumulators(state)
        tokens = torch.zeros(1, 8, dtype=torch.long)

        accumulate_pack_stats(state, tokens, packed_seq_params=None, vp_stage=0)
        accumulate_pack_stats(
            state,
            tokens,
            packed_seq_params={"cu_seqlens_q": torch.tensor([0, 8], dtype=torch.int32)},
            vp_stage=1,
        )
        stats = resolve_global_pack_stats(state, data_parallel_size=1)
        assert stats.microbatches == 0
        assert stats.physical_tokens == 0


class TestPackStatsLoggingAndTrainState:
    def test_fold_and_log_metrics_include_cumulatives(self):
        train_state = TrainState()
        stats = PackRuntimeStats(
            physical_tokens=16,
            real_tokens=12,
            docs=3,
            images=2,
            image_tokens=7,
            loss_tokens=5,
            microbatches=1,
        )
        fold_pack_stats_into_train_state(train_state, stats)
        fold_pack_stats_into_train_state(train_state, stats)

        assert train_state.consumed_pack_docs == 6
        assert train_state.consumed_images == 4
        assert train_state.consumed_image_tokens == 14
        assert train_state.consumed_real_tokens == 24
        assert train_state.consumed_pack_skips == 0

        metrics = pack_stats_to_log_metrics(stats, train_state)
        assert metrics["data/padding-fraction"] == 0.25
        assert metrics["data/docs-per-pack"] == 3.0
        assert metrics["data/images"] == 2.0
        assert metrics["data/image-tokens"] == 7.0
        assert metrics["data/real-tokens"] == 12.0
        assert metrics["data/loss-tokens"] == 5.0
        assert metrics["data/images-seen"] == 4.0
        assert metrics["data/image-tokens-seen"] == 14.0
        assert metrics["data/docs-seen"] == 6.0
        assert metrics["data/real-tokens-seen"] == 24.0
        assert metrics["data/skipped-samples"] == 0.0
        assert metrics["data/skipped-overlong"] == 0.0
        assert metrics["data/skipped-samples-seen"] == 0.0

    def test_skip_stats_resolve_and_log_metrics(self):
        state = SimpleNamespace()
        reset_pack_stats_accumulators(state)
        tokens = torch.zeros(1, 10, dtype=torch.long)
        packed = {"cu_seqlens_q": torch.tensor([0, 10], dtype=torch.int32)}
        accumulate_pack_stats(state, tokens, packed_seq_params=packed, vp_stage=0)
        accumulate_pack_skip_stats(
            state,
            {"overlong": 2, "max_num_images": 1, "unknown_reason": 9},
            vp_stage=0,
        )
        stats = resolve_global_pack_stats(state, data_parallel_size=1)
        assert stats.skipped_samples == 3
        assert stats.skipped_by_reason["overlong"] == 2
        assert stats.skipped_by_reason["max_num_images"] == 1
        assert "unknown_reason" not in stats.skipped_by_reason

        train_state = TrainState()
        fold_pack_stats_into_train_state(train_state, stats)
        accumulate_pack_stats_for_log_interval(state, stats)
        metrics = consume_pack_stats_log_metrics(state, train_state)
        assert metrics["data/skipped-samples"] == 3.0
        assert metrics["data/skipped-overlong"] == 2.0
        assert metrics["data/skipped-max_num_images"] == 1.0
        assert metrics["data/skipped-geometry"] == 0.0
        assert metrics["data/skipped-samples-seen"] == 3.0
        assert train_state.consumed_pack_skips == 3

    def test_interval_accumulate_and_consume(self):
        state = SimpleNamespace()
        train_state = TrainState(
            consumed_images=10,
            consumed_image_tokens=20,
            consumed_pack_docs=4,
            consumed_real_tokens=30,
        )
        accumulate_pack_stats_for_log_interval(
            state,
            PackRuntimeStats(
                physical_tokens=10,
                real_tokens=10,
                docs=2,
                images=1,
                image_tokens=3,
                loss_tokens=4,
                microbatches=1,
            ),
        )
        accumulate_pack_stats_for_log_interval(
            state,
            PackRuntimeStats(
                physical_tokens=10,
                real_tokens=8,
                docs=1,
                images=1,
                image_tokens=2,
                loss_tokens=3,
                microbatches=1,
            ),
        )
        metrics = consume_pack_stats_log_metrics(state, train_state)
        assert metrics["data/padding-fraction"] == 0.1
        assert metrics["data/docs-per-pack"] == 1.5
        assert metrics["data/images"] == 2.0
        assert metrics["data/images-seen"] == 10.0
        assert metrics["data/skipped-samples"] == 0.0
        # Interval cleared after consume.
        assert consume_pack_stats_log_metrics(state, train_state) == {}

    def test_train_state_roundtrip_and_legacy_checkpoint(self):
        original = TrainState(
            step=3,
            consumed_train_samples=12,
            consumed_pack_docs=5,
            consumed_images=9,
            consumed_image_tokens=11,
            consumed_real_tokens=13,
            consumed_pack_skips=7,
        )
        restored = TrainState()
        restored.load_state_dict(original.state_dict())
        assert restored.consumed_pack_docs == 5
        assert restored.consumed_images == 9
        assert restored.consumed_image_tokens == 11
        assert restored.consumed_real_tokens == 13
        assert restored.consumed_pack_skips == 7

        legacy = {
            "step": torch.tensor(1, dtype=torch.int64),
            "consumed_train_samples": torch.tensor(2, dtype=torch.int64),
            "skipped_train_samples": torch.tensor(0, dtype=torch.int64),
            "consumed_valid_samples": torch.tensor(0, dtype=torch.int64),
            "floating_point_operations_so_far": torch.tensor(0.0, dtype=torch.float64),
            "do_train": torch.tensor(True, dtype=torch.bool),
            "do_valid": torch.tensor(False, dtype=torch.bool),
            "do_test": torch.tensor(False, dtype=torch.bool),
        }
        legacy_state = TrainState()
        legacy_state.load_state_dict(legacy)
        assert legacy_state.consumed_pack_docs == 0
        assert legacy_state.consumed_images == 0
        assert legacy_state.consumed_image_tokens == 0
        assert legacy_state.consumed_real_tokens == 0
        assert legacy_state.consumed_pack_skips == 0
