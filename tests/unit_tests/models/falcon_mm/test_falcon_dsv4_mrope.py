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

"""Unit tests for Falcon-MM DSv4 multimodal RoPE (WP6)."""

import pytest
import torch
from megatron.core.packed_seq_params import PackedSeqParams

from megatron.bridge.models.falcon_mm.rope import (
    FALCON_DSV4_DEFAULT_MROPE_SECTION,
    FALCON_DSV4_IMAGE_TOKEN_ID,
    FALCON_DSV4_VISION_START_TOKEN_ID,
    build_falcon_dsv4_mrope_position_ids,
    expand_scalar_position_ids,
    interleaved_mrope_section_for_rotary_dim,
    is_mrope_position_ids,
    resolve_falcon_dsv4_mrope_position_ids,
)
from megatron.bridge.training.utils.packed_seq_utils import partition_packed_batch_contiguous


@pytest.mark.unit
class TestFalconDsv4MropeHelpers:
    def test_default_section_matches_interleaved_r64(self):
        assert FALCON_DSV4_DEFAULT_MROPE_SECTION == [11, 11, 10]
        assert interleaved_mrope_section_for_rotary_dim(64) == [11, 11, 10]
        assert interleaved_mrope_section_for_rotary_dim(32) == [6, 5, 5]

    def test_expand_scalar_equal_axes(self):
        scalar = torch.arange(5).unsqueeze(0)
        expanded = expand_scalar_position_ids(scalar)
        assert expanded.shape == (3, 1, 5)
        assert torch.equal(expanded[0], expanded[1])
        assert torch.equal(expanded[1], expanded[2])

    def test_text_only_equal_axes(self):
        input_ids = torch.tensor([[10, 11, 12, 13]], dtype=torch.long)
        position_ids = build_falcon_dsv4_mrope_position_ids(
            input_ids,
            spatial_merge_size=1,
        )
        assert position_ids.shape == (3, 1, 4)
        assert torch.equal(position_ids[0], position_ids[1])
        assert torch.equal(position_ids[1], position_ids[2])
        assert position_ids[0, 0].tolist() == [0, 1, 2, 3]

    def test_visual_sequence_distinct_hw(self):
        # vision_start, four image pads (2x2 grid), vision_end, then text
        vision_start = FALCON_DSV4_VISION_START_TOKEN_ID
        image_pad = FALCON_DSV4_IMAGE_TOKEN_ID
        vision_end = 128002
        input_ids = torch.tensor(
            [[vision_start, image_pad, image_pad, image_pad, image_pad, vision_end, 42]],
            dtype=torch.long,
        )
        image_grid_thw = torch.tensor([[1, 2, 2]], dtype=torch.long)
        position_ids = build_falcon_dsv4_mrope_position_ids(
            input_ids,
            spatial_merge_size=1,
            image_grid_thw=image_grid_thw,
        )
        assert position_ids.shape == (3, 1, 7)
        assert position_ids[0, 0, 0].item() == 0  # vision_start
        image_positions = position_ids[:, 0, 1:5]
        assert image_positions[0].tolist() == [1, 1, 1, 1]
        assert image_positions[1].tolist() == [1, 1, 2, 2]
        assert image_positions[2].tolist() == [1, 2, 1, 2]
        # Tokens after the visual span share equal T/H/W axes again.
        assert torch.equal(position_ids[:, 0, 5], position_ids[:, 0, 5].new_tensor([3, 3, 3]))
        assert torch.equal(position_ids[:, 0, 6], position_ids[:, 0, 6].new_tensor([4, 4, 4]))

    def test_packed_sample_reset(self):
        vision_start = FALCON_DSV4_VISION_START_TOKEN_ID
        image_pad = FALCON_DSV4_IMAGE_TOKEN_ID
        vision_end = 128002
        # Two logical samples concatenated: [start, pad, end, text] + [text, text]
        input_ids = torch.tensor(
            [[vision_start, image_pad, vision_end, 7, 8, 9]],
            dtype=torch.long,
        )
        image_grid_thw = torch.tensor([[1, 1, 1]], dtype=torch.long)
        packed = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=torch.tensor([0, 4, 6], dtype=torch.int32),
            cu_seqlens_kv=torch.tensor([0, 4, 6], dtype=torch.int32),
            cu_seqlens_q_padded=torch.tensor([0, 4, 6], dtype=torch.int32),
            cu_seqlens_kv_padded=torch.tensor([0, 4, 6], dtype=torch.int32),
            max_seqlen_q=4,
            max_seqlen_kv=4,
        )
        position_ids = build_falcon_dsv4_mrope_position_ids(
            input_ids,
            spatial_merge_size=1,
            image_grid_thw=image_grid_thw,
            packed_seq_params=packed,
        )
        assert position_ids.shape == (3, 1, 6)
        # Second sample resets to 0.
        assert position_ids[:, 0, 4].tolist() == [0, 0, 0]
        assert position_ids[:, 0, 5].tolist() == [1, 1, 1]

    def test_resolve_reuses_explicit_3d(self):
        explicit = torch.arange(6).view(3, 1, 2)
        out = resolve_falcon_dsv4_mrope_position_ids(
            explicit,
            None,
            spatial_merge_size=1,
            image_token_id=FALCON_DSV4_IMAGE_TOKEN_ID,
            vision_start_token_id=FALCON_DSV4_VISION_START_TOKEN_ID,
        )
        assert out is explicit

    def test_resolve_requires_global_under_contiguous_cp(self):
        with pytest.raises(ValueError, match="Contiguous CP"):
            resolve_falcon_dsv4_mrope_position_ids(
                torch.arange(4).unsqueeze(0),
                None,
                spatial_merge_size=1,
                image_token_id=FALCON_DSV4_IMAGE_TOKEN_ID,
                vision_start_token_id=FALCON_DSV4_VISION_START_TOKEN_ID,
                require_global_positions=True,
            )

    def test_is_mrope_position_ids(self):
        assert is_mrope_position_ids(torch.zeros(3, 1, 4))
        assert not is_mrope_position_ids(torch.zeros(1, 4))
        assert not is_mrope_position_ids(None)


@pytest.mark.unit
class TestContiguousCpKeepsGlobalPositions:
    def test_position_ids_not_sliced(self):
        tokens = torch.arange(8, dtype=torch.long).unsqueeze(0)
        labels = torch.arange(8, dtype=torch.long).unsqueeze(0)
        position_ids = torch.arange(8).view(1, 1, 8).expand(3, -1, -1).clone()
        cu_seqlens = torch.tensor([0, 8], dtype=torch.int32)

        rank0 = partition_packed_batch_contiguous(
            {
                "tokens": tokens.clone(),
                "labels": labels.clone(),
                "position_ids": position_ids.clone(),
                "cu_seqlens_q": cu_seqlens.clone(),
            },
            cp_size=2,
            cp_rank=0,
        )
        assert rank0["tokens"].shape == (1, 4)
        assert rank0["position_ids"].shape == (3, 1, 8)
        assert torch.equal(rank0["position_ids"], position_ids)

        rank1 = partition_packed_batch_contiguous(
            {
                "tokens": tokens.clone(),
                "labels": labels.clone(),
                "position_ids": position_ids.clone(),
                "cu_seqlens_q": cu_seqlens.clone(),
            },
            cp_size=2,
            cp_rank=1,
        )
        assert rank1["tokens"].shape == (1, 4)
        assert rank1["position_ids"].shape == (3, 1, 8)
