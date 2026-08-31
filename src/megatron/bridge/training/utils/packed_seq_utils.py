# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from __future__ import annotations

import inspect
from dataclasses import replace

import megatron.core
import torch
from megatron.core import parallel_state
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.utils import get_batch_on_this_cp_rank
from packaging.version import Version as PkgVersion

from megatron.bridge.training.utils.visual_inputs import GenericVisualInputs


PackedMetadataValue = torch.Tensor | int | None

# Sequence-length metadata keys — excluded from token-dimension slicing.
_SEQLEN_KEYS = frozenset(
    {
        "cu_seqlens",
        "cu_seqlens_unpadded",
        "cu_seqlens_argmin",
        "cu_seqlens_unpadded_argmin",
        "max_seqlen",
        "cu_seqlens_q",
        "cu_seqlens_kv",
        "cu_seqlens_q_padded",
        "cu_seqlens_kv_padded",
        "max_seqlen_q",
        "max_seqlen_kv",
        "token_count",
        "attention_mask",
    }
)
_MIN_MCORE_THD_CP_VERSION = PkgVersion("0.18.0")


def partition_packed_batch_contiguous(
    batch: dict[str, torch.Tensor],
    cp_size: int,
    *,
    cp_rank: int | None = None,
) -> dict[str, torch.Tensor]:
    """Slice a consecutive [start, end) token window for this CP rank.

    Only data tensors (tokens, labels, loss_mask, position_ids, etc.) are sliced.
    Sequence-length metadata (cu_seqlens, max_seqlen, ...) is intentionally kept
    at global values — the DSv4 CSA compressor needs global sequence boundaries to
    correctly exchange boundary hidden states between adjacent CP ranks.
    This mirrors how zigzag mode leaves cu_seqlens untouched.

    The packed sequence length must be divisible by cp_size — ensure
    packed_sequence_size = N * cp_size when running pack_sft_data.

    Args:
        batch: Packed training batch with token tensors and optional metadata.
        cp_size: Context-parallel world size.
        cp_rank: Context-parallel rank. Uses MCore parallel state when omitted.

    Returns:
        The batch with data tensors sliced to this rank's contiguous window.
    """
    if cp_rank is None:
        cp_rank = parallel_state.get_context_parallel_rank()

    _data_val = next((v for k, v in batch.items() if v is not None and k not in _SEQLEN_KEYS), None)
    if _data_val is None:
        return batch  # middle PP stage with no data tensors — nothing to slice

    total_tokens = _data_val.size(1)
    if total_tokens % cp_size != 0:
        raise RuntimeError(
            f"Contiguous CP partitioning requires packed sequence length ({total_tokens}) "
            f"to be divisible by cp_size ({cp_size}). "
            "Set packed_sequence_size to a multiple of cp_size when running pack_sft_data."
        )
    local_len = total_tokens // cp_size
    start = cp_rank * local_len
    end = start + local_len

    for key, val in batch.items():
        if val is None or key in _SEQLEN_KEYS:
            continue
        batch[key] = val[:, start:end].contiguous()

    return batch


def _find_image_placeholder_runs(
    full_input_ids: torch.Tensor,
    image_token_id: int,
) -> list[tuple[int, int]]:
    """Return consecutive ``[start, end)`` image-placeholder spans.

    ``full_input_ids`` is already on CUDA in the VLM batch path.  Detect run
    boundaries with tensor operations and transfer only the compact boundary
    vector to the host.  A Python token-by-token scan with ``Tensor.item()``
    would otherwise introduce one CUDA synchronization per packed token.
    """
    flat = full_input_ids.reshape(-1)
    is_image = flat == image_token_id
    padded = torch.nn.functional.pad(is_image, (1, 1), value=False)
    boundaries = torch.nonzero(padded[1:] != padded[:-1], as_tuple=False).flatten()
    boundary_values = boundaries.tolist()
    return list(zip(boundary_values[0::2], boundary_values[1::2], strict=True))


def slice_visual_inputs_contiguous(
    visual_inputs: GenericVisualInputs | None,
    full_input_ids: torch.Tensor,
    start: int,
    end: int,
    *,
    image_token_id: int,
    spatial_merge_size: int,
) -> GenericVisualInputs | None:
    """Slice visual payloads to match a contiguous CP token window.

    Under contiguous context parallelism, each rank embeds only the pixel patches
    whose placeholder tokens fall in ``[start, end)`` of the full packed stream.

    Args:
        visual_inputs: Visual modality tensors, or ``None``.
        full_input_ids: Full pack token ids before CP slicing, shape ``[1, L]``.
        start: Inclusive start of this rank's token window.
        end: Exclusive end of this rank's token window.
        image_token_id: Placeholder token id for merged visual tokens.
        spatial_merge_size: Patch merger spatial merge factor.

    Returns:
        A new ``GenericVisualInputs`` with sliced ``pixel_values`` and
        ``image_grid_thw``, or ``visual_inputs`` unchanged when there is nothing
        to slice.

    Raises:
        ValueError: If placeholder run count does not match ``image_grid_thw``.
        NotImplementedError: If video fields are present.
        AssertionError: If kept placeholder tokens do not match kept patch rows.
    """
    if visual_inputs is None or visual_inputs.pixel_values is None or visual_inputs.image_grid_thw is None:
        return visual_inputs
    if visual_inputs.pixel_values_videos is not None or visual_inputs.video_grid_thw is not None:
        raise NotImplementedError("Video visual inputs are not supported for contiguous CP slicing yet.")

    pixel_values = visual_inputs.pixel_values
    image_grid_thw = visual_inputs.image_grid_thw
    merge_sq = spatial_merge_size * spatial_merge_size

    placeholder_runs = _find_image_placeholder_runs(full_input_ids, image_token_id)
    num_images = image_grid_thw.shape[0]
    if len(placeholder_runs) != num_images:
        raise ValueError(
            f"Found {len(placeholder_runs)} image placeholder runs but image_grid_thw has {num_images} rows."
        )

    kept_pixel_chunks: list[torch.Tensor] = []
    kept_grid_rows: list[torch.Tensor] = []
    kept_placeholder_tokens = 0
    patch_offset = 0
    # ``image_grid_thw`` is also already on CUDA.  Transfer every patch count
    # together instead of synchronizing once per image with ``prod().item()``.
    patch_counts = image_grid_thw.prod(dim=-1).tolist()

    for image_idx in range(num_images):
        span_start, span_end = placeholder_runs[image_idx]
        grid = image_grid_thw[image_idx]
        num_patches = patch_counts[image_idx]
        patch_start = patch_offset
        patch_end = patch_offset + num_patches
        patch_offset = patch_end

        overlap_start = max(span_start, start)
        overlap_end = min(span_end, end)
        if overlap_start >= overlap_end:
            continue

        kept_placeholder_tokens += overlap_end - overlap_start
        patch_slice_start = patch_start + (overlap_start - span_start) * merge_sq
        patch_slice_end = patch_start + (overlap_end - span_start) * merge_sq
        kept_pixel_chunks.append(pixel_values[patch_slice_start:patch_slice_end])

        if overlap_start == span_start and overlap_end == span_end:
            kept_grid_rows.append(grid)
        else:
            kept_visual_tokens = overlap_end - overlap_start
            kept_grid_rows.append(
                torch.tensor(
                    [1, 1, kept_visual_tokens * merge_sq],
                    dtype=image_grid_thw.dtype,
                    device=image_grid_thw.device,
                )
            )

    if kept_pixel_chunks:
        new_pixel_values = torch.cat(kept_pixel_chunks, dim=0)
        new_image_grid_thw = torch.stack(kept_grid_rows, dim=0)
    else:
        feature_dim = pixel_values.shape[1]
        new_pixel_values = pixel_values.new_zeros((0, feature_dim))
        new_image_grid_thw = image_grid_thw.new_zeros((0, 3))

    expected_visual_tokens = new_pixel_values.shape[0] // merge_sq
    if kept_placeholder_tokens != expected_visual_tokens:
        raise AssertionError(
            f"Kept placeholder token count ({kept_placeholder_tokens}) does not match "
            f"kept patch rows ({new_pixel_values.shape[0]}) divided by merge**2 "
            f"({expected_visual_tokens})."
        )

    return replace(
        visual_inputs,
        pixel_values=new_pixel_values,
        image_grid_thw=new_image_grid_thw,
    )


def get_thd_cp_partition_indices(
    cu_seqlens: torch.Tensor,
    *,
    total_tokens: int,
    cp_group: torch.distributed.ProcessGroup,
    device: torch.device,
) -> torch.Tensor:
    """Return MCore's context-parallel partition indices for a THD stream.

    Args:
        cu_seqlens: Physical cumulative sequence offsets for the packed stream.
        total_tokens: Total padded token count before CP partitioning.
        cp_group: Context-parallel process group.
        device: Device on which the returned indices will be consumed.

    Returns:
        Long tensor containing this CP rank's indices into the full stream.

    Raises:
        RuntimeError: If the installed Megatron-Core version does not expose
            THD partitioning through ``get_batch_on_this_cp_rank``.
    """
    mcore_version = PkgVersion(megatron.core.__version__)
    supports_thd_partitioning = "is_hybrid_cp" in inspect.signature(get_batch_on_this_cp_rank).parameters
    if mcore_version < _MIN_MCORE_THD_CP_VERSION or not supports_thd_partitioning:
        raise RuntimeError(
            "THD context-parallel partitioning requires Megatron-Core >= 0.18.0 with "
            "get_batch_on_this_cp_rank(..., is_hybrid_cp=...); "
            f"found {megatron.core.__version__}. Please upgrade Megatron-Core."
        )

    cu_seqlens = cu_seqlens.to(device=device)
    index_batch = {
        "tokens": torch.arange(total_tokens, device=device, dtype=torch.long).unsqueeze(0),
        "cu_seqlens": cu_seqlens.unsqueeze(0) if cu_seqlens.dim() == 1 else cu_seqlens,
        "cu_seqlens_padded": None,
    }
    partitioned_batch = get_batch_on_this_cp_rank(index_batch, is_hybrid_cp=False, cp_group=cp_group)
    return partitioned_batch["tokens"].squeeze(0).to(device=device, dtype=torch.long)


def get_packed_seq_q_cu_seqlens(
    packed_seq_params: PackedSeqParams,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Return unpadded and physical query cumulative offsets.

    Args:
        packed_seq_params: MCore THD sequence metadata.

    Returns:
        Unpadded query offsets and physical offsets. Physical offsets use the
        padded metadata when available and otherwise fall back to unpadded offsets.
    """
    cu_seqlens = packed_seq_params.cu_seqlens_q
    cu_seqlens_padded = getattr(packed_seq_params, "cu_seqlens_q_padded", None)
    if cu_seqlens_padded is None:
        cu_seqlens_padded = cu_seqlens
    return cu_seqlens, cu_seqlens_padded


def get_packed_seq_cp_partition_indices(
    packed_seq_params: PackedSeqParams,
    *,
    total_tokens: int,
    cp_size: int,
    cp_rank: int,
    device: torch.device,
    cp_group: torch.distributed.ProcessGroup | None = None,
) -> torch.Tensor:
    """Return MCore's partition indices for packed CP.

    Args:
        packed_seq_params: MCore THD metadata for the full packed stream.
        total_tokens: Total padded token count before CP partitioning.
        cp_size: Context-parallel world size.
        cp_rank: Context-parallel rank.
        device: Device on which the returned indices will be consumed.
        cp_group: Context-parallel process group. Uses MCore parallel state when omitted.

    Returns:
        Long tensor containing this CP rank's indices into the full stream.

    Raises:
        ValueError: If packed query boundaries are unavailable or the requested
            rank and size do not match the context-parallel group.
    """
    _, cu_seqlens = get_packed_seq_q_cu_seqlens(packed_seq_params)
    if cu_seqlens is None:
        raise ValueError("Packed CP partitioning requires cu_seqlens_q metadata.")

    if cp_size < 1 or not 0 <= cp_rank < cp_size:
        raise ValueError(f"Invalid context-parallel rank {cp_rank} for size {cp_size}.")
    if cp_group is not None and (cp_group.size() != cp_size or cp_group.rank() != cp_rank):
        raise ValueError(
            f"Context-parallel group has rank {cp_group.rank()} and size {cp_group.size()}, "
            f"but rank {cp_rank} and size {cp_size} were requested."
        )
    if cp_size == 1:
        return torch.arange(total_tokens, device=device, dtype=torch.long)
    if cp_group is None:
        cp_group = parallel_state.get_context_parallel_group()
        if cp_group.size() != cp_size or cp_group.rank() != cp_rank:
            raise ValueError(
                f"Context-parallel group has rank {cp_group.rank()} and size {cp_group.size()}, "
                f"but rank {cp_rank} and size {cp_size} were requested."
            )
    return get_thd_cp_partition_indices(
        cu_seqlens,
        total_tokens=total_tokens,
        cp_group=cp_group,
        device=device,
    )


def unpack_mcore_thd_tensor_for_position_ids(
    tensor: torch.Tensor,
    packed_seq_params: PackedSeqParams,
) -> tuple[torch.Tensor, torch.Tensor, list[int], list[int]]:
    """Reconstruct logical rows from a single-row MCore THD tensor.

    This is intended for model-specific position-ID builders that require a
    conventional batch dimension. Attention still consumes the original THD
    tensor and metadata.

    Args:
        tensor: Packed tensor with shape ``[1, total_padded_tokens]``.
        packed_seq_params: Current MCore THD sequence metadata.

    Returns:
        Padded logical rows, their boolean attention mask, padded row starts,
        and unpadded row lengths.

    Raises:
        ValueError: If the tensor or packed metadata is inconsistent.
    """
    if tensor.dim() != 2 or tensor.size(0) != 1:
        raise ValueError("MCore THD position preparation expects a tensor with shape [1, total_tokens].")
    cu_seqlens, cu_seqlens_padded = get_packed_seq_q_cu_seqlens(packed_seq_params)
    if not isinstance(cu_seqlens, torch.Tensor) or cu_seqlens.dim() != 1 or cu_seqlens.numel() < 2:
        raise ValueError("MCore THD position preparation requires 1D cu_seqlens_q metadata.")
    if not isinstance(cu_seqlens_padded, torch.Tensor) or cu_seqlens_padded.shape != cu_seqlens.shape:
        raise ValueError("cu_seqlens_q_padded must match cu_seqlens_q when provided.")

    lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    padded_starts = cu_seqlens_padded[:-1].tolist()
    if not lengths or any(length <= 0 for length in lengths):
        raise ValueError("MCore THD position preparation requires non-empty packed rows.")
    if any(start < 0 or start + length > tensor.size(1) for start, length in zip(padded_starts, lengths)):
        raise ValueError("Packed sequence metadata exceeds the THD tensor length.")

    max_length = max(lengths)
    rows = torch.zeros((len(lengths), max_length), dtype=tensor.dtype, device=tensor.device)
    attention_mask = torch.zeros((len(lengths), max_length), dtype=torch.bool, device=tensor.device)
    for row_idx, (start, length) in enumerate(zip(padded_starts, lengths)):
        rows[row_idx, :length] = tensor[0, start : start + length]
        attention_mask[row_idx, :length] = True
    return rows, attention_mask, padded_starts, lengths


def repack_mcore_thd_position_ids(
    position_ids: torch.Tensor,
    *,
    padded_starts: list[int],
    lengths: list[int],
    total_length: int,
) -> torch.Tensor:
    """Scatter logical-row MRoPE positions back into a single THD row.

    Args:
        position_ids: Position tensor with shape ``[axes, rows, max_length]``.
        padded_starts: Start offset of each row in the padded THD tensor.
        lengths: Unpadded length of each logical row.
        total_length: Padded THD tensor length.

    Returns:
        Position tensor with shape ``[axes, 1, total_length]``. Alignment gaps
        remain zero because they are excluded by packed metadata and loss masks.

    Raises:
        ValueError: If row metadata and position IDs are inconsistent.
    """
    if position_ids.dim() != 3 or position_ids.size(1) != len(lengths):
        raise ValueError("Logical-row position IDs must have shape [axes, rows, max_length].")
    if len(padded_starts) != len(lengths):
        raise ValueError("Packed row starts and lengths must contain the same number of entries.")

    packed_position_ids = torch.zeros(
        (position_ids.size(0), 1, total_length),
        dtype=position_ids.dtype,
        device=position_ids.device,
    )
    for row_idx, (start, length) in enumerate(zip(padded_starts, lengths)):
        packed_position_ids[:, 0, start : start + length] = position_ids[:, row_idx, :length]
    return packed_position_ids


def _squeeze_metadata(value: PackedMetadataValue) -> PackedMetadataValue:
    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        return value
    return value.squeeze()


def _as_python_int(value: PackedMetadataValue, *, field_name: str) -> int | None:
    """Normalize scalar packed-sequence metadata to MCore's integer contract."""
    value = _squeeze_metadata(value)
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"{field_name} must contain exactly one value, got shape {tuple(value.shape)}")
        return int(value.item())
    return int(value)


def _as_python_bool(value: PackedMetadataValue, *, field_name: str) -> bool | None:
    """Normalize scalar packed-sequence metadata to MCore's boolean contract."""
    value = _squeeze_metadata(value)
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"{field_name} must contain exactly one value, got shape {tuple(value.shape)}")
        return bool(value.item())
    return bool(value)


def get_packed_seq_params(batch: dict[str, PackedMetadataValue]) -> PackedSeqParams:
    """Build packed sequence parameters from a batch dictionary.

    Current MCore-style metadata is passed through directly after squeezing
    possible batch dimensions. Legacy Bridge metadata is still converted by
    removing any padding marked by -1 values.

    Args:
        batch: A dictionary containing packed-sequence metadata. Current keys
            are ``cu_seqlens_q``, ``cu_seqlens_kv``, optional padded variants,
            ``max_seqlen_q``, ``max_seqlen_kv``, and optional ``total_tokens``
            (required for hybrid SSM/Mamba models to generate ``seq_idx``).
            Legacy ``cu_seqlens`` / ``cu_seqlens_unpadded`` batches are also
            accepted for offline packed SFT compatibility.

    Returns:
        PackedSeqParams with identical q/kv parameters and `qkv_format` set to
        "thd".
    """
    if "cu_seqlens_q" in batch:
        cu_seqlens_q = _squeeze_metadata(batch["cu_seqlens_q"])
        cu_seqlens_kv = _squeeze_metadata(batch.get("cu_seqlens_kv"))
        max_seqlen_q = _as_python_int(batch.get("max_seqlen_q"), field_name="max_seqlen_q")
        max_seqlen_kv = _as_python_int(batch.get("max_seqlen_kv"), field_name="max_seqlen_kv")
        return PackedSeqParams(
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv if cu_seqlens_kv is not None else cu_seqlens_q,
            cu_seqlens_q_padded=_squeeze_metadata(batch.get("cu_seqlens_q_padded")),
            cu_seqlens_kv_padded=_squeeze_metadata(batch.get("cu_seqlens_kv_padded")),
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv if max_seqlen_kv is not None else max_seqlen_q,
            total_tokens=batch.get("total_tokens"),
            qkv_format="thd",
            **(
                {"pad_between_seqs": _as_python_bool(batch.get("pad_between_seqs"), field_name="pad_between_seqs")}
                if hasattr(PackedSeqParams, "pad_between_seqs")
                else {}
            ),
            # cp_partition_mode available in dev MCore only.
            **(
                {
                    "cp_partition_mode": batch.get("cp_partition_mode", "zigzag"),
                }
                if hasattr(PackedSeqParams, "cp_partition_mode")
                else {}
            ),
        )

    cu_seqlens_padded = batch["cu_seqlens"].squeeze()
    cu_seqlens_unpadded = batch.get("cu_seqlens_unpadded")
    if cu_seqlens_unpadded is not None:
        cu_seqlens_unpadded = cu_seqlens_unpadded.squeeze()

    cu_seqlens_argmin = batch.get("cu_seqlens_argmin")
    cu_seqlens_unpadded_argmin = batch.get("cu_seqlens_unpadded_argmin")

    # note: if argmin is not pre-computed in the dataloader, torch.argmin here will incur a
    # device-to-host synchronization, which can slow down training
    if cu_seqlens_argmin is not None:
        cu_seqlens_padded = cu_seqlens_padded[: cu_seqlens_argmin.item()]
    else:
        cu_seqlens_padded = cu_seqlens_padded[: torch.argmin(cu_seqlens_padded)]

    if cu_seqlens_unpadded is not None:
        if cu_seqlens_unpadded_argmin is not None:
            cu_seqlens_unpadded = cu_seqlens_unpadded[: cu_seqlens_unpadded_argmin.item()]
        else:
            cu_seqlens_unpadded = cu_seqlens_unpadded[: torch.argmin(cu_seqlens_unpadded)]

    max_seqlen = _as_python_int(batch.get("max_seqlen"), field_name="max_seqlen")
    total_tokens = batch.get("total_tokens")

    # When cu_seqlens_unpadded is present (pad_seq_to_mult > 1), pass both unpadded and padded
    # for proper THD CP support. Otherwise, just use cu_seqlens_padded to avoid slower TE kernel.
    if cu_seqlens_unpadded is not None:
        return PackedSeqParams(
            cu_seqlens_q=cu_seqlens_unpadded,
            cu_seqlens_kv=cu_seqlens_unpadded,
            cu_seqlens_q_padded=cu_seqlens_padded,
            cu_seqlens_kv_padded=cu_seqlens_padded,
            max_seqlen_q=max_seqlen,
            max_seqlen_kv=max_seqlen,
            total_tokens=total_tokens,
            qkv_format="thd",
            # cp_partition_mode available in dev MCore only.
            **(
                {
                    "cp_partition_mode": batch.get("cp_partition_mode", "zigzag"),
                }
                if hasattr(PackedSeqParams, "cp_partition_mode")
                else {}
            ),
        )
    else:
        return PackedSeqParams(
            cu_seqlens_q=cu_seqlens_padded,
            cu_seqlens_kv=cu_seqlens_padded,
            max_seqlen_q=max_seqlen,
            max_seqlen_kv=max_seqlen,
            total_tokens=total_tokens,
            qkv_format="thd",
            # cp_partition_mode available in dev MCore only.
            **(
                {
                    "cp_partition_mode": batch.get("cp_partition_mode", "zigzag"),
                }
                if hasattr(PackedSeqParams, "cp_partition_mode")
                else {}
            ),
        )
