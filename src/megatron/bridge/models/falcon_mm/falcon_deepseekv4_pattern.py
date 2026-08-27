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

"""Hybrid layer-pattern generation for DeepSeek-V4 (DSv4) Falcon-MM models."""

_COMPRESS_RATIO_TO_ATTENTION: dict[int, str] = {
    0: "W",
    4: "C",
    128: "H",
}

_MTP_SEGMENT = "WE"


def _attention_letter(compress_ratio: int, layer_index: int) -> str:
    """Map a compress ratio to its hybrid attention symbol."""
    try:
        return _COMPRESS_RATIO_TO_ATTENTION[compress_ratio]
    except KeyError as exc:
        raise ValueError(f"Unsupported compress_ratio {compress_ratio} at layer index {layer_index}") from exc


def _pipeline_stage_lengths(num_layers: int, pipeline_model_parallel_size: int) -> list[int]:
    """Return decoder-layer counts per pipeline stage (earlier stages take remainder)."""
    base = num_layers // pipeline_model_parallel_size
    remainder = num_layers % pipeline_model_parallel_size
    return [base + (1 if stage_index < remainder else 0) for stage_index in range(pipeline_model_parallel_size)]


def _layer_symbols(compress_ratios: list[int]) -> str:
    """Build the attention+MoE symbol string for all decoder layers."""
    symbols: list[str] = []
    for layer_index, compress_ratio in enumerate(compress_ratios):
        attention = _attention_letter(compress_ratio, layer_index)
        symbols.append(f"{attention}E")
    return "".join(symbols)


def falcon_deepseekv4_layer_pattern(
    compress_ratios: list[int],
    *,
    pipeline_model_parallel_size: int = 1,
    mtp_num_layers: int = 0,
    num_hash_layers: int = 0,
) -> str:
    """Build a Megatron-Core ``hybrid_layer_pattern`` string for DSv4.

    Each Hugging Face decoder layer maps to two pattern symbols: an attention
    letter (``W``, ``C``, or ``H``) followed by ``E`` (MoE MLP). Pipeline
    stage boundaries use ``|``; MTP depth segments are appended after ``/``.

    Args:
        compress_ratios: Per-layer attention compression ratios (0, 4, or 128).
        pipeline_model_parallel_size: Number of pipeline-parallel stages.
        mtp_num_layers: Number of multi-token prediction (MTP) depth layers.
        num_hash_layers: Minimum decoder layers required on pipeline stage 0
            for hash-routed MoE co-location with embeddings.

    Returns:
        Hybrid layer pattern string using only ``W``, ``C``, ``H``, ``E``,
        ``|``, and ``/``.

    Raises:
        ValueError: If inputs are invalid or constraints are violated.
    """
    if not compress_ratios:
        raise ValueError("compress_ratios must not be empty")
    if pipeline_model_parallel_size < 1:
        raise ValueError(f"pipeline_model_parallel_size must be >= 1, got {pipeline_model_parallel_size}")
    if mtp_num_layers < 0:
        raise ValueError(f"mtp_num_layers must be >= 0, got {mtp_num_layers}")

    num_layers = len(compress_ratios)
    stage_lengths = _pipeline_stage_lengths(num_layers, pipeline_model_parallel_size)
    if stage_lengths[0] < num_hash_layers:
        raise ValueError(
            f"Pipeline stage 0 has {stage_lengths[0]} decoder layers but "
            f"num_hash_layers={num_hash_layers} requires at least that many on stage 0"
        )

    layer_symbols = _layer_symbols(compress_ratios)

    if pipeline_model_parallel_size == 1:
        main_pattern = layer_symbols
    else:
        stage_segments: list[str] = []
        offset = 0
        for stage_length in stage_lengths:
            stage_segments.append(layer_symbols[offset : offset + 2 * stage_length])
            offset += 2 * stage_length
        main_pattern = "|".join(stage_segments)

    if mtp_num_layers == 0:
        return main_pattern

    mtp_suffix = "/".join([_MTP_SEGMENT] * mtp_num_layers)
    return f"{main_pattern}/{mtp_suffix}"
