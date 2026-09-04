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

"""Parsing and pipeline partitioning for Falcon DeepSeek-V4 patterns."""

_ATTENTION_TO_COMPRESS_RATIO = {"W": 0, "C": 4, "H": 128}
_COMPRESS_RATIO_TO_ATTENTION = {ratio: symbol for symbol, ratio in _ATTENTION_TO_COMPRESS_RATIO.items()}
_FFN_SYMBOLS = frozenset({"-", "E"})


def _pipeline_stage_lengths(num_decoder_layers: int, pipeline_model_parallel_size: int) -> list[int]:
    """Return decoder-layer counts per pipeline stage (earlier stages take remainder)."""
    base = num_decoder_layers // pipeline_model_parallel_size
    remainder = num_decoder_layers % pipeline_model_parallel_size
    return [base + (1 if stage_index < remainder else 0) for stage_index in range(pipeline_model_parallel_size)]


def falcon_deepseekv4_pattern_from_compress_ratios(
    compress_ratios: list[int],
    *,
    ffn_symbol: str = "E",
) -> str:
    """Build an explicit Falcon DSv4 pattern from CSA compression ratios.

    Args:
        compress_ratios: Per-decoder-layer attention compression ratios (0, 4, or 128).
        ffn_symbol: FFN component after every attention symbol (``E`` or ``-``).

    Returns:
        Hybrid layer pattern string with one attention/FFN pair per ratio.
    """
    if not compress_ratios:
        raise ValueError("compress_ratios must not be empty")
    if ffn_symbol not in _FFN_SYMBOLS:
        raise ValueError(f"Unsupported Falcon DSv4 FFN symbol {ffn_symbol!r}")

    symbols: list[str] = []
    for layer_index, compress_ratio in enumerate(compress_ratios):
        try:
            attention = _COMPRESS_RATIO_TO_ATTENTION[compress_ratio]
        except KeyError as exc:
            raise ValueError(f"Unsupported compress_ratio {compress_ratio} at layer index {layer_index}") from exc
        symbols.append(f"{attention}{ffn_symbol}")
    return "".join(symbols)


def falcon_deepseekv4_pattern_metadata(pattern: str) -> tuple[list[int], list[int]]:
    """Decode a Falcon DSv4 component pattern into decoder-layer metadata.

    A logical decoder layer is encoded as an attention symbol (``W``, ``C``,
    or ``H``) followed by either a dense MLP (``-``) or MoE (``E``). Dense and
    MoE FFN symbols may be mixed. Pipeline separators are ignored. MTP suffixes
    are intentionally excluded because they do not describe the main decoder
    stack.

    Args:
        pattern: Explicit Falcon DSv4 hybrid pattern.

    Returns:
        Per-decoder-layer CSA compression ratios and MoE flags (1 for ``E``,
        0 for ``-``).

    Raises:
        ValueError: If the pattern is not a sequence of attention/FFN pairs.
    """
    main_pattern = pattern.split("/", maxsplit=1)[0].replace("|", "")
    if not main_pattern:
        raise ValueError("Falcon DSv4 hybrid pattern must not be empty")
    if len(main_pattern) % 2 != 0:
        raise ValueError("Falcon DSv4 hybrid pattern must contain attention/FFN pairs")

    compress_ratios: list[int] = []
    moe_layer_freq: list[int] = []
    for offset in range(0, len(main_pattern), 2):
        attention = main_pattern[offset]
        feed_forward = main_pattern[offset + 1]
        if attention not in _ATTENTION_TO_COMPRESS_RATIO:
            raise ValueError(f"Unsupported Falcon DSv4 attention symbol {attention!r} at offset {offset}")
        if feed_forward not in _FFN_SYMBOLS:
            raise ValueError(f"Unsupported Falcon DSv4 FFN symbol {feed_forward!r} at offset {offset + 1}")
        compress_ratios.append(_ATTENTION_TO_COMPRESS_RATIO[attention])
        moe_layer_freq.append(1 if feed_forward == "E" else 0)
    return compress_ratios, moe_layer_freq


def falcon_deepseekv4_pipeline_pattern(
    pattern: str,
    *,
    pipeline_model_parallel_size: int,
    num_hash_layers: int = 0,
) -> str:
    """Insert pipeline-stage separators into an explicit Falcon DSv4 pattern."""
    if pipeline_model_parallel_size < 1:
        raise ValueError(f"pipeline_model_parallel_size must be >= 1, got {pipeline_model_parallel_size}")

    pattern_parts = pattern.split("/")
    main_pattern = pattern_parts[0].replace("|", "")
    compress_ratios, moe_layer_freq = falcon_deepseekv4_pattern_metadata(main_pattern)
    stage_lengths = _pipeline_stage_lengths(len(compress_ratios), pipeline_model_parallel_size)
    if stage_lengths[0] < num_hash_layers:
        raise ValueError(
            f"Pipeline stage 0 has {stage_lengths[0]} decoder layers but "
            f"num_hash_layers={num_hash_layers} requires at least that many on stage 0"
        )
    if num_hash_layers > 0 and any(flag == 0 for flag in moe_layer_freq[:num_hash_layers]):
        raise ValueError(
            f"Hash routing requires the first {num_hash_layers} decoder layers to use MoE (E) FFN symbols"
        )

    segments: list[str] = []
    offset = 0
    for stage_length in stage_lengths:
        segments.append(main_pattern[offset : offset + 2 * stage_length])
        offset += 2 * stage_length
    pipeline_pattern = "|".join(segments)
    if len(pattern_parts) == 1:
        return pipeline_pattern
    return "/".join([pipeline_pattern, *pattern_parts[1:]])
