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

"""DeepSeek-V4 tokenizer + Qwen3-VL image-processor shim for Falcon-MM."""

from __future__ import annotations

from typing import Any

import torch

from megatron.bridge.models.falcon_mm.data.visual_markers import (
    FALCON_IMAGE_MARKER as _FALCON_IMAGE_MARKER,
)
from megatron.bridge.models.falcon_mm.data.visual_markers import (
    VisualMarkerIds,
    resolve_dsv4_marker_ids,
)


class FalconDeepseekv4Processor:
    """Combine a DeepSeek-V4 tokenizer with a Qwen-VL image processor.

    Document text arrives with one ``FALCON_IMAGE_MARKER`` block per image
    (same assembly contract as the Qwen Falcon-MM path). Text segments are
    tokenized with ``add_special_tokens=False``; each marker is expanded into
    the same triple the Qwen processor emits natively:
    ``[vision_start_id] + [image_pad_id] * (prod(grid) // spatial_merge_size**2)
    + [vision_end_id]``.
    """

    def __init__(
        self,
        tokenizer: Any,
        image_processor: Any,
        *,
        spatial_merge_size: int = 1,
    ) -> None:
        """Initialize the processor shim.

        Args:
            tokenizer: DeepSeek-V4 ``PreTrainedTokenizerFast`` instance.
            image_processor: Qwen-VL image processor for patchify and grids.
            spatial_merge_size: Spatial merge factor for visual token counting.
        """
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.spatial_merge_size = spatial_merge_size

        self.marker_ids: VisualMarkerIds = resolve_dsv4_marker_ids(tokenizer)

    @property
    def image_token_id(self) -> int:
        """Placeholder id whose positions receive scattered patch embeddings."""
        return self.marker_ids.image_pad_id

    def __call__(
        self,
        text: list[str] | None = None,
        images: list[Any] | None = None,
        padding: bool = False,
        return_tensors: str = "pt",
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Tokenize one document and optionally patchify its images.

        Args:
            text: Single-element list containing the full document string.
            images: PIL images referenced by markers in ``text``, in order.
            padding: Ignored; only unpadded single sequences are supported.
            return_tensors: Must be ``"pt"`` when set.
            min_pixels: Optional lower bound forwarded to the image processor.
            max_pixels: Optional upper bound forwarded to the image processor.
            **kwargs: Ignored.

        Returns:
            Dict with ``input_ids`` shaped ``[1, S]``. When ``images`` is
            non-empty, also includes ``pixel_values`` and ``image_grid_thw``.

        Raises:
            ValueError: On invalid batch size, marker/image count mismatch, or
                unsupported ``return_tensors``.
        """
        del padding, kwargs

        if text is None:
            raise ValueError("text must be provided.")
        if len(text) != 1:
            raise ValueError(f"FalconDeepseekv4Processor supports batch size 1; got {len(text)}.")
        if return_tensors not in ("pt", None):
            raise ValueError(f"Unsupported return_tensors={return_tensors!r}; expected 'pt'.")

        document = text[0]
        image_list = list(images or [])
        marker_count = document.count(_FALCON_IMAGE_MARKER)
        if marker_count != len(image_list):
            raise ValueError(f"Found {marker_count} image marker(s) in text but {len(image_list)} image(s).")

        text_parts = document.split(_FALCON_IMAGE_MARKER)
        if len(text_parts) != len(image_list) + 1:
            raise ValueError("Image marker split produced an unexpected number of text segments.")

        visual_output: dict[str, Any] = {}
        image_grids: torch.Tensor | None = None
        if image_list:
            processor_kwargs: dict[str, Any] = {
                "images": image_list,
                "return_tensors": return_tensors,
            }
            if min_pixels is not None:
                processor_kwargs["min_pixels"] = min_pixels
            if max_pixels is not None:
                processor_kwargs["max_pixels"] = max_pixels
            visual_output = self.image_processor(**processor_kwargs)
            image_grids = visual_output["image_grid_thw"]

        merge_factor = self.spatial_merge_size**2
        start_id = self.marker_ids.vision_start_id
        pad_id = self.marker_ids.image_pad_id
        end_id = self.marker_ids.vision_end_id
        token_segments: list[torch.Tensor] = []
        for index, part in enumerate(text_parts):
            if part:
                segment_ids = self.tokenizer(part, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
                token_segments.append(segment_ids.to(dtype=torch.long))
            if index < len(image_list):
                assert image_grids is not None
                grid = image_grids[index]
                num_visual_tokens = int(grid.prod().item()) // merge_factor
                token_segments.append(
                    torch.cat(
                        [
                            torch.tensor([start_id], dtype=torch.long),
                            torch.full((num_visual_tokens,), pad_id, dtype=torch.long),
                            torch.tensor([end_id], dtype=torch.long),
                        ]
                    )
                )

        input_ids = torch.cat(token_segments).unsqueeze(0)
        result: dict[str, torch.Tensor] = {"input_ids": input_ids}
        if image_list:
            result["pixel_values"] = visual_output["pixel_values"]
            result["image_grid_thw"] = visual_output["image_grid_thw"]
        return result
