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

"""Image header helpers and MINT document assembly for Falcon-MM.

Header-only :func:`image_page_sizes` is the size source for every Falcon image
payload (JPEG/PNG/WebP one page, multi-page TIFF N pages). Pixel decode is
deferred until after packing via :meth:`ImageDocument.decode_rgb_pages`, which
returns RGB PIL images ready for the HF processor.

MINT shards are loaded as Energon ``CrudeWebdataset``
(``subflavors.crude_type=mint``); :func:`assemble_mint_sequence` is called from
the cooker, not from a Webdataset sample loader.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from typing import Any

from PIL import Image, ImageSequence


class _ShortReprBytes(bytes):
    """``bytes`` with a short ``__repr__`` / ``__str__`` for error dumps.

    Energon's sample formatters walk dataclass fields and call ``repr`` on
    values. A normal ``bytes`` TIFF payload would otherwise flood the terminal.
    """

    def __repr__(self) -> str:
        return f"<bytes len={len(self)}>"

    def __str__(self) -> str:
        return self.__repr__()


def image_page_sizes(data: bytes) -> tuple[tuple[int, int], ...]:
    """Read ``(width, height)`` for every page from image headers only.

    JPEG/PNG/WebP yield one page. Multi-page TIFF yields one entry per IFD.
    Does not decompress pixel data.
    """
    page_sizes: list[tuple[int, int]] = []
    with Image.open(io.BytesIO(data)) as image:
        for frame in ImageSequence.Iterator(image):
            # ``frame.size`` comes from the container header / TIFF IFD.
            page_sizes.append(frame.size)
    if not page_sizes:
        raise ValueError("Image contained no pages")
    return tuple(page_sizes)


def smart_resize(
    width: int,
    height: int,
    *,
    factor: int,
    min_pixels: int,
    max_pixels: int,
) -> tuple[int, int]:
    """Factor-align image size for packing estimates; return ``(width, height)``.

    Math matches HuggingFace / ``qwen_vl_utils`` ``smart_resize``, but the
    **argument and return order follow PIL** ``Image.size`` / ``Image.resize``
    (``width, height``). Qwen's copy is the opposite::

        # transformers / qwen_vl_utils (tensor CHW convention)
        smart_resize(height, width, ...) -> (resized_height, resized_width)
        # call site: height, width = image.shape[-2:]

        # this helper (PIL convention)
        smart_resize(width, height, ...) -> (resized_width, resized_height)
        # call site: width, height = image.size

    Decoding to exactly this size makes the processor's own resize a no-op
    (already factor-aligned and within ``[min_pixels, max_pixels]``), so the
    preencode packing-length estimate stays exact and pixels are resampled
    once (original -> target).
    """
    if min(height, width) <= 0:
        raise ValueError(f"Invalid image size {(width, height)}")
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return w_bar, h_bar


def decode_pil_page(frame: Image.Image, *, target_size: tuple[int, int] | None = None) -> Image.Image:
    """Decode one opened PIL frame to RGB, resized to ``target_size`` when given.

    ``target_size`` must be the :func:`smart_resize` output for the page so the
    HF processor's resize becomes a no-op. JPEG ``draft()`` is used when the
    target is smaller than the full frame so the decompressor can avoid a
    full-resolution RGB buffer. Other codecs still decompress then resize.
    """
    if (
        target_size is not None
        and frame.format in {"JPEG", "MPO"}
        and (target_size[0] < frame.size[0] or target_size[1] < frame.size[1])
    ):
        frame.draft("RGB", target_size)
    image = frame.convert("RGB")
    if target_size is not None and image.size != target_size:
        image = image.resize(target_size, Image.Resampling.BILINEAR)
    return image


@dataclass(frozen=True)
class ImageDocument:
    """Encoded image bytes plus per-page ``(width, height)`` from headers.

    ``page_sizes`` is header-only so cookers can budget skip / packing length
    before pixel decode. JPEG/PNG have length 1; multi-page TIFF has one entry
    per page.

    ``data`` is a :class:`bytes` subclass with a short ``__repr__`` so Energon
    error / watchdog dumps do not print raw TIFF payloads.
    """

    data: bytes
    page_sizes: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        # Frozen dataclass: wrap via object.__setattr__.
        if not isinstance(self.data, _ShortReprBytes):
            object.__setattr__(self, "data", _ShortReprBytes(self.data))

    def __repr__(self) -> str:
        return f"ImageDocument(pages={len(self.page_sizes)}, nbytes={len(self.data)}, page_sizes={self.page_sizes!r})"

    @classmethod
    def from_bytes(cls, data: bytes) -> ImageDocument:
        """Parse headers into an :class:`ImageDocument` without pixel decode."""
        payload = _ShortReprBytes(data)
        return cls(data=payload, page_sizes=image_page_sizes(payload))

    def decode_rgb_pages(self, *, target_sizes: list[tuple[int, int] | None] | None = None) -> list[Image.Image]:
        """Decode every page to an RGB PIL image.

        Each page is converted to RGB and resized to its ``target_sizes`` entry
        (the processor's smart-resize target, so the processor resize is a
        no-op). Float normalization stays deferred to the HF image processor.
        Pillow ``DecompressionBombError`` propagates.
        """
        if target_sizes is not None and len(target_sizes) != len(self.page_sizes):
            raise ValueError(f"Got {len(target_sizes)} target sizes for {len(self.page_sizes)} pages")
        pages: list[Image.Image] = []
        with Image.open(io.BytesIO(self.data)) as image:
            for page_index, frame in enumerate(ImageSequence.Iterator(image)):
                # convert() / draft copies, so pages stay valid after close.
                target = target_sizes[page_index] if target_sizes is not None else None
                pages.append(decode_pil_page(frame, target_size=target))
        if len(pages) != len(self.page_sizes):
            raise ValueError(f"Image decode produced {len(pages)} pages but headers listed {len(self.page_sizes)}")
        return pages


@dataclass(frozen=True)
class ImagePageRef:
    """One page inside a shared :class:`ImageDocument`, without pixel data."""

    document: ImageDocument
    page_index: int

    @property
    def size(self) -> tuple[int, int]:
        """Return ``(width, height)`` for this page."""
        return self.document.page_sizes[self.page_index]

    def __repr__(self) -> str:
        width, height = self.size
        return (
            f"ImagePageRef(page_index={self.page_index}, size=({width}, {height}), nbytes={len(self.document.data)})"
        )


def assemble_mint_sequence(
    *,
    texts: list[Any],
    images: list[Any],
    captions: list[Any],
    tiff_payload: ImageDocument,
    sample_key: Any = None,
) -> list[Any]:
    """Assemble one MINT document into text + :class:`ImagePageRef` parts.

    Walks the parallel ``texts``/``images`` lists: each non-empty text segment
    is appended, and each non-``None`` ``images[i]`` contributes an
    :class:`ImagePageRef` followed by its caption (when present). Raises
    ``ValueError`` when the page count and the image references disagree.
    """
    page_count = len(tiff_payload.page_sizes)
    sequence: list[Any] = []
    page_index = 0
    for i, text in enumerate(texts):
        if text:
            sequence.append(text)
        if i < len(images) and images[i] is not None:
            if page_index >= page_count:
                raise ValueError(f"MINT sample has more image references than image pages: {sample_key!r}")
            sequence.append(ImagePageRef(tiff_payload, page_index))
            if page_index < len(captions) and captions[page_index]:
                sequence.append(captions[page_index])
            page_index += 1
    if page_index != page_count:
        raise ValueError(f"MINT sample has {page_count} image pages but {page_index} image references: {sample_key!r}")
    return sequence
