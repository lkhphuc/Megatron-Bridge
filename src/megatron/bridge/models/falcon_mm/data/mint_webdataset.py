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

"""Energon sample loader for MINT-1T-ArXiv image-text interleaved documents.

Each raw webdataset sample stores a ``<key>.json`` document metadata blob of
the form ``{"texts": [...], "images": [...], "captions": [...]}`` (``texts``
and ``images`` are parallel lists; a non-``None`` ``images[i]`` means an image
follows ``texts[i]``) plus a multi-page ``<key>.tiff`` whose k-th page is the
k-th non-``None`` image with caption ``captions[k]``.

The loader normalizes that layout into Energon's standard
:class:`~megatron.energon.flavors.interleaved.InterleavedSample`: a
``sequence`` of ``str`` (text segments and captions) and ``torch.Tensor``
(RGB images, ``[C, H, W]`` float32 in ``[0, 1]``) parts in document order.
"""

import inspect
import io
import re
from typing import Any, Union

import numpy as np
import torch
from megatron.energon.epathlib.epath import EPath
from megatron.energon.flavors.interleaved import InterleavedSample
from megatron.energon.flavors.webdataset import DefaultDecoderWebdatasetFactory
from PIL import Image, ImageSequence
from webdataset.autodecode import Decoder


class tiffhandler:
    """Webdataset decoder handler for multi-page ``.tiff``/``.tif`` members."""

    EXTENSIONS = {"tiff", "tif"}

    def __call__(self, key, data):
        """Decode tiff bytes into a list of RGB ``[C, H, W]`` float tensors.

        Returns ``None`` for keys that are not tiff members.
        """
        extension = re.sub(r".*[.]", "", key).lower()
        if extension not in self.EXTENSIONS:
            return None
        pages = []
        with Image.open(io.BytesIO(data)) as image:
            for frame in ImageSequence.Iterator(image):
                # convert() copies, so pages stay valid after the file closes.
                rgb = np.asarray(frame.convert("RGB"), dtype=np.float32) / 255.0
                pages.append(torch.from_numpy(rgb).permute(2, 0, 1))
        return pages


def mint_interleaved_sample_loader(sample: dict[str, Any]) -> dict[str, Any]:
    """Assemble the ``InterleavedSample.sequence`` for one MINT document.

    Walks the parallel ``texts``/``images`` lists: each non-empty text segment
    is appended, and each non-``None`` ``images[i]`` contributes the next tiff
    page followed by its caption (when present). Raises ``ValueError`` when the
    page count and the image references disagree.
    """
    meta = sample["json"]
    pages = sample["tiff"]
    texts = meta.get("texts") or []
    images = meta.get("images") or []
    captions = meta.get("captions") or []
    sequence: list[Union[torch.Tensor, str]] = []
    page_index = 0
    for i, text in enumerate(texts):
        if text:
            sequence.append(text)
        if i < len(images) and images[i] is not None:
            if page_index >= len(pages):
                raise ValueError(f"MINT sample has more image references than tiff pages: {sample.get('__key__')!r}")
            sequence.append(pages[page_index])
            if page_index < len(captions) and captions[page_index]:
                sequence.append(captions[page_index])
            page_index += 1
    if page_index != len(pages):
        raise ValueError(
            f"MINT sample has {len(pages)} tiff pages but {page_index} image references: {sample.get('__key__')!r}"
        )
    return {"sequence": sequence}


class MintInterleavedWebdataset(DefaultDecoderWebdatasetFactory[InterleavedSample]):
    """Webdataset factory for MINT-1T-ArXiv interleaved document samples.

    Decodes raw shards in place (no tar rewriting) into Energon's standard
    ``InterleavedSample``: the ``json`` member is decoded to a dict by
    webdataset's built-in basic handlers, the ``tiff`` member is decoded by
    :class:`tiffhandler`, and :func:`mint_interleaved_sample_loader` assembles
    the interleaved sequence. The sample loader is hardcoded (like
    ``CrudeWebdataset``), so ``.nv-meta/dataset.yaml`` needs no ``field_map``.

    Args:
        path: Root path of the webdataset shards.
        auto_decode: Whether to install the custom tiff decoder.  Passed
            through to the parent class.
        **kwargs: Forwarded to ``DefaultDecoderWebdatasetFactory.__init__``.
            ``decoder``, ``auto_decode``, and ``field_map`` keys, if present,
            are silently dropped because this class installs its own.
    """

    __sample_type__ = InterleavedSample

    def __init__(self, path: EPath, *, auto_decode: bool = True, **kwargs):
        kwargs.pop("decoder", None)
        kwargs.pop("auto_decode", None)
        kwargs.pop("field_map", None)
        parent_parameters = inspect.signature(DefaultDecoderWebdatasetFactory.__init__).parameters
        if "decoder" in parent_parameters:
            # Energon 7.4+ accepts a decoder object directly.
            kwargs["decoder"] = None
        else:
            # Earlier 7.x releases construct their optional PyAV decoder when
            # auto_decode is true, even for image-only datasets.
            kwargs["auto_decode"] = False
        super().__init__(
            path,
            sample_loader=mint_interleaved_sample_loader,
            part_filter=lambda part: part in ("json", "tiff"),
            **kwargs,
        )
        if auto_decode:
            self._decoder = Decoder([tiffhandler()])
