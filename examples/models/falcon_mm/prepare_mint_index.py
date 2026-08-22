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

"""Index raw MINT-1T-ArXiv webdataset tars in place for Energon loading.

Usage::

    uv run python examples/models/falcon_mm/prepare_mint_index.py \
        --raw-dir workspace/data/mint1t_arxiv/raw

The raw tars are used as-is: this script only builds Energon's ``.nv-meta``
index and per-tar ``.idx`` files next to the shards, then writes the
``MintInterleavedWebdataset`` sample-loader YAML.
"""

import logging
from argparse import ArgumentParser
from pathlib import Path

from megatron.bridge.data.energon import prepare_webdataset


logger = logging.getLogger(__name__)
_DATASET_YAML = """\
__module__: megatron.bridge.models.falcon_mm.data.mint_webdataset
__class__: MintInterleavedWebdataset
subflavors: {}
"""


def prepare_mint_dataset(raw_dir: str, *, train_pattern: str, val_pattern: str, num_workers: int) -> None:
    """Index the raw tars and write the Bridge sample-loader YAML.

    Args:
        raw_dir: Directory containing the raw MINT-1T-ArXiv tar shards.
        train_pattern: Shard-path regex assigned to the ``train`` split.
        val_pattern: Shard-path regex assigned to the ``val`` split.
        num_workers: Number of parallel shard-indexing workers.
    """
    prepare_webdataset(raw_dir, {"train": train_pattern, "val": val_pattern}, num_workers=num_workers)

    metadata_dir = Path(raw_dir) / ".nv-meta"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    (metadata_dir / "dataset.yaml").write_text(_DATASET_YAML, encoding="utf-8")


def main() -> None:
    """Parse arguments and run Energon indexing."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="workspace/data/mint1t_arxiv/raw", help="Directory with raw tar shards")
    parser.add_argument("--train-pattern", default="arXiv_src_0001.*", help="Shard regex for the train split")
    parser.add_argument("--val-pattern", default="arXiv_src_0002.*", help="Shard regex for the val split")
    parser.add_argument("--num-workers", type=int, default=4, help="Workers used by Energon indexing")
    args = parser.parse_args()

    prepare_mint_dataset(
        args.raw_dir,
        train_pattern=args.train_pattern,
        val_pattern=args.val_pattern,
        num_workers=args.num_workers,
    )
    logger.info("Done. Set dataset.path=%s", args.raw_dir)


if __name__ == "__main__":
    main()
