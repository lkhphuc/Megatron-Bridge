# Falcon-MM data contract and mixture reproduction

Terse reference for the Falcon-MM Energon data plane (web-document pretraining
style, native THD packing). Code: `src/megatron/bridge/models/falcon_mm/data/`.

## Sample sources → sample types

| Source | On-disk format | Energon loader | Sample type |
| --- | --- | --- | --- |
| MINT-1T-ArXiv interleaved docs | original tars, `<key>.json` + `<key>.tiff` (multi-page) | `MintInterleavedWebdataset` (`mint_webdataset.py`), indexed in place | Energon standard `InterleavedSample(sequence)` |
| Text-only docs (fineweb) | `{"text": ...}` JSONL + `.idx` | Energon built-in `DefaultCrudeJsonlDatasetFactory` | `CrudeSample` dict |
| Legacy ChatML (mock) | `conversation.json` (+ media) webdataset | `ChatMLWebdataset` | `ChatMLSample` (pass-through to Qwen encoder) |

MINT sample schema (`<key>.json`): `{"texts": [...], "images": [...], "captions": [...]}`;
`texts`/`images` are parallel lists, a non-None `images[i]` means an image sits after
`texts[i]`; the k-th non-None image = tiff page k = `captions[k]`. The loader's
`mint_interleaved_sample_loader` normalizes this into `InterleavedSample.sequence`:
`str` text parts (segments and captions) and `[C,H,W]` float image tensors in
document order.

## Encoder contract (`FalconTaskEncoder`, document path)

- Assembly: walk `sequence`; text parts verbatim, each image part →
  `<|vision_start|><|image_pad|><|vision_end|>`; blocks joined with `\n`; documents are
  never truncated.
- One processor call per sample (`min_pixels`/`max_pixels` bounds); all markers are kept,
  so the images list always matches the markers one-to-one.
- Row contract (`QwenVLPreparedSequence`): `input_ids`, `position_ids` (arange),
  `labels` = shift-left with `-100` on skipped/visual token ids, `loss_mask` =
  `labels != -100` (pretraining — full LM loss on every real-token target; ignored
  positions are excluded from both the loss numerator and the per-token denominator),
  `visual_values` over `QWEN_VISUAL_KEYS` with flattened grids. No `attention_mask`
  key — THD-only path: the pack carries per-segment causal structure in `cu_seqlens`
  and emits `attention_mask=None`; an all-ones mask would be dropped by the packer
  and a non-trivial one rejected, so the row omits it explicitly.
- Guards (all `SkipSample`, warnings rate-limited to the first 32 per worker): images >
  `max_num_images` (10), visual tokens > `max_visual_tokens` (16384), aligned length >
  `seq_length` — checked after processor tokenization, i.e. on the exact expanded token
  count (visual tokens included).
- Packing: inherited Qwen hooks (per-worker FFD over `packing_buffer_size` candidates);
  physical MBS=1; `calculate_per_token_loss=True`, `ddp.average_in_collective=False`.
- Encoder construction is polymorphic: `build_energon_task_encoder` calls
  `task_config.build_task_encoder(...)`; `FalconEnergonTaskEncoderConfig` overrides it.

## Reproduce the current mixture

```bash
# 1. MINT tars (2 shards) — used as-is
hf download mlfoundations/MINT-1T-ArXiv --repo-type dataset \
  --include "arXiv_src_0001_001_wds.tar" --include "arXiv_src_0002_001_wds.tar" \
  --local-dir workspace/data/mint1t_arxiv/raw

# 2. Index in place (writes .idx + .nv-meta only; train=0001, val=0002)
uv run python examples/models/falcon_mm/prepare_mint_index.py   # → 312 train / 382 val docs

# 3. Text-only fineweb docs (450 train / 50 val) + Energon JSONL index
uv run python - <<'EOF'
import json
from datasets import load_dataset
from megatron.energon.flavors.jsonl.jsonl_prepare import JsonlPreparator

docs = (d["text"][:12000] for d in load_dataset(
    "HuggingFaceFW/fineweb", "sample-10BT", split="train", streaming=True))
docs = [t for t in docs if len(t.strip()) >= 200][:500]
for name, rows in (("train", docs[:450]), ("val", docs[450:])):
    path = f"workspace/data/text_web/{name}.jsonl"
    with open(path, "w") as f:
        for t in rows:
            f.write(json.dumps({"text": t}) + "\n")
    JsonlPreparator.prepare_dataset(path)
EOF

# 4. Mixture YAML: workspace/data/falcon_pretrain_mixture.yaml
#    (blend_epochized; MINT x2 + text x1 ≈ 58/42; container paths /opt/Megatron-Bridge/...)

# 5. Train (defaults point at the mixture)
bash examples/models/falcon_mm/sft_energon.sh
```

## Todo: parquet input (fineweb native format)

- Large-scale pretraining text-only dataset like fineweb releases in parquet format, which is currently not supported.
  This is why we convert it to lightweight jsonl. In the future Energon 8, it will be supported by default.
- Energon support Primary & Auxiliary datasets, where the media can stay on separate auxiliary dataset and be shared
  with other primary datasets. We should really be looking into using this (e.g download OmniCorpus image separately, 
  build caption/interleave separately)
- Use Energon media files's metadata and cache pool.
- Skip dataloader process on non-first PP rank or skip heavy media processing on other ranks at least.
