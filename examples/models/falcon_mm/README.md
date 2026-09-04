# Falcon-MM data contract and mixture reproduction

Terse reference for the Falcon-MM Energon data plane (web-document pretraining
style, native THD packing). Code: `src/megatron/bridge/models/falcon_mm/data/`.

## Sample sources → cookers


| Source              | On-disk format                              | Energon loader                  | `subflavors.crude_type` | Cooker                  |
| ------------------- | ------------------------------------------- | ------------------------------- | ----------------------- | ----------------------- |
| MINT-1T-ArXiv       | tar webdataset: `<key>.json` + `<key>.tiff` | `CrudeWebdataset`               | `mint`                  | `cook_mint_interleaved` |
| CC12M (caption)     | tar webdataset: `<key>.jpg` + `<key>.txt`   | `CrudeWebdataset`               | `caption`               | `cook_caption_pair`     |
| Text-only (FineWeb) | JSONL lines `{"text": "..."}`               | crude JSONL / `CrudeWebdataset` | `text`                  | `cook_text_document`    |
| ChatML (mock)       | `conversation.json` (+ media)               | `ChatMLWebdataset`              | —                       | Qwen ChatML path        |


`energon prepare` does **not** prompt for subflavors. After prepare, edit
`.nv-meta/dataset.yaml` (or set `subflavors` on a Metadataset blend entry).

MINT json schema: `{"texts": [...], "images": [...], "captions": [...]}` —
parallel `texts`/`images`; non-`None` `images[i]` means tiff page *k* (and
`captions[k]`) follows `texts[i]`. The cooker builds
`FalconDocumentSample.sequence` with text strings and header-only
`ImagePageRef`s (no pixel decode at load).

CC12M (and other LAION-style caption WDS) is a single image–text pair per
sample: optional `.json` metadata is ignored; the cooker builds
`[ImagePageRef, caption]`.

Per-source image budgets (area = H×W, not edge length) live in `subflavors`.
Encoder defaults use early-fusion patchify
(`patch_size=32`, `spatial_merge_size=1`, `temporal_patch_size=1`,
`min_pixels = patch_size**2 * merge_size = 1024`) so tiny pages are not
aggressively upscaled. Raising `min_pixels` **upscales** small images — it
does **not** drop them. To filter out unusable tiny media, do it per source
(omit the source, or add a source-specific skip), not via a large global
`min_pixels`.

```yaml
subflavors:
  crude_type: mint
  max_pixels: 1048576   # ~1024×1024 cap (downscale)
  # min_pixels: 1024    # optional; omit to use Falcon encoder default
  max_num_images: 10
```



## 1. Prepare MINT-1T-ArXiv

```bash
# Download (full dump; use --include to pull a subset of tars if needed)
hf download --repo-type dataset mlfoundations/MINT-1T-ArXiv \
  --local-dir /mnt/datasets/MINT-1T-arxiv

# Index in place. Choose: 11. Crude sample (plain dict for cooking)
energon prepare /mnt/datasets/MINT-1T-arxiv
```

Then edit `/mnt/datasets/MINT-1T-arxiv/.nv-meta/dataset.yaml` from the stub
Energon wrote to:

```yaml
__module__: megatron.energon
__class__: CrudeWebdataset
subflavors:
  crude_type: mint
  max_pixels: 1048576
```



## 2. Prepare CC12M captions (jpg + txt WebDataset)

```bash
# Full dump is ~1.2 TB; pull a subset of shards first if needed.
hf download --repo-type dataset laion/conceptual-captions-12m-webdataset \
  --include "data/00000.tar" "data/00001.tar" \
  --local-dir /mnt/datasets/conceptual-captions-12m-webdataset

# Or a brace range (example: first 10 shards):
# hf download --repo-type dataset laion/conceptual-captions-12m-webdataset \
#   --include "data/0000{0..9}.tar" \
#   --local-dir /mnt/datasets/conceptual-captions-12m-webdataset

# Index in place. Choose: 11. Crude sample (plain dict for cooking)
# Upstream last shard data/01099.tar is missing tar end-of-archive
# (two 512-byte zero blocks); Energon fails with
# "Unexpected EOF while reading header" until you append them:
#   printf '\0%.0s' {1..1024} >> /mnt/datasets/conceptual-captions-12m-webdataset/data/01099.tar
energon prepare /mnt/datasets/conceptual-captions-12m-webdataset/data \
  --fix-duplicates --non-interactive --split-ratio 98,1,1 \
  --sample-type CrudeWebdataset --force-overwrite
```

Then edit `.nv-meta/dataset.yaml` (under the prepared path) to:

```yaml
__module__: megatron.energon
__class__: CrudeWebdataset
subflavors:
  crude_type: caption
  max_pixels: 1048576
```

Each sample is `{key}.jpg` + `{key}.txt` (+ optional `{key}.json` with width/height).



## 3. Prepare FineWeb text (parquet → JSONL)

Energon does not load FineWeb parquet natively yet; convert each parquet to
JSONL with a string `"text"` field, then prepare.

JSONL indexing is **one** `.jsonl` **file → one sibling** `.jsonl.idx`. There is no
“prepare this folder of JSONLs” command. Prefer concatenating shards into one
file for a single text source; or prepare each shard and blend paths in a
Metadataset.

```bash
# Example: small FineWeb sample dump
hf download --repo-type dataset HuggingFaceFW/fineweb \
  --include "sample/10BT/*" \
  --local-dir /mnt/datasets/fineweb/sample-10BT

# Convert every parquet under the download to {"text": ...} JSONL shards
mkdir -p /mnt/datasets/fineweb/sample-10BT-jsonl
uv run python - <<'EOF'
from pathlib import Path
import json
import pyarrow.parquet as pq

src = Path("/mnt/datasets/fineweb/sample-10BT")
dst = Path("/mnt/datasets/fineweb/sample-10BT-jsonl")
dst.mkdir(parents=True, exist_ok=True)
for parquet_path in sorted(src.rglob("*.parquet")):
    out = dst / (parquet_path.stem + ".jsonl")
    table = pq.read_table(parquet_path, columns=["text"])
    with out.open("w", encoding="utf-8") as f:
        for text in table.column("text").to_pylist():
            if not isinstance(text, str):
                continue
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
    print(f"wrote {out}")
EOF

# Option A (recommended): concat shards, then prepare the single file.
cat /mnt/datasets/fineweb/sample-10BT-jsonl/*.jsonl \
  > /mnt/datasets/fineweb/sample-10BT-all.jsonl
energon prepare /mnt/datasets/fineweb/sample-10BT-all.jsonl
# → creates /mnt/datasets/fineweb/sample-10BT-all.jsonl.idx

# Option B: index every shard without concatenating (then blend in YAML).
# for f in /mnt/datasets/fineweb/sample-10BT-jsonl/*.jsonl; do
#   uv run python -c "from megatron.energon.flavors.jsonl.jsonl_prepare import JsonlPreparator as P; \
#     print('$f', P.prepare_dataset('$f'))"
# done
```

JSONL prepare does **not** create `.nv-meta/dataset.yaml` — only the `.idx`.
Put `crude_type: text` on the Metadataset blend entry. Mixture `path` must be
the `.jsonl` **file** (Option A) or each shard path (Option B), not the folder.

## 4. Sample Metadataset mixture

Save as e.g. `/mnt/datasets/falcon_pretrain_mixture.yaml` and point training at
it with `dataset.path=/mnt/datasets/falcon_pretrain_mixture.yaml`.

Blend-entry `subflavors` merge with (and override) each dataset’s
`.nv-meta/dataset.yaml`, so you can keep dataset yamls minimal and put
`crude_type` / `max_pixels` only here if you prefer.

```yaml
__module__: megatron.energon
__class__: MetadatasetV2
splits:
  train:
    blend:
      - weight: 2.0
        path: /mnt/datasets/MINT-1T-arxiv
        subflavors:
          crude_type: mint
          max_pixels: 1048576
      - weight: 1.0
        path: /mnt/datasets/conceptual-captions-12m-webdataset/data
        subflavors:
          crude_type: caption
          max_pixels: 1048576
      - weight: 1.0
        path: /mnt/datasets/fineweb/sample-10BT-all.jsonl
        subflavors:
          crude_type: text
  val:
    blend:
      - weight: 1.0
        path: /mnt/datasets/MINT-1T-arxiv
        split_part: val
        subflavors:
          crude_type: mint
          max_pixels: 1048576
      - weight: 1.0
        path: /mnt/datasets/conceptual-captions-12m-webdataset/data
        split_part: val
        subflavors:
          crude_type: caption
          max_pixels: 1048576
      - weight: 1.0
        path: /mnt/datasets/fineweb/sample-10BT-all.jsonl
        subflavors:
          crude_type: text
```

Train (example — 8×B200 Falcon DSv4 MoE):

```bash
ENERGON_PATH=/mnt/datasets/falcon_pretrain_mixture.yaml \
  bash examples/models/falcon_mm/thd_energon.sh
```

The default recipe,
`falcon_deepseekv4_moe_35b_a3b_pretrain_8gpu_b200_bf16_config`, uses
the DSv4 Flash-Base tokenizer, TP1/PP1/EP8, selective recompute
`mla_up_proj`+`mhc`, sequence length 1024, and packing buffer 64. Full-Flash
`falcon_deepseekv4_flash_pretrain_8gpu_b200_bf16_config` is for weight-import
bring-up only. Override with `RECIPE=...` / `SEQ_LENGTH=...` as needed.
The dense 1B recipe is a smaller bring-up option; resize the mesh only after
checking model fit and the intended data semantics.



## Encoder contract (`FalconTaskEncoder`)

**Pack-then-decode** (crude → cooker → preencode → pack → postencode):

1. `preencode_sample` — assemble markers, estimate exact length from text +
  image headers, record per-page smart-resize targets → `FalconPendingSample`
  (no HF processor / no `pixel_values`).
2. `select_samples_to_pack` — FFD on exact preencode `packing_length`.
3. `postencode_sample` — decode `ImagePageRef` to the preencode smart-resize
  targets (so the processor's own resize is a no-op and lengths match by
  construction), run the processor, build `QwenVLPreparedSequence`.

- Document assembly: text verbatim; each image →
`<|vision_start|><|image_pad|><|vision_end|>`; blocks joined with `\n`.
- Visual markers are trunk-unified (see `data/visual_markers.py`): every
  image span is `[vision_start][image_pad × N][vision_end]` with single
  special tokens as markers. Qwen uses its native triple; the DSv4 trunk maps
  them to the reserved placeholders `<｜place▁holder▁no▁1｜>` /
  `<｜place▁holder▁no▁0｜>` / `<｜place▁holder▁no▁2｜>` (ids 128001 / 128000 /
  128002); `image_pad` (128000) is the model-side scatter target
  (`FalconDeepseekv4Provider.image_token_id`).
- Row: `input_ids`, `position_ids`, `labels` (shift-left, `-100` on skipped /
visual ids), `loss_mask`, `visual_values`. No `attention_mask` (THD /
`cu_seqlens` only).
- Guards (`SkipSample`): `max_num_images`, `max_visual_tokens`, length >
`seq_length`, `DecompressionBombError`.
- Recipe dataset defaults (`recipes/falcon_mm/dataset.py`):
`packing_buffer_size=16` (enables THD packing); Energon-required
`shuffle_buffer_size=100` and `max_samples_per_sequence=100` (shuffle vs
sequential-read tradeoff — set explicitly, no Energon API default).
Pack-then-decode keeps the shuffle buffer cheap, so these are not RAM
panic valves; override via CLI / `make_falcon_energon_dataset(...)` when
sample size differs a lot from typical MINT pages.



## DSv4 multimodal RoPE (mRoPE)

The Falcon DSv4 trunk enables Qwen3.5-compatible multimodal RoPE by default:

- Provider gate: ``mrope_section=[11, 11, 10]`` with ``mrope_interleaved=True``.
  Keep ``position_embedding_type="rope"`` (DSv4 uses layer-local rotary
  modules; do not switch the embedding type literal to ``mrope``).
- Position shape: ``[3, B, S]`` axes = temporal / height / width. Text-only
  tokens use equal T/H/W (scalar RoPE equivalence). Visual tokens use distinct
  H/W coordinates from ``image_grid_thw``.
- Construction: ``FalconDeepseekv4Model`` (and contiguous-CP ``vlm_step``)
  reuses Qwen's ``get_rope_index`` with DSv4 marker ids
  (``vision_start=128001``, ``image_pad=128000``). Ordinary 2D dataloader
  arange rows are ignored when mRoPE is enabled.
- Contiguous CP: tokens and visual payloads are rank-local; ``position_ids``
  stay global on every CP rank. Set ``mrope_section=None`` to restore legacy
  scalar DSv4 RoPE/YaRN without multimodal indices.

Video inputs remain rejected on this trunk.


## DSv4 hash MoE + VL routing (`moe_vl_routing`)

Text-only DeepSeek-V4 hash layers route with frozen `tid2eid[input_ids]`.
Identical `image_pad` ids would otherwise collapse every visual token onto the
same experts. Falcon opt-in `moe_vl_routing=True` (default on the shipped DSv4
multimodal recipes) keeps text on `tid2eid` and routes `image_token_id` pads
with `topk(scores + expert_bias)` using **one shared** `expert_bias` (no
`bias_vl`). Non-hash layers are unchanged and still use fused top-k with that
same bias. Hash-layer bias starts at zeros when loading a text base that has
no hash bias keys. This is **not** weight-compatible with DeepSeek Vision-Exp
dual-bias checkpoints.



## Todo

- Native parquet / Energon 8 (drop FineWeb JSONL conversion).
- Primary + Auxiliary datasets (shared media, separate caption/interleave).
- Energon media metadata + cache pool.
- Skip heavy media work on non-first PP ranks.
