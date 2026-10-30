# Qwen3-VL Energon THD performance backlog

Status: 2026-09-23. Scope: 30B-A3B MXFP8 pretraining on 8 B200 GPUs with
Energon-native 4K THD packs. These are proposed development priorities for
feature parity with the tuned text recipe, **not estimates of recoverable
throughput**.

## Measured context

| Workload | Tokens/s/GPU | Important shape differences |
| --- | ---: | --- |
| Qwen3 text mock | ~23,000 | MBS 4; no vision work; expert overlap, delayed wgrad, and TE graphs enabled |
| Qwen3-VL mock | ~6,500 | MBS 2; one image per sequence; TE graphs enabled for router and preprocessing |
| Qwen3-VL Energon THD, all-to-all | ~2,845 | MBS 1; 16 images per pack; eager execution |
| Qwen3-VL Energon THD, HybridEP | ~3,099 | Same model, 4K data, and batch shape; HybridEP dispatcher |

The 4K THD mock had 1,024 images and 224,256 real tokens per optimizer step;
200,704 of those tokens were image tokens. Only the two THD rows form a
dispatcher A/B. The text, VLM mock, and THD runs have different work per
token, so their throughput ratios do not measure the cost of any one missing
feature.

GBS 512 versus 64 primarily changes the number of gradient-accumulation
microbatches on these 8 GPUs. It is not an independent explanation for a
large steady-state tokens/s difference. MBS 4 versus 1 can affect kernel
efficiency and launch overhead and should be measured separately.

## Proposed priorities

1. **P1 — Expert-parallel communication overlap for packed VLM.**
   [Config validation](../../../../src/megatron/bridge/training/config.py)
   currently clears `overlap_moe_expert_parallel_comm` for Energon-native
   packs. The [VLM schedule-plan path](../../../../src/megatron/bridge/training/vlm_step.py)
   does not carry visual inputs or packed-sequence metadata into the combined
   schedule. Implement a VLM-aware plan that preserves both, then verify loss
   and gradients against eager packed training before benchmarking overlap.
   This is a substantial code feature, not a safe CLI override.

2. **P1 — Scoped CUDA graphs for Energon THD.** The
   [THD recipe](../../../../src/megatron/bridge/recipes/qwen_vl/b200/qwen3_vl.py)
   disables graphs, and [config validation](../../../../src/megatron/bridge/training/config.py)
   rejects them for native Energon packing. Start with TE router/preprocessing
   scopes; evaluate attention only after dynamic THD boundaries and visual
   input shapes have a capture-safe representation. Prove graph replay and
   compare steady-state time with eager on the same packed batches. The
   throughput benefit is unknown until this A/B runs.

3. **P2 — Delayed expert weight-gradient compute.** Native Energon packing
   also clears `delay_wgrad_compute`. Resolve the schedule-plan input problem
   above first, then enable this independently of expert overlap in a matched
   A/B. Do not count its potential gain twice with overlap.

4. **Investigate — Padding work in MoE dispatch.** Fixed-width THD packs mask
   padded positions for MoE auxiliary statistics, but current MCore may still
   dispatch those positions to experts. Profile this on a padded pack before
   proposing a Bridge or upstream MCore change. Record padding fraction and
   real tokens/s when changing the packing buffer or sequence length.

## Structural work and measurement rule

A trainable vision encoder and projector, image preprocessing, and the chosen
number and resolution of images are part of the VLM workload. They are not
missing optimizations. The current image-heavy THD mock is especially unlike
the one-image VLM mock. Treat its measured throughput as specific to that
mixture, not as a general THD overhead.

For each feature A/B, hold the model, trainable towers, image mixture,
sequence length, packing buffer, precision, MBS, GBS, dispatcher, and runtime
fixed except for the feature under test. Report steady-state step time,
real tokens/s/GPU, images/s/GPU, padding fraction, peak memory, and finite
loss. Profile vision, decoder, data, and optimizer time before assigning a
fraction of the text-to-VLM gap to any backlog item.
