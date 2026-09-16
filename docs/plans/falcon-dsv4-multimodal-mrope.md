# Falcon-MM DSv4 Multimodal RoPE Plan

## Status (2026-09-17)

| WP | Name | Status | Notes |
| --- | --- | --- | --- |
| WP0 | Architecture / contract | Done | See [`falcon_dsv4_multimodal_mrope_wp0_contract.md`](falcon_dsv4_multimodal_mrope_wp0_contract.md); several “current code facts” are historical — contracts still bind |
| WP1 | Reference oracle | Done | `dsv4_mrope.py` + `test_dsv4_mrope.py` (oracle, THW, inverse, compressed first-token) |
| WP2 | Hybrid position plumbing | Done | Optional `position_ids` through HybridModel / HybridStack / layers / recompute; aligns with upstream Qwen3.5 hybrid land (`#6315`) |
| WP3 | Unfused DSv4 multimodal rotary | Done | Explicit T/H/W on Q / shared KV / compressed / indexer / inverse O |
| WP4 | CSA + distributed positions | Partial | First-token compressed coords + contiguous-CP *contract* coded; **CP (incl. load-balanced indexer) not product-validated** — Falcon CP was already unproven pre-mRoPE |
| WP5 | Fused MLA/mRoPE | Done (CP1 path) | `fused_dsv4_mrope` + raw `out=` path; inverse folded into sparse Function via `MultimodalOutputRopeParams` (parity with upstream `#7036` scalar `OutputRopeParams`) |
| WP6 | Falcon-MM Bridge | Done | Provider defaults, `rope.py`, VLM step global positions, docs |
| WP7 | Integration / parity / regression | Partial | Unit + Energon E2E loss match on B200; fused multimodal slightly slower than scalar RoPE; **CP matrix not run** |
| WP8 | Upstream / submodule pin | Not started | MCore work lives on `falcon` (rebased onto `fix-csa`); Bridge still consumes a local submodule pin |

Companion handoff: [`falcon_dsv4_multimodal_mrope_wp6_wp7_report.md`](falcon_dsv4_multimodal_mrope_wp6_wp7_report.md).  
MCore developer note: `3rdparty/Megatron-LM/docs/developer/dsv4_mrope.md`.

### Upstream alignment (MCore)

Falcon mRoPE was rebased onto `fix-csa` so it shares the post-`#6315` hybrid stack and the CSA fixes that landed around it:

- **`#6315`** — Qwen3.5 → HybridModel; generic `position_ids` propagation (WP2 overlap, not redundant with DSv4 apply path).
- **`#7036`** — Recompute / fold inverse RoPE inside the fused sparse `Function` (`OutputRopeParams`). Multimodal now uses `MultimodalOutputRopeParams` with the same ownership contract.
- **`#6058`** — Load-balanced CP DSA indexer (`dsa_cp_balance_indexer`). Multimodal wires `gather_idx` physical rows into `_rope_positions`; **not E2E-validated** (same as contiguous CP for Falcon).

Notable MCore commits on `falcon` (newest first): load-balanced indexer multimodal RoPE → fold multimodal inverse into fused Function → fused explicit rotary → explicit multimodal + CSA coords → hybrid position metadata.

### Future work (post-WP7)

1. **CP validation (blocking for any CP claim)** — Contiguous THD CP and balanced indexer were never signed off for Falcon DSv4 even before mRoPE. Need CP=2+ smoke (scalar and multimodal), then balanced+mRoPE if CP is required.
2. **Close remaining WP7 gaps** — Distributed attention harness flakes; fused vs unfused A/B on the training recipe; document FLOP / throughput gap vs scalar RoPE (~130 vs ~135 TFLOP observed after rebase; MSA non-mRoPE ~150).
3. **Further inverse-RoPE perf** — `out=` removed temp+copy on the multimodal fused path; still behind true inplace scalar `mla_rope_apply_raw_`. Optional: drop `O` undo via precomputed `delta` (upstream TODO in `_undo_output_rope_for_backward`).
4. **WP8** — Land MCore PR(s), pin submodule in Bridge, refresh Falcon README / verification card; no production edits that only exist inside Bridge’s `3rdparty/` tree.
5. **Still unsupported / rejected** — Sequence parallel, configured CUDA graphs, zigzag / rank-local CP position buffers, video, MTP position shifting (unless separately scoped).
6. **Open contract questions** — Authoritative public Falcon checkpoint `mrope_section`; CSA group alignment at image/modality boundaries; E2E parity authority; FLOP-accounting policy.

## Purpose

This document is the shared execution brief for adding Qwen3.5-style multimodal rotary position embeddings (mRoPE) to the DeepSeek-v4 attention trunk used by Falcon-MM. It is intended to be read by every implementation and validation agent before starting a scoped subtask.

The document records the common goal, known architectural facts, system boundaries, integration contracts, open questions, expected result, and independent work packages. It deliberately avoids fixing implementation details that should be decided after an agent inspects the exact code and tests in its assigned area. **Prefer the Status section above for what is already landed; use the WP sections below as historical scope definitions.**

## Goal

Enable Falcon-MM with the DeepSeek-v4 trunk to consume the same temporal, height, and width position indices used by the Qwen3.5 multimodal path and apply those positions to the rotary portion of DeepSeek-v4 Q/K/V attention.

The completed system should:

- preserve the DeepSeek-v4 separation between non-rotary Q/K content and its dedicated rotary subspace;
- support Qwen3.5-style T/H/W frequency assignment for multimodal tokens;
- preserve ordinary text-only behavior when the three position axes are equal;
- preserve the DeepSeek-v4 forward and inverse rotary transformations around attention;
- work across the required DeepSeek-v4 attention modes, including sliding and compressed sparse attention;
- provide compatible fused and unfused implementations;
- work with the Falcon-MM input, packing, pipeline-parallel, and context-parallel paths selected for support;
- avoid changing checkpoint parameter layouts unless an independently justified architectural requirement is discovered;
- include enough focused correctness evidence to distinguish positional correctness from conversion, packing, distributed-layout, and kernel issues.

## Expected Result

At completion, a Falcon-MM DSv4 model can receive image/text sequences and Qwen-compatible T/H/W position IDs. The configured rotary subspace of each relevant attention layer uses those positions consistently for query, shared key/value, compressed representations, and the inverse transformation of the attention output.

Expected externally observable behavior:

- Text-only inputs produce the same behavior as scalar RoPE/YaRN within the agreed numerical tolerance because `T == H == W` for text tokens.
- Visual tokens use distinct temporal, height, and width coordinates.
- Fused and unfused paths agree within a documented tolerance for supported dtypes and layouts.
- Packed and distributed layouts associate every local Q, KV, compressed-KV, and output row with the intended original position.
- Existing DeepSeek-v4 and Falcon-MM behavior remains unchanged when multimodal RoPE is disabled.
- Provider and model configuration clearly express whether multimodal RoPE is enabled and how its rotary channels are assigned.
- Validation covers forward results, backward gradients where applicable, and a bounded end-to-end training or model-forward workflow.

## Architectural Context

### Ordinary RoPE

For rotary width `R`, RoPE has `R / 2` distinct inverse frequencies. With a 64-dimensional rotary subspace:

```text
inverse frequencies: [32]
position IDs:        [B, S] or an equivalent packed representation
angles:              [B, S, 32]
cos/sin after layout:[S, B, 1, 64]
```

The same inverse-frequency spectrum can be used to build or cache a one-dimensional coordinate lookup table. The token-aligned cos/sin tensors are produced by looking up the positions required by the current input layout.

### Multimodal RoPE

mRoPE supplies three positions per token:

```text
position IDs:    [3, B, S]
axis 0:          temporal
axis 1:          height
axis 2:          width
raw frequencies:[3, B, S, R / 2]
```

All three axes use the same inverse-frequency spectrum. A frequency-assignment rule selects which axis supplies each rotary frequency pair. For ordinary text tokens the three positions are equal, so the axis assignment reduces to ordinary RoPE.

Qwen3.5 uses an interleaved T/H/W frequency assignment. For a 64-dimensional rotary subspace, its commonly used section is `[11, 11, 10]`, whose entries sum to the 32 frequency pairs. This mRoPE-axis interleaving is distinct from the memory layout described by `rotary_interleaved`.

### Qwen3.5 and DeepSeek-v4 rotary subspaces

Qwen3.5 obtains a 64-dimensional rotary subspace as a fraction of a standard attention head. DeepSeek-v4 represents its positional Q/K subspace explicitly through its MLA/DSv4 attention geometry. Equal rotary widths do not by themselves imply equal `rotary_percent` settings: each agent must interpret that field relative to the tensor on which the corresponding rotary embedding module operates.

### DeepSeek-v4 shared K/V and inverse rotation

The current DSv4 hybrid attention path uses a shared projected representation as both key and value after applying RoPE to its positional portion. As a result, the rotary basis is carried through value aggregation. DSv4 applies the inverse rotation to the corresponding portion of the attention output before its grouped output projection.

Any multimodal extension must treat the forward and inverse transformations as one contract. Changing Q/K positions without applying the matching output-side inverse positions would be incomplete.

### Compressed sparse attention

The current CSA implementation compresses a group of tokens using learned weighting and assigns a scalar RoPE position associated with the beginning of the group. In the current contiguous scalar scheme, a group `g` with ratio `r` maps to position `g * r`.

Extending this contract to T/H/W requires an explicit decision about how a compressed representation obtains multimodal coordinates, including groups that cross image rows, images, markers, modality boundaries, padding, or packed-sample boundaries. Agents should preserve and test the current scalar behavior before proposing or implementing a different semantic rule.

Qwen Sparse Attention and DSv4 CSA should not be assumed equivalent. A block-level indexer may use a representative coordinate while final attention still operates on the original tokens. DSv4 CSA can instead create an actual compressed KV row, making its assigned position part of the attention representation itself.

### YaRN

YaRN changes the inverse-frequency spectrum and may also apply an attention/concentration scaling factor. DSv4 currently has layer-dependent rotary behavior for compressed and non-compressed attention and enforces its own rotation-magnitude contract. The multimodal design must preserve the exact layer-specific RoPE/YaRN semantics discovered in the implementation.

## Known Code Anchors

Agents must inspect the current revision rather than relying only on these paths or this summary.

### Megatron Bridge

- `src/megatron/bridge/models/falcon_mm/falcon_deepseekv4_provider.py`
- `src/megatron/bridge/models/falcon_mm/falcon_deepseekv4_bridge.py`
- `src/megatron/bridge/models/falcon_mm/modelling_falcon_mm/model_deepseekv4.py`
- `src/megatron/bridge/models/falcon_mm/data/energon.py`
- `src/megatron/bridge/models/falcon_mm/data/deepseek_v4_processor.py`
- `src/megatron/bridge/models/qwen_vl/modelling_qwen3_vl/rope.py`
- `src/megatron/bridge/models/qwen_vl/modelling_qwen3_vl/model.py`
- `src/megatron/bridge/models/qwen_vl/qwen35_vl_provider.py`
- `tests/unit_tests/models/falcon_mm/`
- `tests/unit_tests/models/qwen_vl/`

### Megatron Core submodule

- `megatron/core/models/common/embeddings/rotary_pos_embedding.py`
- `megatron/core/models/common/embeddings/yarn_rotary_pos_embedding.py`
- `megatron/core/models/common/embeddings/rope_utils.py`
- `megatron/core/fusions/fused_mrope.py`
- `megatron/core/fusions/fused_mla_yarn_rope_apply.py`
- `megatron/core/models/hybrid/hybrid_model.py`
- `megatron/core/models/hybrid/hybrid_block.py`
- `megatron/core/transformer/experimental_attention_variant/deepseek_v4_hybrid_attention.py`
- `megatron/core/transformer/experimental_attention_variant/csa.py`
- `megatron/core/transformer/experimental_attention_variant/csa_utils/`
- the focused tests adjacent to these components

The Bridge repository treats `3rdparty/Megatron-LM/` as an upstream submodule. Production MCore changes should be developed in the upstream repository or an appropriate MCore checkout and consumed by Bridge through a submodule revision update. Agents must not make Bridge-owned commits that directly modify files inside `3rdparty/Megatron-LM/`.

## Ownership Boundary

### Megatron Core owns

- generic model and hybrid-stack propagation of positional metadata;
- DSv4/MLA attention interpretation of scalar or multimodal positions;
- ordinary RoPE and YaRN frequency construction used inside DSv4 attention;
- partial-subspace forward and inverse rotary application;
- CSA compressed-position semantics and distributed layout handling;
- fused-kernel interfaces, capability checks, fallback behavior, and kernel tests;
- generic correctness behavior that can apply beyond Falcon-MM.

### Megatron Bridge owns

- enabling the feature for the Falcon-MM DSv4 model;
- mapping model/configuration inputs onto MCore configuration fields;
- generating or reusing Qwen-compatible T/H/W positions from Falcon-MM inputs;
- Falcon-specific token IDs, image grids, markers, wrapper behavior, and recipes;
- Bridge-level unit, configuration, integration, conversion, and functional tests;
- user-facing Falcon-MM documentation and examples.

The exact division can be adjusted if code inspection reveals an existing generic extension point, but model-specific conditionals should not be introduced into shared infrastructure solely for Falcon-MM.

## Shared Contracts To Establish

These contracts should be documented by the responsible agent and referenced by downstream work.

### Position representation

Define the accepted shapes and meanings for at least:

- unpacked scalar positions;
- unpacked T/H/W positions;
- packed THD positions;
- context-parallel local and global position ownership;
- padding and static-capacity rows;
- boundary rows introduced by CSA/CP communication.

The contract should specify whether a component receives original position IDs, materialized rotary embeddings, raw per-axis frequencies, gathered cos/sin values, or another representation.

### Rotary geometry

Record:

- total attention-head width;
- non-positional Q/K width;
- positional Q/K width;
- number of frequency pairs;
- the meaning of `rotary_percent` at each construction site;
- T/H/W section and assignment semantics;
- rotary coordinate layout;
- layer-specific base/scaling behavior;
- output inverse-rotation behavior.

### Compatibility and fallback

Define behavior when:

- multimodal positions are absent;
- two-dimensional scalar position IDs are supplied;
- fusion is requested but unsupported for a shape, dtype, device, or layout;
- position IDs and packed metadata disagree;
- a compressed group or boundary row cannot be assigned a valid position;
- existing text-only DSv4 configurations are loaded.

## Open Questions

These are investigation items, not decisions made by this plan. Status as of
2026-09-17 is noted where known; see WP0 contract for authoritative wording.

1. What is the authoritative Falcon-MM target configuration for rotary width, section, base, YaRN parameters, and supported attention-layer types? **Open** (dev default `[11,11,10]`).
2. Should Falcon-MM generate T/H/W positions in data preparation, in the model wrapper, or through a shared helper used by both, and how should already-provided positions be handled? **Resolved (WP6)** — wrapper + VLM step via `rope.py`.
3. What position representation should be propagated through HybridModel and HybridStack to minimize coupling while supporting recomputation, pipeline stages, and CUDA graph paths? **Resolved (WP2 / `#6315`)** — optional integer `position_ids`; CUDA graphs still rejected for explicit multimodal.
4. How should T/H/W rows be partitioned and recovered for packed THD and each supported context-parallel layout? **Contract written; Falcon CP not validated.**
5. What is the intended multimodal coordinate of a CSA-compressed group, especially when the group crosses spatial or modality boundaries? **Contract: first token; alignment policy Open.**
6. Should input construction prevent or align compression groups that cross particular boundaries? **Open.**
7. How should CSA boundary rows and compressed rows obtain their position metadata without host synchronization or incompatible CUDA graph behavior? **Implemented for eager path; graphs rejected.**
8. Should the fused MLA path consume raw per-axis frequencies, token-aligned cos/sin values, explicit position IDs plus cached tables, or another representation? **Resolved** — physical-row angles from layer spectrum + `fused_dsv4_mrope`.
9. Which existing fused mRoPE utilities can be reused without violating MLA partial-slice, inverse-rotation, or retained-output requirements? **Resolved** — dedicated DSv4 kernel; generic mRoPE only for capability checks; inverse via `MultimodalOutputRopeParams` / `#7036`.
10. What numerical tolerance is appropriate for each dtype and fused/unfused comparison? **Documented in MCore fused tests.**
11. Which combinations of SBHD, THD, CP, compression ratio, recomputation, and quantization are required for the first supported release? **CP1 SBHD/THD shipped; CP deferred until Falcon CP is validated.**
12. Does the feature change FLOP accounting materially enough to require a Bridge FLOPs-calculator update? **Open** (small observed TFLOP/s drop).
13. What reference implementation or model output is authoritative for end-to-end multimodal parity? **Open**; Energon loss-match resume is interim evidence.

## Work Packages

Each package below is intended to be assignable to one agent. An assigned agent must read this entire document, inspect the current code and tests in its scope, record assumptions, and avoid implementing another package unless coordination explicitly changes its scope.

### WP0: Architecture and contract discovery

**Objective**

Turn the shared architectural assumptions into a concise, code-grounded contract that downstream agents can implement and test.

**Inputs**

- This plan.
- Current Falcon-MM DSv4, Qwen3.5 mRoPE, DSv4 attention, CSA, YaRN, packing, and CP implementations.
- Authoritative target model configuration or reference implementation when available.

**Outputs**

- A written position-shape and rotary-geometry contract.
- A diagram or pseudocode trace from Falcon input positions through Q/K/shared-V and inverse output rotation.
- A list of confirmed support requirements and unresolved architecture questions.
- Proposed focused test cases expressed independently of a particular kernel implementation.

**Scope**

- Read-only architectural analysis and contract documentation.
- Verification of tensor dimensions, position semantics, layer variants, and existing fallbacks.

**Out of scope**

- Production implementation.
- Kernel optimization.
- Broad refactoring.

**Dependencies**

- None.

**Completion criteria**

- Downstream agents can implement against unambiguous tensor and semantic contracts.
- Every unresolved choice is explicitly marked rather than silently assumed.

### WP1: Reference rotary and mRoPE oracle

**Objective**

Create a small correctness oracle for scalar RoPE, T/H/W mRoPE, layer-specific YaRN behavior, partial-subspace rotation, and inverse rotation.

**Inputs**

- This plan and WP0 contract.
- Existing MCore RoPE, mRoPE, and YaRN implementations and tests.

**Outputs**

- Focused reference helpers or test-local implementations.
- Tests for scalar equivalence, T/H/W assignment, untouched non-rotary channels, forward/inverse behavior, and relevant dtypes/layouts.
- Explicit expected shapes and tolerances.

**Scope**

- Mathematical correctness tests and minimal reusable reference code.
- CPU tests where possible and focused GPU tests where required by existing modules.

**Out of scope**

- DSv4 attention integration.
- CSA grouping semantics.
- Production fused kernels.

**Dependencies**

- WP0 contract, or clearly documented provisional assumptions if work begins in parallel.

**Completion criteria**

- The oracle can independently judge subsequent unfused and fused implementations.
- Text positions with equal T/H/W reproduce scalar behavior.

### WP2: Hybrid positional-metadata plumbing

**Objective**

Allow the generic MCore hybrid model/stack/layer path to carry the positional metadata required by DSv4 attention without changing behavior for existing callers.

**Inputs**

- This plan and WP0 position contract.
- Current HybridModel, HybridStack, wrapped hybrid layers, transformer-layer call contracts, recomputation, PP, and CUDA graph paths.

**Outputs**

- MCore changes that propagate the agreed optional metadata.
- Focused tests for ordinary calls, wrapped layers, recomputation-relevant calls, pipeline stages, and absence of metadata.
- A short interface note for the attention-integration agent.

**Scope**

- Generic optional metadata propagation through MCore hybrid components.
- Backward-compatible call signatures and tests.

**Out of scope**

- Creating Falcon T/H/W indices.
- Applying RoPE inside DSv4 attention.
- CSA or fused-kernel changes.

**Dependencies**

- WP0 position contract.

**Completion criteria**

- A DSv4 attention layer can receive the intended metadata through the normal Falcon HybridModel path.
- Existing callers that omit it remain covered and unchanged.

### WP3: DSv4 unfused multimodal rotary integration

**Objective**

Implement the correctness-first, unfused DSv4 attention path for multimodal positions, including the matching inverse output transformation.

**Inputs**

- This plan.
- WP0 contract.
- WP1 oracle.
- WP2 interface, when available.
- Current DSv4 hybrid attention and its ordinary RoPE/YaRN behavior.

**Outputs**

- MCore attention changes for the agreed position forms.
- Forward and inverse partial-subspace rotation for applicable DSv4 attention modes.
- Focused forward and gradient tests against the oracle.
- Legacy scalar/absent-position regression tests.

**Scope**

- DSv4 Q, shared-KV, and output rotary behavior.
- Unpacked and packed layouts selected in WP0, beginning with the smallest correctness surface.
- Exact preservation of non-rotary channels.

**Out of scope**

- Production fused-kernel implementation.
- Falcon data/model configuration.
- CSA compressed-position policy beyond interfaces required to keep work separable.

**Dependencies**

- WP0 and WP1.
- WP2 for end-to-end propagation; direct layer tests may proceed earlier.

**Completion criteria**

- The unfused implementation matches the oracle.
- Forward followed by the matching inverse transformation obeys the documented contract.
- Text-only legacy behavior is preserved.

### WP4: CSA compressed and distributed position semantics

**Objective**

Extend CSA grouping, compression, boundary exchange, packed layout, and context-parallel paths so every compressed or exchanged row receives the intended positional metadata.

**Inputs**

- This plan.
- WP0 contract and its decision on compressed-group coordinates.
- WP1 oracle.
- Current CSA compressor, indexer, masking, THD layout kernels, and CP utilities.

**Outputs**

- MCore implementation of the agreed compressed-position contract.
- Tests for supported compression ratios, group boundaries, packed sequences, padding/static capacity, boundary rows, and selected CP configurations.
- Documentation of causal visibility versus assigned rotary position.

**Scope**

- Position metadata associated with compressed KV and indexer representations.
- Layout transformations required by CSA and CP.
- Consistency between fused-preparation and unfused/reference paths.

**Out of scope**

- Changing the learned compression algorithm unless separately approved.
- Falcon-specific marker or image-grid processing.
- General fused MLA rotary optimization owned by WP5.

**Dependencies**

- WP0.
- Coordination with WP3 and WP5 on their input representation.

**Completion criteria**

- Every valid compressed and boundary row has a deterministic, tested position.
- CP and non-CP paths implement the same documented semantics.
- Padding rows are safe and cannot cause invalid lookup or leakage.

### WP5: Fused MLA/mRoPE implementation

**Objective**

Provide a fused implementation of the agreed DSv4 multimodal rotary contract with safe fallback to the unfused path.

**Inputs**

- This plan.
- WP0 contract.
- WP1 oracle and tolerances.
- WP3 unfused behavior.
- WP4 distributed and compressed position representation.
- Existing fused MLA RoPE and fused mRoPE utilities.

**Outputs**

- MCore fused API and kernel changes or justified reuse of existing kernels.
- Forward, inverse, in-place/out-of-place, and gradient tests as required by the attention contract.
- Capability validation and fallback behavior for unsupported inputs.
- Focused performance measurements sufficient to detect a severe regression.

**Scope**

- Partial-subspace multimodal rotation required by MLA/DSv4.
- Supported SBHD, THD, CP, dtype, and retained-output cases defined by WP0.
- Kernel correctness and integration interface.

**Out of scope**

- Falcon position generation.
- Changing attention or CSA semantics.
- Unrelated kernel refactors.

**Dependencies**

- WP0 and WP1.
- WP3 provides the reference production behavior.
- WP4 provides compressed/distributed row semantics.

**Completion criteria**

- Fused forward and gradients match the unfused baseline within documented tolerances.
- Inverse rotation does not corrupt tensors retained by attention backward.
- Unsupported cases take an explicit, tested fallback or error path.

### WP6: Falcon-MM Bridge integration

**Objective**

Configure Falcon-MM DSv4 for multimodal RoPE and provide Qwen-compatible T/H/W positions to the MCore interface.

**Inputs**

- This plan.
- WP0 contract.
- WP2 MCore propagation interface.
- WP3/WP5 configuration and attention interfaces.
- Current Falcon-MM processor, data, model wrapper, provider, bridge, recipes, and Qwen mRoPE helpers.

**Outputs**

- Bridge/provider configuration changes.
- Falcon model/data integration for creation, acceptance, or reuse of T/H/W positions.
- Focused unit tests for provider values, position construction, text-only fallback, visual sequences, and packed inputs selected for support.
- Updated Falcon-MM documentation describing the position contract and limitations.

**Scope**

- Falcon-specific policy and integration.
- Reuse of generic Qwen-compatible position logic where its token/grid contract matches Falcon inputs.
- Backward compatibility when the feature is disabled or scalar positions are supplied.

**Out of scope**

- Reimplementing DSv4 attention in Bridge.
- Direct production edits inside the MCore submodule.
- Unrelated Falcon vision or routing changes.

**Dependencies**

- WP0 and WP2.
- An agreed MCore branch/revision exposing the WP3/WP5 interfaces.

**Completion criteria**

- Falcon image/text inputs produce the agreed T/H/W positions and deliver them to every applicable DSv4 attention layer.
- Existing Falcon text-only and configuration tests remain valid.
- Model documentation no longer states that the trunk lacks mRoPE once support is enabled.

### WP7: Integration, parity, and regression validation

**Objective**

Validate the combined MCore and MBridge implementation across the supported execution matrix and produce an evidence-backed readiness report.

**Inputs**

- This plan and WP0 contracts.
- Outputs of WP1 through WP6.
- Target model/configuration and representative text/image data.

**Outputs**

- A test matrix with commands, environment, results, tolerances, and artifacts.
- Forward and backward comparisons for fused/unfused and distributed variants.
- Text-only regression evidence.
- Multimodal functional evidence.
- A list of remaining limitations, failures, or follow-up work.

**Scope**

- Focused unit and functional tests rather than the full repository suite.
- At minimum, the combinations declared supported by WP0.
- Failure isolation across input construction, attention, CSA, CP, fusion, and Bridge integration.

**Out of scope**

- Broad feature additions while validating.
- Silently changing tolerances or semantics to make failing tests pass.

**Dependencies**

- All implementation packages required by the declared support matrix.

**Completion criteria**

- The expected result section is supported by reproducible evidence.
- Failures are attributed to a component and routed back to its owner.
- The report clearly separates correctness, numerical variance, unsupported combinations, and performance observations.

### WP8: Upstream/downstream integration and documentation

**Objective**

Coordinate the MCore contribution, Bridge submodule update, release-facing documentation, and final review without mixing ownership between repositories.

**Inputs**

- Accepted outputs from WP2 through WP7.
- MCore and Megatron Bridge contribution requirements.
- Existing Falcon-MM examples and documentation.

**Outputs**

- Reviewable MCore change set or PR with focused tests.
- Bridge submodule update after the MCore revision is available.
- Reviewable Bridge change set or PR with focused tests and documentation.
- Dependency/order notes for reviewers and CI operators.

**Scope**

- Integration sequencing, documentation, and change-set hygiene.
- Ensuring each repository contains only the changes it owns.

**Out of scope**

- Redesigning implementation choices already validated by component owners.
- Unrelated dependency or CI-workflow changes.

**Dependencies**

- WP2 through WP7 as applicable.

**Completion criteria**

- MCore and Bridge changes are independently reviewable.
- Bridge points to an MCore revision containing the required upstream capability.
- Public documentation states supported and unsupported modes accurately.

## Suggested Coordination Order

The packages are intentionally separable, but not all are independent:

```text
WP0 architecture contract
 ├── WP1 reference oracle
 ├── WP2 hybrid plumbing
 └── WP4 CSA position investigation

WP1 + WP2 ──> WP3 unfused DSv4 integration
WP3 + WP4 ──> WP5 fused integration
WP2 + WP3 ──> WP6 Falcon-MM integration
WP3 + WP4 + WP5 + WP6 ──> WP7 validation
WP7 ──> WP8 upstream/downstream integration
```

Parallel work is reasonable when agents state provisional assumptions and avoid committing incompatible public interfaces. WP0 owns resolution of cross-package semantic disagreements; component agents own decisions internal to their scope as long as they satisfy the shared contracts.

## Minimum Validation Matrix

The final supported matrix should be decided from the target requirements and implementation constraints. The validation owner should consider at least:

| Dimension | Cases to evaluate |
| --- | --- |
| Input modality | text only, one image, multiple images, video if declared supported |
| Position form | absent/default, scalar, T/H/W |
| Layout | unpacked SBHD, packed THD |
| Attention mode | non-compressed/sliding, each supported CSA compression mode |
| Rotary type | ordinary RoPE, applicable YaRN mode |
| Execution | unfused, fused |
| Parallelism | CP1, each declared CP layout, applicable PP stages |
| Autograd | forward, backward, selective recomputation where supported |
| Dtype | FP32 reference, BF16/FP16 as supported, relevant quantized execution |
| Regression | existing DSv4, existing Falcon-MM, text-only equivalence |

Not every cross-product entry must be supported. Unsupported combinations must be documented and enforced rather than left accidental.

## Shared Agent Handoff Template

Every agent should finish its package with the following information:

```markdown
### Work package
WP<number>: <name>

### Revision and environment
- Repository/revision:
- Relevant dependency revisions:
- Hardware/software requirements:

### Contract used
- Input shapes and meanings:
- Output shapes and meanings:
- Assumptions:
- Resolved open questions:
- Remaining open questions:

### Changes
- Files changed:
- Public/internal interfaces changed:
- Behavior intentionally unchanged:

### Verification
- Commands run:
- Results:
- Numerical tolerances:
- Artifacts/logs:

### Handoff
- Downstream package inputs now available:
- Known limitations:
- Follow-up owner:
```

## Guardrails

- Read the current artifacts before planning or editing; paths and behavior may have changed since this plan was written.
- Follow the repository's `AGENTS.md` and the relevant skills for model support, testing, parity, performance, and MCore dependency updates.
- Do not modify production files inside Bridge's `3rdparty/Megatron-LM/`; develop MCore changes upstream.
- Do not run the full test suite; use focused tests for the assigned package.
- Do not add dependencies, change public conversion APIs, or modify CI workflows without the approvals required by `AGENTS.md`.
- Preserve unrelated user changes in both repositories and coordinate before touching overlapping files.
- Treat exact checkpoint round-trip, forward parity, distributed correctness, and performance as separate evidence categories.
- Do not infer multimodal CSA semantics from scalar positions or from another sparse-attention architecture without confirming the intended model contract.

## Definition of Done

The overall project is done when:

1. The position and rotary contracts are documented and reflected in code.
2. Falcon-MM constructs or accepts correct Qwen-compatible T/H/W positions.
3. MCore carries those positions to every applicable DSv4 attention layer.
4. Q, shared KV, compressed representations, and inverse output rotation use mutually consistent positions and frequency parameters.
5. Fused and unfused paths satisfy the agreed correctness tolerances.
6. Packed, CSA, and distributed paths satisfy the declared support matrix.
7. Existing text-only and non-mRoPE behavior remains covered.
8. Focused end-to-end validation produces finite, stable, reproducible results.
9. MCore and Bridge changes land through their proper repository boundaries.
10. Documentation accurately states the configuration, supported modes, fallbacks, and known limitations.
