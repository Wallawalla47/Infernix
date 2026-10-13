# Request-invariant exact route: plan

Status: design for review (2026-10-13). Temporary: remove once implemented and the contract lives
in [model quality §6.1](model-quality.md#6-numerical-acceptance-policy-and-fast-paths) and
[engine architecture](engine-architecture.md).

## 1. Contract

With `--disable-fast-paths`, a request's output (tokens and every reported probability) is a
function of its own input, sampling options, the artifact, the engine options and the build. It
does not depend on concurrency, the other requests, their timing, lane assignment, preemption,
or prefix-cache state.

Engine options (KV format, `--prefill-chunk`, speculative backend and draft count) may change
the bits: they are configuration, not load. Prompt and generated tokens need not agree: a token
computed by prefill may differ from the same token computed by decode, because GDN's chunked
prefill and recurrent decode round differently.

Decode uses 16-bit activations at every width, as a single request does today (decided
2026-10-13). The current batched 4- and 8-bit decode routes may return as qualified fast paths.

## 2. Measured baseline (2026-10-13)

Measured teacher-forced on exact log-probabilities (`infernix_decode_quality_forced`, 96 segments,
INT8 KV; `profiles/quality/invariance-27b-*`):

| Variation | Qwen3.8-27B NVFP4 | Qwen3.8-Flash-Next IQ3_S, FP8 (IQ3_S session) |
|---|---|---|
| Repeat run | identical | identical |
| Concurrency 2, 4, 8 (plain) | differs from the first batched step; mean \|Δlogp\| 0.014 at 2-4, 0.067 at 8 | identical (FP8 at 8 unmeasured) |
| MTP alone vs MTP at concurrency 4 | differs | identical |
| Prefix hit, resume on the cold run's chunk grid | identical | identical |
| Prefix hit, resume off the grid (3,000, 3,072, 5,000, 5,120 of 8,192) | differs, as much as a chunk-size change | differs |
| `--prefill-chunk` 2,048 or 8,192 vs 4,096 | differs (an option: allowed) | identical |

No variant is systematically worse; a few knife-edge positions move by up to 20 nats in every
variant.

## 3. Sources of load dependence (27B)

| Area | Mechanism | Reference |
|---|---|---|
| Decode linears | Route per column count T = batch: GEMV (T=1), SIMT (2-4), sliced-K MMA (5-16); NVFP4 MLP moves to 4-bit activations at T≥5 (gate/up) and T≥8 (down), FP8 gate/up to 8-bit at T≥5 | `linear_swiglu/nvfp4/*_plan.cpp`, `linear_add/*/*_plan.cpp`, `attn_input_proj/fp8/*_plan.cpp` |
| GDN input projection + conv | B=1 fused with the unrounded FP32 projection; B≥2 rounds the projection to BF16 first | `gdn_input_proj/fp8/fp8_gdn_conv_plan.cpp`, `gdn_input_proj.h:88-101` |
| Decode attention | Key-split target `clamp(2·SMs/(4·B))`: rows longer than 1,280 keys (B=8) to 5,376 (B=2) split differently | `causal_cache/int8/plan.cpp:26-37`, `common/causal_partition.h` |
| Verify width | DFlash2 per-batch-size tree table; tree-width controller adaptive to timing and acceptance; n-gram family and tree on/off set by the other rows | `decode.cpp:499-511, 735-796`, `tree_width_controller.cpp` |
| Verify kernels | Attention kernel instance by round width W (padded to the widest row); FP8 projections 16-bit up to T=7, 8-bit from 8 | `bf16/plan.cpp:28-41`, `gdn_input_proj.h:60-62` |
| Prefill splits | Exact taps split the chunk window; sibling coalescing inserts taps into another lane's prefill | `prefill.cpp:595-605, 701-712`, `hybrid_program.cpp:551-567` |
| Prefix resume | Snapshot chosen by cost and cache contents, resuming anywhere; endpoints include decode-computed state; KV blocks deduplicated across lineages | `binding.cpp:91`, `hybrid_program.cpp:913-923, 1233-1307`, `hybrid_cache.cpp:191-196` |
| Preemption | Victim chosen by KV pressure; replay recomputes generated tokens as chunked prefill | `engine_core.h:1836-1876`, `replay.cpp:109-347` |

Already invariant per row: norms, RoPE, the GDN recurrence (decode and verify use the same
token-by-token form), the LM head up to 8 columns, embedding, sampling and constraints. Nothing
depends on `--max-concurrency` itself: CUDA graphs are captured per exact batch size.

## 4. Design

### 4.1 Column-invariant decode and verify kernels

**Principle.** Every Op in the decode and verify phases computes each column with arithmetic
that depends only on that column's inputs: a fixed K split and accumulation order, and a fixed
activation precision. How many columns share the launch, and how they are padded, does not enter.
Tensor-core MMA columns are independent, so one sliced-K MMA instance used for every width has
this property (the LM head's `launch_tile<8,16,1>` already does, up to 8 columns).

**Linear projections.** For each decode and verify projection, one 16-bit-activation route for
every width from 1 to the widest verify round. The candidates are the existing sliced-K A16
instances, widened to cover T=1, or a multi-column GEMV that keeps the T=1 kernel's per-column
reduction (weights loaded once, one accumulator set per column). The second keeps today's
single-request bits but runs CUDA-core FMAs, so it suits plain decode (B≤8) and not wide
verification. The choice is per projection, by measurement (§5).

| Projection | Today | Exact route |
|---|---|---|
| NVFP4 MLP gate/up + SwiGLU (layers 0-55) | GEMV / SIMT / sliced-K A16, then A4 from T=5 | one A16 route for all widths |
| NVFP4 MLP down (layers 0-55) | GEMV / SIMT / sliced-K A16, then A4 from T=8 | one A16 route |
| FP8 MLP (layers 56-63) | SIMT A16, then A8 from T=5 | one A16 route |
| FP8 attention in/out and GDN out projections | GEMV / SIMT / sliced-K, A8 from 17 | one A16 route |
| GDN input projection + conv | fused FP32 at B=1, BF16-rounded at B≥2 | one form for all B; keeping the fused FP32 form keeps today's single-request bits |
| LM head | invariant to 8 columns | extend to the widest verify round |

**Attention.** Each row's key partition depends only on its own visible-key count: the split
target is today's B=1 target (85 splits on 170 SMs), whatever the batch. Its merge order is fixed.
The envelope keeps setting capacity and strides only. Verify columns use one grouped instance for
every round width, so padding a round to a wider row changes nothing for the others.

### 4.2 Speculative decoding

In the exact route a row's verify width and topology depend only on its own state:

- a fixed chain of the configured draft count, so no per-batch-size tree table and no timed
  controller;
- an n-gram proposal sized to its own row, not the batch's longest;
- trees only in configurations where every row's tree is fixed by the options.

Today's adaptive and batch-coupled choices become fast paths. With §4.1, a row's verified token
then has the same bits whatever was accepted around it. If the plain-decode and verify kernels
also share arithmetic, speculative output equals plain output bit for bit. That is optional, and
would be measured.

### 4.3 Prefill boundaries, prefix resume and replay

- **One grid.** In the exact route, a prompt's prefill calls start at multiples of the effective
  chunk from position 0. Exact taps, rewrite frontiers and sibling coalescing may not split a call
  off the grid; a tap that wants an off-grid position is taken at the grid position below it.
  Vision items still split by content, which the request determines.
- **Resume on the grid only.** A request resumes only from a snapshot at a grid position that
  prefill computed. It recomputes from there; at most one chunk of cached work is repeated.
  Endpoint snapshots holding decode-computed state are not resumed from in the exact route.
- **KV blocks keep their lineage.** A block computed by prefill on the grid holds the same bits
  as the request's own computation. Blocks from decode, verify or replay commits do not, so the
  exact route reuses only the first kind.
- **Replay reproduces the original computation.** The prompt part resumes from a grid snapshot as
  above. Generated tokens re-run as teacher-forced decode-phase columns, which reproduce the
  original bits by §4.1, rather than as chunked prefill. Several replayed tokens can share one
  verify-width pass.

These rules cost cache effectiveness, not arithmetic speed, and the prefix-caching work owns that
trade-off: its numbers come from its agentic A/B.

### 4.4 What becomes a fast path

- **`batched-low-precision-decode`:** today's 4- and 8-bit batched decode routes.
- **`batch-split-attention`:** splits by batch.
- **Adaptive speculation:** tree tables, the timed controller, batch-coupled n-gram families.
- **Free cache resume:** off-grid and endpoint resumes, and cross-lineage KV blocks.

Each is gated like any other fast path. The decode gate measures plain and speculative decode at
one row, so these routes, which act only at batch sizes above one, need the invariance harness's
concurrency arms added to the gate's fast run.

## 5. Measurements that decide the design

| Question | Measurement | Decides |
|---|---|---|
| Can one A16 route match today's T=1 GEMV and batched routes? | Op benchmark per projection at T = 1, 2, 4, 8 and verify widths: candidate route vs today's route | the route per projection (§4.1); the single-request decode cost |
| Cost of per-row attention partitions | Attention Op at B = 8, 1K-64K keys: B=1 target vs batch target | §4.1 attention |
| End-to-end cost | `infernix_bench` decode tok/s at concurrency 1, 2, 4, 8, plain and MTP; prefill unchanged | acceptance of the whole change |
| Cache cost of grid-only resume | the prefix-caching session's agentic A/B with grid-only resume | §4.3 |

Expected costs:

- **Plain decode at one row:** about 0 % if the multi-column GEMV keeps the T=1 kernel; unknown if a
  sliced-K route replaces it.
- **Concurrency 5-8:** slower than today, which runs 4-bit activations there; the size is the main
  unknown.
- **Speculative rounds:** small.
- **Exact route vs fast route:** 20-60 % slower in vLLM and SGLang's batch-invariant modes; the
  default (fast) route keeps today's speed through the fast paths of §4.4.

## 6. Verification

- **Op tests.** Each changed Op is checked against its oracle as today, plus column invariance:
  the same column's output is bit-identical at every width from 1 to the maximum, at random
  positions and with random neighbours.
- **Engine invariance tests.** A real test on the 27B: the exact route's forced log-probabilities
  are bit-identical across concurrency 1/2/4/8, MTP alone vs at concurrency 4, a prefix hit
  resuming on the grid from a prefill snapshot, a forced preemption, and a cache filled by
  unrelated requests.
- **Quality.** The exact route's bits change once, so the release-to-release identity of
  model-quality §6.1 is re-based, the floors of §6.6 are re-measured, and the decode gate's floor
  (16-bit decode at every width) is re-measured.

## 7. Order of work

1. Op measurements (§5, rows 1-2) and the route choice per projection.
2. Column-invariant linear and attention routes for plain decode, with the A16 policy at every
   width. This delivers concurrency invariance for plain decode, the most common case.
3. Speculative width and verify invariance (§4.2).
4. Grid-only prefill splits, resume and KV lineage, and decode-phase replay (§4.3), coordinated
   with the prefix-cache work.
5. The fast paths of §4.4, each qualified, so the default route keeps today's speed.
6. Contract and documentation: AGENTS.md, model-quality §6.1, engine architecture; remove this
   plan.
