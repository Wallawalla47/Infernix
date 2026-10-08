# Linear tuning and performance reports

This document owns the tuning extents, workload priorities, dispatch tradeoffs, and final report
format for Linear and related token projections. [Op development](op-development.md) owns the
shared implementation boundaries, candidate workflow, and numerical qualification rules.
[Linear benchmark](linear-benchmark.md) defines the logical metrics;
[bench/README.md](../../bench/README.md#linear-op-benchmark) provides executable commands.

## 1. Workload and tuning extent

For Linear and related token projections, the latency-sensitive **hot interval** is `1 <= T <= 128`,
where `T` is the token-column extent of the actual Op call. This covers a target verification block
of 16 columns (15 drafts plus one anchor) across eight concurrent requests. For other Ops, derive
the measured axes and extents from their actual public inputs and product workload; preserve any
batch, sequence, or state dimensions required by their semantics.

The large-T optimization anchors for Linear and related token projections are `T=512` and
`T=1024`; report both separately. These tuning targets do not restrict supported extents. Other
valid extents, including `129..511` and values above 1024, use the selected broad routes and receive
supporting measurements where a transition or a specific performance question warrants them.

## 2. Small-T priorities

For Linear hot-interval tuning, use the following ordered priorities rather than a single weighted
score. Apply them to valid extents of the actual Op and the workload it serves.

| Priority | Token extents | Selection objective |
|---|---|---|
| 1 | `T=1` | Pursue the best repeatable measured latency independently; gains elsewhere do not compensate for a material regression here. A dedicated implementation is appropriate. |
| 2 | `T=4,8` | Optimize and report each core speculative point separately. Prefer a shared implementation when performance is equivalent. |
| 3 | Other concurrency-derived points below | Cover them efficiently with a small set of capacities, tiles, and broad intervals; add a useful schedule when these points materially lag. |
| 4 | Remaining valid integers through 128 | Maintain efficient interval coverage; modest local costs can justify fewer instances and branches. |

For active concurrency `B=1..8`, ordinary decode and proposal calls that process one column per
request contribute `T=B`. Target verification with three drafts contributes `T=4B`; seven drafts
contribute `T=8B`; fifteen drafts contribute `T=16B`. Their union gives the priority points:

- Ordinary decode / per-request proposal columns: `1,2,3,4,5,6,7,8`.
- MTP with three drafts: `4,8,12,16,20,24,28,32`.
- DFlash/DFlash2 with seven drafts: `8,16,24,32,40,48,56,64`.
- DFlash/DFlash2 with fifteen drafts: `16,32,48,64,80,96,112,128`.

Use the highest applicable priority for an overlapping point. Proposal and fused/stateful Ops use
their own actual extents; verification width does not describe every call in a speculative round.
For workloads such as Vision, choose priority points from their actual use rather than applying
the speculative workload priorities solely because they share a Linear geometry.

## 3. Dispatch and interval quality

Reusable templates live in the format directories under [src/ops/linear/](../../src/ops/linear/).
Schedules describe computation organization, operands provide explicit inputs, launchers validate
constraints and launch kernels, and Output/Epilogue define output mapping and fused computation.
Each semantic Op selects its own production instances. A template's availability neither registers
a public shape nor implies that production dispatch uses it.

CTA-local sliced-K reduces partial sums within a block. Cross-CTA split-K requires caller-owned
workspace and a final reduction; apply the epilogue once, after the complete reduction.
New configuration/epilogue combinations receive development qualification; permanent tests cover
public behavior and selected production routes.

Measure every valid integer in the hot interval, respecting the Op's alignment contract. A
temporary private-launcher sweep can establish candidate crossovers and the pointwise performance
envelope. Dense measurement does not require dense dispatch or matching the fastest candidate at
every point. Prefer capacity or interval implementations beyond the core points; an isolated small
gain at an ordinary point does not justify another route. Full/tail specializations are useful
when their repeatable benefit justifies them. Review compiled instance count separately from host
branch count: generating many exact-T instances through a compact template still carries cost.
Each retained specialization or boundary should identify the priority point or useful interval it
protects and the measured cost of merging it. Set neither a universal route-count limit nor a
universal percentage threshold; record absolute latency, relative changes, measurement uncertainty,
and the implementation complexity relevant to the choice.

During candidate selection, review the full pointwise curve, core points individually, affected
priority points and intervals,
and the largest adjacent-extent increase and route/schedule seams. Per-priority normalized averages
may summarize results but do not replace those comparisons. Reasonable tile-count and CTA-wave
steps, and modest capacity-tail overhead, are acceptable within the hot interval. Prioritize broad
inefficient intervals and poorly served priority points; interval smoothness is not a completion
requirement. Explain material steps and retained tradeoffs using the available evidence, with
targeted candidate comparisons when they can change the selection. Never omit, interpolate over,
or replace an observed point with an invented value.

Beyond the hot interval, optimize both `T=512` and `T=1024` for Linear and related projections;
for other Ops, select a small number of anchors from their actual bulk workload. Compare each
anchor separately so an average cannot hide a regression at the other. Optimize throughput against
the roofline of the execution resource used by the selected route. Use sparse supporting points
around relevant transitions and as few broad routes as the evidence permits; a reasonable transition discontinuity is acceptable here. A permissive public
policy does not prove that a particular accelerator route ran, so roofline evidence must identify
and measure the implementation that production dispatch actually selects. These are completion
requirements for the large-extent region, not a mandatory position in the development order.

## 4. Report format

This section applies when a retained absolute-performance report is requested; routine changes do
not require a new report or plot. Task summaries follow [AGENTS.md](../../AGENTS.md#verification-and-reporting),
including comparative results, regressions, and limitations.

A retained Linear performance report describes the final implementation's absolute performance.
Use the [Q4 6144×5120 report](examples/q4-linear.md) as a worked example, with this structure:

Reports for shapes registered later live beside it; the
[Q6 34816×5120 report](examples/q6-linear.md) records the fused gate/up projection at Q6.

1. State the format/layout, N/K, input/output types, activation policy, GPU/toolchain, timing
   boundary, cache conditions, warmup/repetitions, and latency statistic.
2. Plot the final latency at every valid T through 128 on linear axes, marking priority points
   and emphasizing 1/4/8. Show 512 and 1024 separately. Preserve measured steps and retain the
   figure as SVG alongside the report document.
3. Tabulate priority points and both bulk anchors: latency, logical GB/s, bandwidth utilization,
   useful TFLOP/s, and Tensor Core utilization. Use the existing benchmark's logical byte/FLOP
   accounting and explicitly name the hardware peak denominators. T=1 emphasizes bandwidth;
   512/1024 emphasize Tensor Core efficiency. Use the dense peak for the actual MMA input and
   accumulator precision; show a dash when the selected path does not use Tensor Cores. Label a
   sustained-read reference separately from the nominal bandwidth reference.
4. Briefly explain the final curve's material steps and known limitations, summarize numerical
   qualification, and provide reproduction commands and links to the implementation.

These utilization figures use logical workload divided by elapsed time and the stated peak,
rather than hardware counters. Retained reports omit old-implementation comparisons, speedup
ratios, aggregate improvement scores, and candidate-search history. Candidate measurements remain
working evidence for dispatch decisions; the report presents the resulting implementation.

## 5. BF16 Vision tower problems

The BF16 Vision towers of Qwen3.8-Flash-Next and of 27B artifacts that keep Vision in BF16 (for
example Quasar) register eight contiguous BF16 problems: patch embedding `[1152,1536]`, fused QKV
`[3456,1152]`, attention output `[1152,1152]`, MLP fc1 `[4304,1152]` and fc2 `[1152,4304]`,
merger fc1 `[4608,4608]`, and merger fc2 `[2560,4608]` (Flash-Next) or `[5120,4608]` (27B). The
tower encodes one item per call, so T is that item's raw-patch count P for the first five problems
and its merged-token count V = P/4 for the merger.

| Extent | Source |
|---|---|
| P = 256..65,536 for images | `min_pixels` 65,536 (256 patches); the 16,384-merged-token item cap |
| P from 16 per temporal patch for video | video `shortest_edge` 4,096 pixels per frame |
| V = P/4, up to 16,384 | merger input |

The small-T priorities of §2 do not apply: Vision has no speculative or per-request columns. The
tuning anchors are P = 1,024, 4,096 and 16,384 for the tower problems and V = 256, 1,024 and
4,096 for the merger, reported separately, with P = 65,536 as the large-item bound. T <= 8 occurs
only for the merger of tiny inputs and keeps the runtime-shape fallback's skinny GEMV, so its output
is unchanged by the registration.

All routes above T = 8 are instances of the BF16 TMA MMA template. The 4304 problems use its
tail-capable form: both operands are read through two-dimensional tensor maps in 64-element K
boxes, the hardware zero-fills the out-of-bounds part of fc1's last row tile and of fc2's last K
tile, and the store skips rows at or beyond N. Large-T routes rasterize row-fast: every row tile of
one token tile runs before the next token tile, so the activation, which exceeds L2 at large P,
is read from DRAM about once while the at most 47 MB weight stays in L2.

Selection uses complete-Op cold-L2 CUDA-graph timings of private-launcher candidates at
T = 9..65,536 (to 16,384 for the merger), the fallback, and the registered route, taking per
interval the fastest candidate within a small tolerance.

## 6. BF16 text problems of Qwen3.8-Flash-Next

Recipe B keeps two text projection groups in BF16 (the converter's per-class 8-bit rule does not
apply to them): the QSA group `[13952,2560]` (query, gate, key, value and the index projections in
one object, 12 layers) and the GDN a/b projections `[96,2560]` (36 layers). T is a call's columns:
1-8 in decode and verification, 9-255 for CPU-assisted prefill calls and short prompts, and the
prefill chunk (typically 1,024-4,096) otherwise. T <= 8 keeps the fallback's skinny GEMV, so decode
and verification outputs are unchanged; larger T routes per token interval to the fastest schedule
of the V0 tile sweep (within 2 % at each measured T).

Measured with that sweep (RTX 5090, cold-L2 CUDA graphs, median of 7): `[13952,2560]` 303.6 ->
45.5 us at T = 9, 5,473 -> 347 us at T = 1,024, 21,491 -> 1,229 us at T = 4,096 (17.5x), 85,492 ->
4,836 us at T = 16,384; `[96,2560]` 222-443 us -> 6.8-36 us from T = 9 to 8,192 (the fallback's
32x32 tiles leave most SMs idle at N = 96).
