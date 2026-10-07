# Research notes

Records for implementation candidates that were built, measured, and then **not** routed, plus the
mechanism notes those measurements depend on. An entry exists so a later round can see what was already
tried and on what evidence, instead of rebuilding the same candidate. Every number says what its timing
boundary was: a complete public Op, a kernel inside it, or a one-side probe. The reports under
`docs/performance/` own published results; this file does not replace them.

## The two Q5 parent shapes

These two shapes sit next to each other in `ops/linear/q5/q5_rowsplit_gemm_simt.cuh` and are easy to
describe wrongly, which has already happened once in this branch's own documents. Both are used by the two
Q4/Q5 input projections, and they are not the same shape.

**split4** (`q5_rowsplit_gemm_simt_split4_kernel`): **one CTA owns one output row** (`row = blockIdx.x`),
and **its four warps split the K dimension**: `chunk = threadIdx.x >> 5` selects the warp's K quarter, each
warp reduces its own partial sums with `warp_reduce_sum`, and the four partials meet in `s_part[4][kTt]`
behind `__syncthreads()`. It stages no weights and no activations in shared memory - both the quantized
weights and the activations are read from global memory. It is instantiated per exact column count
(`kTt`), which is why it covers exactly the live columns.

**c4 SIMT** (`q5_rowsplit_gemm_simt_kernel` with `kTt = 4`): **one output row per warp**, up to **four
columns** per column tile (`blockIdx.y` selects the tile), with the **quantized weight planes staged in
shared memory** (`s_nib`, `s_hi`, `s_sc`, filled by `q5_simt_issue_slab` through a cp.async pipeline) and
**activations read from the input tensor** (`q5_simt_consume_slab` reads `x0 + tt*k + xoff`). It does not
share an activation slab across rows; that was the removed row-block shape's mechanism, not this one.

## Q5 row-block small-T shape (`ops/linear/q5/q5_rowsplit_rowblock_small_t.cuh`, removed)

The shape staged one activation slab per block in shared memory and let `kRowsPerBlock` warps read it, so
one warp still owned one output row but the repeated activation loads were divided by the row count. It was
introduced to serve the Q5 parent of the two Q4/Q5 input projections at `T=7..9`. It is not routed at any
column count, and the file was removed from the production tree (git history keeps it, including the
`__syncwarp()` fence its pipeline needed).

What decided it, with the timing boundary of each number:

| Measurement, and its boundary | row-block | other shape | reading |
|---|---|---|---|
| GDN Q5 parent, `T=7/8/9`, **one-side probe** (us) | 62.7 / 62.7 / 79.1 | 52.5 / 60.6 / 70.9 (split4, one-side probe) | row-block loses at every count |
| GDN Q5 parent, `T=10/12`, **kernel duration inside the complete Op** (nsys per-instance minimum, us) | 91.4 / 102.3 | - | the decisive numbers below |
| attention complete projection, `T=7/8/9`, **complete public Op** | - | - | split4 is 18.8% / 14.6% / 5.4% faster than what was routed (row-block at 7/8, c4 at 9) |

Two caveats belong with that table, and both are the reason the row-block is not routed. Where each
number lives: the one-side probe rows are the R7 round's probe logs
(`profiles/bench/input_proj_r7/data/results_probe_*.txt`), the kernel-level row is
`profiles/bench/input_proj_r7/data/nsys/kernels.txt`, the attention row is
`profiles/bench/input_proj_r7/data/attn_ab.txt`, and this review round's complete-Op numbers are
`profiles/bench/input_proj_r7_review/data/`.

- A **one-side probe** ranked the row-block *ahead* of the c4 tile at `T=10` and `T=12`, and the
  row-block's own kernel-level numbers (the 91.4 / 102.3 us row) came out of the same attempt. What
  settled it was the complete public Op, which ranked the row-block behind at both counts, in both forms.
  A one-side or single-kernel ranking is therefore not used to move a route boundary in these Ops.
- The **narrow-tile counterparts and the complete-Op numbers of that reverted attempt are not in this
  bundle**: the complete-Op CSVs were written under the same file names as the runs that followed and were
  overwritten, and the "the projection op's own bench agrees" sentence that used to sit in the round report
  actually pointed at two-node **probe** sums (105.7 / 128.3 us), which is a different instrument and must
  not be quoted as the projection bench. The numbers are left as they were recorded rather than retimed;
  the decision they supported is unchanged, and its own evidence is listed below.

The current band ends were decided in the review round from a complete-Op comparison with its raw CSVs in
this bundle (`data/raw_s2/`, `data/raw_s2b/`), not from the numbers above.

## Q5 split4 band ends, per parent

The split4 shape is the Q5 parent mechanism for the low column counts of both fused projections, and the
two parents end their bands at different counts because their row counts differ (12288 against 7168):

- **GDN parent: split4 through `T=10`.** At `T=10` split4 beats the c4 tile in the complete Op for every
  organisation that exposes 10 aggregate columns - `B=1/W=10` (105.7 against 110.3 us cold), `B=2/W=5`
  (104.9 against 109.7) and `B=5/W=2` (104.2 against 108.1), in both forms and under both cache policies,
  and by more warm (about 93 against 104 us). At `T=11` and `T=12` it loses (118.0 against 111.9 and 140.5
  against 113.9 us cold), so the band ends at 10.
- **attention parent: split4 through `T=9`.** `T=10` ties with the c4 tile (77.06 us both) and `T=11`/`T=12`
  lose (85.0 against 79.1 and 95.5 against 81.2 us), so the band ends at 9.

Both ends are crossovers between two legal shapes, not limits of either shape: both shapes are correct at
every count in `[2,15]` for the GDN parent and `[2,12]` for the attention parent.

## Fused projection + convolution for the Q4/Q5 GDN input projections

The op either projects into a workspace and then runs the convolution as its own step, or fuses the
convolution into the projection epilogue (which is what the resolutions `T=1,2,3,5,6` already do). Three
attempts to widen the fused route were built and measured; all three were slower, so the materialized route
stays for the rest:

| Candidate | routed form (us, cold) | candidate (us, cold) | note |
|---|---|---|---|
| `T=4`, `B=1`, Snapshot / Record | 52.2 / 50.9 | 54.5 / 53.0 | Snapshot needs no workspace (81920 -> 0 B) and one fewer node (5 -> 4) |
| `T=7`, `B=1`, Snapshot | 69.4 | 81.2 | |
| `T=8`, `B=1`, Snapshot | 76.3 | 91.4 | |
| aggregate 8, `B=2/W=4`, Snapshot | 75.0 | 95.5 | batched fused organisation |
| aggregate 8, `B=8/W=1`, Snapshot | 75.0 | 106.2 | same |

Every number in that table is a **complete public Op** median from the trial and adopted builds
alternating in one window (`data/raw_a/`, `data/raw_b/`, `data/raw_c/`).

What can and cannot be concluded:

- The candidates above **did not win**, and that is all those measurements say. They rule out those
  specific implementations, not fusion in general.
- For `T=7`/`T=8` and for the batched organisation, the candidates drive the Q4 side through the row-split
  SIMT kernel while the routed form uses the K-split MMA kernel there (7..12), and they move the
  convolution into the projection epilogue. That is an implementation difference between those candidates
  and the routed form, and it is consistent with the sizes of the losses.
- For `T=4` it does **not** explain anything: the routed form at `T=4` is `launch_t4_pdl`, whose Q4 side is
  **also the row-split SIMT kernel** (`Q4GdnSimtR8C4Schedule`) with the dependent-launch pair. So "the
  candidate replaced the K-split Q4 with SIMT" is wrong for `T=4`; what is left there is the measurement:
  the fused candidate is 4-5% slower in both forms while being structurally simpler.
- The **Snapshot** `T=4` prototype removes the projection workspace (81920 -> 0 bytes). The **Record**
  form has no such saving to report: its materialized route writes the caller-owned `conv_record` through
  its own publish path, and its public temporary-workspace query is 0 in both builds. The two forms are
  not interchangeable here.

**Not implemented:** fusing the K-split MMA Q4 side with the Q5 split4 side and a sequence-collecting
convolution. Nothing measured above excludes it; it needs a different kernel rather than another
instantiation of the existing template.

The three candidates' minimal patches, their per-configuration measured table and their trial builds' test
logs are kept with the review bundle under `profiles/bench/input_proj_r7_review/data/prototypes/`. The
measured prototype sources were reverted before the final build and are not retained; that directory's
README marks the patches as reconstructions and lists which trial-build artifacts do still exist.

## Decode-round measurement setup (perf/decode-round-overheads)

The two entries below were measured on the RTX 5090 with the NVIDIA ModelOpt Qwen3.8-27B artifact
(`qwen3_8_27b_nvfp4-nvidia.ninfer`: NVFP4 MLP, FP8 attention/GDN projections), DFlash2 K=7 with the
optimized proposal head, INT8 KV, CUDA 13.4, Windows WDDM. The **controlled** numbers are
`infernix_bench -pg 16384,256` and `-pg 60000,256` with greedy decoding, so every arm decodes the same
token stream (identical round and acceptance counts in every run); each value is the mean of 6-9
repetitions from arms interleaved in one window, standard deviation below 0.02 ms. The candidates were
selected per process from one binary (commit `f2c991d0` on branch `perf/decode-round-overheads`, not
merged; it holds every candidate below behind `INFERNIX_AB_ITEMS`) so arms differ only in the item. Kernel
attributions come from Nsight Systems `--cuda-graph-trace=node` captures of one 128-token decode at
16K context, compared kernel by kernel as the interval between consecutive kernel end times.

## Programmatic dependent launch with an entry-time trigger (replaced)

The first PDL form had every decode-path kernel call `griddepcontrol.launch_dependents` at entry,
then wait, and hinted the first four K phases of each FP8 SIMT warp's weight rows into L2 before the
wait. It was correct (byte-identical greedy output) and **10.4 % slower per round at 16K, 9.4 % at
60K**. The profile showed kernel lifetimes overlapping heavily (50 ms of summed kernel time in an
18.4 ms round) and the loss concentrated in bandwidth-bound kernels whose successors had launched
early: the 80-CTA NVFP4 down projection (+1.25 ms per round, 37 -> 57 us each), the FP8 out
projection (+0.32 ms), the GDN record kernel (+0.16 ms) and attention (+0.10 ms). With an entry
trigger the next kernels cascade onto the SMs the running grid leaves idle and issue their weight
loads and L2 prefetches while it is still streaming, so they compete for its bandwidth rather than
fill idle time.

The adopted form triggers dependents after the main loop of every streaming kernel (NVFP4/FP8/Q8/Q4
GEMMs, GDN record, drafter sliding-window attention and convolution prepare) and keeps the entry
trigger only for short kernels. It measured 2.5 % faster at 16K and 2.3 % at 60K. Two further
variants were measured on that form and not kept: launching KV-cache attention as a dependent (0.5 %
slower at 16K, equal at 60K; it has no weights to stage) and a two-phase L2 prefetch in the FP8 SIMT
kernel before the wait (no measurable change against removing it).

## Fused NVFP4 RMSNorm + SwiGLU MLP + down (not routed)

A fused Op computed `residual += down(SwiGLU(gate_up(RMSNorm(residual))))` for the NVFP4 AllowA4
route at T=5..128: one kernel for Offset RMSNorm plus NVFP4 activation quantization (the D=5120
RMSNorm launcher's reduction order, reused from the fused attention-input route), the W4A4 gate/up
MMA with an epilogue that wrote the down projection's NVFP4 codes and scales directly (16-row groups,
the same `quantize_nvfp4_k16` arithmetic), and the unchanged down MMA reading that plane. It was
byte-identical to the unfused composition at every tested T (5..128, eager and graph).

| Arm (controlled, ms/round) | 16K | 60K |
|---|---:|---:|
| without PDL: fusion vs master | -0.36 % | -0.32 % |
| with PDL: fusion + PDL vs PDL alone | +0.7 % | +0.7 % |

Without PDL the fusion removes two small launches per layer. With PDL those kernels already overlap
the neighbouring GEMMs' weight staging, and the fusion only lengthens the serial path: the fused
norm/quantize kernel (8 CTAs, 3.4 us) cost 218 us per round against 196 us for the separate norm and
quantize, and the quantizing epilogue added 103 us to the gate/up MMA against 80 us for the quantize
kernel it replaced, plus about 60 us across the following GEMMs. It was removed with PDL adopted; the
measured source is commit `f2c991d0`.

## DFlash2 proposal filtered by the request's top_k / min_p / top_p (not adopted)

The DFlash2 candidate selector draws each proposal from `softmax(edge / temperature)` over its 16
candidates. The candidate restricted that distribution exactly as the target sampler restricts its
own candidates (top_k, then min_p against the best weight, then the shortest top_p prefix of the
pre-truncation weight), renormalized it, and retained the restricted q for acceptance, so target
sampling stayed exact. It passed an FP64 oracle for the filtered q and draw.

Open-loop controlled test (production thinking sampling: temperature 1.0, top_k 20, top_p 0.95;
DFlash2 K=7 plus ngram 15/12, INT8 KV, one lane; 16 coding prompts x 2 seeds x 1536 output tokens
per arm, three rounds with the arm order alternated, 96 requests and about 140K tokens per arm):

| Arm | Output tok/s | Tokens per round | Neural acceptance |
|---|---:|---:|---:|
| master proposal | 208.1 | 3.358 | 33.19 % |
| filtered proposal | 206.5 | 3.331 | 32.91 % |

The per-round paired acceptance differences were +0.37, -0.26 and -0.99 points. The draft's own
top_p tail over 16 candidates carries little mass, so trimming it barely moves q toward the target
distribution. A closed-loop agentic pair had shown +2.0 points; its control arm produced 25 % less
text and three more cache-accounting request failures, and the controlled test attributes that
difference to content rather than the filter.

A proposal temperature scale (the draft softmax at `temperature * s`, retained q unchanged in
meaning) was measured the same way, on the build with the adopted round changes:

| Arm | Output tok/s | Neural acceptance |
|---|---:|---:|
| s = 1 (master) | 213.7 | 33.19 % |
| s = 0.7 | 213.0 | 33.04 % |
| s = 0.5 | 207.5 | 31.71 % |

Sharpening does not raise acceptance; at 0.5 it lowers it. Neither form was kept.

## 16-bit activations for the NVFP4 gate/up projection at decode widths (not adopted)

Measured on the RTX 5090 with the NVIDIA NVFP4 artifact (`qwen3_8_27b_nvfp4-nvidia.ninfer`),
DFlash2 K=7 with the proposal head, INT8 KV, CUDA 13.4 on Windows, master `2ca38d1d`.

The NVFP4 MLP routes use 4-bit activations (A4) from 5 columns (gate/up) and 8 columns (down). As
complete public Ops on cold L2, the fused A16 gate/up kernel was faster than A4 through 8 columns
and slower above, while the A16 down projection was far slower from 5 columns:

| Op, T (us, median) | A4 | A16 |
|---|---:|---:|
| SwiGLU gate/up 34816x5120, T=5 / 8 / 9 / 16 | 66.1 / 66.1 / 66.1 / 66.1 | 62.0 / 64.1 / 76.4 / 88.7 |
| LinearAdd down 5120x17408, T=4 / 6 / 8 | 36.2 / 58.1 (A16 route) / 37.5 | 36.1 / 58.1 / 58.1 |

Only the gate/up projection was therefore tried with A16 at up to 8 columns (every C=1 neural
round). In the decode graph it was slower, not faster: the A4 MMA kernel stages its weights before
its PDL wait and was charged 60.3 us per call, the A16 kernel 63.6 us (Nsight, 16K context). Greedy
`infernix_bench -pg 16384,512`, three interleaved passes: 14.957 against 14.764 ms per round (+1.3 %).
Sampled acceptance (`acc_ab.py`: 24 prompts x 2 seed sets x 1536 tokens, temperature 1.0, top_p
0.95, top_k 20, thinking on, about 66K tokens per arm): 3.468 against 3.443 tokens per round
(+0.7 %, within the spread of this test) and 15.13 against 14.90 ms per round (+1.5 %); output
tok/s 228.1 against 230.1. The 4-bit activation rounding of the gate/up input does not cost
measurable acceptance, so the slower A16 kernel does not pay.

## L2 weight prefetch of the next projection during latency-bound kernels (not adopted)

Same configuration. Idea: the 31.5 MB FP8 output projection of each GDN layer follows about 14 us
of conv, record and gating kernels that leave DRAM mostly idle, and fits the 96 MB L2. With its
weights L2-resident the out-projection Op takes 16.6 us instead of 23.9 us at T=8 (cold vs warm
L2, complete Op), so a perfect prefetch would save at most about 0.35 ms per round (2.4 %).

A prefetch kernel issued after the GDN projection, as a PDL consumer that issues its prefetches
before waiting for the producer, was measured in three forms (greedy `-pg 16384,512`, identical
token streams, ms per round against 14.764):

| Form | ms/round | change |
|---|---:|---:|
| `cp.async.bulk.prefetch.L2` from 2 CTAs, after the projection | 18.33 | +24.1 % |
| same, before the GDN norm/gating | 18.51 | +25.4 % |
| same, attention layers before the attention kernel | 15.39 | +4.2 % |
| `prefetch.global.L2::evict_last` per line, 64 CTAs, after the projection | 14.817 | +0.36 % |
| same, 16 / 170 CTAs | 15.046 / 14.833 | +1.9 % / +0.47 % |
| same, before the norm/gating | 14.802 | +0.26 % |
| same, attention layers | 14.786 | +0.15 % |

A bulk prefetch holds its grid until the transfer drains, so the next kernel waits for all of it.
The per-line form does land: Nsight shows the GDN out-projection at 17.9 us instead of 21.4 us,
but the record kernel slowed from 9.7 to 11.2 us behind the prefetch traffic and the extra
3.4 us node sits on the dependency chain. The remaining upside, under 1 % per round, needs the
prefetch fused into the GDN kernels across Op ownership, and was not pursued.

## 8-bit P×V in the fast INT8 prompt kernel (opt-in, `--int8-prefill-8bit-pv`)

The fast INT8 prompt kernel accumulates P×V on FP16 Tensor Cores (FP32 accumulation, 209.5
TFLOPS on RTX 5090). The opt-in variant multiplies P' = P × (V group scale), quantized per row,
64-dimension group and 64-key tile to unsigned 8-bit codes (step = row maximum / 255, round to
nearest), by the stored signed INT8 V codes on `mma.m16n8k32.u8.s8` (838 TOPS) and rescales the
INT32 sums in FP32. SageAttention2 quantizes P to FP8 E4M3 instead; that needs V in FP8 too, and
the INT8-G64 V codes are not exact in E4M3 above 16, so integer codes were chosen to keep V exact.

RTX 5090, CUDA 13.4, `qwen3_8_27b_nvfp4-nvidia.ninfer`, INT8 KV. Per attention layer
(`infernix_causal_softmax_attention_bench --entry append --geometry d256-h24-kv4 --tokens 3584
--execution graph --cache cold`, two passes, medians): 584/564 µs at an empty context (-3.5 %),
4990→4455 (-10.7 %), 9346→8332 (-10.8 %), 18263→16207 (-11.3 %) and 37064→34082 µs (-8.0 %) at
16K/32K/64K/128K. End to end (`infernix_bench -p 16384,65536 --prefill-chunk 4096`, two passes):
prefill +1.5 % at 16K and +3.9 % at 64K. Perplexity on `ninfer-ppl-1m-v1` (full corpus):
4.90771 → 4.90551 with 4096/2048 windows and 4.904120 → 4.851635 with 65536/32768
(english_reference 7.42 → 7.26, code 1.849 → 1.810, Chinese and long-form within 0.03 %).

The 64K change is far larger than the per-chunk error the random-data op test sees (relative L2
0.0020 against 0.0017 for the FP16 path), and it grows with context. The likely cause, not
isolated: real attention rows within a 64-key tile span many orders of magnitude, and every
probability below half a code step of the tile's largest rounds to zero while the softmax
denominator keeps it, so each tile's diffuse tail is dropped rather than averaged. Lower
perplexity on this corpus does not make it more exact. Not tried: FP8 P with a split (hi/lo) V,
or two u8 terms per probability (16-bit fixed point at half the INT8 rate, still twice the FP16
rate), either of which would keep the tail.

## Static fan-out draft trees (superseded by per-round lattice trees)

The first tree-verification build (`feat/tree-verification`, commit `062830eb`) verified one fixed
topology per engine, the root fan-out: the main chain plus B-1 side branches of D proposals that all
start at the anchor, with INT8 KV only. Measured on the decode-saturation suite (DFlash2 K=7 with
the proposal head, n-gram 15/12, INT8 KV, two interleaved passes, against the same build with B=1):

| C | B=2 D=3 ms/round | B=2 D=3 tok/s | B=3 D=3 ms/round | B=3 D=3 tok/s |
|---|---:|---:|---:|---:|
| 1 | +2.7 % | -1.8 % | +4.1 % | +3.8 % |
| 2 | +4.5 % | +6.6 % | +7.1 % | +6.9 % |
| 4 | +9.1 % | -0.8 % | +12.0 % | -1.8 % |
| 8 | +14.6 % | -5.0 % | +23.8 % | -10.3 % |

Two things limited it. Alternatives exist only for the first proposal, whatever the drafter's
confidence there, while rejections are spread over every depth: in a one-request chain run of the
same suite 26 % of the rounds end at the first proposal and 69 % at depths 2..7. And one width for
every batch size makes the tree pay its full per-column cost at C=4 and C=8, where rounds are no
longer bound by weight streaming. The replacement builds each row's tree per round from the
drafter's lattice, best first by the probability that a column is reached and accepted, so side
branches start at whichever depths the drafter is unsure of, and its width comes from a
per-batch-size table or, with `--draft-tree-nodes auto`, from the measured cost and acceptance of
each width per batch size and context ([tree verification](docs/maintainer/tree-verification.md)).

## One KV stream for wide verification rows (not adopted)

Verification blocks wider than 8 columns run the grouped attention kernels in 8-column token tiles,
one CTA per tile, each reading the row's KV. At long context a 12- or 16-column tree's attention
costs half as much again as the chain's, so serving every column of a block from one KV stream
looked like the way to make trees pay there. Two INT8 forms of the pipelined kernel
(`grouped_pipelined.cuh`, 24/4 geometry, same partition, bitwise-equal partials) were measured:
one CTA of 96 rows (six QK row tiles, eight warps, one CTA per SM), and one CTA of two 48-row
groups (16 warps, each group exactly an 8-token CTA, both reading the same double-buffered K/V
tiles). Per layer call, append entry, cold cache, graph launches, one row, us (baseline / 96-row
CTA / two groups):

| Columns | 2K | 16K | 64K | 131K |
|---|---|---|---|---|
| 8 | 23.7 | 45.7 | 107.1 | 189.0 |
| 12 | 27.8 / 31.3 / 27.2 | 56.4 / 58.4 / 55.9 | 154.2 / 141.9 / 148.0 | 293.4 / 259.2 / 273.0 |
| 16 | 29.8 / 39.5 / 31.3 | 58.5 / 68.2 / 62.0 | 156.8 / 160.9 / 162.4 | 297.6 / 289.7 / 297.6 |
| 24 (16-column tiles) | 32.0 / 41.6 / 35.4 | 72.8 / 94.8 / 92.8 | 221.5 / 275.0 / 289.9 | 414.3 / 518.8 / 549.5 |

Reading the KV once leaves 16 columns as slow as before. The two 8-column CTAs of a split run
together and the second one's KV reads hit the 96 MB L2 (two DRAM reads of a 131K INT8 row would
need more bandwidth than the card has in 298 us). The extra time is the tile's own work, which
grows with the rows: QK and above all P x V on FP16 Tensor Cores with FP32 accumulation (209.5
TFLOPS); 16 columns at 131K are about 123 us of P x V at that peak. NVFP4 (QK and P x V on FP16)
and K8V4 (FP8 QK, FP16 P x V) nearly double from 8 to 16 columns (209 to 406 us and 154 to 283 us
at 131K), so they are compute-bound too. Both INT8 forms were reverted. What would still lower
the cost of verification columns at long context: cheaper P x V arithmetic (FP16 accumulation per
key tile, or 8-bit P as in `--int8-prefill-8bit-pv`; both change precision) or a warp-specialized
kernel nearer the Tensor Core peak. FP16 accumulation was measured later and gained nothing: the
INT8 kernel is latency-bound at low occupancy rather than Tensor Core-bound (see "FP16 P×V
accumulation in the INT8 verification kernel" below).

## FP8 LM head schedules at 42-64 columns

The [248320,5120] FP8 LM head took the 64-column MMA schedule from 42 columns (1145-1150 us at
42-64 against 818-820 us at 40), which tree rounds of 12 columns at four requests and 16-column n-gram
or tree rounds at three or four requests hit. The sliced-K route now runs to 64 columns. RTX 5090,
`infernix_linear_bench --qtype FP8 --n 248320 --k 5120`, graph execution, cold cache, median us;
variants are (K warps, minimum blocks per SM, stages, row tiles) of `Fp8A16SlicedKMmaSchedule`.
The variants were measured in two sweeps; 4,2,1,2 and 2,2,1,2, measured in both, moved by up to
5 % between them, and the table shows the second sweep, which also measured 2,2,2,2:

| Columns | MMA schedule | 4,2,1,2 | 8,1,1,2 | 4,2,1,1 | 8,2,1,1 | 4,1,2,2 | 2,2,1,2 | 2,2,2,2 | 2,4,1,2 | 2,1,1,2 | 2,2,1,1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 42 | 1148 | 902 | | | | | 920 | 919 | 918 | 920 | 1240 |
| 48 | 1145 | 949 | | | | | 967 | 964 | 966 | 967 | 1358 |
| 49 | 1146 | 1037 | 1195 | 1367 | 1703 | 1013 | 1018 | 963 | 1016 | 1018 | 1474 |
| 56 | 1148 | 1148 | 1285 | 1504 | 1860 | 1255 | 1090 | 994 | 1078 | 1088 | 1603 |
| 57 | 1148 | 1203 | 1373 | 1717 | 2013 | 1354 | 1098 | 1059 | 1090 | 1092 | 1689 |
| 64 | 1150 | 1283 | 1434 | 1848 | 2083 | 1361 | 1162 | 1109 | 1166 | 1170 | 1807 |

Up to 48 columns the existing four-warp tile stays (two K warps are 0-2 % slower from 25 to 48
columns); from 49 two K warps with a double-buffered stage are fastest at every width. Above 64
columns the 96- and 128-column MMA schedules are unchanged.

## NVFP4 MLP schedules at decode widths

At 8-32 tokens the NVFP4 A4 down projection ([5120,17408]) takes about 37.5 us and the fused
gate/up + SwiGLU ([34816,5120]) about 66 us, 1.34 and 1.52 TB/s of weight streaming. The down
projection's 32x64 tiles give only 80 CTAs on 170 SMs, so smaller tiles looked like a gain. RTX
5090, `infernix_nvfp4_linear_add_bench --n 5120 --k 17408 --policy a4` and
`infernix_nvfp4_linear_swiglu_bench --policy a4`, cold weights (256 MiB flush), median of 30
repeats, mean of two passes in opposite variant order, us. Variants are (block tokens, block rows,
block K, token warps, row warps, stages, minimum blocks per SM) of `Nvfp4A4MmaSchedule`; every
variant keeps each output's K order, so all are bit-identical. The bench's medians move in steps of
about 0.7 us.

| Down projection, tokens | 8 | 12 | 16 | 24 | 32 | 48 | 64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 32,64,256,2,4,3,2 (routed to 64) | 37.5 | 38.2 | 38.1 | 38.2 | 38.1 | 38.0 | 38.2 |
| 32,32,256,2,4,3,2 | 37.5 | 37.9 | 37.8 | 37.5 | 38.0 | 44.0 | 50.4 |
| 32,32,256,2,2,3,2 | 37.5 | 37.6 | 38.1 | 37.9 | 37.7 | 44.3 | 48.5 |
| 32,32,256,2,2,4,3 | 37.5 | 38.2 | 37.6 | 37.6 | 37.9 | 44.0 | 48.1 |
| 32,64,256,2,4,4,2 | 37.5 | 38.2 | 37.9 | 38.2 | 38.0 | 37.9 | 38.1 |
| 32,32,512,2,2,3,2 | 36.2 | 37.8 | 37.9 | 38.2 | 38.1 | 54.6 | 58.3 |
| 32,32,256,2,1,4,4 | 36.9 | 37.9 | 38.1 | 37.9 | 38.2 | 44.4 | 48.4 |

Twice the CTAs left the time unchanged: the projection streams at the same rate from 80 or 160
CTAs, at about 75 % of DRAM peak, so the memory system rather than idle SMs limits it. Only 512 K
per stage, which issues longer contiguous weight reads per CTA, helped, at 8 tokens; six
interleaved passes of 50 repeats measured 36.6 against 37.6 us (-2.6 %) at 8 tokens and no change
at 12 and 16 (-0.1 %, -0.5 %), so the [5120,17408] projection takes it up to 8 tokens. The attention output projection ([5120,6144], 17.8-19.7 us) gained
nothing from any variant.

| Gate/up, tokens | 8 | 12 | 16 | 24 | 32 | 48 | 64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 32,128,256,2,4,2,1 to 32; 64,128,256,4,4,2,1 to 64 (routed) | 65.4 | 66.1 | 66.1 | 66.1 | 65.4 | 68.1 | 66.1 |
| 64 rows, 4 row warps, 2 stages, 2 blocks per SM | 70.2 | 69.5 | 70.1 | 69.5 | 70.1 | 66.1 | 66.0 |
| 64 rows, 2 row warps, 2 stages, 2 blocks per SM | 70.3 | 69.5 | 71.2 | 70.3 | 71.3 | 66.1 | 66.2 |
| 128 rows, 4 row warps, 3 stages, 1 block per SM | 65.4 | 66.1 | 66.1 | 66.2 | 66.1 | 68.2 | 66.1 |
| 64 rows, 4 row warps, 3 stages, 2 blocks per SM | 64.7 | 65.4 | 65.3 | 65.4 | 65.3 | 72.2 | 72.3 |
| 64 rows, 2 row warps, 3 stages, 3 blocks per SM | 66.1 | 66.1 | 66.1 | 66.1 | 66.1 | 74.3 | 71.6 |

Each variant row uses 32-token blocks with 2 token warps up to 32 tokens and 64-token blocks with
4 token warps above. 64-row tiles with a third stage at two CTAs per SM, over six interleaved
passes: 65.3 against 66.1 us at 8 tokens (-1.3 %), 64.9 against 66.1 at 12 (-1.8 %) and 66.06
against 66.14 at 16 (-0.1 %); they are routed up to 32 tokens. Above 32 tokens the same tiles are
6-9 % slower, but with two stages they are not: six interleaved passes of 64-token tiles
(64,64,256,4,2,2,2 and 64,64,256,4,4,2,2 against the routed 64,128,256,4,4,2,1) measured 66.1
against 68.0-68.2 us at 33, 40, 47, 48 and 49 tokens (-2.4 to -3.0 %), 66.2 against 66.2 at 56,
66.1 against 66.8 at 63 and equal at 64. The 128-row tile takes 68.2 us in most passes at 33-49
tokens, the 64-row tiles 66.1 in every pass; the eight-warp 64,64,256,4,2,2,2 is routed for 33-64
tokens.

## FP16 P×V accumulation in the INT8 verification kernel (not adopted)

The entry above ends with cheaper P×V as the remaining lever for verification columns at long
context. The pipelined INT8 kernel (`grouped_pipelined.cuh`: 2-8 columns, and the 8-column tiles
of wider rows) accumulated each 32-key tile's P×V with FP16 accumulators (`mma.m16n8k16`, twice
the FP32-accumulate rate on RTX 5090) and added the tile into the FP32 output, as the fast INT8
prompt kernel does. Per layer call (append entry, 24/4 geometry, cold cache, graph launches, one
row, two passes in opposite order), us, FP32 → FP16 accumulation:

| Columns | 2K | 16K | 64K | 131K |
|---|---|---|---|---|
| 2 | 14.2 → 15.0 | 33.6 → 33.4 | 94.8 → 94.8 | 173.2 → 174.8 |
| 8 | 24.2 → 25.0 | 45.5 → 45.7 | 109.3 → 109.0 | 189.1 → 189.3 |
| 12 | 28.5 → 29.3 | 56.2 → 57.2 | 155.2 → 156.3 | 293.7 → 296.5 |
| 16 | 29.6 → 31.3 | 57.9 → 58.0 | 158.2 → 158.6 | 299.9 → 297.9 |
| 24 | 31.9 → 33.4 | 72.3 → 72.6 | 222.1 → 222.1 | 416.9 → 412.6 |

Two rows moved by -1.1 % to +5.2 %. Nsight Compute, 16 columns over 131K keys: 350.9 → 346.6 us,
DRAM throughput 40 %, SM throughput 51 → 46 %, issue slots busy 36 → 46 %, no eligible warp in
62 → 51 % of cycles, theoretical occupancy 33 % (two CTAs per SM, limited by both registers and
shared memory). The kernel waits on latency, not on Tensor Core throughput, so halving the P×V
cost does not shorten it; the extra time of 16 columns over 8 comes from more dependent work per
KV byte at low occupancy, not from a saturated Tensor Core. Reverted, with no option, since it
gave no speed for its lower precision. What could still help: more warps in flight per SM, or a
warp-specialized form in which producer warps load and decode K/V while consumer warps run the
MMAs.

## Where a full INT8 prompt chunk's attention time goes (no change made)

Nsight Compute on the fast INT8 prompt kernel for one 3584-column chunk over 32K cached keys
(RTX 5090, `infernix_causal_softmax_attention_bench --entry append --geometry d256-h24-kv4
--kv-dtype int8 --tokens 3584 --context 32768 --fast-prompt`, eager, cold cache; 10.1 ms at the
profiler's 2.36 GHz): the Tensor pipe is active in 55.6 % of cycles (FP16 HMMA for P×V 37.1 %,
INT8 IMMA for QK 18.5 %) and issue slots are busy 50 % of the time. One eight-warp CTA fits an SM
(255 registers per thread, 100 KB of shared memory), so each scheduler has two warps, 0.66 of
them eligible on average, and none in 49 % of cycles; the main stall is a fixed-latency
dependency wait, then math-pipe throttle. K/V come from L2 (98.6 % hit rate, DRAM 1 %); 2.9 M
local loads are register spills.

Of 8.15 G executed warp instructions the MMAs are 0.56 G (HMMA 0.37 G, IMMA 0.19 G). The V decode
(`causal_prompt_i8_fast_decode_v_pair`: one LOP3, two PRMT, two HADD2 and two HMUL2 per four codes)
is 2.9 G, 36 % of all instructions (LOP3 0.42 G, PRMT, HADD2 and HMUL2 0.84 G each), and every one
of a CTA's eight warps decodes the whole V tile, so seven eighths of it repeats. FMUL 0.97 G, FFMA
0.85 G and I2FP 0.37 G are the QK group scales, the online softmax and the output rescale.
Decoding each V tile once per CTA into shared memory (32 KB more, which fits at one CTA per SM, at
twice the shared-memory bytes per V fragment), or splitting the warps into decode producers and
MMA consumers, would remove close to a third of the issued instructions; that is the largest lever
left for long prefill, and the numerics would be unchanged. Folding the V group scale into P
instead (as the 8-bit P×V form does) would remove most HMUL2s but round differently.

## Key splits for the fast INT8 prompt kernel: the plan's cost model

The fast NVFP4 prompt kernel's plan (waves x one CTA's sweep / splits, plus 1 % per extra split)
was first reused for the INT8 kernel. Over long contexts it was right, but the INT8 kernel also
runs over short ones (NVFP4 only above 2048 keys), and there its splits cost more than they saved:
each split writes and merges an FP32 row per column and query head, which the plan ignored. Per
layer call against the unsplit kernel (RTX 5090, append entry, 24/4 geometry, cold cache, graph,
four passes in opposite order): 900-1280 columns over 1K cached keys were 16-21 % slower, 512-640
columns over 2K 6-13 % slower.

Every CTA shape and split count was then forced (8 warps with 1-8 splits, 4 warps with 1-7) at
257-1536 columns over 0-64K cached keys, one pass. Splitting a four-warp CTA was never the fastest
choice. The best eight-warp split count, and its gain over the best unsplit launch:

| Cached keys | 257-384 columns | 512 | 576-640 | 768-896 | 1024-1280 | 1408-1536 |
|---|---|---|---|---|---|---|
| 0-512 | none | none | none | none | none | none |
| 1K | 2: -3 to -7 % | none | none | none | none | none |
| 2K | 2: -10 to -12 % | 3: -4 % | none | none | none | none |
| 4K | 2: -14 to -19 % | 3: -12 % | none | none | 2: -7 to -9 % | none |
| 8K | 2: -18 to -23 % | 5: -18 % | 4: -6 % | none | 2: -14 to -15 % | none |
| 32K | 7: -26 to -33 % | 5: -30 % | 4: -15 to -16 % | none | 2: -20 to -21 % | none |
| 64K | 7: -29 to -35 % | 5: -32 % | 4: -16 to -18 % | none | 2: -20 to -22 % | none |

(448 columns, between the four- and eight-warp shapes, gains 3-8 % with five splits from 16K.)
A plan in units of one eight-warp CTA's sweep over one visible key (about 59 ns): waves x keys /
splits, plus half a key's sweep per column and split for 24 query heads, with the four-warp CTA at
80 % of the eight-warp per-key time and never split, chooses within 0.14 % of the fastest measured
configuration on average and 3.8 % at worst (448 columns over 8K keys). The NVFP4 kernel keeps its
own plan.

One forced configuration hung the GPU: four-warp CTAs with eight splits (reached only at 257 and
320 columns over at least 4K keys). The kernel ran at 100 % for over a minute; killing the process
caused a TDR with a microcode reset ('UCodeReset TDR occurred', nvlddmkm event 153) and the machine
had to be restarted. The four-warp kernel with up to seven splits and the eight-warp kernel with
eight completed. The cause was not found by reading the code: the cache is populated through the
codec, the workspace arena checks capacity, every barrier is CTA-uniform, and the kernel's only
data-dependent loop was the FP16 range rescale (`while (ldexpf(vmax, -shift) > 8)`), which never
ends if a V scale reads as infinity. The loop now halves the scale clamped to the largest finite FP16
value (at most 13 halvings), only eight-warp CTAs split, and the four-warp split instance is no
longer compiled. A first version that bounded the loop's condition instead made the four-warp
unsplit kernel spill 8 bytes and run 3-6 % slower; the clamp inside the rarely taken branch does
not.

## Decoding each INT8 V tile once per CTA (not adopted)

The profile above suggested decoding each V tile once per eight-warp CTA instead of in every warp.
Adding a 32 KB FP16 V buffer does not fit: sm_120 allows 99 KB of shared memory per block and the
kernel uses 98 KB. The measured form decoded in place instead: after every warp's QK of a tile, each
thread read one key's 64 codes of one group into registers and, after a barrier, wrote them as
FP16 (the same values as the register decode) over the tile's consumed K and V codes, exactly
32 KB, in an even/odd-column, key-swizzled layout that every warp then loaded with ldmatrix.trans.
It passed the attention suite. Per layer against the register decode (append entry, 24/4 geometry,
cold cache, graph, three alternating passes): 3584 columns -0.2 % over 128K keys, -1.2 % over 32K,
+1.1 % over 8K and +1.3 % over an empty context; 2048 columns +1.5 to +2.4 % everywhere; 4096 columns
-0.5 % to +1.8 %. The removed decode instructions had been filling issue slots the Tensor Cores
were not using: with two warps per scheduler the Tensor pipe idles while a warp runs its softmax,
and the two added barriers per tile and a larger spill (24 instead of 8 bytes of stack) took back
what the decode saved.
Overlapping one tile's softmax with another tile's P×V would need a third K/V stage, which the 99
KB limit also rules out for this CTA shape.

## Decoupled warps in the fast INT8 prompt kernel (not adopted)

The INT8 profile's math-pipe throttle suggested the eight warps run each 64-key tile in phase
(QK, softmax and PV at the same time), because every tile ends at a CTA barrier. The measured form
removed that barrier: each of the two stages got an mbarrier completed by the loading warp's
cp.async copies (`cp.async.mbarrier.arrive.noinc`), warps waited on it by phase parity, counted
themselves off the stage with a shared atomic, and the warp that released a stage last loaded the
page two tiles ahead into it, so warps could drift up to a tile apart. It passed the attention
suite and spilled nothing. Per layer against the barrier form (append entry, 24/4 geometry, cold
cache, graph, three alternating passes): +3 to +6 % from 8K to 128K keys at every width
(3584 columns +4.6 % over 128K) and +0.8 to +2.2 % at 2048-4096 columns over an empty context.
The warp that releases a stage last is the slowest one, and the whole 33 KB page load (68 copies
per lane) lands on it, which lengthens exactly the critical path the drift was meant to shorten.
Spreading the load over all warps would need a third stage (a warp could only load into a stage
every warp has left), which the 99 KB limit rules out (the kernel uses 98 KB), and TMA cannot
zero-fill the keys past a CTA's last visible key, which keep masked columns finite.

## FP8 P×V in the fast NVFP4 prompt kernel's register decode (not adopted)

The fast NVFP4 prompt kernel decodes V in registers in every warp. An 8-bit form rounded each
probability (as 256 p against the tile's own row maximum) and each decoded V value to E4M3 and ran
P×V on block-scaled E4M3 Tensor Cores with FP32 accumulation, twice the rate of the FP16-accumulate
P×V it replaced (1013 against 508 TFLOPS dense on RTX 5090; FP16 with FP32 accumulation runs at
254). The k32 MMA consumes the lane's own QK keys in a permuted order (2t, 2t+1, 8+2t, 9+2t per
half) that two consecutive k16 V fragments already match, so no probability or V value crosses
lanes. Relative L2 error against the oracle was 0.025-0.036 (the FP16 form: 0.0017), and it was
slower per layer: 0 to +5 % (3584 columns +1.3 % over 128K keys, 512-1024 columns +3 to +5 % from
8K). The extra E4M3 conversion of every decoded value, done again in each of the eight warps, cost
more issue slots than the halved MMA time saved; see the decode-once kernel below for the form
that does pay.

## The MX-FP8 tiled prompt kernel: FP16 P×V, cost-aware splits and issue order

The FP8 and K8V4 prompt kernel (NInfer's `mxfp8_tiled_mma.cuh`: QK on block-scaled FP8 Tensor
Cores, each V tile decoded once per CTA into a shared FP16 arena) accumulated P×V with FP16 MMAs
in FP32. On RTX 5090 that form runs at half the FP16 rate (`mma_rates.cu`, dense mma.sync:
FP16 with FP16 accumulation 508 TFLOPS, with FP32 accumulation 254; E4M3 block-scaled with FP32
accumulation 1013, plain E4M3 with FP32 accumulation 507). All measurements below are per
attention layer (append entry, 24/4 geometry, cold cache, graph, three alternating passes).

- FP16 accumulation per 64-key tile, promoted into the FP32 output once per tile under a per-tile
  power-of-two V shift that keeps the partial inside FP16 (K8V4: largest scale at most 128; FP8:
  largest row scale below 2): K8V4 -7 to -19 % (3584 columns -15.5 % over 128K keys), FP8 -7 to
  -18 %, at every width and context measured. The four FP16 partial registers made the kernel
  spill 24-40 bytes (none before). Taking alpha into the accumulator before P×V and re-deriving
  the shift after QK (to shorten live ranges) kept the spill and ran 1-5 % slower, so the fused
  promotion stayed.
- Longest-first issue order (the first QHeads CTAs take every head's last row block): K8V4 3584
  columns -19 % over an empty context and -10 % over 8K, 1024 columns -14 % over an empty
  context, -0.3 to -3 % from 32K; FP8 alike. Slower: K8V4 512 columns over an empty context +3 %
  and FP8 256 columns +3.5 % (passes from +1.4 %).
- Split count from a cost model instead of the fewest waves per split: NInfer's partition chose
  the split count that minimizes waves per split and ignored storing and merging every split's
  FP32 rows, so a chunk over a short context split into up to eight (4096 columns over an empty
  context: 1731 us against 820 us for 3584 columns). The kernel and the merge now take the
  cheapest count up to the partition's bound under the fast INT8 kernel's measured model (waves x
  keys per split, plus half a key's sweep per column and split for 24 heads). Together with the
  two changes above, against master: K8V4 -5.5 to -58 % (3584 columns -31.5 % over an empty
  context, -26 % over 8K, -22 % over 32K, -15 % over 128K; 4096 columns over an empty context
  -58 %), FP8 -5.7 to -59 %, faster at every point measured (256-4096 columns, 0-128K keys).
- Split workspace: NInfer's partition reserves FP32 partials for its full split target, about
  550 MiB at 3328-column launches with 4096-token prefill chunks (169 MiB at its default 1024),
  which comes out of the KV cache. FP8 and K8V4 keep that (their workspace is master's); NVFP4 on
  this kernel keeps its old 64 MiB budget (below). The budget costs every format alike: with 64 MiB
  the same kernel takes 17-28 % more time at 1024-2048 columns over 32K-128K keys (K8V4 +27.7 % at
  1024 over 128K, FP8 +25.0 %, NVFP4 +27.5 %), 3-14 % more over 8K, 3-6 % more at 4096 columns over
  32K-128K, and the same at 256-512 columns and full 3584-column chunks. The split cost model
  predicted these penalties (+29 % predicted at 1024 over 128K) and put a 192-256 MiB budget within
  5 % of unbounded everywhere.
- Split-workspace sweep (`--prefill-split-workspace-mib`, one envelope bound for all four prompt
  kernels): 3 interleaved passes per budget, H24, widths 256-4096 over 0/8K/32K/128K keys, against
  16 GiB (unbounded). Mean over 1024/1408/2048 columns at 8K/32K/128K keys: 64 MiB INT8
  +5/+17/+20 %, FP8 +8/+19/+23 %, K8V4 +10/+20/+25 %, NVFP4 +7/+19/+25 %; 128 MiB 0/+2/+4-5 %;
  192 MiB and up within 0.7 %. Worst cells: 64 MiB +23-27 % (1024 columns over 128K); 128 MiB
  +10-13 % (3072 columns over 128K); 192-256 MiB +4.5-5.9 % (4096 columns over 128K); 384 MiB
  +0.8-2.3 %. No split (0) makes 256-column FP8/K8V4/NVFP4 launches over 128K 3.1-3.3 times slower
  and INT8 1024-column ones 1.6 times. The partials of one launch at 256 MiB peak at 233-242 MiB
  (1408-2560 columns); the tiled kernel always writes one split's rows (85 MiB at 3584 columns), so
  budgets under that are equivalent for full chunks. Startup workspace (4096-token chunks,
  `--max-context 220000`, DFlash2): 380 MiB at 64 MiB, 515 MiB at 256 MiB, 809 MiB at 16 GiB (FP8
  and K8V4 before the bound). End to end, fresh 64K and 128K prompts (whose last chunks are 1024
  and 2048 tokens wide) prefill within 0.3 % at all three budgets for all four formats (2 runs
  each). Default 256 MiB.

## NVFP4 on the decode-once kernel (replaces the register-decode fast kernel)

The fast NVFP4 prompt kernel decoded V in registers in each of its eight warps and accumulated
P×V in FP16 per tile; it ran about as fast as the MX-FP8 kernel did with FP32 accumulation (3584
columns over 128K keys: 38.6 against 38.7 ms). The MX-FP8 kernel gained a key-format policy and
runs NVFP4 keys: Q enters as the same two NVFP4 terms (q_terms.cuh) and QK runs on block-scaled
FP4 Tensor Cores straight from the stored codes and scales; V (the same NVFP4-G16 codec as K8V4's)
is decoded once per CTA into the FP16 arena. 88 KB of shared memory, one CTA per SM, 56-96 bytes
of stack (the register-decode kernel: 120-216). Its attention-suite error against the oracle is the
register-decode kernel's (relative L2 0.0017). Against master's fast kernel, with the issue order,
split cost and split budget: -6 to -26 % wherever it runs (3584 columns -7.8 % over an empty
context, -14.6 % over 8K, -15.5 % over 32K, -17.5 % over 128K; 1024-2048 columns -6 to -16 % from
8K keys; 0.32-0.63 times NInfer's tiled NVFP4 kernel).
Before the issue order and split cost it was 13 % slower at 3584 columns over an empty context and
2.2 times slower at 4096. Its splits keep the old kernel's 64 MiB partial budget (one split's
FP32 rows are always written for the merge, so wide launches reserve up to about 85 MiB, against
64 MiB before); without the budget it reached the same 550 MiB as FP8 and K8V4 and took a further
14-22 % less time at 1024-2048 columns over 32K-128K keys (2048 columns over 128K: 18.3 against
22.9 ms), the width of a follow-up prompt over a long cached prefix (the default budget is now
256 MiB, split-workspace sweep above). Launches over at most 768 visible keys keep the tiled
kernel (the threshold re-tune is below).

## INT8 on the decode-once kernel (not adopted)

The same kernel with an INT8 key policy (one INT8 Q code and FP32 scale per row and 64-dimension
group, s8 QK per group with both group scales in FP32) and INT8 V decoded once into the arena
used exactly the 99 KB of shared memory a block may have and 0-8 bytes of stack, and matched the
fast INT8 kernel's error (relative L2 0.0017). With NInfer's split counts it took 9-20 % less
time than the fast INT8 kernel for 1024-2048 columns from 8K visible keys, but 7 % more for
1408-2048 columns over an empty context, 9 % more for 3584 columns over an empty context and about
the same for full chunks over long contexts. Those wins came from splitting into up to eight
(363 MiB of partials, against the fast kernel's 64 MiB budget): within the same 64 MiB it was
0.7-2.2 % slower at 1024-2048 columns over 32K-128K keys and still 6-7 % slower over an empty
context, so INT8 keeps the fast kernel. The fast INT8 kernel takes the same gain from a larger
split budget: at 256 MiB it takes 14-17 % less time at 1024-2048 columns over 32K-128K keys than
at 64 MiB (split-workspace sweep above). A first version also merged the splits with the inverse D256 rotation K8V4 and
NVFP4 need; INT8 rotates only Q and K, and the suite caught it (relative L2 1.41, the
norm-preserving mismatch).

## 8-bit P×V for NVFP4 and K8V4 (E4M3, opt-in)

In the decode-once kernel the V arena holds E4M3 bytes instead of FP16 (half the shared memory)
and P×V runs on block-scaled E4M3 Tensor Cores with FP32 accumulation, four times the rate of the
NInfer FP32-accumulate form. Probabilities are formed as 256 p against the tile's own row maximum:
the softmax reference follows each tile's row maximum (held within 2^40 of the running maximum,
so the FP32 accumulator and row sum cannot overflow), which keeps values down to 2^-17 of that
tile's largest instead of flushing a tile's whole diffuse tail below the running maximum's E4M3
range. V decodes as in the FP16 form, scaled by the power of two that brings the tile's largest
scale into [32, 64) (shifting small scales up as well), and is rounded once to E4M3; the shift
returns exactly through the MMA's UE8M0 B scale. Transposed byte-pair loads of the arena split
into an even- and an odd-dimension n8 tile in the permuted key order (2t, 2t+1, 8+2t, 9+2t per
half) that the lane's own QK probabilities already hold. Error against the oracle: relative L2
0.026-0.037 (FP16 form 0.0017-0.0018), within 4.5 % of the largest reference value pointwise; the
suite checks these cases against a criterion derived from E4M3 rounding of both operands. Per
layer, against the default FP16 form: K8V4 0 to -20 % (no change at 256 columns over an empty
context, -6 % at 512-1408 there) and NVFP4 0 to -20 % (no change where the tiled kernel still runs,
at most 768 visible keys); 3584 columns over 128K keys -18.9 % and -17.3 %. The same E4M3 rounding inside the register-decode NVFP4 kernel
did not pay (above).

## KL divergence of the 8-bit P×V forms (and the instrument for it)

Perplexity sees one token's probability, so it cannot tell a small systematic shift from corpus
noise: the bundled `ninfer-ppl-1m-v1` moves by up to 0.36 nats per token on single streams for any
small numeric change. `infernix-perplexity --save-top-tokens` and `--kl-reference` measure the
distribution shift directly: the reference run (BF16 KV) records each scored position's 32 most
probable tokens and their log-probabilities, and a test run scores exactly those candidates plus its
own top-1 and reports KL(reference || test) over them and one bucket for all other tokens, by
context length, with top-1 agreement and the mean target-token NLL change. Merging the tail makes
the number a lower bound; the 32 reference tokens carry 0.952 of the reference mass at 64K.

Two artifacts were needed. On the production `qwen3_8_27b_nvfp4-nvidia` artifact (W4A4) everything
sits at a floor: KL(BF16 || INT8 KV) = 0.0835 and KL(INT8 KV || INT8 KV with 8-bit P×V) = 0.0782 per
stream, i.e. the change under test is indistinguishable from the A4 activation noise. On
`qwen3_8_27b_q6-dflash2` (BF16 activations) the floor is 20-30 times lower and the change stands
out. 64K windows, full corpus, 1,044,876 scored tokens, against a BF16 KV reference:

| KV + P×V form | mean KL | median | top-1 | 32-64K bucket |
|---|---|---|---|---|
| INT8, FP16 P×V | 0.00514 | 0.00012 | 98.92 % | 0.0071 |
| INT8, 8-bit P×V | 0.01108 | 0.00025 | 98.56 % | 0.0161 |
| INT8, original prompt kernel | 0.00505 | 0.00012 | 98.92 % | 0.0068 |
| NVFP4, FP16 P×V | 0.02055 | 0.00105 | 97.41 % | 0.0272 |
| NVFP4, 8-bit P×V | 0.02027 | 0.00107 | 97.40 % | 0.0271 |
| K8V4, FP16 P×V | 0.01337 | 0.00069 | 97.95 % | 0.0179 |
| K8V4, 8-bit P×V | 0.01408 | 0.00072 | 97.89 % | 0.0188 |

The E4M3 forms (NVFP4, K8V4) are numerically equivalent to FP16 P×V by this measure: within 1.05×
overall, within 1.13× in every bucket, within 0.06 pp of top-1. INT8's integer form doubles the
divergence, worse on all sixteen streams, with the growth concentrated in long context (the 32-64K
bucket ×2.28) — the same tail-flushing effect the 64K perplexity first showed, now measured without
corpus noise. The fast INT8 kernel's own FP16 P×V matches the original INT8 kernel (0.0051 against
0.0050), so the entire cost belongs to the 8-bit P. Since the production artifact hides all of this,
the criterion was fixed before the Q6 runs: a KV format keeps 8-bit P×V on by default only if its
mean KL stays within 1.10× its FP16 form's, the same holds in the longest buckets, and top-1
agreement drops by at most 0.1 pp. NVFP4 and K8V4 pass; INT8 fails, so `EngineOptions::
prefill_8bit_pv` is a three-way `PrefillPv8 {Auto, On, Off}` whose Auto keeps FP16 P×V for INT8 and
the E4M3 form for NVFP4 and K8V4, with `--prefill-8bit-pv` / `--no-prefill-8bit-pv` forcing either.
End to end (one request, DFlash2, 4096-token chunks, two runs each) the default forms remove 4.7 %
(NVFP4) and 5.0 % (K8V4) of prefill time at 64K tokens and 6.8 % and 7.5 % at 128K; INT8's opt-in
8-bit P×V would remove 4.2 % and 5.1 %.

## NVFP4 fast-prompt threshold re-tuned to 768 visible keys

The 2048-visible-key threshold of 2026-09 was set against the register-decode fast NVFP4 kernel,
whose launch cost made it lose to the tiled kernel below that. The decode-once kernel changed the
crossover: its fast launch is a single 128-row CTA per SM whose cost barely depends on the visible
length, and the tiled kernel scales with it instead. Forcing the fast kernel at every visible count
(3 passes, `--entry append`, cold cache, CUDA Graph, 256-3584 columns over 0-8192 cached keys, both
H24 and H16 geometries, new threshold-0 build against the same commit's tiled kernel): equal at
512-768 visible keys, up to 1.6x faster from 1024, up to 3.4x at 8192, and up to 1.6x slower for
256-512-column launches over an empty context, where the tiled kernel's narrower launch wins. A
threshold of 768 keys is within 0.03 % of the per-shape best over the whole sweep in both
geometries; 2048 costs 2.3-2.7 % of the total and up to 64 % in its worst cell (3584 columns over
512 keys, 2048 columns over none). 640 keys is marginally better for H24 but 11 % off best for
H16 at 768 columns, so 768 is the joint choice.

## Register spills in the MX-FP8 tiled prompt kernel: the NVFP4 slab loop

`ptxas -v` on the H24 and H16 instantiations showed the NVFP4 key path spilling 112-176 bytes per
thread (FP8 and K8V4 24-40, the FP16-P×V forms only; the 8-bit P×V forms of FP8 and K8V4 spill
nothing) at the 255-register cap. The spills are the fragment addresses and scale words of the
fully unrolled four-slab QK loop: at 255 registers the allocator gave up on live ranges that span
all four slabs, which also serialized their loads. Rolling that loop (`#pragma unroll 1`) removes
almost all of them (112-176 → 8-16 bytes) and is faster: per attention layer, 3 interleaved passes
over 256-3584 columns at 0-64K keys, H24 FP16 P×V -4.9 % (geomean, -4.3 to -5.7 % per shape), H24
8-bit P×V -3.7 %, H16 FP16 P×V -3.7 %, H16 8-bit P×V -4.2 %; end to end, NVFP4 prefill is 1.0 %
faster at 64K and 1.3-1.5 % at 128K (two runs each). Each slab still issues its sixteen MMAs
between two loads, so the Tensor Cores stay busy.

Two other ideas were built and dropped. Precomputing the PV V-fragment address bases (the swizzle's
XOR touches only address bits 4-6, so four per-lane bases plus immediates replace four XORs per
iterations) is neutral for FP8 (x1.0009) and 0.85 % faster for K8V4's FP16 P×V, both within pass
noise, and does not remove the remaining 24-40 bytes of FP8/K8V4 spill: 2-3 spill operations per
64-key tile are not worth a second code path. Combined with the rolled slab loop it changed nothing
further. The 32-register `bf[2][QKNt][2]` double-buffered key fragment was left alone: unrolling the
pipelined FP8 QK loop by two is what makes the load/mma overlap work, and the measured spills there
are small.

## Vector-quantized KV: the 8-bit PV form, and what its FP16 path's spills are worth

The VQ2 and K4V2 caches (the Vq2 and Q4KeyVq2Value storages) run a prompt kernel derived from the
fast INT8 one, and their V rows already sit in the INT8-G64 tile layout the INT8 8-bit PV form
consumes, so that form ports directly: each group's probabilities, scaled by that key's row scale,
become u8 codes against their row maximum over the tile, and `mma.m16n8k32.s32.u8.s8.s32` multiplies
them by the tile's INT8 V rows with exact INT32 accumulation — no V code is decoded again. The VQ
attention test's prompt cases run both PV forms at both tile widths (32 keys under 1536 visible,
64 above), with and without key splits, through the exact-window band and a stale window, and
through the cached entry, all against the FP64 oracle. On Q6 (BF16 activations, full corpus, 64K
windows, against a BF16 KV reference) the form costs VQ2 0.03666 → 0.03778 mean KL (1.03x, every
bucket within 1.05x, top-1 within 0.04 pp) and K4V2 0.02124 → 0.02319 (1.09x overall, 1.11x in the
32-64K bucket), both reproducible to four digits across two runs. VQ2 therefore takes the 8-bit
form by default alongside NVFP4 and K8V4, while K4V2 stays opt-in against the 1.10x longest-bucket
bound the other formats were held to. Per attention layer the form is 3.3-18 % faster at 3584-key
chunks over long context (geomean 0.97 over 256-3584 keys at 0-64K keys) and up to 5 % slower for
512-1024-key calls over an empty context; end to end it removes 5.0 % (VQ2) and 5.3 % (K4V2) of one
request's prefill time at 64K tokens and 6.8 % and 7.2 % at 128K. A budget sweep on VQ2 confirms the
256 MiB split-workspace default for these formats as well: 64 MiB is 14-26 % slower for
1024-2048-token chunks over 32-128K cached keys, 128 MiB 1-6.6 %, 256 MiB within 0.1 % of
unbounded, and full 3584-token chunks are unaffected.

The FP16 PV path spills 36-96 bytes of stack per thread (the 8-bit form: median 2), so three ways to
remove that were built and measured. Rolling either the group loop or the pass loop of the PV nest
removes the spills completely (median 0) and is decisively worse: 1.20-1.76x per shape, geomean
1.44, on both formats and in all three rolled variants, while the unchanged 8-bit path in the same
binaries measures identical within 0.7 % — the difference is the change, not the build. Shortening
the scale fragments' live range instead — filling them per pass, or loading each key step's pair
where it is used, both with the loops left unrolled — does not reduce the spills at all (median 36 →
48 bytes). The unrolled nest is what lets ptxas software-pipeline the ldmatrix/decode/mma chain, and
its spill traffic is far cheaper than losing that schedule, so the loops stay as they are. The whole
opportunity is bounded anyway: the near-spill-free 8-bit form is only 2.8 % faster per layer than
the FP16 one, and part of that gap is its different Tensor Core rate rather than the spills.
