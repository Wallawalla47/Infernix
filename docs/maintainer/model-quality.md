# Model-Level Numerical Quality

Op qualification ([Op development](op-development.md#6-qualification)) proves each kernel against a
naive FP32/FP64 oracle. This reference covers the step after it: whether a whole model, run
through the public Engine, still produces the same output distribution, how to measure that
without being misled by noise, and the results recorded so far. It applies to every change that
alters arithmetic at model scale: kernels, fusions, routes, quantization and chunking.

## 1. Rounding noise at model scale

On artifacts with 4-bit (NVFP4) activations, a change that only reorders arithmetic still moves
the output distribution measurably. A 1-ulp difference near an E2M1 rounding boundary flips a
whole FP4 step, and 64 layers compound it. The prefill chunk size is the cleanest example: it
changes only the order of operations (the GDN state is FP32, the convolution history keeps the
BF16 projections one pass uses, and activation divisors are static), yet:

| Official Qwen3.8-27B NVFP4 artifact, 409,687-token corpus, 14,336-token windows | Against one pass per window |
|---|---|
| Chunks of 256, 1024, 4096 or 7168 tokens (INT8 KV; 256 and 1024 also BF16 KV) | KL divergence 0.025-0.033, top-1 agreement about 0.960, perplexity -0.6 % to +0.8 % with no trend in the chunk size |
| BF16 KV against INT8 KV, both one pass | KL 0.026 (the same size) |
| Q6 artifact (16-bit activations), 256-token chunks | KL 0.0021, top-1 agreement 0.991 (15 times smaller) |

Two NVFP4 runs differ from each other more (KL about 0.026) than either differs from the same
weights run with 16-bit activations at the 4-bit sites (KL about 0.020): each run scatters around the 16-bit answer,
and the chunk size only picks the scatter. So:

- A perplexity difference under about 1 % between two NVFP4 builds or settings is noise unless it
  holds at several chunk sizes. Compare means over at least five or six chunk sizes, paired by
  chunk size, and report the spread.
- Prefer KL divergence to a fixed reference over perplexity: it measures the distance from the
  intended distribution, not whether noise happened to favour the corpus.
- The best reference for a given weight set is the same artifact with 16-bit activations
  (`EngineOptions::a16_activations`), scored in one pass with BF16 KV.
- Exact outputs (for example BF16 KV prefill, where no Infernix kernel differs from NInfer) must
  still match exactly; noise tolerance applies only to paths whose arithmetic changed.

## 2. Tools

| Tool | Measures |
|---|---|
| `infernix-perplexity --prefill-chunk N` | perplexity at a chosen prefill pass size ([guide](../perplexity.md)) |
| `infernix-perplexity --a16-activations --save-top-tokens FILE` | the 16-bit-activation reference distribution |
| `infernix-perplexity --kl-reference FILE` | KL divergence and top-1 agreement against a saved reference |
| `infernix_decode_quality_forced` | the decode route's log-probability of forced corpus tokens, paired position by position between builds ([benchmarks](../../bench/README.md#decode-quality)) |
| `infernix_decode_quality_gen` / `infernix_decode_quality_judge` | end-to-end greedy decode through the production speculative path, judged token by token against the 16-bit reference (Qwen3.8-Flash-Next: every expert bank W4A16) ([benchmarks](../../bench/README.md#decode-quality)) |
| `tools/quality/fast_path_gate.py` | the acceptance decisions of section 6 from the tools above |

Perplexity scoring runs prefill only; decode kernels, speculative verification and n-gram drafting
need the decode tools. The generator uses only Engine API that NInfer also has, so the
same source builds in an NInfer checkout for a like-for-like comparison.

## 3. Recorded results

Measured on 2026-10-05 on an RTX 5090 (CUDA 13.4, Windows), Gold-Star-Infer at `5fbfc3db` against
NInfer `68c54356` plus the Windows port (with only a `--prefill-chunk` option added to its
perplexity tool), official Qwen3.8-27B NVFP4 artifact unless stated. The reference in these runs
lifted only the 4-bit (MLP) sites to 16-bit activations; sites that run 8-bit activations (FP8
projections) kept them. `a16_activations` lifts those too, so it is a stricter reference: the KL
figures in sections 1, 3.1 and 3.4 are against the narrower one, and the decode comparison (3.3)
was repeated with `a16_activations`.

### 3.1 Where the activation error comes from

Only the MLP runs 4-bit activations on these artifacts. Each group alone at 16-bit activations,
KL to the reference at one pass / 4096-token chunks, with uncached prefill time for the corpus:

| 16-bit activations at | KL | Top-1 agreement | Prefill time |
|---|---|---|---|
| none (production) | 0.0200 / 0.0231 | 0.966 | 1.0x |
| attention, GDN, output projections (no 4-bit sites) | unchanged (bit-identical) | | 1.0x |
| MLP gate and up | 0.0146 / 0.0134 | 0.974 | 2.6x |
| MLP down | 0.0169 / 0.0184 | 0.969 | 1.4x |
| every 4-bit site (the reference here) | 0 | 1 | 3.0x |

`a16_activations` also lifts the 8-bit sites, so it is at least as slow as the last row.

### 3.2 Infernix against NInfer, prefill

Perplexity at six chunk sizes (512, 1024, 2048, 3584, 7168 and one 14,336-token pass), paired by
chunk size:

| KV | Infernix mean | NInfer mean | Infernix - NInfer | By chunk size |
|---|---|---|---|---|
| BF16 | 2.33446 | 2.33446 | 0 | identical at all six |
| INT8 | 2.32850 | 2.33129 | -0.12 % | Infernix lower at 5 of 6; t = -1.4, not significant |
| NVFP4 | 2.33694 | 2.33643 | +0.02 % | Infernix lower at 3 of 6; spread 0.54 %; worst +0.89 % at 1024 |

Infernix changes no arithmetic shared by every KV format, and its INT8 and NVFP4 prompt-attention
kernels are as accurate as NInfer's within this resolution.

### 3.3 Infernix against NInfer, decode

32 corpus segments of 1,536 tokens, 384 greedy tokens each (12,288 tokens per build), DFlash2 with
7 drafts and the proposal head, INT8 KV, judged on each build's own prefixes. The same generated
tokens were judged twice: by the study's reference (4-bit sites at 16 bits) and by the committed
`infernix_decode_quality_judge` (`a16_activations`, every site at 16 bits):

| Build | Agreement, 4-bit sites lifted | Regret | Agreement, `a16_activations` | Regret |
|---|---|---|---|---|
| NInfer | 0.9736 | 0.0080 | 0.9743 | 0.0073 |
| Infernix | 0.9736 | 0.0086 | 0.9762 | 0.0072 |
| Infernix with n-gram drafting (15 drafts, min match 12) | 0.9729 | 0.0087 | 0.9764 | 0.0071 |

Greedy decode chooses the reference's top token at least as often as NInfer's under both
references; the differences are a few tenths of a percent and change order between them, so they
are within noise. Builds share their first
54-68 generated tokens on average before rounding noise makes them diverge; Infernix with and
without n-gram drafting shares 267 of 384.

### 3.4 Rejected: adaptive (4 or 6) activation block scaling

Each 16-value activation block chooses the scale that maps its largest magnitude to 6 or to 4,
whichever reconstructs the block with less squared error. On the official artifact, over seven
chunk sizes, KL to the reference fell 4.3 % (not significant) and top-1 agreement rose 0.10
points at all seven, with perplexity unchanged. On the production NVIDIA artifact KL rose 12 %
(0.0191 to 0.0214) and perplexity 0.6 %. With `infernix_bench` (3 alternating passes, INT8 KV,
DFlash2), prefill was 0.4-0.75 % slower at 16K, 64K and 2K prompts in every pass and decode about
0.4 % slower per round. Not adopted: `quantize_nvfp4_k16` keeps the single amax-to-6 scale.

Also rejected for the same trade-off: 16-bit MLP activations in prefill (table 3.1). An 8-bit
activation path would need FP4 x FP8 block-scaled MMA, which takes UE8M0 scales per 32 values,
not NVFP4's E4M3 scales per 16.

### 3.5 Rejected: truncated QSA index planes

QSA selects the blocks a query attends to by scoring every complete 4-token block against its pooled
index key, one 128-wide BF16 key per block per QSA layer (256 B per block, 13 planes per lane, 832 B
per token; [the design record](qwen3_8-flash-next-design.md#96-long-context-qsa-kv-host-tier) owns the
plane's geometry). A coarser plane would read fewer bytes and return frames to the expert cache if the
key's low-order bits carried little ranking information. The idea comes from Fathom (arXiv
2609.17652), which stores a uniform 4-bit key channel-major as bit planes and picks a per-query read
depth by water-filling; its 1.67x is for an index in pinned **host** memory, and its own
device-resident control is no faster. This engine's pooled plane is device memory in every page
space, so only the byte and agreement questions remain. Fedor Suchkov evaluated them on the
`kido5217/Infernix` fork ([issue #5](https://github.com/kido5217/Infernix/issues/5), map
[#36](https://github.com/kido5217/Infernix/issues/36)) against the pre-registered bar of
[#39](https://github.com/kido5217/Infernix/issues/39) — selected-set agreement with the full-BF16
top-512 selection, per layer per step, must reach a cross-layer mean of 0.95, with no layer below
0.90 and a best-to-worst layer spread of at most 0.10 — on keys, selections and FP32 scores captured
from the engine at 16K and 140K contexts on an RTX 5090 (140K is the bar's >= 128K point;
[#40](https://github.com/kido5217/Infernix/issues/40); prototype and raw per-layer output on that
fork's unmerged branch `proto/qsa-index-truncation`,
[#41](https://github.com/kido5217/Infernix/issues/41)):

| store per block | bytes | vs BF16 | mean agreement, >= 128K | worst layer |
|---|---|---|---|---|
| BF16 leading 12 bits | 192 | 1.33x | 0.9618 | 0.9485 |
| BF16 leading 10 bits | 160 | 1.6x | 0.8575 | |
| BF16 leading 8 bits | 128 | 2.0x | 0.5203 | |
| uniform 4-bit code (3 planes) | 64 | 4.0x | 0.8421 | |
| uniform 4-bit code (4 planes) | 80 | 3.2x | **0.9213** | **0.8951** |
| uniform 4-bit code (2 planes) | 48 | 5.3x | 0.6997 | |
| uniform 4-bit code (1 plane) | 32 | 8.0x | 0.4491 | |

Not adopted. The 4-bit store is the arm the bar gated, and it fails both agreement conditions; the
only *byte-saving* arm that meets the three numeric conditions is the 12-bit BF16 cut, which saves
1.33x — about 0.9 % of the expert cache at 256K with four lanes, and far less at shorter contexts —
while requiring the pooled plane's layout to change and every reader (the pooling writer, the
Program's plane allocation, selection, attention and prompt) to unpack it. At 16K the 4-bit store
does clear the bar (0.9536 mean, worst layer 0.9209); the failure is at long context, where the
saving would matter. Fathom prices bit extraction at one to two integer operations per bit read, four times the per-bit cost of a
nibble scan, and this engine's own scan curve (2x time over 16x bytes from 8K to 128K) is the
signature of a latency-dominated scan rather than a bandwidth-bound one — so a smaller plane's byte
saving is plausible as a time saving but unevidenced, and it was not required to be.

The water-filling read depth that motivated the per-query budget degenerates on this index: the
budget the bar hands it is the depth rule's own maximum, so the constraint never binds and depth 4
lands on every channel. That is a measured property of the rule on these keys, not an implementation
shortcut.

Support would require a representation that reaches 0.95 agreement at a materially larger byte
saving (at least 3x), or index-plane pressure the deployed contexts do not have: the plane is 208 MiB
per lane at 256K and 832 MiB at 1M, against an expert pool of roughly 8,300 frames ([the design
record](qwen3_8-flash-next-design.md) §4.2). The bar's numbers come from a CPU-only prototype over
captures from the fork's engine revision `51601ea2` (2026-10-13, a descendant of this repository's
`bbc2771c`), not from the GPU measurement the rest of this section describes; scan time was not
measured and was deliberately non-gating.

## 4. Qwen3.8-Flash-Next: NVFP4 artifact against the Unsloth UD-Q4_K_XL GGUF

This compares Infernix's Flash-Next artifact, which imports NVIDIA's NVFP4 checkpoint
(`nvidia/Qwen3.8-Flash-Next-NVFP4`) bit-exactly
([design §6.1](qwen3_8-flash-next-design.md#61-recipes-qwen3_8_flash_next_nvfp4-a-and-qwen3_8_flash_next_nvfp4_dense8-b)),
with `unsloth/Qwen3.8-Flash-Next-GGUF` `UD-Q4_K_XL`. That GGUF is the file the Strata engine runs
on this machine. The quality figures were measured. The speed figures are **estimates, not
measurements**: no UD-Q4_K_XL support exists in Infernix.

### 4.1 Measured quality

Teacher-forced on identical token ids
([design §16.5](qwen3_8-flash-next-design.md#165-precision-boundaries-and-measured-quality-2026-10-04-rtx-5090)),
recorded 2026-10-07. Infernix ran recipe B (`e3be72cde`) with INT8 KV; Strata 0.1.40 ran UD-Q4_K_XL
with INT8 KV. The texts were 2,557 positions of code, a document and a chat transcript, all ≤ 1,024
tokens.

| | Perplexity (Infernix / Strata) | ΔNLL (Infernix − Strata) | Top-1 agreement | KL(Strata ‖ Infernix) mean / p99 |
|---|---:|---:|---:|---:|
| Code | 1.9050 / 1.8994 | +0.003 ± 0.013 | 95.7 % | 0.023 / 0.30 |
| Document | 9.7812 / 9.7209 | +0.006 ± 0.014 | 86.4 % | 0.052 / 0.59 |
| Chat | 3.4774 / 3.8520 | **−0.102 ± 0.023** | 90.7 % | 0.106 / 1.64 |
| All | 4.6635 / 4.8435 | **−0.038 ± 0.011** | | |

Infernix ties Strata on code and the document and is better on chat. Recipe B (8-bit dense) cost
+0.008 ± 0.010 nats against recipe A on an earlier checkpoint with the same BF16 dense weights;
recipe A was not converted from NVIDIA's checkpoint. The figures recorded here before 2026-10-07
(recipe A 4.564 against Strata 0.1.39's 4.864) were measured on `RadixArk/Qwen3.8-Flash-Next-NVFP4`,
whose routed experts carry a different calibration.

**This compares engines as well as weights.** Several of Strata's numerical choices differ from
Infernix's:
- It rounds 195 of the file's Q8_0 tensors to BF16 when it packs them: every hyper-connection
  up/down matrix, `output_hc_*` and the PLE value projection (`--compat-bf16`).
- It keeps an FP32 residual stream. In Infernix that measured +0.030 ± 0.009 nats worse.
- It quantizes expert activations to Q8_1.

How much of the chat gap comes from the weights alone has not been measured.

### 4.2 What the file contains

These facts come from the GGUF headers (4 shards, 111.3 GB).

| Tensor class | GGUF format | Infernix recipe B |
|---|---|---|
| Routed experts | Q4_K gate/up (Q5_K in layer 2); Q5_1 down (Q8_0 in layers 2, 4, 30, 46, 47). 71.73 GiB | NVFP4, 63.3 GiB |
| Bytes per expert | 3,072,000; 3,584,000 in the Q8_0-down layers; 3,993,600 in layer 2. **+13 %** on average | 2,764,800 |
| All dense projections, `lm_head`, embedding | Q8_0: the same codes and FP16 scale per 32 as `q8_g32_fp16`, so the same bytes | `q8_g32_fp16` |
| Router, GDN a/b, HC inject | F32 holding exact BF16 values | BF16 |
| QSA indexer | BF16 | BF16 |
| PLE n-gram table | IQ4_NL, 28.8 GB (90 B per row) | FP8, 51.2 GB |
| MTP layer, vision tower | **absent** | present |

At 4.5 bits per weight, Q4_K is the same size as NVFP4. The dense path has the same bytes in both
artifacts. The differences are the larger down projections, plus the format of the arithmetic.

Average bits per weight of the text decoder (176.944 G weights in each; scales counted, vision and
the MTP drafter excluded since the GGUF files hold neither):

| | Routed experts | Dense and other | PLE n-gram table | All | Without the n-gram table |
|---|---:|---:|---:|---:|---:|
| `nvidia/Qwen3.8-Flash-Next-NVFP4` | 4.50 | 16.00 | 8.00 | 5.83 | 4.95 |
| Infernix recipe B + n-gram volume | 4.50 | 10.34 | 8.19 (4 KiB block padding) | 5.73 | 4.73 |
| Unsloth UD-Q4_K_XL | 5.10 | 8.90 | 4.50 | 5.03 | 5.25 |

### 4.3 Conclusion

- **Quality.** A bit-exact UD-Q4_K_XL artifact in Infernix would likely land near Strata's quality,
  perhaps slightly above it, because Infernix can run every Q8_0 tensor exactly. It would likely
  stay below the NVFP4 artifacts, particularly on chat. This is unmeasured. Before such an
  artifact could be adopted, it would have to pass the same §16.5 comparison.
- **Speed** (estimates, against recipe B on the RTX 5090; nothing measured). Offloading narrows
  the gap. The causes are:
  - 13 % more bytes per expert;
  - about 12 % fewer VRAM frames (roughly 9,400 → 8,300), so more misses, each carrying more
    bytes;
  - expert GEMMs in prefill would run on INT8 tensor cores (half the FP4 rate) instead of
    `mxf4nvf4`.

  | Workload | Estimated slowdown |
  |---|---|
  | Warm decode (dense-path-bound, identical bytes) | ~5-10 % |
  | Cold or first-time decode and MTP verification (miss- and link-bound) | ~10-25 % |
  | Link-bound prefill | ~12-15 % |
  | GPU-bound long-prompt prefill | ~5-15 % |

  Strata's 72-80 tok/s on this file therefore reflects its engine, not the format.
- **Overall.** Such an artifact would probably be both slower and lower quality than recipe B.
  Its value would be a like-for-like comparison with Strata and a smaller n-gram table on disk.
  It is not built.

### 4.4 What support would require

Neither path below is built.

- **Converter.**
  - The existing GGUF source (`tools/convert/sources/gguf.py`) dequantizes to FP32. Using it here
    would quantize the values a second time, so a bit-exact artifact needs an encoded import of
    the raw blocks instead.
  - A `qwen4exp` name map, which must undo llama.cpp's export transforms (for example, `ssm_a`
    stored as −exp(A_log)). It is verified against the NVIDIA checkpoint's unquantized tensors.
  - Q8_0 copies into `q8_g32_fp16` exactly, provided no code is −128.
  - New formats `q4_k`, `q5_k`, `q5_1` and `iq4_nl`, each with an exact decode oracle.
  - An expert record layout for the mixed formats.
  - MTP and vision taken from the NVIDIA checkpoint.
- **Runtime.**
  - A second implementation family in `offloaded_sparse_moe`, which today is fixed to the NVFP4
    record (`kRecordBytes`). It needs:
    - a GPU narrow route;
    - a GPU wide route (INT8 MMA);
    - AVX-VNNI CPU miss kernels;
    - canonical arithmetic (integer A8 block products and a fixed-order epilogue), so placement
      invariance still holds.
  - Frame-pool size classes for the three record sizes.
  - IQ4_NL row decoding in the PLE gather.
  - About 72.4 GiB of pinned RAM instead of 64.5 GiB. About 77 GiB must be free at startup, or
    81 GiB with `infernix-serve`. Alternatively, the SSD expert tier.
  - Rough size of the work: 2-3 weeks, mostly the expert kernels.
- **Native GGUF loading.**
  - Loading this one file in place, as Strata does, is feasible. It is an explicit product change:
    `.ninfer` is the only artifact, and runtime repacking is forbidden. The 2-byte-aligned 34-byte
    Q8_0 blocks would slow the dense kernels unless repacked.
  - The preferred form is a thin `.ninfer`. It would reference the GGUF's byte ranges for the
    experts and the PLE table and embed only the repacked dense weights and the MTP layer. It adds
    about 6 GB on disk instead of about 106 GB. Each expert matrix is a whole number of 4 KiB
    pages, but tensor bases are only 32-byte aligned, so reads need aligned over-reads.
  - Arbitrary GGUFs would need every ggml type on every kernel route, plus each architecture's
    export conventions. That is not recommended.

## 5. Repeating the measurements

```bat
rem 16-bit reference, one pass per window
infernix-perplexity model.ninfer --text corpus.txt --context 14336 --stride 7168 --kv-dtype bf16 ^
  --prefill-chunk 14336 --a16-activations --save-top-tokens ref.top
rem a build or setting under test, at several chunk sizes
infernix-perplexity model.ninfer --text corpus.txt --context 14336 --stride 7168 --kv-dtype int8 ^
  --prefill-chunk 4096 --kl-reference ref.top
```

INT8, NVFP4, FP8 and K8V4 KV round the chunk down to whole 896-token prompt-attention waves, so
use a window that is a multiple of 896 (14,336 is 16 waves) to keep the one-pass reference one
pass. The decode-quality commands are in [the benchmark guide](../../bench/README.md#decode-quality).

## 6. Numerical acceptance policy and fast paths

This section is the policy for every change that alters model output. It fixes when a numerically
different route may ship, what it is measured against, how much it may cost, and how to test that.
`tools/quality/fast_path_gate.py` implements the decisions; its constants are the ones below.

### 6.1 The exact route and the fast paths

- **The exact route** is what runs with `--disable-fast-paths` (`EngineOptions::fast_paths`
  false). For a fixed artifact, Engine options (including the prefill chunk and the seed) and
  request, its output is bit-identical from run to run and **from release to release**. Where the
  artifact's formats support it (`LoadSummary::request_invariance_capable`: every decode
  projection registers a column-invariant route, today NVFP4 and FP8 dense text decoders with an
  FP8 or Q8 head, and every Qwen3.8-Flash-Next artifact) it is also **request-invariant**:
  concurrency, the other requests, lane
  assignment, prefix-cache
  contents and resource pressure do not change a request's output. Elsewhere its output also
  depends on the batch composition. Prompt and decode need not agree with each other. An option
  whose effect depends on load (lossy Host-tier prefix state, whose use depends on eviction) is
  refused with `--disable-fast-paths` at startup, never silently overridden. A
  change that alters any exact-route output bit is a *numerics change*. It is allowed only to fix
  a defect or to improve precision, and only when the gates of 6.5, with the previous exact route
  in the default role and the new one in the fast role, show the new route's estimated excess is
  at most zero and its bound within budget. Speed alone never justifies an exact-route numerics
  change: a faster route whose arithmetic differs is a fast path. The exact route is the quality
  reference's route and every gate's default run. Prompt attention there runs P×V in FP16 for every
  KV format; until 2026-10-13 NVFP4, K8V4 and VQ2 KV defaulted to their 8-bit forms, which are now
  the prompt-pv8 fast path. The fast INT8 and NVFP4 prompt kernels, measured when adopted, remain
  part of the exact route.
- **The fast paths** (`infernix::FastPath`) are routes whose arithmetic differs from the exact
  route's. By default (no `--disable-fast-paths`) the Engine runs every fast path qualified for the
  loaded **configuration**: the attention geometry, the activation class (the narrowest activation
  precision any text-decoder linear permits: 4, 8 or 16 bits) and the KV format (on Qwen3.5, also
  whether the text MLP gate/up is groupwise Q4, the format q4-w4a8 runs on). A path runs only
  in the configurations the registry (6.7) lists, so each configuration gets the fast paths that
  meet its own budget and no others; startup logs the selection (`fast paths | ...`) and
  `LoadSummary::fast_paths` reports it. A fast path stays deterministic for a fixed configuration
  and keeps every invariant the exact route guarantees: exact token constraints, sampling and
  speculative acceptance; Replay recovery and prefix-cache equivalence; masked-prefix and
  live-length invariance of attention; and whatever invariance an Op contract states. The
  request-coupling paths (6.7) give up request invariance, and only that. Only
  approximate arithmetic changes: reduction order, split schedules, private precision. A fast path
  never changes an exact codec, a stored representation, or the artifact.

A fast path is qualified for a configuration only when all three hold there:

1. **Op qualification.** Its kernels pass their suite's named criterion against the independent
   oracle over the full conformance domain ([Op development](op-development.md#6-qualification)).
   A fast path that changes an Op's arithmetic profile gets its own named criterion with a
   numerical derivation, under the same rules as any other route.
2. **Model-level acceptance, jointly.** The gates of 6.5 pass with the exact route in the default
   role and, in the fast role, the paths already qualified for the configuration **together with**
   the candidate, so paths that each fit the budget cannot add up past it. The prefill criteria
   apply when any selected path changes a prefill kernel, the decode criteria when one changes a
   decode, verification or drafting kernel; both when unsure.
3. **A resolved speedup** at the scope it claims, measured as any performance change
   ([Op development](op-development.md#7-performance-evidence), `bench/README.md`): alternating
   rounds, the gain ahead in every round. A fast path with no measured gain is removed.

Qualification evidence comes from a real artifact of the configuration; the quality tools'
`--fast-path NAME` options run a candidate before it is listed. The configurations, the kernels'
reach and the evidence are in [6.7](#67-fast-path-registry).

**Paths that leave attention and the KV cache alone** add the same excess whatever the KV format:
their kernels neither read nor write the cache nor change attention arithmetic (activation
precision, linear-layer tiles, GDN, the MLP). Two-term A4 residual activations added 2.2-2.5e-4
decode TV excess on INT8, NVFP4, K8V4 and K4V2 KV alike (6.7). The gate is joint, though, and
each format's other qualified paths already use a different share of its budgets, so the format
with the tightest budgets is not always the one with the least room. Such a path is qualified per
KV format in three steps:

1. Gate it on INT8 KV, the format with the tightest budgets (the lowest floor), in every arm where
   it acts. Its **increment** per criterion is that estimate minus the same arm's estimate with the
   qualified paths alone.
2. Gate it also on the qualified format with the least **headroom** on any criterion (its budget
   minus the 95 % bound of its qualified joint arm; on the 27B in 2026-10, NVFP4 KV's spec top-1,
   0.05 points).
3. Register it for each remaining format whose qualified 95 % bounds plus the INT8 increments stay
   within that format's budgets on every criterion; gate any other format as in step 1.

A path that changes attention or KV arithmetic (`prompt-pv8`, `decode-split64`,
`batch-split-attention`, adaptive speculation's verify kernels, `free-cache-resume`, Replay over
quantized KV) is gated on every format.

### 6.2 What is measured

- **The reference** for an artifact is the same artifact with 16-bit activations in every linear
  (`--a16-activations`), BF16 KV, each window or sequence in one pass. It holds the weights fixed,
  so it measures what the route's arithmetic costs, not what the weight format costs.
- **Cost, not difference.** Every route scatters around the reference: on the official 27B NVFP4
  artifact, two chunk sizes differ from each other by more (KL 0.025-0.033, section 1) than either
  differs from the reference (KL 0.020-0.023). The direct divergence between two routes is mostly
  noise both carry. A fast path is therefore charged only for its **excess**: how much further
  from the reference it is than the exact route, in KL divergence and in top-1 disagreement.
- **The noise floor** of an artifact and KV format is the exact route's own divergence from the
  reference, averaged over the tested chunk sizes. It is the computation noise the configuration
  already carries: **activation quantization** (4-bit activations raise it most), **KV-cache
  quantization** (the reference keeps BF16 KV, so INT8, FP8 or NVFP4 KV each raise it) and
  rounding. A fast path's errors are computation errors of the same kind, so where activations or
  KV are already rounded coarsely at every layer, a slightly different kernel rounding is lost in
  that noise. Budgets are fractions of the floor: a fast path may cost little where the arithmetic
  is precise and more with 4-bit activations or KV (6.6), and never more than a fixed ceiling.
- **Weight quantization is not part of the floor.** The reference keeps the artifact's own
  weights, so weight error is in neither route. It is a fixed change to the function the model
  computes, not noise in the computation: a model with 4-bit weights but 16-bit activations
  computes precisely, and nothing in it masks a kernel's extra rounding.
- **Prefill** is measured by `infernix-perplexity` (teacher-forced, every position of every window)
  at four chunk sizes, 896, 1,792, 3,584 and 7,168 tokens, with 14,336-token windows and a
  7,168-token stride over the full `ninfer-ppl-1m-v1` corpus (1.04 M scored tokens, 144 windows on
  Qwen3.8-27B). Three measures are paired window by window: KL divergence from the reference over
  its 64 most probable tokens plus one bucket for the rest (a lower bound that must resolve at least
  90 % of the reference's probability; 32 tokens resolve 95 % on Qwen3.8-27B), top-1 disagreement
  with the reference, and the corpus tokens' NLL (perplexity). NLL catches what truncated KL can
  miss, probability moved within the unsaved tail; in expectation it measures the same excess, but
  about ten times more noisily (6.4), so it is a gross cap rather than a bounded criterion.
- **Decode** is measured teacher-forced through the production decode route by
  `infernix_decode_quality_forced`: each of 1,024 segments prefills a corpus prompt (cycling through
  512, 2,048 and 8,192 tokens) and forces the next 64 corpus tokens one constrained step each
  (`TokenConstraint`). The segments lie back to back in the decode corpus `infernix-decode-v1`
  (`eval/corpora/decode-4m`: PG-19, WikiText-103 validation, Chinese Wikipedia and SGLang source,
  none of it in the prefill corpus), so no two share text. Every position has the same context in
  both routes, so the measures pair position by position (65,536 per route). The reference at
  those positions is the same 16-bit-activation, BF16-KV route, scored in one pass (`--reference`), and each forced step also
  reports the log-probabilities of the reference's 64 most probable tokens and its own largest one
  (`TokenConstraint::observe`). Decode therefore has the prefill measures (KL from the reference
  over its top 64 plus one bucket, top-1 disagreement, the forced tokens' NLL) and total variation
  over the same coarsening, which carries its distribution criterion (6.3). Its floor is the exact
  route's own decode divergence from the reference, measured in the same run; decode at one row uses
  16-bit activations where prefill uses the artifact's, so the decode floor can be lower.
  Free-running greedy decoding, judged on each route's own prefixes
  ([decode-quality pair](../../bench/README.md#decode-quality)), is an optional supplementary check:
  routes diverge after a few dozen tokens, which makes it far less sensitive.

### 6.3 Criteria

| Criterion | Statistic | Budget |
|---|---|---|
| Prefill mean KL excess | one-sided 95 % upper bound of the paired per-window excess KL(ref ‖ fast) − KL(ref ‖ default) | 10 % of the floor's KL, at least 1e-4 and at most 3e-3 nats |
| Prefill top-1 disagreement excess | the same bound for disagreement with the reference's top token | 10 % of the floor's top-1 disagreement, at least 0.05 and at most 0.5 points |
| Prefill corpus NLL excess | the paired mean excess NLL of the corpus tokens (log of the perplexity ratio), a point estimate; its bound is reported | at most 25 % of the floor's KL (at least 1e-4) |
| Prefill tail | 99.9th-percentile KL, mean over chunk sizes, fast / default | at most 1.25 |
| Prefill context buckets | each context-length bucket with at least 20,000 positions: excess mean KL | at most 25 % of that bucket's default KL (at least 1e-4) |
| Decode mean TV excess | one-sided 95 % upper bound of the paired per-segment excess total variation from the reference at the teacher-forced positions | 10 % of the decode floor's total variation, at least 1e-4 and at most 5e-3 |
| Decode top-1 disagreement excess | the same bound for disagreement with the reference's top token | 10 % of the decode floor's top-1 disagreement, at least 0.05 and at most 0.5 points |
| Decode KL and NLL excess | the paired mean excess KL and NLL of the forced positions, point estimates; their bounds are reported | each at most 25 % of the decode floor's KL (at least 1e-4) |
| Decode tail | 99.9th-percentile KL over the forced positions, fast / default | at most 1.25 |
| Decode prompt lengths | each prompt length's excess mean total variation | at most 25 % of that prompt length's default total variation (at least 1e-4) |
| Free-running decode (optional) | the same bound for the rate of generated tokens that are not the reference's top token, and for mean regret | 10 % of the default's rate (0.05 to 0.5 points); 10 % of the default's regret |

The bounded criteria carry the decision; the tail, bucket and prompt-length caps are point
estimates, like an Op criterion's gross pointwise cap, so damage confined to rare positions or long
contexts cannot hide in a mean. Decode has the prefill criteria at the forced positions, with its
own floor, except that total variation, not KL, carries the bounded distribution criterion. KL is
unbounded per position: the few positions whose most probable token flips under any rounding
change contribute several nats each, and they set its standard error so that no feasible decode
sample resolves a 10 % budget (6.4). Total variation counts each such position at most once and
still measures how much probability a fast path moves, the quantity recent work recommends for
estimating changed decisions (6.6); KL stays as a gross cap. Prefill keeps KL as its bounded
criterion: with about 250 times as many positions it resolves its budget. The decode gate first
bounded the forced tokens' NLL (until 2026-10-13), then their KL (2026-10-13); neither could resolve
a change that also alters the cached context (6.4, 6.7). The budgets' lower limits sit at the
measurement's resolution. The KL ceiling keeps any one fast path, on any artifact, below one step of the
weight-quantization ladder (q8_0 to q6_K costs about 4e-3 on Llama 3 8B in llama.cpp's published
measurements) and under half the 7e-3 that Fireworks calls a high-quality *whole* quantized
deployment (6.6).

### 6.4 Statistical method

Scoring windows are independent (each starts from empty state) and so are decode segments; they are
the units. For each unit the gate sums both routes over the chunk sizes, then forms the
ratio estimator (total excess over total positions) and its cluster-robust standard error. A
criterion passes when the one-sided 95 % upper bound, estimate + 1.645 standard errors, is within
budget, with at least 30 units. This is a non-inferiority test: the fast path carries the burden of
proof, and noise widens the bound, never the budget. A fast path whose effect cannot be resolved
needs more units (a longer corpus or more segments), not a looser criterion. A fast path that leaves
a phase bit-identical passes that phase with a zero bound, and the gate reports it as identical.

**Validation on real runs** (2026-10-13, official Qwen3.8-27B NVFP4 artifact, the prefill
protocol of 6.2 at 896 and 3,584 tokens against INT8 KV's exact route, decided with
`compare-prefill`; budgets: KL 1.55e-3, top-1 0.43 points, corpus NLL cap 3.87e-3):

| Candidate | KL excess (95 % bound) | Top-1 excess, points (bound) | Corpus NLL excess | Decision |
|---|---|---|---|---|
| A/A: the exact route at 1,792 and 7,168 tokens (reordering only) | 2.6e-5 (3.1e-4) | 0.004 (0.037) | 1.7e-3 | pass |
| The same pairs reversed | −2.6e-5 (2.6e-4) | −0.004 (0.030) | −1.7e-3 | pass |
| FP8 KV | 7.5e-5 (6.3e-4) | 0.046 (0.086) | 2.2e-3 | pass |
| INT8 KV with 8-bit P×V | 4.7e-5 (3.7e-4) | 0.020 (0.051) | 1.8e-3 | pass |
| Control: NVFP4 KV posing as a fast path over INT8 KV (a known loss) | **2.31e-3 (2.82e-3)** | 0.379 (0.417) | 2.7e-3 | **fail** |

- **The A/A comparisons pass** with KL and top-1 bounds about a fifth of their budgets, so a pure
  reordering does not fail spuriously, and the gate resolves an effect of about 3e-4.
- **The NVFP4-KV control fails**, so a real loss of 15 % of the floor is caught. Controls test the gate, not a KV format: a user's KV format is in both routes and sets the floor (6.2), so NVFP4 KV is gated against its own exact route and only a fast path's extra error counts.
- **FP8 KV and 8-bit P×V really are cheap** on this 4-bit-activation artifact: their costs are
  0.5 % and 0.3 % of the floor, with bounds well inside the budget. They would be acceptable fast
  paths here, but not necessarily on an artifact with a lower floor.
- **Corpus NLL is a gross cap, not a bounded criterion.** The A/A moved it by 1.7e-3 (perplexity
  +0.17 %), with a bound of 3.7e-3, larger than the KL budget. A perturbation of the logits changes
  the observed token's NLL at first order but the excess KL from the reference only at second
  order, so the same quantity measured on corpus tokens is about ten times noisier.

**Decode validation** (2026-10-13, same artifact, INT8 KV, plain decode, 64 forced tokens per
segment against the reference's top 64, decided with `evaluate-decode` and `compare-decode`). The
current protocol: 1,024 segments of the decode corpus (6.2), with these floors:

- total variation 0.0243;
- KL 0.0341 (512-token prompts 0.0055, 2,048 0.0119, 8,192 0.0852);
- top-1 disagreement 2.27 %;
- reference top mass 0.956.

Budgets: TV 2.43e-3, top-1 0.227 points, KL and NLL caps 8.5e-3.

| Candidate | TV excess (95 % bound) | Top-1 excess, points (bound) | KL excess (bound) | Decision |
|---|---|---|---|---|
| A/A: the exact route at concurrency 2 (batched 16-bit decode: rounding only) | 5.5e-5 (1.45e-4) | −0.012 (0.039) | 1.9e-4 (1.57e-3) | pass |
| `decode-split64` with `prompt-pv8` | −1.8e-4 (3.01e-4) | −0.090 (0.017) | −4.6e-4 (3.27e-3) | pass |
| Control: NVFP4 KV posing as a fast path over INT8 KV (a known loss) | **5.21e-3** (5.68e-3) | **0.49** (0.61) | 2.4e-3 (6.65e-3) | **fail** (also the 512-token bucket) |

Statistics compared on the same runs (bound as a share of its budget; the control must fail):

| Statistic | A/A | `decode-split64` + `prompt-pv8` | NVFP4-KV control |
|---|---|---|---|
| Mean KL (the 2026-10-13 criterion) | 52 % | 109 % | 222 % |
| KL, 10 % trimmed per segment (Yuen) | 0 % | 3 % | **56 %, passes** |
| Mean total variation (the criterion) | 6 % | 12 % | 234 % |
| Mean Jensen-Shannon | 24 % | 50 % | 207 % |

On the first protocol (below) total variation gives 2 %, 56 % and 287 %, and trimmed KL fails the
control (115 %) only because that protocol's KL budget is smaller.

- **The variance sits in a few chaotic long-context segments.** Five 8,192-token segments of the
  Chinese Wikipedia and SGLang parts of the corpus carry about 90 % of the KL variance of every
  comparison, the A/A's included. In them, the whole segment moves by up to 1.8 nats per position
  between runs: the most probable token flips under any rounding change. The decode floor's
  99.9th-percentile KL is 10.8 nats (prefill 0.9).
- **More text alone did not help KL.** The four times larger sample did not narrow the KL
  standard errors: 2.27e-3 against 2.62e-3 for the pair, 8.4e-4 against 8.0e-4 for the A/A. Its
  long-context text is more chaotic than the prefill corpus's (twice the KL floor).
- **Total variation fixes the resolution without dropping data.** It is bounded by 1 per position,
  so a flipped position counts at most once; the mean stays a mean over every segment. Trimming
  segments is too blunt: it discards the segments where a fast path's damage could sit, and here
  it passed the control. Jensen-Shannon is bounded too, but it is dominated by the same flips.
- **A change confined to decode kernels** was already resolved by KL (`decode-split64` alone,
  first protocol: bound 6 % of the budget).

The first protocol: 256 segments, cut evenly from the prefill corpus, judged on mean KL. Decode
floor KL 0.0167, top-1 disagreement 1.96 %, reference top mass 0.964; budgets KL 1.67e-3, top-1
0.196 points, NLL cap 4.17e-3:

| Candidate | KL excess (95 % bound) | Top-1 excess, points (bound) | NLL excess | Decision |
|---|---|---|---|---|
| A/A: the exact route at concurrency 2 | −7e-6 (1.31e-3) | −0.04 (0.05) | −4.1e-4 | pass |
| `decode-split64` alone | −5.6e-4 (9.6e-5) | −0.006 (0.089) | −3.7e-4 | pass |
| `decode-split64` with `prompt-pv8` | −1.19e-3 (**3.12e-3**) | 0.22 (**0.42**) | 2.4e-3 | **fail** |
| Control: NVFP4 KV posing as a fast path over INT8 KV | −4.2e-4 (3.23e-3) | **0.90** (1.13) | **8.0e-3** | **fail** |

Its top-1 bound also fails the pair under the current criteria: 256 segments are too few for
top-1. Before 2026-10-13 the decode gate bounded the forced tokens' NLL (resolution about ±9e-4
for a decode-only change). NVFP4 KV's NLL bound was 1.06e-2 and the INT8 pair's 1.19e-2, against a
1.55e-3 budget.

### 6.5 Testing compliance

`tools/quality/fast_path_gate.py` (Python 3.11, standard library only) runs the measurements,
reuses any run already present in its directory, and prints each criterion with its estimate, bound,
budget and result. Exit status 0 is a pass, 1 a failed criterion, 2 an unusable measurement.

```bat
rem Prefill: reference, then default and fast-path runs at four chunk sizes per KV format
py -3.11 -m tools.quality.fast_path_gate prefill model.ninfer ^
  --corpus eval/corpora/perplexity-1m/manifest.json --kv int8 bf16 --out profiles/quality/prefill
rem Decode: the reference at the forced positions, then default and fast-path teacher-forced decoding
rem (the decode corpus and 1,024 segments by default)
py -3.11 -m tools.quality.fast_path_gate decode model.ninfer --kv int8 --mode plain ^
  --out profiles/quality/decode-plain
rem Default route unchanged between two builds (perplexity runs, forced runs or generations)
py -3.11 -m tools.quality.fast_path_gate identical old-run new-run
```

The default run uses the exact route (`--disable-fast-paths`); the fast run uses every path
already qualified for the configuration, plus each `--fast-path NAME` candidate, so a candidate is
always measured together with the paths it will run beside. Without a candidate the gate re-checks
the configuration's current selection. `--mode` selects the decode route a fast path touches:
`plain` (no speculation, single-column kernels), `dflash2` (DFlash2 with 7 drafts and the optimized
proposal head), `ngram` (DFlash2 with n-gram drafting) or `mtp`; `--free-running` adds the greedy
generations and the judge. The request-coupling paths act only between requests, so the decode
gate runs both routes under the same Engine conditions: `--concurrency N` (requests per round),
`--kv-tokens N` (a short KV pool, so the fast run preempts) and `--prefix-reuse --warm-tokens N`
(each segment resumes from a warm request's cache entry). A schedule path that widens prefill
steps is qualified with prefill `--fast-chunk N`: every fast run prefills at that width against
the default run at each `--chunk` (a separate `--out`).
`evaluate-prefill DIR`, `evaluate-decode DEFAULT.json FAST.json --reference REFERENCE.json` and
`evaluate-free-running judge.json` decide existing runs. `compare-prefill --base RUN... --candidate
RUN...` decides any two sets of runs scored against one reference, paired in order, and
`compare-decode --base A.json --candidate B.json --reference REFERENCE.json` two teacher-forced runs
that observed one reference: a numerics change to the exact route (6.1), or a control. On
Qwen3.8-27B on an RTX 5090 the prefill gate takes 27 minutes per KV format (eight runs) after a
13-minute reference recorded once per artifact, and the decode gate 18 minutes per run after a
22-minute reference (60 minutes for a first gate; a later candidate reuses the reference and,
while the exact route is unchanged, its run).

The exact route needs no budget: `identical` compares two builds' perplexity runs window by window
(NLL, KL and top-1 agreement exactly), their teacher-forced decode runs position by position, or
their generations token by token. Run it at the production chunk size and KV format for any change
that could touch default arithmetic.

**Coverage.** Both Engine models score teacher-forced and have the 16-bit-activation reference.
Qwen3.8-Flash-Next's reference runs every dense linear A16 and every routed expert bank W4A16 (the
placement-invariant W4A16 arithmetic of its design §16.2.1), so its floor includes what the W4A4
experts' 4-bit activations cost; pass `--ngram-volume` to the gate when the volume is not
`<artifact>.ngram`. Its Engine scoring is checked against the Forward's own teacher-forced logits
(`infernix_qwen4_exp_score_real_test`): on 3,000 tokens the perplexity matches, top-1 agrees at
every position, and the RAM and SSD expert tiers give the same output. Its qualified paths are
three request-coupling paths and the three Q8 dense-kernel paths (6.7).

### 6.6 Calibration and comparison with other engines

**Floors measured on this machine** (RTX 5090, CUDA 13.4, 2026-10-12, full corpus and windows as
in 6.2, budgets from 6.3):

| Configuration | Floor: KL / top-1 disagreement | KL budget | Top-1 budget |
|---|---|---|---|
| Qwen3.8-27B NVFP4 (official artifact: 4-bit MLP, 8-bit projection activations), INT8 KV | 0.0155 / 4.30 % (0.0151-0.0159 across the four chunk sizes) | 1.55e-3 | 0.43 points |
| The same, BF16 KV | 0.0153 / 4.29 % | 1.53e-3 | 0.43 points |
| The same, FP8 KV (896 and 3,584 tokens only) | 0.0155 / 4.34 % | 1.55e-3 | 0.43 points |
| The same, NVFP4 KV (896 and 3,584 tokens only) | 0.0178 / 4.67 % | 1.78e-3 | 0.47 points |
| Qwen3.8-Flash-Next NVFP4 (NVIDIA's checkpoint: W4A4 experts), INT8 KV, quick corpus (2026-10-13; 1,792 and 3,584 tokens gave identical windows) | 0.0320 / 6.30 % | 3e-3 (ceiling) | 0.5 points (ceiling) |
| Qwen3.8-27B Q6 (16-bit activations), BF16 KV, 256-token chunks against one pass (section 1, an earlier corpus) | about 0.002 / 0.9 % | about 2e-4 | 0.09 points |

The floor is set mostly by activation precision: 16-bit activations give about an eighth of the
4-bit-activation artifact's. KV quantization adds to it: INT8 and FP8 KV add about 1 % to BF16
KV's on the 27B NVFP4 artifact, whose activations dominate, and NVFP4 KV adds 16 %.

A kernel's own formats do not fix its floor. Attention kernels are specific to their KV format but
run under any activation precision; linear kernels are specific to their weight and activation
formats but run with any KV format; norms, RoPE, the GDN layers and sampling serve every
configuration. The same fast path therefore meets a smaller budget where everything else is more
precise, and qualifying it on the highest-precision configuration its kernels run in usually covers
the others; the gate measures rather than assumes it, since 4-bit activations can amplify a small
change.

**How the budgets compare.** Published practice, surveyed on 2026-10-12, sorts into four levels:

| Level | Examples | Size |
|---|---|---|
| Task accuracy | MLPerf Inference (99 % of the FP16 reference's ROUGE or exact match, 99.9 % for one Llama-2-70B variant); NVIDIA's NVFP4 claims (about 1 point of FP8); Red Hat's quantization study (≥ 99 % recovery); TensorRT-LLM accuracy tests (3-point detectable effect on 0-100, α 0.05) | blind to kernel-level change: benchmark noise is 1-2 points |
| Whole-deployment distribution | Fireworks: generation KL < 7e-3 for a high-quality quantized deployment; Helcig et al. 2026: accuracy holds below KL 1e-2, "distribution-lossless" at acceptance ≥ 0.99 (about KL 2e-3 to 4e-3) | a whole quantization, not one kernel |
| Engine CI on non-bit-exact paths | vLLM: each side's token in the other's top-5 at the first divergence, GSM8K within 8 % (relative) of ground truth; SGLang: top-5 log-probabilities within 0.05 (prefill) and 0.06-0.08 (decode) of Hugging Face, GSM8K floors; llama.cpp: kernel NMSE 1e-7 to 2e-2 against the CPU backend, KL and perplexity reported for new quantizations with no threshold | catches gross errors, no distribution budget |
| Bit-exact | vLLM batch invariance, SGLang deterministic inference (opt-in, 20-60 % slower) | exact |

Strata keeps its default byte-identical to the last release and admits a numerics change by
comparing its KL against a reference path or the engine's run-to-run floor (KL about 6e-4 there),
with no fixed bound: its two non-exact defaults moved first-token or teacher-forced KL by a mean of
4e-3 and 5.8e-3 (top-1 agreement 97.7 %), and an experimental fast route at KL 6.3e-2 stayed opt-in.

**Choice of measure.** Compression studies use KL and top-1 "flips" as distribution measures
([Dutta et al. 2024](https://arxiv.org/abs/2407.09141)). More recent work finds that KL converts
into changed decisions only through its square root, with a factor that varies across models and
corpora, and recommends total variation for estimating how many decisions changed
([2610.00694](https://arxiv.org/html/2610.00694)); KL's unboundedness lets low-probability
mismatches dominate it ([2609.07664](https://arxiv.org/html/2609.07664v1)). llama.cpp reports
mean KL and "same top p" with per-token Gaussian standard errors, which ignore the correlation
within a sequence. Infernix's units are whole windows and segments. Robust estimators of a
heavy-tailed mean exist (trimmed mean, median-of-means, Catoni;
[Lugosi and Mendelson's survey](https://paperswithcode.com/paper/mean-estimation-and-regression-under-heavy)).
For decode, a bounded measure keeps the estimand a plain mean and validated better (6.4).

Infernix is stricter than each of these on what a change costs in fidelity to the reference. The
ceiling alone (3e-3 excess KL, 0.5 points of top-1) is below the 5.8e-3 Strata adopted and
Fireworks' 7e-3 for a whole deployment, the floor-relative budget is far below it on high-precision
artifacts, and the decision uses a 95 % confidence bound rather than a point estimate. The measures
differ in one respect: Strata compares the two routes directly, and that direct divergence also
contains rounding noise both routes carry, while Infernix charges only the excess distance from a
fixed reference. A fast path that merely reorders arithmetic on a 4-bit artifact can therefore
differ from the exact route by as much as two chunk sizes do (KL 0.025-0.033 on the 27B NVFP4
artifact, section 1) while costing no measurable fidelity; Infernix accepts that, because the
exact route itself carries the same scatter. No engine surveyed publishes a
per-change KL budget for kernel optimizations; scaling the budget with the artifact's own floor
follows the observation (llama.cpp's quantization ladder, Unsloth, Helcig et al.) that what a
model already loses to quantization sets the scale at which further error is visible.

### 6.7 Fast-path registry

Each entry records what the fast path changes, where its kernels run, the configurations it is
qualified for, and the evidence that admitted it. The configurations are the ones
`models/qwen3_5/program/planning/fast_paths.cpp` lists (`kQualified`); a configuration not listed
runs the path's exact counterpart. Qwen3.8-Flash-Next has three of the request-coupling paths
below and its own kernel paths (`models/qwen4_exp/program/fast_paths.cpp` lists their
configurations).

| Fast path | Kernels run with | Qualified configurations |
|---|---|---|
| `decode-split64` | INT8 KV, 24 query heads | 24 heads, 4-bit activations, INT8 KV |
| `prompt-pv8` | INT8 KV (fast prompt kernel), NVFP4 KV (fast kernel), K8V4, VQ2, K4V2 KV | 24 heads, 4-bit activations, with INT8, NVFP4, K8V4, VQ2 or K4V2 KV |
| `batched-decode`, `batch-split-attention` | request-invariance-capable artifacts | 24 heads, 4-bit activations, with every KV format (BF16, INT8, FP8, NVFP4, K8V4, VQ2, K4V2) |
| `batched-a8-decode`, `batched-a4-decode` | the same, with 8-bit (FP8) or 4-bit (NVFP4) activations | none (each fails, below) |
| `batched-a4-residual-decode` | the same, with 4-bit (NVFP4) MLP activations | 24 heads, 4-bit activations, with every KV format (BF16, INT8, FP8, NVFP4, K8V4, VQ2, K4V2) |
| `batched-a8-residual-decode` | the same, with 8-bit (FP8) input-projection activations | 24 heads, 4-bit activations, with every KV format (BF16, INT8, FP8, NVFP4, K8V4, VQ2, K4V2) |
| `adaptive-speculation` | the same, with speculative decoding | 24 heads, 4-bit activations, with every KV format (BF16, INT8, FP8, NVFP4, K8V4, VQ2, K4V2) |
| `free-cache-resume` | the same, with the prefix cache | 24 heads, 4-bit activations, with every KV format (BF16, INT8, FP8, NVFP4, K8V4, VQ2, K4V2) |
| `preemption` | request-invariance-capable artifacts | 24 heads, 4-bit activations, with every KV format (BF16, INT8, FP8, NVFP4, K8V4, VQ2, K4V2) |
| Qwen3.8-Flash-Next: `adaptive-speculation` (MTP), `free-cache-resume`, `preemption` | every Flash-Next artifact | INT8 KV |
| Qwen3.8-Flash-Next: `q8-sliced-k`, `gdn-conv-fused`, `qsa-prepare-fused` | Q8 dense weights (`qsa-prepare-fused`: BF16, INT8, FP8, NVFP4 or K8V4 KV) | Q8 dense, 4-bit experts, every KV format (`qsa-prepare-fused`: the five it serves) |
| `fp8-a8-pair` | row-FP8 dense weights (recipe C) | none (op level only, below) |
| `q4-w4a8` | groupwise Q4 MLP gate/up | none (fails, below) |
| `idle-wide-prefill` | Qwen3.8-27B / Qwen3.6-35B-A3B generation engines | 24 heads, 4-bit activations, with INT8, NVFP4, K8V4 or K4V2 KV |
| `adaptive-mtp` | Qwen3.8-27B / Qwen3.6-35B-A3B with MTP | none (no speed gain shown, below) |

**The request-coupling paths.** These were the default behaviour until the exact route became
request-invariant (2026-10-14). Each makes a request's arithmetic depend on the other requests of
the Engine, and each acts only where the artifact can be request-invariant
(`FastPath` documents each; `ExecutionSchedule` holds the flag each clears):
- `batched-decode`: decode and verify projections pick a route by the round's width (single-column
  GEMVs, small-T SIMT, MMA tiles) at 16-bit activations; the GDN input projection rounds to BF16
  above one row.
- `batched-a8-decode` and `batched-a4-decode`: with `batched-decode`, wide rounds take the
  artifact's 8-bit activations in FP8 projections (from 5 columns, or 17 in wide verification) and
  4-bit activations in NVFP4 projections (the MLP gate/up from 5, down from 8). Each **fails**
  the decode gate on the official 27B NVFP4 with INT8 KV (2026-10-14; 1,024 segments, against the
  2.43e-3 TV and 0.22-point top-1 budgets of a 2.43e-2 floor, jointly with the qualified paths
  that act in the arm), so both stay off and wide rounds run 16-bit activations:

  | Path, arm | TV excess (bound) | Top-1 excess, points (bound) | KL excess / cap | NLL excess |
  |---|---|---|---|---|
  | both, plain | 1.62e-2 (1.70e-2) | +1.84 (1.99) | 1.11e-2 / 8.6e-3 | -1.0e-3 |
  | `batched-a4-decode`, plain | 1.57e-2 (1.64e-2) | +1.81 (1.96) | 1.08e-2 / 8.6e-3 | -1.4e-3 |
  | `batched-a8-decode`, plain | 9.6e-4 (1.46e-3) | +0.04 (0.15) | -9.5e-4 / 8.6e-3 | -7.4e-3 |
  | `batched-a8-decode`, spec | 3.85e-3 (4.38e-3) | +0.43 (0.55) | 1.2e-3 / 8.6e-3 | -5.9e-3 |

  The plain arm (concurrency 8 with a four-lane pool) passes 8-bit activations because few of its
  rounds are wide enough to take them; DFlash2 with n-gram copies at concurrency 4 verifies 32
  columns, and fails. The 4-bit activations alone add two thirds of the floor at every prompt
  length. Perplexity cannot see either: the NLL excess is negative in every arm.
- `batched-a4-residual-decode`: with `batched-decode`, wide rounds of the NVFP4 MLP projections
  (gate/up and down from 9 columns) quantize each column's activations into NVFP4 terms that
  sum to it, the first being the 4-bit encoding and each later one the NVFP4 encoding of what the
  earlier ones left (`LinearPolicy::AllowA4Residual`), and multiply every term by the same weight
  tile into one accumulator. Weight traffic is the 4-bit route's; the activation error is roughly
  the 4-bit route's raised to the term count.
- `batched-a8-residual-decode`: with `batched-decode`, wide rounds of the FP8 attention and GDN
  input projections and MLP gate/up (from 17 columns) quantize each column into two E4M3 terms,
  the first the 8-bit encoding at the column's scale and the second what it left at a sixteenth of
  that scale (`LinearPolicy::AllowA8Residual`); each term has its own accumulator, summed exactly
  before the scales apply. Its error is about the 8-bit route's squared. Output and down
  projections keep 16-bit activations: two terms never beat A16 on their 5,120 outputs.
- `batch-split-attention`: decode attention sizes a row's key partition from the batch.
- `adaptive-speculation`: DFlash2 tree tables and controller, and a round's longest n-gram copy,
  set every row's verify width.
- `free-cache-resume`: prefill splits at taps and sibling boundaries, and resumes from any tap or
  endpoint (including decode-computed state).
- `preemption`: KV grows per unit, and a short pool pauses residents, which replay their generated
  tokens as prefill.

Qwen3.8-Flash-Next decodes request-invariant at every batch already (its plain and single-row MTP
output was identical across concurrency 1-8 and to plain decode, 2026-10-14), so only three apply
(`ProgramOptions` holds them): `adaptive-speculation` is its two-row MTP rounds, which share one
draft length chosen from both rows (without it two rows decode plain, as three or more do);
`free-cache-resume` is prefill calls, taps and resumes off its 16-token GDN sub-chunk grid,
sibling coalescing, and endpoints and blocks holding decode-computed state; `preemption` as above.
Its kernel path `q8-sliced-k` (below) couples requests too: its routes depend on the round's width.

**Evidence** (2026-10-14, RTX 5090, CUDA 13.4; official 27B NVFP4 and Qwen3.8-Flash-Next NVIDIA
NVFP4 Dense8; 1,024 segments of the decode corpus, 65,536 forced positions per route). Each
configuration was first shown request-invariant on 96 segments (the exact route bit-identical at
concurrency 8 with a four-lane pool, under DFlash2 with n-gram copies at 4, or Flash-Next MTP at 2,
and resuming from a warm request), so its one exact run is every arm's default. The arms ran
jointly, each with every path that acts there: **plain**, concurrency 8 with a four-lane pool
(`batched-decode`, `batch-split-attention`, and `preemption`, which acted only in its own arm
below); **spec**,
DFlash2 with 7 drafts and n-gram copies at concurrency 4 (adding `adaptive-speculation`;
Flash-Next: MTP with 4 drafts at concurrency 2); **cache**, each segment resuming at 3,000 tokens
from a warm request, its whole prompt for shorter ones, so off the chunk grid or from its endpoint
(adding `free-cache-resume`; 1,023,000 prompt tokens reused per run). Bounds are one-sided 95 %:

| Model, KV | Arm | TV excess (bound) / budget | Top-1 excess, points (bound) / budget | KL excess / cap |
|---|---|---|---|---|
| 27B, INT8 | plain | -1.7e-4 (3.3e-4) / 2.4e-3 | -0.06 (0.05) / 0.22 | -1.2e-3 / 8.6e-3 |
| 27B, INT8 | spec | -2.1e-4 (3.3e-4) / 2.4e-3 | -0.10 (0.01) / 0.22 | -1.3e-3 / 8.6e-3 |
| 27B, INT8 | cache | -2.0e-4 (3.6e-4) / 2.4e-3 | -0.10 (0.01) / 0.22 | -3.0e-3 / 8.6e-3 |
| 27B, NVFP4 | plain | 2.5e-4 (5.8e-4) / 2.9e-3 | +0.09 (0.19) / 0.28 | 1.3e-3 / 8.8e-3 |
| 27B, NVFP4 | spec | 2.3e-4 (5.5e-4) / 2.9e-3 | +0.13 (0.23) / 0.28 | 1.7e-3 / 8.8e-3 |
| 27B, NVFP4 | cache | 6.5e-5 (5.4e-4) / 2.9e-3 | +0.01 (0.12) / 0.28 | 2.6e-4 / 8.8e-3 |
| 27B, K8V4 | plain | 3.2e-4 (6.8e-4) / 2.7e-3 | -0.04 (0.07) / 0.26 | 3.7e-3 / 8.2e-3 |
| 27B, K8V4 | spec | 2.5e-4 (6.3e-4) / 2.7e-3 | -0.05 (0.07) / 0.26 | 2.0e-3 / 8.2e-3 |
| 27B, K8V4 | cache | 5.8e-4 (1.0e-3) / 2.7e-3 | -0.08 (0.04) / 0.26 | 4.0e-3 / 8.2e-3 |
| 27B, K4V2 | plain | 3.1e-4 (6.9e-4) / 2.6e-3 | -0.09 (0.02) / 0.24 | -1.2e-3 / 8.9e-3 |
| 27B, K4V2 | spec | 2.2e-4 (5.8e-4) / 2.6e-3 | -0.08 (0.02) / 0.24 | -1.5e-3 / 8.9e-3 |
| 27B, K4V2 | cache | 1.1e-4 (4.2e-4) / 2.6e-3 | -0.00 (0.10) / 0.24 | -4.5e-3 / 8.9e-3 |
| Flash-Next, INT8 | plain | 0 (identical: nothing preempted) / 5.0e-3 | 0 / 0.50 | 0 / 2.1e-2 |
| Flash-Next, INT8 | spec | 6.0e-5 (1.1e-4) / 5.0e-3 | +0.01 (0.01) / 0.50 | 1.0e-4 / 2.1e-2 |
| Flash-Next, INT8 | cache | 3.4e-4 (1.0e-3) / 5.0e-3 | +0.08 (0.20) / 0.50 | 1.5e-3 / 2.1e-2 |

**Preemption** has its own arm, since the four-lane pool never ran short: concurrency 8 with a
two-lane pool (16,640 tokens), jointly with `batched-decode` and `batch-split-attention`. On the 27B
every fast run paused and replayed (342 preemptions; 1,115,648 tokens replayed on INT8), the exact
route stayed bit-identical to its plain run under the same pool, and every criterion passes:

| 27B, KV | TV excess (bound) / budget | Top-1 excess, points (bound) / budget | KL excess / cap | p99.9 KL ratio |
|---|---|---|---|---|
| INT8 | -1.3e-4 (3.9e-4) / 2.4e-3 | -0.07 (0.04) / 0.22 | -3.1e-3 / 8.6e-3 | 0.93 |
| NVFP4 | 2.3e-5 (3.4e-4) / 2.9e-3 | +0.05 (0.16) / 0.28 | 2.5e-3 / 8.8e-3 | 1.19 |
| K8V4 | -6.3e-5 (2.7e-4) / 2.7e-3 | -0.09 (0.01) / 0.26 | -4.3e-4 / 8.2e-3 | 1.07 |
| K4V2 | 1.5e-4 (4.7e-4) / 2.6e-3 | -0.04 (0.06) / 0.24 | 1.2e-4 / 8.9e-3 | 1.08 |

Flash-Next's run under the same pool never preempted: it binds a request's whole prompt plus one
round, so only decode growth can run short, and 64 forced tokens rarely need a page. Its arm makes a
shortage certain instead: every prompt 8,200 tokens (a binding takes 129 pages and decode crosses
into a 130th), a 388-page pool (three bindings and one spare page) at concurrency 8, 256 segments
(16,384 forced positions per route). The fast run paused 96 times and replayed 792,480 tokens; the
exact run, holding whole extents, never paused. It passes: TV excess -8.7e-5 (bound 1.0e-4) against
5.0e-3, top-1 -0.04 points (bound 0.05) against 0.50, KL excess 3.7e-5 against a 1.9e-2 cap, p99.9 KL
ratio 0.93.

Every prompt-length bucket and p99.9 KL ratio passes too (largest ratio 1.13, 27B K8V4 spec). The
closest bound is 27B NVFP4 spec's top-1 at 82 % of its budget. Speed (official 27B, INT8 KV, 2,048
prompt, 256 output tokens, three rounds; the qualified set against the exact route): plain decode
81.83 vs 73.17 tok/s at one request (+11.8 %), +11.0 % at two, +6.9 % at four and +0.3 % at eight
(the 4- and 8-bit activations the old route used there are not qualified); DFlash2 with n-gram
copies 853 vs 523 tok/s at two requests (+63 %) and 583 vs 329 at four (+77 %). VQ2, BF16 and
FP8 KV were qualified later (below).

**`batched-a4-residual-decode`** (2026-10-14, the same artifact, corpus and arms): two NVFP4 terms
per column, from 9 columns in the MLP. Its op error against the 16-bit oracle is 0.9 % relative L2
on the [5120, 17408] down projection (one-term A4 6-12 %, A16 0.16 %); the gate/up suite's tiny
activations put its second term's scales near E4M3's subnormal range, so its error there (1.6-2.2 %)
does not fall with a third term. The plain arm never reaches 9 columns, so it also re-qualifies
`batched-decode` after its [5120, 17408] down tile changed (8 K-warps at 5-16 columns, 35.6-51.8
against 41.6-60.8 us). Every arm passes, jointly with the paths above:

| 27B, KV | Arm | TV excess (bound) / budget | Top-1 excess, points (bound) / budget | KL excess / cap |
|---|---|---|---|---|
| INT8 | plain | -2.2e-4 (3.1e-4) / 2.4e-3 | -0.10 (0.01) / 0.22 | -9.1e-4 / 8.6e-3 |
| INT8 | spec | 4.2e-5 (5.2e-4) / 2.4e-3 | -0.04 (0.06) / 0.22 | -8.3e-4 / 8.6e-3 |
| NVFP4 | plain | 2.3e-4 (5.5e-4) / 2.9e-3 | +0.09 (0.19) / 0.28 | 1.1e-3 / 8.8e-3 |
| NVFP4 | spec | 4.7e-4 (7.8e-4) / 2.9e-3 | +0.12 (0.22) / 0.28 | 2.1e-3 / 8.8e-3 |
| K8V4 | plain | 2.6e-4 (6.3e-4) / 2.7e-3 | -0.04 (0.07) / 0.26 | 2.8e-3 / 8.2e-3 |
| K8V4 | spec | 4.9e-4 (8.5e-4) / 2.7e-3 | +0.01 (0.13) / 0.26 | 2.8e-3 / 8.2e-3 |
| K4V2 | plain | 2.4e-4 (6.0e-4) / 2.6e-3 | -0.08 (0.03) / 0.24 | -1.8e-3 / 8.9e-3 |
| K4V2 | spec | 4.4e-4 (8.1e-4) / 2.6e-3 | -0.08 (0.02) / 0.24 | -1.2e-3 / 8.9e-3 |

Against the spec arms without it, the residual route adds about 2.5e-4 TV excess (a tenth of the
budget). Speed (INT8 KV, as above, the qualified set with and without it; the old route's 4- and
8-bit activations for reference): plain decode at eight requests 70.3 tok/s either way (64.7 before
the down tile, 72.6 on the old route); DFlash2 with n-gram copies 1,787 vs 1,280 tok/s at one request
(+40 %), 1,217 vs 852 at two (+43 %) and 887 vs 604 at four (+47 %), against the old route's 1,995,
1,650 and 1,286. Three terms ran 1,696, 1,074 and 783 and were not needed.

**`batched-a8-residual-decode`** (2026-10-15, the same artifact, corpus and arms): two E4M3 terms
per column in the FP8 attention and GDN input projections from 17 columns (the official 27B's MLP
is NVFP4, and its output projections keep 16-bit activations). Its op error is 0.14-0.22 % relative
L2 against the FP64 oracle (one-term A8 2.5-2.8 %). It passed under the KV-independent rule
(6.1), jointly with every path above: on INT8 the plain arm never reaches 17 columns (its verdict
is the qualified arm's to the bit), and the spec arm adds 2.2e-5 TV and 6.2e-5 KL excess; NVFP4
KV, the format with least headroom, passes its spec arm with the closest margin of any registered
path; K8V4 and K4V2 are registered by their qualified bounds plus the INT8 increment (closest: K8V4
spec top-1, 1.28e-3 of 2.65e-3):

| 27B, KV | Arm | TV excess (bound) / budget | Top-1 excess, points (bound) / budget | KL excess / cap |
|---|---|---|---|---|
| INT8 | plain | -2.2e-4 (3.1e-4) / 2.4e-3 | -0.10 (0.01) / 0.22 | -9.1e-4 / 8.6e-3 |
| INT8 | spec | 6.4e-5 (5.6e-4) / 2.4e-3 | -0.07 (0.04) / 0.22 | -7.7e-4 / 8.6e-3 |
| NVFP4 | spec | 4.4e-4 (7.5e-4) / 2.9e-3 | +0.14 (0.24) / 0.28 | 2.3e-3 / 8.8e-3 |

The same change retiled the two-term NVFP4 kernels above 32 columns without changing their
arithmetic (each output's K order is the tile's; op error identical at every width): gate/up
takes one token tile per round to 128 columns (64x128 to 64, 96x128 to 96, single-stage 128x128
to 128; 74, 95-101 and 134-138 us against 85, 138-142 and 144 us), and the [5120, 17408] down
projection keeps 32-token tiles to 64 columns (44-46 against 59-63 us). Speed (INT8 KV, 2,048
prompt, 256 output tokens, three rounds; DFlash2 with n-gram copies at one, two and four requests,
and plain decode at eight): the retiled qualified set ran 1,788, 1,288 and 922 tok/s and 70.4
(1,787, 1,217, 887 and 70.3 before); with two-term A8 1,915, 1,548 and 1,034 and 70.4, against the
old route's 1,995, 1,650, 1,286 and 72.6 (-4.0, -6.2, -19.6 and -3.0 %).

**Rejected** (2026-10-15): one-term A4 gate/up beside two-term down, with "four over six" block
scales (each 16-value block keeps the better of its max-to-6 and max-to-4 encodings). The scale
choice halves one-term op error (gate/up 9.8 to 5.0 %, down 7-9 to 5-7 % relative L2), and the mix
ran 1,944, 1,575 and 1,109 tok/s with two-term A8 (+1.5, +1.7 and +7.2 %), but it fails the INT8
spec arm: TV excess 1.14e-2 (bound 1.20e-2) against 2.4e-3 and top-1 +1.17 points against 0.22,
about two thirds of what one-term A4 in the whole MLP adds.

**What else was tried for wide rounds** (2026-10-15, the same artifact):
- Two-term NVFP4 MMA tiles at 65-128 columns with 32x64 warp tiles (one weight fragment feeding
  both terms of two token fragments), with 128-wide K stages three deep, or with both: none beat
  the tiles above by more than 2 % (gate/up 129.6-136.4 against 131.7-135.8 us at 97-128 columns,
  slower at 65-96; down no faster).
- The warp-specialized TMA route with two activation planes (three 128-K stages over one
  128-token tile, which the MMA tile's shared memory held one of) takes gate/up at 97-128
  columns: 101-103 against 128-138 us, with identical op error at every width (the same K order
  per output). DFlash2 with n-gram copies ran 1,051 against 1,025 tok/s at four requests (+2.5 %,
  within one run-to-run standard deviation of 28-30) and 1,534 against 1,538 at two.
- Two-term FP8 input projections take 32x128 tiles at 65-104 columns (attention: 58-62 against
  78-83 us at 65-96, 89-91 against 105 at 97-104) and at 65-96 (GDN: 87-91 against 91-99), with
  identical op error at every width. End to end nothing measurable changed at two and four
  requests (1,551 and 1,037 against 1,548 and 1,034 tok/s; their rounds are rarely that wide).
- A per-token activation scale for one-term A4 (no clipping, the widest E4M3 range). On real
  activations (DFlash2 with n-gram copies, four requests) the artifact's static scale leaves no
  block subnormal and clips 0.046 % of gate/up blocks (all but five by at most 2x) and 0.001 % of
  down blocks. Removing the clipping with a 4x or 16x wider static range moves one-term A4's INT8
  spec TV excess from 1.73e-2 to 1.72e-2 and 1.68e-2 and its top-1 excess from +2.02 to +1.92 and
  +1.86 points, against 2.4e-3 and 0.22: one-term's failure is the 4-bit grid itself, so the
  per-token scale was not built.

**VQ2 KV** (2026-10-15, the same artifact, corpus and arms; 1,024 segments): the exact route is
request-invariant there (96 segments bit-identical at concurrency 8, under DFlash2 with n-gram
copies at 4, resuming from a warm request, and under the two-lane pool), and every path that acts
passes jointly in each arm (the fast run paused and replayed 342 times in the preemption arm).
Its NLL excess is the largest of any format (2.8-7.4e-3, the cache arm's at 76 % of its budget):

| 27B, VQ2 | TV excess (bound) / budget | Top-1 excess, points (bound) / budget | KL excess / cap | NLL excess / budget |
|---|---|---|---|---|
| plain | 1.6e-4 (5.7e-4) / 3.0e-3 | -0.00 (0.11) / 0.28 | 2.6e-3 / 9.7e-3 | 5.1e-3 / 9.7e-3 |
| spec | 4.7e-4 (9.0e-4) / 3.0e-3 | +0.02 (0.14) / 0.28 | 3.7e-3 / 9.7e-3 | 4.5e-3 / 9.7e-3 |
| cache | 1.1e-4 (4.9e-4) / 3.0e-3 | -0.05 (0.06) / 0.28 | -1.2e-4 / 9.7e-3 | 7.4e-3 / 9.7e-3 |
| preemption | 2.6e-4 (6.8e-4) / 3.0e-3 | -0.04 (0.07) / 0.28 | 1.8e-3 / 9.7e-3 | 2.8e-3 / 9.7e-3 |

**BF16 and FP8 KV** (2026-10-15, the same artifact, corpus and paths), shortened at the user's
request: 512 segments (the first 512 of the same reference; bounds about 1.4 times wider than at
1,024) and no separate plain arm (the preemption arm runs eight requests per round as well). The
exact route is request-invariant on both (96 segments bit-identical under DFlash2 with n-gram
copies at 4, resuming from a warm request, and under the two-lane pool). Every arm passes, but
these formats' low floors make the KL and NLL budgets tight (2.4-2.5e-3), and the cache arm sits
close to them on point estimates whose 95 % bounds exceed the budget (BF16 KL 2.34e-3 of 2.47e-3,
FP8 NLL 2.345e-3 of 2.359e-3) while its TV excess is nil: a 1,024-segment cache arm would settle
whether `free-cache-resume` keeps that margin.

| 27B, KV | Arm | TV excess (bound) / budget | Top-1 excess, points (bound) / budget | KL excess / cap | NLL excess / budget |
|---|---|---|---|---|---|
| BF16 | spec | 1.8e-4 (2.9e-4) / 2.3e-3 | +0.03 (0.11) / 0.21 | 6.0e-5 / 2.5e-3 | 1.0e-3 / 2.5e-3 |
| BF16 | cache | -0.9e-5 (1.8e-4) / 2.3e-3 | +0.03 (0.12) / 0.21 | 2.3e-3 / 2.5e-3 | 2.1e-3 / 2.5e-3 |
| BF16 | preemption | -1.1e-4 (4.4e-5) / 2.3e-3 | -0.04 (0.03) / 0.21 | 3.8e-6 / 2.5e-3 | 3.1e-5 / 2.5e-3 |
| FP8 | spec | 3.0e-4 (4.2e-4) / 2.4e-3 | -0.09 (-0.00) / 0.22 | 2.0e-3 / 2.4e-3 | -7.7e-5 / 2.4e-3 |
| FP8 | cache | -0.0e-5 (1.6e-4) / 2.4e-3 | -0.04 (0.06) / 0.22 | 1.2e-3 / 2.4e-3 | 2.3e-3 / 2.4e-3 |
| FP8 | preemption | -2.2e-5 (9.7e-5) / 2.4e-3 | +0.01 (0.09) / 0.22 | 1.8e-3 / 2.4e-3 | -1.1e-3 / 2.4e-3 |

The preemption arms paused and replayed 85 (BF16) and 170 (FP8) times. With BF16 the default KV
of the CLI and server, the default route now runs the qualified fast paths.

**Qualification.** These act only above one request, so the decode gate runs them with
`--concurrency`, `--kv-tokens` (a short pool), or `--prefix-reuse --warm-tokens` (an off-grid
resume) on both routes. Where a path is not listed, the default route runs its request-invariant
counterpart, which costs decode throughput (27B: 10 % at one request, 6-12 % at two to eight),
prefix-cache reuse (no endpoints, so a conversation's next turn resumes from its last grid tap
and recomputes the previous output) or admission (whole extents held without `preemption`).

**`decode-split64`: INT8 KV decode attention, 64-key splits.** Measured on 2026-10-13 on an RTX
5090 (CUDA 13.4, Windows).
- **Selected by:** `CausalAttentionExecutionEnvelope::decode_split64` (`make_int8_kv_causal_plan`).
- **Scope:** INT8-G64 KV with 24 query heads (Qwen3.8-27B), single-column rows: plain decode, and
  the MTP drafter's single-column steps. Under speculative decoding those steps change only
  proposals, never output.
- **What changes:** each key split covers at least 64 keys instead of 128, so a short row spreads
  over twice as many CTAs. Only the softmax merge order changes.
- **Op qualification:** 29 cases in `run_int8_decode_split64_cases`:
  - 1 to 32,769 keys around split and tile boundaries, appended and cached, on fragmented pages;
  - an envelope far past the populated keys;
  - graph replay across short and long envelopes;
  - batched decode of up to 8 rows with mixed lengths.

  The worst case uses 8.6 % of the int8-g64 relative-L2 limit and 8.2 % of its gross cap; the
  default schedule's single-column cases reach 31 %.
- **Model-level acceptance** (official 27B NVFP4 artifact, INT8 KV):
  - prefill gate: bit-identical, so it passes trivially;
  - teacher-forced plain decode (KL gate, 6.4): KL excess −5.6e-4, bound 9.6e-5, against a 1.67e-3
    budget; top-1 excess −0.006 points, bound 0.089, against 0.196; every other criterion passes.
    The NLL gate it first passed (excess −3.7e-4, bound 5.6e-4) was superseded.
- **Speed:** plain decode, INT8 KV, three alternating rounds:

  | Prompt | Default → fast path | Rounds |
  |---|---|---|
  | 512 tokens | 81.76 → 82.06 tok/s, +0.37 % | every round ahead |
  | 4,096 tokens | +0.05 % | unresolved |
  | 16,384 tokens | no change | |

  The attention Op itself takes 16.2 → 13.7 µs at 512 keys and 16.0 → 14.2 µs at 2,048 keys, and
  is unchanged from 4,096. Longer rows already saturate the split target, so they keep their
  partition.
- **Not yet qualified:** 24-head artifacts with 8- or 16-bit activations, where the kernel runs
  unchanged but the budget is smaller (about 2e-4 at 16 bits); none exists on this machine.

**`prompt-pv8`: 8-bit P×V in prompt attention.**
- **Selected by:** `CausalAttentionExecutionEnvelope::fast_prompt_pv8`.
- **What changes**, by KV format:
  - INT8 (fast prompt kernel): each row's probabilities, scaled by the V group scale, become 8-bit
    codes per 64-key tile against the exact stored V codes; probabilities below half a code step of
    their tile's largest round to zero.
  - NVFP4 (fast kernel, over more than 768 visible keys) and K8V4: probabilities against each
    tile's own row maximum and decoded V under a per-tile power-of-two shift are rounded to E4M3,
    on block-scaled E4M3 Tensor Cores with FP32 accumulation.
  - VQ2 and K4V2: the VQ prompt kernel's integer form over the tile's INT8 V rows.
- **Op qualification:** each form has its own named criterion in the attention suite
  (`attention_criterion` with 8-bit P×V).
- **History.** Until 2026-10-13 NVFP4, K8V4 and VQ2 KV ran these forms by default, adopted on
  measurements against a BF16-KV reference on a model with 16-bit activations: NVFP4 and K8V4
  within 1.1× FP16 P×V's KL at every context length, VQ2 within 1.03×, all three removing 5-7 % of
  end-to-end long-prompt prefill time (up to a fifth of prompt attention). INT8 doubled FP16's KL
  at 64K context (up to 5 % faster prefill) and K4V2 raised it 1.09× (1.11× at 32-64K; 5.3-7.2 %
  faster at 64K-128K), so those stayed FP16. These measurements predate this policy and do not
  qualify a configuration.
- **Model-level acceptance** (2026-10-13, RTX 5090, CUDA 13.4; official 27B NVFP4 artifact; the
  full prefill protocol of 6.4: 144 windows, 4.18 million positions per route, chunks of 896,
  1,792, 3,584 and 7,168 tokens). Every prefill criterion passes for every KV format; KL and
  top-1 are excess over the exact route with their one-sided 95 % bounds:

  | KV | Floor KL | KL excess (bound) / budget | Top-1 excess, points (bound) / budget | Corpus NLL excess / cap | p99.9 KL ratio |
  |---|---|---|---|---|---|
  | INT8 | 0.0155 | 2.1e-4 (4.4e-4) / 1.55e-3 | 0.024 (0.046) / 0.43 | 8.9e-4 / 3.9e-3 | 1.019 |
  | NVFP4 | 0.0174 | 4.3e-4 (8.7e-4) / 1.74e-3 | 0.007 (0.027) / 0.47 | −3.2e-4 / 4.4e-3 | 1.018 |
  | K8V4 | 0.0165 | 2.7e-4 (6.4e-4) / 1.65e-3 | −0.001 (0.020) / 0.45 | −4e-6 / 4.1e-3 | 1.024 |
  | VQ2 | 0.0203 | 4.6e-4 (8.6e-4) / 2.03e-3 | 0.020 (0.041) / 0.50 | 1.2e-3 / 5.1e-3 | 1.043 |
  | K4V2 | 0.0169 | 1.2e-4 (3.3e-4) / 1.69e-3 | −0.005 (0.014) / 0.45 | 2.1e-4 / 4.2e-3 | 1.017 |

  The worst bound is half its budget (NVFP4 KL, 0.50; VQ2 0.43). Every context bucket's excess
  is under a fifth of its cap (largest INT8 at 2-4K, 1.1e-3 against 5.8e-3). An earlier partial
  run (INT8 KV, 896 and 3,584 tokens only) gave KL excess 4.7e-5 (bound 3.7e-4).

  NVFP4, K8V4, VQ2 and K4V2 KV select no decode fast path, and prompt-pv8 changes only prompt
  kernels, so the prefill criteria decide them. INT8 KV also runs `decode-split64`, so the pair
  needs the decode criteria too, and it **fails** them. Under the first, NLL-based decode gate:
  NLL excess 2.4e-3 with bound 1.19e-2 against a 1.55e-3 budget. Under the KL decode gate (6.4),
  first protocol: KL excess −1.19e-3 with bound 3.12e-3 against 1.67e-3, and top-1 excess 0.22
  points with bound 0.42 against 0.196. Under the current protocol (1,024 segments of the decode
  corpus): KL excess −4.6e-4 with bound 3.27e-3 against 3e-3 (109 %). Under the total-variation
  decode criterion that replaced KL (6.3), on the same runs, every criterion passes: TV excess
  −1.8e-4 with bound 3.0e-4 against 2.43e-3, top-1 excess −0.09 points with bound 0.017. The
  criterion changed after this pair had failed; the A/A and NVFP4-KV controls of 6.4 validate it
  on two protocols. prompt-pv8 was qualified for INT8 KV on 2026-10-14 on that evidence, by the
  user's decision, which also weighed how far inside other engines' acceptance criteria the pair
  sits (6.6): prefill KL excess 1.4 % of the floor (bound 2.8 %), decode TV bound 12 % of its budget.
- **Speed** (`infernix_bench`, prefill chunk 4,096, three repetitions per run, two runs per route
  in exact / fast / fast / exact order; every fast run beats every exact run):

  | KV | 16,384-token prompt | 65,536-token prompt |
  |---|---|---|
  | INT8 | 11,662 → 11,849 tok/s, +1.6 % | 8,514 → 8,801 tok/s, +3.4 % |
  | NVFP4 | 11,970 → 12,157 tok/s, +1.6 % | 9,247 → 9,669 tok/s, +4.6 % |
  | K8V4 | 11,926 → 12,126 tok/s, +1.7 % | 9,059 → 9,506 tok/s, +4.9 % |
  | VQ2 | 11,263 → 11,471 tok/s, +1.9 % | 7,987 → 8,437 tok/s, +5.6 % |
  | K4V2 | 11,300 → 11,530 tok/s, +2.0 % | 7,939 → 8,393 tok/s, +5.7 % |

  The path acts only in the prompt (tiled) attention kernels; decode throughput was not measured.
- **Not yet qualified:** 24-head artifacts with 8- or 16-bit activations (smaller budgets), and
  other attention geometries; none exists on this machine.

**`adaptive-mtp`: MTP draft length chosen per round (candidate, not qualified).**
- **Selected by:** `--fast-path adaptive-mtp` with `--spec mtp` on Qwen3.8-27B / Qwen3.6-35B-A3B;
  the policy is `models/qwen3_5/program/speculative/mtp_draft_policy.h`.
- **What changes:** `--draft-tokens N` becomes a ceiling. Each all-neural round of one row runs a
  draft length K in 1..N, and a round of several rows one common K among 1, 2, 3 and N, whichever
  maximizes the rows' expected committed tokens per unit of round time. Expected tokens come from
  each request's decayed per-position acceptance (carried through preemption and Replay with the
  request); a round of K drafts costs the fixed model 1 + 0.2 K plain rounds (calibrated on the
  captured graphs), so the choices repeat run to run.
  Every 16 choices the next longer K runs for two rounds so its acceptance stays measured. A choice
  reads only earlier rounds: drafts, acceptance and sampling are unchanged, but the verification
  width, and with it the target's linear routes, varies by round. Copy (n-gram) rounds keep their
  widths and propose at N.
- **Cost:** one Forward/Finish graph pair per extra (K, batch size): K < N for one row, K in 1..3
  for every batch size (about +4 MiB of graph allowance each).
- **Speed** (2026-10-15, RTX 5090, INT8 KV, `infernix_bench` pp2048+tg256 on the bench corpus,
  `--draft-tokens 5`): slower than a fixed five drafts everywhere measured. Official 27B NVFP4: 312.6
  against 326.1 tok/s at one request (−4.1 %), 311.8 against 314.9 at two (−1.0 %), 273.6 against
  276.9 at four (−1.2 %); 35B-A3B: 892.7 against 928.4 at one (−3.9 %). The corpus accepts almost
  every draft (mean accepted length 5.9 of 6), so the policy keeps K = 5 and loses only its probe
  rounds; text with low acceptance, where shorter drafts pay, was not measured. Not gated: no
  speed gain is shown.

**`idle-wide-prefill`: wider prefill steps while no request waits to decode (qualified
2026-10-15).**
- **Selected by:** `--fast-path idle-wide-prefill` on Qwen3.8-27B / Qwen3.6-35B-A3B generation
  engines; the rules are `models/qwen3_5/program/planning/idle_prefill.h`.
- **What changes:** a staged prefill step whose permit is reserved while no other lane is
  generating (Active or Pending) takes the idle width instead of one chunk. Startup plans 4096,
  2048 and 1024 tokens, each clamped and rounded to whole prompt-attention waves as the chunk is
  (3584 and 1792 on the RTX 5090 with 24 heads), then down to a multiple of the chunk, and keeps
  the widest whose larger prefill buffers and workspace keep the KV capacity (an explicit one
  whole, an automatic one down to every lane's full context); the `fast paths |` startup line
  reports it. Beside a lane that waits to decode the step stays one chunk, so that lane's stall
  is unchanged. Only the pass boundaries move, and every boundary stays on the chunk grid from the
  step's start: a wide step ends where a chunk-wide prefill would realize the lane's next flexible
  prefix-cache tap, and exact taps split it where they split a chunk, so snapshots land at the same
  positions. A Vision prompt, a binding's first step, a recovering request and Replay keep the
  chunk, as does a step whose extra KV pages are neither free nor evictable cache.
- **Cost:** prefill buffers and workspace at the idle width (with automatic KV capacity, fewer
  cached KV pages); a request admitted during a wide step waits up to one wide step.
- **Speed** (2026-10-15, RTX 5090, official Qwen3.8-27B NVFP4, INT8 KV, `infernix_bench`, one
  24,576-token prompt alone): 10,547.6 → 11,533.4 tok/s (+9.3 %). The upstream measurement (NInfer
  5bcef9e2c): 35B-A3B 1.41 → 1.26 s, 27B NVFP4 3.28 → 3.06 s, with the decode stall behind a
  13.8k-token prompt unchanged.
- **Evidence** (2026-10-15, RTX 5090, official Qwen3.8-27B NVFP4, quick corpus, its own a16
  reference). Scoring engines plan no idle width, so the gate runs the step the path takes: every
  fast run prefills at the idle width (`--fast-chunk 3584`, jointly with the qualified paths)
  against the exact route at the 896- and 1,792-token chunks below it (36 windows, 0.52 million
  positions per route). Every criterion passes:

  | KV | KL excess (bound) / budget | Top-1 excess, points (bound) / budget | NLL excess | p99.9 KL ratio |
  |---|---|---|---|---|
  | INT8 | 4.0e-4 (7.3e-4) / 1.2e-3 | +0.03 (0.09) / 0.41 | 5.1e-4 | 1.05 |
  | NVFP4 | 3.3e-4 (6.7e-4) / 1.4e-3 | +0.02 (0.09) / 0.45 | −3.4e-4 | 1.01 |
  | K8V4 | 2.8e-4 (9.4e-4) / 1.3e-3 | +0.05 (0.12) / 0.43 | −1.5e-3 | 1.04 |
  | K4V2 | −3.1e-4 (1.9e-4) / 1.4e-3 | −0.02 (0.04) / 0.44 | 1.3e-3 | 1.02 |

  A first run with `--fast-path idle-wide-prefill` at each chunk was void: the scoring engine
  never widened a step, so its fast runs measured only the qualified paths. The evaluation then
  needed `evaluate-prefill --fast-chunk` (fixed the same day: the pair check had required equal
  chunks).
- **Qualified:** 24 heads, 4-bit activations, with INT8, NVFP4, K8V4 or K4V2 KV.

**`q8-sliced-k`, `gdn-conv-fused`, `qsa-prepare-fused`: Qwen3.8-Flash-Next dense kernels (qualified 2026-10-15).**
- **What changes:** `q8-sliced-k` runs NInfer's GEMV, sliced-K and grouped sliced-K tables for the
  Q8 [16384, 2560] GDN projection and the [2560, 6144] GDN and QSA output projections, where they
  are faster than the exact routes (above T = 128, and at T = 321-384 and 449-511 on [16384,
  2560], they keep the exact routes). Their reduction order depends on the call's width, so it
  couples a request's decode to the other requests of its round (`request_invariant` is false).
  `gdn-conv-fused` feeds plain decode's GDN convolution the projection's FP32 q/k/v; and
  `qsa-prepare-fused` keeps FP32 between the QSA layer's norm, rotation and append steps. With them
  MTP output no longer equals plain decode.
- **Evidence** (RTX 5090, CUDA 13.4, Qwen3.8-Flash-Next NVIDIA NVFP4 Dense8, INT8 KV, the three
  jointly with the qualified paths; the exact route was bit-identical to the 2026-10-14 controls,
  so their exact run is the decode default):

  | Arm | KL or TV excess (bound) / budget | Top-1 excess, points (bound) / budget | Other |
  |---|---|---|---|
  | prefill, quick corpus, chunks 896-7,168 | KL −4.2e-4 (−1.1e-4) / 3.0e-3 | −0.06 (0.02) / 0.50 | NLL −2.2e-3, p99.9 KL ratio 0.96 |
  | decode plain, concurrency 8, four-lane pool | TV −1.0e-3 (8.0e-4) / 5.0e-3 | −0.13 (0.16) / 0.50 | KL −2.2e-3 / 2.1e-2, p99.9 ratio 1.01 |
  | decode MTP, concurrency 2 | TV −8.6e-4 (9.4e-4) / 5.0e-3 | −0.18 (0.12) / 0.50 | KL −3.3e-3 / 2.1e-2, p99.9 ratio 1.02 |

  Every prompt-length and context bucket passes.
- **The other KV formats** (2026-10-15, the same artifact, the three jointly). `qsa-prepare-fused`
  prepares every K/V append, so it is gated on each format it serves. Two reductions, chosen by
  the user to save GPU time:
  - Prefill runs at the 896- and 7,168-token chunks only, not all four (INT8 KV's floor was
    identical at 1,792 and 3,584, and none of the paths changes the chunk schedule).
  - Decode runs on BF16 (the lowest floor) and NVFP4 (the lossiest KV it serves) only, at 256
    segments with their own reference. FP8 and K8V4 rest on their prefill gates: a decode append
    is a prefill append's arithmetic, since `infernix_qsa_test` shows `qsa_prepare`'s columns
    bit-identical alone, batched and in a 896-token chunk on every format it serves.

  | KV | Arm | KL or TV excess (bound) / budget | Top-1 excess, points (bound) / budget | Other |
  |---|---|---|---|---|
  | BF16 | prefill | KL −1.3e-4 (3.1e-4) / 3.0e-3 | +0.06 (0.12) / 0.50 | perplexity ratio 0.9998, p99.9 KL ratio 0.94 |
  | FP8 | prefill | KL 2.7e-4 (7.1e-4) / 3.0e-3 | 0.00 (0.08) / 0.50 | perplexity ratio 0.9985, p99.9 KL ratio 1.00 |
  | NVFP4 | prefill | KL −3.6e-4 (4.8e-4) / 3.0e-3 | −0.05 (0.04) / 0.50 | perplexity ratio 0.9992, p99.9 KL ratio 0.95 |
  | K8V4 | prefill | KL 1.0e-4 (8.1e-4) / 3.0e-3 | −0.01 (0.08) / 0.50 | perplexity ratio 1.0009, p99.9 KL ratio 1.00 |
  | BF16 | decode plain c8 | TV −1.8e-4 (6.0e-4) / 5.0e-3 | +0.10 (**0.46**) / 0.50 | KL −1.8e-5, p99.9 KL ratio 0.94 |
  | NVFP4 | decode plain c8 | TV −3.3e-4 (5.4e-4) / 5.0e-3 | −0.21 (0.17) / 0.50 | KL −1.9e-4, p99.9 KL ratio **1.244** / 1.25 |

  Every criterion and bucket passes. Two are close: BF16's top-1 bound (92 % of its budget; the
  estimate is +0.10 points, the interval wide at 256 segments) and NVFP4's p99.9 KL ratio, which
  at 16,384 positions per route rests on about 16 of them. The BF16 floor (prefill KL 0.0316,
  decode TV 0.084) keeps its budgets at the ceilings, like INT8's. On VQ2 and K4V2 KV, where
  `qsa-prepare-fused` has no kernel, `q8-sliced-k` and `gdn-conv-fused` are registered from the
  INT8 increments (6.1, step 3): no other path is qualified there, and the increments' upper
  bounds (prefill KL 1.1e-4 below zero; decode TV 8.0e-4 and 9.4e-4) fit any budget above INT8's.
  Those formats' floors were not measured; their KV is lossier than INT8's, so they are taken to
  be at least INT8's.
- **Speed** (`infernix_bench`, two rounds): plain decode +1.4 %, MTP (four drafts) +3.4 to
  +4.2 %, prefill +0.3 to +1.0 % (qwen3_8-flash-next-design.md §19.3.22).
- **Not qualified:** 16-bit expert activations, and other dense formats (no kernels).

**`fp8-a8-pair`: two E4M3 activation terms in recipe C's FP8 projections (op level only).**
- **What changes:** prefill-width calls of Flash-Next's row-FP8 dense problems carry each 16-bit
  activation as two E4M3 terms on the 8-bit Tensor Core path, admitted per problem where faster
  than the exact route (cold L2, graph): [16384, 2560] from T = 17, [13312, 2560] from 97,
  [2560, 6144] and [2560, 640] from 513, [1280, 2560] from 1,025 (0.65-0.93 of the exact route
  beyond, except [2560, 6144] at T = 1,040-1,088: 1.18).
- **Op qualification:** `infernix_linear_fp8_a8_test`, the A16 criterion at every problem's
  admission seams and tiles (worst relative L2 0.78 of the criterion), outliers and K tails.
- **Model level:** not gated (no recipe-C artifact on this machine), so it runs in no
  configuration.

**`q4-w4a8`: int8 activations in the groupwise Q4 MLP gate/up (fails; not qualified).**
- **Selected by:** `--fast-path q4-w4a8` on artifacts whose text MLP gate/up is RowSplit
  `Q4_G64_FP16` (the official Qwen3.8-27B groupwise recipe; `FastPathConfiguration::q4_gate_up`),
  through `ExecutionSchedule::linear_routes` to `ops::linear_swiglu` with `LinearRoutes::Q4W4A8`.
- **What changes:** calls of at least 256 columns (prefill chunks; decode and verification stay
  narrower and keep their bits) quantize each token's activation to int8 against its own absmax and
  run an `mma.s8` kernel over the sign-extended Q4 codes: each 64-element group's dot product is
  exact in integers, its FP16 weight scale is folded in FP32, and the token scale applies at the
  end. The artifact's gate/up permits only 16-bit activations; the path overrides that permission
  for its calls, which is what the gate must show to be cheap. Within the route a column's bits do
  not depend on T or its place in the call.
- **Op qualification:** `infernix_linear_swiglu_q4_w4a8_test`, the int8 per-token criterion
  (relative L2 4e-2; its derivation is in the suite), at T 1-4,096, graph replay, exact bits below
  the threshold, and the A16 criterion on inputs int8 represents exactly. Passes (2026-10-15).
- **Cost:** 5,124 bytes of workspace per column of a gate/up call (37 MB at a 7,168-token chunk),
  below the exact route's 44.6 MB materialized-projection reservation up to about 8,700 columns.
- **Speed** (2026-10-15, RTX 5090, official Qwen3.8-27B groupwise artifact, INT8 KV,
  `infernix_bench`): prefill 3,349.7 → 3,741.8 tok/s at 4,096 tokens (+11.7 %) and 3,250.8 →
  3,624.5 at 16,384 (+11.5 %); decode 91.00 → 90.97 tok/s (unchanged: its calls are narrower than
  the threshold). Op level (cold L2, graph), the gate/up call at 0.71-0.80 of the exact route from
  T = 256. The upstream measurement (giveen/ninfer-ext b8a4ec826): 202 → 313 TFLOP/s at T = 1,024
  and 183 → 331 at 4,096.
- **Model-level acceptance: fails** (2026-10-15, quick corpus, the artifact's own a16 reference,
  four chunk sizes, 36 windows, 1.04 million positions per route). The artifact runs 16-bit
  activations, so its floor is small, and int8 per-token activations add several times it:

  | KV | Floor KL / top-1 | KL excess (bound) / budget | Top-1 excess, points (bound) / budget | NLL excess (perplexity ratio) |
  |---|---|---|---|---|
  | INT8 | 7.6e-4 / 0.93 % | 4.2e-3 (4.7e-3) / 1.0e-4 | +2.02 (2.24) / 0.09 | 2.5e-3 (1.0025) |
  | NVFP4 | 3.5e-3 / 2.24 % | 3.7e-3 (4.0e-3) / 3.5e-4 | +1.25 (1.38) / 0.22 | 1.3e-3 (1.0013) |

  Every criterion fails in both formats (p99.9 KL ratio 11.9 and 2.1), so the path runs in no
  configuration.
