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
- The best reference for a given weight set is the same artifact on the lossless route
  (`EngineOptions::lossless`: every operation of the model correctly rounded;
  [definition](lossless-route.md)), scored in one pass with BF16 KV.
- Exact outputs (for example BF16 KV prefill, where no Infernix kernel differs from NInfer) must
  still match exactly; noise tolerance applies only to paths whose arithmetic changed.

## 2. Tools

| Tool | Measures |
|---|---|
| `infernix-perplexity --prefill-chunk N` | perplexity at a chosen prefill pass size ([guide](../perplexity.md)) |
| `infernix-perplexity --lossless --save-top-tokens FILE` | a reference distribution over a perplexity corpus |
| `infernix-perplexity --kl-reference FILE` | KL divergence and top-1 agreement against a saved reference |
| `infernix_decode_quality_gen` / `infernix_decode_quality_judge` | end-to-end greedy decode through the production speculative path, judged token by token against a reference ([benchmarks](../../bench/README.md#decode-quality)) |
| `infernix-certify` | references, noise twins, teacher-forced prefill and decode arms, and reference-conformance captures |
| `tools/quality/iqs.py`, `certify.py`, `iqs_e2e.py`, `shadow/` | decisions over stored runs, the certification run, its end-to-end checks and its reference conformance |
| `infernix-certify reference --format llama-kld`, `prefill`/`decode --llama-base`, `llama-compare` | the exact reference as llama.cpp's `_logits_` base file, which `llama-perplexity --kl-divergence` scores llama.cpp against, and Infernix arms scored against it with llama.cpp's definitions and summary (`apps/certify/llama_kld.h`) |
| `tools/quality/llama_kld.py` | those arms' chunk-bootstrap intervals, paired differences, and llama-perplexity's printout read line by line |

Perplexity scoring runs prefill only; decode kernels, speculative verification and n-gram drafting
need the decode tools. The generator uses only Engine API that NInfer also has, so the
same source builds in an NInfer checkout for a like-for-like comparison.

## 3. Recorded results

Measured on 2026-10-05 on an RTX 5090 (CUDA 13.4, Windows), Gold-Star-Infer at `5fbfc3db` against
NInfer `68c54356` plus the Windows port (with only a `--prefill-chunk` option added to its
perplexity tool), official Qwen3.8-27B NVFP4 artifact unless stated. The reference in these runs
lifted only the 4-bit (MLP) sites to 16-bit activations; sites that run 8-bit activations (FP8
projections) kept them. `a16_activations` (16-bit activations at every site; replaced on 2026-10-20
by the precise reference route and on 2026-10-21 by the lossless route) lifted those too, so it
was a stricter reference: the KL figures in sections 1, 3.1 and 3.4 are against the narrower one, and
the decode comparison (3.3) was repeated with `a16_activations`.

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

`a16_activations` also lifted the 8-bit sites, so it was at least as slow as the last row.

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
rem the reference, one pass per window
infernix-perplexity model.ninfer --text corpus.txt --context 14336 --stride 7168 --kv-dtype bf16 ^
  --prefill-chunk 14336 --lossless --save-top-tokens ref.top
rem a build or setting under test, at several chunk sizes
infernix-perplexity model.ninfer --text corpus.txt --context 14336 --stride 7168 --kv-dtype int8 ^
  --prefill-chunk 4096 --kl-reference ref.top
```

INT8, NVFP4, FP8 and K8V4 KV round the chunk down to whole 896-token prompt-attention waves, so
use a window that is a multiple of 896 (14,336 is 16 waves) to keep the one-pass reference one
pass. The decode-quality commands are in [the benchmark guide](../../bench/README.md#decode-quality).

## 6. Numerical acceptance policy and fast paths

This section is the policy for every change that alters model output: when a numerically different
route may ship and what it must show. 6.1 defines the routes and how a fast path qualifies; 6.7 and
6.8 record the paths and the exact-route changes with their evidence. The model-level quality
standard a path is certified against is under revision and is not published here.

### 6.1 The reproducible route and the fast paths

- **The reproducible route** (called the exact route until 2026-10-21; the dated records below keep
  that name) is what runs with `--reproducible` (`EngineOptions::fast_paths`
  false). For a fixed artifact, Engine options (including the prefill chunk and the seed) and
  request, its output is bit-identical from run to run **on a given engine build and GPU** (another
  engine version may give other bits: a fix, a precision change or a speed change may alter them, and
  says so where it lands), and it is
  **request-invariant** on every artifact: concurrency, the other requests, lane assignment,
  prefix-cache contents and resource pressure do not change a request's output. Qwen3.5 decodes and
  verifies every projection under `LinearPolicy::A16ColumnInvariant`: on the problem's dedicated
  column-invariant route, or, where its format and shape register none, on its exact one-column
  route once per column. Such an artifact is unsupported: it decodes very slowly, and startup
  warns `UNSUPPORTED ARTIFACT: decode and verification will run VERY SLOWLY` naming the
  projections. Every official and published 27B/35B artifact has a dedicated route for every
  projection. The MTP drafter and its proposal head run their batched draft calls under the same
  policy: a sampled request accepts or rejects against its drafts, so they must not depend on the
  other rows either. The DFlash and DFlash2 drafters run their context appends and proposals one
  row per call on the exact route (every round verifies the fixed chain and the drafter runs for
  every row), and their attention takes the context capacity as its envelope rather than the
  round's longest row, whose length would otherwise pick its split route; so each row's drafts
  depend only on that row. (Before 2026-10-21 the round's envelope reached them: a sampled request
  admitted beside a longer one drafted with another split order and diverged within five tokens,
  while greedy requests stayed identical.)
  Qwen3.8-Flash-Next decodes rounds of up to eight rows on routes whose column results do not
  depend on the round's width up to eight columns (dense projections, router, hyper-connections,
  QSA attention, the narrow expert route). Recipe C's exact route diverged from run to run: its FP8
  shared expert's A16 GEMM, refilled by TMA, released shared-memory stages without the async-proxy
  fence, and when a forked expert call's staged pass shared the SMs a refill overwrote stages still
  being read (op-development.md 4.3). Its producer now stages with cp.async and every TMA-fed GEMM
  fences; its forced-decode audit then passes at one, four and eight requests, under MTP and from a
  warm prefix (below), as IQ3_S's and the FP8 import's do at one and eight. Its real tests are still
  to be rerun.
  **Speculation does not change its output** (the owner's decision of 2026-10-21): a greedy request
  accepts a draft exactly when it is the verify column's argmax, which is the decode column's; a
  sampled request draws every token by Gumbel-max (`SamplingConfig::gumbel_max`: the truncated
  support's token maximizing log p + g, with g a Gumbel variate keyed by the seed, the produced
  token's position and the token id), so a verify column returns the token plain decoding draws
  there and a draft is accepted exactly when it equals it. The default route keeps speculative
  rejection sampling (same distribution, other text for a seed). The real test
  `infernix_qwen3_5_speculation_invariance_real_test` checks MTP, DFlash2 and DFlash2 with n-gram
  proposals (or the drafters it is given) against one plain run, greedy and sampled, and that a
  raw one-token prompt verifies drafts too.
  Prompt and decode need not agree with each other. An option
  whose effect depends on load (lossy Host-tier prefix state, whose use depends on eviction) is
  refused with `--reproducible` at startup, never silently overridden. A
  change that alters any exact-route output bit is a *numerics change*: a defect fix, a precision change
  or a speed change (the owner's decision of 2026-10-25). It keeps the route deterministic and
  request-invariant, every Op it changes meets its criterion against its oracle
  ([op development](op-development.md)), and the configuration's prefill and decode arms run once
  on the reproducible route for that change against the stored reference, recorded beside the
  previous build's (a fix or precision change does not raise the point estimates of mean KL or flip rate,
  or the corpus NLL; a speed change records what it costs); the release note declares it and the
  golden reproducible hashes are recorded again (6.8 records each one). The reference runs its own
  kernels on every model, so an exact-route change leaves it unchanged. A faster route that couples
  requests or falls below an Op's criterion is a fast path instead. Prompt
  attention on the exact route runs P×V in FP16 for every
  KV format; until 2026-10-13 NVFP4, K8V4 and VQ2 KV defaulted to their 8-bit forms, which are now
  the prompt-pv8 fast path. The fast INT8 and NVFP4 prompt kernels, measured when adopted, remain
  part of the exact route.
- **The fast paths** (`infernix::FastPath`) are routes whose arithmetic differs from the exact
  route's. By default (no `--reproducible`) the Engine runs every fast path qualified for the
  loaded **configuration**, the kernels the artifact runs and its KV format. On Qwen3.5 that is the
  attention geometry, the activation class (the narrowest activation precision any text-decoder
  linear permits: 4, 8 or 16 bits), the KV format and whether the text MLP gate/up is groupwise Q4
  (the format q4-w4a8 runs on); on Qwen3.8-Flash-Next it is the dense kernel family of the GDN
  q/k/v/z projection (Q8, BF16, row FP8, NVFP4 or GGUF, every GGUF block format being one family),
  whether the routed experts take 4-bit activations, and the KV format. A configuration names
  kernels, never a checkpoint, so every artifact that runs the same kernels shares its paths. A path runs only
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
2. **Path fidelity, jointly.** The configuration's arms on the default route, which runs every path
   already qualified there **together with** the candidate (the measurement tools' `--fast-path
   NAME` adds a candidate before it is listed), meet the model-level quality standard against the
   stored reference (`tools/quality/certify.py`), on every KV format of the representative's family
   when the path changes attention or KV arithmetic. Paths that each fit the budget therefore cannot
   add up past it. Every route is judged against the reference alone: the reproducible route is an
   implementation like any other, never a baseline (owner, 2026-10-24).
3. **A resolved speedup** at the scope it claims, measured as any performance change
   ([Op development](op-development.md#7-performance-evidence), `bench/README.md`): alternating
   rounds, the gain ahead in every round. A fast path with no measured gain is removed.

**Approval follows the kernels a path changes** (since 2026-10-18). A configuration is the set of
kernels an artifact runs, never a checkpoint: another training instance, derivative or conversion
that runs the same kernels on the same formats inherits every qualification (the Swift and
Uncensored NVFP4 Dense8 conversions take Dense8's), and the next release certification measures it
with every other artifact. A registration made from a similar configuration's timing needs (a) the
kernels the path changes to run at the same shapes in both, (b) no cost the two would pay
differently (workspace taken from the expert cache or KV pool, more CUDA graphs, host work), (c) a
source gain clear of its run-to-run spread with room to spare, and (d) for a request-coupling path,
the same mechanism of gain (prefill not repeated, MTP kept at two rows, requests admitted under a
short pool). Each 6.7 entry names the configuration whose timing it takes. A kernel's gain at other
shapes does not carry: `gdn-f16-accumulate` gained 0.24 % on the 27B and nothing on Flash-Next.

A path that changes no logit (`sampled-mtp-drafts`: drafts from the head's top tokens, verified by
rejection sampling; `block-verification`: sampled chains with proposal laws accepted by block
verification, Sun et al., arXiv 2403.10444; `univer-trees`: sampled fixed-table trees of top ranks
and residual draws verified by UniVer, arXiv 2605.04543) is decided by its distribution test
(`iqs_e2e.py specdist`); it is registered on every configuration whose drafter it runs on.

**A failing path** (owner, 2026-10-20): a certification that fails a registered path removes it from
that configuration; a candidate that fails is not registered there. Either way the next more precise
variant of the path that keeps most of its speed (more residual terms, a narrower width range, the
fast arithmetic only where its gain is measured, a more precise reduction) is implemented, measured
with the same arms and timed. A failed run is never repeated to look for a pass.

### 6.7 Fast-path registry

Each entry records what the fast path changes, where its kernels run, the configurations it is
qualified for, and the evidence that admitted it. The entries up to 2026-10-22 were admitted under
the earlier gates: the excess of the default route over the reproducible route against the
lossless or precise reference (`tools/quality/fast_path_gate.py`, since removed), with the budgets of
2026-10-16 (mean KL 1.5e-3, top-1 disagreement 0.5 points, TV 4e-3). Their records stay as that
evidence. The first certification under the current standard decides every registered path again
against the stored reference; a path stays registered until a certification fails it (6.1). The
configurations are the ones `models/qwen3_5/program/planning/fast_paths.cpp` lists (`kQualified`); a configuration not listed
runs the path's exact counterpart. Qwen3.8-Flash-Next has three of the request-coupling paths
below and its own kernel paths, among them `expert-a4-residual`
(`models/qwen4_exp/program/fast_paths.cpp` lists their configurations). Since 2026-10-18 a
registration follows the kernels (6.1, items 1-4): the entries name the gate and the timing each
configuration takes.

**Re-scored under the single budgets (2026-10-16).** Every saved gate run in `profiles/quality`
was decided again by `fast_path_gate.py` with that day's budgets. Every registered configuration
still passes, and no failed candidate passes: the bounded criteria of every registered path stay
within budget (the closest, Flash-Next Dense8's kernel paths with BF16 KV, decode top-1 at 0.92;
27B prompt-pv8 with decode-split64, decode top-1 0.84), and the failed `batched-a4-decode`,
`batched-a8-decode` (spec top-1 at 1.11), `q4-w4a8` and the rejected two-term A8 mixes still fail.
At decode KL and NLL caps of twice the KL budget (3e-3), the 27B request-coupling arms on K8V4
(plain, cache) and VQ2 (plain, cache, spec) failed on those point estimates alone, inside their
standard error, which set the caps at 1e-2. Spec arms were scored against the exact run of
their configuration in plain decode, which the request-invariant exact route makes the same
tokens. The recipe C gates (`expert-a4-residual`, `fp8-a8-pair`) are re-scored in their entries.

The configurations by the artifacts that have them: **NVFP4 27B** (24 heads, 4-bit activations:
the official, NVIDIA, quasar and uncensored Qwen3.8-27B NVFP4 artifacts), **groupwise 27B** (24
heads, 16-bit activations, Q4 gate/up), **35B-A3B** (16 heads, 16-bit activations); for
Qwen3.8-Flash-Next **Dense8** (Q8 dense, 4-bit experts: NVIDIA, uncensored and Swift NVFP4
Dense8), **IQ3_S** (GGUF dense, 16-bit experts: every GGUF import), **FP8** (BF16 dense, 16-bit
experts: the official FP8 import), **recipe A** (BF16 dense, 4-bit experts) and **recipe C** (FP8
dense, 16-bit experts). "Every format" is BF16, INT8, FP8, NVFP4, K8V4, VQ2 and K4V2 KV. The
groupwise 27B and the 35B-A3B became request-invariant on the reproducible route in 2026-10 (6.1), so
their request-coupling paths are listed like the others (since 2026-10-22). The NVFP4 27B variants select identical paths (each startup's
`fast paths |` line, 2026-10-18).

| Fast path | Kernels run with | Qualified configurations |
|---|---|---|
| `decode-split64` | INT8 KV, 24 query heads | NVFP4 27B and groupwise 27B, INT8 KV |
| `prompt-pv8` | INT8 KV (fast prompt kernel), NVFP4 KV (fast kernel), K8V4, VQ2, K4V2 KV | NVFP4 27B, groupwise 27B and 35B-A3B, with INT8, NVFP4, K8V4, VQ2 or K4V2 KV |
| `batched-decode`, `batch-split-attention` | every Qwen3.5 artifact | 24 heads, 4-bit activations, with every KV format (BF16, INT8, FP8, NVFP4, K8V4, VQ2, K4V2) |
| `single-column-decode` | Qwen3.5 artifacts without `batched-decode`: decode and verify calls of one column | groupwise 27B, every format (2026-10-22, below); not the 35B-A3B (1 % slower at four requests) |
| `batched-a8-decode`, `batched-a4-decode` | the same, with 8-bit (FP8) or 4-bit (NVFP4) activations | `batched-a8-decode`: 24 heads, 4-bit activations, every KV format, by the owner's decision (2026-10-24) although it fails (below); `batched-a4-decode`: none (fails, below) |
| `batched-a4-residual-decode` | the same, with 4-bit (NVFP4) MLP activations | 24 heads, 4-bit activations, with every KV format (BF16, INT8, FP8, NVFP4, K8V4, VQ2, K4V2) |
| `batched-a8-residual-decode` | the same, with 8-bit (FP8) input-projection activations | 24 heads, 4-bit activations, with every KV format (BF16, INT8, FP8, NVFP4, K8V4, VQ2, K4V2) |
| `adaptive-speculation` | the same, with speculative decoding | 24 heads, 4-bit activations, with every KV format (BF16, INT8, FP8, NVFP4, K8V4, VQ2, K4V2); groupwise 27B and 35B-A3B, every format (2026-10-22, below) |
| `free-cache-resume` | the same, with the prefix cache | 24 heads, 4-bit activations, with every KV format (BF16, INT8, FP8, NVFP4, K8V4, VQ2, K4V2); groupwise 27B and 35B-A3B, every format (2026-10-22, below) |
| `preemption` | every Qwen3.5 artifact | 24 heads, 4-bit activations, with every KV format (BF16, INT8, FP8, NVFP4, K8V4, VQ2, K4V2); groupwise 27B and 35B-A3B, every format (2026-10-22, below) |
| Qwen3.8-Flash-Next: `adaptive-speculation` (MTP), `free-cache-resume`, `preemption` | every Flash-Next artifact | Dense8, IQ3_S, FP8, recipe A, recipe C (since 2026-10-21) and NVIDIA FP4-Local (16- and 4-bit expert activations), every format |
| Qwen3.8-Flash-Next: `q8-sliced-k`, `gdn-conv-fused`, `qsa-prepare-fused` | Q8 dense weights (`qsa-prepare-fused`: BF16, INT8, FP8, NVFP4 or K8V4 KV) | Q8 dense, 4-bit experts, every KV format (`qsa-prepare-fused`: the five it serves) |
| `fp8-a8-pair` | row-FP8 dense weights (recipe C) | FP8 dense, 16-bit experts, every KV format (INT8 fails its decode tail, below; registered by the owner's decision, 2026-10-24) |
| `fp8-a8-pair-block32` | the same, its two terms scaled per 32-value block (it takes precedence over `fp8-a8-pair`) | wherever `fp8-a8-pair` is, by the owner's decision (2026-10-25) ahead of its gate: tied in speed (recipe C, INT8 KV, pp4096 and pp16384 within 0.03 %, 4 blocks) and never less precise (a value is exact within 2^14 of its block's maximum instead of its column's) |
| `q5-ksplit-mma`, `fp16-gdn-state` (Qwen3.5) | the groupwise artifacts' Q5 projections; every GDN state pool | by the owner's decision (2026-10-25) ahead of their gates, where measured faster with INT8 KV (MTP 3 decode, 3 blocks): groupwise 27B `q5-ksplit-mma` +4.5 % and `fp16-gdn-state` +0.7 %, NVFP4 27B `fp16-gdn-state` +0.7 %; not the 35B-A3B (no gain beyond spread) |
| `q4-w4a8` | groupwise Q4 MLP gate/up | none (fails, below) |
| `idle-wide-prefill` | Qwen3.8-27B / Qwen3.6-35B-A3B generation engines | NVFP4 27B, groupwise 27B and 35B-A3B, every format |
| `adaptive-mtp` | Qwen3.8-27B / Qwen3.6-35B-A3B with MTP and `--draft-tokens auto` | NVFP4 27B and 35B-A3B, every format |
| `adaptive-dflash` | Qwen3.8-27B / Qwen3.6-35B-A3B with DFlash or DFlash2 and `--draft-tokens auto` | NVFP4 27B and 35B-A3B, every format (2026-10-22, below); not the groupwise 27B |
| `gdn-tree-closed-form` | tree rounds' GDN records of at most 16 columns (wider rows walk) | NVFP4 27B and 35B-A3B, every format, by the owner's decision (2026-10-25) ahead of its gate, where measured faster (ABBA, INT8 KV, greedy pp1280+tg256, fixed trees of 16, 6 blocks): NVFP4 27B DFlash2 decode +1.19 % (6 of 6, at least +1.01 %), 35B-A3B DFlash rounds 1.0-1.7 % shorter in 6 of 6 (decode +1.6 % where the greedy text matched); its kernel 0.65-0.93 of the walk's time at 1-4 rows (`gdn_replay_bench --component tree`) |
| `block-verification` | sampled chain rounds (DFlash2, sampled MTP) | none: re-timed after its keys were made round-unique (2026-10-25, ABBA, sampled, 6 blocks): Dense8 MTP no resolved difference (median +1.9 %, worst -14.9 %), Dense8 DFlash2 `auto` -14.2 % (6 of 6 worse), NVFP4 27B DFlash2 7 no resolved difference (median -0.1 %, worst -4.8 %) |
| Qwen3.8-Flash-Next: `expert-a4-residual` | artifacts with NVFP4 W4A16 experts (recipe C; NVIDIA FP4-Local at its default 16-bit expert activations), expert calls of more than eight columns | FP8 dense (recipe C) and MXFP8 dense (FP4-Local, by recipe C's kernel verdicts and its own timing), 16-bit expert activations, every KV format (BF16 by FP4-Local's gates against the precise reference, below) |
| Qwen3.8-Flash-Next: `expert-a4-residual-3` | the same calls as `expert-a4-residual`, three NVFP4 terms per x and h row (its more precise form, design §19.3.23; it takes precedence where both are selected) | none: two terms pass on every configuration and format they run in (2026-10-21), so it stays an unregistered candidate (`--fast-path expert-a4-residual-3`) for a cell where two terms fail; three terms cost 7-8 points of prefill speed (below) |
| Qwen3.8-Flash-Next: `sampled-mtp-drafts` | MTP rounds with a sampled request | every Flash-Next configuration, every format (distribution and greedy identity, below; no logit changes) |
| Qwen3.8-Flash-Next: `dflash2-pairs` | two-row rounds with the DFlash2 drafter (`--spec dflash2 --draft-model`) | every Flash-Next configuration, every format, by the owner's decision (2026-10-24) ahead of its gate (not run); timed on Dense8, two requests, INT8 KV: decode +28 % median, 4 of 4 blocks |
| Qwen3.8-Flash-Next: `screened-lm-head`, `f16-accumulate-prefill` | greedy decode and verification rounds of at most 16 columns through the LM head's 2-bit screen (artifacts with `--screen-heads`); FP8, MXFP8 and Q8 dense prefill calls of 256+ columns | NVIDIA FP4-Local (MXFP8 dense, 16-bit experts), INT8 KV, by the owner's decision (2026-10-25) ahead of their gates: `screened-lm-head` plain decode +7.5 %, MTP 3 +3.7 % (4 of 4 blocks); `f16-accumulate-prefill` pp32768 +5.3 % (3 of 3), pp4096 neutral; not Dense8 or recipe C (no gain beyond spread) |
| Qwen3.8-Flash-Next: `ggml-bf16-prefill` | GGUF dense projections in prefill calls of 32+ columns, GGUF expert calls of 14,336+ columns | IQ3_S, every format (BF16 KV since 2026-10-20, against the precise reference; below) |
| `gdn-f16-accumulate` | chunked GDN calls (16+ tokens) of every Qwen3.5-family and Flash-Next artifact | NVFP4 27B, groupwise 27B, FP8 and IQ3_S, every format (FP8 with BF16 KV by its prefill gate against the precise reference, 2026-10-21; IQ3_S with BF16 since 2026-10-20, below); not the 35B-A3B, Dense8 or FP4-Local (no gain beyond spread, below; recipe C and recipe A are timed in the 2026-10-20 queue) |

**`sampled-mtp-drafts`: drafts drawn from the draft head's top four tokens (Flash-Next, qualified
2026-10-17).** It changes no logit: a sampled request's drafts are drawn from the head's four most
probable tokens under its sampler and verified by sparse rejection sampling against that proposal,
and its rounds verify every draft (no `--spec-min-p` cut). The output distribution stays the
sampler's, but a seeded request's text changes, so it cannot be the exact route; greedy requests
keep the argmax drafts. Its gate is therefore the distribution, not the model-level arms
(`tests/models/qwen4_exp/test_speculative_sampling_real.cpp`, Dense8, INT8 KV, exact route otherwise):
2,000 seeded requests x 8 tokens per arm at T 1.0 / top-p 0.95 / top-k 20, two-sample chi-square
against plain decoding at each position and jointly for the first two tokens (family alpha 1e-3,
Bonferroni 1.1e-4): sampled drafts least p 0.21 (7,115 rounds, 5,717 of 12,545 drafts accepted),
one-hot drafts (positive control) 0.037, plain decoding at T 0.8 (power) 1.2e-29. Greedy output
equals the one-hot route on INT8 and NVFP4 KV, and no KV format enters the draft or its
proposal, so every format is registered. Speed (INT8 KV, MTP 4 with the proposal head, 1,536
tokens, cold expert cache, against the default route): AIME 153.5 -> 154.6 / 155.8 -> 165.4,
code 144.4 -> 150.8 / 147.6 -> 153.1, story 137.1 -> 139.9 / 141.0 -> 139.7 tok/s (seeds 7 / 43):
geomean +2.7 %, 5 of 6 cells faster, worst -0.9 % (each arm decodes its own text). The switch
prototype measured +9.1 % (6 of 6) on a warm cache; greedy MTP decode is unchanged (150.3 against
150.3 tok/s). The other configurations (2026-10-18; INT8 KV, `--draft-tokens auto` with the
proposal head, T 1.0, seed 7, 1,024 tokens, the exact route against the exact route plus the path,
ABBA): IQ3_S math 166.4 -> 197.6 (+18.8 %), story 130.4 -> 159.6 (+22.4 %), code 155.8 -> 162.2
tok/s (+4.1 %, overlapping); FP8 math 48.0 -> 57.8 (+20.4 %), story 41.7 -> 48.2 (+15.6 %), every
run ahead, code 46.1 -> 40.1 (-13.0 %, overlapping: one exact run at 52.5 against 39.7-40.1 for the
other three; FP8 decode is SSD-bound and its runs vary by up to 30 %). Tokens per round did not
rise with it (IQ3_S math 2.84 and 2.63 exact against 2.42 and 2.78 with the path), so the gain is
not explained by acceptance alone and was not attributed further. Recipe A, recipe C and NVIDIA
FP4-Local (MXFP8 dense, with 16- and with 4-bit expert activations) are registered by the rule for a
path that changes no logit (6.1), with these as the nearest timings.

**`gdn-f16-accumulate`: FP16-accumulate chunked GDN (qualified on the 27B, 2026-10-17).** See
`ops::GdnChunkArithmetic::Fp16Accumulate`. Op: 1.20x at the 27B's shapes, and closer to the FP64
oracle than the exact route (output relative RMS 2.08e-3 against 3.60e-3, state 8.6e-4 against
2.6e-3). Prefill (INT8 KV, default route against default + the path, ABBA): 27B pp4096 12,956 ->
12,986 and pp16384 11,965 -> 11,994 tok/s (+0.24 %, both runs of each arm ahead); Flash-Next
Dense8 pp16384 6,574 -> 6,564 (-0.15 %, ranges overlap), pp65536 11,186 -> 11,214 (+0.26 %), so it
is not registered on Dense8. Exact route against exact route plus the path, pp16384, INT8 KV
(2026-10-18): IQ3_S 4,227.5 -> 4,236.4 (+0.21 %), FP8 2,529.8 -> 2,538.2 (+0.33 %, the lower fast run
0.3 tok/s above the higher exact one), groupwise 27B +0.12 %, every run ahead in each; 35B-A3B
+0.05 %, within spread, not registered. IQ3_S and FP8 take the NVFP4 27B's gates below (one gate
per kernel, 6.1 item 1), the groupwise 27B its own prefill gate (under `idle-wide-prefill`); the
path touches no attention or KV, so every format is registered with the INT8 timing. **IQ3_S with
BF16 KV was removed (2026-10-19):** gated there alone (under `ggml-bf16-prefill`, below) it fails the
prefill gate (KL bound 7.5e-3 of 1.5e-3; the IQ3_S exact route with BF16 KV is the reference, floor 0)
and the decode gate (TV bound 4.6e-3 of 4e-3); the 27B's BF16 floor is not 0, and there it passed.
`ggml-bf16-prefill` fails the same cell by the same amount: at floor 0 any change of IQ3_S's
arithmetic does. Against the precise reference (2026-10-20) the prefill cell passes (under
`ggml-bf16-prefill`); the decode cell (TV floor 0.029 against the exact kernels) is to be gated
against it. FP8 with BF16 KV failed the same decode gate against the exact kernels (2026-10-19, plain,
256 segments; floor KL 7.4e-3, TV 0.032; TV excess bound 4.9e-3 of 4e-3, top-1 bound 0.85 points of
0.5). Against the precise reference its prefill gate passes (2026-10-21, quick corpus, chunks 896
and 7,168, floor KL 6.7e-3): KL excess -1.6e-4 (bound 1.6e-4), top-1 +0.01 points (bound 0.05), NLL
3.1e-4, p99.9 KL ratio 0.97, both context buckets within budget; a path acting only in chunked GDN
calls is decided by its prefill gate (as IQ3_S's), so it stays registered.
Gates on the official 27B NVFP4, jointly with the qualified paths (the path
touches no attention or KV: reduced prefill gate on every format, decode on INT8 and on BF16, the
lowest floor):

| 27B, KV | Prefill KL excess (bound) / 1.5e-3 | Prefill top-1 excess, points (bound) | Corpus NLL excess / 3e-3 |
|---|---|---|---|
| BF16 | -5.0e-4 (3.8e-5) | +0.01 (0.07) | 9.1e-4 |
| INT8 | 7.6e-5 (3.8e-4) | +0.05 (0.10) | -3.9e-4 |
| FP8 | -1.6e-4 (2.9e-4) | +0.08 (0.14) | -3.6e-4 |
| NVFP4 | 5.8e-4 (1.0e-3) | +0.04 (0.11) | -9.5e-4 |
| K8V4 | 3.8e-4 (9.0e-4) | +0.05 (0.09) | 5.9e-4 |
| VQ2 | 9.1e-5 (3.0e-4) | +0.00 (0.05) | -6.9e-4 |
| K4V2 | -1.1e-4 (3.9e-4) | -0.02 (0.04) | 1.6e-3 |

Decode, plain, 256 segments: INT8 TV excess 2.6e-4 (bound 5.2e-4) / 4e-3, top-1 +0.05 points
(0.23) / 0.5, KL 6.3e-5; BF16 TV -2.8e-4 (2.6e-5), top-1 -0.12 points (0.07), KL -1.4e-4. Every
tail and context criterion passes. These are joint excesses over the exact route, so they do not
isolate the path's own effect; BF16 KV, where the qualified paths add least, shows every estimate
below zero.

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
- `expert-a4-residual` (Qwen3.8-Flash-Next with W4A16 experts): expert calls of more than eight
  columns (prefill, wide verification) take each x and h row as two terms of the same A4 residual
  profile, with an FP32 row scale of amax / (6 · 448) in place of an input scale (the export has
  none), on the FP4 Tensor Cores (design §19.3.23). Without it W4A16 experts run the exact integer
  limb route. An expert's route depends on its column count in the call, which other requests'
  columns share in batched rounds.
- `single-column-decode` (Qwen3.5, without `batched-decode`): decode and verify calls of one column
  (a request's plain decode step alone in its round) run `batched-decode`'s single-column routes and
  down-projection tiles; every wider round keeps the column-invariant route, and the drafters keep
  their column-invariant copies. A step's route depends on whether another request shares its round.
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
Its kernel path `q8-sliced-k` (below) couples requests too: its routes depend on the round's width;
so does `ggml-bf16-prefill`, whose routes depend on a prompt's call boundaries and MoE grouping.

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

**Flash-Next on the other KV formats** (2026-10-16, RTX 5090, CUDA 13.4, NVIDIA NVFP4 Dense8, 256
segments, the single budgets of 2026-10-16). Two arms per format, each fast run with the format's qualified
kernel paths. **Joint**: MTP with 4 drafts at concurrency 2, every segment resuming at 3,000 tokens
(its whole prompt for shorter ones) from a warm request (`adaptive-speculation`,
`free-cache-resume`, `preemption`; 255,000 prompt tokens reused per run), against one plain exact
run at concurrency 8, the format's default for every arm (the exact route is request-invariant).
**Preemption increment**: the forced-preemption arm above, with the qualified paths alone in the
default role (6.1). Bounds are one-sided 95 %:

| KV | Arm | TV excess (bound) / 4.0e-3 | Top-1 excess, points (bound) / 0.50 | KL excess / 1.0e-2 | NLL excess / 1.0e-2 | p99.9 KL ratio / 1.25 | Verdict |
|---|---|---|---|---|---|---|---|
| BF16 | joint | -2.3e-4 (6.1e-4) | -0.17 (0.18) | -1.2e-4 | -2.2e-3 | 0.97 | pass |
| BF16 | preemption increment | -2.9e-4 (1.9e-4) | -0.16 (0.07) | -8.0e-4 | 2.0e-3 | 0.93 | pass |
| FP8 | joint | -6.9e-5 (7.2e-4) | +0.45 (**0.83**) | -1.4e-4 | -4.3e-3 | 1.08 | **fails** |
| FP8 | preemption increment | -3.7e-4 (1.9e-4) | -0.12 (0.13) | -8.1e-4 | -1.6e-3 | 0.95 | pass |
| NVFP4 | joint | 6.4e-4 (1.5e-3) | +0.01 (0.39) | 6.9e-4 | -1.2e-3 | 1.03 | pass |
| NVFP4 | preemption increment | 1.9e-4 (7.7e-4) | +0.04 (0.28) | -5.6e-4 | -4.9e-4 | 0.96 | pass |
| K8V4 | joint | 3.8e-4 (1.2e-3) | +0.09 (**0.51**) | 1.1e-3 | 1.7e-4 | **1.33** | **fails** |
| K8V4 | preemption increment | 4.3e-4 (9.4e-4) | -0.08 (0.16) | 1.7e-5 | 3.0e-3 | 0.98 | pass |
| VQ2 | joint | 1.4e-4 (8.3e-4) | -0.02 (0.36) | 5.3e-4 | 7.0e-4 | 1.23 | pass |
| VQ2 | preemption increment | -3.7e-5 (4.8e-4) | -0.01 (0.22) | 1.2e-3 | 9.9e-5 | 1.07 | pass |
| K4V2 | joint | -1.8e-4 (5.7e-4) | +0.10 (0.46) | 5.0e-5 | -2.7e-3 | 1.02 | pass |
| K4V2 | preemption increment | 3.6e-4 (8.2e-4) | -0.08 (0.13) | 1.1e-3 | 4.4e-4 | 1.03 | pass |

Every prompt-length bucket passes. At 256 segments `adaptive-speculation` and `free-cache-resume`
pass on BF16, NVFP4, VQ2 and K4V2 and fail narrowly on FP8 and K8V4; `preemption` (which acts only
in its own arm) passes on all six. The joint fast run also holds the format's kernel paths, whose
decode was gated on BF16 and NVFP4 only. Attribution
(2026-10-16, the joint arm's conditions, 256 segments; top-1 in points, bound in parentheses):

| KV | Run | TV excess (bound) | Top-1 excess (bound) / 0.50 | p99.9 KL ratio / 1.25 | Verdict |
|---|---|---|---|---|---|
| FP8 | kernel paths alone vs exact | -4.5e-5 (7.7e-4) | +0.26 (**0.69**) | 1.08 | fails |
| FP8 | `qsa-prepare-fused` alone vs exact | -2.7e-4 (4.9e-4) | +0.12 (**0.56**) | 0.94 | fails |
| FP8 | coupling paths over the kernel paths | -2.4e-5 (7.8e-4) | +0.18 (**0.55**) | 1.00 | fails |
| K8V4 | kernel paths alone vs exact | -1.3e-4 (6.9e-4) | -0.09 (0.32) | **1.36** | fails |
| K8V4 | `qsa-prepare-fused` alone vs exact | 8.9e-4 (1.8e-3) | +0.34 (**0.75**) | **1.36** | fails |
| K8V4 | coupling paths over the kernel paths | 5.1e-4 (1.2e-3) | +0.18 (**0.54**) | 0.98 | fails |

KL and NLL pass everywhere. The kernel paths, registered on FP8 and K8V4 from prefill and from
decode on BF16 and NVFP4 only, fail their first decode measurement on these formats, through
`qsa-prepare-fused`, which has its own kernel per KV format; the coupling paths' increment fails
top-1 narrowly on both. At 256 segments the top-1 bound sits 0.35-0.44 points above its estimate on
every format, so these are small positive estimates (+0.12 to +0.34) that the sample cannot clear.
`qsa-prepare-fused`'s own decode gate at 1,024 segments (plain at concurrency 8, the shared INT8
coupling reference) settles it: it passes on FP8 (TV bound -2.5e-5, top-1 bound -0.002 points, KL
excess -5.3e-3, NLL -1.8e-2, p99.9 ratio 0.94) and K8V4 (TV bound 6.2e-5, top-1 bound 0.14 points,
KL -2.7e-3, NLL -9.4e-3, p99.9 ratio 1.10), so its registrations stand. The exact route gives the
same bits under MTP at concurrency 2 as plain at concurrency 8 on FP8 and K8V4 too (96 segments).
The joint arm at 1,024 segments (2026-10-17, against those exact runs; 1,023,000 prompt tokens
reused per run) passes on both formats, every excess estimate negative: FP8 TV bound 8.9e-4, top-1
bound 0.075 points, p99.9 ratio 1.04; K8V4 TV bound -3.6e-4, top-1 bound 0.10 points, p99.9 ratio
1.05. `adaptive-speculation` and `free-cache-resume` are qualified on FP8 and K8V4 by this
verdict, which replaces the 256-segment one. Before the increment rule, the BF16 forced-preemption arm against the exact route failed
(TV bound 4.57e-3, KL excess 1.23e-2, p99.9 ratio 1.63), and the kernel paths alone failed it the same
way; its exact route was bit-identical with and without the short pool and across reruns (6.1).

**Flash-Next IQ3_S, FP8, recipe A and recipe C** (2026-10-17/18, the kernel rule of 6.1 item 2).
On IQ3_S (GSQ-RCO IQ3_S) and FP8 (the official FP8 import) the prefix-cache and preemption real
tests pass on the exact route with the candidates (bit-level resume and Replay equivalences), and
one INT8 joint decode gate each (MTP 4 at concurrency 2, every segment resuming from a warm
3,000-token entry, 256 segments, `adaptive-speculation` + `free-cache-resume` + `preemption`)
passes; the other formats take Dense8's per-format verdicts above:

| Artifact | TV excess (bound) / 4.0e-3 | Top-1 excess, points (bound) / 0.50 | KL excess / 1.0e-2 | NLL excess / 1.0e-2 | p99.9 KL ratio / 1.25 |
|---|---|---|---|---|---|
| IQ3_S | 4.4e-4 (6.4e-4) | -0.03 (0.09) | 2.4e-4 | -9.6e-5 | 1.12 |
| FP8 | 3.7e-4 (5.6e-4) | +0.13 (0.26) | 1.2e-4 | 7.6e-4 | 1.02 |

Speed, each path alone against the exact route (INT8 KV, `infernix-serve`, MTP `auto`, ABBA):
- `free-cache-resume` (three conversations, time to each second turn's first token): IQ3_S
  1.65-1.67 and 0.66 s -> 0.10-0.12 s for the second and third (the first, 1.66 s, is cold in both
  arms), FP8 3.8-5.7 s -> 0.26-0.51 s.
- `adaptive-speculation` (two concurrent requests): IQ3_S 160 -> 200 tok/s (+25 %), every run
  ahead; FP8, whose SSD-bound decode varies run to run, over three ABBA blocks of 2 + 2 runs:
  +2.6, +3.8 and +7.3 % (37.4 -> 39.2 tok/s overall), every block ahead.
- `preemption`: with four requests that run to their 512-token limit under a 1,280-token pool it
  costs IQ3_S 3.0 % (158.4 -> 153.5 tok/s, every run behind; Replay redoes work) and FP8 is mixed.
  Where it acts in practice, eight brief requests asking for 4,096 tokens each under an
  8,192-token pool (the exact route holds whole extents and admits about two at once): IQ3_S 118.1
  -> 131.3 tok/s (+11.3 %), FP8 26.7 -> 28.3 (+5.9 %), every run ahead. It is registered for that
  regime, which agent clients produce (`max_tokens` 64,000 on short turns).

Recipe A (BF16 dense, 4-bit experts) is registered for the three coupling paths and sampled drafts
without runs of its own, at the owner's request (2026-10-18): `free-cache-resume` takes FP8's timing
(the same dense kernels), `adaptive-speculation` Dense8's and IQ3_S's (4-bit experts streamed as
Dense8's), `preemption` IQ3_S's and FP8's, and the quality verdicts Dense8's. Its exact-route
equivalences were not run on the artifact. Recipe C first failed (2026-10-18): its real tests
diverged on the exact route at token 29 (a resumed request against a cold one, and concurrent
against solo), so its exact route was not request-invariant, and its joint gate failed the p99.9 KL
ratio (1.33) against the 16-bit reference. A bisect (2026-10-20, plain decode at
one request, INT8 KV, two identical runs of 48 forced segments per arm) found its plain decode
nondeterministic: 5-9 segments differed while the shared expert ran beside a forked expert call's
staged pass, and none when the shared expert waited for that pass, ran before the routed experts or
ran with no fork, the three bit-identical to each other. Landing, promotions, the CPU-served misses
and L1-cached job records were not the cause (each switched off left 8-33 differing). The shared
expert first waited for the staged pass (which hid the race). The cause (found 2026-10-20 with the
FP4-Local session and tile-ai/tilelang #3529): the shared expert's FP8 A16 GEMM refilled its
shared-memory stages with TMA, and its consumer warps released a stage after `ldmatrix` and
`ld.shared` reads without `fence.proxy.async`, so with the staged pass's kernels on the same SMs a
refill could overwrite a stage still being read; `infernix_fp8_moe_overlap_test` reproduces it
(recipe C's FP8 [1280,2560] and [2560,640] beside staged W4A16 and W4A4 calls: the old kernel
fails, so does the TMA producer without the fence, the fenced one and a cp.async producer pass),
and every other TMA-fed route (BF16 A16, FP8 A8, FP8 A8 pair, NVFP4 A4) now fences: in the same
harness the unfenced FP8 A8 route differed in 7-147 of 200 overlapped calls and the fenced one in
none (`infernix_tma_moe_overlap_test`), while the unfenced BF16 A16, pair and NVFP4 A4 routes never
differed there. The producer now stages with
cp.async (the consumer and every output bit unchanged), and the wait is gone. Without it (48 forced
segments, INT8 KV) recipe C is bit-identical between two runs at one request and at eight, at one,
four and eight requests, MTP at one and four, from a warm 3,072-token prefix at a 1,024-token
chunk, and with the waiting build; the unfenced TMA producer differs in 6 of 48 segments between two
identical runs, the fenced one in none. Before that, with the wait, the exact-route audit (96 forced segments, INT8 KV) finds no
segment differing between two identical runs at one request, at four requests, under a
16,384-token KV pool at four requests, or between MTP at one and four requests. At eight requests
(48 segments, INT8 KV, the same ordering), two identical runs agree with each other and with one
request on recipe C, IQ3_S and the FP8 import; on recipe C and IQ3_S so do runs with the shared expert
on the main stream or before the routed experts. (`infernix_decode_quality_forced` now waits up to an
hour for admission: the FP8 import's prefills read about 100 GiB from the SSD expert tier, and its
eight-request runs had expired at the default 30 s.) The first warm-prefix arm proved nothing: the
exact route resumes only at prefill-chunk boundaries, and with the tool's 4,096-token chunk no
3,000- or 3,072-token prefix, nor any 512- or 2,048-token prompt, held one (`reused_prompt_tokens`
0). At a 1,024-token chunk for both runs, a warm 3,072-token prefix is reused (49,152 tokens over the
16 long segments) and no segment differs from the cold run, which itself equals the 4,096-token-chunk
run. With the fence (2026-10-21) its prefix-cache and preemption real tests pass with the candidates,
and the INT8 joint decode gate (MTP 4 at concurrency 2, every segment resuming from a warm
3,000-token entry, 255,000 prompt tokens reused per run, 256 segments, precise reference, floor TV
0.037) passes: TV excess 8.4e-4 (bound 1.2e-3), top-1 +0.07 points (bound 0.31), KL 3.4e-4, NLL
3.8e-4, p99.9 KL ratio 0.81, every prompt bucket within its budget. No fast run preempted, so
`preemption` rests on the preemption real test's Replay equivalence and the kernel rule (6.1, item 2:
Replay runs the exact route's own prefill kernels), as on IQ3_S and FP8; no forced-preemption
increment arm was run on recipe C. The three coupling paths are registered on every format, the other
formats taking Dense8's verdicts. Speed, `adaptive-speculation` alone against the exact route (two
concurrent requests, INT8 KV, MTP `auto`, 512 tokens each, `infernix-serve`, ABBA): 114.3 / 114.1 ->
118.6 / 118.6 tok/s (+3.8 %), every run ahead; `free-cache-resume` takes FP8's timing and
`preemption` IQ3_S's and FP8's (6.1, item 4).

NVIDIA FP4-Local (MXFP8 dense, 2026-10-20/21): after the TMA proxy-fence fix (design §6.1.2) its exact
route is deterministic and request-invariant alone and at eight concurrent requests (BF16 KV, 48
segments), and the prefix-cache and preemption real tests pass with the candidates at both expert
activation precisions. One INT8 joint decode gate per expert activation precision, as IQ3_S's and
FP8's (MTP 4 at concurrency 2, every segment resuming from a warm 3,072-token entry, a 9,728-token
pool, 256 segments; fast role the coupling paths with the configuration's qualified paths,
`expert-a4-residual` at 16 bits):

| Expert activations | Reference | TV excess (bound) / 4.0e-3 | Top-1 excess, points (bound) / 0.50 | KL excess / 1.0e-2 | NLL excess / 1.0e-2 | p99.9 KL ratio / 1.25 | Verdict |
|---|---|---|---|---|---|---|---|
| 4-bit | 16-bit | 6.0e-6 (2.0e-5) | +0.01 (0.03) | -6.1e-6 | -1.3e-4 | 1.00 | pass |
| 16-bit | 16-bit | 1.2e-3 (1.6e-3) | +0.13 (0.35) | 4.4e-4 | -9.0e-4 | **1.27** | fails |
| 16-bit | precise | 4.8e-4 (8.3e-4) | 0.00 (0.22) | 2.2e-4 | -9.0e-4 | 1.16 | pass |

The 4-bit pass, against the 16-bit reference with a nonzero floor (TV 0.070), stands. Against that
reference the 16-bit cell failed only its tail ratio, not one outlier: each route's 24 largest KLs
lay in 23 segments, the ratio was 1.31 and 1.19 on alternate halves of the segments, and it came from
the 512-token prompts (1.47; 2,048: 1.12; 8,192: 0.97). Against the precise reference (floor TV
0.033, KL 7.0e-3) it passes, so the 1,024-segment run approved for it was not needed. No fast run
preempted (the pool left about 800 tokens beside an 8,320-token lane), so `preemption` is decided by
its forced-preemption increment (6.1; INT8 KV, precise reference, 2026-10-21), which passes at both
precisions: each fast run paused 96 times and replayed 792,480 tokens, the default runs never paused;
16-bit TV excess 4.8e-5 (bound 1.5e-4), top-1 +0.02 points (0.07), KL 9.7e-5, NLL -3.2e-4, p99.9 ratio
1.00 (floor TV 0.038); 4-bit TV -6.8e-6 (1.7e-4), top-1 -0.04 points (0.04), KL -1.8e-4, NLL -5.1e-5,
p99.9 ratio 1.00 (floor TV 0.074). The three coupling paths are registered at both precisions on every
format, the other formats taking Dense8's verdicts (6.1, item 2). Speed,
`adaptive-speculation` alone against the exact route (two concurrent requests, INT8 KV, MTP `auto`,
256 tokens, ABBA): 16-bit 84.0 / 84.1 -> 125.7 / 126.1 tok/s (+49.8 %), 4-bit 125.6 / 125.3 -> 162.1 /
162.2 (+29.3 %), every run ahead; `free-cache-resume` takes FP8's timing and `preemption` IQ3_S's and
FP8's (6.1, item 4: the same mechanisms, prefill not repeated and requests admitted under a short
pool).

Every prompt-length bucket and p99.9 KL ratio passes too (largest ratio 1.13, 27B K8V4 spec). The
closest bound is 27B NVFP4 spec's top-1 at 82 % of its budget. Speed (official 27B, INT8 KV, 2,048
prompt, 256 output tokens, three rounds; the qualified set against the exact route): plain decode
81.83 vs 73.17 tok/s at one request (+11.8 %), +11.0 % at two, +6.9 % at four and +0.3 % at eight
(the 4- and 8-bit activations the old route used there are not qualified); DFlash2 with n-gram
copies 853 vs 523 tok/s at two requests (+63 %) and 583 vs 329 at four (+77 %). VQ2, BF16 and
FP8 KV were qualified later (below).

**The groupwise 27B's and the 35B-A3B's request-coupling paths** (2026-10-22, RTX 5090, CUDA 13; against
the lossless decode references, 256 segments). Once their reproducible routes became request-invariant,
each coupling path was timed alone against the default route (ABBA, serve scenarios; a path is gated when
faster in both pairs somewhere and slower in both nowhere): `free-cache-resume` second-turn time to first
token -95.6 % (groupwise 27B) and -58.8 % (35B), `preemption` aggregate throughput under a short pool
+203 % and +109 %, `adaptive-speculation` +1.18 % (groupwise, MTP; DFlash2 +0.52 %, mixed) and +7.96 %
(35B, MTP). `batched-decode` and `batch-split-attention` were mixed or behind (groupwise plain decode
+12.8 % at one request but -7.1 % prefill and -19.1 % decode at four requests), so they are not gated.
The three paths were gated jointly (concurrency 4, prefix reuse with 3,000 warm tokens, a 16,640-token
pool so the fast run preempts, groupwise DFlash2 with n-gram copies, 35B MTP; the reproducible route as the
default arm): INT8 KV TV excess 1.07e-4 (groupwise, p99.9 ratio 1.15) and 8.6e-5 (35B). On the other six
formats both pass (`.local/gpu-queue/stage2a`): the largest TV excess bound 5.7e-4 (35B NVFP4; groupwise at
most 1.4e-4), the largest top-1 bound 0.24 points (35B BF16; groupwise 0.22, VQ2), p99.9 ratios 0.87-1.12;
every fast run preempted (21-28 times) and resumed 192,000 prompt tokens from the prefix cache. The three
paths are registered for both artifacts on every format.

**`single-column-decode`** (2026-10-22, RTX 5090, CUDA 13): `batched-decode`'s routes in one-column rounds
only, since `batched-decode` itself was +12.8 % at one request but -19.1 % at four on the groupwise 27B.
`batched-decode`'s apparent -7.1 % prefill was no prefill work: nsys traces show the same 747 prefill
kernels in both routes, with the three compute-bound weight GEMMs uniformly slower after the faster decode.
Timed alone against the default route (ABBA, INT8 KV, pp512 + tg512, three repetitions): groupwise 27B
decode +14.5 % at one request and +0.05 % at four, end to end +13.9 % and +0.06 % (ahead in both pairs).
Its prefill (-6.4 % at one request, -0.4 % at four) follows the board's power limit (517.5 W): sampled
clocks put the faster decode at 513-515 W with the SM clock power-capped near 2,470 MHz (the default route
decodes at ~480 W near 2,880 MHz), and the next prefill starts at that clock. 35B-A3B: +0.55 % at one
request but -1.0 % at four (both pairs, unexplained; its decode never reached the power cap), so it is not
registered there. Gated on the groupwise 27B per KV format on plain one-request decode (every round one
column, the path's whole effect), 256 segments, against the lossless decode reference, with the
reproducible route as the base arm (INT8 the stored plain arm; the other six new plain runs, each
bit-identical to step 2a's n-gram joint base runs): all seven pass, the largest TV excess bound 1.65e-4
(NVFP4), the largest top-1 bound 0.16 points (VQ2), p99.9 ratios 0.84-1.13 (K4V2 the highest). Registered
for the groupwise 27B on every format.

**`batched-a4-residual-decode`** (2026-10-14, the same artifact, corpus and arms): two NVFP4 terms
per column, from 9 columns in the MLP. Its op error against the 16-bit oracle is 0.9 % relative L2
on the [5120, 17408] down projection (one-term A4 6-12 %, A16 0.16 %); the gate/up suite's tiny
activations put its second term's scales near E4M3's subnormal range, so its error there (1.6-2.2 %)
does not fall with a third term. The plain arm never reaches 9 columns, so it also re-qualifies
`batched-decode` after its [5120, 17408] down tile changed (8 K-warps at 5-16 columns, 35.6-51.8
against 41.6-60.8 us). That tile is `batched-decode`'s own route (`ops::LinearRoutes::WideDownTile`,
decode and verification rounds only): until 2026-10-18 it also served exact-route prefill calls of
5-16 columns, an unqualified exact-route numerics change since `1fe367b4`, and those now run the
release's 4 K-warp tiles again. Every arm passes, jointly with the paths above:

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
  - teacher-forced plain decode (KL gate): KL excess −5.6e-4, bound 9.6e-5, against a 1.67e-3
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
- **Groupwise 27B** (24 heads, 16-bit activations, 2026-10-18): the same kernel at the same shapes,
  so it takes this gate (6.1 item 1); timed on the artifact, +0.25 % at 512-token prompts with every
  run ahead and level at 4,096, so it is registered on INT8 KV.

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
  full prefill protocol: 144 windows, 4.18 million positions per route, chunks of 896,
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
  NLL excess 2.4e-3 with bound 1.19e-2 against a 1.55e-3 budget. Under the KL decode gate,
  first protocol: KL excess −1.19e-3 with bound 3.12e-3 against 1.67e-3, and top-1 excess 0.22
  points with bound 0.42 against 0.196. Under the current protocol (1,024 segments of the decode
  corpus): KL excess −4.6e-4 with bound 3.27e-3 against 3e-3 (109 %). Under the total-variation
  decode criterion that replaced KL, on the same runs, every criterion passes: TV excess
  −1.8e-4 with bound 3.0e-4 against 2.43e-3, top-1 excess −0.09 points with bound 0.017. The
  criterion changed after this pair had failed; the A/A and NVFP4-KV controls validate it
  on two protocols. prompt-pv8 was qualified for INT8 KV on 2026-10-14 on that evidence, by the
  user's decision, which also weighed how far inside other engines' acceptance criteria the pair
  sits: prefill KL excess 1.4 % of the floor (bound 2.8 %), decode TV bound 12 % of its budget.
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
- **Groupwise 27B** (24 heads, 16-bit activations): the same kernels at the same shapes, so it
  takes these gates and timings on the same five formats (6.1 items 1 and 4); its own prefill gate
  with the path passes (under `idle-wide-prefill`).
- **35B-A3B** (16 heads, 16-bit activations; 2026-10-18): its own prefill gate with the path passes
  (under `idle-wide-prefill`), and its 16-head shapes are its own, so each format was timed
  (pp16384, ABBA, every run ahead): INT8 +0.24 %, NVFP4 +1.51 %, K8V4 +1.59 %, VQ2 +0.15 %, K4V2
  +0.27 %.

**`adaptive-mtp`: MTP draft length chosen per round (qualified 2026-10-18 on the NVFP4 27B and the
35B-A3B).**
- **Selected by:** `--spec mtp --draft-tokens auto` on Qwen3.8-27B / Qwen3.6-35B-A3B, where it is
  registered (it is `auto`'s policy: a fixed `--draft-tokens N` refuses it); it acts where
  speculation is adaptive (`adaptive-speculation` selected, or an artifact that cannot be
  request-invariant). Where it does not act (another configuration, or `--reproducible`),
  `auto` drafts the configuration's fastest single length every round: 5 on the NVFP4 27B, 3 on the
  groupwise 27B, the 35B-A3B and unmeasured configurations (`auto_fixed_draft_tokens`). The policy is
  `models/qwen3_5/program/speculative/mtp_draft_policy.h`.
- **What changes:** the ceiling N is 5. Each all-neural round of one row runs a
  draft length K in 1..N, and a round of several rows one common K among 1, 2, 3 and N, whichever
  maximizes the rows' expected committed tokens per unit of round time. Expected tokens come from
  each request's decayed per-position acceptance (carried through preemption and Replay with the
  request). A round's time is learned from the rounds themselves: a decayed (0.98 per cycle) fit of
  start-to-start cycle time against K per batch size, used once it has 24 cycles and some spread of
  K, with cycles more than three times the fit's prediction (host gaps) left out; before that a
  round of K drafts costs 1 + 0.2 K plain rounds. The learned cost depends on measured times, so the
  choices no longer repeat exactly run to run (two identical exact-route runs on the 27B: 152.0 and
  139.4 tok/s). Every 16 choices the next longer K runs for two rounds so its acceptance stays
  measured. A choice reads only earlier rounds: drafts, acceptance and sampling are unchanged; only
  the verification width varies by round, and every width is one a fixed `--draft-tokens K` already
  runs through routes qualified at that width, so the path changes no kernel arithmetic and is
  decided by speed (6.1 item 3). Copy (n-gram) rounds keep their widths and propose at N.
- **Cost:** one Forward/Finish graph pair per extra (K, batch size): K < N for one row, K in 1..3
  for every batch size (about +4 MiB of graph allowance each).
- **Speed** (2026-10-18, RTX 5090, INT8 KV, `infernix_bench` tg512 on the bench corpus at
  concurrency 1, 2 and 4 on the default route, two runs per setting; `infernix` sampled text, math,
  story and code at seeds 7 and 43, 1,024 tokens; tok/s):

  | Artifact | Setting | Fixed 3 | Fixed 5 | `auto` |
  |---|---|---|---|---|
  | NVFP4 27B | concurrency 1 | 142.5 | 139.0 | **147.1** |
  | NVFP4 27B | concurrency 2 | **137.5** | 133.3 | 134.8 |
  | NVFP4 27B | concurrency 4 | 136.3 | 141.2 | **142.8** |
  | NVFP4 27B | sampled text, six runs summed | 875.5 | 895.5 | **899.3** |
  | 35B-A3B | concurrency 1 | 776.9 | 794.3 | **803.1** |
  | 35B-A3B | concurrency 2 | **575.0** | 504.1 | 564.3 |
  | 35B-A3B | concurrency 4 | 348.6 | 302.9 | **368.3** |
  | 35B-A3B | sampled text, six runs summed | 3,781 | 3,494 | **3,900** |
  | 35B-A3B | exact route, tg512 (fixed 4: 756.4) | 775.9 | 794.2 | **825.3** |
  | 35B-A3B | exact route, pp2048+tg512 (fixed 4: **800.9**) | 746.9 | 730.0 | 713.8 |

  Against the best single fixed length overall (the 27B's 5, the 35B's 3), the geometric mean over
  these settings is +2.1 % and +2.0 %. It is not the fastest everywhere: at concurrency 2 it is
  1.9-2.0 % behind fixed 3 on both, and on the 35B's predictable pp2048+tg512 text it chose short
  drafts too often ([0, 144, 138, 24, 171] rounds at K = 1..5) and trails fixed 4 by 10.9 %. The
  27B's exact-route sweep ran `auto` with `adaptive-speculation` (needed there for the policy to
  act) against fixed lengths without it, so it is not used. The other KV formats take the INT8
  timing: a format changes only the round cost the policy learns. The groupwise 27B, timed the same
  way (2026-10-18), is not registered: against fixed 3, its best single length overall, `auto` is
  +2.7 % at one request (146.9 against 143.0, from runs of 163.8 and 130.0), -4.7 % at two, -2.0 %
  at four and +3.6 % on sampled text (762 against 736 summed), a geometric mean of -0.15 %. The two
  one-request runs show the learned cost can settle on a worse length in one run than another (also
  the 27B's 152.0 and 139.4 above). Re-timed 2026-10-25 (`infernix_decode_quality_gen`, 24 slices of
  1,280 tokens of Infernix's code continued greedily and of its docs sampled, 384 tokens each, one,
  two and four requests, A B B A, decode tok/s): against fixed 3 `auto` is +10.3 / +4.4 / -4.8 % on
  code and -1.5 / +0.2 / +0.6 % on docs (geometric mean +1.4 %); with its batched rounds at a fixed 3
  they equal fixed 3 (within 0.2 % at two and four requests on the bench corpus), leaving code's
  +10.3 % against docs' -1.5 % at one request. Not
  a consistent gain, so the groupwise 27B keeps fixed 3. (Three bench-corpus prompts had shown
  +4 to +25 %: two of them continue near-deterministically, where longer drafts always pay.) Earlier (2026-10-15, the fixed cost model, bench corpus at 5.9 of
  6 accepted): 1.0-4.1 % slower than five fixed drafts, from its probe rounds.

**`adaptive-dflash`: the block drafters' verify width chosen per round from the proposal (qualified
2026-10-22 on the NVFP4 27B and the 35B-A3B).**
- **Selected by:** `--spec dflash|dflash2 --draft-tokens auto` on Qwen3.8-27B / Qwen3.6-35B-A3B, where
  it is registered (it is `auto`'s policy: a fixed `--draft-tokens N` refuses it); it acts where
  speculation is adaptive (`adaptive-speculation` selected). Where it does not act (the groupwise
  27B, `--reproducible`), `auto` verifies the configuration's fastest single fixed length every
  round: 7 on the 27B, 3 on the 35B-A3B (`auto_fixed_draft_tokens`). The policy is
  `models/qwen3_5/program/speculative/dflash_width_policy.h`
  ([DFlash](dflash.md#automatic-verify-width-qwen38-27b--qwen36-35b-a3b)).
- **What changes:** an all-neural chain round of one row proposes the whole block (15 drafts on the
  27B, 7 on the 35B-A3B) with each draft's probability under the drafter, then verifies the width
  with the most expected tokens per round time among 2-8, 10, 12, 14 and 16 columns (up to the
  block's), its leading min(W - 1, limit) drafts. Expected tokens come from the request's
  calibration of its drafts' probability deciles (Qwen3.8-Flash-Next's policy,
  `runtime/speculative/draft_length.h`), a round's time from the configuration's measured costs in
  plain rounds: NVFP4 27B 1 + 0.18 + 0.02 per draft, 35B-A3B 1 + 0.2 + 0.15 per draft per row
  (the groupwise 27B's 0.69 and 0.003 per draft to 7, 0.05 past them, serve a forced
  `--fast-path adaptive-dflash`). Every eighth choice of a row runs the next wider width. Rounds of
  several rows: the NVFP4 27B runs them as plain 7-draft rounds (one graph, `dflash_batch_drafts`),
  the 35B-A3B chooses among 2, 4 and 8 columns with each draft costed once per row. A choice reads the
  round's own proposal, never the target's draws, so acceptance and sampling are unchanged; every
  width is one a fixed `--draft-tokens K` verifies, through routes qualified at that width, so the
  path changes no kernel arithmetic and is decided by speed (6.1 item 3). The choices depend only on
  the proposals and the request's own rounds, not on measured time, so a workload repeats run to
  run. Automatic DFlash2 trees resolve to chains beside it (below).
- **Cost:** a one-row round's proposal and verification are two graphs with a host wait between;
  one proposal graph per drafter profile for the batch sizes that split, one verify/finish pair per
  width and batch size it serves (about 2.5 MiB of graph allowance each).
- **Correctness:** greedy streams against the target's own greedy choice at every position
  (`infernix_qwen3_5_dflash{2,}_block*_greedy_real_test`, six prompts of 160 tokens): NVFP4 27B
  one request 17 of 960 positions disagree (worst margin 0.50 nats; fixed chains 3.0-3.3 % on this
  test), two streams with `--ngram-draft-tokens auto` 20 of 960 (worst 0.50), six copy-heavy
  rewrite prompts at two streams 0 of 960 (27 copy rounds); 35B-A3B one request 55 of 960 against 48
  for fixed 7-draft chains (2026-10-21), two streams with plain 3-draft batches 54 of 960 (worst
  1.63).
- **Speed** (RTX 5090, INT8 KV, main 0e91f862, `infernix_decode_quality_gen` with N workers keeping
  N requests decoding: 1,280-token slices of Infernix's code continued greedily and of its Markdown
  docs sampled at temperature 1 with top-k 20 and top-p 0.95, 384 tokens each; 24 slices on the
  27B, 48 on the 35B-A3B; wall-clock tok/s):

  | Artifact | Requests | Text | Fixed 3 | Fixed 7 | Fixed 15 | MTP `auto` | `auto` |
  |---|---|---|---:|---:|---:|---:|---:|
  | NVFP4 27B | 1 | code | | 280.8 | 315.6 | | **329.5** |
  | NVFP4 27B | 1 | docs, sampled | | 176.3 | **178.9** | | 178.3 |
  | NVFP4 27B | 2 | code | | 507.6 | 504.4 | | **513.7** |
  | NVFP4 27B | 2 | docs, sampled | | **321.1** | 282.3 | | 314.7 |
  | NVFP4 27B | 4 | code | | **775.1** | 662.7 | | 725.1 |
  | NVFP4 27B | 4 | docs, sampled | | 508.0 | 409.8 | | **508.3** |
  | 35B-A3B | 1 | code | 615.0 | | | 655.4 | **686.4** |
  | 35B-A3B | 1 | docs, sampled | 391.2 | | | **441.3** | 428.6 |
  | 35B-A3B | 2 | code | 818.6 | | | **898.8** | 833.2 |
  | 35B-A3B | 2 | docs, sampled | 529.5 | | | **654.9** | 554.0 |
  | 35B-A3B | 4 | code | 940.4 | | | **1,070.5** | 926.6 |
  | 35B-A3B | 4 | docs, sampled | 611.9 | | | **872.1** | 679.3 |

  Against the configuration's best single fixed length overall (the 27B's 7, the 35B-A3B's 3) the
  geometric mean over these settings is +1.6 % and +6.1 % (against the 27B's 15: +8.2 %). It is not
  the fastest everywhere. The 27B's batched rounds are fixed 7's rounds (equal time per row-round
  within 0.5 %), so its batched differences are the texts the streams diverge into: four requests on
  code -6.5 % over these 24 slices, -0.5 % over 72 (791.7 against 795.9); two requests on sampled
  docs -2.0 %. The 35B-A3B is 1.5 % behind fixed 3 with four requests on code. MTP `auto` is the
  35B-A3B's faster drafter with several requests (+8-28 %) and on sampled docs. Long context, one
  request on code (four 64K and 128K-token prompts, decode tok/s): the 27B's `auto` 279.3 and 214.8
  against fixed 7's 269.9 and 216.0 and fixed 15's 261.1 and 204.6 (the policy narrows its widths
  as the context grows). The other KV formats take the INT8 timing: a format changes only the
  absolute round time, not the widths' order.

  Rejected: the 27B's split batched rounds (2026-10-21: a 15-draft proposal and a host wait cost
  0.5-0.9 ms a batched round, 1-8 % behind fixed 7 whatever widths they verified, also with a fixed
  8-column batch width); plain batched rounds of 3 or 5 drafts on the 35B-A3B (its chosen widths are
  -1.2 to +11.2 % against plain 3, geometric mean +3.8 %, and 3-36 % ahead of plain 5); the
  35B-A3B's 15-draft block (756.4 / 448.7 at one request: its 16-column proposal costs more than its
  drafts add); costs 0.01 and 0.005 per draft on the 27B (334.0 and 344.3 on code against 361.4 at
  0.02, decode tok/s); registering the groupwise 27B (one request: `auto` 300.0 / 171.3 against
  fixed 7's 300.8 / 184.6; re-timed 2026-10-25 on code and docs at one, two and four requests,
  decode tok/s: the 15-draft block +8.1 % on code and -4.8 % on docs at one request, its batched
  widths -4.6 to -14.5 %; batched rounds plain at 7 equal fixed 7 (-0.2 to +1.4 %); a 7-draft block
  equals fixed 7 at one request (-0.2 / +0.1 %): no variant is a consistent gain); a lower copy floor (one-row copies of 8-15 drafts replacing the block:
  no round changed on the copy workload); DFlash2 trees on the block: 15 drafts with 20- and
  24-column trees ran 1.3 / 2.5 % and 1.3 % behind the chain (same-text estimate -1.3 / -1.9 % and
  -1.0 %), and `auto` with automatic trees chose trees in 704 of 1,547 code rounds and 122 of 3,119
  docs rounds (-1.2 % / +2.2 %) because the tree controller sees a chain round only when the policy
  verifies the whole block, so automatic trees resolve to chains beside the policy. At 7 drafts
  automatic trees gain +11.5 % on code and +6.1 % on sampled docs against the 7-draft chain, and
  on sampled docs 7 drafts with trees (187.1) beat `auto` (178.3): trees chosen by the width policy
  on a short main chain are the open follow-up. A fixed MTP draft cost in place of `adaptive-mtp`'s
  learned fit did not help (NVFP4 27B at 0.06: 263.1 / 144.2 against 263.2 / 150.5 decode tok/s;
  35B-A3B at 0.19: 729.2 / 466.3 against 735.5 / 470.0), so `adaptive-mtp` keeps its fit.
  Measurement note: the earlier in-order gen tool let finished requests keep their lanes until the
  oldest one completed, so batched occupancy followed the finishing order (four requests on sampled
  docs: 3.40 rows a round for `auto` against 3.60 for fixed 7, 464 against 478 tok/s); every batched
  number above is from the worker tool.

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
  positions. A binding's first step takes the idle width too when no other lane is generating and
  the free KV pages alone (no eviction) hold its wide extent plus every reused block as if restored
  from the Host tier (2026-10-16; before, it kept one chunk). A Vision prompt, a recovering request
  and Replay keep the chunk, as does a step whose extra KV pages are neither free nor evictable
  cache.
- **Cost:** prefill buffers and workspace at the idle width (with automatic KV capacity, fewer
  cached KV pages); a request admitted during a wide step waits up to one wide step.
- **Speed** (2026-10-15, RTX 5090, official Qwen3.8-27B NVFP4, INT8 KV, `infernix_bench`, one
  24,576-token prompt alone): 10,547.6 → 11,533.4 tok/s (+9.3 %). The upstream measurement (NInfer
  5bcef9e2c): 35B-A3B 1.41 → 1.26 s, 27B NVFP4 3.28 → 3.06 s, with the decode stall behind a
  13.8k-token prompt unchanged. The wide first step (2026-10-16, same machine and artifact,
  `infernix_bench` ABBA, two runs per arm, spread under 0.6 %): pp1024 11,744 → 14,013 tok/s
  (+19.3 %), pp2048 +11.5 %, pp3072 +5.2 %, pp4096 +0.5 %, pp8192 +1.8 %. Its steps are the
  idle-width steps the evidence below already gated.
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
  | BF16 (2026-10-16) | −1.0e-3 (1.2e-4) / 1.3e-3 | −0.02 (0.07) / 0.50 | 1.6e-3 | 0.86 |
  | FP8 (2026-10-16) | −3.6e-4 (2.2e-4) / 1.3e-3 | +0.03 (0.11) / 0.50 | 9.8e-5 | 0.89 |
  | VQ2 (2026-10-16) | −1.1e-4 (1.6e-4) / 1.6e-3 | +0.04 (0.09) / 0.50 | −9.7e-4 | 1.05 |

  BF16, FP8 and VQ2 ran the same protocol a day later (the same reference; the top-1 budget is the
  absolute floor's 0.5 points by then, though each bound is within the old 10 % budget too).
  A first run with `--fast-path idle-wide-prefill` at each chunk was void: the scoring engine
  never widened a step, so its fast runs measured only the qualified paths. The evaluation then
  needed `evaluate-prefill --fast-chunk` (fixed the same day: the pair check had required equal
  chunks).
- **Groupwise 27B and 35B-A3B** (2026-10-18, their own evidence; both 16-bit activations, neither
  request-invariant). The same prefill gate on INT8 KV, the idle width (3,584) against the 896- and
  1,792-token chunks, jointly with `prompt-pv8` and `gdn-f16-accumulate`, passes on both:

  | Artifact | KL excess (bound) / 1.5e-3 | Top-1 excess, points (bound) / 0.50 | NLL excess | p99.9 KL ratio / 1.25 |
  |---|---|---|---|---|
  | Groupwise 27B | 2.7e-4 (4.5e-4) | +0.15 (0.18) | 3.4e-4 | 1.75 (passes: the fast p99.9 KL, 0.025, is below the 0.05 floor) |
  | 35B-A3B | 5.9e-4 (7.6e-4) | +0.26 (0.31) | -1.0e-4 | 1.23 |

  Speed, pp16384 ABBA, every run ahead: groupwise 27B +3.31 %, 35B-A3B +16.9 %. The other formats
  take the NVFP4 27B's per-format verdicts above (the path changes the step width, not a kernel).
- **Qualified:** NVFP4 27B, groupwise 27B and 35B-A3B, with every KV format.

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
- **`qsa-prepare-fused` elsewhere** (2026-10-18; it serves every dense format, and Dense8's gates
  cover its kernel): timed alone against the exact route (INT8 KV, ABBA) it is not faster on IQ3_S
  (pp16384 +0.15 %; plain decode after a 2,048-token prompt 122.4 -> 121.8 tok/s, -0.47 %, every
  run behind) or FP8 (pp16384 -0.04 %, MTP decode +0.47 %, both overlapping), so it is not
  registered there. Its MTP timings move with the text it changes, not with its own cost: IQ3_S MTP
  4 tg512 ran 241.8 -> 198.3 tok/s (-18.0 %, 450 -> 489 rounds), while Dense8 (beside `q8-sliced-k`
  and `gdn-conv-fused`) ran 116.3 -> 139.3 (+19.8 %, 579 -> 447 rounds; one exact run cold at 104.8),
  each arm's rounds also taking different expert-cache time on its own text. Dense8's plain decode,
  where acceptance cannot enter, is 102.4 -> 102.6 tok/s (+0.23 %, every run ahead) and its pp16384
  level (-0.88 %, overlapping, from one exact run at 7,034 against 6,911-6,913 for the other three),
  so the Dense8 registration stands.
- **Not qualified:** 16-bit expert activations, and other dense formats (no kernels).

**`expert-a4-residual`: two-term A4 activations in recipe C's wide expert calls (qualified
2026-10-15).**
- **What changes:** expert calls of more than eight columns on W4A16 experts take x and h as two
  NVFP4 terms of the A4 residual profile with an FP32 row scale, on the FP4 Tensor Cores, instead of
  the exact limb route (design §19.3.23).
- **History:** two terms first failed the INT8 prefill gate on mean KL (excess bound 8.92e-4
  against the then budget of 6.11e-4, four chunk sizes). Three terms on both sides, the only
  variant of a one-chunk screen within that budget, then passed every gate below except BF16 KV
  (prefill KL bound 4.24e-4, decode top-1 bound 2.95e-3 of 3.31e-3, pp32768/65536 +25.7/+26.9 %).
  With the absolute floor, and the single budgets that replaced it, two terms pass, and they replaced three, which cost 7-8 points of
  prefill speed.
- **Op qualification:** `infernix_offloaded_moe_wide_test` X1 against the exact canonical
  arithmetic: mean relative L2 0.31-0.41 % per case, worst entry 2.0 % (bounds 2^-7 and 2^-5).
- **Artifact:** orcarouter Qwen3.8-Flash-Next Uncensored NVFP4, converted with recipe C (FP8 dense,
  W4A16 experts), the protocol reduced as the user approved (prefill quick corpus at 896 and 7,168
  tokens; decode MTP at concurrency 2 over 256 segments; the reference and exact runs shared).
- **INT8 KV:** prefill KL excess 7.82e-4 (95 % bound 9.07e-4; budget 1.0e-3 when gated, 1.5e-3
  since 2026-10-16), top-1 excess 1.89e-3 (bound 2.41e-3 of 5.0e-3), perplexity ratio 0.99967,
  p99.9 KL ratio 1.08; decode TV excess 9.9e-4 (bound 1.37e-3 of 4.0e-3), top-1 excess 2.08e-3
  (bound 4.41e-3 of 5.0e-3, the closest), p99.9 KL ratio 1.09, every prompt length.
- **The other formats (prefill):** FP8, K8V4, NVFP4, K4V2 and VQ2 pass (KL bounds 7.3e-4, 3.8e-4,
  4.6e-4, 6.6e-4 and 6.1e-4; floors 7.5e-3, 1.21e-2, 1.49e-2, 1.53e-2 and 2.17e-2, all above
  INT8's 6.11e-3); they take INT8's decode verdict (6.1), INT8 being the passing format with the
  lowest floor.
- **BF16 KV failed against the 16-bit reference and passes against the precise one:** recipe C's
  exact route with BF16 KV was that reference itself (16-bit activations everywhere, chunking
  bit-exact), so its floor was 0 and no format noise absorbed any of the path's error. The fast
  route's whole divergence (KL 6.8e-3, top-1 2.9 %, about INT8 KV's own; three terms 6.35e-3)
  counted as excess, 4.5 times the 1.5e-3 budget. Against the precise reference (2026-10-21, NVIDIA
  FP4-Local, the same expert kernels; 6.1, item 1) the exact route has a floor (prefill KL 5.2e-3,
  decode TV 0.033) and the path passes both gates: prefill (quick corpus, 896 and 7,168 tokens, 36
  windows) KL excess 3.6e-4 (bound 4.6e-4), top-1 8.6e-4 (1.3e-3), NLL 5.6e-4, p99.9 ratio 1.03;
  decode (MTP at concurrency 2, 256 segments) TV excess 4.4e-4 (bound 8.0e-4), top-1 7.9e-4
  (3.1e-3), KL 2.7e-4, p99.9 ratio 0.97. It is registered on BF16 KV for FP4-Local and recipe C; the
  KV format does not enter the expert kernels, so each one's timing holds there.
- **Speed:** `infernix_bench` INT8 KV, chunk 4,096, ABBA (2026-10-14/15, before main 2ec268c2,
  whose changes leave recipe C's route alone): pp4096 1,631 → 1,637 tok/s (level), pp16384 6,175 →
  6,380 (+3.3 %), pp32768 6,620 → 8,803 (+33 %), pp65536 6,730 → 9,067 (+35 %); short prompts are
  bound by the expert stream over the x8 link.
- **NVIDIA FP4-Local (2026-10-20):** its 16-bit expert activations run the same NVFP4 W4A16 banks
  through the same kernels, so it takes recipe C's verdicts (6.1, item 1: the gates follow the
  kernels a path changes), on the same formats, and its own BF16 gates against the precise reference
  (above) add BF16 KV to both. Timed on its own (`infernix_bench`, INT8 KV, n-gram rows from the volume, three
  repetitions, the default route against it plus the path, ABBA): pp32768 8,389 / 8,385 → 11,301 /
  11,298 tok/s (+34.7 %, every run ahead), pp8192 3,352 → 3,358 (+0.2 %, every run ahead), tg256
  plain 109.7 / 110.0 → 109.8 / 109.9 (level). One request's expert calls stay at eight columns or
  fewer in decode (MTP rounds: five), so the path does not act there; the MTP figures (151.4 /
  161.6 → 146.6 / 150.5) follow `--draft-tokens auto`, whose rounds averaged 2.29-3.02 tokens in
  identical configurations, and are not a measurement of the path.

**`fp8-a8-pair`: two E4M3 activation terms in recipe C's FP8 projections (recipe C with BF16, FP8, NVFP4, K8V4, VQ2 or K4V2 KV).**
- **What changes:** prefill-width calls of Flash-Next's row-FP8 dense problems carry each 16-bit
  activation as two E4M3 terms on the 8-bit Tensor Core path, admitted per problem where faster
  than the exact route (cold L2, graph): [16384, 2560] from T = 33, [13312, 2560] from 97,
  [2560, 6144] and [2560, 640] from 513, [1280, 2560] from 1,025 (0.65-0.93 of the exact route
  beyond, except [2560, 6144] at T = 1,040-1,088: 1.18).
- **Arithmetic (since 2026-10-17):** both terms' per-column scales are powers of two (the least
  2^s with max * 2^-s <= 448), so x * 2^-s is exact, the high term keeps a BF16 value's top four
  significant bits and the low term the remaining four: the pair represents every value within
  2^14 of its column maximum exactly (smaller ones to 2^-20 of the maximum). The earlier
  max/448 scales were not powers of two and left about 2^-8 relative error per value. The GEMM is
  unchanged; op speed is within 0-1.6 % of the earlier pair (0.70-0.79 of the exact route at
  4,096 columns). A variant with a power-of-two scale per 32 values of K, applied by the
  block-scaled MMA, was nondeterministic on [2560, 6144] (about one 128 x 128 tile in a hundred
  differed between identical calls, cause not found) and was dropped.
- **Op qualification:** `infernix_linear_fp8_a8_test`: the quantizer against the exact oracle
  (bit-exact decoding wherever a value is within 2^14 of its column maximum, on every problem and
  the outlier activations), determinism across repeated calls in both tile widths, and the A16
  criterion at every problem's admission seams and tiles, outliers and K tails.
- **Model level with power-of-two scales (2026-10-17, beside two-term `expert-a4-residual`):**
  prefill (quick corpus, 896 and 7,168 tokens) passes on every format but BF16 (INT8 KL excess
  8.7e-4, bound 9.8e-4, against 8.0e-4 for `expert-a4-residual` alone; FP8 bound 8.2e-4, NVFP4
  6.7e-4, K8V4 3.9e-4, VQ2 9.5e-4, K4V2 9.1e-4). BF16 fails by construction, as
  `expert-a4-residual` does (the exact route is the reference; KL excess 6.4e-3: any change of
  bits moves near-tie expert routing). The INT8 decode gate over 1,024 segments fails on its tail
  alone: p99.9 KL ratio 1.270 against 1.25; TV bound 2.0e-3, top-1 bound 4.50e-3 of 5.0e-3, NLL
  excess 9.8e-3 under the 1.0e-2 cap. Its increment over `expert-a4-residual` alone passes every
  criterion (TV bound 5.7e-4, top-1 bound 1.6e-3, NLL 8.5e-3, p99.9 ratio 1.08); the joint tail
  fails on the 1.18 that `expert-a4-residual` already uses. Speed beside it (ABBA, as below):
  pp32768 8,752 → 9,221 tok/s (+5.4 %), pp65536 9,011 → 9,556 (+6.0 %).
- **Decode on each format (2026-10-17).** INT8's tail failure would pass to every format by 6.1's
  rule, so the other formats whose prefill passed ran their own decode gate (MTP at concurrency 2,
  256 segments, beside two-term `expert-a4-residual`; the reference shared, each format's exact run
  its own):

  | KV | TV excess (bound) / 4.0e-3 | Top-1 excess (bound) / 5.0e-3 | NLL excess / 1.0e-2 | p99.9 KL ratio / 1.25 | Verdict |
  |---|---|---|---|---|---|
  | FP8 | 3.7e-4 (7.9e-4) | -1.4e-3 (1.1e-3) | 1.2e-3 | **1.316** | fails |
  | NVFP4 | 3.8e-4 (9.2e-4) | 1.5e-3 (**4.44e-3**) | 9.3e-4 | 1.12 | passes, close |
  | K8V4 | 4.5e-4 (9.2e-4) | -1.5e-3 (1.3e-3) | -1.1e-3 | 1.08 | passes |
  | VQ2 | 1.7e-4 (6.6e-4) | -4.9e-4 (2.3e-3) | 9.4e-4 | 1.04 | passes |
  | K4V2 | 9.6e-5 (5.3e-4) | -7.3e-4 (1.7e-3) | -5.2e-4 | **1.18** | passes, close |

  FP8's tail rests on about 16 positions with a small default p99.9 (0.247 against 0.326), and
  NVFP4's top-1 and K4V2's tail are within 11 % of their limits, so those three ran the decode gate
  over 1,024 segments (the shared 1,024-segment reference), whose verdicts replace these:

  | KV | TV excess (bound) / 4.0e-3 | Top-1 excess (bound) / 5.0e-3 | NLL excess / 1.0e-2 | p99.9 KL ratio / 1.25 | Verdict |
  |---|---|---|---|---|---|
  | FP8 | 2.4e-3 (3.84e-3) | 1.7e-3 (3.9e-3) | 6.7e-4 | 1.12 | passes |
  | NVFP4 | 3.7e-4 (2.3e-3) | 5.3e-4 (3.2e-3) | -5.1e-3 | 1.15 | passes |
  | K4V2 | 1.8e-3 (3.3e-3) | 2.0e-3 (4.3e-3) | 7.1e-3 | 0.93 | passes |

  `fp8-a8-pair` is registered on FP8, NVFP4, K8V4, VQ2 and K4V2 (FP8's TV bound, 0.96 of its
  budget, is the closest margin), and since 2026-10-21 on BF16: against the precise reference its
  prefill gate beside `expert-a4-residual` passes (quick corpus, 896 and 7,168 tokens: KL excess bound
  4.5e-4, top-1 1.6e-3, p99.9 KL ratio 1.03), its decode gate (MTP at concurrency 2) was a near pass at
  256 segments and passes at 1,024 (TV excess bound 2.7e-3, top-1 3.4e-3, KL 4.2e-3, NLL excess
  -2.6e-3, p99.9 KL ratio 0.84; prompts 512 / 2,048 / 8,192 TV excess -1.5e-4 / 1.6e-3 / 2.7e-3). Speed beside `expert-a4-residual` was measured on INT8 KV (+5.4 /
  +6.0 % at 32K / 64K above); the other formats were not timed. It is request-coupling: a dense
  call takes the pair route from its problem's admission width, which batched rounds share.
- **A third h term for `expert-a4-residual` (2 + 3, tried 2026-10-17, not kept).** Recipe C, INT8
  KV, ABBA: two terms alone 8,736 / 8,997 tok/s (pp32768 / pp65536), 2 + 3 alone 8,548 / 8,801
  (-2.2 %), 2 + 3 with `fp8-a8-pair` 9,016 / 9,334 (+3.2 % / +3.7 % over two terms alone). Its
  prefill gate beside `fp8-a8-pair` passes on every format but BF16 with less error (INT8 KL
  excess 6.8e-4 against 8.7e-4 at 2 + 2), but its INT8 decode gate over 1,024 segments fails more
  clearly than 2 + 2's: top-1 bound 5.66e-3, p99.9 ratio 1.56 (TV excess 2.6e-3 against 8.2e-4).
  The decode result runs against the prefill one and is not explained; run-to-run variation of
  the wide expert route was not measured. Two terms stay.
- **Model level with max/448 scales (2026-10-15/16):** gated on the orcarouter recipe C artifact jointly with
  `expert-a4-residual` at three terms (the variant qualified then), the reduced protocol as above.
  Prefill passes on every format but BF16 (INT8 KL bound 6.61e-4 of 1.0e-3, against 4.24e-4 for
  three-term `expert-a4-residual` alone). The INT8 decode gate fails on its tail: p99.9 KL ratio
  1.35 against 1.25 (three terms alone 1.07); TV and top-1 pass (top-1 bound 4.29e-3 of 5.0e-3).
  Beside two-term `expert-a4-residual` (2026-10-16, INT8 KV): prefill KL excess 1.006e-3 (bound
  1.14e-3), which failed the 1.0e-3 budget it was gated under and passes the single 1.5e-3 budget,
  every other criterion passing; decode fails its tail again (p99.9 KL ratio 1.266 against 1.25),
  TV and top-1 passing. The p99.9 of 256 segments rests on about 16 positions, so the INT8 decode
  gate was repeated over 1,024 segments (65,536 forced positions per route), its verdict alone
  counting: it fails three criteria (2026-10-16). Top-1 disagreement excess bound 5.31e-3 against
  5.0e-3, forced-token NLL excess 1.23e-2 against the 1.0e-2 cap, p99.9 KL ratio 1.261 against
  1.25; TV excess bound 3.36e-3 of 4.0e-3, KL excess 2.8e-3 of 1.0e-2 and every prompt length pass
  (floor: KL 2.11e-2, TV 3.91e-2, top-1 3.67 points). The fuller sample confirmed the tail and
  moved top-1 and NLL past their limits, so it is not a small-sample artifact. Attribution over
  the same segments (2026-10-16): two-term `expert-a4-residual` alone passes (TV bound 2.38e-3,
  top-1 bound 4.41e-3, NLL excess 1.2e-3, p99.9 ratio 1.18), and `fp8-a8-pair`'s increment over it
  passes every criterion but NLL (1.11e-2 against the 1.0e-2 cap, bound 2.0e-2; TV bound 1.98e-3,
  top-1 bound 2.57e-3, p99.9 ratio 1.07). The joint failure is `fp8-a8-pair`'s NLL shift on top of
  the top-1 and tail budget `expert-a4-residual` already uses. A bit-exact alternative cannot
  recover its speed: the exact route's BF16 MMAs run at 227-243 TFLOP/s
  at 2,048-16,384 columns, the Tensor Core pipe 91.7 % busy (ncu, [16384, 2560] at 4,096), level
  with the BF16 GEMM of the same shape, so the gain is the FP8 Tensor Core rate (0.69-0.78 of the
  exact route's time at 4,096 columns). Speed beside two-term
  `expert-a4-residual` (`infernix_bench`, INT8 KV, chunk 4,096, ABBA): pp32768 8,732 → 9,277 tok/s
  (+6.2 %), pp65536 9,022 → 9,619 (+6.6 %). The other formats were not gated with these scales.

**`ggml-bf16-prefill`: BF16 tensor-core tiles in Flash-Next's GGUF prefill (IQ3_S with INT8, FP8,
NVFP4, K8V4, VQ2 or K4V2 KV, qualified 2026-10-19; BF16 KV, 2026-10-20).**
- **What changes:** eager prefill calls of 32 or more columns run the GGUF dense projections
  (`ops::ggml_linear`, `LinearRoutes::GgmlBf16Wide`) as BF16 m16n8k16 MMAs with FP32 accumulation over
  the exact route's decoded planes: each code is exact in BF16, x is read as stored, each
  sub-block's FP32 sum is scaled by d · sub (exact in binary32), the K-quants subtract
  dmin · min · Σx, and κ applies at the end. The GGUF expert banks' MoE calls of 14,336 or more
  columns run the same tiles over expert planes decoded into the experts' workspace
  (`WideA16::kGgmlBf16`). The exact route rounds every activation column to 21-bit integers and sums
  exactly; this one keeps x exact and rounds the sums. A prompt's bits depend on its call boundaries
  and the layer walk's MoE grouping, so it couples requests (`request_invariant` is false). Decode,
  verification and drafting keep their kernels (graph calls never take it).
- **Op qualification:** `infernix_ggml_linear_test` against an FP64 oracle with the bound
  (3 K / 32 + 64) · 2⁻²⁴ · Σ|wx| (plus 2⁻⁸ |ref| for BF16 outputs) on every GGUF format, K, width,
  input permutation and the tiled layout; calls under 32 columns keep the exact bits. With FP32
  outputs its largest error is 1.4-5.9e-7 of Σ|wx|, the exact route's 2.2e-7 to 6.4e-6; with BF16
  outputs 99.8-100 % of outputs equal the exact route's. `infernix_offloaded_moe_ggml_test` holds
  the expert route to the exact one (every column within 2⁻⁶ normwise; 99.77-99.86 % of outputs
  equal) and a binary64 chain on sampled (token, slot) pairs, at 14,373 columns.
- **Prefill gate** (RTX 5090, CUDA 13.4, the GSQ-RCO IQ3_S import, quick corpus, 14,336-token windows
  at chunks 896 and 7,168, so the walk's MoE groups reach the expert tiles; jointly with the
  qualified paths, of which `gdn-f16-accumulate` acts in prefill):

  | KV | Floor KL / top-1 | KL excess (bound) / 1.5e-3 | Top-1 excess, points (bound) / 0.5 | NLL excess | p99.9 KL ratio | Result |
  |---|---|---|---|---|---|---|
  | BF16 | 0 / 0 | 5.7e-3 (7.0e-3) | +2.58 (2.88) | 3.9e-4 | ∞ (floor 0) | fails |
  | INT8 | 5.3e-3 / 2.48 % | 4.9e-4 (5.5e-4) | +0.11 (0.16) | 2.5e-5 | 1.11 | passes |
  | FP8 | 6.6e-3 / 2.80 % | 4.6e-4 (6.5e-4) | +0.08 (0.13) | 1.6e-4 | 1.13 | passes |
  | K8V4 | 1.09e-2 / 3.68 % | 2.5e-4 (5.4e-4) | −0.03 (0.03) | 7.1e-4 | 1.03 | passes |
  | NVFP4 | 1.36e-2 / 4.08 % | 3.5e-4 (7.3e-4) | +0.01 (0.08) | 5.6e-4 | 1.02 | passes |
  | K4V2 | 1.40e-2 / 3.96 % | 4.0e-4 (7.8e-4) | +0.01 (0.07) | 2.9e-4 | 1.09 | passes |
  | VQ2 | 2.00e-2 / 4.68 % | 0.6e-4 (3.7e-4) | +0.05 (0.10) | 3.2e-4 | 0.99 | passes |

  Every context bucket passes on the six. BF16 KV fails, and not through this path alone: the IQ3_S
  exact route with BF16 KV is the reference itself (floor 0), so any change of its arithmetic counts
  whole, and every change measured there moves it by about the same amount. Against the reference
  (mean KL, top-1 agreement, NLL change): `gdn-f16-accumulate` alone 5.9e-3, 97.38 %, +5.2e-4; this
  path alone (after `gdn-f16-accumulate` was removed there, below) 5.5e-3, 97.43-97.44 %, −0.9e-4 and
  −4.8e-4 at the two chunks; both together 5.6-5.7e-3, 97.40-97.45 %, +3.0e-4 / +4.7e-4. Either one
  takes about what INT8 KV's quantization takes (its floor, 5.3e-3), and the second adds nothing
  measurable (its increment over the first: KL −1.9e-4, bound 1.2e-4): most likely near-tied expert
  routing that any perturbation flips, as for `expert-a4-residual` (not measured). An increment
  measured first suggested the excess was `gdn-f16-accumulate`'s alone; the path's own run shows it
  is not. With floor 0 the absolute budgets (1.5e-3) admit no change of IQ3_S's arithmetic on BF16 KV.
- **Against the precise reference** (2026-10-20; the same artifact, corpus, chunks and joint
  set, 36 windows, 522,334 positions per route), the cell is decided on fidelity instead:

  | KV | Floor KL / top-1 | KL excess (bound) / 1.5e-3 | Top-1 excess, points (bound) / 0.5 | NLL excess | p99.9 KL ratio | Result |
  |---|---|---|---|---|---|---|
  | BF16 | 5.66e-3 / 2.60 % | −1.4e-4 (4.2e-5) | −0.05 (−0.01) | 3.9e-4 | 1.00 | passes |
  | INT8 | 5.74e-3 / 2.62 % | −0.8e-4 (−0.7e-5) | −0.02 (0.03) | 2.5e-5 | 0.93 | passes |
  | VQ2 | 2.01e-2 / 4.69 % | −1.8e-4 (8.3e-5) | −0.00 (0.05) | 3.2e-4 | 1.02 | passes |

  Every context bucket passes on the three. BF16 KV is registered for this path, whose kernels see
  no KV format and which changes no decode kernel (the decode criteria are not required), with
  INT8's timing. `gdn-f16-accumulate` is registered on IQ3_S with BF16 KV on the same footing
  (2026-10-20): it changes only chunked GDN calls, which decode reaches only through the prompt's
  state, and the prefill gate's later positions carry that state through many chunks; its BF16
  decode gate had failed against the old reference (above), whose exact-kernel scoring is the very
  arithmetic the path replaces, while at Op level the path is closer to FP64 than the exact route.
- **Decode gate** (plain, 256 segments, prompts of 512-8,192 tokens, whose prefill takes the dense
  tiles but not the expert tiles): INT8, the passing format with the lowest floor, passes (TV excess
  5.4e-4, bound 9.3e-4 of 4e-3; top-1 +0.10 points, bound 0.32; KL 2.4e-4; NLL −2.9e-4; p99.9 KL
  ratio 1.09; every prompt length). BF16, run as well, fails on a nonzero floor (TV 0.029): TV excess
  3.9e-3 (bound 4.34e-3 of 4e-3), top-1 +0.51 points (bound 0.75), p99.9 KL ratio 1.47, the excess
  in the 2,048- and 8,192-token prompts (TV 6.0e-3 and 5.3e-3, against 4.3e-4 at 512). The same
  holds for each path alone: `gdn-f16-accumulate` TV excess 4.2e-3 (bound 4.6e-3), top-1 +0.62
  points (0.84); this path alone 4.0e-3 (bound 4.45e-3), +0.48 points (0.70), p99.9 KL ratio 1.17,
  again in the 2,048- and 8,192-token prompts: a fixed set of fragile segments. No decode,
  verification or drafting kernel changes, so the decode criteria are not required (6.1, item 2).
- **`gdn-f16-accumulate` removed from IQ3_S with BF16 KV** (2026-10-19, the owner's decision): alone
  it fails both gates there (above); its registration rested on the 27B's gates (one gate per
  kernel), whose BF16 floors are not 0. IQ3_S with BF16 KV now runs both paths' exact counterparts.
- **Speed** (`infernix_bench`, INT8 KV, three rotated rounds, design §19.3.29): against the exact
  route, pp32768 3,839 → 5,113 tok/s (+33.2 %), pp65536 4,809 → 6,924 (+44.0 %), every round ahead;
  the dense tiles alone +20.5 % / +25.9 %. The other formats take INT8's timing (6.1, item 4): the
  path's kernels see no KV format, at the same shapes and widths. Server time to first token,
  `windows-2026.10.18` against this build (the path together with design §19.3.29's exact items;
  the README workload, INT8 KV): 128K 29.3-29.5 → 19.7-19.9 s, 250K 58.8-58.9 →
  39.2-39.4 s, 8K 2.16-2.17 → 1.62-1.65 s. Decode is unchanged: on the same text (the exact route, a
  fixed draft length, ABBA) the branch's second session ran 182.6 tok/s against the release's 182.3
  and 183.6; its first, right after a build, ran 174.5 (design §19.3.29).
- **Not qualified:** BF16 KV (above); other dense families (no kernels).

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

### 6.8 Exact-route numerics changes

A change to exact-route bits (6.1) is recorded here with its gate, and so is the acceptance of a
reference ([lossless route](lossless-route.md)).

**Qwen3.8-Flash-Next precise reference (accepted 2026-10-20; the gate reference since, by the
owner's decision).** Its three checks:
1. *More precise:* on the official FP8 import's blocks, with identical inputs, the mean relative
   error against the binary64 evaluation at the same BF16 points (`tools/flash_next/reference.py
   --bf16-boundaries`) falls from the exact route's to the reference's: projections 4.7e-3 → 3.6e-4,
   MoE 2.0e-3 → 6.2e-5, whole block 2.4e-3 → 4.2e-4; attention's largest error is about unchanged
   (5.7e-3 → 5.3e-3). Model-level KL cannot rank the two (it saturates), hence the block check.
2. *Still has teeth:* VQ2 KV measured as if it were a fast path (IQ3_S prefill, quick corpus) fails
   every criterion: KL excess 1.4e-2 against the 1.5e-3 budget.
3. *Sane floors:* the IQ3_S exact route's prefill floors against it are KL 5.7e-3 on BF16 and INT8
   KV and 2.0e-2 on VQ2, nonzero, where the exact-kernel reference gave BF16 KV a floor of 0.

**Faster reference arithmetic (2026-10-20).** The reference route spent over 99 % of its GPU time in
binary64 dot products (Nsight Systems over one gate window, RTX 5090, CUDA 13.4: recipe C 538 s, routed
experts 57.5 % and dense projections 42 %; IQ3_S 492 s, 59 % and 40 %), which the RTX 5090 runs at 1/64
of its binary32 rate. They now form each product exactly in binary32 (BF16 x times a weight of at most
16 significant bits; Q8 and dense NVFP4 weights split by an FMA; GGUF codes, with each sub-block's scale
and offset applied to its sums) and sum in double-binary32 arithmetic (Sum2 blocks of 32 terms joined by
AccurateDWPlusDW): within (4096 + 2 K) u² of the exact sum relative to the terms' magnitudes, against
binary64's 2 K 2^-53.
- Op oracles: all 255,750 outputs of `infernix_precise_linear_test` (every dense and GGUF format, BF16
  and FP32 outputs, widths 1-37) are correctly rounded, none lies within the route's error of a tie,
  and none differs from the binary64 loop; every routed-expert output of the four bank kinds equals the
  correctly rounded oracle, bit-identical in every placement.
- Against the binary64 reference on the IQ3_S prefill gate's corpus (quick, 14,336/7,168, BF16 KV,
  261,167 positions): 21 of 36 windows are bit-identical throughout. In the other 15 a rounding falls
  differently once (first differing scored position 0 to 6,003) and the difference propagates through
  later layers and tokens: 75.3 % of positions keep bit-identical target log-probabilities and top-64
  lists, KL from the binary64 reference is 1.47e-4, top-1 agreement 99.79 %, corpus NLL 1.226203
  against 1.226187. Two references that each round correctly except within their own error of a tie
  differ wherever an exact value lies inside either error band; over the window's billions of
  roundings a few do. Which op rounds differently first was not traced.
- Check 1 on the FP8 import is unchanged: KL against the binary64 evaluation 1.6477e-2, as before, and
  the block errors against it equal the binary64 reference's to three digits (projections 3.59e-4,
  MoE 6.2e-5, whole block 4.18e-4).
- Speed: recipe C's window takes 82 s of GPU time (538 s, 6.5x: NVFP4 experts 5.9x gate/up and 7.3x
  down, FP8 projections 9.9x, Q8 4.0x) and IQ3_S's 153 s (492 s, 3.2x: GGUF experts 3.3x gate/up and
  3.9x down, GGUF projections 2.8x, BF16 5.5x); the GGUF decoders are now the slowest part. The IQ3_S
  gate corpus scored at 98.3 tok/s (38.6 before) with the first version of the kernels, before the
  decode improvements the window profiles measure.
The owner adopted this arithmetic as the gate reference on 2026-10-20, with the IQ3_S prefill gate
(checks 2 and 3 and the `ggml-bf16-prefill` BF16 verdict) re-run against it the same evening (quick
corpus, 36 windows, chunks 896 and 7,168, `ggml-bf16-prefill` + `gdn-f16-accumulate`; the reference
pass took 37 minutes): the floors are BF16 5.67e-3, INT8 5.74e-3 and VQ2 2.01e-2 (check 3; 5.7e-3,
5.7e-3 and 2.0e-2 under the binary64 sums), VQ2's default runs as the candidate against BF16's still
fail every criterion (check 2: KL excess 1.44e-2, bound 1.68e-2, p99.9 ratio 3.64), and the joint
paths pass on BF16 (KL excess -1.43e-4, bound 4.1e-5; top-1 -0.04 points, bound 0.009; p99.9 ratio
1.007), INT8 (KL -7.9e-5, bound -6.0e-6) and VQ2 (KL -1.77e-4, bound 8.2e-5), as they did under the
binary64 sums.

**35B-A3B router logits in BF16 (2026-10-22, owner-approved).** The sparse MoE router rounds each
logit (and the shared expert's gate logit) to BF16 before the top-8 choice and the softmax, as the model
computes them (lossless-route.md §2: `l = bf16(x W_router)`); the reproducible route had kept the FP32
sums. Gate, main's reproducible route as base, against the lossless references (quick corpus, chunk 896;
decode at four requests, 256 segments): prefill KL excess -5.2e-4 (bound -1.6e-4) on BF16 KV and -6.1e-4
(-3.8e-4) on INT8, top-1 -0.10 points, corpus NLL -2.6e-4 and -3.7e-6, p99.9 ratio 0.84 and 0.88; decode
TV -9.5e-4 (bound -7.0e-4), top-1 -0.12 points, p99.9 ratio 0.73. It lowers the route's processing cost
against the design reference from 4.24e-3 to 3.72e-3; the MoE block's other BF16 rounding points
gave no further gain (tested the same day). Speed (ABBA, INT8 KV, one request): plain decode -0.01 %
(mixed), prefill +0.06 to +0.14 %, 16,384-token prompt decode +0.21 %; MTP decode -3.6 % on the bench's
one prompt came from that text's acceptance (597 rounds against 576, time per round unchanged), and over
16 prompts of the bench corpus the pooled acceptance length is 2.219 against 2.172 (+2.1 %; per prompt
+0.030, standard error 0.035; ahead on 9, behind on 6), so it costs nothing.

**35B-A3B Q8 shared-expert prefill (`4f0bb18f`, NInfer `610f79e9d`).** The Q8 gate/up and down
prefill kernels of `Qwen3.6-35B-A3B` apply each 32-wide group's FP16 scale to the FP32 partial sum
and pass the SiLU output as a BF16 high/low pair, where the old route rounded each code × scale to
BF16 before the MMA. Against the FP64 oracle the worst relative L2 error fell 3× (Q4/Q5) and 5× (Q8);
prefill got 2.3-2.6 % slower, decode is unchanged. It shipped without the model-level gate 6.1 asks
for; the gate was run on 2026-10-18 (`compare-prefill`, the build before `4f0bb18f` as base and the
current one as candidate, both `--reproducible`; `qwen3_6_35b_a3b.ninfer`, the quick corpus,
14,336-token windows at stride 7,168, chunks 896-7,168, 36 windows, 1,044,668 positions per route).
The reference route runs the same shared-expert kernels as the route that writes it, so the gate ran
against each build's reference:

| KV | Reference from | KL excess (bound) | Top-1 excess (bound) | Corpus NLL excess | p99.9 KL ratio | Verdict |
|---|---|---:|---:|---:|---:|---|
| INT8 | base build | 5.8e-5 (1.7e-4) | 2.1e-4 (3.8e-4) | -6.3e-5 | 1.01 | pass |
| INT8 | new build | -3.4e-5 (2.4e-5) | -3.0e-4 (-1.1e-4) | -6.3e-5 | 1.00 | pass |
| BF16 | base build | 7.7e-4 (1.0e-3) | 5.0e-3 (5.5e-3) | -3.0e-5 | 1.38 | fails top-1 and tail |
| BF16 | new build | -9.2e-4 (-7.0e-4) | -5.1e-3 (-4.5e-3) | -3.0e-5 | 0.72 | pass |

The KL and top-1 excesses change sign with the reference and are about equal in size, so they
measure which route the reference shares, not fidelity; BF16 KV shows it most because no KV
quantization hides it (the new build's BF16 KL against the old reference, 3.19e-3, is its INT8 KL).
The corpus NLL, which no reference enters, is lower with the new route in both formats, as the Op
oracle predicts. Under 6.1's rule for a change to the reference route, the new route is the more
precise one (its Op error is lower), so the new-build rows decide: every excess estimate is at most
zero and every criterion passes on both formats. The change stays on the exact route as a
precision fix, and the new route is the reference.
