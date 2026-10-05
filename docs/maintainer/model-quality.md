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
- Exact outputs (for example BF16 KV prefill, where no fork kernel differs from upstream) must
  still match exactly; noise tolerance applies only to paths whose arithmetic changed.

## 2. Tools

| Tool | Measures |
|---|---|
| `ninfer-perplexity --prefill-chunk N` | perplexity at a chosen prefill pass size ([guide](../perplexity.md)) |
| `ninfer-perplexity --a16-activations --save-top-tokens FILE` | the 16-bit-activation reference distribution |
| `ninfer-perplexity --kl-reference FILE` | KL divergence and top-1 agreement against a saved reference |
| `ninfer_decode_quality_gen` / `ninfer_decode_quality_judge` | end-to-end greedy decode through the production speculative path, judged token by token against the 16-bit reference ([benchmarks](../../bench/README.md#decode-quality)) |

Perplexity scoring runs prefill only; decode kernels, speculative verification and n-gram drafting
need the decode-quality pair. The generator uses only Engine API that upstream also has, so the
same source builds in an upstream checkout for a like-for-like comparison.

## 3. Recorded results

Measured on 2026-10-05 on an RTX 5090 (CUDA 13.4, Windows), Gold-Star-Infer at `5fbfc3db` against
upstream `68c54356` plus the Windows port (with only a `--prefill-chunk` option added to its
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

### 3.2 Fork against upstream, prefill

Perplexity at six chunk sizes (512, 1024, 2048, 3584, 7168 and one 14,336-token pass), paired by
chunk size:

| KV | Fork mean | Upstream mean | Fork - upstream | By chunk size |
|---|---|---|---|---|
| BF16 | 2.33446 | 2.33446 | 0 | identical at all six |
| INT8 | 2.32850 | 2.33129 | -0.12 % | fork lower at 5 of 6; t = -1.4, not significant |
| NVFP4 | 2.33694 | 2.33643 | +0.02 % | fork lower at 3 of 6; spread 0.54 %; worst +0.89 % at 1024 |

The fork changes no arithmetic shared by every KV format, and its INT8 and NVFP4 prompt-attention
kernels are as accurate as upstream's within this resolution.

### 3.3 Fork against upstream, decode

32 corpus segments of 1,536 tokens, 384 greedy tokens each (12,288 tokens per build), DFlash2 with
7 drafts and the proposal head, INT8 KV, judged on each build's own prefixes. The same generated
tokens were judged twice: by the study's reference (4-bit sites at 16 bits) and by the committed
`ninfer_decode_quality_judge` (`a16_activations`, every site at 16 bits):

| Build | Agreement, 4-bit sites lifted | Regret | Agreement, `a16_activations` | Regret |
|---|---|---|---|---|
| Upstream | 0.9736 | 0.0080 | 0.9743 | 0.0073 |
| Fork | 0.9736 | 0.0086 | 0.9762 | 0.0072 |
| Fork with n-gram drafting (15 drafts, min match 12) | 0.9729 | 0.0087 | 0.9764 | 0.0071 |

Greedy decode chooses the reference's top token at least as often as upstream's under both
references; the differences are a few tenths of a percent and change order between them, so they
are within noise. Builds share their first
54-68 generated tokens on average before rounding noise makes them diverge; the fork with and
without n-gram drafting shares 267 of 384.

### 3.4 Rejected: adaptive (4 or 6) activation block scaling

Each 16-value activation block chooses the scale that maps its largest magnitude to 6 or to 4,
whichever reconstructs the block with less squared error. On the official artifact, over seven
chunk sizes, KL to the reference fell 4.3 % (not significant) and top-1 agreement rose 0.10
points at all seven, with perplexity unchanged. On the production NVIDIA artifact KL rose 12 %
(0.0191 to 0.0214) and perplexity 0.6 %. With `ninfer_bench` (3 alternating passes, INT8 KV,
DFlash2), prefill was 0.4-0.75 % slower at 16K, 64K and 2K prompts in every pass and decode about
0.4 % slower per round. Not adopted: `quantize_nvfp4_k16` keeps the single amax-to-6 scale.

Also rejected for the same trade-off: 16-bit MLP activations in prefill (table 3.1). An 8-bit
activation path would need FP4 x FP8 block-scaled MMA, which takes UE8M0 scales per 32 values,
not NVFP4's E4M3 scales per 16.

## 4. Repeating the measurements

```bat
rem 16-bit reference, one pass per window
ninfer-perplexity model.ninfer --text corpus.txt --context 14336 --stride 7168 --kv-dtype bf16 ^
  --prefill-chunk 14336 --a16-activations --save-top-tokens ref.top
rem a build or setting under test, at several chunk sizes
ninfer-perplexity model.ninfer --text corpus.txt --context 14336 --stride 7168 --kv-dtype int8 ^
  --prefill-chunk 4096 --kl-reference ref.top
```

INT8, NVFP4, FP8 and K8V4 KV round the chunk down to whole 896-token prompt-attention waves, so
use a window that is a multiple of 896 (14,336 is 16 waves) to keep the one-pass reference one
pass. The decode-quality commands are in [the benchmark guide](../../bench/README.md#decode-quality).
