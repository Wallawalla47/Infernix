# Qwen3.5-122B-A10B with offloaded experts on one RTX 5090: implementation plan

**Status: in progress (2026-10-22): runs end to end on the RTX 5090 (§0), on main 3d1931c7.** This is a temporary planning document for running
[ornith-ai/Ornith-1.5-122B-NVFP4](https://huggingface.co/ornith-ai/Ornith-1.5-122B-NVFP4) (revision
`f049036f`), a training instance of `Qwen3_5MoeForCausalLM` at the Qwen3.5-122B-A10B geometry, with
its routed experts in host RAM behind a VRAM expert cache, as Qwen3.8-Flash-Next runs. When the work
lands, the stable content moves into the active authorities listed in §17 and this file is removed.

The owner's terms (2026-10-21/22):

- every weight except the MTP drafter is imported **bit-exactly** from the NVFP4 checkpoint;
- the MTP drafter may be re-quantized;
- the FP8 projections use the checkpoint's **8-bit activations** (static input scales) where the
  other Qwen3.5 artifacts use their low-bit activations, in prefill. Decode and verification run
  16-bit activations on the reproducible route, as on every Qwen3.5 artifact;
- Vision is included, with the vision tower offloaded to RAM;
- the implementation is **highly optimised from the first build**: every kernel below comes with its
  throughput budget, tensor-core mapping, occupancy and layout, and its first GPU build is checked
  with Nsight Compute.

---

## 0. Status after the first hardware runs (2026-10-22)

RTX 5090, i9-13900K, 96 GB DDR5-5800, PCIe link measured 27.6-27.7 GB/s; the artifact
`Ornith-1.5-122B-NVFP4-Infernix-0000{1,2,3}-of-00003.infernix` (75.1 GB, 69.9 GiB); bf16 KV unless stated.

**Built and verified.**

- Conversion and `verify_qwen3_5`: 12,336 expert records, 1,178 tensors and 240 FP8 input scales
  equal the checkpoint word for word; only the MTP (777) and DFlash2 (55) tensors are re-quantized.
- Startup 16-19 s (weights 12 s at 5.5 GiB/s; the RAM ledger keeps every expert in pinned RAM;
  4,450-4,845 VRAM frames, 36-39 % of the experts; a saved expert state restores in 0.5 s).
- Op tests at the 122B shapes: the offloaded MoE against the CPU engine bit for bit at 1-160
  columns on every placement (a wide-route bug for the shared job's x rows was found and fixed),
  RoPE (32, 2) fixed instances, the 12,288-channel conv split, the Q8 MTP attention input
  [17408, 3072] (oracle and column invariance), the DFlash2 drafter at H 3072 / 6 layers (dynamic
  grouped convolutions, Q8 SwiGLU [18432, 3072], context K/V materialization over 5 or 6 layers,
  Q/K/V input, pack-tail norm), and `linear_topk` over the NVFP4 head (exact rg16 FP32 logits).
- Real-model checks: plain, `--reproducible` (two runs bit-identical across cache states),
  `--lossless` (eager rounds), MTP, DFlash2, vision with the tower offloaded (chart facts correct);
  the exact route's request invariance with no speculation, MTP and DFlash2; preemption and Replay
  (none, MTP); the MoE real test (MTP, prefix reuse, vision, 262,144 context). That test's append
  check had failed on main too since the 2026-10-05 prefix cache (an endpoint is published only
  64 tokens past the root; its first turn was 6 tokens): its first turn is now a 64+-token message,
  with `free-cache-resume` requested as the other prefix tests do.
- Non-real `ctest`: 241 of 243 pass. `infernix_hybrid_cache_defaults_test` fails on main as well
  (3d1931c7 accepts `--draft-tokens auto` with DFlash2; the test still expects a refusal).
  `infernix_cuda_stack_check`: the 149 new kernel keys (this geometry's attention, RoPE, conv, Q8
  and drafter shapes, the rg16 head, the FP32-scale FP8 route, Flash-Next's renamed wide down
  instance at its old 40 bytes) are recorded after review, 86 of them equal to their 16-head
  siblings; five
  existing kernels' frames grew and stay flagged until an A/B against main decides them: the 27B
  FP8 residual-add A8 TMA instances (32-48 → 48-88 bytes: the FP8 Ops read the per-row scale format,
  BF16 or FP32, from a runtime flag) and Flash-Next's wide gate/up instance (24 → 32 bytes, now
  templated on the expert geometry).
- No fast path is qualified for this configuration yet (D9), so the default route runs the same
  kernels as `--reproducible`; every number below is that route.

**Measured against §4's targets** (`infernix_bench`, one request; run-to-run spread about 5 %):

| Measure | Measured | Must / Target |
|---|---:|---:|
| Plain decode, short context, warm cache (tg128) | 80.4-82.6 tok/s (CLI 69.6-74.0) | ≥ 100 / ≥ 140 |
| Prefill, fresh 32K prompt (pp32768, chunk 16,384) | 6,183-6,519 tok/s | ≥ 6,000 / ≥ 9,000 |
| Prefill pp4096 | 1,619-1,750 tok/s | — |
| Best speculation | first runs: none faster than plain decode (below); 2026-10-25: gated MTP 2 drafts 67-103 tok/s on corpus prompts, +14 % over plain (§12.2) | ≥ 140 / ≥ 200 |

- **Decode is bound by the misses.** Decode hit rate 73.5-78 % (layers 0-19 at 58-65 %, 20-47 at
  77-87 %), about 84-100 misses per token. The CPU serves ~80 % of them (`misses / 3` rounds to 0 on
  the link at ~2 misses per layer) at 85-90 µs each with six workers (~60 GB/s, ~65 % of the DRAM
  bandwidth); the per-layer CPU leg (~150-350 µs) is the critical path against ~90 µs of GPU work.
  Flash-Next measured more workers slower. No speed in this section uses `--expert-tail-skip` (off by
  default; it changes the output and lowers quality). For reference only, R 7/6/5 decoded
  76.5/84.3/86.7 tok/s against 73.9 on one prompt.
- **Speculation does not pay here.** Each verify column brings its own experts' misses (~10 ms a
  column against 13.8 ms a plain token): MTP 1/2/3 drafts 71.6/66.8/52.9 tok/s (acceptance
  1.75/2.25/2.68), DFlash2 1/2/4/7/15 65.8/61.3/57.1/35.2/19.2 (1.67-3.19) against plain 72.5 on
  one prompt. `--draft-tokens auto` with DFlash2 decodes without a drafter on this configuration
  (fast_paths.cpp auto_drafts_nothing; the drafter is then not bound); MTP's confidence gate (§12.2)
  later made 2 drafts pay.
- **Prefill is compute-bound, not link-bound.** A 32K chunk (each expert streamed once instead of
  twice) gave 6,892 against 6,517 tok/s (+6 %), so §10's layer walk is worth at most a few per cent
  at 32K. The wide expert route reaches 41/89/110 TFLOP/s at 1K/4K/16K columns (bench
  `--geometry 122b`); under ncu (16K) it is issue-bound (issue slots 50 %, INT tensor pipe ~50 %,
  96 registers, 35 % occupancy): about 125 instructions per 16-element block per warp, of them
  8 MMAs, 32 the int64 epilogue and ~66 the operand decode (LOP3/SHF/MOV/LDSM). §8.6's int32
  epilogue alone barely changes the issue count; it needs the B decode moved out of the consumer
  warps (decoded once per stage) to become MMA-bound.

**Remaining work, in the order the measurements suggest.**

0. Before merging: the A/B against main of the five flagged kernels (27B FP8 prefill residual add,
   Flash-Next wide prefill); a slowdown there moves the scale format into a template parameter
   (BF16 instances then compile exactly as before) and restores the wide kernel's old code.
1. The wide expert route (§8.6 C3 plus the per-stage B decode): prefill +20-30 %.
2. Lending the whole wide prefill workspace from the frames (the MoE buffers are lent already, but
   the GDN/attention buffers at 16,384 columns keep the 1.84 GiB peak): ~370 frames for decode.
3. Decode misses: next-layer link prefetch (route prediction) and the CPU leg's latency; the only
   path found toward the Must decode target. (Measured in §0.1: prefetch and split misses lose,
   because the link is already the critical leg.)
4. Speculation's `auto` with a no-draft choice and miss-aware round costs (§12), once verify rounds
   are cheaper. (§0.1: the gated MTP rule is +5.5 % on sampled requests, -2 % greedy.)
5. Elastic KV, the layer walk, Flash-Next's Program on the shared offload engine (bit-identical),
   the user guide (§17).

### 0.1 Second pass (2026-10-22, measured the same day from 20:25)

Same machine; the artifact reconverted with `--proposal` and `sampling.json` (verified word for word
again: 12,336 experts, 1,178 tensors, 240 input scales; 793 s). Timings are infernix_bench tg256
greedy (one warmup, three repetitions) or the CLI's sampled request (the card's defaults, 256 tokens),
each arm twice in ABBA order; bf16 KV.

**Landed (on by default).**

- **Verification padding mask.** A verification round's columns past a row's drafts route no expert
  (`ops::MoeColumnMask`: no dispatch entry, no miss; y = 0); live columns keep their bits. MTP 2 drafts
  with the confidence gate: 79.6 / 77.5 tok/s with the mask, 47.8 / 47.8 without.
- **Confidence-gated MTP (§12.2)** is MTP's rule with offloaded experts. Sampled requests: MTP 2 drafts
  gated 79.4 tok/s mean of four runs (77.5-80.8) against plain decode's 75.3 (74.1-76.1), +5.5 %, every
  gated run above every plain one; 3 drafts 76.0, a fixed 0.9 probability threshold 74.5 / 73.8.
  Greedy (bench): gated 2 drafts 91.5 against plain 93.5 (-2 %), ungated 82.8, gated 1 draft 93.0. A
  zero-draft MTP round still runs the drafter (its MoE layer misses like a target layer), which the
  rule's baseline does not see: that is the greedy loss. `--lm-head-draft` changes nothing measurable
  (gated 2 drafts 93.2 / 92.5 against 92.8 / 93.2). The greedy bench decodes after a one-token seed
  prompt; on real prompts gated MTP wins greedy too (2026-10-25, §12.2), so `--spec mtp
  --draft-tokens auto` drafts 2 behind the gate.
- **Elastic text KV (§11.1)**: the first 32,768 tokens backed at startup, growth from the expert frames
  in 4,096-token steps, shrink at lane release and after 60 s idle. The MoE real test grows the pool
  512 -> 704 pages for a 40,960-token prompt and back, outputs identical.
- **SSD tier**: with `--expert-ram-mib 40000` (the rest read from the artifact) the reproducible
  route's tokens equal all-in-RAM's; decode 33.2 tok/s against 62.2 (reproducible route). The tier's
  RAM copies measured 4,887 CPU experts/s against ~10,300 for pinned RAM (not investigated).
- **`--lossless` scoring** reads a streamed window's experts from the stream ring: 106 tok/s over four
  4,096-token windows (bf16 KV; no earlier figure at this route). Lossless decode 4.7 tok/s (5.1 in
  the first slot, another prompt).
- **CPU ISA**: AVX-512 needs F, BW, VL and VNNI; the three CPU expert tests pass under Intel SDE 10.13.1
  (Sapphire Rapids) with avx512-vnni exercised, bit-identical to scalar.
- **Memory, sampling defaults, `--lm-head-draft`, whole-pass workspace lending** as coded (§10, §11).

**Measured and rejected (removed; the code is kept in `.local/ornith/removed/` for a later attempt).**

| Change | Plain decode tg256 (base 90.6 / 91.3) | Note |
|---|---:|---|
| Split misses (CPU gate/up + GPU down over link-staged down rows) | 83.2 / 83.2 | also MTP 2 gated 67.7 / 69.0 |
| Decode prefetch, 1 slot + 2 CPU warm hints | 80.0 / 80.0 | 2 slots: 79.4 / 79.1; with split 69.1 / 69.3 |
| Miss split by probe costs instead of the 1/3 divisor | 82.5 / 91.3 | no consistent gain |
| L2 warm chain 64 / 80 MiB (default 40) | 89.9 / 90.5, 89.5 / 89.1 | 40 MiB kept |
| CPU workers 4 / 8 / 10 / 12 (default 6) | 84.9 / 85.2, 90.5 / 90.4, 89.9 / 89.9, 89.9 / 93.1 | 6 kept |
| Wide A16 variant 2 (op bench, 122B geometry) | 1,024 tokens +4.3 %, 4,096 +3.0 %, 16,384 -0.1 % time | pp32768 6,508 against 6,530 |

Why split misses lose: an Nsight Systems trace of decode shows the link, not the CPU, as each layer's
critical leg. The stage kernel (the link-kept misses' copies) runs 252 µs per layer on average against a
58 µs CPU wait after the resident pass, because the 1/3 rule already balances the link (27.5 GB/s) and
the CPU (~53 GB/s); split misses add their down rows to the busy link (stage 428 µs per layer). Decode is
bound by miss bytes over both paths (about 96 misses of 5.3 MB per token at ~80 GB/s combined), plus
~60 µs of dense work and a ~18 µs routing/plan kernel per layer.

**No regression on main's models** (main 1acbc468 against the branch, infernix_bench ABBA, tok/s):

| Model | pp4096 main / branch | tg256 main / branch | MTP 3 tg256 main / branch |
|---|---:|---:|---:|
| Qwen3.8-27B NVFP4 | 13,311, 13,229 / 13,276, 13,268 | 82.20, 82.24 / 82.16, 82.19 | 129.53, 129.42 / 129.51, 129.48 |
| Qwen3.6-35B-A3B | 23,937, 23,985 / 23,972, 23,978 | 395.8, 396.8 / 396.6, 396.8 | 677.6, 678.1 / 678.5, 679.0 |
| Qwen3.8-Flash-Next NVFP4 Dense8 | 1,662, 1,662 / 1,664, 1,664 | 136.5, 139.2 / 138.6, 138.6 | 160.7, 160.2 / 157.8, 160.0 |

Normalized SASS: 5,918 of main's 6,002 `infernix-serve` kernels have an identical twin; the others are
the shared offloaded-MoE kernels (geometry and mask templates) and kernels whose shared-memory or
parameter offsets moved with the same instructions. Flash-Next's one low MTP run (157.8) is within
its ±1.1 spread and the other branch run equals main.
---

## Contents

0. [Status after the first hardware runs](#0-status-after-the-first-hardware-runs-2026-10-22)
1. [Deliverable, fidelity and acceptance](#1-deliverable-fidelity-and-acceptance)
2. [Model and checkpoint facts](#2-model-and-checkpoint-facts)
3. [What vLLM, SGLang and KTransformers do](#3-what-vllm-sglang-and-ktransformers-do)
4. [Performance model and targets](#4-performance-model-and-targets)
5. [Design overview and decisions](#5-design-overview-and-decisions)
6. [Artifact and conversion](#6-artifact-and-conversion)
7. [Ownership](#7-ownership)
8. [Kernels](#8-kernels)
9. [Decode round](#9-decode-round)
10. [Prefill](#10-prefill)
11. [Expert cache, memory and residency](#11-expert-cache-memory-and-residency)
12. [Speculative decoding](#12-speculative-decoding)
13. [Vision and vision offload](#13-vision-and-vision-offload)
14. [Numerics and qualification](#14-numerics-and-qualification)
15. [Implementation phases](#15-implementation-phases)
16. [Risks and open questions](#16-risks-and-open-questions)
17. [Documentation changes](#17-documentation-changes)
18. [Sources](#18-sources)

---

## 1. Deliverable, fidelity and acceptance

### 1.1 Deliverable

`Qwen3_5MoeForCausalLM` at the 122B-A10B geometry through the public Engine, the CLI,
`infernix-serve` and offline CausalScoring, with:

- Text, the MTP layer, and Vision (selected with `--vision`, its tower kept in RAM with
  `--vision-offload`);
- routed experts in pinned host RAM, a VRAM expert cache sharing device memory with the KV cache,
  misses served by the CPU in place and by SM-staged PCIe copies, and the SSD expert tier when RAM
  is short (all inherited from Flash-Next, §7);
- every KV profile (`bf16`, `int8`, `fp8`, `nvfp4`, `k8v4`, `vq2`, `k4v2`);
- concurrency 1-8, the hybrid prefix cache, preemption with Replay, constrained decoding, tool calls
  and reasoning channels, as for the other Qwen3.5 artifacts.

This is not a new architecture: Ornith-1.5-122B is a Qwen3.5-122B-A10B fine-tune with the
unchanged mathematics of [the Qwen3.5 reference](qwen3_5-model.md). The work extends the `qwen3_5`
model to this geometry and gives it offloaded routed experts. The same code then runs
`Qwen/Qwen3.5-122B-A10B` and other instances of it; their checkpoints would need their own recipes.

### 1.2 Fidelity

| Tensor class | Checkpoint form | Stored | Executed as |
|---|---|---|---|
| Routed experts (48 × 256) | ModelOpt NVFP4, `W4A16_NVFP4`: E2M1 codes, E4M3 scale per 16, FP32 `weight_scale_2` | Exact: `nvfp4_mul` banks in `nvfp4_expert_rg16_v1` (multiplier = `weight_scale_2`, no reciprocal) | Canonical W4A16 (Flash-Next design §16.2.1), BF16 activations |
| Shared expert (48) | The same | Exact: one more `nvfp4_expert_rg16_v1` record per layer, device-resident | The same arithmetic, as a resident expert job |
| `lm_head` | NVFP4, `W4A16_NVFP4` | Exact: `nvfp4_mul` in a new dense row-group layout (§6.3) | Canonical W4A16 with FP32 logits |
| GDN `in_proj_qkv`, `in_proj_z`, `out_proj`; attention `q/k/v/o_proj` | FP8 E4M3, FP32 per-tensor `weight_scale`, static FP32 `input_scale` (W8A8) | Exact: new `fp8_e4m3fn_row_f32` (§6.2); `input_scale` as each input's activation divisor | **Prefill: W8A8 static** (E4M3 activations at the checkpoint's input scale, FP8 tensor cores, FP32 accumulation). **Decode and verification: W8A16** (BF16 activations, column-invariant), as every Qwen3.5 artifact's FP8 projections |
| Router, `shared_expert_gate`, GDN `in_proj_a/b`, norms, conv, `A_log`, `dt_bias`, embedding | BF16 | BF16 (direct) | BF16 activations; router and shared-gate logits rounded once to BF16, as the model computes them (main 75029865: `RouterLogits::kBf16` on the offloaded route) |
| Vision tower | BF16 | BF16 (direct) | BF16 |
| MTP layer (dense, 256 experts) | BF16 (`ignore: mtp*`) | **Re-quantized**: `q8_g32_fp16` dense, `q4_g64_fp16` (`grouped_mse`) experts; router and gate BF16 | Acceptance only, never output |

"Exact route" in this plan means the bit-reproducible, request-invariant route
(`--reproducible`), not exact arithmetic. The routes for this artifact are:

| Route | Prefill activations | Decode / verify activations |
|---|---|---|
| Default (qualified fast paths on) | A8 at the checkpoint's scales | BF16, plus any fast path qualified here |
| `--reproducible` | A8 at the checkpoint's scales | BF16 |
| `--lossless` ([definition](lossless-route.md)) | BF16, every operation correctly rounded; weights decoded exactly, input scales unused | the same: prefill and decode compute the same values |
| `--reference` (scoring with `--lossless`; lossless-route.md §3b) | the lossless route with the declared lossy steps computed exactly: the FP8 sites' static A8 (each input divided by its `input_scale`, E4M3, the scale applied in the correctly rounded epilogue) | — (scoring only) |

A route's processing cost is D(lossless, route) − D(lossless, design reference), measured with
`compare-prefill`/`compare-decode` on an `infernix-perplexity --reference` or an
`infernix_decode_quality_forced --reference --observe` run as the base (main ccbc95c2). The
precise reference this plan first extended was retired on main; `--lossless` is the gates'
reference.

The 8-bit activations are the checkpoint's declared arithmetic and the way vLLM and SGLang serve
it (§3). Infernix's Qwen3.5 exact route already runs an artifact's permitted 8-bit and 4-bit
activations in prefill. Its decode and verification parameters are rewritten to 16-bit
column-invariant routes (`src/models/qwen3_5/execution/parameters.cpp`, `make_invariant`). This
artifact follows the same rule (decision D3, §5.2). 8-bit decode activations are a later
fast-path candidate (§14.5).

### 1.3 Baselines

Nothing else runs this checkpoint on one 32 GB card with its own weights:

- **vLLM** has no routing-aware expert offload. Its UVA `--cpu-offload-gb` reads every offloaded
  weight over PCIe on each forward.
- **SGLang's Paged Experts** rejects NVFP4 and speculative decoding.
- **KTransformers' NVFP4 CPU path** folds the scales into BF16 group scales, which is lossy. Its
  layerwise prefill supports FP8/RAWINT4 only.
- Both vLLM and SGLang need versions newer than the model card names: vLLM ≥ 0.23 and SGLang main
  (§3).

The acceptance report therefore compares against:

| Baseline | What it shows |
|---|---|
| llama.cpp `--n-cpu-moe` with a 4-bit GGUF of this model (Ornith's or Unsloth's Qwen3.5-122B-A10B), measured on the target machine | The common single-GPU deployment. Comparison only: it needs its own GGUF (~70 GB) and a llama.cpp build, neither used by the conversion, and the owner's approval when M8 starts |
| KTransformers + SGLang, if it runs this checkpoint on the target machine | The hybrid CPU/GPU reference design. Comparison only: a software install, approved with the GGUF |
| Published, other hardware | vLLM FP8 all-resident on 2× RTX PRO 6000: 106 tok/s decode at 1K context, 81.7 at 256K, TTFT 2.1 s at 32K. KTransformers Qwen3-Next-80B FP8 on 1× RTX 5090 + 2× EPYC 9355 (24 DDR5 channels): 75.9 tok/s decode, 6,233 tok/s prefill at 32K |
| Infernix Flash-Next Dense8 on the same machine | The internal reference for the offload mechanism: README 157 tok/s decode at 8K (MTP auto); 11,211 tok/s pp32768 |

### 1.4 Targets

The targets apply to Text with thinking off, at C = 1 unless stated, in the primary configuration:
`--kv-dtype int8 --max-context 262144`, MTP `--draft-tokens auto`, RTX 5090 on PCIe 5.0 x8,
i9-13900K, 96 GB DDR5-5800. They come from the model in §4. They are acceptance gates set before
any measurement; a missed target is reported, never moved.

| Metric | Must | Stretch |
|---|---:|---:|
| Plain decode, 4K context (warm cache) | ≥ 100 tok/s | ≥ 140 |
| Best speculation (MTP auto + n-gram), mixed chat/code corpus | ≥ 140 tok/s | ≥ 200 |
| Plain decode at 128K actual context, relative to 4K | ≥ 70 % | ≥ 80 % |
| Aggregate decode at C = 8 | ≥ 300 tok/s | ≥ 450 |
| Prefill, fresh 32K prompt | ≥ 6,000 tok/s | ≥ 9,000 |
| Time to first token at 128K | reported | — |
| Reproducible route vs `--lossless`, and the static A8 sites' share of it (vs `--reference`) | reported (§14.4) | — |

The report gives:

- the median and range of at least three runs, and the worst case;
- the expert hit/miss composition and the per-kernel time split from one nsys trace;
- every KV profile after the primary one;
- both baselines on the target machine.

### 1.5 Non-goals

- Tuned A16 prefill routes for the FP8 projections. Prefill runs A8; `--lossless` covers 16-bit
  prefill with exact arithmetic.
- A generic offloading framework for arbitrary models. The offload machinery becomes one
  model-independent component shared by the two Programs that use it (§7).
- Copying vLLM, SGLang or KTransformers code. They were studied for mechanisms and measurements.
- GPUs other than `sm_120a`, multi-GPU, or PCIe x16. The link is x8 by necessity.

---

## 2. Model and checkpoint facts

### 2.1 Geometry and byte budget

| Quantity | Value |
|---|---|
| Hidden H / layers | 3072 / 48 (36 GDN, 12 gated attention at 3, 7, …, 47) |
| Attention | 32 Q heads, 2 KV heads (**GQA group 16**), head dim 256, output gate, partial RoPE 64 of 256, interleaved MRoPE [11, 11, 10], θ = 1e7 |
| GDN | 16 key heads, 64 value heads, dims 128/128, conv width 4 over 12,288 channels, FP32 state |
| MoE | 256 routed experts, top-8, width I = 1024; shared expert width 1024 with a sigmoid gate |
| Vocabulary | 248,320, untied, BF16 embedding |
| MTP | one gated-attention block with its own 256-expert MoE, `fc` [3072, 6144] |
| Vision | 27 blocks, width 1152, MLP 4304, merger to 3072 |

| Item | Size |
|---|---|
| One expert record (gate/up 2048 × 3072 + down 3072 × 1024, E2M1 + E4M3 per 16) | **5,308,416 B = 1,296 × 4 KiB** (gate/up 3,538,944; down 1,769,472) |
| All routed experts (12,288) | **65.23 GB = 60.75 GiB** |
| Routed expert bytes touched per token (48 × 8) | **2.04 GB** (1.54× Flash-Next's 1.33 GB) |
| FP8 projections (GDN 88.1 MB, attention 78.6 MB per layer) | 4,114,612,224 B, equal to the checkpoint's F8_E4M3 total |
| Dense bytes per decoded token (FP8 + NVFP4 head 429 MB + shared experts 255 MB + router 76 MB + GDN a/b 28 MB) | **4.90 GB** |
| Embedding (BF16) | 1.53 GB; one 6 KB row per token |
| GDN state per sequence | 151 MB (FP32), read and written every decode token |
| KV per context token and KV head: `bf16` 1024 B, `int8` 528, `fp8` 516, `k8v4` 402, `nvfp4` 288, `k4v2` 196, `vq2` 132 | Read per decoded token at context L: 24 × per-head bytes × L (`int8`: 1.66 GB at 128K) |
| MTP layer | 5.05 GB in BF16; 1.40 GB as the q8/q4 drafter |

Prefill compute per token:

- 8.2 GFLOP in the FP8 projections;
- 7.25 GFLOP in routed experts;
- 0.9 GFLOP in the shared expert;
- attention at about 0.39 MFLOP per context position. At 16K mean context (a 32K prompt) that is
  6.4 GFLOP per token, equal to each of the two largest terms.

### 2.2 What the checkpoint actually contains (ranged reads, 2026-10-22)

Headers of all eight shards plus every FP32 scale (about 20 MB) and a sample of 127 E4M3
block-scale tensors (28 MB) were read; scripts in `.local/ornith/scripts`.

- **Static input scales are shared where inputs are.** q/k/v carry one `input_scale` in all 12
  attention layers; `in_proj_qkv` and `in_proj_z` share one in all 36 GDN layers. One E4M3 encoding
  of each layer input therefore serves its whole fused projection.
- **Weight scales differ within fused parents**, so each part keeps its own multiplier:
  - only 2 of 156 FP8 `weight_scale` values are powers of two, and 8 are BF16-exact;
  - per-tensor input scales range 0.0025 to 0.189, so a static clip at ±448·s.
  - Neither `fp8_e4m3fn_row_bf16` (BF16 scales) nor a shared parent scale is exact; vLLM and SGLang
    re-quantize the merged shards to their largest scale, which changes weights (§3).
- **NVFP4 global scales:** 163 of 36,864 expert `weight_scale_2` are powers of two; `lm_head`'s
  (7.01e-5) is not. The existing `nvfp4` format (a rounded reciprocal divisor) is therefore not
  exact; `nvfp4_mul` is.
- **Gate and up share `weight_scale_2`** in all 12,336 routed and shared experts.
- **Dead experts.** At least 12 experts have every matrix scale below 1e-20 (smallest 2.06e-37,
  layer 6 expert 188; others at layers 3, 4, 5, 7, 8, 10, 13). Their outputs underflow to zero or
  subnormals. They are imported as stored, and the placement-invariance tests include one (§14.2).
- **Subnormal block scales are common:** 14.8 % of sampled E4M3 block scales are subnormal. vLLM's
  Marlin repack flushes block scales below 2⁻⁶ to zero unless a global factor lifts them; the
  canonical integer arithmetic keeps them exactly.
- **Block-scale exponent spread** (max − min exponent over runs of 16 consecutive blocks of a row):
  - ≤ 3 in 95.4 % of groups, ≤ 4 in 96.3 %, ≤ 5 in 96.8 %;
  - the wide groups concentrate in a few `down_proj` matrices (4 of 127 sampled tensors have
    > 80 % wide groups, the median tensor 0 %).
  - This decides the wide-route epilogue (§8.6).
- **MTP and Vision are BF16;** MTP holds 768 BF16 expert tensors. There are no `k_scale`/`v_scale`
  tensors despite the FP8 `kv_cache_scheme`.

### 2.3 Tokenizer, template and generation defaults

- `tokenizer.json`, `vocab.json`, `merges.txt`, `tokenizer_config.json`, `generation_config.json`
  and the processor configs are byte-identical to `Qwen/Qwen3.5-122B-A10B`'s.
- `chat_template.jinja` differs in one place: Ornith renders `<think>\n…\n</think>\n\n` in **every**
  past assistant turn, while Qwen's renders it only after the last user query. The copy embedded
  in `tokenizer_config.json` is Qwen's.
- The converter packs `chat_template.jinja` (`tools/convert/resources.py`), and the frontend renders
  the artifact's template. Ornith's own template is therefore used without an override, and
  agentic turns keep a stable prefix across turns. That raises prefix-cache reuse when the client
  returns `reasoning_content`; a client that drops it gets empty think blocks.
- Tool calls use the Qwen3.5 XML form (`<tool_call><function=…><parameter=…>`) that the existing
  `tool_call_parser` handles.
- `generation_config.json` sets temperature 0.6, top-k 20, top-p 0.95; the card recommends 1.0 with
  presence penalty 1.5 for general use.
- M2 checks the frontend tests on this template: multi-turn with tools, reasoning, continuation and
  media.

---

## 3. What vLLM, SGLang and KTransformers do

Read on 2026-10-22: vLLM main `0746329d` and tags v0.19.1-v0.32.0; SGLang main `3438ed66` and
v0.5.9; KTransformers main `a694e718`. Nothing was run.

**Version gate.** vLLM v0.19.1, the version the card names, drops `W4A16_NVFP4` layers. Their
support arrived in v0.22 (MoE) and v0.23 (`lm_head`). SGLang 0.5.9 has no
`ModelOptMixedPrecisionConfig`, and on SM120 main crashes at warmup unless
`--moe-runner-backend marlin` is set (PR #45520).

| Mechanism | vLLM / SGLang | This plan |
|---|---|---|
| Fused FP8 shards with different weight scales | Dequantized and re-quantized to the largest scale; input scales maxed. This changes stored weights | Per-row FP32 multipliers, every shard exact (§6.2) |
| Static FP8 activation quantization | `fl32(x · fl32(1/s))`, clamp ±448, RN to E4M3. SGLang runs a separate quant kernel per projection; vLLM fuses it into the norm under Inductor | The same rule, bit for bit (§14.1), in prefill, fused into every producer: norms, gated norm, attention epilogue. Decode keeps BF16 activations (D3) |
| FP8 GEMM, sm_120 | SGLang: warp-per-row streaming GEMV with evict-first loads at M = 1 (cuBLAS reached "only 50-70 % of DRAM bandwidth"); cuBLAS or CUTLASS beyond. vLLM: FlashInfer `bmm_fp8` / CUTLASS | FP8 tensor-core decode and prefill routes at the roofline, one arithmetic class at every width (§8.2) |
| W4A16 NVFP4 MoE, sm_120 | Marlin: dequantize to FP16, gate/up global scale collapsed to gate's, global scale rounded to BF16 (SGLang), subnormal block scales flushed (vLLM) | Canonical exact integer arithmetic, same bits on GPU and CPU (§8.6) |
| Dense W4A16 (shared expert, head) | Marlin FP4 | Canonical arithmetic: shared expert as a resident expert job, head with FP32 logits |
| QK norm + RoPE + gate split | One fused kernel (vLLM `fused_qk_rmsnorm_rope_gate`, SGLang `fused_qk_gemma_rmsnorm_rope_gate`), normed values rounded to BF16 before rotation | Fused, with the Qwen3.5 boundaries (§8.4) |
| GDN decode | One packed kernel: unpack, L2 norm, gating, recurrence (FLA `fused_recurrent_…_packed_decode`); vLLM's MTP variant fuses the gated norm too | One kernel from conv to the quantized out-projection input (§8.3) |
| GDN prefill | FLA chunked (chunk 64); vLLM ≥ 0.30 on CUDA 13 uses a CuTe-DSL SM120 kernel (3.8-4.5× kernel speed) | Infernix's chunked kernel at 64 heads; the CuTe numbers are the bar (§10) |
| Shared expert | Auxiliary stream overlapping router and routed experts | Resident job in the routed kernels (§8.6), no extra stream |
| MTP | BF16 drafter (4.8 GB); SGLang NEXTN 3/1/4; accept length 2.5-3.3 on 122B/397B | q8/q4 drafter (1.4 GB), Infernix `auto` (§12) |
| Prefix cache with GDN state | State at block boundaries (vLLM ≈ 2K-token blocks for this model); SGLang `extra_buffer` every 256 tokens | Existing hybrid prefix cache |
| Expert offload | vLLM: whole-weight UVA only. SGLang Paged Experts: GPU-side LRU, SM float4 gather from pinned host, prefill in waves; rejects NVFP4 and speculation. KTransformers: fixed or frequency placement, CPU experts with **deferral** of the lowest-weight experts to the next layer (inexact, up to 1.45× decode) | Flash-Next's LFRU cache, CPU in-place misses, SM-staged copies, layer-walk prefill (§11, §10). Deferral changes the mathematics and is out of scope, as in the [GLM-5.3](glm5_3-flash-design.md) and [DeepSeek-V4.1](deepseek_v4_1-flash-design.md) designs |

Lessons kept:

- streaming GEMVs reach the DRAM roofline only with evict-first weight loads and enough bytes in
  flight per SM;
- QK norm + RoPE + gate and the whole GDN decode step each belong in one kernel;
- shared-expert work overlaps the routed experts;
- MTP with 3-4 drafts pays on this family.

---

## 4. Performance model and targets

### 4.1 Decode at C = 1

GPU bytes per plain decode token at 8K context, at 1.70 TB/s achievable on the RTX 5090:

| Term | Bytes | Time |
|---|---:|---:|
| FP8 projections | 4.11 GB | 2.42 ms |
| `lm_head` (NVFP4) | 0.43 GB | 0.25 ms |
| Shared experts, router, GDN a/b | 0.36 GB | 0.21 ms |
| GDN state, read + write | 0.30 GB | 0.18 ms |
| KV (`int8`, 8K) | 0.10 GB | 0.06 ms |
| Routed experts hit in VRAM, hit rate h | h × 2.04 GB | h × 1.20 ms |
| **Total at h = 0.9** | **7.1 GB** | **4.2 ms** |

That is a 240 tok/s ceiling with no misses and no launch gaps.

**Misses.** At hit rate h there are 384 (1 − h) misses per token, about 0.8 per layer at h = 0.9.

- A CPU miss reads one 5.31 MB record at ~55 GB/s (six workers, exact W4A16, memory-bound):
  ~100 µs.
- A staged miss copies it at ~24 GB/s: ~220 µs.
- The startup divisor rule d = max(1, ⌊1.65 + C·t⌋) then gives d ≈ 3, so a layer's first misses go
  to the CPU.
- The CPU leg starts after the router, ~70 µs into a ~90 µs layer. A layer with one miss therefore
  waits ~60-80 µs, and 57 % of layers have at least one miss at h = 0.9. That costs ~1.8-2.4 ms per
  token.

**Launch gaps.** Seven kernels per layer (§9.1) with PDL and graphs cost ~0.4-0.8 ms per token.

| Hit rate h | Projected plain decode, 4K-8K |
|---:|---:|
| 0.92 | 135-160 tok/s |
| 0.88 | 115-135 tok/s |
| 0.80 | 85-105 tok/s |

**The hit rate is the dominant unknown.**

- Flash-Next measures 88-93 % with 38 % of its experts cached.
- This model caches about 37 % (~4,600 frames, §11.1), but its 256-expert top-8 router spreads
  tokens over 3.1 % of a layer's experts instead of 2.0 %.
- M0 estimates the curve from Qwen3.6-35B-A3B route traces (same router family) before anything
  else is built (§15).

### 4.2 Speculation, concurrency and context

- **Speculation.** Flash-Next's verify union grows 10 → 27 distinct experts per layer from W = 1
  to 4. With 8 routed experts this model should see roughly 8 → 22. At 75 % acceptance and K ≈ 3,
  rounds amortize the 4.9 GB of dense bytes over ~2.6 tokens. That projects 1.35-1.5× plain
  (150-200 tok/s at h ≈ 0.88-0.92).
- **C = 8.** Eight rows share every dense byte. Misses grow with the union of rows' experts and the
  CPU caps at 16 jobs per call. Aggregate throughput is bounded by the expert traffic, about
  2.04 GB per row-token at the union's hit rate.
- **Long context.** Full attention reads all KV each step: +1.0 ms per token at 128K with `int8`
  (−20 % against 8K), +0.53 ms with `nvfp4`, +0.25 ms with `vq2`. QSA spared Flash-Next this cost.
  The elastic KV cache also takes frames from the expert cache while long contexts are live.
  Hence the 70 % Must for 128K.

### 4.3 Prefill

Per token at a 32K fresh prompt, on the exact route:

| Term | FLOP | Rate | Time |
|---|---:|---:|---:|
| FP8 projections, A8 | 8.2 G | ~400 TFLOP/s (FP8 MMA; BF16 MMA measures 227-243) | 21 µs |
| Routed + shared experts, canonical W4A16 wide route | 8.15 G | 74 TFLOP/s as built (Flash-Next, integer-issue-bound) → **≥ 160 with §8.6's epilogue** | 110 → **51 µs** |
| Attention, 16K mean context, FP16 P×V | 6.4 G | ~200 TFLOP/s | 32 µs |
| GDN chunked, norms, routing, conv | — | — | ~10 µs |
| **Total** | | | **173 → 114 µs: 5.8K → 8.8K tok/s** |

- **Link.** The layer walk streams each non-resident expert once per span (up to 65,536 tokens):
  42 GB at 27.6 GB/s is 1.5 s, or 47 µs per token at 32K. The ring keeps it under compute
  (Flash-Next §19.3.x F2/F4).
- **A8's share.** BF16 projections at the measured 230 TFLOP/s would add ~15 µs per token (−12 %).
  Decode would gain little: the projections are bandwidth-bound either way, and A8 there saves
  only the E4M3 → BF16 widening. That is why decode keeps the 16-bit rule (D3).
- **The expert epilogue is the largest lever:** a 2× faster exact wide route is +50 %.
- **TTFT at 128K.** The mean context is 64K and attention dominates (~130 µs per token), so about
  30 s on the exact route. The `prompt-pv8` fast path, if qualified for this geometry, removes part
  of it.

---

## 5. Design overview and decisions

### 5.1 Placement

```text
 ┌──────────────────────────────── RTX 5090, 32 GB ───────────────────────────────────┐
 │ FP8 projections 4.11 GB · NVFP4 head 0.43 · shared-expert records 0.25 · router    │
 │ MTP drafter 1.40 (q8 dense, q4 experts) · GDN state 151 MB/lane · workspaces       │
 │ ┌──────── frame pool ≈ 4,600 × 5,308,416 B (≈ 22.8 GiB) ────────────────────────┐ │
 │ │ cached experts │ 64 staging slots │ KV pages + loans │ prefill ring │ vision  │ │
 │ └───────────────────────────────────────────────────────────────────────────────┘ │
 └──────▲─────────────────────▲────────────────────────▲──────────────────────────────┘
        │ copy engine         │ SM stage kernel        │ zero-copy
 ┌──────┴─────────────────────┴────────────────────────┴─────────────┐  ┌────────────┐
 │ 96 GB host: 12,288 pinned expert records (60.75 GiB),             │  │ artifact   │
 │ BF16 embedding (1.42 GiB), vision tower (0.84 GiB, offloaded),    │  │ (SSD tier  │
 │ mailboxes, host KV / prefix tiers; CPU expert team (exact W4A16)  │  │ if RAM is  │
 └───────────────────────────────────────────────────────────────────┘  │ short)     │
                                                                        └────────────┘
```

### 5.2 Decisions

| # | Decision | Why |
|---|---|---|
| D1 | Run in `src/models/qwen3_5` (`Qwen3_5MoeForCausalLM`), not a new family | Same mathematics; AGENTS.md: another training instance reuses the architecture's code |
| D2 | Move Flash-Next's model-independent offload machinery into one shared component (`src/runtime/expert_offload/`) that both Programs instantiate; make `offloaded_sparse_moe`'s expert geometry a template parameter with two instantiations | One implementation instead of two; Flash-Next stays bit-identical (§15 M1) |
| D3 | The FP8 projections follow the existing Qwen3.5 exact-route rule: **static A8 in prefill** (the checkpoint's input scales), **A16 column-invariant routes in decode and verification**. Static A8 decode is a fast-path candidate (§14.5), qualified by model-quality §6 | Consistent with every Qwen3.5 artifact. Decode is bandwidth-bound, so A8 there would save only a few per cent while BF16 is more precise. The first draft's A8-decode exception was withdrawn after the owner asked (2026-10-22) why the exact route would not keep BF16 like the other models |
| D4 | A8 rule: `e4m3_rn_satfinite(fl32(x · r))`, `r = fl32(1 / input_scale)` computed once on the host, x the BF16 producer output | Bit-identical to vLLM's, SGLang's and TRT-LLM's activation codes for the same x |
| D5 | Per-row FP32 weight multipliers; the A8 epilogue is the FP8 library's `fl32(fl32(acc · s_in) · m_row)` (the activation scale is the token's, here the static s_in for every token) | Every shard exact; every tuned FP8 kernel serves the format unchanged. (The first draft folded α_row = fl32(m_row · s_in) at load; that needs a second scale plane and a new epilogue in every kernel for no measured gain) |
| D6 | One canonical exact W4A16 arithmetic for routed experts, the shared expert and the head | Placement, width and request invariance for free; no Marlin-style losses |
| D7 | Attention gets a GQA group-16 instantiation of every KV profile's decode/verify/prompt kernels, and the group-16 decode tile is M = 16 rows per KV head per column | A whole MMA tile per KV head at T = 1: the KV stream is read once for 16 heads |
| D8 | Embedding and vision tower host-pinned; vision offload borrows expert frames | 2.3 GiB more VRAM for ~450 frames; Flash-Next's measured mechanism |
| D9 | No fast paths at first. Candidates are listed in §14.5 and qualified per model-quality §6 only after the exact route meets its targets | The exact route is every gate's default run |

---

## 6. Artifact and conversion

### 6.1 Recipe `qwen3_5_moe_nvfp4_modelopt`

- **What it reads.** A new recipe in `tools/convert/official_recipes.py` with the `qwen3_5.py`
  adapter. It reads ModelOpt `MIXED_PRECISION` Qwen3.5 MoE checkpoints with **per-expert** tensors
  (`experts.N.{gate,up,down}_proj.*`); today's adapter reads only fused `gate_up_proj`. It also
  refuses any layer whose `quantized_layers` entry is neither `FP8` nor `W4A16_NVFP4`.
- **Expert import.** It reuses `sources/modelopt.py`'s per-expert bank import (now wired only into
  `qwen4_exp.py`). Without `input_scale` the bank is W4A16, with multiplier = `weight_scale_2`
  exactly; today the ModelOpt branch assumes W4A4 and requires the scale.
- **Checks before writing.** It verifies, and refuses otherwise:
  - gate/up multipliers equal;
  - every fused FP8 parent's parts share one input scale;
  - every global scale finite and positive.
- **Drafter.** MTP is re-quantized by the existing drafter methods: `q8_g32_fp16` dense, and
  `q4_g64_fp16` with `grouped_mse` for the experts (today Qwen4Exp-only). Router and shared gate
  stay BF16.
- **Resources.** Ornith's `chat_template.jinja` and the vision processor configs are packed.
- **Artifact.** About 73.9 GB in `-0000i-of-0000N` files.
- **Conversion cost.** Peak RAM ~16 GB (streamed per shard), about 25 minutes at NVMe speed, plus
  the 77.6 GB source download.

### 6.2 Format `fp8_e4m3fn_row_f32`

- **Definition.** `W[n, k] = e4m3fn(c[n, k]) · m[n]`, with E4M3FN codes in row-major layout and one
  FP32 multiplier per row. The import writes each source matrix's `weight_scale` into its rows'
  words unchanged.
- **Why a new format.** By [tensor formats §3.4](tensor-formats.md#34-fp8_e4m3fn_row_bf16) an FP32
  multiplier is a different format, not a variant of `fp8_e4m3fn_row_bf16`.
- **Parents.** Fused parents simply concatenate rows:
  - GDN [12288 qkv | 8192 z] × 3072;
  - attention [8192 query | 512 k | 8192 output gate | 512 v] × 3072: the adapter takes the query
    and gate rows of each head out of `q_proj` (which interleaves them per head) as two logical
    matrices, as for every Qwen3.5 artifact, and their rows keep `q_proj`'s multiplier.
- **Use auxiliary.** As built, the Use carries a new FP32 role `activation_input_scale = input_scale`
  (a multiplier, unlike NVFP4's `activation_input_divisor`; fused inputs must store bit-equal values).
  The first draft read: `activation_input_divisor = input_scale` (FP32, as NVFP4 A4
  inputs already do) and the `AllowA8` permission. An A8 input with a stored divisor quantizes
  statically, as NVFP4 A4 inputs with divisors do. An `AllowA8` input without the role (a per-channel
  export, ModelOpt `FP8_PER_CHANNEL_PER_TOKEN`: dealignai's Ornith-1.5-122B-UNCENSORED-NVFP4, whose rows
  each carry their own multiplier) encodes each activation column with its own scale
  fl32(max |x| / 448) (`Weight::input_scale` 0); a fused parent's inputs are all static (one scale) or
  all per-token.
- **Producers and oracle.** The only producer is `import_encoded`. The exact decode oracle is
  registered in C++ and Python, and `verify_artifact` compares codes and multipliers word for word.

### 6.3 Head layout `nvfp4_rg16_v1`

- **Layout.** One dense `nvfp4_mul` matrix in the record layout of `nvfp4_expert_rg16_v1`'s matrices:
  16-row groups of 144-byte units (128 code bytes + 16 E4M3 scales), g-major then block-ascending,
  with the FP32 multiplier in the tail.
- **Size.** For [248320, 3072] that is 15,520 row groups × 192 blocks × 144 B = 429.1 MB.
- **Why.** The canonical narrow and limb-MMA routes then read the head exactly as they read an
  expert.

### 6.4 Residency

| Object | Residency |
|---|---|
| Routed expert banks | `host_pinned` (Flash-Next's `RegisteredHostBuffer` load: unbuffered reads, one `cudaHostRegister`) |
| Embedding | `host_pinned`, zero-copy row gather |
| Vision tower | `host_pinned` under `--vision-offload` |
| Everything else (FP8 parents, head, shared-expert records, router, MTP drafter) | `device` |

---

## 7. Ownership

| Component | Owner | Location |
|---|---|---|
| Config acceptance of this geometry, bindings, call order, execution composition | Model | `src/models/qwen3_5/` (`execution/`, `load/`, `program/`) |
| Expert cache policy (LFRU, warm start), residency tables, host and SSD tiers, expert store, memory plan, VRAM monitor, route trace, startup link/CPU calibration, promotion/settle engine | **New shared runtime component**, instantiated per Program (no shared mutable state) | `src/runtime/expert_offload/`, moved from `src/models/qwen4_exp/program/{expert_cache,expert_residency,host_expert_tier,vram_monitor,route_trace,cpu_rates}*`, `expert_store.*`, `memory_plan.*` |
| Router/top-k/residency classification, narrow/wide expert kernels, CPU engine, mailboxes, combine | Op `offloaded_sparse_moe`, geometry as a template parameter: `ExpertGeometry<2560, 640>` and `<3072, 1024>` | `src/ops/offloaded_sparse_moe/` |
| `fp8_e4m3fn_row_f32`, `nvfp4_rg16_v1`, static A8 divisors on Uses | Artifact registries and converter | `src/artifact/`, `tools/convert/`, `tools/artifact/` |
| FP8 A8-static prefill and A16 decode families with fused epilogues; norm+quant; group-16 attention; canonical W4A16 head | Ops | `src/ops/linear/fp8/` (new static route), `src/ops/rmsnorm*`, `src/ops/softmax_attention/`, `src/ops/offloaded_sparse_moe/` (head entry) |
| Program integration: frame pool instance, round hooks, staging, graphs, vision frame loans | qwen3_5 Program | `src/models/qwen3_5/program/` |

**Size of the move.**

- **Moved with namespace changes:** about 8,000 lines.
- **Rewritten for the Qwen3.5 Program:** about 2,050 lines that are `qwen4_exp::ProgramImpl`
  members (CPU/link rates, prefill walk, promotions, loans, split policy, `Forward::moe`).
- **Not moved:** Flash-Next's calibrated expert order is shape-guarded data, and the n-gram, PLE,
  QSA and hyper-connection code stay in `qwen4_exp`.

---

## 8. Kernels

Every entry gives the budget, the mapping and the invariance class. All dense routes keep **one
arithmetic class at every width**: each output column has the same bits for T = 1 and T = 16,384.
Exact-route request invariance needs that; Flash-Next's FP8-row and Q8 seams broke it (§14.3).

### 8.1 RMSNorm + static E4M3 encode

**Not built (2026-10-22).** The static encode is one bandwidth-trivial elementwise pass inside each
A8 projection call (`fp8_a8.cu`, the rule of D4), as every FP8 A8 route already quantizes; prefill
is compute-bound, so a producer fusion saves under 1 % there. Kept below as the fusion candidate if
a prefill profile shows otherwise.

- **What it computes.** `offset_rmsnorm` in FP32, rounded to BF16 (the semantic boundary). In
  prefill it then applies D4's rule with the layer's r.
- **Outputs.**
  - Prefill: the input norm writes E4M3 codes (3 KB per column) for the FP8 parent, and on GDN
    layers BF16 as well for `in_proj_a/b`.
  - Decode and verification: BF16 only (D3).
  - The post-attention norm writes BF16 only, since the MoE takes BF16.
- **Fusion.** In decode, fused with the previous layer's MoE combine and residual add: one kernel
  reads the experts' outputs, the CPU outputs and the residual, and writes the residual and the
  normalized column. In prefill, fused with the residual add.
- **Budget.** Bandwidth trivial; it exists to save a launch and a pass.

### 8.2 FP8 projections (`fp8_e4m3fn_row_f32`): A8 prefill, A16 decode

There are only three shapes: [20480, 3072] (GDN q/k/v/z), [17408, 3072] (attention q/gate/k/v) and
[3072, 8192] (both output projections, with residual add). Each has two route families over the
same stored weights.

**As built (2026-10-22).** The three shapes run the FP8 library's tuned families (`src/ops/linear/
fp8/`), which now read FP32 or BF16 row multipliers (`Fp8RowScales`): the column-invariant sliced-K
MMA for decode and verification, GEMV/SIMT/sliced-K/MMA A16 tiles, and A8 MMA/TMA/split-K tiles
for prefill (`fp8_attn_input_n17408_k3072.cu`, `fp8_gdn_input_n20480_k3072.cu`,
`fp8_linear_add_n3072_k8192.cu`), with the 27B parents' width bands as the starting tables. The
persistent decode mapping and partitioned K below are the plan if the M0 sweep shows these miss
their targets; the GDN conv runs on the materialized projection (`gdn_projected_conv` at 12,288
channels).

- **Prefill (A8).** `mma.sync` m16n8k32 E4M3 × E4M3 → FP32: the weight is A (16 rows × 32 K),
  activations B (32 K × 8 columns). Epilogue `fl32(acc · α_row)`, α_row = fl32(m_row · s_in).
- **Decode and verification (A16, column-invariant).** E4M3 weights widened exactly to BF16 in
  registers, BF16 m16n8k16 MMAs against the BF16 activations, FP32 accumulation, epilogue
  `fl32(acc · m_row)`.
- **Fixed K order.** The K order is fixed per shape and family: K is cut into P fixed partitions
  (P = 1 for K = 3072, P = 4 for K = 8192). Each partition is one ascending MMA chain, and
  partitions are summed in a fixed tree.
  - Within each family every width runs this order. Decode/verify columns therefore have the same
    bits at every round width (up to 64 verify columns), and prefill columns at every chunk width:
    the MXFP8 TMA route's rule (Flash-Next design §6.1.2).
  - Prefill and decode differ from each other, which the exact-route contract allows: a prompt
    token need not equal the same token computed by decode.
- **Decode, T = 1-16.**
  - **Mapping.** Persistent CTAs, 2 per SM in one wave (decode-tile lesson: "a second wave of even
    2 CTAs costs 25-30 %"). Each CTA takes 16-row tiles from a work list. Weights stream through a
    4-stage cp.async ring with an L2 evict-first policy; the BF16 activations (≤ 16 × 8192 × 2 B)
    are staged once per CTA.
  - **ALU.** The widening is one `cvt` per two weights, well under the stream rate. Decode costs
    the same as an A8 decode route would, within the few per cent §4.3 puts on it.
  - **K = 8192.** 192 row tiles × 4 partitions = 768 work items on the persistent grid, since 192
    whole-row tiles would leave 22 SMs doing two. Each partition's FP32 partial goes to global
    memory, and the last finisher of a row tile (an atomic counter) sums the four in the fixed
    order (p0 + p1) + (p2 + p3), so the bits do not depend on scheduling.
  - **Target.** ≥ 92 % of DRAM bandwidth: 38 µs for [20480, 3072] at T = 1.
- **Prefill.**
  - **Mapping.** TMA 128 × 128 tiles, 3-4 stages, warp-specialized producer (proxy fence before
    each stage release, op-development §4.3), same partitions and chains.
  - **Target.** ≥ 400 TFLOP/s, about twice the exact BF16 MMA route's 227-243.
- **Epilogues.**
  - **GDN parent.** Writes q/k/v straight into the convolution's input layout, and z and the BF16
    a/b (a mixed-format second operand computed in the same launch from the BF16 x).
  - **Attention parent.** Writes q, gate, k, v for §8.4.
  - **Output projections.** Add the residual (`linear_add`).

### 8.3 GDN at 64 value heads

- **Decode: one kernel per layer.** A cluster of 4 CTAs per value head, each owning a 32-wide v
  slice of the head's state: 256 CTAs. The gated norm's sum of squares over the head's 128 outputs
  is reduced through distributed shared memory, in a fixed order. In one pass it does:
  - convolution update + SiLU from the BF16 projection outputs and the 3-column conv state;
  - q/k L2 norm and gating (`g`, `β` in FP32);
  - the recurrence over the head's 64 KB FP32 state slice (read once, written once);
  - `plain_rmsnorm(o) · SiLU(z)`, rounded to BF16: the A16 `out_proj` input. The prefill chunk
    kernel's epilogue encodes E4M3 at `out_proj`'s r instead.
- **Decode budget.** 8 MB of state traffic per layer: 4.9 µs at the roofline.
- **Verify.** Verification writes ReplaySSM records instead of state (the existing contract).
- **Prefill.** The existing chunked kernel family, retuned for 64 heads. vLLM's CuTe-DSL SM120
  chunk kernel (3.8-4.5× FLA) is the speed bar.

### 8.4 Gated attention, GQA group 16

- **Projection epilogue / small fused kernel.** q/k `offset_rmsnorm` in FP32 → BF16 (the boundary
  both reference runtimes keep), partial interleaved MRoPE on 64 of 256 dims, then KV append in the
  configured profile.
- **Decode/verify.**
  - **Mapping.** One CTA per (KV head, split): M = 16 × T query rows, so T = 1 is exactly one m16
    tile and each KV byte serves 16 heads.
  - **Splits.** Split-K over the context, with the split count from the context **capacity**
    envelope (the exact route's rule, model-quality §6.1) and the merge in a fixed order.
  - **Merge epilogue.** Applies `· sigmoid(gate)` and writes BF16 for the A16 `o_proj`. The prompt
    kernels' epilogue encodes E4M3 at `o_proj`'s r instead.
  - **Existing kernels at group 16.** Every quantized grouped kernel caps a CTA at 64 rows (4
    columns at group 16), so widths 5-8 move to the parallel form with a 4-column tile. The
    int8-pipelined kernel admits ≤ 3 columns, vq2/k4v2 need their row-count switch, and the
    nvfp4/k8v4 parallel `query_tile` cases are re-cut.
- **Prompt kernels.** Already generic in group (one CTA per Q head); they need instantiation only.
- **Size of the change.** About 45 host-side dispatch/plan sites and about 350-400 new template
  instantiations across the seven profiles. These are counted, not guessed, from today's 16/2
  cross-product.
- **Budget.** Decode at 128K `int8` reads 1.66 GB per token: ≥ 90 % of DRAM bandwidth is the
  target (1.08 ms).

### 8.5 Router and routing

- **Router.** The `projection_fp32` route ([257, 3072] BF16 → FP32 logits) fused with the
  post-attention norm's output. `kMaxK = 3072` is exactly its limit.
- **Routing.** The Op's route/dispatch kernel already computes Qwen3.5's routing: softmax over
  256, top-8 by repeated argmax with lower id winning ties, renormalized, sigmoid shared gate. It
  runs top-8 instead of top-10, with the CPU job plan fused for calls ≤ 32 columns.

### 8.6 Experts (canonical W4A16 at H = 3072, I = 1024)

**Arithmetic.** Flash-Next design §16.2.1 unchanged, with K = 3072 (x) and K = 1024 (h). The
int64 bound holds: 192 blocks give |S| < 2⁵⁵. The record is `nvfp4_expert_rg16_v1` at this
geometry (5,308,416 B).

**Narrow route (≤ 8 columns per expert; decode and verify).**

- **Gate/up.** The current kernel stages a whole 16-intermediate slice (55,296 B here). That
  breaks its two-CTAs-per-SM assert, and the W4A16 variant would need ~92.5 KB.
- **Fix.** Stream units through a 4-stage cp.async ring (~24 KB of shared memory, 3 CTAs per SM),
  with the 9 jobs × 64 slices = 576 slice tasks pulled from a work counter by a persistent grid.
- **Shared expert.** Its record is the ninth job, weighted by `sigmoid(gate)`.
- **Down.** A PDL dependent of gate/up, 16-row tiles of 9,216 B, persistent with a work counter.
- **Budget.** 9 × 5.31 MB = 47.8 MB per layer at h = 1: 28 µs.

**Wide route (prefill, > 8 columns).** It keeps the exact limb MMAs: two m16n8k32 s8 per block
(X = 2¹⁴·H1 + 2¹¹·H0 + 8·L1 + L0). It replaces the per-block int64 epilogue, which leaves the
Flash-Next kernel integer-issue-bound at 74 TFLOP/s.

- **Per-tile multipliers.** For each (row, group of 16 blocks) the kernel decodes the 16 E4M3
  scales once per tile into integer multipliers M_b = s_b · 2^(q_b − q_min) in shared memory.
- **Exact int32 accumulation.** The two limb sums are accumulated separately in int32:
  `acc_hi += first_b · M_b`, `acc_lo += second_b · M_b`.
- **Why it is exact.** |first| < 2^17.6 and |second| < 2^18.6, so 16 blocks fit int32 when the
  group's exponent spread is ≤ 4.
- **Flush.** After each group, `S += (2048 · acc_hi + acc_lo) · 2^q_min` in int64.
- **Fallback.** Groups with a wider spread take today's per-block int64 path. Both paths compute
  the same exact integer, so the bits are identical.
- **Selection (revised 2026-10-22).** A per-matrix flag cannot work: the record has no tail (5,308,416
  bytes, a multiple of 4 KiB, in the `nvfp4_expert_rg16_v1` layout that Flash-Next's published
  artifacts share), and with 96.3 % of sampled groups qualifying (§2.2) almost no matrix has every
  one of its ~24,000 (row, group) pairs qualify. The test runs in the kernel instead, per row and
  stage of 8 blocks: q_base = max q_b − 5 keeps both limb accumulators in int32 (8 · 2^18.6 · 15 ·
  2^5 < 2^31). Each warp decodes its 16 rows' M_b and q_base for the stage into a warp-local table.
  A block below q_base (with a nonzero scale) is folded into the int64 sum directly, a rare,
  divergent case.
- **Registers.** In the current schedule (16 consumer warps of 32 × 16, one 544-thread CTA per SM:
  at most ~120 registers per thread) the two int32 accumulators and the separate high and low MMA
  outputs add 40-60 registers to the int64 sums' 32, so the variant needs a re-tiled schedule (fewer
  columns per thread, or the sums in shared memory). The schedule is chosen with ncu on the first
  build, not before (PROGRESS C3).
- **Instruction count.** Per element and block it drops from ~6 instructions to 2 IMADs plus an
  amortized flush and table read (~3.2), making the kernel MMA-bound. Target ≥ 160 TFLOP/s,
  verified with ncu on the first build.
- **Measured instead (2026-10-22, §0).** The built kernel spends ~2 instructions per element and
  block on the int64 fold (SHL + IMAD.WIDE) and ~66 of ~125 per block and warp on operand decode;
  the int32 epilogue would not lower the first. A variant 2 attacked the second (the two column
  warps of a row group sharing one decode of each stage's B units in shared memory behind a pair
  barrier, one ldmatrix.x4 for the high and low limbs, independent high and low MMAs; about 78
  instead of ~125 instructions per block and warp, the same bits). Measured slower (§0.1: +3-4 %
  time at 1,024-4,096 tokens, equal at 16,384; pp32768 -0.3 %), it was removed: the pair barriers
  and the shared-memory round trip cost what the saved decode instructions gained.

**Head.** The same canonical arithmetic writes FP32 logits, `fl32(S) · 2^(emax − 157) ⊗ m`.

- dp4a GEMV at T ≤ 8;
- limb MMAs (with the same epilogue) at T > 8;
- 429 MB at ≥ 92 % of bandwidth: 0.26 ms.
- Greedy rounds fuse the argmax (the existing `linear_topk` contract).

### 8.7 CPU expert engine

- **Code.** The Op's W4A16 engine at the new geometry: AVX-VNNI `vpdpbusd` (256-bit) on the
  i9-13900K, AVX-512 VNNI where present (tested under Intel SDE). Records stream once per miss;
  three byte products per weight (Flash-Next §16.2.1).
- **Workers.** Six by default, dynamic handout, worker 0 the miss service. Flash-Next measured
  that more workers lose.
- **Budget.** Memory-bound at ~55 GB/s: ~100 µs per 5.31 MB record at T = 1.
- **Startup calibration.** The engine measures its rate per column count for the divisor and the
  CPU cap.

### 8.8 MTP drafter

- **Dense.** The q8 dense projections at the 122B shapes: attention [17408, 3072] and [3072, 8192],
  `fc` [3072, 6144], shared expert. They use the dedicated column-invariant Q8 routes, which need
  registering at these shapes.
- **Experts.** The q4 experts run on `resident_moe` (runtime-shaped up to H 4096 / I 2048; tuning
  for 3072/1024 is unmeasured).
- **Attention.** Uses the group-16 kernels.

### 8.9 Vision

- **Kernels.** The Qwen3.5 tower kernels at 1152/4304 already run Flash-Next's BF16 tower. Only the
  merger `fc2` [3072, 4608] is a new BF16 shape.
- **Offload.** Weights stream per encode window from pinned host RAM (§13).

---

## 9. Decode round

### 9.1 Per-layer sequence (C = 1, plain, 8K, h = 1)

| # | Kernel | GDN layer | Attention layer |
|---|---|---:|---:|
| K1 | previous MoE combine + CPU outputs + residual + input norm (BF16) | 3 µs | 3 µs |
| K2 | FP8 input projection, A16 (+ a/b) | 38 µs | 32 µs |
| K3 | GDN decode step → BF16 / QK norm + RoPE + KV append | 6 µs | 3 µs |
| K3b | — / group-16 attention + gate | — | 7 µs |
| K4 | FP8 output projection, A16, + residual | 15 µs | 15 µs |
| K5 | post-attention norm + router + route/dispatch + CPU plan | 4 µs | 4 µs |
| K6 | expert gate/up (routed hits + shared) | 19 µs | 19 µs |
| K7 | expert down (PDL) | 9 µs | 9 µs |
| **Layer** | | **~94 µs** | **~92 µs** |

That is 4.5 ms for 48 layers at the §8 budgets, plus the head (0.26 ms) and sampling. **Launches are seven per layer
and every kernel is a PDL dependent**, so its prologue (weight prefetch into the ring) overlaps
its predecessor's tail. The CPU wait sits in K1 of the next layer: its spare CTAs prefetch that
layer's projection weights into L2, which measured +1.6 % on Flash-Next.

### 9.2 Round mechanics, inherited from Flash-Next

These are as built (Flash-Next design §19.2-19.3); each was measured there and nothing here is
re-tried.

- **Graphs.** One CUDA graph per batch size, and per (B, W) for verify rounds. CPU requests carry
  device-counted sequence numbers, and per-round tables go up with `upload_pinned` outside the
  graph (+8.7 % MTP on Flash-Next).
- **Misses.**
  - **Staged.** A 16-CTA SM stage kernel reads non-resident records zero-copy into 64 staging slots
    sized by width (+7.2 %); a fork stream computes the staged pass beside the resident one.
  - **CPU.** The CPU takes `min(cap 16, M − ⌊M/d⌋)` misses, fewest columns first.
  - **S4 landing.** Staged misses land in reserved free frames, up to 16 per layer (+8.6 % plain,
    +48.7 % MTP cold).
- **Admission and eviction.** Global LFRU f/(now − last + 1), promotion on CPU-served misses (one
  per layer per token while frames fill, then every 4th, scaled by the link), a prompt-route
  credit of 256, warm start from saved state. Experts used only by rejected drafts earn nothing.
  The policy runs synchronously at commit (+1.4 % / +3.7 %).
- **Shared expert.** It runs inside the routed kernels as a resident job (§8.6). Flash-Next needed
  a side stream because its shared expert was a dense projection.
- **Rejected on Flash-Next, not re-tried without new evidence:**
  - next-layer prefetch (recall 65 % on a saturated link);
  - more than six CPU workers, and the CPU taking every miss;
  - deferral;
  - victim landing, adaptive promotion schedules;
  - megakernel fronts;
  - graph splitting;
  - tail skip as a default (it failed the gate; it stays an approximate opt-in).

---

## 10. Prefill

- **Layer walk.** Spans of ≥ 256-column calls up to 65,536 tokens, run layer-major so experts
  stream once per span (Flash-Next F4: 1.85-4.64×).
- **Ring streaming.** Non-resident experts are copied by the copy engine two layers ahead into ring
  halves lent from frames, with the ring sized to the worst layer (F2: +39-46 %, worst-layer sizing
  +9.2 %).
- **Chunk.** Flash-Next's 16,384 is the starting default. Offloaded experts make small chunks
  link-bound there: pp32768 measured 5,567 tok/s at 1,024 against 11,211 at 16,384. The resident
  Qwen3.5 artifacts keep 1,024 (no gain past it on the 27B), and M8 measures both for this model.
  The exact route's bits follow the chunk grid on Qwen3.5, as today (Flash-Next design §19.3,
  "The prefill chunk"). The A8 projections, the canonical experts and the head add no seam of
  their own (§8.2, §8.6).
- **CPU assist.** For short prompts, capped at 1,536 × 27.5 GB/s / link columns (Flash-Next P7).
- **Prefill never promotes.** The prompt's routes credit the cache instead.
- **Kernels.** The A8 TMA projections (§8.2), the exact wide experts with the int32-group epilogue
  (§8.6), the group-16 prompt attention with FP16 P×V, and the GDN chunk kernel at 64 heads.

---

## 11. Expert cache, memory and residency

### 11.1 VRAM (32,607 MiB card, C = 1, MTP, `int8` KV, 262K max context)

| Term | MiB |
|---|---:|
| In use before loading (CUDA/WDDM, measured on Flash-Next) | 547 |
| FP8 projections, head, shared-expert records, router, a/b, norms | 4,675 |
| MTP drafter | 1,335 |
| KV base (first 32K tokens; grows from the expert cache) | 430 |
| GDN state + conv state + io | 160 |
| Workspaces (prefill chunk 16K) | ~1,240 |
| Staging slots (64 × 5.06 MiB) | 324 |
| Expert tables | 4 |
| Graph reserve | 256 |
| Headroom (no display) | 256 |
| **Expert frames** | **~23,380 → ~4,600 frames (37.5 % of 12,288)** |

- **Without MTP.** About 4,880 frames.
- **Workspaces.** The 1,240 MiB is Flash-Next's ledger figure; M4 re-derives it for this model.
- **At 256K actual context.** The KV grows by ~3.0 GiB and takes ~600 frames while live (the
  elastic pool, §0.1; before it, the whole `--max-context × --max-concurrency` KV was allocated at
  startup).
- **Inherited from Flash-Next.** One `plan_device` ledger function, the spill guard, the VMM frame
  pool that follows free VRAM, and `--vram-past-budget` (~120 more frames here).
- **Second frame pool.** Flash-Next keeps one frame pool per record stride. Here the shared-expert
  and drafter banks are resident outside it, so there is one stride.

### 11.2 Host RAM (96 GB; Windows pins up to 76.0 GiB here)

| Term | GiB |
|---|---:|
| Expert records (pinned) | 60.75 |
| Embedding (pinned) | 1.42 |
| Vision tower (pinned, `--vision-offload`) | 0.84 |
| Mailboxes, load staging, lock overhead (0.25 %) | ~0.6 |
| `infernix-serve` prefix-cache host tier (default) | 4 |
| Headroom (`--ram-headroom-mib`) | 2 |
| **Available RAM needed** | **~66 (CLI) / ~70 (serve)** |

That is the same class as Flash-Next's 69.3 / 73.3 GiB. The pinned total (~63.6 GiB) is under the
76 GiB ceiling, so no record is unpinned. With less RAM the SSD tier serves the rest from the
artifact in place.

---

## 12. Speculative decoding

- **MTP.** The plan was Flash-Next's `auto` policy: maximize E[tokens]/(1 + cK) with EWMA
  acceptance, a per-width cost line, a probe every 8th round, and draft cuts below p = 0.5 on
  one-row rounds. As built, the adaptive policies are fast paths not yet qualified here, so a fixed
  draft count runs, and `auto` decodes without a drafter (§0); §12.2 is the gated replacement.
- **Miss cost.** The draft cost includes the verify union's misses: the expected miss count from
  the union's cache state is an input of the cost line, as Flash-Next's.
- **N-gram drafts.** `--ngram-draft-tokens` (prompt lookup) as on the other Qwen3.5 artifacts:
  agentic code edits repeat their context.

### 12.1 DFlash2 (owner request 2026-10-22)

The artifact also carries [incoai/Ornith-1.5-122B-DFlash2](https://huggingface.co/incoai/Ornith-1.5-122B-DFlash2)
(revision `9baefc1b`), selected at startup instead of MTP (one speculative backend per Engine).

- **Drafter.** 6 sliding-window (2,048) layers, hidden 3072, 32 Q / 8 KV heads of 128, MLP 9216,
  two-tap dynamic convolutions (groups of 16), a rank-256 top-16 candidate selector, mask token
  248077, conditioning taps at target layers 1, 7, 14, 20, 26, 32, 39, 45, block 16. Its card
  reports acceptance lengths of 5.4-9.2 at 15 drafts.
- **Stored** as the official 27B artifact's drafter (`_dflash_drafters`): `q8_g32_fp16` projections
  (0.80 GB), BF16 kernel and selector projections, norms, base kernels and codebooks (0.26 GB).
  The mask row comes from the target's exact BF16 embedding (the card's `mask_embedding.safetensors`
  is only needed by GGUF targets).
- **Kernels at this geometry.** The drafter attention is the 32/8/128 sliding-window family the
  27B drafter runs. New work: `context_kv_materialize` at hidden 3072 (registered for 5120/2560
  only), the Q8 column-invariant routes at the drafter's shapes, the selector's codebook top-k over
  248,320 rows at rank 256, and the eight 3072-wide taps.
- **The offload changes the economics.** A 16-column verify round routes up to 128 expert slots
  per layer, so the union is a large share of a layer's 256 experts and most misses are staged.
  The `auto` policy (the 27B/35B DFlash2 `auto`: split rounds, n-gram copies, trees) therefore
  takes the measured round time per verify width, misses included, as its cost line, as
  Flash-Next's MTP `auto` does. It cuts to fewer drafts or to plain decode when acceptance does
  not pay for the misses, and it is re-fitted by the startup calibration and the running EWMA.
  The miss expectation for a width comes from the union's residency, as in §12.

### 12.2 Confidence-gated MTP and masked padding (measured 2026-10-22, §0.1)

The first runs (§0) priced a verified draft column at ~10 ms of misses against a 13.8 ms plain
token, so a draft pays only when it is very likely accepted (break-even acceptance ≈ 0.7 at one
draft, higher for later ones), and an MTP round averaged 0.75. Two changes target that:

- **Padding routes nothing.** A round verifies at its family's width; columns past a row's extent
  (budget, n-gram copies, the gate below) used to be routed like real tokens and brought their own
  misses. The offloaded MoE now takes the round's valid-column counts (`ops::MoeColumnMask`, the
  verification's `target_valid_columns` read at execution): a masked column has no dispatch entry,
  combines to 0. Live columns' bits are unchanged, so the exact route's
  invariance and the committed text are unaffected.
- **Confidence gate.** Each MTP proposal step also writes the drafter's probability of its draft
  (`ops::argmax_with_probability`: the argmax's own pass over the proposal head's rows also sums the
  softmax denominator in one deterministic order, into the round's egress). Before the next round, each row keeps the prefix that maximizes expected committed
  tokens per unit of round time, `(1 + Σ_j Π_{i≤j} a(q_i)) / (1 + c·e)` (`MtpConfidencePolicy`):
  a(q) is the acceptance learned per probability bucket (prior: the bucket's midpoint), c the cost of
  one verified draft column relative to a round without drafts, fitted from consecutive rounds'
  start-to-start times against their verified columns (prior 0.7 with offloaded experts). An
  extent-0 round costs about a plain round plus the MTP head; the gate reads only the drafter's
  probabilities and earlier rounds, so sampling and acceptance are unchanged.
- **Measured (§0.1).** The mask is on always and the gate is MTP's rule with offloaded experts: 2
  drafts gated +5.5 % over plain decode on the card's sampled requests (79.4 against 75.3 tok/s),
  -2 % greedy (91.5 against 93.5; ungated 82.8). The greedy loss is the drafter's own round cost
  (its MoE layer's misses), which a zero-draft round pays and the rule's baseline does not count;
  skipping the drafter in rounds the rule expects to leave empty would need the MTP layer's KV
  appended without its MoE and head, a separate round kind.
- **`auto` (2026-10-25, Ornith-1.5-122B, INT8 KV, one request, `infernix_bench -pg 1280,256` on
  three bench-corpus prompts, ABBA, four pairs each greedy and at temperature 1.0).** Against plain
  decode: gated 2 drafts a geometric mean 14.0 % faster over the eight pairs (greedy +11 to +26 %,
  sampled -1.1 to +25 %), gated 1 draft 8.7 % (worst -4.3 %), adaptive-mtp forced at a ceiling of
  2 13.9 % (worst +0.0 %, greedy worst +1.2 %) and of 3 13.3 % (worst -7.9 %); the control against
  itself +0.2 to +1.1 %. The greedy loss above came from the one-token seed prompt, whose text
  accepts fewer drafts. MTP's `auto` drafts 2 behind the gate (`auto_fixed_draft_tokens`; the tie
  does not justify registering adaptive-mtp here); the drafter takes 266 expert frames (1.3 GiB).
  On `infernix_decode_quality_gen`'s workload (16 slices of 1,280 tokens of Infernix's code, greedy,
  and docs, sampled; 384 tokens each; decode tok/s) at one request gated MTP 2 decodes 82.5 / 72.2
  tok/s, `adaptive-speculation` beside it -0.5 %, DFlash2's adaptive-dflash (forced) -0.3 % / -3.1 %
  (with `--lm-head-draft` +1.0 % / -1.3 %). With two requests on docs every drafter loses to plain
  decode (47.6 tok/s per request): gated MTP 2 about -6 %, adaptive-dflash -15.6 %, its batched
  rounds plain at one draft -18.3 %; each row's verify columns bring their own misses, and a
  draft-free MTP round still runs the drafter. Over one and two requests MTP's 2 drafts gain about
  3.6 % and adaptive-dflash loses about 1.6 % (eight corpus pairs at one request: +13.5 %), so
  DFlash2's `auto` decodes without its drafter. Skipping the drafter in batched rounds (a round kind
  that appends the MTP layer's KV without its MoE and head) is the open item.
  DFlash2 with the forced adaptive policy and `auto`: 68.8 / 69.1 tok/s sampled, below plain.
- **The drafter's round cost (candidates §11 OR-09, unmeasured; three output-neutral switches, read in
  `models/qwen3_5/program/planning/startup.h`).** The drafter's MoE is resident here (`resident_moe`),
  but each round still runs the alignment over every verify column, then each draft step's layer and
  proposal head:
  - `INFERNIX_TMP_QWEN35_MTP_PAD_MASK`: the alignment's padding columns (past a row's licensed tokens)
    route no resident expert.
  - `INFERNIX_TMP_QWEN35_MTP_STEP_GATE`: Finish is captured as a step chain (the alignment, then each draft
    behind a device gate), whose gates stop where no later draft could be verified. The threshold
    (`MtpConfidencePolicy::step_threshold`) is the lowest bucket edge below which every bucket's
    acceptance times the best bucket's is at most the column cost; a row's drafts end at its first
    one below it.
  - `INFERNIX_TMP_QWEN35_MTP_SKIP_ROUNDS` (with the step chain): `MtpDraftRegime` runs rounds without
    drafting (the alignment's KV only) while their measured tokens per second beat drafting rounds',
    probing the other regime 4 rounds in 32; the release log reports "MTP draft-free rounds: X of Y".
    The aim is greedy decode at plain speed without unloading the drafter, and the sampled gain kept.
- **DFlash2.** The adaptive-dflash policy already chooses each round's verify width from the
  block's per-draft probabilities after a split proposal (a host wait that costs nothing next to a
  14 ms round). It is not qualified here, so the A/B forces it (`adaptive-speculation`,
  `adaptive-dflash`); with offloaded experts it may now also choose width 1 (no draft verified: the
  narrowest family with every row masked), and the offloaded-expert configuration's costs price a draft
  column at 0.72 and the proposal at 0.12 of a plain round (from the fixed-length runs).

---

## 13. Vision and vision offload

- **The tower.** Qwen3.5's, unchanged except for the merger output width. Vision is selected at
  startup, and `--vision-offload` keeps its 0.84 GiB in pinned RAM.
- **Frame loans.** Today the Qwen3.5 overlay evicts weights to make room (`EvictableWeightPool`).
  With an expert cache it instead **borrows expert frames** for the encode window and the output
  handoff, as `qwen4_exp/program/vision_program.cpp` does: `lend`/`give_back` with
  lowest-summed-score runs. That is about 360 lines to port into the shared component's loan API.
- **Ordering.** The evictable ladder stays for artifacts without an expert cache. Vision encode
  windows run between rounds, so a loan never overlaps an expert kernel's frame reads (the
  existing epoch rule).

---

## 14. Numerics and qualification

### 14.1 Arithmetic definitions

| Site | Definition |
|---|---|
| A8 encode (prefill) | `c = e4m3_rn_satfinite(fl32(x · r))`, x the BF16 producer output, r = fl32(1/s_in) |
| FP8 projection, prefill (A8) | `y = bf16_rn(fl32(acc · α_row))`, acc the fixed-partition FP32 chain of E4M3 MMAs (§8.2), α_row = fl32(m_row · s_in) |
| FP8 projection, decode and verification (A16) | `y = bf16_rn(fl32(acc · m_row))`, acc the fixed-partition FP32 chain of BF16 MMAs over the exactly widened weights |
| Routed / shared expert | Flash-Next §16.2.1 (exact integer sums, one rounding per output) |
| Head | the same, FP32 output |
| Router | FP32 logits (`projection_fp32`), softmax/top-8/renormalize in FP32 |
| MoE combine | The existing Qwen3.5 combine order and precision, unchanged (the Op's combine kernel reproduces it; M4 checks it against the `sparse_moe` reference at the 35B shape) |

### 14.2 Op qualification ([op development](op-development.md))

- **Oracles.**
  - **FP8 projections (A8 prefill, A16 decode).** Naive FP64 over decoded codes and multipliers. The exact E4M3 encode is
    checked against a reference encoder on every finite BF16 value × 8 scales, including
    saturation.
  - **Experts and head.** Exact integer oracles; the block-scale sample covers subnormal scales,
    spreads > 4 (the fallback) and a dead expert.
  - **Attention group 16.** FP64 per KV profile at T = 1, 4, 8, 16 and verify windows.
  - **GDN at 64 heads.** FP64.
- **Width invariance.** Every route seam is compared bit for bit: T = 1…16 against prefill widths.
  Placement invariance (frame vs staging vs CPU, every ISA) is checked on real records, including
  the dead expert.
- **The FP8 MMA accumulator.** The sm_120 FP8 tensor-core accumulation is characterized against
  FP64 at the three shapes in M0. The error enters the op tolerance; determinism and invariance do
  not depend on it.

### 14.3 Exact route at model level

- **Determinism and request invariance.** The forced-decode audit at 1, 4 and 8 requests, under
  MTP, from a warm prefix, under KV pressure, and with misses forced to the CPU and to staging,
  must give identical tokens and log-probabilities.
- **Coverage.** Every KV profile.
- **Real tests.** The qwen3_5 prefix-cache, preemption, Replay and constrained-decoding real tests
  pass on the artifact.
- **Lessons applied up front.** Proxy fences on every TMA producer; the shared expert inside the
  routed kernel (no fork-stream race); one route family per shape; the split count from the
  capacity envelope.

### 14.4 Quality

- **Lossless route.** Extended to this geometry: `lossless_linear` reads `fp8_e4m3fn_row_f32` (the
  exact sum times the row's FP32 multiplier, sign included, rounded once) and the dense
  `nvfp4_rg16_v1` head (the record arithmetic with the matrix's multiplier); the offloaded MoE's
  lossless pass reads the bank's geometry at run time (3072/1024 records); the router is
  `lossless_router` on the BF16 logits; the shared expert is its record through the same pass;
  attention at GQA group 16 and GDN at 64 value heads through the lossless attention and gated
  delta rule. Each kernel matches its exact oracle at these shapes.
- **Design reference.** The FP8 sites' declared step is the checkpoint's static A8: the input
  multiplied by fl32(1/s) and rounded to E4M3 (saturating), exact in BF16, with s applied to every
  column's exact sum in the epilogue (`LosslessActivationScale::column_scales`). The routed and
  shared experts are W4A16 (no declared activation step).
- **Measurement.** Reproducible route vs `--lossless` (KL, top-1, TV on the quick corpus and
  decode corpus), and its processing cost against `--reference`, so the static A8 sites' share
  is separated from the rest.
- **Comparison point.** A sanity comparison against vLLM's output distribution is out of reach on
  this machine; published task scores are the only external check.

### 14.5 Fast-path candidates (after the targets are measured)

- static A8 decode and verification of the FP8 projections. The static-scale counterpart of
  `batched-a8-decode`, which failed the 27B's speculative decode arm with dynamic scales; worth a
  few per cent of decode at most (§4.3);
- `prompt-pv8` at group 16;
- `decode-split64`;
- a BF16-MMA expert wide route (approximate, prefill only);
- `expert-a4-residual` (and its three-term form): implemented as an unqualified candidate
  (`--fast-path`). The routed and shared experts' wide calls run Flash-Next's grouped mxf4nvf4 GEMM
  on two (three) NVFP4 terms per x and h row at this geometry (`wide_expert.h` A4Planes,
  `wide_expert.cuh` ScheduleResidual: Flash-Next's 64 x 128 tile, two stages, two CTAs a SM; warps
  skip 16-column MMA tiles past a tile's valid columns). Narrow experts (the narrow ring included)
  and CPU-served misses stay exact, so a column's result depends on its expert's column count in the
  call, not on where its record was read. Op oracle: `test_offloaded_moe_a4_residual_122b`;
  speed: `infernix_offloaded_moe_wide_bench --geometry 122b --activation a16x2` against `a16`, then
  pp4096/pp32768 with and without the path;
- `expert-tail-skip` remains an approximate opt-in.

Each needs model-quality §6's kernel-based gate on every KV format and a resolved speedup.

---

## 15. Implementation phases

Serial, one session; GPU work joins the agreed queue. Estimates are working days at Flash-Next's
pace.

| Phase | Work | Done when | Days |
|---|---|---|---:|
| **M0** Measurements that decide designs | 77.6 GB source download. Full block-scale statistics (per-matrix fallback share). Route traces from Qwen3.6-35B-A3B replayed through `tools/expert_cache_replay` at 4,600 frames → hit-rate curve and §1.4 sanity. Microbenchmarks: FP8 m16n8k32 throughput and accumulation error at the three shapes; the int32-group wide-epilogue prototype at 3072/1024 under ncu; CPU W4A16 rate and staged-copy rate for 5.31 MB records | Each design point in §8 confirmed or revised, with numbers recorded here | 2-3 |
| **M1** Shared offload component | Move §7's pieces to `src/runtime/expert_offload/`; `ExpertGeometry` template in the Op (both instantiations compiled, Flash-Next dispatching on its banks) | Flash-Next bit-identical (golden forced-decode hashes on recipe B and FP4-Local, real tests) and its decode/prefill A/B within spread | 3-4 |
| **M2** Formats and conversion | `fp8_e4m3fn_row_f32`, static A8 divisors on Uses, `nvfp4_rg16_v1`, ModelOpt Qwen3.5-MoE per-expert source, drafter methods for qwen3_5, recipe, `verify_artifact`, frontend tests on Ornith's template | Artifact converted; every object verified against the checkpoint | 2-3 |
| **M3** Dense path | §8.1-8.5 and §8.8: norm+encode, the FP8 families (3 shapes; A8 prefill and A16 decode/verify, fused epilogues), GDN at 64 heads, group-16 attention × 7 profiles, router, Q8 drafter shapes. Each kernel to its §8 budget under ncu | Op oracles and width-invariance tests pass; per-kernel times recorded against budget | 8-11 |
| **M4** Offloaded MoE in Qwen3.5 | §8.6-8.7 kernels; Program integration (round hooks, graphs, staging, promotions, warm start, VRAM monitor, elastic KV, ledgers), prefill walk + ring + CPU assist, head | Engine runs the artifact end to end; placement invariance on real records; first nsys trace | 5-7 |
| **M5** MTP | Drafter, `auto` policy with miss-cost input | Greedy MTP equals plain decode; acceptance measured | 1-2 |
| **M6** Vision | Frame loans, merger shape | Vision real tests, with and without offload | 1-2 |
| **M7** `--lossless` and `--reference` | `lossless_linear` gains `fp8_e4m3fn_row_f32` and the dense `nvfp4_rg16_v1` head (written); the offloaded lossless pass at 3072/1024 (written); the Qwen3.5 lossless layer with offloaded experts; lossless attention at GQA group 16; the design reference's static A8 step | Each lossless kernel matches its exact oracle at these shapes; `--lossless` and `--reference` score the quick corpus | 2-4 |
| **M8** Qualification, tuning, docs | §14.3 audit on every profile, §14.4 quality, §1.4 targets with baselines (download approval first), docs (§17) | Report with every finding, met or missed | 4-6 |
| **Total** | | | **28-42** |

The critical path is M3 (group-16 attention across seven profiles is the largest single item)
and M4 (the Program integration). M0's hit-rate curve and wide-epilogue prototype are the two
results that can change the targets or the design; they come first deliberately.

---

## 16. Risks and open questions

1. **Hit rate.** If routing locality is lower than Flash-Next's, plain decode falls toward the
   h = 0.80 row (85-105 tok/s) and the Must target is at risk. M0 measures it first.
2. **A8 decode as a fast path.** It may not pass the gate (dynamic one-term A8 failed the 27B's
   speculative arm). Decode then stays A16, which §4.3 puts within a few per cent.
3. **FP8 accumulation on sm_120.** The tensor core's internal FP32 accumulation of FP8 products
   may be narrower than IEEE FP32. That does not threaten determinism; it widens the op tolerance
   and is characterized in M0.
4. **Wide-epilogue gain.** Matrices with wide block-exponent spread take the slower exact path.
   The sample says 3.7 % of groups, concentrated in a few down projections. M0's full statistics
   confirm the per-matrix share.
5. **Long context.** Full attention costs ~1 ms per token at 128K `int8`. `nvfp4`/`vq2` KV and
   `decode-split64` are the levers, and the 128K target allows for it.
6. **Static activation clipping.** The per-tensor input scales clip at 448·s. Outliers above it
   saturate exactly as in vLLM/SGLang. They affect prefill only (D3); the processing cost against
   `--reference` (§14.4) shows the size.
7. **Windows pinning.** The pinned total (~63.6 GiB) is within the ceiling, but another process's
   shared-GPU-memory use lowers the room. The ledger reports it, and the SSD tier covers the rest.
8. **Upstream drift.** vLLM and SGLang both needed post-release fixes for this checkpoint. If
   Ornith republishes (e.g. a corrected template), recipe and resource checks pin the revision.

---

## 17. Documentation changes

When the work lands:

- [qwen3_5-model.md](qwen3_5-model.md): the 122B-A10B geometry row and offloaded-expert execution.
- [tensor-formats.md](tensor-formats.md) and [storage-layouts.md](storage-layouts.md):
  `fp8_e4m3fn_row_f32`, `nvfp4_rg16_v1`, expert records at 3072/1024.
- [weight-conversion.md](../weight-conversion.md): the recipe and static A8 divisors.
- [engine-architecture.md](engine-architecture.md): the `src/runtime/expert_offload/` ownership row.
- A user guide `docs/qwen3_5-122b.md` (requirements, convert, run, VRAM/RAM, performance), the
  README model and benchmark tables, and `docs/cli.md` for any new option.
- The Flash-Next design's references to the moved files.

This file is then removed.

---

## 18. Sources

- [ornith-ai/Ornith-1.5-122B-NVFP4](https://huggingface.co/ornith-ai/Ornith-1.5-122B-NVFP4)
  (revision `f049036f`), its `config.json`, shard headers and scale tensors;
  [Qwen/Qwen3.5-122B-A10B](https://huggingface.co/Qwen/Qwen3.5-122B-A10B) for the resource
  comparison.
- vLLM `vllm/model_executor/models/qwen3_5.py`, `qwen3_next.py`, `qwen3_5_mtp.py`,
  `layers/quantization/modelopt.py`, `layers/quantization/utils/marlin_utils_fp4.py`,
  `layers/fused_qk_norm_rope.py`, `csrc/quantization/w8a8/fp8/common.cu` (main `0746329d`).
- SGLang `python/sglang/srt/models/qwen3_5.py`, `qwen3_5_mtp.py`, `layers/quantization/modelopt_quant.py`,
  `kernels/jit/csrc/gemm/sm120_fp8_gemv.cuh`, `layers/moe/paged_experts/`, `layers/moe/kt_ep_wrapper.py`
  (main `3438ed66`); PRs [#45520](https://github.com/sgl-project/sglang/pull/45520),
  [#45298](https://github.com/sgl-project/sglang/pull/45298); issue
  [#19603](https://github.com/sgl-project/sglang/issues/19603).
- KTransformers `kt-kernel` ([README](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/README.md),
  PR [#2176](https://github.com/kvcache-ai/ktransformers/pull/2176)); [LMSYS blog](https://www.lmsys.org/blog/2025-10-22-KTransformers/).
- [Millstone: Qwen3.5-122B-A10B FP8 on 2× RTX PRO 6000](https://www.millstoneai.com/inference-benchmark/qwen3-5-122b-a10b-fp8-2x-rtx-pro-6000-blackwell);
  [rtx6kpro results](https://github.com/local-inference-lab/rtx6kpro/blob/master/benchmarks/results.md).
- Infernix: [Flash-Next design](qwen3_8-flash-next-design.md) §6.1.2, §16.2.1, §19.2-19.3 (as-built
  offload, measured wins and rejections); [model quality](model-quality.md) §6;
  [op development](op-development.md).
