# DeepSeek-V4.1-Flash-1017 on one RTX 5090: design

**Status: design proposal, not implemented.** Nothing in this document is built. It is a temporary
planning document for adding `DeepseekV41ForCausalLM` (DeepSeek-V4.1-Flash, model type
`deepseek_v41`) to Infernix. Its first artifact is a per-expert-bit EXL3 build of the 1017
checkpoint; the lossless MXFP4 (NVFP4-valued) build is deferred to a later milestone. When the
implementation lands, its stable content moves into the active authorities named in
[§24](#24-documentation-and-authority-changes) and this file is removed. The design reuses the
Qwen3.8-Flash-Next work and the GLM-5.3-Flash design wherever they transfer; they are cited as
**FN §x** ([qwen3_8-flash-next-design.md](qwen3_8-flash-next-design.md)) and **GLM §x**
([glm5_3-flash-design.md](glm5_3-flash-design.md)). DwarfStar (`antirez/ds4`), the local engine
whose V4.1 implementation this design studied most closely, is cited as **DS4** (§4.1).

**Revised 2026-10-24** after Strata and Project Maya published their V4.1 engines (§4.3) and a survey
of the published 1017 quantizations (§5.7). The 1017 release changes the Engram tables too (§2.6).
The owner then made the exact prompt rule the default (decision g, §1.2) and chose **EXL3 experts
with a bit width per expert, built for from the start**, as the first artifact (decision f); the
MXFP4 artifact is deferred (milestone M11). Sections written for the MXFP4 records (§5.1-§5.6,
§7.3's `mxfp4_expert_v1`, §9.5's FP4 rows) stay as that artifact's design. A review of DeepSeek's own
serving code (DeepInfer, the kernel libraries it pins, deepseek-recipe; §4.4) then corrected and
extended the model facts, the state rings, the attention and top-k arithmetic, the cache rules and
the frontend.

Labels used for numbers: **measured** (on this machine, with the method named), **upstream** (read
from the reference code, config, checkpoint headers or another project's published record),
**model** (derived from measured or upstream inputs by the stated arithmetic), **estimated**
(judgement; uncertain by the stated range). A milestone named in brackets replaces each model or
estimate with a measurement.

**The first artifact** (decision f): `DeepSeek-V4.1-Flash-1017-EXL3-2.75-Infernix`, its routed
experts taken as stored from HeNryous's per-expert EXL3 pack of 1017
([`HeNryous/DeepSeek-V4.1-Flash-1017-EXL3-2.75bpw-DGX-Spark`](https://huggingface.co/HeNryous/DeepSeek-V4.1-Flash-1017-EXL3-2.75bpw-DGX-Spark):
each expert 2, 3, 4 or 5 bits, 2.75 bits on average and 3.10 per decoded token, `mul1` codebook;
published KL 0.0201 and top-1 95.96 % against DeepSeek's reference code). Its Engram table is the
published 4-bit **IQ4_NL** table of 1017 (ISTA-DASLab's or Unsloth's, 110.6 GB; a 112.4 GB volume
against 209.7 GB for the FP8 rows; §7.5). Everything else comes from the official release: the dense
tensors bit-exactly and the DSpark dense weights; the drafter's experts are the pack's (2 bits; they
change acceptance only). sudoingx's 2.8
pack, made by the same pipeline, is accepted as a second source. Its expert records live in pages of
a shared pool (§7.3), so any mix of widths, including a uniform one, uses the same machinery.

**The deferred artifact.** "NVFP4 DeepSeek-V4.1-Flash" names one set of expert values, whichever
file it comes from:

- DeepSeek trained and released the routed experts as **MXFP4** (E2M1 codes, one UE8M0 scale per
  32 weights). NVIDIA's [`nvidia/DeepSeek-V4.1-Flash-NVFP4`](https://huggingface.co/nvidia/DeepSeek-V4.1-Flash-NVFP4)
  (ModelOpt, September checkpoint) re-encodes them as NVFP4 **losslessly**: "All 16,986,931,200
  weight blocks passed lossless conversion" (its card). The values are identical; only the block
  scale layout differs (§7.1).
- The October update [`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
  changes weights only (§2.6). No NVFP4 export of it exists yet (§4.2). Infernix's recipe imports
  either source, so a 1017 NVFP4 artifact needs no NVIDIA export: the expert values of the 1017
  release are the NVFP4 build's values by construction.
- Per the owner's decision (c) the artifact stores those values in their original MXFP4 form, 5.6 %
  smaller than NVFP4 records, unless M3 finds the MXFP4 kernels slower (§7.3, §9.6).

**Deferred MXFP4 artifact (M11):** `DeepSeek-V4.1-Flash-1017-Infernix`, converted from the official
1017 release: MXFP4 expert records striped over three NVMe volumes, the FP8 Engram volume, and every
dense tensor as stored (FP8 with 32 × 32 block scales, BF16, FP32). The September NVFP4 export is
the second supported source and the reference for the NVFP4 path.

The target machine is the one measured for the GLM design (GLM §3):

- one RTX 5090 (32 GB) on a **PCIe 5.0 x8** link, fixed (the other x8 lanes serve a network card);
- an i9-13900K (8 P-cores, 16 E-cores, AVX-VNNI, no AVX-512) with 96 GB of DDR5-5800;
- Windows 11;
- four NVMe drives (three Samsung 990 PRO and an Optane P5800X), three of them behind the chipset.

---

## Contents

1. [Goals, decisions, baselines and acceptance](#1-goals-decisions-baselines-and-acceptance)
2. [Model facts](#2-model-facts)
3. [Machine facts](#3-machine-facts)
4. [What other implementations do, and what transfers](#4-what-other-implementations-do-and-what-transfers)
5. [Performance model](#5-performance-model)
6. [Design overview](#6-design-overview)
7. [Artifacts, records, volumes and conversion](#7-artifacts-records-volumes-and-conversion)
8. [Ownership in Infernix](#8-ownership-in-infernix)
9. [Decode pipeline and canonical arithmetic](#9-decode-pipeline-and-canonical-arithmetic)
10. [Expert residency: VRAM, RAM and SSD](#10-expert-residency-vram-ram-and-ssd)
11. [Multi-SSD expert streaming](#11-multi-ssd-expert-streaming)
12. [CPU expert engine](#12-cpu-expert-engine)
13. [CSA2 attention and the hierarchical indexer](#13-csa2-attention-and-the-hierarchical-indexer)
14. [Engram](#14-engram)
15. [Single-pass mHC](#15-single-pass-mhc)
16. [Prompt processing: the encoder walk and the decoder rule](#16-prompt-processing-the-encoder-walk-and-the-decoder-rule)
17. [Speculative decoding: DSpark](#17-speculative-decoding-dspark)
18. [Prefix cache, preemption and invariance](#18-prefix-cache-preemption-and-invariance)
19. [Memory plans](#19-memory-plans)
20. [Numerics and qualification](#20-numerics-and-qualification)
21. [Frontend and serving](#21-frontend-and-serving)
22. [Configuration surface](#22-configuration-surface)
23. [Implementation plan](#23-implementation-plan)
24. [Documentation and authority changes](#24-documentation-and-authority-changes)
25. [Risks, open questions, options and rejected alternatives](#25-risks-open-questions-options-and-rejected-alternatives)
26. [Sources](#26-sources)

---

## 1. Goals, decisions, baselines and acceptance

### 1.1 Deliverable

The product supports `DeepseekV41ForCausalLM` through the public Engine, `infernix`,
`infernix-serve`, `infernix_bench` and offline CausalScoring, with:

- the Text model: 40 layers in a 20-layer causal encoder and a 20-layer decoder, CSA2 sparse
  attention with the two-level indexer, single-pass mHC, Engram, and the 384-expert MoE;
- the full-model prompt rule, computed exactly by the decoder suffix, by default, and DeepSeek's
  served rule (decoder SWA bounded replay) as a startup option (§16, decision g);
- DSpark speculative decoding chosen per round by the cost model, and n-gram copy proposals (§17);
- the model's own KV formats: FP8 window entries, FP4 compressed entries, MXFP4 index keys (§13.5);
- the hybrid prefix cache in `infernix-serve` (§18);
- the DeepSeek encoding: thinking, reasoning effort 1-100, DSML tool calls, tool results and
  reminders in the OpenAI and Anthropic protocol surfaces (§21);
- routed experts as **EXL3 records of 2-5 bits per weight, one width per expert**, held in a **page
  pool** shared by every tier (§7.3, §10);
- routed experts in **three tiers**: a VRAM expert cache, pinned and unpinned host RAM, and expert
  volumes striped over **one to four NVMe drives** (§10-§11);
- the Engram tables in their own volume: the first artifact plans the published **IQ4_NL** table,
  and **FP8 and MXFP4 rows** stay selectable per run like Flash-Next's n-gram tables (§14);
- the EXL3 recipe: routed and drafter experts from a per-expert EXL3 pack of 1017, everything else
  from the official release, and a **streaming conversion** that fits the disks free at conversion
  time (§7.1, §7.6);
- an **opt-in approximate expert mode** (GLM §9.8's shape), off by default and labelled (§9.8).

Deferred to M11: the MXFP4 artifact, its recipe for the official release (September or 1017) and
NVIDIA's ModelOpt NVFP4 export, and the MXFP4 and NVFP4 expert routes.

**Fidelity.** Every stored tensor is imported bit-exactly: dense FP8 codes and scales, BF16 and FP32
tensors verbatim, Engram rows verbatim in the chosen format, and the pack's EXL3 expert tensors
verbatim (trellis codes and sign-scale vectors). Every FP8 linear and every expert input takes
DeepSeek's own activation quantization (MXFP8: E4M3 with one power-of-two scale per 32 activations,
decision b). FP8 linears compute it with exact integer sums (**MX8**, §9.5); EXL3 experts apply the
GLM design's canonical E3A16 arithmetic to those MX8 values (§9.5). So an expert's output is the same
bits from any page, a staging slot, the CPU or any landing order. The experts are the pack's
approximation of DeepSeek's (§5.7); everything else reproduces the reference forward up to the
accumulation order the reference leaves to its tensor cores. The deviations from
`inference/model.py` are listed in §20.4.

**Vision** (DeepSeek-ViT, 32 layers) is deferred: the artifact keeps the tower, the binder refuses
`--vision` for this family until a later track adds it (§23).

### 1.2 Owner decisions (2026-10-19 and 2026-10-24)

| # | Question | Decision | Implemented in |
|---|---|---|---|
| a | Prompt rule | ~~DeepSeek's decoder bounded replay by default~~, superseded by g: both rules exist and are switchable (`--prompt-decoder replay\|full`); `full` is computed exactly by the 2,541-position decoder suffix DS4 uses, whose output equals running every prompt token through all 40 layers | §16 |
| b | Activation arithmetic | **DeepSeek's 8-bit activations** (MXFP8) at every FP8 and FP4 linear | §9.5 |
| c | Expert storage (the MXFP4 artifact, deferred) | **The original MXFP4 form**, provided it is not slower than NVFP4 records; M11 decides by kernel measurement, NVFP4 records remain the fallback and serve non-MXFP4-representable builds | §7.3, §9.6 |
| d | Engram tables | **Both FP8 (source) and MXFP4 (NVIDIA) rows**, read from the volume, chosen per run as Flash-Next's n-gram row formats are | §14 |
| d (2026-10-24) | Engram table of the first artifact | **The published IQ4_NL table** (ISTA-DASLab's or Unsloth's 1017 table) instead of the 209.7 GB FP8 volume; FP8 and MXFP4 rows stay selectable per run, and §20.3 measures what the table costs | §7.5, §14.5 |
| e | Disk space | **A conversion route designed for the disks free when the artifact is made**, not a fixed plan | §7.6 |
| f (2026-10-24) | Expert format of the first artifact | **EXL3 with a bit width per expert, from the start**: HeNryous's 2.75 pack (sudoingx's 2.8 accepted), the page pool in the shared offload component from M2; the lossless MXFP4 artifact is not needed initially (M11) | §5.7, §7, §10 |
| — (2026-10-24) | Disk space | The owner frees space when the artifact work is started; the planner (§7.6) sizes the plan then | §7.6 |
| — (2026-10-24) | Shared pieces | **This track builds the shared EXL3 codec, offload extraction and page pool first; GLM reuses them** | §8.2, §23 |
| g (2026-10-24) | Prompt-rule default | **`full`, the exact rule, by default**; bounded replay stays as `--prompt-decoder replay`. Maya measured replay against the exact rule at KL 1.6e-2 / 6.6e-3 / 3.2e-4 after 4K / 8K / 26K-token prompts (its engine's own summation-order noise: 6.4e-4 / 6.1e-3 / 8.8e-5), with greedy text diverging at tokens 2 and 8; here `full` costs +7-13 % prompt time (§4.3) | §16 |

### 1.3 Baselines

All comparisons run on this machine with the same prompts, output caps and sampling.

| Baseline | What it shows |
|---|---|
| **Strata** (`Niko1221/Strata` 0.2.4.x, Windows CUDA packages built for `sm_120`) and **Project Maya** (`maya-ds4` v1.1.26) | **The main baselines**: the local engines closest to this design in mechanism (VRAM expert cache, CPU lane, SSD reads) that run V4.1 natively on Windows (§4.3). Neither runs per-expert EXL3 packs, so each is compared with its nearest build in expert bytes per pick (Maya: uniform EXL3 2.9 and Maya-S; Strata: ISTA's 2.25 set), with quality beside speed |
| **JigSawPT's llama.cpp fork** (`dsv41-porte`) with the MXFP4 experts streamed from NVMe | The only published run on an RTX 5090 (§4.2: 5-10 tok/s with 125.7 GiB of RAM); 1.4× this artifact's bytes per pick, the same values as the deferred MXFP4 artifact |
| **DS4** (`antirez/ds4`, MIT) with its V4.1 Q2 (341 GiB file, 152 GiB of weights) and Q4 (483 / 294 GiB) GGUFs, CUDA, SSD streaming | The local engine studied most closely (§4.1). Its CUDA V4.1 path targets the DGX Spark's unified memory and Linux; on this machine it would run under WSL2 with the GGUF on an ext4 volume, and its discrete-GPU expert cache is an unmerged pull request (DS4 #647). Compared as each engine runs in practice: Q2 has about half the expert bytes of NVFP4 and measurably lower quality (§4.1), so the report gives quality beside speed |
| **ik_llama.cpp** and the other engines of §4.2 that run V4.1 with expert offload | Their own numbers in §4.2; run here where they support Windows or WSL2 |
| **Infernix Flash-Next** on this machine | Not a quality baseline (another model); the reference for what each mechanism achieved here |
| **DeepSeek's reference** (`inference/model.py`) and vLLM/SGLang | Need four or more datacenter GPUs; reference logits only, through §20.2's offline reference forward |

### 1.4 Targets

No measurement of this model exists on this machine. The MXFP4 targets were fixed on 2026-10-19
from §5's model (and their prompt rows moved to the full rule with decision g, thresholds unchanged).
The first artifact's targets were fixed on 2026-10-24 from §5.7's model, before any measurement, at
the same ratios to the model as the MXFP4 targets (Must ≈ 0.7 × the central decode, Stretch ≈ 1.15 ×;
prompts likewise).
Primary configuration: the EXL3 per-expert artifact with the IQ4_NL Engram volume, three expert
volumes (§11.4; drives chosen when the owner frees space), the full prompt rule,
`--max-context 131072`, C = 1, ≈ 88 GiB of RAM available (other programs closed), thinking at the
encoding's default effort (75), DeepSeek's sampling (temperature 1.0, top_p 1.0; §22), 4K prompt,
512 output tokens; greedy results are reported beside.

| Metric | Must | Stretch | Model (§5.7), central (range) | Deferred MXFP4 artifact: Must / Stretch / model |
|---|---:|---:|---:|---:|
| Plain decode, warm cache, C = 1, non-repeating session (§20.5) | ≥ 14 tok/s | ≥ 23 | 20.0 (16.0-25.0) | 9 / 15 / 12.8 |
| Best speculative mode, same corpus | ≥ plain, never slower beyond noise | ≥ 27 | 27-29 with DSpark at acceptance 0.7-0.8, before the drafter's pages are charged; a greedy upper bound (§5.4) | plain / 17 / 13.6-16.2 |
| Decode, three volumes relative to one | ≥ 1.6× | ≥ 1.75× | 1.75× | 1.8× / 2.0× / 2.0× |
| Fresh 32K prompt | ≥ 2,200 tok/s | ≥ 3,800 | 3,040 | 2,000 / 3,500 / 2,500 |
| Time to first token, fresh 4K prompt | ≤ 10 s | ≤ 6 s | 7.6 s | 15 / 9 / 13.0 s |
| Time to first token, 50-token append to a cached conversation (an agent turn: tools declared, the previous reply re-rendered unchanged) | ≤ 2.6 s | ≤ 1.5 s | 2.0 s | 6 / 3.5 / 4.7 s |
| Decode at 128K actual context, relative to 4K | ≥ 90 % | ≥ 97 % | ≈ 98 % | the same |
| Execution fidelity against the reference forward over the artifact's values (§20.3) | within the gate | — | — | the same |
| Quality against the original model (§20.3) | reported, beside the pack's published KL 0.0201 / top-1 95.96 % | — | — | lossless by construction |

The acceptance report gives the median and range of at least three runs, the worst case, the expert
service composition (VRAM hits, RAM experts computed by the CPU, RAM experts staged over the link,
SSD reads and bytes per token, Engram reads), and the device each number came from. A missed target
is reported as missed; targets are not changed after measuring.

These targets are below the GLM design's (25 tok/s) because this model reads 3.3 GB of routed
experts per decode token even at 3.10 bits per pick (MXFP4: 4.5 GB; GLM's EXL3 build: 3.2 GB) from a
bank of 188 GB (MXFP4: 289 GB; GLM: 115 GB), and this machine holds about half of it in memory (MXFP4:
37 %; GLM: 87-96 %). That is a property of the model on this hardware; §5.6 lists the hardware
changes that move it.

### 1.5 Non-goals

- Multi-GPU, AMD GPUs, GPUs other than `sm_120a`.
- Any change to the weights, dynamic top-k, expert deferral or skipping in the default
  configuration; the one exception is the opt-in, labelled approximate mode (§9.8).
- Requantizing the experts at conversion: compact expert formats are imported from published builds
  as stored (§5.7).
- The MXFP4 artifact in the first track (decision f; M11).
- DeepSeek's encoder SWA bounded replay for cache restores: Infernix restores the encoder exactly
  from bounded snapshots (§18.2).
- Writing the expert or Engram volumes at runtime (read-only, FN §19.3.7 U3).
- The vision tower in this track.
- A generic model graph or plugin layer. Shared offload components are extracted only because
  explicit families use them (§8.2).

---

## 2. Model facts

### 2.1 Configuration (upstream)

From the 1017 repository's `config.json`, `inference/config.json`, `inference/model.py`,
`inference/kernel.py`, `inference/engram.py` and the safetensors headers (read 2026-10-18 by HTTP
range requests), and the technical report (arXiv 2609.19969, §2 and §4.2.1). The September and 1017
`config.json` files are byte-identical.

| Quantity | Value |
|---|---|
| Layers | 40 + 3 DSpark blocks (indices 40-42, tensors `mtp.0-2`). Layers 0-19 form the causal encoder, 20-39 the decoder (CED) |
| Hidden width | H = 5,120; RMSNorm eps 1e-20, weight multiplies |
| Residual | mHC, 4 streams (20,480 wide), 24 mixing coefficients per sublayer from one projection, Sinkhorn 20 iterations, eps 1e-6, **single pass** (§2.2) |
| Attention | 64 query heads × 512 (last 64 dimensions rotated, interleaved pairs); one 512-wide KV entry per position, used as key **and** value; q_lora rank 1,280; output low-rank in 8 groups (each 8 heads × 512 → 1,024), then 8,192 → 5,120; a learned sink logit per head; softmax scale 512^−½ |
| Attention sources | layers 0-1: window of 128 only. Layers 2-19: window + compressed entries at ratio 2 (Full at 2, 8, 14; Reuse otherwise). Layers 20-39: window + compressed entries at ratio 1 (Full at 20; Reindex at 24, 28, 32, 36; Reuse otherwise). DSpark: window only |
| Indexer | 32 heads × 128, top-512 compressed positions per query. Decoder: two levels, layer 20 keeps 2,048 blocks of 8 positions (16,384 candidates), Reindex layers score only those |
| RoPE | layers with compression: θ = 160,000 with YaRN (factor 16 over 65,536; β 32 / 1); layers 0-1 and DSpark: θ = 10,000 without YaRN |
| MoE | every layer: 384 routed experts, top-6, plus 1 shared expert; width 2,304; SwiGLU clamp 10 (gate ≤ 10, up ∈ [−10, 10]) |
| Router | FP32 logits; score √softplus; selection on score + `bias` (text) or + `bias_vl` (image-span tokens); weights = selected scores normalized (+1e-20) × 1.5 |
| Engram | before layers 1 and 14: n-grams of orders 2-4 over a 99,092-id compressed vocabulary, 8 hash heads per order, 24 rows of 256 values per layer and token; tables of 384,006,168 and 384,016,682 rows |
| DSpark | 3 blocks (window attention, mHC, MoE of 128 experts top-3), conditioned on layers 37-39; 5 draft positions per pass (anchor + 4 noise tokens, id 128,799); Markov head of rank 256; confidence head |
| Vocabulary | 129,280 (128,000 BPE + 1,283 added); untied embedding and head; final RMSNorm after the last collapse |
| Context | 1,048,576 positions |
| Stored formats (official release) | routed and DSpark experts MXFP4; attention, shared experts, Engram projections and DSpark dense FP8 E4M3 with one UE8M0 scale per 32 × 32 block; Engram tables FP8 E4M3 with one UE8M0 scale per 32 values; norms, routers, compressors, indexer keys and weights, head and embedding BF16; mHC parameters, sinks and router biases FP32 |
| Vision | DeepSeek-ViT, 32 layers × 1,024, plus an aligner (0.97 GB BF16) |

### 2.2 Mathematics (upstream, `inference/model.py` and `kernel.py` at the 1017 revision)

Written once here so every Op and oracle uses the same statement. Shapes per token; S is the
residual, 4 streams of H.

**Activation quantization MX8** (`act_quant` with `scale_fmt` ue8m0; applied to the input of every
FP8 and FP4 linear, to the window entries and to the attention output before `wo_a`):

```text
per 32-element block b of a BF16 vector x:
  a_b = max(max_k |x_k|, 1e-4)                      FP32
  s_b = 2^ceil(log2(fl32(a_b · fl32(1/448))))        exponent of the product, +1 if its mantissa ≠ 0
  q_k = e4m3_rn(clamp(x_k / s_b, −448, 448)) · s_b   the linear's input (E4M3 codes and s_b)
Linear(W, x) = bf16( Σ_k W_k · q_k )                 W decoded exactly: code × block scale
```

FP4 weights take the same MX8 input (the reference casts them to FP8 and runs an FP8 GEMM). The V4
report (§5.2.1) trains the experts with MXFP4 QAT on the FP8 training framework, so this cast is
part of the trained forward, not an inference shortcut.

**Embedding.** S_s = embed(t) for s = 0..3; the first sublayer's collapse coefficients are one-hot
on stream 0.

**Single-pass mHC** (`Block.forward`, `hc_split_sinkhorn`). For a sublayer k entered with stream
state S:

```text
m      = (fn_k · flat(S)) · rsqrt(mean(flat(S)²) + 1e-20)      FP32; fn_k [24 × 20,480]
pre'   = sigmoid(m[0:4] · σ0 + β[0:4]) + 1e-6                   handed to the NEXT sublayer
post   = 2 · sigmoid(m[4:8] · σ1 + β[4:8])
comb   = Sinkhorn(softmax_row(m[8:24] · σ2 + β[8:24], 4 × 4) + 1e-6)
           one column normalization, then 19 × (row, column), each with + 1e-6
x      = bf16( Σ_s pre_s · S_s )        pre: the coefficients this sublayer RECEIVED
y      = sublayer(norm(x))
S'_k   = bf16( post_k · y + Σ_j comb[j][k] · S_j )             FP32, one rounding
```

A layer is the attention sublayer then the FFN sublayer. The report (§2.4.1, Eq. 6) states the shift:
X_{l+1} = B_l X_l + C_l F_l(A_{l−1} X_l). The head collapses with the last FFN's pre', then
`logits = head(norm(x))` in FP32.

**Engram** (layers 1 and 14, before the block):

```text
rows   = 24 rows of 256 (orders 2..4 × 8 heads), decoded to BF16 (code × UE8M0 scale)
kv     = Linear(W_e, concat(rows))                 6,144 → 25,600 (FP8, MX8 input)
key_s  = kv[s·H : (s+1)·H] (FP32), value = kv[4H : 5H]
dot_s  = Σ_d S_s,d · (q_w · k_w)_s,d · key_s,d · rstd(S_s) · rstd(key_s) · H^−½      eps 1e-20
g_s    = sigmoid( sign(dot_s) · √max(|dot_s|, 1e-6) )   (0 on image tokens)
S_s    ← bf16(S_s + g_s · value)
```

**Attention** (layer l, query position t, x the normed collapse):

```text
c_q    = q_norm(Linear(W_qa, x))                                  1,280
q      = RoPE_l(Linear(W_qb, c_q))          64 × 512, BF16
w_t    = MX8(RoPE_l(kv_norm(Linear(W_kv, x))))                    window entry of layer l
E(t)   = { w_j : t−127 ≤ j ≤ t } ∪ { c_p : p ∈ Sel_l(t) }          each entry is key and value
o_h    = Σ_{e∈E(t)} softmax_e(q_h · e / √512; sink_h) · e         exp(sink_h) joins the denominator
o_h    ← RoPE_l^−1(o_h) on its last 64 dimensions
y      = Linear(W_ob, concat_g Linear(W_oa,g, MX8(o_g)))          8 groups of 4,096 → 1,024; 8,192 → 5,120
```

**Compressor** (Full layers: 2, 8, 14 at ratio 2; 20 at ratio 1), per group of r tokens:

```text
r = 2:  k_i = W_ck x_i, z_i = W_cg x_i (FP32);  c = bf16( Σ_i softmax_i(z_i) ⊙ k_i )   per channel
r = 1:  c = Linear_bf16(W_ck, x)
c      = norm(c);  k_idx = MXFP4(RoPE_l(k_norm(W_k c)))            index key (Full layers)
entry  = FP4(RoPE_l(c)) at the group's first position: E2M1 with one E4M3 scale per 16
```

**Indexer** (Full and Reindex layers):

```text
q_h     = MXFP4(RoPE_l(W_qb_idx c_q))   32 × 128;   w_h = (W_w x)_h · 128^−½ · 32^−½
score_p = Σ_h w_h · relu(q_h · k_idx,p)       over compressed positions visible to t
Full, r = 2 (2, 8, 14):    Sel = top-512 of all visible positions
Full, r = 1 (20):          Sel = top-512 of all visible positions; separately, blocks of 8 positions
                           scored by their maximum, the block holding the newest position always
                           kept, and the top-2,048 blocks form the candidates C(t) for later layers
Reindex (24, 28, 32, 36):  Sel = top-512 of its own scores within C(t)
                           (128^−½ · 32^−½ = 2^−6 exactly: a power of two that changes no selection;
                           DeepInfer never applies it)
Reuse:                     Sel = the latest Full or Reindex layer's Sel
```

**MoE:**

```text
z     = W_r x  (FP32);  s = √softplus(z);  top6 = argtop6(s + bias)     bias_vl on image tokens
w_i   = 1.5 · s_i / (Σ_top6 s + 1e-20)
E(x, w) = Linear(W_d, bf16( w · silu(min(g, 10)) · clamp(u, −10, 10) )),  g, u = Linear(W_g, x), Linear(W_u, x)
y     = bf16( Σ_i E_i(x, w_i) [FP32, ascending expert id] + E_shared(x, 1) )
```

The routing weight multiplies h **before** the down projection, inside its MX8 input.

**DSpark** (after the target's forward of the committed token):

```text
main_x = norm(Linear(W_main, concat(mean_s S^(37), mean_s S^(38), mean_s S^(39))))   S^(l): stream state entering layer l
inputs = embed(anchor), embed(noise) × 4, each as 4 streams
block  = mHC + attention over {kv_norm(W_kv main_x) of the last 128 committed positions} ∪ {the 5 block entries,
         no causal mask} + MoE(128 experts, top-3)
logit_i = head(norm(collapse))_i + head_m(embed_m(x_{i−1}))   Markov bias of rank 256, drafts sampled in order
conf_i  = W_c · [h_i ; embed_m(x_{i−1})]
```

### 2.3 The prompt rule (CED)

The decoder's compressed entries come from layer 20's compressor applied to layer 20's input, which
depends on the encoder only. Every layer, encoder or decoder, still keeps its **own** window of the
last 128 positions, built from its own input.

- **Full model** (the reference, `model.py`): every prompt token runs all 40 layers. Exact
  computation needs less: decoder layer l (20..39) must evaluate the last 1 + (39 − l) · 127 prompt
  positions and only form window entries for the 127 before them, because layer l's window at a
  position depends on layer l − 1's output 127 positions earlier. Layer 20 touches 2,541 positions,
  layer 39 128. DS4 computes exactly this suffix (`ds41_decoder_prepare`, read 2026-10-19), with the
  same output as the full forward.
- **Bounded replay** (DeepSeek's serving rule, report §3.2.2): only the last **140** prompt positions
  run the decoder, with window attention truncated at the replay start s (query i sees window entries
  max(s, i − 127) … i). DeepSeek states it "is not mathematically equivalent" to the full forward,
  simulates it during post-training, and serves it; decoder windows are never cached. The report and
  vLLM #56752 speak of 128 positions, but DeepSeek's own engine keeps 140
  (`DEEPINFER_YOCO_PREFILL_KEEP_TOKENS=140` in DeepInfer's README; its code requires at least window
  + drafts + 1 = 134, so a sequence that backtracks after verification still has a full window).
  Infernix records the length as a model parameter, default 140.
- The two rules give the same encoder state and decoder compressed entries; they differ in the
  decoder windows at the prompt's end, hence in every token decoded after a prompt longer than the
  replay length. For a prompt of at most 140 tokens following nothing cached they coincide.
- DeepSeek's replay start is the current sequence length minus 140; after a recompute that length
  includes generated tokens, so DeepSeek's own replay is not preemption-invariant. Infernix's replay
  is defined on the prompt (§18.3). 140 is the default of DeepInfer's README and benchmark; its code
  enforces only ≥ 134 and nothing derives 140.

Infernix runs the full model by default and bounded replay on request (decision g, §16).

### 2.4 Parameters and bytes (upstream sizes from the safetensors headers; model elsewhere)

| Item | Parameters | Stored bytes | Bytes read per decode token |
|---|---:|---:|---:|
| One routed expert: gate, up [5,120 → 2,304], down [2,304 → 5,120] | 35,389,440 | MXFP4 **18,800,640** (17.93 MiB); NVFP4 19,906,584 (+5.9 %) | — |
| Routed bank, 40 × 384 | 543.6 B | MXFP4 **288.8 GB**; NVFP4 305.8 GB; HeNryous's per-expert EXL3 **187.6 GB** (2-5 bits, 2.75 mean) | 240 experts: 4.51 / 4.78 GB; EXL3 ≈ 3.29 GB (3.10 bits per pick) |
| DSpark experts, 3 × 128 (MXFP4 in both sources) | 13.6 B | 7.22 GB | ≤ 45 per draft round |
| Attention, 40 layers (wq_a, wq_b, wkv, wo_a, wo_b; FP8 + 32 × 32 scales) | 5.06 B | 5.07 GB | same |
| Shared experts, 40 (FP8) | 1.42 B | 1.42 GB | same |
| Routers (BF16), mHC (FP32), compressors and indexers (BF16, FP8) | 0.08 B, 0.04 B, 0.04 B | 0.16, 0.16, 0.08 GB | same (compressors and indexers on their layers) |
| Engram projections (FP8), 2 | 0.31 B | 0.31 GB | same |
| Engram tables, 2 | 196.6 B | FP8 rows **202.8 GB**; NVIDIA MXFP4 rows 104.4 GB | 48 rows: 12.7 KB (FP8) / 6.5 KB |
| Head (BF16) | 0.66 B | 1.32 GB | same |
| **Dense text total, VRAM-resident** (head included, embedding and Engram tables excluded) | **7.6 B** | **8.5 GB** | **8.5 GB** |
| Token embedding (BF16) | 0.66 B | 1.32 GB | one row |
| DSpark dense (attention, shared experts, `main_proj`, Markov head, confidence) | 0.6 B | 0.70 GB | per draft round |
| Vision tower and aligner (BF16) | 0.49 B | 0.97 GB | — |
| **Whole checkpoint** | 552 B + 196 B Engram | official 510.3 GB; NVIDIA 429.0 GB | |

The NVIDIA export differs from the official release only in the expert scale layout (an E4M3 scale
per 16 weights, an FP32 `weight_scale_2` and a calibrated FP32 `input_scale` per matrix) and in the
Engram tables of its current revision (MXFP4 since 2026-10-14; FP8 at revision `3431dde3`).

### 2.5 Consequences

- **Dense weights are the larger part of a token's GPU reads**: 8.5 GB against 4.5 GB of routed
  experts. With every expert in VRAM a token would take ≈ 8.4 ms (≈ 120 tok/s); the GPU is not
  where this model is limited on this machine.
- **With the per-expert EXL3 records about half the bank fits in VRAM and RAM** (8-10 % in VRAM,
  §19), and a token reads ≈ 28 experts, ≈ 0.38 GB, from the volumes (model, §5.7). With MXFP4
  records about 7.5 % fit in VRAM and 29 % in RAM, so ≈ 63 % live only on the SSDs, and a token
  reads ≈ 43 experts, ≈ 0.8 GB (central hit rate, §5.2).
- **A miss moves 8.9-22.2 MB** (13.7 MB per pick on average; MXFP4 18.8 MB): ≈ 0.5 ms over the x8
  link and ≈ 0.97 ms from three striped volumes (MXFP4: 0.68 and 1.33 ms).
- **The KV cache is negligible**: 890 B per token of compressed entries and index keys (the
  report's figure, which §13.5 reproduces), 117 MB at 128K and 0.89 GB at 1M, plus 2.9 MB of
  windows per sequence. Long context does not compete with the expert cache.
- **There is no recurrent state** (unlike Flash-Next's GDN or GLM's KDA): a prefix snapshot is the
  windows and a few partial groups, ≈ 3-9 MB (§18).
- **Engram costs I/O operations, not bandwidth**: 48 row reads per token, each one 4 KiB read
  from the Engram volume; 1.57 M reads for a fresh 32K prompt before deduplication (§14).
- **A prompt is a pass over the bank.** A token routes 6 of 384 experts per layer, so a prompt of
  T tokens touches 1 − e^(−6T/384) of each layer's experts: 54 % at 50 tokens, 95 % at 200. Every
  fresh prompt of a few hundred tokens or more reads most of the SSD-resident bank once (§5.5).

### 2.6 The 1017 checkpoint

Compared on 2026-10-18 through the Hugging Face API: `config.json`, `tokenizer.json`,
`tokenizer_config.json`, `encoding/`, `inference/` and the safetensors index are byte-identical to
the September release; all 48 shards keep their sizes and change their SHA-256. It is a weights-only
update, and it changes **every** tensor sampled: range reads on 2026-10-24 compared six 64 KiB
samples of each of both Engram tables' rows and scales, an Engram `wkv`, DSpark's `main_proj`, an
attention projection and a routed expert, and none of the 48 samples were equal. An Engram volume
therefore belongs to one checkpoint (§7.5), and so does a quantized build. The 1017 repository drops the
community `chat_template.jinja` the September one gained (§21). DeepSeek reports DeepSWE v1.1 77.0
(from 74.2), Terminal-Bench 4.0 42.3 (from 31.2, another harness) and shorter outputs.

---

## 3. Machine facts

The platform and storage were measured for the GLM design on 2026-10-10 (GLM §3.1-§3.3) and the
pinning ceiling for Flash-Next on 2026-10-14 (FN §19.3.7); free space and RAM were read again on
2026-10-18. No new probe was run for this design.

| Item | Value | Source |
|---|---|---|
| GPU link | PCIe 5.0 x8, fixed; copy-engine H2D 27.6 GB/s | GLM §3.1 |
| VRAM | 32,607 MiB; Windows budget 31,419 MiB with no display on the card | GLM §3.1 |
| Host DRAM | 2 × 48 GB DDR5-5800, 82-85 GB/s read | GLM §3.1 |
| CPU | i9-13900K, 8 P + 16 E cores, AVX-VNNI, no AVX-512 | GLM §3.1 |
| RAM visible / available | 95.8 GiB; 24.2 GiB free on 2026-10-18 with other sessions running, ≈ 88 GiB with them closed (the plans below assume the latter and print both) | `Win32_OperatingSystem` |
| Pin room | 75.2-76.1 GiB: Windows page-locks at most the adapter's `SharedSystemMemory` (76.61 GiB); experts past it live as unpinned, locked records | FN §19.3.7 |
| Free disk (2026-10-24) | `C:` 141 GB, `D:` 144, `E:` 18, `G:` 80: **383 GB in all** (`df`), against 895 GB on 2026-10-18 |
| Free disk (2026-10-18) | `C:` 80.5 GB (system, CPU lanes), `D:` 300.4, `E:` 405.6 (repository and builds), `G:` 108.7 (Optane, holds Flash-Next's n-gram volumes) | `Get-PSDrive` |
| One drive, large reads | 990 PRO 5.7-6.0 GB/s; Optane 7.0-7.1 GB/s; 1 MiB reads saturate at QD 4, 2 MiB at QD 2 | GLM §3.3 |
| Drive sets | `D:` + `G:` 12.73 GB/s; `D:` + `E:` + `G:` **14.15** (the chipset's DMI link binds); `C:` + `D:` + `E:` + `G:` **19.27** | GLM §3.3 |
| Random 4 KiB reads | ≈ 4.3-4.8 µs each from one issuer (Flash-Next n-gram reads); deeper rings measured faster | FN §12.4, §19.3.8 F7 |
| Windows power throttling | must be exempted (37.6 → 98.7 tok/s on Flash-Next) | FN §19.3.19 |

What this decides:

- **The SSD set is DMI-bound at three chipset drives** (14.15 GB/s); `C:` is the only drive that
  bypasses the chipset and adds its full rate (19.27 GB/s for four), but it is the system drive with
  80 GB free (§11.4).
- **Units of ≈ 1.5 MiB at QD 3-4 per drive saturate the drives** (between the 1 MiB and 2 MiB rows
  of GLM §3.3), so a record striped as 12 units keeps every drive busy (§11).
- **The pin room, not RAM, bounds the pinned tier**: about 72 GiB of experts can be pinned; the rest
  of the tier is unpinned locked records, served by the CPU in place and bounced through pinned slots
  for DMA (FN §19.3.7).
- **RAM shared with other sessions changes the plan at every startup**; the ledger reads it then
  (§19.2).

---

## 4. What other implementations do, and what transfers

Numbers in this section are the sources' own; none were reproduced here. [M] = the source reports a
run with its hardware, [C] = a claim without one. Sources were read on 2026-10-19 unless marked.

### 4.1 DwarfStar (`antirez/ds4`)

DS4 (MIT, read at `main` `fc80bd69`, 2026-10-07) is a narrow engine for a few models: Metal first,
then CUDA (the DGX Spark is its main CUDA target), then ROCm on Strix Halo. It runs only its own
GGUFs and implements V4.1's text inference end to end: tokenizer, encoding, CED, CSA2 with the
two-level indexer, Engram, mHC, SSD streaming, a server and an agent.

**What it ships for V4.1** (`docs/MODELS.md`, `gguf-tools/deepseek41_quantize.py`):

| Target | File | Main weights | Routed experts | Other tensors |
|---|---:|---:|---|---|
| `ds41f-q2` | 341 GiB | 152 GiB | IQ2_XXS gate/up, Q2_K down, imatrix-calibrated | Q8_0 attention, shared experts and head; F16 compressors, indexers and Engram projections; **Engram FP8 rows kept verbatim** (189 GiB, `e4m3_e8m0_32_row264`) |
| `ds41f-q4` | 483 GiB | 294 GiB | Q4_K | as Q2 |

Every routed expert and dense FP8 tensor is requantized: MXFP4 codes are not on Q4_K's uniform grid,
and FP8 blocks with 32 × 32 power-of-two scales are not Q8_0 blocks, so neither file holds the
released weights. DSpark is not supported for V4.1; vision runs on Metal only.

**Quality on DeepSeek's own continuations** (API log-probabilities as reference;
`QA_BEFORE_RELEASES.md` §17), [M]:

| Model, execution | Fixture | Mean NLL | API top-token agreement |
|---|---|---:|---:|
| Q2, M5 SSD | general 100 | 0.3646 | 2,697 / 2,994 |
| Q4, M5 SSD | general 100 | 0.2465 | 2,896 / 2,994 |
| Q2, M3 Ultra resident | extended 6 (64K-96K) | 0.5586 | 327 / 384 |
| Q4, M3 Ultra resident | extended 6 | 0.3665 | 360 / 384 |

Q2 costs 0.12-0.19 nats of NLL against Q4 on DeepSeek's own text; Q4 itself is a requantization.

**Speed** [M]: DGX Spark (GB10, 128 GB unified memory), Q2 with SSD streaming and an automatic
≈ 80 GiB expert cache: ≈ 100 tok/s prefill for a 3.2K prompt and **9.3 tok/s decode** at 32K
context; with a 64 GiB cache, a fresh 32K prompt at 384 tok/s, 2K at 85 tok/s. Two Sparks with
network tensor parallelism: 21.8 tok/s decode, 411 tok/s for a fresh 32K prompt. M3 Ultra (512 GB),
Q4 resident: 18.7 tok/s decode, 342-716 tok/s prefill.

**On a discrete RTX 5090.** DS4 documents no V4.1 path for discrete GPUs; its CUDA V4.1 code is
written around the Spark's unified memory (`cudaDevAttrIntegrated` branches in `ds4_cuda.cu`).
For V4 Flash a user measured 0.74-0.84 tok/s on an RTX 5090 with SSD streaming because the CUDA
expert cache was a no-op stub (issue #639, 2026-08-03); the open PR #647 adds a device LRU (GB10:
1.03 → 2.96 tok/s at an 81 % hit rate). Issue #477 measured +52 % decode on a discrete card from
keeping the model in the page cache when RAM exceeds the model. DS4 is Linux/macOS; on this machine
it would run under WSL2 with the GGUF on an ext4 volume.

**Mechanisms, and what this design does with them:**

| DS4 practice | Evidence | Disposition |
|---|---|---|
| **Exact decoder suffix.** For prompts of ≥ 8,192 tokens, decoder layer l evaluates only the last 1 + (39 − l) · 127 positions after warming its window with the 127 before them ("the exact 2541-token dependency suffix"); long prompts defer the decoder to the last sweep | `ds41_decoder_prepare`, `ds41_graph_prefill_sweep` | **Adopted** as `--prompt-decoder full` (§16.3), for every prompt length, not just ≥ 8K. DS4 does not implement DeepSeek's bounded replay |
| Engram rows read with `pread` from a region past the mapped weights, never mapped; a thread prefetches layer 1's rows before a prefill sweep and layer 14's while layers 2-13 run | `ds4_engram.c`, `ds4.c` | **Adopted** through Flash-Next's n-gram machinery (§14), which already reads rows through a deep ring of direct 4 KiB reads |
| Requantized Q2/Q4 experts and Q8_0 dense | above | **Not the default** (decision c and §1.5); a DS4-GGUF import is option O6 (§25.3) |
| Read-ahead for prefill with two 8 MiB staging buffers, from 2K tokens; "below 2K, unused expert reads outweigh the overlap"; short appends below 256 tokens in exact 2-8-row chunks through the cache | `QA_BEFORE_RELEASES.md` §17 | §16: the walk reads only the experts a prompt touches; read-ahead only for experts a chunk certainly needs |
| Scalar, batched and tensor-parallel execution "are not numerically identical" | `docs/MODELS.md` | Infernix keeps every route bit-identical (MX8, §9.5) |
| Static profile ("hotlist") first, recency as tie-break in the Metal expert cache; issue #636 (2026-07-31) measured on other models that LRU beat the best static policy by 8 points (MiniMax) and recency beat static pinning 62 % to 34 % (Kimi-K3) | `ds4_streaming_hotlist.inc`, issue #636 | Consistent with FN L4: LFRU, no static hot set by default (§10.2) |
| DSpark for V4 Flash with a confidence threshold (0.4 on one Spark) and a scheduler that backs off | `docs/SPECULATIVE_DECODING.md` | The confidence head feeds §17's cost model |

### 4.2 Other engines and artifacts

| Engine or artifact | Mechanism | Result |
|---|---|---|
| **JigSawPT's llama.cpp fork on one RTX 5090** (`dsv41-porte` with the "Crow" expert-streaming patches; 125.7 GiB RAM; llama.cpp discussion #28766, 2026-09-11/23) | MXFP4 experts (269 GiB): 18 GiB VRAM cache, 72 GiB pinned host tier, the rest streamed from NVMe; Engram mmapped from disk | [M] decode **5.12 tok/s on new content** (time to first token 8.56 s) and 21.27 tok/s on content already resident; host tier 0 / 32 / 64 / 72 GiB with direct I/O: 3.9 / 6.2 / 8.0 / 8.3 tok/s (buffered 3.0-4.3); **VRAM hit rate 62.7 %**, 59 → 66 % from 13 to 22 GiB with throughput flat; 26 new experts per token late in a reply; per remap call 54 % NVMe, 26 % PCIe, 20 % compute, the disk at 43 % of its rate; a layer predictor reached 70.9 % recall at depth 6 but prefetch lost 7-19 %; DSpark at draft length 2 accepted 51-79 % (97 % on verbatim text) for **−4 % cold and ±0 resident**; output not reproducible above ~1,024 tokens unless one I/O thread is used |
| **ik_llama.cpp** (V4.1 merged 2026-09-21, PR #2455) | experts on the CPU except six layers, Q3_K_M; Engram rows from the page cache | [M] 14.13 tok/s plain, 19.86 with DSpark n_max 3 (60.1 % accepted), 14.33 with n_max 5 (45.5 %), on a 32-core, 251 GB, two-GPU host |
| llama.cpp PR #28696 (open) | Engram tables as Q8_0 on the CPU, mmap | [M] GB10 Q2_K 2.8 tok/s; DSpark block 5 accepted 43.6 %, block 3 59.1 %; "Corpus NLL against the reference is not done" |
| **Xolotl** (`Indras-Mirror/Strata-DSV4.1`, a Strata fork) | RTX 4090, 90 GB DDR4, one NVMe; GSQ mixed ternary, 2- and 3-bit experts (~152 GiB): hottest in VRAM, a 64 GB pinned RAM cache, the rest on NVMe; Engram (~200 GB) on NVMe | [M as stated; benchmark file unavailable] decode 26.8 tok/s, prefill 180-320 tok/s |
| FreeToken (arXiv 2608.16157), DeepSeek-**V4**-Flash (284B, 256 experts, MXFP4) | one global LRU; RTX 5090 at x16 (52.7 GB/s), 180 GiB RAM | [M] decode miss rate at 11 % cache: **39 %** (KTransformers 59 %, llama.cpp 89 %); 22-25 tok/s |
| hipfire, V4-Flash-0731 (MQ2) on Strix Halo | expert cache by budget | [M] hit rate 40.5 / 59.4 / 74.4 / 86.5 / 91.0 % at 4 / 8 / 16 / 32 / 64 GiB; a learned next-layer predictor (recall@6 75.9 %) ran 0.86-0.93× and is off |
| KTransformers | V4-Flash only (RTX 5090 and ≥ 200 GB RAM) | [C] "20+ tok/s"; no V4.1 |
| cachedMoE (`acupof-ai/cachedMoE`, MIT), Strix Halo 128 GB | the native FP4/FP8 checkpoint with its experts streamed from NVMe | [M] exact routing 9.20 tok/s over an eight-turn conversation with two drives, 7.61 with one; a 4K prompt ≈ 29 s, 17K ≈ 45 s; its approximate modes (miss masking, top-K acceptance for DSpark) failed its own quality gates |
| vLLM / SGLang | datacenter; **bounded replay implemented** (vLLM #56752: layers 21-39 run on the last 128 prompt tokens; GSM8K 0.9249 on vs 0.9295 off, SE 0.0075; SGLang #42901 also trims layer 20's non-KV work, TTFT −5 / −8 / −10 % at 16K / 64K / 200K); Engram in pinned host memory (vLLM #61530: MXFP4 tables 12.2 instead of 23.6 GiB per rank, GSM8K 0.967 vs 0.966); FlashMLA's sparse kernels are SM90/SM100 only, sm_120 has open blockers (#56837, #56461) | reference only |
| **diffbot EXL3 3 bpw** (two RTX PRO 6000) | Engram int4 in pinned host RAM, read over UVA | [M] 74.4 tok/s at 2K, prefill ≈ 10K tok/s. **KL against the original** (neutral / code / agentic): EXL3 3 bpw 0.0545 / 0.0321 / 0.0101; **NVIDIA NVFP4 as served by vLLM (W4A4 + MXFP4 Engram) 0.0352 / 0.0197 / 0.0089**; the original's second run (noise floor) 0.0158 / 0.0100 / 0.0055 |
| OpensourceWTF EXL3 streaming repack (M5 Max, 128 GiB) | 13.3 MB expert records streamed from SSD into a slot cache; Engram FP8 rows from SSD | [M] 661.5 tok/s prefill and 36.8 tok/s decode at 16K |
| ds41rt / cuteafd | an RTX 5090 coordinator plus four DGX Spark expert ranks over RoCE | [M] 77-110 tok/s; not one machine |

**Quantized artifacts of 1017** (Hugging Face API, 2026-10-19): NVIDIA has published only the
September NVFP4 export. `LibertAIDAI/DeepSeek-V4.1-Flash-1017-NVFP4` (2026-10-17) is a community
ModelOpt-style export: lossless MXFP4 → NVFP4 experts ("0 mismatches over 135.90 × 10⁹
elements") with `input_scale` values, MXFP4 Engram tables (cosine 0.9934 to the FP8 tables), DSpark
experts in MXFP4, 399.51 GiB; it publishes no KL. By 2026-10-24 about 470 repositories carried V4.1
builds: Unsloth's nine UD GGUF sets, ISTA-DASLab's GSQ-RCO GGUFs, Maya's, DS4's, two dozen EXL3
packs (diffbot's moved to 1017 on 2026-10-21) and several trellis formats. §5.7 compares the ones
with published quality against their speed here.

**DSpark acceptance** [M]: vLLM issue #56797 (2 × 8 H20, k = 5, GSM8K at temperature 0) measured a
mean acceptance length of 2.817 with per-position survival 0.724 / 0.516 / 0.301 / 0.180 / 0.096;
chat traffic 2.45-2.69, agentic traffic 3.43 (another commenter), 3.28 on GB200 with bounded replay
(PR #56752). diffbot reports 4.1 accepted tokens per step on agentic runs.

What transfers:

- **The NVFP4 export's expert values are the original's, but its served form is not.** diffbot's
  table puts vLLM's W4A4 + MXFP4-Engram serving at KL 0.035 against a run-to-run floor of 0.016.
  The excess comes from 4-bit activations and the lossy Engram tables, not the weights. Decision b
  (8-bit activations) and the FP8 Engram default keep Infernix at the original's arithmetic; the
  W4A4 activation path is only a fast-path candidate (§9.5, §25.3).
- **The closest measured run is JigSawPT's**: the same card and expert format, decode SSD-bound
  (54 % of a remap waiting on NVMe, the disk at 43 % of its rate), a VRAM hit rate of 59-66 % at
  13-22 GiB. It calibrates §5.2 and shows what one drive with an idle queue costs.
- **Speculation pays only at short drafts here**: every verify column adds the union of its experts
  (JigSawPT −4 % at draft length 2 cold; ik_llama +41 % at n_max 3 with experts on a big CPU, a
  different regime). §17 chooses the draft length per round from the confidence head and the
  measured union.
- **Lookahead prefetch lost** on this model family wherever measured (JigSawPT, hipfire, Maya on
  GLM): off by default (§11.3).
- **Bounded replay costs little** on GSM8K (−0.005, within one standard error) and is what DeepSeek
  and both datacenter engines serve. Token-level measurements since then show a larger change (Maya,
  §4.3), so the default is the exact rule (decision g) and replay an option.
- **Engram is random row reads from disk everywhere** (DS4, JigSawPT, Xolotl, OpensourceWTF) or
  pinned host memory where RAM allows (vLLM). No engine found it a bottleneck in decode; in prompts
  it can be one where reads are issued from too few threads (Maya, §4.3).
- **Other engines are not reproducible across their own execution modes** (DS4 scalar vs batched,
  JigSawPT with more than one I/O thread). Infernix's placement invariance (FN §16.2) is the
  difference this design keeps.

### 4.3 Strata and Project Maya

Project Maya (`maya-ds4`, MIT; read at v1.1.26, `f26d8ba`, 2026-10-23) added V4.1 on 2026-10-18.
Strata (`Niko1221/Strata`; read at `7dd5cf18`, 2026-10-24) imported Maya's engine at v1.1.17
(2026-10-20), released it as 0.2.4 and has since ported Maya's later Windows changes with its own
A/Bs. Both run V4.1 on Windows and Linux, CUDA and HIP, on one or several GPUs:

- the most-used experts sit in VRAM, placed by a routing profile and swapped by heat;
- every other routed expert is computed by a CPU lane, from RAM or from the GGUF files on the SSD.

Neither has a run on an RTX 5090. The full reviews are kept with this design's working notes.

**Variants they run.**

| Build | Routed experts | Dense; Engram | Engine | Published quality against the original |
|---|---|---|---|---|
| Unsloth UD-Q8_K_XL / UD-Q4_K_XL | MXFP4 (the original values), 288.8 GB | BF16 / Q8_0; MXFP8, 202.8 GB | both | the "original" both measure against |
| ISTA-DASLab GSQ-RCO "1.75 / 2.00 / 2.25" | per-layer mixes of IQ1_S to IQ3_XXS: 1.69 / 1.94 / 2.19 bits, 114.6 / 132.1 / 149.1 GB | Q5_K and Q8_0, 6.2 GB; IQ4_NL, 110.6 GB | Strata (its shipped sets) | Strata, top-64 KL against its own UD-Q8_K_XL run: 0.217 / 0.170 / 0.107; perplexity +21 / +15.5 / +7.3 % (code +8.9 / +8.6 / +3.2 %); ISTA's task average 98.7 / 100 % / not run |
| Maya-S, Maya-XS | IQ2_XXS / IQ3_XXS mix, 2.49 bits; IQ1_S, 1.56 bits | Q8_0; Q6_K, 161.3 GB | both | against DeepSeek's reference: top-1 92.6 / 87.5 %, NLL +3.2 / +8.7 %, KL 0.120 / 0.276; the engine on the original weights 96.4 % / 0.047 |
| Unsloth UD IQ1_S to Q3_K_XL | 1.70-4.00 bits | mixed; IQ4_NL, Q5_K, Q6_K or MXFP8 | both | Unsloth's own (§5.7) |
| EXL3 packs (LibertAIDAI 2.9 / 3.5 and others) | trellis, `mul1` codebook | EXL3; official FP8 | Maya; Strata reads Maya's `mul1` | none against the reference |
| TQK trellis | K bits per value, converted layer by layer by Strata | — | Strata | none |

**Measured on one GPU** (the sources' hardware; decode tok/s unless marked):

| Machine | Build | Result |
|---|---|---|
| RTX 5070 12 GB, Ryzen 5 7600, 64 GB, one 3.6 GB/s NVMe (Strata's owner) | ISTA 2.00 | 4.3-4.9 with Strata's RAM store, 1.8-2.2 memory-mapped; prompts 121 tok/s at 4K, 231 at 8K |
| RTX 3090 24 GB, Ryzen 7 9700X, 128 GB (96 GB of experts copied), NVMe | Maya-S | 16.9-18.1 |
| RTX 4000 Ada 20 GB, 255 GB (every CPU-lane expert in RAM) | Maya-S; Maya-XS | 11.5-16.4; 16.2 plain, 19.4-22.9 with DSpark |
| Radeon AI PRO R9700 32 GB, 188 GB | UD-Q4_K_XL (MXFP4) | 14.0; prompts 378 tok/s at 8K |
| L40S 48 GB, 504 GB (every expert in RAM) | UD-Q8_K_XL (MXFP4) | 14.2-15 with 32 lane threads, 26.5 with 28; 41 with DSpark; prompts ≈ 1,000 tok/s at 20K |
| RTX PRO 6000 96 GB, 128 GB, 54 % of the experts in VRAM | UD-IQ2_XXS | 78.6-80.7; 91-92 with DSpark; a 28K prompt at 3,930-4,050 tok/s |
| 7900 XTX 24 GB with 64 GB; RTX 3090 with 64 GB and a 2.9 GB/s drive | Maya-S; EXL3 2.9 | 2.6-3.0; 4.9 (SSD-bound) |

**Mechanisms, and what this design does with them:**

| Practice | Evidence | Disposition |
|---|---|---|
| **Exact decoder suffix by default** (`PROMPT_SKIP=1`): from prompts of 3,692 tokens, rows before the last (40 − 20) · 127 + 128 = 2,668 run layers 0-19 and layer 20's KV work only. DeepSeek's 128-row replay is opt-in (`=2`) | Exact skip against no skip (summation order only): last-row KL 1.5e-4 (Maya's maintainer); 6.4e-4 / 6.1e-3 / 8.8e-5 at 4K / 8K / 26K (Maya #342); prompts 30-54 % faster. Replay against the exact skip (#342, 7900 XTX): KL 1.6e-2 / 6.6e-3 / 3.2e-4, greedy text diverging at tokens 2 and 8, prompt time −24 / −24 / −15 %: 25× that engine's noise at 4K, at it at 8K, 3.6× at 26K | `full` is this design's exact rule with a tighter suffix: 2,541 rows, because each decoder layer forms its own window entries (their layer 20 skips its window and pays 127 more rows). Adopted as the default (decision g, §16.1) |
| **The CPU lane computes every non-VRAM expert in decode.** Streaming RAM experts over PCIe (Maya) or zero-copy reads (Strata) lost, and made output depend on the start or on timing | Maya: streaming 62.7 against 78.6-80.7 tok/s; its per-layer timing underestimated the CPU's cost 2-3×; KL 1.8e-2 between two starts. Strata `c8ead784`: zero-copy slower and timing-dependent | Here the split cannot change output (MX8, §9.5), and FN L2 measured a link share beside the CPU faster than the CPU alone. The split's calibration (§9.7) is checked by whole-token A/B in M5, which is how Maya v1.1.25 fixed its choice |
| **Leave the host room** | 28 instead of 32 lane threads: 15 → 26.5 tok/s; two cores left free: 42 → 68.6; a 1 ms Windows timer: 40.6 → 68.6 (Maya; Strata's 6-core PC saw no change) | §12.3's team leaves the service, SSD, Engram and engine threads their own cores; waits spin then park and never rely on 15.6 ms sleeps; M5 sweeps the team size |
| **The RAM tier as private copies** (Maya's `--mmap-experts off`, Strata's unbuffered RAM store) beat the page cache; Maya's swap-managed tier lost to it | 12.1 against 5.75, 55.5 against 33.5, 23.2 against 18.9 tok/s; Strata 4.3-4.9 against 1.8-2.2; Maya's tier 1.4 against 2.2-2.4 ("the tier's moves lag behind") | The pinned and locked tier (§10.4) is the copy model. Its admission must follow a conversation as quickly as a page cache would; M1's trace replay measures the lag |
| **Placement by a shipped routing profile**, refreshed by use | A stale profile gave 42.3 % VRAM hits, a refreshed one 75.5 %, decode +59-64 % (Maya PR #399) | LFRU with the saved expert state (§10.2, §10.5) learns the profile; no shipped list |
| **Next-layer lookahead prefetch** | Guess precision by rank 97 / 91 / 81 / 70 / 57 / 41 %; read-ahead into Strata's store lost (4.5 against 4.7 tok/s) | Off by default (§11.3), unchanged |
| **Engram rows** requested at sampling and collected at layers 1 and 14; on Windows, synchronous `ReadFile` threads are the queue depth | A 28K prompt needs 1.36 M rows: 10 / 32 / 64 / 128 reader threads took 17.1 / 14.1 / 13.4 / 13.2 s (Maya PR #361); decode ≈ 1.4 ms per token from the page cache; unbuffered reads +2-5 % (Maya), −7.6 % (Strata's PC) | The overlapped direct-read ring (§14.4) is the deep queue; M0 measures the prompt read rate it reaches (§14.7) |
| **IQ4_NL Engram table** (ISTA's, 110.6 GB) | +0.003 top-64 KL against MXFP8 with 2-bit experts (Strata `226d73b4`); larger near lossless (§5.7) | A third Engram row format (§7.5), lossy and labelled |
| **DSpark** switched on or off once: by a load-time rule (Strata: trunk experts ≤ 1.5 × VRAM) or by an A/B (Maya) | −59 / −61 / −18 / −20 % on cards whose experts do not fit, +37 to +55 % where they do; the drafter's 6-10 GB of VRAM displaces experts | §17's per-round cost model, which also charges the pages the drafter holds (§19.1: ≈ 350 fewer pages; MXFP4 133 frames) |
| **Conversation slots**: windows, partial groups and compressed KV per conversation (≈ 17 MB per 6.5K tokens), restored in 0.1-0.5 s against 75-200 s of re-reading | Maya PR #356, Strata `cdc3a265` | The prefix cache (§18) covers it; the sizes confirm §18.2's |
| **Activations**: int8 (max/127 per 32) in decode, FP16 in prompts; one rounding shared by both lanes | The original weights through Maya's engine: KL 0.047, top-1 96.4 % against DeepSeek's reference; the shared rounding made 96 of 96 tokens placement-independent (50 of 96 before) | Supports decision b: MX8 (§9.5) uses the reference's own activation rounding on every route |
| **Windows**: WDDM holds submitted work until the host queries it (two GPUs hung); pin caps of ≈ 84-91 GB per process on a 192 GB machine | Maya #303, #384, GLM-era #147 | One GPU here; no mailbox wait depends on unsubmitted work (an M5 test); pin room per FN §19.3.7 |
| **A/B practice**: three alternating pairs, medians, a switch adopted only if faster in every pair; ≈ 5 % run-to-run spread | Strata `6c6267f3`, `a2aeb4ec` | Compatible with Infernix's speed rules |

Neither engine gives the same bits across placement, lanes or starts (both document differences);
neither stripes volumes or prices DSpark per round. What they offer that this design lacked is a
set of compact expert formats with published quality; §5.7 weighs them.

### 4.4 DeepSeek's own serving code

DeepSeek published its serving engine and API front end in October 2026, after this design's first
draft. They are the best evidence of how DeepSeek runs V4.1, read twice (2026-10-21 and 2026-10-24):

- **DeepInfer** (`deepseek-ai/DeepInfer`, Apache-2.0, at `9bd9cf24`): the offline inference engine of
  DeepSeek's production stack for V4.1 only, on SM100 and Ascend, many GPUs (expert, data and context
  parallelism). It has no `sm_120` path; it pins DeepGEMM `1fc9c669`, FlashMLA `2e5429fc` and
  DeepSelect `bfa4507d` (SM100-only) and TileKernels 2.1.0, all MIT.
- **deepseek-recipe** (`deepseek-ai/deepseek-recipe`, MIT, at `935817c6`): the API front end, which
  converts Chat Completions, Responses and Messages requests, renders V4.1 prompts and parses the
  output stream.

Nothing in them runs on this machine, and their throughput machinery (expert parallelism, EPLB, Mega
MoE, disaggregation, the KV store) has no single-GPU counterpart. What they settle:

| Topic | Finding | Where it lands |
|---|---|---|
| Prompt rule | DeepSeek replays 140 positions, not 128; the replay start is the sequence length − 140 | §2.3, §16 |
| Prompt/decode equivalence | asserted bitwise for the full rule (`test_kl.py`), with fixed splits and one attention order | §18.3, a requirement here |
| Verification room | the window ring is 134 slots and the compressor keeps 7 tokens of state | §13.1 |
| Model facts | layer 20 selects from all positions; the 2^−6 indexer factor; the RoPE table; the Engram compressed-vocabulary rules; the DSpark confidence head and its fixed 5-entry block | §2.2, §13.4, §13.7, §14.2, §17 |
| Served arithmetic | eleven places where DeepSeek's server and `model.py` part, and the ones where they agree | §20.4 |
| Kernel structure | one query per CTA with the heads as MMA rows, a canonical slot order, threshold-filter top-k, fused epilogues, Engram rows fed to the GEMM undecoded | §13, §14.6, §15 |
| Speculation | confidence-chosen verify lengths, expected-acceptance calibration, the drafter window written inside verification | §17 |
| Frontend | effort and thinking mapping, sampling (which Infernix's top-20 domain cannot reproduce), control tokens by id, DSML rules, join rules, response-format rendering, prefix identity | §21, §22 |
| Not found | expert routing statistics, Engram read rates, any rationale for 140, data beyond 128K | — |

### 4.5 Lessons from Flash-Next and the GLM design

GLM §4.2 lists the 22 Flash-Next lessons (L1-L22) measured on this machine. They hold here; these
change weight:

| Lesson | Consequence here |
|---|---|
| L2: misses are a host-DRAM bandwidth problem | The SSD set binds first (≈ 0.8 GB of SSD reads per token at the central rate, §5.3); DRAM carries landings, CPU reads and DMA (≈ 30-40 GB/s, §5.1) |
| L5, L6: promotions cost a miss's bytes; none after prefill | A promotion is 8.9-22.2 MB (MXFP4 18.8 MB, 6.8× Flash-Next's); the interval is recalibrated (§10.2) |
| L7: next-layer prediction recall is modest | Confirmed for this family by JigSawPT and hipfire; lookahead off by default (§11.3) |
| L8: verify windows add misses only for rejected drafts | Here every verify column also adds its own experts' union (§5.4); the draft length is priced per round (§17.3) |
| L13: placement invariance needs exact arithmetic | MX8 on every route (§9.5) |
| L16: short prompts pay a whole expert pass | Worse: a 200-token append reads ≈ 155 GB from the volumes (§5.5); the prefix cache and append-only processing are first-class (§16, §18) |
| L19: the divisor rule beats cost models for the CPU/link split | Kept, fed by measured costs per width (§9.7) |
| Pin room (FN §19.3.7, 2026-10-14) | The RAM tier is pinned up to ≈ 72 GiB and unpinned past it (§10.4) |

---

## 5. Performance model

The script is kept with this design's working notes; M0 commits it as
`tools/deepseek_v41/decode_model.py`. §5.1-§5.6 model the MXFP4 records (the deferred artifact) and
carry the mechanisms every build shares; §5.7 applies the same model per expert format, and its
per-expert EXL3 row is the first artifact's.

### 5.1 Per-token work at C = 1, plain decode

Let h_v be the VRAM hit rate, h_s the share read from SSD and h_r = 1 − h_v − h_s the share served
from RAM, over the 240 routed experts of a token.

| Resource | Work per token | Rate | Time |
|---|---|---|---:|
| GPU HBM: dense FP8/BF16 weights, mHC, windows, KV | 8.5 GB | ~1.55 TB/s sustained | ≈ 5.5 ms (§9.3: ≈ 6.3 ms with launches and Engram) |
| GPU HBM: hit experts | 240 h_v × 18.8 MB | same | 2.9 h_v ms |
| CPU lane: RAM- and SSD-served experts it computes | 0.35 / 0.45 / 0.60 ms per expert (§12.2) | 8 P-cores (+ E-cores) | |
| PCIe H2D: staged misses | 18.8 MB each | 27.6 GB/s | 0.68 ms each |
| SSD: 240 h_s records, each striped over the volumes | 240 h_s × 18.8 MB | 14.15 GB/s (three volumes) | 319 h_s ms |
| Host DRAM: landings, CPU reads, DMA source | ≈ (2 h_s + h_r) × 240 × 18.8 MB | ~80 GB/s, shared | not binding (≈ 35 GB/s at the central case) |

Per MoE layer the critical path is

```text
t_layer = t_pre + max(t_hits, t_host)
t_host  = min over n_c of max((n_c + k) · c_cpu, (n_r − n_c) · c_dma, k · t_record + 0.1 ms + c_cpu / 8) + 30 µs
t_token = Σ_l t_layer + t_head + t_engram + t_boundary
```

with the layer's n_r RAM misses and k SSD misses multinomial over its 6 experts (exact expectation),
n_c RAM misses computed by the CPU and the rest staged, and SSD records consumed unit by unit as
they land (§11.1, GLM §11.4), so only one unit's CPU work trails the last read. t_pre ≈ 0.10 ms (attention,
mHC and router weights, 134 MB, before routing); t_record = 1.33 ms on three volumes.

### 5.2 Hit-rate model

No routing traces of this model are recorded here; M1 records them. The evidence:

| Source | Model | Capacity | Hit rate |
|---|---|---:|---:|
| JigSawPT (§4.2), RTX 5090, VRAM cache | V4.1-Flash MXFP4 | 13 / 18 / 22 GiB of 269 GiB (4.8 / 6.7 / 8.2 %) | 59 / 62.7 / 66 % |
| FreeToken (§4.2), global LRU | V4-Flash | 11 % | 61 % |
| hipfire (§4.2) | V4-Flash MQ2 | ≈ 5.6 / 11 / 22 / 45 % (4-32 GiB of ≈ 72 GiB, estimated) | 40.5 / 59.4 / 74.4 / 86.5 % |
| DS4 PR #647, LRU | V4-Flash IQ2 | ≈ 27 % / 100 % (20 / 100 GB) | 51.5 % (220 tokens from cold) / 81 % short, 96-98 % long |
| Maya (§4.3), L40S, profile and heat swaps, agent traffic at 112-127K context | V4.1-Flash MXFP4 | 9.2 % | ≈ 40-45 % of picks from VRAM |
| Maya PR #399, the same profile stale and refreshed | V4.1-Flash EXL3 | not stated | 42.3 / 75.5 % |

This family shows **much stronger locality than GLM-5.3-Flash** at equal capacity (GLM §5.2: ≈ 35-56 %
at 11 %). Maya's agent traffic sits below hipfire's curve (≈ 40-45 % at 9.2 %, inside the pessimistic
scenario below). This machine's capacities for MXFP4 records (§19): VRAM ≈ 1,160 frames (7.5 %), RAM ≈ 4,490 slots (29.2 %)
with 88 GiB available, 37 % together (33.8 % with 79 GiB).

| Scenario | h_v | h_r | h_s | SSD reads per token | SSD bytes per token |
|---|---:|---:|---:|---:|---:|
| Optimistic | 0.58 | 0.30 | 0.12 | 29 | 0.54 GB |
| Central | 0.48 | 0.34 | 0.18 | 43 | 0.81 GB |
| Pessimistic | 0.40 | 0.34 | 0.26 | 62 | 1.17 GB |

### 5.3 Decode projection

Plain decode, tok/s (model), MXFP4 records:

| Scenario | One volume (`D:`) | Two (`D:` + `G:`) | **Three (`D:` + `E:` + `G:`)** | Four (with `C:`) | Whole bank in memory |
|---|---:|---:|---:|---:|---:|
| Optimistic | 8.9 | 15.8 | **16.9** | 20.3 | 28.7 |
| Central | 6.3 | 11.8 | **12.8** | 15.5 | 21.6 |
| Pessimistic | 4.6 | 8.9 | **9.7** | 12.1 | 16.9 |

With NVFP4 records every cell is 4-5 % lower (central, three volumes: 12.2). Central, three volumes,
by SSD share: h_s 0.08 → 17.5, 0.12 → 15.4, 0.18 → 12.8, 0.26 → 10.1, 0.35 → 8.0; by CPU cost per
expert: 0.30 ms → 13.2, 0.45 → 12.8, 0.60 → 12.3, 0.80 → 11.9. **The SSD share decides; the CPU
kernel matters little** (unlike GLM, where it was worth a drive), because the SSD term dominates and
the CPU's cost is near its DRAM floor.

**Calibration.** JigSawPT's one-card run measured 8.3 tok/s with a 72 GiB pinned tier and
125.7 GiB of RAM; this model gives 6.3 for one volume with 96 GB. The model is therefore not
optimistic for one drive; the projected gain comes from striping (2.0× from one volume to three),
record-unit consumption, the CPU lane and the smaller MXFP4 records. GLM's model was 15-45 %
optimistic against strata-glm's measured run (GLM §4.4); §25.1 carries the same risk here, and M4's
first end-to-end numbers refit the overhead terms.

### 5.4 Speculation

A verify round of W = K + 1 columns reads the union of its columns' experts per layer,
U(W) = 6 (1 + σ (W − 1)), where σ is the share of a further column's experts not already in the
round: 0.58 measured on MiniMax-M2.5 (DS4 #636), ≈ 0.875 on GLM-5.3-Flash (Maya: consecutive tokens
share ~1 of 8, GLM §4.3). It yields τ(K) = 1 + Σ_{i≤K} survival_i tokens. With vLLM's measured DSpark
survival (§4.2) and the central case, three volumes (tok/s; plain 12.8):

| σ | K = 1 | K = 2 | K = 3 | K = 5 |
|---:|---:|---:|---:|---:|
| 0.6 | 16.2 | 15.8 | 14.4 | 11.4 |
| 0.8 | 14.6 | 13.6 | 12.1 | 9.3 |
| 0.95 | 13.6 | 12.3 | 10.7 | 8.1 |

With an agentic acceptance profile (survival 0.9, 0.8, 0.7, 0.6, 0.5) and σ = 0.6-0.8, K = 2-3 gives
16.1-19.3. **Short drafts pay; DeepSeek's five-token blocks lose here.** JigSawPT's −4 % at K = 2
fits σ near 1 in this model, with one drive and an idle queue (an inference, not a measurement of σ).
§17 picks K per round, including 0.

**These projections are a greedy upper bound.** vLLM #56797's survival rates come from GSM8K at
temperature 0, but thinking-mode traffic samples at temperature 1.0 (DeepSeek's API forces it,
deepseek-recipe `options.rs`), where acceptance is lower by an amount nobody has published. §1.4's
speculative row and §5.7's DSpark figures inherit the same optimism; M7 decides the default at
temperature 1.0, top_p 1.0 with thinking on, and reports greedy beside it.

### 5.5 Prefill and time to first token

A prompt of T new tokens touches 1 − e^(−6T/384) of each encoder layer's experts, and of each decoder
layer's experts over the positions the decoder evaluates (128 under bounded replay; the exact
suffix under the full rule). Each touched expert is read once per walk span (§16.2): SSD-resident
ones cross the volumes, every expert not in VRAM crosses the link or is computed by the CPU (thin
experts only), and the GPU computes the rest at the exact routes' rates: experts ≈ 65 TFLOP/s (FN
§16.2.1's measured exact W4A16 wide route, whose two-MMA limb form MX8 shares), dense ≈ 130 TFLOP/s
(estimated from FN §19.3.8 F10's 96-169 TFLOP/s for 8-bit dense prompt routes). Popular experts are
the resident ones, so short prompts find more of theirs in memory.

| New tokens T | Experts touched | SSD bytes | Three volumes: TTFT | Four volumes: TTFT | Full rule, three volumes |
|---:|---:|---:|---:|---:|---:|
| 16 | 3,400 (22 %) | 26 GB | 2.1 s | 1.6 s | 2.1 s |
| 50 | 8,330 (54 %) | 63 GB | 4.7 s | 3.5 s | 4.7 s |
| 100 | 12,140 (79 %) | 120 GB | 8.8 s | 6.5 s | 8.8 s |
| 200 | 13,980 (91 %) | 155 GB | 11.2 s | 8.3 s | 12.1 s |
| 500 - 32,768 | 14,320 (93 %) | 161 GB | 11.7 s | 8.6-9.3 s | 13.0 s |
| 131,072 | 14,320 | 161 GB | ≈ 25 s (compute) | ≈ 25 s | ≈ 25 s |

- A fresh 32K prompt: ≈ 13.0 s under the full rule (the default), **≈ 2,500 tok/s**; ≈ 11.7 s and
  2,800 tok/s under replay (compute ≈ 6 s, hidden under the bank pass);
  128K in one walk span: ≈ 25 s, ≈ 5,200 tok/s, compute-bound by the experts' exact wide route. At
  1M, eight 128K spans and the indexer's O(T²) scoring: ≈ 3.5 minutes (estimated). The W4A4 fast
  path (§25.3 O2) would cut the expert compute about fourfold (FN §16.2.1: 1.65 against 6.24 ms per
  call) where a prompt is compute-bound.
- **Any fresh prompt above ~200 tokens costs a bank pass, ≈ 11-13 s on three volumes**, and the
  link (≈ 9 s for the non-VRAM experts) is the next floor. Short appends cost 2-9 s. DS4 measured
  2.9 / 5.4 / 8.7 s for 16 / 43 / 114-token appends with full GLM-5.3 IQ2 streamed on an M5 Max
  (`docs/SSD_STREAMING.md`), the same shape.
- The full rule adds ≈ 1.3 s from 200 tokens: the exact suffix touches ≈ 1,000 more decoder experts
  than bounded replay.

### 5.6 What moves the ceiling

| Lever | Effect (model, central) |
|---|---|
| **More RAM**: 192 GB (≈ 170 GiB of tier, ≈ 71 % of the bank with VRAM) | h_s ≈ 0.08: decode 12.8 → ≈ 17.5 tok/s; fresh-prompt TTFT ≈ 5-6 s |
| 256 GB (the whole bank in memory) | Decode ≈ 21.6 tok/s (CPU lane and link bound); TTFT link-bound ≈ 9 s |
| **`C:` as a fourth volume** (the only drive off the chipset; 80 GB free today) | Decode +21 % (15.5); TTFT −20-25 % |
| More drives behind the chipset | Nothing beyond three: DMI binds at 14.15 GB/s (§3) |
| A CPU with AVX-512 or more cores | +3 % (the CPU is near its DRAM floor) |
| **Smaller experts** (published compact builds, lossy) | 3 bits: +60 % decode, TTFT about halved; 2.2 bits: +150 %; quality and the choice in §5.7 |

The link width is fixed; nothing here is tuned to a different link.

### 5.7 Expert formats: speed against quality

On this machine the bytes per routed pick decide decode (§5.3): a smaller record puts more of the bank
in memory and makes each miss cheaper. This section weighs the published 1017 builds (surveyed on
2026-10-24 through the Hugging Face API, GGUF headers read by range and EXL3 bit maps) with §5's
model. Every row keeps the official dense weights and the FP8 Engram volume, so only the routed
experts change. A build is imported as stored; nothing is requantized at conversion.

**Quality.** No panel measures all the builds against one reference, so each figure is its source's
own, against the original model; compare within a panel only.

| Panel | Method | Builds (expert bits per pick: KL / top-1) |
|---|---|---|
| U: Unsloth | wiki1100 against DeepSeek's `model.py` in F32; each set's own dense and Engram formats | UD-IQ1_M 1.95: 0.314 / 87.1 %; UD-IQ2_XXS 2.31: 0.203 / 90.5 %; UD-Q2_K_XL 2.81: 0.138 / 91.5 %; UD-IQ3_XXS 3.30: 0.063 / 94.6 %; UD-IQ3_S 3.73: 0.032 / 96.0 %; UD-Q3_K_XL 4.00: 0.020 / 96.5 %; MXFP4 (UD-Q8_K_XL, the runtime's own floor) 4.25: 0.005 / 98.5 % |
| S: sudoingX | 254,846 positions, 65 % code, against `model.py` | EXL3 with per-expert bits (sudoingx 2.8, HeNryous 2.75; 3.10 bits per pick): 0.020 / 96.0 %; uniform EXL3 2.9: 0.030 / 94.8 % |
| K: kyuz0 short100 | full vocabulary, against `model.py` | diffbot EXL3 K3, 3.01: 0.034 / 95.4 %; DS4 Q2, 2.25: 0.130 / 90.0 % |
| D: diffbot | its neutral / code / agentic corpora against its own run of the original (floor 0.008 / 0.005 / 0.005) | EXL3 K3: 0.052 / 0.031 / 0.011 |
| L: Lucebox | top-256 KL, fineweb-edu | UD-Q2_K_XL: 0.114 / 85.7 %; Maya-S, 2.49: 0.195 / 82.0 %; dsx-t16 trellis, 1.92: 0.196 / 80.9 %; UD-IQ1_M: 0.275 / 78.8 % |
| M: Maya | against DeepSeek's reference (its engine's floor 0.047 / 96.4 %) | Maya-S, 2.49: 0.120 / 92.6 %, NLL +3.2 %; Maya-XS, 1.56: 0.276 / 87.5 %, NLL +8.7 % |
| St: Strata | top-64 renormalized KL against its own UD-Q8_K_XL run; ISTA's task average | ISTA IQ2_XS, 2.19: 0.107, perplexity +7.3 % (code +3.2 %); IQ2_XXS, 1.94: 0.170, +15.5 %, tasks 100 %; IQ1_M, 1.69: 0.217, +21 % (plain prose +53 %), tasks 98.7 % |

The Engram format matters once the experts are near-lossless. With Q8 experts, Unsloth's plot puts
an IQ4_NL table at ≈ 0.018 against 0.005 for MXFP8 (read off the chart), and TrellisMX measured a
4-bit Engram alone at 0.045 on its panel; with 2-bit experts the difference is ≈ 0.003-0.007
(Strata, dsx). The comparison below therefore holds the Engram rows at FP8, so that only the experts
differ; the first artifact's own table is chosen in the decision further down (IQ4_NL), and it does
not change the speeds.

**Speed** (model; three volumes; the range is the SSD-miss rate × 1.4 to × 0.7; prompts under the
full rule, the default; replay would take 0.3-1.3 s off a fresh 1-4K prompt):

| Build (1017) | Bits per pick | MB per expert | Bank in memory | SSD reads per token | Plain decode, tok/s | Four volumes | TTFT: 50-token append / fresh 1-4K | 32K prompt, tok/s | Quality |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| MXFP4, this design | 4.25 | 18.8 | 37 % | 41 | 13.1 (10.2-16.8) | 15.7 | 4.9 / 13.1 s | 2,490 | lossless |
| UD-IQ3_S experts | 3.73 | 16.5 | 42 % | 35 | 15.5 (12.2-19.7) | 18.2 | 3.5 / 10.6 s | 3,080 | U 0.032 |
| UD-IQ3_XXS experts | 3.30 | 14.6 | 47 % | 30 | 18.5 (14.7-23.2) | 21.4 | 2.4 / 8.6 s | 3,810 | U 0.063 |
| **EXL3, per-expert bits (HeNryous 2.75: the first artifact)** | 3.10 | 13.7 | 51 % | 28 | **20.0** (16.0-25.0) | 22.7 | 2.0 / 7.6 s | 3,040 | S 0.020 |
| EXL3 K3 (diffbot) | 3.01 | 13.3 | 52 % | 27 | 20.9 (16.7-26.1) | 23.6 | 1.9 / 7.2 s | 3,040 | K 0.034; D 0.052 / 0.031 / 0.011 |
| UD-Q2_K_XL experts | 2.81 | 12.4 | 56 % | 24 | 23.2 (18.7-28.8) | 26.3 | 1.8 / 6.4 s | 5,080 | U 0.138; L 0.114 |
| Maya-S experts | 2.49 | 11.0 | 63 % | 20 | 27.6 (22.5-33.9) | 30.6 | 1.6 / 5.6 s | 5,080 | M 0.120; L 0.195 |
| ISTA GSQ-RCO IQ2_XS | 2.19 | 9.7 | 71 % | 15 | 32.9 (27.0-39.9) | 35.6 | 1.4 / 4.9 s | 5,080 | St 0.107 |
| ISTA GSQ-RCO IQ2_XXS | 1.94 | 8.6 | 81 % | 10 | 38.9 (32.2-46.7) | 41.1 | 1.3 / 4.3 s | 5,080 | St 0.170 |
| ISTA GSQ-RCO IQ1_M | 1.69 | 7.5 | 93 % | 5 | 46.6 (38.8-55.6) | 47.9 | 1.2 / 3.7 s | 5,080 | St 0.217 |

Model inputs (`quant_model.py`, kept beside §5's script):

- hipfire's capacity curve (§5.2) for every build, so a format changes only its bytes and its CPU
  cost; for MXFP4 it reproduces the central scenario (h_v 0.49, h_s 0.17);
- the CPU's cost per expert at an estimated 25-42 GB/s of weight bytes (halving it costs ≤ 10 %);
- prompt compute at 65 TFLOP/s for MX8 and the exact GGUF routes, and 33 for EXL3's exact E3A16
  (four int8 products per weight against two, GLM §9.5);
- a per-expert-bit build at its usage-weighted bytes per pick. Its hottest experts are its widest
  (mostly K4), so VRAM holds about 8 % of the experts rather than the 10 % this assumes: slightly
  optimistic, inside the range.

Cross-check: with the capacities of Maya's RTX 3090 run (12.6 GB of experts in VRAM, 96 GB in RAM)
and one 7 GB/s drive (the run does not state its drive), the model gives Maya-S 18.0 tok/s; the run
measured 16.9-18.1 (§4.3). With a 5 GB/s drive the model gives 14.7.

**What the tables say.**

1. Decode follows the share of the bank that fits in ≈ 100 GiB of VRAM and RAM: 37 % at 4.25 bits,
   52 % at 3, 71 % at 2.2. Three bits cut the SSD reads per token from 41 to 27 (+60 % decode); 2.2
   bits cut them to 15 (+150 %).
2. Quality falls slowly to about 3 bits per pick and quickly below. On Unsloth's panel KL doubles from
   3.30 to 2.81 bits and top-1 drops 3 points. Per-expert bit plans and trellis codes move that knee
   lower (S: 0.020 at 3.10 bits used against 0.030 for uniform EXL3 2.9).
3. A fresh prompt stops being a bank pass: TTFT for 1-4K tokens nearly halves at 3 bits. Long prompts
   become compute-bound, where EXL3's exact arithmetic is the slow one (3,040 against 3,800-5,080
   tok/s at 32K, still above MXFP4's 2,490).
4. DSpark is more likely to pay with compact experts. The model gives +37-48 % at 3 bits (+36-46 %
   for the per-expert build) for acceptance 0.7-0.8 (MXFP4: +21-28 %), before the drafter's pages
   are charged. That matches
   Strata's and Maya's gains where the experts fit and losses where they do not (§4.3); §17's cost
   model decides per round.

**Decision f (2026-10-24): EXL3 with a bit width per expert, built for from the start.**

- **The first artifact**: HeNryous's 1017 pack. Each expert has one width: K2 / K3 / K4 / K5 =
  6,271 / 6,726 / 2,295 / 68 experts (2.75 bits mean, 3.10 per pick), `mul1` codebook, the plan in
  `exl3_moe_plan.json`; the mix varies by layer (one layer holds 374 K2 experts, another 153 K4 or
  K5). Published: KL 0.0201, top-1 95.96 % (S), against 0.030 for uniform EXL3 2.9 on the same panel
  and 0.097 against 0.114 on instruction-following replies. sudoingx's 2.8 pack (5,709 / 7,056 /
  2,553 / 42, KL 0.0198) is the same pipeline and an accepted second source. With the official dense
  tensors and DSpark dense weights from DeepSeek's release, the pack's 2-bit drafter experts and the
  IQ4_NL Engram table (below), the model gives ≈ 20 tok/s plain decode (16-25), 22.7 with four
  volumes, 7.6 s for a fresh 1-4K prompt, 2.0 s for a short append and 3,040 tok/s at 32K. The
  Engram format does not change these: a row is one 4 KiB read whatever its format.
- **The analysis behind it.** The first recommendation was diffbot's uniform K3 pack (≈ 21 tok/s;
  K 0.034, D 0.052 / 0.031 / 0.011, measured with the released FP8 dense weights), with per-expert
  bits as a later step. Per-expert bits cost ≈ 4 % of decode in the model (the hot experts carry
  more bits: 3.10 bits per pick against 3.01) and ≈ 5-6.5k more lines, for KL 0.020 against ≈ 0.03
  on the packs' own panel (diffbot's K3 was not measured on it) and 17 GB less disk. Built from the
  start, the page pool replaces the fixed frames once instead of being retrofitted, and a uniform
  pack becomes its one-width case.
- **Deferred**: the lossless MXFP4 artifact (≈ 13 tok/s; M11).
- **Not chosen**: ISTA's GSQ-RCO IQ2_XS as a speed build (≈ 33 tok/s, 4.9 s for a fresh prompt;
  perplexity +7.3 %, code +3.2 %, agent +2.2 %; it would reuse FN §16.2.2's exact GGUF arithmetic
  with two new formats, `ggml_iq2_xxs` and `ggml_iq2_xs`) stays option O6; builds below 2 bits per
  pick (KL ≥ 0.17; plain-prose perplexity +15-53 %); NVFP4 records (the MXFP4 values in 5.9 % more
  bytes); Q4_K or Q4_K_M requantizations (more bytes than MXFP4, and lossy).

**What the first artifact needs** (estimated):

| Piece | Work | Size |
|---|---|---:|
| EXL3 codec and routes | The GLM design's E3A16 codec, narrow and wide routes and CPU kernel, built here for K = 2-6 (GLM reuses them), at H = 5,120 and I = 2,304 and GLM's geometry. One MoE call mixes widths, so the routes take K per job; MX8 → E3A16 chaining (§9.5); E3A16's exact oracle qualifies every K | ≈ 5-6k lines |
| Records and volumes | `exl3_expert_v1` records in pages (§7.3); one bank per width (the volume header already holds several banks) and an expert → (bank, index) table | ≈ 0.5-1k |
| **Page pool** (the largest new piece) | Frames, RAM slots, staging, landing and bounce slots all assume one record size in the shared offload component. Per-expert widths need **pages** of 4,456,448 B (1,088 × 4 KiB): a K2 record takes 2 pages, K3 3, K4 4 and K5 5, with < 0.6 % padding; records addressed through a per-record page list, so the pool never fragments; four 1.06 MiB stripe units per page; eviction by value per page. An artifact with one record size uses one page per record, which is today's frame pool, so Qwen4Exp's tests stay unchanged | ≈ 2.5-3.5k |
| Policy | LFRU by value per byte, re-chosen by M1's trace replay | ≈ 0.3k and the replay |
| Recipe and converter | Reads the plan and the pack's expert tensors (its dense weights are EXL3 K5; the official FP8 ones are used instead), range-reads the official dense tensors, copies the IQ4_NL table's rows into the Engram volume (an FP8 or MXFP4 volume from the official rows remains possible) | ≈ 1-1.5k |

**The Engram table** (owner, 2026-10-24): the FP8 rows are 202.8 GB as released and a 209.7 GB
volume, so the first artifact plans the published **IQ4_NL** table of 1017 (110.6 GB, a 112.4 GB
volume). The other published 1017 tables are Unsloth's Q6_K (161.3 GB) and Q5_K (135.2 GB) and
LibertAIDAI's MXFP4 (104.5 GB). What 4 bits cost depends on how good the experts are: +0.003 top-64
KL with 2-bit experts (Strata), +0.007 on Lucebox's panel with ≈ 2-bit experts (dsx), and with
near-lossless Q8 experts ≈ 0.018 against 0.005 for the FP8 table on Unsloth's chart, where Q5_K and
Q6_K sit near 0.008 (read off the chart, not published numbers); TrellisMX measured its own 4-bit
trellis table alone at 0.045 on its panel. Against this artifact's ≈ 0.020 from its experts, the
IQ4_NL table may add a noticeable share, so §20.3 measures it separately; a Q5_K, Q6_K or FP8 volume
replaces it per run (`--engram-volume`, §14.5) without reconverting the experts.

Disk: ≈ 192 GB of expert volumes (the 188 GB main bank and a 3.4 GB drafter bank), 112.4 GB of
IQ4_NL Engram volume and ≈ 11 GB of main artifact, **≈ 315 GB** (≈ 412 GB with FP8 rows). Download
≈ 313 GB: the pack's expert tensors (≈ 191 GB), the IQ4_NL table (110.6 GB) and the official dense
and DSpark dense tensors by range read (≈ 11 GB). The quality report against the original model
needs reference logits, which §20.2 computes by streaming the official shards one at a time (510 GB
read, FP8 Engram shards included, ≈ 15 GB on disk at once).

**A hybrid (option O11).** The official MXFP4 records on a profile's hottest 10-20 % of experts and
ISTA's IQ2_XS records on the rest model at 22-25 tok/s, with the 2-bit error on 28-43 % of picks. It
is measured nowhere, and a profile that misses a workload moves the error onto that workload's hottest
experts, so it stays an option.

---

## 6. Design overview

```text
 ┌───────────────────────────────────── RTX 5090, 32 GB ─────────────────────────────────────┐
 │ dense FP8 / BF16 8.0 GiB · windows 2.9 MB/lane · compressed KV + index keys 890 B/token   │
 │ decode workspace · mailboxes · residency[43 × 384] · page lists · land_seq[record]        │
 │ ┌── page pool: ≈ 4,900 pages of 4.25 MiB (≈ 20.3 GiB), 2-5 per record, elastic ──────┐   │
 │ │ cached experts │ staging │ DSpark experts (bounded) │ KV pages + loans │ walk lease  │   │
 │ └────────────────────────────────────────────────────────────────────────────────────┘   │
 └──────▲───────────────────────▲─────────────────────────▲───────────────────────────────────┘
        │ copy engine           │ mapped mailboxes        │ copy engine (landed records)
 ┌──────┴───────────────────────┴─────────────────────────┴─────────┐   ┌──────────────────────┐
 │ 96 GB host                                                        │   │ expert volume 0      │
 │  RAM tier: ≈ 17,300 pinned pages (≈ 72 GiB) + unpinned pages      │◄──┤ expert volume 1      │
 │  embedding (pinned) · Engram row cache · landing pages            │   │ expert volume 2      │
 │  transfer agent: VRAM policy, promotions, staging DMA             │   │  4 units per page    │
 │  SSD agent: per-volume IOCP queues; Engram reader ring            │◄──┤ Engram volume        │
 │  CPU expert team: MX8 + E3A16 exact, from RAM or landing pages    │   │  FP8, MXFP4 or IQ4_NL│
 │  engine worker: tokens, Engram hashes, rounds                     │   └──────────────────────┘
 └───────────────────────────────────────────────────────────────────┘
```

One decode round at C = 1:

```text
engine: sample t → Engram hashes for t (48 rows) → row reads (cache, else the Engram volume)
embed(host row) → 4 streams
per layer l (40):
  Engram (l = 1, 14) → collapse (pre from the previous sublayer) → norm → attention:
     q / window entry (MX8) → [Full: compressor + index keys] → [Full/Reindex: indexer → top-512]
     → sparse attention (window 128 + selected) with sink → inverse RoPE → wo_a / wo_b (MX8)
     ‖ mixes from the stream (post, comb, next pre: off the critical path)
  residual update → collapse → norm → MoE:
     router(l) → residency → GPU jobs | CPU jobs | SSD jobs
     shared expert ‖ routed experts (MX8 → E3A16 narrow, K per job) ‖ CPU results ‖ landed records
     → combine (FP32, ascending id)
     ‖ mixes for the next sublayer
head: collapse with the last pre → norm → head (FP32 logits) → sample → DSpark draft (if chosen)
```

The ideas that carry the design:

- **Exact arithmetic makes placement free** (FN §5, L13): every FP8 linear computes DeepSeek's
  MXFP8 product exactly and every EXL3 expert applies E3A16 to the same MX8 inputs, so a VRAM page,
  a staging slot, the CPU from a RAM page or a landing page give the same bits, and policy may move
  work anywhere (§9.5).
- **The SSDs are the bottleneck; everything is sized to read fewer, smaller, fuller records**:
  per-expert EXL3 records (3.10 bits per pick against MXFP4's 4.25), three volumes striped at
  1.06 MiB units (§11), units consumed as they land, LFRU by value per page with about half the bank
  in memory, the prefix cache, short drafts only (§17).
- **Pages make record size free**: one pool of 4.25 MiB pages serves every tier, so a record of any
  width takes its pages and nothing fragments (§7.3, §10).
- **Three tiers, exclusive, one policy owner per boundary** (GLM §6): the transfer agent owns
  VRAM↔RAM, the SSD agent RAM↔SSD; the device reads residency words.
- **The model's own compressed KV is tiny**, so long context costs almost nothing and prefix
  snapshots are a few MB: the prefix cache can keep whole conversations (§18).
- **Prompt rule as a declared model mode**: the exact full model by default, DeepSeek's bounded
  replay on request; both are deterministic, request-invariant and computed with the same kernels
  (§16).

---

## 7. Artifacts, records, volumes and conversion

### 7.1 Sources and recipes

**The EXL3 recipe** (`deepseek_v4_1_flash_exl3`, the first artifact) takes three sources at pinned
revisions:

| Source | Taken from it | Checks |
|---|---|---|
| **`HeNryous/DeepSeek-V4.1-Flash-1017-EXL3-2.75bpw-DGX-Spark`** (default) or `sudoingx/DeepSeek-V4.1-Flash-1017-EXL3-2.8bpw` | `exl3_moe_plan.json`; every routed expert's EXL3 tensors (`layers.N.ffn.experts.E.w1/w3/w2`: trellis codes, `suh`, `svh`) and the drafter's (`mtp.0-2.ffn.experts.*`), copied verbatim | the plan's width equals each tensor's; codebook `mul1` and the producing ExLlamaV3 version are ones M0 qualified; the base revision equals the official source's |
| **`deepseek-ai/DeepSeek-V4.1-Flash-1017`** | every dense tensor, the DSpark dense weights, the Engram hash parameters, the tokenizer and encoding; the FP8 Engram tables only if an FP8 or MXFP4 volume is made; never its routed experts | as the MXFP4 recipe below; the pack's own dense weights (EXL3) are not read |
| **The IQ4_NL Engram table**: `ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF` (`DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf`) or `unsloth/DeepSeek-V4.1-Flash-1017-GGUF` (`engram/…-engram-IQ4_NL.gguf_file`, one tensor of 768,022,850 rows that the converter splits at layer 1's 384,006,168) | the rows, copied verbatim into the Engram volume (§7.5) | M0: the two files' row data compared (their headers differ); sampled rows decoded and compared with the official 1017 FP8 rows of the same index, which must agree far better than with September's; row counts and order against the official primes and offsets |

A uniform EXL3 pack (diffbot's K3) is the same recipe with one width and no plan file. M0 pins the
pack's tensor names, tile order and Hadamard scale against ExLlamaV3's code (GLM §9.5's list).

**The MXFP4 recipe** (`deepseek_v4_1_flash`, deferred to M11) imports any of these sources; M0 pins
each revision.

| Source | Routed experts | Engram tables | Notes |
|---|---|---|---|
| **`deepseek-ai/DeepSeek-V4.1-Flash-1017`** (default, official entry `deepseek_v4_1_flash_1017`) | MXFP4 | FP8 | weights-only update of September |
| `deepseek-ai/DeepSeek-V4.1-Flash` (official entry `deepseek_v4_1_flash_0910`) | MXFP4 | FP8 | |
| `nvidia/DeepSeek-V4.1-Flash-NVFP4` (official entry `deepseek_v4_1_flash_nvfp4_nvidia`) | NVFP4, lossless re-encoding, plus `input_scale` | MXFP4 (current) or FP8 (revision `3431dde3`) | September weights |
| `LibertAIDAI/DeepSeek-V4.1-Flash-1017-NVFP4` | NVFP4, lossless re-encoding of 1017, plus `input_scale` | MXFP4 | community; accepted, not an official entry |

**Every source yields the same expert values**, and the recipe proves it while converting:

- An MXFP4 source is copied: codes and UE8M0 scales verbatim, negative zero (code 0x8)
  canonicalized to 0x0 (the same value).
- An NVFP4 source is recovered to MXFP4 block by block. A 32-weight block (two NVFP4 16-blocks with
  E4M3 scales s_a, s_b and the matrix's FP32 `weight_scale_2` g) decodes to v_j = e2m1(c_j) ·
  e4m3(s) · g. The converter takes k = ⌊log₂ max_j |v_j|⌋ − 2 (the OCP MXFP4 scale rule) and accepts
  the block only if every v_j / 2^k is an E2M1 value; the codes are then re-derived and the block
  stored as MXFP4. M0 checks on sampled official blocks that DeepSeek's own scales follow the same
  rule (all-zero blocks aside), so a September artifact made from NVIDIA's export is byte-identical
  to one made from DeepSeek's release; M4 checks that equality on the whole bank.
- If any block of a bank fails the check (a lossy third-party NVFP4 build), the whole bank is stored
  as NVFP4 records instead (§7.3) and the conversion report says so.
- `input_scale` values of an NVFP4 source are kept in the bank's side table; only the W4A4 fast-path
  candidate (§25.3 O2) would use them.

### 7.2 Tensor classes

| Tensor class | Stored (official) | Bytes | Infernix form |
|---|---|---:|---|
| Routed experts, 40 × 384 (`w1` gate, `w3` up, `w2` down) | EXL3 pack: trellis codes of K = 2-5 bits per weight with FP16 `suh` / `svh`, one K per expert (official: MXFP4, packed I8 codes [2,304 × 2,560] / [5,120 × 1,152], E8M0 scales [2,304 × 160] / [5,120 × 72]) | 187.6 GB (288.8 GB) | `exl3_expert_v1` records in pages (§7.3), one bank per width in the expert volumes (§7.4); MXFP4: `mxfp4_expert_v1` records (M11) |
| DSpark experts, 3 × 128 | EXL3 pack: K = 2 (official: MXFP4) | 3.4 GB (7.22 GB) | `exl3_expert_v1` records, DSpark bank |
| Attention `wq_a`, `wq_b`, `wkv`, `wo_a`, `wo_b`; shared experts; indexer `wq_b`; Engram `wkv`; DSpark attention, shared experts and `main_proj` | FP8 E4M3 with E8M0 per 32 × 32 | 7.5 GB | new Linear format `fp8_e4m3fn_b32_e8m0` with MX8 activations (§9.5), device-resident; `wo_a` applied per group (8 diagonal blocks of [1,024 × 4,096]) |
| Router `gate.weight` (BF16 [384 × 5,120]); `bias`, `bias_vl` (FP32) | | 0.16 GB | direct; FP32 logits |
| Compressor `wkv`, `wgate` (BF16), indexer `weights_proj`, `wk` (BF16), norms | | 0.08 GB | direct; FP32 (ratio 2) and BF16 (ratio 1, indexer) products |
| mHC `fn`, `base`, `scale`; `attn_sink` (FP32) | | 0.16 GB | direct |
| Head (BF16) | | 1.32 GB | `projection_fp32` (FP32 logits), device |
| Token embedding (BF16) | | 1.32 GB | **host-resident**, pinned (one row per token) |
| Engram tables | FP8 rows with E8M0 per 32 | 202.8 GB | Engram volume, FP8 or MXFP4 rows (§7.5) |
| Engram `q_weight`, `k_weight` (BF16 [4 × 5,120]); DSpark Markov head, confidence head, norms (BF16) | | 0.14 GB | direct |
| Vision tower and aligner (BF16) | | 0.97 GB | stored, not loaded (§1.1) |

The converter refuses any tensor outside this inventory, any dense FP8 block whose scale is not a
finite power of two, NaN codes, and a config that differs from the pinned one in any field.

### 7.3 Expert records

**`exl3_expert_v1`** (the first artifact; to be defined byte-exactly in
[storage layouts](storage-layouts.md)). One expert of width K ∈ {2, 3, 4, 5}, read in place by both
processors (FN §6.2's principle):

- **Contents**: first the three matrices' FP16 sign-scale vectors (gate and up `suh` [5,120] and
  `svh` [2,304], down `suh` [2,304] and `svh` [5,120]: 44,544 B), so a job's input rotations start
  with the first unit; then the gate and up trellis tiles interleaved by 16-channel groups (16 × 16
  tiles of 256 weights, 32 · K bytes each, the pack's bits verbatim), then down. M3 fixes the tile
  order within a group for the narrow and wide routes, as GLM §7.2 does.
- **Sizes**: 4,423,680 · K + 44,544 B: 8,891,904 (K2), 13,315,584 (K3), 17,739,264 (K4) and
  22,162,944 B (K5).
- **Pages**: a record of width K takes **K pages of 4,456,448 B** (1,088 × 4 KiB), with 21-119 KB of
  padding (≤ 0.54 %). The pages need not be adjacent: every tier names a record by its page list,
  so records of any width share one pool and it never fragments (§10.1).
- **Stripe units**: four per page, 1,114,112 B (272 × 4 KiB) each, so a record is 8-20 units;
  gate/up units come first, and no tile crosses a unit boundary (≤ 159 B of padding per unit), so a
  landed unit is a whole set of work items.
- An artifact with one record format, such as the deferred MXFP4 one or Qwen4Exp's, uses pages of
  its record size: one page per record, which is today's frame.

**`mxfp4_expert_v1`** (the deferred MXFP4 artifact, M11; to be defined byte-exactly in
[storage layouts](storage-layouts.md)). Both processors read a record in place (FN §6.2's
principle): the GPU from a frame or a staging slot, the CPU from a RAM slot, a landing slot or an
unpinned record.

- **Matrices**: gate/up first, rows interleaved by groups of 16 (16 gate rows, then the same 16
  channels' up rows), then down.
- **Units of work**: for each 16-row group g and 32-element K block b, one 272-byte tile: 256 bytes
  of codes (16 rows × 32 nibbles), then 16 bytes of UE8M0 scales (row i at byte i). Tiles are
  g-major, then b ascending. The intra-tile code order is chosen in M3 so a warp's loads land in
  `mma.sync.m16n8k32` operand order and a CPU vector load covers one row group's 32 weights
  (FN §6.2's quad scheme at 32-element blocks); no tensor-core layout leaks into the record.
- **Sizes**: gate/up 288 row groups × 160 blocks × 272 B = 12,533,760 B; down 320 × 72 × 272 B =
  6,266,880 B; **18,800,640 B per record**, the official bytes rearranged.
- **Stripe units**: 12 per record, 8 for gate/up (each 36 row groups: 288 channels' gate and up rows)
  and 4 for down (each 80 row groups: 1,280 output rows). A unit holds 1,566,720 B, padded to
  1,568,768 B (383 × 4 KiB) so every unit is a direct-I/O read: **18,825,216 B (17.95 MiB) per padded
  record**, +0.13 %. RAM slots and VRAM frames hold the padded record, so a landed record becomes a
  slot without moving bytes.

**`nvfp4_expert_rg16_v1`** (FN §6.2, generalized to H = 5,120, I = 2,304) is the fallback: 144-byte
units of 16 rows × 16 K with E4M3 scales, 19,906,560 B per record (4,860 × 4 KiB), 12 units of
1,658,880 B; the matrices' `weight_scale_2` multipliers sit in the record's tail as in FN.

**Decision c, made by measurement.** The two records hold the same values. MXFP4 is 5.6 % smaller,
which §5.3 turns into +4-5 % decode and a 5.6 % shorter bank pass. Its kernels should also be
cheaper: a block scale is an exponent (a shift) for every 32 weights instead of an E4M3 value
multiplied into an int64 for every 16 (§9.5), and the scale bytes halve. M3 times the narrow, wide
and CPU routes on both records at DeepSeek's shapes; M5 runs both artifacts end to end. MXFP4 stays
the default unless it is slower end to end beyond run-to-run spread, in which case the recipe writes
NVFP4 records and §25.2 records why.

### 7.4 Expert volumes

The routed banks live outside the `.infernix` file set, in **N expert volumes**
(`<name>.xv-<i>-of-<N>`, 1 ≤ N ≤ 4), each a plain file for its own drive, exactly as GLM §7.3
defines them: a 4 KiB header (magic, artifact id, volume index and count, weight vector, banks with
record bytes, unit size and record count, CRC32C), weighted placement of unit j of record r on
volume P[(j + r) mod 12], computable offsets and no per-record table.

- **Banks** (EXL3 artifact): one bank per width, K2 / K3 / K4 / K5 with 6,271 / 6,726 / 2,295 / 68
  records of 2-5 pages, numbered layer-major and expert-minor within each, and a DSpark bank of 384
  K2 records; the artifact holds the expert → (bank, index) table (15,744 entries). Unit j of a
  record goes to volume P[(j + r) mod 12] as before, so an 8-unit record spreads over every volume.
- **Sizes** (EXL3 artifact): 188.2 GB for the main banks and 3.4 GB for the DSpark bank, **191.7 GB**:
  63.9 GB per volume at w = (4, 4, 4).
- **Banks and sizes of the MXFP4 artifact (M11)**: the main bank (15,360 records) and the DSpark bank
  (384 records); 15,744 padded records = **296.4 GB**: 98.8 GB per volume at w = (4, 4, 4). NVFP4
  fallback: 313.0 GB.
- **Writers**: the converter (§7.6) and `python -m tools.artifact.restripe` (GLM §7.3), which
  re-stripes existing volumes for another N or weight vector without the source. Runtime never
  writes a volume.

### 7.5 Engram volumes

The Engram tables live in their own volume, `<name>.engram`, in the role Flash-Next's `.ngram`
volume has (FN §6.3, §12.2):

| Row format | Row | Rows per 4 KiB block | Volume | Source |
|---|---:|---:|---:|---|
| `fp8_e4m3_e8m0_32` | 264 B (256 codes, 8 scales) | 15 | **209.7 GB** | the official release; NVIDIA at `3431dde3` |
| `mxfp4_e8m0_32` | 136 B (128 code bytes, 8 scales) | 30 | **104.9 GB** | NVIDIA's current revision; LibertAIDAI 1017; or the converter (`--engram-format mxfp4`) |
| `ggml_iq4_nl` | 144 B (8 IQ4_NL blocks of 18 B) | 28 | **112.4 GB** | ISTA's or Unsloth's 1017 table (110.6 GB as stored), rows copied verbatim: **the first artifact's planned table** |
| `ggml_q5_k`, `ggml_q6_k` (option O4) | 176 B / 210 B (one 256-value block) | 23 / 19 | 136.8 / 165.6 GB | Unsloth's 1017 tables (135.2 / 161.3 GB as stored), rows copied verbatim |

- Rows never straddle a block; each table's rows are contiguous in row order, so one row is one
  4 KiB direct read and prompt reads of neighbouring rows coalesce.
- The header names the format, both tables' row counts and their 24 primes each, and the artifact id
  it was made for.
- The converter makes an MXFP4 volume from FP8 rows with ModelOpt's MXFP4 rule (UE8M0 scale
  ⌊log₂ amax⌋ − 2 per 32, E2M1 round-to-nearest-even; M0 reads ModelOpt's code and checks it against
  NVIDIA's rows of the September table, which must match byte for byte).
- **Sharing**: the 1017 tables differ from September's (§2.6), so a volume belongs to one checkpoint.
  Every artifact of that checkpoint (the MXFP4 build and a compact build, §5.7) binds the same volume
  with `--engram-reuse`, as Flash-Next's derivatives bind NVIDIA's n-gram volume.
- The IQ4_NL format is lossy against the FP8 tables: Strata measured +0.003 top-64 KL with 2-bit
  experts, and near-lossless experts show more (§5.7). The first artifact plans it to save 97 GB
  (owner, 2026-10-24); §20.3 measures its share of the artifact's KL, and the Q5_K, Q6_K or FP8 rows
  replace it per run if that share matters.
- Runtime selection follows decision d (§14.5).

### 7.6 Conversion under the disk limits (decision e)

The official source is 510.3 GB in 48 shards: 46 of at most 7.41 GB and the two Engram shards
(47 and 48) of 101.54 GB each, one table per shard (Hugging Face API, 2026-10-24; this section
earlier said all 48 were at most 7.41 GB). NVIDIA's is 429.0 GB. The source and the outputs cannot
coexist on these disks, so the conversion route is **shard by shard**, and its plan is computed from
the free space at the time it runs; the owner frees the space when the artifact work starts.

- **The EXL3 artifact** reads ≈ 313 GB: the pack's expert tensors (≈ 191 GB) by range read from its
  shards, the official dense and DSpark dense tensors (≈ 11 GB) by range read, and the IQ4_NL table
  (110.6 GB) streamed by range read straight into the Engram volume. It writes ≈ 11 GB of main
  artifact (dense 8.5, embedding 1.3, DSpark dense 0.7, vision 1.0), 191.7 GB of expert volumes and
  112.4 GB of IQ4_NL Engram volume: **≈ 315 GB** (with an FP8 volume instead, the two official Engram
  shards are streamed the same way: ≈ 405 GB read, ≈ 412 GB written).
- **The MXFP4 artifact (M11)** reads the 510 GB release and writes ≈ 11 GB of main artifact,
  296.4 GB of expert volumes and the same Engram volume: **≈ 517 GB**. One Engram volume serves both
  artifacts of 1017 (§7.5).

Steps (both artifacts):

1. **Plan.** Read the index and every shard header by HTTP range requests (≈ 15 MB of metadata);
   compute every output offset (expert units to volume offsets by §7.4, Engram rows to blocks,
   dense tensors to the main artifact); read each target drive's free space; choose the volume
   weight vector from free space and measured bandwidth (§11.4), the Engram format (FP8 unless the
   owner chose MXFP4, or space requires it, which the planner states), and a staging directory with
   room for two shards. If the outputs plus two shards do not fit the drives named, it refuses and
   names the shortfall per drive.
2. **Allocate.** Create every output at its final size; process shards in index order, which is
   layer order, so each volume is written front to back (no zero-filled gaps on NTFS).
3. **Per shard**: download it into staging (resumable; the LFS SHA-256 is checked), convert its
   tensors (copy EXL3 expert tensors or copy or recover MXFP4 blocks into their units, pack Engram
   rows, write dense tensors), read every written range back and compare it with the source bytes in
   memory, record the shard and the CRC32C of each written unit in a journal, then delete the shard.
   One shard downloads while the previous converts: peak staging ≈ 15 GB, peak RAM ≈ 4 GB (the
   converter runs at below-normal priority and reads the machine's free RAM before starting, since
   other sessions may need it). The two 101.5 GB Engram shards are never staged: their rows stream
   by range read into the Engram volume in 1 GB pieces, the SHA-256 is computed over the stream, and
   the journal records each piece. Tensors needed from a shard without the rest of it (the official
   dense tensors for the EXL3 artifact) are range-read, as `tools/convert/fetch_tensors.py` does for
   Flash-Next.
4. **Resume**: an interrupted run restarts from the journal at the first incomplete shard.
5. **Verify**: `tools.deepseek_v41.verify_artifact` checks headers, sizes, the journal's CRC32Cs on
   every unit and, with the source reachable again, sampled tensors by range read.

**Disk plan for the EXL3 artifact** (an example; step 1 recomputes it from the free space when the
conversion runs):

| Item | Size | Drive |
|---|---:|---|
| Expert volumes 0-2, w = (4, 4, 4) | 3 × 63.9 GB | three of `D:`, `E:`, `G:` (and `C:` as a fourth, §11.4) |
| Engram volume, IQ4_NL | 112.4 GB (FP8: 209.7 GB) | one drive, or striped (§14.7) |
| Main artifact | ≈ 11 GB | any |
| Shard staging (two shards) | ≈ 15 GB | any |

The 2026-10-18 plan for the MXFP4 artifact put 98.8 GB of expert volume on each of `D:`, `E:` and
`G:`, the Engram volume and main artifact on `E:`, and the staging on `D:`. If a drive has less room
when the conversion runs, the planner weights units away from it (for example w = (5, 4, 3), ≈ 6 %
slower records, §11.4); with less room overall it proposes a 4-bit Engram volume (MXFP4 −105 GB,
IQ4_NL −97 GB). Downloading ≈ 313 GB (EXL3 with the IQ4_NL table) or 510 GB (MXFP4) needs the
owner's go-ahead when the conversion is scheduled; at 1 Gb/s that is ≈ 42 or 70 minutes. The
quality report's reference logits (§20.2) stream the 510 GB release once more, without keeping it.

**Space on 2026-10-24** (383 GB free, §3; staging ≈ 15 GB on top of each):

| Build | FP8 Engram | 4-bit Engram (IQ4_NL / MXFP4) |
|---|---:|---:|
| EXL3 with per-expert bits (the first artifact) | 412 GB | 315 / 308 GB |
| EXL3 K3 (§5.7) | 433 GB | 335 / 328 GB |
| MXFP4 (M11) | 517 GB | 420 / 412 GB |
| MXFP4 and an EXL3 build, one shared Engram volume | 720-740 GB | 615-643 GB |

With its IQ4_NL table the first artifact (≈ 315 GB plus staging) would have fitted the 383 GB free
on 2026-10-24 only by using `C:` and striping the Engram volume, since `E:` had 18 GB and no drive
would keep 112 GB free beside an expert volume; with FP8 rows it needed ≈ 45 GB more. The owner
frees space when the artifact work starts (§1.2).

### 7.7 Other artifacts

- **NVIDIA's September NVFP4 export** and **LibertAIDAI's 1017 NVFP4** (M11) convert to the same
  expert records as their official counterparts (§7.1) and bring MXFP4 Engram tables; an FP8 volume
  made from the official release can replace those (§14.5).
- **Other EXL3 packs of 1017** (diffbot's uniform K3, sudoingx's 2.8) use the EXL3 recipe unchanged.
- **GGUF compact builds** (ISTA's GSQ-RCO, Unsloth's UD, Maya's and DS4's) would take their routed
  experts as stored and everything else from the official release; option O6 (§5.7 compares them).

---

## 8. Ownership in Infernix

### 8.1 New family

This is an **explicit product change**: a new mathematical architecture (`DeepseekV41ForCausalLM`,
model type `deepseek_v41`).

| Component | Owner | Location |
|---|---|---|
| Config interpretation (exact key set, refusals), mathematics, binding, block order, the two prompt rules, DSpark alignment | Model | `src/models/deepseek_v41/` (`config`, `weights`, `load`, `execution/forward`, `execution/parameters`) |
| Program: lanes, windows and compressed planes, Engram rows, prefix-cache binding, DSpark drafter, graphs, expert-offload composition | Program | `src/models/deepseek_v41/program/` |
| Engine instance | Runtime | `src/runtime/engine/deepseek_v41_instance.{h,cpp}`; a new `Architecture` value in `src/models/registry.cpp`; an `EngineCore` arm |
| MX8 activation codec and exact block combination | Ops (common) | `src/ops/common/mx8.h`, beside `exact_combine.h`; the scalar form is the exact oracle |
| EXL3 codec (trellis decode for K = 2-6, `mul1` codebook, 128-point Hadamard, E3A16) | Ops (common), built here and reused by GLM | `src/ops/common/exl3/` (GLM §8.3); the scalar form is the exact oracle |
| Tokenizer pre-tokenizer, DeepSeek encoding, DSML tool-call and reasoning parsing | Frontend | §21 |
| Recipes (EXL3 first, MXFP4 in M11), streaming conversion, MX recovery (M11), expert and Engram volume writers, verification, reference forwards | Converter / tools | `tools/convert/deepseek_v41.py`, `official_recipes.py`, `tools/artifact/expert_volume.py`, `tools/artifact/restripe.py`, `tools/deepseek_v41/` |

DeepSeek's reference code and DS4 are MIT-licensed; any code ported from them keeps its notice.

### 8.2 Shared components

Three explicit families now need host-offloaded experts: Qwen4Exp (built), GLM-5.3-Flash (designed)
and this one. **This track builds the shared pieces first and GLM reuses them** (owner, 2026-10-24):
the offload extraction with its page pool, and the EXL3 codec with the expert routes, for K = 2-6
so that GLM's 6-bit head and 5-bit projections are covered too. Expert volumes and the readers come
with the extraction in this track's M2; `mhc` and the radix top-k come with its M3. As GLM §8.2
describes, there is no base class, virtual model interface or registry:

- **`src/models/offload/`**: expert cache, residency controller, RAM tier with pinned and unpinned
  records, expert store, state file, ledger terms, route trace, VRAM monitor, parameterized by an
  `ExpertBankGeometry` (banks, experts per bank, record format, record and unit bytes, H, I).
  Qwen4Exp keeps every decision bit for bit (M2's exit criterion, GLM §23).
- **The page pool, built into that extraction from the start** (decision f): one page size per
  artifact; a record of any format takes ⌈bytes / page⌉ pages named by a page list; VRAM, pinned and
  unpinned RAM, staging, landing and bounce storage are all pages; eviction weighs value per page.
  A single-format artifact (Qwen4Exp, GLM's EXL3 build, the MXFP4 artifact) chooses its record size
  as the page size, one page per record, so its behaviour is today's frames and slots. GLM's EXL3
  records take the `exl3_expert_v1` layout (§7.3), its 36 KiB vectors leading the record in place of
  GLM §8.2's frame side entry, so one set of expert routes reads both models.
- **Expert volumes and the striped reader** (GLM §7.3, §11.3), shared with GLM.
- **The `mhc` Op** (GLM §15) with a *pre source* input: same sublayer (GLM) or the previous one
  (this model, §15).
- **The radix top-k select** moves to `ops/common` (GLM §13.3) and serves `qsa`, GLM's indexer and
  this model's indexer.
- **The row-volume reader** behind Flash-Next's n-gram table (`NgramVolume`, `ReadOnlyFile`'s ring,
  the gated upload) serves the Engram volume (§14); the PLE Op itself is not shared.

### 8.3 Op changes

| Op | Change |
|---|---|
| New Linear format **`fp8_e4m3fn_b32_e8m0`** | MX8 activations (§9.5); decode route (T ≤ 8, CUDA-core integer) and prompt route (int8 tensor cores), bit-identical; grouped application for `wo_a`; generalizes `fp8_block/fp8_exact.h` (128 × 128 BF16 scales) to 32 × 32 power-of-two scales |
| `offloaded_sparse_moe` | Expert format **`exl3_mul1` with a width per record** (K = 2-5, E3A16 after MX8, §9.5) on the narrow, wide and CPU routes, records read through page lists; `mxfp4` and `nvfp4` with the **MX8 activation class** (beside today's A4 and A16) in M11; geometry from the bank; routing function `sqrtsoftplus_bias_topk_scaled` with a second bias for image tokens; the routing weight applied to h before the down projection; SwiGLU clamp 10; ungated shared expert in FP8; landing-aware jobs and partial records (GLM §11.4); a second bank for DSpark (128 experts, top-3) |
| New **`csa_attention`** | Window ring plus selected compressed entries, one shared 512-wide key/value, per-head sink, inverse RoPE on the output; decode and prompt kernels (§13.3, §13.6) |
| New **`csa_compressor`** | Ratio-2 gated pooling and ratio-1 projection, norm, RoPE, FP4 packing, index keys (§13.2) |
| New **`csa_indexer`** | MXFP4 query/key scores, head weights, top-512, layer 20's block candidates, Reindex masks (§13.4) |
| `mhc` (shared with GLM) | Single-pass variant (§15) |
| New **`engram`** | Row decode (FP8 or MXFP4 to BF16), the `wkv` projection, per-stream gates and the stream update (§14.6) |
| New DSpark Ops | Attention over the main-feature window and the bidirectional block; Markov and confidence heads (§17) |
| Core | `StripedRecordReader` (GLM §11.3); the row-volume reader generalized from `NgramVolume` |

---

## 9. Decode pipeline and canonical arithmetic

### 9.1 Execution agents

As GLM §9.1: the compute stream with one captured graph per (batch B, width W); a transfer agent
(VRAM policy, promotions, staged DMA of RAM misses and landed records); an SSD agent (one IOCP queue
per volume, demand and prompt reads, landing ring, refills); the CPU expert team (service thread plus
calibrated workers); and the engine worker, which also hashes Engram rows and reads them through the
row-volume ring (FN §12.3). Host agents spin only while a request is active; the process is exempt
from power throttling (L11).

### 9.2 Host-device protocol

Flash-Next's mapped-memory protocol (FN §8.2) with GLM §9.2's additions (`landing[slot]` masks and
sequence words). New: the Engram rows of a round arrive through the gated upload of FN §12.3 S2
(`upload_pinned_when`), read by the `engram` kernel of layer 1 and layer 14.

### 9.3 Per-layer kernel sequence and budget (T = 1)

Budget = bytes at 1.5 TB/s + 0.5 µs ramp under PDL, as in FN §8.3. "Pre" = before routing.

| # | Kernel | Reads | Budget |
|---|---|---:|---:|
| K1 | Mixes from the stream entering the sublayer (24 dot products over 20,480, Sinkhorn): **concurrent**, off the critical path, because this sublayer's collapse uses the pre it received (§15) | 2.0 MB | (2 µs) |
| K2 | Collapse with the received pre and `attn_norm` (partial sums from the producer's epilogue); MX8 of x; `wq_a` [1,280 × 5,120] and `wkv` [512 × 5,120] FP8 in one launch | 9.2 MB | 6.6 µs |
| K3 | `q_norm`, MX8, `wq_b` [32,768 × 1,280] FP8; epilogue: RoPE on q; window entry (`kv_norm`, RoPE, MX8 → ring) | 41.9 MB | 28 µs |
| K3f | Full layers only: compressor (`wkv`/`wgate` BF16; pooling when a group completes; norm; RoPE; FP4) and index key (`wk`, `k_norm`, RoPE, MXFP4); Full and Reindex layers: indexer query (`wq_b` FP8 [4,096 × 1,280], `weights_proj` BF16) | 5.4-16 MB | 4-11 µs |
| K4 | Full and Reindex layers: indexer scores and top-512 (layer 20 also forms the 2,048 candidate blocks) | 4.5-8.9 MB at 128K; 1.1 MB on Reindex layers | 1-6 µs |
| K5 | Sparse attention: 64 heads over ≤ 128 window entries (528 B each) and ≤ 512 compressed entries (288 B each), sink, inverse RoPE; MX8 of each group's output | 0.2 MB | 3-4 µs |
| K6 | `wo_a` grouped (FP8, 8 × [1,024 × 4,096]) then `wo_b` [5,120 × 8,192]; epilogue: residual update with K1's post and comb, collapse for the FFN with the attention sublayer's pre, `ffn_norm` partial sums | 75.4 MB | 51 µs |
| K7 | Router: FP32 logits [384 × 5,120] (BF16 weights), √softplus, top-6 with bias, weights, residency classification, jobs, mailboxes; FFN mixes run concurrently | 3.9 MB + 2.0 MB | 6 µs |
| | **Pre** | ≈ 134 MB | **≈ 100 µs** (+ 5-17 µs on Full/Reindex layers) |
| K8 | Shared expert (FP8 gate/up [4,608 × 5,120], down [5,120 × 2,304]) ‖ routed GPU jobs (E3A16 narrow on MX8 inputs, 8.9-22.2 MB each; MXFP4 in M11: 18.8 MB) | 35.4 MB + 13.7 MB per hit on average | 24 µs + 9.1 µs per hit (MXFP4 12.6) |
| K9 | Combine (FP32, ascending expert id) with CPU and landed outputs; residual update; collapse for the next layer's attention; `attn_norm` partial sums | 0.1 MB | 2 µs + waits |

Layers 1 and 14 add the `engram` kernel before K1: `wkv` FP8 [25,600 × 6,144] (157 MB, ≈ 105 µs),
gates and the stream update. The head (BF16 [129,280 × 5,120], 1.32 GB) takes ≈ 0.88 ms.

**Per token (GPU work only):** with MXFP4 records at h_v = 0.48, 40 × (≈ 105 + 24 + 2.9 hits × 12.6
+ 2) µs + Engram 0.21 ms + head 0.88 ms ≈ **7.8 ms**; with the EXL3 records at h_v ≈ 0.58,
40 × (≈ 105 + 24 + 3.5 hits × 9.1 + 2) µs + 1.09 ms ≈ **7.6 ms** (assuming the narrow E3A16 route
streams at memory speed, as GLM §9.3 does; M6 measures). About 420 launches; ≈ 4.2 ms precede
routing (t_pre of §5.1).

**Rules**, from FN §8.3: PDL with pre-dependency weight prefetch at ≤ 2 CTAs per SM; evict-first
weight streams, evict-last activations, windows and next-layer routers; deterministic fixed-order
reductions; grids in multiples of 170 SMs; no grid barriers. **Stall-time warming** (GLM §9.3)
matters more here: a layer waits ≈ 1-3 ms on SSD records (§5.3), and while K9 waits, idle CTAs pull
the next layer's K2-K3 weights (≈ 51 MB) into the 96 MB L2. Estimated saving ≈ 30-40 µs per layer,
≈ 1.4 ms per token (≈ 2 % of a ~78 ms token; M6 measures).

### 9.4 Router (K7)

Phase 1: warp-per-row FP32 logits of the 384 router rows: x (BF16) and the BF16 weights give exact
products, summed exactly and rounded once to FP32 (FN §16.5: FP32 logits avoid BF16 ties). Phase 2
(last CTA by arrival counter), per column:

1. s_e = √softplus(z_e) with the canonical `exp`/`log1p` of `ops/common/canonical_math.h` and
   torch's threshold (softplus(z) = z for z > 20); DS4 had to fix its Metal router for small logits
   (commit `0719a0ba`), so M3's oracle covers z ≤ −20 exhaustively over BF16-representable sums.
2. top-6 of s_e + bias_e (bias_vl_e on the whole image span), lower id winning exact ties (as
   `model.py` and DeepSeek's fused router both do); w_i = fl32(fl32(s_i / (Σ s + 1e-20)) · 1.5), the
   division first, as both do.
3. Union over the T columns; classification from the residency word and tier tables, as GLM §9.4:
   `READY` pages → GPU job; RAM pages → CPU or DMA job (§9.7); unpinned pages → CPU job; landing
   pages → CPU job per unit; absent everywhere → demand read, then a landing-page job.
4. Outputs: GPU job list, `miss_req[l]`, the route log with the scores of ranks 7-16 (near misses,
   §10.2).
5. Prediction (lookahead, off by default, §11.3): layer l + 1's router applied to this layer's MoE
   input from L2-warm weights.

### 9.5 The canonical arithmetic MX8

**Requirement.** Every FP8 and FP4 linear computes DeepSeek's MXFP8 product (decision b), and its
BF16 output is bit-identical on every route and processor: GPU narrow and wide routes, CPU on every
ISA, from VRAM pages, staging, RAM pages, unpinned pages or landing pages in any unit
order (FN §16.2's requirement, extended to dense layers as FN §16.2.2 did).

**Activation encoding** of a BF16 column x of K elements, per 32-element block b (the reference's
`act_quant`, §2.2):

```text
a_b  = max(max_k |x_k|, fl32(1e-4))                                   exact
e_b  = exponent(fl32(a_b · fl32(1/448))) + (mantissa ≠ 0)              integer
     = max(⌈log₂(max_k |x_k| / 448)⌉, −22)                             the same, for every input
q_k  = e4m3_rn(clamp(x_k · 2^−e_b, −448, 448))                          exact scaling, one rounding
X_k  = q_k · 2^9 ∈ ℤ, |X_k| ≤ 229,376 < 2^18;  x̂_k = X_k · 2^(e_b − 9)
```

fl32(1/448) = (1/448)(1 + 3 · 2^−26) never moves a power-of-two bin for a normal FP32 amax (the same
holds for fl32(1/6) in E2M1 packing), and the 1e-4 floor is an exponent of at least −22. `model.py`,
TileKernels and DeepGEMM (by an integer rule on the bits, without a float multiply) all compute this;
§20.1's MX8 oracle checks the three forms.

**Weights.** MXFP4: c2 = 2 · e2m1(code) ∈ {0, ±1, ±2, ±3, ±4, ±6, ±8, ±12} with a UE8M0 exponent u per
32 weights along K. NVFP4: c2 with Ŝ (E4M3 · 2^9) per 16 and the matrix multiplier (FN §16.2.1).
Dense FP8: W = e4m3 · 2^9 ∈ ℤ (|W| < 2^18) with a UE8M0 exponent per 32 × 32 block.

**Row product**, for one output row r:

```text
MXFP4:  P_b = Σ_{k∈b} c2_rk · X_k             |P_b| ≤ 32 · 12 · 229,376 < 2^27, exact int32
FP8:    P_b = Σ_{k∈b} W_rk · X_k              |P_b| < 2^41, exact int64
S       = Σ_b P_b · 2^(E_b − E_min)           E_b = u_b + e_b (+ format constants); exact, int128 where the
                                              spread needs it (exact::add_shifted, FN §16.2.2)
y       = bf16_rn( fl32_rn(S · 2^(E_min − κ)) )        κ: the format's fixed power of two; one rounding to FP32,
                                                        one to BF16 (FP32 output where the reference keeps FP32)
```

Integer steps are exact; the two roundings at the end are fixed. The result depends only on the
stored words and x, so **any split of K, rows or columns, any order and any processor give the same
bits**. NVFP4 records use FN §16.2.1's form with the MX8 encoding in place of A16; the result equals
the MXFP4 record's bit for bit (the values are identical and both sums are exact), which M3 tests.

**The expert chain** (semantic boundaries as in the reference):

```text
x (BF16) → MX8 → gate, up (BF16 outputs) → FP32: g ← min(g, 10), u ← clamp(u, ±10)
  → h = bf16( w_i · silu_c(g) · u )   (FP32, canonical exp of FN §16.2; shared expert: w = 1)
  → MX8 → down → BF16 y_i → combine: bf16( Σ_i y_i [FP32, ascending id] + y_shared )
```

**EXL3 experts (the first artifact).** Gate, up and down are EXL3 matrices of the expert's width K,
computed with GLM §9.5's canonical **E3A16** arithmetic on the MX8 values of their inputs. x̂ =
MX8(x) is exact in BF16, so E3A16 takes it as its BF16 input unchanged: the rotation u =
H(x̂ ⊙ suh) in fixed-order FP32, the 14-bit block quantization of u (F = 15 for the 5,120-wide inputs,
16 for the 2,304-wide down input), exact integer dot products with the codebook's exact values, one
rounding to FP32, the output rotation and `svh` in fixed-order FP32, one rounding to BF16. The
chain above holds with "MX8 → E3A16" in place of the MXFP4 product, h rounded to BF16 and MX8 again
before down, and the same clamps (GLM's experts clamp alike). K only changes where each weight's
16-bit trellis state is cut from the stream (bit K · i), so one kernel serves every width with K
per job, and the result depends only on the stored words and x: the same bits on every route and
processor, as for MX8. Relative to the pack's own runtime (FP16 activations through ExLlamaV3 or its
vLLM plugin), this adds DeepSeek's MXFP8 rounding of every expert input, as the reference does
before its FP4 experts (decision b), and E3A16's own differences (the codebook's exact values,
14-bit block activations; GLM §9.5). §20.3 measures the result against the original model.

**Fidelity to the reference.** `model.py` accumulates each 32-block product on tensor cores in FP32,
scales it and adds the blocks in FP32; its result is a rounding of the exact sum S, so the two differ
by the reference's own accumulation error (relative ≲ 2^−24 per block added). §20.1 bounds it against
an FP64 evaluation of the same quantized operands.

**Why exactness is cheap here**: per-block products fit int32 for FP4 weights; the GPU narrow route
uses `dp4a` with X as three byte limbs (FN §16.2.1's narrow form); the wide route uses two
`mma.m16n8k32` per 32-block with FN §16.2.1's limb split (X has 18 bits instead of 21); the CPU uses
two int16 limbs with `vpdpwssd` (FN §16.2.2's CPU GGUF form). Dense FP8 rows take FN §16.2.2's
base-64 digit planes on the tensor cores. Rejected, as FN §16.2.2 rejected them: FP8 or FP4 tensor
cores with FP32 accumulation (undocumented order on `sm_120`, not reproducible on the CPU). DeepSeek's
server uses exactly that, the SM100 block-scaled MMA with UE8M0 scales on both operands and no
software promotion; `sm_120`'s warp-level block-scaled MMA has the same undocumented accumulation.

**Compiler and instruction rules** are FN §16.2's: `-ffp-contract=off -fno-fast-math`, explicit
`__fmaf_rn` / `__fmul_rn` / `__fadd_rn`, and `cvt.rn.satfinite.e4m3x2.f32` checked exhaustively
against the CPU's integer E4M3 rounding.

**Other operations** keep the reference's precision at each boundary: router and compressor products
exact then one rounding to FP32; norms in FP32 with one rounding; RoPE in FP32 with the result
rounded to BF16 (`apply_rotary_emb` writes back to BF16); attention, indexer, mHC and Engram as
§13-§15 state.

### 9.6 Expert routes

| Route | When | Form |
|---|---|---|
| Narrow (GPU) | n ≤ 8 columns for the expert (decode, verification) | EXL3: CUDA-core integer per GLM §9.5 (funnel shift to the state, `IMUL` by the codebook multiplier, `DP4A` byte sum, `IMAD` into the block sum), the state decode shared by the n columns, K per job, the record read through its page list. MXFP4 (M11): X limbs × codes via `dp4a`. Work items per gate/up unit and down unit (§7.3), so an SSD record's landed units are consumed as they arrive |
| Wide (GPU) | n > 8 (prompts, wide verify) | EXL3: int8 tensor cores with four products per weight and activation (GLM §9.5), decoded tiles reused over the CTA's token tile; experts of one width grouped per launch. MXFP4 (M11): two k32 MMAs per block |
| CPU | RAM, unpinned and landed records | EXL3: GLM §12.2's AVX2 + AVX-VNNI E3A16 kernel per width (MXFP4 in M11: §12.2), and a scalar form (the oracle) |

All three return the same bits (§9.5); the route depends only on n, never on placement.

### 9.7 Miss arbitration

- **RAM misses**: the divisor rule (L19) fed by the measured pair, per width. EXL3: the CPU team's
  ≈ 0.35 ms (K2) to ≈ 0.8 ms (K5) per expert, compute-bound (estimated at 30 GB/s of weight bytes,
  §5.7), against the link's 0.32-0.80 ms, so about an even split; MXFP4 (M11): ≈ 0.35-0.45 ms against
  0.68 ms, the CPU taking about 60 % (d ≈ 2.5). The split table is calibrated at start per (column
  count, width) and kept across starts (GLM §12.3).
- **SSD misses**: CPU jobs first, consuming units as they land; staged DMA after landing only beyond
  the CPU cap (16 jobs per call, FN §19.3.16).
- **Unpinned records** (past the pin room): CPU jobs in place; DMA only through bounce slots
  (FN §19.3.7).
- **Multi-column experts** (verify windows): the CPU decodes codes once for all columns; beyond the
  calibrated column count the expert goes to DMA.

### 9.8 Opt-in approximate expert mode

GLM §9.8 applies unchanged in structure: off by default; when on, an expert at rank ≥ k_l (by
selection score) that is not available within the chosen scope is left out of the combine, without
renormalization unless M7a shows renormalization lowers KL; the shared expert is never skipped; a
reproducible `static` rule exists for evaluation; caches are credited as if the expert had run;
prompts stay exact; every surface labels the mode; CausalScoring refuses the `residency` rule.

What differs here:

- **The `ssd` scope is the useful one.** It removes the reads that bind (§5.3): leaving out SSD-only
  experts at ranks 5-6 would remove about a third of the SSD reads, ≈ +20 % decode (model). The
  `vram` scope would skip ~half of every token's experts.
- **Quality is a real risk.** √softplus top-6 weights are fairly flat; issue DS4 #636 measured an
  11.4 % block error from dropping one of eight sigmoid-routed experts on MiniMax-M2.5. M7a measures
  the band sensitivity on this model before any preset ships.
- **Gates** are GLM §9.8's (mean KL ≤ 0.02 to the exact model, p99 ≤ 0.3, top-1 ≥ 95 %, the
  long-answer loop test), with replayed residency at this machine's capacities.

---

## 10. Expert residency: VRAM, RAM and SSD

### 10.1 Tiers

Every tier is made of **pages** (§7.3, §8.2): 4.25 MiB for the EXL3 artifact, where a record takes
2-5 pages through its page list; one page per record (17.95 MiB) for the MXFP4 artifact.

| Tier | Capacity on this machine, EXL3 artifact (MXFP4) | Holds | Policy owner |
|---|---|---|---|
| VRAM pages | ≈ 4,900 (≈ 20.3 GiB): ≈ 1,250-1,600 experts, 8-10 % (1,160 frames, 7.5 %); elastic with KV, DSpark pages and the walk lease | Hot experts; staging; loans | Transfer agent |
| Pinned RAM pages | ≈ 17,300 (≈ 72 GiB, the pin room) (≈ 4,000 slots) | Warm experts; the landing pages; the demotion list | SSD agent (admission), transfer agent (promotion source) |
| Unpinned pages | ≈ 1,900 (≈ 8 GiB) at 88 GiB available (≈ 450 records) | The coldest RAM experts, CPU-served in place | SSD agent |
| Expert volumes | all 15,744 records, read-only | Everything | — |

Residency is **exclusive with shadows** (FN §19.3.7; GLM §10.1): a VRAM resident keeps its RAM copy
only as a shadow, the first RAM victim; demotions are capped at 32 per decode boundary.

### 10.2 VRAM policy

**Baseline: LFRU over one global pool** (FN §9.3). M1 replays the GLM §10.2 candidates on this
model's traces and adopts one only if it beats LFRU by ≥ 5 % fewer misses per token overall and is
not worse by more than 2 % on any trace category: LRU; LFRU with near-miss credit (ranks 7-16 at
c ∈ {0.1, 0.25, 0.5}); protection of depth-1 predictions; aged LFU; soft per-layer shares from
routing counts; a static hot set with an LRU remainder. Static profiles start behind recency on
this family (DS4 #636, §4.1).

**Promotions** come from CPU-served misses that outrank the victim and from landed records (copied
from the landing pages). A promotion moves 8.9-22.2 MB (MXFP4: 18.8 MB, 6.8× Flash-Next's), so the
pace is re-swept in M5 ({1, 2, 4, 8} per layer call until the pages fill, then every N tokens), with
GLM §10.2's admission threshold (m ∈ {0, 2, 4, 6} aged routes) and free-page refill after a prompt.
No promotions after a walked prompt (L6).

**Value per page.** With records of 2-5 pages, every policy above ranks experts by expected hits
per page (a K5 expert must earn 2.5 × a K2 expert's hits to keep its pages), and a victim set is
chosen until enough pages are free. M1's trace replay picks the policy with the widths in place; an
artifact with one width ranks exactly as before.

### 10.3 Epochs, staging and loans

Unchanged from FN §9.2-§9.5 (frame epochs, slack frames, KV loans, the elastic VMM pool, display
headroom), with pages in place of frames. Staging uses pages with a staging role, at most 128
(0.53 GiB: 25-64 records; MXFP4: 32 frames, 0.56 GiB).

### 10.4 RAM tier

- **Sizing**: FN §19.3.7's ledger: available RAM at startup − `--ram-headroom-mib` (2 GiB) −
  embedding (1.23 GiB pinned) − Engram row cache (0.26 GiB) − prefix Host tier (§18) − Program
  buffers − 1 GiB growth − 1.5 GiB of process memory. The expert share is pinned up to the pin room
  (shared system memory − NON_LOCAL usage − 1 GiB, ≈ 72 GiB of experts) and unpinned past it; the
  landing pages (256, 1.06 GiB: 51-128 records; MXFP4: 64 slots, 1.12 GiB) and 128 bounce pages
  (MXFP4: 32 slots) come out of the pinned part.
- **At 88 GiB available**: ≈ 79.8 GiB of experts → ≈ 17,350 pinned and ≈ 1,900 unpinned pages, about
  41 % of the EXL3 bank (with VRAM about half, §5.7); MXFP4: ≈ 4,000 pinned slots and ≈ 450 unpinned
  records (29.2 % of the bank with the ring excluded). **At 79 GiB**: ≈ 37 % (MXFP4 26 %). **With other sessions holding
  RAM** (24 GiB free on 2026-10-18) the tier shrinks to what is left above the headroom, or startup
  refuses below FN's minimum, naming the cause.
- **Commit charge** (GLM §10.4): the ledger plans against the commit limit as well; the 200 GB page
  file on `G:` covers it.
- **Policy**: decayed LFU (FN §19.3.7) as baseline; M1 replays LRU, CLOCK and LFRU on the two-level
  simulation and adopts the one with the fewest SSD reads per token on held-out traces.
- **Admission**: a landed record is admitted when it outranks the RAM victim; demotions of VRAM
  victims without a host copy follow FN.

### 10.5 Warm start, prompt seeding and refill

As GLM §10.5: the expert state file (VRAM and RAM rankings, use counts) saved at shutdown and every
10 minutes between requests; startup fills VRAM then RAM in ranking order with coalesced reads at full
striped bandwidth (≈ 106 GB in ≈ 7.5 s at 14.15 GB/s, overlapped with pinning); prompt routes seed
both policies with a weight M1 fits; the prompt landing reserve is off by default (Maya, GLM §4.3);
idle refill at full bandwidth.

### 10.6 DSpark experts

The 384 DSpark records form their own bank: in the EXL3 artifact the pack's K2 drafter experts,
2 pages each (3.4 GB in all), so the whole drafter fits in 768 pages (3.2 GiB); in the MXFP4 artifact
the official MXFP4 ones. The drafter never waits for them (§17.2): a bounded VRAM share (default
192 pages, 0.80 GiB, 96 experts, taken from the page pool; M7 sweeps 0-768; MXFP4: 96 frames,
1.68 GiB) ranked by LFRU weighted by acceptance, and a RAM share (default the rest of the drafter,
576 pages, 2.4 GiB, taken from the tier only while DSpark is on; MXFP4: 192 slots, 3.4 GiB).

---

## 11. Multi-SSD expert streaming

GLM §11 applies in full: one to four volumes, read-only, every record read uses every volume, demand
reads never queue behind more than ≈ 0.3 ms of other reads per drive, failed reads fail only the
rounds that need them, IOCP on Windows and `io_uring` or a `pread` pool on Linux behind Core's
`StripedRecordReader` with one `DirectReadQueue` per volume. What changes:

### 11.1 Reading a record

An EXL3 record read is 8-20 unit reads of 1,114,112 B issued at once, landing in its 2-5 landing
pages. With w = (4, 4, 4) on three drives each serves a third of the units; at the DMI-bound
≈ 4.7 GB/s per drive a K3 record (12 units) completes in ≈ 0.95 ms, a K5 in ≈ 1.6 ms, and the first
gate/up units in ≈ 0.25 ms (model; R0 measures). An MXFP4 record (M11) is 12 units of 1,568,768 B in
one 17.95 MiB slot, ≈ 1.33 ms. Gate/up units are issued first, so CPU and GPU consumers start on
them while the down units are in flight (GLM §11.4).

### 11.2 Depth and priority

- **Per-volume caps**: demand 6, prompt stream 8, background 1, total 10 (1.06 MiB EXL3 units at the
  1 MiB/QD4 knee of GLM §3.3, MXFP4's 1.5 MiB units between it and the 2 MiB/QD2 knee; §11.5
  calibrates).
- **Priorities**: Engram row reads on the Engram volume's drive > demand (current layer) > prompt
  stream > lookahead > background refill. A 4 KiB Engram read queued behind four units would wait
  ≈ 0.9 ms (MXFP4 1.3 ms); it is issued ahead of queued units instead (§14.7).

### 11.3 Lookahead

Off by default (`--ssd-lookahead off`): every measurement on this family lost (JigSawPT −7 to −19 %,
hipfire 0.86-0.93×; §4.2), as Maya's did on GLM. With SSD-bound decode the drives are busy ≈ 90 % of
a token, so lookahead can only use the remaining ≈ 10 % (t_pre gaps between layers) and costs
bandwidth when wrong. GLM §11.5's budgeted design is built in M5 and measured on three volumes; it is
enabled only if it gains.

### 11.4 Placement on this machine

For the EXL3 artifact (record times for a K3 record of 13.4 MB; space for its 191.7 GB of volumes):

| Configuration | Bandwidth (§3) | Record time | Free space needed per drive | Advice |
|---|---:|---:|---:|---|
| **Three chipset drives (`D:`, `E:`, `G:`), w = (4, 4, 4)** | 14.15 GB/s | 0.95 ms | 63.9 GB | **Default.** Keep the Engram volume and the repository's builds off the busiest volume where space allows (builds add tail latency; §11.5 warns) |
| The same, w = (5, 4, 3) | 14.15 | 1.0 ms (+6 %) | 79.9 / 63.9 / 47.9 GB | When one drive is short of space |
| Two drives, w = (6, 6) or (7, 5) | 12.73 | 1.05-1.17 ms | 95.8-111.8 GB | Decode 18.9 tok/s (model) |
| `C:` + three chipset drives, w = (3, 3, 3, 3) | 19.27 | 0.69 ms (−27 %) | 47.9 GB | Fastest (22.7 tok/s, model); needs room on the system drive |
| One volume | 5.9-7.0 | 1.9-2.3 ms | 191.7 GB | Supported; 11.4 tok/s (0.57× three volumes) |

The MXFP4 artifact (M11) needs 98.8 GB per drive at w = (4, 4, 4) (1.33 ms records), 74.1 GB on four
drives and 296.4 GB on one. Which drives hold the volumes is decided when the owner frees the space
(§7.6); free space on 2026-10-24 was 141 / 144 / 18 / 80 GB on `C:` / `D:` / `E:` / `G:`.

### 11.5 Calibration, failures and integrity

GLM §11.7-§11.8 unchanged: a startup read of 64 units per volume at QD 1 and QD 8, the physical-disk
and root-port topology check with warnings, a retry then a recoverable round failure on read errors,
header and sampled-CRC checks at startup, a full offline check in `verify_artifact`, no per-read
hashing.

---

## 12. CPU expert engine

### 12.1 Role

The CPU computes about half of the RAM misses (MXFP4: about 60 %), every unpinned record's expert,
and every SSD miss within the job cap, in place: the same bits as the GPU.

### 12.2 Kernels and cost on the i9-13900K

**EXL3 (the first artifact).** GLM §12.2's E3A16 kernel (AVX2 + AVX-VNNI, a scalar form as the exact
oracle) with K per job and the record read through its page list: per weight a 16-bit state cut
from the stream, the `mul1` multiply and byte sum, an int16 product into the block sum; the input
rotations and block quantization once per job. Unlike the MXFP4 kernel it is **compute-bound**:
GLM §12.2 estimates 0.2-0.4 ms per 6.3 MB GLM expert, which scales to ≈ 0.4-0.85 ms for a DeepSeek
K3 expert; §5.7 carries 30 GB/s of weight bytes (≈ 0.35 ms at K2, 0.49 at K3, 0.8 at K5) and shows
that half that rate costs ≈ 10 % of decode. The 13900K has no AVX-512, which adrienbrault's EXL3 CPU
worker used to reach DRAM speed (GLM §4.4); M0's probe measures the P- and E-core rates per width.

**MXFP4 (M11).**

`cpu/mx8_expert.cpp`: an AVX2 + AVX-VNNI kernel and a scalar form (the exact oracle, shared with
`ops/common/mx8.h`). Per 32 weights of one row and one column: the 16 code bytes decoded to int16 c2
values by `vpshufb` and widening, two `vpdpwssd` per limb against X's two int16 limbs (X = 512 · hi +
lo), and one shift-add of the block sum into the row's int64 accumulator: ≈ 14 vector µops per 32
weights, ≈ 0.44 per weight (estimated).

**Estimate** (model, M0 measures): 35.4 M weights × ≈ 0.15 cycles per weight on a P-core's vector
ports ≈ 1 ms of one P-core, ≈ 0.13 ms over 8 P-cores; the 18.8 MB record takes ≈ 0.31 ms at 60 GB/s of
DRAM. **The CPU lane is DRAM-bound**, as Flash-Next's was (≈ 61 GB/s on its FP8 route), so §5 carries
0.35 / 0.45 / 0.60 ms with overheads; a second column adds only products (≈ 20 %).

### 12.3 Execution

GLM §12.3 unchanged: atomic item counters, phase A (gate/up and h), one barrier, phase B (down);
landing-aware items that skip units not yet landed; ~48 pieces per call claimed in turn; workers on
physical cores only (never SMT siblings); P-cores first, E-cores swept 0-16 in M5; a per-count split
table calibrated at start and kept across starts.

### 12.4 Prompt CPU assist

Prompt experts the CPU computes cheaper than the link can deliver them: the thin ones. Under bounded
replay every decoder expert sees ≈ 2 of the 128 replayed tokens (6 · 128 / 384), so **the CPU computes
the decoder replay's non-VRAM experts from RAM and landing slots**, layer by layer, while the GPU runs
the replay's dense work, taking most of ≈ 6,600 expert transfers per prompt off the link (§16.4).
Under the full rule (the default) the decoder suffix gives an expert about 2 tokens at layer 39 and
40 at layer 20, so the CPU takes the thin experts of the later decoder layers by the same cut. Any
column count is allowed, since every MX8 route is exact (GLM §12.4's argument); the cut per layer
comes from the calibrated cost per expert-row.

---

## 13. CSA2 attention and the hierarchical indexer

No `sm_120` kernels exist for either: at the commits DeepInfer pins, FlashMLA's sparse kernels and
DeepSelect are built for SM100 only (tcgen05, tensor memory, 2-CTA clusters and up to 227 KB of shared
memory per block), and vLLM's `sm_120` route has open blockers for V4.1's shapes (§4.2). Both are new
Infernix Ops. What transfers from DeepSeek's kernels is their structure, not their code: the 64
heads as the MMA's M dimension with each KV block dequantized once for all heads; the 512-wide dot
product in two fixed-order halves; RoPE and MX8 fused into the loads and epilogues; the indexer's
epilogue with the head weights in registers (relu, weight and head sum fused); DeepSelect's
threshold-filter streaming top-k (§13.4); the fused router. On `sm_120` a 64-head query's 64 KB of Q
and 128 KB of FP32 O do not fit one block's shared memory, so heads or columns are split, with the
canonical merge of §13.3. One converter-side layout choice also transfers: FlashMLA permutes the rows
of the q projection and the columns of `wo_a` in whole 32-groups so the kernel's loads and FP8 stores
coalesce; the permutation is exact and is decided in M3.

### 13.1 Per-sequence state

| State | Layers | Per token | At 128K | Stored form |
|---|---|---:|---:|---|
| Window ring | every layer (and DSpark's over main features) | bounded: (128 + K_max + 1) × 528 B per layer, 134 slots at K_max = 5 | 2.8 MB (+0.2 MB) | E4M3 codes + UE8M0 per 32 |
| Compressed entries | Full layers 2, 8, 14 (ratio 2) and 20 (ratio 1) | 3 × 144 + 288 = 720 B | 94 MB | E2M1 + E4M3 per 16, 288 B per entry, paged |
| Index keys | the same four layers | 3 × 34 + 68 = 170 B | 22 MB | E2M1 + UE8M0 per 32, 68 B per key, paged |
| Compressor state | 2, 8, 14 | — | ≈ 86 KB | a ring of ratio + K_max = 7 tokens per layer (28 KB), FP32 (kv, gate score) per token, indexed by absolute position |

The persistent part is **890 B per token**, the report's figure.

**Room for verification** (DeepInfer `worker/cache_engine.py`, the SWA block size and compress
state size): a verify round of K drafts writes K + 1 window entries. With exactly 128 slots, column
P + j would overwrite the entry of P + j − 128 that an earlier column still reads, and a rejected
column would clobber a committed entry the next query needs. Verification writes every column before
any reads, so a window ring needs at least 128 + K_max slots; Infernix uses 128 + K_max + 1 = 134,
DeepSeek's size (its extra slot absorbs the overshoot of copies at 128-token boundaries). The
compressor keeps per-token state for ratio + K_max = 7 positions (DeepSeek's kernel asserts
ratio + verify width − 1 with a width of 6), so a rejection is only a frontier move, and the DSpark
window has the same room. A unit test runs K + 1 = 6 verify columns in one slot with every rejection
count against sequential decode, bitwise (as DeepInfer's `compress.py` tests do).

### 13.2 Compressor and index keys (`csa_compressor`)

- **Ratio 2** (layers 2, 8, 14): per token, k_i = W_ck x_i and z_i = W_cg x_i as FP32 (exact BF16
  products, one rounding), held in the open group; when a token closes a pair,
  c = Σ_i softmax_i(z_i) ⊙ k_i in FP32 (the max subtracted, canonical exp, the two terms summed in
  order 0, 1, one division), rounded to BF16, RMSNorm over the BF16 values; the
  index key comes from this pre-RoPE c; RoPE at the pair's first position; FP4 packing (amax per 16
  with the floor 6 · 2^−9, s = e4m3_rn(amax / 6), codes E2M1 round-to-nearest-even of value / s).
- **Ratio 1** (layer 20): c = W_ck x as a BF16 linear, then the same norm, key, RoPE and packing.
- Layer 20's work runs for **every prompt token** (§16.2): it is the decoder's KV.
- The packing rules are deterministic functions of c, exhaustively checked between the CPU form and
  the GPU form. DeepSeek's server packs with a reciprocal multiply (§20.4); M0 counts where the two
  forms' codes differ on real latents.

### 13.3 Decode attention (`csa_attention`)

Per layer and token: Q is 64 × 512 BF16; the entries are the window ring (≤ 128) and the selected
compressed entries (≤ 512), each one 512-vector used as key and value.

- **Canonical slot order** (FlashMLA's, which decode and prefill share): the window entries in
  ascending position at the front of 128 slots, the compressed selections in ascending position in
  512 slots, invalid slots at the tail of each list with P = 0: ten blocks of 64 entries. The place of
  an entry in the P × V reduction changes its rounding, so prompt/decode equality (§18.3) needs one
  order; `model.py` itself orders the decode ring with invalid slots first and the prompt window with
  them last.
- Entries decoded to BF16 in the tile loader (E4M3 · 2^k and E2M1 · E4M3 values are exact in BF16),
  through the exact conversion instructions; rows padded to an odd number of 16-byte chunks (576 B
  for the 528-byte entries) against bank conflicts; index −1 maps to a zero-filled load, so an empty
  slot can never produce a NaN.
- **Canonical arithmetic**: per 64-entry block, scores q · e in FP32 with a fixed reduction order,
  × 512^−½, and a partial (m_b, l_b, o_b) in FP32, where l_b sums the unrounded exponentials and P is
  rounded to FP16 for P × V with FP32 accumulation (Infernix's exact attention convention,
  model-quality §6.1; the reference rounds P to BF16, and both keep the denominator unrounded, §20.4).
  The ten partials merge unnormalized in ascending block order, then the sink's exp(sink_h − m) joins
  the denominator. Rescale decisions are per head, never shared across a warp's heads as FlashMLA's
  lazy rescale does.
- **Decode at C = 1** spreads (block, head group) work items over ≈ 40 CTAs and merges in block
  order; DeepSeek runs one CTA per query (its batches fill the GPU), which at C = 1 would occupy one
  SM (≈ 10-20 µs per layer, estimated, against the 3-4 µs budget of §9.3). The prompt kernel (§13.6)
  computes the same partials inside one CTA per query. ≈ 84 MFLOP and 0.2 MB per layer and token.
- Output: inverse RoPE on o's last 64 dimensions at the query position (FP32, rounded to BF16), then
  MX8 per group into `wo_a` (K6), fused into the attention epilogue with `model.py`'s two BF16
  roundings kept in registers (DeepSeek's server skips them, §20.4); q's RoPE is fused into the Q
  load, as FlashMLA does.
- Reuse layers take the latest Full or Reindex layer's selection.

### 13.4 The indexer (`csa_indexer`)

| Step | Precision |
|---|---|
| q_h = MXFP4(RoPE(W_qb_idx c_q)), w_h = bf16(W_w x)_h · 2^−6 (128^−½ · 32^−½, exact) | FP8 linear with MX8 input; BF16 `weights_proj` as the reference |
| q_h · k_p per head and position | **exact**: E2M1 × E2M1 products, integer sums per 32-block, power-of-two scales |
| score_p = Σ_h w_h · relu(q_h · k_p) | FP32, heads in ascending order |
| Ratio-2 Full layers: top-512 of the visible positions | DeepSelect's threshold-filter streaming top-k (stream the scores in blocks, keep what beats a running threshold, compact the candidates with a radix select) on a 64-bit total-order key, (orderable FP32 score ≪ 32) \| ~position, so ties go to the lower position whatever the visit order or launch shape; sized for `sm_120`'s shared memory, no clusters; re-sorted by position |
| Layer 20: top-512 of all visible positions; block maximum over 8 positions, the newest block pinned, top-2,048 blocks as the candidates for later layers | same |
| Reindex: its own scores inside the candidates, top-512 | same |

- **Layer 20 selects from all positions**, not from inside its own candidates: `model.py` masks
  scores to C(t) only in the later layers, and DeepInfer does the same (`index_attn.py`). An earlier
  version of this design said "top-512 within C(t)" for layer 20; the two differ only when ties at
  the 512th score spread over more than ≈ 2,047 blocks. Scanning only C(t) at layer 20 is allowed as a
  shortcut when the tie rule keeps it exact.
- **Deviation.** `model.py` computes q · k with a BF16 einsum, the weighted product and the head sum
  in BF16, which rounds scores into ties that `torch.topk` breaks arbitrarily; DeepSeek's server also
  ranks BF16 scores (§20.4). Infernix keeps FP32 logits, which make fewer ties, and reports selection
  agreement against the reference's BF16 scores (§20.1).
- **Decode cost**: layer 20 scans every position (68 B each: 8.9 MB at 128K, 68 MB at 1M), the
  ratio-2 Full layers half of them, the Reindex layers 16,384 candidates (1.1 MB).
- Selections depend only on the stored keys and the query (FN §8.10's rule).

### 13.5 KV formats

The model defines and was trained with its KV formats (report §2.4.4): FP8 window entries (528 B),
FP4 compressed entries (288 B, an E4M3 scale per 16, no global scale), MXFP4 index keys (68 B).
Infernix stores exactly these codes. `--kv-dtype` accepts only `model` for this family; any other
value is refused at startup. KV at 128K: 117 MB; at 1M: 0.89 GB; windows: 3.0 MB per sequence with
the verification room. DeepInfer stores the same three formats (528 / 288 / 68 B).

### 13.6 Prompt kernels

- **Window attention**: per query the last 128 positions (causal in the chunk plus the ring), query
  tiles of 64 sharing their windows.
- **Compressed attention**: **one query per CTA, the 64 MQA heads as the MMA rows** (an m64 tile
  exactly), over that query's own 512 selections and its window, in the same per-query order as the
  decode kernel; FlashMLA's sparse prefill has this structure (per-query index lists over a gathered
  buffer of window and compressed entries). An earlier draft cited a union tile of 64 queries with a
  per-query mask as FN §19.3.8 F5's kernel; Flash-Next rejected the union (FN §8.10) and built one CTA
  per (column, KV head). The per-query order is also what makes prompt and decode state equal (§18.3).
- **Indexer scoring**: a register-tiled exact int8 MMA over queries × positions per head, query
  sub-chunks of 2,048 (GLM §13.3) and a streaming top-k, never materializing the score matrix. At
  128K the three ratio-2 layers score ≈ 17 T multiply-adds each (≈ 0.1 s on int8 tensor cores,
  estimated); layer 20 scores only the decoder's positions. DeepInfer instead materializes BF16
  scores per chunk of at most 8,192 queries (under a 10 GiB cap; its kernel down-clocks on larger
  chunks). An option for M8: FP32 scores per sub-chunk (≈ 128 MB for 512 queries at 128K), which
  makes the exact tie rule a plain sort; chosen by measurement.
- **Attention arithmetic**: ≈ 84 MFLOP per query and layer, ≈ 1.1 s for a 128K prompt over the
  encoder (estimated).

### 13.7 Long context

Default `--max-context 131072`, up to 1,048,576. At 1M the compressed planes take 0.89 GB of VRAM
and the windows are unchanged: no host KV tier is needed (unlike GLM §13.6). The decode indexer scan
grows to ≈ 170 MB per token at 1M (≈ 0.11 ms); prompts grow quadratically in the indexer (§5.5).

- **Position bound.** A verify round reaches K_max + 1 positions past the committed length and the
  drafter's block 5 more, so the RoPE tables, rings and DSpark block cover `--max-context` + K_max + 5
  positions; near the end K is clamped instead (DeepInfer allows max length + verify width + draft
  block).
- **RoPE table** (M0 check). `model.py` and DeepInfer build inv_freq in FP32 on the device with
  `pow`, apply YaRN in FP32 (no mscale), form each angle as the FP32 product position × inv_freq and
  take cos and sin of that FP32 angle. At 128K one ulp of an inv_freq word moves an angle by ≈ 0.01 rad
  (≈ 0.08 rad at 1M; model), so the table is defined bit for bit as torch builds it: M0 dumps
  `model.py`'s `freqs_cis` for both θ values at positions up to 1M and checks Infernix's table against
  it, and if CPU and CUDA `pow` disagree the artifact stores the 2 × 32 inv_freq words.

---

## 14. Engram

### 14.1 Requirements

- 24 rows per Engram layer and token (orders 2-4 × 8 heads), 256 values each; the two layers use
  the same history with different multipliers and primes.
- Rows are **token-determined**: known as soon as a token is sampled (decode) or tokenized
  (prompts).
- Image-span tokens are dead: they join no n-gram and receive no contribution.
- The tables stay off the page cache; RAM is fully budgeted.

### 14.2 Hashing (engine worker)

- The **compressed token map** (129,280 → 99,092 ids) is built by the converter from `tokenizer.json`
  with the reference normalizer (NFKC, NFD, accents stripped, lowercase, whitespace collapsed, the
  single-space sentinel) and stored in the artifact, checked against `engram_compressed_vocab_size`.
  The rules that decide it (DeepInfer `engram_metadata.py`, `engram.py`): each id is decoded alone with
  special tokens kept; text containing U+FFFD keys on the raw token string; a normalization that
  yields the empty string keys on the decoded text; new compressed ids are assigned in order of first
  appearance; the pad is token 2's compressed id. M0 compares the converter's map with `engram.py`'s
  and DeepInfer's on all 129,280 ids.
- The **multipliers** are `numpy.random.default_rng(10007 · layer_id)` draws (odd, bounded by
  (int64 max // 99,092) // 2); the converter stores them, so Infernix never reimplements NumPy's
  generator. The primes are the 48 consecutive distinct primes above 15,999,999 in (layer, order,
  head) order; primes and offsets are stored and checked against `engram_num_embeddings` (each
  table's 24 primes sum to its row count).
- Per position: the compressed ids of the token and its 3 predecessors (the pad id where the history
  is shorter or crosses a dead token); a rolling XOR of id × multiplier in int64 (overflow-free by the
  multipliers' bound); for order n and head h, row = (hash mod prime_{n,h}) + offset_{n,h}.
- The last 3 compressed ids are carried across rounds and kept in prefix snapshots.
- Exact oracle: `inference/engram.py` itself, on a multilingual corpus with image spans and history
  boundaries.

### 14.3 Volume layout

§7.5: FP8 rows 15 per 4 KiB block, MXFP4 rows 30; table-major; a header naming the format.

### 14.4 Reads

Flash-Next's row machinery (FN §12.3, §19.3.4, §19.3.8 F7 and F9) carries over: a direct-mapped host
row cache (2^20 rows: 264 MB for FP8 rows, 136 MB for MXFP4), misses read as 4 KiB direct reads
through the 64-deep polled ring with per-block deduplication, and a volume that is never mapped.

- **Decode**: the engine hashes and reads the sampled token's 48 rows right after sampling. Layer 1's
  rows are needed after layer 0 (≥ 0.13 ms of GPU work plus layer 0's host-served experts, ≈ 1-3 ms
  here); layer 14's after 13 more layers. The gated upload (FN §12.3 S2) lets the round's graph launch
  before the rows land; the `engram` kernel waits on the gate word. Expected exposure: none.
- **Verification**: draft tokens' rows are read while the drafter runs (FN S2).
- **Prompts**: every row of a chunk is hashed and read at staging, deduplicated per block (FN measured
  −20 to −50 % reads from deduplication on its table) and issued from several threads (F7); layer 1
  needs the first chunk's rows before the walk reaches it, layer 14's rows have layers 1-13 of time.
- **Placement**: the Engram volume shares `E:` with an expert volume; its 4 KiB reads go ahead of
  queued expert units (§11.2).

### 14.5 Formats and runtime choice (decision d)

- The artifact's `engram_table.format` records its own table: `fp8_e4m3_e8m0_32` (official sources),
  `mxfp4_e8m0_32` (NVIDIA's current revision, LibertAIDAI) or `ggml_iq4_nl` (ISTA's or Unsloth's
  1017 table: **the first artifact's**, decided 2026-10-24); `ggml_q5_k` and `ggml_q6_k` with option
  O4.
- **`--engram-volume PATH` replaces the table at run time with a volume of any of these formats**, as
  `--ngram-volume` does for Flash-Next (FN §12.5): an FP8 volume in place of the IQ4_NL or an MXFP4
  table restores DeepSeek's own table; a smaller format in place of a larger one saves disk, a lossy
  change that the startup line and the request log label.
- `--engram-reuse PATH` binds an existing volume at conversion after checking sampled rows (equal FP8
  rows, or FP8 rows that MXFP4-quantize to the artifact's codes by ModelOpt's rule);
  `verify_artifact --engram-rows all` checks every row.
- Row decode on the device: FP8 rows bf16(e4m3 · 2^(u−127)), MXFP4 rows bf16(e2m1 · 2^(u−127)), both
  exact. IQ4_NL rows are d · `kvalues_iq4nl[code]` (binary16 × int8), exact in FP32 and rounded once
  to BF16; this format's BF16 values are defined by that rounding.
- `--engram-residency auto|ram|volume`: `ram` only when the whole table fits beside the planned
  expert tier, which never happens on this machine (97-195 GiB), so `auto` reads the volume.
- The prefix cache's fingerprint names the table format, so blocks computed with one table are never
  reused with the other (FN §12.5).

### 14.6 The `engram` kernel

Per Engram layer and token: the 24 rows into MX8 per 32 over the 6,144 values, `wkv` FP8
[25,600 × 6,144] (157 MB), FP32 keys and value, per-stream rstd and gate (signed square root with the
1e-6 clamp, sigmoid), and the stream update with one rounding to BF16: ≈ 0.11 ms per layer per token,
memory-bound on `wkv`.

- **FP8 and MXFP4 rows need no decode.** `model.py` decodes a row to BF16 and re-quantizes it with MX8;
  for a row stored as E4M3 (or E2M1) codes with a UE8M0 scale per 32, that returns the same codes and
  exponent unless MX8's floor binds (u + ⌈log₂(c_max/448)⌉ < −22). The kernel therefore forms the MX8
  operand from the stored codes and exponents by integer shifts and takes the decode path only for
  floor-binding blocks, which M0 counts in the table; DeepSeek's server feeds the stored codes
  straight to its GEMM (§20.4). IQ4_NL rows decode to BF16 (§14.5) and then take MX8.
- **In production** the tables sit in host RAM sharded over the ranks, the GPU reads rows directly
  over RDMA, the hashes are computed on the GPU, and both layers' rows are requested at the start of
  each forward and consumed at layers 1 and 14, as §14.4 does. DeepInfer has no row cache or
  deduplication and publishes no read rates.
- Engram sits between the previous sublayer's residual update and the next mixing step, so layers
  1 and 14 cannot use the producer-epilogue collapse of §15.1; their collapse is a separate step.
  Image-span tokens add nothing and need no row reads.

### 14.7 Cost and exposure

- **Decode**: 48 rows per token, ≈ 196 KB of 4 KiB reads (≈ 2.5 MB/s at 13 tok/s): no bandwidth to
  speak of; the latency is hidden (§14.4) as long as Engram reads never wait behind expert units,
  hence their priority.
- **Prompts**: ≤ 1.57 M reads per 32K before deduplication; at an estimated 0.5-1 M random reads per
  second from one 990 PRO (R0 measures the Windows ring at QD 64), ≈ 1-3 s, overlapping the bank pass.
  The rate is not a given: Maya's 64 synchronous reader threads reached ≈ 0.1 M rows per second
  (1.36 M rows in 13.4 s for a 28K prompt, §4.3), which here would add ≈ 15 s to a 32K prompt. M0's
  probe must show ≥ 0.8 M reads per second, or the Engram rows are striped over the drives as the
  expert units are. A compact build (§5.7) makes this more pressing, because its bank pass is
  shorter.

---

## 15. Single-pass mHC

### 15.1 Kernels (`mhc`, shared with GLM §15)

- **Mixes off the critical path.** A sublayer's input collapse uses the pre the previous sublayer
  computed (§2.2), so its own mixes (24 dot products over the 20,480-wide stream, the rsqrt of the mean
  square, gates, Sinkhorn) run concurrently with its attention or MoE; only its residual update (post,
  comb) and the next sublayer (pre) wait for them. Reads: fn (1.97 MB FP32) and the stream (40 KB).
- **Residual update and collapse in the producer's epilogue** (GLM §15.1): the kernel that produces y
  (K6's `wo_b`, K9's combine) writes S'_k = post_k · y + Σ_j comb[j][k] · S_j for its rows in FP32 with
  one rounding to BF16, then the next sublayer's input x = Σ_s pre_s · S'_s (with the pre that
  sublayer will use, already computed), and the partial sums of squares for its RMSNorm and for the
  next mixes, taken over the BF16-rounded x and S' (as `model.py` and DeepGEMM's fused mHC both do).
  DeepGEMM's `mega_mhc` fuses the same epilogue (post, collapse, norm, cast, statistics) in one pass.
  The Engram layers 1 and 14 are the exception (§14.6).
- **Sinkhorn**: one warp in FP32, one column normalization then 19 row/column rounds with eps 1e-6,
  reproduced iteration for iteration.
- **Cost**: ≈ 2 µs per sublayer, ≈ 0.16 ms per token, none of it on the critical path except the
  epilogues.

### 15.2 Numerics

- Mixing weights, Sinkhorn, pre, post and comb in FP32; streams in BF16; the update in FP32 with one
  rounding, exactly as the reference (GLM's reference rounded twice; this one does not).
- Fixed orders: s = 0..3 for the collapse, j = 0..3 for the comb sum, then + post · y; qualified
  against FP64 with the reference's single rounding. DeepSeek's server forms fl32(post · y) first and
  adds comb[j] · S_j for j = 0..3 by fused multiply-adds, and computes the mixes on TF32 tensor cores
  with approximate exp and reciprocal (§20.4); Infernix keeps `model.py`'s FP32.
- Confirmed constants (DeepInfer): post scale 2.0, 20 Sinkhorn iterations, eps 1e-6 for pre and
  Sinkhorn, norm eps 1e-20 with the norm weight folded into fn.
- The first sublayer's pre is one-hot on stream 0; the head collapses with the last FFN's pre.

---

## 16. Prompt processing: the encoder walk and the decoder rule

### 16.1 The two rules (decision a)

| `--prompt-decoder` | Prompt tokens through the encoder | Decoder positions evaluated | Equals |
|---|---|---|---|
| `replay` | all | the last min(T, 140) (`--replay-keep N`, default 140, at least 134), window attention truncated at the replay start | DeepSeek's served model (report §3.2.2; DeepInfer; vLLM #56752 and SGLang with 128) |
| **`full`** (default, decision g) | all | the exact suffix: layer l evaluates the last 1 + (39 − l) · 127 positions after forming the 127 window entries before them | the reference forward (every prompt token through all 40 layers), output for output (DS4's method) |

- Both rules are deterministic and request-invariant (§18.3); the rule is part of the prefix cache's
  fingerprint, the startup line and the request log.
- The rule is a declared model mode, chosen by the owner, not a fast path: `--reproducible`
  does not change it, and each rule has its own exact route.
- **CausalScoring runs the full rule.** Replay defines the state at the end of a prompt, not a logit
  at every position; scoring each position under replay would cost a decoder replay per position.
  CausalScoring therefore computes every position through all 40 layers, as the reference does, and
  says so in its output.

**Measured since decision a** (Maya #342, §4.3): replay against the exact rule gave KL 1.6e-2 /
6.6e-3 / 3.2e-4 after 4K / 8K / 26K-token prompts, with greedy text diverging at tokens 2 and 8, for
24 / 24 / 15 % less prompt time on an SSD-bound 7900 XTX. The same engine's exact rule differed from
its own full forward by 6.4e-4 / 6.1e-3 / 8.8e-5 (summation order), so replay's effect is clear at
4K, within that noise at 8K and 3.6 × it at 26K; Infernix's rules have no such noise (§18.3), so
here the difference is replay's alone. On this machine §5.5's model gives the full rule +7-13 %
prompt time: a 4K prompt 11.8 → 13.1 s with MXFP4 records and 6.4 → 7.2 s with 3-bit ones; 32K at
2,770 → 2,490 tok/s. Replay is the model DeepSeek serves and simulated in post-training, and GSM8K
did not separate the two (§4.2); a KL says that they differ, not which is better. But both engines
that measured it default to the exact rule, quality is measured by teacher-forced scoring, which runs
every position through all 40 layers (§5.7's figures, and CausalScoring here), and the cost here is
small. The owner therefore made `full` the default on 2026-10-24 (decision g), with `replay` kept as
an option.

### 16.2 The encoder walk

- Layer-major **walk spans** (FN §19.3.8 F4) over layers 0-19, plus layer 20's prefix work for every
  prompt token: the collapse, `attn_norm`, the compressor and the index keys, which are the decoder's
  KV (SGLang #42901 trims the rest of layer 20 the same way).
- **Spans up to 128K tokens** (Flash-Next's `kWalkMaxTokens` is 65,536; this family's KV is small
  enough to double it): the walk's stream state is 40 KB per token (4 × 5,120 BF16), 5.2 GB at 128K,
  lent from pages (FN F4's lease); the decoder and the head run after the span that holds the
  prompt's end.
- **MoE groups** of up to 16,384 columns (FN F4) and **expert groups** of G experts in ascending id
  (GLM §17.2) with an FP32 accumulator per token: bit-identical to whole-layer processing, with
  ≈ 0.85 GiB of staging instead of a whole layer's experts.
- **Expert supply**: route-aware, only the experts the span touches; SSD-resident records read once
  per span at full striped bandwidth into a deep landing ring (GLM §17.2: 64-96 records); the next
  layer's SSD records read ahead (a prompt touches nearly every expert, so these reads are not
  guesses); RAM records prestaged while attention runs.
- **Engram rows** read at staging (§14.4).
- **Chunk grid**: fixed absolute boundaries, for the exact route's request invariance (engine
  architecture §3.1).

### 16.3 The decoder under each rule

- **replay**: after the span holding the prompt's end, decoder layers 20-39 run on the last
  min(T, 140) positions as narrow work (≤ 140 columns), window attention truncated at the replay
  start s (entries max(s, i − 127) … i), compressed entries from layer 20 for every position. When a
  prompt continues a cached prefix with T < 140, the replay reaches into cached positions; their
  layer-20 inputs come from the snapshot (§18.2).
- **full**, a fresh prompt or one resumed from a tap inside an earlier prompt: layer l evaluates its
  suffix; layer 20 needs layer-20 inputs for the last 2,541 positions, from the walk or, for cached
  positions, from the snapshot. A prompt shorter than a layer's suffix runs that layer on all of its
  tokens.
- **full**, a prompt that continues an endpoint (the end of an earlier prompt or of a reply): the
  endpoint's snapshot holds the exact decoder windows (§18.2), so the new tokens run all 40 layers
  as decode does and no suffix is recomputed. Under the full rule these windows are the reference
  computation; under replay they are redefined at every prompt end, so replay cannot do this.
- Replay's decoder experts are thin (≈ 2 tokens each): the CPU computes the non-VRAM ones (§12.4).
  The full rule's are 2-40 tokens each from layer 39 to layer 20.
- DSpark's window seeds (main-feature entries of the last 128 positions) come from the decoder
  positions either rule evaluates.

### 16.4 Short appends

An append of T tokens to a cached conversation runs the same walk over T tokens; only the experts
those tokens touch are read (54 % of the bank at 50 tokens, §5.5), the narrow routes serve experts
with n ≤ 8 columns, and the CPU takes thin non-resident ones. Under `full` that is all (§16.3);
under `replay` the decoder also replays 140 positions, touching ≈ 89 % of each decoder layer's
experts whatever T is (this was missing from earlier versions of §5.5, which counted only the new
tokens for both rules). Model, three volumes (`append_model.py` in the working notes):

| Append of T tokens | EXL3 artifact, `full` | EXL3, `replay` | MXFP4, `full` | MXFP4, `replay` |
|---:|---:|---:|---:|---:|
| 16 | 1.0 s | 2.3 s | 2.2 s | 5.0 s |
| 50 | 2.0 s | 3.4 s | 4.9 s | 7.4 s |
| 100 | 4.5 s | 5.3 s | 8.9 s | 9.9 s |
| 200 | 7.0 s | 6.5 s | 12.3 s | 11.6 s |

So for chat and agent turns, mostly short appends, the exact rule is the faster one; from about
200 new tokens replay saves ≈ 0.5-1 s, as it does on fresh prompts (§16.1). A continuing turn also
skips re-reading the model's previous reply only if that decode-computed state is published and the
next prompt re-renders the reply unchanged (§18.3; true for agent turns with tools).

### 16.5 After prefill

No promotions after a walked prompt (L6); the prompt's routes seed both policies; when the walk
returns its lent pages, the free-page refill (§10.2) fills them with the hottest RAM experts.

---

## 17. Speculative decoding: DSpark

### 17.1 Expected value

- Every verify column brings its own experts (§5.4): short drafts can pay (K = 1: +6-27 %, model),
  DeepSeek's five-token blocks lose on this machine; JigSawPT measured −4 % / ±0 at K = 2 on the same
  card (§4.2).
- DSpark's confidence head predicts each draft position's acceptance, which with the measured union
  growth is what the cost model needs each round.
- Strata and Maya measured DSpark's sign flip with VRAM pressure: −18 to −61 % where the trunk's
  experts did not fit, +37 to +55 % where they did (§4.3). Both decide once, at load or by an A/B. The
  cost model here decides per round and also charges the pages the drafter holds (§19.1). With
  compact experts (§5.7) it is more likely to pay: +36-46 % for the EXL3 artifact in the model.

### 17.2 The drafter

- Structure §2.2. Its window of main-feature entries (3 layers × 134 × 528 B, the verification room
  of §13.1) is seeded by the prompt's decoder positions. As in DeepInfer (`models/mtp/dflash.py`),
  every verify column's main-feature key is written into this window inside the verify forward; the
  next draft pass's block keys overwrite the rejected columns' slots before anything reads them, so
  there is no separate append step.
- **Confidence head** (DeepInfer, matching §2.2): sigmoid(clamp(([h ; e_m] · w) / T_step, ±60)) with h
  the collapsed hidden state **before** the final norm, e_m the Markov embedding of the previous
  draft, w cast to BF16, an FP32 result and a per-step temperature (default 1.0); a draft's survival
  is the running product. DeepSeek samples drafts with the request's sampler, one random stream per
  draft position.
- **Draft experts**: residency-bounded routing (FN §11.2, GLM §10.6): top-3 among the DSpark experts
  in the VRAM pool, plus the best-scored missing one computed by the CPU from the RAM pool (Maya's
  `MTP_KEEP`, GLM §4.3). This changes acceptance only, never verified output.
- **Markov head**: draft i adds head_m(embed_m(draft_{i−1})) in order (≈ 66 MB of BF16 per step);
  drafts are argmax on greedy rounds and use Infernix's sampled-draft path otherwise.
- **Cost per draft round**: three layers' attention and shared experts (≈ 0.5 GB of FP8), the head over
  5 positions (1.32 GB, once) and the Markov steps: ≈ 1.5 ms (model).

### 17.3 Draft length

FN §11.3's policy maximizes E[tokens(K)] / E[t_round(K)] with t_round(K) = t_gpu(K + 1) +
t_host(U_{K+1}) + t_ssd(U_{K+1}) + t_draft(K): U from the online union measurement (σ, §5.4), each
position's acceptance from the confidence head, calibrated online against observed outcomes (DSpark's
sequential temperature scaling); K ∈ {0, …, 5} with K = 0 always a candidate; experts used only by
rejected columns are not credited to the caches. Expected: K ∈ {0, 1, 2}, more on verbatim text and
code edits.

- **The drafter always computes its trained 5-entry block.** Block attention inside the drafter is
  non-causal, so a shorter block changes every draft, the first included. K only truncates what is
  verified (and the Markov and head steps past K). DeepInfer ties the block length to its draft-count
  setting, so its published nextn = 3 chart ran an untrained 3-entry block.
- **Calibration signal** (DeepInfer `fused_sample.py`): on sampled rounds the verifier knows each
  draft's acceptance probability, Σ min(p, q) (for a point-mass draft, p of the draft token), and
  calibrates the confidence head against that instead of the 0/1 outcome: the same mean, lower
  variance, so K settles faster. Greedy rounds observe 0/1 as before.
- **DeepSeek's production figures**: drafting 5 and verifying a confidence-chosen prefix cut the
  verified drafts per step from 5 to 3.2-4.4 while the accepted length fell only from 3.29 to 3.20
  (EP64 serving; its throughput gain ranged from −1.5 % at 16 sequences per rank to +11 % at 96).
  Here the union cost makes every verified column far dearer, which is why K is chosen per round.

### 17.4 N-gram copy proposals

`--ngram-draft-tokens` through the same cost model; they need no drafter or DSpark pages.
JigSawPT measured 97 % acceptance on verbatim text.

### 17.5 Default

`--spec dspark` exists from M7. The default becomes `auto` (the cost model, including off) only if
M7 measures it faster than plain decode on the NRS corpus; otherwise speculation stays off by
default.

---

## 18. Prefix cache, preemption and invariance

### 18.1 Blocks

64-token content-addressed blocks of the four Full layers' compressed entries and index keys: 57 KB
per block (890 B per token). A 32K conversation is 29 MB.

### 18.2 Snapshots

At the hybrid prefix cache's tap points (spec §17's binding, as for Qwen4Exp):

| Part | Size | Why |
|---|---:|---|
| Encoder windows (layers 0-19) and compressor state | 1.41 MB + ≤ 12 KB (at most one open group per layer; none at even positions) | resume the encoder exactly; DeepSeek's lossy encoder replay is not needed |
| Endpoints: the last position's head input (the collapsed hidden state, 10 KB) | 10 KB | an exact repeat of a whole prompt (or "regenerate") samples its first token without recomputing; otherwise the resume backs off one position, which the verification room of §13.1 allows |
| Engram history | 12 B | the n-gram state |
| DSpark window (main-feature entries) | 0.2 MB | when DSpark is on |
| `full`, at endpoints (the end of a prompt or a reply): decoder windows (layers 20-39) | 1.41 MB | an append continues as decode does (§16.3) |
| `full`, at taps inside a prompt: layer-20 inputs of the last 2,540 positions | 104 MB | a resume there recomputes the exact suffix |
| `replay`: layer-20 inputs of the last 139 positions | 5.7 MB | a short append's replay reaches into cached positions |

A snapshot is ≈ 3 MB at an endpoint under the default `full` rule (≈ 106 MB at a tap inside a
prompt) and ≈ 7.3 MB under `replay`, where decoder windows are never stored because replay
recomputes them by definition. DeepSeek instead caps a prefix hit at L − 134 positions so the tail
re-runs the whole model; here a 134-token re-run would touch ≈ 88 % of each layer's experts, so the
snapshots are kept. DeepSeek's cap also sits uneasily with its own 140-position replay (its hit logic
is not in the released code), so its served output after a cache hit is not pinned down; Infernix
claims request invariance for its own rules (§18.3), not bit-parity with DeepSeek after hits.

### 18.3 Request invariance

Both rules make a request's output a function of the request, the artifact and the options only
(model-quality §6.1): the encoder's arithmetic is exact; the replay's decoder state depends only on
the prompt's last 140 tokens and the encoder state, the suffix's on the last 2,541; a snapshot
restores exactly what a cold run computes at the same point.

**Prompt/decode equivalence is a requirement of this family.** Every piece of published state
(encoder windows, compressor state, compressed entries, index keys, Engram history, layer-20
inputs, and under `full` the decoder windows) must be the same bits whether a prompt walk (any
chunking), a decode step or a verify column computed it. The pieces already hold: the linears and
experts are width-invariant (MX8 and E3A16, §9.5), mHC and norms work per token, and attention runs
one query at a time in one canonical slot order and block merge in prompts and decode (§13.3,
§13.6). What this adds: the compressor pools and the indexer sums its heads in a fixed per-query
order on every route, and top-k ties resolve by a total-order key (§13.4). DeepSeek's engine asserts
the same property: DeepInfer's `tests/models/test_kl.py` (one case: 3 image-and-text prompts of
16,384 tokens per rank, 1,024 teacher-forced tokens, 5-token DSpark drafts, two-way context
parallelism, bounded replay off, so the full rule) requires bitwise-equal FP32 token
log-probabilities between decoding and one full prefill, in its batch-invariant mode. That mode fixes
every split and disables split-KV attention but leaves DeepSelect's tie order alone, so DeepSeek's
equality relies on no ties at a selection cutoff; Infernix's total-order key removes that caveat.
DeepSeek does not run the mode in production (it is slower).

Consequences:

- The fixed chunk grid is unnecessary for this family: any chunking gives the same state.
- **Decode-computed endpoints are published.** A continuing turn walks only the new message, not the
  model's previous reply; without this, a reply of 200 tokens or more costs a bank pass on the next
  turn (§5.5: ≈ 7-13 s). This applies where the next prompt re-renders the reply as the ids the model
  generated: agent turns with tools declared (earlier reasoning stays in the prompt) and chat-mode
  turns. A thinking chat without tools drops the reply's reasoning, so its next prompt departs right
  after the previous `<｜Assistant｜>` and the reply's answer is walked again whatever is published;
  newline stripping and re-serialized DSML values can also cut reuse inside a reply (§21, M9
  measures how often on real agent transcripts).
- Engine architecture §3.1 publishes only prefill-computed state on the chunk grid, a rule made for
  the GDN families, whose decode and chunked prefill cannot match. This family is an exception with
  this reason (§24).
- Under `replay` the decoder windows are redefined at every prompt end, so only the encoder-side
  state is published, and the DSpark window is reseeded from the replay at a resume, as a cold run
  does.
- M0 and M2 test it: one sequence computed as T = 1 decode steps, as verify widths 2-6 and as prompt
  walks of several chunkings, its published state compared bitwise (the Python reference first, then
  the kernels).

### 18.4 Preemption

The exact route never pauses a request: the Program reserves its whole KV extent (engine architecture
§3.1). Here that extent is 890 B per token, so reserving a 128K request costs 117 MB; the `preemption`
fast path (Replay recovery) is not registered for this family.

### 18.5 Tiers

Host tier default 2 GiB (`--host-context-mib`; ≈ 350 endpoint snapshots per GiB under the default
rule), and the file tier as the spec defines it. DeepSeek keeps persistent KV on SSD for at least
72 hours (report §3.2.1; DeepInfer's deployment notes give 3 days for agent traffic and 7 for chat);
at 29 MB per 32K conversation the file tier can keep many agent sessions for days.

---

## 19. Memory plans

### 19.1 VRAM (32,607 MiB card, no display, C = 1)

| Item | Default, GiB | With DSpark, GiB |
|---|---:|---:|
| In use before loading (context, libraries) | 0.55 | 0.55 |
| Dense text: attention 4.72, shared experts 1.32, head 1.23, routers and mHC 0.29, Engram `wkv` 0.29, compressors and indexers 0.08 | **7.94** | 7.94 |
| DSpark dense | — | 0.65 |
| Compressed entries and index keys, 128K (elastic) | 0.11 | 0.11 |
| Windows (40 layers, + 3 DSpark) | < 0.01 | < 0.01 |
| Decode workspace, mailboxes, residency tables and page lists, io | 0.70 | 0.70 |
| Staging (128 pages) | 0.53 | 0.53 |
| CUDA graph reserve; headroom | 0.25; 0.25 | 0.25; 0.25 |
| DSpark expert share (192 pages, §10.6) | — | 0.80 |
| **Page pool** (of the 30.68 GiB budget) | **20.35** | 18.90 |
| Pages of 4.25 MiB | **≈ 4,900** (≈ 1,250-1,600 experts, 8-10 % of 15,360) | ≈ 4,550 |

The MXFP4 artifact (M11) stages 32 frames (0.56 GiB) and holds 96 DSpark frames (1.68 GiB): a frame
pool of 20.32 / 17.99 GiB, **1,159** / 1,026 frames of 17.95 MiB (7.5 %). At 1M actual context the
KV takes 0.72 GiB more (−173 pages; MXFP4 −41 frames). A further lane costs its windows and KV extent
only (≈ 3 MB + 890 B per token): there is no recurrent state.

### 19.2 Host RAM (95.8 GiB visible)

| Item | GiB |
|---|---:|
| RAM expert tier: pinned up to the pin room, unpinned past it; includes 256 landing pages (1.06) and 128 bounce pages (0.53) (MXFP4: a 64-slot landing ring, 1.12, and 32 bounce slots, 0.56) | the rest |
| Token embedding, pinned | 1.23 |
| Engram row cache | 0.26 (FP8 rows) / 0.13 (MXFP4) / 0.14 (IQ4_NL) |
| Prefix Host tier | 2.0 (default) |
| Program pinned buffers | 0.25 |
| Process (tokenizer, server, frontend) | 1.5 |
| Reserve `--ram-headroom-mib`; growth allowance | 2.0; 1.0 |
| DSpark RAM share (with DSpark only) | 2.4 (MXFP4: 3.37) |

| Available at startup | Expert tier | Pinned / unpinned | EXL3: pages, share of experts, with VRAM | MXFP4: slots (ring excluded), share, with VRAM |
|---:|---:|---:|---:|---:|
| 88 GiB (other programs closed) | 79.8 GiB | ≈ 72 / 7.8 GiB | ≈ 19,200, ≈ 41 %, ≈ 51 % | 4,485, 29.2 %, 36.7 % |
| 79 GiB | 70.8 GiB | 70.8 / 0 | ≈ 17,050, ≈ 37 %, ≈ 46 % | 3,972, 25.9 %, 33.4 % |
| 50 GiB (other sessions running) | 41.8 GiB | 41.8 / 0 | ≈ 10,050, ≈ 21 %, ≈ 30 % | 2,318, 15.1 %, 22.6 % |

Every GiB other programs release becomes ≈ 240 more pages (MXFP4: ≈ 57 slots); the startup ledger
prints the split. The EXL3 shares are estimates: the tier holds warm experts of mixed widths.

### 19.3 Disk

| Item | Size | Where (default) |
|---|---:|---|
| Source shards during conversion | ≤ 2 × 7.4 GB at a time; the Engram shards are streamed | any drive (deleted shard by shard, §7.6) |
| Main artifact | ≈ 11 GB | any |
| Expert volumes, w = (4, 4, 4) | 3 × 63.9 GB (MXFP4, M11: 3 × 98.8 GB) | three drives chosen when the owner frees space (§11.4) |
| Engram volume (FP8 / MXFP4 / IQ4_NL) | 209.7 / 104.9 / 112.4 GB, one per checkpoint | one drive, or striped (§14.7) |
| Expert state file | < 1 MB | beside the artifact |
| Prefix-cache file tier | 29 MB per 32K conversation | configurable |

---

## 20. Numerics and qualification

### 20.1 Op oracles

Each Op follows [op development](op-development.md): an independent naive FP32/FP64 oracle, exact
oracles for exact transforms, packed inputs decoded independently with their stored scales, real
model shapes.

| Op | Oracle and criteria |
|---|---|
| MX8 codec (`ops/common/mx8.h`) | **Exact, exhaustive** over every BF16 input and block exponent: the CPU integer form, the GPU `cvt.rn.satfinite` form, a Python port of `act_quant` and DeepGEMM's integer exponent rule agree, including the 1e-4 floor (exponent −22) and the ceiling rule at powers of two |
| `fp8_e4m3fn_b32_e8m0` Linear, decode and prompt routes | **Bit-exact** between routes and against the scalar form at T = 1..8 and prompt widths, every DeepSeek shape including grouped `wo_a`; column invariance for T ≤ 8; FP64 oracle of the same quantized operands within the FP32 rounding bound |
| EXL3 codec (shared with GLM) at K = 2-5 | **Exact**: the scalar E3A16 form against GLM §20.1's oracle (an FP64 evaluation of ExLlamaV3's decode for the codebook values, exact integer sums), every width, both input widths (5,120 and 2,304), and the pack's sampled experts decoded by ExLlamaV3 at the producing version within its FP16 rounding of the codebook |
| `offloaded_sparse_moe`, EXL3 records in pages, MX8 → E3A16 | **Bit-exact** across GPU pages (any page order in the page list), staging, CPU on every ISA, unpinned pages, every unit landing order, n = 1..8 and the wide route; calls mixing all four widths; the routing, clamp and combine checks of the MXFP4 row below |
| `offloaded_sparse_moe`, MXFP4 and NVFP4 records, MX8 (M11) | **Bit-exact** across GPU frame, staging, CPU on every ISA, unpinned record, every unit landing order, n = 1..8 and the wide route; MXFP4 and NVFP4 records of one expert give the same bits; routing exact against an FP64 √softplus/bias/top-6 oracle on near-tie-free inputs, ties by lower id, logits beyond ±20; clamp boundaries (±10, NaN, Inf); the routing weight before the down projection |
| `csa_compressor`, index keys | FP64 pooling and norm; FP4 and MXFP4 packing exact against a Python port of `fp4_act_quant`, exhaustive over block exponents |
| `csa_indexer` | exact q · k against integer arithmetic; FP64 weighted sums; exact selection on near-tie-free fixtures; selection agreement on real activations, also against the reference's BF16 scores (reported) |
| `csa_attention` | FP64 softmax attention with the sink over the decoded entries; replay's window truncation; inverse RoPE |
| `mhc` | FP64 including all 20 Sinkhorn rounds; one-rounding stream update |
| `engram` | hash ids exact against `engram.py`; row decode exact for both formats; the block against FP64 |
| DSpark attention, Markov and confidence heads | FP64 |
| Records and volumes | Every EXL3 record's tensors equal the pack's bytes and its width equals the plan's; every unit read through the striped reader equals the converter's bytes, in permuted completion orders and at fault seams; page lists address every page of every width; Engram volumes equal the source rows (FP8) or ModelOpt's MXFP4 rows; (M11) MX recovery from NVIDIA's NVFP4 equals the official MXFP4 bytes on the whole September bank |

### 20.2 Reference forwards and block checks

`tools/deepseek_v41/reference_forward.py` (a qualification tool, not a product route): teacher-forced
and layer-major, with FP64 accumulation over `model.py`'s quantized operands (MX8 activations, FP8 and
FP4 KV entries, BF16 roundings where `model.py` has them). It has both prompt rules and two weight
sources:

- **the converted artifact**, streaming each layer's weights through an independent Python decoder
  (for the EXL3 artifact, GLM §20.1's FP64 EXL3 decoder of each expert's matrix, applied after MX8)
  and Engram rows from the volume: the reference of the execution gate;
- **the official release**, streaming one shard at a time by download and delete (510 GB read, at
  most ≈ 15 GB on disk; the Engram rows by range read): the original model's logits for the quality
  report, without an MXFP4 artifact. Layer-major order lets every scored text pass a layer before
  its shard is deleted. `block_check.py` compares each layer type, the head and the DSpark blocks with an
FP64 evaluation on the model's own activations; M0 checks Transformers 5.6's `deepseek_v41`
implementation against `model.py` and uses its modules for single-module agreement where they
match.

### 20.3 Gates (fixed now, before any measurement)

Teacher-forced on frozen texts in the DeepSeek encoding with thinking (code, documents, chat, CJK, a
DSML tool transcript; ≥ 2,500 scored positions), including ≥ 512 scored positions after a prompt of
≥ 32K tokens, where layer 20's candidate cut and every top-512 bind; DS4's official-continuation
fixtures (DeepSeek API log-probabilities, `deepseek-v4.1-flash-20260910`) are scored as an external
check for the September checkpoint.

| Gate | Against | Mean KL | p99 per-token KL | Top-1 agreement | Perplexity change |
|---|---|---:|---:|---:|---:|
| **Execution**, each prompt rule | the reference forward of the same rule over the artifact's own values (§20.2) | ≤ 0.001 nats | ≤ 0.03 | ≥ 99.5 % | ≤ 0.1 % |
| (M11) MXFP4 records against NVFP4 records | each other | bit-identical | | | |
| (M11) Official source against NVIDIA source (September, FP8 Engram) | each other | bit-identical | | | |

The execution gate is tighter than GLM §20.3's because Infernix computes the reference's own
quantized arithmetic; what remains is the accumulation order, the indexer's score precision
(§20.4) and, for EXL3 experts, E3A16's 14-bit block activations against the reference's FP64
evaluation of the same decoded matrices (GLM §9.5 bounds them far below the format's error).

**Reported without a gate** (the owner's choices): **the EXL3 artifact against the original model**:
its logits against the official release's (§20.2) on the same texts (KL, top-1, perplexity by
category: code, agent, documents, chat, CJK, prose), beside the pack's published KL 0.0201 and
top-1 95.96 % (§5.7), with **the IQ4_NL table's share** separated: the artifact-weights reference
forward run once with the IQ4_NL rows and once with the official FP8 rows (streamed from the release),
so the experts' and the table's contributions are reported apart; replay against full (KL and top-1
on the same texts, and the official fixtures' NLL under each); MXFP4 Engram tables against FP8 (KL,
top-1, perplexity), beside diffbot's 0.035 for vLLM's W4A4 + MXFP4-Engram serving; and, for the
September checkpoint, Infernix's NLL on the official fixtures beside DS4's Q2 and Q4 references
(§4.1).

### 20.4 Deviations from `inference/model.py`

| Point | `model.py` | Infernix | Why |
|---|---|---|---|
| FP8/FP4 linear accumulation | FP32 tensor-core products per 32-block, FP32 block adds | exact sums, one rounding | placement invariance; more precise |
| Indexer scores | BF16 einsum, BF16 product and head sum | exact q · k, FP32 weighted sum | fewer ties; selection agreement against BF16-rounded scores is reported (§20.1). DeepSeek's server also ranks BF16 scores (DeepGEMM's MQA logits are BF16 on SM100); an earlier version of this row claimed FP32 training kernels |
| Top-k ties | `torch.topk` (unspecified) | lower position or lower expert id | determinism |
| Attention P × V | BF16 P | FP16 P, FP32 accumulation | Infernix's exact attention convention; FP64-qualified |
| softplus, exp, SiLU, sigmoid | torch CUDA | canonical implementations | CPU/GPU identity |
| mHC sums, norms | torch's order | fixed orders | determinism |

Every BF16, FP8 and FP4 rounding point of `model.py` is kept.

**DeepSeek's server differs from `model.py` too** (DeepInfer at `9bd9cf24` with the libraries it pins:
DeepGEMM `1fc9c669`, FlashMLA `2e5429fc`, DeepSelect `bfa4507d`, TileKernels 2.1.0; read 2026-10-24).
Infernix follows `model.py`, the published reference; these are the places where "the served model"
and the reference part, so the §20.3 reports can explain small differences from API log-probabilities:

| Point | `model.py` | DeepSeek's server |
|---|---|---|
| FP8 and FP8 × FP4 GEMMs | FP32 products per 32-block, FP32 block adds | the hardware block-scaled MMA (UE8M0 per 32 on both operands), FP32 accumulation in an undocumented order, no split-K |
| mHC mixing coefficients | FP32 | a TF32 tensor-core GEMM (`fn` read at a 10-bit mantissa), 16 fixed split-K partials in batch-invariant mode and a token-count-dependent split otherwise; approximate exp and reciprocal in sigmoid and Sinkhorn, except on layers 0, 1 and 14, whose Sinkhorn (TileKernels) divides: two Sinkhorn arithmetics in one forward |
| Routed experts after gate and up | BF16 outputs clamped, SwiGLU, h rounded to BF16, MX8 for down; combine in ascending id, shared expert last | the same clamp; approximate exp and reciprocal in SiLU; (silu · up) · w in FP32 quantized to FP8 **without** the BF16 rounding of h; combine with the shared expert first, then routed experts in score order |
| Attention output | FP32 → BF16, inverse RoPE → BF16, MX8 | inverse RoPE in FP32 and MX8 straight from FP32, skipping both BF16 roundings |
| Compressor pooling | exact functions | fast exp and division (the same max-subtracted two-term form) |
| FP4 packing of compressed entries | s = e4m3(amax / 6), codes e2m1(x / s) | s = e4m3(min(max(amax, 6 · 2^−9) · fl32(1/6), 448)), codes e2m1(x · fl32(1/s)): the scales are always equal for BF16 inputs; codes differ only where x / s lands exactly on an E2M1 midpoint and s is not a power of two |
| Indexer scores | q · k rounded to BF16, times the BF16 weight, heads summed in FP32, one rounding | relu and a BF16 rounding of each q · k, then a BF16 multiply-add chain over heads in four interleaved partial sums; the factor 2^−6 omitted (it changes no ranking) |
| Top-k ties (indexer, candidate blocks) | `torch.topk` | DeepSelect: deterministic, but ties at the cutoff fill in the kernel's visit order and depend on the launch shape |
| Engram rows into `wkv` | FP8 rows decoded to BF16, MX8 | the stored FP8 codes and scales fed straight to the FP8 GEMM: the same values except in 32-blocks where MX8's 2^−22 floor binds |
| Speculative sampling | — | residual max(q − e^(log p − 5·10^−5), 0), which keeps outputs within ≈ 5·10^−5 relative of exact; Infernix's sampled verification stays exact |

Confirmed equal to `model.py`: MX8 (all three implementations compute e_b = max(⌈log₂(a_b/448)⌉, −22),
DeepGEMM by an integer rule on the bits), the `wo_a` → `wo_b` chain (DeepGEMM rounds to BF16 before
the cast, "bitwise" as a BF16 output followed by the standalone cast), the MXFP4 index-key floor
(6 · 2^−126 in all three), the router (√softplus with the z > 20 threshold, lower id on ties,
w = (s / (Σ + 1e-20)) · 1.5), attention's BF16 P with an FP32 denominator over the unrounded
exponentials, and the KV formats. M0 counts the FP4 packing differences on real latents and the
floor-binding Engram blocks in the FP8 table; whether DeepSeek trained with the reciprocal FP4 form
or the TF32 mixes is not known (TileKernels ships backward kernels too).

### 20.5 Engine and system tests

- **Placement invariance end to end**: greedy ids identical with no VRAM experts, small and full
  VRAM caches; RAM tier caps of 16, 48 GiB and full; unpinned records forced; one, two and three
  volumes; lookahead off and on; every CPU ISA; random unit landing order; CPU cap 0 and 16;
  shuffled and fragmented page assignments (the same expert in different pages across runs).
- **Page pool**: under eviction pressure with all four widths no page is shared or leaked (an audit
  after every round in the test build); a single-width artifact (Qwen4Exp) keeps every decision bit
  for bit.
- **Record formats (M11)**: an MXFP4 and an NVFP4 artifact of one checkpoint give identical ids.
- **Prompt rules**: replay is identical across chunk sizes and prefix-cache states; the full rule
  matches the reference forward's full-prompt logits within the execution gate; replay matches the
  reference's replay variant. Before any kernel, a CPU test checks the replay's visible-window sets
  against a naive mask oracle for every replay start and chunk split, as DeepInfer's
  `examples/yoco/yoco_equiv.py` does in seconds (it catches off-by-one rectangles that model gates see
  only as small KL).
- **Prompt/decode equivalence** (§18.3): one sequence as T = 1 decode steps, as verify widths 2-6
  with every rejection count, and as prompt walks of several chunkings gives bitwise-identical
  published state and logits.
- **Prefix cache**: cold against warm resume at every tap point and at prompt and reply endpoints
  under both rules.
- **Speculation**: greedy speculative output equals plain greedy output (verify widths ≤ 8 columns are
  column-invariant, FN §11.3). Sampled speculation is checked statistically with DeepInfer's
  probability-integral-transform test (`tests/models/test_sampler.py`: 128 trajectories × 2,048
  tokens per setting at greedy, T 1.0 / top_p 0.95, T 8.0 / top_p 0.95 and T 1.0 / top_p 1.0, with 5
  drafts, Kolmogorov-Smirnov on the PIT values).
- **The full rule's suffix**: a bitwise self-check that the suffix walk (§2.3) equals running every
  prompt position through all 40 layers, as DeepSeek's mask-against-compact replay test does
  (`examples/yoco/run_yoco_equiv_l2.sh`, which also compares the drafter's confidences).
- **Engram**: a run-time volume swap gives the labelled output; Engram read faults reach the
  recoverable path.
- **Faults, tier audit and policy conformance** as GLM §20.4 (unit read errors, a stalled or missing
  volume, a silent CPU service; every expert in exactly one tier; host decisions equal the replay tool
  step for step).
- **Speed corpora**: FN's non-repeating session (NRS) for decode, reported by halves; a fixed
  teacher-forced routing corpus for hit-rate comparisons (L20).

---

## 21. Frontend and serving

- **Tokenizer.** Byte-level BPE (128,000 BPE tokens + 1,283 added) with DeepSeek's
  pre-tokenizer: `\p{N}{1,3}` isolated, CJK and kana runs isolated, the third split regex, then
  byte-level without its own regex; an empty normalizer. The scanner gains this variant only; other
  patterns stay refused. Exact-oracle test against Hugging Face `tokenizers` over a multilingual corpus
  and every special token. BOS 0; EOS 1, the only stop token (there is no generation config); pad 2.
- **Encoding.** `encoding/encoding.py` of the 1017 revision (with its 2026-10-16 fix) is the
  authority; the September repository's community Jinja template (which adds a "medium" alias and
  predates the fix) is not used. Infernix implements the encoding natively, checked byte for byte
  against `encoding/tests` and generated cases:
  - BOS, `<｜System｜>`, `<｜User｜>`, `<｜Assistant｜>`, and the `latest_reminder` role;
  - thinking mode ends the generation prompt with `<｜Assistant｜><think>`, chat mode with
    `<｜Assistant｜></think>`; earlier turns drop their thinking unless tools are declared, except
    the assistant turns after the last user turn, which keep it (`encoding.py`; the recipe drops
    those too); Infernix's `preserve_thinking` maps to `drop_thinking = False`;
  - the reasoning effort (1-100; low / high / max = 50 / 75 / 100; default high) as the encoding's
    line at index 0 in thinking mode;
  - tools under `## Tools` in the system prompt, `response_format` under `## Response Format`;
  - tool calls in DSML (`<｜DSML｜ calls>`, `invoke name=`, `parameter name= string=`), results as
    `<tool_result>` blocks inside user turns in call order;
  - JSON written as Python's `json.dumps(ensure_ascii=False)` writes it: `, ` and `: ` separators,
    non-ASCII unescaped, member order kept, Python's float repr (`1e-05`, `1e+16`), integers beyond
    64 bits exact; deepseek-recipe's `json.rs` agrees except on positive exponents and big integers,
    where `encoding.py` decides.
- **Lowering rules for this family** (the recipe decides lowering; Infernix's generic rules differ):
  - content parts, merged consecutive user or tool-result turns, Anthropic system arrays, assistant
    text blocks and thinking blocks are joined with a blank line (`\n\n`; Responses reasoning entries
    with `\n`), where Infernix's generic rule joins without a separator; consecutive assistant
    messages merge into one turn with `\n\n`;
  - Chat Completions accepts the `developer` role (rendered `<｜System｜>`, mid-conversation if not
    first, as Responses `instructions` are) and DeepSeek's `latest_reminder` role; an empty system
    message is dropped;
  - `response_format` (Chat `json_schema` / `json_object`, Responses `text.format`, Anthropic
    `output_config.format`) is **rendered** as `## Response Format` after the tool prompt in the first
    system turn, as the model was trained, beside the grammar constraint; this family is an
    exception to serving.md's "the schema is not inserted";
  - a tool definition renders as `{"name", "description", "parameters"}` in that order (empty
    description and `{}` parameters when absent), the recipe's form; namespace tools render as
    `{namespace}::{name}` with the namespace description prepended, and `::` inside a name is refused;
    the Responses `apply_patch` custom tool renders with DeepSeek's fixed description and an
    `{input: string}` schema;
  - mid-conversation tool additions: `encoding.py` renders the full `## Tools` template on any system
    message that carries tools, the recipe an `### Updated Tool Schemas` block; M9 compares both before
    deferred tools are offered.
- **Second oracle: deepseek-recipe** (DeepSeek's API front end, MIT, Rust; reviewed 2026-10-21 and
  2026-10-24). It converts Chat Completions, Responses and Messages requests into one conversation,
  renders V4.1 prompts with explicit control-token ranges and parses the output stream. Its
  exact-string unit tests are ported (with the notice) as fixtures beside the five official cases.
  Where it and `encoding.py` disagree, `encoding.py` decides rendering and the recipe decides API
  behaviour: a `latest_reminder` after the last user message (`encoding.py` emits the assistant
  opener first), consecutive reminders (the recipe keeps the first), numeric effort and task tokens
  (`<｜action｜>` …, only in `encoding.py`).
- **Control tokens are recognized by token id, never by spelling.** The template's own control
  tokens are encoded as tokens; user text, tool results and tool arguments are encoded as ordinary
  text, so a control spelling inside them stays text (as DeepSeek's API does). `</think>` and
  `｜DSML｜` are added tokens with special = false, so the output parser tests added-vocabulary
  membership, not the special flag. One exception: the grounding tokens 129271-129278 (`<｜ref｜>`,
  `<｜box｜>`, `<｜point｜>`, `<｜polygon｜>` and their closers) found in assistant history are model
  output and are encoded as tokens.
- **Protocol mapping** (DeepSeek's API, deepseek-recipe `protocol/reasoning.rs`):
  - OpenAI `reasoning_effort` minimal / low → 50, medium / high / xhigh → 75, ultra / max → 100,
    `none` → chat mode; thinking is on by default at 75; Infernix keeps a numeric 1-100 extension
    (`encoding.py` accepts integers). An earlier version of this section mapped medium to 62, the
    September community template's alias.
  - Anthropic: effort from `output_config.effort`; `thinking.type: adaptive` means enabled;
    `thinking.budget_tokens` is Infernix's own thinking budget (the early close), not an effort table.
  - Chat Completions also takes DeepSeek's top-level `thinking` object (`enabled` / `adaptive` /
    `disabled`, with `budget_tokens`), which DeepSeek's clients send; today Infernix ignores unknown
    fields, so `disabled` would leave thinking on. As in the API, an explicit `thinking` overrides the
    effort (enabled with effort `none` thinks at 75); a conflict with `enable_thinking` or template
    kwargs is still refused.
  - Thinking budget: the early close is the single `</think>` token (id 128822), after the
    `--thinking-budget-message` text when one is set; the recipe has no budget mechanism of its own.
  - Reasoning pass-back: with tools declared, V4.1 keeps all earlier reasoning in the prompt. A client
    that omits `reasoning_content` changes the rendered prompt at that turn: reuse stops there and the
    model leaves its trained format. DeepSeek's API rejects the omission; tool-call ids carry an `ET_`
    marker when a turn produced no reasoning, so an honest empty reasoning is not mistaken for an
    omitted one. Infernix restores reasoning it generated itself from the ids it issued (as it already
    does from Anthropic signatures), and otherwise renders what the client sent.
  - `max_tokens` defaults: 8,192 without thinking, 65,536 with thinking, all remaining context at
    max effort (the API's 131,072, capped by the context), with Infernix's answer-room rule.
  - Thinking is native and can be switched off (unlike GLM).
- **Tool calls.** A `dsml` parser with tolerant recovery and a constrained-decoding grammar (M9 checks
  whether xgrammar's `converter_ext/deepseek.cc` covers V4.1's leading-space tags), with DeepSeek's
  API's recognition rules (deepseek-recipe `stream/output_parser.rs`): markers by token identity; a
  call block that opens inside unclosed reasoning is parsed as calls; both closers
  `</｜DSML｜tool_calls>` (V4) and `</｜DSML｜ calls>` (V4.1) are accepted; output after the block and
  a stray `</think>` in content are dropped; newlines are stripped at the start of a new turn's
  reasoning and of content (not of a prefix continuation) and consumed directly before `</think>` and
  before the call opener, and a run of newlines inside text is held until a non-newline byte arrives;
  `string="true"` values are JSON-escaped and `string="false"` values pass through raw. The recipe's
  laxness is not copied: Infernix keeps its validation (an undeclared tool name, a malformed attribute
  or an invalid raw value makes the region ordinary content unless tolerant mode recovers it; a
  repeated parameter keeps the last value at the first position), and the Qwen heuristics (the
  Markdown-fence rule, "`</think>` closes only before a line break") are dropped for this family.
  Arguments are **streamed as they are generated** (the tool name as soon as it is parsed, then
  argument fragments) only on the constrained route, where truncation by the output limit is the
  only possible failure and is reported as such; the unconstrained route keeps publishing completed
  calls, as serving.md's contract says. Forced tool choice (`required` or
  a named tool) is DeepSeek's prefill of `\n\n<｜DSML｜ calls>\n` after the assistant header in chat
  mode; the API rejects it in thinking mode, while Infernix's grammar can force calls after
  `</think>`, and M9 decides between the two with a check of call quality. DeepInfer's structured
  output shows the thinking-mode form: the grammar starts after a trigger token sequence, and an
  optional bound token forces the trigger token by token once reached; drafts are not constrained,
  masks apply per verify step with rollback. Here: bound `</think>`, trigger the DSML call opener (or
  the start of the JSON). This is advertised OpenAI and Anthropic behaviour, so the schema tests and
  the serving documentation change together (AGENTS.md).
- **Tokens never generated**: DeepSeek's evaluation masks BOS 0, PAD 2 and ids 129262-129264 (129264
  is the image placeholder) and can suppress EOS as the first token; the family masks the same ids
  (a product choice recorded in M9).
- **Output stream.**
  - Detokenization flushes pending bytes before every added token and never skips added tokens,
    including the special = true grounding tokens, so a reply re-encodes to the same ids; only EOS
    ends the turn and is withheld.
  - Log probabilities, if offered: each token once, in the channel its bytes land in (Chat's
    `reasoning_content` field for reasoning, DeepSeek's extension), marker tokens nowhere, values from
    the target distribution (the verify column under speculation); token text as UTF-8 or `\xNN` per
    byte, with `bytes`.
  - Usage: Infernix's exact counts, with the API's two differences documented (the API counts
    `reasoning_tokens` by re-encoding the emitted reasoning, without `</think>`); Chat usage adds
    `prompt_cache_hit_tokens` and `prompt_cache_miss_tokens`, which DeepSeek clients read.
  - Tool-call ids follow the API's `call_NN_` form; the `ET_` empty-reasoning check also accepts the
    underscore-stripped form some clients send back.
- **Prefix identity.** The effort line sits near the start in thinking mode and the tool prompt and
  response format in the first system turn, so switching thinking or effort misses the whole cached
  prompt; Infernix keeps the declarations in the prompt under `tool_choice: none` and named choices
  (as `encoding.py` renders them), which keeps the prefix stable. Mode and effort are logged per
  request so misses can be attributed. A model reply is reused next turn only where its re-render
  equals the generated ids: thinking chats without tools drop the reply's reasoning, so the next prompt
  departs right after the previous `<｜Assistant｜>` (§18.3); M9 measures how often re-renders match on
  real agent transcripts.
- **Deferred tools and prefix continuation** (M9 or later, product choices): the API appends
  discovered tool schemas as a mid-conversation `<｜System｜>` block ("### Updated Tool Schemas"), so
  the prefix never changes as the tool set grows, which suits this family's cheap prefix cache;
  whether V4.1 was trained on that block is unknown, so M9 checks it before it is offered.
  `prefix: true` on a final assistant message continues it without a closing token or EOS; the API
  refuses it with tool-call history, `json_object` or tools.
- **Owner decisions for M9** (DeepSeek's API differs from Infernix's generic serving contract):
  stop strings only in answer content (the API never matches them in reasoning or tool markup, and
  thinking is long and on by default); the `[tool_error]` prefix on failed tool results (both
  DeepSeek references render none); a final assistant message without `prefix` on Chat (the API opens
  a new turn, Infernix's contract prefills); tool-result images before the vision track (a 400 ends
  an agent session, placeholder text keeps it going).
- **Images.** Refused for this family until the vision track: the encoding's image tokens are
  recognized and rejected with a clear error. For that track (O7): 14-pixel patches, downsample 3, at
  most 1,024 tokens per image, h · (w + 1) + 2 tokens, images below 544² upscaled; DeepSeek's API
  re-encodes images as WebP at quality 90; id 129264 is `<｜deepseek_image｜>` in the 1017 tokenizer
  (`<｜image｜>` in the recipe's copy), so prompts are compared by token id.

---

## 22. Configuration surface

| Option | Default | Meaning |
|---|---|---|
| `--prompt-decoder full\|replay` | `full` | Decision g (§16); `full` equals the reference forward; `replay` is DeepSeek's served rule and **changes output** |
| `--replay-keep N` | 140 | Positions the `replay` rule runs through the decoder (DeepSeek's served value; at least 134); part of the prefix-cache fingerprint |
| `--expert-volume PATH` (repeat) | required | The artifact's expert volumes, any order; startup checks ids, sizes and headers |
| `--engram-volume PATH`, `--engram-residency auto\|ram\|volume` | the artifact's own; auto | Decision d (§14.5); a volume of the other format **changes output** and is labelled |
| `--expert-ram-mib N\|auto`, `--expert-pin-mib N`, `--ram-headroom-mib N` | auto, auto, 2048 | RAM tier (FN §19.3.7) |
| `--vram-headroom-mib N\|auto`, `--expert-state FILE\|off` | auto, beside the artifact | As FN |
| `--ssd-lookahead off\|1\|2\|3` | off | §11.3 |
| `--cpu-experts auto\|off\|N`, `--cpu-expert-cores p\|pe` | auto, auto | §12.3 |
| `--spec off\|dspark\|auto`, `--dspark-expert-pages N\|auto`, `--draft-tokens auto\|N` | off until M7 | §17 |
| `--ngram-draft-tokens N` | off | Copy proposals |
| `--kv-dtype model` | `model` | The only accepted value (§13.5) |
| `--max-context N` | 131072 | Up to 1,048,576 |
| `--prefill-chunk N`, `--prefill-group N` | auto, 48 | Chunking; experts per prompt group |
| `--host-context-mib N` | 2048 | Prefix Host tier |
| `--reasoning-effort 1..100\|minimal\|low\|medium\|high\|xhigh\|ultra\|max\|off` | high (75) | Request default; requests may override; names map as DeepSeek's API does (§21) |
| Sampling preset for the family | temperature 1.0, top_p 1.0, min_p 0, no penalties; top_k 20 (see the note) | DeepSeek's API settings (deepseek-recipe `request/options.rs`). The API also forces temperature 1.0 and top_p ≥ 0.95 in thinking mode, replaces top_p with 1.0 outside it, and drops penalties; Infernix honours request values and documents the difference. **Note**: the API samples the full vocabulary, but Infernix's sampling pipeline has an exact top-20 candidate domain (`src/runtime/contract/sampling.cpp`), so "no top_k" cannot be represented. M4 measures the probability mass outside the top 20 at temperature 1.0 on the NRS corpus (mean, p99, share of positions above 1 %) with the reference forward; if it matters, this family gets a full-vocabulary route (129,280 candidates) for the sampler and DSpark's sampled verification (§25.2 question 15). A top-20 cut raises DSpark acceptance, so M7's figures say which domain they used |
| `--approx-experts off\|ssd\|vram`, `--approx-keep-map`, `--approx-min-weight θ`, `--approx-experts-rule residency\|static` | off | §9.8; **changes output** |

Converter: `--recipe deepseek_v4_1_flash_exl3` with `--experts REPO@REV` (the EXL3 pack) and
`--source REPO@REV` (the official release), or `--recipe deepseek_v4_1_flash` (M11, or an official
entry, §7.1) with `--source REPO@REV`; either source streamed shard by shard (§7.6) or given as
`DIR`; `--expert-volumes P0,P1,…` and
`--expert-volume-weights w0,w1,…` (default chosen from free space and bandwidth); `--engram-volume PATH`,
`--engram-format fp8|mxfp4` (default: the source's), `--engram-gguf FILE` (a published IQ4_NL table,
the EXL3 recipe's default, or with O4 a Q5_K or Q6_K one; rows copied verbatim), `--engram-reuse PATH`;
`--expert-record mxfp4|nvfp4|auto` (M11); `--staging DIR`; `--plan` (print the disk plan and stop).
Tools: `tools.artifact.restripe`, `tools.deepseek_v41.verify_artifact`,
`tools.deepseek_v41.reference_forward` (`--weights artifact|official`),
`tools.deepseek_v41.block_check`, `tools.deepseek_v41.decode_model`; `infernix-calibrate` gains the
striped-read, Engram-read and CPU MX8 and E3A16 probes.

---

## 23. Implementation plan

| # | Milestone | Deliverables | Exit criteria |
|---|---|---|---|
| **M0** | Facts | ~~September-vs-1017 comparison of tensors and Engram rows~~ (done 2026-10-24, §2.6); the EXL3 pack: revision, base revision, tensor names, `exl3_moe_plan.json` against the tensors, tile order, bit order and Hadamard scale against ExLlamaV3 at the producing version (GLM §9.5's list), sampled experts decoded by ExLlamaV3 and by an independent FP64 decoder; the IQ4_NL Engram table (§7.1's checks: ISTA's against Unsloth's row data, sampled rows against the official 1017 FP8 rows, the split between the two tables); Transformers 5.6 `deepseek_v41` against `model.py`; encoding fixtures and tokenizer files; DS4's suffix rule against `model.py`; C++ probes (`tools/deepseek_v41_probe/`): striped 1.06 MiB reads over three drives through IOCP, Engram 4 KiB reads under concurrent unit reads (≥ 0.8 M per second, §14.7), CPU E3A16 per width on P- and E-cores, CPU MX8 for the dense rows, the pin room with this plan's buffers | §2, §3, §5.7 inputs and §7.1's rules confirmed or revised; §5.7 re-run |
| **M1** | Traces and policy | Route traces from M4's engine on NRS and a broad corpus (code, chat, CJK, tool use, long context); replay of the §10.2 and §10.4 candidates by value per page with the pack's widths; the union growth σ and lookahead recall | Policies chosen by the fixed rules; §5.2 replaced by measured hit rates |
| **M2** | Shared components | `src/models/offload/` **with the page pool** (§8.2): pages, page lists, value-per-page eviction, paged staging, landing and bounce storage; expert volumes with width banks, the striped reader and the row-volume reader; built here, GLM reuses them (owner, 2026-10-24) | Qwen4Exp's tests unchanged (one page per record); greedy ids identical; tg512 within noise; the page audit of §20.5 |
| **M3** | Ops | MX8 codec; `fp8_e4m3fn_b32_e8m0` Linear; the EXL3 codec at K = 2-6 (built here, reused by GLM) and `offloaded_sparse_moe`'s `exl3_mul1` routes (narrow, wide, CPU) with mixed widths and page lists, also qualified at GLM's expert geometry; router; `csa_compressor`, `csa_indexer`, `csa_attention`; `mhc`; `engram`; DSpark Ops | Every §20.1 qualification passes; per-width kernel times recorded |
| **M4** | Artifact and functional model | The EXL3 recipe, the streaming converter with its disk planner (run when the owner has freed the space), volumes, restripe, `verify_artifact`; Engine, CLI, bench and CausalScoring with one volume; both prompt rules; both reference forwards (artifact, official) | Every tensor round-trips; §20.2 checks; the execution gates for both rules; the §20.3 reports, the EXL3 artifact against the original model first; first end-to-end numbers |
| **M5** | Tiers at scale | Three volumes, landing pages, partial records, M1's policies, promotion sweep, CPU sweep per width, unpinned pages, warm start, lookahead A/B | §20.5 invariance and fault tests; decode Must target; three volumes ≥ 1.6× one |
| **M6** | Decode speed | §9.3's kernels and fusions, stall-time warming; nsys attribution of one token | Per-kernel times against the budget; anything > 30 % over gets ncu before M7 |
| **M7** | Speculation | DSpark drafter (bounded experts, Markov and confidence heads, the main-feature window written inside the verify forward), the cost model with σ and confidence calibrated on expected acceptance, n-gram proposals | Default chosen by measurement at DeepSeek's sampling (temperature 1.0, thinking on), greedy reported beside; never slower than plain beyond noise |
| **M7a** | Approximate mode | §9.8 | Mode off bit-identical to a build without it; presets meet the gates |
| **M8** | Prompts and long context | 128K walk spans, prompt kernels, the CPU decoder replay, the full rule's suffix, indexer prompt scoring, 1M context | Prompt Must targets; decode at 128K ≥ 90 % of 4K |
| **M9** | Frontend and serving | Tokenizer, encoding with control tokens by id, DSML parser with DeepSeek's API rules and streamed arguments, protocol mappings (§21), the family's sampling preset, protocol schema tests, prefix-cache binding for both rules with decode-computed endpoints | Tokenizer and encoding oracles exact against `encoding/tests` and deepseek-recipe's fixtures; serving schema tests; prefix-cache real tests |
| **M10** | Report | Strata, Maya, DS4 under WSL2 and the §4.2 engines that run here, head to head; the §1.4 table filled | Every result reported, favourable or not |
| **M11** (deferred) | The MXFP4 artifact | `mxfp4_expert_v1` and `nvfp4_expert_rg16_v1` records (one page each), MX8 MXFP4 and NVFP4 routes on the GPU and the CPU, MX recovery, the official and NVIDIA recipes, the MXFP4-against-NVFP4 timings (decision c); the official scale rule, NVFP4 → MXFP4 recovery and ModelOpt's MXFP4 Engram rule on sampled blocks | The MXFP4 rows of §20.1 and §20.3; its §1.4 column |

Critical path: M0 → M2 ∥ M3 → M4 → M1 → M5 → M6, then M7 and M8; M9 can start now (the tokenizer
and encoding files are read). Rough size: ~35-39k new or moved lines (CSA2 Ops ~5k; MX8 codec and the
FP8-b32 Linear ~3k; the EXL3 codec and expert routes with the CPU kernel at K = 2-6 ~5-6k; the page
pool ~2.5-3.5k; Engram ~1.5k; DSpark ~2.5k; mHC ~1k and volumes with the reader ~2.5k, shared with
GLM; offload extraction ~3.5k moved; frontend ~3k with deepseek-recipe's rules; converter and tools
including the streaming conversion and both reference forwards ~4.5k; tests ~9k), plus the
≈ 313 GB download and GPU time for the campaigns. About 9-11k of these lines (offload extraction,
page pool, volumes and readers, mHC, the EXL3 codec and expert routes) are the shared pieces GLM's
track then reuses. M11 adds ~4-5k later. At Flash-Next's pace (~34k lines in about six intensive
days, GLM §23) the page pool and per-expert widths are about a day of the total; the GPU
qualification and policy campaigns cost more than the code.

---

## 24. Documentation and authority changes

When the implementation lands: a user guide `docs/deepseek_v4_1-flash.md` (hardware advice, the cost
of a fresh prompt, the prompt rules, Engram formats); README models and benchmarks; `docs/cli.md` and
`docs/serving.md` (expert and Engram volumes, `--prompt-decoder`, DSML, reasoning effort); a
mathematics reference `docs/maintainer/deepseek_v41-model.md` from §2.2-§2.3;
[storage layouts](storage-layouts.md) (`exl3_expert_v1` and its pages, width banks, the Engram row
formats, `fp8_e4m3fn_b32_e8m0`; with M11, `mxfp4_expert_v1` and `nvfp4_expert_rg16_v1` at this
geometry); [tensor formats](tensor-formats.md) (EXL3 with a width per record, MX8, FP8 with 32 × 32
power-of-two scales; MXFP4 with M11); [op development](op-development.md) (MX8 and E3A16
qualification); [artifact container](artifact-container.md) (expert and Engram volumes, the expert →
bank table); [engine architecture](engine-architecture.md) (the new architecture, prompt rules as a
model mode, shared offload components and the page pool, and this family's exception to §3.1's
prefill-only publication: prompt/decode equivalence lets it publish decode-computed endpoints,
§18.3); [paged KV](paged-kv-cache.md) (compressed planes and windows with verification room);
[hybrid prefix cache](hybrid-prefix-cache-spec.md) (the binding and its snapshots);
[model quality](model-quality.md) (§6 for this family's configurations); the conversion guide (the
streaming conversion). This file is then removed.

---

## 25. Risks, open questions, options and rejected alternatives

### 25.1 Risks

| Risk | Effect | Mitigation |
|---|---|---|
| §5's model is optimistic, as GLM's was by 15-45 % against strata-glm | EXL3 decode ≈ 14-17 tok/s central (MXFP4 ≈ 9-11) | M4's first numbers refit the overhead terms; M6 attributes one token with nsys |
| Hit rates below central (JigSawPT's 59-66 % VRAM hits came with a 125.7 GiB host; Maya's agent traffic sat below hipfire's curve) | decode toward the low end of §5.7's range (EXL3 16 tok/s) | M1 first; the policy candidates; the RAM lever of §5.6 in the guide |
| Too little disk when the conversion runs (383 GB free on 2026-10-24 against ≈ 315 GB plus staging, ≈ 412 GB with FP8 Engram rows) | slower weighting (+6 % record time), a 4-bit Engram volume, or no conversion | the owner frees space first (§1.2); the planner (§7.6) |
| The EXL3 artifact's quality here differs from the pack's published figure (MX8 inputs and E3A16 instead of the pack's FP16 runtime; official dense weights instead of its EXL3 K5 dense; the IQ4_NL table instead of the FP8 rows it was measured with; a single-source panel) | the first artifact is worse, or better, than chosen for | §20.3's report against the original model in M4, with the table's share separated, before anything else is tuned; sudoingx's pack and diffbot's K3 use the same recipe; a Q5_K, Q6_K or FP8 table per run (O4) |
| The pack's format details (tile order, bit order, Hadamard scale, codebook at the producing ExLlamaV3 version; a vLLM plugin's tensor naming) | wrong expert values | M0 checks them against ExLlamaV3 and an independent FP64 decoder, and the plan file against every tensor |
| Value-per-page eviction ranks wide experts badly (thrash between widths) | more SSD reads than §5.7 assumes | M1's trace replay with the widths; a single-width artifact behaves as today |
| Other sessions holding RAM on this machine | a smaller tier, more SSD reads | the startup ledger; the guide states it |
| Drive thermal throttling under sustained reads; `E:` busy with builds | stalls | per-volume telemetry and warnings (§11.5) |
| Engram reads queued behind expert units | layer-1 stalls | read priority (§11.2); measured in M0 |
| (M11) MXFP4 kernels slower than NVFP4 | decision c falls back | M11 timings; NVFP4 records built anyway |
| (M11) Some NVIDIA NVFP4 blocks not MX-recoverable | that source stores NVFP4 records | per-bank fallback (§7.1) |
| Router near-ties (JigSawPT: the 6th-to-7th margin is about 1 %; 208 of 240 choices agreed between their runs) | differences from the reference at near-ties | exact FP32 logits, lower-id ties; agreement reported |
| The full rule's suffix costs more than modelled (the decoder's thick experts at layers 20-25 on the wide route) | prompt targets of §1.4 missed | M8 measures it; `--prompt-decoder replay` stays available |
| EXL3's CPU decode on AVX2 slower than estimated (unmeasured on this CPU) | the EXL3 artifact ≈ 5-10 % slower (§5.7) | M0's CPU probe; the link takes more of the split |
| The Engram read rate in prompts (Maya: ≈ 0.1 M rows per second) | +15 s on a 32K prompt | M0's probe (§14.7) |
| The ≈ 313 GB download (510 GB for M11) is interrupted | time | the resumable shard journal |
| DSpark slower than plain | no speculative gain | off by default; K = 0 in the cost model |
| Upstream ambiguities (encoding details, Transformers against `model.py`) | wrong frontend or mathematics | M0 oracles against `encoding/tests` and `model.py` |

### 25.2 Open questions for M0

1. ~~Did the 1017 update change the Engram tables and the DSpark weights?~~ Yes, every sampled tensor
   (§2.6, 2026-10-24).
2. (M11) Do DeepSeek's MXFP4 scales follow ⌊log₂ amax⌋ − 2 on every non-zero block, so that
   recovery is byte-identical?
3. (M11) Does every block of NVIDIA's NVFP4 export recover to MXFP4?
4. Does Transformers 5.6 implement `deepseek_v41` as `model.py` does (the mHC shift, the indexer)?
5. The Windows QD-64 4 KiB read rate on the Engram drive with concurrent 1.06 MiB unit reads.
6. The CPU E3A16 cost per expert of each width (and MX8 for the dense rows) on P- and E-cores.
7. The pin room and commit limit with this plan's buffers.
7a. The EXL3 pack's format facts (§7.1, §25.1) and its base revision of 1017.

Questions for the owner:

8. ~~Prompt rule, activations, records, Engram formats, the disk route~~ — decided 2026-10-19 (§1.2).
9. ~~Disk~~ — the owner frees space when the artifact work starts (2026-10-24, §1.2); the planner
   sizes the plan then (§7.6), including whether `C:` holds expert units (+13 % decode for the EXL3
   artifact as a fourth volume, model).
10. When the artifact work starts: the owner says when (the ≈ 313 GB download and the conversion
    need that go-ahead; the converter's RAM and disk use are bounded, and it checks other sessions'
    RAM first).
11. ~~Decision f~~ — EXL3 with a bit width per expert from the start, decided 2026-10-24 (§1.2, §5.7).
12. ~~Decision g: the prompt-rule default~~ — `full`, decided 2026-10-24 (§1.2).
13. ~~The lossless MXFP4 build~~ — not needed initially (2026-10-24): M11; the quality report's
    reference logits come from streaming the official shards (§20.2).
14. ~~Order with the GLM track~~ — this track builds the shared EXL3 codec, offload extraction and
    page pool first; GLM reuses them (2026-10-24, §8.2). When the work starts is question 10.
15. The sampling domain (§22): keep Infernix's top-20 candidate domain for this family, or build a
    full-vocabulary route as DeepSeek's API samples; M4 measures the mass outside the top 20 first.
16. The serving differences from DeepSeek's API listed in §21's last bullet (stop-string scope, the
    `[tool_error]` prefix, a final assistant message on Chat, tool-result images); decided at M9.

### 25.3 Options (each behind its own measured gate)

| ID | Option | Gate |
|---|---|---|
| O1 | A fourth expert volume on `C:` (+13 % decode for the EXL3 artifact, model) | the owner frees space; M5 A/B |
| O2 | A W4A4 fast path for prompt experts (NVIDIA's activation quantization; input scales from an NVFP4 source or calibrated), ≈ 4× expert compute where a prompt is compute-bound | model-quality §6 prompt gates; never in decode |
| O3 | Draft-informed SSD prefetch: the main routers of early layers on draft tokens start reads before verification | M1 recall ≥ 0.5 for SSD-tier experts; M7 gain |
| O4 | The Engram table if IQ4_NL's share (§20.3) matters: Unsloth's Q5_K (136.8 GB volume) or Q6_K (165.6 GB), or the FP8 rows (209.7 GB), swapped per run; MXFP4 (104.9 GB) if disk is shorter still | the owner's choice after the §20.3 report |
| O5 | Mirrored expert volumes with per-unit routing (GLM O8) | M5 A/B; twice the disk |
| O6 | A GGUF compact build (§5.7): ISTA's GSQ-RCO IQ2_XS as a speed build (≈ 33 tok/s, perplexity +7.3 %) or another published set; experts as stored in the page pool, everything else official; `ggml_iq2_xxs` and `ggml_iq2_xs` | an owner request; §20.3's report |
| O10 | ~~EXL3 with per-expert bits~~ | adopted as the first artifact (decision f) |
| O11 | Hybrid: official MXFP4 records on a profile's hottest 10-20 % of experts, a 2-bit build on the rest (§5.7); the page pool already holds mixed formats, the MXFP4 routes come with M11 | unmeasured anywhere; §20.3's report on several workloads |
| O7 | Vision (DeepSeek-ViT, 0.97 GB) | a later track |
| O8 | Lookahead on three volumes | M5 gain |
| O9 | Linux `io_uring` path for the striped and row readers | Linux users; the same tests |

### 25.4 Rejected alternatives

| Alternative | Reason |
|---|---|
| FP8 or FP4 tensor cores with FP32 accumulation as the arithmetic | Not reproducible on the CPU or across routes (FN §16.2.2) |
| BF16 activations (A16) as the exact route | Decision b: DeepSeek's trained MXFP8 forward is the model; Infernix's A16 precedent (FN §16.2.2) was for checkpoints trained in BF16 |
| NVIDIA's W4A4 as the exact route | Served that way it measured KL 0.035 against a 0.016 floor (with MXFP4 Engram, §4.2); the 1017 release has no calibrated input scales |
| Requantizing the experts at conversion | Published builds with measured quality exist (§5.7); Infernix imports one as stored instead of producing an unmeasured one |
| DeepSeek's encoder SWA bounded replay for cache restores | Snapshots make restores exact for ≈ 1.4 MB of encoder windows |
| Page-cache or memory-mapped expert tiers | Pinned and locked records measured faster (FN §19.3.7); DS4's page-cache gain (#477) is the gain over O_DIRECT re-reads, which Infernix's RAM tier already has |
| Keeping the source shards during conversion | Does not fit the disks (§7.6) |
| Engram tables in RAM | 97-195 GiB do not fit beside the expert tier |
| A static hot-expert list as the primary cache key | Loses to recency on this family (DS4 #636) |
| Five-token DSpark drafts by default | The union cost (§5.4); JigSawPT measured losses at two |
| A host KV tier for long context | The compressed KV is 0.89 GB at 1M |
| `--kv-dtype` variants | The model defines its KV formats |

---

## 26. Sources

- Model: [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
  and [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
  (`config.json`, `inference/model.py`, `kernel.py`, `engram.py`, `convert.py`, `encoding/`,
  `tokenizer.json`, safetensors headers), read 2026-10-18; technical report arXiv 2609.19969
  (17 September 2026); DSpark arXiv 2607.05147; DeepSeek-V4 arXiv 2606.19348; Engram arXiv
  2601.07372; mHC arXiv 2512.24880; DeepSeek API news and change log.
- NVFP4: [nvidia/DeepSeek-V4.1-Flash-NVFP4](https://huggingface.co/nvidia/DeepSeek-V4.1-Flash-NVFP4)
  (card, `config.json`, `hf_quant_config.json`, headers), read 2026-10-18;
  [LibertAIDAI/DeepSeek-V4.1-Flash-1017-NVFP4](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-NVFP4)
  (card), read 2026-10-19.
- DwarfStar: [antirez/ds4](https://github.com/antirez/ds4) at `fc80bd69` (`README.md`,
  `docs/MODELS.md`, `SSD_STREAMING.md`, `PERFORMANCE.md`, `SPECULATIVE_DECODING.md`,
  `CUDA_MULTI_GPU.md`, `DGX_SPARK.md`, `QA_BEFORE_RELEASES.md` §7, §8, §17, `ds4.c`, `ds4_engram.c`,
  `ds4_cuda.cu`, `gguf-tools/README.md`, `gguf-tools/deepseek41_quantize.py`; issues #477, #636,
  #639; PR #647), read 2026-10-19.
- DeepSeek's serving code: [deepseek-ai/DeepInfer](https://github.com/deepseek-ai/DeepInfer) (Apache-2.0;
  read at `774ce244` on 2026-10-21 and `9bd9cf24` on 2026-10-24: the V4.1 model, cache engine, YOCO
  replay, DSpark, sampling, structured output, `tests/`, `examples/yoco/`,
  `doc/inference_deployment_qa.md`); the libraries it pins,
  [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) `1fc9c669`,
  [FlashMLA](https://github.com/deepseek-ai/FlashMLA) `2e5429fc`,
  [DeepSelect](https://github.com/deepseek-ai/DeepSelect) `bfa4507d` and
  [TileKernels](https://github.com/deepseek-ai/TileKernels) 2.1.0 (`4d5ad0f`), and the 1017
  repository's `inference/kernel.py` and `engram.py`, read 2026-10-24; and
  [deepseek-ai/deepseek-recipe](https://github.com/deepseek-ai/deepseek-recipe) (MIT, at `935817c6`:
  protocol mapping, V4.1 rendering, the streaming parser); cachedMoE (`acupof-ai/cachedMoE`, MIT:
  Strix Halo, NVMe expert streaming, exact routing 9.20 tok/s with two drives and 7.61 with one,
  ≈ 29 s for a 4K prompt).
- Other engines and artifacts, read 2026-10-19:
  [JigSawPT/deepseek-v41-flash-on-5090](https://github.com/JigSawPT/deepseek-v41-flash-on-5090) and
  [llama.cpp discussion #28766](https://github.com/ggml-org/llama.cpp/discussions/28766);
  [llama.cpp PR #28696](https://github.com/ggml-org/llama.cpp/pull/28696);
  [ik_llama.cpp issue #2438](https://github.com/ikawrakow/ik_llama.cpp/issues/2438);
  [Indras-Mirror/Strata-DSV4.1](https://github.com/Indras-Mirror/Strata-DSV4.1); FreeToken
  ([arXiv 2608.16157](https://arxiv.org/html/2608.16157v1));
  [hipfire-models/deepseek-v4-flash-0731](https://huggingface.co/hipfire-models/deepseek-v4-flash-0731);
  KTransformers' DeepSeek-V4-Flash guide; vLLM [PR #56752](https://github.com/vllm-project/vllm/pull/56752),
  [PR #61530](https://github.com/vllm-project/vllm/pull/61530), PR #61554, issues
  [#56461](https://github.com/vllm-project/vllm/issues/56461),
  [#56797](https://github.com/vllm-project/vllm/issues/56797),
  [#56837](https://github.com/vllm-project/vllm/issues/56837), the
  [V4.1 recipe](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash) and
  [blog post](https://vllm.ai/blog/2026-10-07-deepseek-v41-flash);
  [SGLang PR #42901](https://github.com/sgl-project/sglang/pull/42901);
  [diffbot's EXL3 3 bpw pack](https://huggingface.co/diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000);
  [OpensourceWTF's streaming repack](https://huggingface.co/OpensourceWTF/DeepSeek-V4.1-Flash-streaming-repack-exl3-3.0bpw);
  [hughmadden/ds41rt-rtx5090](https://github.com/hughmadden/ds41rt-rtx5090);
  [unsloth/DeepSeek-V4.1-Flash-1017-GGUF](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF).
- Strata and Project Maya, read 2026-10-24: [Niko1221/Strata](https://github.com/Niko1221/Strata) at
  `7dd5cf18` (`src/dsv41`, `tools/dsv41`, `strata_control/dsv41_fit.py`, `docs/DEEPSEEK41.md`, release
  notes 0.2.4-0.2.4.2, the commit history since 2026-10-19);
  [Project Maya](https://github.com/mw00/project-maya) v1.1.26 at `f26d8ba` (`README.md`, `CHANGELOG.md`,
  `docs/SETTINGS.md`, `docs/SETUP.md`, `src/ds4`, `tools/ds4`; issues and pull requests #271-#448).
- 1017 quantizations and their quality, read 2026-10-24:
  [ISTA-DASLab GSQ-RCO GGUF](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF),
  [Unsloth's GGUF](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF) and its
  [guide](https://unsloth.ai/docs/models/deepseek-v4.1-flash),
  [Maya's GGUF](https://huggingface.co/peasantsmith/DeepSeek-V4.1-Flash-1017-Maya-GGUF),
  [kyuz0's GGUF](https://huggingface.co/kyuz0/deepseek-v4.1-flash-gguf) and
  [short100 panel](https://huggingface.co/datasets/kyuz0/deepseek-v4.1-flash-1017-qa),
  [diffbot's EXL3 pack](https://huggingface.co/diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000)
  (1017 since 2026-10-21),
  [HeNryous's per-expert EXL3 2.75](https://huggingface.co/HeNryous/DeepSeek-V4.1-Flash-1017-EXL3-2.75bpw-DGX-Spark)
  (`README.md`, `exl3_moe_plan.json`),
  [sudoingx's EXL3 2.8](https://huggingface.co/sudoingx/DeepSeek-V4.1-Flash-1017-EXL3-2.8bpw),
  [LibertAIDAI's EXL3 2.9](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw),
  [lxe's dsx-t16](https://huggingface.co/lxe/DeepSeek-V4.1-Flash-1017-dsx-t16),
  [Lucebox's ROCMFP GGUF](https://huggingface.co/Lucebox/DeepSeek-V4.1-Flash-1017-ROCMFP-GGUF),
  [brandonmusic's TrellisMX](https://huggingface.co/brandonmusic/DeepSeek-V4.1-Flash-1017-TrellisMX-K2),
  [prometheusAIR's GGUF](https://huggingface.co/prometheusAIR/DeepSeek-V4.1-Flash-1017-GGUF); GGUF
  headers read by HTTP range; the survey's tables are kept with this design's working notes.
- Infernix: the [Flash-Next design](qwen3_8-flash-next-design.md), the [GLM-5.3-Flash design](glm5_3-flash-design.md)
  (including its storage measurements of 2026-10-10), [model quality](model-quality.md) §6 and
  [engine architecture](engine-architecture.md).
