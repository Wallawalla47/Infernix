# Qwen3.8-Flash-Next on one RTX 5090: design

**Status: design proposal, partly implemented.** [§19.1](#191-implementation-status-and-handoff) lists what exists
and what needs the target machine. This is a temporary planning document for the
Qwen3.8-Flash-Next product change. When the implementation lands, its stable content moves into
the active authorities named in [§20](#20-documentation-and-authority-changes) and this file is
removed.

**Revision 2026-10-03.** This version was revised after an expert review for implementability and
peak performance on the target hardware, and after three requirement changes: NVIDIA's quantized
tensors stay bit-exact with NVIDIA's activation formats, QSA supports every Infernix KV profile, and
BF16 tensors may move to 8 bits only where that helps significantly.
[Appendix A](#appendix-a-review-findings-and-changes) lists every finding and what changed.

This design adds `Qwen4ExpForCausalLM` (Qwen3.8-Flash-Next, about 180B parameters) to Infernix.
The target system is:

- one RTX 5090 (32 GB);
- 96 GB of DDR5 host RAM;
- one NVMe SSD.

It adds three mechanisms that Infernix does not have today:

1. routed experts held in pinned host RAM, with misses served by the CPU in place and by
   copy-engine prefetch over PCIe;
2. a VRAM expert cache that shares device memory with the KV cache;
3. the n-gram PLE embedding table kept on NVMe and read row by row while the engine runs.

The goal is the fastest single-GPU implementation of this model on this hardware, **without
changing any tensor that NVIDIA quantized** (§1.1). The reference point is Strata running Unsloth
UD-Q4_K_XL, a 4-bit model comparable to NVFP4, at **72-80 tok/s decode** on the target machine.

---

## Contents

1. [Goals, baselines and acceptance](#1-goals-baselines-and-acceptance)
2. [Model facts that decide performance](#2-model-facts-that-decide-performance)
3. [What existing implementations do](#3-what-existing-implementations-do)
4. [Performance model](#4-performance-model)
5. [Design overview](#5-design-overview)
6. [Artifact and conversion](#6-artifact-and-conversion)
7. [Ownership in Infernix](#7-ownership-in-infernix)
8. [Decode pipeline](#8-decode-pipeline)
9. [Expert cache and the shared VRAM frame pool](#9-expert-cache-and-the-shared-vram-frame-pool)
10. [CPU expert engine](#10-cpu-expert-engine)
11. [Speculative decoding](#11-speculative-decoding)
12. [PLE n-gram table on NVMe](#12-ple-n-gram-table-on-nvme)
13. [Prefill](#13-prefill)
14. [Calibration and adaptive tuning](#14-calibration-and-adaptive-tuning)
15. [Memory plans](#15-memory-plans)
16. [Numerics and qualification](#16-numerics-and-qualification)
17. [High-reward options](#17-high-reward-options)
18. [Configuration surface](#18-configuration-surface)
19. [Implementation plan](#19-implementation-plan) ([status and handoff](#191-implementation-status-and-handoff), [next-phase designs](#193-next-phase-designs-2026-10-04))
20. [Documentation and authority changes](#20-documentation-and-authority-changes)
21. [Risks and open questions](#21-risks-and-open-questions)
22. [Sources](#22-sources)
- [Appendix A. Review findings and changes](#appendix-a-review-findings-and-changes)

---

## 1. Goals, baselines and acceptance

### 1.1 Deliverable

The product supports `Qwen4ExpForCausalLM` through the public Engine, CLI, `infernix-serve`, and
offline CausalScoring. It includes:

- the Text model;
- the MTP layer;
- QSA with **every Infernix KV profile** (`bf16`, `int8`, `fp8`, `nvfp4`, `k8v4`, `vq2`, `k4v2`;
  §8.10);
- Vision, which can be selected at startup.

**Fidelity to the NVIDIA checkpoint** (`nvidia/Qwen3.8-Flash-Next-NVFP4`) is a requirement:

- **Everything NVIDIA stores at 8 bits or less is imported bit-exactly and executed in NVIDIA's
  activation format:**
  - Routed experts are W4A4 NVFP4. The E2M1 codes, E4M3 block scales, FP32 `weight_scale_2` and
    each expert's own FP32 `input_scale` are preserved. Activations are quantized to NVFP4 at every
    routed-expert input with those calibrated scales (§16.2).
  - MTP routed experts are 128×128 block-scaled FP8, with the activation scheme the checkpoint
    declares (§6.1).
  - The PLE n-gram table is per-tensor FP8.
- **Tensors NVIDIA keeps in BF16 stay BF16, or move to 8 bits only where that helps
  significantly** and passes the quality gate (§16.3). Their activations stay BF16. Nothing moves
  below 8 bits.

"Bit-exact" applies to the stored tensors. NVIDIA's runtimes cannot be reproduced bit for bit on
a CPU: their activation quantizers use approximate hardware reciprocals, their tensor-core
accumulation order is unspecified, and they differ from each other at rounding boundaries. This
design therefore quantizes activations with the ModelOpt reference rule in exact IEEE arithmetic,
and computes every routed expert exactly (§16.2).

Two conversion recipes implement this (§6.1):

| Recipe | Dense weights | Status |
|---|---|---|
| **A, exact** | BF16, as in the checkpoint | Supported |
| **B, 8-bit dense** | The large BF16 matrices as 8-bit weights with BF16 activations (W8A16) | **Recommended** if it passes §16.3. Projected at 1.35-1.6× recipe A for plain decode and 1.1-1.45× for speculative decode, depending on the hit rate (§4.2). |

The supported machine is an RTX 5090 with:

- at least 96 GB of host RAM;
- PCIe 5.0 x16, or 4.0 x16 with reduced performance;
- an NVMe SSD for the n-gram table.

Concurrency stays within the existing product range of 1-8.

### 1.2 Baselines

All comparisons run on the same machine, with the same prompts and the same output caps:

| Baseline | Weights | What it shows |
|---|---|---|
| **Strata, user-measured** | UD-Q4_K_XL (Q4_K/Q5_K gate/up, Q5_1/Q8_0 down), with its own MTP | **72-80 tok/s decode on RTX 5090 + 96 GB**. This is the main baseline. |
| ninfer-ext `fc305acd` | NVFP4 experts (lossy scale handling, §3.2), Q8 dense | 98.5 tok/s serving at C=1, 185 at C=8. Measured on a 247 GiB host, so it is not the target memory class. |
| FreeToken offload/hybrid | NVFP4 experts, BF16 dense | 61-67 tok/s on a 5090 with 128 GB DDR5 (community report, verified) |
| SGLang, **all experts resident**, RTX PRO 6000 (sm_120, same 1.79 TB/s) | NVFP4 experts, BF16 dense | TPOT 11.4 ms (≈ 87 tok/s) plain; 6.0 ms (≈ 165 tok/s) with MTP accepting 3.3 of 4. This is a generic-framework ceiling with no offload at all. |
| Strata IQ2_XS | ~2.35 bpw | 165-179 tok/s on a 5090. This uses 2-bit experts, so it is **not** a like-for-like quality baseline. It shows the dense-path ceiling when 71% of the experts fit in VRAM. |

### 1.3 Targets

The targets assume Text with thinking off, a 4K prompt and 256 output tokens, at C=1 unless stated
otherwise, in the **primary configuration**: recipe B with `--kv-dtype int8 --max-context 262144`
(§18). That is the full-context, 8-bit-KV setup users run on this machine, and Strata is compared
in its equivalent (262K context, Q8 KV). The other KV profiles are measured and reported after it.
The targets are acceptance gates, not claims.

- The **Must** column beats the Strata baseline by 1.5-2×.
- The **Stretch** column is the §4.2 target-budget projection for recipe B at a 0.9 hit rate, which
  is below the replay traces' 0.969. Recipe A's projections are reported beside it, without
  separate targets.
- These targets were revised on 2026-10-03, before any measurement, because the fidelity
  requirement changed the recipe (Appendix A).

| Metric | Must | Stretch | Recipe A projection (h = 0.80 to replay) |
|---|---:|---:|---:|
| Decode, no speculation, C=1, 4K context | ≥ 120 tok/s (1.5× Strata Q4) | ≥ 200 | ~120-135 |
| Decode, best speculation, C=1, mixed chat corpus | ≥ 160 tok/s (2× Strata Q4) | ≥ 340 | ~195-285 |
| Decode at 128K and 256K actual context, relative to 4K | ≥ 85% | ≥ 92% | ~98% |
| Aggregate decode, C=8 | ≥ 300 tok/s | ≥ 600 | — |
| Prefill, 32K fresh prompt | ≥ 6,000 tok/s | ≥ 9,000 | — |
| Quality, recipe B vs recipe A (exact) | within the §16.3 gate | — | — |
| Quality, mean KL vs the BF16 reference model, fixed corpus | ≤ KL of UD-Q4_K_XL | — | — |

The acceptance report covers:

- the full distribution: median and range of at least three runs;
- the worst case;
- expert hit and miss composition;
- the device every number was measured on.

It compares against both Strata UD-Q4_K_XL and ninfer-ext on the target machine, and reports
recipes A and B separately. If a target is missed, the report says so; the target is not changed
after the measurement.

### 1.4 Non-goals

- Copying Strata or ninfer-ext code. Both were studied for mechanisms and measurements only.
- Any tensor below its NVIDIA precision, or below 8 bits if NVIDIA keeps it in BF16.
- Multi-GPU, AMD, or GPUs other than `sm_120a`.
- A generic offloading framework for other models. The mechanisms are written for this
  architecture, with Ops and core primitives kept model-independent where they are naturally so.

---

## 2. Model facts that decide performance

The mathematics is defined by the upstream Transformers `qwen4_exp` model and by SGLang's
`qwen4_exp_mtp`. The fork's [Qwen4Exp reference](https://github.com/giveen/ninfer-ext/blob/main/docs/maintainer/qwen4-exp-model.md)
summarizes it accurately; it is used here only as a reading aid.

| Quantity | Value |
|---|---|
| Parameters | ~176B: 125B main, 51B n-gram table, ~4B MTP. ~6B active per token. |
| Layers | 48: 36 GDN and 12 QSA. Layers 3, 7, …, 47 are attention; PLE sits in GDN layer 1. |
| Hidden width | H = 2560 |
| Hyper-connection residual | 4 streams (10,240 wide). Two HC mixers per layer, each with rank 320. |
| QSA | 24 Q heads, 2 KV heads, head dimension 256. Indexer has 4×128 query and 1×128 key. Budget 2048 tokens in 4-token blocks. |
| MoE | 512 routed experts, top-10, width 640, plus a gated shared expert of width 640 |
| PLE | One injection before block 1. 16 rows of 160 values per token, from a 320,001,536-row table. Dilated conv state of 9×10,240. |
| MTP | One full QSA block with its own 512-expert MoE |
| Vocabulary | 248,320. Untied embedding and head, no final norm. |

**NVIDIA checkpoint composition** (`quant_algo: MIXED_PRECISION`): only three tensor groups are
quantized.

- Routed experts: NVFP4, W4A4, with MSE-calibrated weight scales.
- MTP routed experts: FP8 with 128×128 block scales, copied from `Qwen/Qwen3.8-Flash-Next-FP8`.
- The n-gram table: per-tensor FP8, copied from the same checkpoint.

Everything else, including the shared experts, is BF16.

**Byte budget**, worked out from the shapes above:

| Item | Size |
|---|---|
| One routed expert (gate/up and down: 4,915,200 weights; E2M1 plus E4M3 per 16) | **2,764,800 B** (2.64 MiB) |
| One MTP routed expert (FP8 E4M3 plus 300 FP32 block scales) | 4,916,400 B |
| Routed expert weights touched per token (48 × 10) | **1.33 GB** |
| All routed experts: 48 × 512 main, plus 512 MTP | 67.9 GB + 2.5 GB |
| Dense Text weights (4.31B params, including `lm_head`) | **8.6 GB** in BF16 (recipe A), **4.4 GB** with 8-bit dense (recipe B, §6.1) |
| Token embedding | 1.27 GB in BF16. Only one row per token is read. |
| N-gram table | 51.2 GB of FP8 E4M3 codes plus one BF16 scalar scale |
| KV per context token (12 + 1 MTP attention layers, 2 KV heads, D=256), plus BF16 pooled index keys | 14.6 KB (`int8`, the primary profile), 27.5 KB (`bf16`), 14.2 KB (`fp8`), 8.3 KB (`nvfp4`), 4.3 KB (`vq2`); every profile in §8.10 |
| GDN recurrent state per sequence | 113 MB (FP32) |

What follows from these facts:

- **Experts cannot all be resident.** In the primary configuration (`int8` KV, 262K context,
  §18), VRAM holds 8,130 experts (33.1%) with recipe A and 9,652 (39.3%) with recipe B at a 4K
  context. The KV host tier keeps that above 7,911 and 9,433 at 256K (§15.1).
- **The whole expert set fits in 96 GB of RAM, but not much else does.** About 20 GB remains for
  everything else. The 51 GB n-gram table therefore has to stay on NVMe, as required.
- **Every miss costs host DRAM bandwidth**, whether it crosses PCIe (a DMA read of DRAM) or is
  computed by the CPU (a CPU read of DRAM). At these capacities a plain-decode token is
  **latency-bound** on its few misses. The DRAM ceiling binds for wide verify windows at low hit
  rates and for prefill (§4.1).
- **Dense weights dominate the GPU's per-token bytes:**
  - recipe A: 8.6 of 10.2 GB;
  - recipe B: 4.4 of 6.0 GB.

  The dense path, not the experts, sets the decode ceiling (§8.3).

---

## 3. What existing implementations do

### 3.1 Strata

Strata is GGUF-based with custom CUDA kernels.

| Mechanism | Detail | Consequence for NVFP4 on 32 GB |
|---|---|---|
| Expert placement | All experts pinned in RAM. VRAM slots are filled in the order of a shipped routing profile (all 24,576 experts ranked; `data/expert-profile.bin`). **Misses are never admitted on demand.** Every 4 rounds, `adapt()` (`src/program/generate.cpp`) pairs each layer's hottest non-resident experts (decayed count ≥ 2) against that layer's coldest residents and swaps while candidate ≥ victim + 1.5. It applies at most 96 swaps (highest gain first), then multiplies every count by 0.7. The victim is evicted at once and the newcomer admitted when its copy lands. Since issue #463 the next round **waits** for those swaps. The learned ranking can be saved and reloaded (`--expert-profile-save`). | A good **bandwidth** trade: in replay it makes ~7× fewer promotions than LRU (§9.3). Its counts decay by half about every 8 rounds, so it is mostly recency-driven, and it has 0-15% more misses than LRU. At 2.77 MB per expert, 96 blocking swaps cost ~5 ms. |
| Miss service | Distinct misses are split between CPU in-place compute and an in-graph SM copy kernel that reads mapped host memory (`pcie_frac`, 0.55 by default for IQ packs) | CPU plus PCIe co-service is the right primitive, but the copy kernel occupies SMs and serializes after the hit kernels. |
| Hit/miss resolution | **One GPU→host→GPU handshake per layer, even at a 100% hit rate.** A doorbell kernel writes the activation and ids to mapped memory, the host builds the plan, and a single-thread GPU spin kernel waits. A device-side planner exists but is off by default. | 48 serialized round trips per window, on the critical path. |
| Kernels | llama.cpp-style dp4a MMVQ, FP32 gate/up intermediates through global memory, BF16 hyper-connections (~1.27 GB read per window). About 2,000 graph nodes per window. | On a 5090 a verify round takes ~14-15 ms against a ~2 ms bandwidth floor: it is overhead-bound even at 98% hits. |
| Speculation | MTP with up to 3 drafts, Q2_0 MTP experts in VRAM, a reduced draft head (106K ids), and suffix lookup. 1.6-1.8× on a 5070. Drafts run serially after verify, with a stream sync per step. | Gains depend on a high hit rate. Strata measured the missed-expert union at 1.75 / 2.4 / 3.05× one token's for windows of 2 / 3 / 4 tokens. |
| PLE | 28.8 GB IQ4_NL repack. `O_DIRECT` `pread` on 16 threads, a 1M-row host CLOCK cache, and a **synchronous** gather before each window. | NVMe latency on the critical path whenever the row cache misses. |
| KV | INT8 KV. Above 64K context only 32,768 cells per attention layer stay in VRAM, as a CLOCK cache of 4-cell blocks over an authoritative host copy (96-99.4% block hits on a 5090). | **This is why Strata keeps a large expert cache at 262K context**: KV costs ~0.4 GiB of VRAM instead of several GB. Adopted for long contexts (§9.6). |
| Tuning | Offline `--calibrate`: sweeps `pcie_frac`, `spec_min_p` and pool workers, and keeps a value only if it is > 3% faster (no published gain). Startup probes: PCIe (best of 4 × 256 MiB bursts), CPU ISA and topology, cache auto-sizing, prefill chunk planning. Online: the adaptive tier, `DraftPolicy` EMAs. | §14 replaces this with host calibration, startup validation and online adaptation. |

Strata's printed hit rate is hits / (hits + CPU-served misses). PCIe-served misses (the default
`pcie_frac` share, 0.55) count in neither term, so the true VRAM hit rate is lower than printed.
On 12 GB cards, profile-only caching gave ~0.50 and adaptive caching ~0.72. A user-measured printed
hit rate above 80% at 262K context with Q8 on a 5090 is consistent with KV streaming keeping the
cache large; replay at comparable capacity gives the Strata policy 0.87-0.92 (§9.3).

**Forks.**

- **architectds/Strata** (`best`, 10 commits ahead of v0.1.38) adds three things:
  - **prefill CPU assist** — the CPU computes the thinnest non-resident experts (≤ 8 tokens,
    fewest first) while the copy engine streams the rest, balanced online by per-layer EMAs of
    CPU µs per unit vs GPU µs per streamed expert. Claimed 1.35-1.49× for 200-1,000-token prompts
    and ≈ 1.0× at 4K, measured on PCIe 3.0 with DDR4;
  - **equal-size prompt chunks** with a short-tail exception, plus a 1024-token-step search for the
    largest chunk that expert-slot lending allows (+21-59% at 9K-100K with `auto:16384`);
  - a **VMM-backed elastic expert cache** (`LEND`/`RECLAIM`) for on-demand vision.

  There is nothing new on decode or cache policy. The ideas are folded into §9.2 and §13.
- **chimpera/strata-nvfp4** (reported, not verified) runs Qwen3.8 NVFP4 experts with ~7,200-7,350
  slots on a 5090. It tunes the adaptive tier to every 2 rounds, decay 0.92 and 192 swaps, and
  reports 30-40% fewer misses than Strata's.

**Strata v0.1.39 (2026-10-04).** Strata was re-read at `6f32ec0`: 217 commits after `99f3dbd`,
v0.1.39 plus the #465, #583 and #646 follow-ups. The numbers below are Strata's own, measured on its
cards and GGUF packs. The table above stays as read at `99f3dbd`, but one statement in its
Speculation row is now out of date: "drafts run serially after verify, with a stream sync per
step". With the setup default `--spec 4 --spec-min-p 0.5`, each draft step is a launch followed by
a host spin on a mapped word, and the window is cut at the first draft with probability below 0.5.
With min_p 0 the steps launch back to back.

- **Decode (#646: `cfd3b72`, `055122c`, `deee447`, `f945515`).** No CUDA speed numbers published.
  - **Handshake.** A verify graph with no host handshake, used only when 100% of a stage's experts
    are resident. Elsewhere the doorbell copies x only when a routed id misses.
  - **Concurrency.**
    - The shared expert runs on a stream forked at the MLP mixer output.
    - A one-launch parallel resident planner replaces a one-thread loop.
  - **Kernels.**
    - Sub-warp packing for the K = 640 down kernel.
    - IQ codebooks staged in shared memory, kept only in the grouped kernels: single-matrix mmvq
      measured 10-20% slower with staging at 3-8 columns.
    - A batched PLE K/V projection.
  - **Drafting and submission.**
    - MTP steps launched back to back, with per-step PLE prefetch.
    - `cudaStreamQuery` kicks for WDDM submission.
  - **Exactness.** Fusions that broke bit-exactness became opt-in.
- **QSA.**
  - `cce52db`: past the register kernel's reach, the top-k uses 1,024 threads with a histogram per
    warp. Selection is 9-12× faster at 262K-524K cells, and a 243K-token prompt rose from 743 to
    934 tok/s (RTX 3060).
  - sm_90+ decode had already moved to an 8-CTA cluster top-k and multi-CTA block scoring that reads
    each key once per window (`bbe3d2a`, `a20f3b5`, just before the window). On an RTX 5070 the
    top-k fell from 200 to 22 µs at 262K.
  - `fd95405`: decode attention reduce-scatters the 12 heads' sums in `warp_sum`'s pairing order
    and issues four cells' V loads together, bit-exact.
- **Prefill (#583).**
  - The streamed ring is a byte budget shared with the chunk. The auto chunk is the largest one that
    keeps the ring full: a ring slot was worth ~0.53 tok/s, a chunk token ~0.05.
  - RTX 5070: 32K prompts +3 to +18.5%, 4K unchanged. Long-prompt bits change (another cached and
    streamed mix), gated by teacher-forced KL.
  - Chunks above 8,192 stay opt-in because chunk order changes output.
  - `d541220`, just before the window: one event per 16-expert group replaces a wait/record pair per
    expert, because each pair left ~10 µs of GPU idle under WDDM. RTX 5090 prompt time fell 8.6% at
    32K and 12.6% at 2K.
- **Batch slots on one GPU (#465).**
  - Policy:
    - A lone request runs the MTP path; two or more decode plain one-token rows; a request left
      alone returns to MTP.
    - Slots decode between a long prompt's chunks for half of each chunk's time.
    - A long read yields at a chunk boundary to a waiting prompt under half its length.
  - Measured on a 12 GB RTX 5070:
    - Two concurrent requests: 61.0 against 71.6 tok/s one at a time.
    - A lone request: 11-24% slower with slots configured.
    - The last of four first tokens arrives at 1.8 s instead of 11.2 s.
  - Setup recommends slots only where the cache still holds half the experts.
- **Memory and host.**
  - Opt-in hot VRAM resize through a VMM-segmented cache (#533): 4.3 GiB freed in 78 ms, decode 44 →
    33 tok/s; grown back in 92 ms with identical tokens.
  - P-cores plus half the E-cores on hybrid CPUs (#642), under Strata's static work split.
  - Linux THP and a host-RAM check.
  - Under WDDM, a device `cudaMalloc` failed after ~45 GiB of host pages were registered, so the
    head now loads before the arena (#620).
- **Serving and quality.**
  - Opt-in `effort_position: end` (#458), the Responses API, and literal think tags encoded as text.
  - A 256-repeat stop, added after non-finite q8_1 scales produced NaN (#606).
  - A restore-speed check (#528), and slot and conversation-cache metrics.
  - UD-IQ4_XS is a regular choice.
  - `f23ea57`: a control that changes only the prefill chunk moves teacher-forced KL by
    0.023-0.038 at 8.5K-99K tokens (V100).
- **Unchanged:** the adaptive tier, `pcie_frac` and `DraftPolicy`. There is still no prefill CPU
  assist in Strata.

Most of #646 removes a host handshake this design never had. The transferable mechanisms:
- a codebook held in registers, which exposed a local-memory table in our expert kernels;
- long-context QSA selection;
- prefill streaming and lending;
- a warm start;
- C > 1 scheduling around long prompts.

They are ranked in §19.3.6.

### 3.2 ninfer-ext

ninfer-ext is a fork of NInfer.

| Mechanism | Detail | Consequence |
|---|---|---|
| Recipe | Expert codes and block scales imported as NVFP4, but `weight_scale_2` is stored as Infernix's FP32 **divisor**, a rounded reciprocal of NVIDIA's multiplier, and each layer quantizes activations with **one** input scale, the minimum over its experts' `input_scale`. **Dense layers re-quantized to Q8** (attention, GDN, HC, shared expert, PLE projections). `lm_head` Q6, embedding Q8, router BF16. MTP experts **requantized** from block FP8 to NVFP4. The n-gram table is FP8 with the global scale copied onto every row. | 5.3 GB of device weights, about 3.9 GB of dense reads per token. Neither the represented expert weights nor their activations are NVIDIA's exactly, and the MTP experts lose precision. The per-row scale plane doubles the PLE I/O ranges. |
| Cache | A global LRU stamped per layer call, with no frequency term. A 1024-thread single-CTA resolve kernel radix-selects over all slots for every layer that has misses. | Measured 0.805 hits at C=1 with 6,347 slots (26% of experts), MTP K=3. |
| Miss service | **PCIe only.** A 64-CTA SM copy kernel reads zero-copy at ~36 GB/s after router → top-k → resolve. Hit GEMVs overlap the fetch, but those GEMVs take ~15 µs while one miss takes ~77 µs. | Every layer with a miss stalls. There is no CPU compute and no cross-layer prediction. |
| Host sync | `device.synchronize()` every round, followed by a host PLE gather (one column per call) and ingress build | GPU idle between rounds |
| Graph | ~1,550-1,600 kernel nodes per decode token. Decode expert GEMVs are SIMT FP32, with no tensor cores. | Measured 10.2 ms per token against a ~3.2 ms byte floor |
| MTP | Its own expert bank competes in the same LRU, a full Q6 `lm_head` read per draft, and serial PLE gathers. Adaptive MTP falls back to plain decode at C ≥ 2. | MTP **slower** than plain at C=1 on chat (82 vs 98.5 tok/s) |
| Prefill | Whole-layer staging banks, a W4A4 tensor-core route for staged experts, hundreds of `cudaMemcpyAsync` calls per layer, and two stream syncs per forward | 3,400-3,700 tok/s at 4K-64K, 431 tok/s at 512 |

### 3.3 Other engines

| Engine | Relevant mechanism | Measured on this model |
|---|---|---|
| **FreeToken** | Pinned host expert banks with a global VRAM LRU. Misses either copied by an SM gather kernel (`offload`), computed on the CPU (`cpu`), or split between them by the ratio of PCIe to CPU bandwidth (`hybrid`, "q\*"). PLE `disk` backend: io_uring + `O_DIRECT` into pinned staging, then one H2D copy and dequant inside the graph. | 61-67 tok/s on a 5090 with 128 GB. One published nsys profile (Gen5 x8, 16% of experts cached): expert transfer 52.8% of the token at 20.5 GB/s with **no overlap**; BF16 dense GEMVs, mostly hyper-connection shapes, 22.5%; ~1,852 kernels per token; disk PLE 0.76 ms per token. |
| **FreeToken trace replay** (same community repo, 4,008 slots) | Cache policies compared on this model's routing traces | Fewer misses than LRU: Belady **+52-62%**, two-hit probation +1-16%, decayed global LFU −8% to +17%, resident LFU **−47% to −61%**. **Recency matters; pure frequency loses.** |
| **SGLang** | `qwen4_exp.py` with NEXTN MTP (3 steps, 4 draft tokens). PLE either pinned in host RAM with a Triton gather from the host pointer, or a mapped file that **does not work on a discrete 5090**: it needs pageable host-table access. Lookup on a side stream. Fused n-gram hash. Expert offload only through the generic KTransformers integration. | All-resident on an RTX PRO 6000: TPOT 11.4 ms (≈ 87 tok/s) plain, 6.0 ms (≈ 165 tok/s) with MTP (accepting 2.9-3.3 of 4). KTransformers + SGLang on 2× 5070 Ti: 36-37 tok/s. |
| **vLLM** | Native `qwen4_exp`: Triton QSA indexer, CuTe-DSL HC kernels. The QSA `qkv_proj` explicitly bypasses FP4. Its NVFP4 MoE backends reduce each expert's gate/up `input_scale` pair to one value, and some share one activation scale across a layer's experts. The Engram/PLE table is pinned on the CPU and read through UVA, prefetched on a side stream during the previous layer. Disk/mmap PRs are open. Expert offload is only generic (`--cpu-offload-gb`, layer-group prefetch); "CPU offload: TODO" in its tracking issue. | No single-5090 number found. DGX Spark NVFP4: 34-44 tok/s. |
| **llama.cpp** | `qwen4exp` merged (HC, QSA with pooled keys, one PLE layer). Experts placed by `-ncmoe`, with the CPU computing the experts that are not on the GPU. | ~43-52 tok/s on a 5090 with 128 GB (community reports, not verified) |
| **TensorSharp** (C#, ggml backends) | `qwen4exp` from GGUF. `--n-cpu-moe` keeps whole layers' experts in RAM and multiplies them on the host at a **graph seam per offloaded layer** (the GPU pauses after the router). A selected-expert VRAM cache exists, without lookahead prefetch. CPU kernels are ggml's K-quant dot products (no W4A4 integer path, no AVX-512-specific kernels). QSA selection applied as an attention mask, so it saves no KV bandwidth. KV is F16 and device-only. Its CPU worker team starts the calling thread immediately and hands out chunks atomically; spin versus sleep and the thread count are calibrated (a cliff above ~56-64 threads). Verify uses **exact verify-row kernels**, so speculative output equals plain output, at a measured 2-11% verify cost. | No 5090 number. UD-Q2_K_XL on 3× A40 (layer split, all resident): 49 tok/s plain; shared MTP head 83 tok/s on code copying but **44-46 vs 52 tok/s on prose**, so its speculation needs a cost governor. On a 48 GB Mac it runs larger than memory by reading experts and n-gram rows from SSD. |

Literature mechanisms relevant to batch-1 decode with a hot GPU cache:

| Work | Mechanism | Used here |
|---|---|---|
| Eliseev & Mazur (2312.17238), FATE (2502.12224) | Next layer's gate applied to the current hidden state to predict the next layer's experts; FATE reports cosine similarity > 0.83 and 97% prefetch accuracy | §8.7. Recall must be measured: the HC streams may weaken it on this model. |
| SP-MoE (2510.10302) | Prefetch the experts implied by draft tokens before verification | §11.4 |
| Fiddler (2402.07033), KTransformers (SOSP'25), FreeToken q\* | CPU computes misses in place, or a balanced CPU/PCIe split | §10, §8.6 |
| HOBBIT (2411.01433) | Low-precision copies for less important missed experts | Rejected: it changes NVIDIA's weights (§1.1) |
| AdapMoE (2408.10284), FATE | Per-layer cache allocation; shallow layers miss more | One global pool. Hard per-layer quotas lost 1-35% in replay (§9.3). |
| Zhang, "Reproducible evaluation of MoE expert caching" (2608.07911) | Event-atomic replay. LFRU, f / (age + 1), is the best causal policy in 12 of 13 workloads. 84-97% of the Belady gap comes from victim ranking. A learned next-use predictor did **worse** than LFRU. | Base policy of §9.3 |
| SeqMoE (2609.12978) | A sequence predictor of expert activations several tokens ahead drives a probabilistic Belady (reported 91.7% / 97.0% hits at 25% / 45% residency) | Research option behind M1's predictor evaluation (§9.3) |
| Local routing consistency (2505.16056) | Models with shared experts show weaker token-to-token expert reuse | Explains the modest LRU hit rates; Belady's gap is the headroom (§9.3) |
| DeepSeek Engram (2601.07372), InferenceX PR #3080 | Token-determined table rows can be prefetched; tiered caching driven by Zipfian n-gram reuse | §12 |

### 3.4 Lessons that shape this design

1. **Hits must not involve the host.** The residency lookup belongs on the device inside the graph,
   and the host should hear about a layer only when it has a CPU-served miss (Strata's flaw 1).
2. **Misses are a host-DRAM bandwidth problem, not a PCIe-versus-CPU problem.** DRAM has to be kept
   busy during the GPU-only phases (prefetch) as well as during the MoE phase (CPU compute)
   (ninfer-ext: PCIe only; Strata: no prefetch).
3. **The dense path sets the decode ceiling.** NVIDIA keeps every dense tensor in BF16: 8.6 GB per
   token against the experts' 1.33 GB, and 22.5% of FreeToken's token time. Under the fidelity
   requirement the levers are 8-bit dense weights where quality allows (recipe B halves the bytes),
   kernels at the bandwidth roofline, and fewer kernels per layer: 8-10 instead of ~32-40 (all
   three engines).
4. **LRU is not the best policy; a recency × frequency hybrid is.** On this model's traces LFRU has
   15-22% fewer misses than LRU at every capacity tested (§9.3). Pure LFU and per-layer
   partitioning lose, and Belady still has 2-3× fewer misses. Promotion runs on the copy engine in
   the background, and a slot is reused only after its epoch has retired (Strata's #463 blocking).
5. **Speculation does not multiply misses per accepted token.** In replay, verify windows of 1, 2
   and 4 tokens have the same misses per accepted token (§4.2). Only rejected drafts add misses. The
   drafter must never wait on the host, and the draft length comes from a cost model that includes
   misses (ninfer-ext's MTP regression).
6. **NVMe reads must stay off the critical path.** Every token is on the host before its rows are
   needed, and a column's 2,560 B crosses PCIe in ~1-2 µs. So the host hashes and reads the rows,
   hidden behind the draft and layer 0 by a gate before `ple_embed` (§12.3). A device row cache
   does not pay, because decode misses are compulsory (§12.4). Both engines gather synchronously
   before the window.
7. **Scale handling decides fidelity.** Engines that re-derive NVIDIA's scales (ninfer-ext's reciprocal
   divisor and per-layer input scale, the merged scales of vLLM's MoE backends) silently run a
   different model. This design imports every quantized tensor bit-exactly, keeps each expert's own
   activation scale, and computes routed experts with exact integer block products (§16.2).

---

## 4. Performance model

### 4.1 Bound

For one decode round with T token columns (T = 1 for plain decode, K+1 for a speculative verify),
the round time is approximately:

```text
t_round ≈ t_gpu(T) + Σ_layers t_exposed(l) + t_draft

t_gpu(T)     = Σ_kernels (bytes_k / BW + t_k)                         (§8.3 budget)
t_exposed(l) = max(0, t_service(misses of l) − t_overlap(l))          t_service from §8.6
misses per layer ≈ U_T · (1 − h),   U_T = distinct experts per layer for T columns
```

Two ceilings bound the misses:

```text
host DRAM:  Σ_l (CPU-served + DMA'd + promoted bytes)  ≤  BW_dram · t_round     (≈ 75-90 GB/s measured on desktop DDR5)
PCIe:       Σ_l (DMA'd + promoted bytes)               ≤  BW_pcie · t_round     (≈ 56 GB/s copy engine)
```

**Boundary cost** is measured on the RTX 5090 (§22, sm120-fp4 micro-floor report):

| Measurement | Cost |
|---|---|
| Isolated graph-replayed kernel | ≈ 5 µs + bytes / 1.6 TB/s. An empty kernel costs 2.82 µs. |
| PDL-chained pair at ≤ 2 CTAs/SM | +0 µs |
| PDL-chained pair at 4 CTAs/SM | +2 µs |
| Cooperative grid barrier | 1.4-2.0 µs |
| A 21 MB weight-streaming kernel alone | 18.2 µs (1.17 TB/s) at 4 CTAs/SM; 29.4 µs at 1 CTA/SM with a shallow pipeline |
| Long streaming kernel | 87-92% of the 1.79 TB/s datasheet bandwidth (215 MB at 1.54 TB/s) |

There are four levers, each attacked by the design:

| Lever | Mechanism |
|---|---|
| **h**: hit rate | Frame pool (§9.1); LFRU (§9.3); prefetch converting misses into staged hits (§8.7); KV loans and the KV host tier keep the pool large (§9.2, §9.6) |
| **t_exposed**: the miss stall | CPU W4A4 in place, with no handshake for hit-only layers; x pre-published; warming (H2); arbiter (§8.6); stall-time L2 warming of the next layer (§8.3) |
| **t_gpu**: dense path and kernel latency | Recipe B (8-bit dense); PDL-chained kernels at ≤ 2 CTAs/SM with pre-dependency weight prefetch; per-kernel budget (§8.3); megakernel option (H1) |
| **Tokens per round** | MTP with a miss-aware draft length (§11) |

### 4.2 Projection

**Hit-rate evidence.** This model's published routing traces (FreeToken community repo: 3 × 383
single-client decode tokens, English technical writing) were replayed through candidate policies
(§9.3) at the primary configuration's frame counts (§15.1):

| Frames (share of experts) | LRU hit rate | LFRU hit rate (misses per token) | Belady |
|---|---:|---:|---:|
| 9,652 (39.3%): recipe B, 4K context | 0.963 | **0.969** (14.7) | 0.985 |
| 9,433 (38.4%): recipe B, 256K context, KV host tier | 0.960 | **0.967** (15.8) | 0.984 |
| 8,130 (33.1%): recipe A, 4K context | 0.939 | **0.950** (23.9) | 0.976 |
| 7,911 (32.2%): recipe A, 256K context, KV host tier | 0.935 | **0.946** (25.7) | 0.975 |

An independent data point: llama.cpp's per-layer LRU on this model (PR #27861) reports 81% / 86.9%
/ 90.2% / 98.5% at 64 / 96 / 128 / 384 slots per layer. Recipes A and B correspond to ~170 and ~200
slots per layer, between the 90.2% and 98.5% points; LFRU over one global pool should do better
than per-layer LRU. Other replay findings:

- Reuse distance has a median of 3 tokens.
- Verify windows of 1, 2 and 4 tokens give the same misses per **accepted** token.
- Long, diverse sessions will miss more than these short traces, so the projection spans a range of
  hit rates. M1 replays broad traces (§19).

**GPU assumptions** (from the §8.3 budget; t_gpu(4) reads U_4 = 28 distinct experts per layer):

| Recipe, case | t_gpu(1) | t_gpu(4) | Drafting (3 steps) | Accepted per round |
|---|---:|---:|---:|---:|
| B, target (§8.3 budget, PDL overlap holds) | 4.5 ms | 6.1 ms | 0.6 ms | 2.8 |
| B, conservative (isolated-kernel costs, no overlap) | 6.2 ms | 7.8 ms | 0.85 ms | 2.6 |
| A, target | 7.3 ms | 8.9 ms | 0.78 ms | 2.8 |
| A, conservative | 8.9 ms | 10.4 ms | 1.0 ms | 2.6 |

**Miss model:**

- A cold CPU-served expert takes 42 µs (W4A4 on the CPU is DRAM-bound); the handshake 6 µs. The
  CPU serves a layer's misses one expert after another.
- The GPU's hit work overlaps the CPU service by 21.5 µs per layer at T = 1 and 56.5 µs at T = 4.
- Prefetch stages each miss with probability 0.8 (recall), up to what the copy engine lands in one
  layer: about 1.5 experts per layer for recipe B and 2.6 for recipe A at T = 1. Recipe A's layer
  is longer, so its prefetch window holds more.
- An on-demand DMA (12 µs request + 55 µs per expert) replaces CPU service only when it finishes
  earlier and prefetch has left the copy engine idle.
- Misses per layer are Poisson; rejected drafts add 15%. The host-DRAM ceiling is 80 GB/s.
- Warming (H2) and the megakernel (H1) are not included.

**Projected decode in the primary configuration at a 4K context, target (conservative), tok/s:**

| Hit rate h | Recipe B plain | Recipe B MTP K=3 | Recipe A plain | Recipe A MTP K=3 |
|---|---:|---:|---:|---:|
| Replay (B 0.969 / A 0.950) | ~220 (~160) | ~415 (~300) | ~135 (~110) | ~285 (~225) |
| 0.90 | ~200 (~150) | ~345 (~275) | ~130 (~110) | ~265 (~220) |
| 0.85 | ~185 (~145) | ~270 (~235) | ~125 (~110) | ~230 (~200) |
| 0.80 | ~165 (~135) | ~210 (~195) | ~120 (~105) | ~195 (~175) |

How to read the table:

- **The dense path sets the ceiling.**
  - Recipe A reads 8.6 GB of BF16 dense weights per token, so even all-resident decode tops out
    near 137 tok/s; recipe B's ceiling is 222 tok/s.
  - Recipe B halves those bytes and frees ~1,520 more frames, which is why it is recommended
    (§6.1) if it passes the quality gate.
- **Speculation pays at every row**, more so with recipe A, whose dense read is amortized over
  ~2.8 tokens. At h ≤ 0.85 a 4-column verify puts ~5 misses per layer on the host, and the DRAM
  ceiling starts to bind. That is where the arbiter's DMA overflow, H2 and H3 matter.
- **References.** SGLang with all experts resident on a 96 GB sm_120 card (BF16 dense, generic
  kernels) measures 87 tok/s plain and ~165 with MTP. The Strata baseline is 72-80 tok/s.
- **The largest risk is the GPU budget.** It is the gap between the target and conservative
  columns: whether PDL chaining hides kernel ramp-up as it measured on this card. M6 measures it
  first, and H1 is the fallback.
- **Long context.** In the primary configuration the KV host tier keeps ≥ 9,433 frames at 256K, and
  QSA's index scans and host-tier fetches add ~0.16 ms per token. Recipe B then decodes at ~97% of
  its 4K rate on these traces (~210 tok/s plain, ~405 with MTP). Without the tier, 256K `int8` KV
  would cost 1,381 frames (LFRU 0.953 at 8,300 frames).

---

## 5. Design overview

```text
 ┌──────────────────────────────── RTX 5090, 32 GB ────────────────────────────────┐
 │ Dense Text: BF16 8.6 GB (A) or 8-bit 4.4 GB (B) · MTP dense · MTP q4 experts    │
 │ GDN and conv state · decode workspace · n-gram landing area (io buffer)         │
 │ ┌─────── frame pool: 8,153 (A) or 9,675 (B) × 2,764,800 B (21-25 GiB) ────────┐ │
 │ │ cached experts │ staging │ KV pages + loans │ prefill arena │ vision        │ │
 │ └─────────────────────────────────────────────────────────────────────────────┘ │
 │ residency[49×512] · land_seq[frame] · KV block tables                           │
 └──────▲─────────────────────▲─────────────────────────▲──────────────────▲───────┘
        │ copy engine (DMA)   │ zero-copy reads/writes  │ zero-copy        │ WriteValue
        │                     │ (mailboxes, CPU y_e)    │ (PLE rows)       │ (residency)
 ┌──────┴─────────────────────┴─────────────────────────┴───────────┐  ┌────┴───────────┐
 │ 96 GB host                                                       │  │ NVMe           │
 │  pinned expert banks on huge pages: 24,576 NVFP4 records         │  │ n-gram table   │
 │    (67.9 GB)                                                     │  │ 52.4 GB FP8,   │
 │  embedding rows · n-gram row cache · mailboxes · host tiers      │  │ 4 KiB blocks   │
 │  transfer agent ─ cache policy, all DMA, residency updates       │  └────▲───────────┘
 │  CPU expert engine ─ spin workers, exact W4A4 arithmetic         │       │
 │  engine worker ─ n-gram ids, row cache, batched reads ───────────────────┘
 └──────────────────────────────────────────────────────────────────┘
```

A decode round at C = 1 without speculation. The host loop is not on the critical path, and the
host learns about a layer only when it has a CPU-served miss:

```text
round r: embed(host row) ─ layer 0 ─ [layer 1: ple_embed reads rows the host uploaded with the io prefix, §12.3]
 per layer l:
   K1a/K1b HC_attn ─ K2/K3 (GDN) or K2q/K2b/K3a/K3b (QSA) ─ K4 out_proj + inject ─ K5a/K5b HC_mlp (x → host)
   K6 router(l, l+1): top-10 ─ residency ─ jobs ─ [miss_req → CPU] ─ [prefetch_req → agent]
   K7a gate/up slices: A4(x) ─ exact W4A4 ─ SwiGLU ─ A4(h)
   K7b down row tiles ─ combine with the CPU's y_e (waits only if the layer had a CPU miss) ─ inject
 head ─ sample ─ token egress ─ round_done; host: n-gram ids(t+1), row cache, NVMe reads (§12.3)
```

Four ideas carry the design:

- **Exact expert arithmetic makes placement free.** A routed expert's output is the same bits
  whether a cache frame, a staging frame or the CPU computed it (§16.2). The cache, the prefetcher
  and the miss arbiter can therefore move work anywhere, every round, without changing the model.
- **The device only looks up residency; the host owns policy.** Hits never involve the host. All
  DMA, LFRU accounting and frame reuse run in one host transfer agent (§9.4).
- **The dense path runs at the bandwidth roofline.** Every kernel streams weights with PDL
  pre-dependency prefetch, and any stall prefetches the next layer (§8.3). Recipe B halves the
  dense bytes where quality allows (§6.1).
- **Device memory is one pool.** Expert frames, KV, prefill arenas and vision encode windows share
  it, and reserved but unwritten KV holds experts on loan (§9.2).

---

## 6. Artifact and conversion

### 6.1 Recipes `qwen3_8_flash_next_nvfp4` (A) and `qwen3_8_flash_next_nvfp4_dense8` (B)

The source is [nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4).
Both recipes import every quantized NVIDIA tensor bit-exactly. They differ only in the BF16
tensors that recipe B moves to 8 bits. M0 confirms each source form from `hf_quant_config.json`
and the tensor dtypes before the recipe is frozen.

| Tensor class | Source form | Recipe A | Recipe B | Reason |
|---|---|---|---|---|
| Routed experts (48 × 512) | ModelOpt NVFP4: E2M1 codes, E4M3 per 16, FP32 `weight_scale_2` and FP32 `input_scale` per matrix | **Exact import** as `nvfp4_mul` (below), layout `nvfp4_expert_rg16_v1` (§6.2). Every expert's `input_scale` is kept per matrix as an FP32 model-role tensor. | Same | NVIDIA's weights and NVIDIA's activation calibration, with nothing re-derived |
| MTP routed experts (512) | Qwen's block-scaled FP8 per expert: E4M3FN `{gate,up,down}_proj.weight` and a multiplier per 128 × 128 tile (`FP8_PB_WO`; BF16 in NVIDIA's checkpoint, widened exactly), in `model-fp8-mtp-ple.safetensors` | **`q4_g64_fp16`, MSE-chosen group scales** (`grouped_mse`) from the exact values, 1.34 GB, all device-resident (§11.2) | Same | MTP affects only acceptance, never output; Strata stores these experts at 2.25 bits. The resident expert kernel reads Q4 or Q8 banks, so no format stores them exactly. |
| N-gram table (128 shards) | FP8 E4M3 plus one BF16 scalar scale | **Exact import** into the NVMe volume of §12.2. One scalar scale, no per-row plane. | Same | One I/O per row instead of two |
| GDN q/k/v/z projection and `out_proj`; QSA QKVG and `o_proj`; shared experts; HC mixers (down and up); PLE key/value projections; `lm_head` | BF16 | BF16 | **8-bit, W8A16.** `fp8_e4m3fn_row_bf16` (producer `fp8_row_maxabs`) per class; `q8_g32_fp16` (`grouped_absmax`) for a class where FP8 fails §16.3 and Q8 passes; BF16 for a class where neither passes. | 4.2 GB fewer dense bytes per token and ~1,520 more frames (§4.2). These classes are 97% of the dense bytes. |
| Router, shared-expert gate, GDN `a`/`b`, QSA indexer projection | BF16 | BF16 | BF16 | Small (3% of dense bytes), and routing- or selection-sensitive |
| MTP dense (its QSA block, HC, shared expert, projections) | BF16 | `q8_g32_fp16` (option H13 adopted); router and shared-expert gate stay BF16 | Same | Affects only acceptance; halves the drafter's dense bytes per step |
| Proposal head (`--proposal`, for `--lm-head-draft`) | `lm_head` rows of the 131,072 most frequent tokens | `q4_g64_fp16`, 178 MB | Same | Draft head of the MTP drafter only |
| Token embedding | BF16 | BF16, **host-resident** (§6.3) | Same | Saves 1.27 GB of VRAM, about 480 frames |
| Norms, `A_log`, `dt_bias`, conv weights | BF16 / FP32 | Direct | Direct | — |
| Vision | BF16 | BF16 | BF16 | Not on the decode path; both deployed artifacts contain it. With `--vision-offload` (on by default for this model) the weights stay in pinned host RAM and stream per encode window; only the output handoff and oversize windows borrow frames (§9.2, §19.3.2). |

Recipe A is the bit-exact artifact (`Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix`) and recipe B the 8-bit
dense one (`...-Dense8-Infernix`); both share the drafter, the proposal head and the n-gram volume.
`python -m tools.flash_next.verify_artifact` checks an artifact against the checkpoint object by
object (direct casts byte for byte, BF16 widened to FP32 by value, expert banks decoded back to their
codes, scales and multipliers), the n-gram volume row by row, and lists the re-quantized parameters
per component (proposal bindings are counted apart). On 2026-10-08 recipe A had 24,576 routed experts
and 1,616 tensors with no mismatch, 320,001,536 identical n-gram rows, and only the 1,556 MTP
parameters re-quantized; recipe B had 987 tensors with no mismatch and its 629 text-model
parameters (`lm_head` included) re-quantized as well. The PLE hash tables (`layer_multipliers`, `ngram_heads_*`) are derived from the
config, which conversion checks against the checkpoint's buffers.

Recipe A runs its BF16 dense classes on the BF16 linear routes (`src/ops/linear/bf16`; NInfer's tuned selectors
where they measured fastest, the sweep's tiles elsewhere; at T ≤ 8 one bit-identical route family
per shape, so a column's bits do not depend on the decode width), the fused hyper-connection mixer
with a BF16 codec, a register-streamed BF16 SwiGLU for the shared expert at 1..16 columns, and the
tensor-core FP32 projection for `lm_head` at 1..16 columns (701-710 µs against a ~709 µs DRAM floor).
It decodes ~18 % slower than recipe B end to end (README, "Bit-exact against Dense8") at the same
measured quality.

**Two new weight formats** are needed, each defined in [tensor formats](tensor-formats.md) with an
exact decode oracle:

- **`nvfp4_mul`**: the words of `nvfp4` (E2M1 codes, one E4M3FN scale per 16) with an FP32
  **multiplier** `m_w` per matrix: `W[n,k] = e2m1(c[n,k]) · e4m3fn(s[n,⌊k/16⌋]) · m_w`.
  - Infernix's `nvfp4` divides by `d_w`. Its ModelOpt importer stores `fl(1 / weight_scale_2)` as
    `d_w`, which equals NVIDIA's weight only when the scale is a power of two. ninfer-ext inherited
    the same rounding (§3.2).
  - By the rule of [tensor formats §3.4](tensor-formats.md#34-fp8_e4m3fn_row_bf16), a multiplier
    coefficient is a different format, not a variant of `nvfp4`.
  - Its only producer is `import_encoded`.
- **The checkpoint's block-scaled FP8 MTP experts** (`W[n,k] = e4m3fn(c[n,k]) · s[⌊n/128⌋, ⌊k/128⌋]`,
  edge blocks truncated) are decoded exactly by the converter (`dequantize_fp8_block128` in
  `tools/convert/sources/modelopt.py`) and re-quantized by both recipes (§6.1, `q4_g64_fp16`). A stored
  `fp8_e4m3fn_block128_f32` format with a `block128_scale_v1` layout was planned here and registered,
  but no recipe produced it; it was removed on 2026-10-08 (audit M10).

**Model-role tensors** beside each layer's expert bank:

- `weight_scale_2` (as `m_w`) and `input_scale` (as `g`) for every expert's gate, up and down
  matrices: 6 × 512 FP32 words per layer.
- The loader derives `α = fl32(m_w · g)` once per matrix on the host. CPU and GPU read the same
  words, so the epilogue multiplier is identical on both.
- ModelOpt's export unifies the input scale of matrices that share an input (gate and up). M0
  verifies this. If an expert's gate and up scales differ, x is quantized once per scale.

**Considered and rejected for recipe B:** importing the dense FP8 tensors of
`Qwen/Qwen3.8-Flash-Next-FP8`. They are official, but the 128×128 FP8 blocks are coarser than
`q8_g32_fp16`, and the import would need a second ~180 GB source checkpoint. If a class fails the
gate as Q8, it would fail as block FP8.

### 6.1.1 Recipe `qwen3_8_flash_next_nvfp4_orcarouter` (C)

The source is [orcarouter/Qwen3.8-Flash-Next-Uncensored-NVFP4](https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored-NVFP4)
(revision `cddc6ec5`): an abliterated (refusal-removed) Qwen3.8-Flash-Next exported **weight-only** by
llm-compressor (`compressed-tensors`, no activation scales), for runtimes that keep activations in
BF16. Recipe C imports every quantized tensor bit-exactly and runs it as the checkpoint is served:
experts W4A16 (§16.2.1), dense projections W8A16. Its BF16 classes follow recipe B.

| Tensor class | Source form | Recipe C | Reason |
|---|---|---|---|
| Routed experts (48 × 512) | `nvfp4-pack-quantized`: `weight_packed` (E2M1, two per byte, low nibble first), E4M3 `weight_scale` per 16, FP32 `weight_global_scale` (a **divisor**) per matrix | **Exact import** of the codes and block scales as `nvfp4_mul` in `nvfp4_expert_rg16_v1`; the multiplier is `fl32(1 / weight_global_scale)`, the import's one rounding. No `expert_input_scales`: the bank's use is A16-only, which selects the W4A16 arithmetic | The checkpoint's weights; its activations are BF16, so no scale is borrowed or calibrated |
| GDN q/k/v (`in_proj_qkv`) and z, `out_proj`; QSA q/gate (`q_proj`), k, v, `o_proj`; shared experts | `naive-quantized` FP8 E4M3 per output row, BF16 row scales | **Exact import** as `fp8_e4m3fn_row_bf16`, A16-only (W8A16). The QSA query/gate/key/value form one parent and the BF16 indexer another (explicit recipe groups); the model runs them as two projections | The checkpoint's weights and its serving arithmetic; FP8-row A16 routes for the five Flash-Next shapes (`src/ops/linear/fp8/shapes`, selected by `bench/ops/fp8_flash_next_sweep.cu`) |
| HC mixers, PLE key/value projections, `lm_head` | BF16 | `q8_g32_fp16`, as recipe B | Recipe B's dense-byte saving where the checkpoint is BF16 |
| Router, shared-expert gate, GDN `a`/`b`, QSA indexer, token embedding, vision, norms | BF16 | BF16 (`A_log`, `dt_bias` widened exactly to FP32) | As recipes A and B |
| MTP (dense and 512 routed experts, fused `gate_up_proj` [E, 2I, H] with the gate rows first, `down_proj` [E, H, I]) | BF16 | Recipe A's drafter: `q8_g32_fp16` dense, `q4_g64_fp16` (MSE) experts | Affects only acceptance |
| N-gram table (128 shards) | **BF16** (the source keeps it unquantized) | FP8 per tensor in BF16 arithmetic: `s = bf16(amax / 448)`, `code = e4m3_rn_satfinite(bf16(v / s))` (`qwen4_exp.NgramTable`) | This is the rule NVIDIA's FP8 table follows: the abliteration leaves the table untouched, and on all 327,680 sampled values (and its scale word) the result equals NVIDIA's codes, so recipe C reuses recipes A and B's 52 GB volume (the converter checks 512 rows; `verify_artifact` checks every row) |

The configs differ only in the attention layer name (`qwen_sparse_attention` for `full_attention`).
Recipes A and B refuse a weight-only bank, and recipe C refuses a ModelOpt one.

**Verification (2026-10-09).** `verify_artifact` finds every expert (24,576), every imported tensor
(1,323) and every n-gram row (320,001,536) equal to the checkpoint and the reused volume.
Teacher-forced on the frozen texts (2,557 positions, INT8 KV, `tools.flash_next.strata_compare`):
perplexity 4.772 against Strata 0.1.40 UD-Q4_K_XL's 4.844 (ΔNLL −0.015 ± 0.010 nats; code +0.010,
document +0.010, chat −0.052) and recipe B's 4.653 on the same build; the gap to recipe B is the
abliterated weights. The qwen4_exp prefix-cache, preemption and decide real tests pass on it.

**Column invariance of the FP8 routes.** The first sweep paired a T = 1 GEMV with 8 values per lane
in 4 accumulator chains with a SIMT tile of 16 values in one chain. Their sums round differently, and
greedy MTP left plain decode at the 67th generated token. The test and the sweep had not shown it:
patterned E4M3 codes with uniform activations sum exactly in FP32 in any order. Both now use every
finite E4M3 code and activations over 25 binades. A column's order is fixed by the values per lane
and the chain count alone, so each shape takes its GEMV and its SIMT tile from one family; 16 values
in one chain is the fastest family on all four CUDA-core shapes:

| Shape | T = 1 GEMV, before → after | SIMT T = 2..8 |
|---|---:|---:|
| 16384 × 2560 | 27.3 → 27.2 µs | 29.3-31.4 µs |
| 13312 × 2560 | 23.1 → 22.9 µs | 25.2-29.2 µs |
| 2560 × 6144 | 12.9 → 12.9 µs | 14.8-17.0 µs |
| 1280 × 2560 | 5.1 → 4.7 µs | 7.3-9.3 µs |

The shared expert's down (2560 × 640) runs one tensor-core tile family at every width. On the real
model, 128 teacher-forced positions as calls of 1 and of 5 columns give equal logits (they differed at
every position before), and greedy MTP equals plain decode over 128 tokens.

### 6.2 Expert storage layout `nvfp4_expert_rg16_v1`

The pinned host bank is read by the GPU (DMA into a frame, then the expert kernels) and by the CPU
(in-place computation). Only one copy fits in 96 GB, and Infernix forbids runtime repacking. The
artifact therefore stores each expert as one contiguous record of 2,764,800 B (= 675 × 4 KiB)
that both processors use directly.

- **Matrices.** The gate/up matrix (1,280 rows, K = 2,560) comes first, with rows interleaved:
  row 2i = gate_i, row 2i+1 = up_i. The down matrix (2,560 rows, K = 640) follows.
- **Units.** Rows form groups of 16. For each row group g and 16-element block b there is one
  144-byte unit: 128 bytes of codes, then 16 bytes of E4M3FN scales (row i of the group at
  byte i).
- **Unit order.** g-major, then b ascending, for both matrices. A 32-intermediate gate/up slice σ
  (row groups 4σ…4σ+3) is one contiguous 92,160-byte run. A 16-row down tile is one contiguous
  5,760-byte run.
- **Codes inside a unit.** The 128 code bytes are four 32-byte quads, q = 0..3, each covering
  k = 16b + 4q … 16b + 4q + 3. In quad q, byte j holds two nibbles:
  - the low nibble is row (j div 4) at k = 16b + 4q + (j mod 4);
  - the high nibble is row 8 + (j div 4) at the same k.

  The 32-bit word at offset 32q + 4i therefore holds four consecutive k of row i (low nibbles)
  and of row i + 8 (high nibbles).

| Matrix | Units per row group | Bytes per row group | Row groups | Bytes |
|---|---|---|---|---|
| gate/up | 160 | 23,040 | 80 | 1,843,200 |
| down | 40 | 5,760 | 160 | 921,600 |
| **Record total** | | | | **2,764,800 B = 675 × 4 KiB** |

How each consumer reads it:

| Consumer | Access |
|---|---|
| DMA | One contiguous copy per expert into a frame |
| GPU narrow route (§8.5) | A warp's 32 four-byte loads cover one unit's codes in 128 contiguous bytes. A thread's word decodes to exactly one register pair of the A fragment of `mma.sync.m16n8k32.s8` (rows g and g+8, four consecutive k), or feeds `dp4a` for one or two columns. |
| GPU wide route (prefill, n > 8) | Units are converted into the `mxf4nvf4` MMA operand order in shared memory. No tensor-core layout leaks into the record. |
| CPU | A 32-byte quad becomes one AVX-512 vector with lane r holding row r's four codes, after one broadcast load, one masked 4-bit shift and one AND. `vpshufb` decodes it and `vpdpbusd` multiplies (§10.2). |

The layout id and its byte-exact definition are recorded in [storage layouts](storage-layouts.md).
MTP experts use their source layout (row-major codes, row-major block scales), since only the GPU
reads them.

### 6.3 Residency classes in the v3 container

The artifact needs three residency classes. Each is an artifact/materialization concern, not model
mathematics:

| Class | Backing | Users |
|---|---|---|
| `device` (existing) | Uploaded at load | Dense weights |
| `host_pinned` | Read unbuffered (16 overlapped 8 MiB reads) into one committed pageable block whose pages workers zero ahead of the reads, then `cudaHostRegister`ed once, whole (`RegisteredHostBuffer`; `cudaMallocHost` zeroed and locked 63 GiB serially first, 7 s of a 20.9 s load). Never staged through the page cache. Device code reads it at `WeightParent::device`, which differs from the host address under WDDM. | Routed expert banks (NVFP4), embedding, and the vision tower under `--vision-offload` (856 MiB, §19.3.2) |
| `stream` | Never materialized. The Model holds an open file region plus its layout. | N-gram table |

Residency is a property of a Use and the startup options, not of the stored bytes. The same artifact
loads all-device on a hypothetical larger card. The n-gram table is written as its own volume file
(`*.ninfer.ngram`), so it can be placed on a different NVMe drive and a 4 KiB-aligned layout is
guaranteed.

---

## 7. Ownership in Infernix

This is an **explicit product change**: a new model architecture. It follows the existing
boundaries ([engine architecture](engine-architecture.md)).

| Component | Owner | Location (proposed) |
|---|---|---|
| Mathematics, config, binding, block order, MTP alignment, frontend reuse (Qwen3.5 tokenizer, template, vision) | Model | `src/models/qwen4_exp/` |
| Host-pinned expert banks, embedding bank, n-gram file region | Immutable Model data, through a new artifact residency class (§6.3) | `src/artifact/materializer.*` |
| Weight formats `nvfp4_mul` and `fp8_e4m3fn_block128_f32`, layout `nvfp4_expert_rg16_v1`, exact import of ModelOpt multipliers and input scales | Artifact format and layout registries; converter source | `src/artifact/formats.cpp`, `layouts.cpp`; `tools/convert/sources/` |
| Pinned huge-page host arena; unbuffered batched file reads (`ReadOnlyFile`); `upload_pinned`, and the planned `upload_pinned_when` gate and `publish_pinned_word` (§12.3); mapped mailbox and ring primitives; spin-worker pool with barriers; planned `download_pinned` (prefix, only if M0 confirms D2H FIFO stalls) and the `Weight` L2 class `Stream`/`Reuse` (`src/core/weight.h`, Q8 Phase 2) | Core (model-independent physical and transfer primitives) | `src/core/arena.*` (pinned buffers), `read_only_file*`, `device.*`; as built the mailbox and spin-worker team live in the Op (`src/ops/offloaded_sparse_moe/cpu/miss_service.*`, `expert_team.*`) |
| Frame pool, residency table, staging, loans and frame leases (§9.2), epochs; transfer agent (policy, DMA, residency writes); n-gram row cache, reads and gate state (`NgramVolume`, §12.3); prefix-cache binding (§19.3.1); vision window and lane RoPE state (§19.3.2); CPU-engine lifetime | **Program** (mutable state, placement, agents; all allocated at startup) | `src/models/qwen4_exp/program/` |
| Cache policy algorithms (LFRU, shadow replay) | Program, behind a policy interface conformance-tested against `tools/expert_cache_replay` | `src/models/qwen4_exp/program/expert_cache/` |
| Router + top-k + residency classification + prediction; narrow-route expert kernels (GPU and CPU); wide-route grouped GEMM; combine; the MTP drafter's masked routing | **Op family** `offloaded_sparse_moe` (closed contract: output = MoE(x) independent of residency; residency, agents and mailboxes are execution resources). The drafter's routing mask is a semantic input, because it changes the result. | `include/infernix/ops/offloaded_sparse_moe.h`, `src/ops/offloaded_sparse_moe/{route,gpu,cpu,wide}/` |
| HC mixer and HC inject | Op `hyper_connection` | `include/infernix/ops/hyper_connection.h` |
| QSA prep, index-key pooling, block select, sparse attention over every KV profile | Ops `qsa_prep`, `qsa_select`; extended `softmax_attention` consumer with the existing VQ2/K4V2 window read rule | `include/infernix/ops/qsa.h` |
| PLE gather, projections, gate and dilated conv (stateful) | Op `ple_ngram_injection` | `include/infernix/ops/ple.h` |
| N-gram row ids | Model frontend `NgramHash::row_ids`, called by the Program on the host (§12.3); exact integer oracle in `infernix_qwen4_exp_ngram_hash_test` | `src/models/qwen4_exp/frontend/ngram_hash.*` |
| Canonical routed-expert arithmetic: A4 quantizer, E2M1/E4M3 integer decode, int64 block accumulation, `exp_c` (one header compiled for CPU and GPU) | Op common code | `src/ops/common/canonical_math.h` |
| Draft-length policy with miss cost; joint policy at B ≥ 2 (concurrency S7) | Program | `program/program_impl.h` (`choose_draft_length`); planned `src/models/qwen4_exp/program/draft_policy.{h,cpp}` |
| Calibration | Runtime (profile schema), Core (probes), product (`infernix-calibrate`) | §14.7 |

Residency is an **execution resource** for `offloaded_sparse_moe`, like a stream or a workspace.
By the [Op rules](op-development.md#2-op-admission-and-semantic-boundary), it may choose the
implementation but must not change the result. That is the basis for the placement-invariance
requirement (§16.2).

Placement invariance is a deliberate exception to the rule that a contract must not freeze bitwise
equality ([op development §3](op-development.md#3-contract-headers)). Residency changes every
round, so without it the output would depend on cache history, prefetch timing and CPU load. The
exception covers only the arithmetic of the routed-expert narrow route (§16.2). Every other Op
keeps the ordinary oracle-and-tolerance contract.

If option H1 (megakernel) is adopted, Ops expose device-callable **tile functions** next to their
host entries. Each tile function is qualified through the same contract. The Model owns the
instruction sequence, exactly as it owns the call order today. No Op becomes model-labelled.

---

## 8. Decode pipeline

This section specifies a decode round at the level an implementer needs: agents, the host-device
protocol, every kernel with its budget, the router, expert and arbitration algorithms, and QSA
over every KV profile.

### 8.1 Execution agents

| Agent | Thread / stream | Role |
|---|---|---|
| **Compute stream** | CUDA stream owned by the Program | Runs the captured decode graph for each exact (batch B, window T). Contains every GPU kernel of a round. |
| **Transfer agent** | 1 pinned host thread + 1 CUDA copy stream | Owns all host→device expert traffic and the cache policy: prefetch, on-demand miss DMA, promotions, evictions, epoch retirement, residency-table updates (§9.4). **A GPU kernel cannot start a copy-engine DMA**, so every DMA in this design is issued by this thread. |
| **CPU expert engine** | N_w pinned spin workers | Computes CPU-served misses (§10), and warms predicted misses into its caches (§10.4) |
| **Engine worker** | Existing | Launches graphs and commits rounds. It also hashes and reads the n-gram rows (§12.3; planned: while it waits for the GPU); a dedicated NVMe agent was rejected. It does no expert planning. |

The default core budget is: engine worker 1, transfer agent 1, and CPU workers = physical
P-cores − 2 (as built: six workers, the measured optimum, §19.2). All agents spin only while a
request is active, and park on a futex when the Engine is idle.

### 8.2 Host–device protocol

All GPU↔host signalling uses **mapped pinned memory** (`cudaHostAlloc(..., cudaHostAllocMapped)`).
Each exchange has the same structure:

- a producer writes the payload;
- then it writes a 32-bit **sequence word** with release semantics;
- the consumer spins on the sequence word with acquire loads, then reads the payload.

| Region | Location | Producer → consumer | Content |
|---|---|---|---|
| `x_host[L][T][2560]` BF16 | host, mapped | GPU (HC_mlp epilogue) → CPU workers | FFN input of each layer, **written every round** before routing is known. That is 5 KiB per layer at T=1 and keeps x off the miss critical path. The CPU quantizes it to A4 with each missed expert's own scale (§10.3). |
| `miss_req[L]` | host, mapped | GPU (router kernel) → CPU workers | `{round, n_miss, expert_id[], column_mask[]}` + `seq` |
| `miss_out[L][slot][T][2560]` BF16 | host, mapped | CPU → GPU (K7b) | Expert outputs y_e, then `done_seq` |
| `prefetch_req[L]` | host, mapped | GPU (router kernel) → transfer agent | Up to 16 `{expert_id, predicted weight}` for layer l+1 (and l+2 if enabled) + `seq` |
| `route_log[round][L][T×10]` u16 | host, mapped ring | GPU (router kernel) → transfer agent | Every routed id. Used for LFRU accounting and shadow replay. |
| `residency[49×512]` u32 | **device** | transfer agent (via `cuStreamWriteValue32` on the copy stream) → GPU (router kernel) | Expert state and frame (§9.4) |
| `land_seq[frame]` u32 | device | copy stream (`cuStreamWriteValue32` after the DMA) → GPU | Landing ticket for in-flight DMAs |
| `round_started`, `round_done` u64 | host, mapped | GPU (first and last kernel of each round) → all agents | Safe frame reuse (§9.5) and the agents' watchdog |
| `agent_error` u32 | host, mapped | GPU (bounded-spin timeout) → engine worker | Device-side timeout report |
| gate ready word u32 | host, pinned | engine worker (`publish_pinned_word`) → GPU (gate kernel) | Strictly increasing round value; the rows follow in the io landing area (planned, n-gram S2, §12.3) |

**Ordering rules.**

- **Device → host.** The writer thread stores the payload, executes `fence.sc.sys`
  (`__threadfence_system()`), then `st.release.sys` of `seq`.
- **Payload written by an earlier kernel.** Payloads written by an earlier kernel of the round
  (`x_host` by K5b) are fenced with `fence.sc.sys` by each writing thread before that kernel ends.
  The later publisher's release then covers them.
- **Host → device.** x86 store order is total. The CPU writes the payload, then `done_seq` with a
  release store. The GPU reads `done_seq` with `ld.acquire.sys` and only then reads the payload.
- **Polling discipline.** Exactly **one thread per CTA** (lane 0 of warp 0) polls a sequence
  word, with `__nanosleep(32-128)` backoff after the first 2 µs. It then releases the CTA through a
  shared-memory flag and `__syncthreads()`. No warp-wide polling over PCIe is allowed. When many
  CTAs wait for the same host word (K7b's row tiles waiting for `done_seq`), one designated CTA
  polls the host word and republishes it to a device-memory flag, which the others poll in L2.
  Exception: each of the n-gram gate's ≤ 16 CTAs polls the pinned ready word itself (no
  co-residency needed; §19.3.4).
- **Bounded spins.** Every device spin has an iteration bound worth about 2 s. On expiry the thread
  writes `agent_error`, skips the computation (outputs are undefined), and the engine worker turns
  that round into an Engine-wide failure ([engine architecture §7.4](engine-architecture.md)). Host
  agents have a 2 s watchdog on `round_done` progress. Exception (planned, n-gram S2, §12.3): the
  gate before `ple_embed` (`upload_pinned_when`) traps after 120 s, above Windows' 60 s disk
  timeout (decided default, 2026-10-04).

### 8.3 Per-layer kernel sequence and time budget

Decode at T=1. Bytes are what each kernel must read from HBM with every expert resident. A
kernel's budget is its bytes at 1.5 TB/s, which is 90% of this card's measured 1,674.5 GB/s
sustained read ([linear benchmark §9](linear-benchmark.md)), plus 0.5 µs for ramp and tail under
PDL chaining, plus any serial phase it contains. Recipe B is listed first, recipe A in brackets.

**GDN layer (36 of 48):**

| # | Kernel (one launch each) | Reads | Budget |
|---|---|---|---|
| K1a | **HC_attn mixer, phase 1.** Grouped offset RMSNorm of R (stream sums of squares come from the previous layer's K7b), z = W_down [324×10240] · norm(R) as split-K partials | 3.3 MB [6.6] | 2.7 µs [4.9] |
| K1b | **HC_attn mixer, phase 2.** Every CTA reduces the partials in fixed order (13 KB from L2), computes the gates, its rows of W_up [10240×320], and the collapse to x [2560] | 3.3 MB [6.6] | 2.7 µs [4.9] |
| K2 | **GDN input projection.** [16,384×2560] (q,k,v,z) + BF16 [96×2560] (a,b). Row-local epilogue only: causal conv (state update) + SiLU on q,k,v; β = sigmoid(b), g = −exp(A_log)·softplus(a + dt_bias). Any row tiling, so the grid fills all SMs. In layer 1 only, the PLE kernel (§12.3) runs before K1a. | 42.4 MB [84.4] | 28.8 µs [56.8] |
| K3 | **Gated delta recurrence.** One CTA per (value head, 32-row slice of the 128×128 FP32 state): 192 CTAs. The rows of a head's state update independently given k. The prologue applies per-head L2 norm to q and k (and the 1/√128 q scale), recomputed per CTA at negligible cost. Outputs o (unnormalized). | 6.3 MB | 4.7 µs |
| K4 | **out_proj** [2560×6144]. Prologue: per-head sigmoid-gated RMSNorm of o with z (each CTA recomputes the 48 head norms; 6,144 values). Epilogue: HC inject `R += inject_attn ⊗ y` + partial stream sums of squares for K5a. | 15.7 MB [31.5] | 11.0 µs [21.5] |
| K5a, K5b | **HC_mlp mixer** (as K1a, K1b). K5b's epilogue writes x to `x_host[l]` (§8.2). | 6.6 MB [13.2] | 5.4 µs [9.8] |
| K6 | **Router** (§8.4): routers l and l+1 (BF16), top-10, residency, job list, mailbox, prefetch requests, route log. Router l comes from L2 (§8.4). Serial phase ~2 µs. | 2.6 MB | 4.5 µs |
| K7a | **Experts, gate/up** (§8.5): 20 slices per routed job, exact W4A4, SwiGLU, A4(h); shared expert's gate/up | 21.7 MB [25.0] | 15.0 µs [17.2] |
| K7b | **Experts, down and combine** (§8.5): row tiles over all jobs, combine with any CPU outputs, HC inject `R += inject_mlp ⊗ y_moe`, stream sums of squares for the next K1a. Waits on the CPU only if this layer has CPU misses. | 10.9 MB [12.5] | 8.7 µs [9.8] |
| | **Layer total** | **113 MB [189]** | **≈ 84 µs [134]** |

**QSA layer (12 of 48).** K2-K4 become five kernels:

- K2q: QKVG [13,312×2560] plus the BF16 index projection [640×2560], projection only;
- K2b: QSA prep:
  - Q/K offset-RMSNorm and partial MRoPE;
  - K/V append in the configured KV profile (§8.10);
  - raw index-key append;
  - pooled index-key completion when a 4-token block closes;
- K3a: block scores over the pooled index keys + exact top-512 radix select;
- K3b: sparse attention over ≤ 2,051 selected tokens (GQA 24/2, D=256) + sigmoid output gate;
- K4: `o_proj` [2560×6144] with the HC inject epilogue.

At a 4K context that is 107 MB [174] and ≈ 87 µs [132] per layer. A GDN layer has 10 launches and
a QSA layer 12: 507 launches per token, including PLE, head and sampling.

**Per token:**

| Item | Recipe B | Recipe A |
|---|---:|---:|
| 36 GDN layers | 3.01 ms | 4.82 ms |
| 12 QSA layers, 4K context | 1.04 ms | 1.58 ms |
| `lm_head` (0.64 GB 8-bit / 1.27 GB BF16) + sampling | 0.45 ms | 0.87 ms |
| PLE (§12.3) | 0.02 ms | 0.04 ms |
| **t_gpu (T=1, all hits)** | **≈ 4.5 ms (≈ 222 tok/s ceiling)** | **≈ 7.3 ms (≈ 137 tok/s)** |

The totals are 6.0 GB and 10.2 GB per token. Their pure-bandwidth floors at the sustained read
rate are 3.6 ms and 6.1 ms.

**Rules that every decode kernel follows.** This is the "HBM never idle" baseline.

1. **PDL with pre-dependency weight prefetch.**
   - Kernels launch as programmatic dependents ([op development](op-development.md#programmatic-dependent-launch))
     at ≤ 2 resident CTAs per SM, where a chained pair measured +0 µs (4 CTAs per SM: +2 µs).
   - Before `griddepcontrol.wait`, each CTA issues L2 prefetches (`cp.async.bulk.prefetch.L2`) or
     shared-memory copies of the **static weights it will read**. The next kernel's weight stream
     therefore starts while the previous kernel drains.
   - Every streaming kernel keeps ≥ 32 KB of loads in flight per SM, with TMA bulk copies or deep
     `cp.async` rings. At 1 CTA per SM, a shallow pipeline measured 1.6× slower on this card.
   - The router (K6) prefetches only its own weights. Expert weights depend on routing, so K6's
     last CTA prefetches the GPU jobs' first units as soon as routing is known.
2. **Kernel boundaries instead of grid barriers.** A cooperative grid barrier measured 1.4-2.0 µs
   on this card, more than a PDL boundary. The HC mixers are therefore two kernels each. If M6 finds
   that PDL overlap does not hold in the full graph, they fall back to one cooperative kernel each,
   which is the structure the conservative column of §4.2 assumes.
3. **Stall-time warming.** A kernel that waits on the host (K7b for CPU misses) executes
   `griddepcontrol.launch_dependents` before it waits. The next kernel's CTAs then start on the
   free slots and prefetch their weights, and the waiting CTAs prefetch the following kernel's
   weights into L2 (bounded to half of L2). The stall becomes progress on the next layer. The
   n-gram gate before `ple_embed` (§12.3) is a plain launch: a late row read stalls layer 1
   without warming.
4. **Deterministic reductions.** Routed-expert sums are exact int64 (§16.2), so their order is
   free. Every floating-point split-K or cross-CTA reduction writes partials to workspace and sums
   them in a fixed order. No floating-point atomics are used, and results are run-to-run
   deterministic.
5. **Cache policy.**
   - Weight streams use the L2 evict-first policy (an `L2::cache_hint` policy from `createpolicy`,
     or the eviction-priority qualifier where the PTX ISA allows it for the load width).
   - Activations, state, the residency table and the next layer's router use evict-last. K6(l)
     loads router l+1 that way, so K6(l+1) reads it from L2. That saves 126 MB per token.
6. **Grid sizing and waiting.**
   - Grids are multiples of 170 SMs × resident CTAs per SM, which avoids wave quantization.
   - A CTA waits only on the host or on the copy engine, never on another CTA of its own grid,
     except through the last-arriver pattern. Partial co-residency can therefore never deadlock.

The budget is a planning tool, not a promise. M6 publishes measured per-kernel times against it,
and any kernel more than 30% over budget gets an ncu investigation before the next milestone.

**As built and planned (2026-10-04).** As built, each HC mixer is five launches plus an inject, the
shared expert adds split and SiLU kernels, and dense Q8 takes 3.09 ms per main forward at T = 1
(measured). The Q8 track (§19.3.3) moves toward the sequence:

- **K1a/K1b** become Op `hyper_connection_mix` (Phase 1b): FP32 split-K partials, one K slice per
  stream × 41 groups of 8 rows, summed by K1b in the fixed order s = 0..3. **Deviation:** K1a takes
  each stream's sum of squares from its own slice, not the previous epilogue, so the Ops stay closed
  (≤ 0.3 µs exposed, est.). Est. 7.6-8.9 µs per mixer, against ~11.4-12.9 µs for the five-launch
  mixer after Phase 1a (budget 5.4 µs for K1a + K1b, recipe B): −0.34 to −0.42 ms per forward.
- **K4/K7b injects:** `hyper_connection_linear_inject` (epilogue of the [2560, 6144] projections)
  and an inject form of `moe_combine`, still its own kernel; no stream sums. The shared expert's
  gate/up and SiLU·up become one `linear_swiglu` call.
- **Numerics.** Each fusion drops BF16 intermediates (Rn, z, m, u, y before inject, gate, up), as
  [op development §6.1](op-development.md) requires: more precise than HEAD and upstream.
- **Rules.** 1: PDL weights-first for the Q8 kernel, fused Ops and router. 3: extra
  `cpu_wait_kernel` CTAs prefetch the next layer's mixer and ≤ 16 MB of its input projection into
  L2 during the wait (18.6 µs per layer, measured). 5, deviation: evict-first only for main-forward
  weights, as the drafter re-reads its own ([13952, 2560]: 16.6 µs from L2, 29.6 from DRAM,
  measured). 6, deviation: 160-CTA grids leave 10 SMs idle (< 0.1 µs to gain, est.).

**Decided (default, 2026-10-04):** the Phase 1b rounding changes, this mixer and statistic, that PDL
scope. Alternatives: a three-kernel mixer, the producer-epilogue statistic, PDL everywhere.

### 8.4 Router kernel (K6)

**Inputs:**

- x_l [T, 2560] BF16;
- router W_l (from L2) and the next layer's router W_{l+1} (BF16, 512 × 2560 each), plus the
  shared-expert gate row;
- the residency table;
- the cost-model constants: expected CPU service time, and the expected landing time of each
  `LOADING` frame, published by the transfer agent.

**Phase 1** (all CTAs, ~128): warp-per-row GEMV of the 1,025 rows. Logits go to workspace.

**Phase 2** (last CTA, selected by an atomic arrival counter):

1. **Routing.** For each column t: top-10 of the 512 logits, with the lower id winning exact ties;
   route weights = softmax over the 10 selected logits (`norm_topk_prob`); shared-expert gate =
   sigmoid of its logit.
2. **Union.** The distinct experts over the T columns, with per-expert column masks and n_e.
3. **Classification** of each distinct expert from its residency entry:
   - a GPU job (`READY`);
   - a gated GPU job (`LOADING`, with landing predicted sooner);
   - a CPU miss;
   - or an on-demand DMA (§8.6).

   Experts with n_e > 8 are always GPU jobs. If such an expert is not resident, it becomes an
   on-demand DMA with a gated job.
4. **Outputs.**
   - The job list for K7a and K7b: frame address, gate ticket, column mask, route weights, the
     matrices' α and input scales.
   - `miss_req[l]`, written with release **only if** there is a CPU miss.
   - The route log.
5. **Prediction.** Top-k′ of W_{l+1}·x_l per column (k′ = 16 by default), unioned. Non-resident,
   non-loading experts go to `prefetch_req[l]` in order of predicted weight.
6. **Expert weight prefetch.** L2 prefetch of the GPU jobs' first units, so K7a starts on warm
   lines.

Phase 2 is serial but small: about 2 µs at T=1 and about 4 µs at T=8.

### 8.5 Expert kernels (K7a, K7b)

Every routed expert on the narrow route (n_e ≤ 8) is computed with the canonical arithmetic of
§16.2: A4 activations quantized with the expert's own `input_scale`, exact integer block
products, int64 accumulation, and BF16 outputs. The same arithmetic runs on the CPU (§10), so a
routed expert's output does not depend on where it was computed.

**K7a: gate/up, SwiGLU and A4(h).** A work item is (job, slice σ) for a 32-intermediate slice:
20 items per routed job, plus the shared expert's slices. That is 220 items at T=1. Persistent
CTAs pull items from an atomic counter in job-rank order. A routed item:

1. waits on its job's `land_seq` ticket if the job is gated (one thread polls device memory);
2. streams its 92,160-byte run (row groups 4σ…4σ+3, §6.2) through a multi-stage shared-memory
   ring;
3. meanwhile quantizes x_t to A4 with the job's gate/up `input_scale`. Every item of the job
   recomputes this from the 5 KiB x in L2, hidden under the stream's start;
4. forms each block's exact product P (int32) with `dp4a` for n ≤ 2, or `mma.sync.m16n8k32.s8`
   with block-separated columns for n ≥ 3 (M3 fixes the crossover), and accumulates
   P · Ŝ_w · Ŝ_a into int64 with `mad.wide.s32`;
5. converts S to BF16 y with the matrix's α (§16.2) for its 32 gate and 32 up rows per column;
6. forms h = bf16(silu(y_gate) · y_up), then quantizes its two complete 16-blocks of h to A4 with
   the job's down `input_scale`, and writes them to workspace: 18 bytes per column.

Shared-expert items run a dense BF16 (recipe A) or 8-bit (recipe B) GEMV with BF16 activations
and write h_shared in BF16: NVIDIA keeps the shared expert in BF16, so it has no A4 step.

**K7b: down, combine and HC inject.** A work item is a tile of 16 output rows: 160 items, one CTA
per SM.

1. Before `griddepcontrol.wait`, the CTA copies its tile's down weights for every `READY` job into
   shared memory with TMA bulk copies: one 5,760-byte run per routed job, plus 16 shared-expert rows.
   That is 68 KB [78 KB] per CTA, all in flight at once, so one CTA per SM is enough. Gated jobs are
   read only after their ticket.
2. After the wait, it reads every job's A4 h (360 bytes per job and column) and h_shared.
3. It computes each routed job's 16 output rows exactly (§16.2) as BF16 y_e, and the shared
   expert's rows.
4. If the layer has CPU-served experts, it waits for `done_seq` (one poller CTA republishes it to
   a device flag, §8.2) and reads the CPU's BF16 y_e rows for its tile zero-copy: 32 bytes per
   expert and column.
5. **Combine** in fixed rank order:
   - y_moe[r] = Σ_{i=1..10} w_i · y_{e_i}[r], in FP32;
   - plus sigmoid(shared gate) · y_shared[r];
   - rounded to BF16 at the model's semantic boundary.
6. It applies `R += inject_mlp ⊗ y_moe` for its rows and writes per-stream partial sums of squares
   for the next layer's K1a.

Every input of the combine is the same bits wherever its expert ran, and the order is fixed by
rank. The combine is therefore placement-invariant.

**Wide route (n_e > 8).** Used by prefill and by wide verify batches. The job's K7a items and K7b
tiles use the block-scaled tensor-core MMA of prefill (§13): `mma … kind::mxf4nvf4.block_scale`
with the same canonical A4 activations. Its accumulation order belongs to the tensor core, so it is
qualified against the FP64 oracle, not bit-exactly. The route depends only on n_e, which does not
depend on placement, and the CPU never serves a wide-route expert.

### 8.6 Miss arbitration

For each layer, the router's Phase 2 assigns every non-resident narrow-route expert to the
cheapest predicted service. The inputs are the cost-model coefficients the calibration fitted
(§14), refreshed online by the transfer agent:

| Service | Predicted completion | When chosen |
|---|---|---|
| CPU, warmed (H2) | t_handshake + n_cpu · t_warm | Expert is in the CPU's warmed set |
| CPU, cold | t_handshake + Σ bytes / BW_cpu(n_cpu) | Default |
| Wait for in-flight DMA | landing time published by the agent | `LOADING` and earlier than the CPU option |
| On-demand DMA + GPU | t_req + bytes / BW_pcie, behind the copy engine's queue | Only when the CPU queue for this layer exceeds the DMA completion time (several cold misses in one layer). The single host-to-device copy engine is shared with prefetch, so the agent's published queue length enters the estimate. |
| Split CPU + DMA (H3) | max of the two halves | Option H3 |

The objective is the earliest time at which **all** of the layer's experts are available,
because the combine waits for the last one. Ties go to the CPU, which needs no copy.
Every decision and its outcome (predicted vs actual completion) is logged into per-layer
histograms. The online model corrects its coefficients from those, and M5 reports the prediction
error.

### 8.7 Lookahead prefetch

Between the router of layer l and the router of layer l+1 lies one full layer of GPU work
(≈ 84 µs with recipe B, ≈ 134 µs with recipe A; §8.3). During that window, host DRAM and PCIe
would otherwise be idle.

1. **Prediction.** K6 of layer l evaluates layer l+1's router on x_l (§8.4). Its top-k′ per column
   is unioned, with k′ = 16 by default (M1 fixes it). The non-resident, non-loading experts are
   published to `prefetch_req[l]` in order of predicted weight.
2. **Transfer.** The transfer agent (§9.4) issues one contiguous DMA per predicted expert into a
   staging frame. It stops when the next DMA could not land before layer l+1's K6, using the
   calibrated DMA completion model. At ~50 GB/s that is about 1.5 experts per layer with recipe B
   and 2.6 with recipe A.
3. **Warm list.** Predicted experts beyond the DMA budget go to the CPU engine's warm list (§10.4,
   option H2).
4. **Use.** If layer l+1 routes to a staged expert, it is a GPU job. A still-loading expert is
   handled by the arbiter (§8.6). An expert staged but not used stays in its staging frame until it
   is overwritten, or until the policy promotes it by changing the frame's role, with no copy
   (§9.3).

Prefetch is a **latency** tool. Replay shows that perfectly predicted next-layer experts change the
miss *count* by < 0.1% (§9.3). Its value is turning critical-path misses into staged hits. Each
layer's prediction recall and useful-byte ratio (staged bytes used ÷ staged bytes) are published.
Prefetch is disabled automatically for a layer whose useful-byte ratio stays below 0.3, which saves
DRAM bandwidth for the CPU engine. Depth-2 prediction is option H4.

### 8.8 Batch and verify windows

With T columns (C ≤ 8 lanes and/or a K+1 verify window), the router computes the union of experts
with per-expert column masks. Each distinct expert is fetched or computed **once** for all its
columns:

- experts with n_e ≤ 8 use the narrow route (CPU-servable);
- experts with n_e > 8 use the wide route (GPU only).

CPU cost scales with distinct misses, not columns, because the CPU narrow kernel is DRAM- or
L2-bound up to n = 8.

### 8.9 Round boundary

With the current engine, the round boundary is a host synchronization:
[`ProgramImpl` decode](../../src/models/qwen3_5/program/decode.cpp) builds host ingress, launches the
graph, calls `device.synchronize()`, then commits. This design keeps that contract in the baseline,
but moves work off it:

1. The decode graph ends with these steps:
   - sampling;
   - the token written to the mapped egress slot;
   - `round_done` (§8.2).
2. Ingress: the embedding row is host-resident. The engine worker hashes the n-gram row ids and
   reads the rows (§12.3). As built, the rows go up with the io prefix before the launch. Planned
   (n-gram S2): verify rounds read them after the launch, behind a gate before `ple_embed`.

The engine's per-round host time is measured in M6. If the GPU idles more than 3% of a token
between rounds, option H6 (chained rounds) is built.

### 8.10 QSA over every KV profile

QSA reads its cache in two separate ways:

- **Selection (K3a)** reads only the pooled index keys: one 128-wide key per 4-token block,
  materialized by K2b when the block completes, RMS-normed and RoPE'd as the reference defines.
  - They form a separate BF16 plane per attention layer, 64 B per token. They are activations of a
    BF16 layer, so they stay BF16 under the fidelity requirement.
  - The plane is **independent of the KV profile.** Changing `--kv-dtype` therefore never changes
    which tokens a query attends to; it changes only how their keys and values are represented.
- **Attention (K3b)** reads the K and V rows of the ≤ 2,051 selected tokens from the paged KV
  store, in the profile the Engine selected.

Every Infernix profile (`--kv-dtype`) is supported, and the Main Text and MTP layers use the same
profile. **`int8` at a 262,144-token context is the primary configuration** (§18): it is what users
of this machine run, and it is optimized first. The other profiles follow in this order: `fp8` and
`k8v4`, then `bf16`, then `nvfp4`, then `vq2` and `k4v2`.

| Profile | K+V per token and KV head | Per context token (13 layers + index keys) | 32K / 128K / 256K context |
|---|---:|---:|---|
| `bf16` (BF16 K, FP16 V) | 1,024 B | 27.5 KB | 0.90 / 3.60 / 7.20 GB |
| **`int8` (INT8-G64), primary** | 528 B | 14.6 KB | 0.48 / 1.91 / 3.82 GB |
| `fp8` (E4M3FN-row256) | 516 B | 14.2 KB | 0.47 / 1.87 / 3.74 GB |
| `nvfp4` (NVFP4-G16) | 288 B | 8.3 KB | 0.27 / 1.09 / 2.18 GB |
| `k8v4` (FP8 K, NVFP4 V) | 402 B | 11.3 KB | 0.37 / 1.48 / 2.96 GB |
| `k4v2` (4-bit Lloyd-Max K, VQ2 V) | 196 B | 5.9 KB | 0.19 / 0.78 / 1.55 GB, plus a 15 MB exact window per sequence |
| `vq2` | 132 B | 4.3 KB | 0.14 / 0.56 / 1.12 GB, plus a 15 MB exact window per sequence |

How each part handles the profiles:

- **Append (K2b)** writes K/V through the existing per-profile append routines, including the
  VQ2/K4V2 exact-window slots ([paged KV §9.3](paged-kv-cache.md#93-vq2k4v2-exact-recent-key-window)).
  Stored codes are therefore identical to the dense-attention models'.
- **Decode attention (K3b).**
  - It walks the selected 4-token blocks through the block table and loads each block with a
    **per-profile tile loader**: BF16/FP16 direct; INT8-G64 and FP8-row codes times their scales;
    E2M1 times E4M3; K8V4's mixed pair; or VQ2/K4V2 codes in the Hadamard-rotated domain, with q
    rotated once and the output rotated back.
  - The loaders reuse the dequantization routines of the existing `softmax_attention` consumers.
    Only the gather and the online softmax are new, and they are shared by every profile.
  - Split-K over the selected blocks merges partials in fixed order.
- **Exact window.** For VQ2/K4V2, a selected key at position j is read from the exact INT8-G64
  window when j < 64 or j ≥ q − 768 and its slot tag matches, and from codes otherwise. That is the
  existing position-only rule, so the representation does not depend on how the context was
  chunked.
- **Prefill.**
  - While the visible context is ≤ 2,051 tokens, every token is selected, and QSA equals dense
    causal attention. The existing prompt kernels serve it, with their P×V options
    (`--prefill-8bit-pv`) unchanged.
  - Beyond that, the prompt kernel processes a tile of queries against the union of their selected
    blocks with a per-query mask. It uses the same tile loaders.
- **Cost.** K3b reads 2.2 MB per layer per token with `int8`, 4.2 MB with `bf16` and 0.54 MB with
  `vq2`. The decode-time difference across profiles is ≤ 0.04 ms per token. The profiles' real
  effect is **VRAM**: at 256K, `int8` KV is 3.8 GB, 1,381 frames, before the host tier (§9.6) takes
  most of it off the device (§15.1).

Each profile is qualified against the FP64 attention oracle with that profile's existing decode
criterion. QSA adds two requirements: selection equality across profiles on near-tie-free
fixtures, and dense equivalence while n ≤ 2,051 (§16.1).

---

## 9. Expert cache and the shared VRAM frame pool

### 9.1 Frames

After dense weights and fixed state, device memory is **one allocation** (the frame pool) divided
into frames of exactly 2,764,800 B = 675 × 4 KiB, one expert record each. Frame i is at
`base + i × 2,764,800`. Roles change at runtime through the Program's frame table, never through
allocation:

| Role | Contents |
|---|---|
| `expert` | A cached expert. Always **clean**: the host bank holds an identical copy. |
| `staging` | A prefetch or on-demand DMA target (§8.7). It becomes `expert` by a role change. |
| `kv` | KV pages. The paged KV store is backed by frames, and a KV page size that divides 2,764,800 B is chosen per KV format: 4 KiB multiples whose count divides 675 = 3³ × 5². |
| `loan` | A frame inside a KV or prefill reservation that currently holds a clean expert (§9.2) |
| `prefill` | Prefill staging and workspace (§13) |
| `vision` | Lent to a vision request: its output handoff (5,120 B per image token, held until its last prefill chunk is enqueued) and, only when the encode window does not fit `work_` in one step, the window's weight staging and activations (placement L, §19.3.2). Offloaded tower weights stay in pinned host RAM. |

The frame count is a startup decision (§15.1). In the primary configuration (`int8` KV, 262K
context) the pool has 8,153 frames with recipe A and 9,675 with recipe B, of which the KV
reservation can take up to 1,381 as the context fills. Frames are 4 KiB-aligned slices of one
2 MiB-page allocation, so a frame spans 2-3 GPU pages. The TLB reach of 2 MiB pages covers the
whole pool.

**MTP experts do not use frames.** An FP8 MTP expert is 4,916,400 B, 1.78 frames. The drafter's
experts therefore live in a separate fixed pool of slots (128 by default, 0.59 GiB, sized by
calibration), with their own LFRU in the transfer agent. The drafter never waits for them (§11.2).

### 9.2 Lending reserved memory, the key to sharing VRAM with KV

Infernix's admission invariant requires an Active request to hold its **complete** KV growth
reservation ([engine architecture §1](engine-architecture.md#1-产品执行模型)). In other engines, a
long-context reservation removes expert slots even while those KV pages are empty. ninfer-ext lost
19% at C=8 that way.

Because every cached expert is clean, a frame that is reserved for KV but not yet written can hold
an expert **on loan**:

- **Reclaim is instantaneous and cannot fail.** Clear the residency entry and switch the frame's
  role. There is no write-back and no allocation, so the reservation guarantee is preserved
  exactly.
- **Reclaim is requested ahead of need.** The Program asks the transfer agent to reclaim a loaned
  frame two rounds (2 × max T tokens) before a lane's KV frontier can reach it. The evicted
  expert's retire epoch (§9.5) has then passed before the first KV write. Graphs address KV and
  experts through page and frame tables, so the graphs stay valid.
- **The same mechanism covers the rest:** prefill arenas (§13) and vision encode windows borrow
  frames and return them, the borrowed experts being re-admitted lazily by the policy instead of
  re-copied eagerly. Strata re-copies them eagerly after every prompt, which costs short prompts
  up to a third of their speed. Vision weights stay in pinned host RAM (§19.3.2).
- **Status (2026-10-04).** Built: no loans; KV (whole extent), io, `work_`, MTP buffers and 64
  staging slots come before the frames, which take free VRAM less 384 MiB (§19.2). Planned by the
  vision track (§19.3.2): `ExpertResidency::lend` / `give_back` and a `kLoaned` frame state.
  - `choose_run` takes the contiguous run with the lowest summed LFRU score, never loaned frames,
    avoiding in-flight load targets. `lend` evicts the run's experts (no write-back) and shrinks
    the policy capacity; `give_back` restores it and drains the deferred-load queue.
  - Streams wait only on promotion batches that target the run (no queueing behind up to ~77 ms of
    promotions); `give_back` needs compute ordered after every access; graphs read frames through
    tables, so a loan invalidates none. A per-load serial closes an ABA hole in load publication
    that exists today and that LIFO reuse of returned frames would make likely.
  - First users: a vision output handoff (2 frames per 1K image tokens) and windows that do not
    fit `work_` in one step. Decided (default, 2026-10-04): vision never borrows the last 25 %.
  - Not planned yet: KV and prefill-arena loans. The prefix cache takes no frames (§19.3.1); under
    KV loans its Host-backed blocks yield, Free lanes' latest paths and spare COW pages do not.

Result: memory that a 256K-capable configuration reserves for KV keeps serving experts until the
context actually grows into it. In the primary configuration that is 3.8 GB, 1,381 frames, of
`int8` KV (7.2 GB with `bf16`, 1.1 GB with `vq2`; §8.10).

### 9.3 Replacement and admission policy

#### Evidence: is LRU the best policy? No.

Two independent replays of this model's routing traces were run for this design. The
continuous-session replay is [`tools/expert_cache_replay`](../../tools/expert_cache_replay/replay.py). Both used the
FreeToken community repo's `warmup`, `A` and `B` traces (383 decode tokens each, 48 layers, top-10,
batch 1). Both used event-atomic semantics: a layer's 10 experts must be resident together, so the
current group is never evicted.

- **Continuous session.** The cache is warmed on `warmup`, then A and B are scored as one session,
  at five capacities.
- **Per-request reset at 4,008 slots.** Each request starts from the recorded engine cache state.
  This reproduces the repo's exact LRU copy plans and Belady counts.

The table shows misses per token, with the change against LRU in brackets. Promotions are the
copies into VRAM.

| Policy (continuous session) | 16.3% (4,008) | 25.8% (6,347) | 32.6% (8,000) | 41.5% (10,200) | 48.8% (12,000) |
|---|---:|---:|---:|---:|---:|
| LRU (global) | 104.6 | 49.9 | 30.2 | 14.9 | 8.5 |
| **LFRU** (Zhang 2026: f / (age + 1), global f never reset) | **88.4 (−15%)** | **42.0 (−16%)** | **25.0 (−17%)** | **12.5 (−16%)** | **6.6 (−22%)** |
| ARC | 81.1 (−22%) | 40.6 (−19%) | 27.9 (−8%) | 14.2 (−5%) | 7.2 (−15%) |
| S3-FIFO | 94.2 (−10%) | 45.7 (−8%) | 26.6 (−12%) | 12.8 (−14%) | 7.2 (−15%) |
| SLRU (80% protected) | 93.8 (−10%) | 47.0 (−6%) | 27.3 (−10%) | 13.8 (−7%) | 6.1 (−28%) |
| Decayed LFU, half-life 64 tokens | 103.8 (−1%) | 47.4 (−5%) | 26.4 (−13%) | 13.6 (−9%) | 7.6 (−11%) |
| LRU-2 | 105.9 (+1%) | 54.0 (+8%) | 32.1 (+6%) | 17.0 (+14%) | 7.0 (−18%) |
| W-TinyLFU, 1% window (this implementation) | 149.8 (+43%) | 76.4 (+53%) | 41.6 (+38%) | 15.3 (+3%) | 7.6 (−11%) |
| SIEVE | 172.4 (+65%) | 98.3 (+97%) | 65.4 (+117%) | 13.3 (−11%) | 7.0 (−18%) |
| LRU, equal per-layer quotas | 107.8 (+3%) | 57.6 (+15%) | 36.8 (+22%) | 20.0 (+34%) | 11.2 (+32%) |
| **Strata's adaptive exchange** (CPU-served misses, profile seed) | 119.9 (+15%), promotions 16.9 | 62.5 (+25%), 9.3 | 37.0 (+23%), 5.1 | 16.4 (+10%), **2.0** | 8.6 (+1%), 1.0 |
| Static profile, hindsight frequency of A+B (not deployable) | 118.7 | 54.9 | 29.6 | 11.1 | 4.1 |
| **Belady MIN** (offline optimum) | 46.0 (−56%) | 20.2 (−60%) | 11.8 (−61%) | 5.6 (−62%) | 3.2 (−62%) |

The primary configuration's own capacities, 32-39% of the experts, were replayed separately
(§4.2). LFRU has 17-18% fewer misses than LRU there too: 14.7 against 17.9 per token at recipe B's
9,652 frames.

The per-request replay at 16.3% gives:

| Policy | warmup | A | B |
|---|---:|---:|---:|
| LFRU | −9.8% | −9.2% | −20.9% |
| S3-FIFO | −0.1% | −0.2% | −18.4% |
| ARC | +5.0% | +5.3% | −3.6% |
| LRU-2 | +10.9% | +12.1% | −11.3% |
| Static warmup profile applied to A / B | — | +9.3% | +82% |
| Belady | −51.7% | −51.6% | −61.5% |

**LFRU is the only policy better than LRU in every replay, method and capacity.** That agrees with
the one rigorous published comparison: Zhang (arXiv 2608.07911) found LFRU the best causal policy in
12 of 13 workloads on Qwen3-30B-A3B.

Further results:

- **Next-layer prediction does not reduce misses.** Protecting perfectly predicted next-layer
  experts changes misses by < 0.1%. Its value is hiding latency, which is how §8.7 uses it.
- **Longer horizons do help when combined with LFRU.** Perfect protection of experts used within
  the next 1 token gives a further −2.5 points; within the next 5 tokens, −8 to −10 points.
- **Admission filtering alone is worth ≈ 0 at batch 1.** Belady with bypass equals Belady with
  forced admission, so the gain lies in choosing victims.
- **Speculation.** Per-layer unions over 2 / 4 / 5-token windows leave misses per accepted token
  unchanged for every policy.

**Caveats.**

- The traces are short, and all three are English technical writing.
- The per-request and continuous replays disagree on ARC and S3-FIFO. On these traces only LFRU is
  robust.
- A replay gain does not guarantee a wall-clock gain. FreeToken's two-hit policy won in replay and
  lost 4.8% live, for unmeasured reasons. Policy bookkeeping must therefore stay off the critical
  path. M5 runs a live A/B.

#### Policy

1. **Victim ranking: LFRU**, one global pool over the 24,576 Text experts. The MTP pool runs the
   same policy separately.
   - The score is `f / (now − last + 1)`. The clock advances once per routed layer call (48 ticks
     per round, plus the MTP layer's ticks in its own pool), which is what the replay of §9.3
     measured. Scores compare as IEEE binary64, and ties evict the lower expert id, so host and
     replay decisions are identical step for step.
   - `f` is a global per-expert count kept for all 24,576 ids, including evicted ones (96 KB, kept by
     the transfer agent on the host).
   - `f` halves exactly every 8,192 ticks (~170 plain rounds; `ExpertResidency::kLfruHalvingPeriod`). It is
     a binary64 fraction, and a halved count below 2^-10 becomes 0. The saved state keeps counts rounded.
     Evidence (2026-10-10, `local/workdirs/strata-forks-review/research/b4_decay.py`; the replay's
     `LFRU(cap, halving)` reproduces the engine decision for decision):
     - The trace was greedy plain decode of 14 mixed requests on the NVIDIA artifact (AIME, code in three
       languages, stories, translation, SQL/CSV, chat, review, logic).
     - Each fold warms on 3 requests and scores the next 5.
     - At 8,000 frames, misses per token against never halving:

       | Fold | Never halving | Halving every 8,192 ticks | LRU |
       |---|---:|---:|---:|
       | 0 | 52.99 | 48.10 (−9.2 %) | 50.88 |
       | 5 | 39.01 | 34.97 (−10.4 %) | 36.89 |
       | 10 | 44.72 | 39.51 (−11.7 %) | 42.90 |

       Fold 5's 8,192-tick figure comes from the integer-halving run; the fractional run was folds 0 and 10.
     - At 9,600 frames, halving every 4,096 ticks gives −7.2 %.
     - On this mixed trace, never-halved LFRU loses to plain LRU by 4-5 %: global counts stay stuck on the
       previous topic's experts. Periods of 4,096-16,384 ticks are within 1.5 % of each other.
     - Integer halving (c / 2) is unstable at short periods: at 1,024 ticks it is +12 to +45 % worse,
       because single uses truncate to 0.
     - Fractional halving is smooth: 1,024 ticks gives −6 %.
     - The FreeToken traces above were not re-run with halving (they are not in the tree).
   - No per-layer quotas, because hard partitioning lost 3-34%.
   - Experts of the current and predicted groups are protected.
2. **Admission: promote on miss, within a promotion budget.**
   - Every CPU-served miss is a promotion candidate. The transfer agent (§9.4) copies it into a frame
     only if its LFRU score beats the current victim's and the round's DRAM/PCIe token bucket has
     room. The device only reads residency entries.
   - At recipe B's capacity that is ~15 promotions per token (~8 GB/s at 200 tok/s), so the bucket
     rarely binds.
   - When it does bind (miss storms after a topic switch, or a long context that has shrunk the
     pool), promotion degrades into **lazy, batched promotion ranked by the same score**. This is
     Strata's mechanism, which replay shows makes ~7× fewer copies than LRU. At recipe B's 9,652
     frames, lazy LFRU (half-life 128 rounds, margin 0.5, ≤ 32 per round) had 12% more misses than
     on-demand LFRU, with 56% of its promotions.
3. **Free promotion of staged experts.** Prefetched or on-demand-DMA'd experts already occupy a
   frame. Promoting them is a role change with no copy (§8.7).
4. **Seed and persistence.**
   - The cache starts from the saved LFRU state of the previous run (counts and ranking).
   - Failing that, it starts from a shipped profile built from broad traces.
   - A static profile alone loses badly when the workload differs: +82% misses on B.
5. **Policy telemetry and selection on the host, off the critical path.**
   - The route log (§8.2) carries every routed id (48 × 10 × T × 2 bytes per round) to the host.
   - A low-priority host thread replays them through shadow copies of LRU, LFRU with 2-3 half-life
     settings, and a windowed Belady over the last ~1K tokens.
   - The results are published as metrics: the live policy's gap to LRU and to Belady.
   - If a shadow variant beats the live parameters by > 5% over a long window, the parameters are
     switched. This uses the set-dueling idea of CPU caches, applied to parameters, not policy
     families.
   - This replaces Strata's offline parameter calibration with continuous measurement on the
     user's own workload.
6. **Research option, decided in M1:** protection driven by multi-token prediction.
   - Perfect 1-5-token lookahead is worth 2.5-10 points on top of LFRU. A real predictor only pays
     if it is precise.
   - Zhang's learned next-use regressor made things worse. SeqMoE reports large gains with a
     sequence model (not replicated).
   - M1 evaluates candidate predictors on a public trace set that includes router inputs and hidden
     states (`aswinkumar99/qwen3.8-flash-next-expert-traces`). The candidates are: the main model's
     routers applied to MTP-draft hidden states, and multi-layer gate lookahead.
   - A predictor ships only as **protection** layered over LFRU, never as a replacement ranking.
     That keeps the learning-augmented robustness: with a bad predictor it degrades to LFRU.

### 9.4 Residency state machine and the policy engine

**The device only looks up residency; all policy runs on the host, in the transfer agent.** The
agent receives every routed id through `route_log` (§8.2) one round after the GPU used it. It keeps:

- the LFRU state;
- the frame table;
- the shadow replays (§9.3).

It issues every copy. This keeps the decode kernels free of policy bookkeeping. FreeToken's two-hit
policy, for example, won in replay and lost 4.8% live.

**Residency entry** (`residency[layer × 512 + expert]`, u32, device memory; layer 48 is the MTP
layer, whose entries index MTP pool slots instead of frames):

| Bits | Field |
|---|---|
| 31-30 | state: `ABSENT` = 0, `LOADING` = 1, `READY` = 2 |
| 29-24 | 6-bit generation, incremented on every state change, so a stale ticket cannot be mistaken |
| 23-0 | frame index (≤ 16M) |

`READY` covers both cached and staged experts. The difference (whether the frame counts toward the
policy's resident set or is a staging frame) is policy state, kept only on the host. The device
treats both as "use this frame".

**Transitions.** All transitions are issued by the transfer agent, in this order on its copy stream:

```text
ABSENT  → LOADING(f)  : agent picks frame f (free list), WriteValue(entry, LOADING|gen|f),
                        memcpyAsync(frame f ← host record), WriteValue(land_seq[f], ticket),
                        WriteValue(entry, READY|gen+1|f)
READY(f) → ABSENT     : WriteValue(entry, ABSENT|gen+1) + event; frame f is reused only under the
                        rule of §9.5
LOADING(f) → (cancel) : not allowed; a started load always completes, then may be evicted
```

**Device use of an entry** (in the router kernel, once per selected expert per round):

- `READY(f)`: a GPU job on frame f.
- `LOADING(f)`: the router consults the agent's published expected-landing time. If that is earlier
  than the CPU's expected finish for this layer's misses (both from the cost model, §8.6), the
  expert becomes a GPU job **gated on `land_seq[f]`**. Otherwise it is a CPU miss; the load still
  completes and the expert is used next time.
- `ABSENT`: CPU miss, unless the miss arbiter (§8.6) requests an on-demand DMA.

**Agent loop** (one pinned thread, never blocking on the GPU):

1. **Prefetch requests** for layer l+1 (`prefetch_req[l]`, seq advanced). Take a free staging
   frame, or evict the lowest-score staging expert, and issue the ABSENT → LOADING → READY sequence.
   Stop when the per-layer DMA budget (bytes the copy engine can land before layer l+1's router,
   from calibration) is used.
2. **On-demand misses** requested by the arbiter: same sequence, top priority.
3. **Route log** for completed rounds: update LFRU counts and recency, and run the shadow replays.
4. **Promotions.** For each CPU-served expert of the last round whose LFRU score beats the current
   victim's, within the token bucket (§9.3), set the victim ABSENT, then load the newcomer into a
   free or retired frame.
5. **Retirement.** Frames whose retire epoch ≤ `round_done` return to the free list. An admitted
   expert that found no free frame waits in a FIFO **deferred-load queue**, stays CPU-served, and
   loads as soon as a frame retires; if the policy evicts it first it simply leaves the queue.
6. **Loans.** Process KV-reclaim requests from the Program (§9.2): evict experts from the frames
   being reclaimed, and hand the frames over once their retire epoch passes.

Commands are batched with `cudaMemcpyBatchAsync` when more than one copy is pending. The agent's CPU
cost is budgeted: < 15% of one core at 300 tok/s. M5 measures it.

### 9.5 Epochs and safe frame reuse

The copy stream (where residency writes execute) and the compute stream (where rounds read them)
are not ordered with each other. Frame reuse is therefore keyed to **observed** device progress, not
to enqueue order:

1. Two counters are maintained. The first kernel of every round writes `round_started = r`, and the
   last writes `round_done = r`. Both are mapped host memory (§8.2).
2. To evict, the agent enqueues `WriteValue(entry, ABSENT|gen)` followed by an event. It polls the
   event without blocking.
3. When the event has completed, every round that starts from then on reads `ABSENT`. The agent then
   samples `r_s = round_started`.
4. Any round that may still use the old frame has an index ≤ r_s + D, where D is the maximum number
   of rounds in flight: 1 today, 2 with option H6. D covers a round whose `round_started` write was
   still crossing PCIe when it was sampled. The frame returns to the free list when
   `round_done ≥ r_s + D`.

**Slack frames.** Evicted frames are unusable for D + 1 rounds, so the pool keeps slack frames
outside the policy's capacity: at least the peak admissions per round × (D + 1). With too little
slack, admitted experts queue and are CPU-served although the policy counts them as hits. In the
host simulation (`tests/models/qwen4_exp/test_expert_cache.cpp`), 12 slack frames for ~27 peak
admissions per round left 12% of policy hits CPU-served; 48 slack frames served 98% from the GPU.
At recipe B's ~15 promotions per token, D = 1 and storm peaks of ~60, that is ~120 frames (1.4% of
the pool). M5 sets the value from the measured admission distribution.

As built, the Program is synchronous: `after_round` runs on a quiescent compute stream, and its
`on_quiescent` frees every retired frame and drains the queue before the next round. Slack
frames would never hold an expert, so it uses none (slack 0). The controller is still built with one
round in flight (`rounds_in_flight = 1`, `expert_residency.cpp:33-34`). Because `after_round`
always reaches `on_quiescent` (`:130`), no evicted frame is pending at a round boundary; vision
lending asserts this (§19.3.2). The code had reserved min(F/8, 512) = 512 frames. An asynchronous
transfer agent that overlaps rounds needs slack again.

When the engine worker has synchronized the compute stream (no round in flight, for example
between requests or before a prefill), every pending frame is reusable at once. The same rule
covers KV-loan reclaim (§9.2) and staging-frame reuse.

No kernel ever waits for a promotion. A round that needs an expert still loading serves it from the
CPU, or waits on the landing ticket only when the arbiter predicts that is faster (§8.6).

### 9.6 Long context: QSA KV host tier

QSA attends to at most 2,051 selected tokens per query. All other KV is only scanned through its
128-dimensional pooled index key (one per 4 tokens). Strata exploits this: above 64K it keeps
32,768 cells per layer in VRAM over an authoritative host copy, with 96-99.4% block hits on a
5090. That is why its expert cache survives a 262K context.

This design adopts the same idea in Infernix's paged-KV terms, for every KV profile:

- **Index-key plane: always fully device-resident.** At 256K it is 0.22 GB (BF16, §8.10).
- **K/V pages:**
  - The authoritative copy lives in a pinned host arena, using the existing host-replica machinery
    ([paged KV §5.3](paged-kv-cache.md#53-host-replica)). At 256K it is 3.6 GB with the primary
    `int8` profile, 7.0 GB with `bf16`, 3.5 GB with `fp8`, 2.0 GB with `nvfp4` and 0.9 GB with
    `vq2`.
  - The device holds a page cache of hot 4-token blocks: CLOCK replacement, residency resolved in
    the graph.
  - Missed blocks are fetched zero-copy inside the attention kernel. A block is 4 tokens × 2 KV
    heads in the profile's bytes per layer: 4.2 KB with `int8`, 8 KB with `bf16`, 1 KB with
    `vq2`. The fetch is latency-bound, not bandwidth-bound.
  - The VQ2/K4V2 exact window is sequence state and stays on the device.
- **Writes** go to both copies at append time.
- **Enabling.** The tier is configured when `--max-context` exceeds a threshold (64K by default,
  fixed in M8 by measurement), as in the primary configuration. Below it, KV is fully
  device-resident. A sequence spills to the host only once its context outgrows the device cache.
- **Payoff.** In the primary configuration at 256K, a device cache of 32K tokens per layer keeps
  0.67 GB of `int8` KV and index keys on the device instead of 3.8 GB. That returns ~1,140 frames
  and keeps recipe B at 9,433 frames, against 9,652 at a 4K context (LFRU 0.967 vs 0.969, §4.2).
  With `bf16` it returns ~2,200 frames; with `vq2` the tier matters little (1.1 GB at 256K).
- **Host RAM.** The primary configuration's host tier is 3.6 GB at 262K. A `bf16` tier would be
  7.0 GB and leave ~15 GB of RAM for the OS (§15.2); the startup plan check decides whether it
  fits.

---

## 10. CPU expert engine

### 10.1 Role

At this design's capacity, a layer has 0 CPU-served misses most of the time and 1 miss most of the
rest (§4.2). The CPU engine is therefore optimized for the **latency of one or two experts**, not
for throughput:

- 2.765 MB per expert;
- critical path = handshake + compute + result visibility.

The engine computes NVIDIA's W4A4 arithmetic exactly, as integer dot products (§16.2). That makes
it cheap in compute on every supported ISA:

- **A cold miss** is DRAM-bound: ~35-40 µs on dual-channel DDR5, plus ~1-2 µs of A4 quantization
  and barrier.
- **A warmed miss** (§10.4) is compute-bound. The original estimate was 3-5 µs. The first measurement
  (below) puts it nearer 7-16 µs on 14 desktop cores, so M3/M5 must measure it.

PCIe DMA is used for prefetch, promotions and the arbiter's overflow case (§8.6).

### 10.2 Kernels

The kernels implement the canonical arithmetic of §16.2 on the `nvfp4_expert_rg16_v1` layout. One
vector lane holds one row, so 16 rows of a group fill an AVX-512 register. For each expert and
column:

1. quantize x to A4 with the expert's gate/up `input_scale`;
2. for each 144-byte unit: decode the codes, form the four k-quads' integer products, and scale
   each block's integer sum into int64 accumulators;
3. convert to BF16 y, apply SwiGLU, and quantize h to A4 with the expert's down `input_scale`;
4. compute the down rows the same way and write BF16 y_e.

| ISA (hosts) | Decode a 32-byte quad | Products | Block scaling into int64 |
|---|---|---|---|
| AVX-512 VNNI (Zen 4/5, Xeon) | `vbroadcasti64x4`, masked 4-bit `vpsrlw` on the upper half, `vpandd`: lane r = row r's four codes; `vpshufb` maps a code to 2·e2m1 + 12 as u8 | `vpdpbusd` with the activation's four bytes broadcast (`{1to16}`); subtract 12 · Σa per block | `vpmulld` (P · Ŝ_a), two `vpmuldq` for even and odd lanes (× Ŝ_w), two `vpaddq` |
| AVX-VNNI-INT8 (Arrow Lake, Lunar Lake) | Two 8-row halves; signed table | `vpdpbssd`, signed × signed, no bias | AVX2 `vpmuldq` |
| AVX-VNNI (Alder Lake, Raptor Lake) | Two 8-row halves | `vpdpbusd` (VEX) with the bias | AVX2 |
| AVX2 only (Zen 3) | Two 8-row halves | `vpmaddubsw`, which cannot saturate (a pair sums to at most 576), then `vpmaddwd` | AVX2 |

**Estimated cost** (to be measured in M3), AVX-512 at T = 1: about 34 vector operations per
144-byte unit. That is ~13 bytes per cycle, several times one core's share of DRAM bandwidth. At
T = 8 it is about 118 operations per unit, which is still DRAM-bound across 14 workers. Cold
misses are therefore DRAM-bound on every ISA up to T = 8. AVX2-only hosts need about twice the
operations.

The A4 quantizer uses IEEE `vdivps` and the canonical E4M3/E2M1 rounding (§16.2): 160 + 40 block
scales and 3,200 element divisions per expert and column, ~0.3-0.5 µs.

**First measurement, and a risk to the T = 8 claim.** These figures show which mechanisms matter;
they set no value. Worker count, prefetch distance, ISA variant and every rate used by the cost
model come from `host_probe` and calibration on the target (§14.2). They come from the implemented
kernels (`src/ops/offloaded_sparse_moe/cpu/`) on the development VM, not the target: a 4-vCPU Xeon
at 2.1 GHz with AVX-512 VNNI. They were taken with `tools/flash_next_probe/host_probe.cpp`.

| n (columns) | warm, µs per expert per core | warm GB/s per core | cold GB/s per core | cold GB/s, 4 threads |
|---|---|---|---|---|
| 1 | 227 | 12.2 | 5.2 | 22.2 |
| 2 | 397 | 7.0 | 4.0 | 17.6 |
| 4 | 687 | 4.0 | 2.7 | 11.0 |
| 8 | 1,385 | 2.0 | 1.7 | 6.8 |

The VM's DRAM read rate was 9.2 GB/s for one thread and 33.8 GB/s for four.

- At n = 1 a core is ~1.3× faster than its DRAM share, so cold misses are DRAM-bound, as assumed.
- From n = 2 the kernel is **compute-bound**. Cost grows almost linearly with n because each column
  repeats the product and block-scaling work: ~6× the n = 1 cost at n = 8, against the estimated 3.5×.
- Scaled to a 5+ GHz desktop core, n = 8 is ~5 GB/s per core, ~70 GB/s on 14 workers. That is only
  barely at the DRAM rate, with no margin.
- **Software prefetch matters.** Without it, one VM core read cold records at about half its DRAM
  rate. Prefetching 2-4 KiB ahead roughly doubled cold n = 1 throughput there; n = 8 did not change.
  The distance is therefore a runtime parameter (`prefetch_bytes`; the 2,048 default is a placeholder).
  `host_probe` sweeps it per worker count, and calibration fixes it for the host.
- **Action (M3):** amortize decode across columns (decode a quad once into registers, then loop
  columns) and keep column-pair products in 16-bit lanes before widening. Re-measure on the target
  with the probe. If n ≥ 4 stays compute-bound there, give such experts to the GPU route, either as a
  staging-frame DMA or by waiting on the landing ticket (§8.6). Record the decision here.

Compilation rules: `-ffp-contract=off -fno-fast-math`; explicit FMA intrinsics only, for `exp_c`.

### 10.3 Execution

- **Static ownership.** Worker w owns the same parts of **every** expert:
  - gate/up 16-intermediate units [40w/N_w, 40(w+1)/N_w): each unit is two row groups and exactly
    one A4 block of h;
  - down row groups [160w/N_w, 160(w+1)/N_w).

  The data a worker warms (§10.4) is therefore the data it later computes.
- **Per layer:**
  1. **Phase A** over every missed expert of the layer: A4(x), the owned gate/up units, SwiGLU,
     A4 of the owned h blocks into shared memory.
  2. One spin barrier (one cache line per worker, sense-reversing).
  3. **Phase B** over every missed expert: the owned down rows, using all of A4(h).

  int64 sums are exact, so how rows and blocks are split among workers can never change a bit of
  the result.
- **Handshake.**
  - Workers spin on `miss_req[l].seq` with `_mm_pause` (Intel) / `pause` (AMD).
  - They read x from `x_host[l]`, which K5b already wrote (§8.2).
  - Each worker writes its BF16 output rows to `miss_out[l]`. The last worker to finish (atomic
    counter) writes `done_seq` with a release store.
- **Memory.**
  - Expert banks sit on 1 GiB hugetlbfs pages when available, else on 2 MiB pages (THP with
    `MADV_POPULATE_WRITE`, then `cudaHostRegister`).
  - The arena is interleaved across both CCDs' memory paths by construction: one NUMA node on
    desktop parts.
  - Software prefetch runs two units ahead.
- **Topology.**
  - Workers are pinned one per physical core; SMT siblings stay idle.
  - On hybrid CPUs, E-cores are used only if calibration (§14) shows they raise the 1-2-expert
    service rate. A slow core lengthens every barrier.
  - On dual-CCD Ryzen, workers are split evenly across CCDs, because each CCD has its own memory
    fabric bandwidth limit.
- **Idle.** Workers park on a futex when no request is active.
- **Team mechanics** (TensorSharp's measured lessons, §3.3):
  - the thread that receives a request starts computing at once instead of waking others first;
  - work beyond the static ownership (for example a second expert while a worker is still on the
    first) is handed out by an atomic chunk counter;
  - spin-then-park thresholds and the worker count come from calibration (§14.2). Adding workers
    past the DRAM knee lengthens barriers without adding bandwidth.

### 10.4 Predicted-miss warming (high-reward option H2)

When the router of layer l publishes its prediction for layer l+1 (§8.4), the transfer agent
splits the predicted non-resident experts in two:

- those its DMA budget can land before layer l+1's router, which are prefetched to VRAM;
- the rest, which become a **warm list** for the CPU engine.

While the CPU engine is otherwise idle, each worker loads its own parts of the warm-list experts
into its private L2:

- 2.765 MB / N_w ≈ 200 KB per worker per expert at N_w = 14;
- capped at half of the L2 (Zen 4/5 1 MiB, Lion Cove 3 MiB per P-core).

If layer l+1 then misses on a warmed expert, the computation streams from L2 instead of DRAM. With
integer W4A4 kernels this is the largest single gain in the miss path: an estimated 3-5 µs instead
of ~40 µs.

Warming uses DRAM bandwidth only in the GPU-only phase of each layer, when DRAM is otherwise idle.
Calibration measures warmed versus cold service time, and M5 decides adoption (§17).

---

## 11. Speculative decoding

### 11.1 Expected acceptance

On this model, SGLang's all-resident MTP (3 steps, 4 draft tokens, greedy) accepts **2.9-3.3
tokens per round** on GSM8K, random and ShareGPT prompts. Strata reports 2.4-3.2 tokens per pass.
The projection (§4.2) uses 2.8: one verified token plus draft positions 1-3 accepted with
cumulative probabilities of about 0.80, 0.59 and 0.41.

Speculation is not free on an offloaded MoE. Each verify column routes its own experts, and
ninfer-ext's MTP ran slower than plain decode on chat (§3.2). This design therefore never
speculates blindly: the draft length comes from a cost model that includes misses (§11.3).

### 11.2 A drafter that never waits on the host

The MTP drafter (one QSA block, its MoE, and the optimized proposal head) runs entirely on the GPU
with **residency-bounded routing**. Its tensors are NVIDIA's: BF16 dense weights, and FP8 128×128
block-scaled routed experts with the activation scheme the checkpoint declares (§6.1).

- **Draft routing.** Router logits are computed over all 512 MTP experts. The top-10 are taken
  among experts whose entry is `READY`. The weights are the softmax over those 10 logits. This is a
  draft-only approximation: it changes acceptance, never the verified output.
- **MTP experts.** They live in the MTP pool (§9.1). Its LFRU weights each use by measured
  acceptance, so the resident set tracks the drafter's true routing. A routed MTP expert that is
  not resident is loaded by the transfer agent in the background, within the promotion budget, for
  later rounds: one 4.9 MB DMA, ~90 µs.
- **Draft head.** The existing optimized proposal head is used: a reduced vocabulary whose rows
  are read from `lm_head`, in its recipe format. A full `lm_head` read per draft step is avoided.
- **Budget.** ≈ 0.2 ms per draft step with recipe B and ≈ 0.26 ms with recipe A:
  - 0.17 GB of BF16 MTP dense weights;
  - ~49 MB of resident FP8 experts;
  - the proposal-head rows, 84 MB in 8-bit or 168 MB in BF16 for a 32K-token vocabulary.
- **PLE rows.** As built, the host reads a verify round's rows after drafting. Planned: anchor rows
  during drafting, draft-column rows during layer 0, behind a gate before `ple_embed` (§12.3).

### 11.3 Draft length

Every round, the policy picks the draft length K (and, under option H9, a tree shape) that
maximizes expected tokens per second:

```text
E[tokens(K)] / E[t_round(K)],
    t_round(K) = t_gpu(K+1) + t_miss(U_{K+1}, h_now) + t_draft(K)
```

- **t_gpu(T)** and **t_draft** come from the calibration's Stage 2 fit (§14.2).
- **U_T** (distinct experts per layer for T columns) and **h_now** come from the router kernel's
  counters, as EWMAs.
- **E[tokens(K)]** comes from the existing per-position acceptance estimates.
- **t_miss** charges each draft position its expected **rejected-path** misses, measured online.
  The replay shows that accepted tokens add no misses per token (§4.2).

Experts used only by rejected draft columns are not credited as uses in the LFRU state, so a
rejected path cannot pollute the cache.

N-gram copy proposals ([ngram](../ngram.md)) and tree verification stay available, and the same
cost model decides them.

K = 0 (plain decode) is always a candidate, so the policy turns speculation off where it does not
pay. TensorSharp's shared-MTP head is the counterexample this guards against on this model: 1.7×
on code copying, but 44-46 against 52 tok/s on prose (§3.3).

**Verify-width invariance (option, decided in M7).** Exact canonical expert arithmetic makes each
expert's output independent of placement, but dense kernels at T = K + 1 may reduce in a different
order than at T = 1. Speculative greedy output can then differ from plain greedy output at near-ties.
- TensorSharp's exact verify-row kernels (one fixed per-column reduction order at every width) cost
  2-11% of verify time.
- The option applies that rule to every dense GEMV and attention kernel on the verify path. It also
  requires the GPU wide route (n > 8, §8.5) to use the §16.2 arithmetic.
- M7 measures the cost and the observed divergence rate without the option. The option is adopted if
  the cost is below ~3% of a round.
- **Decided (2026-10-04): not adopted; one row needs nothing.** The dense Q8 routes are
  column-invariant up to 8 columns (SIMT), and one row verifies at most 5 (MTP) or 8 (n-gram)
  columns. Greedy MTP output equalled plain decode in every C = 1 run. Only multi-row blocks of
  more than 8 columns reach the MMA tiles (§19.2). Splitting those into ≤ 8-column SIMT launches
  would cost about one extra pass over the dense weights per round (estimated, not measured).
  That is above the 3 % bar for a property that does not change quality.

**More than one row (C > 1, §19.3.5).** Each row picks K as if alone and the round is padded to
1 + max K: fillers run the whole forward, their experts are uncredited recurring misses, the 8-job
CPU cap binds, and calls of ≥ 7 columns do not fork. Measured: two cold MTP rows ran 24.9 + 21.7
tok/s (older build) against 72 aggregate for plain C = 2.

- **Decided (default, 2026-10-04): the speculation gate.** Only one-row rounds draft; at B ≥ 2 rows
  decode plain (≤ 8 columns, SIMT routes) and the drafter follows by the W = 1 catch-up, so greedy
  output at C ≤ 8 should equal C = 1 (inference, checked by byte identity). Expected C = 2: ~68-72
  tok/s fill, ~120-135 warm, against ~40-55 at HEAD (model). No product option: an environment
  override in a local measurement build toggles it until S7 decides.
- **Planned, only if it beats the gate at C = 2 and C = 4 in fill, turnover and warm:** masked
  fillers (route −1: no jobs, no n-gram reads) and a joint policy. For each W a Dinkelbach search
  takes each row's K_b as the prefix with p_{b,j} ≥ λ·c_B, maximizing (B + Σ p) / (1 + g_B
  (W − 1) + c_B ΣK); W = 1 stays a candidate, and B = 1 keeps the policy above
  (`kWidthCost = g_1 + c_1`). g_B and c_B are fitted per cache regime from the round table. Model:
  +20-30 % at C = 2 in fill, turnover and warm; ~+16 % at C = 4 cold or turnover; ~0 at C = 4 warm.
  Blocks of B·W > 8 columns may then differ from C = 1 at near-ties: dense Q8 becomes
  column-invariant to 16 columns only with Q8 Phase 1a-ii (§19.3.3; today it switches to MMA tiles
  above 8 columns), and BF16 dense, QSA verification attention and the GDN record are not. **Decided
  (default, 2026-10-04):** no separate invariance task.

### 11.4 Overlap

- Drafting for round r+1 runs on the compute stream right after verify r's sampling, in the same
  graph when K is fixed for that topology.
- No stream synchronization happens per draft step; Strata synchronizes once per draft step.
- The commit fold of the recurrent states uses the existing asynchronous commit path.
- **Verify-union prefetch** (SP-MoE's idea, adapted). While the drafter runs, layer 0's router is
  evaluated on the draft tokens' embeddings. Its predicted non-resident experts go to the
  prefetch ring before the verify forward starts. From layer 1 on, the per-layer lookahead
  (§8.7) predicts for all T columns.

---

## 12. PLE n-gram table on NVMe

### 12.1 Requirements

- 16 rows of 160 FP8 values each per token: 2.5 KiB of payload.
- Rows depend on `t[p-2], t[p-1], t[p]`. EOS restarts the window, as defined in the model reference.
- The rows are consumed **once**, by the PLE kernel that runs before layer 1's K1a. In decode, the
  hiding window is layer 0: ≈ 84 µs with recipe B and ≈ 134 µs with recipe A on the §8.3 budget,
  longer when layer 0 has a CPU miss. In the replay traces, layer 0 is among the most-missed
  layers.
- Random 4 KiB reads at QD1 on current PCIe 4.0/5.0 NVMe drives take 33-44 µs with polled io_uring
  completion (published figures, other drives); this machine's Windows path is unmeasured at QD 1
  (§12.4).
- The table must stay off the page cache. RAM is fully budgeted (§15.2).

### 12.2 NVMe layout

- Rows are 160 B, 25 rows per 4 KiB block, and never straddle a block. That is a 2.4% padding cost:
  12,800,062 blocks, 52.4 GB on disk.
- The single BF16 scale is held in the Model.
- **One row is one 4 KiB `O_DIRECT` read.** Rows of one head are contiguous in the head's prime
  range, so prefill reads that touch nearby rows coalesce into one block.
- The file is its own volume (`*.ninfer.ngram`). It may live on a different NVMe drive from the
  expert banks, which are only read at load.

### 12.3 Host-hashed row cache and gated reads

This replaces the earlier design (a 64 MiB device L0 cache probed by kernels, a 2 GiB host L1 and an
io_uring NVMe agent). Every token is on the host before its rows are needed, a column's 2,560 B
crosses PCIe in ~1-2 µs, and decode misses are compulsory (§12.4), so neither pays. Steps, tests
and measurements: §19.3.4.

**As built.** The engine worker hashes 16 rows per column (`NgramHash::row_ids`: heads 0-7 the
bigram, 8-15 the trigram; EOS closes windows and pads the start; vision placeholders hash as tokens)
and `NgramVolume::read_rows` probes a direct-mapped, pageable 2^20-row cache (~170 MB, alive with
the Program), reading each distinct missing 4 KiB block once through a 64-deep polled ring (S1).
Prefill and forced-token calls read their rows before the launch and upload them with the io
(`upload_pinned`; prefill: the whole layout). Verification rounds (S2), and plain decode rounds
with a row missing from the host cache (S4b), read them after the launch: the verify graph runs `upload_pinned_when` (Core) before `ple_embed`, which copies the
rows from the pinned io once the pinned ready word equals the round's io `gate` word. The anchor
column (committed tokens) and every column of an n-gram copy row are read while the drafter runs;
the draft columns after the verify launch, while the GPU embeds and runs layer 0. A round that
warms or captures its graph reads everything first. `ple_embed`, a plain launch at the top of
layer 1, is the only reader.

**Planned.**

- **S0, first.** Permanent `NgramVolume` counters and a per-request diagnostic line; temporary
  host-phase timers, event-timed after-draft GPU idle, an n-gram-cold toggle, and S0b, a
  `device_.flush()` probe after the verify launch. S0's table replaces the §12.4 estimates and
  decides S3.
- **S1.** `ReadOnlyFile::read_direct_blocks`: a rolling, polled ring of 64 in-flight 4 KiB reads,
  no per-call events, no deadline; misses deduped per block; the bounce buffer (up to ~268 MB
  today) fixed at 256 KiB; counters on a per-request diagnostic line. S1b: a standalone
  multi-issuer probe (1, 2, 4 issuers); ≥ 1.6× one issuer triggers S4c.
  **S1 as built and measured (2026-10-05).** Win32: one OVERLAPPED per ring slot, completion
  polled with `HasOverlappedIoCompleted` and `YieldProcessor`; POSIX: serial reads. Misses are
  deduplicated by sorted `(block << 32 | index)` keys. Cold 16K-token prompt, ABBA, 64 new tokens:
  NVMe reads 108,468 → 86,388 (−20 %), read time ~1.84 → ~1.46 s, prefill 507.1 → 518.5 tok/s
  (+2.2 %; runs 494.5-513.7 against 503.8-526.2), ids identical. Decode looked −0.8 % on those
  1.6 s windows; a 512-token ABBA settled it: 80.1 / 80.2 against 80.3 / 79.9 tok/s (−0.06 %,
  inside the spread), ids identical. Tests: `infernix_read_only_file_test` (ring wrap, offsets,
  errors) and `infernix_qwen4_exp_ngram_volume_test` (closed-form row oracle: dedup, ring wrap,
  cache hits, slot collisions, bounds, wrong-size volume).
- **S2, gated verification.** A Core gate, `upload_pinned_when`, copies the rows in the verify graph
  before `ple_embed` once a pinned ready word equals the round's device word (strictly increasing,
  so a stale word never matches). The engine thread reads anchor columns while the drafter runs and
  draft columns while layer 0 runs, then publishes (`publish_pinned_word`: `_mm_sfence`, store). A
  scope guard zero-fills and publishes before rethrowing, so errors keep their class. ~3-4 µs per
  verify round; other calls unchanged.
- **S3, conditional.** A new exact Op, `speculative_assemble_verify_tokens`, builds the verify ids
  on the device: no host round trip between drafting and verification (~10-15 µs of nodes). Only if
  S0's events still measure ≥ 0.15 ms of after-draft GPU idle per round after S0b and S2.
- **Parked (S4):** draft-token mailbox, plain-round gate, multi-issuer reads, gated prefill.
  **Rejected:** an agent thread (the engine thread already spins there), speculative prefetch, a
  larger or associative decode cache, filler-row reuse, a read deadline that fails the round.

**Decided (default, 2026-10-04):** gate trap 120 s, above Windows' 60 s disk timeout, verified once
with a 3-5 s producer delay on an otherwise idle GPU (2 s if WDDM preemption fails); an internal
process-wide fault seam for the engine test; S3 only past its threshold; S4 parked.

### 12.4 Exposure and its acceptance

- **Inputs.** ~4.3-4.8 µs per random 4 KiB read, one issuer (measured, §19.2); QD-1 latency
  unmeasured. First-time text hits 1.5-64 %, within 0-2.5 pp of an infinite cache; repeated text
  misses 0.3-0.7 rows per MTP round (simulated, real tokenizer).
- **Today every read is exposed.** The ~0.5 ms after-draft gap (§19.2) was measured on tg512, whose
  repetitions regenerate one text into a cache that outlives the request: host turnaround, not
  reads. First-time text adds ~48-75 reads, ~0.2-0.37 ms, per MTP round (est.).
- **With the gate** reads hide behind block 0 (~0.5-0.59 ms at W = 4-5) and the draft (~0.8-1.2 ms,
  est.); a slow read stalls the GPU at layer 1, not the engine. Reads over 100 ms per round warn,
  which also shows NVMe power-state exits (50-150 ms on some drives).
- **Expected (est.), after S2 + S3 (S3 only past its S0 threshold):** warm MTP +0.5-1.9 % (S2 alone
  ~0-0.15 %), first-time MTP +1.2-3.4 % (S2 alone +0.7-1.7 %), plain 0-0.3 % (S0b flush only),
  cold prefill −87 to −148 ms per 4K chunk (S1, 29-50 % fewer reads).
- **Acceptance.** Greedy ids unchanged (C = 1 plain and MTP, the C = 2 MTP pair); ABBA, ≥ 5 reps.
  S0b: warm MTP better by > 2 × the pooled standard error, nothing worse beyond its noise. S1:
  cold-prefill block reads equal the distinct missing blocks (−29 to −50 % per 4K chunk) and pp4096
  is not slower; decode within noise; bounce memory fixed at 256 KiB. S2: n-gram-cold MTP faster
  beyond noise, warm within noise, mean gate wait ≤ 10 µs. S3 (built only if S0's events still
  measure ≥ 0.15 ms of after-draft idle after S0b and S2): warm and cold MTP faster beyond noise,
  expert hit rate within ±0.5 pp, C = 2 ids identical. Refit `kWidthCost` if its slope moves by more
  than 0.02. tg512 and repeated serve requests are n-gram-warm, so n-gram changes are judged on
  n-gram-cold runs.

---

## 13. Prefill

- **Chunks.** A prompt of n tokens is read as ⌈n / max⌉ **equal** chunks. `max` is the largest
  chunk, searched in 1K steps, whose arena can be lent from frames (§9.2) while keeping a floor of
  resident experts; it is 16K-32K for a solo long prompt. A short tail (< 1K) that moves only its
  own routed experts stays separate. architectds/Strata measured +21-59% at 9K-100K from these two
  rules.
- **Expert supply.** There are two regimes, with the threshold from calibration (Strata uses 1,024
  tokens; the fork 3,072 with CPU assist):
  - **Streamed walk** (chunk ≥ threshold). Almost every expert is used. While layer l computes,
    the transfer agent DMAs **all non-resident experts of layer l+1** into a double-buffered
    staging region of frames: one contiguous `cudaMemcpyBatchAsync` per layer, one record per
    copy. Resident experts are used in place through the frame table, with no device-to-device copy.
  - **Routed staging** (chunk < threshold). Layer l's router publishes the used set to the agent,
    which DMAs only those experts. The thinnest non-resident experts are computed by the **CPU
    engine** at the same time (below).
- **Expert compute.** Assignments are grouped by expert.
  - n_e ≤ 8: narrow route, exact, shared with the CPU (§16.2).
  - n_e > 8: wide route, a block-scaled tensor-core grouped GEMM (`mma … kind::mxf4nvf4.block_scale`)
    on the canonical A4 activations, quantized with each expert's own `input_scale`.
  - A4 is NVIDIA's activation format for these experts, so it is the model's semantics here, not an
    optional speed permission. There is no A16 expert route.
- **CPU assist for thin experts** (routed staging only). An expert routed to a few tokens costs a
  full 2.77 MB copy, but only a DRAM-bound CPU GEMV. Non-resident experts are sorted by n_e, and
  the thinnest (n_e ≤ 8) go to the CPU engine. The cut is chosen so the CPU and copy-engine halves
  finish together, from per-layer EMAs of measured service rates.
  - architectds/Strata measured 1.35-1.49× for 200-1,000-token prompts, but on PCIe 3.0 with DDR4.
    At Gen5 x16 the copy half is ~3× cheaper, so the gain is expected to be smaller. M8 measures
    it.
  - ninfer-ext prefills 512 tokens at 431 tok/s, which is where this matters most.
- **Bound.** At large chunks, PCIe bounds the streamed walk:

  | Chunk | Streamed bytes (60% non-resident) | PCIe time at ~52 GB/s | Rate cap |
  |---|---:|---:|---:|
  | 16K | 40.8 GB | 0.78 s | ~21K tok/s |
  | 8K | 40.8 GB | 0.78 s | ~10.5K tok/s |

  GPU compute is ~12 GFLOP per token:
  - The **dense layers dominate**: 7.3 GFLOP per token. They run on BF16 tensor cores (209.5
    TFLOP/s peak, ~130-150 achievable), in both recipes. Recipe B dequantizes its 8-bit weights to
    BF16 inside the GEMM, because its activations stay BF16. That is ~50-55 µs per token.
  - The experts' 4.7 GFLOP run at the ~900-985 TFLOP/s this card measures for A4 GEMM
    ([linear benchmark §9](linear-benchmark.md)): ~5 µs per token.
  - Attention and GDN chunk kernels add ~10 µs per token.

  Large chunks are therefore bound by dense compute at ~15K tok/s, just below the PCIe cap.
  FP8 activations (W8A8) would be ~4× faster for the dense layers, but they are not NVIDIA's
  activation format. The Must target of 6,000 tok/s at 32K leaves margin.
- **QSA prefill.** Pooled, normalized, RoPE'd index keys are materialized **once per 4-token
  block** when the block completes. They are stored in a BF16 index-key plane beside KV (§8.10), so
  select scans one 128-wide key per block. This is the same insight as ninfer-ext's `697ff0c7`,
  built into the KV layout rather than recomputed per call.
- **PLE rows** for a chunk are hashed and read on the host before its launch and uploaded with the
  io layout (§12.3). After n-gram S1, each distinct 4 KiB block is read once (−29 to −50 % reads
  per cold 4K chunk, −87 to −148 ms). Reading chunk i+1's rows while chunk i computes (gated
  prefill, S4d) is parked (§19.3.4).
- **MTP KV for the prompt** is built in batch from the chunk residuals.

---

## 14. Calibration and adaptive tuning

Every supported machine has the same GPU. The rest of the platform differs, and the CPU/host side
sets the miss cost, the prefetch window, the PLE latency and the prefill CPU share:

- PCIe link (Gen5 x16 vs Gen4, chipset vs CPU lanes);
- DDR5 speed and channel population;
- CPU ISA, core count and P/E or CCD topology;
- NVMe latency and polled-queue support;
- huge-page configuration;
- memlock limits.

The engine therefore **calibrates itself to the host**. Calibration only chooses among
configurations that give the same numerical results: it changes speed, never output (§14.6).

### 14.1 Three layers

| Layer | When | Budget | Decides |
|---|---|---|---|
| **Startup probe** | Every start | ≤ 3 s | Validates the loaded calibration profile against the hardware and seeds the online cost models with today's measurements. Without a profile it gives safe model-based defaults. |
| **`infernix-calibrate`** | Once per machine and artifact, and again when hardware, driver or BIOS changes (the startup probe says when) | `--quick` ~2 min; `--full` ~15-25 min | Host capability measurements, CPU kernel variant and worker set, page backing, NVMe I/O mode, and structural choices that need a restart or reallocation. Verified end to end. |
| **Online adaptation** | Continuously while serving | Off the critical path | Continuous, workload-dependent parameters: miss arbiter split (§8.6), prefetch budget and k′ (§8.7), cache parameters (§9.3 shadow replay), draft length (§11.3), prefill CPU share (§13). These start from the calibrated values instead of generic defaults. |

**Why both an explicit calibration and online adaptation.**

- Online adaptation can only explore what it can change safely mid-request: continuous knobs with
  smooth effects.
- Worker count and placement, page backing (THP vs hugetlbfs 2 MiB or 1 GiB), the CPU kernel
  variant, NVMe I/O mode, the RAM split between the prefix Host tier and the KV host tier, and the
  long-context KV-tier threshold are **structural**. They need a restart or a reallocation, or they
  behave discontinuously. Exploring them online would cause latency spikes.
- A calibrated starting point also means the first requests run at full speed. Online estimators
  would otherwise have to converge from generic defaults.

### 14.2 What `infernix-calibrate` measures and decides

**Stage 0: inventory.** Instant, nothing timed. It reads:

- GPU UUID, VBIOS, driver and CUDA versions, and the negotiated PCIe link (generation and width,
  from NVML);
- CPU model, ISA flags, P/E or CCD topology (cpuid, sysfs);
- installed memory, DIMM count and speed (sysfs/SMBIOS when readable);
- NUMA layout;
- NVMe model and firmware;
- huge-page pools, THP mode, `RLIMIT_MEMLOCK`;
- the kernel version.

Stage 0 also emits **actionable system advice** before measuring:

- a link trained below Gen5 x16;
- DIMMs below rated speed, or a 4-DIMM configuration known to drop speed;
- no 2 MiB / 1 GiB huge pages configured;
- a memlock limit below the plan;
- NVMe without polled queues;
- background load on the GPU or CPU.

**Stage 1: microbenchmarks** (~60-90 s, GPU and host quiescent). Each runs ≥ 5 interleaved
repetitions and reports median and range:

| Measurement | Method | Used for |
|---|---|---|
| H2D DMA bandwidth vs copy size (64 KiB-16 MiB), D2H | Copy engine from the pinned arena, each page backing | Prefetch and promotion budget; detects small-copy penalties |
| Copy engines: `asyncEngineCount`, and H2D throughput with concurrent D2H | Device attribute, then concurrent copies | Whether prefetch, on-demand DMA and promotions share one host-to-device engine, as expected on GeForce (§8.6) |
| SM zero-copy read bandwidth and latency from mapped host memory | Kernel reading the expert arena | KV-tier block fetch (§9.6) |
| DRAM read bandwidth vs threads (per P-core, E-core, CCD) | Streaming reads over the pinned arena | Worker set; per-CCD placement (Zen CCDs have separate bandwidth limits) |
| **DRAM contention curve:** CPU read bandwidth while H2D DMA runs at 0 / 25 / 50 / 100% | Concurrent runs | Arbiter and token-bucket model (§8.6, §8.7). Host DRAM is the shared bottleneck. |
| CPU W4A4 expert kernel: each compiled variant (AVX-512 VNNI, AVX-VNNI-INT8, AVX-VNNI, AVX2), T ∈ {1, 2, 4, 8}, worker counts, SMT on/off, prefetch distance, cold and L2-warmed | Real expert records | Kernel variant, worker set, service-rate table t_cpu(n_experts, T), warmed-miss time for H2 |
| GPU↔CPU mailbox round trip | Mapped-memory publish → host spin → completion → device acquire | Handshake term in the cost model |
| NVMe random 4 KiB unbuffered reads on the n-gram file: QD-1 latency, and batches of 64-4,096 with 1, 2 and 4 issuers (n-gram S1b) | Random row ids | Whether multi-issuer reads (S4c) pay; expected exposure behind the gate (§12.4) |
| Huge pages | DMA and CPU bandwidth over THP 2 MiB, hugetlbfs 2 MiB and 1 GiB; `cudaHostRegister` time of the 70 GB arena for each backing | Page backing of the expert arena; startup time |
| Host RAM budget | Free and reclaimable RAM after the pinned expert arena, the page cache the n-gram volume needs and the OS reserve | One joint budget for the pinned arena, Vision pins, the prefix Host tier (§19.3.1) and the KV host tier (the n-gram row cache is a fixed ~168 MB, §12.3), so pinning never starves the page cache (TensorSharp sizes these separately and can over-commit) |
| Worker wake-up | Spin-then-park threshold and futex wake latency per core type | The CPU team's idle policy (§10.3) |
| Copy-engine submission latency | `cudaMemcpyBatchAsync` call → `land_seq` visible on the device, for 1-16 experts | Prefetch window model (§8.7) |
| GPU sanity | HBM read probe vs the 1,674.5 GB/s sustained-read reference, memory clock under CUDA load, fixed-shape decode kernels vs the §8.3 budget | Detects power or memory-clock limits, a downtrained link or a busy GPU, and refuses to calibrate on a disturbed machine |

**Stage 2: model-level cost fit** (~1-2 min). The real decode graph runs with **synthetic routing**
at controlled miss rates (0, 1, 2, 4 misses per layer; T = 1, 4, 8). This fits the online cost
model's coefficients to this machine:

- t_gpu(T);
- exposed stall per CPU-served miss;
- the hit-GEMV overlap;
- the prefetch window per layer;
- the drafting cost.

The fitted model replaces the generic assumptions in §4.2 for this host. It also produces an honest
**predicted tok/s versus hit rate** curve for the machine.

**Stage 3: end-to-end A/B of structural choices** (`--full` only, ~10-20 min).

- **Method.**
  1. The cost model ranks candidates and predicts the best.
  2. Only the top 2-3 per choice are verified end to end, on a bundled calibration corpus (chat,
     code, CJK, 32K document), through the public Engine.
  3. Runs are interleaved A/B/A/B, with ≥ 3 repetitions each.
  4. Coordinate descent runs over these choices:
     - worker set (incl. E-cores, SMT);
     - CPU kernel variant;
     - page backing;
     - n-gram read issuers (S1b);
     - prefetch on/off, k′ and depth;
     - RAM split between the prefix Host tier and the KV host tier;
     - KV-tier threshold;
     - maximum prefill chunk and the CPU-assist threshold;
     - maximum draft length and n-gram copy proposals.
- **Acceptance rule.** A non-default choice is kept only if its median beats the default by more
  than the larger of 2% and twice the measured run-to-run spread. Otherwise the default stays.
  Strata uses a fixed 3% margin; this rule also respects noise.
- **Reporting.** The report lists every candidate measured, including the losers.

**Stage 4: cache parameters from the user's own routing** (seconds, CPU only).

- If routing-trace recording is enabled (expert ids only, opt-in), the recorded traces are replayed
  through the §9.3 policy family with `tools/expert_cache_replay`.
- This fixes the LFRU parameters and promotion budget for the user's real workload, rather than for
  the calibration corpus.
- The online shadow replay keeps them current afterwards.

### 14.3 Calibration profile

The output is a JSON profile in the engine's state directory, next to the saved cache state. It
contains:

- **Fingerprint.**
  - Hardware: GPU UUID, PCIe link, CPU model and microcode, memory size and speed, NVMe model and
    firmware.
  - Software: driver, CUDA, kernel.
  - Artifact id and engine build.
- **Measurements**, with medians and ranges.
- **Fitted cost-model coefficients.**
- **Chosen settings**, each with its evidence (measured gain and margin) or "default kept".
- **System advice**, and whether it was applied.

At start, the engine behaves as follows:

1. **Fingerprint matches.** It loads the profile, runs the ≤ 3 s probe (PCIe, DRAM, NVMe QD1/QD32,
   mailbox), and compares.
   - **Drift ≤ 15%:** use the profile. Online models take today's numbers.
   - **Drift > 15%:** use today's numbers in the cost models, keep the structural choices, and log
     a recommendation to recalibrate. Typical causes are a downtrained link, a changed memory
     profile, or a thermally limited CPU.
2. **Fingerprint differs** (hardware, driver or artifact changed). It ignores the profile, uses the
   probe plus model-based defaults, and recommends `infernix-calibrate`.

Explicit command-line options always override profile values. `--calibration off` ignores the
profile, which is needed for reproducible benchmarking. `--calibration PATH` selects one.

### 14.4 What stays fixed for the RTX 5090

These choices depend only on the GPU and the model, and are made once at development time with
Infernix's route-development evidence ([op development](op-development.md)). They are never
calibrated per machine:

- kernel variants, tiles and fused-route gating;
- GEMM and attention schedules;
- graph topology;
- frame size and the expert record layout;
- the device memory plan's structure.

A different GPU is outside the product scope (§1.4).

### 14.5 Mapping from Strata's tuning

| Strata | This design |
|---|---|
| `--calibrate`: `pcie_frac` {0, 0.2, 0.35, 0.55, 0.75}, `spec_min_p` {0.3, 0.5, 0.7}, workers {default, ⅔, ½}; keep if > 3% | Miss split and draft length become **online** cost-model decisions, seeded by Stage 2. Workers are chosen in Stage 1 by measured service rate and verified in Stage 3. The acceptance rule is noise-aware. |
| PCIe probe, which only matters below 20 GB/s | Full DMA size curve, plus the DRAM contention curve that the arbiter needs |
| CPU ISA detection | ISA dispatch plus measured variant choice per ISA |
| Static PLE I/O settings (256 in flight, 1M-row cache) | A fixed 2^20-row host cache and a 64-deep rolling read ring (§12.3); issuer count from the S1b probe |
| 4 KiB fallback when no huge pages | Huge pages measured. Missing huge pages produce explicit advice, not a silent slowdown. |
| Prefill chunk ladder, ring sizes | Planned per request from the frame pool (§13). Ring sizes are fixed. |
| Adaptive tier constants | LFRU with Stage 4 fit and online shadow replay |

### 14.6 Safety rules

- Calibration and online adaptation choose only among configurations that are numerically
  identical:
  - routed-expert placement, which the exact arithmetic makes invisible (§16.2);
  - different workers, page backing, I/O, prefetch and cache settings.

  Choices that change output are never made by calibration: the recipe (A or B) and the KV
  profile. Every CPU kernel variant computes the same exact result.
- Calibration runs only on a quiescent machine. It aborts and reports if the GPU is in use or CPU
  load is above a threshold.
- Every probe result and online estimate is visible in the startup log and the metrics endpoint.
  A performance report always states the calibration profile and conditions it ran under.

### 14.7 Ownership

| Component | Owner |
|---|---|
| Probe primitives: copy timing, DRAM streams, mailbox round trip, NVMe I/O | Core (model-independent) |
| Calibration profile schema, fingerprint, drift check | Runtime (common execution contract) |
| Consuming the profile when planning frames, workers, I/O, prefetch and cost models | The Qwen4Exp Program at startup |
| `infernix-calibrate` (CLI, corpus, Stage 3 orchestration through the public Engine, report) | Product (`apps/`) |
| Trace replay for Stage 4 | `tools/expert_cache_replay` |

---

## 15. Memory plans

### 15.1 VRAM: 32,607 MiB card, C=1, MTP, primary configuration

The primary configuration is `--kv-dtype int8 --max-context 262144` (§18). KV pages are backed by
frames (§9.1), so the frame pool below includes the KV reservation.

| Item | Recipe A, GiB | Recipe B, GiB |
|---|---:|---:|
| CUDA context, libraries, graphs | 0.90 | 0.90 |
| Dense Text (`lm_head` included) | 8.02 | 4.10 |
| MTP dense (BF16) | 0.16 | 0.16 |
| MTP expert pool (128 × 4.9 MB FP8) | 0.59 | 0.59 |
| GDN, PLE, conv state | 0.12 | 0.12 |
| Workspace (decode), mailboxes, residency/score tables | 0.60 | 0.60 |
| Slack (includes the 0.06 GiB of the removed device row cache; n-gram rows land in the decode io buffer, §12.3) | 0.46 | 0.46 |
| **Frame pool, including the KV reservation** | **≈ 20.99** | **≈ 24.91** |
| Frames (2.64 MiB each) | 8,153 | 9,675 |

The 262,144-token `int8` reservation is 3.82 GB, 1,381 frames, including the BF16 index keys.
Loans keep its unwritten part serving experts (§9.2), and the KV host tier caps what a long context
keeps on the device (§9.6):

| Actual context | KV on the device | Expert frames, recipe B | Expert frames, recipe A |
|---|---:|---:|---:|
| 4K | 0.06 GB | **9,652** (39.3%) | **8,130** (33.1%) |
| 32K | 0.48 GB | 9,502 | 7,980 |
| 128K, KV host tier | 0.56 GB | 9,473 | 7,951 |
| 256K, KV host tier | 0.67 GB | **9,433** (38.4%) | **7,911** (32.2%) |
| 256K without the host tier | 3.82 GB | 8,294 | 6,772 |

Up to 128 frames act as staging at any time. For comparison, ninfer-ext has 6,347 slots (26%),
with Q8 dense weights.

**Other KV profiles** (recipe B; recipe A has 1,522 fewer in every cell). These are the frames once
the context has filled, without the host tier:

| `--kv-dtype` | 32K | 128K | 256K |
|---|---:|---:|---:|
| `bf16` | 9,349 | 8,373 | 7,071 |
| `int8` (primary) | 9,502 | 8,984 | 8,294 |
| `fp8` | 9,506 | 8,999 | 8,323 |
| `k8v4` | 9,541 | 9,139 | 8,605 |
| `nvfp4` | 9,576 | 9,280 | 8,886 |
| `k4v2` | 9,599 | 9,388 | 9,107 |
| `vq2` | 9,618 | 9,467 | 9,265 |

Each GB of KV costs ~362 frames. The host tier holds every profile near its 4K frame count, at the
host-RAM cost in §15.2.

### 15.2 Host RAM: 96 GB (2 × 48 GB = 96 GiB; 95.8 GiB visible)

| Item | GB |
|---|---:|
| Expert banks: 24,576 × 2.7648 MB NVFP4, pinned huge pages (MTP experts are device-resident q4, §6.1) | 67.95 |
| Embedding (BF16), pinned | 1.27 |
| N-gram row cache, 2^20 rows, pageable (§12.3) | 0.17 |
| Mailboxes, io and landing buffers | 0.2 |
| Prefix-cache Host tier, pinned slabs (§19.3.1) | 4.3 (4 GiB default) |
| Vision offload: pinned tower 0.90 (856 MiB) + live media ≥ 0.4 (§19.3.2) | 0-1.3 |
| QSA KV host tier (§9.6) at 262K: **3.6 (`int8`, primary)**; 0.9 (`vq2`), 3.5 (`fp8`), 7.0 (`bf16`) | 0-7.0 |
| Process: tokenizer, server, frontend | ~1.5 |
| **Total Infernix** | **~79 in the primary configuration at 262K** (~80.3 with Vision); ~75.4 below the tier threshold; ~82.4 with a `bf16` tier |
| OS, desktop, page cache headroom (of 102.8 GB = 95.8 GiB visible) | ~24 in the primary configuration (~22.5 with Vision); ~27.5 or ~20.5 |

Additional rules:

- **Loading.** Experts are read with `O_DIRECT` straight into the pinned arena (~10 s from a 7 GB/s
  drive). Loading never doubles memory through the page cache. Calibration measures
  `cudaHostRegister` time for each page backing (§14.2).
- **Startup check.** Startup verifies `RLIMIT_MEMLOCK` and `MemAvailable` against this plan,
  including the host tier the chosen context and KV profile need, and fails with a precise message.
  It does not swap. As built, the RAM ledger (§19.3.7, R4) plans every allocation against one
  reserve (`--ram-headroom-mib`, default 2 GiB), and the loader re-checks it against memory
  available when it pins; 64.47 GiB is pinned (§19.2). The prefix tier is resolved after model
  and Vision load: the request, or `min(4 GiB, available − reserve − vision media reserve)`. An
  explicit size that would leave less than the reserve plus the vision media reserve available
  fails at startup with the ledger's message; under 126 slabs (≈ 118 MB) the tier is off with a
  warning. **Decided (default, 2026-10-04):** 4 GiB; alternative 8-12 GiB for ~100K-token agentic
  sessions when ≥ 16 GiB stays free.
- **Machines with less than ~88 GB of RAM** are outside the supported range. Exclusive VRAM/host
  residency (an expert lives in exactly one tier) could support 64 GB machines later. It is listed
  as a future option and is not designed here.

---

## 16. Numerics and qualification

### 16.1 Op oracles

Every new Op is qualified against a naive FP32/FP64 oracle at the real shapes, under the
[Op development](op-development.md) rules:

| Op | Oracle | Boundaries |
|---|---|---|
| `offloaded_sparse_moe` (route) | FP64 logits; exact top-10 with lower-id ties | T ∈ {1, 2, 4, 8, 16, 64}; near-tie fixtures |
| `offloaded_sparse_moe` (A4 quantizer) | **Exact**: a scalar implementation of the §16.2 rule | Zero blocks, saturating blocks (scale > 448), subnormal scales, E2M1 and E4M3 ties, every E4M3 scale word |
| `offloaded_sparse_moe` (narrow, GPU and CPU) | **Exact**: an independent scalar implementation of §16.2 (integers and IEEE operations, no SIMD). Also the FP64 oracle with exactly decoded weights and the same A4 activations, which bounds the canonical rounding. | Every residency mix (all-frame, all-staging, all-CPU, random); n = 1-8; every CPU ISA variant; **bit-exact equality across routes** |
| `offloaded_sparse_moe` (wide) | FP64, exactly decoded weights, canonical A4 activations | n = 9-8192 |
| MTP experts (`q4_g64_fp16`, re-quantized from the checkpoint's block-scaled FP8) | FP64 with exactly decoded weights; activations per the declared scheme | Drafter shapes, T = 1-4 |
| `hyper_connection` | FP64 | BF16 or 8-bit weights decoded exactly; grouped norm per stream; split-K partial order fixed |
| `qsa_prep`, `qsa_select` | FP64 block scores | Near-tie allowance as in the existing selector contracts; dense equivalence while n ≤ 2051; pooled key completion at block boundaries; identical selection under every KV profile |
| QSA attention | FP64 attention over each profile's decoded K/V | Every `--kv-dtype`, with that profile's existing decode criterion; VQ2/K4V2 exact-window positions |
| `ple_ngram_injection` | FP64 | FP8 rows decoded with the scalar scale; conv state transition; EOS restarts |
| `ngram_row_ids` | **Exact integer** | Signed 64-bit products, primes 20,000,003-20,000,171, offsets; EOS and sequence start |

The fork's FP64 Python reference may be consulted as a cross-check of the published mathematics.
Infernix's oracle is written from the upstream definitions, consistent with [§1.4](#14-non-goals).

### 16.2 The canonical routed-expert arithmetic

**Requirement.** For every routed expert computed on a **narrow route** (the expert has n ≤ 8 token
columns in this forward), its BF16 output is **bit-identical** whichever of these computed it:

- the GPU from a cache frame;
- the GPU from a staging frame;
- the CPU from the host record.

Experts with n > 8 columns always use the GPU **wide route** (§8.5). The CPU never serves them. The
route is decided by n, which does not depend on placement, so outputs remain placement-invariant.

**Why exactness is cheap here.**

- Doubled E2M1 values are small integers, and E4M3FN values times 2⁹ are integers.
- Every term of a W4A4 dot product is therefore an integer multiple of 2⁻²⁰, and a whole row fits an
  int64 accumulator exactly.
- Integer addition is associative, so any split, order, instruction set or processor gives the
  same sum.
- Only two roundings remain, both at the end and both fixed.

**Definitions.** For a code word c and an E4M3FN scale word w (exponent e = bits 6:3, mantissa
m = bits 2:0):

```text
c2(c) = 2 · e2m1(c)             ∈ {0, ±1, ±2, ±3, ±4, ±6, ±8, ±12}    (c = 0x0 and 0x8 give 0)
Ŝ(w)  = e4m3fn(w) · 2⁹          = (e == 0) ? m : (8 + m) << (e − 1)     ∈ [0, 229,376]
```

**A4 quantization** of a BF16 vector v with the matrix's FP32 `input_scale` g, per 16-element
block. This is ModelOpt's reference rule, evaluated in exact IEEE FP32 with round-to-nearest-even:

```text
amax = max_j |v_j|                                  exact
s    = e4m3fn_satfinite_rn( amax / fl(6 · g) )      0 ≤ s ≤ 448
if s == 0:  every code of the block is 0
else:       d      = fl(e4m3fn(s) · g)
            code_j = e2m1_satfinite_rn( v_j / d )   ties to even on the E2M1 grid, |value| ≤ 6
```

**Checked against ModelOpt's source (M0, done).** ModelOpt 0.47's FP4 fake quantizer
(`fp4_fake_quant_kernel` via `fp8_quantize_scale` and `fp4_round_magnitude`) computes the same s, d
and E2M1 rounding, ties to even included, except for **one guard**: a dequantized scale d < 1e-5
is replaced by 1.0, which zeroes any block with amax below ~6·10⁻⁵. This rule does not adopt the
guard: TensorRT-LLM's deployed quantizer has none, and zeroing such blocks is a calibration-time
convenience rather than part of the NVFP4 format. `tools/flash_next/a4_reference.py` ports both
rules (bit-equal to the C++ quantizer on 327,680 blocks). They agree on every synthetic block
outside the guard, rounding ties included. Two checks remain for the GPU host (M3): the port against
ModelOpt's own kernel, and the guard's frequency on recorded activations.

**Row product.** For one output row r, one column, K inputs in K/16 blocks:

```text
P_b = Σ_{j<16} c2(w[r,16b+j]) · c2(a[16b+j])                      |P_b| ≤ 2,304, exact int32
S   = Σ_b P_b · Ŝ(ws[r,b]) · Ŝ(as[b])                               exact int64, any order
y   = bf16_rn( (fl32_rn(S) · 2⁻²⁰) ⊗ α ),   α = fl32(weight_scale_2 · input_scale)
```

- Each term is below 2⁴⁷ and K/16 ≤ 160, so |S| < 2⁵⁵.
- The int64 → FP32 conversion is the sum's only rounding. The scaling by 2⁻²⁰ is exact.
- ⊗ is one FP32 multiply, then one BF16 rounding.
- α is computed once per matrix by the loader (§6.1), so both processors use the same word.

**SwiGLU** between the projections, on the BF16 outputs g_i = y_gate,i and u_i = y_up,i widened
exactly:

```text
silu_c(g) = g ⊘ (1 ⊕ exp_c(−g))                  if g ≥ 0
           = (g ⊗ e) ⊘ (1 ⊕ e),  e = exp_c(g)     if g < 0
h_i = bf16_rn( silu_c(g_i) ⊗ u_i )
```

- ⊘, ⊕ and ⊗ are IEEE FP32 operations, with the parenthesization exactly as written.
- The two branches keep SiLU's sign and magnitude for large negative g. The single formula gives
  exp_c(−g) = +∞ and returns −0 already at g = −90.5.
- `exp_c` is one shared implementation in `ops/common/canonical_math.h`:
  - Cody-Waite range reduction;
  - a fixed degree-7 Taylor polynomial evaluated by explicit `fmaf` (degree 6 measured 2.65 ulp
    against FP64 over every BF16 input; degree 7 measures 0.857 ulp);
  - exponent reconstruction by two exact power-of-two scalings, so subnormal results are formed
    correctly;
  - defined overflow (+∞) and underflow (+0).
- Its inputs are BF16, so CPU-GPU equality of SiLU is verified **exhaustively** over all 65,536
  inputs.
- h is then quantized to A4 with the down matrix's `input_scale`, and the down product above gives
  the expert output y_e in BF16.

**Semantic boundaries.** They follow the structure of NVIDIA's deployed NVFP4 MoE: BF16 GEMM
outputs, and A4 at each expert GEMM input.

```text
x (BF16) → A4 → gate/up → BF16 → SwiGLU → BF16 h → A4 → down → BF16 y_e → combine (FP32, rank order) → BF16
```

Nothing else is frozen: tiling, split points, processing order, staging and the int64 reduction
order are free. The result is a function of the stored words, the input scales and x only.

**Compiler and instruction rules.**

- **CPU:** `-ffp-contract=off -fno-fast-math`; explicit FMA intrinsics only, in `exp_c`.
- **GPU:** the canonical header uses `__fmaf_rn`, `__fmul_rn`, `__fadd_rn`, `__fdiv_rn` and
  integer intrinsics.
- `cvt.rn.satfinite.e4m3x2.f32` and `cvt.rn.satfinite.e2m1x2.f32` must match the CPU's integer
  implementations. M3 verifies this exhaustively over every FP32 input the quantizer can produce.

**Fidelity to NVIDIA's runtimes.** NVIDIA's deployed quantizers compute the block scale and
element codes with approximate reciprocals (TensorRT-LLM's FP4 quantization kernel uses
`rcp.approx`). They can differ from the rule above, and from each other, only when a value lies
within a few ULPs of a rounding boundary. M3 reports two numbers on activations recorded from real
prompts:

- the A4 code mismatch rate against ModelOpt's reference quantizer and against the
  TensorRT-LLM/FlashInfer quantizer;
- the expert-output difference against a CUTLASS NVFP4 GEMM.

**Cost gate (M3).** K7a and K7b are measured against their §8.3 byte budgets at n = 1, 2, 4 and 8.

- The integer products and int64 accumulation cost about 2 integer instructions per weight byte at
  n = 1 and about 6 at n = 8. That is an estimated 5-18% of the card's integer issue rate at
  1.5 TB/s.
- A kernel more than 10% over budget gets kernel work (IMMA at smaller n, deeper pipelining), not a
  different arithmetic.
- The last resort, if a gap remains at some n ≥ 2, is to give experts with that n to the wide
  route. They become GPU-only, and their misses go to DMA. The decision and its measured cost are
  recorded here.

**What placement invariance buys:**

- Greedy output does not depend on cache contents, prefetch timing or the CPU/GPU split. Strata
  measures only 95-97.7% top-1 agreement between cache states.
- No round ever waits for a promotion.
- Every residency route is tested by exact comparison: all-GPU vs all-CPU vs random mixes
  (§16.4).

### 16.2.1 The canonical W4A16 arithmetic (experts without activation scales)

Some NVFP4 exports are weight-only: their routed experts carry codes, block scales and a global
weight scale but no activation scale, because the serving runtime keeps activations in BF16
(orcarouter's Qwen3.8-Flash-Next-Uncensored-NVFP4, recipe C of §6.1). Such experts use this
arithmetic instead of §16.2's A4. It keeps the same requirement: a narrow-route expert's BF16
output is bit-identical on the GPU (frame or staging) and the CPU (every ISA).

**Encoding.** A BF16 column v of K elements (x, K = 2,560, or h, K = 640) is encoded once:

```text
M_j  = 128 | mantissa_j (normal), mantissa_j (subnormal)      e_j = max(E_j, 1)
emax = max_j e_j          (1 for an all-zero column; a column with Inf or NaN is "non-finite")
X_j  = sign_j · rne( M_j · 2^13 / 2^(emax − e_j) )            |X_j| < 2^21
```

- v_j = X_j · 2^(emax − 147) exactly for every element within 2^13 of the column's largest
  exponent; smaller ones are rounded to the nearest multiple of 2^(emax − 147), ties to even
  (a relative 2^−21 of the column's largest magnitude, far below the outputs' BF16 rounding).
  21 bits, rather than the 23 of the first version, let the wide route form a block's products
  with two 32-deep integer MMAs (below); the golden expert outputs did not change, since their
  columns span fewer than 13 binades.
- The narrow routes store X as three bytes: lo and mid, the unsigned low bytes of its two's
  complement, and hi = X >> 16 in [−32, 31], so X = lo + 256·mid + 65536·hi.

**Row product.** With c2 and Ŝ as in §16.2:

```text
P_b = Σ_{j<16} c2(w[r,16b+j]) · X[16b+j]       |P_b| ≤ 12 · 16 · (2^21 − 1) < 2^29, exact int32
S   = Σ_b P_b · Ŝ(ws[r,b])                     |S| < 2^55, exact int64, any order
y   = bf16_rn( (fl32_rn(S) · 2^(emax − 157)) ⊗ m )
```

- m = fl32(1 / weight global scale), the multiplier stored in the record's tail (§6.2); the
  checkpoint's global scale divides, so this reciprocal is the import's only rounding.
- The scaling by 2^(emax − 157) (2^−1 for the doubled code, 2^−9 for Ŝ, 2^(emax − 147) for X) is
  two exact power-of-two multiplies, so a subnormal product is rounded once; a non-finite column
  gives the canonical NaN (0x7FC0) in every row.
- The byte limbs make every product a byte product: the GPU uses dp4a (signed codes times unsigned
  lo and mid bytes, and times signed hi bytes); the CPU uses vpdpbusd / vpmaddubsw with unsigned
  activation bytes times signed codes for lo and mid, and biased codes (c2 + 12) times signed hi
  bytes minus 12 · Σ hi. P_b is assembled in wrapping int32, which is exact because the true value
  fits.

**SwiGLU and h.** y_gate and y_up are the BF16 outputs above; h_i = swiglu_bf16(y_gate,i, y_up,i)
as in §16.2; h is then encoded as a column of 640 with its own exponent, and the down product gives
the expert output in BF16. Semantic boundaries:

```text
x (BF16) → A16 → gate/up → BF16 → SwiGLU → BF16 h → A16 → down → BF16 y_e → combine
```

**Wide route.** Experts with more than eight columns use an integer tensor-core grouped GEMM with
the same exact sums (`wide_expert_a16.cuh`). X is split as 2^14·H1 + 2^11·H0 + 8·L1 + L0 (H1
signed, L1 a byte, H0 and L0 three bits); per 16-element block, two `mma.m16n8k32` with
B = [8·c2 | c2] (|8·c2| ≤ 96 stays a signed byte) give c2·(X >> 11) from A = [H1 | H0] and
c2·(X & 2047) from A = [L1 | L0], so P_b = 2048·first + second exactly, and S += P_b·Ŝ in int64.
The route therefore returns the narrow route's and the CPU engine's bits: a W4A16 expert's output
depends neither on placement nor on how many columns share its call.

- *Why exact (finding, 2026-10-09).* The first wide route was a BF16 tensor-core GEMM qualified
  against FP64 (mean relative L2 7.1e-5, as accurate as the exact arithmetic). It made an expert's
  bits depend on its column count in the call (eight columns: exact; nine: FP32 tensor-core
  order), so KV blocks that one request computed in a 476-column call and another reused differed
  from that request's cold 512-column computation: the prefix-cache real test failed its
  Host-block-restore (token 13) and restart-resume (token 27) checks, which pass with the exact
  route. (W4A4 is canonical on every route, so recipes A and B never had this.)
- *Speed (RTX 5090, `infernix_offloaded_moe_wide_bench --activation a16`, T = 4,096, records in
  frames, 8 staging passes).* One call: three k16 MMAs per block 7.24 ms; two k32 per block
  6.24 ms (a k32 MMA costs a k16 one's 13 cycles per SMSP; a BF16 m16n8k16 costs 26); 128-element
  stages with 16 consumer warps 5.45 ms. W4A4 takes 1.65 ms. Rejected: four stages at one CTA
  per SM alone (6.67 ms), 16 warps at 64-element stages (6.47), entry-major x planes read through
  tensor maps (GEMMs ~1.4 % faster, +0.25 ms per call to encode x per entry, 8x the x workspace).
  The profile shows the integer pipe ~50 % active and ~50 % issue utilization.
- *End to end (same day, orca artifact, INT8 KV, `infernix_bench -r 3`, ABBA, exact route vs the
  BF16 route's binary).* pp1024 (chunk 1024) 380.7 vs 373.2 tok/s (+2.0 %), pp4096 (chunk 4096)
  1,659.7 vs 1,658.8 (0.0 %), pp16384 (chunk 4096) 6,176 vs 6,378 (−3.2 %): cold chunks wait on the
  expert link, the warm chunks of a long prompt on these GEMMs. The first exact version (three k16
  MMAs) measured −13.0 % on pp16384.

**Cost.** Three byte products per weight instead of one: the GPU narrow kernels stay bound by the
record reads they share with A4, while the CPU's integer work per miss triples (§10.2); recipe C
measures its effect on decode.

### 16.3 Recipe qualification

**Recipe A** is qualified by exactness, not by a quality threshold:

- **Word equality.** Every quantized tensor's words equal the checkpoint's: expert codes, block
  scales, `weight_scale_2` and `input_scale`; MTP codes and block scales; n-gram codes and scale.
  So do all BF16 tensors. The check compares per-tensor hashes and the decode oracle.
- **Reference agreement.** On the fixed corpus (code, prose, chat, CJK, long-context NIAH),
  CausalScoring reports mean and p99 KL, and top-1 agreement, against an independent runtime
  serving the same NVIDIA checkpoint (SGLang or vLLM on a larger GPU). The expected result is noise
  from the runtimes' different activation rounding, and the report states it.

**Recipe B** is qualified against recipe A on the same corpus. The thresholds are fixed now,
before any measurement:

| Metric, recipe B vs recipe A | Threshold |
|---|---|
| Mean KL | ≤ 0.005 nats |
| p99 per-token KL | ≤ 0.08 nats |
| Top-1 agreement | ≥ 98% |
| Perplexity increase ([perplexity](../perplexity.md) tooling) | ≤ 0.5% |
| Each 8-bit class alone (only that class in 8 bits) | mean KL ≤ 0.002 nats |

Classes are tried as `fp8_e4m3fn_row_bf16` first, then `q8_g32_fp16`, then kept in BF16 (§6.1).
Recipe B ships only if the full recipe passes. The §1.3 comparison with Strata UD-Q4_K_XL uses the
same metrics against BF16 reference logits, where its outputs can be obtained.

### 16.4 Engine-level and system tests

- **Exactness.**
  - Every narrow-route expert's output is bit-identical across batch shapes (T = 1 to verify
    windows, C = 1-8) for the same inputs, because the canonical arithmetic is per column.
  - Speculative and prefill agreement follow Infernix's existing criterion. Dense layers select
    different tiles at different widths, so greedy decode is compared with a fresh one-token
    prefill of the committed prefix. The disagreement rate and its largest margins are reported, as
    in [tree verification](tree-verification.md).
- **Placement invariance end to end.** Greedy outputs are byte-identical across:
  - forced all-CPU experts (`--expert-frames 0`);
  - a small cache;
  - the full cache;
  - prefetch on and off;
  - H2 warming on and off;
  - arbiter decisions forced to each service;
  - every CPU ISA variant the host supports.
- **Exact import.** Recipe A's word equality (§16.3) runs on the real artifact.
- **KV profiles.** Decode and prefill run under every `--kv-dtype`; selection is identical across
  profiles on near-tie-free fixtures.
- **PLE.** Host-cache hits and NVMe reads give exact equality, and a delayed, poisoned producer
  behind the gate gives the same ids (§19.3.4 tests 2, 3 and 5).
- **Policy conformance.** For recorded route logs, the transfer agent's LFRU victim and promotion
  decisions equal `tools/expert_cache_replay`'s decisions, step for step. Ties are broken by expert
  id.
- **Protocol stress.** Injected random delays (0-500 µs) are applied in:
  - CPU workers;
  - the transfer agent;
  - the n-gram reads (producer delay behind the gate);
  - DMA completion.

  Outputs stay exact, and no frame is reused before its retire epoch. The latter is checked by a
  debug build that poisons retired frames.
- **Fault injection.** Each of these reaches the Engine-wide failure path within the 2 s bound:
  - CPU worker death;
  - an agent stall;
  - a device spin timeout.

  An NVMe read error is rethrown unchanged after the gate is published and takes the worker-crash
  path (§12.3; §19.3.4 test 5c).

### 16.5 Precision boundaries and measured quality (2026-10-04, RTX 5090)

Every boundary was checked against the upstream code (Transformers `qwen4_exp` with its Qwen3.5 and
Qwen3-Next parents, SGLang `qwen4_exp.py`) and against Strata's kernels. A boundary is never
less precise than upstream. Where more precision was possible, it was **measured** on real text
and kept only if it did not hurt.

**Method.** Three frozen texts are scored teacher-forced on identical token ids: 511 positions
of C++ (`src/core/layout.cpp`), 1,023 of a document (`README.md`) and 1,023 of a chat transcript
in the model's chat template. Perplexity is taken over the actual next tokens. Differences are
paired per position, as the mean ΔNLL ± its standard error. Tools:
`infernix_qwen4_exp_forward_real_test --dump-logits` and `tools/flash_next/strata_compare.py`.
Strata runs the user's own `strata-unsloth-ud-q4_k_xl.json` (UD-Q4_K_XL, INT8 KV) plus
`--short-read 4096` and `STRATA_LOGPOS`, as in its `docs/UNSLOTH_Q4.md`. Per-op error is checked
separately on Infernix's own taps with `tools/flash_next/block_check.py` (FP64 oracle) and
`upstream_check.py` (Transformers modules in BF16 with the real weights).

| Boundary | Transformers / SGLang | Strata | Infernix | Evidence and decision |
|---|---|---|---|---|
| Dense weights | BF16 | Q4_K-Q8_0 mixes | BF16, word-exact | The main reason Infernix beats Strata on chat (below) |
| Routed experts | NVFP4 W4A4 | Q4_K/Q5_K gate-up, Q5_1/Q8_0 down, Q8_1 activations | NVFP4 W4A4, canonical arithmetic | No 8- or 16-bit activation path is needed: Infernix's overall quality is already better than Strata's |
| Residual stream (4 × 2,560) | BF16 | FP32 | **BF16** | FP32 tried. All op chains stayed exact (median residual error 0.105 % → 0.03 %), but perplexity got **worse**: +0.030 ± 0.009 nats overall, +0.051 ± 0.015 on chat. The model expects the BF16 rounding it was trained and calibrated with. Rejected |
| Router logits | BF16 Linear output, FP32 softmax | FP32 | **FP32** (`projection_fp32`) | BF16 logits make experts within one BF16 step (0.03 at logit ≈ −4.3) tie, and the lower id always wins. Measured BF16 − FP32: +0.008 ± 0.008 nats (code +0.022 ± 0.010). No cost. Kept |
| LM-head logits | BF16 | FP32 | **FP32** (`projection_fp32`) | Perplexity unchanged (BF16 − FP32 = −0.0002 ± 0.0004 nats). BF16 logits create exact ties that flip greedy top-1 at 1.3 % of positions. The 1 MB output row is free next to the 1.27 GB weight read. Kept, subject to its decode kernel matching the BF16 GEMV's bandwidth (M6) |
| RMSNorm / hyper-connection norm | FP32 inside, one BF16 rounding | FP32 | Same | Attention-side mixer error 0.13-0.23 % vs Transformers' 0.27-0.35 % |
| GDN gated norm | Three roundings (normalized value, weight product, output) | – | One rounding | GDN error 0.35-0.50 % vs Transformers' 0.42-0.65 % |
| QSA v, gate, gated product | BF16 | – | BF16 | 0.43-1.8 % vs Transformers' 0.48-3.2 %. The excess over other ops is cancellation in deep layers' `o_proj` sums: emulating these three BF16 roundings leaves ≤ 0.09 % |
| QSA indexer pooled key, scores | FP32 mean → BF16, FP32 scores | – | Same | – |
| KV cache | BF16 | INT8 | INT8-G64 with Hadamard keys (primary), BF16 | INT8 costs +0.005 perplexity overall against BF16 KV (4.542 → 4.564) |

**Result against Strata** (Infernix recipe A with INT8 KV, Strata UD-Q4_K_XL with INT8 KV):

| Text | Positions | Perplexity Infernix | Perplexity Strata | ΔNLL (Infernix − Strata) | Top-1 = next: Infernix / Strata | Top-1 agreement | KL(Strata ‖ Infernix) mean / p99 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Code | 511 | 1.9052 | 1.9051 | +0.000 ± 0.016 | 84.0 % / 84.5 % | 95.1 % | 0.043 / 0.55 |
| Document | 1,023 | 9.5874 | 9.7671 | −0.019 ± 0.017 | 51.7 % / 52.4 % | 84.1 % | 0.074 / 0.84 |
| Chat | 1,023 | 3.3611 | 3.8692 | **−0.141 ± 0.024** | 71.0 % / 70.6 % | 90.7 % | 0.109 / 1.67 |
| All | 2,557 | 4.564 | 4.864 | **−0.064 ± 0.012** | | | |

Infernix ties Strata on code and the document, and is significantly better on chat. KL is taken over
Strata's top 20 plus a bucket for the rest. With BF16 KV, Infernix's overall perplexity is 4.542.
These texts are short (≤ 1,024 tokens); long-context quality is measured with M8.

---

## 17. High-reward options

These options are **not** in the baseline. Each has an expected gain, a cost, and a measurable gate
that decides whether it is built. The baseline (§8-§13) must work without any of them. Gains are
model estimates for recipe B at C=1 unless marked otherwise.

| ID | Option | Expected gain | Cost / risk | Gate (measured in) |
|---|---|---|---|---|
| **H1** | **Persistent decode megakernel.** One persistent kernel per round executes an instruction stream of the round's Op tiles. Ops provide device-callable tile functions in addition to their host entries. An on-GPU scheduler resolves dependencies with counters. Weights for instruction i+1 stream in while instruction i computes, and host waits (CPU misses, PLE rows) are just dependencies. | Insurance against the conservative column of §4.2. If PDL overlap holds, it removes the ~0.5 µs residual of each of 507 kernels: 0.25-0.4 ms per token (5-8%). If PDL overlap fails, it recovers most of the 1.7 ms gap between the columns (up to ~35%). Hazy Research reached 78% of H100 bandwidth at batch 1 against < 50% for vLLM/SGLang; Lucebox runs a hybrid DeltaNet/attention model as one persistent kernel on an RTX 3090. Counterpoint: CUDA graphs alone gave 1.26× on H100 but only 1.03× on L4, so kernel-boundary gains are hardware-specific. | Very high: a new execution framework. Op tiles must remain individually qualifiable, and graph capture becomes instruction-list capture. On sm_120's 99 KB of shared memory, pages are 8 KiB, or weights stream into registers. | M6 nsys attribution: kernel boundary and tail gaps > 10% of the token after the baseline's PDL and prefetch measures |
| **H2** | **CPU cache warming for predicted misses** (§10.4). Workers pre-load the predicted CPU-served experts' parts into their own L2 caches during the GPU's dense phase, so the miss computes from cache. | Warmed miss ~3-5 µs instead of ~40 µs with the integer kernels. +0.5% at the replay hit rate; at h = 0.90, +5% plain and +16% with MTP; at h = 0.85, +10% and +30% (§4.2 model). | Low. DRAM bandwidth is wasted on mispredictions, but DRAM is idle then. | M5: warmed-miss service time and prediction recall. Keep it if the exposed stall falls by > 25%. |
| **H3** | **Intra-expert CPU + DMA split.** For a layer with one cold miss, the CPU computes part of the expert while the copy engine lands the rest into a staging frame, and the GPU computes that part. int64 partial sums add exactly (§16.2), so any row or block split is placement-invariant. | One cold miss ~33 µs instead of ~48 µs; more at low hit rates | Medium. Two handshakes, and it competes with prefetch for the copy engine. | M5: only if cold (unwarmed) misses remain > 20% of the exposed stall |
| **H4** | **Two-layer-ahead prediction** (router of l+2 applied to x_l) | Doubles the prefetch window, so more misses become staged hits | Low. Lower recall, wasted DMA. | M1 recall@k′ and M5 useful-byte ratio > 0.5 |
| **H5** | **Multi-token protection predictor** (§9.3, policy item 6) | Up to 2.5-10 points of hit rate on top of LFRU, with perfect prediction | Research | M1 on the hidden-state trace set |
| **H6** | **Chained rounds (asynchronous scheduling).** Enqueue round r+1 before r completes. The device carries the sampled or accepted tokens. The host commits round r while r+1 runs. This needs a device-side accepted-prefix fold and an abort path for rounds launched past a stop. | Hides the host's per-round work (ingress build, commit, output): 2-6% at 4.5 ms tokens | High. Engine transaction changes; see [engine architecture §6](engine-architecture.md). | M6 nsys: GPU idle between rounds > 3% of the token |
| **H8** | **Prompt-seeded cache.** Prefill already routes every prompt token. At the end of prefill, the prompt's per-layer expert counts seed the LFRU state, and the top uncached experts are promoted while the copy engine is otherwise idle, before the first decode token (MoE-Infinity's request-level activation idea). | Fewer cold misses in the first hundreds of decode tokens after a topic switch | Low | M1 replay with prefill routing: hit rate of the first 256 decode tokens with and without seeding |
| **H9** | **Draft trees** (the existing DFlash2-style tree verification over MTP top-2 at the first one or two positions) | +10-25% accepted tokens per round in the literature. Here each extra column adds ~2 distinct experts per layer, ~0.2 ms with recipe B, so a tree must add more than ~12% accepted tokens to pay. | More columns, so more misses on rejected branches | M7 cost model: chosen per round only when expected tokens/s rise |
| **H11** | **Exclusive residency** (an expert lives in VRAM *or* RAM) for 64 GB hosts | Supports smaller machines | Write-back and eviction complexity | Outside the current product scope |
| **H12** | **BF16 GDN recurrent state** (opt-in; changes numerics) | Halves 6.3 MB of state traffic per layer and sequence: ~2% at C=1, ~5% at C=8. Measured on DGX Spark for this model: +6.8% at 1 stream, +8.5% at 8. | The reference keeps FP32 state. INT8 state lost up to 21.6 accuracy points in the literature unless quantized with decay awareness, so BF16 is the floor. | Off by default. Adopted as an option only if it passes the §16.3 recipe-B thresholds against FP32 state. |
| **H13** | **MTP dense weights in 8 bits** (recipe B's rule applied to the drafter) | MTP dense weights are read three times per round: −0.17 ms per round, ~2.5% of an MTP round | Changes only draft acceptance, never output | M7: acceptance unchanged within noise, and a round-time gain ≥ 2% |

Considered and rejected:

- **Reduced-precision miss copies** (HOBBIT-style, a 2-bit host copy of cold experts; formerly
  H10). They change NVIDIA's weights (§1.1).
- **W4A16 experts.** BF16 activations against NVFP4 weights would be closer to the BF16 source
  model, at ~0.5% instead of 15-17% normwise error per GEMM in one sm_120 study. They were also
  faster at batch 1 (b12x: 37.6 vs 43.8 µs for an 8-expert MoE). They are not NVIDIA's W4A4 model,
  which the requirement fixes.
- **FP8 activations for BF16 layers (W8A8).** Dense prefill would be ~4× faster in compute (§13),
  but BF16 layers keep BF16 activations by requirement.
- **BAR-mapped mailboxes** (gdrcopy-style; formerly H7). gdrcopy builds on GPUDirect RDMA, which
  requires data-center or professional RTX GPUs; GeForce cards are not supported. Mailboxes stay in
  mapped host memory, with device-local republishing for wide waits (§8.2).
- **Lossless entropy coding of NVFP4 codes in VRAM** to fit more experts. E2M1 codes of
  MSE-calibrated weights carry roughly 3.5-3.8 bits of entropy, a ≤ 12% capacity gain. Decoding at
  1.5 TB/s would cost more GPU time than the extra hit rate saves.
- **GPU-side expert skipping or dynamic top-k.** It changes the model's mathematics.
- **Mixed SM copy kernels for prefetch.** They steal SMs and HBM bandwidth from the decode kernels.
  The copy engine is free.
- **Floating-point atomics in the expert kernels** (as in b12x's W4A4 path and MonoMoE). They make
  results run-to-run nondeterministic: b12x gave 20 different outputs in 20 identical calls. The
  exact int64 sums of §16.2 remove the need for them.

---

## 18. Configuration surface

These are the startup options added for this architecture, with their defaults. The exact names
are final when `--help` is written. Unspecified values come from the calibration profile (§14),
then from model-based defaults. Every option that affects output is marked.

The **primary configuration** for this model, used by the targets (§1.3) and documented in its
guide, is `--kv-dtype int8 --max-context 262144` at C=1, with the recipe-B artifact.

| Option | Default | Meaning |
|---|---|---|
| `--kv-dtype bf16\|int8\|fp8\|nvfp4\|k8v4\|vq2\|k4v2` | existing (`bf16`) | **Changes output.** Every profile is supported (§8.10); `int8` is the primary configuration. |
| `--max-context N` | existing | Up to 262,144. Loans keep the unused reservation serving experts (§9.2). |
| `--expert-frames auto\|N` | `auto` | Frame pool size; `auto` takes all device memory left after the plan of §15.1 |
| `--mtp-expert-slots auto\|N` | `auto` (128) | MTP expert pool (§9.1); affects acceptance only |
| `--cpu-experts auto\|off\|N` | `auto` | CPU expert engine worker count; `off` forces DMA service for every miss (testing) |
| `--expert-prefetch auto\|off\|1\|2` | `auto` | Lookahead depth (layers) of §8.7 |
| `--cpu-warming auto\|off` | `auto` | Option H2 |
| `--cache-policy lfru\|lru` | `lfru` | §9.3; `lru` is kept for A/B and replay validation |
| `--cache-state PATH\|off` | state dir | Load and save the LFRU state and ranking |
| `--record-routing PATH` | off | Opt-in route log, expert ids only, for calibration Stage 4 and M1 |
| `--ngram-volume FILE` | `<artifact>.ngram` | N-gram table volume, which may sit on another NVMe drive (§12.3) |
| `--no-prefix-reuse` | existing (reuse on in `infernix-serve`; `infernix` keeps the cache off) | Disables prefix reuse, for strict run-to-run reproducibility. **Reuse changes output at near-ties:** a resume equals its capturing lineage, not an uncached run (§19.3.1) |
| `--host-context-mib N`, `--device-snapshot-slots N` | 4096 and 0 for this model | Prefix Host tier (§15.2); Device snapshot slots, 41.8 frames each; 0 MiB needs ≥ 1 slot |
| `--prefix-cache-file PATH` | existing (off) | Prefix-cache persistence; for this model blocks also store `mtp_next` and snapshots their meta. Decided (default, 2026-10-04): built after P6, since any rebuild invalidates it (§19.3.1) |
| `--kv-capacity N\|auto` | existing | Plus one copy-on-write page per lane (0.34 frames each) for prefix resumes (§19.3.1) |
| `--vision-offload auto\|on\|off` | `auto`: on here, off for Qwen3.5 | Vision weights in pinned host RAM, streamed per window (§19.3.2) |
| `--kv-host-tier auto\|off\|TOKENS` | `auto` (65,536) | QSA KV host tier threshold (§9.6) |
| `--hugepages auto\|1g\|2m\|thp` | `auto` | Backing of the pinned expert arena |
| `--calibration auto\|off\|PATH` | `auto` | §14.3 |
| `--spec mtp`, `--draft-tokens`, `--fixed-draft`, `--ngram-draft-tokens`, `--draft-tree-nodes` | existing | Existing speculation options; draft length policy of §11.3 |

The recipe is chosen at conversion, not at startup: `qwen3_8_flash_next_nvfp4` (A, exact) or
`qwen3_8_flash_next_nvfp4_dense8` (B, 8-bit dense). It is recorded in the artifact and **changes
output**.

Command: `infernix-calibrate ARTIFACT [--quick|--full] [--corpus DIR] [--out PATH]` (§14.2).

---

## 19. Implementation plan

Each milestone has deliverables and exit criteria measured on the target machine, in the primary
configuration (§18) unless stated otherwise. Later milestones may change because of earlier
findings. Such changes, and the evidence for them, are recorded here rather than silently absorbed.

| # | Milestone | Deliverables | Exit criteria |
|---|---|---|---|
| **M0** | Source and machine facts | NVIDIA's `hf_quant_config.json`, tensor dtypes and scales: one `weight_scale_2` per matrix; whether gate and up share `input_scale`; the MTP experts' activation scheme. The A4 rule of §16.2 checked against ModelOpt's reference quantizer. Target machine inventory: CPU ISA, DRAM bandwidth, PCIe, copy engines, NVMe QD1/QD16 latency, memory clock under CUDA load. | Facts recorded; §4, §6.1 and §8.3 assumptions confirmed or revised |
| **M1** | Routing traces, policy and predictors | Broad traces (code, chat, CJK, long context, tool use; C = 1-8; MTP with rejected drafts) from the public hidden-state trace set, Strata's `--dump-routing`, or M4's engine. Replay extended with prediction and DMA-budget modelling. | LFRU (or a better causal policy) confirmed on held-out traces, with its gap to LRU and Belady reported; recall@k′ and useful-byte ratio for depth 1 and 2; §4.2 projection updated with the real h |
| **M2** | Artifact | Recipes A and B. Formats `nvfp4_mul` and `fp8_e4m3fn_block128_f32` with exact decode oracles; `nvfp4_expert_rg16_v1` records; n-gram volume with 4 KiB blocks; residency classes; `O_DIRECT` huge-page loader. | Recipe A word equality with the checkpoint (§16.3); load time; host RAM peak ≤ §15.2 plan |
| **M3** | Ops with oracles | `offloaded_sparse_moe`: route, A4 quantizer, narrow GPU (`dp4a`/IMMA), narrow CPU (AVX-512 VNNI, AVX-VNNI-INT8, AVX-VNNI, AVX2), wide, combine. `hyper_connection`; QSA with the `int8` profile first, then the others; PLE; `ngram_row_ids`. Core mailbox, ring and spin-pool primitives with stress tests. | All Op qualifications pass. **Placement invariance** bit-exact across GPU frame, GPU staging and every CPU ISA for n = 1-8. Exhaustive equality of E4M3/E2M1 conversion and SiLU between CPU and GPU. Quantizer mismatch rates against ModelOpt and TensorRT-LLM reported. §16.2 cost gate evaluated. CPU narrow kernel ≥ 80% of measured DRAM bandwidth cold, on each ISA. Mailbox round trip measured. |
| **M4** | Functional model, CPU-only experts | Prefill, decode, MTP and Vision through the Engine with every routed expert CPU-served; CausalScoring | Model oracle agreement; recipe A reference agreement and recipe B gate (§16.3); first end-to-end numbers |
| **M5** | Expert cache | Frame pool, MTP pool, residency table and state machine, transfer agent, LFRU, loans, epochs, prefetch, arbiter, H2 warming experiment | Host policy decisions **identical** to `tools/expert_cache_replay` on recorded routing; live hit composition within 2 points of replay; no round ever blocks on a promotion; agent CPU < 15% of a core; H2 and prefetch adoption decided by measurement |
| **M6** | Decode performance | §8.3 kernels with PDL, pre-dependency prefetch and stall-time warming; K7a/K7b; device-driven PLE; nsys/ncu attribution of one token | Per-kernel times vs the §8.3 budget published; plain decode ≥ 120 tok/s (Must); H1, H6 and H12 gates evaluated |
| **M7** | Speculation | Residency-bounded drafter, miss-aware draft length, n-gram copy; tree option H9 and H13 evaluated | Speculative decode ≥ 160 tok/s (Must); speculation never slower than plain beyond noise on any corpus category |
| **M8** | Prefill and long context | Layer-streamed prefill with host-issued DMA, CPU thin-expert assist, equal chunk planning, QSA pooled-key plane, QSA prompt kernels (`int8` first), KV host tier | Prefill Must target; short-prompt (512-2K) prefill reported; decode at 128K and 256K actual context ≥ 85% of 4K; the other KV profiles reported |
| **M9** | Calibration | §14: startup probe, `infernix-calibrate --quick/--full`, profile and drift check, Stage 2 fit, Stage 3 A/B | On at least two different hosts, calibrated ≥ uncalibrated (or tie within noise) on the calibration **and** held-out corpora; every candidate reported; recalibration reproduces the choices |
| **M10** | Head-to-head | Campaign vs Strata UD-Q4_K_XL at 262K context with 8-bit KV (the user's Strata configuration) and vs ninfer-ext, on the target machine; serving C = 1-8; 4K/32K/128K/256K actual context; every KV profile; quality report | §1.3 table filled with every result, favorable or not |

The milestones on the critical path to the decode targets are M2 → M3 → M5 → M6. M1 runs in
parallel and must finish before the policy is frozen in M5.

### 19.1 Implementation status and handoff

The first pieces were built on a development VM without a GPU or Hugging Face access: a 4-vCPU
Xeon at 2.1 GHz with AVX-512 VNNI. nvcc 13.4 from PyPI was used to compile only: the project's CMake
build configures and compiles there with a stub `libcuda.so`. Everything below compiles, and its
host tests pass there. Nothing has run on the RTX 5090 or touched the real checkpoint.

| Piece | Where | Verified on the VM | Remaining on the target machine |
|---|---|---|---|
| Canonical W4A4 arithmetic (§16.2): E4M3/E2M1 encoders, A4 quantizer, `exp_c`, `silu_c`, epilogues | `src/ops/common/canonical_math.h` | Encoders against grid enumeration; BF16 rounding; `exp_c`/SiLU exhaustively over BF16 against FP64; exact int64 against FP64 | GPU build of the same header; exhaustive CPU-GPU equality (M3) |
| CPU narrow expert engine: scalar, AVX2, AVX-VNNI, AVX-512 VNNI | `src/ops/offloaded_sparse_moe/cpu/`; test `infernix_offloaded_moe_cpu_test` | Bit equality across ISAs, batch and split invariance, golden output hashes (`kGolden1`, `kGolden4`) with GCC, Clang and -O0 | Throughput on the target (`host_probe`); the multi-column optimization of §10.2; the worker team and handshake (§10.3); AVX-VNNI-INT8 |
| GPU narrow route, first version: quantize, gate/up + SwiGLU + A4(h), and down kernels using `dp4a` and the canonical header | `src/ops/offloaded_sparse_moe/cuda/narrow_expert.*`; test `infernix_offloaded_moe_cuda_test` | Compiled for `sm_120a` inside the project's CMake build (nvcc 13.4); the canonical header has no contractible float expressions | Run the test: it must reproduce `kGolden1`/`kGolden4`, equal the CPU engine bit for bit (device and mapped-host records), and match the canonical scalar functions exhaustively over BF16 and around every encoder boundary. Then the optimized K7a/K7b of §8.5 (tickets, PDL, TMA ring, persistent items, IMMA for n ≥ 3), qualified against this version |
| Formats `nvfp4_mul`, `fp8_e4m3fn_block128_f32`; layouts `nvfp4_expert_rg16_v1`, `block128_scale_v1`; codecs; ModelOpt source readers | `tools/artifact/`, `tools/convert/sources/modelopt.py`; tests in `tests/artifact/`, `tests/convert/` | Word-exact round trips; C++ and Python agree on rg16 row sums | The recipes A and B themselves (M2). C++ registration is done: `QType::NVFP4_MUL` and `FP8_E4M3FN_BLOCK128_F32`, layouts `ExpertRg16` and `Block128Scale`, `expert_bank_planes` and `block128_planes`. The reader test and `infernix_artifact_flash_next_interop_test` (Python writer to C++ reader) pass. [Tensor formats](tensor-formats.md) §3.5-3.6 and [storage layouts](storage-layouts.md) §6-7 define them |
| CPU worker team (§10.3): static ownership, Phase A/B barrier, caller as worker 0, spin-then-park; allocation-free miss path | `src/ops/offloaded_sparse_moe/cpu/expert_team.*`; test `infernix_offloaded_moe_team_test`; latency section in `host_probe` | Output equals `expert_forward` bit for bit for 1-40 workers and mixed jobs; 60 rounds with parking; ThreadSanitizer clean in `--quick` mode (1, 3 and 4 workers, 12 rounds) | Pinning to physical cores and CCD split; the mailbox handshake with the GPU (`miss_req`, `done_seq`, §8.2); latency of 1-2 cold misses on the target (`host_probe` team section) |
| Host expert cache: LFRU, residency words, frame epochs, deferred-load controller | `src/models/qwen4_exp/program/expert_cache/`; test `infernix_qwen4_exp_expert_cache_test` | Victims identical to `tools/expert_cache_replay` on a 1,200-group fixture; device/agent simulation with no frame reused early | The transfer agent around it: copy stream, `cuStreamWriteValue32`, route log, loans (M5) |
| PLE n-gram row ids (§12): exact multipliers, per-head primes, EOS-closed windows | `tools/flash_next/ngram.py`; `src/models/qwen4_exp/frontend/ngram_hash.*`; tests `test_flash_next_ngram.py`, `infernix_qwen4_exp_ngram_hash_test` | Python loop specification equals an independent tensor formulation, and its primes equal `sympy.nextprime`. C++ equals the Python fixture on 800 positions across two layers | Real config values from `inspect_checkpoint` (`ngram_size`, `heads_per_ngram`, base, divisor, seed, EOS); the device kernel (§12.3) qualified against this |
| M0 source facts | `tools/flash_next/inspect_checkpoint.py` | Synthetic checkpoints | Run on the real checkpoint and record the facts in §6.1 |
| A4 rule against ModelOpt | `tools/flash_next/a4_reference.py` | ModelOpt's rule ported from its source: one guard differs (§16.2) | Compare with ModelOpt's kernel on the GPU; guard frequency on real activations |
| M0 machine facts | `tools/flash_next_probe/` (`run_m0.sh`) | Host and NVMe probes run; GPU probe compiled for `sm_120a` | Run everything; revise §4, §8.3, §8.6 and §12 from the results |

**Findings so far that change the plan:**

- **The CPU expert kernel is compute-bound from n = 2** (§10.2). T = 8 is not DRAM-bound as
  assumed. Multi-column work and the n ≥ 4 route decision are added to M3.
- **CPU software prefetch roughly doubled cold single-core throughput on the VM** (§10.2). Its
  distance is now a runtime parameter, swept by `host_probe` and chosen on the target.
- **The pool needs slack frames:** at least peak admissions per round × (D + 1) (§9.5).
- **The SiLU and `exp_c` definitions were corrected** (§16.2) before any GPU code depended on them.
- **ModelOpt's 1e-5 scale guard** is a known, deliberate difference (§16.2).

**Suggested order on the target machine:**

1. `tools/flash_next_probe/run_m0.sh` (M0), then update §4, §6.1 and §8.3 from the results. Every
   VM number in this document is replaced by the target's: DRAM bandwidth and contention, CPU
   kernel and team rates, prefetch distance, NVMe latency, PCIe and copy-engine behavior.
2. C++ format registration, then recipe A conversion (M2).
3. Run `infernix_offloaded_moe_cuda_test`. It must reproduce `kGolden1` and `kGolden4` bit for bit. Then optimize K7a/K7b against it (M3).
4. Optimize the CPU kernel's multi-column path and re-measure it.

### 19.2 Status on the RTX 5090 (2026-10-04)

Work on the target machine (Windows, CUDA 13.4, RTX 5090, 96 GB DDR5), branch
`claude/wonderful-ritchie-65xtnh`. Qwen4Exp is its own architecture (`src/models/qwen4_exp`), as
the user asked, not part of the Qwen3.5 family.

| Milestone | State |
|---|---|
| M2 recipe B | Done (`qwen3_8_flash_next_nvfp4_dense8`): recipe A with the dense projection classes in `q8_g32_fp16`, passing the quality gate (ΔNLL +0.008 ± 0.010 nats) at +22 % decode (below). |
| Source correction (2026-10-07) | Every conversion and measurement before 2026-10-07 used a local checkpoint that was in fact `RadixArk/Qwen3.8-Flash-Next-NVFP4` (the same Model Optimizer recipe for the routed experts, its own calibration, MTP experts left BF16 and fused), not NVIDIA's; it had been taken for NVIDIA's because the MTP layout was read from it. The artifacts were rebuilt from `nvidia/Qwen3.8-Flash-Next-NVFP4` (MTP experts block-FP8, §6.1) and the published measurements repeated; the older figures in this log describe the RadixArk weights. |
| M2 recipe A | Done. Converted with no requantization: 24,576 experts, 1,638 tensors, every input scale and all 320,001,536 n-gram rows word-exact. Loads in 25 s with unbuffered reads straight into 64.47 GiB of pinned memory (8.03 GiB device). The loader refuses a pinned block that would leave less than 8 GiB of RAM. |
| M3 ops | Forward ops in place and checked per op chain against the FP64 reference on the model's own activations (§16.5): `hyper_connection`, `ple`, `qsa` (BF16 and INT8 KV), `offloaded_sparse_moe` layer kernels, `projection_fp32`, `rows`. Standalone oracle tests per op are still to be written. |
| M4 functional | Done for text generation through the public Engine (`infernix`, `infernix-serve`, `infernix_bench`): `EngineCore` over the hybrid-manager surface, whole-extent KV reservation at admission, `--ngram-volume`. MTP drafter (`--spec mtp`) since the third session. Not yet: CausalScoring, vision, prefix cache. |
| M5 expert cache | Free VRAM (less a 384 MiB reserve and 64 staging slots) as frames (9,443 on the 5090 with recipe B), LFRU via `expert_cache::CacheController` with no slack frames, promotions on a copy stream: one per layer call per generated token until the frames fill, then one every fourth token; 16 per layer call in prefill. Misses are shared between the CPU expert engine (six workers) and a PCIe stage through device slots (below). No prefetch, loans, warm list or saved profile yet. |
| M6 decode | CUDA graphs per decode batch size; skinny BF16 GEMV for every unregistered dense shape; expert kernels stage their weight slices with `cp.async`. |
| M7 speculation | Verify rounds with GDN replay records and fold, PLE/QSA commit-after-acceptance, on-device acceptance and graphs per (B, W); MTP drafter (resident q4 experts, own QSA KV layer, catch-up after commit, chained draft graph, proposal head) with an acceptance-driven draft length, and n-gram copy proposals. Greedy output equals plain decode. tg512 133.5 tok/s with MTP against 88.7 plain (C = 1). |

**Measured** (`infernix_bench`, INT8 KV, C = 1, greedy, warm cache, recipe A): prefill 167-180 tok/s
at 512 tokens; decode 42.8 tok/s for 128 tokens and 32.5 tok/s for 512 tokens. Progression of
plain decode in this session: 2.9 tok/s (zero-copy experts, generic dense GEMM) → 4.8 (skinny
GEMV) → 15.1 (expert cache) → 23.1 (staged expert slices) → 26.4 (graphs, cold cache) → 32-43
(warm). A warm profile with few promotions runs at 14.9 ms per round (≈67 tok/s): dense GEMV
5.5 ms, experts 4.2 ms, LM head and router 1.8 ms, small kernels 2.2 ms, gaps 1.2 ms. The
remaining gap to it is PCIe traffic: about 16 % of routed experts miss (≈77 per token) and the
LFRU promotes heavily, so misses and promotions together move 0.2-0.4 GB per token.

**Decisions and their reasons:**

- **Promotion budget.** Decode tok/s at 512 tokens fell as the per-layer promotion budget rose
  (budget 1: 32.5; 2: 31.2; 4: 30.3), while the hit rate barely moved (83.5-84.5 %). A promotion
  costs the same PCIe bytes as serving that expert once zero-copy, so it pays only if the expert is
  reused while resident. Decode uses 1 per layer call, prefill 16.
- **Admission doorkeeper (rejected).** Admitting only experts used at least N times did not reduce
  churn: N = 2 and 3 gave the same 17,397 promotions per 512-token request and 32.2-32.3 tok/s as
  N = 1, N = 5 cut promotions by 3 % with no speed change, and budget 2 with N = 3 was slower
  (30.9). Use counts grow too fast for a count threshold; the churn comes from LFRU's
  recency-dominated score (f / (age + 1) with 48 ticks per token).
- **Strata-style adaptive policy (rejected).** Decayed use counts (×0.915 per round), free frames
  filled at once, and every 4 rounds up to 96 swaps of the hottest misses for the coldest residents
  when the candidate leads by 1.5. Same bench, same build, LFRU vs adaptive: on 512-token
  generations LFRU won, 32.05 vs 29.57 tok/s (83.4 % vs 77.9 % hits, 17.3 K vs 8.0 K promotions per
  request). On the 128-token test the adaptive policy won, 59.99 vs 47.07 tok/s, but that test
  replays the identical greedy sequence every repetition. The adaptive policy then settles at
  97.1 % hits with no promotions, against LFRU's 93.3 % with 2.6 K promotions. That is
  memorization of one sequence, not tracking a drifting working set, so LFRU stays. The tg128
  numbers above come from that replay effect, so only the 512-token figure is a fair decode number.
- **Sampling logits.** Generation samples BF16-rounded logits through the existing sampler, whose
  top-k path keys on BF16; FP32 and BF16 logits measured equal in perplexity (§16.5). An FP32 sampler
  is a backlog item for exact greedy ties.
- **KV append.** Decode appends all sequences' K/V through device-chosen table rows
  (`kv_cache_append_batch`) so a round has no host-dependent value and can be a CUDA graph.

#### Miss path and speculative verification (2026-10-04, second session)

**Platform.** The 5090 links at **PCIe Gen5 x8**, not x16 (`nvidia-smi`; likely lane sharing with
an M.2 slot on the Z790 board). Copy-engine H2D DMA measures 27.6 GB/s. A probe
(`local/workdirs/fn/zc/zc_probe.cu`) shows that SM reads of pinned host memory depend on the
**access window**, not the CTA count alone:

| Pattern | GB/s |
|---|---:|
| 8-32 CTAs, interleaved 4-16 KiB chunks of one region (window ≤ ~0.5 MiB) | 23.5-24.0 |
| 16 CTAs, each streaming its own 16 MiB | 23.1 |
| 64-170 CTAs on separate regions, or 64 KiB chunks over ≥ 32 CTAs | 11.3-15.7 |
| The expert kernels' pattern: 40 CTAs per expert, each staging its own 46 KiB slice | 11.9-13.9 |

An nsys profile of decode put 38.5 of 48 ms per round in `gate_up`/`down` on a cold cache: the
zero-copy misses ran at ~10 GB/s.

**Staged misses (adopted).** `moe_experts` now runs each layer's jobs in passes of up to 64.
A 16-CTA stage kernel copies the pass's non-resident records into 64 device slots (177 MB, shared
by every layer) with interleaved 16 KiB chunks. The expert kernels then read VRAM, so every miss
crosses PCIe once at ~24 GB/s. Placement does not change a bit: the new layer-route test checks
0, 1, 3 and 64 slots against the CPU engine, and greedy token ids are unchanged. Same build,
`INFERNIX_Q4_NOSTAGE` A/B, identical cache decisions:

| Workload | Zero-copy misses | Staged misses |
|---|---:|---:|
| `infernix_bench` tg512 | 31.39 tok/s | 35.25 (+12.3 %) |
| `infernix_bench` tg128 | 45.49 | 47.14 (+3.6 %) |
| `infernix` code-rewrite prompt, cold cache | 25.2 | 29.9 (+19 %) |

**Promotion interval (adopted: 4 once the frames are full).** Promotions are about 30 % of
PCIe bytes per token. Promoting every N decode rounds, tg512 after staging:

| N | tok/s | Hits | Promotions / 512 tokens |
|---:|---:|---:|---:|
| 1 | 35.19 | 83.2 % | 17.6 K |
| 2 | 35.87 | 81.0 % | 9.6 K |
| 4 | 36.24 | 79.7 % | 5.1 K |
| 8 | 32.68 ± 4.59 | 78.0 % | 2.4 K |

Sparse promotion fills a cold cache slowly; N = 8 never warmed up in this bench. So every round
promotes until the promotions reach the frame count, and every fourth round after that. Final
build: tg512 36.25 tok/s.

**Speculative verification with n-gram copy proposals (M7, functional).** Verify rounds of B
sequences × W ≤ 16 columns run as CUDA graphs per (B, W). The round leaves every state in place
and records what its commit needs:

- GDN: `causal_conv1d_silu_from_states` (new read-only form) and `gated_delta_net_replay_record`
  into `GdnReplayRecords`. The commit folds the accepted prefix with `GdnReplayFoldPlan`, which
  now registers the 36-layer geometry.
- PLE: `ple_conv_inject` with no destination slots. Its convolution inputs are recorded, and
  `ple_conv_commit` applies the prefix.
- QSA: `qsa_pool_keys` with `update_tails = false`. The raw index keys are recorded, and
  `qsa_commit_tails` applies the prefix. Pooled keys and K/V past the prefix are rewritten before
  any later query reads them.

Acceptance is `speculative_accept_greedy_drafts` on the device, so greedy and sampled rows are
both handled. Experts routed only by rejected columns are not credited to the cache. The n-gram
proposer is Qwen3.5's (`--ngram-draft-tokens`, `--ngram-min-match`); for this model it may run
without a neural drafter. Each state Op is checked bit for bit against its committing form
(`infernix_speculative_state_ops_test`).

End to end, greedy token ids with K = 7 at min-match 12 or 4 equal plain decode across 24-32
verify rounds, with full and partial acceptance. Results:

- **Code-rewrite prompt:** 92 % acceptance, 7.46 tokens per round, 33.2 tok/s against 30.5
  plain.
- **Prose:** no drafts, no cost (27.1 vs 27.0 tok/s).
- **Cost of a verify round.** Before staging, a W = 8 round cost 170-390 ms, 5-8× a decode round,
  because its 80 routed columns miss far more experts. Verification pays only with staged (or
  CPU-served) misses and high acceptance.

**Next-layer prediction (studied, not adopted).** Router l applied to earlier activations of the
same token, against the true top-10 (450-token code text, all 48 layers):

| Input to router l | Recall@10 | @16 | @24 |
|---|---:|---:|---:|
| Layer l−1's MoE input | 65 % | 78 % | 85 % |
| Layer l's mixer input | 55 % | 68 % | 77 % |

Some layers are far worse: layer 39 reaches 31 % with the previous-layer input. With PCIe already
saturated by misses, wrong prefetches would cost about as much as they save, so lookahead prefetch
waits for a better predictor or spare bandwidth.

**CPU expert team throughput** (`fn/cpu/cpu_bench.cpp`, cold 2.76 MB records, i9-13900K
AVX-VNNI, DDR5-5800):

- One column: 52-55 µs per expert with 8 workers (~50 GB/s), against ~115 µs for a staged miss
  over PCIe.
- More workers are slower (E-cores, barriers): 12-24 workers give 59-83 µs per expert.

CPU and PCIe draw DRAM bandwidth in parallel, so splitting a layer's misses between them should
roughly halve the miss stall.

**CPU-served misses (adopted, §10.3).** For each decode or verify layer call:

1. One GPU kernel picks up to `cpu_expert_jobs` non-resident experts. It takes all but
   misses / `cpu_pcie_divisor` of them (§19.3.12), the fewest-column ones first, each with at most 8 columns.
2. It writes x and a request into mapped host memory (`offloaded_moe::MissRequest`), fences at
   system scope, and publishes a device-counted sequence number. Graph replays stay valid.
3. A host service thread (`CpuMissService`, worker 0 of the existing `CpuExpertTeam`) computes
   those experts from the pinned bank into mapped memory. Meanwhile the GPU stages the remaining
   misses and computes everything else.
4. A one-CTA kernel waits for the answer and places the outputs before the combine. The wait
   lasts while the service's heartbeat word changes; a silent host fails the call through its
   error word after 1 s (§19.3.7, R9; it trapped after 2 s before).

The outputs are bit-identical: the layer-route test checks CPU channels of 2 and 8 jobs against
the CPU engine, and greedy token ids are unchanged. Prefill chunks (more than 64 columns) stay on
the GPU. Same build, tg512:

| CPU workers / job cap / PCIe divisor | tok/s |
|---|---:|
| none (staged misses only) | 36.25 |
| 8 / 4 / 3 | 43.84 |
| 4 / 4 / 3 | 47.55 |
| 5 / 4 / 3 | 48.98 |
| 6 / 4 / 3 | 49.27-49.35 |
| 7 / 4 / 3 | 47.88 |
| 6 / 4 / 2 | 47.22 |
| 6 / 4 / 0 (CPU takes every miss up to the cap) | 48.16 |
| 6 / 8 / 0 | 47.66 |
| **6 / 6 / 3 (default)** | **49.80** |

Six workers beat eight, and keeping a third of the misses on PCIe beats giving them all to the
CPU. Both suggest the host's DRAM is shared by the two paths.

Further measurements:

- **Cold-cache CLI.** Greedy ids are unchanged. With the defaults: code prompt 42.5 tok/s (30.5
  staged only), prose 43.4 (27.0), code with n-gram drafts 44.1 (33.2).
- **Pinning (rejected).** Pinning the six workers to P-cores (even logical CPUs) was slower:
  46.65 vs 50.35 tok/s.
- **Row cache and wait overlap (+1.4 %).** An nsys profile of tg512 showed ~2 ms per round of
  host time between rounds, with a long tail, largely the 16 synchronous unbuffered n-gram row
  reads per token. Two changes ship together:
  - the volume now keeps a direct-mapped 2^20-row host cache and reads one call's missing blocks
    together (`ReadOnlyFile::read_direct_batch`, overlapped I/O);
  - the CPU wait now runs after the shared expert (`moe_experts(..., wait_for_cpu = false)` +
    `moe_experts_cpu_wait`).

  tg512 went from 49.71 and 49.63 (two runs of the previous build) to 50.35.
- **Profile at the defaults** (tg512, nsys, per round): dense GEMV 5.5 ms, stage 3.2, CPU wait
  2.9, expert kernels 3.5, LM head and router 1.8, small kernels ~2.1, intra-graph gaps 0.8.

#### Recipe B (`qwen3_8_flash_next_nvfp4_dense8`, 2026-10-04)

**Format choice: `q8_g32_fp16` for every class, not FP8 rows.** An int8 code with an FP16 scale
per 32 weights has far finer resolution than an E4M3 code with one scale per row (3 mantissa bits).
It costs 1.06 instead of 1.0 bytes per weight, and the user asked for the highest quality. The
recipe converts:

- the hyper-connection mixers (`down` with its `inject` group, and `up`; K 320 padded to 384);
- GDN q/k/v/z and `out_proj`;
- QSA `o_proj`;
- shared experts;
- PLE projections.

As §6.1 requires, the router, shared-expert gate, GDN a/b and every MTP tensor stay BF16. Two
further classes from the §6.1 table:

- **`lm_head` (8-bit since the second cut).** `projection_fp32` gained a `q8_g32_fp16` form: the
  same warp-per-row kernel and fixed per-K order, with weights decoded exactly. Logits stay FP32,
  so verify acceptance and scoring keep their semantics. Against the BF16-head recipe B the 8-bit
  head costs ΔNLL +0.0004 ± 0.0002 nats over 2,557 positions (KL 0.0001, top-1 agreement
  98.8-99.6 %). Against recipe A the total is +0.0080 ± 0.0097. It frees ~600 MB and 216 frames
  (9,007).
- **QSA QKVG group (stays BF16).** It is stored together with the indexer projection, which must
  stay BF16. Splitting the group changes the artifact layout of both recipes, for ~0.3 GB per
  token, so it waits.

The Q8 linear route now admits unregistered shapes through its runtime-shape templates: predicated
SIMT for T ≤ 8 and MMA tiles beyond. That also puts prefill dense GEMMs on tensor cores. The
converter's `--ngram-reuse` binds the new artifact to recipe A's n-gram volume (same checkpoint
words), so no second 52 GB volume is written.

**Quality gate (passed).** Teacher-forced, INT8 KV, the three frozen texts (2,557 positions),
against recipe A:

| Text | ppl A | ppl B | ΔNLL (B − A) | Top-1 agreement | KL(A‖B) mean |
|---|---:|---:|---:|---:|---:|
| code | 1.9052 | 1.9119 | +0.0035 ± 0.0129 | 95.3 % | 0.029 |
| doc | 9.5874 | 9.5871 | −0.0000 ± 0.0140 | 86.0 % | 0.070 |
| chat | 3.3611 | 3.4198 | +0.0173 ± 0.0186 | 92.1 % | 0.090 |
| **all** | **4.5639** | **4.5988** | **+0.0076 ± 0.0097** | | 0.070 |

The difference is within one standard error. KL ~0.07 is the size of the gap between two valid
FP64 references of this W4A4 model (§16.5). Recipe B stays ahead of Strata (UD-Q4_K_XL, INT8
KV, 4.864) by ~0.056 nats.

**Speed** (same build, INT8 KV, C = 1):

| | Recipe A | Recipe B |
|---|---:|---:|
| Expert frames | 7,739 | 8,791 |
| Hit rate, 512-token generation | 79.7 % | 90.2 % |
| tg512 | 50.35 tok/s | **61.46** (+22 %) |
| tg128 | 58.2 | 62.6 |
| pp512 (chunk 4096) | 168 | 219 |
| pp4096 (chunk 4096) | 428 | 573 (+34 %) |
| Cold-cache CLI, code / prose | 42.9 / 43.8 | 45.0 / 45.8 |

Recipe B is the recommended artifact. It converts in ~6 minutes beside an existing recipe A
volume:

```text
python -m tools.convert --model <Qwen3.8-Flash-Next-NVFP4> --recipe qwen3_8_flash_next_nvfp4_dense8 \
  --components text,vision,mtp --device cpu --out <dir>/qwen3_8_flash_next_nvfp4_dense8.ninfer \
  --ngram-reuse <recipe A volume>.ngram
```

**Prefill.** Chunk 4096 beats 1024 by 1.8× on recipe A (pp4096 427.5 vs 240.8 tok/s), because
each chunk moves every non-resident expert once.

An nsys profile of one 4,096-token prompt on recipe B (chunk 4096, 7.4 s of kernels) put 81 % in
the experts:

| Kernel | s |
|---|---:|
| `stage_kernel` | 2.67 |
| `gate_up_kernel` | 2.49 |
| `down_kernel` | 0.81 |
| QSA attention | 0.69 |
| BF16 SIMT GEMM (the QSA QKVG group) | 0.28 |
| Q8 tensor-core GEMMs (every other dense class) | 0.17 |

**Staging overlap (adopted).** Prefill chunks now split the 64 staging slots into two halves and
stage pass p+1 on a side stream while pass p computes, ordered by events. The stream and events
are Program-owned, and decode and verify graphs never use the path. Greedy ids are unchanged.
Recipe B:

| Prompt | Before | After |
|---|---:|---:|
| pp512, chunk 4096 | 219 tok/s | 265 (+21 %) |
| pp4096, chunk 4096 | 566-573 | 673 (+18 %) |
| pp4096, chunk 1024 | 325 | 413 (+27 %) |

Each gate/up CTA quantizes its expert's activation columns to A4 itself. Hoisting that out would
save ~20 % of `gate_up` by instruction count; the int64 per-column accumulation dominates. The
real lever is the A4 tensor-core wide route of §13 for experts with more than 8 columns.

**Prefill chunk versus decode** (recipe B, max-ctx 8192). Larger chunks take workspace from the
expert frames:

| Chunk | pp4096 | tg512 | Frames |
|---:|---:|---:|---:|
| 1024 | 413 tok/s | 61.46 (max-ctx 4096) | 8,791 |
| 2048 | 571 | 60.37 | 8,614 |
| 4096 | 659 | 58.63 | 8,300 |

For long agentic prompts, 2048-4096 is the better setting. A 10K-token prompt saves ~9 s, against
~0.4 s lost over 500 generated tokens. The shared product default stays 1024; lending frames to
the prefill arena (§9.2) would remove the trade-off.

**T = 1 expert kernels (adopted).** The gate/up and down kernels are templated on their pass
width. A one-token call has one column per job, and the one-column instantiation quantizes,
accumulates and reduces one column instead of eight. Per decode round: `gate_up` 3.25 → 2.06 ms,
`down` 1.08 → 0.84 ms. The arithmetic is unchanged, and the layer-route test is bit-exact at
T = 1. (Correction: an earlier version attributed the gain to occupancy. On sm_120, with 100 KB
of shared memory per SM and ~1 KB reserved per CTA, `gate_up` fits one CTA per SM at every width
(~51 KB at one column, ~88 KB at eight). Only `down` gains occupancy, at three CTAs per SM.)

**Runtime-K Q8 GEMV for T = 1 (reverted, inconclusive).** It replaced the predicated SIMT route
for single-token calls and measured slower (3.72 vs 3.36 ms per round). That profile overlapped
another session's GPU work, though, so the result needs a clean re-measure.

**Skinny Q8 GEMV for T ≤ 8 (rejected).** A second try streamed each row once per warp in
16-code chunks for every T ≤ 8 call. Clean nsys profile (tg512, dense8h): 3.39 ms per round of
Q8 dense kernels against 3.36 ms for the SIMT route, so it was removed.

**VRAM reserve (adopted: 384 MiB).** The expert cache took all free VRAM less 1.5 GiB, which
left ~1.5 GiB unused (the user saw it). Same build, dense8h, tg512:

| Reserve | Frames | VRAM free after load | tg512 |
|---:|---:|---:|---:|
| 1,536 MiB | 9,007 | 1,525 MiB | 62.25 tok/s |
| 384 MiB | 9,443 | 367-377 MiB (98.8 % used) | 63.39 (+1.8 %) |

384 MiB held through decode, a 4,096-token prefill chunk at max-ctx 8192 and n-gram verification
graphs (W = 8) with no allocation failure, so it is the default.

**CPU expert workers with dynamic scheduling (kept at 6).** The user noted that Strata keeps the
CPU busier. The team used a static split (worker w owned units 40w/N..), so E-cores held the
P-cores back and more workers ran slower. It now hands out quantize items, gate/up units and
down row groups through atomic counters (output bits unchanged; the team test passes). In
`cpu_bench`, 12-24 workers then beat 6 (cold records 41 vs 52 µs per expert, warm 26 vs 40).
End to end it does not pay: tg512 with 6 / 12 / 16 / 24 workers measured 62.25 / 62.01 / 61.51 /
58.89 ± 1.61 tok/s. With two thirds of the misses on the CPU and the rest staged over PCIe from
the same DDR5, the two paths share DRAM bandwidth, and extra spinning threads only add
contention. Six workers stay the default; their low total CPU load is the measured optimum, not
idle capacity.

**Measurement caveat.** `infernix_bench` decodes greedily from a fixed corpus prompt, so any change
in rounding (a dense kernel's reduction order, an 8-bit head) changes the generated text. That
changes routing and the hit rate with it. The same `dense8` artifact and frame count measured
90.2 % hits in one build and 85.7 % in another. Kernel changes are therefore judged by per-round
kernel time in nsys. tg throughput across numerically different builds is reported with its hit
rate, and differences of a few percent are not attributed.

#### MTP drafter (2026-10-04, third session)

`--spec mtp --draft-tokens K` (K ≤ 5 at the product surface), optionally with `--lm-head-draft`.
As built, which differs from §11.2 where noted:

- **Mathematics** (Strata's vLLM transcription, `mtp.hpp`): cell c pairs the main model's final
  multi-stream residual R_c (before the final mixer) with the token t_{c+1}, at rope position c:
  `R = fc_hidden(per stream of RMSNorm_{S·H}(R_c)) + fc_embedding(RMSNorm_H(embed(t_{c+1})))`,
  every RMSNorm with the unit offset, then one QSA block (its own K/V and index keys) with the
  hyper-connection mixers, a 512-expert top-10 MoE with the shared expert, its own final mixer
  and the text model's head. The output predicts t_{c+2}; a chained step feeds the block's output
  residual and its draft token to the next cell.
- **Experts all resident instead of a pool.** The 512 routed experts are `q4_g64_fp16` (1.34 GB)
  and always in VRAM, so routing is exact and no residency-bounded approximation exists. A new
  Op, `resident_moe_experts`, streams each selected expert's rows (Q4 or Q8 row-split codecs,
  FP32 accumulation, FP32 SwiGLU); FP64-oracle test `infernix_resident_moe_test`. Lane l reads
  chunks l, l + 32, ... of a row, so the activation is staged in shared memory **swizzled** per
  chunk; stored in order, every product's read was a 32-way (Q4) or 16-way (Q8) bank conflict
  and the Op ran at ~200 GB/s. With the swizzle the products and their order are unchanged (the
  output bits are identical) and one draft step's call (T = 1, 10 experts, L2 cold,
  `infernix_resident_moe_bench`) takes 29.2 µs instead of 128.1 µs (Q4; Q8 47.1 instead of
  70.8). End to end (2026-10-10, Dense8, INT8 KV, MTP 4 drafts with the proposal head,
  pg1024+256 on corpus text, 4 old/new pairs × 3 reps): decode 144.1 → 146.5 tok/s, +1.61 to
  +1.70 % per pair, identical speculative counts. Issuing all of a warp's weight loads before its
  products measured slower (37.5 µs; more registers, fewer resident warps).
- **Real QSA attention**, not Strata's dense window: the drafter's KV layer is a 13th layer of
  every page group, read through the same block tables; its tails are a 13th slab.
- **Cells.** Prefill chunks write the drafter's K/V for every cell whose next token is known, from
  the chunk's live residuals (in sub-chunks of 512 columns), and keep the last residual as the
  lane's pending cell. Forced tokens do the same. A verification round exports its W final
  residuals; after its commit, a K/V-only call over the W cells records the index keys and
  `qsa_commit_tails` keeps the committed ones, and the last committed residual becomes pending.
  Each decode round then runs K chained full steps from the pending cell; chained cells past it
  leave the tails alone and are rewritten by the next catch-up before any query reads them.
- **Draft head.** The text head (8-bit `lm_head`, 248,320 rows) or, with `--lm-head-draft`, the
  converter's `--proposal` head: `lm_head` rows of the 131,072 most frequent tokens in
  `q4_g64_fp16` (178 MB) and their token ids (`projection_fp32` gained the Q4 form, `argmax` an
  id-map overload).
- **Graphs.** The K-step chain is one CUDA graph per batch size and the catch-up one per
  (batch, width), as for verification.
- **N-gram proposals** stay available beside MTP: a longer copy proposal replaces a round's MTP
  drafts.

**First results** (dense8m, INT8 KV, C = 1, greedy, cold-cache CLI, K = 3, text head). Greedy
token ids equal plain decode on both prompts (300 and 200 tokens); `resident_moe_experts` and
the Q4/Q8 `projection_fp32` forms pass their FP64-oracle tests.

| Prompt | Acceptance | Tokens per round | Accepted by position (of rounds) | MTP tok/s (hit rate) | Plain tok/s (hit rate) |
|---|---:|---:|---|---:|---:|
| Code rewrite | 98.7 % | 3.96 | 74, 74, 74 of 75 | 33.5 (37.3 %) | 46.3 (69.3 %) |
| Story (prose) | 39.9 % | 2.19 | 59, 32, 17 of ~91 | 32.5 (55.9 %) | 49.5 (73.8 %) |

On a cold cache MTP was slower than plain for three reasons:

- **Promotions per round, not per token.** A round promoted one expert per layer, so a
  four-token round warmed the cache a quarter as fast (4,367 against 9,869 promotions over the
  code prompt). Promotions now follow the tokens a round advances.
- **Verify rounds miss more.** A W = 4 round routes several times the distinct experts of a
  decode round.
- **Fewer frames.** The drafter's experts, head workspace and KV layer took ~930 frames (8,517
  against 9,443).

The draft-head workspace was also sized for 512 columns (763 MB with the text head); it is now
sized per sequence, since full steps run one column each.

#### Decode-round bundle (2026-10-04, third session)

A read-only review of the open items, with an adversarial check of each estimate, ranked these
bit-exact fixes first. All of them landed together:

- **Router projection.** The BF16 `projection_fp32` form ran 17 CTAs for a 513-row router,
  each warp making ~40 dependent DRAM round trips (~22 µs × 48 per round). Decode-width calls now
  give each warp one row (129 CTAs) and issue a row's chunk loads before its FMAs, in the same
  order. Dynamic shared memory follows the live columns in every form (the 8-bit head had 40 KB
  per CTA at T = 1).
- **Per-round copies.** Decode and verification rounds copy only the used prefix of the io
  buffer, not every n-gram row sized by the prefill chunk (2.6 MB at chunk 1024, ~10 MB at 4096).
  The route log comes back as a 2D copy of the used rows.
- **Slack frames 0** (§9.5), giving the policy +512 frames.
- **CPU wait copy.** Batched 16-byte loads replace ten dependent 2-byte PCIe reads per
  CPU-served column.
- **Fork of stage and compute** in one-pass decode and verification calls (graph-captured). The
  misses stage and compute on a side stream while the resident experts compute and the shared
  expert runs; the join is in `moe_experts_cpu_wait`.
- **Register-only `route_kernel`**, register-preloaded `hyper_connection_norm`, and the CPU
  team's A4 quantize split into block slices.
- **MTP fixes:** promotions follow tokens per round, and the draft-head workspace is sized per
  sequence.

Results (dense8m, INT8 KV, C = 1). The greedy token ids equal the previous build on the code and
story prompts and on the bench corpus, so the comparisons below see the same text:

| Workload | Before | After |
|---|---:|---:|
| tg512 plain (hit rate) | 70.29 tok/s (91.9 %) | **83.30** (93.0 %, +18.5 %) |
| Cold-cache CLI, code prompt | 46.3 | 56.7 (+22 %) |
| Cold-cache CLI, story prompt | 49.5 | 57.0 (+15 %) |

The changes were measured together, not one at a time, so the split between them is not
measured. (The 70.29 baseline is higher than the 62-63 tok/s dense8h figures above because the
reverted skinny GEMV changed the greedy text, and with it the hit rate; see the measurement
caveat.)

**MTP sweep** (same build, tg512 on the bench corpus, warm cache, `infernix_bench -n 512 -r 2`).
Greedy output equals plain decode:

| Mode | tok/s | Draft acceptance | Hits | Frames |
|---|---:|---:|---:|---:|
| Plain | 83.30 | — | 93.0 % | 9,443 |
| K = 2, `--lm-head-draft` | 111.47 | 73.0 % | 89.9 % | 8,733 |
| **K = 3, `--lm-head-draft`** | **113.96** (+37 %) | 63.7 % | 89.8 % | 8,733 |
| K = 4, `--lm-head-draft` | 106.50 | 56.3 % | 89.3 % | 8,730 |
| K = 3, 8-bit text head | 110.38 | 63.7 % | 89.8 % | 8,797 |

K = 3 with the proposal head is the best setting: 114 tok/s against Strata's 72-80 tok/s with
its MTP. On this corpus the proposal head drafts the same tokens as the full head, and it is 3 %
faster.

Cold-cache CLI with K = 3 and `--lm-head-draft`:

- **Code:** 67.4 tok/s (93.9 % acceptance, 3.82 tokens per round) against 56.7 plain.
- **Story:** 47.9 tok/s (39.1 % acceptance, 2.16 tokens per round) against 57.0 plain, so prose
  is 16 % slower. A fixed K costs speed on low-acceptance text; the draft-length policy of §11.3
  (K = 0 included) is the remedy.

**Draft-length policy (adopted, §11.3 simplified).** `--draft-tokens` is now the maximum K.

- **Choice of K.** Each round, each row drafts the K in [0, max] that maximizes
  E[tokens(K)] / (1 + c·K). E[tokens(K)] = 1 + Σ_j Π_{i≤j} a_i, from the row's EWMA of the
  conditional acceptance a_i at each draft position (prior 0.75, weight 0.1). c = 0.38 is the
  measured cost of a draft column in plain rounds: warm tg512 W = 3, 4, 5 rounds took 1.84, 2.13
  and 2.54 plain rounds.
- **Probing.** Every 8th round a row drafts one token more than its choice, so the estimate for
  the next position stays current. K = 0 then probes one draft.
- **K = 0 rounds** are plain decode rounds. The drafter writes the round's cell afterwards (a
  W = 1 catch-up), so it stays current and drafting resumes at no cost.
- **Rejected first version.** Measuring round time per width (EWMA) made the policy settle on
  short drafts: the W = 4 time measured on the cold cache never refreshed once K dropped. tg512
  fell to 101.7 tok/s against 114 at fixed K = 3.

Same build, dense8m, INT8 KV, C = 1, `--lm-head-draft`; greedy ids equal plain decode in every
run:

| Workload | Fixed K = 3 | Policy, max 3 | Policy, max 4 | Plain |
|---|---:|---:|---:|---:|
| tg512, warm | 113.96 | 114.18 | **115.06** | 83.30 |
| Code CLI, cold | 67.4 | — | 67.4 (4.54 tokens per round) | 56.7 |
| Story CLI, cold | 47.9 | — | 55.1 (settles on K = 1) | 57.0 |

The policy gets the best warm throughput and removes most of the prose loss. Cold-cache prose is
still 3 % slower than plain, because a cold verify round costs more than the warm slope c
assumes: the story's W = 4 rounds took ~2.6 plain rounds on the cold cache. A cost that follows
the hit rate is the next refinement.

**MTP round profile and two fixes.** nsys of warm tg512, MTP max 3, `--lm-head-draft`, per
generated token: 11.0 ms wall against 8.5 ms GPU busy, so the GPU was idle 22.5 % of the time.

| Kernel | ms per token |
|---|---:|
| `gate_up` | 2.90 |
| `stage_kernel` (overlapped by the fork) | 2.71 |
| Q8 dense | 1.63 |
| `down` | 1.01 |
| `cpu_wait` | 0.31 |
| Drafter's own kernels | ~0.3 |

- **Occupancy (adopted).** A W = 4 verify round has ~30 jobs × 40 gate/up CTAs, about seven
  waves at one CTA per SM. The one-column CTA's activations and partial sums now share storage,
  separated by one more barrier. At 49,376 B, two CTAs fit per SM, and calls of up to four columns
  use the one-column kernels, one pass per column over the staged weights. The layer-route test
  gained a four-column case and passes. Greedy ids are unchanged. tg512: plain 83.30 → 85.46,
  MTP max 4 115.06 → **121.20**.
- **Copy-engine stall (found, fixed below).** On average 4.7 ms per round passed between a
  verification's last kernel and its commit's first. The host was in `cudaStreamSynchronize`,
  and the compute stream's next 4-byte host-to-device copy waited on the copy engine behind the
  ~40 × 2.76 MB promotion copies issued a moment earlier: about 3.4 ms of the 4.7. Small per-round
  host-to-device copies (round inputs, commit counts, drafter ids, the residency table) share the
  copy engine, which runs FIFO across streams, with bulk promotions. So every round waited for
  the previous round's promotions to land. Plain decode pays the same every fourth round, when
  48 promotions (132 MB, ~4.8 ms) go out.
- **Pinned uploads (adopted).** A Core primitive, `upload_pinned`, copies pinned (UVA-mapped)
  host memory to the device with a small kernel in stream order. Every per-round staging copy
  uses it; prefill chunks keep the bulk copy. GPU idle time fell from 22.5 % to 10.9 % of MTP
  decode. Greedy ids are unchanged.

  | Workload (same flags) | Before | After |
  |---|---:|---:|
  | tg512 plain | 85.46 | **88.81** |
  | tg512 MTP max 4, `--lm-head-draft` | 121.20 | **131.78** |
  | Cold CLI code, plain / MTP | 57.2 / 67.6 | **64.8 / 75.3** |
  | Cold CLI story, plain / MTP | 57.5 / 56.8 | **63.9 / 62.3** |

  The remaining host gaps are ~0.58 ms per round after a verification (the cache policy) and
  ~0.56 ms after drafting (verification staging, n-gram rows).
- **CPU jobs per layer call (adopted: 8).** After these fixes `cpu_wait` costs 0.35 ms per token,
  so the CPU had slack. tg512 with MTP max 4, CPU jobs / PCIe divisor:

  | Jobs / divisor | tok/s |
  |---|---:|
  | 6 / 3 (previous default) | 131.96 ± 0.75 |
  | **8 / 3** | **133.55 ± 0.49** |
  | 6 / 5 | 131.87 ± 0.57 |
  | 8 / 5 | 133.06 ± 0.54 |
  | 8 / 0 | 130.54 ± 0.43 |

  Plain decode is neutral at 8 / 3 (89.04 against 88.98). The split is not the bottleneck any
  more, because the fork overlaps the PCIe stage with the hit compute. With 8 / 3 as the default
  (greedy ids unchanged): tg512 plain 88.70, MTP max 4 133.54. Cold CLI code plain 64.7 / MTP
  81.9, story plain 63.9 / MTP 65.4, so MTP is no longer slower on cold prose.
- **Reused n-gram read events (rejected).** The after-draft host gap (~0.5 ms per round) is the
  verification's n-gram rows: new trigrams of draft tokens, read from the NVMe volume. Keeping the
  64 Win32 events of a batched read alive instead of creating them per call changed nothing
  (tg512 89.99 / 138.84 against 89.96 / 138.70), so the time is in the submissions and the device.
- **I/O ring for the n-gram rows (rejected).** A standalone probe timed batches of random
  4 KiB unbuffered reads from the volume's NVMe drive (200 batches each, median). Overlapped
  `ReadFile` took 87 / 308 / 381 µs for 16 / 64 / 80 reads; one `SubmitIoRing` took
  72 / 276 / 344 µs. Both scale at ~4.3-4.8 µs per read, the drive's random-read rate, so the
  after-draft gap is device-bound. An I/O ring would save ~30 µs per round, which is not worth a
  second read path. Fewer reads or overlap are what remain: the anchor column's 16 rows are known
  before drafting, and each draft column's rows as soon as its draft step finishes.
- **Maximum draft length 5** measured the same as 4 (tg512 138.67 against 138.78; code CLI 82.9
  against 83.7). The policy rarely drafts a fifth token, so 4 is the recommendation.
- **Deferred cache update (adopted).** A round's `after_round` (routes, LFRU, promotions; ~0.5 ms
  of host time) now runs in its commit, after the commit has enqueued its GPU work (the GDN fold,
  tail commits, the drafter's catch-up), so the policy overlaps that work. A discarded round skips
  it. Greedy ids are unchanged. tg512: plain 88.70 → **89.96**, MTP max 4 133.54 → **138.70**.
  Cold CLI: code 65.5 plain / 83.7 MTP, story 64.4 / 66.5.
- **Few-row Q8 route for the hyper-connection down projections (adopted).**
  - **Problem.** The MTP-round profile showed the [324, 10240] and [320, 10240] projections
    (96 calls per token) on 41 eight-row CTAs at ~0.4 TB/s: 9.5 µs per call, ~0.9 ms per round,
    a fifth of all Q8 time.
  - **Change.** The generic route now gives shapes with fewer than 64 eight-row blocks and K ≥ 4096
    one CTA per row, whose eight warps split K. This is the existing SIMT kernel with its
    in-CTA reduction (`SimtR1T4W8` / `SimtR1T8W8`).
  - **Microbenchmark.** CUDA graph of 200 calls, weights cycled through 256 MB so they come
    from DRAM: T = 1 7.5 → 4.7 µs, T = 4 9.8 → 7.0, T = 5 12.1 → 8.5, T = 8 12.9 → 10.9.
    The route is slower on [1024, 4096] and [1280, 2560], hence the threshold. Deeper pipelines
    (20 groups × 3 stages, or 2 rows × 4 warps) were no faster.
  - **Tests.** The FP64-oracle conformance passes. A new test pins column invariance for 1-8
    columns on both routes, which the one-row speculation equality relies on (§11.3).
  - **End to end.** The projection now rounds differently, so tg512 (empty prompt) generates a
    different text with a lower expert hit rate: 88.4 % against 93.0 %, and 24.6 K against
    15.6 K CPU-served misses per 512 tokens. Same binary, routes toggled, in ABBA order:

    | Workload | Previous route | New route |
    |---|---:|---:|
    | tg512 plain | 89.89 / 89.70 | 86.96 / 86.77 |
    | tg512 MTP max 4 | 138.62 / 138.69 | 134.00 / 133.90 |
    | Code prompt, cold CLI, plain | 65.5 / 65.6 | **66.3 / 66.5** |
    | Code prompt, cold CLI, MTP | 83.6 / 84.0 | 84.1 / 84.0 |

    The code prompt's greedy ids are identical on both routes, and in plain and MTP runs, with
    equal hit rates (69.2-69.3 %). On it, the new route is 1.3 % faster plain; MTP is within noise.
    The kernel saving predicts ~0.27 ms per plain token (~2.4 % on a warm cache). The tg512 drop
    is the new text's hit rate, not engine speed.
  - **Measurement rule.** tg512's text depends on every rounding. A change that alters rounding is
    judged on a workload whose greedy ids stay identical, with tg512 reported beside it.

**Long prompt.** 7,448-token prompt (past the 2,048-token QSA budget, so the drafter's indexer
selects), max-context 16384, chunk 4096, cold cache, 200 tokens:

- **Output:** greedy ids with MTP equal plain decode.
- **Decode:** 68.4 % draft acceptance, 2.51 tokens per round, 55.9 tok/s against 52.7 plain.
- **Prefill:** the drafter's prompt K/V barely costs anything: 506 against 510 tok/s here, and
  pp4096 664.7 against 665.5 tok/s.
- **Frames:** 8,194 with the drafter against 8,912 at this context.

**Serving at the recommended settings (final build, ecaf50e16).** `infernix-serve --max-concurrency 1
--spec mtp --draft-tokens 4 --lm-head-draft`, temperature 0, 200 tokens. The code and story requests
were sent together, so they queued, and then again each alone:

- **Output:** byte-identical queued and alone, for both prompts.
- **Decode:** first requests on a fresh server 70.6 tok/s (code) and 61.8 (story). Repeated on the
  warm cache: **147.2** and **97.5**. MTP acceptance 92.6 % and 61.6 %.

**Two lanes (sanity check only).** `infernix-serve --max-concurrency 2 --spec mtp --draft-tokens 3
--lm-head-draft`, temperature 0, 200 tokens, code and story requests sent together on a cold
server, then each alone.

- **Robustness:** no crash or hang. MTP acceptance was 94.8 % and 63.0 %.
- **Output equality (explained: verification block width).** The code answer is identical
  concurrent and alone; the story diverges after ~30 tokens at a near-tie. Plain C = 2 decode is
  identical for both prompts. With n-gram copy proposals and no MTP (`--ngram-draft-tokens 7`),
  the story also diverges, and it had no drafts of its own: it only rode in the code row's
  verification rounds. So the cause predates MTP and is not corruption.
  - **Cause, probed at the op.** On the Flash-Next dense Q8 shapes, an output column is
    bit-identical to the same column computed alone for blocks of 1-8 columns, which use the SIMT
    routes. From 9 columns the MMA tiles take over, and 40-45 % of BF16 outputs differ by
    rounding. That holds on all eight classes probed ([16384,2560] … [10240,320]; T = 9, 10, 16,
    40).
  - **Why C = 1 matches.** One row verifies at most 8 columns: MTP with up to 4 drafts uses at
    most 5, n-gram 8. Two rows verify 2 × W columns, so they cross into MMA.
  - **Contract.** NInfer promises no bit-identity across verification widths
    ([n-gram guide](../ngram.md)), so this is the documented behaviour, not a defect. The probe
    does not exclude other width-dependent routes (QSA verification attention, GDN record) from
    also contributing.
  - **Possible change.** Verification blocks of 9-16 columns could run as two SIMT launches of
    ≤ 8 columns to keep concurrency-invariant greedy output. Each launch reads the dense Q8
    weights again. Not done: it buys determinism, not quality, and two lanes are slower than one
    anyway.
- **Throughput:** concurrent decode reached 24.9 + 21.7 tok/s on the cold cache, below one cold
  request (~55-67 tok/s). Two unrelated sequences double the distinct experts per round. C > 1
  under MTP is not tuned; C = 1 is the recommended setting for now.

**Next.** Five tracks, planned and adversarially reviewed on 2026-10-04, are in §19.3 with their
decisions, steps, tests and exit criteria. Listed by track with expected gains (est.); the work and
merge order is §19.3.0's:

1. **Prefix cache** (§19.3.1), P0-P6, ~12 days: a 32K chat turn from 50-75 s to ~2.4-2.8 s to first
   token; P7 (CPU assist) if its gate passes.
2. **Concurrency gate and diagnostics** (§19.3.5, S1-S2), ~1 day: C = 2 with MTP from ~40-55 to
   ~68-72 tok/s cold.
3. **Q8 kernels** (§19.3.3), Phase 0 and 1a, ~5 days, bit-exact at T ≤ 8: −0.7 to −0.9 ms per main
   forward at T = 1 (−1.1 to −1.5 ms at W = 4), and no T = 9-64 MMA cliff (~14-20 ms per forward,
   inferred); then 1b and 2.
4. **N-gram reads** (§19.3.4), S0-S2, ~3 days: +0.7-1.7 % on first-time MTP text.
5. **Vision** (§19.3.2), ~12.5-13.5 days, a new capability; V3 (the load-serial ABA fix) merges
   early, and V0, V1, V3 and V5 can be developed in parallel (§19.3.0).
6. **Concurrency S3-S5** (§19.3.5), ~4-4.5 days: C = 1 MTP fill 69/63 → 89/78 tok/s and C = 4 fill
   85 → 107 (model, DRAM 70 / 55 GB/s; S3 gains nothing at C = 1 at 55 GB/s); S6 + S7 last, only if
   they beat the gate at C = 2 and C = 4 in every regime and the user accepts concurrency-dependent
   output at near-ties.

Tracks share code; the rules and merge order are in §19.3.0 (risk in §21). Not in a track yet: the
A4 tensor-core wide route for experts with > 8 columns (§13; `gate_up` + `down` are 45 % of a 4K
prompt); frames lent to the prefill arena (§9.2); BF16 tensor-core routes for text prefill, whose
QSA QKVG group runs on the scalar fallback at 13.2 TFLOP/s (measured; ~0.45 s per 8K prefill, est.;
decided (default, 2026-10-04): a separate task with its own quality gate, after vision V0); the QSA
QKVG split from the indexer (~1 % of decode; reconversion); plain-round commit overlap
(`after_round` ~0.5 ms of host time with no GPU work queued); the CPU miss service's idle
`sleep_for(50 µs)` (≥ 1 ms on Windows, est.).

**For the user:** an x16 link would roughly double miss bandwidth.

### 19.3 Next-phase designs (2026-10-04)

On 2026-10-04 the user asked for five next items, designed in parallel (fan-out allowed): a prefix
cache, vision with the tower offloaded, faster dense Q8 decode kernels for the small Flash-Next
shapes, n-gram row reads overlapped with the round, and speculative throughput at C > 1. Each was
designed read-only against `46a56fc8f` and revised after adversarial review (two rounds for the
prefix cache and vision, one for the other three), whose findings were checked in code. Nothing is
implemented. Numbers are **measured** (with the source), **model** (`concurrency_model.py`, about
±30 %) or **estimated**. The plans and their review dispositions are in `local/workdirs/fn/plans/`.
§19.3.6 adds the options taken from Strata v0.1.39 on 2026-10-04 (`strata-review.md`,
`strata-amendments.md` there) and names the steps they amend.
Step and measurement names are local to each track (prefix P0-P9 and M0-M9, vision V0-V8, VT1-VT9
and VM1-VM8, kernels Phase 0-3 and M0-M4, n-gram S0-S4 and W1-W7, concurrency S1-S7 and the deferred
ragged layout P6), not the milestones of §19; outside their own section they are qualified with the
track name.

**Can Qwen4Exp use Infernix's existing prefix cache?** Yes, with no new system but not as-is: a
Qwen4Exp binding plus four generic index and cost extensions, estimated to bring a 32K chat turn's
first token from 50-75 s to ~2.4-2.8 s with decode unchanged (§19.3.1, Answer).

### 19.3.0 Order of work and integration

#### Tracks

Efforts are estimates in engineer-days; gains are estimates unless marked. Concurrency pairs such
as 75/64 are model results (`concurrency_model.py`, ±30 %) at 70/55 GB/s of host DRAM bandwidth;
figures that include verification rounds are already scaled by ~0.8 for the model's measured
optimism (§19.3.5), plain-round figures are not. The Strata v0.1.39 review (2026-10-04) added
steps to the kernels, n-gram, concurrency and prefix tracks and proposed three items outside them
(§19.3.6). The rows below include those additions.

| Track | Goal | Steps | Effort | Expected gain | Main risk | Depends on |
|---|---|---|---|---|---|---|
| **Prefix cache** (§19.3.1) | Resume a turn from cached KV blocks and state snapshots instead of re-prefilling the conversation | P0 prep (`mtp_cells`, CPU-served calls ≤ 8 columns, QSA tails fix); P1 generic index and cost extensions; P2 state image; P3 binding and orchestration; P4 per-layer pipelining, copy-engine order; P5 tap-aware call planner; P6 Engine, serve, docs; persistence (`--prefix-cache-file`, X12) after P6; P7 prefill CPU assist (gated by prefix M6); P8 C > 1 extras (after the concurrency track); P9 shared Host-tier helper | P0-P6 ≈ 12 d (including persistence), with P7 ≈ 15 d; P8 and P9 1.5 d each; ~3-4K lines + ~1.3K test lines | 32K chat turn 2.4-2.8 s vs 50-75 s; 50-token tool result after a 32K echo 0.85-0.95 s (0.3-0.45 s with P7); decode unchanged | MTP drafter-state edge cases, invisible at C = 1 greedy (bitwise drafter oracle D1); copy-engine FIFO stalls; pinned Host RAM | `program_impl.h` split (`54deaa2bc`, done); concurrency S3-S5 before P7, the whole concurrency track before P8; vision's prompt RoPE state for image suffixes |
| **Vision** (§19.3.2) | Images and video for Qwen4Exp, tower weights (898 MB BF16) in pinned host RAM, streamed per encode window | V0 tensor-core BF16 routes for 8 tower shapes; V1 QSA M-RoPE op; V2 RoPE plumbing (`[T,3]` in every call); V3 load serial (ABA fix); V4 frame lending; V5 shared tower refactor, loading, flags; V6 Program window; V7 FP64 references and real tests; V8 measurements and docs | ≈ 12.5-13.5 d | No reconversion. Tower GEMMs for a 256-token image 6-9 ms vs 65 ms on today's scalar fallback; offload adds ~22-30 ms per small image, ~2 ms at ≥ 1K image tokens; resident weights would cost 325 frames ≈ 1.3 % of all decode | M-RoPE plumbing touches every attention call (text logits must stay bitwise equal) and is invisible below 2,051 tokens; three-stream ordering with lent frames; Qwen3.5 and Quasar requalification | Prefix P0 tails fix in `qsa.cu`; n-gram S1 de-duplication; concurrency S4 in the frame pool |
| **Q8 kernels** (§19.3.3) | Dense Q8 decode near DRAM bandwidth at T ≤ 8 (small shapes run at 0.6-0.7 TB/s, **measured**) and off the MMA-tile cliff at T = 9-64 | Phase 0 bit-exact non-Q8 fixes (one small dispatch kernel with the route log, router reorder, RMSNorm d = 10240; E2M1 decode in registers in the expert kernels, the GDN control GEMV on more SMs, QSA attention accumulators in registers); 1a-i register-streamed K1 for all 10 shapes, bit-exact at T ≤ 8; 1a-ii T = 9-64; 1b fused mixer, inject and SwiGLU (rounding change); 2 PDL, L2 warming in `cpu_wait`, `Weight` L2 class, the shared expert on its own graph branch; 3 tensor-core K2 only on trigger | 11.5-15 d, + 2-3 d for Phase 3 | Per main forward: Phase 0 −0.28 to −0.4 ms plain, −0.33 to −0.6 ms MTP, plus −0.18 to −0.42 ms at T = 1 and −0.7 to −1.6 ms per MTP round from the 2026-10-04 items; 1a −0.41 to −0.51 ms at T = 1 (of 3.09), −0.75 to −0.91 at W = 4 (of 3.62), ≈ −11 to −17 ms at T = 9-16; 1b −0.55 to −0.7 ms; 2 −0.15 to −0.45 ms. End to end typically 50-80 % of kernel deltas | The T > 8 case rests on one measured point ([2560,2560], 31.1 µs at 12-16 columns); K1 bit-exactness; 1b moves the tg512 text, hit-rate and acceptance baselines | Nothing for Phase 0 and 1a; `moe_layer.cu` with concurrency S3 and S6 |
| **n-gram overlap** (§19.3.4) | Hide PLE row reads and the after-draft host turnaround | S0 attribution, flush probe (three sites) and a temporary draft-probability log; S1 deduplicated, ring-buffered volume reads; S1b multi-issuer probe; S2 gated verification (rows read after the graph launch); S3 device-assembled verification (conditional); S4d cross-chunk prefill reads after S1; S4a-c parked follow-ons | ~7-7.5 d + 5-6 h GPU | With S2 + S3: warm tg512 MTP +0.5-1.9 %, first-time text MTP +1.2-3.4 %; S2 alone ~0-0.15 % warm and +0.7-1.7 % first-time (S3 is built only past its threshold); plain 0-0.3 % (S0b only); cold prefill −87 to −148 ms per 4K chunk | Gains near run-to-run noise (±0.4-0.75 tok/s); long gate waits under WDDM; host stalls move inside the graph | Core `device` primitives (additive); `verify()` and `mtp_draft` shared with concurrency S1, S6, S7 |
| **Concurrency** (§19.3.5) | C > 1 with speculation no slower than plain C > 1, and faster cold C = 1 | S1 per-round speculation gate (draft only when B = 1); S2 diagnostics; S3 CPU job capacity 8 → 32 with one swept cap (12-24 accepted if faster), parallel plan and wait; S4 fill-phase landing; S4b warm start (saved LFRU state, bulk fill at load); S5 fork at every width; S8 decode share between prefill chunks; S6 masked fillers + S7 joint draft lengths, only if they beat the gate; ragged layout (concurrency P6) deferred | S1-S5 5-6 d + ~5 h GPU; S4b 1-1.5 d, S8 0.5-1 d + ~1.5 h GPU; S6-S7 2.5-3 d more | Model: C = 2 fill ~43 → 75/64 (S1) → 80/69 (S1-S4) → 97/85 (S1-S7); C = 2 warm ~95 → 136/122 → 163/158; C = 1 MTP fill 69/63 → 89/78 (S3, S4); S4b: the first ~200-500 tokens after a restart near the warm rate when the saved state matches the workload (est.); S8: a decoding lane beside a long cold prompt gets ~1.2-3 s of decode per chunk instead of one round per chunk (est.) | Model uncertainty; CPU and PCIe paths share DRAM, so S3 gains nothing at C = 1 at 55 GB/s; S7 gives up C > 1 = C = 1 greedy equality at near-ties (MMA tiles above 8 columns) | Owns `cpu_plan`, `cpu_wait`, `stage_kernel`; frame pool with vision lending; kernels 1a-ii if S7 lands |

#### Shared code and conflicts

| Shared code | What the tracks change | Rule |
|---|---|---|
| `ProgramImpl` (`program.cpp`) | All five | Its declaration moved to `program/program_impl.h` on the base branch (`54deaa2bc`, which is prefix P0 step 1, so prefix P0 starts at its step 2), and every track branches from that commit. New code goes into track-owned files (`prefix/`, `vision_window`, `rope_positions`, `draft_policy`) |
| `decode`, `verify`, `mtp_draft`, `commit`; `kWidthCost` | Concurrency S1 gate, S6/S7 joint policy; n-gram S2/S3 restructure `verify` and `mtp_draft` and re-measure the width cost; prefix `mtp_cells` and the cancelled-round repair; vision RoPE staging for verify, drafts and catch-up | Concurrency S1 first, then n-gram S2/S3; vision V2 rebases; S6/S7 last. A change to round cost re-measures W = 3/4/5 |
| Io layout and uploads | Vision `rope`/`block_rope` before `ngram`; n-gram gate word; prefix prefill uploads of `io_prefix(width)`, with `upload_pinned` up to 64 columns | Vision's io-placement assertion runs after every merge |
| `execution/forward.{h,cpp}` | Prefix `layer_waits`; vision scatter, M-RoPE, MTP visual input; kernels `mix`, inject, `linear_swiglu`; n-gram gate before `ple_embed`; concurrency counters, landing, verify extents | Merge order below; the text bit-identity gate after each merge |
| `moe_layer.cu`, `offloaded_sparse_moe.h`, `miss_request.h`, `miss_service` | Kernels: small dispatch, `moe_combine` inject form, `L2Warm` in `cpu_wait`. Concurrency: cap 32, block-parallel plan that publishes only chosen columns, multi-CTA `cpu_wait`, landing, fork lists, route mask. Prefix P7: wide-call gather and per-layer cap J | Concurrency owns plan, wait and stage: S3-S5 land before kernels Phase 2 and prefix P7, which rebase onto them, and one `cpu_wait` grid serves both copying and L2 warming. Kernels Phase 0 lands before S6, which then skips id −1 in the small dispatch too |
| `expert_cache`, `expert_residency` | Vision load serial and lending; concurrency all-column stats, device counters, landing reservations, S4b warm-start seed and save | Vision V3 first (the ABA hole is latent today); whichever of lending and landing merges second adds a test that lends during the fill phase; S4b follows S4 and keys the fill-phase rule on seeded frames too |
| `src/ops/qsa/` | Prefix tails fix (`tail_kernel`, `commit_tail_kernel`); vision M-RoPE (`qsa_index_query`, `pool_kernel`, `QsaBatch`); kernels Phase 0 attention registers (`attention_kernel`); the proposed long-context selection (§19.3.6, `select_kernel`) | Prefix P0 first, and P0 adds the new FP64 QSA oracle test with the tails case (4-blocks completed inside a call or commit); vision V1 extends that test with VT2's M-RoPE cases rather than creating a second QSA test, and keeps the tails case; kernels Phase 0 and the selection work rerun that test and do not create another |
| Engine scheduler (`src/runtime/engine/scheduler.h`) | Concurrency S8 decode budget between prefill units | Per-model default: Qwen3.5 keeps today's 1:1 alternation (share 0) and its scheduler tests unchanged |
| `NgramVolume::read_rows` | n-gram S1 dedupe, ring, counters; vision V6 in-call de-duplication | n-gram S1 owns it; vision V6 drops its copy |
| `src/core/device.{h,cu}` | n-gram `upload_pinned_when`, `publish_pinned_word`; prefix `download_pinned` (only if prefix M0 confirms D2H FIFO stalls) | Additive |
| Copy-engine FIFO | Prefix restores and write-through; vision weight stream; promotions; landing | Per-round small H2D copies use `upload_pinned`; bulk H2D copies queue after the inputs of the call they feed; bulk D2H copy-outs (prefix endpoint images, write-through) never precede a round's readbacks, which include n-gram S3's drafts D2H and concurrency S2's counter download: prefix M0 decides `download_pinned` for them or ≤ 4 MB write-through pacing |
| Host RAM (96 GB) | Model 64.5 GiB pinned; vision +856 MiB weights and ≥ 0.4 GB media; prefix Host tier 4 GiB; n-gram row cache ~168 MB; 8 GiB loader floor | Prefix resolves its tier after Vision is loaded, with a media reserve (§19.3.1) |

#### Execution

- **Worktrees.** One worktree and branch per track under `E:\NInfer-V3\local\workdirs`:
  `NInfer-V3-fn-prefix`, `-fn-vision`, `-fn-kernels`, `-fn-ngram`, `-fn-conc`; the base branch
  `claude/wonderful-ritchie-65xtnh` stays in `NInfer-V3-flashnext`.
- **GPU and builds.** Every GPU run holds `E:\NInfer-V3\local\gpu.lock` as a hidden detached job,
  its rig dry-run validated first; model runs wait for ≥ 72 GiB free RAM, and runs with the prefix
  Host tier wait for ≥ 77 GiB (64.5 GiB model + 8 GiB loader floor + 4 GiB tier), ≥ 78 GiB with
  Vision as well (+0.84 GiB tower + the media reserve, §19.3.1); `fn\tools\gpu.ps1 -Model` (fixed at
  72 GiB today) must take that threshold. Timing runs also hold `fn\locks\quiet.lock`: no build
  starts and running builds finish first, since a build disturbs the CPU-served experts. At most two
  builds run at once, one during a model run (≥ 20 GiB free). Helpers: `fn\tools\gpu.ps1` (GPU lock,
  `-Model`, `-Bench`, writes `<log>.exit`) and `fn\tools\build.ps1` (a worktree's `build-windows`
  targets under a build slot).
- **Volume probe.** n-gram S1b (CPU only, saturates `G:` for a few minutes) runs while holding
  `gpu.lock` and `fn\locks\quiet.lock`, so no model run reads the volume meanwhile (§19.3.4).
- **Artifacts.** `E:\NInfer-V3\out\flash-next\qwen3_8_flash_next_nvfp4_dense8m.ninfer`, volume
  `G:\infernix\qwen3_8_flash_next_nvfp4.ninfer.ngram`, INT8 KV, unless a plan says otherwise.

#### Merge order and documentation

Tracks merge onto the base branch when ready, within the rules above; each merge rebases, builds
and re-runs the global gates on the merged tree. Rounding changes come late: kernels 1a-ii (item 4)
keeps C = 1 greedy ids but moves MTP acceptance, so tracks re-measure their MTP speed baselines
after it; vision V0 (item 5) moves only the Qwen3.5 and Quasar towers; Phase 1b (item 6) moves the
tg512 text, after which every track re-baselines its identical-id workloads. Expected order:

1. Concurrency S1 + S2: the default C > 1 policy and the diagnostics the others measure with.
2. Kernels Phase 0 and 1a-i (bit-exact); vision V3; prefix P0 and P1.
3. n-gram S0-S2 and S4d; concurrency S3-S5, S4b and S8 (S8 after S1 + S2).
4. Kernels 1a-ii (rounds the C = 1 drafter catch-up and C ≥ 2 verification).
5. Prefix P2-P6; vision V0-V2 and V4-V8 (V0 changes the Qwen3.5 and Quasar tower routes).
6. Kernels Phase 1b (rounding change), then re-baselining, then Phase 2 (PDL for K1, the fused Ops
   and the router; `cpu_wait` L2 warming; `Weight::l2`), which §19.3.3 makes depend on 1b.
7. Once their gates pass: prefix persistence (`--prefix-cache-file`, X12), n-gram S3, concurrency
   S6 + S7 (also after the user's determinism decision, §19.3.5), prefix P7 and P8, Q8 Phase 3 and
   n-gram S4 items only on their triggers; prefix P9 last.

**Documentation.** In its branch, a track edits only its own §19.3.x, its own subsection of the
user guide (`docs/qwen3_8-flash-next.md`) and documents or parts no other track edits (the Hybrid
spec's Qwen4Exp binding section, its own curves in `linear-tuning.md`, its own entries in
`tests/README.md`; Op contract comments change with their code under the shared-code rules above).
Every other section of this file (§3-§18, §19.2's results and "Next" list, §20, §21), this
§19.3.0, `docs/cli.md`, `docs/serving.md` and README change on the base branch at merge time, by
the merging track, so parallel edits never meet.

#### Acceptance rules for every track

- **Bit-exact claims.** Where a step claims bit-exactness, C = 1 greedy token ids are unchanged on
  tg512 (bench corpus) and the code and story CLI prompts, plain and MTP; a difference is a defect,
  not noise. MTP ids equal plain ids at C = 1 after every step; with the speculation gate,
  concurrent output at C ≤ 8 is byte-identical to each request alone.
- **Rounding changes.** Kernels 1a-ii: FP64-oracle conformance at T = 9-64 and column invariance
  to 16/64; C = 1 ids identical, MTP acceptance reported; forward-real errors and perplexity no
  worse, since prefill remainder calls of 9-64 columns change route too. Kernels 1b: forward-real
  residual and logit errors no larger, and perplexity on the frozen texts not higher beyond noise
  (§16.5). Vision V0: the 8 tower shapes against the FP64 Linear oracle (VT1), Qwen3.5
  requalified (VT8). Concurrency S7: only with the user's determinism decision (§19.3.5).
- **Speed** is judged on prompts whose greedy ids stay identical between the arms, with tg512 and
  its hit rate reported beside them (§19.2 measurement rule); n-gram-sensitive changes on
  n-gram-cold runs, since tg512 repetitions and repeated serve requests are n-gram-warm (§19.3.4).
  Arms run ABBA on one binary with temporary toggles; median, range and worst case are reported.
- **Every adverse result is recorded**, with rejected approaches and failed attempts: in the
  track's §19.3.x while on its branch, and in §19.2 (status and measurements) when the track
  merges, as §19.3.2-§19.3.5 and §20 specify. Acceptance criteria are fixed before measuring and
  not changed after (§1.3).

#### Default decisions

These defaults were set on 2026-10-04, before any measurement; the user may override any of them.

| Track | Decided (default, 2026-10-04) | Alternative |
|---|---|---|
| Prefix | Host tier 4 GiB by default; `--host-context-mib` may set more, with the startup guard | 8-12 GiB for ~100K-token agentic sessions, if ≥ 16 GiB stays free after model and Vision |
| Prefix | Zero Device snapshot slots (needs the P1 index extension) | Qwen3.5-style C + 1: ~84 frames at C = 1, ≈ 0.35 % decode |
| Prefix | Exactness as NInfer: a resume equals the capturing request's own computation, may differ from an uncached run at near-ties; `--no-prefix-reuse` for strict reproducibility | Cold-run equality: resume only from grid-aligned flexible taps of cold pure-prefill lineages with no exact tap before F (E3) and refuse every other resume, endpoints and exact taps included, so echo turns lose endpoint reuse |
| Prefix | P7 prefill CPU assist implemented, kept only if prefix M6 shows ≥ 10 % TTFT gain for 64-1,024-token suffixes, no regression ≥ 2,048, tg512 unchanged | No P7 (it changes shared offloaded-MoE Op code) |
| Prefix | Exact Structural and Explicit taps kept even when they add a call (~1.3 s once per new preamble) | Demote them to flexible when they cost > 0.1 s |
| Prefix | Generation-opener tap kept after endpoint resumes (~25-40 ms and 116 MB Host per turn), with a fallback counter | Drop it as Qwen3.5 does; revisit if fallbacks stay ≈ 0 |
| Prefix | Not now: a decode-time `</think>` tap; idle-time re-render warming (a product change, ~1-2 s per chat turn, needs preemptible background lane work) | A flexible `</think>` tap (~4.5 ms stall and 116 MB Host per turn); warming designed as a separate product change |
| Prefix | QSA tails fix in P0 (drafter-only; exact prepends, fixes today's step-0 pooled-key corruption) | Leave it and count stale pooled keys |
| Prefix | `download_pinned` for per-round readbacks only if prefix M0 confirms D2H FIFO stalls | Pace write-through in ≤ 4 MB pieces |
| Prefix | Later: the shared Host-tier helper P9 (sibling binding now); persistence (`--prefix-cache-file`) after P6, since any rebuild invalidates it; C > 1 extras P8 (in-flight coalescing, blocked-head prefetch) after the concurrency track | Each now; P9 touches Qwen3.5 and needs a 27B rerun |
| Prefix | Serving prefill chunk decided after prefix M9 (4096 costs ~491 frames, ~4.6 % tg512, against 1024), fixed per Engine and gated by the long-prompt chunk check (§19.3.1 M9) | Fix 2048 or 4096 now |
| Vision | `--vision-offload auto\|on\|off`, `auto` = on for Flash-Next, off for Qwen3.5 | A model-dependent `on\|off`, or off everywhere (325 frames resident) |
| Vision | Keep the 16,384 merged-token per-item cap (~25-40 s prefill plus a 5-9 s tower at that size) | A lower default such as 4,096, trading resolution for TTFT |
| Vision | MTP visual embeddings follow vLLM and Qwen3.5 | Placeholder embeddings; only acceptance changes |
| Vision | Video in scope (same code, one 4-frame smoke test) | Images only first |
| Vision | Frame floor 25 % (frames vision may never borrow); `kVisionStepSeconds` 0.25 s initially | Another floor; shorter steps (less stall for other lanes, more steps) |
| Vision | V0 registers all 8 shapes, including Quasar's merger [5120,4608]; Qwen3.5 vision is requalified | The 7 Flash-Next shapes only; Quasar's merger fc2 [5120,4608] stays on the fallback |
| Vision | Tensor-core routes for Flash-Next's text BF16 prefill projections (~0.45 s per 8K prefill) are a separate later task with its own quality gate | Fold them into V0 (text prefill numerics change) |
| Q8 | Phase 1b rounding changes accepted; all are more precise than HEAD and upstream | Stop at Phases 0, 1a and 2 (bit-exact except 1a-ii's T = 9-64 rounding change) |
| Q8 | The two-kernel split-K mixer of §8.3, with the norm statistic computed inside K1a | The three-kernel K1 mixer, or §8.3's producer-epilogue sums |
| Q8 | T = 17-64 chosen per shape by the sweep (MMA tile or 16-column K1 slices) | MMA tiles above 16 columns |
| Q8 | Recipe A served by the fused Ops' composed route (no new BF16 kernels, today's speed) | Fused BF16 kernels for recipe A |
| Q8 | No separate full C = 2 invariance task: with the speculation gate, C ≤ 8 rounds stay at T ≤ 8, where dense Q8 is column-invariant, so C ≤ 8 output is expected to equal C = 1 (inference; concurrency campaign step 1 checks byte identity; revisit if S7 replaces the gate) | Invariance of BF16 dense shapes, QSA verification attention and the GDN record |
| Q8 | Phase 0 bit-exact non-Q8 fixes included in this track | A separate track |
| Q8 | Phase 2: PDL for K1, the fused Ops and the router; a `Weight` L2 class (`Stream`/`Reuse`) added to Core, drafter evict-last measured | PDL for every Flash-Next decode kernel; no Core field |
| Q8 | Thread-block clusters allowed for HC down if Phase 3 triggers | No clusters |
| n-gram | Gate trap 120 s, verified once with a 3-5 s producer delay while the GPU is otherwise idle; 2 s if WDDM preemption fails | 2 s like `cpu_wait`: a 2-120 s NVMe stall then loses the context |
| n-gram | Internal process-wide fault seam (`qwen4_exp::testing::set_ngram_faults`) for the engine test | Env var read at Program construction (one load per case), or primitive tests only |
| n-gram | S3 built only if W1's after-draft GPU idle is still ≥ 0.15 ms per round after the S0b flush probe and S2 | Another threshold, or build it (~2 days) regardless |
| n-gram | S4a-c (mailbox, plain gate, multi-issuer reads) parked until their triggers fire; S4d built after S1 as the cross-chunk prefill read (§19.3.6) | Build some now; S4d as the within-chunk gate |
| n-gram | S1b runs while holding `gpu.lock` and `fn\locks\quiet.lock` | A separately agreed window |
| n-gram | §12.3 and §12.4 rewritten to the host-hashed, gated design, as the single authority | Keep the L0 device cache and NVMe agent as the target |
| Concurrency | Optimise C = 2 first, check C = 4 | C = 4, or C ≥ 7, as the primary target (then S5 and campaign step 5 extend to C = 8) |
| Concurrency | The speculation gate (S1) is the default C > 1 policy; S6 + S7 replace it only if they beat it at C = 2 and C = 4 in fill, turnover and warm **and** the user accepts concurrency-dependent greedy output at near-ties (open; asked at the decision point after S5) | Never speculate at B ≥ 2; or split wide verification blocks into ≤ 8-column SIMT launches (one extra dense pass per round) to keep equality with C = 1 |
| Concurrency | Higher CPU caps (12-24 jobs per layer call) accepted if measured faster, despite more CPU load and power; cap, divisor and worker count stay internal defaults, and the gate stays internal until S7 decides | Keep 8; CLI options (`--cpu-expert-jobs` and so on) and a documented gate option |
| Concurrency | Fill-phase landing (S4): every staged miss is cached while frames are free | Keep LFRU promotions within the budget |
| Concurrency | S5 compacted grids, subject to the C = 1 no-regression gates | Fork at every width without the lists |
| Concurrency | S4b warm start on by default from the user's own saved state; a shipped fallback profile only if the replay gate passes (§19.3.5) | Start empty, as today |
| Concurrency | S8 decode share for Qwen4Exp at C ≥ 2, its value chosen by S8's measurement (0.5 proposed); Qwen3.5 keeps 1:1 | Today's 1:1 alternation for both |

### 19.3.1 Prefix cache

Design, reviewed twice adversarially, nothing implemented (designed against `46a56fc8f`, 2026-10-04;
`file:line` at that commit; the base split `54deaa2bc` moved `ProgramImpl` into
`program/program_impl.h`, so a `program.cpp:N` citation is now `program_impl.h:N−2` and Files lists
naming `program.cpp` mean `program_impl.h`). **Measured** means §19.2 unless stated ("Hybrid spec"
is [hybrid-prefix-cache-spec.md](hybrid-prefix-cache-spec.md), the existing cache's authority);
**estimated** means arithmetic, not executed. Local IDs: MTP rules MR1-MR9, guarantees E1-E6, phases
P0-P9, tests U/X/D1, measurements M0-M9 (not the §19 milestones). Recipe B dense8m, INT8 KV, C = 1,
chunk 4096 unless stated.

#### Answer

**Qwen4Exp reuses Infernix's Hybrid prefix cache**: no new system, but a Qwen4Exp physical binding
(as Qwen3.5 has) and four small generic extensions to the index and cost model.

| Layer | Verdict | Evidence |
|---|---|---|
| Engine surface (`HybridResourceManager`; `EngineCore` admission, progress, terminal flow; protocol usage; logs) | **Reused unchanged**; Qwen4Exp already runs on it with stub hooks | `engine.cpp:161-163`; `qwen4_exp/program/program.h:295-302` |
| Frontend keys and hints (`block_hashes`, `block_extras`, `tap_hints`, generation opener) | **Reused unchanged** (`qwen3_5::make_frontend`) | `qwen4_exp_instance.cpp:44-55` |
| Index, block hash, tap planner, cost model, Host-restore planner (`src/runtime/prefix_cache/*`) | **Reused with four extensions** | `publish_snapshot` needs a staging slot and a Device tail (`prefix_index.cpp:889-896`); Host-only nodes adopt the inserting page (`prefix_index.cpp:485-496`); cost is `chunks × chunk_seconds` (`cost.h`) |
| Physical binding (page references, slabs, write-through, restore batches, persistence) | **New sibling of `HybridPrefixCache`**, same mechanisms | That binds Qwen3.5's `LogicalKVPageStore`/`StateImageStore` (`hybrid_cache.h:91-96`); Qwen4Exp has a `DeviceKVPagePool` with move-only leases (`paged_kv_cache.h:155-178`) and per-lane slots (`program.cpp:188-213`) |
| Orchestration (quote, stage, activate, taps, publication, endpoint, release) | **New**, a line-by-line mimic of `qwen3_5/program/prefix/hybrid_program.cpp` | — |

Four Qwen4Exp facts change the binding, not the design:

1. **VRAM is expert frames** (~362 per GB, §15.1): cached blocks use idle KV-pool pages, snapshots
   go to pinned Host RAM, and the only new VRAM is one copy-on-write page per lane (0.34 frames).
2. **A call's fixed cost is staging its distinct non-resident experts** (~1.28 s from ≳ 200
   tokens, ~0.12 s GPU-staged at 5 tokens, ~25-40 ms CPU-served; estimated): exact taps must not
   add calls, tiny calls are CPU-served, and the post-hit lever is CPU service of thin experts (P7).
3. **The drafter's KV is a 13th layer of the page group** (`program.cpp:211-238`): a block's last
   MTP cell encodes the next block's first token, so that token is recorded per node and snapshot.
4. **Copy engines run FIFO across streams** (H2D measured; D2H by M0): restores queue behind their
   call's inputs, and bulk copy-outs never precede a round's readbacks.

**Expected** (estimated): first token of a 32K chat turn in ~2.4-2.8 s instead of 50-75 s; of a
50-token tool-result turn in ~0.85-0.95 s (~0.3-0.45 s with P7). Decode is unchanged: no frames are
taken, and fewer prefill calls churn the expert cache less.

#### Workload and costs

- **Turns.** OpenAI-chat and Anthropic clients (e.g. Qwen Code) resend the conversation plus a new
  user or tool turn; contexts are 8K-64K+.
- **Echo.** The template keeps reasoning in history by default (`preserve_thinking` undefined or
  true, and always for turns after the last user query), so history matches the last generation only
  if the client echoes it: Anthropic tool loops do; OpenAI clients often drop `reasoning_content`
  (the turn re-renders as `<think>\n\n</think>\n\n{content}`). Trimming, the `\n\n<tool_call>`
  prefix and `tojson` arguments (`chat_template.jinja:115-140`) can make even echoes diverge.
- **Reuse points.** An exact echo resumes from the previous **endpoint**; anything else from the
  previous **opener tap** (the generation prompt, or the first assistant turn after the last user
  query when history is rewritten, `chat_template.cpp:452-476`), kept after endpoint resumes.

| Quantity | Value |
|---|---|
| GPU-staged call of t tokens | `cost(t) ≈ 1.28 s × g(t) + 1.17 ms × t` + attention, `g(t) = 1 − (1 − ρ)^t`, ρ ≈ 10/512 (top-10 of 512): g(5) 0.09, g(50) 0.63, g(200) 0.98 (estimated; fitted to measured pp512 1.93 s, pp4096 6.09 s, chunk-1024 pp4096 9.92 s; M6 refits ρ) |
| CPU-served call ≤ 8 columns | ~25-40 ms decode-hot, more prompt-hot (≤ 8 CPU misses per layer, `miss_request.h:11`); needs `columns ≤ max_columns = lanes × max_width_` (`program.cpp:347`), which is 1 without speculation: hence P0 step 3 |
| 30K re-prefill today | 8 calls ≈ 10.2 s + 35 s ≈ 45 s + attention; 45-75 s at the measured 400-690 tok/s |
| INT8 KV incl. MTP (13 layers) | 14,560 B/token; 931,840 B per 64-token page group; Host slab 933,888 B (256-B packed planes, 4 KiB-aligned record) |
| State image | 115,673,088 B (110.3 MiB) = 124 slabs ≈ 41.8 frames; 125 slabs with a tail |
| Copies at ~26 GB/s (H2D measured 27.6; D2H assumed similar) | Image 4.45 ms; page 36 µs; 30K tokens (469 pages, 437 MB) 16.8 ms |
| Host enqueue | `copy_page` = 65 per-plane memcpys, ~0.1-0.3 ms; image ≈ 150 segments, ~0.3 ms |
| Restore vs recompute, 30K tokens | ~21 ms vs ~45-75 s, ≈ 2,000-3,500× (27B ≈ 300×, Hybrid spec §3.3) |

#### State and image layout

| State (48 layers = 36 GDN + 12 QSA) | Bytes | Lives in | Capture → restore |
|---|---|---|---|
| Main QSA K/V, scales, pooled index keys, 12 layers | 13,440 B/token | Page group, planes 0-59 | Block published at commit, written through at release → shared read-only; Host-only pages H2D per attention layer |
| MTP K/V + pooled keys (13th layer) | 1,120 B/token | Planes 60-64 | Same block → final MTP group (read only after layer 47) |
| GDN recurrent ×36 FP32 `[128,128,48]`, conv ×36 BF16 `[10240,3]` | 113.2 + 2.2 MB | Lane slot | Per layer part, one event per layer |
| PLE dilated-conv history `[10240,9]` | 184,320 B | `ple_backing_` | Part before layer 1 |
| QSA raw-key tails ×12 `[128,3]` | 9,216 B | `tails_backing_` | With its attention layer |
| MTP tails `[128,3]` + saved residual `[10240]` | 768 + 20,480 B | 13th tails slab; `mtp_saved_` | Final MTP group |
| Snapshot tail page (F % 64 ≠ 0) | 931,840 B | Lane page | Always D2H into a tail slab (an endpoint may also hand the page over as Device tail) → straight into the lane's first private page (H2D per layer, or D2D) |
| Snapshot meta | 64 B | Side table + 256-B header part | Read on the host at quote and activation |
| n-gram rows, proposer, sampling counts, RNG, expert residency, HC streams | — | Derivable or global | Not captured: rows re-hashed, proposer rebuilt (O(n)), counts zeroed |

Meta: `mtp_written` (cell F−1 final), `mtp_next` (`history[F]`, the token that cell encodes),
`lineage_echo` (captured after an endpoint resume), optional `mtp_accept[8]` (§11.3 EWMA, if
measured useful). Copies are byte copies, never requantized; experts are placement-invariant
(§16.2), so expert residency, which is not captured, affects only speed. **Image layout**
(`prefix/state_image.{h,cpp}`): `header | layers 0..47: GDN (recurrent, conv) or QSA tail, PLE
before layer 1 | MTP: saved, tails`, parts 256-B aligned; byte o is in slab `o / slab_bytes` (124
non-contiguous slabs, as Qwen3.5's `hybrid_host_layout.cpp`), one `cudaMemcpyAsync` per segment.
Optional Device slots use the same packed layout, allocated before the frames. Page records move by
plane range (`copy_to_host_records` / `copy_from_host_records`) into any page, cache-owned or
lane-private.

#### Generic extensions (`src/runtime/prefix_cache/`, ~170 lines; Qwen3.5 behaviour unchanged)

| # | Extension |
|---|---|
| 1 | **Host-born snapshots.** `reserve_host_image(tail, claim)` takes free slabs (the image's, then one tail slab when `tail`), GDSF-evicting like `begin_snapshot_host_fill` but never past `claim` (`estimate_priority`); `release_host_image` returns them. `publish_host_snapshot(anchor, frontier, tail, optional tail_device_id, reservation, kind)` publishes a landed image and tail (`device_slot = kNoId`, Host Resident); a non-empty `tail` requires the reservation's tail slab and `host_blocks`; a duplicate frees its slabs and Device tail and returns the existing snapshot with `created = false`. The Host tail is mandatory: `snapshot_valid`, `refresh_tail`, tail eviction and `begin_snapshot_host_fill` assume a Host-resident snapshot has one (`prefix_index.cpp:124-129, 369-379, 629-637, 734-736`). `check_invariants` counts reserved slabs |
| 2 | **Zero Device snapshot slots** are legal (`acquire_device_slot` returns `nullopt`) |
| 3 | **`insert_block(…, bool attach)`**: `false` pins an existing child without adopting `device_id`; the caller keeps its page. With `true` a Host-only child adopts it, and so does a child whose Device copy is unpinned, which `replaced_device_id` hands back for release (hybrid-prefix-cache-spec §7.5: a recomputed block never holds two pages). Qwen3.5 passes `true` |
| 4 | **Saturating call cost** (`cost.h`): ρ = `call_route_fraction` (default 0), `call_seconds(t) = chunk_seconds × (ρ > 0 ? 1 − (1 − ρ)^t : 1)`; `prefill_seconds` charges `⌊s/c⌋ × call_seconds(c) + call_seconds(s mod c)` plus token and attention terms. ρ = 0 is today's formula |

#### Binding `Qwen4ExpPrefixCache` (`src/models/qwen4_exp/program/prefix/`)

Files `prefix_cache.{h,cpp}`, `prefix_program.cpp`, `state_image.{h,cpp}`, `call_plan.{h,cpp}`,
`persist.cpp`. The binding owns `block id → DeviceKVPageLease`, `node → mtp_next` (4 B), the slab
pool, the write-through queue, restore batches with per-layer events, copy-outs, capture records,
the `PrefixIndexBackend` callbacks and `poll`/`drain`/`clear`.

- **Pages without refcounts.** Path pins keep a pinned node's Device copy (`prefix_index.h:24-27`);
  a lane holds private leases plus its path, and `insert_block` moves its lease to a new node or an
  attaching Host-only node. `mtp_draft`'s KV limit uses the mapped page count (`program.cpp:1403`).
- **Spare pages.** Pool = `⌈kv_tokens/64⌉ + C` (`kv_tokens` = `--kv-capacity` or
  `max_context × C`; `+ C` is new, `program.cpp:183-186`), shown by `memory()` and in `--help`.
- **Lane-resident reuse.** After an endpoint publishes with `created = true` the slot still holds
  its image: the lane records `resident = {snapshot, epoch}` (cleared by any reset, restore or
  prefill), an admission from it skips the image restore, and the quote lowers that candidate's
  `restore_bytes` by `image_bytes`. Duplicates (another lineage) get no shortcut.

**MTP rules.** Cell c pairs the residual at c with the token at c + 1, at KV slot c; pooling runs
when a column completes a 4-block, reading positions before the call start from the tails
(`qsa.cu:103-138`).

| Rule | Content |
|---|---|
| MR1 | `Lane::mtp_written` becomes `mtp_cells` (cells `[0, mtp_cells)` final). Settled invariant: `saved = residual(state_tokens − 1)`, `mtp_cells ∈ {state_tokens − 1, state_tokens}`; non-last prefill or forced calls and commit catch-ups give `state_tokens`, the last prompt call `state_tokens − 1` |
| MR2 | A cancelled plain round leaves `saved = residual(state_tokens − 2)` (`decode()` advances at submission, `program.cpp:599`, and the row skips catch-up). The commit's cancelled branch copies the row's `mtp_residuals_` column into `saved_column(lane)` and sets `mtp_cells = state_tokens − 1`. Cancelled verify rounds commit 0 columns: nothing to repair |
| MR3 | Finish flush: if `mtp_live && mtp_cells == state_tokens − 1 && history.size() > state_tokens`, one kv-only cell (cell `state_tokens − 1`, residual `saved`, token `history[state_tokens]`, ~0.2 ms), as the next round would write. Every normal endpoint is then `mtp_written`; one at F % 64 == 0 is never lost |
| MR4 | Snapshot at F: `mtp_written = (mtp_cells == F)`, `mtp_next = history[F]` (non-last prefill taps are written, `prompt[F]`). Node b: `mtp_next = publisher.history[64(b+1)]` |
| MR5 | Device adoption only on a matching next token: `attach = !mtp \|\| lane.history[64(b+1)] == node.mtp_next`, so a node's Device and Host copies are byte-identical; a mismatch stays private (`mtp_branch_mismatches`) |
| MR6 | Resume at F, k = ⌊F/64⌋. **F % 64 ≠ 0:** share `[0, k)`, private snapshot tail, `mtp_cells = F` if `mtp_written && prompt[F] == mtp_next`, else F − 1 (the prepend rewrites cell F−1 privately). **F % 64 == 0, `prompt[F] == node(k−1).mtp_next`:** share `[0, k)`, `mtp_cells = F` (writes are at cells ≥ F and 4 \| F, so page k−1 is untouched). **Mismatch:** share `[0, k−1)`, COW the anchor, `mtp_cells = F − 1`. COW is rare: echo endpoints, opener taps (`prompt[F] = <\|im_start\|>`) and structural taps match; mismatches count `mtp_continuation_mismatches` |
| MR7 | Prepends re-pool the 4-block holding F−1 from the tails, but `tail_kernel`/`commit_tail_kernel` write nothing when the last cell completes a 4-block (`qsa.cu:141-174`). The **P0 tails fix** always leaves the latest block's q % 4 < 3 keys (main layers never read them): prepends become exact, and today's drafter-only step-0 corruption goes away (after a verify catch-up ending at a block end, step 0 re-pools a committed MTP key from stale tails; beyond 2,051 tokens). Without it, mismatched resumes at F % 4 == 0 get a stale MTP key (counted) |
| MR8 | A lane continuing differently from block b's publisher reads one off-branch MTP cell per boundary: drafter-only, unrepairable without the residual, counted (`mtp_branch_mismatches`) |
| MR9 | Drafts, verification records and acceptance statistics are never cached |

#### Flows

| Flow | Mechanism |
|---|---|
| **Quote** (mirror of `hybrid_quote`) | (1) `poll()` lands copy-outs, publishes their snapshots, retires write-through. (2) **Drain endpoint copy-outs the prompt extends:** on a block-hash and tail match (O(F/64)), `cudaEventSynchronize` the last event (≤ 4.5 ms; endpoints go first on the transfer stream), then `poll()`; unrelated requests at C > 1 do not wait. (3) `match(tokens, block_hashes, block_extras, n)`; drop candidates with filling blocks or a frontier strictly inside a Vision span; lane-resident adjustment; `choose`. (4) **Fit:** COW source per MR6, `shared = k − (cow == anchor ? 1 : 0)`, need `(E − shared) + host_only([0, shared)) ≤ available_pages() + device_evictable_blocks() − evictable_on_path([0, k)) − (cow on Device and unpinned ? 1 : 0)`, replacing `available_pages() < pages` (`program.cpp:405`). Try the chosen candidate, the others deepest first, then the root; `PermanentlyInfeasible` only if the root does not fit an empty pool (`feasible`, `program.cpp:399-402`); if nothing fits and write-through is pending, `drain()` and retry, else `TemporarilyBlocked`. (5) Report `reusable_prompt_tokens = F`, `prefix_reuse_path = PrivateEndpoint` for an endpoint candidate, else `SharedStablePrefix`, `service_work_quanta` = suffix-plan calls + `effective_output − 1` |
| **Reserve, stage** (mirror of `hybrid_stage`; never waits) | Re-validate against a fresh match; `acquire_path([0, k))` (an anchor being COWed included), `pin_snapshot`. Evict the Device LRU (backed first) until `need` fits; `reserve(need)`: `E − shared` private leases into `lane.pages`, Host-only cache-owned leases into the binding (`begin_device_fill`). Only if something is Host-only, one restore batch for the first call: Host-only blocks → new cache pages; a Host-only COW source (tail or anchor slab) → **straight into the lane's first private page**, not adopted; a Host-only image → the lane slot (skipped when lane-resident). Forward-order groups, one event per layer (48), then a final MTP group (MTP planes, saved, tails; one event) |
| **Activate** (mirror of `hybrid_activate`) | Block table = shared handles + private leases (`tables_->publish(row, 0, span<const DeviceKVPageHandle>)`); a Device-resident COW source is copied with `copy_page` on the compute stream, then unpinned. Lane: `state_tokens = F`, `history = prompt`, `mtp_cells` per MR6, proposer rebuilt; `reset_slot` for a root admission, only `token_counts` zeroed on a resume; slot from lane-resident state, the restore batch or a Device slot (D2D). Taps: `plan_taps` without opener erasure, then the call planner. Path pins move to the lane; `note_hit`; supersession as in Qwen3.5. `reused_prompt_tokens = F` (today 0, `program.cpp:515`); diagnostics `cached_prefix_tokens`, `restored_host_bytes` |
| **Restore waits** | First call: table publish and io upload (compute stream) → `ev_inputs` → restore stream waits → restore groups → forward, so no restore precedes its call's inputs. `ForwardBatch::layer_waits` (`std::array<std::span<const cudaEvent_t>, 2>`): `[0]` restore events (48 + MTP), `[1]` this slot's copy-out events (tap image, endpoint in flight); `Forward::run` waits on `[i][l]` (i = 0 restore, 1 copy-out) just before layer l touches its state or pages, and on the MTP entries before the MTP chunk. Events are taken by ticket at call time (`take_layer_ready`, the use-after-free fix of Hybrid spec §16.4). A lane released before its first pass orders the compute stream after the batch. Until P4, the first call waits for the whole batch and the next call for the whole copy-out |
| **Block publication** (mirror of `hybrid_publish_blocks`) | Publish block b iff 64(b+1) ≤ P: P = `state_tokens` without a drafter; `mtp_cells` when finishing or in Prefill; `min(mtp_cells, state_tokens − 1)` in Decode (step 0 rewrites cell `state_tokens − 1`). **Invariant:** every remaining write of a live lane (main K/V, prepend, step 0, drafts, catch-up) is at ≥ P and its pooled key in page ⌊cell/64⌋ ≥ ⌊P/64⌋ (4 \| 64), so no published block is written; U3 covers P % 64 == 63, where the first revision was off by one. Insert at every commit (`commit`, `commit_verified`), prefill call and `append_forced`; record `mtp_next`, apply MR5, `publish_pending()`. An optional step-0 `write_cell` fix would make Decode P = `mtp_cells` |
| **Taps** (Host-born, no Device page) | (1) Check the Host-image budget (`estimate_priority`) and `reserve_host_image(tail = p % 64 ≠ 0, claim)`, else count `taps_skipped`. (2) After a compute event the transfer stream copies the partial page into the tail slab first (36 µs; columns < p are final, later writes hit only don't-care regions), then the image parts in forward order, one event per layer group: the next call's `layer_waits[1]`, with the tail-page event at layer 0; part l (≤ 3.2 MB, 0.12 ms) lands long before compute reaches layer l (estimated). (3) The anchor is already a node; on landing `poll` calls `publish_host_snapshot(…, Tap\|Boundary)` and supersedes, with no wait for the tail block (Qwen3.5 waits for it or for finish: for the opener, the whole decode). (4) With Device slots: the existing D2D path |
| **Finish, abort** | Abort only with no unit in flight (rows settle first, prefill calls synchronize). (1) MR3 flush, publish remaining blocks, abandon unpublishable taps and their reservations. (2) **Endpoint** iff F ≥ deepest snapshot + 64 and `mtp_cells ≥ 64⌊F/64⌋` (Qwen3.5's `backend_caught_up`); after a cancelled plain round at F % 64 == 0 it fails: `endpoint_skipped_mtp`. (3) `reserve_host_image`; tail D2H (36 µs), then image D2H (4.5 ms); the partial page is also handed over as Device tail (no copy). (4) On landing `publish_host_snapshot(…, Endpoint)`, supersede the resume snapshot, `lane.resident` only if `created` |
| **Release, eviction** | Write-through of Device-only path blocks after the endpoint copy-out; `release_path` deepest first into the Device LRU; private pages freed except a handed-over tail. `Disabled` requests (`allow_prefix_reuse = false`, warmup) insert nothing. A later lane-resident admission waits on the copy-out via `layer_waits[1]`. Eviction is the index's: Device LRU (backed first), Host GDSF (superseded first, dead-KV sweep), supersession, slots when D > 0. `hybrid_reclaim_device_kv` is never reached (leases never grow); `hybrid_prefetch` is P8 |

#### Checkpoints

Where snapshots are taken and what they cost (estimated):

| Checkpoint | Where | Kind | Compute | Copy | Notes |
|---|---|---|---|---|---|
| Generation opener | `n − ~5` (`<\|im_start\|>assistant\n<think>\n`), or the first assistant turn of a rewritten tool loop | Exact | CPU-served 5-token call, ~25-40 ms (~0.12 s GPU-staged before P0) | Tail 36 µs + image 4.5 ms, overlapping the tail call (P4) | Kept after endpoint resumes |
| Client breakpoint, `cache_control`, structural (end of tools, of the leading system block) | Anywhere | Exact | 0 if no new call, else Δ | Same | Explicit, Structural always kept; Automatic with Δ > 0.1 s become flexible |
| Prompt tail | Start of the final call | Flexible | 0 | Image 4.5 ms | Absorbed by the opener in chat (proximity rule) |
| Ladder `n − G·2^k`, G = max(4096, 2·chunk) | First call boundary ≥ p | Flexible | 0 | 4.5 ms + 116 MB Host each | ~log2(n/G) per cold long prompt |
| Endpoint (end of turn, consistent abort) | Frontier after MR3 | — | ~0.2 ms flush if pending | Tail + image from the idle slot | F ≥ deepest + 64, all full blocks below F published |
| Blocks | Every 64 final positions | — | O(1) host | 0.93 MB per new block at release (30K first turn: 437 MB ≈ 17 ms) | Mismatching duplicates and duplicates of a pinned Device copy stay private; an unpinned Device copy is replaced by the request's page. A request writes 1-3 images (0.12-0.35 GB) plus its new blocks, all D2H |

#### Copy-engine discipline

**Facts.** H2D copies share one FIFO copy engine across streams (measured: a 4-byte round input
waited ~3.4 ms behind promotions until `upload_pinned`). Rounds also make small D2H readbacks
(verify, drafts, sample: `program.cpp:1178-1180, 1457-1459, 1539-1540`; route log:
`expert_residency.cpp:101-102`); whether D2H queues behind bulk D2H on another stream is unverified
(a 27B trace shows restores overlapping a pass).

**Rules:**

1. Restores follow the first call's inputs.
2. Per lane, the transfer stream runs endpoint tail, endpoint image, then write-through.
3. M0 precedes P4. If D2H stalls, P4 adds a Core `download_pinned` beside `upload_pinned`
   (`src/core/device.{h,cu}`; a kernel storing into mapped pinned memory) for those readbacks at
   decode and verify widths (output-neutral, re-checked by tg512 and greedy ids; prefill keeps
   bulk copies). If `download_pinned` is rejected, `poll()` paces write-through in ≤ 4 MB pieces,
   at most one outstanding per round.
4. At C > 1 a restore still shares the x8 link with other lanes' staging (bandwidth; M4).

**Expert cache** (estimated): a 30K re-prefill makes ~6,100 promotions (16 per layer per call,
~65 % of 9,443 frames), replacing decode-hot experts with prompt-hot ones; a hit runs one call or
small calls on decode budgets (M2 (b)). If loans (§9.2) arrive, Free lanes' latest paths keep
their pages, other (Host-backed) cached blocks yield, and spare COW pages are never lent.

#### Exactness

| # | Guarantee | Basis |
|---|---|---|
| E1 | **Byte-exact storage**: restored pages and images equal the capturing lane's state at F, and a node's Device and Host copies are identical (MR5). Host, Device and lane-resident resumes of one snapshot generate identical tokens at any C and sampling mode, given the same schedule and seed | Byte copies of same-profile records; exact token + extra identity; lane-resident only for the lane's own snapshot |
| E2 | **Lineage equivalence**: a resume at F computes exactly what the capturing lineage computed up to F, then the resumed plan. A cold oracle must repeat every call boundary of that lineage below F (taps, demotions, earlier turns' decode and verify routes) and F | Ops depend on inputs, call width and start; experts are placement-invariant (§16.2). The existing "chunk-decomposition baseline" (Hybrid spec §13.3) |
| E3 | **Bit-identical to a cold prefill** for a flexible tap at a multiple of `--prefill-chunk` in a cold pure-prefill lineage, with no exact tap before F | Calls start at 0 and step by `chunk_`; the planner keeps that grid. X2 |
| E4 | **Not guaranteed**: equality with an unsplit cold prefill in general, or with re-prefilling an endpoint. Equally precise orders; greedy can flip at near-ties, like the C > 1 verification-width caveat (§19.2) | GDN token-recurrent below 16 columns and chunked relative to the call start above; QSA split-K up to 170 columns; Q8 SIMT vs MMA above 8; distinct decode and verify routes |
| E5 | **Speculation**: at C = 1 greedy, output after a restore equals plain decode whatever the drafter state (MR2, MR7, MR8 errors are invisible). At C > 1 greedy (Q8 MMA above 8 columns) and sampled at any C, drafter state changes the realization (draft lengths, verify width, sample path), not the distribution | `speculative_round.h:90-101`; greedy = plain at C = 1 (measured) |
| E6 | **Precision**: nothing is requantized or narrowed; CPU-served experts are bit-identical (§16.2) | E1 |

`--no-prefix-reuse` gives the deterministic cold decomposition for strict run-to-run reproducibility,
documented as a reproducibility trade-off, not a quality one.

#### Memory

- **VRAM: 0 frames, plus C spare pages** (0.34 frames each); cached blocks use idle pool pages. The
  whole-extent reservation stays (`E = ⌈min(max_context, n + effective_output)/64⌉` pages,
  `program.cpp:394-395`): at C = 1 a large `max_tokens` (default 8192) or 32-64K reserves the whole
  row and evicts other conversations' Host-backed Device blocks (~20 ms per 30K to restore).
- **Footprint lemma** (estimated): during admission a resume holds at most **E + 1** pages, namely
  `shared` path pages, `E − shared` private pages and at most one Device-resident COW source (the
  Device tail if F % 64 ≠ 0, the mismatched anchor if F % 64 == 0: exclusive cases; Host-only
  sources go straight into the private page; tap tails are Host-born). A whole-extent resume
  therefore fits at any C, all lanes at once; the source pin drops after activation, leaving E.
- **Device snapshot slots: 0.** `--device-snapshot-slots N` adds packed slots (115.7 MB = 41.8
  frames each; D2D capture under the existing slot policy); N ≥ 1 is required with
  `--host-context-mib 0`.
- **Host tier: one pinned slab pool, 4 GiB** (the §15.2 row; it replaced the earlier 2.0 GB
  checkpoint tier). Of 95.8 GiB the model pins 64.5 GiB, leaving ≈ 31 GiB; Vision adds 856 MiB of
  tower weights and ≥ 0.4 GB of pageable media buffers (§19.3.2). Resolved in the Program
  constructor after materialization:
  `host_bytes = explicit, or min(4 GiB, available − reserve − media reserve)` (the RAM ledger's
  reserve, §19.3.7), where the media
  reserve is the pageable live-media budget (`--media-live-mib`, ≥ 402 MB for the prompt cap) with
  Vision and 0 without; Vision pins only its 856 MiB of tower weights, which are already counted
  (§19.3.2). An explicit value breaking that guard fails at startup with the ledger's
  message; the default clamps to whole slabs and below 126 slabs
  (≈ 118 MB) disables the tier with a warning; with the default 0 Device snapshot slots the prefix
  cache is then off (no snapshot store), and the warning says so and names
  `--device-snapshot-slots` as the alternative. The startup ledger logs model and Vision pins, the
  reserve, the n-gram row cache and the tier.
- **4,096 MiB = 4,599 slabs:** ~5 working sets of a 32K conversation (500 blocks + 3 snapshots
  ≈ 875 slabs) or one 128K conversation plus ~16 snapshots; pinned in ≤ 4 GiB chunks no slab
  crosses (WDDM rule, Hybrid spec §5.4). Index: `max_nodes = kv_pages + slabs + 1`,
  `max_snapshots = D + slabs/124 + 1`, `block_bytes = 931,840`, `image_bytes = 115,673,088`,
  `image_slabs = 124`, `host_blocks = slabs != 0`.

#### Prefill integration

**Call planner** (`prefix/call_plan.{h,cpp}`, pure): `plan_calls(F, n, c, exact taps T ⊂ (F, n),
cost)` uses boundaries `{F} ∪ T ∪ {n}`, each segment running c-token calls plus one remainder call.
An exact tap is kept if Δ = Σ cost(with) − Σ cost(without) ≤ 0.1 s (`split_budget`; the opener at
n − 5 costs one CPU-served 5-token call), Explicit and Structural taps regardless (≈ +1.28 s
mid-chunk, once per distinct prefix); others become flexible, realized at the first call boundary
≥ p or the start of the final call. Plans depend neither on C nor on speculation (X9); for F = 0
without exact taps they are today's calls, so E3 holds and cold throughput is unchanged.

**Advance-prefill loop** (`program.cpp:495-536`): each Engine step takes the lane's next planned
call; consecutive ≤ 8-column calls share a step (bounded at ~50 ms). After each call: advance
`state_tokens` and `mtp_cells`, publish blocks, realize taps, and run `after_round` with 16
promotions per layer above 8 columns but `decode_budget` for small calls (no expert-cache churn).
**Small calls** (P0 step 3): `max_columns = max(8, lanes × max_width_)` CPU-serves every ≤ 8-column
call (opener tails, forced tokens, tiny suffixes) at any C and `--spec`, bit-identically (§16.2;
the service already takes 64 columns). **Uploads:** `run` uploads all of `io_layout_` today
(`program.cpp:979`, ~10.5 MB of n-gram rows at chunk 4096); prefill and forced calls upload
`io_prefix(width)` plus the MTP ids and cells (`8 × (width + 1)` B), with `upload_pinned` up to 64
columns so they never queue behind copy-engine traffic.

**Cost model** (`CacheCostModel` in `ProgramOptions`; Qwen4Exp gets `ContextMachineCostModel{}`
today, `engine.cpp:181`) ranks choices and values snapshots, never decides feasibility:

| Field | Value |
|---|---|
| `chunk_seconds` | 0.1645 (was 1.28) |
| `chunk_tokens` | `--prefill-chunk` |
| `token_seconds` | 2.23e-5 (was 1.17e-3) |
| `call_route_fraction` (ρ) | 10/512 = 0.0195 until M6 refits it |
| `attention_pair_seconds` | 1.06e-9 (was 0) |
| `h2d_bytes_per_second` | 26e9 |
| `transfer_batch_seconds` | 20e-6 |
| `prefix_span_seconds` (Qwen4Exp `CallCost::span_seconds`) | 1.67 |

Refitted 2026-10-05 on the Gold port (RTX 5090, INT8 KV, `infernix_bench` pp, `fn/rigs/port/gate3.bat`).
After the layer walk and expert streaming (§19.3.8 F2-F6), a prompt costs about 1.67 s per walk
span to stream the experts, 0.16 s per further call, 22 us per token and 1.06 ns per attention pair.
Measured against modelled: pp4096 in 4 and 8 calls 2.55 s and 3.03 s (model -4.8 % and +1.7 %),
pp16384 2.70 s (+5.0 %), pp32768 4.36 s (-1.5 %), pp131072 19.0 s (0.0 %). Single-call prompts of
512 to 4096 tokens took 2.1-2.5 s, which the model underestimates by 6-25 %. They pay the whole span
at a width the fit does not cover. Because the walk ends its span at every cut, a cut followed by a
GPU-staged call costs a span as well as a call. The planner therefore demotes automatic exact taps
that are not free, and coalescing charges a new tap the same amount. The old constants overpriced
tokens about 50-fold, so predicted coalescing waits exceeded the queue-timeout limit for shared
prefixes beyond about 4K tokens.

After P7, an EWMA of measured call seconds per width class feeds `index.set_cost` and the planner.

#### Prefill CPU assist (P7, gated by M6)

After a hit, TTFT is mostly staging of the suffix's distinct non-resident experts (100-3,000-token
tool results: ~1.2-4.8 s, estimated). A thin expert (n_e ≤ 8) costs a full 2.76 MB over the x8 link
(~27.6 GB/s) but only a DRAM-bound CPU GEMV, and DRAM (~70 GB/s practical, estimated) works in
parallel with the link. §13's "routed staging + CPU assist" is not built: prefill stays on the GPU
(`program.cpp:332`), `cpu_served` needs `columns ≤ max_columns` (`moe_layer.cu:741-743`),
`cpu_plan_kernel` copies all x columns (`:575-577`) into a `[H, max_columns]` buffer, and jobs are
≤ 8 of ≤ 8 columns today (concurrency S3 raises the cap before P7 lands). The architectds/Strata
fork claims 1.35-1.49× at 200-1,000 tokens and ≈ 1.0× at 4K, on PCIe 3 / DDR4 (§3.1).
**Op extension** (`src/ops/offloaded_sparse_moe/`): (1) `cpu_served` means "the CPU channel
exists"; above `max_columns` the plan kernel gathers only the selected jobs' columns (≤ J × 8),
remapping `request->column`, and y becomes `[H, J × 8]`; (2) a per-layer cap J for wide calls
(start 48), thinnest misses first, sized by `want = misses × r_cpu / (r_cpu + r_pcie)` from
per-layer EMAs of CPU and staging µs per expert uploaded per call, replacing `pcie_divisor` for
wide calls (decode keeps today's rule); (3) qualification by exact oracle (identical outputs with
assist on and off at 32, 256 and 2,048 columns, plus the W4A4 oracle) and M6. **Expected**
(estimated, uniform routing): 45 tokens ≈ 180 thin misses per layer; CPU ~20/ms + PCIe ~10/ms →
~6 ms per layer, ≈ 0.3 s vs ~0.8 s; 500 tokens ~1.35 vs 1.87 s; ≥ 3K tokens ≈ 1.0×; it also
speeds every cold prompt ≤ ~2K and the last chunk of long ones. **Risks:** DRAM contention with the
GPU's staging reads; worker scheduling on P- and E-cores.

*P7 as implemented* (branch `claude/fn-prefix-p7`, over concurrency S3, 2026-10-05):

- **Op.** `kMaxCpuJobs` 32 → 256 and `kMaxCpuCallColumns` 256. The plan publishes the chosen jobs'
  x columns compacted (the request names them by compact index, in column order), so the mapped x
  holds at most `kMaxCpuXColumns` = 256 columns whatever the call's width. `MoeCpuChannel` /
  `CpuMissService::Options` gain `wide_from` and `wide_jobs`: a call of at least `wide_from`
  columns takes `wide_jobs` as its cap instead of `max_jobs`. Deviation from the plan above: the
  split keeps the decode rule, want = min(cap, M − M / divisor) with divisor 3 (the CPU takes about
  two thirds of the thinnest misses); the per-layer EMA split is not built, since this rule already
  doubles short-prompt prefill (below). The CPU call bookkeeping grows to 1,280 bytes.
- **Program.** `ProgramOptions::cpu_assist_jobs` (256; 0 disables): the service's `max_columns`
  becomes 255 (`kAssistMaxColumns`) and `wide_from` the decode width + 1, so prefill calls of
  9-255 columns are assisted; wider chunks route nearly every expert and stream or stage on the GPU
  (F2).
- **Tests.** Layer test: an assist service (cap 256 from 9 columns) with the count oracle by width.
  `forward_real --cpu-columns` adds 24- and 100-column calls through a service configured as the
  Program's: 2,980 and 5,495 experts CPU-served, logits bitwise equal to the GPU route.
- **Measured** (2026-10-05, ABBA in one build with a temporary toggle, `infernix_bench`, chunk 1024,
  three repetitions): pp64 82.7 → 167.0 tok/s (+102 %), pp128 126.7 → 246.4 (+94 %), pp200 158.8 →
  287.9 (+81 %); the code prompt at `--prefill-chunk 128` prefills 62.9 → 88.5 tok/s plain and
  62.6 → 96.5 MTP with identical greedy ids; tg512 101.14 → 101.07 (noise). M6's suffix gate after
  a prefix hit is not measured yet (it needs the P5 planner on the same branch).

#### Flags, Engine wiring and Vision

| Flag | Qwen4Exp semantics |
|---|---|
| `--no-prefix-reuse` | Disables the cache: `allow_prefix_reuse = false` → `publish_continuation = false` (today hard-coded false, `program.cpp:389`) |
| `--host-context-mib N` | **Default 4096**, clamped as above; 0 = Device-only (requires `--device-snapshot-slots ≥ 1`) |
| `--device-snapshot-slots N` | **Default 0** (Qwen3.5: C + 1) |
| `--cache-taps-per-request`, `--cache-tap-ladder`, `--cache-tap-min-gap` | Unchanged: 8 / max(4096, 2·chunk) / max(1024, chunk) |
| `--prefix-cache-file PATH` | Unchanged; Qwen4Exp blocks persist `mtp_next`, snapshots their meta (after P6) |
| `--kv-capacity N\|auto` | `auto` = `max_context × C`; both get the C spare pages (documented) |
| `--use-original-prefix-caching` | Rejected for Qwen4Exp with a clear message (the flag and the original cache were removed on 2026-10-09) |

**Engine.** `normalize_engine_options` fills `device_snapshot_slots = C + 1` and the 8 GiB Host
default and rejects 0 (`model_instance.cpp:180, 188, 199`); these move to per-model resolution in
`construct_model` (Qwen3.5 unchanged) and `construct_qwen4_exp`, returned to `Engine::options()`.
`persists_prefix_cache` and `report_prefix_cache_save` handle the Qwen4Exp core
(`engine.cpp:228-266`); `shutdown_cleanup` saves before clearing; `infernix` (one request, Engine
default Legacy) keeps the cache off. **Vision.** Block extras compare exactly; no tap lands strictly
inside a span (`TapExclusion`) and resumes inside spans are rejected at quote; activation drops
payloads of items ending ≤ F (`hybrid_program.cpp:937-974`), so only the rest is encoded; suffix
RoPE positions come from the Vision plan's Program-owned prompt RoPE state (`rope_of(lane, index)`,
`rope_delta`), recomputed per request, never cached; Host RAM is budgeted jointly; P8 coalescing is
off for Vision prompts; the Vision work removes the media rejection (`program.cpp:375`).

#### Phases

Each phase builds and tests alone, in worktree `local\workdirs\NInfer-V3-fn-prefix`; GPU runs take
the GPU lock and run as hidden console jobs. Effort (estimated): P0-P6 ≈ 12 engineer-days, with P7
≈ 15; ≈ 3-4K new lines plus ~1.3K of tests.

| Phase | Work | Exit |
|---|---|---|
| **P0** prep (0.75 d) | (1) done on the base branch: `ProgramImpl` moved into `program/program_impl.h` (`54deaa2bc`, §19.3.0); (2) `mtp_cells` (MR1) with debug invariant checks at commit and finish, MR2 repair; (3) `max_columns = max(8, lanes × max_width_)`; (4) QSA tails fix (MR7) with an oracle case for blocks completed inside a call or commit | tg512 and code-prompt greedy ids unchanged at C = 1, plain and MTP; acceptance ≥ before; a ≤ 8-column forced call bit-identical with the CPU on and off; QSA op test passes |
| **P1** generic extensions (1.25 d) | `prefix_index.{h,cpp}`, `cost.h`, Qwen3.5's `insert_block` caller: extensions 1-4, reserved slabs in the invariants | U1, U4; `infernix_prefix_cache_index_test` and Qwen3.5 unit tests unchanged |
| **P2** state image, page records (1 d) | `prefix/state_image.{h,cpp}`; slots allocated before the frames: layout, per-part D2H/H2D with per-layer events and the MTP group, page records into any page by plane range, optional Device slots | X8 |
| **P3** binding and orchestration, unpipelined (3.5 d) | `prefix/prefix_cache.{h,cpp}`, `prefix_program.cpp`; `program.cpp` (`plan_request`, `quote`, `reserve`, `progress`, `advance_prefill`, `commit*`, `append_forced`, `finish`, `abort`, `release`, `fail_all_cleanup`, `usage`, `memory`); `ProgramOptions` (enable, host bytes, slots, taps, cost). Spare pages; chunked pinning and the Host guard; quote; reserve/activate (whole-batch wait); publication (MR4, MR5); Host-born flexible taps; finish/abort. Stats: `admissions`, `free_device_snapshot_slots`, `host_tail_restores`, `mtp_continuation_mismatches`, `mtp_branch_mismatches`, `endpoint_skipped_mtp`, `endpoint_mismatch_fallbacks`, `taps_skipped`. Test hooks: `force_call_boundaries`, `defer_mtp_cell_at`, `delay_streams(restore_ms, transfer_ms)` (a `cudaLaunchHostFunc` sleep heading each batch), `drafter_state(lane)` | X1, X2, X4, X10, X11, X13, X15-X18, D1 |
| **P4** pipelining, copy-engine discipline (2.5 d) | M0 first; `ForwardBatch::layer_waits`, ticketed events; restore after the first call's inputs; copy-outs in `layer_waits[1]`; io prefix uploads; `download_pinned` or write-through pacing; failure cleanup | X7, X9, X14; `compute-sanitizer memcheck` on X9 and X14; M3 shows per-layer overlap |
| **P5** call planner, exact taps (1.5 d) | `prefix/call_plan.{h,cpp}`, `advance_prefill`, `test_call_plan.cpp`: Δ admission, opener kept with `lineage_echo`, structural and explicit taps, Vision `TapExclusion`, quanta from the plan | U2, X3, X5, X6; cold-prompt calls unchanged for F = 0 |
| **P6** Engine, serve, docs (1.5 d) | `model_instance.cpp`, `qwen4_exp_instance.cpp`, `engine.cpp`, `serve_options.cpp`: per-model resolution, Host default and guard, `--help`, `MaterializationDiagnostics`. Docs: a "Qwen4Exp binding" section in the Hybrid spec (the authority), §19.2 status, user-guide trade-offs | Serve protocol usage check; `git diff --check` |
| Persistence (after P6, decided default; its effort is inside P6's 1.5 d and the P0-P6 ≈ 12 d total; merges after P6) | `prefix/persist.cpp` (`mtp_next`, snapshot meta), Engine save/report wiring | X12 |
| **P7** prefill CPU assist (2.5-3 d, gated; after concurrency S3-S5, §19.3.0) | `moe_layer.cu`, `miss_request.h`, `miss_service.{h,cpp}`, per-layer EMAs, Op tests; the wide-call gather and cap J build on S3's block-parallel `cpu_plan_kernel` and its CPU job capacity (8 → 32), not on today's plan | M6: TTFT ≥ 10 % better for 64-1,024-token suffixes, no regression ≥ 2,048, tg512 unchanged |
| **P8** C > 1 extras (1.5 d) | In-flight coalescing (`hybrid_await_sibling` mirror), blocked-head prefetch | Concurrent shared-prefix scenario at C = 2 |
| **P9** shared helper (1.5 d, later) | Slab pool, write queue, restore batch/events, persistence I/O into `runtime/prefix_cache/host_tier.{h,cpp}` for both bindings | Qwen3.5 real tests (27B) + Qwen4Exp tests |

**P0 as built** (branch `claude/fn-prefix-cache`; design decisions, results are recorded with the
measurements):

- **MR1.** `stage_mtp_chunk` prepends iff `mtp_cells < begin` and sets `mtp_cells = begin +
  columns` (`state_tokens` for non-last prompt calls and forced tokens, `state_tokens − 1` for the
  last prompt call); the drafter's pending-cell write and the catch-up set `state_tokens`. The
  settled invariant is checked after every commit (plain and verified) and at `finish`, at O(rows).
  A violation cannot change output (verification rejects bad drafts), so it does not fail the
  request: drafting is switched off for the rest of that request (`mtp_live = false`) and a Warning
  diagnostic reports the defect.
- **MR2** runs in `commit`'s cancelled branch for plain rounds only, on the compute stream: the
  round's `residual_out` column of that row (`mtp_residuals_`) becomes the saved column and
  `mtp_cells = state_tokens − 1`. The released lane does not use it today; it makes P3's abort
  endpoint consistent.
- **Small calls.** `max_columns = max(kMaxCpuColumns, lanes × max_width_)`: 8 for plain C = 1
  (was 1), 8 for MTP at C = 1 with 4 drafts (was 5); larger products are unchanged.
- **MR7 tails fix**: only the loop bounds of the existing `tail_kernel` and `commit_tail_kernel`
  change (`base = last − last % R`, up to `min(last, base + R − 2)`), and the `qsa_pool_keys`
  contract states the new tail. Main layers never read a tail from a call that starts at a 4-block
  start, so the text model's bits are unchanged; only the drafter's re-pools of a block completed
  inside one call or commit change, and pooled keys matter only once selection starts (more than
  2,051 visible tokens). Short contexts therefore keep their MTP drafts exactly; long ones may
  change acceptance.
- **Tests.** `infernix_qsa_test` (`tests/ops/test_qsa.cpp`) is the QSA FP64 oracle test of the
  shared-code rule (§19.3.0): index queries; pooled keys from the paged plane; the tails exactly
  after every call and commit; rewriting the last position of a block completed inside a call,
  across calls or in a commit (draft step 0 without tail update, prepend with) re-pools the same
  key; single-call invariance; attention per KV profile. Its selection fixture uses an exact value
  grid, so the selected set, including the lower-id tie rule, is decided exactly instead of being
  guarded against near ties. The CPU on/off exit check is Forward-level
  (`infernix_qwen4_exp_forward_real_test TOKENS --cpu-columns`: 1, 2, 5 and 8-column calls after a
  prefix, logits bitwise with the Program's CPU service and without), since the Program has no
  CPU on/off switch and a test-only one would sit in the shared `program_impl.h`.

**P1 as built** (`a519a4a67`): `reserve_host_image` / `release_host_image` /
`publish_host_snapshot` (a reservation's slabs are owned by nothing until published or released;
reservations evict like `begin_snapshot_host_fill` but never past their claim and are never
evicted while outstanding; `check_invariants` counts them), zero Device slots legal,
`insert_block(…, attach)`, and `CacheCostModel::call_route_fraction` / `call_seconds` with
`prefill_seconds` bit-identical to the per-chunk formula at ρ = 0. Tests U1 (Host-born publish with
and without a tail, duplicates, refused publications, claims, reservations under eviction, the
Device tail of a Host-born endpoint evicted and planned as a slab restore, `attach = false`, zero
slots) and U4 (closed form, saturation, subadditivity, ρ = 0 identity) in
`infernix_prefix_cache_index_test`.

**P2 as built**: `prefix/state_image.{h,cpp}`. The layout follows the forward pass: the header
(256 B: magic, version, a fingerprint of every part, frontier, `mtp_written`, `mtp_next`,
`lineage_echo`, `mtp_accept`), then per decoder layer its GDN recurrent and conv state or its QSA
tail, the PLE history in the group of the layer it precedes, then the MTP group (saved residual,
MTP tail); 115,673,088 bytes for the real geometry with MTP. `LaneStateImage` copies a lane to a
segmented Host image (by group or whole), to and from a packed Device buffer, and restores by
group. `kv_page_geometry` / `kv_layer_planes` are now the Program's single definition of the page
group's plane order. Test X8 (`infernix_qwen4_exp_prefix_state_test`, synthetic pools of the real
geometry, no model): Host, packed and write-through round trips byte-exact without touching other
lanes, header validation, and per-KV-layer page moves byte-exact (931,840-byte INT8 records).
Packed Device slots are not allocated by the Program (Device snapshot slots stay 0, below).

**P3 + P4 as built** (one step, pipelined from the start): `prefix/prefix_cache.{h,cpp}` (the
binding) and `prefix/prefix_program.cpp` (the Program side).

- **Snapshots are Host-born only.** A capture reserves Host slabs and copies the lane's image and
  its partial page from the lane, one group (decoder layer, then MTP) at a time on the transfer
  stream with an event after each; the next call of that lane waits per layer
  (`ForwardBatch::layer_waits[1]`). The tail page's planes of a layer are copied with that layer
  (`DeviceKVPagePool::copy_to_host_records` gained the plane-range form). An endpoint hands its
  partial page over as the snapshot's Device tail; a tap keeps it and the lane's later writes wait.
  The Host tier is therefore required: `--host-context-mib 0` or Device snapshot slots are refused
  (decided default 0 slots; packed slots are a later option).
- **Restores** are one batch per admission on the restore stream, in forward order: a decoder
  layer's group holds that layer's planes of every restored page (Host-only shared blocks into new
  cache pages, a Host copy-on-write source into the lane's first private page) and its image parts;
  the MTP group holds the MTP planes and image parts. One event per group; the first call waits
  per layer (`layer_waits[0]`), by ticket, so a landed batch's recycled events are never waited on.
- **Admission.** The quote chooses with the index (lane-resident candidates' restore bytes lowered
  by the image) and checks fit as plan §7.1 (copy-on-write source per MR6). Reserve re-selects,
  **pins the selection's path and snapshot before making room** (eviction could otherwise drop
  the blocks it is about to map), evicts, reserves `need`, and activates: the table maps shared
  pages, then private leases; a Device copy-on-write source is a `copy_page` on the compute stream;
  `mtp_cells` follows MR6; taps are planned beyond the frontier.
- **Waiting for an endpoint.** A prompt that continues an endpoint still being copied out (its
  anchor on the matched path and its tail continuing the prompt) waits for that copy (≤ one image's
  copy time) and is matched again; other prompts never wait.
- **Publication** at every prefill call, plain and verified commit and finish, at the plan §7.4
  frontier; MR5 decides adoption of a Host-only twin by `mtp_next`. Blocks past the prompt (the
  generated output) chain their lookup hash from the previous block's
  (`block_lookup_hash(previous, tokens, extra)`) and carry the prompt's trailing Vision key, as
  Qwen3.5's `hybrid_publish_blocks` does; the lane's hash list grows with them. (Found by the real-artifact
  test `infernix_qwen4_exp_prefix_cache_real_test`: the first version published only the prompt's blocks, so a finish never had the
  path an endpoint needs and turn 2 resumed from the last tap.) Taps are realized at prefill
  call boundaries only (flexible): exact taps, and the opener, wait for P5's call planner.
- **Finish / consistent abort.** The MR3 flush (one kv-only MTP cell from the saved residual), the
  last blocks, the endpoint when F ≥ the lineage's deepest snapshot + 64 and every full block is
  published, then write-through of the lane's Device-only path blocks after the endpoint (one
  transfer stream). The lane records the endpoint capture: while nothing else is admitted into the
  slot, a resume from that snapshot skips the image restore (lane-resident), only if the capture
  created the snapshot and it is at the lane's frontier (`deepest == F`). When the endpoint is not
  captured (within 64 tokens of the deepest snapshot, unfinished MTP blocks), the latest capture is
  an earlier tap whose state the slot no longer holds; the first version recorded it anyway, and a
  turn-2 resume from it skipped the restore and ran on the turn-1 end state (the real-artifact
  test: outputs differed from the second token).
- **Engine wiring (minimal; P6 completes it).** `ContextCacheMode::Hybrid` enables the cache with a
  4 GiB default Host tier and the tap and cost defaults of plan §11; there are no stats or
  persistence yet.

**P5 as built** (branch `claude/fn-prefix-p5`, on the Vision head; 2026-10-05):
`prefix/call_plan.{h,cpp}` (`plan_calls`, `CallCost`, `inside_exclusion`) and the Program wiring.

- **One plan per admission.** Every selection (snapshot or root) carries a `CallPlan` from its
  frontier: `plan_taps` with the prompt's hints, the matched frontiers and the Vision spans as
  exclusions, then `plan_calls`. A lane without reuse (or with the cache off) gets the plain grid.
  `advance_prefill` runs the lane's next planned call (`Lane::calls`) and throws if the plan does
  not continue the lane's state. Quanta are the plan's call count + output − 1 + Vision steps (the
  quote's resumed-request quanta previously omitted the Vision steps).
- **Δ admission** as designed: boundary taps (Explicit, Structural) always cut; the opener and
  Automatic taps, in position order, cut when they add ≤ `split_budget` = 0.1 s of fixed call
  cost, else become flexible. A call of ≤ 8 columns costs `served_call_seconds` = 0.035 s (the
  CPU-served estimate); wider calls `CacheCostModel::call_seconds`. A split that only shifts the
  grid (same call count, saturated calls) has Δ ≈ 0 and is kept.
- **Realization.** After a call ending at B: taps at or before B are due; a flexible tap is also
  due at the start of the final call. A boundary strictly inside a Vision item realizes nothing
  (its flexible taps wait for the next boundary). Calls of ≤ 8 columns promote at the decode rate
  (`decode_budget`), wider calls 16 per layer.
- **Opener kept** (no erasure after endpoint resumes, as decided). The image header gained bit 2
  `opener` (a Tap at a GenerationOpener hint); an activation from a snapshot with `opener` and
  `lineage_echo` counts `endpoint_mismatch_fallbacks`.
- **Vision.** The quote drops candidates whose frontier lies strictly inside an item, then re-plans
  the encode window from the frontier (`plan_vision(prompt, reused)`): items ending at or before F
  are not encoded again (previously every item was re-encoded on a resume). The lendable-frames
  check uses the re-planned window.
- **Not built:** consecutive ≤ 8-column calls sharing one Engine step. Taps keep ≥ 64 tokens apart
  (except Explicit), so two consecutive small calls do not arise in practice.
- **Tests:** U2 `infernix_qwen4_exp_call_plan_test` (grids, coverage, Δ admission and demotion against
  an independent enumeration over 4,000 random plans, refused inputs, span membership). The real
  test adds X5 (chat turn 2 without the reasoning block resumes at ≥ turn 1's opener), X6 (two
  sessions sharing only the system block resume at the same structural frontier) and X3 in its
  public-API form: a repeated chat prompt resumes at its own opener (an exact off-grid split of
  the capturing run) and must generate the same ids. The D1/X3 Program oracle with
  `force_call_boundaries` is not built.
- **Found by the real test:** with user turns shorter than 64 tokens the opener lies within
  `kMinimumTapSeparation` of the structural tap, and `plan_taps` keeps only the earlier tap of the
  cluster (as designed), so such a turn resumes at the end of the system block. The test's turns
  are longer than 64 tokens.

**P6 as built** (same branch, 2026-10-05). Per-model resolution lives in
`normalize_engine_options(options, Architecture)`; the Engine reads the artifact's architecture
first (an unreadable artifact falls back to the Qwen3.5 rules and fails in construction). Qwen4Exp:
Host tier default 4 GiB (`kDefaultQwen4ExpHybridHostCacheBytes`), 0 refused; no Device snapshot
slots (refused if given); Legacy enabled refused (`infernix` and `infernix_bench` already disable the
cache; the two real tests that used the default now disable it); `--prefix-cache-file` refused
until persistence. The RAM ledger gained a `prefix cache` term: the Host tier is planned with the
experts (with the default, ~73.3 GiB must be available for `infernix-serve`). `hybrid_stats` and the
materialization diagnostics (`cached_prefix_tokens`, `restored_host_bytes`) are filled. Docs: Hybrid
spec §17 (the binding's contract), `docs/qwen3_8-flash-next.md` "Prefix cache", `docs/serving.md`
flag rows, `infernix-serve --help`. Qwen4Exp-only counters (`endpoint_mismatch_fallbacks`, MTP
continuation and branch mismatches) are not in `RuntimeStats` yet.

**Persistence as built** (same branch, 2026-10-05): `prefix/persist.cpp` mirrors Qwen3.5's file
(tables first, then slabs, footer) with its own magic and version; blocks carry `mtp_next`, and a
snapshot's meta is read back from its image's header slab (a header that fails the layout
fingerprint or disagrees with the saved frontier fails the load, which then restores nothing). The
Program saves in `shutdown_cleanup` after releasing every lane (Device work synchronized, transfers
drained, the product's `PrefixCacheSaveControl` honoured); the instance attaches the file after the
Program starts, with `hybrid_cache_fingerprint(options, "qwen4_exp")`; the Engine reports the save
for either core. X12 in the real test: Engine 1 serves two chat turns and stops; Engine 2 restores
the file and resumes turn 2 at its opener with Engine 1's ids.

#### Tests

`infernix_qwen4_exp_prefix_cache_real_test` (`tests/models/qwen4_exp/test_prefix_cache_real.cpp`)
takes `INFERNIX_QWEN4_ARTIFACT` and a scenario selector, like the Qwen3.5 hybrid real test; Engines
are built sequentially (64.5 GiB pinned each); prompts are raw tokens unless the template is
needed. **D1, the drafter oracle** ("+D1"; E5 hides drafter errors at C = 1 greedy): after a
resume's first suffix call, compare bitwise with a cold Program run whose `force_call_boundaries`
repeat the capturing lineage's boundaries below F and F itself (`defer_mtp_cell_at = F` for
mismatched continuations): MTP-plane bytes of cells < `mtp_cells`, MTP tails, saved residual, the
first K drafts, and for X3 the whole main state (GDN, PLE, tails, main KV).

| ID | Scenario | Assertion |
|---|---|---|
| X1 restore-exact | A = 1,500 tokens (chunk 512); B = A[0:1400] + 300 new, greedy 64; Engine 1 resumes from Device / lane-resident, Engine 2 is forced to a Host restore by a pressure prompt | Equal `reused`; Engine 2 `host_image_restores`, `host_block_restores` > 0; identical ids; +D1; plain and MTP |
| X2 grid tap = cold | A leaves a flexible tap at 2·chunk; B = A[0:2·chunk] + 200; a fresh Engine runs B cold | Identical first-token logits and 256 ids (E3) |
| X3 split equivalence | Pure-prefill A with an exact off-grid tap F; resume B at F vs cold B with A's boundaries below F and F | Bitwise main + drafter state after the first suffix call; equal ids (E2) |
| X4 endpoint echo | Turn 2 = turn 1 + its 128 generated tokens + 64 new | Reuse = endpoint F; `mtp_continuation_mismatches == 0`; ids equal a Host-restored rerun (E1); +D1; agreement with cold reported, not asserted (E4) |
| X5 chat no-echo | Two template turns without `reasoning_content` | Reuse ≥ the turn-1 opener; the opener adds one small call and no other |
| X6 shared preamble | Three sessions sharing a ~6K system + tools block | Sessions 2-3 reuse ≥ the structural frontier; preamble pages counted once |
| X7 eviction | `--host-context-mib 512`, pool = max_context, three alternating conversations | Outputs equal cold runs with identical decomposition; `host_snapshot_evictions > 0`; fallbacks; debug `check_invariants` clean |
| X8 Host round trip | P2 GPU test on synthetic pools: lane state → slots → Host → another lane; page → slab → another page | Byte-exact |
| X9 concurrency | C = 2 plain, two conversations resumed concurrently from the Host; C = 2 MTP robustness | C = 2 ids == C = 1 ids; no crash or hang; no Filling block mapped |
| X10 cancel mid-prefill | Cancel after the first call; resend | Reuses the abort endpoint; ids equal an uncached run with the same boundary; +D1 |
| X11 Disabled | `allow_prefix_reuse=false`, serve warmup | Tree and snapshot counts unchanged |
| X12 persist | Save at `stop()`, restart, resume | Reuse after restart; ids equal the pre-restart resume |
| X13 whole-extent resume | C = 1, `n + max_tokens ≥ max_context`; F % 64 == 0 and ≠ 0; MTP and plain | Reuse > 0 in all four; no quote/reserve mismatch |
| X14 COW race | Host-only 64-aligned anchor, mismatching continuation, `delay_streams(restore 30 ms)` | +D1 bitwise; ids equal the undelayed run |
| X15 Host tail | Endpoint at F % 64 ≠ 0; a pressure admission evicts its Device tail; resume | `host_tail_restores > 0`; ids equal a Device-tail resume; +D1 |
| X16 immediate next | Next request with zero delay after finish, then with `delay_streams(transfer 30 ms)` | Reuse = endpoint F in both; the Host image equals a synchronous D2H of the slot at finish |
| X17 cancelled rounds | Abort after a cancelled plain and a cancelled verify round, at F % 64 ≠ 0 and == 0 | Consistent endpoint (+D1) at ≠ 0; `endpoint_skipped_mtp` at == 0 after a plain round |
| X18 short output | `max_tokens 1`, n % 64 == 0 | MR3 flush; the next echo request reuses F = n |
| U1-U4 (Host-only) | U1 index: Host-born publish with and without a tail, duplicates, claims, eviction with reservations outstanding, Device-tail eviction keeping the snapshot with a tail-slab restore plan, `attach = false`, zero slots. U2 planner: coverage of [F, n) without overlap, determinism, grid for F = 0, exact taps as boundaries, Δ accounting, demotion. U3 the P rule incl. P % 64 == 63, the MR2 repair state, the MR3 flush. U4 `call_seconds` saturation; ρ = 0 equals today's `prefill_seconds` | Unit assertions |

Fault paths: a failed Host pin at startup gives a clear error; a forced CUDA error in a restore
gives an Engine-wide failure with no leaked pins after recovery; `compute-sanitizer memcheck` on X9
and X14 checks event lifetimes.

#### Measurements

Runs take `E:\NInfer-V3\local\gpu.lock`, run as hidden console jobs, are dry-run validated before
arming, and A/B within one binary (control `--no-prefix-reuse`), except M2 (a): spare pages and P0
changes have no toggle, so it compares the pre-cache build, alternating the two binaries ABBA on
identical-id workloads. Base command:

```text
build-windows\bin\infernix-serve.exe E:\NInfer-V3\out\flash-next\qwen3_8_flash_next_nvfp4_dense8m.ninfer ^
  --ngram-volume G:\infernix\qwen3_8_flash_next_nvfp4.ninfer.ngram --kv-dtype int8 --max-context 65536 ^
  --prefill-chunk 4096 --max-concurrency 1 --spec mtp --draft-tokens 4 --lm-head-draft ^
  --host-context-mib 4096 --request-log-jsonl profiles\bench\fn_prefix\<arm>\request_log.jsonl
```

| ID | What | How | Report |
|---|---|---|---|
| M0 | Copy-engine behaviour (before P4) | ~100-line `local\workdirs\fn\prefix\m0_copy_engines.cu`: `asyncEngineCount`; 4-byte D2H latency on one stream during a 116 MB D2H on another; the same for H2D; `upload_pinned` under bulk H2D; D2H during H2D | Latencies, FIFO yes/no per direction → copy-engine rule 3 |
| M1 | Multi-turn TTFT, 8K and 32K | `local\workdirs\fn\prefix\multiturn_ttft.py` (stdlib HTTP + SSE): one conversation grown to ~8K then ~32K, frozen texts; 6 turns × 4 types: OpenAI without reasoning echo, Anthropic with thinking echoed, 50-token and 3K-token tool results; greedy, 300 output tokens; zero think time (quote step 2) | TTFT both arms and ratio, reused tokens, prefill s, restored bytes, `endpoint_mismatch_fallbacks`; worst cases |
| M2 | Decode impact | (a) `infernix_bench` tg512 (§19.2 command) and the cold code-prompt CLI vs the pre-cache build, then `--device-snapshot-slots 1` vs 0; (b) M1 logs, cached vs uncached turns; (c) restore speed (Strata's #528 check, with interference held fixed): two alternating 32K conversations with the same unrelated request between turns, lane-resident or Device resume against a forced Host restore (X1's two-Engine setup with long outputs), ≥ 256 decode tokens after each resume, ABBA | tok/s, hit rate, frames, compared only on identical-id workloads; per-turn decode tok/s and hit rate; (c) ids equal (E1), decode tok/s and hit rate per arm |
| M3 | Attribution | `nsys profile -t cuda,nvtx`: one Host-restore turn at 32K, one tap-heavy cold 32K prompt | Restore ms, layer waits, copy-out overlap, call times |
| M4 | Copy-outs vs decode, C = 2 | One lane decodes while the other finishes a 32K turn (write-through + image), before and after the rule-3 change | Round-time distribution during vs outside the copy |
| M5 | Tap overhead | Cold 16K with taps vs `--cache-taps-per-request 0` | Prefill s (target < 0.5 %) |
| M6 | Suffix cost curve, P7 gate | After 32K and 8K endpoint resumes: suffixes of 1-2,048 tokens (powers of 2) of tool-output text; P7 on/off (temporary env toggle, ABBA); with and without `--spec` | TTFT per length, per-call ms, CPU-served experts per layer, staged bytes; refit ρ |
| M7 | Agentic mix | `bench/agentic_ab` (`--scale 0.3`, `--max-context 65536`); needs per-arm extra args in `runner.py` | Cache tokens, TTFT p50/p90 continuing vs new, decode tok/s, `endpoint_mismatch_fallbacks` |
| M8 | Existing TTFT cases (secondary) | `py -3.13 tools\bench\run_serve_ttft.py`, cases `session-hot-continuation`, `shared-tools-sequential`, `resume-after-interference-state-host`, `cancel-after-first` | TTFT per role |
| M9 | Chunk trade-off with the cache | M1 + M7 at `--prefill-chunk` 1024 / 2048 / 4096 (§19.2: 8,791 / 8,614 / 8,300 frames, tg512 61.46 / 60.37 / 58.63; the 1024 row at max-ctx 4096, the others at 8192). Before a default changes, a long-prompt quality check: teacher-forced (`--dump-logits`) 8K and 32K prompts at each candidate chunk, against a rounding-order-only control (Strata `f23ea57`: a chunk-only change moved KL by 0.023-0.038, argmax agreement 94.8-97.9 %) and bitwise reruns | TTFT p50/p90 vs decode tok/s → serving chunk, fixed per Engine; pairwise KL and argmax agreement within the control's band |

| Prediction (estimated; chunk 4096, attention excluded) | Today | P0-P6 | + P7 |
|---|---:|---:|---:|
| 8K chat turn, no echo, ~800-token suffix | ~13-15 s | ~2.3 s | ~2.0-2.2 s |
| 32K chat turn, no echo, ~800-token suffix | 50-75 s | ~2.4-2.8 s | ~2.1-2.6 s |
| 32K echo turn, 50-token tool result | 50-75 s | ~0.85-0.95 s | ~0.3-0.45 s |
| 32K echo turn, 500-token tool result | 50-75 s | ~1.9 s | ~1.35 s |
| 3K-token tool result | 50-75 s | ~4.8 s | ~4.5 s |
| New subagent after a 20K shared preamble, 1.5K task | ~35 s | ~3.0 s | ~2.8 s |

`--spec none` is predicted the same (small calls CPU-served after P0), and tg512 unchanged.

**Acceptance (fixed before measuring, §19.3.0).** P0-P6 are kept when: M2 (a) tg512 and the cold
code-prompt run are unchanged within run-to-run noise on identical ids; M2 (c) restored and
resident arms decode the same ids at the same speed and hit rate within noise (a slower restored arm
is a defect: lost residency, a rebuilt proposer, n-gram re-hashing or MTP state); M5 tap overhead
is < 0.5 % of cold prefill; and every M1 turn type and M7 show lower TTFT than the
`--no-prefix-reuse` arm, reported against the predictions above with worst cases. P7 keeps its own
M6 gate.

#### Rejected alternatives

| Alternative | Why not |
|---|---|
| A new cache system | Nothing needs one: all sequence state is fixed-size per lane, KV is causal and page-local, and the QSA ratio (4) divides the page (64) |
| Legacy (`--use-original-prefix-caching`); moving `LogicalKVPageStore`/`KVAddressSpaceStore` to core | Legacy is driven by ~20 Qwen3.5-only hooks; the move is a large refactor of the production 27B path, while path pins already guarantee shared-page lifetime |
| Device snapshot slots by default (`C + 1`) | 84 frames at C = 1, ≈ 0.35 % decode (estimated from the §19.2 reserve experiment), ~30 ms per 1,000-token answer, to save ≤ 4 ms per non-resident resume; still available by flag |
| `--kv-capacity auto` = free VRAM | 1 GB = 362 frames (~1.5 % decode, estimated) to save ~20 ms per Host restore |
| Aligning resumed calls to the absolute chunk grid | No exactness gain (E2 is lineage-based); can add a wide call (~1.3 s) |
| Multi-call narrow suffix route (≤ 8-column calls) | Dominated: each call re-stages its distinct experts and the CPU takes ≤ 8 misses per layer per call; 50 prompt-hot tokens as 7 narrow calls ≈ 0.6-0.7 s vs ~0.3 s for one assisted call. A single small call stays CPU-served |
| MTP cell c at KV slot c + 1 | A QSA op change (pooled 4-blocks would straddle pages) plus requalification; the per-node next token solves it in the binding |
| "Every prefill call leaves its last MTP cell pending" (first revision) | Withdrawn: changed cold-run drafter order for nothing; MR4 makes boundary snapshots exact |
| Zero-split GDN state tap (Hybrid spec §7.2-7.3) | Not built: automatic exact taps are admitted only at Δ ≤ 0.1 s (the opener costs ~25-40 ms). The remaining cost is ~1.3 s once per new prefix, for an Explicit or Structural tap that splits a chunk. That does not justify a GDN kernel state tap, which is not implemented for any model |
| Review proposals | **Two spare pages per lane:** one suffices (lemma); the second served a removed early-tail D2D copy. **E1 for C = 1 greedy only:** MR5 removes the mechanism, so E1 holds at any C (E5 and the drafter risk stay C = 1 greedy). **Lane-resident pending capture as a candidate, or publishing with host = Filling:** needs a snapshot state `snapshot_valid` lacks; the bounded drain closes the gap. **Capping a resumed lane at pool − 1 pages:** silently cuts whole-extent output. **Pacing bulk D2H as primary:** D2H FIFO unverified, and `download_pinned` fixes it without delaying write-through; pacing is the fallback. **Single-kernel page copy:** `copy_page`'s ~0.1-0.3 ms per Device-COW resume is immaterial against ≥ 25 ms of suffix |

#### Risks

| Risk | Mitigation |
|---|---|
| Pinned Host tier on top of 64.5-65.3 GiB pushes Windows into the pagefile | 4 GiB default, joint guard with Vision, clamp and log; explicit values fail loudly |
| MTP edge cases (MR1-MR8) wrong, invisible at C = 1 greedy | Explicit `mtp_cells`; U3; D1 in X1, X3, X4, X10, X14, X15, X17; counters |
| Cross-stream races (COW source, copy-outs, restores) | Host-only COW sources straight into the private page per layer; ticketed events; `layer_waits[1]`; `delay_streams` (X14, X16); sanitizer |
| Copy-engine FIFO stalls (restores ahead of inputs, bulk D2H ahead of readbacks) | Submission order; transfer-stream order; M0 → `download_pinned` or pacing; M4 |
| Cached ≠ uncached at near-ties (E4) surprises users | NInfer's contract, documented; `--no-prefix-reuse` |
| Whole-extent reservations evict other conversations' Device blocks | Host-backed (~20 ms per 30K); lease windows deferred |
| P7 gains less than predicted (DRAM contention) | M6 gate; P7 is independent of P0-P6 |
| The opener after endpoint resumes costs ~30 ms per agentic turn for nothing if echoes are always exact | Counter; drop later if M1/M7 show ≈ 0 fallbacks |
| Duplication with Qwen3.5's binding; persistence invalidated by every rebuild | P9; expected (useful across restarts of one binary) |

#### Decisions and open questions

| Question | Decided (default, 2026-10-04) | Alternative |
|---|---|---|
| Host tier size | **4 GiB**, more by `--host-context-mib` | 8-12 GiB for ~100K-token agentic sessions if ≥ 16 GiB stays free after model and Vision |
| Device snapshot slots | **0** (frame-free; needs P1) | `C + 1` as Qwen3.5 (~84 frames ≈ 0.35 % decode) |
| Exactness contract | **As NInfer** (E1-E6): a resume equals the capturing request's own computation and may differ from an uncached run at near-ties; `--no-prefix-reuse` for strict reproducibility | Cold equality: resume only from grid-aligned flexible taps of cold pure-prefill lineages with no exact tap before F (E3) and refuse every other resume, endpoints and exact taps included (echo turns lose endpoint reuse) |
| P7 prefill CPU assist (shared offloaded-MoE Op change) | **Implemented, kept only if M6 passes** (≥ 10 % TTFT gain for 64-1,024-token suffixes, no regression ≥ 2,048, tg512 unchanged): it is the main post-hit lever and also speeds short cold prompts | No P7 (it changes shared offloaded-MoE Op code) |
| Exact Structural/Explicit taps that add a call (~1.3 s once per new preamble) | **Kept** | Demote when Δ > 0.1 s |
| Generation opener after endpoint resumes | **Kept**: ~25-40 ms + 116 MB Host per turn; it rescues echoes that diverge inside the generated turn, saving the tool result's re-prefill; break-even ~1-3 % mismatches for 1-3K-token results (estimated); each snapshot records `lineage_echo`, and an admission that resumes from an opener snapshot with `lineage_echo` set counts `endpoint_mismatch_fallbacks` (reported by M1 and M7) | Drop as Qwen3.5 does (`hybrid_program.cpp:986-996`, a ~15 ms split on 27B), or later for echo lineages if fallbacks stay ≈ 0 |
| Decode-time `</think>` tap | **Not for now** | Flexible tap at `</think>` (~4.5 ms stall + 116 MB Host per turn) for echo mismatches in content or tool calls |
| QSA tails fix | **In P0** (exact mismatched prepends; fixes today's step-0 pooled-key corruption; drafter-only) | Leave it, count stale prepends |
| `download_pinned` for per-round readbacks | **Only if M0 confirms D2H FIFO stalls** (decode path, output-neutral) | Write-through pacing |
| Code sharing | **Sibling binding now, shared Host-tier helper (P9) later** | Extract now (touches Qwen3.5, needs a 27B rerun) |
| Persistence (`--prefix-cache-file`) | **After P6** (any rebuild invalidates the file) | Inside P6 |
| Serving prefill chunk | **After M9.** 4096 costs ~491 frames (~4.6 % tg512) vs 1024 (§19.2; that 1024 row was at max-ctx 4096); with long cold prefills rare, 2048 may be better. The chosen chunk is fixed per Engine (never derived from free frames or loans, which would break E3) and passes M9's long-prompt quality check | Switch now |
| Idle-time re-render warming (product change) | **Not designed** | Background prefill of the predicted re-rendered history after a turn, so a no-echo suffix is only the new message (~1-2 s per chat turn); needs preemptible background lane work |
| C > 1 extras (P8) | **After the concurrency track** (§19.3.5) | Now |

### 19.3.2 Vision and vision offload

Design, reviewed twice (18 findings, all confirmed in code and folded in), nothing implemented
(worktree `NInfer-V3-flashnext`, HEAD `46a56fc8f`, 2026-10-04). The investigation was read-only: no
build, no GPU run, no weights loaded, one CPU-only query of the nsys trace
`local/workdirs/fn/prof/prefill1.sqlite`. Labels: **measured** (with its source), **code**
(`file:line`; paths beginning `qwen3_5/` are relative to `src/models/`, other model paths to
`src/models/qwen4_exp/`; bare Qwen3.5 file names in the baseline rows are under
`src/models/qwen3_5/frontend/`, `program/` or `execution/`), **estimated**. Local IDs: steps V0-V8,
tests VT1-VT9, measurements VM1-VM8 (not the §19 milestones). This delivers the Vision part of
§1.1 and refines the
`vision` frame role of §9.1-§9.2: the encode normally runs in the prefill workspace, and frames
are lent only for the output handoff and oversize windows.

In short: the tower is Qwen3.5's except for its output width, and the artifacts already hold it (no
reconversion). The work is tensor-core routes for the tower's projections (V0), M-RoPE through
every text call and the QSA indexer, an encode window that borrows device memory without VMM, and
two expert-cache correctness fixes. Offload is on by default.

#### Reused from Qwen3.5

Qwen4Exp already aliases the Qwen3.5 frontend (`using Frontend = qwen3_5::Frontend`,
`program/program.h:50-54`). After renaming, transformers' Qwen4Exp vision classes
(`modeling_qwen4_exp.py:1688-2015`) equal Qwen3.5's: rotary, MLP, patch embedding, merger,
attention (scale 72^-0.5, `cu_seqlens` segments), block, forward, `get_rope_index`, processors.
Only `out_hidden_size` differs, and Infernix reads it from `merger_fc2.weight.n`.

| Item | Flash-Next (checkpoint, code and artifact reports) |
|---|---|
| Vision config | depth 27, hidden 1152, intermediate 4304, 16 heads (D72), patch 16, temporal 2, merge 2, 2,304 position embeddings (48 × 48), out 2560, `gelu_pytorch_tanh`, no deepstack; the converter writes `model_type "qwen4_exp_vision"` |
| Token ids | image 248056, video 248057, vision_start 248053, vision_end 248054 |
| Text RoPE | `mrope_interleaved`, `mrope_section [11,11,10]`, rotary 64 of head 256, θ = 1e7; pair i uses axis i % 3 for all 32 pairs |
| Preprocessors | image `{shortest_edge 65536, longest_edge 16777216}`, video `{4096, 25165824}`, byte-identical to Qwen3.8-27B's; largest image 16,384 merged tokens |
| Weights | 333 BF16 tensors (excluded from quantization), **897,862,112 B**: prelude (patch [1152,1536] + bias, position table [2304,1152]) 8,849,664; per layer (qkv [3456,1152], proj [1152,1152], fc1 [4304,1152], fc2 [1152,4304], biases, 2 LN) 30,479,008; merger (LN, fc1 [4608,4608], fc2 [2560,4608], biases) 66,079,232 |
| Artifacts | Both deployed reports (`qwen3_8_flash_next_nvfp4{,_dense8m}.ninfer.conversion.json`) contain `vision`: 333 objects `('bf16','cast_direct','contiguous_le_v1')` and both preprocessor resources, built by `tools/convert/qwen4_exp.py` through the Qwen3.5 builder. The Flash-Next recipes assign no vision format (§6.1; never below upstream precision). Qwen3.5's official recipes quantize vision through `_optional` (patch Q6, merger Q8, QKV/fc1 Q4, proj/fc2 Q5), which is why only Q-formats have specialised vision shapes. |

| Infernix Qwen3.5 piece | As built (the baseline) |
|---|---|
| Frontend | Media acquisition and `MediaPreprocessCache` (`--media-cache-mib`, `--media-live-mib`). The processor smart-resizes and packs BF16 `[raw_patches, 1536]` as **pageable** payloads (`prepared_prompt.h:35-47`). `--vision-max-merged N` caps an item at N × 1024 px; hard caps are 32,768 merged tokens per prompt and 16,384 per item. `assign_positions` (`processor.cpp:634-708`) stores M-RoPE positions axis-major `[3,n]` with `rope_delta`: text has equal axes, an image run `(cur, cur+y, cur+x)` then `cur += max(gh,gw)`, a text-only prompt the index on every axis. `block_hashes` / `block_extras` hold a cumulative vision key per 64-token block. |
| Vision control (`vision_control.cpp:37-211`) | 2-D RoPE ids; 4-tap bilinear indices and weights into the 48 × 48 table; scatter indices; segments (`segment_length = h·w`, `segment_count = t`) |
| Tower (`VisionContext::encode`, `qwen3_5/execution/vision.cpp:322-464`) | One item per call: pageable H2D → patch projection + bias → bilinear position embedding → 27 × [LN → fused QKV + bias → 2-D RoPE (D72, θ 10,000) → non-causal attention per segment (`packed_softmax_attention`, equal-length form) → proj + bias + residual → LN → fc1 + bias → tanh-GELU → fc2 + bias + residual] → merger LN → view [4608, V] → fc1 → exact GELU → fc2 → BF16 [out, V]. Residual `x` 2,304 B/patch across all layers; the MLP scope (MLP-up 8,608 + MLP-norm 2,304 B/patch) peaks on top of `x`, so 13,216 B/patch plus a 1,280 B/patch handoff, which `encode()` requires as its output (`vision.cpp:333-341`). |
| Injection | `ops::embedding` → `ops::scatter`; `[T,3]` RoPE for multimodal chunks, else `positions + rope_delta`; the MTP takes visual embeddings at image cells; at most one item per chunk (one-item handoff). |
| Offload (commit `5f7350f`) | HostPinned weights. A VMM `EvictableWeightPool` lends the weight arena's tail and re-uploads it from a mirror afterwards. `VisionWeightStream` streams prelude, merger and two layer slots on `transfer_stream`; it uploads prelude and merger before any compute fence exists (safe only because the overlay VA is fresh) and re-streams the tower per item. The window ends with a host sync, a D2H of each item to pinned memory and a re-upload during prefill. |

**How features enter Qwen4Exp** (transformers reference):

- **Scatter before the stream expand** (`masked_scatter` precedes `repeat(1,1,hc)`): in Infernix
  between `ops::embedding` and `ops::hyper_connection_expand` (`execution/forward.cpp:108-111`).
- **PLE** hashes `input_ids`, so placeholders are hashed; `stage_ngram` hashes `lane.history`,
  which holds them. No change.
- **Positions.** Prompt: 3-D `get_rope_index`; decode: `index + rope_deltas`. The indexer query
  rotates by the current token's M-RoPE, a pooled block key by its **first token's**. Visibility,
  blocks and selection stay in KV-index order.
- **MTP** is not defined by transformers. Following vLLM and Infernix's Qwen3.5 convention, cell c
  uses the target's RoPE at c and the input embedding of token c+1, visual where c+1 is an image
  token. Only acceptance depends on this.

#### Blockers in Qwen4Exp today (code)

| Blocker | Where |
|---|---|
| Engine refuses `--vision` (frontend `vision_enabled = false`); config refuses vision; Program refuses media | `runtime/engine/qwen4_exp_instance.cpp:22, 46-55`; `config.cpp:234`; `program.cpp:375` |
| No vision resources, weights or parameters bound | `load.cpp:209-224`, `weights.h`, `config.h`, `execution/parameters.h` |
| One `positions` array is KV index and RoPE position in io, sequence, decode, verify, MTP and catch-up staging; main RoPE is 1-D, though `ops::rope` already has Text M-RoPE `[T,3]` (D256/R64; 24/2 heads take the generic route) | `program.cpp:243-251, 940-972, 1112-1146, 1337-1505`; `forward.cpp:374` |
| The QSA indexer rotates by KV index; QSA equals causal attention below 2,051 visible tokens (B 2048, R 4), so wiring errors are invisible there | `src/ops/qsa/qsa.cu:62-67, 87, 112-133`; `include/infernix/ops/qsa.h:20, 29-30` |
| MTP input is a token gather; `reserve()` keeps only token ids and the `PreparedPrompt` dies at return | `forward.cpp:510-511`; `program.cpp:423-476` |
| Decode and verify upload only the io prefix before `ngram`, MTP drafts only the words before `drafts`; MTP sub-chunks slice `positions` along dim 0 (`kMtpChunkColumns = 512`) and check `numel`; `ops::rope` needs contiguous positions | `program.cpp:863-865, 991, 1146, 1417`; `forward.cpp:34, 176-178, 195`; `src/ops/wrapper/rope.cpp:75-78` |
| Every vision projection runs `bf16_general_gemm`, a scalar 32×32-tile `fmaf` kernel without MMA; `include/infernix/ops/linear.h:79` lists only 3 BF16 problems | `src/ops/linear/bf16/bf16_dispatch.cpp:13-28`, `bf16_general.cuh:15-75` |
| Load publication ignores the generation (ABA); `FramePool::acquire` is LIFO; entries go `Ready` at issue | `expert_residency.cpp:84-89`; `expert_cache.cpp:144-149, 193-199` |
| Only compute is synchronized at round ends: up to 16 promotions per layer × 48 = 768 × 2,764,800 B ≈ 2.1 GB ≈ 77 ms of DMA stay in flight after a prefill chunk | `device.cu:175`; `program.cpp:509-510, 523, 822` |
| PLE row reads have no in-call de-duplication | `program/ngram_volume.cpp:47-95` |

**The ABA hole exists without vision.** Batch B1 loads K into frame f; K is evicted, f goes to K2,
K2 is evicted and K re-enters f, all while B1 is in flight; B1's completion then publishes K@f
while a later copy still writes f. Today that needs B1 in flight over ~3 rounds and a fresh key
evicted, so it is very unlikely (estimated); lending makes it likely (frames return LIFO, hot keys
come straight back). The load serial below closes both. §9.4's device entry has a generation; the
host's publication test does not use it.

**VRAM order:** state, PLE history and tails, KV, io, `work_` (`Forward::workspace_bytes(columns_)`
+ sampling + verify + MTP workspace), MTP buffers, 64 staging slots, then frames =
(free − 384 MiB) / 2,764,800. From §19.2's chunk table (8,791 → 8,300 frames for chunk
1024 → 4096) `work_` grows ~442 KB per column: **~0.5 GB at chunk 1024, ~1.9 GB at 4096**
(estimated; the implementation logs `work_capacity_`).

#### Ownership and the shared tower

| Piece | Design |
|---|---|
| Shared, Program-free tower `src/models/qwen3_5/execution/vision_tower.{h,cpp}` (new) | `VisionWorkspacePlan` moves into it from `qwen3_5/program/planning/startup.h:108-116`, so Qwen4Exp does not include `qwen3_5/program/internal.h`. `VisionTowerItem{patches BF16 [1536, P]; const VisionItemControl* (grid, position ids, interpolation, segments); Tensor output BF16 [out, V], caller-owned}`. `plan_vision_pass(config, params, item_patches) → VisionPassLayout{bytes}`. `VisionTowerPass(device, config, weights, items, DeviceSpan arena, layout, VisionWeightStream* /* null when resident */)` is resumable and layer-major: `advance(end_stage)` over `stages(depth) = depth + 2` (0: patch and position embedding of every item; 1-27: that layer for every item; 28: the mergers into their outputs), `next_stage()`. No internal handoff. |
| Batch invariance by construction | Every GEMM, norm, RoPE and attention call runs per item with its own T = P_i. Attention uses the equal-length form, one call per item, whose tile depends only on the segment length (`src/ops/softmax_attention/dense/packed/launch.cu:54-75`); the `cu_seqlens` form always uses 64 × 64 tiles (`kernel.cuh:16-17`) and is not used. An item's embedding is bitwise the same alone or with other items, whatever Linear routes do with T, and each layer's weights still stream once for all items. |
| Pass layout | `x` for every item plus per-layer scope tensors shared and sized for the largest item: the MLP scope (10,912·P_max) dominates attention (9,216) and patch (~3,112); the merger's normalized buffer is P_max-sized, its hidden aliases `x`. `pass_bytes ≈ 2,304·ΣP + 10,912·P_max` (one item: 13,216·P). |
| Qwen3.5 contract | `VisionContext::encode` keeps its plan, `bind_output` and handoff check and becomes a one-item pass over its layout: same op sequence, bit-identical output. Using the pass in Qwen3.5's offload window (no per-item re-stream) is optional and out of scope. |
| `VisionWeightStream` | The constructor takes `std::span<const cudaEvent_t> wait_before_first_upload`, records a compute fence and waits on it and those events **before the first upload**. Order: prelude → L0 … L26 → merger, so the merger no longer delays layer 0. New `finish(cudaStream_t compute)` records the final event and makes compute wait; the destructor syncs the transfer stream if `finish` was not called (failure paths). Qwen3.5: same bytes, fresh VA, unchanged behavior. |
| Also exposed | `make_vision_overlay_layout(plan, prelude, per_layer, merger objects)` (`qwen3_5/load/vision_overlay.cpp:18-69`); `qwen3_5::parse_vision_config(json, model_type)` (`qwen3_5/config.cpp:183-213`) |
| Qwen4Exp owns | Config and binding; the window (`program/vision_window.{h,cpp}`, new); frame lending (`program/expert_cache/*`, `program/expert_residency.*`); RoPE staging (`program/rope_positions.{h,cpp}`, new); the scatter and MTP visual embeddings (`execution/forward.*`) |

#### V0: tensor-core BF16 routes for the tower

- **Why.** The fallback runs Flash-Next's BF16 QSA projection [13952,2560] in prefill at **13.2
  TFLOP/s** (22.2 ms per call at T = 4096, measured in `prefill1.sqlite`). At that rate a 1K-token
  image (P = 4,096) spends ~260 ms in GEMMs; tensor cores would take ~25-35 ms (estimated).
- **Problems [N,K]:** patch [1152,1536], QKV [3456,1152], proj [1152,1152], fc1 [4304,1152], fc2
  [1152,4304], merger fc1 [4608,4608], merger fc2 [2560,4608], and [5120,4608], the merger fc2 of
  Qwen3.8-27B-Quasar (the published Qwen3.5-architecture artifact whose BF16 vision also runs the
  fallback, `tests/ops/linear/test_bf16_a16.cpp:278-297`; its other six shapes change route anyway,
  so registering the seventh moves its whole tower off the fallback). Registered for every positive
  T as the BF16 contract requires (`include/infernix/ops/linear.h:79` updated):
  `src/ops/linear/bf16/shapes/n1152_k1536.cu` … `n5120_k4608.cu`, `bf16_shapes.h`,
  `bf16_dispatch.cpp` (`kShapes`), `sources.cmake`.
- **Kernels.** All N and K except 4304 are multiples of 64, so the existing `Bf16A16MmaSchedule` /
  `Bf16A16TmaMmaSchedule` instances apply (K static per instance). The MMA launchers need complete
  row and K tiles (`bf16_template_launch.cuh:71-72`, `bf16_a16_tma_mma.cuh:56`), so 4304 gets a
  tail-capable TMA variant: out-of-bounds boxes zero-fill (fc1's N tail, fc2's K tail; the 8,608 B
  row stride is 16-byte aligned), the epilogue guards rows ≥ N, the K loop runs ⌈K/BK⌉ tiles.
  T ≤ 8 reuses the skinny GEMV (every K is a multiple of 8). Per-T selection comes from complete-op
  cold CUDA-graph measurements, as for the Q-format vision shapes.
- **Qualification.** `run_bf16_linear_case` against the FP64 Linear oracle at T ∈ {1, 2, 8, 9, 63,
  64, 65, 255, 256, 1024, 1025, 4096, 4100, 16384}, plus 65,536 sampled; the vision shapes leave
  `run_general_bf16_linear` for registered cases. Bench `infernix_linear_bench --qtype bf16 --policy
  a16`; curves in `docs/maintainer/linear-tuning.md`. Expected 100-150 TFLOP/s (estimated).
  Flash-Next's text BF16 shapes are untouched, so text stays bit-identical.
- **Implementation decisions (V0, 2026-10-04).** No new kernel: every route is an instance of an
  existing BF16 template.
  - The 4304 tail lives inside the existing TMA MMA template, selected by a schedule trait
    (`Bf16A16TmaTailMmaSchedule`, `kTail`): 2-D tensor maps read 64-element K boxes with hardware
    zero fill, the grid and K loop round up, and the store predicates rows ≥ N
    (`bf16_finish_fragment<FullTokens, FullRows>`). Non-tail instantiations compile to the
    previous code. Collective and fragment epilogues reject the tail form at compile time, so the
    fused BF16 Ops that share the template cannot select it by accident.
  - T ≤ 8 returns the fallback's skinny GEMV (`select_bf16_general_launch`), so those outputs stay
    bit-identical; only the merger of tiny inputs reaches it, too rarely to tune. (The registered
    GEMV/SIMT schedules need K to be a multiple of 128 or more, which fc2's 4304 is not.)
  - Large-T routes rasterize row-fast (`Bf16A16TmaR64T128K64S2Rows` and its tail form): the
    activation (up to 564 MB, fc2's input at P = 65,536) is read about once from DRAM while the ≤ 47 MB weight
    stays in L2; the token-fast order would re-read it once per row tile (18-80 times).
  - Per-T selection from a temporary private-launcher sweep (cold-L2 CUDA graphs, existing TMA,
    cp.async MMA and sliced-K schedules, row-fast and grouped rasters; deleted after use): each
    interval takes the fastest candidate within 2 % at each measured T, seams at the last measured
    T. The sweep's winners were up to 1.6× faster than the first default (the largest 64 × {32, 64,
    128} tile that still launched about two waves).
  - Measured (VM1, 2026-10-05, RTX 5090, graph execution): registered against the fallback
    9.6-22.6× over all 8 shapes and T ∈ {64 … 65,536}, 12.4-18.3× at T ≥ 1,024 (acceptance ≥ 5×);
    e.g. fc1 [4304, 1152] 15.5× at T = 4,096, the merger [5120, 4608] 247 TFLOP/s at T = 65,536.
    The BF16 Linear, LinearAdd and attention-input op tests pass at the final selectors; the
    Quasar real-model regression (identical text) ran before the sweep's selectors were pasted.
  - On Windows the eager TMA routes share one descriptor staging buffer, which requires them to run
    on the compute stream; V6's window runs the tower GEMMs there.

#### M-RoPE through the text path

Every Forward and MTP call carries `rope_positions` I32 `[T,3]` (axis-major, contiguous) beside the
unchanged `positions` I32 `[T]`, the KV index for append, visibility, selection and tails. Text
lanes stage three equal axes, which give bit-identical output in the fixed route
(`rope.cuh:64-71`), the generic route (`:120-127`) and QSA. Because every call has the same form, no
graph's topology depends on whether a lane has media. Cost: 12 B of io per column (96 B per decode
round at C = 8, 12 KB per 1,024-column chunk). The io layout puts the arrays where decode, verify
and draft uploads reach:

```text
IoLayout: ids | positions | rope [3·columns_] | block_rope [3·lanes] | slots | rows | columns | ngram |
          vision_columns [columns_] | mtp_ids | mtp_cells | mtp_rope [3·(columns_+1)] |
          mtp_block_rope [3·⌈(columns_+1)/512⌉] | mtp_vision_columns [columns_+1]
```

| Piece | Design |
|---|---|
| `qsa_index_query(q, norm_weight, rope_positions, geometry, stream)` | I32 `[T,3]` axis-major, pair i on axis i % 3 (`ops::rope` TextMrope convention, `src/ops/kernel/rope.cuh:52-71`). A contract change with no 1-D mode and no compatibility path: Qwen4Exp is the only caller. |
| `QsaBatch`, `pool_kernel` | New `rope_positions` `[T,3]` and `block_start_rope` I32 `[3, batch]`: the RoPE position of token `R·⌊start/R⌋` of each sequence (start = first column), read only when that token precedes the call. Block b rotates by token `first = R·b`: `rope_positions[a·T + t − (p − first)]` if `first ≥ start`, else `block_start_rope[a·batch + sequence]`; only one block per sequence can begin before the call. `qsa_commit_tails` and `qsa_attention` are unchanged. Contract: "pooled key b = RoPE(norm(…), RoPE position of token R·b)"; "QSA equals causal attention while p+1 ≤ B+R−1" stays. `rope_angle` computes `float(position) * inv(i)`, so equal axes give identical angles. |
| Prompt RoPE (`program/rope_positions.{h,cpp}`) | `reserve()` consumes the `PreparedPrompt`, so `Lane` keeps `LaneRope{prompt: I32 [3, prompt_tokens] axis-major, empty for text-only; delta: rope_delta, 0 for text-only}`, moved in only for prompts with media (12 B/token). One pure, unit-tested helper `rope_of(lane_rope, prompt_tokens, index)` returns `prompt[a·n + index]` for a prompt token of a media prompt, else `index + delta` on all three axes. It serves prefill chunks, forced tokens (`append_forced` begins ≥ `prompt_tokens`), decode, verify, MTP prompt cells and drafts, catch-up and every block start. |
| io placement | `rope` and `block_rope` precede `ngram`, so `io_prefix(columns)` uploads them in decode and verify; the rest is prefill-only (`run()` uploads everything). Axis a of a `cols`-column call sits at word offset `a·cols`, so `[cols,3]` is contiguous. A constructor check asserts that `block_rope` ends at or before `ngram`. `mtp_io_` gains `rope_cells` and `block_rope_cells` [K][3][lanes] before `drafts`, inside the existing `4·mtp_io_.drafts` upload; draft step j of lane b uses `rope_of(cell + j)`, block start `rope_of(R·⌊(cell+j)/R⌋)`. |
| Block starts | `block_rope[a·batch + b] = rope_of(lane_b, R·⌊start_b/R⌋)` for every call: chunks need not start at multiples of 4 (`--prefill-chunk` is any value ≥ 1; forced tokens and verify windows start anywhere). |
| MTP sub-chunks | The 512-column loop slices along dim 0, which `[cells,3]` cannot do contiguously. `kMtpChunkColumns` moves to `forward.h`; `stage_mtp_chunk` writes sub-chunk k as its own `[n_k,3]` block at word offset `3·512·k`, with block start `rope_of(R·⌊(first+512k)/R⌋)` (with a prepended cell, `first+512k` is not a multiple of R); `MtpChunk` carries the base and `block_start_rope [3, subchunks]`; the geometry check compares `ne[0]`, not `numel`. |
| Forward (`execution/forward.{h,cpp}`) | `ForwardBatch` gains `rope_positions`, `block_start_rope`, `const VisionInput* vision`; `MtpCall` gains `rope_positions`, `block_start_rope`, `input_embeddings`. `AttentionCall` passes them to `ops::rope` (`forward.cpp:374`), `QsaBatch` and `qsa_index_query`; KV append keeps `call.positions`. `ForwardTap` gains `qsa_index_queries` (rotated `[Di, heads, T]` per attention layer) for VT6. |

#### Embedding injection and the MTP mapping

- **Token-order handoff.** Encoded items go back to back in prompt order into one BF16
  `[2560, V_encoded]` handoff. With `rank(j)` = visual tokens in `[first_encoded_token, j)`, the
  visual tokens of any range `[a, b)` are the contiguous slice `[rank(a), rank(b))`; chunk-local
  columns are a host list. The handoff lives until the lane's prefill ends, so unlike Qwen3.5 a
  chunk need not stop at an item boundary.
- **Main chunk.** `ForwardBatch::vision = {handoff slice view, columns I32 [m] in
  io.vision_columns}`: `ops::embedding` → `ops::scatter` → `hyper_connection_expand`.
- **MTP cells.** Cell i sits at `first + i` and takes token `first + i + 1`
  (`program.cpp:1341-1350`). In a non-last chunk the next token is known, so the last cell takes
  **the next chunk's first token**. The visual range is token range `[first+1, first+cells+1)`:
  `[begin+1, begin+width]` without a prepended cell, `[begin, begin+width]` with one, read from the
  global handoff by `rank`. `stage_mtp_chunk` writes per-sub-chunk local columns into
  `mtp_vision_columns`; `input_embeddings` is `ops::embedding(ids)` → `ops::scatter` into `x0`
  before `embedding_norm`. Decode, verify and draft cells take generated text: no visual input.
- **PLE cost.** A 1,024-token image chunk issues ~16 rows × 1,024 lookups over ~48 distinct rows:
  ~16,000 4 KiB reads at ~4.3 µs (≈ 70 ms) today, ~48 with exact in-call de-duplication
  (estimated). The n-gram plan's S1 (§19.3.4) de-duplicates per 4 KiB block in
  `NgramVolume::read_rows`, which covers this; Vision depends on it and adds no second version.

#### Loading

| Item | Change |
|---|---|
| Config (`config.cpp:234`) | With `options.vision`, require the `vision` component, parse with `parse_vision_config(…, "qwen4_exp_vision")`, check out = `text.hidden_size` (on the bound `merger_fc2`), D72, θ_vision 10,000, `rope.pair_axes[i] == i % 3` |
| Binding (`load.cpp`) | `bind_vision`: a ~45-line copy of `qwen3_5/load/vision.cpp` with the same parameter names; residency `HostPinned` when `overlay_vision()`, else `Device`. `bind_resources` adds both preprocessor resources (`qwen3_5::parse_resources` validates them). Pinned vision objects join the existing pinned materialization; `LoadPlan` records `make_vision_overlay_layout(...)`; `Model` keeps `VisionOverlayAssets{pool = nullptr, pinned_block, layout}`. **No `EvictableWeightPool`, no VMM.** |
| Parameters (`execution/parameters.cpp`) | `qwen3_5::execution::VisionParameters`: fused QKV through `ops::prepare_linear_weight({q,k,v})`, joined bias |
| Engine (`qwen4_exp_instance.cpp`) | Drop the rejection; `vision_enabled = options.enable_vision`; pass `vision_max_merged_tokens` to `make_frontend`; report `vision` in `weight_formats` |
| Frames | The handoff needs frames. The cache is always on (`ProgramOptions::expert_cache = true`, no setter), so frames are 0 only below the 384 MiB reserve; startup then fails with "vision needs at least N expert frames". |
| Host RAM | Offload pins 856 MiB more, 64.47 → ~65.3 GiB, inside the loader's 8 GiB-free rule on this 95.8 GiB machine (estimated) |

#### The encode window and its memory

Items to encode are those with `token_end > reused`, and a window is always **one** pass. Window
bytes `W_ws = (offload ? staging 135,886,912 : 0) + pass_bytes`, handoff `5,120·V_encoded` B,
estimated GPU time `t = Σ_i [842 MFLOP·P_i / R_gemm + 27·4·P_i²·1152 / R_attn]`, at least 32.5 ms
with offload (`R_gemm`, `R_attn` measured in VM2, documented in code). Two placements:

- **W (workspace).** If `W_ws ≤ work_capacity_` and `t ≤ kVisionStepSeconds`, the whole window is
  enqueued in one `progress()` call from a `work_` scope. `work_` is idle between units
  (`Forward::run` resets it; every user is stream-ordered on compute). Only the handoff is lent; no
  other expert is evicted.
- **L (lease).** Otherwise one run of `⌈(handoff + W_ws) / 2,764,800⌉` frames is lent, handoff
  first, and the window runs in steps of `⌈27·kVisionStepSeconds / t⌉` layers, one per `progress()`
  call returning `InProgress` (supported: `hybrid_resource_manager.h:200-203`,
  `engine_core.h:1646-1647`). The engine runs other lanes' units between steps; lease and stream
  state persist on the Program; lent frames are disjoint from decode memory; the non-handoff part
  goes back at completion.

| Item | Design |
|---|---|
| `LaneVision` (`std::unique_ptr` in `Lane`, filled at `reserve()` from the moved `PreparedPrompt`) | The `VisionControlPlan` and media payloads (freed when the encode ends); the encoded item range and the handoff `FrameLease` (held until the last prefill chunk is enqueued); per-item token spans for `rank`; timings. Released by `release(index)` (finish, abort, discard, cancelled commits), `release_all()` (compute first waits on any live weight stream's final event, then syncs) and the failure path of `progress()`. |
| `kVisionStepSeconds` | 0.25 s initially, tuned by VM7. A 1,024-token prefill chunk already blocks other lanes for ~1.5-2.5 s (§19.2 rates). |
| `progress(cancellation)` | (1) Nothing to encode: `Published`. (2) Cancellation: abort path, `release(lane)`, `Aborted`. (3) First call: lend the handoff (W) or the run (L) through `ExpertResidency::lend`, whose waits apply to compute and, under L, to `transfer_stream`; construct `VisionWeightStream` (offload) with the lease's wait events; take `stream.window_weights(params)`; start a CUDA event timer. (4) `pass.advance(next end stage)`. (5) When done: `stream.finish(compute)`, return the non-handoff lease (L) stream-ordered, release payloads, record timings, `Published`; else `InProgress`. (6) Abort and failure (RAII `WindowGuard`): compute waits on the stream's final event, every lease goes back stream-ordered, the vision state is cleared. |
| Stream ordering | The first upload waits for a compute fence recorded at window start, which also covers MTP catch-up kernels still queued from `commit()` that use `work_`; under L it also waits for the lease's in-flight promotion events. Compute waits on every prelude, layer and merger upload and on the final event in `finish`. Later promotions into returned frames wait on `table_ready_`, recorded on compute after `give_back`. No host sync is needed. |
| Patch upload | Stays pageable (`copy_host`, `cudaMemcpyAsync`), so it can queue behind the shared H2D FIFO when another lane has just issued promotions (VM4). Pinned payloads would change Qwen3.5's frontend and the media-cache budget: not taken. |
| Telemetry (`GenerationTimings`, `include/infernix/types.h:879-898`) | `vision_seconds` and `vision_offload_window_seconds` (GPU event time), `vision_offload_evict_seconds` (lend bookkeeping), `restore_seconds` = 0, `evicted_bytes` = frames lent × stride, `staged_bytes`. The request log line adds placement, frames lent, experts evicted and step count. |
| Admission | `plan_request` replaces the media rejection: builds the `VisionControlPlan`, requires vision enabled, computes placement and frames, adds `service_work_quanta += steps` (the Qwen3.5 hybrid adds one per item). `feasible()`: frames needed ≤ frames − floor (25 %), else `PermanentlyInfeasible`. `quote()`: `residency_->lendable() ≥ needed`, else `TemporarilyBlocked` (other lanes may hold handoff leases). |

#### Frame lending and the two expert-cache fixes

The fixes: **lease waits** (window writes into lent frames could race promotion DMAs still landing
there (review finding R2-1, `local/workdirs/fn/plans/vision.md` §14), so a lease makes its users
wait on every in-flight batch that targets the run) and the **load serial** (the ABA hole above).
The rest is the lending API.

| Component | Change |
|---|---|
| `FramePool`, `LfruPolicy` (`program/expert_cache/expert_cache.{h,cpp}`, CUDA-free) | `FramePool`: state `kLoaned`, `lend(first, count)` (free frames only; the controller evicts held ones first), `give_back(first, count)`, `loaned_count()`. `LfruPolicy`: `erase` public as `evict(key)`; `set_capacity(n, victims&)` shrinks by evicting the lowest-score residents, or grows. |
| `CacheController` | Maintains `frame_key_[frame]` on load and evict. `choose_run(count, busy)`: the contiguous run with the minimum summed LFRU score, by sliding window over ~9,400 frames (~10 µs, estimated); free frames score 0, **loaned frames are ineligible**, busy frames (targets of in-flight loads) are avoided while a run without them exists. `lend`: policy `evict` plus an `Absent` entry write per held key, `set_capacity(capacity − count)`, frames to `kLoaned`. `give_back`: frames to free, `set_capacity(+count)`, `drain_queue`. |
| No pending frames at a boundary | `after_round` always calls `on_quiescent`, with slack 0 and `rounds_in_flight = 1` (§9.5 as built); `lend` asserts `pending_count() == 0` |
| **Load serial (ABA fix)** | A host `std::vector<std::uint64_t> load_serial_` per key, set when a copy is issued. `complete_load(key, frame, serial)` is true only if the entry is `Ready` and frame and serial both match. `Batch.loads` becomes `(key, frame, serial)` and `before_round` calls `complete_load` instead of the state-and-frame test, which also closes the case without lending. |
| `ExpertResidency::lend(count, compute, std::span<const cudaStream_t> writers) → FrameLease` | Busy bitmap from `in_flight_`; `choose_run`, `controller.lend`; apply the `Absent` writes to `table_host_` and `upload_table(compute)`; for every in-flight batch with a target in the run, `cudaStreamWaitEvent` on compute and each writer (normally none, thanks to the run choice, so the window does not queue behind the ~77 ms backlog); returns `{first, count, DeviceSpan(frame_base + first·stride, count·stride)}` |
| `ExpertResidency::give_back(lease, compute)` | The caller guarantees compute is ordered after every access. Loads released by `drain_queue` go through the `issue(commands, compute)` path of `after_round`: `table_ready_` on compute, then the copy stream waits. |
| Accounting, graphs | `lendable()` = frames − loaned − floor; stats gain `lent_frames`, `lend_evictions`. Decode graphs reach frames through `frame_base` and per-layer tables (data), so lending invalidates no graph. |

#### Costs and the offload default

Frame 2,764,800 B; weights 897,862,112 B; staging 135,886,912 B (50 frames). At the measured 27.6
GB/s (§19.2, PCIe Gen5 x8) the stream takes 32.5 ms per window: prelude 0.32 ms, 1.10 ms per layer,
merger 2.39 ms. GEMM per patch: 27 × 30.45 MFLOP (layers) + 3.54 MFLOP (patch) + 16.5 MFLOP (merger,
per patch) = **842 MFLOP**; attention 27·4·P²·1152 FLOP. Only the fallback rate is measured; the
other rates are estimates.

| P (V tokens) | GEMM | Attn | GEMM, fallback 13.2 TF/s | GEMM, TC 100-150 TF/s | Attn, 60-120 TF/s | W_ws (offload) | Handoff frames |
|---|---:|---:|---:|---:|---:|---:|---:|
| 256 (64), smallest image | 0.22 TF | 0.01 TF | 16 ms | 1.5-2 ms | < 1 ms | 139 MB | 1 |
| 1,024 (256) | 0.86 | 0.13 | 65 ms | 6-9 ms | 1-2 ms | 149 MB | 1 |
| 4,096 (1,024), 1024² px | 3.45 | 2.09 | 261 ms | 23-34 ms | 17-35 ms | 190 MB | 2 |
| 8,160 (2,040), 1080p | 6.87 | 8.28 | 521 ms | 46-69 ms | 69-138 ms | 244 MB | 4 |
| 16,384 (4,096) | 13.8 | 33.4 | 1.05 s | 92-138 ms | 0.28-0.56 s | 352 MB | 8 |
| 65,536 (16,384), item cap | 55.2 | 534 | 4.2 s | 0.37-0.55 s | 4.5-8.9 s | 1,002 MB (363 frames) | 31 |

- **Placement.** At chunk 1024 (`work_` ≈ 0.5 GB) with 0.25 s steps, images up to P ≈ 8-12K
  (V ≈ 2-3K tokens) take W: one step; only the 1-4-frame handoff run is lent, so at most that many
  experts are evicted. The P = 65,536 item lends ~394 frames (4.2 %) and runs in ~20-40 steps.
- **Offload overhead against resident, TC routes.** Stream-bound below ~1.1 ms of compute per
  layer: +30 ms at P = 256, +22 ms at P = 1,024. Compute-bound above: +2 ms at P ≥ 4,096 (prelude
  and layer 0 exposed ~1.4 ms, merger tail ≤ 0.6 ms). On the fallback every size is compute-bound
  at ~2 ms, so V0 makes small images relatively more stream-bound.
- **TTFT** (§19.2: pp4096 413 tok/s at chunk 1024, 659 at chunk 4096; pp512 265). LM prefill of
  1,024 image tokens takes ~1.6-2.5 s; the tower is 2-4 % of it with TC routes, 12-15 % on the
  fallback. A 64-token image plus a short question has TTFT ≈ 0.3-0.5 s; offload's +30 ms is 6-10 %.
- **Resident alternative.** 325 frames lost permanently ≈ 1.3 % of decode on every request, text
  included (scaled from §19.2's 436-frame reserve A/B, +1.8 % tg512). Resident wins only for many
  tiny images with short answers; offload wins once a small image's answer exceeds ~250-300 tokens
  at ~134 tok/s, and always for text traffic.
- **H2D FIFO.** After another lane's prefill chunk, up to ~77 ms of promotion DMA can sit ahead of
  the weight stream (C > 1 only; VM4).

#### Prefix cache, flags, serving and video

- **Prefix keys.** `block_hashes` and the cumulative vision `block_extras`
  (`qwen3_5/program/prefix/block_keys.cpp:24-82`) bind content, grid and placement; text after an
  image carries its key. §19.3.1 compares them exactly; no vision mechanism is needed.
- **Frontier.** The hybrid quote drops candidates whose frontier is strictly inside a vision span
  (`qwen3_5/program/prefix/hybrid_program.cpp:291-295`; §19.3.1 `TapExclusion`), so encoded
  items are exactly those with `token_begin ≥ F`; `plan_request` asserts no straddle. Service
  units: one per window step (Qwen3.5: one per item). §19.3.1 disables its C > 1 coalescing (P8)
  for vision prompts.
- **After reuse.** The new prompt's positions cover the whole conversation, earlier images
  included, with the prompt's `rope_delta`, so nothing RoPE-related is cached. The prepended MTP
  cell F−1 takes token F; if an item starts at F, token F's embedding is handoff column 0, so no
  bridge is needed. Multi-turn reuse is the largest vision speed-up: an image's LM prefill takes
  seconds. **Host RAM:** Vision pins only the tower weights (856 MiB); the handoff lives in lent
  frames and payloads are pageable (≥ 0.4 GB live media), budgeted jointly with §19.3.1's tier.
- **Flags.** Unchanged: `--vision`, `--vision-max-merged N` (default 32,768, clipped by the item cap
  to 16,384; the guide documents TTFT per 1K merged tokens), `--media-cache-mib`,
  `--media-live-mib` (≥ 131,072 patches × 3,072 B = 402 MB for the prompt cap),
  `--media-preprocess-threads`. New `--vision-offload auto|on|off` (§18):
  `include/infernix/types.h:375` becomes an enum, `src/models/load_options.h:10-20` resolves `auto`
  per architecture, parsing in `src/serve/serve_options.cpp:626-643, 786-787` and
  `apps/cli/options.cpp:99, 185`, tests `test_cli_options.cpp` and `test_serve_options.cpp`, no
  shim. **Protocol** unchanged (`image_url`, `input_image`, `video_url`, `input_video`,
  `vision_disabled`).
- **Docs.** `docs/qwen3_8-flash-next.md:10` ("Not yet supported: vision") becomes usage notes (TTFT
  per image token, offload default, frames lent, steps for huge items, video token cost: 2 fps, up
  to 768 frames); `docs/cli.md:231-233`; the README Flash-Next section;
  `docs/maintainer/linear-tuning.md`; the `include/infernix/ops/linear.h` and `qsa.h` contracts; at
  merge time on the base branch (§19.3.0): §9.1-§9.2 (vision frame role, lending, ABA fix) and
  §19.2 (status and a summary of VM1-VM8).
- **Video** takes the same paths (per-frame spans and timestamps in the frontend, `segment_count =
  t` in the per-item equal-length attention, per-frame grids in `get_rope_index`): no extra
  mechanism, one 4-frame smoke test, and a documented token cost (2 fps, up to 768 frames).

#### V1-V6 as built (2026-10-05)

- **V1.** `qsa_index_query(q, norm_weight, rope_positions)` and `QsaBatch::{rope_positions,
  block_start_rope}` (read by `qsa_pool_keys` only): pair i rotates by axis i % 3. VT2 adds distinct
  random axes to the index-query cases and real M-RoPE image spans to every pooling case, plus an
  image case (a 1,024-column prefill through two images, decode W = 1, verification W = 5 / 8 at
  B = 2, a row starting at position 3). Passes.
- **V2.** `program/rope_positions.{h,cpp}` (`LaneRope`, `rope_of`, `block_start_rope`, `stage_rope`,
  `stage_mtp_chunk_rope`); io gains `rope` and `block_rope` before `ngram` (a constructor check),
  `mtp_rope` and `mtp_block_rope` after the MTP cells; `mtp_io_` gains the draft steps' RoPE words
  before `drafts`. `kMtpChunkColumns` lives in `forward.h`. VT3 (`infernix_qwen4_exp_rope_positions_test`,
  host, against an independent port of the frontend's position rule) passes. **Text gate:** greedy
  ids of code / story / long, plain and MTP, are identical to the base build (6 of 6).
- **V3.** `CacheController::complete_load(key, frame, serial)`; `Command::serial` on copies; the
  residency publishes a landed batch's load only when it is the key's newest copy in that frame.
  VT4's ABA case without lending passes.
- **V4.** `FramePool::lend/give_back` (state `kLoaned`), `CacheController::choose_run/lend/give_back/
  frame_key` (per-frame keys kept on load, eviction and relocation), `ExpertResidency::lend/
  give_back/lendable` with the lease waits; `resize` (memory R2) leaves the pool unchanged while
  frames are lent, so the next boundary after the return resizes. VT4's lending cases pass.
- **V5.** Deviation from the design: Qwen3.5 keeps its workspace plan and handoff placement and calls
  the shared stages (`qwen3_5/execution/vision_tower.{h,cpp}`: `vision_embed`, `vision_layer`,
  `vision_merge`); only Qwen4Exp runs the layer-major `VisionTowerPass` over its own layout. Both
  models use one op sequence, and Qwen3.5's output is bit-identical by construction without
  re-planning its memory. `VisionWeightStream` moved to `vision_weight_stream.{h,cpp}` (Program-free)
  with the fence, the prelude → layers → merger order, `finish()` and the failure-path sync.
  `make_vision_overlay_layout` and `parse_vision_config` are shared; Qwen4Exp binds the tower under
  Qwen3.5's parameter names and keeps `VisionOverlayAssets{pool = nullptr}`.
  `--vision-offload auto|on|off` (`auto`: on for Qwen4Exp, off for Qwen3.5) with option tests.
- **V6.** `program/vision_window.{h,cpp}` (pure planning: placement W/L, frames, steps,
  `visual_slice`; host test `infernix_qwen4_exp_vision_window_test`) and `program/vision_program.cpp`
  (plan, reserve, stepped encode, release, staging). `ForwardBatch::vision` and `MtpCall::vision`
  scatter visual embeddings after the token embedding. First GPU smoke (Flash-Next, dense8m, INT8
  KV): the chart answer equals Quasar's ("NIFER VISION 731；3；左侧"), the natural image answers
  correctly; ids identical with offload on and off and between plain and MTP. **Open:** the encode
  window takes 1.4 s for either image with offload on or off, against ~10-40 ms estimated; being
  diagnosed (eager module loading and an nsys trace).

#### Steps

Each step builds and tests on its own. V0, V1, V3 and V5 can run in parallel; V2 needs V1, V4 needs
V3, V6 needs V0-V5, and V7-V8 follow V6. Total ≈ 12.5-13.5 working days plus V7's long CPU reference
runs. Across tracks (§19.3.0): V1 rebases onto prefix P0's QSA tails fix and QSA oracle test, which
land first. V6 needs n-gram S1 (§19.3.4). V2 rebases after concurrency S1 and n-gram S2/S3, which
restructure `verify` and `mtp_draft`. V3 merges early (merge order item 2); V0-V2 and V4-V8 merge
in item 5. Whichever of V4 and concurrency S4 (fill-phase landing) merges second adds a VT4 case
that lends during the fill phase with landing reservations outstanding.

| Step | Work | Main files | Effort, GPU | Exit criteria |
|---|---|---|---|---|
| **V0** | BF16 TC routes for the 8 shapes, 4304 tail TMA variant, tests, bench, contract, tuning doc | `src/ops/linear/bf16/{shapes/*.cu, bf16_a16_tma_mma.cuh, bf16_mma_common.cuh, bf16_dispatch.cpp, bf16_shapes.h, sources.cmake}`, `include/infernix/ops/linear.h`, `test_bf16_a16.cpp`, `linear-tuning.md` | 1.5-2 d, yes | VT1; VM1 ≥ 5× fallback at T ≥ 1,024 |
| **V1** | QSA M-RoPE op and its FP64 cases | `qsa.h`, `qsa.cu`, the QSA oracle test from prefix P0 (§19.3.0), `tests/ops/*.cmake` | 1 d, yes | VT2 |
| **V2** | RoPE plumbing: uniform `[T,3]`, `LaneRope`/`rope_of`, io and `mtp_io_` placement, block starts, MTP sub-chunks, Forward/MtpCall/AttentionCall/QsaBatch wiring, ForwardTap index queries | `program.cpp`, `program/rope_positions.{h,cpp}`, `execution/forward.{h,cpp}`, `test_rope_positions.cpp` | 1.5 d, gate | VT3; VM3 (ids and teacher-forced logits bitwise equal HEAD) |
| **V3** | Load serial (ABA fix), independent of vision | `expert_cache.{h,cpp}`, `expert_residency.{h,cpp}`, `test_expert_cache.cpp` | 0.5 d, no | VT4 ABA cases; LFRU-equals-replay unchanged |
| **V4** | Lending API, `quote`/`feasible` hooks, host tests | same, `program.cpp` | 1 d, no | VT4 lending cases |
| **V5** | Shared tower refactor (Qwen3.5 encode on top), weight-stream fixes, overlay builder, config parse; Qwen4Exp config, binding, resources, parameters; engine and frontend flags; `--vision-offload` | `qwen3_5/execution/{vision_tower.*, vision.*, vision_overlay.*}`, `qwen3_5/load/vision_overlay.*`, `qwen3_5/config.*`, `qwen4_exp/{config.cpp, load.cpp, weights.h, execution/parameters.*}`, `qwen4_exp_instance.cpp`, `types.h`, `load_options.h`, `serve_options.cpp`, `apps/cli/options.cpp`, option tests | 1.5-2 d, Qwen3.5 regression | VT8 (Qwen3.5 embedding bit-identical); option tests |
| **V6** | Program window: `LaneVision`, placements W/L, steps and `InProgress`, cancellation, guards, scatter, MTP visual embeddings, release paths, telemetry. Needs n-gram S1's de-duplication (§19.3.4) and adds no copy of it | `program/vision_window.{h,cpp}`, `program.cpp`, `execution/forward.{h,cpp}` | 2.5 d, yes | VT7, including C = 2, stress and lease accounting |
| **V7** | FP64 tower and multimodal reference, real tests | `tools/flash_next/reference.py`, `test_vision_real.cpp`, `test_forward_real.cpp`, `tests/test_flash_next_tools.py` | 2 d + long CPU runs, yes | VT5, VT6 a-c, VT9 |
| **V8** | Measurements and docs | docs listed above | 1 d, yes | VM1-VM8 recorded in §19.3.2, adverse results and rejected approaches included; defaults confirmed or revised; §9.1-§9.2 and §19.2 updated on the base branch at merge (§19.3.0) |

#### Tests

| # | Test | Checks |
|---|---|---|
| VT1 | Op BF16 routes (`tests/ops/linear/test_bf16_a16.cpp`) | The 8 shapes at V0's T set against the FP64 Linear oracle, full comparison to T = 4,100 and sampled beyond; 4304 tails at T not a tile multiple; weight preservation |
| VT2 | QSA M-RoPE (cases added to the FP64 QSA oracle test that prefix P0 creates with its tails case, §19.3.0; QSA has no standalone oracle test at HEAD, §19.2 M3) | Naive FP64 oracle for `qsa_index_query` and `qsa_pool_keys`: distinct axes (an image grid inside a sequence); a block start before the call (start ≡ 1, 2, 3 mod 4); prefill width 1,024, decode W = 1, verify W ∈ {5, 8}, B ∈ {1, 2}; pooled planes read back from the paged plane; the prefix P0 tails fix (`tail_kernel`, `commit_tail_kernel`), which lands first, stays checked against the same oracle (§19.3.0) |
| VT3 | RoPE staging (`tests/models/qwen4_exp/test_rope_positions.cpp`, new, host) | Over a synthetic image prompt from the real `assign_positions`: `rope_of` for prompt tokens, forced tokens past the prompt, decode, verify and draft cells; MTP sub-chunk blocks (prepend; `first` not a multiple of 4 or 512) and block starts; visual columns for chunks and MTP cells, including an image starting exactly at a chunk boundary and a last MTP cell whose token is the next chunk's first image token; io invariants (`rope`, `block_rope` before `ngram`; draft RoPE before `drafts`) |
| VT4 | Expert cache (`test_expert_cache.cpp`, host) | LFRU-equals-replay unchanged; the first revision's "victims equal replay at changed capacity" check is dropped (`replay.py:565` runs one fixed capacity). `choose_run` picks the minimum-score run, skips loaned and avoids busy frames; `lend` evicts every held key in the run (entries `Absent`, no `frame_key_` into it) and shrinks capacity; `set_capacity` evicts the excess; `give_back` restores capacity and drains; **ABA**: load K→f (serial s1), lend f, give back, re-admit K→f (s2), then `complete_load(K, f, s1)` false and `(K, f, s2)` true; the no-lending variant K→K2→K; `loaned_count() == 0` after every release path |
| VT5 | Tower oracle (`reference.py` gains CPU FP64 `vision_tower(store, patches, grid)` over `model.visual.*`, BF16 rounding at block boundaries as for text, checked once against transformers `Qwen4ExpVisionModel` in FP32); `test_vision_real.cpp` (skips without `INFERNIX_QWEN4_ARTIFACT`) | 256 × 256 (P = 256) and 448 × 320 (odd grid) by relative RMS; offload against resident **bit-identical**; an item alone against the same item in a 2-item pass **bit-identical**; placement W against L bit-identical |
| VT6 | Multimodal model oracle (`test_forward_real.cpp`; `reference.py --patches npz --grid t,h,w` with an independent `get_rope_index` port, scatter before expand, 3-axis RoPE for main q/k, indexer query and pooled keys) | (a) ~150-token prompt with a 64-token image: per-layer residuals and last logits, plus the wiring check selection cannot hide: rotated index queries (new tap) and the pooled-key plane read from the KV pool against the reference's M-RoPE values. (b) Teacher-forced continuation of (a): W = 1 decode steps until ≥ 2 pooled blocks complete, the first starting inside the prompt, then one W = 5 verify-shaped call; logits there (`block_start_rope`, `index + delta`, verify widths). (c) **Mandatory** > 2,051 tokens: ~200 text, a 32 × 32 merged-grid image (1,024 tokens), ~900 text (~2,130 in all); last-position logits and residual RMS, so selection runs over M-RoPE pooled keys. CPU FP64 run 15-40 min (estimated), only while the engine holds no host RAM. |
| VT7 | Engine end to end (GPU, gpu.lock) | Text gate: tg512, code and story greedy ids equal HEAD, teacher-forced logits bitwise equal (`--dump-logits`, three frozen texts). Image chart question, greedy: sensible answer; ids equal with offload on and off, and plain against `--spec mtp` (C = 1). Image at a chunk boundary with `--prefill-chunk 128`: no error, MTP acceptance close to chunk 1024. **C = 2:** lane B decodes greedily while lane A (an image, `max_tokens 1`) is admitted right after B's prefill chunk with the promotion backlog live; B's ids equal B alone (B is at batch 1, so exactly). **Stress:** 10 back-to-back image admissions, each right after another lane's prefill chunk; each image's ids equal its C = 1 resident run. **Lease accounting:** cancel a large image after its first `InProgress`; run abort, discard, finish and `fail_all`; after each, frames lent = 0 and capacity restored (idle log line). Serve `image_url` and `input_image`; one 4-frame video. After §19.3.1: a second turn on the same image whose `cached_tokens` cover it and whose answer equals a cold run. |
| VT8 | Qwen3.5 regression | `test_vision_workspace` and the vision op tests; a Qwen3.5 image embedding bit-identical before and after the refactor; a Qwen3.5 image smoke if an artifact is available; a Quasar vision smoke (its BF16 tower changes route; within oracle tolerance) |
| VT9 | Python | `py_compile` and `tests/test_flash_next_tools.py` for the reference changes |

#### Measurements (in order)

| # | Measurement | Decides |
|---|---|---|
| VM1 | `infernix_linear_bench --qtype bf16 --policy a16`, 8 shapes, T ∈ {64, 256, 1024, 4096, 16384, 65536}, fallback against registered, TFLOP/s | V0 acceptance: ≥ 5× the fallback at T ≥ 1,024 (target) |
| VM2 | nsys of the tower at P = 1,024 / 4,096 / 16,384 | `R_gemm`, `R_attn` |
| VM3 | Text gate: tg512 plain and MTP, pp4096; per-round nsys time of the rope, index-query and pool kernels | Same ids and logits as HEAD; kernels within ±1 % |
| VM4 | Window at P ∈ {256, 1,024, 4,096, 8,160, 16,384, 65,536} and 4 × 1,024 in one prompt: resident against offload, W against L, step durations, start delay behind the promotion FIFO at C = 2, pageable patch copy | W/L threshold; the FIFO risk |
| VM5 | Image TTFT: preprocess, window, LM prefill, first decode; n-gram reads per chunk before and after de-duplication | Where image TTFT goes |
| VM6 | tg512 at 9,443 against 9,118 frames, same build | The ~1.3 % resident cost; the offload default |
| VM7 | C = 2: lane B's longest decode-round gap while lane A encodes a 16K-token item, against `kVisionStepSeconds` | The step length |
| VM8 | Expert hit rate over the first 64 decode tokens after an image, against a text prompt of equal length (L especially) | Recovery cost of lending |

#### Risks

| Risk | Mitigation |
|---|---|
| The RoPE and QSA plumbing touches every attention call; a mismatch silently changes text | Uniform `[T,3]`, the io-placement assertion, VT3, the bitwise teacher-forced logit gate |
| M-RoPE wiring errors are invisible below 2,051 tokens | VT6a taps on index queries and pooled keys; the mandatory VT6c |
| Ordering across three streams (compute, transfer, promotion copy) with lending | Fence before the first upload, lease waits on targeted batches, load serials, the final transfer event on every exit path, C = 2 and stress id equality |
| Tower speed depends on V0; slower routes shift large-image TTFT and the W/L threshold | VM1-VM2 before the defaults are fixed |
| Huge items: a 16K-token image still costs ~5-9 s of tower (attention-dominated; D72 padded to 128 in the packed kernel, `kernel.cuh:18`) plus ~25-40 s of LM prefill | Steps keep other lanes live; TTFT stays high (documented) |
| The tower refactor, upload order and BF16 routes also change Qwen3.5 offload and Quasar's vision numerics (within oracle tolerance) | VT8 requalifies Qwen3.5 vision |
| Host RAM ≈ 65.3 GiB pinned plus media buffers (≥ 0.4 GB live) and the n-gram row cache: inside the loader floor but tight (estimated) | Joint budget with §19.3.1's Host tier |
| Pageable patch upload may queue behind promotion DMA at C > 1 | Measured (VM4), not fixed |

#### Decisions and rejected alternatives

| Question | Decided (default, 2026-10-04) | Alternative |
|---|---|---|
| Offload default | `--vision-offload auto\|on\|off`, `auto` = on for Flash-Next, off for Qwen3.5 | Model-dependent `on\|off`; `off` everywhere |
| Per-item cap | Keep 16,384 merged tokens (~25-40 s of prefill plus a 5-9 s tower at that size) | A lower default such as 4,096, a documented resolution-for-TTFT trade |
| MTP visual embeddings | vLLM / Qwen3.5 convention | Placeholder embedding (only acceptance changes) |
| Video | In scope: same code, one smoke test | Images only first |
| Frame floor vision never borrows | 25 % | Another share |
| `kVisionStepSeconds` | 0.25 s initially, tuned by VM7 (a few hundred ms of stall per step, against ~2 s per prefill chunk today) | Shorter steps (less stall for other lanes, more steps) |
| V0 shapes | All 8, including Quasar's merger [5120,4608]; Qwen3.5 vision requalified (VT8) | The 7 Flash-Next shapes; Quasar's merger fc2 stays on the fallback |
| Text BF16 prefill projections on tensor cores | A separate later task with its own quality gate | Fold into V0 (changes text numerics) |

| Rejected or deferred | Reason |
|---|---|
| Quantized vision weights (Q8 to halve the stream, or Q4/Q5) | Rejected: below upstream BF16 (user rule) |
| Elastic resident tower (keep the 325 frames lent between close images) | Deferred: saves ≤ 30 ms per small image, adds policy |
| Zero-copy weight reads inside the GEMMs | Rejected: the tiles re-read weights, and the x8 link cannot sustain it |
| Content-keyed embedding cache | Deferred: with TC routes the tower is ≤ 70 ms for ≤ 1K-token images against seconds of LM prefill, and repeats are covered by prefix reuse; stronger if V0 slips (~0.3 s per 1K-token image) |
| VMM eviction (Qwen3.5 style) | Rejected: frames are clean caches, so lending needs no restore |
| Fall back to `work_` or the 384 MiB reserve when frames = 0 (review) | Rejected: nothing regresses (Qwen4Exp has no vision today); frames are 0 only below the reserve; the handoff must survive prefill chunks, which reset `work_` (`forward.cpp:107`); the reserve is measured driver and graph headroom (367-377 MiB free during load, §19.2), not slack |
| 1-D RoPE array for lanes without media (review) | Replaced by uniform `[T,3]`: a decode, verify or draft graph mixes lanes; equal axes are bit-identical in every RoPE implementation involved (`rope.cuh:64-71, 120-127`, `qsa.cu:62-67`) |
| Column-invariant vision Linear routes in T (review) | Not adopted: the per-item pass gives batch invariance without constraining routes or testing every schedule |
| Per-lane device plane of RoPE positions, I32 [3, max_context] (review) | Not adopted: a gather kernel in every forward call (graph nodes included) and a QSA contract reading lane planes; host `rope_of` covers every case for 12 B of io per column; prefix reuse needs no state either way |
| Capacity events in the replay tool (review) | The replay-equality assertion for lent capacity is dropped instead (VT4) |

#### Incidental findings (outside this deliverable)

- **Text BF16 prefill projections use the same fallback.** In `prefill1.sqlite`,
  `bf16_general_gemm_kernel` takes 557.5 ms of 14,821 ms of kernel time (3.8 %): the QSA QKVG
  projection [13952,2560], 24 calls × 22.2 ms at T = 4096, and [96,2560], 72 × 0.34 ms (measured).
  Tensor-core routes would recover ~0.45 s per 8K-token prefill (estimated), image prompts
  included. Prefill numerics would change, so this is the separate task above.
  **Done 2026-10-05** (branch `claude/fn-bf16-text`): both problems are registered BF16 problems
  ([linear tuning §6](linear-tuning.md)); T <= 8 is unchanged. Op-level: 21.5 -> 1.23 ms per
  `[13952,2560]` call at T = 4,096, ~0.24 s per 4K chunk over the 12 QSA layers.
  End to end (dense8m, ABBA): pp200 +4.6 %, pp4096 +0.3 %, pp16384 +0.6 %, tg512 unchanged; paired
  NLL +0.011 +/- 0.009 (rounding level). The long-prompt gain is far below the op-level estimate;
  the cause is not established (a profile of both builds would show where the 22 ms went).
- **Prefill io upload through the copy-engine FIFO (unverified).** `run()` uploads the io layout
  with `cudaMemcpyAsync` (`program.cpp:979`), sharing the FIFO with up to ~77 ms of promotions
  issued after the previous chunk; decode avoids this with `upload_pinned` (§19.2). Worth checking
  in a prefill trace; not part of this plan.

### 19.3.3 Dense Q8 decode kernels

Design, adversarially reviewed, nothing implemented or run on the GPU (worktree
`NInfer-V3-flashnext`, designed against `46a56fc8f`, 2026-10-04; `file:line` references are at that
commit, where `program.cpp:N` is now `program/program_impl.h:N−2`, §19.3.1). Labels: **measured**
(an nsys trace or a §19.2 microbenchmark), **verified** (read in the code), **estimate** (the fit
below), **inference** (from one measured point or from code structure; M1/M2 confirm it). Trace
analysis: `fn/plans/q8-trace-analysis.py`.

The dense Q8 decode routes lose time to how they schedule memory, not to compute, and fall off a
cliff above 8 columns. Estimated savings per main forward:

| Phase | Content | Numerics | T = 1 | W = 4 | T = 9-16 |
|---|---|---|---|---|---|
| 0 | Fused small MoE dispatch with the route log; router weight loads before x staging; preloaded RMSNorm for the MTP `hidden_norm`; since 2026-10-04 (§19.3.6): E2M1 decode in registers in the expert kernels, the GDN control GEMV on more SMs, QSA attention accumulators in registers | Bit-exact | −0.28 to −0.4 ms (plain), plus −0.18 to −0.42 ms from the 2026-10-04 items | −0.33 to −0.6 ms, plus −40 µs per round for `hidden_norm` (MTP), plus −0.7 to −1.6 ms per W = 4-5 round from the 2026-10-04 items | — |
| 1a | K1, a register-streamed Q8 kernel, for all 10 Flash-Next Q8 geometries | Bit-exact at T ≤ 8; T = 9-64 rounds differently where K1 replaces the MMA tiles (1a-ii), column-consistent with T ≤ 8 | −0.41 to −0.51 ms (of 3.09) | −0.75 to −0.91 ms (of 3.62) | ≈ −11 to −17 ms (inference; HEAD's MMA times unmeasured but for one shape) |
| 1b | Split-K HC mixer (§8.3 K1a/K1b), inject folded into its producers, shared expert through `linear_swiglu`: 10 fewer launches per layer | Rounding change, more precise than HEAD and upstream | −0.55 to −0.7 ms | −0.55 to −0.7 ms | — |
| 2 | PDL weights-first, L2 warming during `cpu_wait_kernel`, a per-weight L2 class; the shared expert on its own graph branch (2026-10-04) | Bit-exact | −0.15 to −0.45 ms (unmeasured mechanisms), plus −0.1 to −0.25 ms for the branch | — | — |
| 3 | Tensor-core K2, only if its trigger fires | Rounding change | — | — | — |

End-to-end gains are typically 50-80 % of kernel deltas: the fork overlaps work, CPU-served misses
are often the long pole, and the GPU is ~11 % idle in MTP decode.

#### Shapes, widths and measured times

Scope: every `dense8m` Q8_G32_FP16 RowSplit weight, verified in
`out/flash-next/qwen3_8_flash_next_nvfp4_dense8m.ninfer.conversion.json`, which lists exactly these
Q8 matrices plus the 8-bit head. The 8-bit head uses `projection_fp32`; the main QSA projection
[13952, 2560] and GDN control [96, 2560] are BF16. Sources: **A** `fn/mtp/prof/mtp3b.sqlite` (MTP
max 3, T ≤ 4, 400 rounds, before the few-row route of `ecaf50e16`); **B**
`fn/prof/bench3_old.sqlite` (plain tg512, SIMT); **C** `fn/prof/bench4.sqlite` (plain tg512, the
rejected skinny GEMV: a warp per 2 rows, register loads, 16-code chunks); **D** §19.2
microbenchmarks (CUDA graph, weights cycled through 256 MB). µs per call; per-forward totals use B
at T = 1 and A's T = 4 at W = 4 (D's 7.0 for HC down).

| Weight [N, K] | MB | Calls: main / drafter per round | T = 1 (B) | T = 1-3 / 4 (A) | T = 5 / 8 (D) | Skinny T = 1 (C), CTAs | Per forward T = 1 / W = 4 | Why HEAD is slow (verified; slope inferred) |
|---|---:|---|---:|---:|---:|---:|---:|---|
| GDN q/k/v/z [16384, 2560] | 44.56 | 36 / — | 31.14 | 34.08 / 34.88 | n.m. | **28.09** (1024) | 1,121 / 1,257 | 2048 short CTAs, 4.016 waves over 510 slots; every CTA re-reads x |
| GDN and QSA output [2560, 6144] | 16.71 | 36 + 12 / 1 per step | 11.97 | 12.70 / 12.83 | n.m. | 11.94 (160) | 575 / 617 | 320 CTAs at 2 per SM (118 registers): 1.40 TB/s, near the fit |
| HC down + inject [324, 10240]; final mixer down [320, 10240] | 3.53; 3.48 | 96 + 1 / 2 per step and 1 per catch-up; 1 per step | 4.7 (D) | 7.0 at T = 4 (D) | 8.5 / 10.9 | 11.42 (21) | 456 / 679 | Few-row route: 324 one-row CTAs re-read x, 6.6 MB of L2 → SM traffic per column; inference: the +0.75 µs per column slope |
| HC up [10240, 320], row stride 384 | 3.48 | 97 / 3 per step, 1 per catch-up | 5.50 | 5.98 / 6.05 | n.m. | **3.52** (640) | 534 / 588 | 1280 CTAs (80 registers, 3 per SM, 2.51 waves), 340 B per warp, 31 % of lane-phases busy; flat 5.5-6.1 µs, a latency signature |
| Shared expert gate/up [1280, 2560] | 3.48 | 48 / 1 per step | 4.89 | 5.63 / 5.92 | 7.7 / 7.4 | 4.16 (80) | 235 / 284 | 160 CTAs; 3 stages per warp, 2 in flight, so two serial DRAM trips; T = 5-8 on `SimtR8T8` (102-118 registers): 5.3 → 7.7 µs |
| Shared expert down [2560, 640] | 1.74 | 48 / 1 per step | 2.91 | 3.30 / 3.46 | n.m. / 4.0 | **2.56** (160) | 140 / 166 | One partial stage, one trip: 2.9-3.5 µs against a 2.4 µs fit |
| PLE key/value [12800, 2560] | 34.82 | 1 / — | 25.15 | 28.2 | n.m. | **22.46** (800) | 25 / 28 | As [16384, 2560] |
| MTP attention group [13952, 2560] | 37.95 | — / 1 per step, 1 per catch-up | — | 29.6 DRAM, **16.6** L2 | — | — | — | — |
| MTP embedding / hidden projection [2560, 2560] | 6.96 | — / per step and catch-up, at T and 4T columns | — | 4.9-5.2 at 4 columns; **31.1** at 12-16 (MMA) | — | — | — | The MMA cliff (below) |
| **Total** | | | | | | | **3,086 / 3,619** | |

Widths, verified: the adaptive probe is capped at the maximum (`program.cpp:1387`), the catch-up
runs `run_mtp` at T = W × batch (`:1488-1497`), the hidden projection has 4T columns
(`hn.view({H, S*T})`, `forward.cpp:520-521`).

| Workload | Main forward T | Drafter columns |
|---|---|---|
| C = 1 plain / n-gram 7 / MTP max 3 / MTP max 4 | 1 / ≤ 8 / ≤ 4 / ≤ 5 | Step: T = 1, hidden projection 4. Catch-up: T ≤ 4 / 5, hidden projection ≤ 16 / 20 |
| C = 2 MTP max 3 (§19.2 two-lane run) / max 4 / n-gram 7 | ≤ 8 / ≤ 10 / ≤ 16 | Catch-up ≤ 8 / 10, hidden projection ≤ 32 / 40 |
| C = 3-8 with speculation | 12-64 | Up to 160 |

Under the speculation gate (§19.3.5), main forwards exceed 8 columns only if S6 + S7 replace it;
the gated W = 1 catch-up still has 4B hidden-projection columns (inference: B ≥ 3 crosses 8).

**Calibrated floor (measured fit).** Trace C fits **t ≈ 1.4 µs + bytes / 1.6745 TB/s** within 0.27
µs ([16384, 2560] 28.09 against 28.0, PLE 22.46 / 22.2 (the largest residual), HC up 3.52 / 3.5,
shared down 2.56 / 2.4). 1,674.5 GB/s is the sustained-read probe ([linear benchmark
§9](linear-benchmark.md)); 1.6 TB/s overshoots [16384, 2560] by 1.2 µs, and an earlier 0.9 µs fixed
cost (L2-resident tiny kernels) was too optimistic. The fit holds for grids of at least one wave at
2 CTAs per SM ([2560, 6144] at 160 one-per-SM CTAs is 0.5 µs above it). K1 is the same class of
kernel and cannot beat it.

**SIMT route at T ≤ 8 (verified).** `select_q8_generic` (`q8_dispatch.cpp:43-50`) uses
`SimtR8T4`/`SimtR8T8` (eight rows per CTA, a warp per row, 1 KB of codes per stage, 2 stages, `cg`),
or the few-row `SimtR1T4W8`/`SimtR1T8W8` when ⌈N/8⌉ < 64 (N ≤ 504) and K ≥ 4096 (a row per CTA,
eight warps splitting K on scale-pair boundaries). Codes and scales reach shared memory by
`cp.async`; an iteration issues stage s + 1, waits `cp_wait<S-1>` and consumes s; x is loaded inside
the consume loop, a dependent L2 trip (`q8_a16_simt.cuh:162-188`). Lane l takes the 8-code chunk at
k = 1024s + 256p + 8l in an `fmaf` chain in increasing k, then `warp_reduce_sum` (`shfl_down`
16 → 1) and K-warp partials in warp order (`:212-232`). Small shapes reach 0.6-0.7 TB/s, big
1.3-1.4.

**Drafter L2 reuse (measured, A).** 397 of 1,290 [13952, 2560] calls take ~16.6 µs (2.28 TB/s,
above DRAM peak), each ~21 kernels (median 167 µs) after a ~29.6 µs call of the same weight: the
catch-up's projection, then draft step 1's. Drafter HC up takes 4.32 µs against 5.98 in the main
forward. Evict-first on drafter weights would destroy this.

**Lessons of §19.2's attempts.** Skinny (net 3.39 against 3.36 ms per round) shows register-direct
streaming reaching the DRAM peak, but +4.0 µs on HC down (21 CTAs: it needs a K split), and its
16-code mapping changed the text. Runtime-K `q8_a16_gemv` (HC up 4.39, gate/up 6.74, [2560, 6144]
15.2 µs; 3.73 ms per round in `fn/prof/bench3.sqlite`, 3.72 in §19.2, which marks that profile
inconclusive because it overlapped another session's GPU work) suggests, without a clean
measurement, that two rows per CTA without a K split lose on long shapes.

**The MMA cliff above 8 columns.** Measured (A): `q8_a16_mma_kernel`, grid 80, 128 threads, 343
calls in 400 rounds at 31.12 µs (p10-p90 30.6-31.7, 0.22 TB/s), each after `rmsnorm_generic_kernel`
with grid 3 (201) or 4 (142), i.e. the MTP `hidden_norm` and then the hidden projection at 12 or 16
columns (`forward.cpp:517, 521`). Verified: T = 9-64 selects `MmaR32T64` (`q8_dispatch.cpp:47`;
`MmaR32T96` and `MmaR32T128` above) = `Q8A16MmaSchedule<32, 64, 64, 32, 16, 2, 3>`: 4 warps, 32 rows
per CTA, BlockK 64, 2 weight stages (`q8_schedule.cuh:73`), no K split (`q8_mma_launch.cuh:18-20`).
0.78 µs per 64-wide K step is about one DRAM latency each, so narrow or long-K shapes become chains
of 40-160 steps: ≈ 14-20 ms per forward at T = 9-64 against ~3.6 ms at T ≤ 8 (per-shape inference in
the K1 table). It is reached by C = 2 with MTP max 4 (T = 10) or n-gram 7 (T = 16), by C ≥ 3 with
any speculation, and at C = 1 only by the catch-up hidden projection (16 columns at max 3, 20 at max
4). It is not shown to explain the §19.2 two-lane throughput: `--draft-tokens 3` kept that run's
main verification at T ≤ 2 × 4 = 8 (in its build `5429c31a4` too) and only the catch-up (≤ 32
columns) hit MMA, so §19.2's stated cause (twice the distinct experts) is not contradicted. For the
same reason the MTP story divergence in that run did not come from the dense Q8 MMA tiles, as
§19.2's "Two rows verify 2 × W columns, so they cross into MMA" assumes; only the n-gram run (16
columns) crossed. Another width- or batch-dependent route (BF16 dense, QSA verification attention,
the GDN record) must explain it (inference), and the gate's byte-identity check (§19.3.5, campaign
step 1) must cover it.

**Constraints.** FP32 accumulation and exact decode of codes × FP16 scales; per-column invariance for
T = 1..8 (`q8_a16_column_invariance`, `tests/ops/linear/test_q8_a16.cpp:86`), on which the C = 1
speculation equality rests (§11.3); fixed Q8_G32_FP16 storage, row stride `padded_k` (384 for
K = 320); [op development](op-development.md) qualification (naive FP64 oracle, exact decode,
graph-captured cases where PDL applies); no shims; no fused Op keeps a BF16 rounding for parity
(op development §6.1).

#### Phase 0: bit-exact non-Q8 fixes

Trace A, per main forward: router `projection_kernel` (grid 129) 6.63 µs × 50.2 = 333 µs;
`route_kernel` 4.64 µs × 50.2 = 233; count + scan + scatter 0.72 + 2.06 + 0.75 µs × 48 = 169;
`cpu_plan_kernel` 7.64 × 48 = 367; route → count gap mean **10.54 µs** (median 6.62, p90 29.54;
6.24 in B).

| Item | Evidence | Change | Estimate per forward |
|---|---|---|---|
| Route-log copy and dispatch memset/count/scan/scatter | D2D `cudaMemcpyAsync` (`forward.cpp:427-428`), counts memset (`moe_layer.cu:709`), three kernels (`:712-719`). Inference: the D2D node queues behind promotions on the copy engine (the p90) | For `entries ≤ 1024` (T ≤ 102 at top-10), one `dispatch_small_kernel<<<1,1024>>>`: zero counts in shared memory, shared-atomic count, today's scan, offsets/cursor/jobs/job_count, scatter, ids to an optional `route_log` destination. Larger calls keep today's path; the memcpy and memset nodes leave decode graphs. Rejected: zeroing in `count_kernel` races (160 CTAs at a 4096-token chunk) | −0.25 to −0.45 ms (MTP), −0.2 to −0.25 (plain) |
| Router [513, 2560] BF16 | Row loads already precede FMAs (`projection_fp32.cu:66-71`), but x is staged first by a runtime-bound `uint4` loop (`:51-55`) and `__syncthreads` (`:56`). 5.2 / 5.7 / 6.6 / 7.3 µs at T = 1-4 | Weight loads to registers first, then x as one unrolled `cp.async` batch, then sync | −0.08 to −0.15 ms |
| MTP `hidden_norm`, d = 10240 | Generic route for d > 8192 (`rmsnorm.cu:115-136`), median 16.1-16.5 µs × 3.2 per round (48.5 µs per forward) | Preloaded variant for 8192 < d ≤ 16384: each thread loads its 40 strided elements before summing; same order and shared tree (`rmsnorm.cuh:265-287`) | −40 µs per round (MTP) |
| E2M1 decode in the GPU expert kernels (2026-10-04, §19.3.6) | `canon::e2m1_x2` indexes `constexpr signed char kMag[8]` (`canonical_math.h:106-110`), which compiles to a stack store of the table and a local byte load per lookup. `cuobjdump` of the current `moe_layer.obj`: `gate_up_kernel<1>` 688 STL.64 + 656 LDL.S8 + 32 LDL.U8, `down_kernel<1>` 160 + 160, the 8-column instances the same plus spills. Each `unit_sums` unit does 8 lookups against ~5 LDS and 2 dp4a, so the compute after `stage_wait` is load/store-bound. Strata keeps codebooks in registers (`prmt`); its SYCL port measured an expert down call 80.8 → 43.1 µs and decode +7.5 % from that change (`20a2ccf`) | `canon::e2m1_x2` without a memory table: a 64-bit-constant shift or one `prmt` over the magnitude bytes {0, 1, 2, 3 \| 4, 6, 8, 12} with the sign applied by mask, fixed at compile time. It also serves `quantize_a4_block` and the CPU scalar reference. Exact integer transform | −0.3 to −0.38 ms of kernel time at T = 1 (realized −0.1 to −0.3 ms, less where CPU misses are the long pole); −0.5 to −1.3 ms per W = 4-5 round; prefill 8-column kernels unmeasured |
| GDN control [96, 2560] (2026-10-04, moved from "listed") | BF16 skinny GEMV on 6 CTAs (16 rows per CTA, 10 dependent 16-byte loads per lane): 3.21 µs at T = 1, 5.41 µs at MTP widths, × 36. Strata spread its equivalent over 4× the blocks (`cfd3b72`) | A small-N instantiation: one row per warp, all chunk loads issued before the FMAs in today's lane order and butterfly, 24-48 CTAs; the [13952, 2560] QSA projection keeps today's instance (872 CTAs, near the floor) | −0.04 to −0.05 ms (T = 1), ~−0.12 ms (W = 4-5) |
| QSA decode attention (2026-10-04) | `float acc[kMaxGroup = 16]` with runtime-bound loops lives in local memory (~2,498 LDL/STL per instantiation); the value loop is one page-table load and one dependent V load per token; 11.5 µs per call at T = 1, 16.2 µs at MTP widths, 12-14 calls per forward | Group size as a template parameter (accumulators in registers); four cells' V loads issued together and folded in order; the 12 head sums reduce-scattered in `warp_sum`'s xor order (Strata `fd95405`); the serial softmax sum keeps its order | −0.04 to −0.07 ms (T = 1), −0.1 to −0.2 ms (W = 4-5); does not grow with context (attention ≤ 2,051 tokens) |

Listed, not scheduled: the fused `gdn_gating_proj` form of the GDN control (a rounding change, a
further −0.06 to −0.08 ms); `split_rows` after GDN/QSA projections
(2.5 + 1.7 + 1.2 µs per GDN layer, QSA 2.0); `recurrent_record_kernel` (5.4 µs × 35.5);
`cpu_plan_kernel`; `route_kernel` (already a warp per column: a 10-step top-k latency, withdrawn).

**Phase 0 as implemented (step K0, branch `claude/fn-kernels`; built, not yet measured).** Rule
(user, 2026-10-04): reuse Infernix's existing kernels, templates and instances before adding a
kernel. Every item is a change to, or a new instance of, the kernel that serves the call today, so
Phase 0 adds no parallel kernel except the one-CTA dispatch, which shares the scan of the existing
route.

| Item | Implementation | Existing kernels considered |
|---|---|---|
| E2M1 decode | `canon::e2m1_x2` selects the magnitude by one `__byte_perm` over the bytes {0, 1, 2, 3 \| 4, 6, 8, 12} on the device and a 64-bit-constant shift on the host; new `canon::e2m1_x2_quad` decodes the four codes of a `row_quad` word with two byte permutes (magnitude and negated magnitude) and a sign mask. Exact integer transform, so `gate_up`/`down` (1 and 8 columns), `quantize_a4_block`, the CPU engine and `narrow_expert` keep their bits | — (a defect fix in the shared function) |
| MoE dispatch | `moe_dispatch(..., route_log, stream)`: calls of at most 1,024 entries run `dispatch_small_kernel` (one 1,024-thread CTA: shared-memory count, the scan, the scatter, the route-log copy); larger calls keep memset + count + scan + scatter, with the route log written by `count_kernel`. Both routes share one scan (`dispatch_scan`: warp-shuffle inclusive scans of counts and used flags, then of the warp totals), which also replaces the 20-barrier Hillis-Steele scan of `scan_kernel`. The model passes the route-log slot instead of enqueuing a D2D copy, so decode graphs lose the memset and copy nodes | `sparse_moe_prefill_select_count_kernel` / `sparse_moe_prefill_scan_kernel` (Qwen3.5): fixed 256 experts and top-8, fused with the router's top-k, and a tile-job layout (`route_job_columns`, tile bases) that `MoeDispatch`'s per-expert jobs and entry list do not have |
| Router [513, 2560] | `projection_kernel` (both mappings) loads its first row's chunks into registers before staging x, stages x as one unrolled `cp.async` batch, and reads segment fields at compile-time indices only; arithmetic unchanged | — (the existing kernel, reordered) |
| MTP `hidden_norm`, d = 10240 | `rmsnorm_generic_kernel<Epilogue, Preload>`: a preloading instance (Preload = 64, 256 threads) for 8192 < d ≤ 16384, ungated epilogues; same per-thread order and shared tree as the generic instance | `rmsnorm_cta_bf16x2_kernel<…, 512, 10, true, 10240>` fits the shape and alignment but sums bf16x2 pairs and reduces by `block_reduce_sum`, another order: it would change the drafter's bits (MTP acceptance), which Phase 0 must not. It remains the natural rounding-change option if one is ever wanted |
| GDN control [96, 2560] | `bf16_skinny_gemv_kernel<Tokens, Warps, RowsPerWarp, Preload>`: the existing kernel gains warp/row/preload parameters; the small-N instance (2 warps × 1 row, Preload = 16 chunks per lane, N ≤ 256 and K ≤ 4,096) runs 48 CTAs for the control and issues every chunk load before the FMAs; the default instance (8 × 2, loop) is unchanged | The registered BF16 schedules (`Bf16A16SlicedKMma`, `Bf16A16Gemv`, `Bf16A16Simt`, as for [256, 5120]) accumulate in another k order or by MMA, so registering [96, 2560] would change the GDN gating bits; `gdn_gating_proj` (fused) is a rounding change and stays listed |
| QSA decode attention | `attention_kernel<Storage, Group>` with Group ∈ {4, 8, 12, 16} (Qwen4Exp: 12); the head sums reduce-scatter in `warp_sum`'s xor order; the next token's key loads issue before the current token's sums; value loads issue in batches of 8 cells (the plan said 4; a scheduling choice) and fold in token order | — (the existing kernel, templated) |

A temporary toggle patch (`fn/rigs/kernels/k0-base-toggle-temporary.patch`, never committed)
restores each pre-K0 route under `INFERNIX_K0_BASE` for M0's same-binary A/B.

#### Phase 1a: kernel K1, `q8_a16_stream_kernel`

**Keep the T ≤ 8 arithmetic, change the memory schedule.** K1 keeps both SIMT routes' per-(row,
column) arithmetic: lane-to-k mapping on scale-pair boundaries, exact w = float(int8) · scale, an
`fmaf` chain in k order, an xor butterfly 16, 8, 4, 2, 1 (or a transposed butterfly over a value
set; it reproduces lane 0 of `shfl_down` bit for bit since IEEE addition commutes), K-warp partials
in warp order, `__float2bfloat16_rn`. So **T ≤ 8 is bit-exact with HEAD**, few-row route included:
C = 1 greedy ids cannot change and tg512 stays a valid A/B. **T = 9-16 changes rounding** (MMA → K1)
but is column-consistent with T ≤ 8: at C = 1 only the catch-up hidden projection moves (acceptance
may change, ids cannot); at C ≥ 2 verification rounding changes and dense Q8 becomes
concurrency-invariant to T = 16 (the column-invariant range §11.3 relies on grows from 8 to 16).

**Mechanism.** One launch, no shared-memory weight staging: all of a warp's weight loads are in
registers before any FMA, and several rows share each x load.

**Schedule.** `Q8A16StreamSchedule` has static `K` (K % 64 == 0), `R` rows per warp, `KSplit` (1,
or 8 warps splitting K on scale-pair boundaries), `Warps`, `MaxCols` 4/8/16 with `ColsPerPass` 4/8
(two passes over **register-resident** weights give 16), `Window` (8-code chunks per lane per row in
flight; `all` for K ≤ 2560 at R ≤ 2), `XStage` (`Shared`: `cp.async` once per CTA per pass;
`Global`: L1 path two chunks ahead; `Fp32Shared` for fused Ops), `Grid` (`Plain`; `Persistent` as a
sweep candidate only), `RowMap` (`Consecutive`, `Pair<M>`, `StreamQuad<H>`) and `Epilogue`. A warp
(1) loads the codes of its first `Window` chunks of every row (8 B per lane per chunk) and their FP16
scales (lanes 4j..4j+3 share a 16 B sector) with the default L2 policy, which may precede the PDL
wait because weights are immutable (op development §4.3); (2) waits on PDL as a consumer (Phase 2)
and stages pass p's x as an unrolled `cp.async` batch; (3) streams fully unrolled: per live column t
(predicated) and row r it decodes `w_i` once per chunk and runs `acc[r][t] = fmaf(w_i, x_i,
acc[r][t])` for i = 0..7, unpacking each x chunk once for all R rows and issuing chunk j + Window
while j is consumed; (4) reduces by a transposed butterfly over R × ColsPerPass values (KSplit > 1:
partials in shared memory, warp 0 adds in warp order), runs the epilogue, and starts the next pass
on the same registers; (5) triggers dependents after the last pass.

Every instance computes a column identically, so its bits do not depend on T within 1..16 (1..64
where K1 slices are selected). Registers ≤ 128 for 2 CTAs per SM, ≤ 168 for 1, zero spills
(`-Xptxas -v`); an instance that cannot hold two passes' weights ([2560, 6144] at R = 2 needs 96
code registers) uses R = 1. New files `q8_a16_stream.cuh`, `q8_stream_launch.cuh`, `q8_stream.cu`
(`src/ops/linear/q8/`); `ld_nc_na_u64` (`ld.global.nc.L1::no_allocate.v2.u32`) and an L2-policy
load in `src/ops/common/memory.cuh`. Fused Ops reuse the header, as `gdn_input_proj/q8` does.

**Dispatch.** Ten exact entries in `q8_shapes.h` and `kShapes` (`q8_dispatch.cpp:24-33`), one file
per shape under `src/ops/linear/q8/shapes/`. Each `select_*` returns K1 MaxCols 4 at T ≤ 4 and 8 at
T ≤ 8 (step 1a-i; [2560, 6144] only if M1 shows K1 faster by ≥ 0.8 µs per call at T = 4, otherwise
it keeps HEAD's SIMT route at T ≤ 8); K1 MaxCols 16 at T = 9-16 (1a-ii) unless M1 shows HEAD's MMA
tile faster there by more than 0.3 µs (inference: only HC up, MMA ~5 µs, is a candidate); at
T = 17-64 per shape M1's winner between the MMA tile and K1 in 16-column slices
(`for_each_token_slice`); above 64 today's MMA tiles. Expected (inference): K1 slices win for the HC
down shapes (T = 32: 2 × ~6 µs against ~100), [1280, 2560], [2560, 6144] and [2560, 2560] (20
columns: ~6 + ~5.6 against ~31); MMA wins above T ≈ 24 for [16384, 2560], [13952, 2560], [12800,
2560] and [10240, 320]. The sliced-K MMA schedules (`Q8SlicedKDefault<8|16,…>`) join the T ≤ 16
sweep as a baseline that needs no new code; since they are not bit-exact with K1, they must win
by > 0.5 µs.

Default configuration (sweep candidates in brackets); K1 estimates are 1.4 + bytes / 1.6745 TB/s
plus an exposed issue tail of ~(2 + T + 1.125·T/R) lane-ops per weight on 21,760 FP32 lanes at
~2.6 GHz (hidden on big shapes until T ≈ 16, ~0.1 µs per column on small ones); HEAD's MMA at
T = 9-64 is inferred at ~0.6-0.8 µs per K step per CTA:

| Shape | R, KSplit, warps, CTAs | Chunks per lane per row; x; bytes in flight per SM | K1 µs T = 1 / 4 / 8 / 16 | HEAD MMA: CTAs × K steps; per call; per forward |
|---|---|---|---|---|
| [16384, 2560] | 2, 1, 8, 1024 | 10, Window all; Shared 40 KB per 8-column pass; ~2 × 40 KB | 28.0-28.5 / 28.3-29.0 / 28.5-29.5 / 29-33 | 512 CTAs on ~510 slots; 45-60 µs; 1.6-2.2 ms |
| [12800, 2560] | 2, 1, 8, 800 | as above | 22.2-22.7 / 22.5-23.2 / 22.8-23.8 / 23.5-27 | — |
| [13952, 2560] | 2, 1, 8, 872 | as above | 24.1-24.6 / 24.4-25.2 (DRAM; HEAD median 26.9) / — / — | — |
| [2560, 2560] | 2, 1, 8, 160 | as above | ≈ HEAD (partly L2) / — / — / 6.6-7.5 | measured 31.1 at 12-16 |
| [2560, 6144] | 2 (1 at MaxCols 16), 1, 8, 160 (1 per SM) | 24, Window 12; Shared ≤ 96 KB [Global]; ~48 KB | 11.4-11.9 / 11.6-12.1 / 12.0-12.6 / 12.5-14 | 80 × 96; 60-75 µs; 2.9-3.6 ms |
| [324, 10240], [320, 10240] | 2 [4], 8, 8, 162 [81] | 5 per slice; Global two chunks ahead; ~20 KB [40] | 3.6-4.0 / 3.9-4.6 / 4.4-6.0 / 6-9 | 11 × 160; 80-125 µs; 7.8-12.1 ms |
| [10240, 320] | 4, 1, 8, 320 | 1-2 (lanes 0-7 hold 2); Shared 5 KB per pass; ~20 KB | 3.5-3.7 / 3.8-4.1 / 4.2-4.6 / 5.0-5.8 | 320 × 6; 5-7 µs |
| [1280, 2560] | 2, 1, 4, 160 | 10; Shared 40 KB per pass; ~20 KB | 3.5-3.8 / 3.8-4.2 / 4.2-4.6 / 5.0-5.8 | 40 × 40; 24-31 µs; 1.2-1.5 ms |
| [2560, 640] | 4, 1, 4, 160 | 2-3; Shared; ~11 KB | 2.4-2.6 / 2.6-2.8 / 2.9-3.2 / 3.4-4.0 | 80 × 10; 6-8 µs |

- **Little's law.** 160 CTAs need ~10.5 GB/s per SM, ~10.5 KB in flight at ~1 µs, so ~20 KB is
  enough; HC down at R = 4 (81 CTAs) needs ~20.6 GB/s per SM (~21 KB) and has 40 KB.
- **Grid sizing (recorded deviation from §8.3 rule 6).** 160-CTA grids idle 10 SMs; rebalancing
  2560 rows over 170 CTAs needs uneven rows for < 0.1 µs (estimate). [16384, 2560] keeps the plain
  1024-CTA grid, on which skinny measured at the fit.
- **HC down x re-read.** With an in-CTA K split, K1 re-reads x per row group: at R = 2, 162 CTAs ×
  20 KB × T = 26.5 MB at T = 8 and 53 MB at T = 16 (half at R = 4). The T = 6-16 cost rests on
  unmeasured L2 → SM bandwidth (HEAD's +0.75 µs per column back-solves to ~8.5 TB/s, inference).
  Phase 1b's split-K reads each stream slice once: 0.8 MB × T.

**Per main forward (estimate).** T = 1: −0.41 to −0.51 ms of 3.09 ([16384, 2560] −95 to −113 µs, HC
down −68 to −107, HC up −175 to −194, gate/up −52 to −67, shared down −15 to −24, PLE −3); W = 4:
−0.75 to −0.91 ms of 3.62; T = 9-16: ≈ −11 to −17 ms (HEAD ~14-20 → K1 ~3.1-3.8 ms; inference, gated
by M1). [2560, 6144] is excluded: K1 serves it at T ≤ 8 only if M1 shows a gain of ≥ 0.8 µs per call
at T = 4 (it would add 0 to −27 µs at T = 1, −34 to −58 µs at W = 4).

**The few-row route of `ecaf50e16` is superseded.** Its branch (`q8_dispatch.cpp:44`), `SimtR1T4W8`
/ `SimtR1T8W8` and their launchers are deleted: among `dense8m` weights only the HC-down shapes meet
its predicate, and K1 takes them bit-exactly at T ≤ 8, faster (4.7 → 3.6-4.0 µs at T = 1, estimate)
and without the cliff above 8. At implementation, grep the recipe A and `dense8h` conversion reports
for any other N < 505, K ≥ 4096 Q8 matrix; `select_q8_generic` stays for unregistered geometries.
§19.2's idea of running 9-16-column blocks as two ≤ 8-column launches is superseded too.

**K1 as built and measured (M1 sweep, 2026-10-04).** `q8_a16_stream_kernel`
(`ops/linear/q8/q8_a16_stream.cuh`, schedules in `q8_schedule.cuh`) is bit-identical to the SIMT
routes for T = 1..8 on all ten shapes (0 differing outputs; the [324, 10240] and [320, 10240] K
split reproduces the few-row W8 route's warp-order sum). The sweep (a graph of 200 calls cycling
≥ 256 MiB of weight copies, two runs agreeing within 0.5 %) gave K1 / previous route:

| Shape | T = 1-8 | T = 9-64 (K1 MaxCols 16 against the MMA tile) |
|---|---|---|
| [16384, 2560], [12800, 2560], [13952, 2560] | 0.72-0.96 (faster) | 4.1-19.9 (much slower) |
| [10240, 320] | 0.76-0.92 | 1.7-5.5 (slower) |
| [2560, 2560] | 0.91-0.98, but 1.03 at T = 4 | 1.5-6.9 (slower) |
| [1280, 2560] | 0.84-0.95, but 1.03 at T = 4 | 0.39-0.93 to T = 48, 1.02 at 64 |
| [2560, 640] | 0.88-0.96 to T = 7, 1.00 at 8 | 0.83-0.99 to T = 16, slower beyond |
| [324, 10240], [320, 10240] | 0.83-0.92 at T ≤ 2, 1.0 at 3, 1.04-1.16 at 4-8 | 0.15-0.46 (much faster) |
| [2560, 6144] | 1.16-1.29 (slower everywhere) | 1.7-8.7 (slower) |

Two expectations above did not hold: K1 MaxCols 16 is far slower than the MMA tiles at T = 9-16 on
the large shapes (the estimate of −11 to −17 ms per forward there was wrong), and the few-row route
is still the fastest for the HC downs at T = 4-8, so it is kept, not deleted. Each shape therefore
routes to its measured winner: K1 at T ≤ 8 except [2560, 6144] (SIMT) and T = 4 of [2560, 2560] and
[1280, 2560] (SIMT), the few-row route at T = 4-8 of the HC downs; K1 slices beyond 8 only for the
HC downs (to 64), the shared expert gate/up (to 48) and down (to 16); the MMA tiles elsewhere.
K1 and the SIMT routes reduce a column identically, so mixing them by width changes no bits;
column invariance now holds to 8 on every shape and to 64 / 48 / 16 where K1 runs beyond 8.
With the final routes no shape and width is more than 2 % slower than its previous route (second
sweep, two runs within 1.3 %). End to end (infernix_bench tg512 ABBA, frozen binaries, K0 against
K0 + K1, 2026-10-05): plain 97.29 → 99.80 tok/s (+2.6 %), MTP 149.62 → 151.44 (+1.2 %); generated
ids identical in both modes.

#### Phase 1b: fused Ops (rounding changes, more precise)

Op development §6.1 forbids a fused route from keeping intermediate BF16 roundings for parity, so
each fusion drops at least one. For T > 16 and for BF16 weights (recipe A) each Op runs a composed
route (today's kernels inside the Op, caller workspace, BF16 intermediates as a private precision
choice under the Op's criterion), so the Ops serve any T and weight with no new BF16 kernels.

**`hyper_connection_mix`** (new, §8.3 K1a/K1b): `(residual, norm_weight, down, up, streams, rank,
eps, x, Tensor* inject, WorkspaceArena&, stream)` plus
`hyper_connection_mix_workspace_capacity_bytes(...)`. Contract:
Rn = per-stream offset RMSNorm(R); z = W_down·Rn; m = SiLU(z[:rank]/S); inject = 2σ(z[rank:]/S);
u = W_up·m; x = (1/S)·Σ_s σ(u[sH+d])·Rn[sH+d]. x is BF16 [H, T]; inject FP32 [S, T], optional (the
final mixer passes none). `Forward::mix` (`forward.cpp:223-238`) becomes one call: 5 → 2 launches.
Today the five kernels cost, per forward (A): `norm_kernel` 1.36 µs, `gates_kernel` 0.84,
`collapse_kernel` 1.51, each × 104.7 (143 / 88 / 158 µs), plus the two projections.

| Kernel (T ≤ 16) | Grid | Steps | Output |
|---|---|---|---|
| K1a `hc_down_partials`, [324 or 320, 10240] | 4 K slices (one per stream, K = 2560) × 41 row groups of 8 rows = 164 CTAs × 8 warps; a row per warp, 10 chunks per lane | Weights first (20 KB per CTA); PDL wait; stage R[s, pass columns] and the norm weight; block-reduce each column's sum of squares in fixed order to inv[s, t]; write Rn = R·inv·(1 + w) in **FP32** to shared memory (80 KB per 8-column pass: 1 CTA per SM, 164 ≤ 170); K1 with `Fp32Shared` | FP32 `partial[s][row][t]` (4 × 324 × T × 4 B) in workspace; row group 0 also writes inv[s][t] |
| K1b `hc_gates_up_collapse`, [10240, 320] | 160 CTAs × 16 warps; a CTA owns 16 d, a warp one d with `StreamQuad<2560>` rows (d, H+d, 2H+d, 3H+d) | Weights first; PDL wait; read all partials (5.2 KB × T per CTA); z = Σ_s partial[s] in order s = 0..3; m = SiLU(z/S) in FP32 shared memory; CTA 0 writes inject[s, t] = 2σ(z[rank+s]/S) when requested; K1 over W_up rows | The butterfly gathers a column's 4 streams into one lane, which forms Rn from R and inv in FP32 and writes x[d, t] = bf16((1/S)·Σ_s σ(u_s)·Rn_s) |

- **Deviation from §8.3**, which takes the stream sums of squares from the previous kernel's
  epilogue: per-CTA statistics keep the Ops closed (no hidden cross-Op output), overlap the weight
  latency (≤ 0.3 µs exposed, estimate), and are CTA-local with one K slice per stream.
- **Cost.** ≈ 7.6-8.9 µs per mixer against Phase 1a's ~11.4-12.9 (five kernels and four edges):
  −0.34 to −0.42 ms per forward. K1b's partial reads are redundant (160 × 5.2 KB × T: 6.7 MB at
  T = 8, 13 MB at 16, from L2) and hidden at T ≤ 8; if M1 shows them exposed at 9-16, K1a gets a
  last-arriver reduction with a Program-owned zeroed counter (permitted by §8.3 rule 6, new here).

**Inject folded into its producers, shared expert through `linear_swiglu`.** Folding the inject into
the [2560, 6144] projection epilogue and `moe_combine` replaces the deferred `inject_norm` of an
earlier draft (see Rejected); the MTP input fusion (`forward.cpp:523`, inject = ones) keeps
`hyper_connection_inject`.

| Op | Math | T ≤ 16 | T > 16 | Model sites | Removes; estimate per forward |
|---|---|---|---|---|---|
| `hyper_connection_linear_inject(x, w, inject, residual, Tensor* y_tap, ws, stream)` | R[sH+d, t] = bf16(R[sH+d, t] + inject[s, t]·(W·x)[d, t]) from the FP32 product; `y_tap` = bf16(W·x) for `ForwardTap` only | K1 on [2560, 6144]; the epilogue lane for (d, t) reads 4 residual values and inject[·, t] and writes 4 | MMA to BF16 workspace, then the private inject kernel | GDN, QSA and MTP attention outputs (`forward.cpp:343`, `:409`; injects `:141`, `:539`); `Forward::attention` and `Forward::gdn` take `inject` and `residual` and return nothing | `inject_kernel` (1.46 µs × 103.7 = 152 µs) less ~0.1-0.2 µs per epilogue call: ≈ −0.12 to −0.13 ms with the row below |
| `moe_combine` inject form (`offloaded_sparse_moe.h:134`; adds `inject`, `residual`, `y_tap`) | R = bf16(R + inject ⊗ (Σ_i w_i·y_i + s·y_shared)), FP32 sum, no BF16 y | Today's combine (1.27 µs × 50.2 = 64 µs) with the inject | Same | `forward.cpp:481`, `:569` | The non-inject form, which has no caller left |
| `linear_swiglu`, Q8 profile [1280, 2560] → [640] (`q8_linear_swiglu_plan.cpp:68-80`, `linear_swiglu.h`) | silu(gate)·up with gate and up unrounded (upstream rounds both) | K1 with `Pair<640>`, R = 2: lane-local | `Q8SwiGluPairedRows<640>` | `forward.cpp:466-476`, `:555-565` | `split_kernel` (grid 5, median 1.50 µs: 75-156 µs) and `silu_and_mul_kernel` (~60 µs): ≈ −0.12 to −0.14 ms |

**Totals.** Per layer 10 fewer launches (3 + 3 mixer, 2 inject, split, SiLU): ~480 per forward of
~1,800 per round; with ~80 µs of removed graph edges, −0.55 to −0.7 ms per forward. The drafter uses
the same Ops. `hyper_connection_norm`, `_gates` and `_collapse` become private kernels.

#### Phase 2: bit-exact latency work

| Item | Mechanism | Estimate per forward |
|---|---|---|
| PDL weights-first | K1, K1a, K1b, the fused Ops and the router launch with `pdl::launch_with(Dependency::Programmatic, …)`: weights, wait, `trigger_dependents()` after the streaming loop (`enter_streaming`) | −0.3 to −0.6 µs on each of ~200-290 small calls: −0.06 to −0.15 ms, measured per call site (op development §4.3: an entry-time trigger measured about 10 % slower on the RTX 5090 DFlash2 round) |
| L2 warming while the CPU is the long pole (§8.3 rule 3) | `cpu_wait_kernel` runs after the fork join (`moe_layer.cu:747-755`) with the GPU otherwise idle: 18.56 µs × 48 = 891 µs per forward (median 6.94, p90 49.6). It gains an optional `L2Warm{const void* ptr[4]; std::size_t bytes[4];}`: the next layer's attn-mixer W_down and W_up (7.0 MB) and the first ≤ 16 MB of its input projection. Concurrency S3 lands first (§19.3.0) and makes the wait `min(8, kMaxCpuJobs)` CTAs that all poll and copy jobs. Phase 2 adds P CTAs to that grid, which issue `cp.async.bulk.prefetch.L2.global` for 1/P of the spans and exit (one SM's prefetch rate is unknown), so one `cpu_wait` grid serves copying and L2 warming. No extra launch, no semantic effect | −0.1 to −0.3 ms (unmeasured); only waits ≥ ~4 µs gain; 23 MB stays under half of the 96 MB L2 |
| L2 class per weight | New Core `Weight` field `l2` ∈ {`Stream`, `Reuse`} set at Qwen4Exp binding: main-forward dense weights `Stream` (K1 adds an evict-first `createpolicy` hint), drafter dense weights `Reuse` (default policy). Narrows §8.3 rule 5. Drafter evict-last is a measured variant, kept only if M4 shows a gain | Protects the measured drafter L2 reuse |
| Shared expert on its own branch (2026-10-04, Strata `cfd3b72`) | The shared expert depends only on the MLP mixer output, but runs on the main stream after the resident expert pass (`forward.cpp:458-477`), so it overlaps staging and CPU misses but not the routing chain (router, route, dispatch, `cpu_plan`; a 10.5 µs mean route → count gap) or the resident experts. Fork it at the mixer output onto a Program-owned stream (not the miss-staging fork stream, where it would queue behind `stage_kernel`), join before `moe_experts_cpu_wait`; two events per layer, graph-captured | −0.1 to −0.25 ms at T = 1 in GPU-bound layers (estimate); less in miss-heavy MTP rounds and after 1b's two-kernel shared chain (~8-9 µs). Kept only if M4 shows it beyond noise |

*Phase 2 as implemented so far* (2026-10-05):

- **L2 warming in the CPU wait** (branch `claude/fn-kernels-l2`, over concurrency S3): `MoeL2Warm`
  (four spans) in `MoeExpertSource`; when a call published a CPU request, `cpu_wait_kernel` runs 8
  more CTAs that issue `cp.async.bulk.prefetch.L2.global` in 16 KiB pieces over their share of the
  spans, then exit. Forward warms the next layer's `attn_hc` down and up and the first 16 MiB of its
  mixer input projection. Measured (ABBA in one build with a temporary toggle; outputs identical):
  tg512 plain 100.65/100.98 → 102.52/102.54 tok/s (+1.7 %), tg512 MTP 152.93/153.02 → 153.80/153.87
  (+0.6 %), cold CLI code plain 74.3/74.5 → 75.5/75.4 (+1.4 %), MTP 89.7/89.8 → 89.7/90.2 (+0.2 %).
- **Shared expert on its own branch: rejected.** Built (Program-owned stream, fork at the MoE input,
  join before the CPU wait, the shared ops on a sub-arena carved before the fork), ids identical,
  but slower: tg512 MTP 135.5/134.6 → 132.2/132.8 tok/s (−2.0 %; the code and tg512-plain arms of
  that run overlapped builds, a rerun is in the k2pdl rig). The shared chain competes with the
  routing chain and the resident experts for SMs.
- **PDL for K1** (branch `claude/fn-kernels-2`): weights first, `pdl::wait_for_dependencies`
  before x or the output, `trigger_dependents` after the passes, `pdl::launch_with` (programmatic
  only in captured graphs); a captured producer → K1 case in the Q8 linear test. Measurement pending.
- **PDL for the fused mixer, and its per-column chain** (2026-10-10, landed): both kernels launch
  through `pdl::launch_consumer` and call `pdl::enter()` after their weight loads (K1a also loads
  its norm weights first). K1a issues every column's R loads before using any, keeps each column's
  per-thread sum of squares in registers and reduces all columns behind one barrier (instead of a
  block sum with two barriers per column), and scales Rn with preloaded norm weights. K1b loads its
  norm weights and R before the barrier, and in both kernels lanes 4-7 of each quarter warp read a
  chunk's two float4 halves in the other order (no 2-way bank conflict). Products and their order
  are unchanged: outputs are bit-identical. `infernix_hyper_connection_bench` (new: a CUDA graph of
  48 calls with distinct weights, per-call time), Q8, HEAD → new: T = 1 11.08 → 10.56 µs, T = 2
  13.66 → 12.35, T = 5 21.53 → 17.88, T = 8 29.42 → 21.92 (BF16: 13.8 → 13.1, 24.8 → 20.9 at
  T = 5). PDL alone measured 10.34 µs at T = 1 but 20.61 at T = 5; the chain rework alone 11.11
  and 18.63. End to end (Dense8, INT8 KV, pg1024+256, 4 old/new pairs × 3 reps): MTP decode
  146.5 → 147.5 tok/s (+0.70 %, pairs +0.57..+0.80 %, ranges disjoint); plain 123.3 → 123.5
  (+0.10 %, pairs −0.06..+0.29 %: not resolved, the T = 1 op gain does not carry over).
- Not built yet: the `Weight` L2 class.

#### Phase 3: tensor-core K2 (conditional)

`mma.m16n8k16` BF16 → FP32 per 32-code group with exact int8 → BF16 decode (`q8_bf16_pair_from_s8`),
scale per group after the MMA: ~3 lane-ops per weight, flat in T, not bit-exact with K1. To keep the
per-column invariance required above, K2 serves a shape at every T from 1 to 16 or not at all, never
at T = 5-8 beside K1 at T ≤ 4. Adopting it is a rounding change gated like Phase 1b (§19.3.0).
Trigger, after Phase 1: T = 5 Q8 time per round > T = 4 by 10 %, T = 16 > T = 8 by 25 %, or big
shapes at T = 16 compute-bound (> 1.15 × the fit). HC down then needs a cross-CTA K split, as in
K1a, or a cluster.

#### Rejected

| Item | Reason |
|---|---|
| 16-code skinny mapping | Changes the text; K1 reaches the same bandwidth exactly |
| One row per CTA for long K | x re-read per CTA (HEAD's few-row HC down route) |
| Deeper `cp.async` stage counts | §19.2 records no gain |
| Cross-CTA split-K inside `ops::linear` | The A16 overload promises zero workspace; split-K lives in the mixer Op, which owns workspace |
| Deferred `inject_norm` (earlier draft) | The residual taps, `residual_out` (set on every MTP decode and verification call), the `mtp_chunk` copies and the MTP `residual_out` read the residual between blocks, and `ym`/`inject` live in a `work_.scope()` that rolls back (`forward.cpp:116`) |
| Persistent 340-CTA [16384, 2560] grid as default | 1024 row groups / 340 = 3.01 iterations: 4 CTAs run a 4th (~1-1.5 µs tail × 36); HEAD's 2048 on 510 slots has the same flaw. Balanced sweep candidate only |
| Two 8-column launches for T = 9-16 (§19.2) | Reads the weights twice; two passes in one launch over register-resident weights read them once |
| Evict-first on every K1 weight | Removes the drafter's L2 reuse |
| Producer-epilogue stream sums for K1a (§8.3 as written) | Hidden cross-Op output for ≤ 0.3 µs |

**Shared-expert criticality.** The shared expert runs while the host computes CPU-served misses;
when the CPU is the long pole, gate/up and down savings partly become a longer `cpu_wait`, which
Phase 2's warming turns into progress. Per-round kernel time and wall clock are reported together.

#### Tests and measurements

| Test | Cases |
|---|---|
| `tests/ops/linear/test_q8_a16.cpp` | Add [2560, 2560] and [13952, 2560] to `kGeometries` (keep [2560, 4096] only if an artifact uses it). FP64-oracle conformance at every seam: T = 1, 4, 5, 8, 9, 15, 16, 17; 24, 32, 33, 64, 65 where slices are selected; an interior value. `q8_a16_column_invariance` over every Flash-Next shape at T = 2..16 (≤ 64 where slices are selected), with the policy overload. A graph-captured producer → K1 case (pattern `linear_add_test_common.cpp:371-388`): eager launches are never programmatic (`pdl.cuh:40-64`) |
| `tests/ops/linear_swiglu/test_q8_a16.cpp` | [1280, 2560] at T = 1..16, 17, 33, 512 against FP64; invariance T = 1..16; one graph-captured case |
| New `tests/ops/test_hyper_connection.cpp` | `hyper_connection_mix`: FP64 closed formula from R at Q8 and BF16 weights, T ∈ {1, 4, 5, 8, 9, 16, 17, 64}, with and without inject, invariance T = 1..16, graph-captured. `hyper_connection_linear_inject`: FP64 with untouched rows, `y_tap` on and off, aliasing rejection |
| MoE and Phase 0 (`tests/ops/test_offloaded_moe_layer.cu`, `tests/ops/test_rmsnorm.cpp`) | `moe_combine` inject form against FP64. `moe_dispatch` against an exact oracle (counts, offsets, jobs, job_count, route log exact; entries per expert as a multiset) at entries 1, 10, 1024 (small kernel) and 1025, 40960 (multi-CTA path), plus a graph replay that restarts counts from zero. RMSNorm d = 10240 at rows 1..5 against FP64; `projection_fp32` [513, 2560] at T = 1..8 if absent |
| Development only, then deleted | Each K1 instance bitwise equal to HEAD's SIMT route, 10 shapes × T = 1..8 (evidence, not qualification); preloaded RMSNorm equal to generic. `compute-sanitizer` memcheck and initcheck on K1 (one small, one big shape) at T = 1, 5, 7, 9, 13 (partial passes) and K1a/K1b at T = 1, 7, 13 |
| Model | `test_forward_real` (real artifact, FP64 reference, `--residuals --blocks`) before and after 1a-ii and before and after 1b (§19.3.0 applies the rounding-change gates to both): residual-tap and logit error must not grow, `--dump-logits` perplexity must not rise beyond noise. Greedy MTP ids equal plain at C = 1 after every phase |

M0-M4 are local to this section, not the §19 milestones; each run follows §19.3.0's execution rules
(`local/gpu.lock`, plus `fn\locks\quiet.lock` for timing runs) with a dry-run-validated rig.

| Id | Scope | Method and pass criterion |
|---|---|---|
| M0 | Phase 0 | Greedy ids identical on tg512 and the code and story CLI, plain and MTP; tg512 plain and MTP max 4 `--lm-head-draft`, ABBA; nsys route → count gap. 2026-10-04 items: `cuobjdump -sass` shows no local memory in the `gate_up_kernel`/`down_kernel` instances and `attention_kernel`; the offloaded-MoE layer-route and Op tests bit-exact; the GDN control instance bitwise equal to today's at T = 1..8 (development check); the FP64 QSA oracle test per KV profile; ncu load/store throughput of `gate_up`/`down` and nsys per-kernel times (gate_up 39.2, down 16.4, control 3.21, attention 11.5 µs at T = 1) before and after |
| M1 | Op sweep (temporary executable, op development §7) | All K1 candidates, HEAD SIMT, MMA and `Q8SlicedKDefault` × 10 shapes × T ∈ {1..8, 9, 12, 16, 20, 24, 32, 48, 64}; ⌈256 MiB / bytes⌉ + 1 weight copies cycled by a 200-call CUDA graph (weights from DRAM, x in L2), 3 warm-up + 10 timed replays; median, min/max, GB/s, % of 1,674.5. HEAD MMA at T = 9-64 first: every T > 8 claim stands or falls with it. **Go/no-go at T = 1:** HC up 3.72, shared down 2.76, [16384, 2560] 28.3, PLE 22.7 µs (skinny + 0.2), others fit + 0.3; a shape that misses gets one ncu look before it is registered. HC down > 5 µs at T = 8: one `lts__t_bytes` ncu run. 1b chain graphs: unfused, K1a/K1b, three-kernel; T = 1, 4, 5, 8, 16; Programmatic against Serialized |
| M2 | Phase 1a end to end | 1a-i: greedy ids **and** MTP acceptance identical to HEAD (tg512, code, story; plain and MTP); any difference is a defect. 1a-ii: C = 1 ids identical, acceptance reported, forward_real error and perplexity no worse; C = 2 serve with the speculation gate off (§19.3.5 S1's local environment override), with `--draft-tokens 4` and with `--ngram-draft-tokens 7`, plus C = 3 with the gate on (W = 1 catch-up hidden projection of 12 columns): tok/s per request and an nsys of verification rounds. Speed: tg512 plain and tg512 MTP max 4 `--lm-head-draft`, cold CLI code and story, serve warm, ABBA. nsys per-round Q8 time and per-shape times at W = 1, 4, 5; drafter timings ([13952, 2560] second call ~16.6 µs, HC up ~4.3) to catch lost L2 reuse |
| M3 | Phase 1b | M2 checks plus forward_real error and perplexity; speed from per-round nsys kernel time; end to end on prompts whose greedy ids stay identical (check code, story, the 7,448-token prompt; report which), tg512 and its hit rate beside them (§19.2 measurement rule) |
| M4 | Phase 2 | Greedy ids identical. PDL per call site (Programmatic against Serialized, then nsys per round); warming on/off with HC down/up times after a wait; `Stream`/`Reuse` against default and drafter evict-last, with drafter timings; the shared-expert branch on/off (nsys overlap with routing and resident experts, tg512 plain and MTP ABBA) |

#### Steps, risks and decisions

| Step | Content | Depends on | Agent-days |
|---|---|---|---:|
| 0 | Phase 0 (with the 2026-10-04 E2M1, GDN control and QSA attention items), tests, M0 | — | 2.5-3 |
| 1a-i | K1 MaxCols 4/8, 10 registrations, few-row removal, tests, M1 (T ≤ 8), M2 | — | 2.5-3 |
| 1a-ii | MaxCols 16 two-pass instances, T = 17-64 slices, invariance to 16/64, M1 (T > 8), M2 at C = 2 | 1a-i | 1-1.5 |
| 1b | `hyper_connection_mix`, `hyper_connection_linear_inject`, `moe_combine` inject form, `linear_swiglu` profile, model composition and workspace sizing (`forward.h`), superseded public Ops removed, tests, M3 | 1a-ii | 3.5-4.5 |
| 2 | PDL, `cpu_wait` warming, `Weight::l2`, the shared-expert branch, graph-captured tests, M4 | 1a (and 1b); concurrency S3-S5, whose `cpu_wait` grid Phase 2 rebases onto (§19.3.0) | 2-3 |
| 3 | K2, only on its trigger | 1-2 | 2-3 |

Step 0 is independent; 1a-i alone is a complete bit-exact deliverable. In its branch each step
records its results and decisions (the K1a statistic, K1a's four K slices, inject placement, the
rule-6 deviation, rejected items) in this §19.3.3, and in its own subsection of
`docs/qwen3_8-flash-next.md` if C ≥ 2 guidance changes. The §8.3 deviation notes, §19.2 results
and the §19.2 "Next" list change on the base branch at merge time (§19.3.0).

Other files: `q8_schedule.cuh` (`Q8A16StreamSchedule`), `q8_instances.cuh` (`SimtR1T*W8`
removed), `q8_launch.h`, `q8_shapes.h`, `q8_dispatch.cpp`, `q8_simt.cu` (the r1_w8 launchers go)
and `sources.cmake` in `src/ops/linear/q8/`; `include/infernix/ops/linear.h`, `linear_swiglu.h`,
`hyper_connection.h` (norm/gates/collapse leave the contract) and `offloaded_sparse_moe.h`
(`moe_combine` inject form, `moe_dispatch` route-log output, `cpu_wait` `L2Warm`);
`src/ops/hyper_connection/hyper_connection.cu`, new `hyper_connection_mix_q8.cu` and
`hyper_connection_linear_inject_q8.cu`, and its `sources.cmake`;
`src/ops/linear_swiglu/q8/q8_linear_swiglu_plan.cpp` and its decode kernel file;
`src/ops/offloaded_sparse_moe/cuda/moe_layer.cu`; `src/ops/projection_fp32/projection_fp32.cu`;
`src/ops/kernel/rmsnorm.cuh` and `src/ops/launcher/rmsnorm.cu`; `src/core/weight.h` and the
Qwen4Exp binding that sets `l2`; `src/models/qwen4_exp/execution/forward.cpp` and `forward.h`
(`workspace_bytes`, `mtp_workspace_bytes`); `tests/ops/test_offloaded_moe_layer.cu`,
`tests/ops/test_rmsnorm.cpp` and `tests/ops/tests.cmake`. The 2026-10-04 items add
`src/ops/common/canonical_math.h` (`e2m1_x2`), `src/ops/linear/bf16/bf16_general.cu` (the small-N
instance), `src/ops/qsa/qsa.cu` (`attention_kernel`), and a Program-owned stream with two events
per layer for the shared-expert branch (`program/program_impl.h`, `execution/forward.cpp`).

**Risks.** One measured MMA point carries the T > 8 claims, and the traces predate the few-row route
and cover T ≤ 4 (T = 5-8 baselines are microbenchmarks); the 1.4 µs fixed cost is fitted without
PDL. HC down at T ≥ 6 in 1a rests on unmeasured L2 → SM bandwidth until 1b, and K1b's redundant
partial reads may be exposed. K1a's 80 KB per pass forces 1 CTA per SM; two-pass MaxCols 16
instances may spill or lose occupancy (checked per instance); ~30 K1 instances plus the fused ones
cost build time and binary size. Bit-exactness at T ≤ 8 depends on mapping, FMA order, `w = q·s` and
butterfly pairing: an M2 mismatch is a defect, not noise. 1a-ii moves catch-up rounding at C = 1
(acceptance) and verification rounding at C ≥ 2; 1b moves the tg512 text, hit-rate and acceptance
baselines. Warming after a short wait competes with the next kernels, and a mis-sized set could
evict activations; PDL early launches can slow neighbours (DFlash2). Realization is typically
50-80 % of kernel deltas.

Each decision below is **Decided (default, 2026-10-04)**; the user may override it.

| Question | Decided (default, 2026-10-04) | Alternative |
|---|---|---|
| Phase 1b rounding changes | Accepted: Rn, z, m, u in FP32, injects from FP32 products and sums, gate/up unrounded; includes the inject folding (a rounding change under op development §6.1, though a review thought it bit-exact) | Stop at Phases 0, 1a and 2 (bit-exact except 1a-ii's T = 9-64 rounding change) |
| Mixer structure | §8.3 two-kernel split-K mixer with K1a's in-kernel norm statistic (−3.5 to −4.3 µs per mixer, 4× less x traffic) | Three-kernel K1-based mixer (norm; down + gates; up + collapse), or §8.3's producer-epilogue statistic |
| T = 17-64 | Per-shape M1 choice between MMA tiles and 16-column K1 slices (C ≥ 3; the C = 1 max-4 catch-up at 20 columns) | MMA tiles above 16 columns |
| Recipe A (BF16 dense) | Served by the fused Ops' composed route, no new kernels, today's speed | Fused BF16 kernels |
| Full C = 2 invariance | No separate task: the speculation gate (§19.3.5) keeps C ≤ 8 main forwards at T ≤ 8, where dense Q8 is column-invariant | Also make BF16 dense, QSA verification attention and the GDN record width-invariant (§11.3 option) |
| Phase 0 | Included here | Tracked separately |
| PDL scope | K1, the fused Ops and the router | Every Flash-Next decode kernel |
| `Weight::l2` | Added in Phase 2; drafter evict-last tried as a measured variant | No Core change; one policy |
| Clusters (DSMEM) for HC down | Allowed if Phase 3 triggers | Cross-CTA K split only |

### 19.3.4 N-gram row reads overlapped with the GPU

Design, adversarially reviewed, nothing implemented (worktree `NInfer-V3-flashnext`, HEAD
`46a56fc8f`, 2026-10-04). It drops the earlier draft's dedicated n-gram thread, "warm" prefetch
jobs and filler-row reuse, and no longer gates plain rounds by default (parked as S4b). Bare line
numbers are Qwen4Exp `program.cpp` at `46a56fc8f`. Since `54deaa2bc` that code is in
`program/program_impl.h`, where every cited line sits two lines earlier (sync 1180 is
`program_impl.h:1178`). **Simulated** means CPU only, real tokenizer, exact row-id specification,
no weights or GPU (`ngram-overlap-sim.py`, `ngram-warm-repeat-sim.py` in `local/workdirs/fn/plans`).

#### Premise (corrects §19.2)

- **§19.2's after-draft gap was measured on n-gram-warm text.** The ~0.56 ms gap and the "reused
  n-gram read events" A/B are warm tg512, whose repetitions all generate the same tokens and drafts
  (1-token seed, temperature 0, a priming run and one warm-up repetition first, policy state reset
  at admission 449-450, exact expert arithmetic), and the row cache, a Program member (158, 1549),
  outlives requests. The serve warm figures (147.2 / 97.5 tok/s) repeat one request: also warm.
- **Warm repeats leave 0.3-0.7 reads per MTP round**, all direct-mapped conflict misses
  (simulated: a 512-token repetition at ~3.5 tokens per W = 5 round, rejected drafts as new
  n-grams, replayed four times through the 2^20-row cache). Prose / C++ / story: **0.67** / 0.55 /
  0.33 per round, or 98 / 81 / 49 per repetition against 10,962 / 7,761 / 6,775 on the first pass
  (prose: ~75 per round).
- **So the warm gap is host turnaround, not NVMe.** Between sync 1459 and the verify start: the
  spin-sync wake, the `mtp_drafts_` copy and proposals, staging with an O(context) history copy into
  `sequence_` (1127), hashing and probes, two `upload_pinned` launches, `before_round`
  (`expert_residency.cpp:72-93`), graph launch, route download, five tail operations, and WDDM
  submission, only at sync 1180 (this path never calls `device_.flush()`). The warm reads add ~10-50
  µs per round on average at an unmeasured QD-1 latency (estimated 20-80 µs each). nsys inflates
  host gaps here (Qwen3.5: 0.80 ms under nsys, ~0.30 ms in logs), so the real gap may be ~0.2-0.4
  ms.
- **First-time text reads rows** (new serve request, agentic turn, cold CLI; simulated, cold cache).
  Decode misses are compulsory, so a larger or associative cache does not help decode (an infinite
  cache gains 0-2.5 pp). Capacity helps only re-prefill (30K prose re-read: 85.4 / 92.2 / 96.0 % at
  2^20 / 2^21 / 2^22), which the prefix cache (§19.3.1) addresses.

  | Stream | Tokens | Hit rate, 2^20 direct-mapped | Hit rate, infinite cache | Misses per token (of 16) |
  |---|---:|---:|---:|---:|
  | C++ (`program.cpp`) | 24,879 | 63.6 % | 64.2 % | 5.8 |
  | Python (`reference.py`) | 6,605 | 43.3 % | 43.5 % | 9.1 |
  | Prose (design doc) | 72,207 | 44.4 % | 46.9 % | 8.9 |
  | Generated code | 295 | 30.7 % | 30.7 % | 11.1 |
  | Generated story | 201 | 1.5 % | 1.5 % | 15.8 |
  | All, one cache | 104,192 | 49.7 % (2^22: 51.3 %) | 51.9 % (LRU at 2^20 equal) | 8.0 |

- **Per-round reads on first-time text** (estimated, one issuer at the §19.2 probe's ~4.3-4.8 µs
  per read). Reads add **~0.2-0.37 ms per MTP round** on top of the turnaround. The probe had one
  issuing thread, so "device-bound" is unproven: ~210K IOPS also fits a per-thread I/O-stack cost.

  | Round | Rows | Reads (miss share m) | Single-issuer time |
  |---|---:|---:|---:|
  | Plain, C = 1 | 16 | 6-16 | 25-87 µs |
  | MTP W = 4, prose-like (m ≈ 0.9), C = 1 | 64 | ~58 | ~270 µs |
  | MTP W = 5, code-like (m ≈ 0.6), C = 1 | 80 | ~48 | ~220 µs |
  | MTP W = 5, generated prose (simulated first pass), C = 1 | 80 | ~72-75 | ~340 µs |
  | MTP W = 4, C = 2 | 128 | ~115 | ~520 µs |

- **Hiding windows** (estimated). Rows are consumed once, by `ple_embed` at the top of layer 1
  (`forward.cpp:117, 245`; one PLE layer), after the embedding, HC expand and all of layer 0, a
  most-missed layer. Block 0 of a W = 4 / 5 round is ~0.50 / 0.59 ms (2.13 / 2.54 plain rounds of
  ~11.2 ms over 48 layers); a draft step is ~0.2-0.3 ms (§12.1's ≈ 84-134 µs layer-0 window is the
  §8.3 plain-decode budget at T = 1, not this measured verify-round average; S0's events decide). At
  C = 1 the ≤ 64 draft-column rows (≤ ~300 µs cold) fit in block 0 with ~0.2 ms margin, and the
  anchor column's rows fit in the draft.

Two levers follow: hide the reads (first-time text), and remove the host round trip between
drafting and verification (all text, including the headline benchmarks).

#### Plan

| Step | What | Helps | Condition | Effort |
|---|---|---|---|---|
| **S0** | Counters, host-phase timers, event-timed GPU idle, n-gram-cold toggle; flush probe (S0b) | attribution; maybe the warm gap | always first | 0.5 day + ~1 h GPU |
| **S1** | One read per distinct 4 KiB block; 256 KiB, 64-deep rolling ring; no per-call events. S1b: multi-issuer probe | cold prefill (−29 to −50 % reads per chunk), memory | always | 1 day + ~0.5 h GPU; S1b 2 h |
| **S2** | Gate kernel before `ple_embed` in verify graphs; engine-thread reads after the launch; anchor reads during drafting; no new thread | n-gram-cold MTP | always | 1.5 days + ~1.5 h GPU |
| **S3** | Device-assembled verification: no host sync between drafting and verification | every MTP workload | W1 after-draft GPU idle ≥ 0.15 ms per round, measured with S0's events after S0b and S2 | 2 days + ~1.5 h GPU |
| **S4** | Mailbox, plain gate, multi-issuer reads (parked); S4d cross-chunk prefill reads (scheduled since 2026-10-04) | C ≥ 2, plain, prefill | S4a-c each on its trigger; S4d after S1 | 0.5-1 day each; S4d 1-1.5 days |
| Docs | §12.3 "As built" and §12.4 updated as each step lands (§5, §7, §8.1, §11.2, §12.3 and §12.4 already state this design, and since 2026-10-04 so do §3.4, §8.9, §13, §14, §15.1 and §16.4); at merge, the M6 row's "device-driven PLE" and §19.1's "device kernel (§12.3)" are corrected; §19.2 results, corrections and "Next" list; the diagnostic line in `docs/qwen3_8-flash-next.md`; `tests/README.md` | one authority | as each step lands | 0.5 day |

Total ~5.5 days and ~4-5 h of GPU rig time, plus ~1.5-2 days and ~1 h for the 2026-10-04
additions (S4d, the S0 draft-probability log, the third flush site). No numerical change: the
same bytes reach `ple_embed`, the same ids reach the verify graph, and the filler rule is
unchanged.

**S0: measure first** (`program/ngram_volume.{h,cpp}`, `program.cpp`). Permanent: `NgramVolume`
counters (rows requested, cache hits, NVMe block reads, read ns; today's `hits_`/`misses_` are never
reported) and a per-request line beside `report_cache` (826-851): "n-gram rows: N requested,
H % host-cache hits, R NVMe reads, T ms of reads". S2 extends it with the reads behind the gate and
the gate's waits when the gate exists; no field is printed before its mechanism. Temporary (never committed; kept as a local patch
re-applied to each step's measurement build through S3, because the S2 and S3 acceptance rows use
the events, phase timers and cold toggle): `Clock::now()` phases per round (MTP: sync-1459
return → staged → launched → tail enqueued → sync-1180 return; plain: `sample()` return → next
launch; commit: enter → GPU enqueues done → after `settle_round`); events after the draft graph and
before `verify()`'s first upload (and `sample()` → next plain launch) for GPU idle without nsys; a
cold toggle (`NgramVolume::invalidate()` at each admission, so tg512 reads like first-time text);
**S0b**, `device_.flush()` after `replay()` in `verify()` (1148-1152) and `run_decode()` (993),
and, since 2026-10-04, in `commit_verified` before `settle_round`, where the fold, tails, PLE commit
and catch-up are enqueued and ~0.5 ms of host work follows with no flush. Each site gets one flush,
never a poll: Strata kicks all three sites (`cfd3b72`) and measured ~−3 % when it polled the final
wait (`0e23f57`). Also temporary since 2026-10-04: an online-softmax top-1 probability in the
drafter's argmax over the proposal rows, logged per draft as (step, p_j, accepted), with the target
p and the drafter's top-16 q for sampled rows. An offline replay of that log decides the
confidence-width and sampled-draft options (§19.3.6) and costs ~0.5 day.

**S1: volume reads.** Today (`ngram_volume.cpp:47-91`) misses are per occurrence: a cold 4,096-token
chunk makes 65,536 reads for 32,598 (C++) or 46,138 (prose) distinct blocks; the bounce buffer grows
to (misses + 1) × 4 KiB and never shrinks (~268 MB); `read_direct_batch`
(`read_only_file_win32.cpp:125-180`) drains each group of 64 overlapped reads before the next and
creates Win32 events per call. Core `ReadOnlyFile::read_direct_blocks(offsets, block_bytes, ring,
consume)` replaces it (only user `NgramVolume`): at most `ring.size() / block_bytes` unbuffered
reads in flight; `consume(i, bytes)` runs on the caller in completion order, then the block is
reused; returns when all are done; a failed or short read stops issuing, drains and throws
`system_error` / `runtime_error`. Windows: one `OVERLAPPED` per slot, `hEvent = nullptr`,
`HasOverlappedIoCompleted` polled with `_mm_pause()` (no syscall), `GetOverlappedResult(...,
FALSE)`, reissue; no deadline, as today. POSIX (WSL): serial `pread`. `read_rows` keeps its
signature: copy hits; sort misses as one u64 key (block < 2^24, index < 2^20); read each distinct
block into `ring_` (64 × 4 KiB = **256 KiB**, allocated once), whose `consume` copies every row it
serves and fills the cache. Keys reach ≤ 0.5-1 MB per 4K chunk; dedupe saves 50 % / 29 % of a cold
chunk's reads (~148 / 87 ms). **S1b** (standalone, CPU and NVMe only, while no engine reads G:):
random 4 KiB batches of 64, 256 and 4,096 with 1, 2 and 4 issuers, 64-deep rings; ≥ 1.6× one
issuer → S4c. Vision V6 (§19.3.2) relies on this dedupe for image chunks (~16,000 reads to ~48) and
adds no second version.

#### S2 and S3: gated, then device-assembled, verification

- **Core gate** (`src/core/device.{h,cu}`). `upload_pinned_when(dst, pinned_src, bytes,
  pinned_ready, device_expected, wait, stream)` copies `bytes` (multiple of 16) once `*pinned_ready`
  equals a device word staged earlier in stream order; optional `wait` (device U64[2]) sums CTA 0's
  wait ns and count. `publish_pinned_word(ready, value)` is `_mm_sfence()` (orders non-temporal
  stores memcpy may use; an x86 release fence emits nothing), then a volatile store.
  `min(16, ceil(bytes / 32 KiB))` CTAs of 256 threads; each CTA's thread 0 polls with
  `__nanosleep(200)` (no co-residency needed; ≤ 16 pollers instead of §8.2's one designated
  poller, and a trap at 120 s instead of §8.2's ~2 s bounded spin with `agent_error`: both
  deliberate exceptions to §8.2), then `__threadfence_system()`, `__syncthreads()`, then `__ldcv`
  loads, which never use a stale cache line from an earlier round (eight 16-byte vectors per thread
  per batch), as in the CPU miss handshake (`moe_layer.cu:575-640`). Trap after
  `kPinnedGateTimeoutNs` = 120 s. `ple_embed` is a plain launch without PDL (`ple.cu:181`), so
  stream order holds it behind the copy. 12,800 B at C = 1, W = 5 (one CTA) to 327,680 B (8 lanes ×
  W = 16, `ngram_draft_tokens ≤ 15`; 10 CTAs); ~3-4 µs per round, net ~+2-3 µs (~0.01 %).
- **Forward hook.** `ForwardBatch::ngram_gate` (`NgramRowGate`: `pinned_rows`, `pinned_ready`,
  device `expected`, `wait`); `Forward::ple` (`forward.cpp:240-245`) runs the gate into `ngram_rows`
  before `ple_embed`; `forward_call` (1019-1041) sets it for verification only.
- **Program state.** `gate_host_` (`PinnedHostBuffer(64)`, zeroed and published at allocation, as
  `cudaMallocHost` does not zero); `io_layout_.gate`, a U32 between `columns` and `ngram`
  (256-aligned) in the uploaded prefix (`io_device_` zero-filled once; a gated `io_prefix` ends at
  `io_layout_.ngram`, so the prefix upload shrinks by the rows); `gate_wait_` (`DeviceBuffer(16)`,
  read by a 16 B D2H at request finish); `gate_sequence_` (u32, +1 per gated round, skips 0;
  strictly increasing, so a stale word never matches; wraps after ~1.3 years at 100 rounds/s).
- **S3 Op** `speculative_assemble_verify_tokens(anchors [B], step_drafts [B,S], copy_drafts [K,B],
  extents [B], from_device [B], verify_ids [W,B], drafts [K,B], stream)`, all I32, in
  `include/infernix/ops/speculative_round.h`: drafts are `step_drafts[j*B + b]` (step-major,
  1441-1453) when `from_device[b]`, else `copy_drafts[b*K + j]`; K = W - 1; one CTA per row; exact
  oracle; wrapper, launcher and kernel in `src/ops/*/speculative_round.*`. Qwen3.5's
  `speculative_prepare_verify_inputs` takes drafts by row with anchor fillers, so it does not fit.

Round protocol, MTP or copy-proposal verification with W > 1 (W = 1 keeps the plain path):

| Phase | S2 | S3 (changes to S2) |
|---|---|---|
| Plan | Reorder `decode()`, a pure refactor: `wanted`, `steps = max(wanted)`, extents (`min(limit, wanted[b])` or a longer copy proposal; they read only `history`), sources (MTP, copy, none), copy tokens and W before `mtp_draft` (568-589 above 566) | same |
| Begin | g = ++`gate_sequence_` into `host_io() + io_layout_.gate`; arm the scope guard | same |
| Stage | as today, without the `sequence_` copy and `stage_ngram` | every pinned byte once per round, before the first upload that reads it: io (anchor and copy-row ids, positions, columns, slots, rows, gate word), spec (extents, lengths, anchors, new `from_device` and `copy_drafts`), mtp (ids, cells) |
| Draft | draft graph, drafts D2H (1457), `device_.flush()`; read column 0 (h[p-2], h[p-1], h[p]: committed tokens) of every row and all columns of copy rows into landing column `b*W + j`; sync 1459. With `steps == 0` these reads move to Reads | today's `mtp_draft` up to the draft graph (saved-column D2D, uploads, kv-only pending cells when unwritten), then D2H to `mtp_host_`, `cudaEventRecord(drafts_ready_)` before any verify work, flush; set `mtp_written` now (it describes enqueued work); no sync |
| Verify | `upload_pinned` of io up to `io_layout_.ngram`, spec, `before_round`, `replay` (gate inside), route download, tail, D2H, flush | the verify body starts with the assemble op (`mtp_device(drafts)` and spec into io ids and spec drafts) |
| Reads | MTP columns 1..W-1: (h[p-1], h[p], d0), (h[p], d0, d1), then (d[j-3], d[j-2], d[j-1]); fillers d.back(), the anchor when n = 0 (1129); 16 rows, 2,560 B per column | anchor and copy-row columns; if `steps > 0`, `cudaEventSynchronize(drafts_ready_)` (spin), then MTP columns from `mtp_host_` |
| Finish | `publish_pinned_word(gate_host_, g)`; sync 1180 and the rest as today | n is the staged extent; `pending_tokens_` comes from `licensed` |
| Exception | scope guard: zero-fill unwritten columns, publish g so an enqueued gate passes, rethrow unchanged | same |
| Why safe | the engine thread reads where it already spins idle (`cudaDeviceScheduleSpin`, `device.cu:28-29`); every call that writes or uploads the landing area ends in a stream sync (509, 636, 1180, 1459, 1540); commit, which leaves its spec `commit` and mtp `up_ids` uploads pending, never touches it. So a gate never reads the landing area across calls, and prefill and plain decode still write rows before their own upload | `drafts_ready_` precedes the gate in stream order (no deadlock); io `slots`/`rows` are uploaded twice but staged once; the previous commit's pending spec `commit` and mtp `up_ids` uploads cover regions this round does not write; `host_configs_` and `table_host_` keep one writer; the rule is documented in `verify()`. Copy-only mode (no drafter) omits the assemble op and stages as S2 |

**S3 timeline at C = 1** (estimated). Draft graph ~0.8-1.2 ms; D2H and event ~3-8 µs; 2-3 uploads
~3-5 µs; assemble ~2 µs; embedding; block 0 ~0.5 ms; gate ~3 µs (waits only if rows are late);
layers 1-47; acceptance. Anchor reads end during drafting; draft-column reads (≤ 64 rows: ≤ ~300 µs
cold, ~1-5 µs warm) start ~5 µs after it and end inside block 0. New nodes cost ~10-15 µs per round.
`before_round` runs ~1 ms earlier in GPU time, so a promotion landing during drafting is published
a round later. A D2H bubble > 15 µs (events) would become an SM-write kernel like `upload_pinned`.

**S4** (a-c parked until a trigger fires; d scheduled after S1 since 2026-10-04):

| Item | Mechanism | Trigger or bound |
|---|---|---|
| a. Mailbox | Core `publish_pinned` kernel after each draft step writes its tokens and a sequence word (`moe_layer.cu:575-585` pattern); draft columns read during drafting; ~2-3 µs per step | S3 mean gate wait > 10 µs per round at C = 1 on cold text, or any C ≥ 2 use (§19.3.5) |
| b. Plain-round gate | plain graphs gated, reads after launch; ~3 µs per round plus trap exposure | ≤ 0.8 % cold, −0.03 % warm; adopt only if cold plain gains beyond noise and warm does not regress |
| c. Multi-issuer reads | fixed issuer pool in `NgramVolume` for batches ≥ 256 reads (prefill, C ≥ 4); decode stays single-issuer | S1b ≥ 1.6× |
| d. Cross-chunk prefill reads (2026-10-04; replaces the within-chunk gate) | Every prompt token is known up front. So while chunk i computes, the engine thread hashes and reads chunk i+1's rows into a second landing area. The area is uploaded in stream order before chunk i+1, after `advance_prefill`'s sync, so the GPU no longer idles during the reads (today `stage_ngram` runs after the sync, `program_impl.h:493-510, 923-936`). The first chunk's rows are still read before its launch, as today. Strata's CUDA path has done this since 0.1.13: 762 → 790 tok/s at chunk 4096, bit-identical (RTX 5070, 32K prompt). Strata's short first chunk is rejected, since each of our calls pays ~1.28 s of staging. The earlier within-chunk gate (`run()` splitting its copy around the landing area, a multi-CTA gate hiding the reads behind block 0, ~2-3 %) remains the alternative | Built after S1; 2-4 % per cold 4K chunk after S1's dedupe (estimate), more once chunks take ~2 s (§19.3.6 prefill item), ~0 on row-warm text |

#### Stalls, errors and rejected options

- **Slow reads stay a delay.** Reads block without deadline, as today; a stall now holds the GPU at
  layer 1 instead of the engine before launch, for the same round time. The 120 s trap is above
  Windows' default 60 s disk timeout, so the OS completes or fails the read first; it only turns a
  producer bug into an error instead of a hang. Reads > 100 ms in a gated round emit a Warning
  diagnostic (silent today).
- **Errors keep their class.** The guard publishes, so the stream drains, then rethrows. Row-id and
  vocabulary errors (`out_of_range`, `invalid_argument`) take the `logic_error` recover path
  (`engine_core.h:2419-2435`: fail the active requests, `release_all`, keep the worker); I/O errors
  (`system_error`, `runtime_error`) end the worker and set `failed_` (same file, 2232 and
  2436-2453), so the engine needs a restart and model reload. Both release every lane (745-753); a
  verify round changes no lane state before sync 1180 and cannot be redone (acceptance bumps
  `token_counts`), so nothing stale resumes. Cancel, abort or a terminal at commit find the round
  and its gate complete; Program destruction finds every gate complete (each gated round ended at
  sync 1180); process exit takes the context.
- **Rejected:** a dedicated n-gram thread (a ninth spinner, a queue and cross-thread errors where
  the engine thread already idles; its borrowed 50 µs sleep is ≥ 1 ms on Windows, inferred);
  speculative top-k prefetch (k × 16 extra reads on an IOPS-limited drive, nothing left to hide); a
  larger or associative decode cache (compulsory misses); filler-row reuse (≤ 24 rows per row per
  round after dedupe, C > 1 only; filler ids and rows would disagree and change routing); a host
  read deadline that zero-fills and fails the round, or converting post-launch errors to
  `runtime_error` (both turn a slow read or a recoverable error into a model reload); gating plain
  rounds by default.

#### Expected gain (estimated; S0 replaces it)

| Per MTP round at C = 1 (~22-28 ms) | n-gram-warm (tg512, serve repeat) | n-gram-cold (first-time text) |
|---|---:|---:|
| Exposed reads / host turnaround today | ~10-50 µs / ~0.15-0.4 ms | 200-370 µs / same |
| S0b flush; S1 dedupe | −0-50 µs; ~0 | same; ~0 at C = 1, prefill −87 to −148 ms per cold 4K chunk (~1.4-2.4 % of ~6.2 s) |
| S2 gate (+2-3 µs) | **~0-0.15 %** | **+0.7-1.7 %** |
| S3 assembly (+10-15 µs) | **+0.5-1.7 %** | the same, on top of S2 |
| `sequence_` copy removed (S2) | 3-50 µs per row per round at 16K-128K context | same |
| **Throughput, S2 + S3** (plain: 0-0.3 %, S0b only) | tg512 MTP ~+0.5-1.9 % (~+0.7-2.5 tok/s at 134) | ~+1.2-3.4 % |

#### Tests

| # | Test (GPU ones under the lock) | Cases and oracle |
|---|---|---|
| 1 | Core reader, CPU: new `tests/test_read_only_file.cpp` (`tests/cmake/CoreTests.cmake`) | Closed-form block pattern; 1, 63, 64, 65, 1,000 offsets in random order with repeats; rings of 1, 4, 64; each index consumed once with its bytes; a past-EOF offset throws after draining |
| 2 | `NgramVolume`, CPU: new `tests/models/qwen4_exp/test_ngram_volume.cpp` | Synthetic volume (60-byte header, row r byte k = f(r, k)), oracle f. Random rows; duplicates and two rows of one block make one read (counters); > 64 blocks wrap the ring; hit after miss; two ids on one cache slot; id ≥ rows throws `out_of_range`; size mismatch throws at open |
| 3 | Gate primitive, GPU (`tests/test_device.cpp`) | 2,560, 12,800, 327,680 B; a host producer writes poison, waits 2 ms, writes data, publishes; copy equals data, wait count > 0 and wait ≥ the delay; eager, and a graph replayed with three sequences and new data, the old word left in place (stale-word rejection). The trap is not tested (it destroys the context). Once, with the GPU otherwise idle (a TDR resets the display GPU): a 3-5 s producer delay, to confirm WDDM preemption for the 120 s trap |
| 4 | Assemble op, GPU (`tests/ops/test_speculative_round.cpp`) | B = 1-8, W = 2-16, extents 0..K, mixed `from_device`, S ≥ the largest device extent; oracle a host loop of `verify()`'s rule (1122-1135); exact |
| 5 | Engine, real artifact, GPU: new `tests/models/qwen4_exp/test_engine_ngram_gate_real.cpp`, skip 77 without `INFERNIX_QWEN4_ARTIFACT`; volume from `INFERNIX_QWEN4_NGRAM` (default `<artifact>.ngram`), as in `infernix_qwen4_exp_forward_real_test`; one Engine, MTP, C = 2 | Internal seam `qwen4_exp::testing::set_ngram_faults({delay_us, poison, fail_round})`, process-wide, read per gated round, never set by product code. (a) Delay 5 ms + poison (constant FP8 fill, `sfence`, wait, real rows): ids equal faults-off, gate waits > 0 every gated round. (b) A second request admitted mid-decode puts a prefill gather between gated rounds: ids equal faults-off. (c) `system_error` at gated round N: the request fails with it, no hang (test timeout), no sticky CUDA error; runs last since the worker ends |
| 6 | Product checks, CLI | `--print-token-ids` equal to the baseline build: code and story at C = 1, plain and `--spec mtp`; the MTP pair at C = 2; MTP ids still equal plain at C = 1 |
| 7 | Existing | `infernix_qwen4_exp_forward_real_test` calls `read_rows` with the same signature; `infernix_qwen4_exp_ngram_hash_test` unchanged |

Tests 1 and 2 land with S1, tests 3 and 5 with S2, and test 4 with S3; tests 6 and 7 run at every
step.

#### Measurement plan and acceptance

GPU runs use the `gpu.lock` protocol (timing runs also hold `fn\locks\quiet.lock` and model runs
wait for ≥ 72 GiB free RAM, §19.3.0), detached hidden rigs validated by a dry run, and ABBA order on
one binary with temporary toggles, ≥ 5 reps (run-to-run noise is ±0.4-0.75 tok/s). Per arm: tok/s
median, range and worst rep (§19.3.0), plus mean ± sd for S0b's pooled-standard-error test,
after-draft GPU idle (events), host-phase µs, reads per round with exposed and hidden µs, gate
waits, expert hit rate, greedy ids.

| ID | Workload | n-gram state | Decides |
|---|---|---|---|
| W1 | tg512 MTP max 4, `--lm-head-draft`, ≥ 5 reps | warm | headline, S0b, S3 |
| W2 | W1 with the cold toggle | cold | S2, S3 |
| W3 | tg512 plain, warm and cold | both | S0b; no regression |
| W4 | cold CLI code and story, 400 tokens, plain and MTP, new process each | cold, expert-cold | first use |
| W5 | infernix-serve C = 1: unrelated warm-up, then code and story first time, then repeats | cold, then warm | serving |
| W6 | C = 2 MTP pair | mixed | ids vs baseline C = 2 |
| W7 | pp4096, cold n-gram cache | cold | S1 |

| Step | Acceptance (fixed before measuring) |
|---|---|
| All | Greedy ids identical (test 6); any difference blocks the step |
| S0 | Deliver the attribution table (W1, W2: reads, host phases, GPU idle per round; W3: `sample`-to-launch gap), replacing the estimates above |
| S0b | Per flush site (verify, decode, commit): W1 better by > 2 × the pooled standard error; no workload worse beyond its own noise |
| S1 | W7's NVMe block reads per chunk equal its distinct missing blocks (S0 counters), the C++ and prose chunks show −50 % / −29 %, and W7 is not slower; W1-W3 within noise; peak bounce memory 256 KiB |
| S2 | W2, W4-MTP and W5-first faster beyond noise; W1 within noise (expected −0.01 to +0.15 %: the gate's +2-3 µs against 10-50 µs of hidden conflict reads); mean C = 1 gate wait ≤ 10 µs per round |
| S3 | Go: W1 after-draft idle ≥ 0.15 ms per round after S0b and S2. Accept: W1 and W2 faster beyond noise, hit rate within ±0.5 pp, W6 ids identical |
| S4d | W7 (cold n-gram cache, a multi-chunk prompt) faster beyond noise; W7 row-warm within noise; prefill logits bitwise identical (`--dump-logits`); a delayed-producer test of the second landing area (poison, then real rows) gives the same ids |
| Width cost | After S2 and after S3, re-measure W = 3/4/5 round costs; if the slope moves > 0.02, update `kWidthCost` (1607) and c = 0.38 in §19.2 |
| Record | Every result, adverse ones included, in this §19.3.4 while the track is on its branch (§19.3.0), then in §19.2 at merge on the base branch, correcting two claims there: that the after-draft gap is device-bound n-gram reads (the "Pinned uploads" sentence "~0.56 ms after drafting (verification staging, n-gram rows)", and the "Reused n-gram read events" and "I/O ring" bullets), and "reused events changed nothing" (measured n-gram-warm) |

**Measurement rule** (already in §19.3.0 and §12.4; added to §19.2 at merge): tg512 repetitions and
repeated serve requests are n-gram-warm, so n-gram-sensitive changes are judged on n-gram-cold runs.

#### Risks, decisions and open questions

| Risk | Handling |
|---|---|
| Gains near noise: warm S2 ~0, S3 ~0.5-1.7 %; S3 may be inconclusive (a similar Qwen3.5 host-gap item was ≤ 0.2 % and dropped) | The S3 go rule stops it being built on an unmeasured premise |
| A multi-second read stall keeps one gate polling under WDDM; avoiding a TDR relies on Blackwell compute preemption (unverified); a TDR resets the whole GPU and other processes, a 2 s trap only this context | Expected waits are µs; one 3-5 s producer-delay test (below) |
| Stalls and NVMe errors surface inside or after the verify graph (GPU idles at layer 1) | Same total time; error classes and handling unchanged; nsys shows the time elsewhere |
| S3: the expert table is published ~1 ms earlier, which can lower the hit rate slightly; a pinned byte rewritten under a pending upload corrupts it | ±0.5 pp hit-rate gate; single-staging rule and region check, documented in `verify()`; all later staging, including vision's `rope`/`block_rope` regions and prefix prefill uploads (§19.3.0), must keep the rule |
| C ≥ 2: concurrency S1's speculation gate (the default C > 1 policy, merged before n-gram S2; §19.3.0, §19.3.5) makes B ≥ 2 rounds plain, so only B = 1 rounds are gated. Test 5 and W6 therefore interleave B = 1 gated rounds with B = 2 plain rounds. The C = 2 read estimate, S4a's C ≥ 2 trigger and the 327,680 B maximum apply only if S6 + S7 replace the gate. High-miss text: one issuer can exceed block 0 | The gate waits, never worse than today; S4a or S4c |
| A process-wide test seam in product code | Narrow, internal namespace, documented |
| Block 0 and draft times are layer averages | S0's events check them before S2 and S3 are judged |

| Decided (default, 2026-10-04) | Alternative |
|---|---|
| Gate trap 120 s, verified once by a gate test with a 3-5 s producer delay while the GPU is otherwise idle; 2 s if WDDM preemption fails that test | 2 s like `cpu_wait`: bounded GPU waits, but a 2-120 s NVMe stall then loses the context and needs a model reload |
| Internal process-wide fault seam for the engine test: one model load, public options untouched | A test-only environment variable read at Program construction (one load per configuration), or primitive tests only |
| S3 only if S0's events measure ≥ 0.15 ms per round of after-draft GPU idle after S0b and S2 | Another bar, or build S3 (~2 days) unconditionally |
| S4a-c parked until their triggers fire; S4d built after S1 as the cross-chunk read (2026-10-04, §19.3.6) | Build some now; S4d as the within-chunk gate, or after the prefix cache (§19.3.1) |
| §12.3 and §12.4 state this host-hashed, gated design as the single authority (done); the L0 device cache, device-driven requests and NVMe agent are removed from the other sections listed in the Docs step (most on 2026-10-04, the rest at merge), since every token is on the host before its rows are needed and a column's 2.5 KB crosses PCIe in ~1-2 µs | Keep §12.3 as a long-term device-driven target |

**Open: S1b timing.** It reads G: heavily for a few minutes and needs a window with no engine runs.
Proposed default: run it holding `gpu.lock` and `fn\locks\quiet.lock`, since its issuer threads
compete with builds for CPU (§19.3.0).

**Outside this task.** Plain rounds without a drafter leave the GPU idle during commit (no GPU work
before `settle_round`, 645-698 and 814-821, while `after_round` takes ~0.5 ms of host), possibly a
larger plain item; the S0 plain timers measure it. The miss service's `sleep_for(50 µs)` after 20 ms
idle (`miss_service.cpp:111-116`) sleeps ≥ 1 ms, up to 15.6 ms, on Windows (inferred, unmeasured):
cheap to probe. Drafting on the device right after acceptance, as Qwen3.5 does, would remove the
commit-to-draft gap; a larger design. NVMe power-state exits (50-150 ms on some drives, §12.4;
unmeasured on G:) would hit a request's first gated round, and the slow-read warning will show them.

#### S2 as built (branch `claude/fn-ngram-s2`, 2026-10-05)

- **Core.** `upload_pinned_when(dst, pinned_src, bytes, pinned_ready, device_expected, wait_stats,
  wait_row, stream)` and `publish_pinned_word` (`core/device`), as designed: ≤ 16 CTAs of 256
  threads, 32 KiB each; every CTA's thread 0 polls with `__nanosleep(200)`, then
  `__threadfence_system()`, then `__ldcv` loads in batches of eight 16-byte vectors; a wait records
  its ns and a count in `wait_stats[2 * *wait_row]` (CTA 0); trap after 120 s. Test cases in
  `infernix_device_test`: 2,560 / 12,800 / 327,680 B with a 2 ms producer delay over poison, a
  pre-published word (no wait), three graph replays with the old word left published, and a
  misaligned size. The 3-5 s producer-delay TDR probe was not run.
- **Program.** `IoLayout::gate` (a U32 before the rows); `gate_host_` (pinned ready word),
  `gate_sequence_` (strictly increasing, skips 0), `gate_stats_` (per-lane device wait counters,
  zeroed at admission). `decode()` fixes each row's draft count and source before drafting
  (a pure reorder), so `mtp_draft` reads the anchor and copy-row columns while it waits; the
  `sequence_` history copy is gone (`verify_token` builds each window). `verify()` uploads the io
  only up to the rows, launches, enqueues the tail, flushes, reads the rest, publishes, then
  synchronizes; `GateRelease` publishes over zeroed rows if anything throws in between.
  `report_ngram` adds "G gated rounds: R NVMe reads (T ms) behind the gate, the GPU waited in W
  (mean X us)"; a gated round's reads above 100 ms warn.
- **Measured** (one binary, `TEMP-S2` toggle: A reads every row before the launch, B gated; cold
  CLI = new process each, n-gram and expert cold; RTX 5090, CUDA 13.4):

| Workload | A (rows before launch) | B (gated) | B - A |
  |---|---:|---:|---:|
  | Cold CLI story, MTP, 400 tokens, 6 + 6 runs ABBA | 115.50 tok/s (115.3-115.7) | 119.73 (119.3-120.2) | **+3.67 %** (2 x pooled SE 0.31 tok/s) |
  | Cold CLI code, MTP, 400 tokens, 6 + 6 runs ABBA | 131.88 (131.3-132.5) | 132.37 (131.9-132.9) | +0.37 % (+0.48 tok/s against 2 x SE 0.47: at the noise edge) |
  | tg512 MTP (n-gram warm), 2 + 2 files of 3 reps | 139.33 | 139.42 | +0.06 % (within noise) |

  Greedy ids are identical in every run, MTP equals plain at C = 1, the C = 2 serve identity is 6/6
  in both arms, and `prefix_real` passes all 27 checks. The story request's line: 229 gated
  rounds, 4,016 reads (95.9 ms) behind the gate, the GPU waited in 151 rounds for 223 us on
  average, so about two thirds of the read time is hidden. **The acceptance's "mean gate wait <=
  10 us" is not met** (story ~147 us, code ~27 us per gated round): a decode round's ~17 reads take
  ~420 us, ~24 us each, which is close to serial latency, not the probe's ~4.5 us per read. Why the
  overlapped ring does not overlap here is open (next item); the cold code prompt reads few rows in
  decode (403 behind the gate), so it gains little.

#### S4b as built: gated plain rounds (2026-10-05)

The plain decode graph carries the same gate before `ple_embed`. A replayed plain round whose rows
are all in the host cache (`NgramVolume::cached`, a probe of the hashed row ids) stages them before
the launch as before; a round with a miss reads its rows after the launch, while the sampling work
is queued, and publishes before its sync (`sample`'s `before_wait` hook). The first version gated
every replayed round and lost 1.2 % on warm tg512 plain (110.26 -> 108.98 tok/s, sd 0.06; its cold
story gain was +2.0 %); the cache probe removed that.

Measured (one binary, temporary toggle; cold CLI, 400 tokens, 6 + 6 runs ABBA; tg512 2 + 2 files of
3 reps): cold story plain 95.37 -> 97.47 tok/s (**+2.20 %**, ranges 95.3-95.6 / 97.2-97.8), cold
code plain 83.15 -> 83.30 (+0.18 %, within noise: few decode reads), tg512 plain (n-gram warm)
110.31 -> 110.36 (+0.04 %). Greedy ids identical, MTP equals plain, C = 2 serve identity 6/6 in
both arms. Story: 391 gated rounds, 5,960 reads (167.8 ms) behind the gate, the GPU waited in 277
rounds (mean 231 us): about 60 % of the read time is hidden; the rest waits on reads at ~28 us
each.

**Read cost (open).** Alone, one thread issues these overlapped unbuffered reads at 4.7-5.0 us each
(batches of 17-64, no synchronous completions; 9.3 us on an E-core, 8.5 us with half the CPUs
spinning, <= 5.8 us with 64 GiB pinned). Inside the engine they cost ~22-28 us each, and pinning
the engine thread to a P-core or raising its priority changed nothing. Untested: contention with
the GPU's zero-copy staging and the CPU expert team, and parallel issuers (S4c).

#### S0 as built (branch `claude/fn-ngram`)

**Permanent part** (`ab20254e3`).
- `NgramVolume::counters()` replaces the unreported hit/miss pair: rows requested, rows served by
  the host cache, 4 KiB blocks read from the volume, and the wall time of those reads.
- `stage_ngram` adds each call's traffic to the lane it stages: prefill chunks, forced tokens, plain
  rounds and verification rounds.
- `finish()` reports the request's traffic as an Info diagnostic beside the expert-cache line:
  "n-gram rows: N requested, H % host-cache hits, R NVMe reads, T ms of reads". Every read precedes
  its launch until S2, which adds its gate fields to the line. Until S1, R counts one read per
  missed occurrence.

**Temporary part.** One commit labelled `TEMPORARY`, reverted once the S0 campaign has run and
cherry-picked onto the S1-S3 measurement builds, whose acceptance rows use the same events, timers
and toggle. The code is `program/s0_probe.h` plus one-line hooks marked `TEMPORARY`. The Program
reads the environment once, at construction:

| Variable | Effect |
|---|---|
| `INFERNIX_TMP_ARMS=a,b,...` | Arm per admission, cycled. Letters: `v` flush after the verify graph launch; `d` flush after the plain graph launch; `c` flush in `commit_verified` before `settle_round` (A11); `C` n-gram cold (`NgramVolume::invalidate()` at admission); `K` MTP draft length fixed at `--draft-tokens`. `-` is the baseline |
| `INFERNIX_TMP_ARMS_FROM=n` | The first n admissions (bench priming, warm-up) run arm `pre` |
| `INFERNIX_TMP_PHASES=1` | Host clock and CUDA timing events at fixed points of each round |
| `INFERNIX_TMP_DRAFT_LOG=path` | The A12 draft-probability log (TSV, appended) |
| `INFERNIX_TMP_FIXED_K=k` | MTP draft length fixed at k for every arm |

- **Per-request line** (`S0 req=...`, printed at finish). It gives the arm, the MTP and plain round
  counts, the n-gram rows and reads per round, and a **wall-clock decode rate**: decode tokens over
  the time from the first `decode()` entry to the last commit return. `infernix_bench`'s decode rate
  divides by the time inside `decode()` only, so it omits commit and scheduler time, where the
  commit-site flush acts.
- **Phases.** With `INFERNIX_TMP_PHASES=1`, the line adds the mean µs and count of every consecutive
  host-point and GPU-event pair.
  - Host points: decode entry, drafts enqueued, draft sync return, verify staged, verify launched,
    tail enqueued, verify sync return, plain staged, plain launched, sample enqueued, sample sync
    return, decode return, commit entry, commit GPU work enqueued, `settle_round` return, commit
    return.
  - GPU events: decode entry, after the draft graph, after the drafts D2H, before the verify
    uploads, after the licensed D2H, before the plain upload, after the sampled D2H, commit entry,
    commit GPU work enqueued.
  - The GPU pair "after the drafts D2H → before the verify uploads" is the **after-draft GPU idle**
    that S3's go rule reads. "After the sampled D2H → before the next plain upload" is the plain
    gap.
  - Events are recorded outside graph capture, and the chain restarts after prefill and
    forced-token calls.
- **A12 draft log, as built.** It deviates from "an online softmax in the drafter's argmax" so that
  temporary code stays out of the argmax Op.
  - When logging, each draft step copies the draft head's BF16 logits into a probe buffer inside
    the draft graph. After the draft sync, the host computes p1, the softmax top-1 probability at
    T = 1, over the proposal rows (over the public tokens without a proposal head).
  - For sampled rows the log also holds the drafter's sampler distribution q16: the top 16 after
    the request's top-k, min-p and top-p. It also holds the target's sampler distribution p, the
    top 20, at the draft's verify column. From these it gives Σ min(p, q16) and p(draft).
    Penalties are not modelled; the log runs use none.
  - The host work costs ~1 MB of D2H and ~0.5 M exponentials per round, so log runs are never
    timing runs.
- **Scope.** The probe is one per Program, not one per lane, so the S0 runs use C = 1.

**Method and decision rules, fixed before measuring.**
- **Arms.** They alternate per request inside one `infernix_bench` process, in mirrored order
  (A B C … C B A), after the priming request and one warm-up. All arms therefore share one model
  load, one expert-cache history and the same greedy text.
- **Repetitions.** 10 reps per arm for the flush A/B runs, which leave the phase timers off. The
  attribution runs, with phases, use 4 reps per arm.
- **Speed metric.** The wall-clock decode rate, with the bench rate reported beside it.
- **Flush adoption.** A flush site is adopted if its deciding workload improves on the wall metric
  by more than 2 × the pooled standard error, √(sd_A²/n_A + sd_B²/n_B). Neither its cold variant
  nor the bench metric may be worse by more than 2 × its own pooled standard error.
- **Deciding workloads.** W1 decides the verify and commit sites, with W2 as the cold variant. W3
  plain decides the decode site, with W3 cold as the variant: that site runs only in W = 1 rounds,
  which MTP tg512 rarely has, so reading §7.3's "W1 improves" literally would test a path W1
  barely executes.
- **A12.** O2 is built only if the replay shows ≥ 3 % on prose (story, tg512) net of draft cost.
  O3 is built only if Σ min(p, q16) exceeds p(draft) by ≥ 5 acceptance points on sampled prose.
- **Workloads as rigged.**
  - W1, W2: tg512 MTP 4, arms `-`, `C`, `v`, `c`.
  - W3: tg512 plain, arms `-`, `C`, `d`, `Cd`.
  - W4: CLI code and story, 400 tokens, plain and MTP, one process each, phases on.
  - W5: infernix-serve C = 1 with phases on: an unrelated warm-up, then code and story for the first
    time, then repeated.
  - W7 baseline for S1: pp4096 at chunk 4096, cold against warm.
  - A12 logs: tg512, and CLI code and story, greedy and sampled (T = 0.7, top-p 0.8, top-k 20),
    with arms `-` and `K`.

### 19.3.5 Speculation and throughput at C > 1

Design, adversarially reviewed (revision 2), nothing implemented or run on the GPU (worktree
`NInfer-V3-flashnext`, HEAD `46a56fc8f`, 2026-10-04). Labels: **measured** (log named), **code**
(`file:line` at HEAD), **model** (`fn/plans/concurrency_model.py`, a planning tool of about
±30 %), **inference** (names the check that settles it). Code references are at `46a56fc8f`. Since
`54deaa2bc` (the base commit, §19.3.0) `ProgramImpl` is in `program/program_impl.h`, where
`program.cpp:N` is line N − 2 (e.g. `choose_draft_length` 561, the n-gram proposer 579,
`decode_budget` 792-798, the chain copies 1408-1409, `kWidthCost` 1605).

#### Why C = 2 with speculation is slow

A speculative C = 2 round costs more than the two rows' rounds run one after the other:

1. **Per-row draft lengths.** Each row picks K as if alone (`choose_draft_length`, slope
   `kWidthCost = 0.38`, §19.2); the round width is 1 + max K and the other rows are padded to it.
2. **Fillers** (the last draft, or the anchor) run the whole forward, routing included. Experts
   only fillers use become jobs and misses, are never credited (§11.3), and miss every round.
3. **CPU cap per layer call.** At most `cpu_expert_jobs = 8` misses go to the CPU whatever B is;
   the rest queue on the PCIe stage at ~115 µs each.
4. **No fork at ≥ 7 columns:** `forked()` requires `max_jobs = min(512, 10·T) ≤ 64`
   (`moe_layer.cu:108-111`), so wider calls stage and compute serially.
5. **Promotions cross PCIe twice.** Until the frames are full, `decode_budget` promotes one expert
   per layer per advanced token (`program.cpp:794-800`; the longest row's advance, 1 for a plain
   round of any B): a 2.76 MB copy of an expert usually just staged over the same link. Every cold
   test so far ran in this **fill** phase, where the x8 link limits verification, **at C = 1 too**.

Calls of ≤ 4 columns use one-column expert kernels (two CTAs per SM, one pass per column); wider
calls use 8-column kernels (one CTA per SM). Both stage a job's weight slice once
(`moe_layer.cu:394-397`); the 8-column kernel repeats only the dp4a work per column
(`moe_layer.cu:323-351`), so its cost follows the **job count** much more than the column count.

#### Measured facts

Recipe B (dense8m), INT8 KV, max-context 4096, greedy, 200 tokens; code and story requests sent
together to a fresh server, then each alone (`fn/mtp/{c1,c2,c2p,c2n}/serve.err`):

| Run (build) | Mode | Code tok/s | Story tok/s | Notes |
|---|---|---:|---:|---|
| c1 (`6e5d93500-dirty`, committed as `ecaf50e16`, §19.2; ≈ HEAD) | C = 1, MTP max 4, `--lm-head-draft` | 70.6 fill / 147.2 warm | 61.8 turnover / 97.5 warm | queued; acceptance 92.6 / 61.6 % |
| c2 (`5429c31a4-dirty`, older) | C = 2, MTP max 3 | 24.9 | 21.7 | 39.7 tok/s aggregate at batch 2.00 |
| c2p (`69b57d6d4`) | C = 2, plain | 35.9 | 36.1 | alone and warm: 82.8 / 82.6 |
| c2n (`69b57d6d4`) | C = 2, n-gram K = 7, min-match 4 | 27.0 | 21.7 | code: 24 rounds, 134/168 accepted; story: no drafts |

- **Per-request tok/s** is output over Σ `decode_ns` of the rounds the row joined. `decode()` and
  `verify()` synchronize, so it covers the forward and excludes commit-side work (catch-up, GDN
  fold, `settle_round`). Both rows are charged the whole round, so the aggregate ≈ the sum.
- **Cache counters** are global deltas since admission (overlapping at C > 1), credited columns
  only. Fill-phase promotions: c1 code 8,378 over ~49 rounds, c2 code 8,624 over ~55, c2p 9,059
  over 199, against 8,664-9,381 frames. c1 story is the **turnover** regime (frames full, new
  topic, one promotion per layer every 4 tokens): 3,069 promotions, 58.5 % hits.
- **Plain C = 2, fill:** 199 rounds in 5.54 s of `decode_ns`, **27.8 ms per B = 2 round** (≈ 72
  tok/s); (194,423 − 4,549) / 199 / 48 ≈ **19.9** distinct live experts per layer against 10 at
  B = 1.
- **N-gram C = 2, fill:** 24 rounds yield 158 tokens; less the other 41 tokens' plain rounds,
  **~260 ms per B = 2, W = 8 round** against ~56 ms alone and warm; joint aggregate ≈ 36 tok/s.
- **MTP C = 2 (older build):** ~150 ms per verification round, without the pinned uploads, two-CTA
  kernels, 8 CPU jobs, deferred cache update and few-row Q8 route (+24 % at C = 1 together); S2
  re-measures at HEAD. Revision 1's ρ_2 = 27.8 / 15.1 mixed serve `decode_ns` with CLI wall time on
  different texts; S2 measures ρ_B (the period of a B-row round over that of a one-row round) within
  one metric.

#### Where the time goes

Distinct experts per layer, 512 routed tokens of the frozen code text (`fn/predict/routes.bin`):

| Rows (live widths) | Distinct per layer | Share used by one column |
|---|---|---|
| 1 row, W = 1 / 2 / 3 / 4 / 5 / 8 | 10 / 16.6 / 21.9 / 27.0 / 31.4 / 42.6 | 1.00 / 0.80 / 0.72 / 0.69 / 0.66 / 0.59 |
| 2 rows, 1 + 1 / 5 + 1 masked / 5 + 5 padded | 18.3 (unrelated, measured: 19.9) / 36.6 / 52.2 | 0.91 / 0.66 / 0.59 |
| 4 rows, W = 1 / 5 + 1 + 5 + 1 / 5 × 4 | 31.7 / 61.0 / 83.4 | 0.80 / 0.60 / 0.53 |
| 8 rows, W = 1 / 5 | 53.9 / 125.3 | 0.71 / 0.44 |

A draft column adds ~5 experts per layer; unrelated rows are almost fully additive; one request's
512-token working set (309.5 per layer, 14.9 K in all) exceeds the ~8,700-9,400 frames; most jobs
have one column, so a CPU cap fills mostly with one-column jobs.

- **Expert kernels cost per job plus per column.** Profiles (§19.2): warm C = 1 plain ~2.9 ms per
  round for 480 one-column jobs (~6 µs each); W = 4 on the 8-column kernel, before the two-CTA fix,
  12.9 ms for ~1,270 jobs and 1,920 entries. Assumed until S2: 8-column c_job ≈ 8 µs, c_col ≈ 1.5
  µs; one-column c_job ≈ 3 µs, c_col ≈ 3 µs. So a filler costs little compute; it costs its
  **filler-only experts**, each a job and usually a recurring miss (52-115 µs). Revision 1's 2.4 ms
  per filler column overstated the compute.
- **CPU jobs** (`fn/cpu/run.log`, cold, 8 workers): 51.5-59.9 µs per expert at one column, 102-124
  at four, so **52 + 18·(n − 1) µs**. 12-24 workers were slower (this log: 75-88 µs per expert at
  one column, 175-190 at four; §19.2's `cpu_bench` summary: 59-83 at one column). Six workers won
  end to end (§19.2), and only warm one-column records (`dyn.log`) favoured more, so no gain from
  more workers is assumed. A higher cap pulls in 2-5-column jobs (70-124 µs). The split adapts to
  each call's misses M, so the cap binds only when M − M/3 > cap (M ≥ 13 at cap 8): C = 1 MTP cold
  or turnover (M ≈ 13-15), C = 4 plain (≈ 15), padded C = 2 verification; not plain B = 2
  (M ≈ 7-8). CPU, stage and promotions share host DRAM: in §19.2's tg512 MTP sweep, giving the CPU
  every miss up to the cap (jobs / divisor 8 / 0) ran 130.5 tok/s against 133.6 at 8 / 3.
- **The x8 link** (27.6 GB/s; ~100 µs per 2.76 MB record). The table below gives fill-phase link
  time per round. C = 1 cold MTP sits on the link limit: a higher cap alone moves little, removing
  the second crossing (S4) frees it. In turnover and warm, promotions fall to ~54 per round, ~5 ms
  of link per MTP round.

  | Round (fill) | Staged | Promotions | Link | Round |
  |---|---:|---:|---:|---:|
  | C = 1 plain / C = 2 plain | ~48-50 / ~130 | 48 / 48 | ~9.7 / ~18 ms | 15.1 (CLI) / 27.8 ms |
  | C = 1 MTP K = 4, cap 8 | ~310 | ~217 | **~53 ms** | ~54 ms (CLI 83.7 tok/s) |
  | C = 2 MTP K = (4, 0), cap 16, no landing (model) | ~270 | ~217 | ~49 ms | ~53 ms |

- **Fork and empty CTAs.** Calls of ≥ 7 columns run serially and lose the overlap of staging with
  resident compute: ~0.1-0.2 ms per layer at C = 2 cold (model). In a warm C = 2 verification
  round this loss is the largest term. Forked calls launch `(40, max_jobs)` CTAs per kernel per
  phase: 8,000 per layer for ~31 jobs at T = 5, and 16,000 for ~45 at T = 10 once S5 item 1 lets it
  fork (serial today: 8,000, review finding 7); an 8-column CTA reserves ~72 KB, so the early
  exits run one per SM in turn. Unmeasured (S2).
- **Fillers.** c2n 8 + 1 real of 16 columns, 44 %; MTP K = (4, 1) 5 + 2 of 10, 30 %; (4, 0) 40 %.
  Besides its filler-only experts (their count is unknown until S2 measures it), each filler costs
  ~0.19 ms of non-MoE work (dense, QSA, GDN record, LM-head column, catch-up cell) plus a drafter
  step per extra W; NVMe rows for the first `ngram_size − 1` fillers (~4.5 µs per row; later ones
  hit the host row cache); and B·W > 8 puts dense Q8 on the MMA tiles (§19.2, "Two lanes"). That
  is a rounding change and, until §19.3.3's K1 covers T = 9-16, also a cost: ≈ 14-20 ms per
  forward against ~3.6 ms at T ≤ 8 (§19.3.3, inference). The round model has no MMA term, so the
  T > 8 estimates below (C = 2 HEAD K = (4, 1); C = 2 S1-S6 and S1-S7; C = 4 S1-S7) are optimistic
  by up to that amount per round.

**Round model** (revision 2):

```text
t_layer = max(t_cpu, t_stage, t_res) + 30 us        (forked; serial: max(t_cpu, t_stage + t_res))
  t_cpu   = sum over CPU jobs (52 + 18 (n - 1)) us + wait copy (1 us per job / wait CTAs)
  t_stage = staged x 115 us + staged jobs x (c_job + c_col x columns per job)
  t_res   = resident jobs x (c_job + c_col x columns per job)
t_layer >= (CPU jobs + staged + promotions / 48) x 2.76 MB / DRAM
round    = max(fixed(T) + 48 t_layer, link, DRAM)
  fixed(T) = 7.0 + 0.12 T [+ 1.0 + 0.07 T for verification] + 0.35 ms per draft step
  link     = (48 x staged + promotions) x 2.76 MB / 27.6 GB/s
  DRAM     = (48 x (CPU + staged) + promotions) x 2.76 MB / DRAM
```

DRAM is the host DRAM bandwidth that CPU jobs, staging and promotions share (70 or 55 GB/s in the
tables); h is the live expert hit rate. CPU jobs take the fewest-column misses first, with the
column distribution above. Regimes as (live hit rate, promotions per layer per advanced token): fill
(0.54, 1), turnover (0.58, 0.25), warm (0.84, 0.25); with S4, fill promotions are max(0,
budget − landed).

| Calibration | Model, DRAM 70 / 55 GB/s | Measured |
|---|---:|---:|
| C = 1 plain warm (h = 0.884) / fill (h = 0.693) | 11.2 / 12.0; 16.3 / 18.4 ms | 11.5; 15.1 ms (CLI) |
| C = 2 plain fill (h = 0.595) | 23.3 / 28.9 ms | 27.8 ms (serve) |
| C = 1 MTP K = 4 code, fill | 86 / 79 tok/s | CLI 83.7 (h ~0.69); serve 70.6 (h 0.54) |
| C = 1 MTP K = 4 code, warm | 181 / 170 tok/s | serve 147 |

At 70 GB/s plain rounds fit within −8 % to +16 % ((measured − model) / measured); at 55 GB/s plain
fill is 22 % high (18.4 against 15.1 ms) and the others within 5 %. Verification rounds are
~20 % optimistic against serve. The per-layer DRAM floor explains most of the plain C = 2 fill time.
Rejected variant: promotions as full link contention on the stage give C = 1 MTP fill 66 tok/s
against 83.7 measured.

**Estimates** (model, aggregate tok/s, DRAM 70 / 55; a code and a story row; conditional
acceptance code ~0.95, story 0.62, 0.5, 0.45, 0.4). Verification figures are undiscounted; expect
~0.8×. Expected aggregate tok/s, verification discounted, DRAM 70 / 55:

| Workload | HEAD | After S1 | After S1-S4 | After S1-S7 (if accepted) |
|---|---:|---:|---:|---:|
| C = 1 MTP, fill | 69 / 63 (serve measured 70.6) | same | 89 / 78 | same as S1-S4 |
| C = 1 MTP, turnover | 84 / 78 | same | 93 / 78 | same |
| C = 1 MTP, warm | 145 / 136 (measured 147) | same | same | same |
| C = 2, fill | ~43 (old build measured ~40) | 75 / 64 (plain measured 72) | 80 / 69 | 97 / 85 |
| C = 2, turnover | ~47 | 84 / 72 | 84 / 72 | 103 / 86 |
| C = 2, warm | ~95 | 136 / 122 | 136 / 122 | 163 / 158 |
| C = 4, fill / turnover / warm | — | 85 / 100 / 211 | 107 / 115 / 211 (DRAM 70) | 124 / 134 / 213 |

Undiscounted model figures (plain rows need no discount):

| Configuration | Fill | Turnover | Warm |
|---|---:|---:|---:|
| C = 1 plain; with S3 + S4 | 52 / 45; 57 / 51 | 59 / 52 | 83 / 77 |
| C = 1 MTP K = 4: HEAD; S3 (cap 16); S3 + S4 | 86 / 79; 95 / 79; **111 / 97** | 105 / 98; 116 / 98; 116 / 98 | 181 / 170 |
| C = 2 HEAD, per-row K = (4, 1), padded | 54 | 59 | 119 |
| C = 2 S1 (gate); S1 + S3 + S4 | 75 / 64; 80 / 69 | 84 / 72 | 136 / 122 |
| C = 2 S1-S6, K = (4, 0); S1-S7, best K | 119 / 104; 121 / 106 (K = (4, 1)) | 126 / 105; 129 / 108 | 196 / 187; 204 / 197 |
| C = 4 S1 (gate), cap 8; S1 + S3 (cap 24) + S4 | 85 / 85; 107 / 91 | 100 / 96; 115 / 96 | 211 / 184 |
| C = 4 S1-S7, K = (4, 0, 4, 0) | 155 / 134 | 167 / 140 | 266 |

- **The gate is most of the near-term C = 2 gain:** ~40-55 → the plain-C = 2 level.
- **S3 and S4 matter mostly at C = 1 cold and at C = 4.** S3 does nothing for plain B = 2, and at
  C = 1 its gain needs DRAM headroom (none at 55 GB/s). Without S3 and S4 the best joint K on a
  cold cache is short, (1, 0) or (2, 0); with them long drafts pay again.
- **Speculation at B ≥ 2 (S6 + S7)** beats the gate by +20-30 % at C = 2, ~+16 % at C = 4 cold or
  turnover, ~0 at C = 4 warm (model), at the price of concurrency-dependent greedy output.

#### Steps

Each step is a separate commit with its own tests and A/B; S2-S6 change no token at fixed draft
decisions. S1-S5 take ~5-6 working days and ~5 h of GPU; S4b and S8, added from the Strata review
on 2026-10-04 (§19.3.6), take ~1.5-2.5 days and ~1.5 h more; S6-S7 ~2.5-3 more, only if accepted.
Docs: per step, results (including adverse ones) in this §19.3.5 (§19.3.0). At merge time, on the
base branch: §10.3 (cap, plan, wait), §9 (landing; §9.3 item 4 for S4b), §11.3 (gate, joint
policy), the scheduler description for S8 and the §19.2 "Next" list, plus the C > 1 advice in the
user guide (`docs/qwen3_8-flash-next.md`): C > 1 mainly buys TTFT for queued requests, and each
lane costs ~387 frames (~1.5 % decode for a request alone, estimate) at 64K INT8 without
`--kv-capacity`, ~42 frames (~0.2 %) with a fixed `--kv-capacity`; a startup line prints frames
per lane (~0.6 day in all).

| Step | Changes a token? | Affects C = 1? |
|---|---|---|
| S1 | no; C > 1 output should equal C = 1 | no |
| S2 | no | adds a report line |
| S3 | no | speed only (cold and turnover MTP; faster plan and wait) |
| S4 | no | speed only (fill phase) |
| S4b | no (placement-invariant) | speed only (first tokens after a restart); +~1 s at startup |
| S5 | no | speed only (launch shapes; n-gram W ≥ 7 forks) |
| S8 | no (prefill keeps its chunk grid) | no (C ≥ 2 only) |
| S6 | no for valid columns | no (one row has no fillers) |
| S7 | yes at B ≥ 2 (MMA above 8 columns) | no (B = 1 keeps `choose_draft_length`) |

**S1. Speculation gate (default; 0.25 day).** In `ProgramImpl::decode`, `speculate = batch == 1`;
`wanted[b] = speculate && lane.mtp_live ? choose_draft_length(lane) : 0` and the n-gram proposer
runs only when `speculate` (`program.cpp:563, 581`). B ≥ 2 rounds decode plain, sharing the dense
reads without padding; the W = 1 catch-up keeps the drafter current. `mtp_draft` still runs for
unwritten pending cells (a lane fresh from prefill). S1 moves its early `return`
(`program.cpp:1413`) ahead of the per-row chain copies (`program.cpp:1410-1411`) when `steps == 0 &&
!unwritten`, since those copies only serve drafting. `mtp_policy_rounds` advances only in B = 1
rounds. A local build reads an environment override (as the `INFERNIX_Q4_*` sweeps did), not
committed. **Numerics** unchanged: B ≥ 2 rounds are plain with T = B ≤ 8 on SIMT dense routes,
column-invariant for 1-8 columns (§19.2); experts are batch-invariant; MTP equals plain at C = 1
(measured). So C ≤ 8 greedy output equals C = 1 (**inference**, checked in campaign step 1).
**Expected:** plain C = 2 measured 72 on a plain-only server; with MTP loaded (~700 fewer frames, a
catch-up per round) ~68-72 fill and ~120-135 warm, against ~40-55 at HEAD. User guide: `--spec` at
`--max-concurrency ≥ 2` speculates only while one request decodes.

*S1 as implemented* (branch `claude/fn-conc`, 2026-10-04; built, not yet measured):

- **Gate.** As planned: `speculate = batch == 1` gates both `choose_draft_length` and the n-gram
  proposer, and `mtp_draft` returns before the chain copies when `steps == 0` and no pending cell
  is unwritten. No product option.
- **Per-lane startup line** (Strata A13, §19.3.6). The Program prints `expert cache: F frames of
  2.64 MiB at C lanes; each lane holds N frames (KV x MiB; recurrent state, records and workspace
  y MiB)`, with `KV in the fixed --kv-capacity pool` when `--kv-capacity` is set. N is exact, to
  one decimal: every device buffer allocated per lane (its KV pool share unless the pool is fixed;
  GDN state, PLE convolution, QSA tails; logits and penalty counts; GDN, PLE and QSA verification
  records; drafter residuals and records), plus the workspace's per-lane growth,
  (capacity(C) − capacity(1)) / (C − 1), or capacity(2) − capacity(1) at C = 1. The arena and the
  line share one sizing function, `ProgramImpl::workspace_capacity(lanes)`.
- **Tests.** No unit test, as planned: the gate is decision code inside `ProgramImpl`. The rig's
  byte-identity check against each request alone at C = 1 and its per-(B, W) round counts cover
  it.
- **Measurement build, never committed** (`fn/rigs/conc/s1-meas-temporary.patch`). The override
  `INFERNIX_Q4_CONC_UNGATED` (any value) lets B ≥ 2 rounds speculate as before S1. A temporary
  per-(B, W) round table is printed with each request's cache line: rounds, submit time, and S2's
  period (one `decode()` entry to the next, dropping intervals with a prefill or forced-token
  call) with the tokens those rounds committed. S2 replaces it with the permanent table.

**S2. Diagnostics and kernel constants (0.75 day)**, kept as the C > 1 tuning diagnostic (§11.3
asks for U_T and h_now):

1. **Device counters.** `MoeExpertSource::counters` (u64*, nullptr disables): `cpu_plan_kernel`
   adds CPU jobs and their columns, `stage_kernel` block 0 staged and (S4) landed misses. One
   4 × u64 array per Program via `ForwardExperts`, downloaded beside the route log; per-round deltas
   on the host. This fixes the host counter's lag (the service writes `done` before its counter).
2. **All-column stats** in `ExpertResidency::after_round` (with `spec_layout_.extents`): distinct
   experts, distinct non-resident at round start, experts used only by fillers (j > n_b) or only by
   rejected drafts (`Stats::routed_all`, `misses_all`, `filler_only`, `rejected_only`).
3. **Round table** `round_stats_[B][W]`: count; Σ period, Σ period·ΣK, ΣK, ΣK² (fit of time against
   total drafts); Σ committed tokens, all-column misses, CPU jobs and columns, staged, landed,
   promotions, filler-only experts. **Period** = time between successive `decode()` entries with no
   `prefill()` or `append_forced()` between. `report_cache` prints request and since-start tables,
   one line per (B, W) with ≥ 5 rounds, e.g. `rounds B2W1 187x 26.9 ms, 7.9 miss/layer (5.1 CPU,
   1.4 col/job), 2.8 staged, 48 promo`.
4. **Microbenchmark** `bench/ops/offloaded_moe_bench.cu` (E = 512, top-k 10, fixtures from
   `tests/ops/offloaded_moe_fixtures.h`): `moe_experts` at T ∈ {1, 2, 5, 8, 10, 16, 40} × resident
   share {0.5, 0.9}, CPU off, forked and serial, plus job_count = 0 per grid size; CUDA events over
   a graph of 50 calls. It yields c_job, c_col per family and the empty-grid time per
   (T, max_jobs), refits the model and decides S5's compaction.
5. **Rigs** (`fn/conc/`, not in the repository): the three regimes, each request's table logged.

**S3. One larger CPU cap, parallel plan and wait (1-1.5 days).** Today `cpu_plan_kernel` is one CTA
whose thread 0 picks `want = min(cap, M − M/divisor)` fewest columns first in up to 8 serial passes
on the critical path, then copies **all** T columns of x to mapped memory; `cpu_wait_kernel` is one
CTA; `CpuCall` sits in a fixed 256-byte tail (`8 + 4·kMaxCpuJobs` bytes: overflow above 62).

1. `kMaxCpuJobs` 8 → 32: `MissRequest` ~1.3 KB, mapped y 2 × 2560 × 32 × 8 = 1.31 MB, the team's
   `max_jobs` and the wait slot table grow. The tail is sized from `sizeof(CpuCall)` rounded to
   256 B, with `static_assert(sizeof(CpuCall) <= 1024)`.
2. **One cap for every B:** `ProgramOptions::cpu_expert_jobs` = the sweep winner (expected 12-16),
   clamp to `kMaxCpuJobs` kept. No per-B table: the split adapts to M, and eager calls (prefill
   tails, forced tokens with T ≤ lanes × max_width) use the same cap. A per-B rule only if the
   sweep shows real B dependence, and it must then define the eager calls' value.
3. `MoeCpuChannel::max_job_columns` (default 8) bounds the width loop; an option only if < 8 wins.
4. **Block-parallel plan:** thread j mod 256 flags misses and columns; a block reduction gives M and
   `want`; for w = 1..max_job_columns a block prefix sum ranks w-column misses in job order and
   takes those with rank < want − n, n being the misses already taken at narrower widths. The
   selection and its order in `call->job` are today's.
5. **Publish only the chosen columns** (a shared bitmap, T ≤ 64) at their original offsets; the
   request format is unchanged. At T = 10 the mapped write per layer falls from ~51 to ~5-15 KB.
6. **Multi-CTA wait:** `min(8, kMaxCpuJobs)` CTAs all poll `done` (same 2 s trap); CTA c copies
   jobs i ≡ c (mod gridDim.x), 8× the mapped reads in flight.

CPU and GPU results are bit-identical (§16.2; the layer-route test, §19.2), so no token changes;
C = 1 cold and turnover MTP change speed, plain C = 1 (M ≈ 3) only gets the faster plan and wait.
**Sweep** (environment overrides): cap {8, 12, 16, 24, 32} × divisor {2, 3, 4} × job columns {1, 2,
8} × workers {6, 8}, staged: cap × divisor (8 columns, 6 workers) on C = 1 MTP fill and turnover and
C = 4 plain fill, then columns and workers on the top three, then every workload. Verification-heavy
rounds must be in it; the single winner is committed.

*S3 as implemented* (branch `claude/fn-conc-s3`, 2026-10-05):

- **Capacity.** `kMaxCpuJobs` 32 and `kMaxCpuCallColumns` 128 (8 lanes × verify width 16; the
  service and `moe_experts` refuse more). `MissRequest` 1,296 B. The service's mapped y and its
  team are sized by the configured cap, not by `kMaxCpuJobs`. `CpuCall` (136 B) still fits the
  256-byte workspace tail (static assertion).
- **Job columns.** `MoeCpuChannel::max_job_columns` (1..8, default 8) bounds the width loop;
  `CpuMissService::Options::max_job_columns` sets it.
- **Plan.** One CTA of 256 threads: each thread counts its jobs' misses (M) and eligible misses
  per width; thread 0 forms `want` and each width's take and first slot; then, per chunk of 256
  jobs, a warp ballot per width ranks every miss among the same-width misses before it (earlier
  chunks, lower warps, lower lanes), so the n-th chosen miss lands in today's slot (fewest
  columns first, job order within a width) without a serial walk. It publishes only the x
  columns the chosen jobs read (a shared bitmap; 16-byte stores at their original offsets).
- **Wait.** `kWaitCtas = min(8, kMaxCpuJobs)` CTAs; a CTA with no job (index ≥ n) exits at once,
  the others poll `done` (2 s trap) and place jobs c, c + 8, ...
- **Tests.** `test_offloaded_moe_layer.cu`: services with cap 32 / divisor 0 and cap 24 /
  divisor 2 / 2-column jobs, on the pass, serial and fork routes; an oracle of the number of
  CPU-served experts (min(want, eligible misses)); shapes with more misses than the largest cap
  (E 96, T 10: ten published columns of 10) and with more than 256 jobs (E 400, T 64).
- **Sweep tooling (never committed).** `fn/rigs/conc/s3_sweep_patch.py` adds TEMP-S3 environment
  overrides (`INFERNIX_Q4_CPU_JOBS`, `_DIVISOR`, `_WORKERS`, `_JOB_COLUMNS`) to a measurement
  build; `s3_sweep.bat` (stage 1, cap × divisor, the 8/3 default three times as drift control)
  and `analyze_s3.py`.

**S4. Fill-phase landing (1.5 days).** While free frames remain, a staged miss lands in a reserved
free frame instead of a staging slot and becomes resident, replacing its later promotion.
CPU-served misses are still promoted within the budget.

- **Op.** `MoeExpertSource` gains `landing` (device frames for the call, −1 = none), `landed`
  (device log of the landed experts) and `landing_slots`. The n-th staged miss in job order goes
  to `frame_base + landing[n] × stride` when `n < landing_slots && landing[n] ≥ 0`, else the next
  staging slot (with S5, else zero-copy); `job_records[j]` gets that address, block 0 writes
  `landed[n] = expert`, and the copy loop reads per-miss destinations from shared memory. Forked
  calls only (eager prefill passes nullptr). The device table is not touched within the round:
  `pass_record` resolves a not-yet-resident expert through `job_records`.
- **Program.** `ForwardExperts` carries the fields (not for `eager_chunk_`). `CacheController`
  gains `reserve_free(n, out)` (pops free frames, LFRU untouched), `adopt(key, frame, out)` (Ready
  there, `LfruPolicy::promote`, a victim evicted as in `on_route`, a queued load dequeued),
  `release(frame)` and `free_frames()`. `ExpertResidency::before_round(landing)` (decode and
  verify only) first releases a discarded round's reservations, then reserves
  `min(kLandingSlots = 16, free / 48)` frames per layer, uploads the pinned [48][16] table with
  `upload_pinned` (3 KB) and memsets the landed log, outside the graph at fixed addresses; the log
  (3 KB) returns with the route download. `after_round`, per layer: count hits against the
  round-start table; adopt the landed pairs (`table[key] = frame`, `table_dirty_`); call `on_route`
  with budget **max(0, budget − landed_l)**; release unused reservations; count `Stats::landed`.
  `decode_budget`'s fill test becomes `promotions + landed < frames`.
- **Effect** (model): C = 1 MTP stages ~4.8-6.5 misses per layer against a budget of ~4.5, so
  promotions fall to ~0 and link time per round from ~53 to ~23 ms; the fill ends sooner. Fill-phase
  admission becomes every staged miss instead of the LFRU's highest-score misses (LFRU score
  f / (now − last + 1), §9.3); LFRU eviction governs once full; with no free frames nothing changes.
  Bit-neutral (a landed frame holds the identical record). C = 1: the first ~100-200 decode tokens
  after start and every cold benchmark.

*S4 as implemented* (branch `claude/fn-conc-s4`, over S4b and S3, 2026-10-05):

- **Op.** `MoeExpertSource::landing` / `landed` / `landing_slots` (≤ `kMaxLandingSlots` = 32), honoured
  only on the fork route (decode and verification); the pass and serial routes ignore them.
  `stage_kernel` now ranks its misses with warp ballots in chunks of 256 jobs (no serial walk), the
  n-th miss (CPU-served jobs excluded) lands in `landing[n]` when that is a frame (block 0 logs
  `landed[n]`), others take the next staging slot; per-miss destinations sit in shared memory.
- **Program.** `CacheController::reserve_free` (pops free frames while the policy has room),
  `adopt` (READY in its frame, admitted with an LFRU victim outside the layer's group, a queued
  load dropped by `drain_queue`, which now also skips keys that are no longer absent; an in-flight
  copy of the key never publishes since adoption bumps its serial), `release`, `free_frames`.
  `ExpertResidency::before_round(compute, landing)` (decode and verify only) returns any
  unsettled reservations, reserves min(16, free / layers) frames per layer, uploads the
  [48][16] table with `upload_pinned` and clears the landed log; the log returns with the route
  download; `after_round` adopts each layer's landed experts before `on_route`, whose budget
  becomes max(0, budget − landed). Reservations are also returned before a resize or a lend.
  `decode_budget` counts `promotions + landed + seeded`; the per-request report adds "landed".
- **Tests.** Layer test: an exact oracle of the landed experts (the first staged misses in job
  order, CPU-selected misses excluded, a −1 slot skipped, nothing on non-fork routes) and their
  frame bytes. Expert-cache test: reserve/adopt/release semantics.
- **Measured** (2026-10-05, cold CLI code prompt, `--expert-state off`, ABBA in one build with a
  temporary toggle): plain 74.6/74.7 → 81.3/80.8 tok/s (+8.6 %), MTP 89.8/89.2 → 133.1/133.1
  (+48.7 %); ~5.7-6.3 K landed experts replace ~55-60 % of the promotions. The 7,448-token prompt's
  MTP decode 60.6 → 91.0 tok/s (one run per arm). C = 4 plain fill through `infernix-serve` 57.8 →
  73.5 tok/s aggregate (one run per arm), outputs identical. Greedy ids of code and story, plain
  and MTP, equal the pre-S4 build. **Open:** the long prompt's ids varied between runs in both
  arms; the same variation appears without S4 and is traced to the wide prefill route (§19.3.8,
  "F1 nondeterminism"), not to landing, which is decode-only.

**S4b. Warm start (2026-10-04, from Strata; 1-1.5 days).** §9.3 item 4 plans seed and persistence,
but `LfruPolicy::seed` has no caller, so every start begins with 0 resident experts. Strata fills
VRAM at start from a ranking: the user's saved state if one exists (VRAM experts, then counted
routing, written at quit and every 10 minutes through a temporary file and a rename, #477), else a
shipped profile. It published no gain.

- **Save.** Per key count and last use, the resident set and the clock (~300 KB), with artifact
  identity, recipe and frame count. Written via temporary file and rename at stop and every
  N minutes between requests (never during a round).
- **Load.** After the frames exist and before the first admission:
  - reject a file from another artifact, recipe or frame count;
  - seed `LfruPolicy` with the counts scaled or capped, so stale experts are not sticky under
    f / (age + 1);
  - copy the residents on the copy stream (9,443 × 2.76 MB = 26.1 GB, ~0.95 s at 27.6 GB/s) and
    upload the table.
- **Budget rule.** `decode_budget` keys the fill rate on `promotions < frames`, and S4 makes it
  `promotions + landed < frames`. S4b adds the seeded frames. Otherwise a seeded cache would keep
  promoting at the link-bound fill rate (~53 ms of link per MTP round) for ~9,443 promotions.
- **Fallback profile.** A shipped profile, built from our own mixed code, prose and multilingual
  route logs (never Strata's data), only if `tools/expert_cache_replay` shows seed ≥ empty over the
  first 256 tokens on a mismatched workload. §9.3 measured a static profile at +82 % misses on B.
- **Effect** (estimate; C = 1, the first ~200 tokens after a restart, partly to ~500).
  - Same workload as the saved state: up to the measured cold-to-warm gap (cold CLI code
    64.7 / 81.9 tok/s plain / MTP against tg512 warm 88.7 / 133.5). That is an upper bound, since
    tg512 warm is partly replay.
  - Unrelated workload: about the turnover regime (model C = 1 MTP 69/63 → 84/78 at HEAD, but only
    89/78 → 93/78 after S3 + S4).
  - Bit-neutral: expert arithmetic is placement-invariant. Cold benchmarks state whether a seed
    was used.

*S4b as implemented* (branch `claude/fn-conc-s4b`, 2026-10-05):

- **State.** `expert_cache::SavedState` = every key's LFRU count and the resident keys by score,
  highest first (`ExpertResidency::saved_state`). File `NINFQ4ES` v1 (`expert_cache/expert_state.{h,cpp}`):
  magic, version, an identity string, key count, counts, ranking, footer; written through `.tmp` and
  a rename; a load returns the state or the reason it was refused (missing, other identity or key
  count, truncated). The identity is the artifact's absolute path, size and modification time: the
  experts and records, not the build. Last-use times and the clock are not saved: the policy's clock
  restarts at 0 and every seeded count is capped at 16 (`kSeedCountCap`), so an expert used heavily
  in an old session ranks by recent use soon. Frame count is not part of the identity: a state
  ranks every resident, and a smaller cache loads its best prefix.
- **Load.** In the Program constructor after the frames and tables exist:
  `CacheController::seed(ranked, counts)` sets the counts, then loads the best-ranked keys, up to
  half the frames (`kSeedShare`), into free frames while the policy has room (copy commands on the copy stream); the Program waits for them and
  uploads the table, and logs `expert cache warm start: N of F frames from FILE (X GiB in Y s)` or
  why it started empty.
- **Save.** `Program::shutdown_cleanup` (the Engine's orderly stop) and `Program::maintain` between
  requests (when rounds ran since the last save and 10 minutes passed); never during a round.
- **Budget.** As first built, `decode_budget`'s fill test was `promotions + seeded < frames`, as
  planned above. Measured on a workload unrelated to the saved state it regressed (below), because
  a seeded cache left the fill phase at once and replaced the irrelevant seeded experts only at the
  steady-state rate. Since 2026-10-05 seeded frames do not end the fill phase (`promotions + landed <
  frames`): promotions only follow misses, so a matching workload promotes little either way.
- **Interface.** `EngineOptions::expert_state_path` (empty: off; `infernix_bench` leaves it empty).
  CLI and serve: `--expert-state FILE|off`, default `<artifact>.expert-state`.
- **Tests.** `test_expert_cache.cpp`: seed fills the policy's capacity, skips duplicates, residents
  and out-of-range keys, copies each key into its own frame; the state file round-trips and is
  refused for another identity, key count, a missing or a truncated file, with no `.tmp` left.
  Option parsing in the CLI and serve option tests. Model-level check (cold/warm/warm/cold, ids and
  speed): rig `fn/rigs/conc/s4b.bat`.
- **Measured** (2026-10-05, RTX 5090, recipe B dense8m, INT8 KV, cold CLI code prompt, 400 tokens,
  a fresh process per run, order cold / warm / warm / `off`; the warm runs start from the state the
  cold run of the same prompt saved, so this is the same-workload best case):
  - plain 72.9 / 73.5 → 107.6 / 107.9 tok/s (+47 %), hit rate 69.2 % → 91.2-91.6 %;
  - MTP max 4 `--lm-head-draft` 88.8 / 89.1 → 193.7 / 197.0 tok/s (+119 %), hit rate 61.8 % →
    86.3-86.5 %;
  - the warm start loaded every frame (9,389 plain, 8,672 MTP; 24.2 / 22.3 GiB) in 0.96 / 0.88 s;
    the state file is 133-136 KB; greedy ids equal in all four runs of each mode.
  - Unrelated workload (the story prompt warm-started from the code prompt's state, order off /
    state / state / off) with the first fill rule: plain 72.9/72.6 → 62.7/62.9 tok/s (−13.7 %, hit
    rate 73.7 → 60.7 %), MTP 74.8/74.9 → 62.3/62.7 (−16.5 %, 71.1 → 55.7 %); ids equal. The fill rule
    was changed (Budget, above) and both workloads are re-measured.
  - With the new fill rule (and S4's landing, which uses free frames) the regression on the
    unrelated workload stayed for a full seed, because a full cache leaves landing no free frame.
    Seed share sweep (one run per cell, from the code prompt's state, tok/s): code plain off 78.7,
    share 1.0 105.8, 0.5 97.6, 0.25 90.8; code MTP 129.4 / 178.5 / 168.2 / 156.0; story plain 80.2 /
    76.0 / 81.1 / 80.9; story MTP 95.9 / 77.8 / 92.3 / 96.0. **Decision:** a warm start fills half
    the frames (`kSeedShare` = 0.5, the best-ranked experts): +24 % / +30 % on the matching workload,
    within about 4 % of a cold start on the unrelated one.
  - Not yet measured: the 10-minute save in a server.

**S5. Fork at every width, compacted phase lists (1.5 days).** (1) `forked()` needs only a fork
stream and `staging_slots > 0`. (2) S3's kernel becomes `plan_kernel`, run before every fork (also
without CPU serving): in job order it writes `resident[]`, `staged[]` and counts after `CpuCall`
and resolves each record (resident → frame; staged i → landing frame, else slot i − landed, else
zero-copy from the pinned record, the existing third branch as overflow); `stage_kernel` copies
from the list. (3) `gate_up_kernel`/`down_kernel` list mode (`for (i = blockIdx.y; i < *count;
i += gridDim.y)`, `__syncthreads()` between jobs); forked phases launch `(40, G)` with
`G = min(max_jobs, kPhaseRows[family])`, from 8 (one-column) and 4 (8-column), tuned by S2; serial
and eager paths unchanged. (4) `kMaxPassJobs` (512) still bounds `max_jobs`. Item 1 is needed for
B ≥ 2 speculation and helps plain B ≥ 7 and C = 1 n-gram W ≥ 7; items 2-3 ship only if empty grids
cost ≥ ~5 µs per layer at T = 5 (max_jobs 50) or T = 10, else item 1 ships alone with the
overflow. Bit-neutral: only which CTA computes a job changes. C = 1: if items 2-3 ship, every forked
call's launch shape changes; with item 1, n-gram W ≥ 7 rounds start forking.

**S8. Decode share between prefill chunks (2026-10-04, from Strata #465; 0.5-1 day; after S1 + S2).**

- **Today.** `Scheduler::choose_execution` (`scheduler.h:248-255`) alternates one prefill unit with
  one decode round, and Qwen4Exp ignores `PrefillStepWidth` and the `ExecutionTiming*` argument
  (`program.cpp:41-44`). One call costs ~1.28 s + 1.17 ms per token (§19.3.1 `cost(t)`): ~2.5 s
  at the default chunk 1024 and ~6.1 s at 4096. So a lane decoding beside a long cold prompt gets
  one round per 2.5-6 s.
- **Change.** After each non-last prefill unit, the Scheduler runs decode units until a budget of
  share × that unit's wall time is spent. For Qwen4Exp the wall time is
  `PrefillProgress.timing.submit_host_ns`, which is wall time because `advance_prefill`
  synchronizes after non-last chunks; otherwise an engine clock.
  - No decode follows the last chunk.
  - The prompt keeps its chunk grid, so its arithmetic and E3 are unchanged.
  - Per-model default: Qwen4Exp non-zero (0.5 proposed, Strata's default); Qwen3.5 0, today's 1:1,
    because its prefill units are asynchronous.
  - A serve flag sets the share.
  - Shrinking the chunk instead is no substitute: every call re-pays ~1.28 s of staging.
- **Interactions.** Decode rounds between chunks run in a turnover-like regime, since each chunk's
  16 promotions per layer replace decode-hot experts. If frames are later lent to a prefill arena
  (§19.3.6), these rounds see the lent frames as non-resident; this is bit-neutral here, unlike in
  Strata, which needs `--no-prefill-borrow` for exactness.
- **Effect** (estimate; C ≥ 2 while one lane reads a long cold prompt; aggregate roughly neutral;
  C = 1 unchanged).
  - Chunk 1024: ~1.2 s of decode after each chunk (~75-170 tokens) instead of one round per ~2.5 s.
    The long prompt's TTFT grows ~48 % (79 → ~117 s for 32K).
  - Chunk 4096: ~3 s of decode per chunk; TTFT 49 → ~70 s (+43 %).
- **Optional companion.** A length-aware yield: serve the lane whose remaining prompt is under half
  the current one's, at most twice per request, as Strata's BYIELD does. Plain
  `--prefill-round-robin` would also interleave two long prompts and roughly double the first's
  TTFT. Neither preempts at C = 1.

**Decision point after S5** (campaign steps 1-5). S6 + S7 proceed only if the user wants B ≥ 2
speculation (the open question at the end), S2's ungated runs show ≥ ~3 filler-only experts per
layer per padded round, and the refit model still predicts ≥ 10 % over the gate at C = 2.

**S6. Masked fillers (only with S7; 1-1.5 days).** `ops::moe_route` takes an optional `MoeColumnMask
{ extents /* I32 [B] */, width }`: column b·W + j is valid iff j ≤ extents[b] (column n_b yields the
correction or bonus token); masked columns route −1 with weight 0. `count_kernel`/`scatter_kernel`
skip ids < 0; `combine_kernel` skips them, so a masked y is `bf16(s · shared)`, finite. CPU plan,
stage and expert kernels see only valid entries; `max_jobs` stays the static bound. `Forward::moe`
masks when `batch.verify` (`ForwardVerify::extents`; not the drafter); `verify()` passes the
uploaded `spec_layout_.extents` and stages n-gram rows for n_b + 1 columns, zero-filling fillers.
Route log and taps hold −1 (consumers skip it; `after_round` already skips −1, since it compares ids
as unsigned against E, `expert_residency.cpp:117-118`); filler-only counts become 0, rejected-draft
counts remain, and masked columns give no route data (P2b uses live and rejected columns). Valid
columns stay bit-identical (**inference**, checked by the forward test): at fixed (B, W) every dense
route and tile is unchanged and per-column; experts are placement- and batch-invariant; causal state
Ops read only positions at or before a column of their own row, and fillers come after their row's
valid columns, so they never enter a valid column's or another row's state; filler residuals reach
only the drafter's catch-up K/V at uncommitted positions, rewritten before any read.

**S7. Joint draft lengths with an exact search over W (1-1.5 days)**, `draft_policy.{h,cpp}`, pure
host:

```text
choose_joint(rows b: a_{b,1..Kmax}, limit_b; g_B, c_B; probe flags) -> (W, K_b)
  p_{b,j} = prod_{i<=j} a_{b,i}                      (decreasing in j)
  best = (W = 1, K = 0, R = B)                       (a plain round, cost 1 plain-B round)
  for W = 2 .. 1 + Kmax:
      cost0 = 1 + g_B (W - 1)     padded columns: dense, QSA, GDN record, LM head, catch-up, drafter step
      lambda = best.R
      repeat at most 4 times (Dinkelbach):
          K_b = #{ j <= min(W - 1, limit_b) : p_{b,j} >= lambda c_B }   (a prefix of each row)
          if max_b K_b < W - 1: W is dominated by a smaller one; stop
          R = (B + sum_b sum_{j<=K_b} p_{b,j}) / (cost0 + c_B sum_b K_b);  lambda = R
      keep the best (W, K, R)
  probe: on a row's kProbeInterval-th policy round, K_b = min(K_b + 1, W - 1, limit_b)  (never widens W)
```

- **Cost** follows W through g_B and live drafts through c_B; revision 1's separable rule ignored
  the W term and chose K = (4, 0) on a cold cache (67 tok/s against 78 plain, model). At most 4·Kmax
  passes (Kmax widths, up to 4 Dinkelbach iterations each) of B × Kmax comparisons per round. At
  B = 1 it reduces to `choose_draft_length` with `kWidthCost = g_1 + c_1`; B = 1 keeps calling that
  function, so decisions and probes are equal.
- **Integration** at B > 1: row limits (`budgets`) before drafting, `choose_joint`, draft
  `steps = W − 1`, rows with K_b = 0 join as masked rows. N-gram: `Lane::ngram_accept[16]`, an EWMA
  updated in `verify()`, turns a proposal into an acceptance vector cut by the same rule; a row
  takes the source with the larger p-weighted gain (B = 1 unchanged).
- **Constants.** P2a fits g_B and c_B per B and regime from the S2 table (period against W − 1 and
  ΣK, in plain-B periods), from g_B ≈ 0.54 ms / t_plain(B) (0.19 ms per column + 0.35 ms drafter
  step) and c_B ≈ (0.38 · t_plain(1) − 0.54 ms) / t_plain(B). P2b, only if P2a mis-tunes across
  regimes: a miss-aware c_B updated every round from S2's counters, from the per-live-column cost
  48 (10 c_e + u (m c_miss(M_B) + (1 − m) c_hit)) (not the kernel constant c_col above). Here c_e
  is an expert entry's compute, c_hit a resident new expert's job cost, u the new distinct experts
  per live or rejected column, m their miss share and c_miss(M_B) the marginal miss cost at the
  round's miss level.
- **Output.** Draft lengths change only which positions are verified; greedy output at C > 1 can
  differ from C = 1 at near-ties once B·W > 8.

**Tests** (layer and Op tests bit-exact against the CPU-engine or FP64 oracle):

| Step | Tests |
|---|---|
| S1 | None in code (decision inside `ProgramImpl`); campaign step 1 checks byte identity and that no B ≥ 2 round has W > 1 |
| S2 | `test_offloaded_moe_layer.cu`: counters equal the expected CPU and staged counts (`Config{3, &service_two}` stages misses − 2, serves 2) |
| S3 | `test_offloaded_moe_layer.cu`, `test_offloaded_moe_team.cpp`: 16- and 32-job services; T = 8 and 16 at E = 512, top-k 10 (multi-column CPU jobs); served count = `min(cap, M − ⌊M/div⌋, misses of ≤ max_job_columns columns)` from the S2 counters (the knobs' contract); `max_job_columns = 1` serves one-column jobs only; team test to 32 jobs |
| S4 | `test_expert_cache.cpp`: reserve/adopt/release keep `resident_count() ≤ capacity()` and `free + held + pending = frames`; an adopted key is Ready at its frame; an adopted queued key is dequeued; adopting into a full LFRU evicts exactly one victim. Layer: fork with 3 landing frames and 9 misses; `landed` = the first 3 staged misses in job order; each landing frame memcmp-equals the host record; `landing = nullptr` matches today; after vision V4: lending during the fill phase with reservations outstanding (§19.3.0) |
| S4b | `test_expert_cache.cpp`: save and load round trip (counts, last use, residents, clock); a file with another artifact identity, recipe or frame count is rejected; seeded `resident_count() ≤ capacity()`; the fill-phase test counts seeded frames. Engine (real artifact): greedy ids seeded = empty on code and story |
| S5 | Layer: fork with 3 slots and 9 misses (zero-copy overflow); T = 8 and 16 at E = 512 (max_jobs > 64); CPU flags present (lists exclude CPU jobs); landing with overflow; existing configurations pass, and `Config{3, nullptr, true}`, silently serial today, forks |
| S8 | Scheduler unit test (`src/runtime/engine`): with a share s and a fake prefill unit of t seconds, decode units run until s·t is spent, none after the last chunk; share 0 reproduces today's 1:1 order (Qwen3.5 tests unchanged) |
| S6 | `test_offloaded_moe_cuda.cu`: the mask writes −1, dispatch excludes it. Layer (B = 2, W = 4, extents {3, 0}; B = 3, W = 5): no job for masked-only experts; valid columns exact with and without the mask; masked columns equal the FP64 shared-only formula. `test_forward_real.cpp`, the actual claim: bit-identical logits for every valid column with and without the mask at B = 2, W = 4 (T = 8, SIMT) and W = 5 (T = 10, MMA); masked logits finite |
| S7 | `test_draft_policy.cpp` (host): B = 1 equals `choose_draft_length` over an acceptance grid; B ≤ 4, Kmax ≤ 4 equals brute force over (W, K) on random inputs; K monotone in acceptance; limits respected; probes never widen W |

**Files.** `include/infernix/ops/offloaded_sparse_moe.h` (`MoeExpertSource` counters and landing
fields, `MoeCpuChannel::max_job_columns`, `MoeColumnMask`, documentation);
`src/ops/offloaded_sparse_moe/cuda/moe_layer.cu`; `src/ops/offloaded_sparse_moe/cpu/miss_request.h`
(`kMaxCpuJobs`, `static_assert`) and `miss_service.{h,cpp}`;
`src/models/qwen4_exp/execution/forward.{h,cpp}` (`ForwardExperts`, `ForwardVerify::extents`,
`ForwardTap` documentation); `program/program_impl.h` and `program.{h,cpp}`;
`program/expert_residency.{h,cpp}`; `program/expert_cache/expert_cache.{h,cpp}`; new
`program/draft_policy.{h,cpp}` in `program_sources.cmake`; the tests named above, new
`test_draft_policy.cpp` in `tests.cmake`; new `bench/ops/offloaded_moe_bench.cu` in
`bench/ops/benchmarks.cmake`. S4b: the save/load in `program/expert_cache/` (a new
`expert_cache/warm_state.{h,cpp}`) and its call sites in `program/program_impl.h`; S8:
`src/runtime/engine/scheduler.h`, the per-model default in the Qwen4Exp and Qwen3.5 instances and a
serve option in `src/serve/serve_options.cpp`.

**Deferred and rejected.** P6, a ragged layout of only Σ(1 + n_b) columns (per-row offsets in every
state Op, new graph keys), only if S2 shows filler non-MoE work above ~5 % of a C ≥ 4 round
(~1-3 ms). Rejected: sub-rounds or time-slicing (batched rounds share ~7-9 ms of dense, head and
small kernels; the gate keeps that without padding); a per-B cap table; revision 1's P5, the summed
advance as promotion budget (more link bytes in exactly the link-bound rounds); ≤ 8-column SIMT
splits (an extra dense pass; only relevant if S7 lands); narrower prefill steps beside decoding
lanes (each call re-pays ~1.28 s of staging). The first revision also rejected scheduler changes
as "a TTFT/ITL trade-off outside this plan" with "0.5-1.6 s stalls"; §19.3.1's `cost(t)` gives
~2.5 s per 1024-token call and ~6.1 s per 4096, so the rejection was withdrawn on 2026-10-04 and
S8 adds a decode share; more than 64 staging
slots (staged misses stay well under 64 to C = 8 at W = 5, model; S5's overflow covers the tail);
landing into victim frames outside the fill phase (a later layer's victim may still be read this
round; ~5 ms of link per MTP round is not worth the hazard).

#### Measurement campaign and acceptance

Rules (§19.3.0): GPU lock and the quiet lock (`fn\locks\quiet.lock`: no build during timing runs,
which disturbs the CPU-served experts and the S3 sweep), ≥ 72 GiB free RAM, hidden detached rigs
dry-run-validated before arming, ABBA where the text is identical, one server start per
configuration. Regimes: **fill** (fresh server or CLI process), **turnover** (an unrelated 600-token
request first; it fills the frames within ~200 tokens), **warm** (repeated). Workloads: C = 1
queued; C = 2 code + story and code + code (different prompts); C = 4 two of each. Rig
`fn/conc/serve_conc.ps1` (C, plain / MTP / n-gram, pair, regime); C = 1 CLI cold runs and tg512 as
in §19.2. Metrics: **joint-phase tok/s** (Σ tokens / Σ periods of B = C rounds); **makespan
throughput** (output / (last completion − first submit), like-for-like with C = 1 queued); per
request tok/s, TTFT, acceptance, hits; the S2 table. Output: gated runs must be byte-identical to
each request alone; ungated, a mismatch fails only if the runs' (B, W) sequences match.

| # | Campaign step | GPU |
|---|---|---|
| 1 | S1 + S2 baseline, gate toggled: C = 1, 2 (both pairs), 4; plain, MTP gated and ungated, n-gram ungated at C = 2; all regimes. Yields HEAD's real C = 2 MTP figure, the gate's gain and byte identity, ρ_B, filler-only counts, CPU jobs and columns per call, the refit model. Since 2026-10-04 also: C = 1 queued as the baseline (makespan and per-request TTFT), and a lone request at `--max-concurrency 2` against 1 (the per-lane frame cost; Strata measured −11 % / −24 % per request with 2 / 4 slots on a card where most experts miss) | 1.5-2 h |
| 2 | S2 microbenchmark | ~10 min |
| 3 | S3 sweep; the winner then: tg512 plain and MTP max 4 within noise, cold CLI code and story plain and MTP not slower, C = 2 and 4 gated not slower in any regime | ~2 h |
| 4 | S4 A/B (off/on, budget rule vs unchanged): C = 1 MTP and plain fill, C = 2 and 4 gated fill; turnover and warm as no-change checks | ~45 min |
| 5 | S5 A/B: C = 1 MTP and plain, warm and fill; C = 1 n-gram K = 7 code; C = 8 plain if C ≥ 7 matters; ungated C = 2 MTP | ~45 min |
| 6 | S6 + S7 vs the gate (after the decision point): C = 2 both pairs, C = 4, all regimes, 3 repetitions, per-row K distribution | ~1.5 h |
| 7 | S4b A/B (seeded against empty, new process per run): first 256 and 512 decode tokens of code and story after a restart, with the saved state from the same workload and from an unrelated one; startup time; the replay gate for any shipped profile (CPU) | ~45 min |
| 8 | S8 at C = 2: one lane decoding while the other reads a cold 32K prompt, share {0, 0.25, 0.5} at chunk 1024 and 4096: the decoder's ITL p50/p99 and tokens per chunk gap, the long prompt's TTFT, aggregate tok/s | ~45 min |

Acceptance. **Every step, C = 1:** greedy ids unchanged; tg512 plain and MTP within noise (ABBA);
cold CLI code and story not slower. **S1:** C = 2 and 4 makespan ≥ HEAD's ungated MTP in every
regime; output byte-identical to alone; reported against C = 1 queued, with the lone-request cost
of `--max-concurrency 2`. **S3, S4, S5:** each ≥ its predecessor on every workload
and regime; ids identical where (B, W) sequences match; one cap value. **S4b:** greedy ids seeded =
empty; the first 256 tokens faster beyond noise with the same-workload state and not slower with
an unrelated one; a file from another artifact or frame count rejected. **S8:** output
byte-identical to each request alone; the decoder's p99 gap during a long read below one chunk
call; the share recorded with the long prompt's TTFT cost; C = 1 and Qwen3.5 unchanged. **S6 + S7** replace the gate
only if joint-phase **and** makespan throughput beat it at C = 2 and C = 4 in fill, turnover and
warm, with no regime worse; otherwise neither is merged.

#### Corrections to the review

All eleven findings of the adversarial review of revision 1 were checked against the code and
adopted. Findings 2, 3 and 9 carry a correction of magnitude, attribution or fact (table); 1, 6
and 7 carry the additions in the last row. The rest are adopted as stated: 4 (CPU cost grows with
columns, and more workers do not help: the 52 + 18·(n − 1) µs model and the column sweep), 5 (the
per-entry cost does not fit the 8-column kernels: per-job and per-column terms, refitted by S2), 8
(the single-thread `cpu_plan` selection: S3 item 4), 10 (masked equals unmasked tests and the
hidden `CpuCall` limit: the S6 forward test, S3 item 1) and 11 (byte identity at C > 1 and the
aggregate metric: the gated/ungated output rule and the two throughput metrics).

| Finding | Correction |
|---|---|
| 2: one cap, `kMaxCpuJobs` 32, sweep at C = 1 | The reviewer's C = 1 gain from the cap (89.9 → 112.1) omits promotions and the DRAM floor; with them cap ≥ 12 gives 86 → 95 at 70 GB/s and nothing at 55. The large C = 1 gain needs S4 (111 / 97) |
| 3: double crossing; replace P5 | P5 is replaced by S4, but the plain C = 2 gap is not "the 4.8 ms per round of promotions": as a link floor they do not bind (18 ms against 23-28 ms), as full contention they predict 66 tok/s against 83.7. The per-layer DRAM floor gives 23.3 / 28.9 ms against 27.8 |
| 9: period and counter attribution | `decode_ns` is not "submit only": it includes the forward (the conclusion, one metric, stands) |
| Others | 1: S1 keeps `mtp_draft(steps = 0)` for pending cells. 6: the separable rule scores K = (4, 0) at 67 (no fork) or 74 (fork) against 78 plain; the masked-route-data claim is dropped. 7: serial T = 10 launches 8,000 CTAs, not 10,240. Order: S2 runs with the gate before S3 and S4, since the sweep needs per-call counts and the baseline precedes any change |

#### Risks, decisions and open questions

| Risk | Handling |
|---|---|
| Model error ±30 %, verification ~20 % optimistic, kernel constants assumed | S2 refits them; every step is accepted on measurement only |
| Gate byte identity is an inference | Campaign step 1; a mismatch fails S1 |
| CPU, stage and promotions share host DRAM: S3 gains nothing at C = 1 at 55 GB/s and adds CPU load and power | Staged sweep including C = 1 MTP; the cap rises only if measured faster |
| S4: a discarded round leaves reservations | `before_round` releases them first |
| S4's landing reservations and vision's frame lending (§9.2, §19.3.2) both draw on free frames during the fill phase | Whichever of lending and landing merges second adds a test that lends during the fill phase with landing reservations outstanding (§19.3.0) |
| S7: the model has no MMA-tile term. Dense Q8 at T = 9-16 measured 0.22 TB/s ([2560, 2560], 12-16 columns; ~14-20 ms per forward estimated against ~3.6 ms at T ≤ 8, §19.3.3); C = 4 at W = 5 is T = 20 | Measure S7 after §19.3.3's K1 (T ≤ 16) and its T = 17-64 choice (**inference**) |
| S7: greedy output at C > 1 differs from C = 1 at near-ties | Decision point, open question below |
| S4b: stale seeded counts keep cold experts resident; a mismatched state costs more than an empty start | Counts scaled or capped at load; LFRU adapts from the first round; the replay gate before any shipped profile; campaign step 7 measures the unrelated-state case |
| S8: decode rounds between chunks and the chunk's promotions evict each other's experts | Measured in campaign step 8 (decoder hit rate and the prompt's TTFT per share); the share is a flag |
| Shared code: S3 and prefix P7 both change `cpu_plan_kernel`'s x publication; S3 and Q8 Phase 2 both change `cpu_wait_kernel`'s grid; S6's −1 skip must also cover Q8's fused small-dispatch kernel and `moe_combine` | This track owns plan, wait and stage: S3-S5 land first and Q8 Phase 2 and prefix P7 rebase onto them; one `cpu_wait` grid serves copying and L2 warming; Q8 Phase 0 lands before S6 (§19.3.0) |

| Decided (default, 2026-10-04) | Alternative |
|---|---|
| Optimise C = 2 first, check C = 4 | C = 4, or C ≥ 7, as the primary target (then S5 and campaign step 5 extend to C = 8) |
| The gate is the C > 1 policy; S6 + S7 replace it only if they beat it at C = 2 and C = 4 in all three regimes **and** the user accepts concurrency-dependent greedy output at near-ties (open question below) | Never speculate at B ≥ 2; or split wide verification blocks into ≤ 8-column SIMT launches (one extra dense pass per round) to keep equality with C = 1 |
| A higher CPU cap (12-24 jobs) is accepted if measured faster | Keep 8 to bound CPU load and power |
| Fill-phase landing (S4) accepted: every staged miss is cached while frames are free (bit-neutral; the first ~100-200 tokens after a start) | The LFRU's highest-score picks within the promotion budget |
| S5's compacted grids accepted subject to the C = 1 no-regression gates (bit-exact launch-shape change) | S5 item 1 only |
| CPU cap, divisor and workers stay internal defaults; the gate stays internal (an environment override in a local measurement build) until S7 decides | `--cpu-expert-jobs` and the like; a documented gate option |
| S4b warm start on by default from the user's own saved state; no shipped profile unless the replay gate passes (2026-10-04) | Start empty, as today; or a shipped profile without the gate |
| S8 decode share for Qwen4Exp at C ≥ 2, value from campaign step 8 (0.5 proposed); Qwen3.5 keeps 1:1; a length-aware yield only as an option (2026-10-04) | Keep 1:1 alternation; plain `--prefill-round-robin` as the Qwen4Exp default |

**Open, asked at the decision point after S5:** is concurrency-dependent greedy output at
near-ties acceptable if S7 wins (+20-30 % at C = 2, model)?

Dense Q8 needs no separate concurrency-invariance task (§19.3.3): the gate keeps C ≤ 8 rounds at
T ≤ 8. The prefix cache's C > 1 extras (P8, §19.3.1) follow this track.

#### Two-row speculation (measured and adopted, 2026-10-07)

The gate left two decoding requests with one token each per round: in the agentic replay against
Strata (README) one request decoded 123 tok/s and two together 151. Measured on the RTX 5090
(Gen5 x8), INT8 KV, MTP with the proposal head, AIME long-decode prompts sampled at temperature
1.0, decode rate from the server's 5 s records in which all rows decoded (`fn/conc`):

| Rows | Drafts per row (fixed, one length for both) | Combined tok/s | Acceptance |
|---|---|---:|---:|
| 1 | the row's own policy | 135-139 | 74 % |
| 2 | 0 (the gate) | 143-144 | |
| 2 | 1 | 163 | 64 % |
| 2 | 2 | 160 | 59 % |
| 2 | 3 (8 columns) | 120 | 54 % |
| 4 | 0 | 197 | |
| 4 | 1 | 155 | |

Against a plain two-row round a round with K drafts per row took 1.44 (K = 1) and 1.95 (K = 2)
times as long; K = 3 reached 8 columns, past the 7-column fork and CPU-job limits, and took 3.1.

**Decided:** two-row rounds draft one length for both rows (`choose_pair_draft_length`): the K in
[0, 2] maximizing both rows' expected tokens per round time at `kPairWidthCost = 0.47` per draft,
probing one more every `kProbeInterval` rounds. Rounds of three or more rows stay plain. A pair
round has at most 6 columns, inside the column-invariant dense routes, which answers the open
question above without a compromise: greedy output at C = 2 was byte-identical to C = 1 on two
prompts (3,474 and 12,086 characters). Gate build against the pair policy on the same workload:
C = 1 139.2 / 138.6, C = 2 143.8 / **164.9 (+14.7 %)**.

The C = 4 runs found a coalescing defect (§19.3.1), fixed with this change: a fresh request
waiting for a sibling's in-flight capture at or before the sibling's state split the sibling's
remaining calls at that position, so its next call ended where the lane already was ("the lane's
prefill plan does not continue its state"), failing every running request. It needs a request
that shares a prefix with a lane whose capture there has not landed; four requests over three
prompts sent together (two identical) reproduced it on two of three fresh servers, and none of six
after the fix.

#### Confidence cut for one-row drafts (measured and adopted, 2026-10-07)

The agentic replay against Strata showed Strata's drafts accepted more often (72 % against 62 %
per draft). Both engines verify every draft and accept it with the target's probability, so this
was a matter of which drafts are offered, not of output quality: Strata only offers a draft while
the draft layer's probability of it is at least `--spec-min-p` (0.70 in the configuration
measured), while Infernix chose the length from per-position acceptance alone. The proposal head
was ruled out first (`--lm-head-draft` on and off: 76.1 % and 76.1 %).

The draft head now also writes each draft's log-probability (`ops::top_logprobs` over the head's
rows, K = 1, beside the argmax), and a one-row round verifies its drafts only up to the first below
probability 0.5 (`kDraftMinLogprob`). C = 1, AIME prompts sampled, MTP 4 with the proposal head:

| Variant | tok/s | Acceptance |
|---|---:|---:|
| No cut | 139.5 | 76.1 % |
| Cut 0.5, the cost policy's length | **145.6** | 82.1 % |
| Cut 0.7, the cost policy's length | 142.6 | 86.3 % |
| Cut 0.5, the most drafts (4) | 140.7 | 75.6 % |
| Cut 0.7, the most drafts (Strata's form) | 144.4 | 86.0 % |

A confirmation (no cut / cut 0.5 alternated twice, eight rounds each) reproduced 138.8 and 147.2
tok/s to the decimal (+6.1 %). Greedy output with the cut is byte-identical to without. Pair rounds
keep their length: their columns' n-gram rows are staged at offsets that depend on the width
before the drafts are known.

The production build, which computes the log-probabilities every round, repeated it: C = 1
sampled 147.2 tok/s with 81.8 % accepted. At C = 2 greedy decode ran 162.3 tok/s before and 162.2
after in one session, so the extra head pass costs pair rounds nothing measurable. C = 2 sampled
gave 160.8 tok/s against 164.9 for the pair policy in an earlier session, on different sampled
text; no same-session sampled run separates the two. The prefix-cache, preemption and decide real
tests pass.

### 19.3.6 Strata-derived options (2026-10-04)

Strata was re-read at `6f32ec0` (§3.1, "Strata v0.1.39"). The review ran six area sweeps: expert
cache, prefill, speculation, serving, kernels, and vision/KV/quality. An adversarial verifier
checked each sweep against both code bases, and its "missed" items were re-checked in Strata's
history. The full ranked review is `local/workdirs/fn/plans/strata-review.md`; the per-track
changes with their acceptance measurements are `strata-amendments.md`. Nothing is implemented.
Labels are as in §19.3, and no Strata code is copied (§1.4).

Strata's numbers are its own and measure a mechanism, not a prediction for this card. Our
baselines: tg512 88.7-90.0 plain and 133.5-138.7 MTP; serve warm 147.2 / 97.5; cold first request
70.6 / 61.8; pp4096 413 at chunk 1024 and 659-673 at 4096.

**Adopt, ranked.** Each item names the step it amends; the amended text is in that section.

1. **E2M1 decode in registers in the GPU expert kernels** (§19.3.3 Phase 0).
   - Defect: `canon::e2m1_x2`'s 8-entry table compiles to a stack store plus a local load on every
     lookup. `cuobjdump` counts 688 STL + 688 LDL in `gate_up_kernel<1>` and 160 + 160 in
     `down_kernel<1>`.
   - Strata keeps codebooks in registers; its SYCL port measured an expert call 80.8 → 43.1 µs and
     decode +7.5 % from that change.
   - Estimate: −0.1 to −0.3 ms per plain token and −0.5 to −1.3 ms per W = 4-5 round. Bit-exact,
     0.5 day.
2. **QSA block selection that scales with context.** Not in a track; a proposed "QSA long context"
   item.
   - Today `select_kernel` gives each column one CTA, which scores every block and runs a
     single-histogram radix select. Measured ~19 µs + 32 ns per block per launch.
   - Extrapolation: 1.0 / 3.4 / 12.8 ms per plain token at 8K / 32K / 128K, against an 11.5 ms
     token. That alone would miss M8's ≥ 85 %-of-4K target.
   - Strata on sm_90+ scores on many CTAs and selects with an 8-CTA cluster; `cce52db` adds a
     per-warp-histogram top-k (9-12× at 262K-524K cells).
   - Steps:
     - Q0: nsys at 8K / 32K / 128K (~1 h). Method: one `infernix_bench -r 1 --warmup 0 -pg P,128`
       process per context and mode (plain; `--spec mtp --draft-tokens 4 --lm-head-draft`), dense8m,
       INT8 KV, the serving chunk 4096 and one `--max-ctx 132096` for every context (equal frames).
       Prompts are the first P ids of an untiled corpus of the repository's docs
       (`tools/bench/make_bench_corpus.py --source-text`). `nsys --cuda-graph-trace=node`; the
       measured prompt's selects are the eager 32-column launches; decode selects are grouped per
       graph launch (12 = a main or verify forward of width gridX, fewer = an MTP draft chain).
       The prompt's last chunk is the "prefill chunk at P".
     - Q1: prefill groups of ≥ 170 columns (0.5 day, exact; ~5× on prefill selection).
     - Q2: multi-CTA scoring with today's per-block arithmetic, plus a cluster or two-pass top-k
       (2-3 days). Ids stay bit-exact.
   - Estimate: plain decode +8 % at 8K, +27 % at 32K, 2× at 128K. It lands after prefix P0 and
     extends P0's QSA oracle test.
   - **Q1 + Q2 design (2026-10-04, one step; built, measurements pending).** One selection path,
     `ops::detail::qsa_select` (`src/ops/qsa/qsa_select.{h,cu}`), serves prompt chunks, decode,
     drafts and verify rounds. `select_kernel` is gone from `qsa.cu`; it lives on verbatim as the
     reference in `tests/ops/test_qsa_select.cu` (the prefix branch's P0 test is not on this branch;
     the two QSA tests merge at rebase).
     - Groups of 128 columns share one score scratch (128 × (max_context / R + 1) FP32: 33.6 MB at
       262K, 16.9 MB at 132K), plus 1 MB of global-histogram state. A prompt chunk of 4,096 runs 32
       groups. This replaces the plan's "≥ 170 columns, one CTA per column": the multi-CTA kernels
       fill the SMs at any column count, so the group size only bounds scratch and launches.
     - Scoring, `qsa_score_kernel`: a tile is up to 16 consecutive columns of one row; its CTAs
       split the row's blocks, a warp reads 8 pooled keys once and scores them for every column of
       the tile. Per column the 32 lane partials (8 blocks × 4 head slots, each lane's 4-product
       chain in j order from 0) are reduce-scattered: at each xor offset a lane keeps half its
       values and adds the partner's copy (own + partner). That is exactly the set of additions
       `warp_sum`'s butterfly makes at that lane, so every dot is bitwise the old one, at 31
       shuffles per 32 sums instead of 160. ReLUs join in head order from 0, times `rsqrtf(Di)`.
       Scores, ids and counts are bitwise today's.
     - Selection, exact over the 64-bit order key of `score_id_order.cuh` (ties to the lower id,
       so the 512th key is unique and any split gives the same set): a 2,048-bin histogram of the
       scores' ordered bits [30:20] (exponent and 3 mantissa bits; scores are ≥ +0) finds the bin
       of the top-th score; scores above it are selected, the bin's scores are candidates (up to
       4,096 keys in shared memory, with their AND/OR). An 8-bit radix select over the candidates
       starts at the highest bit in which they differ and stops once a digit's keys are all taken.
       A bin beyond 4,096 keys (heavy ties, e.g. zero scores at the threshold) runs the same radix
       select over the column's scores instead. Ids are written in ascending block order.
     - Selection kernel (linear_topk's split-and-merge): up to 8 CTAs per column add their slice
       histograms into a global one; the last to arrive finds the bin, filters the column into a
       bitmap and the candidate buffer, selects and compacts. The score kernel zeroes the
       histograms and arrival counters each call, so stale arena bytes and graph replays are safe.
     - Q2 decision (2026-10-05): a second variant, one 8-CTA thread-block cluster per column
       (histograms reduced through distributed shared memory, candidates gathered in rank 0, no
       global state), was built and measured against this one behind a temporary toggle, then
       deleted. nsys, per layer call (8K / 32K / 128K, plain): decode select 12.3 / 13.8 / 23.7 µs
       global against 16.1 / 17.2 / 23.8 µs cluster; prompt select 0.20 / 0.45 / 1.74 against
       0.54 / 0.82 / 2.15 µs per prompt token. infernix_bench decode (pg N,256, int8 KV, two runs
       each): 4K 90.23 / 89.86 tok/s, 8K 90.87 / 90.29, 32K 80.64 / 80.48. Greedy ids of both
       equal the Q0 kernel's (8K and 32K, plain and MTP, 256 tokens). Against Q0 the decode rate
       rises 4.9 / 10.6 / 34.6 % at 4K / 8K / 32K and prefill 0.5 / 1.2 / 3.9 %; 32K decode is
       89 % of 4K decode (M8 ≥ 85 %; Q0 70 %). 128K was not benchmarked end to end.
     - Launches: the score kernel is a programmatic dependent (streaming: it triggers after its
       loop), the select kernel enters with `pdl::enter()`; 2 launches per layer call and group.
     - Expected (estimates, before measurement), per layer call at W = 1: pooled keys
       N × 256 B (0.5 / 2.1 / 8.4 / 16.8 MB at 8K / 32K / 128K / 262K, floor 0.3 / 1.3 / 4.9 /
       9.9 µs at 1.7 TB/s); score kernel ~2.5 / 3 / 6.5 / 11 µs, select ~5 / 5 / 6 / 8 µs (latency
       of barriers and the candidate radix), plus ~1-2 µs of launch gap: ~8 / 9 / 13 / 20 µs
       against 107 / 393 / 1,559 µs measured in Q0, ~0.1 / 0.11 / 0.16 ms per plain token. Verify
       widths up to 4 cost about the same (keys are read once). Prompt chunk of 4,096: compute-bound
       scoring (~2.2× the FP32 FMA floor of 512 FMAs per block and column), ~0.35 ms per layer at
       8K and ~3 ms at 128K (Q0 at chunk 1024: 2.75 / 40.8 ms per 1,024 columns); a 128K prompt's
       selection ~0.6 s instead of 31.6 s.
   - **Q3: prompt rows on the FP32 pipes (2026-10-06, item 2).** The 128K profile put QSA
     scoring at 73 ms per late 4,096-token call (13 layers), about five times its FP32 FMA
     floor: the reduce-scatter's shuffles and selects cost more issue slots than the FMAs.
     - `qsa_score_wide_kernel` serves rows of at least 32 columns. A CTA scores a 16-column row
       tile against key tiles of 64 blocks staged as FP32 in shared memory (rows of 32 float4
       lanes plus one pad, conflict-free; 99 KiB per block on sm_120 leaves room for one key
       buffer, so the next tile travels in registers). A thread owns one column's four head dots
       against four blocks: it forms each dot's 32 lane partials itself (the j chain from 0, as
       lane l would) and adds them in the butterfly's tree, leaves in bit-reversed lane order so
       that a six-entry stack per dot holds every open subtree. FP32 addition is commutative, so
       every sum is bitwise the butterfly's. 188 registers, no local memory.
     - Bits: `infernix_qsa_select_test` compares scores and ids bit for bit with the original
       kernel on wide cases (partial tiles, two rows, zero-heavy and tied keys, 1 and 3 index
       heads, ratio 8, a 4,096-column chunk at 27K blocks); all pass. The ~40K prompt's greedy
       ids are identical with and without it.
     - Speed, one 4,096-column selection (score and select): 1.119 → 1.006 ms at 8K (1.11x),
       2.39 → 2.08 ms at 32K (1.15x), 6.50 → 5.35 ms at 110K (1.22x), 14.04 → 11.44 ms at 250K
       (1.23x). pp131072 (ABBA): 6,935 / 6,948 → 7,020 / 7,012 tok/s (+1.1 %).
     - The gain is a third of what the scoring alone predicted: the select kernel (a histogram
       pass and a filter pass over the scores, ~1.5 µs per prompt token at 110K) now dominates.
       Fusing the bin histogram into the score kernel would save one pass, but per-column
       histograms (16 × 2,048 bins) do not fit beside the staged tiles; not done.
3. **Warm start** (§19.3.5 S4b).
   - Save the LFRU state; at load, seed it and bulk-fill the frames (26.1 GB, ~0.95 s).
     `LfruPolicy::seed` has no caller today.
   - Estimate, first ~200-500 tokens after a restart: up to +37 % plain and +63 % MTP with the
     user's own state; ~0-24 % with an unrelated one. Bit-neutral, 1-1.5 days.
   - A shipped profile only behind a replay gate.
4. **Prefill track.** Not in a track; a proposed "prefill" item (M8), steps F0-F6.
   - F1, the A4 wide route (§13): gate_up + down from 3.3 s to ≤ 0.3 s per 4K chunk.
   - F2, a copy-engine streamed ring of frames across layers (§13 streamed walk). One event per
     layer or group, never per copy: Strata measured ~10 µs of WDDM idle per wait/record pair.
   - F3, frames lent to the prefill arena (§9.2, after vision V4).
   - Estimate: pp4096 413-673 → ~2,100-2,400 tok/s at the x8 link floor (~1.63 s per 4K chunk),
     and 32K prompts ~3× faster. ~12-17 days.
   - Rules:
     - The chunk is fixed per Engine from configuration and prompt length, never from free frames
       or loans. Chunk size changes bits, and E3 holds only on a fixed grid.
     - io is enqueued before any ring DMA of its chunk.
     - The walk is gated on the DMA-able share of non-resident experts.
     - At merge, §13's "⌈n/max⌉ equal chunks" with a lending-derived max is replaced by a chunk
       fixed per Engine (ring depth first, then the largest chunk).
   - F1 changes prefill bits once and is qualified against the FP64 oracle. Ring and lending are
     placement changes and stay bit-exact.
5. **Decode share between prefill chunks** (§19.3.5 S8).
   - At C ≥ 2, decoding lanes run for share × each chunk's wall time instead of one round per
     2.5-6 s chunk call (Strata's #465 default share is 0.5).
   - Estimate: ~1.2-3 s of decode per chunk; the long prompt's TTFT +43-48 %. 0.5-1 day.
   - It corrects the earlier "0.5-1.6 s stalls" rejection.
6. **Shared expert on its own graph branch from the mixer output** (§19.3.3 Phase 2, measured in
   M4). Estimate −0.1 to −0.25 ms per T = 1 forward; bit-exact.
7. **GDN control GEMV on more SMs** (§19.3.3 Phase 0, moved from "listed"). Estimate −0.04 to
   −0.05 ms at T = 1 and −0.12 ms at W = 4-5; bit-exact.
8. **The next prefill chunk's PLE rows read during the current chunk** (§19.3.4 S4d, reshaped from
   the within-chunk gate and scheduled after S1). Strata CUDA measured +3.7 % at chunk 4096.
   Estimate 2-4 % per cold 4K chunk.
9. **QSA KV host tier** (§9.6, not in a track).
   - Estimate: ~+4-5 % decode on every request at `--max-context 262144` as built (~1,139 frames
     returned); 0 at the guide's 16K.
   - Worth building only for long max-context serving. Its pinned memory is allocated before the
     frames: Strata #620 found that WDDM refused a `cudaMalloc` after ~45 GiB of host
     registration.
10. **QSA decode attention: register accumulators, batched V loads** (§19.3.3 Phase 0, low
    priority). Estimate −0.04 to −0.07 ms at T = 1; bit-exact.
11. **Measurement amendments.**
    - A third S0b flush site at commit (§19.3.4 S0).
    - A temporary draft-probability log for the replays of options 2 and 3 below (§19.3.4 S0).
    - C = 1 queued and lone-request baselines for S1, and per-lane cost in the docs (§19.3.5 S1,
      campaign step 1, docs).
    - A restore-speed criterion with interference held fixed (§19.3.1 M2 (c)).
    - A long-prompt chunk check against a rounding-order-only control (§19.3.1 M9). Strata
      `f23ea57`: a chunk-only change moved KL by 0.023-0.038.

Already planned and confirmed by Strata, no change:
- n-gram S2: Strata gathers rows after layer 0 starts.
- Concurrency S1: Strata switches between solo MTP and plain batch rows.
- Prefix taps: Strata's ~118 MB turn checkpoint matches our 115.7 MB image.
- Vision offload: Strata's resident encoder with a static reserve stalls on 12 GB cards.
- Batch output equal to solo: Strata needs four flags for this, we need none.

**Options, ranked** (gains estimated; each with the measurement that decides it):

1. **KV loans** (§9.2). Up to +4-6 % at 262K max-context while the context is short. Mostly
   superseded by item 9 above. 3-5 days.
2. **Confidence-gated verify width.** Strata `--spec-min-p 0.5`; for us an on-device column mask
   (S6), so S3 keeps no host decision. 0-5 % on prose. Built only if the S0 log's replay shows
   ≥ 3 % net of draft cost: Strata never published a CUDA A/B.
3. **Sampled drafts for T > 0** through the existing sparse rejection Op. The sign is unknown, −2 to
   +8 % on sampled prose: Σ min(p, q) can fall below p(argmax q). Built only if the replay shows
   ≥ ~5 acceptance points.
4. **MTP-or-lookup choice at C = 1** (S7's `ngram_accept` at B = 1).
   - A copy proposal only when its first token equals the first MTP draft.
   - Copy windows capped at W ≤ 6.
   - +0-5 % on edit sessions; lets n-gram default on.
5. **Length-aware prefill yield at C ≥ 2** (with S8). Never at C = 1, which would be a product
   change.
6. **No promotions between prefill chunks.** Up to ~77 ms of link per chunk once F2 makes the link
   the bound.
7. **Prefill io off the copy-engine FIFO** (`upload_pinned`). 0-77 ms per non-last chunk; one
   nsys first.
8. **Ring before chunk, chosen once per Engine** (#583's rule). A few %.
9. **Expert-kernel structure after item 1.**
   - ncu first, then a cp.async ring, PDL for `down`'s weights, and parallel `h` quantization.
   - Plus a carveout co-residency check.
   - −0.3 to −0.8 ms at T = 1, low confidence.
10. **One routing launch** (Phase 0 dispatch merged with S3/S5's `plan_kernel`). −0.05 to −0.15 ms.
11. **Large pages for the pinned bank.**
    - ≤ ~1-2 %, unmeasured in both engines. A locked-memory `cpu_bench` A/B first.
    - Allocation order under WDDM (Strata #620).
12. **Lane affinity for resumed turns** (prefix P8). ~4.5 ms per resumed turn at C ≥ 2.
13. **Miss-service diagnostics.** Log the exception that `miss_service.cpp:141-145` swallows; opt-in
    breadcrumbs.
14. **BF16 KV recommended at ≤ 32K max-context.** −0.6 % decode for upstream KV precision; needs a
    measured quality gain.
15. **`effort_position: end`.** Opt-in, after P6, only if the request log shows effort switches,
    and with an effort-following quality check.
16. **UD-IQ4_XS as a second Strata baseline** in M10.
17. **Elastic VRAM through VMM segments.** A capability without a product requirement.

Unchanged by Strata:
- n-gram S4a (mailbox), kept parked on its trigger.
- Prefix P7 (CPU assist): Strata still has none.
- Q8 Phase 1b. Strata's two-kernel HC read measured −4 to −6 % on an Intel B70, but its
  column-sliced GR down measured +7.8 % on the same card, so M1 decides the form.

**Skip.**

| Item | Reason |
|---|---|
| Zero-doorbell verify graph | We have it in general form: hits never involve the host |
| CPU/PCIe split, adaptive tier | Have the split; the adaptive tier measured 5.5 points of hit rate below LFRU |
| Hybrid-CPU worker rule | Our measured optimum and dynamic handout (6/12/16/24 workers: 62.25/62.01/61.51/58.89). An optional EcoQoS check for hidden launches |
| Router lookahead | No file tier; recall@10 65 % (§19.2) |
| Hit-rate PCIe share, window profiler | Our hit rate never excluded PCIe misses; S2 covers the profiler |
| Mmapped arena, multi-GPU helpers | Breaks clean frames; one GPU |
| BF16X2 activations, finite q8_1 scales | Not our numerics (BF16 activations; saturating A4 encoders) |
| Draft vocabulary per language | 131K proposal rows: 99.09 % held-out coverage, worst domain 97.53 %; full-head fallback |
| Window routing, batched PLE, one-graph draft chain, GDN commit split, sub-warp packing | Have |
| GPU-side commit/draft overlap | ~0 until the next draft is enqueued before `settle_round` |
| Dropping the drafter catch-up on stale K/V | ≤ ~1 % of a B ≥ 2 round; relaxing MR5/MR6 breaks E1 for sampled requests |
| 256-repeat stop, `/metrics` | No loop observed; `max_tokens` defaults to 8192; it truncates legitimate output and changes an external contract. The request log covers metrics |
| Q8 vision weights, lower vision token cap | Below upstream precision; we have `--vision-max-merged` |
| BF16 rounding of q·s in the Q8 MMA route | ≪ 0.001 nats for recipe B (≤ 2^-9 relative per weight) |
| UD per-layer precision exceptions | They are routed-expert formats; ours are NVIDIA's NVFP4, imported exactly |

**Changes at merge time** (on the base branch, §19.3.0):
- §9.3 item 4 (S4b built).
- §9.2 status (prefill loans, once the prefill item is scheduled).
- §13 "Chunks" (the per-Engine rule) and "PLE rows" (S4d).
- §19.2's "Next" list and "not in a track yet" line.
- The user guide's C > 1 and context advice.

### 19.3.7 Memory: VRAM sizing with overflow protection, and an SSD expert tier (2026-10-04)

The user's two memory requirements of 2026-10-04, designed read-only against `1dae6914c` and
revised after two adversarial reviews (rev 2). The full plan, its code check and the review
dispositions are `local/workdirs/fn/plans/memory-tiers.md` (§ numbers below without a prefix are
that plan's); the investigation notes are `mem-vram-notes.md`, `mem-host-notes.md` and
`mem-ssd-notes.md` there. Step, test, measurement and acceptance names are local to this track:
R0-R14, RT1-RT15, RM0-RM10, A1-A5. Labels as in §19.3 (measured, model, estimated), plus
**[K]** for external knowledge a named probe settles. Branch `claude/fn-memory`; status and
measured results are under "Status" at the end of this section.

**Requirements.**

- **U1.** "Make sure the VRAM sizing logic assesses the used VRAM at startup in case a user has a
  monitor plugged in so some VRAM is reserved for display functionality etc. to make sure that the
  engine doesn't overflow available VRAM when running."
- **U2.** "Add another tier to the expert cache to page the least frequently used experts to SSD if
  there isn't enough system RAM free. Assess available RAM on startup leaving an amount (e.g.
  2 GiB) extra free for the system when working out how much free RAM there is for holding
  experts."
- **U3** (later the same day). "One option to consider in your design of the 3 tier expert system
  is whether the experts that sit on the SSD should just be read from the source artifact when
  needed rather than creating a new separate cache that could result in some significant extra
  SSD writes that might want to be avoided to reduce wear on the SSD. But the most important aspect
  is speed." **Binding rule:** the SSD tier is the source artifact, read in place. No step adds an
  SSD cache, spill file, swap file or any write path for expert data; evicting an expert from RAM
  is a discard. The only writes this track makes are the small expert-state file (kilobytes to a
  few MB per shutdown) and logs. Within that rule the fastest design wins: R14's mirror is an
  opt-in, one-time copy the user makes to a second drive, never created automatically, and avoiding
  Windows page-file writes (on `G:` here) is one reason for the RAM reserve and the hard minimum
  working set (RM7 measures paging).
- Standing rules: no precision change; greedy output independent of placement; every trade-off
  documented. Feature 2 changes nothing in full mode (same kernels, load path, pinned bytes and
  speed). Feature 1 deliberately gives up some frames for display headroom on every machine; that
  cost is the trade-off U1 asks for and has its own acceptance arm (A3a).

#### Answer

- **Feature 1, VRAM.** Before any weight is read, the device is queried: device-wide free and total
  VRAM, on Windows the OS budget and this process's usage (DXGI, adapter matched by LUID), and
  whether a display is attached (DXGI outputs, NVML as a second signal; unknown counts as attached
  on Windows). One sizing function, used at startup and at runtime, gives the expert frames; the
  fixed allocations are checked before the 64 GiB load, every device allocation is checked for a
  silent spill into system memory, and one VRAM ledger line is logged. At runtime the frame pool
  sits on CUDA virtual memory (64 MiB chunks under one fixed virtual address, so kernels and
  graphs are unchanged); a monitor thread shrinks it at the next round boundary (or at once when
  idle) when device-wide free VRAM falls below half the display headroom, and regrows it after 30 s
  of slack. No frame floor: Infernix never keeps VRAM the display needs.
- **Feature 2, SSD tier.** A startup RAM ledger computes the RAM for routed experts as available
  RAM − a reserve (`--ram-headroom-mib`, default 2 GiB, replacing the loader's fixed 8 GiB for every
  pinned allocation) − every other planned host allocation. **Full mode** (all 24,576 experts fit
  with ≥ 1 GiB margin) is today's path. **Tier mode** has three levels, VRAM frames ← pinned host
  slots ← the artifact on SSD read in place with unbuffered I/O (U3: no SSD cache and no writes;
  eviction from RAM is a discard). Residency is exclusive under pressure, the RAM tier is ranked by decayed LFU (the
  user's "least frequently used"; the half-life, or LRU/CLOCK on a ≥ 5 % win, is chosen by replay),
  and RAM and VRAM are pre-filled from the saved expert state shared with concurrency S4b. The
  pre-fill is exclusive too (2026-10-08): RAM is read in the order saved ranking, then the saved
  use counts, then the unused keys interleaved across layers; the frames' warm start loads its
  seeds from RAM, whose copies are then released and refilled with the next keys of that order.
  The tier's decayed LFU is seeded with the saved counts (capped at 16), and the demotion list is
  refilled at every boundary. Before, the seeds stayed in RAM as shadows (10.9 of 36.8 GiB on
  2026-10-07), the first victims were the best-ranked pre-filled keys (equal zero scores, lowest
  slot first), and the never-refilled demotion list made every VRAM victim without a host copy an
  SSD drop after the first 32 demotions.
  SSD-only experts in decode and verification calls become CPU jobs, so their reads overlap all of
  the layer's GPU work; prefill and overflow use a device fetch channel. A failed read fails only
  the affected requests.
- **Expected decode speed** (estimated, several-fold uncertain; RM3 replaces it):

  | Machine | Plain decode, central (range) | MTP, central (range) |
  |---|---|---|
  | 96 GB (this machine) | full mode, today's speed | today's |
  | 64 GB | ~0 SSD reads per token once warm: ≈ full mode | ≈ full mode |
  | 48 GB | −5 % (−2 to −20 %) | −9 % (−3 to −30 %) |
  | 32 GB | −22 % (−12 to −40 %) | −34 % (−20 to −53 %) |

  A 4 GiB prefix Host tier costs a further 3-8 points at 32-48 GB. pp4096 at chunk 1024 (today
  9.92 s): ≈ 10-12 s at 48 GB and ≈ 11-22 s at 32 GB (estimated).
- **Cost of Feature 1 on every machine** (estimated from the measured −1.8 % tg512 per 436 frames,
  §19.2): −49 frames headless and −146 with a display when the elastic pool works (−0.2 / −0.6 %
  tg512 at C = 1); −340 frames (−1.3 %) with a display in warning-only mode.
- **Effort.** R0-R11 and R13 ≈ 23 engineer-days plus ~8 h of GPU time; R12 (+2 d) and R14 (+1.5 d)
  only on their measured triggers.

#### Feature 1: startup sizing, flags and the elastic frame pool

- **Core primitive** `src/core/vram_budget.{h,cpp}`: a `VramSnapshot` (device free and total,
  DXGI LOCAL budget and usage, display state, output count, D3DKMT `Demoted`) from a
  `VramBudgetSource` per device context. Windows: `IDXGIFactory4::EnumAdapterByLuid`,
  `QueryVideoMemoryInfo`, `EnumOutputs`, the budget-change event and an `IsCurrent()` refresh for
  hot-plug; NVML through `LoadLibrary`/`dlopen` only, so a missing DLL never stops startup. A
  process-wide test seam supplies fake budgets (RT7, RT9, RT13).
- **One sizing function** (§3.2): `frame_bytes(F, P, B) = min(F + P − H_d, B + P − 64 MiB) − R`, with
  F device-wide free VRAM, P bytes the pool maps now, B the OS budget slack (`+∞` without a budget
  source), H_d the display headroom and R the internal reserve still unspent. The display headroom
  applies to device free only, so a budget below free is not stacked on it. R_total =
  max(256 MiB, 128 MiB + g × min(graph bound, 32)) for the elastic pool (no cap in warning-only
  mode), g = 4 MiB per graph executable until RM0d calibrates it; R_used tracks Infernix's growth after
  sizing, so a runtime grow never spends the reserve.
- **Fail fast.** `construct_qwen4_exp` evaluates the function with the dense arena and
  `ProgramImpl::device_plan(options).fixed_bytes` (the single place that computes the constructor's
  sizes) before `plan_load` reads 64 GiB; a negative result fails with the ledger and the largest
  contributors, fewer than 2,048 frames warns. Today's silent `frames = 0` goes away.
- **Spill guard.** Every startup device allocation step (dense arena, each `device_plan`
  allocation, each frame chunk) must lower device free by at least its size − 32 MiB (and raise
  LOCAL usage, if RM0a shows it counts CUDA). A spill fails startup with the ledger and the
  optional "Prefer No Sysmem Fallback" advice; during a grow it unmaps the chunk and stops growing.
  Every startup pinned host allocation (embedding, tier slots, n-gram cache, prefix Host tier,
  vision tower) happens before the final frame sizing (WDDM late pinning, Strata #620).
- **Flags.** `--vram-headroom-mib N|auto` (existing name; `auto` is new and the default) for
  Qwen4Exp: 512 MiB with a display and the elastic pool, 1,024 MiB with a display in warning-only
  mode, 256 MiB headless. Qwen3.5 keeps its meaning (`auto` = 1 GiB); the field moves from
  `KvCapacityPolicy` to `EngineOptions::vram_headroom` and "requires `--kv-capacity auto`" moves
  into Qwen3.5's validation. Qwen4Exp's `--kv-capacity auto` is defined as `max_context × C`.
- **VMM frame pool** `src/core/vmm_arena.{h,cu}` (§3.4): one virtual range of the card's size,
  64 MiB chunks mapped and unmapped at the top, helpers shared with `evictable_weight_pool.cu`.
  A shrink by k frames runs in two phases: at boundary b the policy evicts the k lowest-score
  residents anywhere (dropped with a host copy, demoted otherwise); at the first boundary where
  every demotion D2H has completed, surviving top-region experts are relocated device to device
  (~3 µs each), one table upload publishes them, and only after host-observed completion are the
  top chunks unmapped and released. Grow maps ≤ 4 chunks per boundary during rounds, all when idle,
  each with the spill guard. The capacity API (`LfruPolicy::set_capacity`, `FramePool` unbacked
  frames, `CacheController::resize`/`relocate`) is shared with vision V4, built by whichever lands
  first. Without VMM the pool stays one `cudaMalloc` and the monitor only warns.
- **Monitor** `program/vram_monitor.{h,cpp}` (§3.5): wakes on the DXGI budget event or 1 s; shrinks
  if F < H_d/2, B < 0 or `Demoted` rose; regrows when the target exceeds the pool by ≥ 2 chunks for
  30 s. Targets are applied at the expert-quiescent round boundary, or through a new idle
  `maintain()` hook in the engine worker's idle wait. Display hot-plug changes H_d. Below 25 % of
  the startup frames it warns once per episode. Linux applies the law only with a display; TCC and
  headless Linux need none.
- **WDDM behaviour** (§3.6): under the default sysmem-fallback policy an over-committed allocation
  silently lands in shared system memory; the pre-check and the spill guard turn that into a
  startup error, and a runtime grow stops at the spill. Pressure from other processes is met by
  shrinking before VidMm contention (device free is the primary signal; budget and `Demoted` are
  secondary). Infernix never changes driver settings; the docs mention "Prefer No Sysmem Fallback" as
  optional.

#### Feature 2: the SSD expert tier

- **Regimes** (§4.1), with H host slots, H_res = H − 176 (128 ring, 16 prefetch, 32 demotion
  slots) and V_min the smallest runtime frame count: **full** (expert RAM ≥ 64,801 MiB + 1 GiB),
  **tier without steady-state SSD** (V_min + H_res ≥ 24,576), **tier with SSD**, and **refuse**
  below H_min = 1,024 slots (2.64 GiB).
- **RAM ledger** `models/qwen4_exp/memory_plan.{h,cpp}` (§4.2): a pure, unit-tested function over a
  Core host-memory snapshot (Windows `ullAvailPhys`, commit, `PrivateUsage`; Linux `MemAvailable`
  under the cgroup limit). Planned: embedding 1.18 GiB, vision tower and media reserve, the prefix
  Host tier, n-gram row cache 0.16 GiB, Program buffers, load staging 0.26 GiB, 1.0 GiB of process
  growth until RM7 measures it, and 0.25 % lock overhead. `--expert-ram-mib N` caps the result (to
  keep RAM for other programs or simulate a smaller machine). The commit limit is honoured, pinning
  re-checks available RAM before each 1.32 GiB chunk, a failed full-mode pin falls back to tier mode
  before any read, and a hard minimum working set keeps Infernix's own pageable pages resident.
  Explicit and default sizes of the prefix tier and vision are honoured first (Q6).
- **Residency** (§4.3): exclusive with shadows (a VRAM resident keeps its RAM copy only as a shadow;
  shadows are the first RAM victims). Slots carry pins (`H2D`, `Queue`, `D2H`, `Use`, `Prefetch`)
  and load serials, so a slot is never rewritten while read and never publishes stale bytes; the
  12 transitions T1-T12 are each an RT2 case. A VRAM victim without a host copy is demoted from a
  published frame (frame held, D2H on its own stream) when it outranks the RAM victim; when the
  boundary's demotion allowance (32 per decode boundary, unlimited at prefill chunk boundaries and
  shrinks) is spent, VRAM admissions wait instead of dropping it.
- **Storage** (§4.4): `artifact::Residency::Streamed` and a long-lived `StreamSource` (per-part
  direct, deny-write handles); `ExpertStore` (Model data) holds each record's ≤ 2 aligned file
  segments (two records straddle part files: L22 E305, L45 E103). Core `DirectReadQueue` (IOCP on
  Windows, an `O_DIRECT` `pread` pool on POSIX) with demand-before-prefetch priority, sub-reads,
  transient-error backoff, `cancel_all` and a fault seam. Measured so far on `E:` (990 PRO, Python
  probe): 0.566 ms p50 at QD 1, 5.8 GB/s (~2,100 experts/s) from QD 4.
- **Op contract** (§4.5): `MoeExpertSource` gains a per-layer `host_table` of record pointers; null
  (full mode) keeps today's `host_records + e × stride` code, a null entry means SSD-only. The CPU
  request carries record pointers when tiered.
- **Demand path** (§4.6). The plan kernel classifies resident / RAM miss / SSD-only jobs. In decode
  and verification, SSD-only experts are chosen first as CPU jobs: the CPU service asks a
  `RecordProvider` for the read, computes from the landed slot, and answers with a status word on
  failure. Prefill, overflow and CPU-off calls publish their SSD-only experts to a mapped
  `MoeFetchChannel`; a host agent reads them into a 128-slot landing ring in job order, one device
  poller mirrors `landed` into L2, the stage kernel copies each record as it lands, and SSD-only jobs
  run last. Waits are bounded by a host heartbeat (1 s without progress), never a trap; the existing
  2 s `cpu_wait` trap becomes the same wait, and both channel sequences become wrap-safe. Exposed
  cost per SSD read ≈ 0.6-0.7 ms (estimated).
- **RAM policy** (§4.7): decayed LFU, score `Σ w · 2^−(t − t_use)/h`, kept as the time-invariant key
  `v = log2 s + t/h` (O(1) per use) with lazy min-heaps for victims. Clock: +1 per decode token
  (verification: + accepted tokens, live columns only); a prefill chunk of T tokens advances
  min(T, 16) with weight Δ/T per use. Default h = 64 until RM0f; LRU or CLOCK replaces it only with
  ≥ 5 % fewer SSD reads (Q13). Demand landings are admitted if they outrank the RAM victim, prefetches
  at 1.25 × its score.
- **Warm start and prefetch** (§4.9): S4b's expert state file gains a RAM-tier section; startup
  pre-fills VRAM then RAM in file order with coalesced reads; a shipped profile only behind the
  replay gate. Background prefetch: one 256 KiB sub-read in flight during rounds, QD 4 when idle,
  prompt-informed after prefill.
- **Prefill** (§4.10): route-aware fetch with SSD-last jobs, and a prompt landing reserve (up to
  min(1,024, H_res/8) slots taken per chunk) so later chunks and the first decode tokens find the
  prompt's cold experts in RAM. In tier mode with SSD reads the log recommends
  `--prefill-chunk 4096`.
- **Failures** (§4.12): the Program checks the tier's error word after each synchronize and throws
  `runtime::RecoverableExecutionError`, which the engine worker handles like an OOM recovery (fails
  the active requests, keeps serving); a read is retried once, a key that fails twice is unreadable
  for 60 s. Every copy stays clean with respect to the artifact (design §9.1's "the host bank holds
  an identical copy" becomes this invariant).
- **Diagnostics**: `report_cache` and the request log gain SSD reads, stall ms, host hits,
  demotions and prefetches; startup prints the RAM and VRAM ledgers; an internal route trace
  (`ProgramOptions::route_trace`, R0) records every round's routes for the replay tool.

#### Flags

| Flag | Default | Meaning |
|---|---|---|
| `--vram-headroom-mib N\|auto` | auto | Device VRAM left free for the display and other programs (Qwen4Exp); Qwen3.5 unchanged |
| `--ram-headroom-mib N` | 2048 | Physical RAM left available after every pinned allocation, all models; replaces the fixed 8 GiB |
| `--expert-ram-mib N\|auto` | auto | RAM for routed experts; N caps it (N ≥ 65,825 gives full mode if the ledger allows; more than the ledger allows is an error) |
| `--expert-state FILE\|off` | S4b's path next to the artifact | Saved expert state (VRAM residents; in tier mode also the RAM ranking) |
| `--expert-mirror DIR` | none | R14 only: a second copy of the parts for split reads, made once by the user (U3: Infernix never writes it) |

All are parsed by `infernix`, `infernix-serve` and `infernix_bench` and carried in `EngineOptions`.
Internal `ProgramOptions` for tests and A/B only: `vram_monitor`, `tier_exclusive`,
`tier_cpu_ssd_jobs`, `tier_prefetch`, `tier_reserve`, `tier_lookahead`, `route_trace` and the fault,
budget and sequence seams.

#### Steps

| Step | Content | Tests | Effort |
|---|---|---|---|
| **R0** | Probes RM0a-RM0e and RM0g (`tools/flash_next_probe/vram_probe.cpp`, `expert_read_probe.cpp`); the internal route-trace hook and a long engine-captured trace; the two-level replay `tools/expert_cache_replay/host_tier.py` | Probe logs; ids identical with the hook on | 1 d + ~2 h GPU |
| R1 | `vram_budget`, `device_plan`, the sizing function, pre-check, spill guard, ledger, `--vram-headroom-mib N\|auto` | RT7; smoke at auto/0/2048 | 1.5 d |
| R2 | `vmm_arena`, capacity API, two-phase resize | RT8 | 2 d |
| R3 | Runtime monitor, `maintain()` hook, warning-only fallback | RT9, RT13 | 1.5 d |
| R4 | Host-memory snapshot, `memory_plan`, shared reserve with per-chunk re-checks, flags | RT1, RT14 (part) | 1.25 d |
| R5 | Op contract (`host_table`, CPU record pointers), heartbeat wait, wrap-safe sequence | RT5, RT6 (CPU); RM1 | 1.25 d |
| R6 | Streamed residency, `StreamSource`, `ExpertStore` | RT3, RT4 | 1.5 d |
| R7 | `DirectReadQueue` | RT3b | 1 d |
| R8 | CUDA-free RAM-tier controller and its Python reference | RT2 | 2.5 d |
| R9 | Fetch channel, SSD-first CPU jobs, SSD-last permutation (with concurrency S5 item 1 if not landed) | RT6 | 2.5 d |
| R10 | Program integration: slot pool, agent, boundary protocol, demotions, state file, recoverable failures | RT10-RT12, RT15 | 4 d |
| R11 | Prefill landing reserve, prefetch modes, diagnostics | RT10 ext.; RM3/RM4 | 2 d |
| R12 | Router lookahead, only if RM3 stalls ≥ 3 % at 29,000 MiB and cold-tail recall ≥ 0.5 | RM9 | 2 d |
| R13 | Docs (§9.1-9.2, §15, §19.2, user guide, `cli.md`, `serving.md`) and the campaign | RM1-RM8 | 1 d + ~5-6 h GPU |
| R14 | Mirror split reads, only if RM3 stalls ≥ 10 % and a second drive exists | RM10 | 1.5 d |

Tests (§6): RT1 ledger; RT2 every slot transition, lazy heaps against a naive recompute and the
Python reference over 2,000 rounds; RT3/RT3b/RT4 streamed records equal the pinned bank bytes and
the read queue's ordering, retries and cancellation; RT5/RT6 bit-exact routes with a null table, a
permuted table, the fetch channel with a delayed out-of-order responder, heartbeat and wrap seams;
RT7-RT9 sizing, resize and monitor law with fake sources; RT10 greedy ids identical in full mode
and at 43,000 / 29,000 / 13,000 MiB caps across every tier option; RT11 fault injection (the next
request completes with correct ids, no trap); RT12 slot integrity audits (with the host bit-flip
triage rule); RT13 VRAM pressure; RT14 startup; RT15 lifecycle.

#### Measurements and acceptance (fixed before measuring)

Workloads: tg512 only for the full-RAM no-regression check and Feature 1; tier speed on a
**non-repeating session** (NRS: 12 distinct code, story, chat and multilingual prompts, ~5K
generated tokens, disjoint from the trace and profile prompts), first and second halves reported,
C = 1, 2, 4, equal warm-up in both arms. ABBA on one binary with temporary toggles; median, range
and worst case.

- RM0a-RM0g probes (R0): DXGI/NVML/VMM facts, sysmem fallback (RM0b, only with the user's consent),
  VMM speed parity (RM0c, tg512 with R2's toggle), graph bytes (RM0d), expert reads (RM0e), the
  two-level replay (RM0f) and pin time (RM0g). RM1 full-RAM regression; RM2 Feature 1 cost and a
  2 GiB ballast; RM3 tier decode at the caps; RM4 tier prefill; RM5 topic switch; RM6 fault timing;
  RM7 startup and paging; RM8 state, profile and trace; RM9/RM10 only with R12/R14.
- **A1** RT10-RT13 pass with identical ids, no CUDA trap in any fault case, the engine serves the
  next request after a read failure.
- **A2** full RAM with Feature 2 code at equal frames (`--vram-headroom-mib 128`): median of ≥ 3
  ABBA pairs ≥ −0.5 % against `dev`, worst pair ≥ −1.0 %, load time ≤ +5 %, pinned bytes unchanged,
  identical ids.
- **A3** Feature 1: A3a the default configuration's frame cost and tg512 change within the
  predicted −0.2 to −1.3 % (worse than −1.5 % is a defect to explain); A3b a 2 GiB ballast absorbed
  within two boundaries plus the unmap, no `Demoted` rise after the shrink, regrow follows; A3c an
  idle start never grows into the internal reserve. If VMM fails RM0a or RM0c, warning mode is
  documented.
- **A4** tier, on NRS second halves: A4a at 43,000 MiB < 0.1 SSD reads per token and ≥ 97 % of full
  mode at equal frames; A4b ≤ 0.8 ms per SSD read at C = 1 (at 29,000 and 13,000 MiB); A4c plain
  ≥ 80 % of full mode at 29,000 MiB (central prediction 91 %, pessimistic 74 %); A4d pp4096 at
  29,000 MiB ≤ 1.6× full mode at chunk 1024 and ≤ 1.4× at 4096; 13,000 MiB reported only.
- **A5** tier-mode startup time per cap; more than 1.5× today's 25 s is a defect to explain.

#### Decisions and rejected alternatives

| Decision | Rejected (reason) |
|---|---|
| Device-wide free VRAM as the primary runtime signal; DXGI budget and `Demoted` secondary | Budget slack alone (reacts only once VidMm trims); NON_LOCAL (fires on our own pins); PDH (~10 ms per collect) |
| One sizing function with the internal reserve tracked as spent | A separate grow rule that could spend the reserve |
| VMM pool at a fixed address, shrink evicts the least valuable experts and relocates survivors | Evicting the top region as it stands; a segmented `cudaMalloc` pool (segment table in kernels and graphs); shrinking KV or `work_` |
| No frame floor (0 frames still works through staging slots and the CPU) | A floor that keeps VRAM the display needs (U1) |
| Fail fast before reading weights | Today's silent `frames = 0` after 25 s of loading |
| Explicit tier with unbuffered reads of the artifact in place on `E:` | An OS mapped-file tier (Strata's `--mmap-experts`: 7.54 vs 12.18 tok/s with identical flags); a copy on `G:` (63 GiB, contention with the n-gram volume and page file); any SSD expert cache, spill or swap file (U3: SSD writes and wear for no gain, since the artifact already holds the exact bytes) |
| Exclusive residency with shadows, pins and admission throttling | Inclusive or independent tier (2-2.5× more SSD reads in the 512-token replay); a fixed demotion cap that drops victims |
| SSD-only experts as CPU jobs in decode; fetch channel for prefill and overflow | Fetch channel for every call (exposed wait on the serial route); CPU jobs only (caps, no prefill) |
| Decayed LFU with O(1) updates and lazy heaps, h by replay | LFRU for RAM (time-varying order); long-half-life LFU (+30 % reads in the 512-token replay); per-layer quotas |
| Full mode keeps today's pinned Model banks and kernel code (null host table) | An identity table in full mode |
| No precision change for SSD experts; deny-write handles | Lower-bit SSD copies; per-read hashing (~0.25 ms CPU) |
| RAM tier sized at startup only | Runtime RAM grow and shrink (Q10) |
| Recoverable round failure | Engine-wide failure on any read error |

#### Risks and open questions

Main risks: VMM release on WDDM unproven or VMM frames slower (warning-only fallback); DXGI usage
excluding CUDA allocations (spill guard by device free); long device spins during NVMe stalls under
WDDM (heartbeat waits and a host I/O bound below the TDR bound unless preemption is verified,
shared with n-gram S2); the hit model resting on one short trace (long-trace replay, NRS, labelled
uncertainty); the 2 GiB reserve paging Infernix or the desktop (hard minimum working set, RM7);
`E:`'s ~2,100 reads/s at C > 1 or in prefill; slot protocol bugs (pins, serials, audits); code
conflicts with concurrency S3-S5/S4b, vision V3/V4, prefix P6 and n-gram S2 (merge order below);
host RAM bit flips on this machine (triage rule).

Defaults decided before measuring (the user may override): Q1 display headroom 512 / 1,024 / 256
MiB (on this machine the 5090 drives no display, the monitor is on the Intel UHD 770, so `auto` is
the headless value); Q2 obey the OS budget with a 64 MiB margin; Q3 runtime shrink and regrow on if
RM0a and RM0c pass; Q4 "Prefer No Sysmem Fallback" as optional advice only; Q5 the 2 GiB reserve
for every model's pinned check; Q6 prefix tier and vision before experts; Q7 a shipped profile only
behind the RM0f gate; Q8 unchanged prefill chunk with a 4096 hint; Q9 heartbeat device wait, I/O
bound per n-gram S2's gate decision, never a trap; Q10 no runtime RAM-tier resizing; Q11 Linux/WSL
supported with a native filesystem; Q12 R12 and R14 only on their triggers; **Q13** decayed LFU
stays unless RM0f shows LRU or CLOCK with ≥ 5 % fewer SSD reads, then that becomes the default with
LFU selectable, reported as a deviation from the literal request.

#### Integration

Merge order (extends §19.3.0): R0's trace hook and R4 early (before prefix P6 and vision V5 size
their host memory, which then use the ledger); R1 after concurrency S1/S2; R2 after vision V3
(whichever of R2 and V4 lands second reuses the capacity API); R3; R5 and R9 after concurrency
S3-S5, which own `cpu_plan`, `cpu_wait` and `stage_kernel`; R6-R8 any time; R10-R11 after R5-R9 and
S4b (extending its state file and fill rule); R12 and R14 last. Shared rules: the I/O timeout
follows n-gram S2's gate decision; demotions obey the copy-engine FIFO rule through the decode
allowance; kernels Phase 2's L2 warming in `cpu_wait` is unaffected. In-branch this track edits only
this section and its user-guide subsection; §9, §15, §19.3.0, `cli.md`, `serving.md` and README
change at merge time.

#### R0 implementation decisions

- **Route trace** (`program/route_trace.{h,cpp}`, internal `ProgramOptions::route_trace`, set only
  through the process-wide `testing::set_route_trace` seam that `construct_qwen4_exp` reads). The
  format is documented in `route_trace.h`: a 64-byte header (layers, experts, top-k, frames, max
  columns, lanes, max width, MTP and n-gram draft tokens, prefill chunk), then one record per
  round (kind, rows, width, tokens advanced, the promotion budget the cache applied, row 0's first
  position for prefill chunks, lanes, live mask, every layer's routes) written right before
  `after_round` applies the same host route log, so the trace is exactly what the VRAM policy saw.
  It also records each CUDA graph executable the Program instantiates (family, batch, width,
  device free before its capture and after its instantiation and first launch), which is RM0d's
  measurement; the family and shape follow from the graph's slot, so no call site changes. Both
  are bit-neutral: host reads after the round's synchronize, and `cudaMemGetInfo` outside capture.
- **Capture** (`infernix_qwen4_exp_route_trace_real_test`): the public Engine over a prompt list,
  trace off then on with identical ids required; `--mtp`, `--ngram`, `--concurrency` (prompts
  submitted together in FIFO order) and `--single-pass` (graph measurements); the trace is
  re-read and checked per lane against each request's prompt and generated tokens.
- **Probes** `tools/flash_next_probe/vram_probe.cpp` (RM0a, the RM0c read proxy) and
  `expert_read_probe.cpp` (RM0e, RM0g) are CMake executables built beside the tests
  (`probes.cmake`). The read probe locates the `nvfp4_expert_rg16_v1` records through the artifact
  directory, opens the parts unbuffered with the tier's deny-write sharing, completes reads on one
  I/O completion port (an `O_DIRECT` thread per I/O on POSIX) into `cudaHostAlloc(Portable |
  Mapped)` slots, and splits quarters on page boundaries (169, 169, 169, 168 pages).
- **Replay model** (`tools/expert_cache_replay/host_tier.py`), where the plan left a choice:
  - The VRAM part is the engine's `LfruPolicy` decision for decision (binary64 scores, ties to the
    lower key), with the tier rules: admission only with a host copy, the budgeted loop for every
    call, the fill rate keyed on free frames. A promotion is visible from the next round, or one
    later when its victim's demotion holds the frame.
  - The demotion list carries a swap: a victim demoted to admit an expert from RAM may use one of
    the 32 list slots, because the promoted expert's own RAM slot becomes a shadow (exclusive) or
    is freed (strict) when its copy is published and refills the list at the next round start.
    So an exclusive swap never sends a victim to SSD for lack of a slot; the 1.25 × rule decides
    only when no free, shadow or list slot is left. Prefill boundaries can exceed the 32 list
    slots (16 promotions × 48 layers), after which a demotion evicts the lowest RAM resident or the
    victim is dropped by the 1.25 × rule; R10 sizes the list for prefill or keeps that behaviour.
  - Only the last 128 landings of a round (the ring) can be admitted at its boundary; landings
    beyond the ring in a prefill chunk are lost unless they are among the first K reserve
    landings. A prefill chunk's reserve takes free slots before evicting residents.
  - Victims are shadows first, without a score comparison (dropping a shadow loses nothing now);
    a demand landing is admitted when a slot is free, a shadow can go or it outranks the lowest
    resident. For LRU and CLOCK, admission, demotion and prefetch comparisons use recency.
  - Priors seed VRAM with the ranking's top V keys at zero LFRU counts and RAM with the next
    H_res (inclusive: the top H_res) at ranks below any real use.
  - Q13 is applied per cell and over all V and RAM sizes of one prior and mode: another policy is
    adopted only with strictly fewer and at most 95 % of the best half-life's SSD reads.

#### R4 implementation decisions

- **Snapshot** (`core/host_memory`): `host_memory_snapshot()` reads total and available physical
  memory, available commit (Windows `ullAvailPageFile`; Linux MemAvailable plus free swap) and the
  process's private bytes; on Linux available memory is lowered to the cgroup v2 limit and
  `RLIMIT_MEMLOCK` is read. `reserve_process_working_set` sets the hard minimum working set.
- **Ledger** (`models/qwen4_exp/memory_plan`, pure, RT1): planned = the other pins (the token
  embedding) and later pins (a 256 MiB allowance for the Program's pinned and mapped buffers),
  each with 0.25 % lock overhead, plus the n-gram row cache, load staging (264 MiB) and 1 GiB of
  growth. Full mode needs the banks with their lock overhead plus a 1 GiB margin, in both physical
  memory and commit after the reserve. Today's model needs about 69.3 GiB available at the
  default reserve (the old fixed rule needed 72.5 GiB).
- **One reserve.** `kPinnedHostReserveBytes` is gone: `MaterializationPlan::host_reserve_bytes`
  (from `EngineOptions::ram_headroom_bytes`, default `kDefaultRamHeadroomBytes` = 2 GiB) and
  `later_pinned_bytes` are re-checked against memory available at the moment of the pin, so other
  programs' growth since the snapshot is seen. Qwen3.5 passes the same reserve for its pinned
  vision tower (its check falls from 8 GiB to the shared default). The materializer still pins
  one block; chunked pinning and per-chunk re-checks arrive with the tier's slot pool (R10), which
  is the first allocation large enough to need them.
- **Tier mode refuses** until R10 with the ledger line and the reserve in use. `--expert-ram-mib`
  and `--expert-state` are not parsed yet: until the tier exists the first could only force a
  refusal and the second would do nothing, so both land with R10 (a deviation from the R4 row).
- **Working set.** Startup sets a hard minimum working set of the growth term (1 GiB); a refusal
  is logged as a warning, never fatal.

#### R1 implementation decisions

- **Source** (`core/vram_budget`): `VramSnapshot` (device free/total, DXGI LOCAL budget and usage,
  display state, outputs) from `cudaMemGetInfo`, DXGI (`EnumAdapterByLuid` on the CUDA LUID,
  `QueryVideoMemoryInfo`, `EnumOutputs`, recreated when `IsCurrent()` is false) and NVML
  `DisplayActive` loaded at runtime. `change_event()` and the D3DKMT `Demoted` statistic of §3.1
  belong to the monitor and arrive with R3. `spill_shortfall` / `SpillGuard` are the spill test
  (32 MiB tolerance, a second reading before a spill is reported).
- **Sizing** (`memory_plan`, pure, RT7): §3.2's function with P = 0 (no VMM pool yet); not elastic,
  so the automatic headroom is the warning-only value (1 GiB with a display or unknown, 256 MiB
  headless) and the graph reserve counts every graph (`graph_bound`). g stays 4 MiB until RM0d.
  The 64 MiB chunk grid of §3.2 arrives with the VMM pool (R2); a single `cudaMalloc` needs none.
- **Device plan** (`Program::plan_device`): one static function computes every fixed allocation of
  the Program, with the layouts in it (state pool, KV pool and tables, io, verification records and
  round arrays, MTP buffers, workspace, staging, residency tables), from the options and the
  configuration alone (the expert record stride from the format's geometry). The constructor
  allocates exactly from it and throws `logic_error` if the bytes it allocated differ, so the
  startup estimate cannot drift. Option validation moved into it, so a bad option fails before the
  weights are read.
- **Startup order.** `construct_qwen4_exp`: RAM ledger, then the VRAM pre-check (dense bytes from
  the load plan plus the device plan) before `materialize_model`; a negative frame budget fails
  with the contributors, fewer than 2,048 frames warn. A spill guard wraps the dense upload (fatal),
  the Program's fixed allocations (fatal) and the frames (given back once with the shortfall plus
  64 MiB, then fatal). One `VRAM ledger` line follows the Program; `MemorySummary` now carries the
  after-weights and after-startup free memory, the headroom (`vram_headroom_bytes`, renamed from
  `kv_capacity_headroom_bytes`) and the slack for Qwen4Exp.
- **Flag.** `--vram-headroom-mib N|auto` sets `EngineOptions::vram_headroom_bytes` (empty = auto)
  in `infernix`, `infernix-serve` and `infernix_bench`; `KvCapacityPolicy` no longer carries a headroom.
  Qwen3.5 reads the same option for `--kv-capacity auto` and refuses it with an explicit capacity
  (moved from the parsers into its Engine validation). The fixed 384 MiB
  `expert_cache_reserve_bytes` is gone.

#### R0 measurements (RTX 5090, headless, Windows WDDM, 2026-10-04)

- **RM0a.** DXGI LOCAL `CurrentUsage` counts CUDA allocations (1,024 of 1,024 MiB), and
  `cudaMemGetInfo` free equals the OS budget minus that usage (budget 31,419 of 32,579 MiB with no
  display): free memory and the budget bound coincide on this card. VMM: create 77 / map 1 /
  access 4 / unmap 60 / release 162 us (p50, idle); from another thread during launches unmap
  0.9 ms and release 0.27 ms p50 (p99 7.6 / 9.8 ms) and the launch loop slows 9 %, so resizes
  unmap on the engine thread at a boundary; unmap does not wait for running kernels; released
  memory returns at once. VMM reads equal `cudaMalloc` reads (streaming -0.01 %, gather +0.03 %).
  Queries cost under 1 us idle and 3-7 us during launches; the DXGI budget-change event fires.
- **RM0d.** CUDA graph executables: 13 at C = 1 (MTP 3 + n-gram 7) take 26 MiB (2.0 MiB each,
  largest 6); 43 at C = 8 (MTP 7 + n-gram 15) take 110 MiB (2.6 MiB each, largest 8). The 4 MiB
  allowance per graph covers them.
- **RM0e.** Whole-record reads from the artifact: QD 1 0.553 ms p50 (4.68 GB/s), QD 8 6.71 GB/s;
  quarter reads are slower (QD 1 0.89 ms per record), a concurrent H2D stream does not slow
  them, and one 256 KiB prefetch sub-read in flight adds ~7 % to a demand read's p50.
- **RM0g.** Pinning runs at 9.6-10 GiB/s, the same for 1.32 GiB chunks and one 13.2 GiB block.
- **Bit-neutrality.** The route trace leaves greedy ids unchanged (12 prompts, plain and MTP).

#### R2 and R3 implementation decisions

- **Pool** (`core/vmm_arena`, helpers shared with `EvictableWeightPool` in `core/cuda_vmm.h`): one
  address range the size of the card, 64 MiB chunks mapped and released at the top.
  `ExpertResidency` starts with no frames; `resize()` grows chunk by chunk (each chunk touched,
  then checked by the spill guard: a spilled or refused chunk ends the growth) and shrinks by
  `CacheController::resize` (lowest-score experts evicted wherever they are, survivors above the
  new top moved below it by device copies on the compute stream, one table upload), then waits
  for the compute stream and unmaps. Without VMM the frames are one allocation of the first size.
- **One phase, not two.** In full RAM mode every expert keeps its host copy, so a shrink never
  demotes: phase B of §3.4 (waiting for demotion copies) has nothing to wait for, and the shrink
  completes at the boundary that decides it (about 1-3 ms). The two-phase form returns with the
  SSD tier's demotions (R10).
- **Monitor** (`program/vram_monitor`): one thread, woken by the DXGI budget-change event or every
  second, keeps the latest snapshot and passes it to the Program. The law (`VramControl`, pure,
  RT9) shrinks at once when free memory is below half the headroom or usage exceeds the OS budget,
  to one chunk below the sizing function's target; grows once the target has stayed two chunks
  above the pool for 30 s, by at most four chunks per round boundary while requests run and to the
  target when idle. The reserve is the sizing reserve less this process's usage growth other than
  the pool's (DXGI usage counts CUDA, RM0a). The D3DKMT `Demoted` trigger is not used: on this
  driver free memory is the budget less usage, so budget pressure already shows as low free
  memory.
- **Boundaries.** The Program applies the decision after every `after_round` (prefill chunks,
  forced tokens, decode and verification settle): no round's kernels are in flight there, and the
  commit-phase work still running (GDN fold, tails, the drafter's catch-up) reads no frame. An idle
  engine is woken by the monitor (`Program::set_maintenance_waker`) and runs `maintain()` from its
  worker under the execution lock (`EngineCore`, only for Programs that provide it).
- **Headroom.** With the elastic pool the automatic headroom is 512 MiB with a display and 256 MiB
  without; without VMM it stays 1 GiB with a display and the monitor only warns, once per
  pressure episode, with the headroom that would have avoided it.

#### Status

R0 implemented on `claude/fn-memory` (base `1dae6914c`) and run: the probes, both traces and both
graph captures pass (above). The replay (RM0f) runs after the timing work, which it would disturb.

R4 implemented on `claude/fn-memory` (`a3713f5f6`): `--ram-headroom-mib` in `infernix`,
`infernix-serve` and `infernix_bench`; RT1 in `infernix_qwen4_exp_memory_plan_test` passes.

R1 implemented on `claude/fn-memory` (`d79ad7172`): startup VRAM sizing, pre-check, spill guard,
ledger and `--vram-headroom-mib N|auto`; RT7 and the engine smoke pass.

R2 and R3 implemented on `claude/fn-memory`: the VMM frame pool with resize, the monitor, the law
and the idle hook; RT8 (host), RT9 and RT13 (engine, fake source) are the tests.

R7 and R8 implemented on `claude/fn-memory-tier` (from the integration branch, 2026-10-05):

- **R7** `core/direct_read_queue` as planned (IOCP; a `pread` pool on POSIX, where an issued read
  cannot be cancelled, so the timeout is not enforced there and `cancel_all` waits). Two depth caps
  plus a common one, 256 KiB prefetch sub-reads, transient backoff within `transient_bound`, one
  retry for other errors and for a timeout (after `CancelIoEx`), short reads fail at once. RT3b
  (`infernix_direct_read_queue_test`) passes; the POSIX path is compiled only, not run.
- **R8** `program/expert_cache/host_tier` as planned, with these choices: a ring or prefetch slot
  stays `Free` while the agent writes it (the engine thread knows a landing only through its
  serial at the boundary; `Landing` marks a demotion target during its D2H); a free resident slot
  is used before any victim; the demotion list is refilled from free slots first. RT2
  (`infernix_qwen4_exp_host_tier_test`) checks the LFU against the naive sum (< 1e-9 relative), every
  transition and a 2,000-round randomized run against a naive victim scan with the audit after
  every round. **Deviation:** no Python reference fixtures; the C++ naive oracles decide the same
  victims.
- **R6** `artifact::Residency::Streamed`: the binder locates the object and the materializer
  reads nothing; `MaterializedArtifact::stream_source()` keeps the file paths (entry and parts,
  headers validated through `Reader::file_path`) and each streamed object's file segments after
  the Reader is gone. `LoadOptions::stream_experts` binds the 48 banks `Streamed`; the load reads
  each bank's scale tail into Model memory, `expert_bank_layout` gives the bank planes without
  records, and the Model owns an `ExpertStore`: each record's one or two 4 KiB-aligned segments
  (the constructor rejects any other). The product never sets the option until R10, and the
  Program has no tier yet, so it is reachable only from tests. RT4
  (`infernix_qwen4_exp_expert_store_real_test`): every record of the first, last and straddling
  layers read through R7 from the store's segments equals the pinned load's bytes.
- **R5** (part): `MoeExpertSource::host_table` (device array of host record pointers; null keeps
  today's `host_records + e x stride`), read by `record_of`, the stage kernel's zero-copy and
  staging copies and the CPU plan, which writes `MissRequest::record[j]` and `tiered`; the CPU
  service then uses those pointers. The CPU request sequence skips 0 when it wraps (an existing
  defect: after 2^32 requests a call's CPU jobs got no outputs). RT5: the layer test's routes with
  the records in shuffled host slots reached only through the table (zero-copy, staged, fork, CPU
  jobs, prefill assist). **Not done:** the per-layer table in `Forward` and promotions through
  the host-pointer mirror (R10).
- **R9** (2026-10-05, op level): `cpu_plan` takes SSD-only experts (null table entry) first, every
  one of at most `max_job_columns` columns up to the cap, then host-memory misses under the divisor
  rule over host-memory misses only. `CpuMissService::Options::records` (`RecordProvider`:
  `demand`, `landed`, `wait`, `done`) serves a job without a record: its read is demanded
  before the other jobs run and computed after them; a missing provider or failed read answers an
  errno in the status word. The fetch channel (`MoeFetchChannel`, `cpu/fetch_request.h`,
  host side `cpu/fetch_channel`) follows §4.6.3 with these choices: a one-CTA `fetch_plan_kernel`
  after the CPU plan does the stable fetch-last partition (`cpu_flags` and the CPU call's job
  indices move with it) and publishes the request; the poller is whichever stage CTA holds a
  device token (no dedicated CTA, so no CTA must outlive its own copies), mirroring `landed`
  into the call's workspace; `consumed` is one 64-bit word (sequence << 32 | records), stored by
  the last CTA of each pass with fetch records (so a responder's ring must exceed the largest
  pass's fetch records: 64 in decode, the staging half in prefill); the host's `land` accepts
  any order and publishes the complete prefix. A fetch channel needs staging (the zero-copy route
  could not release ring slots) and at most `kMaxFetch` = 512 jobs. Tier records are read with
  `.cv` loads (their host slots are rewritten; the full-mode bank keeps `.cs`). Waits: the CPU
  wait and the fetch poller end when the host's heartbeat has not changed for 1 s and write
  `kErrorHostSilent` (no trap unless the source has no error word); host failures write their
  errno; an SSD-only expert with no path writes `kErrorUnservedRecord`. Tests (layer test): SSD-only
  sets served by CPU jobs (caps 8 and 256, fork, zero staging) and by the fetch channel (serial
  passes of 1 and 3 slots, overlap, fork, landing, with CPU jobs taking the narrow ones), a
  responder that fails a request or stays silent, an unserved set without a channel, and a silent
  CPU host, each against the CPU engine's bits or the expected error word. **Not done:** the
  tier's agent as responder (R10).
- **R10.1** (2026-10-05, tier mode end to end without demotions): the ledger's tier placement (or
  `--expert-ram-mib N`, new in `infernix`, `infernix-serve` and `infernix_bench`; a cap above what is free is an
  error) loads the banks streamed and gives the Program the expert RAM less its lock overhead. The
  Program pins the tier (refused below 1,024 slots), its ring max(128, CPU cap + 2 x 64 staging
  slots) so one call's CPU jobs and a pass's fetched records always fit, and owns the
  `FetchChannel` the tier's agent answers: a key in RAM lands at its slot at once, the rest are read
  into ring slots that are reused once the device consumed them or the CPU job finished (the agent
  hands out every ring slot, so neither path fails for want of one); the agent spins from
  `begin_round` to `end_round` (2 s without work: 1 ms polls). `ExpertResidency` owns the interplay: the
  tier's boundary and the per-layer host-pointer table upload in `before_round`; in `after_round` the
  round's routed keys feed the decayed LFU (clock + min(live columns, 16)), only keys with a host copy
  are admitted (LFRU admission filter; SSD-only experts reach VRAM through S4 landing), VRAM
  evictions are reported (T3 keeps the RAM copy as a resident, T5 drops), promotions copy from RAM
  slots under Queue (T8/T9) and H2D (T1/T2, counted per key for the ABA re-issue) pins. F2 streams
  only experts with a host copy, one copy per record (adjacent slots of separately pinned chunks
  cannot be one copy: the first GPU run failed with `cudaErrorInvalidValue` until this was fixed). Startup pre-fills RAM
  with the saved ranking, then file order; VRAM seeds only from RAM. Every MoE call has an error word
  (full mode too: a silent CPU host is now recoverable); the Program checks it after each round's
  synchronization and throws `runtime::RecoverableExecutionError`, which the engine handles like a
  recoverable logic error (fails the round's requests, keeps serving); the release then clears the
  prefix cache, since the failed round may have published state it never computed. **Not done:**
  the state file's RAM section (the VRAM ranking leads the pre-fill), prefetch, the prompt landing
  reserve (R11); lending and resizing drop rather than demote; the host-pointer table (196 KiB) is
  outside the Program's device plan. Each request logs its SSD reads, fetched records, admissions and
  demotions. Measured (dense8m, int8 KV, cold, `--expert-state off`, one run each; full / 32 GiB /
  8 GiB): greedy ids identical to full mode for code plain, code MTP and the long prompt at chunk
  4,096 (6/6); tg512 109.6 / 86.3 / 46.3 tok/s; code decode 79.6 / 39.1 / 18.6, MTP 128.1 / 54.1 /
  24.4; long-prompt prefill 977 / 542 / 407 tok/s. Mean read 0.6-0.8 ms in decode, 17-62 ms in
  prefill (queued behind the layer's other reads).
- **R10.2** (2026-10-05, demotions, T4/T6/T7): the LFRU's budgeted loop consults the tier's
  `allow_evict` gate (tier mode always takes that loop; a refused victim stops the layer's
  admissions); after the routes and before `on_quiescent`, a victim without a host copy whose frame
  is published and that outranks the RAM victim is demoted: `CacheController::hold` keeps its frame
  (FramePool `kDemoting`) and in the table, a D2H stream copies it to a demotion-list slot; the next
  `before_round` after the copy publishes the slot (T7), unmaps the frame, releases it and issues
  the queued loads. The allowance is 32 per decode or verification boundary, unlimited at prefill
  chunk boundaries.

### 19.3.8 Prefill (proposed track M8, steps F0 and F1)

Branch `claude/fn-prefill`. The steps are §19.3.6 item 4's F0-F6 (`strata-amendments.md`, "Prefill");
this section records the design decisions of F0 and F1. Results are recorded by the merging session.

#### F0: attribution of a multi-chunk prefill

- **Instrumentation.** Qwen4Exp's prefill had no NVTX ranges, so a timeline could not say what the
  host did while the GPU idled. `advance_prefill` now opens permanent ranges in the infernix domain
  (`core/nvtx.h`, as Qwen3.5 does): `prefill.chunk` (payload: width), `prefill.ple_rows`
  (`stage_sequence`: n-gram hashing and row reads), `device.wait` (the sync after a non-last
  chunk) and `prefill.residency` (`after_round`: host LFRU policy and promotion issue). They are
  chunk-granular, cost nothing without a tool attached, and stay outside the files other tracks own
  (`ngram_volume.cpp`, `expert_residency.cpp`). An earlier ad-hoc patch with formatted range names in
  those files was replaced.
- **Rig.** nsys (`--trace=cuda,nvtx`) of `infernix_bench -p 16384 --prefill-chunk 4096 -r 2
  --warmup 0`: four cold chunks in a fresh process (empty expert cache, cold n-gram rows), then the
  same prompt warm; once per arm of the F1 toggle, plus profiler-free runs of the same workload.
- **Attribution.** `f0_attribution.py` reads the SQLite export. Per chunk window (from a chunk's
  range start to the next one's) it reports GPU busy time (union over all streams), the narrow and
  wide expert kernels, `stage_kernel` and its exposed part (running while no other kernel runs),
  other kernels, the io upload and its queue delay behind promotions (GPU start minus API return,
  and the promotions that finished in between), promotions (count, bytes, span), and idle GPU time
  split by the host range active at the time (`prefill.ple_rows`, `prefill.residency`,
  `device.wait`, other).

#### F1: the wide route

- **Route.** An expert takes the wide route when it has more than eight columns in the call and its
  gate and up input scales are equal; otherwise the narrow route, at any width. Both conditions are
  properties of the call's routing and the stored scales, never of placement, so outputs stay
  placement-invariant. Unequal gate/up scales would need two A operands inside one MMA tile (gate and
  up rows interleave in the record); the narrow route computes such experts exactly instead.
  Measured on 2026-10-04: all 24,576 experts of `Qwen3.8-Flash-Next-NVFP4` share the scale, so the
  rule only covers recipes that do not exist today. The narrow kernels skip wide jobs; the fork
  route (one pass, graph-captured decode) is limited to calls of at most eight columns, where no
  expert can be wide.
- **Reuse.** The GEMM is the warp-specialized TMA pipeline of the dense W4A4 route
  (`nvfp4_a4_tma_kernel`) made grouped and persistent (`cuda/wide_expert.cuh`): one producer warp
  issues the stage loads on full/empty mbarriers, eight consumer warps run
  `mma…kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64` (`mma_nvfp4_e4m3`). Reused unchanged: the
  tensor-map construction (`nvfp4_make_tma_2d`), `nvfp4_tma_load_2d`, the Windows descriptor staging
  (`TmaDescriptorStaging`, one staging per layer call), the mbarrier helpers, the 64-byte-swizzled
  activation tiles with the dense kernel's `ldmatrix` A fragments and scale lanes,
  `nvfp4_prepare_shared`, and the canonical arithmetic of `canonical_math.h`. From Qwen3.5's
  `sparse_moe/prefill`: the device-built list of (expert, column tile) jobs and the grid-stride walk
  over jobs times row blocks, and packed rows grouped by expert instead of a gathered activation copy.
- **Not reused, and why.**
  - `nvfp4_a4_tma_kernel` itself: its B operand is a dense, tensor-mapped weight with persistent
    128-row scale tiles, one matrix per launch. Expert records (`nvfp4_expert_rg16_v1`) sit at
    per-job addresses in frames, staging slots or the host bank, and their row-group units cannot be
    described by a tensor map in the MMA's operand order. Its token tiles are 128 or 256 (tied to the
    tiled activation-scale layout), against ~20 columns per expert at chunk 1024 and ~80 at 4096.
    Parameterizing it would also requalify five dense routes.
  - `nvfp4_a4_mma_kernel` (cp.async): the same weight-layout constraint.
  - `quantize_nvfp4_k16` and `launch_nvfp4_a4_quantize`: their divisor formulation is not the
    canonical A4 rule of §16.2, and they quantize a whole matrix with one scale, not each expert's
    rows with its own.
  - The token-major SwiGLU epilogue: it fuses SiLU on FP32 accumulators, without the BF16 y
    boundary, `swiglu_bf16` or A4(h).
  - `sparse_moe/prefill`'s kernels: BF16-activation Q4/Q8 codecs on BF16 MMA; its router and
    selection duplicate `moe_route`/`moe_dispatch`.
- **Operands.** Activations are the A operand (M = 16 columns), weights B (N = 8 rows), as in the
  dense route.
  - A: one plane row per routed entry in dispatch order (expert-grouped), filled once per call by
    `quantize_kernel` with the canonical rule (`canon::quantize_a4_codes`, the code-returning form of
    `quantize_a4_block`, equal by test). x rows: 1,280 code bytes + 160 scale bytes (59 MB of
    workspace at chunk 4096, ~21 frames). The A4(h) rows (320 + 40 bytes) live in the entry's unused
    narrow h-block row (stride 1,120 bytes), so h costs no workspace. Both planes are read through
    tensor maps; a tile's rows past its expert read the next expert's rows, and the epilogue discards
    them (MMA rows are independent).
  - B: each K tile's eight 144-byte units of a row group are one contiguous 1,152-byte run, moved by
    one `cp.async.bulk` into shared memory unchanged. A thread forms its fragments in registers:
    the record's word at 32q + 4i holds four k of row i (low nibbles) and of row i + 8 (high nibbles),
    so `(w0 & 0x0F0F0F0F) | (w1 & 0x0F0F0F0F) << 4` is a complete B register for rows 0-7 and the
    high-nibble form for rows 8-15. Its eight K slots hold k, k + 4 of each half block in that order,
    and the activation plane stores its codes in the same order (byte i of a block:
    code[8(i/4) + i%4] | code[8(i/4) + 4 + i%4] << 4). Within each 64-wide K step the MMA's four
    16-slot blocks take record blocks 0, 2, 1, 3 (`plane_block`, applied to the activation plane
    too), which puts the four lanes of a group on disjoint banks of the 144-byte-stride units. Weight
    scales are four byte loads per row and K step, assembled with `prmt`.
- **Tiles and pipeline.** 64 columns × 128 rows × 128-wide K tiles, three stages (43 KB, two CTAs
  per SM), eight consumer warps as 2 (columns) × 4 (rows). A warp's 32 rows are one 16-intermediate
  block of h, and gate row 2i and up row 2i + 1 land in the same thread, so the gate/up epilogue is
  registers and shuffles: y = bf16(acc ⊗ α), h = `swiglu_bf16`, the block maximum by two shuffles,
  the canonical scale and codes, the eight code bytes by two OR-shuffles. The down epilogue writes
  y = bf16(acc ⊗ α_down) to the entry's output column. The grid is persistent (two CTAs per SM)
  over the pass's (tile, row block) items; the epilogue uses no shared memory, so the producer
  fills the next item's stages meanwhile.
- **Passes.** The work list is planned once per layer call for the pass size moe_experts stages
  (half the slots with the overlap stream, all of them otherwise), and the wide GEMMs of pass p run
  after the narrow kernels of pass p, before its slots are released. Staging and the pass loop are
  unchanged (F2 replaces them).
- **Numerics.** Up to the accumulator the arithmetic is §16.2's; only the sum is the tensor core's
  FP32 accumulation of exact block products instead of one exact int64 sum. Qualification
  (`test_offloaded_moe_wide.cu`): the code quantizer against the block quantizer (exact); both GEMMs'
  accumulators, read through a probe epilogue, against the exact sums within an FP32 accumulation
  bound; the down epilogue bitwise; end to end against the FP64 chain with the canonical narrow
  arithmetic as the rounding-order control; narrow-route experts bitwise against the CPU engine;
  and bit-identical outputs across placements and pass layouts.
- **Measurement.** A temporary environment toggle, never committed (`f1-toggle-temporary.patch`),
  puts every expert back on the narrow route in the same binary, for ABBA prefill timing, the
  teacher-forced A15 check (8K text at chunk 4096 against a chunk-2048 rounding-order control) and
  re-recorded greedy baselines. The x-plane workspace is the one cost the toggle cannot show; it is
  read from the startup frame count against the base build.

#### F1 results (2026-10-05, RTX 5090, Windows, INT8 KV, toggle build, ABBA)

- **A VRAM defect found and fixed first.** The first measurement had the wide route 76 % slower
  end to end (146 against 613 tok/s) while its kernels were faster: the first wide launch took
  3,336 MiB of device memory. Its record-unit loads used `cp.async.bulk` with a `shared::cluster`
  destination; ptxas guards each such copy with a call to a driver routine (the remote-CTA path),
  and the driver raised the per-thread stack limit from 1 KiB to 14,416 B at that launch
  (× 170 SMs × 1,536 threads). With the expert frames filling VRAM, allocations fell back to WDDM
  system memory and every kernel slowed. The kernel runs without clusters, so the `shared::cta`
  destination is the same copy with no guard (a PTX probe shows one `UBLKCP` and no call); after
  it the stack limit and free memory are unchanged across the first launch. Other `CALL.ABS`
  sites in the binary (e.g. the INT8 and VQ prompt-attention kernels) are float-division slow
  paths and do not raise the stack.
- **Op** (`infernix_offloaded_moe_wide_bench`, frames placement, per layer call): T = 4096 79.2 →
  1.86 ms, T = 1024 20.5 → 3.27 ms, T = 256 6.32 → 4.68 ms, T = 64 2.87 → 2.89 ms (equal).
- **End to end** (`infernix_bench`, two runs per arm, means): pp4096 at chunk 4096 674.1 → 982.6
  tok/s (+45.8 %), pp16384 at chunk 4096 646.8 → 1081.9 (+67.3 %), pp4096 at chunk 1024 420.8 →
  474.7 (+12.8 %); tg512 decode 88.14 → 87.91 tok/s (−0.26 %, within the arms' spread: A 88.06 and
  88.22, B 88.03 and 87.79; decode does not use the wide route). VRAM free after a request
  375-379 MiB in both arms, the same expert frame count.
- **Quality:** teacher-forced logits of the narrow and wide routes were bit-identical on code, doc
  and chat texts at chunk 1024 and an 8K text at chunk 4096 (the artifact's experts all have equal
  gate and up input scales); the op test's FP64 checks pass (C1-C6).
- **Merged without** the temporary toggle; the NVTX ranges of F0 were ported onto the current
  `advance_prefill`.

#### F2 attribution (2026-10-05, dev 9ec2580a5, nsys, four cold then four warm 4096-token chunks)

| Per chunk (ms) | Cold chunks 2-4 | Warm chunks 5-8 |
|---|---:|---:|
| Wall | 3,440-3,578 | 3,141-3,442 |
| `stage_kernel` (SM zero-copy staging) | 2,417-2,568 | 2,072-2,493 |
| Expert kernels (narrow + wide) | 158-160 | 154-160 |
| Other kernels (dense, attention, norms) | 909-914 | 845-914 |
| GPU idle | 14-16 | 16-18 |

Staging is 70 % of a chunk. Its ~315 non-resident experts per layer (870 MB) move at ~17 GB/s
through the SMs, against 27.6 GB/s for copy-engine DMA, and they cannot start before the layer's
router, so the link idles during each layer's ~19 ms of dense and attention work. Floor: 48 × 870 MB /
27.6 GB/s ≈ 1.5 s per chunk if the link never idles (prefill ~2,600 tok/s at chunk 4096, estimate).

#### F2 and F3 as implemented (branch `claude/fn-prefill-f2`, 2026-10-05)

- **Streaming without routing.** A chunk of T ≥ 256 columns routes nearly every expert of every layer
  (P(unused) = (1 − 10/512)^T: 0.7 % at 256, ~10⁻⁹ at 1,024), so a layer's copies need not wait for
  its router: `execution::ExpertStream` copies each layer's non-resident experts, in ascending id, by
  `cudaMemcpyAsync` on its own stream (runs of consecutive experts coalesced into one copy), two
  layers ahead of the compute stream, into half l % 2 of a ring. Events per layer: `landed[l]`
  (compute waits before the layer's experts) and `consumed[l]` (recorded after the layer's
  `moe_experts`; layer l + 2's copies wait for it). The slot tables (I32 [48][512], slot or −1) sit
  in the ring's first bytes, uploaded per chunk on compute after the io upload, so the copies queue
  behind the io on the copy engine. Experts beyond a half's capacity stay misses and are staged as
  before; experts the chunk does not route cost a copy and nothing else.
- **Op.** `MoeExpertSource::prefetched` / `prefetch_base`: a non-resident expert with a slot is
  resolved like a resident one by `stage_kernel` (never staged, never CPU-served; calls with
  CPU-served misses take no table). Results are unchanged by construction (the same record bytes).
- **F3, lent frames.** At the first chunk of ≥ 256 columns the Program lends
  2 × (the largest non-resident count of any layer) + 1 frames (`ExpertResidency::lend`, the
  Vision window's mechanism: the cheapest run is evicted, promotions into it are waited for), and
  returns them when no lane is prefilling (after the prompt's last chunk is enqueued, on release, or
  before a Vision window lends). A Vision quote counts the ring as lendable. The VRAM monitor's
  resize waits while frames are lent. The startup frame count and decode are unchanged; the lent
  experts refill through promotions after the prompt.
- **Options.** `ProgramOptions::prefill_stream` (default on). Chunks below 256 columns keep the
  staging path. A temporary toggle (`INFERNIX_Q4_PREFILL_STREAM_OFF`, never committed) gives the
  A/B in one binary.
- **Tests.** The layer test adds a streamed placement (every second non-resident expert in a slot
  buffer, reverse order) on the serial, pass and fork routes, bitwise against the CPU engine.
  Model gate: rig `fn/rigs/prefill/f2.bat` (prefix exactness, long-prompt ids with the stream on
  and off, prefill ABBA at chunk 4096 and 1024, tg512).

#### F2 results and the F1 nondeterminism (2026-10-05)

- **Speed** (`infernix_bench`, ABBA in one build with a temporary toggle, three repetitions each):
  pp4096 at chunk 4096 1,128.8 → 1,567.3 tok/s (+38.8 %), pp16384 at chunk 4096 1,408.3 → 2,053.5
  (+45.8 %), pp4096 at chunk 1024 508.9 → 526.7 (+3.5 %); tg512 100.3 → 100.1 (−0.25 %, within
  noise). The ring was 769 lent frames and every half filled (384 experts per layer per chunk: a
  cold cache has more misses than that), so a larger ring would gain more.
- **Ring size (2026-10-05).** Halves sized to the worst layer (512 records on a cold cache, 1,025
  lent frames, 2.8 GB) against the 384-record cap, ABBA in one build: pp4096 at chunk 4096 1,522 →
  1,662 tok/s (+9.2 %), pp16384 1,958 → 1,997 (+2.0 %), the 7,448-token CLI prompt 908 / 960 →
  975 / 989 tok/s, decode after it 71.6 / 72.6 → 72.5 / 72.7, ids identical. The cap is gone: a
  half holds every non-resident expert of the worst layer at lend time.
- **Exactness of F2 itself:** with the wide route off (narrow arithmetic everywhere), the
  7,448-token prompt's greedy ids are identical with the stream on and off and equal the pre-F2
  build. The layer test's streamed placements are bitwise exact.
- **F1 nondeterminism (open, blocks merging F2):** with the wide route on, the same prompt's ids with
  the stream on differ from the stream-off run and between repeated runs. The prefix-cache test
  (`infernix_qwen4_exp_prefix_cache_real_test`) fails its tap-resume and Host-restore exactness checks
  with the wide route on and passes with it off, already on dev (5df90b1ce onward); cold prompts in
  fresh processes are reproducible, and the wide route's arithmetic equals the narrow route's on
  this artifact. So the wide route's results depend on timing or placement inside a running
  Program. Excluded so far: records straddling VMM chunks (layer test C7, identical outputs),
  padding after records (the stride equals the record size), the epilogues' row guards and the
  producer's tensor-map acquires (read). Under test: the prefill staging overlap (pass p+1 staged
  on the side stream while pass p computes).

#### Long prompts: attribution and plan (2026-10-05)

- **The bound.** A 4,096-token call routes nearly every expert, so F2 streams every non-resident
  expert of every layer for every call: ~384 per layer × 2.76 MB × 48 layers ≈ 51 GB, ~1.8 s at
  the x8 link's 27.6 GB/s. The call's other work is ~1.0 s (16K prompt, dev 9ec2580a5, nsys,
  `fn/prof/prefill_kernels.py`). So chunk-major prefill is link-bound: pp16384 ≈ 2,000 tok/s,
  about 65 s for a 128K prompt (an extrapolation; not measured end to end).
  - This is also why the BF16 text routes (aa6a85985) moved pp4096 by only +0.3 %: the call
    waits on the link, not on the GEMMs.
- **The call's other work per 4,096 tokens** (same trace, before the BF16 routes; ms):

  | Kernel | ms |
  |---|---:|
  | `bf16_general_gemm` | 270 |
  | QSA `attention_kernel` (FP32 SIMT; flat beyond 2K context, since the budget caps attended tokens) | 245 |
  | `q8_a16_mma` | 176 |
  | Expert GEMMs | ~160 |
  | `split_kernel` | 105 |
  | Router projection | 36 |
  | QSA selection (~85 ms per call at 128K, from Q2's per-token cost) | 10-14 |

- **Plan** (F4, then the compute items F5 and F6; a 128K nsys on current code confirms the split
  first):
  1. F4: link once per span instead of once per call.
  2. F5, F6: then cut the compute that becomes the bound.

- **128K measured** (2026-10-05, dev aa6a85985 + pipelined decode (decode only), `infernix_bench
  -pg 131072,32`, chunk 4096, INT8 KV, docs corpus, nsys `fn/prof/pf128k`):
  - Prefill took 81.9 s (1,600 tok/s; 82.5 s under nsys).
  - Each 4,096-token call takes 2.3-2.5 s of wall time, of which the GPU is busy ~0.83 s; the
    rest waits on the expert link.
  - The n-gram rows cost **21.7 s** of host reads over the prompt: 2.1M rows, 31 % host-cache
    hits, 1.27M 4 KiB reads at ~17 µs each, before each call while the GPU idles.
  - Kernel time of the late calls (100K+ of context) per call:

    | Kernel | ms |
    |---|---:|
    | QSA attention | 248 |
    | `q8_a16_mma` | 175 |
    | `split_kernel` | 98 |
    | QSA scoring | 73 |
    | Wide expert GEMM | 55 |
    | Narrow expert kernels launched per staging pass (nearly every expert takes the wide route) | 47 |
    | Router projection | 36 |
    | BF16 TMA | 16 |

#### F7: n-gram rows read on several threads (re-applied after F9)

**Status.** Re-applied 2026-10-05 after F9, with 8 threads at depth 16 each (`kParallelInFlight`).
The first measurement below was confounded by the mapped volume (F9).

**Probe** (`fn/probes/ioring_bench.cpp`, unmapped volume, random 4 KiB reads):

| Threads × depth | Reads/s |
|---|---:|
| 1 × 64 | ~220K |
| 4 × 64 | 391K |
| 8 × 64 | 663K |
| 8 × 16 | 878K |

The probe ran beside a build, so these are lower bounds.

**Engine** (`fn/rigs/layer/vf.bat`, 128K prompt, `--vram-past-budget`, one run, against
`diag2.bat`'s single-thread F9 build):

| | Single thread (F9) | F7 re-applied |
|---|---:|---:|
| 1.27M NVMe reads | 5.9 s | 1.29 s (~980K reads/s) |
| Host staging per 64K span | 3.0 s | 0.70 s |
| 128K prefill (tok/s) | 5,331 | 6,576 (+23 %) |

pp16384 at `--max-ctx 20480` was 6,051 tok/s; the 40K CLI prompt prefilled at 5.60k tok/s.

**History** (kept): the version first measured below was rejected and reverted.

The volume is an Optane P5800X, about 5 µs per 4 KiB read and over 1M reads/s. The hypothesis was
that one thread submitting overlapped `ReadFile` calls is bound by the call itself (~15-20 µs on
Windows), about 60K reads/s.

**What was tried.** A `read_rows` call with at least 512 distinct missing blocks was split over
8 threads, each with its own 64-block ring. The rows went straight to the output, and the host cache
was filled afterwards on the calling thread. The rows were exact, and `test_ngram_volume.cpp`
passed.

**Result.** In the engine, the 128K prompt's n-gram read time rose from 3.16 s (dev aa6a85985) to
4.09 s.

**Disposition.** The likely causes are contention with the prefill's CPU workers, and a queue
depth (8 × 64) far past what the drive needs. Neither was isolated. The change is reverted, and its
patch is kept at `fn/rigs/layer/f7_parallel_reads.patch`. Read time stays the 128K prompt's
largest host cost (§19.3.8), so a contention-aware design (a persistent reader thread, depth near
the drive's knee) remains open.

#### F4: the layer walk

- **Order.** A span of consecutive calls of one prompt runs layer-major: decoder layer l over
  every chunk of the span, then layer l + 1. Each layer's non-resident experts are streamed once
  per span (F2's ring, two layers ahead; a layer's half is released after the span's last chunk).
- **Same results.** The call grid, and with it every output bit, is the chunk-major one. Chunk c at
  layer l reads exactly what call c reads at layer l: its own residual columns, and the layer's KV,
  GDN, PLE and QSA state after chunks 0..c−1. Only expert placement differs, and the MoE
  arithmetic is placement-invariant.
  - `Forward::run` is now `begin_call`, `embed`, `layer` × 48 and `finish`; the walk calls the
    same steps per (layer, chunk).
- **Spans.** Consecutive calls of at least 256 columns, up to `kWalkMaxTokens` = 65,536 tokens.
  - A span ends at a call whose end realizes a prefix tap: the capture needs the boundary's state
    in every layer. `prefix_tap_due` mirrors `prefix_after_prefill_call`.
  - A single call keeps the chunk-major path.
- **Memory.** The span's residual stream (BF16 [S·H, tokens], 20 KB per token: 1.3 GB at 64K)
  and every chunk's staged io image live in the stream lease after the ring.
  - At the prompt's first walk the lease is lent with that area, and the ring's halves are widened
    by the experts the larger lease evicts.
  - Without enough lendable frames the prompt runs chunk-major.
  - Link time per layer (~38 ms) is below a span's compute per layer from about two chunks on, so
    64K spans are compute-bound.
- **Steps.** Staging reads every chunk's n-gram rows and io into pinned memory and uploads them
  once; each chunk's embedding and layer 0 are enqueued as soon as its io is uploaded.
  - Each later `advance_prefill` enqueues ⌈48 / chunks⌉ layers of every chunk (about one call's
    work), synchronizes, and returns zero processed tokens as a service unit.
  - Other lanes, cancellation and decode rounds interleave as between calls.
  - The last step returns the span's tokens and runs the call-end work once: the route download
    (the span's last chunk), `prefix_after_prefill_call`, and `after_round` with 16 promotions per
    layer per chunk.
- **Interactions.**
  - An abort inside a span leaves no endpoint: the layers are at different positions.
  - Another lane's prefill waits while a walk runs (`context_blocks`).
  - **Two prompts at once (2026-10-08).** A walk first started only when no other lane was
    prefilling, so two long prompts prefilling together both ran chunk-major, re-streaming every
    expert per call: ~2.2K tok/s each, against ~6.9K for one (31 of 244 busy minutes of the
    2026-10-08 serve log were in that state). A walk now starts beside another prefilling lane,
    which waits for its span, and the spans of the two prompts alternate; a held lease without a
    walk area for the span (the other lane's chunk-major calls lent it) goes back and is lent again
    with one, between calls. Measured (`fn/rigs/trim/walk_ab.py`, deployed launcher settings,
    pairs of fresh ~90K prompts sent together, ABBA, two arms each): a pair that prefills together
    takes 26.1 s instead of 72.5-73.1 s (each prompt 6.9-7.0K tok/s instead of 2.3-2.6K); pairs
    whose prompts did not overlap are unchanged; greedy text equal in all 6 prompts of every arm.
    The two prompts finish together (TTFT ~25 s each) where finishing the older first would give
    ~13 and ~26 s; the scheduler's round-robin decides that order.
  - A Vision window cannot take the lease back from a walk.
  - The LFRU state is credited once per span (from the last chunk's routes) instead of once per
    call; every chunk routes nearly every expert, so the credited set is about the same.
- **Toggle.** A temporary environment variable (`INFERNIX_TMP_WALK_OFF`) gave the A/B in one
  binary; it was removed before the commit.

#### F5: QSA prompt attention on Tensor Cores

- **Route.** Exactly the calls the FP32 kernel would run unsplit (`ops/qsa/qsa_prompt.{h,cu}`,
  inside `qsa_attention` after the selection): columns × KV heads above 170, i.e. 86+ columns.
  Decode, verification and short suffixes keep the FP32 kernel.
  - **Why this rule and not a width.** Within that class a column's FP32 result never depended on
    its call's width (the kernel's only width dependence is its split count). The prefix cache
    relies on that: a resumed prompt can compute a position in a call of another width than the run
    that captured the state.
  - The prompt kernel is width-independent too, so the class keeps the property.
  - **Measured failure that set this rule.** A first version used a 256-column threshold inside the
    class. `infernix_qwen4_exp_prefix_cache_real_test` X12 (a restored Host-tier resume against the
    capturing Engine's turn) then failed deterministically: the first token difference came at 30,
    in 2 of 2 runs with the route on and 0 of 2 with it off.
  - `test_qsa.cpp` now also checks that a 100-column call reproduces the same columns of a wider
    call bit for bit.
- **Kernel.** One CTA per (column, KV head):
  - The KV head's 12 query heads are padded to 16 MMA rows.
  - Each warp sweeps its own 16-token tiles of the column's attended list, in the decode kernel's
    order, gathered by `cp.async` into double-buffered swizzled stages.
  - Each warp keeps its own online softmax; the CTA merges the warps' (max, sum, rows) at the end.
  - INT8: 4 warps. BF16 storage: 2 warps, for shared memory.
- **Arithmetic.**
  - Scores: FP32-accumulated m16n8k16. INT8 multiplies the Hadamard-rotated query rows (rounded
    to FP16 once, in the k order that one `ldmatrix` lane of a code row supplies) by the codes
    widened exactly, with the group scales applied per 64-dimension FP32 partial. BF16 storage
    uses BF16 MMA on the stored keys.
  - P × V: FP16 probabilities times FP16 values (INT8: codes times their represented group scale,
    via `ldmatrix.trans`; BF16 storage: the stored FP16 values), FP32 accumulation.
  - A tile whose largest V scale exceeds 256 decodes with the scales divided by 2^s and multiplies
    the probabilities by 2^s (both exact).
- **Qualification.** `test_qsa.cpp`'s FP64 oracle at the same criterion (relative L2 2.8e-3).
  Prompt-route cases: a dense call, the dense/selected boundary inside a call, selected blocks
  over several pages in two rows, and a list length off the tile grid.
- **Expected** (estimate): the 245 ms per call of FP32 SIMT attention falls to ~30-60 ms (MMA work
  ~1.4 ms per layer at peak; the gathers are L2-served up to ~32K of context).

#### F6: vectorized `split_rows`

`split_kernel` copied one BF16 element per thread: ~0.8 ms per prefill split, ~285 GB/s. Where
every boundary, the row count and every pointer are 8-element aligned (all of Qwen4Exp's splits),
a thread now copies 16 bytes. The copy is exact; `test_rows.cpp` checks both routes.

#### F4-F6 speed (2026-10-05, RTX 5090, Windows, INT8 KV, `fn/rigs/layer/speed.bat`)

**Arms and workload.**
- **Arms:**
  - OLD = dev aa6a85985.
  - OFF = the new binary with the walk and the F5 route toggled off (F6 and F7 stay on).
  - ON = the new binary.
- **Workload:** `infernix_bench` over the long-docs corpus, one process per arm and pass, passes in
  ABC CBA order.
- **Configurations:**
  - "long": `--max-ctx 132096`, chunk 4096.
  - "short": `--max-ctx 20480`, chunk 4096.
  - "c1024": `--max-ctx 20480`, chunk 1024.

**Results** (prefill tok/s, mean of 2):

| Case | OLD | OFF | ON | ON / OLD |
|---|---:|---:|---:|---:|
| long pp131072 | 1,828 | 1,706 | 3,373 | 1.85× |
| long pp65536 | 1,727 | 1,614 | 3,238 | 1.88× |
| long pp16384 | 1,123 | 1,093 | 1,994 | 1.78× |
| short pp16384 | 2,207 | 2,189 | 5,680 | 2.57× |
| short pp4096 (one chunk: no walk) | 1,551 | 1,636 | 1,637 | 1.06× |
| c1024 pp16384 | 521 | 512 | 2,420 | 4.64× |
| tg512 | 110.0 | — | 110.1 | 1.00× |

**Reading the results.**
- **Spread.** The two passes of long pp131072 ON differ by 8 % (3,235 and 3,511). The other rows
  are within 4 %.
- **OFF is slower than OLD in the long case** (−5 to −7 %).
  - Cause one: F7's slower n-gram reads (F7 is reverted since).
  - Cause two: the OLD binary lends a smaller ring. That cap was raised (769 → 1,025 lent frames)
    between the builds.
  - The split between the two causes is not measured. The post-build rig re-measures without F7.
- **What remains after the walk.** At 128K, the profile's late chunks were link-bound: ~2.4 s of
  wall time against 0.8 s of GPU busy time. With the walk, that busy time is the bound. Its largest
  parts per 4K chunk are QSA attention (248 ms, which F5 targets), q8 GEMMs (176 ms), elementwise
  work (107 ms, mostly split, which F6 targets), experts (111 ms) and QSA selection (up to 86 ms).

#### F9: the n-gram volume is no longer mapped (2026-10-05)

**Finding.**
- **Profile.** In the 128K profile of the layer walk, each 64K span began with ~11 s of host
  staging while the GPU was idle. Temporary timers split it: 99.8 % was n-gram NVMe reads, 1.27M
  reads at ~17 µs each (~60K reads/s). The chunk-major build showed the same per-read cost; there
  the reads hid behind compute.
- **Probe.** `fn/probes/ioring_bench.cpp` does random 4 KiB unbuffered reads of the real volume
  from one thread at depth 64:

  | Path | Reads/s |
  |---|---:|
  | Overlapped `ReadFile` | 213-228K |
  | IoRing | 223-234K |
  | Either, with 40 GiB pinned | −4 % |
  | Either, with the file also mapped | 49-61K |

- **Cause.** `ReadOnlyFile` always mapped the whole 52 GB file. NgramVolume never reads the
  mapping, but on Windows every unbuffered read of a file with a mapped data section pays cache
  coherency work.

**Change.**
- `ReadOnlyFile` takes a `FileMapping` mode; `None` maps nothing and opens no buffered handle.
  POSIX honours it too.
- NgramVolume opens its volume with `None`.
- `test_read_only_file.cpp` checks `None`: nothing mapped, the live size, and the same bytes
  through both read paths.

**Result** (`fn/rigs/layer/diag2.bat`, 128K prompt, chunk 4096, one run each, against the same
binary before the change):

| | Before | After |
|---|---:|---:|
| Prefill (tok/s) | 3,255 | 5,331 (+64 %) |
| n-gram read time | 21.6 s | 5.9 s (~213K reads/s, the probe's rate) |
| Staging per span | 10.4-11.2 s | 3.0 s |
| Reads behind the decode gate (169) | 8.2 ms | 2.1 ms |

- The new binary also has F8, worth ≤ 0.6 s of the 15 s saved.
- IoRing was not adopted: it adds only ~5 % over overlapped `ReadFile`.

**Remaining.** About 6 s of reads per 128K prompt was still exposed under the walk (3 s per span).
F7, re-applied, cut that to 1.3 s. Overlapping a span's reads with the previous span's compute
would hide part of the rest, but it is worth under 3 % now.

**Optane against a consumer NVMe drive** (`fn/rigs/layer/ssd.bat`, 2026-10-05):
- **Setup.** The same binary with F7 and `--vram-past-budget`, and the volume copied to a
  Samsung 990 PRO 2 TB (D:). ABBA order: Optane, 990 PRO, 990 PRO, Optane.
- **Results:**

  | | Optane P5800X | 990 PRO |
  |---|---:|---:|
  | Probe: 1 thread at depth 64 | ~220K reads/s | ~205K reads/s |
  | Probe: 8 threads at depth 16 | 1.17M reads/s | 1.11M reads/s |
  | pp131072 (tok/s) | 5,767 / 5,770 | 5,766 / 5,768 |
  | pp16384, cold (tok/s) | 3,680 / 3,713 | 3,719 / 3,723 |
  | 128K n-gram read time | 1.13 s | 1.17-1.18 s |
  | tg512 (tok/s) | 111.4 / 111.3 | 111.3 / 111.3 |
  | Load time | 25.1-25.4 s | 25.1-25.4 s |

- **Decode gate.**
  - Per-read latency behind the gate: ~60 µs on the Optane, ~200 µs on the 990 PRO.
  - The GPU waited in 20 of 295 cold rounds on the Optane (mean 28-33 µs) and in 34-44 on the
    990 PRO (54-68 µs): milliseconds per generation.
- **Conclusion.** With F9 and F7 the volume's drive no longer matters on a fast NVMe SSD. The user
  guide now says so.

**The artifact loader as well.**
- **Change.** `InputFile` opens its parts with `FileMapping::None`. `read_exact` (metadata and
  host objects) now reads unbuffered: straight into an aligned destination, else through an 8 MiB
  aligned bounce buffer per thread. The bulk reads were already unbuffered. The artifact reader and
  materialization tests pass.
- **Probe** (`fn/probes/bigread_bench.cpp`, 8 MiB sequential reads, two in flight): a mapped file
  cut the rate 12 % (6.04 against 6.9 GB/s).
- **Load time: no measurable change.** `load_seconds` is bimodal across today's runs with either
  loader: ~21 s or ~25 s, by run time, with the old dev binary too. The four runs of the new loader
  were 25.2-25.8 s. Loading is bound by something other than the reads.

#### F10: q8 MMA tile selection for prefill widths (2026-10-05)

**Problem.** In the 128K profile the q8 GEMMs were 5.65 s, 31 % of GPU time. Every Flash-Next
shape used `r32_t128` above 96 columns:

| Shape | Projection | Rate |
|---|---|---:|
| 324 × 10240 | hyper-connection down | 96 TFLOP/s |
| 10240 × 320 | hyper-connection up | 110 TFLOP/s |
| 16384 × 2560 | GDN query | 169 TFLOP/s |

The 324-row shape had 352 CTAs, against 340 resident.

**Probe.** `tests/ops/linear/tmp_q8_tiles.cpp` (temporary; `fn/rigs/layer/q8tiles2.bat`) ran
every MMA tile launcher on the seven prefill shapes at 128-4096 columns.
- **Bits.** Every tile's output is bit-identical to `r32_t128` at every point. An MMA tile
  accumulates each output over K in the same order, so the selection is speed-only.
- **Best tiles** (change against `r32_t128`):

  | Shape | 128 columns | 4096 columns |
  |---|---|---|
  | 16384 × 2560 | r64_t128, −17 % | r64_t128, −18 % (1,623 against 1,979 µs) |
  | 12800 × 2560 | r48_t64, −20 % | r64_t128, −19 % |
  | 1280 × 2560 | r32_t64 to 512 columns | r64_t128, −16 % |
  | 2560 × 6144 | r32_t64 to 256 columns | r64_t128, −16 % |
  | 2560 × 640 | r32_t64 to 256 columns | r64_t128, −14 % |
  | 10240 × 320 | r48_t64 to 192 columns | r64_t128, −17 % |
  | 324 × 10240 | r32_t64 to 1024 columns, −7 to −30 % | r48_t64, −30 % |

**Change.** The shapes' selectors (`src/ops/linear/q8/shapes/*.cu`) use these tiles above 96
columns. Widths of 96 and below, where decode and verification run, are unchanged.
`infernix_linear_q8_a16_test` passes.

**Expected.** ~30 ms per 4096-token chunk, ~1 s per 128K prompt.

**Result** (`fn/rigs/layer/tiles_ab.bat`; A = F7+F8+F9, B = A + F10; ABBA, `--vram-past-budget`):

| | A | B | Change |
|---|---|---|---:|
| pp131072 (tok/s) | 6,594 / 6,589 | 6,891 / 6,920 | +4.8 % |
| pp16384 (tok/s) | 6,087 / 6,071 | 6,076 / 6,083 | 0.0 % |

- The ~40K prompt's greedy ids are identical in both arms.
- At 16K the walk's four chunks are bound by something other than these GEMMs; not attributed.

#### F8: the router projection for prefill-wide calls

**Problem.** `projection_fp32` (router logits, FP32 SIMT, a bit-invariance contract) took 755 µs
per 4096-token call: 513 rows, K = 2560, 36 ms per chunk over 48 layers. Its narrow mapping
re-reads a CTA's weight rows for every 8 columns.

**Design.** Calls of at least 64 columns use a tiled kernel.
- **Tiling.**
  - Each CTA (512 threads) owns a 32-row × 16-column output tile.
  - K is walked in slices of 256 elements, staged with double-buffered `cp.async`.
  - A slice is exactly one 8-element chunk per lane, so lane l's partial is the narrow kernels'
    lane-l partial, in the same order.
  - A warp owns a 4 × 8 sub-tile and reduces it by the same butterfly.
- **Bits.** An output's summation is unchanged, so its bits are unchanged by construction.

**Rejected versions.**
- Weights staged once per CTA with x read from global: 918 µs, measured beside another GPU job.
- A 1,024-thread CTA: 1,615 µs. Its 64-register cap made the 4 × 8 tile spill.

**Test.** `test_projection_fp32.cpp` checks every column of the wide calls bit-for-bit against the
narrow mapping. Shapes:
- 513 × 2560 at T = 64 and T = 4096;
- four segments at K = 3072, T = 301;
- 129 × 1024 at T = 135.

It also checks the FP64 bound on edge columns.

**Result.** 370 µs per 4096-token call, measured with CUDA events on an idle GPU, against 755 µs
in the nsys profile. That is 2.0× faster, saving ~18 ms per chunk (~0.6 s per 128K prompt). The
end-to-end effect is not measured yet.

#### VRAM item 1: the prefill workspace lent from frames

**Design.**
- The static workspace covers only `max(lanes × W, min(chunk, 512))` columns.
- A wider call swaps in a wide arena, taken from the first of these that is available:
  - the stream lease (after the ring);
  - a frame lease of its own;
  - a `DeviceBuffer` fallback.
- A temporary toggle (`INFERNIX_TMP_WIDE_WORK_OFF`, removed before the commit) restored the static
  arena for the A/B.

**Results** (`post.bat`, `--max-context 49152`, chunk 4096):

| | Static arena (toggle off) | Wide arena (two runs) |
|---|---:|---:|
| Workspace (MiB) | 1,774 | 284 |
| Expert frames | 8,734 | 9,299 (+565, +1.5 GiB) |
| tg512 (tok/s) | 110.1 | 110.5 / 110.7 |
| pp16384 (tok/s) | 5,602 | 5,674 / 5,707 |

- The ~40K prompt's ids are identical in all three runs.
- The tg512 and pp16384 differences are within one run's spread at this short context, so they
  are not claimed as gains.

#### VRAM item 2: the ~1.6 GB "in use before loading" (2026-10-05)

**Probes.** `fn/probes/vram_probe.cu` and `vram_alloc_probe.cu`, plus a temporary ledger line,
gave these readings:

| Situation | Reading |
|---|---|
| nvidia-smi, idle GPU | 0 MiB used, 420 MiB driver reserve |
| A bare CUDA 13.4 context held | 432 MiB used (nvidia-smi); 1,588 MiB used (`cudaMemGetInfo`), exactly the engine's figure |
| Pinning host memory (16 GiB) | no change |
| `CUDA_MODULE_LOADING`: lazy (the default) or `LAZY` | 1,588 MiB |
| `CUDA_MODULE_LOADING=EAGER` | 1,814 MiB |

**The cost is not Infernix's.** It is the WDDM context (~432 MiB), the driver's reserve (420 MiB),
and ~736 MiB that `cudaMemGetInfo` reports as unavailable.

**The unavailable part is usable.**
- `cudaMalloc` handed out 31,616 MiB in 64 MiB pieces, against 30,991 MiB reported free.
- Every piece was written at VRAM speed (≤ 0.10 ms per piece), so no sysmem fallback occurred.
- That leaves ~625 MiB (~236 expert frames) unused. The engine sizes frames from the reported free
  memory, minus headroom and reserve.

**Now an opt-in: `--vram-past-budget`** (the user's decision, 2026-10-05).
- **Finding.** `cudaMemGetInfo`'s free memory is exactly DXGI's budget minus usage: a budget of
  31,419 MiB, against 32,187 MiB of NVML total less the driver reserve.
- **Allowance.** The VRAM source measures one allowance at open: NVML usable − budget − 128 MiB,
  640 MiB here. It reports budget + allowance as the budget, and free memory derived from it.
  Sizing, the control law and the spill guard are unchanged. An OS budget cut still lowers the
  effective budget by the same amount.
- **Results** (`vf.bat`):

  | | Off | On |
  |---|---:|---:|
  | Expert frames | 9,299 | 9,542 (+243) |
  | "In use before loading" | 1,588 MiB | 948 MiB |
  | tg512 (tok/s, one process per arm, within noise) | 110.3 | 111.7 |

  - The ~40K prompt's greedy ids are identical off and on, and identical to `post.bat`.
- **Off by default**, since that memory may be moved to system memory when another program needs
  VRAM.

---

### 19.3.9 Host time between decode rounds (2026-10-05)

**Measured** (n-gram S0 phase timers, warm tg512 plain, per round of ~11.6 ms, dev before S3/S4):
the GPU idles ~0.45 ms per plain round while the host works: ~0.21 ms in `settle_round` (the
round's `after_round`: route statistics, the LFRU policy, landing adoption, table upload,
promotions) after the commit's GPU work, and ~0.25 ms between staging and the graph launch
(io upload, `before_round`, `cudaGraphLaunch`). That is ~4 % of a plain round; MTP rounds pay
the same per round (~1-2 % of a 22-28 ms round). n-gram reads add to the second part on cold
text (S2/S4b move them behind the launch).

**Step 1, done: the LFRU victim search.** `LfruPolicy::select_victims` computed every resident's
score (~9,000, a division each, through scattered `count_`/`last_` loads), built a (score, key)
pair per resident and partially sorted them, each time it needed a victim; the budgeted steady
state asks for one victim per admitted expert, in many layers per round. Most of the cost was the
scattered score computation, not the sort: a one-victim scan alone measured the same. The policy now
keeps each resident's count and last use as doubles in arrays parallel to the resident list (exact:
both are integers below 2^53, so every score is bit-identical to `score()`), computes all scores in
one contiguous loop, marks protected residents, and takes the minimum score and then the lowest key
holding it (one victim) or partially sorts as before (several). Decisions are unchanged: the
conformance test against the replay tool (1,200 groups, 0 mismatching steps) and a single-victim
oracle in the budgeted test pass. A budgeted step at 9,000 residents (host test, 2,000 steps)
went from 22.8 to 10.9 us. Engine effect (Gold port, `fn/rigs/short/combo.bat`, ABBA, the build
before and after): tg512 110.53 / 110.67 → 110.57 / 110.66 tok/s, MTP tg512 140.65 / 140.80 →
140.81 / 140.61, code MTP decode 134.6 / 134.7 → 134.6 / 135.2: neutral within noise (the estimate
was ~0.1 ms per round, ~1 % of a plain round). Kept: decisions are identical and the host step is
half as long.

**Step 2, proposed: pipelined decode.** The next round's input token is on the device when the
round ends; only the host's sync, commit and staging stand between rounds. Launching round r + 1
before the host has committed round r removes the whole gap. Design sketch:
- The decode graph reads its input ids from the sampled-token buffer (a device copy into the io
  ids at the top of the graph), so round r + 1 can be enqueued right behind round r's sampling.
- n-gram rows of round r + 1 depend on token r: they go behind the S2/S4b gate, read once the host
  has token r.
- `after_round(r)` runs on the host while round r + 1 runs: promotions and evictions it decides are
  published one round later. The route log needs two host buffers (round r + 1's download must not
  overwrite round r's before it is read), and S4 landing reservations must not be released by
  round r + 1's `before_round` before round r adopts them.
- A round launched speculatively behind a stop (EOS, a stop string, cancellation) must be
  discarded: its KV cells and recurrent state writes are past the committed frontier, so the lane's
  state must not be published or captured (prefix cache, state image) until the discarded round has
  finished and the state has been restored or the lane released.
- The runtime contract changes: today the worker alternates `decode` and `commit`; a pipelined
  Program needs `decode(r + 1)` admissible before `commit(r)` returns, with the scheduler's lane
  changes (admissions, finishes) applied only at a drained boundary.
- Expected: ~0.4 ms per plain round (+3-4 % plain decode), ~0.3-0.4 ms per MTP round (+1-2 %).
  MTP rounds also wait on the drafts' host round trip (n-gram S3, device-assembled verification).
- Not started: it changes the runtime's execution contract and the S4/VRAM-resize boundary rules,
  so it needs its own reviewed plan before code.

**Option B, built and measured, not kept (2026-10-05/06).** Within the existing contract, one-row
MTP rounds were pipelined in two parts: B1 chained the drafter into the verification on the device
(the host no longer waits for the drafts before launching the round), and B2 committed a round
ahead into a spare recurrent-state slot (a lane-to-slot indirection) with the next drafter launched
behind it (~420 lines in `program_impl.h`).

- Correct: greedy ids identical with B off and on (code, the long prompt through a layer walk, a
  37-token limit, a stop string, a thinking budget), the prefix-cache real test 27/27 in both arms,
  and serve at C = 2 identical to C = 1 (8/8).
- Speed on the Gold port (`fn/rigs/pipe/pipegold.bat`, one binary with temporary toggles):

  | | off | B1 | B1 + B2 |
  |---|---:|---:|---:|
  | Code MTP decode, CLI (3 runs each) | 135.47 | 135.00 | 135.10 |
  | tg512 MTP, bench (2 x 3 runs each) | 140.36 | 140.39 | 141.83 |

  On dev the same split had been tg512 +1.4 % and code decode neutral.
- B1 gains nothing (likely because the drafter is short enough to finish while the host stages the
  round; not profiled). B2
  gains 1 % on the synthetic tg512 and loses 0.3 % on the real code decode (single-run cases lost
  1-1.5 %), for ~420 lines and a state-slot indirection in every path. Not kept; the branch
  `claude/fn-pipe-gold` holds it (with the drafter clamp fix and the layer walk's slot use).

### 19.3.11 Long-context KV in host RAM (§9.6 as built; VRAM item 3, 2026-10-05)

**Why.** The deployed configurations run `--max-context` 220K-500K at concurrency 2-4. At 220K × 2
lanes, INT8 KV is ~6 GB of VRAM, about 2,300 expert frames. QSA attends to at most 2,051 selected
tokens per query, so most of that KV is cold at any moment.

**Model: exclusive page placement in page spaces.**
- A 64-token KV page lives in exactly one space:
  - **space 0**, the device pool (`DeviceKVPagePool`, as today);
  - **space 1**, a pinned, GPU-mapped host pool with the same plane layout;
  - **space 2**, a device pool lent from expert frames during a long prefill.
- A block-table entry carries the space in its top bits (`page >> 28`); the low 28 bits index the
  space's planes.
- The pooled index-key plane is device memory in every space: space 1's pooled plane is a device
  array, since selection scans every pooled key.
- **Writers are unchanged.** KV append and pooled keys only ever write the frontier's pages, which
  are space 0 by rule.
- **Readers translate.** The page spaces are QSA-owned, so only QSA's kernels change:
  - the decode attention kernel (its `cp.async` tile gathers, §19.3.14);
  - the prompt kernel (its `cp.async` gathers; space-1 pages are read zero-copy);
  - the selection kernels and `pool_kernel` (pooled-plane offsets).
- **Space-1 pages are read zero-copy**, as §9.6 planned. A selected 4-token block of one layer is
  ~4.2 KB in INT8.

**Placement and migration** (Program; between calls and rounds, never inside a captured graph):
- **Admission.** A lane's pages are reserved in space 0 up to its device budget; the rest are
  reserved in space 1. The frontier page and the most recent pages are always space 0.
- **Long prefill.**
  - Each chunk (or walk span) writes space-0 pages.
  - After it, completed pages beyond the lane's device budget move to space 2 (D2D), so the prompt
    kernel's dense gathers stay on the GPU.
  - When the prompt ends, space-2 pages move to space 1 (D2H), except those the last chunks
    selected most, which take space-0 pages. The lent frames go back.
- **Decode: CLOCK over pages.**
  - A small kernel after each decode call's selection sets one reference bit per page whose blocks
    the call selected.
  - Between rounds the host promotes referenced space-1 pages into space 0 (H2D on the copy
    engine, a bounded number per round) and demotes unreferenced space-0 pages to space 1.
  - The block tables are updated, so the next replay reads the new placement.

**Interactions.**
- **Prefix cache.** Cached blocks are space-0 leases, as today. A lane's space-1 pages are copied
  to the prefix Host tier's slabs when published (they are already host bytes), never shared
  zero-copy. Restores write space-0 pages.
- **RAM ledger.** The host pool is pinned and counted. The tier turns on only when
  `max_context × concurrency` KV exceeds the device budget.

**Gate K0, measured before migration code.** Decode selections at 64K and 128K context
(a temporary trace, `INFERNIX_TMP_QSA_TRACE`, kept as a patch on branch
`backup/fn-layer-prefill-wip-20261005`; traces in `G:\infernix\qsa_trace_*.bin`; `fn/rigs/layer/k0_sim.py`) give LRU hit rates for 4-token blocks against
64-token pages at 16K/32K/64K tokens of device cache per lane. Pages are the unit if they reach ≥ 90 %
at a 32K budget. Otherwise a block cache inside space-1 pages is added (designed then).

**First: an elastic device pool (steps E1-E2, before K2).** The host tier pays only when the
admitted contexts' KV exceeds the device. The deployed configurations rarely fill their
`--max-context`, yet a fixed pool takes its whole maximum from the expert frames at startup. The
elastic pool does the same job with no kernel change:
- **E1, the pool.** `DeviceKVPagePool` gets a backed-page limit.
  - Pages `[0, backed)` can be handed out; the rest of the capacity has no memory behind it.
  - Allocation is already first-fit over ascending pages, so a pool keeps its pages low and its
    top frees up.
  - The limit rises freely. It falls only over free pages (`can_back`).
- **E2, the Program.**
  - **Backing.** The KV backing is a `VmmRange`: the address space for the whole plan is reserved,
    and 2 MiB chunks are mapped individually. The planes are page-major, so pages `[0, n)` are a
    prefix of every plane.
  - **Base.** Startup maps the block tables and 512 base pages (32K tokens). The plan counts only
    those, so the frames are sized with the rest free.
  - **Growth.**
    - When an admission's reservation is short after cache eviction, the pool grows in 64-page
      units.
    - Growth first uses memory the control law would give the frames anyway. The rest it takes from
      the frames (`ExpertResidency::resize`, one 64 MiB frame chunk over so that much is
      unmapped), at most a quarter of them per growth.
    - Growth is refused while frames are lent. The quote and the prefix selection count what one
      growth could add.
  - **Shrink.**
    - A release lowers the limit to the free top and gives whole frames back at once.
    - After 60 s with every lane free, the maintenance pass releases the cache's idle,
      host-backed device blocks (a later match restores them from the Host tier; nothing is lost),
      then shrinks. The delay spares a follow-up turn a restore plus an expert reload.
  - **The control law.** `VramControl::account_fixed` moves its sizing baseline by each grow or
    shrink, so the KV is never read as the reserve's use.
  - **Toggle.** A temporary `INFERNIX_TMP_KV_ELASTIC_OFF` restored the fixed pool for the A/B; it
    was removed before the commit.
- **Known costs.**
  - A growth synchronizes the device and evicts experts at admission: a few ms plus refills.
  - The idle eviction releases host-backed blocks in the index's value order, not by page
    position, so it can release more than the top needs.
  - A long conversation that pauses for more than 60 s restores its prefix from the Host tier
    (~90 ms for 128K INT8) and reloads the evicted experts.
- **Gate.**
  - `infernix_kv_cache_test` covers the backed limit and the `VmmRange`.
  - The ~40K prompt at `--max-context 262144` must give the same ids elastic and fixed, and the
    same ids as at 49152.
  - tg512 at `--max-ctx 262144`: elastic against fixed, frames and tok/s.
- **Results** (2026-10-05, `fn/rigs/layer/post.bat`, INT8 KV, one run each):
  - `infernix_kv_cache_test` passes.
  - The ~40K prompt gives identical greedy ids in every arm: 49152 and 262144, elastic and fixed.
  - At `--max-context 262144`:

    | | Fixed | Elastic | Change |
    |---|---:|---:|---:|
    | Startup KV (MiB) | 3,360 | 458 | |
    | Expert frames | 8,197 | 9,299 | +1,102 (+2.8 GiB) |
    | tg512 (tok/s) | 103.5 | 109.6 | +5.9 % |
    | 40K-prompt prefill (tok/s) | 2.83k | 2.91k | |

  - The 40K request grew the pool to 704 pages (626 MiB), taking 88 frames.
  - The release shrank it back to 512 pages and returned 63 frames at once.
  - The remaining 25 frames (the chunk taken over the need) wait for the control law's grow delay.

**Steps.**
- **K1:** page spaces in the QSA ops plus their op tests (spaces 0/1/2, every reader, oracle =
  the same KV in space 0).
- **K2:** host pool, admission placement, and exact outputs with the tier forced on: every
  non-frontier page in space 1, greedy ids equal to the device-only run.
- **K3:** long-prefill space 2 and the end-of-prompt migration.
- **K4:** decode CLOCK.
- **K5:** prefix-cache publish and restore.
- **K6:** measurement — frames gained, decode at 4K/64K/128K/220K context against the device-only
  build, and the prefill rate.

### 19.3.12 Short prompts: the link, and a CPU share of the experts (2026-10-06)

**Diagnosis** (Gold port, `infernix_bench`, chunk 4096, INT8 KV, `fn/rigs/short`).

- Prompts of 512, 2,048, 4,096 and 16,384 tokens prefill in 2.57, 2.37, 2.28 and 2.48 s. The cost is
  nearly constant across lengths.
- In nsys, the measured pp4096 call takes 2,563 ms, of which 2,556 ms are H2D copies. Its 411 ms of
  kernels are hidden under them.
- pp16384 is four walk steps of about 615 ms each, every one a copy of 12 layers' experts. Its
  ~1.7 s of kernels are again hidden.
- A wide call routes nearly every expert, so the stream copies one pass of the experts (up to
  63.3 GiB, 24,576 records) at 27.4-27.6 GB/s. That is the rate of PCIe 5.0 x8: this machine gives
  8 of the CPU's lanes to a network card.
- Every prompt up to the 64K walk span therefore pays about 2.3 s, minus what the expert cache holds.

**Requirement: scale with the link.**

- Another machine may run x16, about twice the rate, so nothing may be tuned to x8.
- The Program measures the host-to-device rate at startup, by copying expert records from the
  pinned banks into the miss staging slots before any call uses them: best of three, 64 records,
  a few tens of milliseconds (`core/link_probe`; `ProgramOptions::link_bytes_per_second` fixes it
  in tests). The diagnostic line `host-to-device link: ...` reports it and the choices below.
- It rescales the link-bound costs from the reference rate at which they were fitted (27.5 GB/s): the
  prefix cost's restore rate and the walk-span cost.
- The decode miss split keeps misses / d of a call's misses on the link, d = max(1, floor(1.65 + C t))
  for C experts per second on the CPU team (one-column jobs) and t seconds per record on the link.
  1 + C t balances the two parts; the switches sit between the measured fastest divisors (below):
  d steps from 2 to 3 at C t = 1.35, midway between 3 workers (1.0, d = 2) and 6 workers (1.5-1.7,
  d = 3), so the startup measurement's noise (6 workers measured 15,000-17,200 experts/s) stays
  clear of it; round(1 + C t) put that switch at 1.5, inside the noise. A fractional share
  1 / (1 + C t) was tried and dropped: at a share near 1/2 the noise flips floor(M share) for M = 2
  between startups (3 workers: 100.8 against 107.5 tok/s plain).
  `cpu_pcie_divisor = 0`, the default, derives d; a positive value fixes it.
- The CPU team's decode rate C is measured at startup for the configured workers
  (`program/cpu_rates.cpp`, ~30 ms): the best of 3 batches of 8 rounds of 16 one-column jobs on
  real records from the pinned banks (across layers, past the last-level cache) while the link
  copies records into the staging slots (the staging's DRAM reads compete with the CPU's). Without host records (the SSD tier) the
  i9-13900K's 6-worker rate stands (21,000 experts/s). The prefill CPU split below keeps its fitted
  rates; on measured ones (lower under the link's load: 6 workers 17,000 experts/s and 46,000
  expert columns/s against 21,000 / 66,000) it is not yet qualified.
  Measured (2026-10-09, tg512 cold, Dense8, two passes in opposite orders). With fixed divisors the
  best divisor follows the CPU: 6 workers d = 3 (plain 116.4 tok/s, MTP 167.1; d = 2 152.4 MTP), 3
  workers d = 2 (107.3 / 150.3; d = 3 144.6 MTP), 2 workers d = 2 (100.3 / 138.0; d = 3 120.8 MTP);
  the old rule (d = 3 at x8 whatever the CPU) costs 3 and 2 workers up to 12 %. The measured rates
  were 15,000-17,200 experts/s at 6 workers (C t = 1.5-1.7: d = 3), 9,700-10,000 at 3 (1.0: d = 2)
  and 7,000-7,700 at 2 (0.7-0.8: d = 2), so the derived divisor is the fastest one at each.
- Decode promotes into the expert cache every `4 × 27.5 GB/s / link` tokens once the frames are full
  (4 at x8, where it measured fastest; 2 at x16). A promotion moves the bytes of serving its expert
  once, and on a faster link the rounds leave the link idler, so promotions cost less wall time. The
  x16 value follows from that argument and is not measured.
- The CPU split below takes the measured rate as an input, and so does its width limit.
- Not link-dependent: the prefill promotions per layer (the stream has copied the experts already)
  and the walk's span (bounded by the lent workspace).

**The CPU's capacity** (`tools/flash_next_probe/host_probe`, i9-13900K, AVX-VNNI, 24 workers):

| | Rate |
|---|---|
| DRAM | 82-85 GB/s |
| W4A4 expert, 1-4 columns, cold records | ~30,000 experts/s (DRAM-bound) |
| 8 columns, cold | ~19,000-21,000 experts/s (compute-bound) |
| 8 columns, warm | ~150,000 expert-columns/s |
| The link, for comparison | ~10,000 records/s at x8, ~20,000 at x16 |

An expert with few columns is cheaper to compute on the CPU than to move.

**Placement invariance limits the split to n <= 8.** Experts with at most 8 columns in a call use the
canonical arithmetic (§16.2), which is bit-identical on the CPU and the GPU. Wider ones always take
the GPU's wide route. So the CPU may serve only experts with n <= 8 columns. The split then changes
no output, and greedy output cannot depend on the link or the CPU.

**Design: a gated stream for narrow single-call prompts (256 columns up to the width limit).**

1. After `moe_dispatch`, a small kernel publishes the layer's per-expert column counts to mapped
   memory, and the host waits for them (tens of microseconds).
2. The host chooses the CPU set. It takes non-resident experts with n <= 8, fewest columns first,
   while the CPU's predicted time stays below the link's time for the rest. Rates come from the
   startup link probe and a CPU rate that starts from the probe values above and follows the
   service times observed.
3. The stream copies only the GPU set. The CPU set stays misses, which `cpu_plan_kernel` hands to
   the CPU miss service, fewest columns first.
4. The miss service's limits grow to the call width: 4,096 columns, 512 jobs, and the mapped x
   buffer sized to match.
5. Copies start after the layer's router instead of two layers ahead. The link idles for the
   previous layer's MoE compute plus this layer's attention: about 2 ms against 20+ ms of copies
   at 512 tokens.
6. Wider calls keep the blind stream: the split takes calls of at most `cpu_split_columns` (1,536)
   × 27.5 GB/s / link columns (1,536 at x8, 768 at x16). The bytes gating saves cost less on a
   faster link, while the idle link between layers does not shrink.

**Measured** (2026-10-06, Gold port, `fn/rigs/short/split.bat`, one binary with a temporary
toggle; off = blind stream, on = gated + CPU split at every width up to 4,096):

- Greedy ids identical off and on for chat prompts of ~600, ~2,000 (prose, code) and ~4,000 tokens,
  plain and MTP. The prefix-cache and `/v1/decide` real tests pass with the split on. tg512 110.2
  tok/s (decode does not use it).
- `infernix_bench`, warm expert cache, chunk 4096, ABBA (two runs per arm, mean of 3):

  | Prompt | Off | On | Change |
  |---:|---:|---:|---:|
  | 512 | 2.46 s | 1.57 s | 1.57x |
  | 1,024 | 2.27 s | 1.75 s | 1.30x |
  | 2,048 | 2.18 s | 2.14 s | 1.02x |
  | 4,096 | 2.05 s | 2.35 s | 0.87x |

- CLI, cold expert cache (first call after load, two runs per arm): 512 tokens 146 -> 179 tok/s
  (1.23x); the 2,000-token prompts 0.96x and 0.99x; 4,000 tokens 1.00x.
- So the gated stream pays where a call leaves many experts untouched or narrow and loses at 4,096
  columns, where nearly every expert is touched and the idle link between a layer's routing and
  its copies costs more than the bytes saved. The width limit (1,536 at x8) keeps 512-1,024-column
  calls on the split and leaves 2,000 and above on the blind stream. The x16 limit (768) follows
  from the same model (saved link time halves, the idle time does not) and is not measured.
- With the limit (`fn/rigs/short/combo.bat`, same method): ids identical for the four prompts;
  pp512 2.463 → 1.567 s (1.57x), pp1024 2.272 → 1.752 s (1.30x), pp2048 and pp4096 unchanged
  (2.178 / 2.054 s, the blind stream); CLI cold 512 tokens 147 → 179 tok/s, the 2,000- and
  4,000-token prompts 0.99-1.00x. The prefix-cache and `/v1/decide` real tests pass.

**N-gram reads and the walk (item 3, assessed 2026-10-06; the overlap is not built).**

- A walk span's chunk needs its n-gram rows before its layer-0 pass. Each 4,096-column chunk reads
  about 74,000 rows (~37 ms on Windows), while that pass takes a few milliseconds. Within a span,
  the reads therefore cannot hide behind compute.
- Across spans, the next span's rows could be read while the current span's later layers run: the
  engine thread waits there anyway.
- But a 64K-token span needs about 1.15 million rows. The direct-mapped host cache holds 2^20 rows
  (~170 MB), so a span prefetched into it would evict itself. A separate store would add ~190 MB of
  host RAM.
- The saving is at most about half of the ~1.2 s of reads at 128K (about 3 %), and nothing for
  prompts up to 64K. It is not worth that RAM.
- The large n-gram cost was Linux's serial direct reads: 8 reads in flight against 128 on Windows,
  5.8 s against 0.5 s for a 40K prompt. Kernel AIO fixed it (`read_only_file_posix.cpp`, 9.5x more
  random 4 KiB reads in WSL2).

**Expected gain** (cold cache, CPU rate derated for the DRAM the DMA shares):

- 512 tokens: about 2.5 s to 1.0 s at x8, and about 1.75x at x16.
- Measured routing (route trace of chat prompts, calls of 524-1,024 columns):
  - such a call touches 323-355 of the 512 experts per layer;
  - 42-54 % of the touched experts have n <= 8 (21-65 % by layer).
- The blind stream copies every non-resident expert, touched or not. The gated stream copies only
  touched ones, which alone saves about a third of the link time at these widths.
- With the CPU split, the modelled link-plus-CPU time per call falls 1.7-2.2x against copying the
  touched experts. That is about 2.5-3x against today's full pass, at x8 and x16 alike.

**The split's rates (2026-10-10, `ProgramImpl::SplitPolicy`).** The split now plans with the CPU team's
rates measured at startup with the link busy (`cpu_rates.cpp`: one-column jobs for the per-expert cost,
8-column jobs for the per-column cost), at `kSplitRateShare` = 0.3 of them. Measurements were taken on
Dense8, int8, total time per request, 10 repetitions per arm, i9-13900K with x8 and 6 workers unless noted.

- **The fitted rates overloaded the CPU.**
  - The rates were 21,000 experts/s and 66,000 columns/s. Scaling both down cut short prompts 5-6 %.
  - The best scale fell as calls widened: 0.35 at pp256, 0.2-0.35 at pp512, 0.1-0.2 at pp1024.
  - Turning the split off cost +15 / +7 / +2 %.
- **The probe's rates alone are not enough.** It measures 16,600 experts/s and 47,300 columns/s, and those
  rates gave only −3.3 / −0.3 / −0.1 %.
- **A per-layer trace explained it.**
  - The CPU estimate is 15-40 % low.
  - The GPU waits nearly the whole CPU time (pp256 layer 0: CPU 13.1 ms, wait 11.8 ms). Its own side takes
    1-2 ms, where the model charges a record's link time per kept expert (11-37 ms), because the stream
    copies most experts ahead.
- **A fitted model failed.** A line fitted to measured layer times (fixed plus per kept expert) starved the
  CPU instead (pp256 +13 %). The kept experts' cost is not linear: experts already copied ahead are
  nearly free, the rest cost a full copy.
- **The 0.3 share against the alternatives:**

  | Arm | pp256 | pp512 | pp1024 |
  |---|---|---|---|
  | 6 workers, against the fitted rates | −4.5 % | −6.2 % | −4.4 % |
  | 3 workers, against the measured ones (9,700 / 24,200) | −11.1 % | −16.8 % | −18.9 % |

  Shares 0.4 / 0.5 were better only at pp256 with 6 workers.
- **The 1,536-column cap stays.** At scale 0.2, extending it to 3,072 / 4,096 cost pp2048 +2.4 % with a
  4,096-chunk prompt. At that width the stream of every touched expert already covers the call.

### 19.3.13 Preemption (2026-10-06)

Gold's Engine pauses the youngest resident when an older one cannot obtain the KV pages of its next
unit, and resumes paused requests oldest first. Qwen4Exp used to reserve a request's whole extent
(prompt plus every output token) at binding, so no unit ever ran short and nothing was paused: a
request whose extent did not fit waited at the queue head while the residents' extents were mostly
unwritten.

**Pages per round.**

- A binding reserves the prompt plus one round: `round_positions = max(W, k + 1) + 1` positions
  past the frontier, W the widest verification and k the MTP drafter's steps (the cells a round
  writes), within the extent.
- `reserve_units` grows a decode or control unit's lane to its frontier plus one round before the
  round: it evicts unpinned cached blocks, grows the elastic pool (§19.3.11), reserves and maps the
  missing pages at the end of the lane's block table. A shortage goes back to the Engine, which
  reclaims and then pauses the youngest resident.
- The extent check at admission is unchanged (`feasible`: the extent fits the pool on its own), so
  the oldest resident can always grow to its extent once the younger ones are paused.
- The MTP drafter clamps its cells to the lane's mapped positions: the shared prefix pages plus the
  private ones (`prefix.page_base + pages`). Before this change the clamp counted only the private
  pages, which with a long reused prefix put every draft cell on one position inside a shared cached
  page (fixed in its own commit). Measured with `infernix-serve` (MTP, 4 drafts): a 44,783-token chat,
  then its continuation resuming 44,942 cached tokens. The continuation took 98 rounds for 176 tokens
  before the fix (accepted per draft position 63 / 13 / 0 / 0) and 76 after (65 / 32 / 3 / 0), 22 %
  fewer, with identical tokens; the first, uncached request was the same in both builds.

**Pause.** `start_pause` runs at a committed boundary (no round or transaction open).

- It publishes the lane exactly as a consistent abort does (`prefix_finish`): the pending MTP cell,
  the remaining blocks, the state at the frontier as a Host-born snapshot, and write-through. A
  request the prefix cache does not publish (`allow_prefix_reuse` off, a non-reusable prompt)
  publishes nothing.
- Inside a layer walk's span the layers are at different positions: the walk is abandoned and
  nothing past the span's start is published (its taps were).
- The ResumeState keeps the ledger (prompt and committed output), the frontier, and the request's
  measurements and decisions (timings, speculative statistics, the drafter's acceptance policy,
  `/v1/decide` draws, cache and n-gram counters). The model state is not owned: the cache entry
  stays evictable, so the Engine sees no snapshot handle and records the Replay route.

**Resume.**

- A request paused inside its prompt binds its prompt again; the selection finds the state the pause
  published (or an earlier tap) and prefill continues.
- A request paused after its first token binds its ledger: lookup keys are the prompt's, chained
  over the output blocks as publication chains them (`ledger_prompt`), and the index may return a
  snapshot at the paused frontier itself. The deepest affordable one restores; prefill calls
  (no taps) replay the ledger from there to the frontier with sampling suppressed, after which the
  lane decodes from the ledger's next input. When the restored state is the frontier, nothing
  replays and the Engine counts a snapshot restore.
- A Vision item inside the replayed range is encoded again (the window is planned from the restore
  frontier); penalty counts are rebuilt from the committed output; the prompt readout is not
  repeated.

**Bits.** A restore from the published state is exact: the state image and the blocks are those the
lane held, so greedy output equals an unpaused run. A replay recomputes the replayed positions with
prefill arithmetic, which may round differently from the decode rounds that produced them, so the
tokens after a replay may differ from an unpaused run (as on Qwen3.5).

**Test.** `infernix_qwen4_exp_preemption_real_test` (two lanes, a 48-page pool, prompts of 1,024
tokens and 1,200 outputs each, plain and MTP): the exact path's outputs equal the solo runs; the
replay path keeps every token committed before the pause, completes at full length and reports
the tokens that still equal the solo run.

**Measured** (RTX 5090, dense8m, int8 KV, 2026-10-06): the older lane ran short at frontier 1,596-1,599
with the 50-page pool full and the younger one was paused once in every scenario.

- Exact path, plain and MTP: one snapshot restore; both requests' greedy outputs equal their solo runs
  (1,200 of 1,200 tokens).
- Replay path: one replay restore of ~1,600 tokens; plain decode still equalled its solo run (1,200
  of 1,200), MTP diverged after 572 tokens (replay arithmetic), and both completed at full length.
- `/v1/decide` and prefix-cache real tests pass. The decide test's prefetch scenario needed a longer
  holding prompt (3,300 tokens): a binding no longer holds a request's whole extent, so the waiting
  head would have bound at once.
- Speed against the Gold base (ABBA): tg512 110.73 → 110.70 tok/s, pp4096 2.470 → 2.459 s,
  MTP tg512 140.83 → 140.33 tok/s (-0.35 %, both pairs).

**Faults the tests found.** (1) A paused lane's pages become cached blocks pinned by their in-flight
Host writes; the older lane then found nothing to evict and no resident to pause, and the Engine
failed ("oldest resident cannot obtain its legal unit"). `reserve_units` and `hybrid_reclaim` now drain
pending transfers before reporting a shortage. (2) A resume whose cached state is the paused frontier
itself has no suffix to plan; it decodes at once.

### 19.3.14 QSA decode attention: staged tiles (2026-10-06)

**Profile** (nsys, Gold `283613d8b`, dense8m, `int8` KV, 128 decoded tokens after 8K and 128K
prompts; `fn/prof/dprof`). The attention family is 2.4-4.8 % of a decode token's kernel time. Its
largest part is K3b, `attention_kernel`:
- 45 µs per call, 12-13 calls per main forward: 545 µs of a 14.8 ms plain token at 8K;
- the same at 128K, since attention covers ≤ 2,051 tokens;
- about 15× its §8.3 budget of ~3 µs for 2.2 MB.

In the same profile, selection (`qsa_score_kernel` + `qsa_select_global_kernel`) takes 12 µs per
call at 8K and 24 µs at 128K, and `merge_kernel` 3.3 µs.

The old kernel had 66 CTAs (33 splits × 2 KV heads) at T = 1. Every key and value element was a
dependent chain: block-table entry → page space → code (and scale) load. The keys were fetched one
token ahead per warp and the values in batches of 8 cells, so a 63-token split paid about 16
round-trip latencies. The softmax was serial: 12 threads made 63 `expf` calls each.

**Change** (`attention_kernel<Storage, Group>`, same grid, same split partition):
- **Staging.** Each 64-token tile's K rows, V rows and INT8 group scales are gathered into dynamic
  shared memory by `cp.async`: one round of loads per tile, across the page spaces of §19.3.11.
  - INT8 double-buffers its 33 KiB stages, so the next tile streams under the current one.
  - BF16 uses one 64 KiB stage.
  - The first tile is issued before the query heads load and rotate.
- **Softmax.** The tile maximum is a warp reduction, which is exact, and the exponentials are
  computed in parallel. Each head's l is still summed serially in token order.
- **Launches.** Attention and the merge launch as PDL consumers.
- **Arithmetic is unchanged:** the score's fma order and butterfly, the online-softmax update
  sequence and the in-order value fold. The outputs are therefore bit-identical to the previous
  kernel.

**Verification.** A temporary check ran the previous kernel beside the new one in
`infernix_qsa_test`. All 12 calls matched bit for bit: bf16 and int8; W = 1, 5 and 40, the last with 8
tiles per split; pool pages and host/lent page spaces. The FP64-oracle cases pass.

**Measured** (RTX 5090, dense8m, `int8` KV, 2026-10-06, A = Gold `283613d8b`):
- **Greedy ids are identical, A against B**, on seven workloads: the four short prompts (dense QSA,
  96 tokens), the code prompt with MTP (295 tokens), and the ~40K prompt plain and MTP (selected
  blocks, 96 tokens).
- **Decode, two ABBA pairs** (bench, tok/s, A → B):

  | Context and mode | A | B | Change |
  |---|---:|---:|---:|
  | 8K plain | 96.91 | 100.13 | +3.32 % |
  | 8K MTP | 134.54 | 136.45 | +1.42 % |
  | 128K plain | 59.99 | 61.36 | +2.28 % |
  | 128K MTP | 51.61 | 51.59 | −0.03 % |

  At 128K MTP the B runs were 52.07 and 51.12 against 51.61 and 51.60 for A, so the change is
  neutral there within that spread.
- **Exposed time per call** in the decode profiles (attention end − selection end, median). Under
  PDL a kernel's own duration includes its wait.

  | Workload | Before | After |
  |---|---:|---:|
  | T = 1, 8K and 128K | 45-47 µs | 18.2 µs |
  | MTP at 8K (264 CTAs) | 46 µs | 34 µs |
  | Merge | 3.3 µs | 3.3 µs |

  These are still well above the ~3 µs budget (open; see the next note).

**Instruction count (ncu, op bench, int8, 8K, W = 1).** The staged kernel ran 24 µs for 57K cycles.
Issue slots were 11.7 % busy, with 0.38 eligible warps per scheduler and 4.45M warp instructions
(~8,400 per warp). The time went to two places:
- the value fold (~1/3): 12 broadcast loads and a predicated FMA per head and token;
- the score loop (~1/3): 96 scalar query loads per token.

The kernel was latency-chained, not memory-bound (DRAM 4.6 %).

**Restructured (bit-exact)** in `attention_kernel<Storage, Group, Exact, Stages>`:
- **512 threads.** Sixteen warps score the tokens. In the value phase two threads share each
  dimension, six heads each.
- **Query layout.** A lane reads its eight query values of a head as two 16-byte words,
  `[head][r / 4][lane][r % 4]`.
- **Probabilities** are read four tokens at a time.
- **Exact group.** An instance for exactly 12 query heads per KV head drops the head-loop
  predicates; other groups up to 16 use a bounded instance.

No per-(head, dimension) order changes, so outputs equal the previous kernel's.

Results:
- **Greedy ids** are identical, A (`cbcde52a4`) against B, on fourteen workloads: int8 and bf16, with
  the short prompts, code MTP and ~40K plain and MTP.
- **Op bench** (cold L2, graph, attention and merge), every storage:

  | Width | Before | After |
  |---|---:|---:|
  | W = 1 | ~27 µs | 18.4-22.5 µs |
  | W = 5 | ~47 µs | 37-43 µs |
  | W = 8 | ~74 µs | 49-55 µs |

- **int8 decode ABBA:** 8K plain +0.77 %, MTP +0.92 % (both pairs).
- **Clock caveat.** These runs, and every engine run after 11:44 on 2026-10-06 until the reset, had
  GPU clocks locked near 2.4 GHz (an interrupted ncu session). Same-rig A/B comparisons hold, but
  absolute rates are about 13 % low.

### 19.3.15 Every Infernix KV profile on Flash-Next (2026-10-06)

Until now QSA served only `bf16` and `int8`. Its kernels now take the other five profiles:
- `fp8` (E4M3-row256, K rotated);
- `nvfp4` (NVFP4-G16, K and V rotated);
- `k8v4` (FP8 K, NVFP4 V, both rotated);
- `vq2` and `k4v2` (vector-quantized, both rotated, with the exact recent-key window; below).

The paged planes were already generic (`paged_kv_storage_layout`), and so was the append, except for
the window.

**Decode attention.** One tile layer serves every profile: `KVTile<Storage>` over `StagePlanes`,
with planes `[K rows | V rows | K scales | V scales]` in the paged planes' own byte layout.
- Each cell decodes to its exact represented value in FP32 (code × scale is exact in FP32).
- Rotated values are rotated back on the unnormalized output. That map is linear, so split partials
  still merge by their weights.
- FP8's 2-byte row scales are plain loads, issued after the rows' `cp.async`.

**Prompt route.** `qsa_prompt_kernel<Storage>` is generic over a key code and a value code:
- **FP8 keys** use the INT8 path: an exact `e4m3x2` → `f16x2` widening and one row-scale group.
- **NVFP4 keys** decode to exact FP16 values: E2M1 × E4M3 fits FP16. The query takes a k-step order
  in which one 16-byte ldmatrix chunk feeds two k-steps.
- **FP8 values** use the INT8 path's transposed decode, with a scale guard of 64 instead of 256.
- **NVFP4 values** decode into a per-warp FP16 tile and take the FP16 ldmatrix.trans path. The merged
  rows are rotated back.
- **Scales.** FP8 row scales are gathered as their aligned 4-byte word, plus a parity bit per token.
- **Gather.** Each lane locates its tile row once; the lanes copying a row's chunks receive its
  addresses by shuffle. The first version located every chunk separately, once for K and once for
  V. At W = 256 that made the int8 prompt kernel 436 → 612 µs and bf16 1,093 → 1,829 µs; it was
  fixed before landing (results below).

**Qualification.** `infernix_qsa_test` is storage-generic and covers every profile:
- the FP64 oracle over decoded K/V, with rotations;
- decode, verify and prefill widths, and the prompt route;
- width invariance;
- host and lent page spaces.

`infernix_qsa_attention_bench` (new) times the Op and the selection alone, for any storage, context
and width.

**Quality** (teacher-forced, `infernix_qwen4_exp_forward_real_test --dump-logits`, chunk 256,
dense8m; 3,580 positions over code/doc/chat in the dense regime and the last 1,024 positions of a
16K text in the sparse regime). Measured against bf16 KV:

| `--kv-dtype` | KL mean | dNLL, all | dNLL, long16k (sparse) |
|---|---:|---:|---:|
| int8 | 0.061 | +0.007 ± 0.007 | +0.016 ± 0.014 |
| fp8 | 0.069 | +0.006 ± 0.008 | +0.024 ± 0.013 |
| nvfp4 | 0.070 | +0.004 ± 0.008 | +0.001 ± 0.013 |
| k8v4 | 0.069 | +0.001 ± 0.008 | +0.012 ± 0.015 |

All three new profiles are within noise of int8 on these texts. KV memory per token and KV head is
516 B (fp8), 288 B (nvfp4) and 402 B (k8v4), against int8's 528 B.

**Speed** (B, clocks locked as above): decode at 8K is within ±1 % of int8 for every profile, since
decode attention is latency-bound and format-independent. Prefill at 32K, before the gather fix:

| `--kv-dtype` | Before the prompt route (SIMT kernel) | With the prompt route |
|---|---:|---:|
| fp8 | ~3,770 tok/s | 4,796 tok/s (+27 %) |
| nvfp4 | ~3,770 tok/s | 4,966 tok/s (+31 %) |
| k8v4 | ~3,770 tok/s | 4,886 tok/s (+29 %) |

For comparison, int8 ran 4,773 tok/s (old kernel: 4,914).

**After the gather fix, clocks reset** (F1c, B = the landed build):
- **Prefill ABBA at 32K against the old prompt kernel:** int8 5,179 → 5,559 tok/s (+7.3 %), bf16
  4,946 → 5,366 (+8.5 %). One A run was 4,952 against 5,406, so the int8 gain is uncertain by about
  ±4 %.
- **int8 decode ABBA:** 8K plain +0.25 %, 8K MTP +0.85 %, 128K plain +1.0 %.
- **Greedy ids** are identical A/B on all fourteen int8/bf16 workloads.
- **Per profile (B):**

  | `--kv-dtype` | pp32k tok/s | tg8k | tg8k MTP | tg128k |
  |---|---:|---:|---:|---:|
  | bf16 | 5,384 | 99.2 | 139.9 | 57.5 |
  | int8 | 5,552 | 99.4 | 136.3 | 61.4 |
  | fp8 | 5,551 | 100.1 | 143.6 | 61.8 |
  | nvfp4 | 5,501 | 99.5 | 128.6 | 60.9 |
  | k8v4 | 5,549 | 99.7 | 126.9 | 60.5 |

  MTP rates differ with each profile's draft acceptance, single run each.

**vq2 and k4v2.** These use the exact INT8-G64 recent-key window
([paged KV §9.3](paged-kv-cache.md#93-vq2k4v2-exact-recent-key-window)). A selected key j of
query p is read exactly when `vq_exact_key(j, p)` holds (j < 64 or j ≥ p − 768) and its row
matches the stored codes; otherwise its codes are read. As built, mirroring Qwen3.5:
- **Program.** One window plane set per KV layer (Main Text and MTP), each plane `[leading, 1,088
  slots, KVH, lanes]` (`prefix::KvWindowGeometry`), is bound as `layer.kv.window`.
  - The forward pass sets each call's `slots` to the sequences' state slots.
  - At 2 KV heads this is ~1.1 MB per layer and lane, ~15 MB per lane.
  - A slot reset zeroes the lane's tags.
- **State image.** Five `QsaWindow` parts per KV layer (K/V codes, K/V group scales, tags) join that
  layer's group, so prefix snapshots, pause and replay carry the window like the raw-key tails.
- **Append inside the Op.** For these profiles `qsa_attention` takes the call's K and V
  (`QsaAppend`) and appends them itself:
  - Calls up to `kKVWindowInlineWidth` write their window slots directly.
  - Wider calls (one sequence) stage every column's exact row in the Op's workspace, attend reading
    in-call keys from staging, and commit the slot-bound rows afterwards.

  The forward pass appends them itself only for K/V-only calls.
- **Kernels.** Both QSA kernels expand each tile into the INT8-G64 planes before running the INT8
  arithmetic:
  - **Gather:** paged codes and row scales, plus the exact rows (window slot or staged row) and the
    window tags of the keys the column reads exactly.
  - **Keep or decode:** a row stays when it is staged or its tag matches the stored codes.
    Otherwise its codes are decoded (VQ2 codebook in shared memory, Q4 levels by byte permute), with
    the row scale as all four group scales.
  - **Output:** rotated back.

  The prompt kernel runs three warps per CTA to fit two stages, the codes and the codebook.

**vq2/k4v2 qualification.** `infernix_qsa_test` covers:
- the Op's own append;
- a pre-call window in which every fourth slot is stale (its keys must read their codes);
- decode, verify, two-sequence and prompt-route calls;
- a staged 300-column call;
- calls without a window.

The oracle decodes the VQ2 codebook and Q4 levels independently and reads the call's exact rows back
from the committed window.

**vq2/k4v2 results** (dense8m, 2026-10-06):
- **Quality against bf16** (same texts):

  | `--kv-dtype` | KL mean | dNLL, all | dNLL, long16k |
  |---|---:|---:|---:|
  | vq2 | 0.071 | +0.010 ± 0.008 | +0.019 ± 0.014 |
  | k4v2 | 0.070 | +0.009 ± 0.007 | +0.014 ± 0.014 |

  Both are close to int8. On the 512-token code text vq2 and k4v2 give identical results, since every
  key there is read from the exact window.
- **Speed** (single runs):

  | `--kv-dtype` | pp32k | tg8k | tg128k |
  |---|---:|---:|---:|
  | int8 | 5,557 | 100.9 | 61.6 |
  | vq2 | 5,261 | 99.8 | 63.1 |
  | k4v2 | 5,063 | 94.8 | 56.0 |

  - vq2's smaller KV leaves more expert frames at 128K.
  - k4v2 pays for its Q4 key encoding and decoding.
- **Op bench, decode W = 1** (including the Op's append): vq2 22.5 µs, k4v2 24.6 µs, int8 15.8 µs.
  At W = 256 they are 621 / 866 µs against 227 µs; the append's VQ encoding dominates.
- **Real tests with `INFERNIX_QWEN4_KV=vq2` and `=k4v2`,** plain and MTP: every prefix-cache scenario
  passes, including tap and Host-block resumes identical to the cold run. Preemption passes too:
  - The exact path equals the solo runs.
  - Replay completes every token. MTP replay diverges after 607 tokens with vq2 (replay
    arithmetic, as int8's 572).
- **MTP decode at 8K** (single runs): vq2 114.6 tok/s, k4v2 135.3 tok/s; draft acceptance differs
  per profile.
- **A fault the tests found.** `kv_cache_append_batch` (the drafter's K/V-only calls) refused the
  vector-quantized storages. It now appends them with each sequence's window slots.
- **Unit tests:** QSA, Qwen4Exp, KV-cache append and softmax-attention tests pass (29/29).

**Optimisation round 1** (after landing, 2026-10-06):
- **Changes.**
  - **Decode splits** follow the SM count: `clamp(SMs / (columns × KV heads), 1, 48)`, at least 4
    once a call is split. This replaces 340 CTAs at most 33 splits. A call that fills half the SMs
    unsplit (43+ columns at 2 KV heads on 170 SMs) now takes the prompt route.
  - **Prompt kernel.** One stage at two CTAs per SM when it fits and the call fills a wave, with a
    sequential warp merge whose bits do not depend on the stage count.
  - **Per-format kernel changes.** NVFP4 values decode in half tiles, k8v4 runs three warps, and the
    q load is vectorized.
- **Op bench** (`infernix_qsa_attention_bench`, attention time, cold L2, before → after):
  - W = 1-2: equal.
  - W = 3-8: 5-18 % faster.
  - 43-85 columns: 42-73 % faster, now on the prompt route.
  - W = 256 at 8K: int8 227 → 195 µs, nvfp4 241 → 180, vq2 621 → 395, k4v2 866 → 532.

  At 128K, selection dominates W = 256: 390 µs of the call's 580-700 µs.
- **Quality.**
  - The prompt route is bit-identical to the landed build: the int8, nvfp4 and vq2 chunk-256 dumps
    give the same KL and dNLL as the tables above.
  - Decode at W = 1 rounds differently because the split partition changed. Teacher-forced at
    chunk 1 on the code text, int8 against bf16 KV, before → after: KL 0.0249 → 0.0242, dNLL
    +0.008 → +0.016 ± 0.010. The two builds differ from each other by KL 0.015.
  - Plain greedy ids therefore diverge after 9-87 tokens on all ten int8/bf16 workloads.
  - The MTP code output (W = 5) is identical.
- **Speed** (ABBA, against the landed build `7a4a13b4d`):

  | Workload | Before | After | Change |
  |---|---:|---:|---:|
  | pp32k bf16 | 5,369 | 5,466 | +1.8 % |
  | pp32k int8 | 5,529 | 5,603 | +1.3 % |
  | pp32k fp8 | 5,512 | 5,560 | +0.9 % |
  | pp32k nvfp4 | 5,512 | 5,583 | +1.3 % |
  | pp32k k8v4 | 5,500 | 5,541 | +0.7 % |
  | pp32k vq2 | 5,217 | 5,423 | +4.0 % |
  | pp32k k4v2 | 5,052 | 5,320 | +5.3 % |
  | tg8k int8 | 100.97 | 101.99 | +1.0 % |
  | tg8k int8 MTP | 137.83 | 131.73 | −4.4 % |

  pp32k and tg8k plain are int8 unless named.
- **The tg8k MTP loss is a text change, not a kernel cost.** The corpus continuation diverges, so
  draft acceptance falls from 2.31 to 2.24 tokens per round, with 228 rounds instead of 222.
  Measured on the CLI code prompt, whose output is identical in both builds: MTP decode +0.4 %
  (int8) and +0.8 % (bf16).
- **Correction: the decode split rule broke width invariance and is reverted for narrow calls.**
  - Under the wave rule, a column's split count depended on its call's width. A sequence decoding
    beside others (W = 2: 42 splits) rounded differently from the same sequence alone (W = 1: 48).
  - The Qwen4Exp preemption and prefix-cache real tests caught it. "A equals its solo run" failed,
    with 2 of 1,200 tokens equal after the first difference. These tests had not run before
    `a47578934` landed on Gold.
  - **Now:** calls of at most 8 columns split by one 64-token tile per split, 33 for every width,
    as before round 1 for W ≤ 5. Wider calls keep the wave rule. `test_qsa` now checks that the
    last row of a 2-, 4- and 8-row decode call, and of a 2-row verification call, gives the same bits
    when called alone. The rule it replaces fails that check.
  - `qsa_attention_workspace_bytes` covers every narrower call, because an 8-column call keeps
    more partials than wider calls up to ~21 columns.
  - **Cost against the wave rule** (op bench, int8 8K): W = 1-2 equal, W = 5 +2.4 µs, W = 8
    +11 µs per call. That is under 0.2 % of a round at 12 QSA layers.
  - **Round 1's narrow-call claims no longer apply.** The 5-18 % at W = 3-8, and the W = 1 rounding
    change with its chunk-1 quality numbers, are gone. Decode at W ≤ 5 is again the pre-round-1
    partition. W = 6-8 used 28-21 splits before and now use 33, which also makes C = 6-8 plain
    decode invariant.

### 19.3.16 Decode misses between the CPU and the link (2026-10-06)

**Profile** (nsys, Gold `283613d8b`, int8 KV, 8K context, 128 tokens; `fn/rigs/prof/analyze_critical.py`
and `analyze_misses.py`).
- **The main stream is busy 89 % of decode.** `stage_kernel` takes 34 % of it and `cpu_wait_kernel`
  14 %. At 128K these become 42 % and 11 %.
- **A staged miss costs ~105-120 µs.** That is a 2.76 MB record at the x8 link rate.
- **Plain decode, by staged misses per layer call:**

  | Staged misses | Layer calls |
  |---:|---:|
  | 0 | 47 % |
  | 1 | 30 % |
  | 2 | 16 % |
  | 3-4 | 6 % |

  At 128K, 21 % of calls stage 3-4, since the larger KV leaves fewer frames.
- **The CPU leg ends 30-50 µs after the PCIe leg** in every bucket. The CPU computes about two
  thirds of the misses.
- **MTP verification calls stage up to 30 records (3.5 ms).** Once a call has 12 or more misses, the
  CPU cap of 8 jobs leaves the rest to the link.
- **The first 4-6 decode rounds after a prefill are cold.**
  - Plain rounds take 30-50 ms against ~19 ms in steady state.
  - MTP rounds take 150, 100, 39 and 35 ms against ~20.
  - This happens partly because the prefill stream's lent frames come back empty.

**Cost-based split (rejected).** The plan kernel took the thinnest misses for the CPU while
cpu_us + cpu_column_us × (columns − 1) stayed below pcie_us × (remaining misses). The constants were
105 / 55 / 18 µs, set through a TEMP environment override in one build. Every variant gave identical
ids. tg8k −0.65 % plain and −2.6 % MTP against the divisor rule (4 reps each). The CPU is faster
than those constants: for one or two misses, the divisor rule's all-CPU choice beats staging one of
them. The divisor rule stays.

**CPU job cap** (`ProgramOptions::cpu_expert_jobs`; the §19.3.5 S3 sweep had never run). Caps
8 / 16 / 32, three rotated reps each, using a TEMP override in one build. Every output was
identical. Change against cap 8:

| Workload | 16 | 32 |
|---|---:|---:|
| tg8k plain | +0.3 % | +0.3 % |
| tg8k MTP | +0.6 % | +0.3 % |
| tg128k plain | −0.1 % | −0.5 % |
| cold CLI code, MTP | +2.8 % | +2.9 % |
| serve C = 4 plain, fill | +7.2 % | +7.1 % |
| serve C = 4 plain, warm | +1.3 % | +1.6 % |
| serve C = 2 MTP, fill | −0.2 % | +0.1 % |
| serve C = 2 MTP, warm | −0.1 % | −0.0 % |

- **The default is now 16.** Calls with many misses gain, and every other workload is unchanged
  within run-to-run spread.
- **The tg128k −0.1 %** is inside its reps' spread: 61.2-61.8 against 61.4-61.6.
- **Cap 32** gains nothing more and has the larger tg128k drop.

### 19.3.17 Prefill: batched expert copies, no promotions between walk spans (2026-10-06)

**Profile** (nsys, Gold `a1e40253a`, int8 KV; `fn/rigs/prof/analyze_pf.py`). Prefill under nsys runs
within 2-4 % of its bare rate.

| Prompt | Span | Main stream busy | Largest idle gaps |
|---|---:|---:|---|
| 32K | 4.5 s | 72 % | 0.63 s after the first token is sampled; 8 × ~35 ms before each chunk's embedding (n-gram rows) |
| 128K | 18.3 s | 76 % | 1.4 s between the two 64K walk spans; 1.3 s after sampling; 32 × ~30 ms before the embeddings |

The two large gaps are `prefill.residency`: `after_round` with 16 promotions per layer and chunk,
about 12,300 record copies (31-37 GB) per span.
- **The host was blocked while issuing them.** On Windows, a `cudaMemcpyAsync` into a full copy
  queue waits about one record's transfer time (~94 µs at x8). Issuing the copies took as long as
  running them, and the first token was returned only afterwards.
- **Between spans the copies were pure waste.** The stream lease still held most frames, so the
  promotions churned through the rest, and the next span waited for them.

A probe (`fn/probes/batch_copy_probe.cu`) issued 4,000 records as one `cudaMemcpyBatchAsync`: 1.3 ms
on the host instead of 316-416 ms, at the same 27.5 GB/s.

**Changes.**
- **`core/copy_batch.h`.** `CopyBatch` enqueues copies through `cudaMemcpyBatchAsync`. A copy whose
  destination overlaps a pending one first flushes the pending batch, so the stream-order result
  (the later copy wins) is kept. `infernix_copy_batch_test` checks disjoint, repeated and
  partially overlapping destinations.
- **`ExpertResidency::issue_loads`** enqueues the promotions in groups of 64 records, each with its
  own completion event, so decode rounds publish them as they land rather than all at once at the end.
- **The prefill expert stream** enqueues each layer's runs as one batch.
- **A walk span that is not the prompt's last** promotes nothing; the last span still promotes for
  decode. That changed later in this section: no span promotes now.

**Results** (ABBA, A = Gold `a1e40253a`, int8; both arms still on round 1's wave split rule):

| Workload | Metric | A | B | Change |
|---|---|---:|---:|---:|
| pp32k | prefill (first token) | 5.795 s | 5.297 s | −8.6 % |
| pp128k | prefill (first token) | 19.40 s | 17.03 s | −12.2 % |
| -pg 32768,128 | prefill / decode / total | 4.37 s / 80.4 tok/s / 5.99 s | 3.83 s / 67.6 tok/s / 5.75 s | −12.3 % / −16.0 % / −4.0 % |
| -pg 131072,64 | prefill / decode / total | 17.94 s / 56.5 tok/s / 19.19 s | 15.64 s / 26.8 tok/s / 18.15 s | −12.8 % / −52.7 % / −5.4 % |
| tg8k MTP | prefill / decode / total | 2.48 s / 108.1 / 4.90 s | 2.42 s / 105.8 / 4.88 s | −2.4 % / −2.1 % / −0.3 % |

- **The first token arrives 8.6-12.8 % sooner.**
- **Decode right after a long prefill is slower.** Its staged misses now share the link with the
  promotions that used to delay the first token. After 32K and 128K the first 128 / 64 tokens run
  at 84 % / 47 % of the earlier rate.
- **Prefill plus decode still finishes 4-5 % sooner.**
- **Greedy ids are identical** for the ~40K walk prompt, code MTP and prose2k.
- **In the same rig the preemption and prefix-cache real tests failed.** The cause was round 1's
  width-dependent split rule (§19.3.15, correction), not these copies.

**The next span's n-gram rows are read during the current span.** Before each chunk's embedding,
the host reads its n-gram rows from the volume (IOPS-bound, ~30-70 ms per 4,096-token chunk) while
the device waits: ~0.5 s per 64K span.
- **How.** While a span's steps run, the host reads the following calls' rows (as `walk_span_end`
  would group them) straight into `walk_host_`'s chunk slots, spread over the remaining steps. It
  starts after the span's first device synchronization, once the span's own slots have uploaded.
- **Use.** The next `walk_begin` takes a chunk's rows from its slot when lane, call, position, width
  and an FNV hash of the hashed tokens all match; any other chunk reads as before. No memory is
  added.
- **Not covered.** The prompt's first span and chunk-major prefills are unchanged.
- **Diagnostic.** The prefill stream line counts the chunks.

**Prefetch results** (ABBAABBA, A = `b5bacdb15`):
- pp128k: 17.14 → 16.71 s (−2.5 %).
- pp32k, one span: 5.307 → 5.314 s (+0.1 %). A first ABBA showed +1.1 %; eight runs did not
  reproduce it.
- Greedy ids identical on a ~100K prompt (12 chunks prefetched) and a ~40K prompt.
- Prefix-cache and preemption real tests pass.

**No promotions after a walked prompt.** The prompt's last span promoted 16 experts per layer and
chunk (~6,100 records at 32K, ~12,300 at 128K). The copies no longer delayed the first token, but
they shared the link with the answer's staged misses, and decode right after ran slower.

A sweep with a TEMP cap per layer (one build, outputs identical by construction; `fn/rigs/split`
promo, promo2; 3 rotated reps, then ABAB) compared the current budget with caps of 64, 32 and 0:

| Workload | Current | 0 | Total | Decode rate |
|---|---:|---:|---:|---:|
| -pg 32768,256 | 7.425 s | 7.181 s | −3.3 % | 72.3 → 77.2 tok/s |
| -pg 131072,128 | 18.705 s | 17.272 s | −7.7 % | 38.6 → 65.0 |
| -pg 16384,512 MTP | 7.355 s | 7.128 s | −3.1 % | 108.8 → 114.0 |
| -pg 32768,1536 | 21.854 s | 21.493 s | −1.7 % | 86.7 → 88.3 |
| -pg 131072,512 | 23.161 s | 21.096 s | −8.9 % | 66.3 → 89.5 |

- **Caps 64 and 32 fell between the current budget and zero.**
- **Zero wins for long answers too.** The answer's misses land in the returned frames and pick
  exactly its experts. The prompt's experts would have displaced them: decode after a 128K prompt
  stays at 89 tok/s instead of 66 over 512 tokens.
- **No walk span promotes now.** Its routes still update the cache's scores.
- **CLI check** (~40K prompt, ABBAABBA): the time to first token stays 6.7 s in all 8 runs, decode
  goes 2.2 → 1.5 s, and the total 9.0 → 8.2 s. An earlier single B run showed a 7.3 s prefill
  that did not recur.

**Chunk-major prompts too.** That covers single-call prompts and calls while another lane prefills.
They promoted 16 per layer per call; CPU-served tiny calls keep the decode rate. The sweep
(`fn/rigs/split` cpromo, TEMP per-layer value 16 / 4 / 0, 3 rotated reps, outputs identical)
measured bench total time and serve throughput, change against 16:

| Workload | 4 | 0 |
|---|---:|---:|
| bench -pg 2048,256 (total) | −0.7 % | −0.7 % |
| bench -pg 4096,256 MTP (total) | −0.4 % | −0.8 % |
| serve C = 1 MTP, fill | +1.8 % | +2.6 % |
| serve C = 1 MTP, warm | +1.8 % | +1.0 % |
| serve C = 1 MTP, lone | +0.6 % | +1.2 % |
| serve C = 4 plain, fill | +3.9 % | +4.5 % |
| serve C = 4 plain, warm | +6.2 % | +5.9 % |

**Now no prefill call promotes.** Zero is best or within the reps' spread of 4 everywhere, and it
keeps one rule.

### 19.3.18 Text YaRN beyond 262,144 positions (2026-10-06)

The reference model has one shared rotary embedding. A YaRN `rope_scaling` changes every rotation
it makes: attention Q/K, the QSA indexer's queries, the pooled index keys and the MTP layer. Infernix
used to refuse `rope_yarn_factor` for Qwen4Exp. Now the startup option drives all of those
rotations.

**Contract.**
- `--rope-yarn-factor F`, finite and in [1, 4]. `max_position_embeddings` (262,144) is the original
  span, and theta must be above 1.
- The context ceiling is 262,144 × F, capped at 1,048,576. A `--max-context` above it is refused
  at planning, with or without a factor. The factor does not raise `--max-context` itself.
- Factor 1 keeps every existing rotation bit for bit.

**Implementation.**
- Attention Q/K use `ops::rope` with the model's `PreparedRope`, built once per Forward by
  `prepare_rope(rotary_dim, theta, {F, 262144})`. That is the qualified Qwen3.5 YaRN path:
  correction range, linear ramp, attention scale 1 + 0.1 ln F.
- `QsaGeometry` carries `yarn_factor` and `original_positions`. The index-query and pool kernels
  take the same preparation and follow the attention kernel's coefficients:
  - the FP64 angle, reduced to one turn;
  - FP32 `sincosf`;
  - cosine and sine both multiplied by the attention scale.
- At factor 1 they keep the FP32 `powf` path unchanged.

**Tests.**
- `infernix_qsa_test` gains YaRN index rotations (factor 4, original 4,096), checked against an
  independent YaRN formula in its FP64 oracle.
- `infernix_rope_test` gains Flash-Next YaRN cases: 24 Q / 2 KV heads at 1, 5 and 4,096 tokens, and a
  factor-1 case at position 0.
  - A factor-1 case at position 262,137 failed at first. The FP32 default route is not qualified
    that far, so the case moved to position 0.
- `infernix_qwen4_exp_forward_real_test` takes `--rope-yarn-factor` for teacher-forced scoring.

**Gate** (RTX 5090, int8 KV, `qwen3_8_flash_next_nvfp4_dense8m`; rig `fn/rigs/qsaattn` yarn/yarn2).
- **Factor 1 is unchanged.** Greedy ids are identical to Gold `b454cd836` for prose2k, code MTP and
  the ~40K walk prompt. Prefix-cache and preemption real tests pass.
- **The ceiling holds.** `--max-ctx 300000` without a factor is refused.
- **Quality.** Teacher-forced NLL over the same last 1,023 positions of one 327,680-token code and
  docs stream, at different context lengths and factors. `pair_nll` gives the paired dNLL ± its
  95 % interval:

  | Context | Factor | NLL (nats) | Against |
  |---|---:|---:|---|
  | 16K | 1 | 1.95308 | |
  | 16K | 2 | 1.96378 | +0.0107 ± 0.0137 vs 16K f1; KL 0.078 |
  | 240K | 1 | 1.33160 | |
  | 240K | 2 | 1.34469 | +0.0131 ± 0.0137 vs 240K f1; KL 0.085 |
  | 320K | 2 | 1.33085 | −0.0008 ± 0.0169 vs 240K f1 |
  | 320K | 4 | 1.33653 | +0.0057 ± 0.0141 vs 320K f2 |

  - Inside the native span, factor 2 costs ~0.011-0.013 nats at both lengths. Neither change is
    significant on its own, but they agree. That is why the guide says to enable YaRN only when the
    length is needed.
  - Beyond the native span, YaRN works. At 320K with factor 2, the extra 80K tokens of context score
    the same tail as native 240K. Factor 4 at 320K is within noise of factor 2.
- **Speed** (`infernix_bench -pg 311296,128 --max-ctx 327680 --rope-yarn-factor 2 --prefill-chunk
  4096`, one rep):
  - prefill: 7,955 tok/s (39.1 s);
  - decode: 66.4 tok/s plain, 76.0 tok/s with MTP K4 and the draft head.
  - Against 245,760 tokens at `--max-context 262144` without YaRN (8.2K, 73.7, 87.9), these numbers
    follow the length. The YaRN index rotation's own cost was not measured at equal length.

### 19.3.19 Host threads: power throttling, not placement (2026-10-07)

**Question.** Strata pins its host thread away from CPU 0, where Windows delivers the GPU's
interrupts (`--host-core last`, from eddoursul's fork), and the CPU leg of a decode miss ends
30-50 µs after the PCIe leg (§19.3.16). Gold left its CPU expert threads unpinned.

**Placement A/B, round 1** (i9-13900K: 8 P-cores with two threads each and 16 E-cores; dense8m,
int8 KV; rig `fn/rigs/split/place`; arms rotated over 3 reps; outputs identical in every arm):
the service thread and five workers unpinned, on dedicated P-cores away from CPU 0
(`dedicated`), or with the service thread on CPU 0 (`waiter`). `dedicated` doubled tg8k (plain
46.1 → 93.7 tok/s, MTP 72.3 → 132.2) but cost 0.7-4.1 % in serving. Unpinned runs were erratic
(36.6-62.7 tok/s plain) and far below this design's earlier measurements, and putting the
waiting thread on the interrupt core was as good or better (CLI MTP +57 % against +36 %), so the
interrupt-core hypothesis does not hold here.

**Cause.** The rigs run without a visible window. Windows power throttling
(EcoQoS) treats such a process as background: lower clocks, threads steered to E-cores. Pinning
only hid it for the pinned threads.

**Round 2** (`fn/rigs/split/place2`, every arm but `none` exempt from execution-speed throttling):

| Workload | none | exempt | exempt + P-core set | exempt + dedicated | exempt + waiter |
|---|---:|---:|---:|---:|---:|
| tg8k plain | 37.6 | 98.7 | 98.6 | 95.2 | 94.0 |
| tg8k MTP | 63.8 | 135.6 | 136.1 | 130.9 | 132.1 |
| CLI code MTP | 80.7 | 140.3 | 140.9 | 134.1 | 133.5 |
| serve C = 2 MTP, warm | 63.4 | 107.1 | 107.8 | 103.2 | 104.9 |
| serve C = 4 plain, warm | 81.9 | 143.7 | 143.3 | 138.2 | 138.8 |

- **The exemption is the whole gain** (+69-162 %); a P-core affinity set adds nothing measurable
  and pinning each thread to one core costs 3-4 %.
- **Kept:** `core/power_throttling` exempts the process once, when an Engine is constructed
  (Windows; a no-op elsewhere). No placement code is kept.
- A server in a foreground console was not measured; whether it is throttled depends on Windows'
  foreground heuristics, so the exemption also protects a console left in the background.

**Fused per-head RMSNorm + RoPE in QSA attention (Strata #783 PR-f): rejected.** The Qwen3.5 text
form of `ops::rmsnorm_rope` was extended to (24,2) heads, M-RoPE positions and the generic and
YaRN coefficient routes. It was not bit-identical to the split route for the generic and
prepared coefficient instances (rare single-element differences; the rotation's SASS matched,
so the normalization's FMA contraction differs per instance), and end to end it measured within
noise (tg8k plain +2.4 % on the erratic unthrottled baseline, serve −0.3 to −1.1 %). Not worth a
change of output.

### 19.3.20 NInfer dev's tuned 2560-wide Linear routes (2026-10-07)

NInfer dev (`35e9b5c85..070fa61a3`) tuned Q8 and BF16 routes for the Flash-Next shapes. Op level
(RTX 5090, cold L2, CUDA Graph, two ABBA rounds; `fn/rigs/item7`) on the four production
`ops::linear` problems, NInfer against Gold:

| Problem | T = 1-8 | T = 9-64 | T = 4096 |
|---|---|---|---|
| Q8 2560×6144 (GDN and QSA output) | −10 to −41 % | −45 to −75 % | +0.1 % |
| Q8 16384×2560 (GDN q/k/v/z) | −6.5 to −33 % | −8 to −36 % | −4.5 % |
| BF16 96×2560 (GDN a/b) | −27 to −46 % | −26 to −44 % | +44 % |
| BF16 13952×2560 (QSA group) | 0 to −4.5 % | about 0 | +3.6 % |

**Exactness** (full outputs per width): NInfer's routes are not Gold's bits, and NInfer itself
runs a GEMV at T = 1 and split-K kernels from T = 2, so its T = 1 column differs from the same
column at T = 2-8 (Gold's classes are {1..8} and {9..}). Two variants were built on Gold's
schedules (TEMP `INFERNIX_TMP_UP7`):
- **a**: one NInfer family over T = 1..8, Gold above; Gold's width classes kept; bits differ at
  T ≤ 8 in a few outputs per thousand (at most 5 BF16 steps);
- **b**: NInfer's selector to T = 128 (T = 1 on its T = 2 kernel).

**End to end** (dense8m, int8 KV, process exempt from power throttling, 3 rotated reps):

| Workload | Gold | a | b |
|---|---:|---:|---:|
| tg8k plain | 98.94 | 99.82 (+0.9 %) | 99.80 (+0.9 %) |
| tg8k MTP | 136.64 | 160.18 (+17.2 %) | 160.17 (+17.2 %) |
| serve C = 2 MTP, fill / warm | 86.94 / 108.28 | −0.8 % / +0.8 % | −0.5 % / −0.4 % |
| serve C = 4 plain, fill / warm | 110.24 / 143.86 | −0.8 % / +0.8 % | −0.9 % / −0.8 % |

- The tg8k MTP gain is the text, not the engine: the changed bits change the bench's generated
  text, and its MTP acceptance rose from 0.665 to 0.800. Plain decode (+0.9 %) is the engine's
  gain; serving over 24 prompts is within ±0.9 %.
- Quality (teacher-forced code/doc/chat at `--chunk 8` and `--chunk 32`, which run the decode-width
  routes): KL to Gold 0.055-0.063 with dNLL −0.014 ± 0.009 to +0.048 ± 0.011, its sign flipping
  between texts and chunk widths. Gold's own two chunk widths differ by as much (perplexity 4.558
  against 4.712), so these runs cannot resolve a quality change of this size. Prefix-cache,
  preemption and decide real tests pass under both variants.
- **Not adopted**: about 1 % of engine speed for a change of output that cannot be shown harmless.
- **BF16 96×2560 at T ≤ 8 alone: not adopted either.** NInfer's SIMT schedule (8-row blocks,
  4.6-5.2 µs instead of 6.8-8.8 µs) matched the skinny GEMV bit for bit on the bench's synthetic
  inputs at every T = 1..8, but not on the model's activations: with it alone, greedy prose2k
  diverged from Gold (deterministically, three runs) and the prefix-cache real test's restored
  resume differed from the original turn at token 30. Synthetic-input equality is not evidence of
  bit-exactness for a different reduction order.

### 19.3.21 One-column expert kernels: FP4 half2 unit products (2026-10-10)

Decode and MTP verification run only the one-column narrow kernels (`gate_up_kernel<1>`,
`down_kernel<1>`): node-traced, the 8-column instances never appear at W ≤ 4. ncu of the op bench
(`infernix_offloaded_moe_wide_bench --tokens 1 --placement frames --cold`, which reproduces the
engine's per-layer time) found them issue-bound, not bandwidth-bound: 0.58 TB/s for a layer's 27.6
MB, ~70 SASS instructions per 144-byte unit of which 2 are `dp4a` (24 decode the E2M1 codes to
int8, 7 convert the row scale, 8 rematerialize shared addresses and constants, 6 guard the column
loop), 0.33 instructions per cycle per scheduler at 2 CTAs per SM.

The one-column instances now take a unit's products with the hardware FP4 conversion instead:
- A code byte holds rows i and i + 8 at one k, so `cvt.rn.f16x2.e2m1x2` turns it into a half2 of the
  two rows' E2M1 values. Lane 8q + i takes code word q, i of the unit (rows i and i + 8, k = 4q..4q+3)
  and forms its share with one HMUL2 and three HFMA2 against the activation, stored per element
  as its code in both nibbles (the same conversion broadcasts it). Every value is a multiple of 1/2
  at most 6, so the four-term sum (≤ 144, on the 1/4 grid) is exact in FP16, and four times it is
  the lane's integer share P_q (≤ 576) of the block product P_b of §16.2.
- The lane adds (P_q · Sw) · Sa for both rows to int64 sums; the four lanes of a row are reduced by
  two shuffles at the end. Integer sums are associative, so the row sums, outputs and h blocks are
  exactly those of `unit_sums` and of the CPU engine.
- gate/up quantizes its column with `quantize_a4_codes` (the same scale and codes as
  `quantize_a4_block`); down converts the stored h blocks' doubled values back to codes.
- The 8-column instances (prefill passes) keep `unit_sums`.

Evidence: output hashes equal at T = 1, 2, 3 (op bench, cold and warm); `infernix_offloaded_moe_cuda_test`
and `_layer_test` (decode T = 1 and verification T = 4 against the CPU engine) pass; teacher-forced
FP32 logits of chat.ids (positions 768+) and code.ids (400+) are byte-identical to the previous build
with every call one column (`--chunk 1`) and four (`--chunk 4`). Executed instructions: gate/up
12.3 → 8.1 M per layer call, down 6.8 → 3.6 M. Op bench T = 1 cold 47.7 → 41.5 µs (−13 %), T = 3
84.6 → 72.2 µs (−15 %); warm −15 to −20 %. End to end (Dense8, INT8 KV, pg1024+256 on corpus text,
4 old/new pairs × 3 reps): plain decode 123.3 → 127.9 tok/s (+3.74 %, pairs +3.26..+4.17 %),
MTP 147.5 → 154.2 (+4.55 %, pairs +4.51..+4.60 %), identical speculative counts.

Rejected on the way: staging the slice in 4 `cp.async` groups and taking each as it lands, before
(T = 1 −2.8 %, noise) and after this change (T = 1 +5 %, T = 2 +15 %: the extra barriers cost more
than the overlap gives). The remaining time is the wait for the slice and the 1.18-wave grid (400
CTAs on 340 slots); the CTAs cannot hold a layer's 18.4 MB of gate/up weights at once (2 × 49 KB
per SM), so only streaming work items can remove the tail.

**down as a programmatic dependent of gate/up (2026-10-10).** A layer runs four expert launches
(gate/up and down of the resident and the staged phase), each paying its launch ramp and its slice's
DRAM latency after the previous one ends. `down_kernel` (A4, every width) now launches through
`pdl::launch_consumer` and issues its weight slice before `pdl::wait_for_dependencies()`; `gate_up_kernel`
calls `pdl::trigger_dependents()` once its own slice has landed, so down's CTAs fetch their weights
while gate/up computes and finishes. Everything down reads before the wait (the dispatch, the CPU
flags, the job records, the frames, the weights) was written before gate/up started; h, its output,
is read after the wait, and every CTA waits, so down's completion implies gate/up's. Bits unchanged
(logits byte-identical at `--chunk 1` and `4`; real tests pass). End to end (as above, against
dd03c8136): plain 127.9 → 132.2 tok/s (+3.31 %, pairs +2.76..+3.66 %), MTP 154.1 → 157.6 (+2.26 %,
pairs +2.18..+2.32 %), identical speculative counts. The W4A16 pair (`gate_up_kernel_a16`,
`down_kernel_a16`) follows the same contract: on the uncensored W4A16 artifact plain 125.9 → 129.5
tok/s (+2.83 %, pairs +2.61..+3.05 %), MTP 141.8 → 144.0 (+1.50 %, pairs +1.41..+1.62 %), identical
speculative counts.

gate/up stays an ordinary launch. Measured and rejected: the fork route's resident pass as a
programmatic dependent of the plan kernel before it (`cpu_plan_kernel`/`fetch_plan_kernel` triggering
at entry, gate/up staging its slice before the wait and waiting before its own trigger; a resident job
is never a CPU job, so the pass reads nothing the plans write). Dense8, against 166158ff5: plain −0.17 %
(pairs −0.35..+0.02 %), MTP −0.33 % (all four pairs negative): the early 400-CTA grid takes the SMs
the plan and the fork stream's staging need.

Also measured and rejected (2026-10-10, Dense8, against the PDL pair):
- **One ticketed launch per phase**: gate/up items then down items in one grid, each CTA taking a
  ticket at start, a down item waiting (after staging its weights) only for its own job's gate/up
  items through per-job counters carried in `MoeDispatch` and zeroed by `moe_dispatch`. Correct
  (output hashes equal, layer and CUDA tests pass) and faster in the eager op bench (T = 1 cold
  41.5 → 36.6 µs, against the unchained pair), but end to end plain −0.46 % (pairs −0.78..−0.24 %),
  MTP −0.52 % (all pairs negative): the PDL chain already overlaps down's weight fetch with gate/up,
  and the fused grid runs down items at two CTAs per SM instead of three. (A first build lost 2 KB to
  a static `__shared__` ticket, which dropped it to one CTA per SM.)
- **The top-k route kernel by warp reductions** (`__reduce_max_sync` on order-preserving keys, the
  smallest id by `__reduce_min_sync`, one lane rescanning per round; same selection): plain −0.12 %,
  MTP +0.05 %, noise; the kernel's 4.3 µs is its load and launch, not the selection.

Host time between decode rounds, re-measured without a profiler (a temporary probe, plain pg1024+256 at
132 tok/s): 82-89 µs per round (`settle_round` 42-49, the decode prologue 26.6, the engine ~13), ~1.1 %
of a round; nsys had shown 226-360 µs. Pipelined decode (§19.3.9 step 2) is therefore worth at most
~1.1 % (launch-ahead) or ~0.6 % (deferred settle) and is not pursued.

### 19.4 On the Gold-Star-Infer runtime contract (2026-10-05)

The Flash-Next history (dev through `claude/fn-layer-prefill` 91af38dd0) was replayed onto
Gold-Star-Infer 5fbfc3dbe. That branch carries NInfer's continuation/checkpoint Engine:
resource-pressure preemption with Snapshot/Replay recovery, unit reservations, and a two-step
admission. Only the final commit of the port builds; the replayed commits in between do not
compile on their own, because the new contract changed the Program surface under them.

**How Qwen4Exp maps onto the contract.**

| Contract step | Qwen4Exp |
|---|---|
| `plan_request(PreparedPrompt&&)` | Keeps the prepared prompt on the plan; compiles a token constraint and checks readout tokens against the public token domain. |
| `hybrid_sources` | Offers two sources: the prefix cache's affordable snapshot, chosen without a lane, then the root. Returns none while the request should wait for a prefilling sibling's snapshot (coalescing, below). |
| `hybrid_prefetch` | Copies a blocked FIFO head's Host-only blocks into free and Host-backed Device pages while it waits. |
| `start_binding` | Re-selects the source for the actual lane, crediting a lane-resident snapshot. A changed choice returns no reservation and no shortage, so the Engine falls back to the root. Otherwise it makes room and reserves the pages of the prompt plus one round (§19.3.13); a shortage is reported as `main_kv_pages`. Vision is reserved after the KV. A resumed request binds its ledger (§19.3.13). |
| Begin summary | `{prompt_tokens, reused_tokens, reuse_path}`. The reuse path is `HybridSnapshot`, `HybridEndpoint` or `Root`. |
| `poll_context` | Runs Vision steps, then publishes the bound sequence. On cancellation it releases the binding. |
| `reserve_units` | Grows a decode or control unit's lane to its next round, evicting cached blocks and growing the elastic pool first; a shortage is `main_kv_pages` (§19.3.13). |
| `start_pause` | Publishes the lane's state at its frontier to the prefix cache and releases the lane (§19.3.13). |
| `advance_replay` | One prefill call of the resumed ledger, without sampling (§19.3.13). |
| Checkpoints, capture | Stubs that are never reached: the prefix cache holds every snapshot. |

The ops' new `DeviceExecutionView` parameter is passed as `execution_view().on_stream(s)`. The
recurrent update and the replay record keep a bare stream. Gold's renames (`host_capacity_bytes`,
`--host-context-mib`, `ContextCacheMode::Original`) are taken throughout. The Qwen3.5 tower's
`parse_vision_config` moved to `models/qwen3_5/vision_config.h`, so `config.h` stays free of JSON
as on Gold.

**`/v1/decide` (prompt readouts and token constraints, `program/constraint.cpp`).**

- A readout reads the prompt's next-token distribution from the first token's logits column
  (rounded to BF16 like the sampler's input, before any mask), with `ops::target_logprobs` and
  `ops::top_logprobs`. A request without readout tokens pays nothing.
- A constrained round stages one descriptor per logits column and masks with
  `ops::constrain_logits`: before the sampler (the prompt's first token and plain decode rounds)
  and before the greedy targets and draft acceptance of a verification round (MTP and n-gram
  chains; Qwen4Exp verifies no trees). Column j of a row samples output step (generated tokens) + j.
  An unconstrained round skips the op entirely.
- The probability records of a round's licensed tokens become ConstrainedDraws as their tokens
  commit, so a rejected draft's draw is dropped with it.

**Hybrid cache admission, as in the Qwen3.5 binding.**

- *Prefetch* (hybrid-prefix-cache-spec §6.6): the Engine calls `hybrid_prefetch` for a FIFO head
  that was evaluated but could not bind. Up to 256 Host-only blocks of its chosen path are restored
  into free pages and pages evicted from Host-backed cached blocks. The pool never grows into the
  expert frames for a prefetch. A selection whose path is still landing waits for that batch first.
- *Drain before giving up a source:* when no snapshot candidate fits and Host writes are pending,
  selection drains them and retries the candidates before the root.
- *Coalescing* (§12.2): a fresh request sharing a prefix with a lane still prefilling (media
  agreeing via `media_agreed_prefix`, now shared in `block_keys`) waits for that lane's snapshot at
  the divergence. This happens when waiting saves more than a split costs and the predicted wait is
  within half the queue timeout. The divergence is the block boundary below it, or the end of an
  image both prompts carry. An exact tap is planned there; it splits the lane's pending call when
  needed. A layer walk's enqueued span is never split. An in-flight capture inside the shared prefix
  is waited for too.
- *Boundary snapshots:* taps planned as boundaries (Explicit, Structural, coalescing) are published
  as `SnapshotKind::Boundary`. A later conversation's continuation therefore does not supersede the
  shared state a `/v1/decide` fan-out resumes from. Before, every tap was published as `Tap`.

**Faults the port's verification found (2026-10-05).** Gold's Engine checks results that dev's
ignored, which exposed these:

- **MR3 flush (latent on dev).** `mtp_flush_cell` built its MTP call without the RoPE fields that
  the M-RoPE change (§19.3.2) made mandatory, so the call threw. `finish` swallowed the exception, so
  on dev an MTP request finishing with a pending cell silently skipped its endpoint capture and its
  blocks' write-through. Gold's Engine fails such a request ("terminal native sequence could not
  finish"), which crashed `/v1/decide` with MTP. The flush now stages the cell's RoPE position and
  block start. `finish` and `abort` report a failure through the diagnostic channel.
- **Engine admission without a cache.** Qwen4Exp always runs on the hybrid resource manager, which
  refused to work with the context cache off. That broke the CLI and `infernix_bench`.
- **Prefill interleaving.** A second lane's prefill could be scheduled during another lane's layer
  walk. `context_blocks` now reports such a lane, and the Engine's prefill turn skips blocked lanes.
- **Options validated after the artifact is read.** The Qwen4Exp route skipped the Engine's common
  option checks. They now run, architecture-aware, before the artifact is opened.

**Not carried over:** preemption. Under resource pressure a Qwen4Exp request is not paused and
replayed, because its whole KV extent is reserved at binding. It matters only at concurrency
above 1 when the KV pool is short. A request that does not fit waits in the queue, as on dev.

---

## 20. Documentation and authority changes

When the implementation lands:

- `AGENTS.md` and `README.md`: add `Qwen4ExpForCausalLM` to the product architectures, and the
  host-RAM and NVMe requirements.
- `docs/maintainer/qwen4-exp-model.md`: new model reference (mathematics, config, state, MTP, PLE).
- `docs/maintainer/expert-offload.md`: frame pool, MTP pool, residency state machine, epochs,
  policy, transfer agent, mailboxes, CPU engine, and the canonical routed-expert arithmetic. §8-§12
  and §16.2 of this file move there.
- [Tensor formats](tensor-formats.md): `nvfp4_mul`, with its decode
  oracles.
- [Storage layouts](storage-layouts.md): `nvfp4_expert_rg16_v1`.
- [Engine architecture](engine-architecture.md): Program ownership of host agents and frames;
  residency as an Op execution resource; the placement-invariance exception to the Op contract
  rules (§7).
- [Op development](op-development.md): the same exception, stated where the "no frozen bitwise
  equality" rule lives.
- [Paged KV cache](paged-kv-cache.md): the QSA pooled index-key plane, frame-backed KV pages, and
  the QSA host tier.
- [Artifact container](artifact-container.md): residency classes, the expert records, and the
  n-gram volume.
- [Weight conversion](../weight-conversion.md), [CLI](../cli.md), [serving](../serving.md): recipes A
  and B, the primary configuration, the options of §18, and `infernix-calibrate`.
- `docs/performance/qwen3.8-flash-next.md` and a model card.
- [Hybrid prefix cache spec](hybrid-prefix-cache-spec.md): a "Qwen4Exp binding" section, the
  authority for §19.3.1 (Host-born snapshots, zero Device slots, `insert_block(attach)`, saturating
  `call_seconds`, MTP rules, lineage exactness, call planner, defaults).
- [User guide](../qwen3_8-flash-next.md): prefix cache and its exactness trade-off; vision usage in
  place of "Not yet supported" (TTFT per image token, offload default, frames lent, steps for huge
  items, video cost); C > 1 advice; the n-gram diagnostic line.
- [CLI](../cli.md), [serving](../serving.md), README, `--help`: the §18 rows; `--kv-capacity` adds one
  copy-on-write page per lane; `--use-original-prefix-caching` is rejected for this model.
- Contracts `linear.h`, `qsa.h`, `hyper_connection.h`, `linear_swiglu.h`, `offloaded_sparse_moe.h`,
  `speculative_round.h`; [linear tuning](linear-tuning.md) (BF16 vision, Q8 curves);
  `tests/README.md`; `bench/README.md`: the expert-kernel microbenchmark
  `bench/ops/offloaded_moe_bench.cu` (concurrency S2).
- §19.2: each track's results, adverse ones included, and the n-gram track's corrections (the
  after-draft gap is host turnaround; the "reused events" A/B was n-gram-warm).
- This file is deleted.

---

## 21. Risks and open questions

| Risk | Effect | Mitigation / decision point |
|---|---|---|
| Routing locality is weaker than in the replay traces (h < 0.85 at 38%; the shared expert lowers token-to-token reuse) | Plain decode toward the lower rows of §4.2; smaller speculation gains | M1 decides early. Levers: H2 (largest at low h), H8, H3; compact KV profiles (+150-1,000 frames, §15.1). |
| **Recipe B fails the §16.3 gate** | Recipe A's plain decode is ~120-135 tok/s on the target budget and ~105-110 conservative, below the 120 Must in the conservative case | Per-class fallback keeps only the failing classes in BF16. H1 and H2 recover GPU and miss time. The result is reported against the Must target, which does not change. |
| The §8.3 GPU budget is missed. A ~20 MB kernel measured 1.17 TB/s in isolation on this card (sm120-fp4). | Every row of §4.2 drops ~10% per +0.5 ms | M6 measures PDL overlap first; per-kernel attribution against the budget; ncu on any kernel > 30% over; H1 gate |
| PDL visibility: K7b's pre-dependency phase reads K6's job list, written two kernels earlier | A stale job list, or a lost weight prefetch | K7a triggers its dependents only after its own wait, so K6 has completed. A stress test checks it. Fallback: K7b loads weights after its wait, a few µs per layer. |
| Host-issued DMA submission latency is too high for the one-layer prefetch window | Fewer staged hits | Batch submission; two-layer-ahead (H4); measured by calibration |
| K7b CTAs hold SMs while the CPU computes | No overlap for other work | They trigger their dependents before waiting, so the next layer's kernels start and prefetch (§8.3 rule 3) |
| Next-layer prediction recall is low with hyper-connections | Less prefetch benefit | Prefetch self-disables per layer below useful-byte ratio 0.3 (§8.7) |
| The exact expert arithmetic costs more than its gate allows | Lower narrow-route bandwidth | Kernel work first (IMMA, pipelining); last resort per §16.2: experts with that n go to the wide route, GPU-only |
| The A4 rule disagrees with NVIDIA's runtimes more than rounding-boundary noise | The model is not quite NVIDIA's | M0 checks the rule against ModelOpt's reference quantizer, which calibrated the checkpoint; M3 reports mismatch rates. A systematic difference is resolved toward ModelOpt and documented. |
| Checkpoint facts differ from §6.1: gate and up scales differ, or the MTP activation scheme | Extra A4 quantization per expert, or a different drafter route | Both cases are specified (§6.1); M0 records which applies |
| QSA prompt kernels for the primary `int8` profile beyond 2,051 visible tokens | Long-prompt prefill slower than planned | `int8` first in M3/M8; dense prompt kernels serve contexts ≤ 2,051; M8 reports every profile |
| DRAM bandwidth of the platform (DDR5 speed, 2 vs 4 DIMMs) | Scales cold-miss service and the DRAM ceiling | Calibration advice; 2 × 48 GB preferred |
| GPU memory clock below maximum under CUDA load (P-state) | All bandwidth-bound kernels slower | Calibration GPU sanity check against the 1,674.5 GB/s sustained-read reference; advice |
| Slow or stalling NVMe (QLC, DRAM-less, power-state exits) | Reads exposed before plain launches; gated verify rounds wait at layer 1 (§12.4) | Block dedupe, the gate, the 100 ms warning, counters, documented drive requirement |
| Calibration overfits the bundled corpus or a noisy measurement | Settings slower on real workloads | Held-out check in M9; noise-aware acceptance; defaults on ties |
| 96 GB is tight with a desktop session, Vision pins, the prefix Host tier, or a `bf16` KV host tier at 256K | OOM, swap or pagefile | The 8 GiB-available floor; prefix tier sized after model and Vision load (4 GiB, clamped, logged; §15.2); configurable KV host tier (3.6 GB at 262K for `int8`) |
| Engine contract changes for chained rounds (H6) | Large change to transactions | Built only if its gate passes; otherwise the synchronous boundary stays |
| Prefix cache: drafter state wrong after a resume; copy-engine FIFO (restores ahead of inputs: H2D FIFO measured, §19.2; bulk D2H ahead of per-round readbacks: unverified, prefix M0) | Lower acceptance, invisible at C = 1 greedy; stalled rounds | Bitwise drafter oracle D1, `mtp_cells`, counters; restores after the call's inputs; M0 decides `download_pinned` (§19.3.1) |
| Vision: M-RoPE plumbing in every attention call, invisible below 2,051 tokens; lending across three streams; new BF16 routes also change Qwen3.5 and Quasar vision | Silent output change; corrupt frames; slower images | Bitwise text logits against HEAD, index-key taps, a mandatory > 2K case; load serial, lease waits, C = 2 stress; routes measured first, Qwen3.5 requalified (§19.3.2) |
| Q8: T > 8 costs rest on one measured MMA point; K1b's redundant partial reads; Phase 1b moves tg512 baselines | Smaller gains | M1 sweep with per-shape thresholds; last-arriver fallback; identical-id workloads (§19.3.3) |
| N-gram gate: a long spin under WDDM relies on compute preemption; a TDR resets the whole GPU | Display and other sessions reset | 120 s trap verified once with a 3-5 s delay on an idle GPU, else 2 s (§12.3) |
| Gains near noise (n-gram) and a ±30 % C > 1 model (verification ~20 % optimistic) | Inconclusive steps | Measure-first gates (S0; S3 at ≥ 0.15 ms; concurrency S2 table); ABBA; S6 + S7 only if they beat the gate everywhere |
| Tracks edit the same code | Conflicts, silent regressions | §19.3.0's rules and merge order: `ProgramImpl` declared in `program_impl.h` first (`54deaa2bc`); concurrency owns `cpu_plan`, `cpu_wait` and `stage_kernel` (S3-S5 before Q8 Phase 2 and prefix P7; one `cpu_wait` grid for copying and L2 warming); Q8 Phase 0 before S6, whose −1 skip covers the small dispatch and `moe_combine`; n-gram S1 owns `read_rows` (vision V6 adds no copy); vision's io-placement assertion and the text bit-identity gate after every merge; the prefix tier reserves Vision's pageable media budget (vision rev 2 pins only the tower) |
| Concurrency S7: greedy output at C > 1 differs from C = 1 at near-ties once B·W > 8 | Concurrency-dependent output | Asked at the decision point after S5 (+20-30 % at C = 2, model); otherwise the gate stays (§19.3.5) |

Open questions answered by measurement, not assumption:

- k′ and the depth of prediction;
- LFRU count-halving period, the promotion budget and the shadow-tuning switch threshold;
- the frame pool's KV-loan reclaim frequency at C = 8;
- the MTP pool size against acceptance;
- the KV host tier threshold for each profile, `int8` first;
- the A4 mismatch rate against NVIDIA's runtimes;
- H1-H13 adoption (§17 gates);
- the n-gram multi-issuer read rate (S1b), which decides S4c;
- D2H copy-engine FIFO (prefix M0), the route fraction ρ (M6) and the serving prefill chunk (M9);
- after-draft GPU idle (n-gram S0), QD-1 read latency; c_job, c_col and filler-only experts (S2);
- HEAD's Q8 MMA times at T = 9-64, HC down's L2 bandwidth (Q8 M1); vision `kVisionStepSeconds`.

---

## 22. Sources

- Strata forks: https://github.com/architectds/Strata (`best`, 10 commits ahead of v0.1.38: prefill CPU
  assist, chunk sizing, elastic cache). chimpera/strata-nvfp4 was not read.
- Routing traces and replay: https://github.com/hz1ulqu01gmnZH4/qwen38-freetoken
  (`experiments/2026-09-22_tg-expert-cache-trace-replay/data/*.npz`). Extended replays for this
  design, continuous-session and per-request, are described in §9.3. Zhang, "Reproducible evaluation
  of MoE expert caching", arXiv 2608.07911 (https://github.com/shijiuzhang/moe-cache-eval). SeqMoE,
  arXiv 2609.12978. Angelopoulos et al., "Cache management for MoE LLMs", arXiv 2509.02408.
- Strata: https://github.com/Niko1221/Strata. Read from source at `99f3dbd`: `src/core/verify.cpp`,
  `expert_cache.cpp`, `expert_source.cpp`, `kernels/cpu/pool.cpp`, `ngram/ple_reader.*`,
  `docs/DETAILS.md`, `docs/paper/Strata-Paper.pdf`, and
  `bench/results/2026-09-30-community-rtx-5090/`.
- Strata re-read at `6f32ec0` (2026-10-04; engine 0.1.39 plus the #465, #583 and #646 follow-ups)
  for §3.1 "Strata v0.1.39" and §19.3.6.
  - **History.** The 217 commits of `99f3dbd..6f32ec0` by subject. The diffs or messages of
    `cfd3b72`, `055122c`, `deee447`, `f945515` (#646); `2fbbfa2`, `bebfca9`, `9556298`, `1a56d71`,
    `4795f98` (#465); `895a77b`, `a93c2ac`, `5423a69`, `5f19911` (#583); `cce52db`, `fd95405`,
    `d595a2b` (#533), `7c77eae` (#620), `61479a8` (#458), `1807086` (#528), `e90fbb4` (#587),
    `0e9814a`, `2b47285` (#642), `6d51272`, `d5469a9`, `f23ea57`, `96ccd2a`, `20a2ccf`, `dcf251c`,
    `882764d`, `ccc09c2`, `e809f7f` and `c2d1f19`.
  - **Pre-window commits the review relies on.** `d541220`, `3e31eea`, `bbe3d2a`, `a20f3b5` and
    `0e23f57`.
  - **Docs.** `README.md`, `docs/DETAILS.md`, `docs/HOW_IT_WORKS.md`, `docs/BATCHING.md`,
    `docs/MULTI_GPU.md`, `docs/SECOND_GPU.md`, `docs/UNSLOTH_Q4.md` and `docs/INTEL.md`.
  - **Results.** `bench/results/2026-09-28-prefill-speed/` and `2026-10-03-v100-prompt-attn/`.
  - **Source and setup.** `src/kernels/cuda/qsa_select.cu`, `src/core/mtp.cpp` and
    `src/program/generate.cpp` (scheduling and `adapt()`), and `setup.py`'s defaults.
  - The review, its verifier dispositions and the per-track amendments are in
    `local/workdirs/fn/plans/strata-review.md` and `strata-amendments.md`.
- ninfer-ext: https://github.com/giveen/ninfer-ext. Read from source at `259e819`: `README.md`,
  `docs/maintainer/qwen4-exp-model.md`, `tools/convert/qwen4_exp.py`,
  `src/models/qwen3_5/execution/qwen4_*`, `src/ops/offload_moe/`, and
  `tools/bench/qwen4_prefill_width/`.
- Model sources: https://huggingface.co/Qwen/Qwen3.8-Flash-Next and
  https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4. They could not be fetched from the design
  environment. The quantization composition (W4A4 NVFP4 routed experts with MSE-calibrated weight
  scales; BF16 attention, shared experts and other main-model layers; 128×128 block FP8 MTP experts
  and per-tensor FP8 PLE, both byte-identical to `Qwen/Qwen3.8-Flash-Next-FP8`) is the NVIDIA model
  card's, as quoted by the user. Other facts come from the studied implementations' recorded
  checkpoint constants and are re-verified in M0.
- FreeToken: https://github.com/FlashML-org/FreeToken (offload/cpu/hybrid strategies, disk PLE).
  Community 5090 measurements, profile and trace replay:
  https://github.com/Gamal-ElDeen/FreeToken-serving-nvidia-Qwen3.8-Flash-Next-NVFP4-on-RTX5090-32GB-VRAM-with-128-DDR5,
  https://github.com/hz1ulqu01gmnZH4/qwen38-freetoken, and
  https://github.com/perryh/qwen3.8-flash-next-nvfp4-freetoken.
- SGLang: `python/sglang/srt/models/qwen4_exp*.py`, `docs/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next.mdx`
  (checkpoint composition, RTX PRO 6000 all-resident numbers).
- vLLM: `vllm/models/qwen4_exp/`, `vllm/config/engram.py`, `fused_moe/oracle/nvfp4.py`; tracking
  issue https://github.com/vllm-project/vllm/issues/55922.
- llama.cpp `src/models/qwen4exp.cpp`; KTransformers PR https://github.com/kvcache-ai/ktransformers/pull/2209.
- Papers: arXiv 2312.17238, 2402.07033 (Fiddler), 2401.14361 (MoE-Infinity), 2410.22134 (ProMoE),
  2411.01433 (HOBBIT), 2408.10284 (AdapMoE), 2501.10375 (DAOP), 2502.12224 (FATE), 2502.05370 (fMoE),
  2510.10302 (SP-MoE), 2504.05897 (HybriMoE), 2505.16056 (routing consistency), 2601.07372 (Engram),
  KTransformers (SOSP'25). Most were read through summaries, not in full; claims taken from them are
  re-measured on this model before they decide anything.
- This repository: `docs/maintainer/linear-benchmark.md` §9 (RTX 5090 sustained-read probe of
  1,674.5 GB/s; T=1 GEMV at 90-92% of it; NVFP4 A4 GEMM at 882-985 TFLOP/s),
  `src/core/pdl.cuh`, `src/models/qwen3_5/program/decode.cpp`.
- Kernels and measurements on sm_120, gathered by the review's kernel survey:
  - sm120-fp4, https://github.com/Theodore-Liu/sm120-fp4: `reports/micro-floor-rtx5090-2026-09-29.json`
    (boundary costs of §4.1; streaming rates of 21-215 MB kernels) and its deterministic MoE layer
    and b12x determinism tests.
  - b12x, https://github.com/yatesdr/b12x (`docs/moe-execution-model.md`,
    `docs/stage2-baselines.md`), and FlashInfer's SM12x kernels
    (`flashinfer/fused_moe/cute_dsl/blackwell_sm12x/`, `gemm/kernels/cute_dsl/gemv_bf16_fp4_sm12x.py`).
  - SGLang fused W4A4 MoE for this model on sm_120: https://github.com/sgl-project/sglang/pull/36787.
  - llama.cpp per-layer expert LRU on this model: https://github.com/ggml-org/llama.cpp/pull/27861.
  - MonoMoE, arXiv 2609.04244. GDN decode latency, arXiv 2607.16831. CUDA graphs vs megakernels
    across GPUs, arXiv 2605.30571.
  - Megakernels: Hazy Research, "Look Ma, No Bubbles!" (2025) and
    https://github.com/HazyResearch/Megakernels; Mirage Persistent Kernel,
    https://github.com/mirage-project/mirage (arXiv 2512.22219); Lucebox,
    https://github.com/Luce-Org/lucebox-hub/tree/main/optimizations/megakernel.
  - BF16 GDN state on this model (DGX Spark): https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark.
  - Host-device facts: NVIDIA/gdrcopy README (GPUDirect RDMA GPU requirements), NVIDIA/nvbandwidth,
    the CUDA programming guide on PDL, and published RTX 5090 `deviceQuery` and bandwidth results.
- Quantizer semantics: TensorRT Model Optimizer's NVFP4 quantizer and TensorRT-LLM's FP4
  quantization kernel. §16.2 follows their published rule; M0 re-verifies it against ModelOpt's code.
- This repository, for the fidelity findings: `tools/convert/sources/compressed_tensors.py` and
  `tests/convert/test_sources.py` (ModelOpt multipliers stored as reciprocal divisors),
  [tensor formats](tensor-formats.md) §3.3-§3.4, [paged KV cache](paged-kv-cache.md) §4.3 and §9.3,
  [tree verification](tree-verification.md) (decode-vs-prefill agreement criterion), and
  [CLI](../cli.md) (`--kv-dtype`).

---

## Appendix A. Review findings and changes

The previous version of this design was reviewed against three questions:

- Can it be implemented on the RTX 5090 and CUDA as they are?
- Does it leave performance on the table?
- Is it specific enough to build?

Four requirement changes arrived during the review (A.1). The findings that follow are listed by
severity, and each points to the section that now resolves it.

### A.1 Requirement changes

| # | Requirement | What changed |
|---|---|---|
| R1 | NVIDIA's quantized tensors stay bit-exact, and activations use NVIDIA's formats | The previous recipe moved BF16 dense tensors to NVFP4 and FP8 and requantized the MTP experts to NVFP4; it is withdrawn. Recipe A imports everything exactly (§6.1), with two new formats, `nvfp4_mul` and `fp8_e4m3fn_block128_f32`. Routed experts now compute NVIDIA's W4A4 with each expert's own activation scale, exactly (§16.2), replacing the A16 arithmetic. The A8 CPU route and the reduced-precision miss copies (old H10) are removed. MTP experts keep FP8 in their own pool (§9.1). |
| R2 | QSA supports every Infernix KV profile | §8.10: per-profile tile loaders, profile-independent BF16 index keys, the VQ2/K4V2 window rule, and the host tier for every profile (§9.6) |
| R3 | BF16 tensors may move to 8 bits only where that helps significantly; 8-bit-or-lower tensors stay as NVIDIA stores them | Recipe B (§6.1): W8A16 for the large BF16 matrices, FP8-row then Q8 per class, under a quality gate fixed before measurement (§16.3). It is projected at 1.35-1.6× recipe A for plain decode. |
| R4 | Optimize first for a 262K context with INT8 KV, then the other profiles | The primary configuration (§18) anchors the targets (§1.3), the memory plan (§15.1), the replay and projections (§4.2) and the QSA development order (§8.10) |

### A.2 Errors (the previous text could not be implemented as written)

| # | Finding | Resolution |
|---|---|---|
| E1 | Prefetch was "driven by a device-side scheduler kernel that issues DMA". A CUDA kernel cannot start a copy-engine DMA. | All DMA is issued by a host **transfer agent**. Landing is signalled by stream-ordered `cuStreamWriteValue32` (§8.1, §8.2, §9.4). |
| E2 | The expert kernel could not saturate HBM at T=1. The original grouped kernel gave each expert one CTA. Its first revision used one thread-block cluster per expert, but sm_120's portable cluster size is 8, which leaves 88 of 170 SMs busy for 11 experts. | K7a/K7b: gate/up slices across all SMs, then down row tiles that fuse the combine (§8.5) |
| E3 | The router kernel as specified would run its 2.6 MB GEMV on a handful of SMs | Split GEMV over all SMs, then a last-CTA phase for top-k, classification and publication (§8.4) |
| E4 | The canonical arithmetic was underspecified. Its order could not be both GPU- and CPU-natural, and `expf`, FMA contraction and the h dtype were undefined. | Exact integer W4A4 arithmetic with int64 sums, so the order no longer matters; defined roundings, SiLU, layout and compiler rules (§16.2) |
| E5 | Frames were described as "2 MiB-aligned VMM chunks" while being 2.64 MiB | Frames are 675 × 4 KiB slices of one allocation, and KV page sizes divide a frame (§9.1) |
| E6 | Device-side "recency updates" contradicted the requirement that policy bookkeeping stay off the critical path | The device only reads residency. LFRU, shadow replay and all decisions run in the transfer agent from the route log (§9.4). |
| E7 | The residency states had no `LOADING`, no generation and no reuse rule, which allowed use-after-reuse races | State machine with generations, landing tickets and epoch-based retirement (§9.4, §9.5) |
| E8 | Prefill's wide route defaulted to A16 tensor cores | A4 is NVIDIA's activation format for the experts, so it is the model's semantics on every route (§13) |
| E9 | Infernix's ModelOpt importer stores `fl(1 / weight_scale_2)` as the NVFP4 divisor, which is not NVIDIA's weight unless the scale is a power of two | `nvfp4_mul` keeps the multiplier (§6.1) |
| E10 | "Speculative output equals plain output" and "incremental decode equals prefill" overclaimed. Infernix's dense kernels change tiles with width, and its established criterion is a reported disagreement rate. | Bit-exactness is claimed only for expert outputs across placements and batch shapes; the rest uses the existing criterion (§16.4) |
| E11 | The HC mixers used device-wide barriers, which measured 1.4-2.0 µs on this card, more than a PDL boundary | Two PDL-chained kernels per mixer; the cooperative form is the fallback (§8.3 rule 2) |

### A.3 Performance gaps (the design left speed on the table)

| # | Finding | Resolution |
|---|---|---|
| P1 | HBM sat idle during kernel boundaries and while waiting on the host | Baseline rules: PDL with pre-dependency weight prefetch; stall-time warming through early dependent launch; evict-first weight streams (§8.3) |
| P2 | x was shipped to the host only once a miss was known, so its transfer sat on the miss path | x is written to host memory every layer (5 KiB) before routing (§8.2) |
| P3 | CPU misses always read DRAM cold, with FP32 FMA kernels that were compute-bound on AVX2 | Integer W4A4 kernels, DRAM-bound on every ISA (§10.2); static ownership plus predicted-miss L2 warming, ~3-5 µs per warmed miss (H2, §10.4) |
| P4 | No per-kernel budget: the 4.0 ms GPU assumption was unsupported | Per-kernel byte and time budget, 4.5 ms (B) and 7.3 ms (A) at T=1, grounded in this repo's measured 1,674.5 GB/s sustained read (§8.3) |
| P5 | Recurrence and dense GEMV tiling were constrained by per-head norms in epilogues | Norms move to consumer prologues; recurrence splits state rows across 192 CTAs (§8.3) |
| P6 | Megakernel, chained rounds and BAR mailboxes were unscoped | Specified as options H1 and H6 with measured gates. BAR mailboxes are not available on GeForce and are rejected (§17). |
| P7 | Prompt routing was not used to warm the cache | Option H8 (§17) |
| P8 | The miss decision ignored in-flight DMAs and per-layer completion time | Arbiter minimizes each layer's latest completion, including waiting for in-flight loads and the shared copy engine's queue (§8.6) |
| P9 | Each router was read twice from HBM: once for prediction and once for routing | Router l+1 is loaded evict-last and re-read from L2: 126 MB per token (§8.3 rule 5) |
| P10 | Fixed-order floating-point reductions forced extra passes and partial traffic | Exact int64 expert sums; the down kernel owns rows and fuses the combine (§8.5) |

### A.4 Clarity gaps (not specified well enough to build)

| # | Finding | Resolution |
|---|---|---|
| C1 | No host-device protocol: memory regions, ordering, polling and failure | §8.2 |
| C2 | No configuration surface | §18 |
| C3 | Tests did not cover policy conformance, protocol races or forced placements | §16.4 |
| C4 | PLE hiding assumed an 80-150 µs layer 0 | Exposure model with the budgeted 84 µs (B) or 134 µs (A) layer 0 and measured NVMe latency, and an acceptance rule (§12.4) |
| C5 | Statements that every miss is DRAM-bound | Corrected: plain decode is latency-bound at this capacity (§2, §4.1) |
| C6 | Milestones lacked deliverables | §19 |

### A.5 Projections and targets

- Projections were recomputed with the per-kernel budget, for both recipes, in the primary
  configuration (§4.2).
- The Must targets are unchanged.
- The Stretch targets are the recipe-B projection at a 0.9 hit rate:
  - plain decode 200 tok/s;
  - speculative decode 340 tok/s;
  - C = 8 aggregate 600 tok/s.
- History: the original design set 170 / 230 / 400. The first review revision raised them to
  220 / 330 / 600 on NVFP4 dense weights. R1 forbids those, so plain decode drops to 200.
- The 0.9 hit rate is below the replay traces' 0.969. The targets were set before any measurement
  (§1.3).
