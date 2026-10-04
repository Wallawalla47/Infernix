# Qwen3.8-Flash-Next on one RTX 5090: design

**Status: design proposal, partly implemented.** [§19.1](#191-implementation-status-and-handoff) lists what exists
and what needs the target machine. This is a temporary planning document for the
Qwen3.8-Flash-Next product change. When the implementation lands, its stable content moves into
the active authorities named in [§20](#20-documentation-and-authority-changes) and this file is
removed.

**Revision 2026-10-03.** This version was revised after an expert review for implementability and
peak performance on the target hardware, and after three requirement changes: NVIDIA's quantized
tensors stay bit-exact with NVIDIA's activation formats, QSA supports every NInfer KV profile, and
BF16 tensors may move to 8 bits only where that helps significantly.
[Appendix A](#appendix-a-review-findings-and-changes) lists every finding and what changed.

This design adds `Qwen4ExpForCausalLM` (Qwen3.8-Flash-Next, about 180B parameters) to NInfer.
The target system is:

- one RTX 5090 (32 GB);
- 96 GB of DDR5 host RAM;
- one NVMe SSD.

It adds three mechanisms that NInfer does not have today:

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
7. [Ownership in NInfer](#7-ownership-in-ninfer)
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
19. [Implementation plan](#19-implementation-plan) ([status and handoff](#191-implementation-status-and-handoff))
20. [Documentation and authority changes](#20-documentation-and-authority-changes)
21. [Risks and open questions](#21-risks-and-open-questions)
22. [Sources](#22-sources)
- [Appendix A. Review findings and changes](#appendix-a-review-findings-and-changes)

---

## 1. Goals, baselines and acceptance

### 1.1 Deliverable

The product supports `Qwen4ExpForCausalLM` through the public Engine, CLI, `ninfer-serve`, and
offline CausalScoring. It includes:

- the Text model;
- the MTP layer;
- QSA with **every NInfer KV profile** (`bf16`, `int8`, `fp8`, `nvfp4`, `k8v4`, `vq2`, `k4v2`;
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
  reports 30-40% fewer misses than upstream.

### 3.2 ninfer-ext

ninfer-ext is a fork of NInfer.

| Mechanism | Detail | Consequence |
|---|---|---|
| Recipe | Expert codes and block scales imported as NVFP4, but `weight_scale_2` is stored as NInfer's FP32 **divisor**, a rounded reciprocal of NVIDIA's multiplier, and each layer quantizes activations with **one** input scale, the minimum over its experts' `input_scale`. **Dense layers re-quantized to Q8** (attention, GDN, HC, shared expert, PLE projections). `lm_head` Q6, embedding Q8, router BF16. MTP experts **requantized** from block FP8 to NVFP4. The n-gram table is FP8 with the global scale copied onto every row. | 5.3 GB of device weights, about 3.9 GB of dense reads per token. Neither the represented expert weights nor their activations are NVIDIA's exactly, and the MTP experts lose precision. The per-row scale plane doubles the PLE I/O ranges. |
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
6. **NVMe reads must not wait for the host.** Row identification and the device row cache belong on
   the GPU, and the NVMe latency should be hidden behind layer 0 (both engines gather
   synchronously).
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
 │ Dense Text: BF16 8.6 GB (A) or 8-bit 4.4 GB (B) · MTP dense · MTP FP8 experts   │
 │ GDN and conv state · decode workspace · L0 PLE row cache                        │
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
 │    (67.9 GB) + 512 FP8 MTP experts (2.5 GB)                      │  │ 52.4 GB FP8,   │
 │  embedding rows · L1 PLE row cache · mailboxes · (KV host tier)  │  │ 4 KiB blocks   │
 │  transfer agent ─ cache policy, all DMA, residency updates       │  └────▲───────────┘
 │  CPU expert engine ─ spin workers, exact W4A4 arithmetic         │       │
 │  NVMe agent ─ io_uring SQPOLL/IOPOLL, L1 cache ──────────────────────────┘
 └──────────────────────────────────────────────────────────────────┘
```

A decode round at C = 1 without speculation. The host loop is not on the critical path, and the
host learns about a layer only when it has a CPU-served miss:

```text
round r: embed(host row) ─ [PLE kernel waits for rows: L0 hit or NVMe agent] before layer 1
 per layer l:
   K1a/K1b HC_attn ─ K2/K3 (GDN) or K2q/K2b/K3a/K3b (QSA) ─ K4 out_proj + inject ─ K5a/K5b HC_mlp (x → host)
   K6 router(l, l+1): top-10 ─ residency ─ jobs ─ [miss_req → CPU] ─ [prefetch_req → agent]
   K7a gate/up slices: A4(x) ─ exact W4A4 ─ SwiGLU ─ A4(h)
   K7b down row tiles ─ combine with the CPU's y_e (waits only if the layer had a CPU miss) ─ inject
 head ─ sample ─ n-gram ids(t+1) ─ L0 probe ─ NVMe requests ─ token egress ─ round_done
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
- **Device memory is one pool.** Expert frames, KV, prefill arenas and vision weights share it,
  and reserved but unwritten KV holds experts on loan (§9.2).

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
| MTP routed experts (512) | FP8 E4M3, 128×128 blocks with FP32 multipliers, byte-identical to `Qwen/Qwen3.8-Flash-Next-FP8` | **Exact import** as `fp8_e4m3fn_block128_f32` (below). Activations follow the scheme the checkpoint declares: per-token 1×128 FP8 groups (W8A8) if dynamic, BF16 if weight-only. | Same | Byte-for-byte equal to the source. MTP affects only acceptance, never output. |
| N-gram table (128 shards) | FP8 E4M3 plus one BF16 scalar scale | **Exact import** into the NVMe volume of §12.2. One scalar scale, no per-row plane. | Same | One I/O per row instead of two |
| GDN q/k/v/z projection and `out_proj`; QSA QKVG and `o_proj`; shared experts; HC mixers (down and up); PLE key/value projections; `lm_head` | BF16 | BF16 | **8-bit, W8A16.** `fp8_e4m3fn_row_bf16` (producer `fp8_row_maxabs`) per class; `q8_g32_fp16` (`grouped_absmax`) for a class where FP8 fails §16.3 and Q8 passes; BF16 for a class where neither passes. | 4.2 GB fewer dense bytes per token and ~1,520 more frames (§4.2). These classes are 97% of the dense bytes. |
| Router, shared-expert gate, GDN `a`/`b`, QSA indexer projection | BF16 | BF16 | BF16 | Small (3% of dense bytes), and routing- or selection-sensitive |
| MTP dense (its QSA block, HC, shared expert, router, projections) | BF16 | BF16 | BF16; 8-bit only under option H13 | Affects only acceptance; 0.17 GB |
| Token embedding | BF16 | BF16, **host-resident** (§6.3) | Same | Saves 1.27 GB of VRAM, about 480 frames |
| Norms, `A_log`, `dt_bias`, conv weights | BF16 / FP32 | Direct | Direct | — |
| Vision | BF16 | BF16 | BF16 | Not on the decode path. It is selected at startup, and its device weights are borrowed from the frame pool (§9.2). |

**Two new weight formats** are needed, each defined in [tensor formats](tensor-formats.md) with an
exact decode oracle:

- **`nvfp4_mul`**: the words of `nvfp4` (E2M1 codes, one E4M3FN scale per 16) with an FP32
  **multiplier** `m_w` per matrix: `W[n,k] = e2m1(c[n,k]) · e4m3fn(s[n,⌊k/16⌋]) · m_w`.
  - NInfer's `nvfp4` divides by `d_w`. Its ModelOpt importer stores `fl(1 / weight_scale_2)` as
    `d_w`, which equals NVIDIA's weight only when the scale is a power of two. ninfer-ext inherited
    the same rounding (§3.2).
  - By the rule of [tensor formats §3.4](tensor-formats.md#34-fp8_e4m3fn_row_bf16), a multiplier
    coefficient is a different format, not a variant of `nvfp4`.
  - Its only producer is `import_encoded`.
- **`fp8_e4m3fn_block128_f32`**: E4M3FN codes with one FP32 multiplier per 128×128 block, edge
  blocks truncated: `W[n,k] = e4m3fn(c[n,k]) · s[⌊n/128⌋, ⌊k/128⌋]`. The represented weight is the
  exact product, which binary64 represents exactly. Its only producer is `import_encoded`.

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

### 6.2 Expert storage layout `nvfp4_expert_rg16_v1`

The pinned host bank is read by the GPU (DMA into a frame, then the expert kernels) and by the CPU
(in-place computation). Only one copy fits in 96 GB, and NInfer forbids runtime repacking. The
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
| `host_pinned` | Read with `O_DIRECT` into a huge-page, `cudaHostRegister`ed host arena. Never staged through the page cache. | Routed expert banks (NVFP4 and the FP8 MTP experts), embedding |
| `stream` | Never materialized. The Model holds an open file region plus its layout. | N-gram table |

Residency is a property of a Use and the startup options, not of the stored bytes. The same artifact
loads all-device on a hypothetical larger card. The n-gram table is written as its own volume file
(`*.ninfer.ngram`), so it can be placed on a different NVMe drive and a 4 KiB-aligned layout is
guaranteed.

---

## 7. Ownership in NInfer

This is an **explicit product change**: a new model architecture. It follows the existing
boundaries ([engine architecture](engine-architecture.md)).

| Component | Owner | Location (proposed) |
|---|---|---|
| Mathematics, config, binding, block order, MTP alignment, frontend reuse (Qwen3.5 tokenizer, template, vision) | Model | `src/models/qwen4_exp/` |
| Host-pinned expert banks, embedding bank, n-gram file region | Immutable Model data, through a new artifact residency class (§6.3) | `src/artifact/materializer.*` |
| Weight formats `nvfp4_mul` and `fp8_e4m3fn_block128_f32`, layout `nvfp4_expert_rg16_v1`, exact import of ModelOpt multipliers and input scales | Artifact format and layout registries; converter source | `src/artifact/formats.cpp`, `layouts.cpp`; `tools/convert/sources/` |
| Pinned huge-page host arena; `O_DIRECT` + io_uring reader; mapped mailbox and ring primitives; spin-worker pool with barriers | Core (model-independent physical and transfer primitives) | `src/core/host_pinned_arena.*`, `direct_io_ring.*`, `mapped_mailbox.*`, `spin_worker_pool.*` |
| Frame pool, residency table, staging, loans, epochs; transfer agent (policy, DMA, residency writes); NVMe agent; L0/L1 row caches; CPU-engine lifetime | **Program** (mutable state, placement, agents; all allocated at startup) | `src/models/qwen4_exp/program/` |
| Cache policy algorithms (LFRU, shadow replay) | Program, behind a policy interface conformance-tested against `tools/expert_cache_replay` | `src/models/qwen4_exp/program/expert_cache/` |
| Router + top-k + residency classification + prediction; narrow-route expert kernels (GPU and CPU); wide-route grouped GEMM; combine; the MTP drafter's masked routing | **Op family** `offloaded_sparse_moe` (closed contract: output = MoE(x) independent of residency; residency, agents and mailboxes are execution resources). The drafter's routing mask is a semantic input, because it changes the result. | `include/ninfer/ops/offloaded_sparse_moe.h`, `src/ops/offloaded_sparse_moe/{route,gpu,cpu,wide}/` |
| HC mixer and HC inject | Op `hyper_connection` | `include/ninfer/ops/hyper_connection.h` |
| QSA prep, index-key pooling, block select, sparse attention over every KV profile | Ops `qsa_prep`, `qsa_select`; extended `softmax_attention` consumer with the existing VQ2/K4V2 window read rule | `include/ninfer/ops/qsa.h` |
| PLE gather, projections, gate and dilated conv (stateful) | Op `ple_ngram_injection` | `include/ninfer/ops/ple.h` |
| N-gram row ids | Op `ngram_row_ids` (exact integer oracle) | same |
| Canonical routed-expert arithmetic: A4 quantizer, E2M1/E4M3 integer decode, int64 block accumulation, `exp_c` (one header compiled for CPU and GPU) | Op common code | `src/ops/common/canonical_math.h` |
| Draft-length policy with miss cost | Program speculative backend | `src/models/qwen4_exp/program/speculative/` |
| Calibration | Runtime (profile schema), Core (probes), product (`ninfer-calibrate`) | §14.7 |

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
| **NVMe agent** | 1 pinned host thread, io_uring | PLE row reads (§12) |
| **Engine worker** | Existing | Launches graphs and commits rounds. It does no PLE gathers and no expert planning. |

The default core budget is: engine worker 1, transfer agent 1, NVMe agent 1, and CPU workers =
physical P-cores − 3. All agents spin only while a request is active, and park on a futex when the
Engine is idle.

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
- **Bounded spins.** Every device spin has an iteration bound worth about 2 s. On expiry the thread
  writes `agent_error`, skips the computation (outputs are undefined), and the engine worker turns
  that round into an Engine-wide failure ([engine architecture §7.4](engine-architecture.md)). Host
  agents have a 2 s watchdog on `round_done` progress.

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
3. **Stall-time warming.** A kernel that waits on the host (K7b for CPU misses, the PLE kernel for
   rows) executes `griddepcontrol.launch_dependents` before it waits. The next kernel's CTAs then
   start on the free slots and prefetch their weights, and the waiting CTAs prefetch the following
   kernel's weights into L2 (bounded to half of L2). The stall becomes progress on the next layer.
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
   - the new token's n-gram row ids (§12.3);
   - the L0 probe;
   - NVMe request publication;
   - the token written to the mapped egress slot;
   - `round_done` (§8.2).
2. Ingress needs no host gather: embedding and PLE inputs are device- or agent-produced.

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

Every NInfer profile (`--kv-dtype`) is supported, and the Main Text and MTP layers use the same
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
| `vision` | Vision tower weights and activations while a vision request runs |

The frame count is a startup decision (§15.1). In the primary configuration (`int8` KV, 262K
context) the pool has 8,153 frames with recipe A and 9,675 with recipe B, of which the KV
reservation can take up to 1,381 as the context fills. Frames are 4 KiB-aligned slices of one
2 MiB-page allocation, so a frame spans 2-3 GPU pages. The TLB reach of 2 MiB pages covers the
whole pool.

**MTP experts do not use frames.** An FP8 MTP expert is 4,916,400 B, 1.78 frames. The drafter's
experts therefore live in a separate fixed pool of slots (128 by default, 0.59 GiB, sized by
calibration), with their own LFRU in the transfer agent. The drafter never waits for them (§11.2).

### 9.2 Lending reserved memory, the key to sharing VRAM with KV

NInfer's admission invariant requires an Active request to hold its **complete** KV growth
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
- **The same mechanism covers the rest:** prefill arenas (§13), vision weights, and MTP draft
  buffers borrow frames and return them, the borrowed experts being re-admitted lazily by the
  policy instead of re-copied eagerly. Strata re-copies them eagerly after every prompt, which
  costs short prompts up to a third of their speed.

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
   - For long sessions, `f` is halved every 4,096 rounds (196,608 ticks) to bound staleness. This value is untested
     and is fixed in M1.
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

This design adopts the same idea in NInfer's paged-KV terms, for every KV profile:

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
- **PLE rows** for each draft token are requested from the NVMe agent as soon as the draft token
  exists, so they normally sit in L0 before the verify forward needs them (§12.3).

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
- Measured random 4 KiB reads at QD1 on current PCIe 4.0/5.0 NVMe drives take 33-44 µs with polled
  completion, inside that window.
- The table must stay off the page cache. RAM is fully budgeted (§15.2).

### 12.2 NVMe layout

- Rows are 160 B, 25 rows per 4 KiB block, and never straddle a block. That is a 2.4% padding cost:
  12,800,062 blocks, 52.4 GB on disk.
- The single BF16 scale is held in the Model.
- **One row is one 4 KiB `O_DIRECT` read.** Rows of one head are contiguous in the head's prime
  range, so prefill reads that touch nearby rows coalesce into one block.
- The file is its own volume (`*.ninfer.ngram`). It may live on a different NVMe drive from the
  expert banks, which are only read at load.

### 12.3 Three-level row cache with device-driven requests

| Level | Size (default) | Mechanism |
|---|---|---|
| L0: device | 64 MiB ≈ 400K rows | Open-addressing hash on the row id, 4-way buckets with CLOCK replacement. Probed and filled only by decode/prefill kernels. |
| L1: host | 2 GiB ≈ 13M rows | Pinned, owned by the NVMe agent, CLOCK replacement. Probed before any I/O. |
| L2: NVMe | — | io_uring with registered files and fixed buffers, `O_DIRECT`, SQPOLL on the agent's core, and IOPOLL when the driver has polled queues (`nvme.poll_queues > 0`). Queue depth up to 256. The SQPOLL thread sleeps after `sq_thread_idle`, and waking it costs ~30 µs, so the idle timeout outlasts a request. |

**Decode sequence for token t+1:**

1. The sampling kernel's epilogue computes the 16 row ids (`ngram_row_ids`: exact integer
   arithmetic, XOR of signed 64-bit products, prime modulus).
2. It probes L0. Hits record the L0 slot. For misses, it writes `{row id, landing slot}` records to
   the NVMe request ring (mapped) and advances the ring's sequence word.
3. The NVMe agent spins on the ring:
   - L1 hits are copied into the mapped landing slot;
   - L1 misses are submitted as reads into registered pinned buffers mapped to the device;
   - each completion writes the landing slot's sequence word (release).
4. Layer 0 runs (K1-K7).
5. The PLE kernel (one launch, before layer 1's K1):
   - waits, under the §8.2 polling rules, for its missing rows' sequence words;
   - gathers the 16 rows (L0 or landing slot), decodes each FP8 value times the scalar scale
     (exact in FP32, then rounded to BF16, as the checkpoint defines), and computes the key/value
     projections, gate, grouped norms, dilated conv (state update) and the injection into R;
   - inserts landed rows into L0.

**Speculative windows.** Draft tokens' rows are requested when the draft token is produced (§11.2).
Only the bonus token's rows are requested at the round start.

**Prefill.** For chunk i+1, the row ids are computed on the GPU, deduplicated, and probed in L0/L1.
The misses are sorted by block and read at QD 256 while chunk i computes. Rows then reach the device
in one DMA per chunk. A 32K prompt needs at most 512K distinct rows; at the drive's measured IOPS
that is ~0.5-1 s, hidden behind ~4-5 s of compute.

**Power-state stalls.** The NVMe agent issues a periodic tiny read while a request is active.
Strata measured 50-150 ms autonomous power-state exit stalls on some drives.

### 12.4 Exposure and its acceptance

- An L0 or L1 hit never waits.
- An L2 read exposes `max(0, t_nvme − t_layer0)`. With the GPU at the §8.3 budget, a drive whose
  polled QD1 read latency, plus submission, exceeds ~80 µs (recipe B) exposes part of every L2 miss
  in plain decode.
- Speculative rounds hide draft tokens' reads entirely.

M6 publishes per-token PLE wait time and L0/L1/L2 hit rates for each corpus. If the mean PLE wait
exceeds 1% of the token, calibration (§14) raises the L1 size within the RAM plan. If it is still
above 1%, the documented drive requirement is tightened (a fast-latency NVMe for the table).

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
- **PLE rows** for chunk i+1 are computed on the GPU, deduplicated, probed in L0/L1, sorted by
  block and read at QD 256 while chunk i computes (§12.3).
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
| **`ninfer-calibrate`** | Once per machine and artifact, and again when hardware, driver or BIOS changes (the startup probe says when) | `--quick` ~2 min; `--full` ~15-25 min | Host capability measurements, CPU kernel variant and worker set, page backing, NVMe I/O mode, and structural choices that need a restart or reallocation. Verified end to end. |
| **Online adaptation** | Continuously while serving | Off the critical path | Continuous, workload-dependent parameters: miss arbiter split (§8.6), prefetch budget and k′ (§8.7), cache parameters (§9.3 shadow replay), draft length (§11.3), prefill CPU share (§13). These start from the calibrated values instead of generic defaults. |

**Why both an explicit calibration and online adaptation.**

- Online adaptation can only explore what it can change safely mid-request: continuous knobs with
  smooth effects.
- Worker count and placement, page backing (THP vs hugetlbfs 2 MiB or 1 GiB), the CPU kernel
  variant, NVMe I/O mode, the RAM split between the PLE L1 cache and the KV host tier, and the
  long-context KV-tier threshold are **structural**. They need a restart or a reallocation, or they
  behave discontinuously. Exploring them online would cause latency spikes.
- A calibrated starting point also means the first requests run at full speed. Online estimators
  would otherwise have to converge from generic defaults.

### 14.2 What `ninfer-calibrate` measures and decides

**Stage 0: inventory.** Instant, nothing timed. It reads:

- GPU UUID, VBIOS, driver and CUDA versions, and the negotiated PCIe link (generation and width,
  from NVML);
- CPU model, ISA flags, P/E or CCD topology (cpuid, sysfs);
- installed memory, DIMM count and speed (sysfs/SMBIOS when readable);
- NUMA layout;
- NVMe model and firmware, and whether polled queues are enabled (`nvme poll_queues`);
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
| NVMe random 4 KiB reads at QD 1-256 on the n-gram file, for `O_DIRECT` + io_uring with and without SQPOLL/IOPOLL | Random row ids | I/O mode, queue depth, expected PLE latency; whether layer 0 hides it (§12.3) |
| Huge pages | DMA and CPU bandwidth over THP 2 MiB, hugetlbfs 2 MiB and 1 GiB; `cudaHostRegister` time of the 70 GB arena for each backing | Page backing of the expert arena; startup time |
| Host RAM budget | Free and reclaimable RAM after the pinned expert arena, the page cache the n-gram volume needs and the OS reserve | One joint budget for pinned arena, L1 PLE row cache and KV host tier, so pinning never starves the page cache (TensorSharp sizes these separately and can over-commit) |
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
     - NVMe mode and queue depth;
     - prefetch on/off, k′ and depth;
     - RAM split between the L1 row cache and the KV host tier;
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
   probe plus model-based defaults, and recommends `ninfer-calibrate`.

Explicit command-line options always override profile values. `--calibration off` ignores the
profile, which is needed for reproducible benchmarking. `--calibration PATH` selects one.

### 14.4 What stays fixed for the RTX 5090

These choices depend only on the GPU and the model, and are made once at development time with
NInfer's route-development evidence ([op development](op-development.md)). They are never
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
| Static PLE I/O settings (256 in flight, 1M-row cache) | Measured NVMe mode and queue depth; L1 size from the RAM plan and Stage 3 |
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
| `ninfer-calibrate` (CLI, corpus, Stage 3 orchestration through the public Engine, report) | Product (`apps/`) |
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
| L0 PLE row cache | 0.06 | 0.06 |
| Slack | 0.40 | 0.40 |
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

### 15.2 Host RAM: 96 GB (2 × 48 GB = 96 GiB; ≈ 93 GiB ≈ 100 GB visible)

| Item | GB |
|---|---:|
| Expert banks: 24,576 × 2.7648 MB NVFP4 + 512 × 4.9164 MB FP8, pinned huge pages | 70.5 |
| Embedding (BF16), pinned | 1.27 |
| L1 PLE row cache | 2.1 (configurable 0.5-4) |
| Mailboxes, landing buffers, io_uring buffers | 0.2 |
| Host checkpoint tier (prefix cache), capped | 2.0 (configurable) |
| QSA KV host tier (§9.6) at 262K: **3.6 (`int8`, primary)**; 0.9 (`vq2`), 3.5 (`fp8`), 7.0 (`bf16`) | 0-7.0 |
| Process: tokenizer, server, frontend | ~1.5 |
| **Total NInfer** | **~81 in the primary configuration at 262K**; ~77.5 below the tier threshold; ~84.5 with a `bf16` tier |
| OS, desktop, page cache headroom | ~19 in the primary configuration; ~22 or ~15 |

Additional rules:

- **Loading.** Experts are read with `O_DIRECT` straight into the pinned arena (~10 s from a 7 GB/s
  drive). Loading never doubles memory through the page cache. Calibration measures
  `cudaHostRegister` time for each page backing (§14.2).
- **Startup check.** Startup verifies `RLIMIT_MEMLOCK` and `MemAvailable` against this plan,
  including the host tier the chosen context and KV profile need, and fails with a precise message.
  It does not swap.
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
| MTP experts (`fp8_e4m3fn_block128_f32`) | FP64 with exactly decoded block-scaled weights; activations per the declared scheme | Drafter shapes, T = 1-4 |
| `hyper_connection` | FP64 | BF16 or 8-bit weights decoded exactly; grouped norm per stream; split-K partial order fixed |
| `qsa_prep`, `qsa_select` | FP64 block scores | Near-tie allowance as in the existing selector contracts; dense equivalence while n ≤ 2051; pooled key completion at block boundaries; identical selection under every KV profile |
| QSA attention | FP64 attention over each profile's decoded K/V | Every `--kv-dtype`, with that profile's existing decode criterion; VQ2/K4V2 exact-window positions |
| `ple_ngram_injection` | FP64 | FP8 rows decoded with the scalar scale; conv state transition; EOS restarts |
| `ngram_row_ids` | **Exact integer** | Signed 64-bit products, primes 20,000,003-20,000,171, offsets; EOS and sequence start |

The fork's FP64 Python reference may be consulted as a cross-check of the published mathematics.
NInfer's oracle is written from the upstream definitions, consistent with [§1.4](#14-non-goals).

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
  - Speculative and prefill agreement follow NInfer's existing criterion. Dense layers select
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
- **PLE.** L0, L1 and NVMe service give exact equality.
- **Policy conformance.** For recorded route logs, the transfer agent's LFRU victim and promotion
  decisions equal `tools/expert_cache_replay`'s decisions, step for step. Ties are broken by expert
  id.
- **Protocol stress.** Injected random delays (0-500 µs) are applied in:
  - CPU workers;
  - the transfer agent;
  - the NVMe agent;
  - DMA completion.

  Outputs stay exact, and no frame is reused before its retire epoch. The latter is checked by a
  debug build that poisons retired frames.
- **Fault injection.** Each of these reaches the Engine-wide failure path within the 2 s bound:
  - an NVMe read error;
  - CPU worker death;
  - an agent stall;
  - a device spin timeout.

### 16.5 Precision boundaries and measured quality (2026-10-04, RTX 5090)

Every boundary was checked against the upstream code (Transformers `qwen4_exp` with its Qwen3.5 and
Qwen3-Next parents, SGLang `qwen4_exp.py`) and against Strata's kernels. A boundary is never
less precise than upstream. Where more precision was possible, it was **measured** on real text
and kept only if it did not hurt.

**Method.** Three frozen texts are scored teacher-forced on identical token ids: 511 positions
of C++ (`src/core/layout.cpp`), 1,023 of a document (`README.md`) and 1,023 of a chat transcript
in the model's chat template. Perplexity is taken over the actual next tokens. Differences are
paired per position, as the mean ΔNLL ± its standard error. Tools:
`ninfer_qwen4_exp_forward_real_test --dump-logits` and `tools/flash_next/strata_compare.py`.
Strata runs the user's own `strata-unsloth-ud-q4_k_xl.json` (UD-Q4_K_XL, INT8 KV) plus
`--short-read 4096` and `STRATA_LOGPOS`, as in its `docs/UNSLOTH_Q4.md`. Per-op error is checked
separately on NInfer's own taps with `tools/flash_next/block_check.py` (FP64 oracle) and
`upstream_check.py` (Transformers modules in BF16 with the real weights).

| Boundary | Transformers / SGLang | Strata | NInfer | Evidence and decision |
|---|---|---|---|---|
| Dense weights | BF16 | Q4_K-Q8_0 mixes | BF16, word-exact | The main reason NInfer beats Strata on chat (below) |
| Routed experts | NVFP4 W4A4 | Q4_K/Q5_K gate-up, Q5_1/Q8_0 down, Q8_1 activations | NVFP4 W4A4, canonical arithmetic | No 8- or 16-bit activation path is needed: NInfer's overall quality is already better than Strata's |
| Residual stream (4 × 2,560) | BF16 | FP32 | **BF16** | FP32 tried. All op chains stayed exact (median residual error 0.105 % → 0.03 %), but perplexity got **worse**: +0.030 ± 0.009 nats overall, +0.051 ± 0.015 on chat. The model expects the BF16 rounding it was trained and calibrated with. Rejected |
| Router logits | BF16 Linear output, FP32 softmax | FP32 | **FP32** (`projection_fp32`) | BF16 logits make experts within one BF16 step (0.03 at logit ≈ −4.3) tie, and the lower id always wins. Measured BF16 − FP32: +0.008 ± 0.008 nats (code +0.022 ± 0.010). No cost. Kept |
| LM-head logits | BF16 | FP32 | **FP32** (`projection_fp32`) | Perplexity unchanged (BF16 − FP32 = −0.0002 ± 0.0004 nats). BF16 logits create exact ties that flip greedy top-1 at 1.3 % of positions. The 1 MB output row is free next to the 1.27 GB weight read. Kept, subject to its decode kernel matching the BF16 GEMV's bandwidth (M6) |
| RMSNorm / hyper-connection norm | FP32 inside, one BF16 rounding | FP32 | Same | Attention-side mixer error 0.13-0.23 % vs Transformers' 0.27-0.35 % |
| GDN gated norm | Three roundings (normalized value, weight product, output) | – | One rounding | GDN error 0.35-0.50 % vs Transformers' 0.42-0.65 % |
| QSA v, gate, gated product | BF16 | – | BF16 | 0.43-1.8 % vs Transformers' 0.48-3.2 %. The excess over other ops is cancellation in deep layers' `o_proj` sums: emulating these three BF16 roundings leaves ≤ 0.09 % |
| QSA indexer pooled key, scores | FP32 mean → BF16, FP32 scores | – | Same | – |
| KV cache | BF16 | INT8 | INT8-G64 with Hadamard keys (primary), BF16 | INT8 costs +0.005 perplexity overall against BF16 KV (4.542 → 4.564) |

**Result against Strata** (NInfer recipe A with INT8 KV, Strata UD-Q4_K_XL with INT8 KV):

| Text | Positions | Perplexity NInfer | Perplexity Strata | ΔNLL (NInfer − Strata) | Top-1 = next: NInfer / Strata | Top-1 agreement | KL(Strata ‖ NInfer) mean / p99 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Code | 511 | 1.9052 | 1.9051 | +0.000 ± 0.016 | 84.0 % / 84.5 % | 95.1 % | 0.043 / 0.55 |
| Document | 1,023 | 9.5874 | 9.7671 | −0.019 ± 0.017 | 51.7 % / 52.4 % | 84.1 % | 0.074 / 0.84 |
| Chat | 1,023 | 3.3611 | 3.8692 | **−0.141 ± 0.024** | 71.0 % / 70.6 % | 90.7 % | 0.109 / 1.67 |
| All | 2,557 | 4.564 | 4.864 | **−0.064 ± 0.012** | | | |

NInfer ties Strata on code and the document, and is significantly better on chat. KL is taken over
Strata's top 20 plus a bucket for the rest. With BF16 KV, NInfer's overall perplexity is 4.542.
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
| `--ngram-file PATH` | next to the artifact | N-gram table volume, which may sit on another NVMe drive |
| `--ple-device-cache-mib N` | 64 | L0 row cache (§12.3) |
| `--ple-host-cache-mib N` | 2048 | L1 row cache |
| `--kv-host-tier auto\|off\|TOKENS` | `auto` (65,536) | QSA KV host tier threshold (§9.6) |
| `--hugepages auto\|1g\|2m\|thp` | `auto` | Backing of the pinned expert arena |
| `--calibration auto\|off\|PATH` | `auto` | §14.3 |
| `--spec mtp`, `--draft-tokens`, `--fixed-draft`, `--ngram-draft-tokens`, `--draft-tree-nodes` | existing | Existing speculation options; draft length policy of §11.3 |

The recipe is chosen at conversion, not at startup: `qwen3_8_flash_next_nvfp4` (A, exact) or
`qwen3_8_flash_next_nvfp4_dense8` (B, 8-bit dense). It is recorded in the artifact and **changes
output**.

Command: `ninfer-calibrate ARTIFACT [--quick|--full] [--corpus DIR] [--out PATH]` (§14.2).

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
| **M9** | Calibration | §14: startup probe, `ninfer-calibrate --quick/--full`, profile and drift check, Stage 2 fit, Stage 3 A/B | On at least two different hosts, calibrated ≥ uncalibrated (or tie within noise) on the calibration **and** held-out corpora; every candidate reported; recalibration reproduces the choices |
| **M10** | Head-to-head | Campaign vs Strata UD-Q4_K_XL at 262K context with 8-bit KV (the user's Strata configuration) and vs ninfer-ext, on the target machine; serving C = 1-8; 4K/32K/128K/256K actual context; every KV profile; quality report | §1.3 table filled with every result, favorable or not |

The milestones on the critical path to the decode targets are M2 → M3 → M5 → M6. M1 runs in
parallel and must finish before the policy is frozen in M5.

### 19.1 Implementation status and handoff

A briefing for the next session, with a file map and commands, is in the
[handoff note](qwen3_8-flash-next-handoff.md).

The first pieces were built on a development VM without a GPU or Hugging Face access: a 4-vCPU
Xeon at 2.1 GHz with AVX-512 VNNI. nvcc 13.4 from PyPI was used to compile only: the project's CMake
build configures and compiles there with a stub `libcuda.so`. Everything below compiles, and its
host tests pass there. Nothing has run on the RTX 5090 or touched the real checkpoint.

| Piece | Where | Verified on the VM | Remaining on the target machine |
|---|---|---|---|
| Canonical W4A4 arithmetic (§16.2): E4M3/E2M1 encoders, A4 quantizer, `exp_c`, `silu_c`, epilogues | `src/ops/common/canonical_math.h` | Encoders against grid enumeration; BF16 rounding; `exp_c`/SiLU exhaustively over BF16 against FP64; exact int64 against FP64 | GPU build of the same header; exhaustive CPU-GPU equality (M3) |
| CPU narrow expert engine: scalar, AVX2, AVX-VNNI, AVX-512 VNNI | `src/ops/offloaded_sparse_moe/cpu/`; test `ninfer_offloaded_moe_cpu_test` | Bit equality across ISAs, batch and split invariance, golden output hashes (`kGolden1`, `kGolden4`) with GCC, Clang and -O0 | Throughput on the target (`host_probe`); the multi-column optimization of §10.2; the worker team and handshake (§10.3); AVX-VNNI-INT8 |
| GPU narrow route, first version: quantize, gate/up + SwiGLU + A4(h), and down kernels using `dp4a` and the canonical header | `src/ops/offloaded_sparse_moe/cuda/narrow_expert.*`; test `ninfer_offloaded_moe_cuda_test` | Compiled for `sm_120a` inside the project's CMake build (nvcc 13.4); the canonical header has no contractible float expressions | Run the test: it must reproduce `kGolden1`/`kGolden4`, equal the CPU engine bit for bit (device and mapped-host records), and match the canonical scalar functions exhaustively over BF16 and around every encoder boundary. Then the optimized K7a/K7b of §8.5 (tickets, PDL, TMA ring, persistent items, IMMA for n ≥ 3), qualified against this version |
| Formats `nvfp4_mul`, `fp8_e4m3fn_block128_f32`; layouts `nvfp4_expert_rg16_v1`, `block128_scale_v1`; codecs; ModelOpt source readers | `tools/artifact/`, `tools/convert/sources/modelopt.py`; tests in `tests/artifact/`, `tests/convert/` | Word-exact round trips; C++ and Python agree on rg16 row sums | The recipes A and B themselves (M2). C++ registration is done: `QType::NVFP4_MUL` and `FP8_E4M3FN_BLOCK128_F32`, layouts `ExpertRg16` and `Block128Scale`, `expert_bank_planes` and `block128_planes`. The reader test and `ninfer_artifact_flash_next_interop_test` (Python writer to C++ reader) pass. [Tensor formats](tensor-formats.md) §3.5-3.6 and [storage layouts](storage-layouts.md) §6-7 define them |
| CPU worker team (§10.3): static ownership, Phase A/B barrier, caller as worker 0, spin-then-park; allocation-free miss path | `src/ops/offloaded_sparse_moe/cpu/expert_team.*`; test `ninfer_offloaded_moe_team_test`; latency section in `host_probe` | Output equals `expert_forward` bit for bit for 1-40 workers and mixed jobs; 60 rounds with parking; ThreadSanitizer clean in `--quick` mode (1, 3 and 4 workers, 12 rounds) | Pinning to physical cores and CCD split; the mailbox handshake with the GPU (`miss_req`, `done_seq`, §8.2); latency of 1-2 cold misses on the target (`host_probe` team section) |
| Host expert cache: LFRU, residency words, frame epochs, deferred-load controller | `src/models/qwen4_exp/program/expert_cache/`; test `ninfer_qwen4_exp_expert_cache_test` | Victims identical to `tools/expert_cache_replay` on a 1,200-group fixture; device/agent simulation with no frame reused early | The transfer agent around it: copy stream, `cuStreamWriteValue32`, route log, loans (M5) |
| PLE n-gram row ids (§12): exact multipliers, per-head primes, EOS-closed windows | `tools/flash_next/ngram.py`; `src/models/qwen4_exp/frontend/ngram_hash.*`; tests `test_flash_next_ngram.py`, `ninfer_qwen4_exp_ngram_hash_test` | Python loop specification equals an independent tensor formulation, and its primes equal `sympy.nextprime`. C++ equals the Python fixture on 800 positions across two layers | Real config values from `inspect_checkpoint` (`ngram_size`, `heads_per_ngram`, base, divisor, seed, EOS); the device kernel (§12.3) qualified against this |
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
3. Run `ninfer_offloaded_moe_cuda_test`. It must reproduce `kGolden1` and `kGolden4` bit for bit. Then optimize K7a/K7b against it (M3).
4. Optimize the CPU kernel's multi-column path and re-measure it.

### 19.2 Status on the RTX 5090 (2026-10-04)

Work on the target machine (Windows, CUDA 13.4, RTX 5090, 96 GB DDR5), branch
`claude/wonderful-ritchie-65xtnh`. Qwen4Exp is its own architecture (`src/models/qwen4_exp`), as
the user asked, not part of the Qwen3.5 family.

| Milestone | State |
|---|---|
| M2 recipe A | Done. Converted from `nvidia/Qwen3.8-Flash-Next-NVFP4` with no requantization: 24,576 experts, 1,638 tensors, every input scale and all 320,001,536 n-gram rows word-exact. Loads in 25 s with unbuffered reads straight into 64.47 GiB of pinned memory (8.03 GiB device). The loader refuses a pinned block that would leave less than 8 GiB of RAM. |
| M3 ops | Forward ops in place and checked per op chain against the FP64 reference on the model's own activations (§16.5): `hyper_connection`, `ple`, `qsa` (BF16 and INT8 KV), `offloaded_sparse_moe` layer kernels, `projection_fp32`, `rows`. Standalone oracle tests per op are still to be written. |
| M4 functional | Done for text generation through the public Engine (`ninfer`, `ninfer-serve`, `ninfer_bench`): `EngineCore` over the hybrid-manager surface, whole-extent KV reservation at admission, `--ngram-volume`. Not yet: CausalScoring, vision, prefix cache, MTP. |
| M5 expert cache | First version: free VRAM (less 1.5 GiB) as frames (7,804 on the 5090), LFRU via `expert_cache::CacheController`, promotions on a copy stream, at most one promotion per layer call in decode and 16 in prefill. No prefetch, loans, CPU expert engine or saved profile yet. |
| M6 decode | CUDA graphs per decode batch size; skinny BF16 GEMV for every unregistered dense shape; expert kernels stage their weight slices with `cp.async`. |

**Measured** (`ninfer_bench`, INT8 KV, C = 1, greedy, warm cache, recipe A): prefill 167-180 tok/s
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

**Next, in order of expected gain:** speculative decoding (MTP drafter, verify rounds with GDN
replay and PLE/QSA state rollback; amortizes the 5.5 ms of dense reads over ~2.8 tokens), fewer
misses (admission doorkeeper, saved profile, lookahead prefetch, CPU-served misses), recipe B's
8-bit dense weights (a measured quality gate, §16.3), and tuned dense kernels.

---

## 20. Documentation and authority changes

When the implementation lands:

- `AGENTS.md` and `README.md`: add `Qwen4ExpForCausalLM` to the product architectures, and the
  host-RAM and NVMe requirements.
- `docs/maintainer/qwen4-exp-model.md`: new model reference (mathematics, config, state, MTP, PLE).
- `docs/maintainer/expert-offload.md`: frame pool, MTP pool, residency state machine, epochs,
  policy, transfer agent, mailboxes, CPU engine, and the canonical routed-expert arithmetic. §8-§12
  and §16.2 of this file move there.
- [Tensor formats](tensor-formats.md): `nvfp4_mul` and `fp8_e4m3fn_block128_f32`, with their decode
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
  and B, the primary configuration, the options of §18, and `ninfer-calibrate`.
- `docs/performance/qwen3.8-flash-next.md` and a model card.
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
| NVMe without polled queues, or high latency (QLC, DRAM-less) | PLE wait exposed in plain decode (§12.4) | L1 sizing, telemetry, documented drive requirement |
| Calibration overfits the bundled corpus or a noisy measurement | Settings slower on real workloads | Held-out check in M9; noise-aware acceptance; defaults on ties |
| 96 GB is tight with a desktop session, or with a `bf16` KV host tier at 256K | OOM or swap | Startup plan check; configurable L1, checkpoint tier and KV host tier; the primary `int8` configuration needs 3.6 GB of host tier at 262K |
| Engine contract changes for chained rounds (H6) | Large change to transactions | Built only if its gate passes; otherwise the synchronous boundary stays |

Open questions answered by measurement, not assumption:

- k′ and the depth of prediction;
- LFRU count-halving period, the promotion budget and the shadow-tuning switch threshold;
- the frame pool's KV-loan reclaim frequency at C = 8;
- the MTP pool size against acceptance;
- the KV host tier threshold for each profile, `int8` first;
- the A4 mismatch rate against NVIDIA's runtimes;
- H1-H13 adoption (§17 gates).

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
| R2 | QSA supports every NInfer KV profile | §8.10: per-profile tile loaders, profile-independent BF16 index keys, the VQ2/K4V2 window rule, and the host tier for every profile (§9.6) |
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
| E9 | NInfer's ModelOpt importer stores `fl(1 / weight_scale_2)` as the NVFP4 divisor, which is not NVIDIA's weight unless the scale is a power of two | `nvfp4_mul` keeps the multiplier (§6.1) |
| E10 | "Speculative output equals plain output" and "incremental decode equals prefill" overclaimed. NInfer's dense kernels change tiles with width, and its established criterion is a reported disagreement rate. | Bit-exactness is claimed only for expert outputs across placements and batch shapes; the rest uses the existing criterion (§16.4) |
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
