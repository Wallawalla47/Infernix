# Qwen3.8-Flash-Next on one RTX 5090: design

**Status: design proposal, not implemented.** This is a temporary planning document for the
Qwen3.8-Flash-Next product change. When the implementation lands, its stable content moves into
the active authorities named in [§20](#20-documentation-and-authority-changes) and this file is
removed.

**Revision 2026-10-03.** This version was revised after an expert review for implementability and
peak performance on the target hardware. [Appendix A](#appendix-a-review-findings-and-changes)
lists every finding and what changed.

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

The goal is the fastest single-GPU implementation of this model on this hardware. The reference
point is Strata running Unsloth UD-Q4_K_XL, a 4-bit model comparable to NVFP4, at **72-80 tok/s
decode** on the target machine.

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
19. [Implementation plan](#19-implementation-plan)
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
- QSA with every existing KV profile that its consumers support;
- Vision, which can be selected at startup.

The supported machine is an RTX 5090 with:

- at least 96 GB of host RAM;
- PCIe 5.0 x16, or 4.0 x16 with reduced performance;
- an NVMe SSD for the n-gram table.

Concurrency stays within the existing product range of 1-8.

### 1.2 Baselines

All comparisons run on the same machine, with the same prompts and the same output caps:

| Baseline | Weights | What it shows |
|---|---|---|
| **Strata, user-measured** | UD-Q4_K_XL (Q4_K/Q5_K gate/up, Q5_1/Q8_0 down) | **72-80 tok/s decode on RTX 5090 + 96 GB**. This is the primary target. |
| ninfer-ext `fc305acd` | NVFP4 experts, Q8 dense | 98.5 tok/s serving at C=1, 185 at C=8. Measured on a 247 GiB host, so it is not the target memory class. |
| FreeToken offload/hybrid | NVFP4 experts, BF16 dense | 61-67 tok/s on a 5090 with 128 GB DDR5 (community report, verified). ninfer-ext's README cites 77-79 tok/s on its own machine; no FreeToken source for that number was found. |
| SGLang, **all experts resident**, RTX PRO 6000 (sm_120, same 1.79 TB/s) | NVFP4 experts, BF16 dense | TPOT 11.4 ms (≈ 87 tok/s) plain; ≈ 148 tok/s with MTP. This is a generic-framework ceiling with no offload at all. |
| Strata IQ2_XS | ~2.35 bpw | 165-179 tok/s on a 5090. This uses 2-bit experts, so it is **not** a like-for-like quality baseline. It shows the dense-path ceiling when 71% of the experts fit in VRAM. |

### 1.3 Targets

The targets assume Text with thinking off, a 4K prompt and 256 output tokens, at C=1 unless stated
otherwise. They are acceptance gates, not claims. [§4](#4-performance-model) derives where they
come from: the Must column beats the Strata baseline by 1.5-2×, and the Stretch column is the
model projection at a 0.9 hit rate, which is more conservative than the replay traces.

| Metric | Must | Stretch |
|---|---:|---:|
| Decode, no speculation, C=1, 4K context | ≥ 120 tok/s (1.5× Strata Q4) | ≥ 220 |
| Decode, best speculation, C=1, mixed chat corpus | ≥ 160 tok/s (2× Strata Q4) | ≥ 330 |
| Decode at a 128K context, relative to 4K | ≥ 85% | ≥ 92% |
| Aggregate decode, C=8 | ≥ 300 tok/s | ≥ 600 |
| Prefill, 32K fresh prompt | ≥ 6,000 tok/s | ≥ 9,000 |
| Quality: mean KL vs the BF16 reference, on a fixed corpus | ≤ KL of UD-Q4_K_XL | — |

The acceptance report covers:

- the full distribution: median and range of at least three runs;
- the worst case;
- expert hit and miss composition;
- the device every number was measured on.

It compares against both Strata UD-Q4_K_XL and ninfer-ext on the target machine. If a target is
missed, the report says so; the target is not changed after the measurement.

### 1.4 Non-goals

- Copying Strata or ninfer-ext code. Both were studied for mechanisms and measurements only.
- Experts below NVFP4 precision by default. [§17](#17-high-reward-options) lists this as an
  explicit opt-in (H10).
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

Byte budget, worked out from the shapes above:

| Item | Size |
|---|---|
| One NVFP4 expert (gate/up and down: 4,915,200 weights; E2M1 plus E4M3 per 16) | **2,764,800 B** (2.64 MiB) |
| Routed expert weights touched per token (48 × 10) | **1.33 GB** |
| All routed experts: 48 × 512, or 49 × 512 with MTP | 67.9 GB / 69.4 GB |
| Dense Text weights | ~4.31 B params. 8.6 GB in BF16, 4.4 GB in Q8, **2.8 GB** with the recipe in §6. |
| Token embedding | 1.27 GB in BF16. Only one row per token is read. |
| N-gram table | 51.2 GB of FP8 E4M3 codes plus one BF16 scalar scale |
| KV per context token (12 layers: FP8 K/V plus the indexer key) | ~15.5 KB: 0.5 GB at 32K, 2.0 GB at 128K, 4.1 GB at 256K |
| GDN recurrent state per sequence | 113 MB |

What follows from these facts:

- **Experts cannot all be resident.** About 26 GiB of VRAM is left for experts, which holds about
  10,000 slots: **~40%** of the 25,088 experts, MTP included.
- **The whole expert set fits in 96 GB of RAM, but not much else does.** About 20 GB remains for
  everything else. The 51 GB n-gram table therefore has to stay on NVMe, as required.
- **Every miss costs host DRAM bandwidth**, whether it crosses PCIe (a DMA read of DRAM) or is
  computed by the CPU (a CPU read of DRAM). At ~41% capacity a plain-decode token is
  **latency-bound** on its few misses. The DRAM ceiling binds for wide verify windows at low hit
  rates and for prefill ([§4.1](#41-bound)).

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
| Recipe | Experts imported exact as NVFP4. **Dense layers re-quantized to Q8** (attention, GDN, HC, shared expert, PLE projections). `lm_head` Q6, embedding Q8, router BF16. MTP experts converted from block FP8 to NVFP4. The n-gram table is FP8 with the global scale **copied onto every row**. | 5.3 GB of device weights, about 3.9 GB of dense reads per token. The per-row scale plane doubles the PLE I/O ranges. |
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
| **SGLang** | `qwen4_exp.py` with NEXTN MTP (3 steps, 4 draft tokens). PLE either pinned in host RAM with a Triton gather from the host pointer, or a mapped file that **does not work on a discrete 5090**: it needs pageable host-table access. Lookup on a side stream. Fused n-gram hash. Expert offload only through the generic KTransformers integration. | All-resident on an RTX PRO 6000: 87 tok/s plain, 148 with MTP (accept 2.9-3.3 of 4). KTransformers + SGLang on 2× 5070 Ti: 36-37 tok/s. |
| **vLLM** | Native `qwen4_exp`: Triton QSA indexer, CuTe-DSL HC kernels. The QSA `qkv_proj` explicitly bypasses FP4. The Engram/PLE table is pinned on the CPU and read through UVA, prefetched on a side stream during the previous layer. Disk/mmap PRs are open. Expert offload is only generic (`--cpu-offload-gb`, layer-group prefetch); "CPU offload: TODO" in its tracking issue. | No single-5090 number found. DGX Spark NVFP4: 34-44 tok/s. |
| **llama.cpp** | `qwen4exp` merged (HC, QSA with pooled keys, one PLE layer). Experts placed by `-ncmoe`, with the CPU computing the experts that are not on the GPU. | ~43-52 tok/s on a 5090 with 128 GB (community reports, not verified) |

Literature mechanisms relevant to batch-1 decode with a hot GPU cache:

| Work | Mechanism | Used here |
|---|---|---|
| Eliseev & Mazur (2312.17238), FATE (2502.12224) | Next layer's gate applied to the current hidden state to predict the next layer's experts; FATE reports cosine similarity > 0.83 and 97% prefetch accuracy | §8.7. Recall must be measured: the HC streams may weaken it on this model. |
| SP-MoE (2510.10302) | Prefetch the experts implied by draft tokens before verification | §11.4 |
| Fiddler (2402.07033), KTransformers (SOSP'25), FreeToken q\* | CPU computes misses in place, or a balanced CPU/PCIe split | §10, §8.6 |
| HOBBIT (2411.01433) | Low-precision copies for less important missed experts | Opt-in only (§17, H10), because it changes the represented weights |
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
3. **The dense path is a large share of the token at 4 bits.** The checkpoint keeps every dense
   tensor in BF16: 8.6 GB per token, and 22.5% of FreeToken's token time. Dense weights should be
   NVFP4, hyper-connections FP8, and the per-layer kernel count should fall from ~32-40 to ~10 or
   fewer (all three engines).
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

---

## 4. Performance model

### 4.1 Bound

For one decode round with T token columns (T = 1 for plain decode, K+1 for a speculative verify),
the round time is approximately:

```text
t_round ≈ t_gpu(T) + Σ_layers t_exposed(l) + t_draft + t_boundary

t_gpu(T)    = Σ_kernels max(bytes_k / BW_eff, compute_k) + n_boundaries · t_b      (§8.3 budget)
t_exposed(l) = max(0, t_service(misses of l) − t_overlap(l))      t_service from §8.6
misses per layer ≈ U_T · (1 − h),   U_T = distinct experts per layer for T columns
```

Two ceilings bound the misses:

```text
host DRAM:  Σ_l (CPU-served + DMA'd + promoted bytes)  ≤  BW_dram · t_round     (BW_dram ≈ 70-90 GB/s)
PCIe:       Σ_l (DMA'd + promoted bytes)               ≤  BW_pcie · t_round     (≈ 50 GB/s)
```

At this design's capacity (~41% of experts in VRAM), a T = 1 token is **latency-bound on misses**,
not bandwidth-bound:

- ~10 of 48 layers have a miss;
- each costs one handshake plus one expert's service time;
- the DRAM ceilings bind only for wide verify windows at low hit rates, and for prefill.

There are four levers, each attacked by the design:

| Lever | Mechanism |
|---|---|
| **h**: hit rate | Frame pool of ~10,200 experts (§6, §9.1); LFRU (§9.3); prefetch converting misses into staged hits (§8.7); KV loans and the KV host tier keep the pool large (§9.2, §9.6) |
| **t_exposed**: the miss stall | CPU in place, with no handshake for hit-only layers; x pre-published; warming (H2); arbiter (§8.6); stall-time L2 warming of the next layer (§8.3) |
| **t_gpu**: dense path and kernel latency | NVFP4/FP8 dense (2.8 GB); 7-9 launches per layer with PDL and pre-dependency weight prefetch; per-kernel budget (§8.3); megakernel option (H1) |
| **Tokens per round** | MTP with a miss-aware draft length (§11); trees (H9) |

### 4.2 Projection

**Measured input: misses per token.** This model's published routing traces (FreeToken community
repo: 3 × 383 single-client decode tokens, English technical writing) were replayed through
candidate policies (§9.3). At ~10,200 frames (41.5% of experts):

| | Value | Note |
|---|---|---|
| LRU hit rate | 0.969 (14.9 misses per token) | |
| LFRU hit rate | 0.974 (12.5 misses per token) | |
| Layers with at least one miss | ~10 of 48 per token | |
| Layers with two or more misses | ~5% | |
| Reuse distance | median 3 tokens | 36% of accesses repeat on the next token; 92% within 64 tokens |
| Verify windows (T = 1 / 2 / 4) | **same misses per accepted token** (LRU 14.9 / 14.8 / 14.8) | 27 distinct experts per layer for T = 4, i.e. 2.7× one token |

The trace sample is small and narrow, so long, diverse sessions will miss more. Milestone M1
replays broad traces (§19). The projection therefore spans a range of hit rates.

**Two GPU assumptions:**

- **Target:** the §8.3 budget, with t_gpu(1) = 3.6 ms and t_gpu(4) = 5.1 ms. The draft takes
  0.3 ms (3 steps), and 2.8 tokens are accepted per round (§11.1).
- **Conservative:** t_gpu(1) = 4.0 ms and t_gpu(4) = 6.0 ms, with 0.8 ms drafting and 2.6 accepted
  per round.

**Common assumptions:**

- A cold CPU-served expert takes 42 µs; the handshake 6 µs.
- Hit work overlaps 15 µs per layer at T = 1 and 45 µs at T = 4.
- Prefetch covers up to 1 (T = 1) or 1.5 (T = 4) experts per layer at recall 0.6.
- Misses per layer are Poisson; rejected drafts add 15%.
- Warming (H2), split service (H3) and the megakernel (H1) are **not** included.

| Hit rate h | Misses per token | Plain decode, target (conservative) | MTP K = 3, target (conservative) |
|---|---:|---:|---:|
| 0.974 (LFRU on the replay traces) | 12.5 | ~270 (~240) tok/s | ~510 (~375) tok/s |
| 0.95 | 24 | ~255 (~235) tok/s | ~475 (~355) tok/s |
| 0.90 | 48 | ~230 (~215) tok/s | ~360 (~295) tok/s |
| 0.85 | 72 | ~200 (~190) tok/s | ~265 (~235) tok/s |
| 0.80 | 96 | ~175 (~170) tok/s | ~205 (~190) tok/s |

How to read the table:

- **At h ≥ 0.9 the GPU term dominates.** The decisive work is the dense path, the kernel count and
  per-layer latency (§8.3). That is a different regime from ninfer-ext and FreeToken, whose
  serialized PCIe misses dominate and whose caches hold only 16-26% of experts.
- **Speculation pays at every row.** Its advantage shrinks as h falls. A 4-column verify at
  h ≤ 0.85 puts ~5 misses per layer on the CPU (~230 per round), so the arbiter's DMA overflow, H2
  and H3 matter there.
- **References.** SGLang with all experts resident on a 96 GB sm_120 card (BF16 dense, generic
  kernels) measures 87 tok/s plain and 148 with MTP. Every row above exceeds that, mainly because
  this design reads 4.5 GB per token instead of ~10 GB.
- **Sensitivity.** Each +0.5 ms of t_gpu costs ~12% of plain decode at h = 0.95. The §8.3 budget
  is the largest risk in the projection, and M6 measures it first.

---

## 5. Design overview

```text
 ┌──────────────────────────────── RTX 5090, 32 GB ─────────────────────────────────┐
 │ Dense Text (NVFP4/FP8, 2.8 GB) · MTP dense · heads · QSA KV + index keys · state │
 │ ┌──────────────── frame pool: ~10,200 × 2,764,800 B frames (~26 GiB) ───────────┐ │
 │ │ cached experts │ staging (≤128) │ KV loans │ prefill arena │ vision (on demand)│ │
 │ └────────────────────────────────────────────────────────────────────────────────┘ │
 │ residency[49×512] · land_seq[frame] · L0 PLE row cache · decode workspace          │
 └──────▲──────────────────────▲─────────────────────────▲───────────────────▲───────┘
        │ copy engine (DMA)    │ zero-copy reads/writes  │ zero-copy          │ WriteValue
        │                      │ (mailboxes, CPU y)      │ (PLE rows)         │ (residency)
 ┌──────┴──────────────────────┴─────────────────────────┴───────────┐  ┌────┴──────────┐
 │ 96 GB host                                                         │  │ NVMe          │
 │  pinned expert banks 25,088 × 2,764,800 B (69.4 GB, huge pages)    │  │ n-gram table  │
 │  embedding rows · L1 PLE row cache · mailboxes · (KV host tier)    │  │ 52.4 GB FP8,  │
 │  transfer agent ─ cache policy, all DMA, residency updates         │  │ 4 KiB blocks  │
 │  CPU expert engine ─ spin workers (narrow canonical route)         │  └────▲──────────┘
 │  NVMe agent ─ io_uring SQPOLL/IOPOLL, L1 cache  ────────────────────────────┘
 └────────────────────────────────────────────────────────────────────┘
```

A decode round at C = 1 without speculation. The host loop is not on the critical path, and the
host learns about a layer only when it has a CPU-served miss:

```text
round r: embed(host row) ─ [PLE kernel waits for rows: L0 hit or NVMe agent] before layer 1 ─┐
 per layer l:                                                                                 │
   K1 HC_attn ─ K2/K3 (GDN) or K2q/K2b/K3a/K3b (QSA) ─ K4 out_proj+inject ─ K5 HC_mlp (x→host)
   K6 router(l, l+1): top-10 ─ residency ─ jobs ─ [miss_req → CPU] ─ [prefetch_req → agent]
   K7 experts (clusters) ─ combine (waits CPU only if needed; warms L2 with layer l+1 meanwhile)
 head ─ sample ─ n-gram ids(t+1) ─ L0 probe ─ NVMe requests ─ token egress ─ round_done
```

---

## 6. Artifact and conversion

### 6.1 Recipe `qwen3_8_flash_next_nvfp4`

The source is [nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4).
The recipe imports NVIDIA's quantized tensors exactly and chooses the representation of the
remaining tensors on measured quality per byte. Per-class choices are qualified by the KL ablation
in [§16.3](#163-recipe-qualification) before they become the official recipe.

| Tensor class | Source form | Artifact form | Reason |
|---|---|---|---|
| Routed experts (main) | ModelOpt NVFP4: E2M1 codes, E4M3 per 16, FP32 `weight_scale_2`, FP32 `input_scale` | **Exact import.** Codes and scales bit-identical. Per-matrix FP32 divisor. Per-layer A4 input divisor = min over the layer's experts' `input_scale`, as in the existing NInfer NVFP4 contract. | NVIDIA's calibration, with no requantization loss |
| MTP routed experts | Block FP8 (128×128 tiles) | `nvfp4_mse` per expert, one divisor per matrix | MTP experts become eligible for the same cache and kernels. Draft quality only changes acceptance, never output. |
| GDN and attention projections, shared expert, PLE key/value projections, MTP dense | BF16. The checkpoint is `MIXED_PRECISION`: only routed experts (NVFP4), MTP experts (block FP8) and the n-gram table (FP8) are quantized. Verified in the FreeToken, vLLM and SGLang loaders; M0 re-checks `hf_quant_config.json`. | **NVFP4** (`nvfp4_mse`), with FP8 row as the ablation fallback per class. The QSA QKV projection starts as FP8, because vLLM deliberately keeps it out of FP4; NVFP4 is used there only if the KL ablation allows. | 2.8 GB instead of 4.4 GB (Q8) or 8.6 GB (BF16) of dense reads per token. The 27B `nvfp4` artifact shows this route already in production. |
| QSA indexer projection, GDN `a`/`b` gates | BF16 | BF16 | Small and selection-sensitive |
| HC low-rank down/up and inject | BF16 | **FP8 E4M3 row-scaled**, with NVFP4 as an ablation candidate | The largest dense read per token in both studied engines. Rank-320 matrices are sensitive, so FP8 is the default until the KL ablation says otherwise. |
| Router (`[512,2560]`) and shared-expert gate | BF16 | BF16 | 63M params, and top-k sensitive |
| `lm_head` | BF16 | **NVFP4**, with FP8 row as a fallback by ablation | 0.36 GB instead of 1.27 GB. The optimized proposal head reuses it. |
| Token embedding | BF16 | BF16, **host-resident** (§6.3) | Saves 1.27 GB of VRAM, about 460 expert slots |
| Norms, `A_log`, `dt_bias`, conv weights | BF16 / FP32 | Direct | — |
| N-gram table (128 shards) | FP8 E4M3 plus one BF16 scalar scale | **Exact FP8 import**, NVMe layout of §12.2. One scalar scale. **No per-row scale plane.** | One I/O per row instead of two |
| Vision | BF16 | Existing NInfer vision formats | Selected at startup, and its device weights are borrowed from the frame pool (§9.2) |

### 6.2 Expert storage layout: one layout for two consumers

The pinned host bank is read by the GPU (DMA into a frame, then the narrow GEMV or the wide GEMM)
and by the CPU (in-place narrow GEMV). Only one copy fits in 96 GB, and NInfer forbids runtime
repacking. The artifact therefore stores each expert as one contiguous record of 2,764,800 B
(= 675 × 4 KiB) in the layout **`nvfp4_rg16_kmajor`** defined in §16.2:

- gate/up rows interleaved (gate_i, up_i);
- 16-row groups;
- one 144-byte unit (codes + scales) per row group and 16-element block.

The FP32 matrix divisors and the per-layer A4 input divisor live in a small separate table.

| Consumer | Access |
|---|---|
| DMA | One contiguous copy per expert into a frame |
| GPU narrow route | Units staged into shared memory with 16-byte `cp.async`; lane-per-row canonical arithmetic (§16.2) |
| GPU wide route (prefill, n > 8) | Units converted into MMA operand order in shared memory. No tensor-core layout leaks into the record. |
| CPU | One AVX-512 vector per row group; a worker's row band is one contiguous stream on huge pages |

The layout id and its byte-exact definition are recorded in [storage layouts](storage-layouts.md).

### 6.3 Residency classes in the v3 container

The artifact needs three residency classes. Each is an artifact/materialization concern, not model
mathematics:

| Class | Backing | Users |
|---|---|---|
| `device` (existing) | Uploaded at load | Dense weights |
| `host_pinned` | Read with `O_DIRECT` into a 2 MiB-page, `cudaHostRegister`ed host arena. Never staged through the page cache. | Routed expert banks, embedding |
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
| Pinned huge-page host arena; `O_DIRECT` + io_uring reader; mapped mailbox and ring primitives; spin-worker pool with barriers | Core (model-independent physical and transfer primitives) | `src/core/host_pinned_arena.*`, `direct_io_ring.*`, `mapped_mailbox.*`, `spin_worker_pool.*` |
| Frame pool, residency table, staging, loans, epochs; transfer agent (policy, DMA, residency writes); NVMe agent; L0/L1 row caches; CPU-engine lifetime | **Program** (mutable state, placement, agents; all allocated at startup) | `src/models/qwen4_exp/program/` |
| Cache policy algorithms (LFRU, shadow replay) | Program, behind a policy interface conformance-tested against `tools/expert_cache_replay` | `src/models/qwen4_exp/program/expert_cache/` |
| Router + top-k + residency classification + prediction; narrow-route expert GEMV (GPU and CPU); wide-route grouped GEMM; combine | **Op family** `offloaded_sparse_moe` (closed contract: output = MoE(x) independent of residency; residency, agents and mailboxes are execution resources) | `include/ninfer/ops/offloaded_sparse_moe.h`, `src/ops/offloaded_sparse_moe/{route,gpu,cpu,wide}/` |
| HC mixer and HC inject | Op `hyper_connection` | `include/ninfer/ops/hyper_connection.h` |
| QSA prep, index-key pooling, block select, sparse attention | Ops `qsa_prep`, `qsa_select`; extended `softmax_attention` consumer | `include/ninfer/ops/qsa.h` |
| PLE gather, projections, gate and dilated conv (stateful) | Op `ple_ngram_injection` | `include/ninfer/ops/ple.h` |
| N-gram row ids | Op `ngram_row_ids` (exact integer oracle) | same |
| Canonical narrow-route arithmetic (shared CPU/GPU helpers, `exp_c`) | Op common code | `src/ops/common/canonical_math.h` |
| Draft-length policy with miss cost | Program speculative backend | `src/models/qwen4_exp/program/speculative/` |
| Calibration | Runtime (profile schema), Core (probes), product (`ninfer-calibrate`) | §14.7 |

Residency is an **execution resource** for `offloaded_sparse_moe`, like a stream or a workspace.
By the [Op rules](op-development.md#2-op-admission-and-semantic-boundary), it may choose the
implementation but must not change the result. That is the basis for the placement-invariance
requirement (§16.2).

If option H1 (megakernel) is adopted, Ops expose device-callable **tile functions** next to their
host entries. Each tile function is qualified through the same contract. The Model owns the
instruction sequence, exactly as it owns the call order today. No Op becomes model-labelled.

---

## 8. Decode pipeline

This section specifies a decode round at the level an implementer needs: agents, the host-device
protocol, every kernel with its budget, and the router, expert and arbitration algorithms.

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
| `x_host[L][T][2560]` | host, mapped | GPU (HC_mlp epilogue) → CPU workers | FFN input of each layer, **written every round** before routing is known. That is 5 KiB per layer at T=1 and keeps x off the miss critical path. |
| `miss_req[L]` | host, mapped | GPU (router kernel) → CPU workers | `{round, n_miss, expert_id[], column_mask[]}` + `seq` |
| `miss_out[L][slot][T][2560]` FP32 | host, mapped (or device memory via BAR, option H7) | CPU → GPU (combine) | Expert outputs, then `done_seq` |
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
  (`x_host` by K5) are fenced with `fence.sc.sys` by each writing thread before that kernel ends.
  The later publisher's release then covers them.
- **Host → device.** x86 store order is total. The CPU writes the payload, then `done_seq` with a
  release store. The GPU reads `done_seq` with `ld.acquire.sys` and only then reads the payload.
- **Polling discipline.** Exactly **one thread per CTA** (lane 0 of warp 0) polls a host-resident
  sequence word, with `__nanosleep(32-128)` backoff after the first 2 µs. It then releases the CTA
  through a shared-memory flag and `__syncthreads()`. No warp-wide polling over PCIe is allowed.
- **Bounded spins.** Every device spin has an iteration bound worth about 2 s. On expiry the thread
  writes `agent_error`, skips the computation (outputs are undefined), and the engine worker turns
  that round into an Engine-wide failure ([engine architecture §7.4](engine-architecture.md)). Host
  agents have a 2 s watchdog on `round_done` progress.

### 8.3 Per-layer kernel sequence and time budget

Decode at T=1. Bytes are what each kernel must read from HBM with every expert resident. The
budget is at 1.5 TB/s, which is 90% of this card's measured 1,674.5 GB/s sustained read
([linear benchmark §9](linear-benchmark.md)), plus ~1 µs per boundary.

**GDN layer (36 of 48):**

| # | Kernel (one launch each) | Reads | Budget |
|---|---|---|---|
| K1 | **HC_attn mixer.** Grouped offset RMSNorm of R (stream sums of squares come from the previous K7 epilogue), W_down [324×10240] FP8 by split-K, gates, W_up [10240×320] FP8, collapse to x [2560]. Persistent cooperative kernel with 2 device-wide barriers. | 6.6 MB | 6 µs |
| K2 | **GDN input projection.** [16,384×2560] NVFP4 rows (q,k,v,z) + [96×2560] BF16 (a,b). Row-local epilogue only: causal conv (state update) + SiLU on q,k,v; β = sigmoid(b), g = −exp(A_log)·softplus(a + dt_bias). Any row tiling, so the grid fills all SMs. In layer 1 only, the PLE kernel (§12.3) runs before K1. | 24.1 MB | 17 µs |
| K3 | **Gated delta recurrence.** One CTA per (value head, 32-row slice of the 128×128 FP32 state): 192 CTAs. The rows of a head's state update independently given k. The prologue applies per-head L2 norm to q and k (and the 1/√128 q scale); this is recomputed per CTA, at negligible cost. Outputs o (unnormalized). | 6.3 MB | 5 µs |
| K4 | **out_proj** [2560×6144] NVFP4. Prologue: per-head sigmoid-gated RMSNorm of o with z (each CTA recomputes the 48 head norms; 6,144 values). Epilogue: HC inject `R += inject_attn ⊗ y` + partial stream sums of squares for K5. | 8.8 MB | 7 µs |
| K5 | **HC_mlp mixer** (as K1). Epilogue writes x to `x_host[l]` (§8.2). | 6.6 MB | 6 µs |
| K6 | **Router** (§8.4): router l and l+1 [1024×2560] BF16, top-10, residency, job list, mailbox, prefetch requests, route log | 5.2 MB | 5 µs |
| K7 | **Experts** (§8.5): narrow-route hits + shared expert; fused combine + HC inject `R += inject_mlp ⊗ y_moe` + stream sums of squares for the next K1. Waits on the CPU only if this layer has CPU misses. | 30.4 MB | 21 µs |
| | **Layer total** | **88 MB** | **≈ 67 µs** |

**QSA layer (12 of 48).** K2/K3 become four kernels:

- K2q: QKVG [13,312×2560], FP8 by recipe default (NVFP4 if the §16.3 ablation allows), plus
  the index projection [640×2560] BF16, projection only;
- K2b: QSA prep:
  - Q/K offset-RMSNorm and partial MRoPE;
  - K/V append (FP8 row-scaled);
  - raw index-key append;
  - pooled index-key completion when a 4-token block closes (§13);
- K3a: block scores over the pooled index keys + exact top-512 radix select;
- K3b: sparse attention over ≤ 2,051 selected tokens (GQA 24/2, D=256) + sigmoid output gate.

That is ≈ 98 MB and ≈ 74 µs per layer at short context, or ≈ 82 MB and 66 µs with an NVFP4
QKVG. A GDN layer has 7 launches and a QSA layer 9: about 366 launches per token, including
embedding, PLE, head and sampling.

**Per token:**

| Item | Budget |
|---|---|
| 48 layers (36 × 67 µs + 12 × 74 µs) | ≈ 3.30 ms |
| PLE (18 MB) | 0.02 ms |
| `lm_head` 0.36 GB NVFP4 + sampling | 0.25 ms |
| Embedding (host row) | 0.005 ms |
| **t_gpu (T=1, all hits)** | **≈ 3.6 ms (≈ 280 tok/s ceiling)** |

The total is 4.7 GB per token. Its pure-bandwidth floor is 2.8 ms at the sustained read rate.

**Rules that every decode kernel follows.** This is the "HBM never idle" baseline.

1. **PDL with pre-dependency weight prefetch.**
   - Kernels launch as programmatic dependents ([op development](op-development.md#programmatic-dependent-launch)).
   - Before `griddepcontrol.wait`, each CTA issues L2 prefetches (`cp.async.bulk.prefetch.L2`) of
     the **static weights it will read**. The next kernel's weight stream therefore starts while
     the previous kernel drains.
   - The router (K6) prefetches only its own weights. K7's expert weights depend on routing, so they
     are prefetched by K6's last block as soon as routing is known, before K7 starts.
2. **Stall-time warming.** A kernel that waits on the host (K7 for CPU misses, the PLE kernel for
   rows) issues L2 prefetches of the **next layer's K1-K4 weights** while it waits (≈ 46 MB,
   bounded to half of L2). The stall is converted into progress on the next layer.
3. **No float atomics.** Split-K and cross-CTA reductions write partials to workspace. The last CTA
   (atomic counter) reduces them in fixed order. Results are run-to-run deterministic.
4. **Streaming cache hints.** Weight loads use `L2::evict_first`. Activations, state and the
   residency table use the default or `evict_last`.
5. **Grid sizing.** Every kernel's grid is a multiple of 170 SMs × resident CTAs per SM. Device-wide
   barriers and per-expert counters (§8.5) then never deadlock, because all CTAs are co-resident.
   Wave quantization is avoided.

The budget is a planning tool, not a promise. M6 publishes measured per-kernel times against it,
and any kernel more than 30% over budget gets an ncu investigation before the next milestone.

### 8.4 Router kernel (K6)

**Inputs:**

- x_l [T, 2560] BF16;
- router W_l and the next layer's router W_{l+1} (BF16, 512 × 2560 each);
- the residency table;
- the cost-model constants (expected CPU service time; expected landing time per `LOADING` frame,
  published by the transfer agent).

**Phase 1** (all CTAs, ~128): warp-per-row GEMV of the 1,024 rows. Logits go to workspace.

**Phase 2** (last CTA, selected by an atomic arrival counter):

1. **Routing.** For each column t: top-10 of the 512 logits, with the lower id winning exact ties;
   route weights = softmax over the 10 selected logits (`norm_topk_prob`).
2. **Union.** The distinct experts over the T columns, with per-expert column masks and n_e.
3. **Classification** of each distinct expert from its residency entry: GPU job (`READY`), gated GPU
   job (`LOADING` and landing predicted sooner), CPU miss, or on-demand DMA (§8.6). Experts with
   n_e > 8 are always GPU jobs. If such an expert is not resident, it becomes an on-demand DMA with
   a gated job.
4. **Outputs.**
   - The job list for K7: frame pointer, mask, weights, gate ticket.
   - `miss_req[l]`, written with release **only if** there is a CPU miss.
   - The route log.
5. **Prediction.** Top-k′ of W_{l+1}·x_l per column, unioned. Non-resident, non-loading experts go
   to `prefetch_req[l]`.
6. **Expert weight prefetch.** L2 prefetch of the GPU jobs' frames, so K7 starts on warm lines.

Phase 2 is serial but small: about 2 µs at T=1 and about 4 µs at T=8.

### 8.5 Expert kernel (K7)

**Narrow route (n_e ≤ 8), cluster-per-expert:**

1. Each GPU job gets one thread-block cluster of S CTAs (S = 8 or 16, chosen at development time).
   The shared expert is one more cluster.
2. **Phase A.** CTA j computes gate/up row groups [80j/S, 80(j+1)/S) with the canonical arithmetic
   (§16.2), and writes its slice of h (FP32) to its shared memory.
3. **Cluster barrier.** Each CTA copies the full h (640 × n_e FP32) from its peers through
   distributed shared memory.
4. **Phase B.** CTA j computes down row groups [160j/S, 160(j+1)/S). y_e (FP32) goes to workspace.

**Fallback if the toolchain's cluster limits do not fit:** per-expert ready counters in global
memory. Down tiles spin on their expert's counter, and co-residency is guaranteed by rule 5 of §8.3.

**Wide route (n_e > 8):** the same grouped block-scaled tensor-core GEMM as prefill (§13). It uses
A4 where the Use allows `AllowA4`, and BF16-dequantized A16 otherwise. Wide-route experts never go
to the CPU, so their arithmetic needs no CPU twin.

**Combine.** This is the last-arriving cluster, selected by an atomic counter:

1. Wait on `miss_out[l].done_seq` if the layer had CPU misses (rule 2 of §8.3 applies while
   waiting).
2. Read the CPU's FP32 outputs zero-copy.
3. Form y_moe = Σ_rank w_i · y_{e_i} in fixed rank order (FP32), add sigmoid(shared_gate · x) ·
   y_shared, and round to BF16 at the model's semantic boundary.
4. Apply `R += inject_mlp ⊗ y_moe`.
5. Produce the next layer's stream sums of squares.

### 8.6 Miss arbitration

For each layer, the router's Phase 2 assigns every non-resident narrow-route expert to the
cheapest predicted service. The inputs are the cost-model coefficients the calibration fitted
(§14), refreshed online by the transfer agent:

| Service | Predicted completion | When chosen |
|---|---|---|
| CPU, warmed (H2) | t_handshake + n_cpu · t_warm | Expert is in the CPU's warmed set |
| CPU, cold | t_handshake + Σ bytes / BW_cpu(n_cpu) | Default |
| Wait for in-flight DMA | landing time published by the agent | `LOADING` and earlier than the CPU option |
| On-demand DMA + GPU | t_req + bytes / BW_pcie, contended | Only when the CPU queue for this layer exceeds the DMA completion time (several cold misses in one layer) |
| Split CPU + DMA (H3) | max of the two halves | Option H3 |

The objective is the earliest time at which **all** of the layer's experts are available,
because the combine waits for the last one. Ties go to the CPU, which needs no copy.
Every decision and its outcome (predicted vs actual completion) is logged into per-layer
histograms. The online model corrects its coefficients from those, and M5 reports the prediction
error.

### 8.7 Lookahead prefetch

Between the router of layer l and the router of layer l+1 lies one full layer of GPU work
(≈ 65-70 µs, §8.3). During that window, host DRAM and PCIe would otherwise be idle.

1. **Prediction.** K6 of layer l evaluates layer l+1's router on x_l (§8.4). Its top-k′ per column
   is unioned, with k′ = 12 by default (M1 fixes it). The non-resident, non-loading experts are
   published to `prefetch_req[l]` in order of predicted weight.
2. **Transfer.** The transfer agent (§9.4) issues one contiguous DMA per predicted expert into a
   staging frame. It stops when the next DMA could not land before layer l+1's K6, using the
   calibrated DMA completion model. At ~50 GB/s that is about one expert per layer.
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

The frame count is a startup decision (§15.1). Frames are 4 KiB-aligned slices of one 2 MiB-page
allocation, so a frame spans 2-3 GPU pages. The TLB reach of 2 MiB pages covers the whole pool.

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

Result: memory that a 256K-capable configuration reserves for KV (4.1 GB, about 1,550 frames) keeps
serving experts until the context actually grows into it.

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

| Policy (continuous session) | 16.3% (4,008) | 25.8% (6,347) | 32.6% (8,000) | **41.5% (10,200)** | 48.8% (12,000) |
|---|---:|---:|---:|---:|---:|
| LRU (global) | 104.6 | 49.9 | 30.2 | **14.9** | 8.5 |
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

1. **Victim ranking: LFRU**, one global pool over the Text and MTP experts.
   - The score is `f / (now − last + 1)`, with time in decode rounds.
   - `f` is a global per-expert count kept for all 25,088 ids, including evicted ones (100 KB, kept by the
     transfer agent on the host).
   - For long sessions, `f` is halved every 4,096 rounds to bound staleness. This value is untested
     and is fixed in M1.
   - No per-layer quotas, because hard partitioning lost 3-34%.
   - Experts of the current and predicted groups are protected.
2. **Admission: promote on miss, within a promotion budget.**
   - Every CPU-served miss is a promotion candidate. The transfer agent (§9.4) copies it into a frame
     only if its LFRU score beats the current victim's and the round's DRAM/PCIe token bucket has
     room. The device only reads residency entries.
   - At this design's capacity that is ~12.5 promotions per token (~7 GB/s at 200 tok/s), so the
     bucket rarely binds.
   - When it does bind (miss storms after a topic switch, or a long context that has shrunk the
     pool), promotion degrades into **lazy, batched promotion ranked by the same score**. This is
     Strata's mechanism, which replay shows makes ~7× fewer copies than LRU. Lazy LFRU (half-life
     128 rounds, margin 0.5, ≤ 32 per round) matched the best on-demand policy at 41.5% with half
     the promotions.
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

**Residency entry** (`residency[layer × 512 + expert]`, u32, device memory):

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
5. **Retirement.** Frames whose retire epoch ≤ `round_done` return to the free list.
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

This design adopts the same idea in NInfer's paged-KV terms:

- **Index-key plane: always fully device-resident.** At 262K it is ~0.1 GB in FP8.
- **K/V pages:**
  - The authoritative copy lives in a pinned host arena (~4 GB at 262K, FP8). §15.2 has room for it.
  - The device holds a page cache of hot 4-token blocks: CLOCK replacement, residency resolved in
    the graph.
  - Missed blocks are fetched zero-copy inside the attention kernel. A block is 4 tokens ×
    1,040 B per layer, a few KB, so latency- rather than bandwidth-bound.
- **Writes** go to both copies at append time.
- **Enabling.** The tier is used only when the configured context needs more than a threshold
  (64K by default, fixed in M8 by measurement). Below it, KV is fully device-resident.
- **Payoff.** At 262K, ~3.5 GB, about 1,300 frames, return to the expert pool. In the replay that
  is the difference between ~8,900 and ~10,200 frames: roughly 20 vs 12.5 misses per token with
  LFRU.

---

## 10. CPU expert engine

### 10.1 Role

At this design's capacity, a layer has 0 CPU-served misses ~80% of the time and 1 miss most of the
rest (§4.2). The CPU engine is therefore optimized for the **latency of one or two experts**, not
for throughput:

- 2.765 MB per expert;
- critical path = handshake + compute + result visibility.

A cold miss is DRAM-bound at ~40-45 µs on dual-channel DDR5. A warmed miss (§10.4) is compute-bound
at ~12-18 µs (to be measured in M3/M5). PCIe DMA is used for prefetch, promotions and the arbiter's
overflow case (§8.6).

### 10.2 Kernels

The **narrow route (A16)** implements the canonical arithmetic of §16.2 on the `nvfp4_rg16_kmajor`
layout. One vector holds 16 rows (AVX-512) or 8 rows (AVX2). A tile loop runs over k with one
broadcast and n FMAs per column.

| ISA | Decode | Expected per-core rate (to be measured) |
|---|---|---|
| AVX-512F/BW (Zen 4/5, Xeon) | 8 B load → `vpmovzxbd`+shift → `vpermps` 16-entry table | ~40-50 G weights/s/core from L2 |
| AVX2+FMA (Arrow Lake, Zen 3) | 4 B per half → 3-bit magnitude `vpermps` + sign XOR | ~20-25 G weights/s/core |

The **A8 route** (opt-in only) quantizes the activation to int8 per 16-block, maps E2M1×2 to int8
and uses `vpdpbusd` (AVX-512 VNNI / AVX-VNNI). It is not placement-invariant (§16.2), so it is
offered only under an explicit `AllowA8` Use and a user flag.

Compilation rules: `-ffp-contract=off -fno-fast-math`, and explicit FMA intrinsics only.

### 10.3 Execution

- **Static row-band ownership.** Worker w owns the same row band of **every** expert: gate/up row
  groups [80w/N_w, 80(w+1)/N_w) and down row groups [160w/N_w, 160(w+1)/N_w). The data a worker
  warms (§10.4) is therefore the data it later computes. A layer's misses are processed expert by
  expert, in rank order:
  1. phase A (gate/up bands → SwiGLU on the owning worker);
  2. a spin barrier (one cache line per worker, sense-reversing);
  3. h exchange through shared memory;
  4. phase B (down bands).
- **Handshake.** Workers spin on `miss_req[l].seq` with `_mm_pause` (Intel) / `pause` (AMD). They
  read x from `x_host[l]`, which is already written by K5 (§8.2). Each writes its output rows to
  `miss_out[l]`. The last worker to finish (atomic counter) writes `done_seq` with a release store.
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

### 10.4 Predicted-miss warming (high-reward option H2)

When the router of layer l publishes its prediction for layer l+1 (§8.4), the transfer agent
splits the predicted non-resident experts in two:

- those its DMA budget can land before layer l+1's router, which are prefetched to VRAM;
- the rest, which become a **warm list** for the CPU engine.

While the CPU engine is otherwise idle, each worker loads its own row bands of the warm-list experts
into its private L2:

- 2.765 MB / N_w ≈ 200 KB per worker per expert at N_w = 14;
- capped at half of the L2 (Zen 4/5 1 MiB, Lion Cove 3 MiB per P-core).

If layer l+1 then misses on a warmed expert, the computation streams from L2 instead of DRAM.

Warming uses DRAM bandwidth only in the GPU-only phase of each layer, when DRAM is otherwise idle.
Calibration measures warmed versus cold service time, and M5 decides adoption (§17).

---

## 11. Speculative decoding

### 11.1 Expected acceptance

On this model, SGLang's all-resident MTP (3 steps, 4 draft tokens, greedy) accepts **2.9-3.3
tokens per round** on GSM8K, random and ShareGPT prompts. Strata reports 2.4-3.2 tokens per pass.
The projection (§4.2) uses 2.8.

### 11.2 A drafter that never waits on the host

The MTP drafter (one QSA block, its MoE, and the optimized proposal head) runs entirely on the GPU
with **residency-bounded routing**:

- **Draft routing.** Router logits are computed over all 512 MTP experts. The top-10 are taken
  among experts whose entry is `READY`. The weights are the softmax over those 10 logits. This is a
  draft-only approximation: it changes acceptance, never the verified output.
- **MTP experts.** They share the frame pool and LFRU (§9.3). Their recency updates are weighted
  by measured acceptance, so the drafter's routing tracks the true routing.
- **Draft head.** The existing optimized proposal head (reduced vocabulary) is used, not a full
  `lm_head` read per draft step.
- **Budget.** ≈ 85 µs per draft step: ~45 MB of dense and proposal-head weights, plus resident
  experts.
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
- The rows are consumed **once**, by the PLE kernel that runs before layer 1's K1. In decode, the
  hiding window is layer 0: ≈ 65-70 µs on the §8.3 budget, longer when layer 0 has a CPU miss. In
  the replay traces, layer 0 is among the most-missed layers.
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
| L2: NVMe | — | io_uring with registered files and fixed buffers, `O_DIRECT`, SQPOLL on the agent's core, and IOPOLL when the driver has polled queues (`nvme.poll_queues > 0`). Queue depth up to 256. |

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
   - gathers the 16 rows (L0 or landing slot), dequantizes FP8 with the scalar scale, and computes
     key/value projections, gate, grouped norms, dilated conv (state update) and the injection
     into R;
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
  polled QD1 read latency exceeds ~60 µs exposes part of every L2 miss in plain decode.
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
  - n_e ≤ 8: narrow route, shared with the CPU, placement-invariant.
  - n_e > 8: wide route, a block-scaled tensor-core grouped GEMM. It uses **A4**
    (`mma … kind::mxf4nvf4.block_scale`, with the per-layer input divisor) when the artifact's Use
    allows `AllowA4`. The official recipe enables that for routed experts, from NVIDIA's calibrated
    `input_scale`, subject to the §16.3 KL check. Otherwise it uses A16 (BF16-dequantized weights),
    which is ~4-5× slower in compute.
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

  GPU compute is ~12 GFLOP per token, ~0.2 s per 16K tokens at the ~900-985 TFLOP/s this card
  measures for A4 GEMM ([linear benchmark §9](linear-benchmark.md)), plus attention and GDN chunk
  kernels. A16 expert GEMMs would add ~0.8 s per 16K chunk. With A4, large chunks are bound by PCIe
  and attention. The Must target of 6,000 tok/s at 32K leaves margin.
- **QSA prefill.** Pooled, normalized, RoPE'd index keys are materialized **once per 4-token
  block** when the block completes. They are stored in an FP8 index-key plane beside KV, so select
  scans one 128-wide key per block. This is the same insight as ninfer-ext's `697ff0c7`, built into
  the KV layout rather than recomputed per call.
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
| SM zero-copy read bandwidth and latency from mapped host memory | Kernel reading the expert arena | KV-tier block fetch (§9.6) |
| DRAM read bandwidth vs threads (per P-core, E-core, CCD) | Streaming reads over the pinned arena | Worker set; per-CCD placement (Zen CCDs have separate bandwidth limits) |
| **DRAM contention curve:** CPU read bandwidth while H2D DMA runs at 0 / 25 / 50 / 100% | Concurrent runs | Arbiter and token-bucket model (§8.6, §8.7). Host DRAM is the shared bottleneck. |
| CPU NVFP4 expert GEMV: each compiled variant (AVX-512 / AVX2; A16, and A8 only if the recipe allows it and the user opted in), T ∈ {1, 2, 4, 8}, worker counts, SMT on/off, prefetch distance | Real expert records | Kernel variant, worker set, service-rate table t_cpu(n_experts, T) |
| GPU↔CPU mailbox round trip | Mapped-memory publish → host spin → completion → device acquire | Handshake term in the cost model |
| NVMe random 4 KiB reads at QD 1-256 on the n-gram file, for `O_DIRECT` + io_uring with and without SQPOLL/IOPOLL | Random row ids | I/O mode, queue depth, expected PLE latency; whether layer 0 hides it (§12.3) |
| Huge pages | DMA and CPU bandwidth over THP 2 MiB, hugetlbfs 2 MiB and 1 GiB | Page backing of the expert arena |
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
  - placement-invariant A16 routes (§16.2);
  - different workers, page backing, I/O, prefetch and cache settings.

  The A8 CPU route changes results, so it is offered only when the artifact's Use permits A8 **and**
  the user explicitly enables it.
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

### 15.1 VRAM: 32,607 MiB card, C=1, MTP, 32K context, FP8 KV

| Item | GiB |
|---|---:|
| CUDA context, libraries, graphs | 0.9 |
| Dense Text (NVFP4 / FP8 HC / BF16 router) | 2.6 |
| MTP dense, proposal head | 0.3 |
| KV + index keys, 32K (main + MTP) | 0.5 |
| GDN, PLE, conv state | 0.12 |
| Workspace (decode), mailboxes, residency/score tables | 0.6 |
| L0 PLE row cache | 0.06 |
| Slack | 0.4 |
| **Frame pool** | **≈ 26.3** |
| Frames (2.64 MiB) | ≈ 10,200: **40.7%** of 25,088 experts, of which ≤ 128 are staging |

For comparison, ninfer-ext has 6,347 slots (26%) and Strata at NVFP4 sizes would have about 8,700
slots (35%).

At a 256K context the KV reservation is 4.1 GB. Through loans (§9.2) it costs expert frames **only
as the context actually fills**. Above the long-context threshold, the QSA KV host tier (§9.6) keeps
most of it off the device altogether.

### 15.2 Host RAM: 96 GB (≈ 93 GiB visible)

| Item | GB |
|---|---:|
| Expert banks, 25,088 × 2.7648 MB, pinned 2 MiB pages | 69.4 |
| Embedding (BF16), pinned | 1.27 |
| L1 PLE row cache | 2.1 (configurable 0.5-4) |
| Mailboxes, landing buffers, io_uring buffers | 0.2 |
| Host checkpoint tier (prefix cache), capped | 2.0 (configurable) |
| QSA KV host tier (§9.6), only above the long-context threshold | 0-4.1 (at 262K, FP8) |
| Process: tokenizer, server, frontend | ~1.5 |
| **Total NInfer** | **~76.5**, or ~80.6 at 262K with the KV host tier |
| OS, desktop, page cache headroom | ~19, or ~15 at 262K |

Additional rules:

- **Loading.** Experts are read with `O_DIRECT` straight into the pinned arena (~10 s from a 7 GB/s
  drive). Loading never doubles memory through the page cache.
- **Startup check.** Startup verifies `RLIMIT_MEMLOCK` and `MemAvailable` against this plan and
  fails with a precise message. It does not swap.
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
| `offloaded_sparse_moe` (narrow, GPU and CPU) | FP64, NVFP4 decoded independently from codes, block scales and divisors; BF16 inputs | Every residency mix: all-frame, all-staging, all-CPU, random mixes; n = 1-8 columns; **plus bit-exact equality across routes** (§16.2) |
| `offloaded_sparse_moe` (wide) | Same FP64 oracle; A4 route against the represented A4 activations | n = 9-8192 |
| `hyper_connection` | FP64 | FP8 weights decoded exactly; grouped norm per stream; split-K partial order fixed |
| `qsa_prep`, `qsa_select` | FP64 block scores | Near-tie allowance as in the existing selector contracts; dense equivalence while n ≤ 2051; pooled key completion at block boundaries |
| `ple_ngram_injection` | FP64 | FP8 rows decoded with the scalar scale; conv state transition; EOS restarts |
| `ngram_row_ids` | **Exact integer** | Signed 64-bit products, primes 20,000,003-20,000,171, offsets; EOS and sequence start |

The fork's FP64 Python reference may be consulted as a cross-check of the published mathematics.
NInfer's oracle is written from the upstream definitions, consistent with [§1.4](#14-non-goals).

### 16.2 Placement invariance: the canonical expert arithmetic

**Requirement.** For every routed expert computed on a **narrow route** (the expert has n ≤ 8
token columns in this forward), its FP32 output vector is **bit-identical** whichever of these
computed it:

- the GPU from a cache frame;
- the GPU from a staging frame;
- the CPU from the host record.

Experts with n > 8 columns always use the GPU **wide route** (tensor cores, §8.5). The CPU never
serves them. The route is decided by n, which does not depend on placement, so outputs remain
placement-invariant.

**Canonical arithmetic** for one matrix row r: K inputs, K/16 blocks, E2M1 codes c, E4M3 block
scales s, FP32 matrix divisor d, FP32 input column x (BF16 inputs widened exactly; for the down
projection, x is the FP32 SwiGLU output h):

```text
acc = 0.0f
for b in 0 .. K/16-1:                      # increasing block order
    p = 0.0f
    for j in 0 .. 15:                      # increasing element order, sequential chain
        p = fmaf(e2m1(c[r,16b+j]), x[16b+j], p)
    acc = fmaf(p, e4m3(s[r,b]), acc)
y[r] = acc / d                             # IEEE division, round to nearest even
```

The SwiGLU between the two projections is:

```text
h_i = (g_i / (1 + exp_c(-g_i))) * u_i
```

- The division and multiplication are IEEE FP32, with the parenthesization exactly as written.
- `exp_c` is one shared implementation, in `ops/common/canonical_math.h`, with no FMA contraction
  outside explicit `fmaf`:
  - Cody-Waite range reduction;
  - a fixed degree-6 polynomial evaluated by explicit `fmaf`;
  - exponent reconstruction by integer add.
- Neither `__expf` nor the libm `expf` is used.
- The expert output y_e (2560 FP32) is the semantic boundary returned to the combine. The combine
  always runs on the GPU, in fixed rank order (§8.5).

Both compilers must be prevented from contracting or reassociating:

- **GPU:** `--fmad=false` for the canonical translation unit, or explicit `__fmaf_rn` /
  `__fmul_rn` / `__fadd_rn` / `__fdiv_rn` intrinsics everywhere in it;
- **CPU:** `-ffp-contract=off -fno-fast-math`, and explicit `_mm512_fmadd_ps` / `_mm256_fmadd_ps`.

**Layout that makes the canonical order natural on both processors:** `nvfp4_rg16_kmajor`.

- Rows are grouped by 16.
- For each row group g and block b there is one 144-byte unit:
  - 128 bytes of codes, k-major. For each of the 16 k positions, 8 bytes hold the 16 rows' nibbles;
    row i is nibble i, with the low nibble first.
  - 16 bytes of E4M3 scales, row i at byte i.
- Units are ordered g-major, then b. For the gate/up matrix, rows alternate gate and up
  (row 2i = gate_i, row 2i+1 = up_i), so a row group holds 8 complete SwiGLU pairs.

Sizes:

| Matrix | Units per row group | Bytes per row group | Row groups | Bytes |
|---|---|---|---|---|
| gate/up | 160 | 23,040 | 80 | 1,843,200 |
| down | 40 | 5,760 | 160 | 921,600 |
| **Record total** | | | | **2,764,800 B = 675 × 4 KiB** |

How each processor uses the layout:

- **CPU.** One AVX-512 vector holds the 16 rows of a group. Each k step:
  1. load 8 bytes and expand nibbles to 16 byte lanes;
  2. `vpermps` with a 16-entry E2M1 table;
  3. one `vfmadd` per column with broadcast x[k].

  AVX2 uses two 8-row halves and a 3-bit-magnitude table plus a sign XOR.
- **GPU narrow route.**
  - 144-byte units are staged into shared memory with 16-byte `cp.async` (or TMA bulk copies) in a
    multi-stage pipeline.
  - Lane i of a half-warp owns row i and runs the canonical chain from shared memory; the other
    half-warp owns the next row group.
  - Each lane interleaves the chains of its n columns, plus a second row group when n is small, to
    hide FMA latency.

**Cost gate (M3).** The narrow-route GPU kernel is measured against a free-order variant of the
same kernel at T = 1, 2, 4, 8 on the real expert shapes. If the canonical kernel loses more than 5%
bandwidth at T = 1, or more than 10% at T = 4, the requirement is downgraded:

- GPU and CPU are then qualified separately against the FP64 oracle;
- greedy outputs may then differ between cache states at near-ties;
- §9.5 then also needs a determinism mode (`--deterministic`) that blocks on promotions, as Strata
  does.

The downgrade and its measured cost are recorded here.

**What placement invariance buys:**

- Greedy output does not depend on cache contents, prefetch timing or the CPU/GPU split. Strata
  measures only 95-97.7% top-1 agreement between cache states.
- No round ever waits for a promotion.
- Every residency route is tested by exact comparison: all-GPU vs all-CPU vs random mixes
  (§16.4).

### 16.3 Recipe qualification

The reference is the BF16 checkpoint run through the engine's CausalScoring path at
all-BF16-equivalent precision, or published BF16 logits where available. On a fixed corpus (code,
prose, chat, CJK, long-context NIAH), measure:

- mean and p99 KL against the reference;
- top-1 agreement;
- perplexity (existing [perplexity](../perplexity.md) tooling).

Run the comparison for:

- the full recipe;
- each dense class ablated back to BF16 or FP8 (HC, `lm_head`, GDN, attention);
- Strata UD-Q4_K_XL, scored on the same corpus by the same metric where its outputs can be obtained.

A class stays NVFP4 only if its KL contribution is below the threshold fixed **before** the
measurement. Proposed threshold: total KL ≤ that of UD-Q4_K_XL, and each class ≤ 20% of the total.

### 16.4 Engine-level and system tests

- **Exactness.**
  - Incremental decode equals one-shot prefill on the narrow routes.
  - Speculative output equals plain output (greedy).
- **Placement invariance end to end.** Greedy outputs are byte-identical across:
  - forced all-CPU experts (`--expert-frames 0`);
  - a small cache;
  - the full cache;
  - prefetch on and off;
  - H2 warming on and off;
  - arbiter decisions forced to each service.
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

---

## 17. High-reward options

These options are **not** in the baseline. Each has an expected gain, a cost, and a measurable gate
that decides whether it is built. The baseline (§8-§13) must work without any of them. Gains are
model estimates at C=1 unless marked otherwise.

| ID | Option | Expected gain | Cost / risk | Gate (measured in) |
|---|---|---|---|---|
| **H1** | **Persistent decode megakernel.** One persistent kernel per round executes an instruction stream of the round's Op tiles. Ops provide device-callable tile functions in addition to their host entries. An on-GPU scheduler resolves dependencies with counters. Weights for instruction i+1 stream into shared memory while instruction i computes, and host waits (CPU misses, PLE rows) are just dependencies. | Removes most of the ~370 kernel boundaries per token. Hazy Research measured large gains on a 1B model at batch 1 (see §22); here ~0.2-0.4 ms per token (5-12%). | Very high: a new execution framework. Op tiles must remain individually qualifiable. Graph capture is replaced by instruction-list capture. | M6 nsys attribution: kernel boundary and tail gaps > 10% of the token after the baseline's PDL and prefetch measures. |
| **H2** | **CPU cache warming for predicted misses** (§10.4). Workers pre-load the predicted CPU-served experts' row bands into their own L2 caches during the GPU's dense phase, so the miss computes from cache. | Warmed miss ~12-18 µs instead of ~42 µs. At ~10 missing layers per token, −0.2 to −0.3 ms (6-9%). | Low. Wasted DRAM bandwidth when mispredicted, but DRAM is idle then. | M5: warmed-miss service time and prediction recall. Keep it if the exposed stall falls by > 25%. |
| **H3** | **Intra-expert CPU + DMA split.** For a layer with a single miss, the CPU computes the gate/up half while the copy engine lands the down half into a staging frame. The GPU computes down from the CPU's h, which crosses PCIe as 2.5 KB. | One cold miss ~32 µs instead of ~42 µs; more at low hit rates | Medium. Two handshakes. The h exchange is FP32, so invariance holds. | M5: only if cold (unwarmed) misses remain > 20% of the exposed stall |
| **H4** | **Two-layer-ahead prediction** (router of l+2 applied to x_l) | Doubles the prefetch window, so more misses become staged hits | Low. Lower recall, wasted DMA. | M1 recall@k′ and M5 useful-byte ratio > 0.5 |
| **H5** | **Multi-token protection predictor** (§9.3, policy item 6) | Up to 2.5-10 points of hit rate on top of LFRU, with perfect prediction | Research | M1 on the hidden-state trace set |
| **H6** | **Chained rounds (asynchronous scheduling).** Enqueue round r+1 before r completes. The device carries the sampled or accepted tokens. The host commits round r while r+1 runs. This needs a device-side accepted-prefix fold and an abort path for rounds launched past a stop. | Hides the host's per-round work (ingress build, commit, output). Expected 2-6% at 3.5 ms tokens. | High. Engine transaction changes; see [engine architecture §6](engine-architecture.md). | M6 nsys: GPU idle between rounds > 3% of the token |
| **H7** | **BAR-mapped mailboxes** (gdrcopy-style). CPU results and done flags are written directly into device memory, so the GPU polls local memory instead of reading host memory over PCIe. | ~2-4 µs less per missing layer | Medium. Kernel module, Resizable BAR, write-combining stores. | M3 mailbox round trip: keep it if it is > 30% faster |
| **H8** | **Prompt-seeded cache.** Prefill already routes every prompt token. At the end of prefill, the prompt's per-layer expert counts seed the LFRU state, and the top uncached experts are promoted while the copy engine is otherwise idle, before the first decode token (MoE-Infinity's request-level activation idea). | Fewer cold misses in the first hundreds of decode tokens after a topic switch | Low | M1 replay with prefill routing: hit rate of the first 256 decode tokens with and without seeding |
| **H9** | **Draft trees** (DFlash2-style lattice over MTP top-2 at the first one or two positions) | +10-25% accepted tokens per round in the literature | More columns, so more misses on rejected branches | M7 cost model: it is chosen per round only when expected tokens/s rise |
| **H10** | **Reduced-precision miss copies** (HOBBIT-style: a 2-bit host copy of cold experts, served when a miss is on the critical path) | Halves the bytes of cold misses | **Changes model output.** RAM for the extra copies. | Never default. A labeled product option, after the KL check. |
| **H11** | **Exclusive residency** (an expert lives in VRAM *or* RAM) for 64 GB hosts | Supports smaller machines | Write-back and eviction complexity | Outside the current product scope |

Considered and rejected:

- **Lossless entropy coding of NVFP4 codes in VRAM** to fit more experts. E2M1 codes of
  MSE-calibrated weights carry roughly 3.5-3.8 bits of entropy, a ≤ 12% capacity gain. Decoding at
  1.5 TB/s would cost more GPU time than the extra hit rate saves.
- **GPU-side expert skipping or dynamic top-k.** It changes the model's mathematics.
- **Mixed SM copy kernels for prefetch.** They steal SMs and HBM bandwidth from the decode kernels.
  The copy engine is free.

---

## 18. Configuration surface

These are the startup options added for this architecture, with their defaults. The exact names
are final when `--help` is written. Unspecified values come from the calibration profile (§14),
then from model-based defaults. Every option that affects output is marked.

| Option | Default | Meaning |
|---|---|---|
| `--expert-frames auto\|N` | `auto` | Frame pool size; `auto` takes all device memory left after the plan of §15.1 |
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
| `--deterministic` | off | Only if the §16.2 cost gate downgrades invariance: blocks on promotions |
| `--cpu-a8` | off | **Changes output.** A8 CPU route (§10.2), only with an `AllowA8` Use |
| `--spec mtp`, `--draft-tokens`, `--fixed-draft`, `--ngram-draft-tokens`, `--draft-tree-nodes` | existing | Existing speculation options; draft length policy of §11.3 |

Command: `ninfer-calibrate ARTIFACT [--quick|--full] [--corpus DIR] [--out PATH]` (§14.2).

---

## 19. Implementation plan

Each milestone has deliverables and exit criteria measured on the target machine. Later milestones
may change because of earlier findings. Such changes, and the evidence for them, are recorded here
rather than silently absorbed.

| # | Milestone | Deliverables | Exit criteria |
|---|---|---|---|
| **M0** | Source and machine facts | NVIDIA NVFP4 `hf_quant_config.json` and tensor dtypes; target machine inventory (CPU ISA, DRAM bandwidth, PCIe, NVMe QD1/QD16 latency, memory clock under CUDA load) | Facts recorded; §4 and §8.3 assumptions confirmed or revised |
| **M1** | Routing traces, policy and predictors | Broad traces (code, chat, CJK, long context, tool use; C = 1-8; MTP with rejected drafts) from the public hidden-state trace set, Strata's `--dump-routing`, or M4's engine. Replay extended with prediction and DMA-budget modelling. | LFRU (or a better causal policy) confirmed on held-out traces, with its gap to LRU and Belady reported; recall@k′ and useful-byte ratio for depth 1 and 2; §4.2 projection updated with the real h |
| **M2** | Artifact | Recipe; `nvfp4_rg16_kmajor` expert records; n-gram volume with 4 KiB blocks; residency classes; `O_DIRECT` huge-page loader | Exact-import tests; load time; host RAM peak ≤ §15.2 plan |
| **M3** | Ops with oracles | `offloaded_sparse_moe` (route, narrow GPU, narrow CPU A16/A8, wide, combine), `hyper_connection`, QSA, PLE, `ngram_row_ids`; core mailbox, ring and spin-pool primitives with stress tests | All Op qualifications pass; **placement invariance** bit-exact across GPU frame, GPU staging and CPU for T = 1-8; §16.2 cost gate evaluated; CPU narrow kernel ≥ 80% of measured DRAM bandwidth cold (AVX-512), shortfall recorded for AVX2; mailbox round trip measured |
| **M4** | Functional model, CPU-only experts | Prefill, decode, MTP and Vision through the Engine with every routed expert CPU-served; CausalScoring | Model oracle agreement; recipe KL report (§16.3); first end-to-end numbers |
| **M5** | Expert cache | Frame pool, residency table and state machine, transfer agent, LFRU, loans, epochs, prefetch, arbiter, H2 warming experiment | Host policy decisions **identical** to `tools/expert_cache_replay` on recorded routing; live hit composition within 2 points of replay; no round ever blocks on a promotion; agent CPU < 15% of a core; H2 and prefetch adoption decided by measurement |
| **M6** | Decode performance | §8.3 kernels with PDL, pre-dependency prefetch and stall-time warming; cluster expert kernel; device-driven PLE; nsys/ncu attribution of one token | Per-kernel times vs the §8.3 budget published; plain decode ≥ 120 tok/s (Must); H1, H6 and H7 gates evaluated |
| **M7** | Speculation | Residency-bounded drafter, miss-aware draft length, n-gram copy; tree option H9 evaluated | Speculative decode ≥ 160 tok/s (Must); speculation never slower than plain beyond noise on any corpus category |
| **M8** | Prefill and long context | Layer-streamed prefill with host-issued DMA, CPU thin-expert assist, equal chunk planning, QSA pooled-key plane, KV host tier | Prefill Must target; short-prompt (512-2K) prefill reported; 128K decode ≥ 85% of 4K |
| **M9** | Calibration | §14: startup probe, `ninfer-calibrate --quick/--full`, profile and drift check, Stage 2 fit, Stage 3 A/B | On at least two different hosts, calibrated ≥ uncalibrated (or tie within noise) on the calibration **and** held-out corpora; every candidate reported; recalibration reproduces the choices |
| **M10** | Head-to-head | Campaign vs Strata UD-Q4_K_XL and ninfer-ext on the target machine; serving C = 1-8; 4K/32K/128K; quality report | §1.3 table filled with every result, favorable or not |

The milestones on the critical path to the decode targets are M2 → M3 → M5 → M6. M1 runs in
parallel and must finish before the policy is frozen in M5.

---

## 20. Documentation and authority changes

When the implementation lands:

- `AGENTS.md` and `README.md`: add `Qwen4ExpForCausalLM` to the product architectures, and the
  host-RAM and NVMe requirements.
- `docs/maintainer/qwen4-exp-model.md`: new model reference (mathematics, config, state, MTP, PLE).
- `docs/maintainer/expert-offload.md`: frame pool, residency state machine, epochs, policy,
  transfer agent, mailboxes, CPU engine, canonical arithmetic. §8-§12 and §16.2 of this file move
  there.
- [Storage layouts](storage-layouts.md): `nvfp4_rg16_kmajor`.
- [Engine architecture](engine-architecture.md): Program ownership of host agents and frames.
  Residency as an Op execution resource.
- [Artifact container](artifact-container.md), [storage layouts](storage-layouts.md): residency
  classes, `expert_record_nvfp4_*`, the n-gram volume.
- [Weight conversion](../weight-conversion.md), [CLI](../cli.md), [serving](../serving.md): recipe and
  options (`--expert-cache`, `--cpu-experts`, `--ngram-file`, `--ple-row-cache`, `--calibration`), and
  `ninfer-calibrate`.
- `docs/performance/qwen3.8-flash-next.md` and a model card.
- This file is deleted.

---

## 21. Risks and open questions

| Risk | Effect | Mitigation / decision point |
|---|---|---|
| Routing locality is weaker than in the replay traces (h < 0.85 at 41%; the shared expert lowers token-to-token reuse) | Plain decode toward the lower rows of §4.2; smaller speculation gains | M1 decides early. Levers: H2, H8, H3; HC and `lm_head` NVFP4 if quality allows (+~400 frames). Opt-in H10 only. |
| The §8.3 GPU budget is missed (small-GEMV efficiency, boundaries) | Every row of §4.2 drops ~12% per +0.5 ms | M6 per-kernel attribution against the budget; ncu on any kernel > 30% over; H1 gate |
| Thread-block cluster or DSMEM limits on sm_120 differ from the plan | Cluster-per-expert kernel unavailable at S = 16 | Fallback per-expert counters (§8.5); S = 8 |
| Host-issued DMA submission latency is too high for the one-layer prefetch window | Fewer staged hits | Batch submission; two-layer-ahead (H4); measured by calibration |
| Spinning kernels hold SMs while the CPU computes | No overlap for other work | Waiting is done by the last cluster only; meanwhile it L2-warms the next layer (§8.3 rule 2) |
| Next-layer prediction recall is low with hyper-connections | Less prefetch benefit | Prefetch self-disables per layer below useful-byte ratio 0.3 (§8.7) |
| The canonical arithmetic costs more than its gate allows | Lower narrow-route bandwidth | §16.2 downgrade rule, with `--deterministic`, disclosed |
| AVX2-only hosts are compute-bound on the narrow route at n > 1 | Slower CPU misses in verify windows | A8 opt-in; arbiter shifts load to DMA; reported per host class |
| DRAM bandwidth of the platform (DDR5 speed, 2 vs 4 DIMMs) | Scales cold-miss service and the DRAM ceiling | Calibration advice; 2 × 48 GB preferred |
| GPU memory clock below maximum under CUDA load (P-state) | All bandwidth-bound kernels slower | Calibration GPU sanity check against the 1,674.5 GB/s sustained-read reference; advice |
| NVIDIA's dense tensors differ from the assumed BF16 | Recipe table changes | M0 reads `hf_quant_config.json`; the recipe follows the source |
| NVMe without polled queues, or high latency (QLC, DRAM-less) | PLE wait exposed in plain decode (§12.4) | L1 sizing, telemetry, documented drive requirement |
| Calibration overfits the bundled corpus or a noisy measurement | Settings slower on real workloads | Held-out check in M9; noise-aware acceptance; defaults on ties |
| 96 GB is tight with a desktop session | OOM or swap | Startup plan check; configurable L1, checkpoint tier and KV host tier; server-only operation recommended |
| Engine contract changes for chained rounds (H6) | Large change to transactions | Built only if its gate passes; otherwise the synchronous boundary stays |

Open questions answered by measurement, not assumption:

- k′ and the depth of prediction;
- LFRU count-halving period, the promotion budget and the shadow-tuning switch threshold;
- the frame pool's KV-loan reclaim frequency at C = 8;
- H1-H9 adoption (§17 gates).

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
  environment. Facts here come from the two studied implementations' recorded checkpoint constants
  and are re-verified in M0.
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

---

## Appendix A. Review findings and changes

The previous version of this design was reviewed against three questions:

- Can it be implemented on the RTX 5090 and CUDA as they are?
- Does it leave performance on the table?
- Is it specific enough to build?

The findings are listed by severity. Each points to the section that now resolves it.

### A.1 Errors (the previous text could not be implemented as written)

| # | Finding | Resolution |
|---|---|---|
| E1 | Prefetch was "driven by a device-side scheduler kernel that issues DMA". A CUDA kernel cannot start a copy-engine DMA. | All DMA is issued by a host **transfer agent**. Landing is signalled by stream-ordered `cuStreamWriteValue32` (§8.1, §8.2, §9.4). |
| E2 | "One grouped kernel runs gate/up, SwiGLU and down with the intermediate in shared memory" cannot saturate HBM at T=1. Ten experts would occupy ten CTAs. | A cluster-per-expert kernel exchanges h through distributed shared memory, or falls back to per-expert ready counters (§8.5). |
| E3 | The router kernel as specified would run its 2.6 MB GEMV on a handful of SMs | Split GEMV over all SMs, then a last-CTA phase for top-k, classification and publication (§8.4) |
| E4 | The canonical arithmetic was underspecified: its order could not be both GPU- and CPU-natural; `expf`, FMA contraction and the h dtype were undefined | Exact definition, the `nvfp4_rg16_kmajor` layout that makes it natural on both, compiler rules, the n ≤ 8 route rule and a measured cost gate (§16.2) |
| E5 | Frames were described as "2 MiB-aligned VMM chunks" while being 2.64 MiB | Frames are 675 × 4 KiB slices of one allocation, and KV page sizes divide a frame (§9.1) |
| E6 | Device-side "recency updates" contradicted the requirement that policy bookkeeping stay off the critical path | The device only reads residency. LFRU, shadow replay and all decisions run in the transfer agent from the route log (§9.4). |
| E7 | The residency states had no `LOADING`, no generation and no reuse rule, which allowed use-after-reuse races | State machine with generations, landing tickets and epoch-based retirement (§9.4, §9.5) |
| E8 | Prefill's wide route defaulted to A16 tensor cores: about 4-5× slower in compute than the A4 the checkpoint was calibrated for | A4 wherever the Use allows `AllowA4`, which the official recipe sets for routed experts after the KL check (§13) |
| E9 | The recipe kept QSA QKV in FP8 while the budget assumed NVFP4 | Budget uses FP8 by default (§8.3) |

### A.2 Performance gaps (the design left speed on the table)

| # | Finding | Resolution |
|---|---|---|
| P1 | HBM sat idle during kernel boundaries and while waiting on the host | Baseline rules: PDL with pre-dependency weight prefetch; stall-time L2 warming of the next layer; evict-first weight streams (§8.3) |
| P2 | x was shipped to the host only once a miss was known, so its transfer sat on the miss path | x is written to host memory every layer (5 KiB) before routing (§8.2) |
| P3 | CPU misses always read DRAM cold | Static row-band ownership plus predicted-miss L2 warming (H2, §10.4) |
| P4 | No per-kernel budget: the 4.0 ms GPU assumption was unsupported | Per-kernel byte and time budget (≈ 3.6 ms at T=1), grounded in this repo's measured 1,674.5 GB/s sustained read (§8.3) |
| P5 | Recurrence and dense GEMV tiling were constrained by per-head norms in epilogues | Norms move to consumer prologues; recurrence splits state rows across 192 CTAs (§8.3) |
| P6 | Megakernel, chained rounds and BAR mailboxes were unscoped | Specified as options H1, H6 and H7, with measured gates (§17) |
| P7 | Prompt routing was not used to warm the cache | Option H8 (§17) |
| P8 | The miss decision ignored in-flight DMAs and per-layer completion time | Arbiter minimizes each layer's latest completion, including waiting for in-flight loads (§8.6) |

### A.3 Clarity gaps (not specified well enough to build)

| # | Finding | Resolution |
|---|---|---|
| C1 | No host-device protocol: memory regions, ordering, polling and failure | §8.2 |
| C2 | No configuration surface | §18 |
| C3 | Tests did not cover policy conformance, protocol races or forced placements | §16.4 |
| C4 | PLE hiding assumed a 80-150 µs layer 0; the faster GPU budget shrinks it to ~65-70 µs | Exposure model and an acceptance rule (§12.4) |
| C5 | Statements that every miss is DRAM-bound | Corrected: plain decode is latency-bound at this capacity (§2, §4.1) |
| C6 | Milestones lacked deliverables | §19 |

### A.4 Projections and targets

- Projections were recomputed with the per-kernel budget (§4.2).
- The Must targets are unchanged.
- The Stretch targets were raised to the model's projection at a 0.9 hit rate:
  - plain decode 220 tok/s;
  - speculative decode 330 tok/s;
  - C = 8 aggregate 600 tok/s.

  The 0.9 hit rate is below the replay traces' 0.974. The targets were set before any measurement
  (§1.3).
