# Qwen3.8-Flash-Next on one RTX 5090: design

**Status: design proposal, not implemented.** This is a temporary planning document for the
Qwen3.8-Flash-Next product change. When the implementation lands, its stable content moves into
the active authorities named in [§17](#17-documentation-and-authority-changes) and this file is
removed.

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
14. [Memory plans](#14-memory-plans)
15. [Numerics and qualification](#15-numerics-and-qualification)
16. [Implementation plan](#16-implementation-plan)
17. [Documentation and authority changes](#17-documentation-and-authority-changes)
18. [Risks and open questions](#18-risks-and-open-questions)
19. [Sources](#19-sources)

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
come from.

| Metric | Must | Stretch |
|---|---:|---:|
| Decode, no speculation, C=1, 4K context | ≥ 120 tok/s (1.5× Strata Q4) | ≥ 170 |
| Decode, best speculation, C=1, mixed chat corpus | ≥ 160 tok/s (2× Strata Q4) | ≥ 230 |
| Decode at a 128K context, relative to 4K | ≥ 85% | ≥ 92% |
| Aggregate decode, C=8 | ≥ 300 tok/s | ≥ 400 |
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
- Experts below NVFP4 precision by default. [§18](#18-risks-and-open-questions) lists this as a
  possible explicit opt-in.
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
- **Every miss is limited by host DRAM bandwidth.** This holds whether the miss crosses PCIe (a DMA
  read of DRAM) or is computed by the CPU (a CPU read of DRAM). It is the central constraint of
  this design ([§4](#4-performance-model)).

---

## 3. What existing implementations do

### 3.1 Strata

Strata is GGUF-based with custom CUDA kernels.

| Mechanism | Detail | Consequence for NVFP4 on 32 GB |
|---|---|---|
| Expert placement | All experts pinned in RAM; VRAM slots are filled from a ranked routing profile. Every 4 rounds an adaptive thread swaps experts per layer using decayed LFU (factor 0.7, gain ≥ 1.5×, ≤ 96 swaps). Since issue #463 the next round **waits** for those swaps. | Sound idea. At 2.77 MB per expert, 96 blocking swaps cost ~5 ms. |
| Miss service | Distinct misses are split between CPU in-place compute and an in-graph SM copy kernel that reads mapped host memory (`pcie_frac`, 0.55 by default for IQ packs) | CPU plus PCIe co-service is the right primitive, but the copy kernel occupies SMs and serializes after the hit kernels. |
| Hit/miss resolution | **One GPU→host→GPU handshake per layer, even at a 100% hit rate.** A doorbell kernel writes the activation and ids to mapped memory, the host builds the plan, and a single-thread GPU spin kernel waits. A device-side planner exists but is off by default. | 48 serialized round trips per window, on the critical path. |
| Kernels | llama.cpp-style dp4a MMVQ, FP32 gate/up intermediates through global memory, BF16 hyper-connections (~1.27 GB read per window). About 2,000 graph nodes per window. | On a 5090 a verify round takes ~14-15 ms against a ~2 ms bandwidth floor: it is overhead-bound even at 98% hits. |
| Speculation | MTP with up to 3 drafts, Q2_0 MTP experts in VRAM, a reduced draft head (106K ids), and suffix lookup. 1.6-1.8× on a 5070. Drafts run serially after verify, with a stream sync per step. | Gains depend on a high hit rate. Strata measured the missed-expert union at 1.75 / 2.4 / 3.05× one token's for windows of 2 / 3 / 4 tokens. |
| PLE | 28.8 GB IQ4_NL repack. `O_DIRECT` `pread` on 16 threads, a 1M-row host CLOCK cache, and a **synchronous** gather before each window. | NVMe latency on the critical path whenever the row cache misses. |
| KV | INT8 KV, with only the hot part in VRAM above 64K and the rest streamed from RAM | Not needed at our KV sizes ([§14](#14-memory-plans)). |

Strata's true VRAM hit rate is lower than it reports, because PCIe-served misses are excluded from
the reported ratio. On 12 GB cards, profile-only caching gave ~0.50 and adaptive caching ~0.72.

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
| Eliseev & Mazur (2312.17238), FATE (2502.12224) | Next layer's gate applied to the current hidden state to predict the next layer's experts; FATE reports cosine similarity > 0.83 and 97% prefetch accuracy | §8.4. Recall must be measured: the HC streams may weaken it on this model. |
| SP-MoE (2510.10302) | Prefetch the experts implied by draft tokens before verification | §11.3 |
| Fiddler (2402.07033), KTransformers (SOSP'25), FreeToken q\* | CPU computes misses in place, or a balanced CPU/PCIe split | §10, §8.3 |
| HOBBIT (2411.01433) | Low-precision copies for less important missed experts | Opt-in only (§18), because it changes the represented weights |
| AdapMoE (2408.10284) | Per-layer cache allocation by dynamic programming | Covered by the global policy plus per-layer floors (§9.4) |
| Local routing consistency (2505.16056) | Models with shared experts show weaker token-to-token expert reuse | Explains the modest LRU hit rates; Belady's gap is the headroom (§9.4) |
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
4. **Cache policy must start from recency, never block, and earn every refinement in trace replay.**
   On this model LRU is a strong baseline, pure LFU loses, and Belady shows ~50-60% fewer misses
   are possible (FreeToken traces). Promotion runs on the copy engine in the background, and a slot
   is reused only after its epoch has retired (Strata's #463 blocking).
5. **Speculation pays only when its verify cost includes the misses.** The drafter must never wait
   on the host, and the draft length has to come from a cost model that includes misses
   (ninfer-ext's MTP regression).
6. **NVMe reads must not wait for the host.** Row identification and the device row cache belong on
   the GPU, and the NVMe latency should be hidden behind layer 0 (both engines gather
   synchronously).

---

## 4. Performance model

### 4.1 Bound

For one decode round with T token columns (T=1 for plain decode, K+1 for a speculative verify), the
round time is approximately:

```text
t_round ≈ t_gpu + Σ_layers t_exposed_miss(l) + t_draft + t_boundary

t_gpu          = (B_dense + B_hit) / (η · 1.79 TB/s) + 48 · t_fixed
B_hit          = U_T · h · 48 · 2.765 MB      U_T: distinct experts per layer for T columns
misses per layer: m = U_T · (1 − h)
t_exposed_miss(l) = max(0, t_cpu(m_cpu) + t_handshake − t_hit_gemv(l))
m = m_prefetched + m_cpu,   m_prefetched ≤ window(l) · BW_pcie / 2.765 MB
```

The host-DRAM bound holds whatever the mix of CPU and PCIe service:

```text
miss bytes per round = Σ_l m_l · 2.765 MB  ≤  BW_dram · t_round        (BW_dram ≈ 70-90 GB/s on dual-channel DDR5)
```

Three levers decide the result. The design attacks all three:

| Lever | Mechanism |
|---|---|
| **h**: hit rate | More VRAM for experts (§6, §9), a frequency-plus-recency policy (§9), and lookahead prefetch counted as an effective hit (§8.4) |
| **t_exposed_miss**: the miss stall | CPU in place; a single handshake only for missing layers; overlap with the hit GEMVs (§8, §10) |
| **t_fixed**: per-layer fixed latency | ≤ 10 fused kernels per layer with PDL; no host involvement on hits (§8.6) |

### 4.2 Projection

The assumptions behind this projection are uncertain and are tested in milestone M1/M2
([§16](#16-implementation-plan)):

- dense 2.8 GB per token;
- η = 0.75;
- t_fixed = 25 µs per layer;
- CPU service at 65 GB/s while contended;
- PCIe DMA at 50 GB/s;
- a prefetch window of about one GPU layer;
- next-layer recall of 0.7;
- a handshake of 6 µs;
- U_4 = 3.05 × 10, Strata's measured union;
- 2.6 accepted tokens per MTP round, Strata's 5090 measurement.

| Hit rate h (VRAM, before prefetch) | Plain decode, projected | MTP K=3, projected | Miss DRAM traffic (plain) |
|---|---:|---:|---:|
| 0.70 | ~150 tok/s | ~150 tok/s | ~60 GB/s |
| 0.80 | ~200 tok/s | ~200 tok/s | ~55 GB/s |
| 0.90 | ~230 tok/s | ~290 tok/s | ~35 GB/s |
| 1.00 (all resident, impossible here) | ~250 tok/s | ~380 tok/s | 0 |

How to read the table:

- **Plain decode** can clear the Must target even at h=0.7, provided the miss stall is hidden as
  described. ninfer-ext reaches only 98 tok/s at h≈0.8 because each miss is a serialized
  ~77 µs PCIe stall.
- **Speculative decoding** only beats plain decode once h ≳ 0.85, because a 4-column verify roughly
  triples the distinct misses. This is why ninfer-ext's MTP lost to plain decode. Speculation
  therefore needs the miss-aware draft-length controller in [§11](#11-speculative-decoding), and
  its gains depend directly on the cache.
- The hit rate at 40% capacity is the main unknown. Measured points so far:
  - FreeToken LRU: 0.72-0.76 at 16% capacity;
  - ninfer-ext's LRU: 0.805 at 26% capacity, MTP K=3;
  - Strata adaptive caching: 0.72 at about 18% (IQ3_XXS on 12 GB).

  From that trend, LRU alone should reach roughly 0.85-0.88 at 40%. The §9 policy and prefetch,
  credited as effective hits, aim at 0.90+. Milestone M1 tests that expectation with real routing
  traces before the cache is built.
- **Calibration of the GPU term.** SGLang with every expert resident on an sm_120 card with the
  same 1.79 TB/s spends 11.4 ms per token. That run reads ~10 GB per token (BF16 dense) through
  generic kernels (~5.5 ms byte floor). This design reads ~4.1 GB per token at h=1 (~2.3 ms floor).
  The assumed t_gpu ≈ 4 ms requires ~58% of peak bandwidth plus 25 µs per layer. NInfer reaches
  comparable efficiency today on Qwen3.6-35B-A3B (338 tok/s, 40 MoE layers). M6 verifies this
  assumption first.

---

## 5. Design overview

```text
 ┌──────────────────────────── RTX 5090, 32 GB ───────────────────────────────┐
 │ Dense Text (NVFP4/FP8, 2.8 GB) · MTP dense · head · QSA KV · GDN/PLE state  │
 │ ┌───────────────────── frame pool (~26 GiB, 2.64 MiB frames) ─────────────┐ │
 │ │ hot experts (~9,400) │ prefetch staging (≤128) │ KV loans │ prefill arena│ │
 │ └─────────────────────────────────────────────────────────────────────────┘ │
 │ residency table [49×512] · device PLE row cache · miss / ngram mailboxes     │
 └─────────▲───────────────────────▲──────────────────────────▲───────────────┘
   copy engine (DMA)          mapped mailboxes            mapped row buffers
           │ prefetch / promote    │ act + ids ↓ / y ↑        │ rows ↑
 ┌─────────┴───────────────────────┴──────────────┐  ┌────────┴──────────────┐
 │ 96 GB host: 25,088 NVFP4 experts pinned on 2 MiB│  │ NVMe: n-gram table    │
 │ pages (69.4 GB) · embedding rows · row cache     │  │ 51.2 GB FP8, 4 KiB-   │
 │ CPU expert engine (spin workers, AVX-512/AVX2)   │◄─┤ packed, io_uring      │
 └──────────────────────────────────────────────────┘  └───────────────────────┘
```

Decode round at C=1 with no speculation. Every arrow is device-driven, and the host loop does not
sit on the critical path:

```text
sample(t) ─► ngram ids (GPU) ─► device row-cache probe ─► miss ids → NVMe ring ─┐
          │                                                                       │ (hidden by layer 0)
          ▼                                                                       ▼
 embed (host row, zero-copy) ─► layer 0 ─► [PLE waits for rows] ─► layer 1 ... layer 47 ─► head ─► sample(t+1)

 per layer l:  HC_attn ─► GDN/QSA ─► HC_mlp ─► router+top-k+residency (1 kernel)
                 │                                   ├─ hits ─► grouped NVFP4 GEMV (GPU)
                 │                                   ├─ CPU misses ─► mailbox ─► CPU engine ─► mapped y
                 │                                   └─ next-layer prediction ─► DMA prefetch → staging
                 └──────────── combine(hits, staged, CPU y, shared) ─► HC inject (fused)
```

---

## 6. Artifact and conversion

### 6.1 Recipe `qwen3_8_flash_next_nvfp4`

The source is [nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4).
The recipe imports NVIDIA's quantized tensors exactly and chooses the representation of the
remaining tensors on measured quality per byte. Per-class choices are qualified by the KL ablation
in [§15.3](#153-recipe-qualification) before they become the official recipe.

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

The pinned host bank is read by both the GPU (DMA into a VRAM frame, then GEMV/GEMM) and the CPU
(in-place AVX GEMV). Only one copy can exist in 96 GB, and NInfer forbids runtime repacking. The
artifact therefore stores every expert as a contiguous, frame-sized record that serves both
consumers:

```text
expert record (2,764,800 B, 4 KiB aligned, padded to the frame size):
  gate_up codes  [1280 rows × 2560]  row-interleaved gate/up pairs, 16-row × 512-col tiles
  gate_up scales                     E4M3, tile-major, adjacent to their code tile
  down codes     [2560 rows × 640]   same tiling
  down scales
  (FP32 matrix divisors and the A4 divisor live in a small separate device/host table)
```

- **One DMA per expert:** a single contiguous `cudaMemcpyAsync` from pinned host memory into a
  frame.
- **GPU decode:** a warp reads a 16-row tile with 16-byte vector loads, and the scales sit in the
  same cache lines as their codes.
- **GPU prefill:** tiles are loaded to shared memory and converted there into the block-scaled MMA
  operand order. No tensor-core layout leaks into the stored record.
- **CPU:** a core's row band is one contiguous memory stream, which suits hardware prefetchers and
  2 MiB pages.

The tile shape is fixed by a layout benchmark in M3 and recorded in
[storage layouts](storage-layouts.md) as a new layout id (`expert_record_nvfp4_t16x512_v1`).

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
| Mathematics, config, binding, PLE n-gram ids, HC, block order, MTP alignment | Model | `src/models/qwen4_exp/` (config, load, execution, frontend reuse of the Qwen3.5 tokenizer/template/vision) |
| Host-pinned expert bank, embedding bank, n-gram file region | Immutable Model data, through a new artifact residency class | `src/artifact/materializer.*` |
| 2 MiB pinned host arena; `O_DIRECT` + io_uring reader; mapped mailbox ring; spin-worker pool | Core primitives (raw transfer and host execution, model-independent) | `src/core/host_pinned_arena.*`, `src/core/direct_io_ring.*`, `src/core/mapped_mailbox.*`, `src/core/spin_worker_pool.*` |
| Expert cache, frame pool, residency table, epochs, promotion policy, prefetch staging, device and host PLE row caches, CPU miss-server thread lifetimes | **Program** (mutable state and placement; allocated at startup) | `src/models/qwen4_exp/program/` |
| Routed MoE with residency-aware execution (GPU hits, staged hits, CPU misses) | **Op** `offloaded_sparse_moe` | `include/ninfer/ops/offloaded_sparse_moe.h`, `src/ops/offloaded_sparse_moe/{gpu,cpu}/` |
| Router with residency resolution and next-layer prediction | Part of the same Op family | same |
| HC mixer (grouped norm, low-rank gates, collapse) and HC inject/combine | Op `hyper_connection` | `include/ninfer/ops/hyper_connection.h` |
| QSA indexer append, pooled-key maintenance, block select, sparse attention | Ops `qsa_index`, `qsa_select`; extended `softmax_attention` consumer | `include/ninfer/ops/qsa.h` |
| PLE gather-from-mapped-rows, key/value projection, gate, dilated conv | Op `ple_ngram_injection` (stateful: conv history) | `include/ninfer/ops/ple.h` |
| N-gram row id computation | Op `ngram_row_ids` (exact integer oracle) | same |
| Draft-length policy with miss cost | Program speculative backend | `src/models/qwen4_exp/program/speculative/` |

Residency is an **execution resource** for `offloaded_sparse_moe`, like a stream or a workspace.
By the [Op rules](op-development.md#2-op-admission-and-semantic-boundary), it may choose the
implementation but must not change the result. That gives this design a testable requirement:
**placement-invariant expert arithmetic** ([§15.2](#152-placement-invariance)).

The Program plans every allocation at startup, including the frame pool, staging, mailboxes, row
caches, and the pinned host arena. Nothing allocates during decode.

---

## 8. Decode pipeline

### 8.1 Streams and agents

| Agent | Role |
|---|---|
| Compute stream | The captured decode graph per exact batch size and window |
| Copy stream (DMA) | Prefetch into staging frames; background promotions. Uses the copy engine, not SMs. |
| CPU expert engine | N−2 pinned spin workers (§10) |
| NVMe agent | One pinned thread on io_uring with SQPOLL and IOPOLL (§12) |
| Engine worker | Unchanged. It launches graphs and commits rounds. It does **not** gather PLE rows or plan experts. |

All communication between the GPU and the host agents uses **mapped mailboxes**: pinned host
memory mapped into the device address space, holding sequence-numbered slots. The device publishes
with `st.release.sys` after `__threadfence_system()`. The host polls with acquire loads. The device
waits on host completions with a bounded `ld.acquire.sys` spin, inside the consuming kernel's
prologue, so no separate spin kernel is needed. A host-side watchdog turns an agent failure into an
Engine-wide failure ([engine architecture §7.4](engine-architecture.md)).

### 8.2 Router, top-k and residency in one kernel

One kernel per layer does all of the following:

1. computes router logits (BF16 `[512,2560]` GEMV);
2. computes the softmax, top-10 and renormalization (exact lower-id tie rule);
3. looks up each selected expert in the device residency table `res[layer][expert] → frame | STAGED | ABSENT`;
4. classifies every selected expert as a **hit**, a **staged hit** (landed by prefetch), or a
   **CPU miss**;
5. writes a compact job list for the GEMV kernel;
6. only if CPU misses exist, publishes one mailbox record: activation (BF16, T×2560), miss ids,
   route weights, layer and epoch;
7. computes the next layer's predicted top-k (§8.4) and publishes prefetch requests.

When a layer has no CPU miss, the host never hears about it. That removes Strata's 48 per-window
handshakes in the common case.

### 8.3 Expert execution and combine

- **Hits and staged hits.** One grouped NVFP4 kernel runs gate/up, SwiGLU, and down for all
  resident jobs of the layer. The intermediate stays in shared memory or registers, not in global
  memory. Decode (T ≤ 8 per expert) uses a W4A16 GEMV path with FP32 accumulation. Wider verify
  windows and C=8 jobs switch to block-scaled MMA (`mma.sync … kind::mxf4nvf4.block_scale`), with
  activations kept at A16 or quantized to A4 only where the recipe Use allows `AllowA4`. The
  shared expert is fused into the same launch as job 0.
- **CPU misses.** These run concurrently with the GPU kernel ([§10](#10-cpu-expert-engine)). The CPU
  writes its BF16 or FP32 outputs (T×2560 per expert) straight into mapped memory.
- **Combine.** The combine is fused into the HC_mlp inject epilogue. It waits on the layer's CPU
  completion sequence (when misses existed), reads CPU outputs straight from mapped memory with
  zero-copy coalesced loads (no copy kernel), forms the route-weighted sum in a **fixed expert-rank
  order**, adds the gated shared expert, and applies `R += inject ⊗ y`.

The exposed stall per layer is about max(0, t_cpu + ~6 µs − t_gpu_hits). Hits run at HBM speed:
about 1.5 µs per resident expert at T=1.

**Miss arbiter.** When a layer's CPU-bound misses exceed what the CPU can finish within the hit
kernel's time, the router kernel sends part of them as **on-demand DMA** into staging frames. The
GPU then computes those experts when they land. The split balances the two finish times using the
measured CPU and PCIe service rates, bounded by their shared DRAM bandwidth. This is FreeToken's q\*
and Strata's `pcie_frac` idea, decided per layer from the actual miss count instead of as a global
fraction. In the common case of 0-2 misses per layer, everything goes to the CPU.

### 8.4 Lookahead prefetch

Every layer's MoE phase is preceded by its HC_attn mixer, its mixer (GDN or QSA), and its HC_mlp.
That is ~60-80 µs of pure GPU time, during which host DRAM and PCIe sit idle in both studied
engines. This design fills that window:

- **Prediction.** After layer l's router, the same kernel evaluates layer l+1's router on layer l's
  FFN-side mixed input `x_l`. With hyper-connections the residual moves slowly, so this
  approximates layer l+1's input. The cost is one extra 512×2560 BF16 GEMV per layer (2.6 MB, about
  2 µs).
- **Request.** The predicted top-k′ experts (k′=10-14, tuned by M2 traces) that are neither resident
  nor already staged are written to the prefetch ring, ranked by predicted route weight.
- **Transfer.** The copy stream, driven by a tiny device-side scheduler kernel that reads the ring,
  issues one contiguous DMA per expert into free staging frames. The transfer stops at a
  per-layer budget, so it never spills into layer l+1's MoE phase. Within that window roughly one
  to two experts fit (≈80 µs × 50 GB/s ≈ 4 MB).
- **Landing.** A landed expert is marked `STAGED` with an epoch. If the actual routing selects it,
  it is a staged hit on the GPU. Otherwise it is simply overwritten later.
- **Promotion is free.** A staged expert that turns out hot is promoted by swapping frame roles in
  the table. No copy is needed.

Depth-2 prediction (layer l+2) is evaluated in M2 and used only if its recall/byte justifies the
DRAM it consumes. The prefetch budget is shared with the CPU engine's DRAM bandwidth through a
global token bucket. Prefetch runs only while the CPU engine is idle, which is exactly the
non-MoE part of each layer. As a result, DRAM is kept busy for nearly the whole token rather than
only during MoE phases.

### 8.5 Batch and verify windows

With T columns (C ≤ 8 lanes and/or a K+1 verify window), the router kernel produces the union of
experts. Each distinct expert is fetched or computed **once** for all columns routed to it, on both
the GPU and the CPU. CPU cost scales with distinct misses, not with columns, because the CPU GEMV
is DRAM-bound up to T≈8.

### 8.6 Kernel count and launch overhead

The target is **≤ 10 kernels per layer** in the decode graph, compared with ~32-40 in the studied
engines:

| # | Kernel |
|---|---|
| 1 | HC_attn: grouped offset RMSNorm, W_down, sigmoid/silu gates, W_up, collapse (one kernel, FP8 weights) |
| 2-4 | GDN: fused qkv\|z\|a\|b NVFP4 GEMV with conv + L2-norm + gating epilogue; recurrence + gated sigmoid norm; out_proj with HC inject epilogue. QSA instead uses four kernels: fused QKVG + index projection + norm/RoPE + KV/index append; block score + top-512 select; sparse attention + gate; out_proj + inject. |
| 5 | HC_mlp (as 1) |
| 6 | Router + top-k + residency + prediction + mailbox publish (§8.2) |
| 7 | Grouped hit-expert kernel + shared expert |
| 8 | Combine + HC inject (waits on CPU completion only when misses exist) |

Every kernel uses programmatic dependent launch ([op development](op-development.md#programmatic-dependent-launch))
so prologues overlap the previous kernel's tail. A persistent per-layer **megakernel** is a phase-2
option, adopted only if nsys attribution in M6 shows launch and tail gaps above ~15% of a token.
It is not built speculatively.

### 8.7 Round boundary without host synchronization

The decode graph ends with these steps:

1. sampling (existing sampling Op);
2. the n-gram row ids of the new token, computed on the device (§12.3);
3. the device row-cache probe;
4. the NVMe request publication;
5. writing the sampled token to a mapped output slot.

The Engine worker reads the token without a full device synchronize, with an event query or
host-side wait on the mapped slot, commits through the existing transaction path, and enqueues the
next graph. The next round's embedding and PLE consume device-resident data that the previous graph
already produced. Host gathers and ingress builds that ninfer-ext performs between rounds disappear.

---

## 9. Expert cache and the shared VRAM frame pool

### 9.1 Frames

After dense weights and fixed state, device memory is one **frame pool** of 2.64 MiB frames (one
expert record each, 2 MiB-aligned VMM chunks underneath). A frame's role changes at runtime through
mappings, never through allocation:

| Role | Contents |
|---|---|
| `expert` | A cached expert. Always **clean**: the host bank holds an identical copy. |
| `staging` | A prefetch or promotion target (§8.4) |
| `kv` | Paged KV pages (several pages per frame) |
| `loan` | A frame inside a KV or prefill reservation that currently holds a clean expert |
| `prefill` | Prefill staging and workspace (§13) |
| `vision` | Vision tower weights and activations while a vision request runs |

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
- **Reclaim happens at worker boundaries.** When a lane's KV frontier will cross into a loaned frame
  during the next unit, the Program reclaims it before launch. Graphs address KV and experts
  through page and frame tables, so the graphs stay valid.
- **The same mechanism covers the rest:** prefill arenas (§13), vision weights, and MTP draft
  buffers borrow frames and return them, the borrowed experts being re-admitted lazily by the
  policy instead of re-copied eagerly. Strata re-copies them eagerly after every prompt, which
  costs short prompts up to a third of their speed.

Result: memory that a 256K-capable configuration reserves for KV (4.1 GB, about 1,550 frames) keeps
serving experts until the context actually grows into it.

### 9.3 Epochs: never block on the cache

- Each decode round has an epoch number. A kernel reads the residency table at its epoch.
- An eviction unmaps an entry immediately for future epochs. The frame becomes reusable only after
  the last epoch that could read it retires (an event-ordered retire queue).
- A promotion copies into a free or retired frame on the copy stream. The entry is published with a
  device-side flag write ordered after the copy.
- **No round ever waits for a promotion.** Until the copy lands, a request for that expert is served
  as a CPU miss. Strata's #463 wait exists to make outputs deterministic; here determinism comes
  from placement-invariant arithmetic instead (§15.2).

### 9.4 Replacement and admission policy

**Evidence on this model.** FreeToken's trace replay of Qwen3.8-Flash-Next routing (§3.3) shows:

- LRU is a strong baseline;
- frequency-only policies lose, by as much as 61% more misses;
- Belady's offline optimum misses **52-62% less than LRU**.

The headroom is real, but it comes from **short-range reuse plus knowledge of the next uses**, not
from long-run popularity. The shared expert weakens token-to-token locality (routing-consistency
study, §3.3). The policy therefore starts from recency and adds only the signals that replay shows
to help. Every frame holds an expert of the same size, and every miss costs the same DRAM bytes, so
no size or cost weighting is needed.

**Base: segmented recency over all 49 layers (one global pool).**

- An expert enters a probationary LRU segment, and is promoted to a protected LRU segment on its
  second reference within the segment's lifetime.
- Victims come from the probationary tail first.
- Ghost entries for recently evicted ids adapt the split between the two segments, as ARC does.
- The clock is a **round counter**, not a per-layer-call counter. ninfer-ext's per-layer stamping
  distorts age across layers.

**Refinements, each kept only if M1 replay shows a gain:**

1. **Prediction-informed protection.** Experts named by the next-layer prediction (§8.4) for the
   current round cannot be evicted during that round. Belady's advantage is knowing the next use;
   prediction is the online approximation of that knowledge.
2. **Prefetch as admission.** Staged experts that are actually used are promoted by remapping, with
   no copy. Staged experts that were never used expire without disturbing the resident set.
3. **Per-layer floors.** A layer keeps a minimum resident count (AdapMoE-style allocation,
   recomputed from replayed per-layer hit curves). This stops one layer's burst from starving
   another layer.
4. **Long-term prior.** A persisted per-user decayed usage score (half-life ~16K tokens) seeds the
   cache at startup and breaks ties among probationary victims. It is only a tiebreaker and never
   overrides recency: replay showed that frequency used as the primary key loses.
5. **MTP experts.** They sit in the same pool. Their recency updates are weighted by measured draft
   acceptance.

**Mechanics.**

- The router kernel updates recency state on the device, with no host traffic.
- Victim selection is amortized. A device-side sweep keeps a small precomputed victim list per
  round. ninfer-ext instead runs a single-CTA radix select over all slots for every layer that has
  misses.
- Promotion of a CPU-served miss is **bandwidth-budgeted**. It draws on the DRAM token bucket shared
  with prefetch, runs only in GPU-only phases, and is capped per round. Under heavy CPU miss load,
  promotions defer rather than compete.
- On shutdown, and periodically while idle, the state is saved and seeds the next start. This
  generalizes Strata's `--expert-profile-save`.

**Selection by trace-driven simulation before implementation (M1).** The candidates are:

- LRU (ninfer-ext, FreeToken);
- Strata's per-layer decayed-LFU exchange;
- ARC, SLRU, LRU-2 and W-TinyLFU;
- the policy above, with each refinement switched on and off;
- Belady's optimum, as the upper bound.

They run on held-out traces at 30-45% capacity, C ∈ {1, 4, 8}, and window T ∈ {1, 4}. Replay
reports misses per token **and** DRAM bytes, so prefetch traffic is counted. The shipped policy is
the best candidate on held-out traces. If nothing beats LRU by more than the replay noise, the
product ships LRU plus prediction-informed protection and prefetch, and says so.

---

## 10. CPU expert engine

### 10.1 Why compute on the CPU at all

A miss served over PCIe costs 2.765 MB of DRAM read **plus** 2.765 MB of PCIe transfer **plus** an
SM or copy-engine slot, and the GPU cannot start that expert until the whole record arrives. A miss
served on the CPU costs 2.765 MB of DRAM read, while the GPU works on the hits in parallel, and only
T×2560 values cross PCIe. For critical-path misses the CPU is strictly better. PCIe is reserved for
*predicted* misses (prefetch) and promotions, which are off the critical path.

### 10.2 Kernels

The primary route is **A16**, and it is placement-invariant (§15.2):

1. Decode each 16-value block: E2M1 nibbles → FP32 through a 16-entry LUT (`vpermps`/`vpshufb`), then
   multiply by the E4M3 block scale (LUT).
2. Accumulate FP32 FMAs against the BF16 activation widened to FP32, using a **canonical reduction
   order**. The GPU kernel uses the same order (§15.2).

| ISA | Implementation |
|---|---|
| AVX-512F + BW | 16-wide FMAs, `vpermb` decode |
| AVX2 + FMA (Arrow Lake, Zen 3) | 8-wide FMAs, `vpshufb` decode |

There is also an optional **A8** route, used only when a recipe Use allows `AllowA8` for routed
experts. It quantizes the activation to int8 per 16-block, maps E2M1×2 to int8 {0, ±1, ±2, ±3, ±4,
±6, ±8, ±12}, and computes integer dot products with `vpdpbusd` (AVX-512 VNNI / AVX-VNNI). The
scales are applied per block. It is about 2× less compute than A16, but it is not placement-invariant
against the GPU A16 path, so it is opt-in.

Compute is not the limit. A 16-core Zen 5 sustains far more FP32 FMAs than 80 GB/s of NVFP4 needs,
and the A16 route should be DRAM-bound on AVX-512. M3 verifies that on AVX2 hosts. If the A16 route
turns out compute-bound there, the cost model in §11.2 learns that from measured service rates.

### 10.3 Execution

- **Workers.** One pinned spin worker per physical core minus two: one core for the NVMe agent and
  one for the Engine worker. SMT siblings stay idle, and hybrid E-cores are measured separately.
- **Partitioning.** A layer's misses are split across workers in two phases, with an atomic counter
  barrier between them:
  1. gate/up row bands of all missed experts → SwiGLU on the owning worker;
  2. down row bands.

  The partition is static per (miss count, T) and balanced by bytes.
- **Activation input.** The worker reads the activation from the mailbox record (T×2560 BF16).
- **Output.** Results go back into the layer's mapped output slab, followed by the completion
  sequence number.
- **Memory.** Expert banks sit on 2 MiB pages, prefaulted and locked (the transparent huge page
  madvise used by ninfer-ext was worth +11-32%). Explicit hugetlbfs is used when configured. Reads
  use non-temporal software prefetch two bands ahead.
- **Idle behavior.** Workers spin while a decode is active and park on a futex between requests, so
  an idle server does not burn CPU.
- **Calibration at startup.** A 2-second probe measures single-core and all-core NVFP4 GEMV
  throughput, DRAM bandwidth with and without concurrent DMA, and PCIe H2D bandwidth. These numbers
  feed the cost model and the reports.

---

## 11. Speculative decoding

### 11.1 A drafter that never waits on the host

The MTP drafter runs entirely on the GPU, using **residency-bounded routing**:

- The MTP MoE takes the top-10 over the experts that are resident (or staged) and renormalizes over
  that set.
- Non-resident MTP experts are simply not used during drafting. Draft quality affects only
  acceptance; the verify step keeps the output distribution exact.
- MTP experts carry the acceptance-weighted cache priority from §9.4, so the drafter's routing stays
  close to the true routing.

The draft also reuses the existing optimized proposal head (reduced vocabulary) rather than a full
`lm_head` read per draft step (ninfer-ext's cost).

PLE rows for draft tokens are requested from NVMe when each draft token is produced. They are
therefore usually already in the device row cache when the verify forward needs them (§12.3).

### 11.2 Miss-aware draft length

Every round, the policy picks K by maximizing expected tokens per second:

```text
E[tokens(K)] / E[t_round(K)],
    t_round(K) = t_gpu(K+1) + t_miss(U_{K+1}, h_now) + t_draft(K)
```

- **U_T and h_now** are tracked online from the router kernel's counters: an EWMA of distinct
  experts per layer as a function of T, and the true VRAM, staged and CPU composition.
- **E[tokens(K)]** comes from the existing per-position acceptance estimates.
- **t_cpu** comes from the measured CPU service rate (§10.3).

This generalizes NInfer's adaptive MTP and Strata's `DraftPolicy` by **including the miss cost that
grows with the window**. That cost is exactly what made ninfer-ext's MTP lose on chat. When the
cache is cold or the topic shifts, K falls toward 0-1. When the cache is warm, K rises to 3-5.

N-gram copy proposals ([ngram](../ngram.md)) and DFlash-style tree verification stay available, and
the same cost model decides them. A tree's extra columns cost misses too.

### 11.3 Overlap

- Drafting for round r+1 starts on the compute stream right after verify r's sampling, in the same
  graph when K is fixed for the topology.
- The commit fold of the recurrent states follows the existing asynchronous commit path.
- No stream synchronization happens per draft step. Strata synchronizes once per draft step.
- **Verify-union prefetch** (SP-MoE's idea, adapted). While the drafter runs, layer 0's router is
  evaluated on the draft tokens' embeddings. Its predicted non-resident experts are DMA'd before the
  verify forward starts. From layer 1 on, the per-layer lookahead (§8.4) already predicts for all T
  columns.

---

## 12. PLE n-gram table on NVMe

### 12.1 Requirements

- 16 rows of 160 FP8 values each per token: 2.5 KiB of payload.
- Rows depend on `t[p-2], t[p-1], t[p]`. EOS restarts the window, as defined in the model reference.
- The rows are consumed **once**, before block 1's attention mixer. In decode they are not needed
  until layer 0 has finished, about 80-150 µs after sampling.
- The table must stay off the page cache. RAM is fully budgeted (§14.2).

### 12.2 NVMe layout

- Rows are 160 B, 25 rows per 4 KiB block, and never straddle a block. That is a 2.4% padding cost,
  stored on disk as ~52.4 GB.
- The single BF16 scale is held in the Model.
- **One row is one 4 KiB `O_DIRECT` read.** Rows of one head are contiguous in the head's prime
  range, so prefill reads that touch nearby rows coalesce into one block.
- The file can live on a different NVMe drive from the expert banks, which are only read at load.

### 12.3 Three-level row cache with device-driven requests

| Level | Size (default) | Mechanism |
|---|---|---|
| L0: device | 64 MiB ≈ 400K rows | Open-addressing hash on the row id with CLOCK replacement. Probed by the decode graph itself. |
| L1: host | 2 GiB ≈ 13M rows | Pinned. The NVMe agent probes it before issuing I/O. The default size is set by the RAM plan. |
| L2: NVMe | — | io_uring with registered fixed buffers, `O_DIRECT`, SQPOLL on the agent's core, IOPOLL when the device supports polled queues. Queue depth up to 256. |

Decode sequence:

1. The sampling kernel's epilogue computes the 16 row ids for the new token on the device. The
   `ngram_row_ids` Op is exact integer arithmetic: XOR of signed 64-bit products and a prime
   modulus.
2. The kernel probes L0 and writes hits' row indices.
3. For L0 misses it publishes `(row id, destination)` records to the NVMe mailbox ring.
4. The NVMe agent spins on the ring. It serves L1 hits by `memcpy` into the mapped landing buffer.
   For L1 misses it submits reads that land directly in registered pinned buffers mapped to the
   device. Completion sets a per-request sequence number.
5. Layer 0 runs: HC, GDN, MoE, typically 80-150 µs. A polled 4 KiB random read on a modern
   Gen4/Gen5 NVMe takes about 40-90 µs.
6. The PLE kernel before block 1 waits, with a bounded acquire spin, for the token's 16 rows. It
   dequantizes FP8 on the device, applies the PLE mathematics, and inserts the rows into L0. There is
   no host decode of rows. Both studied engines dequantize on the CPU.

Additional rules:

- **Speculative windows.** Draft tokens' rows are requested at draft time (§11.1). Only the bonus
  token's rows arrive at round start, and those are hidden by layer 0.
- **Prefill.** For chunk i+1, the row ids are computed on the GPU, deduplicated, and probed in L0/L1.
  The misses are sorted by block and read with QD256 while chunk i computes. A 32K prompt needs at
  most 512K distinct rows. At ~600K-1M IOPS that is ≤ 1 s of I/O, hidden behind ~4-5 s of compute.
- **Power-state stalls.** The NVMe agent issues a periodic tiny read while a request is active.
  Strata measured 50-150 ms autonomous power-state exit stalls on some drives.

Expected exposure: zero for L0/L1 hits, and for most L2 misses at T=1. When the drive is slower than
layer 0, the residual wait is visible in the per-round telemetry, broken out per layer. It is not
hidden in an average.

---

## 13. Prefill

- **Chunks.** 8K by default. 16K-32K for a solo long prompt, with the arena borrowed from frames
  (§9.2).
- **Experts.** Layer-streamed. For each layer:
  1. the router determines the set of experts used by the chunk;
  2. resident experts are used in place, with no device-to-device copy (they are addressed through
     the frame table);
  3. non-resident experts are DMA'd as **one contiguous copy per expert record** into a
     double-buffered staging region, with layer l+1's copies overlapping layer l's compute.

  Assignments are grouped by expert and run through a block-scaled NVFP4 tensor-core grouped GEMM:
  - **A4** activations where the Use allows it, with the per-layer divisor;
  - otherwise **A16**, by dequantizing the weights to BF16 in shared memory.
- **CPU help for thin experts.** An expert routed to only a few tokens in a chunk costs more PCIe
  bytes than CPU work. For chunks below ~1K tokens, experts with ≤ τ assigned tokens are computed on
  the CPU in place (τ from the cost model). This lifts short-prompt prefill, which in ninfer-ext is
  431 tok/s at 512 tokens.
- **Bound.**

  | Chunk | Per-chunk time | Effective rate |
  |---|---|---|
  | 16K | ≈ 1.5 s of PCIe (48 × 1.42 GB × 60% non-resident / ~52 GB/s) | ~10K tok/s |
  | 8K | ~0.8 s | ~10K tok/s |

  At large chunks the GPU's FP4 tensor-core compute and QSA attention become the limit. The Must
  target of 6,000 tok/s at 32K leaves margin for those.
- **QSA prefill.** Pooled, normalized, RoPE'd index keys are materialized **once per 4-token block**
  when the block completes, and stored in an index-key plane beside KV. Select then scans one
  128-wide key per block. This is the same insight as ninfer-ext's `697ff0c7`, built into the KV
  layout rather than recomputed per call.
- **MTP KV for the prompt** is built in batch from the chunk residuals.

---

## 14. Memory plans

### 14.1 VRAM: 32,607 MiB card, C=1, MTP, 32K context, FP8 KV

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
as the context actually fills**.

### 14.2 Host RAM: 96 GB (≈ 93 GiB visible)

| Item | GB |
|---|---:|
| Expert banks, 25,088 × 2.7648 MB, pinned 2 MiB pages | 69.4 |
| Embedding (BF16), pinned | 1.27 |
| L1 PLE row cache | 2.1 (configurable 0.5-4) |
| Mailboxes, landing buffers, io_uring buffers | 0.2 |
| Host checkpoint tier (prefix cache), capped | 2.0 (configurable) |
| Process: tokenizer, server, frontend | ~1.5 |
| **Total NInfer** | **~76.5** |
| OS, desktop, page cache headroom | ~19 |

Additional rules:

- **Loading.** Experts are read with `O_DIRECT` straight into the pinned arena (~10 s from a 7 GB/s
  drive). Loading never doubles memory through the page cache.
- **Startup check.** Startup verifies `RLIMIT_MEMLOCK` and `MemAvailable` against this plan and
  fails with a precise message. It does not swap.
- **Machines with less than ~88 GB of RAM** are outside the supported range. Exclusive VRAM/host
  residency (an expert lives in exactly one tier) could support 64 GB machines later. It is listed
  as a future option and is not designed here.

---

## 15. Numerics and qualification

### 15.1 Op oracles

Every new Op is qualified against a naive FP32/FP64 oracle at the real shapes, under the
[Op development](op-development.md) rules:

| Op | Oracle | Boundaries |
|---|---|---|
| `offloaded_sparse_moe` | FP64. NVFP4 independently decoded from codes, block scales and divisors; BF16 inputs. | Every residency mix: all-hit, all-CPU, all-staged, mixed. T ∈ {1, 2, 4, 8, 16, 64, 512, 8192}. Exact top-10 lower-id ties. |
| `hyper_connection` | FP64 | FP8 weights decoded exactly; grouped norm per stream |
| `qsa_index` / `qsa_select` | FP64 block scores | Near-tie allowance as in the existing selector contracts; dense equivalence while n ≤ 2051 |
| `ple_ngram_injection` | FP64 | FP8 rows decoded with the scalar scale; conv state transition; EOS restarts |
| `ngram_row_ids` | **Exact integer** | Signed 64-bit products, primes 20,000,003-20,000,171, offsets; EOS and sequence start |
| CPU NVFP4 kernels | Same FP64 oracle, independently | A16 and A8 routes separately |

The fork's FP64 Python reference may be consulted as a cross-check of the published mathematics.
NInfer's oracle is written from the upstream definitions, consistent with [§1.4](#14-non-goals).

### 15.2 Placement invariance

**Requirement.** For the A16 routes, an expert's output is **bit-identical** whether it ran on:

- the GPU from a cache frame;
- the GPU from a staging frame;
- the CPU.

**Mechanism.** Both implementations follow one canonical arithmetic:

- decode each block to FP32;
- within each 16-block, a fixed 16-term FMA chain;
- across the 160 (gate/up) or 40 (down) blocks of a row, a fixed pairwise tree;
- FP32 SwiGLU with the same `expf` polynomial (no fast-math intrinsics in either path);
- a fixed expert-rank order in the combine.

**Why it is worth the constraint:**

- Cache contents depend on timing, so without invariance two identical requests can produce
  different tokens. Strata measures 95-97.7% top-1 agreement between cache states.
- With invariance:
  - greedy outputs do not depend on cache state, prefetch timing, or the CPU/GPU split;
  - no round has to wait for a promotion to stay deterministic (§9.3);
  - every residency route can be tested by exact comparison against the all-GPU route.

**Cost.** If the canonical order costs more than ~5% of the hit GEMV's bandwidth in M3
measurements, the design falls back to oracle-tolerance qualification for the GPU route. That
change, and its effect on determinism, will be stated explicitly.

### 15.3 Recipe qualification

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

### 15.4 Engine-level tests

- Incremental decode equals one-shot prefill (exact on A16 routes).
- Speculative output equals plain output (greedy, exact).
- Cache-cold equals cache-warm equals all-CPU forced mode (exact on A16).
- PLE served from L0, L1 and NVMe gives exact equality.
- Fault injection:
  - an NVMe read error;
  - CPU worker death;
  - a watchdog timeout.

  Each must reach the Engine-wide failure path.

---

## 16. Implementation plan

Each milestone has exit criteria measured on the target machine. Later milestones may change
because of earlier findings. Such changes and the evidence for them are recorded here, not silently
absorbed.

| # | Milestone | Exit criteria |
|---|---|---|
| **M0** | Source facts. Record NVIDIA NVFP4 `hf_quant_config.json`, tensor dtypes, and the actual machine: CPU ISA, DRAM bandwidth, PCIe, NVMe latency at QD1 and QD16. | Recorded facts; §4 assumptions confirmed or revised |
| **M1** | Routing traces and cache simulation. Collect traces from the BF16 or NVFP4 model on a corpus: from an existing engine's routing dump, a Transformers run on a larger host, or M4's first functional engine. Build the trace simulator (§9.4 candidates and Belady), plus next-layer prediction recall. | Hit-rate curves at 30-45% capacity. Shipped policy chosen, with its gap to LRU and to Belady reported. Prediction recall@k′ and useful-byte ratio measured. §4 projection updated with the real h. |
| **M2** | Converter recipe, expert record layout, NVMe table layout, residency classes, loader with `O_DIRECT` pinned load | Exact import tests; load time; RAM peak ≤ plan |
| **M3** | Ops with oracles: `offloaded_sparse_moe` (GPU A16/A4 + CPU A16/A8), `hyper_connection`, QSA, PLE, `ngram_row_ids`. Placement-invariance test. CPU kernel bandwidth on target CPUs. | All Op qualifications pass. CPU NVFP4 GEMV ≥ 80% of measured DRAM bandwidth (AVX-512) or measured shortfall recorded (AVX2). |
| **M4** | Functional model, all-host-experts mode: no cache, every expert CPU-served. Prefill + decode + MTP + Vision through the Engine; CausalScoring perplexity. | Model oracle agreement; recipe KL report (§15.3); first end-to-end numbers |
| **M5** | Frame pool, residency table, epochs, §9.4 policy, loans, lookahead prefetch | Measured hit composition matches the M1 simulation within 2 points. No round blocks on promotions. |
| **M6** | Decode graph fusion (≤ 10 kernels per layer, PDL), device-driven PLE, host-sync-free round boundary. nsys attribution of one token. | Plain decode Must target (≥ 120 tok/s). Attribution table published. |
| **M7** | Speculative decoding with residency-bounded drafter and miss-aware K; n-gram copy | Speculation Must target (≥ 160 tok/s). Spec never below plain on any corpus category by more than noise, shown by repeated runs. |
| **M8** | Prefill streaming, thin-expert CPU help, QSA pooled-key plane | Prefill Must target; short-prompt prefill reported |
| **M9** | Head-to-head campaign vs Strata UD-Q4_K_XL and ninfer-ext on the target machine; serving C=1..8; 4K/32K/128K; quality report | §1.3 table filled with every result, favorable or not |

---

## 17. Documentation and authority changes

When the implementation lands:

- `AGENTS.md` and `README.md`: add `Qwen4ExpForCausalLM` to the product architectures, and the
  host-RAM and NVMe requirements.
- `docs/maintainer/qwen4-exp-model.md`: new model reference (mathematics, config, state, MTP, PLE).
- `docs/maintainer/expert-offload.md`: frame pool, residency, epochs, policy, mailboxes, CPU engine.
  §8-§12 of this file move there.
- [Engine architecture](engine-architecture.md): Program ownership of host agents and frames.
  Residency as an Op execution resource.
- [Artifact container](artifact-container.md), [storage layouts](storage-layouts.md): residency
  classes, `expert_record_nvfp4_*`, the n-gram volume.
- [Weight conversion](../weight-conversion.md), [CLI](../cli.md), [serving](../serving.md): recipe and
  options (`--expert-cache`, `--cpu-experts`, `--ngram-file`, `--ple-row-cache`).
- `docs/performance/qwen3.8-flash-next.md` and a model card.
- This file is deleted.

---

## 18. Risks and open questions

| Risk | Effect | Mitigation / decision point |
|---|---|---|
| Routing locality is weaker than expected (h < 0.8 at 40%; the shared expert lowers token-to-token reuse) | Plain decode stays near the projection's 0.7 row; speculation gains shrink | M1 decides early. Fallback levers: HC and lm_head NVFP4 if quality allows (+~400 frames); opt-in reduced-precision copies of the **coldest** experts (HOBBIT-style mixed precision) as an explicit, labeled product option, never default. |
| Next-layer prediction recall is low for this architecture | Less DRAM time-shifting | Prefetch is budgeted and self-disabling when its measured useful-byte ratio drops below a threshold |
| AVX2-only hosts are compute-bound in A16 | CPU misses slower | A8 opt-in route. Cost model shifts misses to PCIe. Reported per host class. |
| DRAM bandwidth of the user's platform (DDR5 speed, 2 vs 4 DIMMs) | Directly scales miss service | M0 measures. 4-DIMM DDR5 often runs at lower speeds, so the 96 GB configuration (2×48 GB) is preferred. |
| NVIDIA's dense tensors are already NVFP4 or excluded differently than assumed | Recipe table changes | M0 reads `hf_quant_config.json`; recipe follows the source |
| Mapped-memory spin waits inside graphs, and driver behavior | Hangs or latency spikes | Bounded spins with a watchdog; fall back to `cuStreamWaitValue32` graph memory-op nodes if measured better |
| NVMe without polled queues, or high latency (QLC, DRAM-less) | PLE wait exposed | L1 cache size; telemetry; documented drive requirement |
| Placement-invariant arithmetic costs too much | Lower hit GEMV throughput | §15.2 decision rule, disclosed |
| 96 GB is tight with a desktop session | OOM or swap | Startup plan check; configurable L1 and checkpoint tier; recommended server-only operation |

Open questions answered by measurement, not assumption:

- optimal k′ and depth of prediction;
- segment split, per-layer floors and the weight of the long-term prior;
- frame pool versus KV loan reclaim frequency at C=8;
- whether a persistent megakernel is worth it (M6 attribution).

---

## 19. Sources

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
