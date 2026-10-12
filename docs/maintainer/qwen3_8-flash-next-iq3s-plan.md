# Qwen3.8-Flash-Next GSQ-RCO IQ3_S: implementation plan

**Status: plan, not implemented.** This is a temporary planning document for a bit-exact Infernix
artifact of [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)
`IQ3_S/` with an MTP drafter sourced from Qwen's checkpoint, and for the optional RAM-resident n-gram
table. When the work lands, its stable content moves into the authorities named in
[§14](#14-documentation-and-authority-changes) and this file is removed. The Flash-Next design is
cited as **FN §x** ([qwen3_8-flash-next-design.md](qwen3_8-flash-next-design.md)) and the GLM-5.3
design as **GLM §x** ([glm5_3-flash-design.md](glm5_3-flash-design.md)).

Code references are to the newest local tree, branch `Infernix` at `217135bba` (worktree
`E:\NInfer-V3\local\workdirs\NInfer-V3-issue1`, 22 commits ahead of `origin/main`). The companion
plan for the official FP8 checkpoint is [qwen3_8-flash-next-fp8-plan.md](qwen3_8-flash-next-fp8-plan.md);
§13 lists the work both plans share.

Labels: **measured** (this machine, method named), **upstream** (read from the source file or
code), **model** (arithmetic on measured or upstream inputs), **estimated** (judgement, with a
range). A milestone in brackets replaces each model or estimate with a measurement.

---

## Contents

1. [Deliverable, baselines and acceptance](#1-deliverable-baselines-and-acceptance)
2. [Source facts](#2-source-facts)
3. [What other implementations do](#3-what-other-implementations-do)
4. [Design decisions](#4-design-decisions)
5. [Artifact and conversion](#5-artifact-and-conversion)
6. [Arithmetic contracts](#6-arithmetic-contracts)
7. [Routed experts](#7-routed-experts)
8. [Dense path](#8-dense-path)
9. [PLE n-gram table: IQ4_NL rows and RAM residency](#9-ple-n-gram-table-iq4_nl-rows-and-ram-residency)
10. [MTP drafter from Qwen's checkpoint](#10-mtp-drafter-from-qwens-checkpoint)
11. [Performance model](#11-performance-model)
12. [Qualification](#12-qualification)
13. [Milestones](#13-milestones)
14. [Documentation and authority changes](#14-documentation-and-authority-changes)
15. [Risks, options and rejected alternatives](#15-risks-options-and-rejected-alternatives)
16. [Sources](#16-sources)

---

## 1. Deliverable, baselines and acceptance

### 1.1 Deliverable

- A recipe `qwen3_8_flash_next_gsq_rco_iq3s` producing `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix`:
  every tensor of the GGUF stored **bit-exactly** (codes and scales verbatim; only lossless
  re-layouts: plane splits, whole-row permutations, per-expert records, alignment padding), plus an
  IQ4_NL n-gram volume holding shard 2's rows verbatim.
- An MTP drafter and proposal head built from Qwen's MTP tensors (§10). The GGUF has none
  (llama.cpp's exporter sets `supports_mtp_export = False`). The drafter only proposes; verification
  keeps the target's output, so it does not affect exactness.
- Vision from the repository's BF16 `mmproj` (exact), as a later milestone.
- Runtime support through the public Engine for every route Flash-Next has today: plain and MTP
  decode, prefill, the hybrid prefix cache, C ≤ 8, VRAM/RAM/SSD expert tiers.
- `--ngram-residency auto|ram|volume`: the whole n-gram table in host RAM, or on NVMe as today (§9).

### 1.2 Baselines (RTX 5090 on PCIe 5.0 x8, i9-13900K, 96 GB DDR5-5800)

| Baseline | Decode | Prefill | Source |
|---|---|---|---|
| Infernix recipe B (NVFP4 experts, 8-bit dense), INT8 KV | plain 132.2, MTP 157.6 tok/s (pg1024+256 corpus) | pp131072 5,767 tok/s; agentic no-hit prefill 5,879 tok/s | measured, FN §19.3.21, README |
| Strata 0.1.38-0.1.39, this GGUF (IQ3_S), RTX 5090 | 161-193 tok/s (9950X3D/9700X, Linux); 127-142 (0.1.33, Windows, shared GPU) | 3,072-6,036 tok/s (4K-128K) | upstream, Strata `bench/results/2026-10-0{3,5}*` |
| llama.cpp #29887 GPU expert cache, Flash-Next Q4_0, RTX 5090, PCIe 4.0 | 67.8 tok/s | — | upstream, PR #29887 |

Strata's numbers are on different CPUs (Zen 4/5 with AVX-512), use GPU/CPU paths with different
rounding (q8_1 vs Q8_K activations), and by default skip low-rank misses (route tail skip, which
changes outputs). Before the design is frozen, M0 measures Strata on this machine with the same file,
with tail skip both on and off.

### 1.3 Acceptance

- **Must, exactness.** `verify_artifact` reconstructs every GGUF tensor's block bytes from the
  artifact and compares them byte for byte (transformed tensors: value for value through their
  stated exact maps, §5.3). Every n-gram row matches. Every Op passes its oracle (§12).
- **Must, invariance.** Greedy output is independent of expert placement (VRAM frame, staging, CPU
  on every ISA), decode width, C ≤ 8, and MTP on/off at C = 1, as for recipes A–C (FN §16.4).
- **Must, quality.** The §16.5 protocol (KL to the 16-bit reference on the frozen texts at several
  prefill chunk sizes) is reported against recipe B. ISTA's own claim (IQ3_S task average 93.26
  against 93.12 BF16) is upstream only. Adoption as a recommended artifact needs KL no worse than
  recipe B's, or a documented trade.
- **Targets (estimated, §11).** Plain decode ≥ recipe B (≥ 132 tok/s at C = 1), stretch 150; MTP
  ≥ 160; pp131072 ≥ 5,800 tok/s; a cold 16K prompt faster than recipe B when the table is in RAM.
  These are gates for the performance milestones, not completion conditions for correctness.

### 1.4 Non-goals

- Arbitrary GGUFs or other mixes. The code is per-format and per-shape, as `ops::linear` is today.
  The `IQ3_XXS`/`IQ2_XS`/`Q2_0` builds of the same repository use the same formats (some only IQ2_S
  or Q2_0), so they come almost free; that is checked, not promised.
- Loading the `.gguf` in place (§15.3).
- Reproducing llama.cpp's logits bit for bit: llama.cpp quantizes activations to Q8_1, Infernix keeps
  its exact A16 arithmetic (§6).

---

## 2. Source facts

Read on 2026-10-10 from the GGUF headers of the local `GSQ-RCO-Coder` build (same writer, same type
histogram), from ISTA's IQ3_S `tensor-allocation/*.rco-allocation.txt`, and from llama.cpp
`3cf03257` (vendored at `E:\Strata\Strata\third_party\llama.cpp`).

### 2.1 Files

| File | Size | Content |
|---|---:|---|
| `IQ3_S/…-00001-of-00002.gguf` | 54.8 GB | 1,223 tensors, 67 metadata keys, arch `qwen4exp`; SHA256 `4c1eb2ce…` (upstream) |
| `IQ3_S/…-00002-of-00002.gguf` | 28,800,138,240 B tensor | one tensor `per_layer_token_embd.weight` [160 × 320,001,536] IQ4_NL at file offset 192; SHA256 `316b46f3…`. **Already on disk**: `E:\Strata\Strata-data\models\coder-IQ1_M\…-00002-of-00002.gguf` has the same SHA256 (measured) |
| `mmproj-Qwen3.8-Flash-Next-BF16.gguf` | 0.91 GB | vision tower, BF16 |

Only shard 1 (54.8 GB) needs downloading; M0 pins the repository revision.

### 2.2 Types (shard 1, 1,223 tensors)

BF16 484, F32 292, Q6_K 129, IQ4_NL 86, Q4_K 47, IQ4_XS 45, IQ2_S 40, Q5_K 35, IQ3_XXS 34,
IQ3_S 20, Q2_0 9, F16 1, Q8_0 1. Per class (upstream allocation file):

| Class | Types (count of layers) |
|---|---|
| `token_embd` [2560 × 248,320] | IQ4_XS |
| `output` (lm_head) | Q6_K |
| GDN `attn_qkv` [10240 × 2560] | Q6_K 22, Q5_K 10, Q4_K 2, IQ4_XS 2 |
| GDN `attn_gate` (z) [6144 × 2560] | Q6_K 14, Q4_K 10, IQ4_XS 8, Q5_K 4 |
| GDN `ssm_out` [2560 × 6144] | Q6_K 29, Q5_K 4, Q4_K 3 |
| QSA `attn_q` [12288 × 2560] | Q6_K 5, Q5_K 3, IQ4_XS 3, Q4_K 1 |
| QSA `attn_k`, `attn_v` [512 × 2560] | Q6_K (k 11, v 10), IQ4_XS 1 (k), Q5_K 2 (v) |
| QSA `attn_output` [2560 × 6144] | Q6_K 11, Q4_K 1 |
| shared `ffn_gate_shexp`, `ffn_up_shexp` [640 × 2560] | Q4_K/IQ4_XS/Q6_K/Q5_K mixed per layer |
| shared `ffn_down_shexp` [2560 × 640] | IQ4_NL 47, Q8_0 1 (layer 47) |
| routers, HC mixers (`hc_*_down/up/inject`, `output_hc_*`), GDN α/β, indexer q/k, PLE key/value | BF16 |
| norms, `ssm_a`, `ssm_conv1d`, `ssm_dt.bias` | F32 |
| `ple_conv1d` | F16 |

### 2.3 Routed experts: seven per-layer combinations

Gate and up always share a type (upstream). Bytes per expert follow from the block sizes
(IQ2_S 2.5, IQ3_XXS 3.0625, IQ3_S 3.4375, IQ4_XS 4.25, IQ4_NL 4.5, Q2_0 2.25 bits per weight):

| gate/up + down | Layers | Record bytes (model) | 4 KiB pages |
|---|---|---:|---:|
| IQ2_S + IQ4_NL | 2-6, 9, 11, 15, 19, 23, 24, 26, 32, 34, 39 (15) | 1,945,600 | 475 |
| IQ3_XXS + IQ4_NL | 0, 7, 14, 16, 18, 22, 31, 33, 37, 38, 41-43, 46 (14) | 2,176,000 | 531.25 |
| IQ3_S + IQ4_NL | 17, 25, 27-29, 35, 40, 44, 45 (9) | 2,329,600 | 568.75 |
| IQ2_S + Q2_0 | 1, 8, 10, 13, 36 (5) | 1,484,800 | 362.5 |
| IQ3_XXS + Q2_0 | 12, 20, 30 (3) | 1,715,200 | 418.75 |
| IQ3_S + Q2_0 | 21 (1) | 1,868,800 | 456.25 |
| IQ4_XS + IQ4_NL | 47 (1) | 2,662,400 | 650 |

All experts: 50.03 GB (46.6 GiB), mean 2.036 MB per expert, against 2,764,800 B for NVFP4 (−26 %).
The tensors are fused 3-D: `ffn_gate_exps`/`ffn_up_exps` [2560, 640, E], `ffn_down_exps` [640, 2560, E]
(ggml order), so each expert is a contiguous slice of each tensor.

### 2.4 Formats (upstream `ggml-common.h`, llama.cpp `3cf03257`)

| Type | Block | Bytes | Value |
|---|---:|---:|---|
| Q2_0 | 64 | 18 | `(c − 1) · d`, c ∈ {0..3}, fp16 d (llama.cpp #24448) |
| IQ4_NL | 32 | 18 | `d · kvalues_iq4nl[nibble]`, 16 values in [−127, 113]; low nibbles are elements 0-15, high 16-31 |
| IQ4_XS | 256 | 136 | `d · (ls − 32) · kvalues_iq4nl[nibble]`, 6-bit ls per 32 |
| IQ2_S | 256 | 82 | `d · (0.5 + s) · 0.25 · iq2s_grid[10-bit][j] · ±1`, 4-bit s per 16, grid values {8, 25, 43}, explicit sign bits |
| IQ3_XXS | 256 | 98 | `d · (0.5 + s) · 0.5 · iq3xxs_grid[8-bit][j] · ±1`, 4-bit s per 32, signs 7 bits + parity |
| IQ3_S | 256 | 110 | `d · (1 + 2s) · iq3s_grid[9-bit][j] · ±1`, 4-bit s per 32, explicit sign bits |
| Q4_K | 256 | 144 | `d · sc · q − dmin · m`, 6-bit sc and m per 32 |
| Q5_K | 256 | 176 | as Q4_K with a fifth bit |
| Q6_K | 256 | 210 | `d · sc · (q − 32)`, int8 sc per 16 |
| Q8_0 | 32 | 34 | `d · q`; equals Infernix `q8_g32_fp16` byte for byte when no code is −128 |

Every value is **an integer code times an integer sub-scale times one fp16 scale** (Q4_K/Q5_K: two
such terms). This is what makes the exact arithmetic of §6 possible.

### 2.5 Export transforms to undo (upstream `conversion/qwen.py`, `conversion/qwen4exp.py`)

| Transform | Tensors | Plan |
|---|---|---|
| `+1` on every `*norm.weight` except `linear_attn.norm.weight` (Qwen3-Next rule), and explicitly on PLE `norm_key/query/conv` and indexer `q/k_layernorm` | F32 gammas | §5.3 |
| `A_log → −exp(A_log)` | `ssm_a` F32 | bind A directly (§8.4) |
| GDN V-head reorder (ggml tiled order) | rows of `attn_qkv` V part, `attn_gate`, α, β, `ssm_dt`, `ssm_a`, conv V channels; **columns** of `ssm_out` | §5.3, §8.4 |
| indexer `index_qk_proj` split into `indexer.q_proj` / `indexer.k_proj` | BF16 | concatenate rows |
| `ple.conv1d` squeezed; PLE shards concatenated into one table | F16, IQ4_NL | §5.3, §9 |
| MTP dropped | — | §10 |

---

## 3. What other implementations do

Read 2026-10-10. [M] = the source reports a run, [C] = a claim, [I] = our inference.

| Source | Technique | Takeaway here |
|---|---|---|
| **Strata** (`E:\Strata\Strata`, 0.1.42) | Per-layer-sized expert slots that never move between layers (`expert_cache.hpp`); raw GGUF blocks per expert `[gate|up|down]`; grouped expert kernels with grid staging in shared memory only for IQ2_* (not IQ3_*); IQ4_NL via `__byte_perm` in registers; `kvalues` in `__constant__` was 2.12× slower [M]; decode-once-apply-to-all-columns with a 16-lane sub-warp that reproduces the 32-lane add tree; fused SwiGLU; PDL with weight loads before `griddepcontrol.wait`; opt-in int8 `mma.sync` prefill with smem codebooks and `cp.async` 4 stages (+4-11 % [M]); Q8_0/Q6_K plane repacks +10-12 % [M]; CPU AVX2 `vpgatherdd` grid decode chosen per core (IQ3_S row 0.34 → 0.19 ms on a 14900KF P-core [M]); host-dequantized PLE rows; `--ple-io ram` (map + VirtualLock after a threaded fault-in) | Per-layer slot sizing, plane repacks, decode-once, smem grids per format by measurement, byte-perm IQ4_NL, CPU gather choice per core: adopted. Strata's GPU/CPU paths round differently (q8_1 vs Q8_K), so output depends on placement (its issues #152, #410): rejected (§6). |
| **llama.cpp** mainline | MMVQ dp4a against Q8_1; signs via `__vcmpne4`/`__vsub4`; IQ4 via two `__byte_perm` 8-entry lookups; MMQ loads i-quants into int8 tiles and reuses the Q8_0 int8-MMA; Q2_0 symbols via `__byte_perm(…, 0x5140/0x7362)` (+36 % tg, #25603 [M]); GPU LRU for host experts (#29887): 67.8 tok/s at 89 % hits [M] | Sign and nibble tricks adopted verbatim in the weight-decode stage. Q8_1 activations rejected (§6). |
| **ik_llama.cpp** | Row-interleaved repacks (`_R4/_R8`) so one activation load feeds 4-8 rows; for slow-to-unpack i-quants, unpack a row block once to 8-bit and run one int8 GEMM over all columns (#515, IQ2_XXS pp 2.7× [M]); AVX-VNNI `dpbusd` kernels for IQ3_S/IQ3_XXS/IQ4_NL/Q4_K/Q5_K/Q6_K R4 (#1446-#1482) | Converter-side row interleave of gate/up (§5.4) and of CPU expert rows; unpack-once-then-int8 for CPU multi-column misses and prefill CPU share (§7.4). |
| **KTransformers** (SOSP'25) | AMX/AVX-512 experts; expert deferral (changes the mathematics) | No AMX/AVX-512 here; deferral rejected (FN §1.4). |
| QTIP / EXL3 / AQLM | Computed or tiny codebooks; codebooks ≤ 8 KB in shared memory are fast, 2^16 entries are not [C] | Confirms that IQ2_S's 8 KB grid in shared memory is fine. |

---

## 4. Design decisions

| # | Decision | Reason |
|---|---|---|
| D1 | **Exact import** of every GGUF block; Infernix-native re-layouts chosen by the converter (planes, interleaves, records), never by the runtime | AGENTS: one artifact, no runtime repacking; bit-exactness is checkable byte for byte |
| D2 | **One canonical arithmetic per class, exact up to a single final rounding** (§6): exact integer products of the stored codes with the existing A16 activation encoding, sub-scales applied in integers, and an exact 128-bit combine of the fp16 scales | Placement invariance (FN §16.2.1, GLM L13) without constraining reduction order; each route (SIMT, IMMA, CPU) may split work freely |
| D3 | **Expert records per layer**, each a whole number of 4 KiB pages; frames in **size classes** with one global LFRU (§7.5) | Seven record sizes; one max-size frame wastes 23 % of VRAM (§11.1) |
| D4 | **Dense layers**: per-(format, shape) routes in `ops::linear`, SIMT and column-invariant for T ≤ 8, tensor-core for prefill; multi-format **segmented** launches where the converter cannot group same-format parents (§8.2) | RCO varies the format per tensor and per layer |
| D5 | **The n-gram table in IQ4_NL**, verbatim rows, 45 per 4 KiB block; `--ngram-residency` RAM mode holding the whole table (26.8 GiB) in locked host memory (§9) | Shard 2 is IQ4_NL; the RAM mode removes NVMe from the PLE path |
| D6 | **MTP from Qwen's checkpoint**, re-quantized as recipes A–C do (`q4_g64_fp16` experts, `q8_g32_fp16` dense) (§10) | The GGUF has no MTP; the existing drafter path is measured and qualified |
| D7 | **Keep the BF16 tensors BF16** (HC mixers, routers, indexer, PLE key/value) | Bit-exactness; a re-quantized variant is a separate gated option (§15.2) |

---

## 5. Artifact and conversion

### 5.1 Recipe

| Class | GGUF | Stored as | Kernel family |
|---|---|---|---|
| Routed experts | IQ2_S/IQ3_XXS/IQ3_S/IQ4_XS + IQ4_NL/Q2_0 | `ggml_expert_v1` records (§5.4), per-layer format pair in the bank's metadata | `offloaded_sparse_moe` ggml family (§7) |
| Dense projections, shared experts, lm_head | Q4_K/Q5_K/Q6_K/IQ4_XS/IQ4_NL | new formats `ggml_q4_k`, `ggml_q5_k`, `ggml_q6_k`, `ggml_iq4_xs`, `ggml_iq4_nl` in plane layouts (§5.2) | `ops::linear`, `projection_fp32` (lm_head) |
| `ffn_down_shexp` layer 47 | Q8_0 | `q8_g32_fp16` (exact; converter asserts no −128 code) | existing Q8 |
| BF16 tensors | BF16 | BF16 | existing |
| `token_embd` | IQ4_XS | `ggml_iq4_xs`, host-resident | host embedding gather with exact decode (§8.5) |
| F32 norms | F32 (some `+1`) | per §5.3 | existing norms, or a direct-gamma FP32 variant |
| `ssm_a` | F32 (−exp(A_log)) | FP32, role `ssm_a` | GDN A-direct variant (§8.4) |
| `ssm_conv1d`, `ssm_dt.bias` | F32 | BF16 if every value round-trips exactly, else FP32 | existing |
| `ple_conv1d` | F16 | FP32 (exact widening, the precedent for BF16 → FP32) | PLE conv FP32 variant |
| n-gram table | IQ4_NL | volume `iq4_nl_rows_v1` (§9.1) | `ple_embed` IQ4_NL variant |
| MTP, proposal head | — (Qwen checkpoint) | as recipes A–C (§10) | existing |
| Vision | `mmproj` BF16 | BF16 | existing (`--vision-offload`) |

### 5.2 Formats and plane layouts

Each new format is defined in [tensor formats](tensor-formats.md) with an **exact decode oracle**
(Python, `tools/artifact/codecs/ggml_*.py`) and an **exact inverse** to GGUF block bytes. The stored
layout splits each block into aligned planes so every kernel load is a vector load (GGUF blocks are
2-byte aligned: 18, 34, 82, 98, 110, 136, 144, 176, 210 bytes):

- per row, planes `codes` (16-byte aligned), `signs`/`qh` (where present), `sub_scales`,
  `d`/`dmin` (fp16), each contiguous over the row's blocks;
- rows of a matrix in tiles of 16 (`rg16`), the unit the SIMT routes read;
- Strata measured +10-12 % for this split on Q8_0 and bitwise equality on Q6_K (§3).

The converter writes planes from raw blocks; `verify_artifact` re-assembles the blocks and compares
bytes. No runtime repacking.

### 5.3 Name map and exact transforms

`tools/convert/sources/gguf.py` gains `qwen4exp_name_map` (all classes of §2.2) and a raw-block
reader (`GGUFSource.read_blocks`) that returns stored bytes, never dequantized values. Transforms,
each exact and checked by the converter:

- **Norm `+1`.** For each shifted F32 gamma, compute g − 1 exactly in FP64. If it is representable
  in the format the Infernix binding uses (BF16 for the HC/QSA/indexer/PLE norms) and round-trips,
  store that; otherwise store FP32 g − 1 with an FP32-gamma norm, or the GGUF value with a
  direct-gamma norm variant. Which tensors carry the shift is decided per tensor in M1 and checked
  against Qwen's BF16 values of the same tensors (§10.1 fetches them).
- **V-head order.** Row-wise tensors are permuted back to Hugging Face order. This is exact: whole
  rows, and every K-block lies within one row. `ssm_out`'s permutation is on its **input** axis at
  128-element head granularity, but Q4_K/Q5_K/Q6_K superblocks span 256 elements (two heads), so it
  **cannot** be undone exactly. `ssm_out` stays in GGUF order, and the GDN output stage writes head h
  to slot π(h) (§8.4).
- **Indexer split**: the q and k rows are concatenated into the QSA indexer parent.
- **F16, F32**: exact widening, or exact narrowing when every value round-trips (checked).

### 5.4 Expert records (`ggml_expert_v1`)

One record per expert, `record_bytes(layer)` a multiple of 4,096:

```text
[ gate/up rows, interleaved in 16-row groups (g0..g15, u0..u15, g16..), split into planes per group ]
[ down rows, in 16-row groups, split into planes ]
[ padding to 4 KiB ]
```

- Gate/up interleaving gives one pass over x for both and a fused SwiGLU, as `nvfp4_expert_rg16_v1`
  does (FN §6.2) and Strata's S2 pack does. Interleaving whole rows is a lossless re-layout.
- Planes per 16-row group put `codes` first (the bulk), then `signs`/`qh`, `sub_scales`, `d`, all
  16-byte aligned. Bank metadata records the per-layer format pair and plane offsets.
- Padding overhead is 0-0.2 % per record (§2.3). The bank tail keeps no multiplier plane: GGUF scales
  live in the blocks.
- The converter writes 32-expert chunks as `import_expert_bank` does (`tools/convert/qwen4_exp.py`).
  `verify_artifact` decodes every record back to the GGUF slices.

---

## 6. Arithmetic contracts

### 6.1 Expert arithmetic: exact up to one rounding

Reuse the A16 activation encoding of FN §16.2.1 unchanged: a BF16 column v (K = 2,560 for x, 640 for
h) becomes integers X_j (|X_j| < 2^21) with one column exponent emax; v_j = X_j · 2^(emax−147)
exactly for elements within 2^13 of the column maximum. X is stored as three byte limbs (lo, mid
unsigned; hi signed).

For every format, an element's weight is w = Σ_t d_t · κ_t · m_t · c, where c is a small integer code
(|c| ≤ 127), m_t an integer sub-scale, d_t an fp16 scale and κ_t a power of two fixed by the format
(t ranges over one term, or two for Q4_K/Q5_K mins). The canonical value of one output row is

```text
P_s   = Σ_{j ∈ sub-block s} c_j · X_j                         exact: per byte limb in int32, limbs combined in int64
Q_b   = Σ_{s ∈ superblock b} m_s · P_s                        exact int64
R     = Σ_b mant(d_b) · Q_b · 2^(exp(d_b) − e_min)             exact int128 (fp16 exponents span ≤ 30 binades)
y     = bf16_rn( fl32_rn(R) · 2^(e_min + emax − 147 + log2 κ) )   one rounding to FP32, one to BF16
```

- Bounds (model): IQ4_NL |c| ≤ 127 over 32 elements gives 127 · 32 · 2^21 < 2^33, so P is formed per
  byte limb in int32 and combined in int64 (as FN §16.2.1 does). IQ2_S/IQ3_* |c| ≤ 62 per 16/32;
  Q6_K |q − 32| ≤ 32 per 16; Q2_0 |c − 1| ≤ 2 per 64. Q4_K/Q5_K add Σ_s m_s · ΣX_s for the min term.
- **Any order.** Every sum before R's rounding is exact, so SIMT, IMMA and CPU routes may split K,
  rows and columns however they like and still return the same bits. Column count, placement and
  route never change a bit, which is FN's invariant and the reason GLM L13 adopted exact arithmetic.
- The int128 combine runs once per superblock per output (10 per row at K = 2,560), so its cost is
  negligible. On the CPU it is two `_addcarry_u64`/`_mulx` steps (MSVC has no `__int128`).
- SwiGLU, h encoding and the down product follow FN §16.2.1 exactly:
  `x (BF16) → A16 → gate/up → BF16 → SwiGLU → BF16 h → A16 → down → BF16 y_e → combine`.

**Oracle.** Exact rationals (Python `Fraction`) over the decoded GGUF values and the encoded X, then
the same two roundings. Production output must be **bit-equal** to the oracle, not just close (§12.1).

### 6.2 Dense arithmetic

Dense layers run only on the GPU, so they need column invariance (T ≤ 8 on SIMT routes, FN §19.2) but
not CPU equality.

- **Decode and verify (T ≤ 8): exact-integer SIMT, the same contract as §6.1.** The bytes per token
  are ~4.2 GB (§11.2); the integer work is ≤ 3 dp4a per 4 weights per column, below DRAM time at
  T ≤ 8 (model: ~1.3 ms of integer issue against ~2.7 ms of DRAM at T = 8). One contract for experts
  and dense also means one oracle family.
- **Prefill (tensor cores):** codes are exact in BF16 (|c| ≤ 127 < 2^8); a BF16 `mma.m16n8k16` per
  16-element sub-block, with the integer sub-scale and fp16 scale applied in an FP32 epilogue per
  k16 step. Qualified normwise against FP64 (FN's dense rule), not bit-equal to the SIMT route; this
  is the same split recipe B already has between its SIMT and tensor-core routes.
  - Option, gated: exact IMMA prefill with the §6.1 sums (three byte-limb MMAs per k32). It is chosen
    only if M5 measures it within 10 % of the BF16 route.
- `lm_head` (Q6_K): a `projection_fp32` variant (FP32 logits), tensor-core at 1..16 columns as recipe
  A's BF16 projection is (701-710 µs against a 709 µs DRAM floor).

### 6.3 PLE rows and embedding

- n-gram rows: `out = bf16_rn(fl32(d) · kvalue)`. The product is exact in FP32, then one rounding at
  the same BF16 boundary `ple_embed` has today (`include/infernix/ops/ple.h`).
- Embedding (IQ4_XS): decoded exactly to FP32 on the host. Whether the first consumer takes FP32 rows
  or a BF16 rounding is a semantic-boundary decision measured in M2 by KL (one extra rounding of
  2,560 values per token).

---

## 7. Routed experts

### 7.1 Op contract changes (`include/infernix/ops/offloaded_sparse_moe.h`)

- `MoeExpertSource` gains `ExpertFormat format` (`nvfp4_rg16` or `ggml{gate_up, down}`) and a
  per-call `RecordGeometry` (plane offsets, record bytes), replacing the global
  `kRecordBytes`/`kGateUpBytes` constants (`cpu/w4a4_expert.h:16-36`, `wide_expert.cuh:44-58`,
  `moe_layer.cu:390, 1763-1767`).
- `stage_kernel` copies `record_bytes` from the geometry. `launch_pass` and the wide `prepare`/`run`
  dispatch on the format.
- The NVFP4 family is unchanged. The ggml family is a second implementation behind the same entry
  points, as `model-quality.md` §4.4 anticipated.
- Kernels are templated on `<GateUpFormat, DownFormat, Columns>`. Seven format pairs × narrow
  (Columns 1..8) × wide × CPU.

### 7.2 GPU narrow route (decode, ≤ 8 columns)

One CTA per (expert, 16-row group), `gate_up` then `down`, PDL-chained, as `gate_up_kernel<Columns>`
and `down_kernel<Columns>` are today:

- **Weight decode once, apply to every column** (Strata's `Split::load`/`apply`), so the column loop
  never re-decodes.
- **Grids**: IQ2_S's 8 KB grid staged in shared memory per CTA; IQ3_S (2 KB) and IQ3_XXS (1 KB)
  chosen between shared memory and L1-cached global by microbenchmark (Strata: staging helped grouped
  kernels and hurt single-matrix ones). IQ4_NL values via the two-`__byte_perm` lookup in registers,
  never `__constant__` (divergent indices serialize: 2.12× slower, Strata).
- **Signs**: IQ2_S/IQ3_S explicit bits and IQ3_XXS parity bits through `__vcmpne4`/`__vsub4`
  (branch-free). Q2_0 symbols via `__byte_perm(…, 0x5140/0x7362)` (llama.cpp #25603).
- **Products**: dp4a of the decoded int8 codes against X's byte limbs (3 dp4a per 4 weights per
  column), int32 per sub-block, int64 per superblock, int128 combine (§6.1).
- **x staging**: X limbs for up to 8 columns in shared memory once per CTA (≤ 61 KB at 8 columns of
  2,560 × 3 bytes).
- **Layer-adaptive**: the format pair is fixed per layer, so each layer's launch is one template
  instance; the graph captures the right instance per layer.
- Budget (M3 gate): each kernel within 10 % of its byte-time at 1.55 TB/s for n = 1, 2, 4, 8 (FN's
  cost-gate rule, §16.2).

### 7.3 GPU wide route (prefill, > 8 columns)

- Integer tensor cores: `mma.sync.m16n8k32.s8.u8/s8` (k16 for per-16 sub-scale formats: IQ2_S, Q6_K),
  one MMA per X limb (3 per k-step), int32 per sub-block, int64/int128 combine.
- **Decode to an int8 code tile in shared memory, double-buffered**; X tiles via `cp.async` 4 stages
  and `ldmatrix` (Strata's `moe_fused_iq.cu` pattern). Grids in shared memory.
- Same exact sums as §7.2, so wide and narrow return the same bits.
- Cost (model): 3 IMMA per 32 K-elements per 8 columns, against 4 for FN's W4A16 wide route (2 MMAs
  per 16 elements with the `[8·c2 | c2]` packing). So it is no more expensive than the W4A16 route
  recipe C already runs.

### 7.4 CPU expert engine (i9-13900K: AVX2 + AVX-VNNI, no AVX-512)

- One engine per format pair under `cpu/ggml_expert.{h,cpp}`, built `/fp:precise`, with ISA variants
  scalar (oracle), AVX2 and AVX-VNNI. AVX-512 VNNI is optional for other machines.
- **Products**: `vpdpbusd` (AVX-VNNI) with unsigned X lo/mid bytes against signed codes, and the
  biased-code trick for the signed hi limb (FN §16.2.1). That gives the same P_s as the GPU.
- **Grid decode**: IQ2_S/IQ3_* via scalar index loads packed with `_mm256_set_epi64x` or
  `vpgatherdd`, chosen per core type at startup (Strata measured gather 0.34 → 0.19 ms on a Raptor
  Lake P-core); signs via `pshufb + and + cmpeq + vpsignb` (ggml `bit_selector`). IQ4_NL/IQ4_XS via
  `vpshufb` on `kvalues`; Q2_0 via shifts and `vpsubb`.
- **Multi-column**: unpack a 16-row group once into int8 codes in L1 (ik_llama.cpp #515), then
  `vpdpbusd` over all columns (Strata: 2.0-2.4× at 3 tokens).
- Team, phases, miss service and fetch channel unchanged (`cpu/expert_team.cpp`, `miss_service.cpp`).
  `CpuExpertJob` carries the record geometry. `cpu_rates.cpp` measures each format pair at startup
  (the decode-divisor input).

### 7.5 Residency with seven record sizes

Today the frame pool, staging slots, `ExpertStore`, host tier, SSD tier, VRAM sizing and the PCIe
divisor all use one `record_stride` (`program_impl.h:543-561, 750-792`, `load.cpp:540`,
`expert_store.cpp`, `host_expert_tier.cpp`, `qwen4_exp_instance.cpp:234`).

- **Per-layer geometry** everywhere a stride is read. The divisor `d = floor(1.65 + C·t)` uses that
  layer's `t = record_bytes/link`.
- **VRAM: size classes, one global LFRU.** The VMM arena is cut into one slab per record size
  (seven). Each slab's capacity is set at startup from the expert-state file (or the route-trace
  profile) in proportion to that class's byte-weighted demand. Eviction picks the lowest LFRU score
  within the needed class; at epoch boundaries a slab may cede whole 2 MiB granules to another class
  when the score gap exceeds a threshold. FN's rule that hard per-layer quotas lose (L21) concerns
  per-layer quotas; classes group 1-15 layers, and M0 replays three variants before freezing:
  - (a) one max-size frame (2,662,400 B), wasting 23 %;
  - (b) static classes;
  - (c) classes with rebalancing.
- **Host and SSD tiers**: records are contiguous per layer in the pinned arena and the artifact, each
  layer its own segment. The SSD tier reads in place (FN L12).
- **Staging**: 64 slots of the maximum record size (170 MB).
- `tools/expert_cache_replay/host_tier.py` replaces `RECORD_BYTES = 2_764_800` with per-layer sizes
  read from the artifact, and gains `--frame-model max|classes|rebalance`.

---

## 8. Dense path

### 8.1 Linear routes

New folders `src/ops/linear/{q4k,q5k,q6k,iq4xs,iq4nl}/`, each with `*_dispatch.cpp`, `*_shapes.h`
and `shapes/nN_kK.cu`, registered in `linear.cpp`'s `QType` switch (the FP8-row folder is the
template). The shapes that occur:

| Shape (n × k) | Classes | Formats |
|---|---|---|
| 10240 × 2560 | GDN `attn_qkv` | Q6_K, Q5_K, Q4_K, IQ4_XS |
| 6144 × 2560 | GDN `attn_gate` | Q6_K, Q4_K, IQ4_XS, Q5_K |
| 2560 × 6144 | `ssm_out`, `attn_output` | Q6_K, Q5_K, Q4_K |
| 12288 × 2560 | QSA `attn_q` (q + gate) | Q6_K, Q5_K, IQ4_XS, Q4_K |
| 512 × 2560 | `attn_k`, `attn_v` | Q6_K, IQ4_XS, Q5_K |
| 640 × 2560 | shared gate/up | Q4_K, IQ4_XS, Q6_K, Q5_K |
| 2560 × 640 | shared down | IQ4_NL (Q8_0 existing) |
| 248320 × 2560 | `lm_head` | Q6_K (`projection_fp32`) |

Each shape has a SIMT route family for T = 1..8 (column-invariant, exact §6.2) and a tensor-core
route for prefill, with tiles from a sweep bench (`bench/ops/ggml_flash_next_sweep.cu`, modelled on
`fp8_flash_next_sweep.cu`). Kernels use PDL with weight loads before `griddepcontrol.wait`
(already used for Flash-Next and Strata) and evict-first weight streams (FN L18).

### 8.2 Fusion across formats

Today the converter groups q/k/v/z (GDN), QSA q/gate/k/v/index and shared gate+up into parents and
splits rows after (`recipe.group`, `ops::split_rows`, `execution/forward.cpp`). In this file, members
of a group often differ in format (e.g. layer 0: `attn_qkv` Q6_K, `attn_gate` Q4_K).

- **Bring-up (M2)**: separate launches, one per member; up to ~84 more launches per token (model).
- **Optimized (M5)**: a **segmented linear**, one launch whose CTAs are assigned to segments by a
  prefix table, each segment with its own format and pointer, writing into one output with the same
  row offsets `split_rows` uses. Per-segment code paths are the per-format SIMT tiles of §8.1. A
  graph captures one segmented launch per group. Gate: adopted if the per-token time saved is
  ≥ 2 % at C = 1.
- Same-format groups stay single parents.

### 8.3 BF16 classes

HC mixers, routers, indexer, GDN α/β and PLE key/value use the existing BF16 routes. That includes
recipe A's fused hyper-connection mixer with the BF16 codec, measured ~18 % slower decode than
recipe B's 8-bit dense overall (FN §6.1). The HC mixers are ~1.26 GB per token in BF16 (model), the
one class where this artifact reads more than recipe B (§11.2).

### 8.4 GDN, QSA, norms

- **GDN A-direct**: the gate `g = A · softplus(a + dt_bias)` with A read from `ssm_a`. The kernel
  drops its `−exp(A_log)` (llama.cpp computes exactly this). A per-artifact role selects the variant.
- **GDN output order**: the gated-norm output store writes V head h to slot π(h) (π = ggml's tiled
  order, defined by `reorder_linear_attention_v_heads`). That is an index change in the store, at no
  cost.
- QSA, indexer and HC Ops are unchanged. Norm variants (FP32 gamma, direct gamma) are added only for
  the tensors §5.3 cannot narrow exactly.

### 8.5 Token embedding

Host-resident as today (FN §6.3). The engine worker gathers IQ4_XS rows (136 B per 256 values,
1,360 B per token) and decodes them exactly. This adds ~1 µs per token on the host (model), already
off the GPU's critical path.

---

## 9. PLE n-gram table: IQ4_NL rows and RAM residency

### 9.1 Volume

- New `NgramTableConfig.format = iq4_nl` with `row_bytes = 90` (5 blocks of fp16 d + 16 nibble
  bytes); 45 rows per 4 KiB block (4,050 B used, 1.1 % padding): 7,111,146 blocks, 29.13 GB. The
  converter copies shard 2's rows verbatim. The header carries the format and the volume id.
  `config.cpp:78-112` accepts both formats.
- Nibble order is llama.cpp's (elements 0-15 low nibbles, 16-31 high), decoded by an exact oracle
  and the kernel.
- `ple_embed` gains an IQ4_NL variant (§6.3); `ngram_scale` disappears for this format.
- The converter's `--ngram-reuse` works by volume id; `verify_artifact` compares every row with the
  GGUF.

### 9.2 Residency modes

`--ngram-residency auto|ram|volume` (CLI, serve, `include/infernix/types.h`):

- **volume** is today's path: unbuffered reads, a 2^20-row cache, gated verify reads (FN §12.3).
  It works with 90-byte rows unchanged apart from the geometry.
- **ram** holds the whole table in host memory:
  - The volume is read at load into a **locked** host arena with 16 overlapped 8 MiB unbuffered
    reads (the `ReadOnlyFile::read_direct` path used for experts): ~4-5 s from the Optane `G:` at
    ~7 GB/s (model).
  - `NgramVolume::read_rows` becomes a gather of 16 × 90 B per column. The 4 KiB padding is kept, so
    a row's address is `block · 4096 + (r mod 45) · 90`.
  - The arena is **pinned** (`cudaHostRegister`) because step 2 below needs it. Locked memory and
    pinned memory cost the same RAM.
- **auto** chooses ram when the ledger fits (§9.4) after experts are fully resident. Otherwise it
  uses volume, with a larger set-associative row cache sized from the remaining headroom
  (`--ngram-cache-mib`, default 160 MiB as today).

### 9.3 What RAM residency changes on the decode path

- **Step 1 (M6).** Rows are always available, so the host gathers them **before** every launch,
  including verify rounds. The anchor and copy-proposal columns are known before drafting; draft
  columns are gathered in ~1-2 µs after drafting. The gate machinery stays for draft columns but
  never waits on I/O.
  - What this removes: NVMe latency on decode misses (S4b), and, for prefill, the cold-row reads. S1
    measured these at ~1.46 s for a cold 16K prompt (FN §12.3, 2026-10-05). How much of that is still
    exposed after the later prefill work is measured in M0; it is not assumed.
- **Step 2 (option, M8 gate).** Device-side PLE:
  - The `ngram_row_ids` hash (exact int64, FN §16.1) runs on the GPU;
  - `ple_embed` reads the 16 rows of each column from the pinned table over UVA: 1,440 B per
    column, latency-bound at ~1-2 µs within layer 0.
  - This removes the host from the PLE path and lets the verify graph launch without waiting for the
    drafter's rows. It is FN's planned "device-driven PLE" (M6 there).
  - Zero-copy is wrong for bulk transfers (FN L3) but right for a few cache lines per column.
  - Gate: adopted when S0-style events show ≥ 0.1 ms of after-draft GPU idle per verify round that
    step 2 removes.

### 9.4 Host RAM ledger

95.8 GiB visible (measured, GLM §3.1). Model:

| Item | Volume mode | RAM mode |
|---|---:|---:|
| Expert records, pinned (24,576 × 2.04 MB with padding) | 46.6 GiB | 46.6 GiB |
| n-gram table | 0.16 GiB (row cache, pageable) | 26.8 GiB (pinned) |
| Embedding (IQ4_XS) | 0.31 GiB | 0.31 GiB |
| Vision tower (`--vision-offload`) | 0.84 GiB | 0.84 GiB |
| Prefix-cache Host tier (default) | 4.0 GiB | 4.0 GiB |
| Program allowance, lock overhead, load staging | ~0.6 GiB | ~0.6 GiB |
| Headroom (`--ram-headroom-mib` default) | 2.0 GiB | 2.0 GiB |
| **Total** | **~54.5 GiB** | **~81.2 GiB** |

Recipe B needs 69.3 GiB today (GLM §3.1). The RAM mode needs ~12 GiB more than recipe B. It fits a
quiet machine (95.8 GiB visible). It does **not** fit the 79 GiB available measured with other
sessions running (GLM §3.1), unless the prefix tier shrinks to 1-2 GiB or vision offload is off.

`HostMemoryDemand` (`models/qwen4_exp/memory_plan.h`) gains an `ngram_table` term placed after
experts. `auto` never displaces experts to the SSD tier to make room for the table. The pinned ceiling
on Windows is measured first (GLM R0: 67 % pinned works today; this plan pins ~78.5 GiB, 82 %,
counting the prefix Host tier).

---

## 10. MTP drafter from Qwen's checkpoint

### 10.1 Source

- No local checkpoint has the MTP head: the NVIDIA checkpoint the current recipes read is gone, and
  no BF16/FP8 Qwen checkpoint is local (measured).
- Fetch only what is needed from **`Qwen/Qwen3.8-Flash-Next-FP8`**, pinned to one revision with
  SHA256 per object:
  - the 31 dense `mtp.*` tensors (BF16);
  - the 512 MTP experts `mtp.layers.0.mlp.experts.{e}.{gate,up,down}_proj.weight` plus
    `weight_scale_inv` (block FP8, 2.52 GB; upstream index). These are the tensors and the form the
    existing converter already decodes exactly from NVIDIA's checkpoint (`fp8_block_values`,
    `dequantize_fp8_block128`, FN §6.1);
  - for §5.3's checks, the BF16 dense tensors whose GGUF counterparts carry transforms (norms,
    `A_log`, conv). Total ~3-6 GB.
- The fetch is a small tool, `tools/convert/fetch_tensors.py`: it reads safetensors headers, then
  HTTP range-reads the named tensors into a local partial checkpoint directory with `config.json`.
  Strata's `tools/mtp_fetch.py` does this from the BF16 repository (5.2 GB).
- Alternative source: Strata's local `E:\Strata\Strata-data\mtp\tensors\*.bin` (BF16, from the BF16
  repository at a pinned revision). Used only if the fetch is unavailable; its provenance is
  recorded.

### 10.2 Conversion and runtime

- The builder accepts a **drafter source** (`--drafter-source DIR`) separate from the base. Today
  `_Qwen4ExpBuilder.validate_source` refuses alternate sources (`tools/convert/qwen4_exp.py:320`);
  this narrows that refusal to the base model.
- The drafter recipe is `_flash_next_drafter` unchanged (`official_recipes.py:278`): experts
  `q4_g64_fp16` (`grouped_mse`, 1.34 GB resident), dense `q8_g32_fp16`, router and shared gate BF16.
- **Proposal head**: the 131,072 most frequent `lm_head` rows, `q4_g64_fp16`. They are taken from the
  GGUF's Q6_K `output` decoded exactly, so the draft head approximates this target's head, not
  Qwen's BF16 one.
- MTP shares the token embedding and `lm_head` (`mtp_use_dedicated_embeddings: false`). It uses the
  IQ4_XS embedding gather and the Q6_K head at runtime.
- Draft policy unchanged (`choose_draft_length`, FN §11.3).
- Expected acceptance (estimated): Strata measured 0.85-0.89 step-1..3 offline greedy acceptance
  with a 2.25-bit MTP drafter against this exact IQ3_S target. FN measured 75.6-81.8 % sampled on
  AIME with q4 experts against NVFP4. M7 measures it here.

---

## 11. Performance model

### 11.1 VRAM frames

Recipe B holds ~9,400 frames of 2.76 MB (~26.0 GB). This artifact's GPU dense bytes are ~4.2 GB
against recipe B's ~4.4 GB (§11.2), so ~26.2 GB is available for experts (model):

| Frame model | Frames | Share of expert bytes |
|---|---:|---:|
| Ideal byte packing | ~12,870 | 52 % |
| (b)/(c) size classes, 2-5 % fragmentation | ~12,250-12,600 | 50-51 % |
| (a) one max-size frame | ~9,840 | 40 % |

For comparison, Strata reports 10,733-12,331 slots on its 5090 runs (upstream).

### 11.2 Dense bytes per token (model, from §2.2 shapes and bits per weight)

| Class | This artifact | Recipe B |
|---|---:|---:|
| GDN projections (36 layers) | 1.58 GB | ~2.07 GB (8-bit) |
| QSA projections and indexer (12) | 0.48 GB | ~0.60 GB |
| Shared experts (48) | 0.14 GB | ~0.24 GB |
| Routers (BF16) | 0.13 GB | 0.13 GB |
| HC mixers | **1.27 GB (BF16)** | ~0.67 GB (Q8) |
| lm_head | 0.52 GB (Q6_K) | ~0.67 GB (Q8) |
| **Total** | **~4.2 GB** | **~4.4 GB** |

### 11.3 Decode (C = 1)

- LFRU replay on Flash-Next (FN §9.3) gives miss rates of 18.4 / 5.2 / 1.4 % of activations at
  16.3 / 32.6 / 48.8 % capacity. At ~50 % capacity against recipe B's ~38 %, the replay model
  predicts **about one third of recipe B's misses**, each 26 % smaller. Live cold-start rates are
  higher than replay (recipe B tg512 hit rate 88 % measured), but the ratio is the claim, and M0
  replays it on real traces with real sizes.
- Hit-expert bytes: 480 × 2.04 MB = 0.98 GB per token, against 1.33 GB (−0.2 ms at 1.55 TB/s).
- The dense path is unchanged within ±5 %, provided the k-quant/i-quant SIMT routes reach the Q8/FP8
  routes' DRAM efficiency. **This is the largest risk** (§15.1).
- **Estimate**: plain decode 135-160 tok/s against 132.2. MTP gains as for recipe B, since verify
  windows add no misses per accepted token (FN L8).
- The CPU miss rate per byte for i-quants is lower than W4A4's (grid decode); with ~1/3 of the misses
  it matters less. It is measured by `cpu_rates.cpp`.

### 11.4 Prefill

- **Link**: every expert not in VRAM crosses PCIe once per walk span. That is ~0.49 × 50 GB = 24.5 GB
  at 27.6 GB/s = 0.9 s per span, against ~1.5 s for recipe B (model).
- **Compute**: exact IMMA expert GEMMs at ≤ the W4A16 wide-route cost (§7.3). Dense prefill on BF16
  tensor cores with an FP32 epilogue per k16 (+~25 % FMA work against a pure BF16 GEMM, model).
- **Estimate**: pp131072 within ±10 % of recipe B's 5,767 tok/s. Short prompts (one walk span) gain
  the most, since streaming is ~40 % shorter.
- **RAM-resident n-gram table**: removes the cold-row reads of each prompt (up to ~1.46 s for 16K
  tokens at S1's rates; current exposure measured in M0).

---

## 12. Qualification

Per [Op development](op-development.md) §6; claims follow [model quality](model-quality.md).

### 12.1 Codecs and Ops

| Item | Oracle | Criterion | Coverage |
|---|---|---|---|
| Each `ggml_*` format | Exact decode in Python (`Fraction`/FP64) from the GGUF block bytes; cross-checked against `gguf-py` `dequantize` (no Q2_0 there: independent oracle) | Exact, every element of every tensor in the artifact; round-trip to GGUF bytes | All 1,223 tensors |
| `offloaded_sparse_moe` ggml family | §6.1 exact rationals, two roundings | **Bit-equal**, every output | 7 format pairs × columns 1..8, 9, 16, 476 × routes (narrow, staged, wide, CPU scalar/AVX2/AVX-VNNI) × random residency mixes; real geometry; non-finite columns |
| Linear (new formats) | FP64 over the exact decoded weights | SIMT: bit-equal to §6.2's exact contract; tensor-core: normwise with gross pointwise cap (FN rule) | Every (format, shape) of §8.1; T = 1..8 boundaries ± 1, prefill tiles |
| Segmented linear | Composition of per-segment oracles | Bit-equal to separate launches | Every group of §8.2 |
| `ple_embed` IQ4_NL | Exact decode | Bit-equal | Random rows + boundary rows |
| `NgramVolume` RAM backend | Volume backend | Byte-equal rows; counters | Closed-form row oracle (existing test) |
| GDN A-direct, output permutation | FP64 / exact index map | Normwise / exact | Real shapes |

### 12.2 Model and system

- `verify_artifact` (§1.3) over the whole artifact and volume.
- Greedy invariance: all-GPU vs all-CPU vs random placements, C = 1..8, MTP on/off
  (`test_forward_real`, `vram_pressure_real`, `expert_fault_real` with the new artifact).
- Quality: the §16.5 KL protocol against recipe B; llama.cpp b10221 (local binary) on the same GGUF
  as an external sanity check (top-1 agreement and KL, not bit-equality, because of Q8_1).
- Performance: tg512 ABBA, the agentic A/B (`bench/agentic_ab`), pp131072, a cold 16K prompt with
  and without RAM residency, and Strata on this machine with the same file, all against recipe B.
  Kernel changes are judged by per-round kernel time, not greedy throughput (FN L20).

---

## 13. Milestones

Estimates are focused engineering time, ±50 %.

| M | Content | Exit gate | Est. |
|---|---|---|---|
| **M0** facts and models | Download shard 1 (pinned, SHA256); full header dump; confirm §2. Strata on this machine (tail skip on/off), recipe B re-measured the same day. Replay with per-layer sizes (`host_tier.py`), three frame models (§7.5). Current exposure of n-gram reads in cold prefill. Windows pinned ceiling at ~78.5 GiB. | §11 table replaced by measurements; frame model chosen | 3 d |
| **M1** formats and converter | `ggml_*` codecs and planes; raw-block GGUF reader; `qwen4exp` name map and transforms; `ggml_expert_v1` records; IQ4_NL volume; `verify_artifact` extension | Bit-exact verify of the whole artifact and volume | 5-7 d |
| **M2** correctness bring-up | Correctness-first kernels: SIMT dense for every (format, shape), generic narrow expert kernel, scalar CPU engine; GDN A-direct and permutation; IQ4_NL `ple_embed`; IQ4_XS embedding; per-layer strides with one max-size frame | Model runs end to end; greedy invariance; KL measured (first quality read: stop here if KL is unacceptable) | 6-8 d |
| **M3** expert kernels | Tuned narrow (per format, smem grids), wide IMMA, AVX2/AVX-VNNI CPU engines; `cpu_rates` per format | Bit-equality matrix (§12.1); each kernel ≤ 110 % of byte-time | 8-10 d |
| **M4** residency | Size-class frames (per M0), per-layer stride in store/host/SSD tiers, divisor, staging, replay tool, expert state | Tier tests pass; replay ≈ live within FN's tolerance | 4-5 d |
| **M5** dense performance | Tuned SIMT and tensor-core tiles per shape; segmented linear; Q6_K `lm_head` projection; PDL chain audit | Decode ≥ recipe B, or attributed by nsys/ncu | 5-7 d |
| **M6** n-gram RAM residency | `--ngram-residency`, ledger term, RAM backend, pre-launch gather | RAM = volume rows; cold 16K prompt measured | 2-3 d |
| **M7** MTP | `fetch_tensors.py`, `--drafter-source`, drafter and proposal head; acceptance and draft-policy check | MTP greedy = plain; acceptance and tok/s reported | 2-3 d |
| **M8** options and vision | Device-side PLE (gated), exact IMMA dense prefill (gated), `mmproj` vision import | Each option by its gate | 3-5 d |
| **M9** documentation | §14; model card | Links and `git diff --check` | 1 d |

Total ≈ 40-55 days. The sequence front-loads the quality read (M2) before kernel investment.

---

## 14. Documentation and authority changes

- [tensor formats](tensor-formats.md): the `ggml_*` formats, plane layouts, `ggml_expert_v1`.
- [Flash-Next design](qwen3_8-flash-next-design.md): the ggml expert family (§16.2.2: the exact
  arithmetic of §6.1), size-class frames, n-gram residency modes, IQ4_NL volume.
- [weight conversion](../weight-conversion.md) and [model quality](model-quality.md) §4.4 (replace
  "neither path is built").
- User guide `docs/qwen3_8-flash-next.md`, `docs/cli.md`, `docs/serving.md`: `--ngram-residency`,
  `--ngram-cache-mib`, the new artifact.
- `model-cards/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix/`: provenance (ISTA revision, Qwen MTP
  revision), the license (Apache-2.0), ISTA's GSQ/RCO and imatrix credits.
- `docs/README.md` row for this plan, removed with it.

---

## 15. Risks, options and rejected alternatives

### 15.1 Risks

| Risk | Effect | Mitigation |
|---|---|---|
| k-quant/i-quant SIMT dense routes below Q8/FP8 DRAM efficiency | Decode below recipe B even with fewer misses | Plane layouts (§5.2), per-shape tiles, M5 attribution; segmented launches |
| CPU grid-decode cost (IQ2_S/IQ3_*) | Lower CPU miss rate; divisor shifts to PCIe | Per-core gather choice, unpack-once multi-column; the divisor already adapts to measured rates |
| Size-class fragmentation | Fewer effective frames | M0 replay picks the frame model; fall back to (a) |
| Windows pinned ceiling below ~78.5 GiB | RAM mode unavailable | `auto` falls back to volume plus a larger row cache |
| MTP acceptance lower against this target | Smaller MTP gain | Measured in M7; draft policy adapts |
| GSQ-RCO quality below recipe B | Artifact not recommended | M2 quality read before kernel work |

### 15.2 Options (each behind its own gate)

- `IQ3_S-dense8` variant: the HC mixers, PLE key/value and router re-quantized to recipe B's 8-bit
  classes (−0.6 GB per token). Not bit-exact, so it is a separate recipe name, qualified by §16.5.
- Exact IMMA dense prefill (§6.2).
- Device-side PLE (§9.3).
- A route-tail skip à la Strata is **not** an option: it changes outputs.

### 15.3 Rejected

- **Loading the GGUF in place**: `.infernix` is the only C++ artifact and runtime repacking is
  forbidden (AGENTS). The blocks' 2-byte alignment would also slow every kernel.
- **Q8_1/Q8_K activations** (llama.cpp, Strata): less accurate than A16, and Strata shows the two
  forms disagree across placements.
- **Dequantizing to FP32 and re-quantizing** (the current `gguf.py` route): a second quantization,
  not bit-exact.
- **One max-size frame without measurement**: 23 % fewer frames (§11.1).
- **Expert deferral, SER-style expert dropping**: change the mathematics.

---

## 16. Sources

- ISTA-DASLab GGUF repository and `tensor-allocation/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.rco-allocation.txt` (read 2026-10-10).
- llama.cpp `3cf03257` (vendored in `E:\Strata\Strata\third_party\llama.cpp`): `ggml-common.h`,
  `ggml-cuda/{vecdotq.cuh,mmq.cuh,dequantize.cuh}`, `conversion/{qwen.py,qwen4exp.py}`,
  `src/models/qwen4exp.cpp`; PRs #24448, #25603, #25707, #26348, #27742, #27794, #29887.
- Strata `E:\Strata\Strata` (HEAD `61b3fb5d`): `tools/iq_pack.py`, `src/kernels/cuda/iq_kernels.cu`,
  `moe_fused_iq.cu`, `native_mmvq.cu`, `src/kernels/cpu/iq_avx2.cpp`, `src/ngram/ple_reader.cpp`,
  `tools/mtp_fetch.py`, `bench/results/`.
- ik_llama.cpp discussions #164, PRs #515, #1446, #1467, #1474, #1482.
- Qwen/Qwen3.8-Flash-Next-FP8 `config.json` and `model.safetensors.index.json` (read 2026-10-10).
