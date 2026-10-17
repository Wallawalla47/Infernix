# Qwen3.8-Flash-Next official FP8: implementation plan

**Outcome (2026-10-17): implemented as one recipe.** `qwen3_8_flash_next_fp8` is F-A (every
checkpoint tensor bit-exact, the BF16 n-gram table) with `q2_g64_fp16` drafter experts; F-B
(Dense8) and its recipe were dropped on 2026-10-12, so the F-B entries below are history. The
current figures are in the [user guide](../qwen3_8-flash-next.md#official-fp8-checkpoint).

**Status (2026-10-11): partly implemented** in worktree `iq3s-gsq-rco` (not committed):

- **M0, source.** The checkpoint is downloaded at the pinned revision to `C:\models\Qwen3.8-Flash-Next-FP8`
  (all 131 shards, 185.6 GB, every shard's SHA256 equal to the LFS hash). All 48 x 512 routed
  experts are E4M3FN `[640, 2560]` / `[2560, 640]` codes with BF16 `[5, 20]` / `[20, 5]` tile scales;
  every scale word is positive, normal and finite; the per-row exponent spread is at most 4
  (0: 128,629 rows, 1: 542,690, 2: 60,560, 3: 5,302, 4: 99); 629 M sampled gate codes hold no NaN.
  Route-trace replays of the FP8 capacities are not done.
- **M1.** `fp8_e4m3fn_b128` in `fp8_expert_b128_v1` (Python codec, layout, C++ registration and
  geometry), the converter's bank import and recipes F-A/F-B, `verify_artifact`'s bank check.
  Codec exact against PyTorch on every code; two real MTP experts round-trip byte for byte.
- **M2/M3.** The exact arithmetic (`src/ops/fp8_block/fp8_exact.h`, sharing
  `src/ops/common/exact_combine.h` with the GGUF route), GPU narrow kernels for every column count
  (`dp4a` over three base-64 weight digits), the CPU route (scalar, AVX2, AVX-VNNI, AVX-512 VNNI)
  through the GGUF route's team, and loader support (`exact_experts`). Op tests pass:
  `infernix_offloaded_moe_fp8_test` (every width 1-300 and placement, CPU-served included, bit-equal
  to the scalar reference, itself equal to a bignum sum) and the FP8 case of
  `infernix_offloaded_moe_ggml_cpu_test` (every ISA; 1.0 ms per expert on one AVX-VNNI thread).
- **F-B artifact.** `E:\Infernix-staging\Qwen3.8-Flash-Next-FP8-Dense8-Infernix-00001-of-00005.infernix` (129.8 GB,
  five files; routed experts 112.6 GiB in 4.69 MiB records). `verify_artifact` (121 s): 24,576
  experts and 987 tensors equal to the checkpoint, 629 text and 1,556 MTP tensors re-quantized by
  the recipe. It runs with recipe B's n-gram volume: its header and volume id match the
  artifact's table and 50,000 sampled rows (`verify_ngram`, 7.8 s) equal this checkpoint's table;
  the full 320M-row comparison is pending.
- **Windows pinned ceiling (M0).** WDDM counts every CUDA pin of a process in the GPU's NON_LOCAL
  segment and refuses pins just past its per-process budget: 75.86 GiB budget, pins failed at
  76.07 GiB of usage with 5.3 GiB of RAM still free (both `cudaHostAlloc` and `cudaHostRegister`).
  The RAM ledger now keeps every pin within the budget less the usage (FN §19.3.7, R4 "Pin room");
  before that, F-B's automatic 77 GiB of expert slots failed to pin at startup. F-B now pins 73.9 GiB
  of expert slots (15,700 experts, 64 % of the bank) by default, more than §8.5's 64 GiB.
- **First measurements** (F-B, default options, INT8 KV, one request, the expert state of earlier
  runs; RTX 5090 on PCIe 5.0 x8, i9-13900K):

  | `infernix_bench` | F-B | Model (§10) | Recipe B, same day |
  |---|---:|---:|---:|
  | tg512 plain | 41.1 tok/s | ~79 central, ~62 pessimistic | 120.5 |
  | tg512 MTP 4, `--lm-head-draft` | 50.3 tok/s (76 % accepted) | ~95 central | 176.3 |
  | pp4096, `--prefill-chunk 4096` | 448 tok/s | — | 1,679 |

  Decode is below the pessimistic model. The VRAM cache holds 5,303 frames (§8.1 modelled 5,290)
  but hits 66 % of routed experts in tg512 against the central 82 %, and about 149 misses per token
  go to the CPU engine (~76K per 512 tokens), against the 82 modelled. With 1.0 ms per expert per
  thread (4.9 GB/s), the CPU engine at ~150 experts per token is the decode's critical path; M5's
  divisor and `cpu_rates` (still calibrated for 2.6 MB NVFP4 and ~2 MB GGUF records) are next.
  Prefill reads about 38 GiB from `E:` per 4K-token prompt (~4.3 GB/s; 8.1-8.6K reads of 64 ms
  mean latency), so it is bound by the single volume: M4 (striped volumes, S3) is the lever.
  Not done: F-A conversion (disk), greedy invariance on F-B, the KL protocol, pp131072, route-trace
  replays.
- **M4/S3, second drive (built, not yet measured).** In place of §8.3's striped copies in 1.1 MiB
  units, an *expert stripe* holds only the tail of every record on a second drive
  (`tools/flash_next/expert_stripe.py`, format in `expert_store.h`, `--expert-stripe` on every
  CLI): the tier reads the head from the artifact and the tail from the stripe at once, at depth 8
  per drive. No second copy of the experts (56 GiB for F-B's tails on `D:` instead of 113 GiB on
  two drives) and no new read engine: an `ExpertStore` record takes a third segment. Records split
  this way are never read as runs of neighbours (the heads are not contiguous). Tests: the tier's
  synthetic stripe (straddling records, refusals) and the writer against the artifact's bytes.
  Measurement (pp4096, tg512, the split fraction) waits for the GPU and for `D:` to be free.
- **BF16 n-gram table (built, not yet run; user request).** `NgramRowFormat::Bf16`: a volume of the
  checkpoint's BF16 rows (`tools/flash_next/ngram_bf16.py`, 320-byte rows, ~109 GB) replaces any
  artifact's FP8 or IQ4_NL table through `--ngram-volume`; `resolve_ngram_volume` reads the format
  from the row size in the header (no format change), the Program sizes its n-gram buffers from the
  effective table, and the PLE reads BF16 rows in place as its embedding. The prefix-cache file's
  fingerprint names the table. Qualification queued: a BF16 volume of the FP8 table's own values
  (`--dequantize`) must give recipe B's FP8-volume greedy text and perplexity bit for bit. The real
  BF16 volume is built: `D:\Infernix-staging\Qwen3.8-Flash-Next-BF16.ngram` (109.2 GB, 205 s) from
  the local Swift 1.5 NVFP4 export, whose 128 n-gram shards are byte-identical to Qwen's BF16 table
  (no download); 1,000,000 sampled rows verified. Its KL against recipe B's FP8 table is queued.
- **Causal scoring for Qwen4Exp (built, not yet run).** The KL protocol needs teacher-forced
  distributions of the FP8 artifact, which only runs in the SSD tier; the earlier route
  (`infernix_qwen4_exp_forward_real_test --dump-logits`) pins every expert. The Engine's
  CausalScoring now runs on Qwen4Exp (`program/score.cpp`): the prompt prefills on lane 0 through
  the generation path (tier, streaming, layer walks), every call keeping its final residuals
  (`ForwardBatch::residual_out`, a `[S*H, max_context]` buffer in the device plan), then
  `Forward::head_logits` (final mixer and LM head) runs in 64-column FP32 tiles, with
  `ops::target_logprobs` / `ops::top_logprobs` now accepting FP32 logits (oracle tests added).
  `infernix-perplexity` gains `--ngram-volume` and `--expert-stripe`. Qualification:
  `infernix_qwen4_exp_score_real_test` against the Forward's dump of the same ids (IQ3_S, every
  expert in RAM and through the tier), then the KL runs (recipe B's top-32 as the reference).
  **Qualified (2026-10-11):** IQ3_S chat ids, 1,023 positions, RAM and tier 8 GiB alike: perplexity
  3.6877 against the reference's 3.6877, |Δ logprob| mean 0.00005 (max 0.0035), top-1 agreement
  100 %, top-8 ids equal at all 428 separated positions. (The first run failed: `causal_score`
  cleared its scoring flag before the head tiles, so they read an empty residual view; fixed.)
- **KL protocol, first results** (quick 1M corpus, 261,167 tokens, context 4096 / stride 2048, INT8
  KV, chunk 4096; reference recipe B's top-32): recipe B perplexity 3.5077 (756 tok/s); F-B
  perplexity 3.4863, mean KL 0.0488, top-1 agreement 92.2 % (195 tok/s through the tier); GSQ-RCO
  IQ3_S perplexity 3.5223, mean KL 0.0905, top-1 agreement 89.3 % (1,083 tok/s). F-B is lower than
  recipe B in every domain (Chinese 4.348 / 4.386, English long-form 6.583 / 6.600, English
  reference 3.086 / 3.118, code 1.661 / 1.666). With the BF16 n-gram table (`--ngram-volume`, same
  reference): recipe B 3.5094, KL 0.0421, top-1 92.6 %; F-B 3.4863, KL 0.0490; IQ3_S 3.5198, KL
  0.0893. The BF16 table moves recipe B's distribution about as far as F-B's experts do, at the
  same perplexity. The BF16 route is exact (bf16q, 2026-10-12): recipe B with its FP8 volume and
  with a BF16 volume holding that table's dequantized values gives the same 400-token greedy text and
  the same mean NLL to every printed digit (1.5817995854760287 over 10,325 tokens), so the KL move is
  the BF16 table's values, not the route.
- **Tier-measured CPU rate (built, measured: no gain).** On F-B the team measured 6,682 experts/s on
  the tier's RAM copies (16 workers), so decode keeps 1/2 of the misses on the link (5 before).
  tg512 40.6 tok/s (41.1 before), MTP 42.8 (50.3 before, −15 %), pp4096 448 (448). The MTP
  regression is unexplained; the earlier divisor stays the better measured setting.
- **Speed work (2026-10-12, user request; built and measured).** On F-A (INT8 KV, BF16 table, 3
  runs, against the F-A run below with the earlier binaries): tg512 26.4 -> 38.3 tok/s (+45 %), MTP 39.0 -> 46.0 (+18 %, 78 % accepted),
  pp4096 449 -> 646 (+44 %). The CPU team's tier-measured rate is 12,545 experts/s at 16 workers
  (6,682 before, on F-B); the Op bench's team rates are 14,790 at 16 workers and 15,970 at 24.
  Exactness: `infernix_offloaded_moe_ggml_cpu_test` (every ISA against the scalar oracle),
  `infernix_offloaded_moe_fp8_test` and the tier's `--fetch` test (read-ahead: 20 reads for 20
  records, an unadopted guess released) pass. Not separated: how much of pp4096's gain is the
  read-ahead against the CPU route. F-B was not re-measured: its files are no longer on `E:`.
- **Long-prompt prefill (2026-10-12, measured).** A 32K prompt's debug log showed the layer walk
  streaming the 15,647 RAM-resident experts once (71.7 GiB) while the 6,686 SSD-only experts were
  fetched again by every 4,096-token call: 57,019 SSD reads (261 GiB) for one prompt, 784 tok/s.
  Two changes:
  - *SSD-only experts in the walk's stream* (`ExpertStream::SsdReads`, `HostExpertTier::stream_read`):
    a span's stream plans them into ring slots too (at most a quarter of the stream slots per
    layer), the tier reads each into a stream slot two layers before its copy, and the slot comes
    back once the copy has landed; the MoE reads them from the ring like any streamed record. The
    slots (640, at most 1/8 of the tier's) are lent from the RAM residents for the span
    (`HostTier::lend`, the lowest-ranked leave RAM) and returned at the next boundary outside a
    walk, where admissions refill them. A fixed pool instead measured F-A tg512 38.0 -> 35.0
    (-8 %, fewer RAM residents) and was replaced. Single calls keep the fetch channel (streaming
    them measured pp4096 649 -> 563: the host waited for the drive while enqueueing).
  - *FP8 tensor-core tiles* for prefill jobs of more than 8 columns (`fp8_expert.cuh`, `fp8t`):
    nine int8 MMAs per digit x limb plane, each accumulating over a 128-element scale tile, the
    tile's P added with `fp8b128::add_tile`: the same integers. Op bench (512 experts, top-10, one
    4,096-token chunk, all resident): 56.1 -> 13.9 ms per layer call (4.0x); bit-equal at 40 and
    300 columns (`infernix_offloaded_moe_fp8_test`).
  Measured (F-A, INT8 KV, BF16 table): 32K prompt 784 -> 2,360 tok/s (3.0x; SSD reads 54,602 ->
  4,654 plus 6,738 stream reads), the same 32 greedy tokens; pp4096 649 -> 658; tg512 38.0 -> 37.7.
  Bounds from the measurements above: decode misses ~163 experts per token at 66 % VRAM hits, each
  reading 4.9 MB of DRAM whether the CPU computes it or the link copies it, so DDR5-5800 caps plain
  decode near 60 tok/s at this hit rate; pp4096 reads ~38 GiB from one 990 PRO at ~4.3 GB/s of its
  ~7. Changes: (1) the CPU route's weight decode (`fp8_expert.cpp`): w = A + 2^8 B by exponent, each
  a signed 4-bit mantissa times a power of two at most 128 from two `vpshufb` tables, multiplied by
  `vpmaddubsw` on even and odd bytes (columns split to match): about 17 instructions per 32 codes
  instead of ~68, the same integers; (2) the GPU kernels' digits from a 256-entry shared table and
  byte permutes (~11 instructions per code word instead of ~60), decoded once per slice for all of a
  pass's columns (they were decoded per column); (3) SSD read-ahead in the tier's agent
  (`HostExpertTier::Options::ahead_min`, `ahead`): a fetch request of 64 or more records queues up to
  32 of the next layer's records without a host copy behind its own, adopted by the next request.
  Not done: an IMMA wide route (the GPU's ~0.8 s per 4K chunk, estimated from the dp4a count, is far
  below the I/O's measured ~9 s; the whole pp4096 now takes ~6.3 s); the read depth (8 per drive).
- **User decision (2026-10-12): the published FP8 artifact is F-A with the BF16 n-gram table**, and
  its repository is to host the BF16 volume for every Flash-Next artifact's model card. The artifact
  format gains `ngram_table.format = "bf16"` (320-byte rows, no `ngram_scale`; `config.cpp`), and
  `--ngram-reuse` binds a BF16 volume to the FP8 checkpoint when its sampled rows quantize to the
  checkpoint's FP8 codes (ModelOpt's rule with the checkpoint's scale; `NgramTable.bf16_rows_match`).
  Checked on the real files: 512 + 20,000 rows, no mismatch. No dedicated F-A against F-B study:
  the Dense8 classes' cost was measured on recipe B (FN §6.1).
- **F-A artifact (2026-10-12).** `E:\Infernix-staging\Qwen3.8-Flash-Next-FP8-Infernix-00001-of-00005.infernix`
  (133.3 GB, five files; conversion 1,627 s on the CPU), bound to
  `D:\Infernix-staging\Qwen3.8-Flash-Next-BF16.ngram`. `verify_artifact` (301 s): 24,576 experts and
  1,615 tensors equal to the checkpoint, 1,556 MTP tensors re-quantized, and all 320,001,536 n-gram
  rows of the BF16 volume quantize to the checkpoint's FP8 codes. Measured (INT8 KV, one request,
  BF16 table; F-B's figures are with the FP8 table): tg512 26.4 tok/s (F-B 40.6), MTP 39.0 (42.8,
  78 % accepted), pp4096 449 (448); perplexity 3.4870, KL 0.0496, top-1 92.1 % (F-B with the same
  BF16 table: 3.4863, 0.0490). Plain decode is 35 % below F-B, more than §10's ~20 % model; the
  BF16 dense reads and the smaller frame count are the modelled causes, not yet separated from the
  table's reads. With the speed work's binaries: a two-drive stripe of F-A's experts
  (`expert_stripe`, a 60.5 GB tail on `D:`, verified with no mismatch, removed afterwards) gives pp4096 743 tok/s against
  588 unstriped in the same run (+26 %; the earlier unstriped run gave 646, so the run-to-run
  spread is large) and tg512 40.5. Greedy invariance on F-A: the default tier, 24 GiB and 8 GiB of
  expert RAM, and 8 GiB with the stripe generate the same 400 tokens.

Changes against this plan: the record keeps gate, up and down rows row-major (not gate/up
interleaved in 16-row groups), so rows are addressed as the GGUF banks' are; the scale-exponent
spread limit is 62 (the exact sum's shift bound), not 64; outputs round once to binary32 at the
result's binade (`exact::to_f32_scaled_rn`), equal to §6.1's form whenever the result is normal. No
wide IMMA route yet: the narrow kernels' estimated cost per 4,096-token chunk is far below the
expert stream's, so it waits for a measurement (M3 gate).

This is a temporary planning document for an Infernix artifact of
[Qwen/Qwen3.8-Flash-Next-FP8](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8) that imports the
official block-FP8 routed experts bit-exactly and runs them with exact arithmetic on one RTX 5090.
When the work lands, its stable content moves into the authorities named in
[§13](#13-documentation-and-authority-changes) and this file is removed. The Flash-Next design is
**FN §x** ([qwen3_8-flash-next-design.md](qwen3_8-flash-next-design.md)), the GLM-5.3 design
**GLM §x** ([glm5_3-flash-design.md](glm5_3-flash-design.md)), and the companion IQ3_S plan
**IQ §x** ([qwen3_8-flash-next-iq3s-plan.md](qwen3_8-flash-next-iq3s-plan.md)).

Code references are to branch `Infernix` at `217135bba` (worktree
`E:\NInfer-V3\local\workdirs\NInfer-V3-issue1`). Labels as in the IQ plan: **measured**, **upstream**,
**model**, **estimated**; a bracketed milestone replaces each model with a measurement.

---

## Contents

1. [Deliverable, baselines and acceptance](#1-deliverable-baselines-and-acceptance)
2. [Source facts](#2-source-facts)
3. [What other implementations do](#3-what-other-implementations-do)
4. [Design decisions](#4-design-decisions)
5. [Artifact and conversion](#5-artifact-and-conversion)
6. [Expert arithmetic](#6-expert-arithmetic)
7. [Expert kernels](#7-expert-kernels)
8. [Residency: VRAM, RAM and SSD](#8-residency-vram-ram-and-ssd)
9. [Everything else: dense, PLE, MTP, vision](#9-everything-else-dense-ple-mtp-vision)
10. [Performance model](#10-performance-model)
11. [Qualification](#11-qualification)
12. [Milestones](#12-milestones)
13. [Documentation and authority changes](#13-documentation-and-authority-changes)
14. [Risks, options and rejected alternatives](#14-risks-options-and-rejected-alternatives)
15. [Sources](#15-sources)

---

## 1. Deliverable, baselines and acceptance

### 1.1 Deliverable

Two recipes over one source, sharing the expert banks, the drafter and the n-gram volume (the
pattern of recipes A and B, FN §6.1):

- **`qwen3_8_flash_next_fp8` (F-A, exact)**: every tensor of the official checkpoint imported
  bit-exactly, except the MTP drafter, which is re-quantized as in every Flash-Next recipe. BF16 dense
  on recipe A's BF16 routes.
- **`qwen3_8_flash_next_fp8_dense8` (F-B, recommended)**: F-A with recipe B's 8-bit dense classes
  (`fp8_e4m3fn_row_bf16`/`q8_g32_fp16`, qualified on this model at the same measured quality,
  FN §6.1, README "Bit-exact against Dense8").

Runtime: every route Flash-Next has today, with the routed experts on a three-tier store (VRAM
frames, pinned RAM, SSD), because 121 GB of FP8 experts exceed this machine's 96 GB of RAM.

### 1.2 Baselines (RTX 5090 on PCIe 5.0 x8, i9-13900K, 96 GB DDR5-5800)

| Baseline | Decode | Prefill | Source |
|---|---|---|---|
| Recipe B (NVFP4 experts, all in RAM) | plain 132.2, MTP 157.6 tok/s | pp131072 5,767 tok/s | measured, FN §19.3.21 |
| Recipe B in SSD-tier mode (dense8m, cold), expert RAM full / 32 GiB / 8 GiB | tg512 109.6 / 86.3 / 46.3 tok/s | long prompt 977 / 542 / 407 tok/s | measured, FN §19.3.7 (older build) |
| vLLM, 2 × RTX PRO 6000, FP8, MTP | ~200 tok/s | 9.8-12.9K tok/s | upstream (`net-snix/qwen38-flashnext-vllm`); all experts resident, not comparable |

No FP8 result on a 32 GB GPU was found (upstream search, 2026-10-10).

### 1.3 Acceptance

- **Must, exactness.**
  - `verify_artifact` matches every expert's codes and scales, every imported dense tensor (F-A),
    and every n-gram row.
  - Ops are **bit-equal** to their exact oracles (§11).
  - Greedy output is independent of placement (VRAM, staging, CPU, SSD-landed), decode width,
    C ≤ 8 and MTP on/off at C = 1.
- **Must, quality.** The FN §16.5 KL protocol for F-A and F-B against recipe B. The expected result
  (estimated) is better than recipe B, since FP8 128×128 experts are far finer than NVFP4. The size
  of that gain is the reason to build this artifact.
- **Targets (model, §10).** These are gates, not completion conditions; M0 replaces them with replay
  measurements.
  - F-B plain decode ≥ 80 tok/s central, MTP ≥ 95, on two expert volumes.
  - pp131072 ≥ 5,000 tok/s.
  - Short-prompt TTFT ≤ 6 s.

### 1.4 Non-goals

- Official-style dynamic FP8 activations as the default: Infernix keeps exact A16 (§6). They are
  offered only as a gated prefill permission (§14.2).
- Re-quantizing the experts to fit RAM. That would be a different artifact; NVFP4 recipes A–C already
  are.

---

## 2. Source facts

Read 2026-10-10 from the repository's `config.json`, `model.safetensors.index.json` (31 MB, fetched
through `/resolve/`) and the HF API. No safetensors header was read, so items marked **[deduced]**
come from the config and byte arithmetic and are confirmed in M0.

| Item | Fact |
|---|---|
| Size | 131 shards, `total_size` 185,502,232,570 B; BF16 5,487,198,064 params, F8_E4M3 174,512,783,360, I64 35 (upstream) |
| Quantization | `quant_method: fp8`, `weight_block_size: [128, 128]`, `activation_scheme: dynamic`, `weight_per_tensor: false`; `modules_to_convert: ["ple.ple_embedding.ngram_embedding"]`; `modules_to_not_convert`: 943 entries, covering every dense linear, the MTP dense module, `lm_head`, `embed_tokens` and vision (upstream) |
| **FP8 tensors** | Only the routed experts (main and MTP) and the n-gram table. **All dense projections are BF16**, byte-for-byte the original checkpoint's [deduced: same name set and shard numbers as the BF16 repository] |
| Routed experts | Per expert, not fused: `model.language_model.layers.{l}.mlp.experts.{e}.{gate,up,down}_proj.weight` (E4M3) and `.weight_scale_inv`; gate/up [640, 2560], down [2560, 640] [deduced]; 5 × 20 / 20 × 5 scale tiles per matrix; **scales 2-byte (BF16)** [deduced: the byte arithmetic only closes with 2-byte scales] |
| Expert bytes | 120,795,955,200 B of codes + 14,745,600 B of scales; 4,915,800 B per expert (model) |
| MTP experts | `mtp.layers.0.mlp.experts.{e}.*`: block FP8, 2.52 GB, the same form NVIDIA's checkpoint carried and the converter already decodes exactly (FN §6.1) |
| n-gram table | `…layers.1.ple.ple_embedding.ngram_embedding.shard_{0..127}.weight` (E4M3) and **one** `ngram_embedding.weight_scale` (2-byte scalar): the form recipes A–C import. NVIDIA's NVFP4 checkpoint carried it under exactly these names (upstream index), and FN found NVIDIA's codes equal to the per-tensor rule on 327,680 sampled values |
| Dense BF16 bytes | 10,974,396,128 B including embedding and `lm_head` (2.54 GB), vision, MTP dense (model) |

**Correction to FN §6.1.** FN rejected "importing the dense FP8 tensors of
`Qwen/Qwen3.8-Flash-Next-FP8`" because 128×128 blocks are coarser than `q8_g32_fp16`. The checkpoint
has **no** FP8 dense tensors: its dense linears are excluded from conversion. The rejection stands for
recipe B on other grounds, but its premise is corrected (§13).

---

## 3. What other implementations do

| Source | Technique | Takeaway |
|---|---|---|
| **vLLM** | SM120 dense block-FP8 via CUTLASS (#22131: 8.5-57.9 % over Triton on a 5090); MoE falls back to Triton `fused_moe` on SM120 (no CUTLASS grouped FP8 builder for SM120, issue #43507); `VLLM_PLE_CPU_OFFLOAD=1` puts the n-gram table in RAM; no routed-expert offload | All experts resident: needs ≥ 2 × 96 GB GPUs for this checkpoint |
| **CUTLASS example 87** | `Sm120BlockwiseScaleConfig<1,128,128>`, MMA tile 128×128×128, grouped variant with device-side TMA descriptor updates; runtime scale-block sizes unsupported on RTX 50 | Reference for the gated A8 prefill option only |
| **sm_120 FP8 MMA rates** (FlashInfer #3628, RTX 5060 Ti) | `QMMA.F32.E4M3` with FP32 accumulate at **half rate** (~102 TFLOP/s); the block-scaled `QMMA.SF…E8` (ue8m0) at full rate (~202); Hopper/Ada FP8 accumulators are narrower than FP32 (DeepSeek-V3: ~14 bits; PyTorch: e8m13) | A8 on tensor cores is neither exact nor reproducible on the CPU; the gated option uses the block-scaled atom with unit scales and promotes to FP32 every 128 K |
| **KTransformers** | AVX2 backend with block-FP8 experts (`--kt-method FP8`); no qwen4_exp support found | CPU FP8 experts are a known pattern; ours must be exact |
| **llama.cpp #29887** | GPU LRU over host experts, misses uploaded per call: 67.8 tok/s at 89 % hits on a 5090 (Q4_0) | Confirms the cache-plus-host design; Infernix already exceeds it |
| **Edge0** (arXiv 2609.18063), **SeqMoE** (2609.12978), **MoE-CORE** (2610.01950) | Trained per-layer pre-routers predicting the next layer one token ahead; multi-layer prediction with deadline-scheduled prefetch and probabilistic Belady eviction (96.97 % hits at 45 % residency); non-uniform per-layer cache capacity | Lookahead for SSD-tier reads (§8.4); options behind replay gates |
| **GLM §3.3, §11** (this machine) | Two 990 PRO volumes add to 11.7 GB/s; a third pays only off-chipset; 1-2 MiB reads at QD 2-4 saturate a drive; striped records land in ~1.2 ms | The SSD tier stripes expert volumes over two drives (§8.3) |

---

## 4. Design decisions

| # | Decision | Reason |
|---|---|---|
| D1 | **Exact import** of every FP8 code and BF16 scale in a new expert format `fp8_e4m3fn_b128` with record layout `fp8_expert_b128_v1` (§5.2). This reinstates, with a producer this time, the format FN removed on 2026-10-08 (audit M10) | Bit-exact, verifiable |
| D2 | **Exact W8A16 arithmetic** (§6): integer FP8 values times the existing A16 encoding, exact per 128-block in int64, exact across blocks in int128, one rounding | Placement and batch invariance (FN §16.2.1, GLM L13); free choice of route per column count |
| D3 | **Three-tier residency is the normal mode**, with **two striped expert volumes** (§8) | 112.6 GiB of experts against 95.8 GiB of visible RAM |
| D4 | **No new dense kernels**: F-A uses recipe A's BF16 routes and F-B recipe B's 8-bit routes | The dense tensors are the same BF16 tensors NVIDIA's checkpoint carried |
| D5 | **Reuse the existing n-gram volume** (`--ngram-reuse`, then every row verified) | Same codes and scale (§2) |
| D6 | **MTP drafter as today** (`q4_g64_fp16` experts from the checkpoint's FP8 MTP experts); exact FP8 MTP experts are an option (§9.3) | Measured path; 1.34 GB against 2.52 GB of VRAM |
| D7 | Build the shared offload work once with the IQ3_S and GLM plans (§12.1) | Record geometry, exact-combine library, striped volumes, replay tool |

---

## 5. Artifact and conversion

### 5.1 Recipes

| Class | Source | F-A | F-B |
|---|---|---|---|
| Routed experts (48 × 512) | E4M3 + BF16 scale per 128×128 | **exact**, `fp8_e4m3fn_b128` in `fp8_expert_b128_v1` | same |
| GDN, QSA, shared experts, HC mixers, PLE key/value, `lm_head` | BF16 | BF16 | recipe B's 8-bit classes |
| Router, shared gate, GDN α/β, indexer | BF16 | BF16 | BF16 |
| Token embedding | BF16 | BF16, host-resident | same |
| Norms, `A_log`, `dt_bias`, conv | BF16/FP32 | direct | direct |
| n-gram table | E4M3 + BF16 scalar | existing volume (`--ngram-reuse`) | same |
| MTP | FP8 experts, BF16 dense | `_flash_next_drafter` unchanged | same |
| Vision | BF16 | BF16 (`--vision-offload`) | same |

### 5.2 Format and record

- **`fp8_e4m3fn_b128`**: `W[n,k] = e4m3fn(c[n,k]) · s[⌊n/128⌋, ⌊k/128⌋]`, s BF16 (FP32 if M0 finds
  FP32). The shapes here have no edge tiles. Defined in [tensor formats](tensor-formats.md) with an
  exact decode oracle. The converter refuses E4M3 NaN codes (`S.1111.111`).
- **`fp8_expert_b128_v1`**, one record per expert:

  ```text
  [ gate/up rows interleaved in 16-row groups (g0..g15, u0..u15, …) : 3,276,800 B ]
  [ down rows in 16-row groups                                     : 1,638,400 B ]
  [ scale plane: gate 5×20, up 5×20, down 20×5, BF16 verbatim       :       600 B ]
  [ padding to 4 KiB                                               :     3,496 B ]   = 4,919,296 B (1,201 pages)
  ```

  Interleaving keeps each row's scale tile ⌊row/128⌋ and gives one pass over x for gate and up, as
  `nvfp4_expert_rg16_v1` does. 24,576 records = 120.9 GB (112.6 GiB).
- The converter imports raw bytes per expert (`import_expert_bank`, `tools/convert/qwen4_exp.py:226`,
  with a block-FP8 encoder), checks each row's scale exponent spread (§6.1 precondition), and writes
  the banks **striped over the expert volumes** named at conversion (§8.3).

### 5.3 Source acquisition

No Flash-Next source checkpoint is on disk (measured: the NVIDIA checkpoint the existing recipes
used is gone).

- Download, at a pinned revision with SHA256, the shards holding experts, MTP and dense tensors:
  ~134 GB. Skip shards 00005-00037 if M0's header read shows they hold only n-gram shards, which the
  existing volume replaces.
- Free space (measured, GLM §3.2): `E:` 292 GB, `C:` 243 GB, `D:` 144 GB, `G:` 128 GB.
- Suggested placement:
  - source on `E:` (134 GB);
  - artifact expert volumes striped over `D:` and `E:` (60.5 GB each);
  - dense part and drafter beside the first volume;
  - n-gram volume stays on `G:` (Optane).
- The source can be deleted after `verify_artifact`.

---

## 6. Expert arithmetic

### 6.1 Canonical W8A16 (exact up to one rounding)

The activation encoding is FN §16.2.1's, unchanged: X_j integers with |X_j| < 2^21 and one column
exponent emax, stored as three byte limbs. Every E4M3FN value is an integer multiple of 2^−9:

```text
w_int = s · (8 + m) · 2^(e−1)    (normal, e ∈ [1, 15])        |w_int| ≤ 14 · 2^14 = 229,376 < 2^18
w_int = s · m                     (subnormal, e = 0)
P_b   = Σ_{k ∈ K-block b} w_int · X_k       exact int64: |P_b| < 2^18 · 2^21 · 2^7 = 2^46
T_b   = P_b · mant(s_b)                       exact int64: < 2^54 (BF16 mantissa, 8 bits)
R     = Σ_b T_b · 2^(exp(s_b) − e_min)        exact int128 when the row's scale exponents span ≤ 64
y     = bf16_rn( fl32_rn(R) · 2^(e_min − 7 − 9 + emax − 147) )
```

- Notation: a BF16 scale is s = mant(s) · 2^(exp(s) − 7), with mant(s) ∈ [128, 255] (subnormal
  scales are refused) and exp(s) unbiased; e_min is the smallest exp(s_b) of the row.
- **Precondition, checked by the converter**: within each row's 20 (gate/up) or 5 (down) scale
  tiles, the exponent spread is ≤ 64. Real block scales span a few binades; a violation refuses the
  conversion with the tensor named.
- SwiGLU and h follow FN §16.2.1: `x → A16 → gate/up → BF16 → SwiGLU → BF16 h → A16 → down → BF16`.
- **Any order**: all sums before R's rounding are exact, so every route may split K, rows and
  columns freely and return the same bits.
- This is the same contract as IQ §6.1 (integer codes, integer sub-scale, floating block scale). One
  shared `exact_combine` library serves both plans.

### 6.2 Why not FP8 tensor cores with FP32 accumulation

- Their accumulation is narrower than FP32 and undocumented on sm_120. FP32-accumulate QMMA runs at
  half rate (FlashInfer #3628).
- The CPU cannot reproduce it, so placement invariance would fail.
- FN's 476-column finding shows that a non-exact wide route breaks KV reuse equality.
- It stays a gated prefill permission (§14.2).

---

## 7. Expert kernels

`offloaded_sparse_moe` gains the format field and per-call record geometry described in IQ §7.1
(shared work). FP8 adds one format family, `<Fp8B128, Columns>`.

### 7.1 GPU narrow route (≤ 8 columns)

- **Decode**: byte → w_int through a 256-entry int32 table in shared memory (1 KB), or arithmetically
  (`e = b>>3 & 15`, `m = b & 7`); M3 measures both.
- **Products**:
  - n = 1: `IMAD.WIDE` (w_int × X as 32×32 → 64) into an int64 accumulator per 128-block. That is
    one product per weight, against a DRAM time of 1 byte per weight.
  - n = 2..8: integer MMA `m16n8k32` with w split into three signed 6-bit limbs and X into its three
    byte limbs (9 MMAs per k32 tile of 16 rows × 8 columns), padded columns masked.
  - Exactness makes the route choice free; M3 picks per n by measurement.
- One CTA per (expert, 16-row group), gate/up then down, PDL-chained. Weight streams evict-first;
  X limbs staged in shared memory.
- Budget gate: ≤ 110 % of byte-time at 1.55 TB/s for n = 1, 2, 4, 8.

### 7.2 GPU wide route (prefill)

- The same 9-MMA limb product on `mma.sync` int8. Tiles decoded to int8 limbs in shared memory,
  double-buffered; X by `cp.async`.
- Cost (model): about 2.25× FN's W4A16 wide route per weight.
- In a 64K-token walk span each expert sees ~1,280 columns, and the expert compute is ~58 T integer
  ops per span: ~0.3 s at an estimated 200 TOPS, against ~3.4 s of streaming. So prefill stays
  link-bound (§10.3).

### 7.3 CPU engine (AVX2 + AVX-VNNI)

- **Decode**: 16 FP8 bytes → three limb byte vectors via 16-bit-lane arithmetic (`vpsllvd` on
  widened lanes, then pack).
- **Products**: `vpdpbusd` with unsigned X lo/mid bytes against signed limbs, and biased limbs
  against the signed hi byte (FN §16.2.1's trick). 9 byte products per weight per column.
- **Multi-column**: decode a 16-row group once, then apply to every column.
- **Cost (model)**: ~1.4 M `vpdpbusd` per expert-column plus decode, about 0.3-0.4 ms per expert per
  P-core. That is comparable to its DRAM time (4.9 MB at ~12-15 GB/s per core). M3 measures it.
  `cpu_rates.cpp` feeds the divisor, which moves misses to PCIe if the CPU is compute-bound.

---

## 8. Residency: VRAM, RAM and SSD

### 8.1 Capacities (model)

| Tier | Capacity | Share of 24,576 experts |
|---|---|---:|
| VRAM frames, F-B (~26.0 GB, as recipe B) | ~5,290 frames | 21.5 % |
| VRAM frames, F-A (BF16 dense, ~4.2 GB more) | ~4,430 frames | 18.0 % |
| Pinned RAM, expert cap 64 GiB (FN pins 64.47 GiB today) | ~13,970 records | 56.8 % |
| SSD only (tiers exclusive, FN L12) | rest | ~21.7 % (F-B) |

### 8.2 Policies

FN's built SSD tier (R0-R10.2, FN §19.3.7) is reused unchanged in its mechanisms:
- LFRU for VRAM;
- decayed LFU for RAM (re-qualified in M0 replay; GLM's evidence says decayed LFU lost on GLM's
  drift-shaped locality);
- promotions 1 per layer call, no promotions after prefill (L5, L6);
- warm start from the expert state file (L9).

New:
- **Per-layer strides** are trivial here (one record size), but the stride must come from the bank,
  not from `kRecordBytes` (IQ §7.5).
- `tools/expert_cache_replay/host_tier.py` replays FP8 record sizes with `--vram`/`--ram-slots`
  (M0).

### 8.3 Striped expert volumes

- GLM §7.3 and §11 design them: records striped in ~1.1 MiB units over two drives, a read engine
  with 2-4 reads in flight per drive, partial-record consumption, calibration and failure handling.
  They are **not built**. This plan builds them as a shared component (`core/` reads,
  `models/qwen4_exp/program/host_expert_tier` consumer), usable by GLM later.
- Placement on this machine: `D:` + `E:` (11.7 GB/s measured together); `G:` holds the n-gram volume.
  A record lands in ~0.45 ms (4.92 MB / 11.7 GB/s, model). One volume: ~0.85 ms.
- Exactness is unaffected: SSD-landed records are computed by the same CPU or GPU kernels.

### 8.4 Lookahead for SSD-tier experts

- The GPU spends ~0.15-0.2 ms per layer (model, from FN's per-layer budget), so a 0.45 ms SSD read
  must be predicted **2-3 layers ahead** to hide.
- Baseline predictor: router l+2 and l+3 applied to layer l's MoE input (FN L7 measured 65 % recall
  @10 one layer ahead; farther is lower).
- Options behind M0 replay gates: an Edge0-style trained pre-router per layer; SeqMoE's multi-layer
  prediction with deadline scheduling.
- Prefetch is bandwidth-budgeted below demand reads and lands in RAM, never in VRAM directly (GLM
  §11.5).

### 8.5 Host RAM ledger (model, 95.8 GiB visible)

| Item | GiB |
|---|---:|
| Expert records (cap) | 64.0 |
| Embedding BF16 | 1.18 |
| Vision tower | 0.84 |
| Prefix-cache Host tier | 4.0 |
| n-gram row cache (pageable) | 0.16 |
| SSD landing ring, staging, allowances | ~1.5 |
| Headroom | 2.0 |
| **Total** | **~73.7** |

A RAM-resident n-gram table (47.7 GiB in FP8) does not fit beside a useful expert tier. Every GiB
given to the table instead of experts moves ~218 records to the SSD, so `--ngram-residency auto`
(IQ §9.2) keeps the table on its volume for this artifact.

---

## 9. Everything else: dense, PLE, MTP, vision

### 9.1 Dense

F-A binds the BF16 dense tensors to recipe A's routes, F-B to recipe B's (§5.1). Both families exist,
are tuned and are qualified. The converter checks each BF16 tensor's name, shape and dtype against
the recipe; `verify_artifact` compares bytes.

### 9.2 PLE

- `--ngram-reuse G:\ninfer\qwen3_8_flash_next_nvfp4.ninfer.ngram` (52.4 GB, present). The converter
  checks geometry and 512 sampled rows; `verify_artifact` checks all 320,001,536 rows and the scale
  word against this checkpoint's shards.
- If any row differs, a new volume is written from this checkpoint. That is not expected (§2).

### 9.3 MTP

- **Default**: the existing drafter recipe (`_flash_next_drafter`, `official_recipes.py:278`) from
  this checkpoint's MTP tensors: experts decoded exactly (`dequantize_fp8_block128`) and re-quantized
  to `q4_g64_fp16` (1.34 GB resident), dense `q8_g32_fp16`, proposal head `q4_g64_fp16` from the
  BF16 `lm_head`.
- **Option**: exact FP8 MTP experts (2.52 GB). They run through `resident_moe_experts` with an FP8
  variant of the §6.1 arithmetic, costing ~240 frames more than q4 (model). Adopted if the measured
  acceptance gain beats the frame loss in tok/s (M7).

### 9.4 Vision

As recipes A–C: BF16, `--vision-offload` on by default.

---

## 10. Performance model

### 10.1 Hit rates (model; M0 replays real traces)

FN's LFRU replay measured 18.4 / 5.2 / 1.4 % misses at 16.3 / 32.6 / 48.8 % VRAM capacity. Live
cold-start rates are worse than replay (recipe B tg512 88 % hits at ~38 % capacity, measured). For
the F-B capacities (VRAM 21.5 %, VRAM + RAM 78 %), over 480 routed activations per token:

| Scenario | h_v | h_r | h_s | RAM-served per token | SSD reads per token |
|---|---:|---:|---:|---:|---:|
| Optimistic (replay-like) | 0.88 | 0.115 | 0.005 | 55 | 2.4 |
| Central | 0.82 | 0.17 | 0.01 | 82 | 4.8 |
| Pessimistic | 0.75 | 0.23 | 0.02 | 110 | 9.6 |

### 10.2 Decode, C = 1 (model)

Per token, using GLM §5.3's critical-path form:

| Term | Optimistic | Central | Pessimistic |
|---|---:|---:|---:|
| GPU dense (F-B 4.4 GB) + attention, router, head | ~3.8 ms | ~3.8 ms | ~3.8 ms |
| Host service, RAM-served + SSD-landed bytes at ~65 GB/s effective | 4.4 ms | 6.6 ms | 9.0 ms |
| Exposed SSD (half the reads unhidden, 0.45 ms each) | 0.5 ms | 1.1 ms | 2.2 ms |
| Round overheads (from recipe B's measured 7.56 ms) | ~1.2 ms | ~1.2 ms | ~1.2 ms |
| **Token** | **~9.9 ms** | **~12.7 ms** | **~16.2 ms** |
| **Plain tok/s, F-B** | **~100** | **~79** | **~62** |
| MTP (+15-25 %, L8) | ~120 | ~95 | ~72 |

F-A: ~2.7 ms more GPU time and ~17 % fewer frames: ~80 / 64 / 50 tok/s plain (model).

Levers (model, as GLM §5.5): PCIe x16 (move the NIC) gives +10-15 %; a third volume on `C:` cuts the
SSD term by ~1/3; 192 GB of RAM holds the whole bank (h_s = 0), for ~105 tok/s central.

### 10.3 Prefill (model)

- **Streaming**: ~0.785 × 120.9 GB = 95 GB per walk span at 27.6 GB/s = 3.4 s. The SSD share is
  ~26 GB at 11.7 GB/s = 2.2 s, concurrent with it.
- **Long prompts**: recipe B's pp131072 takes ~22.7 s (two spans), set by dense and attention compute
  that this artifact shares, so the 2 × 3.4 s stream hides under it. **≈ 5,000-5,700 tok/s.**
- **Short prompts** (one span) pay the whole stream. FN measured ~2.3 s for NVFP4 (L16); scaled by
  bytes, that gives **~5-6 s TTFT**. The prefix cache and the prefill CPU split (FN §19.3.12) shorten
  repeated and short prompts.

---

## 11. Qualification

| Item | Oracle | Criterion | Coverage |
|---|---|---|---|
| `fp8_e4m3fn_b128` | Exact decode (Python, FP64) | Exact, every value, every expert | 24,576 + 512 MTP experts |
| `offloaded_sparse_moe` FP8 family | §6.1 exact rationals and two roundings | **Bit-equal** | Columns 1..8, 9, 16, 476; narrow SIMT and IMMA, staged, wide, CPU scalar/AVX2/AVX-VNNI; random residency mixes incl. SSD-landed; non-finite columns; scale-spread boundary |
| `exact_combine` | Exact rationals | Bit-equal | Random and adversarial exponent spreads up to the precondition |
| Striped volumes | Single-file reads | Byte-equal records | Every record at conversion; injected read failures fail only the round (FN L12) |
| Model | `verify_artifact`; greedy invariance (placements, C ≤ 8, MTP); KL protocol vs recipe B; tg512, agentic A/B, pp131072, short-prompt TTFT on one and two volumes | as FN | — |

---

## 12. Milestones

Estimates are focused engineering time, ±50 %.

### 12.1 Shared foundation (with the IQ3_S plan; build once)

| M | Content | Est. |
|---|---|---|
| S1 | `MoeExpertSource` format field and `RecordGeometry`; per-bank strides through store, host tier, SSD tier, staging, VRAM sizing, divisor; replay tool with per-layer sizes | 4 d |
| S2 | `exact_combine` (int64/int128, single rounding) on CUDA and x86 (`_addcarry_u64`), with oracle tests | 2 d |
| S3 | Striped expert volumes and read engine (GLM §11), calibration probe | 5-7 d |

### 12.2 FP8-specific

| M | Content | Exit gate | Est. |
|---|---|---|---|
| **M0** | Confirm §2 from shard headers (scale dtype, shapes); download; record route traces on recipe B; replay FP8 capacities (F-A/F-B; RAM cap 56-64 GiB; 1 and 2 volumes); measure the Windows pinned ceiling | §10 replaced by replay; volume layout fixed | 2-3 d (+ download) |
| **M1** | `fp8_e4m3fn_b128` codec, record layout, striped bank writer, recipes F-A/F-B, `verify_artifact` | Whole-artifact verify; n-gram reuse verified | 3-4 d |
| **M2** | Correctness-first experts: SIMT narrow, scalar CPU, wide via SIMT | Model runs; greedy invariance; KL measured | 3-4 d |
| **M3** | Tuned narrow (IMAD n = 1, IMMA n ≥ 2), wide IMMA, AVX-VNNI engine, `cpu_rates` | Bit-equality matrix; kernels ≤ 110 % of byte-time | 6-8 d |
| **M4** | SSD lookahead (baseline predictor), prefetch budget; policy re-qualification in replay | Exposed SSD time per token measured | 3-4 d |
| **M5** | Performance campaign: divisor, promotion interval, prefill CPU split recalibrated for 4.92 MB records | §1.3 targets, or attribution | 3-5 d |
| **M6** | Options: exact FP8 MTP experts; trained pre-router; A8 prefill permission | Each by its gate | 3-5 d |
| **M7** | MTP acceptance and draft policy check; documentation (§13); model cards | — | 2 d |

Total ≈ 30-40 days, plus 11-13 days of shared foundation if the IQ3_S plan has not built it. If both
plans are approved, building FP8 first exercises the shared plumbing with a single expert format.

---

## 13. Documentation and authority changes

- [tensor formats](tensor-formats.md): `fp8_e4m3fn_b128`, `fp8_expert_b128_v1`.
- [Flash-Next design](qwen3_8-flash-next-design.md):
  - §6.1: the recipes, and the correction of the "dense FP8 tensors" premise (§2);
  - §16.2.x: the exact W8A16 contract;
  - §19.3.7: striped volumes and lookahead.
- `docs/qwen3_8-flash-next.md`, `docs/cli.md`, `docs/serving.md`: the artifacts, SSD-volume flags,
  expected performance by RAM size and volume count.
- `model-cards/Qwen3.8-Flash-Next-FP8-Infernix/` and `…-FP8-Dense8-Infernix/`: provenance (Qwen
  revision), license.
- `docs/README.md` row for this plan, removed with it.

---

## 14. Risks, options and rejected alternatives

### 14.1 Risks

| Risk | Effect | Mitigation |
|---|---|---|
| Live miss rates at the pessimistic end | ~60 tok/s plain | Replay in M0 before kernel work; lookahead; promotions policy; more volumes or RAM (hardware advice) |
| CPU exact-FP8 cost above DRAM time | Lower CPU miss throughput | Divisor shifts misses to PCIe; multi-column decode-once |
| Windows pinned ceiling | Smaller RAM tier | Pageable CPU-only RAM tier (GLM §10.4) |
| Short-prompt TTFT ~5-6 s | Worse interactive feel than recipe B | Prefix cache; prefill CPU share; documented |
| Scale dtype or spread differs from §2 | Converter refusal | M0 header read; FP32 scales fit the same int128 combine |

### 14.2 Options (each behind its own gate)

- **A8 prefill permission**: dynamic 1×128 E4M3 activations (the official serving arithmetic) through
  `QMMA.SF` with unit ue8m0 scales and FP32 promotion every 128 K, prefill wide route only.
  - Not exact and not batch-invariant, so it is off by default.
  - Gate: prefill turns out compute-bound (it is link-bound in §10.3).
- Exact FP8 MTP experts (§9.3).
- Trained pre-router or multi-layer predictor (§8.4).
- Non-uniform per-layer RAM capacity (MoE-CORE), if replay shows per-layer skew that LFU misses.

### 14.3 Rejected

- **Re-quantizing experts to fit RAM**: that is recipes A–C.
- **Canonical fixed-order FP32 instead of exact integers**: cheaper on the CPU (one FMA per weight per
  column), but it fixes one reduction tree for every route, and the tensor-core wide route cannot
  follow it. FN's 476-column finding is the precedent.
- **FP8 tensor cores with FP32 accumulate as the default**: half rate on sm_120, not exact, not CPU
  reproducible (§6.2).
- **mmap of the expert volumes**: FN F9 measured mapped files slowing unbuffered reads 4×; the SSD
  tier reads unbuffered in place.
- **GPUDirect Storage**: not available on Windows. DirectStorage → D3D12 → CUDA interop is plausible
  but has no LLM implementation; not pursued.

---

## 15. Sources

- Qwen/Qwen3.8-Flash-Next-FP8: README, `config.json`, `model.safetensors.index.json`, HF API (read
  2026-10-10); nvidia/Qwen3.8-Flash-Next-NVFP4 index (same names for the FP8 MTP and n-gram tensors).
- vLLM #22131, #43507, recipes.vllm.ai (Qwen3.8-Flash-Next); SGLang cookbook;
  `net-snix/qwen38-flashnext-vllm`.
- CUTLASS `examples/87_blackwell_geforce_gemm_blockwise`; FlashInfer issue #3628; DeepSeek-V3
  report (arXiv 2412.19437); PyTorch blog on matmul accumulation precision.
- KTransformers AVX2 tutorial; llama.cpp #29887; arXiv 2609.18063 (Edge0), 2609.12978 (SeqMoE),
  2610.01950 (MoE-CORE).
- This machine's measurements: GLM §3 (link, DRAM, drives), FN §19.3.7 (SSD tier), FN §19.3.21
  (recipe B).
