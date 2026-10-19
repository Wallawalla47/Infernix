# DeepSeek-V4.1-Flash-1017 on one RTX 5090: design

**Status: design proposal, not implemented.** Nothing in this document is built. It is a temporary
planning document for adding `DeepseekV41ForCausalLM` (DeepSeek-V4.1-Flash, model type
`deepseek_v41`) to Infernix, with the NVFP4 build as the artifact. When the implementation lands,
its stable content moves into the active authorities named in
[§24](#24-documentation-and-authority-changes) and this file is removed. The design reuses the
Qwen3.8-Flash-Next work and the GLM-5.3-Flash design wherever they transfer; they are cited as
**FN §x** ([qwen3_8-flash-next-design.md](qwen3_8-flash-next-design.md)) and **GLM §x**
([glm5_3-flash-design.md](glm5_3-flash-design.md)). DwarfStar (`antirez/ds4`), the local engine
whose V4.1 implementation this design studied most closely, is cited as **DS4** (§4.1).

Labels used for numbers: **measured** (on this machine, with the method named), **upstream** (read
from the reference code, config, checkpoint headers or another project's published record),
**model** (derived from measured or upstream inputs by the stated arithmetic), **estimated**
(judgement; uncertain by the stated range). A milestone named in brackets replaces each model or
estimate with a measurement.

**The artifact.** "NVFP4 DeepSeek-V4.1-Flash" names one set of expert values, whichever file it
comes from:

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

**Default artifact:** `DeepSeek-V4.1-Flash-1017-Infernix`, converted from the official 1017
release: MXFP4 expert records striped over three NVMe volumes, the FP8 Engram volume, and every
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
- DeepSeek's prompt rule (decoder SWA bounded replay) by default and the full-model rule as a
  startup option (§16);
- DSpark speculative decoding chosen per round by the cost model, and n-gram copy proposals (§17);
- the model's own KV formats: FP8 window entries, FP4 compressed entries, MXFP4 index keys (§13.5);
- the hybrid prefix cache in `infernix-serve` (§18);
- the DeepSeek encoding: thinking, reasoning effort 1-100, DSML tool calls, tool results and
  reminders in the OpenAI and Anthropic protocol surfaces (§21);
- routed experts in **three tiers**: a VRAM expert cache, pinned and unpinned host RAM, and expert
  volumes striped over **one to four NVMe drives** (§10-§11);
- the Engram tables in their own volume, with **FP8 and MXFP4 row formats** selectable like
  Flash-Next's n-gram tables (§14);
- one recipe for the official DeepSeek release (September or 1017) and NVIDIA's ModelOpt NVFP4
  export, and a **streaming conversion** that fits the disks free at conversion time (§7.6);
- an **opt-in approximate expert mode** (GLM §9.8's shape), off by default and labelled (§9.8).

**Fidelity.** Every stored tensor is imported bit-exactly: dense FP8 codes and scales, BF16 and FP32
tensors verbatim, Engram rows verbatim in the chosen format, and routed experts as a lossless
re-encoding that the converter checks block by block. Every FP8 and FP4 linear takes DeepSeek's own
activation quantization (MXFP8: E4M3 with one power-of-two scale per 32 activations, decision b)
and computes it with exact integer sums (**MX8**, §9.5), so an expert's output is the same bits from
a VRAM frame, a staging frame, the CPU or any landing order, and Infernix reproduces the reference
forward up to the accumulation order the reference leaves to its tensor cores. The deviations from
`inference/model.py` are listed in §20.4.

**Vision** (DeepSeek-ViT, 32 layers) is deferred: the artifact keeps the tower, the binder refuses
`--vision` for this family until a later track adds it (§23).

### 1.2 Owner decisions (2026-10-19)

| # | Question | Decision | Implemented in |
|---|---|---|---|
| a | Prompt rule | **DeepSeek's decoder bounded replay by default**: prompt tokens run the 20 encoder layers; only the last 128 run the decoder. **Switchable** to the full model (`--prompt-decoder full`), computed exactly by the 2,541-position decoder suffix DS4 uses, whose output equals running every prompt token through all 40 layers | §16 |
| b | Activation arithmetic | **DeepSeek's 8-bit activations** (MXFP8) at every FP8 and FP4 linear | §9.5 |
| c | Expert storage | **The original MXFP4 form**, provided it is not slower than NVFP4 records; M3 decides by kernel measurement, NVFP4 records remain the fallback and serve non-MXFP4-representable builds | §7.3, §9.6 |
| d | Engram tables | **Both FP8 (source) and MXFP4 (NVIDIA) rows**, read from the volume, chosen per run as Flash-Next's n-gram row formats are | §14 |
| e | Disk space | **A conversion route designed for the disks free when the artifact is made**, not a fixed plan | §7.6 |

### 1.3 Baselines

All comparisons run on this machine with the same prompts, output caps and sampling.

| Baseline | What it shows |
|---|---|
| **JigSawPT's llama.cpp fork** (`dsv41-porte`) with the same MXFP4 experts streamed from NVMe | **The main baseline**: the only published run on an RTX 5090 with the experts in the same format (§4.2: 5-10 tok/s with 125.7 GiB of RAM). Identical expert bytes, so speed is compared without a quality footnote |
| **DS4** (`antirez/ds4`, MIT) with its V4.1 Q2 (341 GiB file, 152 GiB of weights) and Q4 (483 / 294 GiB) GGUFs, CUDA, SSD streaming | The local engine studied most closely (§4.1). Its CUDA V4.1 path targets the DGX Spark's unified memory and Linux; on this machine it would run under WSL2 with the GGUF on an ext4 volume, and its discrete-GPU expert cache is an unmerged pull request (DS4 #647). Compared as each engine runs in practice: Q2 has about half the expert bytes of NVFP4 and measurably lower quality (§4.1), so the report gives quality beside speed |
| **ik_llama.cpp** and the other engines of §4.2 that run V4.1 with expert offload | Their own numbers in §4.2; run here where they support Windows or WSL2 |
| **Infernix Flash-Next** on this machine | Not a quality baseline (another model); the reference for what each mechanism achieved here |
| **DeepSeek's reference** (`inference/model.py`) and vLLM/SGLang | Need four or more datacenter GPUs; reference logits only, through §20.2's offline reference forward |

### 1.4 Targets

Fixed on 2026-10-19 from §5's model; no measurement of this model exists on this machine.
Primary configuration: the 1017 artifact (MXFP4 records, FP8 Engram volume), three expert volumes
on `D:` + `E:` + `G:` (§11.4), bounded replay, `--max-context 131072`, C = 1, ≈ 88 GiB of RAM
available (other programs closed), thinking at the encoding's default effort, 4K prompt, 512 output
tokens.

| Metric | Must | Stretch | Model (§5), central (range) |
|---|---:|---:|---:|
| Plain decode, warm cache, C = 1, non-repeating session (§20.5) | ≥ 9 tok/s | ≥ 15 | 12.8 (9.7-16.9) |
| Best speculative mode, same corpus | ≥ plain, never slower beyond noise | ≥ 17 | 13.6-16.2 with DSpark K = 1 at the measured acceptance (§5.4) |
| Decode, three volumes relative to one | ≥ 1.8× | ≥ 2.0× | 2.0× |
| Fresh 32K prompt | ≥ 2,000 tok/s | ≥ 3,500 | 2,800 |
| Time to first token, fresh 4K prompt | ≤ 15 s | ≤ 9 s | 11.7 s |
| Time to first token, 50-token append to a cached conversation | ≤ 6 s | ≤ 3.5 s | 4.7 s |
| Decode at 128K actual context, relative to 4K | ≥ 90 % | ≥ 97 % | ≈ 98 % |
| Execution fidelity against the reference forward (§20.3) | within the gate | — | — |

The acceptance report gives the median and range of at least three runs, the worst case, the expert
service composition (VRAM hits, RAM experts computed by the CPU, RAM experts staged over the link,
SSD reads and bytes per token, Engram reads), and the device each number came from. A missed target
is reported as missed; targets are not changed after measuring.

These targets are far below the GLM design's (25 tok/s) because this model reads 4.5 GB of routed
experts per decode token (GLM's EXL3 build: 3.2 GB) from a bank of 289 GB (GLM: 115 GB), and this
machine holds about 37 % of it in memory (GLM: 87-96 %). That is a property of the model on this
hardware; §5.6 lists the hardware changes that move it.

### 1.5 Non-goals

- Multi-GPU, AMD GPUs, GPUs other than `sm_120a`.
- Any change to the weights, dynamic top-k, expert deferral or skipping in the default
  configuration; the one exception is the opt-in, labelled approximate mode (§9.8).
- Requantizing the experts to fewer bits in the default artifact. DS4's Q2 recipe shows what that
  costs on this model (§4.1); smaller recipes are options with their own gates (§25.3).
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
Full, r = 1 (20):          blocks of 8 positions scored by their maximum; the block holding the newest
                           position is always kept; top-2,048 blocks form the candidates C(t);
                           Sel = top-512 within C(t)
Reindex (24, 28, 32, 36):  Sel = top-512 of its own scores within C(t)
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
- **Bounded replay** (DeepSeek's serving rule, report §3.2.2): only the last n_win = 128 prompt
  positions run the decoder, with window attention truncated at the replay start s (query i sees
  window entries max(s, i − 127) … i). DeepSeek states it "is not mathematically equivalent" to the
  full forward, simulates it during post-training, and serves it; decoder windows are never cached.
- The two rules give the same encoder state and decoder compressed entries; they differ in the
  decoder windows at the prompt's end, hence in every token decoded after a prompt of more than 128
  tokens. For a prompt of at most 128 tokens following nothing cached they coincide.

Infernix runs bounded replay by default and the full model on request (decision a, §16).

### 2.4 Parameters and bytes (upstream sizes from the safetensors headers; model elsewhere)

| Item | Parameters | Stored bytes | Bytes read per decode token |
|---|---:|---:|---:|
| One routed expert: gate, up [5,120 → 2,304], down [2,304 → 5,120] | 35,389,440 | MXFP4 **18,800,640** (17.93 MiB); NVFP4 19,906,584 (+5.9 %) | — |
| Routed bank, 40 × 384 | 543.6 B | MXFP4 **288.8 GB**; NVFP4 305.8 GB | 240 experts: 4.51 / 4.78 GB |
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
- **About 7.5 % of the experts fit in VRAM and about 29 % in RAM** (§19), so ≈ 63 % live only on
  the SSDs. At the central hit rate (§5.2) a token reads ≈ 43 experts, ≈ 0.8 GB, from the volumes.
- **A miss moves 18.8 MB**: 0.68 ms over the x8 link, ≈ 1.33 ms from three striped volumes.
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
update; whether the Engram tables and DSpark weights changed is not stated, and M0 compares sampled
tensors and Engram rows of the two releases by range reads (§7.5). The 1017 repository drops the
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
| vLLM / SGLang | datacenter; **bounded replay implemented** (vLLM #56752: layers 21-39 run on the last 128 prompt tokens; GSM8K 0.9249 on vs 0.9295 off, SE 0.0075; SGLang #42901 also trims layer 20's non-KV work, TTFT −5 / −8 / −10 % at 16K / 64K / 200K); Engram in pinned host memory (vLLM #61530: MXFP4 tables 12.2 instead of 23.6 GiB per rank, GSM8K 0.967 vs 0.966); FlashMLA's sparse kernels are SM90/SM100 only, sm_120 has open blockers (#56837, #56461) | reference only |
| **diffbot EXL3 3 bpw** (two RTX PRO 6000) | Engram int4 in pinned host RAM, read over UVA | [M] 74.4 tok/s at 2K, prefill ≈ 10K tok/s. **KL against the original** (neutral / code / agentic): EXL3 3 bpw 0.0545 / 0.0321 / 0.0101; **NVIDIA NVFP4 as served by vLLM (W4A4 + MXFP4 Engram) 0.0352 / 0.0197 / 0.0089**; the original's second run (noise floor) 0.0158 / 0.0100 / 0.0055 |
| OpensourceWTF EXL3 streaming repack (M5 Max, 128 GiB) | 13.3 MB expert records streamed from SSD into a slot cache; Engram FP8 rows from SSD | [M] 661.5 tok/s prefill and 36.8 tok/s decode at 16K |
| ds41rt / cuteafd | an RTX 5090 coordinator plus four DGX Spark expert ranks over RoCE | [M] 77-110 tok/s; not one machine |

**Quantized artifacts of 1017** (Hugging Face API, 2026-10-19): NVIDIA has published only the
September NVFP4 export. `LibertAIDAI/DeepSeek-V4.1-Flash-1017-NVFP4` (2026-10-17) is a community
ModelOpt-style export: lossless MXFP4 → NVFP4 experts ("0 mismatches over 135.90 × 10⁹
elements") with `input_scale` values, MXFP4 Engram tables (cosine 0.9934 to the FP8 tables), DSpark
experts in MXFP4, 399.51 GiB; it publishes no KL. Unsloth's 1017 GGUF is marked work in progress;
bartowski, ubergarm and AesSedai have none.

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
  and both datacenter engines serve: the default (decision a).
- **Engram is random row reads from disk everywhere** (DS4, JigSawPT, Xolotl, OpensourceWTF) or
  pinned host memory where RAM allows (vLLM); no engine found it a bottleneck.
- **Other engines are not reproducible across their own execution modes** (DS4 scalar vs batched,
  JigSawPT with more than one I/O thread). Infernix's placement invariance (FN §16.2) is the
  difference this design keeps.

### 4.3 Lessons from Flash-Next and the GLM design

GLM §4.2 lists the 22 Flash-Next lessons (L1-L22) measured on this machine. They hold here; these
change weight:

| Lesson | Consequence here |
|---|---|
| L2: misses are a host-DRAM bandwidth problem | The SSD set binds first (≈ 0.8 GB of SSD reads per token at the central rate, §5.3); DRAM carries landings, CPU reads and DMA (≈ 30-40 GB/s, §5.1) |
| L5, L6: promotions cost a miss's bytes; none after prefill | A promotion is 18.8 MB (6.8× Flash-Next's); the interval is recalibrated (§10.2) |
| L7: next-layer prediction recall is modest | Confirmed for this family by JigSawPT and hipfire; lookahead off by default (§11.3) |
| L8: verify windows add misses only for rejected drafts | Here every verify column also adds its own experts' union (§5.4); the draft length is priced per round (§17.3) |
| L13: placement invariance needs exact arithmetic | MX8 on every route (§9.5) |
| L16: short prompts pay a whole expert pass | Worse: a 200-token append reads ≈ 155 GB from the volumes (§5.5); the prefix cache and append-only processing are first-class (§16, §18) |
| L19: the divisor rule beats cost models for the CPU/link split | Kept, fed by measured costs per 18.8 MB expert (§9.7) |
| Pin room (FN §19.3.7, 2026-10-14) | The RAM tier is pinned up to ≈ 72 GiB and unpinned past it (§10.4) |

---

## 5. Performance model

The script is kept with this design's working notes; M0 commits it as
`tools/deepseek_v41/decode_model.py`.

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

This family shows **much stronger locality than GLM-5.3-Flash** at equal capacity (GLM §5.2: ≈ 35-56 %
at 11 %). This machine's capacities (§19): VRAM ≈ 1,160 frames (7.5 %), RAM ≈ 4,490 slots (29.2 %)
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

- A fresh 32K prompt: ≈ 11.7 s, **≈ 2,800 tok/s** (compute ≈ 6 s, hidden under the bank pass);
  128K in one walk span: ≈ 25 s, ≈ 5,200 tok/s, compute-bound by the experts' exact wide route. At
  1M, eight 128K spans and the indexer's O(T²) scoring: ≈ 3.5 minutes (estimated). The W4A4 fast
  path (§25.3 O2) would cut the expert compute about fourfold (FN §16.2.1: 1.65 against 6.24 ms per
  call) where a prompt is compute-bound.
- **Any fresh prompt above ~200 tokens costs a bank pass, ≈ 11-12 s on three volumes**, and the
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
| Smaller experts (requantized, lossy; DS4 Q2 halves the bytes) | +60-80 % decode, at DS4's measured +0.12-0.19 nats NLL; option O6 only |

The link width is fixed; nothing here is tuned to a different link.

---

## 6. Design overview

```text
 ┌───────────────────────────────────── RTX 5090, 32 GB ─────────────────────────────────────┐
 │ dense FP8 / BF16 8.0 GiB · windows 2.9 MB/lane · compressed KV + index keys 890 B/token   │
 │ decode workspace · mailboxes · residency[43 × 384] · land_seq[frame]                      │
 │ ┌── frame pool: ~1,160 × 17.95 MiB MXFP4 records (≈ 20.3 GiB), elastic ──────────────┐   │
 │ │ cached experts │ staging │ DSpark experts (bounded) │ KV pages + loans │ walk lease  │   │
 │ └────────────────────────────────────────────────────────────────────────────────────┘   │
 └──────▲───────────────────────▲─────────────────────────▲───────────────────────────────────┘
        │ copy engine           │ mapped mailboxes        │ copy engine (landed records)
 ┌──────┴───────────────────────┴─────────────────────────┴─────────┐   ┌──────────────────────┐
 │ 96 GB host                                                        │   │ expert volume 0 (D:) │
 │  RAM tier: ~4,000 pinned slots (≈ 72 GiB) + unpinned records      │◄──┤ expert volume 1 (E:) │
 │  embedding (pinned) · Engram row cache · landing ring             │   │ expert volume 2 (G:) │
 │  transfer agent: VRAM policy, promotions, staging DMA             │   │  12 units per record │
 │  SSD agent: per-volume IOCP queues; Engram reader ring            │◄──┤ Engram volume (E:)   │
 │  CPU expert team: MX8 exact, from RAM slots or landing slots      │   │  FP8 or MXFP4 rows   │
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
     shared expert ‖ routed experts (MX8 narrow) ‖ CPU results ‖ landed records → combine (FP32, ascending id)
     ‖ mixes for the next sublayer
head: collapse with the last pre → norm → head (FP32 logits) → sample → DSpark draft (if chosen)
```

The ideas that carry the design:

- **Exact arithmetic makes placement free** (FN §5, L13): every FP4 and FP8 linear computes
  DeepSeek's MXFP8 product exactly, so a VRAM frame, a staging frame, the CPU from a RAM slot or a
  landing slot give the same bits, and policy may move work anywhere (§9.5).
- **The SSDs are the bottleneck; everything is sized to read fewer, smaller, fuller records**:
  MXFP4 records (−5.6 % bytes), three volumes striped at 1.5 MiB units (§11), units consumed as they
  land, LFRU with a 37 % memory share, the prefix cache, short drafts only (§17).
- **Three tiers, exclusive, one policy owner per boundary** (GLM §6): the transfer agent owns
  VRAM↔RAM, the SSD agent RAM↔SSD; the device reads residency words.
- **The model's own compressed KV is tiny**, so long context costs almost nothing and prefix
  snapshots are a few MB: the prefix cache can keep whole conversations (§18).
- **Prompt rule as a declared model mode**: DeepSeek's bounded replay by default, the exact full
  model on request; both are deterministic, request-invariant and computed with the same kernels
  (§16).

---

## 7. Artifacts, records, volumes and conversion

### 7.1 Sources and the one recipe

One recipe, `deepseek_v4_1_flash`, imports any of these sources; M0 pins each revision.

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
| Routed experts, 40 × 384 (`w1` gate, `w3` up, `w2` down) | MXFP4: packed I8 codes [2,304 × 2,560] / [5,120 × 1,152], E8M0 scales [2,304 × 160] / [5,120 × 72] | 288.8 GB | `mxfp4_expert_v1` records (§7.3) in the expert volumes (§7.4) |
| DSpark experts, 3 × 128 | MXFP4 (in every source) | 7.22 GB | same records, DSpark bank |
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

**`mxfp4_expert_v1`** (to be defined byte-exactly in [storage layouts](storage-layouts.md)). Both
processors read a record in place (FN §6.2's principle): the GPU from a frame or a staging slot, the
CPU from a RAM slot, a landing slot or an unpinned record.

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

- **Banks**: the main bank (15,360 records, numbered layer-major, expert-minor) and the DSpark bank
  (384 records).
- **Sizes**: MXFP4 15,744 padded records = **296.4 GB**: 98.8 GB per volume at w = (4, 4, 4). NVFP4
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

- Rows never straddle a block; each table's rows are contiguous in row order, so one row is one
  4 KiB direct read and prompt reads of neighbouring rows coalesce.
- The header names the format, both tables' row counts and their 24 primes each, and the artifact id
  it was made for.
- The converter makes an MXFP4 volume from FP8 rows with ModelOpt's MXFP4 rule (UE8M0 scale
  ⌊log₂ amax⌋ − 2 per 32, E2M1 round-to-nearest-even; M0 reads ModelOpt's code and checks it against
  NVIDIA's rows of the September table, which must match byte for byte).
- **Sharing**: if M0 finds the 1017 Engram tables equal to September's (sampled rows by range read),
  one FP8 volume serves every artifact of either checkpoint, bound with `--engram-reuse` as
  Flash-Next's derivatives bind NVIDIA's n-gram volume.
- Runtime selection follows decision d (§14.5).

### 7.6 Conversion under the disk limits (decision e)

The official source is 510.3 GB in 48 shards of at most 7.41 GB; NVIDIA's is 429.0 GB. The default
outputs are ≈ 11 GB of main artifact (dense 8.5, embedding 1.3, DSpark dense 0.7, vision 1.0),
296.4 GB of expert volumes and 209.7 GB of FP8 Engram volume: **≈ 517 GB**. On 2026-10-18 the four
drives had ≈ 895 GB free in all, so the source and the outputs cannot coexist. The conversion route
is therefore **shard by shard**, and its plan is computed from the free space at the time it runs.

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
   tensors (copy or recover expert blocks into their units, pack Engram rows, write dense tensors),
   read every written range back and compare it with the source bytes in memory, record the shard
   and the CRC32C of each written unit in a journal, then delete the shard. One shard downloads while
   the previous converts: peak staging ≈ 15 GB, peak RAM ≈ 4 GB (the converter runs at below-normal
   priority and reads the machine's free RAM before starting, since other sessions may need it).
4. **Resume**: an interrupted run restarts from the journal at the first incomplete shard.
5. **Verify**: `tools.deepseek_v41.verify_artifact` checks headers, sizes, the journal's CRC32Cs on
   every unit and, with the source reachable again, sampled tensors by range read.

**Disk plan on this machine** (2026-10-18 free space; recomputed by step 1 when the conversion
runs):

| Item | Size | Drive |
|---|---:|---|
| Expert volume 0, w = 4 | 98.8 GB | `D:` (300.4 free) |
| Expert volume 1, w = 4 | 98.8 GB | `E:` (405.6 free) |
| Expert volume 2, w = 4 | 98.8 GB | `G:` (108.7 free; 9.9 GB left) |
| Engram volume, FP8 | 209.7 GB | `E:` |
| Main artifact | ≈ 11 GB | `E:` (E: then 86 GB free) |
| Shard staging (two shards) | ≈ 15 GB | `D:` |

If `G:` has less room when the conversion runs (it also holds Flash-Next's n-gram volumes), the
planner moves units to `D:` (for example w = (5, 4, 3): 123.5 / 98.8 / 74.1 GB, ≈ 6 % slower records,
§11.4); with less room overall it proposes the MXFP4 Engram volume (−105 GB). Downloading 510 GB
needs the owner's go-ahead when the conversion is scheduled; it takes ≈ 70 minutes at 1 Gb/s.

### 7.7 Other artifacts

- **NVIDIA's September NVFP4 export** and **LibertAIDAI's 1017 NVFP4** convert to the same expert
  records as their official counterparts (§7.1) and bring MXFP4 Engram tables; an FP8 volume made
  from the official release can replace those (§14.5).
- **DS4's Q2 and Q4 GGUFs** are a different, requantized model: option O6 (§25.3) imports them with
  Infernix's exact GGUF arithmetic (FN §16.2.2) where its block formats cover them (Q2_K, Q4_K; IQ2_XXS
  is not built), at DS4's measured quality.
- **EXL3 builds** (diffbot 3 bpw, KL 0.0545 neutral) would need the GLM design's EXL3 kernels; not
  planned.

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
| Tokenizer pre-tokenizer, DeepSeek encoding, DSML tool-call and reasoning parsing | Frontend | §21 |
| Recipe, streaming conversion, MX recovery, expert and Engram volume writers, verification, reference forward | Converter / tools | `tools/convert/deepseek_v41.py`, `official_recipes.py`, `tools/artifact/expert_volume.py`, `tools/artifact/restripe.py`, `tools/deepseek_v41/` |

DeepSeek's reference code and DS4 are MIT-licensed; any code ported from them keeps its notice.

### 8.2 Shared components

Three explicit families now need host-offloaded experts: Qwen4Exp (built), GLM-5.3-Flash (designed)
and this one. The shared pieces are extracted by whichever of the two new families is implemented
first, as GLM §8.2 describes, with no base class, virtual model interface or registry:

- **`src/models/offload/`**: expert cache, residency controller, RAM tier with pinned and unpinned
  records, expert store, state file, ledger terms, route trace, VRAM monitor, parameterized by an
  `ExpertBankGeometry` (banks, experts per bank, record format, record and unit bytes, H, I).
  Qwen4Exp keeps every decision bit for bit (M2's exit criterion, GLM §23).
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
| `offloaded_sparse_moe` | Expert formats `mxfp4` and `nvfp4` with the **MX8 activation class** (beside today's A4 and A16); geometry from the bank; routing function `sqrtsoftplus_bias_topk_scaled` with a second bias for image tokens; the routing weight applied to h before the down projection; SwiGLU clamp 10; ungated shared expert in FP8; landing-aware jobs and partial records (GLM §11.4); a second bank for DSpark (128 experts, top-3) |
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
| K8 | Shared expert (FP8 gate/up [4,608 × 5,120], down [5,120 × 2,304]) ‖ routed GPU jobs (MX8 narrow, 18.8 MB each) | 35.4 MB + 18.8 MB per hit | 24 µs + 12.6 µs per hit |
| K9 | Combine (FP32, ascending expert id) with CPU and landed outputs; residual update; collapse for the next layer's attention; `attn_norm` partial sums | 0.1 MB | 2 µs + waits |

Layers 1 and 14 add the `engram` kernel before K1: `wkv` FP8 [25,600 × 6,144] (157 MB, ≈ 105 µs),
gates and the stream update. The head (BF16 [129,280 × 5,120], 1.32 GB) takes ≈ 0.88 ms.

**Per token (GPU work only, h_v = 0.48):** 40 × (≈ 105 + 24 + 2.9 hits × 12.6 + 2) µs + Engram 0.21 ms
+ head 0.88 ms ≈ **7.8 ms**, about 420 launches. Of it ≈ 4.2 ms precede routing (t_pre of §5.1).

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
2. top-6 of s_e + bias_e (bias_vl_e on image-span tokens), lower id winning exact ties;
   w_i = 1.5 · s_i / (Σ s + 1e-20) in FP32.
3. Union over the T columns; classification from the residency word and tier tables, as GLM §9.4:
   `READY` frame → GPU job; RAM slot → CPU or DMA job (§9.7); unpinned record → CPU job; landing
   slot → CPU job per unit; absent everywhere → demand read, then a landing-slot job.
4. Outputs: GPU job list, `miss_req[l]`, the route log with the scores of ranks 7-16 (near misses,
   §10.2).
5. Prediction (lookahead, off by default, §11.3): layer l + 1's router applied to this layer's MoE
   input from L2-warm weights.

### 9.5 The canonical arithmetic MX8

**Requirement.** Every FP8 and FP4 linear computes DeepSeek's MXFP8 product (decision b), and its
BF16 output is bit-identical on every route and processor: GPU narrow and wide routes, CPU on every
ISA, from a frame, a staging slot, a RAM slot, an unpinned record or a landing slot in any unit
order (FN §16.2's requirement, extended to dense layers as FN §16.2.2 did).

**Activation encoding** of a BF16 column x of K elements, per 32-element block b (the reference's
`act_quant`, §2.2):

```text
a_b  = max(max_k |x_k|, fl32(1e-4))                                   exact
e_b  = exponent(fl32(a_b · fl32(1/448))) + (mantissa ≠ 0)              integer
q_k  = e4m3_rn(clamp(x_k · 2^−e_b, −448, 448))                          exact scaling, one rounding
X_k  = q_k · 2^9 ∈ ℤ, |X_k| ≤ 229,376 < 2^18;  x̂_k = X_k · 2^(e_b − 9)
```

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

**Fidelity to the reference.** `model.py` accumulates each 32-block product on tensor cores in FP32,
scales it and adds the blocks in FP32; its result is a rounding of the exact sum S, so the two differ
by the reference's own accumulation error (relative ≲ 2^−24 per block added). §20.1 bounds it against
an FP64 evaluation of the same quantized operands.

**Why exactness is cheap here**: per-block products fit int32 for FP4 weights; the GPU narrow route
uses `dp4a` with X as three byte limbs (FN §16.2.1's narrow form); the wide route uses two
`mma.m16n8k32` per 32-block with FN §16.2.1's limb split (X has 18 bits instead of 21); the CPU uses
two int16 limbs with `vpdpwssd` (FN §16.2.2's CPU GGUF form). Dense FP8 rows take FN §16.2.2's
base-64 digit planes on the tensor cores. Rejected, as FN §16.2.2 rejected them: FP8 or FP4 tensor
cores with FP32 accumulation (undocumented order on `sm_120`, not reproducible on the CPU).

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
| Narrow (GPU) | n ≤ 8 columns for the expert (decode, verification) | CUDA-core integer, X limbs × codes via `dp4a`; work items per gate/up unit and down unit (§7.3), so an SSD record's landed units are consumed as they arrive |
| Wide (GPU) | n > 8 (prompts, wide verify) | int8 tensor cores, two k32 MMAs per block; codes decoded once per tile and reused over the CTA's token tile |
| CPU | RAM, unpinned and landed records | AVX2 + AVX-VNNI (§12) and a scalar form (the oracle) |

All three return the same bits (§9.5); the route depends only on n, never on placement.

### 9.7 Miss arbitration

- **RAM misses**: the divisor rule (L19) fed by the measured pair: the CPU team's ≈ 0.35-0.45 ms per
  expert against the link's 0.68 ms, so the CPU takes about 60 % (d ≈ 2.5, from a per-count split
  table calibrated at start and kept across starts, GLM §12.3).
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

| Tier | Capacity on this machine | Holds | Policy owner |
|---|---|---|---|
| VRAM frames (17.95 MiB) | ≈ 1,160 (7.5 %), elastic with KV, DSpark pool and the walk lease | Hot experts; staging; loans | Transfer agent |
| Pinned RAM slots | ≈ 4,000 (≈ 72 GiB, the pin room) | Warm experts; the landing ring; the demotion list | SSD agent (admission), transfer agent (promotion source) |
| Unpinned records | ≈ 450 (≈ 8 GiB) at 88 GiB available | The coldest RAM experts, CPU-served in place | SSD agent |
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
from the landing slot). A promotion moves 18.8 MB, 6.8× Flash-Next's, so the pace is re-swept in M5
({1, 2, 4, 8} per layer call until the frames fill, then every N tokens), with GLM §10.2's admission
threshold (m ∈ {0, 2, 4, 6} aged routes) and free-frame refill after a prompt. No promotions after a
walked prompt (L6).

### 10.3 Epochs, staging and loans

Unchanged from FN §9.2-§9.5 (frame epochs, slack frames, KV loans, the elastic VMM pool, display
headroom). Staging slots are frames with a staging role, at most 32 (0.56 GiB).

### 10.4 RAM tier

- **Sizing**: FN §19.3.7's ledger: available RAM at startup − `--ram-headroom-mib` (2 GiB) −
  embedding (1.23 GiB pinned) − Engram row cache (0.26 GiB) − prefix Host tier (§18) − Program
  buffers − 1 GiB growth − 1.5 GiB of process memory. The expert share is pinned up to the pin room
  (shared system memory − NON_LOCAL usage − 1 GiB, ≈ 72 GiB of experts) and unpinned past it; the
  landing ring (64 slots, 1.12 GiB) and 32 bounce slots come out of the pinned part.
- **At 88 GiB available**: ≈ 79.8 GiB of experts → ≈ 4,000 pinned slots and ≈ 450 unpinned records
  (29.2 % of the bank with the ring excluded). **At 79 GiB**: ≈ 26 %. **With other sessions holding
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

The 384 DSpark records form their own bank. The drafter never waits for them (§17.2): a bounded VRAM
pool (default 96 frames, 1.68 GiB, carved from the frame pool; M7 sweeps 0-192) ranked by LFRU
weighted by acceptance, and a RAM pool (default 192 slots, 3.4 GiB, taken from the tier only while
DSpark is on).

---

## 11. Multi-SSD expert streaming

GLM §11 applies in full: one to four volumes, read-only, every record read uses every volume, demand
reads never queue behind more than ≈ 0.3 ms of other reads per drive, failed reads fail only the
rounds that need them, IOCP on Windows and `io_uring` or a `pread` pool on Linux behind Core's
`StripedRecordReader` with one `DirectReadQueue` per volume. What changes:

### 11.1 Reading a record

A record read is 12 unit reads of 1,568,768 B issued at once, landing at their offsets in one
contiguous 17.95 MiB landing slot. With w = (4, 4, 4) on `D:` + `E:` + `G:` each drive serves four
units; at the DMI-bound ≈ 4.7 GB/s per drive the record completes in ≈ 1.33 ms and its first gate/up
units in ≈ 0.35 ms (model; R0 measures). Gate/up units are issued first, so CPU and GPU consumers
start on them while the down units are in flight (GLM §11.4).

### 11.2 Depth and priority

- **Per-volume caps**: demand 6, prompt stream 8, background 1, total 10 (1.5 MiB units between the
  1 MiB/QD4 and 2 MiB/QD2 knees of GLM §3.3; §11.5 calibrates).
- **Priorities**: Engram row reads on the Engram volume's drive (`E:`) > demand (current layer) >
  prompt stream > lookahead > background refill. A 4 KiB Engram read queued behind four 1.5 MiB units
  would wait ≈ 1.3 ms; it is issued ahead of queued units instead (§14.7).

### 11.3 Lookahead

Off by default (`--ssd-lookahead off`): every measurement on this family lost (JigSawPT −7 to −19 %,
hipfire 0.86-0.93×; §4.2), as Maya's did on GLM. With SSD-bound decode the drives are busy ≈ 90 % of
a token, so lookahead can only use the remaining ≈ 10 % (t_pre gaps between layers) and costs
bandwidth when wrong. GLM §11.5's budgeted design is built in M5 and measured on three volumes; it is
enabled only if it gains.

### 11.4 Placement on this machine

| Configuration | Bandwidth (§3) | Record time | Free space needed per drive | Advice |
|---|---:|---:|---:|---|
| **`D:` + `E:` + `G:`, w = (4, 4, 4)** | 14.15 GB/s | 1.33 ms | 98.8 GB | **Default.** `E:` also holds the Engram volume and the repository (builds can add tail latency; §11.5 warns) |
| `D:` + `E:` + `G:`, w = (5, 4, 3) | 14.15 | 1.41 ms (+6 %) | 123.5 / 98.8 / 74.1 GB | When `G:` is short of space |
| `D:` + `G:`, w = (6, 6) or (7, 5) | 12.73 | 1.56-1.73 ms | 148-173 GB on `D:` | `G:` cannot hold half the bank (108.7 GB free); (7, 5) needs 123.5 GB on `G:` |
| `C:` + `D:` + `E:` + `G:`, w = (3, 3, 3, 3) | 19.27 | 1.03 ms (−23 %) | 74.1 GB | Fastest; needs 74 GB on the system drive (80.5 GB free today): opt-in only after freeing space there |
| One volume | 5.9-7.0 | 2.7-3.2 ms | 296.4 GB | Supported; ≈ half of three-volume decode |

### 11.5 Calibration, failures and integrity

GLM §11.7-§11.8 unchanged: a startup read of 64 units per volume at QD 1 and QD 8, the physical-disk
and root-port topology check with warnings, a retry then a recoverable round failure on read errors,
header and sampled-CRC checks at startup, a full offline check in `verify_artifact`, no per-read
hashing.

---

## 12. CPU expert engine

### 12.1 Role

The CPU computes about 60 % of the RAM misses, every unpinned record's expert, and every SSD miss
within the job cap, in place, with MX8: the same bits as the GPU.

### 12.2 Kernels and cost on the i9-13900K

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
the replay's dense work, taking most of ≈ 6,600 expert transfers per prompt off the link (§16.4). Any
column count is allowed, since every MX8 route is exact (GLM §12.4's argument); the cut per layer
comes from the calibrated cost per expert-row.

---

## 13. CSA2 attention and the hierarchical indexer

No `sm_120` kernels exist for either: FlashMLA's sparse kernels are SM90/SM100 only and vLLM's
`sm_120` route has open blockers for V4.1's shapes (§4.2). Both are new Infernix Ops.

### 13.1 Per-sequence state

| State | Layers | Per token | At 128K | Stored form |
|---|---|---:|---:|---|
| Window ring | every layer (and DSpark's over main features) | bounded: 128 × 528 B per layer | 2.7 MB (+0.2 MB) | E4M3 codes + UE8M0 per 32 |
| Compressed entries | Full layers 2, 8, 14 (ratio 2) and 20 (ratio 1) | 3 × 144 + 288 = 720 B | 94 MB | E2M1 + E4M3 per 16, 288 B per entry, paged |
| Index keys | the same four layers | 3 × 34 + 68 = 170 B | 22 MB | E2M1 + UE8M0 per 32, 68 B per key, paged |
| Open compressor groups | 2, 8, 14 | — | 24 KB | FP32 kv and gate score of the incomplete pair |

The persistent part is **890 B per token**, the report's figure.

### 13.2 Compressor and index keys (`csa_compressor`)

- **Ratio 2** (layers 2, 8, 14): per token, k_i = W_ck x_i and z_i = W_cg x_i as FP32 (exact BF16
  products, one rounding), held in the open group; when a token closes a pair,
  c = Σ_i softmax_i(z_i) ⊙ k_i in FP32 (canonical exp, fixed order), rounded to BF16, RMSNorm; the
  index key comes from this pre-RoPE c; RoPE at the pair's first position; FP4 packing (amax per 16
  with the floor 6 · 2^−9, s = e4m3_rn(amax / 6), codes E2M1 round-to-nearest-even of value / s).
- **Ratio 1** (layer 20): c = W_ck x as a BF16 linear, then the same norm, key, RoPE and packing.
- Layer 20's work runs for **every prompt token** (§16.2): it is the decoder's KV.
- The packing rules are deterministic functions of c, exhaustively checked between the CPU form and
  the GPU form.

### 13.3 Decode attention (`csa_attention`)

Per layer and token: Q is 64 × 512 BF16; the entries are the window ring (≤ 128) and the selected
compressed entries (≤ 512), each one 512-vector used as key and value.

- Entries decoded to BF16 in the tile loader (E4M3 · 2^k and E2M1 · E4M3 values are exact in BF16).
- Scores q · e in FP32 with a fixed reduction order, × 512^−½; an online softmax in FP32 with the
  canonical exp and the sink's exp(sink_h − m) in the denominator; P × V with P in FP16 and FP32
  accumulation, Infernix's exact attention convention (model-quality §6.1; the reference casts P to
  BF16, §20.4).
- ≤ 640 entries per query: one CTA per group of heads with a fixed merge order; ≈ 84 MFLOP and
  0.2 MB per layer and token.
- Output: inverse RoPE on o's last 64 dimensions at the query position (FP32, rounded to BF16), then
  MX8 per group into `wo_a` (K6).
- Reuse layers take the latest Full or Reindex layer's selection.

### 13.4 The indexer (`csa_indexer`)

| Step | Precision |
|---|---|
| q_h = MXFP4(RoPE(W_qb_idx c_q)), w_h = (W_w x)_h · 128^−½ · 32^−½ | FP8 linear with MX8 input; BF16 `weights_proj` as the reference |
| q_h · k_p per head and position | **exact**: E2M1 × E2M1 products, integer sums per 32-block, power-of-two scales |
| score_p = Σ_h w_h · relu(q_h · k_p) | FP32, heads in ascending order |
| Ratio-2 Full layers: top-512 of the visible positions | exact radix select, lower position on ties, re-sorted by position |
| Layer 20: block maximum over 8 positions, the newest block pinned, top-2,048 blocks, then top-512 inside | same |
| Reindex: its own scores inside the candidates, top-512 | same |

- **Deviation.** `model.py` computes q · k with a BF16 einsum, the weighted product and the head sum
  in BF16, which rounds scores into ties that `torch.topk` breaks arbitrarily. Infernix keeps FP32
  logits, as the training-time indexer kernels do (DeepGEMM's indexer returns FP32 logits), and
  reports selection agreement against the reference's BF16 scores (§20.1).
- **Decode cost**: layer 20 scans every position (68 B each: 8.9 MB at 128K, 68 MB at 1M), the
  ratio-2 Full layers half of them, the Reindex layers 16,384 candidates (1.1 MB).
- Selections depend only on the stored keys and the query (FN §8.10's rule).

### 13.5 KV formats

The model defines and was trained with its KV formats (report §2.4.4): FP8 window entries (528 B),
FP4 compressed entries (288 B, an E4M3 scale per 16, no global scale), MXFP4 index keys (68 B).
Infernix stores exactly these codes. `--kv-dtype` accepts only `model` for this family; any other
value is refused at startup. KV at 128K: 117 MB; at 1M: 0.89 GB; windows: 2.9 MB per sequence.

### 13.6 Prompt kernels

- **Window attention**: per query the last 128 positions (causal in the chunk plus the ring), query
  tiles of 64 sharing their windows.
- **Compressed attention**: each query's own 512 selections; a tile of 64 queries processes the union
  of its selections with a per-query mask (FN §19.3.8 F5's QSA prompt kernel).
- **Indexer scoring**: a register-tiled exact int8 MMA over queries × positions per head, query
  sub-chunks of 2,048 (GLM §13.3) and a streaming top-k, never materializing the score matrix. At
  128K the three ratio-2 layers score ≈ 17 T multiply-adds each (≈ 0.1 s on int8 tensor cores,
  estimated); layer 20 scores only the decoder's positions.
- **Attention arithmetic**: ≈ 84 MFLOP per query and layer, ≈ 1.1 s for a 128K prompt over the
  encoder (estimated).

### 13.7 Long context

Default `--max-context 131072`, up to 1,048,576. At 1M the compressed planes take 0.89 GB of VRAM
and the windows are unchanged: no host KV tier is needed (unlike GLM §13.6). The decode indexer scan
grows to ≈ 170 MB per token at 1M (≈ 0.11 ms); prompts grow quadratically in the indexer (§5.5).

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
- The **multipliers** are `numpy.random.default_rng(10007 · layer_id)` draws; the converter stores
  them, so Infernix never reimplements NumPy's generator. Primes and offsets are stored and checked
  against `engram_num_embeddings` (each table's 24 primes sum to its row count).
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

- The artifact's `engram_table.format` records its own table: `fp8_e4m3_e8m0_32` (official sources)
  or `mxfp4_e8m0_32` (NVIDIA's current revision, LibertAIDAI).
- **`--engram-volume PATH` replaces the table at run time with a volume of either format**, as
  `--ngram-volume` does for Flash-Next (FN §12.5): an FP8 volume in place of an MXFP4 table restores
  DeepSeek's own table; an MXFP4 volume in place of an FP8 table halves the volume, a lossy change
  that the startup line and the request log label.
- `--engram-reuse PATH` binds an existing volume at conversion after checking sampled rows (equal FP8
  rows, or FP8 rows that MXFP4-quantize to the artifact's codes by ModelOpt's rule);
  `verify_artifact --engram-rows all` checks every row.
- Row decode on the device: FP8 rows bf16(e4m3 · 2^(u−127)), MXFP4 rows bf16(e2m1 · 2^(u−127)), both
  exact.
- `--engram-residency auto|ram|volume`: `ram` only when the whole table fits beside the planned
  expert tier, which never happens on this machine (97-195 GiB), so `auto` reads the volume.
- The prefix cache's fingerprint names the table format, so blocks computed with one table are never
  reused with the other (FN §12.5).

### 14.6 The `engram` kernel

Per Engram layer and token: the 24 rows decoded to BF16, MX8 per 32 over the 6,144 values, `wkv` FP8
[25,600 × 6,144] (157 MB), FP32 keys and value, per-stream rstd and gate (signed square root with the
1e-6 clamp, sigmoid), and the stream update with one rounding to BF16: ≈ 0.11 ms per layer per token,
memory-bound on `wkv`.

### 14.7 Cost and exposure

- **Decode**: 48 rows per token, ≈ 196 KB of 4 KiB reads (≈ 2.5 MB/s at 13 tok/s): no bandwidth to
  speak of; the latency is hidden (§14.4) as long as Engram reads never wait behind expert units,
  hence their priority.
- **Prompts**: ≤ 1.57 M reads per 32K before deduplication; at an estimated 0.5-1 M random reads per
  second from one 990 PRO (R0 measures the Windows ring at QD 64), ≈ 1-3 s, overlapping the bank pass.

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
  next mixes.
- **Sinkhorn**: one warp in FP32, one column normalization then 19 row/column rounds with eps 1e-6,
  reproduced iteration for iteration.
- **Cost**: ≈ 2 µs per sublayer, ≈ 0.16 ms per token, none of it on the critical path except the
  epilogues.

### 15.2 Numerics

- Mixing weights, Sinkhorn, pre, post and comb in FP32; streams in BF16; the update in FP32 with one
  rounding, exactly as the reference (GLM's reference rounded twice; this one does not).
- Fixed orders: s = 0..3 for the collapse, j = 0..3 for the comb sum, then + post · y; qualified
  against FP64 with the reference's single rounding.
- The first sublayer's pre is one-hot on stream 0; the head collapses with the last FFN's pre.

---

## 16. Prompt processing: the encoder walk and the decoder rule

### 16.1 The two rules (decision a)

| `--prompt-decoder` | Prompt tokens through the encoder | Decoder positions evaluated | Equals |
|---|---|---|---|
| **`replay`** (default) | all | the last min(T, 128), window attention truncated at the replay start | DeepSeek's served model (report §3.2.2; vLLM #56752; SGLang) |
| `full` | all | the exact suffix: layer l evaluates the last 1 + (39 − l) · 127 positions after forming the 127 window entries before them | the reference forward (every prompt token through all 40 layers), output for output (DS4's method) |

- Both rules are deterministic and request-invariant (§18.3); the rule is part of the prefix cache's
  fingerprint, the startup line and the request log.
- The rule is a declared model mode, chosen by the owner, not a fast path: `--disable-fast-paths`
  does not change it, and each rule has its own exact route.
- **CausalScoring runs the full rule.** Replay defines the state at the end of a prompt, not a logit
  at every position; scoring each position under replay would cost a decoder replay per position.
  CausalScoring therefore computes every position through all 40 layers, as the reference does, and
  says so in its output.

### 16.2 The encoder walk

- Layer-major **walk spans** (FN §19.3.8 F4) over layers 0-19, plus layer 20's prefix work for every
  prompt token: the collapse, `attn_norm`, the compressor and the index keys, which are the decoder's
  KV (SGLang #42901 trims the rest of layer 20 the same way).
- **Spans up to 128K tokens** (Flash-Next's `kWalkMaxTokens` is 65,536; this family's KV is small
  enough to double it): the walk's stream state is 40 KB per token (4 × 5,120 BF16), 5.2 GB at 128K,
  lent from frames (FN F4's lease); the decoder and the head run after the span that holds the
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
  min(T, 128) positions as narrow work (≤ 128 columns), window attention truncated at the replay
  start s (entries max(s, i − 127) … i), compressed entries from layer 20 for every position. When a
  prompt continues a cached prefix with T < 128, the replay reaches into cached positions; their
  layer-20 inputs come from the snapshot (§18.2).
- **full**: layer l evaluates its suffix; layer 20 needs layer-20 inputs for the last 2,541 positions,
  from the walk or, for cached positions, from the snapshot. A prompt shorter than a layer's suffix
  runs that layer on all of its tokens.
- Replay's decoder experts are thin (≈ 2 tokens each): the CPU computes the non-VRAM ones (§12.4).
  The full rule's are 2-40 tokens each from layer 39 to layer 20.
- DSpark's window seeds (main-feature entries of the last 128 positions) come from the decoder
  positions either rule evaluates.

### 16.4 Short appends

An append of T tokens to a cached conversation runs the same walk over T tokens; only the experts
those tokens touch are read (54 % of the bank at 50 tokens, §5.5), the narrow routes serve experts
with n ≤ 8 columns, and the CPU takes thin non-resident ones. The guide states the cost plainly:
2-9 s for 16-100 tokens on three volumes (model).

### 16.5 After prefill

No promotions after a walked prompt (L6); the prompt's routes seed both policies; when the walk
returns its lent frames, the free-frame refill (§10.2) fills them with the hottest RAM experts.

---

## 17. Speculative decoding: DSpark

### 17.1 Expected value

- Every verify column brings its own experts (§5.4): short drafts can pay (K = 1: +6-27 %, model),
  DeepSeek's five-token blocks lose on this machine; JigSawPT measured −4 % / ±0 at K = 2 on the same
  card (§4.2).
- DSpark's confidence head predicts each draft position's acceptance, which with the measured union
  growth is what the cost model needs each round.

### 17.2 The drafter

- Structure §2.2. Its window of main-feature entries (3 layers × 128 × 528 B) is appended once per
  committed token and seeded by the prompt's decoder positions.
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

### 17.4 N-gram copy proposals

`--ngram-draft-tokens` through the same cost model; they need no drafter or DSpark frames.
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
| Encoder windows (layers 0-19) and open compressor groups | 1.35 MB + 24 KB | resume the encoder exactly; DeepSeek's lossy encoder replay is not needed |
| Engram history | 12 B | the n-gram state |
| DSpark window (main-feature entries) | 0.2 MB | when DSpark is on |
| `replay`: layer-20 inputs of the last 127 positions | 5.1 MB | a short append's replay reaches into cached positions |
| `full`: layer-20 inputs of the last 2,540 positions | 104 MB | the exact suffix of a short append |

A snapshot is ≈ 6.7 MB under the default rule and ≈ 106 MB under the full rule. Decoder windows are
never in a snapshot: replay recomputes them by definition, and the full rule recomputes them through
its suffix.

### 18.3 Request invariance

Both rules make a request's output a function of the request, the artifact and the options only
(model-quality §6.1): the encoder runs on the fixed chunk grid with exact arithmetic; the replay's
decoder state depends only on the prompt's last 128 tokens and the encoder state, the suffix's on the
last 2,541; a snapshot restores exactly what a cold run computes at the same grid point. As the engine
requires (engine architecture §3.1), only prefill-computed state is published, on the chunk grid.

### 18.4 Preemption

The exact route never pauses a request: the Program reserves its whole KV extent (engine architecture
§3.1). Here that extent is 890 B per token, so reserving a 128K request costs 117 MB; the `preemption`
fast path (Replay recovery) is not registered for this family.

### 18.5 Tiers

Host tier default 2 GiB (`--host-context-mib`; ≈ 160 default-rule snapshots per GiB), and the file
tier as the spec defines it. DeepSeek keeps persistent KV on SSD for at least 72 hours (report
§3.2.1); at 29 MB per 32K conversation the file tier can keep many agent sessions for days.

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
| Decode workspace, mailboxes, residency tables, io | 0.70 | 0.70 |
| Staging (32 slots) | 0.56 | 0.56 |
| CUDA graph reserve; headroom | 0.25; 0.25 | 0.25; 0.25 |
| DSpark expert pool (96 frames, §10.6) | — | 1.68 |
| **Frame pool** (of the 30.68 GiB budget) | **20.32** | 17.99 |
| Frames of 17.95 MiB | **1,159** (7.5 % of 15,360) | 1,026 |

At 1M actual context the KV takes 0.72 GiB more (−41 frames). A further lane costs its windows and KV
extent only (≈ 3 MB + 890 B per token): there is no recurrent state.

### 19.2 Host RAM (95.8 GiB visible)

| Item | GiB |
|---|---:|
| RAM expert tier: pinned up to the pin room, unpinned past it; includes the 64-slot landing ring (1.12) and 32 bounce slots (0.56) | the rest |
| Token embedding, pinned | 1.23 |
| Engram row cache | 0.26 (FP8 rows) / 0.13 (MXFP4) |
| Prefix Host tier | 2.0 (default) |
| Program pinned buffers | 0.25 |
| Process (tokenizer, server, frontend) | 1.5 |
| Reserve `--ram-headroom-mib`; growth allowance | 2.0; 1.0 |
| DSpark RAM pool (with DSpark only) | 3.37 |

| Available at startup | Expert tier | Pinned / unpinned | Slots (ring excluded) | Share of 15,360 | With VRAM frames |
|---:|---:|---:|---:|---:|---:|
| 88 GiB (other programs closed) | 79.8 GiB | ≈ 72 / 7.8 GiB | 4,485 | 29.2 % | 36.7 % |
| 79 GiB | 70.8 GiB | 70.8 / 0 | 3,972 | 25.9 % | 33.4 % |
| 50 GiB (other sessions running) | 41.8 GiB | 41.8 / 0 | 2,318 | 15.1 % | 22.6 % |

Every GiB other programs release becomes ≈ 57 more slots; the startup ledger prints the split.

### 19.3 Disk

| Item | Size | Where (default) |
|---|---:|---|
| Source shards during conversion | ≤ 2 × 7.4 GB at a time | `D:` (deleted shard by shard, §7.6) |
| Main artifact | ≈ 11 GB | `E:` |
| Expert volumes, w = (4, 4, 4) | 3 × 98.8 GB | `D:`, `E:`, `G:` |
| Engram volume (FP8 / MXFP4) | 209.7 / 104.9 GB | `E:` |
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
| MX8 codec (`ops/common/mx8.h`) | **Exact, exhaustive** over every BF16 input and block exponent: the CPU integer form, the GPU `cvt.rn.satfinite` form and a Python port of `act_quant` agree, including the 1e-4 floor and the ceiling rule at powers of two |
| `fp8_e4m3fn_b32_e8m0` Linear, decode and prompt routes | **Bit-exact** between routes and against the scalar form at T = 1..8 and prompt widths, every DeepSeek shape including grouped `wo_a`; column invariance for T ≤ 8; FP64 oracle of the same quantized operands within the FP32 rounding bound |
| `offloaded_sparse_moe`, MXFP4 and NVFP4 records, MX8 | **Bit-exact** across GPU frame, staging, CPU on every ISA, unpinned record, every unit landing order, n = 1..8 and the wide route; MXFP4 and NVFP4 records of one expert give the same bits; routing exact against an FP64 √softplus/bias/top-6 oracle on near-tie-free inputs, ties by lower id, logits beyond ±20; clamp boundaries (±10, NaN, Inf); the routing weight before the down projection |
| `csa_compressor`, index keys | FP64 pooling and norm; FP4 and MXFP4 packing exact against a Python port of `fp4_act_quant`, exhaustive over block exponents |
| `csa_indexer` | exact q · k against integer arithmetic; FP64 weighted sums; exact selection on near-tie-free fixtures; selection agreement on real activations, also against the reference's BF16 scores (reported) |
| `csa_attention` | FP64 softmax attention with the sink over the decoded entries; replay's window truncation; inverse RoPE |
| `mhc` | FP64 including all 20 Sinkhorn rounds; one-rounding stream update |
| `engram` | hash ids exact against `engram.py`; row decode exact for both formats; the block against FP64 |
| DSpark attention, Markov and confidence heads | FP64 |
| Records and volumes | MX recovery from NVIDIA's NVFP4 equals the official MXFP4 bytes on the whole September bank; every unit read through the striped reader equals the converter's bytes, in permuted completion orders and at fault seams; Engram volumes equal the source rows (FP8) or ModelOpt's MXFP4 rows |

### 20.2 Reference forwards and block checks

`tools/deepseek_v41/reference_forward.py` (a qualification tool, not a product route): teacher-forced
and layer-major, streaming each layer's weights from the converted artifact through an independent
Python decoder and Engram rows from the volume, with FP64 accumulation over `model.py`'s quantized
operands (MX8 activations, FP8 and FP4 KV entries, BF16 roundings where `model.py` has them). It has
both prompt rules. `block_check.py` compares each layer type, the head and the DSpark blocks with an
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
| **Execution**, each prompt rule | the reference forward of the same rule | ≤ 0.001 nats | ≤ 0.03 | ≥ 99.5 % | ≤ 0.1 % |
| MXFP4 records against NVFP4 records | each other | bit-identical | | | |
| Official source against NVIDIA source (September, FP8 Engram) | each other | bit-identical | | | |

The execution gate is tighter than GLM §20.3's because Infernix computes the reference's own
quantized arithmetic; what remains is the accumulation order and the indexer's score precision
(§20.4).

**Reported without a gate** (the owner's choices): replay against full (KL and top-1 on the same texts,
and the official fixtures' NLL under each); MXFP4 Engram tables against FP8 (KL, top-1, perplexity),
beside diffbot's 0.035 for vLLM's W4A4 + MXFP4-Engram serving; and, for the September checkpoint,
Infernix's NLL on the official fixtures beside DS4's Q2 and Q4 references (§4.1).

### 20.4 Deviations from `inference/model.py`

| Point | `model.py` | Infernix | Why |
|---|---|---|---|
| FP8/FP4 linear accumulation | FP32 tensor-core products per 32-block, FP32 block adds | exact sums, one rounding | placement invariance; more precise |
| Indexer scores | BF16 einsum, BF16 product and head sum | exact q · k, FP32 weighted sum | the training kernels' FP32 logits; fewer ties; agreement reported |
| Top-k ties | `torch.topk` (unspecified) | lower position or lower expert id | determinism |
| Attention P × V | BF16 P | FP16 P, FP32 accumulation | Infernix's exact attention convention; FP64-qualified |
| softplus, exp, SiLU, sigmoid | torch CUDA | canonical implementations | CPU/GPU identity |
| mHC sums, norms | torch's order | fixed orders | determinism |

Every BF16, FP8 and FP4 rounding point of `model.py` is kept.

### 20.5 Engine and system tests

- **Placement invariance end to end**: greedy ids identical with `--expert-frames 0`, small and full
  VRAM caches; RAM tier caps of 16, 48 GiB and full; unpinned records forced; one, two and three
  volumes; lookahead off and on; every CPU ISA; random unit landing order; CPU cap 0 and 16.
- **Record formats**: an MXFP4 and an NVFP4 artifact of one checkpoint give identical ids.
- **Prompt rules**: replay is identical across chunk sizes and prefix-cache states; the full rule
  matches the reference forward's full-prompt logits within the execution gate; replay matches the
  reference's replay variant.
- **Prefix cache**: cold against warm resume at every tap point under both rules.
- **Speculation**: greedy speculative output equals plain greedy output (verify widths ≤ 8 columns are
  column-invariant, FN §11.3).
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
    `<｜Assistant｜></think>`; earlier turns drop their thinking unless tools are declared;
  - the reasoning effort (1-100; low / high / max = 50 / 75 / 100; default high) as the encoding's
    line at index 0 in thinking mode;
  - tools under `## Tools` in the system prompt, `response_format` under `## Response Format`;
  - tool calls in DSML (`<｜DSML｜ calls>`, `invoke name=`, `parameter name= string=`), results as
    `<tool_result>` blocks inside user turns in call order.
- **Protocol mapping.** OpenAI `reasoning_effort` low / medium / high → 50 / 62 / 75 (62 is the
  September template's alias), a numeric 1-100 extension, `none` → chat mode; the Anthropic thinking
  budget → effort by a documented table. Thinking off is native here (unlike GLM).
- **Tool calls.** A `dsml` parser with tolerant recovery and a constrained-decoding grammar (M9 checks
  whether xgrammar's `converter_ext/deepseek.cc` covers V4.1's leading-space tags). This is advertised
  OpenAI and Anthropic behaviour, so the schema tests and the serving documentation change together
  (AGENTS.md).
- **Images.** Refused for this family until the vision track: the encoding's image tokens are
  recognized and rejected with a clear error.

---

## 22. Configuration surface

| Option | Default | Meaning |
|---|---|---|
| `--prompt-decoder replay\|full` | `replay` | Decision a (§16); **changes output**; `full` equals the reference forward |
| `--expert-volume PATH` (repeat) | required | The artifact's expert volumes, any order; startup checks ids, sizes and headers |
| `--engram-volume PATH`, `--engram-residency auto\|ram\|volume` | the artifact's own; auto | Decision d (§14.5); a volume of the other format **changes output** and is labelled |
| `--expert-ram-mib N\|auto`, `--expert-pin-mib N`, `--ram-headroom-mib N` | auto, auto, 2048 | RAM tier (FN §19.3.7) |
| `--vram-headroom-mib N\|auto`, `--expert-state FILE\|off` | auto, beside the artifact | As FN |
| `--ssd-lookahead off\|1\|2\|3` | off | §11.3 |
| `--cpu-experts auto\|off\|N`, `--cpu-expert-cores p\|pe` | auto, auto | §12.3 |
| `--spec off\|dspark\|auto`, `--dspark-expert-frames N\|auto`, `--draft-tokens auto\|N` | off until M7 | §17 |
| `--ngram-draft-tokens N` | off | Copy proposals |
| `--kv-dtype model` | `model` | The only accepted value (§13.5) |
| `--max-context N` | 131072 | Up to 1,048,576 |
| `--prefill-chunk N`, `--prefill-group N` | auto, 48 | Chunking; experts per prompt group |
| `--host-context-mib N` | 2048 | Prefix Host tier |
| `--reasoning-effort 1..100\|low\|high\|max\|off` | high (75) | Request default; requests may override |
| `--approx-experts off\|ssd\|vram`, `--approx-keep-map`, `--approx-min-weight θ`, `--approx-experts-rule residency\|static` | off | §9.8; **changes output** |

Converter: `--recipe deepseek_v4_1_flash` (or an official entry, §7.1) with `--source REPO@REV`
(streamed shard by shard, §7.6) or `--source DIR`; `--expert-volumes P0,P1,…` and
`--expert-volume-weights w0,w1,…` (default chosen from free space and bandwidth); `--engram-volume PATH`,
`--engram-format fp8|mxfp4` (default: the source's), `--engram-reuse PATH`;
`--expert-record mxfp4|nvfp4|auto`; `--staging DIR`; `--plan` (print the disk plan and stop). Tools:
`tools.artifact.restripe`, `tools.deepseek_v41.verify_artifact`, `tools.deepseek_v41.reference_forward`,
`tools.deepseek_v41.block_check`, `tools.deepseek_v41.decode_model`; `infernix-calibrate` gains the
striped-read, Engram-read and CPU MX8 probes.

---

## 23. Implementation plan

| # | Milestone | Deliverables | Exit criteria |
|---|---|---|---|
| **M0** | Facts | September-vs-1017 comparison of tensors and Engram rows (range reads); the official MXFP4 scale rule on sampled blocks; NVFP4 → MXFP4 recovery on sampled NVIDIA blocks; ModelOpt's MXFP4 Engram rule against NVIDIA's rows; Transformers 5.6 `deepseek_v41` against `model.py`; encoding fixtures and tokenizer files; DS4's suffix rule against `model.py`; C++ probes (`tools/deepseek_v41_probe/`): striped 1.5 MiB reads on `D:` + `E:` + `G:` through IOCP, Engram 4 KiB reads on `E:` under concurrent unit reads, CPU MX8 at 18.8 MB per expert on P- and E-cores, the pin room with this plan's buffers | §2, §3, §5 inputs and §7.1's rules confirmed or revised; §5 re-run |
| **M1** | Traces and policy | Route traces from M4's engine on NRS and a broad corpus (code, chat, CJK, tool use, long context); replay of the §10.2 and §10.4 candidates; the union growth σ and lookahead recall | Policies chosen by the fixed rules; §5.2 replaced by measured hit rates |
| **M2** | Shared components | `src/models/offload/`, expert volumes, the striped reader and the row-volume reader, unless GLM's track built them first | Qwen4Exp's tests unchanged; greedy ids identical; tg512 within noise |
| **M3** | Ops | MX8 codec; `fp8_e4m3fn_b32_e8m0` Linear; MXFP4 and NVFP4 experts with MX8 on the GPU and the CPU; router; `csa_compressor`, `csa_indexer`, `csa_attention`; `mhc`; `engram`; DSpark Ops; the MXFP4-against-NVFP4 kernel timings (decision c) | Every §20.1 qualification passes; the timing table recorded |
| **M4** | Artifact and functional model | Recipe and official entries, the streaming converter with its disk planner, MX recovery, volumes, restripe, `verify_artifact`; Engine, CLI, bench and CausalScoring with one volume; both prompt rules; the reference forwards | Every tensor round-trips; §20.2 checks; the execution gates for both rules; the §20.3 reports; first end-to-end numbers |
| **M5** | Tiers at scale | Three volumes, landing ring, partial records, M1's policies, promotion sweep, CPU sweep, unpinned records, warm start, lookahead A/B, the MXFP4 and NVFP4 artifacts end to end | §20.5 invariance and fault tests; decode Must target; three volumes ≥ 1.8× one |
| **M6** | Decode speed | §9.3's kernels and fusions, stall-time warming; nsys attribution of one token | Per-kernel times against the budget; anything > 30 % over gets ncu before M7 |
| **M7** | Speculation | DSpark drafter (bounded experts, Markov and confidence heads), the cost model with σ and confidence, n-gram proposals | Default chosen by measurement; never slower than plain beyond noise |
| **M7a** | Approximate mode | §9.8 | Mode off bit-identical to a build without it; presets meet the gates |
| **M8** | Prompts and long context | 128K walk spans, prompt kernels, the CPU decoder replay, the full rule's suffix, indexer prompt scoring, 1M context | Prompt Must targets; decode at 128K ≥ 90 % of 4K |
| **M9** | Frontend and serving | Tokenizer, encoding, DSML parser, protocol schema tests, prefix-cache binding for both rules | Tokenizer and encoding oracles exact; serving schema tests; prefix-cache real tests |
| **M10** | Report | DS4 under WSL2 and the §4.2 engines that run here, head to head; the §1.4 table filled | Every result reported, favourable or not |

Critical path: M0 → M2 ∥ M3 → M4 → M1 → M5 → M6, then M7 and M8; M9 can start now (the tokenizer
and encoding files are read). Rough size: ~30-34k new or moved lines (CSA2 Ops ~5k; MX8 codec, the
FP8-b32 Linear and the expert routes with the CPU kernel ~6k; Engram ~1.5k; DSpark ~2.5k; mHC ~1k and
volumes with the reader ~2k, shared with GLM; offload extraction ~3.5k moved; frontend ~2.5k;
converter and tools including the streaming conversion and the reference forward ~4k; tests ~8k),
≈ 6-7k fewer if GLM's track builds the shared pieces first, plus the 510 GB download and GPU time for
the campaigns.

---

## 24. Documentation and authority changes

When the implementation lands: a user guide `docs/deepseek_v4_1-flash.md` (hardware advice, the cost
of a fresh prompt, the prompt rules, Engram formats); README models and benchmarks; `docs/cli.md` and
`docs/serving.md` (expert and Engram volumes, `--prompt-decoder`, DSML, reasoning effort); a
mathematics reference `docs/maintainer/deepseek_v41-model.md` from §2.2-§2.3;
[storage layouts](storage-layouts.md) (`mxfp4_expert_v1`, `nvfp4_expert_rg16_v1` at this geometry,
the Engram row formats, `fp8_e4m3fn_b32_e8m0`); [tensor formats](tensor-formats.md) (MXFP4, MX8,
FP8 with 32 × 32 power-of-two scales); [op development](op-development.md) (MX8 qualification);
[artifact container](artifact-container.md) (expert and Engram volumes);
[engine architecture](engine-architecture.md) (the new architecture, prompt rules as a model mode,
shared offload components); [paged KV](paged-kv-cache.md) (compressed planes and windows);
[hybrid prefix cache](hybrid-prefix-cache-spec.md) (the binding and its snapshots);
[model quality](model-quality.md) (§6 for this family's configurations); the conversion guide (the
streaming conversion). This file is then removed.

---

## 25. Risks, open questions, options and rejected alternatives

### 25.1 Risks

| Risk | Effect | Mitigation |
|---|---|---|
| §5's model is optimistic, as GLM's was by 15-45 % against strata-glm | decode ≈ 9-11 tok/s central | M4's first numbers refit the overhead terms; M6 attributes one token with nsys |
| Hit rates below central (JigSawPT's 59-66 % VRAM hits came with a 125.7 GiB host) | decode toward 10 tok/s | M1 first; the policy candidates; the RAM lever of §5.6 in the guide |
| `G:` short of space when the conversion runs | slower weighting (+6 % record time) or a fourth volume | the planner (§7.6) |
| Other sessions holding RAM on this machine | a smaller tier, more SSD reads | the startup ledger; the guide states it |
| Drive thermal throttling under sustained reads; `E:` busy with builds | stalls | per-volume telemetry and warnings (§11.5) |
| Engram reads queued behind expert units | layer-1 stalls | read priority (§11.2); measured in M0 |
| MXFP4 kernels slower than NVFP4 | decision c falls back | M3 and M5 timings; NVFP4 records built anyway |
| Some NVIDIA NVFP4 blocks not MX-recoverable | that source stores NVFP4 records | per-bank fallback (§7.1) |
| Router near-ties (JigSawPT: the 6th-to-7th margin is about 1 %; 208 of 240 choices agreed between their runs) | differences from the reference at near-ties | exact FP32 logits, lower-id ties; agreement reported |
| Replay costs more quality than DeepSeek's GSM8K suggests | the default questioned | the §20.3 report; `--prompt-decoder full` |
| The 510 GB download is interrupted | time | the resumable shard journal |
| DSpark slower than plain | no speculative gain | off by default; K = 0 in the cost model |
| Upstream ambiguities (encoding details, Transformers against `model.py`) | wrong frontend or mathematics | M0 oracles against `encoding/tests` and `model.py` |

### 25.2 Open questions for M0

1. Did the 1017 update change the Engram tables and the DSpark weights?
2. Do DeepSeek's MXFP4 scales follow ⌊log₂ amax⌋ − 2 on every non-zero block, so that recovery is
   byte-identical?
3. Does every block of NVIDIA's NVFP4 export recover to MXFP4?
4. Does Transformers 5.6 implement `deepseek_v41` as `model.py` does (the mHC shift, the indexer)?
5. The Windows QD-64 4 KiB read rate on `E:` with concurrent 1.5 MiB unit reads.
6. The CPU MX8 cost per 18.8 MB expert on P- and E-cores.
7. The pin room and commit limit with this plan's buffers.

Questions for the owner:

8. ~~Prompt rule, activations, records, Engram formats, the disk route~~ — decided 2026-10-19 (§1.2).
9. Whether to free ≈ 75 GB on `C:` for a fourth expert volume (+21 % decode, model).
10. When to schedule the 510 GB download and conversion; the converter's RAM and disk use are bounded,
    and it checks other sessions' RAM first.

### 25.3 Options (each behind its own measured gate)

| ID | Option | Gate |
|---|---|---|
| O1 | A fourth expert volume on `C:` | the owner frees space; M5 A/B |
| O2 | A W4A4 fast path for prompt experts (NVIDIA's activation quantization; input scales from an NVFP4 source or calibrated), ≈ 4× expert compute where a prompt is compute-bound | model-quality §6 prompt gates; never in decode |
| O3 | Draft-informed SSD prefetch: the main routers of early layers on draft tokens start reads before verification | M1 recall ≥ 0.5 for SSD-tier experts; M7 gain |
| O4 | The MXFP4 Engram volume as the default (−105 GB) | the owner's choice after the §20.3 report |
| O5 | Mirrored expert volumes with per-unit routing (GLM O8) | M5 A/B; twice the disk |
| O6 | Import DS4's Q2/Q4 GGUFs (requantized experts, about half or the same bytes) with exact GGUF arithmetic | an owner request; quality against the official continuations; IQ2_XXS kernels for Q2 |
| O7 | Vision (DeepSeek-ViT, 0.97 GB) | a later track |
| O8 | Lookahead on three volumes | M5 gain |
| O9 | Linux `io_uring` path for the striped and row readers | Linux users; the same tests |

### 25.4 Rejected alternatives

| Alternative | Reason |
|---|---|
| FP8 or FP4 tensor cores with FP32 accumulation as the arithmetic | Not reproducible on the CPU or across routes (FN §16.2.2) |
| BF16 activations (A16) as the exact route | Decision b: DeepSeek's trained MXFP8 forward is the model; Infernix's A16 precedent (FN §16.2.2) was for checkpoints trained in BF16 |
| NVIDIA's W4A4 as the exact route | Served that way it measured KL 0.035 against a 0.016 floor (with MXFP4 Engram, §4.2); the 1017 release has no calibrated input scales |
| Requantizing the experts at import (Q2, Q4, EXL3) in the default | Changes the model; DS4 measured +0.12-0.19 nats NLL for Q2 against Q4 |
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
- Infernix: the [Flash-Next design](qwen3_8-flash-next-design.md), the [GLM-5.3-Flash design](glm5_3-flash-design.md)
  (including its storage measurements of 2026-10-10), [model quality](model-quality.md) §6 and
  [engine architecture](engine-architecture.md).
