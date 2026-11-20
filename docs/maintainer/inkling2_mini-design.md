# Inkling 2 Mini on one RTX 5090: design

**Status: design proposal, not implemented.** Nothing in this document is built. It is a temporary planning
document for adding `Inkling2LoopForConditionalGeneration` (Thinking Machines' Inkling 2 Mini, model type
`inkling2_loop`) to Infernix. When the implementation lands, its stable content moves into the active
authorities named in [§22](#22-documentation-and-authority-changes) and this file is removed. A new mathematical
architecture is a product change ([AGENTS.md](../../AGENTS.md)): this design does not authorize implementation,
and its first open decision ([§23.1](#231-owner-decisions)) is whether the family is added at all.

Labels used for numbers: **upstream** (read from the checkpoint, its configs or another engine's code or
reports), **measured** (on this machine, with the method named; most are Infernix's own records on other models),
**model** (derived from upstream or measured inputs by the stated arithmetic), **estimated** (judgement; uncertain
by the stated range). A milestone in brackets replaces a model or estimate with a measurement.

The model was released on 2026-11-20. Its reference code was then in open pull requests: the looped model in
Transformers (huggingface/transformers#51050), CUDA implementations in vLLM (vllm-project/vllm#69028) and SGLang
(sgl-project/sglang#51926), and TokenSpeed's (lightseekorg/tokenspeed#3682, built on its merged Inkling 2 Small
support). Every "upstream" fact comes from those sources on that day; [§24](#24-sources) says which were read and which
only skimmed. M0 re-reads the merged versions.

The target machine is the project's: one RTX 5090 (32 GB, `sm_120a`, ~1.79 TB/s nominal and ~1.67 TB/s measured
read rate, 170 SMs, ~99 KB shared memory per CTA and about the same per SM, 96 MB L2), PCIe 5.0 x8, Windows 11.
The whole model is resident in VRAM; nothing is offloaded.

---

## Contents

1. [Goals, baselines and acceptance](#1-goals-baselines-and-acceptance)
2. [Model facts](#2-model-facts)
3. [Other implementations and Infernix's own record](#3-other-implementations-and-infernixs-own-record)
4. [Performance model](#4-performance-model)
5. [Design overview](#5-design-overview)
6. [Ownership in Infernix](#6-ownership-in-infernix)
7. [State, KV and the prefix cache](#7-state-kv-and-the-prefix-cache)
8. [Decode pipeline](#8-decode-pipeline)
9. [Attention Ops](#9-attention-ops)
10. [MoE Ops](#10-moe-ops)
11. [Dense projections and the head](#11-dense-projections-and-the-head)
12. [Residual mixing, message gates and short convolutions](#12-residual-mixing-message-gates-and-short-convolutions)
13. [Speculative decoding: DSpark](#13-speculative-decoding-dspark)
14. [Prefill](#14-prefill)
15. [Vision](#15-vision)
16. [Numerics and qualification](#16-numerics-and-qualification)
17. [Frontend and serving](#17-frontend-and-serving)
18. [Artifact and conversion](#18-artifact-and-conversion)
19. [Configuration surface](#19-configuration-surface)
20. [Memory plans](#20-memory-plans)
21. [Implementation plan](#21-implementation-plan)
22. [Documentation and authority changes](#22-documentation-and-authority-changes)
23. [Open decisions, risks and rejected alternatives](#23-open-decisions-risks-and-rejected-alternatives)
24. [Sources](#24-sources)

---

## 1. Goals, baselines and acceptance

### 1.1 Deliverable

The product supports `Inkling2LoopForConditionalGeneration` through the public Engine, `infernix`,
`infernix-serve`, `infernix_bench`, the launcher and offline CausalScoring, with:

- the text model: 24 physical layers, 48 layer visits per token, every weight as the checkpoint stores it
  (NVFP4 and MXFP8 experts bit-exact; BF16 matrices packed losslessly at load where faster, as Flash-Next's are);
- the model's own expert arithmetic: NVFP4 experts at 4-bit activations with the per-token four-over-six rule,
  MXFP8 experts at MXFP8 activations, as declared in `hf_quant_config.json` and as the model was trained
  ([§2.4](#24-numerics-the-checkpoint-declares));
- the reproducible route (`--reproducible`: deterministic and request-invariant), the default route and the
  lossless route (`--lossless`);
- the bundled DSpark drafter as a speculative backend (`--spec dspark`, chosen by `--spec auto`) with the
  calibrated per-request draft length, and n-gram copy proposals;
- the existing KV profiles where they apply to 128-wide heads, `int8` and `bf16` first ([§7.6](#76-kv-profiles));
- the hybrid prefix cache, preemption with Replay, one to eight lanes;
- the Inkling chat format: reasoning effort, thinking channel, tool declarations and JSON tool calls in the
  OpenAI and Anthropic surfaces ([§17](#17-frontend-and-serving));
- Vision (image input) as the second track ([§15](#15-vision)).

### 1.2 Baselines

| Baseline | What it shows |
|---|---|
| vLLM with PR #69028 on this machine (WSL): Triton relative-bias attention, native NVFP4 Triton expert GEMMs or `--moe-backend humming`, decode CUDA graphs, BF16 or FP8 KV, DSpark | Same weights and activation quantization on this GPU class: speed compared without a quality footnote |
| SGLang with PR #51926 on this machine: FlashInfer SM120 grouped NVFP4/MXFP8 GEMMs tuned on an RTX 5090, a routed-quantization CUDA kernel, low-M BF16 GEMVs, FP32 head GEMV, DSpark | The other `sm_120` implementation; its DSpark reports mean acceptance 3.93-4.07 on GSM8K |
| TokenSpeed (PR #3682) | Not runnable here (its fast path is a gfx950 decode monokernel on 4× MI350X); it is the strongest published decode design for this model: 465.6 tok/s per user at C = 1 with DSpark (3.59 tokens per verify step on random prompts) |
| llama.cpp with the GGUF `Q8_0` / `IQ4_NL` quants | The common local path; re-quantized weights and different arithmetic, compared with a quality column |
| Infernix Qwen3.6-35B-A3B and Qwen3.8-27B NVFP4 on this machine | Not a quality baseline: what this engine already achieves per layer and per byte ([§4.2](#42-calibration)) |
| Transformers PR #51050 | Correctness reference for the layer mathematics only: it dequantizes the experts and runs them at BF16 activations, which is not the model's expert arithmetic |

### 1.3 Targets

Primary configuration: the official checkpoint converted bit-exactly, INT8-G64 KV, C = 1, `--max-context
262144`, text, the template's default reasoning effort (0.9), 4K prompt, 512 output tokens.

| Metric | Must | Stretch | Model (§4), central (range) |
|---|---:|---:|---:|
| Plain decode, C = 1, 4K context | ≥ 185 tok/s | ≥ 240 | 205 (185-230) |
| Best speculative mode (DSpark, calibrated length), code-like text | ≥ 1.2× plain | ≥ 1.7× plain | 1.4× (1.2-1.65×) |
| Plain decode at 128K context, relative to 4K | ≥ 70 % | ≥ 85 % | ~79 % |
| Prefill, 8K fresh prompt | ≥ 9,000 tok/s | ≥ 16,000 | 13,000 (9,000-18,000) |
| Time to first token, 4K fresh prompt | ≤ 0.45 s | ≤ 0.3 s | 0.33 |
| Steady plain decode, C = 8, aggregate | ≥ 2.5× C = 1 | ≥ 3.5× | ~3× (2.6-3.6×: 8 rows touch ~38-52 experts per visit and 8 windows) |
| Against vLLM's and SGLang's SM120 paths on this machine | faster at every measured point | ≥ 2× decode at C = 1 | not modelled |
| Fidelity | every Op within its criterion; the model-level gates of [§16.4](#164-model-level-quality) | — | — |

Each must sits at or below the model's low end; the stretch values need the per-visit cost at the low end of
[§4.3](#43-plain-decode). The acceptance report gives the median and range of at least three interleaved runs (the
first discarded), the worst case, and the round anatomy (per-visit time by kernel from an nsys main-stream frontier, verify widths,
acceptance). A missed target is reported as missed; targets are not changed after measuring.

### 1.4 Non-goals

- Inkling 2 Small (`inkling2`, 48 layers, no loop): too large for one RTX 5090 at its precision; the layer Ops
  would serve it, the family composition would not.
- The variable-loop research checkpoint's shared-cache mode (`loop_share_cache`, which vLLM, SGLang and TokenSpeed
  expose): it changes the model. The pass count itself is a config value (D8).
- Re-quantizing the BF16 attention projections or the head by default (lossy). An opt-in recipe is an owner
  decision (D7).
- Audio, video input, multi-GPU, GPUs other than `sm_120a`.
- A generic model graph, layer registry or plugin path ([AGENTS.md](../../AGENTS.md)).

---

## 2. Model facts

### 2.1 Configuration (upstream)

| Quantity | Value |
|---|---|
| Architecture / model type | `Inkling2LoopForConditionalGeneration` / `inkling2_loop` (text `inkling2_loop_text`, vision `inkling2_vision`) |
| Parameters | 26.4 B total; 3.4 B active per layer pass; ~5.4 B touched per token (card). Trained with 384 experts per layer, pruned to the 128 most used at mid-training |
| Hidden width H | 2304; RMSNorm eps 1e-6 |
| Layers | 24 physical. Prelude 0-5, core 6-17 run `num_passes` = 3 times with shared weights, coda 18-23: **48 visits** per token |
| Attention layers | sliding window 1024: 0-4, 6-10, 12-16, 18-22 (20 layers, **40 visits**); full: 5, 11, 17, 23 (**8 visits**: 5 and 23 once, 11 and 17 three times) |
| Heads | 24 query, 8 KV (group 3), head dimension 128, scale 1/√128; no RoPE |
| Relative bias | 16 relative dimensions per head, profile bank P [16, 1024] per layer, extent 1024 (also for full layers) |
| Clips | q scaled-L2-clipped, k L2-clipped to 22.627 (= 16√2); relative states scaled-L2-clipped to 8.0 |
| Log scaling (full layers) | τ = 1 + 0.1 ln(max(1, (pos + 1) / 8192)) on q and on the relative states |
| Short convolutions | kernel 4, depthwise, causal |
| Attention-residual blocks | 6 layers per block |
| MoE | 128 routed experts, top-8, no shared expert, width 1152; gate softcap 4, up softcap 25; route scale 8; router bias |
| Vocabulary | 201,024 rows stored (200,058 real); embedding RMS-normed; untied head; final logit softcap 30 |
| Context | 1,048,576 positions (no RoPE: no YaRN question) |
| Loop | `core_start` 6, `core_end` 18, `num_passes` 3, `gate_rank` 64, `is_varloop` false |
| Drafter (`mtp_config`) | 3 dense layers (0, 2 sliding 1024; 1 full), Hq 24 / Hkv 8 / D 128, RoPE θ 1e6, MLP 9216, block 8, Markov head `rnn_hidden` rank 256, targets: inputs of visits 5, 23, 41 and the final state (48) |
| Vision | 23-layer ViT, width 1024, 16 heads, MLP ratio 4, patch 20, learned 32×32 position table, 2×2 merge, projector 4096 |

### 2.2 Mathematics (upstream: Transformers PR `modeling_inkling2_loop.py`, vLLM PR, SGLang, TokenSpeed)

Written once here so every Op and oracle uses the same statement. Per token; `rms(x) = x / sqrt(mean(x²) + eps)`
(unweighted), `norm_w(x) = w ⊙ rms(x)`, `softcap_c(x) = c·tanh(x / c)`, `clip_t(x) = x · min(1, t / (‖x‖₂ + 1e-6))`.
M0 pins every rounding point listed in [§2.5](#25-rounding-points-pinned-at-m0).

**Embedding.** `e = norm_embed(E[token])`, BF16.

**Block attention residuals.** The stream is a list of completed blocks plus a partial sum `h`. A *mix* with a
learned FP32 query `w` over sources `S = [b₁, …, b_n, h]` is

```text
mix_w(S) = Σ_j softmax_j( w · rms(s_j) / √H ) · rms(s_j)
```

(FP32; the same `rms(s_j)` serves as key and value). Full-attention layers have three queries for their attention
input (q, k and v inputs), sliding layers one; every layer has one for its MLP input; `output_query` serves the
final state. A layer whose index is a multiple of 6 is a *block boundary*: its input `h` becomes a block and the
partial sum restarts at its attention output.

| Visit (layer, pass) | Blocks seen by the attention input | Blocks seen by the MLP input |
|---|---|---|
| 0 | none: input is `e` | `[e]` + partial |
| 1-4 | `[e]` + h | `[e]` + h |
| 5 (full) | `[e]` + h (3 queries) | `[e]` + h |
| 6, every pass | `[e]` + h | `[e, h6]` + partial |
| 7-11 | `[e, h6]` + h | `[e, h6]` + h |
| 12, every pass | `[e, h6]` + h | `[e, h6, h12ᵖ]` + partial |
| 13-17 | `[e, h6, h12ᵖ]` + h | same |
| 18 | `[e, h6, h12³]` + h | `[e, h6, h12³, h18]` + partial |
| 19-23 | `[e, h6, h12³, h18]` + h | same |
| final | `mix_out([e, h6, h12³, h18] + h)` then `norm_final` | — |

At most 5 sources. `h6` (the prelude's output) is the same in every pass; `h12ᵖ` is pass p's own.

**Loop passes.** Pass 1 runs layers 6-17 on `(h6, [e])`. Passes p = 2, 3 restart from the same `(h6, [e])` and, at
every core layer i, read four *messages* from the previous pass's two half-blocks:

```text
m_{i,k} = mix_{u_{i,k}}([h12^{p-1}, h18^{p-1}])        k ∈ {q, k, v, mlp},  u: [12, 4, H] FP32
gate_k(x, m) = (1 - tanh(F_k z)) ⊙ x + tanh(G_k z) ⊙ m,   z = W_down,k [rms(x); rms(m)]   (W_down 64×2H, F, G H×64)
```

Sliding core layers: `x_q, x_k, x_v = norm_attn(gate_q(x, m_q)), norm_attn(gate_k(x, m_k)), norm_attn(gate_v(x, m_v))`
with `x` the attention mix; full core layers gate their three mixes, then apply `norm_q`, `norm_k`, `norm_v`. The MLP
input becomes `gate_mlp(mix, m_mlp)` before `norm_mlp`. Every pass has its own KV and conv state.

**Attention (one visit).** Input rows `x_q, x_k, x_v` (all equal on pass-1 sliding visits).

```text
q = clip_22.6(s_q ⊙ W_q x_q)             [24×128]   ; full layers: q ← τ(pos) q
k = clip_22.6(sconv_k(W_k x_k))          [8×128]
v = rms(sconv_v(W_v x_v))                [8×128]     (per head, no weight)
r = clip_8(s_r ⊙ W_r x_q)                [24×16]     ; full layers: r ← τ(pos) r
g = W_gate x_q  [3072],   σ_sink = W_sink x_q  [24]
score(i, j, h) = q_h(i)·k_{h/3}(j) / √128 + bias_h(i, i - j)
bias_h(i, d) = r_h(i) · P[:, d]  for 0 ≤ d < 1024, else 0
a_h(i) = Σ_j softmax over {score(i, ·, h), σ_sink,h(i)} restricted to the visible keys  ×  v_{h/3}(j)
         (the sink takes probability mass and contributes no value)
visible: sliding j ∈ (i - 1024, i];  full j ≤ i
o = sconv_o(W_o (σ(g) ⊙ a))
```

Sliding layers' K/V convolutions are static: `sconv(x)_t = (x_t + Σ_{d=0}^{3} c_d x_{t-d}) / √5` per channel. Full
layers' are dynamic: taps `c_{d,channel}^static + (W_tap x_k)_{head·4+d}` with a 32-row `W_tap` per K and V, and
residual scale `1/√9`. The post-attention and post-MLP convolutions are static, width H.

**Sublayer residuals.** `h ← h + o` (or `h ← o` at a block boundary), then the MLP mix, then
`h ← h + sconv_mlp(MoE(norm_mlp(·)))`, rounded to BF16 at the layer end.

**MoE.**

```text
logits = W_router x  (FP32),  s = σ(logits)
chosen = top8(s + bias)            (selection only)
γ_e = 8 · s_e / (Σ_{e∈chosen} s_e + 1e-20)
y_e = W_gate/up,e A4(x)            interleaved rows: (g_0, u_0, g_1, u_1, …)
h_e = bf16(softcap_4(g) ⊙ σ(g) ⊙ softcap_25(u) ⊙ γ_e)        (γ applied before the down projection)
out = bf16( Σ_{e∈chosen} W_down,e A4(h_e) )                   (FP32 sum in slot order, one rounding)
```

where `A4` is the activation quantization of [§2.4](#24-numerics-the-checkpoint-declares) (NVFP4 layers) or its
MXFP8 counterpart (layers 20-23). γ before the down projection and an FP32 combine are what the CUDA engines compute:
vLLM's expert block "matches bitwise given the same inputs" the reference, and TokenSpeed's Inkling2 kernels apply
the "capped SwiGLU x route weight with its quantization" and a "slot-order fold". Transformers applies γ after the
down projection on dequantized weights.

**Head.** `logits = softcap_30(W_head norm_final(mix_out(...)))` over the 200,058 real rows (FP32 output).

**DSpark drafter** ([§13](#13-speculative-decoding-dspark)) and **Vision** ([§15](#15-vision)) are stated in their
sections.

### 2.3 Checkpoint inventory (upstream, safetensors headers)

| Tensor group | Format | Bytes | Read per token (decode, short context) |
|---|---|---:|---:|
| Embedding [201,024 × 2304] | BF16 | 883 MiB | one row |
| Attention per layer: q 3072, k 1024, v 1024, r 384, gate 3072, sink 24 rows × 2304; o 2304 × 3072 (26.9 M params) | BF16 | 51.3 MiB (×24) | 53.8 MB × 48 visits = **2.58 GB** |
| Dynamic conv taps (full layers) 2 × 32 × 2304 | BF16 | 0.3 MiB (×4) | negligible |
| Experts, layers 0-19: w13 [128, 2304, 1152 B] + E4M3 scales /16; w2 [128, 2304, 576 B] + scales; per-expert `amax`, one `outer_scale_denominator` | NVFP4 | 546.8 MiB (×20) | 8 × 4.48 MB × 44 visits = **1.58 GB** |
| Experts, layers 20-23: w13 [128, 2304, 2304], w2 [128, 2304, 1152] E4M3 + UE8M0 /32 | MXFP8 | 1002 MiB (×4) | 8 × 8.21 MB × 4 visits = **0.26 GB** |
| Router [128, 2304] + FP32 bias | BF16 | 0.6 MiB (×24) | 28 MB |
| Message gates, core layers: 4 × (W_down 64 × 4608, F 2304 × 64, G 2304 × 64) | BF16 | 4.5 MiB (×12) | 113 MB (passes 2-3) |
| Attn-res queries, message queries (2 × [24, 2304]), norms, convs, P | FP32 / BF16 | ~3 MiB | small |
| Head [201,024 × 2304] | BF16 | 883 MiB | 922 MB (200,058 rows) |
| Vision tower + projector | BF16 | ~606 MiB | per image |
| Drafter (`mtp.safetensors`) | BF16 | 606 MiB | per draft round |
| **Total** | | **18.19 GiB + 0.59 GiB** | **~5.66 GB** + KV (as stored; §4.1 for packed) |

Layout facts the converter must honour ([§18](#18-artifact-and-conversion)): w13 rows interleave gate and up
(`g_0, u_0, g_1, …`); NVFP4 nibbles hold the even element in the low half; the per-expert NVFP4 factor is
`amax_e / outer_scale_denominator`; the two message-query tensors store `[q, mlp]` and `[k, v]` row pairs per core
layer; the head and embedding are padded to 201,024 rows.

### 2.4 Numerics the checkpoint declares

`hf_quant_config.json` assigns each tensor a compute dtype. The experts are not "weights to dequantize": their group
declares `required_activation_quantization`, and Thinking Machines states that the experts ran NVFP4 (last four
layers MXFP8) "on all forward passes from pretraining through RL" and are deployed "in the same numeric precision
they were trained on". So the activation quantization is part of the model:

- **NVFP4 layers (0-19), four-over-six.** Per token row x (BF16): `amax = max|x|`; `enc = min(1536 / amax, FP32_MAX)`
  (rounded division), `outer = amax · (1/1536)`. Per 16-element block with max `b`: two candidate E4M3 scales
  `s6 = e4m3(min(b·enc/6, 448))`, `s4 = e4m3(min(b·enc/4, 448))`; each gives E2M1 codes
  `c = e2m1_rn_satfinite(x / (s·outer))` (reciprocal form in the reference); the block keeps the candidate whose
  absolute reconstruction error `Σ|c·s·outer - x|` (a fixed pairwise order) is smaller, ties to the 6.0 candidate.
  An activation's value is `c · s · outer`.
- **MXFP8 layers (20-23).** Per 32 values: `scale = 2^ceil(log2(amax/448))` (UE8M0, clamped), values
  `e4m3_rn(clamp(x / scale, ±448))`.
- **Weights.** NVFP4 `c · s_block · (amax_e / denominator)`; MXFP8 `e4m3 · 2^(e-127)`.
- Everything else is BF16 with FP32 where the reference upcasts (norms, mixes, router, gates, convs, softmax).

Two facts follow. The exact route runs the experts on native FP4/FP8 tensor cores, because that is the model's
arithmetic, not a fast path. And the activation quantization depends only on its own token row, so it is
request-invariant by construction: no residual-term machinery is needed (Infernix's two-term A4 paths exist because
Qwen's checkpoints were not trained at A4; one-term A4 failed their quality gate).

### 2.5 Rounding points pinned at M0

The engines disagree or the sources are silent on these; M0 fixes each from the merged reference and the release
statement, records it in the model reference, and the oracles follow it:

| Point | Candidates seen |
|---|---|
| γ placement and combine | inside the activation, FP32 combine in slot order, rounded once (vLLM; TokenSpeed's Inkling2 kernels) vs after the down projection (Transformers) |
| gate/up accumulator rounding | `bf16(acc · w_e · outer)` (vLLM: expert then row scale) vs `bf16(acc)` then the outer scales in the SwiGLU (SGLang SM120) vs `bf16(acc · (xa · wa) / (1536 · WD))` (TokenSpeed) |
| Expert SiLU-gate transcendentals | `tanh.approx` / `ex2.approx` (vLLM's kernels, which it reports bit-equal to the reference) vs correctly rounded |
| Relative-bias table | `bf16(r · P)` (Transformers eager, SGLang `RelLogitsProj`) vs FP32 in the score |
| Message gate | FP32, every operation rounded separately, no FMA contraction (TokenSpeed `message_gate`) |
| Mix association | two-pass softmax (Transformers) vs online partials then combine (TokenSpeed, Kimi-K3's split kernels): FP32-level differences |
| Conv state | BF16 inputs, FP32 arithmetic, BF16 output (Transformers); BF16 rings (vLLM, TokenSpeed) |
| Message-query row order | `[q, mlp]` / `[k, v]` pairs per layer (Transformers PR conversion) |
| DSpark sliding window | anchored at the block start for every block row (inferred from SGLang's `window_anchor_at_start` flag) vs per row |
| DSpark block attention | non-causal within the block (DFlash's rule, inferred from SGLang reusing `DFlashDraftModel.forward`) vs causal |

### 2.6 Frontend facts (upstream)

- **Tokenizer.** Byte-level BPE, 199,998 vocabulary entries + 60 added tokens; split regex of the o200k family
  (case-aware letter runs with English contractions, 1-3 digit numbers, punctuation runs with trailing `[\r\n/]*`,
  newline runs, whitespace); `ignore_merges: true` (a pre-token found whole in the vocabulary is one token); no
  normalizer.
- **Chat template.** Messages are `<|message_{user,model,system,tool}|>NAME?<|content_{text,thinking,image,…}|>…<|end_message|>`;
  the first non-system message is preceded by `<|message_system|><|content_text|>Thinking effort level: E<|end_message|>`
  (E in [0, 0.99], default 0.9; none/minimal/low/medium/high/xhigh/max map to 0/0.1/0.2/0.7/0.9/0.99/0.99); tools as
  `<|message_system|>tool_declare<|content_xml|>` + sorted compact JSON; assistant tool calls as
  `<|message_model|>NAME<|content_invoke_tool_json|>{"name":…,"args":{…}}<|end_message|>`; every assistant turn ends
  with `<|content_model_end_sampling|>` (EOS 200006). Images are `<|content_image|><|unused_200054|>` (one placeholder
  expanded to the image's feature count).
- **Output parsing.** `tokenizer_config.json` carries a declarative `response_template`: regex-delimited
  `reasoning_content` (`<|content_thinking|>`), `content` (`<|content_text|>`, `tool_error`, `invoke_tool_text`) and
  `tool_calls` (`<|content_invoke_tool_json|>` JSON transformed to `{name, arguments}`). TokenSpeed ships an
  `inkling2_json` grammar for constrained tool-call framing.
- **Images.** `Inkling2ImageProcessor`: RGB, rescale 1/255, normalize mean/std 0.5 (the tower renormalizes to CLIP
  statistics), patch 20, merge 2 (40-pixel grid), sides 40-4096 px.

---

## 3. Other implementations and Infernix's own record

### 3.1 What the other engines do

| Engine | Experts | Attention | Decode structure | Notes |
|---|---|---|---|---|
| vLLM PR, SM120/121 | native NVFP4 Triton GEMMs (`dot_scaled`), four-over-six quantizer fused with the SwiGLU; MXFP8 dequantized to BF16 in Triton; optional Humming NVFP4 | portable Triton relative-bias kernels; a shared-memory overflow on SM12x (`k_group_partials`) needed a fix | decode-only CUDA graphs; GB10 recipe disables prefix caching | BF16 projections with 1-16 rows on a tensor-core GEMV; FP32 head; GB200: 5.86 ms per verify step at 1 request, 51K context |
| SGLang PR, SM120/121 | FlashInfer block-scaled grouped GEMMs, (swap_ab, tile_n, tile_k) tuned on an RTX 5090 for N 2304, K 1152/2304; a CUDA routed-quantization kernel; FP32 one-shot combine; NVFP4 MegaMoE elsewhere pads 1152 → 1280 | FA4 with a bias/sink score_mod (datacenter); Triton elsewhere | loop controllers fused and fitted in SM120 shared memory; low-M BF16 GEMVs (cuBLAS up to 1.4× slower at 2-16 rows) | DSpark mean acceptance 3.93 (GB300) / 4.07 (H200) on GSM8K greedy |
| TokenSpeed PR, gfx950 | NVFP4 GEMM tiles reading the exact BF16 values of the codes, epilogue `bf16(acc·(xa·wa)/(1536·WD))`; MXFP8 ring 2 × 42 KB | split-KV decode at 8 splits (window) and a fixed 32 at the served context (full), 128-key blocks | **decode monokernel**: one persistent launch per layer, stages with tagged mailboxes, bit for bit its regular path; a later pass's core visit runs as two launches; conv rings of 3 + verify block | 465.6 tok/s per user at C = 1 (4× MI350X, DSpark, 3.59 tokens per verify step); attention-residual partials swept once per block set; conv checkpoints paged on the KV planes |
| Transformers PR | experts dequantized to BF16, BF16 activations | eager only (bias + sink) | none | layer-mathematics reference |
| llama.cpp (GGUF) | re-quantized Q8_0 / IQ4_NL | ggml | — | not the trained arithmetic |

None of the published paths runs near this GPU's byte floor (SGLang's B300 chart: 29.7 ms per token plain at C = 1,
~40× its floor). The gap everywhere is fixed per-step cost; TokenSpeed's monokernel is the one design aimed at it.

### 3.2 Lessons from Infernix's record, applied here

Measured results from earlier Infernix work, mostly on Qwen3.8-Flash-Next (**FN** =
[qwen3_8-flash-next-design.md](qwen3_8-flash-next-design.md)), and the decision each forces here. They are applied
from the first implementation.

| # | Recorded result (measured) | Decision here |
|---|---|---|
| L1 | A resident weight-streaming kernel costs ~1.4 µs + bytes / 1.67 TB/s; an empty graph-replayed kernel 2.82 µs; a BF16 GEMV reaches 1.51-1.54 TB/s at T = 1 (FN §19.3.3, §4.1) | the performance model uses these, not the 1.79 TB/s spec ([§4](#4-performance-model)) |
| L2 | ≥ 32 KB of loads in flight per SM decides T ≤ 8 speed (a 21 MB stream: 18.2 µs at 4 CTAs/SM vs 29.4 at 1) (FN §8.3) | every decode GEMV and expert kernel is sized for ≥ 32 KB in flight per SM |
| L3 | A second partial wave costs a call 25-30 %; few-row matrices need their own mapping (one row per warp, all loads issued before the FMAs) (FN §19.3.3) | grids fixed per (B, T) at startup to whole waves; the router (128 rows), sink (24), relative (384), tap (64) and gate-down (64) rows use the few-row mapping |
| L4 | `bf16_p12` (lossless 12-bit BF16, packed at load, 12.02 bits a weight on Flash-Next) on [16384, 2560]: 1.26× at T = 1-4, 1.13-1.19× at 5-16, 0.87-0.96× at 33-64, 0.67-0.91× at 65-1,024, 0.92-0.99× beyond; its skinny route (T ≤ 8, and 16 columns a pass under `A16ColumnInvariant`) gives the BF16 skinny route's bits. Output projections [2560, 6144] gained 1.00-1.06× at T ≤ 8 only with a later direct MMA; few-row matrices lose (a [96, 2560] control 0.43-0.97×). A matrix is packed only when every reader takes the packed form. Plain decode +3.7 % on recipe A (kernels alone +2.4 %, the rest freed VRAM) (FN §19.3.46) | the K2 parents, the head and the embedding are packed where M1's per-shape sweep shows the decode and verify widths faster; W_o only if its direct MMA wins; routers and gates stay BF16. Prefill pays the packed routes' 0.67-0.93× at 65-1,024 columns on those projections (no second BF16 copy) ([§11](#11-dense-projections-and-the-head)) |
| L5 | A norm in a GEMV prologue was 2-4× slower; a cluster-fused gated RMSNorm lost 1.1 % (FN §19.3.20f, §19.3.24) | mixers and norms are their own kernels; GEMVs read finished BF16 inputs ([§8.2](#82-the-visit-chain)) |
| L6 | Projection epilogues writing q/k/v/z in their consumers' layouts +1.8 %; fusions that kept overlap won (FN §19.3.20a) | K2 writes q, k, v, r, gate and sink straight into the attention's layouts |
| L7 | PDL pair boundaries cost ~0 µs at ≤ 2 CTAs/SM; a cooperative barrier 1.4-2.0 µs; streaming kernels trigger after their main loop (entry trigger ~10 % slower), short kernels at entry only when their grid covers ≤ half the SMs (FN §4.1, §19.3.33; [op development §4.3](op-development.md#43-kernel-and-common-facilities)) | every chain link follows these trigger rules; weights are fetched before `griddepcontrol.wait` |
| L8 | ptxas hoists `LDG.E.CONSTANT` loads above `griddepcontrol.wait` | data from the previous kernel is read with coherent loads; the PDL SASS audit covers every new kernel |
| L9 | Narrow W4A4: register-resident row pairs for 1-3 columns (35.4 → 30.8 µs at T = 1), s8 `mma.m16n8k16` on doubled codes for 4-8 (ties the floor); E2M1 decode by `cvt.rn.f16x2.e2m1x2` (a `constexpr` table compiled to local memory: 0.58 TB/s) (FN §19.3.21, §19.3.52-53) | the narrow expert route starts from these kernels ([§10.3](#103-narrow-routes-decode-verify)) |
| L10 | Encode x once per (job, column): each CTA quantizing its own x was slower at 2-3 columns; at T = 1 the row-pair kernel hides an in-kernel encode behind its weight loads (FN §19.3.44) | x is quantized once in the MoE mixer; h's quantization point is chosen by an M5 op benchmark ([§10.3](#103-narrow-routes-decode-verify)) |
| L11 | Down as a PDL dependent copying its weights before the wait +3.3 %; down CTAs polling their own job's completion 77.9 → 69.7 µs at 4 columns; one fused ticketed gate/up+down grid lost 0.5 % (FN §19.3.21) | K8 polls per-job counters; gate/up and down stay two kernels |
| L12 | Expert kernels cost per job, not per column; verify rounds average 1.2-1.5 columns per expert; padding columns must route no expert (FN §19.2, §19.3.24; 122B: 47.8 → 79.6 tok/s from a padding mask) | padded rows of a graph batch route nothing; job planning is per expert |
| L13 | Prefill grouped pipeline: 1 producer + 8 consumer warps, `mxf4nvf4` m16n8k64, 64×128×128 tiles, 3 × 43 KB stages, 2 CTAs/SM, SwiGLU and A4(h) in registers: T = 4,096 79.2 → 1.86 ms; dense A4 GEMM with quantization 882-985 TFLOP/s (FN §19.3.8) | the wide expert route reuses this pipeline with Inkling's row scales ([§10.4](#104-wide-routes-prefill)) |
| L14 | Routing for ≤ 32 columns in one 1,024-thread CTA; a depth-4 arg-max tree cut selection 7,400 → 3,500 cycles; router weights before the PDL wait +0.4 % (FN §19.3.33) | K6 = router GEMV + one-CTA route ([§10.2](#102-moe-mixer-router-and-dispatch)) |
| L15 | Decode attention over a small key set is latency-bound: 45 → 18.2 µs from cp.async 64-key tiles, parallel exps, 512 threads, vectorized query loads, register accumulators; tensor-core decode attention pays from ~8 query rows per KV row; 64-key splits beat 128 at short context (FN §19.3.14, §19.3.22) | the window attention starts from that shape ([§9.2](#92-decode-and-verify-t--16)) |
| L16 | Width-dependent routes change bits and broke speculation/plain equality and prefix restores; the exact route's one arithmetic class per width band costs ~10 % at C = 1 on the 27B; Q8 MMA at T = 9-64 was a cliff (FN §19.3.3; [model quality §6](model-quality.md#6-numerical-acceptance-policy-and-fast-paths)) | one column-invariant schedule per decode Op for T ≤ 16 on the reproducible route; width-specialized routes are fast paths |
| L17 | Megakernels lost in the model (-1.8 to -9 %) despite microbenchmark wins: the prototype fused only the latency-bound MoE front (mix, router, route, dispatch), serialized the mix and lost the overlap the separate kernels had; co-residency (register-full CTAs, mismatched shared-memory carveouts, 1,024-thread CTAs) (FN §19.3.33-34) | the PDL chain first; a per-visit persistent kernel is an M8 experiment that removes those causes ([§8.5](#85-the-persistent-visit-kernel-experiment)) |
| L18 | L2 bulk prefetches compete with a kernel's own stream (self-prefetch +29 % slower); weights re-read soon come from L2 at 2.28 TB/s (FN §19.3.20c, §19.3.3) | no next-weight prefetching; the drafter's per-step matrices are sized to stay in L2 ([§13.6](#136-drafter-storage)) |
| L19 | ~99 KB shared memory per CTA; above ~50 KB a CTA runs alone on its SM (-16..-23 % on a fused gate/up); a 2 KB static `__shared__` dropped a grid to 1 CTA/SM; spills 22-31 % slower; `cp.async.bulk` to `shared::cluster` raised the stack to 14 KB a thread | shared-memory budgets per kernel in §9-§10; `shared::cta` only; spill allowlist entries empty |
| L20 | TMA producers raced generic-proxy readers (19/20 corrupted) without `fence.proxy.async.shared::cta` | every TMA ring fences; qualification runs `verify_repeatable` with a side stream |
| L21 | On MoE models verifying all drafts every round lost to MTP; a per-request length calibrated by confidence deciles won +8-11 %; device-side IF nodes stopping drafting past the confidence cut +2.2 %; trees and copies lost on the Flash-Next MoE (FN §19.2, [DFlash](dflash.md), [tree verification](tree-verification.md)) | calibrated length from day one; IF nodes for the Markov steps; trees deferred ([§13.4](#134-draft-length-and-verification)) |
| L22 | Capturing every graph at startup: first request +5.1 % (1.0-1.5 MiB each); idle graphs cost (90 tree graphs, 1.3 %) (FN §19.3.37) | all (B, T) graphs at startup; no graph per unused width |
| L23 | Flash-Next's prefill is bit-identical across chunk widths, so it defaults to 16K chunks (pp32768 5,567 → 11,211 tok/s from 1K to 16K); wider steps while no lane decodes +9.3 %; merging the prompt's last two calls saves 9-12 ms of TTFT (FN §19.3.24, §19.3.45, §19.3.56) | prefill Ops are chunk-width invariant per column ([§14.1](#141-chunk-width-invariance)) |
| L24 | 8-bit activations for dense decode projections lost 15-27 % end to end (FN §19.3.31) | decode projections stay A16 (also under D7) |
| L25 | Exact BF16 KV packed as `bf16p` (784 of 1,024 B per token per head) holds 1.31× the tokens at level speed (FN §19.3.48) | `bf16p` is a second-wave KV profile ([§7.6](#76-kv-profiles)) |
| L26 | Op timing: CUDA events quantize at 2.048 µs, eager benches overstate narrow kernels 1.5×, nsys node tracing inflates graph launch, the first bench run reads 0.5-1.5 % slow, clocks after a faster decode slow the next prefill 6 % | op timing by nsys kernel spans in graph replay with PDL; interleaved A/B, first run discarded |
| L27 | Measure an op-level prototype before quoting an end-to-end gain (a split-K last wave estimated at -6..-9 % measured +5..+13 % slower) | every fusion choice left open below names its op benchmark ([§21](#21-implementation-plan)) |

### 3.3 Ideas taken from TokenSpeed and SGLang

| Idea | Source | Use here |
|---|---|---|
| Attention-residual mixes as online partials over the completed blocks (max, sum, FP32 accumulator per (token, query)), swept once per block set, then a combine that folds the current partial sum and applies the norm | TokenSpeed `attention_residual_partials` / `_combine` (Kimi-K3's split kernels) | prefill and wide verify; at decode widths the direct mix is ≤ 5 rows, so the sweep is an M5 measurement. The pass restart reuses the prelude's statistics ([§12](#12-residual-mixing-message-gates-and-short-convolutions)) |
| A later pass's core visit runs as two parts (attention, MoE), because the gated input needs the whole vector | TokenSpeed monokernel | the same split in the PDL chain ([§8.2](#82-the-visit-chain)) |
| Conv rings of 3 + verify width | TokenSpeed | the StateImage conv rings ([§7.2](#72-stateimage)) |
| ShortConv checkpoints stored per KV page, so a prefix restores at any page boundary | TokenSpeed cache recipe (`kvconv`, `hiddenconv`) | the alternative to StateImage snapshots, kept for M4 ([§7.4](#74-prefix-cache)) |
| A KV-only model resumes a prefix at any token bit for bit: a query row depends only on the K/V rows it attends to | TokenSpeed numerics notes | chunk-width invariance with append-then-attend ([§14.1](#141-chunk-width-invariance)) |
| Full-attention decode with a fixed split count from the served maximum context (32 at 262K), split length from the row's own length | TokenSpeed | request-invariant splits for the 8 global visits ([§9.2](#92-decode-and-verify-t--16)) |
| A persistent per-layer decode kernel, bit for bit equal to the regular path, whose tasks issue their weight loads before polling, with plan-checked deadlock freedom, tagged mailboxes and an instruction-cache warm-up | TokenSpeed monokernel | the M8 experiment ([§8.5](#85-the-persistent-visit-kernel-experiment)) |
| A per-token relative-logit table materialized before the attention kernel ([24 × 1,280] per token) | vLLM `attention.py` | the alternative to the in-kernel bias of [§9.2](#92-decode-and-verify-t--16), compared at M5 |
| A routed-quantization kernel that zeroes each expert's scale tail in the CTA writing that expert's last row, so its work scales with the routed rows | SGLang `routed_quant_sm120.cuh` | the wide route's quantize-permute ([§10.4](#104-wide-routes-prefill)) |
| Grouped NVFP4 tile choices measured on an RTX 5090 for these shapes (128 experts, top 8, N 2304, K 1152/2304, cold weights): swap A/B with tile 32 × 128 to 128 routed rows, 32 × 256 to 1,536, 64 × 256 to 3,072, then 128 × 256 (K ≥ 2048) or swapped 128 × 128 | SGLang `inkling2_grouped_sm120.py` | starting points for the wide route's tile search; the SGLang path is also an M8 baseline |
| A one-row-per-tile SIMT GEMV is fastest at 1-3 rows for 2304 → 32 and 2304 → 128 projections and for the head at one row; cuBLAS takes up to 1.4× as long at 2-16 rows | SGLang `sm12x_bf16_gemv.py` | agrees with L3's few-row mapping for the router and gates |
| Message queries pre-multiplied by `H^-1/2` and controller weights stacked per layer | SGLang `inkling2_loop.py` | the converter stores them stacked; the scale folds into the query if M0 shows it is exact |

---

## 4. Performance model

### 4.1 Bytes per token

The checkpoint stores ~5.66 GB read per decoded token ([§2.3](#23-checkpoint-inventory-upstream-safetensors-headers)).
With the K2 parents and the head packed losslessly (L4, at Flash-Next's 12.02 bits a weight; Inkling's census is M1),
W_o as stored unless its packed route wins:

| Part | As stored | Model (K2 + head packed) |
|---|---:|---:|
| K2 parents (q, k, v, r, gate, sink), 48 visits | 1.90 GB | 1.43 GB |
| W_o, 48 visits | 0.68 GB | 0.68 GB (0.51 if packed) |
| Experts (44 NVFP4 + 4 MXFP8 visits × 8) | 1.84 GB | 1.84 GB |
| Head (200,064 rows: the real 200,058 rounded up to the packed head's multiple of 16) | 0.92 GB | 0.69 GB |
| Routers, message gates, small tensors (BF16) | 0.14 GB | 0.14 GB |
| Window KV (40 visits × 1024 keys), INT8 / BF16 | 0.09 / 0.17 GB | same |
| **Total, INT8 KV, short context** | **5.57 GB** | **4.87 GB** |

Global KV adds 17.3 MB (INT8) or 33.6 MB (BF16) per 1K of context. Model shares at 4K, INT8 KV: projections 43 %,
experts 38 %, head 14 %, KV 2-3 %, other 3 %.

### 4.2 Calibration

Two Infernix measurements on this machine (bytes from each geometry, formats approximated):

| Model | Plain decode | Bytes per token | Bandwidth time | Fixed time per layer |
|---|---:|---:|---:|---:|
| Qwen3.6-35B-A3B `groupwise-int`, 7.7K (2026-09-07 revision) | 338 tok/s | 1.9-2.6 GB | 1.1-1.45 ms of 2.96 | 38-46 µs × 40 |
| Qwen3.8-27B NVFP4-QAD (README, current) | 90.2 tok/s | ~15 GB | 8.4 ms of 11.1 | ~42 µs × 64 |

### 4.3 Plain decode

`t = bytes / 1.67 TB/s + 48 × f(T)`, with f the non-streaming time of one visit, built bottom-up from the recorded
costs of §3.2 (model; the two calibration points above give ~40 µs per layer for Qwen's longer chains):

| Part of f | T = 1 | T = 4 | T = 8 | Basis |
|---|---:|---:|---:|---|
| Attention (latency-bound; window visits dominate) | 10-18 µs | 25-35 | 40-55 | L15: 18.2 µs at T = 1 and 49-55 at W = 8 on Flash-Next's tuned kernel |
| Mixers (2 kernels, 4 on passes 2-3: 3 on average) | 6-12 | 7-13 | 8-14 | ~2-4 µs per small chained kernel (L1, L7) |
| Router and route | 7-14 | 8-15 | 9-17 | Flash-Next: router ~4.3-5 µs (load + launch), one-CTA route, dispatch and CPU plan 6.5-8.7 µs over 512 experts; Inkling's route has 128 experts and no CPU plan, so 3-9 µs |
| Fixed cost of the 4 streaming kernels, per-job expert cost | 6-8 | 8-12 | 10-16 | L1 (~1.4 µs each), L11-L12 (per-job cost grows with columns) |
| **f** | **29-52 µs** | **48-75** | **67-102** | |

Plain decode: 4.87 GB / 1.67 TB/s = 2.92 ms, + 48 × 29-52 µs = 1.39-2.50 ms → 4.31-5.42 ms → **205 tok/s (185-230)**,
central f ≈ 40 µs. As stored (5.57 GB) it would be 3.34 ms + the same f → ~190 tok/s.

### 4.4 Long context

Global KV adds 8 × 2.06 KiB (INT8) or 8 × 4 KiB (BF16) per context token per step: at 128K, +1.25 ms INT8 / +2.4 ms
BF16 → ~165 / ~140 tok/s (79 % / 67 % of 4K at the central f). The 8 global visits' attention at long context is a split-K stream
at the DRAM rate.

### 4.5 Speculative decoding

Verifying T columns reads the projections and head once but the union of the columns' experts. With independent
uniform routing the expected distinct experts per visit are u(T) = 128(1 - (120/128)^{8T}): 8, 15.5, 29, 41, 52
for T = 1, 2, 4, 6, 8 (maxima 8T); real routing is correlated, assumed u(4) ≈ 22, u(8) ≈ 38. Expert bytes per step
≈ u × 0.23 GB. Model weights of §4.1, INT8 KV, f(T) of §4.3:

| Draft length K (T = K + 1) | Verify bytes | Verify time | Draft time | Tokens per step (upstream / assumed) | tok/s |
|---|---:|---:|---:|---:|---:|
| 0 (plain) | 4.9 GB | 4.3-5.4 ms (4.84 at central f) | — | 1 | 185-230 (207) |
| 3 | 8.1-9.7 GB | 7.8-8.7 ms at central f(4) ≈ 61 µs | 0.5-1.2 ms | ~2.7 (assumed) | 270-325 |
| 7 | 11.7-14.9 GB | 11.0-13.0 ms at central f(8) ≈ 84 µs | 0.6-1.5 ms | 3.55-4.07 (vLLM, SGLang: GSM8K greedy); 3.59 (TokenSpeed, random prompts) | 245-350 |

The speculative rows hold f at its central value and vary the expert union and the drafter. Their verify-to-plain ratio
at T = 4 (1.6-1.8×) is below Flash-Next's measured 1.89× at W = 4 (FN §19.2, whose experts were offloaded at ~92 %
hits). Draft time is 0.5-0.6 ms with an 8-bit drafter and an FP8 Markov fold, 1.1-1.4 ms in BF16
([§13.6](#136-drafter-storage)). So **~1.2-1.65× plain** on favourable text (central ~1.4×); the expert union and the
growth of f with T, not the drafter, limit K. The calibrated length settles where a column's marginal cost exceeds its
acceptance gain.

**C = 8 plain.** Eight rows read the projections and head once but touch ~38-52 experts per visit and run eight
windows: 12.4-15.6 GB and f ≈ 67-102 µs → 10.6-14.3 ms for 8 tokens → 560-755 tok/s, **~3× C = 1 (2.6-3.6×)**.

### 4.6 Prefill

Per token: BF16 projections 2.58 GFLOP; experts 5.6 GFLOP FP4 + 0.5 GFLOP FP8; attention 0.5 GFLOP (windows) plus
8 × 24 × 128 × 4 × L/2 FLOP (global, average context L/2). Projections dominate tensor time at short context
(Infernix's BF16 MMA reaches ~170-200 TFLOP/s here), experts run at the recorded 0.9 PFLOP/s A4 rate behind the
grouped pipeline, global attention dominates beyond ~64K. Model: 9,000-18,000 tok/s at 8K, ~6,000-9,000 at 128K [M8].

### 4.7 Lever ranking (plain decode, 4K)

| Lever | Share | Kind |
|---|---:|---|
| Fixed per-visit cost (48 × f) | 30-45 % of time | exact: [§8](#8-decode-pipeline), [§8.5](#85-the-persistent-visit-kernel-experiment) |
| BF16 attention projections | 43 % of bytes after packing | exact packing in the design; beyond it lossy only (D7) |
| Head | 14 % of bytes | exact packing; drafts use a proposal head ([§13.6](#136-drafter-storage)) |
| Expert union in verification | dominates verify steps | exact: calibrated K; trees later |
| Global KV at long context | +17 MB per 1K context (INT8) | INT8 default; `bf16p` for exact BF16 |

---

## 5. Design overview

```text
token ─► embed+norm ─► prelude visits 0-5 ─► core pass 1 (visits 6-17) ─► pass 2 (18-29, messages)
        ─► pass 3 (30-41, messages) ─► coda visits 42-47 ─► final mix + norm ─► head (softcap 30, FP32)
                               taps for DSpark: inputs of visits 5, 23, 41 and the final state

per visit:  [attention mixer] → [projection q,k,v,r,gate,sink(,taps)] → [attention: conv/clip/norm/append,
            window or global, sink, combine, σ(gate)] → [o + post-attn conv + residual] → [MoE mixer: mix,
            gate, norm, A4(x)] → [router + route] → [gate/up + softcapped SwiGLU·γ] → [down]
            (passes 2-3 of the core: each mixer is two kernels)
```

- **One family, explicit composition.** `src/models/inkling2_loop` composes prelude, `num_passes` core passes and
  coda as code with visit-indexed state.
- **State.** Global KV in the Main Text paged pool (8 visit layers); window KV, conv histories, drafter rings and the
  drafter's recurrent seed in the StateImage as position-indexed rings, so speculative rollback is a frontier move
  ([§7](#7-state-kv-and-the-prefix-cache)).
- **Ops.** A new attention family (relative bias, sink, clips, group 3); a resident W4A4 four-over-six / MXFP8 W8A8
  sparse-MoE Op built from the existing narrow and wide kernel cores; packed BF16 projections with fused per-consumer
  epilogues; small mixer Ops.
- **Routes.** Reproducible (request-invariant), default (+ fast paths registered where measured faster), lossless.

---

## 6. Ownership in Infernix

| Area | Location | Notes |
|---|---|---|
| Config, binding, read-only model | `src/models/inkling2_loop/config.*`, `load/`, `model.*` | parses `text_config.loop_config`, `mtp_config`, `vision_config`; rejects configs the Ops do not cover |
| Native parameters, fixed calls | `src/models/inkling2_loop/execution/` | `visit.cpp` (one visit), `loop.cpp` (pass driver, messages), `moe.cpp`, `head.cpp`, `dspark.cpp`, `vision_tower.cpp`, `forward_lossless.cpp` |
| Program | `src/models/inkling2_loop/program/` | planning, stores, transactions, graphs, prefix binding; implements `runtime::EngineProgramContract` |
| State layout | `src/models/inkling2_loop/state/` | StateImage parts ([§7.2](#72-stateimage)) |
| Frontend | `src/models/inkling2_loop/frontend/` | pre-tokenizer and pipeline validation, renderer, output session, image preprocessing |
| Shared BPE core | `src/text/bpe*` (extracted) | the byte-level BPE merge loop, byte map and added-token split move out of `qwen3_5/frontend/tokenizer.cpp`, which then uses them; each family keeps its pre-tokenizer and validation. Two explicit consumers justify the extraction |
| Ops | `src/ops/relbias_attention/`, `src/ops/w4a4_sparse_moe/`, `src/ops/attn_res_mix/`, `src/ops/short_conv/`; contracts in `include/infernix/ops/` | [§9](#9-attention-ops)-[§12](#12-residual-mixing-message-gates-and-short-convolutions) |
| Kernel cores reused | `ops/offloaded_sparse_moe/cuda/narrow_a4_mma.cuh`, `narrow_rows_a4.cuh`, `wide_expert.cuh`; `ops/linear/bf16_p12/`; `ops/common/{rowsplit_ring,mbarrier,mma}.cuh`; `ops/kernel/sampling.cuh` | the expert cores move to `ops/common/` once two Ops use them; the offloaded Op keeps its planning |
| Runtime reuse | `src/runtime/prefix_cache/`, `src/runtime/speculative/` (calibrated length), Scheduler, EngineCore | model-agnostic, unchanged |
| Converter | `tools/convert/inkling2_loop.py` + recipe `inkling2_mini` | [§18](#18-artifact-and-conversion) |
| Tests | `tests/ops/` per Op; `tests/models/inkling2_loop/` | [§16](#16-numerics-and-qualification) |

The Qwen3.5 Program is not extended: the state (window rings instead of GDN state, 48 visits) and the forward differ
throughout, and the rule is explicit implementations per supported architecture.

---

## 7. State, KV and the prefix cache

### 7.1 Pools

| Store | Content | Per token | Notes |
|---|---|---:|---|
| Main Text (paged, P = 64) | global KV of the 8 full-attention visits | INT8-G64 16.5 KiB; BF16 32 KiB | visit-indexed layers 0-7 = (5), (11, p1..p3), (17, p1..p3), (23) |
| Draft Full (paged) | DSpark layer 1 context K/V | BF16 4 KiB | only with `--spec dspark`; BF16 as vLLM's SM12x draft KV |
| StateImage | rings and seeds below | fixed per lane | [§7.2](#72-stateimage) |

### 7.2 StateImage

Every sequence-local state is a ring indexed by absolute position, sized to its reach plus the largest verify width,
so a rejected draft never overwrites a position a later query needs and rollback is only the frontier:

| Part | Shape | INT8 profile | BF16 |
|---|---|---:|---:|
| Window K/V, 40 visits | R = 1040 slots × 8 heads × 128 × (K, V) | 84 MiB (INT8-G64) | 162.5 MiB |
| Conv histories, 48 visits | 4 rings (K 1024, V 1024, attn-out 2304, MLP-out 2304 channels) × 18 slots (3 taps + 15) | 11 MiB | 11 MiB |
| DSpark context rings, layers 0 and 2 | 2 × 1040 × 8 × 128 × (K, V) | 8.1 MiB (BF16 always) | 8.1 MiB |
| DSpark Markov seed `f_rnn`, continuation hidden, the last column's tap features | 4-6 × 2304 | < 0.1 MiB | < 0.1 MiB |
| **Total per lane** | | **~103 MiB** | **~182 MiB** |

R = 1024 + 16 covers T ≤ 16 (position p lives at slot p mod R). The ring holds the KV profile's rows (INT8-G64, or BF16
K and FP16 V as the paged `bf16` profile; BF16 V on the lossless route). The size matches the 27B's StateImage
(186.8 MiB), so the prefix cache's slot and Host-tier policies apply as they are.

### 7.3 Speculative rollback

Verify writes K/V and conv inputs for all T columns into the rings and the Main pages. Commit moves the lane's
frontier to the accepted length; rejected ring slots are overwritten by the next canonical write before any read
([paged KV §9.1](paged-kv-cache.md#91-independent-pool-frontiers)). There is no recurrent fold: nothing in the model
integrates over positions outside the window.

### 7.4 Prefix cache

A snapshot is the StateImage (rings whole at the snapshot frontier: [§14.4](#144-window-rings-and-taps-in-prefill)) plus
the Main and Draft Full pages of its path, as for Qwen3.5's hybrid layers. Taps, endpoint and output-boundary snapshots,
the Host slab tier, persistence and eviction are the [hybrid prefix cache](hybrid-prefix-cache-spec.md)'s, unchanged;
this family supplies the image layout, the tap Op (a Device copy of the image) and its cost coefficients (103 / 182
MiB per image; ~4-8 ms to restore over PCIe against ~0.3 s to recompute 4K).

The alternative (TokenSpeed's): window K/V in a paged pool whose pages behind the window are freed unless a cached
path holds them, with each page carrying its convs' checkpoints, so any page boundary is a restore point and
neighbouring snapshots share window pages. It needs new pool semantics (front trimming, snapshot-held window pages)
and is adopted only if M4's agentic replay shows Host-tier bytes, not hits, limit reuse.

### 7.5 Preemption and Replay

As Qwen3.5: on the reproducible route a request reserves its whole extent and is never paused
([engine architecture §3.1](engine-architecture.md#31-pause-and-resume)); otherwise Replay rebuilds from the deepest
snapshot.

### 7.6 KV profiles

Some paged profiles assume 256-wide heads (FP8-row256; VQ2/K4V2's H256 rotation). Inkling's heads are 128 wide:

| Profile | Status |
|---|---|
| `int8` (G64), `bf16` | first; INT8-G64 is head-width agnostic |
| `bf16p` (exact BF16 in 784/1024 of the bytes) | second wave (L25) |
| FP8 E4M3 with unit scales (vLLM's choice for this model) | candidate: ‖k‖ ≤ 22.6 and V's unit RMS keep every element inside E4M3's range without per-row scales (2 KiB per token per visit, no scale planes); qualified like any profile |
| `fp8` (row), `k8v4` | need a row-128 variant |
| `nvfp4` (G16) | width-agnostic; second wave |
| `vq2`, `k4v2` | need an H128 rotation and retrained codebooks: out of scope until asked |

K is L2-clipped (‖k‖ ≤ 22.6) and V is RMS-normalized per head, so both are bounded: favourable for 8-bit rows. Every
enabled profile is qualified ([model quality §6](model-quality.md#6-numerical-acceptance-policy-and-fast-paths)).

---

## 8. Decode pipeline

### 8.1 Round structure

A decode or verify round of B lanes × T columns runs embed + 48 visits + final mix + head + sampling as one Forward
graph (and the Finish graph for speculative units), per exact B and T bucket, every graph captured at startup (L22),
with PDL between consecutive kernels. Node count target: 24 × 8 + 24 × 10 + ~10 ≈ 440 (Flash-Next's verify rounds
launched ~1,440 kernels).

### 8.2 The visit chain

| Kernel | Reads | Does | Writes |
|---|---|---|---|
| K1 attention mixer | previous visit's per-expert partials (8 × H FP32 per column), conv ring, block sources | expert combine in slot order → BF16; post-MLP conv + residual; block bookkeeping; attention mix (1 or 3 queries); norm(s); DSpark tap copy on visits 5, 23, 41 | x_q (and x_k, x_v), BF16 |
| K2 projection | x_q, x_k, x_v; packed W (8,600 rows, + 64 dynamic-tap rows on full visits) | multi-input GEMV: row ranges take their own input vector; FP32 accumulate; epilogue writes each output in its consumer's layout (L6) | q, k, v, r, gate, sink, taps |
| K3 attention | K2 outputs, ring or pages | per-KV-head prologue: K/V conv (ring history), clips, v-norm, τ, append; relative-bias vector; split-K over keys; the last split CTA combines with the sink and multiplies σ(gate) | a, BF16 [3072] |
| K4 output projection | a, packed W_o | GEMV, epilogue: post-attention conv (per channel, ring), residual add or restart | h (BF16), conv ring |
| K5 MoE mixer | h, block sources | MLP mix, norm, four-over-six (or MXFP8) quantization of x | A4(x), BF16 x for the router |
| K6 router and route | x, W_router | FP32 router GEMV (few-row mapping, weights before the PDL wait); a one-CTA tail: sigmoid, bias, rank-based top-8, γ, per-expert job lists | dispatch |
| K7 gate/up | A4(x), dispatch, expert weights | narrow W4A4 (or W8A8); epilogue: softcapped SwiGLU × γ → BF16 h; per-(column, expert) row amax by `atomicMax` (order-independent, so deterministic) | h_e, amax_e |
| K8 down | h_e, amax_e | polls its job's completion count (L11); A4(h) (where: M5); W4A4 down | per-expert partials FP32 |

The combine of K8's partials moves into the next visit's K1, which needs them first; the order is the fixed slot
order of [§2.5](#25-rounding-points-pinned-at-m0), never arrival order. K2, K7 and K8 stream weights at the GEMV floor;
the others are latency-bound.

**Passes 2-3 of the core (24 visits).** A message gate needs `z = W_down [rms(x); rms(m)]` (64 rows over 4,608 inputs,
590 KB of weights) before `F z` and `G z` (2 × 2304 × 64) can gate any channel, and every consumer of the gated vector
needs all of it. So each mixer splits in two, as TokenSpeed's monokernel splits these visits: K1a (combine, conv,
residual, mixes, the q/k/v messages and the 192 `z` rows spread over the CTAs) → K1b (the F and G products, gating,
norms → x_q, x_k, x_v), and K5a (MLP mix, message, 64 `z` rows) → K5b (gate, norm, quantization). 10 kernels on these
visits, 8 on the others.

### 8.3 Why this split

- Mixers and norms run before the GEMVs, not inside their prologues (L5).
- K2 cannot take the per-head epilogues: a CTA owning a whole head (128 rows × 2304 × 1.5 B packed) leaves 67 CTAs on
  170 SMs (L2, L3). The conv, clips, v-norm and append move into the attention prologue, where each split CTA does
  128-wide vector work beside its key loads; an M5 op benchmark compares a separate head kernel (TokenSpeed's `head`
  stage).
- The attention combine runs in the last split CTA (a counter in a per-launch slot, reset by that CTA): cheaper than a
  cooperative barrier or a second kernel (L7).
- x is quantized once, in K5 (L10). h's quantization needs the row amax over 1152 values written by several K7 CTAs:
  K8 either quantizes its job's h rows in its prologue after polling (one column per job, weights already loading) or a
  small encode kernel runs between K7 and K8 (several columns per job); the M5 op benchmark picks per width class, and
  both give the same bits.

### 8.4 Grids

Each kernel's grid is fixed per (B, T) at startup to whole waves of 170 SMs (L3). The arithmetic units (a GEMV row
block, an expert job, an attention split) are fixed by the Op and its row's own data; the plan only maps units to
CTAs, so a CTA may run several units in sequence without changing any unit's arithmetic or a combine's order. K2 at
T = 1: 8,664 rows in 64-row units on 136 CTAs; at larger T the same units with more columns. Narrow experts: 8 experts
× ⌈2304 / 128⌉ = 144 gate/up units at one column. Window attention: 8 KV heads × 16 split units per lane, i.e. 128
units on 128 CTAs at B = 1 and 1,024 units at B = 8, mapped onto one or more whole waves.

### 8.5 The persistent visit kernel experiment

Not in the first implementation (L17). At M8, with the chain's kernels as its stages, a per-visit persistent kernel
is built and kept only if it wins end to end. It differs from Flash-Next's lost prototype in the recorded causes of
that loss: it covers the whole visit, weight-streaming stages included, and every task issues its weight loads
before it polls, so the latency-bound stages (mixers, routing, the attention prologue) overlap weight streaming
instead of serializing; and it is sized for co-residency from the start (one CTA per SM at a register cap set by
the heaviest stage, one shared-memory carveout for every stage, no 1,024-thread CTAs). Its other rules combine
Infernix's (grid = SM count, data produced in the launch read with `__ldcg`/`cp.async.cg`, launched as a PDL
consumer, counters self-reset) with TokenSpeed's (each stage bit for bit its chain kernel, so it is a pure speed change
on the reproducible route; plan-ordered polls for deadlock freedom; tagged mailboxes instead of cleared scratch; an
instruction-cache warm-up because streaming weights evict the code; placement tuned on a stage timeline). TokenSpeed
ships such a kernel as its decode path for this model on gfx950; no comparison with its regular path was read, and it
runs four-GPU tensor parallelism, so its result does not transfer as a number.

---

## 9. Attention Ops

### 9.1 Contract

One semantic Op family, `relbias_attention`, with two masks:

```text
inputs  q_raw [24×128×T×B], k_raw/v_raw [8×128×T×B], r_raw [24×16×T×B], sink [24×T×B], gate [3072×T×B],
        positions, valid columns, lanes, per layer: s_q [24×128], s_r [24×16], P [16×1024] BF16,
        conv taps (static [4×1024] per K and V, or static + per-column dynamic), mode (sliding | full),
        log scaling (full only), the ring (sliding) or the paged view (full), the conv rings
outputs a [3072×T×B] BF16; mutated: K/V and conv-input rings or pages at the live columns
```

The prologue (conv, clips, v-norm, τ) belongs to the Op because its outputs are observable only through the cache and
the attention output; the cached K/V values are an observable boundary (the KV codec).

### 9.2 Decode and verify (T ≤ 16)

- **Append, then attend.** The T new columns' K and V go through the KV codec into the ring or pages first, and every
  key and value a query sees, from this call or earlier, is the codec's represented value. So a column's result does
  not depend on whether its neighbours arrived in the same call: verify columns equal decode columns bit for bit, as
  speculation invariance needs ([§13.4](#134-draft-length-and-verification)).
- **Shape (L15).** 512-thread CTAs; K/V streamed in 64-key tiles through a 2-stage cp.async ring (64 KB in BF16, 33 KB in
  INT8: one CTA per SM); exponentials in parallel across the tile; query rows loaded vectorized; accumulators in
  registers.
- **Packing.** Per KV head, the 3 query heads × T columns form the rows (3T ≤ 48), so K/V tiles are read once per KV head.
- **Splits.** A split is an arithmetic unit over fixed absolute 64-key blocks, so a query's units depend only on its own
  position (sliding: the ≤ 17 blocks its window touches) and length (full: a fixed split count from the Engine's maximum
  context, TokenSpeed's 32 at 262K, each split's length from the row's own length rounded to 64 keys), never on B or T
  (L16). Units map to CTAs per (B, T) plan ([§8.4](#84-grids)); the last unit of a (head, column) to finish combines all of
  them in unit order.
- **Route per width.** The reproducible route runs one column-invariant schedule at every T ≤ 16 (each column's score,
  softmax and P×V in a fixed order whatever the other columns). A tensor-core verify route (from ~8 query rows per KV
  head, i.e. T ≥ 3) is a fast path, registered where faster (L16).
- **Bias.** Each CTA computes its rows' bias for the distances its keys span: `bias[d] = r · P[:, d]`, 16-deep
  products over ≤ 64 + T distances per tile, P's slice staged beside the tile. Distances ≥ 1024 add nothing.
- **Sink.** The combine adds `exp(sink - m)` to the denominator only.

### 9.3 Prefill

FlashAttention-style tiles: 64 query positions × 3 heads (192 rows) per CTA against 64-key tiles. The chunk's K/V is
appended through the codec first, and every key tile, in the chunk or before it, is read as the codec's represented
value (from the ring or pages, or recomputed in registers by the codec's exact rule), so prefill columns match any
other chunking of the same prompt ([§14.1](#141-chunk-width-invariance)). A (query tile, key tile) pair spans ≤ 127 distances: its
bias tile is one [192 × 16] × [16 × 128] MMA into shared memory, gathered along diagonals. Sliding visits skip key
tiles outside the window; full visits skip the bias beyond distance 1024. P×V in FP16 on the exact route, as Qwen's
prompt attention; longest-first CTA order and proportional shares as the BF16 prompt kernel. One prefill route at every
width (a width threshold broke prefix restores on Flash-Next: L16).

### 9.4 Formats

INT8-G64 and BF16 first (rings and pages), in decode and prefill kernels; other codecs per [§7.6](#76-kv-profiles). The
lossless route has its own attention kernel (exact scores, correctly rounded softmax, BF16 V), as for Qwen.

---

## 10. MoE Ops

### 10.1 Contract

`w4a4_sparse_moe`: inputs x [H × columns] BF16 (or its quantized form), the dispatch, the experts of one layer (NVFP4 or
MXFP8 family), the softcaps; outputs per-expert FP32 partials (decode: combined by the next K1 in slot order) or the
BF16 combined output (prefill). Routes: narrow (decode, verify) and wide (prefill), each qualified against the FP64
oracle of [§16.3](#163-op-oracles).

### 10.2 MoE mixer, router and dispatch

K5 produces the normalized x and its quantized form (codes, block scales, outer scale) once per column (L10). K6 runs
the 128-row FP32 router with the few-row mapping and its weights fetched before the PDL wait, then a one-CTA tail
(≤ 32 columns, L14): sigmoid, bias, top-8 by rank counts (each candidate counts the scores above it; ties by index:
exact, no serial arg-max), γ, and per-expert job lists. Padded rows of a graph batch route no expert (L12).

### 10.3 Narrow routes (decode, verify)

- **NVFP4 W4A4.** Row pairs for 1-3 columns per job and the tensor-core `narrow_a4_mma` core for 4-8 (L9), bit-equal to
  the exact int64 sum, adapted to a per-column outer scale and a per-expert weight factor in the epilogue in the pinned
  order ([§2.5](#25-rounding-points-pinned-at-m0)). kMaxColumns ≥ B_max × T_max = 128, so every decode and verify call
  stays narrow and its bits never depend on other requests. A job of 9-16 columns runs two passes over
  register-resident weights rather than one wide tile (L16's cliff); a job above 16 columns (possible only at C > 1)
  runs successive 16-column passes with the same per-column arithmetic, exact and linear in its columns. A faster wide
  form for such jobs would be a fast path.
- **gate/up epilogue.** Interleaved rows put each (g, u) pair in adjacent accumulators: softcapped SwiGLU × γ in FP32,
  BF16 h, and the row's amax by `atomicMax`.
- **down.** Polls its job's completion count; quantizes h by the four-over-six rule from the finished amax; K = 1152 runs
  as 9 stages of 128 (no masked tail).
- **MXFP8 W8A8 (layers 20-23).** Native `mxf8f6f4` block-scaled MMA, FP32 accumulation in one fixed K order per column;
  same epilogues.
- **Layout.** The converter stores each expert's codes and scales in the narrow kernels' tile order (value-preserving).

### 10.4 Wide routes (prefill)

Grouped GEMMs over the chunk's (token, expert) rows on Flash-Next's pipeline (L13): one producer and eight consumer
warps, `mxf4nvf4` m16n8k64 (`mxf8f6f4` for MXFP8), 64 × 128 × 128 tiles, three ~43 KB stages, 2 CTAs per SM. Inkling
differs in one place: A4(h) needs the row amax over all 18 N-tiles of a (token, expert) row, so the gate/up epilogue
writes BF16 h and an `atomicMax` amax per row, and the down GEMM's producer warps quantize h tiles as they load them,
after the row's job counter completes. The quantize-permute of x (four-over-six per token, then scatter to expert-sorted
rows, zeroing each expert's scale tail in the CTA writing its last row as SGLang does) is one kernel. SGLang's tile
choices measured on an RTX 5090 for these shapes seed the tile search. The combine is FP32 in slot order, rounded once.

---

## 11. Dense projections and the head

- **Packing (L4).** A matrix loads as `bf16_p12` only when every reader takes the packed form and M1's per-shape sweep
  shows the packed routes faster at the decode and verify widths: expected for the K2 parents [8,600 (+64) × 2304]
  (Flash-Next's similar shapes: 1.26× at T = 1-4, 1.13-1.19× at 5-16) and the head; W_o [2304 × 3072] only if a direct
  MMA wins (Flash-Next's output projections needed one to reach 1.00-1.06×); routers and gates stay BF16. The embedding is
  packed for VRAM (its reader is a row gather). Packing runs on the GPU after upload (~30 ms on Flash-Next). The skinny
  packed GEMV gives the BF16 skinny route's bits at T ≤ 8 and runs 16 columns a pass under `A16ColumnInvariant`, so packing
  changes no reproducible-route bit. There is no second BF16 copy: prefill and wide verify (B × T of 33-1,024 columns) pay
  the packed routes' 0.67-0.96× on those projections, a few percent of prefill time.
- **K2.** A multi-input packed GEMV: row ranges bound to one of three input vectors (one pointer on the 20 pass-1 sliding
  visits, three elsewhere); new tuned routes for Inkling's shapes (`bf16_p12_linear_tuned` registers only measured shapes).
- **K4.** GEMV 2304 × 3072 (packed or BF16 per M1) with the post-attention conv and residual in the epilogue (each output
  channel needs only its own history).
- **Few-row matrices (L3).** Router (128), sink (24, inside K2), relative (384, inside K2), taps (64), gate-down (64 per
  gate): one row per warp, every load before the FMAs.
- **Head.** Packed GEMV over 200,064 rows (`projection_fp32` takes packed heads whose row count is a multiple of 16: the
  200,058 real rows plus 6 zero rows, masked before the softcap), softcap 30 in the epilogue, FP32 logits; the
  checkpoint's other padded rows are never stored; one read for all T columns; evict-first loads (read once per step).

---

## 12. Residual mixing, message gates and short convolutions

**`attn_res_mix`.** Per column: ≤ 5 BF16 sources → per-source `rms` (FP32), dot with the fixed query, softmax, weighted
sum; optional message mix over 2 sources and the rank-64 gate; the layer's RMSNorm; BF16 output. Block bookkeeping
(`h6`, `h12ᵖ`, `h18`) is a pointer switch in the visit driver: sources live in a per-column workspace (e, h6, the current
and previous pass's h12 and h18, the partial sum). Two forms:

- direct (decode default): the mix reads its ≤ 5 rows (≤ 23 KB per column);
- partials + combine (TokenSpeed, Kimi-K3): at each block boundary one sweep computes, for every query of the coming
  block's layers, the online-softmax statistics over the completed blocks; each mix then folds only the current partial
  sum. Prefill and wide verify use it; the pass restart reuses the prelude's statistics; decode adopts it if M5 measures
  it faster. Both forms are qualified against the same FP64 oracle; their FP32 association differs (a pinned point).

**Message gates.** FP32 with every operation rounded separately (pinned at M0), in K1a/K1b and K5a/K5b.

**`short_conv`.** Four kinds share one contract (static width 1024 or 2304, dynamic per-head taps, residual scale 1/√5
or 1/√9), always fused into a neighbour (K3 prologue, K4 epilogue, K1) with the ring as state. Prefill runs them over
the chunk with the ring as left history and writes the last 18 positions back.

---

## 13. Speculative decoding: DSpark

### 13.1 Mathematics (upstream: SGLang `inkling2_mtp.py`, `dspark.py` and Triton kernels; vLLM PR; TokenSpeed)

```text
features at processed position t: f_t = [x_in(visit 5), x_in(visit 23), x_in(visit 41), final_state]   (4 × H)
α = softmax(fuse_weight) per row: rows 0-2 for the draft layers, row 3 for the Markov head
c^l_t = norm_dflare,l( Σ_j α_{l,j} f_{t,j} )                        context feature of draft layer l
K_ctx^l = rope(norm_kctx,l(W_kctx,l c^l)),  V_ctx^l = W_vctx,l c^l   (per processed position)
block input (positions F..F+7): [embed(anchor), mask_embed × 7]
draft layer: DFlash-style (input norm, q/k norms, RoPE θ 1e6, attention over context + block,
             SwiGLU 9216); layers 0, 2 sliding 1024 with the window anchored at the block start, layer 1 full
base_i = W_head norm_final(x_i) for i = 1..7         (the target head; the anchor is not sampled)
Markov head (sequential over i):
   u_i = rnn_proj([s_{i-1}, w1[tok_{i-1}], x_i, f_rnn])          (768 = gate, candidate, output)
   g = σ(u_gate);  s_i = bf16(g ⊙ s_{i-1}) + bf16(bf16(1-g) ⊙ bf16(tanh(u_cand)));  ℓ_i = tanh(u_out)
   bias_i = Fold ℓ_i + fold_bias,   Fold = W_head · W_hidden_out  [vocab × 256],  fold_bias = W_head b_hidden_out
   logits_i = softcap_30(base_i + bias_i) over the real vocabulary;   tok_i ~ logits_i
s_0 = 0, tok_0 = anchor, f_rnn = norm_dflare,markov(Σ_j α_{3,j} f_{F-1,j})
```

### 13.2 Ownership

A speculative backend in the family (`execution/dspark.cpp`) sharing the runtime's round driver, commit, the
calibrated-length policy (`runtime/speculative/`) and later tree verification. Context materialization reuses
`context_kv_materialize` where its contract fits (per-layer feature mixtures are new inputs).

### 13.3 Taps

K1 of visits 5, 23 and 41 copies its input row, and the final mix its output, into a feature buffer per committed
column (4 × 4.6 KB), consumed by the drafter's context append and by `f_rnn` from the last committed column. With
DSpark on, the last visit cannot be trimmed to the logit columns in prefill (every column's final state is a feature).

### 13.4 Draft length and verification

- Proposal distribution: the softcapped Markov logits (sampled) or their argmax (greedy); a proposal's confidence is its
  probability there, at no extra cost.
- Length: the per-request length calibrated by confidence deciles (L21), with this model's cost line `t = a + bK` measured
  at M6; device-side IF nodes skip the remaining Markov steps once the confidence falls below the calibrated cut.
- Verification: chains. Trees lost on the Flash-Next MoE (L21) and the expert union makes them costlier here; they are
  measured after M6, not built first.
- Reproducible route: Gumbel-max sampling, so a draft is accepted exactly when it equals the token plain decoding draws,
  and speculation does not change output ([model quality §6.1](model-quality.md#61-the-reproducible-route-and-the-fast-paths)).
  This needs verify columns bit-equal to decode columns: every decode Op is column-invariant (L16), and attention reads
  the verify columns' own keys as stored codec values ([§9.2](#92-decode-and-verify-t--16)).

### 13.5 Draft cost and fusion

Per round: 3 dense layers over the 8 block columns, context appends for the newly committed columns, the head over 7
columns (one read), then 7 sequential Markov steps (a 5120 → 768 projection, the gate nonlinearity, a [vocab × 256]
product with softcap and sampling). One kernel per step, the next step's projection in the same kernel's tail CTA,
gives 7 kernels; the IF nodes stop early.

### 13.6 Drafter storage

| Form | Bytes read per round | Draft time (model) | Output effect |
|---|---:|---:|---|
| BF16 as shipped (Fold in BF16, 102 MB) | 0.52 GB layers + 0.69 GB packed head + 7 × 0.10 GB Fold ≈ 1.9 GB | 1.1-1.4 ms | none on the reproducible route (acceptance only) |
| 8-bit dense layers, FP8 Fold (51 MB: stays in L2 across the 7 steps, L18), Q4 proposal head | ~0.55 GB | 0.5-0.6 ms | acceptance only |

Fold is derived by the converter in FP64 from the head and `hidden_out` and stored rounded as the reference's
`refresh_fold`. Infernix already re-quantizes drafters (DFlash2 companions in 8 bits). The choice is D2.

---

## 14. Prefill

### 14.1 Chunk-width invariance

Every prefill Op computes a column's result independently of the chunk's width and start: no split-K whose split
depends on the call, one attention route at every width, per-token quantization, and attention that appends the chunk's
K/V through the codec before reading it, so in-chunk and cached keys are the same represented values
([§9.3](#93-prefill)). The model has no recurrent layer whose kernel chunking regroups sums (its convs read a 3-position
history), so, as TokenSpeed notes for KV-only models, a prefix resumes at any token bit for bit. Then publication and
resumes are exact at any prefill-computed frontier, chunks can be wide (default 16K, wider while no lane decodes), and
the prompt's last two calls merge (L23). Where an Op cannot be made invariant, the Qwen rule applies instead: a canonical
chunk grid and grid-only publication on the reproducible route.

### 14.2 Pass schedule

For a chunk of C columns: prelude visits over all C, then each core pass over all C, then the coda. Per-column workspace
holds `e`, `h6`, the current and previous pass's `h12` and `h18`, and the partial sum (6 × C × H BF16 = 113 MB at
C = 4,096, 453 MB at 16K), plus the attention-residual statistics and the MoE workspace for C × 8 rows; M4 sets the
default chunk against the KV it displaces.

### 14.3 GEMM routes

| Call | Route |
|---|---|
| K2, K4 projections | the packed TMA MMA ring (warp-specialized, fenced) for packed parents: 0.67-0.93× of BF16 at 65-1,024 columns, 0.92-0.99× beyond on Flash-Next (no BF16 copy is kept); BF16 TMA ring for unpacked W_o |
| Experts | grouped `mxf4nvf4` / `mxf8f6f4` pipeline ([§10.4](#104-wide-routes-prefill)) |
| Router | tiled FP32 router (Flash-Next's took 755 → 370 µs at ≥ 64 columns) fused with top-8 and the quantize-permute |
| Mixers, gates, convs | partials + combine for the mixes; elementwise kernels over the chunk |

### 14.4 Window rings and taps in prefill

Sliding visits attend to the chunk's K/V, held in a workspace buffer as the codec's represented values (a 16K chunk does
not fit the 1,040-slot ring; [§9.3](#93-prefill)), and to the ring's last 1023 positions before the chunk; after the chunk
only its last R positions are written to the ring, and only the positions some read can reach (the next chunk, a planned
tap) are projected into the drafter's context rings (Infernix's DFlash2 ring rule: the drafter's share of a 64K first
token fell 220 → 75 ms). A tap is a Device copy of the StateImage; the rings are whole at any frontier because R ≤ the
chunk width.

---

## 15. Vision

Second track (M7). Upstream (SGLang `inkling2_vision.py`): patches of 20 × 20 × 3 (a 1200 → 1024 linear), the learned
32 × 32 position table bilinearly resized to the image's patch grid, a CLS token per image, 23 pre-norm blocks
(LayerNorm with bias, attention with 2-D rotary on q/k, exact-GELU MLP, layer scale on both branches), attention within
each image only, 2 × 2 merge (4096), `proj_in` 4096 → 4096, GELU, `proj_out` → 2304, replacing the image's placeholder
tokens. The text model has no positional encoding beyond the relative bias, so image tokens are ordinary positions (no
MRoPE). Ops: the existing BF16 GEMMs (packed where faster), a varlen 2-D-rotary attention (as Qwen3.5 Vision's),
LayerNorm and bilinear-resize kernels. Media acquisition and prefix-cache media keys reuse Qwen's.

---

## 16. Numerics and qualification

### 16.1 Routes

| Route | Experts | Projections, attention, small ops | Head |
|---|---|---|---|
| Reproducible | narrow at decode/verify widths (W4A4: exact sums; W8A8: FP32 accumulation in one fixed K order per column), wide grouped MMA in prefill; four-over-six and MXFP8 quantization per token | BF16 A16 (packed storage, same bits), column-invariant decode; request-invariant attention splits; P×V FP16 in prefill | FP32 |
| Default | the reproducible route plus fast paths registered where measured faster (none at first) | | |
| Lossless | correctly rounded: exact products and sums, one rounding per reference operation, quantizers as exact operations | correctly rounded transcendentals and softmax, BF16 KV with exact V | FP32 |

The lossless route is the model-level reference ([lossless route](lossless-route.md)), as for the two existing families.
Its authority for the experts is the declared quantized arithmetic of [§2.4](#24-numerics-the-checkpoint-declares), not
Transformers' dequantized path; open points are D4.

### 16.2 Reproducible route

Request invariance from the start: decode projections under `A16ColumnInvariant`; attention splits from fixed key chunks
and the row's own length; experts narrow at every decode width (kMaxColumns ≥ 128); per-token quantizers; graphs per
exact B; chunk-width-invariant prefill; no pausing.

### 16.3 Op oracles

| Op | Oracle | Criterion |
|---|---|---|
| four-over-six and MXFP8 quantizers | exact: the pinned FP32 operation sequence emulated with exact checks; every code and scale compared | bitwise |
| W4A4 / W8A8 experts | FP64 over the decoded weights and the quantized activations; softcapped SwiGLU and γ in FP64 | narrow W4A4: bitwise to the exact sum; narrow W8A8 and wide: normwise + gross cap |
| relbias_attention | FP64 over the represented q, k, v (after the prologue's own oracle), bias, sink | normwise + gross cap per KV format |
| attn_res_mix (both forms), message gates, short convs | FP64 | pointwise / normwise |
| router + top-8 | FP64 logits; exact ranks with the pinned tie rule | bitwise selection outside documented near ties |
| packed projections and head | FP64; packing round trip exact | normwise; packing bitwise |
| DSpark Markov step | FP64 with the reference's BF16 rounding points | pointwise |

Every pipelined or counter-based route runs `verify_repeatable` with a side stream (L20); every bit-identity claim uses a
fixture where summation order is visible ([op development §6.3](op-development.md#63-acceptance-criteria)).

### 16.4 Model-level quality

- Against the lossless route: KL, top-1 agreement and NLL over the quick corpus at several chunk widths, prefill and
  decode arms ([model quality](model-quality.md)), per KV format.
- Against vLLM and SGLang on this machine (same arithmetic up to the pinned points): greedy text and logit KL on a fixed
  prompt set, as cross-implementation checks, not gates.
- Speculation invariance (plain vs DSpark, greedy and sampled) and request invariance (C = 1/2/4/8, prefix hit/miss) on
  the reproducible route.

---

## 17. Frontend and serving

- **Tokenizer.** The shared BPE core plus an Inkling pre-tokenizer implementing the o200k-family split exactly (a
  hand-written matcher like Qwen's, with the pattern string validated byte for byte against `tokenizer.json`) and
  `ignore_merges`. Qualified against Hugging Face `tokenizers` on a multilingual and code corpus (exact ids).
- **Renderer.** The official `chat_template.jinja` through the existing Jinja engine (it has macros, `namespace`,
  `raise_exception`, `tojson(sort_keys, separators)`, `dictsort(case_sensitive)` and `generation`); boundary hints and
  next-turn rendering as Qwen's.
- **Reasoning effort.** OpenAI `reasoning_effort` and Anthropic thinking budgets map to the template's effort (strings as
  its table; numbers in [0, 0.99]); the default is the template's 0.9 (D6).
- **Output session.** Channels from `response_template`: thinking, content, tool calls (JSON per call). Constrained tool
  calls use the existing JSON grammar inside `<|content_invoke_tool_json|>` (TokenSpeed's `inkling2_json` frames the same).
- **Stops.** EOS `<|content_model_end_sampling|>`; `<|end_message|>` ends a channel, not the turn.

---

## 18. Artifact and conversion

### 18.1 Recipe

`inkling2_mini`: every tensor bit-exact; experts as `nvfp4_e2m1_g16_e4m3` with `amax/denominator` as each expert's FP32
global scale, and `fp8_e4m3fn_g32_e8m0` (MXFP8); BF16 elsewhere (packed at load, not in the file); FP32 queries and router
bias. Derived: the message queries merged to [12, 4, H] in (q, k, v, mlp) order; the head cut to 200,064 rows (the real
200,058 plus 6 zero rows, a multiple of 16 for the packed head) and the embedding to 200,058; DSpark `Fold` and `fold_bias` computed in FP64 and stored rounded as the reference does (D2 may
store FP8). The `.conversion.json` report lists every tensor's hash and the `bf16_p12` census (bits per weight and escape
rate per matrix).

### 18.2 Layouts

Experts in the narrow kernels' tile order (codes and scales), gate/up interleaving kept (the fused epilogue wants it);
attention matrices concatenated per layer into the K2 parent [8,600 (+64) × 2304] and W_o; controller weights stacked per
layer; the head row-major. All value-preserving.

### 18.3 Frontend resources

`tokenizer.json`, `tokenizer_config.json` (with `response_template`), `chat_template.jinja`, `generation_config.json`,
`processor_config.json`, validated at conversion against the pipeline the runtime implements (as
`tools/convert/resources.py` does for Qwen).

---

## 19. Configuration surface

| Option | Values for this family |
|---|---|
| `--spec` | `none`, `dspark`, `auto` (= `dspark` when the artifact carries the drafter) |
| `--draft-tokens` | `auto` (calibrated), 1-7 |
| `--kv-dtype` | `int8` (default), `bf16`, later `bf16p`, `nvfp4`, `fp8`, `k8v4` |
| `--vision` | on when the artifact carries the tower (M7) |
| `--prefill-chunk` | default 16K if chunk-width invariance holds ([§14.1](#141-chunk-width-invariance)) |
| `--reproducible`, `--lossless` | as for every family |

No family-specific tuning switches ship; experiments use temporary `INFERNIX_TMP_*` switches removed before merge.

---

## 20. Memory plans

Model: usable device memory ~30.3 GiB (estimated after the CUDA context and WDDM); weights 18.19 GiB as stored, ~17.5 GiB
with the K2 parents, head and embedding packed (model: −0.65 GiB at 12 bits); drafter 0.59 GiB + Fold 0.10 as shipped
(under D2: 8-bit layers ~0.30, FP8 Fold 0.05, a Q4 proposal head ~0.24, about the same total); workspace and graphs
~1.4 GiB (estimated, 4K chunk; a 16K chunk adds ~0.5 GiB); 2-3 Device snapshot slots of one StateImage each (~0.3 GiB at
INT8) → **~10.4 GiB for lane state and KV**.

| Plan | State (lanes × image) | Main KV per token | Draft KV per token | Total context in ~10.4 GiB |
|---|---:|---:|---:|---:|
| C = 1, INT8, DSpark | 0.10 GiB | 16.5 KiB | 4 KiB | ~525K |
| C = 1, BF16, DSpark | 0.18 GiB | 32 KiB | 4 KiB | ~295K |
| C = 4, INT8, DSpark | 0.40 GiB | 16.5 KiB | 4 KiB | ~510K shared |
| C = 8, INT8, plain | 0.80 GiB | 16.5 KiB | — | ~610K shared |

The advertised 1M context at C = 1 needs `nvfp4` global KV (~9 KiB per token) or INT8 without the drafter and a smaller
workspace [M4 measures the real headroom].

---

## 21. Implementation plan

Each milestone ends with its tests passing; every performance choice left open has an op benchmark before integration
(L27). Work happens in a worktree off main; nothing merges with a temporary switch left in.

| Milestone | Content | Exit |
|---|---|---|
| M0 Reference | pin [§2.5](#25-rounding-points-pinned-at-m0) from the merged references; read vLLM's and TokenSpeed's attention kernels in full; run vLLM's and SGLang's PRs on this machine (WSL) for reference logits on a fixed prompt set; FP64 per-Op reference fixtures | rounding points recorded; the attention contract of [§9.1](#91-contract) confirmed; fixtures in `tests/` |
| M1 Artifact and frontend | recipe, loader bindings, layouts, Fold; `bf16_p12` census and per-shape packed-vs-BF16 sweep at T = 1-16 for K2, W_o and the head; BPE extraction, tokenizer, renderer, output session, parsers | bit-exact conversion report; packing decision per matrix; tokenizer ids equal HF's on the corpus; templates render as Jinja2 does |
| M2 Lossless route | every operation correctly rounded, one lane, no speculation | lossless logits agree with vLLM/SGLang within quantization-level noise; CausalScoring runs |
| M3 Ops | relbias_attention (decode, verify, prefill; INT8, BF16), w4a4_sparse_moe (narrow, wide; NVFP4, MXFP8), packed multi-input projections, attn_res_mix (both forms), short convs, head | each Op qualified; op benchmarks at the real shapes recorded |
| M4 Program | StateImage, Main/Draft pools, rollback, prefix cache binding, Replay, graphs per (B, T), PDL chain, chunk-width invariance | reproducible route request-invariant (C = 1/2/4/8, prefix hit/miss, chunk widths); model-level gates vs lossless |
| M5 Decode performance | nsys round anatomy; f(T) against [§4.3](#43-plain-decode)'s table; the open choices: h quantization point, attention prologue vs head kernel, in-kernel vs materialized relative bias, decode mix form, tensor-core verify attention (fast path) | plain decode ≥ 185 tok/s (must) |
| M6 DSpark | drafter, taps, Markov steps with IF nodes, calibrated length line, speculation-invariance test; D2's form | ≥ 1.2× plain on code-like text (must) |
| M7 Vision | tower, preprocessing, media keys | image prompts match vLLM's text on a fixed set |
| M8 Prefill, long context, persistent kernel | wide routes, prefill attention, chunk defaults; the persistent visit kernel experiment ([§8.5](#85-the-persistent-visit-kernel-experiment)) | targets of [§1.3](#13-targets); the persistent kernel kept only if faster end to end |
| M9 Acceptance and docs | the full report against vLLM, SGLang and llama.cpp; authorities of [§22](#22-documentation-and-authority-changes); this file removed | every target reported, met or missed |

---

## 22. Documentation and authority changes

- `AGENTS.md`, `README.md`: the product line and model list gain `Inkling2LoopForConditionalGeneration` (owner's wording).
- New `docs/maintainer/inkling2-model.md` (mathematics, config, logical parameters, state, DSpark), as `qwen3_5-model.md`
  is for Qwen.
- `engine-architecture.md` §1.1 and §8 (entry points, locations); `tensor-formats.md` (the four-over-six activation rule;
  MXFP8 activations); `lossless-route.md` (an Inkling operations section); `paged-kv-cache.md` §9.2 (window rings in the
  StateImage); `hybrid-prefix-cache-spec.md` §3.2 (the image); `dflash.md` or a new `dspark.md`; `docs/serving.md`,
  `docs/cli.md`; a model card.

---

## 23. Open decisions, risks and rejected alternatives

### 23.1 Owner decisions

| # | Decision | Options (recommendation first) |
|---|---|---|
| D1 | Add the family to the product | yes, per this design / not now |
| D2 | Drafter storage | 8-bit dense layers + FP8 Fold + Q4 proposal head (acceptance-only effect, ~0.6-0.8 ms per round saved) / BF16 as shipped |
| D3 | Default KV | `int8` (as Qwen) / `bf16` |
| D4 | Lossless authority for the expert block's transcendentals, γ placement and accumulator rounding | the trained arithmetic as the reference kernels define it, each operation correctly rounded / Transformers' order |
| D5 | Vision in the first release | second track (M7) / with M4 |
| D6 | Serve default reasoning effort | the template's 0.9 / lower |
| D7 | Opt-in lossy recipe with FP8 attention projections (W8A16; ~1 GB less per token after packing) | not now / build after M9 as a separate artifact |
| D8 | Variable-loop research checkpoint (pass count option) | later / with M4 |

### 23.2 Risks

- **Reference churn.** All four implementations were open PRs on release day and already differ at the pinned points;
  M0 re-reads them. A later change to a rounding point changes oracles, not the design.
- **Per-visit fixed cost.** The model's f = 29-52 µs at T = 1 is built from other kernels' records; the window attention
  (10-18 µs of it) and the route (3-9 µs) are new code. At the high end plain decode is ~185 tok/s, at the must target.
- **Packing ratio and speed.** The model assumes the K2 parents and head pack near Flash-Next's 12.02 bits and run
  1.13-1.26× faster at decode widths; without packing plain decode is ~190 tok/s at the central f. Prefill pays the
  packed routes' 0.67-0.96× at mid widths.
- **Group-3 GQA tiling** under ~99 KB shared memory (vLLM already overflowed it on SM12x).
- **Four-over-six bit-exactness** depends on FP32 sequence details (reciprocal forms, sum order, ties); M0's exact oracle
  settles it.
- **DSpark acceptance** on chat and reasoning text is unpublished; the published 3.55-4.07 is GSM8K at temperature 0
  and 3.59 random prompts.

### 23.3 Rejected alternatives

| Alternative | Why rejected |
|---|---|
| Dequantize experts to BF16 and run A16 (Transformers' route) | not the trained arithmetic; 3.6× expert bytes |
| Window KV in paged pools with every position kept | 192 KiB per token with the global KV: ~55K context at BF16 |
| Window ring of exactly 1024 slots | a rejected draft would overwrite keys the next query needs |
| Conv states as 3-slot buffers with copy-on-commit | a commit copy per visit per lane; the 18-slot ring makes rollback free |
| Mixes and norms recomputed inside GEMV prologues | 2-4× slower prologues (L5) |
| Reusing the Qwen3.5 Program | different state and forward throughout; explicit implementations per architecture |
| Shared KV across loop passes | changes the model (the research checkpoint's option) |
| A persistent decode kernel from the start | Infernix's measured losses (L17); kept as the M8 experiment |
| Split-K decode GEMVs and split-K last waves | measured losses (L27) |
| 8-bit activations for dense decode projections | measured losses (L24) |

---

## 24. Sources

Read (in full, or the named parts in full):

- Checkpoint `thinkingmachines/Inkling-2-Mini` @ 6e7c8248 (2026-11-20): `config.json`, `hf_quant_config.json`,
  `tokenizer.json` (pipeline), `tokenizer_config.json`, `chat_template.jinja`, `processor_config.json`,
  `generation_config.json`, the safetensors headers of `model-00001-of-00001.safetensors` and `mtp.safetensors`; the
  model card.
- Thinking Machines' release post (the training-precision, pruning and looping paragraphs); the Hugging Face blog
  `inkling2mini` (loop, DSpark taps, the SGLang B300 chart).
- huggingface/transformers#51050 (`add-inkling2-mini`): `modular_inkling2_loop.py`, the PR's `conversion_mapping.py` and
  `core_model_loading.py` changes, the model doc; Transformers main `modeling_inkling2.py` (norms, clips, relative
  logits, convolutions, attention, softcapped SwiGLU, experts, router, block attention residual, decoder layers).
- vllm-project/vllm#69028 (WoosukKwon/vllm @ b426b46e): `docs/models/inkling2.md`, `speculator_config.py`,
  `ops/nvfp4.py` (quantizer), `ops/activation.py` (SwiGLU and quantize), `ops/mxfp8.py` (`_mxfp8_block`),
  `ops/moe_sm120.py` (expert GEMMs), `moe.py` (forward), the PR description.
- sgl-project/sglang#51926 (zcnrex/sglang @ 92e894ef): the PR description, `kernels/ops/gemm/inkling2_grouped_sm120.py`;
  SGLang main `models/inkling2_mtp.py`, `kernels/ops/speculative/inkling2_mtp.py`.
- lightseekorg/tokenspeed#3682 (@ b4555ee6): the PR description (performance tables), `docs/design/monokernel.md`, the
  gfx950 Inkling2 monokernel README, `tokenspeed_kernel/ops/residual/README.md`, `docs/design/numerics.md` (trainer
  alignment; prefix hits and retraction).

Skimmed (headers, docstrings, or searched for specific terms):

- Transformers PR `modeling_inkling2_loop.py` (selected ranges); vLLM PR `attention.py`, `loop.py`, `model.py`,
  `sconv.py`, `conv_mix`; SGLang PR `sm12x_bf16_gemv.py`, `routed_quant_sm120.cuh`, `inkling2_experts_sm120.py`,
  `elementwise/inkling2_loop.py`, `attention/inkling2_attn.py`; SGLang main `models/dspark.py`, `models/inkling2_vision.py`;
  TokenSpeed main's Inkling cache recipe (`kv_cache/recipes/inkling.py`), attention backend, drafter, grammar, MoE and
  quantization modules, and the PR's `models/inkling2_loop.py`. No engine's attention kernels were read line by line;
  M0 reads vLLM's and TokenSpeed's before the attention Op's contract is fixed.

Infernix:
- [engine architecture](engine-architecture.md), [op development](op-development.md),
  [paged KV](paged-kv-cache.md), [hybrid prefix cache](hybrid-prefix-cache-spec.md), [DFlash](dflash.md),
  [tree verification](tree-verification.md), [model quality](model-quality.md), [lossless route](lossless-route.md),
  [Qwen3.8-Flash-Next design](qwen3_8-flash-next-design.md) (the measured lessons of [§3.2](#32-lessons-from-infernixs-record-applied-here)),
  [performance](../performance.md).
