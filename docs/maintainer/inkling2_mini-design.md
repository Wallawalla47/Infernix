# Inkling 2 Mini on one RTX 5090: design

**Status: implemented.** This is the design record of `Inkling2LoopForConditionalGeneration` (Thinking Machines'
Inkling 2 Mini, model type `inkling2_loop`) in Infernix. Parts were written before the code: where they differ, the
[user guide](../inkling2-mini.md), the [CLI reference](../cli.md) and the code are the authority.

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
  calibrated per-request draft length (n-gram copy proposals are refused);
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
| NInfer-all (`iamwavecut/ninfer-all` @ 395ce8b8, Apache-2.0; its `docs/inkling2.md`) | Unsloth's UD-IQ4_XS GGUF: NVFP4 experts (20 layers) and Q5_K/Q4_K/IQ4_XS/Q5_0 (4), multiplied as GGUF vector kernels against q8_1 activations (W4A8, not the trained W4A4) up to 8 tokens; expert products summed in 2^-32 fixed point, so the CPU/GPU split cannot change them | F16 KV; FMA kernels (no tensor cores), keys split over ranges at decode, 8 rows a block at prefill; a sliding visit's prefill attends over a staging ring (the window before the chunk copied in, the last window copied back) | FP32 residual rows; ~15-20 launches a visit; at ≤ 16 rows each depth mix spread over one-warp blocks of 128 channels whose readers add the warp sums in the one-block order (same bits); a CUDA graph per batch size | reported on an RTX 5090: 182.1 tok/s decode (near-empty context), 163.6 after 10K, prefill 8,367 tok/s at 2K and 6,664 at 10K; DSpark 1.19× (essay, 1.71 tokens a round); measured on this machine: [§3.4](#34-ideas-from-ninfer-all-reviewed-2026-11-21); also prefix reuse, vision, structured output, host and disk experts. Q8_0 attention and a Q6_K head move ~3.5 GB a token against this design's ~5.1 GB with the head packed (estimate) |

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
| L9 | Narrow W4A4: s8 `mma.m16n8k16` on doubled codes at every width (bc8bd7da retired the register-resident row pairs for 1-3 columns: under VRAM pressure they fell into a slow state that outlasted the process, 80-130 against ~185 tok/s; the tensor-core kernels measured 30.8 → 34.8 µs at T = 1 and faster at T = 2-3); E2M1 decode by `cvt.rn.f16x2.e2m1x2` (a `constexpr` table compiled to local memory: 0.58 TB/s) (FN §19.3.21, §19.3.52-53) | the narrow expert route starts from the tensor-core kernels ([§10.3](#103-narrow-routes-decode-verify)) |
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
| L24 | 8-bit activations for dense decode projections lost 15-27 % end to end (FN §19.3.31) | the default's decode projections stay A16 (also under D7); the opt-in `--a8-activations` gives every dense Q8 projection 8-bit inputs at every width ([§11](#11-dense-projections-and-the-head), 2026-11-23) |
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

### 3.4 Ideas from NInfer-all (reviewed 2026-11-21)

Measured on this machine (2026-11-21, RTX 5090, normal clocks; both engines' benchmarks through their public Engine on
one Inkling-tokenized corpus of English prose, C = 1, 3 repetitions after a warmup; NInfer-all built at 395ce8b8 from
Unsloth's UD-IQ4_XS GGUF; this design's first builds with the head packed):

| Workload | This design | NInfer-all | |
|---|---:|---:|---|
| prefill 2,048 | 752.8 tok/s | 8,542.8 | 11.3× behind (prefill still runs the decode kernels, 16 columns a call) |
| prefill 10,000 | 736.5 | 6,785.6 | 9.2× behind |
| decode 128 | 167.65 | 187.27 | 10.5 % behind; M5's 185 tok/s target missed |
| decode after 10,000 | 154.63 | 168.47 | 8.2 % behind |
| greedy 384-token essay (CLIs) | 167.0 | 185.5 plain, 216.2 with DSpark (window 3; 1.90 tokens a round) | |

Its artifact reads ~3.5 GB a token against this design's ~5.1 GB (estimates from the formats), and neither decode is
near its byte floor (~2.9 ms a token here against 5.96 ms): fixed per-step cost dominates both. Its other leads are the
routes this design has not built yet (prefill, DSpark, prefix reuse, vision, structured output).

| Idea | Source | Use here |
|---|---|---|
| At ≤ 16 rows a depth mix or residual update spreads each row over one-warp blocks; each block keeps its warp's sums and every reader adds them in the one-block kernel's warp order, so the values are the one-block kernel's bits (with batched controllers and a sorted route: decode 88.9 → 111.4 tok/s on an RTX 3090) | `kernels.cu` `mix_split_terms_kernel` / `mix_split_rows_kernel`, `residual_split_kernel` | the visit kernels run one CTA a column; the pinned order (288 groups, bfly32, a fold over 9 warps) allows the same split with one-warp blocks of 256 channels. Probed, not built (2026-11-21, `infernix_inkling2_visit_bench` at T = 1 launched one warp wide): one warp's share takes 0.60-0.86× the 288-thread CTA's time, so the split's ceiling is ~0.23 ms of a 6.3 ms round. A split mix needs every block's partial sums before its weighted sum. As two launches a mix (NInfer-all's form) it would add ~0.29 ms (an estimate: 96 launches at the ~3.1-3.7 µs a near-empty `gate_exit` takes here, not a measured empty dependent launch). One launch whose blocks of a column exchange the partials through global memory (an arrival count and a wait, as the MoE's last CTA does) is not costed; it is the form to try, worth at most the ceiling less the exchanges |
| A sliding visit's prefill attends over a staging ring: the window before the chunk copied in, the chunk's keys appended, the window that ends the chunk copied back to the sequence's ring | `executor.cpp` `visit()` (`ring_copy`, `prefill_ring`) | the M8 prefill route over the StateImage's 17-page window ring ([§9.3](#93-prefill)) |
| A named tool choice forced by the prompt: it ends with `<|message_model|>NAME<|content_invoke_tool_json|>`, the turn resumes inside the call's JSON and the decoder starts in it | `inkling_output.h` `InklingToolCallDecoder` | needs the tool choice at preparation (PromptOptions has none today); with it, a required choice of one name needs no grammar |
| The prompt opens the answer's text block for typed decisions and reranking (the token read is the answer's) | `docs/inkling2.md` | a readout recipe for the frontend, if typed decisions come to this family |
| DSpark drafts for one decoding request at a time, up to 3 drafts a round: a controller picks the width from how far the request's drafts survive and the round times it measured at each width, and a round verifies only through its first draft at probability ≤ 0.5. Reported: greedy 1.14-1.41×, temperature 1 0.99-1.24×, 1.46-2.68 tokens a round | `runtime/engine/inkling2_core.cpp` `round_drafts`, `spark_controller`; `docs/inkling2.md` | §13.4's confidence-calibrated length with the measured cost line covers both rules; its short chains and modest gains are a bound for its W4A8 arithmetic and 15-20 launches a visit, not for this design's draft cost (§13.5). The lone-request rule is the cost line's answer at C > 1 until M6 measures otherwise |
| Rollback is a frontier move over rings sized to the window plus a verify block (window + 32) | `tests/models/inkling2/test_dspark_kernels.cpp` | as [§7.2](#72-stateimage)-[§7.3](#73-speculative-rollback) (1088 = 1024 + 64, conv rings of 32) |
| Prefix reuse resumes only where a state was kept: a sequence's state when its request ends, a checkpoint of its window rings where the prompt's last user turn closes, and pinned host images at shared-prefix ends and of a sequence's state before another request takes it | `docs/inkling2.md` "Prefix reuse"; `inkling2_core.cpp` (Qwen3.5's `RewriteCheckpointKind::TurnClosure`) | the hybrid prefix cache's endpoint and output-boundary snapshots ([§7.4](#74-prefix-cache)); M4's frontend supplies the turn-closure rewrite checkpoint as Qwen3.5's does |

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
| Head (the real 200,058 rows) | 0.92 GB | 0.69 GB |
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
| Config, binding, read-only model | `src/models/inkling2/config.*`, `load.*`, `model.h`, `weights.h` | parses `text_config.loop_config`, `mtp_config`, `vision_config`; rejects configs the Ops do not cover |
| Native parameters, fixed calls | `src/models/inkling2/execution/` | built: `parameters.*` (the native operands), `forward.*` (one call: embed, the 48 visits with the pass restarts and messages, the head; the StateImage layout), `dspark.*`, `vision.*` (the tower, [§15](#15-vision)), `lossless.*` (`--lossless`: one scored window, [lossless route §3a](lossless-route.md#3a-the-route)) |
| Program | `src/models/inkling2/program/` | planning, stores, transactions, graphs, prefix binding; implements `runtime::EngineProgramContract` |
| State layout | `src/models/inkling2/execution/forward.h` (`state_layout`) | StateImage parts ([§7.2](#72-stateimage)) |
| Frontend | `src/models/inkling2/frontend/` | renderer, output session, image preprocessing (the pre-tokenizer is the shared core's `Inkling` pipeline) |
| Shared BPE core | `src/text/tokenizer.{h,cpp}` (moved from `qwen3_5/frontend/`) | the byte-level BPE tokenizer with two fixed pipelines (`TokenizerPipeline::Qwen`: NFC and Qwen's split; `Inkling`: no normalizer, the o200k-style split with case classes, `ignore_merges`), each pipeline's description in `tokenizer.json` checked against it. Two explicit consumers justify the move |
| Ops | `src/ops/w4a4_sparse_moe/`; `src/ops/inkling2/` (`visit.cu`: the mixers, message gates, short convs and the experts' rows fused at the K1/K4-K5 boundaries of [§8.2](#82-the-visit-chain); `attention.cu`: the attention prologue and decode/verify attention); contracts `include/infernix/ops/{w4a4_sparse_moe,inkling2_visit,inkling2_attention}.h` | [§9](#9-attention-ops)-[§12](#12-residual-mixing-message-gates-and-short-convolutions); the fused kernels made separate `attn_res_mix`/`short_conv` Ops unnecessary |
| Kernel cores reused | `ops/offloaded_sparse_moe/cuda/narrow_a4_mma.cuh`, `wide_expert.cuh`; `ops/linear/bf16_p12/`; `ops/common/{rowsplit_ring,mbarrier,mma}.cuh`; `ops/kernel/sampling.cuh` | the expert cores move to `ops/common/` once two Ops use them; the offloaded Op keeps its planning |
| Runtime reuse | `src/runtime/prefix_cache/`, `src/runtime/speculative/` (calibrated length), Scheduler, EngineCore | model-agnostic, unchanged |
| Converter | `tools/convert/inkling2_loop.py` + recipe `inkling2_mini` | [§18](#18-artifact-and-conversion) |
| Tests | `tests/ops/` per Op; `tests/models/inkling2/` | [§16](#16-numerics-and-qualification) |

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
| Window K/V, 40 visits | 17 pages of 64 (R = 1088 positions) × 8 heads × 128 × (K, V) | 88 MiB (INT8-G64) | 170 MiB |
| Conv histories, 48 visits | 4 rings (K 1024, V 1024, attn-out 2304, MLP-out 2304 channels) × 32 slots (3 taps + up to 29 columns) | 19.5 MiB | 19.5 MiB |
| DSpark context rings, layers 0 and 2 | 2 × 1088 × 8 × 128 × (K, V) | 8.5 MiB (BF16 always) | 8.5 MiB |
| DSpark Markov seed `f_rnn`, continuation hidden, the last column's tap features | 4-6 × 2304 | < 0.1 MiB | < 0.1 MiB |
| **Total per lane** | | **~116 MiB** | **~198 MiB** |

A window ring is whole pages, so a 64-key unit of attention is one contiguous page (position p in page (p / 64) mod 17,
at row p mod 64); 1088 = 1024 + 64 lets a call append up to 65 positions a lane before a write reaches a key the call's
first query reads. A conv ring of 32 slots (p at slot p mod 32) holds 3 taps of history plus the call's columns, up to
29 a lane. Each lane's image is contiguous (the kernels take a lane stride), so a snapshot is one copy. The window
rings take the KV profile's window codec ([§7.6](#76-kv-profiles)): BF16 K and FP16 V under `bf16`, the packed form
under `bf16p`, INT8-G64 under every other profile (the column above); a quantized ring's codes and side records sit
beside the image in their own per-lane planes, copied with it. The BF16 size is close to the 27B's StateImage
(186.8 MiB), so the prefix cache's slot and Host-tier policies apply as they are.

### 7.3 Speculative rollback

Verify writes K/V and conv inputs for all T columns into the rings and the Main pages. Commit moves the lane's
frontier to the accepted length; rejected ring slots are overwritten by the next canonical write before any read
([paged KV §9.1](paged-kv-cache.md#91-independent-pool-frontiers)). There is no recurrent fold: nothing in the model
integrates over positions outside the window.

### 7.4 Prefix cache

Built 2026-11-22 (`src/models/inkling2/program/prefix_cache.{h,cpp}`, the Program's admission, prefill and release),
on the [hybrid prefix cache](hybrid-prefix-cache-spec.md)'s index, tap planner, cost model and Host slab store, in the
form Qwen3.8-Flash-Next's binding takes:

- **Pages.** The full visits' pools are one core `DeviceKVPagePool` of page-major planes (`[pages][8][64][row]`: a
  page is one contiguous run in each plane, so Host records copy whole pages), the lanes' block-table rows a
  `KVExecutionTablePool`. With DSpark the pool has two more planes, the drafter's full layer's K and V (BF16), read
  and written through the same rows, so a cached page carries the drafter's context with the model's. A lane
  reserves its whole context at its binding and maps a page when a call (or a draft's block rows) first writes into
  it; an unmapped row names a scratch page. Every lane's context fits the pool, so a binding always gets its pages
  (evicting cached blocks, or waiting for a Host write to land): no request is paused.
- **Snapshots are Host-born.** A tap copies the lane's StateImage (window rings, conv rings) and the drafter's sliding
  rings straight to reserved Host slabs, and its partial page as the tail record; the lane's next call waits for the
  copy. No Device snapshot slots. An image copied after a call ending at `end` serves every frontier in
  `[end - 29, end]` (the conv rings' 32 slots hold the 3 taps before it; 15 with DSpark, whose Markov ring keeps 16
  rows), so a tap splits a prompt call only when it lies further back; the prompt-end tap is the image after the
  prompt's last call, labelled `n - 1`.
- **Only prompt-computed state is published**: blocks whose 64 positions were prefilled, and taps. No endpoint and no
  decode-computed block (they would make a resumed request's output depend on what the cache held), so a resumed
  request's output equals a cold run's, bit for bit, on every route: the prompt route is chunk-invariant. The
  drafter's context depends on its call widths (projections_dmma at 16 columns or fewer, projections_mma above),
  which changes drafts only.
- **Persistence** (`--prefix-cache-file`): the common file format, Inkling's magic and geometry (slab, page record,
  the image's parts); the fingerprint adds the KV format, drafter, `--q8-weights`, `--a16-activations`, `--lossless`
  and vision.
- **SSD tier** (`--ssd-context-gib`; [spec §5.7](hybrid-prefix-cache-spec.md)): the shared `SsdTier` over whole Host
  slabs. An image is many times a prompt's blocks (107-190 MiB against 1-2 MiB per 64 tokens), so a small Host tier holds few
  snapshots: a capture counts its image, tail and path blocks in the room it needs, and when that room is short lands
  the captures and block writes in flight (write-back takes a snapshot only once it and its path are on the Host),
  then starts and waits for the write-backs, so the last request's snapshot is written rather than displaced. A
  promotion lands them the same way before it asks for Host room. The source choice weighs the read: at the cost
  model's 2 GB/s (a modelled break-even, not timed) a default-route prompt under about 2,800 tokens is re-prefilled
  instead; the lossless route's model (about 0.27 s a call, 3.1 ms a token and 0.89 us a query-key pair) prefers the
  read at any length.
- **Not built yet**: next-turn prefill, tool-call holds, queue holds and prefetching for waiting requests.

Sizes: an image is 190 MiB with BF16 windows, 107 MiB with INT8 windows (every profile but `bf16` and `bf16p`) and
158 MiB with `bf16p` windows, plus 9 MiB of drafter rings; restoring a BF16 one takes ~8 ms over the PCIe link,
against ~0.25 s to recompute 4K tokens.

The alternative (TokenSpeed's): window K/V in a paged pool whose pages behind the window are freed unless a cached
path holds them, with each page carrying its convs' checkpoints, so any page boundary is a restore point and
neighbouring snapshots share window pages. It needs new pool semantics (front trimming, snapshot-held window pages)
and is adopted only if M4's agentic replay shows Host-tier bytes, not hits, limit reuse.

### 7.5 Preemption and Replay

As Qwen3.5: on the reproducible route a request reserves its whole extent and is never paused
([engine architecture §3.1](engine-architecture.md#31-pause-and-resume)); otherwise Replay rebuilds from the deepest
snapshot.

### 7.6 KV profiles

`--kv-dtype` selects the full visits' paged pools and, with them, the window codec of the 40 sliding visits' rings
(`window_codec()`, owner's mapping 2026-11-24): `bf16` keeps BF16 K / FP16 V windows, `bf16p` packs its windows as its
pools, and every other profile stores its windows as `int8` (the coarser formats' own codecs would quantize the recent
keys every query reads; `nvfp4` windows measured 3.5 times `int8`'s KL). The windows are a fixed share a lane,
independent of context: 170 MiB as BF16, 88 MiB as INT8-G64, 138 MiB as `bf16p`. The lossless route takes `bf16`
only, so its windows stay BF16. Every profile decodes into the attention kernels' existing tiles (K as BF16 words, V as FP16), so the decode route's
pinned score chain and the prompt route's MMAs are unchanged; the append (the prologue, or the fused prologue inside
the attention launch) encodes each new row with a warp and returns its represented words. The contract and byte layouts
are `inkling2_attention.h`'s `KvCodec`; the codecs are `src/ops/inkling2/kv_codec.cuh`, each the shared codec's
arithmetic at a 128-value row:

| Profile | Row (codes + side) | Encoding | Status |
|---|---|---|---|
| `bf16` (default) | 256 B (BF16 K, FP16 V words) | as stored | built |
| `bf16p` | 192 + 16 B | lossless 12-bit packing of the BF16 values (V's BF16 source), `bf16p_codec.cuh`'s form with a 16-byte record of up to 14 escapes (the D256 form's 6 failed real prompts: position 0's key row on KV head 4 spans 18 binades and needs 7 escapes, the only row of a 2,300- and a 10,000-token prompt needing more than 3); a row with more fails its request (`UnpackableKv`) | built: the output is `bf16`'s |
| `int8` | 128 + 4 B | two 64-value groups, an FP16 scale each (`int8_g64_codec.cuh`'s) | built |
| `fp8` | 128 + 2 B | E4M3 codes, one FP16 scale for the row of 128 (`fp8_e4m3_row_codec.cuh`'s rule at 128) | built |
| `nvfp4` | 64 + 8 B | E2M1 codes, an E4M3 scale per 16 by the five-target least-error rule (`nvfp4_group16_codec.cuh`'s) | built |
| `k8v4` | K 128 + 2, V 64 + 8 B | `fp8` keys, `nvfp4` values | built |
| `vq2` | 32 + 2 B | the rotated row y = 2^-3 H128 x (H4 over a lane's 4 values, then H32 over the lanes), 16 words of 8 as 16-bit codes of `vq2_codec.cuh`'s 512 patterns (σ over 128, the first nearest pattern: each lane of a pair searches 256, ties to the lower), S = fp16_rn(Σy² / Σyc) | built |
| `k4v2` | K 64 + 2, V 32 + 2 B | `q4_lloyd_codec.cuh`'s 4-bit Lloyd-Max levels of the rotated key row, `vq2` values | built |

The represented values every route reads are K = bf16_rn(code value × scale) and V = fp16_rn(code value × scale), each
product exact in FP32; every division is IEEE, so the op test's host encoders reproduce the stored bytes. The rotated
profiles' attention runs in the rotated domain instead of rotating every row back: the query rows are q' =
bf16_rn(2^-4 H128 q) (so q'·y = q·k, H128 H128 = 128 I), the tiles hold the rotated S c, and the output o = 2^-4 H128 o'
is applied once a row (the decode combine's warp, the prompt epilogue's quad shuffles) before the division by the
softmax sum: no row is rotated back on a read, and the fused prologue stays. Their codebooks are
model-free (VQ2's 512 patterns trained on i.i.d. Gaussian 8-value words, `tools/kv_codebook/make_vq2_codebook.py`;
Q4's unit-Gaussian Lloyd-Max levels), and they carry no exact recent-key window: the 40 sliding visits keep the last
1,024 positions exact. K is
L2-clipped (‖k‖ ≤ 22.6) and V is RMS-normalized per head, so both are bounded: favourable for 8-bit rows. Every profile
is deterministic and request-invariant (`--reproducible` accepts it); the lossy ones change the output and are
qualified like any profile ([model quality §6](model-quality.md#6-numerical-acceptance-policy-and-fast-paths)).

Measured quality (2026-11-22, RTX 5090, inkling2-mini, `infernix-perplexity` on the perplexity corpus's quick set,
291,842 tokens, context 4,096 / stride 2,048, each profile's KL from `bf16`'s own top tokens of the same build):

| Profile | PPL | KL from `bf16` | top-1 agreement |
|---|---:|---:|---:|
| `bf16` | 4.03144 | (reference) | |
| `bf16p` | 4.03144 | 0 (−5.8 × 10⁻¹⁸) | 100.00 % |
| `int8` | 4.03172 | 9.41 × 10⁻³ | 96.31 % |
| `fp8` | 4.03503 | 9.62 × 10⁻³ | 96.25 % |
| `k8v4` | 4.03418 | 9.88 × 10⁻³ | 96.14 % |
| `nvfp4` | 4.03641 | 1.069 × 10⁻² | 95.98 % |
| `k4v2` | 4.04620 | 1.403 × 10⁻² | 95.47 % |
| `vq2` | 4.09501 | 2.578 × 10⁻² | 93.74 % |

`bf16p` reading exactly 0 shows the statistic and the reference are sound. The 8-bit profiles and `nvfp4` sit
within 9.4–10.7 × 10⁻³ of each other, about the level `--q8-weights` reads as well: that is this model's response to
any small perturbation on this corpus (a floor), not a ranking among them. `k4v2` and `vq2` are clearly above it.

The table above has BF16 windows under every profile. The windows' own effect (2026-11-24, the same quick set, BF16
references per route; the real INT8 ring gave the emulated ring's total NLL exactly), as each profile now runs it:

| Profile | PPL, BF16 windows → its windows | KL from the same pools with BF16 windows |
|---|---:|---:|
| `bf16p` (`bf16p` windows) | 4.03144 → 4.03144 (= `bf16`, no request failed) | 0 |
| `int8` | 4.0317 → 4.0322 | 1.27 × 10⁻³ |
| `fp8` | 4.0350 → 4.0333 | 1.19 × 10⁻³ |
| `k8v4` | 4.0342 → 4.0349 | 1.11 × 10⁻³ |
| `nvfp4` | 4.0364 → 4.0340 | 0.90 × 10⁻³ |
| `k4v2` | 4.0462 → 4.0486 | 0.77 × 10⁻³ |
| `vq2` | 4.0950 → 4.0934 | 0.75 × 10⁻³ |

INT8 windows add about a tenth of the 8-bit pools' own KL, and top-1 agreement drops 0.04–0.27 points; the PPL moves
both ways by at most 0.0024. Rejected window codecs: `fp8` windows (+1.34 × 10⁻³ under `fp8`), `nvfp4` windows
(+4.42 × 10⁻³, 3.5 times INT8's). Speed (2026-11-25, RTX 5090, INT8 against BF16 windows over the same pools in one
binary, 2 runs an arm): plain decode +0.1 to +0.7 % after 2,048 and 10,000 prompt tokens on `int8`, `fp8` and `nvfp4`
pools, 4 lanes −0.3 to +0.7 %, prefill −0.1 to −0.5 % (the new rows' encode). DSpark speeds move with the text the
windows' rounding leads to (−4.1 to +12 % over 3 prompts × 3 pools, acceptance −0.025 to +0.038); a round's verify
time, fitted post hoc against the tokens a round over those 9 cells, is about +0.2 % at an equal count. Against `bf16`
KV, `int8` (windows included) decodes +0.4 % after 2,048 prompt tokens and −0.2 % after 10,000 and prefills 0.6 % and
1.5 % slower.

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
| K1 attention mixer | previous visit's per-entry expert outputs (8 × H BF16 per column), conv ring, block sources | expert combine in slot order → BF16; post-MLP conv + residual; block bookkeeping; attention mix (1 or 3 queries); norm(s); DSpark tap copy on visits 5, 23, 41 | x_q (and x_k, x_v), BF16 |
| K2 projection | x_q, x_k, x_v; packed W (8,576 rows; the sink's 24 and the full visits' 64 dynamic-tap rows in a small parent) | multi-input GEMV: row ranges take their own input vector; FP32 accumulate; epilogue writes each output in its consumer's layout (L6) | q, k, v, r, gate, sink, taps |
| K3 attention | K2 outputs, ring or pages | per-KV-head prologue: K/V conv (ring history), clips, v-norm, τ, append; relative-bias vector; split-K over keys; the last split CTA combines with the sink and multiplies σ(gate) | a, BF16 [3072] |
| K4 output projection | a, packed W_o | GEMV, epilogue: post-attention conv (per channel, ring), residual add or restart | h (BF16), conv ring |
| K5 MoE mixer | h, block sources | MLP mix, norm, four-over-six (or MXFP8) quantization of x | A4(x), BF16 x for the router |
| K6 router and route | x, W_router | FP32 router GEMV (few-row mapping, weights before the PDL wait); a one-CTA tail: sigmoid, bias, rank-based top-8, γ, per-expert job lists | dispatch |
| K7 gate/up | A4(x), dispatch, expert weights | narrow W4A4 (or W8A8); epilogue: softcapped SwiGLU × γ → BF16 h; per-(column, expert) row amax by `atomicMax` (order-independent, so deterministic) | h_e, amax_e |
| K8 down | h_e, amax_e | polls its job's completion count (L11); A4(h) (where: M5); W4A4 down | per-entry expert outputs, BF16 |

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
- **Launches.** A prologue kernel per (column, KV head) (the K/V convs over their rings, clips, v-norm, τ, the append,
  the final q and r), then the attention kernel per (split, KV head, lane). Built 2026-11-21:
  `src/ops/inkling2/attention.cu`, qualified by `tests/ops/test_inkling2_attention.cu`. A decode-route call of one
  lane with at most 4 columns (no padding, every lane one group) runs the prologue inside the attention kernel instead
  (`AttentionArgs::fused`: each CTA forms its rows' q and r and patches the call's new keys and values into its tile;
  the same bits) and issues its first unit's pages before the dependency wait, since no kernel of the round writes a
  visit's past keys and values. Against the prologue kernel (decode round at 4K, 2026-11-21): −0.44 % at one column,
  −0.17 % at two, −0.09 % at four, +0.25 % at eight; infernix_bench C = 4 and 8 ran 0.8 % and 1.3 % faster with the
  separate prologue, hence the one-lane rule. The early pages: −0.43 % at one column at context 128 (2026-11-22).
- **Shape (L15).** 512-thread CTAs; a unit's K and V pages copied to shared memory by cp.async (one stage; a 2-stage
  ring for multi-unit full splits is an M5 item); scores a thread per (row group, key) as 4 fold chains, the unit
  softmax a warp per row, P×V a thread per (row group, 4 dims); accumulators in registers.
- **Packing.** Per KV head, the 3 query heads × T columns form the rows (3T ≤ 48), so K/V tiles are read once per KV head.
- **Splits.** Units are fixed absolute 64-key pages, so a query's arithmetic depends only on its own position and lane
  (sliding: one split per unit of its window, ≤ 17; full: 32 splits of ⌈units / 32⌉ units of the row's own key count,
  folded in order inside the split), never on B or T (L16). Grids are (18 or 32) × 8 × lanes; the CTA completing a
  (lane, KV head) counter combines each row's splits in split order and resets the counter.
- **Route per width.** The reproducible route runs one column-invariant schedule at every T ≤ 16 (each column's score,
  softmax and P×V in a fixed order whatever the other columns). A tensor-core verify route (from ~8 query rows per KV
  head, i.e. T ≥ 3) is a fast path, registered where faster (L16).
- **Bias.** Each score thread computes its (row, key) bias `bf16(r · P[:, d])`, a 16-deep fma fold, reading P through
  the read-only cache (staging P's slice beside the tile is an M5 item). Distances ≥ 1024 add nothing.
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

BF16 rings; the full pools in every profile of [§7.6](#76-kv-profiles), decoded into the decode and prompt kernels'
tiles by `decode_unit` (`attention_common.cuh`) and encoded by the append. The lossless route
(`execution/lossless.cpp`) reads a scored window's own keys, or in generation the keys before the call from these rings
and pages (holding BF16 V on that route) followed by the call's own: exact scores (lossless_linear), the eager form's
BF16 roundings with the bias, mask and sink and a correctly rounded softmax (`lossless_inkling2_attention_probs`), BF16
V ([lossless route §3a](lossless-route.md#3a-the-route)).

---

## 10. MoE Ops

### 10.1 Contract

`w4a4_sparse_moe`: inputs x [H × columns] BF16 (or its quantized form), the dispatch, the experts of one layer (NVFP4 or
MXFP8 family), the softcaps; outputs one BF16 row per (column, slot) entry, bf16 of the expert output as the trained kernels write it
(decode: combined by the next K1, an FP32 fold over the slots in slot order, one BF16 rounding) or the BF16 combined
output (prefill). Routes: narrow (decode, verify) and wide (prefill), each qualified against the FP64
oracle of [§16.3](#163-op-oracles).

### 10.2 MoE mixer, router and dispatch

K5 produces the normalized x and its quantized form (codes, block scales, outer scale) once per column (L10). K6 runs
the 128-row FP32 router with the few-row mapping and its weights fetched before the PDL wait, then a one-CTA tail
(≤ 32 columns, L14): sigmoid, bias, top-8 by rank counts (each candidate counts the scores above it; ties by index:
exact, no serial arg-max), γ, and per-expert job lists. Padded rows of a graph batch route no expert (L12).

### 10.3 Narrow routes (decode, verify)

- **NVFP4 W4A4.** Since 2026-11-22 the native block-scaled core (`ops/common/a4_native.cuh`): one `mxf4nvf4`
  m16n8k64 MMA a 64-element K step of a 16-row group (the weights as A, E4M3 block scales applied by the MMA, K slot
  blocks 0, 2, 1, 3; x and h staged once in the core's slot order), an FP32 chain a gate/up K half added half 0 +
  half 1, one chain over down's K, then the per-column outer scale and per-expert weight factor in the pinned order
  ([§2.5](#25-rounding-points-pinned-at-m0)). The same chain runs in `single_kernel`, `gate_up_a4` and `down_a4`, the
  only NVFP4 kernels a decode or verify call reaches, so decode = verify and a lane's bits never depend on running
  alone or beside others. It replaced the exact int64 core (`narrow_a4_mma`'s s8 `mma.m16n8k16` on doubled codes,
  L9), which stays as `Nvfp4Core::Exact` (op test and bench; no CLI): an FP32 chain is no longer the exact sum rounded
  once, so the default and `--reproducible` routes' bits may change (deterministic and request-invariant as before);
  on the op test's 589,824 outputs and the forward test's 32-position free-running logits they did not change at all.
  Op test: operand mapping exact on 524,288 one-hot cases, every finite scale-word pair decoding as
  `canon::e4m3_value` (only 0x7F / 0xFF, E4M3FN's NaN, decode differently: the activation quantizers never write
  them, and none of the checkpoint's 1.40 × 10^9 expert scale words is one or has its sign bit set), the FP64
  criteria of the MXFP8 chains (h within 2^-6
  of max|h|, outputs within half a BF16 ulp + 2^-16 max|y|: worst 0.993). Measured (RTX 5090, forward test round at
  context 128, three runs each): 5.283 → 5.078 ms at one column (−3.9 %), 7.448 → 7.236 at three, 10.299 → 9.758 at
  eight; op bench T = 1 cold 34.65 → 30.35 µs (the emptied-unit ceiling 29.05), warm 35.5 → 21.2 µs.
  kMaxColumns ≥ B_max × T_max = 128, so every decode and verify call stays narrow and its bits never depend on other
  requests. A job of 9-16 columns runs two passes over
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
- **One column in one launch (`moe_single`, 2026-11-21).** A decode call of one column runs the router and both expert
  passes as one persistent grid of 144 CTAs (one an SM, the residency checked once by the occupancy API; otherwise the
  three launches): the router's rows on CTAs 0-15 in route_kernel's order, every CTA selecting the column itself, a
  job's 144 gate/up and down row groups on 18 CTAs of 8 warps, each warp streaming its gate/up row group over all of K
  and then its down row group through one 9-stage cp.async ring, so the down weights stream while the CTA waits for
  its job's h. NVFP4: each CTA loads the job's BF16 h row once (its amax from the same loads) and forms the
  four-over-six row itself; MXFP8: each CTA quantizes its 64 intermediates (two whole groups) and publishes them, and
  the down CTAs load the row (2026-11-22, the same bits, −0.031 ms a round at context 128). Bits equal the three launches' (op test: routing and
  outputs byte for byte, graphs interleaving both paths; the forward test's decode guard). Measured (RTX 5090): the op
  bench at T = 1 (8 cold layers a graph) NVFP4 44.4 → 35.2 µs, MXFP8 58.5 → 49.6 µs (808 → 1,019 and 1,123 → 1,327
  GB/s); the decode round −3.0 % at 4K context (6.107 → 5.921 ms), −3.4 % at 128 (5.818 → 5.622 ms), about −0.19 ms a
  round against the op bench's −0.44 ms.
- **2-8 columns in one launch: tried, rejected (2026-11-22).** Verify and multi-request calls read far more experts
  than one column (distinct experts a visit on the golden prompt: 8.0 at one column, 12.2 at two, 15.8 at three, 18.9
  at four, 29.2 at eight; their bytes at 1.5-1.7 TB/s floor a verify round's MoE at 2.1-2.4 ms at three columns
  against the three launches' 3.76 ms). A persistent grid (one CTA an SM, the router's rows on CTAs 0-15, every CTA
  grouping the columns' entries itself, quads of 4 warps taking 16-row groups with K split four ways in exact int64
  quarters, down workers forming each job's four-over-six h rows) gave the three launches' bits but lost at every
  width, also after a latency pass (one fence a warp, h loaded up front, the scales in shared memory): verify round at
  context 128, three runs each, 6.523 → 6.838 ms at 2 columns (+4.8 %), 7.454 → 8.207 at 3 (+10.1 %), 8.053 → 9.006
  at 4 (+11.8 %), 10.304 → 11.996 at 8 (+16.4 %); the MoE alone at 3 columns 3.78 → 4.59 ms. 2-8 columns keep the
  three launches; their gap to the byte floor (3.78 against 2.1-2.4 ms at three columns) stays open for another
  design.

### 10.4 Wide routes (prefill)

Grouped GEMMs over the chunk's (token, expert) rows on Flash-Next's pipeline (L13): one producer and eight consumer
warps, `mxf4nvf4` m16n8k64 (`mxf8f6f4` for MXFP8), 64 × 128 × 128 tiles, three ~43 KB stages, 2 CTAs per SM. Inkling
differs in one place: A4(h) needs the row amax over all 18 N-tiles of a (token, expert) row, so the gate/up epilogue
writes BF16 h and an `atomicMax` amax per row, and the down GEMM's producer warps quantize h tiles as they load them,
after the row's job counter completes. The quantize-permute of x (four-over-six per token, then scatter to expert-sorted
rows, zeroing each expert's scale tail in the CTA writing its last row as SGLang does) is one kernel. SGLang's tile
choices measured on an RTX 5090 for these shapes seed the tile search. The combine is FP32 in slot order, rounded once.

### 10.5 Expert prefetch (decode and verification)

A decode visit's experts are cold (47 other visits stream ~4 GB between two calls of a layer), and the bus sits idle
through attention, W_o and the glue before them. The Forward fills that time from two side streams that join the call's
stream once, at the end of `finish()`, so nothing on the visit's chain waits for them (Basalt's arrangement; the
2026-11-22 in-chain prefetch from mlp_entry's CTA held ~25 us on the critical path and was rejected):

- a call of 1-2 columns: after each visit's MLP input is formed (kept a visit each, `ForwardBuffers::guess_x`), the
  NEXT visit's router runs on it on the guess stream (`moe_route` over a 2-column workspace); at one column its 4
  likeliest experts' whole records (gate/up and down) are the guess, at two each column's 4 likeliest experts' gate/up
  bytes (16 guesses at 2 columns cost more bus than they saved);
- a call of 3-8 columns of one lane (verification): the router records each visit's last column's experts
  (`moe_route`'s `record`, relaxed stores, in descending key) for the next such call, which prefetches the 4 likeliest
  of them; a decode round of 3-8 lanes prefetches nothing
  (its last lane's experts for every lane: 8 lanes 649.6 → 646.4 tok/s in all, 4 lanes 463.1 → 463.6);
- after each visit's K2 the prefetch stream issues `moe_prefetch_l2` (12 CTAs of `cp.async.bulk.prefetch.L2`, 16 KB
  pieces) over the guess's bytes or the record's gate/up bytes.

The guess's size was measured (2026-11-23, context 512 rounds, median of 3, after the visit kernels' 11-23 latency
work): at one column, against 8 gate/up guesses (4.2838 / 4.3428 / 4.9523 ms on `--q8-weights` / `--a8-activations` /
default), 6 gate/up 4.2327 / 4.2287 / 4.8919, 5 4.2426 / 4.2004 / 4.8914, 4 4.2779 / 4.2271 / 4.9237, 8 whole records
4.1985 / 4.3889 / 4.9039, 6 whole 4.2280 / 4.3037 / 4.9071, 4 whole 4.1864 / 4.1931 / 4.8512 (spreads <= 0.011):
4 whole records, tg128 230.0 → 235.7 and 225.2 → 234.2 tok/s. At two columns (against 8 gate/up guesses) 6 gate/up
guesses were +0.07 % on `--q8-weights` and -1.24 % on `--a8-activations`, 8 whole records +1.32 / +2.83 %, 6 whole
+1.14 / +1.16 %, so two columns keep 8 gate/up guesses. Why fewer, whole guesses win at one column is not established
(a guess's bus time competes with W_o and attention beside it; a kept guess also saves its down pass).

The read-ahead's size was measured on Basalt's DSpark fixtures (conv / struct / prose, auto draft length, -r 2; the
texts are the same whatever is prefetched), against the record's 8 gate/up parts: its first 4 gate/up parts were faster
in two slots on `--a8-activations` (struct 667.7 → 669.6 and 667.6 → 670.5 / 667.2 → 669.3 tok/s, prose 231.9 → 234.4
and 231.9 → 234.3 / 232.1 → 234.9, beyond ±0.6-1.5; conv within its ±10) and about even on `--q8-weights` (prose 240.5 →
241.6 and 240.4 → 241.6 beyond ±1.1, 241.0 → 241.0; struct 620.8 → 620.0 and 620.7 → 620.9 within ±1.3, 620.4 → 619.6
beyond ±0.7; conv within its ±17); 4 whole records and 8 whole records were not faster (8 whole the slowest).

No output depends on it (only L2's contents; emitted token ids checked equal with and without it, DSpark and plain,
2026-11-23). Measured (RTX 5090, 2026-11-23, against the same build without it; ranges over two or three slots where
both ran). Context 512 decode rounds (median of 3): default 5.276-5.277 → 5.128-5.134 ms at one column, 6.479-6.481
→ 6.183-6.316 at two; `--q8-weights` 4.630-4.631 → 4.467-4.523, 5.824-5.826 → 5.511-5.609, three columns 6.599 →
6.233, eight 9.403 → 9.160; `--a8-activations` 4.610-4.612 → 4.558-4.560, two columns 5.962 → 5.789, three 6.776 →
6.593, eight 9.635 → 9.467 (three and eight columns replay one call: the record is that call's own, an optimistic
hit rate). infernix_bench tg128 / pp10000+tg128 (tok/s): default 193.0 / 179.4-179.9 → 196.4-198.0 / 182.9-183.9;
`--q8-weights` 214.2-214.3 / 196.9-197.0 → 219.3-220.8 / 198.5-201.6; `--a8-activations` 214.5-214.7 / 197.2-197.8 →
215.3-216.5 / 196.0-198.7. DSpark on Basalt's fixtures (conv / struct / prose): `--q8-weights` (-r 2) 267.5 / 549.2 /
213.6 → 272.9 / 557.1 / 218.9; `--a8-activations` (-r 4) 274.5 / 606.6 / 211.7 → 276.1 / 610.3 / 212.7 (conv within
its ±15, prose within its ±0.7). Prefetched experts that miss cost bus time: a decoy (the same bytes of other
experts) ran 0.21-0.33 ms a round slower than none, and sixteen guesses at two columns lost (a8: 6.04 → 6.27 ms).

---

## 11. Dense projections and the head

- **Packing (L4).** A matrix loads as `bf16_p12` only when every reader takes the packed form and M1's per-shape sweep
  shows the packed routes faster at the decode and verify widths: expected for the K2 parents [8,576 × 2304]
  (Flash-Next's similar shapes: 1.26× at T = 1-4, 1.13-1.19× at 5-16) and the head; W_o [2304 × 3072] only if a direct
  MMA wins (Flash-Next's output projections needed one to reach 1.00-1.06×); routers and gates stay BF16. The embedding is
  packed for VRAM (its reader is a row gather). Packing runs on the GPU after upload (~30 ms on Flash-Next). The skinny
  packed GEMV gives the BF16 skinny route's bits at T ≤ 8 and runs 16 columns a pass under `A16ColumnInvariant`, so packing
  changes no reproducible-route bit. There is no second BF16 copy: prefill and wide verify (B × T of 33-1,024 columns) pay
  the packed routes' 0.67-0.96× on those projections, a few percent of prefill time.
  Measured (2026-11-21, inkling2-mini, 4K context, C = 1, each group packed alone against a 6.305 ms round): the head
  saves 0.129 ms (its sweep predicted 0.125), but the K2 parents cost 0.070 ms more, W_o 0.044 ms and the message
  gates 0.269 ms, although the op sweep (each launch on its own cold copy) had K2 at 0.80× and W_o at 0.71× BF16's time
  at T = 1. Only the head packs (the embedding stays BF16: its gather does not take the packed form yet); packed, it
  also saves 1.8 % of the round at 2 columns and 0.7 % at 8. K2's loss is residency: the packed projections kernel
  takes 153 registers a thread at one column (BF16's 72), so one 8-warp CTA fits an SM instead of three, and (computed
  from those counts) ~14 % of the matrix loads before the PDL wait instead of ~42 %. With 4-warp CTAs (three an SM; same bits) packed K2 saves 1.1 % at one column but
  costs 1.6 % at 2 and 4.4 % at 8, the widths of decode at C ≥ 2 and of DSpark's verify, so K2 stays BF16. Why W_o
  loses is not established.
- **8-bit weights (`--q8-weights`, approximate, opt-in).** At load (`models/q8_weights.h`) the K2 parents, the
  sink/taps parents and the head become Q8_G32_FP16 RowSplit (`ops/q8_g32.h`: int8 codes, an FP16 scale fl16(max / 127)
  per 32 values) and their BF16 copies are freed; the head is then not packed. The fused projections read Q8 storage
  (a weight is code × scale, exact in FP32; the fold is the BF16 route's, so every column's bits stay independent of
  the call's width and other columns), the head runs on them too (`ops::linear` has no Q8 route at its shape). Chosen
  per group in the captured round (2026-11-21, 4K context, against the shipped route, each group alone): K2 -7.6 % at
  one column, -5.6 % at two, -1.3 % at eight; the head (against its packed form) -1.6 %, -1.4 %, -0.2 %; W_o -2.5 %,
  -1.4 %, +2.7 % and the gates' downs -0.1 %, +0.1 %, +0.1 % stay BF16. W_o is an open candidate: its Q8 arm also
  changed its kernel from `ops::linear`'s to the fused projections', so its width-8 loss is not storage alone, and it
  was not re-timed after the wide kernel's spill fix. (A 2,048-column prefill call with every group in Q8 took +0.5 %,
  W_o's time +6.7 %, while the wide Q8 kernel spilled at W_o's and the gates' K: the compiler hoisted every chunk's
  decode out of the pass loop; an empty `asm volatile` on the held codes keeps it in the loop, 0 spills.) Its oracle is the Op test's FP64 sum of the stored codes and scales; model-level closeness to
  BF16 is reported, not judged (the quality gate is Quality Gates'). Shipped set measured (2026-11-21) against the
  default route: in the captured round (4K context) -9.3 % at one column, -7.1 % at two, -1.4 % at eight; through
  infernix_bench (Inkling corpus; two interleaved invocations at C = 1, one at C = 4 and 8) tg128 167.1 → 185.2 tok/s at
  C = 1, 399.8 → 419.4 at C = 4, 531.4 → 549.5 at C = 8; pp2048 1,504.5 → 1,512.3, pp10000 1,457.0 → 1,463.3; load
  +0.03 s (the conversion 12-16 ms); device weights 17,792.6 → 17,173.6 MiB, 620 MiB more VRAM free after load.
  Re-measured at 4dc97419 (the default's decode projections on decode-width tensor-core MMA, the prompt route's on
  tensor cores, the single-column MoE in one launch; the Q8 matrices on the prompt route's Q8 MMA, still on the fold at
  decode widths): tg128 176.6 → 199.4 tok/s at C = 1, 423.2 → 438.6 at C = 4, 601.0 → 586.3 at C = 8 (a loss);
  pp2048 16,362 → 16,010, pp10000 16,276 → 15,928. Since 6a3f2657 the Q8 matrices decode on `projections_dmma` too
  ([§14.1](#141-chunk-width-invariance)): C = 8 586.3 → 613.3 tok/s (above the default's 595.9), C = 1 tg128 199.4 →
  198.0 while the default gained 176.6 → 184.0. On the perplexity corpus's quick
  set (291,842 tokens, context 4,096 / stride 2,048): perplexity 4.0348 → 4.0328, KL(BF16 ‖ Q8) mean 0.0107 (p99
  0.108), top-1 agreement 96.0 % (code 98.2 %, the prose and Chinese streams 95.1-95.6 %). Its FP16 group scales on the
  real matrices: none rounds to zero, 47 of 29.3 M are subnormal.
- **K2.** A multi-input packed GEMV: row ranges bound to one of three input vectors (one pointer on the 20 pass-1 sliding
  visits, three elsewhere); new tuned routes for Inkling's shapes (`bf16_p12_linear_tuned` registers only measured shapes).
- **K4.** GEMV 2304 × 3072 (packed or BF16 per M1) with the post-attention conv and residual in the epilogue (each output
  channel needs only its own history).
- **Few-row matrices (L3).** Router (128), sink (24) and taps (64) in the small parent beside K2 (8,600 rows would not
  be a multiple of 16), relative (384, inside K2), gate-down (64 per gate): one row per warp, every load before the FMAs.
- **Head.** Packed GEMV over the 200,058 real rows (the last 16-row group predicated: `projection_fp32` takes only
  multiples of 16, so the head kernel handles the tail rather than padding with zero rows, which would score 0 and need
  a mask), softcap 30 in the epilogue, FP32 logits; the checkpoint's padded rows (non-zero, unreachable) are never
  stored; one read for all T columns; evict-first loads (read once per step).
- **`--a8-activations` (approximate, opt-in; 2026-11-23).** 8-bit activations in every call, as Basalt runs this model
  (owner, 2026-11-23: "8-bit activations everywhere (including in place of the existing NVFP4 activations) ... I sort
  of want it to run like Basalt's implementation"). One codec throughout, the engine's other `--a8-activations` routes'
  (`ops/ggml/ggml_a8.h`, llama.cpp's Q8_1 with an FP32 scale): each 32 values of a column as int8 codes
  `rne(v * fl32(127 / m))` with the scale `d = fl32(m / 127)`, `m` the slice's largest magnitude.
  - The dense matrices of `--q8-weights` (K2's fused input projections, W_o, the head: Q8_G32_FP16 at load, the flag
    implies it) take 8-bit inputs at every width, prompt and decode: `projections_mma` with `ProjectionArgs::x8`
    (`projections_a8_kernel`, the GGUF wide 8-bit route's arithmetic: per 32-slice in ascending order from +0,
    `acc = fma(fl32(P_s w_s), d_s, acc)`, `P_s` the exact int32 dot on int8 tensor cores, started at 1.5 * 2^23 so
    `fl32(P_s w_s)` is one FFMA); one thread sums an output, so a column's bits depend on its row and its input alone,
    the same at every width and tile shape. The call encodes each distinct input once (`a8_encode_columns`; K2's
    aliased inputs once). Tile: 128 x 128, eight warps of 64 x 32, two stages of 128 elements of K whose rows are one
    line each (ncu, K2 at 4,096 columns: 64-element stages split slice-major issued ~1 sector an L2 request and held
    L2's tags at 80 % with its data path at 21 %; 128-element contiguous stages: 22.6M requests instead of 55.3M, L2
    43 %, the int8 pipe 41 %, issue 47 % with two warps a scheduler).
  - The NVFP4 layers' routed experts: W4A8 instead of the checkpoint's four-over-six W4A4 (`ExpertActivation::A8`,
    the `w4a4_sparse_moe.h` contract): per 16-block in order `acc = fma(P_b S_b, d_s, acc)`, then
    `bf16(fl32(fl32(acc 2^-10) m))`, `m = amax_w / denominator`; the narrow kernels (decode, verification) and the
    prompt route's grouped GEMM (`wide_expert_a8.cuh` with this model's epilogues) give an entry the same bits.
    Residency (ncu 2026-11-23): the narrow gate/up reads the columns' A8 x rows through L1 rather than staging them
    (21 KB a CTA, 16 warps an SM, as the native core's), the narrow down runs 8 row groups a CTA over one h staging (8
    warps an SM; 4 before); experts op bench, cold, bytes a second: 1,248 -> 1,332 GB/s at 3 columns, 1,349 -> 1,426
    at 8 (the checkpoint's native core 1,514 / 1,612: A8's two FFMAs a 16-block per output keep it ~12 % below). The
    prompt GEMM runs `ScheduleA8Tall` (256 rows, 3 stages, one CTA of 16 consumer warps an SM; Flash-Next keeps 128
    rows, 2 stages, 2 CTAs): 4,096 columns 4,040 -> 3,693 us (141 TOPS; the FP4 GEMM 1,789 us, 292 TOPS).
  - Unchanged, as Basalt: the MXFP8 experts (W8A8 already; Basalt requantizes them to Q8_0, a lossy weight change not
    taken here), the message gates and the router (16-bit activations; Basalt FP32), the attention arithmetic.
  - The 8-bit inputs are written by the kernels that produce the BF16 ones, so a call launches no encoder for them:
    K2's by visit_entry (EntryArgs::x8) or, in a message visit, gate_exit (GateExitArgs::x8); the experts' x by
    mlp_entry / gate_exit (ExpertRows::a8); W_o's in a decode call by attention()'s combine (AttentionArgs::out8; a
    prompt call's W_o encodes its input in the projection call). Every producer applies the same codec to the same BF16
    values (the op tests compare their rows with the codec's host definition byte for byte), so where an input is
    encoded changes no bit. A call that still encodes (the head, a prompt call's W_o) does its distinct inputs in one
    launch. A decode call of one column runs the NVFP4 experts in one launch (moe_single's A8 instance: the separate
    kernels' bits).
  Quality (2026-11-23, perplexity quick set, 291,842 tokens, bf16 KV, KL against the default route's logits): PPL
  4.02607 against the default's 4.03144 and `--q8-weights`' 4.03419 (wikitext 5.9536 / 5.9671, pg19 6.2370 / 6.2518,
  zhwiki 5.2716 / 5.2772 lower; ninfer 1.48938 / 1.48914 higher); KL 8.43e-3, top-1 96.5 % (`--q8-weights` alone
  1.06e-2, 96.0 %). The fake-quantized estimate of the dense projections alone (8-bit inputs on the BF16-weight kernels,
  the experts W4A4) read KL 1.07e-2 and PPL 4.0318, per-token scales 1.14e-2 (rejected). A reading, not isolated by
  any arm: the 8-bit expert activations, not the dense route, bring the KL below this model's usual ~1e-2 floor (the
  checkpoint's 4-bit activations may amplify any upstream difference).

### 11.1 LoRA adapters (`--lora`)

User-facing contract in [LoRA adapters](../lora.md#inkling-2). Binding: `models/inkling2/lora.h` (Model data, read at
load planning); operands: `execution/lora_parameters.h` (BF16 factors, or the lossless route's exact form); execution:
`execution/lora_forward.h` and `ops::inkling2::lora_term_add`.

- **Where the term lands.** After each projection's own rounding, as PEFT adds it. The alternative, fusing it into the
  epilogues of `projections`, `projections_mma`, `projections_dmma` and the a8 kernels, would save a launch per group but
  needs an adapter instance of each and changes the order PEFT computes; not attempted, and the cost of the separate
  launch is unmeasured.
- **A launch per group.** K2's outputs are one group (one `lora_term_add`), W_o, a pass's gate downs and their forget
  and message projections others, each at its own padded rank with a compile-time k-step count; `B` is loaded before
  the PDL wait. No group's work runs for a layer the adapter does not touch.
- **The shrink without a launch, where the bits allow it.** `U = A x` joins the base launch as extra BF16 segments
  (`LoraForward::fold_*`) for W_o and the gate downs on either route, and for K2 in prompt calls when the segments fit
  (8 at most; K2's A rows are one stack per input, one segment when the three inputs alias). Q8 storage cannot share a
  launch with BF16. **K2's decode shrink stays a launch of its own**: `dmma_splits` sets the runs of K from the launch's
  total row tiles, 4 for K2's 538 and 8 for A's alone, so inside K2's launch `U` would take another FP32 association.
  The gates' forget and message projections are FP32 launches and `U` is BF16.
- **Routes.** The shrink takes the call's projection route (tensor cores for a prompt call, the decode-width MMA
  otherwise) and reads the BF16 inputs on every route, so `--q8-weights` and `--a8-activations` change only the base
  projection. On `--lossless` `LosslessForward` applies `lossless_lora_shrink` and `lossless_lora_add` to the same
  outputs in the same places.
- **The experts are out of scope**: they and the router are `nn.Parameter`s of the mixture of experts, so a plain PEFT
  adapter holds no factors for them.

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
the chunk with the ring as left history and writes the ring's positions back (the decode and verify kernels take at
most 29 columns a lane, so no write of a call reaches a slot its reads need).

**Directional ablation (`--ablate`, [Ablation sidecars](../ablate.md)).** The sidecar's layers are the 24 physical layers,
cut on every pass. The attn lane is the post-attention conv's BF16 output inside `mlp_entry` (at a boundary it is `cand`
itself); the mlp lane is the post-MLP conv's output inside the next visit's entry tail, or `final_mix` for visit 47,
so its alpha is the previous visit's layer's. Each is a template instance (`EntryCutArgs`, `MlpEntryCutArgs`): the plain
instances' argument structs and code are unchanged. The cut value goes unrounded into the residual sum, which keeps its
one BF16 rounding; the rings keep the uncut inputs. `prompt_hidden_states` returns each physical layer's output on its
last visit (the core's third pass), captured from `in.tail_out` (at a pass restart `h` is `h6`, not the previous
visit's output). `--lossless` refuses `--ablate`, as on every model.

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

After K1 of visits 5, 23 and 41 the raw partial block sum `h` entering the visit (the attention residual's running
partial, BF16 as stored: not the mixed or normed attention input), and after the output mix the final-normed state (the
head's input row), go into a feature buffer per column (4 × 4.6 KB), consumed by the drafter's context append and by
`f_rnn` from the last committed column. With DSpark on, every column's final state is a feature (the forward computes
the output mix over every column anyway).

**Rounding points** (the reference's, `.local` DSPARK_SPEC; the drafter decides only proposals): α = bf16(softmax);
the mix unrounded, the context norm and the context K's head norm rounded to BF16 (the K before RoPE); the anchor row
is the target's *normed* embedding, the mask rows raw; in the Markov step u, g, the new state bf16(kept + blend), ℓ, the
fold bias, the correction, the BF16 sum base + correction and the capped logit are each rounded to BF16. Every
reference drafts by argmax (lowest id on ties): the proposal is one-hot, and verification keeps a draft only when it
equals the target's own draw at its position, so speculation never changes the output.

### 13.4 Draft length and verification

- Proposal: the argmax of the softcapped Markov logits (lowest id on ties), as every reference drafts; a proposal's
  confidence is its softmax probability there (1 / Σ exp(L − L_max)), at no extra cost.
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
`refresh_fold`. Infernix already re-quantizes drafters (DFlash2 companions in 8 bits).

Built (2026-11-22): whenever DSpark is bound, the drafter's dense layers (each layer's q/k/v, the context's k/v, o,
gate/up/down) and the Fold become `q8_g32_fp16` at load (int8, an FP16 scale per 32; 597 → 317 MiB), the Fold's
Markov steps decoding it in place (54 MB, L2-resident across a round's steps). Not an option: only the drafts depend on
them, and a draft is kept only when it equals the target's own token, so no route's output changes (the Engine test's
DSpark = plain holds on every request). Measured (RTX 5090, Basalt's sampled fixtures, 1,200 tokens, 2 repetitions
an arm, slot DS3): a round's draft 1.94-1.98 → 1.84-1.87 ms on conv, 1.73 → 1.62-1.66 on prose, 2.17-2.27 →
2.07-2.13 on struct (about −5 %: the 0.69 GB target head dominates); decode 264.7 → 265.8 tok/s on conv, 191.3 →
194.3 on prose (plain 191.5), 399.4 → 402.9 on struct; pp2048 under DSpark unchanged (16,337-16,344 → 16,339-16,340).
The proposal head is `--lm-head-draft` (below). A plain-round policy that drafted only while the measured
tokens-per-ms gain exceeded plain decoding's was tried and rejected: it chose plain rounds on 339 and 598 of the prose
repetitions' rounds and lost on conv (265.8 → 260.9 tok/s).

**`--lm-head-draft`** (`models/inkling2/draft_head.*`): the artifact stores the vocabulary's rank order
(`proposal/token_rank`, int32 [V]: the model's own argmax counts on a chat corpus, `tools/freq_corpus`, the tokenizer's
special tokens first), and nothing else, so the stored bytes fix neither the list's length nor its format. At load the
Engine takes the first rows of the order, sorts them by id (so ties still break to the lowest id), gathers those rows of
the target head, the Fold and `fold_bias` from their stored BF16 and makes the two matrices 8-bit. A round's draft then
reads only the list: the head on 8-bit inputs, and each Markov step over the list on CTAs of one 32-row batch a warp,
its winning row mapped back to a token id. A target token off the list is never drafted; the round's verification
commits it as after any rejected draft, so no route's output changes.

Where the draft went (2026-11-23, prose, 7 drafts a round, the round's draft time with one part left out): 2.16 ms in
all; the Markov steps 1.27, the three layers 0.46 (attention 0.13, MLP 0.25), the head 0.36. Not the head, as read
above: each step's logits kernel ended in its last CTA merging the 512 CTAs' argmax and softmax partials on one thread
(ncu: 210 us a step at base clocks with the SMs' issue slots 12 % busy). That merge is now a fixed binary tree over the
CTAs in order (the argmax, lowest id on ties, unchanged; the softmax sum associated by the tree), and each warp takes
32 rows at a time, a transposing butterfly leaving one row's dot on each lane, which alone forms its logit (55 us a step
at base clocks before the second change). Drafts only: the drafted tokens and the accept counts matched on the same
text. Measured: a round's draft 2.16 → 1.24 ms (7 drafts, `--q8-weights` and `--a8-activations`); DSpark on Basalt's
fixtures (conv / struct / prose, auto, -r 2) `--q8-weights` 272.9 / 557.1 / 218.9 → 294.5 / 608.2 / 234.2 tok/s,
`--a8-activations` 276.1 / 610.3 / 212.7 → 296.5 / 665.1 / 227.6.

A Q8 head (`--q8-weights`, `--a8-activations`) then runs the drafter's 7 rows on 8-bit inputs (`Forward::project_head`
with the drafter's own 8-bit input rows: `projections_mma`'s tensor cores instead of the skinny route's FMA chains;
the only Q8 route that takes the head's unaligned FP32 rows). Drafts only, as above. Measured (2026-11-23, Basalt's
fixtures, auto, -r 2, against the skinny route on the same build): `--q8-weights` struct 621.4 → 626.8 tok/s, conv
302.8 → 304.9 and prose 242.6 → 242.4 within their ±; `--a8-activations` struct 670.7 → 674.2, prose 234.8 → 235.5,
conv 304.9 → 305.5 (within ±10). The policy's cost constants (0.12 a verify column, 0.18 a proposal) were re-tuned
against the same fixtures (0.17 / 0.29 measured, 0.145 / 0.235, 0.22 / 0.29, and a concave 0.14 / 0.30 + 0.15 for the
first column): none beat them on both routes once the 8-bit head was in, so they stand.

---

## 14. Prefill

### 14.1 Chunk-width invariance

Every prefill Op computes a column's result independently of the chunk's width and start: no split-K whose split
depends on the call, one attention route at every width, per-token quantization, and attention that appends the chunk's
K/V through the codec before reading it, so in-chunk and cached keys are the same represented values
([§9.3](#93-prefill)). The model has no recurrent layer whose kernel chunking regroups sums (its convs read a 3-position
history), so, as TokenSpeed notes for KV-only models, a prefix resumes at any token bit for bit. Then publication and
resumes are exact at any prefill-computed frontier, chunks can be any width (the default is the fastest measured,
4,096 columns today, no wider than the context), and the prompt's last two calls merge (L23). Where an Op cannot be made invariant, the Qwen rule
applies instead: a canonical chunk grid and grid-only publication on the reproducible route.

Built (2026-11-21): the first prefill route keeps the decode route's arithmetic at every width instead of a second
set of kernels. Every decode Op is already column-invariant (attention's units and splits depend only on a row's
position and lane; the experts' exact core is per entry; the skinny projections fold each column alone), so a wide
call gives each column the bits of single-column calls: prefill, decode and verify agree bit for bit, a prefix resumes
exactly at any position, and the chunk width may follow scheduling. A wide call differs only in how state is reached:
the conv rings' three history rows are snapshotted before a call with a lane of more than 29 columns and each ring
keeps its lane's last 32 positions; a lane appending more than a page runs every sliding visit over a staging ring
sized to the chunk (the window before it copied in by position, the last 1,088 positions copied back); attention runs
a lane's columns in groups of 16 and its rows in launches spanning at most 128 columns; the fused projections run a
wide call's passes inside one launch with the weights held in registers (W_o joins them in wide calls); the experts'
workspace and the forward's buffers take the chunk as their capacity. Measured (infernix_bench, C = 1, prefill 2,048 /
10,000): 752.8 / 736.5 tok/s at 16 columns a call, 1,166.6 / 1,140.8 at 128 (narrow passes), 1,505.5 / 1,457.9 at
512 with the wide projections (the default chunk), 1,398.1 / 1,359.3 at 2,048. A 2,048-column call's time goes to
K2 (41 %), the experts (30 %), W_o (12 %), the gates (7 %), attention and the router (5 % each): the skinny fold pins
each projection to SIMT FMAs reading x once per two rows (~6 TFLOPs at K2), and the experts' exact core costs
~3.3 µs a column per NVFP4 visit once every expert is resident.

**The prompt route (2026-11-21).** Prefill and scoring calls (`CallPlan::prompt`) now run their own arithmetic on
tensor cores at every width, while decode, verification and forced tokens keep the decode route: the owner's agreed
reproducible-route contract lets a prompt token differ from the same position computed by decode, and requires only
that each route give a column the same bits in any call (so any chunking of a prompt is exact, and verify columns
equal decode columns). It is the reproducible route's prefill too, with two rules for the work that follows: on
`--reproducible` no request resumes from state the decode route produced (a later turn's prompt containing an earlier
answer is prefilled, or resumed only from prompt-route state), and Replay re-runs generated tokens on the decode
route.

- **Projections and router.** The fused projections (K2, W_o, the gates' downs and forget/gain) and the router's FP32
  logits run `projections_mma` (ops::linear's MMA arithmetic: each output's k16 steps in ascending K chained through
  the FP32 accumulator, no split of K; FP64 oracle: worst 0.06 of a (K/16 + 5) 2^-23 Σ|wx| bound at FP32 output). Under
  `--q8-weights` the Q8 matrices run its Q8 form: per 32-group in ascending order, two chained MMAs from zero over the
  group's codes (as BF16, exact) and the BF16 inputs, then acc = fma(sum, the row's FP16 group scale, acc) (FP64 oracle
  over the represented weights within 2 (K/16 + 5) 2^-24 Σ|wx|). Each column's selection by a warp.
- **Experts.** The NVFP4 layers run offloaded_sparse_moe's grouped `mxf4nvf4` pipeline at Inkling's geometry with this
  Op's epilogues (bf16(acc · rs), the softcapped SwiGLU · γ, four-over-six h by warp, bf16(acc · rs_h); FP64 oracle
  over the kernel's own h within half a BF16 ulp plus the FP32 sum's bound: worst 0.992 of it; on the op test's random
  banks it matched the exact narrow route on every checked output). The MXFP8 layers run a grouped block-scaled FP8
  GEMM (m16n8k32 `mxf8f6f4`, the record's units as A fragments; h's MXFP8 groups by warp into the row the layer's
  input plane frees; outputs within half a BF16 ulp plus the FP32 sum's bound of FP64 over the kernel's h: worst
  0.990; 2 of 129,024 checked outputs differ from the decode route).
- **Attention.** `attention_prompt`: the decode route's prologue, then a CTA of 12 warps per (64 columns, KV head) over
  each row's fixed absolute 64-key units in ascending order: S = q kᵀ by BF16 MMA, the decode route's bias (the fma fold
  over r P, from a per-unit shared copy of the table's columns), online softmax in base 2, p rounded to FP16 for P V by
  FP16 MMA, FP32 accumulation (FP64 oracle within its stated bound: worst 0.612). It still spills (26 STL / 24 LDL at
  168 registers).
- **Measured.** A 2,048-column call (forward test, one eager call): 1,552 ms → 122 ms (16,778 tok/s). Engine
  (infernix_bench, C = 1, RTX 5090, pp2048 / pp10000 tok/s): 9,514 / 8,514 at 512-column chunks, 13,827 / 12,851 at
  1,024, 16,400 / 14,926 at 2,048, 16,387 / 16,312 at 4,096 (the default chunk; a lane's forward buffers take
  1,462 MiB at 4,096 against 318 MiB at 512); pp32 / pp128 / pp512: 1,165 / 3,823 / 10,238; `--q8-weights` pp2048
  16,014.5. One prompt call on each route (forward test): 32 columns 26.0 against 32.9 ms on the decode route, 128
  30.0 against 97.9, 512 46.3 against 351.8.

**The decode route's projections (2026-11-21).** Decode, verification and forced-token calls run the fused projections
(K2, W_o, the gates' downs and forget/gain; BF16 storage) on `projections_dmma`: K's 32-blocks in S equal runs, each
a chain of m16n8k16 MMAs from zero over its blocks in ascending order, the runs' partials added in order. S (1-16) is
a function of K and the call's segments' row counts only (about 2,048 warps a launch), and a call wider than 16
columns runs launches of 16, so a column's bits depend on its row and input alone (verify columns equal decode
columns). The weights stream once per launch by 16-byte `ld.global.cs` loads, four blocks ahead (FP64 oracle: worst
0.965 of a (K/16 + 5) 2^-23 Σ|wx| bound at widths 1-128). It replaced the SIMT fold, and W_o's `ops::linear` call at
widths up to 16. A first instance of `projections_mma`'s arithmetic at decode widths had lost to the fold at widths 1
and 2 (+20.8 %, +18.0 %; −6.4 % at 8), and a first dmma with 4 runs for K ≥ 512 lost at width 1 (+2.8 %); with S
chosen per shape the decode round at 4K context (forward test, graphs; K2 and gates on dmma, W_o still on
`ops::linear`) went 6.190 → 6.099 ms at width 1 (−1.5 %), 6.967 → 6.868 at 2 (−1.4 %), 8.937 → 8.227 at 4 (−7.9 %),
12.399 → 10.574 at 8 (−14.7 %). The decode route's bits changed with it: the forward test's free-running logits moved
by a mean KL of 0.27 (max 3.4) from the fold's, through the routing's amplification of last-bit differences (the
teacher-forced stages stay within their bounds; greedy agreement with the reference from position 23 holds).
`--q8-weights` runs its Q8 matrices on `projections_dmma` too (6a3f2657): a thread's 8 codes of a 32-block widened to
BF16 exactly, the block's two MMAs from zero, then the block sum times the row's FP16 group scale fused into the
accumulator; codes and scales stay raw in flight, 8 blocks a warp (29c91402). Against the fold (decode round at 4K,
2026-11-22): +5.1 % at one column, +2.6 % at two, −1.5 % at four, −4.3 % at eight. The fold below four columns was
built and withdrawn: a call's width depends on the other requests, so the Engine test's `--q8-weights` C = 4 tokens
differed from the request alone.

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

Built with `--vision` (images; no video). The semantics, from transformers' `Inkling2ImageProcessor` and
`Inkling2VisionModel` and SGLang's `inkling2_vision.py` (which agree):

- **Processor** (`frontend/image.{h,cpp}`): the decoded RGB; a transparent image (`media::decode::Policy::keep_alpha`:
  an alpha channel or PNG tRNS) composited over a 204/153 checkerboard of 8-pixel squares,
  `(rgb · a + bg · (255 − a) + 127) / 255`; padded right and bottom to whole 40-pixel cells with −1 on the 0-255
  scale; `(v − 127.5) / 127.5` in FP32, then BF16. One image token per cell. The reference never resizes (the model
  card recommends 40-4,096-pixel sides); an image of more cells than `--vision-max-merged` (default 32,768) is first
  downscaled bicubically (the shared `media::decode::resize_bicubic`, Qwen3.5's) to the largest whole cells that fit
  with its aspect kept (smart_resize's floor), or refused when the request asks (`ImageResizePolicy::RejectOversized`).
- **Prompt** (`frontend/frontend.cpp`): each image part renders the template's `<|content_image|><|unused_200054|>`,
  whose one placeholder becomes the image's cells' tokens; client text is a literal span, so a typed placeholder stays
  text. The context check counts the expanded prompt from the images' geometry before any pixel is decoded.
- **Tower** (`execution/vision.{h,cpp}`), one image at a time: patches of 20 × 20 × 3 in merge-major order (cell
  row-major, then (dy, dx); values (ky, kx, channel)) through the 1200 → 1024 linear, plus the 32 × 32 position table
  resized by separable bilinear taps (`src = (i + 0.5) · 32 / n − 0.5`, two taps clamped to the table); the class token
  `class + cls_position` in front; LayerNorm (ε 1e-5); 23 layers of LayerNorm → fused QKV (+ bias) → the axial rotary
  on interleaved pairs (pairs 0-15 by column + 1, 16-31 by row + 1, θ_j = 10000^(−(j mod 16)/16); the class token at
  (0, 0)) → D64 attention over the whole image (scale 1/8) → output projection → `x + ls₁ ⊙ (y + b)`; LayerNorm → MLP
  (exact GELU) → `x + ls₂ ⊙ (y + b)`; then each cell's 4 consecutive patch rows as one [4096] row through `proj_in`
  (no bias) → exact GELU → `proj_out` → 2304. Token-parallel stages run 4,096 tokens at a time; Q/K/V and the residual
  are whole. Ops: `ops::linear` (BF16), `layer_norm`, `gelu`, `add_bias`, `vision_pos_embed_add`, the packed attention's
  D64 profile (`softmax_attention.h`, added beside Qwen3.5 Vision's D72), and `inkling2_vision.h`'s QKV bias + rotary
  and layer-scaled residual. An image's rows depend on its patches alone (the same bits in any prompt or batch).
- **Prefill** (`program/program.cpp`): the Program owns the tower and its buffers (in its device plan: ~10 KB a patch
  token, 1.31 GiB at 32,768 image tokens, 0.46 GiB at 10,609, a 4,096-pixel square; the tower's weights 0.63 GB); each
  prompt chunk, right after the embedding and its norm, encodes the images it overlaps (an image across chunks once)
  and copies their rows over its embedding columns, as the reference scatters them after `embed_norm`. Scoring does
  the same. The text model has no positional encoding beyond the relative bias, so image tokens are ordinary positions
  (no MRoPE). The images ride in the request's base plan; the prefix cache keys each block by the images that begin
  before its end (a hash of each image's processed patches and its placement), and no resume point lies inside an
  image or past its start in its first block (tap_planner.h media_exclusion).
- **Offload** (`--vision-offload on`; `auto` keeps it off for Inkling 2): the loader binds the tower's weights
  HostPinned (`Model::vision_host()`: each group's objects with their pinned and staging offsets); the tower keeps a
  device region of the patch embedding, position table, class row, pre-norm and projector (uploaded once) and two
  layer slots, ~0.1 GB of device memory for the resident tower's 0.63 GB. Each encode streams the 23 layers through
  the slots on the transfer stream: layers 0 and 1 once the previous encode's kernels are past (an event fence), then
  at the top of layer l the compute stream waits for its slot's upload and, layer l − 1 having been issued, the other
  slot refills with layer l + 1, so an upload overlaps a layer's compute. The kernels and their order are the resident
  tower's, so the rows are the same bits.

Qualification: the processor's BF16 pixel words equal transformers' processor's exactly (`test_vision_real.cpp`, cell
SHA-256s of `tests/fixtures/inkling2/vision_golden.json`); an FP64 oracle of the tower written from this statement
agreed with SGLang's tower run in FP32 to 1.0-1.3e-6 relative RMS on three asymmetric images, and with its BF16 run to
1.2-3.0 % (the BF16 run's own drift), while plausible slips move the rows 8 % (the position table's axes swapped),
18 % (align_corners taps) and 53-81 % (a cell's pixels transposed). The GPU tower against SGLang's FP32 rows (criterion
4 %, stated before the run): 1.46 % and 1.80 % relative RMS (cosine 0.99989, 0.99984) on the fixture's 2 × 3- and
2 × 5-cell images, the same bits re-encoded. Through the Engine (`test_engine_vision_real.cpp`, RTX 5090,
2026-11-22): a red disc on blue asked for its colour answers "red"; two image requests and a text request give their
solo tokens at C = 3; prefill chunks of 16 and 4,096, eager and graphs, the same tokens; DSpark the same tokens as
plain decoding; an Engine without `--vision` refuses images. Offload (2026-11-22): an offloaded tower's rows equal the
resident tower's bit for bit on both fixture images (encoded twice, in either order), 504 MiB of device memory freed,
553 MiB streamed an encode; the 6- and 10-cell images encode in 22.1 and 22.8 ms against 14.3 and 15.8 ms resident
(best of 3: 553 MiB at ~26 GB/s, the link's rate, so a small image's encode is link-bound); through the Engine, the
small and the 1,296-cell image and a sampled request give the resident tower's tokens. Not measured: a large image's
offloaded encode time and prefill throughput with images.

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
- **Output session.** Channels from `response_template`: thinking, content, tool calls (JSON per call). Constrained
  output (built 2026-11-24: `frontend/constraints.cpp`, [constrained-decoding §4.4](constrained-decoding.md#44-inkling-2-mini))
  states the whole turn's block framing as one grammar: thinking blocks, then the answer block holding the content
  constraint, or call blocks holding `{"name": ..., "args": ...}` under the tool policy (TokenSpeed's `inkling2_json`
  frames the same answer block).
- **Thinking budget.** A turn may hold several thinking blocks, so the budget counts them together and holds for the
  whole turn (Qwen's one `<think>` block makes its one-shot control a turn budget; Inkling's does not): once the
  control has closed a block or a block has ended at the budget, `reasoning_ban()` refuses `<|content_thinking|>` at
  every later column (its close is the opener itself, so the ban never lifts). Without it the model could open a new
  block after the control and think uncapped to the output limit (Basalt #588, llama.cpp #32836). Only turns that use
  up their budget sample differently.
- **Stops.** EOS `<|content_model_end_sampling|>`; `<|end_message|>` ends a channel, not the turn.

---

## 18. Artifact and conversion

### 18.1 Recipe

`inkling2_mini` (`tools/convert/inkling2_loop.py`): every tensor bit-exact. The NVFP4 experts are `nvfp4_mul` banks whose
multiplier plane is `fl32(amax / denominator)` for generic decoders, with the checkpoint's exact FP32 `amax` [128, 2] and
denominators [2] stored beside them (`moe/expert_amax`, `moe/expert_denominators`) for the pinned row scale
`(amax_x · amax_w) · rcp_rn(1536 · den)`; the MXFP8 experts are `fp8_e4m3fn_g32_e8m0` banks; BF16 elsewhere (packed at
load, not in the file); FP32 queries and router bias. The text config names each layer's expert arithmetic
(`expert_formats`: four-over-six W4A4 or MXFP8 W8A8), and the expert Uses permit it (`AllowA4`, `AllowA8`, no static
input scales: the activation scales are per token by the model's own rule). Value-preserving transforms only: each loop
layer's cross-pass queries as [4, H] in (q, k, v, mlp) order; the embedding, head and Markov embedding cut to the real
200,058 rows (every token id below that is assigned; the padded rows are non-zero but unreachable); static convs
lag-major [k, C], row 0 the current token; DSpark `Fold` and `fold_bias` derived as `refresh_fold` does at load (FP64
products copied into BF16; D2 may store FP8). The `vision` component (`--components text,dspark,vision`) stores the
tower as the checkpoint does (BF16; the patch embedding's columns in (ky, kx, channel) order, the position table
[1024 = 32 × 32, 1024] row-major by (row, column)), q/k/v and their biases each one parent, and the class token's row
derived as the one BF16 addition `class_embedding + cls_position_embedding` both references make.
`--token-rank FILE` adds `proposal/token_rank` after every other parameter (their bytes and order do not change).
`tools/convert/verify_inkling2.py` re-derives every bound word from the checkpoint with its own reader and fails on any
difference or unconsumed checkpoint tensor (the tower's only when the artifact has no vision component); it checks the
rank is a permutation of the vocabulary with the special tokens first and, given `--token-rank`, equal to that file. Measured
2026-11-22: `out/inkling2-mini/inkling2-mini-vision.infernix` (text, dspark, vision) 20.29 GB, 908 objects, converted in
66 s; the verifier passed (1,168 direct tensors, 24 expert banks, 3 derived).

### 18.2 Layouts

NVFP4 experts in `nvfp4_expert_rg16_v1`, the Flash-Next narrow kernels' layout; MXFP8 experts in `mxfp8_expert_rg16_v1`
(16-row groups of 528-byte units per 32-element K block: 512 code bytes in `mma.m16n8k32` A-fragment order, then the 16
rows' UE8M0 scales; the stored layout is final: another code order, were one ever measured faster, is a value-preserving
permutation an engine can make at load), both with the checkpoint's gate/up row interleaving (2i gate, 2i + 1 up). Attention matrices grouped per layer into the K2 parent [8,576 × 2304] (q, r, gate,
k, v) and a small parent (sink, taps); the four message gates' down projections one parent, each gate's forget and gain
another; the head row-major.

### 18.3 Frontend resources

`tokenizer.json`, `tokenizer_config.json` (with `response_template`), `chat_template.jinja`, `generation_config.json`,
kept byte for byte (the Qwen resource normalizers do not run on this family); `processor_config.json` with the vision
component, kept byte for byte after the converter checks it names the processor the frontend implements
(`Inkling2ImageProcessor`, mean and std 0.5, rescale 1/255, 40-pixel cells).

---

## 19. Configuration surface

| Option | Values for this family |
|---|---|
| `--spec` | `off`, `dspark`, `auto` (= `dspark` when the artifact carries the drafter) |
| `--draft-tokens` | `auto` (calibrated), 1-7 |
| `--kv-dtype` | the full visits' pools ([§7.6](#76-kv-profiles)): `bf16` (default), `bf16p`, `int8` (recommended), `fp8`, `nvfp4`, `k8v4`, `vq2`, `k4v2`; the sliding windows `bf16` under `bf16`, `bf16p` under `bf16p`, `int8` under every other |
| `--vision` | images (no video) from an artifact with the vision component; `--vision-max-merged` image tokens an image (default 32,768); `--vision-offload on` streams the tower from pinned host memory |
| `--prefill-chunk` | default 4,096; a multiple of 128 up to 8,064 ([§14.1](#141-chunk-width-invariance)) |
| `--reproducible`, `--lossless` | as for every family |
| `--lora` | a PEFT adapter of the attention and message-gate projections, on every text route ([§11.1](#111-lora-adapters---lora)) |
| `--ablate` | a direction cut from each layer's attention and MLP writes ([§12](#12-residual-mixing-message-gates-and-short-convolutions)); refused with `--lossless` |
| `--q8-weights` | approximate 8-bit K2, W_o and head ([§11](#11-dense-projections-and-the-head)); refused with `--reproducible` and `--lossless` |
| `--a16-activations` | the routed experts on 16-bit activations (`w4a4_sparse_moe.h`'s `ExpertActivation::A16`): NVFP4 layers on Qwen3.8-Flash-Next's exact W4A16 arithmetic (x and h as 21-bit fixed point per column, exact int64 sums, the multiplier fl32(amax / denominator)), MXFP8 layers W8A16 (E4M3 widened exactly to BF16, a block's two MMAs from zero, its power-of-two scale fused); accepted with `--reproducible`; prompt calls run the narrow kernels (no A16 grouped GEMM: pp2048 16,387 → 3,383 tok/s, tg128 184.9 → 188.1, 2026-11-22); MXFP8 decode at one column runs the separate router and experts. `--a8-activations` ([§11](#11-dense-projections-and-the-head)): 8-bit activations in every call for the dense Q8 matrices and the NVFP4 experts (implies `--q8-weights`); refused with `--reproducible`, `--lossless` and `--a16-activations` |

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
(L27). Work happens in a worktree off main; nothing merges with a temporary switch left in. Order of work: M0, M1, M3,
M4, M5, M6, then M2, M7-M9. The lossless route follows the fast one because the FP64 Op oracles and a private CPU
reference forward (teacher-forced against the Transformers PR's golden completion: 29 of 30 tokens, the miss a 0.15
margin) qualify M3-M5 without it, and its kernels run inside the Program M4 builds.

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
| 8-bit activations for dense decode projections on the default route | measured losses (L24); the opt-in `--a8-activations` takes them on purpose ([§11](#11-dense-projections-and-the-head)) |

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
