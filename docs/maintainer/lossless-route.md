# The lossless route (`--lossless`)

Status: in development (branch `feat/lossless`, 2026-10-21): Qwen3.5 and Qwen3.8-Flash-Next scoring and
generation. The owner's decisions behind it are
recorded in [model quality §6.1](model-quality.md#61-the-reproducible-route-and-the-fast-paths); this document
is the authority for what the route computes and how it is qualified.

## 1. Definition

`--lossless` computes the arithmetic the model was trained in, exactly:

- **The model's operations, in its training framework's order.** For Qwen3.5/3.6/3.8 the authority is
  Hugging Face Transformers' `modeling_qwen3_5.py` and `modeling_qwen3_5_moe.py` (version 5.5): every
  PyTorch operation of the forward pass, with the dtype that operation produces there (BF16 for the
  model's activations, FP32 where the code upcasts). A fused kernel that the model was trained with
  counts as one operation: FLA's gated delta rule (with its in-kernel L2 norm) and flash attention.
- **Each operation correctly rounded.** Its result is the exact value of the operation on its inputs
  as given, rounded once to the operation's dtype (round to nearest, ties to even; IEEE infinities and
  NaN). Elementwise IEEE operations are already correctly rounded; reductions (matrix products, sums,
  means) are exact and then rounded; transcendental functions (exp, sigmoid, SiLU, softplus, rsqrt,
  cos, sin) are correctly rounded binary32 functions. Zeros carry IEEE's signs: a product's zero the
  exclusive or of its operands' signs, a negative value rounding to zero -0, the sum of two zeros as
  IEEE adds them, and a reduction whose exact value is zero +0 (its accumulator starts at +0).
- **Weights decoded exactly.** A quantized weight is its format's value as a real number (a row-split
  code times its FP16 group scale; NVFP4 an E2M1 code times its E4M3 block scale and the global scale;
  row FP8 an E4M3 code times its BF16 row scale). No weight is rounded to BF16 first.
- **BF16 KV with V stored losslessly**, the only KV format of the route: `--kv-dtype bf16`, whose
  pages on this route hold K and V as BF16 bit for bit (elsewhere `bf16` stores V as FP16, which holds
  every BF16 value with 2^-17 <= |v| < 2^16 exactly but rounds the rest).

So the output is a function of the request and the artifact alone: no batch, chunk, prefill/decode,
speculative, placement or build dependence, and anyone can check any value with exact arithmetic.
It is not Hugging Face's bits (theirs depend on GPU, kernels and batch); it is what Hugging Face
would compute if every operation returned its correctly rounded result.

## 2. Qwen3.5 operations

Notation: `bf16(v)` and `fp32(v)` are the correctly rounded value of the exact real v.

**Embedding.** The token's row, decoded exactly and rounded to BF16 (an exact copy for BF16 tables).

**RMSNorm** (`Qwen3_5RMSNorm`, weight w): with x the BF16 input upcast exactly,
`m = fp32(mean(x^2))`, `r = fp32(rsqrt(fp32(m + eps)))`, `y = fp32(x r)`,
`out = bf16(fp32(y fp32(1 + w)))`.

**Gated full attention** (`Qwen3_5Attention`):
- `[q | gate] = bf16(h W_q)`, `k = bf16(h W_k)`, `v = bf16(h W_v)` (each projection one exact sum per
  output, rounded once);
- `q = q_norm(q)`, `k = k_norm(k)` (RMSNorm per head);
- RoPE: `cos, sin = bf16(fp32(cos(fp32(position fp32(inv_freq)))))` and likewise sin, per HF's FP32
  frequency computation (inv_freq = fp32(1 / theta^(2i / d))); `q_rot = bf16(bf16(q cos) + bf16(rotate(q) sin))`
  (the BF16 tensor operations of `apply_rotary_pos_emb`), the rest of the head unrotated;
- attention as one operation: `o = bf16( sum_j p_j v_j / sum_j p_j )` over the causal keys, with
  `s_j = (q . k_j) d^-1/2` exact (d^-1/2 = 2^-4 for d = 256), `m = max_j s_j`,
  `p_j = fp32(exp(fp32(s_j - m)))` (flash attention's binary32 exponentials of a binary32 argument),
  and the two sums exact;
- `o = bf16(o bf16(sigmoid(gate)))`; `out = bf16(o W_o)`.

**Gated DeltaNet** (`Qwen3_5GatedDeltaNet`):
- `mixed = bf16(h W_qkv)`, `z = bf16(h W_z)`, `b = bf16(h W_b)`, `a = bf16(h W_a)`;
- causal conv (`causal_conv1d_fn`, one operation): `c = bf16(silu(fp32(sum_tap w x)))`, the sum exact before
  its one rounding to binary32 (the kernel accumulates in binary32 and applies SiLU to that: SiLU of the exact
  sum rounded once to BF16 differs where the SiLU lands within a binary32 step of a BF16 midpoint, which the
  layer-by-layer comparison with the publisher's code found at one value of the 27B's layer 28);
- `beta = bf16(sigmoid(b))`; `g = fp32(-fp32(exp(A_log)) fp32(softplus(fp32(a + dt_bias))))`;
- the gated delta rule (FLA's kernels, whose L2 norm runs in FP32; the recurrence is
  `torch_recurrent_gated_delta_rule`'s, each operation correctly rounded, since a recurrence has no
  single exact output): `q̂ = fp32(q fp32(rsqrt(fp32(fp32(sum q^2) + eps))))` and `k̂` likewise,
  `q̂ = fp32(q̂ fp32(d_k^-1/2))`; then per head with state S (FP32 [d_k, d_v]), operation by operation as
  in that function: `S = fp32(S fp32(exp(g)))`, `u_j = fp32(sum_i fp32(S_ij k̂_i))` (the products rounded,
  their sum exact and rounded), `δ = fp32(fp32(v - u) beta)`, `S_ij = fp32(S_ij + fp32(k̂_i δ_j))`,
  `o_j = bf16(fp32(sum_i fp32(S_ij q̂_i)))`; prefill runs the same recurrence token by token, so a token's
  bits do not depend on how the prompt was split;
- gated norm (`Qwen3_5RMSNormGated`), o upcast exactly: `n = bf16(fp32(o fp32(rsqrt(fp32(fp32(mean(o^2)) + eps)))))`,
  `y = bf16(w n)`, `out = bf16(fp32(y fp32(silu(z))))`;
- `out = bf16(out W_out)`.

**MLP** (`Qwen3_5MLP`): `g = bf16(x W_gate)`, `u = bf16(x W_up)`, `a = bf16(silu(g))`,
`hm = bf16(a u)`, `out = bf16(hm W_down)`.

**Sparse MoE** (`Qwen3_5MoeSparseMoeBlock`):
- router: `l = bf16(x W_router)`, `p = fp32(softmax(l))` per element (one operation), top-k by value
  (ties: the lower expert id first), `w_i = fp32(p_i / fp32(sum_topk p))`;
- each selected expert e (ascending id): its MLP as above, `y_e = fp32(out_e w_e)`; the block's sum
  starts at BF16 zero and adds `bf16(y_e)` in ascending expert id, each sum rounded to BF16;
- shared expert: its MLP, `s = bf16(bf16(sigmoid(bf16(x w_gate))) out_shared)`;
- `out = bf16(experts + s)`.

**Decoder layer.** `h = bf16(residual + mixer(norm(residual)))`, then the same for the MLP or MoE.

**Head.** `norm(h)`, then logits `fp32(h W_head)` (each an exact sum rounded to binary32: the scoring
and sampling inputs). Transformers returns the head's output in BF16, and training's loss upcasts
those values; the route keeps the exact sum rounded once to FP32 instead (owner's decision,
2026-10-21). `--logits-dtype bf16` (every route, 2026-10-25) gives Transformers' own dtype: each exact
sum correctly rounded to BF16 (once, not through FP32), for comparisons with references and engines
that round the logits. The logits feed nothing inside the model, so no layer is calibrated to their rounding,
and a BF16 rounding of logits of magnitude 8-32 (rms error 0.02-0.04) would add a KL of roughly
1.5e-4 to 6.5e-4 per position (an estimate) to every gate measured against the route. llama.cpp's
KL baselines also use FP32 logits.

## 2a. Qwen3.8-Flash-Next operations (`Qwen4ExpForCausalLM`)

The authority is Transformers' `modeling_qwen4_exp.py` (main at 3bb8b727, 2026-10-20; downloaded to
`.local/hf-src-qwen4exp`), read with the same rules as §1. Two rules settle what the code leaves to the
implementation: a Python scalar meets a BF16 or FP32 tensor as a binary32 value (PyTorch's opmath), and an
operation on BF16 tensors returns the BF16 rounding of its exact value (a `.mean` or `.sum` included).
H is hidden_size, C = hc_count (4) streams, R the C·H-wide residual stream (BF16).

**Embedding and streams.** The token's row decoded exactly and rounded to BF16; R = C copies of it.
Rotary cos and sin as in §2 (the text positions, interleaved MRoPE).

**Hyper-connection** (`Qwen4ExpTextGatedResidual`, one before the mixer and one before the MoE of every
layer, and the final mixer without the inject): n = the grouped RMSNorm of R (each stream of H its own
RMSNorm with 1 + w, §2's arithmetic); d = bf16(n W_down) (C·H to hc_lowrank); d = bf16(d / C);
m = bf16(silu(d)); u = bf16(m W_up); u = bf16(sigmoid(u)); x = bf16(mean over the C streams of
bf16(u n)) (the input of the block); with the inject, j = bf16(n W_inject) (C outputs), j = bf16(j / C),
j = 2 bf16(sigmoid(j)) (exact). After the block's output y: R = bf16(R + bf16(y j_c)) per stream c.

**PLE layer** (the layers listed in ple_layer_ids, before the attention hyper-connection): the n-gram
ids (integer hashing, exact); e = the n-gram rows decoded exactly and rounded to BF16; k = the grouped
RMSNorm of bf16(e W_key) (C·H); v = bf16(e W_value) (H); q = the grouped RMSNorm of R;
g_c = bf16(bf16(sum_i bf16(k_ci q_ci)) / fp32(sqrt(H))) per stream (the sum exact, rounded once);
g_c = bf16(sqrt(bf16(max(|g_c|, fp32(1e-6))))) sign(g_c) (the clamp and the square root each a BF16
operation, sqrt correctly rounded); gated =
bf16(bf16(sigmoid(g_c)) v); c = the grouped RMSNorm of gated; the dilated depthwise causal convolution
of c (dilation ngram_size, kernel ple_conv_kernel_size, zeros before the sequence) rounded to BF16, then
bf16(silu(.)) (F.conv1d and F.silu, two operations: no fused kernel serves a dilated convolution);
out = bf16(gated + conv); R = bf16(R + out).

**Gated DeltaNet** (`Qwen4ExpTextGatedDeltaNet`): as Qwen3.5's (§2), the gated norm's gate being
output_gate_type's function (sigmoid or SiLU) of z, in binary32. The query scale is FLA's
multiplication by fp32(d_k^-1/2) (the kernel the model was trained with; the torch fallback divides).

**QSA attention** (`Qwen4ExpTextAttention` with its indexer):
- indexer: [q_i | k_raw] = bf16(x W_index); q_i = RoPE(RMSNorm(q_i)) per index head; the visible keys
  0..t of query t form b = floor((t + 1) / r) blocks of r (r = indexer_compress_ratio) and a tail;
  each block's key bf16(fp32(mean of its r raw keys)) (the mean rounded to binary32, then to BF16), its
  RMSNorm, RoPE at the block's first position; score_b = fp32(fp32(sum over index heads of
  relu(fp32(q_h . key_b))) / fp32(sqrt(d_index))) (each dot product and the sum exact, rounded once);
  the min(budget / r, b) highest scores (ties: the lower block), their tokens and the tail selected;
- attention: as Qwen3.5's (§2: q, gate, k, v projections, q_norm, k_norm, RoPE, the exact scores and
  correctly rounded exponentials, bf16(sum p v / sum p)) over the selected keys only (SDPA's fused
  operation; eager attention is not the trained kernel); o = bf16(o bf16(sigmoid(gate))); bf16(o W_o).

**Sparse MoE** (`Qwen4ExpTextSparseMoeBlock`): l = bf16(x W_router); p = fp32(softmax(l)); the top k by p
(ties: the lower expert); w = bf16(fp32(p / fp32(sum of the top k))) (**BF16**, unlike Qwen3.5: the code
casts to the logits' dtype); each selected expert in ascending id: its MLP (§2), y_e = bf16(out_e w_e)
(a BF16 product), the block's sum from BF16 zero adding y_e, each sum rounded to BF16; the shared expert
and its gate as in Qwen3.5; out = bf16(experts + shared).

**Head.** x = the final mixer's output (BF16, H); logits fp32(x W_head) as for Qwen3.5.

## 3. Kernels

**Linear** (`lossless_linear`, `include/infernix/ops/lossless_linear.h`): each 64-element block of an
activation column on a 29-bit integer grid relative to its largest exponent (elements 22-43 binades
below it on a second grid 22 binades lower, multiplied only for the tiles that hold them: SiLU outputs
of strongly negative gates are such elements), four byte limbs against
the weights' integer codes on int8 tensor cores, each scale group's sum exact in int64 and accumulated
exactly in int128, one rounding; an output that lost any bit (a block spanning more than 43 binades, a
term more than 64 binades under the largest possible, a non-finite input) is recomputed with a wide
fixed-point accumulator (352 to 608 bits by format). Formats (each weight its format's exact value, tensor-formats.md): row-split
Q2/Q4/Q5/Q6 (per 64) and Q8 (per 32); `nvfp4` (twice the E2M1 code against 16-element groups on
m16n8k16, the E4M3 scale applied per group, the finished sum divided exactly by the matrix divisor:
the quotient keeps a sticky bit); `fp8_e4m3fn_row_bf16` (the E4M3 value times 2^9 as three balanced
7-bit digit planes, the finished sum multiplied exactly by the row's BF16 scale); BF16 (each 64-element
block of a weight row on a 29-bit grid below its largest exponent as five balanced 7-bit digit planes,
groups of 32, weights 22-43 binades below on a lo grid multiplied only for the 64-row tiles holding them;
a row with a weight further below, or a non-finite one, takes the fallback with a 576-bit accumulator;
a scan of the call's rows finds each block's exponent first); `fp8_e4m3fn_g32_e8m0` (MXFP8: the E4M3
value times 2^9 as three digit planes, the 32-element group's power of two its scale); the GGUF blocks of
`ggml_rows_v1` and `ggml_tiles16_v1` (Q2_0, Q4_K, Q5_K, Q6_K, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS, IQ4_NL: the
integer code of `ggml::chunk` against its group's scale mant(d) sub 2^(e(d) - 25) kappa per 16, 32 or 64
elements, and for Q4_K/Q5_K the min term -mant(dmin) min_sub times the group's activation sum, exact per
column and grid from the encoding). The grouped form
(`lossless_linear_grouped`) runs a routed-expert bank: each 64-column tile with its own expert's rows. The
tile mode (`lossless_linear_tiles`) runs offloaded experts: each 64-column tile reads its own expert
record (NVFP4 `nvfp4_expert_rg16_v1` units, the finished sum times the matrix's FP32 multiplier;
`fp8_expert_b128_v1` rows with their BF16 128 x 128 tile scales per 64-element group; GGUF expert rows),
its columns through an input map and written through an output map.

**Offloaded experts** (`offloaded_moe::WideA16::kLossless`, `ops/offloaded_sparse_moe/cuda/lossless_expert.cuh`):
the call reads each job's column count back once (the route is eager), and each staging pass of jobs is
cut into 64-column tiles of one job; column chunks of 2,048 run the gate/up projection (the tokens'
x through the input map), h = bf16(bf16(silu(g)) u) and the down projection (written to each column's
entry). Every bank serves, an A4 bank's experts too (their weights' multipliers, not the A4 alphas), on
the GPU only.

**Sparse MoE** (`lossless_router`, `lossless_moe_route`, `lossless_moe_gather`, `lossless_moe_combine`,
`include/infernix/ops/lossless_ops.h`): the softmax correctly rounded per element (certified in double,
otherwise double-double); each (slot, column) placed in its expert's run of whole tiles (placement is
free: no column's result depends on it; N pairs take at most 64 floor((N + 63 min(E, N)) / 64) gathered
columns, 512 for one decoded column), the grouped projections, then per column the experts in
ascending id summed in BF16 as `index_add_` does, the shared expert and the residual.

**RoPE** (`lossless_rope_frequencies`, `lossless_rope_table`, `lossless_rope_apply`): inv_freq and the
angles correctly rounded per operation, cos and sin correctly rounded binary32 (double `sincos`
certified, otherwise a double-double reduction with pi/2 in four pieces and Taylor sums), one table per
window shared by every attention layer.

**Gated delta rule** (`lossless_gated_delta_rule`): four CTAs per value head (32 state columns each, two
CTAs per SM, so the 27B's 192 CTAs are one wave) run the recurrence token by token, a column's 128 state
rows on 8 threads (16 each, in registers); each
binary32 sum is exact in a 320-bit accumulator, the 8 partial sums added exactly, and rounded once.

**Flash-Next operations** (`lossless_flash.cu`): the grouped RMSNorm (`lossless_rmsnorm_grouped`), the
hyper-connection gates, mix and injection (`lossless_hc_gates`, `lossless_hc_mix`, `lossless_hc_inject`:
sums of BF16 values exact at 2^-134), the PLE gate and its dilated convolution (`lossless_ple_gate`,
`lossless_ple_conv_inject`: products of BF16 values exact at 2^-268), the QSA indexer's pooled keys
(`lossless_qsa_pool`) and block selection (`lossless_qsa_select`: each query's relu-summed head scores
exact, divided correctly rounded, the budget's largest by a radix search on the binary32 bits with
ties to the lower block); a sum with an Inf or NaN term takes IEEE binary32 arithmetic.

**Attention** (`lossless_attention`): each 64-element block of a q or k row on a 29-bit integer grid
below its largest exponent, so a score is exact int64 block sums placed in an int128 (a pair whose row
has an element off its grid is scored product by product instead); the exact maximum (pass A, its own
kernel with only the q and k tiles in shared memory), then (pass B, scores recomputed) fp32(s - m)
exact, p correctly rounded; sum p exact, and sum p v from p on a 2^-48 grid and v on its channel's
grid, the dropped bits bounded; the quotient rounded to BF16 when its error interval allows, else the
query is recomputed exactly (every score and sum exact, the rounding decided by exact comparisons). With
a selection (QSA), a query's keys are its tail and its selected blocks.

## 3a. The route

`EngineOptions::lossless` (`--lossless` of `infernix-perplexity`; `infernix-certify reference`)
selects it on Qwen3.5/3.6/3.8 (`Qwen3_5ForCausalLM`, `Qwen3_5MoeForCausalLM`;
`src/models/qwen3_5/execution/lossless.cpp`): a CausalScoring Engine that scores each window in one pass
(its prefill chunk covers max_context), no fast path, FP32 logits. Its attention reads the window's own
keys, so the KV cache plays no part; GDN and the convolution start from zero state. MLP and MoE passes
split the columns 4,096 at a time (column-independent, so the split changes nothing).

Generation (`infernix --lossless`, `infernix-serve --lossless`; text requests, `--kv-dtype bf16`, no
speculation; decode rounds as CUDA Graphs) runs the same layers in their cache form
(`execution::LosslessCache`): a prefill chunk or a decode round's columns continue their sequences.
Attention stores each column's K and V bit for bit in the paged cache (`lossless_kv_append`; the
Program allocates the bf16 pages with BF16 V, `paged_kv_lossless_storage_layout`) and reads the
sequence's keys 0..position through its block table (`lossless_paged_attention`); the convolution and
the gated delta rule continue from the lane's source state slot and leave their state in its
destination slot (`lossless_causal_conv_slots`, `lossless_gated_delta_rule_slots`, the state pool's own
layouts). Every operation is the window form's operation on the same values, so a token's logits do not
depend on the prefill chunk, on prefill or decode, on the other requests, on the prefix cache (a resumed
state is an exact copy; the Host tier must keep FP32 state) or on preemption and replay. Sampling and
the grammar constraints read the FP32 logits; sampled requests draw by Gumbel-max. The prefix cache's
file records the route, so a lossless Engine never resumes another route's state.

Attention's key ranges: a call with few CTAs (decode) splits each query's keys into ranges over more
CTAs, a pass of partial maxima, a pass of partial int128 sums from the overall maximum, and a combine
kernel adding them; exact sums make the split invisible in the output.

On Qwen3.8-Flash-Next (`Qwen4ExpForCausalLM`; `src/models/qwen4_exp/execution/forward_lossless.cpp`,
TextConfig::lossless) causal scoring takes the same Engine conditions (one call per window, no fast path,
drafter or vision): Forward's layers run the lossless steps of §2a over the call's own columns. The
projections take each weight's exact value whatever its format (BF16, Q8 and Q4 row-split, row FP8,
MXFP8, GGUF segments with the GDN output's block permutation), the QSA attention reads the window's own
keys (each query's blocks selected from the pooled-key scores once the window holds more complete blocks
than the budget), GDN and the PLE convolution start from zero, the routed experts take the offloaded
lossless pass through serial staging passes (the Program turns off the prefill stream, layer walks and
the CPU split), and nothing is written to the KV cache or the recurrent states. Text RoPE without YaRN
only (the interleaved pairs over the three axes). Generation (text, `--kv-dtype bf16`, no drafter, n-gram
drafts or `--expert-tail-skip`, eager rounds) runs the cache form (ForwardBatch::lossless_keys, the call's
largest position + 1): K/V into the pages (BF16 V), each block the call completes pooled from its raw index
keys (those before the call from the lane's tail), normed, rotated at its first token and stored in the
pooled plane (`lossless_qsa_pool_cache`, `lossless_qsa_pooled_store`), the open block's raw keys into the
tail (`lossless_qsa_tails_store`); each query's selection from its scores against its sequence's gathered
pooled keys (`lossless_qsa_pooled_gather`; `lossless_qsa_select` by position, scored 64 query columns at a
time) once it may have more complete blocks than the budget; attention over the cache with that
selection; the PLE history and the GDN state through the lanes' slots (`lossless_ple_conv_inject_slots`,
the Qwen3.5 slot forms). The sampler reads an FP32 copy of the logits. The certification's references are this route
or the design reference (3b), one scoring pass per window (`infernix-certify reference`; model quality
§6).

**Transcendental functions** (`src/ops/lossless/cr_math.cuh`): every one takes a binary32 argument, so
its correct rounding is decidable: the double evaluation is accepted when both ends of its error
interval round to the same binary32 (or BF16) value; otherwise the function is evaluated in
double-double arithmetic (relative error below 2^-94), far closer than any binary32 input's exact
value lies to a rounding midpoint (binary32 worst cases are within about 2^-60), and rounded from it.
rsqrt decides a near-midpoint case exactly instead (y > m exactly when 1 > m^2 v).

## 3b. The design reference (`--reference`)

`--lossless` computes the model at its trained precision. A configuration can also declare lossy steps
of its own: activation quantization that its artifact permits, and the KV-cache format the user
selects. `--reference` (`EngineOptions::reference`) is `--lossless` with each declared lossy step
applied exactly where the configuration declares it, and nothing else changed: the exact computation of
the configuration as designed: the certification's reference wherever the artifact declares activation
quantization or the KV format is coded (model quality §6), so a path's engine error is its distance from
this route and the declared steps' own cost (its distance from `--lossless`) is disclosed, never charged.

**Activation quantization.** Every input whose artifact Use permits a lower precision takes it, at every
width and phase (the reproducible route chooses by call width and runs Qwen3.5 decode at 16 bits; the
design does not depend on either):

| Weight, Use | Activation step | Value of the quantized activation |
|---|---|---|
| NVFP4 dense (Qwen3.5), `AllowA4`, activation divisor D | each 16 consecutive K elements of a token: `amax` = max abs value; `sf = fp32(fp32(D amax) / 6)`; `w = e4m3_rn_satfinite(sf)` (0 makes the block's codes 0); `d = e4m3(w)`; each code `c = e2m1_rn_satfinite(fp32(fp32(v D) / d))` (the reproducible route's divisor rule, `quantize_nvfp4_k16`) | `c d / D`, exactly |
| NVFP4 routed experts (Qwen3.8-Flash-Next W4A4: recipe A, Dense8; FP4-Local with `--expert-a4-activations`), the expert's FP32 `input_scale` g per gate, up and down | `s = e4m3_rn_satfinite(fp32(amax / fp32(6 g)))`, `d = fp32(e4m3(s) g)`, `c = e2m1_rn_satfinite(fp32(v / d))` (NVIDIA ModelOpt's rule, `quantize_a4_block`) | `c e4m3(s) g`, exactly |
| FP8 dense (Qwen3.5), `AllowA8` or `AllowA4` | per token over the projection's whole K: `s = fp32(amax / 448)` (0 for a zero column), `c = e4m3_rn_satfinite(fp32(v fp32(1 / s)))` (`fp8_a8.cu`) | `c s`, exactly |

An NVFP4 weight under `AllowA8`, the paired Q4/Q5 projections, the Qwen3.5 sparse MoE banks and the output
head stay 16-bit, as their only arithmetic (no route quantizes the head's input). The quantizer reads the exact route's BF16 input and the product is the
exact sum of the quantized values times the weights' exact values: the scale factors (`1 / D`, `g`, `s`)
are applied exactly, not as the FP32 products the fast kernels multiply by, and the result is rounded once
to the operation's dtype. Where the gate and up inputs of an expert carry different scales they are
quantized separately; the down projection quantizes its input h with its own scale.

**KV cache.** The route stores K and V through the selected `--kv-dtype`'s own encoder (the production
`kv_cache_append` codecs: each row's codes depend on that row alone), applied to the exactly computed
BF16 K and V, and attention reads the stored codes exactly: each row's decoded values (code times scale,
exact) are taken back from the codec's rotated basis exactly (`k = R^T y`, `R = H_256 / 16`), and the
scores, the softmax and the value sums are the lossless route's (`bf16` stores V as FP16, as the format
defines). The VQ formats keep their exact-window rule (a query reads a key's exact INT8 row inside the
window and its VQ value outside it, by position alone). Flash-Next's QSA selection reads its BF16 pooled
keys, which no KV format changes.

**Scope.** Every artifact and KV format of the reproducible route, in CausalScoring and in the forced
certification tool's references (`infernix-certify reference --design`, one scoring pass per window);
generation is not a use of the route. `bf16` KV rounds V to FP16 (`lossless_round_fp16`); every
other format stores the window's K and V through the production `kv_cache_append` (`lossless_kv_encode`: an
identity block table, so the planes hold production's bytes; the VQ formats' window rows for every key by the
append's own window-copy steps) and `lossless_codec_attention` reads them: a pack kernel decodes each row
exactly (codes times scales; for a rotated plane the exact H_256 / 16 in 128-bit integers) onto
`lossless_attention`'s grid integers (int32, one exponent per row), and the attention's scores, exponentials,
sums and rounding are `lossless_attention`'s own; a key row whose exact value needs more than 29 bits (a rotated FP8 row: an 18-bit
E4M3 code times an 11-bit scale, then the exact H_256 / 16) keeps the bits the shift to the grid would drop as a
second 29-bit limb, scored exactly beside the first, so such rows stay on the main path; a query against a row
needing more than the two limbs, and every query the error interval leaves undecided, takes an exact fallback
that decodes each key row once. On a 16-bit artifact with `bf16` KV the
design reference differs from `--lossless` only by V's FP16 storage. Its bits are a function of the request,
the artifact and the declared steps, so a reference survives rebuilds like `--lossless`'s.

## 4. Qualification

- Every operation bit-equal to an exact oracle (host big-integer or rational arithmetic, independent of
  the kernel), at the models' shapes, through every route the kernel has (the tensor-core path and the
  exact fallback alike).
- Whole blocks against an FP64 evaluation at the same rounding points, to within those points'
  roundings.
- Reference conformance: the route's whole FP32 logits at chosen positions
  (`infernix-certify capture`) equal, bit for bit, the publisher's transformers code evaluated in exact
  arithmetic with each operation rounded to its dtype (`tools/quality/shadow`). A mismatch is localized
  by `python -m tools.quality.shadow.bisect`: with `INFERNIX_LOSSLESS_DUMP_DIR` set the engine writes the
  BF16 residual stream after the embedding, each layer's mixer and each layer, and the tool compares every
  checkpoint with the publisher's code on the first tokens of the same sequence; the first checkpoint
  that differs names the operation to examine.
- Its bits depend on the request and the artifact alone, so the certification stores its references across builds
  and releases (`tools/quality/certify.py`'s reference store) and a release run re-scores one window of
  each to confirm that no bit changed; any change to what this document defines needs the owner's
  approval and a full certification.

## 5. The noise twin

**Attribution.** `EngineOptions::lossless_noise_units` (`ops::LosslessNoiseUnit`: elementwise, attention,
Flash-Next's own operations, the delta rule, MoE, RoPE, linear, GGUF) confines the noise to chosen groups of
operations, and `lossless_noise_scale` thins it: every rounding moves from its nearest neighbour to the other
with that share of the twin's probability (`infernix-certify prefill --lossless --noise-seed N --noise-units
LIST --noise-scale S`). With the noise in one group at a time, the twin's KL splits by group; with all groups
at falling strength, the strength at which it equals a shipped route's KL says how noisy that route is.

`EngineOptions::lossless_noise_seed` (`infernix-certify prefill --lossless --noise-seed N`, CausalScoring
only) makes every correctly rounded result of the route a stochastic rounding of the same exact value:
up with probability equal to the discarded fraction of the result's unit in the last place
(`src/ops/lossless/noise.cuh`; the draw is keyed by the seed and the value being rounded, so a run is
deterministic and equal exact values round alike). It is an ideal implementation of the model at its
dtypes, off by at most one unit in the last place at every rounding and by nothing else. Its distance
from the route is the model's own sensitivity to rounding, the noise floor of every route measured against it.
`infernix_lossless_linear_test` checks it against the exact oracle: every output one of its exact
value's two neighbours, as often away from zero as the discarded fractions predict, the same bits for
one seed.
