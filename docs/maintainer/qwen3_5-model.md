# Qwen3.5 model reference

This reference describes the Qwen3.5 Dense and MoE mathematics implemented by Infernix, including
Text, MTP, Vision and their state semantics. The Qwen3.6 and Qwen3.8 releases used by the official
artifacts are instances of these architectures. Their release names do not choose another graph.

[Engine architecture](engine-architecture.md) defines ownership and execution;
[artifact container](artifact-container.md) defines stored objects and bindings;
[weight conversion](../weight-conversion.md) explains source mapping and recipes.

## Architecture and configuration

The architecture registry recognizes these Text config pairs:

| `architectures[0]` | `model_type` | Feed-forward block |
|---|---|---|
| `Qwen3_5ForCausalLM` | `qwen3_5_text` | Dense SwiGLU |
| `Qwen3_5MoeForCausalLM` | `qwen3_5_moe_text` | Routed SwiGLU experts and a gated shared expert |

Code fixes the pre-norm residual order, gated GQA, zero-centered Text norms, GDN recurrence,
interleaved partial MRoPE, FFN mathematics and component relationships. Config supplies instance
dimensions and layer choices. It does not describe a graph of arbitrary operators.

[`config.cpp`](../../src/models/qwen3_5/config.cpp) defines the exact fields and validation.
Their responsibilities are:

| Config facts | Meaning |
|---|---|
| `hidden_size`, `vocab_size`, `num_hidden_layers`, `max_position_embeddings` | Mathematical extents |
| `layer_types` | One `full_attention` or `linear_attention` entry per Text block |
| `tie_word_embeddings`, `rms_norm_eps` | Training relationship and normalization constant |
| `num_attention_heads`, `num_key_value_heads`, `head_dim` | GQA dimensions; also used by MTP |
| `rope_parameters` | Theta, partial rotary factor and three interleaved MRoPE sections |
| `linear_num_*_heads`, `linear_*_head_dim`, `linear_conv_kernel_dim` | GDN heads, state dimensions and convolution width |
| `intermediate_size` | Dense FFN width |
| `num_experts`, `num_experts_per_tok`, `moe_intermediate_size`, `shared_expert_intermediate_size` | MoE dimensions |

Projection widths, compact attention/GDN layer indices, rotary dimensions and parameter shapes are
derived. The converter expands source shorthand such as a full-attention interval into the explicit
`layer_types` array. Source dtype, quantization metadata, checkpoint name and training-only fields
do not become mathematical config.

Text is required. Optional components target `text` and supply their private data:

- Vision supplies tower dimensions and patch/merge geometry. Its output width comes from Text.
- MTP identifies `Qwen3_5MTP` or `Qwen3_5MoeMTP`. The single-layer predictor inherits Text's
  attention, RoPE and FFN geometry and shares its embedding/output semantics.
- DFlash/DFlash2 supply private draft dimensions, attention pattern, conditioning taps and their
  algorithm parameters. See [DFlash](dflash.md).

Only selected components are semantically bound and materialized. A model's weights and config
remain immutable after loading; request capacity, concurrency, KV codecs and graph profiles are
runtime choices.

### Current official geometries

These are useful workload points, rather than a second config registry:

| Quantity | 27B Dense | 35B-A3B MoE |
|---|---:|---:|
| Hidden width H | 5120 | 2048 |
| Text layers | 64 | 40 |
| Full attention / GDN layers | 16 / 48 | 10 / 30 |
| Q heads / KV heads / head dimension | 24 / 4 / 256 | 16 / 2 / 256 |
| GDN key heads / value heads | 16 / 48 | 16 / 32 |
| GDN key/value head dimension | 128 / 128 | 128 / 128 |
| GDN convolution width | 4 | 4 |
| Dense intermediate width | 17408 | — |
| Routed experts / selected per token | — | 256 / 8 |
| Routed / shared intermediate width | — | 512 / 512 |
| Embedding/output matrix rows | 248320 | 248320 |
| Position capacity | 262144 | 262144 |

Both use `rms_norm_eps=1e-6`, `rope_theta=1e7`, `partial_rotary_factor=0.25` and
`mrope_section=[11,11,10]`. Their full-attention layers occur at indices `3,7,...`; that pattern is
stored in config. Both have untied embedding and output matrices. Qwen3.6-27B and Qwen3.8-27B share
this Dense geometry, while training results, representations and frontend semantics can differ.

Config parsing accepts structural variation. Actual execution remains bounded by implemented
Ops: the current fused input projections have explicit Dense/MoE shapes, GDN uses the implemented
128-dimensional recurrence, and the native sparse-MoE path uses 256 experts with top-8 routing.
The current multimodal RoPE entry requires the implemented interleaved axis mapping. These checks
belong to the relevant consumers; parsing a config is not a promise to execute every size.

## Logical parameters and physical bindings

[`load/`](../../src/models/qwen3_5/load/) expands the selected config into parameter names, shapes
and mathematical input Uses. The converter's
[`qwen3_5.py`](../../tools/convert/qwen3_5.py) maps source tensors into the same roles.

The main namespaces are `text/`, `vision/`, `mtp/`, `dflash/`, `dflash2/` and the optional
`proposal/`. Text block `i` contains `text/layers/i/attention/...` or `.../gdn/...`, plus its Dense
`mlp/...` or `moe/...` parameters. A projection such as `attention/query` has logical shape
`[num_attention_heads * head_dim, H]` and a Use at `text/layers/i/mixer_input`.

Bindings describe the logical values in physical parents. They can use complete objects,
consecutive regions or several ordered parts. The source `q_proj` stores each head's Q rows next
to that head's gate rows; source mapping extracts those rows into separate logical projections.
A simple split at the middle of the source matrix would change the model.

The Dense groupwise attention recipe packs Q/K into Q4 and gate/V into Q5. A supported FP8 or NVFP4
recipe can pack their union into one parent. The same four logical roles then become either the
two-weight or one-weight native fused call. Dense gate/up and MoE expert banks follow their finite
packing and execution rules. [Storage layouts](storage-layouts.md#12-logical-views-and-native-operands)
describes the difference between logical views and native operand restrictions.

`tie_word_embeddings` records a training relationship; bindings choose whether physical data is
shared. MTP and draft components reference Text's embedding/head rather than introduce private
copies by default. Shared weights keep independent Uses and activation permissions. The stored
NVFP4 weight divisor belongs to its parent; an activation divisor belongs to a mathematical input.

## Text block mathematics

In the formulas below, activations are written with feature dimensions first and token columns
last. H is hidden width, D is an attention head dimension, and T is the current token extent.
The two Text norm forms are:

```text
offset_rmsnorm(x,w) = (1 + w) * x / sqrt(mean(x^2) + eps)
plain_rmsnorm(x,w)  =       w  * x / sqrt(mean(x^2) + eps)
```

Text/MTP input, post-attention, final, stem and Q/K norms use the offset form. GDN's internal
output norm uses the plain form. Vision uses LayerNorm with weight and bias.

Every Text block follows:

```text
h = offset_rmsnorm(x, input_norm)
x = x + mixer(h)
h = offset_rmsnorm(x, post_attention_norm)
x = x + ffn(h)
```

The mixer is gated GQA or GDN according to `layer_types`. Dense FFN is
`down(SiLU(gate(h)) * up(h))`; MoE is defined below. Text/MTP projections and GDN convolution are
bias-free. Fusion may combine projection, activation, normalization and residual work while
preserving the declared mathematical and observable precision boundaries.

### Gated full attention

```text
q    = query(h)
gate = gate_projection(h)
k    = key(h)
v    = value(h)

q = offset_rmsnorm(q, query_norm)
k = offset_rmsnorm(k, key_norm)
q, k = partial_interleaved_mrope(q, k)
a = causal_gqa(q, k, v, scale=1/sqrt(D))
y = output(a * sigmoid(gate))
```

Q and gate have one D-dimensional vector per query head. K and V have one per KV head, with each
KV head serving a consecutive group of query heads. The gate is neither normalized nor rotated.
The logical cached values are normalized/rotated K and projected V. Prefill appends a chunk and
uses causal attention; decode appends one column and attends over the visible prefix.

Paging and KV quantization are storage choices. Their contracts are
[`kv_cache_append.h`](../../include/infernix/ops/kv_cache_append.h),
[`softmax_attention.h`](../../include/infernix/ops/softmax_attention.h) and
[paged KV cache](paged-kv-cache.md).

### Gated DeltaNet

Normalized input produces Q, K, V, output gate Z and per-value-head A/B projections. Only Q/K/V
pass through depthwise causal convolution and SiLU. The convolution retains the preceding
`linear_conv_kernel_dim - 1` projected columns. GDN has no RoPE or position-ID input.

For value head j, use Q/K head `j // (value_heads / key_heads)`. Let the recurrent state be
`S[Dv,Dk]`. One token performs:

```text
g     = -exp(A_log) * softplus(a + dt_bias)
alpha = exp(g)
beta  = sigmoid(b)
k     = k / sqrt(sum(k^2) + 1e-6)
q     = q / sqrt(sum(q^2) + 1e-6)
Sbar  = alpha * S
u     = beta * (v - Sbar @ k)
S     = Sbar + u outer k
o     = (S @ q) * (1/sqrt(Dk))
y     = out_projection(plain_rmsnorm(o, gdn_norm) * SiLU(z))
```

The delta error uses the already-decayed state. GDN controls and recurrent state are FP32. The Op
receives raw BF16 q/k; normalization precision and the chunked/recurrent implementation are private
to its qualified route. Large prefill can use parallel state passing, while decode uses recurrence.

Ordinary decode updates the lane's current state. Speculative verification leaves persistent state
unchanged and writes convolution and recurrent records. After output resolution, one Fold applies
the committed prefix of that recorded physical block. The projection/convolution/record fusion
remains in the model's finite execution choices. [ReplaySSM](replayssm-gdn.md) defines the record
and commit semantics, including their numerical boundary.

### Sparse MoE

For token h, with E routed experts and K selected experts:

```text
router_logits = bf16(W_router h)
router_probs = softmax(router_logits)
(values, ids) = topk(router_probs, K)
routing_weight = values / sum(values)

y_e = W_down[e](SiLU(W_gate[e] h) * (W_up[e] h))
y_routed = sum(routing_weight[e] * y_e for e in ids)
shared = W_shared_down(SiLU(W_shared_gate h) * (W_shared_up h))
y = y_routed + sigmoid(W_shared_score h) * shared
```

The router logits are the router projection rounded once to BF16, as the model computes them
(Transformers' `Qwen3_5MoeSparseMoeBlock`): selection, the routing weights and the shared expert's
sigmoid read those values, and ties at that precision go to the lower expert id. Until 2026-10-22
every route ranked FP32 logits; against the lossless reference the BF16 logits moved the 35B-A3B's
reproducible prefill 11 % closer (mean KL 4.62e-3 to 4.10e-3, BF16 KV; 4.80e-3 to 4.19e-3, INT8 KV).
The shared expert always executes and is outside top-k. Logical expert id e identifies router row e
and the corresponding expert matrices. Physical bank ordering and fused dispatch preserve that
relationship. Router loss coefficients and training-only outputs add no inference term.

## Prefill, decode and MTP

Text prefill gathers embedding columns, replaces media placeholders with Vision outputs, and runs
the configured blocks in chunks. It preserves state and positions across chunks. Final offset
RMSNorm and the full output head produce the target distribution as FP32 logits, the head's sums
without a BF16 rounding: the sampler, greedy argmax, draft acceptance, constraints and scoring read
them on every route (since 2026-10-24; `--logits-dtype bf16` rounds each once to BF16 instead, for
comparisons). BF16 logits turned near-equal tokens into exact ties, which
greedy selection breaks by token id: against the lossless reference (64 IQS-1 windows, BF16 KV) the
groupwise Qwen3.8-27B's default prefill top-1 flips were 1.53x the noise twin's with BF16 logits and
0.74x with FP32 (median KL 1.40x to 0.53x), the 35B-A3B's 1.06x and 0.94x. The drafters' proposal
logits stay BF16. Ordinary decode repeats this
schedule for one new token. CUDA Graph replay uses the same selected model implementation.

A generation prompt reads the last layer's output only at each prefill window's last column: the
head on the prompt's last window, the continuation hidden that prefix-cache snapshots and the
zero-suffix sample keep at every pass end, and a rewrite checkpoint. So its last layer computes
keys, values (and a GDN layer's state) over every column but the mixer's output projection and the
MLP over the window's last few columns, and the final norm over its last one (since 2026-11-08,
candidates YC-01; TensorFold `3d4f6aa360`). The width is twenty columns, at which every dense
projection keeps its prompt route's activation precision: 16-bit ones (the groupwise 27B's
`A16Only`, NVFP4 `AllowA8`, which NVFP4 runs at 16 bits at every width), NVFP4 `AllowA4` ones (the
NVIDIA NVFP4 27B's MLP: 4-bit from 8 columns in linear_add) and row-scaled FP8 `AllowA8` ones (the
official NVFP4 27B keeps its last layer in FP8: 8-bit from 20 columns at K = 17,408). The 4-bit
codes use the artifact's fixed activation divisor and the 8-bit ones a per-token scale, so a
column's codes do not depend on the others. It stays off where every column's final hidden is
read: MTP prompt preparation, causal scoring, a drafter tapping the last layer (the 27B's DFlash2
taps layers 5, 19, 33, 47 and 61 of 64); where a projection has a residual-term profile; and where
the FFN is offloaded. A first version trimmed to one column, on the decode routes' 16-bit
activations: the official artifact's `/v1/decide` readouts then differed from CausalScoring by
0.209 logprob, past the readout real test's 0.2, and the owner kept the prompt routes' precision. A
second forced NVFP4's 4-bit route at one column; linear_add's A4 tiles gave wrong sums below eight
columns (its op test). A third, at eight columns, left the FP8 layer out and never engaged there.

Where the NVFP4 MLP runs the two-term A4 residual profile (`AllowA4Residual`: verification and batched decode
rounds of 5-96 columns, `batched-a4-residual-decode`), its input norm and both projections are one Op,
`ops::dense_mlp_fused_rmsnorm_nvfp4` (since 2026-11-08, candidates SI-02+): the norm kernel encodes the gate/up
projection's activation planes from its own BF16 row, the gate/up GEMM's SwiGLU epilogue encodes the down
projection's planes from its staged BF16 product, and the down GEMM reads them. Three launches replace five
(rmsnorm, two quantizers, two GEMMs) with the same bytes: each producer rounds to BF16 where the separate Ops stored
BF16 and encodes it as their quantizer does. The epilogue encodes each 16-value group with four lanes (a group's
exact divisions are its latency; one lane a group added 2.4 us to the GEMM, four add 0.8-1.7 us), the norm with one
lane a group (320 groups over 256 lanes: two rounds, where four lanes a group take five and measured 0.45 us
slower). The NVIDIA NVFP4 27B's MLP takes 100.7 against 105.9 us a layer in DFlash2 rounds (Nsight Systems), and
its DFlash2 decode runs +0.26 to +0.31 % faster (INT8 KV, -p 512 -n 256, ABBA x3; 2026-11-09).

Prompt calls of 129 columns or more (256 until 2026-11-10) where the MLP runs one-term A4 activations (`AllowA4`: both NVFP4 27B
artifacts' prompt routes) take the same Op's prompt profile (since 2026-11-09): the norm kernel writes the gate/up
projection's one plane with its scales in the TMA tile's 256-token tiles, the gate/up projection's TMA GEMM encodes
its SwiGLU product as the down projection's plane (scales row-major to 512 columns, where the down projection runs
its MMA tiles, 256-token tiles above, where it runs its TMA tile; the padding tokens' scales zeroed as the quantizer
zeroes them), and the down projection reads it. The bytes are the separate Ops' (op test at 256-2,048 columns; greedy
ids on the NVIDIA and official artifacts at ~2,000 and ~20,000-token prompts identical to the unfused build). Before,
a 4,096-token prompt spent 4.6 ms in the product's quantizer and 0.8 ms in the input's (63 layers, of 309 ms of
kernels; Nsight Systems). Prefill (INT8 KV, `infernix_bench -p N -n 1`, ABBA x3, 2026-11-09): NVIDIA NVFP4 pp1024
+0.09 to +0.24 %, pp4096 +1.15 to +1.25 %, pp16384 +1.14 to +1.21 %; official NVFP4 pp4096 +0.98 to +1.09 %.

Prompt widths that are not tile multiples (2026-11-10). The 27B's prefill time was a staircase: about 20 us a token
inside a step and +5 to +7 ms where a width crossed 128, 256, 512 or 1,024 columns, because a partly filled last
token tile ran a full tile's tensor-core work in every warp (the gate/up projection is 272 CTAs a token tile, 1.6
waves of one CTA an SM). Three changes, official NVFP4 artifact, INT8 KV, baseline binaries against new, A B B A
x2:

- The NVFP4 A4 and FP8 A8 TMA kernels skip the tensor-core steps of 16-token fragments past the call. Their sums
  were never stored, and each stored output keeps its K order, so no bit changes. The NVFP4 kernel has one
  instance of its K loop per live count (`with_static_count`, `ops/common/mma.cuh`); the FP8 kernel skips only
  warps with no live column. Two forms that lost: a run-time bound read inside the unrolled loops (the
  accumulators left the registers: partly filled FP8 launches took 4 to 5.6 times as long), and FP8 warps with
  some live fragments beside full ones (the 96x256 tile 17 % slower at 257-272 columns, the 192x128 tile 6 %
  slower at 145; cause not found).
- The fused MLP's prompt profile and the standalone gate/up's TMA route start at 129 columns: one partly filled
  256-token tile instead of two 128-token MMA tiles (gate/up 142-153 -> 101-109 us at 129-255 columns). A cut at
  97 or 65 columns measured 5-7 % slower at 72-120 tokens. Same bits.
- FP8 tables: attention input at 513-576 columns on three 192x128 tiles (152-154 against 167-172 us, same bits);
  GDN input at 257-320 on 64x256 tiles and at 385-480 on 96x256 tiles, and the K = 6144 output at 641-768 on the
  128x256 tile. Those three replace launches that had a split-K tail, so the tail's columns take the unsplit K
  order there: their bits change at those widths, on the default route and on `--reproducible` prefill (which
  stays deterministic and request-invariant).

Prefill: 129-255 tokens -7.0 to -9.9 %, 264-1,100 tokens -2.8 to -7.0 % (264 -3.8 %, 520 -3.7 %, 776 -6.5 %,
1,032 -5.6 %), 40 / 72 / 104 tokens level (+0.5 / -0.5 / -0.6 %), 2,048 / 3,584 / 4,096 / 8,192 level (+0.4 / -0.1 /
0.0 / 0.0 %); `--reproducible`: 136 -9.2 %, 264-1,032 -3.9 to -6.4 %, 4,096 -2.0 %. Decode unchanged (81.2 / 81.3
and 72.6 / 72.6 tok/s). Greedy text against the previous build: identical at 196, 382 and 5,856 prompt tokens on
the default route and at 196 and 382 on `--reproducible`; different at 729 tokens on both routes and at 5,856 on
`--reproducible`, and identical there too with the three bit-changing table entries taken out.

The N = 5120 down projection at 129-512 columns (2026-11-11). Its launches there are one wave (80-240 CTAs on 170
SMs), so a call's time is one CTA's K loop of 68 tiles, and the table's one-stage 128x128 tile at two CTAs an SM
(193-384 columns) was the slow choice: with every tile forced in turn, the 64x128 two-stage tile takes 72-75
against 118-122 us at 193-256 columns and the 128x128 two-stage tile 100-104 against 123-130 us at 257-384 (K =
17408; K = 6144 the same shape: 32-34 against 50, 44-46 against 52-55); a 64x64 tile at two CTAs an SM takes 65-67
against 69 us at 129-176. Smaller tiles at three or four CTAs an SM, K blocks of 128 and the TMA tile below 513
columns measured slower. The seven layers whose MLP is FP8 take two 96x256 tiles at 129-160 columns and one
192x128 tile at 161-192 for the gate/up (146 against 164-167 us, 153 against 178). All of these keep the bits (one
K order per output, no split-K tail involved); greedy text equals the previous build's at 196, ~260, 382, 729 and
5,856 prompt tokens on both routes. Prefill against the previous build (official NVFP4 artifact, INT8 KV, A B B A
x2): 200-255 tokens -9.7 to -10.6 %, 264-328 -4.4 to -4.9 %, 136-177 -0.2 to -2.1 %, other widths level;
`--reproducible`: 200 -9.5 %, 264 -4.4 %. Measured and not taken: FP8 entries that would change a split-K tail
(the FP8 MLP's gate/up at 257-288, its down at 129-240: seven layers, 0.1-0.3 % of a prefill each); small tiles
for the K = 6144 FP8 output (none faster); a launch order that runs a row group's partly filled token tile after
its full tiles (gate/up -2 to -4 % at 257-384 columns, level end to end: the partly filled tile's time is its 40
load-and-barrier round trips, not its weights coming from memory).

The NVFP4 prompt routes' weight tensor maps, and the all-NVFP4 recipe's input projections (2026-11-11). Two
changes, both with the same bits:

- The weight-code tensor maps of the prompt TMA schedules (fused gate/up, down, NVFP4 attention input and GDN
  input) ask for L2 promotion at 128 B (`kNvfp4PromptWeightPromotion`), as the plain [34816,5120] route has since
  2026-09-15. Fused gate/up: 129 / 192 / 256 columns -8.3 / -5.7 / -6.0 %, 264-3,584 -0.8 to -2.2 %; down 0 to
  -2 %. 256 B measured the same and 64 B nothing; the activation-code and scale maps nothing; the residual-plane
  schedule of verification rounds was level and stays without it. The same request on the FP8 A8 TMA maps (weight
  or activation; 64, 128 or 256 B) measured within +-0.5 %: not taken.
- An artifact whose attention and GDN input projections are NVFP4 (the all-NVFP4 recipe; the official, NVIDIA and
  uncensored artifacts keep FP8 there) ran them on MMA tiles to 511 columns. Their TMA routes now start at 129.
  GDN input: 68 -> 44-46 us at 129-192 columns, 62-64 -> 46-47 at 193-256, 90-97 -> 74-80 at 257-384, 113-119 ->
  78-91 at 385-511. Attention input: the 256-token tile to 256 columns and from 513 (50-62 -> 40-46 us at 129-256;
  90-102 against 101-125 us at 513-768), the 128-token tile at 257-512 (the 256-token tile is slower at 257-384:
  72-77 against 66-68 us; at 512 itself 86.6 against 80.5). The bytes are the MMA routes':
  `attn_input_proj_bench` and `gdn_input_proj_bench` print an FNV-1a of their outputs (`output_fnv`), equal for
  the old and the new routes at 26 widths from 64 to 1,100.

Prefill against the previous build (INT8 KV, A B B A x2, medians of four runs: about ten of the 600 runs, in both
arms and all at 40-328 tokens, were 1.0-2.7 ms off their arm's other three). All-NVFP4 artifact (both changes): 129-255 tokens -6.3 to -11.6 %,
264-328 -3.1 to -3.7 %, 392-511 -5.1 to -5.8 %, 520-1,100 -1.0 to -2.6 %, 2,048-8,192 -1.1 to -1.6 %, 40 / 104
+0.1 / +0.2 %; `--reproducible` (two runs): 136 -11.5 %, 200 -6.4 %, 264 -3.8 %, 392 -5.4 %, 520-4,096 -1.1 to
-2.8 %. Official artifact (the promotion only): 129-255 -1.2 to -3.0 %, 264-511 0.0 to -0.8 %, 520-8,192 -0.7 to
-1.6 %, 40 / 104 +0.2 / -0.1 %; `--reproducible` 136 / 200 -2.9 / -1.9 %, 264-4,096 -0.2 to -1.0 %. NVIDIA
artifact: 129-255 -1.7 to -3.3 %, 264-8,192 0.0 to -1.0 %, 40 / 104 0.0 / -0.2 %. Decode unchanged on all three
(99.6, 82.8, 86.1 tok/s). Greedy text equals the previous build's at four prompt lengths on each artifact and on
`--reproducible`.

MTP is a one-layer predictor conditioned on a final-normalized target hidden state h_t and the
following token x_(t+1):

```text
e = offset_rmsnorm(composed_embedding(x_(t+1)), pre_fc_norm_embedding)
h = offset_rmsnorm(h_t, pre_fc_norm_hidden)
u = input_projection(concat(e, h))
u = one_gated_attention_block_with_target_ffn(u)
draft_hidden = offset_rmsnorm(u, mtp_final_norm)
draft_logits = target_output_head(draft_hidden)
```

The stem concatenates embedding first, hidden second. MTP has private stem, block, final norm and
KV state. Its attention and Dense/MoE geometry come from Text. It can recursively propose several
tokens using the previous MTP hidden output.

For prompt tokens x_0 through x_n and the target's first generated token x_(n+1), MTP prefill uses:

```text
token inputs    = [x_1, x_2, ..., x_n, x_(n+1)]
hidden inputs   = [h_0, h_1, ..., h_(n-1), h_n]
position inputs = [p_0, p_1, ..., p_(n-1), p_n]
```

Tokens shift; hidden states and positions do not. The whole aligned sequence populates MTP KV.
Shifted media placeholders use their corresponding composed Vision embedding columns.
[`mtp_alignment.h`](../../src/models/qwen3_5/program/speculative/mtp_alignment.h) and the prefill
implementation own chunk-boundary and continuation alignment.

An optional proposal head supplies an indexed vocabulary subset for draft prediction. Its row map
converts proposal rows to actual token IDs. Full target verification continues to use the full
output head. Backend selection, draft width and proposal-head choice are fixed at startup.

## Vision and multimodal positions

The current native processor uses 16×16 spatial patches, pairs of frames, and 2×2 spatial merge.
It consumes bytes already acquired by the CLI or server, decodes images/video frames (JPEG images
through libjpeg with Pillow's settings, and lossy WebP through FFmpeg's VP8 planes with libwebp's
fancy upsampling and YUV conversion, so their pixels equal the Hugging Face processor's input;
lossless WebP and GIF through FFmpeg and PNG through FFmpeg or the Windows build's native decoder,
also equal to Pillow's; other images and every video through FFmpeg, chroma interpolated bilinearly).
An image's alpha is dropped, as `load_image` drops it and most engines do (llama.cpp, Ollama,
LMDeploy, mlx-vlm, TGI; vLLM, SGLang and TensorRT-LLM composite over a background instead); video
frames with alpha are composited over white. It then resizes to
merge-aligned dimensions, normalizes RGB and packs BF16 patch rows. Each row contains
`3 * 2 * 16 * 16 = 1536` values in the defined channel/temporal/spatial order. Images repeat their
frame to supply the temporal pair.

The official Vision towers have depth 27, hidden width 1152, intermediate width 4304 and 16 heads
of dimension 72. The learned position table is a 48×48 grid. Patch projection and interpolated
position addition precede the transformer stack. Each block performs:

```text
h = LayerNorm(x)
q, k, v = qkv_projection(h) + bias
q, k = vision_rope(q, k, two_dimensional_patch_positions)
a = segmented_attention(q, k, v)
x = x + output_projection(a) + bias
h = LayerNorm(x)
x = x + fc2(GELU_tanh(fc1(h) + bias)) + bias
```

Attention segments separate media/frame grids. The merger normalizes, groups 2×2 patches into
width 4608, and computes `fc2(GELU_exact(fc1(merged) + bias)) + bias`, producing Text-width columns.
Those columns replace matching placeholder embeddings. Vision has no autoregressive state.

Frontend admission uses aggregate media bytes, pixels, patch/token budgets and Engine context
capacity. The aggregate limit is 131,072 raw patches / 32,768 merged tokens; a single tower item
uses at most 16,384 merged tokens, additionally bounded by context. Program executes items
sequentially and reuses their output handoff after scattering the previous item.

The prepared prompt stores axis-major `[3,T]` positions in temporal, height, width order:

- Text tokens advance all three axes together.
- Media placeholders use their merged grid coordinates; following text resumes after the maximum
  multimodal coordinate.
- Video timestamps are rendered into prompt text. Position construction uses the prepared media
  grids and token runs.
- `rope_delta = next_multimodal_position - token_count` supplies the offset for subsequent decode.

For the official Text geometry, 64 of each 256-dimensional Q/K head rotate, using 32 frequency
pairs assigned to interleaved sections `[11,11,10]`. Text-only positions reduce to ordinary partial
RoPE because the three axes are equal.

## Vocabulary and frontend resources

Keep three row domains distinct: config `vocab_size` is the embedding/output matrix extent; the
tokenizer defines addressable token IDs; an optional proposal map defines its own shortlist rows.
The official tokenizer has 248,077 addressable IDs within 248,320 matrix rows. Remaining matrix
rows are not assumed zero and do not become public tokens.

Token spelling, special-token roles and EOS are derived from loaded resources. The frontend checks
the tokenizer's supported BPE/normalization semantics and token-domain agreement. The Frontend
compiles the artifact or startup-selected Jinja template and derives exact reuse boundaries from
its rendered token sequence. Mode sampling
presets are architecture-owned, with explicit application/request overrides. Resource loading
does not infer execution identity from tokenizer filenames, release names or sampling values.

## State and numerical boundaries

Each StateImage contains the Text GDN state and continuation hidden. Per GDN layer, recurrent
state has one `[Dv,Dk]` matrix per value head, and convolution history has
`(2 * key_width + value_width) * (kernel_width - 1)` values. Program sizes Device/Host StateImage
capacity under the [context resource contract](resource-scheduling-and-context-cache.md#capacity).
ReplaySSM records are separate pending-round scratch when speculation is enabled.

Text GQA KV grows with visible context. Selected MTP owns separate KV; DFlash backends own the
state described in their reference. Continuation hidden values and prefix checkpoints belong to
Program. Workspace, storage capacities and graph buffers are derived from actual parameters and
startup options by the planners described in [Engine architecture](engine-architecture.md).

Speculation verifies the candidate sequence against processed target distributions, commits the
accepted prefix and continuation state, then publishes output. Greedy acceptance compares a
proposal token with the target argmax for that verify column. Sampling uses the defined proposal
acceptance and correction distribution. This does not impose token or logits equality between
different quantization, prefill or kernel paths.

Each floating-point Op is qualified against its independent mathematical oracle over represented
public inputs. Packed weights are decoded with their stored scales. Exact transforms use exact
oracles; private accumulation, staging and reduction order use the Op's numerical criteria.
[Op development](op-development.md) owns that qualification contract.
