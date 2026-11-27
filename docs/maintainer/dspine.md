# DSpine

DSpine ([Tencent](https://github.com/Tencent/DSpine), "Draft in Parallel, Condition Through
Depth: Adjacent Causal Injection for Speculative Decoding",
[arXiv 2609.36173](https://arxiv.org/abs/2609.36173), Apache-2.0) is a masked-block drafter of the
[DFlash](dflash.md) family. Its layers attend a ring of the target's tapped hidden states as
DFlash2's do, but causally inside the drafted block. After every layer a code bridge writes into each
block column a message from the column before it, so the drafter carries the predicted predecessor's
features through its depth. A serial refinement of the last write then chains the head's candidates
into drafts.

Infernix runs a DSpine drafter as its own artifact beside a Qwen3.5-family model:
`--spec dspine --draft-model PATH`. It is never chosen by `--spec auto`, and `--spec dflash2` keeps
the artifact's bundled DFlash2 drafter as before. The engine refuses a drafter whose hidden width,
vocabulary or target layer count differs from the model's. Tencent publishes one drafter, for
Qwen3.8-27B. Infernix's converter writes it from Tencent's checkpoint (`--components dspine`, recipes
`qwen3_8_27b_dspine_q8`, `_nvfp4_mlp` and `_nvfp4`: [weight conversion](../weight-conversion.md)), and
[WaveCut's NInfer v3 conversion](https://huggingface.co/WaveCut/Qwen3.8-27B-DSpine-b16-NInfer-v3) (recipe
`dspine_q8` of NInfer-all's converter, which `qwen3_8_27b_dspine_q8` reproduces byte for byte but for the
sender table's rounding) loads as it is.

The port follows [NInfer-all](https://github.com/iamwavecut/ninfer-all) (PR #341 and #365, Apache-2.0):
its bridge and refinement Ops and their oracle tests are NInfer-all's, renamed; the drafter's
execution, loading and planning are Infernix's own, over the DFlash2 frame.

| Code | Owns |
|---|---|
| [`models/qwen3_5/config.cpp`](../../src/models/qwen3_5/config.cpp) `parse_dspine_config` | the drafter artifact's config and its refusals |
| [`models/qwen3_5/load/dspine.cpp`](../../src/models/qwen3_5/load/dspine.cpp), [`load.cpp`](../../src/models/qwen3_5/load.cpp) | the bridge's bindings; the drafter artifact's own materialization beside the model's |
| [`models/qwen3_5/execution/parameters.cpp`](../../src/models/qwen3_5/execution/parameters.cpp) | the bridge operands and the head-write table |
| [`models/qwen3_5/execution/draft.cpp`](../../src/models/qwen3_5/execution/draft.cpp) `propose_dspine_rows` | the drafted block, the bridge writes, the candidates and the refinement |
| [`ops/dspine.h`](../../include/infernix/ops/dspine.h) | the bridge and refinement Ops |

## The artifact

The drafter artifact's only component is `dspine`, architecture `DSpineDraftModel`; no component
declares a target (a standalone artifact: [artifact container](artifact-container.md)). Its layers
bind as a DFlash drafter's do under the component `dspine` (`dspine/feature_projection`,
`dspine/layers/i/...`; each layer's context K/V shares the data of its own K/V), and its bridge as:

| Parameter | Shape | Source |
|---|---|---|
| `dspine/bridge/projection` | `[3R, C]` | the gate, predecessor-gate and value weights, one below the other |
| `dspine/bridge/gate_bias` | `[R]` | the gate's bias |
| `dspine/bridge/layers/i/read` | `[C, H]` | `early_read_weights[i]`, and `read_weight` for the last layer |
| `dspine/bridge/layers/i/write` | `[H, R]` | `feature_write_weights[i]` |
| `dspine/bridge/head_write` | `[R, H]` | the last layer's write, transposed |
| `dspine/bridge/sender_table` | `[V, 2R]` | each token's predecessor-gate and value rows |

The token embedding and the output head stay the model's. The drafter reads the head through the
model artifact's DFlash2 Use of it (`dflash2/final_hidden`) where the artifact declares one, else
through the text's.

Tencent's Qwen3.8-27B drafter (revision `260a6c23`, the first whose config names `DSpineDraftModel`, which the
converter keys on; earlier revisions say `DFlashDraftModel` over the same weights): hidden 5120, vocabulary 248320, 5 layers at a
sliding window of 8192, intermediate 17408, 32 query and 8 KV heads of 128, taps `[1,16,31,46,61]`
of 64, a block of 16 (the anchor and 15 masks), mask token 248070, code width 1024, rank 512, norm
epsilon `1e-6`, RoPE theta `1e7`. WaveCut's `dspine_q8` file stores the layers' projections Q8 (groups
of 32, FP16 scales) and the bridge BF16: 2.27 GiB. Infernix's recommended conversion,
`qwen3_8_27b_dspine_nvfp4`, stores the attention output, MLP and feature projection NVFP4 (16-bit
activations) and keeps the fused query/key/value Q8, which `attn_input_proj`'s drafter form and the fused
context materialization read: 1.53 GiB, the fastest of the three formats measured (2026-11-26).

The config accepts 32 query and 8 KV heads of 128, 5 or 6 layers (the fused context
materialization), a rank that is a multiple of 32 up to 512, a block of 2 to 16, a trained window of
2048, 4096 or 8192, norm epsilon `1e-6` and RoPE theta `1e7`, and the head's top 16; it refuses other
values.

## Conditioning and context

The drafter conditions on the target as DFlash2 does: the tapped layers' residuals side by side,
projected and normalized, and each layer's context K and V projected from them into a ring of
`sliding_window` slots per drafter layer and state image (`ops::context_kv_materialize`, which takes
rings of 2048, 4096 and 8192 slots). The ring always holds the trained window, so the drafter's
attention sees exactly what it was trained on:

```text
ring bytes = layers * 2 (K, V) * 2 bytes * KV heads * head_dim * window
           = 5 * 2 * 2 * 8 * 128 * 8192 = 160 MiB per lane and per state image (DFlash2: 40 MiB)
```

A single request's prompt block wider than 2048 columns commits in slices of 2048.

## The drafted block

The block holds the anchor `a` at the frontier F and K masks at `F+1..F+K`, K at most the block less
one. Column i attends the ring and the block columns `j <= i` (`ops::sliding_window_attention`,
`SlidingWindowMask::Causal`). Each layer runs DFlash's backbone (`attn_input_proj`, `rmsnorm_rope`,
`linear_add`, `linear_swiglu`) and then the bridge writes into every column but the anchor's:

```text
rms_i     = sqrt(max(mean_h raw_i^2, FLT_MIN))
p_i       = W_read^l BF16(raw_i / (rms_i + e_b))
code_i    = BF16(sqrt(C) * p_i / (|p_i| + e_b))
[g_i; s_i; w_i] = W_bridge code_i                     (receiver gate, sender gate, sender value)
msg_i     = BF16(sigmoid(g_i + s_(i-1) + b) * w_(i-1))         i >= 1, msg_0 = 0
out_i     = BF16(raw_i + rms_i * W_write^l msg_i)              i >= 1, out_0 = raw_0
```

Column 1 reads its sender rows from the sender table at the anchor. `out` normalized by the next
layer's input norm (after the last layer the final norm, keeping its denominator `d_i`) is the next
layer's input. [`dspine.h`](../../include/infernix/ops/dspine.h) states every rounding point.

The 27B drafter's projections have registered routes, so a round streams each weight once:

- **The bridge:** the BF16 `[1024, 5120]` read, `[1536, 1024]` projection and `[5120, 512]` write take
  the skinny GEMV in launches of 16 columns up to 128 (`ops/linear/bf16/shapes/dspine_bridge.cu`).
  Past eight columns the runtime-shape fallback tiled them with its SIMT GEMM. That cost 2-3 ms a round
  on an RTX 5090 (Gate 2, 2026-11-25); streamed, the 15 products take 0.16 ms at 16 columns.
- **The attention output:** the Q8 `[5120, 4096]` has no `linear_add` form. It projects on `[5120,
  6144]`'s sliced-K schedules (`ops/linear/q8/shapes/n5120_k4096.cu`), then adds.

## Candidates and the refinement

The target's head ranks the 16 best candidates of each mask column's final hidden: the screened head
where the DFlash2 drafter's screen is fitted (RV-41), else `linear_topk`. Draft row i is block column
`n = i+1`; its sender t is the anchor for row 0 and the previous row's draft otherwise:

```text
m_t[r]    = sigmoid(g_n[r] + sender_table[t, r] + b[r]) * sender_table[t, R + r]
edge(t,c) = d_n * u_i[c] + rms_n * sum_r (m_t[r] - msg_n[r]) * TG[C_i[c], r]
TG[v, r]  = BF16( Head(w_final * W_write^last[:, r])[v] )
```

The head-write table TG (`R * V * 2` bytes, 243 MiB for the 27B drafter) is derived once at load by
running the target's head over the last write's R columns. A row at temperature 0 takes the
candidate of greatest edge; at temperature T it draws from `softmax(edge / (d_n * T))` over the 16
candidates, truncated by the request's top-k, top-p and min-p, under the request's seed, the drafted
position, `kSamplePurposeDSpineProposal` and the draft's step in its block as sub-key (as DFlash2's: a later
round's draft at the same position has a smaller step, so no round reuses another's uniform), and records that
law as the proposal's q. A `gumbel_max` row (the gumbel-sampling fast path) instead takes the kept candidate whose
token scores highest under the Gumbel noise its verification gives that token at the produced position (DFlash2's
coupled draw): a draw from the same law, which is the target's own draw whenever the drafter keeps that token. A draft's
confidence (the `--draft-tokens auto` width policy's input) is its probability under
`softmax(edge / d_n)`.

## Verification and rounds

A DSpine round is a DFlash2 chain round: the same sparse frame, rejection sampling against the
recorded q (greedy rows: one-hot), n-gram copy rounds and the `--draft-tokens auto` width policy
(adaptive-dflash: the whole block is proposed, the host chooses the verified width). The block is
causal, so a narrow round's drafts are the leading drafts of a wide one. Tree verification and
`--lm-head-draft` are refused. Where `--draft-tokens auto` or `--ngram-draft-tokens auto` resolve
per configuration, DSpine takes DFlash2's measured values (fixed length, width costs, batch length, copy
window). Measured for DSpine on the NVFP4 27B (2026-11-26, RTX 5090, INT8 KV, `infernix_decode_quality_gen`
on Infernix's code and sampled Markdown docs, ABBA), none of its own beat them:

- **Batch length:** rounds of two and four rows at 0 (the width policy's split rounds), 5, 9, 11 and 15
  drafts lost to 7 by 1.9-4.5 % (geometric mean of code and docs at two and four rows); code favours longer
  batches (15 at two rows +7.9 %), sampled docs 5-7 (9-15 lose 6-16 %).
- **Width costs:** proposal and per-draft costs of (0.3, 0.02), (0.5, 0.01), (0.18, 0.01) and (0.18, 0.03)
  against DFlash2's (0.18, 0.02) moved one request by at most +0.3 % over 72 code and docs slices and 24 chats;
  24-slice runs had suggested +1.4 %, but each setting's text diverges (its widths change the target's routes),
  and that text variance is about ±3 % over 24 slices.
- **Copy windows:** not swept: one-row copy rounds run no drafter.
- **Candidates:** the screened head beats exact `linear_topk` for DSpine too (one request: chats +2.4 %,
  sampled docs +5.1 %).

## State

The ring is the masked draft's local state, one per state image, as DFlash2's: checkpoints, forks,
Host replicas and the hybrid prefix cache carry it. A prefix cache file's fingerprint names the
speculative backend and the drafter artifact's identity. Session persistence refuses DSpine as it
refuses DFlash and DFlash2.

## Tests

| Test | Checks |
|---|---|
| `tests/ops/test_dspine.cpp` | each bridge and refinement Op against an FP64 oracle of its formula |
| `tests/ops/test_sliding_window_attention.cpp` | the causal mask and the 8192 window |
| `tests/ops/test_context_kv_materialize.cpp` | the 8192 ring's slots |
| `tests/ops/linear/test_bf16_a16.cpp`, `test_q8_a16.cpp` | the bridge's and the attention output's routes; a bridge column's bits at every width |
| `tests/test_cli_options.cpp` | the options and their refusals |
