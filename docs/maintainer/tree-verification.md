# DFlash2 tree verification

DFlash2 proposes one conditional path per round and the target verifies it as a causal chain. A
round ends at the first rejected proposal, so a wrong guess at an early position discards the whole
rest of the path. Tree verification lets the target check a small tree of proposals in the same
forward pass: alternatives share the prefix they extend, and the round continues along whichever
alternative the target accepts. Each round builds a new tree for every row from the drafter's
candidate lattice, spending its columns where the drafter is least sure. The output distribution
is unchanged: greedy rounds emit the target's argmax continuation and sampled rounds emit exactly
the processed target distribution (see [Exactness](#exactness)).

This document owns the tree mathematics, the per-round tree builder, acceptance, compaction and
configuration. [DFlash](dflash.md) owns the drafter and the chain round it extends;
[ReplaySSM](replayssm-gdn.md) owns records and Fold, which tree rounds reuse unchanged.

## Scope

- DFlash2 neural rounds only. Ngram copy rounds, MTP and DFlash (v1) keep chain verification.
- Every KV storage (`bf16`, `int8`, `fp8`, `nvfp4`, `k8v4`): each format's grouped and parallel
  verification attention applies per-row ancestor masks, and compaction moves every stored plane.
- Single FP8 or NVFP4 GDN input projections (the tree convolution reads ancestors from the
  materialized projection); startup rejects other parents.
- A tree row holds at most 32 columns (one 32-bit ancestor mask per column) and 8 root-to-leaf
  paths.

## Topology

A row's tree lives in its verification columns. Column 0 is the anchor (the unprocessed committed
token at position F); every other column c has `parent[c] < c` and
`depth[c] = depth[parent[c]] + 1`. Two layout rules are relied on throughout:

1. **The main chain occupies columns 0..K** (`parent[c] = c-1`, K = `--draft-tokens`): it is the
   chain round's proposal, bit for bit.
2. **Every other column lies after the main chain, at depth at most K.** Any accepted path's
   columns c_i satisfy either `c_i = i` (main) or `c_i > K >= i`, so copying accepted side columns
   onto positions `i` never reads a slot the same copy writes ([Compaction](#compaction)).

The builder publishes one `ops::SpeculativeTreeRow` per row (`include/ninfer/ops/speculative_tree.h`):
parents, depths, children in draw order (`first_child`, `next_sibling`, `sibling_index`), the
root-to-leaf paths in leaf column order with the first column each path owns (`owned_from`; a
column belongs to the first path that contains it), the live column count and whether the row is
a tree row. It also writes an I32 `[W,B]` matrix of ancestor masks (bit a of column c is set iff a
is c or an ancestor of c). A chain row of a tree round (one whose extent is below the tree's, see
[Host, frame and graphs](#host-frame-and-graphs)) publishes its chain prefix and is verified as
a chain.

## Configuration

| Option | Meaning | Default |
|---|---|---|
| `--draft-tree-nodes auto` | every all-neural round of up to four rows chooses chain verification or a tree of K+5 or K+9 columns from measured round time and acceptance ([Automatic widths](#automatic-widths)) | off |
| `--draft-tree-nodes LIST` | fixed columns per tree row by batch size: entry c applies to rounds of c rows and the last entry to every larger batch; `0` keeps that batch size on chain verification; nonzero entries are `draft tokens + 2 .. 32` | off |
| `--draft-tree-paths N` | most root-to-leaf paths per tree row, `2..8` (each path is one parallel GDN replay) | `8` |

A tree widens every target projection, the LM head, attention and the per-column target
distribution of a row from K+1 to N columns, and the GDN replay to the row's path count. A column
buys about the same acceptance at any batch size, but rows share the weight streaming of a round
and not its per-column work, so a column costs more of each row's share of the round as the batch
grows, and every column of a verification block wider than 8 costs attention time that grows with
the context ([Measurements](#measurements)). The useful width therefore shrinks with the batch
size and the context length, which automatic widths measure and a fixed table cannot follow. One
CUDA Graph family is captured per distinct width, for the batch sizes that may use it.

## Automatic widths

`--draft-tree-nodes auto` captures, for rounds of up to four rows, the chain family and tree
families of K+5 and K+9 columns (12 and 16 at K=7, the best two widths at short context), and
`TreeWidthController` (`program/speculative/tree_width_controller.h`) picks each all-neural round's
width on the host before the round is submitted:

- **Gain, measured on the generated text.** The builder adds columns best first, so a tree of w
  columns is the first w columns of any wider tree built for the same row, and its acceptance walk
  follows the same path until the path first uses a column >= w (the chain is the tree of K+1
  columns). Every verified tree row therefore gives, from its accepted path in the round's egress,
  the tokens each narrower width would have emitted in that round on the same text. A width's gain
  is its token total over the chain's on the same rows, as a running sum that decays by 1/512 per
  row and counts after 32 rows.
- **Cost, measured per context.** The wall time of every all-neural round is kept per batch size,
  context bucket (largest execution frontier below 8K, 16K, 32K, 64K, 128K, or longer) and width:
  the smallest of its first three rounds, then a running average that moves by an eighth of each
  round's difference and ignores the part of a spike beyond half the estimate.
- **Choice.** A round takes the width with the most gain per second. Widths are timed three times
  in a new bucket, widest first (a wide round also measures every narrower width's gain); the
  widest width runs every 32 rounds and every other width once its timing is 512 rounds old, which
  keeps every estimate current at a cost of about one wide round in 32.

The choice depends only on earlier rounds, never on the draws of the round it sizes, so sampling
stays exact. It does depend on measured time: two runs of a seeded request may verify different
trees and so sample different, equally distributed text. A fixed table keeps seeded runs at one
request reproducible. On a target whose GDN input projections cannot verify trees (see
[Scope](#scope)) automatic mode resolves to chain verification: the Engine's resolved options
report `draft_tree_auto` false and a startup warning says why, while a fixed table fails startup.

## Tree builder

The drafter forward is unchanged: one masked block of K+1 columns produces, for each position
i=1..K, 16 candidates `C_i` and the predecessor-conditional edge lattice `E_i[p,c]`.
`ops::candidate_selector_tree` then builds each row's tree (one warp per row):

```text
law of node v (depth i-1, rank p_v):  q_v(c) = softmax_c(E_i[p_v, c] / temperature)
                                      greedy rows rank branches with the temperature-1 softmax
score(v)    = product of the law values on v's path (score(anchor) = 1)
priority(v) = score(v) * E[law value of v's next child | v's earlier children]
              E = max untaken law value (greedy), or sum_untaken q^2 / sum_untaken q (sampled)
```

1. The main chain is drawn exactly as `candidate_selector_path` draws it: child 0 of each main
   column, counter key `(seed, F+i-1, DFlash2Proposal, 0)`.
2. A tree row (extent W-1) then adds side columns best first until it holds W columns: the node
   with the highest priority (lowest column on ties) takes its next child, drawn without
   replacement from q_v restricted to its untaken ranks with counter key
   `(seed, F+depth(v), DFlash2Proposal, ((v+1) << 5) | j)` for sibling j (greedy rows take the best
   untaken rank). Nodes at depth K take no children, and once a row holds `--draft-tree-paths`
   leaves only leaves may be extended. A row that runs out of extendable nodes stops early and
   publishes its node count as its live columns.
3. Every column stores its lattice row's candidates and the full law of its parent (a one-hot of
   the drawn rank for greedy rows); acceptance derives the without-replacement laws itself.

A child's priority is fixed before it is drawn, from the earlier draws only. The tree's shape can
therefore depend on any draw except a column's own token and the draws that depend on it, which
is what keeps recursive rejection sampling exact. The counter index is unique per (parent,
sibling), so no two columns at one position share a uniform.

## Verification forward

The target evaluates all W columns of a tree row in one forward, as one ragged batch row:

| Quantity | Tree row |
|---|---|
| input id of column c | anchor for c=0, else its draft (columns past the live count repeat the anchor) |
| KV cache slot of column c | `F + c` (sequential; the append contract is unchanged) |
| RoPE position of column c | the anchor's position plus `depth[c]` (written on the device) |
| attention visibility of column c | cache rows `[0,F)` and block rows `F+a` for every bit a of its ancestor mask |
| GDN convolution window of column c | projected inputs of its three nearest ancestors (conv history past the anchor), then c itself |
| GDN recurrence of column c | the state after the path from the anchor to c |
| MLP, projections, norms, LM head | per column, unchanged |

For the main chain the ancestor mask is the causal triangle, the depth is the column and the
ancestors are the preceding columns, so every main-chain quantity has the chain definition.

**Attention.** Every storage's grouped (W<=8) and parallel grouped (W>8) verification kernel has a
tree instance that reads the row's mask for each query column: a key at cache row `k` is visible
to column c iff `k <= F+c` and (`k < F` or bit `k-F` of the mask is set). Prefix KV is read once
per KV head for all W columns. Chain instances carry no mask arithmetic.

**GDN convolution.** Tree rounds use the materialized projection (projection written to the record
plane, then the width-4 convolution). The tree convolution evaluates
`w0*s(-3) + w1*s(-2) + w2*s(-1) + w3*x_c` with the chain kernel's FMA order, each `s` taken from
the row's ancestor chain. Where a chain round would select a fused projection-convolution
schedule, tree rounds differ from it by the one intermediate BF16 rounding of the projection.

**GDN recurrence (path-parallel replay).** The record kernel runs one CTA row per (request, path)
up to `--draft-tree-paths`. Each path replays its column list from the request's initial state; a
column shared by several paths is recomputed identically by each and written only by the path that
owns it. Paths past a row's leaves exit at once. Replay is exact: a path computes the same
sequence of operations a chain round of that path would. The persistent recurrent and convolution
state is not modified by verification (ReplaySSM record mode), so rejected branches need no
rollback.

A tree widens the verification block, which selects other GEMM tiles and attention routes, so the
main-chain columns of a tree round are not bit-identical to those of a chain round, as ngram
rounds at their own width already are not.

## Acceptance

Let `p_c` be the processed target distribution of column c (public vocabulary mask, penalties,
temperature, top-k/top-p/min-p). The penalty overlay of column c is the drafts of its ancestors
and itself, excluding the anchor, which for the main chain is the chain's `drafts[0..c-1]`.

Starting at the anchor, `ops::speculative_accept_sparse_tree` walks down the tree:

```text
v = 0
loop:
    children x_0..x_{m-1} of v in draw order (x_0 ~ q_v, later ones without replacement)
    if m == 0: emit bonus ~ p_v                                        (key: Bonus, index 0)
    r = p_v                                                            (residual target)
    for j = 0..m-1:
        Q_j = q_v restricted to C \ {x_0..x_{j-1}}, renormalized       (Q_0 = q_v)
        accept x_j with probability min(1, r(x_j) / Q_j(x_j))          (key: Accept, index j)
        if accepted: emit x_j, v = column of x_j, continue loop
        r = normalize(max(r - Q_j, 0))
    emit correction ~ r                                                (key: Correction, index 0)
    stop
```

Counter keys use the logical position `L + depth + 1` of the emitted token, exactly as the chain
does (`L` is the old round length). One node per depth is processed in a walk, so no key repeats.
With m=1 the loop is the chain acceptance, bit for bit. Greedy rows accept the child equal to the
(penalty-adjusted) target argmax of `p_v`, if any, and otherwise emit that argmax.

The kernel writes, per row, the accepted path's columns `c_0=0, c_1..c_A`, the licensed tokens, A,
the new anchor and length (unchanged contract), and the first depth i with `c_i != i` (0 when the
path stays on the main chain).

### Exactness

At a node v with m children, the children are a without-replacement sample from q_v whose length
m was decided from earlier draws only. Conditioned on `x_0..x_{j-1}` having been drawn and
rejected (and on any draw elsewhere in the tree that does not depend on `x_j`), `x_j` is
distributed as `Q_j`, and `r` is the residual of the previous rejections. Each iteration is a
standard rejection step with proposal `Q_j` and target `r`, which emits a token distributed as `r`
when it accepts and hands the normalized residual on otherwise; stopping at any step emits from the
current residual. By induction over j the token emitted at v is distributed exactly as `p_v`
(recursive rejection sampling, Jeon et al. 2024, "Recursive Speculative Decoding"), whatever rule
chose m, as long as the rule did not look at the children it chose to draw. The next level
conditions on the emitted token, as in chain speculative sampling, so the whole emitted sequence
has the target's joint distribution. Greedy rows are exact because they emit the argmax at every
step.

## Compaction

After acceptance and before the continuation hidden is selected, one in-graph step makes the
physical block look like a chain round whose accepted path occupied columns 0..A. For each row
whose path leaves the main chain, and each i in 1..A with `c_i != i`, it copies column `c_i` onto
column `i` of:

- the target KV cache slots `F+c_i -> F+i` in every full-attention layer: every stored plane of the
  layer's storage format (codes and scales of K and V);
- every ReplaySSM record plane (key, value, gate, conv) in every GDN layer;
- the verification hidden states (continuation hidden is taken from column A afterwards);
- the DFlash2 pending target features.

By layout rule 2 the sources lie after the main chain and the destinations within it, so the copy
is order-independent. After compaction the host sees an ordinary chain round with A accepted
drafts: commit, Fold (record columns 0..n-1), context catch-up and KV trim are unchanged. KV slots
and records beyond the committed prefix are unreachable, as for rejected chain drafts.

## Host, frame and graphs

- An all-neural round of c rows uses the tree family of width `nodes[c]` when that entry is
  nonzero, or the width the controller chooses in automatic mode; a round with an n-gram copy
  keeps its n-gram family. A row of a tree round is a tree row when its whole main chain fits
  (extent K and
  `F + W <= capacity`): it gets extent W-1 and KV mapped through `F+W`, and its RoPE positions are
  the anchor's (the device adds depths). Any other row (budget tail, capacity) is a chain row with
  its chain prefix and the chain's positions.
- The drafter proposes K+1 columns at its own width in its narrow frame view; the builder writes
  the tree straight into the round's W-wide frame, with the tree rows and masks in device-only
  frame storage. Record width, pending-feature width and workspace plans use W.
- Egress gains each row's accepted path and the depth at which it left the main chain; request
  statistics count tree rounds, rounds that accepted a side branch and the drafts accepted after
  leaving the main chain. A chain round on the same text would have emitted its correction token
  at the branch point instead, so those side drafts are the tree's gain per round. A tree round's
  drafted tokens are its W-1 tree columns.

## Verification

Op tests (`tests/ops/test_speculative_tree.cpp`, `tests/ops/softmax_attention/causal_cache.cpp`):

1. Builder: main chain bit-identical to `candidate_selector_path`; published topology equal to
   the host derivation of its parent array; masks and live counts; and a host replay with FP64
   lattice laws: every side column's parent had the highest priority (within FP32 rounding) when the
   column was added, its rank is the untaken-restricted inverse CDF of the parent's law with the
   column's counter key, and a short row has no extendable node left.
2. Acceptance: greedy walks over per-row trees (main, side, below-main branches, short rows, chain
   rows); chain-shaped tree rows bit-identical to the chain Op; the emitted-token distribution at a
   root fan-out of 1..4 children against `p_v` (chi-square).
3. Attention: random per-row trees, chain-masked and short rows, on every storage's grouped and
   parallel routes, direct and graph-replayed, against the ancestor-visibility FP64 oracle.
4. GDN: every path of every tree row bit-identical to the chain convolution and record on that
   path's columns; idle columns zero; records exact.
5. Compaction: record-plane columns and every KV storage's planes, exact.

Host (`tests/models/qwen3_5/test_tree_width_controller.cpp`): the controller's same-text gains are
the exact token ratios of known accepted paths (and a narrower round measures no wider width),
widths are timed widest first in every new context bucket, a simulated engine settles on the width
with the most tokens per second in each bucket (16 columns at short context, the chain where
columns cost more, 12 columns when it pays most) with the widest width explored every 32 rounds,
and a single twentyfold timing spike does not change the choice.

Real model (`NINFER_TEST_ARTIFACT`, see `tests/README.md`):

6. `test_engine_dflash2_real` with a tree table or `auto` runs the DFlash2 Engine fixture
   (budgets, stops inside a licensed block, penalties, same-seed replay for fixed tables, prefix
   reuse, compact batches) through tree rounds on every KV storage, with and without CUDA Graphs,
   at K=3, 7 and 15.
7. `test_engine_dflash2_tree_greedy_real` compares every token of greedy tree-decoded streams with
   a fresh one-token greedy prefill of the exact committed prefix. Decode rounds round differently
   from one prefill, so chain decoding disagrees with this oracle too. With the 27B NVIDIA artifact
   over all five KV storages (4800 positions per configuration) the disagreement rate was 3.3 %
   for 8-column chains, 3.0 % for 16-column chains and 3.3 % for 16-column trees, which committed
   704 side-branch drafts; disagreements above half a nat numbered 13, 13 and 14. The largest tree
   margin, 2.94 nats on BF16, sits at a position where a fresh decode round of every kind and width
   (tree, 8- and 16-column chain) picks a third token, and chain decoding shows the same effect
   (1.25 nats); a state or compaction error would instead leave the target's choice at most later
   positions.

## Measurements

RTX 5090, CUDA 13.4, Windows; `qwen3_8_27b_nvfp4-nvidia.ninfer` with DFlash2 K=7 and the optimized
proposal head plus n-gram 15/12 (the production decode flags), `--draft-tree-paths 8`.
`tools/bench/run_serve_concurrency.py --suite decode-saturation` runs C concurrent 8192-token
reasoning requests (temperature 0.6, top-p 0.95, top-k 20, presence penalty 1), one server per
point; arms are interleaved, their order alternates between passes, and pass p samples with
`--seed-set p-1`. Decode tok/s, tokens per row-round and ms per round come from the steady window.

With one request a pass is one sampled text, and the text alone moves decode tok/s by several
percent (the chain arm ranged 202.5-211.4 tok/s over four seed sets), so per-pass changes scatter
widely; ms per round is stable to about 0.5 %. INT8 KV, four seed sets:

| C | Columns per row | tok/s | Change (per seed set) | Tokens per row-round | ms per round |
|---|---|---:|---|---:|---:|
| 1 | chain (8) | 208.3 | | 3.03 | 14.56 |
| 1 | 12 | 222.6 | +6.9 % (+5.8..+8.6) | 3.40 | 15.29 |
| 1 | 16 | 226.0 | +8.6 % (+5.0..+13.6) | 3.58 | 15.83 |
| 1 | 20 | 222.8 | +7.0 % (+2.4..+11.5) | 3.68 | 16.53 |
| 2 | chain (8) | 394.7 | | 3.08 | 15.63 |
| 2 | 12 | 406.4 | +3.0 % (+1.1..+4.7) | 3.39 | 16.69 |
| 2 | 16 | 400.4 | +1.5 % (-1.3..+5.5) | 3.49 | 17.42 |
| 3 | chain (8) | 549.6 | | 3.04 | 16.59 |
| 3 | 10 | 554.2 | +0.9 % (-2.8..+3.3) | 3.26 | 17.65 |
| 4 | chain (8) | 686.5 | | 3.02 | 17.61 |
| 4 | 9 | 679.7 | -0.9 % (-4.4..+1.6) | 3.17 | 18.68 |

An earlier pair of passes on seed set 0 alone measured 12 columns at +5.5 % with two requests; its
chain sample (380 tok/s) was the slowest of the five two-request chain samples, so the table above
replaces it.

**Same-text estimate.** A chain round on the tree run's own text would have emitted its correction
where the accepted path left the main chain, so `tree_side_accepted_tokens` is exactly what the
tree added to its rounds. The acceptance gain g = (A+R)/(A-S+R) (A accepted drafts, R rounds, S side
drafts) is measured on one text without the text-to-text spread above, and with the stable round
times it gives the speedup g x ms_chain / ms_tree. One fresh seed set (set 4), per KV format:

| C | Columns | `int8` | `k8v4` | `nvfp4` | Acceptance gain | Round time |
|---|---|---:|---:|---:|---|---|
| 1 | 12 | +6.4 % | +5.8 % | +5.9 % | +11.4..+11.9 % | +5.1..+5.6 % |
| 1 | 16 | +7.3 % | +8.5 % | +7.1 % | +16.8..+18.2 % | +8.8..+9.3 % |
| 1 | 20 | +6.6 % | +5.0 % | +5.4 % | +19.9..+21.6 % | +14.0..+14.8 % |
| 1 | 24 | +5.8 % | | | +21.2 % | +14.5 % |
| 2 | 12 | +4.8 % | +4.1 % | +4.2 % | +12.0..+12.8 % | +6.8..+8.3 % |
| 2 | 16 | +4.5 % | +3.9 % | +3.5 % | +16.5..+16.9 % | +11.5..+12.9 % |
| 3 | 10 | +1.6 % | | | +8.1 % | +6.4 % |
| 3 | 12 | +2.4 % | | | +12.1 % | +9.5 % |
| 4 | 9 | -1.3 % | | | +4.7 % | +6.2 % |
| 4 | 10 | 0.0 % | | | +7.5 % | +7.5 % |

At one request these agree with the four-seed means above within 1.3 points (12, 16 and 20
columns: +6.4, +7.3 and +6.6 % against +6.9, +8.6 and +7.0 %). A width buys the same acceptance at
every batch size (12 columns: +11.8, +12.0 and +12.1 % at one, two and three requests), while its
round time grows with the batch (+5.1, +6.8 and +9.5 %). 24 columns buy no more than 20.

**KV formats.** One request, 12 columns, seed sets 1-3:

| KV | chain tok/s | tree tok/s | Change (per seed set) | Tokens per row-round | ms per round |
|---|---:|---:|---|---|---|
| `int8` | 208.1 | 222.3 | +6.8 % (+8.6, +6.0, +5.8) | 3.03 -> 3.40 | 14.54 -> 15.28 |
| `k8v4` | 210.6 | 231.0 | +9.7 % (+5.2, +7.0, +17.0) | 3.04 -> 3.50 | 14.43 -> 15.16 |
| `nvfp4` | 213.8 | 220.1 | +3.0 % (+7.4, +4.9, -3.4) | 3.08 -> 3.35 | 14.40 -> 15.22 |

The per-round cost is the same within 0.1 ms across formats (+0.74, +0.74 and +0.81 ms); the
spread of the changes is the text.

**Path cap.** One request, 12 columns, INT8 KV, seed set 0 (one text per arm), against the chain at
210.8 tok/s: two paths +0.8 %, three +6.4 %, four +6.1 %, eight +6.6 %, at 15.13, 15.19, 15.27 and
15.26 ms per round. With two paths the builder can open only one side branch and then only extends
leaves, so 15 % of the rounds leave the main chain against 20-23 % with three or more. Eight paths
cost no more per round than four, so the default is the maximum.

**Where a round's time goes.** Nsight Systems (`--cuda-graph-trace node`), greedy one-request
decode at short context, 12 columns against the chain on the same build, INT8 KV: +0.81 ms of
kernel time per round (+5.5 %). The GDN record kernel (one replay per path) adds 0.31 ms and the
tree convolution 0.11 ms; the projections, MLP, norms and LM head of the four extra columns add
0.30 ms; attention adds 0.03 ms (a block wider than 8 takes the parallel grouped route, whose KV
append is a separate kernel in every storage format); the builder, input preparation, acceptance
and compaction add 0.05 ms together.

**Long context.** The suite above runs at 0-9K tokens of context. Greedy `ninfer_bench -pg P,512` on
the bench corpus (INT8 KV, neural rounds only, one request) gives the extra round time against the
chain: 12 columns +6.6, +11.2 and +17.1 % at 16K, 64K and 128K, 16 columns +10.0, +14.7 and +20.3 %.
The corpus is too predictable under greedy decoding (6.9-7.6 tokens per round) for a tree to add
tokens, so these runs measure cost only. At 128K Nsight attributes 2.21 of the 2.82 ms a 12-column
tree adds per round to verification attention (2.80 -> 5.01 ms): every query row's QK and P x V work
grows with the context, and a 12-column block has half as many rows again as the chain's (reading
the KV once for all columns does not help; see RESEARCH_NOTES). The GDN record kernel adds 0.40 ms.
With the acceptance gain measured above (+12 % at 12 columns, +17 % at 16, which does not fall with
context in production logs), trees gain about 5-6 % at 16K and 1-2 % at 64K, and lose 3-4 % at 128K.

**Automatic widths.** The decode-saturation suite, INT8 KV, seed sets 4 and 5, chain, the best
fixed width and `auto` interleaved per pass; direct changes are the mean of the two per-pass
changes, same-text estimates are over both passes:

| C | Fixed width | Fixed: direct (per pass) | Fixed: same-text | Automatic: direct (per pass) | Automatic: same-text |
|---|---|---|---:|---|---:|
| 1 | 16 | +4.4 % (+8.1, +0.7) | +7.3 % | +6.2 % (+7.1, +5.4) | +6.9 % |
| 2 | 12 | +3.1 % (+2.7, +3.5) | +4.9 % | +3.6 % (+3.4, +3.8) | +4.4 % |
| 3 | 12 | +1.4 % (+3.3, -0.5) | +3.0 % | +1.5 % (+1.2, +1.7) | +1.1 % |
| 4 | chain | | | +1.2 % (+1.7, +0.6) | 0.0 % |

By the same-text estimate, which does not carry the text-to-text spread of the direct changes,
automatic widths stay 0.4-1.9 points behind the best fixed width at one to three requests: they
spend about one round in 32 on the widest width and take narrower trees or the chain where the
estimates are close (at three requests the acceptance gain is +9.7 % against the fixed 12
columns' +12.7 %). At four requests they break even, as the fixed tables do.

**Automatic widths at long context.** One request, sampled (thinking on, the model's default
sampling), 3072 output tokens after a 32K, 64K or 128K-token document (`long_niah` haystacks with a
request for a long critical essay), INT8 KV, DFlash2 K=7 without n-gram, one fresh `ninfer`
process per run, two seeds. The text differs between runs, so each run is reported by its
same-text estimate (its side-branch drafts against the chain's measured round time, the mean of
its two runs):

| Context | Fixed 16 columns | Automatic | Automatic tree rounds |
|---|---|---|---|
| 32K | +2.8 %, +5.8 % | +3.2 %, +3.7 % | 74 %, 99 % |
| 64K | +1.3 %, -0.1 % | +0.5 %, +1.8 % | 74 %, 99 % |
| 128K | -5.4 %, -4.2 % | -2.3 %, -0.6 % | 9 %, 8 % |

At 128K the automatic run pays for learning in a fresh process: about 38 rounds of the widest
tree until each width's gain has 32 rows, then the widest width every 32 rounds. A long-running
server learns once. On the bench corpus, whose greedy text the drafter predicts almost entirely
(6.9 tokens per round), 16 columns add 2 % tokens for 9 % round time: the fixed tree decodes 6.3 %
slower than the chain (447.9 against 478.0 tok/s at 2K context) and automatic widths verify chains
in 288 of 297 rounds at the chain's 478.0 tok/s.

**Memory.** A tree family replaces the chain family at the batch sizes that select it: graph memory
grows by 0-2 MiB, a 20-column table adds 9 MiB of record planes (wider than the 16-column n-gram
family), and KV capacity is unchanged.

**Recommendation.** `--draft-tree-nodes auto`. It follows the batch size, the context and the text:
trees where they pay, chain verification where wide blocks cost more than they gain (long context,
text the drafter already predicts). The best fixed table for contexts below about 64K tokens is
`16,12,12,0`: 16 columns with one request (+7-8.5 %), 12 with two (+4-5 %) and three, chain
verification from four requests, where trees break even. It gains up to 2 points more than `auto` at
short context and keeps seeded one-request runs reproducible, but it cannot switch trees off: it
loses 4.2-5.4 % at 128K and 6.3 % on highly predictable text. The three-request gain is small and
not resolved by the direct runs: +2.4 to +3.0 % by the same-text estimate, while seven direct passes
of 10 columns ranged from -2.8 to +4.4 % (mean +0.9 %). The suite is reasoning text at temperature
0.6; trees apply only to neural rounds, so workloads with more n-gram rounds gain less.
