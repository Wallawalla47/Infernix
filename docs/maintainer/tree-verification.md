# Tree verification

DFlash2, DFlash and MTP propose one path per round and the target verifies it as a causal chain. A
round ends at the first rejected proposal, so a wrong guess at an early position discards the whole
rest of the path. Tree verification lets the target check a small tree of proposals in the same
forward pass: alternatives share the prefix they extend, and the round continues along whichever
alternative the target accepts. Each round builds a new tree for every row from the drafter's
candidates (DFlash2's candidate lattice, DFlash's independent positions, the alternatives beside
each MTP draft), spending its columns where the drafter is least sure. The output distribution is unchanged: greedy rounds emit the
target's argmax continuation and sampled rounds emit exactly the processed target distribution
(see [Exactness](#exactness)).

This document owns the tree mathematics, the per-round tree builder, acceptance, compaction and
configuration. [DFlash](dflash.md) owns the drafter and the chain round it extends;
[ReplaySSM](replayssm-gdn.md) owns records and Fold, which tree rounds reuse unchanged.

## Scope

- DFlash2, DFlash and MTP neural rounds, and n-gram copy rounds whose copy sources continue
  differently ([N-gram trees](#n-gram-trees), `--ngram-tree-nodes`, also on Qwen3.8-Flash-Next).
  MTP neural trees need `--fast-path sampled-mtp-drafts` with resident experts (see [MTP](#mtp));
  n-gram trees do not.
- Every KV storage (`bf16`, `int8`, `fp8`, `nvfp4`, `k8v4`, `vq2`, `k4v2`): each format's grouped
  and parallel verification attention applies per-row ancestor masks, and compaction moves every
  stored plane. For `vq2` and `k4v2` it also moves each accepted column's exact-window slot in the
  row's StateImage slot, re-tagged for the destination position (cleared when the source slot no
  longer matches its codes).
- Single FP8, NVFP4 or Q8 GDN input projections (the tree convolution reads ancestors from the
  materialized projection); startup rejects other parents.
- A tree row holds at most 32 columns (one 32-bit ancestor mask per column) and 8 root-to-leaf
  paths.

## Topology

A row's tree lives in its verification columns. Column 0 is the anchor (the unprocessed committed
token at position F); every other column c has `parent[c] < c` and
`depth[c] = depth[parent[c]] + 1`. Two layout rules are relied on throughout:

1. **The main chain occupies columns 0..K** (`parent[c] = c-1`, K = `--draft-tokens`): it is the
   chain round's proposal, bit for bit (for DFlash, on greedy rows; see [Tree builder](#tree-builder)).
2. **Every other column lies after the main chain, at depth at most K.** Any accepted path's
   columns c_i satisfy either `c_i = i` (main) or `c_i > K >= i`, so copying accepted side columns
   onto positions `i` never reads a slot the same copy writes ([Compaction](#compaction)).

The builder publishes one `ops::SpeculativeTreeRow` per row (`include/infernix/ops/speculative_tree.h`):
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
| `--draft-tree-paths N` | most root-to-leaf paths per tree row, `2..8` (n-gram trees too; Qwen3.8-Flash-Next at most 3) | `8` |
| `--ngram-tree-nodes N` | with `--ngram-draft-tokens`: a copy round in which a row's copy sources continue differently verifies them as one tree of at most N columns, `4..32` and, with several requests, at least a batched round's neural draft tokens + 2 (Qwen3.8-Flash-Next 4..32, verified as at most 3 path rows whose columns together fit a one-row round's 16, when they beat the copy by expected tokens per measured round time) ([N-gram trees](#n-gram-trees)); `0` keeps copies on chain verification | `0` |

A tree widens every target projection, the LM head, attention, the GDN replay and the per-column
target distribution of a row from K+1 to N columns. A column
buys about the same acceptance at any batch size, but rows share the weight streaming of a round
and not its per-column work, so a column costs more of each row's share of the round as the batch
grows, and every column of a verification block wider than 8 costs attention time that grows with
the context ([Measurements](#measurements)). The useful width therefore shrinks with the batch
size and the context length, which automatic widths measure and a fixed table cannot follow. One
CUDA Graph family is captured per distinct width, for the batch sizes that may use it; the chain
family serves every batch size (a constrained round verifies the chain).

## Automatic widths

`--draft-tree-nodes auto` captures, for rounds of up to four rows, the chain family and tree
families of K+5 and K+9 columns (12 and 16 at K=7, the best two widths at short context), and
`TreeWidthController` (`program/speculative/tree_width_controller.h`) picks each all-neural round's
width on the host before the round is submitted (beside `--draft-tokens auto`'s `adaptive-dflash`
widths automatic trees resolve to chains: on the NVFP4 27B's 15-draft block, 20- and 24-column
trees ran 1.0-1.9 % slower than the 16-column chain by the same-text estimate, 2026-10-22, one
request, code and sampled docs, and the controller would time the chain only in the rounds the
width policy verifies the whole block; at 7 drafts automatic trees still gained +11.5 % on code
and +6.1 % on sampled docs against the 7-draft chain that day):

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
stays exact. (Proposing first and sizing each DFlash2 round from its own draft probabilities,
calibrated by the same measured gains, measured flat on the 27B, 1.0002x, and was removed
2026-10-28.) It does depend on measured time: two runs of a seeded request may verify different
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

**DFlash.** The DFlash drafter predicts each position of its block on its own, so a tree round
takes, at each position i, the head's 16 most probable tokens and their log-probabilities
(`ops::top_logprobs` over the head's logits; a shortlist head's rows are mapped to vocabulary ids)
as the position's candidates and an edge row that does not depend on the predecessor, and
`ops::candidate_tree` runs the same builder over these independent positions: every node at
depth i-1 has the law `softmax_c(logprob_i(c) / temperature)`. A greedy row's main chain is each
position's most probable token, the chain round's argmax proposal. A sampled row's drafts are
drawn from that 16-candidate law with the counter keys above, where a chain round proposes the
argmax; both are exact under the target's acceptance, which reads each column's candidates and
law in the same frame as DFlash2's. DFlash's chain rounds keep their deterministic-draft
verifier.

### MTP

MTP drafts its chain in the previous round's Finish, one autoregressive step per draft, so a
round's tree cannot draft past a branch: an MTP tree is the chain with **leaves**. Every round of
an engine with MTP tree families drafts through the sparse proposal of `sampled-mtp-drafts`
(each draft drawn from its law over the head's top four ranks, rank 0 with a one-hot law at
temperature 0) and also keeps those ranks' log-probabilities. In a tree round
`ops::candidate_chain_tree` takes each full-window row's chain as proposed and adds side columns
best first with the builder's priority, draws and counter keys, but only as children of chain
columns, which take no children themselves: a sampled row's law at step i is the law its chain draft
was drawn from, and a greedy row ranks its alternatives by the temperature-1 softmax of the
log-probabilities. The verification, acceptance and compaction are the DFlash rounds' (MTP has no
pending features); the alignment of the next proposal then reads the accepted path's tokens from the
licensed tokens (`ops::mtp_prepare_next_round`) and its compacted target hidden states, so the next
chain is drafted after the accepted path. Rows whose chain is shorter than the window (a budget or
context tail, or the prompt's argmax drafts, which carry no proposals) are chain rows of the round,
and a round of a fixed table's tree batch size runs instead of adaptive-mtp's draft-length policy;
automatic widths resolve to chains beside that policy.

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

**GDN recurrence (tree walk).** The record kernel runs the chain kernel's grid, one CTA per
(request, value head, state tile), and each CTA walks its row's tree once, depth first, so every
live column is computed exactly once from its parent's state. A branch node's state tile goes to a
shared-memory slot while its other subtrees run; the walk visits a node's later-drawn children
before its first child (a main column's main-chain child), so a main column's slot is released when
the walk returns to the main chain and a row with s side columns nests at most (s+1)/2 slots. Two
slots cover every tree of up to four side columns (K+5 columns); a deeper nesting rebuilds the
branch state from the initial state along its root path. Replay is exact: every column's state is
the chain recurrence of its root path, bit for bit. The persistent recurrent and convolution state
is not modified by verification (ReplaySSM record mode), so rejected branches need no rollback.

The `gdn-tree-closed-form` fast path (on by default on the NVFP4 27B and the 35B-A3B; `ops::GdnTreeRecordRoute::ClosedForm`; TreeWY,
arXiv 2608.20961) replaces the walk's W dependent steps: along a column's root path the transition
unrolls to `s_c = exp(G_c) s_0 + sum_a exp(G_c - G_a) u_a k_a^T` (G the summed gates), so each
column's delta `u_c` needs only its ancestors' deltas and `s_0 k_c` (one solve level per depth), and
its output `scale (exp(G_c) s_0 q_c + sum_a exp(G_c - G_a) (k_a . q_c) u_a)`. It runs on the walk's
grid with at least the walk's occupancy: each warp of a CTA holds four of the tile's 16 state rows (the
walk's registers), forms `s_0 k_c` and `s_0 q_c` for them and every column in one transposed reduction
per eight columns, and solves its rows' deltas on its own, one warp barrier per depth level; keys and
queries stay in shared memory as raw BF16 with one normalization scale per column (11.7 KiB, six CTAs
per SM); root paths and summed gates take one thread per column. It serves rows of up to 16 columns;
wider rows (K = 15 trees, whose solve runs up to 16 levels deep) walk, which measured faster there. FP32
in another order than the walk, its outputs are within BF16 rounding of the FP64 recurrence but not
bit-identical; the records are the same copies.

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

### UniVer trees (`univer-trees`)

The `univer-trees` fast-path candidate verifies sampled tree rows by UniVer (Weng, Hu, Yairi, arXiv
2605.04543), which composes local optimal-transport plans and reports +4.2..+8.5 % accepted length
over recursive rejection without replacement. It needs its own tree shape, so it applies to the
drafters that build their trees (DFlash2, DFlash) under a fixed `--draft-tree-nodes` table: an MTP
chain is drawn before any sibling exists, and automatic widths read narrower widths' tokens off the
wider tree's walk, which UniVer's decision does not share. Greedy rows, chain rows and Gumbel rows
are unchanged.

- **Builder** (`SpeculativeTreeShape::transport`). The row's shape is built first as a greedy row's
  is, under its sampled law: each node takes its best untaken rank, best first. No draw enters the
  shape. The tokens are then filled parents first: a node's children but the last take its law's
  top ranks in rank order, and its last child is drawn from its law restricted to the other ranks
  (counter key: the chain's for a main column's only child, `((v+1) << 5) | j` otherwise). A node
  with one child therefore proposes as the chain round does, and the main chain is the top rank
  wherever a main column has siblings. The row is marked `transport`, and `sampled` holds each
  residual draw's column.
- **Acceptance.** Allocation runs parents before children from `pt(anchor) = 1` with UniVer's
  local plan (`speculative_accept_sparse_tree`): the residual draw passes with
  `min(1, pt·p(u)/q'(u))`, every deterministic child takes `pt·p(d)` scaled by `(1 - a_u)/Z`, and
  each child's `pt` is its share given the earlier siblings' rejection. Then every column draws one
  uniform `(seed, L+depth+1, Accept, column)`, and the first passing column in post-order (subtrees
  in sibling order, then the node; the anchor always passes) ends the path. A leaf emits the bonus
  from its target; any other node emits a correction from `max(pt·p - q', 0)` over the tokens that
  are not its children.
- **Exactness.** The tree is locally lossless: at every node the expected emitted mass of each
  token over the residual draw is `pt·p`, because the deterministic children and the draw's law
  are fixed before the draw. The paper's two-stage theorem then gives the processed target
  distribution of every emitted token. Shapes may depend on draws elsewhere in the tree, because
  each subtree's expected return given its shape is its `pt`. The builder's shape depends on no
  draw at all.
- **Checks.** `tests/ops/test_speculative_tree.cpp` replays both builders' transport shapes and
  token fills, and tests the first emitted token and every reached column's next token against
  their targets (chi-square) on a two-level tree with deterministic children and residual draws.

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
  keeps its n-gram family, or the n-gram tree family when a row's copy sources continue
  differently ([N-gram trees](#n-gram-trees)). A row of a tree round is a tree row when its whole main chain fits
  (extent K and
  `F + W <= capacity`): it gets extent W-1 and KV mapped through `F+W`, and its RoPE positions are
  the anchor's (the device adds depths). Any other row (budget tail, capacity) is a chain row with
  its chain prefix and the chain's positions.
- A round with a grammar-constrained row verifies the chain whatever the table or controller
  would select: grammar masks follow one proposal chain
  ([Constrained decoding](constrained-decoding.md)), and tree acceptance reads no masks. The
  neural chain family is therefore captured for every batch size, also where the table selects
  only trees. Each family is a Forward/Finish graph pair; tree compaction runs in Finish.
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
   column's counter key, and a short row has no extendable node left. `candidate_tree` (DFlash)
   against the same host replay over FP64 independent-position laws, with its main chain checked
   against the host argmax and inverse CDF directly. `candidate_chain_tree` (MTP): the chain columns
   as given, bit for bit, and the replay with leaf-only side columns, including a row whose chain
   draft is missing from its candidates (no side column).
2. Acceptance: greedy walks over per-row trees (main, side, below-main branches, short rows, chain
   rows); chain-shaped tree rows bit-identical to the chain Op; the emitted-token distribution at a
   root fan-out of 1..4 children against `p_v` (chi-square).
3. Attention: random per-row trees, chain-masked and short rows, on every storage's grouped and
   parallel routes, direct and graph-replayed, against the ancestor-visibility FP64 oracle.
4. GDN: every path of every tree row bit-identical to the chain convolution and record on that
   path's columns, for trees whose walks nest side branches two deep, give a node four children
   and nest three deep (beyond the walk's two slots, so a branch state is rebuilt), up to 16
   columns at K=7; idle columns zero; records exact.
5. Compaction: record-plane columns and every KV storage's planes, exact.
6. N-gram trees: deterministic transport rows (one-hot laws, no residual draw) emit the anchor's
   target distribution first and every reached column's target after it (chi-square), and greedy
   rows take the branch whose drafts are the target tokens; GDN convolution and records on n-gram
   shapes (14-deep main chains with side columns as deep); `speculative_one_hot_proposals` exact.
   Host (`tests/models/qwen3_5/test_ngram_tree.cpp`): every layout rule, the main chain yielding to
   branches, the limits, and `propose_candidates` (propose's choice first, one per source).

Host (`tests/models/qwen3_5/test_tree_width_controller.cpp`): the controller's same-text gains are
the exact token ratios of known accepted paths (and a narrower round measures no wider width),
widths are timed widest first in every new context bucket, a simulated engine settles on the width
with the most tokens per second in each bucket (16 columns at short context, the chain where
columns cost more, 12 columns when it pays most) with the widest width explored every 32 rounds,
and a single twentyfold timing spike does not change the choice.

Real model (`INFERNIX_TEST_ARTIFACT`, see `tests/README.md`):

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
8. `test_engine_ngram_tree_real` (Qwen3.5: DFlash2 and MTP on the NVFP4 27B, DFlash on the 35B-A3B,
   one stream and two) and `test_ngram_tree_real` (Qwen3.8-Flash-Next: copies alone and beside the
   DFlash2 companion): greedy copies of a two-version prompt verified as trees accept the other
   source's branch, and commit the chain arm's tokens up to their first difference, where both
   tokens must be a near-tie of the target (a scoring pass: within 1.5 nats); a state error would
   leave the tree arm's token far below the target's choice. The Flash-Next test scores each arm's
   whole stream with the lossless route instead (at most two more tokens than the chain arm above
   1 nat of regret, none above 5 nats): on FP4-Local with DFlash2 the arms first differ at a token
   whose fast-route scoring gap is 4.5 nats but whose exact regret is 2.2, where both arms' first
   token already has a regret of 1.9.

## N-gram trees

An n-gram copy round verifies one copy: the continuation of the earlier occurrence whose suffix
match is longest (`NgramProposer::propose`). When the text occurs several times with different
continuations (two versions of a file, repeated boilerplate that diverges), the copy is wrong at the
first place its source differs from the one the model is following, and the round ends there.
`--ngram-tree-nodes N` verifies the sources together instead: a row whose sources continue
differently within the tree's depth verifies one tree of their continuations, and the round goes on
along whichever source the target accepts.

**Sources.** `NgramProposer::propose_candidates` returns every source whose suffix match qualifies
(at least `--ngram-min-match` tokens), one per source position, best first in `propose`'s order
(longer match, then longer continuation), each with its own continuation. The round's copy (which
may come from the archive) is the main chain; the row's own index gives the others.

**Tree** (`program/ngram_tree.h`, `build_ngram_tree`). Column 0 is the anchor and columns
1..D the copy's first D drafts, so both layout rules hold. D is the depth limit
(`min(15, N - 2)`, the row's output budget less its bonus, and its context room), shortened while
the other sources' branches would not reach one draft short of it, but never below three quarters
of the columns: a copy is usually right, so the main chain keeps the larger share, and a late
divergence costs it nothing (a floor of half the columns lost to chains at 16 columns,
[Measurements](#n-gram-trees-1)). Each other source then follows the tree while it agrees with it and continues as
a new branch from where it leaves it, at most D deep, in candidate order while columns and
`--draft-tree-paths` leaves last; a node takes at most 16 children. A row with no branch is a chain
and keeps its full copy (up to the n-gram window).

**Exactness.** Every draft is a deterministic proposal. The row is a transport row (UniVer,
[above](#univer-trees-univer-trees)) with no residual draw (`sampled = 0`): a sampled row accepts a
child d of v with probability `pt_v p_v(d)` (Z_v is 1 when nothing is drawn) and otherwise draws
the correction from `p_v` outside v's children, so the emitted sequence has the processed target
distribution; greedy and Gumbel rows take the child equal to the target's token. The laws each
child stores (its parent's children first, one-hot on the first) are never read: without a residual
draw the transport rule needs no proposal probability.

**Rounds.** A copy round in which some row has a tree is an n-gram tree round: it verifies at width
N, every tree row its tree, every other row its chain as a chain row (a copy of at most N-1 drafts,
or the neural proposal of a row without a copy, which N must hold: with several requests startup
requires N >= a batched round's neural draft tokens + 2, `adaptive-dflash`'s batch length where it
applies; a one-row copy round runs no drafter). Rounds with a grammar-constrained row keep chains,
and startup refuses trees with fixed speculation (the reproducible route, or a configuration
without `adaptive-speculation`). Request statistics count
`ngram_tree_rounds`, `ngram_tree_side_rounds` (the accepted path left the copy for another source)
and `ngram_tree_side_accepted_tokens` (the drafts after leaving it: what the copy alone would not
have accepted); `ngram_drafted_tokens` counts every verified tree column.

**Qwen3.5 family.** One `NgramTree` round family of width N, captured for every batch size. The
host writes each row's `ops::SpeculativeTreeRow` and ancestor masks into the round's ingress
(`ngram_tree_rows`, `ngram_tree_masks`) and a tree row's drafts and laws into its copy payload; the
verification reads them there, so one captured graph serves every tree. A batched round still runs
the drafter for its rows without a copy, at its own width; DFlash's argmax drafts then take the
one-hot law of a deterministic draft on the device (`ops::speculative_one_hot_proposals`). MTP
rounds carry every column's law in their own planes (`ngram_tree_candidates`, `ngram_tree_q`): a
tree's parents' laws, an MTP chain's own drafts' proposals. Verification, acceptance, compaction
and the commit are the device-built trees' above.

**Qwen3.8-Flash-Next** (`program/ngram_tree_program.cpp`). QSA's block-sparse attention has no tree
form: a tree column's visible keys, its block count and the pooled key of a block completing inside
the round all follow its own path. A one-row copy round with a tree therefore verifies each
root-to-leaf path as a verification row of its own (at most 3, `--draft-tree-paths` capped), with
every chain kernel, at the width W of the longest path. The rows together verify at most a one-row
round's columns (R x W <= the copy window + 1, 16): two paths of up to 8 columns or three of up to
5, so the plan needs no records, logits, io or graphs beyond a one-row round's, only the side rows'
pages and table rows.

1. Every path row reads the lane's state in place (GDN states, PLE history, QSA tails: verification
   leaves them unchanged) and the lane's KV pages. Side row r maps the round's positions (one or two
   64-position pages) to two pages of its own through a table row of its own: the lane's mapping
   copied on the device, with the lane's last pages copied into its own first (copy-on-write, one
   batched copy of every plane).
2. The forward runs R rows x W columns, as a chain round of R rows, uncaptured, through a view of
   the GDN records of R rows of W columns a layer (within a one-row round's storage). Each tree
   column's target is the logits column of the first path holding it, gathered into a [vocab, N]
   workspace buffer; `speculative_accept_sparse_tree` verifies the tree as above.
3. The row of the first path through the accepted path's last column becomes row 0 of an ordinary
   one-row round of width W: its GDN records re-laid layer by layer into the one-row view, its PLE
   and QSA records, MTP residual columns and DFlash2 features copied into place, and for a side row
   its pages (K/V of every attention layer, MTP cells, pooled keys) copied over the lane's. The
   commit, the MTP catch-up and the DFlash2 context are a one-row round's, unchanged.

A path row costs its whole path: columns shared by several paths are computed once per path, and
every column routes its own experts, so a tree round costs about its path count's chains. A round
verifies a tree only when it beats the round's copy by expected tokens per measured round time
(the copy's tokens at its bucket's acceptance times the request's learned tree gain, over the tree
round's estimated cycle; the copy choice and its measurements are in [Measurements](#n-gram-trees-1)),
so on an offloaded artifact it rarely does. Tree rounds run uncaptured: graphs of every row count and
width captured at startup cost every round (below). KV storages with a per-lane exact window
(`vq2`, `k4v2`) or per-row packed flags (`bf16p`) are refused, as is the lossless route.

**Qwen3.5 family gain gate.** A request learns its tree rounds' tokens over its copies' alone (the
copy stops where the accepted path left it, or goes on past the tree's main chain at the request's
copy acceptance; weight 0.25 from 1.1); below 1 it verifies its copies as chains, with a tree every
8th eligible round. NVFP4 27B, one request: single-source copies with trees of 16 +0.6 % -> +0.2 %
against chains, two versions unchanged (-15.4 %).

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

**Where a round's time goes.** Nsight Systems (`--cuda-graph-trace node`) on `infernix-serve`
during the decode-saturation suite with one, two and four requests (DFlash2 K=7, INT8 KV, no
n-gram so every round is a tree round, 2-4K tokens of context, 5 s windows of 140-320 rounds),
wall time per round with overlapping kernels charged once, against chain rounds of the same build:

| | 1 request | 2 requests | 4 requests |
|---|---|---|---|
| Chain, ms per round | 14.59 | 15.61 | 17.55 |
| 12 columns | +4.5 % | +5.4 % | +10.8 % |
| 16 columns | +8.9 % | +9.8 % | +16.6 % |
| GDN record, ms: chain / 12 / 16 columns | 0.47 / 0.73 / 0.95 | 0.66 / 1.00 / 1.29 | 1.12 / 1.76 / 2.29 |

The tree walk computes each column once, so its record time follows the column count; at one
request it runs one CTA per state tile through 12 or 16 serial steps where the chain runs 8. The
path-parallel replay it replaced, which ran every root-to-leaf path from the initial state on its
own CTAs, took 0.81 / 1.28 / 2.12 ms at 12 columns and 1.00 / 1.65 / 2.91 ms at 16; the walk makes
tree rounds 0.6 / 1.7 / 2.9 % faster at 12 columns and 0.4 / 2.0 / 4.0 % at 16. At 12 columns the
other additions are the tree convolution (0.11-0.12 ms), attention (0.11-0.31 ms; a block wider
than 8 takes the parallel grouped route, whose KV append is a separate kernel in every storage
format), the builder, input preparation, acceptance and compaction (0.08-0.19 ms, measured before
the walk), and at four requests the LM head over 48 columns (1.05 against 0.75 ms; 1.17 ms before
its sliced-K route was extended to 64 columns).

**Long context.** The suite above runs at 0-9K tokens of context. Greedy `infernix_bench -pg P,512` on
the bench corpus (INT8 KV, neural rounds only, one request) gives the extra round time against the
chain: 12 columns +6.6, +11.2 and +17.1 % at 16K, 64K and 128K, 16 columns +10.0, +14.7 and +20.3 %.
The corpus is too predictable under greedy decoding (6.9-7.6 tokens per round) for a tree to add
tokens, so these runs measure cost only. At 128K Nsight attributes 2.21 of the 2.82 ms a 12-column
tree adds per round to verification attention (2.80 -> 5.01 ms): every query row's QK and P x V work
grows with the context, and a 12-column block has half as many rows again as the chain's (reading
the KV once for all columns does not help; see RESEARCH_NOTES). The GDN record kernel added 0.40 ms
with the path-parallel replay (the tree walk adds 0.27 ms at one request, at any context). With the
acceptance gain measured above (+12 % at 12 columns, +17 % at 16, which does not fall with context
in production logs), trees gain about 5-6 % at 16K and 1-2 % at 64K, and lose 3-4 % at 128K.

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

These runs predate the tree walk and the 64-column sliced-K LM head. A later pair of passes (seed
sets 0 and 1, the same suite and flags) measured automatic widths against chain verification with
the build before and after both: at three requests the same-text estimate rose from +1.9 and +2.0 %
to +2.5 and +3.2 %, and at four from +0.1 and +0.4 % to +3.3 and +2.2 %, where automatic widths now
verify trees in 95-99 % of the rounds instead of 14-56 %.

**Automatic widths at long context.** One request, sampled (thinking on, the model's default
sampling), 3072 output tokens after a 32K, 64K or 128K-token document (`long_niah` haystacks with a
request for a long critical essay), INT8 KV, DFlash2 K=7 without n-gram, one fresh `infernix`
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

### N-gram trees

RTX 5090, CUDA 13.4, Windows, 2026-11-07; `infernix_decode_quality_gen` with `prompts`, greedy,
INT8 KV. Two copy workloads made from this repository's sources. *Two versions*: six prompts, each
a source file shown as version 1 and as version 2 (its three most frequent long identifiers renamed,
small numbers changed) and then asked for version 1 again, 400 tokens each; the newer version is
the copy's main chain, so a copy of version 1 meets its other source at every edit. *Single
source*: twelve `bench/decode_quality/copy_prompts.py` prompts (a file, then its header with one
identifier renamed), 512 tokens each, whose sources rarely disagree. One-request runs repeat bit
for bit (two runs each, within 0.3 %); two-request runs vary with scheduling, so both runs are
listed.

**NVFP4 27B** (`qwen3_8_27b_nvfp4-official`), DFlash2 `--draft-tokens auto` with
`--ngram-draft-tokens auto` (63 drafts, 31 in batched rounds); one request decode seconds, two
requests wall seconds:

| Workload | Requests | Chains | Trees of 16 | Trees of 32 |
|---|---|---|---|---|
| Two versions | 1 | 3.03 | 2.53 (-16.5 %) | 2.44 (-19.6 %) |
| Two versions | 2 | 3.17, 3.17 | 3.01, 2.79 | 3.07, 3.07 |
| Single source | 1 | 3.30 | 3.32 (+0.6 %) | 3.31 (+0.4 %) |
| Single source | 2 | 4.16, 4.16 | 4.17, 4.17 | 4.13, 4.13 |

Trees replace a copy round's single source only where another source leaves it within the tree's
depth: on the single-source prompts 3 (one request) and 7-9 (two) of 108-178 copy rounds verified
a tree, none took the other source, and the time is the chains'. With two versions 11-20 rounds
verified a tree and 3-6 took the other source; at one request 32 columns accept 1,914 drafts
against the chains' 1,796 in 49 against 45 copy rounds, although the tree's main chain holds at
most 15 drafts where a chain copies up to 63.

**Main-chain floor.** The builder shortens the main chain for the other sources' branches down to a
floor. Two versions, 16 columns: decode 3.51 s at one request with a floor of half the columns, 2.53
s at three quarters (the default) and 2.72 s without shortening; two requests 2.56 / 2.82, 3.01 /
2.79 and 2.80 / 2.80 s (wall). 32 columns hold a 15-draft main chain and a branch at every floor
(2.44 s). A copy is usually right after its first divergence (the match that took the other source
becomes the next round's main chain), so columns taken from the main chain cost more than the
branch gains.

**Qwen3.8-Flash-Next FP4-Local**, DFlash2 `--draft-tokens auto` (the companion), one request,
decode seconds. As first built, a one-row round took the whole copy (up to 15 drafts) when its
expected tokens per round beat the drafter's at a flat cost a column (`kWidthCost`), and trees
verified rows of up to 16 columns each:

| Workload | DFlash2 alone | + copies of 15 | + copies of 15, trees of 16 |
|---|---:|---:|---:|
| Two versions | 10.48 | 13.10 (+25 %) | 14.00 (+34 %) |
| Single source | 29.22, 29.36 | 32.54, 32.60 (+11 %) | 33.55 (+15 %) |

Copies lost even where they accepted 98 % of their drafts (two versions: 1,693 of 1,726). DFlash2
alone accepts 94 % of its 8-column rounds' drafts on that text (6.8 tokens a round at 29.8 ms); with
copies, if its 123 remaining rounds cost the same, each 16-column copy round took about 81 ms for
15.6 tokens, where the flat cost charged about 51: past the narrow expert kernels every column
routes its own experts at a higher marginal cost.

**Copy choice by measured round time** (`ProgramImpl::copy_choice`). Every one-row round's
start-to-start cycle is measured by kind (drafter, copy, tree) and verified columns (EWMAs); an
unmeasured width is interpolated between its kind's measured widths, else taken from another kind
at that width, else extended past the widest by the steeper of the measured slope and the flat
model. A round takes the copy's first e drafts that maximize its expected tokens (its match
bucket's acceptance) per estimated cycle, when they beat by 5 % the drafter's round where a copy
was available (its tokens and columns learned in such rounds: copied text is text the drafter
predicts well); every 8th eligible round runs the drafter instead and every 8th, offset, one draft
more, so both estimates stay current. Copy widths past the drafter's are captured at their first
use. Two runs each (the choices follow measured times, so runs differ), against DFlash2 alone:

| Workload | Copies of 15 enabled, never taken | + copies of 15 | + copies of 15, trees of 16 |
|---|---:|---:|---:|
| Two versions | +1.2 % | +3.0 % (+2.9, +3.1) | +3.1 % (+4.4, +1.9) |
| Single source | +1.4 % | +1.8 % | +1.7 % |
| Prose, twelve 1,024-token slices of `engine-architecture.md` | +0.7 % | +0.9 % (+2.2, -0.4) | +1.2 % |

The never-taken arm (copies enabled, the margin set past any copy: the same 352 rounds and stream)
is the configured cost: a copy window of 15 plans 16-column rounds, whose graph allowance (4 MiB a
graph), records and logits leave 87 fewer experts in VRAM (8,882 against 8,969), decode +1.1 to
+1.4 %, prefill +1.0 %. With MTP instead of DFlash2 (four drafts), copies gain: two versions 12.15 ->
11.16 s (-8.1 %), prose 16.07 -> 15.87 s (-1.3 %).

**Configured cost of trees (Flash-Next).** As first built, trees captured every path count's
verification at every width at startup and planned rows of 16 columns: FP4-Local with one lane
captured 90 CUDA Graphs (335 MiB) against 49 (160 MiB) for copies alone, 35 fewer experts in VRAM,
and on greedy prose where no tree formed (the same 855 rounds) decode +1.6 %, prefill +1.2 %; with
two lanes +1.3 %, and +0.1 % with the tree graphs left out (the expert cache the same), so the
graphs themselves cost every round. Trees now verify uncaptured within a one-row round's columns,
and cost as much as copies alone (the table above).

**Agentic coding (FP4-Local).** `bench/agentic_ab/runner.py`, the owner's launch flags (DFlash2
auto, two lanes, INT8 KV, 262,144-token context), 130 sampled requests per arm, workload seeds 42
and 43, every arm starting from the same saved expert state (`AB_EXPERT_STATE_RESET`); arms in the
order copies, copies + trees, none. Mean over the seeds against no copies (per-seed range), as first
built:

| | Copies of 15 | Copies of 15, trees of 16 |
|---|---|---|
| Decode rounds/s, one request decoding | -4.5 % (-5.6..-3.4) | -9.1 % (-9.2..-8.9) |
| Output tok/s, one request decoding | -2.9 % (-5.5..-0.2) | -9.5 % (-10.5..-8.5) |
| Output tok/s, all decoding | -2.9 % (-4.3..-1.5) | -9.1 % (-9.4..-8.8) |
| Prefill tok/s, no cache hit (client TTFT) | -2.0 % (-2.1..-1.9) | -5.2 % (-6.1..-4.2) |
| Average TTFT | +1.1 % (+0.3..+1.9) | +6.8 % (+6.7..+6.9) |

Copies supplied 0.3-0.5 % of the output (60-87 copy rounds of 35,000-37,000, 29-41 % of their
drafts accepted), and no tree round formed in either seed. By decode seconds over rounds in the
request logs, copies cost nothing a round (15.97 / 16.15 ms against 15.83 / 16.24) and trees 6 %
(17.04 / 17.05 ms). Without the expert-state reset an arm starts from the state the arm before it
saved: in an earlier run the first arm decoded at 90 tok/s (request log, tokens per decode second)
against 133 for the same flags after the reset.

The same sweep with the measured-cost copy choice, trees within a one-row round's columns and copy
widths captured at first use (the same seeds, arms and order):

| | Copies of 15 | Copies of 15, trees of 16 |
|---|---|---|
| Output tok/s, one request decoding | -1.7 % (-2.0..-1.5) | -1.0 % (-1.9..-0.1) |
| Output tok/s, all decoding | -1.9 % (-4.6..+0.8) | -3.2 % (-3.4..-3.0) |
| Prefill tok/s, no cache hit (client TTFT) | +1.5 % (+0.5..+2.4) | -0.2 % (-0.9..+0.4) |
| Decode ms a round (request logs) | +3.1 %, +2.2 % | +1.0 %, +0.4 % |

Copies verified 32-46 rounds a seed (they had 60-87), no tree round formed, and the two copy arms,
which differ only by the unused tree flag, differ by 1-2 % a round: the run-to-run spread of these
sampled texts (completion tokens 83-105 thousand an arm).

**On FP4-Local both off is fastest** for agentic coding and for every DFlash2 copy workload
measured: DFlash2 already predicts copied text well, so copies at best match its rounds, and
enabling them costs about 1 % of VRAM-bound speed. N-gram trees pay on the NVFP4 27B (32 columns),
where copies themselves are cheap, and copies pay beside Flash-Next's MTP drafter.
