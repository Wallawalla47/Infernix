#pragma once

// Inkling 2's DSpark drafter (design docs/maintainer/inkling2_mini-design.md §13; the reference's arithmetic in
// .local's DSPARK_SPEC): the Ops between its BF16 projections (which run on projections_dmma / projections_mma). A
// drafter decides only which tokens are proposed: verification computes every output on the target's decode route, so
// these Ops are deterministic and each row's result depends on its own inputs alone, with the reference's rounding
// points; they are qualified against FP64 oracles, not against another engine's bits.
//
// Shapes are Inkling 2 Mini's drafter: H = 2304, 24 query heads and 8 KV heads of 128, MLP 9216, a block of 8 rows,
// Markov rank 256, 4 context features (the partial block sums entering visits 5, 23 and 41, and the final-normed state).
//
// Drafter caches: a sliding layer's context K and V, BF16 [kv heads][ring][128] a lane (lane_stride words apart),
// position p at slot p % ring; the full layer's in the target's KV pages (its two planes of each page, page-major
// [pages][kv heads][64][128]) through the lane's block-table row, so a page carries the drafter's context with the
// target's (the prefix cache shares and restores them together).

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace infernix::ops::inkling2 {

inline constexpr std::int32_t kDSparkFeatures     = 4;
inline constexpr std::int32_t kDSparkBlock        = 8;
inline constexpr std::int32_t kDSparkHeads        = 24;
inline constexpr std::int32_t kDSparkKvHeads      = 8;
inline constexpr std::int32_t kDSparkHeadDim      = 128;
inline constexpr std::int32_t kDSparkIntermediate = 9216;
inline constexpr std::int32_t kMarkovRank         = 256;
inline constexpr std::int32_t kDSparkWindow       = 1024; // a sliding layer's context keys before the block
// A sliding layer's ring: the window, the block, and a verify call's columns appended before the next block's
// attention reads (positions [F - 1024, F + 8) live, F + 8 <= the newest appended position + 8).
inline constexpr std::int32_t kDSparkRing = 1152;
inline constexpr std::int32_t kMarkovSteps = kDSparkBlock - 1; // drafts a round at most

// The device words of a round (uploaded before the round's launches): the anchor's position F (the block's first)
// and token, and the lane.
struct DSparkRound {
    std::int32_t first  = 0;
    std::int32_t anchor = 0;
    std::int32_t lane   = 0;
};

// The mixed, normalized context features (the reference's _mix_rmsnorm): for each consumer row m of `rows` and column
// c, mix = sum_j fp32(T_j) * alpha[m][j] in tap order (FP32, unrounded), alpha = bf16(softmax_fp32(fuse_weight row m))
// over the 4 taps, out = bf16(mix * rsqrt(sum(mix^2) * (1 / H) + eps) * w_m).
struct DSparkMixArgs {
    const std::uint16_t* features    = nullptr; // BF16 [columns][4][H]
    std::int32_t columns             = 0;
    const std::uint16_t* fuse_weight = nullptr; // BF16 [4][4]: a row per consumer (layers 0-2, the Markov head)
    std::int32_t count               = 0;       // consumers in rows / norms / out
    std::int32_t rows[4]             = {};
    const std::uint16_t* norms[4]    = {};      // BF16 [H]
    std::uint16_t* out[4]            = {};      // BF16 [columns][H], or a ring [ring][H] at positions[c] % ring
    std::int32_t ring[4]             = {};      // 0: dense by column
    const std::int32_t* positions    = nullptr; // [columns] (ring outputs)
    const std::int32_t* lanes        = nullptr; // [columns] (ring outputs): the ring at out + lane * lane_stride
    std::int64_t lane_stride[4]      = {};
    float eps                        = 1e-6F;
};
void dspark_mix(const DSparkMixArgs& args, cudaStream_t stream);

// The full layer's context in INT8 (the target pools' int8 row codec, inkling2_attention.h KvCodec::Int8G64): K and V
// code planes [pages][kv heads][64][128] (int8) and side planes [pages][kv heads][64][4] (two FP16 scales a row, one a
// 64-value group: s = fp16(max |x| / 127), code = clamp(round(x / s), +-127)), page-major as the BF16 cache. A value's
// represented word is bf16(code * s) for K and V alike. Drafts only: no output depends on them.
struct DSparkInt8Cache {
    std::uint8_t *codes_k = nullptr, *codes_v = nullptr, *side_k = nullptr, *side_v = nullptr;
    [[nodiscard]] bool active() const noexcept { return codes_k != nullptr; }
};

// Per (row, head) of x: n = bf16(x * rsqrt(sum(x^2) * (1 / 128) + eps) * w), then NeoX RoPE at the row's position
// (pairs (i, i + 64), angle = fp32(position) * theta^(-i / 64), cos and sin rounded to BF16, FP32 products, one BF16
// rounding). Destination: a drafter cache (keys; `values` copied beside them unchanged, or both encoded into the full
// layer's INT8 planes, `int8`) at the row's position, or a dense [rows][heads][128] buffer. Positions: positions[r], or
// (positions null) *first + r.
struct DSparkRotateArgs {
    const std::uint16_t* x       = nullptr; // BF16 [rows][x_stride]: head h at h * 128
    std::int32_t x_stride        = 0;
    const std::uint16_t* values  = nullptr; // optional BF16 [rows][values_stride] (cache destinations only)
    std::int32_t values_stride   = 0;
    std::int32_t rows            = 0;
    std::int32_t heads           = 0;
    const std::uint16_t* norm    = nullptr; // BF16 [128]
    const std::int32_t* positions = nullptr;
    const DSparkRound* round     = nullptr; // positions null: the block's rows from round->first
    const float* inv_freq        = nullptr; // FP32 [64]: fl32(theta^(-i / 64)) (dspark_inv_freq)
    float eps = 1e-6F;
    std::uint16_t *cache_k = nullptr, *cache_v = nullptr; // a cache destination at the row's lane
    std::int64_t lane_stride = 0;
    const std::int32_t* lanes = nullptr;                  // [rows]: each row's lane (null: `lane`, or round->lane)
    std::int32_t lane = -1, ring = 0;                     // lane -1: round->lane
    // The full layer (ring 0): the cache is a page-major pool [pages][kv heads][64][128], position p of a lane at page
    // table[lane * table_stride + p / 64], row p % 64 (lane_stride unused).
    const std::int32_t* table = nullptr;
    std::int32_t table_stride = 0;
    // The full layer in INT8 (ring 0, values set; cache_k and cache_v then unused).
    DSparkInt8Cache int8;
    std::uint16_t* dense = nullptr;
    // Rows from rounds (positions null): 0, every row the round's; or the rows are blocks of block_rows rows, block i
    // the round at round + i (a round of several rows drafted in one launch), its rows from round[i].first.
    std::int32_t block_rows = 0;
};
void dspark_rotate(const DSparkRotateArgs& args, cudaStream_t stream);
// The RoPE table dspark_rotate reads: inv_freq[i] = fl32(theta^(-i / 64)) in FP64, i = 0 .. 63 (host).
void dspark_inv_freq(double theta, float (&inv_freq)[kDSparkHeadDim / 2]);

// The block's attention: the 8 query rows (positions F .. F + 7) against one layer's cached keys at positions
// [lo, F + 8) of the round's lane (the context before the block and the block's own 8, written there by dspark_rotate;
// every row sees all of them: the block is non-causal), lo = max(0, F - 1024) for a sliding layer, 0 for the full one.
// Query head h reads KV head h / 3. Scores q.k / sqrt(128) by BF16 MMAs (FP32 accumulation over the head dimension in
// order), softmax in FP32 (online over key splits of whole 32-key tiles, at least 128 keys, at most `splits` of them,
// merged in split order), P x V by BF16 MMAs over each p as two BF16 terms (hi = bf16(p), lo = bf16(p - hi); FP32
// accumulation), out = bf16 [8][24 * 128]. `splits` is the launch's (dspark_attention_splits): the split boundaries follow the context, so only
// the drafts' last bits depend on it. The full layer in INT8 (`int8`): the same over its represented words.
struct DSparkAttentionArgs {
    const std::uint16_t* q = nullptr; // BF16 [8][24][128] (normed, rotated)
    const std::uint16_t *cache_k = nullptr, *cache_v = nullptr;
    std::int64_t lane_stride = 0;
    std::int32_t ring        = 0; // 0: the full layer, its keys through the lane's block-table row (dspark_rotate's)
    const std::int32_t* table = nullptr;
    std::int32_t table_stride = 0;
    DSparkInt8Cache int8;         // the full layer in INT8 (ring 0; cache_k and cache_v then unused)
    std::int32_t splits      = 0;
    const DSparkRound* round = nullptr;
    float* scratch           = nullptr; // dspark_attention_scratch_bytes(splits)
    std::uint16_t* out       = nullptr;
    // Several blocks in one launch (a round of several rows): block b's round at round + b, its queries at
    // q + b q_stride, output at out + b out_stride and scratch at scratch + b scratch_stride (words, floats). Each
    // block's arithmetic is its launch's alone.
    std::int32_t blocks = 1;
    std::int64_t q_stride = 0, out_stride = 0, scratch_stride = 0;
};
void dspark_attention(const DSparkAttentionArgs& args, cudaStream_t stream);
[[nodiscard]] std::size_t dspark_attention_scratch_bytes(std::int32_t splits);
// The `splits` a layer's launch takes (and its scratch is sized for): a sliding layer's, or the full layer's at
// `max_context`.
[[nodiscard]] std::int32_t dspark_attention_splits(bool sliding, std::uint32_t max_context);

// The pre-norm block boundary: s = fp32(x) + fp32(residual) (x null: s = residual), residual = bf16(s) (unless x is
// null), normed = bf16(s * rsqrt(sum(s^2) * (1 / H) + eps) * w): the norm reads the FP32 sum.
struct DSparkAddNormArgs {
    std::uint16_t* residual   = nullptr; // BF16 [rows][H]
    const std::uint16_t* x    = nullptr; // BF16 [rows][H] or null
    const std::uint16_t* norm = nullptr; // BF16 [H]
    std::uint16_t* normed     = nullptr; // BF16 [rows][H]
    std::int32_t rows         = 0;
    float eps                 = 1e-6F;
};
void dspark_add_norm(const DSparkAddNormArgs& args, cudaStream_t stream);

// h = bf16(silu_c(g) * u) in FP32, g and u BF16 [rows][width].
void dspark_swiglu(const std::uint16_t* gate, const std::uint16_t* up, std::int32_t rows, std::int32_t width,
                   std::uint16_t* h, cudaStream_t stream);

// The Markov head (TokenSpeed's form): the step-invariant part of the joint projection first, then `steps` sequential
// steps. With s_0 = 0 and tok_0 = the anchor, step i (1 .. steps):
//   u   = bf16(invariant[i] + sum_k W[:, k] [s_{i-1}; w1[tok_{i-1}]]_k)   (FP32; W's first 2 rank columns)
//   g   = bf16(sigmoid_c(u_gate)); s_i = bf16(bf16(g s_{i-1}) + bf16(bf16(1 - g) bf16(tanh_c(u_cand))))
//   l   = bf16(tanh_c(u_out)); corr = bf16(sum_r Fold[v, r] l_r + fold_bias[v]) (FP32 sum)
//   z   = bf16(bf16(base[i][v]) + corr); L = bf16(30 tanh_c(z / 30)); tok_i = argmax L (the lowest id on ties)
//   prob_i = 1 / sum_v exp_c(L_v - L_max)  (the draft-length policy's confidence)
// invariant[i] = sum over W's hidden and feature columns of [x_out[i]; f_rnn] + bias (FP32, unrounded), x_out[i] the
// block's final row i, f_rnn the Markov context of position F - 1.
// Fold is BF16 (`fold`) or 8-bit (`fold_codes` + `fold_scales`: q8_g32.h's Q8_G32_FP16 RowSplit, rank 256 a row, an
// FP16 scale per 32: Fold[v, r] = code * scale, exact in FP32): 51 MB that stay in L2 across a round's steps, where
// the BF16 Fold's 102 MB are read from DRAM at every step. Only the drafts depend on it (a draft is kept only when it
// equals the target's own token), so no route's output does.
struct DSparkMarkovArgs {
    const std::uint16_t* projection = nullptr; // BF16 [768][2 rank + 2 H]
    const std::uint16_t* bias       = nullptr; // BF16 [768]
    const std::uint16_t* block      = nullptr; // BF16 [8][H]: the block's final rows (rows 1 .. steps read)
    const std::uint16_t* f_ring     = nullptr; // BF16 [f_ring_rows][H]: the Markov context by position % rows
    std::int32_t f_ring_rows        = 0;
    std::int64_t f_lane_stride      = 0;       // words between lanes' rings
    const std::uint16_t* embedding  = nullptr; // BF16 [V][rank] (markov w1)
    const std::uint16_t* fold       = nullptr; // BF16 [V][rank], or null with the 8-bit Fold below
    const std::int8_t* fold_codes   = nullptr; // Q8 [V][rank] codes
    const std::uint16_t* fold_scales = nullptr; // Q8 FP16 [V][rank / 32] scales
    const std::uint16_t* fold_bias  = nullptr; // BF16 [V]
    const float* base               = nullptr; // FP32 [steps][V]: the target head over the block's rows 1 .. steps
    std::int32_t vocab              = 0;       // V: the vocabulary, or a shortlist's rows (row_ids)
    const std::int32_t* row_ids     = nullptr; // a shortlist: each row's token id, ascending (null: rows are token ids)
    float softcap                   = 30.0F;
    const DSparkRound* round        = nullptr;
    std::int32_t* tokens            = nullptr; // [steps] out
    float* probabilities            = nullptr; // [steps] out
    float* scratch                  = nullptr; // dspark_markov_scratch_bytes
    // Several blocks in one launch (a round of several rows): block b's round at round + b, its final rows at
    // block + b block_stride, base at base + b base_stride, scratch at scratch + b scratch_stride (words, floats, bytes
    // / 4) and its drafts at tokens and probabilities + b kDSparkBlock. Each block's steps are its own.
    std::int32_t blocks = 1;
    std::int64_t block_stride = 0, base_stride = 0, scratch_stride = 0;
};
void dspark_markov(const DSparkMarkovArgs& args, std::int32_t steps, cudaStream_t stream);
[[nodiscard]] std::size_t dspark_markov_scratch_bytes(std::int32_t vocab);

// out[r][c] = src[ids[r]][c] for BF16 tables of `row_words` words a row (--lm-head-draft's tables, at load).
void dspark_gather_rows(const std::uint16_t* src, std::int32_t row_words, const std::int32_t* ids, std::int32_t rows,
                        std::uint16_t* out, cudaStream_t stream);

// The DFlash2 drafter's committed context columns of a verify call (models/inkling2/execution/dflash2.h): for each row
// b < rows and j < words[b] (its count, at most `width`), the call's column words[padded + b] + j (its first, then its
// next ones) to column b * width + j: `column_words` BF16 words (a multiple of 8, 16-byte aligned columns) of
// `features` into out_features, and its position into out_positions. Exact copies; other output columns unchanged.
void dflash2_gather_columns(const std::uint16_t* features, std::int32_t column_words, const std::int32_t* positions,
                            const std::int32_t* words, std::int32_t padded, std::int32_t rows, std::int32_t width,
                            std::uint16_t* out_features, std::int32_t* out_positions, cudaStream_t stream);

} // namespace infernix::ops::inkling2
