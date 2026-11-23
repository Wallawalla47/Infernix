#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <span>

namespace infernix::ops {

// The --lossless route's Inkling 2 operations (docs/maintainer/lossless-route.md §2b): each the operation of
// modeling_inkling2_loop.py (or, for the routed experts, the declared quantized arithmetic) with every step correctly
// rounded to its dtype. bf16(v) and fp32(v) are the correctly rounded value of the exact v; transcendental functions
// are correctly rounded binary32 (src/ops/lossless/cr_math.cuh); a softmax element is the correctly rounded value of
// its exact quotient. Tensors are [width, columns] with contiguous columns unless noted.

/**
 * RMSNorm over rows of x (BF16 or FP32 [width, rows]): m = fp32(mean(fp32(x^2))) (the sum exact),
 * r = fp32(rsqrt(fp32(m + eps))), y = fp32(x r); v = y (`weight` null: Inkling2LoopUnweightedRMSNorm) or fp32(w y)
 * (w BF16 [width]: Inkling2LoopRMSNorm); out (BF16 or FP32 [width, rows], its columns may be strided) = v or bf16(v).
 */
void lossless_inkling2_rmsnorm(const Tensor& x, const Tensor* weight, float eps, Tensor& out, cudaStream_t stream);

/**
 * A block attention residual (Inkling2LoopBlockAttentionResidual, Inkling2LoopMessageAttentionResidual) at every
 * column: sources s_j (j < 6; BF16 or FP32 [H, T]), n_j = fp32 rms(s_j) (eps 1e-6, as lossless_inkling2_rmsnorm);
 * per query q (FP32 [H], at most 4): l_j = fp32(fp32(q . n_j) fp32(H^-1/2)) (the dot product exact),
 * p = fp32(softmax(l)), out_q = fp32(sum_j p_j n_j) (exact, FP32 [H, T]).
 */
void lossless_inkling2_mix(std::span<const Tensor* const> sources, const Tensor& queries, std::span<Tensor* const> out,
                           cudaStream_t stream);

/**
 * A loop message gate's combine (Inkling2LoopMessageGate after its projections): from fl and gl (FP32 [H, T], the
 * forget and message projections of z, each exact sum rounded once), f = fp32(tanh(fl)), g = fp32(tanh(gl)),
 * out = fp32(fp32(fp32(1 - f) x) + fp32(g m)) (x and m FP32 [H, T]; out FP32, may alias x).
 */
void lossless_inkling2_message_gate(const Tensor& forget, const Tensor& gain, const Tensor& x, const Tensor& m,
                                    Tensor& out, cudaStream_t stream);

/**
 * The L2 clips over head rows of x (BF16 [D, heads, T], row r = t heads + h): y = fp32(x s_h) with s BF16 [heads, D]
 * (Inkling2LoopScaledL2Clip) or y = x (`scale` null: Inkling2LoopL2Clip); nu = fp32(||y||_2) (the sum of squares
 * exact, its square root correctly rounded); c = min(fp32(fp32(1 / fp32(nu + 1e-6)) fp32(t)), 1); o = bf16(fp32(y c)).
 * With log_scale.positions (I32 [T]; Inkling2LoopLogScaledL2Clip), out = bf16(fp32(o tau_t)),
 * tau = fp32(1 + fp32(fp32(alpha) fp32(ln(max(fp32(fp32(p + 1) / floor), 1))))); else out = o. out BF16 like x.
 */
struct LosslessLogScale {
    const Tensor* positions = nullptr;
    float alpha             = 0.0F;
    float floor             = 0.0F;
};
void lossless_inkling2_clip(const Tensor& x, const Tensor* scale, int heads, float threshold,
                            const LosslessLogScale& log_scale, Tensor& out, cudaStream_t stream);

/**
 * Inkling2LoopShortConvolution over one window (zeros before its first column): taps BF16 [k, C] lag-major (row 0
 * the current column), c_t = fp32(sum_{d < k} w[d] x_{t - d}) (exact), out = bf16(fp32(fp32(c_t + x_t) /
 * fp32(sqrt(k + 1)))). x BF16 [C, T] (its columns may be strided), out BF16 [C, T].
 */
void lossless_inkling2_conv(const Tensor& x, const Tensor& taps, Tensor& out, cudaStream_t stream);

/**
 * Inkling2LoopDynamicShortConvolution over one window: dynamic BF16 [heads k, T] (the per-column taps, the
 * exact projection rounded to BF16), taps BF16 [k, C] lag-major; w_{c,d} = fp32(taps[d, c] + dynamic[h(c) k + d])
 * with h(c) = c / (C / heads), c_t = fp32(sum_d fp32(x_{t-d,c} w_{c,d})) (zeros before the window; the sum exact),
 * out = bf16(fp32(fp32(c_t + x_t) / fp32(sqrt(2k + 1)))).
 */
void lossless_inkling2_dynamic_conv(const Tensor& x, const Tensor& dynamic, const Tensor& taps, int heads, Tensor& out,
                                    cudaStream_t stream);

/** The gated output projection's gate: out = bf16(fp32(o fp32(sigmoid(g)))), o and g BF16 [W, T]. */
void lossless_inkling2_sigmoid_gate(const Tensor& o, const Tensor& gate, Tensor& out, cudaStream_t stream);

/**
 * out = the exact a + b rounded to out's dtype (a, b BF16 or FP32 of one shape; out BF16 or FP32, may alias either),
 * or with `via_fp32` and a BF16 out, bf16(fp32(a + b)) (the residual's FP32 sum, then the layer's cast).
 */
void lossless_inkling2_add(const Tensor& a, const Tensor& b, Tensor& out, cudaStream_t stream, bool via_fp32 = false);

/** out = x rounded to out's dtype (BF16 or FP32, contiguous, one shape): `.float()` and `.to(bf16)`, zeros' signs kept. */
void lossless_inkling2_convert(const Tensor& x, Tensor& out, cudaStream_t stream);

/**
 * The attention probabilities of `head_count` consecutive heads from `head` in the eager form
 * (eager_attention_forward with position_bias and sinks), in place: scores (BF16, its columns `ld` apart) holds the
 * heads' query columns head-major, column (h - head) T + t that of head h and query position P_t = query_begin + t,
 * row j the key at position j, j < keys; it holds a_j = bf16(q . k_j) and receives p_j (0 off the visible keys):
 * a = bf16(fp32 a s) (s = fp32(scale)), a = bf16(a + B(d)) for d = P_t - j < extent with B(d) = bf16(sum_c r_c
 * table[c, d]) (r BF16 [rel, heads, T] the head's relative states at column t, table BF16 [rel, extent]; the sum exact),
 * then softmax over the visible keys (j <= P_t, and P_t - j < window when window > 0) and the sink (sinks BF16
 * [heads, T]): p_j = bf16(fp32(softmax_j)), each the correctly rounded quotient.
 */
struct LosslessInkling2Probs {
    Tensor* scores        = nullptr;
    int ld                = 0;
    int keys              = 0;
    int query_begin       = 0;
    int head              = 0;
    int head_count        = 1;
    int heads             = 0;
    const Tensor* relative = nullptr;
    const Tensor* table   = nullptr;
    const Tensor* sinks   = nullptr;
    int window            = 0;
    float scale           = 0.0F;
};
void lossless_inkling2_attention_probs(const LosslessInkling2Probs& a, cudaStream_t stream);

/**
 * Inkling2LoopTopKRouter from logits FP32 [E, T] (the exact x . W rounded once): s = fp32(sigmoid(l)); the top_k
 * experts by fp32(s + bias) (bias FP32 [E]; ties: the lower expert), in descending key order;
 * gamma_j = fp32(fp32(s_j / fp32(fp32(sum_k s_k) + 1e-20)) fp32(scaling)) (the sum exact). weights FP32 and ids I32
 * [top_k, T]. E <= 1024, top_k <= 16.
 */
void lossless_inkling2_router(const Tensor& logits, const Tensor& bias, int top_k, float scaling, Tensor& weights,
                              Tensor& ids, cudaStream_t stream);

/**
 * The experts' activation quantizers on the lossless route: each applies its rule (src/ops/common/canonical_math.h,
 * the declared arithmetic) to BF16 rows and writes the quantized values without their row factor, each exact in BF16.
 * Column c of out takes column in_columns[c] of x (null: c itself). Four-over-six: out = c s (E2M1 code times its
 * block's E4M3 scale) and outer (FP32 [T], out's columns) = amax fp32(1 / 1536); the value is out outer. MXFP8: out
 * = e4m3(c) 2^(e - 127), the value itself.
 */
void lossless_inkling2_quantize_four_over_six(const Tensor& x, const std::int32_t* in_columns, int columns, Tensor& out,
                                              Tensor& outer, cudaStream_t stream);
void lossless_inkling2_quantize_mxfp8(const Tensor& x, const std::int32_t* in_columns, int columns, Tensor& out,
                                      cudaStream_t stream);

/**
 * The softcapped SwiGLU times the routing weight: gate_up BF16 [2 I, C] with gate_i at row 2i and up_i at 2i + 1;
 * sc(v, cap) = fp32(fp32(tanh(fp32(v / cap))) cap); h = bf16(fp32(fp32(fp32(sc(g, gc) fp32(sigmoid(g))) sc(u, uc))
 * gamma_c)), gamma FP32 [C] (each column's weight). out BF16 [I, C].
 */
void lossless_inkling2_swiglu(const Tensor& gate_up, const Tensor& gamma, float gate_cap, float up_cap, Tensor& out,
                              cudaStream_t stream);

/**
 * The routed combine: out[:, t] = bf16(v), v folded from +0 over slots j < top_k in order, v = fp32(v + y_{t,j}),
 * y BF16 [H, entries] with entry slot_column[t top_k + j] (I32). out BF16 [H, T].
 */
void lossless_inkling2_combine(const Tensor& y, const Tensor& slot_column, int top_k, Tensor& out, cudaStream_t stream);

/**
 * The softmaxes (mix, attention probabilities) decide each element's rounding in binary64 or, where that interval
 * holds a midpoint, in double-double; an element even that leaves undecided (its quotient has no proven worst case)
 * is recorded. take_undecided returns whether one was since the last call (synchronous) and clears it; the route
 * fails its request then. force_exact_fallback (tests) sends every softmax through the double-double evaluation.
 */
[[nodiscard]] bool lossless_inkling2_take_undecided();
void lossless_inkling2_force_exact_fallback(bool on);

/** Inkling2LoopSoftcap on FP32 x: out = fp32(fp32(tanh(fp32(x / cap))) cap) (out may alias x). */
void lossless_inkling2_softcap(const Tensor& x, float cap, Tensor& out, cudaStream_t stream);

/**
 * Exact copies of a projection's head rows (x BF16 [D heads, T], column t's head h at rows h D ..) into head planes of
 * `out` (BF16, `heads` planes of plane_elements each): head-major, out[h][t][d] (plane h a contiguous [D, T'] matrix
 * of columns t); transposed, out[h][d][t] (plane h a [T', D] matrix: row d holds the columns' element d), with
 * T' = `stride` the plane's column count (>= T). Elements beyond T are left as they are.
 */
void lossless_inkling2_heads(const Tensor& x, int heads, bool transposed, int stride, Tensor& out, cudaStream_t stream);

// The cache form's state (generation, docs/maintainer/lossless-route.md §3a): exact copies between a call's tensors and
// a lane's stored history, in the layouts the reproducible route keeps (models/inkling2/execution/forward.h StateLayout,
// inkling2_attention.h), the lossless route's values bit for bit.

/**
 * A convolution's input ring: BF16 [rows][C], position p at row p % rows. ring_load writes positions first ..
 * first + count - 1 into columns 0 .. count - 1 of out (BF16 [C, *], columns may be strided; +0 for a negative
 * position: zeros before the sequence); ring_store writes x's columns (BF16 [C, T], column t position first + t,
 * first >= 0) into the ring, only the last min(T, rows) (the ring then holds the call's last rows).
 */
void lossless_inkling2_ring_load(const std::uint16_t* ring, int rows, int first, int count, Tensor& out,
                                 cudaStream_t stream);
void lossless_inkling2_ring_store(const Tensor& x, int first, std::uint16_t* ring, int rows, cudaStream_t stream);

/**
 * Stored keys and values: a row of `dim` (128) BF16 per (position, KV head) in units of 64 positions; a sliding
 * visit's ring [heads][ring_pages][64][dim] (position p in unit (p / 64) % ring_pages, row p % 64), or a full visit's
 * page-major pool [pages][heads][64][dim] through a lane's block-table row (`table`, device I32: position p in page
 * table[p / 64]).
 */
struct LosslessInkling2Keys {
    std::uint16_t* k          = nullptr;
    std::uint16_t* v          = nullptr;
    const std::int32_t* table = nullptr; // a full visit's block-table row; null: a ring
    int ring_pages            = 0;
    int heads                 = 0;
    int dim                   = 0;
};
/**
 * Positions first .. first + count - 1 (first >= 0) into head planes laid out as lossless_inkling2_heads writes them,
 * `stride` columns a plane: kh[h][j][d] (head-major) and vt[h][d][j] (transposed) for column j = position - first.
 */
void lossless_inkling2_keys_load(const LosslessInkling2Keys& source, int first, int count, int stride, Tensor& kh,
                                 Tensor& vt, cudaStream_t stream);
/** k and v (BF16 [dim heads, T], column t position first + t) into their rows (a ring: only the last min(T, its
 * positions)). */
void lossless_inkling2_keys_store(const Tensor& k, const Tensor& v, int first, const LosslessInkling2Keys& destination,
                                  cudaStream_t stream);

} // namespace infernix::ops
