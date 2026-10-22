#include "infernix/ops/lossless_ops.h"

#include "core/device.h"
#include "ops/lossless/cr_math.cuh"
#include "ops/lossless/wide.cuh"

#include <cuda_bf16.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

// The --lossless route's Qwen3.8-Flash-Next operations (include/infernix/ops/lossless_ops.h,
// docs/maintainer/lossless-route.md §2a): the hyper-connection gates, mix and injection, the PLE gate and
// its dilated convolution, the QSA indexer's pooled keys and block selection. Sums are exact (wide.cuh
// accumulators) and rounded once; a sum with an Inf or NaN term takes IEEE binary32 arithmetic, whose
// non-finite results (Inf of the terms' sign, NaN for NaN or Inf - Inf) are the exact operation's.

namespace infernix::ops {
namespace {

using lossless::F32Sum;
using lossless::round_magnitude;
using lossless::Wide;

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string("lossless Flash-Next op: ") + message); }
}

bool contiguous(const Tensor& t, DType dtype) { return t.dtype == dtype && t.is_contiguous(); }

unsigned blocks_for(std::size_t n, int threads) { return static_cast<unsigned>((n + threads - 1) / threads); }

__device__ __forceinline__ float bf16f(std::uint16_t bits) { return __uint_as_float(static_cast<std::uint32_t>(bits) << 16); }

__device__ __forceinline__ std::uint16_t bf16_rn(float v) { return __bfloat16_as_ushort(__float2bfloat16_rn(v)); }

__device__ __forceinline__ bool nonfinite(std::uint16_t bits) { return ((bits >> 7) & 0xFF) == 0xFF; }

// A sum of BF16 values M 2^(e - 134), exact at unit 2^-134 (positions below 255 + 63).
constexpr int kBf16SumWords = 10;
constexpr int kBf16Unit     = -134;

__device__ __forceinline__ void add_bf16(Wide<kBf16SumWords>& acc, std::uint16_t bits) {
    const int field = (bits >> 7) & 0xFF, mant = bits & 0x7F;
    const long long M = field == 0 ? mant : (128 | mant);
    if (M == 0) { return; }
    acc.add((bits & 0x8000U) != 0 ? -M : M, field == 0 ? 1 : field);
}

// --------------------------------------------------------------------------------- hyper-connection

// One thread per (row, column) of dj: rows [0, rank) give mix = bf16(silu(bf16(d / S))), rows
// [rank, rank + S) give inject = 2 bf16(sigmoid(bf16(j / S))).
__global__ void hc_gates_kernel(const std::uint16_t* dj, int rows, int rank, int streams, int columns,
                                std::uint16_t* mix, std::uint16_t* inject) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= static_cast<std::size_t>(rows) * columns) { return; }
    const int r = static_cast<int>(index % rows), t = static_cast<int>(index / rows);
    // d / S: the exact quotient (a power of two times a BF16 value, or a BF16 value over 3) rounded once.
    const std::uint16_t q = lossless::bf16_of_double(static_cast<double>(bf16f(dj[index])) / streams);
    if (r < rank) {
        mix[static_cast<std::size_t>(t) * rank + r] = lossless::cr_silu_bf16(bf16f(q));
    } else if (inject != nullptr) {
        // 2 times a BF16 value at most 1: exact.
        inject[static_cast<std::size_t>(t) * streams + (r - rank)] =
            bf16_rn(2.0F * bf16f(lossless::cr_sigmoid_bf16(bf16f(q))));
    }
}

// x_i = bf16(sum_c bf16(bf16(sigmoid(u_ci)) n_ci) / S), one thread per (i, column).
__global__ void hc_mix_kernel(const std::uint16_t* up, const std::uint16_t* normed, int width, int streams, int columns,
                              std::uint16_t* x) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= static_cast<std::size_t>(width) * columns) { return; }
    const int i = static_cast<int>(index % width), t = static_cast<int>(index / width);
    Wide<kBf16SumWords> acc;
    acc.clear();
    bool special = false;
    float fsum = 0.0F;
    for (int c = 0; c < streams; ++c) {
        const std::size_t at = (static_cast<std::size_t>(t) * streams + c) * width + i;
        const float g = bf16f(lossless::cr_sigmoid_bf16(bf16f(up[at])));
        const std::uint16_t p = bf16_rn(__fmul_rn(g, bf16f(normed[at]))); // the product exact in binary32
        special = special || nonfinite(p);
        fsum    = __fadd_rn(fsum, bf16f(p));
        add_bf16(acc, p);
    }
    if (special) {
        x[index] = bf16_rn(fsum / static_cast<float>(streams));
        return;
    }
    std::uint32_t mag[kBf16SumWords];
    const bool negative = acc.finish(mag);
    // Division by S = 4 (a power of two) moves the unit; another S divides exactly with a sticky bit.
    if ((streams & (streams - 1)) == 0) {
        x[index] = static_cast<std::uint16_t>(round_magnitude(mag, kBf16Unit - (31 - __clz(streams)), negative, true));
    } else {
        std::uint32_t q[kBf16SumWords + 2] = {};
        for (int w = 0; w < kBf16SumWords; ++w) { q[w + 2] = mag[w]; }
        const bool sticky = lossless::divide_small<kBf16SumWords + 2>(q, static_cast<std::uint32_t>(streams));
        x[index] = static_cast<std::uint16_t>(round_magnitude(q, kBf16Unit - 64, negative, true, sticky));
    }
}

// R_ci = bf16(R_ci + bf16(y_i j_c)), one thread per element of R.
__global__ void hc_inject_kernel(const std::uint16_t* y, const std::uint16_t* inject, int width, int streams,
                                 int columns, std::uint16_t* residual) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= static_cast<std::size_t>(width) * streams * columns) { return; }
    const int i = static_cast<int>(index % width);
    const std::size_t sc = index / width; // c + S t
    const int t = static_cast<int>(sc / streams), c = static_cast<int>(sc % streams);
    const float term = bf16f(bf16_rn(__fmul_rn(bf16f(y[static_cast<std::size_t>(t) * width + i]),
                                               bf16f(inject[static_cast<std::size_t>(t) * streams + c]))));
    residual[index] = lossless::bf16_of_double(static_cast<double>(bf16f(residual[index])) + static_cast<double>(term));
}

// --------------------------------------------------------------------------------- PLE

// One warp per (stream, column): g = bf16(sum_i bf16(k_i q_i)), then the scalar steps; s BF16 [S, T].
__global__ void ple_scores_kernel(const std::uint16_t* key, const std::uint16_t* query, int width, int streams,
                                  int columns, float inv_scale_divisor, std::uint16_t* s) {
    const int warp = static_cast<int>((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    const int lane = static_cast<int>(threadIdx.x) & 31;
    if (warp >= streams * columns) { return; }
    const std::size_t base = static_cast<std::size_t>(warp) * width; // (c + S t) * H
    Wide<kBf16SumWords> acc;
    acc.clear();
    bool special = false;
    float fsum = 0.0F;
    for (int i = lane; i < width; i += 32) {
        const std::uint16_t p = bf16_rn(__fmul_rn(bf16f(key[base + i]), bf16f(query[base + i])));
        if (nonfinite(p)) {
            special = true;
            fsum    = __fadd_rn(fsum, bf16f(p));
        }
        add_bf16(acc, p);
    }
    acc.warp_sum();
    special = __any_sync(0xFFFFFFFFU, special);
    if (special) {
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) { fsum = __fadd_rn(fsum, __shfl_xor_sync(0xFFFFFFFFU, fsum, o)); }
    }
    if (lane != 0) { return; }
    std::uint16_t g;
    if (special) {
        g = bf16_rn(fsum);
    } else {
        std::uint32_t mag[kBf16SumWords];
        const bool negative = acc.finish(mag);
        g = static_cast<std::uint16_t>(round_magnitude(mag, kBf16Unit, negative, true));
    }
    // g / fp32(sqrt(H)): a BF16 value over a binary32 value, never within 2^-33 of a BF16 midpoint unless
    // exactly on one, so the binary64 quotient rounds to the correctly rounded BF16.
    const double gq = static_cast<double>(bf16f(g)) / static_cast<double>(inv_scale_divisor);
    g = lossless::bf16_of_double(gq);
    const float gv = bf16f(g);
    // sqrt(bf16(max(|g|, fp32(1e-6)))) sign(g): the BF16 square root of a BF16 value, correctly rounded
    // through binary64 (an irrational root lies far from a midpoint; a rational one is exact).
    const float a = fmaxf(fabsf(gv), 1e-6F);
    const std::uint16_t clamped = isnan(gv) ? static_cast<std::uint16_t>(0x7FC0U) : bf16_rn(a);
    const std::uint16_t root = lossless::bf16_of_double(sqrt(static_cast<double>(bf16f(clamped))));
    const float sign = isnan(gv) ? NAN : (gv > 0.0F ? 1.0F : (gv < 0.0F ? -1.0F : 0.0F));
    s[warp] = bf16_rn(bf16f(root) * sign);
}

// gated_ci = bf16(bf16(sigmoid(s_c)) v_i), one thread per element.
__global__ void ple_gated_kernel(const std::uint16_t* s, const std::uint16_t* value, int width, int streams, int columns,
                                 std::uint16_t* gated) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= static_cast<std::size_t>(width) * streams * columns) { return; }
    const int i = static_cast<int>(index % width);
    const std::size_t sc = index / width;
    const int t = static_cast<int>(sc / streams);
    const float g = bf16f(lossless::cr_sigmoid_bf16(bf16f(s[sc])));
    gated[index] = bf16_rn(__fmul_rn(g, bf16f(value[static_cast<std::size_t>(t) * width + i])));
}

// Products of two BF16 values, M 2^(e - 268), exact at unit 2^-268.
constexpr int kProductWords = 18;
constexpr int kProductUnit  = -268;

// conv = bf16(sum_j w[j C + c] n[c, t - (K - 1 - j) D]); out = bf16(gated + bf16(silu(conv))); R += out.
// Columns form runs of `width` (one per sequence); with `history` (BF16 [C, span, slots], span =
// (K - 1) dilation, oldest first) a run's taps before its first column read its source slot, else zeros.
__global__ void ple_conv_inject_kernel(const std::uint16_t* gated, const std::uint16_t* normalized,
                                       const std::uint16_t* weight, int channels, int taps, int dilation, int columns,
                                       std::uint16_t* residual, int width, const std::uint16_t* history,
                                       const std::int32_t* source_slots) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= static_cast<std::size_t>(channels) * columns) { return; }
    const int c = static_cast<int>(index % channels), t = static_cast<int>(index / channels);
    const int b = t / width, run = t % width, span = (taps - 1) * dilation;
    Wide<kProductWords> acc;
    acc.clear();
    bool special = false;
    float fsum = 0.0F;
    for (int j = 0; j < taps; ++j) {
        const int src = run - (taps - 1 - j) * dilation;
        if (src < 0 && history == nullptr) { continue; }
        const std::uint16_t wb = weight[static_cast<std::size_t>(j) * channels + c];
        const std::uint16_t nb =
            src >= 0 ? normalized[(static_cast<std::size_t>(b) * width + src) * channels + c]
                     : history[(static_cast<std::size_t>(source_slots[b]) * span + span + src) * channels + c];
        if (nonfinite(wb) || nonfinite(nb)) { special = true; }
        fsum = __fadd_rn(fsum, __fmul_rn(bf16f(wb), bf16f(nb)));
        const int fw = (wb >> 7) & 0xFF, mw = wb & 0x7F, fn = (nb >> 7) & 0xFF, mn = nb & 0x7F;
        const long long Mw = fw == 0 ? mw : (128 | mw), Mn = fn == 0 ? mn : (128 | mn);
        if (special || Mw == 0 || Mn == 0) { continue; }
        const long long p = Mw * Mn;
        acc.add(((wb ^ nb) & 0x8000U) != 0 ? -p : p, (fw == 0 ? 1 : fw) + (fn == 0 ? 1 : fn));
    }
    std::uint16_t conv;
    if (special) {
        conv = bf16_rn(fsum);
    } else {
        std::uint32_t mag[kProductWords];
        const bool negative = acc.finish(mag);
        conv = static_cast<std::uint16_t>(round_magnitude(mag, kProductUnit, negative, true));
    }
    const float act = bf16f(lossless::cr_silu_bf16(bf16f(conv)));
    const std::uint16_t out = lossless::bf16_of_double(static_cast<double>(bf16f(gated[index])) + static_cast<double>(act));
    residual[index] = lossless::bf16_of_double(static_cast<double>(bf16f(residual[index])) + static_cast<double>(bf16f(out)));
}

// One thread per (channel, sequence): the destination slot's span columns become the last span of the
// source slot's followed by the run, in ascending order (an in-place slot reads each old column before
// it is overwritten).
__global__ void ple_history_kernel(const std::uint16_t* normalized, std::uint16_t* history, const std::int32_t* source_slots,
                                   const std::int32_t* destination_slots, int channels, int sequences, int width, int span) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= channels * sequences) { return; }
    const int c = i % channels, b = i / channels;
    const std::uint16_t* src = history + static_cast<std::size_t>(source_slots[b]) * span * channels + c;
    std::uint16_t* dst       = history + static_cast<std::size_t>(destination_slots[b]) * span * channels + c;
    for (int k = 0; k < span; ++k) {
        const int s = width + k - span; // the run's column, or (negative) the old history's column k + width
        dst[static_cast<std::size_t>(k) * channels] =
            s >= 0 ? normalized[(static_cast<std::size_t>(b) * width + s) * channels + c]
                   : src[static_cast<std::size_t>(k + width) * channels];
    }
}

// --------------------------------------------------------------------------------- QSA indexer

// bf16(fp32(sum of key(0..ratio-1) / ratio)) (HF's .float().mean().to(bf16); the sum exact).
template <class Key>
__device__ std::uint16_t pooled_mean(Key key, int ratio) {
    Wide<kBf16SumWords> acc;
    acc.clear();
    bool special = false;
    float fsum = 0.0F;
    for (int i = 0; i < ratio; ++i) {
        const std::uint16_t k = key(i);
        special = special || nonfinite(k);
        fsum    = __fadd_rn(fsum, bf16f(k));
        add_bf16(acc, k);
    }
    float mean;
    if (special) {
        mean = fsum / static_cast<float>(ratio);
    } else {
        std::uint32_t mag[kBf16SumWords];
        const bool negative = acc.finish(mag);
        if ((ratio & (ratio - 1)) == 0) {
            mean = __uint_as_float(round_magnitude(mag, kBf16Unit - (31 - __clz(ratio)), negative, false));
        } else {
            std::uint32_t q[kBf16SumWords + 2] = {};
            for (int w = 0; w < kBf16SumWords; ++w) { q[w + 2] = mag[w]; }
            const bool sticky = lossless::divide_small<kBf16SumWords + 2>(q, static_cast<std::uint32_t>(ratio));
            mean = __uint_as_float(round_magnitude(q, kBf16Unit - 64, negative, false, sticky));
        }
    }
    return bf16_rn(mean); // the binary32 mean's BF16 rounding
}

// pooled[:, b] = the mean of keys [r b, r b + r), one thread per (d, b).
__global__ void qsa_pool_kernel(const std::uint16_t* keys, int dim, int ratio, int blocks, std::uint16_t* pooled) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= static_cast<std::size_t>(dim) * blocks) { return; }
    const int d = static_cast<int>(index % dim), b = static_cast<int>(index / dim);
    pooled[index] = pooled_mean(
        [&](int i) { return keys[(static_cast<std::size_t>(b) * ratio + i) * dim + d]; }, ratio);
}

// The cache form (runs of `width` consecutive positions): a column t at position P that completes a block
// (P + 1 multiple of ratio) gets its block's mean, the keys before the run's first position P0 from its
// tail slot (slot P mod ratio); other columns get zeros. Thread d == 0 also writes the column's block-start
// RoPE position (axis-major [T, 3]): its own column's when the block starts in the run, else the run's
// block_start_rope.
__global__ void qsa_pool_cache_kernel(const std::uint16_t* keys, const std::int32_t* positions, const std::uint16_t* tails,
                                      const std::int32_t* tail_slots, const std::int32_t* rope_positions,
                                      const std::int32_t* block_start_rope, int dim, int ratio, int width, int sequences,
                                      std::uint16_t* pooled, std::int32_t* block_rope) {
    const int columns = width * sequences;
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= static_cast<std::size_t>(dim) * columns) { return; }
    const int d = static_cast<int>(index % dim), t = static_cast<int>(index / dim);
    const int b = t / width, p = positions[t], p0 = positions[b * width];
    const bool complete = (p + 1) % ratio == 0;
    if (d == 0) {
        const int start = p + 1 - ratio; // the completed block's first position
        for (int a = 0; a < 3; ++a) {
            block_rope[a * columns + t] = !complete      ? rope_positions[a * columns + t]
                                          : start >= p0 ? rope_positions[a * columns + t - (p - start)]
                                                        : block_start_rope[a * sequences + b];
        }
    }
    if (!complete) {
        pooled[index] = 0;
        return;
    }
    const std::uint16_t* tail = tails + static_cast<std::size_t>(tail_slots[b]) * (ratio - 1) * dim + d;
    pooled[index] = pooled_mean(
        [&](int i) {
            const int q = p + 1 - ratio + i;
            return q >= p0 ? keys[static_cast<std::size_t>(t - (p - q)) * dim + d]
                           : tail[static_cast<std::size_t>(q % ratio) * dim];
        },
        ratio);
}

// Block (P + 1) / ratio - 1 of each completing column into the pooled plane ([dim / ratio, 64, 1, pages]:
// a block's dim values from its first token's slot of its page).
__global__ void qsa_pooled_store_kernel(const std::uint16_t* pooled, const std::int32_t* positions,
                                        const std::int32_t* tables, const std::int32_t* table_rows, int stride, int dim,
                                        int ratio, int width, int columns, std::uint16_t* plane) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= static_cast<std::size_t>(dim) * columns) { return; }
    const int d = static_cast<int>(index % dim), t = static_cast<int>(index / dim);
    const int p = positions[t];
    if ((p + 1) % ratio != 0) { return; }
    const int first = p + 1 - ratio; // the block's first position
    const int page  = tables[static_cast<std::size_t>(table_rows[t / width]) * stride + first / 64];
    plane[(static_cast<std::size_t>(page) * 64 + first % 64) * (dim / ratio) + d] = pooled[index];
}

// The pooled keys of blocks [0, blocks) of row table_rows[sequence] into out ([dim, blocks]); blocks past
// the sequence's complete ones (after its run's last position) are zeros.
__global__ void qsa_pooled_gather_kernel(const std::uint16_t* plane, const std::int32_t* tables,
                                         const std::int32_t* table_rows, const std::int32_t* positions, int width,
                                         int sequence, int stride, int dim, int ratio, int blocks, std::uint16_t* out) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= static_cast<std::size_t>(dim) * blocks) { return; }
    const int d = static_cast<int>(index % dim), blk = static_cast<int>(index / dim);
    if (blk >= (positions[sequence * width + width - 1] + 1) / ratio) {
        out[index] = 0;
        return;
    }
    const int first = blk * ratio;
    const int page  = tables[static_cast<std::size_t>(table_rows[sequence]) * stride + first / 64];
    out[index]      = plane[(static_cast<std::size_t>(page) * 64 + first % 64) * (dim / ratio) + d];
}

// The open block's keys after the run (positions from R floor((P_last + 1) / R)) into the tail slots.
__global__ void qsa_tails_store_kernel(const std::uint16_t* keys, const std::int32_t* positions, std::uint16_t* tails,
                                       const std::int32_t* tail_slots, int dim, int ratio, int width, int columns) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= static_cast<std::size_t>(dim) * columns) { return; }
    const int d = static_cast<int>(index % dim), t = static_cast<int>(index / dim);
    const int b = t / width, p = positions[t];
    const int open = (positions[b * width + width - 1] + 1) / ratio * ratio;
    if (p < open) { return; }
    tails[(static_cast<std::size_t>(tail_slots[b]) * (ratio - 1) + p % ratio) * dim + d] = keys[index];
}

constexpr int kSelectThreads = 256;
constexpr int kSelectWords   = 2048; // 65,536 blocks (262,144 keys at ratio 4)

// One CTA per query t: n = floor((t + 1) / ratio) complete blocks; all when n <= budget, else the budget
// largest of fp32(fp32(sum_h relu(score)) / divisor) (NaN largest, ties the lower block).
__global__ void __launch_bounds__(kSelectThreads)
    qsa_select_kernel(const float* scores, int blocks, int heads, int ratio, int budget, float divisor, int columns,
                      std::uint32_t* select, int words, std::uint32_t* keys_scratch, const std::int32_t* positions) {
    __shared__ std::uint32_t bits[kSelectWords];
    __shared__ int counts[kSelectThreads];
    const int t = static_cast<int>(blockIdx.x), tid = static_cast<int>(threadIdx.x);
    for (int w = tid; w < words; w += kSelectThreads) { bits[w] = 0; }
    __syncthreads();
    const int n = ((positions != nullptr ? positions[t] : t) + 1) / ratio;
    if (n <= budget) {
        for (int b = tid; b < n; b += kSelectThreads) { atomicOr(&bits[b / 32], 1U << (b % 32)); }
    } else {
        std::uint32_t* keys = keys_scratch + static_cast<std::size_t>(t) * blocks;
        for (int b = tid; b < n; b += kSelectThreads) {
            F32Sum sum;
            sum.clear();
            bool has_nan = false;
            for (int h = 0; h < heads; ++h) {
                const float s = scores[(static_cast<std::size_t>(t) * heads + h) * blocks + b];
                if (isnan(s)) { has_nan = true; }
                if (s > 0.0F) { sum.add(s); } // relu
            }
            const float v = has_nan ? NAN : __fdiv_rn(sum.result(), divisor);
            // Nonnegative binary32 values order as their bits (-0 is +0; NaN above every value).
            keys[b] = isnan(v) ? 0xFFFFFFFFU : (v == 0.0F ? 0U : __float_as_uint(v));
        }
        __syncthreads();
        // The largest V with at least `budget` keys >= V.
        std::uint32_t lo = 0U, hi = 0xFFFFFFFFU;
        while (lo < hi) {
            const std::uint32_t mid = lo + (hi - lo) / 2 + 1;
            int c = 0;
            for (int b = tid; b < n; b += kSelectThreads) { c += keys[b] >= mid ? 1 : 0; }
            counts[tid] = c;
            __syncthreads();
            for (int o = kSelectThreads / 2; o > 0; o >>= 1) {
                if (tid < o) { counts[tid] += counts[tid + o]; }
                __syncthreads();
            }
            const int total = counts[0];
            __syncthreads();
            if (total >= budget) {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        const std::uint32_t threshold = lo;
        // Keys above the threshold, then the lowest-indexed keys equal to it until budget: each thread
        // takes a contiguous range so its equal keys' rank follows from the threads before it.
        const int per = (n + kSelectThreads - 1) / kSelectThreads;
        const int first = tid * per, last = min(n, first + per);
        int above = 0, equal = 0;
        for (int b = first; b < last; ++b) {
            above += keys[b] > threshold ? 1 : 0;
            equal += keys[b] == threshold ? 1 : 0;
        }
        counts[tid] = above;
        __syncthreads();
        for (int o = kSelectThreads / 2; o > 0; o >>= 1) {
            if (tid < o) { counts[tid] += counts[tid + o]; }
            __syncthreads();
        }
        const int need = budget - counts[0];
        __syncthreads();
        counts[tid] = equal;
        __syncthreads();
        // Exclusive scan of the equal counts (Hillis-Steele in place).
        for (int o = 1; o < kSelectThreads; o <<= 1) {
            const int v = tid >= o ? counts[tid - o] : 0;
            __syncthreads();
            counts[tid] += v;
            __syncthreads();
        }
        int rank = counts[tid] - equal;
        for (int b = first; b < last; ++b) {
            bool take = keys[b] > threshold;
            if (keys[b] == threshold) {
                take = rank < need;
                ++rank;
            }
            if (take) { atomicOr(&bits[b / 32], 1U << (b % 32)); }
        }
    }
    __syncthreads();
    for (int w = tid; w < words; w += kSelectThreads) { select[static_cast<std::size_t>(t) * words + w] = bits[w]; }
}

__global__ void permute_blocks_kernel(const std::uint16_t* x, const std::int32_t* perm, int block, int k, int columns,
                                      std::uint16_t* out) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= static_cast<std::size_t>(k) * columns) { return; }
    const int r = static_cast<int>(index % k);
    const std::size_t t = index / k;
    out[index] = x[t * k + static_cast<std::size_t>(perm[r / block]) * block + r % block];
}

} // namespace

void lossless_permute_blocks(const Tensor& x, const std::int32_t* perm, int block, Tensor& out, cudaStream_t stream) {
    require(contiguous(x, DType::BF16) && contiguous(out, DType::BF16) && x.ne[0] == out.ne[0] && x.ne[1] == out.ne[1] &&
                perm != nullptr && block > 0 && x.ne[0] % block == 0,
            "x and out BF16 [K, T], K a multiple of the block");
    const std::size_t n = static_cast<std::size_t>(x.ne[0]) * x.ne[1];
    permute_blocks_kernel<<<blocks_for(n, 256), 256, 0, stream>>>(static_cast<const std::uint16_t*>(x.data), perm, block,
                                                                  x.ne[0], x.ne[1], static_cast<std::uint16_t*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

void lossless_hc_gates(const Tensor& dj, int rank, int streams, Tensor& mix, Tensor* inject, cudaStream_t stream) {
    const int rows = dj.ne[0], columns = dj.ne[1];
    require(contiguous(dj, DType::BF16) && contiguous(mix, DType::BF16) && mix.ne[0] == rank && mix.ne[1] == columns &&
                streams > 0 && (rows == rank || rows == rank + streams),
            "dj BF16 [rank (+ S), T], mix BF16 [rank, T]");
    require(inject == nullptr || (rows == rank + streams && contiguous(*inject, DType::BF16) && inject->ne[0] == streams &&
                                  inject->ne[1] == columns),
            "inject BF16 [S, T] from the injection rows");
    const std::size_t n = static_cast<std::size_t>(rows) * columns;
    hc_gates_kernel<<<blocks_for(n, 256), 256, 0, stream>>>(
        static_cast<const std::uint16_t*>(dj.data), rows, rank, streams, columns, static_cast<std::uint16_t*>(mix.data),
        inject != nullptr ? static_cast<std::uint16_t*>(inject->data) : nullptr);
    CUDA_CHECK(cudaGetLastError());
}

void lossless_hc_mix(const Tensor& up, const Tensor& normed, int streams, Tensor& x, cudaStream_t stream) {
    const int width = x.ne[0], columns = x.ne[1];
    require(contiguous(up, DType::BF16) && contiguous(normed, DType::BF16) && contiguous(x, DType::BF16) &&
                up.ne[0] == streams * width && normed.ne[0] == streams * width && up.ne[1] == columns &&
                normed.ne[1] == columns,
            "up and normed BF16 [S*H, T], x BF16 [H, T]");
    const std::size_t n = static_cast<std::size_t>(width) * columns;
    hc_mix_kernel<<<blocks_for(n, 256), 256, 0, stream>>>(static_cast<const std::uint16_t*>(up.data),
                                                          static_cast<const std::uint16_t*>(normed.data), width,
                                                          streams, columns, static_cast<std::uint16_t*>(x.data));
    CUDA_CHECK(cudaGetLastError());
}

void lossless_hc_inject(const Tensor& y, const Tensor& inject, Tensor& residual, cudaStream_t stream) {
    const int width = y.ne[0], columns = y.ne[1], streams = inject.ne[0];
    require(contiguous(y, DType::BF16) && contiguous(inject, DType::BF16) && contiguous(residual, DType::BF16) &&
                inject.ne[1] == columns && residual.ne[0] == streams * width && residual.ne[1] == columns,
            "y BF16 [H, T], inject BF16 [S, T], residual BF16 [S*H, T]");
    const std::size_t n = static_cast<std::size_t>(width) * streams * columns;
    hc_inject_kernel<<<blocks_for(n, 256), 256, 0, stream>>>(static_cast<const std::uint16_t*>(y.data),
                                                             static_cast<const std::uint16_t*>(inject.data), width,
                                                             streams, columns, static_cast<std::uint16_t*>(residual.data));
    CUDA_CHECK(cudaGetLastError());
}

void lossless_ple_gate(const Tensor& key_normed, const Tensor& query_normed, const Tensor& value, int streams,
                       Tensor& scores, Tensor& gated, cudaStream_t stream) {
    const int width = value.ne[0], columns = value.ne[1];
    require(contiguous(key_normed, DType::BF16) && contiguous(query_normed, DType::BF16) &&
                contiguous(value, DType::BF16) && contiguous(gated, DType::BF16) && contiguous(scores, DType::BF16) &&
                key_normed.ne[0] == streams * width && query_normed.ne[0] == streams * width &&
                gated.ne[0] == streams * width && key_normed.ne[1] == columns && query_normed.ne[1] == columns &&
                gated.ne[1] == columns && scores.ne[0] == streams && scores.ne[1] == columns,
            "keys, queries and gated BF16 [S*H, T], value BF16 [H, T], scores BF16 [S, T]");
    const auto divisor = static_cast<float>(std::sqrt(static_cast<double>(width))); // math.sqrt(H) as binary32
    const int warps = streams * columns;
    ple_scores_kernel<<<blocks_for(static_cast<std::size_t>(warps) * 32, 256), 256, 0, stream>>>(
        static_cast<const std::uint16_t*>(key_normed.data), static_cast<const std::uint16_t*>(query_normed.data), width,
        streams, columns, divisor, static_cast<std::uint16_t*>(scores.data));
    CUDA_CHECK(cudaGetLastError());
    const std::size_t n = static_cast<std::size_t>(width) * streams * columns;
    ple_gated_kernel<<<blocks_for(n, 256), 256, 0, stream>>>(static_cast<const std::uint16_t*>(scores.data),
                                                             static_cast<const std::uint16_t*>(value.data), width,
                                                             streams, columns, static_cast<std::uint16_t*>(gated.data));
    CUDA_CHECK(cudaGetLastError());
}

void lossless_ple_conv_inject(const Tensor& gated, const Tensor& normalized, const Tensor& weight, int dilation,
                              Tensor& residual, cudaStream_t stream) {
    const int channels = gated.ne[0], columns = gated.ne[1];
    require(contiguous(gated, DType::BF16) && contiguous(normalized, DType::BF16) && contiguous(weight, DType::BF16) &&
                contiguous(residual, DType::BF16) && normalized.ne[0] == channels && normalized.ne[1] == columns &&
                residual.ne[0] == channels && residual.ne[1] == columns && weight.ne[0] == channels && dilation > 0,
            "gated, normalized and residual BF16 [C, T], weight BF16 [C, K]");
    const int taps = weight.ne[1];
    const std::size_t n = static_cast<std::size_t>(channels) * columns;
    if (n == 0) { return; }
    ple_conv_inject_kernel<<<blocks_for(n, 256), 256, 0, stream>>>(
        static_cast<const std::uint16_t*>(gated.data), static_cast<const std::uint16_t*>(normalized.data),
        static_cast<const std::uint16_t*>(weight.data), channels, taps, dilation, columns,
        static_cast<std::uint16_t*>(residual.data), columns, nullptr, nullptr);
    CUDA_CHECK(cudaGetLastError());
}

void lossless_ple_conv_inject_slots(const Tensor& gated, const Tensor& normalized, const Tensor& weight, int dilation,
                                    int sequences, Tensor& history, const Tensor& source_slots,
                                    const Tensor& destination_slots, Tensor& residual, cudaStream_t stream) {
    const int channels = gated.ne[0], columns = gated.ne[1];
    require(contiguous(gated, DType::BF16) && contiguous(normalized, DType::BF16) && contiguous(weight, DType::BF16) &&
                contiguous(residual, DType::BF16) && normalized.ne[0] == channels && normalized.ne[1] == columns &&
                residual.ne[0] == channels && residual.ne[1] == columns && weight.ne[0] == channels && dilation > 0 &&
                weight.ne[1] >= 2,
            "gated, normalized and residual BF16 [C, T], weight BF16 [C, K >= 2]");
    const int taps = weight.ne[1], span = (taps - 1) * dilation;
    require(sequences >= 1 && columns % sequences == 0, "the columns are `sequences` equal runs");
    require(contiguous(history, DType::BF16) && history.ne[0] == channels && history.ne[1] == span,
            "history BF16 [C, (K - 1) dilation, slots]");
    require(source_slots.dtype == DType::I32 && destination_slots.dtype == DType::I32 &&
                source_slots.is_contiguous() && destination_slots.is_contiguous() &&
                source_slots.numel() >= sequences && destination_slots.numel() >= sequences,
            "source and destination slots I32 [B]");
    if (columns == 0) { return; }
    const std::size_t n = static_cast<std::size_t>(channels) * columns;
    ple_conv_inject_kernel<<<blocks_for(n, 256), 256, 0, stream>>>(
        static_cast<const std::uint16_t*>(gated.data), static_cast<const std::uint16_t*>(normalized.data),
        static_cast<const std::uint16_t*>(weight.data), channels, taps, dilation, columns,
        static_cast<std::uint16_t*>(residual.data), columns / sequences, static_cast<const std::uint16_t*>(history.data),
        static_cast<const std::int32_t*>(source_slots.data));
    CUDA_CHECK(cudaGetLastError());
    ple_history_kernel<<<blocks_for(static_cast<std::size_t>(channels) * sequences, 256), 256, 0, stream>>>(
        static_cast<const std::uint16_t*>(normalized.data), static_cast<std::uint16_t*>(history.data),
        static_cast<const std::int32_t*>(source_slots.data), static_cast<const std::int32_t*>(destination_slots.data),
        channels, sequences, columns / sequences, span);
    CUDA_CHECK(cudaGetLastError());
}

void lossless_qsa_pool(const Tensor& keys, int ratio, Tensor& pooled, cudaStream_t stream) {
    const int dim = keys.ne[0], blocks = pooled.ne[1];
    require(contiguous(keys, DType::BF16) && contiguous(pooled, DType::BF16) && pooled.ne[0] == dim && ratio > 0 &&
                blocks == keys.ne[1] / ratio,
            "keys BF16 [D, T], pooled BF16 [D, T / ratio]");
    if (blocks == 0) { return; }
    const std::size_t n = static_cast<std::size_t>(dim) * blocks;
    qsa_pool_kernel<<<blocks_for(n, 256), 256, 0, stream>>>(static_cast<const std::uint16_t*>(keys.data), dim, ratio,
                                                            blocks, static_cast<std::uint16_t*>(pooled.data));
    CUDA_CHECK(cudaGetLastError());
}

namespace {

void require_runs(const Tensor& keys, const Tensor& positions, int sequences, int ratio) {
    require(contiguous(keys, DType::BF16) && positions.dtype == DType::I32 && positions.is_contiguous() &&
                positions.numel() == keys.ne[1] && sequences >= 1 && keys.ne[1] % sequences == 0 && ratio > 0 &&
                64 % ratio == 0 && keys.ne[0] % ratio == 0,
            "keys BF16 [D, T] in `sequences` runs of consecutive positions (I32 [T]); ratio divides 64 and D");
}

void require_tails(const Tensor& tails, const Tensor& tail_slots, int dim, int ratio, int sequences) {
    require(contiguous(tails, DType::BF16) && tails.ne[0] == dim && tails.ne[1] == ratio - 1 &&
                tail_slots.dtype == DType::I32 && tail_slots.is_contiguous() && tail_slots.numel() >= sequences,
            "tails BF16 [D, ratio - 1, slots] and tail slots I32 [B]");
}

void require_plane(const Tensor& plane, const Tensor& block_tables, int dim, int ratio) {
    require(contiguous(plane, DType::BF16) && plane.ne[0] == dim / ratio && plane.ne[1] == 64 &&
                block_tables.dtype == DType::I32 && block_tables.is_contiguous(),
            "pooled plane BF16 [D / ratio, 64, 1, pages] and block tables I32 [entries, rows]");
}

} // namespace

void lossless_qsa_pool_cache(const Tensor& keys, const Tensor& positions, int sequences, const Tensor& tails,
                             const Tensor& tail_slots, const Tensor& rope_positions, const Tensor& block_start_rope,
                             int ratio, Tensor& pooled, Tensor& block_rope, cudaStream_t stream) {
    require_runs(keys, positions, sequences, ratio);
    const int dim = keys.ne[0], columns = keys.ne[1];
    require_tails(tails, tail_slots, dim, ratio, sequences);
    require(contiguous(pooled, DType::BF16) && pooled.ne[0] == dim && pooled.ne[1] == columns &&
                rope_positions.dtype == DType::I32 && rope_positions.numel() == 3 * columns &&
                block_start_rope.dtype == DType::I32 && block_start_rope.numel() >= 3 * sequences &&
                block_rope.dtype == DType::I32 && block_rope.numel() == 3 * columns,
            "pooled BF16 [D, T]; RoPE positions, block-start RoPE and block_rope I32 axis-major");
    if (columns == 0) { return; }
    const std::size_t n = static_cast<std::size_t>(dim) * columns;
    qsa_pool_cache_kernel<<<blocks_for(n, 256), 256, 0, stream>>>(
        static_cast<const std::uint16_t*>(keys.data), static_cast<const std::int32_t*>(positions.data),
        static_cast<const std::uint16_t*>(tails.data), static_cast<const std::int32_t*>(tail_slots.data),
        static_cast<const std::int32_t*>(rope_positions.data), static_cast<const std::int32_t*>(block_start_rope.data),
        dim, ratio, columns / sequences, sequences, static_cast<std::uint16_t*>(pooled.data),
        static_cast<std::int32_t*>(block_rope.data));
    CUDA_CHECK(cudaGetLastError());
}

void lossless_qsa_pooled_store(const Tensor& pooled, const Tensor& positions, int sequences, const Tensor& block_tables,
                               const Tensor& table_rows, int ratio, Tensor& plane, cudaStream_t stream) {
    require_runs(pooled, positions, sequences, ratio);
    const int dim = pooled.ne[0], columns = pooled.ne[1];
    require_plane(plane, block_tables, dim, ratio);
    require(table_rows.dtype == DType::I32 && table_rows.numel() >= sequences, "table rows I32 [B]");
    if (columns == 0) { return; }
    const std::size_t n = static_cast<std::size_t>(dim) * columns;
    qsa_pooled_store_kernel<<<blocks_for(n, 256), 256, 0, stream>>>(
        static_cast<const std::uint16_t*>(pooled.data), static_cast<const std::int32_t*>(positions.data),
        static_cast<const std::int32_t*>(block_tables.data), static_cast<const std::int32_t*>(table_rows.data),
        block_tables.ne[0], dim, ratio, columns / sequences, columns, static_cast<std::uint16_t*>(plane.data));
    CUDA_CHECK(cudaGetLastError());
}

void lossless_qsa_pooled_gather(const Tensor& plane, const Tensor& block_tables, const Tensor& table_rows,
                                const Tensor& positions, int sequences, int sequence, int ratio, Tensor& out,
                                cudaStream_t stream) {
    const int dim = out.ne[0], blocks = out.ne[1];
    require(positions.dtype == DType::I32 && positions.is_contiguous() && sequences >= 1 &&
                positions.numel() % sequences == 0,
            "positions I32 [T] of `sequences` equal runs");
    require(contiguous(out, DType::BF16) && ratio > 0 && dim % ratio == 0 && table_rows.dtype == DType::I32 &&
                sequence >= 0 && sequence < sequences && sequence < table_rows.numel() &&
                static_cast<std::int64_t>(blocks) * ratio <= static_cast<std::int64_t>(block_tables.ne[0]) * 64,
            "out BF16 [D, blocks] within the block table, table rows I32 [B]");
    require_plane(plane, block_tables, dim, ratio);
    if (blocks == 0) { return; }
    const std::size_t n = static_cast<std::size_t>(dim) * blocks;
    qsa_pooled_gather_kernel<<<blocks_for(n, 256), 256, 0, stream>>>(
        static_cast<const std::uint16_t*>(plane.data), static_cast<const std::int32_t*>(block_tables.data),
        static_cast<const std::int32_t*>(table_rows.data), static_cast<const std::int32_t*>(positions.data),
        static_cast<int>(positions.numel() / sequences), sequence, block_tables.ne[0], dim, ratio, blocks,
        static_cast<std::uint16_t*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

void lossless_qsa_tails_store(const Tensor& keys, const Tensor& positions, int sequences, Tensor& tails,
                              const Tensor& tail_slots, int ratio, cudaStream_t stream) {
    require_runs(keys, positions, sequences, ratio);
    const int dim = keys.ne[0], columns = keys.ne[1];
    require_tails(tails, tail_slots, dim, ratio, sequences);
    if (columns == 0) { return; }
    const std::size_t n = static_cast<std::size_t>(dim) * columns;
    qsa_tails_store_kernel<<<blocks_for(n, 256), 256, 0, stream>>>(
        static_cast<const std::uint16_t*>(keys.data), static_cast<const std::int32_t*>(positions.data),
        static_cast<std::uint16_t*>(tails.data), static_cast<const std::int32_t*>(tail_slots.data), dim, ratio,
        columns / sequences, columns);
    CUDA_CHECK(cudaGetLastError());
}

std::size_t lossless_qsa_select_workspace_bytes(int columns, int blocks) {
    return static_cast<std::size_t>(columns) * std::max(blocks, 1) * sizeof(std::uint32_t);
}

void lossless_qsa_select(const Tensor& scores, int heads, int ratio, int budget, int index_dim, std::uint32_t* select,
                         int select_words, void* workspace, cudaStream_t stream, const Tensor* positions) {
    const int blocks = scores.ne[0], columns = scores.ne[1] / std::max(heads, 1);
    require(contiguous(scores, DType::FP32) && heads > 0 && scores.ne[1] == heads * columns && ratio > 0 && budget > 0 &&
                index_dim > 0 && select_words >= (blocks + 31) / 32 && select_words <= kSelectWords,
            "scores FP32 [blocks, heads * T], select U32 [T][words >= blocks / 32]");
    require(positions != nullptr ? positions->dtype == DType::I32 && positions->is_contiguous() &&
                                       positions->numel() == columns
                                 : blocks == columns / ratio,
            "positions I32 [T] (the scores then cover every query's complete blocks), else blocks = T / ratio");
    if (columns == 0) { return; }
    const auto divisor = static_cast<float>(std::sqrt(static_cast<double>(index_dim)));
    qsa_select_kernel<<<static_cast<unsigned>(columns), kSelectThreads, 0, stream>>>(
        static_cast<const float*>(scores.data), blocks, heads, ratio, budget, divisor, columns, select, select_words,
        static_cast<std::uint32_t*>(workspace),
        positions != nullptr ? static_cast<const std::int32_t*>(positions->data) : nullptr);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace infernix::ops

// This unit's copy of the noise twin's seed (ops/lossless/noise.cuh).
INFERNIX_LOSSLESS_NOISE_UNIT(flash)
