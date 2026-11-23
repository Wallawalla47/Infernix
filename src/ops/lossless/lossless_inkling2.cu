#include "infernix/ops/lossless_inkling2.h"

#include "core/device.h"
#include "ops/common/canonical_math.h"
#include "ops/lossless/cr_math.cuh"
#include "ops/lossless/wide.cuh"

#include <cuda_bf16.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

// The --lossless route's Inkling 2 operations (include/infernix/ops/lossless_inkling2.h, docs/maintainer/
// lossless-route.md §2b), each correctly rounded: IEEE binary32 multiplies, adds and divisions are __fmul_rn /
// __fadd_rn / __fdiv_rn, a BF16 rounding of an exact binary32 value __float2bfloat16_rn, sums exact (wide.cuh) and
// rounded once, transcendental functions cr_math.cuh's, a softmax element its quotient's correct rounding (decided
// by an error bound in binary64, otherwise in double-double).

namespace infernix::ops {
namespace {

using lossless::DD;
using lossless::F32Sum;
using lossless::kF32SumWords;
using lossless::kProductSumUnit;
using lossless::kProductSumWords;
using lossless::ProductSum;
using lossless::round_magnitude;

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string("lossless Inkling 2 op: ") + message); }
}

bool is_matrix(const Tensor& t) { return t.ne[2] == 1 && t.ne[3] == 1; }
bool float_dtype(const Tensor& t) { return t.dtype == DType::BF16 || t.dtype == DType::FP32; }
// Elements between consecutive columns of a [rows, columns] tensor whose rows are contiguous.
int column_stride(const Tensor& t) {
    require(t.nb[0] == static_cast<std::int64_t>(dtype_size(t.dtype)), "rows must be contiguous");
    return static_cast<int>(t.nb[1] / static_cast<std::int64_t>(dtype_size(t.dtype)));
}

__device__ __forceinline__ float bf16_bits_float(std::uint16_t bits) {
    return __uint_as_float(static_cast<std::uint32_t>(bits) << 16);
}
__device__ __forceinline__ std::uint16_t bf16_bits_rn(float v) { return __bfloat16_as_ushort(__float2bfloat16_rn(v)); }

__device__ __forceinline__ float load(const void* x, bool fp32, std::size_t i) {
    return fp32 ? static_cast<const float*>(x)[i] : bf16_bits_float(static_cast<const std::uint16_t*>(x)[i]);
}
__device__ __forceinline__ void store(void* out, bool fp32, std::size_t i, float v) {
    if (fp32) {
        static_cast<float*>(out)[i] = v;
    } else {
        static_cast<std::uint16_t*>(out)[i] = bf16_bits_rn(v);
    }
}

// Sums every thread's accumulator chunks over the block (256 threads) into lane 0 of warp 0's chunks; `scratch`
// holds Words chunks per warp. Every thread must call it.
template <int Words>
__device__ void block_sum(lossless::Wide<Words>& acc, long long (*scratch)[Words]) {
    acc.warp_sum();
    const int warp = static_cast<int>(threadIdx.x) >> 5, lane = static_cast<int>(threadIdx.x) & 31;
    if (lane == 0) {
#pragma unroll
        for (int i = 0; i < Words; ++i) { scratch[warp][i] = acc.chunk[i]; }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        const int warps = static_cast<int>(blockDim.x) >> 5;
        for (int w = 1; w < warps; ++w) {
#pragma unroll
            for (int i = 0; i < Words; ++i) { acc.chunk[i] += scratch[w][i]; }
        }
    }
    __syncthreads();
}

// ---------------------------------------------------------------------------------------------- RMSNorm

// m = fp32(mean(fp32(x^2))) over a row (one warp; every lane returns it): each square rounded to binary32 (exact for
// BF16 x in range), their sum exact, divided by the width, rounded once. Non-finite squares give IEEE's result.
__device__ float row_mean_of_squares(const void* x, bool fp32, std::size_t base, int width, int lane) {
    F32Sum sum;
    sum.clear();
    for (int i = lane; i < width; i += 32) {
        const float v = load(x, fp32, base + static_cast<std::size_t>(i));
        sum.add(__fmul_rn(v, v));
    }
    sum.acc.warp_sum();
    const bool nan = __any_sync(0xFFFFFFFFU, sum.nan), inf = __any_sync(0xFFFFFFFFU, sum.pos_inf);
    if (nan) { return NAN; }
    if (inf) { return INFINITY; }
    std::uint32_t mag[kF32SumWords + 2] = {};
    std::uint32_t total[kF32SumWords];
    (void)sum.acc.finish(total); // squares are nonnegative
#pragma unroll
    for (int i = 0; i < kF32SumWords; ++i) { mag[i + 2] = total[i]; } // 64 bits up: the quotient keeps a sticky bit
    const bool sticky = lossless::divide_small(mag, static_cast<std::uint32_t>(width));
    return __uint_as_float(round_magnitude(mag, -150 - 64, false, false, sticky));
}

// One warp a row: y = fp32(x r), r = fp32(rsqrt(fp32(m + eps))); v = y (`w` null) or fp32(w y); out = v (FP32) or
// bf16(v).
__global__ void rmsnorm_kernel(const void* x, bool x_fp32, int x_ld, const std::uint16_t* w, float eps, void* out,
                               bool out_fp32, int out_ld, int width, int rows) {
    const int row  = static_cast<int>((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    const int lane = static_cast<int>(threadIdx.x) & 31;
    if (row >= rows) { return; }
    const std::size_t xb = static_cast<std::size_t>(row) * x_ld, ob = static_cast<std::size_t>(row) * out_ld;
    const float m = row_mean_of_squares(x, x_fp32, xb, width, lane);
    const float r = lossless::cr_rsqrt(__fadd_rn(m, eps));
    for (int i = lane; i < width; i += 32) {
        float v = __fmul_rn(load(x, x_fp32, xb + i), r);
        if (w != nullptr) { v = __fmul_rn(bf16_bits_float(w[i]), v); }
        store(out, out_fp32, ob + i, v);
    }
}

// ---------------------------------------------------------------------------------------------- softmax

// A softmax element's quotient has no proven worst case (unlike a function of one binary32 argument), so the
// double-double evaluation is accepted only where its own error bound decides the rounding; an element it leaves
// undecided sets g_undecided, which fails the request (lossless_inkling2_take_undecided). g_force_fallback (a test
// hook) sends every softmax through the double-double evaluation.
__device__ unsigned int g_undecided;
__constant__ bool g_force_fallback;

// e / s rounded to binary32 from double-double terms; `terms` the sum's term count (each dd_exp within 2^-94, each of
// the sum's dd additions within 2^-104 of the sum of positive terms, the quotient's few roundings).
__device__ float dd_quotient(DD e, DD s, int terms) {
    const DD q       = lossless::dd_div(e, s);
    const double err = fabs(q.hi) * (0x1p-90 + static_cast<double>(terms) * 0x1p-103) + fabs(q.lo);
    float decided;
    if (!lossless::certain_f32(q.hi, err, decided)) { atomicOr(&g_undecided, 1U); }
    return lossless::f32_of_dd(q);
}

// Correctly rounded binary32 softmax of n <= 8 binary32 logits (one thread). Each p_j = e^(l_j - m) / S is first
// evaluated in binary64 (exp within an ulp, the sum and the quotient adding (n + 2) roundings: relative error below
// (n + 4) 2^-52) and accepted when that interval decides its rounding; otherwise every term is redone in
// double-double. A NaN logit, or a largest logit that is infinite, makes every p NaN (as IEEE evaluation does).
__device__ void softmax_small(const float* l, int n, float* p) {
    float m      = -INFINITY;
    bool invalid = false;
    for (int j = 0; j < n; ++j) {
        invalid = invalid || isnan(l[j]);
        m       = fmaxf(m, l[j]);
    }
    if (invalid || isinf(m)) {
        for (int j = 0; j < n; ++j) { p[j] = NAN; }
        return;
    }
    double e[8], s = 0.0;
    for (int j = 0; j < n; ++j) {
        const double d = static_cast<double>(l[j]) - static_cast<double>(m); // exact
        e[j]           = d < -745.0 ? 0.0 : exp(d);
        s += e[j];
    }
    bool redo = false;
    for (int j = 0; j < n; ++j) {
        const double q = e[j] / s;
        if (!lossless::certain_f32(q, q * static_cast<double>(n + 4) * 0x1p-52, p[j])) { redo = true; }
    }
    if (!redo && !g_force_fallback) { return; }
    DD ed[8], sd = {0.0, 0.0};
    for (int j = 0; j < n; ++j) {
        const double d = static_cast<double>(l[j]) - static_cast<double>(m);
        ed[j]          = d < -700.0 ? DD{0.0, 0.0} : lossless::dd_exp(d);
        sd             = lossless::dd_add(sd, ed[j]);
    }
    for (int j = 0; j < n; ++j) { p[j] = dd_quotient(ed[j], sd, n); }
}

// ---------------------------------------------------------------------------------------------- mix

constexpr int kMaxMixSources = 6;
constexpr int kMaxMixQueries = 4;
constexpr int kMixThreads    = 256;

struct MixArgs {
    const void* source[kMaxMixSources];
    bool fp32[kMaxMixSources];
    int ld[kMaxMixSources];
    int sources;
    const float* queries; // [Q][H]
    int query_count;
    float* out[kMaxMixQueries];
    int out_ld[kMaxMixQueries];
    int hidden;
    float scale; // fp32(H^-1/2)
};

// One CTA a column.
__global__ void __launch_bounds__(kMixThreads) mix_kernel(MixArgs a) {
    __shared__ long long scratch[kMixThreads / 32][kProductSumWords];
    __shared__ float r[kMaxMixSources];
    __shared__ float p[kMaxMixQueries][kMaxMixSources];
    const int t = static_cast<int>(blockIdx.x);
    const int H = a.hidden;
    // The sources' rms factors.
    for (int j = 0; j < a.sources; ++j) {
        const std::size_t base = static_cast<std::size_t>(t) * a.ld[j];
        lossless::Wide<kF32SumWords> acc;
        acc.clear();
        bool nan = false, inf = false;
        F32Sum sq;
        sq.clear();
        for (int i = static_cast<int>(threadIdx.x); i < H; i += kMixThreads) {
            const float v = load(a.source[j], a.fp32[j], base + i);
            sq.add(__fmul_rn(v, v));
        }
        nan = __syncthreads_or(sq.nan ? 1 : 0) != 0;
        inf = __syncthreads_or(sq.pos_inf ? 1 : 0) != 0;
        block_sum<kF32SumWords>(sq.acc, reinterpret_cast<long long (*)[kF32SumWords]>(scratch));
        if (threadIdx.x == 0) {
            float m;
            if (nan) {
                m = NAN;
            } else if (inf) {
                m = INFINITY;
            } else {
                std::uint32_t mag[kF32SumWords + 2] = {};
                std::uint32_t total[kF32SumWords];
                (void)sq.acc.finish(total);
#pragma unroll
                for (int i = 0; i < kF32SumWords; ++i) { mag[i + 2] = total[i]; }
                const bool sticky = lossless::divide_small(mag, static_cast<std::uint32_t>(H));
                m = __uint_as_float(round_magnitude(mag, -150 - 64, false, false, sticky));
            }
            r[j] = lossless::cr_rsqrt(__fadd_rn(m, 1e-6F));
        }
        __syncthreads();
    }
    // The queries' logits, then their softmax.
    for (int q = 0; q < a.query_count; ++q) {
        float l[kMaxMixSources];
        for (int j = 0; j < a.sources; ++j) {
            const std::size_t base = static_cast<std::size_t>(t) * a.ld[j];
            ProductSum dot;
            dot.clear();
            for (int i = static_cast<int>(threadIdx.x); i < H; i += kMixThreads) {
                const float n = __fmul_rn(load(a.source[j], a.fp32[j], base + i), r[j]);
                dot.add(a.queries[static_cast<std::size_t>(q) * H + i], n);
            }
            const bool nan = __syncthreads_or(dot.nan ? 1 : 0) != 0, pinf = __syncthreads_or(dot.pos_inf ? 1 : 0) != 0,
                       ninf = __syncthreads_or(dot.neg_inf ? 1 : 0) != 0;
            block_sum<kProductSumWords>(dot.acc, scratch);
            dot.nan = nan, dot.pos_inf = pinf, dot.neg_inf = ninf;
            l[j] = __fmul_rn(dot.result(), a.scale);
        }
        if (threadIdx.x == 0) { softmax_small(l, a.sources, p[q]); }
        __syncthreads();
    }
    // out_q = fp32(sum_j p_j n_j), each element's sum exact.
    for (int i = static_cast<int>(threadIdx.x); i < H; i += kMixThreads) {
        float n[kMaxMixSources];
        for (int j = 0; j < a.sources; ++j) {
            n[j] = __fmul_rn(load(a.source[j], a.fp32[j], static_cast<std::size_t>(t) * a.ld[j] + i), r[j]);
        }
        for (int q = 0; q < a.query_count; ++q) {
            ProductSum s;
            s.clear();
            for (int j = 0; j < a.sources; ++j) { s.add(p[q][j], n[j]); }
            a.out[q][static_cast<std::size_t>(t) * a.out_ld[q] + i] = s.result();
        }
    }
}

// ---------------------------------------------------------------------------------------------- message gate

__global__ void message_gate_kernel(const float* forget, const float* gain, const float* x, const float* m, float* out,
                                    std::int64_t count) {
    for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; i < count;
         i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        const float f = lossless::cr_tanh(forget[i]), g = lossless::cr_tanh(gain[i]);
        out[i]        = __fadd_rn(__fmul_rn(__fadd_rn(1.0F, -f), x[i]), __fmul_rn(g, m[i]));
    }
}

// ---------------------------------------------------------------------------------------------- L2 clips

// One warp a head row of D <= 128 elements.
__global__ void clip_kernel(const std::uint16_t* x, int x_ld, const std::uint16_t* scale, int heads, int D,
                            float threshold, const std::int32_t* positions, float alpha, float floor_value,
                            std::uint16_t* out, int out_ld, int rows) {
    const int row  = static_cast<int>((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    const int lane = static_cast<int>(threadIdx.x) & 31;
    if (row >= rows) { return; }
    const int t = row / heads, h = row % heads;
    const std::size_t xb = static_cast<std::size_t>(t) * x_ld + static_cast<std::size_t>(h) * D;
    const std::size_t ob = static_cast<std::size_t>(t) * out_ld + static_cast<std::size_t>(h) * D;
    float y[4];
    ProductSum sq;
    sq.clear();
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int i = lane + 32 * k;
        y[k]        = 0.0F;
        if (i < D) {
            const float v = bf16_bits_float(x[xb + i]);
            y[k] = scale != nullptr ? __fmul_rn(v, bf16_bits_float(scale[static_cast<std::size_t>(h) * D + i])) : v;
            sq.add(y[k], y[k]);
        }
    }
    sq.warp_sum();
    float nu;
    if (sq.nan) {
        nu = NAN;
    } else if (sq.pos_inf) {
        nu = INFINITY;
    } else {
        std::uint32_t mag[kProductSumWords];
        (void)sq.acc.finish(mag);
        nu = lossless::sqrt_magnitude(mag, kProductSumUnit);
    }
    float c = __fmul_rn(__frcp_rn(__fadd_rn(nu, 1e-6F)), threshold);
    c       = c > 1.0F ? 1.0F : c; // clamp(max=1) keeps a NaN
    float tau = 1.0F;
    if (positions != nullptr) {
        float q = __fdiv_rn(static_cast<float>(positions[t] + 1), floor_value);
        q       = q < 1.0F ? 1.0F : q;
        tau     = __fadd_rn(1.0F, __fmul_rn(alpha, lossless::cr_log(q)));
    }
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int i = lane + 32 * k;
        if (i >= D) { continue; }
        std::uint16_t o = bf16_bits_rn(__fmul_rn(y[k], c));
        if (positions != nullptr) { o = bf16_bits_rn(__fmul_rn(bf16_bits_float(o), tau)); }
        out[ob + i] = o;
    }
}

// ---------------------------------------------------------------------------------------------- convolutions

// A thread a (channel, column).
__global__ void conv_kernel(const std::uint16_t* x, int x_ld, const std::uint16_t* taps, int k, float divisor,
                            std::uint16_t* out, int out_ld, int C, int T) {
    const std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    if (i >= static_cast<std::int64_t>(C) * T) { return; }
    const int c = static_cast<int>(i % C), t = static_cast<int>(i / C);
    F32Sum s;
    s.clear();
    for (int d = 0; d < k && d <= t; ++d) {
        s.add(__fmul_rn(bf16_bits_float(taps[static_cast<std::size_t>(d) * C + c]),
                        bf16_bits_float(x[static_cast<std::size_t>(t - d) * x_ld + c])));
    }
    const float xt = bf16_bits_float(x[static_cast<std::size_t>(t) * x_ld + c]);
    out[static_cast<std::size_t>(t) * out_ld + c] = bf16_bits_rn(__fdiv_rn(__fadd_rn(s.result(), xt), divisor));
}

__global__ void dynamic_conv_kernel(const std::uint16_t* x, int x_ld, const std::uint16_t* dynamic, int dyn_ld,
                                    const std::uint16_t* taps, int k, int heads, float divisor, std::uint16_t* out,
                                    int out_ld, int C, int T) {
    const std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    if (i >= static_cast<std::int64_t>(C) * T) { return; }
    const int c = static_cast<int>(i % C), t = static_cast<int>(i / C);
    const int h = c / (C / heads);
    F32Sum s;
    s.clear();
    for (int d = 0; d < k; ++d) {
        const float w = __fadd_rn(bf16_bits_float(taps[static_cast<std::size_t>(d) * C + c]),
                                  bf16_bits_float(dynamic[static_cast<std::size_t>(t) * dyn_ld + h * k + d]));
        const float v = d <= t ? bf16_bits_float(x[static_cast<std::size_t>(t - d) * x_ld + c]) : 0.0F;
        s.add(__fmul_rn(v, w));
    }
    const float xt = bf16_bits_float(x[static_cast<std::size_t>(t) * x_ld + c]);
    out[static_cast<std::size_t>(t) * out_ld + c] = bf16_bits_rn(__fdiv_rn(__fadd_rn(s.result(), xt), divisor));
}

// ---------------------------------------------------------------------------------------------- elementwise

__global__ void sigmoid_gate_kernel(const std::uint16_t* o, const std::uint16_t* g, std::uint16_t* out, int width,
                                    int o_ld, int g_ld, int out_ld, std::int64_t count) {
    for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; i < count;
         i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        const std::int64_t t = i / width, e = i % width;
        const float s        = lossless::cr_sigmoid(bf16_bits_float(g[t * g_ld + e]));
        out[t * out_ld + e]  = bf16_bits_rn(__fmul_rn(bf16_bits_float(o[t * o_ld + e]), s));
    }
}

__global__ void add_kernel(const void* a, bool a_fp32, const void* b, bool b_fp32, void* out, bool out_fp32,
                           bool via_fp32, std::int64_t count) {
    for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; i < count;
         i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        const double x = static_cast<double>(load(a, a_fp32, i)), y = static_cast<double>(load(b, b_fp32, i));
        if (out_fp32 || via_fp32) {
            // Both inputs are binary32 values, so binary64's rounding of the sum rounds again to the binary32 one.
            const float f = __double2float_rn(__dadd_rn(x, y));
            store(out, out_fp32, i, f);
        } else {
            const DD s = lossless::dd_two_sum(x, y); // exact: x + y = s.hi + s.lo
            static_cast<std::uint16_t*>(out)[i] = lossless::bf16_rne_of_dd(s);
        }
    }
}

__global__ void convert_kernel(const void* x, bool x_fp32, void* out, bool out_fp32, std::int64_t count) {
    for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; i < count;
         i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        store(out, out_fp32, i, load(x, x_fp32, i)); // a binary32's BF16 rounding, or an exact widening
    }
}

__global__ void softcap_kernel(const float* x, float cap, float* out, std::int64_t count) {
    for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; i < count;
         i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        out[i] = __fmul_rn(lossless::cr_tanh(__fdiv_rn(x[i], cap)), cap);
    }
}

// ---------------------------------------------------------------------------------------------- attention probs

constexpr int kMaxRelative  = 32;
constexpr int kProbsThreads = 256;
constexpr int kProbsWarps   = kProbsThreads / 32;

struct ProbsArgs {
    std::uint16_t* scores;
    int ld, keys, query_begin, head, heads, columns, query_columns; // columns: head_count * query_columns
    const std::uint16_t* relative; // [rel][heads][T]: column t's head h at (t heads + h) rel
    int rel, rel_ld;
    const std::uint16_t* table; // [rel][extent]
    int extent;
    const std::uint16_t* sinks; // [heads][T]: (t heads + h)
    int sink_ld;
    int window;
    float scale;
};

// The bias of distance d: bf16 of the exact sum of r_c table[c, d] (BF16 products, exact in binary32).
__device__ __forceinline__ float relative_bias(const float (&r)[kMaxRelative], int rel, const std::uint16_t* table,
                                               int extent, int d) {
    F32Sum s;
    s.clear();
    for (int c = 0; c < rel; ++c) { s.add(__fmul_rn(r[c], bf16_bits_float(table[static_cast<std::size_t>(c) * extent + d]))); }
    std::uint32_t mag[kF32SumWords];
    const bool negative = s.acc.finish(mag);
    if (s.nan || (s.pos_inf && s.neg_inf)) { return NAN; }
    if (s.pos_inf) { return INFINITY; }
    if (s.neg_inf) { return -INFINITY; }
    return bf16_bits_float(static_cast<std::uint16_t>(round_magnitude(mag, -150, negative, true)));
}

// The block's OR, largest value and double-double sum of per-thread values (every thread calls them; `scratch` holds
// one entry a warp). The warps' partial sums are added in warp order: a sum of positive terms in double-double keeps a
// relative error below 2^-103 an addition whatever its tree.
__device__ __forceinline__ bool block_any(bool v, int* scratch) {
    v = __any_sync(0xFFFFFFFFU, v);
    if ((threadIdx.x & 31) == 0) { scratch[threadIdx.x >> 5] = v ? 1 : 0; }
    __syncthreads();
    bool out = false;
    for (int w = 0; w < kProbsWarps; ++w) { out = out || scratch[w] != 0; }
    __syncthreads();
    return out;
}

__device__ __forceinline__ float block_max(float m, float* scratch) {
    for (int o = 16; o > 0; o >>= 1) { m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFFU, m, o)); }
    if ((threadIdx.x & 31) == 0) { scratch[threadIdx.x >> 5] = m; }
    __syncthreads();
    for (int w = 0; w < kProbsWarps; ++w) { m = fmaxf(m, scratch[w]); }
    __syncthreads();
    return m;
}

__device__ __forceinline__ DD block_dd_sum(DD s, double* scratch_hi, double* scratch_lo) {
    for (int o = 16; o > 0; o >>= 1) {
        const DD other = {__shfl_xor_sync(0xFFFFFFFFU, s.hi, o), __shfl_xor_sync(0xFFFFFFFFU, s.lo, o)};
        s              = lossless::dd_add(s, other);
    }
    if ((threadIdx.x & 31) == 0) {
        scratch_hi[threadIdx.x >> 5] = s.hi;
        scratch_lo[threadIdx.x >> 5] = s.lo;
    }
    __syncthreads();
    DD out = {scratch_hi[0], scratch_lo[0]};
    for (int w = 1; w < kProbsWarps; ++w) { out = lossless::dd_add(out, DD{scratch_hi[w], scratch_lo[w]}); }
    __syncthreads();
    return out;
}

// One CTA per (head, query column), its visible keys across the block's threads.
__global__ void __launch_bounds__(kProbsThreads) probs_kernel(ProbsArgs a) {
    __shared__ int shared_flag[kProbsWarps];
    __shared__ float shared_max[kProbsWarps];
    __shared__ double shared_hi[kProbsWarps], shared_lo[kProbsWarps];
    const int c      = static_cast<int>(blockIdx.x);
    const int t      = c % a.query_columns, head = a.head + c / a.query_columns;
    const int lane   = static_cast<int>(threadIdx.x);
    std::uint16_t* row = a.scores + static_cast<std::size_t>(c) * a.ld;
    const int P = a.query_begin + t;
    const int lo = a.window > 0 ? max(0, P - a.window + 1) : 0, hi = min(P, a.keys - 1); // visible keys [lo, hi]
    float r[kMaxRelative];
    for (int i = 0; i < a.rel; ++i) {
        r[i] = bf16_bits_float(a.relative[static_cast<std::size_t>(t) * a.rel_ld + static_cast<std::size_t>(head) * a.rel + i]);
    }
    // Pass 1: the scaled, biased logits in place; their maximum.
    float m      = bf16_bits_float(a.sinks[static_cast<std::size_t>(t) * a.sink_ld + head]);
    const float sink = m;
    bool nan     = isnan(m);
    for (int j = lo + lane; j <= hi; j += kProbsThreads) {
        const double prod = static_cast<double>(bf16_bits_float(row[j])) * static_cast<double>(a.scale); // exact
        float v           = bf16_bits_float(lossless::bf16_rne_of_double(prod));
        const int d       = P - j;
        if (d < a.extent) {
            const double sum = static_cast<double>(v) + static_cast<double>(relative_bias(r, a.rel, a.table, a.extent, d));
            v                = bf16_bits_float(lossless::bf16_rne_of_double(sum)); // two BF16 values: exact in binary64
        }
        row[j] = static_cast<std::uint16_t>(__float_as_uint(v) >> 16);
        nan    = nan || isnan(v);
        m      = fmaxf(m, v);
    }
    nan = block_any(nan, shared_flag);
    m   = block_max(m, shared_max);
    const bool invalid = nan || isinf(m);
    // Pass 2: the sum of exponentials in double-double (each term within an ulp of binary64).
    DD s = {0.0, 0.0};
    if (!invalid) {
        for (int j = lo + lane; j <= hi; j += kProbsThreads) {
            const double d = static_cast<double>(bf16_bits_float(row[j])) - static_cast<double>(m);
            s              = lossless::dd_add(s, {d < -745.0 ? 0.0 : exp(d), 0.0});
        }
        s = block_dd_sum(s, shared_hi, shared_lo);
        const double ds = static_cast<double>(sink) - static_cast<double>(m);
        s               = lossless::dd_add(s, {ds < -745.0 ? 0.0 : exp(ds), 0.0});
    }
    // Pass 3: each probability decided in binary64 (relative error below 2^-50: an ulp of the term, an ulp of the
    // sum, the quotient's rounding; the double-double sum's own error, at most 2^20 additions of 2^-103, is far below
    // them); a row with an undecided one has all its quotients redone in double-double.
    const double S = s.hi + s.lo;
    bool undecided = false;
    if (!invalid) {
        for (int j = lo + lane; j <= hi; j += kProbsThreads) {
            const double d = static_cast<double>(bf16_bits_float(row[j])) - static_cast<double>(m);
            const double q = (d < -745.0 ? 0.0 : exp(d)) / S;
            float p;
            undecided = undecided || !lossless::certain_f32(q, q * 0x1p-50, p);
        }
    }
    undecided = block_any(undecided, shared_flag) || g_force_fallback;
    DD sd     = {0.0, 0.0};
    if (undecided && !invalid) {
        for (int j = lo + lane; j <= hi; j += kProbsThreads) {
            const double d = static_cast<double>(bf16_bits_float(row[j])) - static_cast<double>(m);
            sd             = lossless::dd_add(sd, d < -700.0 ? DD{0.0, 0.0} : lossless::dd_exp(d));
        }
        sd = block_dd_sum(sd, shared_hi, shared_lo);
        const double ds = static_cast<double>(sink) - static_cast<double>(m);
        sd              = lossless::dd_add(sd, ds < -700.0 ? DD{0.0, 0.0} : lossless::dd_exp(ds));
    }
    // Pass 4: the probabilities (BF16 of the binary32 rounding), zeros off the visible keys.
    for (int j = lane; j < a.keys; j += kProbsThreads) {
        if (j < lo || j > hi) {
            row[j] = 0;
            continue;
        }
        if (invalid) {
            row[j] = 0x7FC0U;
            continue;
        }
        const double d = static_cast<double>(bf16_bits_float(row[j])) - static_cast<double>(m);
        float p;
        if (!undecided) {
            p = __double2float_rn((d < -745.0 ? 0.0 : exp(d)) / S); // decided in pass 3
        } else {
            p = dd_quotient(d < -700.0 ? DD{0.0, 0.0} : lossless::dd_exp(d), sd, hi - lo + 2);
        }
        row[j] = bf16_bits_rn(p);
    }
}

// ---------------------------------------------------------------------------------------------- router

constexpr int kMaxRouterExperts = 1024;
constexpr int kMaxRouterTopK    = 16;
constexpr int kRouterWarps      = 4;

// One warp a column: keys fp32(sigmoid(l) + bias) (a NaN key ranks below every other), the top_k by key (ties to
// the lower expert) in descending order, gamma from their sigmoids.
__global__ void __launch_bounds__(32 * kRouterWarps)
    router_kernel(const float* logits, int logits_ld, const float* bias, int experts, int top_k, float scaling,
                  float* weights, int weights_ld, std::int32_t* ids, int ids_ld, int columns) {
    __shared__ float keys[kRouterWarps][kMaxRouterExperts];
    __shared__ float sel[kRouterWarps][kMaxRouterTopK];
    __shared__ int sel_id[kRouterWarps][kMaxRouterTopK];
    const int warp = static_cast<int>(threadIdx.x) >> 5, lane = static_cast<int>(threadIdx.x) & 31;
    const int t    = static_cast<int>(blockIdx.x) * kRouterWarps + warp;
    if (t >= columns) { return; }
    const float* l = logits + static_cast<std::size_t>(t) * logits_ld;
    for (int e = lane; e < experts; e += 32) {
        const float key = __fadd_rn(lossless::cr_sigmoid(l[e]), bias[e]);
        keys[warp][e]   = key != key ? -INFINITY : key;
    }
    __syncwarp();
    for (int e = lane; e < experts; e += 32) {
        const float key = keys[warp][e];
        int rank        = 0;
        for (int f = 0; f < experts; ++f) {
            const float other = keys[warp][f];
            rank += (other > key || (other == key && f < e)) ? 1 : 0;
        }
        if (rank < top_k) {
            sel_id[warp][rank] = e;
            sel[warp][rank]    = lossless::cr_sigmoid(l[e]);
        }
    }
    __syncwarp();
    if (lane == 0) {
        F32Sum sum;
        sum.clear();
        for (int j = 0; j < top_k; ++j) { sum.add(sel[warp][j]); }
        const float den = __fadd_rn(sum.result(), 1e-20F);
        for (int j = 0; j < top_k; ++j) {
            weights[static_cast<std::size_t>(t) * weights_ld + j] = __fmul_rn(__fdiv_rn(sel[warp][j], den), scaling);
            ids[static_cast<std::size_t>(t) * ids_ld + j]         = sel_id[warp][j];
        }
    }
}

// ---------------------------------------------------------------------------------------------- quantizers

constexpr int kQuantThreads = 256;

// A CTA a column: the row amax (an exact maximum), then a thread a 16-element block under canon's rule.
__global__ void __launch_bounds__(kQuantThreads)
    four_over_six_kernel(const std::uint16_t* x, int x_ld, const std::int32_t* in_columns, int K,
                         std::uint16_t* out, int out_ld, float* outer) {
    __shared__ float warp_max[kQuantThreads / 32];
    const int c   = static_cast<int>(blockIdx.x);
    const int src = in_columns != nullptr ? in_columns[c] : c;
    if (src < 0) { // a padding column: zeros
        for (int i = static_cast<int>(threadIdx.x); i < K; i += kQuantThreads) { out[static_cast<std::size_t>(c) * out_ld + i] = 0; }
        if (threadIdx.x == 0) { outer[c] = 0.0F; }
        return;
    }
    const std::uint16_t* row = x + static_cast<std::size_t>(src) * x_ld;
    float amax = 0.0F;
    for (int i = static_cast<int>(threadIdx.x); i < K; i += kQuantThreads) { amax = fmaxf(amax, fabsf(bf16_bits_float(row[i]))); }
    for (int o = 16; o > 0; o >>= 1) { amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFU, amax, o)); }
    if ((threadIdx.x & 31) == 0) { warp_max[threadIdx.x >> 5] = amax; }
    __syncthreads();
    amax = warp_max[0];
    for (int w = 1; w < kQuantThreads / 32; ++w) { amax = fmaxf(amax, warp_max[w]); }
    const canon::FourOverSixRow r = canon::four_over_six_row(amax);
    if (threadIdx.x == 0) { outer[c] = r.outer; }
    for (int b = static_cast<int>(threadIdx.x); b < K / 16; b += kQuantThreads) {
        float v[16];
#pragma unroll
        for (int j = 0; j < 16; ++j) { v[j] = bf16_bits_float(row[16 * b + j]); }
        const canon::FourOverSixBlock q = canon::four_over_six_block(v, r);
        const float s                   = canon::e4m3_value(q.scale_word);
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            // The code's signed E2M1 value (a negative-zero code keeps its sign), times its scale: exact.
            const float mag  = __fmul_rn(0.5F, static_cast<float>(canon::e2m1_x2(q.code[j] & 7U)));
            const float code = (q.code[j] & 8U) != 0 ? -mag : mag;
            out[static_cast<std::size_t>(c) * out_ld + 16 * b + j] = bf16_bits_rn(__fmul_rn(code, s));
        }
    }
}

// A thread a 32-element group.
__global__ void mxfp8_kernel(const std::uint16_t* x, int x_ld, const std::int32_t* in_columns, int K, int columns,
                             std::uint16_t* out, int out_ld) {
    const std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const int groups     = K / 32;
    if (i >= static_cast<std::int64_t>(groups) * columns) { return; }
    const int c = static_cast<int>(i / groups), g = static_cast<int>(i % groups);
    const int src = in_columns != nullptr ? in_columns[c] : c;
    if (src < 0) { // a padding column: zeros
#pragma unroll
        for (int j = 0; j < 32; ++j) { out[static_cast<std::size_t>(c) * out_ld + 32 * g + j] = 0; }
        return;
    }
    float v[32];
#pragma unroll
    for (int j = 0; j < 32; ++j) { v[j] = bf16_bits_float(x[static_cast<std::size_t>(src) * x_ld + 32 * g + j]); }
    const canon::Mxfp8Group q = canon::mxfp8_group(v);
    const float scale         = q.exponent == 0 ? 0x1p-127F : __uint_as_float(static_cast<std::uint32_t>(q.exponent) << 23);
#pragma unroll
    for (int j = 0; j < 32; ++j) {
        const float mag = canon::e4m3_value(q.code[j] & 0x7FU);
        const float val = (q.code[j] & 0x80U) != 0 ? -mag : mag;
        out[static_cast<std::size_t>(c) * out_ld + 32 * g + j] = q.exponent == 255 ? 0x7FC0U : bf16_bits_rn(__fmul_rn(val, scale));
    }
}

// ---------------------------------------------------------------------------------------------- experts

__device__ __forceinline__ float softcap_rn(float v, float cap) {
    return __fmul_rn(lossless::cr_tanh(__fdiv_rn(v, cap)), cap);
}

__global__ void swiglu_kernel(const std::uint16_t* gate_up, int gu_ld, const float* gamma, float gate_cap,
                              float up_cap, std::uint16_t* out, int out_ld, int I, int columns) {
    const std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    if (i >= static_cast<std::int64_t>(I) * columns) { return; }
    const int c = static_cast<int>(i / I), e = static_cast<int>(i % I);
    const float g = bf16_bits_float(gate_up[static_cast<std::size_t>(c) * gu_ld + 2 * e]);
    const float u = bf16_bits_float(gate_up[static_cast<std::size_t>(c) * gu_ld + 2 * e + 1]);
    const float a = __fmul_rn(softcap_rn(g, gate_cap), lossless::cr_sigmoid(g));
    out[static_cast<std::size_t>(c) * out_ld + e] = bf16_bits_rn(__fmul_rn(__fmul_rn(a, softcap_rn(u, up_cap)), gamma[c]));
}

__global__ void combine_kernel(const std::uint16_t* y, int y_ld, const std::int32_t* slot_column, int top_k,
                               std::uint16_t* out, int out_ld, int H, int columns) {
    const std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    if (i >= static_cast<std::int64_t>(H) * columns) { return; }
    const int t = static_cast<int>(i / H), e = static_cast<int>(i % H);
    float v = 0.0F;
    for (int j = 0; j < top_k; ++j) {
        const int col = slot_column[static_cast<std::size_t>(t) * top_k + j];
        v             = __fadd_rn(v, bf16_bits_float(y[static_cast<std::size_t>(col) * y_ld + e]));
    }
    out[static_cast<std::size_t>(t) * out_ld + e] = bf16_bits_rn(v);
}

__global__ void heads_kernel(const std::uint16_t* x, int x_ld, int D, int heads, int T, bool transposed, int stride,
                             std::uint16_t* out) {
    const std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    if (i >= static_cast<std::int64_t>(D) * heads * T) { return; }
    const int d = static_cast<int>(i % D), h = static_cast<int>((i / D) % heads), t = static_cast<int>(i / (static_cast<std::int64_t>(D) * heads));
    const std::uint16_t v = x[static_cast<std::size_t>(t) * x_ld + static_cast<std::size_t>(h) * D + d];
    const std::size_t plane = static_cast<std::size_t>(h) * D * stride;
    out[plane + (transposed ? static_cast<std::size_t>(d) * stride + t : static_cast<std::size_t>(t) * D + d)] = v;
}

// The cache form's state copies (exact).
__global__ void ring_load_kernel(const std::uint16_t* ring, int rows, int C, int first, int count, std::uint16_t* out,
                                 int out_ld) {
    const std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    if (i >= static_cast<std::int64_t>(C) * count) { return; }
    const int e = static_cast<int>(i % C), c = static_cast<int>(i / C);
    const int p = first + c;
    out[static_cast<std::size_t>(c) * out_ld + e] = p < 0 ? std::uint16_t{0} : ring[static_cast<std::size_t>(p % rows) * C + e];
}

__global__ void ring_store_kernel(const std::uint16_t* x, int x_ld, int C, int from, int count, int first, int rows,
                                  std::uint16_t* ring) {
    const std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    if (i >= static_cast<std::int64_t>(C) * count) { return; }
    const int e = static_cast<int>(i % C), c = from + static_cast<int>(i / C);
    ring[static_cast<std::size_t>((first + c) % rows) * C + e] = x[static_cast<std::size_t>(c) * x_ld + e];
}

constexpr int kKeyUnitRows = 64; // positions a KV unit (page) holds
constexpr int kKeyDim      = 128;

__device__ __forceinline__ std::size_t key_row(const LosslessInkling2Keys& s, int position, int head) {
    const int unit = position / kKeyUnitRows, row = position % kKeyUnitRows;
    if (s.table != nullptr) {
        const auto page = static_cast<std::size_t>(s.table[unit]);
        return ((page * s.heads + head) * kKeyUnitRows + row) * kKeyDim;
    }
    return ((static_cast<std::size_t>(head) * s.ring_pages + unit % s.ring_pages) * kKeyUnitRows + row) * kKeyDim;
}

// A block: 64 positions of one head. K rows go straight to the head-major plane; V rows through shared memory, so the
// transposed plane's writes are rows of 64 consecutive positions.
__global__ void __launch_bounds__(256) keys_load_kernel(LosslessInkling2Keys s, int first, int count, int stride,
                                                       std::uint16_t* kh, std::uint16_t* vt) {
    __shared__ std::uint16_t tile[kKeyUnitRows][kKeyDim + 2];
    const int j0 = static_cast<int>(blockIdx.x) * kKeyUnitRows, h = static_cast<int>(blockIdx.y);
    const std::size_t plane = static_cast<std::size_t>(h) * kKeyDim * stride;
    for (int idx = static_cast<int>(threadIdx.x); idx < kKeyUnitRows * kKeyDim; idx += 256) {
        const int jj = idx / kKeyDim, d = idx % kKeyDim;
        if (j0 + jj >= count) { continue; }
        const std::size_t r = key_row(s, first + j0 + jj, h);
        kh[plane + static_cast<std::size_t>(j0 + jj) * kKeyDim + d] = s.k[r + d];
        tile[jj][d] = s.v[r + d];
    }
    __syncthreads();
    for (int idx = static_cast<int>(threadIdx.x); idx < kKeyUnitRows * kKeyDim; idx += 256) {
        const int d = idx / kKeyUnitRows, jj = idx % kKeyUnitRows;
        if (j0 + jj >= count) { continue; }
        vt[plane + static_cast<std::size_t>(d) * stride + j0 + jj] = tile[jj][d];
    }
}

__global__ void keys_store_kernel(const std::uint16_t* k, int k_ld, const std::uint16_t* v, int v_ld, int from, int count,
                                  int first, LosslessInkling2Keys d) {
    const std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const int row = d.heads * kKeyDim;
    if (i >= static_cast<std::int64_t>(row) * count) { return; }
    const int e = static_cast<int>(i % row), t = from + static_cast<int>(i / row);
    const int h = e / kKeyDim, dd = e % kKeyDim;
    const std::size_t r = key_row(d, first + t, h);
    d.k[r + dd] = k[static_cast<std::size_t>(t) * k_ld + e];
    d.v[r + dd] = v[static_cast<std::size_t>(t) * v_ld + e];
}

unsigned grid_for(std::int64_t count, int threads) {
    const std::int64_t blocks = (count + threads - 1) / threads;
    return static_cast<unsigned>(blocks < 1 ? 1 : blocks);
}
unsigned stride_grid(std::int64_t count) {
    const std::int64_t blocks = (count + 255) / 256;
    return static_cast<unsigned>(blocks < 1 ? 1 : (blocks > 8192 ? 8192 : blocks));
}

} // namespace

void lossless_inkling2_rmsnorm(const Tensor& x, const Tensor* weight, float eps, Tensor& out, cudaStream_t stream) {
    require(is_matrix(x) && is_matrix(out) && x.ne[0] == out.ne[0] && x.ne[1] == out.ne[1],
            "rmsnorm: x and out are [width, rows] alike");
    require(float_dtype(x) && float_dtype(out), "rmsnorm: x and out BF16 or FP32");
    require(weight == nullptr || (weight->dtype == DType::BF16 && weight->numel() == x.ne[0]),
            "rmsnorm: the weight is BF16 [width]");
    const int width = static_cast<int>(x.ne[0]), rows = static_cast<int>(x.ne[1]);
    if (rows == 0) { return; }
    const int threads = 256, per_block = threads / 32;
    rmsnorm_kernel<<<(rows + per_block - 1) / per_block, threads, 0, stream>>>(
        x.data, x.dtype == DType::FP32, column_stride(x),
        weight != nullptr ? static_cast<const std::uint16_t*>(weight->data) : nullptr, eps, out.data,
        out.dtype == DType::FP32, column_stride(out), width, rows);
    CUDA_CHECK(cudaGetLastError());
}

void lossless_inkling2_mix(std::span<const Tensor* const> sources, const Tensor& queries, std::span<Tensor* const> out,
                           cudaStream_t stream) {
    require(!sources.empty() && sources.size() <= kMaxMixSources, "mix: 1 to 6 sources");
    require(!out.empty() && out.size() <= kMaxMixQueries && queries.dtype == DType::FP32 && queries.is_contiguous(),
            "mix: 1 to 4 FP32 queries");
    MixArgs a{};
    a.hidden  = static_cast<int>(sources[0]->ne[0]);
    const int T = static_cast<int>(sources[0]->ne[1]);
    require(queries.numel() == static_cast<std::int64_t>(out.size()) * a.hidden, "mix: queries are FP32 [Q, H]");
    a.sources = static_cast<int>(sources.size());
    for (int j = 0; j < a.sources; ++j) {
        const Tensor& s = *sources[j];
        require(is_matrix(s) && float_dtype(s) && s.ne[0] == a.hidden && s.ne[1] == T, "mix: sources [H, T] alike");
        a.source[j] = s.data;
        a.fp32[j]   = s.dtype == DType::FP32;
        a.ld[j]     = column_stride(s);
    }
    a.queries     = static_cast<const float*>(queries.data);
    a.query_count = static_cast<int>(out.size());
    for (int q = 0; q < a.query_count; ++q) {
        require(is_matrix(*out[q]) && out[q]->dtype == DType::FP32 && out[q]->ne[0] == a.hidden && out[q]->ne[1] == T,
                "mix: outputs FP32 [H, T]");
        a.out[q]    = static_cast<float*>(out[q]->data);
        a.out_ld[q] = column_stride(*out[q]);
    }
    a.scale = static_cast<float>(std::pow(static_cast<double>(a.hidden), -0.5));
    if (T == 0) { return; }
    mix_kernel<<<T, kMixThreads, 0, stream>>>(a);
    CUDA_CHECK(cudaGetLastError());
}

void lossless_inkling2_message_gate(const Tensor& forget, const Tensor& gain, const Tensor& x, const Tensor& m,
                                    Tensor& out, cudaStream_t stream) {
    for (const Tensor* t : {&forget, &gain, &x, &m, static_cast<const Tensor*>(&out)}) {
        require(t->dtype == DType::FP32 && t->is_contiguous() && t->numel() == x.numel(),
                "message gate: FP32 [H, T] contiguous operands");
    }
    const std::int64_t n = x.numel();
    if (n == 0) { return; }
    message_gate_kernel<<<stride_grid(n), 256, 0, stream>>>(static_cast<const float*>(forget.data),
                                                            static_cast<const float*>(gain.data),
                                                            static_cast<const float*>(x.data),
                                                            static_cast<const float*>(m.data),
                                                            static_cast<float*>(out.data), n);
    CUDA_CHECK(cudaGetLastError());
}

void lossless_inkling2_clip(const Tensor& x, const Tensor* scale, int heads, float threshold,
                            const LosslessLogScale& log_scale, Tensor& out, cudaStream_t stream) {
    require(is_matrix(x) && is_matrix(out) && x.dtype == DType::BF16 && out.dtype == DType::BF16 &&
                x.ne[0] == out.ne[0] && x.ne[1] == out.ne[1] && heads > 0 && x.ne[0] % heads == 0,
            "clip: x and out BF16 [D heads, T] alike");
    const int D = static_cast<int>(x.ne[0]) / heads, T = static_cast<int>(x.ne[1]);
    require(D <= 128, "clip: head rows of at most 128");
    require(scale == nullptr || (scale->dtype == DType::BF16 && scale->numel() == static_cast<std::int64_t>(heads) * D),
            "clip: the scale is BF16 [heads, D]");
    const Tensor* p = log_scale.positions;
    require(p == nullptr || (p->dtype == DType::I32 && p->numel() >= T && log_scale.floor > 0.0F),
            "clip: the log scale's positions are I32 [T] and its floor positive");
    const int rows = heads * T;
    if (rows == 0) { return; }
    clip_kernel<<<(rows + 7) / 8, 256, 0, stream>>>(
        static_cast<const std::uint16_t*>(x.data), column_stride(x),
        scale != nullptr ? static_cast<const std::uint16_t*>(scale->data) : nullptr, heads, D, threshold,
        p != nullptr ? static_cast<const std::int32_t*>(p->data) : nullptr, log_scale.alpha, log_scale.floor,
        static_cast<std::uint16_t*>(out.data), column_stride(out), rows);
    CUDA_CHECK(cudaGetLastError());
}

void lossless_inkling2_conv(const Tensor& x, const Tensor& taps, Tensor& out, cudaStream_t stream) {
    require(is_matrix(x) && is_matrix(out) && x.dtype == DType::BF16 && out.dtype == DType::BF16 &&
                x.ne[0] == out.ne[0] && x.ne[1] == out.ne[1],
            "conv: x and out BF16 [C, T] alike");
    require(taps.dtype == DType::BF16 && taps.is_contiguous() && taps.ne[0] == x.ne[0] && taps.ne[1] >= 1,
            "conv: taps BF16 [k, C]");
    require(x.data != out.data, "conv: out may not alias x");
    const int C = static_cast<int>(x.ne[0]), T = static_cast<int>(x.ne[1]), k = static_cast<int>(taps.ne[1]);
    const float divisor = static_cast<float>(std::sqrt(static_cast<double>(k + 1)));
    const std::int64_t n = static_cast<std::int64_t>(C) * T;
    if (n == 0) { return; }
    conv_kernel<<<grid_for(n, 256), 256, 0, stream>>>(static_cast<const std::uint16_t*>(x.data), column_stride(x),
                                                      static_cast<const std::uint16_t*>(taps.data), k, divisor,
                                                      static_cast<std::uint16_t*>(out.data), column_stride(out), C, T);
    CUDA_CHECK(cudaGetLastError());
}

void lossless_inkling2_dynamic_conv(const Tensor& x, const Tensor& dynamic, const Tensor& taps, int heads, Tensor& out,
                                    cudaStream_t stream) {
    require(is_matrix(x) && is_matrix(out) && x.dtype == DType::BF16 && out.dtype == DType::BF16 &&
                x.ne[0] == out.ne[0] && x.ne[1] == out.ne[1] && heads > 0 && x.ne[0] % heads == 0,
            "dynamic conv: x and out BF16 [C, T] alike");
    require(x.data != out.data, "dynamic conv: out may not alias x");
    const int C = static_cast<int>(x.ne[0]), T = static_cast<int>(x.ne[1]);
    require(taps.dtype == DType::BF16 && taps.is_contiguous() && taps.ne[0] == C, "dynamic conv: taps BF16 [k, C]");
    const int k = static_cast<int>(taps.ne[1]);
    require(is_matrix(dynamic) && dynamic.dtype == DType::BF16 && dynamic.ne[0] == static_cast<std::int64_t>(heads) * k &&
                dynamic.ne[1] == T,
            "dynamic conv: the per-column taps are BF16 [heads k, T]");
    const float divisor = static_cast<float>(std::sqrt(static_cast<double>(2 * k + 1)));
    const std::int64_t n = static_cast<std::int64_t>(C) * T;
    if (n == 0) { return; }
    dynamic_conv_kernel<<<grid_for(n, 256), 256, 0, stream>>>(
        static_cast<const std::uint16_t*>(x.data), column_stride(x), static_cast<const std::uint16_t*>(dynamic.data),
        column_stride(dynamic), static_cast<const std::uint16_t*>(taps.data), k, heads, divisor,
        static_cast<std::uint16_t*>(out.data), column_stride(out), C, T);
    CUDA_CHECK(cudaGetLastError());
}

void lossless_inkling2_sigmoid_gate(const Tensor& o, const Tensor& gate, Tensor& out, cudaStream_t stream) {
    require(is_matrix(o) && is_matrix(gate) && is_matrix(out) && o.dtype == DType::BF16 && gate.dtype == DType::BF16 &&
                out.dtype == DType::BF16 && o.ne[0] == gate.ne[0] && o.ne[0] == out.ne[0] && o.ne[1] == gate.ne[1] &&
                o.ne[1] == out.ne[1],
            "sigmoid gate: o, gate and out BF16 [W, T] alike");
    const std::int64_t n = o.numel();
    if (n == 0) { return; }
    sigmoid_gate_kernel<<<stride_grid(n), 256, 0, stream>>>(
        static_cast<const std::uint16_t*>(o.data), static_cast<const std::uint16_t*>(gate.data),
        static_cast<std::uint16_t*>(out.data), static_cast<int>(o.ne[0]), column_stride(o), column_stride(gate),
        column_stride(out), n);
    CUDA_CHECK(cudaGetLastError());
}

void lossless_inkling2_add(const Tensor& a, const Tensor& b, Tensor& out, cudaStream_t stream, bool via_fp32) {
    require(float_dtype(a) && float_dtype(b) && float_dtype(out) && a.is_contiguous() && b.is_contiguous() &&
                out.is_contiguous() && a.numel() == b.numel() && a.numel() == out.numel(),
            "add: contiguous BF16 or FP32 operands of one shape");
    const std::int64_t n = a.numel();
    if (n == 0) { return; }
    add_kernel<<<stride_grid(n), 256, 0, stream>>>(a.data, a.dtype == DType::FP32, b.data, b.dtype == DType::FP32,
                                                   out.data, out.dtype == DType::FP32, via_fp32, n);
    CUDA_CHECK(cudaGetLastError());
}

void lossless_inkling2_convert(const Tensor& x, Tensor& out, cudaStream_t stream) {
    require(float_dtype(x) && float_dtype(out) && x.is_contiguous() && out.is_contiguous() && x.numel() == out.numel(),
            "convert: contiguous BF16 or FP32 operands of one shape");
    const std::int64_t n = x.numel();
    if (n == 0) { return; }
    convert_kernel<<<stride_grid(n), 256, 0, stream>>>(x.data, x.dtype == DType::FP32, out.data, out.dtype == DType::FP32, n);
    CUDA_CHECK(cudaGetLastError());
}

void lossless_inkling2_attention_probs(const LosslessInkling2Probs& a, cudaStream_t stream) {
    require(a.scores != nullptr && a.scores->dtype == DType::BF16 && a.relative != nullptr && a.table != nullptr &&
                a.sinks != nullptr,
            "attention probs: the scores, relative states, table and sinks are needed");
    require(a.relative->dtype == DType::BF16 && a.table->dtype == DType::BF16 && a.sinks->dtype == DType::BF16 &&
                a.table->is_contiguous(),
            "attention probs: BF16 relative states, table and sinks");
    const int rel = static_cast<int>(a.table->ne[0]);
    require(rel <= kMaxRelative && a.relative->ne[0] == static_cast<std::int64_t>(rel) * a.heads,
            "attention probs: relative states [rel heads, T] with rel <= 32");
    require(a.head >= 0 && a.head_count > 0 && a.head + a.head_count <= a.heads && a.sinks->ne[0] == a.heads &&
                a.keys <= a.ld && a.scores->ne[1] % a.head_count == 0,
            "attention probs: shapes");
    ProbsArgs p{};
    p.scores      = static_cast<std::uint16_t*>(a.scores->data);
    p.ld          = a.ld;
    p.keys        = a.keys;
    p.query_begin = a.query_begin;
    p.head        = a.head;
    p.heads       = a.heads;
    p.columns     = static_cast<int>(a.scores->ne[1]);
    p.query_columns = p.columns / a.head_count;
    p.relative    = static_cast<const std::uint16_t*>(a.relative->data);
    p.rel         = rel;
    p.rel_ld      = column_stride(*a.relative);
    p.table       = static_cast<const std::uint16_t*>(a.table->data);
    p.extent      = static_cast<int>(a.table->ne[1]);
    p.sinks       = static_cast<const std::uint16_t*>(a.sinks->data);
    p.sink_ld     = column_stride(*a.sinks);
    p.window      = a.window;
    p.scale       = a.scale;
    require(a.relative->ne[1] >= p.query_columns && a.sinks->ne[1] >= p.query_columns,
            "attention probs: a relative row and sink a query column");
    if (p.columns == 0) { return; }
    probs_kernel<<<p.columns, kProbsThreads, 0, stream>>>(p);
    CUDA_CHECK(cudaGetLastError());
}

void lossless_inkling2_router(const Tensor& logits, const Tensor& bias, int top_k, float scaling, Tensor& weights,
                              Tensor& ids, cudaStream_t stream) {
    require(is_matrix(logits) && logits.dtype == DType::FP32 && bias.dtype == DType::FP32 &&
                bias.numel() == logits.ne[0] && logits.ne[0] <= kMaxRouterExperts,
            "router: FP32 logits [E <= 1024, T] and bias [E]");
    require(top_k > 0 && top_k <= kMaxRouterTopK && top_k <= logits.ne[0], "router: 1 <= top_k <= 16");
    const int T = static_cast<int>(logits.ne[1]);
    require(weights.dtype == DType::FP32 && ids.dtype == DType::I32 && weights.ne[0] == top_k && ids.ne[0] == top_k &&
                weights.ne[1] == T && ids.ne[1] == T,
            "router: weights FP32 and ids I32 [top_k, T]");
    if (T == 0) { return; }
    router_kernel<<<(T + kRouterWarps - 1) / kRouterWarps, 32 * kRouterWarps, 0, stream>>>(
        static_cast<const float*>(logits.data), column_stride(logits), static_cast<const float*>(bias.data),
        static_cast<int>(logits.ne[0]), top_k, scaling, static_cast<float*>(weights.data), column_stride(weights),
        static_cast<std::int32_t*>(ids.data), column_stride(ids), T);
    CUDA_CHECK(cudaGetLastError());
}

void lossless_inkling2_quantize_four_over_six(const Tensor& x, const std::int32_t* in_columns, int columns, Tensor& out,
                                              Tensor& outer, cudaStream_t stream) {
    require(is_matrix(x) && is_matrix(out) && x.dtype == DType::BF16 && out.dtype == DType::BF16 &&
                x.ne[0] == out.ne[0] && x.ne[0] % 16 == 0 && out.ne[1] >= columns && (in_columns != nullptr || x.ne[1] >= columns),
            "four-over-six: x and out BF16 [K, *], K a multiple of 16");
    require(outer.dtype == DType::FP32 && outer.numel() >= columns, "four-over-six: outer FP32 [columns]");
    if (columns == 0) { return; }
    four_over_six_kernel<<<columns, kQuantThreads, 0, stream>>>(
        static_cast<const std::uint16_t*>(x.data), column_stride(x), in_columns, static_cast<int>(x.ne[0]),
        static_cast<std::uint16_t*>(out.data), column_stride(out), static_cast<float*>(outer.data));
    CUDA_CHECK(cudaGetLastError());
}

void lossless_inkling2_quantize_mxfp8(const Tensor& x, const std::int32_t* in_columns, int columns, Tensor& out,
                                      cudaStream_t stream) {
    require(is_matrix(x) && is_matrix(out) && x.dtype == DType::BF16 && out.dtype == DType::BF16 &&
                x.ne[0] == out.ne[0] && x.ne[0] % 32 == 0 && out.ne[1] >= columns && (in_columns != nullptr || x.ne[1] >= columns),
            "mxfp8: x and out BF16 [K, *], K a multiple of 32");
    const std::int64_t n = (x.ne[0] / 32) * static_cast<std::int64_t>(columns);
    if (n == 0) { return; }
    mxfp8_kernel<<<grid_for(n, 256), 256, 0, stream>>>(static_cast<const std::uint16_t*>(x.data), column_stride(x),
                                                       in_columns, static_cast<int>(x.ne[0]), columns,
                                                       static_cast<std::uint16_t*>(out.data), column_stride(out));
    CUDA_CHECK(cudaGetLastError());
}

void lossless_inkling2_swiglu(const Tensor& gate_up, const Tensor& gamma, float gate_cap, float up_cap, Tensor& out,
                              cudaStream_t stream) {
    require(is_matrix(gate_up) && is_matrix(out) && gate_up.dtype == DType::BF16 && out.dtype == DType::BF16 &&
                gate_up.ne[0] == 2 * out.ne[0] && gate_up.ne[1] >= out.ne[1],
            "swiglu: gate_up BF16 [2 I, C] and out BF16 [I, C]");
    const int I = static_cast<int>(out.ne[0]), C = static_cast<int>(out.ne[1]);
    require(gamma.dtype == DType::FP32 && gamma.numel() >= C, "swiglu: gamma FP32 [C]");
    const std::int64_t n = static_cast<std::int64_t>(I) * C;
    if (n == 0) { return; }
    swiglu_kernel<<<grid_for(n, 256), 256, 0, stream>>>(static_cast<const std::uint16_t*>(gate_up.data),
                                                        column_stride(gate_up), static_cast<const float*>(gamma.data),
                                                        gate_cap, up_cap, static_cast<std::uint16_t*>(out.data),
                                                        column_stride(out), I, C);
    CUDA_CHECK(cudaGetLastError());
}

void lossless_inkling2_combine(const Tensor& y, const Tensor& slot_column, int top_k, Tensor& out, cudaStream_t stream) {
    require(is_matrix(y) && is_matrix(out) && y.dtype == DType::BF16 && out.dtype == DType::BF16 && y.ne[0] == out.ne[0],
            "combine: y BF16 [H, entries] and out BF16 [H, T]");
    const int H = static_cast<int>(out.ne[0]), T = static_cast<int>(out.ne[1]);
    require(slot_column.dtype == DType::I32 && slot_column.numel() >= static_cast<std::int64_t>(T) * top_k,
            "combine: slot_column I32 [T top_k]");
    const std::int64_t n = static_cast<std::int64_t>(H) * T;
    if (n == 0) { return; }
    combine_kernel<<<grid_for(n, 256), 256, 0, stream>>>(static_cast<const std::uint16_t*>(y.data), column_stride(y),
                                                         static_cast<const std::int32_t*>(slot_column.data), top_k,
                                                         static_cast<std::uint16_t*>(out.data), column_stride(out), H, T);
    CUDA_CHECK(cudaGetLastError());
}

void lossless_inkling2_heads(const Tensor& x, int heads, bool transposed, int stride, Tensor& out, cudaStream_t stream) {
    require(is_matrix(x) && x.dtype == DType::BF16 && out.dtype == DType::BF16 && heads > 0 && x.ne[0] % heads == 0 &&
                stride >= x.ne[1] && out.numel() >= static_cast<std::int64_t>(x.ne[0]) * stride,
            "heads: x BF16 [D heads, T] and out BF16 planes of D x stride");
    const int D = static_cast<int>(x.ne[0]) / heads, T = static_cast<int>(x.ne[1]);
    const std::int64_t n = x.numel();
    if (n == 0) { return; }
    heads_kernel<<<grid_for(n, 256), 256, 0, stream>>>(static_cast<const std::uint16_t*>(x.data), column_stride(x), D,
                                                       heads, T, transposed, stride,
                                                       static_cast<std::uint16_t*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

void lossless_inkling2_ring_load(const std::uint16_t* ring, int rows, int first, int count, Tensor& out,
                                 cudaStream_t stream) {
    require(ring != nullptr && rows > 0 && count >= 0 && is_matrix(out) && out.dtype == DType::BF16 && out.ne[1] >= count,
            "ring_load: a ring of positive rows and out BF16 [C, >= count]");
    if (count == 0) { return; }
    const int C = static_cast<int>(out.ne[0]);
    const std::int64_t n = static_cast<std::int64_t>(C) * count;
    ring_load_kernel<<<grid_for(n, 256), 256, 0, stream>>>(ring, rows, C, first, count,
                                                           static_cast<std::uint16_t*>(out.data), column_stride(out));
    CUDA_CHECK(cudaGetLastError());
}

void lossless_inkling2_ring_store(const Tensor& x, int first, std::uint16_t* ring, int rows, cudaStream_t stream) {
    require(ring != nullptr && rows > 0 && first >= 0 && is_matrix(x) && x.dtype == DType::BF16,
            "ring_store: x BF16 [C, T] at a position >= 0 and a ring of positive rows");
    const int C = static_cast<int>(x.ne[0]), T = static_cast<int>(x.ne[1]);
    const int from = T > rows ? T - rows : 0, count = T - from;
    if (count <= 0) { return; }
    const std::int64_t n = static_cast<std::int64_t>(C) * count;
    ring_store_kernel<<<grid_for(n, 256), 256, 0, stream>>>(static_cast<const std::uint16_t*>(x.data), column_stride(x), C,
                                                            from, count, first, rows, ring);
    CUDA_CHECK(cudaGetLastError());
}

void lossless_inkling2_keys_load(const LosslessInkling2Keys& source, int first, int count, int stride, Tensor& kh,
                                 Tensor& vt, cudaStream_t stream) {
    const std::int64_t plane_elements = static_cast<std::int64_t>(kKeyDim) * stride;
    require(source.k != nullptr && source.v != nullptr && source.dim == kKeyDim && source.heads > 0 && first >= 0 &&
                count >= 0 && stride >= count && (source.table != nullptr || source.ring_pages > 0),
            "keys_load: 128-wide rows of a ring or a pool, positions >= 0, planes of >= count columns");
    require(kh.dtype == DType::BF16 && vt.dtype == DType::BF16 && kh.numel() >= plane_elements * source.heads &&
                vt.numel() >= plane_elements * source.heads,
            "keys_load: BF16 planes of D x stride per head");
    if (count == 0) { return; }
    const dim3 grid(static_cast<unsigned>((count + kKeyUnitRows - 1) / kKeyUnitRows), static_cast<unsigned>(source.heads));
    keys_load_kernel<<<grid, 256, 0, stream>>>(source, first, count, stride, static_cast<std::uint16_t*>(kh.data),
                                               static_cast<std::uint16_t*>(vt.data));
    CUDA_CHECK(cudaGetLastError());
}

void lossless_inkling2_keys_store(const Tensor& k, const Tensor& v, int first, const LosslessInkling2Keys& destination,
                                  cudaStream_t stream) {
    const LosslessInkling2Keys& d = destination;
    require(d.k != nullptr && d.v != nullptr && d.dim == kKeyDim && d.heads > 0 && first >= 0 &&
                (d.table != nullptr || d.ring_pages > 0),
            "keys_store: 128-wide rows of a ring or a pool at a position >= 0");
    require(is_matrix(k) && is_matrix(v) && k.dtype == DType::BF16 && v.dtype == DType::BF16 &&
                k.ne[0] == static_cast<std::int64_t>(d.heads) * kKeyDim && v.ne[0] == k.ne[0] && v.ne[1] == k.ne[1],
            "keys_store: k and v BF16 [dim heads, T]");
    const int T = static_cast<int>(k.ne[1]);
    const int ring_positions = d.table != nullptr ? T : d.ring_pages * kKeyUnitRows;
    const int from = T > ring_positions ? T - ring_positions : 0, count = T - from;
    if (count <= 0) { return; }
    const std::int64_t n = static_cast<std::int64_t>(d.heads) * kKeyDim * count;
    keys_store_kernel<<<grid_for(n, 256), 256, 0, stream>>>(static_cast<const std::uint16_t*>(k.data), column_stride(k),
                                                            static_cast<const std::uint16_t*>(v.data), column_stride(v),
                                                            from, count, first, d);
    CUDA_CHECK(cudaGetLastError());
}

bool lossless_inkling2_take_undecided() {
    unsigned int flag = 0;
    CUDA_CHECK(cudaMemcpyFromSymbol(&flag, g_undecided, sizeof(flag)));
    if (flag != 0) {
        const unsigned int zero = 0;
        CUDA_CHECK(cudaMemcpyToSymbol(g_undecided, &zero, sizeof(zero)));
    }
    return flag != 0;
}

void lossless_inkling2_force_exact_fallback(bool on) {
    CUDA_CHECK(cudaMemcpyToSymbol(g_force_fallback, &on, sizeof(on)));
}

void lossless_inkling2_softcap(const Tensor& x, float cap, Tensor& out, cudaStream_t stream) {
    require(x.dtype == DType::FP32 && out.dtype == DType::FP32 && x.is_contiguous() && out.is_contiguous() &&
                x.numel() == out.numel() && cap > 0.0F,
            "softcap: FP32 operands of one shape and a positive cap");
    const std::int64_t n = x.numel();
    if (n == 0) { return; }
    softcap_kernel<<<stride_grid(n), 256, 0, stream>>>(static_cast<const float*>(x.data), cap,
                                                       static_cast<float*>(out.data), n);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace infernix::ops
