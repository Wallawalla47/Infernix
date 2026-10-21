#include "infernix/ops/lossless_ops.h"

#include "core/device.h"
#include "ops/lossless/cr_math.cuh"
#include "ops/lossless/wide.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>
#include <string>

// The --lossless route's gated delta rule (include/infernix/ops/lossless_ops.h): Hugging Face's recurrence
// operation by operation, every binary32 product and sum correctly rounded. Each value head's 128 state
// columns are split over 4 CTAs of 32 columns; a column's 128 state rows over 8 threads of one warp (16
// rows each, in registers), which combine their partial sums exactly before the one rounding. The CTAs
// run the sequence token by token; the L2 norms of q and k are block reductions (binary32 squares of
// BF16 values are exact) in a fixed-point accumulator. Sums of binary32 values go through a 320-bit
// fixed point from 2^-150 (lossless::F32Sum) and are rounded once; a non-finite term gives IEEE's result.

namespace infernix::ops {
namespace {

using lossless::F32Sum;
using lossless::round_magnitude;
using lossless::Wide;

constexpr int kDk = 128, kDv = 128;
constexpr int kSlices        = 8;                 // threads per state column
constexpr int kRowsPerThread = kDk / kSlices;     // 16
constexpr int kThreads       = 256;               // 32 columns x 8 slices
constexpr int kColumnsPerCta = kThreads / kSlices;
constexpr int kGroups        = kDv / kColumnsPerCta; // CTAs per value head
constexpr int kWarps         = kThreads / 32;
constexpr int kSumWords      = lossless::kF32SumWords;

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string("lossless_gated_delta_rule: ") + message); }
}

__device__ __forceinline__ float bf16_bits_float(std::uint16_t bits) {
    return __uint_as_float(static_cast<std::uint32_t>(bits) << 16);
}

// Block-wide fp32(sum of squares) of up to 256 BF16 values (one per thread, zeros elsewhere), exact
// before the rounding; every thread returns it.
__device__ float block_sum_squares(float x, long long (&scratch)[kWarps][kSumWords]) {
    const int tid = static_cast<int>(threadIdx.x), lane = tid & 31, warp = tid >> 5;
    F32Sum s;
    s.clear();
    s.add(__fmul_rn(x, x)); // exact: a BF16 square has at most 16 significant bits
    s.acc.warp_sum();
    if (lane == 0) {
#pragma unroll
        for (int i = 0; i < kSumWords; ++i) { scratch[warp][i] = s.acc.chunk[i]; }
    }
    // Non-finite squares: any lane's flags.
    const bool nan_any = __syncthreads_or(s.nan ? 1 : 0) != 0;
    const bool inf_any = __syncthreads_or(s.pos_inf ? 1 : 0) != 0;
    F32Sum total;
    total.clear();
#pragma unroll
    for (int i = 0; i < kSumWords; ++i) {
        long long v = 0;
#pragma unroll
        for (int w = 0; w < kWarps; ++w) { v += scratch[w][i]; }
        total.acc.chunk[i] = v;
    }
    total.nan     = nan_any;
    total.pos_inf = inf_any;
    __syncthreads(); // scratch is reused
    return total.result();
}

// The 8 slice threads' partial sums of a column added exactly (lanes xor 1, 2, 4 stay in the group);
// every thread of the group returns the rounded total.
__device__ __forceinline__ float group_result(F32Sum& part) {
#pragma unroll
    for (int i = 0; i < kSumWords; ++i) {
#pragma unroll
        for (int o = 1; o < kSlices; o <<= 1) { part.acc.chunk[i] += __shfl_xor_sync(0xFFFFFFFFU, part.acc.chunk[i], o); }
    }
    int flags = (part.nan ? 1 : 0) | (part.pos_inf ? 2 : 0) | (part.neg_inf ? 4 : 0);
#pragma unroll
    for (int o = 1; o < kSlices; o <<= 1) { flags |= __shfl_xor_sync(0xFFFFFFFFU, flags, o); }
    part.nan     = (flags & 1) != 0;
    part.pos_inf = (flags & 2) != 0;
    part.neg_inf = (flags & 4) != 0;
    return part.result();
}

__global__ __launch_bounds__(kThreads) void gated_delta_rule_kernel(const std::uint16_t* query, const std::uint16_t* key,
                                                                    const std::uint16_t* value, const float* g,
                                                                    const std::uint16_t* beta, float* state,
                                                                    std::uint16_t* out, int key_heads, int value_heads,
                                                                    int columns, int q_ld, int k_ld, int v_ld) {
    __shared__ float qhat[kDk], khat[kDk];
    __shared__ long long scratch[kWarps][kSumWords];
    const int hv = static_cast<int>(blockIdx.x) / kGroups, group = static_cast<int>(blockIdx.x) % kGroups;
    const int hk = hv / (value_heads / key_heads);
    const int tid = static_cast<int>(threadIdx.x), lane = tid & 31, warp = tid >> 5;
    const int slice = lane % kSlices;
    const int j  = group * kColumnsPerCta + warp * (32 / kSlices) + lane / kSlices; // this thread's column
    const int i0 = slice * kRowsPerThread;                                           // its first state row
    float S[kRowsPerThread]; // S_(i0 + r, j)
    const float* st = state != nullptr ? state + static_cast<std::size_t>(hv) * kDk * kDv : nullptr;
#pragma unroll
    for (int r = 0; r < kRowsPerThread; ++r) { S[r] = st != nullptr ? st[(i0 + r) * kDv + j] : 0.0F; }
    const float scale = 0x1.6a09e6p-4F; // fp32(1 / sqrt(128)): HF's double scale rounded to binary32
    for (int t = 0; t < columns; ++t) {
        // L2 norms: threads 0-127 hold element tid of q and k (d_k = 128).
        const float qe = tid < kDk ? bf16_bits_float(query[static_cast<std::size_t>(t) * q_ld + hk * kDk + tid]) : 0.0F;
        const float ke = tid < kDk ? bf16_bits_float(key[static_cast<std::size_t>(t) * k_ld + hk * kDk + tid]) : 0.0F;
        const float rq = lossless::cr_rsqrt(__fadd_rn(block_sum_squares(qe, scratch), 1e-6F));
        const float rk = lossless::cr_rsqrt(__fadd_rn(block_sum_squares(ke, scratch), 1e-6F));
        if (tid < kDk) {
            qhat[tid] = __fmul_rn(__fmul_rn(qe, rq), scale);
            khat[tid] = __fmul_rn(ke, rk);
        }
        __syncthreads();
        const float decay = lossless::cr_exp(g[static_cast<std::size_t>(t) * value_heads + hv]);
        const float b     = bf16_bits_float(beta[static_cast<std::size_t>(t) * value_heads + hv]);
        const float vj    = bf16_bits_float(value[static_cast<std::size_t>(t) * v_ld + hv * kDv + j]);
        F32Sum u;
        u.clear();
#pragma unroll
        for (int r = 0; r < kRowsPerThread; ++r) {
            S[r] = __fmul_rn(S[r], decay);
            u.add(__fmul_rn(S[r], khat[i0 + r]));
        }
        const float delta = __fmul_rn(__fsub_rn(vj, group_result(u)), b);
        F32Sum o;
        o.clear();
#pragma unroll
        for (int r = 0; r < kRowsPerThread; ++r) {
            S[r] = __fadd_rn(S[r], __fmul_rn(khat[i0 + r], delta));
            o.add(__fmul_rn(S[r], qhat[i0 + r]));
        }
        const float oj = group_result(o);
        if (slice == 0) {
            out[static_cast<std::size_t>(t) * value_heads * kDv + hv * kDv + j] =
                __bfloat16_as_ushort(__float2bfloat16_rn(oj));
        }
        __syncthreads(); // qhat and khat are rewritten next token
    }
    if (state != nullptr) {
        float* so = state + static_cast<std::size_t>(hv) * kDk * kDv;
#pragma unroll
        for (int r = 0; r < kRowsPerThread; ++r) { so[(i0 + r) * kDv + j] = S[r]; }
    }
}

} // namespace

void lossless_gated_delta_rule(const Tensor& query, const Tensor& key, const Tensor& value, const Tensor& g,
                               const Tensor& beta, float* state, Tensor& out, cudaStream_t stream) {
    require(query.dtype == DType::BF16 && key.dtype == DType::BF16 && value.dtype == DType::BF16 &&
                out.dtype == DType::BF16 && g.dtype == DType::FP32 && beta.dtype == DType::BF16 &&
                query.nb[0] == 2 && key.nb[0] == 2 && value.nb[0] == 2 && query.nb[1] % 2 == 0 &&
                key.nb[1] % 2 == 0 && value.nb[1] % 2 == 0 && query.ne[2] == 1 && key.ne[2] == 1 &&
                value.ne[2] == 1 && out.is_contiguous() && g.is_contiguous() && beta.is_contiguous(),
            "q, k, v BF16 columns (strided), out BF16, g FP32 and beta BF16 contiguous");
    const int columns = query.ne[1];
    require(query.ne[0] % kDk == 0 && key.ne[0] == query.ne[0] && value.ne[0] % kDv == 0 && out.ne[0] == value.ne[0] &&
                key.ne[1] == columns && value.ne[1] == columns && out.ne[1] == columns,
            "q, k [128 Hk, T] and v, out [128 Hv, T]");
    const int key_heads = query.ne[0] / kDk, value_heads = value.ne[0] / kDv;
    require(value_heads % key_heads == 0 && g.ne[0] == value_heads && beta.ne[0] == value_heads &&
                g.ne[1] == columns && beta.ne[1] == columns,
            "g and beta [Hv, T], Hv a multiple of Hk");
    gated_delta_rule_kernel<<<value_heads * kGroups, kThreads, 0, stream>>>(
        static_cast<const std::uint16_t*>(query.data), static_cast<const std::uint16_t*>(key.data),
        static_cast<const std::uint16_t*>(value.data), static_cast<const float*>(g.data),
        static_cast<const std::uint16_t*>(beta.data), state, static_cast<std::uint16_t*>(out.data), key_heads,
        value_heads, columns, static_cast<int>(query.nb[1] / 2), static_cast<int>(key.nb[1] / 2),
        static_cast<int>(value.nb[1] / 2));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace infernix::ops
