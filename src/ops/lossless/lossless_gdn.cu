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
// run the sequence in passes of kChunk tokens: a pass first stages its tokens' normalized q and k (one
// warp per token: the L2 norms' binary32 squares of BF16 values are exact, summed in a fixed-point
// accumulator), gates and this CTA's values in shared memory, then runs them token by token, which needs
// no block barrier: only the state update is sequential. Sums of binary32 values go through a 320-bit
// fixed point from 2^-150 (lossless::F32Sum) and are rounded once; a non-finite term gives IEEE's result.

namespace infernix::ops {
namespace {

using lossless::add_i128;
using lossless::F32Sum;
using lossless::I128;
using lossless::magnitude_i128;
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
constexpr int kChunk         = 32;                // tokens staged per pass
// A token's staged q or k: slice s's 16 rows at s kSliceStride, so the 8 slices of a column read their
// rows as float4 from distinct banks.
constexpr int kSliceStride = kRowsPerThread + 4;
constexpr int kStagedRow   = kSlices * kSliceStride;

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string("lossless_gated_delta_rule: ") + message); }
}

__device__ __forceinline__ float bf16_bits_float(std::uint16_t bits) {
    return __uint_as_float(static_cast<std::uint32_t>(bits) << 16);
}

// fp32(1 / sqrt(fp32(sum of squares) + 1e-6)) of one 128-element BF16 vector held four per lane
// (element lane + 32 i), the sum exact before its rounding; every lane of the warp returns it.
__device__ __forceinline__ float warp_inverse_norm(const float (&x)[4]) {
    F32Sum s;
    s.clear();
#pragma unroll
    for (int i = 0; i < 4; ++i) { s.add(__fmul_rn(x[i], x[i])); } // exact: a BF16 square has at most 16 significant bits
    s.acc.warp_sum();
    // Non-finite squares: any lane's flags.
    s.nan     = __any_sync(0xFFFFFFFFU, s.nan) != 0;
    s.pos_inf = __any_sync(0xFFFFFFFFU, s.pos_inf) != 0;
    return lossless::cr_rsqrt(__fadd_rn(s.result(), 1e-6F));
}

// A staged q or k element's position: element e is slice e / 16's row e % 16.
__device__ __forceinline__ int staged(int e) { return e / kRowsPerThread * kSliceStride + e % kRowsPerThread; }

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

// A column's dot product: the binary32 sum of its 8 slice threads' 16 terms each (128), exact before its
// one rounding, as group_result(F32Sum) gives it. Fast path: every term M 2^(f - 150) (M < 2^24, f its
// exponent field, 1 for subnormals) placed exactly in a 128-bit integer whose unit is 2^(F - 150 - kLowBits),
// F the group's largest field: a term's shift f - F + kLowBits lies in [0, kLowBits], every shift a register
// operation (F32Sum's chunk index is dynamic, so its words live in local memory), 128 terms below 2^(25 +
// kLowBits + 7) fit, and the 8 threads add their integers with 4 shuffles a level. Where any lane of the warp
// holds a non-finite term or a nonzero term more than kLowBits binades below its group's largest, the whole
// warp takes F32Sum (group_result shuffles over the warp).
constexpr int kLowBits = 88;

__device__ __forceinline__ void add_term(I128& acc, long long m, int shift) {
    unsigned long long lo;
    long long hi;
    if (shift >= 64) {
        lo = 0;
        hi = m * (1LL << (shift - 64)); // |m| < 2^24, shift - 64 <= 24
    } else if (shift > 0) {
        lo = static_cast<unsigned long long>(m) << shift;
        hi = m >> (64 - shift);
    } else {
        lo = static_cast<unsigned long long>(m);
        hi = m < 0 ? -1 : 0;
    }
    acc = add_i128(acc, I128{lo, hi});
}

__device__ __forceinline__ int term_field(std::uint32_t bits) {
    const int field = static_cast<int>((bits >> 23) & 0xFF);
    return field == 0 ? 1 : field;
}

__device__ __forceinline__ float group_sum(const float (&t)[kRowsPerThread]) {
    int top = 1;
    bool special = false;
#pragma unroll
    for (int r = 0; r < kRowsPerThread; ++r) {
        const std::uint32_t bits = __float_as_uint(t[r]);
        special |= ((bits >> 23) & 0xFF) == 0xFF;
        if ((bits & 0x7FFFFFFFU) != 0) { top = max(top, term_field(bits)); }
    }
#pragma unroll
    for (int o = 1; o < kSlices; o <<= 1) { top = max(top, __shfl_xor_sync(0xFFFFFFFFU, top, o)); }
    bool low = false;
#pragma unroll
    for (int r = 0; r < kRowsPerThread; ++r) {
        const std::uint32_t bits = __float_as_uint(t[r]);
        low |= (bits & 0x7FFFFFFFU) != 0 && term_field(bits) < top - kLowBits;
    }
    if (__any_sync(0xFFFFFFFFU, special || low)) {
        F32Sum exact;
        exact.clear();
#pragma unroll
        for (int r = 0; r < kRowsPerThread; ++r) { exact.add(t[r]); }
        return group_result(exact);
    }
    I128 acc{0ULL, 0LL};
#pragma unroll
    for (int r = 0; r < kRowsPerThread; ++r) {
        const std::uint32_t bits = __float_as_uint(t[r]);
        if ((bits & 0x7FFFFFFFU) == 0) { continue; }
        const long long M = ((bits >> 23) & 0xFF) == 0 ? (bits & 0x7FFFFF) : (0x800000 | (bits & 0x7FFFFF));
        add_term(acc, (bits >> 31) ? -M : M, term_field(bits) - top + kLowBits);
    }
#pragma unroll
    for (int o = 1; o < kSlices; o <<= 1) {
        const unsigned long long lo = __shfl_xor_sync(0xFFFFFFFFU, acc.lo, o);
        const long long hi          = __shfl_xor_sync(0xFFFFFFFFU, acc.hi, o);
        acc                         = add_i128(acc, I128{lo, hi});
    }
    std::uint32_t mag[4];
    const bool negative = magnitude_i128(acc, mag);
    return __uint_as_float(round_magnitude(mag, top - 150 - kLowBits, negative, false));
}

// The state of sequence b: FP32 [128 (k), 128 (v), Hv] at `base` (null: zeros, not kept) or, with slots,
// at slot slots[b] of a pool of such states.
__device__ __forceinline__ std::size_t state_offset(const int* slots, int b, int value_heads) {
    return slots != nullptr ? static_cast<std::size_t>(slots[b]) * kDk * kDv * value_heads : 0;
}

// blockIdx.y: the sequence (columns [b width, (b + 1) width)). Two CTAs per SM (at most 128 registers; the
// staged tokens take about 44 KB of shared memory each): the 27B's 48 value heads are 192 CTAs, one wave on
// 170 SMs, not 1.13 waves at one CTA per SM.
constexpr int kMinCtasPerSm = 2;
__global__ __launch_bounds__(kThreads, kMinCtasPerSm) void gated_delta_rule_kernel(const std::uint16_t* query, const std::uint16_t* key,
                                                                    const std::uint16_t* value, const float* g,
                                                                    const std::uint16_t* beta, const float* state_in,
                                                                    float* state_out, const int* source_slots,
                                                                    const int* destination_slots, std::uint16_t* out,
                                                                    int key_heads, int value_heads, int width, int q_ld,
                                                                    int k_ld, int v_ld) {
    __shared__ __align__(16) float qhat[kChunk][kStagedRow];
    __shared__ __align__(16) float khat[kChunk][kStagedRow];
    __shared__ float vs[kChunk][kColumnsPerCta];
    __shared__ float decay_s[kChunk], beta_s[kChunk];
    const int hv = static_cast<int>(blockIdx.x) / kGroups, group = static_cast<int>(blockIdx.x) % kGroups;
    const int b  = static_cast<int>(blockIdx.y);
    const int hk = hv / (value_heads / key_heads);
    const int tid = static_cast<int>(threadIdx.x), lane = tid & 31, warp = tid >> 5;
    const int slice = lane % kSlices;
    const int j  = group * kColumnsPerCta + warp * (32 / kSlices) + lane / kSlices; // this thread's column
    const int i0 = slice * kRowsPerThread;                                           // its first state row
    const int jl = j - group * kColumnsPerCta;                                       // its column in the CTA
    float S[kRowsPerThread]; // S_(i0 + r, j), stored at j 128 + i0 + r of its head
    const std::size_t head = static_cast<std::size_t>(hv) * kDk * kDv + static_cast<std::size_t>(j) * kDk + i0;
    if (state_in != nullptr) {
        const float4* st = reinterpret_cast<const float4*>(state_in + state_offset(source_slots, b, value_heads) + head);
#pragma unroll
        for (int r = 0; r < kRowsPerThread / 4; ++r) {
            const float4 v4 = st[r];
            S[4 * r]        = v4.x;
            S[4 * r + 1]    = v4.y;
            S[4 * r + 2]    = v4.z;
            S[4 * r + 3]    = v4.w;
        }
    } else {
#pragma unroll
        for (int r = 0; r < kRowsPerThread; ++r) { S[r] = 0.0F; }
    }
    const float scale = 0x1.6a09e6p-4F; // fp32(1 / sqrt(128)): HF's double scale rounded to binary32
    for (int c0 = 0; c0 < width; c0 += kChunk) {
        const int n = min(kChunk, width - c0);
        // Stage the pass: each warp normalizes its tokens' q and k (L2 norms of the head's 128 elements).
        for (int tt = warp; tt < n; tt += kWarps) {
            const std::size_t t = static_cast<std::size_t>(b) * width + c0 + tt;
            float qe[4], ke[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                qe[i] = bf16_bits_float(query[t * q_ld + hk * kDk + lane + 32 * i]);
                ke[i] = bf16_bits_float(key[t * k_ld + hk * kDk + lane + 32 * i]);
            }
            const float rq = warp_inverse_norm(qe), rk = warp_inverse_norm(ke);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                qhat[tt][staged(lane + 32 * i)] = __fmul_rn(__fmul_rn(qe[i], rq), scale);
                khat[tt][staged(lane + 32 * i)] = __fmul_rn(ke[i], rk);
            }
            if (lane == 0) {
                decay_s[tt] = lossless::cr_exp(g[t * value_heads + hv]);
                beta_s[tt]  = bf16_bits_float(beta[t * value_heads + hv]);
            }
        }
        for (int e = tid; e < n * kColumnsPerCta; e += kThreads) {
            const int tt = e / kColumnsPerCta, c = e % kColumnsPerCta;
            const std::size_t t = static_cast<std::size_t>(b) * width + c0 + tt;
            vs[tt][c] = bf16_bits_float(value[t * v_ld + hv * kDv + group * kColumnsPerCta + c]);
        }
        __syncthreads();
        for (int tt = 0; tt < n; ++tt) {
            const float decay = decay_s[tt], bt = beta_s[tt], vj = vs[tt][jl];
            float kh[kRowsPerThread], qh[kRowsPerThread];
#pragma unroll
            for (int r = 0; r < kRowsPerThread / 4; ++r) {
                const float4 k4 = *reinterpret_cast<const float4*>(&khat[tt][slice * kSliceStride + 4 * r]);
                const float4 q4 = *reinterpret_cast<const float4*>(&qhat[tt][slice * kSliceStride + 4 * r]);
                kh[4 * r] = k4.x, kh[4 * r + 1] = k4.y, kh[4 * r + 2] = k4.z, kh[4 * r + 3] = k4.w;
                qh[4 * r] = q4.x, qh[4 * r + 1] = q4.y, qh[4 * r + 2] = q4.z, qh[4 * r + 3] = q4.w;
            }
            float terms[kRowsPerThread];
#pragma unroll
            for (int r = 0; r < kRowsPerThread; ++r) {
                S[r]     = __fmul_rn(S[r], decay);
                terms[r] = __fmul_rn(S[r], kh[r]);
            }
            const float delta = __fmul_rn(__fsub_rn(vj, group_sum(terms)), bt);
#pragma unroll
            for (int r = 0; r < kRowsPerThread; ++r) {
                S[r]     = __fadd_rn(S[r], __fmul_rn(kh[r], delta));
                terms[r] = __fmul_rn(S[r], qh[r]);
            }
            const float oj = group_sum(terms);
            if (slice == 0) {
                out[(static_cast<std::size_t>(b) * width + c0 + tt) * value_heads * kDv + hv * kDv + j] =
                    __bfloat16_as_ushort(__float2bfloat16_rn(oj));
            }
        }
        __syncthreads(); // the next pass overwrites the staged tokens
    }
    if (state_out != nullptr) {
        float4* so = reinterpret_cast<float4*>(state_out + state_offset(destination_slots, b, value_heads) + head);
#pragma unroll
        for (int r = 0; r < kRowsPerThread / 4; ++r) {
            so[r] = make_float4(S[4 * r], S[4 * r + 1], S[4 * r + 2], S[4 * r + 3]);
        }
    }
}

struct Shape {
    int key_heads, value_heads, columns;
};

Shape check(const Tensor& query, const Tensor& key, const Tensor& value, const Tensor& g, const Tensor& beta,
            const Tensor& out) {
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
    return {key_heads, value_heads, columns};
}

void launch(const Tensor& query, const Tensor& key, const Tensor& value, const Tensor& g, const Tensor& beta,
            const float* state_in, float* state_out, const int* source_slots, const int* destination_slots,
            const Shape& shape, int sequences, Tensor& out, cudaStream_t stream) {
    if (shape.columns == 0) { return; }
    gated_delta_rule_kernel<<<dim3(shape.value_heads * kGroups, sequences), kThreads, 0, stream>>>(
        static_cast<const std::uint16_t*>(query.data), static_cast<const std::uint16_t*>(key.data),
        static_cast<const std::uint16_t*>(value.data), static_cast<const float*>(g.data),
        static_cast<const std::uint16_t*>(beta.data), state_in, state_out, source_slots, destination_slots,
        static_cast<std::uint16_t*>(out.data), shape.key_heads, shape.value_heads, shape.columns / sequences,
        static_cast<int>(query.nb[1] / 2), static_cast<int>(key.nb[1] / 2), static_cast<int>(value.nb[1] / 2));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void lossless_gated_delta_rule(const Tensor& query, const Tensor& key, const Tensor& value, const Tensor& g,
                               const Tensor& beta, float* state, Tensor& out, cudaStream_t stream) {
    const Shape shape = check(query, key, value, g, beta, out);
    require(reinterpret_cast<std::uintptr_t>(state) % 16 == 0, "the state is 16-byte aligned");
    launch(query, key, value, g, beta, state, state, nullptr, nullptr, shape, 1, out, stream);
}

void lossless_gated_delta_rule_slots(const Tensor& query, const Tensor& key, const Tensor& value, const Tensor& g,
                                     const Tensor& beta, int sequences, Tensor& states, const Tensor& source_slots,
                                     const Tensor& destination_slots, Tensor& out, cudaStream_t stream) {
    const Shape shape = check(query, key, value, g, beta, out);
    require(sequences >= 1 && shape.columns % sequences == 0, "the columns are `sequences` equal runs");
    require(states.dtype == DType::FP32 && states.is_contiguous() && states.ne[0] == kDk && states.ne[1] == kDv &&
                states.ne[2] == shape.value_heads && reinterpret_cast<std::uintptr_t>(states.data) % 16 == 0,
            "states FP32 [128, 128, Hv, slots], 16-byte aligned");
    require(source_slots.dtype == DType::I32 && destination_slots.dtype == DType::I32 &&
                source_slots.is_contiguous() && destination_slots.is_contiguous() &&
                source_slots.numel() >= sequences && destination_slots.numel() >= sequences,
            "source and destination slots I32 [B]");
    launch(query, key, value, g, beta, static_cast<const float*>(states.data), static_cast<float*>(states.data),
           static_cast<const int*>(source_slots.data), static_cast<const int*>(destination_slots.data), shape,
           sequences, out, stream);
}

} // namespace infernix::ops

// This unit's copy of the noise twin's seed (ops/lossless/noise.cuh).
INFERNIX_LOSSLESS_NOISE_UNIT(gdn)
