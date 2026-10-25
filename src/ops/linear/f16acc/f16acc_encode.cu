// f16-accumulate-prefill's activation encoding (f16acc_plan.h): each BF16 column as FP16 values under a
// power-of-two scale that puts the column's maximum in [1, 2), and the diagnostic scan of that encoding
// (INFERNIX_TMP_ACTIVATION_SCAN, the Qwen3.8-Flash-Next Forward). One CTA per column; the encoder keeps
// its column in registers between the maximum and the stores, so it reads x once.
#include "ops/linear/f16acc/f16acc_plan.h"

#include "core/device.h"
#include "core/pdl.cuh"
#include "ops/common/math.cuh"
#include "ops/common/warp.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <stdexcept>

namespace infernix::ops::detail {
namespace {

// s with maximum * 2^s in [1, 2) (s = -floor(log2 maximum)), within [-126, 126] so that 2^s and 2^-s
// are normal FP32 values; 0 for a zero column. A BF16-subnormal maximum clamps to 126 and lands below
// 1, an infinite one to -126: neither can overflow the encoding.
__device__ __forceinline__ int f16acc_exponent(float maximum) {
    if (maximum == 0.0F) return 0;
    const int biased = static_cast<int>((__float_as_uint(maximum) >> 23) & 0xFFU);
    const int s      = 127 - biased;
    return s < -126 ? -126 : (s > 126 ? 126 : s);
}

__device__ __forceinline__ float f16acc_power(int exponent) {
    return __uint_as_float(static_cast<unsigned>(127 + exponent) << 23);
}

// The maximum over the CTA; every thread returns it.
template <int Threads>
__device__ __forceinline__ float f16acc_block_max(float value, float* warp_maxima) {
    constexpr int kWarps = Threads / 32;
    value                = warp_max(value);
    if ((threadIdx.x & 31U) == 0U) { warp_maxima[threadIdx.x >> 5] = value; }
    __syncthreads();
    float result = 0.0F;
#pragma unroll
    for (int warp = 0; warp < kWarps; ++warp) { result = fmaxf(result, warp_maxima[warp]); }
    return result;
}

// Four BF16 values (8 bytes) per vector; Threads * Vectors vectors cover the column. Each 16-value block is
// written in the ldmatrix B route's order (f16acc_mma.cuh): values 4q, 4q + 1 go to positions 2q, 2q + 1
// and 4q + 2, 4q + 3 to 8 + 2q, 9 + 2q, so an m16n8k16 A fragment's slots (2t, 2t + 1) and (2t + 8, 2t + 9)
// hold values 4t .. 4t + 3, the four codes one ldmatrix lane receives of a weight row.
template <int K, int Threads>
__global__ __launch_bounds__(Threads) void f16acc_encode_kernel(const __nv_bfloat16* __restrict__ input,
                                                                __half* __restrict__ output,
                                                                float* __restrict__ inverses) {
    static_assert(Threads % 32 == 0 && Threads <= 1024);
    static_assert(K % (4 * Threads) == 0);
    constexpr int kVectors = K / 4 / Threads;
    __shared__ float warp_maxima[Threads / 32];

    pdl::enter();
    const int column        = static_cast<int>(blockIdx.x);
    const int tid           = static_cast<int>(threadIdx.x);
    const std::int64_t base = static_cast<std::int64_t>(column) * K;
    const auto* source      = reinterpret_cast<const uint2*>(input + base);
    auto* destination       = reinterpret_cast<uint2*>(output + base);

    float values[kVectors][4];
    float maximum = 0.0F;
#pragma unroll
    for (int item = 0; item < kVectors; ++item) {
        const uint2 packed = source[tid + item * Threads];
        const float2 low   = bf16x2_bits_to_float2(packed.x);
        const float2 high  = bf16x2_bits_to_float2(packed.y);
        values[item][0]    = low.x;
        values[item][1]    = low.y;
        values[item][2]    = high.x;
        values[item][3]    = high.y;
#pragma unroll
        for (int i = 0; i < 4; ++i) { maximum = fmaxf(maximum, fabsf(values[item][i])); }
    }
    maximum            = f16acc_block_max<Threads>(maximum, warp_maxima);
    const int exponent = f16acc_exponent(maximum);
    const float scale  = f16acc_power(exponent);
#pragma unroll
    for (int item = 0; item < kVectors; ++item) {
        // x * 2^s is exact in FP32; the conversion rounds to nearest (f16acc_plan.h).
        const __half2 low  = __floats2half2_rn(values[item][0] * scale, values[item][1] * scale);
        const __half2 high = __floats2half2_rn(values[item][2] * scale, values[item][3] * scale);
        const int vector = tid + item * Threads;
        // Vector v = 4 * block + q: its two pairs are words q and 4 + q of the block's eight.
        auto* words     = reinterpret_cast<unsigned*>(destination);
        const int block = vector >> 2, q = vector & 3;
        words[block * 8 + q]     = *reinterpret_cast<const unsigned*>(&low);
        words[block * 8 + 4 + q] = *reinterpret_cast<const unsigned*>(&high);
    }
    if (tid == 0) { inverses[column] = f16acc_power(-exponent); }
}

template <int K, int Threads>
void launch_encode(const Tensor& x, F16AccumulateWorkspace workspace, cudaStream_t stream) {
    CUDA_CHECK(pdl::launch_consumer({dim3(x.ne[1]), dim3(Threads), 0, stream}, f16acc_encode_kernel<K, Threads>,
                                    static_cast<const __nv_bfloat16*>(x.data), workspace.x, workspace.inverses));
}

constexpr int kScanThreads = 256;

// One column of the scan: its maximum, the encoder's exponent, then per value whether the FP16
// encoding is exact or flushes, the smallest nonzero magnitude and each 32-value chunk's sum of encoded
// magnitudes (eight consecutive lanes hold a chunk's eight vectors).
__global__ __launch_bounds__(kScanThreads) void f16acc_scan_kernel(const __nv_bfloat16* __restrict__ input, int k,
                                                                   F16AccumulateScanRecord* __restrict__ record) {
    constexpr int kWarps = kScanThreads / 32;
    __shared__ float warp_maxima[kWarps];
    __shared__ float warp_minima[kWarps];
    __shared__ float warp_chunks[kWarps];
    __shared__ unsigned warp_counts[kWarps][3];

    const int column        = static_cast<int>(blockIdx.x);
    const int tid           = static_cast<int>(threadIdx.x);
    const int lane          = tid & 31;
    const int warp          = tid >> 5;
    const auto* source      = reinterpret_cast<const uint2*>(input + static_cast<std::int64_t>(column) * k);
    const int vectors       = k / 4;
    const int rounds        = (vectors + kScanThreads - 1) / kScanThreads;
    const auto load_vector  = [&](int index, float (&out)[4]) {
        const uint2 packed = source[index];
        const float2 low   = bf16x2_bits_to_float2(packed.x);
        const float2 high  = bf16x2_bits_to_float2(packed.y);
        out[0]             = low.x;
        out[1]             = low.y;
        out[2]             = high.x;
        out[3]             = high.y;
    };

    float maximum = 0.0F;
    for (int round = 0; round < rounds; ++round) {
        const int index = tid + round * kScanThreads;
        if (index >= vectors) continue;
        float v[4];
        load_vector(index, v);
        for (int i = 0; i < 4; ++i) { maximum = fmaxf(maximum, fabsf(v[i])); }
    }
    maximum            = f16acc_block_max<kScanThreads>(maximum, warp_maxima);
    const float scale  = f16acc_power(f16acc_exponent(maximum));

    float minimum = __uint_as_float(0x7F800000U), chunk_max = 0.0F;
    unsigned nonzero = 0, inexact = 0, flushed = 0;
    for (int round = 0; round < rounds; ++round) {
        const int index = tid + round * kScanThreads;
        float chunk     = 0.0F;
        if (index < vectors) {
            float v[4];
            load_vector(index, v);
            for (int i = 0; i < 4; ++i) {
                const float scaled  = v[i] * scale;
                const float decoded = __half2float(__float2half_rn(scaled));
                chunk += fabsf(decoded);
                if (v[i] == 0.0F) continue;
                ++nonzero;
                minimum = fminf(minimum, fabsf(v[i]));
                if (decoded != scaled) ++inexact;
                if (decoded == 0.0F) ++flushed;
            }
        }
        // Every lane takes part in the shuffles (rounds is uniform); a chunk is eight vectors.
        chunk     = warp_sum<8>(chunk);
        chunk_max = fmaxf(chunk_max, chunk);
    }
    minimum   = -warp_max(-minimum);
    chunk_max = warp_max(chunk_max);
    nonzero   = warp_sum(nonzero);
    inexact   = warp_sum(inexact);
    flushed   = warp_sum(flushed);
    if (lane == 0) {
        warp_minima[warp]    = minimum;
        warp_chunks[warp]    = chunk_max;
        warp_counts[warp][0] = nonzero;
        warp_counts[warp][1] = inexact;
        warp_counts[warp][2] = flushed;
    }
    __syncthreads();
    if (tid != 0) return;
    unsigned long long totals[3] = {0, 0, 0};
    for (int w = 0; w < kWarps; ++w) {
        minimum   = fminf(minimum, warp_minima[w]);
        chunk_max = fmaxf(chunk_max, warp_chunks[w]);
        for (int i = 0; i < 3; ++i) { totals[i] += warp_counts[w][i]; }
    }
    atomicMax(&record->max_bits, __float_as_uint(maximum));
    atomicMax(&record->max_chunk_bits, __float_as_uint(chunk_max));
    if (maximum > 0.0F) {
        atomicMin(&record->min_column_max_bits, __float_as_uint(maximum));
        atomicMax(&record->max_range_bits, __float_as_uint(maximum / minimum));
    }
    atomicAdd(&record->columns, 1ULL);
    atomicAdd(&record->values, static_cast<unsigned long long>(k));
    atomicAdd(&record->nonzero, totals[0]);
    atomicAdd(&record->inexact, totals[1]);
    atomicAdd(&record->flushed, totals[2]);
}

} // namespace

void launch_f16_accumulate_encode(const Tensor& x, F16AccumulateWorkspace workspace, cudaStream_t stream) {
    if (workspace.x == nullptr || workspace.inverses == nullptr) {
        throw std::invalid_argument("f16 accumulate encode requires caller workspace");
    }
    if (x.dtype != DType::BF16 || x.ne[1] <= 0) {
        throw std::invalid_argument("f16 accumulate encode: x must be BF16 [K, T]");
    }
    switch (x.ne[0]) {
    case 640: return launch_encode<640, 160>(x, workspace, stream);
    case 2560: return launch_encode<2560, 128>(x, workspace, stream);
    case 6144: return launch_encode<6144, 256>(x, workspace, stream);
    default: throw std::invalid_argument("f16 accumulate encode: unsupported K");
    }
}

void launch_f16_accumulate_scan(const Tensor& x, F16AccumulateScanRecord* record, cudaStream_t stream) {
    if (record == nullptr || x.dtype != DType::BF16 || x.ne[1] <= 0 || x.ne[0] <= 0 ||
        x.ne[0] % kF16AccumulateChunk != 0 || !x.is_contiguous()) {
        throw std::invalid_argument("f16 accumulate scan: x must be contiguous BF16 [K, T], K a multiple of 32");
    }
    f16acc_scan_kernel<<<x.ne[1], kScanThreads, 0, stream>>>(static_cast<const __nv_bfloat16*>(x.data), x.ne[0],
                                                             record);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace infernix::ops::detail
