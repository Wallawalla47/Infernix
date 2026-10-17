// fp8-a8-pair (LinearRoutes::Fp8A8Pair): the two-term E4M3 activation quantizer with power-of-two
// column scales. One CTA per column writes both terms, so the residual is formed from the high
// term's exact decoded value in registers without a second pass over the activation.
#include "core/weight.h"
#include "ops/linear/fp8/fp8_a8_plan.h"

#include "core/device.h"
#include "core/pdl.cuh"
#include "ops/common/math.cuh"
#include "ops/common/warp.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstdint>
#include <stdexcept>

namespace infernix::ops::detail {
namespace {

// The exact value of a finite E4M3 code (the saturating encoder never produces NaN).
__device__ __forceinline__ float fp8_e4m3_value(unsigned code) {
    const unsigned exponent = (code >> 3) & 0xFU;
    const unsigned mantissa = code & 0x7U;
    const float magnitude =
        exponent == 0U ? static_cast<float>(mantissa) * 0.001953125F // 2^-9: subnormal
                       : __uint_as_float(((exponent + 120U) << 23) | (mantissa << 20));
    return (code & 0x80U) != 0U ? -magnitude : magnitude;
}

// A term's scale: the least power of two 2^s with maximum * 2^-s <= 448 (E4M3's largest finite
// value), s within [-126, 126] so that 2^s and 2^-s are normal FP32 values; 0 for a zero term.
// Power-of-two scales keep x * 2^-s exact, so the high term keeps a BF16 value's top four
// significant bits and the residual its remaining four exactly.
__device__ __forceinline__ float pair_scale(float maximum) {
    if (maximum == 0.0F) return 0.0F;
    const unsigned bits = __float_as_uint(maximum);
    const int exponent  = static_cast<int>((bits >> 23) & 0xFFU) - 127;
    // maximum = 1.f * 2^exponent; 1.f * 2^8 <= 448 exactly when 1.f <= 1.75.
    int s = exponent - 8 + ((bits & 0x7FFFFFU) > 0x600000U ? 1 : 0);
    s     = s < -126 ? -126 : (s > 126 ? 126 : s);
    return __uint_as_float(static_cast<unsigned>(s + 127) << 23);
}

// The maximum of every thread's value; each call needs its own `result` slot.
template <int Threads>
__device__ __forceinline__ float block_max(float value, float* warp_maxima, float* result) {
    constexpr int warps = Threads / 32;
    const int lane      = static_cast<int>(threadIdx.x) & 31;
    const int warp      = static_cast<int>(threadIdx.x) >> 5;
    value               = warp_max(value);
    if (lane == 0) { warp_maxima[warp] = value; }
    __syncthreads();
    if (warp == 0) {
        value = lane < warps ? warp_maxima[lane] : 0.0F;
        value = warp_max(value);
        if (lane == 0) { *result = value; }
    }
    __syncthreads();
    return *result;
}

template <int K, int Threads>
__global__ __launch_bounds__(Threads) void fp8_a8_pair_quantize_kernel(
    const __nv_bfloat16* __restrict__ input, std::uint8_t* __restrict__ codes_hi,
    float* __restrict__ scales_hi, std::uint8_t* __restrict__ codes_lo,
    float* __restrict__ scales_lo) {
    static_assert(Threads % 32 == 0 && Threads <= 1024);
    static_assert(K % (2 * Threads) == 0);
    constexpr int pairs_per_thread = K / 2 / Threads;
    __shared__ float warp_maxima[Threads / 32];
    __shared__ float maxima[2];

    pdl::enter();
    const int token = static_cast<int>(blockIdx.x);
    const int tid   = static_cast<int>(threadIdx.x);
    const std::int64_t base = static_cast<std::int64_t>(token) * K;
    const auto* input_pairs = reinterpret_cast<const std::uint32_t*>(input + base);
    auto* hi_pairs          = reinterpret_cast<std::uint16_t*>(codes_hi + base);
    auto* lo_pairs          = reinterpret_cast<std::uint16_t*>(codes_lo + base);

    float2 values[pairs_per_thread];
    float maximum = 0.0F;
#pragma unroll
    for (int item = 0; item < pairs_per_thread; ++item) {
        values[item] = bf16x2_bits_to_float2(input_pairs[tid + item * Threads]);
        maximum      = fmaxf(maximum, fmaxf(fabsf(values[item].x), fabsf(values[item].y)));
    }
    maximum                = block_max<Threads>(maximum, warp_maxima, &maxima[0]);
    const float scale_hi   = pair_scale(maximum);
    const float inverse_hi = scale_hi > 0.0F ? 1.0F / scale_hi : 0.0F; // exact: a power of two

    // The high term, and the residual its decoded value leaves (in place of the activation).
    float residual_maximum = 0.0F;
#pragma unroll
    for (int item = 0; item < pairs_per_thread; ++item) {
        const int pair = tid + item * Threads;
        const __nv_fp8x2_storage_t code = __nv_cvt_float2_to_fp8x2(
            make_float2(values[item].x * inverse_hi, values[item].y * inverse_hi), __NV_SATFINITE,
            __NV_E4M3);
        hi_pairs[pair]       = static_cast<std::uint16_t>(code);
        const unsigned bits  = static_cast<unsigned>(code);
        values[item].x      -= scale_hi * fp8_e4m3_value(bits & 0xFFU);
        values[item].y      -= scale_hi * fp8_e4m3_value(bits >> 8);
        residual_maximum =
            fmaxf(residual_maximum, fmaxf(fabsf(values[item].x), fabsf(values[item].y)));
    }
    residual_maximum       = block_max<Threads>(residual_maximum, warp_maxima, &maxima[1]);
    const float scale_lo   = pair_scale(residual_maximum);
    const float inverse_lo = scale_lo > 0.0F ? 1.0F / scale_lo : 0.0F;
#pragma unroll
    for (int item = 0; item < pairs_per_thread; ++item) {
        lo_pairs[tid + item * Threads] = static_cast<std::uint16_t>(__nv_cvt_float2_to_fp8x2(
            make_float2(values[item].x * inverse_lo, values[item].y * inverse_lo), __NV_SATFINITE,
            __NV_E4M3));
    }
    if (tid == 0) {
        scales_hi[token] = scale_hi;
        scales_lo[token] = scale_lo;
    }
}

template <int K, int Threads>
void launch_pair_quantize(const Tensor& x, Fp8A8PairWorkspace workspace, cudaStream_t stream) {
    CUDA_CHECK(pdl::launch_consumer({dim3(x.ne[1]), dim3(Threads), 0, stream},
                                    fp8_a8_pair_quantize_kernel<K, Threads>,
                                    static_cast<const __nv_bfloat16*>(x.data), workspace.codes_hi,
                                    workspace.scales_hi, workspace.codes_lo, workspace.scales_lo));
}

} // namespace

void launch_fp8_a8_pair_quantize(const Tensor& x, const Weight& weight,
                                 Fp8A8PairWorkspace workspace, cudaStream_t stream) {
    if (workspace.codes_hi == nullptr || workspace.scales_hi == nullptr ||
        workspace.codes_lo == nullptr || workspace.scales_lo == nullptr) {
        throw std::invalid_argument("fp8 A8 pair requires caller workspace");
    }
    if (x.ne[0] != weight.k || x.ne[1] <= 0) {
        throw std::invalid_argument("fp8 A8 pair quantize: activation does not match K");
    }
    switch (weight.k) {
    case 2560: return launch_pair_quantize<2560, 256>(x, workspace, stream);
    case 6144: return launch_pair_quantize<6144, 256>(x, workspace, stream);
    case 640: return launch_pair_quantize<640, 160>(x, workspace, stream);
    default: throw std::invalid_argument("fp8 A8 pair quantize: unsupported K");
    }
}

} // namespace infernix::ops::detail
