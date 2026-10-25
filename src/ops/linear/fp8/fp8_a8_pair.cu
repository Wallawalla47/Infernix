// fp8-a8-pair (LinearRoutes::Fp8A8Pair): the two-term E4M3 activation quantizer with power-of-two
// column scales, and fp8-a8-pair-block32's (LinearRoutes::Fp8A8PairBlock32) with power-of-two scales
// per 32-value block. One CTA per column writes both terms, so the residual is formed from the high
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

// The least exponent s with maximum * 2^-s <= 448 (E4M3's largest finite value) for a positive
// maximum, within [-126, 126] so that 2^s and 2^-s are normal FP32 values. Power-of-two scales keep
// x * 2^-s exact, so the high term keeps a BF16 value's top four significant bits and the residual
// its remaining four exactly.
__device__ __forceinline__ int pair_exponent(float maximum) {
    const unsigned bits = __float_as_uint(maximum);
    const int exponent  = static_cast<int>((bits >> 23) & 0xFFU) - 127;
    // maximum = 1.f * 2^exponent; 1.f * 2^8 <= 448 exactly when 1.f <= 1.75.
    const int s = exponent - 8 + ((bits & 0x7FFFFFU) > 0x600000U ? 1 : 0);
    return s < -126 ? -126 : (s > 126 ? 126 : s);
}

// 2^s for s within [-126, 127].
__device__ __forceinline__ float power_of_two(int s) {
    return __uint_as_float(static_cast<unsigned>(s + 127) << 23);
}

// A column term's scale 2^pair_exponent(maximum); 0 for a zero term.
__device__ __forceinline__ float pair_scale(float maximum) {
    return maximum == 0.0F ? 0.0F : power_of_two(pair_exponent(maximum));
}

// The maximum over the 16 lanes of this lane's half warp.
__device__ __forceinline__ float half_warp_max(float value) {
#pragma unroll
    for (int offset = 8; offset > 0; offset >>= 1)
        value = fmaxf(value, __shfl_xor_sync(0xFFFFFFFFU, value, offset));
    return value;
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

// fp8-a8-pair-block32: one CTA per column; each half warp owns one 32-value block (pairs
// 16 b .. 16 b + 15 of the column are its lanes' values), so a block's maximum is a four-step
// shuffle and its residual stays in registers.
template <int K, int Threads>
__global__ __launch_bounds__(Threads) void fp8_a8_pair_block32_quantize_kernel(
    const __nv_bfloat16* __restrict__ input, std::uint8_t* __restrict__ codes_hi,
    std::uint8_t* __restrict__ codes_lo, std::uint8_t* __restrict__ scales_hi,
    std::uint8_t* __restrict__ scales_lo, int scale_stride) {
    static_assert(Threads % 32 == 0 && Threads <= 1024);
    static_assert(K % (2 * Threads) == 0 && K % kFp8A8PairBlock32K == 0);
    constexpr int pairs_per_thread = K / 2 / Threads;

    pdl::enter();
    const int token = static_cast<int>(blockIdx.x);
    const int tid   = static_cast<int>(threadIdx.x);
    const std::int64_t base = static_cast<std::int64_t>(token) * K;
    const auto* input_pairs = reinterpret_cast<const std::uint32_t*>(input + base);
    auto* hi_pairs          = reinterpret_cast<std::uint16_t*>(codes_hi + base);
    auto* lo_pairs          = reinterpret_cast<std::uint16_t*>(codes_lo + base);

#pragma unroll
    for (int item = 0; item < pairs_per_thread; ++item) {
        const int pair     = tid + item * Threads;
        const float2 value = bf16x2_bits_to_float2(input_pairs[pair]);
        const float block_maximum = half_warp_max(fmaxf(fabsf(value.x), fabsf(value.y)));
        const int s_hi            = block_maximum == 0.0F ? 0 : pair_exponent(block_maximum);
        const float inverse_hi    = power_of_two(-s_hi);
        // In units of 2^s_hi: x * 2^-s_hi is exact and at most 448, its E4M3 rounding the high
        // term, and the remainder exact in FP32.
        const float2 scaled = make_float2(value.x * inverse_hi, value.y * inverse_hi);
        const __nv_fp8x2_storage_t hi = __nv_cvt_float2_to_fp8x2(scaled, __NV_SATFINITE, __NV_E4M3);
        const unsigned hi_bits        = static_cast<unsigned>(hi);
        const float2 residual = make_float2(scaled.x - fp8_e4m3_value(hi_bits & 0xFFU),
                                            scaled.y - fp8_e4m3_value(hi_bits >> 8));
        const float residual_maximum = half_warp_max(fmaxf(fabsf(residual.x), fabsf(residual.y)));
        // The low term's exponent relative to s_hi, kept so that s_hi + s_rel >= -127 (UE8M0's
        // least scale): below that, the residual's codes shrink toward zero instead of the stored
        // scale overstating them. It binds only for blocks whose maximum is below about 2^-100.
        int s_rel = residual_maximum == 0.0F ? 0 : pair_exponent(residual_maximum);
        s_rel     = s_rel < -127 - s_hi ? -127 - s_hi : s_rel;
        const float inverse_lo        = power_of_two(-s_rel);
        const __nv_fp8x2_storage_t lo = __nv_cvt_float2_to_fp8x2(
            make_float2(residual.x * inverse_lo, residual.y * inverse_lo), __NV_SATFINITE,
            __NV_E4M3);
        hi_pairs[pair] = static_cast<std::uint16_t>(hi);
        lo_pairs[pair] = static_cast<std::uint16_t>(lo);
        if ((tid & 15) == 0) {
            const int block = pair / 16;
            const std::int64_t at =
                (static_cast<std::int64_t>(block / 4) * scale_stride + token) * 4 + block % 4;
            scales_hi[at] = static_cast<std::uint8_t>(s_hi + 127);
            scales_lo[at] = static_cast<std::uint8_t>(s_hi + s_rel + 127);
        }
    }
}

template <int K, int Threads>
void launch_pair_block32_quantize(const Tensor& x, Fp8A8PairBlock32Workspace workspace,
                                  cudaStream_t stream) {
    CUDA_CHECK(pdl::launch_consumer(
        {dim3(x.ne[1]), dim3(Threads), 0, stream}, fp8_a8_pair_block32_quantize_kernel<K, Threads>,
        static_cast<const __nv_bfloat16*>(x.data), workspace.codes_hi, workspace.codes_lo,
        reinterpret_cast<std::uint8_t*>(workspace.block_scales_hi),
        reinterpret_cast<std::uint8_t*>(workspace.block_scales_lo),
        static_cast<int>(fp8_a8_pair_block32_scale_stride(x.ne[1]))));
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

void launch_fp8_a8_pair_block32_quantize(const Tensor& x, const Weight& weight,
                                         Fp8A8PairBlock32Workspace workspace, cudaStream_t stream) {
    if (workspace.codes_hi == nullptr || workspace.codes_lo == nullptr ||
        workspace.block_scales_hi == nullptr || workspace.block_scales_lo == nullptr) {
        throw std::invalid_argument("fp8 A8 pair block32 requires caller workspace");
    }
    if (x.ne[0] != weight.k || x.ne[1] <= 0) {
        throw std::invalid_argument("fp8 A8 pair block32 quantize: activation does not match K");
    }
    switch (weight.k) {
    case 2560: return launch_pair_block32_quantize<2560, 256>(x, workspace, stream);
    case 6144: return launch_pair_block32_quantize<6144, 256>(x, workspace, stream);
    case 640: return launch_pair_block32_quantize<640, 160>(x, workspace, stream);
    default: throw std::invalid_argument("fp8 A8 pair block32 quantize: unsupported K");
    }
}

} // namespace infernix::ops::detail
