#include "infernix/ops/lossless_quantize.h"

#include "core/device.h"
#include "ops/common/canonical_math.h"
#include "ops/linear/nvfp4/nvfp4_codec.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

// The design reference's activation quantizers (docs/maintainer/lossless-route.md §3b): the reproducible
// route's own rules, called on the exact route's BF16 activations, writing each quantized value without its
// activation scale (exact in BF16).

namespace infernix::ops {
namespace {

void require(bool ok, const char* what) {
    if (!ok) { throw std::invalid_argument(std::string("lossless_quantize: ") + what); }
}

void check_pair(const Tensor& x, const Tensor& out, int k_multiple) {
    require(x.dtype == DType::BF16 && x.is_contiguous() && x.ne[2] == 1 && x.ne[3] == 1 && x.ne[1] > 0 &&
                x.ne[0] > 0 && x.ne[0] % k_multiple == 0,
            "x must be contiguous BF16 [K, T] with K a multiple of the block");
    require(out.dtype == DType::BF16 && out.is_contiguous() && out.ne[0] == x.ne[0] && out.ne[2] == 1 && out.ne[3] == 1,
            "out must be contiguous BF16 [K, *]");
    require(out.data != x.data, "x and out may not overlap");
}

// One thread per 16-element block: quantize_nvfp4_k16 (the divisor form), then each code times the
// decoded scale (at most six significant bits: exact in BF16).
__global__ void nvfp4_divisor_kernel(const __nv_bfloat16* __restrict__ x, int blocks, long long total, float divisor,
                                     __nv_bfloat16* __restrict__ out) {
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) { return; }
    const __nv_bfloat16* source = x + i * 16;
    const detail::Nvfp4QuantizedK16 q = detail::quantize_nvfp4_k16<false>(source, divisor);
    __nv_bfloat16* dst = out + i * 16;
    if (q.scale == 0) {
#pragma unroll
        for (int j = 0; j < 16; ++j) { dst[j] = __float2bfloat16_rn(0.0F); }
        return;
    }
    const float d = detail::decode_nvfp4_e4m3(q.scale);
#pragma unroll
    for (int b = 0; b < 8; ++b) {
        const std::uint32_t word = b < 4 ? q.codes_lo : q.codes_hi;
        const float2 v = detail::decode_nvfp4_e2m1x2(static_cast<std::uint8_t>(word >> (8 * (b & 3))));
        dst[2 * b]     = __float2bfloat16_rn(v.x * d);
        dst[2 * b + 1] = __float2bfloat16_rn(v.y * d);
    }
    (void)blocks;
}

// One thread per 16-element block of an output column: quantize_a4_codes with the column's multiplier,
// each code's value times the scale word's (exact in BF16).
__global__ void nvfp4_scaled_kernel(const std::uint16_t* __restrict__ x, const std::int32_t* __restrict__ in_columns,
                                    const float* __restrict__ column_scales, int k, int columns,
                                    std::uint16_t* __restrict__ out) {
    const int blocks = k / 16;
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<long long>(blocks) * columns) { return; }
    const int column = static_cast<int>(i / blocks), block = static_cast<int>(i % blocks);
    const int source = in_columns != nullptr ? in_columns[column] : column;
    std::uint16_t* dst = out + static_cast<std::size_t>(column) * k + 16 * block;
    if (source < 0) {
#pragma unroll
        for (int j = 0; j < 16; ++j) { dst[j] = 0; }
        return;
    }
    const canon::A4Codes q =
        canon::quantize_a4_codes(x + static_cast<std::size_t>(source) * k + 16 * block, column_scales[column]);
    const float d = canon::e4m3_value(q.scale_word);
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        // e2m1_x2 is the doubled value with its sign; its half times d is exact.
        const float v = canon::mul_rn(static_cast<float>(canon::e2m1_x2(q.code[j])) * 0.5F, d);
        dst[j]        = canon::f32_to_bf16_rn(v);
    }
}

// One block per token: fp8_a8_quantize_kernel's rule over the column's whole K.
__global__ void fp8_token_kernel(const __nv_bfloat16* __restrict__ x, int k, __nv_bfloat16* __restrict__ out,
                                 float* __restrict__ scales) {
    constexpr int kThreads = 256;
    __shared__ float warp_maxima[kThreads / 32];
    __shared__ float token_scale;
    const int token = static_cast<int>(blockIdx.x), tid = static_cast<int>(threadIdx.x);
    const __nv_bfloat16* column = x + static_cast<std::size_t>(token) * k;
    float maximum = 0.0F;
    for (int i = tid; i < k; i += kThreads) { maximum = fmaxf(maximum, fabsf(__bfloat162float(column[i]))); }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        maximum = fmaxf(maximum, __shfl_xor_sync(0xFFFFFFFFU, maximum, offset));
    }
    if ((tid & 31) == 0) { warp_maxima[tid >> 5] = maximum; }
    __syncthreads();
    if (tid < 32) {
        maximum = tid < kThreads / 32 ? warp_maxima[tid] : 0.0F;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            maximum = fmaxf(maximum, __shfl_xor_sync(0xFFFFFFFFU, maximum, offset));
        }
        if (tid == 0) { token_scale = maximum > 0.0F ? __fdiv_rn(maximum, 448.0F) : 0.0F; }
    }
    __syncthreads();
    const float scale   = token_scale;
    const float inverse = scale > 0.0F ? __fdiv_rn(1.0F, scale) : 0.0F;
    __nv_bfloat16* dst  = out + static_cast<std::size_t>(token) * k;
    for (int i = tid; i < k; i += kThreads) {
        const float scaled              = __fmul_rn(__bfloat162float(column[i]), inverse);
        const __nv_fp8_storage_t code   = __nv_cvt_float_to_fp8(scaled, __NV_SATFINITE, __NV_E4M3);
        const __half_raw value          = __nv_cvt_fp8_to_halfraw(code, __NV_E4M3);
        dst[i]                          = __float2bfloat16_rn(__half2float(__half(value)));
    }
    if (tid == 0) { scales[token] = scale; }
}

// One thread per element: fp8_a8_static_quantize_kernel's rule, c = e4m3_rn_satfinite(fl32(v r)) with
// r = fl32(1 / s); the first T threads also write every column's scale s.
__global__ void fp8_static_kernel(const __nv_bfloat16* __restrict__ x, long long n, int columns, float scale,
                                  float inverse, __nv_bfloat16* __restrict__ out, float* __restrict__ scales) {
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < columns) { scales[i] = scale; }
    if (i >= n) { return; }
    const float scaled            = __fmul_rn(__bfloat162float(x[i]), inverse);
    const __nv_fp8_storage_t code = __nv_cvt_float_to_fp8(scaled, __NV_SATFINITE, __NV_E4M3);
    const __half_raw value        = __nv_cvt_fp8_to_halfraw(code, __NV_E4M3);
    out[i]                        = __float2bfloat16_rn(__half2float(__half(value)));
}

// BF16 -> FP16 (round to nearest even) -> BF16, exactly.
__global__ void round_fp16_kernel(__nv_bfloat16* __restrict__ x, long long n) {
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) { return; }
    x[i] = __float2bfloat16_rn(__half2float(__float2half_rn(__bfloat162float(x[i]))));
}

} // namespace

void lossless_round_fp16(Tensor& x, cudaStream_t stream) {
    require(x.dtype == DType::BF16 && x.is_contiguous(), "x must be contiguous BF16");
    const long long n = static_cast<long long>(x.numel());
    if (n == 0) { return; }
    constexpr int kThreads = 256;
    round_fp16_kernel<<<static_cast<unsigned>((n + kThreads - 1) / kThreads), kThreads, 0, stream>>>(
        static_cast<__nv_bfloat16*>(x.data), n);
    CUDA_CHECK(cudaGetLastError());
}

void lossless_quantize_nvfp4(const Tensor& x, float divisor, Tensor& out, cudaStream_t stream) {
    check_pair(x, out, 16);
    require(out.ne[1] == x.ne[1], "out has x's columns");
    require(std::isfinite(divisor) && divisor > 0.0F, "the activation divisor is positive and finite");
    const long long total = static_cast<long long>(x.ne[0] / 16) * x.ne[1];
    constexpr int kThreads = 256;
    nvfp4_divisor_kernel<<<static_cast<unsigned>((total + kThreads - 1) / kThreads), kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<int>(x.ne[0] / 16), total, divisor,
        static_cast<__nv_bfloat16*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

void lossless_quantize_nvfp4_scaled(const Tensor& x, const std::int32_t* in_columns, const float* column_scales,
                                    Tensor& out, cudaStream_t stream) {
    check_pair(x, out, 16);
    require(column_scales != nullptr, "each column needs its input scale");
    require(in_columns != nullptr || out.ne[1] <= x.ne[1], "without a column map out's columns are x's");
    const int k = static_cast<int>(x.ne[0]), columns = static_cast<int>(out.ne[1]);
    const long long total = static_cast<long long>(k / 16) * columns;
    constexpr int kThreads = 256;
    nvfp4_scaled_kernel<<<static_cast<unsigned>((total + kThreads - 1) / kThreads), kThreads, 0, stream>>>(
        static_cast<const std::uint16_t*>(x.data), in_columns, column_scales, k, columns,
        static_cast<std::uint16_t*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

void lossless_quantize_fp8_token(const Tensor& x, Tensor& out, Tensor& scales, cudaStream_t stream) {
    check_pair(x, out, 1);
    require(out.ne[1] == x.ne[1], "out has x's columns");
    require(scales.dtype == DType::FP32 && scales.is_contiguous() && scales.numel() == x.ne[1],
            "scales must be FP32 [T]");
    fp8_token_kernel<<<static_cast<unsigned>(x.ne[1]), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<int>(x.ne[0]), static_cast<__nv_bfloat16*>(out.data),
        static_cast<float*>(scales.data));
    CUDA_CHECK(cudaGetLastError());
}

void lossless_quantize_fp8_static(const Tensor& x, float input_scale, Tensor& out, Tensor& scales, cudaStream_t stream) {
    check_pair(x, out, 1);
    require(out.ne[1] == x.ne[1], "out has x's columns");
    require(scales.dtype == DType::FP32 && scales.is_contiguous() && scales.numel() == x.ne[1],
            "scales must be FP32 [T]");
    require(std::isfinite(input_scale) && input_scale > 0.0F, "the static input scale is positive and finite");
    const long long n    = static_cast<long long>(x.numel());
    const long long work = std::max<long long>(n, x.ne[1]);
    const float inverse  = 1.0F / input_scale; // fl32(1 / s), as the reproducible route's encoder takes it
    constexpr int kThreads = 256;
    fp8_static_kernel<<<static_cast<unsigned>((work + kThreads - 1) / kThreads), kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), n, static_cast<int>(x.ne[1]), input_scale, inverse,
        static_cast<__nv_bfloat16*>(out.data), static_cast<float*>(scales.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace infernix::ops
