#include "core/weight.h"
#include "ops/linear/fp8/fp8_a8_plan.h"

#include "core/device.h"
#include "core/pdl.cuh"
#include "ops/common/math.cuh"
#include "ops/common/warp.cuh"
#include "ops/linear/fp8/fp8_instances.cuh"
#include "ops/linear/fp8/fp8_schedule.cuh"
#include "ops/linear/common/epilogue.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cstdint>
#include <stdexcept>

namespace infernix::ops::detail {
namespace {

template <class ActivationGeometry, int Threads = 256>
__global__ __launch_bounds__(Threads,
                             2) void fp8_a8_quantize_kernel(const __nv_bfloat16* __restrict__ input,
                                                            std::uint8_t* __restrict__ codes,
                                                            float* __restrict__ scales,
                                                            std::uint8_t* __restrict__ residual) {
    static_assert((ActivationGeometry::kInputRows % (Threads * 2)) == 0);
    constexpr int pairs_per_token  = ActivationGeometry::kInputRows / 2;
    constexpr int pairs_per_thread = pairs_per_token / Threads;
    constexpr int warps            = Threads / 32;
    __shared__ float warp_maxima[warps];
    __shared__ float token_scale;

    pdl::enter();
    const int token         = static_cast<int>(blockIdx.x);
    const int tid           = static_cast<int>(threadIdx.x);
    const int lane          = tid & 31;
    const int warp          = tid >> 5;
    const auto* input_pairs = reinterpret_cast<const std::uint32_t*>(
        input + static_cast<std::int64_t>(token) * ActivationGeometry::kInputRows);
    auto* output_pairs = reinterpret_cast<std::uint16_t*>(
        codes + static_cast<std::int64_t>(token) * ActivationGeometry::kInputRows);

    float2 values[pairs_per_thread];
    float maximum = 0.0F;
#pragma unroll
    for (int item = 0; item < pairs_per_thread; ++item) {
        const int pair = tid + item * Threads;
        values[item]   = bf16x2_bits_to_float2(input_pairs[pair]);
        maximum        = fmaxf(maximum, fabsf(values[item].x));
        maximum        = fmaxf(maximum, fabsf(values[item].y));
    }
    maximum = warp_max(maximum);
    if (lane == 0) { warp_maxima[warp] = maximum; }
    __syncthreads();
    if (warp == 0) {
        maximum = lane < warps ? warp_maxima[lane] : 0.0F;
        maximum = warp_max(maximum);
        if (lane == 0) { token_scale = maximum > 0.0F ? maximum / 448.0F : 0.0F; }
    }
    __syncthreads();

    const float scale   = token_scale;
    const float inverse = scale > 0.0F ? 1.0F / scale : 0.0F;
    auto* residual_pairs =
        residual == nullptr
            ? nullptr
            : reinterpret_cast<std::uint16_t*>(
                  residual + static_cast<std::int64_t>(token) * ActivationGeometry::kInputRows);
#pragma unroll
    for (int item = 0; item < pairs_per_thread; ++item) {
        const int pair      = tid + item * Threads;
        const float2 scaled = make_float2(values[item].x * inverse, values[item].y * inverse);
        const __nv_fp8x2_storage_t code =
            __nv_cvt_float2_to_fp8x2(scaled, __NV_SATFINITE, __NV_E4M3);
        output_pairs[pair] = code;
        if (residual_pairs != nullptr) {
            // The A8 residual profile: what the code left, in units of scale / 16. E4M3 rounds to
            // within 2^-4 of a value, so the remainder of a value within the row's 448 fits.
            const float2 decoded = __half22float2(__half2(__nv_cvt_fp8x2_to_halfraw2(code, __NV_E4M3)));
            const float2 left    = make_float2(__fmul_rn(__fsub_rn(scaled.x, decoded.x), 16.0F),
                                               __fmul_rn(__fsub_rn(scaled.y, decoded.y), 16.0F));
            residual_pairs[pair] = __nv_cvt_float2_to_fp8x2(left, __NV_SATFINITE, __NV_E4M3);
        }
    }
    if (tid == 0) { scales[token] = scale; }
}

template <class ActivationGeometry>
void launch_quantize_exact(const Tensor& x, Fp8A8Workspace workspace, cudaStream_t stream) {
    constexpr int kThreads = 256;
    CUDA_CHECK(pdl::launch_consumer({dim3(x.ne[1]), dim3(kThreads), 0, stream},
                                    fp8_a8_quantize_kernel<ActivationGeometry, kThreads>,
                                    static_cast<const __nv_bfloat16*>(x.data), workspace.codes,
                                    workspace.scales, workspace.residual));
}

} // namespace

void launch_fp8_a8_quantize(const Tensor& x, const Weight& weight, Fp8A8Workspace workspace,
                            cudaStream_t stream) {
    if (workspace.codes == nullptr || workspace.scales == nullptr) {
        throw std::invalid_argument("fp8 A8 requires caller workspace");
    }
    switch (weight.k) {
    case Fp8Activation5120Geometry::kInputRows:
        launch_quantize_exact<Fp8Activation5120Geometry>(x, workspace, stream);
        return;
    case Fp8Activation6144Geometry::kInputRows:
        launch_quantize_exact<Fp8Activation6144Geometry>(x, workspace, stream);
        return;
    case Fp8Activation17408Geometry::kInputRows:
        launch_quantize_exact<Fp8Activation17408Geometry>(x, workspace, stream);
        return;
    default:
        throw std::invalid_argument("fp8 A8 quantize: unsupported K");
    }
}

} // namespace infernix::ops::detail
