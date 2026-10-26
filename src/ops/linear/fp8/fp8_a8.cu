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

#include <algorithm>
#include <cmath>
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

// A static A8 parent (fp8_e4m3fn_row_f32) encodes with its calibrated input scale s:
// c = e4m3_rn_satfinite(fl32(x * r)), r = fl32(1 / s), and every token's scale is s. Each thread
// encodes eight consecutive values of the [T, K] row-major input.
__global__ __launch_bounds__(256) void fp8_a8_static_quantize_kernel(
    const __nv_bfloat16* __restrict__ input, std::uint8_t* __restrict__ codes,
    float* __restrict__ scales, float scale, float inverse, std::int64_t octets, int tokens) {
    pdl::enter();
    const std::int64_t index =
        static_cast<std::int64_t>(blockIdx.x) * blockDim.x + static_cast<std::int64_t>(threadIdx.x);
    if (index < tokens) { scales[index] = scale; }
    if (index >= octets) { return; }
    const uint4 words = __ldg(reinterpret_cast<const uint4*>(input) + index);
    const std::uint32_t pairs[4]{words.x, words.y, words.z, words.w};
    std::uint16_t out[4];
#pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
        const float2 value = bf16x2_bits_to_float2(pairs[pair]);
        out[pair]          = __nv_cvt_float2_to_fp8x2(
            make_float2(__fmul_rn(value.x, inverse), __fmul_rn(value.y, inverse)), __NV_SATFINITE,
            __NV_E4M3);
    }
    reinterpret_cast<uint2*>(codes)[index] =
        make_uint2(static_cast<std::uint32_t>(out[0]) | (static_cast<std::uint32_t>(out[1]) << 16),
                   static_cast<std::uint32_t>(out[2]) | (static_cast<std::uint32_t>(out[3]) << 16));
}

void launch_quantize_static(const Tensor& x, const Weight& weight, Fp8A8Workspace workspace,
                            cudaStream_t stream) {
    if (workspace.residual != nullptr) {
        throw std::invalid_argument("fp8 A8 quantize: a static input scale has no residual profile");
    }
    if (weight.k % 8 || reinterpret_cast<std::uintptr_t>(x.data) % 16 ||
        reinterpret_cast<std::uintptr_t>(workspace.codes) % 8) {
        throw std::invalid_argument("fp8 A8 static quantize requires K % 8 and aligned buffers");
    }
    constexpr int kThreads    = 256;
    const int tokens          = static_cast<int>(x.ne[1]);
    const std::int64_t octets = static_cast<std::int64_t>(tokens) * weight.k / 8;
    const std::int64_t items  = std::max<std::int64_t>(octets, tokens);
    const float scale         = weight.input_scale;
    const float inverse       = 1.0F / scale;
    CUDA_CHECK(pdl::launch_consumer(
        {dim3(static_cast<unsigned>((items + kThreads - 1) / kThreads)), dim3(kThreads), 0, stream},
        fp8_a8_static_quantize_kernel, static_cast<const __nv_bfloat16*>(x.data), workspace.codes,
        workspace.scales, scale, inverse, octets, tokens));
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
    if (weight.qtype == QType::FP8_E4M3FN_ROW_F32) {
        if (weight.input_scale == 0.0F) {
            // A per-token parent (ModelOpt FP8_PER_CHANNEL_PER_TOKEN, no calibrated scale): each column
            // takes its own scale, max |x| / 448, as the BF16-scaled row format's A8 route encodes. The
            // Qwen3.5-122B-A10B's input projections read K 3072, its output projections K 8192.
            switch (weight.k) {
            case 3072: launch_quantize_exact<Fp8ActivationGeometry<3072>>(x, workspace, stream); return;
            case 8192: launch_quantize_exact<Fp8ActivationGeometry<8192>>(x, workspace, stream); return;
            default: throw std::invalid_argument("fp8 A8 quantize: unsupported per-token K");
            }
        }
        if (!(weight.input_scale > 0.0F) || !std::isfinite(weight.input_scale)) {
            throw std::invalid_argument("fp8 A8 quantize: static parent requires its input scale");
        }
        launch_quantize_static(x, weight, workspace, stream);
        return;
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
