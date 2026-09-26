#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.h"

#include "core/device.h"
#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_a4_tma_launch.h"
#include "ops/kernel/rmsnorm.cuh"
#include "ops/linear/nvfp4/nvfp4_a4_quantize.cuh"
#include "ops/linear/nvfp4/nvfp4_geometry.h"
#include "ops/linear/nvfp4/nvfp4_operands.h"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

using Geometry                = Nvfp4N14336K5120;
constexpr int kBlock          = 256;
constexpr int kPairsPerRow    = Geometry::kInputRows / 2;
constexpr int kPairsPerThread = kPairsPerRow / kBlock;
static_assert(kPairsPerThread == 10);

// The sum and Offset epilogue deliberately use the D=5120 RMSNorm launcher's pair ownership,
// accumulation order, reduction, and BF16 store rounding. The shared row is the exact BF16 image
// which the standalone quantizer would read from the intermediate global allocation. The scales go
// to the tiled plane the TMA route reads; a token past the real count is a tile's padding and gets
// zero scales, as the standalone quantizer writes them.
template <Nvfp4ScaleLayout Layout>
__launch_bounds__(kBlock) __global__
    void rmsnorm_nvfp4_quantize_kernel(const __nv_bfloat162* __restrict__ residual,
                                       const __nv_bfloat162* __restrict__ norm_weight,
                                       std::uint8_t* __restrict__ codes,
                                       std::uint8_t* __restrict__ scales, std::int32_t tokens,
                                       float eps, float input_scale_divisor) {
    static_assert(Layout != Nvfp4ScaleLayout::RowMajor);
    const int token             = static_cast<int>(blockIdx.x);
    const int thread            = static_cast<int>(threadIdx.x);
    constexpr int kGroupsPerRow = Geometry::kGroupsPerRow;
    if (token >= tokens) {
        for (int group = thread; group < kGroupsPerRow; group += kBlock) {
            scales[nvfp4_tiled_scale_offset<Geometry, Layout>(token, group)] = 0;
        }
        return;
    }

    const std::int64_t row_base = static_cast<std::int64_t>(token) * kPairsPerRow;
    __nv_bfloat162 values[kPairsPerThread];
    __nv_bfloat162 weights[kPairsPerThread];
    float sum = 0.0F;
#pragma unroll
    for (int i = 0; i < kPairsPerThread; ++i) {
        const int pair  = thread + i * kBlock;
        values[i]       = residual[row_base + pair];
        weights[i]      = norm_weight[pair];
        const float2 xf = __bfloat1622float2(values[i]);
        sum += xf.x * xf.x + xf.y * xf.y;
    }

    __shared__ float warp_sums[kBlock / kWarpSize];
    __shared__ float inv_shared;
    const float block_sum = block_reduce_sum<kBlock>(sum, warp_sums);
    if (thread == 0) {
        inv_shared = rsqrtf(block_sum / static_cast<float>(Geometry::kInputRows) + eps);
    }
    __syncthreads();
    const float inv = inv_shared;

    __shared__ alignas(16) __nv_bfloat162 staged[kPairsPerRow];
#pragma unroll
    for (int i = 0; i < kPairsPerThread; ++i) {
        const int pair  = thread + i * kBlock;
        const float2 xf = __bfloat1622float2(values[i]);
        const float2 wf = __bfloat1622float2(weights[i]);
        staged[pair] =
            __floats2bfloat162_rn(rmsnorm_epilogue<RmsEpilogue::Offset>(xf.x, inv, wf.x, 0.0F),
                                  rmsnorm_epilogue<RmsEpilogue::Offset>(xf.y, inv, wf.y, 0.0F));
    }
    __syncthreads();

    const auto* staged_bf16 = reinterpret_cast<const __nv_bfloat16*>(staged);
    for (int group = thread; group < kGroupsPerRow; group += kBlock) {
        const Nvfp4QuantizedK16 quantized =
            quantize_nvfp4_k16(staged_bf16 + group * 16, input_scale_divisor);
        auto* code_destination =
            codes + static_cast<std::int64_t>(token) * Geometry::kCodeBytesPerRow + group * 8;
        store_vec(code_destination, make_uint2(quantized.codes_lo, quantized.codes_hi));
        scales[nvfp4_tiled_scale_offset<Geometry, Layout>(token, group)] = quantized.scale;
    }
}

template <Nvfp4ScaleLayout Layout>
void launch_quantize(const Tensor& residual, const Tensor& norm_weight, float eps,
                     float input_scale_divisor, Nvfp4A4Workspace workspace, cudaStream_t stream) {
    const std::int32_t tokens = residual.ne[1];
    const int padded          = nvfp4_a4_padded_tokens(tokens, Layout);
    if (static_cast<std::size_t>(padded) * Geometry::kGroupsPerRow > workspace.scale_bytes) {
        throw std::logic_error("fused rmsnorm NVFP4 quantize: scale plane is too small");
    }
    rmsnorm_nvfp4_quantize_kernel<Layout><<<padded, kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat162*>(residual.data),
        static_cast<const __nv_bfloat162*>(norm_weight.data), workspace.codes, workspace.scales,
        tokens, eps, input_scale_divisor);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void launch_nvfp4_attn_input_fused_rmsnorm_quantize(const Tensor& residual,
                                                    const Tensor& norm_weight, float eps,
                                                    float input_scale_divisor,
                                                    Nvfp4A4Workspace workspace,
                                                    Nvfp4ScaleLayout layout, cudaStream_t stream) {
    switch (layout) {
    case Nvfp4ScaleLayout::Tiled128:
        launch_quantize<Nvfp4ScaleLayout::Tiled128>(residual, norm_weight, eps,
                                                    input_scale_divisor, workspace, stream);
        return;
    case Nvfp4ScaleLayout::Tiled256:
        launch_quantize<Nvfp4ScaleLayout::Tiled256>(residual, norm_weight, eps,
                                                    input_scale_divisor, workspace, stream);
        return;
    case Nvfp4ScaleLayout::RowMajor:
        break;
    }
    throw std::invalid_argument("fused rmsnorm NVFP4 quantize: the TMA route reads a tiled plane");
}

void nvfp4_attn_input_fused_rmsnorm_launch(const Tensor& residual, const Tensor& norm_weight,
                                           float eps, const Weight& weight, Tensor& q, Tensor& gate,
                                           Tensor& k, Tensor& v, WorkspaceArena& workspace,
                                           cudaStream_t stream) {
    const std::int32_t tokens      = residual.ne[1];
    const Nvfp4ScaleLayout layout  = nvfp4_attn_input_scale_layout(tokens);
    auto scope                     = workspace.scope();
    const Nvfp4A4Workspace scratch = allocate_nvfp4_a4_workspace(workspace, tokens, weight.k);
    launch_nvfp4_attn_input_fused_rmsnorm_quantize(residual, norm_weight, eps,
                                                   weight.input_scale_divisor, scratch, layout,
                                                   stream);
    launch_nvfp4_a4_tma_attention(nvfp4_a4_operands(weight, scratch, tokens, layout),
                                  static_cast<__nv_bfloat16*>(q.data),
                                  static_cast<__nv_bfloat16*>(gate.data),
                                  static_cast<__nv_bfloat16*>(k.data),
                                  static_cast<__nv_bfloat16*>(v.data), stream);
}

} // namespace ninfer::ops::detail
