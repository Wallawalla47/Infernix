#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.h"

#include "core/device.h"
#include "core/layout.h"
#include "core/pdl.cuh"
#include "ops/kernel/rmsnorm.cuh"
#include "ops/linear/nvfp4/nvfp4_a4_quantize.cuh"
#include "ops/linear/nvfp4/nvfp4_geometry.h"
#include "ops/linear_add/nvfp4/nvfp4_linear_add_plan.h"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace infernix::ops::detail {
namespace {

using InputGeometry           = Nvfp4N34816K5120; // the gate/up projection's K = 5120
using DownGeometry            = Nvfp4N5120K17408; // the down projection's K = 17408
constexpr int kBlock          = 256;
constexpr int kPairsPerRow    = InputGeometry::kInputRows / 2;
constexpr int kPairsPerThread = kPairsPerRow / kBlock;
constexpr int kTerms          = kNvfp4A4ResidualTerms;
static_assert(kPairsPerThread == 10);

// The D=5120 Offset RMSNorm's arithmetic (rmsnorm_cta_bf16x2_kernel<Offset, 256, 10, true, 5120>: its pair
// ownership, accumulation order, reduction and BF16 rounding), its BF16 row staged in shared memory, then each
// 16-element group encoded as nvfp4_a4_residual_quantize_kernel encodes it (row-major scales): the bytes of
// rmsnorm() followed by that quantizer, without the normalized row in global memory.
__launch_bounds__(kBlock) __global__ void rmsnorm_a4_residual_quantize_kernel(
    const __nv_bfloat162* __restrict__ residual, const __nv_bfloat162* __restrict__ norm_weight,
    std::uint8_t* __restrict__ codes, std::uint8_t* __restrict__ scales, std::int64_t code_plane,
    std::int64_t scale_plane, float eps, float input_scale_divisor) {
    pdl::enter();
    const int token             = static_cast<int>(blockIdx.x);
    const int thread            = static_cast<int>(threadIdx.x);
    constexpr int kGroupsPerRow = InputGeometry::kGroupsPerRow;

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
    if (thread == 0) { inv_shared = rsqrtf(block_sum / static_cast<float>(InputGeometry::kInputRows) + eps); }
    __syncthreads();
    const float inv = inv_shared;

    __shared__ alignas(16) __nv_bfloat162 staged[kPairsPerRow];
#pragma unroll
    for (int i = 0; i < kPairsPerThread; ++i) {
        const int pair  = thread + i * kBlock;
        const float2 xf = __bfloat1622float2(values[i]);
        const float2 wf = __bfloat1622float2(weights[i]);
        staged[pair]    = __floats2bfloat162_rn(rmsnorm_epilogue<RmsEpilogue::Offset>(xf.x, inv, wf.x, 0.0F),
                                                rmsnorm_epilogue<RmsEpilogue::Offset>(xf.y, inv, wf.y, 0.0F));
    }
    __syncthreads();

    // One lane a group: 320 groups over 256 lanes take two rounds, where four lanes a group (the SwiGLU epilogue's
    // quantize_nvfp4_terms_k16_quad) take five and measured slower here (5.52 against 5.08 us a call, 2026-11-09).
    const auto* staged_bf16 = reinterpret_cast<const __nv_bfloat16*>(staged);
    for (int group = thread; group < kGroupsPerRow; group += kBlock) {
        Nvfp4QuantizedK16 terms[kTerms];
        quantize_nvfp4_terms_k16<kTerms>(staged_bf16 + group * 16, input_scale_divisor, terms);
        const std::int64_t code_offset =
            static_cast<std::int64_t>(token) * InputGeometry::kCodeBytesPerRow + group * 8;
        const std::int64_t scale_offset = static_cast<std::int64_t>(token) * kGroupsPerRow + group;
#pragma unroll
        for (int term = 0; term < kTerms; ++term) {
            store_vec(codes + term * code_plane + code_offset, make_uint2(terms[term].codes_lo, terms[term].codes_hi));
            scales[term * scale_plane + scale_offset] = terms[term].scale;
        }
    }
}

struct FusedPlanes {
    Nvfp4A4Workspace input;   // the normalized x, K = 5120
    Nvfp4A4Workspace product; // silu(gate) * up, K = 17408
};

template <class Allocator>
FusedPlanes allocate_planes(Allocator& allocator, std::int32_t tokens) {
    return {allocate_nvfp4_a4_workspace(allocator, tokens, InputGeometry::kInputRows, kTerms),
            allocate_nvfp4_a4_workspace(allocator, tokens, DownGeometry::kInputRows, kTerms)};
}

} // namespace

void launch_nvfp4_rmsnorm_a4_residual_quantize(const Tensor& residual, const Tensor& norm_weight, float eps,
                                               float input_scale_divisor, Nvfp4A4Workspace planes,
                                               cudaStream_t stream) {
    const std::int32_t tokens = residual.ne[1];
    if (residual.ne[0] != InputGeometry::kInputRows || tokens <= 0 || planes.terms != kTerms ||
        planes.code_plane < static_cast<std::size_t>(tokens) * InputGeometry::kCodeBytesPerRow ||
        planes.scale_plane < static_cast<std::size_t>(tokens) * InputGeometry::kGroupsPerRow) {
        throw std::invalid_argument("fused rmsnorm A4 residual quantize: unsupported call or planes too small");
    }
    CUDA_CHECK(pdl::launch_consumer({dim3(static_cast<unsigned>(tokens)), dim3(kBlock), 0, stream},
                                    rmsnorm_a4_residual_quantize_kernel,
                                    static_cast<const __nv_bfloat162*>(residual.data),
                                    static_cast<const __nv_bfloat162*>(norm_weight.data), planes.codes,
                                    planes.scales, static_cast<std::int64_t>(planes.code_plane),
                                    static_cast<std::int64_t>(planes.scale_plane), eps, input_scale_divisor));
}

std::size_t nvfp4_dense_mlp_fused_rmsnorm_workspace_bytes(std::int32_t max_tokens) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_planes<WorkspaceLayoutBuilder>(layout, max_tokens);
    return layout.peak_bytes(1);
}

void nvfp4_dense_mlp_fused_rmsnorm_launch(Tensor& residual, const Tensor& norm_weight, float eps,
                                          const Weight& gate_up, const Weight& down, WorkspaceArena& workspace,
                                          cudaStream_t stream) {
    const std::int32_t tokens = residual.ne[1];
    auto scope                = workspace.scope();
    const auto planes         = allocate_planes<WorkspaceArena>(workspace, tokens);
    launch_nvfp4_rmsnorm_a4_residual_quantize(residual, norm_weight, eps, gate_up.input_scale_divisor, planes.input,
                                              stream);
    nvfp4_linear_swiglu_a4_residual_planes_launch(gate_up, planes.input, planes.product, down.input_scale_divisor,
                                                  tokens, stream);
    nvfp4_linear_add_a4_residual_planes_launch(down, residual, planes.product, tokens, stream);
}

} // namespace infernix::ops::detail
