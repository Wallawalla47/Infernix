#include "core/weight.h"
#include "ops/dynamic_grouped_conv/bf16/bf16_dynamic_grouped_conv_prepare_kernels.h"
#include "core/device.h"
#include "core/pdl.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/rowsplit_mma.cuh"
#include <cuda_bf16.h>
#include <stdexcept>

// The prepare in one kernel for Qwen3.8-Flash-Next's drafter (H = 2560, at most 16 columns): the RMS
// scale factors out of the projection, d = inv * sum_h W[row, h] * (r[h] * w[h]), so every CTA projects
// the raw residual times the norm weight and accumulates each column's sum of squares on the way,
// instead of an RMSNorm kernel, a split-K projection and a reduction. A CTA owns four groups: their
// sixteen coefficient rows (two sides x two taps) over all of H, so it also writes those groups'
// prepared channels and finish deltas.
namespace infernix::ops::detail {
namespace {
constexpr int kBlockK = 64, kWarps = 8, kThreads = kWarps * 32;
// Pipeline depth per warp: three stages fit sm_120a's 99 KB of shared memory at 8 columns, two at 16.
template <int Columns>
constexpr int kStages = Columns <= 8 ? 3 : 2;
constexpr int kGroupsPerCta = 4, kCtaRows = 4 * kGroupsPerCta; // 16: one MMA row block

__device__ __forceinline__ int shared_col(int row, int col) { return gemm_swz64(row, col); }

// One warp's pipeline stage: the CTA's 16 coefficient rows, the columns' residual (converted to
// r * w in place) and the norm weight, over 64 hidden channels.
template <int Columns>
struct Stage {
    __nv_bfloat16 weight[kCtaRows * kBlockK];
    __nv_bfloat16 input[Columns * kBlockK];
    __nv_bfloat16 norm[kBlockK];
};

// The CTA's own 64 channels for the epilogue: the columns' raw residual, the norm weight and the base
// kernel's side-0 taps, so it reads nothing from global memory.
template <int Columns>
struct Own {
    __nv_bfloat16 residual[Columns * kBlockK];
    __nv_bfloat16 norm[kBlockK];
    __nv_bfloat16 base[2 * kBlockK];
};

template <int Columns>
constexpr std::size_t shared_bytes() {
    return sizeof(Stage<Columns>) * kStages<Columns> * kWarps + sizeof(Own<Columns>) +
           sizeof(float) * kWarps * kCtaRows * Columns + sizeof(float) * kWarps * Columns +
           sizeof(float) * kCtaRows * Columns + sizeof(float) * Columns;
}

// Columns: the MMA column width (8 or 16) holding the call's tokens, which are W x B <= Columns.
template <int Hidden, int Columns>
__global__ __launch_bounds__(kThreads, 1) void dynamic_grouped_conv_prepare_fused_kernel(
    const __nv_bfloat16* residual, const __nv_bfloat16* __restrict__ norm,
    const __nv_bfloat16* __restrict__ base, const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ prepared, __nv_bfloat16* __restrict__ finish, float eps, int tokens, int width) {
    constexpr int kGroups = Hidden / 16, kTiles = Hidden / kBlockK, kBlocks = Columns / 8;
    constexpr int kInputVectors = Columns * (kBlockK / 8) / 32; // per lane and tile
    static_assert(kTiles % kWarps == 0 && kInputVectors >= 1);
    extern __shared__ __align__(16) unsigned char smem[];
    auto* stages    = reinterpret_cast<Stage<Columns>*>(smem);
    auto* mine      = reinterpret_cast<Own<Columns>*>(stages + kStages<Columns> * kWarps);
    auto* partial   = reinterpret_cast<float*>(mine + 1);       // [warp][row][column]
    auto* squares   = partial + kWarps * kCtaRows * Columns;    // [warp][column]
    auto* projected = squares + kWarps * Columns;               // [row][column]
    auto* inverse   = projected + kCtaRows * Columns;            // [column]

    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32, gid = lane / 4, lid = lane % 4;
    const int group0 = blockIdx.x * kGroupsPerCta;
    Stage<Columns>* own = stages + warp * kStages<Columns>;
    // The CTA's own channels are hidden tile blockIdx.x, which warp blockIdx.x % 8 converts at its step
    // blockIdx.x / 8.
    const bool owner_warp = warp == static_cast<int>(blockIdx.x) % kWarps;
    const int owner_step  = static_cast<int>(blockIdx.x) / kWarps;

    // The weights (coefficient rows and norm weight): immutable, so staged before the dependency wait.
    auto stage_weights = [&](int slot, int tile) {
        const int k0 = tile * kBlockK;
#pragma unroll
        for (int j = 0; j < kCtaRows * (kBlockK / 8) / 32; ++j) {
            const int item = lane + 32 * j, row = item / 8, kk = item % 8 * 8;
            // Local row c * 4 + g: coefficient c (side x 2 + tap) of the CTA's group g.
            const int global_row = row / kGroupsPerCta * kGroups + group0 + row % kGroupsPerCta;
            cp_async<16, Cache::cg>(&own[slot].weight[row * kBlockK + shared_col(row, kk)],
                                     weight + static_cast<std::int64_t>(global_row) * Hidden + k0 + kk);
        }
        if (lane < kBlockK / 8) { cp_async<16, Cache::cg>(&own[slot].norm[lane * 8], norm + k0 + lane * 8); }
    };
    // Lane `lane` stages residual vectors lane + 32 j: column lane / 8 + 4 j, channels (lane % 8) * 8.
    auto stage_residual = [&](int slot, int tile) {
        const int k0 = tile * kBlockK;
#pragma unroll
        for (int j = 0; j < kInputVectors; ++j) {
            const int item = lane + 32 * j, col = item / 8, kk = item % 8 * 8;
            const int safe = col < tokens ? col : 0;
            cp_async_zfill<16, Cache::cg>(&own[slot].input[col * kBlockK + shared_col(col, kk)],
                                           residual + static_cast<std::int64_t>(safe) * Hidden + k0 + kk,
                                           col < tokens ? 16 : 0);
        }
    };

    float acc[kBlocks][4] = {};
    float square[kInputVectors] = {};
    constexpr int kWarpTiles = kTiles / kWarps;
#pragma unroll
    for (int s = 0; s < kStages<Columns>; ++s) {
        if (s < kWarpTiles) { stage_weights(s, warp + s * kWarps); }
    }
    if (tid < 2 * kBlockK / 8) { // the base kernel's tap 0 and tap 1 (side 0) of the CTA's channels
        const int tap = tid / (kBlockK / 8), kk = tid % (kBlockK / 8) * 8;
        cp_async<16, Cache::cg>(&mine->base[tap * kBlockK + kk], base + tap * Hidden + group0 * 16 + kk);
    }
    pdl::enter_streaming();
#pragma unroll
    for (int s = 0; s < kStages<Columns>; ++s) {
        if (s < kWarpTiles) { stage_residual(s, warp + s * kWarps); }
        cp_commit();
    }
#pragma unroll 1
    for (int t = 0; t < kWarpTiles; ++t) {
        const int slot = t % kStages<Columns>;
        cp_wait<kStages<Columns> - 1>();
        __syncwarp();
        const bool keep = owner_warp && t == owner_step;
        if (keep && lane < kBlockK / 8) {
            *reinterpret_cast<uint4*>(&mine->norm[lane * 8]) = *reinterpret_cast<const uint4*>(&own[slot].norm[lane * 8]);
        }
        // r -> r * w (BF16, the MMA's operand) and the columns' sums of squares of r.
#pragma unroll
        for (int j = 0; j < kInputVectors; ++j) {
            const int item = lane + 32 * j, col = item / 8, kk = item % 8 * 8;
            uint4* at       = reinterpret_cast<uint4*>(&own[slot].input[col * kBlockK + shared_col(col, kk)]);
            const uint4 raw = *at;
            if (keep) { *reinterpret_cast<uint4*>(&mine->residual[col * kBlockK + kk]) = raw; }
            const uint4 nw  = *reinterpret_cast<const uint4*>(&own[slot].norm[kk]);
            const auto* r2  = reinterpret_cast<const __nv_bfloat162*>(&raw);
            const auto* w2  = reinterpret_cast<const __nv_bfloat162*>(&nw);
            uint4 out;
            auto* x2 = reinterpret_cast<__nv_bfloat162*>(&out);
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const float2 r = __bfloat1622float2(r2[e]);
                const float2 w = __bfloat1622float2(w2[e]);
                square[j]      = fmaf(r.x, r.x, square[j]);
                square[j]      = fmaf(r.y, r.y, square[j]);
                x2[e]          = __floats2bfloat162_rn(r.x * w.x, r.y * w.y);
            }
            *at = out;
        }
        __syncwarp();
#pragma unroll
        for (int ki = 0; ki < kBlockK / 16; ++ki) {
            unsigned a[4];
            const int ar = (lane & 7) + ((lane >> 3) & 1) * 8;
            const int ak = ki * 16 + (lane >> 4) * 8;
            ldmatrix_x4(a[0], a[1], a[2], a[3], smem_addr(&own[slot].weight[ar * kBlockK + shared_col(ar, ak)]));
#pragma unroll
            for (int m = 0; m < kBlocks; ++m) {
                unsigned b[2];
                const int br = m * 8 + (lane & 7);
                const int bk = ki * 16 + ((lane >> 3) & 1) * 8;
                ldmatrix_x2(b[0], b[1], smem_addr(&own[slot].input[br * kBlockK + shared_col(br, bk)]));
                mma_bf16(acc[m][0], acc[m][1], acc[m][2], acc[m][3], a[0], a[1], a[2], a[3], b[0], b[1]);
            }
        }
        __syncwarp();
        if (t + kStages<Columns> < kWarpTiles) {
            stage_weights(slot, warp + (t + kStages<Columns>) * kWarps);
            stage_residual(slot, warp + (t + kStages<Columns>) * kWarps);
        }
        cp_commit();
    }
    cp_wait<0>();
    pdl::trigger_dependents();

    // Each column's sum of squares: its 8 lanes (one per channel slice), then the warps, in a fixed
    // order whatever the call's width and batch.
#pragma unroll
    for (int j = 0; j < kInputVectors; ++j) {
        float v = square[j];
        v += __shfl_xor_sync(0xffffffffu, v, 1);
        v += __shfl_xor_sync(0xffffffffu, v, 2);
        v += __shfl_xor_sync(0xffffffffu, v, 4);
        if (lane % 8 == 0) { squares[warp * Columns + lane / 8 + 4 * j] = v; }
    }
#pragma unroll
    for (int m = 0; m < kBlocks; ++m) {
        const int col = m * 8 + 2 * lid;
        float* at     = partial + warp * kCtaRows * Columns;
        at[gid * Columns + col]           = acc[m][0];
        at[gid * Columns + col + 1]       = acc[m][1];
        at[(gid + 8) * Columns + col]     = acc[m][2];
        at[(gid + 8) * Columns + col + 1] = acc[m][3];
    }
    __syncthreads();
    if (tid < Columns) {
        float sum = 0;
        for (int w = 0; w < kWarps; ++w) { sum += squares[w * Columns + tid]; }
        inverse[tid] = rsqrtf(sum / static_cast<float>(Hidden) + eps);
    }
    if (tid < kCtaRows * Columns) {
        float sum = 0;
        for (int w = 0; w < kWarps; ++w) { sum += partial[w * kCtaRows * Columns + tid]; }
        projected[tid] = sum;
    }
    __syncthreads();
    // d[c][g][t] = inv[t] * projected[c * 4 + g][t]; coefficients 2 and 3 (side 1) are the finish deltas.
    if (tid < 2 * kGroupsPerCta * Columns) {
        const int column = tid % Columns, g = tid / Columns % kGroupsPerCta, tap = tid / Columns / kGroupsPerCta;
        if (column < tokens) {
            finish[(column * 2 + tap) * kGroups + group0 + g] =
                __float2bfloat16_rn(inverse[column] * projected[((2 + tap) * kGroupsPerCta + g) * Columns + column]);
        }
    }
    // Prepared channels: 16 per group, the CTA's 64 consecutive ones per column.
    constexpr int kChannels = kGroupsPerCta * 16;
    for (int item = tid; item < kChannels * Columns; item += kThreads) {
        const int column = item / kChannels, channel = item % kChannels, g = channel / 16;
        if (column >= tokens) { continue; }
        const int hidden = group0 * 16 + channel;
        const float w    = __bfloat162float(mine->norm[channel]);
        const float n    = __bfloat162float(mine->residual[column * kBlockK + channel]) * inverse[column] * w;
        float value = (__bfloat162float(mine->base[channel]) +
                       inverse[column] * projected[(0 * kGroupsPerCta + g) * Columns + column]) * n;
        if (column % width > 0) {
            const float previous =
                __bfloat162float(mine->residual[(column - 1) * kBlockK + channel]) * inverse[column - 1] * w;
            value = fmaf(__bfloat162float(mine->base[kBlockK + channel]) +
                             inverse[column] * projected[(1 * kGroupsPerCta + g) * Columns + column],
                         previous, value);
        }
        prepared[static_cast<std::int64_t>(column) * Hidden + hidden] = __float2bfloat16_rn(value);
    }
}

template <int Hidden, int Columns>
void launch(const Tensor& residual, const Tensor& norm, float eps, const Tensor& base, const Weight& weight,
            Tensor& prepared, Tensor& finish, cudaStream_t stream) {
    constexpr std::size_t bytes = shared_bytes<Columns>();
    static const bool configured = [] {
        CUDA_CHECK(cudaFuncSetAttribute(dynamic_grouped_conv_prepare_fused_kernel<Hidden, Columns>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(bytes)));
        return true;
    }();
    (void)configured;
    const int tokens = residual.ne[1] * residual.ne[2];
    CUDA_CHECK(pdl::launch_consumer({dim3(Hidden / 16 / kGroupsPerCta), dim3(kThreads), bytes, stream},
                                    dynamic_grouped_conv_prepare_fused_kernel<Hidden, Columns>,
                                    static_cast<const __nv_bfloat16*>(residual.data),
                                    static_cast<const __nv_bfloat16*>(norm.data),
                                    static_cast<const __nv_bfloat16*>(base.data),
                                    static_cast<const __nv_bfloat16*>(weight.qdata),
                                    static_cast<__nv_bfloat16*>(prepared.data),
                                    static_cast<__nv_bfloat16*>(finish.data), eps, tokens,
                                    static_cast<int>(residual.ne[1])));
    CUDA_CHECK(cudaGetLastError());
}
} // namespace

bool bf16_dynamic_grouped_conv_prepare_fused_supported(int hidden, int tokens) {
    return hidden == 2560 && tokens <= 16;
}

void bf16_dynamic_grouped_conv_prepare_fused_launch(const Tensor& residual, const Tensor& norm, float eps,
                                                    const Tensor& base, const Weight& weight, Tensor& prepared,
                                                    Tensor& finish, cudaStream_t stream) {
    const int tokens = residual.ne[1] * residual.ne[2];
    if (residual.ne[0] != 2560 || tokens > 16) {
        throw std::logic_error("dynamic grouped conv prepare: the fused route takes H 2560 and at most 16 columns");
    }
    if (tokens <= 8) {
        launch<2560, 8>(residual, norm, eps, base, weight, prepared, finish, stream);
    } else {
        launch<2560, 16>(residual, norm, eps, base, weight, prepared, finish, stream);
    }
}
} // namespace infernix::ops::detail
