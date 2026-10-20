#pragma once

// fp8-a8-pair (LinearRoutes::Fp8A8Pair): the A8 TMA GEMM with the activation carried as two E4M3
// terms (fp8_a8_plan.h). Each staged weight tile feeds two accumulator sets, one per term, from the
// same B fragments; the terms are combined in FP32 with their own column scales, then the row
// scale and the single BF16 rounding of the A8 route's epilogue are applied. The exact routes'
// kernels are untouched.

#include "core/tensor.h"
#include "core/weight.h"
#include "ops/common/math.h"
#include "ops/common/token_slices.h"
#include "ops/linear/common/epilogue.cuh"
#include "ops/linear/common/output.cuh"
#include "ops/linear/fp8/fp8_a8_mma_common.cuh"
#include "ops/linear/fp8/fp8_a8_plan.h"
#include "ops/linear/fp8/fp8_a8_tma_mma.cuh"
#include "ops/linear/fp8/fp8_operands.h"
#include "ops/linear/fp8/fp8_schedule.cuh"
#include "ops/linear/fp8/fp8_shared.cuh"

#include <cuda.h>
#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace infernix::ops::detail {

// A TMA schedule whose stages hold both terms' activation tiles beside one weight tile.
template <class Base>
struct Fp8A8TmaPairSchedule : Base {
    static_assert(Base::kSplitWaveCtas == 0, "the pair route has no split-K plan");
    static constexpr int kPipelineBytes =
        Base::kStages * (2 * Base::kBlockTokens + Base::kBlockRows) * Base::kBlockK;
    static constexpr int kOutputStagingBytes = Base::kBlockTokens * (Base::kBlockRows + 8) * 2;
    static constexpr int kStorageBytes =
        kPipelineBytes > kOutputStagingBytes ? kPipelineBytes : kOutputStagingBytes;
    static constexpr int kSharedBytes = kStorageBytes + Base::kBarrierBytes;
    static_assert(kSharedBytes <= 99 * 1024);
};

// Up to kFp8A8PairNarrowTokens columns: 64-column tiles (more CTAs for short prompts); beyond,
// 128-column tiles. Both use 128-row tiles, so every registered pair problem's N is whole tiles.
using Fp8A8PairT64  = Fp8A8TmaPairSchedule<Fp8A8TmaMmaSchedule<64, 128, 128, 2, 4, 3, 1>>;
using Fp8A8PairT128 = Fp8A8TmaPairSchedule<Fp8A8TmaMmaSchedule<128, 128, 128, 2, 4, 2, 1>>;
inline constexpr std::int32_t kFp8A8PairNarrowTokens = 256;

struct alignas(128) Fp8TmaPairDescriptors {
    CUtensorMap activation_hi;
    CUtensorMap activation_lo;
    CUtensorMap weight;
};

#ifdef _WIN32
// The Windows descriptor staging of the pair route (core/tma_descriptor_staging.cuh).
inline TmaDescriptorStaging<Fp8TmaPairDescriptors>& fp8_tma_pair_descriptor_staging() {
    static TmaDescriptorStaging<Fp8TmaPairDescriptors> staging;
    return staging;
}
#endif

// fp8_mma_compute_stage with a second activation plane: both terms' A fragments meet the same B
// fragments, so the weight tile is read from shared memory once per k step.
template <class Schedule>
__device__ __forceinline__ void
fp8_mma_pair_compute_stage(const std::uint8_t* activation_hi, const std::uint8_t* activation_lo,
                           const std::uint8_t* weight_stage,
                           float (&hi)[Schedule::kMmaTokens][Schedule::kMmaRows][4],
                           float (&lo)[Schedule::kMmaTokens][Schedule::kMmaRows][4], int warp,
                           int lane) {
    constexpr int BK        = Schedule::kBlockK;
    constexpr int Tokens    = Schedule::kMmaTokens;
    constexpr int Rows      = Schedule::kMmaRows;
    const int warp_token    = warp / Schedule::kWarpsRows;
    const int warp_row      = warp % Schedule::kWarpsRows;
    const int a_matrix      = lane >> 3;
    const int a_row_offset  = (lane & 7) + ((a_matrix & 1) << 3);
    const int a_column_byte = (a_matrix >> 1) * 16;
    const int b_row_offset  = lane & 7;
    const int b_column_byte = ((lane >> 3) & 1) * 16;
#pragma unroll
    for (int k_step = 0; k_step < Schedule::kMmaK; ++k_step) {
        unsigned b[Rows][2];
        unsigned a_hi[Tokens][4];
        unsigned a_lo[Tokens][4];
#pragma unroll
        for (int mma_row = 0; mma_row < Rows; ++mma_row) {
            const int row           = warp_row * Schedule::kWarpRows + mma_row * 8 + b_row_offset;
            const int physical_byte = fp8_mma_shared_byte<Schedule>(row, k_step * 32 + b_column_byte);
            ldmatrix_x2(b[mma_row][0], b[mma_row][1],
                        smem_addr(weight_stage + row * BK + physical_byte));
        }
#pragma unroll
        for (int mma_token = 0; mma_token < Tokens; ++mma_token) {
            const int row = warp_token * Schedule::kWarpTokens + mma_token * 16 + a_row_offset;
            const int physical_byte = fp8_mma_shared_byte<Schedule>(row, k_step * 32 + a_column_byte);
            ldmatrix_x4(a_hi[mma_token][0], a_hi[mma_token][1], a_hi[mma_token][2],
                        a_hi[mma_token][3], smem_addr(activation_hi + row * BK + physical_byte));
            ldmatrix_x4(a_lo[mma_token][0], a_lo[mma_token][1], a_lo[mma_token][2],
                        a_lo[mma_token][3], smem_addr(activation_lo + row * BK + physical_byte));
        }
#pragma unroll
        for (int mma_token = 0; mma_token < Tokens; ++mma_token) {
#pragma unroll
            for (int mma_row = 0; mma_row < Rows; ++mma_row) {
                mma_fp8_e4m3(hi[mma_token][mma_row][0], hi[mma_token][mma_row][1],
                             hi[mma_token][mma_row][2], hi[mma_token][mma_row][3],
                             a_hi[mma_token][0], a_hi[mma_token][1], a_hi[mma_token][2],
                             a_hi[mma_token][3], b[mma_row][0], b[mma_row][1]);
                mma_fp8_e4m3(lo[mma_token][mma_row][0], lo[mma_token][mma_row][1],
                             lo[mma_token][mma_row][2], lo[mma_token][mma_row][3],
                             a_lo[mma_token][0], a_lo[mma_token][1], a_lo[mma_token][2],
                             a_lo[mma_token][3], b[mma_row][0], b[mma_row][1]);
            }
        }
    }
}

// `operands` carries the high term (x, x_scales); the low term's codes reach the kernel only
// through its tensor map, and its column scales through `scales_lo`.
template <class Schedule, bool FullTokens>
__global__ __launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void
fp8_a8_pair_tma_mma_kernel(
#ifdef _WIN32
    // MSVC cannot pass the over-aligned descriptor struct by value (fp8_a8_tma_mma_kernel).
    const Fp8TmaPairDescriptors* descriptors_pointer,
#else
    const __grid_constant__ Fp8TmaPairDescriptors descriptors,
#endif
    Fp8A8Operands operands, const float* __restrict__ scales_lo, LinearBf16Output output,
    int token_offset, int count) {
#ifdef _WIN32
    const Fp8TmaPairDescriptors& descriptors = *descriptors_pointer;
#endif
    constexpr int BT = Schedule::kBlockTokens, BR = Schedule::kBlockRows;
    constexpr int BK = Schedule::kBlockK, S = Schedule::kStages;
    constexpr int plane_bytes = S * BT * BK;
    const int k               = Schedule::kStaticK ? Schedule::kStaticK : operands.k;
    const int tiles_k         = k / BK;
    const int row_tiles = operands.rows / BR, token_tiles = div_up(count, BT);
    int row_tile, token_tile;
    fp8_mma_tile_coordinates<Schedule>(static_cast<std::int32_t>(blockIdx.x), row_tiles,
                                       token_tiles, row_tile, token_tile);
    const int row_begin   = row_tile * BR;
    const int token_begin = token_offset + token_tile * BT;

    // Every stage base is a multiple of 1,024 bytes, as the 128-byte TMA swizzle requires.
    extern __shared__ __align__(128) unsigned char fp8_pair_shared[];
    auto* activation_hi = fp8_pair_shared;
    auto* activation_lo = activation_hi + plane_bytes;
    auto* weight        = activation_lo + plane_bytes;
    auto* full          = reinterpret_cast<std::uint64_t*>(
        fp8_pair_shared + fp8_tma_scratch_bytes<Schedule, LinearIdentityEpilogue>);
    auto* empty = full + S;
    if (threadIdx.x == 0) {
#pragma unroll
        for (int stage = 0; stage < S; ++stage) {
            cta_mbarrier_init(full + stage, 1);
            cta_mbarrier_init(empty + stage, Schedule::kConsumerWarps);
        }
        cta_mbarrier_fence_init();
    }
    __syncthreads();

    if (threadIdx.x < Schedule::kProducerThreads) {
        if (threadIdx.x == 0) {
#ifdef _WIN32
            acquire_staged_tensor_map(&descriptors.activation_hi);
            acquire_staged_tensor_map(&descriptors.activation_lo);
            acquire_staged_tensor_map(&descriptors.weight);
#endif
            for (int kt = 0; kt < tiles_k; ++kt) {
                const int stage = kt % S;
                cta_mbarrier_wait(empty + stage, 1U ^ ((kt / S) & 1U));
                cta_mbarrier_arrive_expect_tx(full + stage, (2 * BT + BR) * BK);
                fp8_tma_load(activation_hi + stage * BT * BK, &descriptors.activation_hi, kt * BK,
                             token_begin, full + stage);
                fp8_tma_load(activation_lo + stage * BT * BK, &descriptors.activation_lo, kt * BK,
                             token_begin, full + stage);
                fp8_tma_load(weight + stage * BR * BK, &descriptors.weight, kt * BK, row_begin,
                             full + stage);
            }
        }
        return;
    }

    const int tid  = static_cast<int>(threadIdx.x) - Schedule::kProducerThreads;
    const int warp = tid >> 5, lane = tid & 31;
    float hi[Schedule::kMmaTokens][Schedule::kMmaRows][4] = {};
    float lo[Schedule::kMmaTokens][Schedule::kMmaRows][4] = {};
    for (int kt = 0; kt < tiles_k; ++kt) {
        const int stage = kt % S;
        cta_mbarrier_wait(full + stage, (kt / S) & 1U);
        fp8_mma_pair_compute_stage<Schedule>(activation_hi + stage * BT * BK,
                                             activation_lo + stage * BT * BK,
                                             weight + stage * BR * BK, hi, lo, warp, lane);
        // Release only after every lane in this consumer warp has finished its shared reads,
        // ordered before the TMA refill.
        fence_reads_before_async_refill();
        __syncwarp();
        if (lane == 0) cta_mbarrier_arrive(empty + stage);
    }

    // Each term keeps its own column scale until the two are summed in FP32. The row scale and the
    // BF16 rounding follow in the A8 epilogue, which then sees unit column scales.
    const int tokens = token_offset + count;
#pragma unroll
    for (int mma_token = 0; mma_token < Schedule::kMmaTokens; ++mma_token) {
        const int token0 = token_begin + warp / Schedule::kWarpsRows * Schedule::kWarpTokens +
                           mma_token * 16 + (lane >> 2);
        const int token1      = token0 + 8;
        const bool valid0     = FullTokens || token0 < tokens;
        const bool valid1     = FullTokens || token1 < tokens;
        const float scale_hi0 = valid0 ? __ldg(operands.x_scales + token0) : 0.0F;
        const float scale_lo0 = valid0 ? __ldg(scales_lo + token0) : 0.0F;
        const float scale_hi1 = valid1 ? __ldg(operands.x_scales + token1) : 0.0F;
        const float scale_lo1 = valid1 ? __ldg(scales_lo + token1) : 0.0F;
#pragma unroll
        for (int mma_row = 0; mma_row < Schedule::kMmaRows; ++mma_row) {
            float(&h)[4]       = hi[mma_token][mma_row];
            const float(&l)[4] = lo[mma_token][mma_row];
            h[0]               = h[0] * scale_hi0 + l[0] * scale_lo0;
            h[1]               = h[1] * scale_hi0 + l[1] * scale_lo0;
            h[2]               = h[2] * scale_hi1 + l[2] * scale_lo1;
            h[3]               = h[3] * scale_hi1 + l[3] * scale_lo1;
        }
    }
    // All consumers must finish reading staged inputs before the epilogue reuses the storage.
    __syncthreads();
    fp8_finish_mma_tile<Schedule, FullTokens>(output, LinearIdentityEpilogue{}, Fp8IdentityRows{},
                                              fp8_pair_shared, hi, nullptr, operands.scales,
                                              row_begin, token_begin, operands.rows, tokens, warp,
                                              lane);
}

template <class Schedule>
void launch_fp8_a8_pair_tma_mma(const Fp8A8Operands& p, const std::uint8_t* x_lo,
                                const float* scales_lo, LinearBf16Output output,
                                cudaStream_t stream) {
    validate_fp8_operands<Schedule>(p);
    if (x_lo == nullptr || reinterpret_cast<std::uintptr_t>(x_lo) % 16 != 0 ||
        scales_lo == nullptr || reinterpret_cast<std::uintptr_t>(scales_lo) % 4 != 0)
        throw std::invalid_argument("FP8 A8 pair requires an aligned second activation term");
    if (p.rows % Schedule::kBlockRows || p.k % Schedule::kBlockK)
        throw std::invalid_argument("FP8 A8 pair requires complete row/K tiles");
    // Descriptors are launch-owned values, copied into kernel parameters during Graph capture.
    const Fp8TmaPairDescriptors descriptors{
        fp8_tma_map(p.x, p.tokens, p.k, Schedule::kBlockTokens, Schedule::kBlockK),
        fp8_tma_map(x_lo, p.tokens, p.k, Schedule::kBlockTokens, Schedule::kBlockK),
        fp8_tma_map(p.codes, p.rows, p.k, Schedule::kBlockRows, Schedule::kBlockK)};
#ifdef _WIN32
    // Staged once: every token-slice launch below reads the same copy, in stream order.
    const Fp8TmaPairDescriptors* staged =
        fp8_tma_pair_descriptor_staging().stage(descriptors, stream);
#endif
    for_each_token_slice(p.tokens, Schedule::kBlockTokens, [&](int offset, int count) {
        const int blocks  = p.rows / Schedule::kBlockRows * div_up(count, Schedule::kBlockTokens);
        const auto launch = [&]<bool Full>() {
            constexpr auto kernel = fp8_a8_pair_tma_mma_kernel<Schedule, Full>;
            constexpr int bytes =
                fp8_tma_scratch_bytes<Schedule, LinearIdentityEpilogue> + Schedule::kBarrierBytes;
            const int dynamic = fp8_prepare_shared<bytes, kernel, true>();
#ifdef _WIN32
            kernel<<<blocks, Schedule::kThreads, dynamic, stream>>>(staged, p, scales_lo, output,
                                                                    offset, count);
#else
            kernel<<<blocks, Schedule::kThreads, dynamic, stream>>>(descriptors, p, scales_lo,
                                                                    output, offset, count);
#endif
            CUDA_CHECK(cudaGetLastError());
        };
        if (count % Schedule::kBlockTokens == 0)
            launch.template operator()<true>();
        else
            launch.template operator()<false>();
    });
}

// The complete pair route of one registered problem: both terms, then the fused GEMM.
template <class Geometry>
void launch_fp8_a8_pair(const Tensor& x, const Weight& w, Tensor& out, Fp8A8PairWorkspace scratch,
                        cudaStream_t stream) {
    launch_fp8_a8_pair_quantize(x, w, scratch, stream);
    const Fp8A8Operands operands =
        fp8_a8_operands(w, Fp8A8Workspace{scratch.codes_hi, scratch.scales_hi, nullptr}, x.ne[1]);
    const LinearBf16Output output{static_cast<__nv_bfloat16*>(out.data), w.n};
    if (x.ne[1] <= kFp8A8PairNarrowTokens) {
        launch_fp8_a8_pair_tma_mma<Fp8ScheduleInstance<Fp8A8PairT64, Geometry::kInputRows>>(
            operands, scratch.codes_lo, scratch.scales_lo, output, stream);
        return;
    }
    launch_fp8_a8_pair_tma_mma<Fp8ScheduleInstance<Fp8A8PairT128, Geometry::kInputRows>>(
        operands, scratch.codes_lo, scratch.scales_lo, output, stream);
}

} // namespace infernix::ops::detail
