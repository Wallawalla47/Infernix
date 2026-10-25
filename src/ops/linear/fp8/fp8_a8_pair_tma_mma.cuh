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
#include <string>
#include <utility>

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
template <class Schedule, bool FullTokens, bool InPlace = false>
__global__ __launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void
fp8_a8_pair_tma_mma_kernel(
#ifdef _WIN32
    // MSVC cannot pass the over-aligned descriptor struct by value (fp8_a8_tma_mma_kernel).
    const __grid_constant__ TmaDescriptorParameter<Fp8TmaPairDescriptors, InPlace> descriptors_pointer,
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
            acquire_staged_tensor_map<InPlace>(&descriptors.activation_hi);
            acquire_staged_tensor_map<InPlace>(&descriptors.activation_lo);
            acquire_staged_tensor_map<InPlace>(&descriptors.weight);
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
    const auto launch_slices = [&]<bool InPlace>(std::bool_constant<InPlace>, const auto& parameter) {
        for_each_token_slice(p.tokens, Schedule::kBlockTokens, [&](int offset, int count) {
            const int blocks  = p.rows / Schedule::kBlockRows * div_up(count, Schedule::kBlockTokens);
            const auto launch = [&]<bool Full>() {
                constexpr auto kernel = fp8_a8_pair_tma_mma_kernel<Schedule, Full, InPlace>;
                constexpr int bytes =
                    fp8_tma_scratch_bytes<Schedule, LinearIdentityEpilogue> + Schedule::kBarrierBytes;
                const int dynamic = fp8_prepare_shared<bytes, kernel, true>();
                kernel<<<blocks, Schedule::kThreads, dynamic, stream>>>(parameter, p, scales_lo, output,
                                                                        offset, count);
                CUDA_CHECK(cudaGetLastError());
            };
            if (count % Schedule::kBlockTokens == 0)
                launch.template operator()<true>();
            else
                launch.template operator()<false>();
        });
    };
#ifdef _WIN32
    // Staged once: every token-slice launch reads the same copy, in stream order.
    with_tma_descriptor_parameter(fp8_tma_pair_descriptor_staging(), descriptors, stream, launch_slices);
#else
    launch_slices(std::false_type{}, descriptors);
#endif
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

// fp8-a8-pair-block32 (LinearRoutes::Fp8A8PairBlock32): the pair GEMM with both terms' UE8M0 scales
// per 32 values of K (fp8_a8_plan.h), applied by the block-scaled MMA. Each stage holds both terms'
// code tiles, one weight tile and both terms' scale words for its K tile, all filled by TMA, so the
// scales cost one shared word per term and 16-token MMA tile per stage. Both terms accumulate into
// one FP32 set from the same B fragments. Every consumer fences its shared reads (codes and scale
// words) before releasing a stage to the TMA refill (op-development.md 4.3).
//
// Two weight formats share the kernel:
// - row-scaled FP8 (recipe C): the MMA's weight scale is 1, and the BF16 row scale and the single
//   BF16 rounding of the A8 route's epilogue follow;
// - MXFP8 (NVIDIA FP4-Local, kBlockScaledWeights): the weight's own UE8M0 scale of each 32-value group
//   is the MMA's weight scale, so the products are the represented weights' and the epilogue scales
//   nothing. A stage has no room for the scales beside the pipeline (99,888 B for the 64-column tile
//   already), so each consumer lane reads its columns' four group bytes of the next K tile with
//   __ldg while the current tile computes (the immutable plane stays in L1/L2) and packs them into
//   one word per MMA row, whose byte the k32 step selects.
struct Fp8A8PairBlock32Operands {
    const std::uint8_t* x;    // the high term's codes [T][K]
    const std::uint8_t* x_lo; // the low term's codes [T][K]
    const std::uint32_t* block_scales_hi;
    const std::uint32_t* block_scales_lo;
    const std::uint8_t* codes;
    const __nv_bfloat16* scales;              // row FP8: the BF16 row scales
    const std::uint8_t* weight_block_scales;  // MXFP8: the UE8M0 plane [N/16][K/32][16]
    int rows, k, tokens;
};

template <class Base>
struct Fp8A8TmaPairBlock32Schedule : Base {
    static_assert(Base::kSplitWaveCtas == 0, "the pair route has no split-K plan");
    static_assert(Base::kBlockK == kFp8A8PairBlock32K,
                  "a stage's scale word holds its K tile's four 32-value blocks");
    // Both terms' scale words of one stage: one 32-bit word per token and term.
    static constexpr int kScaleStageBytes = Base::kBlockTokens * 4;
    static constexpr int kCodeBytes =
        Base::kStages * (2 * Base::kBlockTokens + Base::kBlockRows) * Base::kBlockK;
    static constexpr int kPipelineBytes      = kCodeBytes + Base::kStages * 2 * kScaleStageBytes;
    static constexpr int kOutputStagingBytes = Base::kBlockTokens * (Base::kBlockRows + 8) * 2;
    static constexpr int kStorageBytes =
        kPipelineBytes > kOutputStagingBytes ? kPipelineBytes : kOutputStagingBytes;
    static constexpr int kSharedBytes = kStorageBytes + Base::kBarrierBytes;
    static_assert(kSharedBytes <= 99 * 1024);
};

// The tile widths and stage counts of the column-scaled pair route (T64: three stages; T128: two).
using Fp8A8PairBlock32T64  = Fp8A8TmaPairBlock32Schedule<Fp8A8TmaMmaSchedule<64, 128, 128, 2, 4, 3, 1>>;
using Fp8A8PairBlock32T128 = Fp8A8TmaPairBlock32Schedule<Fp8A8TmaMmaSchedule<128, 128, 128, 2, 4, 2, 1>>;

struct alignas(128) Fp8TmaPairBlock32Descriptors {
    CUtensorMap activation_hi;
    CUtensorMap activation_lo;
    CUtensorMap weight;
    CUtensorMap scales_hi;
    CUtensorMap scales_lo;
};

#ifdef _WIN32
// The Windows descriptor staging of the block32 pair route (core/tma_descriptor_staging.cuh).
inline TmaDescriptorStaging<Fp8TmaPairBlock32Descriptors>& fp8_tma_pair_block32_descriptor_staging() {
    static TmaDescriptorStaging<Fp8TmaPairBlock32Descriptors> staging;
    return staging;
}
#endif

// A term's scale words [K / 128][stride] as a TMA map whose box is one K tile's `box_tokens` words.
// Tokens at or past T read zero (scale 2^-127 against zero codes).
inline CUtensorMap fp8_tma_scale_map(const std::uint32_t* words, int tokens, int k_tiles,
                                     int stride, int box_tokens) {
    CUtensorMap result{};
    const std::uint64_t dimensions[]{static_cast<std::uint64_t>(tokens),
                                     static_cast<std::uint64_t>(k_tiles)};
    const std::uint64_t strides[]{static_cast<std::uint64_t>(stride) * sizeof(std::uint32_t)};
    const std::uint32_t box[]{static_cast<std::uint32_t>(box_tokens), 1};
    const std::uint32_t steps[]{1, 1};
    const auto status = cuTensorMapEncodeTiled(
        &result, CU_TENSOR_MAP_DATA_TYPE_UINT32, 2, const_cast<std::uint32_t*>(words), dimensions,
        strides, box, steps, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (status != CUDA_SUCCESS) {
        const char* name = nullptr;
        (void)cuGetErrorName(status, &name);
        throw std::runtime_error(std::string("FP8 pair block32 scale TMA descriptor: ") +
                                 (name ? name : "CUDA error"));
    }
    return result;
}

// An MXFP8 lane's weight scales of K tile `kt`: per MMA row, the four 32-value groups' UE8M0 bytes of
// the B column the lane's scale selector serves (column lane >> 2, mma.cuh), group j in byte j.
template <class Schedule>
__device__ __forceinline__ void
fp8_pair_block32_weight_scales(const std::uint8_t* __restrict__ plane, int k, int row_begin, int kt,
                               int warp, int lane, unsigned (&words)[Schedule::kMmaRows]) {
    const int warp_row = warp % Schedule::kWarpsRows;
#pragma unroll
    for (int mma_row = 0; mma_row < Schedule::kMmaRows; ++mma_row) {
        const int row = row_begin + warp_row * Schedule::kWarpRows + mma_row * 8 + (lane >> 2);
        const std::uint8_t* groups =
            plane + (static_cast<std::size_t>(row >> 4) * (k / 32) + static_cast<std::size_t>(kt) * 4) * 16 +
            (row & 15);
        words[mma_row] = static_cast<unsigned>(__ldg(groups)) | (static_cast<unsigned>(__ldg(groups + 16)) << 8) |
                         (static_cast<unsigned>(__ldg(groups + 32)) << 16) |
                         (static_cast<unsigned>(__ldg(groups + 48)) << 24);
    }
}

// One k32 step of a stage: B fragments read once for both terms; every high-term MMA is issued
// before the low-term MMAs into the same accumulators, so dependent MMAs are a whole tile apart.
// KStep selects the step's scale byte of each activation word (and, for MXFP8, of each weight word).
template <class Schedule, bool BlockScaledWeights, int KStep>
__device__ __forceinline__ void
fp8_mma_pair_block32_k_step(const std::uint8_t* activation_hi, const std::uint8_t* activation_lo,
                            const std::uint8_t* weight_stage,
                            const unsigned (&scale_hi)[Schedule::kMmaTokens],
                            const unsigned (&scale_lo)[Schedule::kMmaTokens],
                            const unsigned (&scale_b)[Schedule::kMmaRows],
                            float (&accumulators)[Schedule::kMmaTokens][Schedule::kMmaRows][4],
                            int warp, int lane) {
    constexpr int BK        = Schedule::kBlockK;
    constexpr int Tokens    = Schedule::kMmaTokens;
    constexpr int Rows      = Schedule::kMmaRows;
    constexpr unsigned kUnitScale = 0x7FU;
    const int warp_token    = warp / Schedule::kWarpsRows;
    const int warp_row      = warp % Schedule::kWarpsRows;
    const int a_matrix      = lane >> 3;
    const int a_row_offset  = (lane & 7) + ((a_matrix & 1) << 3);
    const int a_column_byte = (a_matrix >> 1) * 16;
    const int b_row_offset  = lane & 7;
    const int b_column_byte = ((lane >> 3) & 1) * 16;
    unsigned b[Rows][2];
    unsigned a_hi[Tokens][4];
    unsigned a_lo[Tokens][4];
#pragma unroll
    for (int mma_row = 0; mma_row < Rows; ++mma_row) {
        const int row           = warp_row * Schedule::kWarpRows + mma_row * 8 + b_row_offset;
        const int physical_byte = fp8_mma_shared_byte<Schedule>(row, KStep * 32 + b_column_byte);
        ldmatrix_x2(b[mma_row][0], b[mma_row][1], smem_addr(weight_stage + row * BK + physical_byte));
    }
#pragma unroll
    for (int mma_token = 0; mma_token < Tokens; ++mma_token) {
        const int row = warp_token * Schedule::kWarpTokens + mma_token * 16 + a_row_offset;
        const int physical_byte = fp8_mma_shared_byte<Schedule>(row, KStep * 32 + a_column_byte);
        ldmatrix_x4(a_hi[mma_token][0], a_hi[mma_token][1], a_hi[mma_token][2], a_hi[mma_token][3],
                    smem_addr(activation_hi + row * BK + physical_byte));
        ldmatrix_x4(a_lo[mma_token][0], a_lo[mma_token][1], a_lo[mma_token][2], a_lo[mma_token][3],
                    smem_addr(activation_lo + row * BK + physical_byte));
    }
    const auto mma = [&](float (&c)[4], const unsigned (&a)[4], const unsigned (&w)[2], unsigned scale_a,
                         int mma_row) {
        if constexpr (BlockScaledWeights) {
            mma_fp8_e4m3_block_scaled<KStep, KStep>(c[0], c[1], c[2], c[3], a[0], a[1], a[2], a[3], w[0], w[1],
                                                    scale_a, scale_b[mma_row]);
        } else {
            mma_fp8_e4m3_block_scaled<KStep, 0>(c[0], c[1], c[2], c[3], a[0], a[1], a[2], a[3], w[0], w[1],
                                                scale_a, kUnitScale);
        }
    };
#pragma unroll
    for (int mma_token = 0; mma_token < Tokens; ++mma_token) {
#pragma unroll
        for (int mma_row = 0; mma_row < Rows; ++mma_row) {
            mma(accumulators[mma_token][mma_row], a_hi[mma_token], b[mma_row], scale_hi[mma_token], mma_row);
        }
    }
#pragma unroll
    for (int mma_token = 0; mma_token < Tokens; ++mma_token) {
#pragma unroll
        for (int mma_row = 0; mma_row < Rows; ++mma_row) {
            mma(accumulators[mma_token][mma_row], a_lo[mma_token], b[mma_row], scale_lo[mma_token], mma_row);
        }
    }
}

// A stage's k32 steps in order, each with its compile-time scale byte.
template <class Schedule, bool BlockScaledWeights, int... KSteps>
__device__ __forceinline__ void
fp8_mma_pair_block32_compute_stage(const std::uint8_t* activation_hi,
                                   const std::uint8_t* activation_lo,
                                   const std::uint8_t* weight_stage,
                                   const unsigned (&scale_hi)[Schedule::kMmaTokens],
                                   const unsigned (&scale_lo)[Schedule::kMmaTokens],
                                   const unsigned (&scale_b)[Schedule::kMmaRows],
                                   float (&accumulators)[Schedule::kMmaTokens][Schedule::kMmaRows][4],
                                   int warp, int lane, std::integer_sequence<int, KSteps...>) {
    static_assert(sizeof...(KSteps) == 4, "a 128-value K tile is four k32 steps, one scale byte each");
    (fp8_mma_pair_block32_k_step<Schedule, BlockScaledWeights, KSteps>(
         activation_hi, activation_lo, weight_stage, scale_hi, scale_lo, scale_b, accumulators, warp, lane),
     ...);
}

template <class Schedule, bool FullTokens, bool BlockScaledWeights, class Output, bool InPlace = false>
__global__ __launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void
fp8_a8_pair_block32_tma_mma_kernel(
#ifdef _WIN32
    // MSVC cannot pass the over-aligned descriptor struct by value (fp8_a8_tma_mma_kernel).
    const __grid_constant__ TmaDescriptorParameter<Fp8TmaPairBlock32Descriptors, InPlace> descriptors_pointer,
#else
    const __grid_constant__ Fp8TmaPairBlock32Descriptors descriptors,
#endif
    Fp8A8PairBlock32Operands operands, Output output, int token_offset, int count) {
#ifdef _WIN32
    const Fp8TmaPairBlock32Descriptors& descriptors = *descriptors_pointer;
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

    // Every code stage base is a multiple of 1,024 bytes, as the 128-byte TMA swizzle requires; the
    // scale words follow the code planes.
    extern __shared__ __align__(128) unsigned char fp8_pair_block32_shared[];
    auto* activation_hi = fp8_pair_block32_shared;
    auto* activation_lo = activation_hi + plane_bytes;
    auto* weight        = activation_lo + plane_bytes;
    auto* scales_hi =
        reinterpret_cast<std::uint32_t*>(fp8_pair_block32_shared + Schedule::kCodeBytes);
    auto* scales_lo = scales_hi + S * BT;
    auto* full      = reinterpret_cast<std::uint64_t*>(
        fp8_pair_block32_shared + fp8_tma_scratch_bytes<Schedule, LinearIdentityEpilogue>);
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
            acquire_staged_tensor_map<InPlace>(&descriptors.activation_hi);
            acquire_staged_tensor_map<InPlace>(&descriptors.activation_lo);
            acquire_staged_tensor_map<InPlace>(&descriptors.weight);
            acquire_staged_tensor_map<InPlace>(&descriptors.scales_hi);
            acquire_staged_tensor_map<InPlace>(&descriptors.scales_lo);
#endif
#pragma unroll 1
            for (int kt = 0; kt < tiles_k; ++kt) {
                const int stage = kt % S;
                cta_mbarrier_wait(empty + stage, 1U ^ ((kt / S) & 1U));
                cta_mbarrier_arrive_expect_tx(full + stage,
                                              (2 * BT + BR) * BK + 2 * Schedule::kScaleStageBytes);
                fp8_tma_load(activation_hi + stage * BT * BK, &descriptors.activation_hi, kt * BK,
                             token_begin, full + stage);
                fp8_tma_load(activation_lo + stage * BT * BK, &descriptors.activation_lo, kt * BK,
                             token_begin, full + stage);
                fp8_tma_load(weight + stage * BR * BK, &descriptors.weight, kt * BK, row_begin,
                             full + stage);
                fp8_tma_load(scales_hi + stage * BT, &descriptors.scales_hi, token_begin, kt,
                             full + stage);
                fp8_tma_load(scales_lo + stage * BT, &descriptors.scales_lo, token_begin, kt,
                             full + stage);
            }
        }
        return;
    }

    const int tid  = static_cast<int>(threadIdx.x) - Schedule::kProducerThreads;
    const int warp = tid >> 5, lane = tid & 31;
    // The token whose scale word this lane supplies in each 16-token MMA tile (mma.cuh).
    const int scale_token = warp / Schedule::kWarpsRows * Schedule::kWarpTokens +
                            (((lane & 1) << 3) | (lane >> 2));
    float accumulators[Schedule::kMmaTokens][Schedule::kMmaRows][4] = {};
    // MXFP8: this K tile's weight scale words, and the next tile's while this one computes.
    unsigned weight_scales[Schedule::kMmaRows] = {};
    unsigned next_weight_scales[Schedule::kMmaRows] = {};
    if constexpr (BlockScaledWeights) {
        fp8_pair_block32_weight_scales<Schedule>(operands.weight_block_scales, k, row_begin, 0, warp, lane,
                                                 weight_scales);
    }
#pragma unroll 1
    for (int kt = 0; kt < tiles_k; ++kt) {
        const int stage = kt % S;
        if constexpr (BlockScaledWeights) {
            if (kt + 1 < tiles_k) {
                fp8_pair_block32_weight_scales<Schedule>(operands.weight_block_scales, k, row_begin, kt + 1, warp,
                                                         lane, next_weight_scales);
            }
        }
        cta_mbarrier_wait(full + stage, (kt / S) & 1U);
        unsigned scale_hi[Schedule::kMmaTokens], scale_lo[Schedule::kMmaTokens];
#pragma unroll
        for (int mma_token = 0; mma_token < Schedule::kMmaTokens; ++mma_token) {
            scale_hi[mma_token] = scales_hi[stage * BT + scale_token + mma_token * 16];
            scale_lo[mma_token] = scales_lo[stage * BT + scale_token + mma_token * 16];
        }
        fp8_mma_pair_block32_compute_stage<Schedule, BlockScaledWeights>(
            activation_hi + stage * BT * BK, activation_lo + stage * BT * BK,
            weight + stage * BR * BK, scale_hi, scale_lo, weight_scales, accumulators, warp, lane,
            std::make_integer_sequence<int, Schedule::kMmaK>{});
        // Release only after every lane in this consumer warp has finished its shared reads (codes
        // and scale words), ordered before the TMA refill.
        fence_reads_before_async_refill();
        __syncwarp();
        if (lane == 0) cta_mbarrier_arrive(empty + stage);
        if constexpr (BlockScaledWeights) {
#pragma unroll
            for (int mma_row = 0; mma_row < Schedule::kMmaRows; ++mma_row) {
                weight_scales[mma_row] = next_weight_scales[mma_row];
            }
        }
    }

    // All consumers must finish reading staged inputs before the epilogue reuses the storage.
    __syncthreads();
    if constexpr (BlockScaledWeights) {
        // The represented weights carry their scales: the epilogue multiplies by nothing.
        fp8_finish_mma_tile<Schedule, FullTokens>(output, LinearIdentityEpilogue{}, Fp8UnscaledRows{},
                                                  fp8_pair_block32_shared, accumulators, nullptr,
                                                  static_cast<const __nv_bfloat16*>(nullptr), row_begin,
                                                  token_begin, operands.rows, token_offset + count, warp, lane);
    } else {
        fp8_finish_mma_tile<Schedule, FullTokens>(output, LinearIdentityEpilogue{}, Fp8IdentityRows{},
                                                  fp8_pair_block32_shared, accumulators, nullptr,
                                                  operands.scales, row_begin, token_begin,
                                                  operands.rows, token_offset + count, warp, lane);
    }
}

template <class Schedule, bool BlockScaledWeights, class Output>
void launch_fp8_a8_pair_block32_tma_mma(const Fp8A8PairBlock32Operands& p, Output output,
                                        cudaStream_t stream) {
    const auto aligned = [](const void* v, int alignment) {
        return v && reinterpret_cast<std::uintptr_t>(v) % alignment == 0;
    };
    if (p.rows <= 0 || p.k <= 0 || p.tokens <= 0 || !aligned(p.x, 16) || !aligned(p.codes, 16) ||
        !aligned(p.x_lo, 16) || !aligned(p.block_scales_hi, 16) || !aligned(p.block_scales_lo, 16) ||
        (BlockScaledWeights ? !aligned(p.weight_block_scales, 16) || p.rows % 16 != 0 : !aligned(p.scales, 4)))
        throw std::invalid_argument("FP8 A8 pair block32 requires positive N/K/T, an aligned second term, "
                                    "block scales and the weight's scales");
    if constexpr (Schedule::kStaticK > 0) {
        if (p.k != Schedule::kStaticK)
            throw std::invalid_argument("FP8 template static K does not match operands");
    }
    if (p.rows % Schedule::kBlockRows || p.k % Schedule::kBlockK)
        throw std::invalid_argument("FP8 A8 pair block32 requires complete row/K tiles");
    const int k_tiles = p.k / Schedule::kBlockK;
    const int stride  = fp8_a8_pair_block32_scale_stride(p.tokens);
    // Descriptors are launch-owned values, copied into kernel parameters during Graph capture.
    const Fp8TmaPairBlock32Descriptors descriptors{
        fp8_tma_map(p.x, p.tokens, p.k, Schedule::kBlockTokens, Schedule::kBlockK),
        fp8_tma_map(p.x_lo, p.tokens, p.k, Schedule::kBlockTokens, Schedule::kBlockK),
        fp8_tma_map(p.codes, p.rows, p.k, Schedule::kBlockRows, Schedule::kBlockK),
        fp8_tma_scale_map(p.block_scales_hi, p.tokens, k_tiles, stride, Schedule::kBlockTokens),
        fp8_tma_scale_map(p.block_scales_lo, p.tokens, k_tiles, stride, Schedule::kBlockTokens)};
    const auto launch_slices = [&]<bool InPlace>(std::bool_constant<InPlace>, const auto& parameter) {
        for_each_token_slice(p.tokens, Schedule::kBlockTokens, [&](int offset, int count) {
            const int blocks  = p.rows / Schedule::kBlockRows * div_up(count, Schedule::kBlockTokens);
            const auto launch = [&]<bool Full>() {
                constexpr auto kernel =
                    fp8_a8_pair_block32_tma_mma_kernel<Schedule, Full, BlockScaledWeights, Output, InPlace>;
                constexpr int bytes =
                    fp8_tma_scratch_bytes<Schedule, LinearIdentityEpilogue> + Schedule::kBarrierBytes;
                const int dynamic = fp8_prepare_shared<bytes, kernel, true>();
                kernel<<<blocks, Schedule::kThreads, dynamic, stream>>>(parameter, p, output, offset,
                                                                        count);
                CUDA_CHECK(cudaGetLastError());
            };
            if (count % Schedule::kBlockTokens == 0)
                launch.template operator()<true>();
            else
                launch.template operator()<false>();
        });
    };
#ifdef _WIN32
    // Staged once: every token-slice launch reads the same copy, in stream order.
    with_tma_descriptor_parameter(fp8_tma_pair_block32_descriptor_staging(), descriptors, stream,
                                  launch_slices);
#else
    launch_slices(std::false_type{}, descriptors);
#endif
}

// Both terms of x, then the fused GEMM into `output` (64-column tiles up to kFp8A8PairNarrowTokens).
template <class Geometry, bool BlockScaledWeights, class Output>
void fp8_a8_pair_block32_gemm(const Tensor& x, const Weight& w, Output output, Fp8A8PairBlock32Workspace scratch,
                              cudaStream_t stream) {
    launch_fp8_a8_pair_block32_quantize(x, w, scratch, stream);
    const Fp8A8PairBlock32Operands operands{
        scratch.codes_hi,
        scratch.codes_lo,
        scratch.block_scales_hi,
        scratch.block_scales_lo,
        static_cast<const std::uint8_t*>(w.qdata),
        BlockScaledWeights ? nullptr : static_cast<const __nv_bfloat16*>(w.scales),
        BlockScaledWeights ? static_cast<const std::uint8_t*>(w.scales) : nullptr,
        w.n,
        w.k,
        x.ne[1]};
    if (x.ne[1] <= kFp8A8PairNarrowTokens) {
        launch_fp8_a8_pair_block32_tma_mma<Fp8ScheduleInstance<Fp8A8PairBlock32T64, Geometry::kInputRows>,
                                           BlockScaledWeights>(operands, output, stream);
        return;
    }
    launch_fp8_a8_pair_block32_tma_mma<Fp8ScheduleInstance<Fp8A8PairBlock32T128, Geometry::kInputRows>,
                                       BlockScaledWeights>(operands, output, stream);
}

// The complete block32 pair route of one registered row-FP8 problem.
template <class Geometry>
void launch_fp8_a8_pair_block32(const Tensor& x, const Weight& w, Tensor& out,
                                Fp8A8PairBlock32Workspace scratch, cudaStream_t stream) {
    require_fp8_scale_word<__nv_bfloat16>(w);
    fp8_a8_pair_block32_gemm<Geometry, false>(x, w, LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n},
                                              scratch, stream);
}

} // namespace infernix::ops::detail
