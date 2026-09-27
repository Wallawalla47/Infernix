#pragma once

// Small-T row-scaled E4M3 weight x BF16 activation Tensor Core mainloop.
//
// A CTA owns sixteen output rows and splits K across compile-time-selected warps. Persistent E4M3
// codes are widened exactly to BF16 MMA operands; the represented BF16 row multiplier is applied
// once to the complete FP32 dot product. The public activation is never quantized.

#include "core/pdl.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/memory.cuh"
#include "ops/linear/fp8/fp8_a16_codec.cuh"
#include "ops/linear/fp8/fp8_schedule.cuh"
#include "ops/linear/common/epilogue.cuh"
#include "ops/linear/fp8/fp8_operands.h"
#include "ops/linear/fp8/fp8_shared.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {

template <class Schedule, class Output, class Epilogue, class RowPolicy>
__global__
__launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void fp8_a16_sliced_k_mma_kernel(
    Fp8A16Operands operands, Output output, Epilogue epilogue, RowPolicy row_policy,
    int token_offset) {
    // Streams its weights through the main loop: wait for the producer first, and let dependents
    // launch only once that loop is done.
    pdl::enter_streaming();
    const auto* __restrict__ x            = operands.x;
    const auto* __restrict__ weight_codes = operands.codes;
    const auto* __restrict__ row_scales   = operands.scales;
    const int kHidden                     = Schedule::kStaticK ? Schedule::kStaticK : operands.k;
    constexpr int ActiveTokens =
        Schedule::kTokenCapacity ? Schedule::kTokenCapacity : Schedule::kBlockTokens;
    constexpr bool MaskedColumns = !Schedule::kExactTokens;
    constexpr int kTileK         = Schedule::kTileKPerWarp;
    constexpr int kWarps         = Schedule::kKWarps;
    constexpr int kBlockRows     = Schedule::kBlockRows;
    constexpr int kRowTiles      = Schedule::kRowTiles;
    static_assert(kRowTiles == 1 || !RowPolicy::kPaired, "two row tiles store unpaired rows only");
    constexpr int kBlockK        = Schedule::kBlockK;
    const int kGroups            = kHidden / kBlockK;
    constexpr int kBlockTokens   = Schedule::kBlockTokens;
    constexpr int kTokenMmas     = kBlockTokens / 8;
    static_assert(ActiveTokens >= 1 && ActiveTokens <= kBlockTokens);
    static_assert((kWarps & 1) == 0);
    constexpr unsigned kMask = 0xffffffffU;

    union SharedStorage {
        struct {
            std::uint8_t codes[Schedule::kStages][kBlockRows][kBlockK];
            __nv_bfloat16 activations[Schedule::kStages][kWarps][kBlockTokens * kTileK];
        } staging;

        float partial[kWarps * kRowTiles * kTokenMmas * 32 * 4];
    };

    static_assert(sizeof(SharedStorage) == Schedule::kSharedBytes);
    auto& shared = *reinterpret_cast<SharedStorage*>(fp8_shared_storage<Schedule::kSharedBytes>());
    auto& code_shared = shared.staging.codes;
    auto& x_shared    = shared.staging.activations;

    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int gid  = lane >> 2;
    const int lid  = lane & 3;
    const int row0 = static_cast<int>(blockIdx.x) * (kBlockRows / (RowPolicy::kPaired ? 2 : 1));
    const int token_begin = token_offset + static_cast<int>(blockIdx.y) * ActiveTokens;
    const int live_columns =
        MaskedColumns ? min(ActiveTokens, operands.tokens - token_begin) : ActiveTokens;

    const auto stage_activation = [&](int stage, int group_k0) {
        constexpr auto kActivationCache = Schedule::kActivationCache;
        constexpr bool kPadded     = Schedule::kActivationStage == Fp8ActivationStage::PaddedZero;
        constexpr int kStageTokens = kPadded ? kBlockTokens : ActiveTokens;
        constexpr int kItems       = kStageTokens * (kTileK / 8);
        for (int item = lane; item < kItems; item += 32) {
            const int token = item / (kTileK / 8);
            const int k8    = item - token * (kTileK / 8);
            auto* destination =
                &x_shared[stage][warp][token * kTileK + fp8_a16_shared_col_64(token, k8 * 8)];
            if constexpr (!MaskedColumns && (!kPadded || ActiveTokens == kBlockTokens)) {
                cp_async<16, kActivationCache>(
                    destination, x + static_cast<std::int64_t>(token_begin + token) * kHidden +
                                     group_k0 + warp * kTileK + k8 * 8);
            } else {
                const int source_token = token < live_columns ? token : 0;
                cp_async_zfill<16, kActivationCache>(
                    destination,
                    x + static_cast<std::int64_t>(token_begin + source_token) * kHidden + group_k0 +
                        warp * kTileK + k8 * 8,
                    token < live_columns ? 16 : 0);
            }
        }
    };

    const auto stage_codes = [&](int stage, int group_k0) {
        constexpr auto kWeightCache = Schedule::kWeightCache;
#pragma unroll
        for (int row_item = 0; row_item < Schedule::kRowsPerLoaderWarp; ++row_item) {
            const int row = warp * Schedule::kRowsPerLoaderWarp + row_item;
            for (int chunk = lane; chunk < kBlockK / 16; chunk += 32) {
                const int swizzled_chunk = chunk ^ (row & 7);
                cp_async<16, kWeightCache>(
                    &code_shared[stage][row][swizzled_chunk * 16],
                    weight_codes +
                        static_cast<std::int64_t>(row_policy.weight_row(row0, row, operands.rows)) *
                            kHidden +
                        group_k0 + chunk * 16);
            }
        }
    };

    const int b_row                   = lane & 7;
    const int b_k_offset              = ((lane >> 3) & 1) << 3;
    const int warp_k0                 = warp * kTileK;
    float accumulators[kRowTiles][kTokenMmas][4] = {};

#pragma unroll
    for (int stage = 0; stage < Schedule::kStages; ++stage) {
        if (stage < kGroups) {
            stage_codes(stage, stage * kBlockK);
            stage_activation(stage, stage * kBlockK);
            cp_commit();
        }
    }
#pragma unroll
    for (int group_index = 0; group_index < kGroups; ++group_index) {
        const int stage = group_index % Schedule::kStages;
        if (group_index + Schedule::kStages - 1 < kGroups)
            cp_wait<Schedule::kStages - 1>();
        else
            cp_wait<0>();
        __syncthreads();
#pragma unroll
        for (int k_step = 0; k_step < kTileK / 16; ++k_step) {
            const int code_col        = k_step * 16 + lid * 2;
            const auto load_code_pair = [&](int row, int col) {
                const int chunk  = (warp_k0 + col) >> 4;
                const int offset = (chunk ^ (row & 7)) * 16 + (col & 15);
                return static_cast<unsigned>(
                    *reinterpret_cast<const std::uint16_t*>(&code_shared[stage][row][offset]));
            };
            unsigned a[kRowTiles][4];
#pragma unroll
            for (int tile = 0; tile < kRowTiles; ++tile) {
                const int top = tile * 16 + gid;
                a[tile][0]    = fp8_e4m3x2_to_bf16x2_bits(load_code_pair(top, code_col));
                a[tile][1]    = fp8_e4m3x2_to_bf16x2_bits(load_code_pair(top + 8, code_col));
                a[tile][2]    = fp8_e4m3x2_to_bf16x2_bits(load_code_pair(top, code_col + 8));
                a[tile][3]    = fp8_e4m3x2_to_bf16x2_bits(load_code_pair(top + 8, code_col + 8));
            }
#pragma unroll
            for (int token_mma = 0; token_mma < kTokenMmas; ++token_mma) {
                unsigned b0;
                unsigned b1;
                const int row = token_mma * 8 + b_row;
                ldmatrix_x2(
                    b0, b1,
                    smem_addr(
                        &x_shared[stage][warp][row * kTileK + fp8_a16_shared_col_64(
                                                                  row, k_step * 16 + b_k_offset)]));
#pragma unroll
                for (int tile = 0; tile < kRowTiles; ++tile) {
                    mma_bf16(accumulators[tile][token_mma][0], accumulators[tile][token_mma][1],
                             accumulators[tile][token_mma][2], accumulators[tile][token_mma][3],
                             a[tile][0], a[tile][1], a[tile][2], a[tile][3], b0, b1);
                }
            }
        }

        __syncthreads();
        const int next = group_index + Schedule::kStages;
        if (next < kGroups) {
            stage_codes(stage, next * kBlockK);
            stage_activation(stage, next * kBlockK);
            cp_commit();
        }
    }
    pdl::trigger_dependents();

    __syncthreads();
    auto* partial            = shared.partial;
    const auto partial_index = [&](int split, int tile, int token_mma) {
        return (((split * kRowTiles + tile) * kTokenMmas + token_mma) * 32 + lane) * 4;
    };
    if ((warp & 1) != 0) {
#pragma unroll
        for (int tile = 0; tile < kRowTiles; ++tile) {
#pragma unroll
            for (int token_mma = 0; token_mma < kTokenMmas; ++token_mma) {
                const auto& acc = accumulators[tile][token_mma];
                store_vec(partial + partial_index(warp, tile, token_mma),
                          make_float4(acc[0], acc[1], acc[2], acc[3]));
            }
        }
    }
    __syncthreads();

    if ((warp & 1) == 0) {
#pragma unroll
        for (int tile = 0; tile < kRowTiles; ++tile) {
#pragma unroll
            for (int token_mma = 0; token_mma < kTokenMmas; ++token_mma) {
                auto& acc = accumulators[tile][token_mma];
                const float4 partner =
                    load_vec<float4>(partial + partial_index(warp + 1, tile, token_mma));
                acc[0] += partner.x;
                acc[1] += partner.y;
                acc[2] += partner.z;
                acc[3] += partner.w;
                if (warp != 0) {
                    store_vec(partial + partial_index(warp, tile, token_mma),
                              make_float4(acc[0], acc[1], acc[2], acc[3]));
                }
            }
        }
    }
    __syncthreads();

    if (warp == 0) {
        const auto destination =
            linear_output_tile<kBlockRows / (RowPolicy::kPaired ? 2 : 1)>(output, row0);
#pragma unroll
        for (int tile = 0; tile < kRowTiles; ++tile) {
            const int tile_row  = tile * 16;
            unsigned lane_scale = 0;
            if (lid < 2) {
                lane_scale = static_cast<unsigned>(
                    reinterpret_cast<const std::uint16_t*>(row_scales)[row_policy.weight_row(
                        row0, tile_row + gid + lid * 8, operands.rows)]);
            }
            const unsigned top_scale_bits    = __shfl_sync(kMask, lane_scale, lane & ~3);
            const unsigned bottom_scale_bits = __shfl_sync(kMask, lane_scale, (lane & ~3) + 1);
            const float top_scale =
                __bfloat162float(__ushort_as_bfloat16(static_cast<std::uint16_t>(top_scale_bits)));
            const float bottom_scale = __bfloat162float(
                __ushort_as_bfloat16(static_cast<std::uint16_t>(bottom_scale_bits)));

#pragma unroll
            for (int token_mma = 0; token_mma < kTokenMmas; ++token_mma) {
                const auto& acc = accumulators[tile][token_mma];
                float4 sum      = make_float4(acc[0], acc[1], acc[2], acc[3]);
#pragma unroll
                for (int split = 2; split < kWarps; split += 2) {
                    const float4 value =
                        load_vec<float4>(partial + partial_index(split, tile, token_mma));
                    sum.x += value.x;
                    sum.y += value.y;
                    sum.z += value.z;
                    sum.w += value.w;
                }
                const int local_token = token_mma * 8 + 2 * lid;
                const int token0      = token_begin + local_token;
                const int row_a = row_policy.weight_row(row0, tile_row + gid, operands.rows);
                const int row_b = row_policy.weight_row(row0, tile_row + gid + 8, operands.rows);
                if constexpr (RowPolicy::kPaired) {
                    if (local_token < live_columns)
                        epilogue.apply_pair(destination, row_a, token0, sum.x * top_scale,
                                            sum.z * bottom_scale);
                    if (local_token + 1 < live_columns)
                        epilogue.apply_pair(destination, row_a, token0 + 1, sum.y * top_scale,
                                            sum.w * bottom_scale);
                } else {
                    if (local_token < live_columns) {
                        destination.store(row_a, token0,
                                          epilogue.apply(row_a, token0, sum.x * top_scale));
                        destination.store(row_b, token0,
                                          epilogue.apply(row_b, token0, sum.z * bottom_scale));
                    }
                    if (local_token + 1 < live_columns) {
                        destination.store(row_a, token0 + 1,
                                          epilogue.apply(row_a, token0 + 1, sum.y * top_scale));
                        destination.store(row_b, token0 + 1,
                                          epilogue.apply(row_b, token0 + 1, sum.w * bottom_scale));
                    }
                }
            }
        }
    }
}

} // namespace ninfer::ops::detail
