#pragma once

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/linear/common/vector_output.cuh"
#include "ops/linear/fp8/fp8_operands.h"
#include "ops/linear/fp8/fp8_schedule.cuh"
#include "ops/lora/lora_device.cuh"

#include <type_traits>

namespace infernix::ops::detail {

// Epilogues may consume adjacent aligned rows together; the scalar contract remains valid.
template <class Epilogue>
__device__ __forceinline__ float2 fp8_apply_row_pair(Epilogue epilogue, int row, int next_row,
                                                     int token, float2 value) {
    if constexpr (requires { epilogue.apply_row_pair(row, token, value); }) {
        if (next_row == row + 1) return epilogue.apply_row_pair(row, token, value);
    }
    return make_float2(epilogue.apply(row, token, value.x),
                       epilogue.apply(next_row, token, value.y));
}

template <class Schedule>
__device__ __forceinline__ int fp8_mma_shared_byte(int row, int logical_byte) {
    if constexpr (Schedule::kTmaSwizzle) {
        // TMA swizzles bits from the byte address, including the row's K stride.
        return logical_byte ^
               ((((row * Schedule::kBlockK + logical_byte) >> 7) & (Schedule::kSegmentsPerRow - 1))
                << 4);
    }
    const int logical_segment  = logical_byte >> 4;
    const int byte_in_segment  = logical_byte & 15;
    const int physical_segment = logical_segment ^ (row & (Schedule::kSegmentsPerRow - 1));
    return physical_segment * 16 + byte_in_segment;
}

template <class Schedule>
__device__ __forceinline__ void
fp8_mma_tile_coordinates(std::int32_t linear, std::int32_t row_tiles, std::int32_t token_tiles,
                         std::int32_t& row_tile, std::int32_t& token_tile) {
    if constexpr (Schedule::kRaster == Fp8MmaRaster::RowFast) {
        token_tile = linear / row_tiles;
        row_tile   = linear - token_tile * row_tiles;
    } else if constexpr (Schedule::kRaster == Fp8MmaRaster::TokenFast) {
        row_tile   = linear / token_tiles;
        token_tile = linear - row_tile * token_tiles;
    } else {
        constexpr int group_rows       = Schedule::kRasterGroupRows;
        const std::int32_t group_span  = group_rows * token_tiles;
        const std::int32_t group       = linear / group_span;
        const std::int32_t first_row   = group * group_rows;
        const std::int32_t active_rows = min(group_rows, row_tiles - first_row);
        const std::int32_t within      = linear - group * group_span;
        row_tile                       = first_row + within % active_rows;
        token_tile                     = within / active_rows;
    }
}

template <class Schedule>
__device__ __forceinline__ void
fp8_mma_compute_stage(const std::uint8_t* activation_stage, const std::uint8_t* weight_stage,
                      float (&accumulators)[Schedule::kMmaTokens][Schedule::kMmaRows][4], int warp,
                      int lane) {
    constexpr int BK        = Schedule::kBlockK;
    const int warp_token    = warp / Schedule::kWarpsRows;
    const int warp_row      = warp % Schedule::kWarpsRows;
    const int a_matrix      = lane >> 3;
    const int a_row_offset  = (lane & 7) + ((a_matrix & 1) << 3);
    const int a_column_byte = (a_matrix >> 1) * 16;
    const int b_row_offset  = lane & 7;
    const int b_column_byte = ((lane >> 3) & 1) * 16;
    auto load_fragments     = [&](int k_step, unsigned(&a_fragments)[Schedule::kMmaTokens][4],
                              unsigned(&b_fragments)[Schedule::kMmaRows][2]) {
#pragma unroll
        for (int mma_token = 0; mma_token < Schedule::kMmaTokens; ++mma_token) {
            const int row = warp_token * Schedule::kWarpTokens + mma_token * 16 + a_row_offset;
            const int logical_byte  = k_step * 32 + a_column_byte;
            const int physical_byte = fp8_mma_shared_byte<Schedule>(row, logical_byte);
            ldmatrix_x4(a_fragments[mma_token][0], a_fragments[mma_token][1],
                        a_fragments[mma_token][2], a_fragments[mma_token][3],
                        smem_addr(activation_stage + row * BK + physical_byte));
        }
#pragma unroll
        for (int mma_row = 0; mma_row < Schedule::kMmaRows; ++mma_row) {
            const int row           = warp_row * Schedule::kWarpRows + mma_row * 8 + b_row_offset;
            const int logical_byte  = k_step * 32 + b_column_byte;
            const int physical_byte = fp8_mma_shared_byte<Schedule>(row, logical_byte);
            ldmatrix_x2(b_fragments[mma_row][0], b_fragments[mma_row][1],
                        smem_addr(weight_stage + row * BK + physical_byte));
        }
    };

    if constexpr (Schedule::kFragmentPipeline == Fp8MmaFragmentPipeline::PingPong) {
        unsigned a_fragments[2][Schedule::kMmaTokens][4];
        unsigned b_fragments[2][Schedule::kMmaRows][2];
        load_fragments(0, a_fragments[0], b_fragments[0]);
#pragma unroll
        for (int k_step = 0; k_step < Schedule::kMmaK; ++k_step) {
            const int slot = k_step & 1;
            if (k_step + 1 < Schedule::kMmaK) {
                load_fragments(k_step + 1, a_fragments[slot ^ 1], b_fragments[slot ^ 1]);
            }
#pragma unroll
            for (int mma_token = 0; mma_token < Schedule::kMmaTokens; ++mma_token) {
#pragma unroll
                for (int mma_row = 0; mma_row < Schedule::kMmaRows; ++mma_row) {
                    mma_fp8_e4m3(
                        accumulators[mma_token][mma_row][0], accumulators[mma_token][mma_row][1],
                        accumulators[mma_token][mma_row][2], accumulators[mma_token][mma_row][3],
                        a_fragments[slot][mma_token][0], a_fragments[slot][mma_token][1],
                        a_fragments[slot][mma_token][2], a_fragments[slot][mma_token][3],
                        b_fragments[slot][mma_row][0], b_fragments[slot][mma_row][1]);
                }
            }
        }
    } else {
        unsigned a_fragments[Schedule::kMmaTokens][4];
        unsigned b_fragments[Schedule::kMmaRows][2];
#pragma unroll
        for (int k_step = 0; k_step < Schedule::kMmaK; ++k_step) {
            load_fragments(k_step, a_fragments, b_fragments);
#pragma unroll
            for (int mma_token = 0; mma_token < Schedule::kMmaTokens; ++mma_token) {
#pragma unroll
                for (int mma_row = 0; mma_row < Schedule::kMmaRows; ++mma_row) {
                    mma_fp8_e4m3(
                        accumulators[mma_token][mma_row][0], accumulators[mma_token][mma_row][1],
                        accumulators[mma_token][mma_row][2], accumulators[mma_token][mma_row][3],
                        a_fragments[mma_token][0], a_fragments[mma_token][1],
                        a_fragments[mma_token][2], a_fragments[mma_token][3],
                        b_fragments[mma_row][0], b_fragments[mma_row][1]);
                }
            }
        }
    }
}

// Weight rows' words are BF16 or FP32 (fp8_operands.h): Scale is the instance's word type.
// `activation_scales` is `nullptr` (A16 tiles: no per-token scales) or a validated non-null pointer, so
// the choice is the instance's and the A8 epilogue carries no unscaled copy.
template <class Schedule, bool FullTokens, class Epilogue, class Output, class RowPolicy,
          class Scale, class ActivationScales>
__device__ __forceinline__ void
fp8_finish_mma_tile(Output output, Epilogue epilogue, RowPolicy row_policy,
                    unsigned char* shared_raw,
                    float (&accumulators)[Schedule::kMmaTokens][Schedule::kMmaRows][4],
                    ActivationScales activation_scales, const Scale* weight_scales,
                    int row_begin, int token_begin, int rows, int tokens, int warp, int lane) {
    static_assert(std::is_null_pointer_v<ActivationScales> || std::is_same_v<ActivationScales, const float*>);
    constexpr bool kTokenScales = !std::is_null_pointer_v<ActivationScales>;
    constexpr bool PairRows = RowPolicy::kPaired;
    constexpr int BM = Schedule::kBlockTokens, BN = Schedule::kBlockRows;
    constexpr int producers = [] {
        if constexpr (requires { Schedule::kProducerThreads; })
            return Schedule::kProducerThreads;
        else
            return 0;
    }();
    constexpr int THREADS     = Schedule::kThreads - producers;
    const int tid             = warp * 32 + lane;
    const int warp_token      = warp / Schedule::kWarpsRows;
    const int warp_row        = warp % Schedule::kWarpsRows;
    constexpr bool collective = requires {
        epilogue.template finish_tile<Schedule, FullTokens>(output, shared_raw, accumulators,
                                                            row_begin, token_begin, rows, tokens);
    };
    static_assert(!PairRows || collective, "paired rows require a collective epilogue");
    const int accumulator_token = lane >> 2;
    const int accumulator_row   = 2 * (lane & 3);
    constexpr int output_stride = BN + 8;
    auto* shared_output         = reinterpret_cast<__nv_bfloat16*>(shared_raw);
    constexpr bool kLora = kLoraEpilogue<Epilogue>;
    // The term's operands staged in the tile's free shared memory, on the scaled values. A narrow term takes the
    // whole tile's U a pass (lora_add_tokens_tile_staged) when B's rows and one segment's U columns fit; a wide one
    // whose B rows and a 16-token U buffer a token group fit takes the group-staged form
    // (lora_add_tokens_tile_group_staged, low-register: the FP8 tiles' accumulators leave few registers). The others
    // read them from L2 a fragment at a time (the form below), which took the QAD 27B's q/k/v tile 24 % more cycles at
    // rank 32 (ncu, RTX 5090, 2026-11-22) against 19 % for the wide group-staged form at rank 128; the group-staged
    // form on a narrow term kept 442 local loads and stores where the whole-tile form keeps none of its own.
    constexpr int kScratchBytes = [] {
        if constexpr (requires { Schedule::kStorageBytes; })
            return Schedule::kStorageBytes;
        else
            return Schedule::kSharedBytes;
    }();
    // A wide term's whole-tile stage fits the 384-row FP8 tiles' 98,304 bytes only unpadded (lora_add_tokens_tile_
    // staged_swizzled, its row fragments one at a time: low-register); the group-staged form serves the others.
    constexpr bool kStagedLora = [] {
        if constexpr (kLora && requires { Schedule::kWarpsTokens; }) {
            if constexpr (kLoraStageRank<Epilogue> > kLoraNarrowRank)
                return (BN + BM) * (2 * kLoraStageRank<Epilogue>) <= kScratchBytes;
            return (BN + BM) * (2 * kLoraNarrowRank + 16) <= kScratchBytes;
        } else {
            return false;
        }
    }();
    constexpr bool kGroupLora = [] {
        if constexpr (kLora && !kStagedLora && requires { Schedule::kWarpsTokens; }) {
            if constexpr (kLoraStageRank<Epilogue> <= kLoraNarrowRank) { return false; }
            return (BN + Schedule::kWarpsTokens * 2 * 16) * (2 * kLoraStageRank<Epilogue> + 16) <= kScratchBytes;
        } else {
            return false;
        }
    }();
    if constexpr (kLora && !collective && !kGroupLora && !kStagedLora) {
        // --lora, a fragment-wise epilogue: per fragment column its term operands, then per fragment the scaled
        // values plus the term, finished and stored at once, so the accumulators retire as they go (the operations
        // of each element are the scaling pass's, lora_add_tokens_tile's and the epilogue's, as in the else branch).
        const int token_base = token_begin + warp_token * Schedule::kWarpTokens;
#pragma unroll
        for (int mma_row = 0; mma_row < Schedule::kMmaRows; ++mma_row) {
            const int column_row  = warp_row * Schedule::kWarpRows + mma_row * 8;
            const int local_row0  = column_row + accumulator_row;
            const int parent_row0 = row_policy.weight_row(row_begin, local_row0, rows);
            const int parent_row1 = row_policy.weight_row(row_begin, local_row0 + 1, rows);
            float2 weight_scale   = make_float2(1.0F, 1.0F);
            if constexpr (!requires { RowPolicy::kUnscaled; }) {
                weight_scale = make_float2(fp8_widen(weight_scales[parent_row0]),
                                           fp8_widen(weight_scales[parent_row1]));
            }
            const auto column = lora_tokens_column(
                epilogue.lora, row_policy.weight_row(row_begin, column_row, rows),
                row_policy.weight_row(row_begin, column_row + accumulator_token, rows));
#pragma unroll
            for (int mma_token = 0; mma_token < Schedule::kMmaTokens; ++mma_token) {
                const int token0 = token_base + mma_token * 16 + accumulator_token;
                const int token1 = token0 + 8;
                const auto activation_scale = [&](int token) {
                    if constexpr (kTokenScales) {
                        return (FullTokens || token < tokens) ? __ldg(activation_scales + token) : 0.0F;
                    } else {
                        return 1.0F;
                    }
                };
                const float activation_scale0 = activation_scale(token0);
                const float activation_scale1 = activation_scale(token1);
                const auto& acc = accumulators[mma_token][mma_row];
                float value00   = __fmul_rn(__fmul_rn(acc[0], activation_scale0), weight_scale.x);
                float value01   = __fmul_rn(__fmul_rn(acc[1], activation_scale0), weight_scale.y);
                float value10   = __fmul_rn(__fmul_rn(acc[2], activation_scale1), weight_scale.x);
                float value11   = __fmul_rn(__fmul_rn(acc[3], activation_scale1), weight_scale.y);
                if (column.uo >= 0) {
                    float c[4];
                    lora_tokens_fragment(epilogue.lora, column, token0 < tokens ? token0 : -1,
                                         token1 < tokens ? token1 : -1, c);
                    value00 = __fadd_rn(value00, __fmul_rn(c[0], column.scale));
                    value01 = __fadd_rn(value01, __fmul_rn(c[1], column.scale));
                    value10 = __fadd_rn(value10, __fmul_rn(c[2], column.scale));
                    value11 = __fadd_rn(value11, __fmul_rn(c[3], column.scale));
                }
                if (FullTokens || token0 < tokens) {
                    const float2 value = fp8_apply_row_pair(epilogue, parent_row0, parent_row1, token0,
                                                            make_float2(value00, value01));
                    value00            = value.x;
                    value01            = value.y;
                }
                if (FullTokens || token1 < tokens) {
                    const float2 value = fp8_apply_row_pair(epilogue, parent_row0, parent_row1, token1,
                                                            make_float2(value10, value11));
                    value10            = value.x;
                    value11            = value.y;
                }
                auto* destination0 = reinterpret_cast<__nv_bfloat162*>(
                    shared_output + (token0 - token_begin) * output_stride + local_row0);
                auto* destination1 = reinterpret_cast<__nv_bfloat162*>(
                    shared_output + (token1 - token_begin) * output_stride + local_row0);
                *destination0 = __floats2bfloat162_rn(value00, value01);
                *destination1 = __floats2bfloat162_rn(value10, value11);
            }
        }
    } else {
        // --lora with a collective epilogue: the loop below first runs as the scaling pass (the same rounded
        // multiplies), the term is added to the scaled values, and the loop then runs again with unit scales (exact)
        // to finish them.
        if constexpr (kLora) {
#pragma unroll
            for (int mma_token = 0; mma_token < Schedule::kMmaTokens; ++mma_token) {
                const int token0 =
                    token_begin + warp_token * Schedule::kWarpTokens + mma_token * 16 + accumulator_token;
                const int token1 = token0 + 8;
                const auto activation_scale = [&](int token) {
                    if constexpr (kTokenScales) {
                        return (FullTokens || token < tokens) ? __ldg(activation_scales + token) : 0.0F;
                    } else {
                        return 1.0F;
                    }
                };
                const float activation_scale0 = activation_scale(token0);
                const float activation_scale1 = activation_scale(token1);
#pragma unroll
                for (int mma_row = 0; mma_row < Schedule::kMmaRows; ++mma_row) {
                    const int local_row0  = warp_row * Schedule::kWarpRows + mma_row * 8 + accumulator_row;
                    const int parent_row0 = row_policy.weight_row(row_begin, local_row0, rows);
                    const int parent_row1 = row_policy.weight_row(row_begin, local_row0 + 1, rows);
                    float2 weight_scale   = make_float2(1.0F, 1.0F);
                    if constexpr (!requires { RowPolicy::kUnscaled; }) {
                        weight_scale = make_float2(fp8_widen(weight_scales[parent_row0]),
                                                   fp8_widen(weight_scales[parent_row1]));
                    }
                    auto& acc = accumulators[mma_token][mma_row];
                    acc[0]    = __fmul_rn(__fmul_rn(acc[0], activation_scale0), weight_scale.x);
                    acc[1]    = __fmul_rn(__fmul_rn(acc[1], activation_scale0), weight_scale.y);
                    acc[2]    = __fmul_rn(__fmul_rn(acc[2], activation_scale1), weight_scale.x);
                    acc[3]    = __fmul_rn(__fmul_rn(acc[3], activation_scale1), weight_scale.y);
                }
            }
            if constexpr (kGroupLora) {
                // Every warp has passed the main loop's last shared reads (the caller's barrier): the storage is free.
                lora_add_tokens_tile_group_staged<BN, kScratchBytes, Schedule::kWarpsTokens, Schedule::kWarpsRows, true>(
                    epilogue.lora, accumulators,
                    [&](int mt, int i) { return warp_token * Schedule::kWarpTokens + mt * 16 + i; },
                    [&](int mr, int j) { return warp_row * Schedule::kWarpRows + mr * 8 + j; },
                    [&](int i) { return row_policy.weight_row(row_begin, i, rows); }, token_begin, tokens, shared_raw,
                    tid, warp_token, [] { asm volatile("bar.sync 1, %0;" ::"r"(THREADS) : "memory"); });
            } else if constexpr (kStagedLora) {
                // The same: the storage is free.
                lora_add_tokens_tile_staged<BM, BN, kScratchBytes, true>(
                    epilogue.lora, accumulators,
                    [&](int mt, int i) { return warp_token * Schedule::kWarpTokens + mt * 16 + i; },
                    [&](int mr, int j) { return warp_row * Schedule::kWarpRows + mr * 8 + j; },
                    [&](int i) { return row_policy.weight_row(row_begin, i, rows); }, token_begin, tokens, shared_raw,
                    tid, THREADS, [] { asm volatile("bar.sync 1, %0;" ::"r"(THREADS) : "memory"); });
            } else {
                const int token_base = token_begin + warp_token * Schedule::kWarpTokens;
                lora_add_tokens_tile<Schedule::kMmaTokens, Schedule::kMmaRows>(
                    epilogue.lora, accumulators,
                    [&](int mt, int i) {
                        const int token = token_base + mt * 16 + i;
                        return token < tokens ? token : -1;
                    },
                    [&](int mr, int j) {
                        return row_policy.weight_row(row_begin, warp_row * Schedule::kWarpRows + mr * 8 + j, rows);
                    });
            }
        }
#pragma unroll
        for (int mma_token = 0; mma_token < Schedule::kMmaTokens; ++mma_token) {
            const int token0 =
                token_begin + warp_token * Schedule::kWarpTokens + mma_token * 16 + accumulator_token;
            const int token1 = token0 + 8;
            // A16 tiles carry unscaled activations and pass no per-token scales.
            const auto activation_scale = [&](int token) {
                if constexpr (kTokenScales && !kLora) {
                    return (FullTokens || token < tokens) ? __ldg(activation_scales + token) : 0.0F;
                } else {
                    return 1.0F;
                }
            };
            const float activation_scale0 = activation_scale(token0);
            const float activation_scale1 = activation_scale(token1);
#pragma unroll
            for (int mma_row = 0; mma_row < Schedule::kMmaRows; ++mma_row) {
                const int local_row0  = warp_row * Schedule::kWarpRows + mma_row * 8 + accumulator_row;
                const int parent_row0 = row_policy.weight_row(row_begin, local_row0, rows);
                const int parent_row1 = row_policy.weight_row(row_begin, local_row0 + 1, rows);
                const float2 weight_scale = [&] {
                    if constexpr (kLora || requires { RowPolicy::kUnscaled; }) {
                        return make_float2(1.0F, 1.0F);
                    } else {
                        if constexpr (requires { RowPolicy::kContiguousPairs; }) {
                            if constexpr (RowPolicy::kContiguousPairs) {
                                if constexpr (std::is_same_v<Scale, float>)
                                    return load_ldg<float2>(weight_scales + parent_row0);
                                else
                                    return bf16x2_bits_to_float2(
                                        load_ldg<std::uint32_t>(weight_scales + parent_row0));
                            }
                        }
                        return make_float2(fp8_widen(weight_scales[parent_row0]),
                                           fp8_widen(weight_scales[parent_row1]));
                    }
                }();
                // Scale with explicit rounded multiplies: a residual add after them must not contract
                // into one FMA in the FullTokens instance only, or a token's bits would depend on
                // whether its tile is full (upstream Neroued/ninfer#374). Every route that finishes
                // here, and the TMA split-K reduce, rounds the products, then the sum.
                float value00 = __fmul_rn(
                    __fmul_rn(accumulators[mma_token][mma_row][0], activation_scale0), weight_scale.x);
                float value01 = __fmul_rn(
                    __fmul_rn(accumulators[mma_token][mma_row][1], activation_scale0), weight_scale.y);
                float value10 = __fmul_rn(
                    __fmul_rn(accumulators[mma_token][mma_row][2], activation_scale1), weight_scale.x);
                float value11 = __fmul_rn(
                    __fmul_rn(accumulators[mma_token][mma_row][3], activation_scale1), weight_scale.y);
                if constexpr (collective) {
                    accumulators[mma_token][mma_row][0] = value00;
                    accumulators[mma_token][mma_row][1] = value01;
                    accumulators[mma_token][mma_row][2] = value10;
                    accumulators[mma_token][mma_row][3] = value11;
                } else {
                    if (FullTokens || token0 < tokens) {
                        const float2 value = fp8_apply_row_pair(epilogue, parent_row0, parent_row1,
                                                                token0, make_float2(value00, value01));
                        value00            = value.x;
                        value01            = value.y;
                    }
                    if (FullTokens || token1 < tokens) {
                        const float2 value = fp8_apply_row_pair(epilogue, parent_row0, parent_row1,
                                                                token1, make_float2(value10, value11));
                        value10            = value.x;
                        value11            = value.y;
                    }
                    auto* destination0 = reinterpret_cast<__nv_bfloat162*>(
                        shared_output + (token0 - token_begin) * output_stride + local_row0);
                    auto* destination1 = reinterpret_cast<__nv_bfloat162*>(
                        shared_output + (token1 - token_begin) * output_stride + local_row0);
                    *destination0 = __floats2bfloat162_rn(value00, value01);
                    *destination1 = __floats2bfloat162_rn(value10, value11);
                }
            }
        }
    }
    __syncthreads();

    if constexpr (collective) {
        epilogue.template finish_tile<Schedule, FullTokens>(output, shared_raw, accumulators,
                                                            row_begin, token_begin, rows, tokens);
    } else {
        constexpr int stored_rows       = PairRows ? BN / 2 : BN;
        constexpr int vectors_per_token = stored_rows / 8;
        constexpr int output_vectors    = BM * vectors_per_token;
        for (int task = tid; task < output_vectors; task += THREADS) {
            const int token_local = task / vectors_per_token;
            const int row_vector  = task - token_local * vectors_per_token;
            const int token       = token_begin + token_local;
            if constexpr (FullTokens) {
                const uint4 values =
                    load_vec<uint4>(shared_output + token_local * output_stride + row_vector * 8);
                linear_store_bf16_vector(output, row_begin + row_vector * 8, token, values);
            } else if (token < tokens) {
                const uint4 values =
                    load_vec<uint4>(shared_output + token_local * output_stride + row_vector * 8);
                linear_store_bf16_vector(output, row_begin + row_vector * 8, token, values);
            }
        }
    }
}

} // namespace infernix::ops::detail
