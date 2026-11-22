#pragma once

// Reusable row-scaled FP8 CUDA-core mainloop for a compile-time number of BF16 activation
// columns. A CTA owns one row tile and one compile-time token tile, reusing each decoded weight
// pair across that token tile before advancing K. Output, epilogue, and row policies let fused
// consumers retain their observable semantics without duplicating the contraction.

#include "core/pdl.cuh"
#include "ops/linear/fp8/fp8_a16_gemv.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace infernix::ops::detail {

template <int Values>
struct Fp8ActivationPack {
    static_assert(Values == 8 || Values == 16 || Values == 32);
    std::uint32_t words[Values / 2];
};

template <int Values, bool Shared = false>
__device__ __forceinline__ Fp8ActivationPack<Values>
load_fp8_activation_pack(const __nv_bfloat16* pointer) {
    Fp8ActivationPack<Values> result;
#pragma unroll
    for (int chunk = 0; chunk < Values / 8; ++chunk) {
        const uint4 packed =
            Shared ? load_vec<uint4>(pointer + chunk * 8) : load_ldg<uint4>(pointer + chunk * 8);
        result.words[chunk * 4]     = packed.x;
        result.words[chunk * 4 + 1] = packed.y;
        result.words[chunk * 4 + 2] = packed.z;
        result.words[chunk * 4 + 3] = packed.w;
    }
    return result;
}

template <class Schedule>
struct Fp8SimtSharedStorage {
    static constexpr int kValuesPerPhase = 32 * Schedule::kValuesPerLane;
    static constexpr int kActivationElements =
        Schedule::kActivationAccess == Fp8SimtActivationAccess::SharedPhase
            ? Schedule::kBlockTokens * kValuesPerPhase
            : 8;
    alignas(16) __nv_bfloat16 activation[kActivationElements];
};

template <class Schedule, class Output, class Epilogue, class RowPolicy, class Scale = __nv_bfloat16>
__global__
__launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void fp8_a16_simt_kernel(
    Fp8A16OperandsT<Scale> operands, Output output, Epilogue epilogue, RowPolicy row_policy) {
    // Streams its weights through the main loop: wait for the producer first, and let dependents
    // launch only once that loop is done.
    pdl::enter_streaming();
    constexpr bool PairRows               = RowPolicy::kPaired;
    const auto* __restrict__ x            = operands.x;
    const auto* __restrict__ weight_codes = operands.codes;
    const auto* __restrict__ row_scales   = operands.scales;
    const int K                           = Schedule::kStaticK ? Schedule::kStaticK : operands.k;
    const int ActiveTokens = Schedule::kTokenCapacity ? Schedule::kTokenCapacity : operands.tokens;
    const int live_tokens  = Schedule::kExactTokens ? Schedule::kTokenCapacity : operands.tokens;
    static_assert(!PairRows || Schedule::kRowsPerWarp % 2 == 0);
    constexpr int kValuesPerPhase = 32 * Schedule::kValuesPerLane;
    const int kPhases             = K / kValuesPerPhase;
    constexpr int kStoredRowsPerWarp =
        PairRows ? Schedule::kRowsPerWarp / 2 : Schedule::kRowsPerWarp;
    constexpr int kStoredRowsPerCta = Schedule::kWarpsPerCta * kStoredRowsPerWarp;
    const int kRowBlocks            = operands.rows / Schedule::kBlockRows;
    const int kTokenTiles = (ActiveTokens + Schedule::kBlockTokens - 1) / Schedule::kBlockTokens;

    const int linear_block = static_cast<int>(blockIdx.x);
    int row_block;
    int token_tile;
    if (kTokenTiles == 1) {
        row_block  = linear_block;
        token_tile = 0;
    } else if constexpr (Schedule::kBlockOrder == Fp8SimtBlockOrder::RowsContiguous) {
        token_tile = linear_block / kRowBlocks;
        row_block  = linear_block - token_tile * kRowBlocks;
    } else {
        row_block  = linear_block / kTokenTiles;
        token_tile = linear_block - row_block * kTokenTiles;
    }
    const int token0 = kTokenTiles == 1 ? 0 : token_tile * Schedule::kBlockTokens;

    __shared__ Fp8SimtSharedStorage<Schedule> shared;
    const int lane      = static_cast<int>(threadIdx.x) & 31;
    const int warp      = static_cast<int>(threadIdx.x) >> 5;
    const int row_begin = row_block * kStoredRowsPerCta + warp * kStoredRowsPerWarp;
    float accumulators[Schedule::kRowsPerWarp][Schedule::kBlockTokens]
                      [Schedule::kAccumulatorChains] = {};
    using LoraStage = LoraWarpStage<Schedule::kRowsPerWarp, Schedule::kBlockTokens, kLoraStageRank<Epilogue>>;
    [[maybe_unused]] LoraStage* lora_stage = nullptr;
    if constexpr (kLoraEpilogue<Epilogue>) {
        // The term's operands (B rows, U columns) stage in shared memory beside the main loop.
        __shared__ LoraStage lora_stages[Schedule::kWarpsPerCta];
        lora_stage = &lora_stages[warp];
        int term_rows[Schedule::kRowsPerWarp];
#pragma unroll
        for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row)
            term_rows[local_row] = row_policy.weight_row(row_begin, local_row, operands.rows);
        lora_stage_warp_b(epilogue.lora, term_rows, *lora_stage);
        lora_stage_warp_u(epilogue.lora, term_rows, token0, live_tokens, *lora_stage);
    }

#pragma unroll Schedule::kPhaseUnroll
    for (int phase = 0; phase < kPhases; ++phase) {
        if constexpr (Schedule::kActivationAccess == Fp8SimtActivationAccess::SharedPhase) {
            constexpr int kPacksPerToken = kValuesPerPhase / 8;
            constexpr int kPacks         = Schedule::kBlockTokens * kPacksPerToken;
            auto* destination            = reinterpret_cast<uint4*>(shared.activation);
            for (int task = static_cast<int>(threadIdx.x); task < kPacks;
                 task += Schedule::kThreads) {
                const int local_token = task / kPacksPerToken;
                const int local_pack  = task - local_token * kPacksPerToken;
                const int token       = token0 + local_token;
                if (token < ActiveTokens) {
                    destination[task] = load_vec<uint4>(
                        x + static_cast<std::int64_t>(min(token, live_tokens - 1)) * K +
                        phase * kValuesPerPhase + local_pack * 8);
                }
            }
        }

        const int value_begin = phase * kValuesPerPhase + lane * Schedule::kValuesPerLane;
        Fp8CodePack<Schedule::kValuesPerLane> row_codes[Schedule::kRowsPerWarp];
#pragma unroll
        for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
            const int weight_row = row_policy.weight_row(row_begin, local_row, operands.rows);
            row_codes[local_row] = load_fp8_codes<Schedule::kCodeCache, Schedule::kValuesPerLane>(
                weight_codes + static_cast<std::int64_t>(weight_row) * K + value_begin);
        }

        Fp8ActivationPack<Schedule::kValuesPerLane> activation[Schedule::kBlockTokens];
        if constexpr (Schedule::kActivationAccess == Fp8SimtActivationAccess::SharedPhase) {
            __syncthreads();
#pragma unroll
            for (int local_token = 0; local_token < Schedule::kBlockTokens; ++local_token) {
                if (token0 + local_token < ActiveTokens) {
                    activation[local_token] =
                        load_fp8_activation_pack<Schedule::kValuesPerLane, true>(
                            shared.activation + local_token * kValuesPerPhase +
                            lane * Schedule::kValuesPerLane);
                }
            }
        } else {
#pragma unroll
            for (int local_token = 0; local_token < Schedule::kBlockTokens; ++local_token) {
                const int token = token0 + local_token;
                if (token < ActiveTokens) {
                    activation[local_token] = load_fp8_activation_pack<Schedule::kValuesPerLane>(
                        x + static_cast<std::int64_t>(min(token, live_tokens - 1)) * K +
                        value_begin);
                }
            }
        }

        constexpr int kChainMask = Schedule::kAccumulatorChains - 1;
#pragma unroll
        for (int pair = 0; pair < Schedule::kValuesPerLane / 2; ++pair) {
            float2 weights[Schedule::kRowsPerWarp];
#pragma unroll
            for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
                const std::uint32_t word   = row_codes[local_row].words[pair >> 1];
                const std::uint16_t packed = static_cast<std::uint16_t>(word >> ((pair & 1) * 16));
                weights[local_row]         = decode_fp8_e4m3x2(packed);
            }
#pragma unroll
            for (int local_token = 0; local_token < Schedule::kBlockTokens; ++local_token) {
                if (token0 + local_token >= ActiveTokens) { continue; }
                const float2 value = bf16x2_bits_to_float2(activation[local_token].words[pair]);
#pragma unroll
                for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
                    accumulators[local_row][local_token][(2 * pair) & kChainMask] =
                        fmaf(weights[local_row].x, value.x,
                             accumulators[local_row][local_token][(2 * pair) & kChainMask]);
                    accumulators[local_row][local_token][(2 * pair + 1) & kChainMask] =
                        fmaf(weights[local_row].y, value.y,
                             accumulators[local_row][local_token][(2 * pair + 1) & kChainMask]);
                }
            }
        }

        if constexpr (Schedule::kActivationAccess == Fp8SimtActivationAccess::SharedPhase) {
            __syncthreads();
        }
    }
    pdl::trigger_dependents();

    const auto destination =
        linear_output_tile<kStoredRowsPerCta>(output, row_block * kStoredRowsPerCta);
    if constexpr (kLoraEpilogue<Epilogue>) {
        // The rows' values as the no-adapter kernel forms them and the term (warp-wide): paired rows take it on the
        // gate and up values before the SwiGLU, single rows on the epilogue's result (lora_epilogue_apply).
        static_assert(Schedule::kBlockTokens <= 8, "LoRA term: at most 8 columns a SIMT tile");
        static_assert(!requires { Epilogue::kRowTokens; }, "LoRA term: no row-vector epilogue");
        // The warp sums and row scales; each value total * scale is formed where it is consumed, in the block of its
        // epilogue's add, as the no-adapter kernel forms it (the add may fuse the product into one FMA).
        int parent[Schedule::kRowsPerWarp];
        float scales[Schedule::kRowsPerWarp];
        float totals[Schedule::kRowsPerWarp][Schedule::kBlockTokens];
#pragma unroll
        for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
            parent[local_row] = row_policy.weight_row(row_begin, local_row, operands.rows);
            scales[local_row] = fp8_widen(__ldg(row_scales + parent[local_row]));
#pragma unroll
            for (int local_token = 0; local_token < Schedule::kBlockTokens; ++local_token) {
                float total = 0.0F;
#pragma unroll
                for (int chain = 0; chain < Schedule::kAccumulatorChains; ++chain) {
                    total += accumulators[local_row][local_token][chain];
                }
                totals[local_row][local_token] = warp_reduce_sum(total);
            }
        }
        float add[Schedule::kRowsPerWarp][Schedule::kBlockTokens];
        bool has[Schedule::kRowsPerWarp];
        lora_warp_rows_staged<Schedule::kRowsPerWarp, Schedule::kBlockTokens>(epilogue.lora, parent, token0,
                                                                             live_tokens, add, has, *lora_stage);
        if (lane == 0) {
#pragma unroll
            for (int local_token = 0; local_token < Schedule::kBlockTokens; ++local_token) {
                const int token = token0 + local_token;
                if (token >= live_tokens) { continue; }
                if constexpr (PairRows) {
#pragma unroll
                    for (int local_row = 0; local_row < kStoredRowsPerWarp; ++local_row) {
                        const int second = kStoredRowsPerWarp + local_row;
                        const float first  = totals[local_row][local_token] * scales[local_row];
                        const float second_value = totals[second][local_token] * scales[second];
                        epilogue.apply_pair(destination, row_begin + local_row, token,
                                            lora_add(first, add[local_row][local_token], has[local_row]),
                                            lora_add(second_value, add[second][local_token], has[second]));
                    }
                } else {
#pragma unroll
                    for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
                        destination.store(parent[local_row], token,
                                          lora_epilogue_apply(epilogue, parent[local_row], token,
                                                              totals[local_row][local_token] * scales[local_row],
                                                              add[local_row][local_token], has[local_row]));
                    }
                }
            }
        }
        return;
    }
    if constexpr (requires { Epilogue::kRowTokens; }) {
        static_assert(!PairRows, "row-vector finalization does not pair output rows");
        static_assert(Schedule::kBlockTokens == Epilogue::kRowTokens,
                      "row-vector finalization requires one CTA to own the full token row");
#pragma unroll
        for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
            const int parent_row = row_policy.weight_row(row_begin, local_row, operands.rows);
            const float scale    = fp8_widen(__ldg(row_scales + parent_row));
            float projected[Schedule::kBlockTokens];
#pragma unroll
            for (int local_token = 0; local_token < live_tokens; ++local_token) {
                float total = 0.0F;
#pragma unroll
                for (int chain = 0; chain < Schedule::kAccumulatorChains; ++chain) {
                    total += accumulators[local_row][local_token][chain];
                }
                total = warp_reduce_sum(total);
                if (lane == 0) { projected[local_token] = total * scale; }
            }
            if (lane == 0) {
                linear_finish_row(destination, epilogue, parent_row, 0, projected, live_tokens);
            }
        }
    } else if constexpr (PairRows) {
#pragma unroll
        for (int local_token = 0; local_token < Schedule::kBlockTokens; ++local_token) {
            const int token = token0 + local_token;
            if (token >= live_tokens) { continue; }
            float totals[Schedule::kRowsPerWarp];
#pragma unroll
            for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
                float total = 0.0F;
#pragma unroll
                for (int chain = 0; chain < Schedule::kAccumulatorChains; ++chain) {
                    total += accumulators[local_row][local_token][chain];
                }
                totals[local_row] = warp_reduce_sum(total);
            }
            if (lane == 0) {
#pragma unroll
                for (int local_row = 0; local_row < kStoredRowsPerWarp; ++local_row) {
                    const int first_row =
                        row_policy.weight_row(row_begin, local_row, operands.rows);
                    const int second_row = row_policy.weight_row(
                        row_begin, kStoredRowsPerWarp + local_row, operands.rows);
                    const float first =
                        totals[local_row] * fp8_widen(__ldg(row_scales + first_row));
                    const float second = totals[kStoredRowsPerWarp + local_row] *
                                         fp8_widen(__ldg(row_scales + second_row));
                    epilogue.apply_pair(destination, row_begin + local_row, token, first, second);
                }
            }
        }
    } else {
#pragma unroll
        for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
            const int parent_row = row_policy.weight_row(row_begin, local_row, operands.rows);
            const float scale    = fp8_widen(__ldg(row_scales + parent_row));
#pragma unroll
            for (int local_token = 0; local_token < Schedule::kBlockTokens; ++local_token) {
                const int token = token0 + local_token;
                if (token >= live_tokens) { continue; }
                float total = 0.0F;
#pragma unroll
                for (int chain = 0; chain < Schedule::kAccumulatorChains; ++chain) {
                    total += accumulators[local_row][local_token][chain];
                }
                total = warp_reduce_sum(total);
                if (lane == 0) {
                    destination.store(parent_row, token,
                                      epilogue.apply(parent_row, token, total * scale));
                }
            }
        }
    }
}

} // namespace infernix::ops::detail
