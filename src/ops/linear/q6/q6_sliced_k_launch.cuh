#pragma once

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/common/token_slices.h"
#include "ops/linear/common/epilogue.cuh"
#include "ops/linear/common/sliced_k_pipelined.cuh"
#include "ops/linear/q6/q6_operands.h"
#include "ops/linear/q6/q6_a16_sliced_k_mma.cuh"

namespace infernix::ops::detail {

// The schedule's pipelined form (sliced_k_pipelined.cuh): the same K warps and linear reduction,
// its token tile as the capacity, at the schedule's block rows or twice as many.
template <class Schedule>
using Q6SlicedKPipelinedFamily =
    SlicedKPipelinedFamily<SlicedKFormat::Q6, Schedule::kBlockRows, Schedule::kBlockTokens,
                           Schedule::kBlockTokens, Schedule::kKWarps, false, 0,
                           Schedule::kWeightCache, Schedule::kActivationCache>;

template <class Schedule, class Output, class Epilogue = LinearIdentityEpilogue>
void launch_q6_a16_sliced_k_mma(const Q6LinearOperands& operands, Output output, Epilogue epilogue,
                                cudaStream_t stream) {
    validate_q6_operands(operands);
    const bool full_weights = operands.rows % Schedule::kBlockRows == 0 &&
                              operands.k == operands.padded_k &&
                              operands.k % Schedule::kBlockK == 0 &&
                              reinterpret_cast<std::uintptr_t>(operands.scales) %
                                      (Schedule::kKWarps * sizeof(__half)) ==
                                  0;
    // The temporary A/B route (sliced_k_route.h): the pipelined kernel returns these bits.
    if constexpr (Q6SlicedKPipelinedFamily<Schedule>::template fits<Schedule::kBlockRows>()) {
        const SlicedKRoute route = groupwise_sliced_k_route();
        if (route != SlicedKRoute::Release && full_weights) {
            launch_sliced_k_pipelined<Q6SlicedKPipelinedFamily<Schedule>>(
                SlicedKOperands{operands.x, operands.codes, operands.high, operands.scales,
                                operands.rows, operands.k, operands.tokens, operands.padded_k},
                output, epilogue, stream, SlicedKIdentityRows<Schedule::kBlockRows>{},
                pdl::Dependency::Serialized, route);
            return;
        }
    }
    for_each_token_slice(operands.tokens, Schedule::kBlockTokens, [&](int offset, int count) {
        const dim3 grid(static_cast<unsigned>(div_up(operands.rows, Schedule::kBlockRows)),
                        static_cast<unsigned>(div_up(count, Schedule::kBlockTokens)));
        const auto* x = operands.x + static_cast<std::int64_t>(offset) * operands.k;
        // The full kernel copies all warp scales together; the masked kernel
        // copies four-byte scale pairs and supports the minimum input alignment.
        const bool full = full_weights && count % Schedule::kBlockTokens == 0;
        if (full) {
            q6_a16_sliced_k_mma_kernel<Schedule, true><<<grid, Schedule::kThreads, 0, stream>>>(
                x, operands.codes, operands.high, operands.scales, output, epilogue, operands.rows,
                operands.k, count, operands.padded_k, offset);
        } else {
            q6_a16_sliced_k_mma_kernel<Schedule, false><<<grid, Schedule::kThreads, 0, stream>>>(
                x, operands.codes, operands.high, operands.scales, output, epilogue, operands.rows,
                operands.k, count, operands.padded_k, offset);
        }
        CUDA_CHECK(cudaGetLastError());
    });
}

} // namespace infernix::ops::detail
