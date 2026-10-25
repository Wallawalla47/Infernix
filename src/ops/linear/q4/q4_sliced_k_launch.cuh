#pragma once
#include "core/device.h"
#include "core/pdl.cuh"
#include "ops/common/math.h"
#include "ops/common/token_slices.h"
#include "ops/linear/common/epilogue.cuh"
#include "ops/linear/common/sliced_k_pipelined.cuh"
#include "ops/linear/q4/q4_operands.h"
#include "ops/linear/q4/q4_a16_sliced_k_mma.cuh"

namespace infernix::ops::detail {

// The schedule's pipelined form (sliced_k_pipelined.cuh): the same K warps, reduction, token tile
// and static K, at the schedule's block rows or twice as many.
template <class Schedule>
using Q4SlicedKPipelinedFamily =
    SlicedKPipelinedFamily<SlicedKFormat::Q4, Schedule::kBlockRows, Schedule::kBlockTokens,
                           Schedule::kTokenCapacity, Schedule::kKWarps,
                           Schedule::kReduction == Q4SlicedKReduction::Pairwise, Schedule::kStaticK,
                           Schedule::kWeightCache, Schedule::kActivationCache>;

template <class Schedule>
constexpr bool q4_sliced_k_pipelines() {
    if constexpr (Schedule::kStaticK % Schedule::kBlockK != 0) {
        return false;
    } else {
        return Q4SlicedKPipelinedFamily<Schedule>::template fits<Schedule::kBlockRows>();
    }
}

template <class Schedule, class Output, class Epilogue,
          class RowPolicy = Q4IdentityRows<Schedule::kBlockRows>>
void launch_q4_a16_sliced_k_mma(const Q4LinearOperands& operands, Output output, Epilogue epilogue,
                                cudaStream_t stream, RowPolicy row_policy = {},
                                pdl::Dependency dependency = pdl::Dependency::Serialized) {
    validate_q4_operands(operands);
    if constexpr (Schedule::kStaticK > 0) {
        if (operands.k != Schedule::kStaticK || operands.padded_k != Schedule::kStaticK)
            throw std::invalid_argument("Q4 sliced-K static geometry requires K == padded K");
    }
    const int output_rows = row_policy.output_rows(operands.rows);
    const bool full = operands.rows % Schedule::kBlockRows == 0 &&
                      output_rows % RowPolicy::kOutputRowsPerCta == 0 &&
                      operands.k == operands.padded_k && operands.k % Schedule::kBlockK == 0 &&
                      reinterpret_cast<std::uintptr_t>(operands.scales) %
                              (Schedule::kKWarps * sizeof(__half)) ==
                          0;
    // The temporary A/B route (sliced_k_route.h): the pipelined kernel returns these bits.
    if constexpr (q4_sliced_k_pipelines<Schedule>()) {
        const SlicedKRoute route = groupwise_sliced_k_route();
        if (route != SlicedKRoute::Release && full) {
            launch_sliced_k_pipelined<Q4SlicedKPipelinedFamily<Schedule>>(
                SlicedKOperands{operands.x, operands.codes, nullptr, operands.scales, operands.rows,
                                operands.k, operands.tokens, operands.padded_k},
                output, epilogue, stream, row_policy, dependency, route);
            return;
        }
    }
    for_each_token_slice(operands.tokens, Schedule::kTokenCapacity, [&](int offset, int count) {
        const dim3 grid(div_up(output_rows, RowPolicy::kOutputRowsPerCta),
                        div_up(count, Schedule::kTokenCapacity));
        const auto* x          = operands.x + static_cast<std::int64_t>(offset) * operands.k;
        const bool full_tokens = count % Schedule::kTokenCapacity == 0;
        const auto launch      = [&]<bool Full, bool FullTokens>() {
            CUDA_CHECK(pdl::launch_with(
                dependency,
                {grid, dim3(Schedule::kThreads), 0, stream},
                q4_a16_sliced_k_mma_kernel<Schedule, Full, FullTokens, Output, Epilogue, RowPolicy>,
                x, operands.codes, operands.scales, output, epilogue, operands.rows, operands.k,
                count, operands.padded_k, offset, row_policy));
        };
        if (full && full_tokens)
            launch.template operator()<true, true>();
        else if (full)
            launch.template operator()<true, false>();
        else if (full_tokens)
            launch.template operator()<false, true>();
        else
            launch.template operator()<false, false>();
        CUDA_CHECK(cudaGetLastError());
    });
}
} // namespace infernix::ops::detail
