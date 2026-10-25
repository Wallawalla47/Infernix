#pragma once
#include "core/device.h"
#include "ops/common/math.h"
#include "ops/common/token_slices.h"
#include "ops/linear/common/epilogue.cuh"
#include "ops/linear/common/sliced_k_pipelined.cuh"
#include "ops/linear/q5/q5_operands.h"
#include "ops/linear/q5/q5_a16_sliced_k_mma.cuh"

#include <stdexcept>

namespace infernix::ops::detail {

// The schedule's pipelined form (sliced_k_pipelined.cuh): the same K warps, reduction, token tile
// and static K, at the schedule's block rows or twice as many.
template <class Schedule>
using Q5SlicedKPipelinedFamily =
    SlicedKPipelinedFamily<SlicedKFormat::Q5, Schedule::kBlockRows, Schedule::kBlockTokens,
                           Schedule::kTokenCapacity, Schedule::kKWarps,
                           Schedule::kReduction == Q5SlicedKReduction::Pairwise, Schedule::kStaticK,
                           Schedule::kWeightCache, Schedule::kActivationCache>;

template <class Schedule>
constexpr bool q5_sliced_k_pipelines() {
    if constexpr (Schedule::kStaticK % Schedule::kBlockK != 0) {
        return false;
    } else {
        return Q5SlicedKPipelinedFamily<Schedule>::template fits<Schedule::kBlockRows>();
    }
}

// Whether the release kernel's full-weight form serves the call (the pipelined kernel needs it).
template <class Schedule, class RowPolicy>
bool q5_sliced_k_full(const Q5LinearOperands& operands, const RowPolicy& row_policy) {
    const int output_rows = row_policy.output_rows(operands.rows);
    return operands.rows % Schedule::kBlockRows == 0 &&
           output_rows % RowPolicy::kOutputRowsPerCta == 0 && operands.k == operands.padded_k &&
           operands.k % Schedule::kBlockK == 0 &&
           reinterpret_cast<std::uintptr_t>(operands.scales) % (Schedule::kKWarps * sizeof(__half)) ==
               0;
}

inline SlicedKOperands q5_sliced_k_operands(const Q5LinearOperands& operands) {
    return {operands.x,    operands.codes, operands.high,     operands.scales,
            operands.rows, operands.k,     operands.tokens,   operands.padded_k};
}

// The pipelined kernel whatever the temporary route (the q5-ksplit-mma fast path's eight K warps,
// which no release kernel serves); the weights must be full.
template <class Schedule, class Output, class Epilogue,
          class RowPolicy = Q5IdentityRows<Schedule::kBlockRows>>
void launch_q5_a16_sliced_k_pipelined(const Q5LinearOperands& operands, Output output,
                                      Epilogue epilogue, cudaStream_t stream,
                                      RowPolicy row_policy = {}) {
    static_assert(q5_sliced_k_pipelines<Schedule>());
    validate_q5_operands(operands);
    if (!q5_sliced_k_full<Schedule>(operands, row_policy)) {
        throw std::invalid_argument("Q5 pipelined sliced-K requires full weights");
    }
    launch_sliced_k_pipelined<Q5SlicedKPipelinedFamily<Schedule>>(
        q5_sliced_k_operands(operands), output, epilogue, stream, row_policy,
        pdl::Dependency::Serialized, SlicedKRoute::OneWave);
}

template <class Schedule, class Output, class Epilogue,
          class RowPolicy = Q5IdentityRows<Schedule::kBlockRows>>
void launch_q5_a16_sliced_k_mma(const Q5LinearOperands& operands, Output output, Epilogue epilogue,
                                cudaStream_t stream, RowPolicy row_policy = {}) {
    validate_q5_operands(operands);
    if (operands.k % 8 != 0) throw std::invalid_argument("Q5 MMA requires K aligned to 8");
    if constexpr (Schedule::kStaticK > 0) {
        if (operands.k != Schedule::kStaticK || operands.padded_k != Schedule::kStaticK)
            throw std::invalid_argument("Q5 sliced-K static geometry requires K == padded K");
    }
    const bool full = q5_sliced_k_full<Schedule>(operands, row_policy);
    // The temporary A/B route (sliced_k_route.h): the pipelined kernel returns these bits.
    if constexpr (q5_sliced_k_pipelines<Schedule>()) {
        const SlicedKRoute route = groupwise_sliced_k_route();
        if (route != SlicedKRoute::Release && full) {
            launch_sliced_k_pipelined<Q5SlicedKPipelinedFamily<Schedule>>(
                q5_sliced_k_operands(operands), output, epilogue, stream, row_policy,
                pdl::Dependency::Serialized, route);
            return;
        }
    }
    for_each_token_slice(operands.tokens, Schedule::kTokenCapacity, [&](int offset, int count) {
        const int output_rows = row_policy.output_rows(operands.rows);
        const dim3 grid(div_up(output_rows, RowPolicy::kOutputRowsPerCta),
                        div_up(count, Schedule::kTokenCapacity));
        const auto* x          = operands.x + static_cast<std::int64_t>(offset) * operands.k;
        const bool full_tokens = count % Schedule::kTokenCapacity == 0;
        const auto launch      = [&]<bool Full, bool FullTokens>() {
            q5_a16_sliced_k_mma_kernel<Schedule, Full, FullTokens>
                <<<grid, Schedule::kThreads, 0, stream>>>(
                    x, operands.codes, operands.high, operands.scales, output, epilogue,
                    operands.rows, operands.k, count, operands.padded_k, offset, row_policy);
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
