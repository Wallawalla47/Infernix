#pragma once
#include "core/device.h"
#include "core/pdl.cuh"
#include "ops/common/math.h"
#include "ops/common/token_slices.h"
#include "ops/linear/q8/q8_a16_sliced_k_mma.cuh"

namespace infernix::ops::detail {
template <class Schedule, class RowPolicy = Q8SlicedKIdentityRows, class Output, class Epilogue>
void launch_q8_a16_sliced_k_mma(const Q8LinearOperands& operands, Output output, Epilogue epilogue,
                                cudaStream_t stream, RowPolicy row_policy = {},
                                pdl::Dependency dependency = pdl::Dependency::Serialized) {
    validate_q8_operands(operands);
    if (operands.k % 8 != 0) throw std::invalid_argument("Q8 sliced-K MMA requires K aligned to 8");
    if constexpr (Schedule::kStaticK > 0) {
        if (operands.k != Schedule::kStaticK || operands.padded_k != Schedule::kStaticK)
            throw std::invalid_argument(
                "Q8 sliced-K static K requires matching, padding-free operands");
    }
    if constexpr (Schedule::kExactTokens) {
        if (operands.tokens != Schedule::kTokenCapacity)
            throw std::invalid_argument("Q8 sliced-K exact tokens differ from the schedule");
    }
    // Row tiles cover consecutive identity rows; a paired row policy maps each CTA's weight rows
    // onto fewer output rows.
    constexpr int ratio = Schedule::kRowTiles > 1 ? 1 : Schedule::kBlockRows / RowPolicy::kOutputRowsPerCta;
    constexpr int cta_rows = Schedule::kRowTiles > 1 ? Schedule::kBlockRows : RowPolicy::kOutputRowsPerCta;
    static_assert(Schedule::kBlockRows % RowPolicy::kOutputRowsPerCta == 0);
    if (operands.rows % ratio != 0)
        throw std::invalid_argument("Q8 sliced-K row mapping requires complete row pairs");
    for_each_token_slice(operands.tokens, Schedule::kTokenCapacity, [&](int offset, int count) {
        const dim3 grid(div_up(operands.rows / ratio, cta_rows),
                        div_up(count, Schedule::kTokenCapacity));
        const auto launch = [&]<bool Full>() {
            constexpr auto kernel =
                q8_a16_sliced_k_mma_kernel<Schedule, Full, Output, Epilogue, RowPolicy>;
            const int shared = q8_prepare_shared<Schedule::kSharedBytes, kernel>();
            CUDA_CHECK(pdl::launch_with(
                dependency,
                {grid, dim3(Schedule::kThreads), static_cast<std::size_t>(shared), stream}, kernel,
                operands, output, epilogue, row_policy, offset));
        };
        if (operands.rows % Schedule::kBlockRows == 0 && operands.k == operands.padded_k &&
            operands.k % Schedule::kBlockK == 0)
            launch.template operator()<true>();
        else
            launch.template operator()<false>();
    });
}
} // namespace infernix::ops::detail
