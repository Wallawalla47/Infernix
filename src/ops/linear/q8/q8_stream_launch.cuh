#pragma once
#include "core/device.h"
#include "ops/common/math.h"
#include "ops/common/token_slices.h"
#include "ops/linear/q8/q8_a16_stream.cuh"
#include "ops/linear/q8/q8_operands.h"

#include <stdexcept>

namespace ninfer::ops::detail {

// One launch per grid.y-sized token slice; each CTA covers up to kMaxCols columns.
template <class Schedule, class Output, class Epilogue>
void launch_q8_a16_stream(const Q8LinearOperands& operands, Output output, Epilogue epilogue,
                          cudaStream_t stream) {
    validate_q8_operands(operands);
    if (operands.k != Schedule::kK)
        throw std::invalid_argument("Q8 register-streamed route requires its static K");
    for_each_token_slice(operands.tokens, Schedule::kMaxCols, [&](int offset, int count) {
        const dim3 grid(div_up(operands.rows, Schedule::kBlockRows), div_up(count, Schedule::kMaxCols));
        const auto* x = operands.x + static_cast<std::int64_t>(offset) * operands.k;
        const auto launch = [&]<bool FullRows>() {
            constexpr auto kernel = q8_a16_stream_kernel<Schedule, FullRows, Output, Epilogue>;
            const int shared      = q8_prepare_shared<Schedule::kSharedBytes, kernel>();
            kernel<<<grid, Schedule::kThreads, static_cast<std::size_t>(shared), stream>>>(
                x, operands.codes, operands.scales, output, epilogue, operands.rows, operands.padded_k, count,
                offset);
            CUDA_CHECK(cudaGetLastError());
        };
        if (operands.rows % Schedule::kBlockRows == 0)
            launch.template operator()<true>();
        else
            launch.template operator()<false>();
    });
}

template <class Schedule>
void launch_q8_a16_stream_linear(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    launch_q8_a16_stream<Schedule>(q8_linear_operands(x, weight),
                                   LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), out.ne[0]},
                                   LinearIdentityEpilogue{}, stream);
}

} // namespace ninfer::ops::detail
