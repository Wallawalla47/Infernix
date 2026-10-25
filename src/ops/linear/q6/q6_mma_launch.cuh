#pragma once

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/common/token_slices.h"
#include "ops/linear/common/epilogue.cuh"
#include "ops/linear/q6/q6_operands.h"
#include "ops/linear/q6/q6_a16_mma.cuh"
#include "core/temporary_switches.h"

namespace infernix::ops::detail {

template <class Schedule, class Output, class Epilogue = LinearIdentityEpilogue>
void launch_q6_a16_mma(const Q6LinearOperands& operands, Output output, Epilogue epilogue,
                       cudaStream_t stream) {
    validate_q6_operands(operands);
    if (operands.padded_k % Schedule::kBlockK != 0) {
        throw std::invalid_argument("Q6 MMA padded K must contain complete K tiles");
    }
    for_each_token_slice(operands.tokens, Schedule::kBlockTokens, [&](int offset, int count) {
        const dim3 grid(static_cast<unsigned>(div_up(operands.rows, Schedule::kBlockRows)),
                        static_cast<unsigned>(div_up(count, Schedule::kBlockTokens)));
        const auto* x   = operands.x + static_cast<std::int64_t>(offset) * operands.k;
        const bool full = operands.rows % Schedule::kBlockRows == 0 &&
                          count % Schedule::kBlockTokens == 0 && operands.k == operands.padded_k &&
                          operands.k % Schedule::kBlockK == 0;
        const auto launch = [&]<bool Full, bool Fp32Scale>() {
            q6_a16_mma_kernel<Schedule, Full, Fp32Scale><<<grid, Schedule::kThreads, 0, stream>>>(
                x, operands.codes, operands.high, operands.scales, output, epilogue, operands.rows,
                operands.k, count, operands.padded_k, offset);
        };
        const bool fp32_scale = temporary_switches::q56_fp32_scale();
        if (full) {
            fp32_scale ? launch.template operator()<true, true>()
                       : launch.template operator()<true, false>();
        } else {
            fp32_scale ? launch.template operator()<false, true>()
                       : launch.template operator()<false, false>();
        }
        CUDA_CHECK(cudaGetLastError());
    });
}

} // namespace infernix::ops::detail
