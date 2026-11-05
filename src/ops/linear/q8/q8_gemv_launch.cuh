#pragma once
#include "core/device.h"
#include "ops/common/math.h"
#include "ops/linear/q8/q8_a16_gemv.cuh"

namespace infernix::ops::detail {
// The decode GEMV runs as a programmatic dependent of the kernel before it. In Dense8's decode graph it followed the
// HC collapse, the norms and the gate as a serialized launch, 84 such edges a round (the PDL edge census): Dense8 plain
// decode 133.55 -> 134.08 tok/s (+0.4 %, ABBA x4, every pair ahead; struct, whose rounds do not run it, +0.1 %;
// 2026-11-05).
template <class Schedule, class Output, class Epilogue>
void launch_q8_a16_gemv(const Q8LinearOperands& operands, Output output, Epilogue epilogue,
                        cudaStream_t stream) {
    validate_q8_operands(operands);
    if (operands.tokens != 1 || operands.k % 8 != 0)
        throw std::invalid_argument("Q8 GEMV requires T=1 and K aligned to 8");
    if constexpr (Schedule::kStaticK > 0) {
        if (operands.k != Schedule::kStaticK || operands.padded_k != Schedule::kStaticK)
            throw std::invalid_argument(
                "Q8 GEMV static K requires matching, padding-free operands");
    }
    const auto launch = [&]<bool FullRows>() {
        CUDA_CHECK(pdl::launch_consumer({dim3(div_up(operands.rows, Schedule::kBlockRows)), dim3(Schedule::kThreads),
                                         0, stream},
                                        q8_a16_gemv_kernel<Schedule, FullRows, Output, Epilogue>, operands, output,
                                        epilogue));
    };
    if (operands.rows % Schedule::kBlockRows == 0)
        launch.template operator()<true>();
    else
        launch.template operator()<false>();
}
} // namespace infernix::ops::detail
