#pragma once

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/common/token_slices.h"
#include "ops/linear/fp8/fp8_a16_gemv.cuh"
#include "ops/linear/fp8/fp8_a16_simt.cuh"
#include "ops/linear/fp8/fp8_a16_mma.cuh"
#include "ops/linear/fp8/fp8_a16_sliced_k_mma.cuh"
#include "ops/linear/fp8/fp8_a8_mma.cuh"
#include "ops/linear/fp8/fp8_a8_tma_mma.cuh"

namespace infernix::ops::detail {
template <class Schedule, class Output, class Epilogue, class Rows = Fp8IdentityRows,
          class Scale = __nv_bfloat16>
void launch_fp8_a16_gemv(const Fp8A16OperandsT<Scale>& p, Output output, Epilogue epilogue,
                         cudaStream_t stream, Rows rows = {}) {
    validate_fp8_operands<Schedule>(p);
    if (p.tokens != 1 || p.rows % Schedule::kBlockRows || p.k % (32 * Schedule::kValuesPerLane))
        throw std::invalid_argument("FP8 GEMV requires T=1 and complete row/K tiles");
    CUDA_CHECK(pdl::launch_consumer({dim3(p.rows / Schedule::kBlockRows), dim3(Schedule::kThreads), 0, stream},
                                    fp8_a16_gemv_kernel<Schedule, Output, Epilogue, Rows, Scale>, p.x, p.codes, p.scales,
                                    output, epilogue, rows, p.rows, p.k));
}

template <class Schedule, class Output, class Epilogue, class Rows = Fp8IdentityRows,
          class Scale = __nv_bfloat16>
void launch_fp8_a16_simt(const Fp8A16OperandsT<Scale>& p, Output output, Epilogue epilogue,
                         cudaStream_t stream, Rows rows = {},
                         pdl::Dependency dependency = pdl::Dependency::Serialized) {
    validate_fp8_operands<Schedule>(p);
    if (p.rows % Schedule::kBlockRows || p.k % (32 * Schedule::kValuesPerLane) ||
        (Schedule::kTokenCapacity && p.tokens > Schedule::kTokenCapacity) ||
        (Schedule::kExactTokens && p.tokens != Schedule::kTokenCapacity))
        throw std::invalid_argument(
            "FP8 SIMT requires complete row/K tiles and matching token capacity");
    if constexpr (requires { Epilogue::kRowTokens; }) {
        if (p.tokens != Epilogue::kRowTokens)
            throw std::invalid_argument("FP8 row epilogue requires its complete token interval");
    }
    const int capacity = Schedule::kTokenCapacity ? Schedule::kTokenCapacity : p.tokens;
    const int blocks   = p.rows / Schedule::kBlockRows * div_up(capacity, Schedule::kBlockTokens);
    CUDA_CHECK(pdl::launch_with(dependency, {dim3(blocks), dim3(Schedule::kThreads), 0, stream},
                                fp8_a16_simt_kernel<Schedule, Output, Epilogue, Rows, Scale>, p, output,
                                epilogue, rows));
}

template <class Schedule, class Output, class Epilogue, class Rows = Fp8IdentityRows,
          class Scale = __nv_bfloat16>
void launch_fp8_a16_mma(const Fp8A16OperandsT<Scale>& p, Output output, Epilogue epilogue,
                        cudaStream_t stream, Rows rows = {}) {
    validate_fp8_operands<Schedule>(p);
    if (p.rows % Schedule::kBlockRows || p.k % Schedule::kBlockK)
        throw std::invalid_argument("FP8 A16 MMA requires complete row/K tiles");
    for_each_token_slice(p.tokens, Schedule::kBlockTokens, [&](int offset, int count) {
        const dim3 grid(p.rows / Schedule::kBlockRows, div_up(count, Schedule::kBlockTokens));
        const auto launch = [&]<bool Full>() {
            constexpr auto kernel = fp8_a16_mma_kernel<Schedule, Full, Output, Epilogue, Rows, Scale>;
            const int bytes =
                fp8_prepare_shared<fp8_mma_shared_bytes<Schedule, Epilogue>, kernel, true>();
            kernel<<<grid, Schedule::kThreads, bytes, stream>>>(
                p.x, p.codes, p.scales, output, epilogue, rows, p.rows, p.k, offset, count);
            CUDA_CHECK(cudaGetLastError());
        };
        if (count % Schedule::kBlockTokens == 0)
            launch.template operator()<true>();
        else
            launch.template operator()<false>();
    });
}

template <class Schedule, class Output, class Epilogue, class Rows = Fp8IdentityRows,
          class Scale = __nv_bfloat16>
void launch_fp8_a16_sliced_k_mma(const Fp8A16OperandsT<Scale>& p, Output output, Epilogue epilogue,
                                 cudaStream_t stream, Rows rows = {},
                                 pdl::Dependency dependency = pdl::Dependency::Serialized) {
    validate_fp8_operands<Schedule>(p);
    constexpr int capacity =
        Schedule::kTokenCapacity ? Schedule::kTokenCapacity : Schedule::kBlockTokens;
    static_assert(capacity > 0 && capacity <= Schedule::kBlockTokens);
    if (p.rows % Schedule::kBlockRows || p.k % Schedule::kBlockK ||
        (Schedule::kExactTokens && p.tokens != capacity))
        throw std::invalid_argument(
            "FP8 sliced-K requires complete row/K tiles and matching tokens");
    // Static: MSVC 2022 rejects the lambda's use of an automatic constexpr local (C2326).
    static constexpr auto kernel =
        fp8_a16_sliced_k_mma_kernel<Schedule, Output, Epilogue, Rows, Scale>;
    const int bytes              = fp8_prepare_shared<Schedule::kSharedBytes, kernel>();
    for_each_token_slice(p.tokens, capacity, [&](int offset, int count) {
        const dim3 grid(p.rows / Schedule::kBlockRows, div_up(count, capacity));
        CUDA_CHECK(pdl::launch_with(
            dependency,
            {grid, dim3(Schedule::kThreads), static_cast<std::size_t>(bytes), stream}, kernel, p,
            output, epilogue, rows, offset));
    });
}

template <class Schedule, class Output, class Epilogue, class Rows = Fp8IdentityRows,
          class Scale = __nv_bfloat16>
void launch_fp8_a8_mma(const Fp8A8OperandsT<Scale>& p, Output output, Epilogue epilogue,
                       cudaStream_t stream, Rows rows = {},
                       pdl::Dependency dependency = pdl::Dependency::Serialized) {
    validate_fp8_operands<Schedule>(p);
    if (p.rows % Schedule::kBlockRows || p.k % Schedule::kBlockK ||
        p.k / Schedule::kBlockK < Schedule::kStages)
        throw std::invalid_argument("FP8 A8 MMA requires complete row/K tiles and enough K stages");
    if constexpr (requires { Schedule::kTerms; }) {
        if ((Schedule::kTerms > 1) != (p.x_residual != nullptr))
            throw std::invalid_argument("FP8 A8 MMA: activation planes do not match the schedule");
    }
    for_each_token_slice(p.tokens, Schedule::kBlockTokens, [&](int offset, int count) {
        const int blocks  = p.rows / Schedule::kBlockRows * div_up(count, Schedule::kBlockTokens);
        const auto launch = [&]<bool Full>() {
            constexpr auto kernel = fp8_a8_mma_kernel<Schedule, Full, Epilogue, Output, Rows, Scale>;
            const int bytes =
                fp8_prepare_shared<fp8_mma_shared_bytes<Schedule, Epilogue>, kernel>();
            CUDA_CHECK(pdl::launch_with(
                dependency,
                {dim3(blocks), dim3(Schedule::kThreads), static_cast<std::size_t>(bytes), stream},
                kernel, p, output, epilogue, rows, offset, count));
        };
        if (count % Schedule::kBlockTokens == 0)
            launch.template operator()<true>();
        else
            launch.template operator()<false>();
    });
}
} // namespace infernix::ops::detail
