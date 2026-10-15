#include "ops/linear/q8/q8_gemv_launch.cuh"
#include "ops/linear/q8/q8_grouped_sliced_k_launch.cuh"
#include "ops/linear/q8/q8_instances.cuh"
#include "ops/linear/q8/q8_mma_launch.cuh"
#include "ops/linear/q8/q8_shapes.h"
#include "ops/linear/q8/q8_sliced_k_launch.cuh"
#include "ops/linear/q8/q8_stream_launch.cuh"
#include "ops/gdn_input_proj/q8/q8_gdn_conv_columns.cuh"

#include <stdexcept>

namespace infernix::ops::detail {
namespace {
// Qwen3.8-Flash-Next GDN q/k/v/z projection (packed): the register-streamed route up to 8 columns (5-28 %
// faster than SIMT), the MMA tiles beyond (M1 sweep). K1: 2 row(s) per warp, x staged in shared memory per
// pass.
template <int MaxCols, int ColsPerPass>
using Stream = Q8A16StreamSchedule<2560, 2, 1, 8, MaxCols, ColsPerPass, Q8StreamX::Shared, 2>;

// The routes of both tables up to 8 columns, where one call holds the one-position-per-sequence
// columns of a decode round. The fused GDN convolution (launch_q8_n16384_k2560_conv_snapshot) runs
// these same routes with its own row consumer; the contraction and so each projected FP32 value is
// unchanged by the consumer.
template <class Output, class Epilogue>
void launch_decode_routes(const Q8LinearOperands& operands, Output output, Epilogue epilogue,
                          cudaStream_t stream) {
    if (operands.tokens <= 4) return launch_q8_a16_stream<Stream<4, 4>>(operands, output, epilogue, stream);
    launch_q8_a16_stream<Stream<8, 8>>(operands, output, epilogue, stream);
}

// One route table for both outputs: the packed [16384,T] linear result, and the q/k/v rows and z rows
// written straight to their two destinations (gdn_input_proj). The output map changes only where a
// value is stored, so both produce the same bits.
template <class Output>
void launch_routes(const Q8LinearOperands& operands, Output output, cudaStream_t stream) {
    const LinearIdentityEpilogue epilogue{};
    const std::int32_t tokens = operands.tokens;
    if (tokens <= 8) return launch_decode_routes(operands, output, epilogue, stream);
    if (tokens <= 64) return launch_q8_a16_mma<q8_instances::MmaR32T64>(operands, output, epilogue, stream);
    if (tokens <= 96) return launch_q8_a16_mma<q8_instances::MmaR32T96>(operands, output, epilogue, stream);
    // Beyond 96 columns (prefill chunks): the MMA tile sweep of 2026-10-05 (fn/rigs/layer/q8tiles2.bat,
    // RTX 5090). Every MMA tile accumulates each output over K in the same order, so their bits are
    // identical (checked at every swept point); only speed differs.
    launch_q8_a16_mma<q8_instances::MmaR64T128>(operands, output, epilogue, stream);
}

// The q8-sliced-k fast path: NInfer's routes for this problem (NInfer cd521ab7d). The T = 1 GEMV, the
// sliced-K MMA at T = 2-8 (group scales applied after each MMA) and the grouped sliced-K MMA accumulate
// in different orders, so a column's bits depend on the call's T.
using FastGemv    = Q8A16GemvSchedule<8, 1, 2, 2560, 8>;
using FastSliced8 = Q8A16SlicedKMmaSchedule<8, 4, 1, 1, Q8ScaleAccess::Shared, Cache::ca, Cache::cg,
                                            Q8ActivationStage::PaddedZero, 2560>;
using FastGrouped = Q8A16GroupedSlicedKMmaSchedule<16, 2, 2, 2, 1, 2560, Cache::cg, Cache::ca, true>;
using FastSliced24 = Q8A16SlicedKMmaSchedule<24, 4, 1, 1, Q8ScaleAccess::Shared, Cache::ca, Cache::cg,
                                             Q8ActivationStage::PaddedZero, 2560>;
using FastSliced32 = Q8A16SlicedKMmaSchedule<32, 4, 2, 1, Q8ScaleAccess::Shared, Cache::ca, Cache::cg,
                                             Q8ActivationStage::PaddedZero, 2560>;
using FastSliced40 = Q8A16SlicedKMmaSchedule<40, 4, 2, 1, Q8ScaleAccess::Shared, Cache::ca, Cache::cg,
                                             Q8ActivationStage::PaddedZero, 2560>;
using FastMma48 = Q8A16MmaSchedule<64, 48, 128, 16, 24, 1, 2>;
using FastMma63 = Q8A16MmaSchedule<32, 64, 64, 32, 16, 2, 3>;
using FastMma64 = Q8A16MmaSchedule<64, 64, 128, 32, 16, 1, 2>;
using FastMma96 = Q8A16MmaSchedule<64, 96, 64, 64, 16, 2, 2>;
using FastMma128 = Q8A16MmaSchedule<64, 128, 64, 32, 32, 2, 2, Q8MmaFragmentPipeline::Serial, Cache::cg,
                                    Cache::cg, Cache::ca, false, 2560, 20>;
using FastMma512 = Q8A16MmaSchedule<64, 64, 64, 32, 16, 2, 3, Q8MmaFragmentPipeline::Serial, Cache::cg,
                                    Cache::cg, Cache::ca, false, 2560, 20>;

template <class Output, class Epilogue>
void launch_fast_decode_routes(const Q8LinearOperands& operands, Output output, Epilogue epilogue,
                               cudaStream_t stream) {
    if (operands.tokens <= 1) return launch_q8_a16_gemv<FastGemv>(operands, output, epilogue, stream);
    launch_q8_a16_sliced_k_mma<FastSliced8>(operands, output, epilogue, stream);
}

template <class Output>
void launch_fast_routes(const Q8LinearOperands& operands, Output output, cudaStream_t stream) {
    const LinearIdentityEpilogue epilogue{};
    const std::int32_t tokens = operands.tokens;
    if (tokens <= 8) return launch_fast_decode_routes(operands, output, epilogue, stream);
    if (tokens <= 16) return launch_q8_a16_grouped_sliced_k_mma<FastGrouped>(operands, output, epilogue, stream);
    if (tokens <= 24) return launch_q8_a16_sliced_k_mma<FastSliced24>(operands, output, epilogue, stream);
    if (tokens <= 32) return launch_q8_a16_sliced_k_mma<FastSliced32>(operands, output, epilogue, stream);
    if (tokens <= 40) return launch_q8_a16_sliced_k_mma<FastSliced40>(operands, output, epilogue, stream);
    if (tokens <= 48) return launch_q8_a16_mma<FastMma48>(operands, output, epilogue, stream);
    if (tokens <= 63) return launch_q8_a16_mma<FastMma63>(operands, output, epilogue, stream);
    if (tokens <= 64) return launch_q8_a16_mma<FastMma64>(operands, output, epilogue, stream);
    if (tokens <= 96) return launch_q8_a16_mma<FastMma96>(operands, output, epilogue, stream);
    if (tokens <= 128) return launch_q8_a16_mma<FastMma128>(operands, output, epilogue, stream);
    // NInfer's 512-wide tile measured 1.05-1.10 times the exact routes at T = 321-384 and 1.02-1.04 at
    // 449-511 (2026-10-15 sweep, RTX 5090), so the exact routes serve those widths.
    if ((tokens > 320 && tokens <= 384) || (tokens > 448 && tokens < 512)) {
        return launch_routes(operands, output, stream);
    }
    if (tokens <= 512) return launch_q8_a16_mma<FastMma512>(operands, output, epilogue, stream);
    launch_q8_a16_mma<FastMma128>(operands, output, epilogue, stream);
}

void launch_linear(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    launch_routes(q8_linear_operands(x, weight),
                  LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), out.ne[0]}, stream);
}

void launch_linear_fast(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    launch_fast_routes(q8_linear_operands(x, weight),
                       LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), out.ne[0]}, stream);
}
} // namespace

Q8Launch select_q8_n16384_k2560(std::int32_t) { return launch_linear; }

Q8Launch select_q8_n16384_k2560_fast(std::int32_t) { return launch_linear_fast; }

void launch_q8_n16384_k2560_split(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                  LinearRoutes routes, cudaStream_t stream) {
    const LinearBf16SegmentedOutput<10240, 6144> output{
        {static_cast<__nv_bfloat16*>(qkv.data), static_cast<__nv_bfloat16*>(z.data)}};
    if (has_route(routes, LinearRoutes::Q8SlicedK)) {
        launch_fast_routes(q8_linear_operands(x, weight), output, stream);
        return;
    }
    launch_routes(q8_linear_operands(x, weight), output, stream);
}

void launch_q8_n16384_k2560_conv_snapshot(const Tensor& x, const Weight& weight,
                                          const Tensor& conv_weight, Tensor& conv_states,
                                          const Tensor& initial_state_slots,
                                          const Tensor& snapshot_base_slots, Tensor& query,
                                          Tensor& key, Tensor& value, Tensor& z, LinearRoutes routes,
                                          cudaStream_t stream) {
    const Q8LinearOperands operands = q8_linear_operands(x, weight);
    if (operands.tokens < 1 || operands.tokens > 8) {
        throw std::invalid_argument("Q8 [16384,2560] fused GDN convolution covers 1..8 sequences");
    }
    const GdnConvSnapshotColumnsOutput output{
        static_cast<const __nv_bfloat16*>(conv_weight.data),
        static_cast<__nv_bfloat16*>(conv_states.data),
        static_cast<const std::int32_t*>(initial_state_slots.data),
        static_cast<const std::int32_t*>(snapshot_base_slots.data),
        static_cast<__nv_bfloat16*>(query.data),
        static_cast<__nv_bfloat16*>(key.data),
        static_cast<__nv_bfloat16*>(value.data),
        static_cast<__nv_bfloat16*>(z.data),
    };
    const GdnConvSnapshotRowEpilogue epilogue{};
    if (has_route(routes, LinearRoutes::Q8SlicedK)) {
        launch_fast_decode_routes(operands, output, epilogue, stream);
        return;
    }
    launch_decode_routes(operands, output, epilogue, stream);
}

} // namespace infernix::ops::detail
