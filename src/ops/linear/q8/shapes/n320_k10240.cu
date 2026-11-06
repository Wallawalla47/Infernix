#include "ops/linear/q8/q8_shapes.h"
#include "ops/linear/q8/q8_sliced_k_launch.cuh"
#include "ops/linear/q8/q8_stream_launch.cuh"

namespace infernix::ops::detail {
namespace {
// Qwen3.8-Flash-Next final mixer down: the register-streamed route at 1-3 and 9-64 columns (2-7x faster than
// the MMA tile there), the few-row SIMT route at 4-8 (M1 sweep). K1: 2 row(s) per warp, 8 warps splitting each
// row's K, x read through L1.
template <int MaxCols, int ColsPerPass>
using Stream = Q8A16StreamSchedule<10240, 2, 8, 8, MaxCols, ColsPerPass, Q8StreamX::Global, 2>;
} // namespace

Q8Launch select_q8_n320_k10240(std::int32_t tokens) {
    if (tokens <= 3) return launch_q8_a16_stream_linear<Stream<4, 4>>;
    if (tokens <= 4) return launch_q8_a16_simt_r1_t4_w8;
    if (tokens <= 8) return launch_q8_a16_simt_r1_t8_w8;
    if (tokens <= 64) return launch_q8_a16_stream_linear<Stream<16, 8>>;
    if (tokens <= 96) return launch_q8_a16_mma_r32_t96;
    return launch_q8_a16_mma_r32_t128;
}

namespace {
// The q8-sliced-k fast path's routes past 64 columns (candidates NQ-01), as for [324, 10240] (n324_k10240.cu).
template <class Schedule>
void hc_sliced(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    using Instance = typename Schedule::template with_problem<10240, Schedule::kBlockTokens>;
    launch_q8_a16_sliced_k_mma<Instance>(q8_linear_operands(x, weight),
                                         LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), out.ne[0]},
                                         LinearIdentityEpilogue{}, stream, {}, pdl::Dependency::Programmatic);
}
using S16W8   = Q8A16SlicedKMmaSchedule<16, 8, 2, 1, Q8ScaleAccess::Shared>;
using S32W4   = Q8A16SlicedKMmaSchedule<32, 4, 2, 1, Q8ScaleAccess::Shared>;
using S32W4R2 = Q8A16SlicedKMmaSchedule<32, 4, 1, 1, Q8ScaleAccess::Shared, Cache::ca, Cache::cg,
                                        Q8ActivationStage::ActiveOnly, 0, 32, false, 2>;
} // namespace

// Sweep 2026-11-03 (infernix_linear_bench --routes fast, RTX 5090, cold L2, two runs, 64-column steps): against the
// exact routes 128 columns 105.1 -> 19.0 us, 256 105.1 -> 29.3, 512 105.0 -> 48.3, 1,024 107.1 -> 62.0, 1,536 119.4 ->
// 109.3, 2,048 127.4 -> 119.4 (Dense8 prefill: n324_k10240.cu).
Q8Launch select_q8_n320_k10240_fast(std::int32_t tokens) {
    if (tokens <= 64) return select_q8_n320_k10240(tokens);
    if (tokens <= 256) return hc_sliced<S16W8>;
    if (tokens <= 512) return hc_sliced<S32W4>;
    if (tokens <= 2048) return hc_sliced<S32W4R2>;
    return select_q8_n320_k10240(tokens);
}

} // namespace infernix::ops::detail
