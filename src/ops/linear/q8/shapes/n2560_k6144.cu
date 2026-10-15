#include "ops/linear/q8/q8_gemv_launch.cuh"
#include "ops/linear/q8/q8_grouped_sliced_k_launch.cuh"
#include "ops/linear/q8/q8_launch.h"
#include "ops/linear/q8/q8_mma_launch.cuh"
#include "ops/linear/q8/q8_shapes.h"

namespace infernix::ops::detail {

// Qwen3.8-Flash-Next GDN and QSA output projections: the SIMT route up to 8 columns (the register-streamed
// route measured 16-29 % slower, M1 sweep), the MMA tiles beyond.
Q8Launch select_q8_n2560_k6144(std::int32_t tokens) {
    if (tokens <= 4) return launch_q8_a16_simt_r8_t4;
    if (tokens <= 8) return launch_q8_a16_simt_r8_t8;
    if (tokens <= 64) return launch_q8_a16_mma_r32_t64;
    if (tokens <= 96) return launch_q8_a16_mma_r32_t96;
    // Beyond 96 columns (prefill chunks): the MMA tile sweep of 2026-10-05 (fn/rigs/layer/q8tiles2.bat,
    // RTX 5090). Every MMA tile accumulates each output over K in the same order, so their bits are
    // identical (checked at every swept point); only speed differs.
    if (tokens <= 256) return launch_q8_a16_mma_r32_t64;
    return launch_q8_a16_mma_r64_t128;
}

namespace {
// The q8-sliced-k fast path: NInfer's routes for this problem (NInfer 35e9b5c85), whose GEMV, sliced-K
// and grouped sliced-K MMA accumulate in different orders from the exact SIMT and MMA routes.
using FastGemv    = Q8A16GemvSchedule<4, 1, 2, 6144, 4>;
using FastGrouped = Q8A16GroupedSlicedKMmaSchedule<32, 4, 2, 1, 1, 6144, Cache::cg, Cache::ca, true>;
using FastMma     = Q8A16MmaSchedule<32, 32, 128, 16, 16, 2, 3>;

void fast_gemv(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    launch_q8_a16_gemv<FastGemv>(q8_linear_operands(x, weight),
                                 LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), out.ne[0]},
                                 LinearIdentityEpilogue{}, stream);
}

void fast_grouped(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    launch_q8_a16_grouped_sliced_k_mma<FastGrouped>(
        q8_linear_operands(x, weight), LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), out.ne[0]},
        LinearIdentityEpilogue{}, stream);
}

void fast_mma(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    launch_q8_a16_mma<FastMma>(q8_linear_operands(x, weight),
                               LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), out.ne[0]},
                               LinearIdentityEpilogue{}, stream);
}
} // namespace

Q8Launch select_q8_n2560_k6144_fast(std::int32_t tokens) {
    if (tokens == 1) return fast_gemv;
    if (tokens <= 32) return launch_q8_a16_sliced_r16_t16_w8_s2;
    if (tokens <= 50) return launch_q8_a16_sliced_r16_t32_w4_s2;
    if (tokens <= 64) return fast_grouped;
    if (tokens <= 128) return fast_mma;
    // Beyond 128 columns NInfer's table measured 1.0-1.45 times the exact routes (2026-10-15 sweep,
    // RTX 5090), so the exact routes serve them.
    return select_q8_n2560_k6144(tokens);
}

} // namespace infernix::ops::detail
