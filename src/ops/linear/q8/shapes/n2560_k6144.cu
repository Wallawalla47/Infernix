#include "ops/linear/q8/q8_launch.h"
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

} // namespace infernix::ops::detail
