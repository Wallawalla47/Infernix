#include "ops/linear/q8/q8_launch.h"
#include "ops/linear/q8/q8_shapes.h"

namespace ninfer::ops::detail {

// Qwen3.8-Flash-Next GDN and QSA output projections: the SIMT route up to 8 columns (the register-streamed
// route measured 16-29 % slower, M1 sweep), the MMA tiles beyond.
Q8Launch select_q8_n2560_k6144(std::int32_t tokens) {
    if (tokens <= 4) return launch_q8_a16_simt_r8_t4;
    if (tokens <= 8) return launch_q8_a16_simt_r8_t8;
    if (tokens <= 64) return launch_q8_a16_mma_r32_t64;
    if (tokens <= 96) return launch_q8_a16_mma_r32_t96;
    return launch_q8_a16_mma_r32_t128;
}

} // namespace ninfer::ops::detail
