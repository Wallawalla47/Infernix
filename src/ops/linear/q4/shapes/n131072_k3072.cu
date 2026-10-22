#include "ops/linear/q4/q4_shapes.h"

namespace infernix::ops::detail {
// Qwen3.5-122B-A10B's proposal head ([131072, 3072], --lm-head-draft). The 5,120-wide head's
// dynamic-K families: one 214 MB read whatever the width, so the same tiles apply (no static-K
// instance at 3,072).
Q4Launch select_q4_n131072_k3072(std::int32_t tokens) {
    if (tokens <= 1) return launch_q4_a16_gemv_r4_w1_direct;
    if (tokens <= 8) return launch_q4_a16_sliced_r32_t8_w4_s2;
    if (tokens <= 16) return launch_q4_a16_sliced_r32_t16_w4_s1;
    if (tokens <= 32) return launch_q4_a16_sliced_r32_t32_w4_s1;
    if (tokens <= 64) return launch_q4_a16_mma_r64_t64_k128_s2_a1;
    if (tokens <= 80) return launch_q4_a16_mma_r64_t80;
    if (tokens <= 96) return launch_q4_a16_mma_r64_t96;
    if (tokens <= 112) return launch_q4_a16_mma_r64_t112;
    return launch_q4_a16_mma_r64_t128;
}
} // namespace infernix::ops::detail
