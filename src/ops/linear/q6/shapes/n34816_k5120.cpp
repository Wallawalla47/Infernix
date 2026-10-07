#include "ops/linear/q6/q6_shapes.h"

namespace infernix::ops::detail {

// The fused gate_up of a Q6 Qwen3.8-27B MLP (upstream PR #284). It shares K=5120 with the output
// head, so it takes that shape's capacity routes on the unified Q6 templates; the PR's own ladder
// named the pre-unification SIMT and K-split MMA instances, which no longer exist.
Q6Launch select_q6_n34816_k5120(std::int32_t tokens) {
    if (tokens <= 2) return launch_q6_a16_simt_r8_t4;
    if (tokens <= 8) return launch_q6_a16_sliced_r16_t8_w4_s2;
    if (tokens <= 16) return launch_q6_a16_sliced_r32_t16_w4_s2;
    if (tokens <= 32) return launch_q6_a16_sliced_r32_t32_w4_s1;
    if (tokens <= 48) return launch_q6_a16_mma_r64_t48_k128;
    if (tokens <= 56) return launch_q6_a16_mma_r64_t56_k128;
    if (tokens <= 64) return launch_q6_a16_mma_r64_t64_k128;
    if (tokens <= 80) return launch_q6_a16_mma_r64_t80;
    if (tokens <= 96) return launch_q6_a16_mma_r64_t96;
    if (tokens <= 112) return launch_q6_a16_mma_r64_t112;
    return launch_q6_a16_mma_r64_t128;
}

} // namespace infernix::ops::detail
