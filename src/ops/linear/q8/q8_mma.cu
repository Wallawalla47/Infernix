#include "ops/linear/q8/q8_mma_launch.cuh"
#include "ops/linear/q8/q8_instances.cuh"
#include "ops/linear/q8/q8_launch.h"

#include "ops/linear/common/rowsplit_ring_linear.cuh"
#include "ops/linear/q8/q8_ring_codec.cuh"

namespace infernix::ops::detail {

// The wide route's first width on the staging ring (ops/linear/common/rowsplit_ring_linear.cuh): narrower calls
// keep the 64-row MMA tile. Both give the same bits. RTX 5090, linear_bench (CUDA graph, cold L2, A B B A, equal
// output hashes), ring against the MMA tile at 512 / 1,024 / 2,048 / 4,096 columns: [12800, 2560] +6.3 / -5.6 /
// -0.9 / -4.5 %, [17408, 5120] +5.3 / +1.9 / -4.0 / -3.7 %, [2560, 12800] +42.4 / -8.3 / -7.7 / -6.8 %,
// [5120, 17408] -8.4 / -8.8 / -7.0 / -6.4 % (2026-11-10): from 2,048 columns every measured shape gains.
inline constexpr std::int32_t kQ8RingMinTokens = 2048;
// The runtime-shape schedules: FP32 or BF16 output (a vocabulary head's FP32 logits take them).
#define INFERNIX_Q8_MMA_LAUNCHER(Name, Schedule)                                                     \
    void Name(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {                \
        with_linear_output(out, out.ne[0], [&](auto output) {                                      \
            launch_q8_a16_mma<q8_instances::Schedule>(q8_linear_operands(x, w), output,           \
                                                      LinearIdentityEpilogue{}, stream);           \
        });                                                                                        \
    }

INFERNIX_Q8_MMA_LAUNCHER(launch_q8_a16_mma_r32_t64, MmaR32T64)
INFERNIX_Q8_MMA_LAUNCHER(launch_q8_a16_mma_r32_t96, MmaR32T96)
INFERNIX_Q8_MMA_LAUNCHER(launch_q8_a16_mma_r32_t128, MmaR32T128)
INFERNIX_Q8_MMA_LAUNCHER(launch_q8_a16_mma_r48_t64, MmaR48T64)
INFERNIX_Q8_MMA_LAUNCHER(launch_q8_a16_mma_r64_t96, MmaR64T96)
void launch_q8_a16_wide_t128(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    with_linear_output(out, out.ne[0], [&](auto output) {
        const Q8LinearOperands p = q8_linear_operands(x, w);
        if (p.tokens >= kQ8RingMinTokens) {
            validate_q8_operands(p);
            launch_rowsplit_ring_linear<Q8RingFormat, RowSplitRingTile128, kRowSplitRingStages,
                                        kRowSplitRingProducerWarps>(
                p.x, Q8RingFormat::Weights{p.codes, p.scales}, p.rows, p.k, p.padded_k, p.tokens, output,
                LinearIdentityEpilogue{}, stream);
            return;
        }
        launch_q8_a16_mma<q8_instances::MmaR64T128>(p, output, LinearIdentityEpilogue{}, stream);
    });
}
INFERNIX_Q8_MMA_LAUNCHER(launch_q8_a16_mma_r96_t96, MmaR96T96)
INFERNIX_Q8_MMA_LAUNCHER(launch_q8_a16_mma_r128_t64, MmaR128T64)
INFERNIX_Q8_MMA_LAUNCHER(launch_q8_a16_mma_r128_t80, MmaR128T80)
INFERNIX_Q8_MMA_LAUNCHER(launch_q8_a16_mma_r64x16_t48_k128_a1, MmaR64x16T48K128A1)
INFERNIX_Q8_MMA_LAUNCHER(launch_q8_a16_mma_r64x32_t64_k128_a1, MmaR64x32T64K128A1)

#undef INFERNIX_Q8_MMA_LAUNCHER

} // namespace infernix::ops::detail
