#include "ops/linear/q8/q8_mma_launch.cuh"
#include "ops/linear/q8/q8_instances.cuh"
#include "ops/linear/q8/q8_launch.h"

namespace infernix::ops::detail {
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
INFERNIX_Q8_MMA_LAUNCHER(launch_q8_a16_mma_r64_t128, MmaR64T128)
INFERNIX_Q8_MMA_LAUNCHER(launch_q8_a16_mma_r96_t96, MmaR96T96)
INFERNIX_Q8_MMA_LAUNCHER(launch_q8_a16_mma_r128_t64, MmaR128T64)
INFERNIX_Q8_MMA_LAUNCHER(launch_q8_a16_mma_r128_t80, MmaR128T80)
INFERNIX_Q8_MMA_LAUNCHER(launch_q8_a16_mma_r64x16_t48_k128_a1, MmaR64x16T48K128A1)
INFERNIX_Q8_MMA_LAUNCHER(launch_q8_a16_mma_r64x32_t64_k128_a1, MmaR64x32T64K128A1)

#undef INFERNIX_Q8_MMA_LAUNCHER

} // namespace infernix::ops::detail
