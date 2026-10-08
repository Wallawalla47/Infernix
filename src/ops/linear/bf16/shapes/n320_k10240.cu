#include "ops/linear/bf16/bf16_instances.cuh"
#include "ops/linear/bf16/bf16_shapes.h"
#include "ops/linear/bf16/bf16_launch.cuh"

namespace infernix::ops::detail {
// NInfer's tuned routes for this shape (NInfer 81c8ce09), used where they measured fastest.
namespace {
using S0 = Bf16ScheduleInstance<
    Bf16A16GemvSchedule<2, 1, 1, 16, 4, Bf16ActivationAccess::Direct, Bf16WeightCache::Default,
                        Bf16PhaseOrder::Sequential, 1, 1, 4, 2>,
    10240>;
using S1 = Bf16ScheduleInstance<
    Bf16A16SimtSchedule<8, 2, 1, 8, 4, 2, Bf16SimtActivationAccess::DirectStream,
                        Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 4, 1, 2, 2>,
    10240>;
using S2 = Bf16ScheduleInstance<Bf16A16SlicedKMmaSchedule<16, 8, 4, 128, 3>, 10240>;
using S3 = Bf16ScheduleInstance<Bf16A16SlicedKMmaSchedule<16, 16, 4, 128, 3>, 10240>;
using S4 = Bf16ScheduleInstance<Bf16A16SlicedKMmaSchedule<16, 16, 4, 64, 3>, 10240>;
using S5 = Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<32, 64, 64, 16, 8, 4, 1>, 10240>;
using S6 = Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<32, 80, 64, 16, 8, 5, 1>, 10240>;
Bf16Launch ninfer_route(std::int32_t tokens) {
    if (tokens <= 1) return launch_bf16_gemv<S0>;
    if (tokens <= 8) return launch_bf16_simt<S1>;
    if (tokens <= 48) return launch_bf16_sliced_k_mma<S2>;
    if (tokens <= 96) return launch_bf16_sliced_k_mma<S3>;
    if (tokens <= 128) return launch_bf16_sliced_k_mma<S4>;
    if (tokens <= 512) return launch_bf16_tma_mma<S5>;
    return launch_bf16_tma_mma<S6>;
}
} // namespace

// Qwen3.8-Flash-Next: the final mixer's down projection (recipe A), T = a call's columns.
// T <= 8 runs one column-invariant route at every width, so a column's bits do not depend on how
// many columns share the call (C = 1 and C = 2 give the same greedy text). Wider calls take the
// fastest measured route per interval (bench/ops/bf16_flash_next_sweep.cu, RTX 5090, 2 % tolerance,
// seams at the last measured T).
Bf16Launch select_bf16_n320_k10240(std::int32_t tokens) {
    if (tokens <= 8) return ninfer_route(8); // decode 55.7 us vs skinny 138.9 us over T = 1..8
    return ninfer_route(tokens);
}
} // namespace infernix::ops::detail
