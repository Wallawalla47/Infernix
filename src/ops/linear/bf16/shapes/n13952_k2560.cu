#include "ops/linear/bf16/bf16_instances.cuh"
#include "ops/linear/bf16/bf16_shapes.h"
#include "ops/linear/bf16/bf16_launch.cuh"

namespace infernix::ops::detail {
// NInfer's tuned routes for this shape (NInfer 81c8ce09), used where they measured fastest.
namespace {
using S0 = Bf16ScheduleInstance<
    Bf16A16GemvSchedule<4, 1, 1, 16, 4, Bf16ActivationAccess::Direct, Bf16WeightCache::Default,
                        Bf16PhaseOrder::Sequential, 1, 1, 4, 2>,
    2560>;
using S1 = Bf16ScheduleInstance<Bf16A16SlicedKMmaSchedule<16, 8, 4, 128, 2>, 2560>;
using S2 = Bf16ScheduleInstance<
    Bf16A16MmaSchedule<32, 16, 128, 16, 16, 2, 2, Cache::cg, Cache::cg,
                       Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>,
    2560>;
using S3 = Bf16ScheduleInstance<
    Bf16A16MmaSchedule<32, 32, 128, 16, 8, 2, 2, Cache::cg, Cache::cg,
                       Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>,
    2560>;
using S4 = Bf16A16MmaSchedule<64, 16, 128, 16, 8, 2, 2, Cache::cg, Cache::cg,
                              Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>;
using S5 = Bf16A16TmaMmaSchedule<128, 64, 128, 32, 32, 2, 1>;
using S6 = Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<128, 32, 128, 16, 16, 2, 1>, 2560>;
using S7 = Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<128, 48, 64, 32, 16, 2, 1>, 2560>;
using S8 = Bf16A16TmaMmaSchedule<64, 64, 64, 32, 32, 3, 2>;
using S9 = Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<64, 128, 64, 32, 32, 3, 1>, 2560>;
Bf16Launch ninfer_route(std::int32_t tokens) {
    if (tokens <= 1) return launch_bf16_gemv<S0>;
    if (tokens <= 8) return launch_bf16_sliced_k_mma<S1>;
    if (tokens <= 16) return launch_bf16_mma<S2>;
    if (tokens <= 32) return launch_bf16_mma<S3>;
    if (tokens <= 48) return launch_bf16_mma<S4>;
    if (tokens <= 64) return launch_bf16_tma_mma<S5>;
    if (tokens <= 96) return launch_bf16_tma_mma<S6>;
    if (tokens <= 128) return launch_bf16_tma_mma<S7>;
    if (tokens <= 512) return launch_bf16_tma_mma<S8>;
    return launch_bf16_tma_mma<S9>;
}
} // namespace

// Qwen3.8-Flash-Next: the QSA query, gate, key, value and index projections (both recipes), T = a call's columns.
// T <= 8 runs one column-invariant route at every width, so a column's bits do not depend on how
// many columns share the call (C = 1 and C = 2 give the same greedy text). Wider calls take the
// fastest measured route per interval (bench/ops/bf16_flash_next_sweep.cu, RTX 5090, 2 % tolerance,
// seams at the last measured T).
Bf16Launch select_bf16_n13952_k2560(std::int32_t tokens) {
    if (tokens <= 8) return select_bf16_general_launch(tokens); // skinny GEMV (311.5 us over T = 1..8)
    if (tokens <= 128) return ninfer_route(tokens);
    if (tokens <= 192) return launch_bf16_tma_mma<Bf16A16TmaMmaSchedule<128, 64, 64, 32, 32, 4>>; // tma_r128t64k64s4
    if (tokens <= 256) return launch_bf16_tma_mma<Bf16A16TmaMmaSchedule<64, 128, 64, 32, 32, 2>>; // tma_r64t128k64s2
    if (tokens <= 768) return launch_bf16_tma_mma<Bf16A16TmaTailMmaSchedule<128, 128, 64, 32, 3, 1, Bf16MmaRaster::RowFast>>; // tail_r128t128s3_w64x32_rows
    if (tokens <= 1024) return launch_bf16_tma_mma<Bf16A16TmaMmaSchedule<64, 128, 64, 32, 32, 2, 1, Bf16MmaRaster::RowFast>>; // tma_r64t128k64s2_rows
    return launch_bf16_tma_mma<Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3, 1, Bf16MmaRaster::Grouped, 8>>; // tma_r128t128k64s3_w64x32_group8
}
} // namespace infernix::ops::detail
