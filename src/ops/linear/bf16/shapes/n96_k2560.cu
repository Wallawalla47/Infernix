#include "ops/linear/bf16/bf16_instances.cuh"
#include "ops/linear/bf16/bf16_shapes.h"
#include "ops/linear/bf16/bf16_launch.cuh"

namespace infernix::ops::detail {
// NInfer's tuned routes for this shape (NInfer 81c8ce09), used where they measured fastest.
namespace {
using S0 = Bf16ScheduleInstance<
    Bf16A16GemvSchedule<4, 2, 1, 8, 4, Bf16ActivationAccess::Direct, Bf16WeightCache::Default,
                        Bf16PhaseOrder::Sequential, 1, 1, 4, 2>,
    2560>;
using S1 = Bf16ScheduleInstance<
    Bf16A16SimtSchedule<4, 2, 1, 8, 4, 2, Bf16SimtActivationAccess::WarpPacked,
                        Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 4, 1, 2, 2>,
    2560>;
using S2 = Bf16A16SimtSchedule<8, 2, 1, 8, 4, 2, Bf16SimtActivationAccess::WarpPacked,
                               Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 4, 1, 2, 2>;
using S3 = Bf16ScheduleInstance<Bf16A16SlicedKMmaSchedule<16, 8, 8, 64, 2>, 2560>;
using S4 = Bf16ScheduleInstance<Bf16A16SlicedKMmaSchedule<16, 32, 8, 64, 2>, 2560>;
using S5 = Bf16ScheduleInstance<
    Bf16A16MmaSchedule<32, 32, 256, 16, 8, 3, 1, Cache::cg, Cache::cg,
                       Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>,
    2560>;
Bf16Launch ninfer_route(std::int32_t tokens) {
    if (tokens <= 1) return launch_bf16_gemv<S0>;
    if (tokens <= 4) return launch_bf16_simt<S1>;
    if (tokens <= 16) return launch_bf16_simt<S2>;
    if (tokens <= 128) return launch_bf16_sliced_k_mma<S3>;
    if (tokens <= 512) return launch_bf16_sliced_k_mma<S4>;
    return launch_bf16_mma<S5>;
}
} // namespace

// Qwen3.8-Flash-Next: the GDN a/b projections (both recipes), T = a call's columns.
// T <= 8 runs one column-invariant route at every width, so a column's bits do not depend on how
// many columns share the call (C = 1 and C = 2 give the same greedy text). Wider calls take the
// fastest measured route per interval (bench/ops/bf16_flash_next_sweep.cu, RTX 5090, 2 % tolerance,
// seams at the last measured T).
Bf16Launch select_bf16_n96_k2560(std::int32_t tokens) {
    // NInfer's T = 1 route accumulates a column like its T = 8 route (checked bit for bit): 3.20 us vs 4.74.
    if (tokens == 1) return ninfer_route(1);
    if (tokens <= 8) return ninfer_route(8); // decode 33.0 us vs skinny 50.9 us over T = 1..8
    if (tokens <= 384) return ninfer_route(tokens);
    if (tokens <= 512) return launch_bf16_sliced_k_mma<Bf16A16SlicedR32T16W4>; // sliced_r32t16w4
    if (tokens <= 768) return launch_bf16_mma<Bf16A16MmaR32T32K256S3>; // mma_r32t32k256s3
    if (tokens <= 1024) return ninfer_route(tokens);
    if (tokens <= 1536) return launch_bf16_mma<Bf16A16MmaR32T32K128S3>; // mma_r32t32k128s3
    if (tokens <= 2048) return launch_bf16_tma_mma<Bf16A16TmaTailMmaSchedule<64, 32, 32, 16, 3>>; // tail_r64t32s3
    if (tokens <= 3072) return launch_bf16_mma<Bf16A16MmaR32T32K128S3>; // mma_r32t32k128s3
    if (tokens <= 4096) return launch_bf16_tma_mma<Bf16A16TmaTailMmaSchedule<64, 64, 32, 32, 3>>; // tail_r64t64s3
    return launch_bf16_tma_mma<Bf16A16TmaTailMmaSchedule<64, 128, 32, 32, 2, 1, Bf16MmaRaster::RowFast>>; // tail_r64t128s2_rows
}
} // namespace infernix::ops::detail
