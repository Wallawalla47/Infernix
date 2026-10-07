#include "ops/linear/bf16/bf16_instances.cuh"
#include "ops/linear/bf16/bf16_shapes.h"
#include "ops/linear/bf16/bf16_launch.cuh"

namespace infernix::ops::detail {
// Qwen3.8-Flash-Next GDN a/b projections, T = a call's columns. Routes per token interval from the
// tile sweep on the RTX 5090 (the fastest candidate within 2 % at each measured T, seams at the last
// measured T); T <= 8 keeps the runtime-shape fallback's skinny GEMV, so decode and verification
// outputs are unchanged.
Bf16Launch select_bf16_n96_k2560(std::int32_t tokens) {
    if (tokens <= 8) return select_bf16_general_launch(tokens);
    if (tokens <= 768)
        return launch_bf16_sliced_k_mma<Bf16ScheduleInstance<Bf16A16SlicedR32T16W4, 2560>>; // sliced_r32t16w4
    if (tokens <= 1536)
        return launch_bf16_mma<Bf16ScheduleInstance<Bf16A16MmaR32T32K256S3, 2560>>; // mma_r32t32k256s3
    if (tokens <= 2048)
        return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaTailMmaSchedule<64, 32, 32, 16, 3>, 2560>>; // tail_r64t32s3
    if (tokens <= 3352)
        return launch_bf16_mma<Bf16ScheduleInstance<Bf16A16MmaR32T32K128S3, 2560>>; // mma_r32t32k128s3
    if (tokens <= 4096)
        return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaTailMmaSchedule<64, 64, 32, 32, 3>, 2560>>; // tail_r64t64s3
    if (tokens <= 8192)
        return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaTailMmaSchedule<64, 128, 32, 32, 2>, 2560>>; // tail_r64t128s2
    return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaTailMmaSchedule<128, 128, 32, 64, 3, 1, Bf16MmaRaster::RowFast>, 2560>>; // tail_r128t128s3_w32x64_rows
}
} // namespace infernix::ops::detail
