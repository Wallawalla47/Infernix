#include "ops/linear/bf16/bf16_instances.cuh"
#include "ops/linear/bf16/bf16_shapes.h"
#include "ops/linear/bf16/bf16_launch.cuh"

namespace infernix::ops::detail {
// Vision merger fc2 for a 5120-wide text model (Qwen3.8 27B), T = merged tokens. Routes per token interval from the V0 sweep on the RTX 5090 (the fastest
// candidate within 2 % at each measured T, seams at the last measured T).
Bf16Launch select_bf16_n5120_k4608(std::int32_t tokens) {
    if (tokens <= 8) return select_bf16_general_launch(tokens);
    if (tokens <= 32)
        return launch_bf16_sliced_k_mma<Bf16ScheduleInstance<Bf16A16SlicedR32T16W4, 4608>>; // sliced_r32t16w4
    if (tokens <= 64)
        return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<64, 32, 64, 32, 16, 3>, 4608>>; // tma_r64t32k64s3
    if (tokens <= 128)
        return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaTailMmaSchedule<64, 64, 32, 32, 3>, 4608>>; // tail_r64t64s3
    if (tokens <= 256)
        return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<128, 64, 64, 32, 32, 4>, 4608>>; // tma_r128t64k64s4
    if (tokens <= 384)
        return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaTailMmaSchedule<64, 64, 32, 32, 3>, 4608>>; // tail_r64t64s3
    if (tokens <= 512)
        return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3>, 4608>>; // tma_r128t128k64s3_w64x32
    if (tokens <= 768)
        return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<64, 128, 64, 32, 32, 2, 1, Bf16MmaRaster::RowFast>, 4608>>; // tma_r64t128k64s2_rows
    if (tokens <= 1024)
        return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaTailMmaSchedule<128, 128, 32, 64, 3, 1, Bf16MmaRaster::RowFast>, 4608>>; // tail_r128t128s3_w32x64_rows
    if (tokens <= 1152)
        return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<64, 64, 64, 32, 32, 3>, 4608>>; // tma_r64t64k64s3
    if (tokens <= 2048)
        return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3, 1, Bf16MmaRaster::Grouped, 8>, 4608>>; // tma_r128t128k64s3_w64x32_group8
    if (tokens <= 2304)
        return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaTailMmaSchedule<64, 128, 32, 32, 2, 1, Bf16MmaRaster::RowFast>, 4608>>; // tail_r64t128s2_rows
    return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaTailMmaSchedule<128, 128, 64, 32, 3, 1, Bf16MmaRaster::Grouped, 8>, 4608>>; // tail_r128t128s3_w64x32_group8
}
} // namespace infernix::ops::detail
