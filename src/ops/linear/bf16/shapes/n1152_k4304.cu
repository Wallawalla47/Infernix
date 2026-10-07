#include "ops/linear/bf16/bf16_instances.cuh"
#include "ops/linear/bf16/bf16_shapes.h"
#include "ops/linear/bf16/bf16_launch.cuh"

namespace infernix::ops::detail {
// Vision MLP fc2, T = raw patches; 4304 inputs end in a partial K tile. Routes per token interval from the V0 sweep on the RTX 5090 (the fastest
// candidate within 2 % at each measured T, seams at the last measured T).
Bf16Launch select_bf16_n1152_k4304(std::int32_t tokens) {
    if (tokens <= 8) return select_bf16_general_launch(tokens);
    if (tokens <= 256)
        return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaTailMmaSchedule<64, 32, 32, 16, 3>, 4304>>; // tail_r64t32s3
    if (tokens <= 512)
        return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaTailMmaSchedule<64, 64, 32, 32, 3>, 4304>>; // tail_r64t64s3
    if (tokens <= 1152)
        return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaTailMmaSchedule<64, 128, 32, 32, 2>, 4304>>; // tail_r64t128s2
    if (tokens <= 1536)
        return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaTailMmaSchedule<64, 64, 32, 32, 3>, 4304>>; // tail_r64t64s3
    if (tokens <= 2304)
        return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaTailMmaSchedule<128, 128, 64, 32, 3, 1, Bf16MmaRaster::Grouped, 8>, 4304>>; // tail_r128t128s3_w64x32_group8
    if (tokens <= 3072)
        return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaTailMmaSchedule<64, 128, 32, 32, 2>, 4304>>; // tail_r64t128s2
    if (tokens <= 6144)
        return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaTailMmaSchedule<128, 128, 64, 32, 3, 1, Bf16MmaRaster::RowFast>, 4304>>; // tail_r128t128s3_w64x32_rows
    if (tokens <= 8192)
        return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaTailMmaSchedule<64, 128, 32, 32, 2>, 4304>>; // tail_r64t128s2
    if (tokens <= 49152)
        return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaTailMmaSchedule<128, 128, 32, 64, 3, 1, Bf16MmaRaster::RowFast>, 4304>>; // tail_r128t128s3_w32x64_rows
    return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaTailMmaSchedule<128, 128, 64, 32, 3, 1, Bf16MmaRaster::RowFast>, 4304>>; // tail_r128t128s3_w64x32_rows
}
} // namespace infernix::ops::detail
