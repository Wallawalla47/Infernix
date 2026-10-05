#include "ops/linear/bf16/bf16_instances.cuh"
#include "ops/linear/bf16/bf16_shapes.h"
#include "ops/linear/bf16/bf16_launch.cuh"

namespace ninfer::ops::detail {
// Vision MLP fc1, T = raw patches; 4304 rows end in a partial row tile. Routes per token interval from the V0 sweep on the RTX 5090 (the fastest
// candidate within 2 % at each measured T, seams at the last measured T).
Bf16Launch select_bf16_n4304_k1152(std::int32_t tokens) {
    if (tokens <= 8) return select_bf16_general_launch(tokens);
    if (tokens <= 128)
        return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaTailMmaSchedule<64, 32, 32, 16, 3>, 1152>>; // tail_r64t32s3
    if (tokens <= 256)
        return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaTailMmaSchedule<64, 128, 32, 32, 2>, 1152>>; // tail_r64t128s2
    if (tokens <= 384)
        return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaTailMmaSchedule<64, 64, 32, 32, 3>, 1152>>; // tail_r64t64s3
    if (tokens <= 512)
        return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaTailMmaSchedule<128, 128, 32, 64, 3, 1, Bf16MmaRaster::RowFast>, 1152>>; // tail_r128t128s3_w32x64_rows
    if (tokens <= 1024)
        return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaTailMmaSchedule<64, 64, 32, 32, 3>, 1152>>; // tail_r64t64s3
    if (tokens <= 1152)
        return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaTailMmaSchedule<128, 128, 64, 32, 3, 1, Bf16MmaRaster::RowFast>, 1152>>; // tail_r128t128s3_w64x32_rows
    if (tokens <= 6144)
        return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaTailMmaSchedule<64, 128, 32, 32, 2>, 1152>>; // tail_r64t128s2
    return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaTailMmaSchedule<128, 128, 64, 32, 3, 1, Bf16MmaRaster::RowFast>, 1152>>; // tail_r128t128s3_w64x32_rows
}
} // namespace ninfer::ops::detail
