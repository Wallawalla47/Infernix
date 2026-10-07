#include "ops/linear/bf16/bf16_instances.cuh"
#include "ops/linear/bf16/bf16_shapes.h"
#include "ops/linear/bf16/bf16_launch.cuh"

namespace infernix::ops::detail {
// Vision patch embedding, T = raw patches. Routes per token interval from the V0 sweep on the RTX 5090 (the fastest
// candidate within 2 % at each measured T, seams at the last measured T).
Bf16Launch select_bf16_n1152_k1536(std::int32_t tokens) {
    if (tokens <= 8) return select_bf16_general_launch(tokens);
    if (tokens <= 96)
        return launch_bf16_sliced_k_mma<Bf16ScheduleInstance<Bf16A16SlicedR32T16W4, 1536>>; // sliced_r32t16w4
    if (tokens <= 128)
        return launch_bf16_mma<Bf16ScheduleInstance<Bf16A16MmaR32T32K128S3, 1536>>; // mma_r32t32k128s3
    if (tokens <= 384)
        return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<64, 32, 64, 32, 16, 3>, 1536>>; // tma_r64t32k64s3
    if (tokens <= 512)
        return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<64, 64, 64, 32, 32, 3>, 1536>>; // tma_r64t64k64s3
    if (tokens <= 1152)
        return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<64, 128, 64, 32, 32, 2>, 1536>>; // tma_r64t128k64s2
    if (tokens <= 1536)
        return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<64, 64, 64, 32, 32, 3>, 1536>>; // tma_r64t64k64s3
    if (tokens <= 2304)
        return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3>, 1536>>; // tma_r128t128k64s3_w64x32
    if (tokens <= 3072)
        return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<64, 128, 64, 32, 32, 2>, 1536>>; // tma_r64t128k64s2
    if (tokens <= 6144)
        return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaTailMmaSchedule<64, 64, 32, 32, 3>, 1536>>; // tail_r64t64s3
    if (tokens <= 12288)
        return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaTailMmaSchedule<64, 128, 32, 32, 2>, 1536>>; // tail_r64t128s2
    return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3, 1, Bf16MmaRaster::RowFast>, 1536>>; // tma_r128t128k64s3_w64x32_rows
}
} // namespace infernix::ops::detail
