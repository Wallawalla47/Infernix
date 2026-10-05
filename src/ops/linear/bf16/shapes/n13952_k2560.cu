#include "ops/linear/bf16/bf16_instances.cuh"
#include "ops/linear/bf16/bf16_shapes.h"
#include "ops/linear/bf16/bf16_launch.cuh"

namespace ninfer::ops::detail {
// Qwen3.8-Flash-Next QSA projection group (query, gate, key, value and the index projections), T = a
// call's columns. Routes per token interval from the tile sweep on the RTX 5090 (the fastest
// candidate within 2 % at each measured T, seams at the last measured T); T <= 8 keeps the
// runtime-shape fallback's skinny GEMV, so decode and verification outputs are unchanged.
Bf16Launch select_bf16_n13952_k2560(std::int32_t tokens) {
    if (tokens <= 8) return select_bf16_general_launch(tokens);
    if (tokens <= 32)
        return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<64, 32, 64, 32, 16, 3>, 2560>>; // tma_r64t32k64s3
    if (tokens <= 64)
        return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<128, 64, 64, 32, 32, 4, 1, Bf16MmaRaster::RowFast>, 2560>>; // tma_r128t64k64s4_rows
    if (tokens <= 96)
        return launch_bf16_mma<Bf16ScheduleInstance<Bf16A16MmaR64T32K64S3, 2560>>; // mma_r64t32k64s3
    if (tokens <= 128)
        return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaTailMmaSchedule<64, 64, 32, 32, 3>, 2560>>; // tail_r64t64s3
    if (tokens <= 192)
        return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<128, 64, 64, 32, 32, 4>, 2560>>; // tma_r128t64k64s4
    if (tokens <= 256)
        return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaTailMmaSchedule<64, 128, 32, 32, 2>, 2560>>; // tail_r64t128s2
    if (tokens <= 768)
        return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3>, 2560>>; // tma_r128t128k64s3_w64x32
    if (tokens <= 1024)
        return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<64, 128, 64, 32, 32, 2, 1, Bf16MmaRaster::RowFast>, 2560>>; // tma_r64t128k64s2_rows
    return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3>, 2560>>; // tma_r128t128k64s3_w64x32
}
} // namespace ninfer::ops::detail
