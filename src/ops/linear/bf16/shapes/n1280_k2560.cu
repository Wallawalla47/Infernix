#include "ops/linear/bf16/bf16_instances.cuh"
#include "ops/linear/bf16/bf16_shapes.h"
#include "ops/linear/bf16/bf16_launch.cuh"

namespace infernix::ops::detail {
// Qwen3.8-Flash-Next: the shared expert's gate and up projections (recipe A), T = a call's columns.
// T <= 8 runs one column-invariant route at every width, so a column's bits do not depend on how
// many columns share the call (C = 1 and C = 2 give the same greedy text). Wider calls take the
// fastest measured route per interval (bench/ops/bf16_flash_next_sweep.cu, RTX 5090, 2 % tolerance,
// seams at the last measured T).
Bf16Launch select_bf16_n1280_k2560(std::int32_t tokens) {
    if (tokens <= 8) return select_bf16_general_launch(tokens); // skinny GEMV (56.5 us over T = 1..8)
    if (tokens <= 64) return launch_bf16_sliced_k_mma<Bf16A16SlicedR32T16W4>; // sliced_r32t16w4
    if (tokens <= 128) return launch_bf16_mma<Bf16A16MmaR32T32K256S3>; // mma_r32t32k256s3
    if (tokens <= 384) return launch_bf16_tma_mma<Bf16A16TmaMmaSchedule<64, 32, 64, 32, 16, 3>>; // tma_r64t32k64s3
    if (tokens <= 512) return launch_bf16_tma_mma<Bf16A16TmaMmaSchedule<64, 64, 64, 32, 32, 3>>; // tma_r64t64k64s3
    if (tokens <= 1024) return launch_bf16_tma_mma<Bf16A16TmaMmaSchedule<64, 128, 64, 32, 32, 2>>; // tma_r64t128k64s2
    if (tokens <= 1536) return launch_bf16_tma_mma<Bf16A16TmaTailMmaSchedule<64, 64, 32, 32, 3>>; // tail_r64t64s3
    if (tokens <= 2048) return launch_bf16_tma_mma<Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3>>; // tma_r128t128k64s3_w64x32
    if (tokens <= 3072) return launch_bf16_tma_mma<Bf16A16TmaMmaSchedule<64, 128, 64, 32, 32, 2>>; // tma_r64t128k64s2
    return launch_bf16_tma_mma<Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3, 1, Bf16MmaRaster::RowFast>>; // tma_r128t128k64s3_w64x32_rows
}
} // namespace infernix::ops::detail
