#include "ops/linear/bf16_p12/bf16_p12_routes.h"

#include "ops/linear/bf16_p12/bf16_p12_stream.cuh"
#include "ops/linear/bf16_p12/bf16_p12_tma.cuh"

namespace infernix::ops::detail {

// Qwen3.8-Flash-Next: the GDN query, key, value and z projections (recipe A), T = a call's columns. Against the
// BF16 route (cold-L2 medians): 1.26-1.27x at T = 1-4, 1.13-1.19x at 5-16, 1.08-1.10x at 17-32; 0.93-0.96x at
// 33-48, 0.87x at 49-64, 0.67-0.91x at 65-1024 and 0.92-0.98x beyond. Up to T = 8 the skinny GEMV, which gives the
// registered BF16 route's bits (its skinny GEMV), though the stream MMA is 5 % faster at T = 8.
P12Launch select_bf16_p12_n16384_k2560(std::int32_t tokens) {
    if (tokens <= 4) return bf16_p12_skinny_launch<8, 2, 1>(tokens);
    if (tokens <= 8) return bf16_p12_skinny_launch<2, 1, 2>(tokens);
    if (tokens <= 16) return launch_bf16_p12_stream<P12StreamSchedule<1, 16, 8, 1, 2, 3>>;
    if (tokens <= 32) return launch_bf16_p12_stream<P12StreamSchedule<1, 32, 8, 1, 2, 3>>;
    if (tokens <= 48) return launch_bf16_p12_stream<P12StreamSchedule<1, 48, 8, 1, 2, 3>>;
    if (tokens <= 64) return launch_bf16_p12_stream<P12StreamSchedule<1, 64, 8, 1, 2, 3>>;
    if (tokens <= 384) return launch_bf16_p12_tma<P12TmaSchedule<Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3>, 4, 2, 2>>;
    return launch_bf16_p12_tma<
        P12TmaSchedule<Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3, 1, Bf16MmaRaster::RowFast>, 4, 2, 2>>;
}

} // namespace infernix::ops::detail
