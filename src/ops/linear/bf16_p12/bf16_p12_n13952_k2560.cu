#include "ops/linear/bf16_p12/bf16_p12_routes.h"

#include "ops/linear/bf16_p12/bf16_p12_stream.cuh"
#include "ops/linear/bf16_p12/bf16_p12_tma.cuh"

namespace infernix::ops::detail {

// Qwen3.8-Flash-Next: the QSA query, gate, key, value and index projections (recipe A), T = a call's columns.
// Against the BF16 route (cold-L2 medians): 1.18-1.29x at T = 1-4, 1.09-1.10x at 5-7, 1.00-1.01x at 8-12; 0.97x
// at 16, 0.88-0.89x at 17-32, 0.72-0.76x at 33-64, 0.63-0.91x at 65-768 and 0.90-0.99x beyond. Up to T = 8 the
// skinny GEMV, which gives the registered BF16 route's bits (its skinny GEMV).
P12Launch select_bf16_p12_n13952_k2560(std::int32_t tokens) {
    if (tokens <= 4) return bf16_p12_skinny_launch<2, 1, 2>(tokens);
    if (tokens <= 7) return bf16_p12_skinny_launch<8, 2, 1>(tokens);
    if (tokens <= 8) return bf16_p12_skinny_launch<2, 1, 2>(tokens);
    if (tokens <= 16) return launch_bf16_p12_stream<P12StreamSchedule<1, 16, 2, 4, 2, 3>>;
    if (tokens <= 32) return launch_bf16_p12_stream<P12StreamSchedule<1, 32, 2, 4, 2, 2>>;
    if (tokens <= 48) return launch_bf16_p12_stream<P12StreamSchedule<1, 48, 8, 1, 2, 3>>;
    if (tokens <= 64) return launch_bf16_p12_stream<P12StreamSchedule<1, 64, 8, 1, 2, 3>>;
    if (tokens <= 128) return launch_bf16_p12_tma<P12TmaSchedule<Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3>, 4, 2, 2>>;
    if (tokens <= 192) return launch_bf16_p12_tma<P12TmaSchedule<Bf16A16TmaMmaSchedule<128, 64, 64, 32, 32, 4>, 4, 2, 3>>;
    if (tokens <= 256) return launch_bf16_p12_tma<P12TmaSchedule<Bf16A16TmaMmaSchedule<64, 128, 64, 32, 32, 2>, 4, 2, 3>>;
    return launch_bf16_p12_tma<
        P12TmaSchedule<Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3, 1, Bf16MmaRaster::RowFast>, 4, 2, 2>>;
}

} // namespace infernix::ops::detail
