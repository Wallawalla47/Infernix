#include "ops/linear/bf16_p12/bf16_p12_routes.h"

#include "ops/linear/bf16_p12/bf16_p12_direct.cuh"
#include "ops/linear/bf16_p12/bf16_p12_stream.cuh"
#include "ops/linear/bf16_p12/bf16_p12_tma.cuh"

namespace infernix::ops::detail {

// Qwen3.8-Flash-Next: the GDN and QSA output projections (recipe A), T = a call's columns. Against the BF16 route
// (cold-L2 medians): 1.00-1.06x at T = 1-8; 0.93x at 16, 0.87-0.94x at 17-32, 0.60-0.66x at 33-128, 0.74-0.82x at
// 129-512, 0.88-0.92x beyond.
P12Launch select_bf16_p12_n2560_k6144(std::int32_t tokens) {
    if (tokens <= 8) return launch_bf16_p12_direct<P12DirectSchedule<1, 1, 1, 16, 2>>;
    if (tokens <= 16) return launch_bf16_p12_direct<P12DirectSchedule<1, 2, 1, 16, 2>>;
    if (tokens <= 32) return launch_bf16_p12_direct<P12DirectSchedule<1, 4, 1, 8, 2>>;
    if (tokens <= 128) return launch_bf16_p12_stream<P12StreamSchedule<1, 32, 2, 4, 2, 2>>;
    if (tokens <= 512) return launch_bf16_p12_tma<P12TmaSchedule<Bf16A16TmaMmaSchedule<64, 128, 64, 32, 32, 2>, 4, 2, 3>>;
    return launch_bf16_p12_tma<
        P12TmaSchedule<Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3, 1, Bf16MmaRaster::RowFast>, 4, 2, 2>>;
}

} // namespace infernix::ops::detail
