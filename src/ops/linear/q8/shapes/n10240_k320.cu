#include "ops/linear/q8/q8_shapes.h"
#include "ops/linear/q8/q8_stream_launch.cuh"

namespace ninfer::ops::detail {
namespace {
// Qwen3.8-Flash-Next hyper-connection up (K padded to 384): the register-streamed route up to 8 columns (8-24 %
// faster than SIMT), the MMA tiles beyond (M1 sweep). K1: 4 row(s) per warp, x staged in shared memory per
// pass.
template <int MaxCols, int ColsPerPass>
using Stream = Q8A16StreamSchedule<320, 4, 1, 8, MaxCols, ColsPerPass, Q8StreamX::Shared, 2>;
} // namespace

Q8Launch select_q8_n10240_k320(std::int32_t tokens) {
    if (tokens <= 4) return launch_q8_a16_stream_linear<Stream<4, 4>>;
    if (tokens <= 8) return launch_q8_a16_stream_linear<Stream<8, 8>>;
    if (tokens <= 64) return launch_q8_a16_mma_r32_t64;
    if (tokens <= 96) return launch_q8_a16_mma_r32_t96;
    return launch_q8_a16_mma_r32_t128;
}

} // namespace ninfer::ops::detail
