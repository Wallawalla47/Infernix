#include "ops/linear/q8/q8_shapes.h"
#include "ops/linear/q8/q8_instance_launch.cuh"

namespace infernix::ops::detail {
namespace {
using Geometry = Q8N2304K9216;

// Inkling 2 Mini's DFlash2 feature projection (four taps of 2304), the schedule of Qwen3.8-Flash-Next's
// (n2560_k12800.cu): 144 CTAs of one 16-row tile stream the 21 MB weight, each double-buffering its stages, 8 K
// warps through 32 columns (9216 is 18 slices of 512), 4 beyond. Its calls are a commit's columns (8 a row, up to
// 64) and a prompt chunk's. Up to 32 columns a column's bits do not depend on how many share the call.
template <int Columns, int KWarps>
using Sliced = Q8A16SlicedKMmaSchedule<Columns, KWarps, 2, KWarps == 8 ? 2 : 3, Q8ScaleAccess::Shared, Cache::ca,
                                       Cache::cg, Q8ActivationStage::ActiveOnly>;
} // namespace

Q8Launch select_q8_n2304_k9216(std::int32_t tokens) {
    if (tokens <= 8) return launch_q8_a16_sliced<Geometry, 8, Sliced<8, 8>>;
    if (tokens <= 16) return launch_q8_a16_sliced<Geometry, 16, Sliced<16, 8>>;
    if (tokens <= 24) return launch_q8_a16_sliced<Geometry, 24, Sliced<24, 8>>;
    if (tokens <= 32) return launch_q8_a16_sliced<Geometry, 32, Sliced<32, 8>>;
    if (tokens <= 40) return launch_q8_a16_sliced<Geometry, 40, Sliced<40, 4>>;
    if (tokens <= 48) return launch_q8_a16_sliced<Geometry, 48, Sliced<48, 4>>;
    if (tokens <= 56) return launch_q8_a16_sliced<Geometry, 56, Sliced<56, 4>>;
    if (tokens <= 96) return launch_q8_a16_sliced<Geometry, 64, Sliced<64, 4>>;
    if (tokens <= 384) return launch_q8_a16_mma_r32_t64;
    return launch_q8_a16_wide_t128;
}

} // namespace infernix::ops::detail
