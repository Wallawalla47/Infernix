#include "ops/linear/q8/q8_shapes.h"
#include "ops/linear/q8/q8_instance_launch.cuh"

namespace infernix::ops::detail {
namespace {
using Geometry = Q8N2560K12800;

// Qwen3.8-Flash-Next's DFlash2 feature projection: 160 CTAs of one 16-row tile stream the 35 MB weight, so
// each double-buffers its stages. 8 K warps through 32 columns, 4 beyond (sweep of 2026-11-01, RTX 5090,
// cold, graph-timed: at 8 columns 25.2 us against 45.7 for the runtime-shape SIMT route, at 64 68.7 against
// 122 for the MMA tile). Up to 32 columns a column's bits do not depend on how many share the call
// (verification's column invariance needs 8).
template <int Columns, int KWarps>
using Sliced = Q8A16SlicedKMmaSchedule<Columns, KWarps, 2, KWarps == 8 ? 2 : 3, Q8ScaleAccess::Shared, Cache::ca,
                                       Cache::cg, Q8ActivationStage::ActiveOnly>;
} // namespace

Q8Launch select_q8_n2560_k12800(std::int32_t tokens) {
    if (tokens <= 8) return launch_q8_a16_sliced<Geometry, 8, Sliced<8, 8>>;
    if (tokens <= 16) return launch_q8_a16_sliced<Geometry, 16, Sliced<16, 8>>;
    if (tokens <= 24) return launch_q8_a16_sliced<Geometry, 24, Sliced<24, 8>>;
    if (tokens <= 32) return launch_q8_a16_sliced<Geometry, 32, Sliced<32, 8>>;
    if (tokens <= 40) return launch_q8_a16_sliced<Geometry, 40, Sliced<40, 4>>;
    if (tokens <= 48) return launch_q8_a16_sliced<Geometry, 48, Sliced<48, 4>>;
    if (tokens <= 56) return launch_q8_a16_sliced<Geometry, 56, Sliced<56, 4>>;
    // 64-column slices up to 96 columns (72: 101 us, 96: 105, against 123.5), the MMA tiles beyond; every
    // MMA tile accumulates each output over K in the same order, so they differ only in speed (r32_t64
    // to 384 columns, r64_t128 beyond: 2,048 columns 712 us against 844 for r32_t128).
    if (tokens <= 96) return launch_q8_a16_sliced<Geometry, 64, Sliced<64, 4>>;
    if (tokens <= 384) return launch_q8_a16_mma_r32_t64;
    return launch_q8_a16_mma_r64_t128;
}

} // namespace infernix::ops::detail
