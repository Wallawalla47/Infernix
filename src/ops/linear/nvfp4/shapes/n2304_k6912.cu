// Inkling 2 Mini's DFlash2 drafter's MLP down [2304,6912] in NVFP4 (block_scale_k16_m128x4_v1; tools/convert recipe
// inkling2_mini_dflash2_nvfp4_mlp), 16-bit activations at every width (the drafter's Uses are A16Only;
// linear_dynamic_grouped_conv_add's NVFP4 route projects with it, then finishes): the [5120,4096] drafter shape's A16
// sliced-K and MMA ladder, with at most 4 K-warps in its sliced-K schedules (a block K of 256: 6912 = 27 x 256 is no
// multiple of 8 warps' 512). No A4 route and no column-invariant one.
#include "ops/linear/nvfp4/nvfp4_shapes.h"
#include "ops/linear/nvfp4/nvfp4_launch.cuh"
#include "ops/linear/nvfp4/nvfp4_instances.cuh"

namespace infernix::ops::detail {
namespace {
using Geometry = Nvfp4Geometry<2304, 6912>;
Nvfp4Launch select_a16(int tokens) {
    // The 1-2 column GEMV and SIMT schedules read K in phases of 512 (2304 and 6912 are not multiples): every width
    // to 8 takes the 8-token sliced-K tile (the drafter projects 8 columns a row).
    if (tokens <= 8) return nvfp4_linear_a16_sliced_k<Geometry, Nvfp4SlicedInstance<8, 4, 2>>;
    if (tokens <= 16) return nvfp4_linear_a16_sliced_k<Geometry, Nvfp4SlicedInstance<16, 4, 2>>;
    if (tokens <= 24) return nvfp4_linear_a16_sliced_k<Geometry, Nvfp4SlicedInstance<32, 4, 1>>;
    if (tokens <= 32) return nvfp4_linear_a16_sliced_k<Geometry, Nvfp4SlicedInstance<32, 4, 2>>;
    if (tokens <= 48) return nvfp4_linear_a16_sliced_k<Geometry, Nvfp4SlicedInstance<32, 2, 2>>;
    if (tokens <= 64) return nvfp4_linear_a16_sliced_k<Geometry, Nvfp4SlicedInstance<64, 2, 2>>;
    if (tokens <= 128)
        return nvfp4_linear_a16_mma<Geometry, Nvfp4A16MmaSchedule<64, 64, 128, 32, 16, 2, 1>>;
    return nvfp4_linear_a16_mma<Geometry, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>;
}

void launch_a16(const Tensor& x, const Weight& w, Tensor& y, cudaStream_t stream) {
    select_a16(x.ne[1])(x, w, y, stream);
}

bool uses_a4(std::int32_t, std::int32_t) { return false; }
} // namespace

const Nvfp4LinearShape kNvfp4N2304K6912{2304, 6912, launch_a16, nullptr, uses_a4};
} // namespace infernix::ops::detail
