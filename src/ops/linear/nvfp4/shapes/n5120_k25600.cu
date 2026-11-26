// A DSpine drafter's feature projection (five target taps) [5120,25600] in NVFP4 (block_scale_k16_m128x4_v1: the 27B drafter's NVFP4 recipe,
// tools/convert/official_recipes.py qwen3_8_27b_dspine_nvfp4), 16-bit activations at every width (the drafter's Uses
// are A16Only): the [5120,6144] output projection's A16 schedules, its block rows and, wider, the prefill's context columns. No A4 route and no column-invariant
// one (a drafter's projections need neither: they change acceptance, never output).
#include "ops/linear/nvfp4/nvfp4_shapes.h"
#include "ops/linear/nvfp4/nvfp4_launch.cuh"
#include "ops/linear/nvfp4/nvfp4_instances.cuh"

namespace infernix::ops::detail {
namespace {
using Geometry = Nvfp4Geometry<5120, 25600>;
using Gemv =
    Nvfp4A16GemvSchedule<8, 2, 16, 4, Nvfp4ScaleAccess::StagedRaw, Nvfp4CodeCache::Default, 2>;
using C2 = Nvfp4A16SimtSchedule<4, 1, 2, 16, 2, 1, Nvfp4SimtActivationAccess::TokenPacked,
                                Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                                Nvfp4SimtBlockOrder::RowsContiguous, 1>;

Nvfp4Launch select_a16(int tokens) {
    if (tokens == 1) return nvfp4_linear_a16_gemv<Geometry, Gemv>;
    if (tokens <= 2) return nvfp4_linear_a16_simt<Geometry, 2, C2, true>;
    if (tokens <= 8) return nvfp4_linear_a16_sliced_k<Geometry, Nvfp4SlicedInstance<8, 8, 2>>;
    if (tokens <= 16) return nvfp4_linear_a16_sliced_k<Geometry, Nvfp4SlicedInstance<16, 8, 2>>;
    if (tokens <= 24) return nvfp4_linear_a16_sliced_k<Geometry, Nvfp4SlicedInstance<32, 8, 1>>;
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

const Nvfp4LinearShape kNvfp4N5120K25600{5120, 25600, launch_a16, nullptr, uses_a4};
} // namespace infernix::ops::detail
