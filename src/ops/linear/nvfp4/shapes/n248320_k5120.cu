// The Qwen3.8-27B output head [248320,5120] as NVIDIA stores it in NVFP4 (nvfp4_mul in
// block_scale_k16_m128x4_v1: NVIDIA's NVFP4-QAD 27B, bit-exact): FP32 or BF16 logits by out's dtype, 16-bit
// activations at every width (the head's Use is A16Only). Its 715 MB of codes and scales are 56 % of the
// row-scaled FP8 head's 1.27 GB, the bytes a decode or verification call streams.
//
// The column-invariant route (A16ColumnInvariant) streams the matrix once per call through the sliced-K tiles with
// eight K-warps at every width, which reduce K in one order whatever the call's width (the MMA's columns are
// independent), wider calls as 64-column slices. The A16 route (linear_topk's grouped selection on the default
// route, CausalScoring) takes the same tiles through 16 columns and above that the K = 5120 schedule of the 34816-row
// gate/up (four K-warps through 32 columns, then the tensor-core GEMM): the eight-K-warp tiles took 1.08, 1.87 and
// 2.49 ms at 24, 33 and 64 columns against the FP8 head's fused 0.89, 0.90 and 1.13 ms (linear_topk, 2026-11-19).
#include "ops/linear/nvfp4/nvfp4_shapes.h"
#include "ops/linear/nvfp4/nvfp4_launch.cuh"
#include "ops/linear/nvfp4/nvfp4_instances.cuh"
#include "ops/linear/common/output.cuh"

namespace infernix::ops::detail {
namespace {
using Geometry = Nvfp4Geometry<248320, 5120>;
constexpr int kSlicedTokens = 64;

template <class Instance>
void sliced(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    static_assert(Geometry::kInputRows % Instance::kBlockK == 0);
    with_linear_output(out, Geometry::kOutputRows, [&](auto output) {
        launch_nvfp4_a16_sliced_k_mma<Nvfp4ScheduleInstance<Instance, Geometry::kInputRows>>(
            nvfp4_a16_operands(x, w), output, LinearIdentityEpilogue{}, stream);
    });
}

template <class Schedule>
void mma(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    with_linear_output(out, Geometry::kOutputRows, [&](auto output) {
        launch_nvfp4_a16_mma<Nvfp4ScheduleInstance<Schedule, Geometry::kInputRows>>(
            nvfp4_a16_operands(x, w), output, LinearIdentityEpilogue{}, stream);
    });
}

// Eight K-warps at every width (a column's reduction order); the stage count only pipelines the K slices, two
// where both fit in shared memory (the 64-token tile's 70 KB stage fits once).
void launch_invariant(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const int tokens = x.ne[1];
    if (tokens <= 8) return sliced<Nvfp4SlicedInstance<8, 8, 2>>(x, w, out, stream);
    if (tokens <= 16) return sliced<Nvfp4SlicedInstance<16, 8, 2>>(x, w, out, stream);
    if (tokens <= 32) return sliced<Nvfp4SlicedInstance<32, 8, 2>>(x, w, out, stream);
    sliced<Nvfp4SlicedInstance<kSlicedTokens, 8, 1>>(x, w, out, stream);
}

void launch_a16(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const int tokens = x.ne[1];
    if (tokens <= 16) return launch_invariant(x, w, out, stream);
    if (tokens <= 24) return sliced<Nvfp4SlicedInstance<32, 4, 1>>(x, w, out, stream);
    if (tokens <= 32) return sliced<Nvfp4SlicedInstance<32, 4, 2>>(x, w, out, stream);
    if (tokens <= 64) return mma<Nvfp4A16MmaSchedule<32, 64, 128, 16, 16, 1, 3>>(x, w, out, stream);
    if (tokens <= 96) return mma<Nvfp4A16MmaSchedule<64, 96, 128, 64, 16, 1, 2>>(x, w, out, stream);
    mma<Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>(x, w, out, stream);
}

bool uses_a4(std::int32_t, std::int32_t) { return false; }
} // namespace

const Nvfp4LinearShape kNvfp4N248320K5120{248320, 5120, launch_a16, nullptr, uses_a4, launch_invariant, true};
} // namespace infernix::ops::detail
