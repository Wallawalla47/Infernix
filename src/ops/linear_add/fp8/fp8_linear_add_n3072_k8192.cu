// Qwen3.5-122B-A10B residual projections (Ornith-1.5-122B-NVFP4): the attention output projection
// (32 heads of 256) and the GDN output projection (64 value heads of 128), both fp8_e4m3fn_row_f32
// [3072,8192] with FP32 row multipliers, added into the BF16 residual with one rounding. Prefill
// encodes the activations once with the projection's static input scale (static A8); decode and
// verification keep BF16 activations (A16, column-invariant on the exact route). The schedule
// families are the long-K [5120,17408] problem's (fp8_linear_add_*.cu): 3,072 rows give few output
// tiles, so the TMA tiles split K over an underfilled wave.
#include "ops/linear/common/epilogue.cuh"
#include "ops/linear/fp8/fp8_a16_gemv.cuh"
#include "ops/linear/fp8/fp8_a16_ring_mma.cuh"
#include "ops/linear/fp8/fp8_instances.cuh"
#include "ops/linear/fp8/fp8_template_launch.cuh"
#include "ops/linear_add/fp8/fp8_linear_add_plan.h"

namespace infernix::ops::detail {
namespace {

constexpr int kRows = 3072;
constexpr int kK    = 8192;

// This Op admits dense, even-width BF16 residuals. Adjacent MMA rows share one load.
struct ResidualPairEpilogue : LinearResidualAddEpilogue {
    __device__ __forceinline__ float2 apply_row_pair(int row, int token, float2 value) const {
        const auto* pointer =
            residual.data + static_cast<std::int64_t>(token) * residual.leading_dim + row;
        const float2 add = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(pointer));
        return make_float2(value.x + add.x, value.y + add.y);
    }
};

using Tma32x64 = Fp8A8TmaMmaSchedule<32, 64, 128, 1, 2, 3, 2>;
using Small    = Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<64, 128, 128, 2, 4, 3, 1>, 170, 4, 8>;
using Mid      = Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 128, 128, 2, 4, 3, 1>, 170, 4, 8>;
using Wide     = Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<192, 128, 128, 3, 4, 2, 1>, 170, 4, 8>;
using Bulk     = Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, 170, 4, 8>;

struct Destination {
    LinearBf16Output output;
    LinearResidualAddEpilogue epilogue;
};

Destination destination_of(Tensor& residual) {
    auto* data = static_cast<__nv_bfloat16*>(residual.data);
    return {{data, kRows}, {{data, kRows}}};
}

template <class Schedule>
void sliced(const Tensor& x, const Weight& weight, Tensor& residual, cudaStream_t stream) {
    const Destination d = destination_of(residual);
    launch_fp8_a16_sliced_k_mma<Fp8ScheduleInstance<Schedule, kK>>(fp8_a16_operands<float>(x, weight),
                                                                   d.output, d.epilogue, stream);
}

} // namespace

std::size_t fp8_linear_add_n3072_partial_capacity_bytes(std::int32_t max_tokens) {
    // Every split-K band can leave a tail of at most half a wave: 24 to 96 output tiles.
    if (max_tokens > 384) return Bulk::kPartialBytes;
    if (max_tokens > 256) return Wide::kPartialBytes;
    if (max_tokens > 128) return Mid::kPartialBytes;
    return max_tokens > 64 ? Small::kPartialBytes : 0;
}

void fp8_linear_add_n3072_invariant_launch(const Tensor& x, const Weight& weight,
                                           Tensor& residual, cudaStream_t stream) {
    // Eight K-warps over the same K slices, reduced in the same order, at every width; the MMA's
    // columns are independent, and wider calls run as 64-column slices.
    if (x.ne[1] <= 8) return sliced<Fp8SlicedInstance<8, 8, 2>>(x, weight, residual, stream);
    if (x.ne[1] <= 16) return sliced<Fp8SlicedInstance<16, 8, 2>>(x, weight, residual, stream);
    sliced<Fp8SlicedInstance<64, 8, 1>>(x, weight, residual, stream);
}

void fp8_linear_add_n3072_a16_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                     cudaStream_t stream) {
    const Destination d = destination_of(residual);
    const auto operands = fp8_a16_operands<float>(x, weight);
    const int tokens    = x.ne[1];
    if (tokens == 1) {
        using Schedule = Fp8A16GemvSchedule<8, 2, 8, 4, Fp8CodeCache::Default, 2, 2>;
        return launch_fp8_a16_gemv<Fp8ScheduleInstance<Schedule, kK>>(operands, d.output,
                                                                      d.epilogue, stream);
    }
    if (tokens <= 8) return sliced<Fp8SlicedInstance<8, 8, 2>>(x, weight, residual, stream);
    if (tokens <= 16) return sliced<Fp8SlicedInstance<16, 8, 2>>(x, weight, residual, stream);
    const auto ring = [&]<class Schedule>() {
        launch_fp8_a16_ring_mma<Fp8ScheduleInstance<Schedule, kK>>(operands, d.output, d.epilogue,
                                                                  stream);
    };
    if (tokens <= 64) return ring.template operator()<Fp8A16RingT32R64>();
    if (tokens <= 128 || (tokens > 256 && tokens <= 384))
        return ring.template operator()<Fp8A16RingT32R128>();
    if (tokens <= 192) return ring.template operator()<Fp8A16RingT32R64S3>();
    ring.template operator()<Fp8A16RingT64R128>();
}

void fp8_linear_add_n3072_a8_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                    WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope         = workspace.scope();
    const auto scratch = allocate_fp8_a8_workspace(
        workspace, x.ne[1], weight.k, fp8_linear_add_n3072_partial_capacity_bytes(x.ne[1]));
    launch_fp8_a8_quantize(x, weight, scratch, stream);
    auto* data = static_cast<__nv_bfloat16*>(residual.data);
    const LinearBf16Output output{data, kRows};
    const ResidualPairEpilogue epilogue{{{data, kRows}}};
    const auto operands = fp8_a8_operands<float>(weight, scratch, x.ne[1]);
    const auto launch   = [&]<class Schedule>() {
        launch_fp8_a8_tma_mma<Fp8ScheduleInstance<Schedule, kK>>(operands, output, epilogue, stream,
                                                                 scratch.partials);
    };
    const int tokens = x.ne[1];
    if (tokens <= 64) return launch.template operator()<Tma32x64>();
    if (tokens <= 128) return launch.template operator()<Small>();
    if (tokens <= 256) return launch.template operator()<Mid>();
    if (tokens <= 384 || (tokens > 512 && tokens <= 768)) return launch.template operator()<Wide>();
    launch.template operator()<Bulk>();
}

} // namespace infernix::ops::detail
