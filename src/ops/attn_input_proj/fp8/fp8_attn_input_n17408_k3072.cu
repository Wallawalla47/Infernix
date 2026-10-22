// Qwen3.5-122B-A10B attention input projection (Ornith-1.5-122B-NVFP4): one fp8_e4m3fn_row_f32
// parent [17408,3072] in physical order query 8192, key 512, output gate 8192, value 512, each row
// with its FP32 weight multiplier. Prefill encodes the activations once with the layer's static
// input scale (static A8); decode and verification keep BF16 activations (A16, column-invariant on
// the exact route). The schedule families and width bands are the [14336,5120] parent's
// (fp8_attn_input_*.cu).
#include "ops/attn_input_proj/fp8/fp8_attn_input_plan.h"
#include "ops/linear/common/output.cuh"
#include "ops/linear/fp8/fp8_a16_gemv.cuh"
#include "ops/linear/fp8/fp8_a16_mma.cuh"
#include "ops/linear/fp8/fp8_instances.cuh"
#include "ops/linear/fp8/fp8_template_launch.cuh"

namespace infernix::ops::detail {
namespace {

constexpr int kHidden = 3072;
using Output          = LinearBf16SegmentedOutput<8192, 512, 8192, 512>;
using Tma64x128       = Fp8A8TmaMmaSchedule<64, 128, 128, 2, 4, 2, 1>;
using Tma64x256       = Fp8A8TmaMmaSchedule<64, 256, 128, 2, 4, 2, 1>;
using Tma96x256       = Fp8A8TmaMmaSchedule<96, 256, 128, 3, 4, 2, 1>;
using Bulk = Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, 170, 4, 8>;

Output output_of(Tensor& q, Tensor& gate, Tensor& k, Tensor& v) {
    return {static_cast<__nv_bfloat16*>(q.data), static_cast<__nv_bfloat16*>(k.data),
            static_cast<__nv_bfloat16*>(gate.data), static_cast<__nv_bfloat16*>(v.data)};
}

template <class Schedule>
void sliced(const Tensor& x, const Weight& weight, Output output, cudaStream_t stream) {
    launch_fp8_a16_sliced_k_mma<Fp8ScheduleInstance<Schedule, kHidden>>(
        fp8_a16_operands<float>(x, weight), output, LinearIdentityEpilogue{}, stream);
}

template <class Schedule>
void gemm(const Tensor& x, const Weight& weight, Output output, cudaStream_t stream) {
    launch_fp8_a16_mma<Fp8ScheduleInstance<Schedule, kHidden>>(fp8_a16_operands<float>(x, weight), output,
                                                               LinearIdentityEpilogue{}, stream);
}

} // namespace

std::size_t fp8_attn_input_n17408_partial_capacity_bytes(std::int32_t max_tokens) {
    // Bulk runs from T = 289; its 68 row tiles leave a split tail (at most half a wave) at three
    // token tiles already (204 tiles: 34 past one wave).
    return max_tokens > 288 ? Bulk::kPartialBytes : 0;
}

void fp8_attn_input_n17408_invariant_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                            Tensor& gate, Tensor& k, Tensor& v,
                                            cudaStream_t stream) {
    // Four K-warps over the same K slices, reduced in the same order, at every width; the MMA's
    // columns are independent, and wider calls run as 64-column slices.
    const Output output = output_of(q, gate, k, v);
    if (x.ne[1] <= 8) return sliced<Fp8SlicedInstance<8, 4, 1>>(x, weight, output, stream);
    if (x.ne[1] <= 16) return sliced<Fp8SlicedInstance<16, 4, 1>>(x, weight, output, stream);
    if (x.ne[1] <= 32) return sliced<Fp8SlicedInstance<32, 4, 1>>(x, weight, output, stream);
    sliced<Fp8SlicedInstance<64, 4, 1>>(x, weight, output, stream);
}

void fp8_attn_input_n17408_a16_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                      Tensor& gate, Tensor& k, Tensor& v, cudaStream_t stream) {
    const Output output = output_of(q, gate, k, v);
    const int tokens    = x.ne[1];
    if (tokens == 1) {
        using Schedule = Fp8A16GemvSchedule<8, 2, 8, 4, Fp8CodeCache::Default, 2, 6>;
        return launch_fp8_a16_gemv<Fp8ScheduleInstance<Schedule, kHidden>>(
            fp8_a16_operands<float>(x, weight), output, LinearIdentityEpilogue{}, stream);
    }
    if (tokens <= 16) return sliced<Fp8SlicedInstance<16, 4, 1>>(x, weight, output, stream);
    if (tokens <= 24) return sliced<Fp8SlicedInstance<32, 4, 2>>(x, weight, output, stream);
    if (tokens <= 32) return sliced<Fp8SlicedInstance<32, 4, 1>>(x, weight, output, stream);
    if (tokens <= 64)
        return gemm<Fp8A16MmaSchedule<32, 64, 128, 16, 16, 1, 3>>(x, weight, output, stream);
    if (tokens <= 80 || (tokens >= 129 && tokens <= 160))
        return gemm<Fp8A16MmaSchedule<32, 80, 128, 32, 16, 1, 3>>(x, weight, output, stream);
    if (tokens <= 96)
        return gemm<Fp8A16MmaSchedule<64, 96, 128, 64, 16, 1, 2>>(x, weight, output, stream);
    gemm<Fp8A16MmaSchedule<64, 128, 64, 32, 16, 2, 2>>(x, weight, output, stream);
}

void fp8_attn_input_n17408_a8_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                     Tensor& gate, Tensor& k, Tensor& v, Fp8A8Workspace workspace,
                                     cudaStream_t stream) {
    launch_fp8_a8_quantize(x, weight, workspace, stream);
    const Output output = output_of(q, gate, k, v);
    const auto operands = fp8_a8_operands<float>(weight, workspace, x.ne[1]);
    const auto launch   = [&]<class Schedule>() {
        using S = Fp8ScheduleInstance<Schedule, kHidden>;
        if constexpr (S::kTmaSwizzle)
            launch_fp8_a8_tma_mma<S>(operands, output, LinearIdentityEpilogue{}, stream,
                                     workspace.partials);
        else
            launch_fp8_a8_mma<S>(operands, output, LinearIdentityEpilogue{}, stream, {},
                                 pdl::Dependency::Programmatic);
    };
    const int tokens = x.ne[1];
    if (tokens <= 32) return launch.template operator()<Fp8A8T32R32K128>();
    if (tokens <= 96) return launch.template operator()<Fp8A8T32R128K128>();
    if (tokens <= 128) return launch.template operator()<Tma64x128>();
    if (tokens <= 192) return launch.template operator()<Tma64x256>();
    // 68 row tiles of 256: three 96-token tiles give 204 CTAs, 1.2 waves on 170 SMs.
    if (tokens <= 288) return launch.template operator()<Tma96x256>();
    launch.template operator()<Bulk>();
}

} // namespace infernix::ops::detail
