// Qwen3.5-122B-A10B shared expert gate and up stored per-row FP8 (dealignai's
// Ornith-1.5-122B-UNCENSORED-NVFP4): one fp8_e4m3fn_row_f32 parent [2048,3072] (gate rows, then up
// rows), each row with its FP32 multiplier, BF16 activations (W8A16), out = silu(gate) * up [1024,T].
// 6 MiB of codes: decode and verification calls are latency-bound and take the paired-row SIMT tiles
// and the 8-K-warp sliced-K tiles (128 CTAs); prefill takes the MMA tiles of the [34816,5120] parent
// (fp8_linear_swiglu_a16.cu).
#include "ops/linear/fp8/fp8_instances.cuh"
#include "ops/linear/fp8/fp8_template_launch.cuh"
#include "ops/linear_swiglu/fp8/fp8_linear_swiglu_output.cuh"
#include "ops/linear_swiglu/fp8/fp8_linear_swiglu_plan.h"
#include "ops/linear_swiglu/row_major_mma_epilogue.cuh"

namespace infernix::ops::detail {
namespace {

constexpr int kHidden       = 3072;
constexpr int kIntermediate = 1024;
using Simt = Fp8A16SimtSchedule<4, 2, 16, 4, 1, Fp8SimtActivationAccess::TokenPacked, Fp8CodeCache::Default, 1,
                                Fp8SimtBlockOrder::RowsContiguous, 1>;
static_assert(Simt::kRowsPerWarp == 2 && (kIntermediate % Simt::kWarpsPerCta) == 0);

LinearBf16Output output_of(Tensor& out) { return {static_cast<__nv_bfloat16*>(out.data), kIntermediate}; }

template <int Tokens, int KWarps, int Stages>
void sliced(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    using S = Fp8ScheduleInstance<Fp8SlicedInstance<Tokens, KWarps, Stages>, kHidden>;
    launch_fp8_a16_sliced_k_mma<S>(fp8_a16_operands<float>(x, weight), output_of(out), Fp8SwiGluEpilogue{}, stream,
                                   SwiGluRowMajorMmaRows<S>{});
}

template <class Schedule>
void mma(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    using S = Fp8ScheduleInstance<Schedule, kHidden>;
    launch_fp8_a16_mma<S>(fp8_a16_operands<float>(x, weight), output_of(out), SwiGluRowMajorMmaEpilogue{}, stream,
                          SwiGluRowMajorMmaRows<S>{});
}

} // namespace

void fp8_linear_swiglu_n2048_a16_launch(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const int tokens = x.ne[1];
    if (tokens <= 4) {
        using Rows = Fp8SwiGluRows<Simt::kRowsPerWarp / 2, kIntermediate>;
        return launch_fp8_a16_simt<Fp8ScheduleInstance<Simt, kHidden, 4>>(
            fp8_a16_operands<float>(x, weight), output_of(out), Fp8SwiGluEpilogue{}, stream, Rows{});
    }
    if (tokens <= 8) return sliced<8, 8, 1>(x, weight, out, stream);
    if (tokens <= 16) return sliced<16, 8, 1>(x, weight, out, stream);
    if (tokens <= 32) return sliced<32, 4, 1>(x, weight, out, stream);
    if (tokens <= 64) return mma<Fp8A16MmaSchedule<32, 64, 128, 16, 16, 1, 3>>(x, weight, out, stream);
    if (tokens <= 96) return mma<Fp8A16MmaSchedule<64, 96, 128, 64, 16, 1, 2>>(x, weight, out, stream);
    mma<Fp8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>(x, weight, out, stream);
}

void fp8_linear_swiglu_n2048_invariant_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                              cudaStream_t stream) {
    // Eight K-warps over the same K slices, reduced in the same order, at every width; the MMA's
    // columns are independent, and wider calls run as 64-column slices.
    if (x.ne[1] <= 8) return sliced<8, 8, 1>(x, weight, out, stream);
    if (x.ne[1] <= 16) return sliced<16, 8, 1>(x, weight, out, stream);
    sliced<64, 8, 1>(x, weight, out, stream);
}

} // namespace infernix::ops::detail
