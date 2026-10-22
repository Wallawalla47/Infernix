// Qwen3.5-122B-A10B GDN input projection (Ornith-1.5-122B-NVFP4): one fp8_e4m3fn_row_f32 parent
// [20480,3072], query/key/value 12288 (16 key heads and 64 value heads of 128) then z 8192, each
// row with its FP32 weight multiplier. Prefill encodes the activations once with the layer's static
// input scale (static A8); decode and verification keep BF16 activations (A16, column-invariant on
// the exact route). The schedule families and width bands are the [16384,5120] parent's
// (fp8_gdn_input_*.cu).
#include "ops/gdn_input_proj/fp8/fp8_gdn_input_plan.h"
#include "ops/linear/common/output.cuh"
#include "ops/linear/fp8/fp8_a16_gemv.cuh"
#include "ops/linear/fp8/fp8_a16_mma.cuh"
#include "ops/linear/fp8/fp8_a16_simt.cuh"
#include "ops/linear/fp8/fp8_a16_sliced_k_mma.cuh"
#include "ops/linear/fp8/fp8_instances.cuh"
#include "ops/linear/fp8/fp8_template_launch.cuh"

namespace infernix::ops::detail {
namespace {

constexpr int kHidden = 3072;
using Output          = LinearBf16SegmentedOutput<12288, 8192>;
using Tma64x128       = Fp8A8TmaMmaSchedule<64, 128, 128, 2, 4, 2, 1>;
using Tma192x128      = Fp8A8TmaMmaSchedule<192, 128, 128, 3, 4, 2, 1>;
using MidBulk = Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 128, 128, 2, 4, 2, 1>, 170, 4, 8>;
using Bulk    = Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, 170, 4, 8>;

Output output_of(Tensor& qkv, Tensor& z) {
    return {static_cast<__nv_bfloat16*>(qkv.data), static_cast<__nv_bfloat16*>(z.data)};
}

template <class Schedule, int Capacity = 0>
void sliced(const Tensor& x, const Weight& weight, Output output, cudaStream_t stream) {
    launch_fp8_a16_sliced_k_mma<Fp8ScheduleInstance<Schedule, kHidden, Capacity>>(
        fp8_a16_operands<float>(x, weight), output, LinearIdentityEpilogue{}, stream);
}

template <int Capacity>
void small_mma(const Tensor& x, const Weight& weight, Output output, cudaStream_t stream) {
    constexpr int warps = Capacity <= 8 ? 16 : Capacity <= 24 ? 8 : 4;
    sliced<Fp8A16SlicedKMmaSchedule<warps, Capacity, warps == 16 ? 1 : 2>, Capacity>(
        x, weight, output, stream);
}

template <class Schedule>
void gemm(const Tensor& x, const Weight& weight, Output output, cudaStream_t stream) {
    static_assert(12288 % Schedule::kBlockRows == 0 && 8192 % Schedule::kBlockRows == 0);
    static_assert(Schedule::kSharedBytes <= 48 * 1024);
    launch_fp8_a16_mma<Fp8ScheduleInstance<Schedule, kHidden>>(fp8_a16_operands<float>(x, weight), output,
                                                               LinearIdentityEpilogue{}, stream);
}

} // namespace

std::size_t fp8_gdn_input_n20480_partial_capacity_bytes(std::int32_t max_tokens) {
    // Bulk's 80 row tiles leave no split tail at two token tiles (160 tiles) and one from three.
    return max_tokens > 256 ? Bulk::kPartialBytes : 0;
}

void fp8_gdn_input_n20480_invariant_launch(const Tensor& x, const Weight& weight, Tensor& qkv,
                                           Tensor& z, cudaStream_t stream) {
    // Four K-warps over the same K slices, reduced in the same order, at every width; the MMA's
    // columns are independent, and wider calls run as 64-column slices.
    const Output output = output_of(qkv, z);
    if (x.ne[1] <= 8) return sliced<Fp8SlicedInstance<8, 4, 1>>(x, weight, output, stream);
    if (x.ne[1] <= 16) return sliced<Fp8SlicedInstance<16, 4, 1>>(x, weight, output, stream);
    if (x.ne[1] <= 32) return sliced<Fp8SlicedInstance<32, 4, 1>>(x, weight, output, stream);
    sliced<Fp8SlicedInstance<64, 4, 1>>(x, weight, output, stream);
}

void fp8_gdn_input_n20480_a16_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                     cudaStream_t stream) {
    const Output output = output_of(qkv, z);
    const int columns   = x.ne[1];
    if (columns == 1) {
        using Schedule = Fp8A16GemvSchedule<4, 4, 16, 4, Fp8CodeCache::Default, 1, 1>;
        return launch_fp8_a16_gemv<Fp8ScheduleInstance<Schedule, kHidden>>(
            fp8_a16_operands<float>(x, weight), output, LinearIdentityEpilogue{}, stream);
    }
    if (columns <= 4) {
        using Schedule =
            Fp8A16SimtSchedule<8, 2, 16, 4, 1, Fp8SimtActivationAccess::SharedPhase,
                               Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::RowsContiguous, 1>;
        return launch_fp8_a16_simt<Fp8ScheduleInstance<Schedule, kHidden, 4>>(
            fp8_a16_operands<float>(x, weight), output, LinearIdentityEpilogue{}, stream);
    }
    if (columns <= 16) {
        using Schedule = Fp8A16SlicedKMmaSchedule<4, 16, 7, Cache::ca, Cache::cg,
                                                  Fp8ActivationStage::PaddedZero, 1>;
        return sliced<Schedule>(x, weight, output, stream);
    }
    if (columns <= 24) return small_mma<24>(x, weight, output, stream);
    if (columns <= 32) return small_mma<32>(x, weight, output, stream);
    if (columns <= 64)
        return gemm<Fp8A16MmaSchedule<32, 64, 128, 16, 16, 1, 3>>(x, weight, output, stream);
    if (columns <= 96)
        return gemm<Fp8A16MmaSchedule<64, 96, 128, 64, 16, 1, 2>>(x, weight, output, stream);
    gemm<Fp8A16MmaSchedule<64, 128, 64, 32, 16, 2, 2>>(x, weight, output, stream);
}

void fp8_gdn_input_n20480_a8_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                    Fp8A8Workspace workspace, cudaStream_t stream) {
    launch_fp8_a8_quantize(x, weight, workspace, stream);
    const Output output = output_of(qkv, z);
    const auto operands = fp8_a8_operands<float>(weight, workspace, x.ne[1]);
    const auto launch   = [&]<class Schedule>() {
        using S = Fp8ScheduleInstance<Schedule, kHidden>;
        if constexpr (S::kTmaSwizzle)
            launch_fp8_a8_tma_mma<S>(operands, output, LinearIdentityEpilogue{}, stream,
                                     workspace.partials);
        else
            launch_fp8_a8_mma<S>(operands, output, LinearIdentityEpilogue{}, stream);
    };
    const int tokens = x.ne[1];
    if (tokens <= 32) return launch.template operator()<Fp8A8T32R32K128>();
    if (tokens <= 64) return launch.template operator()<Fp8A8T64R128K256>();
    if (tokens <= 128) return launch.template operator()<Tma64x128>();
    if (tokens <= 192) return launch.template operator()<Tma192x128>();
    if (tokens > 384 && tokens <= 512) return launch.template operator()<MidBulk>();
    launch.template operator()<Bulk>();
}

} // namespace infernix::ops::detail
