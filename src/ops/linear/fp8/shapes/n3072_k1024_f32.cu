// Qwen3.5-122B-A10B shared expert down projection stored per-row FP8 (dealignai's
// Ornith-1.5-122B-UNCENSORED-NVFP4): fp8_e4m3fn_row_f32 [3072,1024] with FP32 row multipliers and BF16
// activations (W8A16), its output combined with the routed experts' (offloaded_moe). 3 MiB of codes: a
// decode or verification call is latency-bound, so the narrow widths take the GEMV and the 8-K-warp
// sliced-K tiles (192 CTAs over every SM), prefill the ring tensor-core tiles.
#include "ops/linear/fp8/fp8_shapes.h"
#include "ops/linear/fp8/fp8_launch.cuh"

namespace infernix::ops::detail {
namespace {
constexpr int kK = 1024;
using Geometry   = Fp8Geometry<3072, kK>;

LinearBf16Output output_of(Tensor& out, const Weight& w) {
    return {static_cast<__nv_bfloat16*>(out.data), w.n};
}

template <class Schedule>
void sliced(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    launch_fp8_a16_sliced_k_mma<Fp8ScheduleInstance<Schedule, kK>>(fp8_a16_operands<float>(x, weight),
                                                                   output_of(out, weight), LinearIdentityEpilogue{},
                                                                   stream);
}

template <class Schedule>
void ring(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    launch_fp8_a16_ring_mma<Fp8ScheduleInstance<Schedule, kK>>(fp8_a16_operands<float>(x, weight),
                                                              output_of(out, weight), LinearIdentityEpilogue{}, stream);
}

void launch_a16(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const int tokens = x.ne[1];
    if (tokens == 1) {
        using Schedule = Fp8A16GemvSchedule<8, 2, 8, 4, Fp8CodeCache::Default, 2, 2>;
        return launch_fp8_a16_gemv<Fp8ScheduleInstance<Schedule, kK>>(
            fp8_a16_operands<float>(x, weight), output_of(out, weight), LinearIdentityEpilogue{}, stream);
    }
    if (tokens <= 8) return sliced<Fp8SlicedInstance<8, 8, 2>>(x, weight, out, stream);
    if (tokens <= 16) return sliced<Fp8SlicedInstance<16, 8, 2>>(x, weight, out, stream);
    if (tokens <= 32) return sliced<Fp8SlicedInstance<32, 4, 2>>(x, weight, out, stream);
    if (tokens <= 64) return sliced<Fp8SlicedInstance<64, 4, 1>>(x, weight, out, stream);
    if (tokens <= 256) return ring<Fp8A16RingT32R64S3>(x, weight, out, stream);
    ring<Fp8A16RingT64R128>(x, weight, out, stream);
}

bool uses_a8(std::int32_t, std::int32_t) { return false; }

// A16ColumnInvariant: eight K-warps over the same K slices, reduced in the same order, at every width;
// the MMA's columns are independent, and wider calls run as 64-column slices.
void launch_invariant(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    if (x.ne[1] <= 8) return sliced<Fp8SlicedInstance<8, 8, 2>>(x, weight, out, stream);
    if (x.ne[1] <= 16) return sliced<Fp8SlicedInstance<16, 8, 2>>(x, weight, out, stream);
    sliced<Fp8SlicedInstance<64, 8, 1>>(x, weight, out, stream);
}
} // namespace

const Fp8LinearShape kFp8F32N3072K1024{3072, kK, launch_a16, nullptr, uses_a8, nullptr, launch_invariant};
} // namespace infernix::ops::detail
