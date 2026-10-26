// Qwen3.5-122B-A10B vocabulary head stored per-row FP8 (dealignai's Ornith-1.5-122B-UNCENSORED-NVFP4):
// fp8_e4m3fn_row_f32 [248320,3072], each row with its FP32 multiplier, BF16 activations (the checkpoint
// keeps the head A16), FP32 or BF16 logits. The schedule families are the [248320,5120] head's
// (n248320_k5120.cu) at K 3072.
#include "ops/linear/fp8/fp8_template_launch.cuh"
#include "ops/linear/fp8/fp8_shapes.h"
#include "core/device.h"
#include "ops/common/math.h"
#include "ops/common/token_slices.h"
#include "ops/linear/fp8/fp8_a16_sliced_k_mma.cuh"
#include "ops/linear/fp8/fp8_a16_mma.cuh"

namespace infernix::ops::detail {
namespace {
constexpr int kHidden = 3072;

template <int ActiveTokens, int KWarps, int Stages>
void launch_tile(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    // Above 16 columns two row tiles share the staged activation (same split-K arithmetic), as on the
    // [248320,5120] head.
    using Schedule = Fp8A16SlicedKMmaSchedule<KWarps, ActiveTokens, ActiveTokens <= 8 ? 1 : 2,
                                              Cache::ca, Cache::cg, Fp8ActivationStage::ActiveOnly,
                                              Stages, (ActiveTokens > 16 ? 2 : 1)>;
    static_assert((kHidden % Schedule::kBlockK) == 0);
    with_linear_output(out, weight.n, [&](auto output) {
        launch_fp8_a16_sliced_k_mma<Fp8ScheduleInstance<Schedule, kHidden, ActiveTokens>>(
            fp8_a16_operands<float>(x, weight), output, LinearIdentityEpilogue{}, stream, {},
            pdl::Dependency::Programmatic);
    });
}

constexpr int kSlicedKTokens = 64;

void launch_ksplit(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const int tokens = x.ne[1];
    if (tokens <= 8) return launch_tile<8, 16, 1>(x, weight, out, stream);
    if (tokens <= 16) return launch_tile<16, 8, 1>(x, weight, out, stream);
    if (tokens <= 24) return launch_tile<24, 8, 1>(x, weight, out, stream);
    if (tokens <= 32) return launch_tile<32, 4, 1>(x, weight, out, stream);
    if (tokens <= 40) return launch_tile<40, 4, 1>(x, weight, out, stream);
    if (tokens <= 48) return launch_tile<48, 4, 1>(x, weight, out, stream);
    if (tokens <= 56) return launch_tile<56, 2, 2>(x, weight, out, stream);
    if (tokens <= kSlicedKTokens) return launch_tile<64, 2, 2>(x, weight, out, stream);
    throw std::logic_error("fp8 K-split exceeds shape capacity");
}

using Main128 = Fp8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>;
using Tail96  = Fp8A16MmaSchedule<64, 96, 64, 64, 16, 2, 2>;

template <class Schedule>
void launch_schedule(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    with_linear_output(out, weight.n, [&](auto output) {
        launch_fp8_a16_mma<Fp8ScheduleInstance<Schedule, kHidden>>(fp8_a16_operands<float>(x, weight), output,
                                                                   LinearIdentityEpilogue{}, stream);
    });
}

void launch_a16(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const std::int32_t tokens = x.ne[1];
    if (tokens <= kSlicedKTokens) return launch_ksplit(x, weight, out, stream);
    if (tokens <= Tail96::kBlockTokens) return launch_schedule<Tail96>(x, weight, out, stream);
    if (tokens <= Main128::kBlockTokens) return launch_schedule<Main128>(x, weight, out, stream);
    const std::int32_t tail = tokens % Main128::kBlockTokens;
    if (tail == 0 || tail > Tail96::kBlockTokens) return launch_schedule<Main128>(x, weight, out, stream);
    const std::int32_t prefix = tokens - tail;
    const Tensor input_prefix = x.slice(1, 0, prefix);
    Tensor output_prefix      = out.slice(1, 0, prefix);
    launch_schedule<Main128>(input_prefix, weight, output_prefix, stream);
    const Tensor input_tail = x.slice(1, prefix, tail);
    Tensor output_tail      = out.slice(1, prefix, tail);
    if (tail <= kSlicedKTokens) return launch_ksplit(input_tail, weight, output_tail, stream);
    launch_schedule<Tail96>(input_tail, weight, output_tail, stream);
}

bool uses_a8(std::int32_t, std::int32_t) { return false; }

// A16ColumnInvariant: 16 K-warps over the same K slices, reduced in the same order, at every T: the
// 8-column tile through 8 columns and 16-column slices beyond (the MMA's columns are independent). The
// [248320,5120] head's 32-column two-row-tile slice spills 8 bytes with FP32 row words.
void launch_invariant(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    if (x.ne[1] <= 8) return launch_tile<8, 16, 1>(x, weight, out, stream);
    launch_tile<16, 16, 1>(x, weight, out, stream);
}
} // namespace

const Fp8LinearShape kFp8F32N248320K3072{248320, 3072, launch_a16, nullptr, uses_a8, nullptr, launch_invariant};
} // namespace infernix::ops::detail
