#include "core/weight.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.h"

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/common/warp.cuh"
#include "ops/linear/nvfp4/nvfp4_a16_slice_ring_mma.cuh"
#include "ops/linear/nvfp4/nvfp4_schedule.cuh"
#include "ops/linear/nvfp4/nvfp4_template_launch.cuh"
#include "ops/linear/nvfp4/nvfp4_instances.cuh"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_epilogue.cuh"

#include <cuda_bf16.h>

#include <array>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace infernix::ops::detail {
namespace {

using Geometry              = Nvfp4N34816K5120;
constexpr int kIntermediate = Geometry::kOutputRows / 2;

using Launch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

template <int ActiveTokens>
void launch_exact(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    static constexpr auto kActivationAccess = ActiveTokens <= 4
                                                  ? Nvfp4SimtActivationAccess::SharedPhase
                                                  : Nvfp4SimtActivationAccess::TokenPacked;
    static constexpr int kWarpsPerCta       = ActiveTokens >= 13 ? 16 : (ActiveTokens >= 5 ? 4 : 8);
    using Schedule =
        Nvfp4A16SimtSchedule<kWarpsPerCta, 1, 2, 16, ActiveTokens, 1, kActivationAccess,
                             Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                             Nvfp4SimtBlockOrder::RowsContiguous, 1>;
    launch_nvfp4_a16_simt<Nvfp4ScheduleInstance<Schedule, 5120, ActiveTokens, true>>(
        nvfp4_a16_operands(x, weight),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), kIntermediate},
        Nvfp4SwiGluEpilogue{}, stream, Nvfp4SwiGluRows<1>{});
}

template <std::size_t... Offsets>
constexpr auto make_launchers(std::index_sequence<Offsets...>) {
    return std::array<Launch, sizeof...(Offsets)>{&launch_exact<2 + static_cast<int>(Offsets)>...};
}

constexpr auto kLaunchers = make_launchers(std::make_index_sequence<1>{});

} // namespace

void nvfp4_linear_swiglu_invariant_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                          cudaStream_t stream) {
    // From 17 columns (the reproducible route's verification of the NVFP4 27B MLP) the slice-major ring tile,
    // bit-identical to the sliced-K tiles: 0.47-0.71x their time at 17-129 columns; the reproducible route's
    // DFlash2 decode +19 % (2026-10-25).
    if (x.ne[1] >= 17) {
        nvfp4_linear_swiglu_invariant_ring_launch(x, weight, out, stream);
        return;
    }
    nvfp4_linear_swiglu_invariant_sliced_launch(x, weight, out, stream);
}

void nvfp4_linear_swiglu_invariant_ring_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                               cudaStream_t stream) {
    // The sliced-K tiles' 8 K slices on 128 weight rows (64 outputs) a CTA, 272 CTAs per token tile;
    // 32-token tiles two CTAs an SM up to 32 columns, 64-token tiles one above.
    if (x.ne[1] < 17) throw std::invalid_argument("nvfp4 linear_swiglu ring tile: T must be 17 or more");
    const auto p = nvfp4_a16_operands(x, weight);
    const LinearBf16Output output{static_cast<__nv_bfloat16*>(out.data), kIntermediate};
    if (x.ne[1] <= 32)
        launch_nvfp4_a16_slice_ring_mma<
            Nvfp4ScheduleInstance<Nvfp4A16SliceRingSchedule<8, 128, 32, 8, 1, 4, 2>, 5120>>(
            p, output, Nvfp4SwiGluEpilogue{}, stream, Nvfp4SwiGluMmaRows{});
    else
        launch_nvfp4_a16_slice_ring_mma<
            Nvfp4ScheduleInstance<Nvfp4A16SliceRingSchedule<8, 128, 64, 4, 2, 4, 1>, 5120>>(
            p, output, Nvfp4SwiGluEpilogue{}, stream, Nvfp4SwiGluMmaRows{});
}

void nvfp4_linear_swiglu_invariant_sliced_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                                 cudaStream_t stream) {
    // The tiles run 8 K-warps over the same K slices and reduce them in the same order, and the
    // MMA's columns are independent, so the tile width changes only how many columns share a
    // launch; wider calls run as 64-column slices.
    const auto p = nvfp4_a16_operands(x, weight);
    const LinearBf16Output output{static_cast<__nv_bfloat16*>(out.data), kIntermediate};
    if (x.ne[1] <= 8)
        launch_nvfp4_a16_sliced_k_mma<Nvfp4ScheduleInstance<Nvfp4SlicedInstance<8, 8, 2>, 5120>>(
            p, output, Nvfp4SwiGluEpilogue{}, stream, Nvfp4SwiGluRows<8>{});
    else if (x.ne[1] <= 16)
        launch_nvfp4_a16_sliced_k_mma<Nvfp4ScheduleInstance<Nvfp4SlicedInstance<16, 8, 2>, 5120>>(
            p, output, Nvfp4SwiGluEpilogue{}, stream, Nvfp4SwiGluRows<8>{});
    else
        launch_nvfp4_a16_sliced_k_mma<Nvfp4ScheduleInstance<Nvfp4SlicedInstance<64, 8, 1>, 5120>>(
            p, output, Nvfp4SwiGluEpilogue{}, stream, Nvfp4SwiGluRows<8>{});
}

void nvfp4_linear_swiglu_small_t_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                        cudaStream_t stream) {
    const auto p = nvfp4_a16_operands(x, weight);
    const LinearBf16Output output{static_cast<__nv_bfloat16*>(out.data), kIntermediate};
    if (x.ne[1] <= 2)
        kLaunchers[0](x, weight, out, stream);
    else if (x.ne[1] <= 8)
        launch_nvfp4_a16_sliced_k_mma<Nvfp4ScheduleInstance<Nvfp4SlicedInstance<8, 8, 2>, 5120>>(
            p, output, Nvfp4SwiGluEpilogue{}, stream, Nvfp4SwiGluRows<8>{});
    else if (x.ne[1] <= 16)
        launch_nvfp4_a16_sliced_k_mma<Nvfp4ScheduleInstance<Nvfp4SlicedInstance<16, 8, 2>, 5120>>(
            p, output, Nvfp4SwiGluEpilogue{}, stream, Nvfp4SwiGluRows<8>{});
    else if (x.ne[1] <= 32)
        launch_nvfp4_a16_sliced_k_mma<Nvfp4ScheduleInstance<Nvfp4SlicedInstance<32, 4, 2>, 5120>>(
            p, output, Nvfp4SwiGluEpilogue{}, stream, Nvfp4SwiGluRows<8>{});
    else
        launch_nvfp4_a16_mma<
            Nvfp4ScheduleInstance<Nvfp4A16MmaSchedule<32, 64, 128, 16, 16, 1, 3>, 5120>>(
            p, output, Nvfp4SwiGluEpilogue{}, stream, Nvfp4SwiGluMmaRows{});
}

} // namespace infernix::ops::detail
