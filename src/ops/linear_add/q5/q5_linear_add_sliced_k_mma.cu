#include "ops/linear_add/q5/q5_linear_add_kernels.h"
#include "ops/linear/q5/q5_instances.cuh"
#include "ops/linear/q5/q5_sliced_k_launch.cuh"
#include "ops/lora/lora_device.cuh"

#include <stdexcept>

namespace infernix::ops::detail {
namespace {
template <class Schedule>
void launch_sliced(const Tensor& x, const Weight& w, Tensor& residual, cudaStream_t stream,
                   const LoraAddend* lora) {
    auto* data        = static_cast<__nv_bfloat16*>(residual.data);
    const auto stride = std::int64_t(residual.nb[1] / sizeof(__nv_bfloat16));
    with_lora_epilogue(lora, LinearResidualAddEpilogue{{data, stride, 0}}, [&](auto epilogue) {
        launch_q5_a16_sliced_k_mma<Schedule>(q5_linear_operands(x, w), LinearBf16StridedOutput{data, stride, 0},
                                             epilogue, stream);
    });
}
} // namespace

void q5_linear_add_sliced_r16_t8_launch(const Tensor& x, const Weight& w, Tensor& residual,
                                        cudaStream_t stream, const LoraAddend* lora) {
    launch_sliced<q5_instances::SlicedR16T8W4S2>(x, w, residual, stream, lora);
}

void q5_linear_add_sliced_r16_t16_launch(const Tensor& x, const Weight& w, Tensor& residual,
                                         cudaStream_t stream, const LoraAddend* lora) {
    launch_sliced<q5_instances::SlicedR16T16W4S2>(x, w, residual, stream, lora);
}

void q5_linear_add_sliced_r16_t24_launch(const Tensor& x, const Weight& w, Tensor& residual,
                                         cudaStream_t stream, const LoraAddend* lora) {
    launch_sliced<q5_instances::SlicedR16T24W4S2>(x, w, residual, stream, lora);
}

void q5_linear_add_sliced_r32_t32_w4_s2_launch(const Tensor& x, const Weight& w, Tensor& residual,
                                               cudaStream_t stream, const LoraAddend* lora) {
    launch_sliced<q5_instances::SlicedR32T32W4S2>(x, w, residual, stream, lora);
}

void q5_linear_add_sliced_r32_t24_pairwise_launch(const Tensor& x, const Weight& w,
                                                  Tensor& residual, cudaStream_t stream,
                                                  const LoraAddend* lora) {
    launch_sliced<q5_instances::SlicedR32T24W4S2Pairwise>(x, w, residual, stream, lora);
}

void q5_linear_add_sliced_r32_t32_w4_s1_launch(const Tensor& x, const Weight& w, Tensor& residual,
                                               cudaStream_t stream, const LoraAddend* lora) {
    launch_sliced<q5_instances::SlicedR32T32W4S1>(x, w, residual, stream, lora);
}

void q5_linear_add_sliced_r32_t32_w2_s2_launch(const Tensor& x, const Weight& w, Tensor& residual,
                                               cudaStream_t stream, const LoraAddend* lora) {
    launch_sliced<q5_instances::SlicedR32T32W2S2>(x, w, residual, stream, lora);
}

void q5_linear_add_sliced_r32_t64_w2_s1_launch(const Tensor& x, const Weight& w, Tensor& residual,
                                               cudaStream_t stream, const LoraAddend* lora) {
    launch_sliced<q5_instances::SlicedR32T64W2S1>(x, w, residual, stream, lora);
}

// Every instance below is the sliced-K schedule with four K warps and the linear reduction; block
// rows, token capacity, stages and the token slicing of launch_q5_a16_sliced_k_mma change only
// staging and predication, so a column's bits do not depend on the call's width.
template <class... Schedules>
constexpr bool same_reduction() {
    return ((Schedules::kKWarps == 4 && Schedules::kReduction == Q5SlicedKReduction::Linear) && ...);
}
static_assert(same_reduction<q5_instances::SlicedR16T8W4S2, q5_instances::SlicedR16T16W4S2,
                             q5_instances::SlicedR16T24W4S2, q5_instances::SlicedR32T32W4S2>());

void q5_linear_add_sliced_invariant_launch(const Tensor& x, const Weight& w, Tensor& residual,
                                           cudaStream_t stream, const LoraAddend* lora) {
    const std::int32_t cols = x.ne[1];
    if (cols < 1) { throw std::invalid_argument("q5 linear_add column-invariant route requires T >= 1"); }
    if (cols <= 8) {
        launch_sliced<q5_instances::SlicedR16T8W4S2>(x, w, residual, stream, lora);
    } else if (cols <= 16) {
        launch_sliced<q5_instances::SlicedR16T16W4S2>(x, w, residual, stream, lora);
    } else if (cols <= 24) {
        launch_sliced<q5_instances::SlicedR16T24W4S2>(x, w, residual, stream, lora);
    } else {
        launch_sliced<q5_instances::SlicedR32T32W4S2>(x, w, residual, stream, lora);
    }
}
} // namespace infernix::ops::detail
