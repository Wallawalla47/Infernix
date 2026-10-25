#pragma once
#include "ops/linear/q8/q8_instances.cuh"
#include "ops/linear/q8/q8_sliced_k_launch.cuh"

namespace infernix::ops::detail {
template <class Geometry, int Capacity, class Schedule>
void launch_q8_a16_sliced(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    using Instance = typename Schedule::template with_problem<Geometry::kInputRows, Capacity>;
    launch_q8_a16_sliced_k_mma<Instance>(
        q8_linear_operands(x, weight),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), Geometry::kOutputRows},
        LinearIdentityEpilogue{}, stream, {}, pdl::Dependency::Programmatic);
}

// The same with an FP32 or BF16 output by out's dtype: a vocabulary head, whose logits are FP32.
template <class Geometry, int Capacity, class Schedule>
void launch_q8_a16_sliced_logits(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    using Instance = typename Schedule::template with_problem<Geometry::kInputRows, Capacity>;
    with_linear_output(out, Geometry::kOutputRows, [&](auto output) {
        launch_q8_a16_sliced_k_mma<Instance>(q8_linear_operands(x, weight), output, LinearIdentityEpilogue{}, stream,
                                             {}, pdl::Dependency::Programmatic);
    });
}
} // namespace infernix::ops::detail
