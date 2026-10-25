#include "ops/linear/q6/q6_launch.h"
#include "ops/linear/q6/q6_sliced_k_launch.cuh"
#include "ops/linear/q6/q6_instances.cuh"

namespace infernix::ops::detail {

void launch_q6_a16_sliced_r16_t8_w4_s2(const Tensor& x, const Weight& weight, Tensor& out,
                                       cudaStream_t stream) {
    with_linear_output(out, weight.n, [&](auto output) {
        launch_q6_a16_sliced_k_mma<q6_instances::SlicedR16T8W4S2>(q6_linear_operands(x, weight), output, LinearIdentityEpilogue{}, stream);
    });
}

void launch_q6_a16_sliced_r32_t16_w4_s2(const Tensor& x, const Weight& weight, Tensor& out,
                                        cudaStream_t stream) {
    with_linear_output(out, weight.n, [&](auto output) {
        launch_q6_a16_sliced_k_mma<q6_instances::SlicedR32T16W4S2>(q6_linear_operands(x, weight), output, LinearIdentityEpilogue{}, stream);
    });
}

void launch_q6_a16_sliced_r32_t32_w4_s1(const Tensor& x, const Weight& weight, Tensor& out,
                                        cudaStream_t stream) {
    with_linear_output(out, weight.n, [&](auto output) {
        launch_q6_a16_sliced_k_mma<q6_instances::SlicedR32T32W4S1>(q6_linear_operands(x, weight), output, LinearIdentityEpilogue{}, stream);
    });
}

void launch_q6_a16_sliced_r32_t64_w2_s1(const Tensor& x, const Weight& weight, Tensor& out,
                                        cudaStream_t stream) {
    with_linear_output(out, weight.n, [&](auto output) {
        launch_q6_a16_sliced_k_mma<q6_instances::SlicedR32T64W2S1>(q6_linear_operands(x, weight), output, LinearIdentityEpilogue{}, stream);
    });
}

void launch_q6_a16_sliced_r16_t24_w4_s2(const Tensor& x, const Weight& weight, Tensor& out,
                                        cudaStream_t stream) {
    with_linear_output(out, weight.n, [&](auto output) {
        launch_q6_a16_sliced_k_mma<q6_instances::SlicedR16T24W4S2>(q6_linear_operands(x, weight), output, LinearIdentityEpilogue{}, stream);
    });
}

void launch_q6_a16_sliced_r16_t32_w4_s2(const Tensor& x, const Weight& weight, Tensor& out,
                                        cudaStream_t stream) {
    with_linear_output(out, weight.n, [&](auto output) {
        launch_q6_a16_sliced_k_mma<q6_instances::SlicedR16T32W4S2>(q6_linear_operands(x, weight), output, LinearIdentityEpilogue{}, stream);
    });
}

void launch_q6_a16_sliced_r32_t32_w4_s2(const Tensor& x, const Weight& weight, Tensor& out,
                                        cudaStream_t stream) {
    with_linear_output(out, weight.n, [&](auto output) {
        launch_q6_a16_sliced_k_mma<q6_instances::SlicedR32T32W4S2>(q6_linear_operands(x, weight), output, LinearIdentityEpilogue{}, stream);
    });
}

// The column-invariant routes (q6_dispatch.cpp) mix these instances by width: they must share the
// K-warp count, which sets the reduction order.
static_assert(q6_instances::SlicedR16T8W4S2::kKWarps == 4 &&
              q6_instances::SlicedR32T16W4S2::kKWarps == 4 &&
              q6_instances::SlicedR32T32W4S1::kKWarps == 4);

} // namespace infernix::ops::detail
