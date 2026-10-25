#include "ops/linear/q6/q6_launch.h"
#include "ops/linear/q6/q6_simt_launch.cuh"
#include "ops/linear/q6/q6_instances.cuh"

namespace infernix::ops::detail {

void launch_q6_a16_simt_r8_t4(const Tensor& x, const Weight& weight, Tensor& out,
                              cudaStream_t stream) {
    with_linear_output(out, weight.n, [&](auto output) {
        launch_q6_a16_simt<q6_instances::SimtR8T4>(q6_linear_operands(x, weight), output, LinearIdentityEpilogue{}, stream);
    });
}

void launch_q6_a16_gemv_r4_w2_g16(const Tensor& x, const Weight& weight, Tensor& out,
                                  cudaStream_t stream) {
    with_linear_output(out, weight.n, [&](auto output) {
        launch_q6_a16_gemv<q6_instances::GemvR4W2G16>(q6_linear_operands(x, weight), output, LinearIdentityEpilogue{}, stream);
    });
}

} // namespace infernix::ops::detail
