#include "ops/linear/q6/q6_launch.h"
#include "ops/linear/q6/q6_mma_launch.cuh"
#include "ops/linear/q6/q6_instances.cuh"

namespace infernix::ops::detail {

void launch_q6_a16_mma_r64_t40_k128(const Tensor& x, const Weight& weight, Tensor& out,
                                    cudaStream_t stream) {
    with_linear_output(out, weight.n, [&](auto output) {
        launch_q6_a16_mma<q6_instances::MmaR64T40K128>(q6_linear_operands(x, weight), output, LinearIdentityEpilogue{}, stream);
    });
}

void launch_q6_a16_mma_r64_t48_k128(const Tensor& x, const Weight& weight, Tensor& out,
                                    cudaStream_t stream) {
    with_linear_output(out, weight.n, [&](auto output) {
        launch_q6_a16_mma<q6_instances::MmaR64T48K128>(q6_linear_operands(x, weight), output, LinearIdentityEpilogue{}, stream);
    });
}

void launch_q6_a16_mma_r64_t56_k128(const Tensor& x, const Weight& weight, Tensor& out,
                                    cudaStream_t stream) {
    with_linear_output(out, weight.n, [&](auto output) {
        launch_q6_a16_mma<q6_instances::MmaR64T56K128>(q6_linear_operands(x, weight), output, LinearIdentityEpilogue{}, stream);
    });
}

void launch_q6_a16_mma_r64_t64_k128(const Tensor& x, const Weight& weight, Tensor& out,
                                    cudaStream_t stream) {
    with_linear_output(out, weight.n, [&](auto output) {
        launch_q6_a16_mma<q6_instances::MmaR64T64K128>(q6_linear_operands(x, weight), output, LinearIdentityEpilogue{}, stream);
    });
}

void launch_q6_a16_mma_r64_t72_k128(const Tensor& x, const Weight& weight, Tensor& out,
                                    cudaStream_t stream) {
    with_linear_output(out, weight.n, [&](auto output) {
        launch_q6_a16_mma<q6_instances::MmaR64T72K128>(q6_linear_operands(x, weight), output, LinearIdentityEpilogue{}, stream);
    });
}

void launch_q6_a16_mma_r64_t80(const Tensor& x, const Weight& weight, Tensor& out,
                               cudaStream_t stream) {
    with_linear_output(out, weight.n, [&](auto output) {
        launch_q6_a16_mma<q6_instances::MmaR64T80>(q6_linear_operands(x, weight), output, LinearIdentityEpilogue{}, stream);
    });
}

void launch_q6_a16_mma_r64_t96(const Tensor& x, const Weight& weight, Tensor& out,
                               cudaStream_t stream) {
    with_linear_output(out, weight.n, [&](auto output) {
        launch_q6_a16_mma<q6_instances::MmaR64T96>(q6_linear_operands(x, weight), output, LinearIdentityEpilogue{}, stream);
    });
}

void launch_q6_a16_mma_r64_t128(const Tensor& x, const Weight& weight, Tensor& out,
                                cudaStream_t stream) {
    with_linear_output(out, weight.n, [&](auto output) {
        launch_q6_a16_mma<q6_instances::MmaR64T128>(q6_linear_operands(x, weight), output, LinearIdentityEpilogue{}, stream);
    });
}

void launch_q6_a16_mma_r64_t112(const Tensor& x, const Weight& weight, Tensor& out,
                                cudaStream_t stream) {
    const auto operands = q6_linear_operands(x, weight);
    // The full tile and the capacity tail have different measured warp mappings.
    with_linear_output(out, weight.n, [&](auto output) {
        if (x.ne[1] % 112 == 0)
            launch_q6_a16_mma<q6_instances::MmaR64T112>(operands, output, LinearIdentityEpilogue{}, stream);
        else
            launch_q6_a16_mma<q6_instances::MmaR64T112Partial>(operands, output, LinearIdentityEpilogue{},
                                                               stream);
    });
}

} // namespace infernix::ops::detail
