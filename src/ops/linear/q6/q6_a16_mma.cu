#include "ops/linear/q6/q6_launch.h"

#include "ops/linear/common/rowsplit_ring_linear.cuh"
#include "ops/linear/q6/q6_ring_codec.cuh"
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

void launch_q6_a16_wide_t128(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    // The wide route runs on the staging ring (ops/linear/common/rowsplit_ring_linear.cuh; 128 x 128 tiles),
    // the same bits as the 64-row MMA tile it replaced. RTX 5090, linear_bench (CUDA graph, cold L2, A B B A,
    // equal output hashes), 512 / 1,024 / 2,048 / 4,096 columns: [34816, 5120] -3.9 / -4.9 / -5.0 / -5.1 %
    // (2026-11-10).
    with_linear_output(out, weight.n, [&](auto output) {
        const Q6LinearOperands p = q6_linear_operands(x, weight);
        launch_rowsplit_ring_linear<Q6RingFormat, RowSplitRingTile128, kRowSplitRingStages,
                                    kRowSplitRingProducerWarps>(
            p.x, Q6RingFormat::Weights{p.codes, p.high, p.scales}, p.rows, p.k, p.padded_k, p.tokens, output,
            LinearIdentityEpilogue{}, stream);
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
