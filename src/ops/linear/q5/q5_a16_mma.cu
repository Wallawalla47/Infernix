#include "ops/linear/q5/q5_instance_launch.cuh"

#include "ops/linear/common/rowsplit_ring_linear.cuh"
#include "ops/linear/q5/q5_ring_codec.cuh"

namespace infernix::ops::detail {
void launch_q5_a16_mma_r32_t128(const Tensor& x, const Weight& w, Tensor& out,
                                cudaStream_t stream) {
    launch_q5_a16_mma_instance<q5_instances::MmaR32T128>(x, w, out, stream);
}

void launch_q5_a16_wide_t128(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    // The wide route runs on the staging ring (ops/linear/common/rowsplit_ring_linear.cuh; 128 x 128 tiles),
    // the same bits as the 64-row MMA tile it replaced. RTX 5090, linear_bench (CUDA graph, cold L2, A B B A,
    // equal output hashes), 512 / 1,024 / 2,048 / 4,096 columns: [5120, 17408] -5.9 / -4.6 / -5.8 / -7.1 %;
    // [6144, 5120] at 1,024-4,096 -5.1 / -6.0 / -6.0 % (2026-11-10).
    const Q5LinearOperands p = q5_linear_operands(x, w);
    validate_q5_operands(p);
    launch_rowsplit_ring_linear<Q5RingFormat, RowSplitRingTile128, kRowSplitRingStages,
                                kRowSplitRingProducerWarps>(
        p.x, Q5RingFormat::Weights{p.codes, p.high, p.scales}, p.rows, p.k, p.padded_k, p.tokens,
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n}, LinearIdentityEpilogue{}, stream);
}

void launch_q5_a16_mma_r64_t96_k128_s1_a1(const Tensor& x, const Weight& w, Tensor& out,
                                          cudaStream_t stream) {
    launch_q5_a16_mma_instance<q5_instances::MmaR64T96K128S1A1>(x, w, out, stream);
}
} // namespace infernix::ops::detail
