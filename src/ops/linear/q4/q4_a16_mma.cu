#include "ops/linear/q4/q4_instance_launch.cuh"

#include "ops/linear/common/rowsplit_ring_linear.cuh"
#include "ops/linear/q4/q4_ring_codec.cuh"

namespace infernix::ops::detail {

void launch_q4_a16_mma_r64_t48(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_q4_a16_mma_instance<q4_instances::MmaR64T48K64Wr16Wt16S2A2B2>(x, w, out, stream);
}

void launch_q4_a16_mma_r64_t72(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_q4_a16_mma_instance<q4_instances::MmaR64T72K64Wr32Wt24S2A2B2>(x, w, out, stream);
}

void launch_q4_a16_mma_r64_t80(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_q4_a16_mma_instance<q4_instances::MmaR64T80K64Wr16Wt40S2A2B1>(x, w, out, stream);
}

void launch_q4_a16_mma_r64_t96(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_q4_a16_mma_instance<q4_instances::MmaR64T96K64Wr32Wt16S2A2B1>(x, w, out, stream);
}

void launch_q4_a16_mma_r64_t112(const Tensor& x, const Weight& w, Tensor& out,
                                cudaStream_t stream) {
    // Complete and partial column tiles favor different fragment mappings.
    if (x.ne[1] % 112 == 0) {
        launch_q4_a16_mma_instance<q4_instances::MmaR64T112K64Wr16Wt112S2A2B1>(x, w, out, stream);
    } else {
        launch_q4_a16_mma_instance<q4_instances::MmaR64T112K64Wr32Wt16S2A2B1>(x, w, out, stream);
    }
}

void launch_q4_a16_mma_r64_t120(const Tensor& x, const Weight& w, Tensor& out,
                                cudaStream_t stream) {
    // Complete and partial column tiles favor different fragment mappings.
    if (x.ne[1] % 120 == 0) {
        launch_q4_a16_mma_instance<q4_instances::MmaR64T120K64Wr16Wt120S2A2B1>(x, w, out, stream);
    } else {
        launch_q4_a16_mma_instance<q4_instances::MmaR64T120K64Wr32Wt24S2A2B1>(x, w, out, stream);
    }
}

void launch_q4_a16_wide_t128(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    // The wide route runs on the staging ring (ops/linear/common/rowsplit_ring_linear.cuh; 128 x 128 tiles),
    // the same bits as the 64-row MMA tile it replaced. RTX 5090, linear_bench (CUDA graph, cold L2, A B B A,
    // equal output hashes), 512 / 1,024 / 2,048 / 4,096 columns: [4096, 5120] -3.3 / -2.7 / -2.3 / -2.6 %;
    // [6144, 5120] at 1,024-4,096 -3.0 / -3.4 / -2.9 % (2026-11-10).
    const Q4LinearOperands p = q4_linear_operands(x, w);
    validate_q4_operands(p);
    launch_rowsplit_ring_linear<Q4RingFormat, RowSplitRingTile128, kRowSplitRingStages,
                                kRowSplitRingProducerWarps>(
        p.x, Q4RingFormat::Weights{p.codes, p.scales}, p.rows, p.k, p.padded_k, p.tokens,
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n}, LinearIdentityEpilogue{}, stream);
}

void launch_q4_a16_mma_r32_t32_k128_s2_a2(const Tensor& x, const Weight& w, Tensor& out,
                                          cudaStream_t stream) {
    launch_q4_a16_mma_instance<q4_instances::MmaR32T32K128S2A2>(x, w, out, stream);
}

void launch_q4_a16_mma_r64_t64_k128_s2_a1(const Tensor& x, const Weight& w, Tensor& out,
                                          cudaStream_t stream) {
    launch_q4_a16_mma_instance<q4_instances::MmaR64T64K128S2A1>(x, w, out, stream);
}

void launch_q4_a16_mma_r32_t128_k64_s2_a2(const Tensor& x, const Weight& w, Tensor& out,
                                          cudaStream_t stream) {
    launch_q4_a16_mma_instance<q4_instances::MmaR32T128K64S2A2>(x, w, out, stream);
}

void launch_q4_a16_mma_r32_t32_k64_wr16_wt16_s3_a3_b2(const Tensor& x, const Weight& w, Tensor& out,
                                                      cudaStream_t stream) {
    launch_q4_a16_mma_instance<q4_instances::MmaR32T32K64Wr16Wt16S3A3B2>(x, w, out, stream);
}

void launch_q4_a16_mma_r32_t64_k64_wr16_wt32_s2_a2_b2(const Tensor& x, const Weight& w, Tensor& out,
                                                      cudaStream_t stream) {
    launch_q4_a16_mma_instance<q4_instances::MmaR32T64K64Wr16Wt32S2A2B2>(x, w, out, stream);
}

void launch_q4_a16_mma_r32_t64_k64_wr16_wt32_s3_a3_b2(const Tensor& x, const Weight& w, Tensor& out,
                                                      cudaStream_t stream) {
    launch_q4_a16_mma_instance<q4_instances::MmaR32T64K64Wr16Wt32S3A3B2>(x, w, out, stream);
}

void launch_q4_a16_mma_r64_t64_k64_wr32_wt16_s2_a2_b2(const Tensor& x, const Weight& w, Tensor& out,
                                                      cudaStream_t stream) {
    launch_q4_a16_mma_instance<q4_instances::MmaR64T64K64Wr32Wt16S2A2B2>(x, w, out, stream);
}

} // namespace infernix::ops::detail
