// Qwen3.8-Flash-Next shared expert down projection, MXFP8 with BF16 activations (W8A16, NVIDIA
// FP4-Local). Every width runs the Tensor Core route, whose schedules accumulate a column in one
// order, so a column has the same bits at every T and with either producer (mxfp8_a16_tma_mma.cuh).
// The selection is bench/ops/mxfp8_flash_next_sweep.cu's (RTX 5090, 2026-10-20: the fastest schedule
// per width, merged within 2 %): cp.async at every width (TMA was up to 44 % slower and never faster).
#include "ops/linear/mxfp8/mxfp8_launch.cuh"
#include "ops/linear/mxfp8/mxfp8_shapes.h"
#include "ops/linear/f16acc/f16acc_mma.cuh"
#include "ops/linear/fp8/fp8_a8_pair_tma_mma.cuh"

namespace infernix::ops::detail {
namespace {
using Geometry = Fp8Geometry<2560, 640>;

void launch_a16(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const int tokens = x.ne[1];
    if (tokens <= 64) return mxfp8_linear_a16_tma<Geometry, Mxfp8P2<Mxfp8TmaT16R64>>(x, weight, out, stream);
    if (tokens <= 128) return mxfp8_linear_a16_tma<Geometry, Fp8A16RingT32R64>(x, weight, out, stream);
    mxfp8_linear_a16_tma<Geometry, Mxfp8P2<Mxfp8TmaT64R64>>(x, weight, out, stream);
}

// f16-accumulate-prefill: the exact route's tile at the same widths (not yet swept for this route).
void launch_f16(const Tensor& x, const Weight& weight, Tensor& out, F16AccumulateWorkspace scratch,
                cudaStream_t stream) {
    mxfp8_linear_f16acc<Geometry, Mxfp8P2<Mxfp8TmaT64R64>>(x, weight, out, scratch, stream);
}

// fp8-a8-pair-block32: the activation as two E4M3 terms with per-32 UE8M0 scales, the weight's own group
// scales in the same block-scaled MMA (fp8_a8_pair_tma_mma.cuh); recipe C's pair tiles, not yet swept here.
void launch_pair_block32(const Tensor& x, const Weight& weight, Tensor& out, Fp8A8PairBlock32Workspace scratch,
                         cudaStream_t stream) {
    fp8_a8_pair_block32_gemm<Geometry, true>(x, weight, LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), weight.n},
                                             scratch, stream);
}
} // namespace

const Mxfp8LinearShape kMxfp8N2560K640{2560, 640, launch_a16, launch_f16, kMxfp8F16MinTokens, launch_pair_block32, kMxfp8PairMinTokensN2560};
} // namespace infernix::ops::detail
