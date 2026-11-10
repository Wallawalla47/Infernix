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

// f16-accumulate-prefill: tiles from linear_bench --qtype mxfp8 --routes f16 with each tile forced
// (RTX 5090, 2026-11-10; every tile gives the same output hash). The 128x128 tile (20 CTAs a 128 columns, one
// an SM) takes 7-9 % less than the 64x64 one where its last wave of 170 is at least half full (832-1,024,
// 2,048, 3,072, 4,096 columns) and 19-20 % less at 8,192-16,384 (228 against 285 us); with a thinner last wave
// it is level or slower (1,536: +6.5 %, 2,560: +2.0 %).
void launch_f16(const Tensor& x, const Weight& weight, Tensor& out, F16AccumulateWorkspace scratch,
                cudaStream_t stream) {
    constexpr int kWaveCtas = 170;
    const int ctas          = (Geometry::kOutputRows / 128) * ((x.ne[1] + 127) / 128);
    const bool wide         = x.ne[1] > 768 && (ctas >= 4 * kWaveCtas || 2 * (ctas % kWaveCtas) >= kWaveCtas ||
                                                ctas % kWaveCtas == 0);
    if (!wide) return mxfp8_linear_f16acc<Geometry, Mxfp8P2<Mxfp8TmaT64R64>>(x, weight, out, scratch, stream);
    mxfp8_linear_f16acc<Geometry, Mxfp8Tma<Mxfp8TmaT128R128>>(x, weight, out, scratch, stream);
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
