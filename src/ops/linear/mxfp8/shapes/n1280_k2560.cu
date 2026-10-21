// Qwen3.8-Flash-Next shared-expert gate/up projection, MXFP8 with BF16 activations (W8A16, NVIDIA
// FP4-Local). Every width runs the Tensor Core route, whose schedules accumulate a column in one
// order, so a column has the same bits at every T and with either producer (mxfp8_a16_tma_mma.cuh).
// The selection is bench/ops/mxfp8_flash_next_sweep.cu's (RTX 5090, 2026-10-20: the fastest schedule
// per width, merged within 2 %): cp.async except at 49-64 columns and from 385, where TMA is faster.
#include "ops/linear/mxfp8/mxfp8_launch.cuh"
#include "ops/linear/mxfp8/mxfp8_shapes.h"

namespace infernix::ops::detail {
namespace {
using Geometry = Fp8Geometry<1280, 2560>;

void launch_a16(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const int tokens = x.ne[1];
    if (tokens <= 48) return mxfp8_linear_a16_tma<Geometry, Mxfp8P2<Mxfp8TmaT16R32>>(x, weight, out, stream);
    if (tokens <= 64) return mxfp8_linear_a16_tma<Geometry, Mxfp8Tma<Mxfp8TmaT16R32>>(x, weight, out, stream);
    if (tokens <= 128) return mxfp8_linear_a16_tma<Geometry, Mxfp8P2<Mxfp8TmaT16R64>>(x, weight, out, stream);
    if (tokens <= 256) return mxfp8_linear_a16_tma<Geometry, Mxfp8TmaT32R32>(x, weight, out, stream);
    if (tokens <= 384) return mxfp8_linear_a16_tma<Geometry, Mxfp8P2<Fp8A16RingT32R64S3>>(x, weight, out, stream);
    if (tokens <= 512) return mxfp8_linear_a16_tma<Geometry, Mxfp8Tma<Mxfp8TmaT64R64>>(x, weight, out, stream);
    if (tokens <= 768) return mxfp8_linear_a16_tma<Geometry, Mxfp8Tma<Fp8A16RingT32R64S3>>(x, weight, out, stream);
    if (tokens <= 1024) return mxfp8_linear_a16_tma<Geometry, Mxfp8Tma<Fp8A16RingT64R128>>(x, weight, out, stream);
    if (tokens <= 1536) return mxfp8_linear_a16_tma<Geometry, Mxfp8Tma<Mxfp8TmaT64R64>>(x, weight, out, stream);
    if (tokens <= 2048) return mxfp8_linear_a16_tma<Geometry, Mxfp8Tma<Mxfp8TmaT128R128>>(x, weight, out, stream);
    if (tokens <= 3072) return mxfp8_linear_a16_tma<Geometry, Mxfp8Tma<Fp8A16RingT64R128>>(x, weight, out, stream);
    mxfp8_linear_a16_tma<Geometry, Mxfp8Tma<Mxfp8TmaT64R256>>(x, weight, out, stream);
}
} // namespace

const Mxfp8LinearShape kMxfp8N1280K2560{1280, 2560, launch_a16};
} // namespace infernix::ops::detail
