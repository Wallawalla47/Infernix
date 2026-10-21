// Qwen3.8-Flash-Next GDN query/key/value/z projection, MXFP8 with BF16 activations (W8A16, NVIDIA
// FP4-Local). Every width runs the Tensor Core route, whose schedules accumulate a column in one
// order, so a column has the same bits at every T and with either producer (mxfp8_a16_tma_mma.cuh).
// The selection is bench/ops/mxfp8_flash_next_sweep.cu's (RTX 5090, 2026-10-20: the fastest schedule
// per width, merged within 2 %): cp.async to 192 columns, TMA from there. gdn_input_proj runs the same
// selection into the q/k/v and z destinations, so its rows hold exactly linear's bits.
#include "ops/linear/common/output.cuh"
#include "ops/linear/mxfp8/mxfp8_launch.cuh"
#include "ops/linear/mxfp8/mxfp8_shapes.h"

namespace infernix::ops::detail {
namespace {
using Geometry = Fp8Geometry<16384, 2560>;

template <class Output>
void select(const Mxfp8Operands& p, Output output, cudaStream_t stream) {
    const int tokens = p.tokens;
    if (tokens <= 16) return mxfp8_tma<Geometry, Mxfp8TmaT16R64>(p, output, stream);
    if (tokens <= 32) return mxfp8_tma<Geometry, Mxfp8P2<Fp8A16RingT32R64S3>>(p, output, stream);
    if (tokens <= 64) return mxfp8_tma<Geometry, Fp8A16RingT64R128>(p, output, stream);
    if (tokens <= 192) return mxfp8_tma<Geometry, Mxfp8P4<Mxfp8TmaT64R256>>(p, output, stream);
    if (tokens <= 256) return mxfp8_tma<Geometry, Mxfp8Tma<Mxfp8TmaT64R64>>(p, output, stream);
    if (tokens <= 512) return mxfp8_tma<Geometry, Mxfp8Tma<Fp8A16RingT64R128>>(p, output, stream);
    if (tokens <= 768) return mxfp8_tma<Geometry, Mxfp8Tma<Mxfp8TmaT128R128>>(p, output, stream);
    if (tokens <= 3072) return mxfp8_tma<Geometry, Mxfp8Tma<Fp8A16RingT64R128>>(p, output, stream);
    mxfp8_tma<Geometry, Mxfp8Tma<Mxfp8TmaT128R128>>(p, output, stream);
}

void launch_a16(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    select(mxfp8_operands(x, weight), LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), weight.n}, stream);
}
} // namespace

const Mxfp8LinearShape kMxfp8N16384K2560{16384, 2560, launch_a16};

void mxfp8_gdn_input_n16384_k2560(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z, cudaStream_t stream) {
    select(mxfp8_operands(x, weight),
           LinearBf16SegmentedOutput<10240, 6144>{
               {static_cast<__nv_bfloat16*>(qkv.data), static_cast<__nv_bfloat16*>(z.data)}},
           stream);
}
} // namespace infernix::ops::detail
