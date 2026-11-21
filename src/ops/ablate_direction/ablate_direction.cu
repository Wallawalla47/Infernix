// Rank-1 direction ablation (docs/ablate.md): per token column, s = sum_h d[h] * y[h];
// y[h] -= alpha * s * d[h]. One block per column, 256 threads, H-strided loops; the reduction's
// order is fixed per launch, so ablated runs are bitwise reproducible (the regression tests rely
// on that). H = 2560 on Qwen3.8-Flash-Next (the measured write width); the loop form keeps the
// op generic. The kernel reads y for the dot product, syncs, then reads y again for the update —
// no write lands before the barrier, so both loops see the same values.

#include "infernix/ops/ablate_direction.h"

#include <cuda_bf16.h>

namespace {

constexpr int kThreads = 256;
constexpr int kWarps   = kThreads / 32;

__global__ void ablate_direction_kernel(__nv_bfloat16* __restrict__ y, const float* __restrict__ d,
                                        float alpha, int h) {
    __shared__ float warp_sum[kWarps];
    __nv_bfloat16* column = y + static_cast<size_t>(blockIdx.x) * h;

    float s = 0.0F;
    for (int i = threadIdx.x; i < h; i += kThreads) {
        s += d[i] * __bfloat162float(column[i]);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        s += __shfl_down_sync(0xffffffffu, s, offset);
    }
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) { warp_sum[warp] = s; }
    __syncthreads();
    if (warp == 0) {
        float v = (lane < kWarps) ? warp_sum[lane] : 0.0F;
        for (int offset = 16; offset > 0; offset >>= 1) {
            v += __shfl_down_sync(0xffffffffu, v, offset);
        }
        if (lane == 0) { warp_sum[0] = v; }
    }
    __syncthreads();

    const float g = alpha * warp_sum[0];
    for (int i = threadIdx.x; i < h; i += kThreads) {
        column[i] = __float2bfloat16(__bfloat162float(column[i]) - g * d[i]);
    }
}

}  // namespace

void infernix::ops::ablate_direction(void* y_bf16, const std::uint32_t h, const std::uint32_t t,
                                     const float* const d, const float alpha,
                                     cudaStream_t stream) {
    if (t == 0 || h == 0 || alpha == 0.0F) { return; }
    ablate_direction_kernel<<<t, kThreads, 0, stream>>>(
        static_cast<__nv_bfloat16*>(y_bf16), d, alpha, static_cast<int>(h));
}
