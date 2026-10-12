#pragma once

// infernix::ops - gdn_gating kernel: elementwise GDN gate prep over [48,T].
// Transcendentals use fp32 CUDA math functions, not polynomial approximations.

#include "ops/common/math.cuh"

#include <cuda_bf16.h>

#include <cmath>
#include <cstdint>

namespace infernix::ops {

// a and b hold each column's 48 heads contiguously, at column strides a_ld and b_ld (elements), so
// both may be row slices of one control projection [a; b].
__global__ void gdn_gating_kernel(const __nv_bfloat16* a, const __nv_bfloat16* b, std::int64_t a_ld,
                                  std::int64_t b_ld, const float* A_log, const float* dt_bias, float* g,
                                  float* beta, std::int64_t n) {
    const std::int64_t start  = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = start; i < n; i += stride) {
        const int h           = static_cast<int>(i % 48);
        const std::int64_t t  = i / 48;
        const float av        = __bfloat162float(a[t * a_ld + h]);
        const float bv        = __bfloat162float(b[t * b_ld + h]);
        const float sp = softplus(av + dt_bias[h]);
        g[i]           = -expf(A_log[h]) * sp;
        beta[i]        = sigmoid(bv);
    }
}

} // namespace infernix::ops
