#pragma once

// infernix::ops - one Gated DeltaNet head's decay and update gates, shared by the gdn_gating kernel
// and the Gated DeltaNet batch update that evaluates them in place (same bits either way).

#include "ops/common/math.cuh"

#include <cuda_bf16.h>

#include <cmath>

namespace infernix::ops {

// ADirect: the third input holds A = -exp(A_log) itself (a GGUF import), used as stored.
template <bool ADirect = false>
__device__ __forceinline__ void gdn_gate_values(__nv_bfloat16 a, __nv_bfloat16 b, float A_log, float dt_bias,
                                                float& g, float& beta) {
    const float sp = softplus(__bfloat162float(a) + dt_bias);
    if constexpr (ADirect) {
        g = A_log * sp;
    } else {
        g = -expf(A_log) * sp;
    }
    beta = sigmoid(__bfloat162float(b));
}

} // namespace infernix::ops
