#pragma once

// hyper_connection_inject's value, bf16(R + inject * y) in FP32 with one rounding: its kernel and the
// fused mixer's folded inject (hyper_connection_inject_mix) both form it here, so they round alike.
// Private to the Op.

#include <cuda_bf16.h>

namespace infernix::ops::detail {

__device__ __forceinline__ __nv_bfloat16 hc_injected(float r, float inject, float y) {
    return __float2bfloat16_rn(r + inject * y);
}

} // namespace infernix::ops::detail
