#pragma once

// infernix::ops - sigmoid_mul's arithmetic on a BF16 pair (FP32 expf sigmoid, one rounding), shared by its kernels
// (ops/kernel/sigmoid_gate_mul.cuh) and the producer that encodes its result for a GGUF projection
// (ops/ggml/ggml_encode_producers.cu).

#include "ops/common/math.cuh"

#include <cuda_bf16.h>

namespace infernix::ops {

__device__ __forceinline__ __nv_bfloat162 sigmoid_gate_mul_pair(__nv_bfloat162 gate, __nv_bfloat162 x) {
    const float r0 = __low2float(x) * sigmoid(__low2float(gate));
    const float r1 = __high2float(x) * sigmoid(__high2float(gate));
    return __floats2bfloat162_rn(r0, r1);
}

} // namespace infernix::ops
