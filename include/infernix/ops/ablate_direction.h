#pragma once

// Rank-1 direction ablation (docs/ablate.md): y -= alpha * dot(d, y_col) * d, per token
// column — the runtime form of the output-projection surgery W' = (I - alpha d d^T) W, applied at
// the residual-write site instead of to the weights. Deterministic: fixed thread-to-element mapping
// and a fixed block-reduction order, so ablated runs are bitwise reproducible.

#include <cuda_runtime.h>

#include <cstdint>

namespace infernix::ops {

// y_bf16: BF16 [H, T], token columns contiguous with stride H. d: FP32 [H] device, unit-norm at
// load. alpha: this layer's strength; 0 returns without launching.
void ablate_direction(void* y_bf16, std::uint32_t h, std::uint32_t t, const float* d, float alpha,
                      cudaStream_t stream);

}  // namespace infernix::ops
