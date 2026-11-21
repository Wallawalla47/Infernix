#pragma once

#include "core/weight.h"
#include "core/tensor.h"
#include "infernix/ops/lora.h"

#include <cuda_runtime.h>

namespace infernix::ops::detail {

// `lora`: a LoRA term (include/infernix/ops/lora.h) over the gate/up parent's rows and the call's columns, added to
// each gate and up value before the SwiGLU (the adapter instantiation of the same kernel; null keeps it as it was).
void q4_linear_swiglu_gemv_pair_launch(const Tensor& x, const Weight& w, Tensor& out,
                                       cudaStream_t stream, const LoraAddend* lora = nullptr);
// The folded gate/up ring GEMM (q4_linear_swiglu_gemm_ring.cuh): 64 output rows x 128 columns a CTA, with a
// narrow route for a small remainder past the last whole column tile.
void q4_linear_swiglu_mma_ring_r64_c128_launch(const Tensor& x, const Weight& w, Tensor& out,
                                               cudaStream_t stream, const LoraAddend* lora = nullptr);
void q4_linear_swiglu_small_t_tiled_launch(const Tensor& x, const Weight& w, Tensor& out,
                                           cudaStream_t stream, const LoraAddend* lora = nullptr);
// The same family at any width (Q4LinearSwiGluScheduleId::SmallTTiledColumnInvariant).
void q4_linear_swiglu_small_t_invariant_launch(const Tensor& x, const Weight& w, Tensor& out,
                                               cudaStream_t stream, const LoraAddend* lora = nullptr);

} // namespace infernix::ops::detail
