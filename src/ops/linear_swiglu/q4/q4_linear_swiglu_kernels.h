#pragma once

#include "core/weight.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

namespace infernix::ops::detail {

void q4_linear_swiglu_gemv_pair_launch(const Tensor& x, const Weight& w, Tensor& out,
                                       cudaStream_t stream);
// The folded gate/up ring GEMM (q4_linear_swiglu_gemm_ring.cuh): 64 output rows x 128 columns a CTA, with a
// narrow route for a small remainder past the last whole column tile.
void q4_linear_swiglu_mma_ring_r64_c128_launch(const Tensor& x, const Weight& w, Tensor& out,
                                               cudaStream_t stream);
void q4_linear_swiglu_small_t_tiled_launch(const Tensor& x, const Weight& w, Tensor& out,
                                           cudaStream_t stream);
// The same family at any width (Q4LinearSwiGluScheduleId::SmallTTiledColumnInvariant).
void q4_linear_swiglu_small_t_invariant_launch(const Tensor& x, const Weight& w, Tensor& out,
                                               cudaStream_t stream);

} // namespace infernix::ops::detail
