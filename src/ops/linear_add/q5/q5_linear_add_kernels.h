#pragma once
#include "core/weight.h"
#include "core/tensor.h"
#include "infernix/ops/lora.h"
#include <cuda_runtime.h>

namespace infernix::ops::detail {
// `lora`: a LoRA term (include/infernix/ops/lora.h) over the call's columns, added to each projected value before the
// residual add (the adapter instantiation of the same kernel; null keeps the kernel as it was).
void q5_linear_add_split2_exact_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t,
                                       const LoraAddend* lora = nullptr);
void q5_linear_add_sliced_r16_t8_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t,
                                        const LoraAddend* lora = nullptr);
void q5_linear_add_sliced_r16_t16_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t,
                                         const LoraAddend* lora = nullptr);
void q5_linear_add_sliced_r16_t24_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t,
                                         const LoraAddend* lora = nullptr);
void q5_linear_add_sliced_r32_t32_w4_s2_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t,
                                               const LoraAddend* lora = nullptr);
void q5_linear_add_sliced_r32_t24_pairwise_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t,
                                                  const LoraAddend* lora = nullptr);
void q5_linear_add_sliced_r32_t32_w4_s1_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t,
                                               const LoraAddend* lora = nullptr);
void q5_linear_add_sliced_r32_t32_w2_s2_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t,
                                               const LoraAddend* lora = nullptr);
void q5_linear_add_sliced_r32_t64_w2_s1_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t,
                                               const LoraAddend* lora = nullptr);
void q5_linear_add_sliced_invariant_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t,
                                           const LoraAddend* lora = nullptr);
void q5_linear_add_ksplit_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t,
                                 const LoraAddend* lora = nullptr);
void q5_linear_add_mma_r32_t32_k128_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t,
                                           const LoraAddend* lora = nullptr);
void q5_linear_add_mma_r32_t128_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t,
                                       const LoraAddend* lora = nullptr);
void q5_linear_add_ring_t128_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t,
                                    const LoraAddend* lora = nullptr);
} // namespace infernix::ops::detail
