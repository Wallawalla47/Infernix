#pragma once
#include "core/weight.h"
#include "core/tensor.h"
#include "infernix/ops/lora.h"
#include <cuda_runtime.h>

namespace infernix::ops::detail {
// `lora`: the LoRA term over the logical rows [q 6144, k 1024, gate 6144, v 1024] (the single-parent order;
// include/infernix/ops/lora.h) and the call's columns, added to each projected value before its BF16 store; null
// keeps every kernel as it was.
void q4_q5_attn_input_small_t_launch(const Tensor& x, const Weight& query_key_weight,
                                     const Weight& gate_value_weight, Tensor& q, Tensor& gate,
                                     Tensor& k, Tensor& v, cudaStream_t stream, const LoraAddend* lora = nullptr);
void q4_q5_attn_input_column_invariant_launch(const Tensor& x, const Weight& query_key_weight,
                                              const Weight& gate_value_weight, Tensor& q,
                                              Tensor& gate, Tensor& k, Tensor& v,
                                              cudaStream_t stream, const LoraAddend* lora = nullptr);
void q4_q5_attn_input_mixed_r32_c32_s2_launch(const Tensor& x, const Weight& w0, const Weight& w1,
                                              Tensor& q, Tensor& g, Tensor& k, Tensor& v,
                                              cudaStream_t stream, const LoraAddend* lora = nullptr);
void q4_q5_attn_input_mixed_r32_c64_s3_launch(const Tensor& x, const Weight& w0, const Weight& w1,
                                              Tensor& q, Tensor& g, Tensor& k, Tensor& v,
                                              cudaStream_t stream, const LoraAddend* lora = nullptr);
void q4_q5_attn_input_mixed_ring_c128_launch(const Tensor& x, const Weight& w0, const Weight& w1,
                                               Tensor& q, Tensor& g, Tensor& k, Tensor& v,
                                               cudaStream_t stream, const LoraAddend* lora = nullptr);
// The ring tiles alone over every column (the launcher above sends a short remainder to the narrow routes).
void q4_q5_attn_input_mixed_ring_c128_launch_tiles(const Tensor& x, const Weight& w0, const Weight& w1, Tensor& q,
                                                   Tensor& g, Tensor& k, Tensor& v, cudaStream_t stream,
                                                   const LoraAddend* lora = nullptr);
void q4_q5_attn_input_grouped_mma_r32_c64_s4_launch(const Tensor& x, const Weight& query_key_weight,
                                                    const Weight& gate_value_weight, Tensor& q,
                                                    Tensor& gate, Tensor& k, Tensor& v,
                                                    cudaStream_t stream, const LoraAddend* lora = nullptr);
} // namespace infernix::ops::detail
