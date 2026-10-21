#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "infernix/ops/linear.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace infernix::ops::detail {

// MXFP8 (fp8_e4m3fn_g32_e8m0) projections run with 16-bit activations on registered shapes only and
// need no workspace; the query validates the problem.
[[nodiscard]] std::size_t mxfp8_linear_workspace_capacity_bytes(std::int32_t output_rows, std::int32_t input_rows,
                                                                LinearPolicy policy, std::int32_t min_tokens,
                                                                std::int32_t max_tokens);

void mxfp8_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy, cudaStream_t stream);

// gdn_input_proj's Qwen3.8-Flash-Next MXFP8 [16384,2560] parent: linear's route for the shape, its
// q/k/v rows [0,10240) written to qkv and its z rows to z (validated by the caller).
void mxfp8_gdn_input_n16384_k2560(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z, cudaStream_t stream);

} // namespace infernix::ops::detail
