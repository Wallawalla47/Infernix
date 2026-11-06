#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "infernix/ops/linear.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace infernix::ops::detail {

// MXFP8 (fp8_e4m3fn_g32_e8m0) projections run on registered shapes only. The exact route (16-bit
// activations) needs no workspace; the query covers the f16-accumulate-prefill and block32 pair routes'
// scratch wherever the interval reaches their admissions (the query takes no route flags).
[[nodiscard]] std::size_t mxfp8_linear_workspace_capacity_bytes(std::int32_t output_rows, std::int32_t input_rows,
                                                                LinearPolicy policy, std::int32_t min_tokens,
                                                                std::int32_t max_tokens);

// Under LinearRoutes::F16Accumulate (and not A16ColumnInvariant) a call at or above its problem's
// admission runs the FP16-accumulate route in `workspace`; otherwise under LinearRoutes::Fp8A8PairBlock32
// a call at or above the pair's admission runs the two-term A8 route (mxfp8_shapes.h); every other call
// runs the exact route.
void mxfp8_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy, LinearRoutes routes,
                    WorkspaceArena* workspace, cudaStream_t stream);

// gdn_input_proj's Qwen3.8-Flash-Next MXFP8 [16384,2560] parent: linear's route for the shape, its
// q/k/v rows [0,10240) written to qkv and its z rows to z (validated by the caller). With a route flag
// and a workspace, a call of that route's admission width or more runs it as linear does.
void mxfp8_gdn_input_n16384_k2560(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z, LinearRoutes routes,
                                  WorkspaceArena* workspace, cudaStream_t stream);

// The route scratch the [16384,2560] parent's GDN form needs over the interval (0 below the admissions).
[[nodiscard]] std::size_t mxfp8_gdn_input_workspace_capacity_bytes(std::int32_t max_tokens);

// The exact route's calls of at least this many tokens walk the rows of one token tile before the next
// (mxfp8_launch.cuh Mxfp8RowFast; candidates RV-44): FP4-Local, BF16 KV (the exact route's prefill), -p N -n 1, two
// rotated runs a point (2026-11-04): from 4,096 or 8,192 tokens against never, pp8192 1.000x / 0.999x, pp16384 1.000x /
// 1.000x, pp32768 1.020x / 1.021x (the layer walk's wide calls). Same bits.
inline constexpr std::int64_t kMxfp8RowFastTokens = 16384;

} // namespace infernix::ops::detail
