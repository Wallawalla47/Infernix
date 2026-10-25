#pragma once

// f16-accumulate-prefill on Q8 (q8_g32_fp16; Qwen3.8-Flash-Next Dense8's dense projections): int8 codes
// and FP16-encoded activations on FP16 Tensor Cores with FP16 accumulation over each 32-value group and
// an FP32 carry folded by the group's FP16 scale (ops/linear/f16acc/f16acc_mma.cuh, F16AccScales::Fp16).
// It serves LinearRoutes::F16Accumulate from kQ8F16MinTokens columns on the registered problems; every
// other call keeps the exact Q8 routes.

#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace infernix::ops::detail {

// The route's admission (prefill widths), as the FP8 and MXFP8 problems'. Not yet timed on Q8: the op
// bench (linear_bench --qtype q8 --routes f16) sets it per problem.
inline constexpr std::int32_t kQ8F16MinTokens = 256;

// Whether [n, k] registers the route: Flash-Next's GDN input [16384,2560], QSA group [13952,2560] and
// [12800,2560], [2560,2560], shared expert [1280,2560], output [2560,6144] and [2560,640].
[[nodiscard]] bool q8_f16acc_problem(std::int32_t n, std::int32_t k) noexcept;

// Whether a call of `tokens` columns on [n, k] takes the route: a registered problem from
// kQ8F16MinTokens columns. TEMPORARY (INFERNIX_TMP_Q8_F16ACC_ADMIT=1), until its A/B: past 8,192 tokens
// only the problems the row-fast raster measured faster than the exact one (at 32,768 tokens: [13952,2560]
// 0.79x, [2560,2560] 0.85x, [2560,640] 0.98x; Q8 has since kept its own rasters, f16acc_mma.cuh).
[[nodiscard]] bool q8_f16acc_admits(std::int32_t n, std::int32_t k, std::int32_t tokens) noexcept;

// The encoded activation a call of up to max_tokens columns needs (0 when the route cannot run).
[[nodiscard]] std::size_t q8_f16acc_workspace_capacity_bytes(std::int32_t n, std::int32_t k, std::int32_t max_tokens);

// out [n, T] BF16 on the route (a registered problem, T >= kQ8F16MinTokens), the encoded activation in
// `workspace` (scoped).
void q8_linear_f16acc(const Tensor& x, const Weight& weight, Tensor& out, WorkspaceArena& workspace,
                      cudaStream_t stream);

// The [16384,2560] GDN projection on the route: rows [0,10240) to qkv [10240,T], rows [10240,16384) to
// z [6144,T] (gdn_input_proj), the same bits as q8_linear_f16acc's rows.
void q8_gdn_input_f16acc(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z, WorkspaceArena& workspace,
                         cudaStream_t stream);

} // namespace infernix::ops::detail
