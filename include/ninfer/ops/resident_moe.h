#pragma once

// Routed experts whose banks live whole in device memory: the Qwen4Exp MTP drafter's 512 experts
// (design §11.2). Unlike offloaded_sparse_moe there is no residency, staging or host path, and
// the banks use the row-split grouped-integer codecs.

#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/// Workspace for resident_moe_experts over `pairs` = top_k * T (column, rank) entries.
[[nodiscard]] std::size_t resident_moe_workspace_bytes(std::int32_t pairs, std::int32_t intermediate);

/// For every column t of x (BF16 [H, T]) and rank j of ids (I32 [k, T]) with e = ids[j, t]:
///
///   h      = silu(G_e x_t) * (U_e x_t)            (FP32, I values)
///   out_jt = bf16(D_e h)                           written to outputs[:, t * k + j] (BF16 [H, k*T])
///
/// gate_up is the bank [E * 2I, H]: expert e's I gate rows at e*2I, then its I up rows; down is
/// [E * H, I]. Both are Q4_G64_FP16 or Q8_G32_FP16 row-split weights; each weight is decoded
/// exactly as code * FP16 group scale and every dot product accumulates in FP32 with the group
/// scale applied once per 16-weight (Q8) or 32-weight (Q4) chunk. ids outside [0, E) are a
/// precondition violation. The oracle is the same formula in FP64 over the decoded weights.
void resident_moe_experts(const Tensor& x, const Tensor& ids, const Weight& gate_up, const Weight& down,
                          std::int32_t experts, std::int32_t intermediate, void* workspace,
                          std::size_t workspace_bytes, Tensor& outputs, cudaStream_t stream);

} // namespace ninfer::ops
