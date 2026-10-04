#pragma once

#include "core/tensor.h"
#include "ops/offloaded_sparse_moe/cpu/w4a4_expert.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * Routed experts of one MoE layer with experts held outside device memory
 * (docs/maintainer/qwen3_8-flash-next-design.md §8.4-8.5, §16.2). For each column t:
 *
 *   logits     = FP32 [router; shared gate] x[:, t]             -> projection_fp32
 *   p          = softmax(router logits[0:E, t]);  ids = top-k of p (lower id wins exact ties)
 *   w_i        = p[ids_i] / sum_j p[ids_j]                      -> moe_route
 *   s          = sigmoid(logits[E, t])  (the shared-expert gate)  -> moe_route
 *   y_i        = expert ids_i applied to x[:, t] with the canonical W4A4 arithmetic
 *   y[:, t]    = bf16(sum_i w_i * y_i + s * y_shared[:, t])       -> moe_combine
 *
 * Expert placement is an execution resource, not a semantic input: an expert's record is read
 * from a device frame when `frames[e] >= 0` and from the pinned host bank otherwise, and its
 * BF16 output bits are identical either way and identical to the CPU engine's
 * (ops/offloaded_sparse_moe/cpu/w4a4_expert.h). The routed products are exact integer sums, so
 * no batch shape, split or order changes a bit of an expert output. Router logits stay FP32, as in
 * the Qwen3.5 sparse MoE: rounding them to BF16 would turn experts within one BF16 step of the
 * top-k boundary into ties. Each logit is one FP32 dot product whose order depends only on H, so
 * a column's routing is identical in every batch shape. Routing weights, the shared gate and the
 * combine are FP32 evaluations rounded once; their oracle is the formula in FP64.
 */

struct MoeRouting {
    Tensor ids;         // I32 [k, T]
    Tensor weights;     // FP32 [k, T]
    Tensor shared_gate; // FP32 [T]
};

/// logits: FP32 [E + 1, T], the router rows followed by the shared-expert gate row.
void moe_route(const Tensor& logits, std::int32_t top_k, MoeRouting& routing, cudaStream_t stream);

/// Device-side dispatch of the routed (column, slot) entries by expert. Every array is
/// caller-owned device memory sized for `max_entries = k * T` entries and E experts.
struct MoeDispatch {
    std::int32_t* counts  = nullptr; // [E]
    std::int32_t* offsets = nullptr; // [E + 1]
    std::int32_t* cursor  = nullptr; // [E] scratch
    std::int32_t* entries = nullptr; // [k*T]: t * k + slot, grouped by expert
    std::int32_t* jobs    = nullptr; // [E]: experts with entries, ascending id
    std::int32_t* job_count = nullptr; // [1]
};

[[nodiscard]] std::size_t moe_dispatch_bytes(std::int32_t experts, std::int32_t entries);
[[nodiscard]] MoeDispatch carve_moe_dispatch(void* base, std::int32_t experts, std::int32_t entries);
void moe_dispatch(const MoeRouting& routing, std::int32_t experts, MoeDispatch& dispatch,
                  cudaStream_t stream);

/// The layer's routed expert records: expert e's record is at frame_base + frames[e] *
/// record_stride when frames[e] >= 0, else at host_records + e * record_stride (pinned host
/// memory). scales: device [E] ExpertScales of the layer.
///
/// Non-resident records are first copied to `staging_slots` device slots (staging_base + i *
/// record_stride) by a few CTAs that keep their reads within a compact address window, then read
/// from there: pinned memory read by many CTAs at scattered offsets, as the expert kernels would,
/// runs at under half the copy rate on the RTX 5090 (design section 8.6). The jobs run in passes
/// of staging_slots jobs, so every miss is staged; staging_slots == 0 reads misses zero-copy.
/// The staging slots are scratch of this call and are shared by every layer.
struct MoeExpertSource {
    const std::uint8_t* frame_base   = nullptr;
    const std::int32_t* frames       = nullptr; // device [E]
    const std::uint8_t* host_records = nullptr;
    std::uint64_t record_stride      = 0;
    const offloaded_moe::ExpertScales* scales = nullptr;
    std::uint8_t* staging_base  = nullptr;
    std::int32_t staging_slots  = 0;
};

[[nodiscard]] std::size_t moe_experts_workspace_bytes(std::int32_t max_jobs, std::int32_t entries);

/// x: BF16 [H, T]. outputs: BF16 [H, k*T], column t*k + slot receives expert ids[slot, t]'s
/// output for column t. max_jobs bounds the job count the grid covers (min(E, k*T)).
void moe_experts(const Tensor& x, const MoeDispatch& dispatch, const MoeExpertSource& source,
                 std::int32_t top_k, std::int32_t max_jobs, void* workspace, Tensor& outputs,
                 cudaStream_t stream);

/// y = bf16(sum_slot w * outputs + shared_gate * shared), BF16 [H, T].
void moe_combine(const Tensor& outputs, const MoeRouting& routing, const Tensor& shared, Tensor& y,
                 cudaStream_t stream);

} // namespace ninfer::ops
