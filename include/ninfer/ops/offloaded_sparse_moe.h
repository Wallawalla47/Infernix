#pragma once

#include "core/tensor.h"
#include "ops/offloaded_sparse_moe/cpu/miss_request.h"
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
 *   y_i        = expert ids_i applied to x[:, t] with the W4A4 arithmetic of design §16.2
 *   y[:, t]    = bf16(sum_i w_i * y_i + s * y_shared[:, t])       -> moe_combine
 *
 * Expert placement is an execution resource, not a semantic input: an expert's record is read
 * from a device frame when `frames[e] >= 0` and from the pinned host bank otherwise, and its
 * BF16 output bits are identical either way. An expert with at most eight columns in a call takes
 * the narrow route, whose products are exact integer sums: no batch shape, split or order changes
 * a bit of its output, and the CPU engine (ops/offloaded_sparse_moe/cpu/w4a4_expert.h) returns
 * the same bits. An expert with more columns (and one shared gate/up input scale) takes the wide
 * route: the same A4 activations and BF16 boundaries, with the block products summed by FP32
 * tensor cores (qualified against FP64; design §8.5, §13). Both routes depend only on the column
 * count and the stored scales, never on placement. Router logits stay FP32, as in
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
/// Fills `dispatch` from routing.ids: counts, offsets (exclusive scan, offsets[E] = k*T), jobs in
/// ascending expert id, job_count, and entries grouped by expert (their order within an expert is
/// unspecified). When `route_log` is not null it also receives a copy of the k*T ids. The arrays need
/// no clearing between calls, so a captured call replays correctly.
void moe_dispatch(const MoeRouting& routing, std::int32_t experts, MoeDispatch& dispatch,
                  std::int32_t* route_log, cudaStream_t stream);

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
/// CPU-served misses (design section 10.3). When enabled for a call (max_jobs > 0 and the call
/// has at most max_columns <= kMaxCpuCallColumns columns), up to max_jobs <= kMaxCpuJobs of its
/// non-resident experts with at most max_job_columns columns, the fewest-column ones first, are
/// published as a request in mapped host memory, with the x columns they read (compacted: the
/// request's column k is x's k-th column read by a chosen job, in column order), and computed by
/// the host's expert engine (offloaded_moe::CpuMissService) while the GPU computes the other jobs.
/// Calls of at least wide_from columns (prefill; 0: none) take wide_jobs as their cap instead.
/// moe_experts returns after placing the host's outputs. The arithmetic is the same exact W4A4,
/// so where an expert is computed never changes a bit. A host that does not answer within 2 s
/// traps the kernel.
struct MoeCpuChannel {
    offloaded_moe::MissRequest* request = nullptr; // mapped host memory
    std::uint16_t* x                    = nullptr; // mapped BF16 [H, kMaxCpuXColumns]
    const std::uint16_t* y              = nullptr; // mapped BF16 [H, max(max_jobs, wide_jobs) * kMaxCpuColumns]
    const std::uint32_t* done           = nullptr; // mapped; the host writes the answered sequence
    std::uint32_t* sequence             = nullptr; // device counter of published requests
    std::int32_t layer                  = 0;
    std::int32_t max_jobs               = 0;       // 0 disables
    std::int32_t max_columns            = 0;
    // misses / pcie_divisor of a call's misses stay on the GPU stage (0: the CPU takes up to
    // max_jobs of them), so the CPU and the PCIe stage share the call's misses.
    std::int32_t pcie_divisor           = 3;
    std::int32_t max_job_columns        = offloaded_moe::kMaxCpuColumns; // 1..kMaxCpuColumns
    // Prefill CPU assist (design §19.3.1 P7): calls of at least wide_from columns (0: none) take up
    // to wide_jobs <= kMaxCpuJobs CPU-served misses.
    std::int32_t wide_from              = 0;
    std::int32_t wide_jobs              = 0;
};

/// Device memory a call warms into L2 while it waits for its CPU-served misses (design §19.3.3
/// Phase 2): typically the next layer's first weights. Spans start 16-byte aligned; a span's bytes
/// are rounded down to 16. Warming changes no result.
struct MoeL2Warm {
    static constexpr int kSpans = 4;
    const void* ptr[kSpans]   = {};
    std::size_t bytes[kSpans] = {};
};

struct MoeExpertSource {
    const std::uint8_t* frame_base   = nullptr;
    const std::int32_t* frames       = nullptr; // device [E]
    const std::uint8_t* host_records = nullptr;
    std::uint64_t record_stride      = 0;
    const offloaded_moe::ExpertScales* scales = nullptr;
    std::uint8_t* staging_base  = nullptr;
    std::int32_t staging_slots  = 0;
    MoeCpuChannel cpu;
    // Optional overlap of staging with compute for calls of several passes (prefill chunks): the
    // slots are split in two halves and pass p+1 is staged on `overlap_stream` while pass p
    // computes. The stream and the five events (start, staged[2], consumed[2]) are caller-owned;
    // the stream is otherwise idle. Not for calls captured into a CUDA graph.
    cudaStream_t overlap_stream = nullptr;
    cudaEvent_t overlap_events[5] = {};
    // Optional fork for calls of one pass (decode and verification widths; may be captured into a
    // CUDA graph): the pass's misses are staged and computed on `fork_stream` while the calling
    // stream computes the resident experts. The calling stream joins the fork in
    // moe_experts_cpu_wait (immediately when wait_for_cpu is true). The stream and both events
    // (fork, join) are caller-owned; the stream is otherwise idle.
    cudaStream_t fork_stream = nullptr;
    cudaEvent_t fork_events[2] = {};
    // Optional landing for forked calls (design §19.3.5 S4; ignored by every other route): the n-th
    // staged miss in job order (CPU-served misses excluded) is copied into frame landing[n]
    // (frame_base + landing[n] * record_stride) instead of a staging slot when n < landing_slots
    // and landing[n] >= 0, and stage block 0 writes landed[n] = its expert; other entries of
    // `landed` are left as they are. Device arrays of landing_slots <= kMaxLandingSlots entries.
    // The frames must be free (no expert, not read by this call); the caller adopts the landed
    // experts afterwards. Results are unchanged: a landed record is read exactly as a staged one.
    const std::int32_t* landing = nullptr;
    std::int32_t* landed        = nullptr;
    std::int32_t landing_slots  = 0;
    // Prefetched into L2 by extra CTAs of the CPU wait while the host computes this call's misses
    // (only when a request was published).
    MoeL2Warm l2_warm;
    // Optional streamed records (design §19.3.8 F2): prefetched (device [E]) gives, for a
    // non-resident expert, the slot whose record the caller has already copied to prefetch_base +
    // slot * record_stride (ordered before this call), or -1. Such an expert is read there like a
    // resident one: never staged and never CPU-served (the CPU takes misses the stream left out).
    const std::int32_t* prefetched     = nullptr;
    const std::uint8_t* prefetch_base  = nullptr;
};

[[nodiscard]] std::size_t moe_experts_workspace_bytes(std::int32_t max_jobs, std::int32_t entries);

/// x: BF16 [H, T]. outputs: BF16 [H, k*T], column t*k + slot receives expert ids[slot, t]'s
/// output for column t. max_jobs bounds the job count the grid covers (min(E, k*T)).
///
/// With wait_for_cpu == false the call returns before placing CPU-served outputs; the caller may
/// enqueue unrelated work (the shared expert) and must call moe_experts_cpu_wait with the same
/// arguments before reading `outputs`. The workspace stays reserved until then.
void moe_experts(const Tensor& x, const MoeDispatch& dispatch, const MoeExpertSource& source,
                 std::int32_t top_k, std::int32_t max_jobs, void* workspace, Tensor& outputs,
                 cudaStream_t stream, bool wait_for_cpu = true);

/// Waits for the host's answer to a moe_experts call made with wait_for_cpu == false and places
/// its outputs; a no-op when that call published no request.
void moe_experts_cpu_wait(const Tensor& x, const MoeDispatch& dispatch, const MoeExpertSource& source,
                          std::int32_t max_jobs, void* workspace, Tensor& outputs, cudaStream_t stream);

/// y = bf16(sum_slot w * outputs + shared_gate * shared), BF16 [H, T].
void moe_combine(const Tensor& outputs, const MoeRouting& routing, const Tensor& shared, Tensor& y,
                 cudaStream_t stream);

} // namespace ninfer::ops
