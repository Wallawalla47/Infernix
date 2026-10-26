#pragma once

// A Qwen3.5 MoE layer with offloaded routed experts (docs/maintainer/qwen3_5-122b-offload-design.md
// §8.5-8.6, §9.1): the router and the shared expert's gate as FP32 logits read at BF16
// (projection_fp32, ops::RouterLogits::kBf16), the route and dispatch with the shared expert as the
// call's last job, the routed and shared experts in one ops::moe_experts call over the Program's
// expert sources (runtime::expert_offload::ExpertOffload), and the combine into the residual. Every
// expert route is exact W4A16, so the layer's bits depend on neither the call's width nor where a
// record was read; with an expert-a4-residual fast path (OffloadedExperts::wide_a16) an expert of more
// than eight columns in a call, routed or shared, computes from A4 terms, so its bits then depend on
// its column count in the call (not on where its record was read: CPU-served misses have at most
// eight columns and are exact, as the GPU's narrow experts are).

#include "core/arena.h"
#include "core/device.h"
#include "core/host_memory.h"
#include "infernix/ops/offloaded_sparse_moe.h"
#include "models/qwen3_5/execution/parameters.h"
#include "runtime/expert_offload/expert_offload.h"

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace infernix::models::qwen3_5::execution {

// The Program's offloaded experts as the forward reads them, built once after its ExpertOffload has
// started: every pointer is stable for the Program's life, so captured rounds keep them.
struct OffloadedExperts {
    // The engine (a prefill pass enqueues its route download after its layers).
    runtime::expert_offload::ExpertOffload* engine = nullptr;
    // Per offloaded layer (OffloadedMoeParameters::layer): ExpertOffload::source with the shared job
    // (shared_expert, shared_record; scales holds E + 1 entries).
    std::vector<ops::MoeExpertSource> sources;
    std::int32_t* route_log  = nullptr; // layer l's routed ids at route_log + l * route_stride
    std::size_t route_stride = 0;
    // Graphed rounds' landing frames (ExpertOffload::landing_table, landed_log).
    const std::int32_t* landing = nullptr;
    std::int32_t* landed        = nullptr;
    std::int32_t landing_slots  = 0;
    // --expert-tail-skip's rank for graphed rounds (0: off; MoeExpertSource::tail_skip_rank).
    std::int32_t tail_skip_rank = 0;
    // The wide route of every call (MoeExpertSource::wide_a16): exact, or the expert-a4-residual fast
    // paths' two or three A4 terms (ExecutionSchedule::expert_wide, which sized the workspace).
    ops::offloaded_moe::WideA16 wide_a16 = ops::offloaded_moe::WideA16::kExact;
    // router-mma: calls of at least kRouterMmaMinColumns columns (prefill chunks and walk calls, never a
    // decode or verification round) compute the router and shared-gate logits on tensor cores
    // (ops::projection_fp32_router_mma: BF16 operands, FP32 accumulation, a column's bits the same at
    // every such width).
    bool router_mma = false;
    // The side stream: a graphed round's fork (its first two events) or an eager call's staging
    // overlap (all five).
    cudaStream_t side_stream = nullptr;
    std::array<cudaEvent_t, 5> side_events{};
    // A dense shared expert's stream (null for shared expert banks) and its fork and join events: the
    // shared expert runs there while the routed experts stage and compute (Qwen4Exp's pattern).
    cudaStream_t shared_stream = nullptr;
    std::array<cudaEvent_t, 2> shared_events{};
    // The prefill expert stream (null without one) and the prefill CPU split.
    runtime::expert_offload::ExpertStream* stream      = nullptr;
    const runtime::expert_offload::CpuSplitPolicy* split = nullptr;
    // A gated split's host scratch: the layer's dispatch offsets (pinned), their event, the experts'
    // columns and the CPU's choice.
    PinnedHostBuffer split_offsets{4};
    cudaEvent_t split_ready = nullptr;
    std::vector<std::int32_t> split_columns;
    std::vector<std::uint8_t> split_cpu;
    std::uint64_t split_cpu_experts = 0, split_streamed_experts = 0;
};

// One call's route through the offloaded experts.
struct OffloadedCall {
    bool graphed        = false; // a decode or verification round: one pass, the fork and the landings
    bool streamed       = false; // a prefill chunk whose misses the expert stream copies ahead
    bool split          = false; // its gated layers give the CPU their thinnest misses
    bool release_stream = true;  // each layer's ring half is released once its experts are enqueued
    // A verification round's live columns (rows of mask.width columns; device valid counts per row):
    // the padding past a row's drafts routes nowhere, so it adds no job, miss or prefetch (live
    // columns keep their bits). Empty for every other call.
    ops::MoeColumnMask mask;
    // A pass wider than kOffloadedMoeStaticColumns: the whole pass's workspace, lent from the expert
    // frames (ProgramOffload::begin_pass; prefill_text_chunk and prefill_multimodal_chunk run on it);
    // empty for a narrower call, which uses the Program's workspace.
    DeviceSpan workspace;
};

struct OffloadedMoeContext {
    OffloadedExperts* experts     = nullptr;
    OffloadedCall call;
    const BlockParameters* next = nullptr; // the next layer (its weights are warmed into L2 during the CPU wait)
};

// The widest eager pass (prefill, replay, scoring, forced tokens) the Program's workspace holds. A
// wider pass (a prefill chunk, whose buffers are the Program's largest) runs on a workspace the expert
// cache lends for the pass, so decode keeps that VRAM as cache (design §10). Decode and verification
// rounds are planned at their own widths.
inline constexpr std::int32_t kOffloadedMoeStaticColumns = 256;

// router-mma's narrowest call (OffloadedExperts::router_mma), as Flash-Next's.
inline constexpr std::int32_t kRouterMmaMinColumns = 256;

// The layer's workspace for every call of `first`..`last` columns on the experts' wide route `wide_a16`.
[[nodiscard]] std::size_t offloaded_moe_workspace_bytes(const OffloadedMoeParameters& p, std::int32_t first,
                                                        std::int32_t last, ops::offloaded_moe::WideA16 wide_a16);
// residual += moe(hidden) for BF16 hidden and residual [H, T].
void offloaded_moe(const Tensor& hidden, const OffloadedMoeParameters& p, Tensor& residual,
                   const OffloadedMoeContext& context, WorkspaceArena& workspace, DeviceExecutionView execution);

} // namespace infernix::models::qwen3_5::execution
