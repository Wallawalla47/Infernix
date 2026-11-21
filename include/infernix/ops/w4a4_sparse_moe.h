#pragma once

// Inkling 2's routed experts (docs/maintainer/inkling2_mini-design.md §10): 128 experts, top 8, softcapped SwiGLU,
// every expert device-resident, in the trained arithmetic of its layer:
//
//   NVFP4 (four-over-six W4A4, nvfp4_expert_rg16_v1): x and h are four-over-six NVFP4 rows (canon::four_over_six_*,
//     a per-row amax and an E4M3 scale per 16 elements); a gate/up or down output is
//       bf16((fl32(S) * 2^-20) * rs),  rs = (amax_row * amax_w[e]) * rcp_rn(1536 * denominator),
//     with S the exact int64 sum of the decoded products (ops/common/a4_mma.cuh), so the dot product is exact and
//     rounded once to binary32;
//   MXFP8 (W8A8, mxfp8_expert_rg16_v1): x and h are MXFP8 rows (E4M3 codes, a UE8M0 power of two per 32 elements);
//     a gate/up or down output is bf16(acc), acc the block-scaled FP32 tensor-core sum in one fixed K order.
//
//   h = bf16(((softcap_c(g, gate_cap) * sigmoid_c(g)) * softcap_c(u, up_cap)) * gamma), each product rounded,
//   g and u the BF16 gate and up outputs, gamma the entry's routing weight.
//
// Every entry (column t, slot j) is computed from its own column alone, in an order fixed by the bank's geometry, so
// a column's outputs never depend on the call's other columns (decode and verify columns agree bit for bit). The
// routed outputs are BF16 [entries][H], entry t * top_k + j; the caller combines a column's top_k outputs.

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace infernix::ops::w4a4_moe {

inline constexpr std::int32_t kTopK       = 8;
inline constexpr std::int32_t kMaxColumns = 128; // decode and verify: B lanes x T columns (a workspace may hold
                                                 // more: its capacity)
inline constexpr std::int32_t kMaxCapacity = 8191; // the dispatch's 16-bit entry counts: 8191 * 8 entries

enum class ExpertFormat : std::uint8_t { Nvfp4FourOverSix, Mxfp8 };

inline constexpr std::uint32_t kErrorDownStalled = 1U;

// One layer's experts [E, H, I]: records at records + e * record_stride (the bank's layout), and for NVFP4 the
// checkpoint's FP32 amax [E][2] (gate/up, down) and denominators [2].
struct Bank {
    const std::uint8_t* records  = nullptr;
    std::uint64_t record_stride  = 0;
    ExpertFormat format          = ExpertFormat::Nvfp4FourOverSix;
    const float* amax            = nullptr;
    const float* denominators    = nullptr;
    std::int32_t experts         = 0;
    std::int32_t hidden          = 0;
    std::int32_t intermediate    = 0;
    float gate_softcap           = 4.0F;
    float up_softcap             = 25.0F;
};

// The bytes of one activation row of `width` elements: NVFP4 codes (two a byte, element 2j in byte j's low nibble)
// then one E4M3 scale word per 16 elements; MXFP8 one E4M3 code a byte then one UE8M0 word per 32 elements.
[[nodiscard]] constexpr std::int32_t row_bytes(ExpertFormat format, std::int32_t width) noexcept {
    return format == ExpertFormat::Nvfp4FourOverSix ? width / 2 + width / 16 : width + width / 32;
}

// The columns' quantized x (the MoE mixer's output): rows [T][row_bytes(H)], and for NVFP4 each row's amax [T].
struct Activations {
    const std::uint8_t* rows = nullptr;
    const float* amax        = nullptr;
    std::int32_t columns     = 0;
};

// The routing of the call's columns: experts [T][top_k] in slot order and their weights gamma [T][top_k].
struct Routing {
    const std::int32_t* experts = nullptr;
    const float* gamma          = nullptr;
};

// Device scratch of one call, carved by carve_workspace (sized for `capacity` columns and E experts): the router's
// logits and routing, the dispatch (entries grouped by expert, jobs in ascending expert id), h rows, their quantized
// form and amax, and the jobs' completion counts. The dispatch rebuilds every word a call reads and the router's last
// CTA resets its counter, so a captured call replays correctly; the memory must be zeroed once before its first use.
struct Workspace {
    std::int32_t capacity     = 0;       // the columns a call may have
    float* logits             = nullptr; // [capacity][E] the router's logits
    std::int32_t* routed      = nullptr; // [capacity * kTopK] the router's experts (moe_route's Routing)
    float* gamma              = nullptr; // [capacity * kTopK] their weights
    std::uint32_t* route_done = nullptr; // [1] router CTAs done (its last CTA resets it)
    std::int32_t* counts      = nullptr; // [E]
    std::int32_t* offsets     = nullptr; // [E + 1]
    std::int32_t* entries     = nullptr; // [capacity * kTopK]: t * kTopK + j, grouped by expert
    std::int32_t* jobs        = nullptr; // [E]: experts with entries, ascending
    std::int32_t* job_count   = nullptr; // [1]
    std::uint32_t* job_done   = nullptr; // [E]: per job, gate/up CTAs done (+1 once its h is quantized)
    std::uint16_t* h          = nullptr; // [capacity * kTopK][I] BF16
    std::uint32_t* h_amax     = nullptr; // [capacity * kTopK]: binary32 bits of max |h| (NVFP4)
    std::uint8_t* h_rows      = nullptr; // [capacity * kTopK][row_bytes(I)]
    // [1]: kErrorDownStalled once a down CTA's wait for its job made no progress for 2 s (zero: no error).
    std::uint32_t* error      = nullptr;
};

// The router of `columns` columns of x (BF16 [T][H], the MoE input), then the dispatch (moe_dispatch's) of its routing
// in its last CTA: logits l_e = sum_k x_k W[e, k] in binary32 (BF16 products are exact; one fixed order per column: a
// lane's strided elements in order, then a fixed warp tree); selection keys sigmoid_c(l_e) + bias_e (FP32); the top
// kTopK by key, ties to the lower expert index, in slots of descending key; gamma_j = (s_j / sum8(s)) * route_scale
// over the selected sigmoids s, sum8 the pairwise ((s0 + s2) + (s1 + s3)) + ((s4 + s6) + (s5 + s7)). W: BF16 [E][H]
// (E <= 256), bias FP32 [E]. The routing lands in the workspace (routed_routing).
void moe_route(const std::uint16_t* x, std::int32_t columns, const std::uint16_t* router, const float* bias,
               std::int32_t experts, std::int32_t hidden, float route_scale, const Workspace& workspace,
               cudaStream_t stream);

// The workspace's routing (moe_route's output) as a Routing.
[[nodiscard]] inline Routing routed_routing(const Workspace& workspace) noexcept {
    return Routing{workspace.routed, workspace.gamma};
}

// A workspace for calls of up to `capacity` columns (1..kMaxCapacity).
[[nodiscard]] std::size_t workspace_bytes(std::int32_t experts, std::int32_t intermediate,
                                         std::int32_t capacity = kMaxColumns);
[[nodiscard]] Workspace carve_workspace(void* base, std::int32_t experts, std::int32_t intermediate,
                                        std::int32_t capacity = kMaxColumns);

// Groups the call's (column, slot) entries by expert and clears the per-entry and per-job state the expert passes
// accumulate into (h_amax, job_done), so the call depends on nothing a previous call left.
void moe_dispatch(const Routing& routing, std::int32_t columns, std::int32_t experts, const Workspace& workspace,
                  cudaStream_t stream);

// The routed experts of the dispatched entries: outputs BF16 [columns * kTopK][H].
void moe_experts(const Activations& x, const Routing& routing, const Bank& bank, const Workspace& workspace,
                 std::uint16_t* outputs, cudaStream_t stream);

} // namespace infernix::ops::w4a4_moe
