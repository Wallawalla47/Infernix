#pragma once

// Inkling 2's routed experts (docs/maintainer/inkling2_mini-design.md §10): 128 experts, top 8, softcapped SwiGLU,
// every expert device-resident, in the trained arithmetic of its layer:
//
//   NVFP4 (four-over-six W4A4, nvfp4_expert_rg16_v1): x and h are four-over-six NVFP4 rows (canon::four_over_six_*,
//     a per-row amax and an E4M3 scale per 16 elements), rs = (amax_row * amax_w[e]) * rcp_rn(1536 * denominator).
//     The decode and verify kernels (moe_single, moe_experts) take one of two cores (Activations::core), the same in
//     every kernel a decode or verify call reaches:
//       Native (the default): a gate/up or down output is bf16(acc * rs), acc an FP32 chain of 64-element steps on the
//         block-scaled e2m1 tensor core (mma mxf4nvf4 m16n8k64 with E4M3 block scales, ops/common/a4_native.cuh; K
//         slot blocks 0, 2, 1, 3 of each step): gate/up one chain a K half, acc = half 0 + half 1; down one chain in
//         block order;
//       Exact: bf16((fl32(S) * 2^-20) * rs) with S the exact int64 sum of the decoded products (ops/common/a4_mma.cuh),
//         the dot product rounded once to binary32;
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

// The activations the experts take: the checkpoint's (above) or, with --a16-activations (EngineOptions::
// expert_a16_activations), 16-bit ones:
//   NVFP4 layers: Flash-Next's canonical W4A16 arithmetic (docs/maintainer/qwen3_8-flash-next-design.md §16.2.1): x
//     (BF16, K = H) and h (K = I) each encoded once as X_j = a16_value(v_j, emax), 21-bit fixed point under the
//     column's largest exponent (canon::a16_column_exponent); P_b = sum_j c2(w) X_j exact in int32, S = sum_b P_b
//     e4m3_scaled(s_b) exact in int64; a gate/up or down output is canon::a16_row_output(S, emax, m) with
//     m = fl32(amax_w[e] / denominator) (the weights' c * s * amax / denominator). Exact, so a column's bits depend
//     on its own inputs alone.
//   MXFP8 layers: W8A16, x and h BF16 (no rows).
// or, with --a8-activations (EngineOptions::a8_activations; LOSSY), A8 on the NVFP4 layers (the MXFP8 layers keep
// their checkpoint W8A8): x (K = H) and h (K = I) each quantized per 32-element slice s to int8 codes q with an FP32
// scale d_s (ops/ggml/ggml_a8.h's codec, the dense projections' too), P_b = sum_j c2(w) q_j the exact int32 product
// of 16-element block b (|P_b| <= 12 * 16 * 127), and per output
//   acc = +0; for b = 0 .. K/16 - 1 in order: acc = fma(P_b * S_b, d_s(b), acc)   (P_b * S_b exact)
//   y   = bf16(fl32(fl32(acc * 2^-10) * m))                                         (nvfp4_a8::row_output)
// with S_b = canon::e4m3_scaled of the block's scale word and m = fl32(amax_w[e] / denominator), A16's multiplier:
// the wide 8-bit route's arithmetic (ops/offloaded_sparse_moe/cuda/wide_expert_a8.cuh), so every route that takes it
// gives an entry the same bits. h = the BF16 softcapped SwiGLU times gamma, as on every route, then its slices.
enum class ExpertActivation : std::uint8_t { Checkpoint, A16, A8 };

// The NVFP4 checkpoint arithmetic's core in the decode and verify kernels (the header's NVFP4 contract; no effect on
// A16, MXFP8 or the prompt route).
enum class Nvfp4Core : std::uint8_t { Native, Exact };

// An A16 limb row (a16_limbs.cuh's planes: per 64-element tile 64 H1, 64 L1, 64 H0 | L0 << 4 bytes).
inline constexpr std::int32_t kA16XRowBytes = 2304 / 64 * 192; // 6,912
inline constexpr std::int32_t kA16HRowBytes = 1152 / 64 * 192; // 3,456
// An A8 row (a8_encode_columns' column: K int8 codes, then K / 32 FP32 slice scales).
inline constexpr std::int32_t kA8XRowBytes = 2304 + 2304 / 8; // 2,592
inline constexpr std::int32_t kA8HRowBytes = 1152 + 1152 / 8; // 1,296

// The columns' x (the MoE mixer's output). Checkpoint: rows [T][row_bytes(H)], and for NVFP4 each row's amax [T].
// A16: x16, the BF16 rows [T][H]; NVFP4 layers also rows as limb rows [T][kA16XRowBytes] and emax [T] their exponents.
// A8 (NVFP4 layers): rows as A8 rows [T][kA8XRowBytes].
struct Activations {
    const std::uint8_t* rows    = nullptr;
    const float* amax           = nullptr;
    std::int32_t columns        = 0;
    ExpertActivation activation = ExpertActivation::Checkpoint;
    const std::int32_t* emax    = nullptr;
    const std::uint16_t* x16    = nullptr;
    Nvfp4Core core              = Nvfp4Core::Native;
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
    // A16 NVFP4 (a workspace carved for A16): h as limb rows [capacity * kTopK][kA16HRowBytes], their exponent a16_emax
    // of h_amax's BF16 bits.
    std::uint8_t* h_limbs     = nullptr;
    // A8 NVFP4 (a workspace carved for A8): h as A8 rows [capacity * kTopK][kA8HRowBytes].
    std::uint8_t* h_a8        = nullptr;
    // [1]: kErrorDownStalled once a down CTA's wait for its job made no progress for 2 s (zero: no error).
    std::uint32_t* error      = nullptr;
    // [10]: moe_single's counters (router CTAs, finished CTAs, each job's gate/up row groups), reset by its last CTA.
    std::uint32_t* single     = nullptr;
    // The prompt route's planes (moe_experts_prompt): every dispatched entry's A4(x) and A4(h) rows in dispatch order
    // (the grouped GEMM's tensor-map layout: kPromptXRowBytes and kPromptHRowBytes a row), its work tiles and each
    // job's record.
    std::uint8_t* x_plane             = nullptr; // [capacity * kTopK][kPromptXRowBytes]
    std::uint8_t* h_plane             = nullptr; // [capacity * kTopK][kPromptHRowBytes]
    std::int32_t* tiles               = nullptr; // [2 * (E + capacity * kTopK / 64 + 1)]
    std::int32_t* pass_tiles          = nullptr; // [2]
    const std::uint8_t** job_records  = nullptr; // [E]
};

// The prompt route's plane rows (NVFP4 layers, H = 2304, I = 1152): codes then E4M3 scales, 16-byte strides.
inline constexpr std::int32_t kPromptXRowBytes = 2304 / 2 + 2304 / 16;     // 1,296
inline constexpr std::int32_t kPromptHRowBytes = 1152 / 2 + 80;            // 576 code bytes, an 80-byte scale map

// The router of `columns` columns of x (BF16 [T][H], the MoE input), then the dispatch (moe_dispatch's) of its routing
// in its last CTA: logits l_e = sum_k x_k W[e, k] in binary32 (BF16 products are exact; one fixed order per column: a
// lane's strided elements in order, then a fixed warp tree); selection keys sigmoid_c(l_e) + bias_e (FP32; a NaN key,
// from a non-finite x, ranks below every other, so the slots always hold kTopK distinct experts); the top
// kTopK by key, ties to the lower expert index, in slots of descending key; gamma_j = (s_j / sum8(s)) * route_scale
// over the selected sigmoids s, sum8 the pairwise ((s0 + s2) + (s1 + s3)) + ((s4 + s6) + (s5 + s7)). W: BF16 [E][H]
// (E <= 256), bias FP32 [E]. The routing lands in the workspace (routed_routing).
// `record`, if set, receives the last column's kTopK experts in slot order (relaxed stores: a later call's prefetch
// hint, moe_prefetch_l2 reads it with relaxed loads, so a record rewritten while it is read is a defined, stale hint).
void moe_route(const std::uint16_t* x, std::int32_t columns, const std::uint16_t* router, const float* bias,
               std::int32_t experts, std::int32_t hidden, float route_scale, const Workspace& workspace,
               cudaStream_t stream, std::int32_t* record = nullptr);

// The workspace's routing (moe_route's output) as a Routing.
[[nodiscard]] inline Routing routed_routing(const Workspace& workspace) noexcept {
    return Routing{workspace.routed, workspace.gamma};
}

// A workspace for calls of up to `capacity` columns (1..kMaxCapacity); with A16 also the h limb rows.
[[nodiscard]] std::size_t workspace_bytes(std::int32_t experts, std::int32_t intermediate,
                                         std::int32_t capacity = kMaxColumns,
                                         ExpertActivation activation = ExpertActivation::Checkpoint);
[[nodiscard]] Workspace carve_workspace(void* base, std::int32_t experts, std::int32_t intermediate,
                                        std::int32_t capacity = kMaxColumns,
                                        ExpertActivation activation = ExpertActivation::Checkpoint);

// Groups the call's (column, slot) entries by expert and clears the per-entry and per-job state the expert passes
// accumulate into (h_amax, job_done), so the call depends on nothing a previous call left.
void moe_dispatch(const Routing& routing, std::int32_t columns, std::int32_t experts, const Workspace& workspace,
                  cudaStream_t stream);

// A8: the columns' x (BF16 [T][H]) as A8 rows [T][kA8XRowBytes] (Activations::rows).
void moe_encode_a8(const std::uint16_t* x, std::int32_t columns, std::uint8_t* rows, cudaStream_t stream);

// The routed experts of the dispatched entries: outputs BF16 [columns * kTopK][H]. A8 (NVFP4 layers) at any width.
void moe_experts(const Activations& x, const Routing& routing, const Bank& bank, const Workspace& workspace,
                 std::uint16_t* outputs, cudaStream_t stream);

// A decode call of one column (128 experts) in one launch: moe_route's routing (workspace.routed and gamma for the
// column; not its dispatch) and moe_experts' outputs, bit for bit, as one persistent grid of 144 CTAs (one an SM; the
// router's rows on the first 16, every CTA then selecting the column itself, a job's 144 gate/up and down row groups on
// 18 CTAs of 8 warps, each warp streaming both through one ring). x: the router's BF16 row; rows: the column's
// quantized row. Needs moe_single_supported(bank.format): the device holds the grid resident (otherwise moe_route and
// moe_experts). A8 (NVFP4 layers): rows is the column's A8 row; the separate kernels' A8 bits.
[[nodiscard]] bool moe_single_supported(ExpertFormat format, ExpertActivation activation = ExpertActivation::Checkpoint);

// The expert prefetch (a decode visit's guessed experts into L2 ahead of its experts' launch; models issue it on a
// side stream). The bytes of an expert record's gate/up part (its first part: the gate/up row groups, then down's).
[[nodiscard]] constexpr std::uint64_t gate_up_record_bytes(ExpertFormat format, std::int32_t hidden,
                                                           std::int32_t intermediate) noexcept {
    const bool mx = format == ExpertFormat::Mxfp8;
    return std::uint64_t(2 * intermediate / 16) * std::uint64_t(hidden / (mx ? 32 : 16)) * std::uint64_t(mx ? 528 : 144);
}
// The first `bytes` (a multiple of 16, at most a record) of the records of the distinct experts among experts[0, count)
// (count 1-16; ids outside the bank skipped; read with relaxed loads) into L2, as cp.async.bulk.prefetch.L2 in 16 KB
// pieces from a grid of 12 CTAs (few SM slots beside the call's own kernels). A hint: no output depends on it.
void moe_prefetch_l2(const std::int32_t* experts, std::int32_t count, const Bank& bank, std::uint64_t bytes,
                     cudaStream_t stream);
void moe_single(const std::uint16_t* x, const Activations& rows, const std::uint16_t* router, const float* bias,
                float route_scale, const Bank& bank, const Workspace& workspace, std::uint16_t* outputs,
                cudaStream_t stream);

// After a call that set kErrorDownStalled: a CTA that gave up returned before it counted itself, so the counters a
// call's last CTA resets (route_done, single) may hold that call's counts, and the next call would pass its waits on
// them. Zeroes both on `stream` (the dispatch rebuilds every other word a call reads). The caller runs it once the
// stalled call has finished (after its synchronization) and before any later call.
void moe_clear_counters(const Workspace& workspace, cudaStream_t stream);

namespace testing {
// What a moe_single call whose job 0 stalled at its last row group leaves in the workspace's counters (the router done,
// every CTA but job 0's finished, the other jobs complete), written on `stream` and waited for: the Engine's forced
// stall pairs it with the error word, so a test sees the stall's real aftermath.
void leave_stalled_single_counters(const Workspace& workspace, cudaStream_t stream);
} // namespace testing

// The prompt route (a prefill call's columns, any width). Its arithmetic differs from the narrow route's in the
// router's logits and the experts' sums, and each column is still computed from its own inputs alone, so a prompt
// column's outputs never depend on the call's width or other columns.
//
// moe_select: each column's selection (moe_route's keys, ranks and gamma) from logits FP32 [T][E] computed by the
// caller, into the workspace's routing, then the dispatch.
void moe_select(const float* logits, std::int32_t columns, const float* bias, std::int32_t experts, float route_scale,
                const Workspace& workspace, cudaStream_t stream);

// moe_experts_prompt: NVFP4 layers on block-scaled tensor cores (mma mxf4nvf4, a grouped GEMM over the dispatched
// entries): a gate/up or down output is bf16(acc * rs) with acc the FP32 tensor-core sum of the decoded block
// products in its own fixed K order (the exact sum up to the accumulation's rounding) and rs the narrow route's row
// scale; h and its four-over-six rows as the narrow route forms them. MXFP8 layers: a
// grouped GEMM on block-scaled FP8 tensor cores (mma mxf8f6f4, UE8M0 per 32), each output the FP32 sum over K's
// 32-blocks in ascending order, g, u = bf16(acc), h's MXFP8 groups as the narrow route's, y = bf16(acc). A8 (NVFP4
// layers): the A8 order above on int8 tensor cores (ops/offloaded_sparse_moe/cuda/wide_expert_a8.cuh's grouped GEMM),
// so its outputs are moe_experts' A8 bits. A16 takes moe_experts. Outputs as moe_experts'.
void moe_experts_prompt(const Activations& x, const Routing& routing, const Bank& bank, const Workspace& workspace,
                        std::uint16_t* outputs, cudaStream_t stream);

} // namespace infernix::ops::w4a4_moe
