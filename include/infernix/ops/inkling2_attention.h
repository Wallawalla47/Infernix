#pragma once

// Inkling 2's attention on the decode and verify route (docs/maintainer/inkling2_mini-design.md §9): 24 query heads
// over 8 KV heads of 128, a learned relative bias over 1024 distances, a per-head sink, an output gate, sliding
// (window 1024) or full visits. Two launches:
//
//   prologue  per (column, KV head): the K and V short convs over their input rings (static taps; on full visits plus
//             the column's per-head dynamic taps, (sum + x) / 3), then k = bf16(clip(bf16(conv k), 22.627)) and
//             v = bf16(rms(bf16(conv v))) appended to the visit's KV at the column's position (K as BF16, V as
//             FP16_RNE: the cache's represented values); q = bf16(clip(bf16(q_raw) * s_q, 22.627)) and
//             r = bf16(clip(r_raw * s_r, 8)), each times tau(position) and rounded again on full visits
//   attention per (lane, KV head, split): every query row (a column's 3 heads) against the represented keys of fixed
//             absolute 64-key units, the last CTA of a (lane, KV head) combining each row's splits in split order
//             with the sink, then a = bf16(bf16(acc / l) * sigmoid(gate))
//
// Pinned arithmetic (the reproducible route's bits; .local NUMERICS "Decode attention"): for a row and a key j of a
// unit, dot = (c0 + c1) + (c2 + c3), c_i the fma fold of q.k over dims 32i .. 32i + 31 from +0.0;
// s_j = dot * fl32(1/sqrt 128) + bias_j, bias_j = bf16(the fma fold of r[i] P[i][d] over i in order) for distance
// d = p - j < 1024 (else 0); t_j = s_j * fl32(log2 e); per unit m_u = max over visible keys, p_j = ex2.approx(t_j - m_u),
// l_u = the warp tree (bfly32) of p_lane + p_{lane+32}, acc_u[d] = the fma fold of p_j v_j[d] over keys in order.
// A split folds its units in order (m = max, c = ex2(m_old - m), c_u = ex2(m_u - m), l = l_u c_u + l c,
// acc = acc_u c_u + acc c by fma); the combine takes m over the row's splits, l and acc as fma folds of the splits'
// l and acc times ex2(m_s - m) in split order, then l += ex2(sink * log2 e - m), a = bf16(acc / l).
//
// Units are the absolute KV pages [64u, 64u + 64). A sliding row's splits are its window's units (at most 17), a full
// row's are kFullSplits runs of ceil(units / kFullSplits) units of its own key count, so a row's arithmetic depends only
// on its position and lane: a column gives the same bits alone or in any call. Keys below the lane's first position are
// masked; a lane may have at most kMaxAttentionLaneColumns columns a call (and the conv rings' kMaxLaneColumns).

#include "infernix/ops/inkling2_visit.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace infernix::ops::inkling2 {

inline constexpr std::int32_t kHeads                  = 24;
inline constexpr std::int32_t kKvHeads                = 8;
inline constexpr std::int32_t kGroupHeads             = kHeads / kKvHeads;
inline constexpr std::int32_t kRelativeExtent         = 1024;
inline constexpr std::int32_t kWindow                 = 1024;
inline constexpr std::int32_t kKeyUnit                = 64;  // a unit of keys: one KV page
inline constexpr std::int32_t kWindowRingPages        = 17;  // a sliding visit's ring: 17 pages, 1088 positions
inline constexpr std::int32_t kSlidingSplits          = 18;  // the units a group's columns can touch
inline constexpr std::int32_t kFullSplits             = 32;
inline constexpr std::int32_t kMaxAttentionLaneColumns = 16; // a group's columns
inline constexpr std::int32_t kMaxAttentionColumns     = w4a4_moe::kMaxColumns;
inline constexpr std::int32_t kMaxAttentionLanes       = 8;
// A call's groups: each lane's columns in runs of at most kMaxAttentionLaneColumns.
inline constexpr std::int32_t kMaxAttentionGroups =
    kMaxAttentionLanes + kMaxAttentionColumns / kMaxAttentionLaneColumns;
// The pages of a staging ring that holds a lane's call of up to `columns` positions and the window before it: the
// first query's 1023 earlier keys and the call's own, whole pages.
[[nodiscard]] constexpr std::int32_t staging_ring_pages(std::int32_t columns) noexcept {
    return (kWindow - 1 + columns + 2 * (kKeyUnit - 1)) / kKeyUnit + 1;
}

// One visit's keys and values: K as BF16 words, V as FP16 words, 64 positions a page of [64][128] per KV head.
//   sliding (table null): each lane's ring [kKvHeads][ring_pages][64][128] at k (v) + lane * lane_stride (elements),
//            position p in page (p / 64) % ring_pages (the state's ring has kWindowRingPages; a call writing more
//            than kKeyUnit positions a lane runs over a staging ring of staging_ring_pages(its width), window_copy filling it
//            from the state's ring before and returning its last positions after);
//   full: a head-major page pool [kKvHeads][pool_pages][64][128] and each lane's block table row
//            table[lane * table_stride + p / 64].
struct KvView {
    std::uint16_t* k             = nullptr;
    std::uint16_t* v             = nullptr;
    std::int64_t lane_stride     = 0;
    std::int32_t ring_pages      = kWindowRingPages;
    const std::int32_t* table    = nullptr;
    std::int32_t table_stride    = 0;
    std::int64_t pool_pages      = 0;
};

// Copies positions [from, to) of a lane's sliding K and V from one ring view to another (by position: the views'
// page counts may differ).
struct WindowCopyArgs {
    KvView src, dst;
    std::int32_t src_lane = 0, dst_lane = 0;
    std::int32_t from = 0, to = 0;
};
void window_copy(const WindowCopyArgs& args, cudaStream_t stream);

// K2's outputs for the call's columns (BF16 [T][width]).
struct AttentionInputs {
    const std::uint16_t* q          = nullptr; // [T][24 * 128]
    const std::uint16_t* r          = nullptr; // [T][24 * 16]
    const std::uint16_t* k          = nullptr; // [T][8 * 128]
    const std::uint16_t* v          = nullptr; // [T][8 * 128]
    const std::uint16_t* sink       = nullptr; // [T][24]
    const std::uint16_t* gate       = nullptr; // [T][24 * 128]
    const std::uint16_t* key_taps   = nullptr; // full: [T][8 * 4], head-major, lag-minor
    const std::uint16_t* value_taps = nullptr; // full
};

// A layer's attention parameters (BF16).
struct AttentionLayer {
    bool full                           = false;
    const std::uint16_t* query_scale    = nullptr; // [24][128]
    const std::uint16_t* relative_scale = nullptr; // [24][16]
    const std::uint16_t* relative_table = nullptr; // [16][1024]
    const std::uint16_t* key_conv       = nullptr; // [4][1024], lag-major (row 0 the current position)
    const std::uint16_t* value_conv     = nullptr;
};

// Call scratch for calls of up to `capacity` columns: the final q and r, the splits' statistics of one launch of
// attention rows (a call's groups run in launches spanning at most kMaxAttentionColumns columns each) and the
// per-(group, KV head) counters (each last CTA resets its own). Zero the memory once before the first call.
struct AttentionWorkspace {
    std::int32_t capacity    = 0;
    std::uint16_t* q         = nullptr; // [capacity][24][128]
    std::uint16_t* r         = nullptr; // [capacity][24][16]
    float* stats             = nullptr; // [kMaxAttentionColumns][24][kFullSplits][132]: m, l, -, -, acc[128]
    std::uint32_t* counters  = nullptr; // [kMaxAttentionGroups][kKvHeads]
};
[[nodiscard]] std::size_t attention_workspace_bytes(std::int32_t capacity = kMaxAttentionColumns);
[[nodiscard]] AttentionWorkspace carve_attention_workspace(void* base, std::int32_t capacity = kMaxAttentionColumns);

// A group is a run of one lane's columns: its rows' arithmetic is the same in any group (it depends only on each row's
// position and lane), so a lane's columns may split into groups anywhere.
struct AttentionArgs {
    std::int32_t columns = 0;               // at most the workspace's capacity
    std::int32_t groups  = 0;               // the call's groups; 0: all padding
    std::int32_t max_group_columns = 0;     // the most columns any group has (sizes the CTA)
    ColumnMap map;
    const std::int32_t* groups_columns = nullptr; // [groups][3]: the group's first column, column count and lane
    // The same groups in host memory, which the call's launches of rows split by: needed when the groups span more
    // than kMaxAttentionColumns columns or number more than kMaxAttentionGroups.
    const std::int32_t* host_groups = nullptr;
    std::int32_t stats_base = 0;            // set by attention() for each launch: its first column
    AttentionInputs in;
    AttentionLayer layer;
    Ring key_ring, value_ring;              // the K and V conv input rings (width 1024)
    KvView kv;
    AttentionWorkspace ws;
    std::uint16_t* out = nullptr;           // a [T][24 * 128] (padding columns: zero)
    // attention(): the prologue inside the attention launch (a call without padding columns whose every lane is one
    // group: a lane's second group would read keys its first group's CTAs store): each CTA computes its
    // rows' q and r, and the CTA(s) reading a unit the call's new keys and values in it (stored as the prologue
    // stores them), the same bits (the first split's CTAs also write ws.q and ws.r). attention_prompt ignores it.
    bool fused = false;
};
void attention(const AttentionArgs& args, cudaStream_t stream);

// The prompt route (a prefill or scoring call: one lane's consecutive columns, any width; its groups on the host): the
// same prologue, then a row's scores, softmax and P x V on tensor cores over its fixed absolute 64-key units in
// ascending order (attention_prompt.cu): S = q k^T by BF16 MMA (FP32, from zero over the head dimension), the decode
// route's bias, s and t; per unit m_u over visible keys (a unit with none leaves the row unchanged), m = max, c =
// ex2(m_old - m), p = ex2(t - m) rounded to FP16 for P x V by FP16 MMA (FP32), l = l c + the row sum of p (the four
// lanes' partial sums in xor order), acc = acc c + P V; then l += ex2(sink log2 e - m) and a = bf16(bf16(acc / l) *
// sigmoid(gate)). Other bits than attention()'s; a column's bits depend only on its position and lane.
void attention_prompt(const AttentionArgs& args, cudaStream_t stream);

} // namespace infernix::ops::inkling2
