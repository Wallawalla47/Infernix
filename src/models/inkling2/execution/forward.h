#pragma once

// Inkling 2's forward over one decode or verify call (design docs/maintainer/inkling2_mini-design.md §8): embed, the
// 48 visits (layers 0-5, the core 6-17 three times, 18-23), the output mix and the head. Steppable a visit at a time,
// with every call buffer visible, so a test can inject a reference's inputs between visits; the Program captures the
// same steps into its graphs.
//
// Block sources (Block Attention Residuals, blocks of 6 layers): e (the embedding, block 0), h6 (layer 6's input, the
// prelude's output, the same on every pass), h12[p] (layer 12's input on pass p) and h18[p] (layer 17's output on pass
// p: pass p + 1's message source, and on the last pass layer 18's input). A pass restart (layer 6 on passes 2-3) takes
// h6 as its input while its entry kernel computes the previous visit's output into h18[p - 1].

#include "core/arena.h"
#include "models/inkling2/execution/parameters.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

namespace infernix::models::inkling2 {

inline constexpr std::int32_t kMaxForwardColumns = ops::w4a4_moe::kMaxColumns;

// One lane's state image (design §7.2): each visit's four conv rings ([kRingPositions][width] BF16: K and V inputs
// 1024 wide, the post-attention and post-MLP inputs 2304) and each sliding visit's window K (BF16) and V (FP16)
// ([kKvHeads][kWindowRingPages][64][128] each). Offsets are 16-bit words from the image's start.
struct VisitState {
    std::int64_t key_ring = 0, value_ring = 0, attention_ring = 0, mlp_ring = 0;
    std::int64_t window_k = -1, window_v = -1; // sliding visits
    std::int32_t full_slot = -1;               // full visits: their K/V pool, in visit order
};
struct StateLayout {
    std::vector<VisitState> visits;
    std::int64_t lane_words  = 0;
    std::int32_t full_visits = 0;
};
[[nodiscard]] StateLayout state_layout(const TextConfig& config);

// The state a call runs against: the lanes' images and the full visits' paged K/V, every full visit reading the
// lanes' block table rows (they append the same positions).
struct StateView {
    std::uint16_t* images   = nullptr; // [lanes][lane_words]
    std::int64_t lane_words = 0;
    std::vector<std::uint16_t*> pool_k, pool_v; // per full slot: [kKvHeads][pool_pages][64][128]
    std::int64_t pool_pages     = 0;
    const std::int32_t* table   = nullptr; // [lanes][table_stride]
    std::int32_t table_stride   = 0;
};

// One call: each column's token, lane and position (a lane's columns consecutive, positions ascending by 1; lane -1
// pads), and each lane's first position: where its sequence begins (the first key it attends to), not its first column
// in this call. A lane may have up to kMaxForwardColumns columns: every Op gives a column the same bits in any call,
// so a wide call (prefill) equals single-column calls of the same positions.
struct CallPlan {
    std::vector<std::int32_t> tokens, lane, position;
    std::vector<std::int32_t> first;
    [[nodiscard]] std::int32_t columns() const noexcept { return static_cast<std::int32_t>(tokens.size()); }
    [[nodiscard]] std::int32_t lanes() const noexcept { return static_cast<std::int32_t>(first.size()); }
};

// The call's device buffers ([columns][width] rows unless stated).
struct ForwardBuffers {
    std::int32_t *tokens = nullptr, *lane = nullptr, *position = nullptr, *first = nullptr;
    std::int32_t* groups = nullptr; // [kMaxAttentionGroups][3]: attention's column groups
    // A wide call's state copies: each lane's conv ring histories ([lanes][history_words], the image's 192 rings at
    // kHistorySlots rows each) and its sliding visits' staging ring ([lanes][kKvHeads][kStagingRingPages][64][128] K
    // and V), with the {ring, history, width} descriptors conv_history reads.
    std::uint16_t* history = nullptr;
    std::uint16_t *staging_k = nullptr, *staging_v = nullptr;
    std::int64_t* history_rings = nullptr;
    std::uint16_t *e = nullptr, *h6 = nullptr, *hp = nullptr, *h12[3] = {}, *h18[3] = {};
    float* cand = nullptr;
    std::uint16_t* x[3] = {};
    std::uint16_t *q = nullptr, *r = nullptr, *gate = nullptr, *k = nullptr, *v = nullptr, *sink = nullptr;
    std::uint16_t *key_taps = nullptr, *value_taps = nullptr;
    std::uint16_t *attention = nullptr, *o = nullptr, *xm = nullptr;
    std::uint8_t* rows = nullptr;
    float* amax = nullptr;
    std::uint16_t* experts = nullptr; // [columns * kTopK][H]
    float *mixed = nullptr, *message = nullptr, *fg = nullptr; // [3][columns][H], [3][columns][H], [3][columns][2H]
    std::uint16_t *gate_input = nullptr, *z = nullptr;          // [3][columns][2H], [3][columns][rank]
    std::uint16_t* final_hidden = nullptr;
    float* logits = nullptr;          // [columns][V]
    ops::w4a4_moe::Workspace moe;
    ops::inkling2::AttentionWorkspace attention_ws;
};

// What a visit reads besides the state and the previous visit's MLP tail: its input, where its entry writes the
// previous visit's output, both mixes' completed blocks and (loop passes 2-3) the message sources. inputs() gives the
// forward's own wiring; a test may substitute a reference's rows.
struct VisitInputs {
    const std::uint16_t* input = nullptr;
    std::uint16_t* tail_out    = nullptr;
    ops::inkling2::Blocks attention_blocks, mlp_blocks;
    const std::uint16_t* message_sources[2] = {};
};

// The output mix's sources and where its entry writes the last visit's output.
struct FinalInputs {
    std::uint16_t* tail_out = nullptr;
    ops::inkling2::Blocks blocks;
};

class Forward {
public:
    Forward(const Parameters& parameters, std::int32_t max_lanes);

    // Uploads the call's tokens and column map (synchronous copies) and fixes the call's width.
    void prepare(const CallPlan& call);
    // The same through `staging` (pinned, kUploadWords words, unused by any copy still in flight) on `stream`.
    void upload(const CallPlan& call, std::int32_t* staging, cudaStream_t stream);
    static constexpr std::size_t kUploadWords =
        3 * kMaxForwardColumns + ops::inkling2::kMaxAttentionLanes + 3 * ops::inkling2::kMaxAttentionGroups;
    void embed(cudaStream_t stream);
    void visit(std::uint32_t visit, const StateView& state, const VisitInputs& in, cudaStream_t stream);
    void visit(std::uint32_t v, const StateView& state, cudaStream_t stream) { visit(v, state, inputs(v), stream); }
    // The output mix and the head over columns [head_first, head_first + head_columns) (-1: through the last column;
    // 0: no head, as a prefill chunk that samples nothing), their logits in rows 0.. of buffers().logits.
    void finish(const StateView& state, const FinalInputs& in, cudaStream_t stream, std::int32_t head_first = 0,
                std::int32_t head_columns = -1);
    void finish(const StateView& state, cudaStream_t stream) { finish(state, final_inputs(), stream); }

    [[nodiscard]] const ForwardBuffers& buffers() const noexcept { return b_; }
    [[nodiscard]] const StateLayout& layout() const noexcept { return layout_; }
    [[nodiscard]] std::int32_t columns() const noexcept { return columns_; }
    [[nodiscard]] VisitInputs inputs(std::uint32_t visit) const;
    [[nodiscard]] FinalInputs final_inputs() const;

private:
    // The block sources of a mix at a layer and pass (include_current: the MLP mix of a boundary layer, which counts
    // the layer's own input).
    [[nodiscard]] ops::inkling2::Blocks blocks(std::uint32_t layer, std::uint32_t pass, bool include_current) const;
    [[nodiscard]] std::uint16_t* tail_out(std::uint32_t visit) const;
    // Validates the call, fixes its width and lanes, and returns attention's groups ([groups][3]).
    std::vector<std::int32_t> layout_call(const CallPlan& call);
    // Columns [from, from + columns) (-1: through the last) of x through w.
    void project(const std::uint16_t* x, const Weight& w, void* out, bool fp32, cudaStream_t stream,
                 std::int32_t from = 0, std::int32_t columns = -1) const;
    // A visit's conv ring (kind: 0 K input, 1 V input, 2 W_o output, 3 MLP output), with its history on a wide call.
    [[nodiscard]] ops::inkling2::Ring ring(const StateView& s, std::uint32_t visit, int kind) const;
    [[nodiscard]] ops::inkling2::ColumnMap column_map() const;
    // A wide call's state copies: the conv ring histories (before visit 0) and a sliding visit's staging ring.
    void snapshot_history(const StateView& s, cudaStream_t stream) const;
    void stage_window(const StateView& s, const VisitState& vs, bool in, cudaStream_t stream) const;

    // A lane's columns in the call.
    struct CallLane {
        std::int32_t lane = 0, count = 0, position = 0, first = 0; // position: of its first column
    };

    const Parameters& p_;
    const TextConfig& c_;
    StateLayout layout_;
    std::int64_t history_words_ = 0; // a lane's histories
    std::int32_t max_lanes_ = 0;
    std::int32_t columns_   = 0;
    std::int32_t lanes_     = 0;
    std::int32_t groups_    = 0;
    std::int32_t max_group_columns_ = 0;
    std::vector<CallLane> call_lanes_;
    bool wide_conv_   = false; // a lane has more columns than its conv rings can take without a history
    bool wide_window_ = false; // a lane appends more positions than its window ring can take: staging rings
    DeviceBuffer memory_;
    ForwardBuffers b_;
};

} // namespace infernix::models::inkling2
