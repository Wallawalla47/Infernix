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
#include "models/inkling2/execution/lora_forward.h"
#include "models/inkling2/execution/parameters.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <functional>
#include <memory>
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
    std::vector<std::uint16_t*> pool_k, pool_v; // per full slot: [pool_pages][kKvHeads][64][128] (codec's elements)
    std::int64_t pool_pages     = 0;
    const std::int32_t* table   = nullptr; // [lanes][table_stride]
    std::int32_t table_stride   = 0;
    // The full pools' storage (--kv-dtype; inkling2_attention.h KvCodec) and, quantized, their FP16 scale planes.
    ops::inkling2::KvCodec codec = ops::inkling2::KvCodec::Bf16;
    std::vector<std::uint8_t*> side_k, side_v; // per full slot: [pool_pages][kKvHeads][64][the form's side bytes]
    std::uint32_t* unpackable = nullptr;       // bf16p: [lanes] flags of rows with no packed form
};

// One call: each column's token, lane and position (a lane's columns consecutive, positions ascending by 1; lane -1
// pads), and each lane's first position: where its sequence begins (the first key it attends to), not its first column
// in this call.
//
// Two routes, each giving a column the same bits in any call of its kind (every Op computes a column from its own
// inputs alone): a prompt call (`prompt`: prefill and scoring, every column a prompt token) runs the tensor-core route
// (the projections, router and experts on MMA; design §14), at any width, so any chunking of a prompt gives the same
// bits; any other call (decode, verification, forced tokens) runs the decode route, so a verify column equals the
// decode column. A prompt column need not equal the same position computed by the decode route.
struct CallPlan {
    std::vector<std::int32_t> tokens, lane, position;
    std::vector<std::int32_t> first;
    bool prompt = false;
    [[nodiscard]] std::int32_t columns() const noexcept { return static_cast<std::int32_t>(tokens.size()); }
    [[nodiscard]] std::int32_t lanes() const noexcept { return static_cast<std::int32_t>(first.size()); }
};

// The call's device buffers ([columns][width] rows unless stated).
struct ForwardBuffers {
    std::int32_t *tokens = nullptr, *lane = nullptr, *position = nullptr, *first = nullptr;
    std::int32_t* groups = nullptr; // [kMaxAttentionGroups][3]: attention's column groups
    // A wide call's state copies: each lane's conv ring histories ([lanes][history_words], the image's 192 rings at
    // kHistorySlots rows each) and its sliding visits' staging ring ([lanes][kKvHeads][staging pages][64][128] K
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
    std::uint8_t* rows = nullptr;     // the experts' x rows (checkpoint rows, or A16 limb rows)
    float* amax = nullptr;
    std::int32_t* emax = nullptr;     // A16: the limb rows' exponents
    std::uint16_t* experts = nullptr; // [columns * kTopK][H]
    float *mixed = nullptr, *message = nullptr, *fg = nullptr; // [3][columns][H], [3][columns][H], [3][columns][2H]
    std::uint16_t *gate_input = nullptr, *z = nullptr;          // [3][columns][2H], [3][columns][rank]
    // a8: a call's 8-bit inputs (projections_mma's x8): K2's three ([3][columns][projection_a8_column_words(H)]
    // words), then W_o's and the head's over the same memory; null without a8.
    std::int32_t* x8 = nullptr;
    std::uint16_t* final_hidden = nullptr;
    // DSpark's context features (--spec dspark): each column's partial block sums entering the drafter's target visits
    // (5, 23, 41) and its final-normed state, BF16 [columns][4][H]; null without the drafter.
    std::uint16_t* features = nullptr;
    float* logits = nullptr;          // [columns][V]
    ops::w4a4_moe::Workspace moe;
    ops::inkling2::AttentionWorkspace attention_ws;
    // The expert prefetch (Forward::kGuessColumns): each visit's MLP input of a guessing call [visits][2][H], each
    // visit's guessed experts [visits][16], the experts each visit's last column took in the last verification call
    // [visits][kTopK], and the guesses' router workspace (2 columns).
    std::uint16_t* guess_x       = nullptr;
    std::int32_t* guess_ids      = nullptr;
    std::int32_t* expert_record  = nullptr;
    ops::w4a4_moe::Workspace guess;
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
    // A forward of `max_lanes` lanes and calls of up to `capacity` columns (prefill's chunk; at least
    // kMaxForwardColumns, the decode and verify calls' width, whose logits rows the head takes).
    // `activation`: the routed experts' (--a16-activations: ExpertActivation::A16; A8 with `a8`). `a8`
    // (--a8-activations, LOSSY): the attention input and output projections (K2, W_o) and the head, whose 8-bit
    // weights it needs, take 8-bit activations (projections_mma's x8) in every call, prompt or decode; the message
    // gates and the router keep 16-bit ones (as Basalt runs them).
    Forward(const Parameters& parameters, std::int32_t max_lanes, std::int32_t capacity = kMaxForwardColumns,
            ops::w4a4_moe::ExpertActivation activation = ops::w4a4_moe::ExpertActivation::Checkpoint, bool a8 = false);

    // Uploads the call's tokens and column map (synchronous copies) and fixes the call's width.
    void prepare(const CallPlan& call);
    // The same through `staging` (pinned, upload_words() words, unused by any copy still in flight) on `stream`.
    void upload(const CallPlan& call, std::int32_t* staging, cudaStream_t stream);
    [[nodiscard]] std::size_t upload_words() const noexcept {
        return 3 * static_cast<std::size_t>(capacity_) + ops::inkling2::kMaxAttentionLanes + 3 * static_cast<std::size_t>(max_groups_);
    }
    // The columns a call may have (decode and verify calls: at most kMaxForwardColumns).
    [[nodiscard]] std::int32_t capacity() const noexcept { return capacity_; }
    // The device memory a forward of these lanes and capacity allocates.
    [[nodiscard]] static std::size_t device_bytes(const TextConfig& config, std::int32_t max_lanes, std::int32_t capacity,
                                                  bool features = false,
                                                  ops::w4a4_moe::ExpertActivation activation =
                                                      ops::w4a4_moe::ExpertActivation::Checkpoint,
                                                  bool a8 = false);
    void embed(cudaStream_t stream);
    void visit(std::uint32_t visit, const StateView& state, const VisitInputs& in, cudaStream_t stream);
    void visit(std::uint32_t v, const StateView& state, cudaStream_t stream) { visit(v, state, inputs(v), stream); }
    // The output mix and the head over columns [head_first, head_first + head_columns) (-1: through the last column;
    // 0: no head, as a prefill chunk that samples nothing), their logits in rows 0.. of buffers().logits.
    void finish(const StateView& state, const FinalInputs& in, cudaStream_t stream, std::int32_t head_first = 0,
                std::int32_t head_columns = -1);
    void finish(const StateView& state, cudaStream_t stream) { finish(state, final_inputs(), stream); }
    // The head again over columns [first, first + count) (at most kMaxForwardColumns) of the last finish's final
    // hidden rows, into logits rows 0..: a wide call's logits a slice at a time.
    void head(std::int32_t first, std::int32_t count, cudaStream_t stream);
    // The head's raw products W_head x (FP32, no softcap) of `count` BF16 rows x [count][H] (at most
    // kMaxForwardColumns) into out [count][V]: the DSpark drafter's base logits. With x8 (count 8-bit input rows of
    // ops::inkling2::projection_a8_column_words(H) words) a Q8 head runs on 8-bit inputs: tensor cores at the block's
    // rows, other bits than head()'s (drafts only: no output depends on them). `head`: a shortlist's Q8 head instead
    // of the target's (--lm-head-draft: out is [count][its rows]).
    void project_head(const std::uint16_t* x, std::int32_t count, float* out, cudaStream_t stream,
                      std::int32_t* x8 = nullptr, const Weight* head = nullptr) const;
    // Profiling: `marker` is called with each Op's name right after its launches (null: none), so a caller can record
    // events between them.
    void set_marker(std::function<void(const char*)> marker) { marker_ = std::move(marker); }
    // ScoreOptions::prompt_hidden_states: while set, the calls copy `column` of each layer's output on its last visit
    // (the prelude's and coda's only visit, the core's third pass) to rows[layer][H] (BF16), and hidden_counts()[layer]
    // counts the row written.
    void set_hidden_capture(std::uint16_t* rows, std::int32_t column);
    void clear_hidden_capture() noexcept { hidden_rows_ = nullptr; }
    [[nodiscard]] const std::vector<std::uint8_t>& hidden_counts() const noexcept { return hidden_counts_; }

    [[nodiscard]] const ForwardBuffers& buffers() const noexcept { return b_; }
    // Points the MoE workspaces' error word (a down CTA's lost count sets kErrorDownStalled there) at `word`, a
    // mapped pinned word the host reads after each round's own synchronization (Program::finish_sample). Called
    // before any graph captures a call.
    void set_moe_error_word(std::uint32_t* word) noexcept {
        b_.moe.error   = word;
        b_.guess.error = word;
    }
    [[nodiscard]] const StateLayout& layout() const noexcept { return layout_; }
    [[nodiscard]] std::int32_t columns() const noexcept { return columns_; }
    [[nodiscard]] VisitInputs inputs(std::uint32_t visit) const;
    // A visit's MLP input rows in the last call ([columns][H] BF16): buffers().xm, or a guessing call's visit slot
    // (buffers().guess_x, kept for the expert prefetch).
    [[nodiscard]] const std::uint16_t* mlp_input(std::uint32_t visit) const;
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
    // DSpark's context features: `rows` (a visit's input, or the final state) into the feature slot of its tap, if any.
    void tap(std::uint32_t visit, const std::uint16_t* rows, cudaStream_t stream) const;
    // set_hidden_capture's copy of the visit's output (`rows`: where its successor's entry wrote it).
    void capture_hidden(std::uint32_t visit, const std::uint16_t* rows, cudaStream_t stream);

    // A lane's columns in the call.
    struct CallLane {
        std::int32_t lane = 0, count = 0, position = 0, first = 0; // position: of its first column
    };

    const Parameters& p_;
    const TextConfig& c_;
    StateLayout layout_;
    std::int64_t history_words_ = 0; // a lane's histories
    std::int32_t max_lanes_ = 0;
    std::int32_t capacity_  = 0;
    ops::w4a4_moe::ExpertActivation activation_ = ops::w4a4_moe::ExpertActivation::Checkpoint;
    bool a8_                 = false;
    std::int32_t max_groups_ = 0;
    std::int32_t staging_pages_ = 0;
    std::int32_t columns_   = 0;
    std::int32_t lanes_     = 0;
    std::int32_t groups_    = 0;
    std::int32_t max_group_columns_ = 0;
    std::vector<CallLane> call_lanes_;
    std::vector<std::int32_t> host_groups_; // the call's groups, as uploaded
    std::function<void(const char*)> marker_;
    void mark(const char* op) const {
        if (marker_) { marker_(op); }
    }
    bool wide_conv_   = false; // a lane has more columns than its conv rings can take without a history
    bool wide_window_ = false; // a lane appends more positions than its window ring can take: staging rings
    bool prompt_      = false; // the call's route (CallPlan::prompt)
    bool padding_     = false; // the call has padding columns (lane -1)
    bool one_group_lanes_ = true; // every lane's columns are one attention group (at most 16)
    DeviceBuffer memory_;
    ForwardBuffers b_;
    std::unique_ptr<LoraForward> lora_; // --lora on this route (the lossless route's terms are LosslessForward's)
    std::uint16_t* hidden_rows_ = nullptr;
    std::int32_t hidden_column_ = 0;
    std::vector<std::uint8_t> hidden_counts_;

public:
    // The expert prefetch (design §10.5): a decode call of at most kGuessColumns columns runs the next visit's router on
    // this visit's MLP input (kept a visit each: ForwardBuffers::guess_x) on one side stream, and after each visit's K2
    // prefetches that guess into L2 on another: at one column its kGuessExperts1 likeliest experts' whole records, at
    // two each column's kGuessExperts / 2 likeliest experts' gate/up bytes. A call of kGuessColumns + 1 to
    // kReadAheadColumns columns (verification) prefetches instead the gate/up bytes of the kReadAheadExperts likeliest
    // experts its visit's last column took in the previous such call (the router's record). The side streams join the
    // call's stream once, at the end of finish(). No output depends on it (only what L2 holds).
    static constexpr std::int32_t kGuessColumns     = 2;
    static constexpr std::int32_t kGuessExperts1    = 4;
    static constexpr std::int32_t kGuessExperts     = 8;
    static constexpr std::int32_t kReadAheadColumns = 8;
    static constexpr std::int32_t kReadAheadExperts = 4;

private:
    // The prefetch's two side streams and the events that fork and join them (created with the forward).
    struct SideStreams {
        cudaStream_t guess = nullptr, prefetch = nullptr;
        cudaEvent_t x = nullptr, k2 = nullptr, guessed = nullptr, join_guess = nullptr, join_prefetch = nullptr;
        bool forked_guess = false, forked_prefetch = false; // the current call's
        SideStreams();
        SideStreams(const SideStreams&)            = delete;
        SideStreams& operator=(const SideStreams&) = delete;
        ~SideStreams();
        void release() noexcept;
    };
    std::unique_ptr<SideStreams> side_;
};

} // namespace infernix::models::inkling2
