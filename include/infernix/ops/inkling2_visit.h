#pragma once

// Inkling 2's visit chain outside the projections, attention and experts (docs/maintainer/inkling2_mini-design.md §8,
// §12) on the decode and verify route: one CTA a column, H = 2304, every reduction in the pinned order
// (src/ops/inkling2/visit_math.h). Rows are BF16 or FP32 [T][H], column t contiguous.
//
//   embed        e = bf16(w * rms(E[token]))
//   visit_entry  (K1, K1a) the previous visit's MLP tail: its 8 expert outputs folded in slot order in FP32 and
//                rounded once, the post-MLP conv over the ring, out = bf16(cand + m); then the attention mix over the
//                completed blocks and the visit's input (1 query, or 3 on full layers; layer 0 takes its input
//                unmixed), and either the input norms (x_q, x_k, x_v) or, on loop passes 2-3, the cross-pass message
//                mixes and the gates' inputs bf16([rms(x); rms(m)]) for gate_exit
//   mlp_entry    (K4's epilogue, K5, K5a) the post-attention conv of W_o's output over its ring, cand = h + a (a alone
//                at a block boundary, FP32), the MLP mix over the completed blocks and cand, then the MLP norm and the
//                experts' activation rows (four-over-six NVFP4 with the row amax, or MXFP8), or the mlp message gate's
//                inputs
//   gate_exit    (K1b, K5b) x' = (1 - tanh f) x + tanh(g) m per kind (f, g the gate's forget and gain projections),
//                then the kind's norm (and for the mlp kind the activation rows)
//   final_mix    the last visit's tail, the output mix and the final norm: the head's input
//   softcap      the head's logits capped in place
//
// A short conv's history is keyed by absolute position: column (lane, p) reads positions p - 1 .. p - 3 from the
// call's earlier columns of its lane (recomputed from their inputs), else from its lane's ring (or, when the call
// passes one, the ring's history snapshot), and 0 below the lane's first position; a column among its lane's last
// kRingPositions in the call writes its own raw input to the ring at slot p % kRingPositions, so the ring ends the call
// holding the lane's last kRingPositions positions. A lane of more than kMaxLaneColumns columns would overwrite a slot
// its first columns read: such a call passes each ring's history, taken by conv_history before the call's first
// kernel. A column of lane -1 is padding: its history is 0 and it writes no ring.

#include "infernix/ops/w4a4_sparse_moe.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace infernix::ops::inkling2 {

inline constexpr std::int32_t kHidden         = 2304; // the geometry the kernels are built for
inline constexpr std::int32_t kHeadDim        = 128;
inline constexpr std::int32_t kRelDim         = 16;
inline constexpr std::int32_t kRingPositions  = 32;
inline constexpr std::int32_t kMaxLaneColumns = kRingPositions - 3;
inline constexpr std::int32_t kMaxSources     = 5; // a mix's sources: <= 4 completed blocks and the partial

// Each column's lane and absolute position, and each lane's first position (the conv history below it is 0).
struct ColumnMap {
    const std::int32_t* lane     = nullptr; // [T]; a lane's columns are consecutive, positions ascending by 1
    const std::int32_t* position = nullptr; // [T]
    const std::int32_t* first    = nullptr; // [lanes]
    std::int32_t columns         = 0;       // T
};

// History rows a snapshot keeps: positions p - 1 .. p - 3 of a lane's first column p, at slot p % kHistorySlots.
inline constexpr std::int32_t kHistorySlots = 4;

// One visit's conv ring: per lane [kRingPositions][width] BF16 at base + lane * lane_stride (elements, at least
// kRingPositions * width: a lane's state may hold other parts between), position p at slot p % kRingPositions. A call
// with a lane of more than kMaxLaneColumns columns passes `history`: per lane [kHistorySlots][width] at history +
// lane * history_lane_stride, which reads of positions before the lane's first column take instead of the ring.
struct Ring {
    std::uint16_t* base             = nullptr;
    std::int32_t width              = 0;
    std::int64_t lane_stride        = 0;
    const std::uint16_t* history    = nullptr;
    std::int64_t history_lane_stride = 0;
};

// Copies rows of a lane's rings to their history snapshot (a ring's rows p - 1 .. p - 3 at or above `first`, p the
// lane's first column in the coming call): `rings` device descriptors {ring offset, history offset, width} in 16-bit
// words from the lane's image and history bases.
struct ConvHistoryArgs {
    const std::int64_t* rings     = nullptr; // [count][3]
    std::int32_t count            = 0;
    const std::uint16_t* image    = nullptr; // the lane's state image
    std::uint16_t* history        = nullptr; // the lane's history snapshot
    std::int32_t position         = 0;       // the lane's first column's position
    std::int32_t first            = 0;       // the lane's first position
};
void conv_history(const ConvHistoryArgs& args, cudaStream_t stream);

// A mix's completed blocks, BF16 [T][H] each, in block order (the partial source is the kernel's own).
struct Blocks {
    const std::uint16_t* rows[kMaxSources - 1] = {};
    std::int32_t count = 0;
};

// The previous visit's MLP tail: its routed expert outputs [T * kTopK][H] (entry t * kTopK + j), its cand (FP32
// [T][H]), its post-MLP conv taps [4][H] (lag-major, row 0 the current position) and its ring of combined outputs.
struct MlpTail {
    const std::uint16_t* experts = nullptr;
    const float* cand            = nullptr;
    const std::uint16_t* conv    = nullptr;
    Ring ring;
};

void embed(const std::int32_t* tokens, std::int32_t columns, const std::uint16_t* embedding, const std::uint16_t* norm,
           std::uint16_t* out, cudaStream_t stream);

struct EntryArgs {
    std::int32_t columns = 0;
    ColumnMap map;
    MlpTail tail;                         // tail.experts null: no tail (layer 0)
    std::uint16_t* tail_out    = nullptr; // with a tail: its output bf16(cand + m) [T][H]
    const std::uint16_t* input = nullptr; // the visit's input [T][H]; null: tail_out (a pass restart passes h6)
    Blocks blocks;                        // the attention mix's completed blocks
    const float* queries = nullptr;       // FP32 [nq][H]; null: layer 0, whose x is its input
    std::int32_t nq      = 1;             // 1 (sliding) or 3 (full: q, k, v)
    // Without messages: the norms (sliding: norms[0]; full: q, k, v) and x_q, x_k, x_v [T][H] (sliding writes its
    // one x to each non-null x[i]).
    const std::uint16_t* norms[3] = {};
    std::uint16_t* x[3]           = {};
    // --a8-activations: the x outputs' A8 rows too (K2's 8-bit inputs: ops/ggml/ggml_a8.h's codec of the BF16 x, a
    // column's 2,304 codes then 72 FP32 scales, x8_stride words apart), each non-null x8[i] beside x[i].
    std::int32_t* x8[3]      = {};
    std::int32_t x8_stride   = 0;
    // Loop passes 2-3: the message queries [4][H] (q, k, v, mlp; the first three used here) and the previous pass's
    // two message sources; the kernel then writes mixed (FP32 [nq][T][H]), message (FP32 [3][T][H]) and gate_input
    // (BF16 [3][T][2H]) instead of x.
    const float* message_queries            = nullptr;
    const std::uint16_t* message_sources[2] = {};
    float* mixed              = nullptr;
    float* message            = nullptr;
    std::uint16_t* gate_input = nullptr;
};
void visit_entry(const EntryArgs& args, cudaStream_t stream);

// The experts' activation rows of the MLP input: [T][row_bytes(format, H)], and for NVFP4 each row's amax [T]. With
// emax (an NVFP4 layer under --a16-activations): rows are A16 limb rows [T][w4a4_moe::kA16XRowBytes] and emax [T]
// their exponents (canon::a16_column_exponent of the BF16 row). rows null: none (an MXFP8 layer under A16 reads x).
struct ExpertRows {
    w4a4_moe::ExpertFormat format = w4a4_moe::ExpertFormat::Nvfp4FourOverSix;
    std::uint8_t* rows            = nullptr;
    float* amax                   = nullptr;
    std::int32_t* emax            = nullptr;
    // --a8-activations (NVFP4 layers): rows as A8 rows [T][w4a4_moe::kA8XRowBytes] (ops/ggml/ggml_a8.h's codec of the
    // BF16 MLP input, moe_encode_a8's bits).
    bool a8                       = false;
};

struct MlpEntryArgs {
    std::int32_t columns = 0;
    ColumnMap map;
    const std::uint16_t* o    = nullptr; // W_o's output [T][H]
    const std::uint16_t* conv = nullptr; // the post-attention conv taps [4][H]
    Ring ring;                           // its ring of W_o outputs
    const std::uint16_t* h    = nullptr; // the visit's input
    bool boundary             = false;   // the layer opens a block: cand = a
    float* cand               = nullptr; // FP32 [T][H]
    Blocks blocks;                       // the MLP mix's completed blocks (at a boundary they include h)
    const float* query        = nullptr; // FP32 [H]
    // Without messages: the MLP norm, its BF16 output x [T][H] (the router's input) and the experts' rows.
    const std::uint16_t* norm = nullptr;
    std::uint16_t* x          = nullptr;
    ExpertRows rows;
    // Loop passes 2-3: the mlp message query [H] and the previous pass's two sources; the kernel then writes mixed,
    // message (FP32 [T][H]) and gate_input (BF16 [T][2H]) instead.
    const float* message_query              = nullptr;
    const std::uint16_t* message_sources[2] = {};
    float* mixed              = nullptr;
    float* message            = nullptr;
    std::uint16_t* gate_input = nullptr;
};
void mlp_entry(const MlpEntryArgs& args, cudaStream_t stream);

struct GateExitArgs {
    std::int32_t columns = 0;
    std::int32_t kinds   = 1;          // 3 (attention: q, k, v) or 1 (mlp)
    const float* mixed   = nullptr;    // FP32 [kinds][T][H], or [1][T][H] shared by the kinds (shared_x)
    bool shared_x        = false;
    const float* message = nullptr;    // FP32 [kinds][T][H]
    const float* fg      = nullptr;    // FP32 [kinds][T][2H]: the forget projection, then the gain projection
    const std::uint16_t* norms[3] = {}; // per kind (a sliding layer passes its one norm three times)
    std::uint16_t* x[3]           = {}; // per kind [T][H]
    // --a8-activations: per kind x's A8 rows too (K2's 8-bit inputs, EntryArgs::x8's layout; null: none).
    std::int32_t* x8[3]    = {};
    std::int32_t x8_stride = 0;
    ExpertRows rows;                    // the mlp kind: the experts' rows of x[0] (rows.rows null: none)
};
void gate_exit(const GateExitArgs& args, cudaStream_t stream);

struct FinalArgs {
    std::int32_t columns = 0;
    ColumnMap map;
    MlpTail tail;                     // the last visit's
    std::uint16_t* tail_out = nullptr; // its output [T][H]
    Blocks blocks;
    const float* query        = nullptr; // the output query [H]
    const std::uint16_t* norm = nullptr; // the final norm
    std::uint16_t* out        = nullptr; // the head's input [T][H]
};
void final_mix(const FinalArgs& args, cudaStream_t stream);

// logits[i] = softcap_c(logits[i], cap) over `count` FP32 values, in place.
void softcap(float* logits, std::int64_t count, float cap, cudaStream_t stream);

} // namespace infernix::ops::inkling2
