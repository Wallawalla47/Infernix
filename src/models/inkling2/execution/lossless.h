#pragma once

// Inkling 2's --lossless route (docs/maintainer/lossless-route.md §2b, §3a): every operation of
// modeling_inkling2_loop.py correctly rounded (the routed experts in their declared quantized arithmetic): the exact
// projections of lossless_linear over each weight's exact value, the block attention residuals and the loop messages,
// the L2 clips and log scales, the static and dynamic convolutions, eager attention with the relative bias and the
// sinks, the router, the experts and the softcapped FP32 logits.
//
// Two forms. The window form (causal scoring) runs one sequence's window from zero state over its own keys and writes
// nothing to the KV pool or the lanes' state. The cache form (generation) runs a call continuing one lane's sequence:
// its convolutions continue from the lane's conv rings and its attention reads the keys before the call from the
// lane's window rings (sliding visits) and pages (full visits), and the call leaves its own inputs and keys there, in
// the reproducible route's layouts (forward.h StateLayout, inkling2_attention.h) with the lossless values bit for bit
// (V as its BF16 value, not the bf16 profile's FP16). Every operation is the window form's on the same values, so a
// position's logits do not depend on how its sequence was cut into calls.

#include "core/arena.h"
#include "infernix/ops/lossless_inkling2.h"
#include "models/inkling2/execution/forward.h"
#include "models/inkling2/execution/parameters.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace infernix::models::inkling2 {

// A cache-form call's lane: its state image (layout's), the full visits' K and V pools (per full slot, page-major
// [pages][kKvHeads][64][128]) and its block-table row (device), and the call's first position. The state holds the
// positions before `first` as this route computed them (zero conv history at a fresh start).
struct LosslessLaneState {
    std::uint16_t* image      = nullptr;
    const StateLayout* layout = nullptr;
    std::span<std::uint16_t* const> pool_k, pool_v;
    const std::int32_t* table = nullptr;
    std::int32_t first        = 0;
};

class LosslessForward {
public:
    // Rows of logits one head call gives.
    static constexpr std::int32_t kHeadRows = 64;

    // Calls of up to `max_columns` tokens. `max_keys`: the cache form's context (the keys a call's attention may
    // read; 0: the window form only, its keys the window's). `reference`: the design reference (lossless-route.md §3b)
    // with the bf16 KV profile, V stored as FP16 (lossless_round_fp16; the window form only), otherwise the lossless
    // route.
    LosslessForward(const Parameters& parameters, std::int32_t max_columns, bool reference = false,
                    std::int32_t max_keys = 0);
    [[nodiscard]] static std::size_t device_bytes(const TextConfig& config, std::int32_t max_columns,
                                                  std::int32_t max_keys = 0);

    // The window `tokens` (positions 0 ..) through the final norm; the final rows stay for head().
    void run(std::span<const std::int32_t> tokens, cudaStream_t stream);
    // The cache form: `tokens` at positions lane.first .. continuing the lane's sequence, its state updated.
    void run(std::span<const std::int32_t> tokens, const LosslessLaneState& lane, cudaStream_t stream);
    // Logits FP32 [count][V] (softcapped) of final rows [first, first + count), count <= kHeadRows, into logits().
    void head(std::int32_t first, std::int32_t count, cudaStream_t stream);
    [[nodiscard]] const float* logits() const noexcept { return logits_; }

private:
    // A convolution's input over the call (BF16 [C, history + T]): in the cache form its first `history` columns the
    // lane's inputs before the call (conv_kernel - 1 of them), `call` the call's columns.
    struct ConvInput {
        Tensor all, call;
    };
    // Conv rings (forward.h VisitState): 0 the K input, 1 the V input, 2 W_o's output, 3 the MoE's output.
    enum RingKind : int { kKeyRing = 0, kValueRing = 1, kAttentionRing = 2, kMlpRing = 3 };

    void run_call(std::span<const std::int32_t> tokens, const LosslessLaneState* lane, cudaStream_t stream);
    // One layer visit over the call: h (BF16 [H, T]) in, out (BF16 [H, T]) its output.
    void visit(std::uint32_t visit, const Tensor& h, std::span<const Tensor* const> attention_blocks,
               std::span<const Tensor* const> mlp_blocks, const Tensor* messages, Tensor& out, cudaStream_t stream);
    // A message gate on FP32 x (in place) with message m.
    void message_gate(const GateOperands& g, Tensor& x, const Tensor& m, cudaStream_t stream);
    // The attention sublayer from its normed inputs: into c the post-attention conv's output (BF16 [H, history + T],
    // the call's columns from `history`).
    void attention(std::uint32_t visit, const LayerOperands& l, const Tensor& xq, const Tensor& xk, const Tensor& xv,
                   Tensor& c, cudaStream_t stream);
    // A convolution input of C rows from the arena (its history columns loaded in the cache form).
    [[nodiscard]] ConvInput conv_input(std::int32_t C, std::int32_t T);
    void load_history(const ConvInput& in, std::uint32_t visit, RingKind kind, cudaStream_t stream) const;
    void store_history(const ConvInput& in, std::uint32_t visit, RingKind kind, cudaStream_t stream) const;
    [[nodiscard]] std::uint16_t* ring(std::uint32_t visit, RingKind kind) const;
    // A visit's stored K and V in the lane's state: a sliding visit's window rings, a full visit's pages.
    [[nodiscard]] ops::LosslessInkling2Keys stored_keys(std::uint32_t visit) const;
    // The routed experts from the normed BF16 x: out (BF16 [H, T]).
    void experts(const LayerOperands& l, std::uint32_t layer, const Tensor& x, Tensor& out, cudaStream_t stream);
    void project(const Tensor& x, const Weight& w, std::int32_t row_begin, std::int32_t rows, Tensor& out,
                 cudaStream_t stream);

    const Parameters& p_;
    const TextConfig& c_;
    std::int32_t capacity_ = 0;
    std::int32_t keys_     = 0; // the key planes' capacity (a multiple of 64)
    std::int32_t T_        = 0;
    bool reference_        = false;
    const LosslessLaneState* lane_ = nullptr; // the cache form's lane during run()
    std::int32_t history_          = 0;       // the conv history columns of the call (cache form: conv_kernel - 1)
    bool identity_positions_       = true;    // positions_ holds 0 .. capacity - 1
    DeviceBuffer memory_;
    std::unique_ptr<WorkspaceArena> arena_; // over memory_ after the window tensors
    // The window's tensors.
    std::int32_t* tokens_    = nullptr;
    std::int32_t* positions_ = nullptr;
    std::int32_t* kv_planes_ = nullptr; // {0, .., kv_heads - 1}: the KV heads as tiles of one product
    std::uint16_t* blocks_   = nullptr; // e, h6, h12[3], h18[3] (BF16 [H, capacity] each)
    std::uint16_t* hidden_[2] = {};     // the visits' ping-pong rows
    std::uint16_t* final_    = nullptr; // the final norm's rows
    float* logits_           = nullptr; // [kHeadRows][V]
    // The experts' host tables: each expert's FP32 amax (gate/up, down) and the bank's denominators.
    std::vector<std::vector<float>> amax_;
    std::vector<std::vector<float>> denominators_;
};

} // namespace infernix::models::inkling2
