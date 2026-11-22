#pragma once

// Inkling 2's --lossless route (docs/maintainer/lossless-route.md §2b, §3a): a scored window of one sequence from zero
// state in one pass, every operation of modeling_inkling2_loop.py correctly rounded (the routed experts in their
// declared quantized arithmetic): the exact projections of lossless_linear over each weight's exact value, the block
// attention residuals and the loop messages, the L2 clips and log scales, the static and dynamic convolutions, eager
// attention with the relative bias and the sinks, the router, the experts and the softcapped FP32 logits. It writes
// nothing to the KV pool or the lanes' state.

#include "core/arena.h"
#include "models/inkling2/execution/parameters.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace infernix::models::inkling2 {

class LosslessForward {
public:
    // Rows of logits one head call gives.
    static constexpr std::int32_t kHeadRows = 64;

    // Windows of up to `max_columns` tokens. `reference`: the design reference (lossless-route.md §3b) with the bf16
    // KV profile, V stored as FP16 (lossless_round_fp16), otherwise the lossless route.
    LosslessForward(const Parameters& parameters, std::int32_t max_columns, bool reference = false);
    [[nodiscard]] static std::size_t device_bytes(const TextConfig& config, std::int32_t max_columns);

    // The window `tokens` (positions 0 ..) through the final norm; the final rows stay for head().
    void run(std::span<const std::int32_t> tokens, cudaStream_t stream);
    // Logits FP32 [count][V] (softcapped) of final rows [first, first + count), count <= kHeadRows, into logits().
    void head(std::int32_t first, std::int32_t count, cudaStream_t stream);
    [[nodiscard]] const float* logits() const noexcept { return logits_; }

private:
    struct Visit;
    // One layer visit over the window: h (BF16 [H, T]) in, out (BF16 [H, T]) its output.
    void visit(std::uint32_t visit, const Tensor& h, std::span<const Tensor* const> attention_blocks,
               std::span<const Tensor* const> mlp_blocks, const Tensor* messages, Tensor& out, cudaStream_t stream);
    // A message gate on FP32 x (in place) with message m.
    void message_gate(const GateOperands& g, Tensor& x, const Tensor& m, cudaStream_t stream);
    // The attention sublayer from its normed inputs: c = the post-attention conv's output (BF16 [H, T]).
    void attention(const LayerOperands& l, const Tensor& xq, const Tensor& xk, const Tensor& xv, Tensor& c,
                   cudaStream_t stream);
    // The routed experts from the normed BF16 x: out (BF16 [H, T]).
    void experts(const LayerOperands& l, std::uint32_t layer, const Tensor& x, Tensor& out, cudaStream_t stream);
    void project(const Tensor& x, const Weight& w, std::int32_t row_begin, std::int32_t rows, Tensor& out,
                 cudaStream_t stream);

    const Parameters& p_;
    const TextConfig& c_;
    std::int32_t capacity_ = 0;
    std::int32_t T_        = 0;
    bool reference_        = false;
    DeviceBuffer memory_;
    std::unique_ptr<WorkspaceArena> arena_; // over memory_ after the window tensors
    // The window's tensors.
    std::int32_t* tokens_    = nullptr;
    std::int32_t* positions_ = nullptr;
    std::uint16_t* blocks_   = nullptr; // e, h6, h12[3], h18[3] (BF16 [H, capacity] each)
    std::uint16_t* hidden_[2] = {};     // the visits' ping-pong rows
    std::uint16_t* final_    = nullptr; // the final norm's rows
    float* logits_           = nullptr; // [kHeadRows][V]
    // The experts' host tables: each expert's FP32 amax (gate/up, down) and the bank's denominators.
    std::vector<std::vector<float>> amax_;
    std::vector<std::vector<float>> denominators_;
};

} // namespace infernix::models::inkling2
