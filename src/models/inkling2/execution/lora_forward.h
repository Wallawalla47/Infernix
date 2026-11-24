#pragma once

// --lora on the fast routes (docs/lora.md, design docs/maintainer/inkling2_mini-design.md §11.1): after a Forward's
// projections have written their outputs, the adapter's term for each group: U = bf16(A x), then
// ops::inkling2::lora_term_add over the group's outputs. Owns U's workspace.
//
// U comes out of the base projection launch where that gives the bits of a launch of its own (the fold_* calls, which
// append A's rows as segments before the base launch): BF16 storage (Q8 cannot share a launch with BF16), a segment
// count within kMaxProjectionSegments, and the same arithmetic for A's rows. A prompt call's MMA does not depend on the
// launch's other segments; a decode call's splits K by the launch's total rows (dmma_splits), which W_o and the gate
// downs keep beside their A rows and K2 does not (4 runs for its 538 row tiles, 8 for A's alone), so K2's decode shrink
// stays a launch of its own. The gates' forget and message projections are FP32 launches, and U is BF16: separate too.

#include "core/arena.h"
#include "infernix/ops/inkling2_projections.h"
#include "models/inkling2/execution/lora_parameters.h"

#include <cuda_runtime.h>

#include <array>
#include <cstdint>

namespace infernix::models::inkling2 {

class LoraForward {
public:
    // U's workspace: K2's stacked modules (at most 8), or a smaller group's slots, at the adapter's largest rank.
    static constexpr std::int32_t kSlots = 8;

    LoraForward(const LoraParameters& parameters, const TextConfig& config, std::int32_t capacity);
    [[nodiscard]] static std::size_t device_bytes(std::int32_t rank, std::int32_t capacity);

    // K2's outputs, written by the projections at their [T][rows] buffers (query, relative, gate, sink, key, value and
    // on full layers the two tap projections), from the three BF16 inputs. `key_taps`/`value_taps` may be null.
    struct AttentionOutputs {
        std::uint16_t *query = nullptr, *relative = nullptr, *gate = nullptr, *sink = nullptr;
        std::uint16_t *key = nullptr, *value = nullptr, *key_taps = nullptr, *value_taps = nullptr;
    };

    // Each fold_* appends the group's A rows to the base launch about to run when that keeps every bit, and says so;
    // the matching call below then skips its own shrink.
    [[nodiscard]] bool fold_attention_inputs(std::uint32_t layer, ops::inkling2::ProjectionArgs& k2, bool prompt) const;
    [[nodiscard]] bool fold_output(std::uint32_t layer, ops::inkling2::ProjectionArgs& wo) const;
    [[nodiscard]] bool fold_gate_down(std::uint32_t layer, std::int32_t first_kind, std::int32_t kinds,
                                      ops::inkling2::ProjectionArgs& down) const;

    void attention_inputs(std::uint32_t layer, std::int32_t columns, const std::uint16_t* xq, const std::uint16_t* xk,
                          const std::uint16_t* xv, const AttentionOutputs& out, bool prompt, bool folded,
                          cudaStream_t stream) const;
    // W_o's output (BF16 [T][H]) from its input (BF16 [T][Q]).
    void attention_output(std::uint32_t layer, std::int32_t columns, const std::uint16_t* attention, std::uint16_t* out,
                          bool prompt, bool folded, cudaStream_t stream) const;
    // The message gates' down projections of `kinds` gates from first_kind (0-2 attention's q, k, v, or 3 the mlp's):
    // gate i of the call reads gate_input + i T 2H and writes z + i T rank (rank = the gates' rank).
    void gate_down(std::uint32_t layer, std::int32_t first_kind, std::int32_t kinds, std::int32_t columns,
                   const std::uint16_t* gate_input, std::uint16_t* z, bool prompt, bool folded, cudaStream_t stream) const;
    // Their forget and message projections: gate i reads z + i T rank and writes the forget rows of fg + i T 2H (its
    // first H rows a column) and the message rows after them (FP32).
    void gate_outputs(std::uint32_t layer, std::int32_t first_kind, std::int32_t kinds, std::int32_t columns,
                      const std::uint16_t* z, float* fg, bool prompt, cudaStream_t stream) const;

private:
    // K2's A rows by input: x_q's, x_k's, x_v's, one contiguous stack (LoraBinding::fast_layout).
    struct AttentionStack {
        std::int32_t rank = 0;
        std::array<const std::uint16_t*, 3> a{};
        std::array<std::int32_t, 3> rows{};
        [[nodiscard]] std::int32_t total() const noexcept { return rows[0] + rows[1] + rows[2]; }
    };
    [[nodiscard]] static AttentionStack attention_stack(const LoraLayerTerms<LoraTerm>& l);
    // Where a stack's U goes: one [T][total] block when K2's three inputs are one vector (one segment), otherwise one
    // [T][rows] block per input. Appends the segments.
    void append_stack(const AttentionStack& s, bool aliased, ops::inkling2::ProjectionArgs& args) const;
    [[nodiscard]] std::uint16_t* slot(std::int32_t index) const noexcept {
        return static_cast<std::uint16_t*>(u_.p) + static_cast<std::size_t>(index) * capacity_ * rank_;
    }
    [[nodiscard]] std::uint16_t* stack_rows(std::int32_t rows_before) const noexcept {
        return static_cast<std::uint16_t*>(u_.p) + static_cast<std::size_t>(rows_before) * capacity_;
    }

    const LoraParameters& p_;
    std::int32_t hidden_, query_, gate_rank_, capacity_, rank_;
    DeviceBuffer u_; // [kSlots][capacity][rank] BF16
};

} // namespace infernix::models::inkling2
