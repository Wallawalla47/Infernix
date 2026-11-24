#pragma once

// A LoRA adapter (`--lora`, docs/lora.md) bound to an Inkling 2 model: Model data, read at load planning. Maps every
// module of a Hugging Face PEFT adapter to its layer and projection and checks its shapes against the configuration.

#include "artifact/peft_lora.h"
#include "models/inkling2/config.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace infernix::models::inkling2 {

// One message gate's modules (core layers): down_proj [2H -> rank], forget_gate_proj and message_gate_proj [rank -> H].
struct LoraGate {
    std::optional<std::size_t> down, forget, message;
    [[nodiscard]] bool any() const noexcept { return down || forget || message; }
};

// The PEFT modules (indices into PeftLoraAdapter::modules()) an adapter targets in one text layer.
struct LoraLayer {
    // K2's: q_proj.linear, relative_position_bias.linear, o_proj.gate_proj, sink_proj (x_q), k_proj.linear and
    // k_proj.sconv.weight_proj (x_k), v_proj.linear and v_proj.sconv.weight_proj (x_v). The two weight_proj exist on
    // full-attention layers only.
    std::optional<std::size_t> query, relative, gate, sink, key, value, key_taps, value_taps;
    std::optional<std::size_t> output;  // o_proj.linear
    std::array<LoraGate, 4> gates;      // q, k, v, mlp (core layers)

    [[nodiscard]] bool attention_input() const noexcept {
        return query || relative || gate || sink || key || value || key_taps || value_taps;
    }
    [[nodiscard]] bool any_gate() const noexcept {
        return gates[0].any() || gates[1].any() || gates[2].any() || gates[3].any();
    }
    [[nodiscard]] bool any() const noexcept { return attention_input() || output || any_gate(); }
};

// One module's factors on the fast routes: byte offsets of A and B in the factor buffer and its group's padded rank.
struct LoraFastModule {
    std::uint64_t a = 0, b = 0;
    std::uint32_t rank = 0;
};
struct LoraFastLayout {
    std::vector<LoraFastModule> modules; // by module index
    std::uint64_t bytes = 0;
};

struct LoraBinding {
    std::shared_ptr<const artifact::PeftLoraAdapter> adapter;
    std::vector<LoraLayer> layers; // one per text layer
    // Every module's rank rounded up to a multiple of 16 (ops::LoraAddend's k-step), the largest: on the lossless route
    // every module runs at this rank, its padded rows of A and columns of B zero. On the fast routes each term group
    // runs at its own (fast_layout).
    std::uint32_t padded_rank = 0;

    [[nodiscard]] const artifact::PeftLoraModule& module(std::size_t index) const {
        return adapter->modules().at(index);
    }
    // The fast routes' factor layout: a group (K2's outputs, W_o, a loop pass's gate downs, their forget and message
    // projections; each one launch of the term) runs at the largest padded rank among its modules, and K2's A rows are
    // one stack in input order (x_q's, x_k's, x_v's), so one projection segment computes them.
    [[nodiscard]] LoraFastLayout fast_layout() const;
    // The startup log's description.
    [[nodiscard]] std::string describe() const;
    // The device bytes of the factors (prepare_lora's layout): the route's BF16 A and B, or on the lossless route A's
    // three exact planes and B in FP32. A plan counts this before anything is allocated.
    [[nodiscard]] std::uint64_t device_bytes(bool lossless) const;
};

// One factor region: 256-byte aligned, so every operand keeps the alignment the kernels read it with.
[[nodiscard]] constexpr std::uint64_t lora_region_bytes(std::uint64_t bytes) noexcept { return (bytes + 255) / 256 * 256; }
// B's rows on the BF16 route: the output rows rounded up to 16 (the tensor core's m), the padded rows zero.
[[nodiscard]] constexpr std::uint64_t lora_padded_rows(std::uint64_t rows) noexcept { return (rows + 15) / 16 * 16; }

// Reads `path` against the text configuration. Refuses, naming the module and why: anything but the projections above
// (the routed experts and the router are parameters, which a plain PEFT adapter cannot target; embeddings, the head,
// vision and the drafter are not adapted), a shape that is not this model's, and a rank above ops::kLoraMaxRank.
[[nodiscard]] LoraBinding bind_lora(const std::filesystem::path& path, const TextConfig& text);

} // namespace infernix::models::inkling2
