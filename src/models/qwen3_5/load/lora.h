#pragma once

#include "artifact/peft_lora.h"
#include "models/qwen3_5/config.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace infernix::models::qwen3_5 {

// The PEFT modules (indices into PeftLoraAdapter::modules()) an adapter targets in one text layer.
struct LoraLayer {
    // Gated full attention: q_proj (its rows are each head's query then output gate), k_proj, v_proj, o_proj.
    std::optional<std::size_t> query, key, value, output;
    // The dense MLP: gate_proj, up_proj, down_proj.
    std::optional<std::size_t> gate, up, down;

    [[nodiscard]] bool attention_input() const noexcept { return query || key || value; }
    [[nodiscard]] bool mlp_input() const noexcept { return gate || up; }
    [[nodiscard]] bool any() const noexcept { return attention_input() || output || mlp_input() || down; }
};

// A LoRA adapter (`--lora`) bound to a Qwen3.5 text configuration: every module mapped to its layer and projection
// and checked against the configuration's shapes. Refuses modules this route does not adapt (the linear-attention
// projections, MoE experts, vision, MTP, embeddings and heads), naming them.
struct LoraBinding {
    std::shared_ptr<const artifact::PeftLoraAdapter> adapter;
    std::vector<LoraLayer> layers; // one per text layer
    // Every module's rank rounded up to a multiple of 16 (ops::LoraAddend::rank): the largest.
    std::uint32_t padded_rank = 0;

    [[nodiscard]] const artifact::PeftLoraModule& module(std::size_t index) const {
        return adapter->modules().at(index);
    }
    // A one-line description for the startup log.
    [[nodiscard]] std::string describe() const;
};

[[nodiscard]] LoraBinding bind_lora(const std::filesystem::path& path, const TextConfig& text);

} // namespace infernix::models::qwen3_5
