#include "models/qwen3_5/load/lora.h"

#include "artifact/schema.h"
#include "infernix/ops/lora.h"

#include <algorithm>
#include <cstdio>
#include <regex>
#include <string>

namespace infernix::models::qwen3_5 {
namespace {

std::uint32_t round_up16(std::uint32_t value) { return (value + 15U) / 16U * 16U; }

[[noreturn]] void refuse(const artifact::PeftLoraAdapter& adapter, const std::string& what) {
    throw artifact::ArtifactError("--lora " + adapter.weights_path().string() + ": " + what);
}

} // namespace

LoraBinding bind_lora(const std::filesystem::path& path, const TextConfig& text) {
    auto adapter = std::make_shared<const artifact::PeftLoraAdapter>(path);
    LoraBinding binding;
    binding.layers.resize(text.num_hidden_layers);
    const auto* dense = std::get_if<DenseConfig>(&text.ffn);
    // PEFT's module paths for a Qwen3.5 checkpoint: the multimodal model's "model.language_model.layers.N", or a
    // text-only export's "model.layers.N".
    static const std::regex kModule(R"(^model\.(language_model\.)?layers\.(\d+)\.(self_attn|mlp|linear_attn)\.(.+)$)");
    std::uint32_t max_rank = 0;
    for (std::size_t index = 0; index < adapter->modules().size(); ++index) {
        const artifact::PeftLoraModule& m = adapter->modules()[index];
        std::smatch match;
        if (!std::regex_match(m.module, match, kModule)) {
            refuse(*adapter, "module '" + m.module +
                                 "' is not a text layer's attention or MLP projection (vision, MTP, embeddings and the "
                                 "head are not adapted)");
        }
        const std::uint64_t layer = std::stoull(match[2].str());
        const std::string block = match[3].str(), leaf = match[4].str();
        if (layer >= text.num_hidden_layers) {
            refuse(*adapter, "module '" + m.module + "' names layer " + std::to_string(layer) + " of a " +
                                 std::to_string(text.num_hidden_layers) + "-layer model");
        }
        LoraLayer& target = binding.layers[layer];
        const auto expect = [&](std::uint64_t in, std::uint64_t out) {
            if (m.in != in || m.out != out) {
                refuse(*adapter, "module '" + m.module + "' maps [" + std::to_string(m.in) + "] to [" +
                                     std::to_string(m.out) + "]; this model's projection maps [" + std::to_string(in) +
                                     "] to [" + std::to_string(out) + "] (an adapter for another base model?)");
            }
        };
        const auto place = [&](std::optional<std::size_t>& slot) {
            if (slot) { refuse(*adapter, "module '" + m.module + "' appears twice"); }
            slot = index;
        };
        if (block == "linear_attn") {
            refuse(*adapter, "module '" + m.module + "' is a linear-attention (Gated DeltaNet) projection, which "
                                                     "this route does not adapt");
        }
        if (block == "self_attn") {
            if (text.layer_types.at(layer) != MixerKind::FullAttention || !text.attention) {
                refuse(*adapter, "module '" + m.module + "' targets attention in a linear-attention layer");
            }
            const AttentionConfig& a = *text.attention;
            if (leaf == "q_proj") {
                expect(text.hidden_size, 2 * a.query_width()); // each head's query rows, then its output gate's
                place(target.query);
            } else if (leaf == "k_proj") {
                expect(text.hidden_size, a.key_width());
                place(target.key);
            } else if (leaf == "v_proj") {
                expect(text.hidden_size, a.key_width());
                place(target.value);
            } else if (leaf == "o_proj") {
                expect(a.query_width(), text.hidden_size);
                place(target.output);
            } else {
                refuse(*adapter, "module '" + m.module + "' is not q_proj, k_proj, v_proj or o_proj");
            }
        } else {
            if (dense == nullptr) {
                refuse(*adapter, "module '" + m.module + "' targets an MLP, but this model's are mixtures of experts, "
                                                         "which this route does not adapt");
            }
            if (leaf == "gate_proj") {
                expect(text.hidden_size, dense->intermediate_size);
                place(target.gate);
            } else if (leaf == "up_proj") {
                expect(text.hidden_size, dense->intermediate_size);
                place(target.up);
            } else if (leaf == "down_proj") {
                expect(dense->intermediate_size, text.hidden_size);
                place(target.down);
            } else {
                refuse(*adapter, "module '" + m.module + "' is not gate_proj, up_proj or down_proj");
            }
        }
        max_rank = std::max(max_rank, m.rank);
    }
    binding.padded_rank = round_up16(max_rank);
    if (binding.padded_rank > static_cast<std::uint32_t>(ops::kLoraMaxRank)) {
        refuse(*adapter, "rank " + std::to_string(max_rank) + " exceeds " + std::to_string(ops::kLoraMaxRank));
    }
    // The shrink reads its input in groups of 1024 values.
    constexpr std::uint64_t kGroup = static_cast<std::uint64_t>(ops::kLoraShrinkChunk) * ops::kLoraShrinkGroup;
    const auto shrinkable = [&](std::uint64_t k, const char* what) {
        if (k % kGroup != 0) {
            refuse(*adapter, std::string(what) + " width " + std::to_string(k) + " is not a multiple of 1024");
        }
    };
    shrinkable(text.hidden_size, "the hidden");
    if (text.attention) { shrinkable(text.attention->query_width(), "the attention output"); }
    if (dense != nullptr) { shrinkable(dense->intermediate_size, "the MLP intermediate"); }
    binding.adapter = std::move(adapter);
    return binding;
}

std::string LoraBinding::describe() const {
    std::size_t attention = 0, mlp = 0, layers_adapted = 0;
    for (const LoraLayer& layer : layers) {
        attention += (layer.query ? 1 : 0) + (layer.key ? 1 : 0) + (layer.value ? 1 : 0) + (layer.output ? 1 : 0);
        mlp += (layer.gate ? 1 : 0) + (layer.up ? 1 : 0) + (layer.down ? 1 : 0);
        layers_adapted += layer.any() ? 1 : 0;
    }
    std::uint32_t min_rank = 0xFFFFFFFFU, max_rank = 0;
    float min_scale = 0, max_scale = 0;
    bool first = true;
    for (const auto& m : adapter->modules()) {
        min_rank  = std::min(min_rank, m.rank);
        max_rank  = std::max(max_rank, m.rank);
        min_scale = first ? m.scale : std::min(min_scale, m.scale);
        max_scale = first ? m.scale : std::max(max_scale, m.scale);
        first     = false;
    }
    char text[512];
    std::snprintf(text, sizeof(text),
                  "%zu attention and %zu MLP projections in %zu layers, rank %u%s, scale %g%s, %s factors, base %s",
                  attention, mlp, layers_adapted, max_rank,
                  min_rank != max_rank ? (" (from " + std::to_string(min_rank) + ")").c_str() : "",
                  static_cast<double>(max_scale), min_scale != max_scale ? " (varies)" : "",
                  adapter->dtype().c_str(), adapter->base_model().empty() ? "unnamed" : adapter->base_model().c_str());
    return text;
}

} // namespace infernix::models::qwen3_5
