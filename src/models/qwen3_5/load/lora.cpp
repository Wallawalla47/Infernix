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

// " (it names <base>)" when the adapter names its base model.
std::string base_named(const artifact::PeftLoraAdapter& adapter) {
    return adapter.base_model().empty() ? std::string{} : " (it names " + adapter.base_model() + ")";
}

// A refusal names the adapter, what it adapts that this model or Infernix cannot, and why.
[[noreturn]] void refuse(const artifact::PeftLoraAdapter& adapter, const std::string& what) {
    throw artifact::ArtifactError("LoRA adapter " + adapter.weights_path().parent_path().string() + " is refused: " +
                                  what);
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
    std::string max_rank_module;
    for (std::size_t index = 0; index < adapter->modules().size(); ++index) {
        const artifact::PeftLoraModule& m = adapter->modules()[index];
        std::smatch match;
        if (!std::regex_match(m.module, match, kModule)) {
            refuse(*adapter, "it adapts module '" + m.module +
                                 "', which is not a text layer's attention, linear-attention or MLP projection. Infernix "
                                 "adapts the text layers' q_proj, k_proj, v_proj, o_proj, linear_attn's in_proj_qkv, "
                                 "in_proj_z, in_proj_a, in_proj_b and out_proj, and gate_proj, up_proj and down_proj, not "
                                 "the vision tower, MTP, the embeddings or the head");
        }
        const std::uint64_t layer = std::stoull(match[2].str());
        const std::string block = match[3].str(), leaf = match[4].str();
        if (layer >= text.num_hidden_layers) {
            refuse(*adapter, "it adapts module '" + m.module + "' in layer " + std::to_string(layer) +
                                 ", but this model has " + std::to_string(text.num_hidden_layers) +
                                 " layers: the adapter was trained for another base model" + base_named(*adapter));
        }
        LoraLayer& target = binding.layers[layer];
        const auto expect = [&](std::uint64_t in, std::uint64_t out) {
            if (m.in != in || m.out != out) {
                refuse(*adapter, "its module '" + m.module + "' maps [" + std::to_string(m.in) + "] to [" +
                                     std::to_string(m.out) + "], but this model's projection maps [" + std::to_string(in) +
                                     "] to [" + std::to_string(out) + "]: the adapter was trained for another base model" +
                                     base_named(*adapter));
            }
        };
        const auto place = [&](std::optional<std::size_t>& slot) {
            if (slot) { refuse(*adapter, "it holds module '" + m.module + "' twice"); }
            slot = index;
        };
        if (block == "linear_attn") {
            if (text.layer_types.at(layer) != MixerKind::LinearAttention || !text.gdn) {
                refuse(*adapter, "it adapts linear-attention module '" + m.module +
                                     "', but that layer is a full-attention layer in this model: the adapter was trained "
                                     "for another base model" + base_named(*adapter));
            }
            const GdnConfig& g = *text.gdn;
            if (leaf == "in_proj_qkv") {
                expect(text.hidden_size, 2 * g.key_width() + g.value_width()); // query, key, then value channels
                place(target.gdn_qkv);
            } else if (leaf == "in_proj_z") {
                expect(text.hidden_size, g.value_width());
                place(target.gdn_z);
            } else if (leaf == "in_proj_a") {
                expect(text.hidden_size, g.linear_num_value_heads);
                place(target.gdn_a);
            } else if (leaf == "in_proj_b") {
                expect(text.hidden_size, g.linear_num_value_heads);
                place(target.gdn_b);
            } else if (leaf == "out_proj") {
                expect(g.value_width(), text.hidden_size);
                place(target.gdn_output);
            } else {
                refuse(*adapter, "it adapts linear-attention module '" + m.module +
                                     "'; Infernix adapts in_proj_qkv, in_proj_z, in_proj_a, in_proj_b and out_proj");
            }
        } else if (block == "self_attn") {
            if (text.layer_types.at(layer) != MixerKind::FullAttention || !text.attention) {
                refuse(*adapter, "it adapts attention module '" + m.module +
                                     "', but that layer is a linear-attention (Gated DeltaNet) layer in this model: the "
                                     "adapter was trained for another base model" + base_named(*adapter));
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
                refuse(*adapter, "it adapts attention module '" + m.module +
                                     "'; Infernix adapts q_proj, k_proj, v_proj and o_proj");
            }
        } else {
            if (dense == nullptr) {
                refuse(*adapter, "it adapts MLP module '" + m.module +
                                     "', but this model's MLPs are mixtures of experts. Infernix applies LoRA adapters "
                                     "to the dense models (the 27B) only");
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
                refuse(*adapter, "it adapts MLP module '" + m.module +
                                     "'; Infernix adapts gate_proj, up_proj and down_proj");
            }
        }
        if (m.rank > max_rank) {
            max_rank        = m.rank;
            max_rank_module = m.module;
        }
    }
    binding.padded_rank = round_up16(max_rank);
    if (binding.padded_rank > static_cast<std::uint32_t>(ops::kLoraMaxRank)) {
        refuse(*adapter, "its rank is " + std::to_string(max_rank) + " (module '" + max_rank_module +
                             "'), and Infernix currently supports LoRA adapters up to rank " +
                             std::to_string(ops::kLoraMaxRank));
    }
    // The shrink reads its input in groups of 1024 values.
    constexpr std::uint64_t kGroup = static_cast<std::uint64_t>(ops::kLoraShrinkChunk) * ops::kLoraShrinkGroup;
    const auto shrinkable = [&](std::uint64_t k, const char* what) {
        if (k % kGroup != 0) {
            refuse(*adapter, std::string("this model's ") + what + " width " + std::to_string(k) +
                                 " is not a multiple of 1024, which Infernix's LoRA kernels need");
        }
    };
    shrinkable(text.hidden_size, "the hidden");
    if (text.attention) { shrinkable(text.attention->query_width(), "the attention output"); }
    if (text.gdn) { shrinkable(text.gdn->value_width(), "the linear-attention output"); }
    if (dense != nullptr) { shrinkable(dense->intermediate_size, "the MLP intermediate"); }
    binding.adapter = std::move(adapter);
    return binding;
}

std::string LoraBinding::describe() const {
    std::size_t attention = 0, linear = 0, mlp = 0, layers_adapted = 0;
    for (const LoraLayer& layer : layers) {
        attention += (layer.query ? 1 : 0) + (layer.key ? 1 : 0) + (layer.value ? 1 : 0) + (layer.output ? 1 : 0);
        linear += (layer.gdn_qkv ? 1 : 0) + (layer.gdn_z ? 1 : 0) + (layer.gdn_a ? 1 : 0) + (layer.gdn_b ? 1 : 0) +
                  (layer.gdn_output ? 1 : 0);
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
    const std::string kinds = linear > 0 ? std::to_string(attention) + " attention, " + std::to_string(linear) +
                                               " linear-attention and " + std::to_string(mlp) + " MLP"
                                         : std::to_string(attention) + " attention and " + std::to_string(mlp) + " MLP";
    std::snprintf(text, sizeof(text),
                  "%s projections in %zu layers, rank %u%s, scale %g%s, %s factors, base %s", kinds.c_str(),
                  layers_adapted, max_rank,
                  min_rank != max_rank ? (" (from " + std::to_string(min_rank) + ")").c_str() : "",
                  static_cast<double>(max_scale), min_scale != max_scale ? " (varies)" : "",
                  adapter->dtype().c_str(), adapter->base_model().empty() ? "unnamed" : adapter->base_model().c_str());
    return text;
}

} // namespace infernix::models::qwen3_5
