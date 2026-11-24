#include "models/inkling2/lora.h"

#include "artifact/schema.h"
#include "infernix/ops/lora.h"

#include <algorithm>
#include <cstdio>
#include <initializer_list>
#include <regex>
#include <string>

namespace infernix::models::inkling2 {
namespace {

std::uint32_t round_up16(std::uint32_t value) { return (value + 15U) / 16U * 16U; }

std::string base_named(const artifact::PeftLoraAdapter& adapter) {
    return adapter.base_model().empty() ? std::string{} : " (it names " + adapter.base_model() + ")";
}

[[noreturn]] void refuse(const artifact::PeftLoraAdapter& adapter, const std::string& what) {
    throw artifact::ArtifactError("LoRA adapter " + adapter.weights_path().parent_path().string() + " is refused: " +
                                  what);
}

// The projections Infernix adapts, for a refusal's message.
constexpr const char* kAdapted =
    "Infernix adapts Inkling 2's attention projections (self_attn.q_proj.linear, k_proj.linear, v_proj.linear, "
    "relative_position_bias.linear, sink_proj, o_proj.gate_proj, o_proj.linear), the full-attention layers' dynamic "
    "convolution taps (k_proj.sconv.weight_proj, v_proj.sconv.weight_proj) and the loop's message gates "
    "({q,k,v,mlp}_message_gate.down_proj, forget_gate_proj, message_gate_proj)";

} // namespace

LoraBinding bind_lora(const std::filesystem::path& path, const TextConfig& text) {
    auto adapter = std::make_shared<const artifact::PeftLoraAdapter>(path);
    LoraBinding binding;
    binding.layers.resize(text.num_hidden_layers);
    // PEFT's module paths: Inkling2LoopForConditionalGeneration's "model.language_model.layers.N", or
    // Inkling2LoopForCausalLM's "model.layers.N".
    static const std::regex kModule(R"(^model\.(language_model\.)?layers\.(\d+)\.(.+)$)");
    static const std::regex kGate(R"(^(q|k|v|mlp)_message_gate\.(down_proj|forget_gate_proj|message_gate_proj)$)");
    const std::uint64_t H = text.hidden_size, Q = text.query_width(), KV = text.key_width();
    const std::uint64_t R = text.relative_width(), rank = text.loop.message_rank;
    const std::uint64_t taps = static_cast<std::uint64_t>(text.kv_heads) * text.conv_kernel;
    std::uint32_t max_rank = 0;
    std::string max_rank_module;
    for (std::size_t index = 0; index < adapter->modules().size(); ++index) {
        const artifact::PeftLoraModule& m = adapter->modules()[index];
        std::smatch match;
        if (!std::regex_match(m.module, match, kModule)) {
            // The routed experts and router are parameters (never a PEFT LoRA module); the rest are other components.
            refuse(*adapter, "it adapts module '" + m.module + "', which is not a text layer's projection. " + kAdapted +
                                 "; not the embeddings, the output head, the vision tower or the DSpark drafter");
        }
        const std::uint64_t layer = std::stoull(match[2].str());
        const std::string leaf    = match[3].str();
        if (layer >= text.num_hidden_layers) {
            refuse(*adapter, "it adapts module '" + m.module + "' in layer " + std::to_string(layer) +
                                 ", but this model has " + std::to_string(text.num_hidden_layers) +
                                 " layers: the adapter was trained for another base model" + base_named(*adapter));
        }
        const bool full = text.layer_types.at(layer) == AttentionKind::Full;
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
        std::smatch gate;
        if (leaf == "self_attn.q_proj.linear") {
            expect(H, Q);
            place(target.query);
        } else if (leaf == "self_attn.k_proj.linear") {
            expect(H, KV);
            place(target.key);
        } else if (leaf == "self_attn.v_proj.linear") {
            expect(H, KV);
            place(target.value);
        } else if (leaf == "self_attn.relative_position_bias.linear") {
            expect(H, R);
            place(target.relative);
        } else if (leaf == "self_attn.sink_proj") {
            expect(H, text.heads);
            place(target.sink);
        } else if (leaf == "self_attn.o_proj.gate_proj") {
            expect(H, Q);
            place(target.gate);
        } else if (leaf == "self_attn.o_proj.linear") {
            expect(Q, H);
            place(target.output);
        } else if (leaf == "self_attn.k_proj.sconv.weight_proj" || leaf == "self_attn.v_proj.sconv.weight_proj") {
            if (!full) {
                refuse(*adapter, "it adapts module '" + m.module + "', but layer " + std::to_string(layer) +
                                     " is a sliding-window layer, whose convolutions are static: the adapter was "
                                     "trained for another base model" + base_named(*adapter));
            }
            expect(H, taps);
            place(leaf == "self_attn.k_proj.sconv.weight_proj" ? target.key_taps : target.value_taps);
        } else if (std::regex_match(leaf, gate, kGate)) {
            if (!text.in_core(static_cast<std::uint32_t>(layer))) {
                refuse(*adapter, "it adapts message gate module '" + m.module + "', but layer " + std::to_string(layer) +
                                     " is outside the loop's core: the adapter was trained for another base model" +
                                     base_named(*adapter));
            }
            const std::string kind = gate[1].str(), which = gate[2].str();
            LoraGate& g = target.gates[kind == "q" ? 0 : kind == "k" ? 1 : kind == "v" ? 2 : 3];
            if (which == "down_proj") {
                expect(2 * H, rank);
                place(g.down);
            } else if (which == "forget_gate_proj") {
                expect(rank, H);
                place(g.forget);
            } else {
                expect(rank, H);
                place(g.message);
            }
        } else if (leaf.rfind("mlp", 0) == 0) {
            refuse(*adapter, "it adapts module '" + m.module +
                                 "': Inkling 2's routed experts and router are parameters of the mixture of experts, "
                                 "which a plain LoRA adapter cannot target. " + kAdapted);
        } else {
            refuse(*adapter, "it adapts module '" + m.module + "'. " + kAdapted);
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
    binding.adapter = std::move(adapter);
    return binding;
}

std::string LoraBinding::describe() const {
    std::size_t attention = 0, gates = 0, layers_adapted = 0;
    for (const LoraLayer& layer : layers) {
        for (const auto* slot : {&layer.query, &layer.relative, &layer.gate, &layer.sink, &layer.key, &layer.value,
                                 &layer.key_taps, &layer.value_taps, &layer.output}) {
            attention += slot->has_value() ? 1 : 0;
        }
        for (const LoraGate& g : layer.gates) {
            gates += (g.down ? 1 : 0) + (g.forget ? 1 : 0) + (g.message ? 1 : 0);
        }
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
    char line[512];
    std::snprintf(line, sizeof(line),
                  "%zu attention and %zu message gate projections in %zu layers, rank %u%s, scale %g%s, %s factors, base %s",
                  attention, gates, layers_adapted, max_rank,
                  min_rank != max_rank ? (" (from " + std::to_string(min_rank) + ")").c_str() : "",
                  static_cast<double>(max_scale), min_scale != max_scale ? " (varies)" : "", adapter->dtype().c_str(),
                  adapter->base_model().empty() ? "unnamed" : adapter->base_model().c_str());
    return line;
}

LoraFastLayout LoraBinding::fast_layout() const {
    LoraFastLayout out;
    out.modules.resize(adapter->modules().size());
    const auto reserve = [&](std::uint64_t bytes) {
        const std::uint64_t at = out.bytes;
        out.bytes += lora_region_bytes(bytes);
        return at;
    };
    using Slot = std::optional<std::size_t>;
    // One term group at its own rank; `stacked`: its A rows contiguous (they share K).
    const auto place = [&](std::initializer_list<const Slot*> group, bool stacked) {
        std::uint32_t rank = 0;
        std::uint64_t rows = 0, in = 0;
        for (const Slot* slot : group) {
            if (!*slot) { continue; }
            rank = std::max(rank, round_up16(module(**slot).rank));
            in   = module(**slot).in;
            ++rows;
        }
        if (rank == 0) { return; }
        std::uint64_t a = stacked ? reserve(rows * rank * in * 2) : 0;
        for (const Slot* slot : group) {
            if (!*slot) { continue; }
            const artifact::PeftLoraModule& m = module(**slot);
            LoraFastModule& placed            = out.modules[**slot];
            placed.rank                       = rank;
            if (stacked) {
                placed.a = a;
                a += std::uint64_t{rank} * m.in * 2;
            } else {
                placed.a = reserve(std::uint64_t{rank} * m.in * 2);
            }
            placed.b = reserve(lora_padded_rows(m.out) * rank * 2);
        }
    };
    for (const LoraLayer& l : layers) {
        place({&l.query, &l.relative, &l.gate, &l.sink, &l.key, &l.key_taps, &l.value, &l.value_taps}, true);
        place({&l.output}, false);
        place({&l.gates[0].down, &l.gates[1].down, &l.gates[2].down}, false);
        place({&l.gates[0].forget, &l.gates[0].message, &l.gates[1].forget, &l.gates[1].message, &l.gates[2].forget,
               &l.gates[2].message},
              false);
        place({&l.gates[3].down}, false);
        place({&l.gates[3].forget, &l.gates[3].message}, false);
    }
    return out;
}

std::uint64_t LoraBinding::device_bytes(bool lossless) const {
    if (!lossless) { return fast_layout().bytes; }
    const std::uint64_t r = padded_rank;
    std::uint64_t bytes   = 0;
    for (const auto& m : adapter->modules()) {
        // A's three exact BF16 planes [r, 3 in] and B in FP32 [out, r].
        bytes += lora_region_bytes(r * m.in * 6);
        bytes += lora_region_bytes(m.out * r * 4);
    }
    return bytes;
}

} // namespace infernix::models::inkling2
