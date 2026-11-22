// --lora on Qwen3.5 (models/qwen3_5/load/lora.h): a PEFT adapter's modules bound to a text configuration.
//
// Hugging Face's Qwen3.5 module paths ("model.language_model.layers.N" and the text-only "model.layers.N") map to
// the layer's gated attention (q_proj: each head's query rows then its output gate's, so [2 * query width, hidden]),
// its Gated DeltaNet projections (in_proj_qkv, in_proj_z, in_proj_a, in_proj_b, out_proj) or its dense MLP; ranks
// round up to the term's 16; the shapes must be this model's. Everything this route does not adapt is refused by name:
// attention on a linear-attention layer and linear attention on a full-attention one, MoE MLPs, other leaves (Qwen3-
// Next's in_proj_qkvz among them), vision/MTP/embeddings/head, a layer past the model, ranks above 128 and widths the
// shrink cannot read.

#include "artifact/safetensors_writer.h"
#include "artifact/schema.h"
#include "models/qwen3_5/load/lora.h"

#include <nlohmann/json.hpp>

#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace infernix;
using namespace infernix::models::qwen3_5;
namespace sw = infernix::test::safetensors_writer;
using Json   = nlohmann::json;

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

void rejects(const std::function<void()>& call, const std::string& needle, const std::string& label) {
    try {
        call();
    } catch (const artifact::ArtifactError& error) {
        require(std::string(error.what()).find(needle) != std::string::npos,
                label + ": refused, but the message lacks '" + needle + "': " + error.what());
        return;
    }
    throw std::runtime_error(label + ": accepted");
}

// A small Qwen3.5 text model: hidden 1024, 4 heads of 256 (query width 1024), 2 KV heads, MLP 2048; layers 0-2
// Gated DeltaNet (4 key heads and 8 value heads of 128: q/k/value channels 2048, z 1024, a and b 8), layer 3 full
// attention.
TextConfig small_text(bool moe = false) {
    TextConfig text;
    text.hidden_size       = 1024;
    text.num_hidden_layers = 4;
    text.layer_types       = {MixerKind::LinearAttention, MixerKind::LinearAttention, MixerKind::LinearAttention,
                              MixerKind::FullAttention};
    text.attention         = AttentionConfig{4, 2, 256};
    text.gdn               = GdnConfig{4, 128, 8, 128, 4};
    if (moe) {
        text.ffn = MoeConfig{8, 2, 256, 256};
    } else {
        text.ffn = DenseConfig{2048};
    }
    return text;
}

struct Module {
    std::string path;
    std::uint64_t in, out;
    std::uint32_t rank = 8;
};

// Writes an adapter of these modules (F32 factors) with r = 8 and rank_pattern for the others.
void write_adapter(const std::filesystem::path& dir, const std::vector<Module>& modules) {
    std::vector<sw::TensorSpec> tensors;
    Json ranks = Json::object();
    int seed   = 1;
    for (const Module& m : modules) {
        tensors.push_back({"base_model.model." + m.path + ".lora_A.weight", "F32", {m.rank, m.in},
                           sw::pattern(m.rank * m.in, seed++)});
        tensors.push_back({"base_model.model." + m.path + ".lora_B.weight", "F32", {m.out, m.rank},
                           sw::pattern(m.out * m.rank, seed++)});
        if (m.rank != 8) { ranks[m.path] = m.rank; }
    }
    sw::write(dir / "adapter_model.safetensors", tensors);
    sw::write_json(dir / "adapter_config.json",
                   {{"peft_type", "LORA"}, {"r", 8}, {"lora_alpha", 16}, {"rank_pattern", ranks}, {"bias", "none"}});
}

const std::string kL3 = "model.language_model.layers.3.";
const std::string kL0 = "model.language_model.layers.0.";

void binds_every_projection() {
    const TextConfig text = small_text();
    sw::TempDir dir("lora_binding");
    write_adapter(dir.path, {{kL3 + "self_attn.q_proj", 1024, 2048},
                             {kL3 + "self_attn.k_proj", 1024, 512},
                             {kL3 + "self_attn.v_proj", 1024, 512, 24},
                             {kL3 + "self_attn.o_proj", 1024, 1024},
                             {kL0 + "mlp.gate_proj", 1024, 2048},
                             {"model.layers.0.mlp.up_proj", 1024, 2048}, // the text-only path
                             {kL3 + "mlp.down_proj", 2048, 1024}});
    const LoraBinding binding = bind_lora(dir.path, text);
    require(binding.layers.size() == 4, "one LoraLayer a text layer");
    const LoraLayer& l3 = binding.layers[3];
    const LoraLayer& l0 = binding.layers[0];
    require(l3.query && l3.key && l3.value && l3.output && l3.down && !l3.gate && !l3.up, "layer 3's modules");
    require(l0.gate && l0.up && !l0.down && !l0.attention_input() && !l0.output, "layer 0's modules");
    require(!binding.layers[1].any() && !binding.layers[2].any(), "unadapted layers");
    require(binding.module(*l3.query).module == kL3 + "self_attn.q_proj", "q_proj placed in query");
    require(binding.module(*l0.up).module == "model.layers.0.mlp.up_proj", "text-only path placed");
    require(binding.padded_rank == 32, "the largest rank (24) rounds up to 32");
    require(binding.describe().find("4 attention and 3 MLP projections in 2 layers") != std::string::npos,
            "description: " + binding.describe());
}

// The Gated DeltaNet projections bind in a linear-attention layer (here beside layer 3's attention and an MLP).
void binds_gdn_projections() {
    const TextConfig text = small_text();
    sw::TempDir dir("lora_binding_gdn");
    write_adapter(dir.path, {{kL0 + "linear_attn.in_proj_qkv", 1024, 2048},
                             {kL0 + "linear_attn.in_proj_z", 1024, 1024},
                             {"model.layers.0.linear_attn.in_proj_a", 1024, 8}, // the text-only path
                             {kL0 + "linear_attn.in_proj_b", 1024, 8, 40},
                             {kL0 + "linear_attn.out_proj", 1024, 1024},
                             {kL3 + "self_attn.o_proj", 1024, 1024},
                             {kL0 + "mlp.down_proj", 2048, 1024}});
    const LoraBinding binding = bind_lora(dir.path, text);
    const LoraLayer& l0       = binding.layers[0];
    require(l0.gdn_qkv && l0.gdn_z && l0.gdn_a && l0.gdn_b && l0.gdn_output && l0.down && !l0.attention_input(),
            "layer 0's modules");
    require(l0.gdn_projection() && l0.gdn_control() && l0.gdn_input(), "layer 0's GDN sites");
    require(binding.module(*l0.gdn_a).module == "model.layers.0.linear_attn.in_proj_a", "in_proj_a placed");
    require(binding.module(*l0.gdn_output).module == kL0 + "linear_attn.out_proj", "out_proj placed");
    require(binding.layers[3].output && !binding.layers[3].gdn_input(), "layer 3's attention");
    require(binding.padded_rank == 48, "the largest rank (40) rounds up to 48");
    require(binding.describe().find("1 attention, 5 linear-attention and 1 MLP projections in 2 layers") !=
                std::string::npos,
            "description: " + binding.describe());
}

// Ranks past 64 (the wide kernels') up to the limit, 128, bind; the padded rank rounds the largest up to 16.
void binds_wide_ranks() {
    const TextConfig text = small_text();
    for (const auto& [rank, padded] : {std::pair<std::uint32_t, std::uint32_t>{72, 80}, {96, 96}, {128, 128}}) {
        sw::TempDir dir("lora_binding_wide");
        write_adapter(dir.path, {{kL3 + "mlp.down_proj", 2048, 1024, rank}, {kL0 + "mlp.gate_proj", 1024, 2048}});
        const LoraBinding binding = bind_lora(dir.path, text);
        require(binding.padded_rank == padded, "rank " + std::to_string(rank) + " pads to " + std::to_string(padded));
    }
}

void refuses() {
    const auto refused = [](const std::string& label, std::vector<Module> modules, const std::string& needle,
                            bool moe = false) {
        sw::TempDir dir("lora_binding_refused");
        write_adapter(dir.path, modules);
        const TextConfig text = small_text(moe);
        rejects([&] { (void)bind_lora(dir.path, text); }, needle, label);
    };
    refused("Qwen3-Next's fused leaf", {{kL0 + "linear_attn.in_proj_qkvz", 1024, 4096}},
            "Infernix adapts in_proj_qkv, in_proj_z, in_proj_a, in_proj_b and out_proj");
    refused("linear attention on a full-attention layer", {{kL3 + "linear_attn.in_proj_qkv", 1024, 2048}},
            "that layer is a full-attention layer in this model");
    refused("another model's GDN", {{kL0 + "linear_attn.in_proj_z", 1024, 2048}}, "another base model");
    refused("attention on a GDN layer", {{kL0 + "self_attn.q_proj", 1024, 2048}},
            "that layer is a linear-attention (Gated DeltaNet) layer in this model");
    refused("MoE MLP", {{kL3 + "mlp.gate_proj", 1024, 2048}}, "mixtures of experts", true);
    refused("expert leaf", {{kL3 + "mlp.experts.0.gate_proj", 1024, 2048}}, "Infernix adapts gate_proj");
    refused("attention leaf", {{kL3 + "self_attn.q_norm", 256, 256}}, "Infernix adapts q_proj");
    refused("head", {{"lm_head", 1024, 4096}}, "not a text layer");
    refused("vision", {{"model.visual.blocks.0.attn.qkv", 1024, 3072}}, "not a text layer");
    refused("MTP", {{"mtp.layers.0.mlp.gate_proj", 1024, 2048}}, "not a text layer");
    refused("layer past the model", {{"model.language_model.layers.4.mlp.gate_proj", 1024, 2048}},
            "this model has 4 layers");
    refused("q_proj without its gate rows", {{kL3 + "self_attn.q_proj", 1024, 1024}}, "another base model");
    refused("another model's MLP", {{kL3 + "mlp.down_proj", 4096, 1024}}, "another base model");
    // Infernix's rank limit, named as such with the adapter's rank and its module.
    refused("rank above 128", {{kL3 + "mlp.down_proj", 2048, 1024, 144}},
            "its rank is 144 (module '" + kL3 + "mlp.down_proj'), and Infernix currently supports LoRA adapters up to "
            "rank 128");
    refused("rank 129 (padded 144)", {{kL0 + "mlp.gate_proj", 1024, 2048}, {kL3 + "mlp.down_proj", 2048, 1024, 129}},
            "up to rank 128");
    // Every refusal names the adapter and says it is refused.
    refused("refusal wording", {{kL3 + "mlp.experts.0.up_proj", 1024, 2048}}, " is refused: it adapts MLP module");
    refused("twice", {{kL3 + "mlp.down_proj", 2048, 1024}, {"model.layers.3.mlp.down_proj", 2048, 1024}}, "twice");
    // A hidden width the shrink cannot read (not a multiple of 1024).
    {
        sw::TempDir dir("lora_binding_width");
        write_adapter(dir.path, {{kL3 + "mlp.down_proj", 2048, 1536}});
        TextConfig text  = small_text();
        text.hidden_size = 1536;
        rejects([&] { (void)bind_lora(dir.path, text); }, "multiple of 1024", "hidden width");
    }
}

} // namespace

int main() {
    try {
        binds_every_projection();
        binds_gdn_projections();
        refuses();
        binds_wide_ranks();
    } catch (const std::exception& error) {
        std::cerr << "lora binding test: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
