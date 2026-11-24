// --lora on Inkling 2 (models/inkling2/lora.h): a PEFT adapter's modules bound to the text configuration (host only).
//
// The module paths are transformers' Inkling2Loop ("model.language_model.layers.N" and the text-only "model.layers.N",
// derived from modeling_inkling2_loop.py: no public adapter exists yet). Each leaf maps to its projection with its
// shapes; the dynamic-convolution taps exist on full layers only and the message gates in the loop's core only; ranks
// round up to 16. Everything else is refused by name: the routed experts and the router, the head and embeddings, the
// vision tower, the drafter, a layer past the model, a wrong shape, a rank above 128.

#include "artifact/safetensors_writer.h"
#include "artifact/schema.h"
#include "models/inkling2/lora.h"

#include <nlohmann/json.hpp>

#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace infernix;
using namespace infernix::models::inkling2;
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

// Inkling 2 Mini's geometry.
TextConfig inkling_text() {
    TextConfig text;
    text.hidden_size       = 2304;
    text.num_hidden_layers = 24;
    text.heads             = 24;
    text.kv_heads          = 8;
    text.head_dim          = 128;
    text.relative_dim      = 16;
    text.conv_kernel       = 4;
    text.loop              = {6, 18, 3, 64};
    text.layer_types.assign(24, AttentionKind::Sliding);
    for (const std::size_t full : {5U, 11U, 17U, 23U}) { text.layer_types[full] = AttentionKind::Full; }
    return text;
}

struct Module {
    std::string path;
    std::uint64_t in, out;
    std::uint32_t rank = 8;
};

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

const std::string kFull  = "model.language_model.layers.11."; // a full layer of the core
const std::string kSlide = "model.language_model.layers.7.";  // a sliding layer of the core
const std::string kPre   = "model.layers.0.";                 // the text-only path, a prelude layer

void binds_every_projection() {
    const TextConfig text = inkling_text();
    sw::TempDir dir("inkling_lora");
    write_adapter(dir.path, {{kFull + "self_attn.q_proj.linear", 2304, 3072},
                             {kFull + "self_attn.k_proj.linear", 2304, 1024, 24},
                             {kFull + "self_attn.v_proj.linear", 2304, 1024},
                             {kFull + "self_attn.relative_position_bias.linear", 2304, 384},
                             {kFull + "self_attn.sink_proj", 2304, 24},
                             {kFull + "self_attn.o_proj.gate_proj", 2304, 3072},
                             {kFull + "self_attn.o_proj.linear", 3072, 2304},
                             {kFull + "self_attn.k_proj.sconv.weight_proj", 2304, 32},
                             {kFull + "self_attn.v_proj.sconv.weight_proj", 2304, 32},
                             {kSlide + "q_message_gate.down_proj", 4608, 64},
                             {kSlide + "v_message_gate.forget_gate_proj", 64, 2304},
                             {kSlide + "mlp_message_gate.message_gate_proj", 64, 2304},
                             {kPre + "self_attn.q_proj.linear", 2304, 3072}});
    const LoraBinding b = bind_lora(dir.path, text);
    require(b.layers.size() == 24, "one LoraLayer a text layer");
    const LoraLayer& f = b.layers[11];
    require(f.query && f.key && f.value && f.relative && f.sink && f.gate && f.output && f.key_taps && f.value_taps,
            "the full layer's attention modules");
    require(b.module(*f.key_taps).module == kFull + "self_attn.k_proj.sconv.weight_proj", "k taps placed");
    require(b.module(*f.value_taps).module == kFull + "self_attn.v_proj.sconv.weight_proj", "v taps placed");
    const LoraLayer& s = b.layers[7];
    require(s.gates[0].down && !s.gates[0].forget && s.gates[2].forget && s.gates[3].message && !s.attention_input(),
            "the sliding core layer's gates");
    require(b.layers[0].query && !b.layers[0].any_gate(), "the text-only path");
    require(!b.layers[1].any(), "an unadapted layer");
    require(b.padded_rank == 32, "the largest rank (24) rounds up to 32");
    require(b.describe().find("10 attention and 3 message gate projections in 3 layers") != std::string::npos,
            "the description counts: " + b.describe());
    // Lossless: A's three planes and B in FP32 at the adapter's largest padded rank.
    std::uint64_t exact = 0;
    for (const auto& m : b.adapter->modules()) {
        exact += lora_region_bytes(std::uint64_t{32} * m.in * 6) + lora_region_bytes(m.out * 32 * 4);
    }
    require(b.device_bytes(true) == exact, "the lossless route's factor bytes");
    // The fast routes: each term group at its own rank. Layer 11's K2 group holds the rank-24 key, so 32; its W_o 16;
    // K2's A rows one stack in input order (x_q's, then x_k's, then x_v's).
    const LoraFastLayout fast = b.fast_layout();
    require(fast.bytes == b.device_bytes(false), "the fast layout is the planned bytes");
    require(fast.modules[*f.query].rank == 32 && fast.modules[*f.value_taps].rank == 32, "the K2 group's rank");
    require(fast.modules[*f.output].rank == 16, "W_o runs at its own rank");
    require(fast.modules[*s.gates[0].down].rank == 16, "a gate group's rank");
    const std::uint64_t row = 32 * 2304 * 2;
    require(fast.modules[*f.relative].a == fast.modules[*f.query].a + row &&
                fast.modules[*f.gate].a == fast.modules[*f.query].a + 2 * row &&
                fast.modules[*f.sink].a == fast.modules[*f.query].a + 3 * row &&
                fast.modules[*f.key].a == fast.modules[*f.query].a + 4 * row &&
                fast.modules[*f.key_taps].a == fast.modules[*f.query].a + 5 * row &&
                fast.modules[*f.value].a == fast.modules[*f.query].a + 6 * row &&
                fast.modules[*f.value_taps].a == fast.modules[*f.query].a + 7 * row,
            "K2's A rows are one stack in input order");
}

void refuses() {
    const TextConfig text = inkling_text();
    const auto single = [&](const Module& m, const std::string& needle, const std::string& label) {
        sw::TempDir dir("inkling_lora_refusal");
        write_adapter(dir.path, {m});
        rejects([&] { (void)bind_lora(dir.path, text); }, needle, label);
    };
    single({kSlide + "mlp.experts.gate_up_proj", 2304, 2304}, "parameters of the mixture of experts", "experts");
    single({kSlide + "mlp.router", 2304, 128}, "parameters of the mixture of experts", "router");
    single({"lm_head", 2304, 201024}, "not a text layer's projection", "the head");
    single({"model.language_model.embed_tokens", 2304, 2304}, "not a text layer's projection", "embeddings");
    single({"model.vision_tower.layers.0.attn.qkv", 1024, 3072}, "not a text layer's projection", "vision");
    single({"mtp.layers.0.self_attn.q_proj", 2304, 3072}, "not a text layer's projection", "the drafter");
    single({"model.layers.24.self_attn.q_proj.linear", 2304, 3072}, "this model has 24 layers", "a layer past the model");
    single({kFull + "self_attn.q_proj.linear", 2304, 2048}, "another base model", "a wrong shape");
    single({kSlide + "self_attn.k_proj.sconv.weight_proj", 2304, 32}, "sliding-window layer", "taps on a sliding layer");
    single({kPre + "q_message_gate.down_proj", 4608, 64}, "outside the loop's core", "a gate outside the core");
    single({kFull + "self_attn.q_proj", 2304, 3072}, "Infernix adapts", "the wrapper module, not its linear");
    single({kFull + "self_attn.q_proj.linear", 2304, 3072, 256}, "up to rank 128", "a rank above 128");
}

} // namespace

int main() {
    try {
        binds_every_projection();
        refuses();
    } catch (const std::exception& error) {
        std::cerr << "FAILED: " << error.what() << "\n";
        return 1;
    }
    std::cout << "PASS\n";
    return 0;
}
