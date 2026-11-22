// Hugging Face PEFT LoRA adapters (--lora): the safetensors reader and PeftLoraAdapter's configuration handling.
//
// 1. Factors round-trip exactly from F32, BF16 and F16 files; the header's ranges are validated.
// 2. Modules pair their lora_A and lora_B; rank and alpha come from r / lora_alpha or the first matching
//    rank_pattern / alpha_pattern key (PEFT's `(.*\.)?(key)$` match); the scale is alpha / r, or alpha / sqrt(r)
//    with use_rslora.
// 3. Every feature that is not plain LoRA on linear weights, and every tensor that cannot be placed, is refused by
//    name, as are GGUF adapters and a missing adapter_config.json.

#include "artifact/peft_lora.h"
#include "artifact/safetensors.h"
#include "artifact/schema.h"
#include "safetensors_writer.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace infernix::artifact;
using Json = nlohmann::json;
namespace fs = std::filesystem;

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

// The call must throw ArtifactError whose message contains `needle`.
void rejects(const std::function<void()>& call, const std::string& needle, const std::string& label) {
    try {
        call();
    } catch (const ArtifactError& error) {
        require(std::string(error.what()).find(needle) != std::string::npos,
                label + ": refused, but the message lacks '" + needle + "': " + error.what());
        return;
    }
    throw std::runtime_error(label + ": accepted");
}

using infernix::test::safetensors_writer::pattern;
using infernix::test::safetensors_writer::TensorSpec;
using infernix::test::safetensors_writer::write_json;
using Dir = infernix::test::safetensors_writer::TempDir;

void write_safetensors(const fs::path& path, const std::vector<TensorSpec>& tensors,
                       const std::function<void(Json&)>& edit = {}) {
    infernix::test::safetensors_writer::write(path, tensors, edit);
}

const std::string kPrefix = "base_model.model.";
const std::string kQ      = "model.language_model.layers.3.self_attn.q_proj";
const std::string kDown   = "model.language_model.layers.3.mlp.down_proj";

// Two modules: q_proj rank 16 [128 <- 64], down_proj rank 32 [64 <- 96].
std::vector<TensorSpec> factors(const std::string& dtype) {
    return {{kPrefix + kQ + ".lora_A.weight", dtype, {16, 64}, pattern(16 * 64, 1)},
            {kPrefix + kQ + ".lora_B.weight", dtype, {128, 16}, pattern(128 * 16, 2)},
            {kPrefix + kDown + ".lora_A.weight", dtype, {32, 96}, pattern(32 * 96, 3)},
            {kPrefix + kDown + ".lora_B.weight", dtype, {64, 32}, pattern(64 * 32, 4)}};
}

Json config() {
    return {{"peft_type", "LORA"},
            {"r", 16},
            {"lora_alpha", 32},
            {"bias", "none"},
            {"use_dora", false},
            {"use_rslora", false},
            {"modules_to_save", nullptr},
            {"rank_pattern", {{"down_proj", 32}}},
            {"alpha_pattern", {{"layers.3.mlp.down_proj", 8}}},
            {"target_modules", {"q_proj", "down_proj"}},
            {"base_model_name_or_path", "Qwen/Qwen3.8-27B"},
            {"peft_version", "0.17.1"},
            {"eva_config", nullptr},
            {"loftq_config", Json::object()}};
}


const PeftLoraModule& module_named(const PeftLoraAdapter& adapter, const std::string& name) {
    for (const auto& m : adapter.modules()) {
        if (m.module == name) { return m; }
    }
    throw std::runtime_error("module " + name + " missing");
}

void valid_adapters() {
    for (const std::string dtype : {"F32", "BF16", "F16"}) {
        Dir dir("valid_" + dtype);
        const auto tensors = factors(dtype);
        write_safetensors(dir.path / "adapter_model.safetensors", tensors);
        write_json(dir.path / "adapter_config.json", config());
        for (const fs::path& given : {dir.path, dir.path / "adapter_model.safetensors"}) {
            const PeftLoraAdapter adapter(given);
            require(adapter.modules().size() == 2, "two modules expected");
            require(adapter.dtype() == dtype, "dtype " + dtype);
            require(adapter.base_model() == "Qwen/Qwen3.8-27B", "base model");
            const auto& q    = module_named(adapter, kQ);
            const auto& down = module_named(adapter, kDown);
            require(q.rank == 16 && q.in == 64 && q.out == 128, "q_proj shape");
            require(down.rank == 32 && down.in == 96 && down.out == 64, "down_proj shape (rank_pattern)");
            require(q.scale == 2.0F, "q_proj scale lora_alpha / r");
            require(down.scale == 0.25F, "down_proj scale alpha_pattern / rank_pattern");
            require(adapter.parameter_count() == 16ULL * (64 + 128) + 32ULL * (96 + 64), "parameter count");
            // Exact round trip of every factor.
            for (const TensorSpec& t : tensors) {
                const std::vector<float> read = adapter.file().read_f32(t.name);
                require(read == t.values, "factor " + t.name + " (" + dtype + ") changed on reading");
            }
        }
    }
    // rsLoRA: alpha / sqrt(r).
    {
        Dir dir("rslora");
        write_safetensors(dir.path / "adapter_model.safetensors", factors("BF16"));
        Json c          = config();
        c["use_rslora"] = true;
        write_json(dir.path / "adapter_config.json", c);
        const PeftLoraAdapter adapter(dir.path);
        require(adapter.rslora(), "rslora flag");
        require(module_named(adapter, kQ).scale == 8.0F, "rsLoRA q_proj scale 32 / sqrt(16)");
        require(std::fabs(module_named(adapter, kDown).scale - static_cast<float>(8.0 / std::sqrt(32.0))) == 0.0F,
                "rsLoRA down_proj scale 8 / sqrt(32)");
    }
    // A pattern key matches the module path's tail on a dot boundary only: "proj" does not match "q_proj".
    {
        Dir dir("pattern_boundary");
        write_safetensors(dir.path / "adapter_model.safetensors", factors("BF16"));
        Json c             = config();
        c["alpha_pattern"] = {{"proj", 1}, {"layers.3.mlp.down_proj", 8}};
        write_json(dir.path / "adapter_config.json", c);
        const PeftLoraAdapter adapter(dir.path);
        require(module_named(adapter, kQ).scale == 2.0F, "pattern 'proj' must not match q_proj");
    }
}

void refusals() {
    const auto with_config = [](const std::string& label, const std::function<void(Json&)>& edit,
                                const std::string& needle) {
        Dir dir("refuse");
        write_safetensors(dir.path / "adapter_model.safetensors", factors("BF16"));
        Json c = config();
        edit(c);
        write_json(dir.path / "adapter_config.json", c);
        rejects([&] { PeftLoraAdapter adapter(dir.path); }, needle, label);
    };
    with_config("DoRA", [](Json& c) { c["use_dora"] = true; }, "use_dora");
    // A refusal says the adapter is refused, what it uses, and Infernix's plain-LoRA scope.
    with_config("DoRA wording", [](Json& c) { c["use_dora"] = true; },
                " is refused: it is a DoRA adapter (use_dora is set): its weights carry magnitudes. Infernix applies "
                "plain LoRA only");
    with_config("LoRA bias", [](Json& c) { c["lora_bias"] = true; }, "lora_bias");
    with_config("bias all", [](Json& c) { c["bias"] = "all"; }, "bias");
    with_config("modules_to_save", [](Json& c) { c["modules_to_save"] = {"lm_head"}; }, "modules_to_save");
    with_config("fan_in_fan_out", [](Json& c) { c["fan_in_fan_out"] = true; }, "fan_in_fan_out");
    with_config("layer_replication", [](Json& c) { c["layer_replication"] = {{0, 4}}; }, "layer_replication");
    with_config("trainable tokens", [](Json& c) { c["trainable_token_indices"] = {1, 2}; }, "trainable_token_indices");
    with_config("aLoRA", [](Json& c) { c["alora_invocation_tokens"] = {7}; }, "alora_invocation_tokens");
    with_config("target_parameters", [](Json& c) { c["target_parameters"] = {"experts.gate_up_proj"}; },
                "target_parameters");
    with_config("variant config", [](Json& c) { c["arrow_config"] = {{"top_k", 3}}; }, "arrow_config");
    with_config("not LoRA", [](Json& c) { c["peft_type"] = "IA3"; }, "peft_type");
    with_config("rank mismatch", [](Json& c) { c["r"] = 8; }, "rank");
    with_config("bad pattern", [](Json& c) { c["rank_pattern"] = {{"(", 4}}; },
                " is refused: its adapter_config.json's rank_pattern / alpha_pattern key '(' is not a regular expression");

    const auto with_tensors = [](const std::string& label, std::vector<TensorSpec> tensors, const std::string& needle) {
        Dir dir("tensors");
        write_safetensors(dir.path / "adapter_model.safetensors", tensors);
        write_json(dir.path / "adapter_config.json", config());
        rejects([&] { PeftLoraAdapter adapter(dir.path); }, needle, label);
    };
    {
        auto t = factors("BF16");
        t.push_back({kPrefix + kQ + ".lora_magnitude_vector", "BF16", {128, 1}, pattern(128, 5)});
        with_tensors("DoRA magnitude", t, "not a lora_A or lora_B factor");
    }
    {
        auto t = factors("BF16");
        t.erase(t.begin() + 1); // q_proj's lora_B
        with_tensors("lone lora_A", t, "lacks its lora_B");
    }
    {
        auto t     = factors("BF16");
        t[2].dtype = "F32";
        t[3].dtype = "F32";
        with_tensors("mixed dtypes", t, "mix");
    }
    {
        auto t = factors("BF16");
        t.push_back({"model.embed_tokens.weight", "BF16", {4, 4}, pattern(16, 6)});
        with_tensors("foreign tensor", t, "not a PEFT adapter's");
    }
    {
        auto t   = factors("BF16");
        t[0].shape = {16, 64, 1};
        with_tensors("three-dimensional factor", t, "two-dimensional");
    }

    // GGUF and the paths.
    {
        Dir dir("gguf");
        std::ofstream(dir.path / "adapter.gguf") << "GGUF";
        rejects([&] { PeftLoraAdapter adapter(dir.path / "adapter.gguf"); }, "GGUF", "gguf adapter");
        write_safetensors(dir.path / "adapter_model.safetensors", factors("BF16"));
        rejects([&] { PeftLoraAdapter adapter(dir.path); }, "adapter_config.json", "missing config");
        rejects([&] { PeftLoraAdapter adapter(dir.path / "absent"); }, "does not exist", "missing directory");
    }
}

void safetensors_validation() {
    Dir dir("header");
    const fs::path file = dir.path / "x.safetensors";
    const std::vector<TensorSpec> two{{"a", "F32", {2, 2}, pattern(4, 1)}, {"b", "F32", {2}, pattern(2, 2)}};
    write_safetensors(file, two, [](Json& h) { h["b"]["data_offsets"] = {8, 16}; });
    rejects([&] { SafetensorsFile f(file); }, "overlap", "overlapping ranges");
    write_safetensors(file, two, [](Json& h) { h["b"]["data_offsets"] = {16, 32}; });
    rejects([&] { SafetensorsFile f(file); }, "'b'", "range past the data");
    write_safetensors(file, two, [](Json& h) { h["a"]["shape"] = {3, 2}; });
    rejects([&] { SafetensorsFile f(file); }, "'a'", "shape against bytes");
    write_safetensors(file, two, [](Json& h) { h["a"]["dtype"] = "Q9"; });
    rejects([&] { SafetensorsFile f(file); }, "'a'", "unknown dtype");
    write_safetensors(file, two);
    const SafetensorsFile ok(file);
    require(ok.tensors().size() == 2 && ok.metadata().at("format") == "pt", "valid header");
    require(ok.read_f32("a") == two[0].values && ok.read_f32("b") == two[1].values, "F32 round trip");
    rejects([&] { (void)ok.tensor("c"); }, "'c'", "absent tensor");
    {
        std::ofstream(file, std::ios::binary | std::ios::trunc) << "abc";
        rejects([&] { SafetensorsFile f(file); }, "not a safetensors file", "truncated file");
    }
}

} // namespace

int main() {
    try {
        valid_adapters();
        refusals();
        safetensors_validation();
    } catch (const std::exception& error) {
        std::cerr << "peft lora test: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
