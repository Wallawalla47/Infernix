#include "artifact/peft_lora.h"

#include "artifact/schema.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <map>
#include <regex>
#include <sstream>

namespace infernix::artifact {
namespace {

constexpr std::string_view kPrefix  = "base_model.model.";
constexpr std::string_view kASuffix = ".lora_A.weight";
constexpr std::string_view kBSuffix = ".lora_B.weight";

bool ends_with(std::string_view text, std::string_view suffix) {
    return text.size() >= suffix.size() && text.substr(text.size() - suffix.size()) == suffix;
}

std::string lower(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) { return std::tolower(c); });
    return text;
}

// A configuration value that turns on a feature this route does not implement: a true boolean, or a non-null value
// that is not an empty list, object or string.
bool feature_set(const Json& config, const char* key) {
    if (!config.contains(key)) { return false; }
    const Json& value = config.at(key);
    if (value.is_null()) { return false; }
    if (value.is_boolean()) { return value.get<bool>(); }
    if (value.is_array() || value.is_object() || value.is_string()) { return !value.empty(); }
    return true;
}

// PEFT's pattern lookup (peft.utils.other.get_pattern_key): the first key of rank_pattern / alpha_pattern that matches
// the module path as `(.*\.)?(key)$`.
const Json* pattern_value(const Json& patterns, const std::string& module) {
    if (!patterns.is_object()) { return nullptr; }
    for (const auto& [key, value] : patterns.items()) {
        try {
            if (std::regex_match(module, std::regex("(.*\\.)?(" + key + ")"))) { return &value; }
        } catch (const std::regex_error&) {
            throw ArtifactError("adapter_config.json: pattern '" + key + "' is not a regular expression");
        }
    }
    return nullptr;
}

} // namespace

PeftLoraAdapter::PeftLoraAdapter(const std::filesystem::path& path) {
    std::filesystem::path weights;
    const std::string extension = lower(path.extension().string());
    if (extension == ".gguf") {
        throw ArtifactError(path.string() +
                            ": GGUF LoRA adapters are not read; give the adapter's Hugging Face directory (or its "
                            "adapter_model.safetensors) with adapter_config.json beside it");
    }
    if (std::error_code error; !std::filesystem::exists(path, error)) {
        throw ArtifactError(path.string() + ": the LoRA adapter does not exist");
    }
    if (std::filesystem::is_directory(path)) {
        weights      = path / "adapter_model.safetensors";
        config_path_ = path / "adapter_config.json";
    } else {
        if (extension != ".safetensors") {
            throw ArtifactError(path.string() +
                                ": a LoRA adapter is its Hugging Face directory or its .safetensors weights file");
        }
        weights      = path;
        config_path_ = path.parent_path() / "adapter_config.json";
    }
    if (!std::filesystem::is_regular_file(weights)) {
        throw ArtifactError(weights.string() + ": the adapter's weights file does not exist");
    }
    if (!std::filesystem::is_regular_file(config_path_)) {
        throw ArtifactError(config_path_.string() +
                            ": missing; the adapter's rank, alpha and options come from adapter_config.json beside "
                            "its weights");
    }
    Json config;
    {
        std::ifstream in(config_path_, std::ios::binary);
        std::stringstream text;
        text << in.rdbuf();
        try {
            config = Json::parse(text.str());
        } catch (const Json::exception& error) {
            throw ArtifactError(config_path_.string() + ": not JSON (" + error.what() + ")");
        }
    }
    const auto refuse = [&](const std::string& what) {
        throw ArtifactError(config_path_.string() + ": " + what + "; Infernix applies plain LoRA on linear weights");
    };
    const std::string peft_type = config.value("peft_type", std::string{});
    if (peft_type != "LORA") { refuse("peft_type is '" + peft_type + "', not LORA"); }
    if (feature_set(config, "use_dora")) { refuse("use_dora (DoRA) is set"); }
    if (feature_set(config, "lora_bias")) { refuse("lora_bias is set"); }
    if (config.contains("bias") && config.at("bias").is_string() && config.at("bias").get<std::string>() != "none") {
        refuse("bias is '" + config.at("bias").get<std::string>() + "', not none");
    }
    // Each of these changes the computation. (layers_to_transform and layers_pattern only restrict which modules carry
    // factors, which the tensors present decide here.)
    for (const char* feature : {"modules_to_save", "layer_replication", "fan_in_fan_out", "alora_invocation_tokens",
                                "use_qalora", "target_parameters", "trainable_token_indices", "use_bdlora",
                                "megatron_config"}) {
        if (feature_set(config, feature)) { refuse(std::string(feature) + " is set"); }
    }
    // A variant's own configuration (arrow, KaSA, VeLoRA, MonteCLoRA, ...) may change the computation; only the
    // initialization methods' (LoftQ, EVA, CorDA, LoRA-GA), which act before training, are accepted when set.
    for (const auto& [key, value] : config.items()) {
        if (key.size() > 7 && key.substr(key.size() - 7) == "_config" && key != "loftq_config" &&
            key != "eva_config" && key != "corda_config" && key != "lora_ga_config" && feature_set(config, key.c_str())) {
            refuse(key + " is set");
        }
    }
    rslora_       = config.value("use_rslora", false);
    base_model_   = config.value("base_model_name_or_path", std::string{});
    peft_version_ = config.value("peft_version", std::string{});
    const double default_alpha = config.contains("lora_alpha") ? config.at("lora_alpha").get<double>() : 8.0;
    const std::uint64_t default_rank = config.contains("r") ? config.at("r").get<std::uint64_t>() : 8;
    const Json rank_pattern  = config.contains("rank_pattern") && config.at("rank_pattern").is_object()
                                   ? config.at("rank_pattern")
                                   : Json::object();
    const Json alpha_pattern = config.contains("alpha_pattern") && config.at("alpha_pattern").is_object()
                                   ? config.at("alpha_pattern")
                                   : Json::object();

    file_ = std::make_unique<SafetensorsFile>(weights);
    {
        const auto written = std::filesystem::last_write_time(weights);
        weights_mtime_     = std::chrono::duration_cast<std::chrono::seconds>(written.time_since_epoch()).count();
    }
    // Pair every lora_A with its lora_B; any other tensor is a feature this route does not apply.
    std::map<std::string, std::pair<std::string, std::string>> pairs; // module -> (A, B)
    for (const auto& [name, tensor] : file_->tensors()) {
        std::string_view view = name;
        if (view.substr(0, kPrefix.size()) != kPrefix) {
            throw ArtifactError(weights.string() + ": tensor '" + name + "' is not a PEFT adapter tensor (base_model.model.)");
        }
        view.remove_prefix(kPrefix.size());
        if (ends_with(view, kASuffix)) {
            pairs[std::string(view.substr(0, view.size() - kASuffix.size()))].first = name;
        } else if (ends_with(view, kBSuffix)) {
            pairs[std::string(view.substr(0, view.size() - kBSuffix.size()))].second = name;
        } else {
            throw ArtifactError(weights.string() + ": tensor '" + name +
                                "' is not a LoRA A or B weight (DoRA magnitudes, embeddings, biases and saved modules "
                                "are not applied)");
        }
        if (tensor.dtype != "F32" && tensor.dtype != "BF16" && tensor.dtype != "F16") {
            throw ArtifactError(weights.string() + ": tensor '" + name + "' is " + tensor.dtype +
                                "; LoRA factors are read as F32, BF16 or F16");
        }
        if (dtype_.empty()) {
            dtype_ = tensor.dtype;
        } else if (dtype_ != tensor.dtype) {
            throw ArtifactError(weights.string() + ": the factors mix " + dtype_ + " and " + tensor.dtype);
        }
        if (tensor.shape.size() != 2) {
            throw ArtifactError(weights.string() + ": tensor '" + name + "' is not two-dimensional");
        }
    }
    if (pairs.empty()) { throw ArtifactError(weights.string() + ": the adapter holds no LoRA factors"); }
    for (const auto& [module, names] : pairs) {
        if (names.first.empty() || names.second.empty()) {
            throw ArtifactError(weights.string() + ": module '" + module + "' lacks its " +
                                (names.first.empty() ? "lora_A" : "lora_B") + " weight");
        }
        const SafetensorsTensor& a = file_->tensor(names.first);
        const SafetensorsTensor& b = file_->tensor(names.second);
        if (a.shape[0] != b.shape[1] || a.shape[0] == 0) {
            throw ArtifactError(weights.string() + ": module '" + module + "': lora_A is [" +
                                std::to_string(a.shape[0]) + ", " + std::to_string(a.shape[1]) + "] and lora_B [" +
                                std::to_string(b.shape[0]) + ", " + std::to_string(b.shape[1]) +
                                "]: their ranks differ");
        }
        PeftLoraModule m;
        m.module   = module;
        m.a_tensor = names.first;
        m.b_tensor = names.second;
        m.rank     = static_cast<std::uint32_t>(a.shape[0]);
        m.in       = a.shape[1];
        m.out      = b.shape[0];
        // PEFT's per-module rank and alpha: the pattern's when one matches the module path, else r and lora_alpha.
        const Json* rank_value  = pattern_value(rank_pattern, module);
        const Json* alpha_value = pattern_value(alpha_pattern, module);
        const std::uint64_t rank = rank_value != nullptr ? rank_value->get<std::uint64_t>() : default_rank;
        if (rank != m.rank) {
            throw ArtifactError(config_path_.string() + ": module '" + module + "' is configured with rank " +
                                std::to_string(rank) + " but its factors have rank " + std::to_string(m.rank));
        }
        const double alpha = alpha_value != nullptr ? alpha_value->get<double>() : default_alpha;
        m.alpha            = static_cast<float>(alpha);
        // PEFT computes the scaling in Python floats (doubles, from the configured alpha) and multiplies by it in the
        // factors' dtype.
        const double scale = rslora_ ? alpha / std::sqrt(static_cast<double>(m.rank))
                                     : alpha / static_cast<double>(m.rank);
        m.scale = static_cast<float>(scale);
        modules_.push_back(std::move(m));
    }
}

PeftLoraAdapter::~PeftLoraAdapter() = default;

std::uint64_t PeftLoraAdapter::weights_bytes() const noexcept { return file_->file_bytes(); }

std::uint64_t PeftLoraAdapter::parameter_count() const noexcept {
    std::uint64_t count = 0;
    for (const auto& m : modules_) { count += m.rank * (m.in + m.out); }
    return count;
}

} // namespace infernix::artifact
