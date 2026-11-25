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

// adapter_config.json in its file order: PEFT takes the first rank_pattern / alpha_pattern key that matches a module in
// the order the file lists them, where a sorted map would try them alphabetically.
using ConfigJson = nlohmann::ordered_json;

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
bool feature_set(const ConfigJson& config, const char* key) {
    if (!config.contains(key)) { return false; }
    const ConfigJson& value = config.at(key);
    if (value.is_null()) { return false; }
    if (value.is_boolean()) { return value.get<bool>(); }
    if (value.is_array() || value.is_object() || value.is_string()) { return !value.empty(); }
    return true;
}

// PEFT's pattern lookup (peft.utils.other.get_pattern_key): the first key of rank_pattern / alpha_pattern that matches
// the module path as `(.*\.)?(key)$`. `adapter` names the adapter in a refusal.
const ConfigJson* pattern_value(const ConfigJson& patterns, const std::string& module, const std::string& adapter) {
    if (!patterns.is_object()) { return nullptr; }
    for (const auto& [key, value] : patterns.items()) {
        try {
            if (std::regex_match(module, std::regex("(.*\\.)?(" + key + ")"))) { return &value; }
        } catch (const std::regex_error&) {
            throw ArtifactError("LoRA adapter " + adapter + " is refused: its adapter_config.json's rank_pattern / "
                                "alpha_pattern key '" + key + "' is not a regular expression");
        }
    }
    return nullptr;
}

} // namespace

PeftLoraAdapter::PeftLoraAdapter(const std::filesystem::path& path) {
    std::filesystem::path weights;
    const std::string extension = lower(path.extension().string());
    if (extension == ".gguf") {
        throw ArtifactError("LoRA adapter " + path.string() +
                            " is refused: it is a GGUF adapter (llama.cpp's export), which Infernix does not read; give "
                            "the adapter's Hugging Face PEFT folder (adapter_model.safetensors with adapter_config.json "
                            "beside it)");
    }
    if (std::error_code error; !std::filesystem::exists(path, error)) {
        throw ArtifactError("LoRA adapter " + path.string() + " does not exist");
    }
    if (std::filesystem::is_directory(path)) {
        weights      = path / "adapter_model.safetensors";
        config_path_ = path / "adapter_config.json";
    } else {
        if (extension != ".safetensors") {
            throw ArtifactError("LoRA adapter " + path.string() +
                                " is refused: give the adapter's Hugging Face PEFT folder, or its "
                                "adapter_model.safetensors with adapter_config.json beside it");
        }
        weights      = path;
        config_path_ = path.parent_path() / "adapter_config.json";
    }
    if (!std::filesystem::is_regular_file(weights)) {
        throw ArtifactError("LoRA adapter " + path.string() + " is refused: its weights file " + weights.string() +
                            " does not exist");
    }
    if (!std::filesystem::is_regular_file(config_path_)) {
        throw ArtifactError("LoRA adapter " + path.string() + " is refused: " + config_path_.string() +
                            " is missing, and the adapter's rank, alpha and options come from it");
    }
    ConfigJson config;
    {
        std::ifstream in(config_path_, std::ios::binary);
        std::stringstream text;
        text << in.rdbuf();
        try {
            config = ConfigJson::parse(text.str());
        } catch (const Json::exception& error) {
            throw ArtifactError("LoRA adapter " + config_path_.parent_path().string() + " is refused: " +
                                config_path_.string() + " is not JSON (" + error.what() + ")");
        }
    }
    // A refusal says what the adapter uses and that Infernix applies plain LoRA only.
    const auto refuse = [&](const std::string& what) {
        throw ArtifactError("LoRA adapter " + config_path_.parent_path().string() + " is refused: it " + what +
                            ". Infernix applies plain LoRA only: W x + scale * B (A x) on linear projections, without "
                            "PEFT's variants");
    };
    const std::string peft_type = config.value("peft_type", std::string{});
    if (peft_type != "LORA") {
        refuse("is a PEFT " + (peft_type.empty() ? std::string("adapter of no stated type") : "'" + peft_type + "' adapter") +
               ", not a LoRA one (adapter_config.json's peft_type is '" + peft_type + "')");
    }
    if (feature_set(config, "use_dora")) { refuse("is a DoRA adapter (use_dora is set): its weights carry magnitudes"); }
    if (feature_set(config, "lora_bias")) { refuse("trains a bias beside lora_B (lora_bias is set)"); }
    if (config.contains("bias") && config.at("bias").is_string() && config.at("bias").get<std::string>() != "none") {
        refuse("trains the base model's biases (bias is '" + config.at("bias").get<std::string>() + "', not 'none')");
    }
    // Each of these changes the computation. (layers_to_transform and layers_pattern only restrict which modules carry
    // factors, which the tensors present decide here.)
    struct Feature {
        const char* key;
        const char* what;
    };
    for (const Feature& feature : {
             Feature{"modules_to_save", "carries whole trained copies of modules (modules_to_save is set)"},
             Feature{"layer_replication", "replicates layers (layer_replication is set), changing the model's depth"},
             Feature{"fan_in_fan_out", "stores its factors for Conv1D layers (fan_in_fan_out is set)"},
             Feature{"alora_invocation_tokens",
                     "is an activated LoRA (alora_invocation_tokens is set), applied only after its invocation tokens"},
             Feature{"use_qalora", "is a QA-LoRA adapter (use_qalora is set)"},
             Feature{"target_parameters", "adapts parameters directly (target_parameters is set), not projections"},
             Feature{"trainable_token_indices", "trains token embeddings (trainable_token_indices is set)"},
             Feature{"use_bdlora", "is a BD-LoRA adapter (use_bdlora is set)"},
             Feature{"megatron_config", "keeps Megatron's parallel layers (megatron_config is set)"},
         }) {
        if (feature_set(config, feature.key)) { refuse(feature.what); }
    }
    // A variant's own configuration (arrow, KaSA, VeLoRA, MonteCLoRA, ...) may change the computation; only the
    // initialization methods' (LoftQ, EVA, CorDA, LoRA-GA), which act before training, are accepted when set.
    for (const auto& [key, value] : config.items()) {
        if (key.size() > 7 && key.substr(key.size() - 7) == "_config" && key != "loftq_config" &&
            key != "eva_config" && key != "corda_config" && key != "lora_ga_config" && feature_set(config, key.c_str())) {
            refuse("uses a LoRA variant (" + key + " is set) whose computation differs from plain LoRA");
        }
    }
    rslora_       = config.value("use_rslora", false);
    base_model_   = config.value("base_model_name_or_path", std::string{});
    peft_version_ = config.value("peft_version", std::string{});
    const double default_alpha = config.contains("lora_alpha") ? config.at("lora_alpha").get<double>() : 8.0;
    const std::uint64_t default_rank = config.contains("r") ? config.at("r").get<std::uint64_t>() : 8;
    const ConfigJson rank_pattern  = config.contains("rank_pattern") && config.at("rank_pattern").is_object()
                                   ? config.at("rank_pattern")
                                   : ConfigJson::object();
    const ConfigJson alpha_pattern = config.contains("alpha_pattern") && config.at("alpha_pattern").is_object()
                                   ? config.at("alpha_pattern")
                                   : ConfigJson::object();

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
            throw ArtifactError("LoRA adapter " + weights.string() + " is refused: tensor '" + name +
                                "' is not a PEFT adapter's (their names start with base_model.model.)");
        }
        view.remove_prefix(kPrefix.size());
        if (ends_with(view, kASuffix)) {
            pairs[std::string(view.substr(0, view.size() - kASuffix.size()))].first = name;
        } else if (ends_with(view, kBSuffix)) {
            pairs[std::string(view.substr(0, view.size() - kBSuffix.size()))].second = name;
        } else {
            throw ArtifactError("LoRA adapter " + weights.string() + " is refused: it holds tensor '" + name +
                                "', which is not a lora_A or lora_B factor. Infernix applies plain LoRA only, without "
                                "DoRA magnitudes, biases, embeddings or saved modules");
        }
        if (tensor.dtype != "F32" && tensor.dtype != "BF16" && tensor.dtype != "F16") {
            throw ArtifactError("LoRA adapter " + weights.string() + " is refused: tensor '" + name + "' is " +
                                tensor.dtype + "; Infernix reads LoRA factors stored as F32, BF16 or F16 (not "
                                "quantized ones)");
        }
        if (dtype_.empty()) {
            dtype_ = tensor.dtype;
        } else if (dtype_ != tensor.dtype) {
            throw ArtifactError("LoRA adapter " + weights.string() + " is refused: its factors mix " + dtype_ +
                                " and " + tensor.dtype + "; Infernix reads an adapter whose factors share one type");
        }
        if (tensor.shape.size() != 2) {
            throw ArtifactError("LoRA adapter " + weights.string() + " is refused: tensor '" + name +
                                "' is not two-dimensional, as a LoRA factor is");
        }
    }
    if (pairs.empty()) {
        throw ArtifactError("LoRA adapter " + weights.string() + " is refused: it holds no LoRA factors");
    }
    for (const auto& [module, names] : pairs) {
        if (names.first.empty() || names.second.empty()) {
            throw ArtifactError("LoRA adapter " + weights.string() + " is refused: module '" + module +
                                "' lacks its " + (names.first.empty() ? "lora_A" : "lora_B") + " factor");
        }
        const SafetensorsTensor& a = file_->tensor(names.first);
        const SafetensorsTensor& b = file_->tensor(names.second);
        if (a.shape[0] != b.shape[1] || a.shape[0] == 0) {
            throw ArtifactError("LoRA adapter " + weights.string() + " is refused: module '" + module +
                                "' has lora_A [" + std::to_string(a.shape[0]) + ", " + std::to_string(a.shape[1]) +
                                "] and lora_B [" + std::to_string(b.shape[0]) + ", " + std::to_string(b.shape[1]) +
                                "], whose ranks differ");
        }
        PeftLoraModule m;
        m.module   = module;
        m.a_tensor = names.first;
        m.b_tensor = names.second;
        m.rank     = static_cast<std::uint32_t>(a.shape[0]);
        m.in       = a.shape[1];
        m.out      = b.shape[0];
        // PEFT's per-module rank and alpha: the pattern's when one matches the module path, else r and lora_alpha.
        const ConfigJson* rank_value  = pattern_value(rank_pattern, module, config_path_.parent_path().string());
        const ConfigJson* alpha_value = pattern_value(alpha_pattern, module, config_path_.parent_path().string());
        const std::uint64_t rank = rank_value != nullptr ? rank_value->get<std::uint64_t>() : default_rank;
        if (rank != m.rank) {
            throw ArtifactError("LoRA adapter " + weights.string() + " is refused: adapter_config.json gives module '" +
                                module + "' rank " + std::to_string(rank) + ", but its factors have rank " +
                                std::to_string(m.rank));
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
