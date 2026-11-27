#include "launcher/artifact_scan.h"

#include "launcher/option_support.h"

#include "launcher/sha256.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iterator>
#include <map>
#include <mutex>
#include <set>
#include <tuple>

namespace infernix::launcher {
namespace {

constexpr char kEntryMagic[8]  = {'N', 'I', 'N', 'F', 'E', 'R', '\0', '\x03'};
constexpr char kPartMagic[8]   = {'N', 'I', 'N', 'P', 'R', 'T', '\0', '\x03'};
constexpr char kNgramMagic[8]  = {'N', 'I', 'N', 'F', 'E', 'R', 'N', 'G'};
constexpr std::uint64_t kPayloadAlignment = 4096;
constexpr std::uint64_t kMaxDirectoryBytes = 512ull << 20;
constexpr std::size_t kNgramHeaderBytes   = 68;
constexpr std::uint64_t kMaxAdapterConfigBytes = 4ull << 20; // a PEFT adapter_config.json is about 1 kB
constexpr const char* kAdapterConfig  = "adapter_config.json";
constexpr const char* kAdapterWeights = "adapter_model.safetensors";
constexpr const char* kSidecarFile    = "sidecar.json";
constexpr std::uint64_t kMaxSidecarBytes = 1ull << 20; // a sidecar.json is a few kB
constexpr const char* kSidecarFormat  = "ablate-direction-v2";

// NVIDIA's published FP8 table records no scale in its header; infernix-serve knows it by id.
constexpr const char* kKnownFp8Volumes[] = {"59e5d77e802ac95d8faf0a800b8f2301"};

template <class T>
T read_le(const char* p) {
    T value{};
    std::memcpy(&value, p, sizeof(T));
    return value; // x86-64 and AArch64 targets are little-endian
}

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

std::uint64_t file_size_or_zero(const std::filesystem::path& path) {
    std::error_code error;
    const auto size = std::filesystem::file_size(path, error);
    return error ? 0 : size;
}

bool has_extension(const std::filesystem::path& path, std::string_view extension) {
    return iequals(path_utf8(path.extension()), extension);
}

// "Name-00001-of-00003.infernix" -> "Name"; "name.ninfer" -> "name".
std::string artifact_stem(const std::filesystem::path& path) {
    std::string stem = path_utf8(path.stem());
    const std::size_t of = stem.rfind("-of-");
    if (of != std::string::npos && of >= 6 && stem[of - 6] == '-') {
        const std::string_view part = std::string_view(stem).substr(of - 5, 5);
        const std::string_view count = std::string_view(stem).substr(of + 4);
        const auto digits = [](std::string_view s) {
            return !s.empty() && std::all_of(s.begin(), s.end(), [](char c) { return c >= '0' && c <= '9'; });
        };
        if (digits(part) && digits(count)) { stem.resize(of - 6); }
    }
    return stem;
}

// The formats an artifact binds as its own table (the engine's config.cpp). BF16 P12 is a run-time
// volume only: it replaces a table through --ngram-volume and is never an artifact's own.
std::string ngram_format_name(std::string_view artifact_format) {
    if (artifact_format == "fp8_e4m3fn") { return "fp8"; }
    if (artifact_format == "iq4_nl" || artifact_format == "bf16" || artifact_format == "nvfp4" ||
        artifact_format == "q8_0") {
        return std::string(artifact_format);
    }
    return "unknown";
}

// The table's row width in elements, from the artifact's own row size and format.
std::uint32_t head_width(const NgramTableFacts& table) {
    if (table.format == "fp8") { return table.row_bytes; }
    if (table.format == "bf16") { return table.row_bytes / 2; }
    if (table.format == "iq4_nl") { return table.row_bytes / 18 * 32; }
    if (table.format == "q8_0") { return table.row_bytes / 34 * 32; }
    if (table.format == "nvfp4") { return table.row_bytes * 16 / 9; }
    return 0;
}

std::uint32_t row_bytes_for(std::string_view format, std::uint32_t width) {
    if (format == "fp8") { return width; }
    if (format == "bf16") { return 2 * width; }
    if (format == "iq4_nl") { return width % 32 == 0 ? width / 32 * 18 : 0; }
    if (format == "q8_0") { return width % 32 == 0 ? width / 32 * 34 : 0; }
    if (format == "nvfp4") { return width % 16 == 0 ? width / 2 + width / 16 : 0; }
    if (format == "bf16_p12") { return width % 2 == 0 && width <= 255 ? width + width / 2 + 16 : 0; }
    return 0;
}

std::string architecture_for(std::string_view model_type) {
    if (model_type == "qwen4_exp_text") { return "Qwen4ExpForCausalLM"; }
    if (model_type == "qwen3_5_moe_text") { return "Qwen3_5MoeForCausalLM"; }
    if (model_type == "qwen3_5_text") { return "Qwen3_5ForCausalLM"; }
    if (model_type == "inkling2_loop_text") { return "Inkling2LoopForCausalLM"; }
    return std::string(model_type);
}

void classify_family(ScannedModel& model) {
    const std::string name = lower(model.name + " " + model.file_name);
    const bool says_38 = name.find("3.8") != std::string::npos || name.find("3_8") != std::string::npos;
    const bool says_36 = name.find("3.6") != std::string::npos || name.find("3_6") != std::string::npos;
    if (model.model_type == "qwen4_exp_text") {
        model.family              = "qwen3.8-flash-next";
        model.family_label        = "Qwen3.8-Flash-Next";
        model.offloaded_experts   = true;
        model.needs_ngram_volume  = true;
        model.accepts_draft_model = true;
        return;
    }
    if (model.model_type == "qwen3_5_moe_text") {
        if (model.hidden_size >= 3072) {
            model.family            = "qwen3.5-122b-a10b";
            model.family_label      = "Qwen3.5-122B-A10B";
            model.offloaded_experts = true;
        } else {
            model.family       = "qwen3.6-35b-a3b";
            model.family_label = "Qwen3.6-35B-A3B";
        }
        return;
    }
    if (model.model_type == "qwen3_5_text") {
        if (model.hidden_size == 5120 && (model.has_dflash2 || (says_38 && !says_36))) {
            model.family              = "qwen3.8-27b";
            model.family_label        = "Qwen3.8-27B";
            model.accepts_draft_model = true; // a DSpine drafter (--spec dspine --draft-model)
        } else if (model.hidden_size == 5120) {
            model.family       = "qwen3.6-27b";
            model.family_label = "Qwen3.6-27B";
        } else {
            model.family       = "qwen3.5";
            model.family_label = "Qwen3.5 family";
        }
        return;
    }
    if (model.model_type == "inkling2_loop_text") {
        model.family              = "inkling2-mini";
        model.family_label        = "Inkling 2 Mini";
        model.accepts_draft_model = true; // its DFlash2 companion (--spec dflash2 --draft-model)
        return;
    }
    model.family       = "unknown";
    model.family_label = model.model_type.empty() ? "Unknown" : model.model_type;
}

// An adapter folder: adapter_config.json with adapter_model.safetensors beside it. Nothing when the weights are not
// there (a download still running, or a GGUF adapter's folder); `issue` when the config cannot be read.
std::optional<ScannedAdapter> read_adapter(const std::filesystem::path& config_path, std::optional<std::string>& issue) {
    const std::filesystem::path folder = config_path.parent_path();
    const std::uint64_t weights_bytes  = file_size_or_zero(folder / kAdapterWeights);
    if (weights_bytes == 0) { return std::nullopt; }
    const std::uint64_t config_bytes = file_size_or_zero(config_path);
    if (config_bytes == 0 || config_bytes > kMaxAdapterConfigBytes) {
        issue = std::string(kAdapterConfig) + ": empty or too large";
        return std::nullopt;
    }
    std::ifstream in(config_path, std::ios::binary);
    std::string text(static_cast<std::size_t>(config_bytes), '\0');
    if (!in.read(text.data(), static_cast<std::streamsize>(text.size()))) {
        issue = std::string(kAdapterConfig) + ": cannot be read";
        return std::nullopt;
    }
    Json config;
    try {
        config = Json::parse(text);
    } catch (const Json::exception&) {
        issue = std::string(kAdapterConfig) + ": not JSON";
        return std::nullopt;
    }
    if (!config.is_object()) {
        issue = std::string(kAdapterConfig) + ": not a JSON object";
        return std::nullopt;
    }
    ScannedAdapter adapter;
    adapter.path     = folder;
    adapter.key      = sha256_hex(path_utf8(folder)).substr(0, 32);
    adapter.name     = path_utf8(folder.filename());
    adapter.bytes    = weights_bytes;
    adapter.config   = parse_adapter_config(config);
    adapter.modified = modified_ms(folder / kAdapterWeights);
    return adapter;
}

// A sidecar.json in the folders: nothing when it is another file of that name or a sidecar cutting no layer.
std::optional<ScannedSidecar> read_sidecar(const std::filesystem::path& path, std::optional<std::string>& issue) {
    const std::uint64_t bytes = file_size_or_zero(path);
    if (bytes == 0 || bytes > kMaxSidecarBytes) { return std::nullopt; }
    std::ifstream in(path, std::ios::binary);
    std::string text(static_cast<std::size_t>(bytes), '\0');
    if (!in.read(text.data(), static_cast<std::streamsize>(text.size()))) { return std::nullopt; }
    Json sidecar;
    try {
        sidecar = Json::parse(text);
    } catch (const Json::exception&) {
        return std::nullopt;
    }
    return parse_sidecar(path, sidecar, issue);
}

// A tools/ablate run directory's fit/ folder (one trial sidecar per trial): not walked.
bool ablate_run_fit_folder(const std::filesystem::path& dir) {
    if (!iequals(path_utf8(dir.filename()), "fit")) { return false; }
    const std::filesystem::path run = dir.parent_path() / "run.json";
    const std::uint64_t bytes       = file_size_or_zero(run);
    if (bytes == 0 || bytes > kMaxSidecarBytes) { return false; }
    std::ifstream in(run, std::ios::binary);
    std::string text(static_cast<std::size_t>(bytes), '\0');
    if (!in.read(text.data(), static_cast<std::streamsize>(text.size()))) { return false; }
    try {
        const Json json = Json::parse(text);
        return json.is_object() && json.value("schema", std::string{}).starts_with("infernix-ablate-run-");
    } catch (const Json::exception&) {
        return false;
    }
}

std::optional<std::string> recipe_of(const Json& provenance) {
    if (!provenance.is_object()) { return std::nullopt; }
    const auto it = provenance.find("recipe");
    if (it == provenance.end()) { return std::nullopt; }
    if (it->is_string()) { return it->get<std::string>(); }
    if (it->is_object()) {
        for (const char* key : {"name", "id", "recipe"}) {
            if (it->contains(key) && (*it)[key].is_string()) { return (*it)[key].get<std::string>(); }
        }
    }
    return std::nullopt;
}

// The artifact's tensor formats (src/artifact/formats.cpp) as weight families with their bits per
// weight, scales included. FP32 norms and INT32 indices are not weights a label names.
struct WeightFamily {
    std::string_view format;
    std::string_view label;
    double bits;
};

constexpr WeightFamily kWeightFamilies[] = {
    {"bf16", "BF16", 16.0},
    {"nvfp4", "NVFP4", 4.5},
    {"nvfp4_mul", "NVFP4", 4.5},
    {"fp8_e4m3fn_row_bf16", "FP8", 8.0},
    {"fp8_e4m3fn_row_f32", "FP8", 8.0},
    {"fp8_e4m3fn_b128", "FP8", 8.0},
    {"fp8_e4m3fn_g32_e8m0", "MXFP8", 8.25},
    {"q2_g64_fp16", "Q2", 2.25},
    {"q4_g64_fp16", "Q4", 4.25},
    {"q5_g64_fp16", "Q5", 5.25},
    {"q6_g64_fp16", "Q6", 6.25},
    {"q8_g32_fp16", "Q8", 8.5},
    {"ggml_q2_0", "Q2_0", 2.5},
    {"ggml_q4_k", "Q4_K", 4.5},
    {"ggml_q5_k", "Q5_K", 5.5},
    {"ggml_q6_k", "Q6_K", 6.5625},
    {"ggml_iq2_s", "IQ2_S", 2.5625},
    {"ggml_iq3_xxs", "IQ3_XXS", 3.0625},
    {"ggml_iq3_s", "IQ3_S", 3.4375},
    {"ggml_iq4_xs", "IQ4_XS", 4.25},
    {"ggml_iq4_nl", "IQ4_NL", 4.5},
};

// "NVFP4 + FP8": the weight family holding the most parameters (bytes over bits per weight), and
// the next when it holds a quarter of them or more. Bytes alone would name FP8 for an NVFP4 model
// whose FP8 attention outweighs its four-bit MLPs on disk.
std::optional<std::string> quantization_label(const Json& objects) {
    if (!objects.is_array()) { return std::nullopt; }
    std::map<std::string_view, double> weights;
    double total = 0.0;
    for (const Json& object : objects) {
        if (json_or<std::string>(object, "kind", "") != "tensor") { continue; }
        const std::string format = json_or<std::string>(object, "format", "");
        const auto family        = std::find_if(std::begin(kWeightFamilies), std::end(kWeightFamilies),
                                                [&](const WeightFamily& f) { return f.format == format; });
        if (family == std::end(kWeightFamilies)) { continue; }
        const double count = static_cast<double>(json_or<std::uint64_t>(object, "bytes", 0)) * 8.0 / family->bits;
        weights[family->label] += count;
        total += count;
    }
    if (total <= 0.0) { return std::nullopt; }
    std::vector<std::pair<std::string_view, double>> ranked(weights.begin(), weights.end());
    std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
    std::string label(ranked.front().first);
    if (ranked.size() > 1 && ranked[1].second >= 0.25 * total) { label += " + " + std::string(ranked[1].first); }
    return label;
}

// One scanned file, remembered by size and modification time so later scans only stat it.
struct FileFacts {
    std::uint64_t size = 0;
    std::int64_t modified = 0;
    std::optional<ScannedModel> model;
    std::optional<ScannedDrafter> drafter;
    std::optional<ScannedNgramVolume> volume;
    std::optional<std::string> issue;
    // Continuation files the entry names, with their expected sizes (checked on every scan).
    std::vector<std::pair<std::string, std::uint64_t>> parts;
};

std::mutex g_cache_mutex;
std::map<std::string, FileFacts> g_cache;

FileFacts read_facts(const std::filesystem::path& path, std::uint64_t size, std::int64_t modified) {
    FileFacts facts;
    facts.size     = size;
    facts.modified = modified;
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        facts.issue = "cannot open the file";
        return facts;
    }
    char head[kNgramHeaderBytes] = {};
    in.read(head, sizeof(head));
    const auto got = static_cast<std::size_t>(in.gcount());
    if (got >= 8 && std::memcmp(head, kPartMagic, 8) == 0) { return facts; }
    if (got >= 8 && std::memcmp(head, kNgramMagic, 8) == 0) {
        facts.volume = parse_ngram_header(path, std::string_view(head, got));
        if (!facts.volume) { facts.issue = "unreadable n-gram volume header"; }
        return facts;
    }
    if (got < 32 || std::memcmp(head, kEntryMagic, 8) != 0) {
        if (!has_extension(path, ".ngram")) {
            facts.issue = "not a v3 artifact (an NInfer v2 artifact needs tools/upgrade_ninfer_v2_to_v3.py)";
        } else {
            facts.issue = "not an n-gram volume";
        }
        return facts;
    }
    const auto json_bytes = read_le<std::uint64_t>(head + 8);
    std::array<std::uint8_t, 16> artifact_id{};
    std::memcpy(artifact_id.data(), head + 16, 16);
    if (json_bytes == 0 || json_bytes > kMaxDirectoryBytes || json_bytes > size) {
        facts.issue = "corrupt artifact header";
        return facts;
    }
    std::string text(json_bytes, '\0');
    in.seekg(32);
    in.read(text.data(), static_cast<std::streamsize>(json_bytes));
    if (static_cast<std::uint64_t>(in.gcount()) != json_bytes) {
        facts.issue = "truncated artifact directory";
        return facts;
    }
    try {
        const Json directory = Json::parse(text);
        ParsedEntry entry = parse_artifact_directory(path, artifact_id, directory,
                                                     align_up(32 + json_bytes, kPayloadAlignment));
        const Json& files = directory.at("files");
        for (std::size_t i = 1; i < files.size(); ++i) {
            facts.parts.emplace_back(json_or<std::string>(files[i], "path", ""),
                                     kPayloadAlignment + json_or<std::uint64_t>(files[i], "payload_bytes", 0));
        }
        facts.model   = std::move(entry.model);
        facts.drafter = std::move(entry.drafter);
        const std::uint64_t entry_payload = json_or<std::uint64_t>(files.at(0), "payload_bytes", 0);
        if (size < align_up(32 + json_bytes, kPayloadAlignment) + entry_payload) {
            facts.issue = "the entry file is shorter than its header says (an unfinished download?)";
        }
    } catch (const std::exception& error) {
        facts.issue = std::string("unreadable artifact directory: ") + error.what();
    }
    return facts;
}

} // namespace

const ScannedModel* LibraryScan::model(std::string_view key) const {
    for (const ScannedModel& m : models) {
        if (m.key == key) { return &m; }
    }
    return nullptr;
}

const ScannedDrafter* LibraryScan::drafter(std::string_view key) const {
    for (const ScannedDrafter& d : drafters) {
        if (d.key == key) { return &d; }
    }
    return nullptr;
}

ParsedEntry parse_artifact_directory(const std::filesystem::path& path,
                                     const std::array<std::uint8_t, 16>& artifact_id,
                                     const Json& directory, std::uint64_t entry_payload_offset) {
    ParsedEntry out;
    const Json& components = directory.at("components");
    if (!components.is_object() || components.empty()) {
        throw std::runtime_error("components must be a nonempty object");
    }
    const Json& files = directory.at("files");
    if (!files.is_array() || files.empty()) { throw std::runtime_error("files must be a nonempty array"); }
    std::uint64_t payload = 0;
    for (const Json& file : files) { payload += json_or<std::uint64_t>(file, "payload_bytes", 0); }
    const std::uint64_t bytes = payload + entry_payload_offset + (files.size() - 1) * kPayloadAlignment;
    const Json metadata       = directory.value("metadata", Json::object());
    const std::string key     = hex_string(artifact_id.data(), artifact_id.size());
    const std::string stem    = artifact_stem(path);
    const std::string name    = json_or<std::string>(metadata, "name", stem);
    std::vector<std::string> names;
    for (const auto& [component, _] : components.items()) { names.push_back(component); }

    if (!components.contains("text")) {
        ScannedDrafter drafter;
        drafter.key       = key;
        drafter.name      = name;
        drafter.file_name = path_utf8(path.filename());
        drafter.path      = path;
        drafter.bytes     = bytes;
        for (const std::string& component : names) {
            drafter.kind += (drafter.kind.empty() ? "" : "+") + component;
        }
        const Json config = components.begin()->value("config", Json::object());
        const Json target = config.value("target_config", Json::object());
        if (target.is_object()) {
            const Json architectures = target.value("architectures", Json::array());
            if (architectures.is_array() && !architectures.empty() && architectures[0].is_string()) {
                drafter.target_architecture = architectures[0].get<std::string>();
            }
            drafter.target_hidden_size = json_or<std::uint64_t>(target, "hidden_size", 0);
            drafter.target_layers      = json_or<std::uint64_t>(target, "num_hidden_layers", 0);
            drafter.target_vocab       = json_or<std::uint64_t>(target, "vocab_size", 0);
        }
        if (drafter.kind == "dspine") {
            // A DSpine drafter records no target config: its own width and vocabulary are the target's, and
            // num_target_layers the target's layer count (models/qwen3_5/config.cpp, parse_dspine_config).
            drafter.target_hidden_size = json_or<std::uint64_t>(config, "hidden_size", 0);
            drafter.target_layers      = json_or<std::uint64_t>(config, "num_target_layers", 0);
            drafter.target_vocab       = json_or<std::uint64_t>(config, "vocab_size", 0);
        }
        drafter.modified = modified_ms(path);
        out.drafter      = std::move(drafter);
        return out;
    }

    const Json& text_component = components.at("text");
    const Json text            = text_component.value("config", Json::object());
    ScannedModel model;
    model.key          = key;
    model.name         = name;
    model.file_name    = path_utf8(path.filename());
    model.path         = path;
    model.locations    = {path};
    model.format       = has_extension(path, ".ninfer") ? "ninfer" : "infernix";
    model.model_type   = json_or<std::string>(text, "model_type", "");
    model.architecture = architecture_for(model.model_type);
    model.recipe       = recipe_of(directory.value("provenance", Json::object()));
    model.quantization = quantization_label(directory.value("objects", Json::array()));
    model.bytes        = bytes;
    model.file_count   = files.size();
    model.components   = names;
    model.has_vision   = components.contains("vision");
    model.has_mtp      = components.contains("mtp");
    model.has_dflash   = components.contains("dflash");
    model.has_dflash2  = components.contains("dflash2");
    model.has_dspark   = components.contains("dspark");
    model.has_proposal = text_component.contains("proposal");
    if (model.model_type == "inkling2_loop_text") { // its ranking is a parameter of the text component
        const Json bindings = directory.value("bindings", Json::object());
        model.has_proposal  = bindings.is_object() && bindings.contains("proposal/token_rank");
    }
    model.native_context = json_or<std::uint64_t>(text, "max_position_embeddings", 0);
    model.hidden_size    = json_or<std::uint64_t>(text, "hidden_size", 0);
    model.num_layers     = json_or<std::uint64_t>(text, "num_hidden_layers", 0);
    model.vocab_size     = json_or<std::uint64_t>(text, "vocab_size", 0);
    if (text.contains("num_experts")) { model.num_experts = json_or<std::uint64_t>(text, "num_experts", 0); }
    if (text.contains("ngram_table") && text["ngram_table"].is_object()) {
        const Json& table = text["ngram_table"];
        NgramTableFacts facts;
        facts.format    = ngram_format_name(json_or<std::string>(table, "format", ""));
        facts.rows      = json_or<std::uint64_t>(table, "rows", 0);
        facts.row_bytes = json_or<std::uint32_t>(table, "row_bytes", 0);
        facts.volume_id = lower(json_or<std::string>(table, "volume_id", ""));
        model.ngram_table = facts;
    }
    classify_family(model);
    std::error_code error;
    std::filesystem::path sidecar = path;
    sidecar += ".expert-state";
    if (std::filesystem::is_regular_file(sidecar, error)) { model.expert_state = sidecar; }
    std::filesystem::path volume = path;
    volume += ".ngram";
    if (model.needs_ngram_volume && std::filesystem::is_regular_file(volume, error)) {
        model.own_ngram_volume = volume;
    }
    model.modified = modified_ms(path);
    out.model      = std::move(model);
    return out;
}

std::optional<ScannedNgramVolume> parse_ngram_header(const std::filesystem::path& path,
                                                     std::string_view header) {
    if (header.size() < kNgramHeaderBytes || std::memcmp(header.data(), kNgramMagic, 8) != 0) {
        return std::nullopt;
    }
    const char* p = header.data();
    if (read_le<std::uint32_t>(p + 8) != 1) { return std::nullopt; }
    ScannedNgramVolume volume;
    volume.path      = path;
    volume.file_name = path_utf8(path.filename());
    volume.name      = path_utf8(path.stem());
    volume.rows      = read_le<std::uint64_t>(p + 16);
    volume.row_bytes = read_le<std::uint32_t>(p + 24);
    volume.key       = hex_string(reinterpret_cast<const std::uint8_t*>(p + 44), 16);
    volume.fp8_scale = read_le<std::uint16_t>(p + 60);
    volume.tag       = read_le<std::uint16_t>(p + 62);
    static constexpr const char* kTagged[] = {"fp8", "iq4_nl", "bf16", "nvfp4", "q8_0", "bf16_p12"};
    if (volume.tag >= 1 && volume.tag <= 6) {
        volume.format = kTagged[volume.tag - 1];
    } else {
        // Untagged volumes predate NVFP4 tables; Qwen3.8-Flash-Next rows are 160 elements wide.
        constexpr std::uint32_t kWidth = 160;
        volume.format = volume.row_bytes == row_bytes_for("fp8", kWidth)      ? "fp8"
                        : volume.row_bytes == row_bytes_for("bf16", kWidth)   ? "bf16"
                        : volume.row_bytes == row_bytes_for("iq4_nl", kWidth) ? "iq4_nl"
                                                                              : "unknown";
    }
    volume.expected_bytes = std::uint64_t{read_le<std::uint32_t>(p + 12)} +
                            read_le<std::uint64_t>(p + 36) * read_le<std::uint32_t>(p + 32);
    volume.bytes    = file_size_or_zero(path);
    volume.modified = modified_ms(path);
    return volume;
}

std::optional<int> ngram_rank(const ScannedModel& model, const ScannedNgramVolume& volume) {
    if (!model.needs_ngram_volume || !model.ngram_table) { return std::nullopt; }
    const NgramTableFacts& table = *model.ngram_table;
    if (volume.rows != table.rows) { return std::nullopt; }
    const std::uint32_t width = head_width(table);
    std::string format        = volume.format;
    if (volume.tag == 0) {
        // Untagged: the row size against this model's width decides.
        format = volume.row_bytes == row_bytes_for("bf16", width)     ? "bf16"
                 : volume.row_bytes == row_bytes_for("fp8", width)    ? "fp8"
                 : volume.row_bytes == row_bytes_for("iq4_nl", width) ? "iq4_nl"
                                                                      : "unknown";
    }
    if (format == "unknown" || volume.row_bytes != row_bytes_for(format, width)) { return std::nullopt; }
    if (volume.key == table.volume_id && format == table.format) { return 0; }
    if (format == "fp8" && volume.fp8_scale == 0 &&
        std::find(std::begin(kKnownFp8Volumes), std::end(kKnownFp8Volumes), volume.key) ==
            std::end(kKnownFp8Volumes)) {
        return std::nullopt; // no scale: only the artifact it was written for can read it
    }
    if (format == "fp8") { return 1; }
    if (format == "nvfp4") { return 2; }
    if (format == "iq4_nl") { return 3; }
    if (format == "q8_0") { return 4; }
    if (format == "bf16_p12") { return 5; }
    if (format == "bf16") { return 6; }
    return 6;
}

LibraryScan scan_library(const std::vector<std::filesystem::path>& dirs, int depth) {
    LibraryScan scan;
    scan.scanned_at = now_ms();
    std::set<std::string> visited_files;
    std::lock_guard cache_lock(g_cache_mutex);
    std::map<std::string, FileFacts> seen;

    for (const std::filesystem::path& root : dirs) {
        ScannedDir dir;
        dir.path = root;
        std::error_code error;
        dir.exists = std::filesystem::is_directory(root, error);
        // Read here, on the scan's thread, so no API call waits on the drive (a sleeping network share).
        std::error_code space_error;
        const std::filesystem::space_info space = std::filesystem::space(root, space_error);
        if (!space_error) {
            dir.free_bytes  = space.available;
            dir.total_bytes = space.capacity;
        }
        if (!dir.exists) {
            dir.error = error ? error_text(error) : "folder not found";
            scan.dirs.push_back(std::move(dir));
            continue;
        }
        const auto options = std::filesystem::directory_options::skip_permission_denied;
        std::filesystem::recursive_directory_iterator it(root, options, error);
        if (error) {
            dir.error = error_text(error);
            scan.dirs.push_back(std::move(dir));
            continue;
        }
        // A failed step ends the walk (MSVC's STL and libstdc++ leave the iterator at the end), so the
        // folder says it was scanned only in part instead of losing its later files silently.
        for (; it != std::filesystem::recursive_directory_iterator(); it.increment(error)) {
            if (error) { break; }
            const std::filesystem::directory_entry& entry = *it;
            std::error_code status_error;
            if (entry.is_directory(status_error)) {
                if (it.depth() >= depth || ablate_run_fit_folder(entry.path())) { it.disable_recursion_pending(); }
                continue;
            }
            if (!entry.is_regular_file(status_error)) { continue; }
            const std::filesystem::path& path = entry.path();
            if (iequals(path_utf8(path.filename()), kSidecarFile)) {
                const std::string id = path_utf8(std::filesystem::weakly_canonical(path, status_error));
                if (!visited_files.insert(id).second) { continue; }
                std::optional<std::string> issue;
                if (std::optional<ScannedSidecar> sidecar = read_sidecar(path, issue)) {
                    ++dir.sidecars;
                    scan.sidecars.push_back(std::move(*sidecar));
                }
                if (issue) { scan.issues.push_back({path, *issue}); }
                continue;
            }
            if (iequals(path_utf8(path.filename()), kAdapterConfig)) {
                const std::string id = path_utf8(std::filesystem::weakly_canonical(path, status_error));
                if (!visited_files.insert(id).second) { continue; }
                std::optional<std::string> issue;
                if (std::optional<ScannedAdapter> adapter = read_adapter(path, issue)) {
                    ++dir.adapters;
                    scan.adapters.push_back(std::move(*adapter));
                }
                if (issue) { scan.issues.push_back({path, *issue}); }
                continue;
            }
            if (!has_extension(path, ".infernix") && !has_extension(path, ".ninfer") &&
                !has_extension(path, ".ngram")) {
                continue;
            }
            const std::string id = path_utf8(std::filesystem::weakly_canonical(path, status_error));
            if (!visited_files.insert(id).second) { continue; }
            const std::uint64_t size    = entry.file_size(status_error);
            const std::int64_t modified = modified_ms(path);
            auto cached = g_cache.find(id);
            FileFacts facts = cached != g_cache.end() && cached->second.size == size &&
                                      cached->second.modified == modified
                                  ? cached->second
                                  : read_facts(path, size, modified);
            seen[id] = facts;
            if (facts.issue) { scan.issues.push_back({path, *facts.issue}); }
            if (facts.model) {
                ScannedModel model = *facts.model;
                model.path      = path;
                model.locations = {path};
                for (const auto& [part, expected] : facts.parts) {
                    if (!safe_relative(part)) {
                        // Never looked at, counted or deleted with the model's files.
                        model.missing_files.push_back(part + " (outside the artifact's folder)");
                        continue;
                    }
                    const std::filesystem::path part_path = path.parent_path() / utf8_path(part);
                    const std::uint64_t actual             = file_size_or_zero(part_path);
                    if (actual != expected) {
                        model.missing_files.push_back(part + (actual == 0 ? " (missing)" : " (incomplete)"));
                    }
                    if (actual != 0) { model.part_files.push_back(part_path); }
                }
                if (!model.missing_files.empty()) {
                    scan.issues.push_back({path, "missing or incomplete parts: " +
                                                     std::to_string(model.missing_files.size())});
                }
                ++dir.models;
                auto existing = std::find_if(scan.models.begin(), scan.models.end(),
                                             [&](const ScannedModel& m) { return m.key == model.key; });
                if (existing != scan.models.end()) {
                    existing->locations.push_back(path);
                    // Prefer a complete copy as the one to launch.
                    if (!existing->missing_files.empty() && model.missing_files.empty()) {
                        model.locations = existing->locations;
                        *existing       = std::move(model);
                    }
                } else {
                    scan.models.push_back(std::move(model));
                }
            }
            if (facts.drafter) {
                ++dir.drafters;
                if (!scan.drafter(facts.drafter->key)) {
                    ScannedDrafter drafter = *facts.drafter;
                    drafter.path           = path;
                    scan.drafters.push_back(std::move(drafter));
                }
            }
            if (facts.volume && facts.volume->bytes < facts.volume->expected_bytes) {
                scan.issues.push_back({path, "incomplete n-gram volume: " + format_bytes(facts.volume->bytes) +
                                                 " of " + format_bytes(facts.volume->expected_bytes)});
            } else if (facts.volume) {
                ++dir.ngram_volumes;
                const bool duplicate = std::any_of(scan.ngram_volumes.begin(), scan.ngram_volumes.end(),
                                                   [&](const ScannedNgramVolume& v) {
                                                       return v.key == facts.volume->key;
                                                   });
                if (!duplicate) {
                    ScannedNgramVolume volume = *facts.volume;
                    volume.path               = path;
                    scan.ngram_volumes.push_back(std::move(volume));
                }
            }
        }
        if (error) { dir.error = "scanned only in part: " + error_text(error); }
        scan.dirs.push_back(std::move(dir));
    }
    // Forget files no longer present.
    g_cache = std::move(seen);

    for (ScannedDrafter& drafter : scan.drafters) {
        for (ScannedModel& model : scan.models) {
            if (!drafter_fits(model, drafter)) { continue; }
            drafter.compatible_models.push_back(model.key);
            model.has_draft_companion = true;
        }
    }
    for (ScannedNgramVolume& volume : scan.ngram_volumes) {
        for (const ScannedModel& model : scan.models) {
            const std::optional<int> rank = ngram_rank(model, volume);
            if (!rank) { continue; }
            volume.compatible_models.push_back(model.key);
            if (*rank == 0) { volume.own_for.push_back(model.key); }
        }
    }
    for (ScannedAdapter& adapter : scan.adapters) {
        for (const ScannedModel& model : scan.models) {
            if (adapter_matches(model, adapter)) { adapter.compatible_models.push_back(model.key); }
        }
    }
    std::sort(scan.adapters.begin(), scan.adapters.end(), [](const ScannedAdapter& a, const ScannedAdapter& b) {
        return std::tie(a.name, a.key) < std::tie(b.name, b.key);
    });
    for (ScannedSidecar& sidecar : scan.sidecars) {
        for (const ScannedModel& model : scan.models) {
            if (!sidecar_fits(model, sidecar)) { continue; }
            sidecar.compatible_models.push_back(model.key);
            if (sidecar.artifact_id && *sidecar.artifact_id == model.key) { sidecar.fitted_models.push_back(model.key); }
        }
    }
    std::sort(scan.sidecars.begin(), scan.sidecars.end(), [](const ScannedSidecar& a, const ScannedSidecar& b) {
        return std::tie(a.name, a.key) < std::tie(b.name, b.key);
    });
    std::sort(scan.models.begin(), scan.models.end(), [](const ScannedModel& a, const ScannedModel& b) {
        return std::tie(a.family_label, a.name) < std::tie(b.family_label, b.name);
    });
    return scan;
}

Json to_json(const LibraryScan& scan) {
    Json dirs = Json::array();
    for (const ScannedDir& dir : scan.dirs) {
        dirs.push_back({{"path", path_utf8(dir.path)},
                        {"exists", dir.exists},
                        {"models", dir.models},
                        {"drafters", dir.drafters},
                        {"ngram_volumes", dir.ngram_volumes},
                        {"adapters", dir.adapters},
                        {"sidecars", dir.sidecars},
                        {"free_bytes", dir.free_bytes ? Json(*dir.free_bytes) : Json(nullptr)},
                        {"total_bytes", dir.total_bytes ? Json(*dir.total_bytes) : Json(nullptr)},
                        {"error", dir.error ? Json(*dir.error) : Json(nullptr)}});
    }
    const auto optional_path = [](const std::optional<std::filesystem::path>& path) {
        return path ? Json(path_utf8(*path)) : Json(nullptr);
    };
    Json models = Json::array();
    for (const ScannedModel& m : scan.models) {
        Json locations = Json::array();
        for (const auto& location : m.locations) { locations.push_back(path_utf8(location)); }
        models.push_back({
            {"key", m.key},
            {"name", m.name},
            {"file_name", m.file_name},
            {"path", path_utf8(m.path)},
            {"dir", path_utf8(m.path.parent_path())},
            {"locations", locations},
            {"format", m.format},
            {"family", m.family},
            {"family_label", m.family_label},
            {"model_type", m.model_type},
            {"recipe", m.recipe ? Json(*m.recipe) : Json(nullptr)},
            {"quantization", m.quantization ? Json(*m.quantization) : Json(nullptr)},
            {"bytes", m.bytes},
            {"file_count", m.file_count},
            {"missing_files", m.missing_files},
            {"components", m.components},
            {"has_vision", m.has_vision},
            {"has_mtp", m.has_mtp},
            {"has_dflash", m.has_dflash},
            {"has_dflash2", m.has_dflash2},
            {"has_dspark", m.has_dspark},
            {"has_proposal", m.has_proposal},
            {"option_support", option_support_json(m)},
            {"offloaded_experts", m.offloaded_experts},
            {"needs_ngram_volume", m.needs_ngram_volume},
            {"accepts_draft_model", m.accepts_draft_model},
            {"native_context", m.native_context},
            {"hidden_size", m.hidden_size},
            {"num_layers", m.num_layers},
            {"vocab_size", m.vocab_size},
            {"num_experts", m.num_experts ? Json(*m.num_experts) : Json(nullptr)},
            {"expert_state", optional_path(m.expert_state)},
            {"own_ngram_volume", optional_path(m.own_ngram_volume)},
            {"modified", m.modified},
        });
    }
    Json drafters = Json::array();
    for (const ScannedDrafter& d : scan.drafters) {
        drafters.push_back({
            {"key", d.key},
            {"name", d.name},
            {"file_name", d.file_name},
            {"path", path_utf8(d.path)},
            {"dir", path_utf8(d.path.parent_path())},
            {"bytes", d.bytes},
            {"kind", d.kind},
            {"target_architecture", d.target_architecture ? Json(*d.target_architecture) : Json(nullptr)},
            {"compatible_models", d.compatible_models},
            {"modified", d.modified},
        });
    }
    Json volumes = Json::array();
    for (const ScannedNgramVolume& v : scan.ngram_volumes) {
        volumes.push_back({
            {"key", v.key},
            {"name", v.name},
            {"file_name", v.file_name},
            {"path", path_utf8(v.path)},
            {"dir", path_utf8(v.path.parent_path())},
            {"bytes", v.bytes},
            {"format", v.format},
            {"rows", v.rows},
            {"compatible_models", v.compatible_models},
            {"own_for", v.own_for},
            {"modified", v.modified},
        });
    }
    Json adapters = Json::array();
    for (const ScannedAdapter& a : scan.adapters) {
        adapters.push_back({
            {"key", a.key},
            {"name", a.name},
            {"path", path_utf8(a.path)},
            {"dir", path_utf8(a.path.parent_path())},
            {"bytes", a.bytes},
            {"config", to_json(a.config)},
            {"compatible_models", a.compatible_models},
            {"modified", a.modified},
        });
    }
    const auto optional_text = [](const std::optional<std::string>& text) { return text ? Json(*text) : Json(nullptr); };
    Json sidecars = Json::array();
    for (const ScannedSidecar& c : scan.sidecars) {
        sidecars.push_back({
            {"key", c.key},
            {"name", c.name},
            {"path", path_utf8(c.path)},
            {"dir", path_utf8(c.path.parent_path())},
            {"bytes", c.bytes},
            {"direction_dim", c.direction_dim},
            {"rows", c.rows},
            {"attn_layers", c.attn_layers},
            {"mlp_layers", c.mlp_layers},
            {"artifact_id", optional_text(c.artifact_id)},
            {"fitted_artifact", optional_text(c.fitted_artifact)},
            {"source", optional_text(c.source)},
            {"compatible_models", c.compatible_models},
            {"fitted_models", c.fitted_models},
            {"modified", c.modified},
        });
    }
    Json issues = Json::array();
    for (const ScanIssue& issue : scan.issues) {
        issues.push_back({{"path", path_utf8(issue.path)}, {"message", issue.message}});
    }
    return {{"scanned_at", scan.scanned_at == 0 ? Json(nullptr) : Json(scan.scanned_at)},
            {"scanning", false},
            {"dirs", dirs},
            {"models", models},
            {"drafters", drafters},
            {"ngram_volumes", volumes},
            {"adapters", adapters},
            {"sidecars", sidecars},
            {"issues", issues}};
}

AdapterConfigFacts parse_adapter_config(const Json& config) {
    AdapterConfigFacts facts;
    if (!config.is_object()) { return facts; }
    const auto text = [&](const char* key) -> std::optional<std::string> {
        const auto it = config.find(key);
        return it != config.end() && it->is_string() ? std::optional<std::string>(it->get<std::string>()) : std::nullopt;
    };
    const auto count = [](const Json& value) -> std::optional<std::uint64_t> {
        return value.is_number_unsigned() || (value.is_number_integer() && value.get<std::int64_t>() > 0)
                   ? std::optional<std::uint64_t>(value.get<std::uint64_t>())
                   : std::nullopt;
    };
    facts.peft_type  = text("peft_type");
    facts.base_model = text("base_model_name_or_path");
    if (const auto it = config.find("r"); it != config.end()) { facts.rank = count(*it); }
    if (const auto it = config.find("lora_alpha"); it != config.end() && it->is_number()) {
        facts.alpha = it->get<double>();
    }
    facts.max_rank = facts.rank;
    if (const auto it = config.find("rank_pattern"); it != config.end() && it->is_object()) {
        for (const auto& [key, value] : it->items()) {
            if (const std::optional<std::uint64_t> rank = count(value)) {
                facts.max_rank = std::max(facts.max_rank.value_or(0), *rank);
            }
        }
    }
    if (const auto it = config.find("target_modules"); it != config.end() && it->is_array()) {
        for (const Json& module : *it) {
            if (module.is_string()) { facts.target_modules.push_back(module.get<std::string>()); }
        }
    }
    if (const auto it = config.find("use_rslora"); it != config.end() && it->is_boolean()) {
        facts.rslora = it->get<bool>();
    }
    return facts;
}

Json to_json(const AdapterConfigFacts& facts) {
    const auto optional = [](const auto& value) { return value ? Json(*value) : Json(nullptr); };
    return {{"peft_type", optional(facts.peft_type)},
            {"base_model", optional(facts.base_model)},
            {"rank", optional(facts.rank)},
            {"max_rank", optional(facts.max_rank)},
            {"alpha", optional(facts.alpha)},
            {"target_modules", facts.target_modules},
            {"rslora", facts.rslora}};
}

bool drafter_fits(const ScannedModel& model, const ScannedDrafter& drafter) {
    if (!model.accepts_draft_model) { return false; }
    // The Qwen3.8-27B bundles its DFlash2 drafter and takes a standalone DSpine one (--spec dspine); the other models
    // that take a drafter artifact take DFlash2 companions.
    const bool dspine_target = model.family == "qwen3.8-27b";
    const bool dspine        = drafter.kind == "dspine";
    const bool dflash2       = drafter.kind.find("dflash2") != std::string::npos;
    if (dspine_target ? !dspine : !dflash2) { return false; }
    if (drafter.target_architecture && *drafter.target_architecture != model.architecture) { return false; }
    if (drafter.target_hidden_size != 0 && drafter.target_hidden_size != model.hidden_size) { return false; }
    if (drafter.target_layers != 0 && drafter.target_layers != model.num_layers) { return false; }
    if (drafter.target_vocab != 0 && drafter.target_vocab != model.vocab_size) { return false; }
    return true;
}

bool adapter_matches(const ScannedModel& model, const ScannedAdapter& adapter) {
    if (model.family == "inkling2-mini") {
        if (!adapter.config.base_model) { return false; }
        // "thinkingmachines/Inkling-2-Mini" and its repackagings.
        const std::string base = lower(*adapter.config.base_model);
        return base.find("inkling-2-mini") != std::string::npos || base.find("inkling2-mini") != std::string::npos;
    }
    if (model.model_type != "qwen3_5_text" || model.num_experts || !adapter.config.base_model) { return false; }
    if (model.family != "qwen3.8-27b" && model.family != "qwen3.6-27b") { return false; }
    // "Qwen/Qwen3.8-27B", "unsloth/Qwen3.8-27B-bnb-4bit", ...: the family's name in the base model's.
    return lower(*adapter.config.base_model).find(model.family) != std::string::npos;
}

std::optional<ScannedSidecar> parse_sidecar(const std::filesystem::path& path, const Json& sidecar,
                                            std::optional<std::string>& issue) {
    if (!sidecar.is_object()) { return std::nullopt; }
    const auto format = sidecar.find("format");
    if (format == sidecar.end() || !format->is_string() || format->get<std::string>() != kSidecarFormat) {
        return std::nullopt;
    }
    const auto text = [](const Json& object, const char* key) -> std::optional<std::string> {
        if (!object.is_object()) { return std::nullopt; }
        const auto it = object.find(key);
        return it != object.end() && it->is_string() ? std::optional<std::string>(it->get<std::string>()) : std::nullopt;
    };
    ScannedSidecar result;
    result.path = path;
    result.key  = sha256_hex(path_utf8(path)).substr(0, 32);
    result.name = path_utf8(path.parent_path().filename());
    const auto dim = sidecar.find("direction_dim");
    if (dim == sidecar.end() || !dim->is_number_unsigned() || dim->get<std::uint64_t>() == 0) {
        issue = "ablation sidecar: direction_dim is not a positive integer";
        return std::nullopt;
    }
    result.direction_dim            = dim->get<std::uint64_t>();
    const std::optional<std::string> file = text(sidecar, "direction_file");
    if (!file || !safe_relative(*file)) {
        issue = "ablation sidecar: no direction_file beside it";
        return std::nullopt;
    }
    result.direction_file           = path.parent_path() / utf8_path(*file);
    result.bytes                    = file_size_or_zero(result.direction_file);
    const std::uint64_t row_bytes   = result.direction_dim * 4;
    const bool per_layer            = text(sidecar, "direction_layout").value_or("shared") == "per_layer";
    if (result.bytes == 0 || result.bytes % row_bytes != 0 || (!per_layer && result.bytes != row_bytes)) {
        issue = "ablation sidecar: " + *file + " is not " + (per_layer ? "whole rows" : "one row") + " of " +
                std::to_string(result.direction_dim) + " float32 values";
        return std::nullopt;
    }
    result.rows = result.bytes / row_bytes;
    // The lanes' tables: alpha_attn_per_layer and alpha_mlp_per_layer, or alpha_per_layer for both.
    const auto lane = [&](const char* key, std::uint64_t& cut) -> bool {
        const auto table = sidecar.find(key);
        if (table == sidecar.end()) { return true; }
        if (!table->is_object()) { return false; }
        for (const auto& [layer, alpha] : table->items()) {
            if (!alpha.is_number() || layer.empty() ||
                !std::all_of(layer.begin(), layer.end(), [](char c) { return c >= '0' && c <= '9'; }) ||
                layer.size() > 6) {
                return false;
            }
            result.layer_bound = std::max<std::uint64_t>(result.layer_bound, std::stoull(layer) + 1);
            if (alpha.get<double>() > 0.0) { ++cut; }
        }
        return true;
    };
    std::uint64_t shared_cut = 0;
    if (!lane("alpha_attn_per_layer", result.attn_layers) || !lane("alpha_mlp_per_layer", result.mlp_layers) ||
        !lane("alpha_per_layer", shared_cut)) {
        issue = "ablation sidecar: an alpha table is not layer indices with numbers";
        return std::nullopt;
    }
    result.attn_layers += shared_cut;
    result.mlp_layers += shared_cut;
    if (result.attn_layers == 0 && result.mlp_layers == 0) { return std::nullopt; } // a base direction: no cut
    result.artifact_id = text(sidecar, "artifact_id");
    const auto generation  = sidecar.find("generation");
    if (generation != sidecar.end()) {
        result.source = text(*generation, "source");
        if (const auto fitted = generation->find("fitted_under"); generation->is_object() && fitted != generation->end()) {
            result.fitted_artifact = text(*fitted, "artifact");
        }
    }
    result.modified = modified_ms(path);
    return result;
}

bool sidecar_fits(const ScannedModel& model, const ScannedSidecar& sidecar) {
    return model.hidden_size != 0 && sidecar.direction_dim == model.hidden_size &&
           (sidecar.rows == 1 || sidecar.rows == model.num_layers) && sidecar.layer_bound <= model.num_layers;
}

} // namespace infernix::launcher
