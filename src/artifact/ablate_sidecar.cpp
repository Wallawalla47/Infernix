#include "artifact/ablate_sidecar.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>

namespace infernix::artifact {
namespace {

[[noreturn]] void fail(const std::string& what) { throw std::runtime_error("ablate-sidecar: " + what); }

constexpr std::string_view kFormat = "ablate-direction-v2";
constexpr double kAlphaMax         = 4.0; // beyond this the cut is a projection removal, not an ablation
constexpr double kMinNorm          = 0.5; // a direction this far from unit is a broken file, not a weak cut

constexpr std::array<std::string_view, 11> kKeys{
    "format",          "direction_dtype",      "direction_dim",       "direction_layout",
    "direction_file",  "default_alpha",        "alpha_per_layer",     "alpha_attn_per_layer",
    "alpha_mlp_per_layer", "generation",       "artifact_id"};

const std::string& required_string(const nlohmann::json& cfg, const char* key) {
    if (!cfg.contains(key) || !cfg.at(key).is_string() || cfg.at(key).get_ref<const std::string&>().empty()) {
        fail(std::string(key) + " must be a non-empty string");
    }
    return cfg.at(key).get_ref<const std::string&>();
}

// One alpha table: {"3": 1.9, "4": 0.0} into a dense per-layer vector. Keys must be bare integers
// inside the layer range, values finite inside [0, kAlphaMax]; anything else fails the boot.
void read_table(const nlohmann::json& table, const std::string& what, const std::uint32_t n_layers,
                std::vector<float>& out) {
    if (!table.is_object()) { fail(what + " must be an object of layer -> alpha"); }
    for (const auto& entry : table.items()) {
        std::size_t consumed = 0;
        unsigned long layer  = 0;
        try {
            layer = std::stoul(entry.key(), &consumed);
        } catch (...) {
            consumed = 0;
        }
        if (consumed != entry.key().size() || layer >= n_layers) {
            fail(what + ": key outside 0.." + std::to_string(n_layers - 1) + ": '" + entry.key() + "'");
        }
        if (!entry.value().is_number()) { fail(what + ": layer " + entry.key() + " is not a number"); }
        const double alpha = entry.value().get<double>();
        if (!std::isfinite(alpha) || alpha < 0.0 || alpha > kAlphaMax) {
            fail(what + ": alpha outside [0, " + std::to_string(kAlphaMax) + "] at layer " + entry.key());
        }
        out[layer] = static_cast<float>(alpha);
    }
}

// Reads the direction bytes and normalizes them row by row. A shared direction is one row; a
// per-layer stack is n_layers rows of hidden_size floats each. `required[r]` says whether row r is
// read by a cut: a required row that is non-finite or far from unit fails the boot; an unrequired one is
// zeroed and carried, so a short stack still loads for the layers that do use it.
void read_direction(const std::filesystem::path& path, const std::uint32_t rows, const std::uint32_t hidden_size,
                    const std::vector<bool>& required, std::vector<float>& d) {
    std::ifstream bin(path, std::ios::binary);
    if (!bin) { fail("direction file not found: " + path.string()); }
    std::vector<char> bytes((std::istreambuf_iterator<char>(bin)), std::istreambuf_iterator<char>());
    const std::size_t want = static_cast<std::size_t>(rows) * hidden_size * sizeof(float);
    if (bytes.size() != want) {
        fail(path.filename().string() + " is " + std::to_string(bytes.size()) + " B, expected " +
             std::to_string(want) + " B for " + std::to_string(rows) + " row(s) of " +
             std::to_string(hidden_size) + " float32");
    }
    d.resize(want / sizeof(float));
    std::memcpy(d.data(), bytes.data(), bytes.size());

    for (std::uint32_t row = 0; row < rows; ++row) {
        const bool required_row = row < required.size() && required[row];
        float* v                = d.data() + static_cast<std::size_t>(row) * hidden_size;
        double norm             = 0.0;
        bool finite             = true;
        for (std::uint32_t h = 0; h < hidden_size; ++h) {
            if (!std::isfinite(v[h])) {
                finite = false;
                break;
            }
            norm += static_cast<double>(v[h]) * v[h];
        }
        norm = std::sqrt(norm);
        if (!finite) {
            if (required_row) { fail("non-finite direction value in row " + std::to_string(row)); }
            std::memset(v, 0, hidden_size * sizeof(float));
            continue;
        }
        if (!(norm > kMinNorm)) {
            // An uncut layer may hold a placeholder row; a cut layer may not. That is the difference
            // between a stack the fitter left empty and a stack that silently stops ablating.
            if (required_row) {
                fail("row " + std::to_string(row) + " is degenerate (norm " + std::to_string(norm) +
                     ") but this layer is cut");
            }
            std::memset(v, 0, hidden_size * sizeof(float));
            continue;
        }
        for (std::uint32_t h = 0; h < hidden_size; ++h) {
            v[h] = static_cast<float>(static_cast<double>(v[h]) / norm);
        }
    }
}

} // namespace

AblateSidecar AblateSidecar::load(const std::filesystem::path& json_path, const std::uint32_t n_layers,
                                  const std::uint32_t hidden_size) {
    std::ifstream in(json_path);
    if (!in) { fail("sidecar not found: " + json_path.string()); }
    nlohmann::json cfg;
    try {
        cfg = nlohmann::json::parse(in);
    } catch (const std::exception& error) {
        fail("bad JSON in " + json_path.string() + ": " + error.what());
    }
    if (!cfg.is_object()) { fail(json_path.string() + " must hold a JSON object"); }
    for (const auto& entry : cfg.items()) {
        if (std::find(kKeys.begin(), kKeys.end(), entry.key()) == kKeys.end()) {
            fail("unknown key '" + entry.key() + "' in " + json_path.string());
        }
    }

    const std::string format = cfg.contains("format") && cfg.at("format").is_string()
                                   ? cfg.at("format").get<std::string>()
                                   : std::string();
    if (format != kFormat) {
        fail("unsupported format '" + (format.empty() ? std::string("<missing>") : format) + "': expected " +
             std::string(kFormat));
    }
    if (required_string(cfg, "direction_dtype") != "float32") { fail("direction_dtype must be float32"); }
    if (!cfg.contains("direction_dim") || !cfg.at("direction_dim").is_number_integer() ||
        cfg.at("direction_dim").get<std::int64_t>() != static_cast<std::int64_t>(hidden_size)) {
        fail("direction_dim must be the model's write width " + std::to_string(hidden_size));
    }
    // Layers outside every table stay cut-free. A sidecar that cuts everything by default would turn
    // a typo in the table into a model-wide edit, and the fit could not tell the two apart.
    if (cfg.contains("default_alpha") &&
        (!cfg.at("default_alpha").is_number() || cfg.at("default_alpha").get<double>() != 0.0)) {
        fail("default_alpha must be 0");
    }

    AblateSidecar sidecar;
    sidecar.n_layers    = n_layers;
    sidecar.hidden_size = hidden_size;
    for (auto& table : sidecar.alpha) { table.assign(n_layers, 0.0F); }

    // --- direction: one shared vector, or a per-layer stack -------------------------------------
    std::string layout = "shared";
    if (cfg.contains("direction_layout")) { layout = required_string(cfg, "direction_layout"); }
    if (layout != "shared" && layout != "per_layer") {
        fail("direction_layout must be 'shared' or 'per_layer', got '" + layout + "'");
    }
    sidecar.per_layer_direction = (layout == "per_layer");
    const std::uint32_t rows    = sidecar.per_layer_direction ? n_layers : 1U;

    // --- alphas --------------------------------------------------------------------------------
    const bool has_shared = cfg.contains("alpha_per_layer");
    const bool has_attn   = cfg.contains("alpha_attn_per_layer");
    const bool has_mlp    = cfg.contains("alpha_mlp_per_layer");
    if (has_shared && (has_attn || has_mlp)) {
        fail("name a layer once: give alpha_per_layer alone, or the per-lane tables");
    }
    if (has_shared) {
        read_table(cfg.at("alpha_per_layer"), "alpha_per_layer", n_layers,
                   sidecar.alpha[static_cast<std::size_t>(Lane::Attn)]);
        sidecar.alpha[static_cast<std::size_t>(Lane::Mlp)] = sidecar.alpha[static_cast<std::size_t>(Lane::Attn)];
    }
    if (has_attn) {
        read_table(cfg.at("alpha_attn_per_layer"), "alpha_attn_per_layer", n_layers,
                   sidecar.alpha[static_cast<std::size_t>(Lane::Attn)]);
    }
    if (has_mlp) {
        read_table(cfg.at("alpha_mlp_per_layer"), "alpha_mlp_per_layer", n_layers,
                   sidecar.alpha[static_cast<std::size_t>(Lane::Mlp)]);
    }

    // --- provenance: the engine logs it and does not interpret it, and still requires it ---------
    if (!cfg.contains("generation") || !cfg.at("generation").is_object()) {
        fail("a generation block must name where the direction came from (generation.source)");
    }
    const nlohmann::json& gen = cfg.at("generation");
    if (!gen.contains("source") || !gen.at("source").is_string() ||
        gen.at("source").get_ref<const std::string&>().empty()) {
        fail("generation.source must say what produced the direction (e.g. the seed corpus and the extraction "
             "that cut it)");
    }
    for (const char* key : {"seed", "notes"}) {
        if (gen.contains(key) && !gen.at(key).is_string()) { fail(std::string("generation.") + key + " must be a string"); }
    }
    std::string line = "source=" + gen.at("source").get<std::string>();
    if (gen.contains("seed")) { line += ", seed=" + gen.at("seed").get<std::string>(); }
    if (gen.contains("notes")) { line += ", " + gen.at("notes").get<std::string>(); }
    sidecar.generation = std::move(line);
    if (cfg.contains("artifact_id")) { sidecar.artifact_id = required_string(cfg, "artifact_id"); }

    // A row a cut actually reads must be a real direction; a row on an uncut layer is a placeholder
    // the fitter may leave empty. One shared row serves every layer, so it is required whenever any
    // layer is cut: checking it against layer 0 alone would let a broken file zero out and serve uncut.
    std::vector<bool> required(rows, false);
    for (std::uint32_t l = 0; l < n_layers; ++l) {
        if (sidecar.layer_cut(l)) { required[sidecar.per_layer_direction ? l : 0U] = true; }
    }
    // The direction file sits beside sidecar.json (docs/ablate.md): a file name, not a path elsewhere.
    const std::filesystem::path direction_file(required_string(cfg, "direction_file"));
    if (direction_file.empty() || direction_file.has_parent_path() || direction_file.has_root_path() ||
        direction_file != direction_file.filename() || direction_file == "." || direction_file == "..") {
        fail("direction_file must name a file beside sidecar.json (a file name, not a path)");
    }
    read_direction(json_path.parent_path() / direction_file, rows, hidden_size, required, sidecar.d_host);
    return sidecar;
}

std::filesystem::path AblateSidecar::direction_path(const std::filesystem::path& json_path) noexcept {
    try {
        std::ifstream in(json_path);
        if (!in) { return {}; }
        const nlohmann::json cfg = nlohmann::json::parse(in);
        const std::filesystem::path name(cfg.value("direction_file", std::string()));
        // Only a file beside sidecar.json, as the loader accepts (a path elsewhere is refused there).
        return name.empty() || name != name.filename() ? std::filesystem::path() : json_path.parent_path() / name;
    } catch (...) {
        return {};
    }
}

std::uint32_t AblateSidecar::layers_cut(const Lane lane) const noexcept {
    std::uint32_t n = 0;
    for (std::uint32_t l = 0; l < n_layers; ++l) { n += alpha_for(l, lane) != 0.0F ? 1U : 0U; }
    return n;
}

} // namespace infernix::artifact
