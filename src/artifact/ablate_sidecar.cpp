#include "artifact/ablate_sidecar.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>

namespace infernix::artifact {
namespace {

[[noreturn]] void fail(const std::string& what) { throw std::runtime_error("ablate-sidecar: " + what); }

constexpr double kAlphaMax = 4.0;      // beyond this the cut is a projection removal, not an ablation
constexpr double kMinNorm   = 0.5;     // a direction this far from unit is a broken file, not a weak cut

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
// per-layer stack is n_layers rows of hidden_size floats each. `required[r]` says whether layer r is
// cut: a required row that is non-finite or far from unit fails the boot; an unrequired one is
// zeroed and carried, so a short stack still loads for the layers that do use it.
void read_direction(const std::filesystem::path& json_path, const std::string& file_name,
                    const std::uint32_t rows, const std::uint32_t hidden_size,
                    const std::vector<float>& required, std::vector<float>& d) {
    const auto path = json_path.parent_path() / file_name;
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
        const bool required_row = row < static_cast<std::uint32_t>(required.size()) && required[row] != 0.0F;
        float*        v    = d.data() + static_cast<std::size_t>(row) * hidden_size;
        double        norm = 0.0;
        bool          finite = true;
        for (std::uint32_t h = 0; h < hidden_size; ++h) {
            if (!std::isfinite(v[h])) { finite = false; break; }
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

}  // namespace

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

    const std::string format = cfg.value("format", std::string());
    int version              = 0;
    if (format == "ablate-direction-v1") { version = 1; }
    else if (format == "ablate-direction-v2") { version = 2; }
    else {
        fail("unsupported format '" + (format.empty() ? std::string("<missing>") : format) +
             "': expected ablate-direction-v1 or ablate-direction-v2");
    }

    if (cfg.value("direction_dtype", std::string()) != "float32") { fail("direction_dtype must be float32"); }
    if (cfg.value("direction_dim", 0.0) != static_cast<double>(hidden_size)) {
        fail("direction_dim " + std::to_string(cfg.value("direction_dim", 0.0)) +
             " != model write width " + std::to_string(hidden_size));
    }
    // Layers outside every table stay cut-free. A sidecar that cuts everything by default would turn
    // a typo in the table into a model-wide edit, and the fit could not tell the two apart.
    if (cfg.value("default_alpha", 0.0) != 0.0) {
        fail("format v" + std::to_string(version) + " requires default_alpha 0");
    }

    AblateSidecar sidecar;
    sidecar.format_version = version;
    sidecar.n_layers       = n_layers;
    sidecar.hidden_size    = hidden_size;
    for (auto& table : sidecar.alpha) { table.assign(n_layers, 0.0F); }

    // --- direction: one shared vector, or a per-layer stack (v2 only) ---------------------------
    const std::string layout = cfg.value("direction_layout", std::string("shared"));
    if (version == 1 && !layout.empty() && layout != "shared") {
        fail("v1 has no per-layer direction stack (use v2 for direction_layout per_layer)");
    }
    if (layout != "shared" && layout != "per_layer") {
        fail("direction_layout must be 'shared' or 'per_layer', got '" + layout + "'");
    }
    sidecar.per_layer_direction = (layout == "per_layer");
    const std::uint32_t rows     = sidecar.per_layer_direction ? n_layers : 1U;

    // --- alphas --------------------------------------------------------------------------------
    const bool has_shared = cfg.contains("alpha_per_layer");
    const bool has_attn   = cfg.contains("alpha_attn_per_layer");
    const bool has_mlp    = cfg.contains("alpha_mlp_per_layer");
    if (version == 1) {
        if (has_attn || has_mlp) { fail("v1 does not carry per-lane alpha tables (use v2)"); }
        if (has_shared) {
            // v1 named one alpha for the layer, and the hook applied it at both lanes. Preserved
            // exactly: a v1 sidecar must score the same on a v2 engine as it did on the one it fit on.
            read_table(cfg.at("alpha_per_layer"), "alpha_per_layer", n_layers,
                       sidecar.alpha[static_cast<std::size_t>(Lane::Attn)]);
            sidecar.alpha[static_cast<std::size_t>(Lane::Mlp)] =
                sidecar.alpha[static_cast<std::size_t>(Lane::Attn)];
        }
    } else {
        if (has_shared && (has_attn || has_mlp)) {
            fail("v2 must name a layer once: give alpha_per_layer alone, or the per-lane tables");
        }
        if (has_shared) {
            read_table(cfg.at("alpha_per_layer"), "alpha_per_layer", n_layers,
                       sidecar.alpha[static_cast<std::size_t>(Lane::Attn)]);
            sidecar.alpha[static_cast<std::size_t>(Lane::Mlp)] =
                sidecar.alpha[static_cast<std::size_t>(Lane::Attn)];
        }
        if (has_attn) {
            read_table(cfg.at("alpha_attn_per_layer"), "alpha_attn_per_layer", n_layers,
                       sidecar.alpha[static_cast<std::size_t>(Lane::Attn)]);
        }
        if (has_mlp) {
            read_table(cfg.at("alpha_mlp_per_layer"), "alpha_mlp_per_layer", n_layers,
                       sidecar.alpha[static_cast<std::size_t>(Lane::Mlp)]);
        }
    }

    // --- provenance (v2): the engine ignores this, and still requires it ------------------------
    if (version >= 2) {
        if (!cfg.contains("generation") || !cfg.at("generation").is_object()) {
            fail("v2 requires a generation block naming where the direction came from "
                 "(generation.source)");
        }
        const auto& gen  = cfg.at("generation");
        const auto  src  = gen.value("source", std::string());
        if (src.empty()) {
            fail("v2 generation.source must say what produced the direction (e.g. the seed corpus "
                 "and the extraction that cut it)");
        }
        std::string line = "source=" + src;
        if (gen.contains("seed") && gen.at("seed").is_string()) { line += ", seed=" + gen.at("seed").get<std::string>(); }
        if (gen.contains("notes") && gen.at("notes").is_string()) { line += ", " + gen.at("notes").get<std::string>(); }
        sidecar.generation = std::move(line);
    }

    // A row a cut actually reads must be a real direction; a row on an uncut layer is a placeholder
    // the fitter may leave empty. That is the difference between a half-filled stack and a sidecar
    // that silently stops ablating on the layers it was fitted for.
    std::vector<float> require(n_layers, 0.0F);
    bool any_cut = false;
    for (std::uint32_t l = 0; l < n_layers; ++l) {
        require[l] = sidecar.layer_cut(l) ? 1.0F : 0.0F;
        any_cut    = any_cut || require[l] != 0.0F;
    }
    if (!sidecar.per_layer_direction) {
        // One shared row serves every layer, so "this row's layer" is "any layer": a degenerate shared
        // direction must fail even when layer 0 happens to be uncut. Checking row 0 against layer 0
        // would let a broken file zero out and serve uncut, which is the silent no-op this whole
        // loader exists to refuse.
        require.assign(1, any_cut ? 1.0F : 0.0F);
    }
    read_direction(json_path, cfg.at("direction_file").get<std::string>(), rows, hidden_size, require,
                   sidecar.d_host);

    return sidecar;
}

std::filesystem::path AblateSidecar::direction_path(const std::filesystem::path& json_path) noexcept {
    try {
        std::ifstream in(json_path);
        if (!in) { return {}; }
        const nlohmann::json cfg = nlohmann::json::parse(in);
        const std::string name   = cfg.value("direction_file", std::string());
        return name.empty() ? std::filesystem::path() : json_path.parent_path() / name;
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
