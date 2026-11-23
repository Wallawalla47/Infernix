// The ablation sidecar reader (artifact::AblateSidecar, docs/ablate.md), host only: the format, the two lane
// tables, the per-layer direction stack, the provenance requirement, and every way a sidecar can be wrong.
// A model's device copy (qwen4_exp's binding at load) is not exercised here; the engine boot covers it.
// Every case prints a verdict line.

#include "artifact/ablate_sidecar.h"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

using infernix::artifact::AblateSidecar;

constexpr std::uint32_t kLayers = 4;
constexpr std::uint32_t kH      = 8;

int failures = 0;
int checks   = 0;

void check(const bool ok, const std::string& what) {
    ++checks;
    if (!ok) { ++failures; }
    std::printf("  %-4s %s\n", ok ? "ok  " : "FAIL", what.c_str());
}

// Writes <dir>/d<rows>.f32.bin: row r is the unit vector pointing mostly along axis (r % H).
std::string write_direction(const std::filesystem::path& dir, const std::string& name,
                            const std::uint32_t rows, const bool degenerate_row0 = false) {
    std::vector<float> v(static_cast<std::size_t>(rows) * kH, 0.0F);
    for (std::uint32_t r = 0; r < rows; ++r) {
        float* row = v.data() + static_cast<std::size_t>(r) * kH;
        if (degenerate_row0 && r == 0) { continue; }  // an all-zero row: only legal on an uncut layer
        row[r % kH] = 1.0F;
        row[(r + 1) % kH] = 0.5F;                      // not unit yet; the loader must normalize it
    }
    const auto path = dir / name;
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(float)));
    return name;
}

void write_json(const std::filesystem::path& dir, const std::string& body) {
    std::ofstream out(dir / "sidecar.json");
    out << body;
}

// Returns the loader's exception text, or "" when it accepted the sidecar.
std::string try_load(const std::filesystem::path& dir, AblateSidecar& out) {
    try {
        out = AblateSidecar::load(dir / "sidecar.json", kLayers, kH);
        return {};
    } catch (const std::exception& error) {
        return error.what();
    }
}

bool threw(const std::string& message, const std::string& want, const std::string& label) {
    const bool ok = message.find(want) != std::string::npos;
    check(ok, label + (ok ? "" : "  [got: " + message + "]"));
    if (!ok && message.empty()) { check(false, label + " (it loaded anyway)"); }
    return ok;
}

struct Scratch {
    std::filesystem::path dir;
    explicit Scratch(const std::string& tag) : dir(std::filesystem::temp_directory_path() / ("ablate-v2-" + tag)) {
        std::filesystem::remove_all(dir);
        std::filesystem::create_directories(dir);
    }
    ~Scratch() { std::filesystem::remove_all(dir); }
};

}  // namespace

int main() {
    AblateSidecar s;

    // --- one table for both lanes --------------------------------------------------------------
    std::printf("alpha_per_layer\n");
    {
        Scratch t("both");
        const std::string d = write_direction(t.dir, "d.f32.bin", 1);
        write_json(t.dir, R"({"format":"ablate-direction-v2","direction_file":")" + d +
                            R"(","direction_dtype":"float32","direction_dim":8,"default_alpha":0,
                               "generation":{"source":"test","notes":"one table"},
                               "alpha_per_layer":{"1":1.955,"2":1.955}})");
        const std::string err = try_load(t.dir, s);
        check(err.empty(), "a single table loads" + (err.empty() ? "" : "  [got: " + err + "]"));
        check(!s.per_layer_direction, "the direction is shared by default");
        check(s.alpha_for(1, AblateSidecar::Lane::Attn) == 1.955F, "it cuts the attention lane");
        check(s.alpha_for(1, AblateSidecar::Lane::Mlp) == 1.955F, "and the MLP lane with the same alpha");
        check(s.alpha_for(0, AblateSidecar::Lane::Attn) == 0.0F, "an unlisted layer stays cut-free");
        check(s.artifact_id.empty(), "a sidecar naming no artifact has an empty artifact_id");
        check(s.generation == "source=test, one table", "the provenance line names source and notes");
        double norm = 0.0;
        for (const float v : s.d_host) { norm += static_cast<double>(v) * v; }
        check(std::fabs(norm - 1.0) < 1e-6, "the direction is normalized on load");
    }
    {
        Scratch t("artifact");
        const std::string d = write_direction(t.dir, "d.f32.bin", 1);
        write_json(t.dir, R"({"format":"ablate-direction-v2","direction_file":")" + d +
                            R"(","direction_dtype":"float32","direction_dim":8,"artifact_id":"abc123",
                               "generation":{"source":"test"},"alpha_mlp_per_layer":{"3":1.0}})");
        const std::string err = try_load(t.dir, s);
        check(err.empty() && s.artifact_id == "abc123",
              "artifact_id names the artifact the direction was fitted on" + (err.empty() ? "" : "  [got: " + err + "]"));
    }

    // --- v2: the two lanes decide independently -----------------------------------------------
    std::printf("\nv2 per-lane alphas\n");
    {
        Scratch t("lanes");
        const std::string d = write_direction(t.dir, "d.f32.bin", 1);
        write_json(t.dir, R"({"format":"ablate-direction-v2","direction_file":")" + d +
                            R"(","direction_dtype":"float32","direction_dim":8,"default_alpha":0,
                               "generation":{"source":"test: refuse/comply pairs, own model"},
                               "alpha_attn_per_layer":{"1":2.0,"2":2.0,"3":0.0},
                               "alpha_mlp_per_layer":{"2":1.0}})");
        const std::string err = try_load(t.dir, s);
        check(err.empty(), "v2 loads" + (err.empty() ? "" : "  [got: " + err + "]"));
        check(s.alpha_for(1, AblateSidecar::Lane::Attn) == 2.0F, "layer 1: attention cut");
        check(s.alpha_for(1, AblateSidecar::Lane::Mlp) == 0.0F, "layer 1: MLP untouched  <- the whole point of v2");
        check(s.alpha_for(2, AblateSidecar::Lane::Attn) == 2.0F, "layer 2: attention cut");
        check(s.alpha_for(2, AblateSidecar::Lane::Mlp) == 1.0F, "layer 2: both lanes, different strengths");
        check(!s.layer_cut(0), "layer 0: not cut at all");
        check(s.layer_cut(1), "layer 1: cut on one lane is enough to count");
        check(!s.layer_cut(3), "an explicit 0.0 is not a cut, so the lane stays fused");
    }

    // --- v2: a direction stack, one row per layer ---------------------------------------------
    std::printf("\nv2 per-layer direction stack\n");
    {
        Scratch t("stack");
        const std::string d = write_direction(t.dir, "stack.f32.bin", kLayers);
        write_json(t.dir, R"({"format":"ablate-direction-v2","direction_file":")" + d +
                            R"(","direction_dtype":"float32","direction_dim":8,"default_alpha":0,
                               "direction_layout":"per_layer",
                               "generation":{"source":"test: per-layer extraction"},
                               "alpha_attn_per_layer":{"0":1.0}})");
        const std::string err = try_load(t.dir, s);
        check(err.empty(), "stack loads" + (err.empty() ? "" : "  [got: " + err + "]"));
        check(s.per_layer_direction, "stack flagged per-layer");
        check(s.d_host.size() == kLayers * kH, "stack holds one row per layer");
        bool each_unit = true;
        for (std::uint32_t r = 0; r < kLayers; ++r) {
            double n = 0.0;
            for (std::uint32_t h = 0; h < kH; ++h) {
                const float v = s.d_host[static_cast<std::size_t>(r) * kH + h];
                n += static_cast<double>(v) * v;
            }
            if (std::fabs(n - 1.0) > 1e-6) { each_unit = false; }
        }
        check(each_unit, "every row normalized on its own");
        check(s.host_d(2) == s.host_d(0) + 2 * kH, "row 2 sits two rows in (host view of the same walk)");
        // Row r points mostly along axis r, so column 0 belongs to row 0 only: proof the rows are
        // separate vectors, not one vector read at an offset.
        // Row r is (axis r = 1.0, axis r+1 = 0.5), so after normalization its leading component is
        // 1/sqrt(1.25) = 0.894 on its own axis and 0 on the next row's. A shifted single vector would
        // put row 1's mass on axis 1 and row 0's column 1 would be 0.447, not 0.894.
        const float lead = 1.0F / std::sqrt(1.25F);
        check(std::fabs(s.host_d(0)[0] - lead) < 1e-6F && std::fabs(s.host_d(0)[1] - 0.5F * lead) < 1e-6F,
              "row 0 normalized to its own axis pair");
        check(s.host_d(1)[0] == 0.0F && std::fabs(s.host_d(1)[1] - lead) < 1e-6F,
              "row 1 points on its own axes, so the stack is rows and not one shifted vector");
    }

    // --- the ways it must refuse -------------------------------------------------------------
    std::printf("\nfail-loud contract\n");
    {
        Scratch t("stack-zero");
        const std::string d = write_direction(t.dir, "stack.f32.bin", kLayers, /*degenerate_row0=*/true);
        write_json(t.dir, R"({"format":"ablate-direction-v2","direction_file":")" + d +
                            R"(","direction_dtype":"float32","direction_dim":8,"default_alpha":0,
                               "direction_layout":"per_layer","generation":{"source":"test"},
                               "alpha_attn_per_layer":{"0":1.0}})");
        threw(try_load(t.dir, s), "degenerate", "a cut layer whose row is zero fails the boot");
    }
    {
        // The trap in a shared direction: row 0 is the only row, and "this layer is uncut" means
        // nothing when one vector serves all four. A zeroed file must still fail.
        Scratch t("shared-zero");
        std::vector<float> z(static_cast<std::size_t>(kH), 0.0F);
        { std::ofstream out(t.dir / "d.f32.bin", std::ios::binary);
          out.write(reinterpret_cast<const char*>(z.data()), static_cast<std::streamsize>(z.size() * sizeof(float))); }
        write_json(t.dir, R"({"format":"ablate-direction-v2","direction_file":"d.f32.bin",
                              "direction_dtype":"float32","direction_dim":8,"default_alpha":0,
                              "generation":{"source":"test"},"alpha_attn_per_layer":{"2":1.0}})");
        threw(try_load(t.dir, s), "degenerate",
              "a degenerate SHARED direction fails even when layer 0 is uncut");
    }
    {
        Scratch t("stack-unused-zero");
        const std::string d = write_direction(t.dir, "stack.f32.bin", kLayers, /*degenerate_row0=*/true);
        write_json(t.dir, R"({"format":"ablate-direction-v2","direction_file":")" + d +
                            R"(","direction_dtype":"float32","direction_dim":8,"default_alpha":0,
                               "direction_layout":"per_layer","generation":{"source":"test"},
                               "alpha_attn_per_layer":{"1":1.0}})");
        const std::string err = try_load(t.dir, s);
        check(err.empty(), "an uncut layer may carry a placeholder row" + (err.empty() ? "" : "  [got: " + err + "]"));
    }
    {
        Scratch t("provenance");
        const std::string d = write_direction(t.dir, "d.f32.bin", 1);
        write_json(t.dir, R"({"format":"ablate-direction-v2","direction_file":")" + d +
                            R"(","direction_dtype":"float32","direction_dim":8,"default_alpha":0,
                               "alpha_attn_per_layer":{"1":1.0}})");
        threw(try_load(t.dir, s), "generation", "a sidecar with no provenance is refused");
    }
    {
        Scratch t("empty-source");
        const std::string d = write_direction(t.dir, "d.f32.bin", 1);
        write_json(t.dir, R"({"format":"ablate-direction-v2","direction_file":")" + d +
                            R"(","direction_dtype":"float32","direction_dim":8,"default_alpha":0,
                               "generation":{"notes":"no source named"},
                               "alpha_attn_per_layer":{"1":1.0}})");
        threw(try_load(t.dir, s), "source", "an empty generation.source is refused");
    }
    {
        Scratch t("double");
        const std::string d = write_direction(t.dir, "d.f32.bin", 1);
        write_json(t.dir, R"({"format":"ablate-direction-v2","direction_file":")" + d +
                            R"(","direction_dtype":"float32","direction_dim":8,"default_alpha":0,
                               "generation":{"source":"test"},"alpha_per_layer":{"1":1.0},
                               "alpha_attn_per_layer":{"1":3.0}})");
        threw(try_load(t.dir, s), "name a layer once", "a layer named in two tables is refused, not merged");
    }
    for (const char* format : {"ablate-direction-v1", "ablate-direction-v3"}) {
        Scratch t("format");
        const std::string d = write_direction(t.dir, "d.f32.bin", 1);
        write_json(t.dir, std::string(R"({"format":")") + format + R"(","direction_file":")" + d +
                            R"(","direction_dtype":"float32","direction_dim":8,"generation":{"source":"test"},
                               "alpha_per_layer":{"1":1.0}})");
        threw(try_load(t.dir, s), "unsupported format",
              std::string("another format (") + format + ") is refused, never guessed at");
    }
    {
        // A misspelt lane table would otherwise load as a sidecar that leaves that lane uncut.
        Scratch t("unknown-key");
        const std::string d = write_direction(t.dir, "d.f32.bin", 1);
        write_json(t.dir, R"({"format":"ablate-direction-v2","direction_file":")" + d +
                            R"(","direction_dtype":"float32","direction_dim":8,"generation":{"source":"test"},
                               "alpha_attn_per_layer":{"1":1.0},"alpha_mpl_per_layer":{"1":1.0}})");
        threw(try_load(t.dir, s), "unknown key 'alpha_mpl_per_layer'", "an unknown key is refused by name");
    }
    {
        Scratch t("no-file");
        write_json(t.dir, R"({"format":"ablate-direction-v2","direction_dtype":"float32","direction_dim":8,
                              "generation":{"source":"test"},"alpha_attn_per_layer":{"1":1.0}})");
        threw(try_load(t.dir, s), "direction_file must be", "a sidecar naming no direction file is refused by name");
    }
    {
        Scratch t("notes-type");
        const std::string d = write_direction(t.dir, "d.f32.bin", 1);
        write_json(t.dir, R"({"format":"ablate-direction-v2","direction_file":")" + d +
                            R"(","direction_dtype":"float32","direction_dim":8,
                               "generation":{"source":"test","notes":7},"alpha_attn_per_layer":{"1":1.0}})");
        threw(try_load(t.dir, s), "generation.notes", "provenance notes that are not text are refused");
    }
    {
        Scratch t("default-alpha");
        const std::string d = write_direction(t.dir, "d.f32.bin", 1);
        write_json(t.dir, R"({"format":"ablate-direction-v2","direction_file":")" + d +
                            R"(","direction_dtype":"float32","direction_dim":8,"default_alpha":1.0,
                               "generation":{"source":"test"},"alpha_mlp_per_layer":{"1":1.0}})");
        threw(try_load(t.dir, s), "default_alpha", "a non-zero default would cut every layer the table omits");
    }
    {
        Scratch t("alpha-range");
        const std::string d = write_direction(t.dir, "d.f32.bin", 1);
        write_json(t.dir, R"({"format":"ablate-direction-v2","direction_file":")" + d +
                            R"(","direction_dtype":"float32","direction_dim":8,"default_alpha":0,
                               "generation":{"source":"test"},"alpha_attn_per_layer":{"1":5.0}})");
        threw(try_load(t.dir, s), "outside", "alpha above the ceiling is refused");
    }
    {
        Scratch t("alpha-neg");
        const std::string d = write_direction(t.dir, "d.f32.bin", 1);
        write_json(t.dir, R"({"format":"ablate-direction-v2","direction_file":")" + d +
                            R"(","direction_dtype":"float32","direction_dim":8,"default_alpha":0,
                               "generation":{"source":"test"},"alpha_attn_per_layer":{"1":-0.5}})");
        threw(try_load(t.dir, s), "outside", "a negative alpha is refused (it would add the direction back)");
    }
    {
        Scratch t("layer-key");
        const std::string d = write_direction(t.dir, "d.f32.bin", 1);
        write_json(t.dir, R"({"format":"ablate-direction-v2","direction_file":")" + d +
                            R"(","direction_dtype":"float32","direction_dim":8,"default_alpha":0,
                               "generation":{"source":"test"},"alpha_attn_per_layer":{"99":1.0}})");
        threw(try_load(t.dir, s), "outside 0..3", "a layer key past the model is refused");
    }
    {
        Scratch t("dim");
        const std::string d = write_direction(t.dir, "d.f32.bin", 1);
        write_json(t.dir, R"({"format":"ablate-direction-v2","direction_file":")" + d +
                            R"(","direction_dtype":"float32","direction_dim":16,"default_alpha":0,
                               "generation":{"source":"test"},"alpha_attn_per_layer":{"1":1.0}})");
        threw(try_load(t.dir, s), "direction_dim", "a direction for another model's width is refused");
    }
    {
        Scratch t("dim-fraction");
        const std::string d = write_direction(t.dir, "d.f32.bin", 1);
        write_json(t.dir, R"({"format":"ablate-direction-v2","direction_file":")" + d +
                            R"(","direction_dtype":"float32","direction_dim":8.5,
                               "generation":{"source":"test"},"alpha_attn_per_layer":{"1":1.0}})");
        threw(try_load(t.dir, s), "direction_dim", "a fractional direction_dim is refused");
    }
    {
        Scratch t("size");
        write_json(t.dir, R"({"format":"ablate-direction-v2","direction_file":"short.f32.bin",
                              "direction_dtype":"float32","direction_dim":8,"default_alpha":0,
                              "generation":{"source":"test"},"alpha_attn_per_layer":{"1":1.0}})");
        { std::ofstream out(t.dir / "short.f32.bin", std::ios::binary); out.write("junk", 4); }
        threw(try_load(t.dir, s), "expected", "a direction file of the wrong size is refused");
    }
    {
        Scratch t("missing");
        write_json(t.dir, R"({"format":"ablate-direction-v2","direction_file":"gone.f32.bin",
                              "direction_dtype":"float32","direction_dim":8,"default_alpha":0,
                              "generation":{"source":"test"},"alpha_attn_per_layer":{"1":1.0}})");
        threw(try_load(t.dir, s), "not found", "a missing direction file fails the boot, never a silent no-op");
    }

    std::printf("\n%d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
