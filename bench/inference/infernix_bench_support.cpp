#include "infernix_bench_support.h"
#include "product/constraint_observation.h"
#include "product/speculative_options.h"
#include "product/logits_options.h"
#include "product/rope_yarn_options.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <thread>

namespace infernix::bench {
namespace {
std::string_view constraint_name(const std::optional<OutputConstraint>& constraint) {
    if (!constraint) return "none";
    switch (constraint->kind) {
    case OutputConstraintKind::Grammar:
        return "grammar";
    case OutputConstraintKind::JsonObject:
        return "json_object";
    case OutputConstraintKind::JsonSchema:
        return "json_schema";
    case OutputConstraintKind::Choice:
        return "choice";
    case OutputConstraintKind::Regex:
        return "regex";
    }
    throw std::logic_error("unknown output constraint");
}

std::string constraint_choices(const std::optional<OutputConstraint>& constraint) {
    std::string result = "[";
    if (constraint) {
        for (std::size_t i = 0; i < constraint->choices.size(); ++i) {
            if (i) result += ',';
            result += '"' + json_escape(constraint->choices[i]) + '"';
        }
    }
    return result + ']';
}

int parse_int(std::string_view text, const char* label) {
    if (text.empty()) { throw std::invalid_argument(std::string(label) + " is empty"); }
    int value         = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size()) {
        throw std::invalid_argument(std::string("invalid ") + label + ": " + std::string(text));
    }
    return value;
}

int parse_nonnegative(std::string_view text, const char* label) {
    const int value = parse_int(text, label);
    if (value < 0) { throw std::invalid_argument(std::string(label) + " must be nonnegative"); }
    return value;
}

int parse_positive(std::string_view text, const char* label) {
    const int value = parse_int(text, label);
    if (value <= 0) { throw std::invalid_argument(std::string(label) + " must be positive"); }
    return value;
}

std::uint32_t parse_u32(std::string_view text, const char* label, bool allow_zero = false) {
    if (text.empty()) { throw std::invalid_argument(std::string(label) + " is empty"); }
    std::uint64_t value = 0;
    const auto result   = std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size() ||
        value > std::numeric_limits<std::uint32_t>::max() || (!allow_zero && value == 0)) {
        throw std::invalid_argument(std::string("invalid ") + label + ": " + std::string(text));
    }
    return static_cast<std::uint32_t>(value);
}

float parse_float_in(std::string_view text, const char* label, float lo, float hi) {
    if (text.empty()) { throw std::invalid_argument(std::string(label) + " is empty"); }
    float value       = 0.0F;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size() || !(value >= lo && value <= hi)) {
        throw std::invalid_argument(std::string("invalid ") + label + ": " + std::string(text));
    }
    return value;
}

std::uint64_t parse_u64(std::string_view text, const char* label) {
    std::uint64_t value = 0;
    const auto result   = std::from_chars(text.data(), text.data() + text.size(), value);
    if (text.empty() || result.ec != std::errc{} || result.ptr != text.data() + text.size()) {
        throw std::invalid_argument(std::string("invalid ") + label + ": " + std::string(text));
    }
    return value;
}

KvCacheStorage parse_kv_cache(std::string_view text) {
    if (text == "bf16") { return KvCacheStorage::BFloat16; }
    if (text == "int8") { return KvCacheStorage::Int8Group64; }
    if (text == "fp8") { return KvCacheStorage::Fp8E4M3Row256; }
    if (text == "nvfp4") { return KvCacheStorage::Nvfp4Group16; }
    if (text == "k8v4") { return KvCacheStorage::Fp8KeyNvfp4Value; }
    if (text == "vq2") { return KvCacheStorage::Vq2; }
    if (text == "k4v2") { return KvCacheStorage::Q4KeyVq2Value; }
    throw std::invalid_argument("--kv-dtype must be bf16, int8, fp8, nvfp4, k8v4, vq2, or k4v2");
}

std::vector<int> parse_int_list(std::string_view value, const char* label) {
    std::vector<int> out;
    std::size_t start = 0;
    while (true) {
        const std::size_t comma      = value.find(',', start);
        const std::string_view piece = value.substr(
            start, comma == std::string_view::npos ? std::string_view::npos : comma - start);
        out.push_back(parse_positive(piece, label));
        if (comma == std::string_view::npos) { break; }
        start = comma + 1;
    }
    return out;
}

std::vector<std::pair<int, int>> parse_pair_list(std::string_view value, const char* label) {
    std::vector<std::pair<int, int>> out;
    std::size_t start = 0;
    while (true) {
        const std::size_t semi       = value.find(';', start);
        const std::string_view piece = value.substr(
            start, semi == std::string_view::npos ? std::string_view::npos : semi - start);
        const std::size_t comma = piece.find(',');
        if (comma == std::string_view::npos ||
            piece.find(',', comma + 1) != std::string_view::npos) {
            throw std::invalid_argument(std::string(label) + " expects P,G pairs");
        }
        out.emplace_back(parse_positive(piece.substr(0, comma), "prompt-gen P"),
                         parse_positive(piece.substr(comma + 1), "prompt-gen G"));
        if (semi == std::string_view::npos) { break; }
        start = semi + 1;
    }
    return out;
}

std::string existing_read_path(const std::string& path) {
    if (std::filesystem::exists(path)) { return path; }
#ifdef INFERNIX_SOURCE_DIR
    const auto source_relative =
        (std::filesystem::path(INFERNIX_SOURCE_DIR) / path).lexically_normal();
    if (std::filesystem::exists(source_relative)) { return source_relative.string(); }
#endif
    return path;
}

const char* kind_string(TestKind kind) {
    switch (kind) {
    case TestKind::Prefill:
        return "pp";
    case TestKind::Decode:
        return "tg";
    case TestKind::PrefillDecode:
        return "pp+tg";
    }
    return "unknown";
}

std::uint32_t checked_context(std::uint64_t value, const char* label) {
    if (value == 0 || value > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error(std::string(label) + " exceeds the supported context range");
    }
    return static_cast<std::uint32_t>(value);
}

std::string number(double value) {
    std::ostringstream out;
    out << std::setprecision(10) << value;
    return out.str();
}

std::string format_bytes(std::uint64_t bytes) {
    if (bytes == 0) { return "-"; }
    constexpr double gib = 1024.0 * 1024.0 * 1024.0;
    constexpr double mib = 1024.0 * 1024.0;
    std::ostringstream out;
    if (static_cast<double>(bytes) >= gib) {
        out << std::fixed << std::setprecision(2) << static_cast<double>(bytes) / gib << " GiB";
    } else {
        out << std::fixed << std::setprecision(1) << static_cast<double>(bytes) / mib << " MiB";
    }
    return out.str();
}

std::string rate_cell(const std::vector<double>& values) {
    if (values.empty()) { return "-"; }
    const Stats stats = compute_stats(values);
    std::ostringstream out;
    out << std::fixed << std::setprecision(2) << stats.mean;
    if (stats.count > 1) {
        out << " \u00b1 " << std::fixed << std::setprecision(2) << stats.stddev;
    }
    return out.str();
}

void append_stat(std::ostringstream& out, std::string_view name, const std::vector<double>& values,
                 std::string_view indent) {
    out << indent << '"' << name << "_mean\": ";
    if (values.empty()) {
        out << "null,\n" << indent << '"' << name << "_stddev\": null";
    } else {
        const Stats stats = compute_stats(values);
        out << number(stats.mean) << ",\n"
            << indent << '"' << name << "_stddev\": " << number(stats.stddev);
    }
}

SpeculativeStats aggregate_speculative(const TestResult& result) {
    SpeculativeStats out;
    for (const RepTiming& rep : result.reps) {
        const SpeculativeStats& in = rep.speculative;
        out.enabled                = out.enabled || in.enabled;
        out.backend                = in.backend;
        out.draft_window           = std::max(out.draft_window, in.draft_window);
        out.rounds += in.rounds;
        out.drafted_tokens += in.drafted_tokens;
        out.accepted_tokens += in.accepted_tokens;
        out.fallback_steps += in.fallback_steps;
        out.ngram_rounds += in.ngram_rounds;
        out.ngram_drafted_tokens += in.ngram_drafted_tokens;
        out.ngram_accepted_tokens += in.ngram_accepted_tokens;
        out.tree_rounds += in.tree_rounds;
        out.tree_side_rounds += in.tree_side_rounds;
        out.tree_side_accepted_tokens += in.tree_side_accepted_tokens;
        if (out.accepted_per_position.size() < in.accepted_per_position.size()) {
            out.accepted_per_position.resize(in.accepted_per_position.size());
        }
        for (std::size_t i = 0; i < in.accepted_per_position.size(); ++i) {
            out.accepted_per_position[i] += in.accepted_per_position[i];
        }
        if (out.rounds_by_draft_length.size() < in.rounds_by_draft_length.size()) {
            out.rounds_by_draft_length.resize(in.rounds_by_draft_length.size());
        }
        for (std::size_t i = 0; i < in.rounds_by_draft_length.size(); ++i) {
            out.rounds_by_draft_length[i] += in.rounds_by_draft_length[i];
        }
    }
    return out;
}

std::uint64_t decode_engine_tokens(const TestResult& result, const RepTiming& rep) {
    if (!result.test.has_decode()) { return 0; }
    if (!rep.speculative.enabled) {
        return rep.generated_output_tokens ? rep.generated_output_tokens - 1U : 0U;
    }
    return rep.speculative.rounds + rep.speculative.accepted_tokens +
           rep.speculative.fallback_steps;
}

void append_arena_json(std::ostringstream& out, std::string_view name,
                       const ArenaMemorySummary& arena, std::string_view indent,
                       bool trailing_comma) {
    out << indent << '"' << name << "\": {\"capacity_bytes\": " << arena.capacity_bytes
        << ", \"used_bytes\": " << arena.used_bytes
        << ", \"peak_used_bytes\": " << arena.peak_used_bytes << '}';
    if (trailing_comma) { out << ','; }
    out << '\n';
}

void append_vision_workspace_json(std::ostringstream& out,
                                  const std::optional<VisionWorkspaceMemorySummary>& vision,
                                  std::string_view indent, bool trailing_comma) {
    out << indent << "\"vision_workspace\": ";
    if (!vision) {
        out << "null";
    } else {
        out << "{\"aggregate_prompt_tokens\": " << vision->aggregate_prompt_tokens
            << ", \"max_item_tokens\": " << vision->max_item_tokens
            << ", \"general_capacity_bytes\": " << vision->general_capacity_bytes
            << ", \"encode_peak_bytes\": " << vision->encode_peak_bytes
            << ", \"handoff_offset_bytes\": " << vision->handoff_offset_bytes
            << ", \"handoff_capacity_bytes\": " << vision->handoff_capacity_bytes
            << ", \"handoff_active_bytes\": " << vision->handoff_active_bytes
            << ", \"handoff_peak_bytes\": " << vision->handoff_peak_bytes << '}';
    }
    if (trailing_comma) { out << ','; }
    out << '\n';
}

void append_speculative_json(std::ostringstream& out, const SpeculativeStats& stats,
                             std::string_view indent) {
    out << indent << "\"speculative\": {\n"
        << indent << "  \"backend\": \"" << product::speculative_backend_name(stats.backend)
        << "\",\n"
        << indent << "  \"enabled\": " << (stats.enabled ? "true" : "false") << ",\n"
        << indent << "  \"draft_window\": " << stats.draft_window << ",\n"
        << indent << "  \"rounds\": " << stats.rounds << ",\n"
        << indent << "  \"drafted_tokens\": " << stats.drafted_tokens << ",\n"
        << indent << "  \"accepted_tokens\": " << stats.accepted_tokens << ",\n"
        << indent << "  \"fallback_steps\": " << stats.fallback_steps << ",\n"
        << indent << "  \"ngram_rounds\": " << stats.ngram_rounds << ",\n"
        << indent << "  \"ngram_drafted_tokens\": " << stats.ngram_drafted_tokens << ",\n"
        << indent << "  \"ngram_accepted_tokens\": " << stats.ngram_accepted_tokens << ",\n"
        << indent << "  \"tree_rounds\": " << stats.tree_rounds << ",\n"
        << indent << "  \"tree_side_rounds\": " << stats.tree_side_rounds << ",\n"
        << indent << "  \"tree_side_accepted_tokens\": " << stats.tree_side_accepted_tokens << ",\n"
        << indent << "  \"acceptance_rate\": ";
    if (stats.drafted_tokens == 0) {
        out << "null";
    } else {
        out << number(static_cast<double>(stats.accepted_tokens) /
                      static_cast<double>(stats.drafted_tokens));
    }
    out << ",\n" << indent << "  \"acceptance_length\": ";
    if (stats.rounds == 0) {
        out << "null";
    } else {
        out << number(1.0 + static_cast<double>(stats.accepted_tokens) /
                                static_cast<double>(stats.rounds));
    }
    out << ",\n" << indent << "  \"accepted_per_position\": [";
    for (std::size_t i = 0; i < stats.accepted_per_position.size(); ++i) {
        if (i != 0) { out << ", "; }
        out << stats.accepted_per_position[i];
    }
    out << "],\n" << indent << "  \"rounds_by_draft_length\": [";
    for (std::size_t i = 0; i < stats.rounds_by_draft_length.size(); ++i) {
        if (i != 0) { out << ", "; }
        out << stats.rounds_by_draft_length[i];
    }
    out << "]\n" << indent << '}';
}

void append_timings_json(std::ostringstream& out, const GenerationTimings& timings,
                         std::string_view indent) {
    out << indent << "\"timings\": {" << "\"prepare_seconds\": " << number(timings.prepare_seconds)
        << ", " << "\"vision_seconds\": " << number(timings.vision_seconds) << ", "
        << "\"prefill_seconds\": " << number(timings.prefill_seconds) << ", "
        << "\"decode_seconds\": " << number(timings.decode_seconds) << ", "
        << "\"total_seconds\": " << number(timings.total_seconds) << '}';
}

// The fast paths the Engine runs, comma-separated; "off" on the exact route, "none" when none is
// qualified for the configuration.
std::string fast_path_list(const BenchEnvironment& env) {
    std::string out;
    for (const std::string& path : env.load.fast_paths) { out += (out.empty() ? "" : ",") + path; }
    if (out.empty()) { out = env.load.fast_paths_disabled ? "off" : "none"; }
    return out;
}

const char* ngram_residency_name(NgramResidency residency) {
    switch (residency) {
    case NgramResidency::Ram: return "ram";
    case NgramResidency::Volume: return "volume";
    case NgramResidency::Auto: break;
    }
    return "auto";
}

// A MiB budget as the options spell it; "auto" when unset.
std::string mib_or_auto(const std::optional<std::uint64_t>& bytes) {
    return bytes ? std::to_string(*bytes >> 20) : std::string("auto");
}

// The machine block's numbers: 0 (unknown or not measured) is null; rates in GB/s (10^9).
std::string count_or_null(std::uint32_t value) {
    return value == 0 ? "null" : std::to_string(value);
}

std::string gbps_or_null(double bytes_per_second) {
    return bytes_per_second > 0.0 ? number(bytes_per_second * 1e-9) : "null";
}

std::string machine_json(const BenchMachine& m) {
    return "{\"pcie_generation\": " + count_or_null(m.pcie_generation) +
           ", \"pcie_width\": " + count_or_null(m.pcie_width) +
           ", \"pcie_max_generation\": " + count_or_null(m.pcie_max_generation) +
           ", \"pcie_max_width\": " + count_or_null(m.pcie_max_width) +
           ", \"h2d_gbps\": " + gbps_or_null(m.h2d_bytes_per_second) +
           ", \"ram_read_gbps\": " + gbps_or_null(m.ram_read_bytes_per_second) +
           ", \"ram_read_threads\": " + count_or_null(m.ram_read_threads) + "}";
}

// One kind's executions: count, mean and maximum in milliseconds (null without executions).
std::string duration_json(const DurationCounter& d) {
    const auto ms = [](double ns) { return number(ns * 1e-6); };
    if (d.count == 0) { return "{\"count\": 0, \"mean_ms\": null, \"max_ms\": null}"; }
    return "{\"count\": " + std::to_string(d.count) +
           ", \"mean_ms\": " + ms(static_cast<double>(d.total_ns) / static_cast<double>(d.count)) +
           ", \"max_ms\": " + ms(static_cast<double>(d.max_ns)) + "}";
}

// The test's executions by kind, and its decode rounds by batch size (sizes that ran).
std::string execution_json(const RuntimeExecutionLatency& e) {
    std::string out = "{\"prefill\": " + duration_json(e.prefill) +
                      ", \"replay\": " + duration_json(e.replay) +
                      ", \"control\": " + duration_json(e.control) +
                      ", \"decode\": " + duration_json(e.decode) + ", \"decode_by_batch\": [";
    bool first      = true;
    for (std::size_t b = 0; b < e.decode_by_batch.size(); ++b) {
        if (e.decode_by_batch[b].count == 0) { continue; }
        out += std::string(first ? "" : ", ") + "{\"batch\": " + std::to_string(b + 1) + ", " +
               duration_json(e.decode_by_batch[b]).substr(1);
        first = false;
    }
    return out + "]}";
}

std::string machine_text(const BenchMachine& m) {
    const auto link = [](std::uint32_t generation, std::uint32_t width) {
        return (generation == 0 ? std::string("?") : std::to_string(generation)) + " x" +
               (width == 0 ? std::string("?") : std::to_string(width));
    };
    const auto rate = [](double bytes_per_second) {
        if (bytes_per_second <= 0.0) { return std::string("-"); }
        std::ostringstream out;
        out << std::fixed << std::setprecision(1) << bytes_per_second * 1e-9 << " GB/s";
        return out.str();
    };
    return "PCIe " + link(m.pcie_generation, m.pcie_width) + " (max " +
           link(m.pcie_max_generation, m.pcie_max_width) + "), H2D " +
           rate(m.h2d_bytes_per_second) + ", RAM read " + rate(m.ram_read_bytes_per_second) + " (" +
           std::to_string(m.ram_read_threads) + " threads)";
}

} // namespace

std::uint32_t BenchTest::requested_output_tokens() const {
    return checked_context(static_cast<std::uint64_t>(has_decode() ? n_gen : 0) + 1ULL,
                           "requested benchmark output");
}

std::uint32_t BenchTest::required_context(const SpeculativeOptions& speculative) const {
    const std::uint64_t prompt =
        static_cast<std::uint64_t>(kind == TestKind::Decode ? kDecodeSeedTokens : n_prompt);
    const std::uint64_t decode = static_cast<std::uint64_t>(has_decode() ? n_gen : 0);
    const std::uint64_t mtp_margin =
        speculative.backend == SpeculativeBackend::Mtp ? 2ULL * speculative.draft_tokens : 0;
    return checked_context(prompt + decode + mtp_margin, "benchmark context requirement");
}

std::string usage_text(std::string_view program) {
    if (program.empty()) { program = "infernix_bench"; }
    std::ostringstream out;
    out << "Usage: " << program << " --weights <artifact.ninfer> [options]\n\n"
        << "Product-route throughput benchmark over infernix::Engine. pp measures Engine prefill;\n"
        << "tg measures G generated tokens after an untimed one-token seed prefill.\n\n"
        << "Options:\n"
        << "  --weights <path>            required .ninfer artifact\n"
        << "  --corpus <path>             token-id corpus (default: " << kDefaultCorpusPath << ")\n"
        << "  --corpus-offset <n>         prompts start n tokens into the corpus, wrapping (default: 0)\n"
        << "  --prompt-ids <path>         one exact prompt (comma- or space-separated token ids, e.g. a\n"
           "                             chat-templated fixture) for every test; each -n N prefills it\n"
           "                             and decodes N tokens (no -p / -pg)\n"
        << "  --temperature <f>           sample at this temperature (default: greedy); with --top-p,\n"
           "  --top-p <f> --top-k <n>    --top-k and --min-p (each unset: the model's default)\n"
           "  --min-p <f>\n"
        << "  --seed <n>                  row k of measured repetition r samples with seed\n"
           "                             n + r * concurrency + k (default: 1)\n"
        << "  --stop-at-eos               the model's stop tokens end a request (it may end early)\n"
        << "  --route-trace <path>        write the expert route trace (offloaded experts)\n"
        << "  --ngram-volume <path>       Qwen3.8-Flash-Next n-gram volume: the artifact's, or any FP8,\n"
        << "                              IQ4_NL, BF16 or NVFP4 volume of the table (default: <weights>.ngram)\n"
        << "  --ngram-residency <mode>    Qwen3.8-Flash-Next n-gram table: auto, ram (all of it in pinned\n"
        << "                              RAM) or volume (rows read on demand) (default: auto)\n"
        << "  --expert-stripe <path>      Qwen3.8-Flash-Next expert stripe: the SSD tier reads each record's\n"
        << "                              tail from it (a second drive) beside its head in the artifact\n"
        << "  --ram-headroom-mib <N>      physical Host RAM pinned weights and caches leave free (default "
        << (infernix::kDefaultRamHeadroomBytes >> 20) << ")\n"
        << "  --expert-ram-mib <N|auto>   Qwen3.8-Flash-Next: Host RAM for the routed experts (default auto);\n"
        << "                              less than all of them need reads the rest from the artifact\n"
        << "  --expert-pin-mib <N|auto>   Qwen3.8-Flash-Next: at most this much of the experts' RAM page-locked;\n"
        << "                              the rest stays in RAM for the CPU (default auto: what Windows allows)\n"
        << "  --vram-headroom-mib <N|auto> Qwen3.8-Flash-Next: VRAM the expert cache leaves free for the\n"
        << "                              display and other programs (default auto: 1024 with a display, else 256)\n"
        << "  --vram-past-budget          Qwen3.8-Flash-Next on Windows: also use the VRAM the OS budget withholds\n"
        << "  --expert-tail-skip <1..9>   Qwen3.8-Flash-Next: skip a VRAM-missing expert every token of a decode\n"
        << "                              round routes at top-10 rank R or lower (faster, lower quality; changes\n"
        << "                              the output; default off, Strata's default 7)\n"
        << "  --expert-a4-activations     nvidia/Qwen3.8-Flash-Next-FP4-Local only: routed experts with 4-bit\n"
        << "                              (NVFP4) activations, using the activation scales NVIDIA calibrated\n"
        << "                              for them (changes the output; default: 16-bit, as NVIDIA serves it)\n"
        << "  --embedding-host            Qwen3.5-family artifacts: the token embedding in pinned host memory,\n"
        << "                              read zero-copy (its VRAM goes to the KV cache; same output)\n"
        << "  --concurrency <1..8>     simultaneous requests per repetition (default: 1)\n"
        << "  --grammar-file <path>      constrain output with GBNF\n"
        << "  --json-object              constrain output to a JSON object\n"
        << "  --json-schema-file <path>  constrain output with JSON Schema\n"
        << "  --regex <pattern>          constrain the complete output with regex\n"
        << "  --choice <text>            literal output candidate; repeat to supply choices\n"
        << "  --mixed-constraints        apply constraints to alternate requests; requires "
           "concurrency "
           ">= 2\n"
        << "  -p, --n-prompt <list>       pp lengths, for example 512,2048\n"
        << "  -n, --n-gen <list>          tg lengths, for example 128\n"
        << "  -pg, --prompt-gen <P,G;..>  combined pp+tg tests\n"
        << "  -r, --repetitions <n>       measured repetitions (default: " << kDefaultRepetitions
        << ")\n"
        << "  --warmup <n>                discarded repetitions (default: " << kDefaultWarmup
        << ")\n"
        << "  --rope-yarn-factor <F>      startup-fixed YaRN, finite [1,4] (default 1); ceiling "
           "only\n"
        << "  --max-ctx <tokens>          override auto-sized context capacity\n"
        << "  --prefill-chunk <tokens>    multiple of " << kPrefillChunkAlignment
        << " (default: 16384 for Qwen3.8-Flash-Next, 1024 otherwise)\n"
        << "  --kv-dtype <bf16|int8|fp8|nvfp4|k8v4|vq2|k4v2>  KV cache storage (default: bf16)\n"
        << "  --logits-dtype <fp32|bf16>  the head's logits (default: fp32; bf16 rounds each once,\n"
        << "                              as Transformers returns them, for comparisons)\n"
        << "  --use-original-int8-prefill-kernel  original INT8-KV prompt kernel at the\n"
        << "                              requested chunk (default: fast kernel, wave-aligned\n"
        << "                              chunks); requires --kv-dtype int8\n"
        << "  --use-original-nvfp4-prefill-kernel  tiled NVFP4-KV prompt kernel (default:\n"
        << "                              fast kernel); requires --kv-dtype nvfp4\n"
        << "  --reproducible              the reproducible route (default: every fast path qualified\n"
        << "                              for the configuration, the artifact's kernels and KV)\n"
        << "  --fast-path <name>          qualification: also run this fast path, qualified\n"
        << "                              or not (a registry name; repeatable)\n"
        << "  --prefill-split-workspace-mib <n>  prompt-attention split workspace (default: "
        << kDefaultPrefillSplitWorkspaceMiB << ";\n"
        << "                              128-384 recommended, 0 = no splitting)\n"
        << "  --spec <mtp|dflash|dflash2> speculative backend (default: none)\n"
        << "  --draft-tokens <n>|auto    MTP 1..7 (Qwen3.5/3.6 1..5); DFlash/DFlash2 1..15\n"
        << "                             (Qwen3.8-Flash-Next DFlash2 1..7); n drafts n every round,\n"
        << "                             auto lets MTP (and Qwen3.8-Flash-Next DFlash2) pick each\n"
        << "                             round's length (0..7; Qwen3.5/3.6 1..5 by adaptive-mtp where\n"
        << "                             qualified, else the model's fastest fixed length)\n"
        << "  --draft-model <path>       Qwen3.8-Flash-Next DFlash2: the drafter's companion artifact\n"
        << "  --spec-min-p <p>           Qwen3.8-Flash-Next MTP draft-confidence gate 0..1\n"
        << "                             (default 0.5 with --draft-tokens auto, else 0 (every draft))\n"
        << "  --ngram-draft-tokens <n>   copy proposals 1..63; 0 disables (default: 0)\n"
        << "  --ngram-min-match <n>      copy admission 4..64 (default: 12)\n"
        << "  --draft-tree-nodes <auto|list>  DFlash2 tree verification (default: off): auto\n"
           "                             picks a tree or the single draft each round from\n"
           "                             measured speed (recommended); a list fixes tree sizes\n"
           "                             in tokens by batch size, e.g. 16,12,12,0 (last entry\n"
           "                             repeats, 0 = single draft)\n"
        << "  --draft-tree-paths <n>     most branches per tree 2..8 (default: 8)\n"
        << "  --lm-head-draft             use the optimized proposal head; requires a speculative "
           "backend\n"
        << "  --device <id>               CUDA device ordinal (default: 0)\n"
        << "  --no-cuda-graph             use eager decode\n"
        << "  --profile-measured          bracket one measured repetition with CUDA profiler API\n"
        << "  -o, --output <table|json|csv>  output format (default: table)\n"
        << "  --output-file <path>        write report to a file\n"
        << "  -h, --help                  show this help\n\n"
        << "With no -p/-n/-pg, the matrix is pp" << kDefaultNPrompt << " and tg" << kDefaultNGen
        << ".\n";
    return out.str();
}

BenchOptions parse_args(int argc, char** argv) {
    BenchOptions options;
    bool saw_artifact = false;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        auto value = [&](std::string_view flag) -> std::string {
            if (i + 1 >= argc) {
                throw std::invalid_argument(std::string(flag) + " requires value");
            }
            return argv[++i];
        };
        if (arg == "-h" || arg == "--help") {
            options.help_requested = true;
            return options;
        }
        if (arg == "--weights") {
            options.artifact_path = value("--weights");
            saw_artifact          = true;
        } else if (arg == "--corpus") {
            options.corpus_path = value("--corpus");
        } else if (arg == "--corpus-offset") {
            options.corpus_offset =
                static_cast<std::size_t>(parse_nonnegative(value("--corpus-offset"), "corpus offset"));
        } else if (arg == "--prompt-ids") {
            options.prompt_ids_path = value("--prompt-ids");
        } else if (arg == "--temperature") {
            options.temperature = parse_float_in(value("--temperature"), "temperature", 0.0F, 2.0F);
        } else if (arg == "--top-p") {
            options.top_p = parse_float_in(value("--top-p"), "top-p", 0.0F, 1.0F);
        } else if (arg == "--top-k") {
            options.top_k = static_cast<std::int32_t>(parse_u32(value("--top-k"), "top-k", true));
        } else if (arg == "--min-p") {
            options.min_p = parse_float_in(value("--min-p"), "min-p", 0.0F, 1.0F);
        } else if (arg == "--seed") {
            options.seed = parse_u64(value("--seed"), "seed");
        } else if (arg == "--stop-at-eos") {
            options.stop_at_eos = true;
        } else if (arg == "--route-trace") {
            options.route_trace_path = value("--route-trace");
        } else if (arg == "--ngram-volume") {
            options.ngram_volume_path = value("--ngram-volume");
        } else if (arg == "--ngram-residency") {
            options.ngram_residency = infernix::parse_ngram_residency(value("--ngram-residency"));
        } else if (arg == "--expert-stripe") {
            options.expert_stripe_path = value("--expert-stripe");
        } else if (arg == "--ram-headroom-mib") {
            const std::string text = value("--ram-headroom-mib");
            std::size_t used       = 0;
            unsigned long long mib = 0;
            try {
                mib = std::stoull(text, &used);
            } catch (const std::exception&) { used = 0; }
            if (used != text.size() || text.empty() || text.front() == '-' ||
                mib > (std::numeric_limits<std::uint64_t>::max() >> 20)) {
                throw std::invalid_argument("--ram-headroom-mib must be a MiB count");
            }
            options.ram_headroom_bytes = static_cast<std::uint64_t>(mib) << 20;
        } else if (arg == "--vram-past-budget") {
            options.vram_past_budget = true;
        } else if (arg == "--expert-tail-skip") {
            options.expert_tail_skip = parse_u32(value("--expert-tail-skip"), "expert-tail-skip");
        } else if (arg == "--expert-a4-activations") {
            options.expert_a4_activations = true;
        } else if (arg == "--embedding-host") {
            options.embedding_host = true;
        } else if (arg == "--expert-ram-mib") {
            const std::string text = value("--expert-ram-mib");
            if (text == "auto") {
                options.expert_ram_bytes.reset();
            } else {
                std::size_t used       = 0;
                unsigned long long mib = 0;
                try {
                    mib = std::stoull(text, &used);
                } catch (const std::exception&) { used = 0; }
                if (used != text.size() || text.empty() || text.front() == '-' || mib == 0 ||
                    mib > (std::numeric_limits<std::uint64_t>::max() >> 20)) {
                    throw std::invalid_argument("--expert-ram-mib must be a MiB count or auto");
                }
                options.expert_ram_bytes = static_cast<std::uint64_t>(mib) << 20;
            }
        } else if (arg == "--expert-pin-mib") {
            const std::string text = value("--expert-pin-mib");
            if (text == "auto") {
                options.expert_pin_bytes.reset();
            } else {
                std::size_t used       = 0;
                unsigned long long mib = 0;
                try {
                    mib = std::stoull(text, &used);
                } catch (const std::exception&) { used = 0; }
                if (used != text.size() || text.empty() || text.front() == '-' || mib == 0 ||
                    mib > (std::numeric_limits<std::uint64_t>::max() >> 20)) {
                    throw std::invalid_argument("--expert-pin-mib must be a MiB count or auto");
                }
                options.expert_pin_bytes = static_cast<std::uint64_t>(mib) << 20;
            }
        } else if (arg == "--vram-headroom-mib") {
            const std::string text = value("--vram-headroom-mib");
            if (text == "auto") {
                options.vram_headroom_bytes.reset();
            } else {
                std::size_t used       = 0;
                unsigned long long mib = 0;
                try {
                    mib = std::stoull(text, &used);
                } catch (const std::exception&) { used = 0; }
                if (used != text.size() || text.empty() || text.front() == '-' ||
                    mib > (std::numeric_limits<std::size_t>::max() >> 20)) {
                    throw std::invalid_argument("--vram-headroom-mib must be a MiB count or auto");
                }
                options.vram_headroom_bytes = static_cast<std::size_t>(mib) << 20;
            }
        } else if (arg == "--concurrency") {
            options.concurrency = parse_u32(value("--concurrency"), "concurrency");
        } else if (arg == "--grammar-file" || arg == "--json-schema-file" ||
                   arg == "--json-object" || arg == "--regex" || arg == "--choice") {
            if (options.constraint &&
                !(arg == "--choice" && options.constraint->kind == OutputConstraintKind::Choice))
                throw std::invalid_argument("select only one output constraint");
            if (arg == "--choice") {
                if (!options.constraint) options.constraint = OutputConstraint::choice({});
                options.constraint->choices.push_back(value(arg));
            } else if (arg == "--regex") {
                options.constraint = OutputConstraint::regex(value(arg));
            } else if (arg == "--json-object") {
                options.constraint = OutputConstraint::json_object();
            } else {
                options.constraint      = arg == "--grammar-file" ? OutputConstraint::grammar({})
                                                                  : OutputConstraint::json_schema({});
                options.constraint_file = value(arg);
            }
        } else if (arg == "--mixed-constraints") {
            options.mixed_constraints = true;
        } else if (arg == "-p" || arg == "--n-prompt") {
            auto parsed = parse_int_list(value("--n-prompt"), "n-prompt");
            options.n_prompt.insert(options.n_prompt.end(), parsed.begin(), parsed.end());
        } else if (arg == "-n" || arg == "--n-gen") {
            auto parsed = parse_int_list(value("--n-gen"), "n-gen");
            options.n_gen.insert(options.n_gen.end(), parsed.begin(), parsed.end());
        } else if (arg == "-pg" || arg == "--prompt-gen") {
            auto parsed = parse_pair_list(value("--prompt-gen"), "prompt-gen");
            options.prompt_gen.insert(options.prompt_gen.end(), parsed.begin(), parsed.end());
        } else if (arg == "-r" || arg == "--repetitions") {
            options.repetitions = parse_positive(value("--repetitions"), "repetitions");
        } else if (arg == "--warmup") {
            options.warmup = parse_nonnegative(value("--warmup"), "warmup");
        } else if (arg == "--rope-yarn-factor") {
            options.rope_yarn_factor = product::parse_rope_yarn_factor(value("--rope-yarn-factor"));
        } else if (arg == "--max-ctx") {
            options.max_context = parse_u32(value("--max-ctx"), "max-ctx");
        } else if (arg == "--prefill-chunk") {
            options.prefill_chunk = parse_u32(value("--prefill-chunk"), "prefill-chunk");
            if (options.prefill_chunk == 0) {
                throw std::invalid_argument("--prefill-chunk must be a positive multiple of 128");
            }
        } else if (arg == "--kv-dtype") {
            options.kv_cache = parse_kv_cache(value("--kv-dtype"));
        } else if (arg == "--logits-dtype") {
            options.logits_dtype = product::parse_logits_dtype(value("--logits-dtype"));
        } else if (arg == "--use-original-int8-prefill-kernel") {
            options.original_int8_prefill_kernel = true;
        } else if (arg == "--prefill-split-workspace-mib") {
            options.prefill_split_workspace_mib =
                parse_u32(value("--prefill-split-workspace-mib"), "prefill-split-workspace-mib", true);
        } else if (arg == "--use-original-nvfp4-prefill-kernel") {
            options.original_nvfp4_prefill_kernel = true;
        } else if (arg == "--reproducible") {
            options.fast_paths = false;
        } else if (arg == "--fast-path") {
            const std::string name(value("--fast-path"));
            const std::optional<FastPath> path = parse_fast_path(name);
            if (!path) { throw std::invalid_argument("unknown fast path: " + name); }
            options.fast_path_candidates.push_back(*path);
        } else if (arg == "--spec") {
            options.speculative.backend = product::parse_speculative_backend(value("--spec"));
        } else if (arg == "--draft-tokens") {
            product::apply_draft_tokens(options.speculative, value("--draft-tokens"));
        } else if (arg == "--draft-model") {
            options.speculative.draft_model = std::filesystem::path(std::string(value("--draft-model")));
        } else if (arg == "--spec-min-p") {
            options.speculative.draft_min_p = product::parse_draft_min_p(value("--spec-min-p"));
        } else if (arg == "--ngram-draft-tokens") {
            options.speculative.ngram_draft_tokens =
                parse_u32(value("--ngram-draft-tokens"), "ngram-draft-tokens", true);
        } else if (arg == "--ngram-min-match") {
            options.speculative.ngram_min_match =
                parse_u32(value("--ngram-min-match"), "ngram-min-match");
        } else if (arg == "--lm-head-draft") {
            options.speculative.proposal_head = ProposalHead::Optimized;
        } else if (arg == "--draft-tree-nodes") {
            product::apply_draft_tree_nodes(options.speculative, value("--draft-tree-nodes"));
        } else if (arg == "--draft-tree-paths") {
            options.speculative.draft_tree_paths =
                parse_u32(value("--draft-tree-paths"), "draft-tree-paths");
        } else if (arg == "--device") {
            options.device = parse_nonnegative(value("--device"), "device");
        } else if (arg == "--no-cuda-graph") {
            options.use_cuda_graph = false;
        } else if (arg == "--profile-measured") {
            options.profile_measured = true;
        } else if (arg == "-o" || arg == "--output") {
            const std::string selected = value("--output");
            if (selected == "table") {
                options.output = OutputFormat::Table;
            } else if (selected == "json") {
                options.output = OutputFormat::Json;
            } else if (selected == "csv") {
                options.output = OutputFormat::Csv;
            } else {
                throw std::invalid_argument("--output must be table, json, or csv");
            }
        } else if (arg == "--output-file") {
            options.output_file = value("--output-file");
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(arg));
        }
    }
    if (!saw_artifact) { throw std::invalid_argument("--weights is required"); }
    if (options.concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("--concurrency must be in [1,8]");
    }
    if (options.mixed_constraints && (!options.constraint || options.concurrency < 2)) {
        throw std::invalid_argument(
            "--mixed-constraints requires an output constraint and concurrency >= 2");
    }
    if (options.prefill_chunk % kPrefillChunkAlignment != 0) {
        throw std::invalid_argument("--prefill-chunk must be a multiple of 128");
    }
    if (!options.prompt_ids_path.empty() &&
        (!options.n_prompt.empty() || !options.prompt_gen.empty() || options.n_gen.empty() || options.corpus_offset != 0)) {
        throw std::invalid_argument("--prompt-ids takes -n N tests only, without --corpus-offset");
    }
    if (!options.temperature && (options.top_p || options.top_k || options.min_p)) {
        throw std::invalid_argument("--top-p, --top-k and --min-p need --temperature");
    }
    product::validate_speculative_cli_options(options.speculative);
    return options;
}

std::vector<BenchTest> expand_tests(const BenchOptions& options) {
    std::vector<int> prompts = options.n_prompt;
    std::vector<int> gens    = options.n_gen;
    auto combined            = options.prompt_gen;
    if (prompts.empty() && gens.empty() && combined.empty()) {
        prompts.push_back(kDefaultNPrompt);
        gens.push_back(kDefaultNGen);
    }

    std::vector<BenchTest> tests;
    for (const int p : prompts) {
        tests.push_back({TestKind::Prefill, p, 0, "pp" + std::to_string(p)});
    }
    for (const int g : gens) {
        tests.push_back({TestKind::Decode, 0, g, "tg" + std::to_string(g)});
    }
    for (const auto& [p, g] : combined) {
        tests.push_back(
            {TestKind::PrefillDecode, p, g, "pp" + std::to_string(p) + "+tg" + std::to_string(g)});
    }
    return tests;
}

std::uint32_t resolve_max_context(const std::vector<BenchTest>& tests,
                                  std::optional<std::uint32_t> override_max_context,
                                  const SpeculativeOptions& speculative, bool use_cuda_graph) {
    product::validate_speculative_cli_options(speculative);
    std::uint32_t required = 0;
    std::string driver;
    bool has_decode = false;
    for (const BenchTest& test : tests) {
        const std::uint32_t candidate = test.required_context(speculative);
        if (candidate > required) {
            required = candidate;
            driver   = test.label;
        }
        has_decode = has_decode || test.has_decode();
    }
    if (use_cuda_graph && has_decode) {
        const std::uint32_t candidate = decode_graph_prime_required_context(speculative);
        if (candidate > required) {
            required = candidate;
            driver   = "decode graph prime";
        }
    }
    if (required == 0) { throw std::invalid_argument("no benchmark tests were selected"); }
    if (override_max_context && *override_max_context < required) {
        throw std::invalid_argument("--max-ctx " + std::to_string(*override_max_context) +
                                    " is too small; " + driver + " needs " +
                                    std::to_string(required));
    }
    return override_max_context.value_or(required);
}

void validate_prompt_lengths(const std::vector<BenchTest>& tests, std::size_t corpus_tokens) {
    if (corpus_tokens < static_cast<std::size_t>(kDecodeSeedTokens)) {
        throw std::invalid_argument("corpus is too small to seed decode tests");
    }
    for (const BenchTest& test : tests) {
        if (test.has_prefill() && static_cast<std::size_t>(test.n_prompt) > corpus_tokens) {
            throw std::invalid_argument(test.label + " exceeds the token-id corpus");
        }
    }
}

std::vector<TokenId> load_corpus_ids(const std::string& path) {
    std::ifstream input(existing_read_path(path));
    if (!input) { throw std::runtime_error("failed to open corpus ids file: " + path); }
    std::vector<TokenId> ids;
    std::string word;
    while (input >> word) {
        const int parsed = parse_nonnegative(word, "corpus token id");
        ids.push_back(static_cast<TokenId>(parsed));
    }
    if (ids.empty()) { throw std::invalid_argument("corpus ids file is empty: " + path); }
    return ids;
}

std::vector<TokenId> load_prompt_ids(const std::string& path) {
    std::ifstream input(existing_read_path(path));
    if (!input) { throw std::runtime_error("failed to open prompt ids file: " + path); }
    std::vector<TokenId> ids;
    std::string text((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    std::replace(text.begin(), text.end(), ',', ' ');
    std::istringstream words(text);
    std::string word;
    while (words >> word) { ids.push_back(static_cast<TokenId>(parse_nonnegative(word, "prompt token id"))); }
    if (ids.empty()) { throw std::invalid_argument("prompt ids file is empty: " + path); }
    return ids;
}

std::vector<TokenId> prompt_slice(const std::vector<TokenId>& corpus, int n_prompt) {
    if (n_prompt <= 0 || static_cast<std::size_t>(n_prompt) > corpus.size()) {
        throw std::invalid_argument("invalid prompt slice length: " + std::to_string(n_prompt));
    }
    return {corpus.begin(), corpus.begin() + n_prompt};
}

std::string decode_path_name(bool use_cuda_graph, const SpeculativeOptions& speculative) {
    const std::string execution = use_cuda_graph ? "cuda_graph" : "eager";
    return speculative.backend == SpeculativeBackend::None
               ? execution
               : std::string(product::speculative_backend_name(speculative.backend)) + "_" +
                     execution;
}

std::uint32_t decode_graph_prime_output_tokens(const SpeculativeOptions& speculative) {
    product::validate_speculative_cli_options(speculative);
    const auto widest =
        std::max(product::speculative_verify_drafts(speculative), speculative.ngram_draft_tokens);
    return speculative.backend == SpeculativeBackend::None ? 3 : 2 * (widest + 1) + 1;
}

std::uint32_t decode_graph_prime_required_context(const SpeculativeOptions& speculative) {
    const std::uint64_t outputs = decode_graph_prime_output_tokens(speculative);
    return checked_context(outputs + (speculative.backend == SpeculativeBackend::Mtp
                                          ? 2ULL * speculative.draft_tokens
                                          : 0),
                           "decode graph prime context requirement");
}

Stats compute_stats(const std::vector<double>& values) {
    Stats out;
    out.count = static_cast<int>(values.size());
    if (values.empty()) { return out; }
    for (const double value : values) { out.mean += value; }
    out.mean /= static_cast<double>(values.size());
    if (values.size() > 1) {
        for (const double value : values) { out.stddev += (value - out.mean) * (value - out.mean); }
        out.stddev = std::sqrt(out.stddev / static_cast<double>(values.size() - 1));
    }
    return out;
}

std::vector<double> prefill_tok_s_series(const TestResult& result) {
    std::vector<double> out;
    if (!result.test.has_prefill()) { return out; }
    for (const RepTiming& rep : result.reps) {
        if (rep.timings.prefill_seconds > 0.0) {
            out.push_back(static_cast<double>(result.test.n_prompt) / rep.timings.prefill_seconds);
        }
    }
    return out;
}

std::vector<double> decode_output_tok_s_series(const TestResult& result) {
    std::vector<double> out;
    if (!result.test.has_decode()) { return out; }
    for (const RepTiming& rep : result.reps) {
        if (rep.timings.decode_seconds > 0.0) {
            const auto tokens = rep.generated_output_tokens ? rep.generated_output_tokens - 1U : 0U;
            out.push_back(static_cast<double>(tokens) / rep.timings.decode_seconds);
        }
    }
    return out;
}

std::vector<double> decode_engine_tok_s_series(const TestResult& result) {
    std::vector<double> out;
    if (!result.test.has_decode()) { return out; }
    for (const RepTiming& rep : result.reps) {
        if (rep.timings.decode_seconds > 0.0) {
            out.push_back(static_cast<double>(decode_engine_tokens(result, rep)) /
                          rep.timings.decode_seconds);
        }
    }
    return out;
}

std::vector<double> output_tok_s_series(const TestResult& result) {
    std::vector<double> out;
    for (std::size_t i = 0; i < result.repetition_wall_seconds.size(); ++i) {
        std::uint64_t tokens = 0;
        for (std::uint32_t row = 0; row < result.concurrency; ++row) {
            tokens += result.reps.at(i * result.concurrency + row).generated_output_tokens;
        }
        out.push_back(static_cast<double>(tokens) / result.repetition_wall_seconds[i]);
    }
    return out;
}

template <double GenerationTimings::*Field>
std::vector<double> timing_series(const TestResult& result) {
    std::vector<double> out;
    out.reserve(result.reps.size());
    for (const RepTiming& rep : result.reps) { out.push_back(rep.timings.*Field); }
    return out;
}

std::vector<double> prepare_time_series(const TestResult& result) {
    return timing_series<&GenerationTimings::prepare_seconds>(result);
}

std::vector<double> prefill_time_series(const TestResult& result) {
    return result.test.has_prefill() ? timing_series<&GenerationTimings::prefill_seconds>(result)
                                     : std::vector<double>{};
}

std::vector<double> decode_time_series(const TestResult& result) {
    return result.test.has_decode() ? timing_series<&GenerationTimings::decode_seconds>(result)
                                    : std::vector<double>{};
}

std::vector<double> total_time_series(const TestResult& result) {
    return timing_series<&GenerationTimings::total_seconds>(result);
}

std::string format_table(const BenchEnvironment& env, const std::vector<TestResult>& results) {
    std::ostringstream out;
    out << "infernix_bench product throughput report\n"
        << "  architecture: " << env.load.architecture << '\n'
        << "  model:      " << env.load.model_name << '\n'
        << "  gpu:        " << env.gpu_name << " (device " << env.device_id << ")\n"
        << "  cuda:       runtime " << env.cuda_runtime_version << ", driver "
        << env.cuda_driver_version << '\n'
        << "  machine:    " << machine_text(env.machine) << '\n'
        << "  artifact:   " << env.artifact_path << " (" << env.artifact_file_size_bytes
        << " bytes)\n"
        << "  load:       " << env.load.load_seconds << " s, upload " << env.load.upload_seconds
        << " s, read " << format_bytes(env.load.artifact_bytes_read) << ", H2D "
        << format_bytes(env.load.host_to_device_bytes) << ", staging peak "
        << format_bytes(env.load.peak_staging_bytes) << '\n'
        << "  memory:     weights " << format_bytes(env.memory.weights.capacity_bytes)
        << ", sequence " << format_bytes(env.memory.sequence.capacity_bytes) << ", workspace "
        << format_bytes(env.memory.workspace.capacity_bytes) << ", Vision handoff "
        << (env.memory.vision_workspace
                ? format_bytes(env.memory.vision_workspace->handoff_capacity_bytes)
                : "disabled")
        << " (within workspace), graph allowance "
        << format_bytes(env.memory.cuda_graph_allowance_bytes) << ", KV payload "
        << format_bytes(env.memory.kv_payload_bytes) << '\n'
        << "  corpus:     " << env.corpus_path << " (" << env.corpus_tokens << " tokens)\n"
        << "  config:     max_context=" << env.max_context << " prefill_chunk=" << env.prefill_chunk
        << " concurrency=" << env.concurrency << " constraint=" << constraint_name(env.constraint)
        << (env.mixed_constraints ? " (mixed)" : "")
        << " original_int8_prefill_kernel=" << (env.original_int8_prefill_kernel ? "on" : "off")
        << " prefill_split_workspace_mib=" << env.prefill_split_workspace_mib
        << " original_nvfp4_prefill_kernel=" << (env.original_nvfp4_prefill_kernel ? "on" : "off")
        << " fast_paths=" << fast_path_list(env) << " temporary_switches=\""
        << env.load.temporary_switches << '"' << " expert_tail_skip=" << env.expert_tail_skip
        << " expert_a4_activations=" << (env.expert_a4_activations ? "on" : "off")
        << " ngram_volume=" << (env.ngram_volume_path.empty() ? "default" : env.ngram_volume_path)
        << " ngram_residency=" << ngram_residency_name(env.ngram_residency)
        << " expert_stripe=" << (env.expert_stripe_path.empty() ? "none" : env.expert_stripe_path)
        << " expert_ram_mib=" << mib_or_auto(env.expert_ram_bytes)
        << " expert_pin_mib=" << mib_or_auto(env.expert_pin_bytes)
        << " rope_yarn_factor=" << env.rope_yarn_factor
        << " kv_cache=" << kv_cache_name(env.kv_cache)
        << " logits_dtype=" << product::logits_dtype_name(env.logits_dtype)
        << " spec=" << product::speculative_backend_name(env.speculative.backend)
        << " draft_tokens=" << env.speculative.draft_tokens
        << " draft_auto=" << (env.speculative.draft_tokens_auto ? "on" : "off")
        << " draft_tree=" << product::draft_tree_nodes_text(env.speculative) << '/'
        << env.speculative.draft_tree_paths
        << " ngram_draft_tokens=" << env.speculative.ngram_draft_tokens
        << " ngram_min_match=" << env.speculative.ngram_min_match
        << " spec_min_p=" << product::draft_min_p_text(env.speculative)
        << " proposal_head=" << proposal_head_name(env.speculative.proposal_head)
        << " decode_path=" << decode_path_name(env.use_cuda_graph, env.speculative)
        << " graph_prime="
        << (env.decode_graph_primed
                ? std::to_string(env.decode_graph_prime_output_tokens) + " outputs"
                : "n/a")
        << " repetitions=" << env.repetitions << " warmup=" << env.warmup << "\n\n";

    constexpr std::size_t cols                   = 11;
    const std::array<std::string, cols> headings = {
        "test",           "n_prompt",      "n_gen",    "prefill t/s",   "decode out t/s",
        "decode eng t/s", "batch out t/s", "spec acc", "spec round/fb", "ngram acc/draft/round",
        "work peak"};
    std::vector<std::array<std::string, cols>> rows;
    for (const TestResult& result : results) {
        const SpeculativeStats spec  = aggregate_speculative(result);
        const std::string acceptance = spec.drafted_tokens == 0
                                           ? "n/a"
                                           : number(static_cast<double>(spec.accepted_tokens) /
                                                    static_cast<double>(spec.drafted_tokens));
        const std::string rounds =
            spec.enabled ? std::to_string(spec.rounds) + "/" + std::to_string(spec.fallback_steps)
                         : "n/a";
        rows.push_back({result.test.label, std::to_string(result.test.n_prompt),
                        std::to_string(result.test.n_gen), rate_cell(prefill_tok_s_series(result)),
                        rate_cell(decode_output_tok_s_series(result)),
                        rate_cell(decode_engine_tok_s_series(result)),
                        rate_cell(output_tok_s_series(result)), acceptance, rounds,
                        std::to_string(spec.ngram_accepted_tokens) + "/" +
                            std::to_string(spec.ngram_drafted_tokens) + "/" +
                            std::to_string(spec.ngram_rounds),
                        format_bytes(result.workspace_peak_bytes)});
    }

    std::array<std::size_t, cols> widths{};
    for (std::size_t i = 0; i < cols; ++i) { widths[i] = headings[i].size(); }
    for (const auto& row : rows) {
        for (std::size_t i = 0; i < cols; ++i) { widths[i] = std::max(widths[i], row[i].size()); }
    }
    const auto print_row = [&](const auto& row) {
        for (std::size_t i = 0; i < cols; ++i) {
            if (i != 0) { out << "  "; }
            out << (i == 0 ? std::left : std::right) << std::setw(static_cast<int>(widths[i]))
                << row[i];
        }
        out << '\n';
    };
    print_row(headings);
    for (const auto& row : rows) { print_row(row); }
    return out.str();
}

std::string format_json(const BenchEnvironment& env, const std::string& command,
                        const std::vector<TestResult>& results) {
    std::ostringstream formats;
    formats << '[';
    for (std::size_t i = 0; i < env.load.weight_formats.size(); ++i) {
        if (i != 0) { formats << ','; }
        formats << '"' << json_escape(env.load.weight_formats[i]) << '"';
    }
    formats << ']';

    std::ostringstream out;
    out << "{\n"
        << "  \"schema_version\": " << kSchemaVersion << ",\n"
        << "  \"artifact_type\": \"" << kArtifactType << "\",\n"
        << "  \"tool\": \"infernix_bench\",\n"
        << "  \"command\": \"" << json_escape(command) << "\",\n"
        << "  \"environment\": {\"gpu_name\": \"" << json_escape(env.gpu_name)
        << "\", \"cuda_runtime_version\": \"" << json_escape(env.cuda_runtime_version)
        << "\", \"cuda_driver_version\": \"" << json_escape(env.cuda_driver_version)
        << "\", \"device_id\": " << env.device_id << "},\n"
        << "  \"machine\": " << machine_json(env.machine) << ",\n"
        << "  \"artifact\": {\"path\": \"" << json_escape(env.artifact_path)
        << "\", \"file_size_bytes\": " << env.artifact_file_size_bytes << "},\n"
        << "  \"load\": {\n"
        << "    \"architecture\": \"" << json_escape(env.load.architecture) << "\",\n"
        << "    \"name\": \"" << json_escape(env.load.model_name) << "\",\n"
        << "    \"formats\": " << formats.str() << ",\n"
        << "    \"prefill_signature\": \"" << json_escape(env.load.prefill_signature) << "\",\n"
        << "    \"load_seconds\": " << number(env.load.load_seconds) << ",\n"
        << "    \"upload_seconds\": " << number(env.load.upload_seconds) << ",\n"
        << "    \"artifact_bytes_read\": " << env.load.artifact_bytes_read << ",\n"
        << "    \"host_to_device_bytes\": " << env.load.host_to_device_bytes << ",\n"
        << "    \"peak_staging_bytes\": " << env.load.peak_staging_bytes << ",\n"
        << "    \"device_object_count\": " << env.load.device_object_count << ",\n"
        << "    \"host_object_count\": " << env.load.host_object_count << "\n"
        << "  },\n"
        << "  \"memory\": {\n"
        << "    \"device\": " << env.memory.device << ",\n"
        << "    \"max_context\": " << env.memory.max_context << ",\n"
        << "    \"kv_capacity_mode\": \""
        << (env.memory.kv_capacity_mode == KvCapacityMode::Automatic ? "auto" : "explicit")
        << "\",\n"
        << "    \"kv_capacity\": " << env.memory.kv_capacity << ",\n"
        << "    \"kv_capacity_page_groups\": " << env.memory.kv_capacity_page_groups << ",\n"
        << "    \"kv_capacity_max_page_groups\": " << env.memory.kv_capacity_max_page_groups
        << ",\n"
        << "    \"kv_cache\": \"" << kv_cache_name(env.memory.kv_cache) << "\",\n";
    append_arena_json(out, "weights", env.memory.weights, "    ", true);
    append_arena_json(out, "sequence", env.memory.sequence, "    ", true);
    append_arena_json(out, "workspace", env.memory.workspace, "    ", true);
    append_vision_workspace_json(out, env.memory.vision_workspace, "    ", true);
    out << "    \"minimum_runtime_reservation_bytes\": "
        << env.memory.minimum_runtime_reservation_bytes << ",\n"
        << "    \"kv_capacity_increment_bytes\": " << env.memory.kv_capacity_increment_bytes
        << ",\n"
        << "    \"runtime_reservation_bytes\": " << env.memory.runtime_reservation_bytes << ",\n"
        << "    \"available_after_weights_bytes\": " << env.memory.available_after_weights_bytes
        << ",\n"
        << "    \"available_after_startup_bytes\": " << env.memory.available_after_startup_bytes
        << ",\n"
        << "    \"vram_headroom_bytes\": " << env.memory.vram_headroom_bytes << ",\n"
        << "    \"planned_slack_bytes\": " << env.memory.planned_slack_bytes << ",\n"
        << "    \"cuda_graph_allowance_bytes\": " << env.memory.cuda_graph_allowance_bytes << ",\n"
        << "    \"kv_payload_bytes\": " << env.memory.kv_payload_bytes << "\n"
        << "  },\n"
        << "  \"config\": {\n"
        << "    \"concurrency\": " << env.concurrency << ",\n"
        << "    \"constraint_file\": \"" << json_escape(env.constraint_file) << "\",\n"
        << "    \"constraint_type\": \"" << constraint_name(env.constraint) << "\",\n"
        << "    \"constraint_source\": \""
        << json_escape(env.constraint ? env.constraint->source : "") << "\",\n"
        << "    \"constraint_choices\": " << constraint_choices(env.constraint) << ",\n"
        << "    \"mixed_constraints\": " << (env.mixed_constraints ? "true" : "false") << ",\n"
        << "    \"max_context\": " << env.max_context << ",\n"
        << "    \"rope_yarn_factor\": " << env.rope_yarn_factor << ",\n"
        << "    \"prefill_chunk\": " << env.prefill_chunk << ",\n"
        << "    \"original_int8_prefill_kernel\": "
        << (env.original_int8_prefill_kernel ? "true" : "false") << ",\n"
        << "    \"prefill_split_workspace_mib\": " << env.prefill_split_workspace_mib << ",\n"
        << "    \"original_nvfp4_prefill_kernel\": "
        << (env.original_nvfp4_prefill_kernel ? "true" : "false") << ",\n"
        << "    \"fast_paths\": \"" << fast_path_list(env) << "\",\n"
        << "    \"temporary_switches\": \"" << json_escape(env.load.temporary_switches) << "\",\n"
        << "    \"expert_tail_skip\": " << env.expert_tail_skip << ",\n"
        << "    \"expert_a4_activations\": " << (env.expert_a4_activations ? "true" : "false") << ",\n"
        << "    \"ngram_volume\": \"" << json_escape(env.ngram_volume_path) << "\",\n"
        << "    \"ngram_residency\": \"" << ngram_residency_name(env.ngram_residency) << "\",\n"
        << "    \"expert_stripe\": \"" << json_escape(env.expert_stripe_path) << "\",\n"
        << "    \"expert_ram_mib\": "
        << (env.expert_ram_bytes ? std::to_string(*env.expert_ram_bytes >> 20) : "null") << ",\n"
        << "    \"expert_pin_mib\": "
        << (env.expert_pin_bytes ? std::to_string(*env.expert_pin_bytes >> 20) : "null") << ",\n"
        << "    \"kv_cache\": \"" << kv_cache_name(env.kv_cache) << "\",\n"
        << "    \"logits_dtype\": \"" << product::logits_dtype_name(env.logits_dtype) << "\",\n"
        << "    \"speculative_backend\": \""
        << product::speculative_backend_name(env.speculative.backend) << "\",\n"
        << "    \"draft_tokens\": " << env.speculative.draft_tokens << ",\n"
        << "    \"draft_tokens_auto\": " << (env.speculative.draft_tokens_auto ? "true" : "false") << ",\n"
        << "    \"ngram_draft_tokens\": " << env.speculative.ngram_draft_tokens << ",\n"
        << "    \"ngram_min_match\": " << env.speculative.ngram_min_match << ",\n"
        << "    \"spec_min_p\": "
        << (env.speculative.draft_min_p ? product::draft_min_p_text(env.speculative) : "null")
        << ",\n"
        << "    \"proposal_head\": \"" << proposal_head_name(env.speculative.proposal_head)
        << "\",\n"
        << "    \"use_cuda_graph\": " << (env.use_cuda_graph ? "true" : "false") << ",\n"
        << "    \"decode_path\": \"" << decode_path_name(env.use_cuda_graph, env.speculative)
        << "\",\n"
        << "    \"decode_graph_prime\": {\"primed\": "
        << (env.decode_graph_primed ? "true" : "false")
        << ", \"output_tokens\": " << env.decode_graph_prime_output_tokens << "},\n"
        << "    \"repetitions\": " << env.repetitions << ",\n"
        << "    \"warmup\": " << env.warmup << ",\n"
        << "    \"corpus_path\": \"" << json_escape(env.corpus_path) << "\",\n"
        << "    \"corpus_tokens\": " << env.corpus_tokens << ",\n"
        << "    \"corpus_offset\": " << env.corpus_offset << ",\n"
        << "    \"prompt_ids_path\": "
        << (env.prompt_ids_path.empty() ? std::string("null") : "\"" + json_escape(env.prompt_ids_path) + "\"") << ",\n"
        << "    \"sampling\": {\"temperature\": " << (env.temperature ? number(*env.temperature) : "0")
        << ", \"top_p\": " << (env.top_p ? number(*env.top_p) : "null")
        << ", \"top_k\": " << (env.top_k ? std::to_string(*env.top_k) : "null")
        << ", \"min_p\": " << (env.min_p ? number(*env.min_p) : "null") << ", \"seed\": " << env.seed << "},\n"
        << "    \"stop_at_eos\": " << (env.stop_at_eos ? "true" : "false") << ",\n"
        << "    \"route_trace_path\": "
        << (env.route_trace_path.empty() ? std::string("null") : "\"" + json_escape(env.route_trace_path) + "\"")
        << "\n"
        << "  },\n"
        << "  \"tests\": [\n";

    for (std::size_t i = 0; i < results.size(); ++i) {
        const TestResult& result = results[i];
        out << "    {\n"
            << "      \"label\": \"" << json_escape(result.test.label) << "\",\n"
            << "      \"kind\": \"" << kind_string(result.test.kind) << "\",\n"
            << "      \"n_prompt\": " << result.test.n_prompt << ",\n"
            << "      \"n_gen\": " << result.test.n_gen << ",\n"
            << "      \"requested_output_tokens\": " << result.test.requested_output_tokens()
            << ",\n";
        append_stat(out, "prefill_tok_s", prefill_tok_s_series(result), "      ");
        out << ",\n";
        append_stat(out, "decode_output_tok_s", decode_output_tok_s_series(result), "      ");
        out << ",\n";
        append_stat(out, "decode_engine_tok_s", decode_engine_tok_s_series(result), "      ");
        out << ",\n";
        append_stat(out, "output_tok_s", output_tok_s_series(result), "      ");
        out << ",\n";
        append_stat(out, "prepare_seconds", prepare_time_series(result), "      ");
        out << ",\n";
        append_stat(out, "prefill_seconds", prefill_time_series(result), "      ");
        out << ",\n";
        append_stat(out, "decode_seconds", decode_time_series(result), "      ");
        out << ",\n";
        append_stat(out, "total_seconds", total_time_series(result), "      ");
        out << ",\n      \"workspace_peak_bytes\": " << result.workspace_peak_bytes
            << ",\n      \"workspace_allocator_peak_bytes\": "
            << result.workspace_allocator_peak_bytes << ",\n"
            << "      \"execution\": " << execution_json(result.execution) << ",\n";
        append_speculative_json(out, aggregate_speculative(result), "      ");
        out << ",\n      \"repetition_wall_seconds\": [";
        for (std::size_t r = 0; r < result.repetition_wall_seconds.size(); ++r) {
            if (r) { out << ", "; }
            out << number(result.repetition_wall_seconds[r]);
        }
        out << "],\n      \"reps\": [\n";
        for (std::size_t r = 0; r < result.reps.size(); ++r) {
            const RepTiming& rep = result.reps[r];
            out << "        {\n"
                << "          \"repetition\": " << r / result.concurrency << ",\n"
                << "          \"row\": " << r % result.concurrency << ",\n"
                << "          \"generated_output_tokens\": " << rep.generated_output_tokens << ",\n"
                << "          \"decode_output_tokens\": "
                << (result.test.has_decode()
                        ? std::to_string(
                              rep.generated_output_tokens ? rep.generated_output_tokens - 1U : 0U)
                        : "null")
                << ",\n"
                << "          \"decode_engine_tokens\": "
                << (result.test.has_decode() ? std::to_string(decode_engine_tokens(result, rep))
                                             : "null")
                << ",\n";
            out << "          \"constraint\": "
                << product::constraint_observation_json(rep.constraint).dump() << ",\n"
                << "          \"constraint_draft_wait_exposed_seconds\": "
                << number(rep.constraint_draft_wait_exposed_seconds) << ",\n";
            append_timings_json(out, rep.timings, "          ");
            out << ",\n";
            append_speculative_json(out, rep.speculative, "          ");
            out << "\n        }" << (r + 1 == result.reps.size() ? "" : ",") << '\n';
        }
        out << "      ]\n    }" << (i + 1 == results.size() ? "" : ",") << '\n';
    }
    out << "  ]\n}\n";
    return out.str();
}

namespace {
std::string csv_field(std::string_view value) {
    std::string out = "\"";
    for (const char ch : value) {
        if (ch == '"') { out.push_back('"'); }
        out.push_back(ch);
    }
    out.push_back('"');
    return out;
}
} // namespace

std::string format_csv(const BenchEnvironment& env, const std::vector<TestResult>& results) {
    std::ostringstream out;
    out << "label,kind,n_prompt,n_gen,architecture,prefill_signature,model_name,artifact_path,max_"
           "context,prefill_chunk,concurrency,constraint_type,constraint_file,constraint_source,"
           "constraint_choices,mixed_constraints,rope_yarn_factor,"
           "speculative_"
           "backend,draft_tokens,ngram_draft_tokens,ngram_min_match,spec_min_p,"
           "proposal_head,decode_path,kv_cache,logits_dtype,kv_payload_bytes,load_host_to_device_bytes,"
           "weights_capacity_bytes,sequence_capacity_bytes,workspace_capacity_bytes,"
           "workspace_general_capacity_bytes,vision_handoff_capacity_bytes,"
           "cuda_graph_allowance_bytes,"
           "workspace_peak_bytes,workspace_allocator_peak_bytes,"
           "spec_rounds,spec_fallback_steps,spec_acceptance_rate,"
           "ngram_rounds,ngram_drafted_tokens,ngram_accepted_tokens,"
           "repetitions,prefill_tok_s_mean,prefill_tok_s_stddev,decode_output_tok_s_mean,"
           "decode_output_tok_s_stddev,decode_engine_tok_s_mean,decode_engine_tok_s_stddev,"
           "prepare_seconds_mean,prefill_seconds_mean,decode_seconds_mean,total_seconds_mean,"
           "output_tok_s_mean,output_tok_s_stddev\n";
    const auto mean = [](const std::vector<double>& values) {
        return values.empty() ? std::string() : number(compute_stats(values).mean);
    };
    const auto stddev = [](const std::vector<double>& values) {
        return values.empty() ? std::string() : number(compute_stats(values).stddev);
    };
    for (const TestResult& result : results) {
        const SpeculativeStats spec  = aggregate_speculative(result);
        const std::string acceptance = spec.drafted_tokens == 0
                                           ? std::string()
                                           : number(static_cast<double>(spec.accepted_tokens) /
                                                    static_cast<double>(spec.drafted_tokens));
        out << result.test.label << ',' << kind_string(result.test.kind) << ','
            << result.test.n_prompt << ',' << result.test.n_gen << ',' << env.load.architecture
            << ',' << env.load.prefill_signature << ',' << csv_field(env.load.model_name) << ','
            << csv_field(env.artifact_path) << ',' << env.max_context << ',' << env.prefill_chunk
            << ',' << env.concurrency << ',' << constraint_name(env.constraint) << ','
            << csv_field(env.constraint_file) << ','
            << csv_field(env.constraint ? env.constraint->source : "") << ','
            << csv_field(constraint_choices(env.constraint)) << ','
            << (env.mixed_constraints ? "true" : "false") << ',' << env.rope_yarn_factor << ','
            << product::speculative_backend_name(env.speculative.backend) << ','
            << env.speculative.draft_tokens << ',' << env.speculative.ngram_draft_tokens << ','
            << env.speculative.ngram_min_match << ','
            << (env.speculative.draft_min_p ? product::draft_min_p_text(env.speculative) : "") << ','
            << proposal_head_name(env.speculative.proposal_head) << ','
            << decode_path_name(env.use_cuda_graph, env.speculative) << ','
            << kv_cache_name(env.kv_cache) << ',' << product::logits_dtype_name(env.logits_dtype) << ','
            << env.memory.kv_payload_bytes << ','
            << env.load.host_to_device_bytes << ',' << env.memory.weights.capacity_bytes << ','
            << env.memory.sequence.capacity_bytes << ',' << env.memory.workspace.capacity_bytes
            << ','
            << (env.memory.vision_workspace
                    ? std::to_string(env.memory.vision_workspace->general_capacity_bytes)
                    : std::to_string(env.memory.workspace.capacity_bytes))
            << ','
            << (env.memory.vision_workspace
                    ? std::to_string(env.memory.vision_workspace->handoff_capacity_bytes)
                    : std::string())
            << ',' << env.memory.cuda_graph_allowance_bytes << ',' << result.workspace_peak_bytes
            << ',' << result.workspace_allocator_peak_bytes << ',' << spec.rounds << ','
            << spec.fallback_steps << ',' << acceptance << ',' << spec.ngram_rounds << ','
            << spec.ngram_drafted_tokens << ',' << spec.ngram_accepted_tokens << ','
            << result.reps.size() / result.concurrency << ',' << mean(prefill_tok_s_series(result))
            << ',' << stddev(prefill_tok_s_series(result)) << ','
            << mean(decode_output_tok_s_series(result)) << ','
            << stddev(decode_output_tok_s_series(result)) << ','
            << mean(decode_engine_tok_s_series(result)) << ','
            << stddev(decode_engine_tok_s_series(result)) << ','
            << mean(prepare_time_series(result)) << ',' << mean(prefill_time_series(result)) << ','
            << mean(decode_time_series(result)) << ',' << mean(total_time_series(result)) << ','
            << mean(output_tok_s_series(result)) << ',' << stddev(output_tok_s_series(result))
            << '\n';
    }
    return out.str();
}

std::string json_escape(std::string_view value) {
    std::string out;
    for (const char c : value) {
        switch (c) {
        case '"':
            out += "\\\"";
            break;
        case '\\':
            out += "\\\\";
            break;
        case '\n':
            out += "\\n";
            break;
        case '\r':
            out += "\\r";
            break;
        case '\t':
            out += "\\t";
            break;
        default:
            if (static_cast<unsigned char>(c) < 0x20) {
                std::ostringstream escaped;
                escaped << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<int>(static_cast<unsigned char>(c));
                out += escaped.str();
            } else {
                out += c;
            }
        }
    }
    return out;
}

std::string kv_cache_name(KvCacheStorage storage) {
    switch (storage) {
    case KvCacheStorage::BFloat16:
        return "bf16";
    case KvCacheStorage::Int8Group64:
        return "int8-group64";
    case KvCacheStorage::Fp8E4M3Row256:
        return "fp8-e4m3-row256";
    case KvCacheStorage::Nvfp4Group16:
        return "nvfp4";
    case KvCacheStorage::Fp8KeyNvfp4Value:
        return "k8v4";
    case KvCacheStorage::Vq2:
        return "vq2";
    case KvCacheStorage::Q4KeyVq2Value:
        return "k4v2";
    }
    return "unknown";
}

std::string proposal_head_name(ProposalHead head) {
    switch (head) {
    case ProposalHead::Full:
        return "full";
    case ProposalHead::Optimized:
        return "optimized";
    }
    return "unknown";
}

std::uint64_t file_size_or_zero(const std::string& path) {
    std::error_code error;
    const auto size = std::filesystem::file_size(path, error);
    return error ? 0 : static_cast<std::uint64_t>(size);
}

RuntimeExecutionLatency execution_delta(const RuntimeExecutionLatency& before,
                                        const RuntimeExecutionLatency& after) {
    const auto delta = [](const DurationCounter& b, const DurationCounter& a) {
        return DurationCounter{a.count - b.count, a.total_ns - b.total_ns, a.max_ns};
    };
    RuntimeExecutionLatency out;
    out.prefill = delta(before.prefill, after.prefill);
    out.replay  = delta(before.replay, after.replay);
    out.control = delta(before.control, after.control);
    out.decode  = delta(before.decode, after.decode);
    for (std::size_t b = 0; b < out.decode_by_batch.size(); ++b) {
        out.decode_by_batch[b] = delta(before.decode_by_batch[b], after.decode_by_batch[b]);
    }
    return out;
}

double measure_ram_read_bytes_per_second(std::size_t bytes, std::uint32_t threads) {
    const std::size_t words = bytes / sizeof(std::uint64_t);
    if (words == 0 || threads == 0) { return 0.0; }
    std::vector<std::uint64_t> data(words);
    // One cache line per thread's sum, so the threads do not share a line.
    std::vector<std::uint64_t> sums(static_cast<std::size_t>(threads) * 8, 0);
    const auto run = [&](auto&& work) {
        std::vector<std::thread> workers;
        workers.reserve(threads);
        for (std::uint32_t t = 0; t < threads; ++t) {
            const std::size_t begin = words * t / threads;
            const std::size_t end   = words * (t + 1) / threads;
            workers.emplace_back([&work, &sums, t, begin, end] { work(sums[t * 8], begin, end); });
        }
        for (std::thread& worker : workers) { worker.join(); }
    };
    run([&](std::uint64_t&, std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i) { data[i] = i; }
    });
    double best = 0.0;
    for (int pass = 0; pass < 3; ++pass) {
        const auto started = std::chrono::steady_clock::now();
        run([&](std::uint64_t& sum, std::size_t begin, std::size_t end) {
            std::uint64_t local = 0;
            for (std::size_t i = begin; i < end; ++i) { local += data[i]; }
            sum = local;
        });
        const double seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        if (seconds > 0.0) {
            best = std::max(best, static_cast<double>(words * sizeof(std::uint64_t)) / seconds);
        }
    }
    // The sums are read, so the passes are not optimized away.
    volatile std::uint64_t sink = 0;
    for (std::uint32_t t = 0; t < threads; ++t) { sink = sink + sums[t * 8]; }
    return best;
}

} // namespace infernix::bench
