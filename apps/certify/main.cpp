// infernix-certify: the IQS-2 measurement tool (docs/maintainer/model-quality.md §6). One reference pass per
// window serves every arm: the reference scores each window's positions once (the lossless route, the
// design reference or the noise twin), a prefill arm scores the same windows on a shipped route, and a
// decode arm forces each window's continuation through the decode path at the set's decode starts, every
// forced step observing the reference's top tokens at that position. Arms write one row per position
// (KL over the reference's top tokens plus one bucket for the rest, top-1 flip, NLL change), which
// tools/quality/iqs.py turns into bounds and verdicts.
//
//   infernix-certify reference <artifact> --corpus M --set S --out REF [--design --kv-dtype K]
//                    [--noise-seed N [--noise-units LIST] [--noise-scale S]] [--windows i,j,...]
//                    [--logits-dtype fp32|bf16] (the head's exact sums correctly rounded to it; default fp32)
//   infernix-certify verify    <artifact> --reference REF [--windows i,j,...]
//   infernix-certify capture   <artifact> --corpus M --set S --out PREFIX --windows i,j [--tokens N] [--every K]
//   infernix-certify mask      <artifact> --corpus M --set S --out FILE.json
//   infernix-certify prefill   <artifact> --corpus M --set S --reference REF --out PREFIX [route options]
//   infernix-certify decode    <artifact> --corpus M --set S --reference REF --out PREFIX [route options]
//                    [--windows i,j,...]
//   --reference may repeat: a reference written in parts (reference --windows), each window in one part. Every
//   command scores the windows --windows selects (all by default) and needs those windows in the reference.
//                    [--concurrency N] [--spec mtp|dflash|dflash2] [--draft-tokens N|auto]
//                    [--ngram-draft-tokens N|auto] [--draft-tree-nodes auto|LIST] [--draft-model FILE]
//                    [--prefix-reuse [--warm-tokens N]] [--kv-tokens N]
//   route options: [--reproducible] [--fast-path NAME]... [--kv-dtype K] [--logits-dtype fp32|bf16]
//                  [--prefill-chunk N] [--ngram-volume FILE] [--expert-a4-activations]
//
// llama.cpp's KL-divergence measurement (llama_kld.h): the same exact reference as llama-perplexity's
// `_logits_` base file, which llama-perplexity --kl-divergence scores llama.cpp against, and arms that score
// Infernix against it with llama.cpp's own definitions and summary.
//   infernix-certify reference <artifact> --format llama-kld --out FILE (--text F [--context N] [--chunks N] |
//                    --tokens-from BASE) [--design --kv-dtype K] [--logits-dtype fp32|bf16]
//   infernix-certify prefill   <artifact> --llama-base FILE --out PREFIX [route options | --lossless [--noise-seed N]]
//   infernix-certify decode    <artifact> --llama-base FILE --out PREFIX [route options] [--spec ...]
//   infernix-certify llama-compare --llama-base FILE --logits OTHER --out PREFIX
//   --text is tokenized whole and cut into --context-token chunks (default 512, the first --chunks, default 100;
//   0 takes every whole chunk), as llama-perplexity cuts its -f file; --tokens-from takes another base file's
//   tokens (llama.cpp's tokenization). llama-compare scores a base file llama-perplexity saved of its own logits.

#include "infernix_build_id.h"
#include "distribution.h"
#include "iqs_corpus.h"
#include "llama_kld.h"
#include "infernix/engine.h"
#include "product/logits_options.h"
#include "product/speculative_options.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {

using infernix::TokenId;
using nlohmann::json;
namespace perplexity = infernix::perplexity;
namespace certify    = infernix::certify;
namespace llama      = infernix::certify::llama;

// IQS-2 metrics: KL over the reference's 256 most probable tokens plus one bucket for the rest, in
// prefill (scoring candidates) and decode (observed tokens).
constexpr std::uint32_t kTop = 256;
static_assert(kTop <= infernix::kMaximumScoreTopTokens && kTop <= infernix::kMaximumObservedTokens);

struct Options {
    std::string command;
    std::string artifact;
    std::filesystem::path corpus;
    std::string set = "main";
    std::filesystem::path out;
    std::vector<std::filesystem::path> references; // one reference, possibly in parts (windows split between files)
    bool lossless              = false; // prefill: score on the exact route (lossless, --design, --noise-seed)
    bool design                = false;
    std::uint64_t noise_seed   = 0;
    std::uint32_t noise_units  = 0xFFU; // the operation groups the noise applies to (ops::LosslessNoiseUnit)
    float noise_scale          = 1.0F;  // its strength
    std::string kv             = "bf16";
    std::string logits         = "fp32"; // --logits-dtype, every route and reference
    bool reproducible          = false;
    std::vector<infernix::FastPath> candidates;
    std::optional<std::uint32_t> prefill_chunk;
    std::uint32_t concurrency  = 1;
    // Decode: the speculative options as infernix-serve parses them (no --spec: none).
    infernix::SpeculativeOptions speculative;
    bool prefix_reuse          = false;
    std::uint32_t warm_tokens  = 200;
    std::uint32_t kv_tokens    = 0;
    std::string ngram_volume;
    bool expert_a4_activations = false;
    std::vector<std::uint32_t> windows;
    std::uint32_t capture_tokens = 512;
    std::uint32_t capture_every  = 16;
    // llama.cpp's KL-divergence format: reference --format llama-kld, the arms' --llama-base, llama-compare.
    std::string format = "iqs";
    std::filesystem::path text;
    std::filesystem::path tokens_from;
    std::uint32_t context = 512;
    std::uint32_t chunks  = 100;
    std::filesystem::path llama_base;
    std::filesystem::path logits_file;
};

template <class Integer>
Integer parse_integer(std::string_view text, const char* label) {
    Integer value{};
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size()) {
        throw std::invalid_argument(std::string("invalid ") + label + ": " + std::string(text));
    }
    return value;
}

// A comma-separated list of the noise twin's operation groups (ops::LosslessNoiseUnit): elementwise, attention,
// flash, deltarule, moe, rope, linear, gguf; "all".
std::uint32_t parse_noise_units(std::string_view list) {
    std::uint32_t mask = 0;
    while (!list.empty()) {
        const std::size_t comma = list.find(',');
        const std::string_view name = list.substr(0, comma);
        if (name == "all") { mask |= 0xFFU; }
        else if (name == "elementwise") { mask |= 1U << 0; }
        else if (name == "attention") { mask |= 1U << 1; }
        else if (name == "flash") { mask |= 1U << 2; }
        else if (name == "deltarule") { mask |= 1U << 3; }
        else if (name == "moe") { mask |= 1U << 4; }
        else if (name == "rope") { mask |= 1U << 5; }
        else if (name == "linear") { mask |= 1U << 6; }
        else if (name == "gguf") { mask |= 1U << 7; }
        else { throw std::invalid_argument("--noise-units: unknown group " + std::string(name)); }
        list = comma == std::string_view::npos ? std::string_view{} : list.substr(comma + 1);
    }
    return mask;
}

std::vector<std::uint32_t> parse_list(std::string_view list) {
    std::vector<std::uint32_t> out;
    while (!list.empty()) {
        const std::size_t comma = list.find(',');
        out.push_back(parse_integer<std::uint32_t>(list.substr(0, comma), "window"));
        list = comma == std::string_view::npos ? std::string_view{} : list.substr(comma + 1);
    }
    return out;
}

Options parse(int argc, char** argv) {
    if (argc < 3) {
        throw std::invalid_argument("usage: infernix-certify reference|verify|prefill|decode|capture|mask <artifact> ... | "
                                    "llama-compare ...");
    }
    Options out{.command = argv[1]};
    // llama-compare reads two base files and runs no engine.
    const bool artifact = out.command != "llama-compare";
    if (artifact) { out.artifact = argv[2]; }
    for (int i = artifact ? 3 : 2; i < argc; ++i) {
        const std::string_view option = argv[i];
        const auto value              = [&]() -> std::string_view {
            if (++i >= argc) { throw std::invalid_argument(std::string(option) + " needs a value"); }
            return argv[i];
        };
        if (option == "--corpus") {
            out.corpus = std::string(value());
        } else if (option == "--set") {
            out.set = value();
        } else if (option == "--out") {
            out.out = std::string(value());
        } else if (option == "--reference") {
            out.references.emplace_back(std::string(value()));
        } else if (option == "--lossless") {
            out.lossless = true;
        } else if (option == "--design") {
            out.design = true;
        } else if (option == "--noise-seed") {
            out.noise_seed = parse_integer<std::uint64_t>(value(), "noise seed");
        } else if (option == "--noise-units") {
            out.noise_units = parse_noise_units(value());
        } else if (option == "--noise-scale") {
            out.noise_scale = std::stof(std::string(value()));
        } else if (option == "--kv-dtype") {
            out.kv = value();
        } else if (option == "--logits-dtype") {
            out.logits = std::string(value());
            (void)infernix::product::parse_logits_dtype(out.logits);
        } else if (option == "--reproducible") {
            out.reproducible = true;
        } else if (option == "--fast-path") {
            const std::string_view name = value();
            const std::optional<infernix::FastPath> path = infernix::parse_fast_path(name);
            if (!path) { throw std::invalid_argument("unknown fast path " + std::string(name)); }
            out.candidates.push_back(*path);
        } else if (option == "--prefill-chunk") {
            out.prefill_chunk = parse_integer<std::uint32_t>(value(), "prefill chunk");
        } else if (option == "--concurrency") {
            out.concurrency = parse_integer<std::uint32_t>(value(), "concurrency");
        } else if (option == "--spec") {
            out.speculative.backend = infernix::product::parse_speculative_backend(value());
        } else if (option == "--draft-tokens") {
            infernix::product::apply_draft_tokens(out.speculative, value());
        } else if (option == "--ngram-draft-tokens") {
            infernix::product::apply_ngram_draft_tokens(out.speculative, value());
        } else if (option == "--draft-tree-nodes") {
            infernix::product::apply_draft_tree_nodes(out.speculative, value());
        } else if (option == "--draft-model") {
            out.speculative.draft_model = std::filesystem::path(std::string(value()));
        } else if (option == "--prefix-reuse") {
            out.prefix_reuse = true;
        } else if (option == "--warm-tokens") {
            out.warm_tokens = parse_integer<std::uint32_t>(value(), "warm tokens");
        } else if (option == "--kv-tokens") {
            out.kv_tokens = parse_integer<std::uint32_t>(value(), "KV tokens");
        } else if (option == "--ngram-volume") {
            out.ngram_volume = value();
        } else if (option == "--expert-a4-activations") {
            out.expert_a4_activations = true;
        } else if (option == "--windows") {
            out.windows = parse_list(value());
        } else if (option == "--tokens") {
            out.capture_tokens = parse_integer<std::uint32_t>(value(), "tokens");
        } else if (option == "--every") {
            out.capture_every = parse_integer<std::uint32_t>(value(), "every");
        } else if (option == "--format") {
            out.format = value();
            if (out.format != "iqs" && out.format != "llama-kld") { throw std::invalid_argument("--format takes iqs or llama-kld"); }
        } else if (option == "--text") {
            out.text = std::string(value());
        } else if (option == "--tokens-from") {
            out.tokens_from = std::string(value());
        } else if (option == "--context") {
            out.context = parse_integer<std::uint32_t>(value(), "context");
        } else if (option == "--chunks") {
            out.chunks = parse_integer<std::uint32_t>(value(), "chunks");
        } else if (option == "--llama-base") {
            out.llama_base = std::string(value());
        } else if (option == "--logits") {
            out.logits_file = std::string(value());
        } else {
            throw std::invalid_argument("unknown option " + std::string(option));
        }
    }
    if (out.concurrency == 0 || out.concurrency > 8) { throw std::invalid_argument("--concurrency must be 1..8"); }
    return out;
}

infernix::KvCacheStorage kv_storage(std::string_view name) {
    if (name == "bf16") { return infernix::KvCacheStorage::BFloat16; }
    if (name == "int8") { return infernix::KvCacheStorage::Int8Group64; }
    if (name == "fp8") { return infernix::KvCacheStorage::Fp8E4M3Row256; }
    if (name == "nvfp4") { return infernix::KvCacheStorage::Nvfp4Group16; }
    if (name == "k8v4") { return infernix::KvCacheStorage::Fp8KeyNvfp4Value; }
    if (name == "vq2") { return infernix::KvCacheStorage::Vq2; }
    if (name == "k4v2") { return infernix::KvCacheStorage::Q4KeyVq2Value; }
    throw std::invalid_argument("--kv-dtype must be bf16, int8, fp8, nvfp4, k8v4, vq2 or k4v2");
}

std::string artifact_path(const std::string& artifact) {
    return std::filesystem::absolute(artifact).lexically_normal().string();
}

std::string hex(std::uint64_t value) {
    std::ostringstream out;
    out << std::hex << std::setw(16) << std::setfill('0') << value;
    return out.str();
}

double seconds_since(std::chrono::steady_clock::time_point begin) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
}

void write_json(const std::filesystem::path& path, const json& value) {
    std::ofstream out(path);
    if (!out) { throw std::runtime_error("cannot write " + path.string()); }
    out << value.dump(2) << '\n';
}

json read_json(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) { throw std::runtime_error("cannot read " + path.string()); }
    return json::parse(in);
}

json load_json(const infernix::Engine& engine) {
    const infernix::LoadSummary load = engine.load_summary();
    return json{{"architecture", load.architecture},
                {"model_name", load.model_name},
                {"prefill_signature", load.prefill_signature},
                {"fast_paths", load.fast_paths},
                {"fast_path_configuration", load.fast_path_configuration},
                {"request_invariant", load.request_invariant},
                {"load_seconds", load.load_seconds}};
}

json provenance(const Options& options, const certify::Corpus* corpus) {
    json out{{"tool", "infernix-certify"}, {"command", options.command}};
    if (!options.artifact.empty()) { out["artifact"] = artifact_path(options.artifact); }
#ifdef INFERNIX_BUILD_ID
    out["build"] = INFERNIX_BUILD_ID;
#endif
    if (corpus != nullptr) {
        out["corpus"] = {{"id", corpus->corpus_id},
                         {"manifest", corpus->manifest.string()},
                         {"content_hash", hex(corpus->content_hash)},
                         {"set", corpus->set.name},
                         {"window_tokens", corpus->set.window_tokens}};
    }
    return out;
}

// A route's options shared by the arms.
void apply_route(const Options& options, infernix::EngineOptions& engine_options) {
    engine_options.artifact_path         = options.artifact;
    engine_options.kv_cache              = kv_storage(options.kv);
    engine_options.logits_dtype          = infernix::product::parse_logits_dtype(options.logits);
    engine_options.fast_paths            = !options.reproducible;
    engine_options.fast_path_candidates  = options.candidates;
    engine_options.expert_a4_activations = options.expert_a4_activations;
    if (options.prefill_chunk) { engine_options.prefill_chunk = *options.prefill_chunk; }
    if (!options.ngram_volume.empty()) { engine_options.ngram_volume_path = options.ngram_volume; }
}

bool selected(const Options& options, std::uint32_t window);

// The stored windows of a reference, which may be in parts (each file holds the windows it was written with,
// e.g. a corpus layer added later): indexed like `windows`, filled for the windows this run selects. Every
// stored window must carry the tokens this corpus gives it, and every selected window must be in one part.
std::vector<perplexity::ReferenceWindow> read_reference(const Options& options, const certify::Corpus& corpus,
                                                        const std::vector<certify::Window>& windows) {
    std::vector<perplexity::ReferenceWindow> out(windows.size());
    std::vector<bool> found(windows.size(), false);
    for (const std::filesystem::path& path : options.references) {
        perplexity::ReferenceReader reader(path);
        const perplexity::ReferenceHeader& header = reader.header();
        if (header.corpus_id != corpus.corpus_id || header.mode != corpus.set.name ||
            header.context != corpus.set.window_tokens || header.top_k != kTop) {
            throw std::runtime_error(path.string() + " is a reference for corpus " + header.corpus_id + " set " +
                                     header.mode + ", not this corpus set");
        }
        while (std::optional<perplexity::ReferenceWindow> ref = reader.try_next()) {
            if (ref->stream_index >= windows.size()) {
                throw std::runtime_error(path.string() + " holds window " + std::to_string(ref->stream_index) +
                                         ", beyond this corpus set's " + std::to_string(windows.size()));
            }
            const certify::Window& window = windows[ref->stream_index];
            if (ref->input_hash != window.input_hash || ref->target_begin != 1 || ref->target_end != window.tokens.size()) {
                throw std::runtime_error(path.string() + " window " + std::to_string(ref->stream_index) +
                                         " does not match " + window.id + " (other text or tokenizer)");
            }
            if (found[window.index]) {
                throw std::runtime_error("window " + std::to_string(window.index) + " is in more than one reference part");
            }
            found[window.index] = true;
            if (selected(options, window.index)) { out[window.index] = std::move(*ref); }
        }
    }
    for (const certify::Window& window : windows) {
        if (selected(options, window.index) && !found[window.index]) {
            throw std::runtime_error("no reference part holds window " + std::to_string(window.index) + " (" +
                                     window.id + ")");
        }
    }
    return out;
}

json reference_paths(const Options& options) {
    json out = json::array();
    for (const std::filesystem::path& path : options.references) { out.push_back(std::filesystem::absolute(path).string()); }
    return out;
}

// One scored window of a reference: positions 1 .. window-1 (row r predicts token r + 1).
perplexity::ReferenceWindow score_reference_window(infernix::Engine& engine, const certify::Window& window) {
    infernix::ScoreResult scored =
        engine.score_tokens(window.tokens, 1, infernix::ScoreOptions{.top_k = kTop});
    perplexity::ReferenceWindow out{.stream_index    = window.index,
                                    .target_begin    = 1,
                                    .target_end      = window.tokens.size(),
                                    .input_hash      = window.input_hash,
                                    .target_logprobs = std::move(scored.logprobs),
                                    .top_ids         = std::move(scored.top_ids),
                                    .top_logprobs    = std::move(scored.top_logprobs)};
    return out;
}

infernix::EngineOptions reference_engine(const Options& options, std::uint32_t window_tokens) {
    infernix::EngineOptions engine_options;
    engine_options.artifact_path         = options.artifact;
    engine_options.purpose               = infernix::EnginePurpose::CausalScoring;
    engine_options.max_context           = window_tokens;
    engine_options.kv_capacity           = infernix::KvCapacityPolicy::explicit_capacity(window_tokens);
    engine_options.kv_cache              = options.design ? kv_storage(options.kv) : infernix::KvCacheStorage::BFloat16;
    engine_options.logits_dtype          = infernix::product::parse_logits_dtype(options.logits);
    engine_options.prefill_chunk         = window_tokens;
    engine_options.lossless              = true;
    engine_options.reference             = options.design;
    engine_options.lossless_noise_seed   = options.noise_seed;
    engine_options.lossless_noise_units  = options.noise_units;
    engine_options.lossless_noise_scale  = options.noise_scale;
    engine_options.expert_a4_activations = options.expert_a4_activations;
    engine_options.fast_paths            = false;
    if (!options.ngram_volume.empty()) { engine_options.ngram_volume_path = options.ngram_volume; }
    return engine_options;
}

bool selected(const Options& options, std::uint32_t window) {
    return options.windows.empty() ||
           std::find(options.windows.begin(), options.windows.end(), window) != options.windows.end();
}

// One row per compared position: the arm's per-position output tools/quality/iqs.py reads.
struct RowWriter {
    std::ofstream out;
    std::uint64_t rows = 0, nonfinite = 0;
    explicit RowWriter(const std::filesystem::path& path) : out(path) {
        if (!out) { throw std::runtime_error("cannot write " + path.string()); }
        out << "window,segment,position,kl,top_mass,flip,dnll\n" << std::setprecision(9);
    }
    void add(std::uint32_t window, int segment, std::uint64_t position, const perplexity::PositionDivergence& d) {
        if (!std::isfinite(d.kl) || !std::isfinite(d.delta_nll)) { ++nonfinite; }
        out << window << ',' << segment << ',' << position << ',' << d.kl << ',' << d.reference_top_mass << ','
            << (d.top1_match ? 0 : 1) << ',' << d.delta_nll << '\n';
        ++rows;
    }
};

int run_reference(const Options& options) {
    if (options.out.empty()) { throw std::invalid_argument("reference needs --out"); }
    if (!options.design && options.kv != "bf16") {
        throw std::invalid_argument("the lossless reference uses bf16 KV; a KV format's codec needs --design");
    }
    const certify::Corpus corpus = certify::load_corpus(options.corpus, options.set);
    const auto started           = std::chrono::steady_clock::now();
    infernix::Engine engine(reference_engine(options, corpus.set.window_tokens));
    const std::vector<certify::Window> windows = certify::load_windows(engine, corpus);
    perplexity::ReferenceWriter writer(options.out, perplexity::ReferenceHeader{.corpus_id = corpus.corpus_id,
                                                                                .mode      = corpus.set.name,
                                                                                .context   = corpus.set.window_tokens,
                                                                                .stride    = corpus.set.window_tokens,
                                                                                .top_k     = kTop});
    // --reference REF (with --design): the same pass also measures the design against REF (the lossless
    // reference), its design error, written as an arm (OUT.compare.*), so disclosure costs no second pass.
    std::vector<perplexity::ReferenceWindow> compared;
    std::optional<RowWriter> compare_rows;
    if (!options.references.empty()) {
        if (!options.design) { throw std::invalid_argument("reference --reference compares a design reference: add --design"); }
        compared = read_reference(options, corpus, windows);
        compare_rows.emplace(std::filesystem::path(options.out.string() + ".compare.positions.csv"));
    }
    json window_list = json::array();
    for (std::size_t w = 0; w < windows.size(); ++w) {
        const certify::Window& window = windows[w];
        if (!selected(options, window.index)) { continue; }
        const auto begin = std::chrono::steady_clock::now();
        if (compare_rows) {
            const perplexity::ReferenceWindow& base = compared[w];
            infernix::ScoreResult scored = engine.score_tokens(
                window.tokens, 1, infernix::ScoreOptions{.top_k = kTop, .candidates_per_position = kTop, .candidates = base.top_ids});
            for (std::size_t r = 0; r < scored.logprobs.size(); ++r) {
                compare_rows->add(window.index, -1, r + 1,
                                  perplexity::position_divergence(
                                      std::span<const float>(base.top_logprobs.data() + r * kTop, kTop),
                                      std::span<const float>(scored.candidate_logprobs.data() + r * kTop, kTop),
                                      scored.top_ids[r * kTop] == base.top_ids[r * kTop], base.target_logprobs[r],
                                      scored.logprobs[r]));
            }
            writer.write(perplexity::ReferenceWindow{.stream_index    = window.index,
                                                     .target_begin    = 1,
                                                     .target_end      = window.tokens.size(),
                                                     .input_hash      = window.input_hash,
                                                     .target_logprobs = std::move(scored.logprobs),
                                                     .top_ids         = std::move(scored.top_ids),
                                                     .top_logprobs    = std::move(scored.top_logprobs)});
        } else {
            writer.write(score_reference_window(engine, window));
        }
        window_list.push_back({{"index", window.index}, {"id", window.id}, {"domain", window.domain},
                               {"input_hash", hex(window.input_hash)}, {"seconds", seconds_since(begin)}});
        std::cerr << "reference window " << window.index + 1 << "/" << windows.size() << " (" << window.id << ")\n";
    }
    writer.finish();
    if (compare_rows) {
        json compare         = provenance(options, &corpus);
        compare["arm"]       = "design_error";
        compare["route"]     = "design";
        compare["kv_dtype"]  = options.kv;
        compare["reference"] = reference_paths(options);
        compare["rows"]      = compare_rows->rows;
        compare["nonfinite"] = compare_rows->nonfinite;
        write_json(std::filesystem::path(options.out.string() + ".compare.json"), compare);
    }
    json meta          = provenance(options, &corpus);
    meta["route"]      = options.design ? "design" : "lossless";
    meta["kv_dtype"]   = options.design ? options.kv : "bf16";
    meta["logits_dtype"] = options.logits;
    meta["noise_seed"] = options.noise_seed;
    meta["expert_a4_activations"] = options.expert_a4_activations;
    meta["windows"]    = window_list;
    meta["load"]       = load_json(engine);
    meta["seconds"]    = seconds_since(started);
    write_json(std::filesystem::path(options.out.string() + ".json"), meta);
    return 0;
}

// The release spot check: re-score stored windows with the reference's own settings; any bit that differs
// fails (the lossless route and the design reference never change their bits).
int run_verify(Options options) {
    if (options.references.empty()) { throw std::invalid_argument("verify needs --reference"); }
    // Every part of a reference was written with the same settings; the first part's metadata gives them.
    const json meta = read_json(std::filesystem::path(options.references.front().string() + ".json"));
    options.corpus     = meta.at("corpus").at("manifest").get<std::string>();
    options.set        = meta.at("corpus").at("set").get<std::string>();
    options.design     = meta.at("route") == "design";
    options.kv         = meta.at("kv_dtype").get<std::string>();
    options.logits     = meta.value("logits_dtype", std::string("fp32"));
    options.noise_seed = meta.at("noise_seed").get<std::uint64_t>();
    options.expert_a4_activations = meta.value("expert_a4_activations", false);
    if (options.windows.empty()) { options.windows = {0}; }
    const certify::Corpus corpus = certify::load_corpus(options.corpus, options.set);
    infernix::Engine engine(reference_engine(options, corpus.set.window_tokens));
    const std::vector<certify::Window> windows = certify::load_windows(engine, corpus);
    // A stored window whose text or tokenizer changed fails here, before any scoring.
    const std::vector<perplexity::ReferenceWindow> references = read_reference(options, corpus, windows);
    bool identical = true;
    for (const certify::Window& window : windows) {
        if (!selected(options, window.index)) { continue; }
        const perplexity::ReferenceWindow& stored = references[window.index];
        const perplexity::ReferenceWindow fresh   = score_reference_window(engine, window);
        const auto same = [](const auto& a, const auto& b) {
            return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(a[0])) == 0;
        };
        const bool equal = stored.input_hash == fresh.input_hash && same(stored.target_logprobs, fresh.target_logprobs) &&
                           same(stored.top_ids, fresh.top_ids) && same(stored.top_logprobs, fresh.top_logprobs);
        std::cout << (equal ? "IDENTICAL" : "DIFFERENT") << " window " << window.index << " (" << window.id << ")\n";
        identical = identical && equal;
    }
    return identical ? 0 : 1;
}

// Reference conformance (IQS-2 G1): the lossless route's whole FP32 logits at every --every'th position of the
// first --tokens tokens of the chosen windows, for tools/quality/shadow to compare bit for bit with the
// publisher's model code run in exact arithmetic. PREFIX.json holds the tokens and positions, PREFIX.f32 the
// logits rows (little-endian binary32, one vocabulary-sized row per position, in order).
int run_capture(const Options& options, std::uint32_t tokens, std::uint32_t every) {
    if (options.out.empty()) { throw std::invalid_argument("capture needs --out"); }
    const certify::Corpus corpus = certify::load_corpus(options.corpus, options.set);
    Options exact                = options;
    exact.design                 = false;
    infernix::Engine engine(reference_engine(exact, std::max(tokens, 2U)));
    const std::vector<certify::Window> windows = certify::load_windows(engine, corpus);
    std::ofstream logits_out(std::filesystem::path(options.out.string() + ".f32"), std::ios::binary | std::ios::trunc);
    if (!logits_out) { throw std::runtime_error("cannot write " + options.out.string() + ".f32"); }
    json sequences = json::array();
    std::size_t vocab = 0;
    for (const certify::Window& window : windows) {
        if (!selected(options, window.index)) { continue; }
        const std::vector<TokenId> input(window.tokens.begin(), window.tokens.begin() + tokens);
        infernix::ScoreOptions score;
        // Scored position r predicts token r + 1.
        for (std::uint32_t r = every - 1; r + 1 < tokens; r += every) { score.logits_positions.push_back(r); }
        const infernix::ScoreResult scored = engine.score_tokens(input, 1, score);
        vocab = scored.logits.size() / score.logits_positions.size();
        logits_out.write(reinterpret_cast<const char*>(scored.logits.data()),
                         static_cast<std::streamsize>(scored.logits.size() * sizeof(float)));
        sequences.push_back({{"index", window.index}, {"id", window.id}, {"domain", window.domain}, {"tokens", input},
                             {"positions", score.logits_positions}});
    }
    json meta          = provenance(options, &corpus);
    meta["route"]      = "lossless";
    meta["vocab"]      = vocab;
    meta["sequences"]  = sequences;
    meta["load"]       = load_json(engine);
    write_json(std::filesystem::path(options.out.string() + ".json"), meta);
    return 0;
}

// The positions each window of the set scores, written once per artifact (its template and tokenizer decide where
// the assistant turns lie): a position, the index of a target token in the window, is scored when the model writes
// that token. tools/quality/iqs.py filters every arm of the artifact with it; the references and arms still
// compute every position.
int run_mask(const Options& options) {
    if (options.out.empty()) { throw std::invalid_argument("mask needs --out"); }
    const certify::Corpus corpus = certify::load_corpus(options.corpus, options.set);
    Options exact                = options;
    exact.design                 = false;
    infernix::Engine engine(reference_engine(exact, 128U)); // only its tokenizer is used: the smallest chunk
    const std::vector<certify::Window> windows = certify::load_windows(engine, corpus);
    json entries                               = json::array();
    for (const certify::Window& window : windows) {
        json runs = json::array();
        std::uint64_t scored = 0;
        for (std::size_t t = 1; t < window.generated.size();) {
            if (window.generated[t] == 0) {
                ++t;
                continue;
            }
            std::size_t end = t;
            while (end + 1 < window.generated.size() && window.generated[end + 1] != 0) { ++end; }
            runs.push_back({t, end}); // inclusive
            scored += end - t + 1;
            t = end + 1;
        }
        entries.push_back({{"index", window.index}, {"id", window.id}, {"domain", window.domain},
                           {"positions", window.generated.size() - 1}, {"scored", scored}, {"runs", runs}});
    }
    json meta         = provenance(options, &corpus);
    meta["windows"]   = entries;
    write_json(options.out, meta);
    return 0;
}

int run_prefill(const Options& options) {
    if (options.out.empty() || options.references.empty()) { throw std::invalid_argument("prefill needs --reference and --out"); }
    const certify::Corpus corpus = certify::load_corpus(options.corpus, options.set);
    const auto started           = std::chrono::steady_clock::now();
    infernix::EngineOptions engine_options;
    if (options.lossless) {
        // An exact route against a reference: the noise twin (--noise-seed), or the design reference
        // (--design) against the lossless one (a declared design's error).
        engine_options = reference_engine(options, corpus.set.window_tokens);
    } else {
        if (options.design || options.noise_seed != 0) {
            throw std::invalid_argument("--design and --noise-seed score the exact route: add --lossless");
        }
        apply_route(options, engine_options);
        engine_options.purpose     = infernix::EnginePurpose::CausalScoring;
        engine_options.max_context = corpus.set.window_tokens;
        engine_options.kv_capacity = infernix::KvCapacityPolicy::explicit_capacity(corpus.set.window_tokens);
    }
    infernix::Engine engine(engine_options);
    const std::vector<certify::Window> windows = certify::load_windows(engine, corpus);
    const std::vector<perplexity::ReferenceWindow> references = read_reference(options, corpus, windows);
    RowWriter rows(std::filesystem::path(options.out.string() + ".positions.csv"));
    for (std::size_t w = 0; w < windows.size(); ++w) {
        if (!selected(options, windows[w].index)) { continue; }
        const perplexity::ReferenceWindow& ref = references[w];
        infernix::ScoreOptions score{.top_k = 1, .candidates_per_position = kTop, .candidates = ref.top_ids};
        const infernix::ScoreResult scored = engine.score_tokens(windows[w].tokens, 1, std::move(score));
        const std::size_t positions        = ref.target_logprobs.size();
        for (std::size_t r = 0; r < positions; ++r) {
            const std::span<const float> reference_top(ref.top_logprobs.data() + r * kTop, kTop);
            const std::span<const float> evaluated(scored.candidate_logprobs.data() + r * kTop, kTop);
            rows.add(windows[w].index, -1, r + 1,
                     perplexity::position_divergence(reference_top, evaluated, scored.top_ids[r] == ref.top_ids[r * kTop],
                                                     ref.target_logprobs[r], scored.logprobs[r]));
        }
        std::cerr << "prefill window " << w + 1 << "/" << windows.size() << '\n';
    }
    json meta                = provenance(options, &corpus);
    meta["arm"]              = "prefill";
    meta["route"]            = options.lossless ? (options.noise_seed != 0 ? "noise_twin" : (options.design ? "design" : "lossless"))
                                                : (options.reproducible ? "reproducible" : "default");
    meta["noise_seed"]       = options.noise_seed;
    meta["noise_units"]      = options.noise_units;
    meta["noise_scale"]      = options.noise_scale;
    meta["windows"]          = options.windows;
    meta["kv_dtype"]         = options.kv;
    meta["logits_dtype"]     = options.logits;
    meta["reference"]        = reference_paths(options);
    meta["prefill_chunk"]    = engine.options().prefill_chunk;
    meta["load"]             = load_json(engine);
    meta["rows"]             = rows.rows;
    meta["nonfinite"]        = rows.nonfinite;
    meta["seconds"]          = seconds_since(started);
    write_json(std::filesystem::path(options.out.string() + ".json"), meta);
    return 0;
}

struct Segment {
    std::uint32_t window = 0; // index into the windows
    int index            = 0; // the decode start's index
    std::uint32_t start  = 0;
};

int run_decode(const Options& options) {
    if (options.out.empty() || options.references.empty()) { throw std::invalid_argument("decode needs --reference and --out"); }
    const certify::Corpus corpus = certify::load_corpus(options.corpus, options.set);
    const std::uint32_t forced   = corpus.set.decode_tokens;
    if (forced > infernix::kMaximumConstraintSteps) { throw std::runtime_error("decode_tokens exceeds the constraint step limit"); }
    const auto started = std::chrono::steady_clock::now();
    infernix::EngineOptions engine_options;
    apply_route(options, engine_options);
    engine_options.max_context = (corpus.set.window_tokens + 127) / 128 * 128;
    engine_options.kv_capacity = infernix::KvCapacityPolicy::explicit_capacity(
        options.kv_tokens != 0 ? options.kv_tokens : engine_options.max_context * options.concurrency);
    engine_options.max_concurrency    = options.concurrency;
    engine_options.pending_timeout_ms = 3'600'000;
    if (!options.prefix_reuse) {
        engine_options.context_cache.enabled             = false;
        engine_options.context_cache.host_capacity_bytes = 0;
    }
    engine_options.speculative = options.speculative;
    if (options.speculative.backend != infernix::SpeculativeBackend::None) {
        // The optimized proposal head (--lm-head-draft), as the served configurations run it.
        engine_options.speculative.proposal_head = infernix::ProposalHead::Optimized;
    }
    infernix::Engine engine(engine_options);
    const std::vector<certify::Window> windows = certify::load_windows(engine, corpus);
    const std::vector<perplexity::ReferenceWindow> references = read_reference(options, corpus, windows);

    // Window-major: a window's segments run together, so in a prefix-reuse run they resume from its warm prefix.
    std::vector<Segment> segments;
    for (std::uint32_t w = 0; w < windows.size(); ++w) {
        if (!selected(options, windows[w].index)) { continue; }
        for (std::size_t j = 0; j < corpus.set.decode_starts.size(); ++j) {
            segments.push_back(Segment{.window = w, .index = static_cast<int>(j), .start = corpus.set.decode_starts[j]});
        }
    }
    const auto request_of = [&](const Segment& segment) {
        const certify::Window& window          = windows[segment.window];
        const perplexity::ReferenceWindow& ref = references[segment.window];
        infernix::RequestOptions request;
        request.execution.requested_output_tokens = forced;
        request.execution.sampling.temperature    = 0.0F;
        request.execution.allow_prefix_reuse      = options.prefix_reuse;
        request.stop.include_model_defaults       = false;
        for (std::uint32_t k = 0; k < forced; ++k) {
            const std::size_t token = segment.start + k; // the forced token; its reference row is token - 1
            request.execution.constraint.steps.push_back({window.tokens[token]});
            request.execution.constraint.observe.emplace_back(ref.top_ids.begin() + static_cast<std::ptrdiff_t>((token - 1) * kTop),
                                                              ref.top_ids.begin() + static_cast<std::ptrdiff_t>(token * kTop));
        }
        return request;
    };
    const auto prompt_of = [&](const Segment& segment) {
        const certify::Window& window = windows[segment.window];
        return engine.prepare_tokens(std::vector<TokenId>(window.tokens.begin(), window.tokens.begin() + segment.start),
                                     options.prefix_reuse);
    };

    std::vector<std::optional<infernix::GenerationResult>> results(segments.size());
    if (!options.prefix_reuse) {
        // Rolling lanes: `concurrency` requests in flight, each lane taking the next segment.
        std::atomic<std::size_t> next{0};
        std::vector<std::exception_ptr> errors(options.concurrency);
        std::vector<std::thread> lanes;
        for (std::uint32_t lane = 0; lane < options.concurrency; ++lane) {
            lanes.emplace_back([&, lane] {
                try {
                    for (std::size_t s = next++; s < segments.size(); s = next++) {
                        results[s] = engine.submit(prompt_of(segments[s]), request_of(segments[s])).wait();
                    }
                } catch (...) {
                    errors[lane] = std::current_exception();
                    next         = segments.size();
                }
            });
        }
        for (std::thread& lane : lanes) { lane.join(); }
        for (const std::exception_ptr& error : errors) {
            if (error) { std::rethrow_exception(error); }
        }
    } else {
        // Per window: a warm request publishes its first warm_tokens tokens (off the prefill chunk grid), then
        // its segments run together and resume from that prefix, in groups of `concurrency`.
        const std::size_t per_window = corpus.set.decode_starts.size();
        for (std::size_t base = 0; base < segments.size(); base += per_window) {
            const std::uint32_t w = segments[base].window;
            infernix::RequestOptions warm;
            warm.execution.requested_output_tokens = 1;
            warm.execution.sampling.temperature    = 0.0F;
            warm.execution.allow_prefix_reuse      = true;
            warm.stop.include_model_defaults       = false;
            const std::uint32_t warm_tokens = std::min<std::uint32_t>(options.warm_tokens, corpus.set.decode_starts.front());
            (void)engine.generate(engine.prepare_tokens(std::vector<TokenId>(windows[w].tokens.begin(),
                                                                             windows[w].tokens.begin() + warm_tokens),
                                                        true),
                                  std::move(warm));
            for (std::size_t first = 0; first < per_window; first += options.concurrency) {
                std::vector<infernix::GenerationHandle> handles;
                const std::size_t end = std::min(per_window, first + options.concurrency);
                for (std::size_t j = first; j < end; ++j) {
                    const Segment& segment = segments[base + j];
                    handles.push_back(engine.submit(prompt_of(segment), request_of(segment)));
                }
                for (std::size_t j = first; j < end; ++j) { results[base + j] = handles[j - first].wait(); }
            }
        }
    }

    RowWriter rows(std::filesystem::path(options.out.string() + ".positions.csv"));
    std::uint64_t reused = 0, underflows = 0;
    double prefill_seconds = 0, decode_seconds = 0;
    for (std::size_t s = 0; s < segments.size(); ++s) {
        const Segment& segment                 = segments[s];
        const infernix::GenerationResult& result = *results[s];
        const perplexity::ReferenceWindow& ref = references[segment.window];
        if (result.constrained_draws.size() != forced) {
            throw std::runtime_error("window " + std::to_string(segment.window) + " start " + std::to_string(segment.start) +
                                     " forced " + std::to_string(result.constrained_draws.size()) + " of " +
                                     std::to_string(forced) + " tokens");
        }
        reused += result.reused_prompt_tokens;
        prefill_seconds += result.timings.prefill_seconds;
        decode_seconds += result.timings.decode_share_seconds;
        for (std::uint32_t k = 0; k < forced; ++k) {
            const infernix::ConstrainedDraw& draw = result.constrained_draws[k];
            const std::size_t row                 = segment.start + k - 1;
            underflows += draw.mass < 1.0e-37F ? 1U : 0U;
            const std::span<const float> reference_top(ref.top_logprobs.data() + row * kTop, kTop);
            const bool top1 = !draw.observed_logprobs.empty() && draw.observed_logprobs[0] >= draw.top_logprob;
            rows.add(windows[segment.window].index, segment.index, segment.start + k,
                     perplexity::position_divergence(reference_top, draw.observed_logprobs, top1, ref.target_logprobs[row],
                                                     static_cast<float>(std::log(std::max(static_cast<double>(draw.mass), 1.0e-37)))));
        }
    }
    const infernix::RuntimeStats stats = engine.runtime_stats();
    json meta                    = provenance(options, &corpus);
    meta["arm"]                  = "decode";
    meta["route"]                = options.reproducible ? "reproducible" : "default";
    meta["kv_dtype"]             = options.kv;
    meta["logits_dtype"]         = options.logits;
    meta["reference"]            = reference_paths(options);
    meta["windows"]              = options.windows;
    meta["concurrency"]          = options.concurrency;
    meta["spec"]                 = infernix::product::speculative_backend_name(options.speculative.backend);
    meta["draft_tokens"]         = options.speculative.draft_tokens_auto ? json("auto") : json(options.speculative.draft_tokens);
    meta["ngram_draft_tokens"]   = options.speculative.ngram_draft_tokens_auto ? json("auto")
                                                                               : json(options.speculative.ngram_draft_tokens);
    meta["draft_tree"]           = options.speculative.draft_tree_auto ? json("auto")
                                                                       : json(options.speculative.draft_tree_nodes);
    meta["draft_model"]          = options.speculative.draft_model.string();
    meta["prefix_reuse"]         = options.prefix_reuse;
    meta["warm_tokens"]          = options.prefix_reuse ? options.warm_tokens : 0;
    meta["kv_tokens"]            = options.kv_tokens;
    meta["decode_starts"]        = corpus.set.decode_starts;
    meta["decode_tokens"]        = forced;
    meta["reused_prompt_tokens"] = reused;
    meta["preemptions"]          = stats.preemptions;
    meta["replayed_tokens"]      = stats.replayed_tokens;
    meta["underflows"]           = underflows;
    meta["prefill_seconds"]      = prefill_seconds;
    meta["decode_seconds"]       = decode_seconds;
    meta["load"]                 = load_json(engine);
    meta["rows"]                 = rows.rows;
    meta["nonfinite"]            = rows.nonfinite;
    meta["seconds"]              = seconds_since(started);
    write_json(std::filesystem::path(options.out.string() + ".json"), meta);
    return 0;
}

// ---------------------------------------------------------------------------- llama.cpp's KL divergence

std::string read_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { throw std::runtime_error("cannot read " + path.string()); }
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

// The scored positions of a chunk whose logits llama.cpp's rows hold: n_ctx/2 .. n_ctx-2 (scored position r
// predicts token r + 1).
std::vector<std::uint32_t> llama_positions(const llama::BaseHeader& header) {
    std::vector<std::uint32_t> out;
    for (std::uint32_t r = header.first_row(); r + 1 < header.context; ++r) { out.push_back(r); }
    return out;
}

// An arm's outputs: PREFIX.positions.csv (written as it goes), PREFIX.summary.txt (llama.cpp's summary, also
// printed) and PREFIX.json.
void finish_llama_arm(const Options& options, json meta, const llama::Summary& summary, const llama::PositionWriter& rows) {
    const std::string text = summary.text();
    std::ofstream out(std::filesystem::path(options.out.string() + ".summary.txt"), std::ios::binary);
    out << text;
    if (!out) { throw std::runtime_error("cannot write " + options.out.string() + ".summary.txt"); }
    std::cout << text;
    meta["rows"]    = rows.rows();
    meta["summary"] = summary.to_json();
    write_json(std::filesystem::path(options.out.string() + ".json"), meta);
}

void llama_base_json(json& meta, const Options& options, const llama::BaseHeader& header) {
    meta["llama_base"] = std::filesystem::absolute(options.llama_base).string();
    meta["context"]    = header.context;
    meta["chunks"]     = header.chunks;
    meta["vocab"]      = header.vocab;
}

int run_llama_reference(const Options& options) {
    if (options.out.empty()) { throw std::invalid_argument("reference needs --out"); }
    if (options.text.empty() == options.tokens_from.empty()) {
        throw std::invalid_argument("reference --format llama-kld takes exactly one of --text and --tokens-from");
    }
    if (!options.design && options.kv != "bf16") {
        throw std::invalid_argument("the lossless reference uses bf16 KV; a KV format's codec needs --design");
    }
    const auto started = std::chrono::steady_clock::now();
    llama::BaseHeader header;
    if (!options.tokens_from.empty()) {
        header = llama::BaseReader(options.tokens_from).header();
    } else {
        header.context = options.context;
    }
    if (header.context < 4 || header.context % 128 != 0) {
        throw std::invalid_argument("the llama-kld context must be a multiple of 128");
    }
    infernix::Engine engine(reference_engine(options, header.context));
    if (!options.text.empty()) {
        const std::vector<TokenId> tokens = engine.tokenize_text(read_file(options.text));
        const std::size_t whole           = tokens.size() / header.context;
        const std::size_t chunks          = options.chunks == 0 ? whole : std::min<std::size_t>(options.chunks, whole);
        if (chunks == 0 || (options.chunks != 0 && chunks < options.chunks)) {
            throw std::runtime_error(options.text.string() + " holds " + std::to_string(whole) + " chunks of " +
                                     std::to_string(header.context) + " tokens");
        }
        header.chunks = static_cast<std::int32_t>(chunks);
        header.tokens.assign(tokens.begin(), tokens.begin() + static_cast<std::ptrdiff_t>(chunks * header.context));
    }
    const std::vector<std::uint32_t> positions = llama_positions(header);
    std::optional<llama::BaseWriter> writer;
    for (std::int32_t c = 0; c < header.chunks; ++c) {
        const std::span<const TokenId> chunk = header.chunk(static_cast<std::size_t>(c));
        const infernix::ScoreResult scored =
            engine.score_tokens(std::vector<TokenId>(chunk.begin(), chunk.end()), 1, infernix::ScoreOptions{.logits_positions = positions});
        const std::size_t vocab = scored.logits.size() / positions.size();
        if (!writer) {
            if (header.vocab != 0 && static_cast<std::size_t>(header.vocab) != vocab) {
                throw std::runtime_error(options.tokens_from.string() + " has a vocabulary of " + std::to_string(header.vocab) +
                                         ", the artifact " + std::to_string(vocab));
            }
            header.vocab = static_cast<std::int32_t>(vocab);
            writer.emplace(options.out, header);
        }
        for (std::size_t k = 0; k < positions.size(); ++k) {
            writer->write_row(std::span<const float>(scored.logits.data() + k * vocab, vocab));
        }
        std::cerr << "llama-kld reference chunk " << c + 1 << "/" << header.chunks << '\n';
    }
    writer->finish();
    json meta            = provenance(options, nullptr);
    meta["format"]       = "llama-kld";
    meta["route"]        = options.design ? "design" : "lossless";
    meta["kv_dtype"]     = options.design ? options.kv : "bf16";
    meta["logits_dtype"] = options.logits;
    meta["noise_seed"]   = options.noise_seed;
    meta["source"]       = options.text.empty() ? json{{"tokens_from", std::filesystem::absolute(options.tokens_from).string()}}
                                                : json{{"text", std::filesystem::absolute(options.text).string()}};
    meta["context"]      = header.context;
    meta["chunks"]       = header.chunks;
    meta["vocab"]        = header.vocab;
    meta["load"]         = load_json(engine);
    meta["seconds"]      = seconds_since(started);
    write_json(std::filesystem::path(options.out.string() + ".json"), meta);
    return 0;
}

int run_llama_prefill(const Options& options) {
    if (options.out.empty()) { throw std::invalid_argument("prefill needs --out"); }
    llama::BaseReader base(options.llama_base);
    const llama::BaseHeader& header = base.header();
    const auto started              = std::chrono::steady_clock::now();
    infernix::EngineOptions engine_options;
    if (options.lossless) {
        engine_options = reference_engine(options, header.context);
    } else {
        if (options.design || options.noise_seed != 0) {
            throw std::invalid_argument("--design and --noise-seed score the exact route: add --lossless");
        }
        apply_route(options, engine_options);
        engine_options.purpose     = infernix::EnginePurpose::CausalScoring;
        engine_options.max_context = (header.context + 127) / 128 * 128;
        engine_options.kv_capacity = infernix::KvCapacityPolicy::explicit_capacity(engine_options.max_context);
    }
    infernix::Engine engine(engine_options);
    const std::vector<std::uint32_t> positions = llama_positions(header);
    const std::size_t words                    = header.row_words();
    llama::Summary summary;
    llama::PositionWriter rows(std::filesystem::path(options.out.string() + ".positions.csv"));
    for (std::int32_t c = 0; c < header.chunks; ++c) {
        const std::vector<std::uint16_t> base_rows = base.next_chunk();
        const std::span<const TokenId> chunk       = header.chunk(static_cast<std::size_t>(c));
        const infernix::ScoreResult scored =
            engine.score_tokens(std::vector<TokenId>(chunk.begin(), chunk.end()), 1, infernix::ScoreOptions{.logits_positions = positions});
        if (scored.logits.size() != positions.size() * static_cast<std::size_t>(header.vocab)) {
            throw std::runtime_error("the artifact's vocabulary differs from the base file's " + std::to_string(header.vocab));
        }
        for (std::size_t k = 0; k < positions.size(); ++k) {
            const std::uint32_t r          = positions[k];
            const llama::Position position = llama::compare_logits(
                std::span<const float>(scored.logits.data() + k * static_cast<std::size_t>(header.vocab), static_cast<std::size_t>(header.vocab)),
                std::span<const std::uint16_t>(base_rows.data() + k * words, words), chunk[r + 1]);
            summary.add(position);
            rows.add(static_cast<std::uint32_t>(c), r, position);
        }
        std::cerr << "llama-kld prefill chunk " << c + 1 << "/" << header.chunks << '\n';
    }
    base.finish();
    json meta             = provenance(options, nullptr);
    meta["arm"]           = "llama_prefill";
    meta["route"]         = options.lossless ? (options.noise_seed != 0 ? "noise_twin" : (options.design ? "design" : "lossless"))
                                             : (options.reproducible ? "reproducible" : "default");
    meta["noise_seed"]    = options.noise_seed;
    meta["noise_units"]   = options.noise_units;
    meta["noise_scale"]   = options.noise_scale;
    meta["kv_dtype"]      = options.kv;
    meta["logits_dtype"]  = options.logits;
    meta["prefill_chunk"] = engine.options().prefill_chunk;
    llama_base_json(meta, options, header);
    meta["load"]    = load_json(engine);
    meta["seconds"] = seconds_since(started);
    finish_llama_arm(options, std::move(meta), summary, rows);
    return 0;
}

// The decode path (llama-perplexity -b 1's counterpart): each chunk's scored tokens forced through it in requests
// of kMaximumConstraintSteps steps, each request's first step (its prompt's prefill) unscored, so every scored
// distribution comes from a decode step. A step observes the tokens of the KL sum (base log-probability above
// -16, kl_support), most probable first, kMaximumObservedTokens - 1 per request besides the target: a step with
// more repeats the identical request with the next tokens, and every repeat must reproduce the first one's
// target and largest log-probabilities bit for bit.
int run_llama_decode(const Options& options) {
    if (options.out.empty()) { throw std::invalid_argument("decode needs --out"); }
    if (options.concurrency != 1 || options.prefix_reuse) {
        throw std::invalid_argument("the llama-kld decode arm repeats identical single requests: no --concurrency or --prefix-reuse");
    }
    llama::BaseReader base(options.llama_base);
    const llama::BaseHeader& header = base.header();
    const auto started              = std::chrono::steady_clock::now();
    infernix::EngineOptions engine_options;
    apply_route(options, engine_options);
    engine_options.max_context                       = (header.context + 127) / 128 * 128;
    engine_options.kv_capacity                       = infernix::KvCapacityPolicy::explicit_capacity(engine_options.max_context);
    engine_options.max_concurrency                   = 1;
    engine_options.pending_timeout_ms                = 3'600'000;
    engine_options.context_cache.enabled             = false;
    engine_options.context_cache.host_capacity_bytes = 0;
    engine_options.speculative                       = options.speculative;
    if (options.speculative.backend != infernix::SpeculativeBackend::None) {
        engine_options.speculative.proposal_head = infernix::ProposalHead::Optimized;
    }
    infernix::Engine engine(engine_options);
    constexpr std::uint32_t kSteps = infernix::kMaximumConstraintSteps;
    constexpr std::size_t kSlice   = infernix::kMaximumObservedTokens - 1;
    const std::size_t words        = header.row_words();
    const std::uint32_t first      = header.first_row() + 1; // the first scored token
    llama::Summary summary;
    llama::PositionWriter rows(std::filesystem::path(options.out.string() + ".positions.csv"));
    std::uint64_t requests = 0, observed_tokens = 0;
    double prefill_seconds = 0, decode_seconds = 0;
    struct Step {
        TokenId target = 0;
        std::span<const std::uint16_t> row;
        std::vector<TokenId> support;   // ascending
        std::vector<TokenId> order;     // the support without the target, most probable first
        std::vector<float> log_probs;   // the evaluated run's, in the support's order
        float target_log_prob = 0, top_log_prob = 0;
    };
    for (std::int32_t c = 0; c < header.chunks; ++c) {
        const std::vector<std::uint16_t> base_rows = base.next_chunk();
        const std::span<const TokenId> chunk       = header.chunk(static_cast<std::size_t>(c));
        // A request forces tokens start - 1 .. start + count - 1 and scores start .. start + count - 1.
        for (std::uint32_t start = first; start < header.context; start += kSteps - 1) {
            const std::uint32_t count = std::min(kSteps - 1, header.context - start);
            std::vector<Step> steps(count);
            std::size_t passes = 1;
            for (std::uint32_t k = 0; k < count; ++k) {
                Step& step  = steps[k];
                const std::uint32_t r = start + k - 1; // the row of the position predicting token start + k
                step.target = chunk[start + k];
                step.row    = std::span<const std::uint16_t>(base_rows.data() + (r - header.first_row()) * words, words);
                step.support = llama::kl_support(step.row, header.vocab);
                for (const TokenId token : step.support) {
                    if (token != step.target) { step.order.push_back(token); }
                }
                std::stable_sort(step.order.begin(), step.order.end(), [&](TokenId a, TokenId b) {
                    return llama::row_log_prob(step.row, a) > llama::row_log_prob(step.row, b);
                });
                step.log_probs.assign(step.support.size(), 0.0F);
                passes = std::max(passes, (step.order.size() + kSlice - 1) / kSlice);
            }
            for (std::size_t pass = 0; pass < passes; ++pass) {
                infernix::RequestOptions request;
                request.execution.requested_output_tokens = count + 1;
                request.execution.sampling.temperature    = 0.0F;
                request.execution.allow_prefix_reuse      = false;
                request.stop.include_model_defaults       = false;
                request.execution.constraint.steps.push_back({chunk[start - 1]});
                request.execution.constraint.observe.push_back({chunk[start - 1]});
                std::vector<std::span<const TokenId>> slices(count);
                for (std::uint32_t k = 0; k < count; ++k) {
                    const Step& step        = steps[k];
                    const std::size_t begin = std::min(step.order.size(), pass * kSlice);
                    const std::size_t end   = std::min(step.order.size(), begin + kSlice);
                    slices[k]               = std::span<const TokenId>(step.order.data() + begin, end - begin);
                    std::vector<TokenId> observe{step.target};
                    observe.insert(observe.end(), slices[k].begin(), slices[k].end());
                    request.execution.constraint.steps.push_back({step.target});
                    request.execution.constraint.observe.push_back(std::move(observe));
                }
                const infernix::GenerationResult result = engine.generate(
                    engine.prepare_tokens(std::vector<TokenId>(chunk.begin(), chunk.begin() + (start - 1)), false), std::move(request));
                if (result.constrained_draws.size() != count + 1) {
                    throw std::runtime_error("chunk " + std::to_string(c) + " forced " + std::to_string(result.constrained_draws.size()) +
                                             " of " + std::to_string(count + 1) + " tokens");
                }
                ++requests;
                prefill_seconds += result.timings.prefill_seconds;
                decode_seconds += result.timings.decode_share_seconds;
                for (std::uint32_t k = 0; k < count; ++k) {
                    Step& step                            = steps[k];
                    const infernix::ConstrainedDraw& draw = result.constrained_draws[k + 1];
                    if (draw.observed_logprobs.size() != slices[k].size() + 1) {
                        throw std::runtime_error("a decode step reported the wrong observation count");
                    }
                    if (pass == 0) {
                        step.target_log_prob = draw.observed_logprobs[0];
                        step.top_log_prob    = draw.top_logprob;
                    } else if (std::memcmp(&step.target_log_prob, &draw.observed_logprobs[0], sizeof(float)) != 0 ||
                               std::memcmp(&step.top_log_prob, &draw.top_logprob, sizeof(float)) != 0) {
                        throw std::runtime_error("chunk " + std::to_string(c) + " token " + std::to_string(start + k) +
                                                 ": a repeated identical request decoded different logits");
                    }
                    for (std::size_t j = 0; j < slices[k].size(); ++j) {
                        const auto at = std::lower_bound(step.support.begin(), step.support.end(), slices[k][j]);
                        step.log_probs[static_cast<std::size_t>(at - step.support.begin())] = draw.observed_logprobs[j + 1];
                    }
                    observed_tokens += slices[k].size() + 1;
                }
            }
            for (std::uint32_t k = 0; k < count; ++k) {
                Step& step = steps[k];
                const auto target = std::lower_bound(step.support.begin(), step.support.end(), step.target);
                if (target != step.support.end() && *target == step.target) {
                    step.log_probs[static_cast<std::size_t>(target - step.support.begin())] = step.target_log_prob;
                }
                // The base's most probable token is in the support; the evaluated run's is it when it holds the
                // step's largest log-probability.
                const TokenId top  = llama::base_top(step.row, header.vocab);
                const auto top_at  = std::lower_bound(step.support.begin(), step.support.end(), top);
                const bool same    = step.log_probs[static_cast<std::size_t>(top_at - step.support.begin())] == step.top_log_prob;
                const llama::Position position =
                    llama::compare_observed(step.row, step.support, step.log_probs, step.target_log_prob, same, step.target);
                summary.add(position);
                rows.add(static_cast<std::uint32_t>(c), start + k - 1, position);
            }
        }
        std::cerr << "llama-kld decode chunk " << c + 1 << "/" << header.chunks << " (" << requests << " requests)\n";
    }
    base.finish();
    json meta                  = provenance(options, nullptr);
    meta["arm"]                = "llama_decode";
    meta["route"]              = options.reproducible ? "reproducible" : "default";
    meta["kv_dtype"]           = options.kv;
    meta["logits_dtype"]       = options.logits;
    meta["spec"]               = infernix::product::speculative_backend_name(options.speculative.backend);
    meta["requests"]           = requests;
    meta["observed_tokens"]    = observed_tokens;
    meta["prefill_seconds"]    = prefill_seconds;
    meta["decode_seconds"]     = decode_seconds;
    llama_base_json(meta, options, header);
    meta["load"]    = load_json(engine);
    meta["seconds"] = seconds_since(started);
    finish_llama_arm(options, std::move(meta), summary, rows);
    return 0;
}

// Another base file (llama-perplexity --kl-divergence-base of llama.cpp's own logits, the same tokens) against
// --llama-base, by the arms' definitions: its log-probabilities are its codes' (those more than 16 below its
// maximum read as its floor).
int run_llama_compare(const Options& options) {
    if (options.out.empty() || options.llama_base.empty() || options.logits_file.empty()) {
        throw std::invalid_argument("llama-compare needs --llama-base, --logits and --out");
    }
    llama::BaseReader base(options.llama_base);
    llama::BaseReader other(options.logits_file);
    const llama::BaseHeader& header = base.header();
    const llama::BaseHeader& theirs = other.header();
    if (header.context != theirs.context || header.vocab != theirs.vocab || header.chunks != theirs.chunks) {
        throw std::runtime_error("the base files hold different chunk geometries or vocabularies");
    }
    const auto mismatch = std::mismatch(header.tokens.begin(), header.tokens.end(), theirs.tokens.begin());
    if (mismatch.first != header.tokens.end()) {
        throw std::runtime_error("the base files hold different tokens (first at index " +
                                 std::to_string(mismatch.first - header.tokens.begin()) + ")");
    }
    const auto started      = std::chrono::steady_clock::now();
    const std::size_t words = header.row_words();
    llama::Summary summary;
    llama::PositionWriter rows(std::filesystem::path(options.out.string() + ".positions.csv"));
    std::vector<float> log_probs;
    for (std::int32_t c = 0; c < header.chunks; ++c) {
        const std::vector<std::uint16_t> base_rows  = base.next_chunk();
        const std::vector<std::uint16_t> other_rows = other.next_chunk();
        const std::span<const TokenId> chunk        = header.chunk(static_cast<std::size_t>(c));
        for (std::uint32_t k = 0; k < header.rows_per_chunk(); ++k) {
            const std::span<const std::uint16_t> row(base_rows.data() + k * words, words);
            const std::span<const std::uint16_t> evaluated(other_rows.data() + k * words, words);
            const TokenId target                   = chunk[header.first_row() + k + 1];
            const std::vector<TokenId> support     = llama::kl_support(row, header.vocab);
            log_probs.resize(support.size());
            for (std::size_t j = 0; j < support.size(); ++j) { log_probs[j] = llama::row_log_prob(evaluated, support[j]); }
            const bool same = llama::base_top(evaluated, header.vocab) == llama::base_top(row, header.vocab);
            const llama::Position position =
                llama::compare_observed(row, support, log_probs, llama::row_log_prob(evaluated, target), same, target);
            summary.add(position);
            rows.add(static_cast<std::uint32_t>(c), header.first_row() + k, position);
        }
    }
    base.finish();
    other.finish();
    json meta      = provenance(options, nullptr);
    meta["arm"]    = "llama_compare";
    meta["logits"] = std::filesystem::absolute(options.logits_file).string();
    llama_base_json(meta, options, header);
    meta["seconds"] = seconds_since(started);
    finish_llama_arm(options, std::move(meta), summary, rows);
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse(argc, argv);
        const bool llama_format = options.format == "llama-kld";
        if (llama_format && options.command != "reference") { throw std::invalid_argument("--format belongs to reference"); }
        if (!options.llama_base.empty() && options.command != "prefill" && options.command != "decode" &&
            options.command != "llama-compare") {
            throw std::invalid_argument("--llama-base belongs to prefill, decode and llama-compare");
        }
        if (options.command == "reference") { return llama_format ? run_llama_reference(options) : run_reference(options); }
        if (options.command == "verify") { return run_verify(options); }
        if (options.command == "prefill") { return options.llama_base.empty() ? run_prefill(options) : run_llama_prefill(options); }
        if (options.command == "decode") { return options.llama_base.empty() ? run_decode(options) : run_llama_decode(options); }
        if (options.command == "llama-compare") { return run_llama_compare(options); }
        if (options.command == "capture") { return run_capture(options, options.capture_tokens, options.capture_every); }
        if (options.command == "mask") { return run_mask(options); }
        throw std::invalid_argument("unknown command " + options.command);
    } catch (const std::exception& error) {
        std::cerr << "infernix-certify: " << error.what() << '\n';
        return 2;
    }
}
