// infernix-certify: the IQS-1 measurement tool (docs/maintainer/model-quality.md §6). One reference pass per
// window serves every arm: the reference scores each window's positions once (the lossless route, the
// design reference or the noise twin), a prefill arm scores the same windows on a shipped route, and a
// decode arm forces each window's continuation through the decode path at the set's decode starts, every
// forced step observing the reference's top tokens at that position. Arms write one row per position
// (KL over the reference's top tokens plus one bucket for the rest, top-1 flip, NLL change), which
// tools/quality/iqs.py turns into bounds and verdicts.
//
//   infernix-certify reference <artifact> --corpus M --set S --out REF [--design --kv-dtype K]
//                    [--noise-seed N] [--windows i,j,...]
//   infernix-certify verify    <artifact> --reference REF [--windows i,j,...]
//   infernix-certify capture   <artifact> --corpus M --set S --out PREFIX --windows i,j [--tokens N] [--every K]
//   infernix-certify mask      <artifact> --corpus M --set S --out FILE.json
//   infernix-certify prefill   <artifact> --corpus M --set S --reference REF --out PREFIX [route options]
//   infernix-certify decode    <artifact> --corpus M --set S --reference REF --out PREFIX [route options]
//                    [--concurrency N] [--spec mtp|dflash|dflash2] [--draft-tokens N|auto]
//                    [--ngram-draft-tokens N|auto] [--draft-tree-nodes auto|LIST] [--draft-model FILE]
//                    [--prefix-reuse [--warm-tokens N]] [--kv-tokens N]
//   route options: [--reproducible] [--fast-path NAME]... [--kv-dtype K] [--prefill-chunk N]
//                  [--ngram-volume FILE] [--expert-a4-activations]

#include "infernix_build_id.h"
#include "distribution.h"
#include "iqs_corpus.h"
#include "infernix/engine.h"
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
#include <optional>
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

// IQS-1 metrics: KL over the reference's 256 most probable tokens plus one bucket for the rest, in
// prefill (scoring candidates) and decode (observed tokens).
constexpr std::uint32_t kTop = 256;
static_assert(kTop <= infernix::kMaximumScoreTopTokens && kTop <= infernix::kMaximumObservedTokens);

struct Options {
    std::string command;
    std::string artifact;
    std::filesystem::path corpus;
    std::string set = "main";
    std::filesystem::path out;
    std::filesystem::path reference;
    bool lossless              = false; // prefill: score on the exact route (lossless, --design, --noise-seed)
    bool design                = false;
    std::uint64_t noise_seed   = 0;
    std::string kv             = "bf16";
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
    if (argc < 3) { throw std::invalid_argument("usage: infernix-certify reference|verify|prefill|decode <artifact> ..."); }
    Options out{.command = argv[1], .artifact = argv[2]};
    for (int i = 3; i < argc; ++i) {
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
            out.reference = std::string(value());
        } else if (option == "--lossless") {
            out.lossless = true;
        } else if (option == "--design") {
            out.design = true;
        } else if (option == "--noise-seed") {
            out.noise_seed = parse_integer<std::uint64_t>(value(), "noise seed");
        } else if (option == "--kv-dtype") {
            out.kv = value();
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
    json out{{"tool", "infernix-certify"},
             {"command", options.command},
             {"artifact", artifact_path(options.artifact)}};
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
    engine_options.fast_paths            = !options.reproducible;
    engine_options.fast_path_candidates  = options.candidates;
    engine_options.expert_a4_activations = options.expert_a4_activations;
    if (options.prefill_chunk) { engine_options.prefill_chunk = *options.prefill_chunk; }
    if (!options.ngram_volume.empty()) { engine_options.ngram_volume_path = options.ngram_volume; }
}

std::vector<perplexity::ReferenceWindow> read_reference(const std::filesystem::path& path,
                                                        const certify::Corpus& corpus,
                                                        const std::vector<certify::Window>& windows) {
    perplexity::ReferenceReader reader(path);
    const perplexity::ReferenceHeader& header = reader.header();
    if (header.corpus_id != corpus.corpus_id || header.mode != corpus.set.name ||
        header.context != corpus.set.window_tokens || header.top_k != kTop) {
        throw std::runtime_error(path.string() + " is a reference for corpus " + header.corpus_id + " set " +
                                 header.mode + ", not this corpus set");
    }
    std::vector<perplexity::ReferenceWindow> out;
    for (const certify::Window& window : windows) {
        perplexity::ReferenceWindow ref = reader.next();
        if (ref.stream_index != window.index || ref.input_hash != window.input_hash || ref.target_begin != 1 ||
            ref.target_end != window.tokens.size()) {
            throw std::runtime_error(path.string() + " window " + std::to_string(ref.stream_index) +
                                     " does not match " + window.id + " (other text or tokenizer)");
        }
        out.push_back(std::move(ref));
    }
    reader.finish();
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
    engine_options.prefill_chunk         = window_tokens;
    engine_options.lossless              = true;
    engine_options.reference             = options.design;
    engine_options.lossless_noise_seed   = options.noise_seed;
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
    if (!options.reference.empty()) {
        if (!options.design) { throw std::invalid_argument("reference --reference compares a design reference: add --design"); }
        compared = read_reference(options.reference, corpus, windows);
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
        compare["reference"] = std::filesystem::absolute(options.reference).string();
        compare["rows"]      = compare_rows->rows;
        compare["nonfinite"] = compare_rows->nonfinite;
        write_json(std::filesystem::path(options.out.string() + ".compare.json"), compare);
    }
    json meta          = provenance(options, &corpus);
    meta["route"]      = options.design ? "design" : "lossless";
    meta["kv_dtype"]   = options.design ? options.kv : "bf16";
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
    if (options.reference.empty()) { throw std::invalid_argument("verify needs --reference"); }
    const json meta = read_json(std::filesystem::path(options.reference.string() + ".json"));
    options.corpus     = meta.at("corpus").at("manifest").get<std::string>();
    options.set        = meta.at("corpus").at("set").get<std::string>();
    options.design     = meta.at("route") == "design";
    options.kv         = meta.at("kv_dtype").get<std::string>();
    options.noise_seed = meta.at("noise_seed").get<std::uint64_t>();
    options.expert_a4_activations = meta.value("expert_a4_activations", false);
    if (options.windows.empty()) { options.windows = {0}; }
    const certify::Corpus corpus = certify::load_corpus(options.corpus, options.set);
    if (hex(corpus.content_hash) != meta.at("corpus").at("content_hash").get<std::string>()) {
        throw std::runtime_error("the corpus changed since " + options.reference.string() + " was written");
    }
    infernix::Engine engine(reference_engine(options, corpus.set.window_tokens));
    const std::vector<certify::Window> windows = certify::load_windows(engine, corpus);
    perplexity::ReferenceReader reader(options.reference);
    bool identical = true;
    for (const certify::Window& window : windows) {
        perplexity::ReferenceWindow stored;
        bool found = false;
        try {
            // The stored file holds the windows it was written with, in order.
            while (true) {
                stored = reader.next();
                if (stored.stream_index >= window.index) { break; }
            }
            found = stored.stream_index == window.index;
        } catch (const std::exception&) { found = false; }
        if (!selected(options, window.index)) { continue; }
        if (!found) { throw std::runtime_error(options.reference.string() + " holds no window " + std::to_string(window.index)); }
        const perplexity::ReferenceWindow fresh = score_reference_window(engine, window);
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

// Reference conformance (IQS-1 G1): the lossless route's whole FP32 logits at every --every'th position of the
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
    if (options.out.empty() || options.reference.empty()) { throw std::invalid_argument("prefill needs --reference and --out"); }
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
    const std::vector<perplexity::ReferenceWindow> references = read_reference(options.reference, corpus, windows);
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
    meta["windows"]          = options.windows;
    meta["kv_dtype"]         = options.kv;
    meta["reference"]        = std::filesystem::absolute(options.reference).string();
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
    if (options.out.empty() || options.reference.empty()) { throw std::invalid_argument("decode needs --reference and --out"); }
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
    const std::vector<perplexity::ReferenceWindow> references = read_reference(options.reference, corpus, windows);

    // Window-major: a window's segments run together, so in a prefix-reuse run they resume from its warm prefix.
    std::vector<Segment> segments;
    for (std::uint32_t w = 0; w < windows.size(); ++w) {
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
        for (std::uint32_t w = 0; w < windows.size(); ++w) {
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
                    const Segment& segment = segments[w * per_window + j];
                    handles.push_back(engine.submit(prompt_of(segment), request_of(segment)));
                }
                for (std::size_t j = first; j < end; ++j) { results[w * per_window + j] = handles[j - first].wait(); }
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
    meta["reference"]            = std::filesystem::absolute(options.reference).string();
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

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse(argc, argv);
        if (options.command == "reference") { return run_reference(options); }
        if (options.command == "verify") { return run_verify(options); }
        if (options.command == "prefill") { return run_prefill(options); }
        if (options.command == "decode") { return run_decode(options); }
        if (options.command == "capture") { return run_capture(options, options.capture_tokens, options.capture_every); }
        if (options.command == "mask") { return run_mask(options); }
        throw std::invalid_argument("unknown command " + options.command);
    } catch (const std::exception& error) {
        std::cerr << "infernix-certify: " << error.what() << '\n';
        return 2;
    }
}
