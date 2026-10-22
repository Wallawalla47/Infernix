// Teacher-forced decode quality: the decode route's distribution at the corpus's own next tokens.
// Each segment prefills a corpus prompt and then forces the following corpus tokens through the
// production decode path, one constrained step per token (TokenConstraint: a one-token step forces
// it, and ConstrainedDraw::mass is that token's probability under the round's full next-token
// distribution). Every position therefore has the same context in every build, so two builds'
// log-probabilities pair position by position: unlike free-running greedy decoding, nothing
// diverges after the first rounding difference. tools/quality/fast_path_gate.py pairs a default and a
// fast-path run.
//
// --lossless writes the reference instead: the lossless route's (EngineOptions::lossless,
// Qwen3.5/3.6/3.8 and Qwen3.8-Flash-Next) log-probability of every forced token and its 64 most
// probable tokens there (Engine::score_tokens, BF16 KV, one pass), for the same segments. --observe
// REFERENCE then makes each forced step also report the log-probabilities of the reference's top
// tokens and its own largest log-probability (TokenConstraint::observe), so the gate measures the
// decode route's KL divergence from the reference and its top-1 agreement exactly as the prefill
// gate does. --reference --observe REFERENCE writes the design reference's run instead (the
// lossless route with the configuration's declared steps, --kv-dtype's codec among them; one scoring
// pass per segment, reported as a decode run is): the gate's base arm.
//
// Segments lie back to back from the corpus's first token, no two sharing a token, and cycle
// through the prompt lengths, so the forced steps read short and long contexts. --first-segment N
// runs the set's segments N to N + --segments - 1 (their offsets and prompt lengths as in the whole
// set), so a larger set extends a smaller one: tools/quality/fast_path_gate.py concat-forced joins the
// runs and references of consecutive ranges. --mode selects the decode route: plain (no speculation: one
// single-column round per token), dflash2 (DFlash2, 7 drafts, the optimized proposal head), ngram
// (DFlash2 with n-gram drafting) or mtp (MTP, 4 drafts). Fast paths run as in infernix-serve;
// --reproducible runs the reproducible route and --fast-path NAME also runs that fast path
// whether or not it is qualified (qualification). --expert-a4-activations runs
// nvidia/Qwen3.8-Flash-Next-FP4-Local's routed experts with 4-bit activations (not the reference).
//
// The same segments also measure request invariance: their log-probabilities are printed exactly,
// so `fast_path_gate identical` compares two runs bit for bit. --concurrency N keeps N segments in
// flight, each lane taking the next segment as its previous one completes (their prompt lengths
// differ, so prefills overlap decodes and no lane idles behind a longer prompt), --prefill-chunk N
// sets the prompt chunk, and --prefix-reuse submits the segments in groups of N instead: each
// segment's prompt (or with --warm-tokens N its first N tokens) runs alone first, and then the
// group's measured requests together from the prefix cache. The run's last line reports the summed
// prefill and decode time, which shows which of the two its wall time went to.
//
// Usage: infernix_decode_quality_forced <artifact> <corpus.txt> <out.json> [--segments N]
//          [--first-segment N] [--prompts L1,L2,...] [--forced N]
//          [--lossless | --observe REFERENCE.json [--reference]]
//          [--mode plain|dflash2|ngram|mtp|mtp-ngram] [--kv-dtype int8|bf16|fp8|nvfp4]
//          [--reproducible] [--fast-path NAME]... [--expert-a4-activations]
//          [--concurrency 1..8] [--prefill-chunk N] [--prefix-reuse [--warm-tokens N]]
//          [--ngram-volume FILE] (Qwen3.8-Flash-Next; default <artifact>.ngram)
//          [--expert-pin-mib N] (Qwen3.8-Flash-Next: page-locked expert RAM; the CPU serves the rest)

#include "infernix/engine.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {

struct Options {
    std::string artifact;
    std::string corpus;
    std::string output;
    std::uint32_t segments      = 256;
    std::uint32_t first_segment = 0; // the set's first segment this run computes
    std::vector<std::uint32_t> prompts{512, 2048, 8192};
    std::uint32_t forced  = infernix::kMaximumConstraintSteps;
    std::string mode      = "plain";
    std::string kv        = "int8";
    bool fast_paths       = true;
    std::vector<infernix::FastPath> candidates;
    std::uint32_t concurrency   = 1;
    std::uint32_t prefill_chunk = 4096;
    // KV pool tokens; 0: every lane's whole context. Less makes concurrent requests preempt.
    std::uint32_t kv_tokens     = 0;
    bool prefix_reuse           = false;
    std::uint32_t warm_tokens   = 0; // 0: the whole prompt
    bool lossless               = false; // write the reference (the lossless route)
    bool reference              = false; // the design reference's observed run (with --observe)
    std::string observe;
    std::string ngram_volume;
    bool expert_a4_activations = false;
    // Qwen3.8-Flash-Next: page-locked expert RAM (EngineOptions::expert_pin_bytes); the rest the CPU
    // serves. Empty: the default.
    std::optional<std::uint64_t> expert_pin_mib;
};

std::uint32_t parse_u32(std::string_view text, const char* label) {
    std::uint32_t value = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size() || value == 0) {
        throw std::invalid_argument(std::string("invalid ") + label + ": " + std::string(text));
    }
    return value;
}

Options parse(int argc, char** argv) {
    if (argc < 4) { throw std::invalid_argument("artifact, corpus and output are required"); }
    Options out{.artifact = argv[1], .corpus = argv[2], .output = argv[3]};
    for (int i = 4; i < argc; ++i) {
        const std::string_view option = argv[i];
        const auto value              = [&]() -> std::string_view {
            if (++i >= argc) { throw std::invalid_argument(std::string(option) + " needs a value"); }
            return argv[i];
        };
        if (option == "--segments") {
            out.segments = parse_u32(value(), "segments");
        } else if (option == "--first-segment") {
            out.first_segment = parse_u32(value(), "first segment");
        } else if (option == "--prompts") {
            out.prompts.clear();
            std::string_view list = value();
            while (!list.empty()) {
                const std::size_t comma = list.find(',');
                out.prompts.push_back(parse_u32(list.substr(0, comma), "prompt length"));
                list = comma == std::string_view::npos ? std::string_view{} : list.substr(comma + 1);
            }
        } else if (option == "--forced") {
            out.forced = parse_u32(value(), "forced");
        } else if (option == "--mode") {
            out.mode = value();
        } else if (option == "--kv-dtype") {
            out.kv = value();
        } else if (option == "--ngram-volume") {
            out.ngram_volume = value();
        } else if (option == "--concurrency") {
            out.concurrency = parse_u32(value(), "concurrency");
        } else if (option == "--prefill-chunk") {
            out.prefill_chunk = parse_u32(value(), "prefill chunk");
        } else if (option == "--kv-tokens") {
            out.kv_tokens = parse_u32(value(), "KV tokens");
        } else if (option == "--prefix-reuse") {
            out.prefix_reuse = true;
        } else if (option == "--warm-tokens") {
            out.warm_tokens = parse_u32(value(), "warm tokens");
        } else if (option == "--expert-pin-mib") {
            out.expert_pin_mib = parse_u32(value(), "expert pin MiB");
        } else if (option == "--lossless") {
            out.lossless = true;
        } else if (option == "--reference") {
            out.reference = true;
        } else if (option == "--observe") {
            out.observe = value();
        } else if (option == "--reproducible") {
            out.fast_paths = false;
        } else if (option == "--expert-a4-activations") {
            out.expert_a4_activations = true;
        } else if (option == "--fast-path") {
            const std::string_view name = value();
            const std::optional<infernix::FastPath> path = infernix::parse_fast_path(name);
            if (!path) { throw std::invalid_argument("unknown fast path " + std::string(name)); }
            out.candidates.push_back(*path);
        } else {
            throw std::invalid_argument("unknown option " + std::string(option));
        }
    }
    if (out.prompts.empty()) { throw std::invalid_argument("--prompts needs a length"); }
    if (out.forced > infernix::kMaximumConstraintSteps) {
        throw std::invalid_argument("--forced exceeds the constraint step limit");
    }
    if (out.concurrency > 8) { throw std::invalid_argument("--concurrency must be 1..8"); }
    if (out.warm_tokens != 0 && !out.prefix_reuse) {
        throw std::invalid_argument("--warm-tokens needs --prefix-reuse");
    }
    if (out.lossless && !out.observe.empty()) {
        throw std::invalid_argument("--lossless and --observe exclude each other");
    }
    if (out.lossless && out.expert_a4_activations) {
        throw std::invalid_argument("--lossless and --expert-a4-activations exclude each other");
    }
    if (out.reference && (out.lossless || out.observe.empty())) {
        throw std::invalid_argument("--reference scores the design route against a reference: give --observe, not "
                                    "--lossless");
    }
    return out;
}

infernix::KvCacheStorage kv_storage(std::string_view name) {
    if (name == "int8") { return infernix::KvCacheStorage::Int8Group64; }
    if (name == "bf16") { return infernix::KvCacheStorage::BFloat16; }
    if (name == "fp8") { return infernix::KvCacheStorage::Fp8E4M3Row256; }
    if (name == "nvfp4") { return infernix::KvCacheStorage::Nvfp4Group16; }
    if (name == "k8v4") { return infernix::KvCacheStorage::Fp8KeyNvfp4Value; }
    if (name == "vq2") { return infernix::KvCacheStorage::Vq2; }
    if (name == "k4v2") { return infernix::KvCacheStorage::Q4KeyVq2Value; }
    throw std::invalid_argument("--kv-dtype must be int8, bf16, fp8, nvfp4, k8v4, vq2 or k4v2");
}

// The artifact as the runs record it, so a run and its reference can be matched.
std::string artifact_path(const std::string& artifact) {
    return std::filesystem::absolute(artifact).lexically_normal().string();
}

void write_string(std::ostream& out, const std::string& text) {
    out << '"';
    for (const char c : text) {
        if (c == '\\' || c == '"') { out << '\\'; }
        out << c;
    }
    out << '"';
}

template <class Values>
void write_list(std::ostream& out, const Values& values) {
    out << '[';
    bool first = true;
    for (const auto& value : values) {
        out << (first ? "" : ", ") << value;
        first = false;
    }
    out << ']';
}

// The segments: back to back from the corpus's first token, cycling through the prompt lengths, so
// no two segments share a token and each is its own text. Segment i of a run is the set's segment
// --first-segment + i.
struct Segments {
    const std::vector<infernix::TokenId>& tokens;
    const Options& options;
    std::vector<std::size_t> offsets; // the set's segments 0 .. first + segments - 1

    Segments(const std::vector<infernix::TokenId>& corpus, const Options& run)
        : tokens(corpus), options(run) {
        std::size_t next = 0;
        for (std::uint32_t segment = 0; segment < options.first_segment + options.segments; ++segment) {
            offsets.push_back(next);
            next += set_prompt_tokens(segment) + options.forced;
        }
        if (next > tokens.size()) {
            throw std::runtime_error("the corpus holds " + std::to_string(tokens.size()) + " tokens; " +
                                     std::to_string(options.first_segment + options.segments) +
                                     " segments need " + std::to_string(next));
        }
    }

    [[nodiscard]] std::size_t offset(std::uint32_t segment) const {
        return offsets[options.first_segment + segment];
    }
    [[nodiscard]] std::uint32_t prompt_tokens(std::uint32_t segment) const {
        return set_prompt_tokens(options.first_segment + segment);
    }
    [[nodiscard]] std::uint32_t set_prompt_tokens(std::uint32_t set_segment) const {
        return options.prompts[set_segment % options.prompts.size()];
    }
    [[nodiscard]] std::vector<infernix::TokenId> prompt(std::uint32_t segment) const {
        const auto begin = tokens.begin() + static_cast<std::ptrdiff_t>(offset(segment));
        return {begin, begin + prompt_tokens(segment)};
    }
    // The forced tokens: the corpus continuation after the prompt.
    [[nodiscard]] std::vector<infernix::TokenId> forced(std::uint32_t segment, std::uint32_t count) const {
        const auto begin =
            tokens.begin() + static_cast<std::ptrdiff_t>(offset(segment) + prompt_tokens(segment));
        return {begin, begin + count};
    }
};

std::vector<infernix::TokenId> read_tokens(infernix::Engine& engine, const Options& options) {
    std::ifstream corpus(options.corpus, std::ios::binary);
    if (!corpus) { throw std::runtime_error("cannot read " + options.corpus); }
    std::stringstream text;
    text << corpus.rdbuf();
    std::vector<infernix::TokenId> tokens = engine.tokenize_text(text.str());
    return tokens;
}

// The reference: the lossless route's BF16-KV scoring in one pass per segment.
void write_reference(const Options& options, std::uint32_t longest, std::ostream& out) {
    infernix::EngineOptions engine_options;
    engine_options.artifact_path   = options.artifact;
    engine_options.purpose         = infernix::EnginePurpose::CausalScoring;
    engine_options.max_context     = (longest + options.forced + 64 + 127) / 128 * 128;
    engine_options.kv_capacity     = infernix::KvCapacityPolicy::explicit_capacity(engine_options.max_context);
    engine_options.kv_cache        = infernix::KvCacheStorage::BFloat16;
    engine_options.prefill_chunk   = engine_options.max_context;
    engine_options.lossless        = true;
    engine_options.fast_paths      = false;
    if (!options.ngram_volume.empty()) { engine_options.ngram_volume_path = options.ngram_volume; }
    infernix::Engine engine(engine_options);
    const std::vector<infernix::TokenId> tokens = read_tokens(engine, options);
    const Segments segments(tokens, options);

    constexpr std::uint32_t kTop = infernix::kMaximumObservedTokens;
    static_assert(kTop <= infernix::kMaximumScoreTopTokens);
    out << "{\"artifact\": ";
    write_string(out, artifact_path(options.artifact));
    out << ", \"kind\": \"reference\", \"forced_tokens\": " << options.forced
        << ", \"top_tokens\": " << kTop << ", \"segments\": [";
    for (std::uint32_t segment = 0; segment < options.segments; ++segment) {
        std::vector<infernix::TokenId> input = segments.prompt(segment);
        const std::vector<infernix::TokenId> forced = segments.forced(segment, options.forced);
        input.insert(input.end(), forced.begin(), forced.end());
        const infernix::ScoreResult scored = engine.score_tokens(
            std::move(input), segments.prompt_tokens(segment), infernix::ScoreOptions{.top_k = kTop});
        out << (segment ? ", " : "") << "{\"offset\": " << segments.offset(segment)
            << ", \"prompt_tokens\": " << segments.prompt_tokens(segment) << ", \"logprobs\": ";
        write_list(out, scored.logprobs);
        out << ", \"top_ids\": ";
        write_list(out, scored.top_ids);
        out << ", \"top_logprobs\": ";
        write_list(out, scored.top_logprobs);
        out << '}';
        if ((segment + 1) % 16 == 0) {
            std::cerr << "reference segment " << segment + 1 << "/" << options.segments << '\n';
        }
    }
    out << "]}\n";
}

// Each step's observe list from a reference (written by --lossless for the same segments).
std::vector<std::vector<std::vector<infernix::TokenId>>> read_observe(const Options& options,
                                                                      const Segments& segments) {
    std::ifstream in(options.observe);
    if (!in) { throw std::runtime_error("cannot read " + options.observe); }
    const nlohmann::json reference = nlohmann::json::parse(in);
    if (reference.at("kind") != "reference" || reference.at("forced_tokens") != options.forced) {
        throw std::runtime_error(options.observe + " is not a reference for " +
                                 std::to_string(options.forced) + " forced tokens");
    }
    // Another artifact's top tokens would score this run against the wrong distribution.
    const std::filesystem::path recorded = reference.at("artifact").get<std::string>();
    std::error_code error;
    if (!std::filesystem::equivalent(recorded, options.artifact, error) &&
        artifact_path(recorded.string()) != artifact_path(options.artifact)) {
        throw std::runtime_error(options.observe + " was written for " + recorded.string() + ", not " +
                                 options.artifact);
    }
    const auto& list = reference.at("segments");
    if (list.size() != options.segments) {
        throw std::runtime_error(options.observe + " holds a different number of segments");
    }
    const std::size_t top = reference.at("top_tokens").get<std::size_t>();
    std::vector<std::vector<std::vector<infernix::TokenId>>> observe(options.segments);
    for (std::uint32_t segment = 0; segment < options.segments; ++segment) {
        const auto& item = list[segment];
        if (item.at("offset") != segments.offset(segment) ||
            item.at("prompt_tokens") != segments.prompt_tokens(segment)) {
            throw std::runtime_error(options.observe + " was written for other segments");
        }
        const auto ids = item.at("top_ids").get<std::vector<infernix::TokenId>>();
        if (ids.size() != top * options.forced) {
            throw std::runtime_error(options.observe + " has a short top-token list");
        }
        for (std::uint32_t step = 0; step < options.forced; ++step) {
            observe[segment].emplace_back(ids.begin() + static_cast<std::ptrdiff_t>(step * top),
                                          ids.begin() + static_cast<std::ptrdiff_t>((step + 1) * top));
        }
    }
    return observe;
}

// The design reference's observed run (--reference --observe REFERENCE): the design route
// (EngineOptions::reference; docs/maintainer/lossless-route.md §3b) scores each segment in one pass. Its
// output depends on neither chunking nor prefill versus decode, so each forced position's log-probability,
// the observed log-probabilities of the reference's top tokens and its own largest are those a decode run
// reports; the gate takes the run as the base arm against which a route's processing cost is measured.
void write_design(const Options& options, std::uint32_t longest, std::ostream& out) {
    infernix::EngineOptions engine_options;
    engine_options.artifact_path         = options.artifact;
    engine_options.purpose               = infernix::EnginePurpose::CausalScoring;
    engine_options.max_context           = (longest + options.forced + 64 + 127) / 128 * 128;
    engine_options.kv_capacity           = infernix::KvCapacityPolicy::explicit_capacity(engine_options.max_context);
    engine_options.kv_cache              = kv_storage(options.kv);
    engine_options.prefill_chunk         = engine_options.max_context;
    engine_options.lossless              = true;
    engine_options.reference             = true;
    engine_options.expert_a4_activations = options.expert_a4_activations;
    engine_options.fast_paths            = false;
    if (!options.ngram_volume.empty()) { engine_options.ngram_volume_path = options.ngram_volume; }
    infernix::Engine engine(engine_options);
    const std::vector<infernix::TokenId> tokens = read_tokens(engine, options);
    const Segments segments(tokens, options);
    const auto observe = read_observe(options, segments);
    const std::uint32_t top = observe.empty() || observe[0].empty() ? 0U : static_cast<std::uint32_t>(observe[0][0].size());

    out << "{\"artifact\": ";
    write_string(out, artifact_path(options.artifact));
    out << ", \"mode\":\"reference\", \"kv_dtype\": \"" << options.kv
        << "\", \"expert_a4_activations\": " << (options.expert_a4_activations ? "true" : "false")
        << ", \"fast_paths\": [], \"forced_tokens\": " << options.forced << ", \"segments\": [";
    std::uint64_t underflows = 0;
    for (std::uint32_t segment = 0; segment < options.segments; ++segment) {
        std::vector<infernix::TokenId> input = segments.prompt(segment);
        const std::vector<infernix::TokenId> forced = segments.forced(segment, options.forced);
        input.insert(input.end(), forced.begin(), forced.end());
        infernix::ScoreOptions score{.top_k = 1, .candidates_per_position = top};
        for (const auto& step : observe[segment]) { score.candidates.insert(score.candidates.end(), step.begin(), step.end()); }
        const infernix::ScoreResult scored =
            engine.score_tokens(std::move(input), segments.prompt_tokens(segment), std::move(score));
        out << (segment ? ", " : "") << "{\"offset\": " << segments.offset(segment)
            << ", \"prompt_tokens\": " << segments.prompt_tokens(segment) << ", \"logprobs\": [";
        for (std::size_t i = 0; i < scored.logprobs.size(); ++i) {
            // As a decode run reports it: an FP32 probability below the smallest normal at it.
            const double p = std::exp(static_cast<double>(scored.logprobs[i]));
            underflows += p < 1.0e-37 ? 1U : 0U;
            out << (i ? ", " : "") << std::log(std::max(p, 1.0e-37));
        }
        out << "], \"observed\": ";
        write_list(out, scored.candidate_logprobs);
        out << ", \"top_logprobs\": ";
        write_list(out, scored.top_logprobs);
        out << '}';
        if ((segment + 1) % 16 == 0) { std::cerr << "design segment " << segment + 1 << "/" << options.segments << '\n'; }
    }
    out << "], \"observed_reference\": ";
    write_string(out, options.observe);
    out << ", \"underflows\": " << underflows << ", \"reused_prompt_tokens\": 0, \"preemptions\": 0"
        << ", \"replayed_tokens\": 0}\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse(argc, argv);
        const std::uint32_t longest = *std::max_element(options.prompts.begin(), options.prompts.end());
        std::ofstream out(options.output);
        if (!out) { throw std::runtime_error("cannot write " + options.output); }
        out << std::setprecision(17);
        const auto started = std::chrono::steady_clock::now();
        if (options.lossless || options.reference) {
            if (options.reference) {
                write_design(options, longest, out);
            } else {
                write_reference(options, longest, out);
            }
            if (!out) { throw std::runtime_error("cannot write " + options.output); }
            std::cerr << (options.reference ? "design run" : "reference") << " for " << options.segments
                      << " segments in "
                      << std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count()
                      << " s\n";
            return 0;
        }

        infernix::EngineOptions engine_options;
        engine_options.artifact_path   = options.artifact;
        engine_options.max_context     = (longest + options.forced + 64 + 127) / 128 * 128;
        engine_options.kv_capacity     = infernix::KvCapacityPolicy::explicit_capacity(
            options.kv_tokens != 0 ? options.kv_tokens
                                   : engine_options.max_context * options.concurrency);
        engine_options.kv_cache        = kv_storage(options.kv);
        engine_options.max_concurrency = options.concurrency;
        // A group's requests are submitted together and wait behind each other's prefills, which on an
        // artifact reading its experts from the SSD tier take minutes (Qwen3.8-Flash-Next FP8 at
        // concurrency 8 outlasted the default 30 s).
        engine_options.pending_timeout_ms = 3'600'000;
        engine_options.prefill_chunk   = options.prefill_chunk;
        engine_options.fast_paths            = options.fast_paths;
        engine_options.fast_path_candidates  = options.candidates;
        engine_options.expert_a4_activations = options.expert_a4_activations;
        if (!options.ngram_volume.empty()) { engine_options.ngram_volume_path = options.ngram_volume; }
        if (options.expert_pin_mib) { engine_options.expert_pin_bytes = *options.expert_pin_mib << 20; }
        if (!options.prefix_reuse) {
            engine_options.context_cache.enabled             = false;
            engine_options.context_cache.host_capacity_bytes = 0;
        }
        if (options.mode == "dflash2" || options.mode == "ngram") {
            engine_options.speculative.backend       = infernix::SpeculativeBackend::DFlash2;
            engine_options.speculative.draft_tokens  = 7;
            engine_options.speculative.proposal_head = infernix::ProposalHead::Optimized;
            if (options.mode == "ngram") {
                engine_options.speculative.ngram_draft_tokens = 15;
                engine_options.speculative.ngram_min_match    = 12;
            }
        } else if (options.mode == "mtp" || options.mode == "mtp-ngram") {
            engine_options.speculative.backend       = infernix::SpeculativeBackend::Mtp;
            engine_options.speculative.draft_tokens  = 4;
            engine_options.speculative.proposal_head = infernix::ProposalHead::Optimized;
            // MTP with n-gram copies: the round's longest copy sets its width (adaptive-speculation)
            // on models without DFlash2.
            if (options.mode == "mtp-ngram") {
                engine_options.speculative.ngram_draft_tokens = 15;
                engine_options.speculative.ngram_min_match    = 12;
            }
        } else if (options.mode != "plain") {
            throw std::invalid_argument("--mode must be plain, dflash2, ngram, mtp or mtp-ngram");
        }
        infernix::Engine engine(engine_options);
        const std::vector<infernix::TokenId> tokens = read_tokens(engine, options);
        const Segments segments(tokens, options);
        const auto observe = options.observe.empty()
                                 ? std::vector<std::vector<std::vector<infernix::TokenId>>>{}
                                 : read_observe(options, segments);

        out << "{\"artifact\": ";
        write_string(out, artifact_path(options.artifact));
        out << ", \"mode\":\"" << options.mode << "\", \"kv_dtype\": \"" << options.kv
            << "\", \"expert_a4_activations\": " << (options.expert_a4_activations ? "true" : "false")
            << ", \"fast_paths\": [";
        const infernix::LoadSummary load = engine.load_summary();
        for (std::size_t i = 0; i < load.fast_paths.size(); ++i) {
            out << (i ? ", " : "") << '"' << load.fast_paths[i] << '"';
        }
        out << "], \"forced_tokens\": " << options.forced << ", \"segments\": [";

        std::uint64_t underflows = 0;
        std::uint64_t reused     = 0;
        const auto request_of = [&](std::uint32_t segment, std::uint32_t forced) {
            infernix::RequestOptions request;
            request.execution.requested_output_tokens = forced;
            request.execution.sampling.temperature    = 0.0F;
            request.execution.allow_prefix_reuse      = options.prefix_reuse;
            request.stop.include_model_defaults       = false;
            for (const infernix::TokenId token : segments.forced(segment, forced)) {
                request.execution.constraint.steps.push_back({token});
            }
            if (!observe.empty()) {
                request.execution.constraint.observe.assign(observe[segment].begin(),
                                                            observe[segment].begin() + forced);
            }
            return request;
        };
        double prefill_seconds = 0, decode_seconds = 0;
        const auto write_segment = [&](std::uint32_t segment, const infernix::GenerationResult& result) {
            if (result.constrained_draws.size() != options.forced) {
                throw std::runtime_error("segment " + std::to_string(segment) + " forced " +
                                         std::to_string(result.constrained_draws.size()) + " of " +
                                         std::to_string(options.forced) + " tokens");
            }
            reused += result.reused_prompt_tokens;
            prefill_seconds += result.timings.prefill_seconds;
            decode_seconds += result.timings.decode_share_seconds;
            out << (segment ? ", " : "") << "{\"offset\": " << segments.offset(segment)
                << ", \"prompt_tokens\": " << segments.prompt_tokens(segment) << ", \"logprobs\": [";
            for (std::size_t i = 0; i < result.constrained_draws.size(); ++i) {
                const float mass = result.constrained_draws[i].mass;
                // An FP32 probability below the smallest normal is reported at it.
                underflows += mass < 1.0e-37F ? 1U : 0U;
                out << (i ? ", " : "") << std::log(std::max(static_cast<double>(mass), 1.0e-37));
            }
            out << ']';
            if (!observe.empty()) {
                // Step-major: each step's observed log-probabilities of the reference's top tokens.
                out << ", \"observed\": [";
                bool first = true;
                for (const infernix::ConstrainedDraw& draw : result.constrained_draws) {
                    for (const float value : draw.observed_logprobs) {
                        out << (first ? "" : ", ") << value;
                        first = false;
                    }
                }
                out << "], \"top_logprobs\": [";
                for (std::size_t i = 0; i < result.constrained_draws.size(); ++i) {
                    out << (i ? ", " : "") << result.constrained_draws[i].top_logprob;
                }
                out << ']';
            }
            out << '}';
            if ((segment + 1) % 16 == 0) { std::cerr << "segment " << segment + 1 << "/" << options.segments << '\n'; }
        };
        if (!options.prefix_reuse) {
            // Rolling: `concurrency` requests stay in flight, each lane taking the next segment as its
            // previous one completes, so no lane idles behind a group's longest prompt. Results are
            // written in segment order.
            std::vector<std::optional<infernix::GenerationResult>> results(options.segments);
            std::atomic<std::uint32_t> next{0};
            std::vector<std::exception_ptr> errors(options.concurrency);
            std::vector<std::thread> lanes;
            for (std::uint32_t lane = 0; lane < options.concurrency; ++lane) {
                lanes.emplace_back([&, lane] {
                    try {
                        for (std::uint32_t segment = next++; segment < options.segments; segment = next++) {
                            results[segment] = engine
                                                   .submit(engine.prepare_tokens(segments.prompt(segment), false),
                                                           request_of(segment, options.forced))
                                                   .wait();
                        }
                    } catch (...) {
                        errors[lane] = std::current_exception();
                        next         = options.segments; // stop the other lanes
                    }
                });
            }
            for (std::thread& lane : lanes) { lane.join(); }
            for (const std::exception_ptr& error : errors) {
                if (error) { std::rethrow_exception(error); }
            }
            for (std::uint32_t segment = 0; segment < options.segments; ++segment) {
                write_segment(segment, *results[segment]);
            }
        }
        for (std::uint32_t group = 0; options.prefix_reuse && group < options.segments; group += options.concurrency) {
            const std::uint32_t end = std::min(options.segments, group + options.concurrency);
            // Each warm request publishes its segment's prompt prefix (its first --warm-tokens tokens
            // when set) before the round; the measured requests then resume from them together.
            for (std::uint32_t segment = group; segment < end; ++segment) {
                std::vector<infernix::TokenId> warm = segments.prompt(segment);
                if (options.warm_tokens != 0 && options.warm_tokens < warm.size()) {
                    warm.resize(options.warm_tokens);
                    infernix::RequestOptions request;
                    request.execution.requested_output_tokens = 1;
                    request.execution.sampling.temperature    = 0.0F;
                    request.execution.allow_prefix_reuse      = true;
                    request.stop.include_model_defaults       = false;
                    (void)engine.generate(engine.prepare_tokens(std::move(warm), true), std::move(request));
                } else {
                    (void)engine.generate(engine.prepare_tokens(std::move(warm), true), request_of(segment, 1));
                }
            }
            std::vector<infernix::GenerationHandle> handles;
            for (std::uint32_t segment = group; segment < end; ++segment) {
                handles.push_back(engine.submit(engine.prepare_tokens(segments.prompt(segment), options.prefix_reuse),
                                                request_of(segment, options.forced)));
            }
            for (std::uint32_t segment = group; segment < end; ++segment) {
                write_segment(segment, handles[segment - group].wait());
            }
        }
        out << "], \"observed_reference\": ";
        write_string(out, options.observe);
        const infernix::RuntimeStats stats = engine.runtime_stats();
        out << ", \"underflows\": " << underflows << ", \"reused_prompt_tokens\": " << reused
            << ", \"preemptions\": " << stats.preemptions
            << ", \"replayed_tokens\": " << stats.replayed_tokens << "}\n";
        if (!out) { throw std::runtime_error("cannot write " + options.output); }
        const double seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        std::cerr << "forced " << static_cast<std::uint64_t>(options.segments) * options.forced
                  << " tokens in " << seconds << " s (" << underflows << " underflows; requests' prefill "
                  << prefill_seconds << " s summed over lanes, decode " << decode_seconds
                  << " s of shared rounds)\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "infernix_decode_quality_forced: " << error.what() << '\n';
        return 1;
    }
}
