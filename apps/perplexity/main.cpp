#include "infernix_build_id.h"
#include "corpus.h"
#include "distribution.h"
#include "evaluation.h"

#include "product/rope_yarn_options.h"
#include "infernix/engine.h"
#include "product/logging/logging.h"
#include "product/logging/pretty_format.h"
#include "product/logging/engine_diagnostics.h"
#include "product/logging/startup_log.h"

#include <nlohmann/json.hpp>
#include <spdlog/logger.h>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using json  = nlohmann::json;
using infernix::perplexity::CorpusSelection;
using infernix::perplexity::DivergenceAggregate;
using infernix::perplexity::ScoreAggregate;
using infernix::perplexity::WindowPlan;

struct Options {
    bool help_requested = false;
    std::filesystem::path artifact;
    std::optional<std::filesystem::path> corpus;
    std::optional<std::filesystem::path> text;
    std::optional<std::filesystem::path> output;
    std::optional<std::filesystem::path> save_top_tokens;
    std::optional<std::filesystem::path> kl_reference;
    float rope_yarn_factor              = 1.0F;
    std::uint32_t context               = 4096;
    std::uint32_t stride                = 2048;
    std::optional<std::uint32_t> prefill_chunk;
    int device                          = 0;
    infernix::KvCacheStorage kv           = infernix::KvCacheStorage::Fp8E4M3Row256;
    bool quick                          = false;
    bool original_int8_prefill_kernel   = false;
    infernix::PrefillPv8 prefill_8bit_pv          = infernix::PrefillPv8::Auto;
    bool original_nvfp4_prefill_kernel  = false;
    bool a16_activations                = false;
    infernix::product::LogLevel log_level = infernix::product::LogLevel::Info;
};

std::string usage_text() {
    return "usage: infernix-perplexity <model.ninfer> "
           "(--corpus <manifest.json> [--quick] | --text <utf8-file>)\n"
           "       [--context N] [--stride N] [--device N]\n"
           "       [--prefill-chunk N] (tokens per scoring prefill pass, a multiple of 128;\n"
           "        default the Engine's 1024)\n"
           "       [--rope-yarn-factor F] (startup-fixed, finite [1,4], default 1; ceiling only)\n"
           "       [--kv-dtype bf16|int8|fp8|nvfp4|k8v4|vq2|k4v2] (default fp8)\n"
           "       [--use-original-int8-prefill-kernel (int8 only; default fast kernel)]\n"
           "       [--prefill-8bit-pv | --no-prefill-8bit-pv] (8-bit P*V for INT8 and K4V2;\n"
           "        FP16 for NVFP4, K8V4 and VQ2, which default to 8-bit; see infernix-serve)\n"
           "       [--use-original-nvfp4-prefill-kernel (nvfp4 only; default fast kernel)]\n"
           "       [--a16-activations] (16-bit activations in every linear: the reference for\n"
           "        measuring activation quantization; at least 3x slower on NVFP4 artifacts)\n"
           "       [--output <directory>]\n"
           "       [--save-top-tokens <file>] (record each position's 32 most probable tokens)\n"
           "       [--kl-reference <file>] (KL divergence from a --save-top-tokens run of the\n"
           "        same corpus, context and stride, by context length)\n"
           "       [--log-level trace|debug|info|warning|error|critical|off]\n";
}

[[noreturn]] void usage_error(std::string_view message) {
    throw std::invalid_argument(std::string(message));
}

template <class Integer>
Integer parse_integer(std::string_view text, const char* label) {
    Integer value{};
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size()) {
        usage_error(std::string("invalid ") + label + ": " + std::string(text));
    }
    return value;
}

Options parse_options(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--help") {
        return Options{.help_requested = true};
    }
    if (argc < 2 || std::string_view(argv[1]).starts_with("--")) {
        usage_error("artifact path is required");
    }
    Options out;
    out.artifact = argv[1];
    for (int i = 2; i < argc; ++i) {
        const std::string_view option = argv[i];
        const auto value              = [&](const char* label) -> std::string_view {
            if (++i >= argc) { usage_error(std::string(label) + " requires a value"); }
            return argv[i];
        };
        if (option == "--corpus") {
            out.corpus = std::filesystem::path(value("--corpus"));
        } else if (option == "--text") {
            out.text = std::filesystem::path(value("--text"));
        } else if (option == "--quick") {
            out.quick = true;
        } else if (option == "--rope-yarn-factor") {
            out.rope_yarn_factor = infernix::product::parse_rope_yarn_factor(value("--rope-yarn-factor"));
        } else if (option == "--context") {
            out.context = parse_integer<std::uint32_t>(value("--context"), "context");
        } else if (option == "--stride") {
            out.stride = parse_integer<std::uint32_t>(value("--stride"), "stride");
        } else if (option == "--prefill-chunk") {
            out.prefill_chunk = parse_integer<std::uint32_t>(value("--prefill-chunk"), "prefill-chunk");
        } else if (option == "--device") {
            out.device = parse_integer<int>(value("--device"), "device");
        } else if (option == "--use-original-int8-prefill-kernel") {
            out.original_int8_prefill_kernel = true;
        } else if (option == "--prefill-8bit-pv") {
            out.prefill_8bit_pv = infernix::PrefillPv8::On;
        } else if (option == "--no-prefill-8bit-pv") {
            out.prefill_8bit_pv = infernix::PrefillPv8::Off;
        } else if (option == "--use-original-nvfp4-prefill-kernel") {
            out.original_nvfp4_prefill_kernel = true;
        } else if (option == "--a16-activations") {
            out.a16_activations = true;
        } else if (option == "--kv-dtype") {
            const std::string_view dtype = value("--kv-dtype");
            if (dtype == "bf16") {
                out.kv = infernix::KvCacheStorage::BFloat16;
            } else if (dtype == "int8") {
                out.kv = infernix::KvCacheStorage::Int8Group64;
            } else if (dtype == "fp8") {
                out.kv = infernix::KvCacheStorage::Fp8E4M3Row256;
            } else if (dtype == "nvfp4") {
                out.kv = infernix::KvCacheStorage::Nvfp4Group16;
            } else if (dtype == "k8v4") {
                out.kv = infernix::KvCacheStorage::Fp8KeyNvfp4Value;
            } else if (dtype == "vq2") {
                out.kv = infernix::KvCacheStorage::Vq2;
            } else if (dtype == "k4v2") {
                out.kv = infernix::KvCacheStorage::Q4KeyVq2Value;
            } else {
                usage_error("--kv-dtype must be bf16, int8, fp8, nvfp4, k8v4, vq2, or k4v2");
            }
        } else if (option == "--output") {
            out.output = std::filesystem::path(value("--output"));
        } else if (option == "--save-top-tokens") {
            out.save_top_tokens = std::filesystem::path(value("--save-top-tokens"));
        } else if (option == "--kl-reference") {
            out.kl_reference = std::filesystem::path(value("--kl-reference"));
        } else if (option == "--log-level") {
            out.log_level = infernix::product::parse_log_level(value("--log-level"));
        } else {
            usage_error("unknown option: " + std::string(option));
        }
    }
    if (out.corpus.has_value() == out.text.has_value()) {
        usage_error("exactly one of --corpus and --text is required");
    }
    if (out.quick && !out.corpus) { usage_error("--quick requires --corpus"); }
    if (out.context < 2 || out.stride == 0 || out.stride >= out.context) {
        usage_error("context/stride must satisfy context>=2 and 1<=stride<context");
    }
    return out;
}

std::string kv_name(infernix::KvCacheStorage value) {
    switch (value) {
    case infernix::KvCacheStorage::BFloat16:
        return "bf16";
    case infernix::KvCacheStorage::Int8Group64:
        return "int8-g64";
    case infernix::KvCacheStorage::Fp8E4M3Row256:
        return "fp8-e4m3-r256";
    case infernix::KvCacheStorage::Nvfp4Group16:
        return "nvfp4";
    case infernix::KvCacheStorage::Fp8KeyNvfp4Value:
        return "k8v4";
    case infernix::KvCacheStorage::Vq2:
        return "vq2";
    case infernix::KvCacheStorage::Q4KeyVq2Value:
        return "k4v2";
    }
    throw std::logic_error("unknown KV dtype");
}

std::string safe_component(std::string_view value) {
    std::string out;
    out.reserve(value.size());
    for (const unsigned char c : value) {
        out.push_back(std::isalnum(c) || c == '-' || c == '_' || c == '.' ? static_cast<char>(c)
                                                                          : '-');
    }
    return out.empty() ? "unknown" : out;
}

std::string timestamp() {
    const std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm utc{};
#ifdef _WIN32
    gmtime_s(&utc, &now);
#else
    gmtime_r(&now, &utc);
#endif
    std::ostringstream out;
    out << std::put_time(&utc, "%Y%m%d-%H%M%S");
    return out.str();
}

std::filesystem::path prepare_output_directory(const Options& options,
                                               const infernix::LoadSummary& load,
                                               const CorpusSelection& corpus) {
    std::filesystem::path output = options.output.value_or(
        std::filesystem::path("profiles/perplexity") / safe_component(load.model_name) /
        safe_component(load.prefill_signature) / kv_name(options.kv) /
        safe_component(corpus.corpus_id) / safe_component(corpus.mode) / timestamp());
    if (std::filesystem::exists(output)) {
        if (!std::filesystem::is_directory(output) ||
            std::filesystem::directory_iterator(output) != std::filesystem::directory_iterator()) {
            throw std::runtime_error("output directory exists and is not empty: " +
                                     output.string());
        }
    } else if (!std::filesystem::create_directories(output)) {
        throw std::runtime_error("cannot create output directory: " + output.string());
    }
    return std::filesystem::absolute(output).lexically_normal();
}

double seconds_since(Clock::time_point begin) {
    return std::chrono::duration<double>(Clock::now() - begin).count();
}

json aggregate_json(const ScoreAggregate& value) {
    return json{{"scored_tokens", value.scored_tokens},
                {"total_nll", value.total_nll},
                {"mean_nll", value.mean_nll()},
                {"perplexity", value.ppl()}};
}

struct EvaluationStream {
    infernix::perplexity::CorpusStream source;
    std::vector<infernix::TokenId> tokens;
    std::vector<WindowPlan> windows;
};

int run(const Options& options, const std::shared_ptr<spdlog::logger>& logger,
        infernix::product::StartupLogRenderer& startup_log,
        const std::shared_ptr<infernix::product::TerminalProgress>& progress) {
    const Clock::time_point total_started = Clock::now();
    infernix::EngineOptions engine_options;
    engine_options.artifact_path    = options.artifact;
    engine_options.purpose          = infernix::EnginePurpose::CausalScoring;
    engine_options.device           = options.device;
    engine_options.max_context      = options.context;
    engine_options.rope_yarn_factor  = options.rope_yarn_factor;
    if (options.prefill_chunk) { engine_options.prefill_chunk = *options.prefill_chunk; }
    engine_options.kv_cache         = options.kv;
    engine_options.original_int8_prefill_kernel = options.original_int8_prefill_kernel;
    engine_options.prefill_8bit_pv              = options.prefill_8bit_pv;
    engine_options.original_nvfp4_prefill_kernel = options.original_nvfp4_prefill_kernel;
    engine_options.a16_activations               = options.a16_activations;
    engine_options.startup_observer = startup_log.observer();
    engine_options.diagnostic_observer = infernix::product::engine_diagnostic_observer(logger);
    infernix::Engine engine(std::move(engine_options));
    const infernix::LoadSummary load = engine.load_summary();
    startup_log.engine_ready(load);

    const Clock::time_point preflight_started = Clock::now();
    logger->info("preparing corpus");
    CorpusSelection corpus = options.corpus
                                 ? infernix::perplexity::load_corpus(*options.corpus, options.quick)
                                 : infernix::perplexity::load_custom_text(*options.text);
    std::vector<EvaluationStream> streams;
    streams.reserve(corpus.streams.size());
    std::uint64_t total_scored_tokens = 0;
    std::uint64_t total_input_tokens  = 0;
    std::uint64_t total_windows       = 0;
    for (auto& source : corpus.streams) {
        std::vector<infernix::TokenId> tokens = engine.tokenize_text(source.text);
        if (tokens.size() < 2) {
            throw std::runtime_error("stream tokenized to fewer than two tokens: " + source.id);
        }
        std::vector<WindowPlan> windows =
            infernix::perplexity::plan_windows(tokens.size(), options.context, options.stride);
        total_input_tokens += static_cast<std::uint64_t>(tokens.size());
        total_scored_tokens += static_cast<std::uint64_t>(tokens.size() - 1);
        total_windows += static_cast<std::uint64_t>(windows.size());
        streams.push_back(EvaluationStream{.source  = std::move(source),
                                           .tokens  = std::move(tokens),
                                           .windows = std::move(windows)});
    }
    const double preflight_seconds = seconds_since(preflight_started);
    logger->info("corpus ready | {} streams | {} input tokens | {} scored tokens | {} windows | {}",
                 infernix::product::format_pretty_count(streams.size()),
                 infernix::product::format_pretty_count(total_input_tokens),
                 infernix::product::format_pretty_count(total_scored_tokens),
                 infernix::product::format_pretty_count(total_windows),
                 infernix::product::format_pretty_duration(preflight_seconds));

    const std::filesystem::path output_directory = prepare_output_directory(options, load, corpus);
    const infernix::perplexity::ReferenceHeader reference_header{.corpus_id = corpus.corpus_id,
                                                                .mode      = corpus.mode,
                                                                .context   = options.context,
                                                                .stride    = options.stride};
    std::optional<infernix::perplexity::ReferenceWriter> top_tokens;
    if (options.save_top_tokens) { top_tokens.emplace(*options.save_top_tokens, reference_header); }
    std::optional<infernix::perplexity::ReferenceReader> kl_reference;
    if (options.kl_reference) {
        kl_reference.emplace(*options.kl_reference);
        const infernix::perplexity::ReferenceHeader& saved = kl_reference->header();
        if (saved.corpus_id != reference_header.corpus_id || saved.mode != reference_header.mode ||
            saved.context != reference_header.context || saved.stride != reference_header.stride) {
            throw std::runtime_error("--kl-reference was recorded for corpus " + saved.corpus_id +
                                     " / " + saved.mode + " with context/stride " +
                                     std::to_string(saved.context) + "/" +
                                     std::to_string(saved.stride));
        }
    }
    // The evaluated run's own most probable token, for top-1 agreement.
    const std::uint32_t own_top_tokens =
        top_tokens ? infernix::perplexity::kReferenceTopTokens : (kl_reference ? 1U : 0U);
    DivergenceAggregate overall_divergence;
    const Clock::time_point scoring_started      = Clock::now();
    logger->info("scoring | {} streams | {} tokens | {} windows",
                 infernix::product::format_pretty_count(streams.size()),
                 infernix::product::format_pretty_count(total_scored_tokens),
                 infernix::product::format_pretty_count(total_windows));
    Clock::time_point next_progress = scoring_started + std::chrono::seconds(10);
    ScoreAggregate overall;
    std::map<std::string, ScoreAggregate> domains;
    json stream_reports             = json::array();
    std::uint64_t completed_windows = 0;

    for (std::size_t stream_index = 0; stream_index < streams.size(); ++stream_index) {
        EvaluationStream& stream = streams[stream_index];
        std::ostringstream stream_status;
        stream_status << "  scoring [" << stream_index + 1 << '/' << streams.size() << "] "
                      << infernix::product::format_pretty_text(stream.source.id) << " | "
                      << infernix::product::format_pretty_count(stream.tokens.size()) << " tokens | "
                      << infernix::product::format_pretty_count(stream.windows.size()) << " windows";
        if (progress->enabled()) {
            progress->update(stream_status.str());
        } else {
            logger->debug("{}", stream_status.str());
        }
        const Clock::time_point stream_started = Clock::now();
        ScoreAggregate stream_score;
        DivergenceAggregate stream_divergence;
        json window_reports = json::array();
        for (std::size_t window_index = 0; window_index < stream.windows.size(); ++window_index) {
            const WindowPlan& window = stream.windows[window_index];
            std::vector<infernix::TokenId> input(
                stream.tokens.begin() + static_cast<std::ptrdiff_t>(window.input_begin),
                stream.tokens.begin() + static_cast<std::ptrdiff_t>(window.input_end));
            const Clock::time_point window_started = Clock::now();
            const std::uint64_t input_hash = infernix::perplexity::token_fingerprint(input);
            infernix::ScoreOptions score_options{.top_k = own_top_tokens};
            std::optional<infernix::perplexity::ReferenceWindow> reference;
            if (kl_reference) {
                reference = kl_reference->next();
                if (reference->stream_index != stream_index ||
                    reference->target_begin != window.target_begin ||
                    reference->target_end != window.target_end ||
                    reference->input_hash != input_hash) {
                    throw std::runtime_error("--kl-reference window " +
                                             std::to_string(window_index) + " of " +
                                             stream.source.id + " scored different tokens");
                }
                score_options.candidates_per_position = kl_reference->header().top_k;
                score_options.candidates              = reference->top_ids;
            }
            infernix::ScoreResult scored;
            try {
                scored = engine.score_tokens(std::move(input), window.first_target,
                                             std::move(score_options));
            } catch (const std::exception& error) {
                throw std::runtime_error("scoring " + stream.source.id + " window " +
                                         std::to_string(window_index) + " failed: " + error.what());
            }
            const std::vector<float>& logprobs = scored.logprobs;
            const std::size_t expected         = window.target_end - window.target_begin;
            if (logprobs.size() != expected) {
                throw std::runtime_error("scoring returned an invalid target count for " +
                                         stream.source.id);
            }
            if (top_tokens) {
                top_tokens->write({.stream_index    = static_cast<std::uint32_t>(stream_index),
                                   .target_begin    = window.target_begin,
                                   .target_end      = window.target_end,
                                   .input_hash      = input_hash,
                                   .target_logprobs = logprobs,
                                   .top_ids         = scored.top_ids,
                                   .top_logprobs    = scored.top_logprobs});
            }
            if (reference) {
                const std::size_t k = kl_reference->header().top_k;
                for (std::size_t position = 0; position < expected; ++position) {
                    const std::span<const float> reference_top(
                        reference->top_logprobs.data() + position * k, k);
                    const std::span<const float> evaluated_top(
                        scored.candidate_logprobs.data() + position * k, k);
                    const bool top1_match =
                        scored.top_ids[position * own_top_tokens] == reference->top_ids[position * k];
                    // The predicted token at window input index first_target + position follows
                    // that many tokens of context.
                    stream_divergence.add(window.first_target + position,
                                          infernix::perplexity::position_divergence(
                                              reference_top, evaluated_top, top1_match,
                                              reference->target_logprobs[position],
                                              logprobs[position]));
                }
            }
            ScoreAggregate window_score;
            window_score.add(logprobs);
            stream_score.add(window_score);
            overall.add(window_score);
            domains[stream.source.domain].add(window_score);
            ++completed_windows;
            json window_report            = aggregate_json(window_score);
            window_report["index"]        = window_index;
            window_report["input_begin"]  = window.input_begin;
            window_report["input_end"]    = window.input_end;
            window_report["target_begin"] = window.target_begin;
            window_report["target_end"]   = window.target_end;
            window_report["first_target"] = window.first_target;
            window_report["seconds"]      = seconds_since(window_started);
            window_reports.push_back(std::move(window_report));

            if (Clock::now() >= next_progress) {
                const double elapsed = seconds_since(scoring_started);
                const double rate    = static_cast<double>(overall.scored_tokens) / elapsed;
                const std::uint64_t remaining = total_scored_tokens - overall.scored_tokens;
                const double eta = rate > 0 ? static_cast<double>(remaining) / rate : 0.0;
                std::ostringstream line;
                line << "scoring | " << infernix::product::format_pretty_count(overall.scored_tokens)
                     << '/' << infernix::product::format_pretty_count(total_scored_tokens)
                     << " tokens | " << completed_windows << '/' << total_windows
                     << " windows | PPL " << std::fixed << std::setprecision(4) << overall.ppl();
                if (kl_reference && stream_divergence.positions() > 0) {
                    line << " | stream KL " << std::scientific << std::setprecision(3)
                         << stream_divergence.mean_kl() << std::fixed;
                }
                line
                     << " | " << infernix::product::format_pretty_rate(rate, "tok") << " | elapsed "
                     << infernix::product::format_pretty_duration(elapsed) << " | ETA "
                     << infernix::product::format_pretty_duration(eta);
                if (progress->enabled()) {
                    progress->update("  " + line.str());
                } else {
                    logger->info("{}", line.str());
                }
                next_progress = Clock::now() + std::chrono::seconds(10);
            }
        }
        const double stream_seconds = seconds_since(stream_started);
        progress->clear();
        logger->info("[{}/{}] {} | {} scored tokens | PPL {:.6g} | {}", stream_index + 1,
                     streams.size(), infernix::product::format_pretty_text(stream.source.id),
                     infernix::product::format_pretty_count(stream_score.scored_tokens),
                     stream_score.ppl(), infernix::product::format_pretty_duration(stream_seconds));
        json stream_report               = aggregate_json(stream_score);
        if (kl_reference) {
            logger->info("[{}/{}] {} | KL {:.4e} | top-1 agreement {:.4f} | by context {}",
                         stream_index + 1, streams.size(),
                         infernix::product::format_pretty_text(stream.source.id),
                         stream_divergence.mean_kl(), stream_divergence.top1_agreement(),
                         stream_divergence.bucket_summary());
            stream_report["kl_divergence"] = stream_divergence.to_json();
            overall_divergence.add(stream_divergence);
        }
        stream_report["id"]              = stream.source.id;
        stream_report["domain"]          = stream.source.domain;
        stream_report["path"]            = stream.source.path.string();
        stream_report["input_tokens"]    = stream.tokens.size();
        stream_report["unscored_tokens"] = 1;
        stream_report["seconds"]         = stream_seconds;
        stream_report["windows"]         = std::move(window_reports);
        stream_reports.push_back(std::move(stream_report));
    }

    const double scoring_seconds = seconds_since(scoring_started);
    progress->clear();
    if (top_tokens) { top_tokens->finish(); }
    if (kl_reference) { kl_reference->finish(); }
    logger->info("scoring complete | {} tokens | {} windows | PPL {:.6g} | {} | {}",
                 infernix::product::format_pretty_count(overall.scored_tokens), completed_windows,
                 overall.ppl(), infernix::product::format_pretty_duration(scoring_seconds),
                 infernix::product::format_pretty_rate(
                     static_cast<double>(overall.scored_tokens) / scoring_seconds, "tok"));
    json domain_reports = json::array();
    for (const auto& [domain, aggregate] : domains) {
        json item      = aggregate_json(aggregate);
        item["domain"] = domain;
        domain_reports.push_back(std::move(item));
    }

    json report{
        {"schema_version", 3},
        {"metric",
         {{"name", "fixed-window truncated-context causal perplexity"}, {"log_base", "natural"}}},
        {"artifact",
         {{"path", std::filesystem::absolute(options.artifact).lexically_normal().string()},
          {"architecture", load.architecture},
          {"name", load.model_name},
          {"prefill_signature", load.prefill_signature},
          {"formats", load.weight_formats}}},
        {"corpus",
         {{"id", corpus.corpus_id},
          {"mode", corpus.mode},
          {"source", corpus.source.string()},
          {"stream_count", streams.size()}}},
        {"execution",
         {{"purpose", "causal_scoring"},
          {"device", options.device},
          {"context_tokens", options.context},
          {"rope_yarn_factor", options.rope_yarn_factor},
          {"original_int8_prefill_kernel", options.original_int8_prefill_kernel},
          {"prefill_8bit_pv", infernix::prefill_pv8_name(options.prefill_8bit_pv)},
          {"original_nvfp4_prefill_kernel", options.original_nvfp4_prefill_kernel},
          {"a16_activations", options.a16_activations},
          {"stride_tokens", options.stride},
          {"prefill_chunk_tokens", engine.options().prefill_chunk},
          {"score_tile_tokens", 1024},
          {"kv_dtype", kv_name(options.kv)}}},
        {"timing",
         {{"load_seconds", load.load_seconds},
          {"read_and_tokenize_seconds", preflight_seconds},
          {"score_seconds", scoring_seconds},
          {"total_seconds", seconds_since(total_started)},
          {"scored_tokens_per_second",
           static_cast<double>(overall.scored_tokens) / scoring_seconds}}},
        {"streams", std::move(stream_reports)},
        {"domains", std::move(domain_reports)},
        {"overall", aggregate_json(overall)},
    };
    if (top_tokens) {
        report["top_tokens"] = {{"path", std::filesystem::absolute(*options.save_top_tokens).string()},
                                {"per_position", infernix::perplexity::kReferenceTopTokens}};
    }
    if (kl_reference) {
        json divergence         = overall_divergence.to_json();
        divergence["reference"] = std::filesystem::absolute(*options.kl_reference).string();
        divergence["reference_top_tokens"] = kl_reference->header().top_k;
        divergence["definition"] =
            "KL(reference || this run) over the reference's top tokens plus one bucket for all "
            "others (a lower bound); context is the number of tokens before the predicted one";
        report["kl_divergence"] = std::move(divergence);
    }

    const std::filesystem::path temporary = output_directory / "report.json.tmp";
    const std::filesystem::path final     = output_directory / "report.json";
    {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output) { throw std::runtime_error("cannot create report: " + temporary.string()); }
        output << std::setw(2) << report << '\n';
        output.flush();
        if (!output) { throw std::runtime_error("cannot write report: " + temporary.string()); }
    }
    std::filesystem::rename(temporary, final);

    std::cout << "Perplexity result\n"
              << "artifact: " << load.model_name << '\n'
              << "kv: " << kv_name(options.kv) << ", corpus: " << corpus.corpus_id << " / "
              << corpus.mode << ", context/stride: " << options.context << '/' << options.stride
              << "\n\n";
    std::cout << std::left << std::setw(24) << "domain" << std::right << std::setw(16) << "tokens"
              << std::setw(16) << "mean_nll" << std::setw(16) << "ppl" << '\n';
    for (const auto& [domain, aggregate] : domains) {
        std::cout << std::left << std::setw(24) << domain << std::right << std::setw(16)
                  << aggregate.scored_tokens << std::setw(16) << std::fixed << std::setprecision(6)
                  << aggregate.mean_nll() << std::setw(16) << aggregate.ppl() << '\n';
    }
    std::cout << std::left << std::setw(24) << "overall" << std::right << std::setw(16)
              << overall.scored_tokens << std::setw(16) << std::fixed << std::setprecision(6)
              << overall.mean_nll() << std::setw(16) << overall.ppl() << "\n\n";
    if (kl_reference) {
        std::cout << "KL divergence from reference: mean " << std::scientific
                  << std::setprecision(4) << overall_divergence.mean_kl() << ", top-1 agreement "
                  << std::fixed << std::setprecision(4) << overall_divergence.top1_agreement()
                  << "\nby context: " << overall_divergence.bucket_summary() << "\n\n";
    }
    std::cout << "score rate: " << std::fixed << std::setprecision(1)
              << static_cast<double>(overall.scored_tokens) / scoring_seconds << " tok/s\n"
              << "report: " << final << '\n';
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    Options options;
    try {
        options = parse_options(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "infernix-perplexity: " << error.what() << '\n';
        std::cerr << usage_text();
        return 1;
    }
    if (options.help_requested) {
        std::cout << usage_text();
        return 0;
    }

    infernix::product::LoggingRuntime logging(
        {.logger_name  = "infernix-perplexity",
         .level        = options.log_level,
         .presentation = infernix::product::LogPresentation::Tool});
    const std::shared_ptr<spdlog::logger> logger = logging.logger();
#ifdef INFERNIX_BUILD_ID
    logger->info("build {}", INFERNIX_BUILD_ID);
#endif
    infernix::product::StartupLogRenderer startup_log(logging);
    try {
        return run(options, logger, startup_log, logging.terminal_progress());
    } catch (const std::exception& error) {
        logging.terminal_progress()->clear();
        logger->error("{}", infernix::product::format_pretty_text(error.what()));
        return 1;
    }
}
