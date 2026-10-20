// End-to-end decode quality, generator half. Greedy continuations of fixed segments of a UTF-8
// corpus, written as token ids, through the production decode path: DFlash2 with 7 drafts and the
// optimized proposal head, INT8 KV, CUDA Graph decode; with `mtp`, Qwen3.8-Flash-Next's: its MTP
// drafter with 4 drafts and the optimized proposal head; with `plain`, no speculation (one token per
// round, the single-column decode kernels). Fast paths run as in infernix-serve; `exact` runs the
// exact route (EngineOptions::fast_paths off) and `path=NAME` also runs that fast path whether or
// not it is qualified (qualification). Qwen3.8-Flash-Next's DFlash2 drafter is a companion artifact:
// `draft-model=FILE` selects it (`drafts=N`, 1-7, its draft count, also MTP's; `auto` its per-round length;
// `full-head` drafts with the text head instead of the proposal head). `temperature=T` samples
// instead (top-k 20, top-p 0.95, segment s seeded s + 1: a speed workload; the judge scores greedy
// output only).
// infernix_decode_quality_judge scores the output of any number of builds against one reference.
//
// This file uses only Engine API that NInfer also has, so the same source builds in an
// NInfer checkout (define INFERNIX_DECODE_QUALITY_NINFER there: it drops n-gram and MTP drafting,
// and the fast-path options, which NInfer lacks). That is how bench/README.md compares Infernix's
// decode with NInfer's.
//
// Usage: infernix_decode_quality_gen <artifact> <corpus.txt> <out.txt> <segments> <prompt_tokens>
//                                  <new_tokens> [ngram] [mtp [auto] [min-p=P]] [plain] [exact] [path=NAME]...
//                                  [draft-model=FILE [drafts=N] [auto] [full-head]] (Qwen3.8-Flash-Next DFlash2)
//                                  [temperature=T]
//                                  [ngram-volume=FILE] (Qwen3.8-Flash-Next; default <artifact>.ngram)
//                                  [route-trace=FILE] (Qwen3.8-Flash-Next's route trace, route_trace.h)
// Output, one line per segment: <corpus token offset> <prompt_tokens> <generated ids...>

#include "infernix/engine.h"

#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 7) {
        std::cerr << "usage: infernix_decode_quality_gen <artifact> <corpus.txt> <out.txt> <segments> "
                     "<prompt_tokens> <new_tokens> [ngram] [mtp [auto] [min-p=P]] [plain] [exact] [path=NAME]... "
                     "[kv=bf16|int8|fp8|nvfp4|k8v4|vq2|k4v2]\n";
        return 2;
    }
    try {
        const auto segments      = static_cast<std::uint32_t>(std::stoul(argv[4]));
        const auto prompt_tokens = static_cast<std::uint32_t>(std::stoul(argv[5]));
        const auto new_tokens    = static_cast<std::uint32_t>(std::stoul(argv[6]));
        bool ngram = false, mtp = false, plain = false, exact = false, draft_auto = false, full_head = false;
        std::vector<std::string> paths;
        std::string ngram_volume, route_trace, draft_model;
        std::uint32_t drafts = 7;
        bool drafts_set      = false;
        float temperature    = 0.0F;
        std::optional<float> min_p;
        infernix::KvCacheStorage kv = infernix::KvCacheStorage::Int8Group64;
        for (int i = 7; i < argc; ++i) {
            const std::string flag = argv[i];
            if (flag == "ngram") {
                ngram = true;
            } else if (flag == "mtp") {
                mtp = true;
            } else if (flag == "auto") {
                draft_auto = true;
            } else if (flag.starts_with("min-p=")) {
                min_p = std::stof(flag.substr(6));
            } else if (flag == "plain") {
                plain = true;
            } else if (flag == "exact") {
                exact = true;
            } else if (flag.starts_with("ngram-volume=")) {
                ngram_volume = flag.substr(13);
            } else if (flag.starts_with("route-trace=")) {
                route_trace = flag.substr(12);
            } else if (flag.starts_with("draft-model=")) {
                draft_model = flag.substr(12);
            } else if (flag.starts_with("drafts=")) {
                drafts     = static_cast<std::uint32_t>(std::stoul(flag.substr(7)));
                drafts_set = true;
            } else if (flag == "full-head") {
                full_head = true;
            } else if (flag.starts_with("temperature=")) {
                temperature = std::stof(flag.substr(12));
            } else if (flag.starts_with("path=")) {
                paths.push_back(flag.substr(5));
            } else if (flag.starts_with("kv=")) {
                const std::string name = flag.substr(3);
                if (name == "bf16") {
                    kv = infernix::KvCacheStorage::BFloat16;
                } else if (name == "int8") {
                    kv = infernix::KvCacheStorage::Int8Group64;
                } else if (name == "fp8") {
                    kv = infernix::KvCacheStorage::Fp8E4M3Row256;
                } else if (name == "nvfp4") {
                    kv = infernix::KvCacheStorage::Nvfp4Group16;
                } else if (name == "k8v4") {
                    kv = infernix::KvCacheStorage::Fp8KeyNvfp4Value;
                } else if (name == "vq2") {
                    kv = infernix::KvCacheStorage::Vq2;
                } else if (name == "k4v2") {
                    kv = infernix::KvCacheStorage::Q4KeyVq2Value;
                } else {
                    std::cerr << "unknown KV format " << name << '\n';
                    return 2;
                }
            } else {
                std::cerr << "unknown option " << flag
                          << " (ngram, mtp, auto, min-p=P, plain, exact, path=NAME, draft-model=FILE, drafts=N, "
                             "full-head, temperature=T, ngram-volume=FILE, route-trace=FILE, kv=NAME)\n";
                return 2;
            }
        }
        if (segments == 0 || prompt_tokens == 0 || new_tokens == 0) {
            std::cerr << "segments, prompt_tokens and new_tokens must be positive\n";
            return 2;
        }
        if (plain && (mtp || ngram)) {
            std::cerr << "plain decoding drafts nothing (no mtp or ngram)\n";
            return 2;
        }

        infernix::EngineOptions options;
        options.artifact_path = argv[1];
        options.max_context   = (prompt_tokens + new_tokens + 64 + 127) / 128 * 128;
        options.kv_cache      = kv;
        options.max_concurrency                   = 1;
        options.prefill_chunk                     = 4096;
        options.speculative.backend               = infernix::SpeculativeBackend::DFlash2;
        options.speculative.draft_tokens          = drafts;
        options.speculative.proposal_head         = full_head ? infernix::ProposalHead::Full : infernix::ProposalHead::Optimized;
#if !defined(INFERNIX_DECODE_QUALITY_NINFER)
        // Qwen3.8-Flash-Next's DFlash2 companion artifact (--draft-model) and --draft-tokens auto.
        options.speculative.draft_model       = draft_model;
        options.speculative.draft_tokens_auto = !mtp && draft_auto;
        // `min-p=P`: --spec-min-p P, the selector-probability cut of a one-row round.
        options.speculative.draft_min_p = min_p;
#endif
        if (mtp) {
#if defined(INFERNIX_DECODE_QUALITY_NINFER)
            std::cerr << "MTP drafting is not available in NInfer\n";
            return 2;
#else
            // infernix-serve's production Qwen3.8-Flash-Next drafting (--spec mtp --draft-tokens 4);
            // `drafts=N` another count (with `auto`, its ceiling).
            options.speculative.backend      = infernix::SpeculativeBackend::Mtp;
            options.speculative.draft_tokens = drafts_set ? drafts : 4;
            // `auto`: --draft-tokens auto, each round's own length up to 4 and the --spec-min-p 0.5 cut.
            options.speculative.draft_tokens_auto = draft_auto;
            // `min-p=P`: --spec-min-p P, the draft-confidence cut (a fixed length's default is none).
            options.speculative.draft_min_p = min_p;
#endif
        }
        options.context_cache.enabled             = false;
        options.context_cache.host_capacity_bytes = 0;
        if (ngram) {
#if defined(INFERNIX_DECODE_QUALITY_NINFER)
            std::cerr << "n-gram drafting is not available in NInfer\n";
            return 2;
#else
            // infernix-serve's production n-gram settings.
            options.speculative.ngram_draft_tokens = 15;
            options.speculative.ngram_min_match    = 12;
#endif
        }
        if (plain) { options.speculative = {}; }
#if defined(INFERNIX_DECODE_QUALITY_NINFER)
        if (exact || !paths.empty() || !route_trace.empty()) {
            std::cerr << "fast-path and route-trace options are not available in NInfer\n";
            return 2;
        }
#else
        // Qwen3.8-Flash-Next: the per-round expert routes for tools/expert_cache_replay/host_tier.py.
        options.route_trace_path = route_trace;
        options.fast_paths = !exact;
        for (const std::string& name : paths) {
            const std::optional<infernix::FastPath> path = infernix::parse_fast_path(name);
            if (!path) {
                std::cerr << "unknown fast path " << name << '\n';
                return 2;
            }
            options.fast_path_candidates.push_back(*path);
        }
#endif
        if (!ngram_volume.empty()) { options.ngram_volume_path = ngram_volume; }
        infernix::Engine engine(options);

        std::ifstream corpus(argv[2], std::ios::binary);
        if (!corpus) {
            std::cerr << "cannot read " << argv[2] << '\n';
            return 2;
        }
        std::stringstream text;
        text << corpus.rdbuf();
        const std::vector<infernix::TokenId> tokens = engine.tokenize_text(text.str());
        if (tokens.size() < static_cast<std::size_t>(prompt_tokens) * 2) {
            std::cerr << "the corpus has fewer than twice prompt_tokens tokens\n";
            return 2;
        }

        std::ofstream out(argv[3]);
        const auto started      = std::chrono::steady_clock::now();
        std::uint64_t generated = 0;
        infernix::SpeculativeStats speculative;
        double prefill_seconds = 0.0, decode_seconds = 0.0;
        for (std::uint32_t segment = 0; segment < segments; ++segment) {
            const std::size_t offset =
                static_cast<std::size_t>(segment) * (tokens.size() - prompt_tokens) / segments;
            std::vector<infernix::TokenId> prompt(
                tokens.begin() + static_cast<std::ptrdiff_t>(offset),
                tokens.begin() + static_cast<std::ptrdiff_t>(offset + prompt_tokens));
            infernix::RequestOptions request;
            request.execution.requested_output_tokens = new_tokens;
            request.execution.sampling.temperature    = temperature;
            if (temperature > 0.0F) {
                request.execution.sampling.top_k = 20;
                request.execution.sampling.top_p = 0.95F;
                request.execution.sampling.seed  = segment + 1ULL;
            }
            request.execution.allow_prefix_reuse      = false;
            request.stop.include_model_defaults       = false;
#if defined(INFERNIX_DECODE_QUALITY_NINFER)
            const infernix::GenerationResult result = engine.generate(
                engine.prepare_tokens(std::move(prompt), false), std::move(request));
#else
            // Phase timings only observe (they change no execution): prefill and decode apart.
            const infernix::GenerationResult result =
                engine.submit(engine.prepare_tokens(std::move(prompt), false), std::move(request),
                              infernix::OutputConsumerMode::Aggregate, {.phase_timings = true})
                    .wait();
            prefill_seconds += result.timings.prefill_seconds;
            decode_seconds += result.timings.decode_seconds;
#endif
            out << offset << ' ' << prompt_tokens;
            for (const infernix::TokenId id : result.generated_token_ids) { out << ' ' << id; }
            out << '\n';
            generated += result.generated_token_ids.size();
            speculative.rounds += result.speculative.rounds;
            speculative.drafted_tokens += result.speculative.drafted_tokens;
            speculative.accepted_tokens += result.speculative.accepted_tokens;
            speculative.fallback_steps += result.speculative.fallback_steps;
            const auto& per_position = result.speculative.accepted_per_position;
            if (speculative.accepted_per_position.size() < per_position.size()) {
                speculative.accepted_per_position.resize(per_position.size(), 0);
            }
            for (std::size_t j = 0; j < per_position.size(); ++j) {
                speculative.accepted_per_position[j] += per_position[j];
            }
            std::cerr << "segment " << segment + 1 << "/" << segments << ": "
                      << result.generated_token_ids.size() << " tokens\n";
        }
        const double seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        std::cerr << "generated " << generated << " tokens in " << seconds << " s\n";
        std::cerr << "speculative: " << speculative.rounds << " rounds, " << speculative.fallback_steps
                  << " plain steps, accepted " << speculative.accepted_tokens << " of " << speculative.drafted_tokens
                  << " drafts; rounds accepting position 1..:";
        for (const std::uint64_t n : speculative.accepted_per_position) { std::cerr << ' ' << n; }
        std::cerr << '\n';
#if !defined(INFERNIX_DECODE_QUALITY_NINFER)
        std::cerr << "phases: prefill " << prefill_seconds << " s, decode " << decode_seconds << " s\n";
#endif
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "infernix_decode_quality_gen: " << error.what() << '\n';
        return 1;
    }
}
