// End-to-end decode quality, generator half. Greedy continuations of fixed segments of a UTF-8
// corpus, written as token ids, through the production decode path: DFlash2 with 7 drafts and the
// optimized proposal head, INT8 KV, CUDA Graph decode (`dflash`: the artifact's DFlash drafter
// instead, the Qwen3.6-35B-A3B's; `auto` with either: --draft-tokens auto, `drafts=N` its ceiling,
// else the Engine's); with `mtp`, Qwen3.8-Flash-Next's: its MTP
// drafter with 4 drafts and the optimized proposal head; with `plain`, no speculation (one token per
// round, the single-column decode kernels). Fast paths run as in infernix-serve; `reproducible`
// runs the reproducible route (EngineOptions::fast_paths off) and `path=NAME` also runs that fast
// path whether or not it is qualified (qualification). Qwen3.8-Flash-Next's DFlash2 drafter is a companion artifact:
// `draft-model=FILE` selects it (`drafts=N`, 1-7, its draft count, also MTP's; `auto` its per-round length;
// `full-head` drafts with the text head instead of the proposal head). `concurrency=N` (1-8) keeps N
// segments decoding on an Engine of N lanes (N workers, each taking the next segment as soon as its
// last completes) and reports their wall-clock throughput. `ngram=N` and
// `min-match=M` set the n-gram copy window and minimum match (`ngram`: 15 and 12; `ngram=auto` the
// Engine's), `tree=auto` or
// `tree=N` DFlash2 tree verification. `prompts` reads the corpus as whole prompts, each ended by a line
// "<<<END>>>" (prompt_tokens their limit), for workloads such as copy-heavy edits. `temperature=T` samples
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
//                                  <new_tokens> [ngram | ngram=N] [min-match=M] [tree=auto|N] [prompts]
//                                  [concurrency=N] [mtp [auto] [min-p=P]] [dflash [auto]] [plain] [reproducible]
//                                  [path=NAME]...
//                                  [draft-model=FILE [drafts=N] [auto] [full-head]] (Qwen3.8-Flash-Next DFlash2)
//                                  [temperature=T]
//                                  [ngram-volume=FILE] (Qwen3.8-Flash-Next; default <artifact>.ngram)
//                                  [route-trace=FILE] (Qwen3.8-Flash-Next's route trace, route_trace.h)
// Output, one line per segment: <corpus token offset> <prompt_tokens> <generated ids...>

#include "infernix/engine.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iostream>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 7) {
        std::cerr << "usage: infernix_decode_quality_gen <artifact> <corpus.txt> <out.txt> <segments> "
                     "<prompt_tokens> <new_tokens> [ngram] [mtp [auto] [min-p=P]] [plain] [reproducible] [path=NAME]... "
                     "[kv=bf16|int8|fp8|nvfp4|k8v4|vq2|k4v2]\n";
        return 2;
    }
    try {
        const auto segments      = static_cast<std::uint32_t>(std::stoul(argv[4]));
        const auto prompt_tokens = static_cast<std::uint32_t>(std::stoul(argv[5]));
        const auto new_tokens    = static_cast<std::uint32_t>(std::stoul(argv[6]));
        bool ngram = false, mtp = false, dflash = false, plain = false, reproducible = false, draft_auto = false,
             full_head = false, tree_auto = false, prompt_records = false, ngram_auto = false;
        std::uint32_t ngram_drafts = 15, ngram_min_match = 12, tree_nodes = 0;
        std::vector<std::string> paths;
        std::string ngram_volume, route_trace, draft_model;
        std::uint32_t drafts = 7;
        bool drafts_set      = false;
        std::uint32_t concurrency = 1;
        float temperature    = 0.0F;
        std::optional<float> min_p;
        infernix::KvCacheStorage kv = infernix::KvCacheStorage::Int8Group64;
        for (int i = 7; i < argc; ++i) {
            const std::string flag = argv[i];
            if (flag == "ngram") {
                ngram = true;
            } else if (flag == "ngram=auto") {
                ngram      = true;
                ngram_auto = true;
            } else if (flag.starts_with("ngram=")) {
                ngram        = true;
                ngram_drafts = static_cast<std::uint32_t>(std::stoul(flag.substr(6)));
            } else if (flag.starts_with("min-match=")) {
                ngram_min_match = static_cast<std::uint32_t>(std::stoul(flag.substr(10)));
            } else if (flag == "tree=auto") {
                tree_auto = true;
            } else if (flag.starts_with("tree=")) {
                tree_nodes = static_cast<std::uint32_t>(std::stoul(flag.substr(5)));
            } else if (flag == "prompts") {
                prompt_records = true;
            } else if (flag == "mtp") {
                mtp = true;
            } else if (flag == "dflash") {
                dflash = true;
            } else if (flag == "auto") {
                draft_auto = true;
            } else if (flag.starts_with("min-p=")) {
                min_p = std::stof(flag.substr(6));
            } else if (flag == "plain") {
                plain = true;
            } else if (flag == "reproducible") {
                reproducible = true;
            } else if (flag.starts_with("ngram-volume=")) {
                ngram_volume = flag.substr(13);
            } else if (flag.starts_with("route-trace=")) {
                route_trace = flag.substr(12);
            } else if (flag.starts_with("draft-model=")) {
                draft_model = flag.substr(12);
            } else if (flag.starts_with("concurrency=")) {
                concurrency = static_cast<std::uint32_t>(std::stoul(flag.substr(12)));
                if (concurrency == 0 || concurrency > 8) {
                    std::cerr << "concurrency must be 1..8\n";
                    return 2;
                }
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
                          << " (ngram, ngram=N, min-match=M, tree=auto|N, prompts, mtp, dflash, auto, min-p=P, "
                             "plain, reproducible, path=NAME, draft-model=FILE, drafts=N, full-head, temperature=T, "
                             "ngram-volume=FILE, route-trace=FILE, kv=NAME)\n";
                return 2;
            }
        }
        if (segments == 0 || prompt_tokens == 0 || new_tokens == 0) {
            std::cerr << "segments, prompt_tokens and new_tokens must be positive\n";
            return 2;
        }
        if (plain && (mtp || dflash || ngram)) {
            std::cerr << "plain decoding drafts nothing (no mtp, dflash or ngram)\n";
            return 2;
        }
        if (mtp && dflash) {
            std::cerr << "mtp and dflash are two drafters: choose one\n";
            return 2;
        }

        infernix::EngineOptions options;
        options.artifact_path = argv[1];
        options.max_context   = (prompt_tokens + new_tokens + 64 + 127) / 128 * 128;
        // Every lane's whole context: the Qwen3.5 planner refuses a capacity past it.
        options.kv_capacity   = infernix::KvCapacityPolicy::explicit_capacity(options.max_context * concurrency);
        options.kv_cache      = kv;
        options.max_concurrency                   = concurrency;
        options.prefill_chunk                     = 4096;
        options.speculative.backend      = dflash ? infernix::SpeculativeBackend::DFlash
                                                  : infernix::SpeculativeBackend::DFlash2;
        // `auto` without `drafts=N` leaves the ceiling to the Engine (the drafter's whole block).
        options.speculative.draft_tokens = draft_auto && !drafts_set && !mtp ? 0U : drafts;
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
            // infernix-serve's production n-gram settings, or `ngram=N` / `min-match=M`, or `ngram=auto`
            // (--ngram-draft-tokens auto: the Engine's window, batched window and minimum match).
            options.speculative.ngram_draft_tokens      = ngram_auto ? 0U : ngram_drafts;
            options.speculative.ngram_draft_tokens_auto = ngram_auto;
            options.speculative.ngram_min_match         = ngram_min_match;
#endif
        }
        if (tree_auto || tree_nodes != 0) {
#if defined(INFERNIX_DECODE_QUALITY_NINFER)
            std::cerr << "tree options are not used in NInfer comparisons\n";
            return 2;
#else
            // DFlash2 tree verification: `tree=auto` (--draft-tree-nodes auto) or a fixed width for
            // every batch size.
            options.speculative.draft_tree_auto = tree_auto;
            if (tree_nodes != 0) { options.speculative.draft_tree_nodes.fill(tree_nodes); }
#endif
        }
        if (plain) { options.speculative = {}; }
#if defined(INFERNIX_DECODE_QUALITY_NINFER)
        if (reproducible || !paths.empty() || !route_trace.empty()) {
            std::cerr << "fast-path and route-trace options are not available in NInfer\n";
            return 2;
        }
#else
        // Qwen3.8-Flash-Next: the per-round expert routes for tools/expert_cache_replay/host_tier.py.
        options.route_trace_path = route_trace;
        options.fast_paths = !reproducible;
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
#if !defined(INFERNIX_DECODE_QUALITY_NINFER)
        if (ngram_auto) {
            const infernix::SpeculativeOptions& resolved = engine.options().speculative;
            std::cerr << "ngram auto: window " << resolved.ngram_draft_tokens << ", batched "
                      << resolved.ngram_batch_draft_tokens << ", minimum match " << resolved.ngram_min_match << '\n';
        }
#endif

        std::ifstream corpus(argv[2], std::ios::binary);
        if (!corpus) {
            std::cerr << "cannot read " << argv[2] << '\n';
            return 2;
        }
        std::stringstream text;
        text << corpus.rdbuf();
        // `prompts`: the corpus is a list of whole prompts, each ended by a line "<<<END>>>", and
        // segment s continues prompt s modulo their count (at most prompt_tokens tokens each).
        std::vector<std::vector<infernix::TokenId>> records;
        if (prompt_records) {
            const std::string all = text.str();
            const std::string end = "\n<<<END>>>\n";
            for (std::size_t begin = 0, at = all.find(end); at != std::string::npos;
                 begin = at + end.size(), at = all.find(end, begin)) {
                records.push_back(engine.tokenize_text(std::string_view(all).substr(begin, at - begin)));
                if (records.back().empty() || records.back().size() > prompt_tokens) {
                    std::cerr << "prompt " << records.size() - 1 << " has " << records.back().size()
                              << " tokens, outside 1.." << prompt_tokens << '\n';
                    return 2;
                }
            }
            if (records.empty()) {
                std::cerr << "the prompt file has no prompt ended by <<<END>>>\n";
                return 2;
            }
        }
        const std::vector<infernix::TokenId> tokens =
            prompt_records ? std::vector<infernix::TokenId>{} : engine.tokenize_text(text.str());
        if (!prompt_records && tokens.size() < static_cast<std::size_t>(prompt_tokens) * 2) {
            std::cerr << "the corpus has fewer than twice prompt_tokens tokens\n";
            return 2;
        }

        // Segment s's prompt and request; a prompt record's offset is its index.
        const auto segment_offset = [&](std::uint32_t segment) {
            return prompt_records ? segment % records.size()
                                  : static_cast<std::size_t>(segment) * (tokens.size() - prompt_tokens) / segments;
        };
        const auto segment_request = [&](std::uint32_t segment) {
            infernix::RequestOptions request;
            request.execution.requested_output_tokens = new_tokens;
            request.execution.sampling.temperature    = temperature;
            if (temperature > 0.0F) {
                request.execution.sampling.top_k = 20;
                request.execution.sampling.top_p = 0.95F;
                request.execution.sampling.seed  = segment + 1ULL;
            }
            request.execution.allow_prefix_reuse = false;
            request.stop.include_model_defaults  = false;
            return request;
        };

        std::ofstream out(argv[3]);
        const auto started      = std::chrono::steady_clock::now();
        std::uint64_t generated = 0;
        infernix::SpeculativeStats speculative;
        double prefill_seconds = 0.0, decode_seconds = 0.0;
        // `concurrency` workers each take the next segment as soon as their last one completes, so
        // that many requests decode until the corpus runs out; results are written in segment order.
        std::vector<infernix::GenerationResult> results(segments);
        std::atomic<std::uint32_t> next{0};
        std::mutex progress;
        std::uint32_t completed = 0;
        std::exception_ptr failure;
        const auto worker = [&] {
            try {
                for (std::uint32_t segment = next++; segment < segments; segment = next++) {
                    const std::size_t offset = segment_offset(segment);
                    std::vector<infernix::TokenId> prompt =
                        prompt_records ? records[offset]
                                       : std::vector<infernix::TokenId>(
                                             tokens.begin() + static_cast<std::ptrdiff_t>(offset),
                                             tokens.begin() + static_cast<std::ptrdiff_t>(offset + prompt_tokens));
#if defined(INFERNIX_DECODE_QUALITY_NINFER)
                    infernix::GenerationResult result =
                        engine.generate(engine.prepare_tokens(std::move(prompt), false), segment_request(segment));
#else
                    // Phase timings only observe (they change no execution): prefill and decode apart.
                    infernix::GenerationResult result =
                        engine
                            .submit(engine.prepare_tokens(std::move(prompt), false), segment_request(segment),
                                    infernix::OutputConsumerMode::Aggregate, {.phase_timings = true})
                            .wait();
#endif
                    const std::lock_guard<std::mutex> lock(progress);
                    std::cerr << "segment " << ++completed << "/" << segments << ": "
                              << result.generated_token_ids.size() << " tokens\n";
                    results[segment] = std::move(result);
                }
            } catch (...) {
                const std::lock_guard<std::mutex> lock(progress);
                if (!failure) { failure = std::current_exception(); }
                next = segments;
            }
        };
        {
            std::vector<std::thread> workers;
            for (std::uint32_t w = 0; w < concurrency; ++w) { workers.emplace_back(worker); }
            for (std::thread& thread : workers) { thread.join(); }
        }
        if (failure) { std::rethrow_exception(failure); }
        const double seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        for (std::uint32_t segment = 0; segment < segments; ++segment) {
            const infernix::GenerationResult& result = results[segment];
            const std::size_t offset                 = segment_offset(segment);
            out << offset << ' ' << (prompt_records ? records[offset].size() : prompt_tokens);
            for (const infernix::TokenId id : result.generated_token_ids) { out << ' ' << id; }
            out << '\n';
            generated += result.generated_token_ids.size();
#if !defined(INFERNIX_DECODE_QUALITY_NINFER)
            prefill_seconds += result.timings.prefill_seconds;
            decode_seconds += result.timings.decode_seconds;
#endif
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
#if !defined(INFERNIX_DECODE_QUALITY_NINFER)
            const auto& by_length = result.speculative.rounds_by_draft_length;
            if (speculative.rounds_by_draft_length.size() < by_length.size()) {
                speculative.rounds_by_draft_length.resize(by_length.size(), 0);
            }
            for (std::size_t j = 0; j < by_length.size(); ++j) {
                speculative.rounds_by_draft_length[j] += by_length[j];
            }
            speculative.ngram_rounds += result.speculative.ngram_rounds;
            speculative.ngram_drafted_tokens += result.speculative.ngram_drafted_tokens;
            speculative.ngram_accepted_tokens += result.speculative.ngram_accepted_tokens;
            speculative.tree_rounds += result.speculative.tree_rounds;
            speculative.tree_side_rounds += result.speculative.tree_side_rounds;
            speculative.tree_side_accepted_tokens += result.speculative.tree_side_accepted_tokens;
#endif
        }
        std::cerr << "generated " << generated << " tokens in " << seconds << " s\n";
        std::cerr << "speculative: " << speculative.rounds << " rounds, " << speculative.fallback_steps
                  << " plain steps, accepted " << speculative.accepted_tokens << " of " << speculative.drafted_tokens
                  << " drafts; rounds accepting position 1..:";
        for (const std::uint64_t n : speculative.accepted_per_position) { std::cerr << ' ' << n; }
        std::cerr << '\n';
#if !defined(INFERNIX_DECODE_QUALITY_NINFER)
        // Qwen3.5-family neural rounds by their verified draft count (adaptive lengths and widths).
        if (!speculative.rounds_by_draft_length.empty()) {
            std::cerr << "rounds verifying 1..N drafts:";
            for (const std::uint64_t n : speculative.rounds_by_draft_length) { std::cerr << ' ' << n; }
            std::cerr << '\n';
        }
        std::cerr << "ngram: " << speculative.ngram_rounds << " rounds, accepted " << speculative.ngram_accepted_tokens
                  << " of " << speculative.ngram_drafted_tokens << " drafts; tree: " << speculative.tree_rounds
                  << " rounds, " << speculative.tree_side_rounds << " side, " << speculative.tree_side_accepted_tokens
                  << " side drafts\n";
        {
            // Batch occupancy: decode batch executions and the rows they carried.
            const infernix::RuntimeStats stats = engine.runtime_stats();
            std::cerr << "decode batches: " << stats.decode_rounds << " rounds, " << stats.decode_row_rounds
                      << " rows\n";
        }
        std::cerr << "phases: prefill " << prefill_seconds << " s, decode " << decode_seconds << " s\n";
#endif
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "infernix_decode_quality_gen: " << error.what() << '\n';
        return 1;
    }
}
