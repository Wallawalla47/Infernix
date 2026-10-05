// End-to-end decode quality, generator half. Greedy continuations of fixed segments of a UTF-8
// corpus, written as token ids, through the production decode path: DFlash2 with 7 drafts and the
// optimized proposal head, INT8 KV, CUDA Graph decode. ninfer_decode_quality_judge scores the
// output of any number of builds against one 16-bit-activation reference.
//
// This file uses only Engine API that upstream NInfer also has, so the same source builds in an
// upstream checkout (define NINFER_DECODE_QUALITY_UPSTREAM there: it drops n-gram drafting, which
// upstream lacks). That is how bench/README.md compares this fork's decode with upstream's.
//
// Usage: ninfer_decode_quality_gen <artifact> <corpus.txt> <out.txt> <segments> <prompt_tokens>
//                                  <new_tokens> [ngram]
// Output, one line per segment: <corpus token offset> <prompt_tokens> <generated ids...>

#include "ninfer/engine.h"

#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 7) {
        std::cerr << "usage: ninfer_decode_quality_gen <artifact> <corpus.txt> <out.txt> <segments> "
                     "<prompt_tokens> <new_tokens> [ngram]\n";
        return 2;
    }
    try {
        const auto segments      = static_cast<std::uint32_t>(std::stoul(argv[4]));
        const auto prompt_tokens = static_cast<std::uint32_t>(std::stoul(argv[5]));
        const auto new_tokens    = static_cast<std::uint32_t>(std::stoul(argv[6]));
        const bool ngram         = argc > 7 && std::string(argv[7]) == "ngram";
        if (segments == 0 || prompt_tokens == 0 || new_tokens == 0) {
            std::cerr << "segments, prompt_tokens and new_tokens must be positive\n";
            return 2;
        }

        ninfer::EngineOptions options;
        options.artifact_path = argv[1];
        options.max_context   = (prompt_tokens + new_tokens + 64 + 127) / 128 * 128;
        options.kv_cache      = ninfer::KvCacheStorage::Int8Group64;
        options.max_concurrency                   = 1;
        options.prefill_chunk                     = 4096;
        options.speculative.backend               = ninfer::SpeculativeBackend::DFlash2;
        options.speculative.draft_tokens          = 7;
        options.speculative.proposal_head         = ninfer::ProposalHead::Optimized;
        options.context_cache.enabled             = false;
        options.context_cache.device_state_slots  = 0;
        options.context_cache.host_capacity_bytes = 0;
        if (ngram) {
#if defined(NINFER_DECODE_QUALITY_UPSTREAM)
            std::cerr << "n-gram drafting is not available upstream\n";
            return 2;
#else
            // ninfer-serve's production n-gram settings.
            options.speculative.ngram_draft_tokens = 15;
            options.speculative.ngram_min_match    = 12;
#endif
        }
        ninfer::Engine engine(options);

        std::ifstream corpus(argv[2], std::ios::binary);
        if (!corpus) {
            std::cerr << "cannot read " << argv[2] << '\n';
            return 2;
        }
        std::stringstream text;
        text << corpus.rdbuf();
        const std::vector<ninfer::TokenId> tokens = engine.tokenize_text(text.str());
        if (tokens.size() < static_cast<std::size_t>(prompt_tokens) * 2) {
            std::cerr << "the corpus has fewer than twice prompt_tokens tokens\n";
            return 2;
        }

        std::ofstream out(argv[3]);
        const auto started      = std::chrono::steady_clock::now();
        std::uint64_t generated = 0;
        for (std::uint32_t segment = 0; segment < segments; ++segment) {
            const std::size_t offset =
                static_cast<std::size_t>(segment) * (tokens.size() - prompt_tokens) / segments;
            std::vector<ninfer::TokenId> prompt(
                tokens.begin() + static_cast<std::ptrdiff_t>(offset),
                tokens.begin() + static_cast<std::ptrdiff_t>(offset + prompt_tokens));
            ninfer::RequestOptions request;
            request.execution.requested_output_tokens = new_tokens;
            request.execution.sampling.temperature    = 0.0F;
            request.execution.allow_prefix_reuse      = false;
            request.stop.include_model_defaults       = false;
            const ninfer::GenerationResult result = engine.generate(
                engine.prepare_tokens(std::move(prompt), false), std::move(request));
            out << offset << ' ' << prompt_tokens;
            for (const ninfer::TokenId id : result.generated_token_ids) { out << ' ' << id; }
            out << '\n';
            generated += result.generated_token_ids.size();
            std::cerr << "segment " << segment + 1 << "/" << segments << ": "
                      << result.generated_token_ids.size() << " tokens\n";
        }
        const double seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        std::cerr << "generated " << generated << " tokens in " << seconds << " s\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ninfer_decode_quality_gen: " << error.what() << '\n';
        return 1;
    }
}
