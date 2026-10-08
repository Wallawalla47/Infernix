// End-to-end decode quality, judge half. Scores the greedy continuations that
// infernix_decode_quality_gen wrote, for any number of builds, with one reference: a CausalScoring
// Engine with 16-bit activations in every linear (EngineOptions::a16_activations), BF16 KV, and
// each sequence in one prefill pass. Each build is judged on its own prefixes, so builds whose
// continuations diverge are still compared fairly. Per build it reports how often the generated
// token is the reference's most probable token, the mean reference log-probability of the
// generated tokens, and the mean regret (top log-probability minus the generated token's); per
// pair of builds, the generated prefix they share exactly.
//
// --stored-activations scores with the activations the artifact records instead (every other
// condition the same). It is the reference for models without a 16-bit-activation route
// (Qwen3.8-Flash-Next): it isolates what decoding changes (KV format, chunking, speculation,
// batching) but not the cost of the stored activation precision itself.
//
// Usage: infernix_decode_quality_judge [--stored-activations] <artifact> <corpus.txt> <gen_file>...

#include "infernix/engine.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Segment {
    std::size_t offset          = 0;
    std::uint32_t prompt_tokens = 0;
    std::vector<infernix::TokenId> generated;
};

std::vector<Segment> read_generation(const std::string& path) {
    std::ifstream in(path);
    if (!in) { throw std::runtime_error("cannot read " + path); }
    std::vector<Segment> segments;
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream fields(line);
        Segment segment;
        if (!(fields >> segment.offset >> segment.prompt_tokens)) { continue; }
        infernix::TokenId id = 0;
        while (fields >> id) { segment.generated.push_back(id); }
        segments.push_back(std::move(segment));
    }
    return segments;
}

} // namespace

int main(int argc, char** argv) {
    const bool stored = argc > 1 && std::string(argv[1]) == "--stored-activations";
    if (stored) {
        ++argv;
        --argc;
    }
    if (argc < 4) {
        std::cerr << "usage: infernix_decode_quality_judge [--stored-activations] <artifact> <corpus.txt> <gen_file>...\n";
        return 2;
    }
    try {
        std::vector<std::vector<Segment>> runs;
        std::uint32_t longest = 0;
        for (int i = 3; i < argc; ++i) {
            runs.push_back(read_generation(argv[i]));
            for (const Segment& segment : runs.back()) {
                longest = std::max(longest,
                                   segment.prompt_tokens +
                                       static_cast<std::uint32_t>(segment.generated.size()));
            }
        }
        const std::uint32_t context = (longest + 127) / 128 * 128;

        infernix::EngineOptions options;
        options.artifact_path   = argv[1];
        options.purpose         = infernix::EnginePurpose::CausalScoring;
        options.max_context     = context;
        options.prefill_chunk   = context;
        options.kv_cache        = infernix::KvCacheStorage::BFloat16;
        options.a16_activations = !stored;
        infernix::Engine engine(options);

        std::ifstream corpus(argv[2], std::ios::binary);
        if (!corpus) { throw std::runtime_error(std::string("cannot read ") + argv[2]); }
        std::stringstream text;
        text << corpus.rdbuf();
        const std::vector<infernix::TokenId> tokens = engine.tokenize_text(text.str());

        std::cout << std::fixed << std::setprecision(4);
        for (std::size_t run = 0; run < runs.size(); ++run) {
            std::uint64_t count = 0;
            std::uint64_t agree = 0;
            double logprob      = 0.0;
            double regret       = 0.0;
            std::vector<std::uint32_t> first_disagreement;
            for (const Segment& segment : runs[run]) {
                if (segment.offset + segment.prompt_tokens > tokens.size()) {
                    throw std::runtime_error("a segment lies outside the corpus: was it the same "
                                             "corpus and artifact tokenizer?");
                }
                std::vector<infernix::TokenId> sequence(
                    tokens.begin() + static_cast<std::ptrdiff_t>(segment.offset),
                    tokens.begin() +
                        static_cast<std::ptrdiff_t>(segment.offset + segment.prompt_tokens));
                sequence.insert(sequence.end(), segment.generated.begin(), segment.generated.end());
                const infernix::ScoreResult scored =
                    engine.score_tokens(std::move(sequence), segment.prompt_tokens, {.top_k = 1});
                auto first = static_cast<std::uint32_t>(segment.generated.size());
                for (std::size_t i = 0; i < segment.generated.size(); ++i) {
                    ++count;
                    logprob += scored.logprobs[i];
                    regret += scored.top_logprobs[i] - scored.logprobs[i];
                    if (scored.top_ids[i] == segment.generated[i]) {
                        ++agree;
                    } else if (first == segment.generated.size()) {
                        first = static_cast<std::uint32_t>(i);
                    }
                }
                first_disagreement.push_back(first);
            }
            std::sort(first_disagreement.begin(), first_disagreement.end());
            const std::uint32_t median =
                first_disagreement.empty() ? 0 : first_disagreement[first_disagreement.size() / 2];
            const double n = static_cast<double>(std::max<std::uint64_t>(count, 1));
            std::cout << argv[3 + run] << ": tokens " << count << ", reference-top agreement "
                      << static_cast<double>(agree) / n << ", mean logprob " << logprob / n
                      << ", mean regret " << regret / n << ", median first disagreement "
                      << median << '\n';
        }
        for (std::size_t a = 0; a < runs.size(); ++a) {
            for (std::size_t b = a + 1; b < runs.size(); ++b) {
                double shared       = 0.0;
                const std::size_t n = std::min(runs[a].size(), runs[b].size());
                for (std::size_t s = 0; s < n; ++s) {
                    const auto& x = runs[a][s].generated;
                    const auto& y = runs[b][s].generated;
                    const auto end =
                        std::mismatch(x.begin(), x.begin() + std::min(x.size(), y.size()),
                                      y.begin())
                            .first;
                    shared += static_cast<double>(end - x.begin());
                }
                std::cout << "shared prefix " << argv[3 + a] << " vs " << argv[3 + b]
                          << ": mean " << shared / static_cast<double>(std::max<std::size_t>(n, 1))
                          << " tokens\n";
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "infernix_decode_quality_judge: " << error.what() << '\n';
        return 1;
    }
}
