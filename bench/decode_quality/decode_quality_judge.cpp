// End-to-end decode quality, judge half. Scores the greedy continuations that
// infernix_decode_quality_gen wrote, for any number of builds, with one reference: a CausalScoring
// Engine on the lossless route (EngineOptions::lossless), BF16 KV, and each sequence in one
// prefill pass. Each build is judged on its own prefixes, so builds whose
// continuations diverge are still compared fairly. Per build it reports how often the generated
// token is the reference's most probable token, the mean reference log-probability of the
// generated tokens, and the mean regret (top log-probability minus the generated token's); per
// pair of builds, the generated prefix they share exactly.
//
// --stored-activations scores on the reproducible route with the activations the artifact records
// instead (every other condition the same): it isolates what decoding changes (KV format, chunking,
// speculation, batching) but not the cost of the stored activation precision itself.
//
// --default-route scores on the default route, fast paths included. Neither it nor --stored-activations
// reproduces the generator's decode: on NVIDIA's NVFP4 27B the prompt path quantizes activations to 4 bits
// and decode keeps 16, and these references' kernels follow the sequence's length, which alone moved one
// token's regret from under 1 nat to 19.9 (candidates YK-04, 2026-11-08). A token they rank low too is
// therefore no evidence of a decode-state error; infernix_decode_quality_probe tests that directly.
//
// --report FILE also writes each build's per-segment counts as JSON (segments are independent
// samples, which a comparison pairs between builds).
//
// --ngram-volume FILE names a Qwen3.8-Flash-Next n-gram volume (default <artifact>.ngram).
//
// --prompts reads the corpus as whole prompts, each ended by a line "<<<END>>>", as the generator's
// `prompts` does (a segment's offset is then its prompt's index).
//
// Per build it also reports the largest regret and how many tokens' regret exceeds 1 nat: a state or
// verification error leaves tokens far below the reference's choice, which near-ties do not. So does a
// route's rounding at a position where the model itself is unstable (a one-token shift of the context
// changes its choice): there a decode can pick a confident other token.
//
// Usage: infernix_decode_quality_judge [--stored-activations | --default-route] [--report <file.json>]
//                                      [--ngram-volume <file>] [--prompts] <artifact> <corpus.txt> <gen_file>...

#include "infernix/engine.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
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

struct SegmentScore {
    std::uint64_t tokens = 0;
    std::uint64_t agree  = 0;
    double logprob       = 0.0;
    double regret        = 0.0;
    double worst_regret  = 0.0;
    std::uint64_t above_one = 0;
};

} // namespace

int main(int argc, char** argv) {
    bool stored = false, default_route = false, prompt_records = false;
    std::string report_path;
    std::string ngram_volume;
    while (argc > 1 && std::string(argv[1]).starts_with("--")) {
        const std::string option = argv[1];
        if (option == "--stored-activations") {
            stored = true;
        } else if (option == "--default-route") {
            default_route = true;
        } else if (option == "--prompts") {
            prompt_records = true;
        } else if (option == "--report" && argc > 2) {
            report_path = argv[2];
            ++argv;
            --argc;
        } else if (option == "--ngram-volume" && argc > 2) {
            ngram_volume = argv[2];
            ++argv;
            --argc;
        } else {
            std::cerr << "unknown option " << option << '\n';
            return 2;
        }
        ++argv;
        --argc;
    }
    if (argc < 4 || (stored && default_route)) {
        std::cerr << "usage: infernix_decode_quality_judge [--stored-activations | --default-route] "
                     "[--report <file.json>] [--ngram-volume <file>] [--prompts] <artifact> <corpus.txt> "
                     "<gen_file>...\n";
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
        options.lossless        = !stored && !default_route;
        // Only the default-route reference runs fast paths.
        options.fast_paths      = default_route;
        if (!ngram_volume.empty()) { options.ngram_volume_path = ngram_volume; }
        infernix::Engine engine(options);

        std::ifstream corpus(argv[2], std::ios::binary);
        if (!corpus) { throw std::runtime_error(std::string("cannot read ") + argv[2]); }
        std::stringstream text;
        text << corpus.rdbuf();
        // `--prompts`: each prompt record tokenized on its own, as the generator does.
        std::vector<std::vector<infernix::TokenId>> records;
        if (prompt_records) {
            const std::string all = text.str();
            const std::string end = "\n<<<END>>>\n";
            for (std::size_t begin = 0, at = all.find(end); at != std::string::npos;
                 begin = at + end.size(), at = all.find(end, begin)) {
                records.push_back(engine.tokenize_text(std::string_view(all).substr(begin, at - begin)));
            }
        }
        const std::vector<infernix::TokenId> tokens =
            prompt_records ? std::vector<infernix::TokenId>{} : engine.tokenize_text(text.str());

        std::cout << std::fixed << std::setprecision(4);
        std::vector<std::vector<SegmentScore>> scores(runs.size());
        for (std::size_t run = 0; run < runs.size(); ++run) {
            std::uint64_t count = 0;
            std::uint64_t agree = 0;
            double logprob      = 0.0;
            double regret       = 0.0;
            double worst        = 0.0;
            std::uint64_t above = 0;
            std::vector<std::uint32_t> first_disagreement;
            for (const Segment& segment : runs[run]) {
                SegmentScore& score = scores[run].emplace_back();
                std::vector<infernix::TokenId> sequence;
                if (prompt_records) {
                    if (segment.offset >= records.size() || records[segment.offset].size() != segment.prompt_tokens) {
                        throw std::runtime_error("a segment's prompt is not the corpus's: was it the same "
                                                 "prompt file and artifact tokenizer?");
                    }
                    sequence = records[segment.offset];
                } else {
                    if (segment.offset + segment.prompt_tokens > tokens.size()) {
                        throw std::runtime_error("a segment lies outside the corpus: was it the same "
                                                 "corpus and artifact tokenizer?");
                    }
                    sequence.assign(tokens.begin() + static_cast<std::ptrdiff_t>(segment.offset),
                                    tokens.begin() +
                                        static_cast<std::ptrdiff_t>(segment.offset + segment.prompt_tokens));
                }
                sequence.insert(sequence.end(), segment.generated.begin(), segment.generated.end());
                const infernix::ScoreResult scored =
                    engine.score_tokens(std::move(sequence), segment.prompt_tokens, {.top_k = 1});
                auto first = static_cast<std::uint32_t>(segment.generated.size());
                for (std::size_t i = 0; i < segment.generated.size(); ++i) {
                    ++score.tokens;
                    score.logprob += scored.logprobs[i];
                    const double regret_i = scored.top_logprobs[i] - scored.logprobs[i];
                    score.regret += regret_i;
                    score.worst_regret = std::max(score.worst_regret, regret_i);
                    score.above_one += regret_i > 1.0 ? 1U : 0U;
                    if (regret_i > 1.0) {
                        std::cerr << argv[3 + run] << " segment " << scores[run].size() - 1 << " token " << i
                                  << ": generated " << segment.generated[i] << ", reference top "
                                  << scored.top_ids[i] << ", regret " << regret_i << '\n';
                    }
                    if (scored.top_ids[i] == segment.generated[i]) {
                        ++score.agree;
                    } else if (first == segment.generated.size()) {
                        first = static_cast<std::uint32_t>(i);
                    }
                }
                count += score.tokens;
                agree += score.agree;
                logprob += score.logprob;
                regret += score.regret;
                worst = std::max(worst, score.worst_regret);
                above += score.above_one;
                first_disagreement.push_back(first);
            }
            std::sort(first_disagreement.begin(), first_disagreement.end());
            const std::uint32_t median =
                first_disagreement.empty() ? 0 : first_disagreement[first_disagreement.size() / 2];
            const double n = static_cast<double>(std::max<std::uint64_t>(count, 1));
            std::cout << argv[3 + run] << ": tokens " << count << ", reference-top agreement "
                      << static_cast<double>(agree) / n << ", mean logprob " << logprob / n
                      << ", mean regret " << regret / n << ", worst regret " << worst
                      << ", regrets above 1 nat " << above << ", median first disagreement " << median << '\n';
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
        if (!report_path.empty()) {
            std::ofstream report(report_path);
            if (!report) { throw std::runtime_error("cannot write " + report_path); }
            report << std::setprecision(17) << "{\"reference\": \""
                   << (stored ? "stored_activations" : default_route ? "default_route" : "lossless")
                   << "\", \"runs\": [";
            for (std::size_t run = 0; run < runs.size(); ++run) {
                // Generation files are paths the user chose; JSON-escape the two characters a
                // Windows path may hold that JSON reserves.
                std::string name;
                for (const char c : std::string(argv[3 + run])) {
                    if (c == '\\' || c == '"') { name.push_back('\\'); }
                    name.push_back(c);
                }
                report << (run ? ", " : "") << "{\"file\": \"" << name << "\", \"segments\": [";
                for (std::size_t s = 0; s < scores[run].size(); ++s) {
                    const SegmentScore& score = scores[run][s];
                    report << (s ? ", " : "") << "{\"offset\": " << runs[run][s].offset
                           << ", \"tokens\": " << score.tokens << ", \"agree\": " << score.agree
                           << ", \"logprob\": " << score.logprob << ", \"regret\": " << score.regret
                           << '}';
                }
                report << "]}";
            }
            report << "]}\n";
            if (!report) { throw std::runtime_error("cannot write " + report_path); }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "infernix_decode_quality_judge: " << error.what() << '\n';
        return 1;
    }
}
