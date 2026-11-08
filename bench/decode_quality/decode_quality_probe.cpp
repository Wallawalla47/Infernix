// End-to-end decode quality, probe: is a judged token's large regret a decode-state error? For a
// position the judge flagged (segment s, token t of an infernix_decode_quality_gen output), it runs the
// generator's route and KV format with plain decoding and forces the generated tokens through the decode path
// from several split points: the prompt plus the first t - d generated tokens are prefilled, and the remaining
// d tokens are forced one decode step each, so the distribution at token t comes from the prefill when d = 0
// and from the d-th decode step otherwise. Each split reports the log-probabilities, under that distribution,
// of the generated token and of the prefill's most probable token, and the distribution's largest
// log-probability. A route whose decode reproduces its prefill gives the same figures at every d; the smallest
// d at which the generated token overtakes the prefill's choice is how much decode state the error needs (d =
// 1: one decode step; growing d: state that accumulates). A token that flips back and forth with d, with the
// KV format (--kv) or with a one-token shift of the context (--trim, --pad) is a route's rounding at a
// position where the model is unstable, not a fault in one step. The d = t split (no generated token prefilled)
// replays the generator's own decode, so its generated token must be the most probable at every position
// (reported as "replay": a nonzero regret means the forced replay is not the generator's decode).
//
// Usage: infernix_decode_quality_probe [--route reproducible|default|lossless] [--kv bf16|int8|fp8]
//                                      [--context N] [--prefill-chunk N] [--splits d1,d2,...] [--prompts] [--show]
//                                      [--no-cuda-graph] [--trim N | --pad N]
//                                      [--ngram-volume FILE] <artifact> <corpus.txt> <gen_file> <segment>:<token>...
// --no-cuda-graph decodes eagerly. --trim N drops the prompt's first N tokens and --pad N puts N newline tokens before
// it: the same content at a context N shorter or longer (a fault that follows the context length moves, one that
// follows the content stays).
// --show also prints each target's text: the context's last 24 tokens, the generated token, the prefill's top and
// the next 12 generated tokens, and every earlier place where the longest suffix of the context (up to 32 tokens)
// recurs with the token that followed it there (a copying continuation picks one of those sources).
// --splits defaults to 0,1,2,3,4,6,8,12,16,24,32,48,64,96,128,192,256,384 and always adds d = t. --context and
// --prefill-chunk default to the generator's (its prompt_tokens + new_tokens + 64, rounded up to 128; 4096): pass
// the generator's --context when it differs, so that the replay runs the generator's options.

#include "infernix/engine.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <span>
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

std::vector<std::uint32_t> parse_list(const std::string& text) {
    std::vector<std::uint32_t> values;
    std::istringstream in(text);
    std::string item;
    while (std::getline(in, item, ',')) { values.push_back(static_cast<std::uint32_t>(std::stoul(item))); }
    return values;
}

// One split's figures at the target token.
struct Split {
    float generated = 0.0F; // log p(generated token)
    float choice    = 0.0F; // log p(the prefill's most probable token)
    float top       = 0.0F; // the distribution's largest log-probability
    infernix::TokenId top_token = 0; // d = 0 only (the readout names it)
    // Forced tokens before the target that were not their step's most probable (regret > 1e-3), and the
    // largest such regret.
    std::uint32_t off_top = 0;
    float worst_before    = 0.0F;
};

} // namespace

int main(int argc, char** argv) {
    std::string route = "reproducible", kv_name = "bf16", ngram_volume;
    std::vector<std::uint32_t> splits{0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384};
    bool prompt_records         = false;
    bool show                   = false;
    bool cuda_graph             = true;
    std::uint32_t trim = 0, pad = 0;
    std::uint32_t context       = 0;
    std::uint32_t prefill_chunk = 4096;
    int at                      = 1;
    for (; at < argc && std::string_view(argv[at]).starts_with("--"); ++at) {
        const std::string option = argv[at];
        if (option == "--prompts") {
            prompt_records = true;
        } else if (option == "--show") {
            show = true;
        } else if (option == "--no-cuda-graph") {
            cuda_graph = false;
        } else if (at + 1 < argc && option == "--trim") {
            trim = static_cast<std::uint32_t>(std::stoul(argv[++at]));
        } else if (at + 1 < argc && option == "--pad") {
            pad = static_cast<std::uint32_t>(std::stoul(argv[++at]));
        } else if (at + 1 < argc && option == "--route") {
            route = argv[++at];
        } else if (at + 1 < argc && option == "--kv") {
            kv_name = argv[++at];
        } else if (at + 1 < argc && option == "--context") {
            context = static_cast<std::uint32_t>(std::stoul(argv[++at]));
        } else if (at + 1 < argc && option == "--prefill-chunk") {
            prefill_chunk = static_cast<std::uint32_t>(std::stoul(argv[++at]));
        } else if (at + 1 < argc && option == "--splits") {
            splits = parse_list(argv[++at]);
        } else if (at + 1 < argc && option == "--ngram-volume") {
            ngram_volume = argv[++at];
        } else {
            std::cerr << "unknown option " << option << '\n';
            return 2;
        }
    }
    if (argc - at < 4 || (route != "reproducible" && route != "default" && route != "lossless")) {
        std::cerr << "usage: infernix_decode_quality_probe [--route reproducible|default|lossless] [--kv bf16|int8|fp8] "
                     "[--splits d1,d2,...] [--prompts] [--ngram-volume FILE] <artifact> <corpus.txt> <gen_file> "
                     "<segment>:<token>...\n";
        return 2;
    }
    try {
        const std::vector<Segment> segments = read_generation(argv[at + 2]);
        struct Target {
            std::uint32_t segment = 0, token = 0;
        };
        std::vector<Target> targets;
        std::uint32_t longest = 0;
        for (int i = at + 3; i < argc; ++i) {
            const std::string text = argv[i];
            const std::size_t colon = text.find(':');
            if (colon == std::string::npos) { throw std::invalid_argument("a target is <segment>:<token>: " + text); }
            const Target target{static_cast<std::uint32_t>(std::stoul(text.substr(0, colon))),
                                static_cast<std::uint32_t>(std::stoul(text.substr(colon + 1)))};
            if (target.segment >= segments.size() || target.token >= segments[target.segment].generated.size()) {
                throw std::invalid_argument("target outside the generation file: " + text);
            }
            targets.push_back(target);
            longest = std::max(longest, segments[target.segment].prompt_tokens + pad + target.token + 1);
        }

        infernix::EngineOptions options;
        options.artifact_path = argv[at];
        options.max_context   = context != 0 ? context : (longest + 64 + 127) / 128 * 128;
        if (options.max_context < longest) { throw std::invalid_argument("--context is shorter than a target's context"); }
        options.kv_capacity   = infernix::KvCapacityPolicy::explicit_capacity(options.max_context);
        options.kv_cache      = kv_name == "int8"  ? infernix::KvCacheStorage::Int8Group64
                                : kv_name == "fp8" ? infernix::KvCacheStorage::Fp8E4M3Row256
                                                   : infernix::KvCacheStorage::BFloat16;
        if (kv_name != "bf16" && kv_name != "int8" && kv_name != "fp8") { throw std::invalid_argument("unknown --kv"); }
        options.max_concurrency                   = 1;
        options.use_cuda_graph                    = cuda_graph;
        options.prefill_chunk                     = prefill_chunk;
        options.speculative                       = {};
        options.context_cache.enabled             = false;
        options.context_cache.host_capacity_bytes = 0;
        options.fast_paths                        = route == "default";
        options.lossless                          = route == "lossless";
        if (!ngram_volume.empty()) { options.ngram_volume_path = ngram_volume; }
        infernix::Engine engine(options);

        std::ifstream corpus(argv[at + 1], std::ios::binary);
        if (!corpus) { throw std::runtime_error(std::string("cannot read ") + argv[at + 1]); }
        std::stringstream text;
        text << corpus.rdbuf();
        std::vector<std::vector<infernix::TokenId>> records;
        if (prompt_records) {
            const std::string all = text.str();
            const std::string end = "\n<<<END>>>\n";
            for (std::size_t begin = 0, found = all.find(end); found != std::string::npos;
                 begin = found + end.size(), found = all.find(end, begin)) {
                records.push_back(engine.tokenize_text(std::string_view(all).substr(begin, found - begin)));
            }
        }
        const std::vector<infernix::TokenId> tokens =
            prompt_records ? std::vector<infernix::TokenId>{} : engine.tokenize_text(text.str());

        // One forced run: the prompt plus generated[0, j), then generated[j, t] forced one step each.
        const auto run = [&](const std::vector<infernix::TokenId>& prompt, const std::vector<infernix::TokenId>& generated,
                             std::uint32_t j, std::uint32_t t, infernix::TokenId choice, bool know_choice) {
            std::vector<infernix::TokenId> input = prompt;
            input.insert(input.end(), generated.begin(), generated.begin() + j);
            infernix::RequestOptions request;
            request.execution.requested_output_tokens = t - j + 1;
            request.execution.sampling.temperature    = 0.0F;
            request.execution.allow_prefix_reuse      = false;
            request.stop.include_model_defaults       = false;
            request.execution.readout_tokens          = {generated[t]};
            if (know_choice && choice != generated[t]) { request.execution.readout_tokens.push_back(choice); }
            for (std::uint32_t i = j; i <= t; ++i) {
                request.execution.constraint.steps.push_back({generated[i]});
                std::vector<infernix::TokenId> observe{generated[i]};
                if (i == t && know_choice && choice != generated[t]) { observe.push_back(choice); }
                request.execution.constraint.observe.push_back(std::move(observe));
            }
            const infernix::GenerationResult result = engine.generate(engine.prepare_tokens(std::move(input), false), request);
            if (result.constrained_draws.size() != t - j + 1 || !result.readout) {
                throw std::runtime_error("a forced run returned " + std::to_string(result.constrained_draws.size()) +
                                         " draws of " + std::to_string(t - j + 1) + (result.readout ? "" : ", no readout"));
            }
            Split split;
            const infernix::ConstrainedDraw& last = result.constrained_draws.back();
            split.generated = last.observed_logprobs.at(0);
            split.choice    = last.observed_logprobs.size() > 1 ? last.observed_logprobs[1] : split.generated;
            split.top       = last.top_logprob;
            split.top_token = result.readout->top_token;
            for (std::size_t k = 0; k + 1 < result.constrained_draws.size(); ++k) {
                const infernix::ConstrainedDraw& draw = result.constrained_draws[k];
                const float regret                    = draw.top_logprob - draw.observed_logprobs.at(0);
                if (regret > 1e-3F) {
                    ++split.off_top;
                    split.worst_before = std::max(split.worst_before, regret);
                }
            }
            return split;
        };

        std::cout << std::fixed << std::setprecision(4);
        std::cout << "route " << route << ", KV " << kv_name << ", plain decode" << (cuda_graph ? "" : ", eager")
                  << (trim ? ", trim " + std::to_string(trim) : "") << (pad ? ", pad " + std::to_string(pad) : "") << ", "
                  << argv[at + 2] << '\n';
        for (const Target& target : targets) {
            const Segment& segment = segments[target.segment];
            std::vector<infernix::TokenId> prompt;
            if (prompt_records) {
                if (segment.offset >= records.size() || records[segment.offset].size() != segment.prompt_tokens) {
                    throw std::runtime_error("a segment's prompt is not the corpus's");
                }
                prompt = records[segment.offset];
            } else {
                if (segment.offset + segment.prompt_tokens > tokens.size()) {
                    throw std::runtime_error("a segment lies outside the corpus");
                }
                prompt.assign(tokens.begin() + static_cast<std::ptrdiff_t>(segment.offset),
                              tokens.begin() + static_cast<std::ptrdiff_t>(segment.offset + segment.prompt_tokens));
            }
            if (trim >= prompt.size()) { throw std::invalid_argument("--trim removes the whole prompt"); }
            prompt.erase(prompt.begin(), prompt.begin() + trim);
            const std::vector<infernix::TokenId> newline = engine.tokenize_text("\n");
            if (pad != 0 && newline.size() != 1) { throw std::runtime_error("a newline is not one token"); }
            prompt.insert(prompt.begin(), pad, pad != 0 ? newline[0] : 0);
            const std::uint32_t t = target.token;
            // d = 0 first: the prefill's view and its most probable token.
            const Split prefill = run(prompt, segment.generated, t, t, 0, false);
            const infernix::TokenId choice = prefill.top_token;
            std::cout << "segment " << target.segment << " token " << t << " (context " << prompt.size() + t
                      << "): generated " << segment.generated[t] << ", prefill's top " << choice << '\n';
            if (show) {
                // The text, quoted with escapes so that newlines and spaces stay visible.
                const auto quote = [&](std::span<const infernix::TokenId> ids) {
                    std::string text = engine.detokenize(ids), quoted = "\"";
                    for (const char c : text) {
                        if (c == '\n') { quoted += "\\n"; } else if (c == '"') { quoted += "\\\""; } else { quoted += c; }
                    }
                    return quoted + "\"";
                };
                std::vector<infernix::TokenId> all = prompt;
                all.insert(all.end(), segment.generated.begin(), segment.generated.begin() + t);
                const std::size_t n = all.size();
                const std::size_t from = n > 24 ? n - 24 : 0;
                std::cout << "  context ..." << quote(std::span(all).subspan(from)) << '\n';
                std::cout << "  generated " << quote(std::span(&segment.generated[t], 1)) << ", prefill's top "
                          << quote(std::span(&choice, 1)) << ", then "
                          << quote(std::span(segment.generated).subspan(t + 1, std::min<std::size_t>(12, segment.generated.size() - t - 1)))
                          << '\n';
                // The longest context suffix that recurs earlier, and what followed each recurrence.
                std::size_t best = 0;
                std::vector<std::size_t> ends;
                for (std::size_t k = std::min<std::size_t>(32, n - 1); k >= 2 && best == 0; --k) {
                    for (std::size_t end = k; end < n; ++end) { // all[end - k, end) == all[n - k, n), followed by all[end]
                        if (std::equal(all.begin() + static_cast<std::ptrdiff_t>(end - k), all.begin() + static_cast<std::ptrdiff_t>(end),
                                       all.begin() + static_cast<std::ptrdiff_t>(n - k))) {
                            best = k;
                            ends.push_back(end);
                        }
                    }
                }
                std::cout << "  the context's last " << best << " tokens recur " << ends.size() << " time(s) earlier, followed by:";
                for (const std::size_t end : ends) {
                    std::cout << ' ' << all[end] << '=' << quote(std::span(&all[end], 1)) << " (at " << end << ')';
                }
                std::cout << '\n';
            }
            std::vector<std::uint32_t> ds;
            for (const std::uint32_t d : splits) {
                if (d <= t) { ds.push_back(d); }
            }
            ds.push_back(t);
            std::sort(ds.begin(), ds.end());
            ds.erase(std::unique(ds.begin(), ds.end()), ds.end());
            for (const std::uint32_t d : ds) {
                const Split split = d == 0 ? run(prompt, segment.generated, t, t, choice, true)
                                           : run(prompt, segment.generated, t - d, t, choice, true);
                std::cout << "  d " << std::setw(4) << d << ": log p(generated) " << std::setw(9) << split.generated
                          << ", log p(prefill top) " << std::setw(9) << split.choice << ", top " << std::setw(9)
                          << split.top << ", regret " << std::setw(8) << split.top - split.generated
                          << ", forced tokens off their step's top " << split.off_top << " (worst " << split.worst_before
                          << ")" << (d == t ? "  <- replay of the generator's decode" : "") << '\n';
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "infernix_decode_quality_probe: " << error.what() << '\n';
        return 1;
    }
}
