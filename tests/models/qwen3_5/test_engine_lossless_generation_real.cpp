// Lossless generation on a real Qwen3.5 artifact (INFERNIX_TEST_ARTIFACT, docs/maintainer/lossless-route.md):
// every token is a function of the request and the artifact alone.
//  - Greedy generation picks the lossless scoring route's top token at every position, so prefill in
//    chunks, the KV cache, the recurrent state slots and decode compute what one scoring pass computes.
//  - Sampled requests (temperature 1, fixed seeds, which make any arithmetic difference a token
//    difference) give the same tokens alone, concurrently in another order, with another prefill chunk
//    and resumed from the prefix cache.

#include "infernix/engine.h"

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::uint32_t kContext = 4096;
constexpr std::uint32_t kOutput  = 64;

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

infernix::EngineOptions generation_options(const char* artifact, std::uint32_t chunk, std::uint32_t lanes,
                                           bool prefix_cache) {
    infernix::EngineOptions options;
    options.artifact_path        = artifact;
    options.lossless             = true;
    options.fast_paths           = false;
    options.kv_cache             = infernix::KvCacheStorage::BFloat16;
    options.max_context          = kContext;
    options.kv_capacity          = infernix::KvCapacityPolicy::explicit_capacity(lanes * kContext);
    options.max_concurrency      = lanes;
    options.max_pending_requests = 2 * lanes;
    options.prefill_chunk        = chunk;
    options.enable_vision        = false;
    options.context_cache.enabled = prefix_cache;
    if (prefix_cache) { options.context_cache.host_capacity_bytes = std::size_t{1} << 30U; }
    return options;
}

infernix::RequestOptions request(bool greedy, std::uint64_t seed, bool prefix_reuse = false) {
    infernix::RequestOptions options;
    options.execution.requested_output_tokens = kOutput;
    options.execution.sampling.temperature    = greedy ? 0.0F : 1.0F;
    options.execution.sampling.top_k          = 0;
    options.execution.sampling.top_p          = 1.0F;
    options.execution.sampling.seed           = seed;
    if (greedy) {
        // Greedy picks the scoring pass's top token only without penalties; the MoE model's thinking
        // preset sets a presence penalty (1.5). Sampled requests keep the model's defaults.
        options.execution.sampling.presence_penalty  = 0.0F;
        options.execution.sampling.frequency_penalty = 0.0F;
    }
    options.execution.allow_prefix_reuse      = prefix_reuse;
    options.stop.include_model_defaults       = false;
    options.output.raw                        = true;
    return options;
}

// Prompts of different lengths; the first (about 1,500 tokens) crosses prefill chunks and KV pages.
std::vector<std::string> texts() {
    std::string notes = "Field notes from the river survey, read them and continue the log.\n";
    for (int entry = 0; entry < 40; ++entry) {
        notes += "Entry " + std::to_string(entry) + ": the team logged " + std::to_string(3 * entry + 7) +
                 " birds near marker " + std::to_string(entry % 17) + ", wind from the " +
                 (entry % 2 ? "north" : "south-west") + ", water level " + std::to_string(40 + entry % 9) +
                 " centimetres.\n";
    }
    return {notes,
            "Write a short story about a lighthouse keeper who finds a message in a bottle.",
            "Explain, step by step, how a compiler turns source code into machine instructions.",
            "List twelve uses for a paperclip and give one sentence of explanation for each."};
}

void require_same(const std::string& label, const std::vector<infernix::TokenId>& a,
                  const std::vector<infernix::TokenId>& b) {
    require(!a.empty(), label + ": the reference generated nothing");
    std::size_t i = 0;
    while (i < a.size() && i < b.size() && a[i] == b[i]) { ++i; }
    if (a != b) {
        throw std::runtime_error(label + ": tokens differ at " + std::to_string(i) + " of " +
                                 std::to_string(a.size()) + "/" + std::to_string(b.size()));
    }
    std::cout << label << ": " << a.size() << " tokens identical\n";
}

infernix::EngineOptions scoring_options(const char* artifact, bool lossless) {
    infernix::EngineOptions options;
    options.artifact_path = artifact;
    options.purpose       = infernix::EnginePurpose::CausalScoring;
    options.max_context   = kContext;
    options.prefill_chunk = kContext;
    options.kv_cache      = infernix::KvCacheStorage::BFloat16;
    options.fast_paths    = false;
    options.lossless      = lossless;
    return options;
}

// When generation departs from the scoring pass at output position m: how each computation sees that
// position, so a failure names the form that departs. The lossless and reproducible scoring passes'
// five most probable tokens there; the lossless engine's prompt readout after a prefill (chunk 512) of
// the prompt and the first m outputs; and its token from one decode step after a prefill of the first
// m - 1 outputs.
void diagnose(const char* artifact, const std::vector<infernix::TokenId>& prompt,
              const std::vector<infernix::TokenId>& greedy, std::size_t m, infernix::TokenId scored) {
    std::vector<infernix::TokenId> sequence = prompt;
    sequence.insert(sequence.end(), greedy.begin(), greedy.end());
    const auto first = static_cast<std::uint32_t>(prompt.size());
    constexpr std::uint32_t kTop = 5;
    for (const bool lossless : {true, false}) {
        infernix::Engine engine(scoring_options(artifact, lossless));
        const infernix::ScoreResult score = engine.score_tokens(sequence, first, {.top_k = kTop});
        std::cout << (lossless ? "lossless" : "reproducible") << " scoring at output " << m << ":";
        for (std::uint32_t j = 0; j < kTop; ++j) {
            std::cout << " " << score.top_ids[m * kTop + j] << " (" << score.top_logprobs[m * kTop + j] << ")";
        }
        std::cout << "\n";
    }
    infernix::Engine engine(generation_options(artifact, 512, 1, false));
    std::vector<infernix::TokenId> through = prompt;
    through.insert(through.end(), greedy.begin(), greedy.begin() + static_cast<std::ptrdiff_t>(m));
    infernix::RequestOptions readout = request(true, 0);
    readout.execution.requested_output_tokens = 1;
    readout.execution.readout_tokens          = {greedy[m], scored};
    const auto prefilled = engine.generate(engine.prepare_tokens(through), readout);
    if (prefilled.readout) {
        std::cout << "lossless prefill readout at output " << m << ": token " << greedy[m] << " "
                  << prefilled.readout->logprobs[0] << ", token " << scored << " " << prefilled.readout->logprobs[1]
                  << ", top " << prefilled.readout->top_token << " " << prefilled.readout->top_logprob << "\n";
    }
    if (m > 0) {
        std::vector<infernix::TokenId> before(through.begin(), through.end() - 1);
        infernix::RequestOptions step = request(true, 0);
        step.execution.requested_output_tokens = 2;
        const auto decoded = engine.generate(engine.prepare_tokens(before), step).generated_token_ids;
        std::cout << "lossless decode step at output " << m << ": " << (decoded.size() > 1 ? decoded[1] : -1)
                  << " (after " << (decoded.empty() ? -1 : decoded[0]) << ")\n";
    }
}

} // namespace

int main() {
    const char* artifact = std::getenv("INFERNIX_TEST_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "SKIP: INFERNIX_TEST_ARTIFACT is not set\n";
        return 77;
    }
    try {
        const auto prompts = texts();
        std::vector<std::vector<infernix::TokenId>> tokens;
        std::vector<infernix::TokenId> greedy;
        std::vector<std::vector<infernix::TokenId>> solo;
        {
            infernix::Engine engine(generation_options(artifact, 512, 4, true));
            for (const std::string& text : prompts) { tokens.push_back(engine.tokenize_text(text)); }
            greedy = engine.generate(engine.prepare_tokens(tokens[0]), request(true, 0)).generated_token_ids;
            require(greedy.size() == kOutput, "the greedy request stopped early");
            // Sampled, each alone, then all at once in reverse order.
            for (std::size_t i = 0; i < tokens.size(); ++i) {
                solo.push_back(
                    engine.generate(engine.prepare_tokens(tokens[i]), request(false, 100 + i)).generated_token_ids);
            }
            std::vector<infernix::GenerationHandle> handles;
            for (std::size_t i = tokens.size(); i-- > 0;) {
                handles.push_back(engine.submit(engine.prepare_tokens(tokens[i]), request(false, 100 + i)));
            }
            for (std::size_t j = 0; j < handles.size(); ++j) {
                const std::size_t i = tokens.size() - 1 - j;
                require_same("concurrent request " + std::to_string(i), solo[i], handles[j].wait().generated_token_ids);
            }
            // The prefix cache: a request sharing the long prompt warms it, then the first resumes from it.
            std::vector<infernix::TokenId> other = tokens[0];
            const auto tail = engine.tokenize_text("\nHow many birds were logged in all?");
            other.insert(other.end(), tail.begin(), tail.end());
            // Only its prompt is needed: it fills the prefix cache.
            infernix::RequestOptions warming = request(false, 7, true);
            warming.execution.requested_output_tokens = 1;
            (void)engine.generate(engine.prepare_tokens(other), warming);
            const auto before = engine.runtime_stats();
            const auto warm   = engine.generate(engine.prepare_tokens(tokens[0]), request(false, 100, true));
            require(engine.runtime_stats().reused_prompt_tokens > before.reused_prompt_tokens,
                    "the warm request reused no prefix");
            require_same("warm prefix", solo[0], warm.generated_token_ids);
        }
        {
            // Another prefill chunk and one lane.
            infernix::Engine engine(generation_options(artifact, 1024, 1, false));
            // Sampled (temperature 1) shows any arithmetic difference greedy decoding would.
            require_same("prefill chunk 1024 (sampled)", solo[0],
                         engine.generate(engine.prepare_tokens(tokens[0]), request(false, 100)).generated_token_ids);
        }
        // One lossless scoring pass over the prompt and the greedy tokens: its top token at every
        // generated position is the generated token.
        std::vector<infernix::TokenId> scored;
        {
            infernix::Engine engine(scoring_options(artifact, true));
            std::vector<infernix::TokenId> sequence = tokens[0];
            sequence.insert(sequence.end(), greedy.begin(), greedy.end());
            const auto first = static_cast<std::uint32_t>(tokens[0].size());
            scored = engine.score_tokens(sequence, first, {.top_k = 1}).top_ids;
            require(scored.size() == greedy.size(), "the score has an unexpected shape");
        }
        for (std::size_t k = 0; k < greedy.size(); ++k) {
            if (scored[k] != greedy[k]) {
                diagnose(artifact, tokens[0], greedy, k, scored[k]);
                throw std::runtime_error("generated token " + std::to_string(k) + " is " + std::to_string(greedy[k]) +
                                         ", the scoring route's top token " + std::to_string(scored[k]));
            }
        }
        std::cout << "greedy generation: " << greedy.size() << " tokens equal the scoring route's top tokens\n";
        std::cout << "OK lossless generation\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "lossless generation: " << error.what() << '\n';
        return 1;
    }
}
