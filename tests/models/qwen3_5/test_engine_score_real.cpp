#include "ninfer/engine.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

int main() {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "SKIP: NINFER_TEST_ARTIFACT is not set\n";
        return 77;
    }

    ninfer::EngineOptions options;
    options.artifact_path = artifact;
    options.purpose       = ninfer::EnginePurpose::CausalScoring;
    options.max_context   = 2048;
    options.kv_cache      = ninfer::KvCacheStorage::Fp8E4M3Row256;
    ninfer::Engine engine(options);
    const auto& effective = engine.options();
    if (effective.max_concurrency != 1 || effective.prefill_chunk != 1024 ||
        effective.kv_capacity.mode != ninfer::KvCapacityMode::Explicit ||
        effective.kv_capacity.explicit_tokens != effective.max_context ||
        effective.context_cache.enabled ||
        effective.speculative.backend != ninfer::SpeculativeBackend::None ||
        effective.kv_cache != ninfer::KvCacheStorage::Fp8E4M3Row256) {
        std::cerr << "causal scoring options were not normalized correctly\n";
        return 1;
    }

    std::string text;
    const std::string paragraph =
        "NInfer scores each target token from the preceding hidden state. "
        "Every evaluation window owns fresh state and a fresh KV address space.\n";
    std::vector<ninfer::TokenId> tokens;
    while (tokens.size() < 1537) {
        text += paragraph;
        tokens = engine.tokenize_text(text);
    }
    tokens.resize(1537);

    const std::vector<float> all      = engine.score_tokens(tokens, 1).logprobs;
    const std::vector<float> suffix   = engine.score_tokens(tokens, 513).logprobs;
    const std::vector<float> repeated = engine.score_tokens(tokens, 513).logprobs;
    if (all.size() != 1536 || suffix.size() != 1024 || repeated.size() != suffix.size()) {
        std::cerr << "causal scoring returned an invalid result shape\n";
        return 1;
    }
    for (const float value : all) {
        if (!std::isfinite(value) || value > 0.0F) {
            std::cerr << "causal scoring returned an invalid log probability\n";
            return 1;
        }
    }
    for (std::size_t i = 0; i < suffix.size(); ++i) {
        if (!std::isfinite(suffix[i]) || suffix[i] > 0.0F) {
            std::cerr << "causal scoring returned a non-finite logprob\n";
            return 1;
        }
        if (suffix[i] != repeated[i]) {
            std::cerr << "a repeated score window inherited prior State/KV\n";
            return 1;
        }
    }

    // Distribution outputs: the top tokens of every scored position, then the same tokens as
    // candidates. Both read the same logits and log-normalizer as the target log-probability.
    constexpr std::uint32_t kTop  = 8;
    const ninfer::ScoreResult top = engine.score_tokens(tokens, 513, {.top_k = kTop});
    if (top.logprobs != suffix || top.top_ids.size() != suffix.size() * kTop ||
        top.top_logprobs.size() != top.top_ids.size() || !top.candidate_logprobs.empty()) {
        std::cerr << "top-token scoring returned an invalid result\n";
        return 1;
    }
    for (std::size_t i = 0; i < suffix.size(); ++i) {
        const ninfer::TokenId target = tokens[513 + i];
        for (std::uint32_t k = 0; k < kTop; ++k) {
            const float value = top.top_logprobs[i * kTop + k];
            if (!std::isfinite(value) || value > 0.0F ||
                (k > 0 && value > top.top_logprobs[i * kTop + k - 1]) ||
                (top.top_ids[i * kTop + k] == target && value != suffix[i])) {
                std::cerr << "top tokens are not ordered or disagree with the target\n";
                return 1;
            }
        }
        if (top.top_logprobs[i * kTop] < suffix[i]) {
            std::cerr << "a target is more probable than the top token\n";
            return 1;
        }
    }
    ninfer::ScoreOptions candidates{.candidates_per_position = kTop, .candidates = top.top_ids};
    const ninfer::ScoreResult scored = engine.score_tokens(tokens, 513, candidates);
    if (scored.logprobs != suffix || scored.candidate_logprobs != top.top_logprobs ||
        !scored.top_ids.empty()) {
        std::cerr << "candidate scoring disagrees with the top-token log-probabilities\n";
        return 1;
    }
    const auto rejects = [&](ninfer::ScoreOptions invalid) {
        try {
            (void)engine.score_tokens(tokens, 513, std::move(invalid));
        } catch (const std::invalid_argument&) { return true; }
        return false;
    };
    if (!rejects({.top_k = ninfer::kMaximumScoreTopTokens + 1}) ||
        !rejects({.candidates_per_position = 1, .candidates = {0}}) ||
        !rejects({.candidates_per_position = 1,
                  .candidates = std::vector<ninfer::TokenId>(suffix.size(), -1)})) {
        std::cerr << "invalid distribution outputs were accepted\n";
        return 1;
    }
    std::cout << "OK causal_score_real\n";
    return 0;
}
