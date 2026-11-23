#include "infernix/engine.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

int main() {
    const char* artifact = std::getenv("INFERNIX_TEST_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "SKIP: INFERNIX_TEST_ARTIFACT is not set\n";
        return 77;
    }

    infernix::EngineOptions options;
    options.artifact_path = artifact;
    options.purpose       = infernix::EnginePurpose::CausalScoring;
    options.max_context   = 2048;
    options.kv_cache      = infernix::KvCacheStorage::Fp8E4M3Row256;
    infernix::Engine engine(options);
    const auto& effective = engine.options();
    if (effective.max_concurrency != 1 || effective.prefill_chunk != 1024 ||
        effective.kv_capacity.mode != infernix::KvCapacityMode::Explicit ||
        effective.kv_capacity.explicit_tokens != effective.max_context ||
        effective.context_cache.enabled ||
        effective.speculative.backend != infernix::SpeculativeBackend::None ||
        effective.kv_cache != infernix::KvCacheStorage::Fp8E4M3Row256) {
        std::cerr << "causal scoring options were not normalized correctly\n";
        return 1;
    }

    std::string text;
    const std::string paragraph =
        "Infernix scores each target token from the preceding hidden state. "
        "Every evaluation window owns fresh state and a fresh KV address space.\n";
    std::vector<infernix::TokenId> tokens;
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
    const infernix::ScoreResult top = engine.score_tokens(tokens, 513, {.top_k = kTop});
    if (top.logprobs != suffix || top.top_ids.size() != suffix.size() * kTop ||
        top.top_logprobs.size() != top.top_ids.size() || !top.candidate_logprobs.empty()) {
        std::cerr << "top-token scoring returned an invalid result\n";
        return 1;
    }
    for (std::size_t i = 0; i < suffix.size(); ++i) {
        const infernix::TokenId target = tokens[513 + i];
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
    infernix::ScoreOptions candidates{.candidates_per_position = kTop, .candidates = top.top_ids};
    const infernix::ScoreResult scored = engine.score_tokens(tokens, 513, candidates);
    if (scored.logprobs != suffix || scored.candidate_logprobs != top.top_logprobs ||
        !scored.top_ids.empty()) {
        std::cerr << "candidate scoring disagrees with the top-token log-probabilities\n";
        return 1;
    }
    const auto rejects = [&](infernix::ScoreOptions invalid) {
        try {
            (void)engine.score_tokens(tokens, 513, std::move(invalid));
        } catch (const std::invalid_argument&) { return true; }
        return false;
    };
    if (!rejects({.top_k = infernix::kMaximumScoreTopTokens + 1}) ||
        !rejects({.candidates_per_position = 1, .candidates = {0}}) ||
        !rejects({.candidates_per_position = 1,
                  .candidates = std::vector<infernix::TokenId>(suffix.size(), -1)})) {
        std::cerr << "invalid distribution outputs were accepted\n";
        return 1;
    }

    // Every layer's hidden state at the last prompt token (ScoreOptions::prompt_hidden_states): the scores unchanged,
    // one row per layer. The rows are read at the prompt's last column, in the first prefill chunk and in the second:
    // another token after it (the first target, or a shorter window's narrower chunk) leaves them alone within 1e-2
    // relative L2 per layer (kernels may round a column by its neighbours; a later column is off by order one), and
    // another token at it moves them by order one (an earlier column would not move: the scored prefix is causal).
    const auto worst_relative = [](const infernix::ScoreResult& a, const infernix::ScoreResult& b) -> double {
        const std::size_t layers = a.hidden_state_layers;
        if (layers == 0 || layers != b.hidden_state_layers || a.hidden_states.size() != b.hidden_states.size()) {
            return INFINITY;
        }
        const std::size_t width = a.hidden_states.size() / layers;
        double worst            = 0;
        for (std::size_t layer = 0; layer < layers; ++layer) {
            double diff = 0, norm = 0;
            for (std::size_t i = layer * width; i < (layer + 1) * width; ++i) {
                const double d = static_cast<double>(a.hidden_states[i]) - static_cast<double>(b.hidden_states[i]);
                diff += d * d;
                norm += static_cast<double>(b.hidden_states[i]) * static_cast<double>(b.hidden_states[i]);
            }
            worst = std::max(worst, std::sqrt(diff / std::max(norm, 1e-30)));
        }
        return worst;
    };
    const infernix::ScoreOptions hidden_options{.prompt_hidden_states = true};
    const infernix::ScoreResult hidden = engine.score_tokens(tokens, 513, hidden_options);
    const std::size_t layers           = hidden.hidden_state_layers;
    if (hidden.logprobs != suffix || layers == 0 || hidden.hidden_states.size() % layers != 0 ||
        !std::all_of(hidden.hidden_states.begin(), hidden.hidden_states.end(), [](float v) { return std::isfinite(v); }) ||
        !top.hidden_states.empty()) {
        std::cerr << "the prompt's hidden states changed the scores or have an invalid shape\n";
        return 1;
    }
    std::vector<infernix::TokenId> other = tokens;
    other[513]                           = tokens[513] == tokens[0] ? tokens[1] : tokens[0];
    const infernix::ScoreResult changed  = engine.score_tokens(other, 513, hidden_options);
    // Position 1499, in the second 1,024-token chunk: the 1,537-token window against a 1,501-token one.
    const infernix::ScoreResult wide = engine.score_tokens(tokens, 1500, hidden_options);
    const infernix::ScoreResult narrow =
        engine.score_tokens(std::vector<infernix::TokenId>(tokens.begin(), tokens.begin() + 1501), 1500, hidden_options);
    const auto moved_at = [&](std::uint32_t first_target) {
        std::vector<infernix::TokenId> moved = tokens;
        const std::size_t at                 = first_target - 1U;
        moved[at]                            = tokens[at] == tokens[0] ? tokens[1] : tokens[0];
        return worst_relative(engine.score_tokens(moved, first_target, hidden_options),
                              engine.score_tokens(tokens, first_target, hidden_options));
    };
    const double changed_worst = worst_relative(changed, hidden), chunk_worst = worst_relative(narrow, wide);
    const double moved_first = moved_at(513), moved_second = moved_at(1500);
    std::cout << "hidden states: " << layers << " layers x " << hidden.hidden_states.size() / layers
              << "; another first target token: worst layer relative L2 " << changed_worst
              << (changed.hidden_states == hidden.hidden_states ? " (bit-identical)" : "")
              << "; position 1499 in a narrower second chunk: " << chunk_worst
              << (narrow.hidden_states == wide.hidden_states ? " (bit-identical)" : "")
              << "; another token at the captured position: " << moved_first << " (position 512), " << moved_second
              << " (position 1499)\n";
    if (changed_worst > 1e-2 || chunk_worst > 1e-2 || moved_first < 0.1 || moved_second < 0.1) {
        std::cerr << "the hidden states are not the prompt position's\n";
        return 1;
    }
    std::cout << "OK causal_score_real\n";
    return 0;
}
