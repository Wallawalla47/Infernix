// Prompt readout on the real artifact (/v1/decide's engine path). The oracle is the CausalScoring
// Engine: log p(candidate | prompt) from score_tokens over prompt + [any token] at the last
// position, an independent route over the same weights. The readout must also agree with the
// greedy token it stages, survive a repeat served from either prefix cache (the hybrid cache's
// snapshot restore, and the original cache's zero-suffix route that samples from the cached tail
// hidden), and cost nothing on requests that name no tokens.
// Requires NINFER_TEST_ARTIFACT.

#include "ninfer/engine.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr std::uint32_t kMaxContext = 2048;
constexpr auto kKvStorage           = ninfer::KvCacheStorage::BFloat16;
// BF16 logits quantize near |logit| ~ 30 in steps of 0.125; two routes over the same weights
// differ by accumulation order only.
constexpr float kOracleTolerance = 0.2F;

int fail(const std::string& message) {
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

ninfer::RequestOptions readout_request(std::vector<ninfer::TokenId> tokens,
                                       std::uint32_t outputs = 1) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = outputs;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.allow_prefix_reuse      = true;
    options.execution.readout_tokens          = std::move(tokens);
    options.stop.include_model_defaults       = false;
    return options;
}

std::vector<ninfer::TokenId> prompt_tokens(ninfer::Engine& engine) {
    std::string text;
    const std::string paragraph =
        "Ticket: my payouts have failed for three days and support has not answered.\n"
        "Which queue should it go to? Options: A) payouts B) billing C) fraud. Answer: ";
    std::vector<ninfer::TokenId> tokens;
    while (tokens.size() < 700) {
        text += paragraph;
        tokens = engine.tokenize_text(text);
    }
    return tokens;
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "SKIP: NINFER_TEST_ARTIFACT is not set\n";
        return 77;
    }

    std::vector<ninfer::TokenId> prompt;
    std::vector<ninfer::TokenId> candidates;
    std::vector<std::pair<std::string, ninfer::PromptReadout>> readouts;
    // Hybrid keeps at least one prompt token to prefill: with 256-token chunks and ladder the
    // repeat restores a ladder snapshot at a chunk boundary and prefills the rest. The original cache claims the whole
    // prompt and samples from the cached tail hidden (the zero-suffix route).
    // Only a prompt prefilled in one pass, as the oracle scores it, is compared with the oracle:
    // chunked prefill sums in another order and on this repetitive prompt moved the candidates'
    // log-probabilities by up to 1.6 (0.07-0.22 on the top token) at every chunk size tried
    // (128-640, with and without the grouped small-pass route), identically in both cache modes.
    // The readout itself is exact against a one-pass prefill, so the chunked configuration checks
    // that a restored repeat reads what its own cold run read.
    struct Configuration {
        ninfer::ContextCacheMode mode;
        std::uint32_t prefill_chunk;
    };
    for (const Configuration configuration :
         {Configuration{ninfer::ContextCacheMode::Original, 1024},
          Configuration{ninfer::ContextCacheMode::Hybrid, 256}}) {
        const bool hybrid      = configuration.mode == ninfer::ContextCacheMode::Hybrid;
        const std::string name = std::string(hybrid ? "hybrid" : "original") + " chunk " +
                                 std::to_string(configuration.prefill_chunk);
        ninfer::EngineOptions options;
        options.artifact_path      = artifact;
        options.max_context        = kMaxContext;
        options.kv_capacity        = ninfer::KvCapacityPolicy::explicit_capacity(
            (hybrid ? 4U : 2U) * kMaxContext);
        options.kv_cache           = kKvStorage;
        options.max_concurrency    = 2;
        options.prefill_chunk      = configuration.prefill_chunk;
        options.context_cache.mode = configuration.mode;
        if (hybrid) {
            options.context_cache.hybrid.tap_ladder_tokens  = 256;
            options.context_cache.hybrid.tap_min_gap_tokens = 256;
        }
        ninfer::Engine engine(options);

        if (prompt.empty()) {
            prompt = prompt_tokens(engine);
            for (const char* label : {"A", "B", "C", "a", "0", " A", "payouts", "\n"}) {
                const std::vector<ninfer::TokenId> ids = engine.tokenize_text(label);
                if (ids.size() == 1 &&
                    std::find(candidates.begin(), candidates.end(), ids[0]) == candidates.end()) {
                    candidates.push_back(ids[0]);
                }
            }
            if (candidates.size() < 4) { return fail("the fixture labels did not tokenize"); }
        }

        const auto run = [&](ninfer::RequestOptions request) {
            return engine.generate(engine.prepare_tokens(prompt), std::move(request));
        };
        ninfer::GenerationResult cold = run(readout_request(candidates));
        if (!cold.readout || cold.readout->logprobs.size() != candidates.size() ||
            cold.generated_token_ids.size() != 1) {
            return fail(name + ": a readout request returned no readout or the wrong output count");
        }
        const ninfer::PromptReadout& readout = *cold.readout;
        for (std::size_t i = 0; i < candidates.size(); ++i) {
            const float value = readout.logprobs[i];
            if (!std::isfinite(value) || value > 0.0F || value > readout.top_logprob) {
                return fail(name + ": a readout log-probability is invalid or above the top token's");
            }
            if (candidates[i] == readout.top_token && value != readout.top_logprob) {
                return fail(name + ": the top token read as a candidate disagrees with the top readout");
            }
        }
        if (readout.top_token != cold.generated_token_ids[0]) {
            return fail(name + ": the greedy token is not the readout's top token");
        }
        if (configuration.prefill_chunk >= prompt.size()) { readouts.emplace_back(name, readout); }

        ninfer::GenerationResult warm = run(readout_request(candidates));
        std::cout << name << ": repeat reused " << warm.reused_prompt_tokens << " of "
                  << prompt.size() << " prompt tokens\n";
        if (!warm.readout || warm.reused_prompt_tokens == 0 ||
            (!hybrid && warm.reused_prompt_tokens != prompt.size())) {
            return fail(name + ": the repeated readout did not reuse the cached prompt");
        }
        for (std::size_t i = 0; i < candidates.size(); ++i) {
            if (std::fabs(warm.readout->logprobs[i] - readout.logprobs[i]) > kOracleTolerance) {
                return fail(name + ": the cached-prefix readout disagrees with the cold readout");
            }
        }

        ninfer::RequestOptions plain = readout_request({}, 4);
        if (run(std::move(plain)).readout) {
            return fail(name + ": a request that named no readout tokens returned a readout");
        }

        const auto rejects = [&](ninfer::RequestOptions invalid) {
            try {
                (void)run(std::move(invalid));
            } catch (const std::invalid_argument&) { return true; }
            return false;
        };
        std::vector<ninfer::TokenId> wide(ninfer::kMaximumReadoutTokens + 1);
        for (std::size_t i = 0; i < wide.size(); ++i) {
            wide[i] = static_cast<ninfer::TokenId>(i);
        }
        if (!rejects(readout_request(wide)) ||
            !rejects(readout_request({candidates[0], candidates[0]})) ||
            !rejects(readout_request(candidates, 0)) || !rejects(readout_request({-1}))) {
            return fail(name + ": an invalid readout request was accepted");
        }
    }

    // The oracle: CausalScoring scores the token after the prompt over the same weights.
    ninfer::EngineOptions scoring;
    scoring.artifact_path = artifact;
    scoring.purpose       = ninfer::EnginePurpose::CausalScoring;
    scoring.max_context   = kMaxContext;
    scoring.kv_cache      = kKvStorage;
    ninfer::Engine oracle(scoring);
    std::vector<ninfer::TokenId> scored = prompt;
    scored.push_back(readouts.front().second.top_token);
    const auto count = static_cast<std::uint32_t>(candidates.size());
    const ninfer::ScoreResult expected = oracle.score_tokens(
        scored, static_cast<std::uint32_t>(prompt.size()),
        {.top_k = 1, .candidates_per_position = count, .candidates = candidates});
    int failed = 0;
    for (const auto& [name, readout] : readouts) {
        float worst = 0.0F;
        std::cout << name << " readout vs CausalScoring:";
        for (std::size_t i = 0; i < candidates.size(); ++i) {
            const float delta = readout.logprobs[i] - expected.candidate_logprobs[i];
            worst = std::max(worst, std::fabs(delta));
            std::cout << ' ' << candidates[i] << ':' << readout.logprobs[i] << '/'
                      << expected.candidate_logprobs[i];
        }
        std::cout << "\n  worst |d logprob| = " << worst << ", top " << readout.top_token << ':'
                  << readout.top_logprob << " vs " << expected.top_ids[0] << ':'
                  << expected.top_logprobs[0] << '\n';
        if (worst > kOracleTolerance) {
            failed += fail(name + ": the readout disagrees with CausalScoring");
        }
        if (readout.top_token != expected.top_ids[0] ||
            std::fabs(expected.top_logprobs[0] - readout.top_logprob) > kOracleTolerance) {
            failed += fail(name + ": the readout's top token disagrees with CausalScoring");
        }
    }
    if (failed != 0) { return 1; }
    std::cout << "OK readout_real\n";
    return 0;
}
