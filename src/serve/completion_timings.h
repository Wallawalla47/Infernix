#pragma once

// Completion timings shared by the response adapters. The `timings` object mirrors the
// llama-server shape; prompt counts are reported net of the prefix cache hit (cache_n),
// so the prompt rate measures only the uncached prompt work.

#include "serve/generation_service.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace infernix::serve {

struct CompletionTimings {
    std::uint32_t cache_n          = 0;
    std::uint32_t prompt_n         = 0;
    double prompt_ms               = 0.0;
    double prompt_per_token_ms     = 0.0;
    double prompt_per_second       = 0.0;
    std::uint32_t predicted_n      = 0;
    double predicted_ms            = 0.0;
    double predicted_per_token_ms  = 0.0;
    double predicted_per_second    = 0.0;
    std::uint64_t draft_n          = 0;
    std::uint64_t draft_n_accepted = 0;
};

inline double finite_nonnegative(double value) {
    return std::isfinite(value) && value > 0.0 ? value : 0.0;
}

inline double milliseconds_per_token(std::uint32_t tokens, double milliseconds) {
    return tokens == 0 ? 0.0 : milliseconds / static_cast<double>(tokens);
}

inline double tokens_per_second(std::uint32_t tokens, double milliseconds) {
    return milliseconds > 0.0 ? 1000.0 * static_cast<double>(tokens) / milliseconds : 0.0;
}

inline CompletionTimings make_completion_timings(std::uint32_t prompt_tokens, std::uint32_t cached_tokens,
                                                 std::uint32_t generated_tokens, double prompt_ms,
                                                 double generation_ms, std::uint64_t draft_tokens = 0,
                                                 std::uint64_t accepted_draft_tokens = 0) {
    CompletionTimings timings;
    timings.cache_n             = std::min(cached_tokens, prompt_tokens);
    timings.prompt_n            = prompt_tokens - timings.cache_n;
    timings.prompt_ms           = finite_nonnegative(prompt_ms);
    timings.prompt_per_token_ms = milliseconds_per_token(timings.prompt_n, timings.prompt_ms);
    timings.prompt_per_second   = tokens_per_second(timings.prompt_n, timings.prompt_ms);

    timings.predicted_n                  = generated_tokens;
    timings.predicted_ms                 = finite_nonnegative(generation_ms);
    const std::uint32_t decode_intervals = generated_tokens > 0 ? generated_tokens - 1 : 0;
    timings.predicted_per_token_ms = milliseconds_per_token(decode_intervals, timings.predicted_ms);
    timings.predicted_per_second   = tokens_per_second(decode_intervals, timings.predicted_ms);
    timings.draft_n                = draft_tokens;
    timings.draft_n_accepted       = accepted_draft_tokens;
    return timings;
}

inline nlohmann::json timings_json(const CompletionTimings& timings) {
    nlohmann::json output = {{"cache_n", timings.cache_n},
                             {"prompt_n", timings.prompt_n},
                             {"prompt_ms", timings.prompt_ms},
                             {"prompt_per_token_ms", timings.prompt_per_token_ms},
                             {"prompt_per_second", timings.prompt_per_second},
                             {"predicted_n", timings.predicted_n},
                             {"predicted_ms", timings.predicted_ms},
                             {"predicted_per_token_ms", timings.predicted_per_token_ms},
                             {"predicted_per_second", timings.predicted_per_second}};
    if (timings.draft_n != 0) {
        output["draft_n"]          = timings.draft_n;
        output["draft_n_accepted"] = timings.draft_n_accepted;
    }
    return output;
}

inline CompletionTimings outcome_timings(const GenerationOutcome& outcome) {
    return make_completion_timings(
        static_cast<std::uint32_t>(std::max(0, outcome.prompt_tokens)),
        outcome.metrics.prefix_cache_hit_tokens,
        static_cast<std::uint32_t>(std::max(0, outcome.completion_tokens)),
        outcome.metrics.prompt_wall_seconds * 1000.0,
        outcome.metrics.generation_wall_seconds * 1000.0, outcome.metrics.speculative_draft_tokens,
        outcome.metrics.speculative_accepted_tokens);
}

} // namespace infernix::serve
