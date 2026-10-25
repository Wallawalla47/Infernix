#pragma once

// A resume against a cold run (candidates §15 PR6-02; Penny Royal #115: a DFlash2 that registered decode-time
// checkpoints it never wrote resumed the next turn from a stale state, and the next 16 tokens' log-probabilities
// differed from a cold recompute by 3.99 on average, 0.061 with native MTP). The existing resume tests require
// reuse to happen or compare two resumes with each other, so a stale but self-consistent state passes them.
//
// One probe on one Engine: a request decodes past `start` (its frontier is a decode-computed endpoint where the
// route publishes one: the default route's free-cache-resume; the reproducible route publishes and resumes on
// the prefill chunk grid only), then the continuation start + generated + separator runs resumed (prefix reuse)
// and cold (none), and the cold run's tokens are teacher-forced through prompt readouts both ways. With decode 0
// the state is prefill's alone: the calibration of the default route's bound.

#include "infernix/engine.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace infernix::test {

inline constexpr std::uint32_t kResumeProbeTokens = 16;

struct ResumeProbe {
    std::size_t start        = 0; // prompt tokens before the decoded ones
    std::size_t continuation = 0; // the continuation's prompt tokens
    std::size_t reused       = 0; // prompt tokens the resumed continuation reused
    std::vector<TokenId> resumed;
    std::vector<TokenId> cold;
    // log p(cold[i] | continuation + cold[0..i)) with reuse and without.
    std::vector<float> resumed_logprobs;
    std::vector<float> cold_logprobs;

    [[nodiscard]] double mean_abs_delta() const {
        double sum = 0.0;
        for (std::size_t i = 0; i < cold_logprobs.size(); ++i) {
            sum += std::abs(static_cast<double>(resumed_logprobs[i]) - static_cast<double>(cold_logprobs[i]));
        }
        return cold_logprobs.empty() ? 0.0 : sum / static_cast<double>(cold_logprobs.size());
    }

    // The same tokens and the same log-probability bits.
    [[nodiscard]] bool bit_identical() const {
        return resumed == cold && resumed_logprobs.size() == cold_logprobs.size() &&
               std::memcmp(resumed_logprobs.data(), cold_logprobs.data(), cold_logprobs.size() * sizeof(float)) == 0;
    }
};

inline RequestOptions resume_probe_request(std::uint32_t outputs, bool reuse) {
    RequestOptions options;
    options.execution.requested_output_tokens = outputs;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.sampling.seed           = 0;
    options.execution.allow_prefix_reuse      = reuse;
    options.stop.include_model_defaults       = false;
    return options;
}

inline ResumeProbe resume_probe(Engine& engine, const std::vector<TokenId>& start, std::uint32_t decode,
                                TokenId separator) {
    ResumeProbe probe;
    probe.start                       = start.size();
    std::vector<TokenId> continuation = start;
    const GenerationResult first = engine.generate(engine.prepare_tokens(start), resume_probe_request(decode == 0 ? 1 : decode, true));
    if (decode != 0) {
        if (first.generated_token_ids.size() != decode) { throw std::runtime_error("the probe's decode stopped early"); }
        continuation.insert(continuation.end(), first.generated_token_ids.begin(), first.generated_token_ids.end());
    }
    continuation.push_back(separator);
    probe.continuation = continuation.size();
    const GenerationResult resumed =
        engine.generate(engine.prepare_tokens(continuation), resume_probe_request(kResumeProbeTokens, true));
    const GenerationResult cold =
        engine.generate(engine.prepare_tokens(continuation), resume_probe_request(kResumeProbeTokens, false));
    if (cold.reused_prompt_tokens != 0) { throw std::runtime_error("the cold continuation reused a prefix"); }
    probe.reused  = resumed.reused_prompt_tokens;
    probe.resumed = resumed.generated_token_ids;
    probe.cold    = cold.generated_token_ids;
    for (std::size_t i = 0; i < probe.cold.size(); ++i) {
        std::vector<TokenId> forced = continuation;
        forced.insert(forced.end(), probe.cold.begin(), probe.cold.begin() + static_cast<std::ptrdiff_t>(i));
        for (const bool reuse : {true, false}) {
            RequestOptions options             = resume_probe_request(1, reuse);
            options.execution.readout_tokens   = {probe.cold[i]};
            const GenerationResult read        = engine.generate(engine.prepare_tokens(forced), options);
            if (!read.readout.has_value() || read.readout->logprobs.size() != 1) {
                throw std::runtime_error("a teacher-forced readout returned no log-probability");
            }
            (reuse ? probe.resumed_logprobs : probe.cold_logprobs).push_back(read.readout->logprobs.front());
        }
    }
    return probe;
}

// A long natural prompt: `sentences` numbered sentences on one theme, so decoding past it stays natural text.
inline std::string resume_probe_text(const std::string& theme, int sentences) {
    std::string text = "Write a careful, detailed essay about " + theme + ". Notes:\n";
    for (int i = 1; i <= sentences; ++i) {
        text += std::to_string(i) + ". Point " + std::to_string(i) + " about " + theme +
                " deserves its own paragraph, with an example, a counterexample and a short summary.\n";
    }
    return text;
}

} // namespace infernix::test
