// A long prompt's layer walk interleaved with another lane's decode rounds, on the real artifact
// with SSD-only experts (GitHub issue #5): the walk's steps run while the other lane decodes, and
// each decode round closes the SSD tier's round when it settles. Every walk step must run inside an
// open tier round (its fetch requests are answered only then), so both requests complete with the
// greedy ids each gives alone, and no call reports a silent host.
// Skips (77) unless INFERNIX_QWEN4_ARTIFACT names a Qwen4Exp artifact.
//
//   INFERNIX_QWEN4_ARTIFACT=out.ninfer [INFERNIX_QWEN4_NGRAM=out.ninfer.ngram] infernix_qwen4_exp_walk_interleave_real_test

#include "infernix/engine.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

int failures = 0;

void check(bool ok, const std::string& what) {
    std::printf("%s: %s\n", ok ? "ok" : "FAIL", what.c_str());
    if (!ok) { ++failures; }
}

infernix::PromptInput story() {
    infernix::PromptInput input;
    infernix::ChatMessage message;
    message.role = infernix::ChatRole::User;
    message.parts.push_back({.kind = infernix::MessagePartKind::Text,
                             .text = "Write a long story about a lighthouse keeper who befriends a seagull."});
    input.messages.push_back(std::move(message));
    input.options.enable_thinking = false;
    return input;
}

// Synthetic field reports: varied enough that the prompt is not one repeated n-gram.
std::string reports(std::size_t count) {
    static const char* const kSites[]   = {"north ridge", "river delta", "salt flats", "old quarry", "pine basin",
                                           "coastal shelf", "east meadow", "glacier tongue"};
    static const char* const kEvents[]  = {"a rise in water level", "unusual bird activity", "a minor rockfall",
                                           "steady wind from the west", "a drop in soil moisture", "fresh animal tracks",
                                           "fog that lifted by noon", "a broken fence post"};
    static const char* const kActions[] = {"logged it for review", "replaced the sensor", "photographed the area",
                                           "notified the district office", "took three samples", "marked the trail"};
    std::uint32_t state = 12345;
    const auto next     = [&] { return state = state * 1664525U + 1013904223U; };
    std::string out;
    for (std::size_t i = 0; i < count; ++i) {
        out += "Report " + std::to_string(i + 1) + ": at the " + kSites[next() % 8] + " the team observed " +
               kEvents[next() % 8] + " at " + std::to_string(next() % 24) + ":" + std::to_string(10 + next() % 50) +
               ", measured " + std::to_string(next() % 1000) + " units, and " + kActions[next() % 6] + ".\n";
    }
    return out + "Question: which site was mentioned most often?\nAnswer:";
}

infernix::RequestOptions greedy(std::uint32_t tokens) {
    infernix::RequestOptions request;
    request.execution.requested_output_tokens = tokens;
    request.execution.sampling.temperature    = 0.0F;
    return request;
}

// When a request's generated tokens were committed (its timing observations, published whether or
// not the tokens have visible text yet).
class Timeline final : public infernix::OutputSink {
public:
    void start(infernix::GenerationStart) override {}
    void progress(infernix::PromptProgress) override {}
    void timing(infernix::GenerationTimingObservation timing) override {
        const std::lock_guard<std::mutex> lock(mutex_);
        if (timing.generated_tokens > generated_) {
            generated_ = timing.generated_tokens;
            times_.push_back(Clock::now());
        }
    }
    void publish(infernix::OutputDelta) override {}
    [[nodiscard]] std::uint32_t generated() const {
        const std::lock_guard<std::mutex> lock(mutex_);
        return generated_;
    }
    // When the first token was committed (the prompt's prefill had ended).
    [[nodiscard]] std::optional<Clock::time_point> first() const {
        const std::lock_guard<std::mutex> lock(mutex_);
        return times_.empty() ? std::nullopt : std::optional<Clock::time_point>(times_.front());
    }
    // Commit boundaries in [from, to).
    [[nodiscard]] std::size_t between(Clock::time_point from, Clock::time_point to) const {
        const std::lock_guard<std::mutex> lock(mutex_);
        std::size_t n = 0;
        for (const auto t : times_) { n += t >= from && t < to ? 1 : 0; }
        return n;
    }

private:
    mutable std::mutex mutex_;
    std::vector<Clock::time_point> times_;
    std::uint32_t generated_ = 0;
};

// A greedy request streamed into `line` with live timings.
std::vector<infernix::TokenId> observed(infernix::Engine& engine, infernix::PreparedPrompt prompt,
                                        std::uint32_t tokens, Timeline& line) {
    infernix::GenerationObservationOptions observation;
    observation.live_timings = true;
    return engine.submit(std::move(prompt), greedy(tokens), infernix::OutputConsumerMode::Streaming, observation)
        .wait(&line)
        .generated_token_ids;
}

} // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const char* artifact = std::getenv("INFERNIX_QWEN4_ARTIFACT");
    if (artifact == nullptr) {
        std::printf("SKIP: set INFERNIX_QWEN4_ARTIFACT\n");
        return 77;
    }
    const char* ngram = std::getenv("INFERNIX_QWEN4_NGRAM");
    try {
        std::mutex diagnostics_mutex;
        std::vector<std::string> diagnostics;
        infernix::EngineOptions options;
        options.artifact_path = artifact;
        if (ngram != nullptr) { options.ngram_volume_path = ngram; }
        options.max_context     = 12288;
        options.max_concurrency = 2;
        options.prefill_chunk   = 1024; // the long prompt walks a span of several calls, step by step
        options.kv_cache        = infernix::KvCacheStorage::Int8Group64;
        options.kv_capacity     = infernix::KvCapacityPolicy::explicit_capacity(24576);
        // Fewer experts than the artifact holds stay in RAM: the rest are read from the SSD tier,
        // and the walk's calls publish fetch requests for them.
        options.expert_ram_bytes      = 24ULL << 30;
        options.context_cache.enabled = false; // every prompt prefills in full, the walk included
        options.diagnostic_observer.callback = [&](const infernix::Diagnostic& d) {
            const std::lock_guard<std::mutex> lock(diagnostics_mutex);
            diagnostics.push_back(d.message);
            std::printf("  [engine] %s\n", d.message.c_str());
        };
        infernix::Engine engine(options);

        const std::vector<infernix::TokenId> long_prompt = engine.tokenize_text(reports(260));
        std::printf("long prompt: %zu tokens\n", long_prompt.size());
        check(long_prompt.size() >= 6 * options.prefill_chunk, "the long prompt spans several prefill calls");
        constexpr std::uint32_t kStoryTokens = 400, kAnswerTokens = 24;

        // The short request decodes; once it has committed a few tokens the long prompt is admitted
        // beside it and walks while the decode rounds continue.
        Timeline story_line, answer_line;
        std::optional<std::vector<infernix::TokenId>> story_ids, answer_ids;
        std::string story_error, answer_error;
        std::thread decoder([&] {
            try {
                story_ids = observed(engine, engine.prepare(story()), kStoryTokens, story_line);
            } catch (const std::exception& error) { story_error = error.what(); }
        });
        while (story_line.generated() < 8 && story_error.empty()) { std::this_thread::sleep_for(std::chrono::milliseconds(5)); }
        const auto admitted = Clock::now();
        try {
            answer_ids = observed(engine, engine.prepare_tokens(long_prompt), kAnswerTokens, answer_line);
        } catch (const std::exception& error) { answer_error = error.what(); }
        decoder.join();
        check(story_error.empty(), "the decoding request completed" + (story_error.empty() ? "" : ": " + story_error));
        check(answer_error.empty(), "the walked request completed" + (answer_error.empty() ? "" : ": " + answer_error));
        const auto first = answer_line.first();
        check(first.has_value(), "the walked request committed tokens");
        if (first) {
            const std::size_t during = story_line.between(admitted, *first);
            std::printf("decode commits during the long prefill: %zu (%.1f s)\n", during,
                        std::chrono::duration<double>(*first - admitted).count());
            check(during > 0, "the other lane decoded while the long prompt prefilled");
        }
        bool silent = false;
        {
            const std::lock_guard<std::mutex> lock(diagnostics_mutex);
            for (const std::string& line : diagnostics) {
                silent = silent || line.find("stopped answering") != std::string::npos;
            }
        }
        check(!silent, "no call reported a silent host expert service");

        // Each request alone (the prefix cache is off, so both prefill in full again).
        const auto story_alone  = engine.generate(engine.prepare(story()), greedy(kStoryTokens)).generated_token_ids;
        const auto answer_alone = engine.generate(engine.prepare_tokens(long_prompt), greedy(kAnswerTokens)).generated_token_ids;
        check(story_ids && *story_ids == story_alone, "the decoding request's ids equal its solo run");
        check(answer_ids && *answer_ids == answer_alone, "the walked request's ids equal its solo run");
    } catch (const std::exception& error) {
        std::printf("FAIL: %s\n", error.what());
        return 1;
    }
    if (failures != 0) {
        std::printf("%d checks failed\n", failures);
        return 1;
    }
    std::printf("walk interleave checks passed\n");
    return 0;
}
