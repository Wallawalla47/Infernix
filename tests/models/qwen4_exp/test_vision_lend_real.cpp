// The Vision window's expert frames on the real artifact (design §19.3.2, candidates SS-01): a media
// request whose encode window the expert cache cannot lend must never fail the Engine. The window is
// lent as part of the binding: while other lanes hold the frames the request waits, and one that cannot
// fit with every other lane idle is refused alone (media_budget_exceeded). A test hook
// (qwen4_exp::testing::set_vision_lend_limit) makes the cache that small:
//
//   refused       The window cannot be lent on an idle Engine: the image request fails with
//                 MediaBudgetExceeded, and a text request then equals its solo run.
//   deferred      The same while a long text request holds the other lane: the image request waits for it
//                 and is refused once it ends; the text request completes with its solo ids.
//   contention    No limit: a long text prompt prefills (its stream leases taking frames) while an image
//                 request arrives on the other lane; both complete with their solo ids.
//
// The artifact must carry the vision tower (Dense8). Reproducible route, prefix reuse off (every image
// request encodes), two lanes. Skips (77) unless INFERNIX_QWEN4_ARTIFACT names a Qwen4Exp artifact
// (INFERNIX_QWEN4_NGRAM: its n-gram volume).
//
//   INFERNIX_QWEN4_ARTIFACT=out.ninfer infernix_qwen4_exp_vision_lend_real_test

#include "models/qwen4_exp/program/vision_window.h"
#include "infernix/engine.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace {

std::vector<infernix::TokenId> synthetic_tokens(std::size_t count, std::uint32_t seed) {
    std::vector<infernix::TokenId> tokens;
    tokens.reserve(count);
    std::uint32_t state = seed * 2654435761U + 1U;
    for (std::size_t i = 0; i < count; ++i) {
        state = state * 1664525U + 1013904223U;
        tokens.push_back(static_cast<infernix::TokenId>(1000U + (state >> 8U) % 30000U));
    }
    return tokens;
}

std::vector<std::uint8_t> gradient_ppm(int size, int phase) {
    std::vector<std::uint8_t> ppm;
    const std::string header = "P6\n" + std::to_string(size) + ' ' + std::to_string(size) + "\n255\n";
    ppm.insert(ppm.end(), header.begin(), header.end());
    for (int index = 0; index < size * size; ++index) {
        ppm.push_back(static_cast<std::uint8_t>((index + phase) & 0xff));
        ppm.push_back(static_cast<std::uint8_t>(((index / size) * 3 + phase) & 0xff));
        ppm.push_back(static_cast<std::uint8_t>((index * 7 + phase * 5) & 0xff));
    }
    return ppm;
}

infernix::PromptInput image_prompt() {
    infernix::ChatMessage user;
    user.role = infernix::ChatRole::User;
    infernix::MessagePart media;
    media.kind              = infernix::MessagePartKind::Media;
    media.media.kind        = infernix::MediaKind::Image;
    media.media.bytes       = gradient_ppm(1024, 17); // a screenshot-sized item: 1,024 merged tokens
    media.media.media_type  = "image/x-portable-pixmap";
    media.media.source_name = "screenshot.ppm";
    user.parts.push_back(std::move(media));
    user.parts.push_back(infernix::MessagePart{.kind = infernix::MessagePartKind::Text,
                                               .text = "Describe the colours and layout of this image.",
                                               .media = {}});
    infernix::PromptInput input;
    input.messages.push_back(std::move(user));
    input.options.enable_thinking = false;
    return input;
}

infernix::PromptInput text_prompt() {
    infernix::ChatMessage user;
    user.role = infernix::ChatRole::User;
    user.parts.push_back(infernix::MessagePart{.kind  = infernix::MessagePartKind::Text,
                                               .text  = "Write a short story about a lighthouse keeper.",
                                               .media = {}});
    infernix::PromptInput input;
    input.messages.push_back(std::move(user));
    input.options.enable_thinking = false;
    return input;
}

infernix::RequestOptions greedy(std::uint32_t outputs) {
    infernix::RequestOptions options;
    options.execution.requested_output_tokens = outputs;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.allow_prefix_reuse      = false; // every image request encodes its image
    return options;
}

int check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "  ok  " : "  FAIL", what.c_str());
    return ok ? 0 : 1;
}

// The handle's ids, or the RequestError kind it failed with (nullopt for any other error).
struct Outcome {
    std::optional<std::vector<infernix::TokenId>> ids;
    std::optional<infernix::RequestErrorKind> error;
    std::string what;
};

Outcome outcome_of(infernix::GenerationHandle& handle) {
    Outcome out;
    try {
        out.ids = handle.wait().generated_token_ids;
    } catch (const infernix::RequestError& error) {
        out.error = error.kind();
        out.what  = error.what();
    } catch (const std::exception& error) { out.what = error.what(); }
    return out;
}

bool refused(const Outcome& outcome) {
    return !outcome.ids && outcome.error == infernix::RequestErrorKind::MediaBudgetExceeded;
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
    namespace testing = infernix::models::qwen4_exp::testing;
    int failures      = 0;
    try {
        infernix::EngineOptions options;
        options.artifact_path = artifact;
        if (ngram != nullptr) { options.ngram_volume_path = ngram; }
        options.enable_vision   = true;
        options.max_context     = 32768;
        options.max_concurrency = 2;
        // The long prompt and an image request on the other lane at once (the default is 2,048 tokens).
        options.kv_capacity     = infernix::KvCapacityPolicy::explicit_capacity(32768);
        options.kv_cache        = infernix::KvCacheStorage::Int8Group64;
        options.fast_paths      = false; // a request's ids do not depend on the other lane
        infernix::Engine engine(options);

        constexpr std::uint32_t kTokens = 24;
        const auto submit_image = [&] { return engine.submit(engine.prepare(image_prompt()), greedy(kTokens)); };
        const auto submit_text  = [&] { return engine.submit(engine.prepare(text_prompt()), greedy(kTokens)); };
        // A long prompt: several streamed prefill calls, whose leases take expert frames between them.
        const std::vector<infernix::TokenId> long_tokens = synthetic_tokens(16384, 5);
        const auto submit_long = [&] { return engine.submit(engine.prepare_tokens(long_tokens), greedy(kTokens)); };
        const auto solo = [](infernix::GenerationHandle handle) { return outcome_of(handle); };

        const Outcome image_solo = solo(submit_image());
        const Outcome text_solo  = solo(submit_text());
        const Outcome long_solo  = solo(submit_long());
        failures += check(image_solo.ids && image_solo.ids->size() == kTokens, "the image request runs solo " + image_solo.what);
        failures += check(text_solo.ids.has_value(), "the text request runs solo " + text_solo.what);
        failures += check(long_solo.ids.has_value(), "the long text request runs solo " + long_solo.what);
        if (!image_solo.ids || !text_solo.ids || !long_solo.ids) { return 1; }

        std::printf("== refused\n");
        testing::set_vision_lend_limit(1);
        const Outcome refused_alone = solo(submit_image());
        failures += check(refused(refused_alone), "an image request the cache cannot lend is refused alone (" +
                                                      refused_alone.what + ")");
        const Outcome text_after = solo(submit_text());
        failures += check(text_after.ids == text_solo.ids, "the Engine keeps serving: a text request equals its solo run");

        std::printf("== deferred\n");
        infernix::GenerationHandle long_handle  = submit_long();
        std::this_thread::sleep_for(std::chrono::milliseconds(50)); // the long request binds first
        infernix::GenerationHandle image_handle = submit_image();
        const Outcome long_beside  = outcome_of(long_handle);
        const Outcome image_beside = outcome_of(image_handle);
        failures += check(long_beside.ids == long_solo.ids, "the long request completes with its solo ids");
        failures += check(refused(image_beside), "the image request waits for the other lane, then is refused (" +
                                                     image_beside.what + ")");
        testing::set_vision_lend_limit(0);

        std::printf("== contention\n");
        bool contended = true;
        for (int round = 0; round < 3; ++round) {
            infernix::GenerationHandle text  = submit_long();
            std::this_thread::sleep_for(std::chrono::milliseconds(200 * round)); // a later point of its prefill
            infernix::GenerationHandle image = submit_image();
            const Outcome long_out  = outcome_of(text);
            const Outcome image_out = outcome_of(image);
            contended = contended && long_out.ids == long_solo.ids && image_out.ids == image_solo.ids;
            if (!image_out.ids) { std::printf("        round %d: the image request failed: %s\n", round, image_out.what.c_str()); }
        }
        failures += check(contended, "beside a long prefill both requests complete with their solo ids (3 rounds)");
        const Outcome image_after = solo(submit_image());
        failures += check(image_after.ids == image_solo.ids, "the image request again equals its solo run");
    } catch (const std::exception& error) {
        testing::set_vision_lend_limit(0);
        std::printf("FAIL: %s\n", error.what());
        return 1;
    }
    std::printf(failures == 0 ? "vision lend checks passed\n" : "FAIL: %d checks failed\n", failures);
    return failures == 0 ? 0 : 1;
}
