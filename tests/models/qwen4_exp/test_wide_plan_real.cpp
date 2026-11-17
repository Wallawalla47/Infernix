// The wide prefill workspace against its plan on the real artifact (design §19.3.7, "the wide workspace"; candidates
// QG-01). A prefill call wider than the static arena runs in an arena lent from the expert frames. Each scenario runs
// prompts whose calls are that wide and records every wide scope (testing::record_wide_scopes): where it was entered,
// the columns it was entered with, the widest call it ran (a layer walk's MoE runs over a group of chunks), the bytes
// its arena was given, the bytes the forward's own allocation plan gives it (0 while none is computed) and the
// arena's peak. Every peak stays within what its arena was given (the arena throws past it), and within the plan
// where one is computed. One line is printed per scope, with the plan's headroom over the peak.
//
//   infernix_qwen4_exp_wide_plan_real_test SCENARIO
//   SCENARIO: banks (the experts in RAM as banks), tier (the SSD tier: INFERNIX_EXPERT_RAM_MIB, default 30000),
//             mtp, dflash2 (INFERNIX_QWEN4_DFLASH2 names the drafter companion), two-lanes (two prompts at once),
//             scoring (CausalScoring), a8 (--a8-activations), bf16p (the packed BF16 KV cache), chat (a chat prompt of
//             530-565 tokens with prefix reuse: its opener tap makes the last call a wide pair call, which must run)
// Skips (77) without INFERNIX_QWEN4_ARTIFACT (INFERNIX_QWEN4_NGRAM: its n-gram volume, default <artifact>.ngram), or
// for dflash2 without INFERNIX_QWEN4_DFLASH2.

#include "models/qwen4_exp/program/prefix/test_taps.h"
#include "infernix/engine.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace q4t = infernix::models::qwen4_exp::testing;

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

infernix::RequestOptions one_token(bool reuse = false) {
    infernix::RequestOptions options;
    options.execution.requested_output_tokens = 1;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.allow_prefix_reuse      = reuse; // without it every call is computed, none resumed
    options.stop.include_model_defaults       = false;
    return options;
}

// A chat prompt of one user message whose rendered tokens number from `low` to `high`: its generation opener
// tap cuts the last call into the pair call's head and tail (a wide one past the static arena's 512 columns,
// below the 576 a call streams from on FP4-Local and IQ3_S).
infernix::PromptInput chat_prompt(infernix::Engine& engine, std::size_t low, std::size_t high) {
    infernix::PromptInput input;
    input.options.enable_thinking = false;
    std::string text;
    for (int i = 0; i < 400; ++i) {
        text += "Box " + std::to_string(i) + " goes to shelf " + std::to_string((i * 7) % 13) + ". ";
        input.messages.clear();
        infernix::ChatMessage user;
        user.role = infernix::ChatRole::User;
        user.parts.push_back(infernix::MessagePart{.kind = infernix::MessagePartKind::Text, .text = text});
        input.messages.push_back(user);
        infernix::PromptInput probe = input;
        const std::size_t tokens    = engine.prepare(std::move(probe)).token_ids().size();
        if (tokens >= low && tokens <= high) { return input; }
        if (tokens > high) { break; }
    }
    throw std::runtime_error("no chat prompt rendered to " + std::to_string(low) + ".." + std::to_string(high) + " tokens");
}

const char* site_name(q4t::WideSite site) {
    switch (site) {
        case q4t::WideSite::Call: return "call";
        case q4t::WideSite::WalkSpan: return "walk-span";
        case q4t::WideSite::WalkStep: return "walk-step";
        case q4t::WideSite::PairCall: return "pair";
    }
    return "?";
}

// Prints every recorded scope and checks it; returns the failures.
int check_scopes(const std::string& scenario) {
    const std::vector<q4t::WideScope> scopes = q4t::wide_scopes();
    int failures = 0;
    if (scopes.empty()) {
        std::printf("FAIL %s: no wide scope was recorded (no call ran wider than the static arena)\n", scenario.c_str());
        return 1;
    }
    std::uint64_t widest_peak = 0, widest_given = 0, widest_planned = 0;
    for (const q4t::WideScope& s : scopes) {
        const auto mib = [](std::uint64_t bytes) { return static_cast<double>(bytes) / (1 << 20); };
        std::printf("  %-9s %6d columns (widest call %6d): given %8.1f MiB, planned %8.1f MiB, peak %8.1f MiB", site_name(s.site),
                    s.columns, s.inner, mib(s.given), mib(s.planned), mib(s.peak));
        if (s.peak > 0) {
            std::printf(", given/peak %.2f", static_cast<double>(s.given) / static_cast<double>(s.peak));
            if (s.planned > 0) { std::printf(", planned/peak %.2f", static_cast<double>(s.planned) / static_cast<double>(s.peak)); }
        }
        std::printf("\n");
        if (s.peak > s.given) {
            std::printf("FAIL %s: a %d-column scope's peak exceeds its arena\n", scenario.c_str(), s.columns);
            ++failures;
        }
        if (s.planned > 0 && s.peak > s.planned) {
            std::printf("FAIL %s: a %d-column scope's peak exceeds the forward's plan\n", scenario.c_str(), s.columns);
            ++failures;
        }
        if (s.peak > widest_peak) {
            widest_peak    = s.peak;
            widest_given   = s.given;
            widest_planned = s.planned;
        }
    }
    std::printf("%s: %zu wide scopes; the heaviest peaked at %.1f MiB of %.1f MiB given (%.2fx)", scenario.c_str(), scopes.size(),
                static_cast<double>(widest_peak) / (1 << 20), static_cast<double>(widest_given) / (1 << 20),
                widest_peak > 0 ? static_cast<double>(widest_given) / static_cast<double>(widest_peak) : 0.0);
    if (widest_planned > 0) {
        std::printf(", planned %.1f MiB (%.2fx)", static_cast<double>(widest_planned) / (1 << 20),
                    static_cast<double>(widest_planned) / static_cast<double>(widest_peak));
    }
    std::printf("\n");
    return failures;
}

} // namespace

int main(int argc, char** argv) {
    const char* artifact = std::getenv("INFERNIX_QWEN4_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::printf("skip: INFERNIX_QWEN4_ARTIFACT is not set\n");
        return 77;
    }
    const std::string scenario = argc > 1 ? argv[1] : "banks";
    constexpr std::string_view kScenarios[] = {"banks", "tier", "mtp", "dflash2", "two-lanes", "scoring", "a8", "bf16p", "chat"};
    if (std::find(std::begin(kScenarios), std::end(kScenarios), scenario) == std::end(kScenarios)) {
        std::printf("unknown scenario %s\n", scenario.c_str());
        return 1;
    }
    const char* drafter = std::getenv("INFERNIX_QWEN4_DFLASH2");
    if (scenario == "dflash2" && (drafter == nullptr || *drafter == '\0')) {
        std::printf("skip: INFERNIX_QWEN4_DFLASH2 is not set\n");
        return 77;
    }
    try {
        // Prompt lengths: one call under the chunk, one near it, one past it (a full chunk, then the rest).
        const std::vector<std::size_t> lengths = {3000, 9000, 20000};
        infernix::EngineOptions options;
        options.artifact_path = artifact;
        if (const char* ngram = std::getenv("INFERNIX_QWEN4_NGRAM")) { options.ngram_volume_path = ngram; }
        options.max_context      = 32768;
        options.prefill_chunk    = 16384;
        options.kv_capacity      = infernix::KvCapacityPolicy::automatic();
        options.kv_cache         = infernix::KvCacheStorage::Int8Group64;
        options.max_concurrency  = scenario == "two-lanes" ? 2U : 1U;
        options.max_pending_requests = 4;
        if (scenario == "tier") {
            const char* mib         = std::getenv("INFERNIX_EXPERT_RAM_MIB");
            options.expert_ram_bytes = (mib != nullptr ? std::stoull(mib) : 30000ULL) << 20;
        }
        if (scenario == "mtp") {
            options.speculative.backend      = infernix::SpeculativeBackend::Mtp;
            options.speculative.draft_tokens = 3;
        }
        if (scenario == "dflash2") {
            options.speculative.backend      = infernix::SpeculativeBackend::DFlash2;
            options.speculative.draft_tokens = 3;
            options.speculative.draft_model  = drafter;
        }
        if (scenario == "a8") { options.a8_activations = true; }
        if (scenario == "bf16p") { options.kv_cache = infernix::KvCacheStorage::Bf16Packed; }
        if (scenario == "scoring") {
            options.purpose       = infernix::EnginePurpose::CausalScoring;
            options.max_context   = 9000;
            options.prefill_chunk = 9000;
        }
        infernix::Engine engine(options);
        q4t::record_wide_scopes(true);
        if (scenario == "scoring") {
            (void)engine.score_tokens(synthetic_tokens(9000, 5), 1, infernix::ScoreOptions{});
        } else if (scenario == "chat") {
            q4t::record_wide_scopes(false);
            infernix::PromptInput prompt = chat_prompt(engine, 530, 565);
            q4t::record_wide_scopes(true);
            (void)engine.generate(engine.prepare(std::move(prompt)), one_token(true));
        } else if (scenario == "two-lanes") {
            auto a = engine.submit(engine.prepare_tokens(synthetic_tokens(9000, 1)), one_token());
            auto b = engine.submit(engine.prepare_tokens(synthetic_tokens(9000, 2)), one_token());
            (void)a.wait();
            (void)b.wait();
        } else {
            std::uint32_t seed = 10;
            for (const std::size_t length : lengths) {
                (void)engine.generate(engine.prepare_tokens(synthetic_tokens(length, seed++)), one_token());
            }
        }
        q4t::record_wide_scopes(false);
        std::printf("== %s\n", scenario.c_str());
        int failures = check_scopes(scenario);
        if (scenario == "chat") {
            const std::vector<q4t::WideScope> scopes = q4t::wide_scopes();
            if (std::none_of(scopes.begin(), scopes.end(), [](const q4t::WideScope& s) { return s.site == q4t::WideSite::PairCall; })) {
                std::printf("FAIL chat: the prompt ran no wide pair call\n");
                ++failures;
            }
        }
        if (failures == 0) { std::printf("wide plan checks passed\n"); }
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        q4t::record_wide_scopes(false);
        std::printf("FAIL: %s\n", error.what());
        return 1;
    }
}
