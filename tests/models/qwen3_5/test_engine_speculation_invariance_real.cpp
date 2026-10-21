#include "infernix/engine.h"

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

// Speculation invariance of the reproducible route on a real Qwen3.5 artifact (owner's decision of
// 2026-10-21; docs/maintainer/model-quality.md 6.1): with fast paths off, a request's tokens are the
// same with and without speculation, greedy and sampled. Greedy verification compares each draft
// with the verify column's argmax, which equals the decode column's; sampled requests draw by
// Gumbel-max keyed by (seed, position, token) (ops::SamplingConfig::gumbel_max), so a verify column
// returns the token plain sampling draws there.
//
// INFERNIX_TEST_ARTIFACT names the artifact; the arguments name one or more drafters (mtp, dflash,
// dflash2 or dflash2-ngram: DFlash2 with n-gram copy proposals); the artifact must hold them. One Engine
// decodes plainly, then one per drafter decodes the same requests (one plain run serves every drafter).
namespace {

constexpr std::uint32_t kOutputTokens = 200;
constexpr std::uint32_t kContext      = 4096;

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

infernix::PromptInput prompt(const std::string& text) {
    infernix::PromptInput input;
    infernix::ChatMessage user;
    user.role = infernix::ChatRole::User;
    user.parts.push_back(infernix::MessagePart{.kind = infernix::MessagePartKind::Text, .text = text, .media = {}});
    input.messages.push_back(std::move(user));
    input.options.enable_thinking   = false;
    input.options.preserve_thinking = true;
    return input;
}

infernix::RequestOptions request(float temperature, std::uint64_t seed) {
    infernix::RequestOptions options;
    options.execution.requested_output_tokens = kOutputTokens;
    options.execution.sampling.temperature    = temperature;
    options.execution.sampling.top_k          = 20;
    options.execution.sampling.top_p          = 0.95F;
    options.execution.sampling.seed           = seed;
    options.stop.include_model_defaults       = false;
    options.output.raw                        = true;
    return options;
}

infernix::EngineOptions reproducible_options(const std::filesystem::path& artifact, std::string_view drafter) {
    infernix::EngineOptions options;
    options.artifact_path   = artifact;
    options.fast_paths      = false; // the reproducible route
    options.max_context     = kContext;
    options.kv_capacity     = infernix::KvCapacityPolicy::explicit_capacity(kContext);
    options.max_concurrency = 1;
    options.prefill_chunk   = 1024;
    options.enable_vision   = false;
    if (drafter == "mtp") {
        options.speculative.backend      = infernix::SpeculativeBackend::Mtp;
        options.speculative.draft_tokens = 3;
    } else if (drafter == "dflash") {
        options.speculative.backend      = infernix::SpeculativeBackend::DFlash;
        options.speculative.draft_tokens = 7;
    } else if (drafter == "dflash2" || drafter == "dflash2-ngram") {
        options.speculative.backend      = infernix::SpeculativeBackend::DFlash2;
        options.speculative.draft_tokens = 7;
        if (drafter == "dflash2-ngram") { options.speculative.ngram_draft_tokens = 16; }
    } else if (drafter != "none") {
        throw std::invalid_argument("the drafter must be mtp, dflash, dflash2 or dflash2-ngram");
    }
    return options;
}

std::vector<std::string> prompts() {
    return {
        "Write a short story about a lighthouse keeper who finds a message in a bottle.",
        "Explain, step by step, how a compiler turns source code into machine instructions.",
        // Repetitive text, so n-gram copy proposals are accepted often.
        "Repeat the following list three times, exactly: apples, pears, plums, cherries, grapes, "
        "melons, lemons, limes, oranges, peaches.",
    };
}

std::vector<std::vector<infernix::TokenId>> decode(const std::filesystem::path& artifact, std::string_view drafter) {
    infernix::Engine engine(reproducible_options(artifact, drafter));
    std::vector<std::vector<infernix::TokenId>> out;
    for (const auto& text : prompts()) {
        out.push_back(engine.generate(engine.prepare(prompt(text)), request(0.0F, 0)).generated_token_ids);
        for (std::uint64_t seed : {11ULL, 12ULL}) {
            out.push_back(engine.generate(engine.prepare(prompt(text)), request(1.0F, seed)).generated_token_ids);
        }
    }
    return out;
}

} // namespace

int main(int argc, char** argv) {
    const char* artifact = std::getenv("INFERNIX_TEST_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "SKIP: set INFERNIX_TEST_ARTIFACT to an explicit Qwen3.5 artifact\n";
        return 77;
    }
    try {
        require(argc >= 2,
                "usage: infernix_qwen3_5_speculation_invariance_real_test mtp|dflash|dflash2|dflash2-ngram...");
        // Plain decoding once, then each drafter against it.
        const auto plain = decode(artifact, "none");
        int failed_drafters = 0;
        for (int a = 1; a < argc; ++a) {
            const std::string_view drafter = argv[a];
            const auto speculative         = decode(artifact, drafter);
            require(plain.size() == speculative.size(), "result counts differ");
            int failures = 0;
            for (std::size_t i = 0; i < plain.size(); ++i) {
                const char* kind = i % 3 == 0 ? "greedy" : "sampled";
                std::size_t same = 0;
                while (same < plain[i].size() && same < speculative[i].size() && plain[i][same] == speculative[i][same]) {
                    ++same;
                }
                const bool ok = !plain[i].empty() && plain[i] == speculative[i];
                std::cout << drafter << " prompt " << i / 3 << " " << kind << ": " << (ok ? "identical" : "DIFFERS")
                          << " (" << same << " of " << plain[i].size() << "/" << speculative[i].size()
                          << " tokens agree)\n";
                failures += ok ? 0 : 1;
            }
            if (failures != 0) {
                std::cerr << "FAIL " << drafter << ": " << failures << " requests differ from plain decoding\n";
                ++failed_drafters;
            } else {
                std::cout << "PASS " << drafter << ": speculative decoding returns plain decoding's tokens\n";
            }
        }
        return failed_drafters == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "FAIL " << error.what() << "\n";
        return 1;
    }
}
