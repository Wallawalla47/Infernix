#include "real_test_artifact.h"
#include "infernix/engine.h"

#include <cstdint>
#include <cstdlib>
#include <deque>
#include <exception>
#include <filesystem>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// Speculation invariance of the reproducible route on a real Qwen3.5 artifact (owner's decision of
// 2026-10-21; docs/maintainer/model-quality.md 6.1): with fast paths off, a request's tokens are the
// same with and without speculation, greedy and sampled. Greedy verification compares each draft
// with the verify column's argmax, which equals the decode column's; sampled requests draw by
// Gumbel-max keyed by (seed, position, token) (ops::SamplingConfig::gumbel_max), so a verify column
// returns the token plain sampling draws there.
//
// INFERNIX_TEST_ARTIFACT names the artifact; the arguments name one or more drafters (mtp, mtp-cut: MTP
// with --spec-min-p 0.5, mtp-sampled: MTP with the sampled-mtp-drafts candidate, dflash, dflash2,
// dflash2-ngram: DFlash2 with n-gram copy proposals, or dflash2-ngramN: copies of N drafts at the
// shortest minimum match, e.g. 14, 15, 16, 62, 63 for the verify widths around the GDN record routes'
// boundaries); the artifact must hold them. One Engine
// decodes plainly, then one per drafter decodes the same requests (one plain run serves every drafter).
// The requests include a raw one-token prompt, which must also verify drafts. The mtp, dflash2 and
// dspine and dspine-ngram: the DSpine drafter artifact that INFERNIX_TEST_DSPINE_DRAFTER names (a drafter
// for the artifact's target), 15 drafts a round, the second with n-gram copies of 16. The mtp, dflash2,
// dflash2-ngram, dspine and dspine-ngram drafters also decode every prompt with presence penalty 1.5 (the
// non-thinking default
// of every Qwen3.5-family model) and frequency penalty 0.3, greedy and sampled: a verified draft
// changes the next position's penalties, which speculation must apply as plain decoding does (SGLang
// #46752 shipped DFlash outputs that ignored the penalty).
//
// The plain run and the drafters without a fixed copy width run their requests together on kLanes lanes. The
// reproducible route gives a request the tokens it has alone whatever runs beside it (its request invariance, which
// the invariance entry checks), so every comparison is the one-at-a-time run's; rounds of several rows still draft,
// each row as if alone (dflash.md), and each drafting arm must accept drafts. The dflash2-ngramN arms run one request
// at a time: their rounds are placed at single-row verify widths (the GDN record routes' boundaries), which rounds of
// several rows would move.
namespace {

constexpr std::uint32_t kOutputTokens = 200;
constexpr std::uint32_t kContext      = 4096;
constexpr std::uint32_t kLanes        = 4;

// One lane for the arms that place their rounds at a verify width (dflash2-ngramN), kLanes for the rest.
std::uint32_t lanes_for(std::string_view drafter) {
    return drafter.starts_with("dflash2-ngram") && drafter.size() > std::string_view("dflash2-ngram").size() ? 1U
                                                                                                           : kLanes;
}
constexpr float kPresencePenalty      = 1.5F;
constexpr float kFrequencyPenalty     = 0.3F;

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

infernix::RequestOptions request(float temperature, std::uint64_t seed, bool penalised = false) {
    infernix::RequestOptions options;
    options.execution.requested_output_tokens = kOutputTokens;
    options.execution.sampling.temperature    = temperature;
    options.execution.sampling.top_k          = 20;
    options.execution.sampling.top_p          = 0.95F;
    options.execution.sampling.seed           = seed;
    if (penalised) {
        options.execution.sampling.presence_penalty  = kPresencePenalty;
        options.execution.sampling.frequency_penalty = kFrequencyPenalty;
    }
    options.stop.include_model_defaults = false;
    options.output.raw                  = true;
    return options;
}

bool penalised_drafter(std::string_view drafter) {
    return drafter == "mtp" || drafter == "dflash2" || drafter == "dflash2-ngram" || drafter == "dspine" ||
           drafter == "dspine-ngram";
}

infernix::EngineOptions reproducible_options(const std::filesystem::path& artifact, std::string_view drafter) {
    infernix::EngineOptions options;
    options.artifact_path   = artifact;
    options.context_cache.host_capacity_bytes = infernix::test::kTestHostTierBytes;
    options.fast_paths      = false; // the reproducible route
    options.max_context     = kContext;
    options.kv_capacity     = infernix::KvCapacityPolicy::explicit_capacity(kContext * lanes_for(drafter));
    options.max_concurrency = lanes_for(drafter);
    options.max_pending_requests = 32; // every request of a run is submitted at once
    options.prefill_chunk   = 1024;
    options.enable_vision   = false;
    if (drafter == "mtp" || drafter == "mtp-cut" || drafter == "mtp-sampled") {
        options.speculative.backend      = infernix::SpeculativeBackend::Mtp;
        options.speculative.draft_tokens = 3;
        // The draft-confidence cut verifies only the drafts before the first below 0.5 (the
        // others are padding of the fixed width); sampled drafts verify by sparse rejection
        // sampling, which under Gumbel-max draws keeps plain sampling's tokens.
        if (drafter == "mtp-cut") { options.speculative.draft_min_p = 0.5F; }
        if (drafter == "mtp-sampled") { options.fast_path_candidates = {infernix::FastPath::SampledMtpDrafts}; }
    } else if (drafter == "dflash") {
        options.speculative.backend      = infernix::SpeculativeBackend::DFlash;
        options.speculative.draft_tokens = 7;
    } else if (drafter == "dflash2" || drafter.starts_with("dflash2-ngram")) {
        options.speculative.backend      = infernix::SpeculativeBackend::DFlash2;
        options.speculative.draft_tokens = 7;
        if (drafter == "dflash2-ngram") { options.speculative.ngram_draft_tokens = 16; }
        // dflash2-ngramN: copy windows of N drafts at the shortest minimum match, so copy rounds verify
        // N + 1 columns and their last drafts are often rejected. N = 14..16 and 62..63 put the round
        // on both sides of the GDN record routes' width boundaries (16 columns; the tree closed form's
        // 16-column records) and at the 64-column top, where llama.cpp #31160/#31350 met a state
        // its rollback had not written.
        if (drafter.size() > std::string_view("dflash2-ngram").size()) {
            options.speculative.ngram_draft_tokens =
                static_cast<std::uint32_t>(std::stoul(std::string(drafter.substr(std::string_view("dflash2-ngram").size()))));
            options.speculative.ngram_min_match = 4;
        }
    } else if (drafter == "dspine" || drafter == "dspine-ngram") {
        const char* drafter_artifact = std::getenv("INFERNIX_TEST_DSPINE_DRAFTER");
        if (drafter_artifact == nullptr || *drafter_artifact == '\0') {
            throw std::invalid_argument("dspine needs INFERNIX_TEST_DSPINE_DRAFTER (a DSpine drafter artifact)");
        }
        options.speculative.backend      = infernix::SpeculativeBackend::DSpine;
        options.speculative.draft_model  = drafter_artifact;
        options.speculative.draft_tokens = 15;
        if (drafter == "dspine-ngram") { options.speculative.ngram_draft_tokens = 16; }
    } else if (drafter != "none") {
        throw std::invalid_argument("the drafter must be mtp, mtp-cut, mtp-sampled, dflash, dflash2, dflash2-ngram, "
                                    "dflash2-ngramN, dspine or dspine-ngram");
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

// Each request's label and tokens: per prompt greedy and two sampled seeds, then the one-token
// prompt's, then (penalised runs only) per prompt penalised greedy and sampled. A run without the
// penalised requests is a prefix of one with them.
struct Decoded {
    std::vector<std::string> labels;
    std::vector<std::vector<infernix::TokenId>> tokens;
    std::uint64_t one_token_drafts = 0; // drafts verified for the one-token prompt
    std::uint64_t drafted = 0, accepted = 0; // over every request (MTP/DFlash drafts and n-gram copies)
};

Decoded decode(const std::filesystem::path& artifact, std::string_view drafter, bool penalised) {
    infernix::Engine engine(reproducible_options(artifact, drafter));
    Decoded out;
    // Each request as its label and a call that submits it.
    std::vector<std::function<infernix::GenerationHandle()>> requests;
    const auto add = [&](std::string label, std::function<infernix::GenerationHandle()> submit) {
        out.labels.push_back(std::move(label));
        requests.push_back(std::move(submit));
    };
    const auto chat = [&engine](std::string text, infernix::RequestOptions options) {
        return [&engine, text = std::move(text), options] { return engine.submit(engine.prepare(prompt(text)), options); };
    };
    const std::vector<std::string> texts = prompts();
    for (std::size_t p = 0; p < texts.size(); ++p) {
        const std::string name = "prompt " + std::to_string(p);
        add(name + " greedy", chat(texts[p], request(0.0F, 0)));
        for (std::uint64_t seed : {11ULL, 12ULL}) {
            add(name + " sampled seed " + std::to_string(seed), chat(texts[p], request(1.0F, seed)));
        }
    }
    // A one-token prompt speculates like any other (Strata aece46db: there, position 0 meant "no
    // speculation", so such a prompt never drafted).
    const std::vector<infernix::TokenId> first = engine.tokenize_text("Once");
    require(!first.empty(), "the tokenizer encoded no token");
    const std::size_t one_token = requests.size();
    const infernix::TokenId once = first.front();
    add("one-token greedy", [&engine, once] { return engine.submit(engine.prepare_tokens({once}), request(0.0F, 0)); });
    if (penalised) {
        for (std::size_t p = 0; p < texts.size(); ++p) {
            const std::string name = "prompt " + std::to_string(p);
            add(name + " penalised greedy", chat(texts[p], request(0.0F, 0, true)));
            add(name + " penalised sampled", chat(texts[p], request(1.0F, 11, true)));
        }
    }
    // At most twice the lanes submitted and unfinished, oldest first: a request queued longer than the Engine's admission
    // bound expires (a one-lane arm's last requests did when all sixteen were submitted at once).
    const std::size_t window = 2 * static_cast<std::size_t>(lanes_for(drafter));
    std::deque<infernix::GenerationHandle> in_flight;
    std::size_t submitted = 0;
    for (std::size_t i = 0; i < requests.size(); ++i) {
        for (; submitted < requests.size() && in_flight.size() < window; ++submitted) {
            in_flight.push_back(requests[submitted]());
        }
        const infernix::GenerationResult result = in_flight.front().wait();
        in_flight.pop_front();
        const auto& s = result.speculative;
        if (i == one_token) { out.one_token_drafts = s.drafted_tokens + s.ngram_drafted_tokens; }
        out.drafted += s.drafted_tokens + s.ngram_drafted_tokens;
        out.accepted += s.accepted_tokens + s.ngram_accepted_tokens;
        out.tokens.push_back(result.generated_token_ids);
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
    for (int a = 1; a < argc; ++a) {
        const std::string_view drafter = argv[a];
        const char* dspine = std::getenv("INFERNIX_TEST_DSPINE_DRAFTER");
        if ((drafter == "dspine" || drafter == "dspine-ngram") && (dspine == nullptr || *dspine == '\0')) {
            std::cout << "SKIP: set INFERNIX_TEST_DSPINE_DRAFTER to a DSpine drafter artifact for " << drafter << '\n';
            return 77;
        }
    }
    try {
        require(argc >= 2,
                "usage: infernix_qwen3_5_speculation_invariance_real_test "
                "mtp|mtp-cut|mtp-sampled|dflash|dflash2|dflash2-ngram...");
        // Plain decoding once (with the penalised requests when a drafter needs them), then each
        // drafter against it.
        bool any_penalised = false;
        for (int a = 1; a < argc; ++a) { any_penalised = any_penalised || penalised_drafter(argv[a]); }
        const auto plain_run = decode(artifact, "none", any_penalised);
        const auto& plain    = plain_run.tokens;
        int failed_drafters  = 0;
        for (int a = 1; a < argc; ++a) {
            const std::string_view drafter = argv[a];
            const auto speculative_run     = decode(artifact, drafter, penalised_drafter(drafter));
            const auto& speculative        = speculative_run.tokens;
            require(speculative.size() <= plain.size(), "result counts differ");
            int failures = 0;
            for (std::size_t i = 0; i < speculative.size(); ++i) {
                require(plain_run.labels[i] == speculative_run.labels[i], "request orders differ");
                std::size_t same = 0;
                while (same < plain[i].size() && same < speculative[i].size() && plain[i][same] == speculative[i][same]) {
                    ++same;
                }
                const bool ok = !plain[i].empty() && plain[i] == speculative[i];
                std::cout << drafter << " " << speculative_run.labels[i] << ": " << (ok ? "identical" : "DIFFERS")
                          << " (" << same << " of " << plain[i].size() << "/" << speculative[i].size()
                          << " tokens agree)\n";
                failures += ok ? 0 : 1;
            }
            std::cout << drafter << " one-token prompt: " << speculative_run.one_token_drafts << " drafts verified\n";
            if (speculative_run.one_token_drafts == 0) {
                std::cerr << "FAIL " << drafter << ": a one-token prompt verified no drafts\n";
                ++failures;
            }
            std::cout << drafter << ": " << speculative_run.accepted << " of " << speculative_run.drafted
                      << " drafts accepted over every request (" << lanes_for(drafter) << " lanes)\n";
            if (speculative_run.accepted == 0) {
                std::cerr << "FAIL " << drafter << ": no draft was accepted\n";
                ++failures;
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
