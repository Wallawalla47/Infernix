#include "infernix/engine.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

// The exact route's request invariance on a real Qwen3.5 artifact (docs/maintainer/model-quality.md,
// 6.1): with fast paths disabled a request's tokens do not depend on concurrency, submission order,
// the other requests, the prefix cache's contents or KV pressure. Sampled decoding (temperature 1,
// a fixed seed per request) makes any arithmetic difference visible as a token difference within a
// few hundred tokens, where greedy decoding would hide most of them.
//
// INFERNIX_TEST_ARTIFACT names the artifact; argv[1] selects speculation (none, mtp, dflash or
// dflash2, default none; the artifact must hold that drafter). Under speculation a sampled request
// accepts or rejects against its drafts, so they too must not depend on the other requests.
namespace {

constexpr std::uint32_t kOutputTokens = 256;
constexpr std::uint32_t kContext      = 4096;
constexpr std::uint32_t kConcurrency  = 4;

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

infernix::PromptInput prompt(const std::string& text) {
    infernix::PromptInput input;
    infernix::ChatMessage user;
    user.role = infernix::ChatRole::User;
    user.parts.push_back(
        infernix::MessagePart{.kind = infernix::MessagePartKind::Text, .text = text, .media = {}});
    input.messages.push_back(std::move(user));
    input.options.enable_thinking   = false;
    input.options.preserve_thinking = true;
    return input;
}

infernix::RequestOptions sampled(std::uint64_t seed, bool prefix_reuse = false) {
    infernix::RequestOptions options;
    options.execution.requested_output_tokens = kOutputTokens;
    options.execution.sampling.temperature    = 1.0F;
    options.execution.sampling.top_k          = 0;
    options.execution.sampling.top_p          = 1.0F;
    options.execution.sampling.seed           = seed;
    options.execution.allow_prefix_reuse      = prefix_reuse;
    options.stop.include_model_defaults       = false;
    options.output.raw                        = true;
    return options;
}

infernix::EngineOptions exact_options(const std::filesystem::path& artifact,
                                    infernix::SpeculativeBackend backend, std::uint32_t kv_tokens,
                                    bool prefix_cache) {
    infernix::EngineOptions options;
    options.artifact_path        = artifact;
    options.fast_paths           = false; // the exact route
    options.max_context          = kContext;
    options.kv_capacity          = infernix::KvCapacityPolicy::explicit_capacity(kv_tokens);
    options.max_concurrency      = kConcurrency;
    options.max_pending_requests = 2 * kConcurrency;
    options.prefill_chunk        = 1024;
    options.enable_vision        = false;
    options.speculative.backend  = backend;
    if (backend == infernix::SpeculativeBackend::Mtp) { options.speculative.draft_tokens = 3; }
    if (backend == infernix::SpeculativeBackend::DFlash ||
        backend == infernix::SpeculativeBackend::DFlash2) {
        options.speculative.draft_tokens = 7;
    }
    options.context_cache.enabled = prefix_cache;
    if (prefix_cache) { options.context_cache.host_capacity_bytes = std::size_t{1} << 30U; }
    return options;
}

// Four unrelated prompts of different lengths.
std::vector<std::string> prompts() {
    return {
        "Write a short story about a lighthouse keeper who finds a message in a bottle.",
        "Explain, step by step, how a compiler turns source code into machine instructions, "
        "covering lexing, parsing, semantic analysis, optimisation and code generation.",
        "List twelve uses for a paperclip and give one sentence of explanation for each.",
        "Describe the water cycle to a ten-year-old, then summarise it in one line.",
    };
}

// A long shared document (about 3,000 tokens; 120 entries made 4,199) whose end is off the
// 1,024-token prefill grid, with room in the context for the output.
std::string long_document() {
    std::string text = "Read the following field notes and answer the question after them.\n\n";
    for (int entry = 0; entry < 85; ++entry) {
        text += "Entry " + std::to_string(entry) + ": the survey team logged " +
                std::to_string(3 * entry + 7) + " birds near marker " + std::to_string(entry % 17) +
                ", wind from the " + (entry % 2 ? "north" : "south-west") + ", water level " +
                std::to_string(40 + entry % 9) + " centimetres.\n";
    }
    return text;
}

std::size_t first_difference(const std::vector<infernix::TokenId>& a,
                             const std::vector<infernix::TokenId>& b) {
    std::size_t i = 0;
    while (i < a.size() && i < b.size() && a[i] == b[i]) { ++i; }
    return i;
}

void require_same(const std::string& label, const infernix::GenerationResult& expected,
                  const infernix::GenerationResult& actual) {
    const auto& a = expected.generated_token_ids;
    const auto& b = actual.generated_token_ids;
    require(!a.empty(), label + ": the reference generated nothing");
    if (a != b) {
        throw std::runtime_error(label + ": tokens differ at " +
                                 std::to_string(first_difference(a, b)) + " of " +
                                 std::to_string(a.size()) + "/" + std::to_string(b.size()));
    }
    std::cout << label << ": " << a.size() << " tokens identical\n";
}

infernix::SpeculativeBackend backend_of(std::string_view name) {
    if (name == "none") { return infernix::SpeculativeBackend::None; }
    if (name == "mtp") { return infernix::SpeculativeBackend::Mtp; }
    if (name == "dflash") { return infernix::SpeculativeBackend::DFlash; }
    if (name == "dflash2") { return infernix::SpeculativeBackend::DFlash2; }
    throw std::invalid_argument("speculation must be none, mtp, dflash or dflash2");
}

} // namespace

int main(int argc, char** argv) {
    const char* artifact = std::getenv("INFERNIX_TEST_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "SKIP: set INFERNIX_TEST_ARTIFACT to an explicit Qwen3.5 artifact\n";
        return 77;
    }
    try {
        require(argc <= 2, "usage: infernix_qwen3_5_invariance_real_test [none|mtp|dflash|dflash2]");
        const auto backend = backend_of(argc == 2 ? argv[1] : "none");
        const auto texts   = prompts();
        std::vector<infernix::GenerationResult> solo;

        {
            // Room for every request's whole extent: the exact route never preempts here.
            infernix::Engine engine(
                exact_options(artifact, backend, kConcurrency * kContext, true));
            require(engine.load_summary().request_invariant,
                    "the exact route does not report request invariance");

            // 1. Each request alone.
            for (std::size_t i = 0; i < texts.size(); ++i) {
                solo.push_back(engine.generate(engine.prepare(prompt(texts[i])), sampled(1000 + i)));
            }

            // 2. All of them at once, submitted in reverse order: concurrency, lane assignment and
            //    the batch's other rows do not change a request's tokens.
            std::vector<infernix::GenerationHandle> handles;
            for (std::size_t i = texts.size(); i-- > 0;) {
                handles.push_back(engine.submit(engine.prepare(prompt(texts[i])), sampled(1000 + i)));
            }
            for (std::size_t j = 0; j < handles.size(); ++j) {
                const std::size_t i = texts.size() - 1 - j;
                require_same("concurrent request " + std::to_string(i), solo[i], handles[j].wait());
            }

            // 3. A warm prefix: the second request shares a ~3,000-token document whose end is off
            //    the prefill grid. Resuming from the first request's cache gives the cold tokens.
            const std::string document = long_document();
            const auto cold = engine.generate(
                engine.prepare(prompt(document + "\nWhich marker had the most birds?")),
                sampled(7, false));
            (void)engine.generate(engine.prepare(prompt(document + "\nWhat was the wind like?")),
                                  sampled(8, true));
            const auto before = engine.runtime_stats();
            const auto warm   = engine.generate(
                engine.prepare(prompt(document + "\nWhich marker had the most birds?")),
                sampled(7, true));
            const auto after = engine.runtime_stats();
            require(after.reused_prompt_tokens > before.reused_prompt_tokens,
                    "the warm request reused no prefix");
            require_same("warm prefix", cold, warm);

            // 4. A request admitted while another decodes, at once and some way in: the round's other
            //    row is longer, so anything chosen from the round (an attention envelope, a split
            //    count) would reach the new request's arithmetic. Under DFlash2 that once changed a
            //    sampled request's drafts within five tokens while greedy ones stayed identical. The
            //    requests do not reuse prefixes (sampled's default), so each is the cold request of 1.
            for (const int delay_ms : {0, 400}) {
                auto running = engine.submit(engine.prepare(prompt(texts[2])), sampled(1002));
                std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
                auto admitted = engine.submit(engine.prepare(prompt(texts[3])), sampled(1003));
                (void)running.wait();
                require_same("request admitted " + std::to_string(delay_ms) + " ms into another",
                             solo[3], admitted.wait());
            }
        }

        {
            // 5. A KV pool for two whole extents and four requests: the exact route admits a request
            //    only when its whole extent fits, so none is preempted and every output is the one it
            //    gives alone. The pool and the prefix cache are Engine options, which may change
            //    outputs, so the solo outputs are this Engine's own.
            infernix::Engine engine(exact_options(artifact, backend, 2 * kContext, false));
            std::vector<infernix::GenerationResult> alone;
            for (std::size_t i = 0; i < texts.size(); ++i) {
                alone.push_back(engine.generate(engine.prepare(prompt(texts[i])), sampled(1000 + i)));
            }
            const auto before = engine.runtime_stats();
            std::vector<infernix::GenerationHandle> handles;
            for (std::size_t i = 0; i < texts.size(); ++i) {
                handles.push_back(engine.submit(engine.prepare(prompt(texts[i])), sampled(1000 + i)));
            }
            for (std::size_t i = 0; i < handles.size(); ++i) {
                require_same("small pool request " + std::to_string(i), alone[i], handles[i].wait());
            }
            require(engine.runtime_stats().preemptions == before.preemptions,
                    "the exact route preempted a request");
        }

        std::cout << "OK request invariance\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "request invariance: " << error.what() << '\n';
        return 1;
    }
}
