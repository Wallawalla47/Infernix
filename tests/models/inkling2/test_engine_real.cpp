// Inkling 2 through the public Engine on a real artifact (the Program's first slice, src/models/inkling2/program).
// The Engine without CUDA graphs, prefilling 16 columns a call and serving one request at a time, is the reference;
// every other configuration must give its tokens:
// 1. Greedy: the golden prompt of the forward test's reference dump (its first 24 ids) continues with the dump's next
//    8 ids (the forward's free-running argmax at positions 23-30, which the bit guard's logits confirm).
// 2. Graphs: the Engine with CUDA graphs gives the reference's tokens.
// 3. Request invariance: four sampled requests (temperature 1, a seed each, 24-48 tokens so lanes drop out mid-run in a
//    known order) give their solo tokens two at once (C = 2), all at once in reverse order (C = 4) and twice over at
//    once (C = 8).
// 4. Prefill chunk width: an Engine prefilling 7 columns a call gives the reference's tokens; on a 2,300-token prompt
//    at context 4,096, chunks of 128, 4,096 (the prompt in one call) and 100 (beside a decoding request) give the
//    greedy continuation, and chunks of 128, 100, 1,000 and 4,096 the scoring logits (19 positions across the
//    window's wrap and the chunk edges), of chunks of 16 byte for byte.
// 5. Scoring: a CausalScoring Engine's logits of the 32 golden ids (positions 0-30) equal the forward test's
//    prompt-route logits (INFERNIX_INKLING2_PROMPT_LOGITS_GUARD: scoring runs the prompt route) byte for byte.
// 6. --q8-weights (approximate 8-bit weights; refused with --reproducible) and --a16-activations (16-bit expert
//    activations; accepted with --reproducible, giving the same tokens), both together, and --a8-activations (8-bit
//    activations in every call; refused with --reproducible): under each, an eager Engine at chunks of 16
//    serving one request at a time is the reference, and graphs (alone and C = 4 in reverse order), chunks of 7, and
//    the long prompt in one 4,096-column call give its tokens.
// 7. DSpark (--spec dspark, auto and 7 drafts, graphs and eager; INFERNIX_INKLING2_NO_DSPARK skips it): speculative
//    tokens equal plain decoding's (the golden continuation, a 160-token greedy reply raw and parsed: the parsed one
//    samples under its reasoning ban, the four sampled requests alone
//    and C = 2; under --a16-activations and under --a8-activations the greedy reply equals that route's plain tokens);
//    the acceptance is reported.
// 8. --kv-dtype (the full visits' pools quantized): each format as a variant of 6 (accepted with --reproducible),
//    DSpark (auto, graphs) giving its plain tokens, and a 10,000-token prompt's greedy continuation the same in chunks
//    of 4,096 and 1,000; bf16p (lossless) gives bf16's tokens throughout, the others' agreement with BF16's is reported.
// 9. A stalled MoE call (INFERNIX_INKLING2_STALL_ONLY runs it alone): the request fails with the stall, the next one
//    gives its solo tokens.
// Reported, not judged: a chat turn's channels, a turn whose thinking budget (6) injects the control (the mixed-width
// decode after forced tokens), and how long --q8-weights' tokens agree with the BF16 weights'.
//
//   INFERNIX_INKLING2_ARTIFACT      an artifact converted with the inkling2_mini recipe
//   INFERNIX_INKLING2_REFERENCE     the reference dump's path prefix (<prefix>.bin and <prefix>.json: the golden ids)
//   INFERNIX_INKLING2_PROMPT_LOGITS_GUARD  the forward test's prompt-route logits ([32][V] FP32), for 5

#include "infernix/engine.h"
#include "models/inkling2/program/test_taps.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::uint32_t kContext = 1024;
using Tokens = std::vector<infernix::TokenId>;

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

Tokens golden_ids(const std::string& prefix) {
    std::ifstream meta(prefix + ".json");
    require(static_cast<bool>(meta), "cannot read " + prefix + ".json");
    const auto json  = nlohmann::json::parse(meta);
    const auto entry = json.at("ids");
    std::ifstream bin(prefix + ".bin", std::ios::binary);
    require(static_cast<bool>(bin), "cannot read " + prefix + ".bin");
    Tokens ids(entry.at("shape").at(0).get<std::size_t>());
    bin.seekg(entry.at("offset").get<std::streamoff>());
    bin.read(reinterpret_cast<char*>(ids.data()), static_cast<std::streamsize>(ids.size() * 4));
    require(static_cast<bool>(bin), "the dump's ids are truncated");
    return ids;
}

infernix::EngineOptions engine_options(const std::filesystem::path& artifact, std::uint32_t chunk, bool graphs,
                                       std::uint32_t concurrency, std::uint32_t context = kContext) {
    infernix::EngineOptions options;
    options.artifact_path        = artifact;
    options.max_context          = context;
    options.kv_capacity          = infernix::KvCapacityPolicy::explicit_capacity(concurrency * context);
    options.max_concurrency      = concurrency;
    options.max_pending_requests = 16;
    // The prefix cache stays on (the Engine's default): requests sharing a prompt resume from it, which must not change
    // their tokens; a small Host tier keeps the start quick.
    options.context_cache.host_capacity_bytes = std::size_t{2} << 30U;
    // INFERNIX_INKLING2_NO_PREFIX_CACHE: the cache off (its taps split prompt calls, which changes the drafter's context
    // rows and so its drafts, never the tokens: the drafts then compare with a run of the same calls).
    if (std::getenv("INFERNIX_INKLING2_NO_PREFIX_CACHE") != nullptr) { options.context_cache.enabled = false; }
    options.prefill_chunk        = chunk;
    options.use_cuda_graph       = graphs;
    // Every diagnostic, Debug included (a failed request's cause is one).
    options.diagnostic_observer.callback = [](const infernix::Diagnostic& d) {
        std::cerr << "[engine " << static_cast<int>(d.level) << "] " << d.message << '\n';
    };
    return options;
}

infernix::RequestOptions greedy(std::uint32_t tokens) {
    infernix::RequestOptions options;
    options.execution.requested_output_tokens = tokens;
    options.execution.sampling.temperature    = 0.0F;
    options.stop.include_model_defaults       = false;
    options.output.raw                        = true;
    return options;
}

// Request i: seed 1000 + i, 24 + 8 i tokens.
infernix::RequestOptions sampled(std::size_t i) {
    infernix::RequestOptions options;
    options.execution.requested_output_tokens = static_cast<std::uint32_t>(24 + 8 * i);
    options.execution.sampling.temperature    = 1.0F;
    options.execution.sampling.top_k          = 20;
    options.execution.sampling.top_p          = 1.0F;
    options.execution.sampling.seed           = 1000 + i;
    options.stop.include_model_defaults       = false;
    options.output.raw                        = true;
    return options;
}

infernix::PromptInput chat(const std::string& text) {
    infernix::PromptInput input;
    infernix::ChatMessage user;
    user.role = infernix::ChatRole::User;
    user.parts.push_back({.text = text});
    input.messages.push_back(std::move(user));
    input.options.reasoning_effort = infernix::ReasoningEffort::Low;
    return input;
}

const std::vector<std::string>& texts() {
    static const std::vector<std::string> t = {
        "Write a short story about a lighthouse keeper who finds a message in a bottle.",
        "Explain how a compiler turns source code into machine instructions.",
        "List twelve uses for a paperclip.",
        "Describe the water cycle to a ten-year-old.",
    };
    return t;
}

void require_same(const std::string& label, const Tokens& a, const Tokens& b) {
    std::size_t at = 0;
    while (at < a.size() && at < b.size() && a[at] == b[at]) { ++at; }
    require(!a.empty() && a == b, label + ": tokens differ at " + std::to_string(at) + " of " + std::to_string(a.size()) +
                                      "/" + std::to_string(b.size()));
    std::cout << label << ": " << a.size() << " tokens identical\n";
}

// A long prompt: numbered sentences, the first `count` ids of their raw-text encoding.
constexpr std::size_t kLongPrompt  = 2300;
constexpr std::uint32_t kLongContext = 4096;
Tokens long_prompt(const infernix::Engine& engine, std::size_t count = kLongPrompt) {
    static const char* const adjectives[] = {"quiet", "amber", "distant", "narrow", "restless", "silver", "patient"};
    static const char* const nouns[]      = {"harbour", "ledger", "orchard", "signal", "bridge", "lantern",
                                             "archive", "valley", "engine", "garden", "compass"};
    std::string text;
    for (int i = 0; text.size() < std::max<std::size_t>(24000, count * 12); ++i) {
        text += "Item " + std::to_string(i) + ": the " + adjectives[i % 7] + " " + nouns[i % 11] + " in section " +
                std::to_string(i * 37 % 101) + " was logged at " + std::to_string(i * 13 % 60) + " minutes past " +
                std::to_string(i % 12 + 1) + ".\n";
    }
    Tokens ids = engine.tokenize_text(text);
    require(ids.size() >= count, "the long prompt's text encodes to at least " + std::to_string(count) + " ids");
    ids.resize(count);
    return ids;
}

std::size_t agree(const Tokens& a, const Tokens& b) {
    std::size_t at = 0;
    while (at < a.size() && at < b.size() && a[at] == b[at]) { ++at; }
    return at;
}

// Scoring logits of the long prompt at positions inside the first chunk, across chunk edges (16, 100, 128 columns),
// the window's wrap (1023-1088), the full visits' multi-unit splits and the end.
std::vector<float> long_scores(const std::filesystem::path& artifact, std::uint32_t chunk, const Tokens& prompt,
                               const std::vector<std::uint32_t>& positions) {
    infernix::EngineOptions options = engine_options(artifact, chunk, false, 1, kLongContext);
    options.purpose                 = infernix::EnginePurpose::CausalScoring;
    infernix::Engine engine(options);
    infernix::ScoreOptions score;
    score.logits_positions = positions;
    return engine.score_tokens(prompt, 1, score).logits;
}

std::vector<Tokens> alone(infernix::Engine& engine) {
    std::vector<Tokens> out;
    for (std::size_t i = 0; i < texts().size(); ++i) {
        out.push_back(engine.generate(engine.prepare(chat(texts()[i])), sampled(i)).generated_token_ids);
    }
    return out;
}

void require_alone(const std::string& label, infernix::Engine& engine, const std::vector<Tokens>& reference) {
    const auto mine = alone(engine);
    for (std::size_t i = 0; i < mine.size(); ++i) { require_same(label + ", request " + std::to_string(i), reference[i], mine[i]); }
}

// 9. A stalled MoE call (testing::force_moe_stall writes kErrorDownStalled through the MoE workspace's error word, as a
//    down CTA that waited past its bound sets it): the request holding the lane fails with the Engine's recoverable
//    error instead of committing tokens, and the Engine keeps serving: the next request gives its solo tokens.
void stalled_moe(const std::filesystem::path& artifact) {
    namespace inkling_testing = infernix::models::inkling2::testing;
    infernix::Engine engine(engine_options(artifact, 16, true, 1));
    const Tokens solo = engine.generate(engine.prepare(chat(texts()[1])), sampled(1)).generated_token_ids;
    inkling_testing::force_moe_stall(1);
    std::string error;
    try {
        (void)engine.generate(engine.prepare(chat(texts()[0])), sampled(0));
    } catch (const std::exception& e) { error = e.what(); }
    inkling_testing::force_moe_stall(0);
    std::cout << "stalled MoE: " << (error.empty() ? std::string("the request completed") : error) << '\n';
    require(error.find("down pass waited past its bound") != std::string::npos,
            "stalled MoE: the request did not fail with the stall");
    require_same("stalled MoE: the next request", solo,
                 engine.generate(engine.prepare(chat(texts()[1])), sampled(1)).generated_token_ids);
}

} // namespace

int main() {
    const char* artifact  = std::getenv("INFERNIX_INKLING2_ARTIFACT");
    const char* reference = std::getenv("INFERNIX_INKLING2_REFERENCE");
    const char* guard     = std::getenv("INFERNIX_INKLING2_PROMPT_LOGITS_GUARD");
    if (artifact == nullptr || reference == nullptr) {
        std::cout << "skip: set INFERNIX_INKLING2_ARTIFACT and INFERNIX_INKLING2_REFERENCE\n";
        return 77;
    }
    try {
        if (std::getenv("INFERNIX_INKLING2_STALL_ONLY") != nullptr) {
            stalled_moe(artifact);
            std::cout << "PASS (the stalled-MoE case alone)\n";
            return 0;
        }
        stalled_moe(artifact);
        const Tokens golden = golden_ids(reference);
        require(golden.size() == 32, "the dump holds 32 ids");
        const Tokens prompt(golden.begin(), golden.begin() + 24);
        const Tokens continuation(golden.begin() + 24, golden.end());
        std::vector<Tokens> reference_tokens;
        {
            // The reference: no graphs, 16 columns a prefill call, one request at a time.
            infernix::Engine engine(engine_options(artifact, 16, false, 1));
            const auto first = engine.generate(engine.prepare_tokens(prompt), greedy(8));
            std::cout << "eager greedy: " << first.generated_token_ids.size() << " tokens, finish "
                      << static_cast<int>(first.finish_reason) << ", prompt " << first.prompt.prompt_tokens
                      << ", prefilled " << first.computed_prefill_tokens << ", prefill "
                      << first.timings.prefill_seconds << " s, decode " << first.timings.decode_seconds << " s\n  got     ";
            for (const auto id : first.generated_token_ids) { std::cout << ' ' << id; }
            std::cout << "\n  expected";
            for (const auto id : continuation) { std::cout << ' ' << id; }
            std::cout << '\n';
            require_same("eager: greedy continuation of the golden prompt", continuation, first.generated_token_ids);
            reference_tokens = alone(engine);
        }
        {
            infernix::Engine engine(engine_options(artifact, 16, true, 8));
            require_same("graphs: greedy continuation of the golden prompt", continuation,
                         engine.generate(engine.prepare_tokens(prompt), greedy(8)).generated_token_ids);
            require_alone("graphs alone", engine, reference_tokens);
            {
                // C = 2: requests 1 and 0 at once (request 0 finishes first, its lane emptying under request 1).
                auto second = engine.submit(engine.prepare(chat(texts()[1])), sampled(1));
                auto first  = engine.submit(engine.prepare(chat(texts()[0])), sampled(0));
                require_same("C = 2, request 1", reference_tokens[1], second.wait().generated_token_ids);
                require_same("C = 2, request 0", reference_tokens[0], first.wait().generated_token_ids);
            }
            std::vector<infernix::GenerationHandle> handles;
            for (std::size_t i = texts().size(); i-- > 0;) {
                handles.push_back(engine.submit(engine.prepare(chat(texts()[i])), sampled(i)));
            }
            for (std::size_t j = 0; j < handles.size(); ++j) {
                const std::size_t i = texts().size() - 1 - j;
                require_same("C = 4, request " + std::to_string(i), reference_tokens[i], handles[j].wait().generated_token_ids);
            }
            handles.clear();
            for (int copy = 0; copy < 2; ++copy) {
                for (std::size_t i = 0; i < texts().size(); ++i) {
                    handles.push_back(engine.submit(engine.prepare(chat(texts()[i])), sampled(i)));
                }
            }
            for (std::size_t j = 0; j < handles.size(); ++j) {
                require_same("C = 8, request " + std::to_string(j), reference_tokens[j % texts().size()],
                             handles[j].wait().generated_token_ids);
            }
            // Reported: a chat turn's channels, and a thinking budget's control.
            infernix::RequestOptions turn;
            turn.execution.requested_output_tokens = 64;
            turn.execution.sampling.temperature    = 0.0F;
            const auto reply = engine.generate(engine.prepare(chat("What is 2 + 2? Answer briefly.")), turn);
            std::cout << "chat turn (" << reply.generated_token_ids.size() << " tokens): reasoning "
                      << nlohmann::json(reply.reasoning).dump() << ", content " << nlohmann::json(reply.content).dump()
                      << '\n';
            turn.execution.thinking.budget = 6;
            const auto capped = engine.generate(engine.prepare(chat("Plan a three-day trip to Rome.")), turn);
            std::cout << "thinking budget 6 (" << capped.generated_token_ids.size() << " tokens): applied "
                      << capped.thinking.applied << ", model thinking tokens " << capped.thinking.model_thinking_tokens
                      << ", injected " << capped.thinking.injected_tokens << "; reasoning "
                      << nlohmann::json(capped.reasoning).dump() << ", content "
                      << nlohmann::json(capped.content.substr(0, 160)).dump() << '\n';
        }
        {
            infernix::Engine engine(engine_options(artifact, 7, true, 8));
            require_alone("prefill chunks of 7", engine, reference_tokens);
        }
        if (std::getenv("INFERNIX_INKLING2_NO_DSPARK") == nullptr) {
            // 7. DSpark: verification keeps a draft only when it equals the target's own draw at its position, so
            // speculative output is plain decoding's, token for token (greedy and sampled, auto and fixed lengths,
            // graphs and eager); C = 2 decodes plainly while still appending the drafter's context. Reported: the
            // acceptance.
            Tokens plain_long, plain_reasoning;
            infernix::RequestOptions long_greedy = greedy(160);
            // Parsed output: while the reply's reasoning is open every column samples under its reasoning ban.
            infernix::RequestOptions reasoning = greedy(160);
            reasoning.output.raw               = false;
            {
                infernix::Engine engine(engine_options(artifact, 0, true, 1));
                plain_long = engine.generate(engine.prepare(chat(texts()[1])), long_greedy).generated_token_ids;
                plain_reasoning = engine.generate(engine.prepare(chat(texts()[0])), reasoning).generated_token_ids;
            }
            for (const bool automatic : {true, false}) {
                for (const bool graphs : {true, false}) {
                    if (!graphs && automatic) { continue; }
                    infernix::EngineOptions options = engine_options(artifact, 0, graphs, 2);
                    options.speculative.backend           = infernix::SpeculativeBackend::DSpark;
                    options.speculative.draft_tokens_auto = automatic;
                    options.speculative.draft_tokens      = 7;
                    infernix::Engine engine(options);
                    const std::string label = std::string("dspark (") + (automatic ? "auto" : "7 drafts") +
                                              (graphs ? ", graphs" : ", eager") + ")";
                    require_same(label + ": greedy continuation of the golden prompt", continuation,
                                 engine.generate(engine.prepare_tokens(prompt), greedy(8)).generated_token_ids);
                    const auto result = engine.generate(engine.prepare(chat(texts()[1])), long_greedy);
                    require_same(label + ": a 160-token greedy reply", plain_long, result.generated_token_ids);
                    const auto parsed = engine.generate(engine.prepare(chat(texts()[0])), reasoning);
                    require_same(label + ": a 160-token parsed greedy reply (reasoning ban)", plain_reasoning,
                                 parsed.generated_token_ids);
                    std::cout << label << ": parsed reply " << parsed.speculative.rounds << " rounds, "
                              << parsed.speculative.accepted_tokens << " of " << parsed.speculative.drafted_tokens
                              << " drafts accepted, " << parsed.speculative.fallback_steps << " plain steps\n";
                    const infernix::SpeculativeStats& s = result.speculative;
                    std::cout << label << ": " << s.rounds << " rounds, " << s.drafted_tokens << " drafted, "
                              << s.accepted_tokens << " accepted ("
                              << (s.drafted_tokens == 0 ? 0.0 : 100.0 * s.accepted_tokens / s.drafted_tokens)
                              << " %), " << s.fallback_steps << " plain steps; accepted by position";
                    for (const auto n : s.accepted_per_position) { std::cout << ' ' << n; }
                    std::cout << '\n';
                    require_alone(label, engine, reference_tokens);
                    if (graphs && automatic) {
                        auto second = engine.submit(engine.prepare(chat(texts()[1])), sampled(1));
                        auto first  = engine.submit(engine.prepare(chat(texts()[0])), sampled(0));
                        require_same(label + ", C = 2, request 1", reference_tokens[1], second.wait().generated_token_ids);
                        require_same(label + ", C = 2, request 0", reference_tokens[0], first.wait().generated_token_ids);
                    }
                }
            }
            {
                // --lm-head-draft: the drafter reads a shortlist, and verification still decides every output. On an
                // artifact without a ranking the flag is refused (INFERNIX_INKLING2_RANKED=1: the artifact must have one).
                infernix::EngineOptions options       = engine_options(artifact, 0, true, 2);
                options.speculative.backend           = infernix::SpeculativeBackend::DSpark;
                options.speculative.draft_tokens_auto = true;
                options.speculative.draft_tokens      = 7;
                options.speculative.proposal_head     = infernix::ProposalHead::Optimized;
                std::unique_ptr<infernix::Engine> shortlisted;
                try {
                    shortlisted = std::make_unique<infernix::Engine>(options);
                } catch (const std::exception& error) {
                    if (std::getenv("INFERNIX_INKLING2_RANKED") != nullptr ||
                        std::string(error.what()).find("no token ranking") == std::string::npos) {
                        throw;
                    }
                    std::cout << "--lm-head-draft: refused on an artifact without a ranking\n";
                }
                if (shortlisted) {
                    infernix::Engine& engine = *shortlisted;
                    const auto result = engine.generate(engine.prepare(chat(texts()[1])), long_greedy);
                    require_same("--lm-head-draft: a 160-token greedy reply", plain_long, result.generated_token_ids);
                    require_same("--lm-head-draft: the same request again (from the prefix cache)", plain_long,
                                 engine.generate(engine.prepare(chat(texts()[1])), long_greedy).generated_token_ids);
                    require_same("--lm-head-draft: a parsed greedy reply (reasoning ban)", plain_reasoning,
                                 engine.generate(engine.prepare(chat(texts()[0])), reasoning).generated_token_ids);
                    require_alone("--lm-head-draft", engine, reference_tokens);
                    auto second = engine.submit(engine.prepare(chat(texts()[1])), sampled(1));
                    auto first  = engine.submit(engine.prepare(chat(texts()[0])), sampled(0));
                    require_same("--lm-head-draft, C = 2, request 1", reference_tokens[1], second.wait().generated_token_ids);
                    require_same("--lm-head-draft, C = 2, request 0", reference_tokens[0], first.wait().generated_token_ids);
                    const infernix::SpeculativeStats& s = result.speculative;
                    std::cout << "--lm-head-draft: " << s.rounds << " rounds, " << s.drafted_tokens << " drafted, "
                              << s.accepted_tokens << " accepted, " << s.fallback_steps << " plain steps\n";
                }
            }
            {
                // Under --a16-activations: DSpark (auto, graphs) gives the A16 route's plain tokens (its verify calls
                // run the experts at other widths than plain decoding).
                infernix::EngineOptions plain = engine_options(artifact, 0, true, 1);
                plain.expert_a16_activations  = true;
                Tokens a16_long;
                {
                    infernix::Engine engine(plain);
                    a16_long = engine.generate(engine.prepare(chat(texts()[1])), long_greedy).generated_token_ids;
                }
                infernix::EngineOptions options       = plain;
                options.speculative.backend           = infernix::SpeculativeBackend::DSpark;
                options.speculative.draft_tokens_auto = true;
                options.speculative.draft_tokens      = 7;
                infernix::Engine engine(options);
                const auto result = engine.generate(engine.prepare(chat(texts()[1])), long_greedy);
                require_same("dspark (auto, graphs), a16: a 160-token greedy reply", a16_long, result.generated_token_ids);
                const infernix::SpeculativeStats& s = result.speculative;
                std::cout << "dspark (auto, graphs), a16: " << s.rounds << " rounds, " << s.drafted_tokens << " drafted, "
                          << s.accepted_tokens << " accepted, " << s.fallback_steps << " plain steps\n";
            }
            {
                // Under --a8-activations (--spec auto's drafter beside it): DSpark (auto, graphs) gives the A8 route's
                // plain tokens (its verify calls run every 8-bit kernel at other widths than plain decoding).
                infernix::EngineOptions plain = engine_options(artifact, 0, true, 1);
                plain.a8_activations          = true;
                Tokens a8_long;
                {
                    infernix::Engine engine(plain);
                    a8_long = engine.generate(engine.prepare(chat(texts()[1])), long_greedy).generated_token_ids;
                }
                infernix::EngineOptions options       = plain;
                options.speculative.backend           = infernix::SpeculativeBackend::DSpark;
                options.speculative.draft_tokens_auto = true;
                options.speculative.draft_tokens      = 7;
                infernix::Engine engine(options);
                const auto result = engine.generate(engine.prepare(chat(texts()[1])), long_greedy);
                require_same("dspark (auto, graphs), a8: a 160-token greedy reply", a8_long, result.generated_token_ids);
                const infernix::SpeculativeStats& s = result.speculative;
                std::cout << "dspark (auto, graphs), a8: " << s.rounds << " rounds, " << s.drafted_tokens << " drafted, "
                          << s.accepted_tokens << " accepted, " << s.fallback_steps << " plain steps\n";
            }
        }
        {
            // Wide prefill calls (128 and 100 columns: the conv rings' history snapshot, the window's staging ring,
            // attention's groups) against 16 a call, on a prompt past the window.
            Tokens prompt;
            Tokens oracle_long, oracle_request;
            {
                infernix::Engine engine(engine_options(artifact, 16, true, 1, kLongContext));
                prompt         = long_prompt(engine);
                oracle_long    = engine.generate(engine.prepare_tokens(prompt), greedy(16)).generated_token_ids;
                oracle_request = engine.generate(engine.prepare(chat(texts()[2])), sampled(2)).generated_token_ids;
            }
            for (const std::uint32_t chunk : {128U, 4096U}) { // 4096: the whole prompt in one call
                infernix::Engine engine(engine_options(artifact, chunk, true, 1, kLongContext));
                require_same("prefill chunks of " + std::to_string(chunk) + ": the long prompt's greedy continuation",
                             oracle_long, engine.generate(engine.prepare_tokens(prompt), greedy(16)).generated_token_ids);
            }
            {
                // Chunks of 100 while another request decodes: its prompt and decode rounds interleave with the wide
                // prefill calls.
                infernix::Engine engine(engine_options(artifact, 100, true, 2, kLongContext));
                auto other = engine.submit(engine.prepare(chat(texts()[2])), sampled(2));
                auto mine  = engine.submit(engine.prepare_tokens(prompt), greedy(16));
                require_same("prefill chunks of 100 beside a decoding request: the long prompt", oracle_long,
                             mine.wait().generated_token_ids);
                require_same("prefill chunks of 100 beside a decoding request: the other request", oracle_request,
                             other.wait().generated_token_ids);
            }
            const std::vector<std::uint32_t> positions = {0, 15, 16, 99, 100, 127, 128, 1022, 1023, 1024, 1087,
                                                          1088, 1151, 1200, 2047, 2048, 2200, 2297, 2298};
            const std::vector<float> oracle = long_scores(artifact, 16, prompt, positions);
            for (const std::uint32_t chunk : {128U, 100U, 1000U, 4096U}) {
                const std::vector<float> wide = long_scores(artifact, chunk, prompt, positions);
                require(wide.size() == oracle.size(), "scoring: the logits' sizes differ");
                std::size_t differ = 0;
                for (std::size_t i = 0; i < oracle.size(); ++i) {
                    differ += std::memcmp(&oracle[i], &wide[i], 4) != 0 ? 1 : 0;
                }
                require(differ == 0, "scoring the long prompt in chunks of " + std::to_string(chunk) + ": " +
                                         std::to_string(differ) + " logits differ from chunks of 16");
                std::cout << "scoring the long prompt in chunks of " << chunk << ": " << positions.size()
                          << " logits rows equal chunks of 16 byte for byte\n";
            }
        }
        {
            // 6. --q8-weights (8-bit weights: approximate) and --a16-activations (16-bit expert activations: another
            // arithmetic), each giving other tokens than the default's: the variant's own reference (no graphs, chunks of
            // 16, one request at a time) and every other configuration must agree, as above. --q8-weights is refused
            // with --reproducible; --a16-activations is not, and gives its reference's tokens there.
            const auto variant = [&](const std::string& name, const std::string& flag, bool reproducible,
                                     const std::function<void(infernix::EngineOptions&)>& set) {
                const auto make = [&](std::uint32_t chunk, bool graphs, std::uint32_t concurrency,
                                      std::uint32_t context = kContext) {
                    infernix::EngineOptions options = engine_options(artifact, chunk, graphs, concurrency, context);
                    set(options);
                    return options;
                };
                std::vector<Tokens> reference;
                Tokens greedy8;
                {
                    infernix::Engine engine(make(16, false, 1));
                    greedy8 = engine.generate(engine.prepare_tokens(prompt), greedy(8)).generated_token_ids;
                    std::size_t same = 0;
                    while (same < greedy8.size() && same < continuation.size() && greedy8[same] == continuation[same]) {
                        ++same;
                    }
                    std::cout << name << ": the golden prompt's greedy continuation agrees with the default's for " << same
                              << " of 8 tokens (reported)\n";
                    reference = alone(engine);
                    for (std::size_t i = 0; i < reference.size(); ++i) {
                        std::size_t at = 0;
                        while (at < reference[i].size() && at < reference_tokens[i].size() &&
                               reference[i][at] == reference_tokens[i][at]) {
                            ++at;
                        }
                        std::cout << name << ": sampled request " << i << " agrees with the default's for " << at << " of "
                                  << reference_tokens[i].size() << " tokens (reported)\n";
                    }
                }
                {
                    infernix::EngineOptions strict = make(16, false, 1);
                    strict.fast_paths              = false;
                    if (reproducible) {
                        infernix::Engine engine(strict);
                        require_same(name + ", --reproducible: greedy continuation", greedy8,
                                     engine.generate(engine.prepare_tokens(prompt), greedy(8)).generated_token_ids);
                        std::cout << flag << " with --reproducible: accepted\n";
                    } else {
                        bool threw = false;
                        try {
                            infernix::Engine engine(strict);
                        } catch (const std::invalid_argument& e) {
                            threw = std::string(e.what()).find(flag) != std::string::npos;
                        }
                        require(threw, flag + " was accepted on the reproducible route");
                        std::cout << flag << " with --reproducible: refused\n";
                    }
                }
                {
                    infernix::Engine engine(make(16, true, 8));
                    require_same(name + ", graphs: greedy continuation", greedy8,
                                 engine.generate(engine.prepare_tokens(prompt), greedy(8)).generated_token_ids);
                    require_alone(name + ", graphs alone", engine, reference);
                    std::vector<infernix::GenerationHandle> handles;
                    for (std::size_t i = texts().size(); i-- > 0;) {
                        handles.push_back(engine.submit(engine.prepare(chat(texts()[i])), sampled(i)));
                    }
                    for (std::size_t j = 0; j < handles.size(); ++j) {
                        const std::size_t i = texts().size() - 1 - j;
                        require_same(name + ", C = 4, request " + std::to_string(i), reference[i],
                                     handles[j].wait().generated_token_ids);
                    }
                }
                {
                    infernix::Engine engine(make(7, true, 8));
                    require_alone(name + ", prefill chunks of 7", engine, reference);
                }
                Tokens long_ids, long_greedy;
                {
                    infernix::Engine engine(make(16, true, 1, kLongContext));
                    long_ids    = long_prompt(engine);
                    long_greedy = engine.generate(engine.prepare_tokens(long_ids), greedy(16)).generated_token_ids;
                }
                {
                    infernix::Engine engine(make(4096, true, 1, kLongContext));
                    require_same(name + ", prefill chunks of 4096: the long prompt's greedy continuation", long_greedy,
                                 engine.generate(engine.prepare_tokens(long_ids), greedy(16)).generated_token_ids);
                }
            };
            variant("q8", "--q8-weights", false, [](infernix::EngineOptions& o) { o.q8_weights = true; });
            variant("a16", "--a16-activations", true, [](infernix::EngineOptions& o) { o.expert_a16_activations = true; });
            variant("q8 + a16", "--q8-weights", false, [](infernix::EngineOptions& o) {
                o.q8_weights             = true;
                o.expert_a16_activations = true;
            });
            // --a8-activations (8-bit activations in every call: the dense projections, the head and the NVFP4 experts;
            // implies --q8-weights): request-invariant like the others, refused with --reproducible.
            variant("a8", "--a8-activations", false, [](infernix::EngineOptions& o) { o.a8_activations = true; });

            // 8. --kv-dtype: the full visits' pools in a quantized format (the sliding windows stay BF16 K / FP16 V),
            // deterministic and request-invariant, so accepted with --reproducible: the variant's checks above; DSpark
            // (auto, graphs) gives the format's plain tokens; a 10,000-token prompt's greedy continuation is the same in
            // chunks of 4,096 and 1,000. Reported: how long the format's tokens agree with BF16's.
            constexpr std::size_t kTenThousand    = 10000;
            constexpr std::uint32_t kTenThousandContext = 10240 + 256;
            Tokens ten_k, bf16_ten_k, bf16_long_reply;
            {
                infernix::Engine engine(engine_options(artifact, 4096, true, 1, kTenThousandContext));
                ten_k      = long_prompt(engine, kTenThousand);
                bf16_ten_k = engine.generate(engine.prepare_tokens(ten_k), greedy(32)).generated_token_ids;
                bf16_long_reply = engine.generate(engine.prepare(chat(texts()[1])), greedy(160)).generated_token_ids;
            }
            for (const infernix::KvCacheStorage storage :
                 {infernix::KvCacheStorage::Int8Group64, infernix::KvCacheStorage::Fp8E4M3Row256,
                  infernix::KvCacheStorage::Nvfp4Group16, infernix::KvCacheStorage::Fp8KeyNvfp4Value,
                  infernix::KvCacheStorage::Vq2, infernix::KvCacheStorage::Q4KeyVq2Value,
                  infernix::KvCacheStorage::Bf16Packed}) {
                const std::string name = std::string("kv ") + infernix::kv_cache_cli_name(storage);
                const bool lossless    = storage == infernix::KvCacheStorage::Bf16Packed; // bf16's bits
                const auto set = [storage](infernix::EngineOptions& o) { o.kv_cache = storage; };
                variant(name, "--kv-dtype", true, set);
                infernix::EngineOptions plain = engine_options(artifact, 0, true, 1);
                set(plain);
                Tokens plain_reply;
                {
                    infernix::Engine engine(plain);
                    plain_reply = engine.generate(engine.prepare(chat(texts()[1])), greedy(160)).generated_token_ids;
                    if (lossless) {
                        require_same(name + ": the golden continuation", continuation,
                                     engine.generate(engine.prepare_tokens(prompt), greedy(8)).generated_token_ids);
                        require_alone(name + " alone = bf16", engine, reference_tokens);
                    }
                }
                if (lossless) {
                    require_same(name + ": a 160-token greedy reply = bf16's", bf16_long_reply, plain_reply);
                } else {
                    std::cout << name << ": a 160-token greedy reply agrees with BF16's for "
                              << agree(plain_reply, bf16_long_reply) << " tokens (reported)\n";
                }
                infernix::EngineOptions spec          = plain;
                spec.speculative.backend           = infernix::SpeculativeBackend::DSpark;
                spec.speculative.draft_tokens_auto = true;
                spec.speculative.draft_tokens      = 7;
                {
                    infernix::Engine engine(spec);
                    const auto result = engine.generate(engine.prepare(chat(texts()[1])), greedy(160));
                    require_same(name + ", dspark (auto, graphs): a 160-token greedy reply", plain_reply,
                                 result.generated_token_ids);
                    const infernix::SpeculativeStats& s = result.speculative;
                    std::cout << name << ", dspark (auto, graphs): " << s.rounds << " rounds, " << s.accepted_tokens
                              << " of " << s.drafted_tokens << " drafts accepted\n";
                }
                Tokens ten_k_reply;
                for (const std::uint32_t chunk : {4096U, 1000U}) {
                    infernix::EngineOptions options = engine_options(artifact, chunk, true, 1, kTenThousandContext);
                    set(options);
                    infernix::Engine engine(options);
                    const Tokens reply = engine.generate(engine.prepare_tokens(ten_k), greedy(32)).generated_token_ids;
                    if (chunk == 4096U) {
                        ten_k_reply = reply;
                        if (lossless) {
                            require_same(name + ": the 10,000-token prompt's greedy continuation = bf16's", bf16_ten_k, reply);
                        } else {
                            std::cout << name << ": the 10,000-token prompt's greedy continuation agrees with BF16's for "
                                      << agree(reply, bf16_ten_k) << " of 32 tokens (reported)\n";
                        }
                    } else {
                        require_same(name + ": the 10,000-token prompt in chunks of 1,000 = 4,096", ten_k_reply, reply);
                    }
                }
            }
        }
        {
            // ScoreOptions::argmax_only (the token-ranking tool's device argmax): each position's top_k = 1 token.
            infernix::EngineOptions options = engine_options(artifact, 16, false, 1);
            options.purpose                 = infernix::EnginePurpose::CausalScoring;
            infernix::Engine engine(options);
            infernix::ScoreOptions ranked, device;
            ranked.top_k       = 1;
            device.argmax_only = true;
            const Tokens by_host = engine.score_tokens(golden, 1, ranked).top_ids;
            require_same("scoring: argmax_only against top_k = 1", by_host, engine.score_tokens(golden, 1, device).top_ids);
        }
        if (guard != nullptr) {
            infernix::EngineOptions options = engine_options(artifact, 16, false, 1);
            options.purpose                 = infernix::EnginePurpose::CausalScoring;
            infernix::Engine engine(options);
            infernix::ScoreOptions score;
            for (std::uint32_t p = 0; p < 31; ++p) { score.logits_positions.push_back(p); }
            const auto result = engine.score_tokens(golden, 1, score);
            std::ifstream file(guard, std::ios::binary);
            std::vector<float> want(result.logits.size());
            file.read(reinterpret_cast<char*>(want.data()), static_cast<std::streamsize>(want.size() * 4));
            require(static_cast<bool>(file), "the guard file is shorter than 31 logits rows");
            std::size_t differ = 0;
            for (std::size_t i = 0; i < want.size(); ++i) { differ += std::memcmp(&want[i], &result.logits[i], 4) != 0 ? 1 : 0; }
            require(differ == 0, "scoring: " + std::to_string(differ) + " logits differ from the forward test's");
            std::cout << "scoring: 31 logits rows equal the forward test's prompt route byte for byte\n";
        } else {
            std::cout << "scoring skipped: INFERNIX_INKLING2_PROMPT_LOGITS_GUARD is not set\n";
        }
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    std::cout << "PASS\n";
    return 0;
}
