// Inkling 2 through the public Engine on a real artifact (the Program's first slice, src/models/inkling2/program).
// The Engine without CUDA graphs, prefilling 16 columns a call and serving one request at a time, is the reference;
// every other configuration must give its tokens:
// 1. Greedy: the golden prompt of the forward test's reference dump (its first 24 ids) continues with the dump's next
//    8 ids (the forward's free-running argmax at positions 23-30, which the bit guard's logits confirm).
// 2. Graphs: the Engine with CUDA graphs gives the reference's tokens.
// 3. Request invariance: four sampled requests (temperature 1, a seed each, 24-48 tokens so lanes drop out mid-run in a
//    known order) give their solo tokens all at once in reverse order (C = 4) and twice over at once (C = 8).
// 4. Prefill chunk width: an Engine prefilling 7 columns a call gives the reference's tokens.
// 5. Scoring: a CausalScoring Engine's logits of the 32 golden ids (positions 0-30) equal the forward test's
//    free-running logits (INFERNIX_INKLING2_LOGITS_GUARD) byte for byte.
// Reported, not judged: a chat turn's channels, and a turn whose thinking budget (6) injects the control (the
// mixed-width decode after forced tokens).
//
//   INFERNIX_INKLING2_ARTIFACT      an artifact converted with the inkling2_mini recipe
//   INFERNIX_INKLING2_REFERENCE     the reference dump's path prefix (<prefix>.bin and <prefix>.json: the golden ids)
//   INFERNIX_INKLING2_LOGITS_GUARD  the forward test's free-running logits ([32][V] FP32), for 5

#include "infernix/engine.h"

#include <nlohmann/json.hpp>

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
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
                                       std::uint32_t concurrency) {
    infernix::EngineOptions options;
    options.artifact_path        = artifact;
    options.max_context          = kContext;
    options.kv_capacity          = infernix::KvCapacityPolicy::explicit_capacity(concurrency * kContext);
    options.max_concurrency      = concurrency;
    options.max_pending_requests = 16;
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

} // namespace

int main() {
    const char* artifact  = std::getenv("INFERNIX_INKLING2_ARTIFACT");
    const char* reference = std::getenv("INFERNIX_INKLING2_REFERENCE");
    const char* guard     = std::getenv("INFERNIX_INKLING2_LOGITS_GUARD");
    if (artifact == nullptr || reference == nullptr) {
        std::cout << "skip: set INFERNIX_INKLING2_ARTIFACT and INFERNIX_INKLING2_REFERENCE\n";
        return 77;
    }
    try {
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
            std::cout << "scoring: 31 logits rows equal the forward test's byte for byte\n";
        } else {
            std::cout << "scoring skipped: INFERNIX_INKLING2_LOGITS_GUARD is not set\n";
        }
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    std::cout << "PASS\n";
    return 0;
}
