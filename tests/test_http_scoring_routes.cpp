// A --causal-scoring server over loopback HTTP: its generation routes refuse (501 generation_not_enabled) before
// anything runs, POST /v1/score answers through the backend, and a generation server refuses /v1/score (501
// scoring_not_enabled). No Engine: the backends stand in for GenerationService (the scores themselves are
// Engine::score_tokens', perplexity's path).
#include "http_loopback_harness.h"

#include <nlohmann/json.hpp>

#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using Json = nlohmann::json;
using namespace infernix::serve;
using infernix::serve::test::LoopbackBackend;
using infernix::serve::test::LoopbackServer;

constexpr const char* kModel = "scoring-test-model";

int failures = 0;

void check(bool condition, const std::string& message) {
    if (condition) { return; }
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
}

// A causal-scoring Engine's backend: nothing generates, and /v1/score returns one canned position.
class ScoringBackend final : public LoopbackBackend {
public:
    ScoringBackend() : LoopbackBackend(kModel) { options_.purpose = infernix::EnginePurpose::CausalScoring; }

    const infernix::EngineOptions& engine_options() const override { return options_; }

    GenerationOutcome run(PreparedRequest&, const StreamSink*, std::function<bool()>) override {
        ++generations;
        throw std::logic_error("a scoring server generated");
    }

    ScoreOutcome score(const ScoreRequest& request, std::function<bool()>) override {
        last_continuation = request.continuation;
        ScoreOutcome outcome;
        outcome.top_k         = request.top_k;
        outcome.prompt_tokens = 5;
        outcome.positions.push_back(ScoredPosition{
            .id = 11, .text = " Paris", .logprob = -0.25F, .top = {}});
        return outcome;
    }

    int generations = 0;
    std::string last_continuation;

private:
    infernix::EngineOptions options_;
};

// A generation server's backend: /v1/score keeps GenerationBackend's refusal.
class GeneratingBackend final : public LoopbackBackend {
public:
    GeneratingBackend() : LoopbackBackend(kModel) {}

    GenerationOutcome run(PreparedRequest&, const StreamSink*, std::function<bool()>) override {
        GenerationOutcome outcome;
        outcome.text              = "hello";
        outcome.prompt_tokens     = 8;
        outcome.completion_tokens = 1;
        return outcome;
    }
};

Json chat_body() {
    return Json{{"model", kModel}, {"max_tokens", 4}, {"messages", Json::array({{{"role", "user"}, {"content", "hi"}}})}};
}

Json score_body() {
    return Json{{"model", kModel}, {"apply_chat_template", false}, {"prompt", "The capital of France is"},
                {"continuation", " Paris"}, {"top_k", 0}};
}

Json body_of(const httplib::Result& result) {
    if (!result) { return Json(); }
    return Json::parse(result->body, nullptr, false);
}

void test_scoring_server() {
    ScoringBackend backend;
    const LoopbackServer server(backend);
    for (const char* path : {"/v1/chat/completions", "/v1/responses", "/v1/decide"}) {
        Json body = chat_body();
        if (std::string(path) == "/v1/responses") {
            body = Json{{"model", kModel}, {"input", "hi"}};
        } else if (std::string(path) == "/v1/decide") {
            body = Json{{"model", kModel},
                        {"questions", Json::array({{{"kind", "yes_no"}, {"question", "Is the sky blue?"}}})}};
        }
        const httplib::Result result = server.post(path, body);
        const Json answer            = body_of(result);
        check(result && result->status == 501, std::string(path) + " on a scoring server answers 501");
        check(answer.is_object() && answer.contains("error") &&
                  answer["error"].value("code", std::string()) == "generation_not_enabled",
              std::string(path) + " names generation_not_enabled");
    }
    {
        const httplib::Result result = server.post("/v1/messages", chat_body());
        const Json answer            = body_of(result);
        check(result && result->status == 501, "/v1/messages on a scoring server answers 501");
        check(answer.is_object() && answer.value("type", std::string()) == "error" &&
                  answer["error"].value("message", std::string()).find("--causal-scoring") != std::string::npos,
              "/v1/messages answers an Anthropic error naming --causal-scoring");
    }
    check(backend.generations == 0, "a scoring server ran no generation");

    const httplib::Result scored = server.post("/v1/score", score_body());
    const Json answer            = body_of(scored);
    check(scored && scored->status == 200, "/v1/score on a scoring server answers 200");
    check(answer.is_object() && answer.value("object", std::string()) == "score" &&
              answer["positions"].size() == 1 && answer["positions"][0]["token"] == " Paris" &&
              answer["usage"]["prompt_tokens"] == 5 && answer["usage"]["continuation_tokens"] == 1,
          "/v1/score returns the backend's positions and usage");
    check(backend.last_continuation == " Paris", "/v1/score passes the continuation through");

    Json unknown        = score_body();
    unknown["logprobs"] = true;
    const httplib::Result refused = server.post("/v1/score", unknown);
    check(refused && refused->status == 400 &&
              body_of(refused)["error"].value("code", std::string()) == "unknown_field",
          "/v1/score refuses an unknown field before scoring");
}

void test_generation_server() {
    GeneratingBackend backend;
    const LoopbackServer server(backend);
    const httplib::Result scored = server.post("/v1/score", score_body());
    check(scored && scored->status == 501 &&
              body_of(scored)["error"].value("code", std::string()) == "scoring_not_enabled",
          "/v1/score on a generation server answers 501 scoring_not_enabled");
    const httplib::Result chat = server.post("/v1/chat/completions", chat_body());
    check(chat && chat->status == 200, "a generation server still answers chat completions");
}

} // namespace

int main() {
    try {
        test_scoring_server();
        test_generation_server();
    } catch (const std::exception& exception) {
        std::cerr << "FAIL: " << exception.what() << '\n';
        return 1;
    }
    if (failures != 0) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "http scoring routes: all checks passed\n";
    return 0;
}
