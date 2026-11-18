#include "http_loopback_harness.h"

#include <httplib.h>

#include <chrono>
#include <cstdint>
#include <functional>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

// Audit 2026-11-17 SV-07: a non-streamed request whose generation ends cancelled (the client left, or the
// server is stopping) over loopback HTTP without a model. Anthropic Messages has no cancelled message: it
// answers 499 request_cancelled, not a 500 counted as a render failure. Responses answers its cancelled
// Response but does not store it, so a retrieval (and a previous_response_id chain) cannot find a cut answer.
// Chat Completions answers the cut answer, and with --restore-reasoning does not remember its reasoning: the
// next turn, sending the cut answer back without reasoning, is not given it (an answer that finished is).
namespace {

using Json = nlohmann::json;
using namespace infernix::serve;

constexpr const char* kModel = "cancel-test-model";

int check(bool condition, const std::string& message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

// Answers `text` with `reasoning`, ending `finish` (cancelled unless a test changes it), and records the
// reasoning each prepared request's assistant messages carry (after the server's restore).
class CancelledBackend final : public test::LoopbackBackend {
public:
    CancelledBackend() : LoopbackBackend(kModel) {}

    std::string text                    = "a cut ans";
    std::string reasoning;
    infernix::FinishReason finish       = infernix::FinishReason::Cancelled;
    mutable std::vector<std::string> history_reasoning;

    PreparedRequest prepare(const GenerationRequest& request, GenerationConsumerMode mode,
                            infernix::GenerationObservationOptions observation,
                            std::function<bool()> cancelled) const override {
        history_reasoning.clear();
        for (const ChatTurn& turn : request.messages) {
            if (turn.role == infernix::ChatRole::Assistant) { history_reasoning.push_back(turn.reasoning_content); }
        }
        return LoopbackBackend::prepare(request, mode, std::move(observation), std::move(cancelled));
    }

    GenerationOutcome run(PreparedRequest& prepared, const StreamSink*, std::function<bool()>) override {
        GenerationOutcome outcome;
        outcome.prompt_tokens           = prepared.prompt_tokens;
        outcome.computed_prefill_tokens = static_cast<std::uint32_t>(prepared.prompt_tokens);
        outcome.text                    = text;
        outcome.reasoning               = reasoning;
        outcome.completion_tokens       = 3;
        outcome.finish_reason           = finish;
        return outcome;
    }
};

// One Chat turn answered `answer`, then the next turn sending it back without reasoning: the reasoning that
// next turn's assistant message carries after the server's restore.
std::string reasoning_given_back(const test::LoopbackServer& server, CancelledBackend& backend,
                                 const std::string& question, const std::string& answer,
                                 infernix::FinishReason finish, int& failures) {
    backend.text      = answer;
    backend.reasoning = "thought before " + answer;
    backend.finish    = finish;
    const Json first  = Json::array({Json{{"role", "user"}, {"content", question}}});
    const auto result = server.post("/v1/chat/completions", Json{{"model", kModel}, {"messages", first}});
    failures += check(result && result->status == 200, "Chat: the answer '" + answer + "' was not answered");
    backend.finish    = infernix::FinishReason::StopToken;
    Json next         = first;
    next.push_back(Json{{"role", "assistant"}, {"content", answer}});
    next.push_back(Json{{"role", "user"}, {"content", "and then?"}});
    const auto again = server.post("/v1/chat/completions", Json{{"model", kModel}, {"messages", next}});
    failures += check(again && again->status == 200, "Chat: the turn after '" + answer + "' was not answered");
    failures += check(backend.history_reasoning.size() == 1,
                      "Chat: the turn after '" + answer + "' did not reach the backend with one assistant message");
    return backend.history_reasoning.empty() ? std::string("<none>") : backend.history_reasoning.front();
}

} // namespace

int main() try {
    CancelledBackend backend;
    const test::LoopbackServer server(backend, std::chrono::seconds(30),
                                      [](ServeOptions& options) { options.restore_reasoning = true; });
    int failures = 0;

    const Json messages = Json::array({Json{{"role", "user"}, {"content", "hi"}}});
    {
        const auto result = server.post("/v1/messages", Json{{"model", kModel}, {"max_tokens", 16}, {"messages", messages}});
        failures += check(result && result->status == 499, "Anthropic: a cancelled message did not answer 499");
        if (result) {
            const Json body = Json::parse(result->body, nullptr, false);
            failures += check(body.is_object() && body.value("type", "") == "error" &&
                                  body["error"].value("type", "") == "request_cancelled",
                              "Anthropic: the answer is not a request_cancelled error: " + result->body);
        }
    }
    {
        const auto result = server.post("/v1/responses", Json{{"model", kModel}, {"input", "hi"}, {"store", true}});
        failures += check(result && result->status == 200, "Responses: a cancelled Response was not answered");
        if (result && result->status == 200) {
            const Json body = Json::parse(result->body, nullptr, false);
            failures += check(body.is_object() && body.value("status", "") == "cancelled",
                              "Responses: the answer's status is not cancelled: " + result->body);
            const std::string id = body.is_object() ? body.value("id", "") : "";
            httplib::Client client("127.0.0.1", server.port());
            const auto stored = client.Get("/v1/responses/" + id);
            failures += check(!id.empty() && stored && stored->status == 404,
                              "Responses: a cancelled Response was stored (retrieval answered " +
                                  std::to_string(stored ? stored->status : -1) + ")");
        }
    }
    {
        // The control: a finished answer's reasoning is remembered and given back, so the restore is live.
        const std::string kept = reasoning_given_back(server, backend, "hi", "a full answer",
                                                      infernix::FinishReason::StopToken, failures);
        failures += check(kept == "thought before a full answer",
                          "Chat: a finished answer's reasoning was not given back (got '" + kept + "')");
        const std::string cut = reasoning_given_back(server, backend, "hello", "a cut ans",
                                                     infernix::FinishReason::Cancelled, failures);
        failures += check(cut.empty(), "Chat: a cancelled answer's reasoning was given back: '" + cut + "'");
    }
    if (failures == 0) { std::cout << "cancelled non-streamed answers: ok\n"; }
    return failures == 0 ? 0 : 1;
} catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
}
