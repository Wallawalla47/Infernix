#include "http_loopback_harness.h"

#include <httplib.h>

#include <cstdint>
#include <functional>
#include <iostream>
#include <string>

// Audit 2026-11-17 SV-07: a non-streamed request whose generation ends cancelled (the client left, or the
// server is stopping) over loopback HTTP without a model. Anthropic Messages has no cancelled message: it
// answers 499 request_cancelled, not a 500 counted as a render failure. Responses answers its cancelled
// Response but does not store it, so a retrieval (and a previous_response_id chain) cannot find a cut answer.
// Chat Completions answers the cut answer.
namespace {

using Json = nlohmann::json;
using namespace infernix::serve;

constexpr const char* kModel = "cancel-test-model";

int check(bool condition, const std::string& message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

class CancelledBackend final : public test::LoopbackBackend {
public:
    CancelledBackend() : LoopbackBackend(kModel) {}

    GenerationOutcome run(PreparedRequest& prepared, const StreamSink*, std::function<bool()>) override {
        GenerationOutcome outcome;
        outcome.prompt_tokens           = prepared.prompt_tokens;
        outcome.computed_prefill_tokens = static_cast<std::uint32_t>(prepared.prompt_tokens);
        outcome.text                    = "a cut ans";
        outcome.completion_tokens       = 3;
        outcome.finish_reason           = infernix::FinishReason::Cancelled;
        return outcome;
    }
};

} // namespace

int main() try {
    CancelledBackend backend;
    const test::LoopbackServer server(backend);
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
        const auto result = server.post("/v1/chat/completions", Json{{"model", kModel}, {"messages", messages}});
        failures += check(result && result->status == 200, "Chat: the cut answer was not answered");
    }
    if (failures == 0) { std::cout << "cancelled non-streamed answers: ok\n"; }
    return failures == 0 ? 0 : 1;
} catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
}
