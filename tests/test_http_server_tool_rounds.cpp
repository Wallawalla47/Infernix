// The server-side tool loops of Anthropic Messages (tool_search_tool_regex) and OpenAI Responses (the hosted
// tool_search), over loopback HTTP through the real routes, against a backend that scripts each generation and each
// continuation. Every way a loop ends: the search's generation followed by an answer; a search beside a client call;
// a search the output limit cut; the round limit (Anthropic pause_turn; Responses completed); Responses'
// max_tool_calls; a continuation with no output tokens left; a continuation over the context
// (context_length_exceeded) and one refused otherwise. One streamed answer per API covers the round events. A
// cancelled generation is a client disconnect on these routes (test_openai_responses' cancelled-stream case).
//
// With INFERNIX_TOOL_ROUNDS_DUMP set to a folder, each answer is also written there with its ids and times masked,
// for a byte comparison of two builds.

#include "http_loopback_harness.h"

#include "serve/http_server.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace {

using Json = nlohmann::json;
using namespace infernix::serve;

constexpr const char* kModel = "tool-rounds-test-model";

int check(bool condition, const std::string& message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

// How each continuation is answered.
enum class Continuation { Prepare, ContextFull, Refuse };

class ScriptedBackend final : public test::LoopbackBackend {
public:
    ScriptedBackend() : LoopbackBackend(kModel) {}

    // The next request's script: generation n's outcome, and how its continuations are answered.
    void script(std::function<GenerationOutcome(int)> outcome, Continuation continuation = Continuation::Prepare) {
        outcome_      = std::move(outcome);
        continuation_ = continuation;
        runs_         = 0;
        continuations_ = 0;
    }

    [[nodiscard]] int runs() const { return runs_.load(); }
    [[nodiscard]] int continuations() const { return continuations_.load(); }

    PreparedRequest prepare_continuation(const GenerationRequest&, GenerationConsumerMode,
                                         infernix::GenerationObservationOptions,
                                         std::function<bool()>) const override {
        ++continuations_;
        if (continuation_ == Continuation::ContextFull) {
            throw ApiException(ApiError{.status  = 400,
                                        .type    = "invalid_request_error",
                                        .message = "the continuation exceeds the context",
                                        .code    = "context_length_exceeded"});
        }
        if (continuation_ == Continuation::Refuse) {
            throw ApiException(ApiError{.status = 400, .message = "refused by the test", .code = "invalid_value"});
        }
        PreparedRequest prepared;
        prepared.prompt_tokens = 8;
        return prepared;
    }

    GenerationOutcome run(PreparedRequest& prepared, const StreamSink* sink, std::function<bool()>) override {
        GenerationOutcome outcome = outcome_(runs_++);
        if (sink != nullptr) {
            if (sink->on_start) {
                infernix::GenerationStart start;
                start.prompt.prompt_tokens = static_cast<std::uint32_t>(prepared.prompt_tokens);
                sink->on_start(start);
            }
            if (sink->on_reasoning && !outcome.reasoning.empty()) { sink->on_reasoning(outcome.reasoning); }
            if (sink->on_content && !outcome.text.empty()) { sink->on_content(outcome.text); }
            for (std::size_t index = 0; index < outcome.tool_calls.size(); ++index) {
                if (sink->on_tool_call) {
                    sink->on_tool_call(infernix::ToolCallDelta{.index     = static_cast<std::uint32_t>(index),
                                                               .name      = outcome.tool_calls[index].name,
                                                               .arguments = outcome.tool_calls[index].arguments_json,
                                                               .complete  = true});
                }
            }
        }
        return outcome;
    }

private:
    std::function<GenerationOutcome(int)> outcome_;
    Continuation continuation_ = Continuation::Prepare;
    mutable std::atomic<int> runs_{0};
    mutable std::atomic<int> continuations_{0};
};

GenerationOutcome generation(std::vector<infernix::GeneratedToolCall> calls, std::string text = {},
                             infernix::FinishReason finish = infernix::FinishReason::StopToken) {
    GenerationOutcome outcome;
    outcome.reasoning         = "I need a tool.";
    outcome.text              = std::move(text);
    outcome.prompt_tokens     = 40;
    outcome.completion_tokens = 20;
    outcome.finish_reason     = finish;
    outcome.tool_calls        = std::move(calls);
    return outcome;
}

// ---- masking for the dumps ------------------------------------------------------------------------

void mask(Json& value) {
    static const std::set<std::string> keys{"id",         "call_id",      "tool_use_id", "item_id", "request_id",
                                            "created_at", "completed_at", "created",     "response_id",
                                            "signature"};
    if (value.is_object()) {
        for (auto it = value.begin(); it != value.end(); ++it) {
            if (keys.contains(it.key()) && (it->is_string() || it->is_number())) {
                *it = "<masked>";
            } else {
                mask(*it);
            }
        }
    } else if (value.is_array()) {
        for (Json& item : value) { mask(item); }
    }
}

void dump(const std::string& name, const std::string& body, bool stream) {
    const char* folder = std::getenv("INFERNIX_TOOL_ROUNDS_DUMP");
    if (folder == nullptr) { return; }
    std::ofstream out(std::string(folder) + "/" + name + ".txt", std::ios::binary);
    if (!stream) {
        Json value = Json::parse(body, nullptr, false);
        if (value.is_discarded()) {
            out << body;
            return;
        }
        mask(value);
        out << value.dump(1);
        return;
    }
    for (const test::SseFrame& frame : test::sse_frames(body)) {
        Json value = Json::parse(frame.data, nullptr, false);
        if (!value.is_discarded()) { mask(value); }
        out << frame.event << ' ' << (value.is_discarded() ? frame.data : value.dump()) << '\n';
    }
}

std::vector<std::string> types(const Json& items) {
    std::vector<std::string> out;
    for (const Json& item : items) { out.push_back(item.value("type", "")); }
    return out;
}

std::size_t count(const std::vector<std::string>& values, const std::string& value) {
    return static_cast<std::size_t>(std::count(values.begin(), values.end(), value));
}

// ---- Anthropic Messages ---------------------------------------------------------------------------

Json anthropic_body(int max_tokens = 4096) {
    const auto deferred = [](const char* name) {
        return Json{{"name", name},
                    {"description", std::string("The ") + name + " tool."},
                    {"input_schema", Json{{"type", "object"}}},
                    {"defer_loading", true}};
    };
    return Json{{"model", kModel},
                {"max_tokens", max_tokens},
                {"tools", Json::array({Json{{"name", "get_weather"},
                                            {"description", "Current weather."},
                                            {"input_schema", Json{{"type", "object"}}}},
                                       Json{{"type", "tool_search_tool_regex_20251119"}, {"name", "tool_search_tool_regex"}},
                                       deferred("get_forecast"), deferred("get_tides")})},
                {"messages", Json::array({Json{{"role", "user"}, {"content", "Forecast for Paris?"}}})}};
}

const infernix::GeneratedToolCall kAnthropicSearch{.name = "tool_search_tool_regex", .arguments_json = R"({"pattern":"forecast"})"};
const infernix::GeneratedToolCall kAnthropicClient{.name = "get_forecast", .arguments_json = R"({"city":"Paris"})"};

int test_anthropic(ScriptedBackend& backend, const test::LoopbackServer& server) {
    int failures       = 0;
    const auto request = [&](const std::string& name, const Json& body, int expected_status = 200) -> Json {
        const auto result = server.post("/v1/messages", body);
        if (!result) {
            ++failures;
            std::cerr << "FAIL: anthropic " << name << ": no response\n";
            return Json();
        }
        dump("anthropic_" + name, result->body, false);
        failures += check(result->status == expected_status,
                          "anthropic " + name + ": HTTP " + std::to_string(result->status) + " " + result->body);
        return Json::parse(result->body, nullptr, false);
    };

    // A search, then the answer: two generations, one continuation, the search and its result before the text.
    backend.script([](int n) { return n == 0 ? generation({kAnthropicSearch}) : generation({}, "Sunny."); });
    Json answer = request("search_then_answer", anthropic_body());
    auto blocks = types(answer["content"]);
    failures += check(answer["stop_reason"] == "end_turn" && backend.runs() == 2 && backend.continuations() == 1 &&
                          count(blocks, "server_tool_use") == 1 && count(blocks, "tool_search_tool_result") == 1 &&
                          !blocks.empty() && blocks.back() == "text",
                      "anthropic: a search followed by an answer " + answer.dump());

    // A search beside a client call: the search runs, the turn ends at the client's call.
    backend.script([](int) { return generation({kAnthropicSearch, kAnthropicClient}); });
    answer = request("search_and_client", anthropic_body());
    blocks = types(answer["content"]);
    failures += check(answer["stop_reason"] == "tool_use" && backend.runs() == 1 && backend.continuations() == 0 &&
                          count(blocks, "tool_search_tool_result") == 1 && count(blocks, "tool_use") == 1,
                      "anthropic: a search beside a client call " + answer.dump());

    // A search the output limit cut: it runs, nothing continues.
    backend.script([](int) { return generation({kAnthropicSearch}, {}, infernix::FinishReason::OutputLimit); });
    answer = request("search_cut", anthropic_body());
    blocks = types(answer["content"]);
    failures += check(answer["stop_reason"] == "max_tokens" && backend.runs() == 1 && backend.continuations() == 0 &&
                          count(blocks, "tool_search_tool_result") == 1,
                      "anthropic: a search the output limit cut " + answer.dump());

    // Searches only: the response pauses at the round limit.
    backend.script([](int) { return generation({kAnthropicSearch}); });
    answer = request("round_limit", anthropic_body());
    failures += check(answer["stop_reason"] == "pause_turn" && backend.runs() == 10 && backend.continuations() == 9,
                      "anthropic: the round limit " + answer.dump().substr(0, 400));

    // No output tokens left for the continuation (max_tokens = the search's 20 tokens).
    backend.script([](int) { return generation({kAnthropicSearch}); });
    answer = request("no_tokens_left", anthropic_body(20));
    failures += check(answer["stop_reason"] == "max_tokens" && backend.runs() == 1 && backend.continuations() == 0,
                      "anthropic: no output tokens left " + answer.dump());

    // The continuation is over the context.
    backend.script([](int) { return generation({kAnthropicSearch}); }, Continuation::ContextFull);
    answer = request("context_full", anthropic_body());
    failures += check(answer["stop_reason"] == "model_context_window_exceeded" && backend.runs() == 1 &&
                          backend.continuations() == 1,
                      "anthropic: a continuation over the context " + answer.dump());

    // Any other refusal of the continuation fails the request.
    backend.script([](int) { return generation({kAnthropicSearch}); }, Continuation::Refuse);
    answer = request("continuation_refused", anthropic_body(), 400);
    failures += check(backend.runs() == 1 && backend.continuations() == 1,
                      "anthropic: a refused continuation " + answer.dump());

    // Streamed: the search's round is sent as it completes, then the answer; the stream ends with message_stop.
    backend.script([](int n) { return n == 0 ? generation({kAnthropicSearch}) : generation({}, "Sunny."); });
    Json body      = anthropic_body();
    body["stream"] = true;
    const auto streamed = server.post("/v1/messages", body);
    if (!streamed) {
        failures += check(false, "anthropic: no streamed response");
    } else {
        dump("anthropic_stream", streamed->body, true);
        const auto frames = test::sse_frames(streamed->body);
        failures += check(streamed->status == 200 && !frames.empty() && frames.back().event == "message_stop" &&
                              backend.runs() == 2 && streamed->body.find("tool_search_tool_result") != std::string::npos,
                          "anthropic: a streamed search and answer " + streamed->body.substr(0, 400));
    }
    return failures;
}

// ---- OpenAI Responses -----------------------------------------------------------------------------

Json responses_body(std::optional<int> max_output_tokens = std::nullopt, std::optional<int> max_tool_calls = std::nullopt) {
    const Json crm = {{"type", "namespace"},
                      {"name", "crm"},
                      {"description", "CRM tools for customer lookup."},
                      {"tools", Json::array({Json{{"type", "function"}, {"name", "get_customer"},
                                                  {"parameters", Json{{"type", "object"}}}},
                                             Json{{"type", "function"},
                                                  {"name", "list_open_orders"},
                                                  {"description", "List open orders for a customer."},
                                                  {"parameters", Json{{"type", "object"}}},
                                                  {"defer_loading", true}}})}};
    Json body{{"model", kModel}, {"input", "Which orders are open?"},
              {"tools", Json::array({Json{{"type", "tool_search"}}, crm})}, {"store", false}};
    if (max_output_tokens) { body["max_output_tokens"] = *max_output_tokens; }
    if (max_tool_calls) { body["max_tool_calls"] = *max_tool_calls; }
    return body;
}

const infernix::GeneratedToolCall kResponsesSearch{.name = "tool_search", .arguments_json = R"({"paths":["crm"]})"};
const infernix::GeneratedToolCall kResponsesClient{.name = "crm__list_open_orders", .arguments_json = R"({"customer":"c1"})"};

int test_responses(ScriptedBackend& backend, const test::LoopbackServer& server) {
    int failures       = 0;
    const auto request = [&](const std::string& name, const Json& body, int expected_status = 200) -> Json {
        const auto result = server.post("/v1/responses", body);
        if (!result) {
            ++failures;
            std::cerr << "FAIL: responses " << name << ": no response\n";
            return Json();
        }
        dump("responses_" + name, result->body, false);
        failures += check(result->status == expected_status,
                          "responses " + name + ": HTTP " + std::to_string(result->status) + " " + result->body);
        return Json::parse(result->body, nullptr, false);
    };

    backend.script([](int n) { return n == 0 ? generation({kResponsesSearch}) : generation({}, "Two orders."); });
    Json answer = request("search_then_answer", responses_body());
    auto items  = types(answer["output"]);
    failures += check(answer["status"] == "completed" && backend.runs() == 2 && backend.continuations() == 1 &&
                          count(items, "tool_search_call") == 1 && count(items, "tool_search_output") == 1 &&
                          !items.empty() && items.back() == "message",
                      "responses: a search followed by an answer " + answer.dump());

    backend.script([](int) { return generation({kResponsesSearch, kResponsesClient}); });
    answer = request("search_and_client", responses_body());
    items  = types(answer["output"]);
    failures += check(answer["status"] == "completed" && backend.runs() == 1 && backend.continuations() == 0 &&
                          count(items, "tool_search_output") == 1 && count(items, "function_call") == 1,
                      "responses: a search beside a client call " + answer.dump());

    backend.script([](int) { return generation({kResponsesSearch}, {}, infernix::FinishReason::OutputLimit); });
    answer = request("search_cut", responses_body());
    items  = types(answer["output"]);
    failures += check(answer["status"] == "incomplete" && backend.runs() == 1 && backend.continuations() == 0 &&
                          count(items, "tool_search_output") == 1,
                      "responses: a search the output limit cut " + answer.dump());

    // The round limit ends the response as completed (the last generation's search shown, nothing more).
    backend.script([](int) { return generation({kResponsesSearch}); });
    answer = request("round_limit", responses_body());
    failures += check(answer["status"] == "completed" && backend.runs() == 10 && backend.continuations() == 9,
                      "responses: the round limit " + answer.dump().substr(0, 400));

    backend.script([](int) { return generation({kResponsesSearch}); });
    answer = request("max_tool_calls", responses_body(std::nullopt, 1));
    items  = types(answer["output"]);
    failures += check(answer["status"] == "completed" && backend.runs() == 1 && backend.continuations() == 0 &&
                          count(items, "tool_search_output") == 1,
                      "responses: max_tool_calls " + answer.dump());

    backend.script([](int) { return generation({kResponsesSearch}); });
    answer = request("no_tokens_left", responses_body(20));
    failures += check(answer["status"] == "incomplete" && backend.runs() == 1 && backend.continuations() == 0,
                      "responses: no output tokens left " + answer.dump());

    backend.script([](int) { return generation({kResponsesSearch}); }, Continuation::ContextFull);
    answer = request("context_full", responses_body());
    failures += check(answer["status"] == "incomplete" && backend.runs() == 1 && backend.continuations() == 1,
                      "responses: a continuation over the context " + answer.dump());

    backend.script([](int) { return generation({kResponsesSearch}); }, Continuation::Refuse);
    answer = request("continuation_refused", responses_body(), 400);
    failures += check(backend.runs() == 1 && backend.continuations() == 1,
                      "responses: a refused continuation " + answer.dump());

    backend.script([](int n) { return n == 0 ? generation({kResponsesSearch}) : generation({}, "Two orders."); });
    Json body      = responses_body();
    body["stream"] = true;
    const auto streamed = server.post("/v1/responses", body);
    if (!streamed) {
        failures += check(false, "responses: no streamed response");
    } else {
        dump("responses_stream", streamed->body, true);
        const auto frames = test::sse_frames(streamed->body);
        failures += check(streamed->status == 200 && !frames.empty() && frames.back().event == "response.completed" &&
                              backend.runs() == 2 && streamed->body.find("tool_search_output") != std::string::npos,
                          "responses: a streamed search and answer " + streamed->body.substr(0, 400));
    }
    return failures;
}

} // namespace

int main() try {
    ScriptedBackend backend;
    const test::LoopbackServer server(backend);
    int failures = 0;
    failures += test_anthropic(backend, server);
    failures += test_responses(backend, server);
    if (failures != 0) {
        std::cerr << failures << " server tool round check(s) failed\n";
        return 1;
    }
    std::cout << "server tool rounds: all checks passed\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
}
