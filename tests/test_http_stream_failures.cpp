// How the streaming routes (Chat Completions, Anthropic Messages, Responses) end a stream that fails or is cut, over
// loopback HTTP through the real routes, against a backend that scripts each generation's events and its ending: an
// answer; an API error and an unexpected exception after streamed text; an API error before any event; an event the
// protocol's encoder refuses (a render failure during the generation); terminal output that contradicts what was
// streamed (a render failure of the terminal events); and a cancelled generation. Each case checks how the stream
// ends (its terminal events, the protocol's error event, or cut) and the request's operational log records: done,
// failed or cancelled, and a response failure, with their phases. Refusals before a stream starts are
// test_http_stream_refusals'.
//
// With INFERNIX_STREAM_FAILURES_DUMP set to a folder, each stream and its log records are also written there with
// ids and times masked, for a byte comparison of two builds.

#include "http_loopback_harness.h"

#include "serve/http_server.h"

#include <httplib.h>
#include <nlohmann/json.hpp>
#include <spdlog/logger.h>
#include <spdlog/sinks/base_sink.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using Json = nlohmann::json;
using namespace infernix::serve;

constexpr const char* kModel = "stream-failures-test-model";
constexpr std::chrono::seconds kTimeout{30};

int check(bool condition, const std::string& message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

// The operational log's request records: "done", "failed during PHASE", "cancelled during PHASE" and "response failed
// during PHASE", each without its request number (a done record keeps its protocol, not its timings).
class RecordSink final : public spdlog::sinks::base_sink<std::mutex> {
public:
    std::vector<std::string> take() {
        std::lock_guard lock(records_mutex_);
        return std::exchange(records_, {});
    }

protected:
    void sink_it_(const spdlog::details::log_msg& message) override {
        const std::string line(message.payload.data(), message.payload.size());
        if (!line.starts_with("req#")) { return; }
        const std::size_t space = line.find(' ');
        if (space == std::string::npos) { return; }
        std::string record = line.substr(space + 1);
        if (record.starts_with("done | ")) {
            record = record.substr(0, record.find(" | ", 7));
        } else if (!record.starts_with("failed during ") && !record.starts_with("cancelled during ") &&
                   !record.starts_with("response failed during ")) {
            return;
        }
        std::lock_guard lock(records_mutex_);
        records_.push_back(std::move(record));
    }
    void flush_() override {}

private:
    std::mutex records_mutex_;
    std::vector<std::string> records_;
};

// One generation: what it does with the route's sink, and the outcome it returns (or what it throws).
using Script = std::function<GenerationOutcome(const StreamSink&, int prompt_tokens)>;

class ScriptedBackend final : public test::LoopbackBackend {
public:
    ScriptedBackend() : LoopbackBackend(kModel) {}

    void script(Script script) {
        std::lock_guard lock(mutex_);
        script_ = std::move(script);
    }

    GenerationOutcome run(PreparedRequest& prepared, const StreamSink* sink, std::function<bool()>) override {
        Script script;
        {
            std::lock_guard lock(mutex_);
            script = script_;
        }
        if (sink == nullptr) { throw std::logic_error("a streaming test generation ran without a sink"); }
        return script(*sink, prepared.prompt_tokens);
    }

private:
    std::mutex mutex_;
    Script script_;
};

void start(const StreamSink& sink, int prompt_tokens) {
    if (!sink.on_start) { return; }
    infernix::GenerationStart start;
    start.prompt.prompt_tokens = static_cast<std::uint32_t>(prompt_tokens);
    sink.on_start(start);
}

void content(const StreamSink& sink, const std::string& text) {
    if (sink.on_content) { sink.on_content(text); }
}

GenerationOutcome answer(std::string text, infernix::FinishReason finish = infernix::FinishReason::StopToken) {
    GenerationOutcome outcome;
    outcome.text              = std::move(text);
    outcome.prompt_tokens     = 8;
    outcome.completion_tokens = 4;
    outcome.finish_reason     = finish;
    return outcome;
}

ApiError engine_error() {
    return ApiError{.status = 503, .type = "server_error", .message = "the engine is stopping", .code = "server_stopping"};
}

// How a stream ends: its protocol's terminal events, its protocol's error event, or cut without either.
enum class Ending { Terminal, Error, Cut };

struct Case {
    const char* name;
    Script script;
    Ending ending;
    std::vector<std::string> records;
};

std::vector<Case> cases() {
    return {
        {"answer",
         [](const StreamSink& sink, int prompt) {
             start(sink, prompt);
             content(sink, "Hello");
             return answer("Hello");
         },
         Ending::Terminal, {"done"}},
        {"api_error_after_text",
         [](const StreamSink& sink, int prompt) -> GenerationOutcome {
             start(sink, prompt);
             content(sink, "Hel");
             throw ApiException(engine_error());
         },
         Ending::Error, {"failed during generation"}},
        {"exception_after_text",
         [](const StreamSink& sink, int prompt) -> GenerationOutcome {
             start(sink, prompt);
             content(sink, "Hel");
             throw std::runtime_error("the backend broke");
         },
         Ending::Error, {"failed during generation"}},
        {"api_error_before_events",
         [](const StreamSink&, int) -> GenerationOutcome { throw ApiException(engine_error()); }, Ending::Error,
         {"failed during generation"}},
        // A call delta for a call that was never opened: every encoder refuses it.
        {"refused_event",
         [](const StreamSink& sink, int prompt) -> GenerationOutcome {
             start(sink, prompt);
             content(sink, "Hel");
             if (sink.on_tool_call) {
                 sink.on_tool_call(infernix::ToolCallDelta{.index = 3, .name = {}, .arguments = "{}", .complete = true});
             }
             return answer("Hel");
         },
         Ending::Error, {"failed during response rendering"}},
        // The outcome's text does not continue what was streamed: the terminal events cannot be rendered.
        {"contradicting_terminal",
         [](const StreamSink& sink, int prompt) {
             start(sink, prompt);
             content(sink, "Hello");
             return answer("Bye.");
         },
         Ending::Error, {"done", "response failed during response rendering"}},
        {"cancelled",
         [](const StreamSink& sink, int prompt) {
             start(sink, prompt);
             content(sink, "Hel");
             return answer("Hel", infernix::FinishReason::Cancelled);
         },
         Ending::Cut, {"done", "response failed during transport"}},
    };
}

struct Route {
    const char* name;
    const char* path;
    Json body;
};

std::vector<Route> routes() {
    const Json user = Json::array({Json{{"role", "user"}, {"content", "hello"}}});
    return {
        {"chat", "/v1/chat/completions", Json{{"model", kModel}, {"messages", user}, {"stream", true}}},
        {"anthropic", "/v1/messages",
         Json{{"model", kModel}, {"max_tokens", 16}, {"messages", user}, {"stream", true}}},
        {"responses", "/v1/responses",
         Json{{"model", kModel}, {"input", "hello"}, {"stream", true}, {"temperature", 0.5}}},
    };
}

// The body as received, even when the stream is cut (the HTTP result is then an error).
struct Received {
    bool complete = false;
    int status    = 0;
    std::string body;
};

Received post_stream(int port, const Route& route) {
    httplib::Client client("127.0.0.1", port);
    client.set_connection_timeout(kTimeout);
    client.set_read_timeout(kTimeout);
    Received received;
    const httplib::Result result =
        client.Post(route.path, httplib::Headers{}, route.body.dump(), "application/json",
                    [&](const char* data, std::size_t size) {
                        received.body.append(data, size);
                        return true;
                    });
    received.complete = static_cast<bool>(result);
    if (result) { received.status = result->status; }
    return received;
}

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

void dump(const std::string& name, const Received& received, const std::vector<test::SseFrame>& frames,
          const std::vector<std::string>& records) {
    const char* folder = std::getenv("INFERNIX_STREAM_FAILURES_DUMP");
    if (folder == nullptr) { return; }
    std::ofstream out(std::string(folder) + "/" + name + ".txt", std::ios::binary);
    out << "complete " << received.complete << " status " << received.status << '\n';
    for (const test::SseFrame& frame : frames) {
        Json value = Json::parse(frame.data, nullptr, false);
        if (!value.is_discarded()) { mask(value); }
        out << frame.event << ' ' << (value.is_discarded() ? frame.data : value.dump()) << '\n';
    }
    for (const std::string& record : records) { out << "log " << record << '\n'; }
}

// The stream's last event as its protocol's terminal or error event.
bool terminal_event(const std::string& route, const test::SseFrame& frame) {
    if (route == "chat") { return frame.event.empty() && frame.data == "[DONE]"; }
    if (route == "anthropic") { return frame.event == "message_stop"; }
    return frame.event == "response.completed";
}

bool error_event(const std::string& route, const test::SseFrame& frame) {
    if (route == "chat") {
        const Json value = Json::parse(frame.data, nullptr, false);
        return frame.event.empty() && value.is_object() && value.contains("error");
    }
    if (route == "anthropic") { return frame.event == "error"; }
    return frame.event == "response.failed";
}

int test_route(ScriptedBackend& backend, const test::LoopbackServer& server, RecordSink& records,
               const Route& route) {
    int failures = 0;
    for (const Case& scenario : cases()) {
        const std::string what = std::string(route.name) + " " + scenario.name;
        backend.script(scenario.script);
        (void)records.take();
        const Received received = post_stream(server.port(), route);
        std::vector<test::SseFrame> frames;
        try {
            frames = test::sse_frames(received.body);
        } catch (const std::exception& error) {
            failures += check(false, what + ": " + error.what() + ": " + received.body);
            continue;
        }
        const std::vector<std::string> logged = records.take();
        dump(std::string(route.name) + "_" + scenario.name, received, frames, logged);

        failures += check(!frames.empty(), what + ": no event");
        if (frames.empty()) { continue; }
        const test::SseFrame& last = frames.back();
        switch (scenario.ending) {
        case Ending::Terminal:
            failures += check(received.complete && received.status == 200 && terminal_event(route.name, last),
                              what + ": the stream did not end with its terminal events: " + received.body);
            break;
        case Ending::Error:
            failures += check(received.complete && received.status == 200 && error_event(route.name, last),
                              what + ": the stream did not end with its error event: " + received.body);
            break;
        case Ending::Cut:
            failures += check(!received.complete && !terminal_event(route.name, last) && !error_event(route.name, last),
                              what + ": the cut stream ended as if answered: " + received.body);
            break;
        }
        std::string logged_text;
        for (const std::string& record : logged) { logged_text += "[" + record + "] "; }
        failures += check(logged.size() == scenario.records.size(), what + ": log records " + logged_text);
        for (std::size_t index = 0; index < scenario.records.size() && index < logged.size(); ++index) {
            failures += check(logged[index].starts_with(scenario.records[index]),
                              what + ": log record " + std::to_string(index) + " " + logged_text);
        }
        // An error before any event: Anthropic starts the message first, a Response is created first, and Chat
        // Completions (whose role chunk opens with the first event) sends the error alone.
        if (std::string(scenario.name) == "api_error_before_events") {
            const std::string& first = frames.front().event;
            failures += check(std::string(route.name) == "chat"        ? frames.size() == 1U
                              : std::string(route.name) == "anthropic" ? first == "message_start"
                                                                       : first == "response.created",
                              what + ": the error's opening " + received.body);
        }
    }
    return failures;
}

} // namespace

int main() try {
    ScriptedBackend backend;
    auto records = std::make_shared<RecordSink>();
    const test::LoopbackServer server(backend, kTimeout, {},
                                      std::make_shared<spdlog::logger>("stream-failures", records));
    int failures = 0;
    for (const Route& route : routes()) { failures += test_route(backend, server, *records, route); }
    if (failures != 0) {
        std::cerr << failures << " stream failure check(s) failed\n";
        return 1;
    }
    std::cout << "stream failures: all checks passed\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
}
