#include "http_loopback_harness.h"

#include <chrono>
#include <cstdint>
#include <functional>
#include <iostream>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

// A streamed reply whose only generated token is the lead byte of a multi-byte character (ZD-04;
// llama-server's stream never ended on that shape). The Engine's output session holds the byte back
// as an incomplete character and, at the output limit, publishes U+FFFD in its place
// (test_frontend.cpp covers that); this drives what the route then receives, over loopback HTTP on
// every streaming API: that one U+FFFD delta and the terminal result, or, as a bound, a terminal
// result with no delta at all. Each stream must carry the text and its protocol's terminal events
// for the output limit, and end: the client's read returns long before its timeout, and the next
// request on the same client is answered.
namespace {

using Json = nlohmann::json;
using namespace infernix::serve;
using test::LoopbackServer;
using test::SseFrame;
using Clock = std::chrono::steady_clock;

constexpr const char* kModel       = "lone-token-test-model";
constexpr const char* kReplacement = "\xef\xbf\xbd"; // U+FFFD

int check(bool condition, const std::string& message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

// One generated token, cut at the output limit of one: the U+FFFD delta, or none.
class LoneTokenBackend final : public test::LoopbackBackend {
public:
    LoneTokenBackend() : LoopbackBackend(kModel) {}

    void publish_delta(bool value) {
        std::lock_guard lock(mutex_);
        publish_delta_ = value;
    }

    GenerationOutcome run(PreparedRequest& prepared, const StreamSink* sink,
                          std::function<bool()>) override {
        const bool delta = publishes_delta();
        if (sink != nullptr) {
            if (sink->on_start) {
                sink->on_start(infernix::GenerationStart{
                    .prompt = infernix::PromptSummary{.prompt_tokens = static_cast<std::uint32_t>(
                                                          prepared.prompt_tokens)}});
            }
            if (delta && sink->on_content) { sink->on_content(kReplacement); }
        }
        GenerationOutcome outcome;
        outcome.text                    = delta ? kReplacement : "";
        outcome.prompt_tokens           = prepared.prompt_tokens;
        outcome.completion_tokens       = 1;
        outcome.computed_prefill_tokens = static_cast<std::uint32_t>(prepared.prompt_tokens);
        outcome.finish_reason           = infernix::FinishReason::OutputLimit;
        return outcome;
    }

private:
    bool publishes_delta() const {
        std::lock_guard lock(mutex_);
        return publish_delta_;
    }

    mutable std::mutex mutex_;
    bool publish_delta_ = true;
};

struct Streamed {
    std::vector<SseFrame> frames;
    double seconds = 0.0;
};

// Posts a streaming request; the answer must be a 200 event stream that ended on its own.
std::string stream(const LoopbackServer& server, std::string_view path, const Json& body,
                   Streamed& streamed) {
    const Clock::time_point start = Clock::now();
    const httplib::Result result  = server.post(path, body);
    streamed.seconds              = std::chrono::duration<double>(Clock::now() - start).count();
    if (!result) { return "no HTTP answer (" + httplib::to_string(result.error()) + ")"; }
    if (result->status != 200 ||
        result->get_header_value("Content-Type").find("text/event-stream") == std::string::npos) {
        return "not a 200 event stream (status " + std::to_string(result->status) + ")";
    }
    try {
        streamed.frames = test::sse_frames(result->body);
    } catch (const std::exception& error) { return error.what(); }
    return {};
}

int test_chat(const LoopbackServer& server, bool delta, const std::string& what) {
    const Json body{{"model", kModel},
                    {"stream", true},
                    {"max_tokens", 1},
                    {"messages", Json::array({Json{{"role", "user"}, {"content", "hi"}}})}};
    Streamed streamed;
    if (const std::string error = stream(server, "/v1/chat/completions", body, streamed); !error.empty()) {
        return check(false, what + " chat: " + error);
    }
    std::string text;
    std::string finish;
    for (std::size_t index = 0; index + 1 < streamed.frames.size(); ++index) {
        const Json chunk = Json::parse(streamed.frames[index].data);
        for (const Json& choice : chunk.value("choices", Json::array())) {
            text += choice.value("delta", Json::object()).value("content", "");
            if (choice.contains("finish_reason") && choice["finish_reason"].is_string()) {
                finish = choice["finish_reason"];
            }
        }
    }
    int failures = check(!streamed.frames.empty() && streamed.frames.back().data == "[DONE]",
                         what + " chat: the stream does not end with [DONE]");
    failures += check(finish == "length", what + " chat: no finish_reason length, got '" + finish + "'");
    failures += check(text == (delta ? kReplacement : ""), what + " chat: the text is not the token's U+FFFD");
    failures += check(streamed.seconds < 10.0, what + " chat: the stream took " +
                                                   std::to_string(streamed.seconds) + " s to end");
    return failures;
}

int test_responses(const LoopbackServer& server, bool delta, const std::string& what) {
    const Json body{{"model", kModel}, {"stream", true}, {"max_output_tokens", 16}, {"input", "hi"}};
    Streamed streamed;
    if (const std::string error = stream(server, "/v1/responses", body, streamed); !error.empty()) {
        return check(false, what + " responses: " + error);
    }
    std::string text;
    for (const SseFrame& frame : streamed.frames) {
        if (frame.event == "response.output_text.delta") { text += Json::parse(frame.data).value("delta", ""); }
    }
    const SseFrame* last = streamed.frames.empty() ? nullptr : &streamed.frames.back();
    const Json terminal  = last != nullptr ? Json::parse(last->data) : Json::object();
    int failures         = check(last != nullptr && last->event == "response.incomplete" &&
                                     terminal.at("response").at("status") == "incomplete" &&
                                     terminal.at("response").at("incomplete_details").at("reason") ==
                                         "max_output_tokens",
                                 what + " responses: the stream does not end with response.incomplete "
                                        "for max_output_tokens");
    failures += check(text == (delta ? kReplacement : ""), what + " responses: the text is not the token's U+FFFD");
    failures += check(streamed.seconds < 10.0, what + " responses: the stream took " +
                                                   std::to_string(streamed.seconds) + " s to end");
    return failures;
}

int test_messages(const LoopbackServer& server, bool delta, const std::string& what) {
    const Json body{{"model", kModel},
                    {"stream", true},
                    {"max_tokens", 1},
                    {"messages", Json::array({Json{{"role", "user"}, {"content", "hi"}}})}};
    Streamed streamed;
    if (const std::string error = stream(server, "/v1/messages", body, streamed); !error.empty()) {
        return check(false, what + " messages: " + error);
    }
    std::string text;
    std::string stop_reason;
    for (const SseFrame& frame : streamed.frames) {
        const Json event = Json::parse(frame.data);
        if (frame.event == "content_block_delta" && event.at("delta").value("type", "") == "text_delta") {
            text += event.at("delta").value("text", "");
        }
        if (frame.event == "message_delta") { stop_reason = event.at("delta").value("stop_reason", ""); }
    }
    int failures = check(!streamed.frames.empty() && streamed.frames.front().event == "message_start" &&
                             streamed.frames.back().event == "message_stop",
                         what + " messages: the stream does not run from message_start to message_stop");
    failures += check(stop_reason == "max_tokens",
                      what + " messages: no stop_reason max_tokens, got '" + stop_reason + "'");
    failures += check(text == (delta ? kReplacement : ""), what + " messages: the text is not the token's U+FFFD");
    failures += check(streamed.seconds < 10.0, what + " messages: the stream took " +
                                                   std::to_string(streamed.seconds) + " s to end");
    return failures;
}

} // namespace

int main() try {
    LoneTokenBackend backend;
    int failures = 0;
    for (const bool delta : {true, false}) {
        backend.publish_delta(delta);
        const std::string what = delta ? "a lone U+FFFD delta" : "no delta";
        const LoopbackServer server(backend);
        // Twice on one client: the first stream's end leaves the connection usable.
        for (int pass = 0; pass < 2; ++pass) {
            failures += test_chat(server, delta, what);
            failures += test_responses(server, delta, what);
            failures += test_messages(server, delta, what);
        }
    }
    if (failures == 0) { std::cout << "lone-token stream tests passed\n"; }
    return failures == 0 ? 0 : 1;
} catch (const std::exception& error) {
    std::cerr << "FAIL: uncaught exception: " << error.what() << '\n';
    return 1;
}
