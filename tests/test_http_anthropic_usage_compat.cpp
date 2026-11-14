#include "serve/http_server.h"

#include <nlohmann/json.hpp>
#include <spdlog/logger.h>
#include <spdlog/sinks/null_sink.h>

#include <chrono>
#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

// --claude-code-usage-compat over real HTTP: the route, the option and the streaming encoder
// together. A generation backend streams one ordinary answer through HttpServer::handle_messages,
// so what a client actually receives is compared with the flag off and on. The encoder's own unit
// test cannot show that the ServeOptions field reaches the encoder at all, which is what this
// covers.
namespace {

using Json = nlohmann::json;
using namespace infernix::serve;

constexpr const char* kModel    = "streaming-test-model";
constexpr int kCompletionTokens = 6;
constexpr std::chrono::seconds kTimeout{30};
constexpr const char* kText = "hello there";

int check(bool condition, const std::string& message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

// One prompt's cache accounting, as the Engine reports it for a single generation.
struct Accounted {
    int prompt = 0;
    int reused = 0;
    std::optional<std::uint32_t> created;
    const char* what = "";
};

std::vector<Accounted> accounted() {
    return {
        {400, 0, std::uint32_t{400}, "cold, fully captured"},
        {520, 320, std::uint32_t{180}, "warm, partly captured"},
        {400, 0, std::uint32_t{0}, "cache on, nothing captured"},
        {400, 0, std::nullopt, "no prefix cache"},
        {400, 400, std::uint32_t{0}, "fully reused"},
    };
}

// A backend that streams one ordinary answer with the configured accounting and never generates.
class StreamingBackend final : public GenerationBackend {
public:
    // Where the streamed run fails, to cover a stream that ends with an error event instead of the
    // terminal message_delta: before the Engine's start observation, and after the message started.
    enum class Failure : std::uint8_t { None, BeforeStart, AfterContent };

    void account(Accounted value) {
        std::lock_guard lock(mutex_);
        accounted_ = value;
    }

    void fail(Failure failure) {
        std::lock_guard lock(mutex_);
        failure_ = failure;
    }

    const infernix::EngineOptions& engine_options() const override { return engine_options_; }

    infernix::LoadSummary load_summary() const override {
        infernix::LoadSummary summary;
        summary.model_name = kModel;
        return summary;
    }

    infernix::ModelMetadata model_metadata() const override { return {}; }

    infernix::MemorySummary memory_summary() const override { return {}; }

    infernix::RuntimeStats runtime_stats() const override { return {}; }

    void reset_execution_maxima() noexcept override {}

    bool is_available() const override { return true; }

    bool has_failed() const override { return false; }

    infernix::ModelSamplingDefaults sampling_defaults() const override { return {}; }

    std::vector<infernix::TokenId> tokenize_text(std::string_view) const override { return {}; }

    PreparedRequest prepare(const GenerationRequest&, GenerationConsumerMode,
                            infernix::GenerationObservationOptions,
                            std::function<bool()>) const override {
        PreparedRequest prepared;
        prepared.prompt_tokens = account().prompt;
        return prepared;
    }

    PreparedRequest prepare_continuation(const GenerationRequest&, GenerationConsumerMode,
                                         infernix::GenerationObservationOptions,
                                         std::function<bool()>) const override {
        throw std::logic_error("a single-generation answer continued");
    }

    int count_prompt_tokens(const GenerationRequest&, std::function<bool()>) const override {
        return account().prompt;
    }

    // The Engine's own order: the start observation with the prompt's reused prefix, then the text,
    // then the terminal result carrying the cache creation the run published.
    GenerationOutcome run(PreparedRequest& prepared, const StreamSink* sink,
                          std::function<bool()>) override {
        const Accounted value = account();
        const Failure stage   = failure();
        if (stage == Failure::BeforeStart) { throw ApiException(stream_failure()); }
        if (sink != nullptr) {
            if (sink->on_start) {
                sink->on_start(infernix::GenerationStart{
                    .prompt = infernix::PromptSummary{.prompt_tokens = static_cast<std::uint32_t>(
                                                          prepared.prompt_tokens)},
                    .reused_prompt_tokens = static_cast<std::uint32_t>(value.reused)});
            }
            if (sink->on_content) {
                sink->on_content("hello");
                sink->on_content(" there");
            }
        }
        if (stage == Failure::AfterContent) { throw ApiException(stream_failure()); }
        GenerationOutcome outcome;
        outcome.text              = kText;
        outcome.prompt_tokens     = prepared.prompt_tokens;
        outcome.completion_tokens = kCompletionTokens;
        outcome.computed_prefill_tokens =
            static_cast<std::uint32_t>(prepared.prompt_tokens - value.reused);
        outcome.reused_prompt_tokens        = static_cast<std::uint32_t>(value.reused);
        outcome.cache_created_prompt_tokens = value.created;
        outcome.finish_reason               = infernix::FinishReason::StopToken;
        return outcome;
    }

    bool admit(PreparedRequest&, std::function<bool()>, std::chrono::steady_clock::time_point) const override {
        return true;
    }

    void stop() noexcept override {}

private:
    Accounted account() const {
        std::lock_guard lock(mutex_);
        return accounted_;
    }

    Failure failure() const {
        std::lock_guard lock(mutex_);
        return failure_;
    }

    static ApiError stream_failure() {
        ApiError error;
        error.status  = 500;
        error.type    = "api_error";
        error.message = "the streamed generation failed";
        return error;
    }

    infernix::EngineOptions engine_options_;
    mutable std::mutex mutex_;
    Accounted accounted_ = Accounted{.prompt = 400, .reused = 0, .created = std::uint32_t{400}};
    Failure failure_     = Failure::None;
};

// A server of the real options on a loopback port, with `claude_code_usage_compat` as configured.
class TestServer {
public:
    TestServer(GenerationBackend& backend, bool claude_code_usage_compat) {
        ServeOptions options;
        options.host                     = "127.0.0.1";
        options.port                     = free_loopback_port();
        options.log_stats_interval_ms    = 0;
        options.log_stats_panel          = false;
        options.claude_code_usage_compat = claude_code_usage_compat;
        auto logger                      = std::make_shared<spdlog::logger>(
            "http-usage-compat", std::make_shared<spdlog::sinks::null_sink_mt>());
        server_ = std::make_unique<HttpServer>(options, std::move(logger));
        if (options.port <= 0 || !server_->bind()) {
            throw std::runtime_error("the test server could not bind a loopback port");
        }
        server_->attach(backend);
        listener_ = std::thread([this] { (void)server_->listen(); });
        client_   = std::make_unique<httplib::Client>("127.0.0.1", options.port);
        client_->set_connection_timeout(kTimeout);
        client_->set_read_timeout(kTimeout);
    }

    ~TestServer() {
        server_->stop();
        if (listener_.joinable()) { listener_.join(); }
    }

    TestServer(const TestServer&)            = delete;
    TestServer& operator=(const TestServer&) = delete;

    [[nodiscard]] httplib::Result post(std::string_view path, const Json& body) const {
        return client_->Post(std::string(path), body.dump(), "application/json");
    }

private:
    static int free_loopback_port() {
        httplib::Server probe;
        const int port = probe.bind_to_any_port("127.0.0.1");
        probe.stop();
        return port;
    }

    std::unique_ptr<HttpServer> server_;
    std::thread listener_;
    std::unique_ptr<httplib::Client> client_;
};

Json messages_body(bool stream) {
    const Json user = Json::array({Json{{"role", "user"}, {"content", "hello"}}});
    Json body{{"model", kModel}, {"max_tokens", 32}, {"messages", user}};
    if (stream) { body["stream"] = true; }
    return body;
}

// The frames of an SSE body, each `event: TYPE` then one `data:` JSON line whose type matches.
std::vector<std::string> sse_frames(const std::string& body) {
    std::vector<std::string> frames;
    for (std::size_t begin = 0; begin < body.size();) {
        const std::size_t end = body.find("\n\n", begin);
        if (end == std::string::npos) { throw std::runtime_error("unterminated SSE event"); }
        const std::string frame = body.substr(begin, end - begin);
        const std::size_t data  = frame.find("\ndata: ");
        if (!frame.starts_with("event: ") || data == std::string::npos) {
            throw std::runtime_error("invalid SSE framing");
        }
        if (Json::parse(frame.substr(data + 7)).at("type") != frame.substr(7, data - 7)) {
            throw std::runtime_error("SSE event name and type differ");
        }
        frames.push_back(frame);
        begin = end + 2;
    }
    return frames;
}

Json event_of(const std::string& frame) {
    return Json::parse(frame.substr(frame.find("\ndata: ") + 7));
}

std::vector<std::string> event_types(const std::vector<std::string>& frames) {
    std::vector<std::string> types;
    for (const std::string& frame : frames) {
        types.push_back(frame.substr(7, frame.find("\ndata: ") - 7));
    }
    return types;
}

std::string frame_of(const std::vector<std::string>& frames, std::string_view type) {
    const std::string wanted(type);
    for (const std::string& frame : frames) {
        if (event_of(frame).at("type") == wanted) { return frame; }
    }
    return {};
}

// A frame with the two things that may differ for reasons of their own removed: the per-request id
// an error event carries, and message_start's input_tokens, which is the flag's whole effect. Every
// other byte of the two modes' streams has to match.
std::string normalized_frame(const std::string& frame) {
    Json event = event_of(frame);
    event.erase("request_id");
    if (event.contains("message") && event["message"].is_object()) {
        // Every request gets its own message id, which says nothing about the flag.
        event["message"].erase("id");
        if (event["message"].contains("usage") && event["message"]["usage"].is_object()) {
            event["message"]["usage"].erase("input_tokens");
        }
    }
    return event.dump();
}

bool only_the_start_differs(const std::vector<std::string>& plain,
                            const std::vector<std::string>& compat) {
    if (plain.size() != compat.size()) { return false; }
    for (std::size_t index = 0; index < plain.size(); ++index) {
        if (normalized_frame(plain[index]) != normalized_frame(compat[index])) { return false; }
    }
    return true;
}

Json usage_of(const std::string& frame) {
    const Json event = event_of(frame);
    return event.contains("usage") ? event.at("usage") : event.at("message").at("usage");
}

bool event_stream(const httplib::Result& result) {
    return result->get_header_value("Content-Type").find("text/event-stream") != std::string::npos;
}

// The same streamed answer with the flag off and on: only message_start's input_tokens may differ,
// and the terminal usage stays the exact accounting of the non-streaming response.
int test_streamed_usage(const Accounted& row, StreamingBackend& backend) {
    const std::string what = std::string(row.what);
    backend.account(row);
    const int created = row.created ? static_cast<int>(*row.created) : 0;

    TestServer plain(backend, false);
    TestServer compat(backend, true);
    const httplib::Result plain_answer  = plain.post("/v1/messages", messages_body(true));
    const httplib::Result compat_answer = compat.post("/v1/messages", messages_body(true));
    int failures                        = 0;
    if (!plain_answer || !compat_answer) {
        std::cerr << "FAIL: " << what << ": no HTTP answer\n";
        return 1;
    }
    failures += check(plain_answer->status == 200 && event_stream(plain_answer) &&
                          compat_answer->status == 200 && event_stream(compat_answer),
                      what + ": the answer is not a 200 event stream");

    std::vector<std::string> plain_frames;
    std::vector<std::string> compat_frames;
    try {
        plain_frames  = sse_frames(plain_answer->body);
        compat_frames = sse_frames(compat_answer->body);
    } catch (const std::exception& error) {
        return failures + check(false, what + ": " + error.what());
    }

    // The protocol's event order, unchanged by the flag, and a real stream in both modes.
    const std::vector<std::string> types = event_types(plain_frames);
    failures += check(!types.empty() && types.front() == "message_start" &&
                          types.back() == "message_stop" && types.size() > 3 &&
                          event_types(compat_frames) == types,
                      what + ": the flag changed the event sequence");

    const std::string plain_start_frame  = frame_of(plain_frames, "message_start");
    const std::string compat_start_frame = frame_of(compat_frames, "message_start");
    if (plain_start_frame.empty() || compat_start_frame.empty() ||
        frame_of(plain_frames, "message_delta").empty() ||
        frame_of(compat_frames, "message_delta").empty()) {
        return failures + check(false, what + ": the stream has no message_start or message_delta");
    }
    const Json plain_start  = usage_of(plain_start_frame);
    const Json compat_start = usage_of(compat_start_frame);
    failures += check(plain_start.at("input_tokens") == row.prompt - row.reused &&
                          plain_start.at("cache_read_input_tokens") == row.reused &&
                          plain_start.at("cache_creation_input_tokens").is_null() &&
                          plain_start.at("output_tokens") == 0,
                      what + ": the start without the flag is not the exact provisional split");
    failures += check(compat_start.at("input_tokens") == 0 &&
                          compat_start.at("cache_read_input_tokens") == row.reused &&
                          compat_start.at("cache_creation_input_tokens").is_null() &&
                          compat_start.at("output_tokens") == 0,
                      what + ": the start with the flag lost its provisional 0 or its cache read");

    // Nothing besides message_start's input_tokens differs, the terminal usage least of all.
    failures +=
        check(only_the_start_differs(plain_frames, compat_frames),
              what + ": a field besides message_start input_tokens differs between the modes");

    const std::string plain_terminal  = frame_of(plain_frames, "message_delta");
    const std::string compat_terminal = frame_of(compat_frames, "message_delta");
    const Json terminal               = usage_of(plain_terminal);
    failures += check(!plain_terminal.empty() && plain_terminal == compat_terminal &&
                          terminal.at("input_tokens") == row.prompt - row.reused - created &&
                          terminal.at("cache_read_input_tokens") == row.reused &&
                          terminal.at("cache_creation_input_tokens") ==
                              (row.created ? Json(*row.created) : Json(nullptr)) &&
                          terminal.at("output_tokens") == kCompletionTokens &&
                          event_of(plain_terminal).at("delta").at("stop_reason") == "end_turn",
                      what + ": the terminal usage is not the response's exact accounting");
    failures += check(terminal.at("input_tokens").get<int>() + created + row.reused == row.prompt,
                      what + ": the terminal usage does not sum to the prompt");

    // The flag reaches the streaming encoder only: the aggregate response is the same answer, and
    // its usage is what both streams end with.
    const httplib::Result plain_aggregate_answer = plain.post("/v1/messages", messages_body(false));
    const httplib::Result compat_aggregate_answer =
        compat.post("/v1/messages", messages_body(false));
    if (!plain_aggregate_answer || !compat_aggregate_answer) {
        return failures + check(false, what + ": the non-streaming request got no HTTP answer");
    }
    const Json plain_aggregate  = Json::parse(plain_aggregate_answer->body, nullptr, false);
    const Json compat_aggregate = Json::parse(compat_aggregate_answer->body, nullptr, false);
    // The two answers carry their own message ids, so compare what the flag could reach: the usage,
    // the stop and the text.
    failures +=
        check(plain_aggregate_answer->status == 200 && compat_aggregate_answer->status == 200 &&
                  !event_stream(plain_aggregate_answer) && !event_stream(compat_aggregate_answer) &&
                  compat_aggregate.at("usage") == plain_aggregate.at("usage") &&
                  plain_aggregate.at("usage") == terminal &&
                  compat_aggregate.at("stop_reason") == "end_turn" &&
                  compat_aggregate.at("content") == plain_aggregate.at("content") &&
                  plain_aggregate.at("content").at(0).at("text") == kText,
              what + ": the non-streaming response changed or disagrees with the stream");
    return failures;
}

// A stream the generation breaks off answers with an error event and never with the terminal
// message_delta, in either mode. The compat mode may not turn that into a finished message, and its
// start keeps the provisional 0 there too: on this route no event carries the uncached count.
int test_stream_failure(StreamingBackend& backend) {
    const Accounted row{.prompt = 400, .reused = 320, .created = std::uint32_t{80}};
    backend.account(row);
    int failures = 0;
    for (const auto& [stage, label, start_input] :
         {std::tuple{StreamingBackend::Failure::BeforeStart, "before the Engine start", row.prompt},
          std::tuple{StreamingBackend::Failure::AfterContent, "after the message started",
                     row.prompt - row.reused}}) {
        const std::string what = std::string("a stream failed ") + label;
        backend.fail(stage);
        const TestServer plain(backend, false);
        const TestServer compat(backend, true);
        const httplib::Result plain_answer  = plain.post("/v1/messages", messages_body(true));
        const httplib::Result compat_answer = compat.post("/v1/messages", messages_body(true));
        backend.fail(StreamingBackend::Failure::None);
        if (!plain_answer || !compat_answer) {
            failures += check(false, what + ": no HTTP answer");
            continue;
        }
        // The stream was already committed: the failure is an event, not a status.
        failures += check(plain_answer->status == 200 && event_stream(plain_answer) &&
                              compat_answer->status == 200 && event_stream(compat_answer),
                          what + ": the answer is not a 200 event stream");
        std::vector<std::string> plain_frames;
        std::vector<std::string> compat_frames;
        try {
            plain_frames  = sse_frames(plain_answer->body);
            compat_frames = sse_frames(compat_answer->body);
        } catch (const std::exception& error) {
            failures += check(false, what + ": " + error.what());
            continue;
        }
        const std::vector<std::string> types = event_types(plain_frames);
        const std::string plain_error_frame  = frame_of(plain_frames, "error");
        const std::string compat_error_frame = frame_of(compat_frames, "error");
        failures += check(!types.empty() && types.front() == "message_start" &&
                              types.back() == "error" && types == event_types(compat_frames),
                          what + ": the stream does not end with the same events in both modes");
        // The error event carries its request's own id, so compare its body rather than its bytes.
        failures += check(!plain_error_frame.empty() && !compat_error_frame.empty() &&
                              event_of(plain_error_frame).at("error") ==
                                  event_of(compat_error_frame).at("error") &&
                              only_the_start_differs(plain_frames, compat_frames),
                          what + ": the modes differ in more than message_start input_tokens");
        // No terminal success is invented after the error, in either mode.
        failures += check(frame_of(plain_frames, "message_delta").empty() &&
                              frame_of(plain_frames, "message_stop").empty() &&
                              frame_of(compat_frames, "message_delta").empty() &&
                              frame_of(compat_frames, "message_stop").empty(),
                          what + ": an error stream still reported a terminal usage or stop");
        failures += check(!plain_error_frame.empty() &&
                              event_of(plain_error_frame).at("error").at("message") ==
                                  "the streamed generation failed",
                          what + ": the error event lost the failure");

        const Json plain_start  = usage_of(frame_of(plain_frames, "message_start"));
        const Json compat_start = usage_of(frame_of(compat_frames, "message_start"));
        failures += check(plain_start.at("input_tokens") == start_input &&
                              compat_start.at("input_tokens") == 0 &&
                              compat_start.at("cache_read_input_tokens") ==
                                  plain_start.at("cache_read_input_tokens"),
                          what + ": the start usage of a failed stream is not provisional");
    }
    return failures;
}

} // namespace

int main() try {
    StreamingBackend backend;
    int failures = 0;
    for (const Accounted& row : accounted()) { failures += test_streamed_usage(row, backend); }
    failures += test_stream_failure(backend);
    if (failures == 0) { std::cout << "anthropic usage compatibility stream tests passed\n"; }
    return failures == 0 ? 0 : 1;
} catch (const std::exception& error) {

    std::cerr << "FAIL: uncaught exception: " << error.what() << '\n';
    return 1;
}
