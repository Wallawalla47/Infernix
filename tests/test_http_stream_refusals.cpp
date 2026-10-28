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
#include <tuple>
#include <utility>
#include <vector>

// The streaming routes answer a request refused before its stream starts over real HTTP, against a
// generation backend that refuses at preparation or at admission and never generates.
namespace {

using Json = nlohmann::json;
using namespace infernix::serve;

constexpr const char* kModel        = "refusing-test-model";
constexpr std::uint32_t kRetryAfter = 30;
constexpr std::chrono::seconds kTimeout{30};

int check(bool condition, const std::string& message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

class RefusingBackend final : public GenerationBackend {
public:
    enum class Stage { Prepare, Admission };

    // The next requests fail with `error` at `stage`, as GenerationService throws it there.
    void refuse(Stage stage, ApiError error) {
        std::lock_guard lock(mutex_);
        stage_ = stage;
        error_ = std::move(error);
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
        refuse_at(Stage::Prepare);
        return {};
    }
    PreparedRequest prepare_continuation(const GenerationRequest&, GenerationConsumerMode,
                                         infernix::GenerationObservationOptions,
                                         std::function<bool()>) const override {
        throw std::logic_error("a refused request continued");
    }
    // The template probe at attach: a declared tool renders.
    int count_prompt_tokens(const GenerationRequest& request,
                            std::function<bool()>) const override {
        return request.tools.empty() ? 10 : 20;
    }
    GenerationOutcome run(PreparedRequest&, const StreamSink*, std::function<bool()>) override {
        throw std::logic_error("a refused request ran");
    }
    void admit(PreparedRequest&, std::function<bool()>) const override {
        refuse_at(Stage::Admission);
    }
    void stop() noexcept override {}

private:
    void refuse_at(Stage stage) const {
        std::lock_guard lock(mutex_);
        if (stage_ == stage && error_) { throw ApiException(*error_); }
    }

    infernix::EngineOptions engine_options_;
    mutable std::mutex mutex_;
    Stage stage_ = Stage::Prepare;
    std::optional<ApiError> error_;
};

struct Route {
    const char* name;
    const char* path;
    Json body;
    bool anthropic = false;
};

std::vector<Route> streaming_routes() {
    const Json user = Json::array({Json{{"role", "user"}, {"content", "hello"}}});
    return {
        {"chat completions", "/v1/chat/completions",
         Json{{"model", kModel}, {"messages", user}, {"stream", true}}},
        {"responses", "/v1/responses",
         Json{{"model", kModel}, {"input", "hello"}, {"stream", true}, {"temperature", 0.5}}},
        {"anthropic messages", "/v1/messages",
         Json{{"model", kModel}, {"max_tokens", 16}, {"messages", user}, {"stream", true}}, true},
    };
}

bool event_stream(const httplib::Result& result) {
    return result->get_header_value("Content-Type").find("text/event-stream") != std::string::npos;
}

// The events of an SSE body, each `event: TYPE` then one `data:` JSON line whose type matches.
std::vector<Json> sse_events(const std::string& body) {
    std::vector<Json> events;
    for (std::size_t begin = 0; begin < body.size();) {
        const std::size_t end = body.find("\n\n", begin);
        if (end == std::string::npos) { throw std::runtime_error("unterminated SSE event"); }
        const std::string frame = body.substr(begin, end - begin);
        const std::size_t data  = frame.find("\ndata: ");
        if (!frame.starts_with("event: ") || data == std::string::npos) {
            throw std::runtime_error("invalid SSE framing");
        }
        Json event = Json::parse(frame.substr(data + 7));
        if (event.at("type") != frame.substr(7, data - 7)) {
            throw std::runtime_error("SSE event name and type differ");
        }
        events.push_back(std::move(event));
        begin = end + 2;
    }
    return events;
}

// The queue's own answers reach a streaming request as a plain HTTP status with Retry-After and an
// error body, before any event stream: 503 and 429 on the OpenAI routes, the Anthropic API's 504
// and 529 on Messages.
int test_admission_refusals(RefusingBackend& backend, httplib::Client& client) {
    int failures = 0;
    const ApiError queue_timeout = request_error_to_api_error(
        infernix::RequestError(infernix::RequestErrorKind::QueueTimeout,
                               "inference request expired while waiting for admission"),
        "structured_outputs.grammar", {}, kRetryAfter);
    const ApiError overloaded = request_error_to_api_error(
        infernix::RequestError(infernix::RequestErrorKind::Overloaded, "request queue is full"),
        "structured_outputs.grammar", {}, kRetryAfter);
    for (const auto& [answer, openai_status, anthropic_status] :
         {std::tuple{queue_timeout, 503, 504}, std::tuple{overloaded, 429, 529}}) {
        backend.refuse(RefusingBackend::Stage::Admission, answer);
        for (const Route& route : streaming_routes()) {
            const std::string what = std::string(route.name) + " " + answer.code;
            const httplib::Result result =
                client.Post(route.path, route.body.dump(), "application/json");
            failures += check(static_cast<bool>(result), what + ": no HTTP answer");
            if (!result) { continue; }
            failures += check(result->status == (route.anthropic ? anthropic_status : openai_status),
                              what + ": status " + std::to_string(result->status));
            failures += check(result->get_header_value("Retry-After") == std::to_string(kRetryAfter),
                              what + ": no Retry-After of the admission window");
            failures += check(!event_stream(result) && result->get_header_value("Content-Type")
                                                           .starts_with("application/json"),
                              what + ": the refusal opened an event stream");
            const Json body = Json::parse(result->body, nullptr, false);
            const bool error_body =
                body.is_object() && body.contains("error") && body.at("error").is_object() &&
                (route.anthropic ? body.value("type", "") == "error" &&
                                       body.at("error").contains("type")
                                 : body.at("error").value("code", "") == answer.code);
            failures += check(error_body, what + ": no error body: " + result->body);
        }
    }
    return failures;
}

// A prompt over the context limit is refused at preparation. A streamed Response answers inside its
// event stream (HTTP 200: created, in_progress, then failed carrying context_length_exceeded and the
// message), the one form Codex compacts on; a non-streamed Response and the other routes keep the
// HTTP 400.
int test_context_overflow(RefusingBackend& backend, httplib::Client& client) {
    int failures            = 0;
    const ApiError overflow = request_error_to_api_error(infernix::RequestError(
        infernix::ContextLengthDetails{.prompt_tokens = 9000, .max_context = 8192}));
    backend.refuse(RefusingBackend::Stage::Prepare, overflow);
    for (const Route& route : streaming_routes()) {
        const std::string what       = std::string(route.name) + " context overflow";
        const httplib::Result result =
            client.Post(route.path, route.body.dump(), "application/json");
        failures += check(static_cast<bool>(result), what + ": no HTTP answer");
        if (!result) { continue; }
        if (std::string_view(route.path) != "/v1/responses") {
            failures += check(result->status == 400 && !event_stream(result),
                              what + ": not an HTTP 400 before any stream");
            continue;
        }
        failures += check(result->status == 200 && event_stream(result),
                          what + ": not answered as an event stream");
        std::vector<Json> events;
        try {
            events = sse_events(result->body);
        } catch (const std::exception& error) {
            failures += check(false, what + ": " + error.what());
        }
        const std::vector<std::string> types = {"response.created", "response.in_progress",
                                                "response.failed"};
        failures += check(events.size() == types.size(), what + ": not three events");
        if (events.size() != types.size()) { continue; }
        const std::string id = events[0].at("response").at("id").get<std::string>();
        for (std::size_t index = 0; index < events.size(); ++index) {
            failures += check(events[index].at("type") == types[index] &&
                                  events[index].at("sequence_number") == index &&
                                  events[index].at("response").at("id") == id,
                              what + ": event " + std::to_string(index) + " out of sequence");
        }
        const Json& failed = events[2].at("response");
        failures += check(id.starts_with("resp_") && failed.at("status") == "failed" &&
                              failed.at("output").empty() && failed.at("temperature") == 0.5 &&
                              failed.at("error").at("code") == "context_length_exceeded" &&
                              failed.at("error").at("message") == overflow.message,
                          what + ": the failed Response lost the context error");

        Json aggregate      = route.body;
        aggregate["stream"] = false;
        const httplib::Result plain = client.Post(route.path, aggregate.dump(), "application/json");
        const Json plain_body       = plain ? Json::parse(plain->body, nullptr, false) : Json(nullptr);
        failures += check(plain && plain->status == 400 && !event_stream(plain) &&
                              plain_body.is_object() && plain_body.contains("error") &&
                              plain_body.at("error").value("code", "") == "context_length_exceeded",
                          "a non-streamed Response over the context did not keep its HTTP 400");
    }
    return failures;
}

int free_loopback_port() {
    httplib::Server probe;
    const int port = probe.bind_to_any_port("127.0.0.1");
    probe.stop();
    return port;
}

} // namespace

int main() try {
    RefusingBackend backend;
    ServeOptions options;
    options.host                  = "127.0.0.1";
    options.port                  = free_loopback_port();
    options.log_stats_interval_ms = 0;
    options.log_stats_panel       = false;
    auto logger = std::make_shared<spdlog::logger>("http-stream-refusals",
                                                   std::make_shared<spdlog::sinks::null_sink_mt>());
    HttpServer server(options, std::move(logger));
    if (options.port <= 0 || !server.bind()) {
        std::cerr << "FAIL: the test server could not bind a loopback port\n";
        return 1;
    }
    server.attach(backend);
    std::thread listener([&server] { (void)server.listen(); });
    httplib::Client client("127.0.0.1", options.port);
    client.set_connection_timeout(kTimeout);
    client.set_read_timeout(kTimeout);

    int failures = 0;
    failures += test_admission_refusals(backend, client);
    failures += test_context_overflow(backend, client);

    server.stop();
    listener.join();
    if (failures == 0) { std::cout << "http stream refusal tests passed\n"; }
    return failures == 0 ? 0 : 1;
} catch (const std::exception& error) {
    std::cerr << "FAIL: uncaught exception: " << error.what() << '\n';
    return 1;
}
