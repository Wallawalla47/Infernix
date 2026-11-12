#include "serve/http_server.h"

#include <nlohmann/json.hpp>
#include <spdlog/logger.h>
#include <spdlog/sinks/null_sink.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

// --restore-reasoning over real HTTP (candidates ZA-03): the Chat Completions route remembers each reply's
// reasoning, from an aggregate and from a streamed response, and a later request that sends the reply back
// without it reaches the backend with it, through the option; the counts reach /metrics. The store's own
// test covers the key; this covers that the route records and applies it, and that without the option the
// history reaches the backend as the client sent it while the answers are still counted.
namespace {

using Json = nlohmann::json;
using namespace infernix::serve;

constexpr const char* kModel = "reasoning-restore-test-model";
constexpr std::chrono::seconds kTimeout{30};

int check(bool condition, const std::string& message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

// A reply the backend gives: text, reasoning and tool calls.
struct Reply {
    std::string text;
    std::string reasoning;
    std::vector<infernix::GeneratedToolCall> tool_calls;
};

// Answers each request with the queued reply and records the messages it was asked to prepare.
class RecordingBackend final : public GenerationBackend {
public:
    // /metrics reports the resolved snapshot slots, which a started Engine always has.
    RecordingBackend() { engine_options_.context_cache.hybrid.device_snapshot_slots = 1; }

    void reply(Reply value) {
        std::lock_guard lock(mutex_);
        reply_ = std::move(value);
    }

    [[nodiscard]] std::vector<ChatTurn> last_messages() const {
        std::lock_guard lock(mutex_);
        return messages_;
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

    PreparedRequest prepare(const GenerationRequest& request, GenerationConsumerMode,
                            infernix::GenerationObservationOptions,
                            std::function<bool()>) const override {
        {
            std::lock_guard lock(mutex_);
            messages_ = request.messages;
        }
        PreparedRequest prepared;
        prepared.prompt_tokens = 16;
        return prepared;
    }

    PreparedRequest prepare_continuation(const GenerationRequest&, GenerationConsumerMode,
                                         infernix::GenerationObservationOptions,
                                         std::function<bool()>) const override {
        throw std::logic_error("a single-generation answer continued");
    }

    int count_prompt_tokens(const GenerationRequest&, std::function<bool()>) const override { return 16; }

    GenerationOutcome run(PreparedRequest& prepared, const StreamSink* sink, std::function<bool()>) override {
        Reply value;
        {
            std::lock_guard lock(mutex_);
            value = reply_;
        }
        if (sink != nullptr) {
            if (sink->on_start) {
                sink->on_start(infernix::GenerationStart{
                    .prompt = infernix::PromptSummary{
                        .prompt_tokens = static_cast<std::uint32_t>(prepared.prompt_tokens)}});
            }
            if (!value.reasoning.empty() && sink->on_reasoning) { sink->on_reasoning(value.reasoning); }
            if (!value.text.empty() && sink->on_content) { sink->on_content(value.text); }
        }
        GenerationOutcome outcome;
        outcome.text              = value.text;
        outcome.reasoning         = value.reasoning;
        outcome.tool_calls        = value.tool_calls;
        outcome.prompt_tokens     = prepared.prompt_tokens;
        outcome.completion_tokens = 4;
        outcome.finish_reason     = infernix::FinishReason::StopToken;
        return outcome;
    }

    void admit(PreparedRequest&, std::function<bool()>) const override {}

    void stop() noexcept override {}

private:
    infernix::EngineOptions engine_options_;
    mutable std::mutex mutex_;
    Reply reply_;
    mutable std::vector<ChatTurn> messages_;
};

// A server of the real options on a loopback port, with `restore_reasoning` as configured.
class TestServer {
public:
    TestServer(GenerationBackend& backend, bool restore_reasoning) {
        ServeOptions options;
        options.host                  = "127.0.0.1";
        options.port                  = free_loopback_port();
        options.log_stats_interval_ms = 0;
        options.log_stats_panel       = false;
        options.restore_reasoning     = restore_reasoning;
        auto logger =
            std::make_shared<spdlog::logger>("http-reasoning-restore", std::make_shared<spdlog::sinks::null_sink_mt>());
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

    [[nodiscard]] httplib::Result post(const Json& body) const {
        return client_->Post("/v1/chat/completions", body.dump(), "application/json");
    }

    [[nodiscard]] std::string metrics() const {
        const httplib::Result result = client_->Get("/metrics");
        if (!result || result->status != 200) {
            std::cerr << "/metrics: "
                      << (result ? std::to_string(result->status) + " " + result->body.substr(0, 400)
                                 : "error " + httplib::to_string(result.error()))
                      << '\n';
            return {};
        }
        return result->body;
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

Json user(std::string_view text) { return Json{{"role", "user"}, {"content", text}}; }

Json assistant(std::string_view text) { return Json{{"role", "assistant"}, {"content", text}}; }

Json chat(Json messages, bool stream) {
    Json body{{"model", kModel}, {"max_tokens", 64}, {"messages", std::move(messages)}};
    if (stream) { body["stream"] = true; }
    return body;
}

// Posts one request and requires a 200; returns its body.
std::string answered(const TestServer& server, const Json& body, int& failures, const std::string& what) {
    const httplib::Result result = server.post(body);
    if (!result || result->status != 200) {
        failures += check(false, what + ": no 200 answer" + (result ? ": " + result->body : std::string()));
        return {};
    }
    return result->body;
}

bool has(const std::string& text, std::string_view line) { return text.find(line) != std::string::npos; }

int run(bool restore) {
    const std::string mode = restore ? "with --restore-reasoning" : "without --restore-reasoning";
    RecordingBackend backend;
    TestServer server(backend, restore);
    int failures = 0;

    // Turn 1, aggregate: the reply's reasoning goes to the client and into the store.
    backend.reply({.text = "4", .reasoning = "two and two make four", .tool_calls = {}});
    const std::string first = answered(server, chat(Json::array({user("what is 2+2?")}), false), failures, mode);
    if (!first.empty()) {
        failures += check(Json::parse(first).at("choices").at(0).at("message").at("reasoning_content") ==
                              "two and two make four",
                          mode + ": the first reply lost its reasoning");
    }

    // Turn 2, streamed, echoing turn 1's answer without its reasoning.
    backend.reply({.text = "6", .reasoning = "three and three make six", .tool_calls = {}});
    (void)answered(server, chat(Json::array({user("what is 2+2?"), assistant("4"), user("and 3+3?")}), true),
                   failures, mode);
    std::vector<ChatTurn> seen = backend.last_messages();
    failures += check(seen.size() == 3 && seen[1].reasoning_content == (restore ? "two and two make four" : ""),
                      mode + ": the echoed answer reached the backend with the wrong reasoning");

    // Turn 3, aggregate: both earlier answers sent back without reasoning, the second from a streamed reply.
    backend.reply({.text = "you're welcome", .reasoning = "be polite", .tool_calls = {}});
    (void)answered(server,
                   chat(Json::array({user("what is 2+2?"), assistant("4"), user("and 3+3?"), assistant("6"),
                                     user("thanks")}),
                        false),
                   failures, mode);
    seen = backend.last_messages();
    failures += check(seen.size() == 5 && seen[1].reasoning_content == (restore ? "two and two make four" : "") &&
                          seen[3].reasoning_content == (restore ? "three and three make six" : ""),
                      mode + ": the answers of a streamed and an aggregate reply were not both restored");

    // The same answer after another prompt, and an answer the client sends with its own reasoning.
    (void)answered(server, chat(Json::array({user("what is 1+3?"), assistant("4"), user("and 3+3?")}), false),
                   failures, mode);
    seen = backend.last_messages();
    failures += check(seen.size() == 3 && seen[1].reasoning_content.empty(),
                      mode + ": an answer to another prompt took the reasoning");
    Json own = assistant("4");
    own["reasoning_content"] = "my own notes";
    (void)answered(server, chat(Json::array({user("what is 2+2?"), own, user("and 3+3?")}), false), failures, mode);
    seen = backend.last_messages();
    failures += check(seen.size() == 3 && seen[1].reasoning_content == "my own notes",
                      mode + ": the client's own reasoning was replaced");

    // A tool-calling reply, its call sent back by the id the response gave it.
    backend.reply({.text       = "",
                   .reasoning  = "I need the file",
                   .tool_calls = {{.name = "read", .arguments_json = R"({"path":"a.txt"})"}}});
    const std::string called = answered(server, chat(Json::array({user("read a.txt")}), false), failures, mode);
    if (!called.empty()) {
        const Json message = Json::parse(called).at("choices").at(0).at("message");
        const Json echoed  = Json{
            {"role", "assistant"}, {"content", nullptr}, {"tool_calls", message.at("tool_calls")}};
        const std::string id = message.at("tool_calls").at(0).at("id").get<std::string>();
        backend.reply({.text = "done", .reasoning = "read it", .tool_calls = {}});
        (void)answered(server,
                       chat(Json::array({user("read a.txt"), echoed,
                                         Json{{"role", "tool"}, {"tool_call_id", id}, {"content", "hello"}}}),
                            false),
                       failures, mode);
        seen = backend.last_messages();
        failures += check(seen.size() == 3 && seen[1].reasoning_content == (restore ? "I need the file" : ""),
                          mode + ": the tool-call message reached the backend with the wrong reasoning");
    }

    // Counted with or without the option: answers whose reply had reasoning (turn 2: one; turn 3: two),
    // and the tool-call message; restored only with it.
    const std::string metrics = server.metrics();
    std::string reasoning_lines;
    for (std::size_t begin = 0; begin < metrics.size();) {
        const std::size_t end   = std::min(metrics.find('\n', begin), metrics.size());
        const std::string line  = metrics.substr(begin, end - begin);
        if (line.starts_with("infernix_chat_reasoning_")) { reasoning_lines += "  " + line + '\n'; }
        begin = end + 1;
    }
    std::cout << mode << ": /metrics\n"
              << (metrics.empty() ? std::string("  (no /metrics answer)\n") : reasoning_lines);
    failures += check(has(metrics, "infernix_chat_reasoning_missing_total{message=\"answer\"} 3\n") &&
                          has(metrics, "infernix_chat_reasoning_missing_total{message=\"tool_call\"} 1\n"),
                      mode + ": /metrics does not count the messages that arrived without reasoning");
    failures += check(has(metrics, restore ? "infernix_chat_reasoning_restored_total{message=\"answer\"} 3\n"
                                           : "infernix_chat_reasoning_restored_total{message=\"answer\"} 0\n") &&
                          has(metrics, restore ? "infernix_chat_reasoning_restored_total{message=\"tool_call\"} 1\n"
                                               : "infernix_chat_reasoning_restored_total{message=\"tool_call\"} 0\n"),
                      mode + ": /metrics does not count the restored messages");
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    try {
        failures += run(true);
        failures += run(false);
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    std::cout << (failures == 0 ? "PASS" : "FAIL") << " http_reasoning_restore (" << failures << " failures)\n";
    return failures == 0 ? 0 : 1;
}
