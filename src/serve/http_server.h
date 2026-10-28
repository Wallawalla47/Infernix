#pragma once

#include "serve/console_stats.h"
#include "serve/decide.h"
#include "serve/generation_service.h"
#include "serve/metrics.h"
#include "serve/metrics_json.h"
#include "serve/openai_common.h"
#include "serve/operational_log.h"
#include "serve/openai_responses_store.h"
#include "serve/reasoning_store.h"
#include "serve/request_log.h"
#include "serve/serve_options.h"

#include <httplib.h>

#include <atomic>
#include <condition_variable>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <memory>
#include <string>
#include <string_view>
#include <thread>

namespace infernix::serve {

// infernix-serve's exit status after an Engine-wide failure stopped it, so a launcher can restart
// it; startup and other failures exit with 1.
inline constexpr int kEngineFailureExitStatus = 2;
// How often a listening server checks the Engine for such a failure.
inline constexpr std::chrono::milliseconds kEngineWatchInterval{250};

void write_openai_error(httplib::Response& response, const ApiError& error);
void write_anthropic_error(httplib::Response& response, const ApiError& error,
                           const std::string& request_id);

// cpp-httplib invokes the error handler for every application response with status >= 400. Only
// an empty 413 is its own pre-routing payload-limit rejection; application-authored errors must be
// left untouched.
httplib::Server::HandlerResponse handle_unrendered_http_error(const ServeOptions& options,
                                                              const httplib::Request& request,
                                                              httplib::Response& response);

[[nodiscard]] bool matches_bearer_credential(std::string_view authorization,
                                             std::string_view api_key) noexcept;

// Anthropic SDKs append /v1/<endpoint> to their base URL, so a client given the OpenAI-style base
// URL that already ends in /v1 requests /v1/v1/<endpoint>. Every API route also answers there:
// api_route_pattern("/messages") matches /v1/messages and /v1/v1/messages without adding a capture
// group, and canonical_api_path maps the doubled form back to the /v1/<endpoint> it names.
[[nodiscard]] std::string api_route_pattern(std::string_view endpoint);
[[nodiscard]] std::string_view canonical_api_path(std::string_view path) noexcept;

class HttpServer {
public:
    // `panel` is the console panel for the session statistics; it is drawn only when enabled by
    // the options and supported by the terminal, and may be null.
    HttpServer(ServeOptions options, std::shared_ptr<spdlog::logger> logger,
               std::shared_ptr<product::TerminalPanel> panel = nullptr);

    // Reserves the configured address before model loading. The service is attached only after its
    // Engine is ready, then listen() enters the blocking accept loop on the already-bound socket.
    bool bind();
    void attach(GenerationBackend& service);
    // Returns once stop() or an Engine-wide failure (engine_failed()) ends the accept loop.
    bool listen();
    // Whether listen() returned because the Engine failed: it then answers nothing but errors
    // until the process restarts, so the server stops instead of keeping a dead endpoint up.
    [[nodiscard]] bool engine_failed() const noexcept { return engine_failed_.load(); }
    // Closes the listening socket and stops the attached service's Engine, whose queued and
    // running requests then fail, so listen() returns within about one unit of Engine work.
    // Does not block; callable from any thread.
    void stop();

    [[nodiscard]] const std::string& public_model_id() const noexcept { return public_model_id_; }

private:
    class RequestLifecycle {
    public:
        RequestLifecycle(HttpServer& owner, RequestLogContext context);

        void done(const GenerationOutcome& outcome);
        void failure(const RequestFailure& failure);
        void response_failure(const RequestFailure& failure);

        [[nodiscard]] std::uint64_t request_id() const noexcept { return context_.id; }

    private:
        enum class State : std::uint8_t {
            Pending,
            Done,
            Error,
        };

        [[nodiscard]] bool claim(State terminal) noexcept;

        HttpServer* owner_ = nullptr;
        RequestLogContext context_;
        std::atomic<State> state_{State::Pending};
    };

    [[nodiscard]] std::shared_ptr<RequestLifecycle> begin_request(RequestLogContext context);

    void register_routes();
    void handle_chat_completions(const httplib::Request& req, httplib::Response& res);
    void handle_messages(const httplib::Request& req, httplib::Response& res);
    void handle_count_tokens(const httplib::Request& req, httplib::Response& res);
    void handle_decide(const httplib::Request& req, httplib::Response& res);
    void handle_responses(const httplib::Request& req, httplib::Response& res);
    void handle_response_input_tokens(const httplib::Request& req, httplib::Response& res);
    void handle_response_get(const httplib::Request& req, httplib::Response& res);
    void handle_response_delete(const httplib::Request& req, httplib::Response& res);
    void handle_response_input_items(const httplib::Request& req, httplib::Response& res);
    void handle_response_cancel(const httplib::Request& req, httplib::Response& res);
    void handle_response_compact(const httplib::Request& req, httplib::Response& res);
    void handle_models(const httplib::Request& req, httplib::Response& res) const;
    void handle_model(const httplib::Request& req, httplib::Response& res) const;

    void record_request_start(const RequestLogContext& context);
    [[nodiscard]] infernix::GenerationSchedulingObserver
    scheduling_observer(std::uint64_t request_id, std::string http_request_id);
    [[nodiscard]] infernix::GenerationFirstTokenObserver first_token_observer();
    void record_request_rejected(const RequestRejectionLogContext& context);
    void record_request_done(const RequestLogContext& context, const GenerationOutcome& outcome);
    void record_request_failure(const RequestLogContext& context, const RequestFailure& failure);
    void record_response_failure(std::uint64_t request_id, const RequestFailure& failure);
    void record_throughput(const ThroughputReport& report);
    [[nodiscard]] std::string metrics_json_body() const;
    void run_stats_reporter();
    void stop_stats_reporter();
    void run_engine_watch();
    void stop_engine_watch();

    GenerationBackend* service_ = nullptr;
    ServeOptions options_;
    std::string public_model_id_;
    infernix::ModelMetadata model_metadata_;
    // Probed once through the attached service's prompt path (/v1/models chat_template_caps).
    ChatTemplateCaps chat_template_caps_;
    // /v1/decide answer labels, computed from the loaded tokenizer when the service attaches.
    std::vector<DecideLabel> decide_labels_;
    OpenAIResponsesStore openai_responses_store_;
    // --restore-tool-reasoning: the last 4,096 tool-calling Chat Completions responses' reasoning, at most
    // 64 MiB of it.
    ReasoningStore reasoning_store_{4096, 64U << 20};
    OperationalLog operational_log_;
    JsonlRequestLog request_jsonl_;
    Metrics metrics_;
    RequestHistory metrics_json_history_;
    std::unique_ptr<ConsoleStatsPanel> console_stats_;
    httplib::Server server_;
    std::atomic<std::uint64_t> request_seq_{0};
    std::mutex stats_mutex_;
    std::condition_variable stats_cv_;
    std::thread stats_thread_;
    bool stats_stopping_ = false;
    std::mutex watch_mutex_;
    std::condition_variable watch_cv_;
    std::thread watch_thread_;
    bool watch_stopping_ = false;
    std::atomic<bool> engine_failed_{false};
};

} // namespace infernix::serve
