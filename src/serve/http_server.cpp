#include "serve/http_server.h"

#include "product/logging/logging.h"
#include "serve/anthropic_messages.h"
#include "serve/http_transport.h"
#include "serve/openai_common.h"
#include "serve/request_log.h"

#include <nlohmann/json.hpp>
#include <spdlog/logger.h>

#include <chrono>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

namespace infernix::serve {
namespace {

void write_exception(httplib::Response& res, const std::exception& ex) {
    ApiError error;
    error.status  = 500;
    error.type    = "internal_error";
    error.message = ex.what();
    write_openai_error(res, error);
}

bool is_anthropic_path(std::string_view path) {
    return canonical_api_path(path).starts_with("/v1/messages");
}

bool is_openai_path(std::string_view path) {
    return path.starts_with("/v1/") && !is_anthropic_path(path);
}

void ensure_http_request_id(const httplib::Request& request, httplib::Response& response) {
    if (is_anthropic_path(request.path) && !response.has_header("request-id")) {
        response.set_header("request-id", new_anthropic_request_id());
    } else if (is_openai_path(request.path) && !response.has_header("x-request-id")) {
        response.set_header("x-request-id", new_openai_request_id());
    }
}

ThroughputReport make_throughput_report(const infernix::RuntimeStats& previous,
                                        const infernix::RuntimeStats& current,
                                        double interval_seconds) {
    return ThroughputReport{
        .interval_seconds = interval_seconds,
        .computed_prefill_tokens =
            current.computed_prefill_tokens - previous.computed_prefill_tokens,
        .committed_decode_tokens =
            current.committed_decode_tokens - previous.committed_decode_tokens,
        .decode_rounds     = current.decode_rounds - previous.decode_rounds,
        .decode_row_rounds = current.decode_row_rounds - previous.decode_row_rounds,
        .previous          = previous,
        .current           = current,
    };
}

bool report_has_activity(const ThroughputReport& report) {
    return report.computed_prefill_tokens != 0 || report.committed_decode_tokens != 0 ||
           report.decode_rounds != 0 || report.current.running_requests != 0 ||
           report.current.waiting_requests != 0 || report.current.paused_requests != 0 ||
           report.current.replaying_requests != 0 || report.current.materializing_requests != 0 ||
           report.current.terminal_pending_requests != 0 ||
           report.current.preemptions != report.previous.preemptions ||
           report.current.snapshot_restores != report.previous.snapshot_restores ||
           report.current.replay_restores != report.previous.replay_restores ||
           report.current.replayed_tokens != report.previous.replayed_tokens ||
           report.current.root_selections != report.previous.root_selections ||
           report.current.prefix_selections != report.previous.prefix_selections ||
           report.current.state_forks != report.previous.state_forks ||
           report.current.materialization_state_forks !=
               report.previous.materialization_state_forks ||
           report.current.state_restores != report.previous.state_restores ||
           report.current.partial_tail_cow_pages != report.previous.partial_tail_cow_pages ||
           report.current.device_state_occupied_slots !=
               report.previous.device_state_occupied_slots ||
           report.current.device_main_kv_occupied_pages !=
               report.previous.device_main_kv_occupied_pages ||
           report.current.device_backend_kv_occupied_pages !=
               report.previous.device_backend_kv_occupied_pages ||
           report.current.host_work.engine_boundary_ns !=
               report.previous.host_work.engine_boundary_ns ||
           report.current.host_work.program_submit_ns !=
               report.previous.host_work.program_submit_ns ||
           report.current.host_work.program_post_ns != report.previous.host_work.program_post_ns ||
           report.current.host_work.engine_commit_output_ns !=
               report.previous.host_work.engine_commit_output_ns ||
           report.current.host_work.engine_maintenance_ns !=
               report.previous.host_work.engine_maintenance_ns ||
           report.current.host_work.device_wait_ns != report.previous.host_work.device_wait_ns;
}

const char* endpoint_name(std::string_view request_path) noexcept {
    const std::string_view path = canonical_api_path(request_path);
    if (path == "/v1/chat/completions") { return "openai_chat_completions"; }
    if (path == "/v1/responses") { return "openai_responses"; }
    if (path == "/v1/responses/input_tokens") { return "openai_responses_input_tokens"; }
    if (path == "/v1/messages") { return "anthropic_messages"; }
    if (path == "/v1/messages/count_tokens") { return "anthropic_count_tokens"; }
    if (path == "/v1/decide" || path == "/v1/systemone") { return "decide"; }
    return "http_route";
}

std::string response_request_id(const httplib::Response& response) {
    if (response.has_header("x-request-id")) { return response.get_header_value("x-request-id"); }
    if (response.has_header("request-id")) { return response.get_header_value("request-id"); }
    return {};
}

} // namespace

void write_openai_error(httplib::Response& response, const ApiError& error) {
    response.status = error.status;
    response.set_content(make_error_body(error), "application/json");
}

void write_anthropic_error(httplib::Response& response, const ApiError& api_error,
                           const std::string& request_id) {
    const ApiError error = normalize_anthropic_error(api_error);
    response.status      = error.status;
    response.headers.erase("request-id");
    response.set_header("request-id", request_id);
    response.set_content(make_anthropic_error_body(error, request_id), "application/json");
}

httplib::Server::HandlerResponse handle_unrendered_http_error(const ServeOptions& options,
                                                              const httplib::Request& request,
                                                              httplib::Response& response) {
    ensure_http_request_id(request, response);
    if (!response.body.empty()) { return httplib::Server::HandlerResponse::Unhandled; }

    ApiError error;
    if (response.status == 413) {
        error.status  = 413;
        error.type    = "invalid_request_error";
        error.code    = "request_too_large";
        error.message = "request body exceeds the configured payload limit of " +
                        std::to_string(options.max_request_bytes) + " bytes";
    } else if (response.status == 404 && is_anthropic_path(request.path)) {
        error.status  = 404;
        error.code    = "not_found";
        error.message = "requested Anthropic resource was not found";
    } else {
        return httplib::Server::HandlerResponse::Unhandled;
    }
    if (is_anthropic_path(request.path)) {
        write_anthropic_error(response, error, response.get_header_value("request-id"));
    } else {
        write_openai_error(response, error);
    }
    return httplib::Server::HandlerResponse::Handled;
}

bool matches_bearer_credential(std::string_view authorization, std::string_view api_key) noexcept {
    if (api_key.empty()) { return false; }
    const auto is_whitespace = [](char value) { return value == ' ' || value == '\t'; };
    const auto ascii_equal   = [](char lhs, char rhs) {
        if (lhs >= 'A' && lhs <= 'Z') { lhs = static_cast<char>(lhs - 'A' + 'a'); }
        if (rhs >= 'A' && rhs <= 'Z') { rhs = static_cast<char>(rhs - 'A' + 'a'); }
        return lhs == rhs;
    };

    std::size_t position = 0;
    while (position < authorization.size() && is_whitespace(authorization[position])) {
        ++position;
    }
    constexpr std::string_view scheme = "Bearer";
    if (authorization.size() - position < scheme.size()) { return false; }
    for (std::size_t index = 0; index < scheme.size(); ++index) {
        if (!ascii_equal(authorization[position + index], scheme[index])) { return false; }
    }
    position += scheme.size();
    if (position == authorization.size() || !is_whitespace(authorization[position])) {
        return false;
    }
    while (position < authorization.size() && is_whitespace(authorization[position])) {
        ++position;
    }
    std::size_t end = authorization.size();
    while (end > position && is_whitespace(authorization[end - 1])) { --end; }
    return authorization.substr(position, end - position) == api_key;
}

std::string api_route_pattern(std::string_view endpoint) {
    return "(?:/v1)?/v1" + std::string(endpoint);
}

std::string_view canonical_api_path(std::string_view path) noexcept {
    constexpr std::string_view kVersion = "/v1";
    if (path.starts_with("/v1/v1/")) { path.remove_prefix(kVersion.size()); }
    return path;
}

// /v1/internal/* is the local control plane (the yield broker asks the engine to hand its
// Device expert cache to another GPU tenant). It is not part of the OpenAI-compatible
// surface, so it is exempt from the API key exactly like /health - but it is refused off
// loopback, so a server bound to a routable interface cannot be told to drop its cache by
// a stranger.
bool is_internal_control_path(std::string_view path) {
    return canonical_api_path(path).starts_with("/v1/internal/");
}

bool is_loopback_peer(std::string_view addr) {
    return addr == "127.0.0.1" || addr == "::1" || addr == "::ffff:127.0.0.1";
}

HttpServer::HttpServer(ServeOptions options, std::shared_ptr<spdlog::logger> logger,
                       std::shared_ptr<product::TerminalPanel> panel)
    : options_(std::move(options)), openai_responses_store_(options_.response_store_max_records,
                                                            options_.response_store_max_bytes),
      operational_log_(logger), request_jsonl_(options_.request_log_jsonl, options_.artifact_path,
                                               std::move(logger), options_.request_log_rotation) {
    if (options_.log_stats_panel && panel != nullptr && panel->enabled()) {
        console_stats_ = std::make_unique<ConsoleStatsPanel>(
            std::move(panel), options_.speculative.ngram_draft_tokens != 0);
    }
    const std::size_t queued_requests =
        static_cast<std::size_t>(options_.max_concurrency) + options_.max_pending_requests;
    const std::size_t worker_count = queued_requests + 1;
    server_.new_task_queue         = [queued_requests, worker_count] {
        return new httplib::ThreadPool(worker_count, worker_count, queued_requests);
    };
    server_.set_socket_options(configure_http_server_socket);
    server_.set_payload_max_length(options_.max_request_bytes);
    register_routes();
}

HttpServer::RequestLifecycle::RequestLifecycle(HttpServer& owner, RequestLogContext context)
    : owner_(&owner), context_(std::move(context)) {
    owner_->record_request_start(context_);
}

bool HttpServer::RequestLifecycle::claim(State terminal) noexcept {
    State expected = State::Pending;
    return state_.compare_exchange_strong(expected, terminal, std::memory_order_acq_rel);
}

void HttpServer::RequestLifecycle::done(const GenerationOutcome& outcome) {
    if (claim(State::Done)) { owner_->record_request_done(context_, outcome); }
}

void HttpServer::RequestLifecycle::failure(const RequestFailure& failure) {
    if (claim(State::Error)) { owner_->record_request_failure(context_, failure); }
}

void HttpServer::RequestLifecycle::response_failure(const RequestFailure& failure) {
    owner_->record_response_failure(context_.id, failure);
}

std::shared_ptr<HttpServer::RequestLifecycle> HttpServer::begin_request(RequestLogContext context) {
    return std::make_shared<RequestLifecycle>(*this, std::move(context));
}

void HttpServer::record_request_start(const RequestLogContext& context) {
    request_jsonl_.write_request_start(context);
    operational_log_.request_start(context);
}

void HttpServer::record_request_rejected(const RequestRejectionLogContext& context) {
    metrics_.rejected();
    request_jsonl_.write_request_rejected(context);
    operational_log_.request_rejected(context);
    if (console_stats_) {
        console_stats_->request_rejected(
            make_request_failure(RequestFailurePhase::Prepare, context.error));
    }
}

infernix::GenerationFirstTokenObserver HttpServer::first_token_observer() {
    return [this](const infernix::GenerationFirstTokenObservation& observation) {
        metrics_.first_token(observation);
    };
}

infernix::GenerationSchedulingObserver HttpServer::scheduling_observer(std::uint64_t request_id,
                                                                     std::string http_request_id) {
    if (!request_jsonl_.enabled()) { return {}; }
    return [this, request_id, http_request_id = std::move(http_request_id)](
               const infernix::GenerationSchedulingObservation& observation) {
        request_jsonl_.write_request_scheduling(request_id, http_request_id, observation);
    };
}

void HttpServer::record_request_done(const RequestLogContext& context,
                                     const GenerationOutcome& outcome) {
    metrics_.done(outcome);
    request_jsonl_.write_request_done(context, outcome);
    operational_log_.request_done(context, outcome);
    if (console_stats_) { console_stats_->request_done(outcome); }
}

void HttpServer::record_request_failure(const RequestLogContext& context,
                                        const RequestFailure& failure) {
    metrics_.failed(failure.classification == RequestFailureClass::ClientDisconnected);
    request_jsonl_.write_request_error(context, failure.machine_message);
    operational_log_.request_failure(context, failure);
    if (console_stats_) { console_stats_->request_failure(failure); }
}

void HttpServer::record_response_failure(std::uint64_t request_id, const RequestFailure& failure) {
    metrics_.response_failed();
    operational_log_.response_failure(request_id, failure);
}

void HttpServer::record_throughput(const ThroughputReport& report) {
    request_jsonl_.write_throughput(report);
    operational_log_.throughput(report);
}

void HttpServer::run_stats_reporter() {
    using Clock                     = std::chrono::steady_clock;
    infernix::RuntimeStats previous   = service_->runtime_stats();
    Clock::time_point previous_time = Clock::now();
    const auto interval             = std::chrono::milliseconds(options_.log_stats_interval_ms);
    Clock::time_point next_deadline = previous_time + interval;

    for (;;) {
        {
            std::unique_lock lock(stats_mutex_);
            if (stats_cv_.wait_until(lock, next_deadline, [this] { return stats_stopping_; })) {
                break;
            }
        }

        const infernix::RuntimeStats current = service_->runtime_stats();
        const Clock::time_point now        = Clock::now();
        const ThroughputReport report      = make_throughput_report(
            previous, current, std::chrono::duration<double>(now - previous_time).count());
        if (report_has_activity(report)) { record_throughput(report); }
        if (console_stats_) { console_stats_->runtime(current); }
        previous      = current;
        previous_time = now;
        next_deadline += interval;
        const Clock::time_point after_write = Clock::now();
        if (next_deadline <= after_write) { next_deadline = after_write + interval; }
    }

    const infernix::RuntimeStats current = service_->runtime_stats();
    const Clock::time_point now        = Clock::now();
    ThroughputReport tail              = make_throughput_report(
        previous, current, std::chrono::duration<double>(now - previous_time).count());
    // The exact partial interval remains useful to measurement consumers. Pretty throughput is a
    // fixed-cadence operational record and deliberately has no irregular shutdown tail.
    tail.final_interval = true;
    request_jsonl_.write_throughput(tail);
}

void HttpServer::stop_stats_reporter() {
    if (!stats_thread_.joinable()) { return; }
    {
        std::lock_guard lock(stats_mutex_);
        stats_stopping_ = true;
    }
    stats_cv_.notify_one();
    stats_thread_.join();
}

void HttpServer::register_routes() {
    server_.set_error_handler([this](const httplib::Request& request, httplib::Response& response) {
        return handle_unrendered_http_error(options_, request, response);
    });
    if (options_.enable_cors) {
        server_.set_default_headers(
            {{"Access-Control-Allow-Origin", "*"},
             {"Access-Control-Expose-Headers", "x-request-id, request-id"},
             {"Access-Control-Allow-Headers",
              "Authorization, Content-Type, X-API-Key, anthropic-version, anthropic-beta, "
              "anthropic-user-profile-id"},
             {"Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS"}});
        // CORS preflight: browsers send OPTIONS with no credentials before the real
        // request; answer it without auth so the actual GET/POST can carry the key.
        server_.Options(R"(.*)",
                        [](const httplib::Request&, httplib::Response& res) { res.status = 204; });
    }

    server_.set_pre_routing_handler([this](const httplib::Request& req, httplib::Response& res) {
        ensure_http_request_id(req, res);
        if (options_.api_key.empty() || req.path == "/health" || req.method == "OPTIONS" ||
            is_internal_control_path(req.path)) {
            return httplib::Server::HandlerResponse::Unhandled;
        }
        // Accept both the OpenAI-style bearer token and the Anthropic-style
        // x-api-key header so OpenAI clients and Claude Code (ANTHROPIC_API_KEY
        // -> x-api-key, ANTHROPIC_AUTH_TOKEN -> Authorization: Bearer) both work.
        const bool bearer_ok =
            matches_bearer_credential(req.get_header_value("Authorization"), options_.api_key);
        const bool x_api_key_ok = req.get_header_value("x-api-key") == options_.api_key;
        if (!bearer_ok && !x_api_key_ok) {
            ApiError error;
            error.status  = 401;
            error.type    = "invalid_request_error";
            error.code    = "invalid_api_key";
            error.message = "missing or invalid API key";
            // Render the 401 in the shape the target endpoint speaks.
            if (is_anthropic_path(req.path)) {
                write_anthropic_error(res, error, res.get_header_value("request-id"));
            } else {
                write_openai_error(res, error);
            }
            return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });

    server_.set_exception_handler(
        [this](const httplib::Request& req, httplib::Response& res, std::exception_ptr ep) {
            ensure_http_request_id(req, res);
            try {
                std::rethrow_exception(ep);
            } catch (const ApiException& e) {
                if (e.error().status >= 500) {
                    operational_log_.http_failure(
                        endpoint_name(req.path),
                        make_request_failure(RequestFailurePhase::Http, e.error()),
                        response_request_id(res));
                }
                if (is_anthropic_path(req.path)) {
                    write_anthropic_error(res, e.error(), res.get_header_value("request-id"));
                } else {
                    write_openai_error(res, e.error());
                }
            } catch (const std::exception& e) {
                operational_log_.http_failure(
                    endpoint_name(req.path),
                    make_internal_request_failure(RequestFailurePhase::Http, e.what()),
                    response_request_id(res));
                if (is_anthropic_path(req.path)) {
                    ApiError error;
                    error.status  = 500;
                    error.message = e.what();
                    write_anthropic_error(res, error, res.get_header_value("request-id"));
                } else {
                    write_exception(res, e);
                }
            } catch (...) {
                operational_log_.http_failure(
                    endpoint_name(req.path),
                    make_internal_request_failure(RequestFailurePhase::Http, "unknown error"),
                    response_request_id(res));
                ApiError error;
                error.status  = 500;
                error.type    = "internal_error";
                error.message = "unknown error";
                if (is_anthropic_path(req.path)) {
                    write_anthropic_error(res, error, res.get_header_value("request-id"));
                } else {
                    write_openai_error(res, error);
                }
            }
        });

    server_.Get("/health", [this](const httplib::Request&, httplib::Response& res) {
        const bool available = service_ != nullptr && service_->is_available();
        res.status           = available ? 200 : 503;
        res.set_content(nlohmann::json{{"status", available ? "ok" : "unavailable"}}.dump(),
                        "application/json");
    });
    server_.Get(api_route_pattern("/models"),
                [this](const httplib::Request& req, httplib::Response& res) {
                    handle_models(req, res);
                });
    server_.Get("/metrics", [this](const httplib::Request&, httplib::Response& res) {
        const auto stats = service_->runtime_stats();
        res.set_header("Cache-Control", "no-store");
        res.set_content(metrics_.render(stats, service_->is_available()),
                        "text/plain; version=0.0.4; charset=utf-8");
    });
    server_.Get(api_route_pattern("/internal/residency"),
                [this](const httplib::Request& req, httplib::Response& res) {
                    handle_residency_status(req, res);
                });
    server_.Post(api_route_pattern("/internal/residency"),
                 [this](const httplib::Request& req, httplib::Response& res) {
                     handle_residency_request(req, res);
                 });
    server_.Get(api_route_pattern(R"(/models/(.+))"),
                [this](const httplib::Request& req, httplib::Response& res) {
                    handle_model(req, res);
                });
    server_.Post(api_route_pattern("/chat/completions"),
                 [this](const httplib::Request& req, httplib::Response& res) {
                     handle_chat_completions(req, res);
                 });
    server_.Post(api_route_pattern("/responses"),
                 [this](const httplib::Request& req, httplib::Response& res) {
                     handle_responses(req, res);
                 });
    server_.Post(api_route_pattern("/responses/input_tokens"),
                 [this](const httplib::Request& req, httplib::Response& res) {
                     handle_response_input_tokens(req, res);
                 });
    server_.Post(api_route_pattern("/responses/compact"),
                 [this](const httplib::Request& req, httplib::Response& res) {
                     handle_response_compact(req, res);
                 });
    server_.Post(api_route_pattern(R"(/responses/([^/]+)/cancel)"),
                 [this](const httplib::Request& req, httplib::Response& res) {
                     handle_response_cancel(req, res);
                 });
    server_.Get(api_route_pattern(R"(/responses/([^/]+)/input_items)"),
                [this](const httplib::Request& req, httplib::Response& res) {
                    handle_response_input_items(req, res);
                });
    server_.Get(api_route_pattern(R"(/responses/([^/]+))"),
                [this](const httplib::Request& req, httplib::Response& res) {
                    handle_response_get(req, res);
                });
    server_.Delete(api_route_pattern(R"(/responses/([^/]+))"),
                   [this](const httplib::Request& req, httplib::Response& res) {
                       handle_response_delete(req, res);
                   });
    server_.Post(api_route_pattern("/messages/count_tokens"),
                 [this](const httplib::Request& req, httplib::Response& res) {
                     handle_count_tokens(req, res);
                 });
    server_.Post(api_route_pattern("/messages"),
                 [this](const httplib::Request& req, httplib::Response& res) {
                     handle_messages(req, res);
                 });
    // /v1/systemone is TypeSafe Jev's name for the same request, so a Jev client needs only a URL.
    for (const char* endpoint : {"/decide", "/systemone"}) {
        server_.Post(api_route_pattern(endpoint),
                     [this](const httplib::Request& req, httplib::Response& res) {
                         handle_decide(req, res);
                     });
    }
}

void HttpServer::handle_models(const httplib::Request&, httplib::Response& res) const {
    res.set_content(make_models_list(public_model_id_, unix_time_now(), options_.max_context,
                                     model_metadata_),
                    "application/json");
}

void HttpServer::handle_model(const httplib::Request& req, httplib::Response& res) const {
    const std::string id = req.matches.size() > 1 ? req.matches[1].str() : std::string();
    if (id != public_model_id_) {
        ApiError error;
        error.status  = 404;
        error.type    = "invalid_request_error";
        error.code    = "model_not_found";
        error.message = "model '" + id + "' not found";
        write_openai_error(res, error);
        return;
    }
    res.set_content(make_model_object(public_model_id_, unix_time_now(), options_.max_context,
                                      model_metadata_),
                    "application/json");
}

namespace {

// The elastic expert cache's state as the internal residency endpoint reports it, including the
// SSD tier's pinned Host memory (the level a release can hand back to the OS with the engine
// still serving, design §19.3.7).
nlohmann::json residency_json(const infernix::ResidencyStatus& status) {
    const infernix::HostResidencyStatus& host = status.host;
    return nlohmann::json{
        {"supported", status.supported},
        {"forced", status.forced},
        {"restore_pending", status.restore_pending},
        {"requested_frames", status.requested_frames},
        {"frames", status.frames},
        {"startup_frames", status.startup_frames},
        {"control_target", status.control_target},
        {"frame_limit", status.frame_limit},
        {"frame_bytes", status.frame_bytes},
        {"pool_bytes", status.pool_bytes},
        {"pool_mib", static_cast<std::uint64_t>(status.pool_bytes >> 20)},
        {"last_resize_seconds", status.last_resize_seconds},
        {"host",
         nlohmann::json{
             {"supported", host.supported},
             {"forced", host.forced},
             {"restore_pending", host.restore_pending},
             {"bytes", host.bytes},
             {"mib", host.bytes >> 20},
             {"bytes_max", host.bytes_max},
             {"bytes_floor", host.bytes_floor},
             {"released_bytes", host.released_bytes},
             {"slots", host.slots},
             {"slot_limit", host.slot_limit},
             {"slot_bytes", host.slot_bytes},
             {"slots_per_chunk", host.slots_per_chunk},
             {"resident_keys", host.resident_keys},
             {"key_count", host.key_count},
             {"guard", host.guard},
             {"guard_releases", host.guard_releases},
             {"guard_owned", host.guard_owned},
             {"last_slots", host.last_slots},
             {"last_keys", host.last_keys},
             {"last_op_seconds", host.last_op_seconds},
         }},
    };
}

void write_residency_error(httplib::Response& res, int status, const char* message) {
    ApiError error;
    error.status  = status;
    error.type    = "invalid_request_error";
    error.message = message;
    write_openai_error(res, error);
}

} // namespace

void HttpServer::handle_residency_status(const httplib::Request& req, httplib::Response& res) const {
    res.set_header("Cache-Control", "no-store");
    if (!is_loopback_peer(req.remote_addr)) {
        write_residency_error(res, 403, "the residency control plane is loopback-only");
        return;
    }
    if (service_ == nullptr) {
        ApiError error;
        error.status  = 503;
        error.type    = "internal_error";
        error.message = "inference engine is not attached";
        write_openai_error(res, error);
        return;
    }
    res.set_content(residency_json(service_->residency_status()).dump(), "application/json");
}

void HttpServer::handle_residency_request(const httplib::Request& req, httplib::Response& res) {
    res.set_header("Cache-Control", "no-store");
    if (!is_loopback_peer(req.remote_addr)) {
        write_residency_error(res, 403, "the residency control plane is loopback-only");
        return;
    }
    if (service_ == nullptr) {
        ApiError error;
        error.status  = 503;
        error.type    = "internal_error";
        error.message = "inference engine is not attached";
        write_openai_error(res, error);
        return;
    }
    std::string action   = "release";
    std::uint32_t frames = 0;
    long long wait_ms    = 120000;
    // The SSD tier's pinned Host memory is untouched unless the body names it, so the plain
    // {"action":"release"} of the Device cache keeps meaning exactly that.
    std::string host_memory  = "keep"; // "keep" | "release" | "restore"
    std::uint64_t host_bytes = 0;
    if (!req.body.empty()) {
        try {
            const auto body = nlohmann::json::parse(req.body);
            if (body.contains("action")) { action = body.at("action").get<std::string>(); }
            if (body.contains("frames")) { frames = body.at("frames").get<std::uint32_t>(); }
            if (body.contains("wait_ms")) { wait_ms = body.at("wait_ms").get<long long>(); }
            if (body.contains("host")) { host_memory = body.at("host").get<std::string>(); }
            if (body.contains("host_bytes")) {
                host_bytes = body.at("host_bytes").get<std::uint64_t>();
                if (!body.contains("host") && host_bytes != 0) { host_memory = action; }
            }
        } catch (const std::exception& error) {
            write_residency_error(res, 400,
                                  (std::string("invalid request body: ") + error.what()).c_str());
            return;
        }
    }
    if (action != "release" && action != "restore") {
        write_residency_error(res, 400, "action must be \"release\" or \"restore\"");
        return;
    }
    if (host_memory != "keep" && host_memory != "release" && host_memory != "restore") {
        write_residency_error(res, 400, "host must be \"keep\", \"release\" or \"restore\"");
        return;
    }

    const bool release    = action == "release";
    const bool host_want  = host_memory != "keep";
    const auto started = std::chrono::steady_clock::now();
    const infernix::ResidencyStatus before = service_->residency_status();
    if (!before.supported) {
        res.status = 409;
        res.set_header("Cache-Control", "no-store");
        res.set_content(nlohmann::json{{"ok", false},
                                       {"action", action},
                                       {"error", "this engine has no elastic Device expert cache"},
                                       {"before", residency_json(before)}}
                            .dump(),
                        "application/json");
        return;
    }
    if (host_want && !before.host.supported) {
        res.status = 409;
        res.set_header("Cache-Control", "no-store");
        res.set_content(nlohmann::json{{"ok", false},
                                       {"action", action},
                                       {"host_action", host_memory},
                                       {"error", "this engine has no SSD expert tier with pinned Host slots"},
                                       {"before", residency_json(before)}}
                            .dump(),
                        "application/json");
        return;
    }

    service_->request_residency(release, frames);
    if (host_want) { service_->request_host_residency(host_memory == "release", host_bytes); }
    const auto deadline = started + std::chrono::milliseconds(wait_ms < 0 ? 0 : wait_ms);
    const auto settled  = [&](const infernix::ResidencyStatus& status) {
        // A release is done once the frames are down to what was asked and no restore is queued; a
        // restore is done once the request is withdrawn (the control law may grow the cache later).
        const bool vram_done = release ? (status.frames <= frames && !status.restore_pending)
                                       : (!status.forced && !status.restore_pending);
        // A Host-memory request is done once it has been carried out at an idle boundary; the
        // caller reads the bytes it moved from the before/after blocks.
        const bool host_done = !host_want || (!status.host.forced && !status.host.restore_pending);
        return vram_done && host_done;
    };
    infernix::ResidencyStatus after;
    for (;;) {
        after = service_->residency_status();
        if (settled(after) || std::chrono::steady_clock::now() >= deadline) { break; }
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
    }
    const bool done = settled(after);
    const auto waited = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now() - started)
                            .count();
    res.status = done ? 200 : 202;
    res.set_header("Cache-Control", "no-store");
    res.set_content(nlohmann::json{{"ok", done},
                                   {"action", action},
                                   {"requested_frames", frames},
                                   {"host_action", host_memory},
                                   {"host_bytes_requested", host_bytes},
                                   {"waited_ms", waited},
                                   {"before", residency_json(before)},
                                   {"after", residency_json(after)}}
                        .dump(),
                    "application/json");
}

bool HttpServer::bind() { return server_.bind_to_port(options_.host, options_.port); }
void HttpServer::attach(GenerationService& service) {
    if (service_ != nullptr) {
        throw std::logic_error("HTTP generation service is already attached");
    }
    const infernix::LoadSummary load = service.load_summary();
    // memory_summary waits for the completed worker boundary, including warmup publication.
    const infernix::MemorySummary memory  = service.memory_summary();
    const infernix::RuntimeStats baseline = service.runtime_stats();
    public_model_id_                    = resolve_public_model_id(options_, load.model_name);
    model_metadata_                     = service.model_metadata();
    decide_labels_                      = build_decide_alphabet(
        [&service](std::string_view text) { return service.tokenize_text(text); });
    service_                            = &service;
    metrics_.configure(public_model_id_, service.engine_options(), memory, baseline);
    request_jsonl_.write_server_start(options_, service.engine_options(),
                                      service.sampling_defaults(), public_model_id_, load, memory);
}

bool HttpServer::listen() {
    if (service_ == nullptr) { throw std::logic_error("HTTP generation service is not attached"); }
    if (public_model_id_.empty()) {
        throw std::logic_error("HTTP public model id is not resolved");
    }
    if (console_stats_) { console_stats_->show(); }
    if (options_.log_stats_interval_ms != 0) {
        stats_stopping_ = false;
        stats_thread_   = std::thread([this] { run_stats_reporter(); });
    }
    try {
        const bool result = server_.listen_after_bind();
        stop_stats_reporter();
        return result;
    } catch (...) {
        stop_stats_reporter();
        throw;
    }
}

void HttpServer::stop() {
    server_.stop();
    if (service_ != nullptr) { service_->stop(); }
}

} // namespace infernix::serve
