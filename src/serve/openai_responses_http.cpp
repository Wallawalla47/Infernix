#include "serve/http_server.h"

#include "serve/http_stream_driver.h"
#include "serve/http_transport.h"
#include "serve/openai_common.h"
#include "serve/openai_responses.h"
#include "serve/request_validation.h"
#include "serve/server_tools.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <iterator>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace infernix::serve {
namespace {

using Json = nlohmann::json;

struct PendingResponseStorage {
    std::vector<ChatTurn> input_turns;
    std::vector<Json> input_items;
    OpenAIResponseContext previous_context;
    bool enabled = false;
};

// The generations of one Response. When the model calls a hosted tool_search, the server runs the
// search, appends its tool_search_output and runs another generation, until the model answers or
// calls a client tool, or a limit ends the Response.
struct ResponseRounds {
    GenerationRequest continuation; // the request plus every completed round
    std::vector<ServerToolRound> rounds;
    std::vector<std::vector<std::string>> round_call_ids;
    std::vector<std::vector<std::optional<OpenAIToolSearchExecution>>> round_searches;
    std::size_t hosted_calls = 0;

    GenerationOutcome final_outcome;
    std::vector<std::string> final_call_ids;
    std::vector<std::optional<OpenAIToolSearchExecution>> final_searches;
    std::optional<infernix::FinishReason> finish;

    [[nodiscard]] OpenAIResponsesRound round(std::size_t index) const {
        return OpenAIResponsesRound{.outcome  = &rounds[index].outcome,
                                    .call_ids = round_call_ids[index],
                                    .searches = round_searches[index]};
    }

    [[nodiscard]] std::vector<OpenAIResponsesRound> all() const {
        std::vector<OpenAIResponsesRound> result;
        for (std::size_t index = 0; index < rounds.size(); ++index) {
            result.push_back(round(index));
        }
        result.push_back(OpenAIResponsesRound{
            .outcome = &final_outcome, .call_ids = final_call_ids, .searches = final_searches});
        return result;
    }

    [[nodiscard]] GenerationOutcome totals() const {
        return combine_round_outcomes(rounds, final_outcome);
    }
};

using ContinuationPreparer = std::function<PreparedRequest(const GenerationRequest&)>;
using RoundWriter          = std::function<void(const OpenAIResponsesRound&)>;
// The identifier of a generation's hosted search call `index`: a stream's keeps the Item id it
// streamed the call with.
using HostedSearchId = std::function<std::string(std::size_t index)>;

// Runs `prepared` and every continuation its hosted searches lead to, leaving the final generation
// in `state`; `prepared` ends holding the last generation's request and ingress reservation.
void run_openai_response(GenerationBackend& service, const OpenAIResponsesCreateRequest& request,
                         ResponseRounds& state, PreparedRequest& prepared, const StreamSink* sink,
                         const std::function<bool()>& is_cancelled,
                         const ContinuationPreparer& prepare_next, const RoundWriter& on_round,
                         const HostedSearchId& hosted_search_id) {
    // The protocol's part of run_server_tool_rounds: a hosted tool_search runs (counted against max_tool_calls),
    // any other call is the client's, and each generation keeps its searches' Item ids and results.
    struct Protocol {
        const OpenAIResponsesCreateRequest& request;
        ResponseRounds& state;
        const RoundWriter& on_round;
        const HostedSearchId& hosted_search_id;
        const bool hosted;
        const std::string& search_name;
        std::vector<std::string> call_ids;                               // the current generation's, by call
        std::vector<std::optional<OpenAIToolSearchExecution>> searches; // likewise

        ServerRoundCalls calls(const GenerationOutcome& outcome) {
            const std::size_t count = outcome.tool_calls.size();
            call_ids.assign(count, std::string());
            searches.assign(count, std::nullopt);
            ServerRoundCalls out;
            out.calls.resize(count);
            for (std::size_t index = 0; index < count; ++index) {
                const infernix::GeneratedToolCall& call = outcome.tool_calls[index];
                out.calls[index] = ToolCall{.name = call.name, .arguments_json = call.arguments_json};
                if (!hosted || call.name != search_name ||
                    outcome.finish_reason == infernix::FinishReason::Cancelled) {
                    out.client_calls = true;
                    continue;
                }
                call_ids[index]     = hosted_search_id(index);
                out.calls[index].id = call_ids[index];
                searches[index]     = execute_openai_tool_search(request, out.calls[index]);
                out.results.push_back(searches[index]->turn);
                ++state.hosted_calls;
            }
            return out;
        }

        [[nodiscard]] bool limit_reached() const noexcept {
            return request.max_tool_calls && state.hosted_calls >= static_cast<std::size_t>(*request.max_tool_calls);
        }

        void keep() {
            state.round_call_ids.push_back(std::move(call_ids));
            state.round_searches.push_back(std::move(searches));
            if (on_round) { on_round(state.round(state.rounds.size() - 1U)); }
        }

        void end(ServerToolRound&& round, ServerRoundsEnd end) {
            state.final_outcome  = std::move(round.outcome);
            state.final_call_ids = std::move(call_ids);
            state.final_searches = std::move(searches);
            switch (end) {
            case ServerRoundsEnd::Finished:
            case ServerRoundsEnd::RoundLimit: state.finish = std::nullopt; break;
            case ServerRoundsEnd::OutputLimit: state.finish = infernix::FinishReason::OutputLimit; break;
            case ServerRoundsEnd::ContextWindow: state.finish = infernix::FinishReason::ContextCapacity; break;
            }
        }
    };
    const std::string search_name(kToolSearchFunctionName);
    Protocol protocol{.request          = request,
                      .state            = state,
                      .on_round         = on_round,
                      .hosted_search_id = hosted_search_id,
                      .hosted           = request.prompt.tool_search && request.prompt.tool_search->server,
                      .search_name      = search_name};
    run_server_tool_rounds(service, prepared, sink, is_cancelled, prepare_next, state.continuation, state.rounds,
                           std::span(&search_name, 1), protocol);
}

struct StreamingResponse {
    PreparedRequest prepared;
    PendingResponseStorage storage;
    std::unique_ptr<OpenAIResponsesEventStream> encoder;
    ResponseRounds rounds;
    std::atomic<bool> cancelled{false};
    std::atomic<bool> started{false};
};

ApiError responses_error(ApiError error) {
    if (error.param == "messages") { error.param = "input"; }
    if (error.param == "reasoning_effort") { error.param = "reasoning.effort"; }
    return error;
}

ApiError response_not_found(const std::string& id) {
    ApiError error;
    error.status  = 404;
    error.type    = "invalid_request_error";
    error.param   = "response_id";
    error.code    = "response_not_found";
    error.message = "response '" + id + "' not found";
    return error;
}

OpenAIResponseContext terminal_context(OpenAIResponseContext previous,
                                       std::vector<ChatTurn> input_turns,
                                       std::vector<ChatTurn> output_history) {
    OpenAIResponseContext input =
        append_openai_response_context(std::move(previous), std::move(input_turns));
    return append_openai_response_context(std::move(input), std::move(output_history));
}

void commit_stored_response(OpenAIResponsesStore& store, PendingResponseStorage pending,
                            std::string id, const Json& response,
                            std::vector<ChatTurn> output_history,
                            std::optional<bool> preserve_thinking) {
    if (!pending.enabled) { return; }
    StoredOpenAIResponse stored;
    stored.id                = std::move(id);
    stored.response          = response;
    stored.input_items       = std::move(pending.input_items);
    stored.context           = terminal_context(std::move(pending.previous_context),
                                                std::move(pending.input_turns), std::move(output_history));
    stored.preserve_thinking = preserve_thinking;
    store.put(std::move(stored));
}

OpenAIResponsesRuntimeValues runtime_values(const PreparedRequest& prepared,
                                            const GenerationOutcome* outcome = nullptr) {
    OpenAIResponsesRuntimeValues runtime;
    runtime.temperature = prepared.sampling.temperature;
    runtime.top_p       = prepared.sampling.top_p;
    if (outcome != nullptr) {
        runtime.cached_input_tokens = static_cast<int>(outcome->reused_prompt_tokens);
    }
    return runtime;
}

// A streamed Response refused before its generation, answered as one whole failed event stream.
// It reports the sampling the request named, or the Responses default: the Engine resolves the
// model's own defaults only for a prepared prompt.
void write_failed_response_stream(httplib::Response& res, const std::string& id,
                                  OpenAIResponsesCreateRequest request,
                                  const GenerationRequest& generation, const ApiError& error) {
    OpenAIResponsesRuntimeValues runtime;
    if (generation.sampling.temperature) { runtime.temperature = static_cast<float>(*generation.sampling.temperature); }
    if (generation.sampling.top_p) { runtime.top_p = static_cast<float>(*generation.sampling.top_p); }
    prepare_sse_response(res);
    res.set_content(make_openai_failed_response_stream(id, unix_time_now(), std::move(request),
                                                       runtime, error),
                    "text/event-stream");
}

[[noreturn]] void invalid_query(std::string message, std::string param,
                                std::string code = "invalid_value") {
    bad_request(std::move(message), std::move(param), std::move(code));
}

void reject_unknown_query(const httplib::Request& request,
                          const std::vector<std::string_view>& allowed) {
    for (const auto& [key, value] : request.params) {
        (void)value;
        if (std::find(allowed.begin(), allowed.end(), key) == allowed.end()) {
            invalid_query("unknown query parameter: " + key, key, "unknown_parameter");
        }
    }
}

bool query_bool(const httplib::Request& request, const char* key, bool fallback) {
    if (!request.has_param(key)) { return fallback; }
    const std::string value = request.get_param_value(key);
    if (value == "true") { return true; }
    if (value == "false") { return false; }
    invalid_query(std::string(key) + " must be 'true' or 'false'", key);
}

void validate_retrieve_query(const httplib::Request& request) {
    reject_unknown_query(request, {"include", "include_obfuscation", "starting_after", "stream"});
    if (request.has_param("include") && !request.get_param_value("include").empty()) {
        invalid_query("additional stored Response fields are not available", "include",
                      "include_not_supported");
    }
    (void)query_bool(request, "include_obfuscation", true);
    if (request.has_param("starting_after") && !request.get_param_value("starting_after").empty()) {
        invalid_query("stream event recovery is not available for stored Responses",
                      "starting_after", "stream_recovery_not_supported");
    }
    if (query_bool(request, "stream", false)) {
        invalid_query("stored Response event streams are not retained", "stream",
                      "retrieve_stream_not_supported");
    }
}

void redact_input_image_urls(Json& item) {
    if (!item.is_object()) { return; }
    const auto redact_parts = [](Json& parts) {
        if (!parts.is_array()) { return; }
        for (Json& part : parts) {
            if (part.is_object() && part.value("type", "") == "input_image") {
                part.erase("image_url");
            }
        }
    };
    if (item.contains("content")) { redact_parts(item["content"]); }
    if ((item.value("type", "") == "function_call_output" ||
         item.value("type", "") == "custom_tool_call_output") &&
        item.contains("output")) {
        redact_parts(item["output"]);
    }
}

std::string path_response_id(const httplib::Request& request) {
    return request.matches.size() > 1 ? request.matches[1].str() : std::string();
}

int parse_limit(const httplib::Request& request) {
    if (!request.has_param("limit")) { return 20; }
    const std::string value = request.get_param_value("limit");
    int parsed              = 0;
    const auto result       = std::from_chars(value.data(), value.data() + value.size(), parsed);
    if (result.ec != std::errc{} || result.ptr != value.data() + value.size() || parsed < 1 ||
        parsed > 100) {
        ApiError error;
        error.status  = 400;
        error.param   = "limit";
        error.code    = "invalid_pagination";
        error.message = "limit must be an integer in [1,100]";
        throw ApiException(std::move(error));
    }
    return parsed;
}

Json paginated_input_items(const httplib::Request& request, const std::vector<Json>& stored_items) {
    reject_unknown_query(request, {"after", "include", "limit", "order"});
    bool include_image_urls = false;
    if (request.has_param("include") && !request.get_param_value("include").empty()) {
        if (request.get_param_value("include") != "message.input_image.image_url") {
            invalid_query("only include=message.input_image.image_url is available", "include",
                          "include_not_supported");
        }
        include_image_urls = true;
    }
    const int limit   = parse_limit(request);
    std::string order = request.has_param("order") ? request.get_param_value("order") : "desc";
    if (order != "asc" && order != "desc") {
        ApiError error;
        error.status  = 400;
        error.param   = "order";
        error.code    = "invalid_pagination";
        error.message = "order must be 'asc' or 'desc'";
        throw ApiException(std::move(error));
    }

    std::vector<Json> ordered = stored_items;
    if (order == "desc") { std::reverse(ordered.begin(), ordered.end()); }
    std::size_t begin = 0;
    if (request.has_param("after")) {
        const std::string after = request.get_param_value("after");
        const auto found = std::find_if(ordered.begin(), ordered.end(), [&](const Json& item) {
            return item.contains("id") && item.at("id").is_string() &&
                   item.at("id").get<std::string>() == after;
        });
        if (found == ordered.end()) {
            ApiError error;
            error.status  = 400;
            error.param   = "after";
            error.code    = "invalid_pagination";
            error.message = "after does not identify an input Item in this response";
            throw ApiException(std::move(error));
        }
        begin = static_cast<std::size_t>(std::distance(ordered.begin(), found)) + 1;
    }
    const std::size_t end = std::min(ordered.size(), begin + static_cast<std::size_t>(limit));
    Json data             = Json::array();
    for (std::size_t index = begin; index < end; ++index) {
        Json item = ordered[index];
        if (!include_image_urls) { redact_input_image_urls(item); }
        data.push_back(std::move(item));
    }
    return Json{{"object", "list"},
                {"data", data},
                {"first_id", data.empty() ? Json(nullptr) : data.front().at("id")},
                {"last_id", data.empty() ? Json(nullptr) : data.back().at("id")},
                {"has_more", end < ordered.size()}};
}

} // namespace

void HttpServer::handle_responses(const httplib::Request& req, httplib::Response& res) {
    OpenAIResponsesCreateRequest request;
    OpenAIResponsesResolvedPrompt resolved;
    const std::string id = new_openai_response_id();
    try {
        require_generation();
        RequestLimits limits;
        limits.default_max_tokens = options_.default_max_tokens;
        const auto body           = parse_json_body(req);
        request                   = parse_openai_responses_create_request(body, limits);
        request.prompt.model = resolve_openai_model(request.prompt.model, public_model_id_, options_.any_model_name);
        resolved = resolve_openai_responses_prompt(request.prompt, openai_responses_store_);
        resolved.generation.ngram_session = resolve_ngram_session(req, body, options_);
        if (request.store) {
            openai_responses_store_.check_input_capacity(resolved.parent, request.prompt.input_turns,
                                                         request.prompt.input_items);
        }
    } catch (const ApiException& exception) {
        write_openai_error(res, responses_error(exception.error()));
        return;
    } catch (const std::exception& exception) {
        operational_log_.http_failure(
            "openai_responses",
            make_internal_request_failure(RequestFailurePhase::Http, exception.what()));
        write_openai_error(res, internal_server_error(exception, "server_error"));
        return;
    }

    const std::uint64_t req_id = ++request_seq_;
    const RequestLogMetadata metadata{
        .http_request_id                   = res.get_header_value("x-request-id"),
        .response_id                       = id,
        .model                             = request.prompt.model,
        .stream                            = request.stream,
        .output_tokens_explicit            = request.requested_max_output_tokens.has_value(),
        .preserve_thinking_semantic_change = resolved.preserve_thinking_semantic_change,
        .label                             = request_label(req),
        .client_request_id                 = client_header_id(req, "x-request-id"),
        .session_id                        = client_header_id(req, "x-session-id"),
    };
    PreparedRequest prepared;
    try {
        prepared = service_->prepare(
            resolved.generation,
            request.stream ? GenerationConsumerMode::Streaming : GenerationConsumerMode::Aggregate,
            {.phase_timings = true,
             .scheduling    = scheduling_observer(req_id, metadata.http_request_id),
             .first_token   = first_token_observer(),
             .progress      = progress_observer(req_id)},
            [&req] { return client_disconnected(req); });
    } catch (const ApiException& exception) {
        const ApiError error = responses_error(exception.error());
        record_request_rejected(make_request_rejection_log_context(
            req_id, "openai_responses", resolved.generation, metadata, error));
        if (request.stream && error.code == "context_length_exceeded") {
            // Codex starts compaction only on this failure inside the stream (its HTTP error path
            // does not), so a streamed request answers with the whole failed event stream.
            write_failed_response_stream(res, id, std::move(request), resolved.generation, error);
            return;
        }
        write_openai_error(res, error);
        return;
    } catch (const std::exception& exception) {
        const ApiError error = internal_server_error(exception, "server_error");
        record_request_rejected(make_request_rejection_log_context(
            req_id, "openai_responses", resolved.generation, metadata, error));
        write_openai_error(res, error);
        return;
    }

    const std::int64_t created = unix_time_now();
    auto lifecycle             = begin_request(make_request_log_context(
        req_id, "openai_responses", resolved.generation, metadata, prepared));
    const GenerationConsumerMode consumer_mode =
        request.stream ? GenerationConsumerMode::Streaming : GenerationConsumerMode::Aggregate;
    // A continuation after a hosted search keeps the request's ingress reservation and log
    // identity; its first token is not the Response's.
    const std::string http_request_id = metadata.http_request_id;
    const auto continuation_preparer  = [this, consumer_mode, req_id,
                                         http_request_id](const std::function<bool()>& cancelled) {
        return ContinuationPreparer([this, consumer_mode, req_id, http_request_id,
                                     cancelled](const GenerationRequest& next) {
            return service_->prepare_continuation(
                next, consumer_mode,
                {.phase_timings = true,
                 .scheduling    = scheduling_observer(req_id, http_request_id),
                 .progress      = progress_observer(req_id)},
                cancelled);
        });
    };
    ResponseRounds rounds;
    // Only a hosted tool search continues the conversation; otherwise its copy is not needed.
    if (request.prompt.tool_search && request.prompt.tool_search->server) {
        rounds.continuation = std::move(resolved.generation);
    }
    resolved.generation.messages.clear();

    if (!request.stream) {
        const std::function<bool()> cancelled = [&req] { return client_disconnected(req); };
        try {
            run_openai_response(*service_, request, rounds, prepared, nullptr, cancelled,
                                continuation_preparer(cancelled), {},
                                [](std::size_t) { return new_openai_response_item_id("tsc"); });
        } catch (const ApiException& exception) {
            const ApiError error = responses_error(exception.error());
            lifecycle->failure(make_generation_request_failure(error));
            write_openai_error(res, error);
            return;
        } catch (const std::exception& exception) {
            lifecycle->failure(
                make_internal_request_failure(RequestFailurePhase::Generation, exception.what()));
            write_openai_error(res, internal_server_error(exception, "server_error"));
            return;
        }
        const GenerationOutcome outcome = rounds.totals();
        lifecycle->done(outcome);

        std::optional<BuiltOpenAIResponse> response;
        try {
            const OpenAIResponsesRuntimeValues runtime = runtime_values(prepared, &outcome);
            const std::vector<OpenAIResponsesRound> all = rounds.all();
            response.emplace(make_openai_response_object(id, created, request, runtime, all,
                                                         outcome, rounds.finish));
        } catch (const ApiException& exception) {
            const ApiError error = responses_error(exception.error());
            lifecycle->response_failure(
                make_request_failure(RequestFailurePhase::ResponseRender, error));
            write_openai_error(res, error);
            return;
        } catch (const std::exception& exception) {
            lifecycle->response_failure(make_internal_request_failure(
                RequestFailurePhase::ResponseRender, exception.what()));
            write_openai_error(res, internal_server_error(exception, "server_error"));
            return;
        }

        PendingResponseStorage storage;
        storage.input_turns      = std::move(request.prompt.input_turns);
        storage.input_items      = std::move(request.prompt.input_items);
        storage.previous_context = std::move(resolved.parent);
        // A cancelled response (status "cancelled": the client left, or the server is stopping) is answered but not
        // kept: it would take store room and a later previous_response_id could chain onto a cut answer.
        storage.enabled = request.store && outcome.finish_reason != infernix::FinishReason::Cancelled;
        try {
            commit_stored_response(openai_responses_store_, std::move(storage), id, response->body,
                                   std::move(response->output_history), prepared.preserve_thinking);
        } catch (const ApiException& exception) {
            const ApiError error = responses_error(exception.error());
            lifecycle->response_failure(
                make_request_failure(RequestFailurePhase::ResponseStore, error));
            write_openai_error(res, error);
            return;
        } catch (const std::exception& exception) {
            lifecycle->response_failure(make_internal_request_failure(
                RequestFailurePhase::ResponseStore, exception.what()));
            write_openai_error(res, internal_server_error(exception, "server_error"));
            return;
        }

        try {
            set_ngram_generation_header(res, rounds.final_outcome.ngram_archive);
            set_owned_json_content(res, response->body.dump(), prepared.lifetime);
        } catch (const std::exception& exception) {
            lifecycle->response_failure(make_internal_request_failure(
                RequestFailurePhase::ResponseRender, exception.what()));
            write_openai_error(res, internal_server_error(exception, "server_error"));
        }
        return;
    }

    // The event stream cannot be taken back once committed: wait up to kStreamAdmissionGrace for the
    // Engine's admission of the first generation, so the queue's prompt answers are a plain HTTP status
    // here as on the non-streaming branch. A request still queued then starts its stream (created,
    // then in_progress heartbeats while it waits), and a later queue answer is its response.failed,
    // as a later round's failure (after a server tool) is.
    try {
        (void)service_->admit(prepared, [&req] { return client_disconnected(req); },
                              std::chrono::steady_clock::now() + kStreamAdmissionGrace);
    } catch (const ApiException& exception) {
        const ApiError error = responses_error(exception.error());
        lifecycle->failure(make_generation_request_failure(error));
        write_openai_error(res, error);
        return;
    } catch (const std::exception& exception) {
        lifecycle->failure(
            make_internal_request_failure(RequestFailurePhase::Generation, exception.what()));
        write_openai_error(res, internal_server_error(exception, "server_error"));
        return;
    }

    try {
        auto stream                      = std::make_shared<StreamingResponse>();
        stream->prepared                 = std::move(prepared);
        stream->rounds                   = std::move(rounds);
        stream->storage.input_turns      = std::move(request.prompt.input_turns);
        stream->storage.input_items      = std::move(request.prompt.input_items);
        stream->storage.previous_context = std::move(resolved.parent);
        stream->storage.enabled          = request.store;
        // The searches read the request's tools while the encoder owns its copy.
        auto search_request              = std::make_shared<const OpenAIResponsesCreateRequest>(request);
        stream->encoder         = std::make_unique<OpenAIResponsesEventStream>(
            id, created, std::move(request), runtime_values(stream->prepared));

        prepare_sse_response(res);
        res.set_chunked_content_provider(
            "text/event-stream",
            [this, stream, id, lifecycle, search_request,
             continuation_preparer](std::size_t, httplib::DataSink& sink) -> bool {
                if (stream->started.exchange(true, std::memory_order_acq_rel)) {
                    sink.done();
                    return true;
                }
                SseTransport transport(sink, stream->cancelled);
                transport.set_heartbeat([&] { return stream->encoder->keep_alive(); });
                const auto send_failed = [&](const ApiError& error) {
                    return end_stream_with_error(sink, *lifecycle, [&] {
                        render_and_write(transport, [&] { return stream->encoder->failed(error); });
                    });
                };
                const auto server_error = [](const std::exception& exception) {
                    return internal_server_error(exception, "server_error");
                };

                try {
                    // The Response's created and in_progress events, before the generation.
                    render_and_write(transport, [&] { return stream->encoder->start(); });
                    StreamSink output;
                    output.on_reasoning = [&](const std::string& text) {
                        render_and_write(transport,
                                         [&] { return stream->encoder->reasoning_delta(text); });
                    };
                    output.on_content = [&](const std::string& text) {
                        render_and_write(transport,
                                         [&] { return stream->encoder->content_delta(text); });
                    };
                    output.on_tool_call = [&](const infernix::ToolCallDelta& delta) {
                        render_and_write(transport,
                                         [&] { return stream->encoder->tool_call_delta(delta); });
                    };
                    output.is_cancelled = [&] { return transport.poll(); };
                    const std::function<bool()> cancelled = [&] { return transport.poll(); };

                    run_openai_response(
                        *service_, *search_request, stream->rounds, stream->prepared, &output, {},
                        continuation_preparer(cancelled),
                        [&](const OpenAIResponsesRound& round) {
                            render_and_write(transport,
                                             [&] { return stream->encoder->round(round); });
                        },
                        [&](std::size_t index) { return stream->encoder->hosted_search_id(index); });
                } catch (...) {
                    return end_failed_stream_generation(*lifecycle, responses_error, server_error, send_failed);
                }

                const GenerationOutcome outcome = stream->rounds.totals();
                if (!record_stream_generation(*lifecycle, outcome, stream->cancelled)) { return false; }
                std::optional<OpenAIResponsesStreamFinish> finished;
                try {
                    const std::vector<OpenAIResponsesRound> all = stream->rounds.all();
                    finished.emplace(stream->encoder->finish(all, outcome, stream->rounds.finish));
                } catch (const ApiException& exception) {
                    const ApiError error = responses_error(exception.error());
                    lifecycle->response_failure(
                        make_request_failure(RequestFailurePhase::ResponseRender, error));
                    return send_failed(error);
                } catch (const std::exception& exception) {
                    lifecycle->response_failure(make_internal_request_failure(
                        RequestFailurePhase::ResponseRender, exception.what()));
                    return send_failed(server_error(exception));
                }

                try {
                    commit_stored_response(openai_responses_store_, std::move(stream->storage), id,
                                           finished->response.body,
                                           std::move(finished->response.output_history),
                                           stream->prepared.preserve_thinking);
                } catch (const ApiException& exception) {
                    const ApiError error = responses_error(exception.error());
                    lifecycle->response_failure(
                        make_request_failure(RequestFailurePhase::ResponseStore, error));
                    return send_failed(error);
                } catch (const std::exception& exception) {
                    lifecycle->response_failure(make_internal_request_failure(
                        RequestFailurePhase::ResponseStore, exception.what()));
                    return send_failed(server_error(exception));
                }

                std::string terminal;
                try {
                    terminal = stream->encoder->terminal(finished->response);
                    if (auto comment =
                            ngram_generation_comment(stream->rounds.final_outcome.ngram_archive);
                        !comment.empty()) {
                        terminal.insert(0, comment);
                    }
                } catch (const std::exception& exception) {
                    lifecycle->response_failure(make_internal_request_failure(
                        RequestFailurePhase::ResponseRender, exception.what()));
                    return send_failed(server_error(exception));
                }
                return write_stream_terminal(transport, sink, *lifecycle, finished->events_before_terminal,
                                             terminal);
            },
            stream_release(stream, lifecycle));
    } catch (const std::exception& exception) {
        lifecycle->failure(
            make_internal_request_failure(RequestFailurePhase::ResponseRender, exception.what()));
        write_openai_error(res, internal_server_error(exception, "server_error"));
    }
}

void HttpServer::handle_response_input_tokens(const httplib::Request& req, httplib::Response& res) {
    try {
        RequestLimits limits;
        limits.default_max_tokens = options_.default_max_tokens;
        OpenAIResponsesPromptRequest request =
            parse_openai_responses_input_tokens_request(parse_json_body(req), limits);
        request.model = resolve_openai_model(request.model, public_model_id_, options_.any_model_name);
        OpenAIResponsesResolvedPrompt resolved =
            resolve_openai_responses_prompt(request, openai_responses_store_);
        const int tokens = service_->count_prompt_tokens(
            resolved.generation, [&req] { return client_disconnected(req); });
        res.set_content(make_openai_response_input_tokens_body(tokens), "application/json");
    } catch (const ApiException& exception) {
        write_openai_error(res, responses_error(exception.error()));
    } catch (const std::exception& exception) {
        operational_log_.http_failure(
            "openai_responses_input_tokens",
            make_internal_request_failure(RequestFailurePhase::Http, exception.what()));
        write_openai_error(res, internal_server_error(exception, "server_error"));
    }
}

void HttpServer::handle_response_get(const httplib::Request& req, httplib::Response& res) {
    try {
        validate_retrieve_query(req);
    } catch (const ApiException& exception) {
        write_openai_error(res, exception.error());
        return;
    }
    const std::string id                                     = path_response_id(req);
    const std::shared_ptr<const StoredOpenAIResponse> stored = openai_responses_store_.get(id);
    if (!stored) {
        write_openai_error(res, response_not_found(id));
        return;
    }
    res.set_content(stored->response.dump(), "application/json");
}

void HttpServer::handle_response_delete(const httplib::Request& req, httplib::Response& res) {
    const std::string id = path_response_id(req);
    if (!openai_responses_store_.erase(id)) {
        write_openai_error(res, response_not_found(id));
        return;
    }
    res.set_content(Json{{"id", id}, {"object", "response.deleted"}, {"deleted", true}}.dump(),
                    "application/json");
}

void HttpServer::handle_response_input_items(const httplib::Request& req, httplib::Response& res) {
    const std::string id                                     = path_response_id(req);
    const std::shared_ptr<const StoredOpenAIResponse> stored = openai_responses_store_.get(id);
    if (!stored) {
        write_openai_error(res, response_not_found(id));
        return;
    }
    try {
        res.set_content(paginated_input_items(req, stored->input_items).dump(), "application/json");
    } catch (const ApiException& exception) { write_openai_error(res, exception.error()); }
}

void HttpServer::handle_response_cancel(const httplib::Request& req, httplib::Response& res) {
    const std::string id = path_response_id(req);
    if (!openai_responses_store_.get(id)) {
        write_openai_error(res, response_not_found(id));
        return;
    }
    ApiError error;
    error.status  = 400;
    error.type    = "invalid_request_error";
    error.code    = "background_not_supported";
    error.message = "only background responses can be cancelled; Infernix does not support "
                    "background execution";
    write_openai_error(res, error);
}

void HttpServer::handle_response_compact(const httplib::Request&, httplib::Response& res) {
    ApiError error;
    error.status  = 400;
    error.type    = "invalid_request_error";
    error.param   = "context_management";
    error.code    = "compaction_not_supported";
    error.message = "Responses compaction is not supported";
    write_openai_error(res, error);
}

} // namespace infernix::serve
