#include "serve/http_server.h"

#include "serve/anthropic_messages.h"
#include "serve/http_transport.h"
#include "serve/server_tools.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace infernix::serve {
namespace {

// One Messages response. A request that declares tool search tools may run several generations:
// each that ends in searches only is followed by the searches' results and another generation,
// until the model answers, calls a client tool, or a limit ends the response.
struct AnthropicResponseState {
    explicit AnthropicResponseState(AnthropicMessagesRequest parsed)
        : request(std::move(parsed)), executor(request), continuation(request.generation) {
        rounds.reserve(kMaximumServerToolRounds);
        round_results.reserve(kMaximumServerToolRounds);
    }

    AnthropicMessagesRequest request;
    AnthropicToolSearchExecutor executor;
    GenerationRequest continuation; // the request plus every completed round
    std::vector<ServerToolRound> rounds;
    std::vector<std::vector<std::optional<RequestJson>>> round_results;

    // The final generation and how the response ends.
    GenerationOutcome final_outcome;
    std::vector<ToolCall> final_calls;
    std::vector<std::optional<RequestJson>> final_results;
    std::optional<std::string_view> stop_reason;

    [[nodiscard]] AnthropicResponseSegment round_segment(std::size_t index) const {
        return AnthropicResponseSegment{.outcome        = &rounds[index].outcome,
                                        .calls          = rounds[index].calls,
                                        .search_results = round_results[index]};
    }

    [[nodiscard]] AnthropicResponseSegment final_segment() const {
        return AnthropicResponseSegment{
            .outcome = &final_outcome, .calls = final_calls, .search_results = final_results};
    }

    [[nodiscard]] std::vector<AnthropicResponseSegment> segments() const {
        std::vector<AnthropicResponseSegment> result;
        for (std::size_t index = 0; index < rounds.size(); ++index) {
            result.push_back(round_segment(index));
        }
        result.push_back(final_segment());
        return result;
    }

    [[nodiscard]] GenerationOutcome totals() const {
        return combine_round_outcomes(rounds, final_outcome);
    }
};

using ContinuationPreparer = std::function<PreparedRequest(const GenerationRequest&)>;
using SegmentWriter        = std::function<void(const AnthropicResponseSegment&)>;
// A generation's calls with wire identifiers: a stream's keep those it streamed them with.
using CallIdentifier = std::function<std::vector<ToolCall>(const GenerationOutcome&)>;

// Runs `prepared` and every continuation its searches lead to, leaving the final generation in
// `state`. `prepared` ends holding the last generation's request (and its ingress reservation).
void run_anthropic_response(GenerationBackend& service, AnthropicResponseState& state,
                            PreparedRequest& prepared, const StreamSink* sink,
                            const std::function<bool()>& is_cancelled,
                            const ContinuationPreparer& prepare_next,
                            const SegmentWriter& on_segment, const CallIdentifier& identify) {
    for (;;) {
        GenerationOutcome outcome   = service.run(prepared, sink, is_cancelled);
        std::vector<ToolCall> calls = identify(outcome);
        std::vector<std::optional<RequestJson>> results(calls.size());
        std::vector<ChatTurn> result_turns;
        bool client_calls = false;
        if (outcome.finish_reason != infernix::FinishReason::Cancelled) {
            for (std::size_t index = 0; index < calls.size(); ++index) {
                if (state.executor.search_tool(calls[index].name) == nullptr) {
                    client_calls = true;
                    continue;
                }
                AnthropicToolSearchResult executed = state.executor.execute(calls[index]);
                results[index] = std::move(executed.content);
                result_turns.push_back(std::move(executed.turn));
            }
        }
        // A generation continues only when its turn ended in searches alone; a search in a turn
        // that also calls a client tool, or that the output limit cut, still runs, so the history
        // the client sends back carries its result.
        const bool turn_ended = outcome.finish_reason == infernix::FinishReason::StopToken ||
                                outcome.finish_reason == infernix::FinishReason::None;
        bool continues = !result_turns.empty() && !client_calls && turn_ended;
        std::optional<std::string_view> stop_reason;
        if (continues && state.rounds.size() + 1U >= kMaximumServerToolRounds) {
            continues   = false;
            stop_reason = "pause_turn";
        }
        ServerToolRound round{.outcome = std::move(outcome),
                              .calls   = std::move(calls),
                              .results = std::move(result_turns)};
        std::optional<PreparedRequest> next;
        if (continues) {
            GenerationRequest extended = state.continuation;
            append_server_round(extended, round, state.executor.names());
            if (extended.max_tokens <= 0) {
                continues   = false;
                stop_reason = "max_tokens";
            } else {
                try {
                    next.emplace(prepare_next(extended));
                    state.continuation = std::move(extended);
                } catch (const ApiException& error) {
                    if (error.error().code != "context_length_exceeded") { throw; }
                    continues   = false;
                    stop_reason = "model_context_window_exceeded";
                }
            }
        }
        if (!continues) {
            state.final_outcome = std::move(round.outcome);
            state.final_calls   = std::move(round.calls);
            state.final_results = std::move(results);
            state.stop_reason   = stop_reason;
            return;
        }
        state.rounds.push_back(std::move(round));
        state.round_results.push_back(std::move(results));
        if (on_segment) { on_segment(state.round_segment(state.rounds.size() - 1U)); }
        prepared = std::move(*next);
    }
}

} // namespace

void HttpServer::handle_count_tokens(const httplib::Request& req, httplib::Response& res) {
    const std::string request_id = res.get_header_value("request-id");
    try {
        const AnthropicCountTokensRequest request = parse_anthropic_count_tokens_request(
            parse_json_body(req), {.claude_code_prompt = options_.claude_code_prompt});
        const int input_tokens = service_->count_prompt_tokens(
            request.generation, [&req] { return client_disconnected(req); });
        res.set_content(make_anthropic_count_tokens_response(input_tokens), "application/json");
    } catch (const ApiException& exception) {
        write_anthropic_error(res, exception.error(), request_id);
    } catch (const std::exception& exception) {
        operational_log_.http_failure(
            "anthropic_count_tokens",
            make_internal_request_failure(RequestFailurePhase::Http, exception.what()), request_id);
        ApiError error;
        error.status  = 500;
        error.message = exception.what();
        write_anthropic_error(res, error, request_id);
    }
}

void HttpServer::handle_messages(const httplib::Request& req, httplib::Response& res) {
    const std::string request_id = res.get_header_value("request-id");

    std::shared_ptr<AnthropicResponseState> state;
    try {
        RequestLimits limits;
        limits.default_max_tokens = options_.default_max_tokens;
        limits.claude_code_prompt = options_.claude_code_prompt;
        const auto body           = parse_json_body(req);
        AnthropicMessagesRequest parsed = parse_anthropic_messages_request(body, limits);
        parsed.generation.ngram_session = resolve_ngram_session(req, body, options_);
        state = std::make_shared<AnthropicResponseState>(std::move(parsed));
    } catch (const ApiException& exception) {
        write_anthropic_error(res, exception.error(), request_id);
        return;
    } catch (const std::exception& exception) {
        operational_log_.http_failure(
            "anthropic_messages",
            make_internal_request_failure(RequestFailurePhase::Http, exception.what()), request_id);
        ApiError error;
        error.status  = 500;
        error.message = exception.what();
        write_anthropic_error(res, error, request_id);
        return;
    }
    const AnthropicMessagesRequest& request = state->request;

    const std::uint64_t req_id = ++request_seq_;
    RequestLogMetadata metadata{.http_request_id        = request_id,
                                .model                  = request.model,
                                .stream                 = request.stream,
                                .output_tokens_explicit = request.output_tokens_explicit,
                                .label                  = request_label(req)};
    const GenerationConsumerMode consumer_mode =
        request.stream ? GenerationConsumerMode::Streaming : GenerationConsumerMode::Aggregate;
    PreparedRequest prepared;
    try {
        prepared = service_->prepare(request.generation, consumer_mode,
                                     {.phase_timings = true,
                                      .scheduling    = scheduling_observer(req_id, metadata.http_request_id),
                                      .first_token   = first_token_observer()},
                                     [&req] { return client_disconnected(req); });
    } catch (const ApiException& exception) {
        const ApiError error = normalize_anthropic_error(exception.error());
        record_request_rejected(make_request_rejection_log_context(
            req_id, "anthropic_messages", request.generation, metadata, error));
        write_anthropic_error(res, error, request_id);
        return;
    } catch (const std::exception& exception) {
        ApiError error;
        error.status  = 500;
        error.type    = "internal_error";
        error.message = exception.what();
        record_request_rejected(make_request_rejection_log_context(
            req_id, "anthropic_messages", request.generation, metadata, error));
        write_anthropic_error(res, error, request_id);
        return;
    }

    const AnthropicResponseIdentity identity =
        make_anthropic_response_identity(request_id, request.model);
    metadata.response_id   = identity.message_id;
    const int input_tokens = prepared.prompt_tokens;

    auto lifecycle = begin_request(make_request_log_context(
        req_id, "anthropic_messages", request.generation, metadata, prepared));

    // A continuation after a search keeps the request's ingress reservation and log identity; its
    // first token is not the response's.
    const std::string http_request_id = metadata.http_request_id;
    const auto continuation_preparer  = [this, consumer_mode, req_id,
                                         http_request_id](const std::function<bool()>& cancelled) {
        return ContinuationPreparer([this, consumer_mode, req_id, http_request_id,
                                     cancelled](const GenerationRequest& next) {
            return service_->prepare_continuation(
                next, consumer_mode,
                {.phase_timings = true, .scheduling = scheduling_observer(req_id, http_request_id)},
                cancelled);
        });
    };

    if (!request.stream) {
        const std::function<bool()> cancelled = [&req] { return client_disconnected(req); };
        try {
            run_anthropic_response(*service_, *state, prepared, nullptr, cancelled,
                                   continuation_preparer(cancelled), {},
                                   [&](const GenerationOutcome& outcome) {
                                       return identify_anthropic_tool_calls(outcome, request.tool_searches);
                                   });
        } catch (const ApiException& exception) {
            const ApiError error = normalize_anthropic_error(exception.error());
            lifecycle->failure(make_generation_request_failure(error));
            write_anthropic_error(res, error, request_id);
            return;
        } catch (const std::exception& exception) {
            lifecycle->failure(
                make_internal_request_failure(RequestFailurePhase::Generation, exception.what()));
            ApiError error;
            error.status  = 500;
            error.message = exception.what();
            write_anthropic_error(res, error, request_id);
            return;
        }
        const GenerationOutcome totals = state->totals();
        lifecycle->done(totals);
        try {
            set_ngram_generation_header(res, state->final_outcome.ngram_archive);
            const std::vector<AnthropicResponseSegment> segments = state->segments();
            set_owned_json_content(res,
                                   make_anthropic_messages_response(identity, segments, totals,
                                                                    state->stop_reason,
                                                                    request.hide_thinking),
                                   prepared.lifetime);
        } catch (const ApiException& exception) {
            const ApiError error = normalize_anthropic_error(exception.error());
            lifecycle->response_failure(
                make_request_failure(RequestFailurePhase::ResponseRender, error));
            write_anthropic_error(res, error, request_id);
        } catch (const std::exception& exception) {
            lifecycle->response_failure(make_internal_request_failure(
                RequestFailurePhase::ResponseRender, exception.what()));
            ApiError error;
            error.status  = 500;
            error.message = exception.what();
            write_anthropic_error(res, error, request_id);
        }
        return;
    }

    // The event stream cannot be taken back once committed: wait for the Engine's admission of the
    // first generation, so the queue's own answers are a plain HTTP status here as on the
    // non-streaming branch (a later round's failure, after a server tool, belongs in the stream).
    try {
        service_->admit(prepared, [&req] { return client_disconnected(req); });
    } catch (const ApiException& exception) {
        const ApiError error = normalize_anthropic_error(exception.error());
        lifecycle->failure(make_generation_request_failure(error));
        write_anthropic_error(res, error, request_id);
        return;
    } catch (const std::exception& exception) {
        lifecycle->failure(
            make_internal_request_failure(RequestFailurePhase::Generation, exception.what()));
        ApiError error;
        error.status  = 500;
        error.message = exception.what();
        write_anthropic_error(res, error, request_id);
        return;
    }

    try {
        auto stream  = std::make_shared<HttpGenerationStream>(std::move(prepared));
        auto encoder = std::make_shared<AnthropicMessagesStream>(identity, input_tokens,
                                                                 request.hide_thinking, request.tool_searches);

        prepare_sse_response(res);
        res.set_chunked_content_provider(
            "text/event-stream",
            [this, stream, encoder, lifecycle, state,
             continuation_preparer](std::size_t, httplib::DataSink& sink) -> bool {
                if (stream->started.exchange(true, std::memory_order_acq_rel)) {
                    sink.done();
                    return true;
                }
                SseTransport transport(sink, stream->cancelled);
                const auto send_error = [&](const ApiError& error) {
                    try {
                        if (!encoder->started()) {
                            render_and_write(transport, [&] { return encoder->start(); });
                        }
                        render_and_write(transport, [&] { return encoder->error(error); });
                        sink.done();
                        return true;
                    } catch (const ClientDisconnected&) {
                        lifecycle->response_failure(
                            make_client_disconnected_failure(RequestFailurePhase::Transport));
                        return false;
                    } catch (const ResponseRenderFailure& exception) {
                        lifecycle->response_failure(make_internal_request_failure(
                            RequestFailurePhase::ResponseRender, exception.what()));
                        return false;
                    }
                };

                try {
                    StreamSink output;
                    // Only the first generation starts the message; later ones continue it.
                    output.on_start = [&](const infernix::GenerationStart& start) {
                        if (!encoder->started()) {
                            render_and_write(transport, [&] { return encoder->start(start); });
                        }
                    };
                    output.on_reasoning = [&](const std::string& text) {
                        render_and_write(transport, [&] { return encoder->reasoning_delta(text); });
                    };
                    output.on_content = [&](const std::string& text) {
                        render_and_write(transport, [&] { return encoder->content_delta(text); });
                    };
                    output.on_tool_call = [&](const infernix::ToolCallDelta& delta) {
                        render_and_write(transport, [&] { return encoder->tool_call_delta(delta); });
                    };
                    output.is_cancelled = [&] { return transport.poll(); };
                    const std::function<bool()> cancelled = [&] { return transport.poll(); };

                    run_anthropic_response(
                        *service_, *state, stream->prepared, &output, {},
                        continuation_preparer(cancelled),
                        [&](const AnthropicResponseSegment& segment) {
                            render_and_write(transport, [&] { return encoder->segment(segment); });
                        },
                        [&](const GenerationOutcome& outcome) { return encoder->identify_calls(outcome); });
                } catch (const ClientDisconnected&) {
                    lifecycle->failure(
                        make_client_disconnected_failure(RequestFailurePhase::Transport));
                    return false;
                } catch (const ResponseRenderFailure& exception) {
                    lifecycle->failure(make_internal_request_failure(
                        RequestFailurePhase::ResponseRender, exception.what()));
                    ApiError error;
                    error.status  = 500;
                    error.message = exception.what();
                    return send_error(error);
                } catch (const ApiException& exception) {
                    const ApiError error = normalize_anthropic_error(exception.error());
                    lifecycle->failure(make_generation_request_failure(error));
                    return send_error(error);
                } catch (const std::exception& exception) {
                    lifecycle->failure(make_internal_request_failure(
                        RequestFailurePhase::Generation, exception.what()));
                    ApiError error;
                    error.status  = 500;
                    error.message = exception.what();
                    return send_error(error);
                }

                const GenerationOutcome totals = state->totals();
                lifecycle->done(totals);
                if (state->final_outcome.finish_reason == infernix::FinishReason::Cancelled ||
                    stream->cancelled.load(std::memory_order_acquire)) {
                    lifecycle->response_failure(
                        make_client_disconnected_failure(RequestFailurePhase::Transport));
                    return false;
                }
                std::vector<std::string> terminal;
                try {
                    terminal = encoder->finish(state->final_segment(), totals, state->stop_reason);
                    if (auto comment = ngram_generation_comment(state->final_outcome.ngram_archive);
                        !comment.empty()) {
                        terminal.insert(terminal.begin(), std::move(comment));
                    }
                } catch (const std::exception& exception) {
                    lifecycle->response_failure(make_internal_request_failure(
                        RequestFailurePhase::ResponseRender, exception.what()));
                    ApiError error;
                    error.status  = 500;
                    error.message = exception.what();
                    return send_error(error);
                }
                try {
                    transport.write(terminal);
                    sink.done();
                    return true;
                } catch (const ClientDisconnected&) {
                    lifecycle->response_failure(
                        make_client_disconnected_failure(RequestFailurePhase::Transport));
                    return false;
                }
            },
            [stream, lifecycle](bool successful) {
                stream->cancelled.store(true, std::memory_order_release);
                if (!successful || !stream->started.load(std::memory_order_acquire)) {
                    lifecycle->failure(
                        make_client_disconnected_failure(RequestFailurePhase::Transport));
                }
            });
    } catch (const std::exception& exception) {
        lifecycle->failure(
            make_internal_request_failure(RequestFailurePhase::ResponseRender, exception.what()));
        ApiError error;
        error.status  = 500;
        error.message = exception.what();
        write_anthropic_error(res, error, request_id);
    }
}

} // namespace infernix::serve
