#include "serve/http_server.h"

#include "serve/http_transport.h"
#include "serve/openai_chat.h"
#include "serve/openai_common.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace infernix::serve {
namespace {

std::string sse_error_event(const ApiError& error) {
    return "data: " + make_error_body(error) + "\n\n";
}

} // namespace

void HttpServer::handle_chat_completions(const httplib::Request& req, httplib::Response& res) {
    OpenAIChatRequest request;
    try {
        RequestLimits limits;
        limits.default_max_tokens        = options_.default_max_tokens;
        const auto body                  = parse_json_body(req);
        request                          = parse_chat_completion_request(body, limits);
        request.generation.ngram_session = resolve_ngram_session(req, body, options_);
        request.model = resolve_openai_model(request.model, public_model_id_, options_.any_model_name);
    } catch (const ApiException& exception) {
        write_openai_error(res, exception.error());
        return;
    }
    // Tool-call turns sent back without their reasoning: counted, and with --restore-tool-reasoning
    // given the reasoning this server generated for them.
    const ReasoningRestore reasoning =
        options_.restore_tool_reasoning
            ? reasoning_store_.restore(request.generation.messages)
            : ReasoningRestore{.missing = count_missing_reasoning(request.generation.messages)};
    metrics_.reasoning(reasoning.missing, reasoning.restored);
    const auto remember_reasoning = [this](const GenerationOutcome& outcome, const std::vector<std::string>& ids) {
        if (options_.restore_tool_reasoning && !outcome.tool_calls.empty()) {
            reasoning_store_.remember(ids, outcome.reasoning);
        }
    };

    const std::uint64_t req_id = ++request_seq_;
    RequestLogMetadata metadata{.http_request_id        = res.get_header_value("x-request-id"),
                                .model                  = request.model,
                                .stream                 = request.stream,
                                .output_tokens_explicit = request.output_tokens_explicit,
                                .label                  = request_label(req)};
    PreparedRequest prepared;
    try {
        infernix::GenerationObservationOptions observation{
            .phase_timings   = true,
            .live_timings    = request.stream && request.timings_per_token,
            .prompt_progress = request.stream && request.return_progress,
            .scheduling      = scheduling_observer(req_id, metadata.http_request_id),
            .first_token     = first_token_observer(),
        };
        prepared = service_->prepare(
            request.generation,
            request.stream ? GenerationConsumerMode::Streaming : GenerationConsumerMode::Aggregate,
            std::move(observation), [&req] { return client_disconnected(req); });
    } catch (const ApiException& exception) {
        record_request_rejected(make_request_rejection_log_context(
            req_id, "openai_chat_completions", request.generation, metadata, exception.error()));
        write_openai_error(res, exception.error());
        return;
    } catch (const std::exception& exception) {
        ApiError error;
        error.status  = 500;
        error.type    = "internal_error";
        error.message = exception.what();
        record_request_rejected(make_request_rejection_log_context(
            req_id, "openai_chat_completions", request.generation, metadata, error));
        write_openai_error(res, error);
        return;
    }

    const OpenAIChatResponseIdentity identity = make_openai_chat_response_identity(request.model);
    metadata.response_id                      = identity.id;
    auto lifecycle                            = begin_request(make_request_log_context(
        req_id, "openai_chat_completions", request.generation, metadata, prepared));

    if (!request.stream) {
        GenerationOutcome outcome;
        try {
            outcome = service_->run(prepared, nullptr, [&req] { return client_disconnected(req); });
        } catch (const ApiException& exception) {
            lifecycle->failure(make_generation_request_failure(exception.error()));
            write_openai_error(res, exception.error());
            return;
        } catch (const std::exception& exception) {
            const RequestFailure failure =
                make_internal_request_failure(RequestFailurePhase::Generation, exception.what());
            lifecycle->failure(failure);
            ApiError error;
            error.status  = 500;
            error.type    = "internal_error";
            error.message = exception.what();
            write_openai_error(res, error);
            return;
        }
        lifecycle->done(outcome);
        try {
            set_ngram_generation_header(res, outcome.ngram_archive);
            std::vector<std::string> call_ids;
            set_owned_json_content(res, make_chat_completion_response(identity, outcome, &call_ids),
                                   prepared.lifetime);
            remember_reasoning(outcome, call_ids);
        } catch (const std::exception& exception) {
            lifecycle->response_failure(make_internal_request_failure(
                RequestFailurePhase::ResponseRender, exception.what()));
            ApiError error;
            error.status  = 500;
            error.type    = "internal_error";
            error.message = exception.what();
            write_openai_error(res, error);
        }
        return;
    }

    // The SSE response cannot be taken back once committed: wait for the Engine's admission first,
    // so the queue's own answers (a queue timeout, an orderly stop) are a plain HTTP status here as
    // on the non-streaming branch; after it only mid-generation failures remain, which belong in
    // the stream.
    try {
        service_->admit(prepared, [&req] { return client_disconnected(req); });
    } catch (const ApiException& exception) {
        lifecycle->failure(make_generation_request_failure(exception.error()));
        write_openai_error(res, exception.error());
        return;
    } catch (const std::exception& exception) {
        lifecycle->failure(
            make_internal_request_failure(RequestFailurePhase::Generation, exception.what()));
        ApiError error;
        error.status  = 500;
        error.type    = "internal_error";
        error.message = exception.what();
        write_openai_error(res, error);
        return;
    }

    try {
        const bool return_progress   = request.return_progress;
        const bool timings_per_token = request.timings_per_token;
        auto stream                  = std::make_shared<HttpGenerationStream>(std::move(prepared));
        auto encoder = std::make_shared<OpenAIChatStream>(identity, request.include_usage,
                                                          timings_per_token, return_progress,
                                                          options_.usage_chunk_choice);

        prepare_sse_response(res);
        res.set_chunked_content_provider(
            "text/event-stream",
            [this, stream, encoder, lifecycle, return_progress, timings_per_token,
             remember_reasoning](std::size_t, httplib::DataSink& sink) -> bool {
                if (stream->started.exchange(true, std::memory_order_acq_rel)) {
                    sink.done();
                    return true;
                }
                SseTransport transport(sink, stream->cancelled);
                const auto send_error = [&](const ApiError& error) {
                    try {
                        render_and_write(transport, [&] { return sse_error_event(error); });
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
                    render_and_write(transport, [&] { return encoder->start(); });
                } catch (const ClientDisconnected&) {
                    lifecycle->failure(
                        make_client_disconnected_failure(RequestFailurePhase::Transport));
                    return false;
                } catch (const ResponseRenderFailure& exception) {
                    lifecycle->failure(make_internal_request_failure(
                        RequestFailurePhase::ResponseRender, exception.what()));
                    ApiError error;
                    error.status  = 500;
                    error.type    = "internal_error";
                    error.message = exception.what();
                    return send_error(error);
                }

                GenerationOutcome outcome;
                try {
                    StreamSink output;
                    output.on_start = [&](const infernix::GenerationStart& start) {
                        encoder->note_start(start);
                        if (return_progress) {
                            render_and_write(transport,
                                             [&] { return encoder->initial_prompt_progress(); });
                        }
                    };
                    if (return_progress) {
                        output.on_progress = [&](const infernix::PromptProgress& progress) {
                            render_and_write(transport,
                                             [&] { return encoder->prompt_progress(progress); });
                        };
                    }
                    if (timings_per_token) {
                        output.on_timing = [&](const infernix::GenerationTimingObservation& timing) {
                            encoder->note_timing(timing);
                        };
                    }
                    output.on_content = [&](const std::string& text) {
                        render_and_write(transport, [&] { return encoder->content_delta(text); });
                    };
                    output.on_reasoning = [&](const std::string& text) {
                        render_and_write(transport, [&] { return encoder->reasoning_delta(text); });
                    };
                    output.on_tool_call = [&](const infernix::ToolCallDelta& delta) {
                        render_and_write(transport, [&] { return encoder->tool_call_delta(delta); });
                    };
                    output.is_cancelled = [&] { return transport.poll(); };

                    outcome = service_->run(stream->prepared, &output);
                } catch (const ClientDisconnected&) {
                    lifecycle->failure(
                        make_client_disconnected_failure(RequestFailurePhase::Transport));
                    return false;
                } catch (const ResponseRenderFailure& exception) {
                    lifecycle->failure(make_internal_request_failure(
                        RequestFailurePhase::ResponseRender, exception.what()));
                    ApiError error;
                    error.status  = 500;
                    error.type    = "internal_error";
                    error.message = exception.what();
                    return send_error(error);
                } catch (const ApiException& exception) {
                    lifecycle->failure(make_generation_request_failure(exception.error()));
                    return send_error(exception.error());
                } catch (const std::exception& exception) {
                    lifecycle->failure(make_internal_request_failure(
                        RequestFailurePhase::Generation, exception.what()));
                    ApiError error;
                    error.status  = 500;
                    error.type    = "internal_error";
                    error.message = exception.what();
                    return send_error(error);
                }

                lifecycle->done(outcome);
                if (outcome.finish_reason == infernix::FinishReason::Cancelled ||
                    stream->cancelled.load(std::memory_order_acquire)) {
                    lifecycle->response_failure(
                        make_client_disconnected_failure(RequestFailurePhase::Transport));
                    return false;
                }
                std::vector<std::string> terminal;
                try {
                    terminal = encoder->finish(outcome);
                    remember_reasoning(outcome, encoder->tool_call_ids());
                    if (auto comment = ngram_generation_comment(outcome.ngram_archive);
                        !comment.empty()) {
                        terminal.insert(terminal.begin(), std::move(comment));
                    }
                } catch (const std::exception& exception) {
                    lifecycle->response_failure(make_internal_request_failure(
                        RequestFailurePhase::ResponseRender, exception.what()));
                    ApiError error;
                    error.status  = 500;
                    error.type    = "internal_error";
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
        error.type    = "internal_error";
        error.message = exception.what();
        write_openai_error(res, error);
    }
}

} // namespace infernix::serve
