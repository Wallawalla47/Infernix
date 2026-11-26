#include "serve/http_server.h"

#include "serve/http_stream_driver.h"
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
        require_generation();
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
    // Assistant messages sent back without their reasoning: counted, and with --restore-reasoning given
    // the reasoning this server generated for them. The reply's reasoning is remembered under its tool-call
    // ids, or as an answer to this request's conversation (a continued final message is not a new answer).
    ConversationDigest conversation;
    const ReasoningRestore reasoning = reasoning_store_.restore(request.generation, conversation);
    metrics_.reasoning(reasoning);
    const bool new_answer = request.generation.continuation == infernix::PromptContinuationMode::NewAssistantTurn;
    const auto remember_reasoning = [this, conversation, new_answer](const GenerationOutcome& outcome,
                                                                     const std::vector<std::string>& ids) {
        if (!outcome.tool_calls.empty()) {
            reasoning_store_.remember_tool_calls(ids, outcome.reasoning);
        } else if (new_answer) {
            reasoning_store_.remember_answer(conversation, outcome.text, outcome.reasoning);
        }
    };

    const std::uint64_t req_id = ++request_seq_;
    RequestLogMetadata metadata{.http_request_id        = res.get_header_value("x-request-id"),
                                .model                  = request.model,
                                .stream                 = request.stream,
                                .output_tokens_explicit = request.output_tokens_explicit,
                                .label                  = request_label(req),
                                .reasoning              = reasoning,
                                .client_request_id      = client_header_id(req, "x-request-id"),
                                .session_id             = client_header_id(req, "x-session-id")};
    PreparedRequest prepared;
    try {
        infernix::GenerationObservationOptions observation{
            .phase_timings   = true,
            .live_timings    = request.stream && request.timings_per_token,
            .prompt_progress = request.stream && request.return_progress,
            .scheduling      = scheduling_observer(req_id, metadata.http_request_id),
            .first_token     = first_token_observer(),
            .progress        = progress_observer(req_id),
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
        const ApiError error = internal_server_error(exception);
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
            write_openai_error(res, internal_server_error(exception));
            return;
        }
        lifecycle->done(outcome);
        try {
            set_ngram_generation_header(res, outcome.ngram_archive);
            std::vector<std::string> call_ids;
            set_owned_json_content(res, make_chat_completion_response(identity, outcome, &call_ids),
                                   prepared.lifetime);
            // A cancelled answer is cut: its reasoning is not kept for the next turn to resend.
            if (outcome.finish_reason != infernix::FinishReason::Cancelled) { remember_reasoning(outcome, call_ids); }
        } catch (const std::exception& exception) {
            lifecycle->response_failure(make_internal_request_failure(
                RequestFailurePhase::ResponseRender, exception.what()));
            write_openai_error(res, internal_server_error(exception));
        }
        return;
    }

    // The SSE response cannot be taken back once committed: wait for the Engine's admission, so the
    // queue's answers (a queue timeout, an orderly stop) are a plain HTTP status here as on the
    // non-streaming branch. Unlike Messages and Responses, Chat Completions has no error event clients
    // agree on: a queue timeout as one `data:` error frame after a 200 is recorded by some clients
    // (LiteLLM 1.98, raw HTTP) as a successful empty reply and retried at once (NInfer #420), so this
    // route does not start its stream early (kStreamAdmissionGrace).
    try {
        (void)service_->admit(prepared, [&req] { return client_disconnected(req); },
                              std::chrono::steady_clock::time_point::max());
    } catch (const ApiException& exception) {
        lifecycle->failure(make_generation_request_failure(exception.error()));
        write_openai_error(res, exception.error());
        return;
    } catch (const std::exception& exception) {
        lifecycle->failure(
            make_internal_request_failure(RequestFailurePhase::Generation, exception.what()));
        write_openai_error(res, internal_server_error(exception));
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
                transport.set_heartbeat([&] { return encoder->keep_alive(); });
                const auto send_error = [&](const ApiError& error) {
                    return end_stream_with_error(sink, *lifecycle, [&] {
                        render_and_write(transport, [&] { return sse_error_event(error); });
                    });
                };
                // The assistant-role chunk opens the stream with its first event (prompt progress, the
                // first output delta or the finish), not at admission, so a client's time to its first
                // chunk is the prefill's (as OpenAI's). Until then a heartbeat is an SSE comment.
                bool opened     = false;
                const auto open = [&] {
                    if (opened) { return; }
                    opened = true;
                    render_and_write(transport, [&] { return encoder->start(); });
                };

                GenerationOutcome outcome;
                try {
                    StreamSink output;
                    output.on_start = [&](const infernix::GenerationStart& start) {
                        encoder->note_start(start);
                        if (return_progress) {
                            open();
                            render_and_write(transport,
                                             [&] { return encoder->initial_prompt_progress(); });
                        }
                    };
                    if (return_progress) {
                        output.on_progress = [&](const infernix::PromptProgress& progress) {
                            open();
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
                        open();
                        render_and_write(transport, [&] { return encoder->content_delta(text); });
                    };
                    output.on_reasoning = [&](const std::string& text) {
                        open();
                        render_and_write(transport, [&] { return encoder->reasoning_delta(text); });
                    };
                    output.on_tool_call = [&](const infernix::ToolCallDelta& delta) {
                        open();
                        render_and_write(transport, [&] { return encoder->tool_call_delta(delta); });
                    };
                    output.is_cancelled = [&] { return transport.poll(); };

                    outcome = service_->run(stream->prepared, &output);
                } catch (...) {
                    return end_failed_stream_generation(
                        *lifecycle, [](const ApiError& error) -> const ApiError& { return error; },
                        [](const std::exception& exception) { return internal_server_error(exception); },
                        send_error);
                }

                if (!record_stream_generation(*lifecycle, outcome, stream->cancelled)) { return false; }
                std::vector<std::string> terminal;
                try {
                    // An answer with no streamed delta (an empty one, or tool calls sent whole) opens here.
                    std::string opening;
                    if (!opened) {
                        opened  = true;
                        opening = encoder->start();
                    }
                    terminal = encoder->finish(outcome);
                    if (!opening.empty()) { terminal.insert(terminal.begin(), std::move(opening)); }
                    remember_reasoning(outcome, encoder->tool_call_ids());
                    if (auto comment = ngram_generation_comment(outcome.ngram_archive);
                        !comment.empty()) {
                        terminal.insert(terminal.begin(), std::move(comment));
                    }
                } catch (const std::exception& exception) {
                    lifecycle->response_failure(make_internal_request_failure(
                        RequestFailurePhase::ResponseRender, exception.what()));
                    return send_error(internal_server_error(exception));
                }
                return write_stream_terminal(transport, sink, *lifecycle, terminal);
            },
            stream_release(stream, lifecycle));
    } catch (const std::exception& exception) {
        lifecycle->failure(
            make_internal_request_failure(RequestFailurePhase::ResponseRender, exception.what()));
        write_openai_error(res, internal_server_error(exception));
    }
}

} // namespace infernix::serve
