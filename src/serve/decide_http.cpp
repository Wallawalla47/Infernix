#include "serve/http_server.h"

#include "serve/decide.h"
#include "serve/http_transport.h"
#include "serve/openai_common.h"
#include "serve/request_events.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace infernix::serve {
namespace {

RequestJson error_answer(const std::string& code, const std::string& message) {
    RequestJson answer;
    answer["type"]    = "error";
    answer["code"]    = code;
    answer["message"] = message;
    return answer;
}

GenerationRequest question_request(const DecideRequest& request, const DecideQuestion& question,
                                   std::span<const DecideLabel> labels,
                                   const std::optional<DecideSchedule>& schedule) {
    GenerationRequest generation;
    generation.messages        = decide_messages(request, question, labels);
    generation.enable_thinking = false;
    // Greedy and penalty-free: penalties would push a repeated digit away from the answer.
    generation.sampling.temperature       = 0.0;
    generation.sampling.presence_penalty  = 0.0;
    generation.sampling.frequency_penalty = 0.0;
    // Only the declared tokens may end the run; the checkpoint's EOS is never among them.
    generation.ignore_eos = true;
    if (schedule) {
        generation.token_constraint = schedule->constraint;
        generation.max_tokens = static_cast<int>(schedule->constraint.steps.size());
        if (schedule->terminator >= 0) { generation.stop_token_ids = {schedule->terminator}; }
        return generation;
    }
    // A readout is taken before the one staged token is chosen.
    generation.max_tokens = 1;
    generation.readout_tokens.reserve(question.options.size());
    for (std::size_t i = 0; i < question.options.size(); ++i) {
        generation.readout_tokens.push_back(labels[i].token);
    }
    return generation;
}

ApiError internal_error(std::string message, std::string code) {
    return ApiError{.status  = 500,
                    .type    = "internal_error",
                    .message = std::move(message),
                    .code    = std::move(code)};
}

} // namespace

void HttpServer::handle_decide(const httplib::Request& req, httplib::Response& res) {
    DecideRequest request;
    std::vector<std::optional<DecideSchedule>> schedules;
    try {
        request = parse_decide_request(parse_json_body(req), decide_labels_.size());
        if (!request.model.empty()) { validate_openai_model(request.model, public_model_id_); }
        const DecideEncoder encode = [this](std::string_view text) {
            return service_->tokenize_text(text);
        };
        for (const DecideQuestion& question : request.questions) {
            schedules.push_back(decide_generates(question.kind)
                                    ? std::optional(decide_schedule(question, encode))
                                    : std::nullopt);
        }
    } catch (const ApiException& exception) {
        write_openai_error(res, exception.error());
        return;
    }

    struct QuestionRun {
        GenerationRequest generation;
        std::vector<infernix::MediaGeometry> media;
        std::optional<PreparedRequest> prepared;
        std::shared_ptr<RequestLifecycle> lifecycle;
        std::optional<RequestJson> answer;
        std::optional<ApiError> error;
        int prompt_tokens             = 0;
        int output_tokens             = 0;
        std::uint32_t computed_tokens = 0;
        std::uint32_t cached_tokens   = 0;
    };
    const auto disconnected = [&req] { return client_disconnected(req); };
    std::vector<QuestionRun> runs(request.questions.size());
    for (std::size_t q = 0; q < runs.size(); ++q) {
        runs[q].generation =
            question_request(request, request.questions[q], decide_labels_, schedules[q]);
    }

    const auto prepare = [&](QuestionRun& run) {
        const std::uint64_t req_id = ++request_seq_;
        const RequestLogMetadata metadata{
            .http_request_id        = res.get_header_value("x-request-id"),
            .model                  = public_model_id_,
            .stream                 = false,
            .output_tokens_explicit = true,
        };
        try {
            run.prepared.emplace(service_->prepare(
                run.generation, GenerationConsumerMode::Aggregate,
                infernix::GenerationObservationOptions{.phase_timings = true}, disconnected));
            run.media = run.prepared->media;
            run.lifecycle = begin_request(make_request_log_context(
                req_id, "decide", run.generation, metadata, *run.prepared));
        } catch (const ApiException& exception) {
            record_request_rejected(make_request_rejection_log_context(
                req_id, "decide", run.generation, metadata, exception.error()));
            run.error = exception.error();
        } catch (const std::exception& exception) {
            ApiError error = internal_error(exception.what(), "internal_error");
            record_request_rejected(make_request_rejection_log_context(
                req_id, "decide", run.generation, metadata, error));
            run.error = std::move(error);
        }
    };
    const auto finish = [&](QuestionRun& run, const DecideQuestion& question,
                            const std::optional<DecideSchedule>& schedule) {
        if (!run.prepared) { return; }
        try {
            GenerationOutcome outcome = service_->run(*run.prepared, nullptr, disconnected);
            run.lifecycle->done(outcome);
            run.prompt_tokens   = outcome.prompt_tokens;
            // A readout's one staged token is never returned; a generated run is the answer.
            run.output_tokens   = schedule ? outcome.completion_tokens : 0;
            run.computed_tokens = outcome.metrics.computed_prefill_tokens;
            run.cached_tokens   = outcome.metrics.prefix_cache_hit_tokens;
            if (outcome.finish_reason == infernix::FinishReason::Cancelled) {
                run.error = ApiError{.status  = 499,
                                     .type    = "cancelled",
                                     .message = "the decision was cancelled",
                                     .code    = "cancelled"};
            } else if (schedule) {
                if (outcome.constrained_draws.size() != outcome.generated_token_ids.size()) {
                    run.error =
                        internal_error("the engine returned an incomplete trace", "no_trace");
                } else {
                    run.answer = decide_generated_answer(question, *schedule,
                                                         outcome.constrained_draws, run.media);
                }
            } else if (!outcome.readout ||
                       outcome.readout->logprobs.size() != question.options.size()) {
                run.error = internal_error("the engine returned no readout", "no_readout");
            } else {
                run.answer = decide_answer(question, outcome.readout->logprobs);
            }
        } catch (const ApiException& exception) {
            run.lifecycle->failure(make_generation_request_failure(exception.error()));
            run.error = exception.error();
        } catch (const std::exception& exception) {
            run.lifecycle->failure(
                make_internal_request_failure(RequestFailurePhase::Generation, exception.what()));
            run.error = internal_error(exception.what(), "internal_error");
        }
        run.prepared.reset();
    };

    // The first question runs alone so that its prompt state is cached before its siblings
    // submit: every later question over the same state then reuses the evidence prefix. The rest
    // run in waves of the Engine's lane count, each wave fully submitted before any is awaited.
    const std::size_t wave = std::max<std::size_t>(1, service_->engine_options().max_concurrency);
    prepare(runs.front());
    finish(runs.front(), request.questions.front(), schedules.front());
    for (std::size_t begin = 1; begin < runs.size() && !disconnected(); begin += wave) {
        const std::size_t end = std::min(runs.size(), begin + wave);
        for (std::size_t q = begin; q < end; ++q) { prepare(runs[q]); }
        for (std::size_t q = begin; q < end; ++q) {
            finish(runs[q], request.questions[q], schedules[q]);
        }
    }

    RequestJson answers         = RequestJson::object();
    std::uint64_t input_tokens  = 0;
    std::uint64_t output_tokens = 0;
    std::uint64_t computed      = 0;
    std::uint64_t cached        = 0;
    const ApiError* first_error = nullptr;
    bool any_answer             = false;
    for (std::size_t q = 0; q < runs.size(); ++q) {
        QuestionRun& run      = runs[q];
        const std::string& id = request.questions[q].id;
        input_tokens += static_cast<std::uint64_t>(std::max(0, run.prompt_tokens));
        computed += run.computed_tokens;
        cached += run.cached_tokens;
        if (run.answer) {
            answers[id] = std::move(*run.answer);
            any_answer  = true;
        } else if (run.error) {
            answers[id] = error_answer(run.error->code.empty() ? run.error->type : run.error->code,
                                       run.error->message);
            if (first_error == nullptr) { first_error = &*run.error; }
        } else {
            answers[id] = error_answer("cancelled", "the client disconnected first");
        }
    }
    if (!any_answer && first_error != nullptr) {
        // Nothing was answered: the request failed as a whole, with the first question's cause.
        write_openai_error(res, *first_error);
        return;
    }
    RequestJson body;
    body["model"]                            = public_model_id_;
    body["answers"]                          = std::move(answers);
    body["usage"]["input_tokens"]            = input_tokens;
    body["usage"]["output_tokens"]           = 0;
    body["usage"]["cached_input_tokens"]     = cached;
    body["usage"]["computed_prefill_tokens"] = computed;
    res.set_content(body.dump(-1, ' ', false, RequestJson::error_handler_t::replace),
                    "application/json");
}

} // namespace infernix::serve
