#pragma once

// The steps every streamed generation route's chunked content provider shares (Chat Completions, Anthropic Messages,
// Responses): how a failure ends the stream, how the finished generation is recorded before its terminal events, the
// terminal write, and the provider's release. What a stream writes stays its protocol's: its error event, how it maps
// errors, its terminal events. `Lifecycle` is the server's request lifecycle (done, failure, response_failure).

#include "serve/generation_service.h"
#include "serve/http_transport.h"
#include "serve/request.h"
#include "serve/request_events.h"

#include <httplib.h>

#include <atomic>
#include <exception>
#include <memory>
#include <utility>

namespace infernix::serve {

// Ends the stream with the protocol's error event, which `write_error` renders and writes. True when it was written;
// false when the client left first or the event did not render (a response failure: the request already has its
// record).
template <class Lifecycle, class WriteError>
bool end_stream_with_error(httplib::DataSink& sink, Lifecycle& lifecycle, WriteError&& write_error) {
    try {
        std::forward<WriteError>(write_error)();
        sink.done();
        return true;
    } catch (const ClientDisconnected&) {
        lifecycle.response_failure(make_client_disconnected_failure(RequestFailurePhase::Transport));
        return false;
    } catch (const ResponseRenderFailure& exception) {
        lifecycle.response_failure(
            make_internal_request_failure(RequestFailurePhase::ResponseRender, exception.what()));
        return false;
    }
}

// Ends a stream whose generation phase threw, from the catch (...) handler around it (its events' writes included):
// records the request's failure and sends the error event for it with `send_error` (returning its result). A client
// that left gets nothing; an event that did not render and an unexpected exception are internal errors (`internal`
// maps the exception to the protocol's error); an API error is the protocol's own (`normalize`). Anything else is
// rethrown.
template <class Lifecycle, class Normalize, class Internal, class SendError>
bool end_failed_stream_generation(Lifecycle& lifecycle, Normalize&& normalize, Internal&& internal,
                                  SendError&& send_error) {
    try {
        throw;
    } catch (const ClientDisconnected&) {
        lifecycle.failure(make_client_disconnected_failure(RequestFailurePhase::Transport));
        return false;
    } catch (const ResponseRenderFailure& exception) {
        lifecycle.failure(make_internal_request_failure(RequestFailurePhase::ResponseRender, exception.what()));
        return send_error(internal(exception));
    } catch (const ApiException& exception) {
        decltype(auto) error = normalize(exception.error());
        lifecycle.failure(make_generation_request_failure(error));
        return send_error(error);
    } catch (const std::exception& exception) {
        lifecycle.failure(make_internal_request_failure(RequestFailurePhase::Generation, exception.what()));
        return send_error(internal(exception));
    }
}

// Records the finished generation (`totals`, over every generation of the response; its finish reason is the last
// one's) and whether the stream goes on to its terminal events. False when it ends here instead: the last generation
// was cancelled, or the client left (a response failure after the request's record).
template <class Lifecycle>
[[nodiscard]] bool record_stream_generation(Lifecycle& lifecycle, const GenerationOutcome& totals,
                                            const std::atomic<bool>& cancelled) {
    lifecycle.done(totals);
    if (totals.finish_reason != infernix::FinishReason::Cancelled && !cancelled.load(std::memory_order_acquire)) {
        return true;
    }
    lifecycle.response_failure(make_client_disconnected_failure(RequestFailurePhase::Transport));
    return false;
}

// Writes the stream's terminal events (each an event or a list of them, in order) and ends it. True when they were
// written; false when the client left first.
template <class Lifecycle, class... Events>
bool write_stream_terminal(SseTransport& transport, httplib::DataSink& sink, Lifecycle& lifecycle,
                           const Events&... events) {
    try {
        (transport.write(events), ...);
        sink.done();
        return true;
    } catch (const ClientDisconnected&) {
        lifecycle.response_failure(make_client_disconnected_failure(RequestFailurePhase::Transport));
        return false;
    }
}

// The provider's release: the stream is cancelled (a generation still running stops at its next poll), and a stream
// that never started or did not end successfully is a client disconnect, unless the request already has its record.
template <class Stream, class Lifecycle>
[[nodiscard]] auto stream_release(std::shared_ptr<Stream> stream, std::shared_ptr<Lifecycle> lifecycle) {
    return [stream = std::move(stream), lifecycle = std::move(lifecycle)](bool successful) {
        stream->cancelled.store(true, std::memory_order_release);
        if (!successful || !stream->started.load(std::memory_order_acquire)) {
            lifecycle->failure(make_client_disconnected_failure(RequestFailurePhase::Transport));
        }
    };
}

} // namespace infernix::serve
