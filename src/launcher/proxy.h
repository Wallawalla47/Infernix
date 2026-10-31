#pragma once

// Forwards one HTTP request to the managed infernix-serve on loopback and relays its answer: event
// streams chunk by chunk as they arrive, other bodies whole. A client that disconnects closes the
// upstream connection, which cancels its request in the server. The router lease stays held until
// the upstream exchange has ended.

#include "launcher/router.h"

#include <httplib.h>

#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>

namespace infernix::launcher {

struct ProxyOutcome {
    int status  = 0;
    bool stream = false;
    std::string error; // set when no upstream answer arrived
    // From the request's arrival: the first response byte (a stream's first event) and the end.
    std::optional<std::int64_t> first_byte_ms;
    std::int64_t duration_ms = 0;
    std::uint64_t bytes_out  = 0;
    // The response body (a stream's raw events), when capture was asked for; cut at the limit.
    std::string body;
    bool body_truncated = false;
};

// `finished` runs once the exchange is over (on the thread that finished it). `capture_limit` > 0
// keeps up to that many bytes of the response body in the outcome. `arrived` is when the client's
// request arrived (timings count from it). `label` goes to the server as X-Infernix-Request-Label
// when the client sent none (the server's metrics name the request by it).
void forward_request(const httplib::Request& request, httplib::Response& response, int port,
                     std::shared_ptr<Router::Lease> lease, std::function<void(const ProxyOutcome&)> finished,
                     std::chrono::steady_clock::time_point arrived, std::size_t capture_limit, const std::string& label);

enum class StreamDialect { OpenAI, Anthropic };

// Waits for the model's lease (blocking; `cancelled` turns true when the client leaves).
using AcquireLease =
    std::function<std::shared_ptr<Router::Lease>(const std::function<bool()>& cancelled, Router::Refusal& refusal)>;

// A streamed request whose model must load first: answers 200 with an event stream at once and
// keeps the connection alive while `acquire` waits (an SSE comment ": <loading_note>" every 5 s for
// OpenAI clients, Anthropic's `ping` event for Anthropic ones), so clients with read timeouts do not
// give up during a load of a minute or more; then relays the server's stream. A failed load, or an
// error answer from the server, becomes one error event in the client's protocol.
void forward_after_load(const httplib::Request& request, httplib::Response& response, AcquireLease acquire,
                        StreamDialect dialect, std::string loading_note,
                        std::function<void(const ProxyOutcome&)> finished,
                        std::chrono::steady_clock::time_point arrived, std::size_t capture_limit,
                        const std::string& label);

} // namespace infernix::launcher
