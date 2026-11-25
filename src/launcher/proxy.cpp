#include "launcher/proxy.h"

#include "launcher/util.h"

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>

namespace infernix::launcher {
namespace {

constexpr auto kKeepAliveEvery = std::chrono::seconds(5);

bool hop_by_hop(const std::string& name) {
    static constexpr std::string_view kHeaders[] = {"connection", "keep-alive", "proxy-connection", "transfer-encoding",
                                                    "te", "trailer", "upgrade", "host", "content-length",
                                                    "accept-encoding", "content-encoding"};
    const std::string lowered = lower(name);
    for (std::string_view header : kHeaders) {
        if (lowered == header) { return true; }
    }
    return false;
}

// One upstream exchange: a client thread sends the request to infernix-serve and queues what comes
// back; the client-facing side takes the status, headers and chunks from here.
struct Bridge {
    std::mutex mutex;
    std::condition_variable cv;
    bool have_headers = false;
    bool done         = false;
    bool failed       = false;
    // The upstream ended after its status line, before its answer was whole (a server killed or crashed
    // mid-body): not a complete answer, whatever its status.
    bool cut_off      = false;
    int status        = 0;
    httplib::Headers headers;
    // An event stream's chunks, relayed as they come; a whole answer collects in `body`.
    bool whole = false;
    std::deque<std::string> chunks;
    std::string body;
    std::string error;
    std::atomic<bool> downstream_gone{false};
    std::unique_ptr<httplib::Client> client;
    std::thread thread;
    std::shared_ptr<Router::Lease> lease;
    std::function<void(const ProxyOutcome&)> finished;
    ProxyOutcome outcome;
    std::chrono::steady_clock::time_point arrived;
    std::size_t capture_limit = 0;

    ~Bridge() {
        if (thread.joinable()) { thread.join(); }
        lease.reset();
        outcome.duration_ms = elapsed_ms();
        if (finished) { finished(outcome); }
    }

    [[nodiscard]] std::int64_t elapsed_ms() const {
        return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - arrived).count();
    }

    // Bytes on their way to the client: timed, counted and (when asked) kept.
    void delivered(const std::string& bytes) {
        if (!outcome.first_byte_ms) { outcome.first_byte_ms = elapsed_ms(); }
        outcome.bytes_out += bytes.size();
        if (capture_limit == 0) { return; }
        const std::size_t room = capture_limit > outcome.body.size() ? capture_limit - outcome.body.size() : 0;
        if (bytes.size() > room) { outcome.body_truncated = true; }
        outcome.body.append(bytes, 0, std::min(room, bytes.size()));
    }

    void abandon() {
        downstream_gone = true;
        if (client) { client->stop(); }
        cv.notify_all();
    }
};

// The request for infernix-serve: the client's method, raw target, body and end-to-end headers,
// with the launcher's label when the client sent none.
httplib::Request make_upstream(const httplib::Request& request, const std::string& label) {
    httplib::Request upstream;
    upstream.method = request.method;
    upstream.path   = request.target.empty() ? request.path : request.target;
    for (const auto& [name, value] : request.headers) {
        if (!hop_by_hop(name) && !iequals(name, "REMOTE_ADDR") && !iequals(name, "REMOTE_PORT") &&
            !iequals(name, "LOCAL_ADDR") && !iequals(name, "LOCAL_PORT")) {
            upstream.headers.emplace(name, value);
        }
    }
    if (!label.empty() && !request.has_header("X-Infernix-Request-Label")) {
        upstream.headers.emplace("X-Infernix-Request-Label", label);
    }
    upstream.body = request.body;
    return upstream;
}

std::shared_ptr<Bridge> start_bridge(httplib::Request upstream, int port, std::shared_ptr<Router::Lease> lease,
                                     std::function<void(const ProxyOutcome&)> finished,
                                     std::chrono::steady_clock::time_point arrived, std::size_t capture_limit) {
    auto bridge           = std::make_shared<Bridge>();
    bridge->lease         = std::move(lease);
    bridge->finished      = std::move(finished);
    bridge->arrived       = arrived;
    bridge->capture_limit = capture_limit;
    bridge->client        = std::make_unique<httplib::Client>("127.0.0.1", port);
    bridge->client->set_connection_timeout(std::chrono::seconds(10));
    bridge->client->set_read_timeout(std::chrono::hours(2));
    bridge->client->set_write_timeout(std::chrono::minutes(10));
    bridge->client->set_keep_alive(false);
    // The request goes out as a header write then a body write: with Nagle's rule the body would wait for
    // infernix-serve's delayed ACK of the header.
    bridge->client->set_tcp_nodelay(true);
    // The target is the client's raw (already encoded) path and query: forward it unchanged.
    bridge->client->set_path_encode(false);
    Bridge* raw               = bridge.get();
    upstream.response_handler = [raw](const httplib::Response& answer) {
        {
            std::lock_guard lock(raw->mutex);
            raw->status       = answer.status;
            raw->headers      = answer.headers;
            raw->have_headers = true;
            // A whole answer is sized up front when its length is known, so the relay holds one
            // copy of it (forward_request tells a stream from its content type the same way).
            raw->whole = !lower(answer.get_header_value("Content-Type")).starts_with("text/event-stream");
            const std::string length = answer.get_header_value("Content-Length");
            std::size_t bytes        = 0;
            const auto [end, error]  = std::from_chars(length.data(), length.data() + length.size(), bytes);
            if (raw->whole && error == std::errc{} && end == length.data() + length.size()) {
                raw->body.reserve(std::min<std::size_t>(bytes, std::size_t{1} << 30));
            }
        }
        raw->cv.notify_all();
        return !raw->downstream_gone.load();
    };
    upstream.content_receiver = [raw](const char* data, size_t size, size_t, size_t) {
        {
            std::lock_guard lock(raw->mutex);
            if (raw->whole) {
                raw->body.append(data, size);
            } else {
                raw->chunks.emplace_back(data, size);
            }
        }
        raw->cv.notify_all();
        return !raw->downstream_gone.load();
    };
    bridge->thread = std::thread([raw, upstream = std::move(upstream)]() mutable {
        httplib::Response answer;
        httplib::Error error = httplib::Error::Success;
        // A client gone before the connection opens would leave nothing for stop() to shut down.
        const bool ok = !raw->downstream_gone.load() && raw->client->send(upstream, answer, error);
        {
            std::lock_guard lock(raw->mutex);
            raw->done = true;
            if (!ok && !raw->have_headers) {
                raw->failed = true;
                raw->error  = httplib::to_string(error);
            } else if (!ok && !raw->downstream_gone.load()) {
                raw->cut_off = true;
                raw->error   = httplib::to_string(error);
            }
        }
        raw->cv.notify_all();
    });
    return bridge;
}

// An error as one event of the client's protocol (defined below).
std::string error_event(StreamDialect dialect, const std::string& code, const std::string& message);

// One step of relaying an event stream: waits up to a second for chunks and writes them. A stream the upstream
// cut off ends with an error event of the client's protocol, not as if it were complete.
bool pump_stream(const std::shared_ptr<Bridge>& bridge, httplib::DataSink& sink, StreamDialect dialect) {
    std::unique_lock lock(bridge->mutex);
    bridge->cv.wait_for(lock, std::chrono::seconds(1), [&] { return !bridge->chunks.empty() || bridge->done; });
    while (!bridge->chunks.empty()) {
        std::string chunk = std::move(bridge->chunks.front());
        bridge->chunks.pop_front();
        lock.unlock();
        if (!sink.write(chunk.data(), chunk.size())) {
            bridge->abandon();
            return false;
        }
        bridge->delivered(chunk);
        lock.lock();
    }
    if (bridge->done) {
        const bool cut_off = bridge->cut_off;
        const std::string error = bridge->error;
        lock.unlock();
        if (cut_off) {
            bridge->outcome.status = 502;
            bridge->outcome.error  = "infernix-serve ended the stream early: " + error;
            const std::string event = error_event(dialect, "upstream_unavailable", bridge->outcome.error);
            bridge->delivered(event);
            sink.write(event.data(), event.size());
        }
        sink.done();
        return true;
    }
    lock.unlock();
    if (!sink.is_writable()) {
        bridge->abandon();
        return false;
    }
    return true;
}

// Error text can come from the server's raw output (a message in the system code page) or a cut body: invalid
// UTF-8 is replaced, as everywhere else the launcher writes JSON, instead of throwing and dropping the answer.
std::string dump_lenient(const Json& value) { return value.dump(-1, ' ', false, Json::error_handler_t::replace); }

void write_error(httplib::Response& response, int status, const std::string& message) {
    response.status = status;
    response.set_content(
        dump_lenient({{"error", {{"message", message}, {"type", "server_error"}, {"code", "upstream_unavailable"}}}}),
        "application/json");
}

// An error as one event of the client's protocol.
std::string error_event(StreamDialect dialect, const std::string& code, const std::string& message) {
    if (dialect == StreamDialect::Anthropic) {
        return "event: error\ndata: " +
               dump_lenient({{"type", "error"}, {"error", {{"type", "overloaded_error"}, {"message", message}}}}) +
               "\n\n";
    }
    return "data: " + dump_lenient({{"error", {{"message", message}, {"type", "server_error"}, {"code", code}}}}) +
           "\n\n";
}

// The message of an error answer's JSON body (OpenAI or Anthropic shape), else the body itself.
std::string error_message(const std::string& body, int status) {
    try {
        const Json value = Json::parse(body);
        if (value.contains("error") && value["error"].is_object()) {
            return value["error"].value("message", "HTTP " + std::to_string(status));
        }
    } catch (const Json::exception&) {}
    return body.empty() ? "HTTP " + std::to_string(status) : body.substr(0, 500);
}

} // namespace

void forward_request(const httplib::Request& request, httplib::Response& response, int port,
                     std::shared_ptr<Router::Lease> lease, std::function<void(const ProxyOutcome&)> finished,
                     std::chrono::steady_clock::time_point arrived, std::size_t capture_limit, const std::string& label) {
    auto bridge = start_bridge(make_upstream(request, label), port, std::move(lease), std::move(finished), arrived,
                               capture_limit);

    // Wait for the upstream status line (the server answers once it has admitted the request).
    {
        std::unique_lock lock(bridge->mutex);
        while (!bridge->have_headers && !bridge->done) {
            bridge->cv.wait_for(lock, std::chrono::milliseconds(250));
            if (request.is_connection_closed()) {
                lock.unlock();
                bridge->abandon();
                bridge->outcome.status = 499;
                bridge->outcome.error  = "client disconnected";
                response.status        = 499;
                return;
            }
        }
        if (bridge->failed) {
            bridge->outcome.status = 502;
            bridge->outcome.error  = bridge->error;
            lock.unlock();
            write_error(response, 502, "infernix-serve did not answer: " + bridge->error);
            bridge->delivered(response.body);
            return;
        }
    }

    std::string content_type = "application/json";
    response.status          = bridge->status;
    for (const auto& [name, value] : bridge->headers) {
        if (iequals(name, "content-type")) {
            content_type = value;
            continue;
        }
        if (!hop_by_hop(name)) { response.set_header(name, value); }
    }
    bridge->outcome.status = bridge->status;
    const bool stream      = lower(content_type).starts_with("text/event-stream");
    bridge->outcome.stream = stream;

    if (!stream) {
        std::unique_lock lock(bridge->mutex);
        while (!bridge->done) {
            bridge->cv.wait_for(lock, std::chrono::milliseconds(250));
            if (request.is_connection_closed()) {
                lock.unlock();
                bridge->abandon();
                bridge->outcome.status = 499;
                return;
            }
        }
        if (bridge->cut_off) {
            // A partial body with the upstream's 200 would read as a complete answer (its Content-Length is not
            // relayed): the client gets an error instead.
            bridge->outcome.status = 502;
            bridge->outcome.error  = "infernix-serve ended its answer early: " + bridge->error;
            const std::string message = bridge->outcome.error;
            lock.unlock();
            write_error(response, 502, message);
            bridge->delivered(response.body);
            return;
        }
        std::string body = std::move(bridge->body);
        lock.unlock();
        bridge->delivered(body);
        response.set_content(std::move(body), content_type);
        return;
    }

    const StreamDialect dialect =
        request.path.find("/v1/messages") != std::string::npos ? StreamDialect::Anthropic : StreamDialect::OpenAI;
    response.set_chunked_content_provider(
        content_type, [bridge, dialect](size_t, httplib::DataSink& sink) { return pump_stream(bridge, sink, dialect); },
        [bridge](bool success) {
            if (!success) { bridge->abandon(); }
        });
}

void forward_after_load(const httplib::Request& request, httplib::Response& response, AcquireLease acquire,
                        StreamDialect dialect, std::string loading_note,
                        std::function<void(const ProxyOutcome&)> finished,
                        std::chrono::steady_clock::time_point arrived, std::size_t capture_limit,
                        const std::string& label) {
    // The wait for the model runs on its own thread; the client's stream sends keep-alives meanwhile.
    struct Loading {
        std::mutex mutex;
        std::condition_variable cv;
        bool settled = false;
        std::shared_ptr<Router::Lease> lease;
        Router::Refusal refusal;
        std::atomic<bool> cancelled{false};
        std::thread waiter;
        httplib::Request upstream;
        std::shared_ptr<Bridge> bridge;
        bool converted_error = false; // the server answered with an error, sent as an event
        std::function<void(const ProxyOutcome&)> finished;
        ProxyOutcome outcome; // while no bridge runs (the load failed or the client left)
        std::chrono::steady_clock::time_point arrived;
        std::chrono::steady_clock::time_point last_write = std::chrono::steady_clock::now();
        std::size_t capture_limit = 0;

        ~Loading() {
            cancelled = true;
            if (waiter.joinable()) { waiter.join(); }
            if (!bridge && finished) {
                outcome.stream      = true;
                outcome.duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                          std::chrono::steady_clock::now() - arrived)
                                          .count();
                finished(outcome);
            }
        }
    };
    auto loading           = std::make_shared<Loading>();
    loading->upstream      = make_upstream(request, label);
    loading->finished      = std::move(finished);
    loading->arrived       = arrived;
    loading->capture_limit = capture_limit;
    Loading* raw           = loading.get();
    loading->waiter        = std::thread([raw, acquire = std::move(acquire)] {
        Router::Refusal refusal;
        std::shared_ptr<Router::Lease> lease = acquire([raw] { return raw->cancelled.load(); }, refusal);
        {
            std::lock_guard lock(raw->mutex);
            raw->lease   = std::move(lease);
            raw->refusal = refusal;
            raw->settled = true;
        }
        raw->cv.notify_all();
    });

    const std::string keep_alive = dialect == StreamDialect::Anthropic
                                       ? "event: ping\ndata: {\"type\": \"ping\"}\n\n"
                                       : ": " + loading_note + "\n\n";
    response.status = 200;
    response.set_header("Cache-Control", "no-cache");
    response.set_chunked_content_provider(
        "text/event-stream",
        [loading, dialect, keep_alive](size_t, httplib::DataSink& sink) {
            if (!loading->bridge) {
                std::shared_ptr<Router::Lease> lease;
                {
                    std::unique_lock lock(loading->mutex);
                    loading->cv.wait_for(lock, kKeepAliveEvery, [&] { return loading->settled; });
                    if (!loading->settled) {
                        lock.unlock();
                        if (!sink.write(keep_alive.data(), keep_alive.size())) {
                            loading->cancelled = true;
                            loading->outcome.status = 499;
                            loading->outcome.error  = "client disconnected while the model loaded";
                            return false;
                        }
                        loading->last_write = std::chrono::steady_clock::now();
                        return true;
                    }
                    lease = std::move(loading->lease);
                }
                if (!lease) {
                    // The load failed (or timed out): one error event ends the stream.
                    const Router::Refusal& refusal = loading->refusal;
                    const std::string event = error_event(dialect, refusal.code, refusal.message);
                    loading->outcome.status = refusal.status;
                    loading->outcome.error  = refusal.code + ": " + refusal.message;
                    sink.write(event.data(), event.size());
                    sink.done();
                    return true;
                }
                const int port   = lease->port();
                loading->bridge  = start_bridge(std::move(loading->upstream), port, std::move(lease),
                                                std::move(loading->finished), loading->arrived, loading->capture_limit);
                loading->bridge->outcome.stream = true;
            }
            const std::shared_ptr<Bridge>& bridge = loading->bridge;
            {
                std::unique_lock lock(bridge->mutex);
                if (!bridge->have_headers && !bridge->done) {
                    bridge->cv.wait_for(lock, std::chrono::seconds(1));
                    if (!bridge->have_headers && !bridge->done) {
                        // The server has not admitted it yet (its lanes are busy): keep-alives go on.
                        lock.unlock();
                        if (std::chrono::steady_clock::now() - loading->last_write < kKeepAliveEvery) {
                            return sink.is_writable();
                        }
                        loading->last_write = std::chrono::steady_clock::now();
                        if (!sink.write(keep_alive.data(), keep_alive.size())) {
                            bridge->abandon();
                            return false;
                        }
                        return true;
                    }
                }
                if (bridge->failed) {
                    bridge->outcome.status = 502;
                    bridge->outcome.error  = bridge->error;
                    lock.unlock();
                    const std::string event = error_event(dialect, "upstream_unavailable",
                                                          "infernix-serve did not answer: " + bridge->error);
                    sink.write(event.data(), event.size());
                    sink.done();
                    return true;
                }
                bridge->outcome.status = bridge->status;
                // An error answer (JSON, not a stream) becomes one error event once it is complete.
                if (bridge->status != 200 || loading->converted_error) {
                    loading->converted_error = true;
                    if (!bridge->done) {
                        bridge->cv.wait_for(lock, std::chrono::seconds(1));
                        if (!bridge->done) { return sink.is_writable(); }
                    }
                    // A JSON answer is held whole in `body`; anything else arrived as chunks.
                    std::string body = std::move(bridge->body);
                    for (const std::string& chunk : bridge->chunks) { body += chunk; }
                    bridge->body.clear();
                    bridge->chunks.clear();
                    lock.unlock();
                    const std::string event =
                        error_event(dialect, "http_" + std::to_string(bridge->status), error_message(body, bridge->status));
                    bridge->delivered(event);
                    sink.write(event.data(), event.size());
                    sink.done();
                    return true;
                }
                // A 200 that is not an event stream cannot continue the stream this answer started (pump_stream relays
                // only stream chunks): one error event says so instead of an empty stream.
                if (bridge->whole) {
                    bridge->outcome.status = 502;
                    bridge->outcome.error  = "infernix-serve answered a streamed request with a whole body";
                    lock.unlock();
                    const std::string event = error_event(dialect, "upstream_not_a_stream", bridge->outcome.error);
                    bridge->delivered(event);
                    sink.write(event.data(), event.size());
                    sink.done();
                    return true;
                }
            }
            return pump_stream(bridge, sink, dialect);
        },
        [loading](bool success) {
            if (success) { return; }
            loading->cancelled = true;
            if (loading->bridge) { loading->bridge->abandon(); }
        });
}

} // namespace infernix::launcher
