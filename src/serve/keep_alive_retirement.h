#pragma once

#include <httplib.h>

#include <chrono>
#include <cstddef>
#include <mutex>
#include <string>
#include <unordered_map>

namespace infernix::serve {

// Graceful retirement for idle keep-alive connections.
//
// The HTTP server's keep-alive timeout closes a connection silently: there is no
// response in flight to carry a close signal, so a client that reuses the pooled
// socket right at the deadline races the close and sees a connection reset. The
// documented remedy (nginx, Node) is to make long-idle pooled connections retire
// knowingly: the first request that arrives after a long idle gets `Connection:
// close` on its response, the client stops reusing that socket, and the next
// request opens a fresh connection.
//
// This is the whole mechanism on the Infernix side of the vendored server. A
// connection (keyed by peer address and port) is idle from the end of its last
// response, its whole body written (a stream's last chunk, not its headers), to the
// arrival of its next request, so a long generation, streamed or not, never counts
// as idleness. Call arrived() first in the pre-routing handler and wire respond() as
// the post-routing handler. The hard timeout remains the backstop for sockets that
// never send again.
class KeepAliveRetirement {
public:
    explicit KeepAliveRetirement(std::chrono::steady_clock::duration idle_threshold);

    // Pre-routing: a request arrived; decides whether its response retires the
    // connection (idle for at least the threshold since its last response ended).
    void arrived(const httplib::Request& request);

    // Post-routing, which runs after the server has set its own keep-alive headers
    // and before it writes the response: a retiring response gets `Connection: close`
    // and drops the server's `Keep-Alive` hint (a response must not both invite reuse
    // and announce its end). Every response also records when it has been written.
    void respond(const httplib::Request& request, httplib::Response& response);

private:
    struct Connection {
        std::chrono::steady_clock::time_point finished; // the last response's end, or a request's arrival
        bool retire = false;                            // the request in flight retires the connection
    };

    static constexpr std::size_t kMaxTrackedConnections = 4096;

    void finished(const std::string& key);
    void prune_stale_locked(std::chrono::steady_clock::time_point now);

    const std::chrono::steady_clock::duration idle_threshold_;
    std::mutex mutex_;
    std::unordered_map<std::string, Connection> connections_;
};

} // namespace infernix::serve
