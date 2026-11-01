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
// A stamp alone is decoration: pooled clients exist that ignore the header, and
// RFC 9112 obliges a server that sends it to close. So a retired response is also
// served through a content provider that cancels itself after the last byte —
// the public-API lever that makes the vendored server end the connection loop and
// take the graceful (FIN-first, drained) close path itself. HEAD is the exception
// the library leaves without a lever (its provider never runs): it stays unstamped
// and falls back to the timeout close.
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
    // and before it writes the response: a retiring response gets `Connection: close`,
    // drops the server's `Keep-Alive` hint (a response must not both invite reuse
    // and announce its end), and is served through a provider that closes the socket
    // behind it. Every response also records when it has been written.
    void respond(const httplib::Request& request, httplib::Response& response);

private:
    // Stamps the close and converts the response to a provider that reports itself
    // canceled after the last body byte, so the response reaches the client complete
    // and the server then closes the socket. The plain-body path keeps its
    // Content-Length framing byte-for-byte; a stream keeps its own provider and gets
    // the cancel only after that provider reports done(); an empty body uses the
    // without-length provider, which emits no framing of its own. Requires a
    // non-HEAD request — the library skips providers on HEAD, which would stamp a
    // close the server would not perform.
    static void close_after_response(httplib::Response& response);
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
