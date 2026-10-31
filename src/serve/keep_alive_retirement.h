#pragma once

#include <httplib.h>

#include <chrono>
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
// This is the whole mechanism on the Infernix side of the vendored server: wire
// handle() as the post-routing handler. It tracks when each connection last
// finished a request (keyed by peer address and port), and stamps the response
// when a request arrives after `idle_threshold` of silence. The hard timeout
// remains the backstop for sockets that never send again.
class KeepAliveRetirement {
public:
    explicit KeepAliveRetirement(std::chrono::seconds idle_threshold);

    // Post-routing hook: decide retirement for this request, then record its
    // completion time. Runs after the server has written its own keep-alive
    // headers, so a retired response drops the `Keep-Alive` hint it added — the
    // response must not both invite reuse and announce its end.
    void handle(const httplib::Request& request, httplib::Response& response);

private:
    static constexpr std::size_t kMaxTrackedConnections = 4096;

    void prune_stale_locked(std::chrono::steady_clock::time_point now);

    const std::chrono::seconds idle_threshold_;
    std::mutex mutex_;
    std::unordered_map<std::string, std::chrono::steady_clock::time_point> last_finished_;
};

} // namespace infernix::serve
