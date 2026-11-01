#include "serve/http_server.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#if defined(__linux__) || defined(__APPLE__)
#    include <arpa/inet.h>
#    include <netinet/in.h>
#    include <sys/socket.h>
#    include <unistd.h>
#elif defined(_WIN32)
#    include <winsock2.h>
#    include <ws2tcpip.h>
#endif

namespace {

using Json = nlohmann::json;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

#if defined(__linux__) || defined(__APPLE__) || defined(_WIN32)
// Raw-socket client for the retirement close assertions: an httplib::Client hides its
// socket, and the proof needed is a FIN on the wire, not a client-visible symptom.
// Winsock is already initialized for the process by the httplib::Server instances.
#    if defined(_WIN32)
using NativeSocket               = SOCKET;
constexpr NativeSocket kNoSocket = INVALID_SOCKET;
#    else
using NativeSocket               = int;
constexpr NativeSocket kNoSocket = -1;
#    endif

NativeSocket raw_connect(int port) {
    const NativeSocket sock = ::socket(AF_INET, SOCK_STREAM, 0);
    if (sock == kNoSocket) { return kNoSocket; }
    sockaddr_in address{};
    address.sin_family      = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port        = htons(static_cast<unsigned short>(port));
    if (::connect(sock, reinterpret_cast<const sockaddr*>(&address), sizeof(address)) != 0) {
#    if defined(_WIN32)
        ::closesocket(sock);
#    else
        ::close(sock);
#    endif
        return kNoSocket;
    }
    return sock;
}

void raw_close(NativeSocket sock) {
#    if defined(_WIN32)
    ::closesocket(sock);
#    else
    ::close(sock);
#    endif
}

bool raw_send_request(NativeSocket sock, const char* request) {
    const std::size_t total = std::string(request).size();
    std::size_t sent = 0;
    while (sent < total) {
        const int n = ::send(sock, request + sent, static_cast<int>(total - sent), 0);
        if (n <= 0) { return false; }
        sent += static_cast<std::size_t>(n);
    }
    return true;
}

bool raw_send_get(NativeSocket sock, const char* path = "/ping") {
    const std::string request = std::string("GET ") + path +
                                " HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: keep-alive\r\n\r\n";
    return raw_send_request(sock, request.c_str());
}

// 0 = readable, 1 = timed out, -1 = error.
int raw_wait_readable(NativeSocket sock, int ms) {
    fd_set readable;
    FD_ZERO(&readable);
    FD_SET(sock, &readable);
    timeval timeout{};
    timeout.tv_sec  = ms / 1000;
    timeout.tv_usec = (ms % 1000) * 1000;
    const int ready =
        ::select(static_cast<int>(sock) + 1, &readable, nullptr, nullptr, &timeout);
    return ready > 0 ? 0 : (ready == 0 ? 1 : -1);
}

// Reads one complete Content-Length-framed response; false on timeout or mid-response EOF.
bool raw_read_response(NativeSocket sock, std::string& out) {
    out.clear();
    std::size_t headers_end = std::string::npos;
    std::size_t body_expected = 0;
    for (;;) {
        if (raw_wait_readable(sock, 3000) != 0) { return false; }
        char buffer[4096];
        const int n = ::recv(sock, buffer, static_cast<int>(sizeof(buffer)), 0);
        if (n <= 0) { return false; }
        out.append(buffer, static_cast<std::size_t>(n));
        if (headers_end == std::string::npos) {
            headers_end = out.find("\r\n\r\n");
            if (headers_end == std::string::npos) { continue; }
            const std::size_t cl = out.find("Content-Length: ");
            if (cl == std::string::npos || cl > headers_end) { return false; }
            body_expected = 0;
            for (std::size_t p = cl + 16; p < out.size() && out[p] >= '0' && out[p] <= '9'; ++p) {
                body_expected = body_expected * 10 + static_cast<std::size_t>(out[p] - '0');
            }
        }
        if (out.size() >= headers_end + 4 + body_expected) { return true; }
    }
}

// Reads one complete chunked response through its zero-length terminator.
bool raw_read_chunked(NativeSocket sock, std::string& out) {
    out.clear();
    std::size_t headers_end = std::string::npos;
    for (;;) {
        if (raw_wait_readable(sock, 3000) != 0) { return false; }
        char buffer[4096];
        const int n = ::recv(sock, buffer, static_cast<int>(sizeof(buffer)), 0);
        if (n <= 0) { return false; }
        out.append(buffer, static_cast<std::size_t>(n));
        if (headers_end == std::string::npos) {
            headers_end = out.find("\r\n\r\n");
            if (headers_end == std::string::npos) { continue; }
        }
        // The terminator only counts after the header block: "0\r\n\r\n" can
        // otherwise match inside a Keep-Alive hint ("max=100\r\n\r\n").
        if (out.find("0\r\n\r\n", headers_end + 4) != std::string::npos) { return true; }
    }
}

// True only when the peer has FIN'd: readable, and recv reports EOF.
bool raw_peer_closed(NativeSocket sock, int ms) {
    if (raw_wait_readable(sock, ms) != 0) { return false; }
    char byte = 0;
    return ::recv(sock, &byte, 1, 0) == 0;
}
#endif

} // namespace

int main() {
    int failures = 0;
    using infernix::serve::api_route_pattern;
    using infernix::serve::canonical_api_path;

    // An Anthropic SDK given a base URL ending in /v1 adds one more /v1; only that one is removed.
    failures += check(canonical_api_path("/v1/v1/messages") == "/v1/messages" &&
                          canonical_api_path("/v1/v1/models/qwen") == "/v1/models/qwen",
                      "a doubled /v1 prefix must name the /v1 endpoint");
    failures += check(canonical_api_path("/v1/messages") == "/v1/messages" &&
                          canonical_api_path("/v1/v1/v1/messages") == "/v1/v1/messages" &&
                          canonical_api_path("/v1/v1") == "/v1/v1" &&
                          canonical_api_path("/health") == "/health",
                      "only a single doubled /v1 prefix may be removed");

    // Errors on a doubled path keep the envelope and request-id header of the endpoint it names.
    infernix::serve::ServeOptions options;
    httplib::Request missing_messages;
    missing_messages.path = "/v1/v1/messages/missing";
    httplib::Response missing_response;
    missing_response.status = 404;
    const auto missing_result =
        infernix::serve::handle_unrendered_http_error(options, missing_messages, missing_response);
    failures += check(missing_result == httplib::Server::HandlerResponse::Handled &&
                          !missing_response.body.empty() &&
                          Json::parse(missing_response.body).at("type") == "error" &&
                          missing_response.has_header("request-id") &&
                          !missing_response.has_header("x-request-id"),
                      "a doubled Anthropic path must keep the Anthropic error envelope");
    httplib::Request oversized_responses;
    oversized_responses.path = "/v1/v1/responses";
    httplib::Response oversized_response;
    oversized_response.status = 413;
    const auto oversized_result = infernix::serve::handle_unrendered_http_error(
        options, oversized_responses, oversized_response);
    failures += check(oversized_result == httplib::Server::HandlerResponse::Handled &&
                          Json::parse(oversized_response.body).at("error").at("code") ==
                              "request_too_large" &&
                          oversized_response.has_header("x-request-id"),
                      "a doubled OpenAI path must keep the OpenAI error envelope");

    // Route patterns reach the handler under either prefix and keep the endpoint's own captures.
    httplib::Server server;
    server.Post(api_route_pattern("/messages"),
                [](const httplib::Request&, httplib::Response& res) {
                    res.set_content("messages", "text/plain");
                });
    server.Get(api_route_pattern(R"(/models/(.+))"),
               [](const httplib::Request& req, httplib::Response& res) {
                   const std::string id = req.matches.size() > 1 ? req.matches[1].str() : "";
                   res.set_content(id, "text/plain");
               });
    const int port = server.bind_to_any_port("127.0.0.1");
    failures += check(port > 0, "test server could not bind a loopback port");
    if (port > 0) {
        std::thread listener([&server] { server.listen_after_bind(); });
        server.wait_until_ready();
        httplib::Client client("127.0.0.1", port);
        const auto post_status = [&client](const char* path) {
            const httplib::Result result = client.Post(path, "{}", "application/json");
            return result ? result->status : -1;
        };
        failures += check(post_status("/v1/messages") == 200 &&
                              post_status("/v1/v1/messages") == 200,
                          "the messages route must answer under /v1 and /v1/v1");
        failures += check(post_status("/messages") == 404 &&
                              post_status("/v1/v1/v1/messages") == 404 &&
                              post_status("/v2/v1/messages") == 404,
                          "only the /v1 and single doubled /v1 prefixes may reach a route");
        const httplib::Result model = client.Get("/v1/v1/models/qwen3.8-27b");
        failures += check(model && model->status == 200 && model->body == "qwen3.8-27b",
                          "a doubled prefix must not shift the route's capture groups");
        server.stop();
        listener.join();
    }

    // Idle retirement (the client-races-the-silent-close fix): a connection idle past the
    // threshold since its last response ended gets its next response stamped `Connection: close`,
    // stripped of the server's own Keep-Alive reuse hint, and the request after it answers
    // unstamped on a fresh connection. A response that takes longer than the threshold, computed
    // before its headers or streamed after them, is not idleness: the request right after it keeps
    // the connection.
    {
        using namespace std::chrono_literals;
        constexpr auto kThreshold = 300ms, kLong = 450ms;
        httplib::Server server;
        server.Get("/ping", [](const httplib::Request&, httplib::Response& res) {
            res.set_content("pong", "text/plain");
        });
        server.Get("/computed", [kLong](const httplib::Request&, httplib::Response& res) {
            std::this_thread::sleep_for(kLong);
            res.set_content("computed", "text/plain");
        });
        server.Get("/streamed", [kLong](const httplib::Request&, httplib::Response& res) {
            res.set_chunked_content_provider("text/plain", [kLong](std::size_t, httplib::DataSink& sink) {
                std::this_thread::sleep_for(kLong);
                sink.write("streamed", 8);
                sink.done();
                return true;
            });
        });
        server.set_keep_alive_timeout(30);
        infernix::serve::KeepAliveRetirement retirement(kThreshold);
        std::mutex ports_mutex;
        std::vector<int> ports; // each request's client port: one connection or several
        server.set_pre_routing_handler([&](const httplib::Request& request, httplib::Response&) {
            retirement.arrived(request);
            std::lock_guard lock(ports_mutex);
            ports.push_back(request.remote_port);
            return httplib::Server::HandlerResponse::Unhandled;
        });
        server.set_post_routing_handler([&retirement](const httplib::Request& request,
                                                       httplib::Response& response) {
            retirement.respond(request, response);
        });
        const int port = server.bind_to_any_port("127.0.0.1");
        failures += check(port > 0, "keep-alive test server could not bind a loopback port");
        if (port > 0) {
            std::thread listener([&server] { server.listen_after_bind(); });
            server.wait_until_ready();
            httplib::Client client("127.0.0.1", port);
            client.set_keep_alive(true);
            const auto kept = [](const httplib::Result& result, const char* body) {
                return result && result->status == 200 && result->body == body && !result->has_header("Connection");
            };
            const bool warm           = kept(client.Get("/ping"), "pong");
            const bool computed       = kept(client.Get("/computed"), "computed");
            const bool after_computed = kept(client.Get("/ping"), "pong");
            const bool streamed       = kept(client.Get("/streamed"), "streamed");
            const bool after_streamed = kept(client.Get("/ping"), "pong");
            failures += check(warm && computed && after_computed,
                              "a response computed past the threshold must not count as idleness");
            failures += check(streamed && after_streamed,
                              "a response streamed past the threshold must not count as idleness");
            std::this_thread::sleep_for(kLong);
            const auto retired = client.Get("/ping");
            failures += check(retired && retired->status == 200 &&
                                  retired->get_header_value("Connection") == "close",
                              "the response after a long idle must retire the connection");
            failures += check(retired && !retired->has_header("Keep-Alive"),
                              "a retired response must not also invite reuse");
            const bool next = kept(client.Get("/ping"), "pong");
            failures += check(next, "the request after a retirement must answer unstamped");
            std::lock_guard lock(ports_mutex);
            failures += check(ports.size() == 7 && std::count(ports.begin(), ports.end() - 1, ports.front()) == 6 &&
                                  ports.back() != ports.front(),
                              "the requests up to the retirement must share one connection and the next open another");
            server.stop();
            listener.join();
        }
    }

#if defined(__linux__) || defined(__APPLE__) || defined(_WIN32)
    // A stamp is not enough: a server that sends `Connection: close` must itself close
    // (RFC 9112), and pooled clients exist that ignore the header. The retired response must
    // therefore arrive on a socket the server FINs behind it — with the same framed bytes an
    // unstamped response would carry, for every body shape: plain, empty, and streamed (a
    // stream's own provider keeps its framing; the close fires after its terminator). The
    // two controls after this block fix the attribution: without the handler the response
    // stays unstamped, and the stamp alone never closes.
    {
        httplib::Server server;
        server.Get("/ping", [](const httplib::Request&, httplib::Response& res) {
            res.set_content("pong", "text/plain");
        });
        server.Get("/empty", [](const httplib::Request&, httplib::Response& res) {
            res.set_content("", "text/plain");
        });
        server.Get("/streamed", [](const httplib::Request&, httplib::Response& res) {
            res.set_chunked_content_provider("text/plain", [](std::size_t, httplib::DataSink& sink) {
                sink.write("streamed", 8);
                sink.done();
                return true;
            });
        });
        server.set_keep_alive_timeout(30);
        infernix::serve::KeepAliveRetirement retirement(std::chrono::seconds(1));
        server.set_pre_routing_handler([&retirement](const httplib::Request& request, httplib::Response&) {
            retirement.arrived(request);
            return httplib::Server::HandlerResponse::Unhandled;
        });
        server.set_post_routing_handler([&retirement](const httplib::Request& request,
                                                       httplib::Response& response) {
            retirement.respond(request, response);
        });
        const int port = server.bind_to_any_port("127.0.0.1");
        failures += check(port > 0, "close-assertion test server could not bind a loopback port");
        if (port > 0) {
            std::thread listener([&server] { server.listen_after_bind(); });
            server.wait_until_ready();
            bool warm_ok    = false;
            bool retired_ok = false;
            bool closed     = false;
            {
                const NativeSocket sock = raw_connect(port);
                if (sock != kNoSocket) {
                    std::string response;
                    warm_ok = raw_send_get(sock) && raw_read_response(sock, response) &&
                              response.rfind("HTTP/1.1 200 OK\r\n", 0) == 0 &&
                              response.find("Connection:") == std::string::npos;
                    std::this_thread::sleep_for(std::chrono::milliseconds(1300));
                    retired_ok =
                        raw_send_get(sock) && raw_read_response(sock, response) &&
                        response.find("Connection: close") != std::string::npos &&
                        response.find("Keep-Alive") == std::string::npos &&
                        response.find("Transfer-Encoding") == std::string::npos &&
                        response.find("Content-Length: 4") != std::string::npos &&
                        response.size() == response.find("\r\n\r\n") + 4 + 4 &&
                        response.compare(response.find("\r\n\r\n") + 4, 4, "pong") == 0;
                    closed = raw_peer_closed(sock, 2000);
                    raw_close(sock);
                }
            }
            failures += check(warm_ok,
                              "the first raw keep-alive response must be unstamped and complete");
            failures += check(retired_ok,
                              "the retired response must carry the stamp and the full framed body");
            failures += check(closed,
                              "a stamped response must be followed by the server closing the socket");
            bool empty_closed = false;
            {
                const NativeSocket sock = raw_connect(port);
                if (sock != kNoSocket) {
                    std::string response;
                    (void)(raw_send_get(sock, "/empty") && raw_read_response(sock, response));
                    std::this_thread::sleep_for(std::chrono::milliseconds(1300));
                    empty_closed =
                        raw_send_get(sock, "/empty") && raw_read_response(sock, response) &&
                        response.find("Connection: close") != std::string::npos &&
                        response.find("Content-Length: 0") != std::string::npos &&
                        raw_peer_closed(sock, 2000);
                    raw_close(sock);
                }
            }
            failures += check(empty_closed,
                              "an empty-body retired response keeps Content-Length: 0 and closes");
            bool stream_closed = false;
            {
                const NativeSocket sock = raw_connect(port);
                if (sock != kNoSocket) {
                    std::string response;
                    (void)(raw_send_get(sock, "/streamed") && raw_read_chunked(sock, response));
                    std::this_thread::sleep_for(std::chrono::milliseconds(1300));
                    stream_closed =
                        raw_send_get(sock, "/streamed") && raw_read_chunked(sock, response) &&
                        response.find("Connection: close") != std::string::npos &&
                        response.find("Transfer-Encoding: chunked") != std::string::npos &&
                        response.find("0\r\n\r\n") != std::string::npos &&
                        raw_peer_closed(sock, 2000);
                    raw_close(sock);
                }
            }
            failures += check(stream_closed,
                              "a retired streamed response keeps its terminator and closes behind it");
            server.stop();
            listener.join();
        }
    }

    // Control (a): with no handlers registered, the same long-idle response is not stamped
    // and the socket stays open — the stamp and the close above both come from the
    // retirement mechanism, not from the library's own behavior.
    {
        httplib::Server server;
        server.Get("/ping", [](const httplib::Request&, httplib::Response& res) {
            res.set_content("pong", "text/plain");
        });
        server.set_keep_alive_timeout(30);
        const int port = server.bind_to_any_port("127.0.0.1");
        failures += check(port > 0, "control-A test server could not bind a loopback port");
        if (port > 0) {
            std::thread listener([&server] { server.listen_after_bind(); });
            server.wait_until_ready();
            const NativeSocket sock = raw_connect(port);
            failures += check(sock != kNoSocket, "control-A raw client could not connect");
            bool idle_unstamped = false;
            bool still_open     = false;
            if (sock != kNoSocket) {
                std::string response;
                (void)(raw_send_get(sock) && raw_read_response(sock, response));
                std::this_thread::sleep_for(std::chrono::milliseconds(1300));
                idle_unstamped = raw_send_get(sock) && raw_read_response(sock, response) &&
                                 response.find("Connection:") == std::string::npos;
                still_open = !raw_peer_closed(sock, 300);
                raw_close(sock);
            }
            failures += check(idle_unstamped,
                              "control: a long-idle response with no handler must not be stamped");
            failures += check(still_open,
                              "control: the library must not close an idle socket before its timeout");
            server.stop();
            listener.join();
        }
    }

    // Control (b): the stamp without the close mechanism must not produce the EOF — if the
    // socket closed anyway, the close assertion above would be decoration. The header-only
    // handler below is exactly the pre-close-mechanism behavior.
    {
        httplib::Server server;
        server.Get("/ping", [](const httplib::Request&, httplib::Response& res) {
            res.set_content("pong", "text/plain");
        });
        server.set_keep_alive_timeout(30);
        server.set_post_routing_handler([](const httplib::Request&, httplib::Response& response) {
            response.set_header("Connection", "close");
            response.headers.erase("Keep-Alive");
        });
        const int port = server.bind_to_any_port("127.0.0.1");
        failures += check(port > 0, "control-B test server could not bind a loopback port");
        if (port > 0) {
            std::thread listener([&server] { server.listen_after_bind(); });
            server.wait_until_ready();
            const NativeSocket sock = raw_connect(port);
            failures += check(sock != kNoSocket, "control-B raw client could not connect");
            bool stamped_but_open = false;
            if (sock != kNoSocket) {
                std::string response;
                stamped_but_open = raw_send_get(sock) && raw_read_response(sock, response) &&
                                   response.find("Connection: close") != std::string::npos &&
                                   !raw_peer_closed(sock, 300);
                raw_close(sock);
            }
            failures += check(stamped_but_open,
                              "control: stamping Connection: close alone must not close the socket");
            server.stop();
            listener.join();
        }
    }
#endif

    // Idle sockets must not starve admission (the pool-sizing half of the keep-alive fix):
    // cpp-httplib parks one pool worker per open connection for its whole idle window, so a
    // pool sized to the admitted-request count alone lets that many idle sockets block new
    // requests. The built-in sizing doubles it: with (max_concurrency 1, max_pending 1) the
    // stock "+1" formula owns 3 workers — hold exactly that many idle connections and a new
    // request still finds one of the doubled (4) workers.
    {
        httplib::Server server;
        server.Get("/ping", [](const httplib::Request&, httplib::Response& res) {
            res.set_content("pong", "text/plain");
        });
        server.set_keep_alive_timeout(30);
        infernix::serve::ServeOptions pool_options;
        pool_options.max_concurrency      = 1;
        pool_options.max_pending_requests = 1;
        server.new_task_queue = [&pool_options] { return infernix::serve::new_http_worker_pool(pool_options); };
        const int port = server.bind_to_any_port("127.0.0.1");
        failures += check(port > 0, "pool test server could not bind a loopback port");
        if (port > 0) {
            std::thread listener([&server] { server.listen_after_bind(); });
            server.wait_until_ready();
            std::vector<httplib::Client> idle_clients;
            bool all_warm = true;
            for (int i = 0; i < 3; ++i) {
                idle_clients.emplace_back("127.0.0.1", port);
                idle_clients.back().set_keep_alive(true);
                const auto warm = idle_clients.back().Get("/ping");
                all_warm = all_warm && warm && warm->status == 200;
            }
            failures += check(all_warm, "idle connections could not each get their warm-up response");
            // All three connections are now open and idle — every worker the stock "+1"
            // formula would have built. A fourth request must still be admitted.
            httplib::Client fresh("127.0.0.1", port);
            fresh.set_read_timeout(5, 0);
            const auto admitted = fresh.Get("/ping");
            failures += check(admitted && admitted->status == 200,
                              "idle keep-alive connections must not starve request admission");
            server.stop();
            listener.join();
        }
    }

    // Control: the same hold-and-admit probe against the stock "+1" pool (3 workers, ceiling
    // == base for the (1,1) options above) must starve. If this block passes, the one above
    // proves nothing about the doubling.
    {
        httplib::Server server;
        server.Get("/ping", [](const httplib::Request&, httplib::Response& res) {
            res.set_content("pong", "text/plain");
        });
        server.set_keep_alive_timeout(30);
        server.new_task_queue = [] { return new httplib::ThreadPool(3, 3, 2); };
        const int port = server.bind_to_any_port("127.0.0.1");
        failures += check(port > 0, "control-+1 test server could not bind a loopback port");
        if (port > 0) {
            std::thread listener([&server] { server.listen_after_bind(); });
            server.wait_until_ready();
            std::vector<httplib::Client> idle_clients;
            bool all_warm = true;
            for (int i = 0; i < 3; ++i) {
                idle_clients.emplace_back("127.0.0.1", port);
                idle_clients.back().set_keep_alive(true);
                const auto warm = idle_clients.back().Get("/ping");
                all_warm = all_warm && warm && warm->status == 200;
            }
            failures += check(all_warm, "control-+1: the three idle connections must be served");
            httplib::Client fresh("127.0.0.1", port);
            fresh.set_read_timeout(0, 500000);
            const auto starved = fresh.Get("/ping");
            failures += check(!starved,
                              "control: the +1-formula pool must starve the fourth request");
            server.stop();
            listener.join();
        }
    }

    // --max-http-clients raises the ceiling above the doubled base: with (1,1) the base is
    // 4; hold every base worker with idle connections and a fifth request spawns within the
    // raised ceiling (6).
    {
        httplib::Server server;
        server.Get("/ping", [](const httplib::Request&, httplib::Response& res) {
            res.set_content("pong", "text/plain");
        });
        server.set_keep_alive_timeout(30);
        infernix::serve::ServeOptions pool_options;
        pool_options.max_concurrency      = 1;
        pool_options.max_pending_requests = 1;
        pool_options.max_http_clients     = 6; // base (0) would cap the pool at the doubled 4
        server.new_task_queue = [&pool_options] { return infernix::serve::new_http_worker_pool(pool_options); };
        const int port = server.bind_to_any_port("127.0.0.1");
        failures += check(port > 0, "ceiling test server could not bind a loopback port");
        if (port > 0) {
            std::thread listener([&server] { server.listen_after_bind(); });
            server.wait_until_ready();
            std::vector<httplib::Client> idle_clients;
            bool all_warm = true;
            for (int i = 0; i < 4; ++i) {
                idle_clients.emplace_back("127.0.0.1", port);
                idle_clients.back().set_keep_alive(true);
                const auto warm = idle_clients.back().Get("/ping");
                all_warm = all_warm && warm && warm->status == 200;
            }
            failures += check(all_warm, "the four base-sized idle connections must be served");
            httplib::Client fresh("127.0.0.1", port);
            fresh.set_read_timeout(5, 0);
            const auto admitted = fresh.Get("/ping");
            failures += check(admitted && admitted->status == 200,
                              "--max-http-clients must admit a request beyond the doubled base");
            server.stop();
            listener.join();
        }
    }

    // The ceiling is a real bound (the pool-full case): with every worker held and the queue
    // full, a further connection is refused by the accept loop — closed without a response,
    // not crashed or wedged. ThreadPool(base 1, ceiling 2, queue 1): A and B park the two
    // workers as idle keep-alive sockets, C occupies the single queue slot, D is overflow.
    {
        httplib::Server server;
        server.Get("/ping", [](const httplib::Request&, httplib::Response& res) {
            res.set_content("pong", "text/plain");
        });
        server.set_keep_alive_timeout(30);
        server.new_task_queue = [] { return new httplib::ThreadPool(1, 2, 1); };
        const int port = server.bind_to_any_port("127.0.0.1");
        failures += check(port > 0, "pool-full test server could not bind a loopback port");
        if (port > 0) {
            std::thread listener([&server] { server.listen_after_bind(); });
            server.wait_until_ready();
            httplib::Client a("127.0.0.1", port);
            httplib::Client b("127.0.0.1", port);
            a.set_keep_alive(true);
            b.set_keep_alive(true);
            const auto warm_a = a.Get("/ping");
            const auto warm_b = b.Get("/ping");
            failures += check(warm_a && warm_b && warm_a->status == 200 && warm_b->status == 200,
                              "the two ceiling connections must be served");
            // C is accepted but queued behind the two parked workers: no worker frees within
            // its read timeout, so it goes unanswered while the ceiling is held.
            httplib::Client c("127.0.0.1", port);
            c.set_read_timeout(0, 300000);
            const auto queued = c.Get("/ping");
            failures += check(!queued, "a connection queued behind a full pool must go unanswered");
            // D arrives with the queue full too: the accept loop refuses it outright.
            httplib::Client d("127.0.0.1", port);
            d.set_read_timeout(0, 300000);
            const auto refused = d.Get("/ping");
            failures += check(!refused, "a connection beyond pool and queue must be refused");
            server.stop();
            listener.join();
        }
    }

    // Control: the same overflow probe against an unbounded ceiling (base 1, ceiling 64,
    // queue 1) must admit, not refuse — every dynamic request spawns its own worker. If
    // this block refuses, the refusal above is not attributable to the ceiling.
    {
        httplib::Server server;
        server.Get("/ping", [](const httplib::Request&, httplib::Response& res) {
            res.set_content("pong", "text/plain");
        });
        server.set_keep_alive_timeout(30);
        server.new_task_queue = [] { return new httplib::ThreadPool(1, 64, 1); };
        const int port = server.bind_to_any_port("127.0.0.1");
        failures += check(port > 0, "control-unbounded test server could not bind a loopback port");
        if (port > 0) {
            std::thread listener([&server] { server.listen_after_bind(); });
            server.wait_until_ready();
            httplib::Client a("127.0.0.1", port);
            a.set_keep_alive(true);
            const auto warm_a = a.Get("/ping");
            failures += check(warm_a && warm_a->status == 200, "control: the base connection must be served");
            httplib::Client b("127.0.0.1", port);
            b.set_read_timeout(5, 0);
            const auto spawned = b.Get("/ping");
            failures += check(spawned && spawned->status == 200,
                              "control: an unbounded ceiling must admit where the bounded one refuses");
            server.stop();
            listener.join();
        }
    }

    if (failures == 0) { std::cout << "http routes tests passed\n"; }
    return failures == 0 ? 0 : 1;
}
