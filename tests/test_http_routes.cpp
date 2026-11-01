#include "serve/http_server.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

using Json = nlohmann::json;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

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
