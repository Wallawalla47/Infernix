#include "serve/http_server.h"

#include <nlohmann/json.hpp>

#include <iostream>
#include <string>
#include <thread>

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

    // Idle sockets must not starve admission (the pool-sizing half of the keep-alive fix):
    // cpp-httplib parks one pool worker per open connection for its whole idle window, so a
    // pool sized only to admission lets idle connections block new requests. The ceiling
    // above the base (--max-http-clients territory) must absorb them.
    {
        httplib::Server server;
        server.Get("/ping", [](const httplib::Request&, httplib::Response& res) {
            res.set_content("pong", "text/plain");
        });
        server.set_keep_alive_timeout(30);
        // Two base workers, four headroom workers: hold every base worker with an idle
        // connection and a request must still be admitted through the headroom.
        server.new_task_queue = [] { return new httplib::ThreadPool(2, 6, 16); };
        const int port = server.bind_to_any_port("127.0.0.1");
        failures += check(port > 0, "pool test server could not bind a loopback port");
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
            failures += check(all_warm, "idle connections could not each get their warm-up response");
            // All four connections are now open and idle, holding four workers.
            httplib::Client fresh("127.0.0.1", port);
            fresh.set_read_timeout(5, 0);
            const auto admitted = fresh.Get("/ping");
            failures += check(admitted && admitted->status == 200,
                              "idle keep-alive connections must not starve request admission");
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

    if (failures == 0) { std::cout << "http routes tests passed\n"; }
    return failures == 0 ? 0 : 1;
}
