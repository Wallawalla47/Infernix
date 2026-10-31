#include "serve/http_server.h"

#include <nlohmann/json.hpp>

#include <chrono>
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

    // Idle retirement (the client-races-the-silent-close fix): a request arriving after the
    // graceful-idle threshold gets its response stamped `Connection: close`; a back-to-back
    // request keeps the pool warm, and the request after a retirement answers on a fresh
    // connection. The httplib default (0) must stamp nothing.
    {
        httplib::Server server;
        server.Get("/ping", [](const httplib::Request&, httplib::Response& res) {
            res.set_content("pong", "text/plain");
        });
        server.set_keep_alive_timeout(30);
        server.set_keep_alive_graceful_idle(1);
        const int port = server.bind_to_any_port("127.0.0.1");
        failures += check(port > 0, "keep-alive test server could not bind a loopback port");
        if (port > 0) {
            std::thread listener([&server] { server.listen_after_bind(); });
            server.wait_until_ready();
            httplib::Client client("127.0.0.1", port);
            client.set_keep_alive(true);
            const auto warm = client.Get("/ping");
            failures += check(warm && warm->status == 200 &&
                                  !warm->has_header("Connection"),
                              "a back-to-back keep-alive response must not retire the connection");
            std::this_thread::sleep_for(std::chrono::milliseconds(1300));
            const auto retired = client.Get("/ping");
            failures += check(retired && retired->status == 200 &&
                                  retired->get_header_value("Connection") == "close",
                              "the response after a long idle must retire the connection");
            const auto next = client.Get("/ping");
            failures += check(next && next->status == 200 && !next->has_header("Connection"),
                              "the request after a retirement must answer unstamped on a fresh connection");
            server.stop();
            listener.join();
        }
    }

    if (failures == 0) { std::cout << "http routes tests passed\n"; }
    return failures == 0 ? 0 : 1;
}
