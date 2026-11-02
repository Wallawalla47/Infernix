#include "serve/http_server.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <exception>
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
    server.Get(api_route_pattern("/responses"), infernix::serve::handle_responses_upgrade);
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
        // A Responses WebSocket upgrade (Codex's built-in provider) answers 426 with the OpenAI envelope under either
        // prefix, so the client switches to HTTP at once; a plain GET still finds nothing (IJ-02).
        const httplib::Headers upgrade{{"Connection", "Upgrade"},
                                       {"Upgrade", "WebSocket"},
                                       {"Sec-WebSocket-Version", "13"},
                                       {"Sec-WebSocket-Key", "dGhlIHNhbXBsZSBub25jZQ=="}};
        for (const char* path : {"/v1/responses", "/v1/v1/responses"}) {
            const httplib::Result refused = client.Get(path, upgrade);
            failures += check(refused && refused->status == 426 &&
                                  Json::parse(refused->body).at("error").at("code") == "websocket_not_supported",
                              "a Responses WebSocket upgrade did not answer 426 with the OpenAI error");
        }
        const httplib::Result plain = client.Get("/v1/responses");
        failures += check(plain && plain->status == 404, "a plain GET on /v1/responses did not answer 404");
        server.stop();
        listener.join();
    }

    // The server advertises its keep-alive timeout and closes an idle connection only after the
    // buffer: a request 1.5 s after the first, past the advertised 1 s, still arrives on the same
    // connection (the same client port), so a client that retires at the advertised deadline
    // always closes first.
    {
        httplib::Server server;
        std::vector<int> ports;
        std::mutex ports_mutex;
        server.Get("/ping", [&](const httplib::Request& req, httplib::Response& res) {
            const std::lock_guard<std::mutex> lock(ports_mutex);
            ports.push_back(req.remote_port);
            res.set_content("pong", "text/plain");
        });
        server.set_keep_alive_timeout(1);
        server.set_keep_alive_timeout_buffer(2);
        const int port = server.bind_to_any_port("127.0.0.1");
        failures += check(port > 0, "keep-alive buffer test server could not bind a loopback port");
        if (port > 0) {
            std::thread listener([&server] { server.listen_after_bind(); });
            server.wait_until_ready();
            httplib::Client client("127.0.0.1", port);
            client.set_keep_alive(true);
            const auto first = client.Get("/ping");
            failures += check(first && first->status == 200 &&
                                  first->get_header_value("Keep-Alive").starts_with("timeout=1,"),
                              "the response must advertise the keep-alive timeout, not the buffered close");
            std::this_thread::sleep_for(std::chrono::milliseconds(1500));
            const auto second = client.Get("/ping");
            failures += check(second && second->status == 200, "the request after the advertised timeout failed");
            const std::lock_guard<std::mutex> lock(ports_mutex);
            failures += check(ports.size() == 2 && ports[0] == ports[1],
                              "an idle connection must stay open past the advertised timeout until the buffer ends");
            server.stop();
            listener.join();
        }
    }

    // Idle sockets must not starve admission (the pool-sizing half of the keep-alive fix):
    // cpp-httplib parks one pool worker per open connection for its whole idle window, so the
    // server's pool grows past request admission up to --max-http-clients. With one lane and one
    // waiting request (three base workers), four idle connections must still leave room for a new one.
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
            for (int i = 0; i < 4; ++i) {
                idle_clients.emplace_back("127.0.0.1", port);
                idle_clients.back().set_keep_alive(true);
                const auto warm = idle_clients.back().Get("/ping");
                all_warm = all_warm && warm && warm->status == 200;
            }
            failures += check(all_warm, "idle connections could not each get their warm-up response");
            // All four connections are now open and idle, each holding a worker.
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
    // not crashed or wedged. One lane and no waiting requests give two base workers and a
    // one-request queue; --max-http-clients 3 lets one more worker spawn. A, B and C hold the
    // three workers as idle keep-alive sockets, D occupies the queue slot, E is overflow.
    {
        httplib::Server server;
        server.Get("/ping", [](const httplib::Request&, httplib::Response& res) {
            res.set_content("pong", "text/plain");
        });
        server.set_keep_alive_timeout(30);
        infernix::serve::ServeOptions pool_options;
        pool_options.max_concurrency      = 1;
        pool_options.max_pending_requests = 0;
        pool_options.max_http_clients     = 3;
        server.new_task_queue = [&pool_options] { return infernix::serve::new_http_worker_pool(pool_options); };
        const int port = server.bind_to_any_port("127.0.0.1");
        failures += check(port > 0, "pool-full test server could not bind a loopback port");
        if (port > 0) {
            std::thread listener([&server] { server.listen_after_bind(); });
            server.wait_until_ready();
            std::vector<httplib::Client> held;
            bool all_served = true;
            for (int i = 0; i < 3; ++i) {
                held.emplace_back("127.0.0.1", port);
                held.back().set_keep_alive(true);
                const auto warm = held.back().Get("/ping");
                all_served = all_served && warm && warm->status == 200;
            }
            failures += check(all_served, "the connections up to the ceiling must be served");
            // D is accepted but queued behind the three held workers: no worker frees within
            // its read timeout, so it goes unanswered while the ceiling is held.
            httplib::Client d("127.0.0.1", port);
            d.set_read_timeout(0, 300000);
            const auto queued = d.Get("/ping");
            failures += check(!queued, "a connection queued behind a full pool must go unanswered");
            // E arrives with the queue full too: the accept loop refuses it outright.
            httplib::Client e("127.0.0.1", port);
            e.set_read_timeout(0, 300000);
            const auto refused = e.Get("/ping");
            failures += check(!refused, "a connection beyond pool and queue must be refused");
            server.stop();
            listener.join();
        }
    }

    // A ceiling below admission is raised to it: cpp-httplib's ThreadPool rejects a ceiling below
    // its base, and --max-http-clients must never refuse what --max-concurrency promises.
    {
        infernix::serve::ServeOptions pool_options;
        pool_options.max_concurrency      = 2;
        pool_options.max_pending_requests = 4;
        pool_options.max_http_clients     = 1;
        bool created = false;
        try {
            httplib::TaskQueue* pool = infernix::serve::new_http_worker_pool(pool_options);
            pool->shutdown();
            delete pool;
            created = true;
        } catch (const std::exception&) {}
        failures += check(created, "a --max-http-clients below admission must be raised, not rejected");
    }

    if (failures == 0) { std::cout << "http routes tests passed\n"; }
    return failures == 0 ? 0 : 1;
}
