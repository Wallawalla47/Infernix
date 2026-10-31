#include "serve/http_transport.h"
#include "serve/request.h"

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#if defined(__linux__)
#    include <arpa/inet.h>
#    include <netinet/in.h>
#    include <netinet/tcp.h>
#    include <sys/socket.h>
#    include <unistd.h>
#elif defined(_WIN32)
#    include <winsock2.h>
#    include <ws2tcpip.h>
#endif

namespace {

using infernix::serve::ClientDisconnected;
using infernix::serve::SseTransport;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

int test_sse_transport() {
    using namespace std::chrono_literals;
    int failures = 0;

    std::vector<std::string> writes;
    bool writable = true;
    bool write_ok = true;
    httplib::DataSink sink;
    sink.write = [&](const char* data, std::size_t size) {
        if (!write_ok) { return false; }
        writes.emplace_back(data, size);
        return true;
    };
    sink.is_writable = [&] { return writable; };

    std::atomic<bool> cancelled{false};
    const SseTransport::Clock::time_point start{100s};
    SseTransport transport(sink, cancelled, 5s, start);

    failures += check(!transport.poll(start + 4999ms) && writes.empty() && !cancelled,
                      "SSE transport emitted a heartbeat before the quiet interval");
    failures +=
        check(!transport.poll(start + 5s) &&
                  writes == std::vector<std::string>{std::string(SseTransport::kHeartbeatComment)},
              "SSE transport did not emit the standard comment heartbeat on time");
    failures += check(!transport.poll(start + 9999ms) && writes.size() == 1,
                      "SSE heartbeat did not reset the quiet interval");

    transport.write("data: token\n\n", start + 10s);
    failures += check(!transport.poll(start + 14999ms) && writes.size() == 2,
                      "normal SSE output did not reset the heartbeat interval");
    failures += check(!transport.poll(start + 15s) && writes.size() == 3 &&
                          writes.back() == SseTransport::kHeartbeatComment,
                      "quiet SSE output did not resume heartbeats after a data event");

    transport.write(std::vector<std::string>{"data: one\n\n", "data: two\n\n"}, start + 16s);
    failures +=
        check(writes.size() == 5 && writes[3] == "data: one\n\n" && writes[4] == "data: two\n\n",
              "SSE transport changed ordered multi-event output");

    writable = false;
    failures += check(transport.poll(start + 16s) && cancelled,
                      "unwritable SSE transport did not cancel its request");
    bool cancelled_write_threw = false;
    try {
        transport.write("data: unreachable\n\n", start + 16s);
    } catch (const ClientDisconnected&) { cancelled_write_threw = true; }
    failures += check(cancelled_write_threw && writes.size() == 5,
                      "cancelled SSE transport accepted another event");

    httplib::DataSink failed_sink;
    failed_sink.write       = [](const char*, std::size_t) { return false; };
    failed_sink.is_writable = [] { return true; };
    std::atomic<bool> heartbeat_cancelled{false};
    SseTransport failed_heartbeat(failed_sink, heartbeat_cancelled, 5s, start);
    failures += check(failed_heartbeat.poll(start + 5s) && heartbeat_cancelled,
                      "failed SSE heartbeat did not cancel its request");

    write_ok = false;
    writable = true;
    std::atomic<bool> event_cancelled{false};
    SseTransport failed_event(sink, event_cancelled, 5s, start);
    bool failed_event_threw = false;
    try {
        failed_event.write("data: failed\n\n", start);
    } catch (const ClientDisconnected&) { failed_event_threw = true; }
    failures += check(failed_event_threw && event_cancelled,
                      "failed SSE event write did not cancel its request");

    std::atomic<bool> releaser_cancelled{true};
    SseTransport released(sink, releaser_cancelled, 5s, start);
    failures += check(released.poll(start),
                      "response resource cancellation was not observed by SSE transport");
    return failures;
}

int test_ngram_identity() {
    using namespace infernix::serve;
    int failures = 0;
    ServeOptions options;
    httplib::Request request;
    request.set_header("x-session-affinity", "conversation-a");
    const RequestJson empty = {{"store", false}};
    failures += check(resolve_ngram_session(request, empty, options).key.empty(),
                      "disabled archive bound identity");
    options.speculative.ngram_archive_bytes = 512ULL << 20;
    failures += check(resolve_ngram_session(request, empty, options).key.empty(),
                      "native identity retained without explicit opt-in");
    options.ngram_native_sessions = true;
    failures += check(resolve_ngram_session(request, empty, options).key == "kilo:conversation-a",
                      "native affinity was not bound");
    request.set_header("x-infernix-draft-reset", "1");
    failures +=
        check(resolve_ngram_session(request, empty, options).reset, "native reset was ignored");
    request.headers.clear();
    RequestJson codex = {{"store", false},
                         {"client_metadata", {{"thread_id", "task-a"}, {"session_id", "task-a"}}}};
    failures += check(resolve_ngram_session(request, codex, options).key == "codex:task-a",
                      "Codex conversation was not bound");
    codex["client_metadata"]["session_id"] = "task-b";
    failures += check(resolve_ngram_session(request, codex, options).key.empty(),
                      "contradictory Codex identity accepted");
    const RequestJson claude = {
        {"metadata", {{"user_id", R"({"device_id":"shared-device","session_id":"chat-a"})"}}}};
    failures += check(resolve_ngram_session(request, claude, options).key == "claude:chat-a",
                      "Claude conversation was not bound");
    request.set_header("x-session-affinity", "other-task");
    failures += check(resolve_ngram_session(request, claude, options).key.empty(),
                      "ambiguous native identities merged");
    request.headers.clear();
    for (const RequestJson& body :
         {RequestJson{{"prompt_cache_key", "shared"}},
          RequestJson{{"metadata", {{"user_id", "shared-user"}}}},
          RequestJson{{"metadata", {{"user_id", R"({"device_id":"shared-device"})"}}}}}) {
        failures += check(resolve_ngram_session(request, body, options).key.empty(),
                          "non-conversation metadata authorized retention");
    }
    options.ngram_native_sessions = false;
    request.set_header("x-infernix-draft-session", "fork");
    request.set_header("x-infernix-draft-parent", "root");
    request.set_header("x-infernix-draft-generation", "3");
    auto hints = resolve_ngram_session(request, empty, options);
    failures += check(hints.key == "explicit:fork" && hints.parent == "explicit:root" &&
                          hints.parent_generation == 3,
                      "explicit ancestor snapshot was not preserved");
    request.set_header("x-infernix-draft-session", "different");
    failures += check(resolve_ngram_session(request, empty, options).key.empty(),
                      "duplicate session header accepted");
    request.headers.clear();
    request.set_header("x-infernix-draft-session", std::string(241, 'x'));
    failures += check(resolve_ngram_session(request, empty, options).key.empty(),
                      "unbounded session header accepted");
    return failures;
}

int test_request_label() {
    using namespace infernix::serve;
    int failures = 0;
    httplib::Request request;
    failures += check(request_label(request).empty(), "absent label header gave a label");
    request.set_header("X-Infernix-Request-Label", "bench fl02/a");
    failures += check(request_label(request) == "bench fl02/a", "valid label was not read");
    request.set_header("x-infernix-request-label", "second");
    failures += check(request_label(request).empty(), "repeated label header accepted");
    for (const std::string& value : {std::string{}, std::string(129, 'x'), std::string("tab\there"),
                                     std::string("caf\xc3\xa9")}) {
        request.headers.clear();
        request.set_header("x-infernix-request-label", value);
        failures += check(request_label(request).empty(), "malformed label accepted");
    }
    request.headers.clear();
    request.set_header("x-infernix-request-label", std::string(128, 'x'));
    failures += check(request_label(request).size() == 128, "128-character label rejected");
    return failures;
}

int test_ngram_generation() {
    using namespace infernix;
    using namespace infernix::serve;
    int failures = 0;
    const NgramArchiveStats complete{
        .enabled = true, .bound = true, .published = true, .generation = 9};
    httplib::Response response;
    set_ngram_generation_header(response, complete);
    failures += check(response.get_header_value("X-Infernix-Draft-Generation") == "9",
                      "completed archive generation header absent");
    failures += check(ngram_generation_comment(complete) == ": infernix-draft-generation: 9\n\n",
                      "stream generation is not a standard SSE comment");
    for (int missing = 0; missing < 4; ++missing) {
        auto stats = complete;
        if (missing == 0) { stats.enabled = false; }
        if (missing == 1) { stats.bound = false; }
        if (missing == 2) { stats.published = false; }
        if (missing == 3) { stats.generation = 0; }
        httplib::Response absent;
        set_ngram_generation_header(absent, stats);
        failures += check(!absent.has_header("X-Infernix-Draft-Generation") &&
                              ngram_generation_comment(stats).empty(),
                          "unpublished archive advertised a completed generation");
    }
    return failures;
}

int test_sse_response_headers() {
    httplib::Response response;
    infernix::serve::prepare_sse_response(response);
    return check(response.get_header_value("Cache-Control") == "no-cache" &&
                     response.get_header_value("X-Accel-Buffering") == "no",
                 "SSE response lost its no-cache or anti-buffering header");
}

int test_prompt_json_member_order() {
    httplib::Request request;
    request.body =
        R"({"model":"claude-local","tools":[{"name":"probe","input_schema":{"type":"object","properties":{"zeta":{"type":"string"},"alpha":{"type":"object","properties":{"yankee":{"type":"integer"},"bravo":{"type":"boolean"}}}}}}],"messages":[{"role":"assistant","content":[{"type":"tool_use","id":"toolu_1","name":"probe","input":{"zeta":"last","alpha":{"yankee":2,"bravo":true}}}]}],"max_tokens":32})";

    const auto parsed = infernix::serve::parse_json_body(request);
    return check(
        parsed.at("tools").at(0).at("input_schema").at("properties").dump() ==
                R"({"zeta":{"type":"string"},"alpha":{"type":"object","properties":{"yankee":{"type":"integer"},"bravo":{"type":"boolean"}}}})" &&
            parsed.at("messages").at(0).at("content").at(0).at("input").dump() ==
                R"({"zeta":"last","alpha":{"yankee":2,"bravo":true}})",
        "HTTP request JSON changed prompt-bearing object member order");
}

int test_schema_number_precision() {
    int failures = 0;
    const std::vector<std::pair<std::string, std::string>> cases{
        {R"({"response_format":{"type":"json_schema","json_schema":{"name":"x","schema":{"type":"number","minimum":0.10000000000000001}}}})",
         "response_format.json_schema.schema/minimum"},
        {R"({"text":{"format":{"type":"json_schema","name":"x","schema":{"type":"number","minimum":1e-400}}}})",
         "text.format.schema/minimum"},
        {R"({"output_config":{"format":{"type":"json_schema","schema":{"type":"number","maximum":0.10000000000000001}}}})",
         "output_config.format.schema/maximum"},
        {R"({"tools":[{"type":"function","function":{"name":"x","strict":true,"parameters":{"type":"object","properties":{"x":{"minimum":0.10000000000000001}}}}}]})",
         "tools/0/function/parameters/properties/x/minimum"},
        {R"({"tools":[{"type":"namespace","name":"n","tools":[{"type":"function","name":"x","strict":true,"parameters":{"type":"object","properties":{"x":{"minimum":0.10000000000000001}}}}]}]})",
         "tools/0/tools/0/parameters/properties/x/minimum"},
        {R"({"tools":[{"name":"x","strict":true,"input_schema":{"type":"object","properties":{"x":{"minimum":0.10000000000000001}}}}]})",
         "tools/0/input_schema/properties/x/minimum"},
        {R"({"temperature":0.10000000000000001,"tools":[{"name":"x","strict":false,"input_schema":{"minimum":0.10000000000000001}}]})",
         ""},
        {R"({"response_format":{"json_schema":{"schema":{"type":"number","default":0.10000000000000001,"minimum":0.1}}}})",
         ""},
    };
    for (const auto& [body, expected] : cases) {
        httplib::Request request;
        request.body = body;
        try {
            (void)infernix::serve::parse_json_body(request);
            failures += check(expected.empty(), "HTTP rounded a schema number before validation");
        } catch (const infernix::serve::ApiException& error) {
            failures +=
                check(!expected.empty() && error.error().code == "unsupported_json_schema" &&
                          error.error().param == expected,
                      "schema precision error lost its protocol location");
        }
    }
    return failures;
}

#if defined(__linux__) || defined(_WIN32)
#    if defined(_WIN32)
using NativeSocket               = SOCKET;
using SocketLength               = int;
constexpr NativeSocket kNoSocket = INVALID_SOCKET;
void close_socket(NativeSocket socket) { ::closesocket(socket); }
#    else
using NativeSocket               = int;
using SocketLength               = socklen_t;
constexpr NativeSocket kNoSocket = -1;
void close_socket(NativeSocket socket) { ::close(socket); }
#    endif

class Socket final {
public:
    explicit Socket(NativeSocket descriptor = kNoSocket) : descriptor_(descriptor) {}

    ~Socket() {
        if (valid()) { close_socket(descriptor_); }
    }

    Socket(const Socket&)            = delete;
    Socket& operator=(const Socket&) = delete;

    [[nodiscard]] NativeSocket get() const noexcept { return descriptor_; }
    [[nodiscard]] bool valid() const noexcept { return descriptor_ != kNoSocket; }

private:
    NativeSocket descriptor_ = kNoSocket;
};

template <class T>
bool socket_option_equals(NativeSocket socket, int level, int option, T expected) {
    T actual{};
    SocketLength size = sizeof(actual);
    return ::getsockopt(socket, level, option, reinterpret_cast<char*>(&actual), &size) == 0 &&
           size == static_cast<SocketLength>(sizeof(actual)) && actual == expected;
}

// A boolean socket option, whichever width the platform reports it in.
bool socket_flag(NativeSocket socket, int level, int option) {
    int actual        = 0;
    SocketLength size = sizeof(actual);
    if (::getsockopt(socket, level, option, reinterpret_cast<char*>(&actual), &size) != 0) { return false; }
    return size == 1 ? (actual & 0xFF) != 0 : actual != 0;
}

// A listening IPv4 socket on `address`:`port` (0: any free port), configured as the HTTP listener or
// left with no options (another program's); kNoSocket when it cannot bind.
NativeSocket listen_on(std::uint32_t address, std::uint16_t port, bool configured) {
    const NativeSocket socket = ::socket(AF_INET, SOCK_STREAM, 0);
    if (socket == kNoSocket) { return kNoSocket; }
    if (configured) { infernix::serve::configure_http_server_socket(socket); }
    sockaddr_in at{};
    at.sin_family      = AF_INET;
    at.sin_addr.s_addr = htonl(address);
    at.sin_port        = htons(port);
    if (::bind(socket, reinterpret_cast<const sockaddr*>(&at), sizeof(at)) != 0 || ::listen(socket, 32) != 0) {
        close_socket(socket);
        return kNoSocket;
    }
    return socket;
}

std::uint16_t port_of(NativeSocket socket) {
    sockaddr_in at{};
    SocketLength size = sizeof(at);
    return ::getsockname(socket, reinterpret_cast<sockaddr*>(&at), &size) == 0 ? ntohs(at.sin_port) : 0;
}

// The HTTP listener's port is its own (candidates SR-09): a second listener with the same options
// cannot bind it, nor a wildcard listener beside it or beside another program's loopback listener (the
// Strata #4255 shape, where local clients reached the other program); and a restart right after the
// listener served connections and closed them first (their TIME_WAIT on its side) binds at once.
int test_exclusive_port() {
    int failures = 0;
#    if defined(_WIN32)
    WSADATA winsock{};
    if (::WSAStartup(MAKEWORD(2, 2), &winsock) != 0) { return check(false, "failed to start Winsock for the port test"); }
    struct WinsockCleanup {
        ~WinsockCleanup() { ::WSACleanup(); }
    } cleanup;
#    endif
    std::uint16_t port = 0;
    {
        Socket first(listen_on(INADDR_LOOPBACK, 0, true));
        if (!first.valid() || (port = port_of(first.get())) == 0) {
            return check(false, "failed to bind the HTTP listener for the port test");
        }
        {
            const Socket second(listen_on(INADDR_LOOPBACK, port, true));
            failures += check(!second.valid(), "a second HTTP listener bound a port in use");
        }
        {
            const Socket wildcard(listen_on(INADDR_ANY, port, true));
            failures += check(!wildcard.valid(), "a wildcard HTTP listener bound beside a loopback one");
        }
        // infernix-serve's startup message: the probe finds the bind's error, and says what to do.
        const int error = infernix::serve::probe_bind_error("127.0.0.1", port);
        failures += check(error != 0 && infernix::serve::bind_failure_reason(error).find("--port") != std::string::npos,
                          "the bind probe did not find why a port in use cannot be bound");
        sockaddr_in at{};
        at.sin_family      = AF_INET;
        at.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        at.sin_port        = htons(port);
        for (int connection = 0; connection < 32; ++connection) {
            const Socket client(::socket(AF_INET, SOCK_STREAM, 0));
            if (!client.valid() ||
                ::connect(client.get(), reinterpret_cast<const sockaddr*>(&at), sizeof(at)) != 0) {
                return failures + check(false, "failed to connect to the HTTP listener for the port test");
            }
            const Socket accepted(::accept(first.get(), nullptr, nullptr)); // closed first, below
            failures += check(accepted.valid(), "failed to accept a connection for the port test");
        }
    }
    {
        const Socket restarted(listen_on(INADDR_LOOPBACK, port, true));
        failures += check(restarted.valid(), "a restart could not bind the port right after it served connections");
    }
    const Socket foreign(listen_on(INADDR_LOOPBACK, 0, false));
    const Socket hijack(foreign.valid() ? listen_on(INADDR_ANY, port_of(foreign.get()), true) : kNoSocket);
    failures += check(foreign.valid() && !hijack.valid(),
                      "a wildcard HTTP listener bound beside another program's loopback listener");
    return failures;
}

int test_inherited_socket_liveness() {
    int failures = 0;
#    if defined(_WIN32)
    WSADATA winsock{};
    if (::WSAStartup(MAKEWORD(2, 2), &winsock) != 0) {
        return check(false, "failed to start Winsock for the liveness test");
    }
    struct WinsockCleanup {
        ~WinsockCleanup() { ::WSACleanup(); }
    } cleanup;
#    endif
    Socket listener(::socket(AF_INET, SOCK_STREAM, 0));
    if (!listener.valid()) { return check(false, "failed to create HTTP listener test socket"); }
    infernix::serve::configure_http_server_socket(listener.get());

    sockaddr_in address{};
    address.sin_family      = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port        = 0;
    if (::bind(listener.get(), reinterpret_cast<const sockaddr*>(&address), sizeof(address)) != 0 ||
        ::listen(listener.get(), 1) != 0) {
        return check(false, "failed to bind HTTP listener test socket");
    }
    SocketLength address_size = sizeof(address);
    if (::getsockname(listener.get(), reinterpret_cast<sockaddr*>(&address), &address_size) != 0) {
        return check(false, "failed to inspect HTTP listener test address");
    }

    Socket client(::socket(AF_INET, SOCK_STREAM, 0));
    if (!client.valid() || ::connect(client.get(), reinterpret_cast<const sockaddr*>(&address),
                                     sizeof(address)) != 0) {
        return check(false, "failed to connect HTTP liveness test socket");
    }
    Socket accepted(::accept(listener.get(), nullptr, nullptr));
    if (!accepted.valid()) { return check(false, "failed to accept HTTP liveness test socket"); }

#    if defined(_WIN32)
    // The port is the listener's alone: SO_EXCLUSIVEADDRUSE, never SO_REUSEADDR.
    failures += check(socket_flag(listener.get(), SOL_SOCKET, SO_EXCLUSIVEADDRUSE) &&
                          !socket_flag(listener.get(), SOL_SOCKET, SO_REUSEADDR),
                      "HTTP listener lacks SO_EXCLUSIVEADDRUSE or has SO_REUSEADDR");
    // Windows reports SO_KEEPALIVE as a one-byte BOOLEAN and TCP_MAXRTMS stands for
    // TCP_USER_TIMEOUT.
    failures += check(socket_option_equals<char>(accepted.get(), SOL_SOCKET, SO_KEEPALIVE, 1),
                      "accepted HTTP socket did not inherit SO_KEEPALIVE");
    failures += check(socket_option_equals<DWORD>(accepted.get(), IPPROTO_TCP, TCP_KEEPIDLE, 10),
                      "accepted HTTP socket did not inherit TCP_KEEPIDLE");
    failures += check(socket_option_equals<DWORD>(accepted.get(), IPPROTO_TCP, TCP_KEEPINTVL, 3),
                      "accepted HTTP socket did not inherit TCP_KEEPINTVL");
    failures += check(socket_option_equals<DWORD>(accepted.get(), IPPROTO_TCP, TCP_KEEPCNT, 3),
                      "accepted HTTP socket did not inherit TCP_KEEPCNT");
    failures += check(
        socket_option_equals<DWORD>(accepted.get(), IPPROTO_TCP, TCP_MAXRTMS, 15000),
        "accepted HTTP socket did not inherit TCP_MAXRTMS");
#    else
#        if defined(__linux__)
    // SO_REUSEADDR binds over an earlier server's TIME_WAIT remnants; SO_REUSEPORT would let a
    // second listener share the port.
    failures += check(socket_flag(listener.get(), SOL_SOCKET, SO_REUSEADDR) &&
                          !socket_flag(listener.get(), SOL_SOCKET, SO_REUSEPORT),
                      "HTTP listener lacks SO_REUSEADDR or has SO_REUSEPORT");
#        endif
    failures += check(socket_option_equals(accepted.get(), SOL_SOCKET, SO_KEEPALIVE, 1),
                      "accepted HTTP socket did not inherit SO_KEEPALIVE");
    failures += check(socket_option_equals(accepted.get(), IPPROTO_TCP, TCP_KEEPIDLE, 10),
                      "accepted HTTP socket did not inherit TCP_KEEPIDLE");
    failures += check(socket_option_equals(accepted.get(), IPPROTO_TCP, TCP_KEEPINTVL, 3),
                      "accepted HTTP socket did not inherit TCP_KEEPINTVL");
    failures += check(socket_option_equals(accepted.get(), IPPROTO_TCP, TCP_KEEPCNT, 3),
                      "accepted HTTP socket did not inherit TCP_KEEPCNT");
    failures += check(
        socket_option_equals<unsigned int>(accepted.get(), IPPROTO_TCP, TCP_USER_TIMEOUT, 15000U),
        "accepted HTTP socket did not inherit TCP_USER_TIMEOUT");
#    endif
    return failures;
}
#endif

} // namespace

int main() {
    int failures = test_ngram_identity() + test_request_label() + test_ngram_generation() + test_sse_transport() +
                   test_sse_response_headers() + test_prompt_json_member_order() +
                   test_schema_number_precision();
#if defined(__linux__) || defined(_WIN32)
    failures += test_inherited_socket_liveness();
    failures += test_exclusive_port();
    const std::string in_use = infernix::serve::bind_failure_reason(
#    if defined(_WIN32)
        WSAEADDRINUSE
#    else
        EADDRINUSE
#    endif
    );
    failures += check(in_use.find("in use") != std::string::npos && in_use.find("--port") != std::string::npos,
                      "a bind failure on a port in use does not say so");
#endif
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
