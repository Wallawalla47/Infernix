#include "serve/http_transport.h"

#include "serve/request_validation.h"
#include "text/json_input.h"

#if defined(__linux__)
#    include <netinet/tcp.h>
#    include <sys/socket.h>
#elif defined(_WIN32)
#    include <winsock2.h>
#    include <ws2tcpip.h>
#endif

#include <algorithm>
#include <charconv>
#include <stdexcept>
#include <utility>

namespace infernix::serve {
namespace {

bool has_ngram_generation(const NgramArchiveStats& stats) noexcept {
    return stats.enabled && stats.bound && stats.published && stats.generation != 0;
}

#if defined(__linux__) || defined(_WIN32)
constexpr int kKeepAliveIdleSeconds                = 10;
constexpr int kKeepAliveIntervalSeconds            = 3;
constexpr int kKeepAliveProbeCount                 = 3;
constexpr unsigned int kTcpUserTimeoutMilliseconds = 15000;

template <class T>
void set_socket_option(socket_t socket, int level, int option, const T& value) noexcept {
    (void)::setsockopt(socket, level, option, reinterpret_cast<const char*>(&value),
                       static_cast<int>(sizeof(value)));
}
#endif

} // namespace

void set_ngram_generation_header(httplib::Response& response, const NgramArchiveStats& stats) {
    if (has_ngram_generation(stats)) {
        response.set_header("X-Infernix-Draft-Generation", std::to_string(stats.generation));
    }
}

std::string ngram_generation_comment(const NgramArchiveStats& stats) {
    if (!has_ngram_generation(stats)) { return {}; }
    return ": infernix-draft-generation: " + std::to_string(stats.generation) + "\n\n";
}

RequestJson parse_json_body(const httplib::Request& request) {
    text::ParsedJsonNumbers parsed;
    try {
        parsed = text::parse_json_numbers(request.body);
    } catch (const std::exception&) { bad_request("request body is not valid JSON"); }
    validate_schema_number_input(parsed);
    return std::move(parsed.value);
}

NgramSessionHints resolve_ngram_session(const httplib::Request& request, const RequestJson& body,
                                        const ServeOptions& options) {
    if (options.speculative.ngram_archive_bytes == 0) { return {}; }
    auto valid = [](std::string_view id) {
        // Leaves room for the identity namespace; the archive also bounds the complete key.
        return !id.empty() && id.size() <= 240 &&
               std::none_of(id.begin(), id.end(),
                            [](unsigned char c) { return c <= 32 || c == 127; });
    };
    auto field = [](const RequestJson& object, const char* key) -> std::string_view {
        const auto it = object.find(key);
        return it != object.end() && it->is_string() ? it->get_ref<const std::string&>()
                                                     : std::string_view{};
    };
    NgramSessionHints hints;
    if (request.has_header("x-infernix-draft-session")) {
        const auto key = request.get_header_value("x-infernix-draft-session");
        if (request.get_header_value_count("x-infernix-draft-session") != 1 || !valid(key)) {
            return {};
        }
        hints.key = "explicit:" + key;
        if (request.has_header("x-infernix-draft-reset")) {
            if (request.get_header_value_count("x-infernix-draft-reset") != 1 ||
                request.get_header_value("x-infernix-draft-reset") != "1") {
                return {};
            }
            hints.reset = true;
        }
        const bool parent     = request.has_header("x-infernix-draft-parent");
        const bool generation = request.has_header("x-infernix-draft-generation");
        if (parent != generation || (parent && hints.reset)) { return {}; }
        if (parent) {
            const auto name   = request.get_header_value("x-infernix-draft-parent");
            const auto number = request.get_header_value("x-infernix-draft-generation");
            if (!valid(name) || request.get_header_value_count("x-infernix-draft-parent") != 1 ||
                request.get_header_value_count("x-infernix-draft-generation") != 1) {
                return {};
            }
            const auto [end, error] = std::from_chars(number.data(), number.data() + number.size(),
                                                      hints.parent_generation);
            if (error != std::errc{} || end != number.data() + number.size() ||
                hints.parent_generation == 0) {
                return {};
            }
            hints.parent = "explicit:" + name;
        }
        return hints;
    }
    // Separate local opt-in includes store:false. Only verified conversation IDs
    // are recognized; an API key, cache prefix or device ID is not an identity.
    if (!options.ngram_native_sessions) { return {}; }
    bool ambiguous = false;
    auto select    = [&](std::string_view prefix, std::string_view id) {
        if (!valid(id) || !hints.key.empty()) {
            ambiguous = true;
            return;
        }
        hints.key = std::string(prefix) + std::string(id);
    };
    if (request.has_header("x-session-affinity")) {
        if (request.get_header_value_count("x-session-affinity") != 1) { return {}; }
        select("kilo:", request.get_header_value("x-session-affinity"));
    }
    if (const auto metadata = body.find("client_metadata");
        metadata != body.end() && metadata->is_object()) {
        const auto thread  = field(*metadata, "thread_id");
        const auto session = field(*metadata, "session_id");
        if (!thread.empty()) {
            if (!session.empty() && thread != session) { return {}; }
            select("codex:", thread);
        }
    }
    if (const auto metadata = body.find("metadata");
        metadata != body.end() && metadata->is_object()) {
        const auto user = field(*metadata, "user_id");
        if (!user.empty() && user.size() <= 4096) {
            const auto parsed = RequestJson::parse(user, nullptr, false);
            if (parsed.is_object()) {
                const auto session = field(parsed, "session_id");
                if (!session.empty()) { select("claude:", session); }
            }
        }
    }
    if (request.has_header("x-infernix-draft-parent") ||
        request.has_header("x-infernix-draft-generation")) {
        return {};
    }
    if (request.has_header("x-infernix-draft-reset")) {
        if (request.get_header_value_count("x-infernix-draft-reset") != 1 ||
            request.get_header_value("x-infernix-draft-reset") != "1") {
            return {};
        }
        hints.reset = true;
    }
    return ambiguous ? NgramSessionHints{} : hints;
}

bool client_disconnected(const httplib::Request& request) { return request.is_connection_closed(); }

void prepare_sse_response(httplib::Response& response) {
    response.set_header("Cache-Control", "no-cache");
    response.set_header("X-Accel-Buffering", "no");
}

SseTransport::SseTransport(httplib::DataSink& sink, std::atomic<bool>& cancelled,
                           Clock::duration heartbeat_interval, Clock::time_point now)
    : sink_(sink), cancelled_(cancelled), heartbeat_interval_(heartbeat_interval),
      last_write_(now) {
    if (heartbeat_interval_ <= Clock::duration::zero()) {
        throw std::invalid_argument("SSE heartbeat interval must be positive");
    }
}

bool SseTransport::mark_cancelled() noexcept {
    cancelled_.store(true, std::memory_order_release);
    return true;
}

void SseTransport::write(std::string_view item, Clock::time_point now) {
    if (cancelled_.load(std::memory_order_acquire) || !sink_.write(item.data(), item.size())) {
        mark_cancelled();
        throw ClientDisconnected();
    }
    last_write_ = now;
}

void SseTransport::write(const std::vector<std::string>& items, Clock::time_point now) {
    for (const std::string& item : items) { write(item, now); }
}

bool SseTransport::poll(Clock::time_point now) {
    if (cancelled_.load(std::memory_order_acquire)) { return true; }
    if (sink_.is_writable && !sink_.is_writable()) { return mark_cancelled(); }
    if (now - last_write_ < heartbeat_interval_) { return false; }
    if (!sink_.write(kHeartbeatComment.data(), kHeartbeatComment.size())) {
        return mark_cancelled();
    }
    last_write_ = now;
    return false;
}

void configure_http_server_socket(socket_t socket) noexcept {
    httplib::default_socket_options(socket);
#if defined(__linux__)
    const int enabled = 1;
    // cpp-httplib sets only SO_REUSEPORT. The listener's options pass to the connections it
    // accepts, and Linux binds over their TIME_WAIT remnants only when both sides share a reuse
    // option, so another server that sets just SO_REUSEADDR (llama.cpp, vLLM, uvicorn) could not
    // bind this port for about a minute after Infernix stops, nor Infernix after it (upstream
    // #341). With both set, either kind binds at once.
    set_socket_option(socket, SOL_SOCKET, SO_REUSEADDR, enabled);
    set_socket_option(socket, SOL_SOCKET, SO_KEEPALIVE, enabled);
    set_socket_option(socket, IPPROTO_TCP, TCP_KEEPIDLE, kKeepAliveIdleSeconds);
    set_socket_option(socket, IPPROTO_TCP, TCP_KEEPINTVL, kKeepAliveIntervalSeconds);
    set_socket_option(socket, IPPROTO_TCP, TCP_KEEPCNT, kKeepAliveProbeCount);
    set_socket_option(socket, IPPROTO_TCP, TCP_USER_TIMEOUT, kTcpUserTimeoutMilliseconds);
#elif defined(_WIN32)
    // The same liveness on Windows: keepalive probes find a silently dead peer, and TCP_MAXRTMS
    // (Windows' TCP_USER_TIMEOUT) bounds how long unacknowledged stream writes are retried, so
    // the SSE liveness gate sees a failed write instead of the retransmission window.
    const DWORD enabled = 1;
    set_socket_option(socket, SOL_SOCKET, SO_KEEPALIVE, enabled);
    set_socket_option(socket, IPPROTO_TCP, TCP_KEEPIDLE, static_cast<DWORD>(kKeepAliveIdleSeconds));
    set_socket_option(socket, IPPROTO_TCP, TCP_KEEPINTVL,
                      static_cast<DWORD>(kKeepAliveIntervalSeconds));
    set_socket_option(socket, IPPROTO_TCP, TCP_KEEPCNT, static_cast<DWORD>(kKeepAliveProbeCount));
    set_socket_option(socket, IPPROTO_TCP, TCP_MAXRTMS,
                      static_cast<DWORD>(kTcpUserTimeoutMilliseconds));
#endif
}

void set_owned_json_content(httplib::Response& response, std::string body,
                            std::shared_ptr<RequestLifetime> lifetime) {
    response.set_content(std::move(body), "application/json");
    response.user_data.set("infernix.request_lifetime", std::move(lifetime));
}

} // namespace infernix::serve
