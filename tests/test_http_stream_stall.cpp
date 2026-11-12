#include "http_loopback_harness.h"

#ifdef _WIN32
#    include <winsock2.h>
#    include <ws2tcpip.h>
#else
#    include <arpa/inet.h>
#    include <netinet/in.h>
#    include <sys/socket.h>
#    include <sys/time.h>
#    include <unistd.h>
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// YH-01: a client that stops reading a stream for longer than cpp-httplib's 5-second default write
// timeout keeps its stream (infernix-serve waits up to kWriteTimeoutSec, 60 s, for it). Over loopback
// HTTP without a model: a backend streams 16 MiB of Chat Completions deltas, polling the route's
// liveness check between them as the Engine does, while a raw-socket client with a 4 KiB receive
// buffer reads the start, stops for 7 s, then reads to the end. The stream must carry every delta in
// order, the finish chunk and [DONE]. The backend times its delta writes: one of them must have
// waited past 5 s, so the test shows the failure path really ran (the old default would have
// cancelled that write) rather than the operating system's buffers absorbing the stream.
namespace {

using Json  = nlohmann::json;
using Clock = std::chrono::steady_clock;
using namespace infernix::serve;

constexpr const char* kModel           = "stall-test-model";
constexpr int kDeltas                  = 2048;
constexpr std::size_t kDeltaBytes      = 8192;
constexpr std::size_t kReadBeforePause = 32 * 1024;
constexpr std::chrono::seconds kPause{7};
constexpr std::chrono::milliseconds kOldDefaultWriteTimeout{5000};

int check(bool condition, const std::string& message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

std::string delta_text(int index) {
    char head[16];
    std::snprintf(head, sizeof(head), "<%06d>", index);
    std::string text(head);
    text.resize(kDeltaBytes, static_cast<char>('a' + index % 26));
    return text;
}

class StallBackend final : public test::LoopbackBackend {
public:
    StallBackend() : LoopbackBackend(kModel) {}

    GenerationOutcome run(PreparedRequest& prepared, const StreamSink* sink,
                          std::function<bool()>) override {
        GenerationOutcome outcome;
        outcome.prompt_tokens           = prepared.prompt_tokens;
        outcome.computed_prefill_tokens = static_cast<std::uint32_t>(prepared.prompt_tokens);
        outcome.finish_reason           = infernix::FinishReason::StopToken;
        if (sink != nullptr && sink->on_start) {
            sink->on_start(infernix::GenerationStart{
                .prompt = infernix::PromptSummary{.prompt_tokens = static_cast<std::uint32_t>(
                                                      prepared.prompt_tokens)}});
        }
        for (int index = 0; index < kDeltas; ++index) {
            if (sink != nullptr && sink->is_cancelled && sink->is_cancelled()) {
                outcome.finish_reason = infernix::FinishReason::Cancelled;
                cancelled_            = true;
                return outcome;
            }
            const std::string text = delta_text(index);
            const Clock::time_point start = Clock::now();
            if (sink != nullptr && sink->on_content) { sink->on_content(text); }
            const Clock::duration took = Clock::now() - start;
            {
                std::lock_guard lock(mutex_);
                longest_write_ = std::max(longest_write_, took);
            }
            outcome.text += text;
            ++outcome.completion_tokens;
        }
        finished_ = true;
        return outcome;
    }

    [[nodiscard]] Clock::duration longest_write() const {
        std::lock_guard lock(mutex_);
        return longest_write_;
    }

    [[nodiscard]] bool finished() const noexcept { return finished_.load(); }

    [[nodiscard]] bool cancelled() const noexcept { return cancelled_.load(); }

private:
    mutable std::mutex mutex_;
    Clock::duration longest_write_{};
    std::atomic<bool> finished_{false};
    std::atomic<bool> cancelled_{false};
};

#ifdef _WIN32
using Socket                    = SOCKET;
constexpr Socket kInvalidSocket = INVALID_SOCKET;
void close_socket(Socket socket) { ::closesocket(socket); }
#else
using Socket                    = int;
constexpr Socket kInvalidSocket = -1;
void close_socket(Socket socket) { ::close(socket); }
#endif

// A client socket to 127.0.0.1:`port` with a small receive buffer (set before connecting, so the
// window it advertises stays small) and a receive timeout that bounds a hung stream.
Socket connect_small_window(int port) {
    const Socket socket = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (socket == kInvalidSocket) { throw std::runtime_error("socket() failed"); }
    const int receive_buffer = 4096;
    ::setsockopt(socket, SOL_SOCKET, SO_RCVBUF, reinterpret_cast<const char*>(&receive_buffer),
                 sizeof(receive_buffer));
#ifdef _WIN32
    const DWORD timeout_ms = 30000;
    ::setsockopt(socket, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char*>(&timeout_ms),
                 sizeof(timeout_ms));
#else
    const timeval timeout{.tv_sec = 30, .tv_usec = 0};
    ::setsockopt(socket, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
#endif
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port   = htons(static_cast<std::uint16_t>(port));
    ::inet_pton(AF_INET, "127.0.0.1", &address.sin_addr);
    if (::connect(socket, reinterpret_cast<const sockaddr*>(&address), sizeof(address)) != 0) {
        close_socket(socket);
        throw std::runtime_error("connect() failed");
    }
    return socket;
}

void send_all(Socket socket, const std::string& data) {
    for (std::size_t sent = 0; sent < data.size();) {
        const auto count = ::send(socket, data.data() + sent, static_cast<int>(data.size() - sent), 0);
        if (count <= 0) { throw std::runtime_error("send() failed"); }
        sent += static_cast<std::size_t>(count);
    }
}

// Appends what the socket returns until `stop_at` bytes are held or the peer closes; false on an
// error or the receive timeout.
bool receive(Socket socket, std::string& into, std::size_t stop_at) {
    char buffer[4096];
    while (into.size() < stop_at) {
        const auto count = ::recv(socket, buffer, static_cast<int>(sizeof(buffer)), 0);
        if (count == 0) { return true; }
        if (count < 0) { return false; }
        into.append(buffer, static_cast<std::size_t>(count));
    }
    return true;
}

// The body of a chunked HTTP/1.1 response.
std::string dechunk(const std::string& raw) {
    std::string body;
    for (std::size_t at = 0;;) {
        const std::size_t line_end = raw.find("\r\n", at);
        if (line_end == std::string::npos) { throw std::runtime_error("truncated chunk header"); }
        const std::size_t size = std::stoul(raw.substr(at, line_end - at), nullptr, 16);
        at                     = line_end + 2;
        if (size == 0) { return body; }
        if (at + size + 2 > raw.size()) { throw std::runtime_error("truncated chunk"); }
        body.append(raw, at, size);
        at += size + 2;
    }
}

} // namespace

int main() try {
    StallBackend backend;
    const test::LoopbackServer server(backend);

    const std::string body = Json{{"model", kModel},
                                  {"stream", true},
                                  {"messages", Json::array({Json{{"role", "user"}, {"content", "hi"}}})}}
                                 .dump();
    const std::string request = "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                                "Content-Type: application/json\r\nConnection: close\r\n"
                                "Content-Length: " +
                                std::to_string(body.size()) + "\r\n\r\n" + body;

    const Clock::time_point start = Clock::now();
    const Socket socket           = connect_small_window(server.port());
    std::string raw;
    bool read_ok = false;
    try {
        send_all(socket, request);
        read_ok = receive(socket, raw, kReadBeforePause);
        // The client stops reading; the server fills the socket buffers and its write waits.
        std::this_thread::sleep_for(kPause);
        read_ok = read_ok && receive(socket, raw, std::string::npos);
    } catch (...) {
        close_socket(socket);
        throw;
    }
    close_socket(socket);
    const double seconds = std::chrono::duration<double>(Clock::now() - start).count();
    const double longest = std::chrono::duration<double>(backend.longest_write()).count();
    std::cout << "stall test: " << raw.size() << " bytes in " << seconds << " s, longest delta write "
              << longest << " s\n";

    int failures = check(read_ok, "the stream's read failed or timed out");
    failures += check(raw.starts_with("HTTP/1.1 200"), "the answer is not a 200");
    failures += check(!backend.cancelled() && backend.finished(),
                      "the generation was cancelled while the client paused");
    failures += check(std::chrono::duration<double>(backend.longest_write()) > kOldDefaultWriteTimeout,
                      "no delta write waited past 5 s: the socket buffers absorbed the stream, so the test "
                      "did not reach the write-timeout path (longest " +
                          std::to_string(longest) + " s)");

    const std::size_t header_end = raw.find("\r\n\r\n");
    std::vector<test::SseFrame> frames;
    try {
        if (header_end == std::string::npos ||
            raw.substr(0, header_end).find("Transfer-Encoding: chunked") == std::string::npos) {
            throw std::runtime_error("not a chunked response");
        }
        frames = test::sse_frames(dechunk(raw.substr(header_end + 4)));
    } catch (const std::exception& error) {
        return failures + check(false, std::string("the stream is incomplete: ") + error.what());
    }
    int deltas = 0;
    bool in_order = true;
    std::string finish;
    for (std::size_t index = 0; index + 1 < frames.size(); ++index) {
        const Json chunk = Json::parse(frames[index].data);
        for (const Json& choice : chunk.value("choices", Json::array())) {
            const std::string content = choice.value("delta", Json::object()).value("content", "");
            if (!content.empty()) {
                in_order = in_order && content == delta_text(deltas);
                ++deltas;
            }
            if (choice.contains("finish_reason") && choice["finish_reason"].is_string()) {
                finish = choice["finish_reason"];
            }
        }
    }
    failures += check(deltas == kDeltas && in_order,
                      "the stream lost or reordered deltas: " + std::to_string(deltas) + " of " +
                          std::to_string(kDeltas));
    failures += check(finish == "stop", "no finish_reason stop, got '" + finish + "'");
    failures += check(!frames.empty() && frames.back().data == "[DONE]", "the stream does not end with [DONE]");
    if (failures == 0) { std::cout << "stream stall test passed\n"; }
    return failures == 0 ? 0 : 1;
} catch (const std::exception& error) {
    std::cerr << "FAIL: uncaught exception: " << error.what() << '\n';
    return 1;
}
