#pragma once

// Loopback HTTP harness for the serving routes without a model: a GenerationBackend whose run() a
// test supplies, an HttpServer of the real options on a free loopback port, and an SSE frame reader.

#include "serve/http_server.h"

#include <nlohmann/json.hpp>
#include <spdlog/logger.h>
#include <spdlog/sinks/null_sink.h>

#include <chrono>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace infernix::serve::test {

// Everything but run(): a model named `model_name` with an 8-token prompt that never continues.
class LoopbackBackend : public GenerationBackend {
public:
    explicit LoopbackBackend(std::string model_name) : model_name_(std::move(model_name)) {}

    const infernix::EngineOptions& engine_options() const override { return engine_options_; }

    infernix::LoadSummary load_summary() const override {
        infernix::LoadSummary summary;
        summary.model_name = model_name_;
        return summary;
    }

    infernix::ModelMetadata model_metadata() const override { return {}; }

    infernix::MemorySummary memory_summary() const override { return {}; }

    infernix::RuntimeStats runtime_stats() const override { return {}; }

    void reset_execution_maxima() noexcept override {}

    bool is_available() const override { return true; }

    bool has_failed() const override { return false; }

    infernix::ModelSamplingDefaults sampling_defaults() const override { return {}; }

    std::vector<infernix::TokenId> tokenize_text(std::string_view) const override { return {}; }

    PreparedRequest prepare(const GenerationRequest&, GenerationConsumerMode,
                            infernix::GenerationObservationOptions,
                            std::function<bool()>) const override {
        PreparedRequest prepared;
        prepared.prompt_tokens = 8;
        return prepared;
    }

    PreparedRequest prepare_continuation(const GenerationRequest&, GenerationConsumerMode,
                                         infernix::GenerationObservationOptions,
                                         std::function<bool()>) const override {
        throw std::logic_error("a single-generation answer continued");
    }

    int count_prompt_tokens(const GenerationRequest&, std::function<bool()>) const override { return 8; }

    void admit(PreparedRequest&, std::function<bool()>) const override {}

    void stop() noexcept override {}

private:
    std::string model_name_;
    infernix::EngineOptions engine_options_;
};

class LoopbackServer {
public:
    explicit LoopbackServer(GenerationBackend& backend,
                            std::chrono::seconds client_timeout = std::chrono::seconds(30)) {
        ServeOptions options;
        options.host                  = "127.0.0.1";
        options.port                  = free_loopback_port();
        options.log_stats_interval_ms = 0;
        options.log_stats_panel       = false;
        auto logger                   = std::make_shared<spdlog::logger>(
            "http-loopback", std::make_shared<spdlog::sinks::null_sink_mt>());
        server_ = std::make_unique<HttpServer>(options, std::move(logger));
        if (options.port <= 0 || !server_->bind()) {
            throw std::runtime_error("the test server could not bind a loopback port");
        }
        port_ = options.port;
        server_->attach(backend);
        listener_ = std::thread([this] { (void)server_->listen(); });
        client_   = std::make_unique<httplib::Client>("127.0.0.1", port_);
        client_->set_connection_timeout(client_timeout);
        client_->set_read_timeout(client_timeout);
        client_->set_keep_alive(true);
    }

    ~LoopbackServer() {
        server_->stop();
        if (listener_.joinable()) { listener_.join(); }
    }

    LoopbackServer(const LoopbackServer&)            = delete;
    LoopbackServer& operator=(const LoopbackServer&) = delete;

    [[nodiscard]] int port() const noexcept { return port_; }

    [[nodiscard]] httplib::Result post(std::string_view path, const nlohmann::json& body) const {
        return client_->Post(std::string(path), body.dump(), "application/json");
    }

private:
    static int free_loopback_port() {
        httplib::Server probe;
        const int port = probe.bind_to_any_port("127.0.0.1");
        probe.stop();
        return port;
    }

    std::unique_ptr<HttpServer> server_;
    int port_ = 0;
    std::thread listener_;
    std::unique_ptr<httplib::Client> client_;
};

// One SSE event: its `event:` name (empty when it has none) and its `data:` payload.
struct SseFrame {
    std::string event;
    std::string data;
};

// The events of an SSE body; comments (a heartbeat) are skipped.
inline std::vector<SseFrame> sse_frames(const std::string& body) {
    std::vector<SseFrame> frames;
    for (std::size_t begin = 0; begin < body.size();) {
        const std::size_t end = body.find("\n\n", begin);
        if (end == std::string::npos) { throw std::runtime_error("unterminated SSE event"); }
        const std::string text = body.substr(begin, end - begin);
        begin                  = end + 2;
        if (text.starts_with(":")) { continue; }
        SseFrame frame;
        for (std::size_t line_begin = 0; line_begin <= text.size();) {
            std::size_t line_end = text.find('\n', line_begin);
            if (line_end == std::string::npos) { line_end = text.size(); }
            const std::string line = text.substr(line_begin, line_end - line_begin);
            if (line.starts_with("event: ")) { frame.event = line.substr(7); }
            if (line.starts_with("data: ")) { frame.data = line.substr(6); }
            line_begin = line_end + 1;
        }
        if (frame.data.empty()) { throw std::runtime_error("an SSE event without data"); }
        frames.push_back(std::move(frame));
    }
    return frames;
}

} // namespace infernix::serve::test
