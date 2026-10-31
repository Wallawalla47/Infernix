#pragma once

// Server-sent events for the web UI: every subscriber gets each published event in order through
// its own bounded queue (the oldest events are dropped for a reader that falls far behind).

#include "launcher/util.h"

#include <chrono>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace infernix::launcher {

class EventHub {
public:
    class Subscription {
    public:
        // The next formatted event (`event: NAME\ndata: JSON\n\n`), or nullopt after `timeout` or
        // once the hub closes.
        [[nodiscard]] std::optional<std::string> next(std::chrono::milliseconds timeout);
        void push(std::string event);
        void close();
        [[nodiscard]] bool closed() const;

    private:
        static constexpr std::size_t kLimit = 2000;
        mutable std::mutex mutex_;
        std::condition_variable cv_;
        std::deque<std::string> queue_;
        bool closed_ = false;
    };

    [[nodiscard]] std::shared_ptr<Subscription> subscribe();
    void publish(const std::string& name, const Json& data);
    // Publishes an event already formatted (format, format_data).
    void publish_text(const std::string& event);
    void close_all();

    [[nodiscard]] static std::string format(const std::string& name, const Json& data);
    // An unnamed event: `data: JSON\n\n` (llama.cpp's /models/sse).
    [[nodiscard]] static std::string format_data(const Json& data);

private:
    std::mutex mutex_;
    std::vector<std::weak_ptr<Subscription>> subscribers_;
};

} // namespace infernix::launcher
