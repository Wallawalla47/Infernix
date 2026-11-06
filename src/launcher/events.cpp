#include "launcher/events.h"

#include <algorithm>

namespace infernix::launcher {

std::optional<std::string> EventHub::Subscription::next(std::chrono::milliseconds timeout) {
    std::unique_lock lock(mutex_);
    cv_.wait_for(lock, timeout, [this] { return closed_ || !queue_.empty(); });
    if (queue_.empty()) { return std::nullopt; }
    std::string event = std::move(queue_.front());
    queue_.pop_front();
    return event;
}

void EventHub::Subscription::push(std::string event) {
    {
        std::lock_guard lock(mutex_);
        if (closed_) { return; }
        queue_.push_back(std::move(event));
        while (queue_.size() > kLimit) { queue_.pop_front(); }
    }
    cv_.notify_one();
}

void EventHub::Subscription::close() {
    {
        std::lock_guard lock(mutex_);
        closed_ = true;
    }
    cv_.notify_all();
}

bool EventHub::Subscription::closed() const {
    std::lock_guard lock(mutex_);
    return closed_;
}

std::shared_ptr<EventHub::Subscription> EventHub::subscribe() {
    auto subscription = std::make_shared<Subscription>();
    std::lock_guard lock(mutex_);
    subscribers_.push_back(subscription);
    return subscription;
}

std::string EventHub::format(const std::string& name, const Json& data) {
    // JSON text never holds a raw newline, so one data line carries the whole object.
    return "event: " + name + "\ndata: " + data.dump(-1, ' ', false, Json::error_handler_t::replace) + "\n\n";
}

std::string EventHub::format_data(const Json& data) {
    return "data: " + data.dump(-1, ' ', false, Json::error_handler_t::replace) + "\n\n";
}

void EventHub::publish(const std::string& name, const Json& data) { publish_text(format(name, data)); }

void EventHub::publish_text(const std::string& event) {
    std::vector<std::shared_ptr<Subscription>> live;
    {
        std::lock_guard lock(mutex_);
        subscribers_.erase(std::remove_if(subscribers_.begin(), subscribers_.end(),
                                          [&](const std::weak_ptr<Subscription>& weak) {
                                              auto strong = weak.lock();
                                              if (!strong || strong->closed()) { return true; }
                                              live.push_back(std::move(strong));
                                              return false;
                                          }),
                           subscribers_.end());
    }
    for (const auto& subscriber : live) { subscriber->push(event); }
}

void EventHub::close_all() {
    std::lock_guard lock(mutex_);
    for (const auto& weak : subscribers_) {
        if (auto strong = weak.lock()) { strong->close(); }
    }
    subscribers_.clear();
}

} // namespace infernix::launcher
