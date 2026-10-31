#include "serve/keep_alive_retirement.h"

#include <utility>

namespace infernix::serve {

KeepAliveRetirement::KeepAliveRetirement(std::chrono::seconds idle_threshold)
    : idle_threshold_(idle_threshold) {}

void KeepAliveRetirement::handle(const httplib::Request& request, httplib::Response& response) {
    const auto now = std::chrono::steady_clock::now();
    // One entry per pooled connection, keyed by its peer.
    std::string key = request.remote_addr + ':' + std::to_string(request.remote_port);
    std::lock_guard lock(mutex_);
    const auto seen = last_finished_.find(key);
    if (seen != last_finished_.end() && now - seen->second >= idle_threshold_) {
        // This connection has been silent past the threshold: hand the client a
        // response it can retire on. The silent timeout close is the one death
        // no response can announce, so retire at the moment an announcement
        // is still possible.
        response.set_header("Connection", "close");
        // The server wrote its own keep-alive hint before this handler runs;
        // a response must not invite reuse and announce the end at once.
        response.headers.erase("Keep-Alive");
        // The connection's lifetime is over. A later peer that reuses this
        // address and port must start fresh, not inherit this stamp time.
        last_finished_.erase(key);
        return;
    }
    if (last_finished_.size() >= kMaxTrackedConnections) { prune_stale_locked(now); }
    last_finished_[std::move(key)] = now;
}

void KeepAliveRetirement::prune_stale_locked(std::chrono::steady_clock::time_point now) {
    // An entry silent past the threshold names a connection that is either closed
    // or one request away from retiring; keeping it buys nothing. The map holds
    // only connections seen inside the threshold, so it stays bounded without
    // depending on the server's timeout close ever arriving.
    for (auto entry = last_finished_.begin(); entry != last_finished_.end();) {
        if (now - entry->second >= idle_threshold_) {
            entry = last_finished_.erase(entry);
        } else {
            ++entry;
        }
    }
}

} // namespace infernix::serve
