#include "serve/keep_alive_retirement.h"

#include <utility>

namespace infernix::serve {
namespace {

// One entry per pooled connection, keyed by its peer.
std::string connection_key(const httplib::Request& request) {
    return request.remote_addr + ':' + std::to_string(request.remote_port);
}

} // namespace

KeepAliveRetirement::KeepAliveRetirement(std::chrono::steady_clock::duration idle_threshold)
    : idle_threshold_(idle_threshold) {}

void KeepAliveRetirement::arrived(const httplib::Request& request) {
    const auto now = std::chrono::steady_clock::now();
    std::lock_guard lock(mutex_);
    const auto [entry, first] = connections_.try_emplace(connection_key(request), Connection{.finished = now});
    entry->second.retire      = !first && now - entry->second.finished >= idle_threshold_;
}

void KeepAliveRetirement::respond(const httplib::Request& request, httplib::Response& response) {
    std::string key = connection_key(request);
    {
        std::lock_guard lock(mutex_);
        const auto entry = connections_.find(key);
        if (entry != connections_.end() && entry->second.retire) {
            response.set_header("Connection", "close");
            response.headers.erase("Keep-Alive");
        }
    }
    // The server destroys the Response once its whole body has been written (a stream's
    // last chunk included), calling its resource releaser then; chaining onto that
    // releaser (a public member, though cpp-httplib files it with its private ones)
    // marks the end of every response, streamed or not, without patching the server.
    response.content_provider_resource_releaser_ =
        [this, key = std::move(key), releaser = std::move(response.content_provider_resource_releaser_)](bool success) {
            if (releaser) { releaser(success); }
            finished(key);
        };
}

void KeepAliveRetirement::finished(const std::string& key) {
    const auto now = std::chrono::steady_clock::now();
    std::lock_guard lock(mutex_);
    const auto entry = connections_.find(key);
    if (entry == connections_.end()) { return; }
    if (entry->second.retire) {
        // The connection's lifetime is over. A later peer that reuses this address and
        // port must start fresh, not inherit this connection's idle time.
        connections_.erase(entry);
        return;
    }
    entry->second.finished = now;
    if (connections_.size() >= kMaxTrackedConnections) { prune_stale_locked(now); }
}

void KeepAliveRetirement::prune_stale_locked(std::chrono::steady_clock::time_point now) {
    // An entry idle past the threshold names a connection that is either closed or one
    // request away from retiring; dropping it costs at most that one retirement. The
    // map holds only connections seen inside the threshold, so it stays bounded without
    // depending on the server's timeout close ever arriving.
    for (auto entry = connections_.begin(); entry != connections_.end();) {
        if (now - entry->second.finished >= idle_threshold_) {
            entry = connections_.erase(entry);
        } else {
            ++entry;
        }
    }
}

} // namespace infernix::serve
