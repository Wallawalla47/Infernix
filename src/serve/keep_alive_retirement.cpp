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
            // The library never runs a provider for HEAD, so a HEAD stamp would be a
            // close the server cannot perform — decoration. It falls back to the
            // timeout close instead.
            if (request.method != "HEAD") { close_after_response(response); }
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

void KeepAliveRetirement::close_after_response(httplib::Response& response) {
    response.set_header("Connection", "close");
    // The server wrote its own keep-alive hint before this handler runs;
    // a response must not invite reuse and announce the end at once.
    response.headers.erase("Keep-Alive");

    // The vendored server ends a connection loop only when the response callback
    // reports failure, and its graceful close (shutdown-write, bounded drain, FIN)
    // runs unconditionally behind that loop. A content provider that returns false
    // after the body is the public-API lever producing exactly that: the headers
    // and every body byte are already on the wire, the callback sees a canceled
    // write, and the socket closes without a framing change. serve registers no
    // compressor, so the skipped identity-compressor tail flush costs no bytes.
    auto releaser = std::move(response.content_provider_resource_releaser_);
    if (response.content_provider_) {
        // The stream's own provider owns the body and its framing (chunked or
        // length-framed, done() or full-length writes included). The wrapper only
        // reports cancellation after that provider has taken its own completion
        // path, so every body byte — a stream's terminator included — is already
        // written when the close fires.
        auto original = std::move(response.content_provider_);
        response.content_provider_ =
            [original = std::move(original)](std::size_t offset, std::size_t length,
                                             httplib::DataSink& sink) {
                auto finished = std::make_shared<bool>(false);
                auto done     = std::move(sink.done);
                sink.done     = [finished, done] {
                    *finished = true;
                    done();
                };
                return original(offset, length, sink) && !*finished;
            };
        response.content_provider_resource_releaser_ = std::move(releaser);
        return;
    }
    if (!response.body.empty()) {
        std::string body = std::move(response.body);
        // The body path always carries a Content-Type by the time post-routing
        // runs, and set_content_provider re-stamps the one it is handed, so the
        // header block (including the Content-Length already set for this body)
        // reaches the wire unchanged.
        const std::string content_type = response.get_header_value("Content-Type");
        response.set_content_provider(
            body.size(), content_type,
            [body](std::size_t offset, std::size_t length, httplib::DataSink& sink) {
                sink.write(body.data() + offset, length);
                return false;
            },
            std::move(releaser));
        return;
    }
    // Empty body: the server already framed it with Content-Length: 0. The
    // zero-length known-provider variant is ignored by the library, so the
    // without-length provider is the only lever left; it never emits framing of
    // its own here, but it does set a Content-Type the original response lacked,
    // so keep the header block byte-identical to the unstamped case.
    const bool had_content_type = response.has_header("Content-Type");
    response.set_content_provider(
        response.get_header_value("Content-Type"),
        [](std::size_t, httplib::DataSink&) { return false; }, std::move(releaser));
    if (!had_content_type) { response.headers.erase("Content-Type"); }
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
