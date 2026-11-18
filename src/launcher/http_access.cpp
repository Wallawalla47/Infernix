#include "launcher/http_access.h"

#include "launcher/util.h"

namespace infernix::launcher {

bool is_loopback(const std::string& address) {
    return address == "::1" || address.starts_with("127.") || address.starts_with("::ffff:127.") ||
           address == "localhost";
}

std::string authority_host(std::string_view authority) {
    if (authority.starts_with('[')) {
        const std::size_t close = authority.find(']');
        return lower(authority.substr(1, close == std::string_view::npos ? std::string_view::npos : close - 1));
    }
    return lower(authority.substr(0, authority.find(':')));
}

bool ip_literal(const std::string& host) {
    if (host.find(':') != std::string::npos) { return true; } // IPv6
    int dots = 0;
    for (const char c : host) {
        if (c == '.') {
            ++dots;
        } else if (c < '0' || c > '9') {
            return false;
        }
    }
    return dots == 3;
}

bool loopback_host(const std::string& host) {
    // 127.0.0.0/8 only as an address: "127.example.com" is a name that may resolve anywhere (DNS rebinding).
    return host == "localhost" || host.ends_with(".localhost") || host == "::1" ||
           ((host.starts_with("127.") || host.starts_with("::ffff:127.")) && ip_literal(host));
}

std::optional<std::string> page_refusal(std::string_view host, std::string_view origin, bool has_key) {
    const std::string host_name = authority_host(host);
    if (!has_key && !host_name.empty() && !ip_literal(host_name) && !loopback_host(host_name)) {
        return "open the launcher by IP address or localhost (not " + host_name + "), or send an admin key";
    }
    if (!origin.empty()) {
        const std::size_t scheme         = origin.find("://");
        const std::string_view authority = scheme == std::string_view::npos ? std::string_view() : origin.substr(scheme + 3);
        if (authority.empty() || (!iequals(authority, host) && !loopback_host(authority_host(authority)))) {
            return std::string("the launcher's management API does not answer other web pages");
        }
    }
    return std::nullopt;
}

} // namespace infernix::launcher
