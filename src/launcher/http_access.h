#pragma once

// Who may reach the launcher's HTTP routes: the address and host checks its access control (Launcher::page_checks,
// admin_allowed, from_own_page) is built from.

#include <optional>
#include <string>
#include <string_view>

namespace infernix::launcher {

// A peer address (httplib's remote_addr, numeric) on this computer: 127.0.0.0/8, ::1 or an IPv4-mapped 127 address.
[[nodiscard]] bool is_loopback(const std::string& address);

// The host of a Host header or an origin's authority: lower case, without the port or brackets.
[[nodiscard]] std::string authority_host(std::string_view authority);

// A numeric host: four dotted decimal parts, or IPv6 (any colon).
[[nodiscard]] bool ip_literal(const std::string& host);

// A host that names this computer: localhost and its subdomains, ::1, and numeric 127.0.0.0/8 (also IPv4-mapped).
// A DNS name that only begins "127." (127.evil.com) is not one.
[[nodiscard]] bool loopback_host(const std::string& host);

// Whether a web page may drive the management API, from the request's Host and Origin headers: the message to
// refuse it with, or nullopt. A page's requests carry its Origin, refused unless it is this page's own or a loopback
// one; a DNS-rebinding page names its own domain in Host, refused unless Host is an IP address or localhost (or the
// request carries an admin key, `has_key`).
[[nodiscard]] std::optional<std::string> page_refusal(std::string_view host, std::string_view origin, bool has_key);

} // namespace infernix::launcher
