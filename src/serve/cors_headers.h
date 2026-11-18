#pragma once

// The request headers a page on another origin may send to infernix-serve's API, as CORS preflights are answered
// (Access-Control-Allow-Headers). Named, not "*": a wildcard does not cover Authorization (Fetch). The launcher,
// which proxies these routes, allows them too (with its own api-key header). Header-only: the launcher links
// nothing of serve's.

#include <string_view>

namespace infernix::serve {

inline constexpr std::string_view kCorsAllowHeaders =
    "Authorization, Content-Type, X-API-Key, anthropic-version, anthropic-beta, anthropic-user-profile-id";

} // namespace infernix::serve
