#pragma once

// The request headers a page on another origin may send to infernix-serve's API, as CORS preflights are answered
// (Access-Control-Allow-Headers): the headers the APIs define, named because a wildcard does not cover Authorization
// (Fetch), then "*" for any other a page's client adds (the official JavaScript SDKs send their own, such as
// x-stainless-* and anthropic-dangerous-direct-browser-access). The launcher, which proxies these routes, allows
// them too (with its own api-key header). Header-only: the launcher links nothing of serve's.

#include <string_view>

namespace infernix::serve {

inline constexpr std::string_view kCorsAllowHeaders =
    "Authorization, Content-Type, X-API-Key, anthropic-version, anthropic-beta, anthropic-user-profile-id, *";

} // namespace infernix::serve
