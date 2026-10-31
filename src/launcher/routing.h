#pragma once

// Model-name routing: the `model` a request names, matched against the user's routes (patterns
// with `*` wildcards, case-insensitive, first match wins) before the configurations' own ids.

#include "launcher/store.h"

#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace infernix::launcher {

// `*` matches any run of characters (none included); everything else matches itself, ignoring case.
[[nodiscard]] bool wildcard_match(std::string_view pattern, std::string_view text);

// The first route whose pattern matches `name`.
[[nodiscard]] const ModelRoute* match_route(const std::vector<ModelRoute>& routes, std::string_view name);

} // namespace infernix::launcher
