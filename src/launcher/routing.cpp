#include "launcher/routing.h"

#include <cctype>

namespace infernix::launcher {

bool wildcard_match(std::string_view pattern, std::string_view text) {
    const auto same = [](char a, char b) {
        return std::tolower(static_cast<unsigned char>(a)) == std::tolower(static_cast<unsigned char>(b));
    };
    // Iterative glob with one backtrack point: linear in practice, no recursion.
    std::size_t p = 0, t = 0, star = std::string_view::npos, resume = 0;
    while (t < text.size()) {
        if (p < pattern.size() && pattern[p] == '*') {
            star   = p++;
            resume = t;
        } else if (p < pattern.size() && same(pattern[p], text[t])) {
            ++p;
            ++t;
        } else if (star != std::string_view::npos) {
            p = star + 1;
            t = ++resume;
        } else {
            return false;
        }
    }
    while (p < pattern.size() && pattern[p] == '*') { ++p; }
    return p == pattern.size();
}

const ModelRoute* match_route(const std::vector<ModelRoute>& routes, std::string_view name) {
    for (const ModelRoute& route : routes) {
        if (wildcard_match(route.pattern, name)) { return &route; }
    }
    return nullptr;
}

} // namespace infernix::launcher
