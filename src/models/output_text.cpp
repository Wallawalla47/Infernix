#include "models/output_text.h"

#include <algorithm>
#include <vector>

namespace infernix::models {

std::string consume_generated_utf8(std::string& pending) {
    std::string decoded;
    decoded.reserve(pending.size());
    std::size_t offset = 0;
    while (offset < pending.size()) {
        const auto lead    = static_cast<unsigned char>(pending[offset]);
        std::size_t length = 0;
        if (lead <= 0x7fU) {
            decoded.push_back(pending[offset]);
            ++offset;
            continue;
        } else if (lead >= 0xc2U && lead <= 0xdfU) {
            length = 2;
        } else if (lead >= 0xe0U && lead <= 0xefU) {
            length = 3;
        } else if (lead >= 0xf0U && lead <= 0xf4U) {
            length = 4;
        } else {
            decoded.append(kUtf8Replacement);
            ++offset;
            continue;
        }

        bool malformed = false;
        for (std::size_t index = 1; index < length; ++index) {
            if (offset + index >= pending.size()) {
                pending.erase(0, offset);
                return decoded;
            }
            const auto byte      = static_cast<unsigned char>(pending[offset + index]);
            unsigned int minimum = 0x80U;
            unsigned int maximum = 0xbfU;
            if (index == 1) {
                if (lead == 0xe0U) {
                    minimum = 0xa0U;
                } else if (lead == 0xedU) {
                    maximum = 0x9fU;
                } else if (lead == 0xf0U) {
                    minimum = 0x90U;
                } else if (lead == 0xf4U) {
                    maximum = 0x8fU;
                }
            }
            if (byte < minimum || byte > maximum) {
                // Replace one maximal subpart. The first byte that cannot continue this sequence
                // is deliberately left for the next iteration, so valid following text is kept.
                decoded.append(kUtf8Replacement);
                offset += index;
                malformed = true;
                break;
            }
        }
        if (malformed) { continue; }

        decoded.append(pending, offset, length);
        offset += length;
    }
    pending.clear();
    return decoded;
}

// O(n) for n = min(|text|, |marker|): Knuth-Morris-Pratt over text's last n bytes. (Comparing every size from the
// largest down was quadratic once a long held prefix broke, and a stop string's length is the client's; QA-14.)
std::size_t longest_suffix_prefix(std::string_view text, std::string_view marker) {
    const std::size_t n = std::min(text.size(), marker.size());
    if (n == 0) { return 0; }
    thread_local std::vector<std::size_t> border;
    border.assign(n, 0);
    for (std::size_t i = 1, k = 0; i < n; ++i) {
        while (k != 0 && marker[i] != marker[k]) { k = border[k - 1]; }
        if (marker[i] == marker[k]) { ++k; }
        border[i] = k;
    }
    std::size_t state = 0;
    for (const char c : text.substr(text.size() - n)) {
        while (state != 0 && (state == marker.size() || c != marker[state])) { state = border[state - 1]; }
        if (c == marker[state]) { ++state; }
    }
    return state == marker.size() ? border[state - 1] : state;
}

std::size_t stop_hold_size(std::string_view text, OutputChannel channel, const StopPolicy& policy) {
    std::size_t hold = 0;
    for (const StopString& stop : policy.strings) {
        if (stop.channel != channel) { continue; }
        hold = std::max(hold, longest_suffix_prefix(text, stop.text));
    }
    return hold;
}

} // namespace infernix::models
