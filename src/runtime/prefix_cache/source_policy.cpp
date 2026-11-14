#include "runtime/prefix_cache/source_policy.h"

#include "runtime/prefix_cache/block_hash.h"

#include <algorithm>

namespace infernix::runtime::prefix_cache {

std::vector<std::size_t> source_order(const MatchResult& match, std::optional<std::size_t> choice, bool resumed) {
    std::vector<std::size_t> order;
    order.reserve(match.candidates.size());
    if (choice && *choice < match.candidates.size()) { order.push_back(*choice); }
    for (std::size_t i = 0; i < match.candidates.size(); ++i) {
        if (!choice || i != *choice) { order.push_back(i); }
    }
    if (resumed) {
        std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
            return match.candidates[a].frontier > match.candidates[b].frontier;
        });
    }
    return order;
}

void extend_history_keys(std::span<const TokenId> history, std::vector<std::uint64_t>& hashes,
                         std::vector<std::uint64_t>& extras, std::uint64_t trailing_extra) {
    const std::size_t blocks = history.size() / kBlockTokens;
    if (blocks <= hashes.size()) { return; }
    // Empty extras mean every block's is 0; a non-zero trailing key needs them spelled out.
    if (trailing_extra != 0 && extras.empty()) { extras.assign(hashes.size(), 0ULL); }
    for (std::size_t block = hashes.size(); block < blocks; ++block) {
        const std::uint64_t previous = block == 0 ? kRootLookupHash : hashes[block - 1U];
        hashes.push_back(block_lookup_hash(previous, history.subspan(block * kBlockTokens, kBlockTokens),
                                           trailing_extra));
        if (!extras.empty()) { extras.push_back(trailing_extra); }
    }
}

} // namespace infernix::runtime::prefix_cache
