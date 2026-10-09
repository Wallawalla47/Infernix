#include "models/qwen3_5/program/ledger_splits.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace infernix::models::qwen3_5::detail {

void LedgerSplits::assign(const PreparedPromptData& prompt) {
    tokens_    = prompt.token_ids.size();
    frontiers_ = prompt.identity.rewrite_execution_frontiers;
}

void LedgerSplits::append_generated(std::size_t count, std::optional<std::uint32_t> split_after) {
    if (count > std::numeric_limits<std::size_t>::max() - tokens_) {
        throw std::overflow_error("generated ledger length overflows size_t");
    }
    if (split_after) {
        if (*split_after == 0 || *split_after > count ||
            tokens_ > std::numeric_limits<std::uint32_t>::max() - *split_after) {
            throw std::invalid_argument("generated execution split is outside the appended span");
        }
        const auto frontier = static_cast<std::uint32_t>(tokens_) + *split_after;
        if (!frontiers_.empty() && frontiers_.back() >= frontier) {
            throw std::logic_error("generated execution split is not a new ordered frontier");
        }
        frontiers_.push_back(frontier);
    }
    tokens_ += count;
}

void LedgerSplits::truncate(std::size_t tokens) {
    if (tokens > tokens_) { throw std::out_of_range("cannot extend ledger splits by truncation"); }
    tokens_ = tokens;
    frontiers_.erase(std::upper_bound(frontiers_.begin(), frontiers_.end(), tokens),
                     frontiers_.end());
}

} // namespace infernix::models::qwen3_5::detail
