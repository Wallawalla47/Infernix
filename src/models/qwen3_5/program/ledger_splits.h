#pragma once

// The execution frontiers of a sequence's ledger: the positions where its prefill was split into
// separate passes (prompt rewrite frontiers and generated control spans). Replay splits at the same
// positions, so a recovered state has the same GDN decomposition as the uninterrupted one.

#include "models/qwen3_5/frontend/prepared_prompt.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace infernix::models::qwen3_5::detail {

class LedgerSplits {
public:
    void assign(const PreparedPromptData& prompt);
    // Extends the ledger by `count` generated tokens; `split_after` places a frontier after that
    // many of them.
    void append_generated(std::size_t count,
                          std::optional<std::uint32_t> split_after = std::nullopt);
    void truncate(std::size_t tokens);

    // Ledger tokens covered.
    [[nodiscard]] std::size_t size() const noexcept { return tokens_; }

    [[nodiscard]] std::span<const std::uint32_t> frontiers() const noexcept { return frontiers_; }

private:
    std::size_t tokens_ = 0;
    std::vector<std::uint32_t> frontiers_;
};

} // namespace infernix::models::qwen3_5::detail
