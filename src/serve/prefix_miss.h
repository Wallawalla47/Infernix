#pragma once

// Prefix-reuse miss record (candidates CF-08): explains a large re-prefill by comparing the prompt
// with the contexts (prompt and output) of the last requests. Kept only with the request log.

#include "infernix/types.h"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace infernix::serve {

// Tokens [first, first + token_ids.size()) of one context and their text (invalid UTF-8 at a cut
// replaced).
struct PrefixMissWindow {
    std::uint32_t first = 0;
    std::vector<infernix::TokenId> token_ids;
    std::string text;
};

struct PrefixMiss {
    // Diverged: the prompt departs from the earlier context inside it (the client changed history
    // there). Evicted: the earlier context is a prefix of the prompt (or the prompt of it), yet much
    // less of it was reused.
    enum class Kind : std::uint8_t { Diverged, Evicted };
    Kind kind                                 = Kind::Diverged;
    std::uint64_t earlier_engine_request_id   = 0;
    std::uint32_t earlier_context_tokens      = 0;
    std::uint32_t common_prefix_tokens        = 0;
    std::uint32_t reused_tokens               = 0;
    std::uint32_t computed_prefill_tokens     = 0;
    // Prefix tokens the cache could have supplied but did not: min(common prefix, prompt - 1) less
    // the reused tokens.
    std::uint32_t lost_reuse_tokens           = 0;
    // Diverged only: the window around the divergence in each context.
    std::optional<PrefixMissWindow> earlier;
    std::optional<PrefixMissWindow> current;
};

[[nodiscard]] const char* prefix_miss_kind_name(PrefixMiss::Kind kind) noexcept;

class PrefixMissTracker {
public:
    using Detokenize = std::function<std::string(std::span<const infernix::TokenId>)>;

    // `contexts` recent contexts are kept; a request computing at least `min_tokens` prompt tokens
    // is compared with them; windows hold `window` tokens on each side of a divergence.
    explicit PrefixMissTracker(Detokenize detokenize, std::size_t contexts = 16,
                               std::uint32_t min_tokens = 4096, std::uint32_t window = 64);

    // A settled request: its miss record, if any, against the earlier contexts; then its own
    // context joins them.
    [[nodiscard]] std::optional<PrefixMiss> settle(std::uint64_t engine_request_id,
                                                   std::span<const infernix::TokenId> prompt,
                                                   std::span<const infernix::TokenId> output,
                                                   std::uint32_t reused_tokens,
                                                   std::uint32_t computed_prefill_tokens);

private:
    struct Context {
        std::uint64_t engine_request_id = 0;
        std::vector<infernix::TokenId> tokens;
    };

    [[nodiscard]] PrefixMissWindow window(std::span<const infernix::TokenId> tokens,
                                          std::size_t at) const;

    Detokenize detokenize_;
    std::size_t capacity_;
    std::uint32_t min_tokens_;
    std::uint32_t window_;
    std::mutex mutex_;
    std::deque<Context> contexts_;
};

} // namespace infernix::serve
