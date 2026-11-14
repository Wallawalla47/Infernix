#pragma once

#include "infernix/types.h"
#include "runtime/prefix_cache/prefix_index.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <vector>

namespace infernix::runtime::prefix_cache {

// The source-choice rules every hybrid prefix cache Program follows (hybrid-prefix-cache-spec §6.2,
// §16.10). Qwen3.5 and Qwen3.8-Flash-Next quote, bind and hold through them; each supplies only its
// mechanics (what a source needs, what fits beside the residents now). Kept in one place because the
// two drifted apart once: Flash-Next resumed from its pause captures over its ledger while Qwen3.5 had
// lost both, and two agent conversations on the 27B replayed for minutes.

// The order a request's candidates are tried in. A resumed request takes its deepest first: every
// token above it would be replayed, its pause's own snapshot leaves nothing to replay. Any other
// request takes the cost model's choice (index.choose) first, then the rest in match order.
[[nodiscard]] std::vector<std::size_t> source_order(const MatchResult& match, std::optional<std::size_t> choice,
                                                    bool resumed);

// Whether a candidate that fits beside the residents now may stand in for the preferred one (the
// first of source_order): a resumed request takes no other, any other request one at most a prefill
// chunk short. A shallower source needs the pages of the history it recomputes within its first units
// anyway, and recomputing restorable history costs far more than restoring it (2026-11-14: sources
// 65K and 71K tokens short of the cached history, 63-94 s to the first token). Otherwise the preferred
// source is offered even though it does not fit now, and its binding waits for room.
[[nodiscard]] constexpr bool may_stand_in(std::uint32_t candidate_frontier, std::uint32_t preferred_frontier,
                                          bool resumed, std::uint32_t prefill_chunk) noexcept {
    return resumed ? candidate_frontier == preferred_frontier
                   : static_cast<std::uint64_t>(candidate_frontier) + prefill_chunk >= preferred_frontier;
}

// Extends a prompt's lookup keys (one hash per full block, `extras` empty or one per block) over the
// blocks of a paused request's `history` past them, chaining each hash from the previous block's with
// the prompt's trailing Vision key, as the Programs publish generated blocks. A history no longer than
// the keys' blocks changes nothing.
void extend_history_keys(std::span<const TokenId> history, std::vector<std::uint64_t>& hashes,
                         std::vector<std::uint64_t>& extras, std::uint64_t trailing_extra);

// A pause's resume point is admitted to the Host tier (and, on Qwen3.5, to a Device snapshot slot) over
// any retained snapshot that is not pinned or held; its retention priority is its own afterwards. A
// refused one would make the resume replay.
inline constexpr double kResumePointClaim = std::numeric_limits<double>::infinity();

} // namespace infernix::runtime::prefix_cache
