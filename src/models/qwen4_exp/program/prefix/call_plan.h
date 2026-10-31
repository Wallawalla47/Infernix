#pragma once

// The prefill call planner of Qwen4Exp's prefix cache (docs/maintainer/qwen3_8-flash-next-design.md
// §19.3.1, "Prefill integration"): the calls that prefill a prompt's suffix [F, n), and which exact
// taps split them. Pure: plans depend only on their arguments (not on concurrency or speculation).

#include "runtime/prefix_cache/cost.h"
#include "runtime/prefix_cache/tap_planner.h"

#include <cstdint>
#include <span>
#include <vector>

namespace infernix::models::qwen4_exp::prefix {

// What a call costs beyond its tokens (their per-token and attention terms are equal with and
// without a split, so admission compares fixed costs only).
struct CallCost {
    runtime::prefix_cache::CacheCostModel model; // call_seconds(width) for GPU-staged calls
    std::uint32_t served_columns = 8;            // calls this narrow are CPU-served (no staging)
    double served_call_seconds   = 0.035;        // their fixed cost (estimated, 25-40 ms)
    double split_budget          = 0.1;          // the most an automatic exact tap may add
    // The layer walk streams the experts once per span: a run of consecutive calls at least
    // walk_min_columns wide, cut where it would pass walk_max_tokens (prefill_walk.cpp). A cut costs
    // a span only where it leaves a narrower call between two such calls (taps inside a span are
    // captured in place).
    double span_seconds            = 0.0;
    std::uint32_t walk_min_columns = 256;
    std::uint32_t walk_max_tokens  = 65536;

    [[nodiscard]] double call_seconds(std::uint32_t width) const noexcept;
};

struct CallPlan {
    // Exclusive end of every call, ascending; the last is the prompt length.
    std::vector<std::uint32_t> ends;
    // The taps with their realized placement: an admitted exact tap is the end of a call; a demoted
    // one is flexible.
    std::vector<runtime::prefix_cache::PlannedTap> taps;
    // Fixed cost of the calls (CallCost::call_seconds summed) and of the walk's spans.
    double seconds = 0.0;
};

// Calls covering [frontier, prompt_tokens): boundaries {frontier} ∪ kept exact taps ∪ {prompt_tokens},
// each segment running `chunk`-token calls and one remainder call. Boundary exact taps (Explicit,
// Structural) are always kept; every other exact tap (the generation opener, Automatic) is kept, in
// position order, when the call it starts is CPU-served (at most `served_columns`) and it adds at most
// `split_budget` to the plan, and otherwise becomes flexible.
// Without exact taps the calls are frontier, frontier + chunk, ...; for frontier 0 the cold grid.
// `taps` must be sorted, inside (frontier, prompt_tokens), as plan_taps returns them.
[[nodiscard]] CallPlan plan_calls(std::uint32_t frontier, std::uint32_t prompt_tokens, std::uint32_t chunk,
                                  std::span<const runtime::prefix_cache::PlannedTap> taps, const CallCost& cost);

// Call ends of [frontier, n), `chunk`-token calls and one remainder call between consecutive cuts (sorted,
// inside the range): a plan's calls, or its remaining calls planned again at a narrower chunk with its
// kept exact taps as `cuts`.
[[nodiscard]] std::vector<std::uint32_t> segment_ends(std::uint32_t frontier, std::uint32_t n, std::uint32_t chunk,
                                                      std::span<const std::uint32_t> cuts);

// Whether a frontier lies strictly inside one of the spans (a Vision item's tokens).
[[nodiscard]] bool inside_exclusion(std::uint32_t frontier,
                                    std::span<const runtime::prefix_cache::TapExclusion> exclusions) noexcept;

} // namespace infernix::models::qwen4_exp::prefix
