#include "models/qwen4_exp/program/prefix/call_plan.h"

#include <algorithm>
#include <stdexcept>

namespace infernix::models::qwen4_exp::prefix {

namespace pc = runtime::prefix_cache;

double CallCost::call_seconds(std::uint32_t width) const noexcept {
    if (width == 0) { return 0.0; }
    return width <= served_columns ? served_call_seconds : model.call_seconds(width);
}

namespace {

// Call ends of [frontier, n) cut at `cuts` (sorted, inside the range).
std::vector<std::uint32_t> segment_ends(std::uint32_t frontier, std::uint32_t n, std::uint32_t chunk,
                                        std::span<const std::uint32_t> cuts) {
    std::vector<std::uint32_t> ends;
    std::uint32_t begin = frontier;
    const auto segment  = [&](std::uint32_t end) {
        while (begin < end) {
            begin = end - begin > chunk ? begin + chunk : end;
            ends.push_back(begin);
        }
    };
    for (const std::uint32_t cut : cuts) { segment(cut); }
    segment(n);
    return ends;
}

double seconds_of(std::uint32_t frontier, std::span<const std::uint32_t> ends, std::span<const std::uint32_t> cuts,
                  const CallCost& cost) {
    double total        = 0.0;
    std::uint32_t begin = frontier;
    for (const std::uint32_t end : ends) {
        const std::uint32_t width = end - begin;
        total += cost.call_seconds(width);
        if (width > cost.served_columns && std::binary_search(cuts.begin(), cuts.end(), begin)) {
            total += cost.span_seconds;
        }
        begin = end;
    }
    return total;
}

} // namespace

CallPlan plan_calls(std::uint32_t frontier, std::uint32_t prompt_tokens, std::uint32_t chunk,
                    std::span<const pc::PlannedTap> taps, const CallCost& cost) {
    if (chunk == 0 || frontier >= prompt_tokens) {
        throw std::invalid_argument("Qwen4Exp call plan: an empty suffix or a zero chunk");
    }
    CallPlan out;
    out.taps.assign(taps.begin(), taps.end());
    for (std::size_t i = 0; i < out.taps.size(); ++i) {
        const std::uint32_t p = out.taps[i].position;
        if (p <= frontier || p >= prompt_tokens || (i > 0 && p <= out.taps[i - 1].position)) {
            throw std::invalid_argument("Qwen4Exp call plan: taps must be sorted inside the suffix");
        }
    }
    std::vector<std::uint32_t> cuts;
    for (const pc::PlannedTap& tap : out.taps) {
        if (tap.placement == pc::TapPlacement::Exact && tap.boundary) { cuts.push_back(tap.position); }
    }
    double current = seconds_of(frontier, segment_ends(frontier, prompt_tokens, chunk, cuts), cuts, cost);
    for (pc::PlannedTap& tap : out.taps) {
        if (tap.placement != pc::TapPlacement::Exact || tap.boundary) { continue; }
        std::vector<std::uint32_t> with = cuts;
        with.insert(std::upper_bound(with.begin(), with.end(), tap.position), tap.position);
        const double seconds = seconds_of(frontier, segment_ends(frontier, prompt_tokens, chunk, with), with, cost);
        if (seconds - current <= cost.split_budget) {
            cuts    = std::move(with);
            current = seconds;
        } else {
            tap.placement = pc::TapPlacement::Flexible;
        }
    }
    out.ends    = segment_ends(frontier, prompt_tokens, chunk, cuts);
    out.seconds = current;
    return out;
}

bool inside_exclusion(std::uint32_t frontier, std::span<const pc::TapExclusion> exclusions) noexcept {
    return std::any_of(exclusions.begin(), exclusions.end(), [&](const pc::TapExclusion& span) {
        return frontier > span.begin && frontier < span.end;
    });
}

} // namespace infernix::models::qwen4_exp::prefix
