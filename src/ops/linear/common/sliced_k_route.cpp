#include "ops/linear/common/sliced_k_route.h"

#include "core/temporary_switches.h"

#include <algorithm>
#include <atomic>
#include <stdexcept>
#include <string>
#include <tuple>

namespace infernix::ops::detail {
namespace {

struct Environment {
    SlicedKRoute route = SlicedKRoute::OneWave;
    SlicedKShape forced;
};

// Read once. An invalid shape throws from temporary_switches::describe(), which the Engine and the op
// benches call at startup.
const Environment& environment() {
    static const Environment value = [] {
        Environment out;
        const auto shape = temporary_switches::gw_shape();
        out.forced       = {shape.block_rows, shape.stages};
        return out;
    }();
    return value;
}

// Test override: 0 none, else 1 + SlicedKRoute.
std::atomic<int> override_route{0};
std::atomic<int> override_rows{0};
std::atomic<int> override_stages{0};

} // namespace

SlicedKRoute groupwise_sliced_k_route() noexcept {
    if (const int route = override_route.load(std::memory_order_relaxed); route != 0) {
        return static_cast<SlicedKRoute>(route - 1);
    }
    try {
        return environment().route;
    } catch (...) { return SlicedKRoute::Release; }
}

SlicedKShape forced_sliced_k_shape() noexcept {
    if (override_route.load(std::memory_order_relaxed) != 0) {
        return {override_rows.load(std::memory_order_relaxed),
                override_stages.load(std::memory_order_relaxed)};
    }
    try {
        return environment().forced;
    } catch (...) { return {}; }
}

void set_groupwise_sliced_k_route_for_testing(std::optional<SlicedKRoute> route,
                                              SlicedKShape forced) {
    override_rows.store(forced.block_rows, std::memory_order_relaxed);
    override_stages.store(forced.stages, std::memory_order_relaxed);
    override_route.store(route ? 1 + static_cast<int>(*route) : 0, std::memory_order_relaxed);
}

std::size_t choose_sliced_k_shape(std::span<const SlicedKCandidate> candidates,
                                  int multiprocessors) {
    if (multiprocessors <= 0) { throw std::invalid_argument("sliced-K shape: no multiprocessors"); }
    std::size_t best = candidates.size();
    // Ordered key: fewer waves, more bytes in flight, fewer rows, fewer stages.
    std::tuple<std::int64_t, std::int64_t, int, int> best_key{};
    for (std::size_t index = 0; index < candidates.size(); ++index) {
        const SlicedKCandidate& c = candidates[index];
        if (c.resident <= 0 || c.ctas <= 0) { continue; }
        const std::int64_t slots    = static_cast<std::int64_t>(c.resident) * multiprocessors;
        const std::int64_t waves    = (c.ctas + slots - 1) / slots;
        const std::int64_t per_sm   = std::min<std::int64_t>(
            c.resident, (c.ctas + multiprocessors - 1) / multiprocessors);
        const std::int64_t inflight = per_sm * c.shape.stages * c.weight_slot_bytes;
        const std::tuple<std::int64_t, std::int64_t, int, int> key{waves, -inflight, c.shape.block_rows,
                                                                   c.shape.stages};
        if (best == candidates.size() || key < best_key) {
            best     = index;
            best_key = key;
        }
    }
    if (best == candidates.size()) {
        throw std::logic_error("sliced-K shape: no candidate has a resident CTA");
    }
    return best;
}

} // namespace infernix::ops::detail
