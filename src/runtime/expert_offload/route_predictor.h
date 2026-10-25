#pragma once

// TEMPORARY (INFERNIX_TMP_ROUTE_PROTECT, candidates §13 PUB-22): a routing-sequence predictor whose
// predictions protect resident experts from eviction (SeqMoE, arXiv 2609.12978; protection only: its
// prefetch half is the rejected prefetch class). tools/expert_cache_replay/route_predictor.py's
// TransitionPredictor models it operation for operation (FP32 counts and scores, the same addition
// order), and its replay decides whether this is A/B'd (only on a decode hit-rate gain of 3 points or
// more over the LFRU). It runs on the host beside the policy, so it costs no frames (host memory:
// layers x experts^2 FP32, 50 MiB for Flash-Next).
//
// Per layer, exponentially decayed counts of expert b at a lane's live token t + 1 after expert a at
// its token t (half-life in tokens; each row of a round contributes its last kMaxRoundTokens live
// tokens; counts are rescaled once an increment passes 1e20). After a round is observed,
// predict(layer) names for each of its lanes the `per_layer` experts b with the largest sum, over that
// lane's last token's experts a, of counts[a][b] / out[a] (out[a] the row sum; ties: the lower
// expert), positive scores only. LfruPolicy::step evicts them only when no other resident is left.

#include <cstddef>
#include <cstdint>
#include <span>
#include <unordered_map>
#include <vector>

namespace infernix::runtime::expert_offload::expert_cache {

class RoutePredictor {
public:
    static constexpr std::int32_t kMaxRoundTokens = 256;

    RoutePredictor(std::uint32_t layers, std::uint32_t experts, std::uint32_t top_k, std::uint32_t per_layer,
                   double half_life = 4096.0);

    // One round. Layer l's ids are at routes + l * stride, top_k per column; the columns are
    // `lanes.size()` rows of columns / rows each (no lanes: one row, lane -1); `live` flags the
    // columns that count (empty: all). Ids outside [0, experts) are skipped.
    void observe(const std::int32_t* routes, std::size_t stride, std::int32_t columns,
                 std::span<const std::int32_t> lanes, std::span<const std::uint8_t> live);
    // Layer `layer`'s predicted keys (layer * experts + expert) for the next round, ascending; valid
    // until the next call.
    [[nodiscard]] std::span<const std::uint32_t> predict(std::uint32_t layer);

private:
    void add(std::uint32_t layer, const std::int32_t* from, const std::int32_t* to, float weight);

    std::uint32_t layers_, experts_, top_k_, per_layer_;
    double growth_;
    double weight_ = 1.0;
    std::vector<float> counts_; // [layers][experts][experts]
    std::vector<float> out_;    // [layers][experts]
    // Each lane's last live token: [layers][top_k] ids.
    std::unordered_map<std::int32_t, std::vector<std::int32_t>> last_;
    std::vector<std::int32_t> round_lanes_; // the lanes of the last observed round
    std::vector<std::int32_t> token_;       // scratch: one token's [layers][top_k] ids
    std::vector<float> score_;              // scratch: [experts]
    std::vector<std::uint32_t> order_, keys_;
};

} // namespace infernix::runtime::expert_offload::expert_cache
