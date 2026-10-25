#include "runtime/expert_offload/route_predictor.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace infernix::runtime::expert_offload::expert_cache {

RoutePredictor::RoutePredictor(std::uint32_t layers, std::uint32_t experts, std::uint32_t top_k,
                               std::uint32_t per_layer, double half_life)
    : layers_(layers), experts_(experts), top_k_(top_k), per_layer_(std::min(per_layer, experts)),
      growth_(std::exp2(1.0 / half_life)) {
    if (layers == 0 || experts == 0 || top_k == 0 || per_layer == 0 || !(half_life > 0.0)) {
        throw std::invalid_argument("route predictor: empty geometry or no prediction");
    }
    counts_.assign(static_cast<std::size_t>(layers) * experts * experts, 0.0F);
    out_.assign(static_cast<std::size_t>(layers) * experts, 0.0F);
    token_.resize(static_cast<std::size_t>(layers) * top_k);
    score_.resize(experts);
}

void RoutePredictor::add(std::uint32_t layer, const std::int32_t* from, const std::int32_t* to, float weight) {
    float* counts = counts_.data() + static_cast<std::size_t>(layer) * experts_ * experts_;
    float* out    = out_.data() + static_cast<std::size_t>(layer) * experts_;
    for (std::uint32_t i = 0; i < top_k_; ++i) {
        const std::int32_t a = from[i];
        if (a < 0 || static_cast<std::uint32_t>(a) >= experts_) { continue; }
        for (std::uint32_t j = 0; j < top_k_; ++j) {
            const std::int32_t b = to[j];
            if (b < 0 || static_cast<std::uint32_t>(b) >= experts_) { continue; }
            counts[static_cast<std::size_t>(a) * experts_ + static_cast<std::uint32_t>(b)] += weight;
            out[a] += weight;
        }
    }
}

void RoutePredictor::observe(const std::int32_t* routes, std::size_t stride, std::int32_t columns,
                             std::span<const std::int32_t> lanes, std::span<const std::uint8_t> live) {
    static constexpr std::int32_t kNoLane[] = {-1};
    if (lanes.empty()) { lanes = kNoLane; }
    const auto rows  = static_cast<std::int32_t>(lanes.size());
    const auto width = rows > 0 ? columns / rows : 0;
    round_lanes_.assign(lanes.begin(), lanes.end());
    for (std::int32_t row = 0; row < rows; ++row) {
        // The row's last kMaxRoundTokens live columns, in order.
        std::int32_t first = row * width;
        const std::int32_t end = first + width;
        std::int32_t live_columns = 0;
        for (std::int32_t c = end - 1; c >= first && live_columns < kMaxRoundTokens; --c) {
            if (live.empty() || live[static_cast<std::size_t>(c)] != 0) { ++live_columns; }
            if (live_columns == kMaxRoundTokens) { first = c; }
        }
        std::vector<std::int32_t>& last = last_.try_emplace(lanes[static_cast<std::size_t>(row)]).first->second;
        for (std::int32_t c = first; c < end; ++c) {
            if (!live.empty() && live[static_cast<std::size_t>(c)] == 0) { continue; }
            for (std::uint32_t l = 0; l < layers_; ++l) {
                const std::int32_t* ids = routes + l * stride + static_cast<std::size_t>(c) * top_k_;
                std::copy(ids, ids + top_k_, token_.begin() + static_cast<std::ptrdiff_t>(l) * top_k_);
            }
            if (!last.empty()) {
                const auto weight = static_cast<float>(weight_);
                for (std::uint32_t l = 0; l < layers_; ++l) {
                    add(l, last.data() + static_cast<std::size_t>(l) * top_k_,
                        token_.data() + static_cast<std::size_t>(l) * top_k_, weight);
                }
                weight_ *= growth_;
                if (weight_ > 1e20) {
                    // Rescale everything rather than let the increments overflow FP32.
                    const auto scale = static_cast<float>(1.0 / weight_);
                    for (float& v : counts_) { v *= scale; }
                    for (float& v : out_) { v *= scale; }
                    weight_ = 1.0;
                }
            }
            last = token_;
        }
    }
}

std::span<const std::uint32_t> RoutePredictor::predict(std::uint32_t layer) {
    keys_.clear();
    const float* counts = counts_.data() + static_cast<std::size_t>(layer) * experts_ * experts_;
    const float* out    = out_.data() + static_cast<std::size_t>(layer) * experts_;
    for (const std::int32_t lane : round_lanes_) {
        const auto it = last_.find(lane);
        if (it == last_.end() || it->second.empty()) { continue; }
        std::fill(score_.begin(), score_.end(), 0.0F);
        const std::int32_t* ids = it->second.data() + static_cast<std::size_t>(layer) * top_k_;
        for (std::uint32_t i = 0; i < top_k_; ++i) {
            const std::int32_t a = ids[i];
            if (a < 0 || static_cast<std::uint32_t>(a) >= experts_ || !(out[a] > 0.0F)) { continue; }
            const float total = out[a];
            const float* row  = counts + static_cast<std::size_t>(a) * experts_;
            for (std::uint32_t b = 0; b < experts_; ++b) { score_[b] += row[b] / total; }
        }
        order_.resize(experts_);
        for (std::uint32_t b = 0; b < experts_; ++b) { order_[b] = b; }
        // Best per_layer_ by score, then the lower expert.
        const auto better = [&](std::uint32_t x, std::uint32_t y) {
            return score_[x] != score_[y] ? score_[x] > score_[y] : x < y;
        };
        std::nth_element(order_.begin(), order_.begin() + (per_layer_ - 1), order_.end(), better);
        for (std::uint32_t n = 0; n < per_layer_; ++n) {
            if (score_[order_[n]] > 0.0F) { keys_.push_back(layer * experts_ + order_[n]); }
        }
    }
    std::sort(keys_.begin(), keys_.end());
    keys_.erase(std::unique(keys_.begin(), keys_.end()), keys_.end());
    return keys_;
}

} // namespace infernix::runtime::expert_offload::expert_cache
