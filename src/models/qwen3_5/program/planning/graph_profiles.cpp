#include "models/qwen3_5/program/planning/graph_profiles.h"
#include "models/qwen3_5/program/execution_context.h"
#include <algorithm>
#include <array>
#include <stdexcept>

namespace infernix::models::qwen3_5::detail {
namespace {
// Resource tiers bound inactive attention work. They are not kernel/topology boundaries.
constexpr std::array<std::uint32_t, 7> kCausalVisibleTiers{128,  512,   2048, 4096,
                                                           8192, 16384, 32768};

std::vector<GraphExecutionProfile>
graph_profiles_through(std::uint32_t max_frontier,
                       const std::vector<std::uint32_t>& preferred_ends) {
    std::vector<GraphExecutionProfile> out;
    std::uint32_t begin = 0;
    for (const std::uint32_t preferred_end : preferred_ends) {
        if (begin > max_frontier) { break; }
        const std::uint32_t end = std::min(preferred_end, max_frontier);
        out.push_back({begin, end});
        if (end == max_frontier) { return out; }
        begin = end + 1;
    }
    if (begin <= max_frontier) { out.push_back({begin, max_frontier}); }
    return out;
}

std::vector<GraphExecutionProfile> causal_resource_profiles(std::uint32_t capacity,
                                                            std::uint32_t visible_offset) {
    std::vector<std::uint32_t> ends;
    for (const auto visible : kCausalVisibleTiers)
        if (visible >= visible_offset) ends.push_back(visible - visible_offset);
    return graph_profiles_through(capacity - 1, ends);
}

} // namespace

std::vector<GraphExecutionProfile> ordinary_graph_profiles(std::uint32_t capacity) {
    // E+1 is the one-token visible window; all tiers share one topology per exact B.
    return causal_resource_profiles(capacity, 1);
}

std::vector<GraphExecutionProfile> mtp_graph_profiles(std::uint32_t capacity,
                                                      std::uint32_t draft_window,
                                                      std::uint32_t neural_drafts) {
    if (draft_window == 0 || capacity == 0) { return {}; }
    // The final AR call can see E+2K. Target verify and MTP attention are update-compatible
    // across all resource tiers, independently of the selected KV representation.
    auto profiles = causal_resource_profiles(capacity, 2 * draft_window);
    if (draft_window != neural_drafts) {
        // Unequal verification/next-draft widths cannot share equal-width graphs.
        for (std::size_t i = 0; i < profiles.size(); ++i) {
            profiles[i].topology_class = static_cast<std::uint32_t>(i);
        }
    }
    return profiles;
}

std::vector<GraphExecutionProfile> dflash_graph_profiles(SpeculativeBackend backend,
                                                         std::uint32_t capacity,
                                                         std::uint32_t draft_window) {
    // Ngram copy verification widens the window up to 63 drafts for any batch size.
    if (capacity == 0 || draft_window == 0 || draft_window > 63) {
        throw std::invalid_argument("invalid masked draft graph dimensions");
    }
    if (backend == SpeculativeBackend::DFlash2) {
        auto profiles = graph_profiles_through(capacity - 1, {96, 511, 2047, 8191, 32767});
        for (std::size_t i = 0; i < profiles.size(); ++i) {
            profiles[i].topology_class = static_cast<std::uint32_t>(i);
        }
        return profiles;
    }
    // Retain the draft's resource and topology profiles. Target causal attention contributes
    // no context-dependent topology class.
    auto profiles = graph_profiles_through(
        capacity - 1, {96, 127, 511, 1023, 2047, 4095, 8191, 16383, 32767, 65536, 131072, 196608});
    for (GraphExecutionProfile& profile : profiles) {
        profile.topology_class = profile.max > 96U ? 1U : 0U;
    }
    return profiles;
}

execution::MtpCausalAttentionEnvelopes mtp_causal_attention_envelopes(std::uint32_t max_frontier,
                                                                      std::uint32_t k,
                                                                      std::uint32_t capacity,
                                                                      std::uint32_t next_k) {
    const auto visible = [capacity](std::uint64_t value) {
        return static_cast<std::uint32_t>(std::min<std::uint64_t>(capacity, value));
    };
    execution::MtpCausalAttentionEnvelopes out;
    out.target_verify = {1, visible(static_cast<std::uint64_t>(max_frontier) + k + 1ULL)};
    out.batch         = out.target_verify;
    for (std::uint32_t step = 0; step + 1 < next_k; ++step) {
        out.ar[step] = {1, visible(static_cast<std::uint64_t>(max_frontier) + k + step + 2ULL)};
    }
    return out;
}

execution::DFlashEnvelopes dflash_envelopes(std::uint32_t min_frontier,
                                            std::uint32_t max_frontier) {
    (void)min_frontier;
    return execution::DFlashEnvelopes{
        .local = {0, max_frontier},
        .full  = {0, max_frontier},
    };
}

} // namespace infernix::models::qwen3_5::detail
