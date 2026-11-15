#pragma once

// Whether pausing a lane now would leave its state at its frontier in the prefix cache (design §19.3.13): what
// ProgramImpl::prefix_finish does for a pausing lane, as a pure function of the lane's counters, so that
// ProgramImpl::pause_replay_tokens predicts the pause it prices. Pure: no cache or device state.

#include "runtime/prefix_cache/block_hash.h"

#include <algorithm>
#include <cstdint>

namespace infernix::models::qwen4_exp::prefix {

// A lane between units (no round in flight), with free cache resume.
struct PauseLane {
    std::uint32_t frontier    = 0; // state_tokens
    std::uint32_t deepest     = 0; // frontier of the lineage's deepest snapshot
    std::uint32_t history     = 0; // tokens the lane knows (the prompt, the committed output and its next input)
    std::uint32_t path_blocks = 0; // blocks of the lane's path now (mapped, then published)
    bool mtp_live             = false;
    std::uint32_t mtp_cells   = 0; // MTP cells [0, mtp_cells) are final
};

// The MTP cells after the finish flush (rule MR3): the pending cell is written when its token is known.
[[nodiscard]] inline std::uint32_t flushed_mtp_cells(const PauseLane& lane) noexcept {
    return lane.mtp_live && lane.mtp_cells + 1U == lane.frontier && lane.history > lane.frontier ? lane.frontier
                                                                                                : lane.mtp_cells;
}

// The pause publishes every block its final frontier covers before it captures: the frontier without a live
// drafter, else the flushed MTP cells. The capture is taken past the deepest snapshot once every full block below
// the frontier is final and on the lane's path.
[[nodiscard]] inline bool pause_captures_frontier(const PauseLane& lane) noexcept {
    constexpr std::uint32_t kBlock  = runtime::prefix_cache::kBlockTokens;
    const std::uint32_t cells       = flushed_mtp_cells(lane);
    const std::uint32_t published   = std::min(lane.mtp_live ? cells : lane.frontier, lane.history) / kBlock;
    const std::uint32_t path_blocks = std::max(lane.path_blocks, published);
    const bool full_blocks          = !lane.mtp_live || cells >= lane.frontier / kBlock * kBlock;
    return lane.frontier > lane.deepest && full_blocks && path_blocks >= lane.frontier / kBlock;
}

} // namespace infernix::models::qwen4_exp::prefix
