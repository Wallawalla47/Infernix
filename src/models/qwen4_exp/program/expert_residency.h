#pragma once

// The VRAM expert cache of the Qwen4Exp Program (docs/maintainer/qwen3_8-flash-next-design.md
// §9): a pool of device frames, each holding one routed expert's nvfp4_expert_rg16_v1 record, the
// per-layer frame tables the MoE kernels read (frame or -1: read the pinned bank zero-copy), the
// route log of every round, and the LFRU policy (expert_cache::CacheController) that decides
// promotions and evictions at round boundaries.
//
// Ordering. Rounds run on the compute stream and the Program synchronizes at the end of each. At
// the boundary the policy's evictions are applied to the device tables on the compute stream
// before the next round, and the promotions' copies run on a copy stream that first waits for that
// table update, so no kernel can read a frame while it is overwritten. A promoted expert becomes
// visible in its table only after its copy has completed. Expert outputs are placement-invariant
// (design §16.2), so the round a promotion lands in does not change any result.

#include "core/arena.h"
#include "core/device.h"
#include "models/qwen4_exp/config.h"
#include "models/qwen4_exp/program/expert_cache/expert_cache.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace ninfer::models::qwen4_exp {

class ExpertResidency {
public:
    struct Stats {
        std::uint64_t routed     = 0; // routed (layer, expert) uses, summed over rounds' groups
        std::uint64_t hits       = 0; // of those, resident in a frame at the round's start
        std::uint64_t promotions = 0; // expert copies issued
    };

    // banks[l]: layer l's pinned host records (record_stride bytes apart). frames: device frames
    // to allocate (0 disables the cache). max_columns: the most columns one round routes.
    ExpertResidency(const TextConfig& config, std::vector<const std::uint8_t*> banks, std::uint64_t record_stride,
                    std::uint32_t frames, std::int32_t max_columns);
    ~ExpertResidency();
    ExpertResidency(const ExpertResidency&)            = delete;
    ExpertResidency& operator=(const ExpertResidency&) = delete;

    [[nodiscard]] const std::uint8_t* frame_base() const noexcept;
    [[nodiscard]] std::uint64_t frame_stride() const noexcept { return stride_; }
    [[nodiscard]] const std::int32_t* table(std::uint32_t layer) const noexcept;
    [[nodiscard]] std::int32_t* route_log() noexcept { return static_cast<std::int32_t*>(route_device_.p); }
    [[nodiscard]] std::size_t route_stride() const noexcept { return route_stride_; }
    [[nodiscard]] std::uint32_t frames() const noexcept { return frames_; }

    // Before a round: publishes the loads that completed since the last round.
    void before_round(cudaStream_t compute);
    // After the round's kernels are enqueued: downloads its route log with the round.
    void enqueue_route_download(cudaStream_t compute, std::int32_t columns);
    // After the round completed (the compute stream was synchronized): runs the policy over the
    // round's routes and issues its evictions and promotions. `per_layer_budget` caps the
    // promotions one layer's routes may start. A non-empty `live` (one flag per column) credits
    // only the flagged columns' experts: a verification round's rejected draft columns neither
    // count as uses nor start promotions (design section 11.3).
    void after_round(cudaStream_t compute, std::int32_t columns, std::size_t per_layer_budget,
                     std::span<const std::uint8_t> live = {});

    [[nodiscard]] const Stats& stats() const noexcept { return stats_; }

private:
    void upload_table(cudaStream_t compute);

    const TextConfig& c_;
    std::vector<const std::uint8_t*> banks_;
    std::uint64_t stride_ = 0;
    std::uint32_t frames_ = 0;
    std::uint32_t experts_ = 0, layers_ = 0, top_k_ = 0;
    std::size_t route_stride_ = 0;

    DeviceBuffer frame_memory_;
    DeviceBuffer table_device_;
    PinnedHostBuffer table_host_{1};
    DeviceBuffer route_device_;
    PinnedHostBuffer route_host_{1};
    bool table_dirty_ = true;

    std::unique_ptr<expert_cache::CacheController> controller_;
    std::vector<expert_cache::CacheController::Command> commands_;
    std::vector<std::uint32_t> group_;
    std::vector<std::uint8_t> seen_;
    std::uint64_t round_ = 0;

    cudaStream_t copy_stream_ = nullptr;
    cudaEvent_t table_ready_  = nullptr;
    struct Batch {
        cudaEvent_t done = nullptr;
        std::vector<std::pair<std::uint32_t, std::uint32_t>> loads; // (key, frame)
    };
    std::vector<Batch> in_flight_;
    std::vector<cudaEvent_t> spare_events_;
    Stats stats_;
};

} // namespace ninfer::models::qwen4_exp
