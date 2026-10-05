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
//
// Size. The frames live in a VMM arena whose base never moves (design §19.3.7): resize() backs more
// frames chunk by chunk, or releases the top chunks after evicting the lowest-score experts and
// moving the survivors below the new top, so kernels and captured graphs keep their addresses.
// Without VMM the frames are one allocation of the first size.

#include "core/arena.h"
#include "core/device.h"
#include "core/vmm_arena.h"
#include "core/vram_budget.h"
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
        std::uint64_t lent_frames    = 0; // frames lent now
        std::uint64_t lend_evictions = 0; // experts evicted to lend their frames, since start
    };

    // banks[l]: layer l's pinned host records (record_stride bytes apart). max_columns: the most
    // columns one round routes. The cache starts with no frames: resize() backs them.
    ExpertResidency(const TextConfig& config, std::vector<const std::uint8_t*> banks, std::uint64_t record_stride,
                    std::int32_t max_columns, int device);
    ~ExpertResidency();
    ExpertResidency(const ExpertResidency&)            = delete;
    ExpertResidency& operator=(const ExpertResidency&) = delete;

    // Device bytes besides the frames: the frame tables and the route log.
    [[nodiscard]] static std::uint64_t table_bytes(const TextConfig& config, std::int32_t max_columns) noexcept;
    // Frames the cache uses at most: every routed expert but one.
    [[nodiscard]] static std::uint32_t max_frames(const TextConfig& config) noexcept;
    // Bytes the frames are mapped in (whole chunks); a grow or shrink moves in these steps.
    static constexpr std::size_t kChunkBytes = 64ULL << 20;

    struct Resize {
        std::uint32_t frames  = 0;     // frames backed afterwards
        std::uint64_t spilled = 0;     // bytes of a chunk the driver placed in system memory (given back)
        bool refused          = false; // the device had no memory for the next chunk
    };
    // Backs `frames` frames (clamped to frame_limit()); unchanged while frames are lent. Call only
    // between rounds, with nothing in flight on the compute stream: promotion copies are waited for,
    // then a shrink evicts the
    // lowest-score experts, moves the survivors above the new top below it on `compute` and
    // releases the top chunks; a grow maps chunks one at a time, stops at the first the driver
    // refuses or places in system memory (`vram` tells), and fills the new frames from the queue.
    // Without VMM only the first call allocates.
    Resize resize(std::uint32_t frames, cudaStream_t compute, VramBudgetSource& vram);
    // Frame lending (design §19.3.2, "Frame lending"): a contiguous run of frames serves another use
    // of device memory (the Vision window) and comes back. Lent frames hold no expert and are never
    // resized; decode graphs reach frames through the tables, so lending invalidates none.
    struct FrameLease {
        std::uint32_t first = 0, count = 0;
        DeviceSpan memory; // the run's bytes
        [[nodiscard]] bool valid() const noexcept { return count != 0; }
    };
    // Frames that may be lent now: backed, not lent, above the quarter serving always keeps.
    [[nodiscard]] std::uint32_t lendable() const noexcept;
    // Between rounds (no round in flight): evicts the experts of the cheapest run of `count` frames
    // (avoiding frames that in-flight promotions target while it can), uploads the table on
    // `compute`, and makes `compute` and every `writers` stream wait for each in-flight promotion
    // batch that targets the run.
    FrameLease lend(std::uint32_t count, cudaStream_t compute, std::span<const cudaStream_t> writers);
    // Returns a lease. The caller has ordered `compute` after every access to it; promotions into
    // its frames start after `compute` reaches this point.
    void give_back(FrameLease& lease, cudaStream_t compute);

    // Whether the pool can resize after its first size (VMM).
    [[nodiscard]] bool elastic() const noexcept { return arena_ != nullptr; }
    [[nodiscard]] std::uint32_t frame_limit() const noexcept { return limit_; }
    // Bytes the frames occupy now (whole chunks with VMM).
    [[nodiscard]] std::uint64_t pool_bytes() const noexcept;

    [[nodiscard]] const std::uint8_t* frame_base() const noexcept;
    [[nodiscard]] std::uint64_t frame_stride() const noexcept { return stride_; }
    [[nodiscard]] const std::int32_t* table(std::uint32_t layer) const noexcept;
    [[nodiscard]] std::int32_t* route_log() noexcept { return static_cast<std::int32_t*>(route_device_.p); }
    [[nodiscard]] std::size_t route_stride() const noexcept { return route_stride_; }
    // The last downloaded route log (valid once its round has completed), as after_round reads it.
    [[nodiscard]] const std::int32_t* route_host() const noexcept {
        return static_cast<const std::int32_t*>(route_host_.data());
    }
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
    // Publishes the promotions whose copies have completed.
    void publish_landed();
    // Issues commands_' promotion copies on the copy stream, after the table update on `compute`.
    void issue_loads(cudaStream_t compute);

    const TextConfig& c_;
    std::vector<const std::uint8_t*> banks_;
    std::uint64_t stride_ = 0;
    std::uint32_t frames_ = 0;
    std::uint32_t limit_  = 0;
    std::uint32_t experts_ = 0, layers_ = 0, top_k_ = 0;
    std::size_t route_stride_ = 0;

    std::unique_ptr<VmmArena> arena_;
    DeviceBuffer frame_memory_; // without VMM
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
        struct Load {
            std::uint32_t key, frame;
            std::uint64_t serial;
        };
        std::vector<Load> loads;
    };
    std::vector<Batch> in_flight_;
    std::vector<cudaEvent_t> spare_events_;
    Stats stats_;
};

} // namespace ninfer::models::qwen4_exp
