#pragma once

// The VRAM expert cache of the Qwen4Exp Program (docs/maintainer/qwen3_8-flash-next-design.md
// §9): pools of device frames, each frame holding one routed expert's record, the per-layer frame
// tables the MoE kernels read (frame or -1: read the pinned bank zero-copy), the route log of every
// round, and the LFRU policy (expert_cache::CacheController) that decides promotions and evictions
// at round boundaries.
//
// Pools (size classes; docs/maintainer/qwen3_8-flash-next-design.md §19.3.25). The layers whose
// records share a stride share a pool: its frames have that pitch, and its own LFRU decides among
// its layers' experts. An NVFP4 or block-FP8 artifact has one pool; a GGUF import's layers have up
// to seven record sizes. Frames are sized as average frames (frame_bytes(), the mean of the layers'
// strides): resize(n) gives every layer n / layers frames (one pool: n exactly), so the cache holds
// as many experts as its bytes allow (replay: static classes within 0.14 points of a byte-exact
// LFRU). A layer's table holds frame indices in its pool (frame_base(layer), frame_pitch(layer)).
//
// Ordering. Rounds run on the compute stream and the Program synchronizes at the end of each. At
// the boundary the policy's evictions are applied to the device tables on the compute stream
// before the next round, and the promotions' copies run on a copy stream that first waits for that
// table update, so no kernel can read a frame while it is overwritten. A promoted expert becomes
// visible in its table only after its copy has completed. Expert outputs are placement-invariant
// (design §16.2), so the round a promotion lands in does not change any result.
//
// SSD tier (design §19.3.7, attach_tier). The banks are not in RAM: a key's record is in a frame, in
// a RAM slot of the HostExpertTier, or only in the artifact. The residency then keeps a per-layer
// table of host record pointers (null: SSD-only) beside the frame table, uploaded with it before
// each round after the tier's boundary; promotions copy from RAM slots, pinned in the tier while
// queued and while the copy runs; only keys with a host copy are admitted (SSD-only experts reach
// VRAM through landing); each round's routed keys feed the tier's decayed LFU. A VRAM victim with a
// host copy keeps it (a shadow becomes a resident); one without is demoted when it outranks the
// RAM victim and the boundary's allowance lasts (32 per decode boundary): its frame stays held and
// readable while a D2H stream copies it to a RAM slot, and the next before_round after the copy
// publishes the slot and frees the frame. Evictions the gate refuses stop that layer's admissions.
// Lending and resizing finish pending demotions first and drop, not demote, their victims.
//
// Size. Each pool's frames live in a VMM arena whose base never moves (design §19.3.7): resize()
// backs more frames chunk by chunk, or releases the top chunks after evicting the lowest-score
// experts and moving the survivors below the new top, so kernels and captured graphs keep their
// addresses. Without VMM the frames are one allocation per pool of the first size.

#include "core/arena.h"
#include "core/device.h"
#include "core/vmm_arena.h"
#include "core/vram_budget.h"
#include "runtime/expert_offload/expert_cache.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace infernix::runtime::expert_offload {

class HostExpertTier;

// The routed experts a residency caches: its offloaded layers, each layer's routed experts and the
// experts one column routes.
struct ExpertLayout {
    std::uint32_t layers  = 0;
    std::uint32_t experts = 0;
    std::uint32_t top_k   = 0;
};

class ExpertResidency {
public:
    struct Stats {
        std::uint64_t routed     = 0; // routed (layer, expert) uses, summed over rounds' groups
        std::uint64_t hits       = 0; // of those, resident in a frame at the round's start
        std::uint64_t decode_routed = 0; // routed and hits of decode and verification rounds only
        std::uint64_t decode_hits   = 0;
        std::uint64_t promotions = 0; // expert copies issued
        std::uint64_t lent_frames    = 0; // frames lent now (in their pools' frames)
        std::uint64_t lend_evictions   = 0; // lowest-score experts evicted to lend frames, since start
        std::uint64_t lend_relocations = 0; // experts moved out of lent runs by device copies, since start
        std::uint64_t seeded         = 0; // experts loaded by the warm start
        std::uint64_t landed         = 0; // staged misses landed in reserved frames and adopted (S4)
        std::uint64_t demotions      = 0; // SSD tier: VRAM victims copied to RAM (T4)
        std::uint64_t unpinned_promotions = 0; // SSD tier: promotions of unpinned records (bounced)
    };
    // Landing frames per layer and round (design §19.3.5 S4): free frames are reserved before a
    // decode or verification round, the round's forked MoE calls copy their first staged misses
    // into them instead of staging slots, and the round's settle adopts them as resident.
    static constexpr std::uint32_t kLandingSlots = 16;
    // LFRU counts halve every this many ticks (layer calls, ~170 plain rounds; design §9.3): on
    // greedy decode traces of 14 mixed requests, 9-12 % fewer misses than never halving. A pool
    // ticks once per call of its layers, so its period is this share of its layers.
    static constexpr std::uint32_t kLfruHalvingPeriod = 8192;

    using SavedState = expert_cache::SavedState;
    [[nodiscard]] SavedState saved_state() const;
    // Before the first round: seeds the counts (each capped at `count_cap`, so a stale expert
    // yields to new uses) and loads at most `max_keys` of the ranked keys, best first, into free
    // frames (each pool its share), waiting for the copies. Returns the number loaded.
    std::uint32_t warm_start(const SavedState& state, std::uint32_t count_cap, std::uint32_t max_keys,
                             cudaStream_t compute);

    // banks[l]: layer l's pinned host records (null entries with a tier); host_strides[l]: their
    // record stride, each at most record_stride (empty: every layer's is record_stride). The layers
    // of one stride share a pool of frames of that pitch. max_columns: the most columns one round
    // routes. The cache starts with no frames: resize() backs them.
    ExpertResidency(const ExpertLayout& layout, std::vector<const std::uint8_t*> banks, std::uint64_t record_stride,
                    std::int32_t max_columns, int device, std::vector<std::uint64_t> host_strides = {});
    ~ExpertResidency();
    ExpertResidency(const ExpertResidency&)            = delete;
    ExpertResidency& operator=(const ExpertResidency&) = delete;

    // Device bytes besides the frames: the frame tables and the route log.
    [[nodiscard]] static std::uint64_t table_bytes(const ExpertLayout& layout, std::int32_t max_columns) noexcept;
    // Frames the cache uses at most: every routed expert but one.
    [[nodiscard]] static std::uint32_t max_frames(const ExpertLayout& layout) noexcept;
    // The average frame's bytes for these strides (the sizing unit of frames(), resize() and the
    // device plan): the mean record stride, rounded up.
    [[nodiscard]] static std::uint64_t average_frame_bytes(std::span<const std::uint64_t> strides) noexcept;
    // Bytes a pool's frames are mapped in (whole chunks); a grow or shrink moves in these steps.
    static constexpr std::size_t kChunkBytes = 64ULL << 20;

    struct Resize {
        std::uint32_t frames  = 0;     // frames backed afterwards, over every pool
        std::uint64_t spilled = 0;     // bytes of a chunk the driver placed in system memory (given back)
        bool refused          = false; // the device had no memory for the next chunk
    };
    // Backs `frames` frames over the pools (every layer frames / layers; clamped to frame_limit());
    // unchanged while frames are lent. Call only between rounds, with nothing in flight on the
    // compute stream: promotion copies are waited for, then a shrink evicts each pool's
    // lowest-score experts, moves the survivors above its new top below it on `compute` and
    // releases the top chunks; a grow maps chunks one at a time, stops at the first the driver
    // refuses or places in system memory (`vram` tells), and fills the new frames from the queue.
    // Without VMM only the first call allocates.
    Resize resize(std::uint32_t frames, cudaStream_t compute, VramBudgetSource& vram);
    // Frame lending (design §19.3.2, "Frame lending"): a contiguous run of one pool's frames serves
    // another use of device memory (the prefill stream's ring and areas, the Vision window) and
    // comes back. Lent frames hold no expert and are never resized; decode graphs reach frames
    // through the tables, so lending invalidates none.
    struct FrameLease {
        std::uint32_t pool = 0, first = 0, count = 0;
        DeviceSpan memory; // the run's bytes (whole frames of its pool)
        [[nodiscard]] bool valid() const noexcept { return count != 0; }
    };
    // Bytes that may be lent now as one run: the largest of the pools' runs (backed, not lent,
    // above the quarter each pool always keeps).
    [[nodiscard]] std::uint64_t lendable_bytes() const noexcept;
    // The pitch of the pool a lease comes from (the one with the most lendable bytes), and its layers.
    [[nodiscard]] std::uint64_t lend_pitch() const noexcept;
    [[nodiscard]] std::uint32_t lend_layers() const noexcept;
    // What serving never gives up aside, the most bytes that pool lends with no lease out.
    [[nodiscard]] std::uint64_t lend_capacity_bytes() const noexcept;
    // The layers whose experts a lease's frames held (its pool's).
    [[nodiscard]] std::uint32_t lease_layers(const FrameLease& lease) const noexcept {
        return lease.valid() ? static_cast<std::uint32_t>(pools_[lease.pool].layers.size()) : 0U;
    }
    // Between rounds (no round in flight): evicts the experts of the lending pool's cheapest run of
    // frames covering `bytes` (avoiding frames that in-flight promotions target while it can),
    // uploads the table on `compute`, and makes `compute` and every `writers` stream wait for each
    // in-flight promotion batch that targets the run.
    FrameLease lend(std::uint64_t bytes, cudaStream_t compute, std::span<const cudaStream_t> writers);
    // Splits a lease after its first `head_bytes` (rounded up to whole frames): the lease keeps the
    // head, the tail is returned (invalid when nothing is left).
    [[nodiscard]] FrameLease split(FrameLease& lease, std::uint64_t head_bytes) const;
    // Returns a lease. The caller has ordered `compute` after every access to it; promotions into
    // its frames start after `compute` reaches this point.
    void give_back(FrameLease& lease, cudaStream_t compute);

    // SSD tier mode: records come from `tier` (owned by the caller, outliving this), not the banks.
    // Before the first round and before warm_start.
    void attach_tier(HostExpertTier* tier);
    [[nodiscard]] bool tiered() const noexcept { return tier_ != nullptr; }
    // Layer l's host record pointers (device [E]; null entries: SSD-only), or null without a tier.
    [[nodiscard]] const std::uint8_t* const* record_table(std::uint32_t layer) const noexcept;

    // Whether the pools can resize after their first size (VMM).
    [[nodiscard]] bool elastic() const noexcept { return !pools_.empty() && pools_.front().arena != nullptr; }
    [[nodiscard]] std::uint32_t frame_limit() const noexcept;
    // Bytes the frames occupy now (whole chunks with VMM), over every pool.
    [[nodiscard]] std::uint64_t pool_bytes() const noexcept;
    [[nodiscard]] std::size_t pools() const noexcept { return pools_.size(); }

    // Where layer `layer`'s frames are: frame f at frame_base(layer) + f * frame_pitch(layer).
    [[nodiscard]] const std::uint8_t* frame_base(std::uint32_t layer) const noexcept;
    [[nodiscard]] std::uint64_t frame_pitch(std::uint32_t layer) const noexcept;
    // The average frame's bytes (average_frame_bytes of the layers' strides).
    [[nodiscard]] std::uint64_t frame_bytes() const noexcept { return frame_bytes_; }
    [[nodiscard]] const std::int32_t* table(std::uint32_t layer) const noexcept;
    // The host copy of every layer's table (I32 [layers][experts]), equal to the device tables
    // once before_round has uploaded them.
    [[nodiscard]] const std::int32_t* host_table() const noexcept {
        return static_cast<const std::int32_t*>(table_host_.data());
    }
    [[nodiscard]] std::int32_t* route_log() noexcept { return static_cast<std::int32_t*>(route_device_.p); }
    [[nodiscard]] std::size_t route_stride() const noexcept { return route_stride_; }
    // The last downloaded route log (valid once its round has completed), as after_round reads it.
    [[nodiscard]] const std::int32_t* route_host() const noexcept {
        return static_cast<const std::int32_t*>(route_host_.data());
    }
    // Frames backed now, over every pool.
    [[nodiscard]] std::uint32_t frames() const noexcept;

    // Before a round: publishes the loads that completed since the last round, returns the
    // reservations of a round that was never settled, and with `landing` (decode and verification
    // rounds) reserves min(kLandingSlots, free frames / layers) landing frames per layer (each from
    // its pool) while the policies have room, uploading the landing table (frame or -1) and clearing
    // the landed log. With `walking` (another lane's round between a layer walk's steps) the walk
    // keeps its stream pins and the tier boundary allows no demotions, as at the walk's own step
    // boundaries (tier_step).
    // `stream_slots`: a layer walk's span begins, whose stream lends that many RAM slots for its
    // reads of SSD-only experts (HostExpertTier::lend_stream); a boundary outside a span returns them.
    void before_round(cudaStream_t compute, bool landing = false, bool walking = false, std::uint32_t stream_slots = 0);
    // A layer walk's step boundary, with the device idle, before the step's calls: the SSD tier closes
    // its round (or another lane's round already closed it) and opens the next (landings admitted, no
    // demotions while the walk holds lent frames) and the record pointers it changed are uploaded. A
    // walk's span is one residency round, but a tier round's tickets cover about one pass of the
    // experts: a step's layers once per chunk of the span. The tier answers the device's fetch
    // requests only while its round is open.
    void tier_step(cudaStream_t compute);
    // The landing table and landed log, device I32 [layers][kLandingSlots] at fixed addresses.
    [[nodiscard]] const std::int32_t* landing_table() const noexcept {
        return static_cast<const std::int32_t*>(landing_device_.p);
    }
    [[nodiscard]] std::int32_t* landed_log() noexcept { return static_cast<std::int32_t*>(landed_device_.p); }
    // After the round's kernels are enqueued: downloads its route log (and landed log) with the round.
    void enqueue_route_download(cudaStream_t compute, std::int32_t columns);
    // After the round completed (the compute stream was synchronized): runs the policies over the
    // round's routes and issues their evictions and promotions. `per_layer_budget` caps the
    // promotions one layer's routes may start. A non-empty `live` (one flag per column) credits
    // only the flagged columns' experts: a verification round's rejected draft columns neither
    // count as uses nor start promotions (design section 11.3). A layer's landed experts are adopted
    // first and use up its budget: max(0, per_layer_budget - landed) promotions remain.
    void after_round(cudaStream_t compute, std::int32_t columns, std::size_t per_layer_budget,
                     std::span<const std::uint8_t> live = {});
    // At a prompt's end (design §19.3.27): every key with uses[key] > 0 (the prompt's tokens that
    // routed to it, [layers][experts]) gains `scale * uses[key]` of LFRU count, used now. Nothing is
    // copied: the credit ranks the decode's admissions and evictions.
    void seed_prompt(std::span<const std::uint32_t> uses, double scale);

    [[nodiscard]] const Stats& stats() const noexcept { return stats_; }
    // The layer split of stats().routed and stats().hits: cumulative routed (layer, expert) uses
    // and resident-at-round-start hits per layer, indexed by layer. Exact counters, fixed at
    // construction, so the decode path only adds to them.
    [[nodiscard]] std::span<const std::uint64_t> routed_by_layer() const noexcept { return routed_by_layer_; }
    [[nodiscard]] std::span<const std::uint64_t> hits_by_layer() const noexcept { return hits_by_layer_; }

private:
    using Command = expert_cache::CacheController::Command;
    struct Pool {
        std::uint64_t pitch = 0;           // the layers' record stride
        std::vector<std::uint32_t> layers; // layers whose experts it holds
        std::uint32_t frames = 0, limit = 0;
        std::unique_ptr<VmmArena> arena;
        DeviceBuffer memory; // without VMM
        std::unique_ptr<expert_cache::CacheController> controller;
        std::vector<Command> commands; // scratch: the pool's commands of one operation
        [[nodiscard]] std::uint8_t* base() const noexcept;
    };

    [[nodiscard]] Pool& pool_of(std::uint32_t key) noexcept { return pools_[pool_of_layer_[key / experts_]]; }
    [[nodiscard]] const Pool& pool_of(std::uint32_t key) const noexcept { return pools_[pool_of_layer_[key / experts_]]; }
    // Frames per pool for `frames` average frames.
    [[nodiscard]] std::vector<std::uint32_t> split_frames(std::uint32_t frames) const;
    // Resizes one pool to `frames`; returns false when a grow was refused or spilled (out.*).
    void resize_pool(Pool& pool, std::uint32_t frames, cudaStream_t compute, VramBudgetSource& vram, Resize& out);
    // The pool a lease comes from: the most lendable bytes.
    [[nodiscard]] std::size_t lend_pool() const noexcept;
    [[nodiscard]] std::uint32_t pool_lendable(const Pool& pool) const noexcept;
    void upload_table(cudaStream_t compute);
    // Returns every reserved landing frame to its pool.
    void release_reservations();
    // Publishes the promotions whose copies have completed.
    void publish_landed();
    // Issues the pending loads (load_commands_) on the copy stream, after the table update on `compute`.
    void issue_loads(cudaStream_t compute);
    // Moves a pool's kCopy commands to load_commands_ (keys are global, so the pool is the key's).
    void take_loads(std::vector<Command>& commands);
    // Tier mode: VRAM evictions among a pool's commands (T3/T5), and Queue pins of the controllers' queues.
    void report_evictions(const std::vector<Command>& commands);
    // Applies the pool's commands' relocations and Absent writes to table_host_ and moves the relocated
    // frames on `compute` in one launch; returns the number moved. The caller uploads the table.
    std::uint32_t apply_moves(Pool& pool, cudaStream_t compute);
    void sync_queue_pins();
    // Tier mode, after_round before on_quiescent: each evicted expert is kept in RAM (T3), demoted
    // (T4: frame held, D2H) or dropped (T5).
    void demote_or_drop(Pool& pool, const std::vector<Command>& commands, std::int32_t* table);
    // Tier mode: completes the demotions whose copies finished (all of them, waiting, with `wait`):
    // publishes their slots (T7), frees their frames and loads queued experts into them.
    void finish_demotions(cudaStream_t compute, bool wait);

    ExpertLayout layout_;
    std::vector<const std::uint8_t*> banks_;
    std::vector<std::uint64_t> host_strides_; // per layer: its bank's record stride
    std::uint64_t stride_      = 0;           // the largest record stride (the tier's slot pitch)
    std::uint64_t frame_bytes_ = 0;
    std::uint32_t experts_ = 0, layers_ = 0, top_k_ = 0;
    std::size_t route_stride_ = 0;

    std::vector<Pool> pools_;
    std::vector<std::uint32_t> pool_of_layer_;
    DeviceBuffer table_device_;
    PinnedHostBuffer table_host_{1};
    DeviceBuffer route_device_;
    PinnedHostBuffer route_host_{1};
    bool table_dirty_ = true;
    DeviceBuffer landing_device_;
    PinnedHostBuffer landing_host_{1};
    DeviceBuffer landed_device_;
    PinnedHostBuffer landed_host_{1};
    std::vector<std::uint32_t> reserved_; // [layer][landing_per_layer_] reserved frames (in the layer's pool)
    std::uint32_t landing_per_layer_ = 0;
    bool landing_uploaded_           = false; // the device landing table holds frames
    bool decode_round_               = false; // the round's before_round had `landing`

    std::vector<Command> load_commands_; // kCopy commands of every pool, issued together
    std::vector<std::uint32_t> move_to_, move_from_; // a lend's relocations: destination and source frames
    std::vector<std::uint32_t> group_;
    std::vector<double> amounts_; // seed_prompt's per-key credits of one layer
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
    // Promotions per completion event (~6 ms of link at x8): a large promotion publishes in steps.
    static constexpr std::size_t kLoadsPerEvent = 64;
    Stats stats_;
    std::vector<std::uint64_t> routed_by_layer_, hits_by_layer_; // per layer, indexed by layer

    HostExpertTier* tier_ = nullptr;
    expert_cache::LfruPolicy::Admissible admissible_;
    DeviceBuffer records_device_;       // tier: const uint8_t* [layers][experts]
    PinnedHostBuffer records_host_{1};
    std::vector<std::uint32_t> dirty_;  // keys whose host pointer changed (tier.take_dirty)
    std::vector<std::uint32_t> uses_;   // the round's routed keys (the tier's LFU)
    std::vector<std::uint8_t> queue_pinned_;     // per key: holds the tier's Queue pin
    std::vector<std::uint32_t> queue_pinned_keys_;
    std::vector<std::uint8_t> queue_mark_;
    std::vector<std::uint16_t> copies_in_flight_; // per key: promotion copies holding its H2D pin
    expert_cache::LfruPolicy::Admissible evictable_; // the tier's eviction gate
    cudaStream_t d2h_stream_ = nullptr;
    struct Demotion {
        std::uint32_t key, frame, slot;
        cudaEvent_t done;
    };
    std::vector<Demotion> demotions_;
    std::vector<std::uint8_t> demoting_; // per key
    // Promotions of unpinned records go through the tier's bounce slots, in two halves: the CPU
    // fills one while the link drains the other (bounced_[h]: the half's last copies, recorded).
    cudaEvent_t bounced_[2]  = {nullptr, nullptr};
    bool bounce_pending_[2]  = {false, false};
    std::uint32_t bounce_half_ = 0;
};

} // namespace infernix::runtime::expert_offload
