#pragma once

#include "infernix/types.h"
#include "runtime/contract/resources.h"
#include "runtime/engine/context_cache/context_cost.h"
#include "runtime/engine/context_cache/types.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace infernix::runtime {

// Engine-side driver of the hybrid prefix cache (docs/maintainer/hybrid-prefix-cache-spec.md §8).
// It presents the ResourceManager surface the Engine core uses, so scheduling, binding,
// preemption and commit orchestration are shared by both cache modes. Retention policy lives in
// the Program's prefix index: sources are quotes of its block tree, reclamation evicts its cached
// Device blocks, and no checkpoint is ever published, captured or kept for a paused request (a
// paused request recovers by Replay, from whatever the tree then holds).
template <class Model>
class HybridResourceManager {
public:
    using Program    = typename Model::Program;
    using Base       = typename Model::RequestBasePlan;
    using Handle     = typename Model::CheckpointHandle;
    using OwnerToken = ContinuationOwnerToken;
    using Source     = typename Model::SourceCandidate;

    struct Admission {
        CacheRetentionPriority priority;
        CacheRetentionPriority demand;
        const Base* base       = nullptr;
        std::uint32_t frontier = 0;
        std::optional<Handle> checkpoint;
        std::vector<Handle> sources;
    };

    struct ReclaimCursor {
        std::optional<Admission> admission;
        ReclaimRights rights;
    };

    struct SourceChoice {
        Source source;
        OwnerToken owner           = 0;
        std::uint64_t shared_entry = 0;
        bool take_over             = false;
        std::optional<OwnerToken> resume_owner;
        bool session_hint               = false;
        std::uint64_t publication_order = 0;
        std::uint64_t ordinal           = 0;
    };

    // The manager serves every admission of its core, whether or not a prefix cache exists: with
    // the cache off the Program offers the root alone.
    HybridResourceManager(bool /*enabled*/, ContextMachineCostModel /*costs*/) {}

    // The tree's quotes in preference order (its chosen path, then a root start). Empty while a
    // prefilling sibling is about to publish the snapshot this request should resume from.
    [[nodiscard]] std::vector<SourceChoice> candidates(Program& program, const Base& base,
                                                       std::uint32_t maximum_frontier = UINT32_MAX,
                                                       std::optional<OwnerToken> = std::nullopt,
                                                       std::uint64_t             = 0,
                                                       std::uint64_t = UINT64_MAX) {
        std::vector<SourceChoice> out;
        for (auto& source : program.hybrid_sources(base, maximum_frontier)) {
            out.push_back(SourceChoice{.source = std::move(source)});
        }
        return out;
    }

    [[nodiscard]] bool prepare_source(Program&, const Base&, SourceChoice&, std::uint64_t = 0,
                                      std::optional<ReclaimRights> = std::nullopt) {
        return true;
    }

    // A quote holds no checkpoint, so a waiting request retains nothing: when a lane frees, its
    // source is quoted again from whatever the tree then holds.
    void retain_source(Program&, std::uint64_t, const SourceChoice&) {}

    [[nodiscard]] std::optional<SourceChoice> retained_source(std::uint64_t) const {
        return std::nullopt;
    }

    [[nodiscard]] bool has_source_record(std::uint64_t) const { return false; }

    [[nodiscard]] bool can_transfer_private(OwnerToken, std::uint64_t) const { return true; }

    [[nodiscard]] std::uint32_t source_revocations(std::uint64_t) const { return 0; }

    void binding_started(std::uint64_t, std::span<const Handle> retired, std::optional<Handle>) {
        if (!retired.empty()) {
            throw std::logic_error("hybrid binding retired checkpoint recovery points");
        }
    }

    void release_source(Program&, std::uint64_t) {}

    [[nodiscard]] OwnerToken adopt(Program&, const SourceChoice&, const Base&, std::uint64_t,
                                   std::span<const Handle> carried, std::span<const Handle>) {
        if (!carried.empty()) {
            throw std::logic_error("hybrid binding carried checkpoint recovery points");
        }
        return 0;
    }

    void publish(Program&, OwnerToken, Handle) {
        throw std::logic_error("hybrid mode published a checkpoint");
    }

    void finish(Program&, OwnerToken, const Base&, std::uint64_t) {}

    void abandon(Program&, OwnerToken) {}

    [[nodiscard]] std::vector<Handle> points(OwnerToken) const { return {}; }

    bool recycle_input(Program&, OwnerToken) { return false; }

    void observe_committed(const Base&, std::uint64_t, std::uint32_t, std::uint32_t) {}

    [[nodiscard]] Admission capture_admission(Program&, OwnerToken, const Base& base,
                                              std::uint32_t frontier, bool = true) const {
        return {.base = &base, .frontier = frontier};
    }

    [[nodiscard]] ReclaimCursor begin_reclaim(Program&,
                                              std::optional<Admission> admission = std::nullopt,
                                              ReclaimRights rights               = {}) {
        ReclaimCursor cursor;
        cursor.admission = admission;
        cursor.rights    = rights;
        return cursor;
    }

    // The Program keeps no Host context arena in hybrid mode, so nothing is ever written to it.
    [[nodiscard]] std::optional<std::vector<Handle>>
    host_victims(Program&, std::size_t, std::optional<Handle>, std::span<const Handle> = {},
                 std::span<const Handle> = {}, std::optional<Admission> = std::nullopt,
                 const ReclaimCursor* = nullptr, ReclaimRights = {}) {
        return std::nullopt;
    }

    void commit_host_victims(Program&, std::span<const Handle>, ReclaimCursor&) {}

    bool erase(Program&, Handle, ReclaimRights = {}) { return false; }

    [[nodiscard]] ReclaimProgress reclaim(Program& program, ContextResourceUsage shortage,
                                          std::span<const Handle> = {}) {
        if (program.has_context_transaction()) { return ReclaimProgress::Transferring; }
        return program.hybrid_reclaim(shortage) ? ReclaimProgress::Changed
                                                : ReclaimProgress::Blocked;
    }

    [[nodiscard]] ReclaimProgress reclaim(Program& program, ContextResourceUsage shortage,
                                          std::span<const Handle> excluded, ReclaimCursor&) {
        return reclaim(program, shortage, excluded);
    }

    void release_all(Program&) noexcept {}

    // A blocked FIFO head's Host-only blocks are copied into spare Device cache while it waits
    // (hybrid-prefix-cache-spec §6.6). Matching a long prompt walks its path, so a new attempt
    // needs a new head, progress on the last attempt, or more room.
    void prefetch_blocked_head(Program& program, const Base& base,
                               std::uint64_t publication_order) {
        if (program.has_context_transaction()) { return; }
        if (publication_order == prefetch_order_ && !prefetch_retry_ &&
            program.hybrid_prefetch_room() <= prefetch_room_) {
            return;
        }
        const std::optional<std::uint32_t> started = program.hybrid_prefetch(base);
        prefetch_order_                            = publication_order;
        // A prefetch still in flight, or one that copied blocks, may leave more to copy.
        prefetch_retry_ = !started || *started != 0;
        prefetch_room_  = program.hybrid_prefetch_room();
    }

    void populate_runtime_stats(Program& program, RuntimeStats& out) const noexcept {
        const auto hybrid              = program.hybrid_stats();
        out.hybrid_cached_blocks       = hybrid.device_resident_blocks;
        out.hybrid_evictable_blocks    = hybrid.device_evictable_blocks;
        out.hybrid_tree_blocks         = hybrid.nodes;
        out.hybrid_snapshots           = hybrid.snapshots;
        out.hybrid_host_capacity_bytes = hybrid.host_slab_bytes * hybrid.host_slabs;
        out.hybrid_host_used_bytes =
            hybrid.host_slab_bytes * (hybrid.host_slabs - hybrid.host_free_slabs);
        out.hybrid_snapshot_hits           = hybrid.snapshot_hits;
        out.hybrid_reused_tokens           = hybrid.reused_tokens;
        out.hybrid_blocks_inserted         = hybrid.blocks_inserted;
        out.hybrid_blocks_reattached       = hybrid.blocks_reattached;
        out.hybrid_blocks_duplicate        = hybrid.blocks_duplicate;
        out.hybrid_taps_created            = hybrid.taps_created;
        out.hybrid_taps_skipped            = hybrid.taps_skipped;
        out.hybrid_endpoints_created       = hybrid.endpoints_created;
        out.hybrid_host_image_writes       = hybrid.host_image_writes;
        out.hybrid_host_block_writes       = hybrid.host_block_writes;
        out.hybrid_host_image_restores     = hybrid.host_image_restores;
        out.hybrid_host_block_restores     = hybrid.host_block_restores;
        out.hybrid_prefetched_blocks       = hybrid.prefetched_blocks;
        out.hybrid_host_write_bytes        = hybrid.host_write_bytes;
        out.hybrid_host_restore_bytes      = hybrid.host_restore_bytes;
        out.hybrid_evicted_blocks          = hybrid.evicted_blocks;
        out.hybrid_host_snapshot_evictions = hybrid.host_snapshot_evictions;
        out.hybrid_host_dead_reclaims      = hybrid.host_dead_reclaims;
        out.hybrid_unbacked_node_losses    = hybrid.unbacked_node_losses;
        out.hybrid_held_snapshots          = hybrid.held_snapshots;
        out.hybrid_held_device_evictions   = hybrid.held_device_evictions;
        out.hybrid_held_snapshot_losses    = hybrid.held_snapshot_losses;
        out.hybrid_held_host_refusals      = hybrid.held_host_refusals;
    }

    // Holds the sources of the requests waiting for admission, in admission order
    // (hybrid-prefix-cache-spec §9.6). `ids` names `queue`'s requests; matching walks every
    // queued prompt's path, so the holds are recomputed only when the queue or the tree's
    // snapshots changed.
    void hold_queue(Program& program, std::span<const std::uint64_t> ids,
                    std::span<const Base* const> queue) {
        const std::uint64_t epoch = program.hybrid_cache_epoch();
        if (hold_valid_ && epoch == hold_epoch_ &&
            std::equal(ids.begin(), ids.end(), hold_ids_.begin(), hold_ids_.end())) {
            return;
        }
        program.hybrid_hold_queue(queue);
        hold_ids_.assign(ids.begin(), ids.end());
        hold_epoch_ = epoch;
        hold_valid_ = true;
    }

private:
    // The last prefetch attempt: the head it served, whether to try again and the room it left.
    std::uint64_t prefetch_order_ = 0;
    bool prefetch_retry_          = false;
    std::uint32_t prefetch_room_  = 0;
    // The queue and snapshot epoch the current holds were computed for.
    std::vector<std::uint64_t> hold_ids_;
    std::uint64_t hold_epoch_ = 0;
    bool hold_valid_          = false;
};

} // namespace infernix::runtime
