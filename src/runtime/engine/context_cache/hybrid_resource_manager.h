#pragma once

#include "infernix/types.h"
#include "runtime/contract/resources.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace infernix::runtime {

enum class ReclaimProgress : std::uint8_t { Blocked, Changed, Transferring };

// Engine-side driver of the prefix cache (docs/maintainer/hybrid-prefix-cache-spec.md §8).
// Retention policy lives in the Program's prefix index: sources are quotes of its block tree and
// reclamation evicts its cached Device blocks. A pause publishes the lane's state at its frontier;
// the paused request resumes from the deepest state the tree then holds on its ledger (that
// snapshot, while it is cached), by Replay past it.
template <class Model>
class HybridResourceManager {
public:
    using Program = typename Model::Program;
    using Base    = typename Model::RequestBasePlan;
    using Source  = typename Model::SourceCandidate;
    using Resume  = typename Model::ResumeState;

    // The tree's quotes in preference order (its chosen path, then a root start; a root start
    // alone without a prefix cache). A paused request (`resume`) is quoted over its ledger, never
    // past its paused frontier. Empty while a prefilling sibling is about to publish the snapshot
    // this request should resume from.
    [[nodiscard]] std::vector<Source> candidates(Program& program, const Base& base, const Resume* resume) {
        return program.hybrid_sources(base, resume);
    }

    [[nodiscard]] ReclaimProgress reclaim(Program& program, ContextResourceUsage shortage) {
        if (program.has_context_transaction()) { return ReclaimProgress::Transferring; }
        return program.hybrid_reclaim(shortage) ? ReclaimProgress::Changed
                                                : ReclaimProgress::Blocked;
    }

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
        // Publication refusals (Qwen3.5's Device snapshot slots; other caches have none).
        if constexpr (requires { hybrid.device_slot_refusals; }) {
            out.hybrid_taps_skipped_no_slot       = hybrid.taps_skipped_no_slot;
            out.hybrid_endpoints_skipped          = hybrid.endpoints_skipped;
            out.hybrid_host_image_write_refusals  = hybrid.host_image_write_refusals;
            out.hybrid_host_block_write_refusals  = hybrid.host_block_write_refusals;
            out.hybrid_device_slot_evictions      = hybrid.device_slot_evictions;
            out.hybrid_device_slot_refusals       = hybrid.device_slot_refusals;
            out.hybrid_free_device_snapshot_slots = hybrid.free_device_snapshot_slots;
            out.hybrid_host_write_copies          = hybrid.host_write_copies;
            out.hybrid_host_restore_copies        = hybrid.host_restore_copies;
        }
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
        out.hybrid_tool_call_holds         = hybrid.tool_call_holds;
        const auto& ssd                     = hybrid.ssd;
        out.hybrid_ssd_capacity_bytes       = ssd.capacity_bytes;
        out.hybrid_ssd_used_bytes           = ssd.used_bytes;
        out.hybrid_ssd_only_snapshots       = ssd.ssd_only_snapshots;
        out.hybrid_ssd_write_bytes          = ssd.write_bytes;
        out.hybrid_ssd_read_bytes           = ssd.read_bytes;
        out.hybrid_ssd_image_writes         = ssd.image_writes;
        out.hybrid_ssd_block_writes         = ssd.block_writes;
        out.hybrid_ssd_host_image_demotions = ssd.host_image_demotions;
        out.hybrid_ssd_host_block_demotions = ssd.host_block_demotions;
        out.hybrid_ssd_promotions           = ssd.promotions;
        out.hybrid_ssd_promotion_room_waits = ssd.promotion_room_waits;
        out.hybrid_ssd_snapshot_evictions   = ssd.snapshot_evictions;
        out.hybrid_ssd_failures             = ssd.write_failures + ssd.read_failures;
        out.hybrid_ssd_read_bytes_per_second = ssd.read_bytes_per_second;
        out.hybrid_ssd_budget_bytes_per_hour  = ssd.budget_bytes_per_hour;
        out.hybrid_ssd_budget_available_bytes = ssd.budget_available_bytes;
        out.hybrid_ssd_declined_snapshots     = ssd.declined_snapshots;
        out.hybrid_ssd_packed_images          = ssd.packed_images;
        out.hybrid_ssd_packed_saved_bytes     = ssd.packed_saved_bytes;
        out.hybrid_ssd_policy_nanoseconds     = ssd.policy_nanoseconds;
        for (std::size_t kind = 0; kind < ssd.by_kind.size(); ++kind) {
            const auto& from = ssd.by_kind[kind];
            auto& to         = out.hybrid_ssd_by_kind[kind];
            to.writes            = from.writes;
            to.written_bytes     = from.written_slabs * ssd.slab_bytes;
            to.reads             = from.reads;
            to.first_reads       = from.first_reads;
            to.superseded_unread = from.superseded_unread;
            to.dropped_unread    = from.dropped_unread;
            to.host_losses       = from.host_losses;
        }
    }

    // Holds the sources of the requests waiting for admission, in admission order
    // (hybrid-prefix-cache-spec §9.6). `ids` names `queue`'s requests and `resumes` their paused
    // states (null for a fresh request); matching walks every queued prompt's path, so the holds
    // are recomputed only when the queue or the tree's snapshots changed.
    void hold_queue(Program& program, std::span<const std::uint64_t> ids,
                    std::span<const Base* const> queue, std::span<const Resume* const> resumes) {
        const std::uint64_t epoch = program.hybrid_cache_epoch();
        if (hold_valid_ && epoch == hold_epoch_ && !program.hybrid_holds_stale() &&
            std::equal(ids.begin(), ids.end(), hold_ids_.begin(), hold_ids_.end())) {
            return;
        }
        program.hybrid_hold_queue(queue, resumes);
        hold_ids_.assign(ids.begin(), ids.end());
        hold_epoch_ = epoch;
        hold_valid_ = true;
    }

    // A tool-call hold (HybridPrefixCacheOptions::tool_call_hold): the request finished last
    // ended with a tool call, so its client returns with the next turn within the tool's run time.
    void hold_continuation(Program& program, std::chrono::steady_clock::time_point deadline) {
        program.hybrid_hold_continuation(deadline);
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
