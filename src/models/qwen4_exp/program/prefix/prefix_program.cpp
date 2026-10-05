// The Program side of Qwen4Exp's prefix cache (docs/maintainer/qwen3_8-flash-next-design.md §19.3.1):
// admission against the index, activation with per-layer restores, block publication, Host-born taps
// and endpoints, and release. The binding (prefix_cache.h) moves the bytes.

#include "models/qwen4_exp/program/program_impl.h"
#include "models/qwen3_5/program/prefix/block_keys.h"
#include "runtime/prefix_cache/block_hash.h"

#include <algorithm>

namespace ninfer::models::qwen4_exp::detail {

namespace pc = runtime::prefix_cache;

namespace {

constexpr std::uint32_t kBlock = pc::kBlockTokens;

} // namespace

// ---- construction ------------------------------------------------------------------------------------

void ProgramImpl::create_prefix_cache() {
    if (!options_.prefix_cache) { return; }
    const auto lanes = static_cast<std::uint32_t>(options_.max_concurrency);
    image_layout_    = prefix::plan_state_image(prefix::state_image_spec(c_, mtp_));
    lane_image_      = std::make_unique<prefix::LaneStateImage>(
        image_layout_,
        prefix::LaneStateBuffers{.gdn       = gdn_.get(),
                                 .lanes     = lanes,
                                 .ple       = static_cast<std::byte*>(ple_backing_.p),
                                 .tails     = static_cast<std::byte*>(tails_backing_.p),
                                 .mtp_saved = mtp_ ? static_cast<std::byte*>(mtp_saved_.p) : nullptr});
    prefix::PrefixCacheConfig config;
    config.index.cost = options_.prefix_cost;
    config.taps       = options_.prefix_taps;
    config.kv_pages   = kv_pages_;
    prefix_           = std::make_unique<prefix::PrefixCache>(config, *pool_, kv_geometry_, *lane_image_,
                                                            options_.prefix_host_bytes);
    if (mtp_) {
        // The MR3 flush cell's inputs: token, cell, slot and table row.
        flush_host_   = PinnedHostBuffer(4 * sizeof(std::int32_t));
        flush_device_ = DeviceBuffer(4 * sizeof(std::int32_t));
        CUDA_CHECK(cudaEventCreateWithFlags(&flush_uploaded_, cudaEventDisableTiming));
    }
}

void ProgramImpl::pc_make_room(std::uint32_t pages) {
    pc::PrefixCacheIndex& cache = prefix_->index();
    while (pool_->available_pages() < pages) {
        if (cache.evict_device_blocks(pages - pool_->available_pages()) == 0) { return; }
    }
}

// MR3: one kv-only MTP cell for the pending position (cell state_tokens - 1 from the saved residual
// and token history[state_tokens]), the cell the next round's step 0 or catch-up would write.
void ProgramImpl::mtp_flush_cell(Lane& lane, std::uint32_t index) {
    const cudaStream_t stream = device_.stream;
    CUDA_CHECK(cudaEventSynchronize(flush_uploaded_)); // the previous flush has read the staging words
    auto* words = static_cast<std::int32_t*>(flush_host_.data());
    words[0]    = lane.history[lane.state_tokens];
    words[1]    = static_cast<std::int32_t>(lane.state_tokens - 1U);
    words[2]    = static_cast<std::int32_t>(index);
    words[3]    = static_cast<std::int32_t>(index);
    CUDA_CHECK(cudaMemcpyAsync(flush_device_.p, words, 4 * sizeof(std::int32_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaEventRecord(flush_uploaded_, stream));
    auto* device = static_cast<std::int32_t*>(flush_device_.p);
    execution::MtpCall call{.residuals  = saved_column(index).view({width_, 1}),
                            .ids        = Tensor(device, DType::I32, {1}),
                            .positions  = Tensor(device + 1, DType::I32, {1}),
                            .slots      = Tensor(device + 2, DType::I32, {1}),
                            .table_rows = Tensor(device + 3, DType::I32, {1}),
                            .batch      = 1,
                            .width      = 1,
                            .kv_only    = true};
    forward_->run_mtp(call);
    lane.mtp_cells = lane.state_tokens;
}

// ---- admission ---------------------------------------------------------------------------------------

// Chooses the resume source for `prompt` in lane `lane`: the deepest affordable snapshot, else the
// root, or nothing when even the root does not fit now (design §19.3.1, plan §7.1).
std::optional<PrefixSelection> ProgramImpl::prefix_select(const qwen3_5::PreparedPromptData& prompt,
                                                                       const BasePlanImpl& base,
                                                                       std::uint32_t lane) {
    pc::PrefixCacheIndex& index = prefix_->index();
    const auto n                = static_cast<std::uint32_t>(prompt.token_ids.size());
    const std::uint32_t E       = base.pages;
    const auto fits = [&](PrefixSelection& s) {
        // Entries this admission pins stop being evictable: its unpinned Device path blocks and a
        // Device copy-on-write source.
        std::uint32_t evictable_on_path = 0, host_only = 0;
        const std::uint32_t k = s.frontier / kBlock;
        for (std::uint32_t b = 0; b < k; ++b) {
            const pc::NodeView view = index.node(s.path[b]);
            if (view.pins == 0 && view.device == pc::CopyState::Resident) { ++evictable_on_path; }
            if (b < s.shared && view.device != pc::CopyState::Resident) { ++host_only; }
        }
        if (s.cow == PrefixCow::Tail) {
            const pc::SnapshotView view = index.snapshot(*s.snapshot);
            if (view.tail_device_copy == pc::CopyState::Resident && view.pins == 0) { ++evictable_on_path; }
        }
        s.need                     = (E - s.shared) + host_only;
        const std::uint64_t room   = static_cast<std::uint64_t>(pool_->available_pages()) + index.device_evictable_blocks();
        return static_cast<std::uint64_t>(s.need) + evictable_on_path <= room;
    };
    std::uint32_t cached_tokens = 0; // the prompt prefix held as cached blocks
    const auto planned = [&](PrefixSelection s) {
        s.plan          = plan_prefill(prompt, s.frontier, s.existing, base.reuse);
        s.cached_tokens = cached_tokens;
        return s;
    };
    PrefixSelection root;
    root.need = E;
    if (base.reuse && n > 1) {
        prefix_->poll();
        pc::MatchResult match = index.match(prompt.token_ids, prompt.block_hashes, prompt.block_extras, n);
        if (prefix_->await_extended_endpoint(match.path, prompt.token_ids)) {
            // The endpoint this prompt continues has just been published.
            match = index.match(prompt.token_ids, prompt.block_hashes, prompt.block_extras, n);
        }
        cached_tokens = static_cast<std::uint32_t>(match.path.size()) * kBlock;
        // A frontier strictly inside a Vision item would resume half an image (design §19.3.1).
        const std::vector<pc::TapExclusion> spans = prefix_exclusions(prompt);
        std::erase_if(match.candidates, [&](const pc::MatchCandidate& c) {
            return c.filling_blocks != 0 || prefix::inside_exclusion(c.frontier, spans);
        });
        // A lane-resident snapshot needs no image restore: its state is still in the lane.
        const std::optional<pc::SnapshotRef> resident = prefix_resident(lanes_[lane]);
        for (pc::MatchCandidate& c : match.candidates) {
            if (resident && *resident == c.snapshot && c.restore_bytes >= image_layout_.image_bytes) {
                c.restore_bytes -= image_layout_.image_bytes;
            }
        }
        const pc::AdmissionChoice choice = index.choose(match, n);
        std::vector<std::uint32_t> frontiers;
        for (const pc::MatchCandidate& c : match.candidates) { frontiers.push_back(c.frontier); }
        std::vector<std::size_t> order;
        if (choice.candidate) { order.push_back(*choice.candidate); }
        for (std::size_t i = 0; i < match.candidates.size(); ++i) {
            if (!choice.candidate || i != *choice.candidate) { order.push_back(i); }
        }
        for (const std::size_t i : order) {
            const pc::MatchCandidate& c = match.candidates[i];
            PrefixSelection s;
            s.snapshot = c.snapshot;
            s.frontier = c.frontier;
            s.path.assign(match.path.begin(), match.path.begin() + c.path_blocks);
            const std::uint32_t k = c.frontier / kBlock;
            s.shared              = k;
            if (c.frontier % kBlock != 0) {
                s.cow = PrefixCow::Tail;
            } else if (mtp_ && k > 0 && prefix_->mtp_next(s.path[k - 1]) != prompt.token_ids[c.frontier]) {
                // MR6: the anchor's last MTP cell encodes another continuation; the lane rewrites it
                // in a private copy of the anchor page.
                s.cow    = PrefixCow::Anchor;
                s.shared = k - 1;
            }
            s.image_restore = !(resident && *resident == c.snapshot);
            s.existing      = frontiers;
            if (fits(s)) { return planned(std::move(s)); }
        }
    }
    if (fits(root)) { return planned(std::move(root)); }
    if (prefix_->transfers_pending()) {
        // Blocks pinned only by in-flight copies become evictable once those land.
        prefix_->drain();
        if (fits(root)) { return planned(std::move(root)); }
    }
    return std::nullopt;
}

// Pins what a selection reads (its matched path up to the anchor and the snapshot) before room is
// made for it, so making room cannot evict them.
void ProgramImpl::prefix_pin(const PrefixSelection& s) {
    if (!s.snapshot) { return; }
    const std::span<const pc::NodeRef> path(s.path.data(), s.frontier / kBlock);
    if (!path.empty()) { prefix_->index().acquire_path(path); }
    prefix_->index().pin_snapshot(*s.snapshot);
}

void ProgramImpl::prefix_unpin(const PrefixSelection& s) noexcept {
    if (!s.snapshot) { return; }
    try {
        const std::span<const pc::NodeRef> path(s.path.data(), s.frontier / kBlock);
        if (!path.empty()) { prefix_->index().release_path(path); }
        prefix_->index().unpin_snapshot(*s.snapshot);
    } catch (...) {}
}

// Activates a pinned selection in a lane whose pages are reserved: maps shared pages, stages
// restores and copy-on-write sources, and sets the lane's frontier and MTP state (plan §7.2).
void ProgramImpl::prefix_activate(Lane& lane, std::uint32_t index, const PrefixSelection& s,
                                  DeviceKVPageReservation& reservation, std::uint32_t pages) {
    pc::PrefixCacheIndex& cache = prefix_->index();
    const cudaStream_t stream   = device_.stream;
    // A copy-out of this lane's previous state must land before anything overwrites the slot.
    for (const cudaEvent_t event : prefix_->capture_events(lane.prefix.capture)) {
        CUDA_CHECK(cudaStreamWaitEvent(stream, event, 0));
    }
    const std::uint32_t k = s.frontier / kBlock;
    // The lane keeps the pins of the blocks it maps (prefix_pin took one on each up to the anchor).
    lane.prefix.path.assign(s.path.begin(), s.path.begin() + s.shared);
    lane.pages.clear();
    lane.pages.reserve(pages - s.shared);
    pool_->materialize(reservation, pages - s.shared, lane.pages);
    lane.prefix.page_base = s.shared;
    const bool restores   = s.snapshot && (s.image_restore || s.cow != PrefixCow::None ||
                                         std::any_of(lane.prefix.path.begin(), lane.prefix.path.end(), [&](pc::NodeRef n) {
                                             return cache.node(n).device != pc::CopyState::Resident;
                                         }));
    if (s.snapshot) {
        if (restores) { prefix_->open_restore(stream); }
        for (const pc::NodeRef node : lane.prefix.path) {
            if (cache.node(node).device != pc::CopyState::Resident) {
                prefix_->restore_block(node, pool_->materialize_one(reservation));
            }
        }
        const DeviceKVPageHandle first_private = lane.pages.empty() ? DeviceKVPageHandle{} : lane.pages.front().handle();
        if (s.cow == PrefixCow::Tail) {
            const pc::SnapshotView view = cache.snapshot(*s.snapshot);
            if (view.tail_device_copy == pc::CopyState::Resident) {
                pool_->copy_page(prefix_->page(view.tail_device), first_private, stream);
            } else {
                prefix_->restore_tail_page(*s.snapshot, first_private);
            }
        } else if (s.cow == PrefixCow::Anchor) {
            const pc::NodeRef anchor = s.path[k - 1];
            const pc::NodeView view  = cache.node(anchor);
            if (view.device == pc::CopyState::Resident) {
                pool_->copy_page(prefix_->page(view.device_id), first_private, stream);
            } else {
                prefix_->restore_node_page(anchor, first_private);
            }
        }
        if (s.image_restore) { prefix_->restore_image(*s.snapshot, index); }
        if (restores) { lane.prefix.restore = prefix_->submit_restore(); }
        if (s.cow == PrefixCow::Anchor) {
            // The anchor's copy is stream-ordered (or pinned by the restore batch): drop its pin.
            const std::span<const pc::NodeRef> to_anchor(s.path.data(), k);
            cache.release_path(to_anchor);
            if (k > 1) { cache.acquire_path(to_anchor.first(k - 1)); }
        }
        cache.note_hit(*s.snapshot);
        cache.unpin_snapshot(*s.snapshot); // the restore batch holds its own pin while it lands
        lane.prefix.resume  = s.snapshot;
        lane.prefix.deepest = s.frontier;
        ++prefix_->counters().admissions;
        prefix_->counters().reused_tokens += s.frontier;
    }
    // The slot no longer holds the endpoint image once anything is admitted into it.
    prefix_->forget_capture(lane.prefix.resident_capture);
    lane.prefix.resident_capture = 0;
    if (!s.snapshot) { reset_slot(index); }
    // The block table: shared cache pages, then the private leases.
    std::vector<DeviceKVPageHandle> handles;
    handles.reserve(pages);
    for (const pc::NodeRef node : lane.prefix.path) { handles.push_back(prefix_->page(cache.node(node).device_id)); }
    for (const DeviceKVPageLease& lease : lane.pages) { handles.push_back(lease.handle()); }
    tables_->publish(lane.row.handle(), 0, std::span<const DeviceKVPageHandle>(handles), stream);
    lane.state_tokens = s.frontier;
    if (s.snapshot) {
        // MR6: whether the drafter's cell F - 1 holds this prompt's continuation.
        const prefix::StateImageHeader& meta = prefix_->meta(*s.snapshot);
        // The opener of an echo lineage serves only when the endpoint after it did not: the client
        // did not echo the generated turn exactly (design §19.3.1, "Generation opener").
        if (meta.opener && meta.lineage_echo) { ++prefix_->counters().endpoint_mismatch_fallbacks; }
        const bool continues = static_cast<std::size_t>(s.frontier) < lane.history.size() &&
                               meta.mtp_next == lane.history[s.frontier];
        if (s.cow == PrefixCow::Tail) {
            lane.mtp_cells = meta.mtp_written && continues ? s.frontier : s.frontier - 1U;
        } else if (s.cow == PrefixCow::Anchor) {
            lane.mtp_cells = s.frontier - 1U;
        } else {
            lane.mtp_cells = s.frontier;
        }
        if (mtp_ && lane.mtp_cells != s.frontier) { ++prefix_->counters().mtp_continuation_mismatches; }
        lane.mtp_live = mtp_;
        // Penalty counts restart with the request; the lane's state came from the snapshot.
        CUDA_CHECK(cudaMemsetAsync(static_cast<std::int32_t*>(token_counts_.p) + index * static_cast<std::size_t>(token_domain_),
                                   0, 4ULL * token_domain_, stream));
    }
    // Taps for this prompt beyond the resume frontier, realized at the plan's call boundaries.
    lane.prefix.taps = s.plan.taps;
}

// The prompt's prefill calls from `frontier` (design §19.3.1, "Prefill integration"): with `taps`, the
// prefix taps beyond the frontier (no tap inside a Vision item), exact ones admitted by their cost.
prefix::CallPlan ProgramImpl::plan_prefill(const qwen3_5::PreparedPromptData& prompt, std::uint32_t frontier,
                                           std::span<const std::uint32_t> existing, bool taps) const {
    const auto n = static_cast<std::uint32_t>(prompt.token_ids.size());
    std::vector<pc::PlannedTap> planned;
    if (taps && prefix_) {
        const std::vector<pc::TapExclusion> spans = prefix_exclusions(prompt);
        planned = pc::plan_taps(n, frontier, prompt.tap_hints.hints, existing, spans, prefix_->taps());
    }
    return prefix::plan_calls(frontier, n, static_cast<std::uint32_t>(chunk_), planned,
                              prefix::CallCost{.model = options_.prefix_cost});
}

HybridPrefixCacheStats ProgramImpl::prefix_stats() const noexcept {
    HybridPrefixCacheStats out;
    if (!prefix_) { return out; }
    const pc::PrefixIndexStats index         = prefix_->index().stats();
    const prefix::PrefixCacheCounters& count = prefix_->counters();
    out.nodes                   = index.nodes;
    out.snapshots               = index.snapshots;
    out.device_resident_blocks  = index.device_resident_blocks;
    out.device_evictable_blocks = index.device_evictable_blocks;
    out.host_slabs              = prefix_->host_layout().slabs;
    out.host_free_slabs         = index.host_free_slabs;
    out.host_slab_bytes         = prefix_->host_layout().slab_bytes;
    out.snapshot_hits           = index.snapshot_hits;
    out.reused_tokens           = count.reused_tokens;
    out.blocks_inserted         = count.blocks_inserted;
    out.blocks_reattached       = count.blocks_reattached;
    out.blocks_duplicate        = count.blocks_duplicate;
    out.taps_created            = count.taps_created;
    out.taps_skipped            = count.taps_skipped;
    out.endpoints_created       = count.endpoints_created;
    out.host_image_writes       = count.host_image_writes;
    out.host_block_writes       = count.host_block_writes;
    out.host_image_restores     = count.host_image_restores;
    out.host_block_restores     = count.host_block_restores;
    out.host_write_bytes        = count.host_write_bytes;
    out.host_restore_bytes      = count.host_restore_bytes;
    out.evicted_blocks          = index.device_block_evictions;
    out.host_snapshot_evictions = index.host_snapshot_evictions;
    out.host_dead_reclaims      = index.host_dead_reclaims;
    out.unbacked_node_losses    = index.unbacked_node_losses;
    return out;
}

std::vector<pc::TapExclusion> ProgramImpl::prefix_exclusions(const qwen3_5::PreparedPromptData& prompt) {
    std::vector<pc::TapExclusion> out;
    if (prompt.vision_items.empty()) { return out; }
    for (const auto& range : qwen3_5::detail::vision_ranges(prompt)) { out.push_back({range.begin, range.end}); }
    return out;
}

// ---- publication -------------------------------------------------------------------------------------

// The frontier below which no write of the lane remains (plan §7.4).
std::uint32_t ProgramImpl::prefix_frontier(const Lane& lane, bool finishing) const noexcept {
    if (!lane.mtp_live) { return lane.state_tokens; }
    if (finishing || lane.phase == Phase::Prefill) { return lane.mtp_cells; }
    return std::min(lane.mtp_cells, lane.state_tokens - 1U);
}

void ProgramImpl::prefix_publish(Lane& lane, std::uint32_t frontier) {
    if (!prefix_ || !lane.prefix.reuse) { return; }
    pc::PrefixCacheIndex& cache = prefix_->index();
    // Blocks past the prompt (generated output) chain their lookup hash from the previous block's
    // and carry the prompt's trailing Vision key, as Qwen3.5's publication does.
    const std::uint32_t blocks =
        std::min<std::uint32_t>(frontier, static_cast<std::uint32_t>(lane.history.size())) / kBlock;
    while (lane.prefix.path.size() < blocks) {
        const auto b = static_cast<std::uint32_t>(lane.prefix.path.size());
        const pc::NodeRef parent = b == 0 ? pc::NodeRef{} : lane.prefix.path.back();
        const std::span<const TokenId> tokens(lane.history.data() + static_cast<std::size_t>(b) * kBlock, kBlock);
        const std::uint64_t extra =
            b < lane.prefix.extras.size() ? lane.prefix.extras[b] : lane.prefix.trailing_extra;
        if (b >= lane.prefix.hashes.size()) {
            const std::uint64_t previous = b == 0 ? pc::kRootLookupHash : lane.prefix.hashes[b - 1];
            lane.prefix.hashes.push_back(pc::block_lookup_hash(previous, tokens, extra));
        }
        const std::int32_t next      = lane.history.size() > static_cast<std::size_t>(kBlock) * (b + 1U)
                                           ? lane.history[static_cast<std::size_t>(kBlock) * (b + 1U)]
                                           : -1;
        // MR5: adopt a Host-only twin only when its last MTP cell encodes this lane's next token.
        bool attach = true;
        if (const auto existing = cache.find_child(parent, lane.prefix.hashes[b], tokens, extra); existing && mtp_) {
            attach = prefix_->mtp_next(*existing) == next;
            if (!attach) { ++prefix_->counters().mtp_branch_mismatches; }
        }
        DeviceKVPageLease& lease = lane.pages.at(b - lane.prefix.page_base);
        const pc::InsertResult result =
            prefix_->insert_block(parent, lane.prefix.hashes[b], tokens, extra, lease, attach, next);
        lane.prefix.path.push_back(result.node);
    }
}

// A tap or endpoint snapshot of the lane's state at its frontier F = state_tokens (Host-born).
void ProgramImpl::prefix_capture(Lane& lane, std::uint32_t index, pc::SnapshotKind kind, bool hand_over_tail) {
    const std::uint32_t F = lane.state_tokens;
    const std::uint32_t k = F / kBlock;
    if (F == 0 || lane.prefix.path.size() < k) { return; }
    prefix::PrefixCache::Capture c;
    c.id       = ++next_capture_;
    c.anchor   = k == 0 ? pc::NodeRef{} : lane.prefix.path[k - 1];
    c.frontier = F;
    c.tail.assign(lane.history.begin() + static_cast<std::ptrdiff_t>(k) * kBlock, lane.history.begin() + F);
    if (!c.tail.empty()) {
        DeviceKVPageLease& page = lane.pages.at(k - lane.prefix.page_base);
        c.tail_page             = page.handle();
        if (hand_over_tail) { c.tail_lease.emplace(std::move(page)); }
    }
    c.kind             = kind;
    c.claim            = prefix_->index().estimate_priority(lane.prefix.deepest, F, !c.tail.empty());
    c.meta.frontier    = F;
    c.meta.mtp_written = lane.mtp_live && lane.mtp_cells == F;
    c.meta.mtp_next    = lane.history.size() > F ? lane.history[F] : -1;
    c.meta.lineage_echo = lane.prefix.resume && prefix_->index().valid(*lane.prefix.resume) &&
                          prefix_->index().snapshot(*lane.prefix.resume).kind == pc::SnapshotKind::Endpoint;
    c.meta.opener = kind == pc::SnapshotKind::Tap &&
                    std::any_of(lane.prefix.hints.begin(), lane.prefix.hints.end(), [&](const pc::TapHint& hint) {
                        return hint.kind == pc::TapHintKind::GenerationOpener && hint.position == F;
                    });
    c.meta.mtp_accept = lane.mtp_accept;
    // The lineage's previous snapshot serves only requests diverging before this one.
    if (const auto previous = prefix_->capture_result(lane.prefix.capture); previous && previous->snapshot.valid()) {
        c.supersedes = previous->snapshot;
    } else if (lane.prefix.resume) {
        c.supersedes = *lane.prefix.resume;
    }
    const std::uint64_t id = c.id;
    if (prefix_->capture(index, std::move(c), device_.stream)) {
        prefix_->forget_capture(lane.prefix.capture);
        lane.prefix.capture = id;
        lane.prefix.deepest = F;
    }
}

// After a prefill call ending at the lane's frontier: publish its blocks and realize the taps due
// at this call boundary.
void ProgramImpl::prefix_after_prefill_call(Lane& lane, std::uint32_t index, bool last) {
    if (!prefix_ || !lane.prefix.reuse) { return; }
    prefix_publish(lane, prefix_frontier(lane, false));
    if (last || lane.prefix.taps.empty()) { return; }
    const std::uint32_t B = lane.state_tokens;
    // A boundary inside a Vision item cannot be resumed from: its flexible taps wait for the next.
    if (prefix::inside_exclusion(B, lane.prefix.exclusions)) { return; }
    // An exact tap ends a call; a flexible one is realized at the first boundary at or past it, or
    // at the start of the prompt's final call.
    const bool final_next = lane.next_call + 1U == lane.calls.size();
    bool due              = false;
    std::erase_if(lane.prefix.taps, [&](const pc::PlannedTap& tap) {
        const bool now = tap.position <= B || (final_next && tap.placement == pc::TapPlacement::Flexible);
        due            = due || now;
        return now;
    });
    if (due && B >= lane.prefix.deepest + kBlock) {
        prefix_capture(lane, index, pc::SnapshotKind::Tap, false);
    }
}

// Finish or a consistent abort: the M3 flush, the remaining blocks, the endpoint and write-through.
void ProgramImpl::prefix_finish(Lane& lane, std::uint32_t index) {
    if (!prefix_ || !lane.prefix.reuse) { return; }
    // MR3: write the pending MTP cell when its token is known, so the endpoint is final.
    if (lane.mtp_live && lane.mtp_cells + 1U == lane.state_tokens && lane.history.size() > lane.state_tokens) {
        mtp_flush_cell(lane, index);
    }
    prefix_publish(lane, prefix_frontier(lane, true));
    const std::uint32_t F = lane.state_tokens;
    const bool full_blocks = !lane.mtp_live || lane.mtp_cells >= F / kBlock * kBlock;
    if (F >= lane.prefix.deepest + kBlock && full_blocks && lane.prefix.path.size() >= F / kBlock) {
        prefix_capture(lane, index, pc::SnapshotKind::Endpoint, true);
    } else if (!full_blocks) {
        ++prefix_->counters().endpoint_skipped_mtp;
    }
    // Blocks reach the Host after the endpoint (one transfer stream, FIFO).
    prefix_->write_blocks(lane.prefix.path, device_.stream);
    // The lane's slot holds the image of its latest capture until the lane is reused, but only when
    // that capture is at the lane's frontier: when the endpoint was not captured (too close to the
    // deepest snapshot, unfinished MTP blocks), the latest capture is an earlier tap whose state the
    // slot no longer holds, and a resume from it must restore its image.
    lane.prefix.resident_capture = lane.prefix.deepest == F ? lane.prefix.capture : 0;
}

// The endpoint this lane published at its last finish, while its slot still holds that image: only
// a snapshot the capture created (a duplicate's image came from another lineage).
std::uint64_t ProgramImpl::prefix_trailing_extra(const qwen3_5::PreparedPromptData& prompt) {
    std::uint64_t cumulative = 0;
    for (const auto& range : qwen3_5::detail::vision_ranges(prompt)) {
        cumulative = qwen3_5::detail::accumulate_vision(cumulative, range.key);
    }
    return cumulative;
}

std::optional<pc::SnapshotRef> ProgramImpl::prefix_resident(const Lane& lane) const {
    if (lane.prefix.resident_capture == 0) { return std::nullopt; }
    const auto result = prefix_->capture_result(lane.prefix.resident_capture);
    if (!result || !result->created || !prefix_->index().valid(result->snapshot)) { return std::nullopt; }
    return result->snapshot;
}

void ProgramImpl::prefix_release(Lane& lane) noexcept {
    if (!prefix_) { return; }
    try {
        if (!lane.prefix.path.empty()) { prefix_->index().release_path(lane.prefix.path); }
        prefix_->poll();
    } catch (...) {}
    lane.prefix.path.clear();
    lane.prefix.taps.clear();
    lane.prefix.exclusions.clear();
    lane.prefix.restore.reset();
    lane.prefix.resume.reset();
    lane.prefix.page_base = 0;
    lane.prefix.deepest   = 0;
    lane.prefix.reuse     = false;
}

std::array<std::span<const cudaEvent_t>, 2> ProgramImpl::prefix_waits(const Lane& lane) const noexcept {
    if (!prefix_) { return {}; }
    return {lane.prefix.restore ? prefix_->restore_events(*lane.prefix.restore) : std::span<const cudaEvent_t>{},
            prefix_->capture_events(lane.prefix.capture)};
}

} // namespace ninfer::models::qwen4_exp::detail
