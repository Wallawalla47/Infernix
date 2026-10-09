// Hybrid prefix cache orchestration inside the Qwen3.5 Program
// (docs/maintainer/hybrid-prefix-cache-spec.md). Admission (staging, Host restores, activation),
// block publication, prefill taps and terminal snapshots; the index and physical bindings live in
// hybrid_cache.{h,cpp}.

#include "models/qwen3_5/program/program_impl.h"
#include "models/qwen3_5/program/execution_context.h"
#include "models/qwen3_5/program/context_work.h"
#include "models/qwen3_5/program/internal.h"
#include "models/qwen3_5/execution/vision_overlay.h"
#include "models/qwen3_5/execution/vision.h"
#include "models/qwen3_5/program/vision_control.h"
#include "models/qwen3_5/program/vision_prefill.h"
#include "models/qwen3_5/program/prefix/block_keys.h"
#include "core/device.h"
#include "core/startup.h"
#include "runtime/prefix_cache/block_hash.h"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstdint>
#include <functional>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace infernix::models::qwen3_5::detail {

namespace pc = runtime::prefix_cache;

namespace {

constexpr std::uint32_t kBlock = pc::kBlockTokens;

// Blocks one prefetch batch copies (§6.6): about 20 ms of PCIe for 27B INT8, the longest a quote
// may wait for blocks it is about to resume over.
constexpr std::uint32_t kPrefetchBatchBlocks = 256;

std::uint64_t all_vision_key(std::span<const VisionTokenRange> ranges) noexcept {
    std::uint64_t cumulative = 0;
    for (const VisionTokenRange& range : ranges) {
        cumulative = accumulate_vision(cumulative, range.key);
    }
    return cumulative;
}

bool inside_vision(std::uint32_t frontier, std::span<const VisionTokenRange> ranges) noexcept {
    return std::any_of(ranges.begin(), ranges.end(), [&](const VisionTokenRange& range) {
        return range.begin < frontier && frontier < range.end;
    });
}

bool inside_exclusion(std::uint32_t frontier,
                      std::span<const pc::TapExclusion> exclusions) noexcept {
    return std::any_of(exclusions.begin(), exclusions.end(), [&](const pc::TapExclusion& span) {
        return span.begin < frontier && frontier < span.end;
    });
}

PrefixReusePath reuse_path_for(pc::SnapshotKind kind) noexcept {
    return kind == pc::SnapshotKind::Endpoint ? PrefixReusePath::HybridEndpoint
                                              : PrefixReusePath::HybridSnapshot;
}

} // namespace

// ---- construction ------------------------------------------------------------------------------

void ProgramImpl::create_hybrid_prefix_cache(const StartupObserver& observer) {
    if (!context_cache.enabled || causal_scoring) { return; }
    const DeviceKVPagePool& text_pool = text_kv_pages->physical_pool();
    const KVPageGeometry* backend_geometry =
        backend_kv_pages ? &backend_kv_pages->physical_pool().geometry() : nullptr;
    const HybridHostLayout host = plan_hybrid_host_layout(text_pool.geometry(), backend_geometry,
                                                          state_images->host_layout().image_bytes);
    const std::uint32_t slabs =
        hybrid_host_slabs(host, context_cache.host_capacity_bytes.value_or(0U));
    const HybridPrefixCacheOptions& options = context_cache.hybrid;
    const std::uint32_t device_slots        = options.device_snapshot_slots.value_or(0U);
    if (device_slots == 0 || !options.max_new_taps || !options.tap_ladder_tokens ||
        !options.tap_min_gap_tokens) {
        throw std::logic_error("hybrid prefix cache options are not normalized");
    }
    pc::PrefixIndexConfig config;
    // Every node holds a Device page bundle or a Host slab.
    config.max_nodes             = text_pool.capacity_pages() + slabs + 1U;
    config.max_snapshots         = device_slots + slabs / host.image_slabs + 1U;
    config.host_slabs            = slabs;
    config.image_slabs           = host.image_slabs;
    config.device_snapshot_slots = device_slots;
    config.block_bytes           = host.block_payload_bytes;
    config.image_bytes           = host.image_bytes;
    config.host_blocks           = slabs != 0;
    config.cost                  = hybrid_cost_;
    config.cost.chunk_tokens     = prefill_chunk;
    const pc::TapPlannerConfig taps{
        .max_new_taps   = *options.max_new_taps,
        .ladder_tokens  = *options.tap_ladder_tokens,
        .min_gap_tokens = *options.tap_min_gap_tokens,
    };
    const std::uint64_t host_bytes = static_cast<std::uint64_t>(slabs) * host.slab_bytes;
    std::optional<StartupPhaseScope> phase;
    if (host_bytes != 0) {
        phase.emplace(observer, StartupPhase::HostContextPin, StartupProgressUnit::Bytes,
                      host_bytes);
    }
    // Restores land in forward order: each layer's KV planes or recurrent state.
    const TextConfig& text = parameters.model.config().text;
    std::vector<HybridRestoreLayer> layers;
    layers.reserve(text.layer_types.size());
    for (std::size_t layer = 0; layer < text.layer_types.size(); ++layer) {
        layers.push_back(
            HybridRestoreLayer{.attention = text.layer_types[layer] == MixerKind::FullAttention,
                               .index     = text.compact_layer_indices[layer]});
    }
    hybrid_ = std::make_unique<HybridPrefixCache>(
        config, taps, *text_kv_pages, backend_kv_pages.get(), *state_store, *state_images, host,
        std::move(layers), phase ? &*phase : nullptr);
    if (phase) { phase->complete(host_bytes, host_bytes); }
}

void ProgramImpl::set_hybrid_cost(const pc::CacheCostModel& cost) {
    hybrid_cost_              = cost;
    hybrid_cost_.chunk_tokens = prefill_chunk;
    if (hybrid_) { hybrid_->set_cost(hybrid_cost_); }
}

namespace {

HybridCachePersistence public_result(const HybridPersistResult& result) {
    return HybridCachePersistence{
        .ok                  = result.ok,
        .message             = result.message,
        .blocks              = result.blocks,
        .snapshots           = result.snapshots,
        .bytes               = result.bytes,
        .seconds             = result.seconds,
        .saved_blocks        = result.saved_blocks,
        .saved_snapshots     = result.saved_snapshots,
        .required_host_bytes = result.required_host_bytes,
        .host_bytes          = result.host_bytes,
    };
}

} // namespace

HybridCachePersistence ProgramImpl::attach_hybrid_cache_file(const std::filesystem::path& path,
                                                             std::string fingerprint,
                                                             const StartupObserver& observer) {
    if (!hybrid_) { return {.message = "the hybrid prefix cache is not enabled"}; }
    if (path.empty()) { throw std::invalid_argument("hybrid prefix cache file path is empty"); }
    HybridCachePersistence loaded = public_result(hybrid_->load(path, fingerprint, observer));
    hybrid_file_                  = path;
    hybrid_fingerprint_           = std::move(fingerprint);
    hybrid_saved_writes_          = hybrid_->counters().host_write_bytes;
    return loaded;
}

std::optional<HybridCachePersistence> ProgramImpl::save_hybrid_cache_now(const CancellationView& abandoned) {
    if (!hybrid_ || hybrid_file_.empty() || hybrid_->counters().host_write_bytes == hybrid_saved_writes_) {
        return std::nullopt;
    }
    // As at shutdown: every Host write lands before the slabs are read; lanes pinning a path do not
    // matter, the save only reads.
    device.synchronize();
    if (device.transfer_stream != nullptr) { CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream)); }
    hybrid_->drain();
    const std::uint64_t written = hybrid_->counters().host_write_bytes;
    HybridCachePersistence saved = public_result(hybrid_->save(hybrid_file_, hybrid_fingerprint_, abandoned));
    if (saved.ok) { hybrid_saved_writes_ = written; }
    return saved;
}

void ProgramImpl::save_hybrid_cache_for_shutdown() noexcept {
    if (!hybrid_ || hybrid_file_.empty()) { return; }
    try {
        // Only Host-resident entries are saved, so every Host write must land before the slabs
        // are read. Lanes still pinning a path do not matter: the save only reads.
        device.synchronize();
        if (device.transfer_stream != nullptr) {
            CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream));
        }
        hybrid_->drain();
        // The product may abandon the save (a Ctrl+C during the stop). It counts as running only
        // from here, so an exit never waits for the Device work above.
        const PrefixCacheSaveControl& control = context_cache.hybrid.persistent_save;
        if (!control.begin()) {
            hybrid_shutdown_save_ = HybridCachePersistence{.message = "abandoned before it began"};
            return;
        }

        struct Ended {
            const PrefixCacheSaveControl& control;
            bool saved = false;

            ~Ended() { control.end(saved); }
        } ended{control};

        hybrid_shutdown_save_ = public_result(
            hybrid_->save(hybrid_file_, hybrid_fingerprint_,
                          CancellationView([&control] { return control.abandoned(); })));
        ended.saved = hybrid_shutdown_save_->ok;
    } catch (const std::exception& error) {
        hybrid_shutdown_save_ = HybridCachePersistence{.message = error.what()};
    } catch (...) { hybrid_shutdown_save_ = HybridCachePersistence{.message = "unknown error"}; }
}

std::uint32_t ProgramImpl::hybrid_backend_frontier(std::uint32_t frontier) const noexcept {
    return speculative_backend == SpeculativeBackend::Mtp && frontier != 0 ? frontier - 1U
                                                                           : frontier;
}

// ---- admission ---------------------------------------------------------------------------------

std::vector<SourceCandidate> ProgramImpl::hybrid_sources(const RequestBasePlan& base,
                                                         std::uint32_t maximum_frontier) {
    if (base.impl_ == nullptr || !base.impl_->prompt) {
        throw std::logic_error("admission requires a prepared prompt");
    }
    if (hybrid_) { hybrid_->poll(); }
    const RequestBasePlanImpl& plan  = *base.impl_;
    const PreparedPromptData& prompt = *plan.prompt;
    const auto n                     = static_cast<std::uint32_t>(prompt.token_ids.size());
    if (n == 0 || n != plan.summary.prompt_tokens) {
        throw std::logic_error("hybrid admission prompt does not match its base plan");
    }
    const auto source = [](std::shared_ptr<HybridQuoteImpl> quote, PrefixReusePath path) {
        SourceCandidate candidate;
        candidate.reused_tokens = quote->reuse_frontier;
        candidate.reuse_path    = path;
        candidate.hybrid        = std::move(quote);
        return candidate;
    };
    auto root = std::make_shared<HybridQuoteImpl>();
    // Without a prefix cache every request starts from the root.
    if (!hybrid_ || !plan.allow_prefix_reuse || !prompt.identity.reusable) {
        return {source(std::move(root), PrefixReusePath::Root)};
    }

    pc::PrefixCacheIndex& index = hybrid_->index();
    const bool backend_pool     = backend_kv_pages != nullptr;
    // Device pages a source needs beyond what the pools have free plus the cache can give up:
    // the restored blocks and tail, the private tail copy and the first unit's growth. Entries
    // this source pins stop being evictable: its unpinned Device-resident path blocks and the
    // snapshot's Device-resident tail.
    const auto fits = [&](const pc::MatchCandidate& candidate, std::span<const pc::NodeRef> path) {
        const std::uint32_t frontier = candidate.frontier;
        const std::uint32_t restored =
            candidate.host_only_blocks + (candidate.tail && !candidate.tail_on_device ? 1U : 0U);
        std::uint32_t evictable_on_path = 0;
        for (const pc::NodeRef node : path) {
            const pc::NodeView view = index.node(node);
            if (view.pins == 0 && view.device == pc::CopyState::Resident) { ++evictable_on_path; }
        }
        if (candidate.tail && candidate.tail_on_device &&
            index.snapshot(candidate.snapshot).pins == 0) {
            ++evictable_on_path;
        }
        const std::uint64_t evictable = index.device_evictable_blocks() - evictable_on_path;
        const std::uint32_t first     = std::min(n, frontier + prefill_chunk);
        const auto need               = [&](std::uint32_t reuse) {
            return static_cast<std::uint64_t>(kv_pages_for_frontier(first)) - reuse / kBlock +
                   restored;
        };
        if (need(frontier) > text_kv_pages->physical_pool().available_pages() + evictable) {
            return false;
        }
        return !backend_pool || need(hybrid_backend_frontier(frontier)) <=
                                    backend_kv_pages->physical_pool().available_pages() + evictable;
    };

    // Preparation computed the lookup keys once; every quote of this prompt reads them.
    if (prompt.block_hashes.size() != prompt.token_ids.size() / kBlock) {
        throw std::logic_error("prepared prompt carries no hybrid block keys");
    }
    pc::MatchResult match =
        index.match(prompt.token_ids, prompt.block_hashes, prompt.block_extras, n);
    const auto filling = [](const pc::MatchResult& result) {
        return std::any_of(
            result.candidates.begin(), result.candidates.end(),
            [](const pc::MatchCandidate& candidate) { return candidate.filling_blocks != 0; });
    };
    if (hybrid_->prefetch_landing() && filling(match)) {
        // A prefetch (§6.6) is copying blocks this prompt resumes over: they land within a
        // batch's copy time, and skipping them would pick a shallower source.
        hybrid_->settle_prefetch();
        match = index.match(prompt.token_ids, prompt.block_hashes, prompt.block_extras, n);
    }
    const auto cached          = static_cast<std::uint32_t>(match.path.size()) * kBlock;
    root->cached_prefix_tokens = cached;
    const std::vector<VisionTokenRange> ranges =
        prompt.vision_items.empty() ? std::vector<VisionTokenRange>{} : vision_ranges(prompt);
    // A source keeps at least one prompt token to prefill, and a resumed request's source never
    // passes the frontier it replays to.
    std::erase_if(match.candidates, [&](const pc::MatchCandidate& candidate) {
        return candidate.filling_blocks != 0 || candidate.frontier >= n ||
               candidate.frontier > maximum_frontier || inside_vision(candidate.frontier, ranges);
    });
    const pc::AdmissionChoice choice = index.choose(match, n);
    std::vector<std::size_t> order;
    if (choice.candidate) { order.push_back(*choice.candidate); }
    for (std::size_t i = 0; i < match.candidates.size(); ++i) {
        if (!choice.candidate || i != *choice.candidate) { order.push_back(i); }
    }
    const auto select = [&]() -> std::optional<pc::MatchCandidate> {
        for (const std::size_t i : order) {
            const pc::MatchCandidate& candidate = match.candidates[i];
            if (fits(candidate,
                     std::span<const pc::NodeRef>(match.path.data(), candidate.path_blocks))) {
                return candidate;
            }
        }
        return std::nullopt;
    };
    std::optional<pc::MatchCandidate> selected = select();
    if (!selected && !match.candidates.empty() && hybrid_->transfers_pending()) {
        // Blocks pinned only by their in-flight Host writes become evictable once those land:
        // a cached source is not given up for a root bind while a finished lane's writes land.
        hybrid_->drain();
        selected = select();
    }
    // Coalescing only defers a fresh request; a resumed one replays from what exists.
    if (maximum_frontier == UINT32_MAX &&
        hybrid_await_sibling(prompt, selected ? selected->frontier : 0U)) {
        return {};
    }
    std::vector<SourceCandidate> out;
    if (selected) {
        auto quote                  = std::make_shared<HybridQuoteImpl>();
        quote->snapshot             = selected->snapshot;
        quote->reuse_frontier       = selected->frontier;
        quote->cached_prefix_tokens = cached;
        out.push_back(
            source(std::move(quote), reuse_path_for(index.snapshot(selected->snapshot).kind)));
    }
    out.push_back(source(std::move(root), PrefixReusePath::Root));
    return out;
}

std::optional<std::uint32_t> ProgramImpl::hybrid_prefetch(const RequestBasePlan& base) {
    if (!hybrid_ || !hybrid_->host_tier() || base.impl_ == nullptr || !base.impl_->prompt ||
        !base.impl_->allow_prefix_reuse || !base.impl_->prompt->identity.reusable) {
        return 0U;
    }
    const PreparedPromptData& prompt = *base.impl_->prompt;
    if (has_context_transaction() || hybrid_->restore_open()) { return std::nullopt; }
    hybrid_->poll();
    if (hybrid_->prefetch_landing()) { return std::nullopt; }
    const auto n = static_cast<std::uint32_t>(prompt.token_ids.size());
    if (n == 0 || prompt.block_hashes.size() != prompt.token_ids.size() / kBlock) { return 0U; }

    // The source the admission would choose now, as in hybrid_sources.
    pc::PrefixCacheIndex& index = hybrid_->index();
    pc::MatchResult match =
        index.match(prompt.token_ids, prompt.block_hashes, prompt.block_extras, n);
    const std::vector<VisionTokenRange> ranges =
        prompt.vision_items.empty() ? std::vector<VisionTokenRange>{} : vision_ranges(prompt);
    std::erase_if(match.candidates, [&](const pc::MatchCandidate& candidate) {
        return candidate.filling_blocks != 0 || candidate.frontier >= n ||
               inside_vision(candidate.frontier, ranges);
    });
    const pc::AdmissionChoice choice = index.choose(match, n);
    if (!choice.candidate || match.candidates[*choice.candidate].host_only_blocks == 0) {
        return 0U;
    }
    const std::span<const pc::NodeRef> path(match.path.data(),
                                            match.candidates[*choice.candidate].path_blocks);

    // Pinned while room is made, so no block of this path is evicted to hold another of them.
    // Room comes from free pages and host-backed cache, least recently used first: a prefetch
    // never waits for a transfer and never drops a block's last copy.
    index.acquire_path(path);
    std::uint32_t started = 0;
    try {
        DeviceKVPagePool& text_pool = text_kv_pages->physical_pool();
        DeviceKVPagePool* backend_pool =
            backend_kv_pages ? &backend_kv_pages->physical_pool() : nullptr;
        const auto available = [&] {
            const std::uint32_t text = text_pool.available_pages();
            return backend_pool != nullptr ? std::min(text, backend_pool->available_pages()) : text;
        };
        std::uint32_t wanted =
            std::min(match.candidates[*choice.candidate].host_only_blocks, kPrefetchBatchBlocks);
        while (available() < wanted && index.evict_backed_device_blocks(1) != 0) {}
        wanted = std::min(wanted, available());
        std::optional<DeviceKVPageReservation> text_pages;
        std::optional<DeviceKVPageReservation> backend_pages;
        if (wanted != 0) {
            text_pages = text_pool.reserve(wanted);
            if (backend_pool != nullptr) { backend_pages = backend_pool->reserve(wanted); }
        }
        if (text_pages && (backend_pool == nullptr || backend_pages)) {
            hybrid_->open_restore(device.stream);
            for (const pc::NodeRef node : path) {
                if (started == wanted) { break; }
                if (index.node(node).device != pc::CopyState::Absent) { continue; }
                HybridBlockPages pages{
                    .text = text_kv_pages->materialize_transfer_destination(*text_pages, kBlock)};
                if (backend_pool != nullptr) {
                    pages.backend =
                        backend_kv_pages->materialize_transfer_destination(*backend_pages, kBlock);
                }
                hybrid_->restore_block(node, pages);
                ++started;
            }
            hybrid_->submit_restore();
            hybrid_->detach_prefetch();
        }
    } catch (...) {
        hybrid_->abort_restore();
        index.release_path(path);
        throw;
    }
    index.release_path(path);
    return started;
}

std::uint32_t ProgramImpl::hybrid_prefetch_room() const noexcept {
    if (!hybrid_ || !hybrid_->host_tier()) { return 0U; }
    const std::uint32_t cached = hybrid_->index().device_backed_evictable_blocks();
    std::uint32_t room         = text_kv_pages->physical_pool().available_pages() + cached;
    if (backend_kv_pages) {
        room = std::min(room, backend_kv_pages->physical_pool().available_pages() + cached);
    }
    return room;
}

// The snapshot each queued request would resume from now, as hybrid_sources would choose it, is
// held in admission order (§9.6). A block landing for another admission does not change what the
// request needs, so filling candidates stay eligible.
void ProgramImpl::hybrid_hold_queue(std::span<const RequestBasePlan* const> queue) {
    if (!hybrid_) { return; }
    pc::PrefixCacheIndex& index = hybrid_->index();
    std::vector<pc::SnapshotRef> holds;
    for (const RequestBasePlan* base : queue) {
        if (holds.size() == pc::kMaxQueueHolds) { break; }
        if (base == nullptr || base->impl_ == nullptr || !base->impl_->prompt ||
            !base->impl_->allow_prefix_reuse || !base->impl_->prompt->identity.reusable) {
            continue;
        }
        const PreparedPromptData& prompt = *base->impl_->prompt;
        const auto n                     = static_cast<std::uint32_t>(prompt.token_ids.size());
        if (n == 0 || prompt.block_hashes.size() != prompt.token_ids.size() / kBlock) { continue; }
        pc::MatchResult match =
            index.match(prompt.token_ids, prompt.block_hashes, prompt.block_extras, n);
        const std::vector<VisionTokenRange> ranges =
            prompt.vision_items.empty() ? std::vector<VisionTokenRange>{} : vision_ranges(prompt);
        std::erase_if(match.candidates, [&](const pc::MatchCandidate& candidate) {
            return candidate.frontier >= n || inside_vision(candidate.frontier, ranges);
        });
        const pc::AdmissionChoice choice = index.choose(match, n);
        if (choice.candidate) { holds.push_back(match.candidates[*choice.candidate].snapshot); }
    }
    index.set_queue_holds(holds);
}

std::uint64_t ProgramImpl::hybrid_cache_epoch() const noexcept {
    return hybrid_ ? hybrid_->index().snapshot_epoch() : 0U;
}

bool ProgramImpl::hybrid_await_sibling(const PreparedPromptData& prompt, std::uint32_t reuse) {
    if (hybrid_coalesce_wait_seconds_ <= 0.0) { return false; }
    const auto n = static_cast<std::uint32_t>(prompt.token_ids.size());

    struct Wait {
        std::uint32_t lane   = 0;
        std::uint32_t target = 0;
        bool plan_tap        = false;
    };

    std::optional<Wait> best;
    for (std::uint32_t lane = 0; lane < max_concurrency; ++lane) {
        const HybridLaneState& state  = hybrid_lanes_[lane];
        const RequestControl& request = requests[lane];
        if (!state.active || !state.publish || request.lifecycle != Lifecycle::Prefilling ||
            !request.prefill) {
            continue;
        }
        const RequestControl::Prefill& prefill = *request.prefill;
        // The waiting request keeps at least one prompt token to prefill.
        const std::size_t limit = std::min<std::size_t>(n - 1U, prefill.prompt.token_ids.size());
        const auto equal_tokens = static_cast<std::uint32_t>(
            std::mismatch(prompt.token_ids.begin(), prompt.token_ids.begin() + limit,
                          prefill.prompt.token_ids.begin())
                .first -
            prompt.token_ids.begin());
        // A sibling encoding the same image saves this request the encode as well as the prefill.
        const std::uint32_t shared = media_agreed_prefix(prompt, prefill.prompt, equal_tokens);
        if (shared <= reuse) { continue; }
        const auto consider = [&](const Wait& wait) {
            if (!best || wait.target > best->target) { best = wait; }
        };
        // A new tap goes on the block boundary below the divergence: a frontier inside a block
        // publishes only once the sibling completes that block.
        std::uint32_t aligned = shared / kBlock * kBlock;
        // A block boundary inside an image both prompts carry moves to the image's end: the
        // whole image is shared, and resuming past it saves its encode as well as its prefill.
        for (const pc::TapExclusion& span : state.exclusions) {
            if (span.begin < aligned && aligned < span.end && span.end <= shared) {
                aligned = span.end;
            }
        }
        std::optional<Wait> wait;
        if (aligned > prefill.cursor && aligned > reuse) {
            wait = Wait{.lane = lane, .target = aligned, .plan_tap = true};
        }
        // An exact tap the sibling already plans near the divergence serves instead, unless
        // prefilling the tokens between it and the new tap costs more than the split.
        for (std::size_t tap = state.next_tap; tap < state.taps.size(); ++tap) {
            const pc::PlannedTap& planned = state.taps[tap];
            if (planned.position > shared) { break; }
            if (planned.placement != pc::TapPlacement::Exact ||
                planned.position <= prefill.cursor || planned.position <= reuse) {
                continue;
            }
            if (planned.position >= aligned ||
                hybrid_cost_.prefill_seconds(planned.position, aligned - planned.position) <=
                    hybrid_cost_.chunk_seconds) {
                wait = Wait{.lane = lane, .target = planned.position, .plan_tap = false};
            }
        }
        if (wait && !inside_exclusion(wait->target, state.exclusions) &&
            wait->target > state.last_capture) {
            // Waiting saves this request's prefill of the shared tokens; a new tap costs the
            // sibling one split. The sibling's remaining prefill to the snapshot is the wait,
            // doubled because decode rounds of other lanes interleave with it.
            const double saved = hybrid_cost_.prefill_seconds(reuse, wait->target - reuse);
            const double predicted =
                2.0 * hybrid_cost_.prefill_seconds(prefill.cursor, wait->target - prefill.cursor);
            if (saved > (wait->plan_tap ? hybrid_cost_.chunk_seconds : 0.0) &&
                predicted <= hybrid_coalesce_wait_seconds_) {
                consider(*wait);
                continue;
            }
        }
        // A snapshot the sibling captured inside the shared prefix but has not published yet (its
        // block or its MTP backend page is incomplete) publishes within the sibling's next chunk.
        for (const HybridPendingTap& tap : state.pending) {
            if (tap.frontier > reuse && tap.frontier <= shared) {
                consider(Wait{.lane = lane, .target = tap.frontier, .plan_tap = false});
            }
        }
    }
    if (!best) { return false; }
    // The snapshot the waiting request resumes from is where two conversations diverge: it is
    // published as a boundary, so neither lineage supersedes it.
    HybridLaneState& state = hybrid_lanes_[best->lane];
    const auto first       = state.taps.begin() + static_cast<std::ptrdiff_t>(state.next_tap);
    const auto at = std::upper_bound(first, state.taps.end(), best->target,
                                     [](std::uint32_t position, const pc::PlannedTap& planned) {
                                         return position < planned.position;
                                     });
    if (at != first && std::prev(at)->position == best->target &&
        std::prev(at)->placement == pc::TapPlacement::Exact) {
        std::prev(at)->boundary = true;
    } else if (best->plan_tap) {
        state.taps.insert(at, pc::PlannedTap{.position  = best->target,
                                             .placement = pc::TapPlacement::Exact,
                                             .boundary  = true});
    }
    for (HybridPendingTap& tap : state.pending) {
        if (tap.frontier == best->target) { tap.boundary = true; }
    }
    return true;
}

bool ProgramImpl::hybrid_make_room(std::uint32_t text_pages, std::uint32_t backend_pages) {
    const DeviceKVPagePool& text_pool = text_kv_pages->physical_pool();
    const DeviceKVPagePool* backend_pool =
        backend_kv_pages ? &backend_kv_pages->physical_pool() : nullptr;
    while (text_pool.available_pages() < text_pages ||
           (backend_pool != nullptr && backend_pool->available_pages() < backend_pages)) {
        if (!hybrid_) { return false; }
        if (hybrid_->index().evict_device_blocks(1) != 0) { continue; }
        if (!hybrid_->transfers_pending()) { return false; }
        hybrid_->drain();
    }
    return true;
}

BindingReservation
ProgramImpl::start_hybrid_binding(const RequestBasePlan& base, std::uint32_t lane,
                                  const SourceCandidate& candidate, ResumeState* resume,
                                  ExecutionUnitKind resume_kind, std::uint32_t resume_tokens) {
    if (!candidate.hybrid || !base.impl_ || !base.impl_->prompt) {
        throw std::logic_error("binding requires a prefix cache source");
    }
    if (!hybrid_ && candidate.hybrid->reuse_frontier != 0) {
        throw std::logic_error("a reusing source requires the prefix cache");
    }
    if (hybrid_) { hybrid_->poll(); }
    const PreparedPromptData& prompt = *base.impl_->prompt;
    const HybridQuoteImpl& quote     = *candidate.hybrid;
    const auto n                     = static_cast<std::uint32_t>(prompt.token_ids.size());
    const std::uint32_t reuse        = quote.reuse_frontier;
    const std::uint32_t full         = reuse / kBlock;
    const std::uint32_t tail         = reuse % kBlock;
    if (reuse >= n || (resume && reuse > resume->frontier())) { return {.source_valid = false}; }
    pc::PrefixCacheIndex* const cache = hybrid_ ? &hybrid_->index() : nullptr;

    // The quote may have gone stale since it was made; the Engine then tries its next source.
    std::vector<pc::NodeRef> path;
    pc::SnapshotView snapshot;
    if (reuse != 0) {
        if (!cache->valid(quote.snapshot)) { return {.source_valid = false}; }
        snapshot = cache->snapshot(quote.snapshot);
        if (prompt.block_hashes.size() != prompt.token_ids.size() / kBlock) {
            throw std::logic_error("prepared prompt carries no hybrid block keys");
        }
        const pc::MatchResult match =
            cache->match(prompt.token_ids, prompt.block_hashes, prompt.block_extras, n);
        if (snapshot.frontier != reuse || match.path.size() < full ||
            (full != 0 ? !(snapshot.anchor == match.path[full - 1U]) : snapshot.anchor.valid())) {
            return {.source_valid = false};
        }
        path.assign(match.path.begin(), match.path.begin() + full);
        for (const pc::NodeRef node : path) {
            const pc::CopyState device = cache->node(node).device;
            if (device != pc::CopyState::Absent && device != pc::CopyState::Resident) {
                return {.source_valid = false};
            }
        }
    }

    ContextTransaction transaction;
    transaction.kind             = ContextOperationKind::Bind;
    transaction.lane             = lane;
    transaction.epoch            = lane_epochs[lane];
    transaction.base             = base.impl_;
    transaction.resume           = resume;
    transaction.reuse_frontier   = reuse;
    transaction.backend_frontier = backend_kv_cache() ? hybrid_backend_frontier(reuse) : 0U;
    auto& binding                = transaction.hybrid.emplace();
    binding.quote                = candidate.hybrid;
    plan_binding_units(transaction, base, resume, resume_kind, resume_tokens);
    // The Begin summary names the hybrid path the Engine admitted (initialize_prefill knows only
    // the legacy Checkpoint/Root pair).
    if (reuse != 0) { requests[lane].prefill->reuse = candidate.reuse_path; }
    const UnitDemand& demand = transaction.reservation_demand;

    std::vector<pc::NodeRef> host_only;
    for (const pc::NodeRef node : path) {
        if (cache->node(node).device == pc::CopyState::Absent) { host_only.push_back(node); }
    }
    const bool tail_restore  = tail != 0 && snapshot.tail_device_copy != pc::CopyState::Resident;
    const bool image_restore = reuse != 0 && snapshot.device_slot == pc::kNoId;
    const auto restored = static_cast<std::uint32_t>(host_only.size()) + (tail_restore ? 1U : 0U);
    const auto growth   = [](std::uint32_t frontier, std::uint32_t target) {
        return kv_pages_for_frontier(std::max(frontier, target)) - kv_pages_for_frontier(frontier);
    };
    const std::uint32_t backend_reuse = transaction.backend_frontier;
    const std::uint32_t text_growth   = growth(reuse, demand.main_frontier);
    const std::uint32_t backend_growth =
        backend_kv_addresses ? growth(backend_reuse, demand.backend_frontier) : 0U;
    const std::uint32_t text_need = restored + text_growth + (tail != 0 ? 1U : 0U);
    const std::uint32_t backend_need =
        backend_kv_pages ? restored + backend_growth + (backend_reuse % kBlock != 0 ? 1U : 0U) : 0U;

    // The source is pinned while room is made, so eviction never takes a block it stands on.
    if (cache != nullptr) { cache->acquire_path(path); }
    binding.path = std::move(path);
    if (reuse != 0) {
        cache->pin_snapshot(quote.snapshot);
        binding.snapshot_pinned = true;
    }
    const auto unwind = [&] {
        abort_hybrid_binding(transaction);
        requests[lane] = {};
    };
    try {
        runtime::ContextResourceUsage shortage;
        if (!hybrid_make_room(text_need, backend_need)) {
            const auto missing = [](std::uint32_t need, std::uint32_t available) {
                return need > available ? need - available : 0U;
            };
            shortage.main_kv_pages =
                missing(text_need, text_kv_pages->physical_pool().available_pages());
            if (backend_kv_pages) {
                shortage.backend_kv_pages =
                    missing(backend_need, backend_kv_pages->physical_pool().available_pages());
            }
        }
        if (shortage.main_kv_pages || shortage.backend_kv_pages) {
            unwind();
            // Waiting helps only when the binding fits the pools at all.
            const auto usage = physical_usage();
            const bool capacity_possible =
                text_need <= usage.capacity.main_kv_pages &&
                (!backend_kv_pages || backend_need <= usage.capacity.backend_kv_pages);
            return {.capacity_possible = capacity_possible, .shortage = shortage};
        }
        transaction.reserved_state = state_store->reserve_destination();
        if (!transaction.reserved_state) {
            unwind();
            return {.shortage = {.state_slots = 1}};
        }
    } catch (...) {
        unwind();
        throw;
    }

    context_transaction_.emplace(std::move(transaction));
    auto& tx = *context_transaction_;
    auto& hb = *tx.hybrid;
    try {
        // Host restores run on the restore stream while other lanes keep executing; the restored
        // pages become the cache's Device copies at once, so the forks below can share them. The
        // lane's Device work queues behind the copies when the binding completes (§6.5).
        if (restored != 0 || image_restore) {
            std::optional<DeviceKVPageReservation> text_pages =
                text_kv_pages->physical_pool().reserve(restored);
            std::optional<DeviceKVPageReservation> backend_pages;
            if (backend_kv_pages) {
                backend_pages = backend_kv_pages->physical_pool().reserve(restored);
            }
            if ((restored != 0 && !text_pages) ||
                (backend_kv_pages && restored != 0 && !backend_pages)) {
                throw std::logic_error("hybrid binding lost the room it made for its restore");
            }
            const std::uint64_t block_bytes = hybrid_->host_layout().block_payload_bytes;
            hb.restore_bytes =
                static_cast<std::uint64_t>(restored) * block_bytes +
                (image_restore ? static_cast<std::uint64_t>(hybrid_->host_layout().image_bytes)
                               : 0U);
            hybrid_->open_restore(device.stream);
            hb.restore_staged      = true;
            const auto destination = [&](std::uint32_t columns) {
                HybridBlockPages pages{
                    .text = text_kv_pages->materialize_transfer_destination(*text_pages, columns)};
                if (backend_kv_pages) {
                    pages.backend =
                        backend_kv_pages->materialize_transfer_destination(*backend_pages, columns);
                }
                return pages;
            };
            for (const pc::NodeRef node : host_only) {
                hybrid_->restore_block(node, destination(kBlock));
            }
            if (tail_restore) { hybrid_->restore_tail(quote.snapshot, destination(tail)); }
            if (image_restore) {
                hybrid_->restore_image(quote.snapshot,
                                       state_store->physical_slot(*tx.reserved_state));
                hb.state_restored = true;
            }
            hybrid_->submit_restore();
            ++tx.operations.state_restores;
        }

        // KV: cached full pages are shared; a partial tail is copied into a private page.
        const auto main = text_kv_addresses->create_inactive();
        if (!main) { throw std::logic_error("hybrid binding exhausted bounded KV descriptors"); }
        tx.reserved_kv = SequenceKVBundle{.text = *main};
        if (backend_kv_addresses) {
            const auto backend = backend_kv_addresses->create_inactive();
            if (!backend) {
                throw std::logic_error("hybrid binding exhausted backend descriptors");
            }
            tx.reserved_kv->backend = *backend;
        }
        const pc::SnapshotView landed =
            reuse != 0 ? cache->snapshot(quote.snapshot) : pc::SnapshotView{};
        const auto prepare = [&](KVAddressSpaceStore& addresses, KVAddressSpaceHandle address,
                                 std::uint32_t frontier, std::uint32_t growth_pages, bool backend,
                                 std::optional<KVPagePrefixForkReservation>& fork,
                                 std::optional<KVActivationReservation>& activation) {
            if (frontier == 0) {
                activation.emplace(addresses.prepare_activation(address, growth_pages,
                                                                static_cast<std::int32_t>(lane)));
                return;
            }
            const std::uint32_t full_pages   = frontier / kBlock;
            const std::uint32_t tail_columns = frontier % kBlock;
            std::vector<LogicalKVPageHandle> shared;
            shared.reserve(full_pages);
            for (std::uint32_t page = 0; page < full_pages; ++page) {
                const HybridBlockPages& block = hybrid_->block(cache->node(hb.path[page]).device_id);
                shared.push_back(backend ? *block.backend : block.text);
            }
            std::optional<LogicalKVPageHandle> tail_source;
            if (tail_columns != 0) {
                const HybridBlockPages& block =
                    full_pages < full ? hybrid_->block(cache->node(hb.path[full_pages]).device_id)
                                      : hybrid_->block(landed.tail_device);
                tail_source = backend ? *block.backend : block.text;
            }
            fork.emplace(addresses.prepare_page_prefix_fork(address, shared, tail_source,
                                                            tail_columns, growth_pages,
                                                            static_cast<std::int32_t>(lane)));
        };
        prepare(*text_kv_addresses, tx.reserved_kv->text, reuse, text_growth, false, hb.text_fork,
                tx.text_activation);
        if (tx.reserved_kv->backend) {
            prepare(*backend_kv_addresses, *tx.reserved_kv->backend, backend_reuse, backend_growth,
                    true, hb.backend_fork, tx.backend_activation);
        }
        requests[lane].lifecycle = Lifecycle::Binding;
        return {.reserved = true};
    } catch (...) {
        abort_context();
        throw;
    }
}

void ProgramImpl::abort_hybrid_binding(ContextTransaction& tx) noexcept {
    if (!hybrid_ || !tx.hybrid) { return; }
    auto& hb = *tx.hybrid;
    try {
        // A submitted restore may still be writing the reserved state slot and the destination
        // pages; it returns every destination once its copies are done.
        if (hb.restore_staged) {
            hybrid_->abort_restore();
            hb.restore_staged = false;
        }
        hb.text_fork.reset();
        hb.backend_fork.reset();
        pc::PrefixCacheIndex& index = hybrid_->index();
        if (hb.snapshot_pinned && index.valid(hb.quote->snapshot)) {
            index.unpin_snapshot(hb.quote->snapshot);
        }
        hb.snapshot_pinned = false;
        if (!hb.path.empty()) { index.release_path(hb.path); }
        hb.path.clear();
    } catch (...) { std::terminate(); }
}

void ProgramImpl::complete_hybrid_binding(ContextTransaction& tx, ContextProgress& out) {
    auto& hb                         = *tx.hybrid;
    const HybridQuoteImpl& quote     = *hb.quote;
    const std::uint32_t lane         = tx.lane;
    const std::uint32_t reuse        = quote.reuse_frontier;
    const RequestBasePlanImpl& base  = *tx.base;
    const PreparedPromptData& prompt = *base.prompt;
    const auto n                     = static_cast<std::uint32_t>(prompt.token_ids.size());

    // A Host restore does not hold the binding: the lane's Device work queues behind its prelude
    // now and each layer of the first prefill pass behind that layer's copies (§6.5).
    std::uint64_t ticket = 0;
    if (hb.restore_staged) {
        ticket            = hybrid_->land_restore(device.stream);
        hb.restore_staged = false;
    }
    const auto commit = [&](KVAddressSpaceStore& addresses, LogicalKVPageStore& pages,
                            std::optional<KVPagePrefixForkReservation>& fork,
                            std::optional<KVActivationReservation>& activation) {
        if (fork) {
            if (fork->needs_tail_copy()) {
                pages.physical_pool().copy_page(addresses.page_prefix_fork_tail_source(*fork),
                                                addresses.page_prefix_fork_tail_destination(*fork),
                                                device.stream);
                ++tx.operations.partial_tail_cow_pages;
            }
            addresses.commit_page_prefix_fork(std::move(*fork), device.stream);
            fork.reset();
        } else if (activation) {
            addresses.commit_activation(std::move(*activation), device.stream);
            activation.reset();
        }
    };
    commit(*text_kv_addresses, *text_kv_pages, hb.text_fork, tx.text_activation);
    if (tx.reserved_kv->backend) {
        commit(*backend_kv_addresses, *backend_kv_pages, hb.backend_fork, tx.backend_activation);
    }
    auto history     = std::make_shared<KVHistory>();
    history->text    = tx.reserved_kv->text;
    history->backend = tx.reserved_kv->backend;
    history->owner   = this;
    tx.reserved_kv.reset();
    // From here an abort releases the lane's state and KV.
    tx.adopted                   = true;
    SequenceState& state         = sequences[lane];
    state.kv                     = history;
    const StateImageHandle image = *tx.reserved_state;
    tx.reserved_state.reset();
    state.state = ActiveStateBinding{.read = image, .write = image};
    if (reuse == 0) {
        state_store->activate_reset(image, device.stream);
    } else if (hb.state_restored) {
        state_store->activate_copied(image);
    } else {
        state_images->copy_slot(state_store->physical_slot(hybrid_->image(quote.snapshot)),
                                state_store->physical_slot(image), device.stream);
        state_store->activate_copied(image);
        ++tx.operations.state_forks;
    }
    install_binding(tx);
    // The snapshot image's continuation hidden belongs to the snapshot, not to a committed tail.
    state.tail_hidden_valid = false;
    if (!hybrid_) {
        // Without a prefix cache the lane publishes nothing.
        out.sequence  = sequence_handle(lane);
        out.replaying = requests[lane].lifecycle == Lifecycle::Replaying;
        return;
    }
    pc::PrefixCacheIndex& index = hybrid_->index();

    // Taps are planned over the prompt the lane prefills, past its reuse frontier.
    const std::vector<VisionTokenRange> ranges = vision_ranges(prompt);
    const bool publish = base.summary.publish_continuation && prompt.identity.reusable;
    std::vector<pc::TapExclusion> exclusions;
    exclusions.reserve(ranges.size());
    for (const VisionTokenRange& range : ranges) {
        exclusions.push_back(pc::TapExclusion{range.begin, range.end});
    }
    std::vector<pc::PlannedTap> taps;
    if (publish && !tx.resume) {
        // A request that resumed from a previous generation's endpoint proves its client echoes
        // generated turns token for token (agent loops, preserved reasoning): its own endpoint
        // will serve the next turn, so the generation opener is not worth the prefill split an
        // exact tap costs. Otherwise the history is re-rendered (for example with reasoning
        // stripped) and the next turn diverges at the opener.
        std::vector<pc::TapHint> hints = prompt.tap_hints.hints;
        if (reuse != 0 && index.snapshot(quote.snapshot).kind == pc::SnapshotKind::Endpoint) {
            std::erase_if(hints, [](const pc::TapHint& hint) {
                return hint.kind == pc::TapHintKind::GenerationOpener;
            });
        }
        const std::array<std::uint32_t, 1> existing{reuse};
        taps = pc::plan_taps(n, reuse, hints,
                             reuse != 0 ? std::span<const std::uint32_t>(existing)
                                        : std::span<const std::uint32_t>(),
                             exclusions, hybrid_->taps());
    }
    const std::uint32_t full    = reuse / kBlock;
    HybridLaneState& lane_state = hybrid_lanes_[lane];
    lane_state                  = HybridLaneState{};
    lane_state.taps.swap(taps);
    lane_state.exclusions.swap(exclusions);
    lane_state.prompt_extras = prompt.block_extras;
    lane_state.path.swap(hb.path);
    lane_state.active           = true;
    lane_state.publish          = publish;
    lane_state.path_hash        = full == 0 ? pc::kRootLookupHash : prompt.block_hashes[full - 1U];
    lane_state.trailing_extra   = all_vision_key(ranges);
    lane_state.deepest_snapshot = reuse;
    lane_state.resume_snapshot  = reuse != 0 ? quote.snapshot : pc::SnapshotRef{};
    lane_state.resume_frontier  = reuse;
    lane_state.last_capture     = reuse;
    lane_state.restore_ticket   = ticket;
    lane_state.restore_layers_pending = ticket != 0;

    HybridCacheCounters& counters = hybrid_->counters();
    ++counters.admissions;
    if (reuse != 0) {
        index.note_hit(quote.snapshot);
        ++counters.snapshot_hits;
        counters.reused_tokens += reuse;
        index.unpin_snapshot(quote.snapshot);
        hb.snapshot_pinned = false;
    }
    out.sequence  = sequence_handle(lane);
    out.replaying = requests[lane].lifecycle == Lifecycle::Replaying;
}

// ---- publication -------------------------------------------------------------------------------

void ProgramImpl::hybrid_publish_blocks(SequenceState& sequence) {
    if (!hybrid_ || sequence.lane >= max_concurrency) { return; }
    HybridLaneState& lane = hybrid_lanes_[sequence.lane];
    if (!lane.active || !lane.publish || !sequence.kv) { return; }
    hybrid_->poll();
    std::uint32_t frontier = sequence.text_kv_valid;
    // Every enabled pool must hold the whole block. The DFlash2 draft ring is StateImage content,
    // not block content, so it does not bound block publication.
    if (sequence.kv->backend) { frontier = std::min(frontier, backend_kv_valid(sequence)); }
    frontier =
        std::min<std::uint32_t>(frontier, static_cast<std::uint32_t>(sequence.ledger.size()));
    const std::uint32_t full = frontier / kBlock;
    while (lane.path.size() < full) {
        const auto block = static_cast<std::uint32_t>(lane.path.size());
        const std::span<const TokenId> tokens(
            sequence.ledger.data() + static_cast<std::size_t>(block) * kBlock, kBlock);
        const std::uint64_t extra =
            block < lane.prompt_extras.size() ? lane.prompt_extras[block] : lane.trailing_extra;
        const std::uint64_t hash = pc::block_lookup_hash(lane.path_hash, tokens, extra);
        HybridBlockPages pages{.text = text_kv_addresses->logical_page(sequence.kv->text, block)};
        if (sequence.kv->backend) {
            pages.backend = backend_kv_addresses->logical_page(*sequence.kv->backend, block);
        }
        const pc::NodeRef parent      = lane.path.empty() ? pc::NodeRef{} : lane.path.back();
        const pc::InsertResult result = hybrid_->insert_block(parent, hash, tokens, extra, pages);
        lane.path.push_back(result.node);
        lane.path_hash = hash;
    }
    if (!lane.pending.empty()) { hybrid_publish_pending(sequence, false); }
}

std::optional<std::uint32_t> ProgramImpl::hybrid_copy_tail(const HybridBlockPages& source,
                                                           std::uint32_t columns) {
    const bool backend_pool = backend_kv_pages != nullptr;
    if (!hybrid_make_room(1U, backend_pool ? 1U : 0U)) { return std::nullopt; }
    std::optional<DeviceKVPageReservation> text_reservation =
        text_kv_pages->physical_pool().reserve(1U);
    std::optional<DeviceKVPageReservation> backend_reservation;
    if (!text_reservation) { return std::nullopt; }
    if (backend_pool) {
        backend_reservation = backend_kv_pages->physical_pool().reserve(1U);
        if (!backend_reservation) { return std::nullopt; }
    }
    HybridBlockPages copy{
        .text = text_kv_pages->materialize_transfer_destination(*text_reservation, columns)};
    if (backend_pool) {
        copy.backend =
            backend_kv_pages->materialize_transfer_destination(*backend_reservation, columns);
    }
    // Each tail owns its bundle, so evicting any cache entry returns exactly one page per pool.
    text_kv_pages->physical_pool().copy_page(text_kv_pages->physical(source.text),
                                             text_kv_pages->physical(copy.text), device.stream);
    if (backend_pool) {
        backend_kv_pages->physical_pool().copy_page(backend_kv_pages->physical(*source.backend),
                                                    backend_kv_pages->physical(*copy.backend),
                                                    device.stream);
    }
    return hybrid_->register_tail_copy(copy, columns);
}

void ProgramImpl::hybrid_publish_pending(SequenceState& sequence, bool finishing) {
    HybridLaneState& lane         = hybrid_lanes_[sequence.lane];
    pc::PrefixCacheIndex& index   = hybrid_->index();
    HybridCacheCounters& counters = hybrid_->counters();
    const auto drop               = [&](const HybridPendingTap& tap) {
        (void)state_store->release(tap.image);
        index.release_device_slot(tap.slot);
        ++counters.taps_skipped;
    };
    while (!lane.pending.empty()) {
        const HybridPendingTap tap  = lane.pending.front();
        const std::uint32_t full    = tap.frontier / kBlock;
        const std::uint32_t tail    = tap.frontier % kBlock;
        const bool anchored         = lane.path.size() >= full;
        const bool tail_block_ready = tail == 0 || lane.path.size() > full;
        std::optional<HybridBlockPages> tail_source;
        if (anchored && tail != 0) {
            if (tail_block_ready) {
                tail_source = hybrid_->block(index.node(lane.path[full]).device_id);
            } else if (finishing && sequence.kv && sequence.text_kv_valid >= tap.frontier &&
                       (!sequence.kv->backend ||
                        backend_kv_valid(sequence) >= hybrid_backend_frontier(tap.frontier))) {
                // The finishing lane's own partial page holds the tail; it is copied before the
                // lane releases it.
                HybridBlockPages own{.text =
                                         text_kv_addresses->logical_page(sequence.kv->text, full)};
                if (sequence.kv->backend) {
                    own.backend = backend_kv_addresses->logical_page(*sequence.kv->backend, full);
                }
                tail_source = own;
            }
        }
        if (!anchored || (tail != 0 && !tail_source)) {
            if (!finishing) { return; }
            lane.pending.erase(lane.pending.begin());
            drop(tap);
            continue;
        }
        lane.pending.erase(lane.pending.begin());
        std::optional<std::uint32_t> tail_id;
        if (tail != 0) {
            try {
                tail_id = hybrid_copy_tail(*tail_source, tail);
            } catch (...) {
                drop(tap);
                throw;
            }
            if (!tail_id) {
                drop(tap);
                continue;
            }
        }
        const pc::NodeRef anchor = full == 0 ? pc::NodeRef{} : lane.path[full - 1U];
        const std::span<const TokenId> tail_tokens(
            sequence.ledger.data() + static_cast<std::size_t>(full) * kBlock, tail);
        // publish_snapshot consumes the staging slot and the tail id on every outcome.
        const pc::PublishResult published = index.publish_snapshot(
            anchor, tap.frontier, tail_tokens, tail_id, tap.slot,
            tap.boundary ? pc::SnapshotKind::Boundary : pc::SnapshotKind::Tap);
        if (!published.created) {
            (void)state_store->release(tap.image);
            ++counters.taps_skipped;
            continue;
        }
        hybrid_->attach_image(published.snapshot, tap.image);
        lane.deepest_snapshot = std::max(lane.deepest_snapshot, tap.frontier);
        ++counters.taps_created;
        if (!tap.boundary) {
            // A previous tap still pending when this one was captured is superseded here, before
            // the Host write below can take its slabs.
            hybrid_supersede_tap(lane, tap.frontier);
            lane.tap_snapshot = published.snapshot;
            lane.tap_frontier = tap.frontier;
        }
        (void)hybrid_->start_snapshot_host_write(published.snapshot, device.stream,
                                                 device.transfer_stream);
    }
}

void ProgramImpl::hybrid_capture_tap(SequenceState& sequence, std::uint32_t frontier,
                                     bool boundary) {
    HybridLaneState& lane         = hybrid_lanes_[sequence.lane];
    pc::PrefixCacheIndex& index   = hybrid_->index();
    HybridCacheCounters& counters = hybrid_->counters();
    if (frontier == 0 || frontier > sequence.text_kv_valid) {
        ++counters.taps_skipped;
        return;
    }
    hybrid_supersede_resume(lane, frontier, sequence.ledger);
    hybrid_supersede_tap(lane, frontier);
    const std::optional<std::uint32_t> slot = index.acquire_device_slot(
        index.estimate_priority(lane.last_capture, frontier, frontier % kBlock != 0));
    if (!slot) {
        ++counters.taps_skipped;
        return;
    }
    const std::optional<StateImageHandle> image = state_store->reserve_destination();
    if (!image) {
        index.release_device_slot(*slot);
        ++counters.taps_skipped;
        return;
    }
    try {
        state_images->copy_slot(state_store->physical_slot(sequence.state.write),
                                state_store->physical_slot(*image), device.stream);
        state_store->publish_copied_snapshot(*image);
        lane.pending.push_back(HybridPendingTap{
            .frontier = frontier, .image = *image, .slot = *slot, .boundary = boundary});
    } catch (...) {
        (void)state_store->release(*image);
        index.release_device_slot(*slot);
        throw;
    }
    lane.last_capture = frontier;
    hybrid_publish_pending(sequence, false);
}

void ProgramImpl::hybrid_after_prefill_chunk(SequenceState& sequence, std::uint32_t cursor,
                                             std::uint32_t prompt_tokens) {
    if (!hybrid_ || sequence.lane >= max_concurrency) { return; }
    HybridLaneState& lane = hybrid_lanes_[sequence.lane];
    if (!lane.active || !lane.publish || cursor >= prompt_tokens) { return; }
    // The next chunk reaches the prompt end, so a flexible tap inside it is realized here.
    const bool final_chunk_next = prompt_tokens - cursor <= prefill_chunk;
    bool exact                  = false;
    bool flexible               = false;
    bool boundary               = false;
    while (lane.next_tap < lane.taps.size()) {
        const pc::PlannedTap& tap = lane.taps[lane.next_tap];
        if (tap.placement == pc::TapPlacement::Exact) {
            if (tap.position > cursor) { break; }
            exact    = exact || tap.position == cursor;
            boundary = boundary || (tap.position == cursor && tap.boundary);
        } else if (tap.position > cursor && !final_chunk_next) {
            break;
        } else {
            flexible = true;
        }
        ++lane.next_tap;
    }
    if (!exact && !flexible) { return; }
    if (inside_exclusion(cursor, lane.exclusions) || cursor <= lane.last_capture ||
        (!exact && cursor - lane.last_capture < pc::kMinimumTapSeparation)) {
        ++hybrid_->counters().taps_skipped;
        return;
    }
    hybrid_capture_tap(sequence, cursor, boundary);
}

void ProgramImpl::hybrid_supersede_resume(HybridLaneState& lane, std::uint32_t frontier,
                                          std::span<const TokenId> ledger) {
    pc::PrefixCacheIndex& index = hybrid_->index();
    if (!lane.resume_snapshot.valid() || frontier <= lane.resume_frontier ||
        !index.valid(lane.resume_snapshot)) {
        return;
    }
    // The lineage's tokens past its source, so the endpoints it went past are superseded too.
    const std::size_t end = std::min<std::size_t>(frontier, ledger.size());
    index.supersede(lane.resume_snapshot,
                    end > lane.resume_frontier
                        ? ledger.subspan(lane.resume_frontier, end - lane.resume_frontier)
                        : std::span<const TokenId>{});
    lane.resume_snapshot = {};
}

void ProgramImpl::hybrid_supersede_tap(HybridLaneState& lane, std::uint32_t frontier) {
    pc::PrefixCacheIndex& index = hybrid_->index();
    if (!lane.tap_snapshot.valid() || frontier <= lane.tap_frontier ||
        !index.valid(lane.tap_snapshot)) {
        return;
    }
    index.supersede(lane.tap_snapshot);
    lane.tap_snapshot = {};
}

std::span<const cudaEvent_t> ProgramImpl::hybrid_take_restore_layers(std::uint32_t lane) {
    if (!hybrid_ || lane >= max_concurrency) { return {}; }
    HybridLaneState& state = hybrid_lanes_[lane];
    if (!std::exchange(state.restore_layers_pending, false)) { return {}; }
    return hybrid_->restore_layer_events(state.restore_ticket);
}

void ProgramImpl::hybrid_release_lane(std::uint32_t lane) noexcept {
    if (!hybrid_ || lane >= max_concurrency) { return; }
    HybridLaneState& state = hybrid_lanes_[lane];
    try {
        // The state slot the lane returns may still be a destination of its restore (a lane can
        // end before its first prefill pass waits for the copies), so later Device work waits
        // for whatever of the restore is still landing.
        if (state.restore_ticket != 0) {
            hybrid_->order_after_restore(state.restore_ticket, device.stream);
        }
        pc::PrefixCacheIndex& index = hybrid_->index();
        for (const HybridPendingTap& tap : state.pending) {
            (void)state_store->release(tap.image);
            index.release_device_slot(tap.slot);
        }
        state.pending.clear();
        if (state.active) {
            // The lane's blocks become evictable once its pins go: back them on the Host tier
            // first, in one batch, so eviction drops Device copies instead of losing nodes.
            try {
                hybrid_->start_block_host_writes(state.path, device.stream, device.transfer_stream);
            } catch (...) {}
            index.release_path(state.path);
        }
    } catch (...) { std::terminate(); }
    state = HybridLaneState{};
}

void ProgramImpl::hybrid_finish_lane(SequenceState& sequence, bool endpoint) noexcept {
    if (!hybrid_ || sequence.lane >= max_concurrency) { return; }
    const std::uint32_t lane    = sequence.lane;
    HybridLaneState& lane_state = hybrid_lanes_[lane];
    try {
        if (lane_state.active && lane_state.publish && sequence.kv) {
            hybrid_publish_blocks(sequence);
            hybrid_publish_pending(sequence, true);
            const std::uint32_t frontier         = sequence.text_kv_valid;
            const std::uint32_t backend_frontier = hybrid_backend_frontier(frontier);
            const bool backend_caught_up =
                (!sequence.kv->backend || backend_kv_valid(sequence) >= backend_frontier) &&
                (!is_masked_draft_backend(speculative_backend) ||
                 sequence.dflash_context_frontier == frontier);
            const std::uint32_t anchor_blocks = frontier / kBlock;
            if (endpoint && backend_caught_up && frontier != 0 &&
                frontier >= lane_state.deepest_snapshot + pc::kMinimumTapSeparation &&
                frontier <= sequence.ledger.size() && lane_state.path.size() >= anchor_blocks &&
                sequence.state.read == sequence.state.write &&
                state_store->role(sequence.state.write) == StateImageRole::ActiveMutable) {
                pc::PrefixCacheIndex& index = hybrid_->index();
                // The lineage moves past its source here, so a slot it needs comes from its own
                // lineage before another conversation's snapshot.
                hybrid_supersede_resume(lane_state, frontier, sequence.ledger);
                const std::optional<std::uint32_t> slot =
                    index.acquire_device_slot(index.estimate_priority(
                        lane_state.deepest_snapshot, frontier, frontier % kBlock != 0));
                if (slot) {
                    const std::uint32_t tail = frontier % kBlock;
                    std::optional<std::uint32_t> tail_id;
                    try {
                        if (tail != 0) {
                            // The finishing lane hands its last partial page to the snapshot.
                            HybridBlockPages pages{.text = text_kv_addresses->logical_page(
                                                       sequence.kv->text, anchor_blocks)};
                            std::uint32_t backend_columns = 0;
                            if (sequence.kv->backend) {
                                pages.backend = backend_kv_addresses->logical_page(
                                    *sequence.kv->backend, anchor_blocks);
                                backend_columns = backend_frontier > anchor_blocks * kBlock
                                                      ? backend_frontier - anchor_blocks * kBlock
                                                      : 0U;
                            }
                            tail_id = hybrid_->register_tail(pages, tail, backend_columns);
                        }
                    } catch (...) {
                        index.release_device_slot(*slot);
                        throw;
                    }
                    const StateImageHandle image = sequence.state.write;
                    state_store->freeze(image);
                    const std::span<const TokenId> tail_tokens(
                        sequence.ledger.data() + static_cast<std::size_t>(anchor_blocks) * kBlock,
                        tail);
                    const pc::NodeRef anchor =
                        anchor_blocks == 0 ? pc::NodeRef{} : lane_state.path[anchor_blocks - 1U];
                    const pc::PublishResult published = index.publish_snapshot(
                        anchor, frontier, tail_tokens, tail_id, *slot, pc::SnapshotKind::Endpoint);
                    if (published.snapshot.valid()) {
                        lane_state.deepest_snapshot =
                            std::max(lane_state.deepest_snapshot, frontier);
                    }
                    if (published.created) {
                        hybrid_->attach_image(published.snapshot, image);
                        sequence.state = {};
                        ++hybrid_->counters().endpoints_created;
                        // Superseded first, so the Host write below can take its slabs.
                        hybrid_supersede_resume(lane_state, lane_state.deepest_snapshot,
                                                sequence.ledger);
                        (void)hybrid_->start_snapshot_host_write(published.snapshot, device.stream,
                                                                 device.transfer_stream);
                    } else {
                        state_store->thaw(image);
                    }
                }
            }
            hybrid_supersede_resume(lane_state, lane_state.deepest_snapshot,
                                    sequence.ledger);
        }
    } catch (...) {
        // Terminal publication is optional; a failure keeps whatever was published and releases
        // the lane normally.
    }
    hybrid_release_lane(lane);
}

// ---- Device pressure and statistics ------------------------------------------------------------

bool ProgramImpl::hybrid_reclaim(runtime::ContextResourceUsage shortage) {
    if (!hybrid_) { return false; }
    hybrid_->poll();
    const DeviceKVPagePool& text_pool = text_kv_pages->physical_pool();
    const DeviceKVPagePool* backend_pool =
        backend_kv_pages ? &backend_kv_pages->physical_pool() : nullptr;
    const std::uint32_t text_target = text_pool.available_pages() + shortage.main_kv_pages;
    const std::uint32_t backend_target =
        backend_pool != nullptr ? backend_pool->available_pages() + shortage.backend_kv_pages : 0U;
    std::uint32_t released = 0;
    while (text_pool.available_pages() < text_target ||
           (backend_pool != nullptr && backend_pool->available_pages() < backend_target)) {
        const std::uint32_t count = hybrid_->index().evict_device_blocks(1);
        if (count == 0) {
            // Blocks pinned only by their in-flight Host writes become evictable once those land.
            if (!hybrid_->transfers_pending()) { break; }
            hybrid_->drain();
            continue;
        }
        released += count;
    }
    return released != 0;
}

HybridPrefixCacheStats ProgramImpl::hybrid_stats() const noexcept {
    HybridPrefixCacheStats out;
    if (!hybrid_) { return out; }
    const pc::PrefixIndexStats index    = hybrid_->index().stats();
    const HybridCacheCounters& counters = hybrid_->counters();
    const pc::PrefixIndexConfig& config = hybrid_->index().config();
    out.nodes                           = index.nodes;
    out.snapshots                       = index.snapshots;
    out.device_resident_blocks          = index.device_resident_blocks;
    out.device_evictable_blocks         = index.device_evictable_blocks;
    out.host_slabs                      = config.host_slabs;
    out.host_free_slabs                 = index.host_free_slabs;
    out.host_slab_bytes                 = hybrid_->host_layout().slab_bytes;
    out.free_device_snapshot_slots      = index.free_device_slots;
    out.admissions                      = counters.admissions;
    out.snapshot_hits                   = counters.snapshot_hits;
    out.reused_tokens                   = counters.reused_tokens;
    out.blocks_inserted                 = counters.blocks_inserted;
    out.blocks_reattached               = counters.blocks_reattached;
    out.blocks_duplicate                = counters.blocks_duplicate;
    out.taps_created                    = counters.taps_created;
    out.taps_skipped                    = counters.taps_skipped;
    out.endpoints_created               = counters.endpoints_created;
    out.host_image_writes               = counters.host_image_writes;
    out.host_block_writes               = counters.host_block_writes;
    out.host_image_restores             = counters.host_image_restores;
    out.host_block_restores             = counters.host_block_restores;
    out.prefetched_blocks               = counters.prefetched_blocks;
    out.host_tail_restores              = counters.host_tail_restores;
    out.host_write_bytes                = counters.host_write_bytes;
    out.host_restore_bytes              = counters.host_restore_bytes;
    out.evicted_blocks                  = counters.evicted_blocks;
    out.host_snapshot_evictions         = index.host_snapshot_evictions;
    out.host_dead_reclaims              = index.host_dead_reclaims;
    out.unbacked_node_losses            = index.unbacked_node_losses;
    out.held_snapshots                  = index.held_snapshots;
    out.held_device_evictions           = index.held_device_evictions;
    out.held_snapshot_losses            = index.held_snapshot_losses;
    out.held_host_refusals              = index.held_host_refusals;
    return out;
}

} // namespace infernix::models::qwen3_5::detail
