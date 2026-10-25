// The layer walk of Qwen4Exp's prefill (docs/maintainer/qwen3_8-flash-next-design.md §19.3.8 F4):
// a span of consecutive streamed calls of one prompt runs layer-major, so each decoder layer's
// non-resident experts cross PCIe once per span instead of once per call. A chunk-major prompt of
// 4,096-token calls re-streams nearly every non-resident expert for every call (~51 GB at the x8
// link's 27.6 GB/s, ~1.8 s per call), more than the call's compute; a span of 16 calls streams them
// once and is bound by compute.
//
// The span keeps the prompt's call grid: chunk c of layer l reads exactly what the chunk-major
// call c reads at layer l (its own residual columns, and the layer's KV, GDN, PLE and QSA state
// after chunks 0..c-1), so every output bit equals the chunk-major prefill's. Only placement
// differs (resident, streamed or staged experts), which the MoE's arithmetic does not see. A
// layer's MoE runs once per group of consecutive chunks (walk_.groups, up to kWalkMoeColumns
// columns): every MoE step acts column by column, so grouping changes no bit either.
//
// The span's residual stream (BF16 [S*H, tokens]) and every chunk's staged io image live in the
// stream lease after the ring. Each advance_prefill call enqueues about one call's work (a few
// layers of every chunk) and synchronizes, so cancellation and other lanes interleave as between
// chunk-major calls; the walk's own state stays valid across them. A span ends before a call too
// narrow to stream and at kWalkMaxTokens. A prefix tap due at an inner chunk end is captured in place
// (walk_plan_taps): each layer's state is copied once the walk has enqueued that layer of the chunk,
// before the next chunk overwrites it, and the snapshot is committed after the span. Ending the span
// there instead would stream every non-resident expert once more per tap (a cold 128K prompt's taps
// took ~2.4 s, 14 %, that way).

#include "models/qwen4_exp/program/program_impl.h"

#include "core/temporary_switches.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace infernix::models::qwen4_exp::detail {

namespace pc = runtime::prefix_cache;

namespace {

constexpr std::size_t kWalkAlign = 256;
constexpr std::uint32_t kBlock   = pc::kBlockTokens;

std::size_t walk_align(std::size_t bytes) { return (bytes + kWalkAlign - 1) / kWalkAlign * kWalkAlign; }

} // namespace

std::size_t ProgramImpl::walk_bytes(std::uint32_t tokens, std::size_t chunks) const noexcept {
    // The residual stream, every chunk's io image, and an MoE group's input (xm BF16 [H, columns],
    // inject FP32 [S, columns]).
    const std::size_t moe = static_cast<std::size_t>(kWalkMoeColumns) * (2ULL * c_.hidden_size + 4ULL * c_.hc.streams);
    return walk_align(2ULL * static_cast<std::size_t>(width_) * tokens) + chunks * walk_align(io_layout_.bytes) +
           walk_align(moe);
}

// The exclusive end of the span that starts at lane.next_call; lane.next_call when the call runs
// chunk-major. Lends the stream lease (with the walk's area) when none is held, or lends it again
// with the area when the held one has none for the span (another lane's chunk-major calls lent
// it). Another prefilling lane waits while the span runs (context_blocks), so two long prompts
// walk span by span instead of both re-streaming every expert per call.
std::size_t ProgramImpl::walk_span_end(const Lane& lane, std::uint32_t index) {
    const std::size_t first = lane.next_call;
    if (!expert_stream_ || residency_->frames() == 0 || walk_.active) { return first; }
    std::size_t end     = first;
    std::uint32_t from  = lane.state_tokens;
    std::uint32_t total = 0;
    std::vector<std::uint32_t> tokens{0}; // tokens of the span [first, first + i)
    while (end < lane.calls.size()) {
        const std::uint32_t width = lane.calls[end] - from;
        if (width < static_cast<std::uint32_t>(kStreamMinColumns) || total + width > kWalkMaxTokens) { break; }
        total += width;
        from = lane.calls[end];
        ++end;
        tokens.push_back(total);
    }
    if (end - first < 2) { return first; }
    if (stream_lease_.valid() && walk_bytes(tokens[2], 2) > stream_walk_bytes_) {
        // Between calls nothing reads the lease; its frames go back and come out again with the area.
        return_stream_lease();
    }
    if (!stream_lease_.valid()) {
        // Sized for the prompt's largest span: as many tokens as remain (at most the cap) in calls
        // of the configured chunk, plus a remainder call.
        const std::uint32_t remaining = std::min(lane.prefill_end - lane.state_tokens, kWalkMaxTokens);
        const std::size_t chunks      = remaining / static_cast<std::uint32_t>(chunk_) + 2;
        if (!stream_chunk(static_cast<std::int32_t>(lane.calls[first] - lane.state_tokens),
                          walk_bytes(remaining, chunks))) {
            return first;
        }
    }
    while (end - first >= 2 && walk_bytes(tokens[end - first], end - first) > stream_walk_bytes_) { --end; }
    return end - first >= 2 ? end : first;
}

void ProgramImpl::walk_begin(Lane& lane, std::uint32_t index, std::size_t end) {
    const cudaStream_t s = device_.stream;
    walk_                = LayerWalk{};
    walk_.lane           = index;
    walk_.first_call     = lane.next_call;
    walk_.end_call       = end;
    const std::size_t n  = end - lane.next_call;
    std::int32_t begin = static_cast<std::int32_t>(lane.state_tokens), offset = 0;
    for (std::size_t c = lane.next_call; c < end; ++c) {
        const auto width = static_cast<std::int32_t>(lane.calls[c]) - begin;
        walk_.begins.push_back(begin);
        walk_.widths.push_back(width);
        walk_.offsets.push_back(offset);
        begin += width;
        offset += width;
    }
    walk_.io_stride      = walk_align(io_layout_.bytes);
    walk_.residual_bytes = walk_align(2ULL * static_cast<std::size_t>(width_) * static_cast<std::size_t>(offset));
    walk_.moe_offset     = walk_.residual_bytes + n * walk_.io_stride;
    // MoE groups: consecutive chunks while their columns fit kWalkMoeColumns and the arena the calls
    // run in (WideWork of the span's widest chunk: the wide arena when it exceeds the static one's).
    const std::int32_t widest = *std::max_element(walk_.widths.begin(), walk_.widths.end());
    const std::size_t arena   = widest > plan_.static_columns && wide_work_ > 0 ? wide_work_ : plan_.work;
    for (std::size_t c = 0; c < n;) {
        std::size_t e       = c + 1;
        std::int32_t columns = walk_.widths[c];
        while (e < n && columns + walk_.widths[e] <= kWalkMoeColumns &&
               execution::Forward::moe_workspace_bytes(c_, columns + walk_.widths[e], wide_a16(options_)) <= arena) {
            columns += walk_.widths[e];
            ++e;
        }
        walk_.groups.emplace_back(c, e);
        c = e;
    }
    walk_.layers_per_step =
        static_cast<std::uint32_t>((c_.num_hidden_layers + n - 1) / n); // about one call's work per step
    walk_.mtp.assign(n, std::nullopt);
    walk_.vision.assign(n, std::nullopt);
    walk_.mtp_vision.assign(n, {});
    walk_plan_taps(lane);
    walk_dflash_frontier_ = lane.dflash_frontier;
    walk_.host_slot = host_lanes_[0] = static_cast<std::int32_t>(index);
    if (walk_host_.size() < n * walk_.io_stride) {
        walk_host_     = PinnedHostBuffer(n * walk_.io_stride);
        walk_prefetch_ = WalkPrefetch{};
    }
    walk_.active = true;

    auto* io        = static_cast<std::byte*>(io_device_.p);
    auto* area      = walk_area();
    auto* host_area = static_cast<std::byte*>(walk_host_.data());
    // A tensor (or word pointer) into the io image, moved to chunk c's copy.
    const auto rebase = [&](const void* p, std::byte* slot) -> std::byte* {
        const auto* b = static_cast<const std::byte*>(p);
        return b >= io && b < io + io_layout_.bytes ? slot + (b - io) : const_cast<std::byte*>(b);
    };
    const auto rebase_tensor = [&](Tensor t, std::byte* slot) {
        if (t.data != nullptr) { t.data = rebase(t.data, slot); }
        return t;
    };
    if (cpu_service_) { cpu_service_->wake(); }
    residency_->before_round(s, /*landing=*/false, /*walking=*/false, kStreamReadSlots);
    expert_stream_->begin(ring_span(), residency_->host_table(), s, /*gated=*/false, /*ssd=*/true);
    // INFERNIX_TMP_NGRAM_FIRST_SPAN_ASYNC (candidates §8.3 GX-08): when the previous span prefetched
    // nothing for this one (the prompt's first span), chunk 0 reads its rows here, then a worker reads
    // chunks 1..n-1's into their host slots while this thread stages and launches them. Only the worker
    // touches the volume until it ends; chunk c waits for its rows before its upload.
    const bool async_rows = ngram_first_span_async_ && n > 1 &&
                            !walk_prefetched(index, walk_.first_call, 0, walk_.begins[0], walk_.widths[0]);
    std::atomic<std::size_t> rows_ready{1};
    std::exception_ptr rows_error;
    std::thread rows_worker;
    NgramVolume::Counters rows_before{};
    const auto join_rows = [&] {
        if (!rows_worker.joinable()) { return; }
        rows_worker.join();
        const NgramVolume::Counters& after = volume_.counters();
        lane.ngram.rows += after.rows - rows_before.rows;
        lane.ngram.hits += after.hits - rows_before.hits;
        lane.ngram.reads += after.reads - rows_before.reads;
        lane.ngram.read_ns += after.read_ns - rows_before.read_ns;
        if (rows_error) { std::rethrow_exception(rows_error); }
    };
    struct JoinOnExit {
        std::thread& worker;
        ~JoinOnExit() {
            if (worker.joinable()) { worker.join(); }
        }
    } join_on_exit{rows_worker};
    for (std::size_t c = 0; c < n; ++c) {
        const std::int32_t b = walk_.begins[c], w = walk_.widths[c];
        // Rows the previous span prefetched (or this span's worker read) already stand at this
        // chunk's slot.
        const bool from_worker = async_rows && c > 0;
        if (from_worker) {
            for (std::size_t ready = rows_ready.load(std::memory_order_acquire); ready <= c;
                 ready             = rows_ready.load(std::memory_order_acquire)) {
                rows_ready.wait(ready, std::memory_order_acquire);
            }
            if (rows_error) { join_rows(); }
        }
        const bool prefetched = from_worker || walk_prefetched(index, walk_.first_call, c, b, w);
        {
            const nvtx::ScopedRange rows_range(nvtx::Name::PrefillPleRows, nvtx::Category::Prefill,
                                               static_cast<std::uint64_t>(w));
            stage_sequence(index, b, w, lane.history, !prefetched);
        }
        if (async_rows && c == 0) {
            rows_before = volume_.counters();
            rows_worker = std::thread([&, host_area] {
                try {
                    const std::int32_t context = static_cast<std::int32_t>(c_.ple.ngram.ngram_size) - 1;
                    const std::size_t heads    = hash_.heads();
                    const std::size_t row_bytes = ngram_table().row_bytes;
                    std::vector<std::int32_t> window;
                    std::vector<std::uint32_t> ids;
                    for (std::size_t k = 1; k < n; ++k) {
                        const std::int32_t from = walk_.begins[k], count = walk_.widths[k];
                        window.clear();
                        for (std::int32_t p = from - context; p < from + count; ++p) {
                            window.push_back(p < 0 ? static_cast<std::int32_t>(c_.eos_token_id)
                                                   : lane.history[static_cast<std::size_t>(p)]);
                        }
                        ids.resize(static_cast<std::size_t>(count) * heads);
                        hash_.row_ids(window, static_cast<std::size_t>(count), ids.data());
                        std::byte* rows = host_area + k * walk_.io_stride + io_layout_.ngram;
                        volume_.read_rows(ids, std::span<std::byte>(rows, static_cast<std::size_t>(count) * heads * row_bytes));
                        rows_ready.store(k + 1, std::memory_order_release);
                        rows_ready.notify_all();
                    }
                } catch (...) {
                    rows_error = std::current_exception();
                    rows_ready.store(n, std::memory_order_release);
                    rows_ready.notify_all();
                }
            });
        }
        std::optional<execution::MtpChunk> chunk = stage_mtp_chunk(lane, index, b, w);
        const execution::VisionInput* vision     = stage_vision(lane, static_cast<std::uint32_t>(b), w);
        std::byte* host_slot = host_area + c * walk_.io_stride;
        std::byte* slot      = area + walk_.residual_bytes + c * walk_.io_stride;
        if (prefetched) {
            const std::size_t rows = static_cast<std::size_t>(w) * hash_.heads() * ngram_table().row_bytes;
            const std::size_t tail = io_layout_.ngram + rows;
            std::memcpy(host_slot, io_host_.data(), io_layout_.ngram);
            std::memcpy(host_slot + tail, static_cast<const std::byte*>(io_host_.data()) + tail, io_layout_.bytes - tail);
            ++walk_prefetched_chunks_;
        } else {
            std::memcpy(host_slot, io_host_.data(), io_layout_.bytes);
        }
        upload_pinned(slot, host_slot, io_layout_.bytes, s); // not behind the ring's copies (W7)
        if (chunk) {
            chunk->ids              = rebase_tensor(chunk->ids, slot);
            chunk->positions        = rebase_tensor(chunk->positions, slot);
            chunk->rope_positions   = reinterpret_cast<std::int32_t*>(rebase(chunk->rope_positions, slot));
            chunk->block_start_rope = reinterpret_cast<std::int32_t*>(rebase(chunk->block_start_rope, slot));
            if (chunk->vision != nullptr) {
                const std::int32_t cells     = (chunk->prepend ? 1 : 0) + chunk->columns;
                const std::int32_t subchunks = (cells + execution::kMtpChunkColumns - 1) / execution::kMtpChunkColumns;
                auto& copies                 = walk_.mtp_vision[c];
                copies.assign(chunk->vision, chunk->vision + subchunks);
                for (auto& input : copies) { input.columns = rebase_tensor(input.columns, slot); }
                chunk->vision = copies.data();
            }
            walk_.mtp[c] = *chunk;
        }
        if (vision != nullptr) {
            walk_.vision[c] = execution::VisionInput{vision->embeddings, rebase_tensor(vision->columns, slot)};
        }
        // The chunk's embedding and first layer run while the host stages the next chunk (its MoE
        // once its group's last chunk is staged).
        const WideWork wide(*this, widest);
        walk_mixer(c, 0);
        for (std::size_t g = 0; g < walk_.groups.size(); ++g) {
            if (walk_.groups[g].second == c + 1) { walk_moe(g, 0); }
        }
    }
    join_rows();
    walk_.next_layer = 1;
    walk_prefetch_   = WalkPrefetch{}; // consumed; the slots now hold this span's chunks
    ++walk_spans_;
    walk_calls_ += n;
}

std::uint64_t ProgramImpl::window_key() const noexcept {
    std::uint64_t h = 1469598103934665603ULL; // FNV-1a over the window's token ids
    for (const std::int32_t t : window_) {
        h ^= static_cast<std::uint32_t>(t);
        h *= 1099511628211ULL;
    }
    return h;
}

bool ProgramImpl::walk_prefetched(std::uint32_t index, std::size_t first_call, std::size_t c, std::int32_t begin,
                                  std::int32_t width) {
    const WalkPrefetch& f = walk_prefetch_;
    if (!f.valid || f.lane != index || f.first_call != first_call || c >= f.widths.size() || f.begins[c] != begin ||
        f.widths[c] != width) {
        return false;
    }
    ngram_window(lanes_[index].history, begin, width);
    return window_key() == f.keys[c];
}

// Reads up to `chunks` more chunks of the span after the current one (calls from walk_.end_call, as
// walk_span_end would group them) into walk_host_'s slots. Only after the span's first device
// synchronization: until then its own slots may still be uploading.
void ProgramImpl::walk_prefetch_rows(Lane& lane, std::uint32_t index, std::size_t chunks) {
    WalkPrefetch& f = walk_prefetch_;
    if (!f.valid || f.lane != index || f.first_call != walk_.end_call) {
        f            = WalkPrefetch{};
        f.valid      = true;
        f.lane       = index;
        f.first_call = walk_.end_call;
    }
    const std::size_t slots = walk_host_.size() / walk_.io_stride;
    std::uint32_t tokens    = 0;
    for (const auto w : f.widths) { tokens += static_cast<std::uint32_t>(w); }
    for (std::size_t added = 0; added < chunks && !f.done; ++added) {
        const std::size_t k = f.widths.size(), call = f.first_call + k;
        if (k >= slots || call >= lane.calls.size()) {
            f.done = true;
            break;
        }
        const std::uint32_t from  = lane.calls[call - 1];
        const std::uint32_t width = lane.calls[call] - from;
        if (width < static_cast<std::uint32_t>(kStreamMinColumns) || tokens + width > kWalkMaxTokens) {
            f.done = true;
            break;
        }
        const nvtx::ScopedRange rows_range(nvtx::Name::PrefillPleRows, nvtx::Category::Prefill,
                                           static_cast<std::uint64_t>(width));
        ngram_window(lane.history, static_cast<std::int32_t>(from), static_cast<std::int32_t>(width));
        auto* slot = static_cast<std::byte*>(walk_host_.data()) + k * walk_.io_stride;
        read_ngram_to(lane, static_cast<std::int32_t>(width), slot + io_layout_.ngram);
        f.begins.push_back(static_cast<std::int32_t>(from));
        f.widths.push_back(static_cast<std::int32_t>(width));
        f.keys.push_back(window_key());
        tokens += width;
    }
}

// Chunk c's call as the Forward reads it.
execution::ForwardBatch ProgramImpl::walk_batch(std::size_t c) {
    std::byte* slot      = walk_area() + walk_.residual_bytes + c * walk_.io_stride;
    const std::int32_t w = walk_.widths[c];
    execution::ForwardBatch fb = io_batch(slot, 1, w, 1);
    fb.host_slots      = std::span<const std::int32_t>(&walk_.host_slot, 1);
    fb.host_table_rows = fb.host_slots;
    fb.mtp_chunk       = walk_.mtp[c] ? &*walk_.mtp[c] : nullptr;
    fb.vision          = walk_.vision[c] ? &*walk_.vision[c] : nullptr;
    // Chunk 0 waits per layer for the lane's restore and in-flight capture. Resolved here, not at the
    // walk's start: the walk spans engine steps, and a poll between them retires a landed batch and
    // frees its events (a landed batch needs no wait). A later chunk waits per layer for the tap
    // captured at the previous chunk's end, whose copy of that layer is enqueued by then.
    if (c == 0) {
        fb.layer_waits = prefix_waits(lanes_[walk_.lane]);
    } else if (walk_.taps[c - 1].staged != 0) {
        fb.layer_waits[1] = prefix_->staged_events(walk_.taps[c - 1].staged);
    }
    fb.stream         = true;
    fb.release_stream = c + 1 == walk_.widths.size();
    if (scoring_) { fb.residual_out = score_residuals(walk_.begins[c], w); } // a scored prompt's (causal_score)
    return fb;
}

// The residual columns of chunks [first, end) (consecutive in the area).
Tensor ProgramImpl::walk_residual(std::size_t first, std::size_t end) {
    const std::int32_t columns = walk_.offsets[end - 1] + walk_.widths[end - 1] - walk_.offsets[first];
    return Tensor(walk_area() + 2ULL * static_cast<std::size_t>(width_) * static_cast<std::size_t>(walk_.offsets[first]),
                  DType::BF16, {width_, columns});
}

// Layer l of chunk c up to its MoE (with the embedding before layer 0): the MoE input and inject
// weights go to its columns of the group's area.
void ProgramImpl::walk_mixer(std::size_t c, std::uint32_t l) {
    std::size_t g = 0;
    while (walk_.groups[g].second <= c) { ++g; }
    const auto first        = static_cast<std::size_t>(walk_.offsets[c] - walk_.offsets[walk_.groups[g].first]);
    const std::int32_t w    = walk_.widths[c];
    const std::int32_t H    = static_cast<std::int32_t>(c_.hidden_size), S = static_cast<std::int32_t>(c_.hc.streams);
    std::byte* moe          = walk_area() + walk_.moe_offset;
    std::byte* inject_base  = moe + walk_align(static_cast<std::size_t>(kWalkMoeColumns) * 2ULL * H);
    Tensor xm(moe + first * 2ULL * H, DType::BF16, {H, w});
    Tensor inject(inject_base + first * 4ULL * S, DType::FP32, {S, w});
    execution::ForwardBatch fb = walk_batch(c);
    Tensor residual            = walk_residual(c, c + 1);
    forward_->begin_call(fb, nullptr);
    if (l == 0) { forward_->embed(fb, residual); }
    forward_->layer_mixer(l, fb, residual, xm, inject);
    walk_capture(c, l); // the layer's state (the MoE writes none) at the chunk's end
}

// Layer l's MoE over group g's chunks, then (the last layer) each chunk's MTP cells and, for the
// prompt's last chunk, the logits.
void ProgramImpl::walk_moe(std::size_t g, std::uint32_t l) {
    const auto [first, end]    = walk_.groups[g];
    const std::uint32_t L      = c_.num_hidden_layers;
    const std::int32_t H       = static_cast<std::int32_t>(c_.hidden_size), S = static_cast<std::int32_t>(c_.hc.streams);
    Tensor residual            = walk_residual(first, end);
    const std::int32_t columns = residual.ne[1];
    std::byte* moe             = walk_area() + walk_.moe_offset;
    Tensor xm(moe, DType::BF16, {H, columns});
    Tensor inject(moe + walk_align(static_cast<std::size_t>(kWalkMoeColumns) * 2ULL * H), DType::FP32, {S, columns});
    // The group's last chunk's call opens it (its release_stream ends the span's last layer copies);
    // the route log keeps that chunk's columns, the span's last as walk_step downloads them.
    const execution::ForwardBatch fb = walk_batch(end - 1);
    forward_->begin_call(fb, nullptr);
    forward_->layer_moe(l, fb, xm, inject, residual, walk_.widths[end - 1]);
    if (dflash2_) { dflash2_walk_features(first, end, l, residual); }
    if (l + 1 < L) { return; }
    const bool prompt_end = walk_.end_call == lanes_[walk_.lane].calls.size();
    for (std::size_t c = first; c < end; ++c) {
        const bool logits_here = prompt_end && c + 1 == walk_.widths.size();
        Tensor logits                 = head_logits(1);
        execution::ForwardBatch chunk = walk_batch(c);
        Tensor chunk_residual         = walk_residual(c, c + 1);
        forward_->begin_call(chunk, logits_here ? &logits : nullptr);
        forward_->finish(chunk, chunk_residual, logits_here ? &logits : nullptr);
        if (logits_here) { round_logits(1, device_.stream); }
        walk_capture(c, L); // the chunk's end in the MTP block
    }
}

// A prefix tap due at chunk c's end (walk_plan_taps): copies its state of layer `layer` (L: the MTP
// block) once the walk has enqueued that layer of the chunk, before the next chunk overwrites it.
void ProgramImpl::walk_capture(std::size_t c, std::uint32_t layer) {
    const LayerWalk::Tap& tap = walk_.taps[c];
    if (tap.staged == 0) { return; }
    const Lane& lane = lanes_[walk_.lane];
    std::optional<DeviceKVPageHandle> tail_page;
    if (tap.frontier % kBlock != 0) { tail_page = lane.pages.at(tap.frontier / kBlock - lane.prefix.page_base).handle(); }
    prefix_->capture_group(tap.staged, layer, walk_.lane, tail_page, device_.stream);
}

void ProgramImpl::walk_plan_taps(Lane& lane) {
    const std::size_t n = walk_.widths.size();
    walk_.taps.assign(n, {});
    if (!prefix_ || !lane.prefix.reuse) { return; }
    // As prefix_after_prefill_call after each inner call of a chunk-major prefill: the taps due at the
    // chunk's end are realized there, and captured when the lineage's deepest snapshot is a block back.
    std::uint32_t deepest = lane.prefix.deepest;
    for (std::size_t c = 0; c + 1 < n; ++c) {
        const std::uint32_t B = lane.calls[walk_.first_call + c];
        bool boundary         = false;
        if (!prefix_take_taps(lane, B, walk_.first_call + c + 1, boundary) || B < deepest + kBlock) { continue; }
        const bool tail   = B % kBlock != 0;
        const auto staged = prefix_->open_staged_capture(tail, prefix_->index().estimate_priority(deepest, B, tail));
        if (!staged) { continue; }
        walk_.taps[c] = {B, boundary ? pc::SnapshotKind::Boundary : pc::SnapshotKind::Tap, *staged};
        deepest       = B;
    }
}

void ProgramImpl::walk_commit_taps(Lane& lane) {
    for (LayerWalk::Tap& tap : walk_.taps) {
        if (tap.staged == 0) { continue; }
        // The blocks up to the tap first, as after its chunk-major call (its anchor is the last).
        prefix_publish(lane, tap.frontier);
        const std::uint64_t staged = std::exchange(tap.staged, 0);
        if (lane.prefix.path.size() < tap.frontier / kBlock) {
            prefix_->abandon_staged_capture(staged);
            continue;
        }
        // Every cell up to the span's end is written (the prompt continues past each inner chunk).
        prefix::PrefixCache::Capture capture = prefix_capture_at(lane, tap.frontier, tap.kind, false, lane.mtp_live);
        // Its DFlash2 ring was copied during the span, before the span's context reached the ring.
        capture.meta.draft_frontier = std::min(capture.meta.draft_frontier, walk_dflash_frontier_);
        const std::uint64_t id               = capture.id;
        prefix_->commit_staged_capture(staged, std::move(capture));
        prefix_captured(lane, id, tap.frontier);
    }
}

void ProgramImpl::walk_enqueue(std::uint32_t to_layer) {
    require_tier_round();
    for (std::uint32_t l = walk_.next_layer; l < to_layer; ++l) {
        // Host enqueue of one layer over the span's chunks (payload: the layer).
        const nvtx::ScopedRange layer_range(nvtx::Name::TextLayer, nvtx::Category::Prefill, l);
        for (std::size_t g = 0; g < walk_.groups.size(); ++g) {
            for (std::size_t c = walk_.groups[g].first; c < walk_.groups[g].second; ++c) { walk_mixer(c, l); }
            walk_moe(g, l);
        }
    }
    walk_.next_layer = std::max(walk_.next_layer, to_layer);
}

std::uint32_t ProgramImpl::walk_step_tokens_from_environment() {
    return static_cast<std::uint32_t>(
        temporary_switches::integer("INFERNIX_TMP_DECODE_SHARE_WALK_TOKENS", 0, 0, kWalkMaxTokens));
}

// The layers one walk step enqueues: about one call's work (walk_begin), or with the walk-step switch
// while another lane decodes, about walk_step_tokens_ tokens' (at least one layer).
std::uint32_t ProgramImpl::walk_step_layers(std::uint32_t index) const noexcept {
    if (walk_step_tokens_ == 0) { return walk_.layers_per_step; }
    bool decoding = false;
    for (std::uint32_t i = 0; i < options_.max_concurrency; ++i) {
        if (i != index && lanes_[i].phase == Phase::Decode) { decoding = true; }
    }
    if (!decoding) { return walk_.layers_per_step; }
    std::uint64_t tokens = 0;
    for (const auto w : walk_.widths) { tokens += static_cast<std::uint64_t>(w); }
    const std::uint64_t layers = tokens == 0 ? 1 : static_cast<std::uint64_t>(c_.num_hidden_layers) * walk_step_tokens_ / tokens;
    return static_cast<std::uint32_t>(std::clamp<std::uint64_t>(layers, 1, walk_.layers_per_step));
}

PrefillProgress ProgramImpl::walk_step(Lane& lane, std::uint32_t index, Clock::time_point start) {
    const std::uint32_t L = c_.num_hidden_layers;
    std::uint32_t tokens  = 0;
    for (const auto w : walk_.widths) { tokens += static_cast<std::uint32_t>(w); }
    // The first step already ran layer 0 while staging.
    const std::uint32_t to =
        std::min(L, walk_.next_layer + walk_step_layers(index) - (walk_.steps == 0 ? 1U : 0U));
    if (walk_.steps > 0) {
        // A step boundary, before the step's calls: a decode or verification round of another lane
        // since the last step closed the tier's round when it settled, and the tier answers fetch
        // requests only while one is open (walk_begin's before_round opened the first step's).
        device_.synchronize(); // the boundary needs an idle device: that round's commit may still run
        residency_->tier_step(device_.stream);
        if (cpu_service_) { cpu_service_->wake(); }
    }
    {
        const nvtx::ScopedRange step_range(nvtx::Name::PrefillChunk, nvtx::Category::Prefill,
                                           static_cast<std::uint64_t>(tokens));
        const WideWork wide(*this, *std::max_element(walk_.widths.begin(), walk_.widths.end()));
        walk_enqueue(to);
    }
    ++walk_.steps;
    if (to < L) {
        // While the device runs this step, read the next span's n-gram rows (one span's chunks spread
        // over the remaining steps; ~35-70 ms of NVMe reads per 4,096-token chunk).
        if (walk_.steps > 1 && walk_.end_call < lane.calls.size()) {
            const std::size_t remaining = (L - to + walk_.layers_per_step - 1) / walk_.layers_per_step + 1;
            const std::size_t slots     = walk_host_.size() / walk_.io_stride;
            const std::size_t have      = walk_prefetch_.valid ? walk_prefetch_.widths.size() : 0;
            walk_prefetch_rows(lane, index, slots > have ? (slots - have + remaining - 1) / remaining : 0);
        }
        {
            const nvtx::ScopedRange wait_range(nvtx::Name::DeviceWait, nvtx::Category::Prefill);
            device_.synchronize();
        }
        check_expert_error();
        PrefillProgress out;
        out.summary                 = prefill_summary(lane);
        out.processed_prompt_tokens = 0;
        out.complete                = false;
        out.timing.submit_host_ns   = elapsed_ns(start);
        lane.prefill_ns += out.timing.submit_host_ns;
        return out;
    }
    // The span is enqueued: the lane's state is at its end in every layer.
    const std::int32_t begin = walk_.begins.back(), width = walk_.widths.back();
    const bool last          = walk_.end_call == lane.calls.size();
    lane.state_tokens = lane.calls[walk_.end_call - 1];
    lane.next_call    = walk_.end_call;
    residency_->enqueue_route_download(device_.stream, width);
    expert_stream_->end();
    if (dflash2_) { dflash2_walk_context(lane, index); }
    walk_.active   = false;
    host_lanes_[0] = static_cast<std::int32_t>(index);
    if (last && !prefilling_besides(index)) {
        return_stream_lease();
        return_wide_work();
    }
    walk_commit_taps(lane);
    prefix_after_prefill_call(lane, index, last);
    if (last && lane.vision) { vision_release(lane); }
    if (!last) {
        {
            const nvtx::ScopedRange wait_range(nvtx::Name::DeviceWait, nvtx::Category::Prefill);
            device_.synchronize();
        }
        check_expert_error();
        const nvtx::ScopedRange residency_range(nvtx::Name::PrefillResidency, nvtx::Category::Moe,
                                                static_cast<std::uint64_t>(width));
        // The route log holds the span's last chunk (every chunk routes nearly every expert).
        trace_round(RouteTraceKind::PrefillChunk, 1, width, tokens, 0, static_cast<std::uint32_t>(begin));
        apply_prompt_routes(lane, width, 0);
        apply_vram_target(false);
    }
    // A walk promotes no experts (its routes still count toward the cache's scores). Between spans
    // the stream lease holds most frames, so promotions would churn through the rest (~12,000
    // records, ~1.2 s of link per 64K span) and the next span would wait for them. After the
    // prompt they would fill the returned frames with the prompt's experts while the answer's own
    // staged misses land there anyway: 16 per layer and chunk measured slower than none for every
    // answer length (128K + 512 tokens: 23.16 -> 21.10 s; design §19.3.17).
    return prefill_progress(lane, index, begin, width, tokens, last, 0, start);
}

void ProgramImpl::walk_abandon() noexcept {
    if (!walk_.active) { return; }
    try {
        device_.synchronize();
    } catch (...) {}
    expert_stream_->end();
    for (const LayerWalk::Tap& tap : walk_.taps) {
        if (tap.staged != 0) { prefix_->abandon_staged_capture(tap.staged); }
    }
    walk_          = LayerWalk{};
    walk_prefetch_ = WalkPrefetch{};
}

} // namespace infernix::models::qwen4_exp::detail
