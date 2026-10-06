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
// differs (resident, streamed or staged experts), which the MoE's arithmetic does not see.
//
// The span's residual stream (BF16 [S*H, tokens]) and every chunk's staged io image live in the
// stream lease after the ring. Each advance_prefill call enqueues about one call's work (a few
// layers of every chunk) and synchronizes, so cancellation and other lanes interleave as between
// chunk-major calls; the walk's own state stays valid across them. A span ends where a prefix tap
// is realized (the captured state must be the boundary's in every layer), before a call too narrow
// to stream, and at kWalkMaxTokens.

#include "models/qwen4_exp/program/program_impl.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace ninfer::models::qwen4_exp::detail {

namespace {

constexpr std::size_t kWalkAlign = 256;

std::size_t walk_align(std::size_t bytes) { return (bytes + kWalkAlign - 1) / kWalkAlign * kWalkAlign; }

} // namespace

std::size_t ProgramImpl::walk_bytes(std::uint32_t tokens, std::size_t chunks) const noexcept {
    return walk_align(2ULL * static_cast<std::size_t>(width_) * tokens) + chunks * walk_align(io_layout_.bytes);
}

// The exclusive end of the span that starts at lane.next_call; lane.next_call when the call runs
// chunk-major. Lends the stream lease (with the walk's area) when none is held.
std::size_t ProgramImpl::walk_span_end(const Lane& lane, std::uint32_t index) {
    const std::size_t first = lane.next_call;
    if (!expert_stream_ || residency_->frames() == 0 || walk_.active || prefilling_besides(index)) { return first; }
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
        if (prefix_tap_due(lane, end)) { break; }
    }
    if (end - first < 2) { return first; }
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
    walk_.layers_per_step =
        static_cast<std::uint32_t>((c_.num_hidden_layers + n - 1) / n); // about one call's work per step
    walk_.mtp.assign(n, std::nullopt);
    walk_.vision.assign(n, std::nullopt);
    walk_.mtp_vision.assign(n, {});
    walk_.host_slot = host_lanes_[0] = static_cast<std::int32_t>(index);
    if (walk_host_.size() < n * walk_.io_stride) {
        walk_host_     = PinnedHostBuffer(n * walk_.io_stride);
        walk_prefetch_ = WalkPrefetch{};
    }
    walk_.waits = prefix_waits(lane);
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
    residency_->before_round(s);
    expert_stream_->begin(ring_span(), residency_->host_table(), s);
    for (std::size_t c = 0; c < n; ++c) {
        const std::int32_t b = walk_.begins[c], w = walk_.widths[c];
        // Rows the previous span prefetched already stand at this chunk's slot.
        const bool prefetched = walk_prefetched(index, walk_.first_call, c, b, w);
        {
            const nvtx::ScopedRange rows_range(nvtx::Name::PrefillPleRows, nvtx::Category::Prefill,
                                               static_cast<std::uint64_t>(w));
            stage_sequence(index, b, w, lane.history, !prefetched);
        }
        std::optional<execution::MtpChunk> chunk = stage_mtp_chunk(lane, index, b, w);
        const execution::VisionInput* vision     = stage_vision(lane, static_cast<std::uint32_t>(b), w);
        std::byte* host_slot = host_area + c * walk_.io_stride;
        std::byte* slot      = area + walk_.residual_bytes + c * walk_.io_stride;
        if (prefetched) {
            const std::size_t rows = static_cast<std::size_t>(w) * hash_.heads() * c_.ple.table.row_bytes;
            const std::size_t tail = io_layout_.ngram + rows;
            std::memcpy(host_slot, io_host_.data(), io_layout_.ngram);
            std::memcpy(host_slot + tail, static_cast<const std::byte*>(io_host_.data()) + tail, io_layout_.bytes - tail);
            ++walk_prefetched_chunks_;
        } else {
            std::memcpy(host_slot, io_host_.data(), io_layout_.bytes);
        }
        CUDA_CHECK(cudaMemcpyAsync(slot, host_slot, io_layout_.bytes, cudaMemcpyHostToDevice, s));
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
        // The chunk's embedding and first layer run while the host stages the next chunk.
        const WideWork wide(*this, w);
        walk_pass(c, 0);
    }
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
        if (prefix_tap_due(lane, call + 1)) { f.done = true; }
    }
}

// Layer l of chunk c (with the embedding before layer 0, and the MTP cells and, for the prompt's
// last chunk, the logits after the last layer).
void ProgramImpl::walk_pass(std::size_t c, std::uint32_t l) {
    const std::size_t n   = walk_.widths.size();
    const std::uint32_t L = c_.num_hidden_layers;
    const bool prompt_end = walk_.end_call == lanes_[walk_.lane].calls.size();
    auto* area            = walk_area();
    std::byte* slot       = area + walk_.residual_bytes + c * walk_.io_stride;
    const std::int32_t w  = walk_.widths[c];
    execution::ForwardBatch fb = io_batch(slot, 1, w, 1);
    fb.host_slots      = std::span<const std::int32_t>(&walk_.host_slot, 1);
    fb.host_table_rows = fb.host_slots;
    fb.mtp_chunk       = walk_.mtp[c] ? &*walk_.mtp[c] : nullptr;
    fb.vision          = walk_.vision[c] ? &*walk_.vision[c] : nullptr;
    if (c == 0) { fb.layer_waits = walk_.waits; }
    fb.stream         = true;
    fb.release_stream = c + 1 == n;
    Tensor residual(area + 2ULL * static_cast<std::size_t>(width_) * static_cast<std::size_t>(walk_.offsets[c]),
                    DType::BF16, {width_, w});
    const bool logits_here = l + 1 == L && prompt_end && c + 1 == n;
    Tensor logits(logits32_.p, DType::FP32, {vocab_, 1});
    forward_->begin_call(fb, logits_here ? &logits : nullptr);
    if (l == 0) { forward_->embed(fb, residual); }
    forward_->layer(l, fb, residual);
    if (l + 1 == L) { forward_->finish(fb, residual, logits_here ? &logits : nullptr); }
}

void ProgramImpl::walk_enqueue(std::uint32_t to_layer) {
    for (std::uint32_t l = walk_.next_layer; l < to_layer; ++l) {
        for (std::size_t c = 0; c < walk_.widths.size(); ++c) { walk_pass(c, l); }
    }
    walk_.next_layer = std::max(walk_.next_layer, to_layer);
}

PrefillProgress ProgramImpl::walk_step(Lane& lane, std::uint32_t index, Clock::time_point start) {
    const std::uint32_t L = c_.num_hidden_layers;
    const std::size_t n   = walk_.widths.size();
    std::uint32_t tokens  = 0;
    for (const auto w : walk_.widths) { tokens += static_cast<std::uint32_t>(w); }
    // The first step already ran layer 0 while staging.
    const std::uint32_t to =
        std::min(L, walk_.next_layer + walk_.layers_per_step - (walk_.steps == 0 ? 1U : 0U));
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
        residency_->tier_step(device_.stream);
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
    const std::size_t promotions = kPrefillPromotionsPerLayer * n;
    lane.state_tokens = lane.calls[walk_.end_call - 1];
    lane.next_call    = walk_.end_call;
    residency_->enqueue_route_download(device_.stream, width);
    expert_stream_->end();
    walk_.active   = false;
    host_lanes_[0] = static_cast<std::int32_t>(index);
    if (last && !prefilling_besides(index)) {
        return_stream_lease();
        return_wide_work();
    }
    prefix_after_prefill_call(lane, index, last);
    if (last && lane.vision) { vision_release(lane); }
    if (!last) {
        {
            const nvtx::ScopedRange wait_range(nvtx::Name::DeviceWait, nvtx::Category::Prefill);
            device_.synchronize();
        }
        const nvtx::ScopedRange residency_range(nvtx::Name::PrefillResidency, nvtx::Category::Moe,
                                                static_cast<std::uint64_t>(width));
        // The route log holds the span's last chunk (every chunk routes nearly every expert). No
        // promotions: the stream lease still holds most frames, so they would churn through the
        // rest (~12,000 records, ~1.2 s of link per 64K span), and the next span would wait for
        // them; the prompt's last span promotes.
        trace_round(RouteTraceKind::PrefillChunk, 1, width, tokens, 0, static_cast<std::uint32_t>(begin));
        residency_->after_round(device_.stream, width, 0);
        apply_vram_target(false);
    }
    return prefill_progress(lane, index, begin, width, tokens, last, promotions, start);
}

void ProgramImpl::walk_abandon() noexcept {
    if (!walk_.active) { return; }
    try {
        device_.synchronize();
    } catch (...) {}
    expert_stream_->end();
    walk_          = LayerWalk{};
    walk_prefetch_ = WalkPrefetch{};
}

} // namespace ninfer::models::qwen4_exp::detail
