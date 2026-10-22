#include "runtime/expert_offload/expert_stream.h"

#include "core/copy_batch.h"
#include "core/device.h"

#include <algorithm>
#include <stdexcept>

namespace infernix::runtime::expert_offload {
namespace {

constexpr std::size_t kAlign = 256;

std::size_t align_up(std::size_t bytes) { return (bytes + kAlign - 1) / kAlign * kAlign; }

} // namespace

ExpertStream::ExpertStream(std::uint32_t layers, std::uint32_t experts, std::uint64_t record_stride, RecordOf record_of,
                           bool one_allocation, Hold hold, std::vector<std::uint64_t> host_strides, SsdReads ssd)
    : experts_(experts), layers_(layers), stride_(record_stride), host_strides_(std::move(host_strides)),
      record_of_(std::move(record_of)), hold_(std::move(hold)), one_allocation_(one_allocation), ssd_(std::move(ssd)) {
    if (experts_ == 0 || layers_ == 0 || stride_ == 0 || stride_ % 16 != 0 || !record_of_) {
        throw std::invalid_argument("expert stream: empty geometry or unaligned records");
    }
    if (ssd_.slots && (!ssd_.read || !ssd_.wait || !ssd_.release)) {
        throw std::invalid_argument("expert stream: stream slots without their reads");
    }
    ssd_planned_.resize(layers_);
    ssd_slots_.resize(layers_);
    ssd_requested_.assign(layers_, 0);
    if (!host_strides_.empty() &&
        (host_strides_.size() != layers_ ||
         std::any_of(host_strides_.begin(), host_strides_.end(), [&](std::uint64_t s) { return s == 0 || s > stride_ || s % 16 != 0; }))) {
        throw std::invalid_argument("expert stream: a bank's record stride exceeds the ring's slots");
    }
    tables_host_ = PinnedHostBuffer(static_cast<std::size_t>(layers_) * experts_ * sizeof(std::int32_t));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&start_, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&uploaded_, cudaEventDisableTiming));
    landed_.resize(layers_);
    consumed_.resize(layers_);
    for (std::uint32_t l = 0; l < layers_; ++l) {
        CUDA_CHECK(cudaEventCreateWithFlags(&landed_[l], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&consumed_[l], cudaEventDisableTiming));
    }
    streams_.assign(layers_, 0);
    complete_.assign(layers_, 0);
}

ExpertStream::~ExpertStream() {
    if (stream_ != nullptr) { (void)cudaStreamSynchronize(stream_); }
    for (auto event : bounced_) {
        if (event != nullptr) { (void)cudaEventDestroy(event); }
    }
    for (auto event : landed_) { (void)cudaEventDestroy(event); }
    for (auto event : consumed_) { (void)cudaEventDestroy(event); }
    if (uploaded_ != nullptr) { (void)cudaEventDestroy(uploaded_); }
    if (start_ != nullptr) { (void)cudaEventDestroy(start_); }
    if (stream_ != nullptr) { (void)cudaStreamDestroy(stream_); }
}

void ExpertStream::set_staging(Staging staging) {
    if (active_) { throw std::logic_error("expert stream: staging changed during a chunk"); }
    if (staging.per_part > 0 && (!staging.unpinned || !staging.slot || !staging.copy)) {
        throw std::invalid_argument("expert stream: incomplete staging");
    }
    staging_ = std::move(staging);
    for (auto& event : bounced_) {
        if (event == nullptr && staging_.per_part > 0) { CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming)); }
    }
}

bool ExpertStream::takes(std::uint32_t layer, std::uint32_t e, std::uint32_t taken, std::uint32_t& staged) const {
    if (taken >= half_) { return false; }
    if (record_of_(layer, e) != nullptr) { return true; }
    if (staged < staging_.per_part && staging_.unpinned && staging_.unpinned(layer, e) != nullptr) {
        ++staged;
        return true;
    }
    return false;
}

std::uint32_t ExpertStream::slots_in(std::size_t bytes) const noexcept {
    const std::size_t tables = align_up(static_cast<std::size_t>(layers_) * experts_ * sizeof(std::int32_t));
    if (bytes <= tables) { return 0; }
    return static_cast<std::uint32_t>((bytes - tables) / stride_) & ~1U;
}

void ExpertStream::begin(DeviceSpan ring, const std::int32_t* residency, cudaStream_t compute, bool gated, bool ssd) {
    if (active_) { throw std::logic_error("expert stream: a chunk is already streaming"); }
    const std::uint32_t slots = slots_in(ring.bytes);
    if (slots < 2 || ring.data == nullptr) { throw std::invalid_argument("expert stream: the ring holds no slots"); }
    half_    = slots / 2;
    tables_  = static_cast<std::int32_t*>(ring.data);
    records_ = static_cast<std::uint8_t*>(ring.data) +
               align_up(static_cast<std::size_t>(layers_) * experts_ * sizeof(std::int32_t));
    // The previous chunk's table upload may still be pending: its pinned source is rewritten now.
    if (uploading_) { CUDA_CHECK(cudaEventSynchronize(uploaded_)); }
    // A gated chunk's rows upload on the copy stream: the previous chunk's must have left them.
    if (gated_) { CUDA_CHECK(cudaStreamSynchronize(stream_)); }
    gated_     = gated;
    residency_ = residency;
    for (std::uint32_t l = 0; l < layers_; ++l) { // a gated chunk streams no SSD-only expert
        ssd_planned_[l].clear();
        ssd_slots_[l].clear();
        ssd_requested_[l] = 0;
    }
    if (gated) {
        uploading_ = false;
        std::fill(streams_.begin(), streams_.end(), std::uint8_t{0});
        std::fill(complete_.begin(), complete_.end(), std::uint8_t{0});
        CUDA_CHECK(cudaEventRecord(start_, compute));
        CUDA_CHECK(cudaStreamWaitEvent(stream_, start_, 0));
        active_ = true;
        return;
    }
    auto* tables = static_cast<std::int32_t*>(tables_host_.data());
    // Stream slots are held from a layer's read request to its copy's landing: about kReadAhead + 2
    // layers' worth at once, so each layer takes at most that share.
    const std::uint32_t ssd_cap = ssd && ssd_.slots ? ssd_.slots() / (kReadAhead + 2) : 0;
    for (std::uint32_t l = 0; l < layers_; ++l) {
        std::int32_t* row = tables + static_cast<std::size_t>(l) * experts_;
        const std::int32_t* frames = residency + static_cast<std::size_t>(l) * experts_;
        std::uint32_t taken = 0, staged = 0;
        bool complete = true;
        for (std::uint32_t e = 0; e < experts_; ++e) {
            row[e] = -1;
            if (frames[e] >= 0) { continue; }
            if (taken >= half_) {
                complete = false;
                continue;
            }
            if (takes(l, e, taken, staged)) { // a pinned record, or an unpinned one while staging has room
                row[e] = static_cast<std::int32_t>((l % 2) * half_ + taken++);
                if (hold_) { hold_(l, e); }
            } else if (record_of_(l, e) == nullptr && (!staging_.unpinned || staging_.unpinned(l, e) == nullptr) &&
                       ssd_planned_[l].size() < ssd_cap) { // SSD-only: a stream read
                row[e] = static_cast<std::int32_t>((l % 2) * half_ + taken++);
                ssd_planned_[l].emplace_back(e, row[e]);
            } else {
                complete = false;
            }
        }
        ssd_slots_[l].assign(ssd_planned_[l].size(), -1);
        streams_[l]  = taken > 0 ? 1 : 0;
        complete_[l] = complete ? 1 : 0;
    }
    const std::size_t table_bytes = static_cast<std::size_t>(layers_) * experts_ * sizeof(std::int32_t);
    CUDA_CHECK(cudaMemcpyAsync(tables_, tables_host_.data(), table_bytes, cudaMemcpyHostToDevice, compute));
    CUDA_CHECK(cudaEventRecord(uploaded_, compute));
    uploading_ = true;
    // The ring's previous readers (the last chunk's experts, the lender's last use) come first.
    CUDA_CHECK(cudaEventRecord(start_, compute));
    CUDA_CHECK(cudaStreamWaitEvent(stream_, start_, 0));
    active_ = true;
    enqueue(0);
    if (layers_ > 1) { enqueue(1); }
}

std::uint32_t ExpertStream::plan_layer(std::uint32_t layer, std::span<const std::int32_t> columns,
                                       std::span<const std::uint8_t> cpu) {
    if (!active_ || !gated_ || layer >= layers_ || columns.size() < experts_ || cpu.size() < experts_) {
        throw std::logic_error("expert stream: plan_layer outside a gated chunk");
    }
    auto* row                  = static_cast<std::int32_t*>(tables_host_.data()) + static_cast<std::size_t>(layer) * experts_;
    const std::int32_t* frames = residency_ + static_cast<std::size_t>(layer) * experts_;
    std::uint32_t taken        = 0, staged = 0;
    for (std::uint32_t e = 0; e < experts_; ++e) {
        // An SSD-only expert has no host record to copy: the CPU or the fetch channel serves it.
        const bool copy = columns[e] > 0 && frames[e] < 0 && cpu[e] == 0 && takes(layer, e, taken, staged);
        row[e]          = copy ? static_cast<std::int32_t>((layer % 2) * half_ + taken++) : -1;
        if (copy && hold_) { hold_(layer, e); }
    }
    streams_[layer] = 1; // the MoE waits for the table upload even when nothing is copied
    // The half's previous readers (layer - 2's experts) release it first; the table rides ahead of
    // the copies on the copy stream, so the layer's landed event covers both.
    if (layer >= 2) { CUDA_CHECK(cudaStreamWaitEvent(stream_, consumed_[layer - 2], 0)); }
    CUDA_CHECK(cudaMemcpyAsync(tables_ + static_cast<std::size_t>(layer) * experts_, row, experts_ * sizeof(std::int32_t),
                               cudaMemcpyHostToDevice, stream_));
    enqueue(layer);
    return taken;
}

const std::int32_t* ExpertStream::slots(std::uint32_t layer) const noexcept {
    return tables_ + static_cast<std::size_t>(layer) * experts_;
}

bool ExpertStream::release_landed(bool wait) {
    bool released = false;
    while (!ssd_held_.empty()) {
        const std::uint32_t l = ssd_held_.front();
        if (wait && !released) {
            CUDA_CHECK(cudaEventSynchronize(landed_[l]));
        } else if (cudaEventQuery(landed_[l]) != cudaSuccess) {
            (void)cudaGetLastError(); // cudaErrorNotReady
            break;
        }
        for (std::int32_t& s : ssd_slots_[l]) {
            if (s >= 0) { ssd_.release(s); }
            s = -1;
        }
        ssd_held_.erase(ssd_held_.begin());
        released = true;
    }
    return released;
}

void ExpertStream::request_reads(std::uint32_t layer) {
    if (layer >= layers_ || ssd_requested_[layer] != 0) { return; }
    ssd_requested_[layer] = 1;
    for (std::size_t i = 0; i < ssd_planned_[layer].size(); ++i) {
        std::int32_t s = ssd_.read(layer, ssd_planned_[layer][i].first);
        // Slots come back as earlier layers' copies land (a layer's share always fits once they have).
        while (s < 0) {
            if (!release_landed(true)) { throw std::runtime_error("expert stream: no stream slot comes free"); }
            s = ssd_.read(layer, ssd_planned_[layer][i].first);
        }
        ssd_slots_[layer][i] = s;
    }
}

void ExpertStream::enqueue(std::uint32_t layer) {
    if (layer >= 2) { CUDA_CHECK(cudaStreamWaitEvent(stream_, consumed_[layer - 2], 0)); }
    // The stream reads of this layer and the next kReadAhead (the drive reads while these copy).
    (void)release_landed(false);
    for (std::uint32_t l = layer; l <= layer + kReadAhead && l < layers_; ++l) { request_reads(l); }
    const auto* row = static_cast<const std::int32_t*>(tables_host_.data()) + static_cast<std::size_t>(layer) * experts_;
    // Runs of consecutive experts in consecutive slots and adjacent records of one allocation: one
    // copy per run (a copy may not span two pinned allocations, even adjacent ones), all enqueued as
    // one batch so the host does not wait on the copy queue.
    CopyBatch copies(stream_);
    std::size_t next_ssd = 0; // ssd_planned_[layer] is in ascending expert order, as `row`
    staged_.clear();
    // A bank whose records are narrower than the ring's slots (a GGUF import's layer) copies record by
    // record; equal strides copy runs.
    const std::uint64_t host = host_strides_.empty() ? stride_ : host_strides_[layer];
    for (std::uint32_t e = 0; e < experts_;) {
        if (row[e] < 0) {
            ++e;
            continue;
        }
        if (next_ssd < ssd_planned_[layer].size() && ssd_planned_[layer][next_ssd].first == e) {
            // A stream read: its record once landed (the host waits for the drive here).
            const std::uint8_t* source = ssd_.wait(ssd_slots_[layer][next_ssd]);
            if (source == nullptr) { throw std::runtime_error("expert stream: an SSD-only expert's record could not be read"); }
            const std::size_t bytes = host_strides_.empty() ? stride_ : host_strides_[layer];
            copies.add(records_ + static_cast<std::size_t>(row[e]) * stride_, source, bytes);
            stats_.streamed += 1;
            stats_.ssd += 1;
            stats_.bytes += bytes;
            ++stats_.copies;
            ++next_ssd;
            ++e;
            continue;
        }
        const std::uint8_t* source = record_of_(layer, e);
        if (source == nullptr) { // an unpinned record: bounced below
            staged_.emplace_back(records_ + static_cast<std::size_t>(row[e]) * stride_, staging_.unpinned(layer, e));
            ++e;
            continue;
        }
        std::uint32_t end = e + 1;
        while (host == stride_ && one_allocation_ && end < experts_ && row[end] == row[end - 1] + 1 &&
               record_of_(layer, end) == source + static_cast<std::size_t>(end - e) * stride_) {
            ++end;
        }
        const std::size_t bytes = static_cast<std::size_t>(end - e) * host;
        copies.add(records_ + static_cast<std::size_t>(row[e]) * stride_, source, bytes);
        stats_.streamed += end - e;
        stats_.bytes += bytes;
        ++stats_.copies;
        e = end;
    }
    if (!staged_.empty()) {
        // The layer's part of the bounce: free once its last DMA (layer - 4's, or an earlier
        // chunk's) has read it; the host fills it, then the DMA reads it in stream order.
        const std::uint32_t h = layer % kBounceParts;
        if (bounce_used_[h]) { CUDA_CHECK(cudaEventSynchronize(bounced_[h])); }
        fill_.clear();
        for (std::size_t i = 0; i < staged_.size(); ++i) {
            fill_.emplace_back(staging_.slot(h * staging_.per_part + static_cast<std::uint32_t>(i)), staged_[i].second);
        }
        staging_.copy(fill_);
        for (std::size_t i = 0; i < staged_.size(); ++i) { copies.add(staged_[i].first, fill_[i].first, host); }
        stats_.streamed += staged_.size();
        stats_.bytes += staged_.size() * host;
        stats_.staged += staged_.size();
        stats_.copies += staged_.size();
    }
    copies.flush();
    if (!staged_.empty()) {
        const std::uint32_t h = layer % kBounceParts;
        CUDA_CHECK(cudaEventRecord(bounced_[h], stream_));
        bounce_used_[h] = true;
    }
    CUDA_CHECK(cudaEventRecord(landed_[layer], stream_));
    if (!ssd_planned_[layer].empty()) { ssd_held_.push_back(layer); } // its slots return once landed
}

void ExpertStream::end() noexcept {
    active_ = false;
    // Every read requested is waited for (a read in flight still writes its slot), every copy has
    // landed (the chunk's work waited for each), then the slots go back. Each slot on its own, so
    // one failed wait or release (after a failed call) does not keep the others lent.
    for (std::uint32_t l = 0; l < layers_; ++l) {
        for (std::int32_t& s : ssd_slots_[l]) {
            if (s >= 0 && std::find(ssd_held_.begin(), ssd_held_.end(), l) == ssd_held_.end()) {
                try {
                    (void)ssd_.wait(s);
                    ssd_.release(s);
                } catch (...) {}
                s = -1;
            }
        }
    }
    try {
        while (release_landed(true)) {}
    } catch (...) {}
}

void ExpertStream::before_experts(std::uint32_t layer, cudaStream_t compute) {
    if (active_ && streams_[layer] != 0) { CUDA_CHECK(cudaStreamWaitEvent(compute, landed_[layer], 0)); }
}

void ExpertStream::after_experts(std::uint32_t layer, cudaStream_t compute) {
    if (!active_) { return; }
    CUDA_CHECK(cudaEventRecord(consumed_[layer], compute));
    if (!gated_ && layer + 2 < layers_) { enqueue(layer + 2); }
}

} // namespace infernix::runtime::expert_offload
