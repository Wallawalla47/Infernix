#include "models/qwen4_exp/execution/expert_stream.h"

#include "core/copy_batch.h"
#include "core/device.h"

#include <algorithm>
#include <stdexcept>

namespace infernix::models::qwen4_exp::execution {
namespace {

constexpr std::size_t kAlign = 256;

std::size_t align_up(std::size_t bytes) { return (bytes + kAlign - 1) / kAlign * kAlign; }

} // namespace

ExpertStream::ExpertStream(std::uint32_t layers, std::uint32_t experts, std::uint64_t record_stride, RecordOf record_of,
                           bool one_allocation)
    : experts_(experts), layers_(layers), stride_(record_stride), record_of_(std::move(record_of)),
      one_allocation_(one_allocation) {
    if (experts_ == 0 || layers_ == 0 || stride_ == 0 || stride_ % 16 != 0 || !record_of_) {
        throw std::invalid_argument("expert stream: empty geometry or unaligned records");
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
}

ExpertStream::~ExpertStream() {
    if (stream_ != nullptr) { (void)cudaStreamSynchronize(stream_); }
    for (auto event : landed_) { (void)cudaEventDestroy(event); }
    for (auto event : consumed_) { (void)cudaEventDestroy(event); }
    if (uploaded_ != nullptr) { (void)cudaEventDestroy(uploaded_); }
    if (start_ != nullptr) { (void)cudaEventDestroy(start_); }
    if (stream_ != nullptr) { (void)cudaStreamDestroy(stream_); }
}

std::uint32_t ExpertStream::slots_in(std::size_t bytes) const noexcept {
    const std::size_t tables = align_up(static_cast<std::size_t>(layers_) * experts_ * sizeof(std::int32_t));
    if (bytes <= tables) { return 0; }
    return static_cast<std::uint32_t>((bytes - tables) / stride_) & ~1U;
}

void ExpertStream::begin(DeviceSpan ring, const std::int32_t* residency, cudaStream_t compute, bool gated) {
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
    if (gated) {
        uploading_ = false;
        std::fill(streams_.begin(), streams_.end(), std::uint8_t{0});
        CUDA_CHECK(cudaEventRecord(start_, compute));
        CUDA_CHECK(cudaStreamWaitEvent(stream_, start_, 0));
        active_ = true;
        return;
    }
    auto* tables = static_cast<std::int32_t*>(tables_host_.data());
    for (std::uint32_t l = 0; l < layers_; ++l) {
        std::int32_t* row = tables + static_cast<std::size_t>(l) * experts_;
        const std::int32_t* frames = residency + static_cast<std::size_t>(l) * experts_;
        std::uint32_t taken = 0;
        for (std::uint32_t e = 0; e < experts_; ++e) {
            const bool stream = frames[e] < 0 && taken < half_ && record_of_(l, e) != nullptr;
            row[e]            = stream ? static_cast<std::int32_t>((l % 2) * half_ + taken++) : -1;
        }
        streams_[l] = taken > 0 ? 1 : 0;
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
    std::uint32_t taken        = 0;
    for (std::uint32_t e = 0; e < experts_; ++e) {
        // An SSD-only expert has no host record to copy: the CPU or the fetch channel serves it.
        const bool copy = columns[e] > 0 && frames[e] < 0 && cpu[e] == 0 && taken < half_ &&
                          record_of_(layer, e) != nullptr;
        row[e]          = copy ? static_cast<std::int32_t>((layer % 2) * half_ + taken++) : -1;
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

void ExpertStream::enqueue(std::uint32_t layer) {
    if (layer >= 2) { CUDA_CHECK(cudaStreamWaitEvent(stream_, consumed_[layer - 2], 0)); }
    const auto* row = static_cast<const std::int32_t*>(tables_host_.data()) + static_cast<std::size_t>(layer) * experts_;
    // Runs of consecutive experts in consecutive slots and adjacent records of one allocation: one
    // copy per run (a copy may not span two pinned allocations, even adjacent ones), all enqueued as
    // one batch so the host does not wait on the copy queue.
    CopyBatch copies(stream_);
    for (std::uint32_t e = 0; e < experts_;) {
        if (row[e] < 0) {
            ++e;
            continue;
        }
        const std::uint8_t* source = record_of_(layer, e);
        std::uint32_t end = e + 1;
        while (one_allocation_ && end < experts_ && row[end] == row[end - 1] + 1 &&
               record_of_(layer, end) == source + static_cast<std::size_t>(end - e) * stride_) {
            ++end;
        }
        const std::size_t bytes = static_cast<std::size_t>(end - e) * stride_;
        copies.add(records_ + static_cast<std::size_t>(row[e]) * stride_, source, bytes);
        stats_.streamed += end - e;
        ++stats_.copies;
        e = end;
    }
    copies.flush();
    CUDA_CHECK(cudaEventRecord(landed_[layer], stream_));
}

void ExpertStream::before_experts(std::uint32_t layer, cudaStream_t compute) {
    if (active_ && streams_[layer] != 0) { CUDA_CHECK(cudaStreamWaitEvent(compute, landed_[layer], 0)); }
}

void ExpertStream::after_experts(std::uint32_t layer, cudaStream_t compute) {
    if (!active_) { return; }
    CUDA_CHECK(cudaEventRecord(consumed_[layer], compute));
    if (!gated_ && layer + 2 < layers_) { enqueue(layer + 2); }
}

} // namespace infernix::models::qwen4_exp::execution
