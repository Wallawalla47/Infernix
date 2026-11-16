#pragma once

// Many copies into device memory enqueued with few driver calls. A loop of cudaMemcpyAsync blocks the
// calling thread whenever the stream's copy queue is full (on Windows about one 2.76 MB record's
// transfer time per call, so issuing 12,000 expert records held the host ~1.2 s); one
// cudaMemcpyBatchAsync of the same copies returns in about a millisecond, and the copies run at the
// same link rate. Sources are read in stream order, as with cudaMemcpyAsync. Copies within one
// driver batch run in no particular order, so a copy whose destination overlaps an earlier pending
// one first enqueues the pending copies as a batch: the stream-order result (the later copy wins)
// is kept. A source in registered host memory is passed at its device address
// (cudaHostGetDevicePointer): the batch faults on the host address where the device maps the
// memory elsewhere (WDDM), and copies from the device address at the full link rate.

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <vector>

namespace infernix {

class CopyBatch {
public:
    // Distinct: the caller guarantees that no two copies of the batch write the same bytes (the paged KV pool's
    // record copies, whose pages and records it validates distinct), so no overlap is tracked: a restore of 750
    // 27B blocks is 48,000 copies.
    enum class Destinations : std::uint8_t { MayOverlap, Distinct };
    explicit CopyBatch(cudaStream_t stream, Destinations destinations = Destinations::MayOverlap) noexcept
        : stream_(stream), distinct_(destinations == Destinations::Distinct) {}
    CopyBatch(const CopyBatch&)            = delete;
    CopyBatch& operator=(const CopyBatch&) = delete;

    void add(void* destination, const void* source, std::size_t bytes);
    // Enqueues the pending copies on the stream; nothing when none are pending.
    void flush();
    // The driver calls the batch has made (each flush with copies pending, and each one an overlap forced).
    [[nodiscard]] std::uint32_t calls() const noexcept { return calls_; }

private:
    // Whether a pending copy writes any of [destination, destination + bytes).
    [[nodiscard]] bool pending(const void* destination, std::size_t bytes) const;

    cudaStream_t stream_;
    bool distinct_ = false;
    std::uint32_t calls_ = 0;
    std::vector<void*> destinations_;
    std::vector<const void*> sources_;
    std::vector<std::size_t> sizes_;
    std::map<std::uintptr_t, std::uintptr_t> ranges_; // pending destinations: begin -> end
};

} // namespace infernix
