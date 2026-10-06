#pragma once

// Many copies into device memory enqueued with few driver calls. A loop of cudaMemcpyAsync blocks the
// calling thread whenever the stream's copy queue is full (on Windows about one 2.76 MB record's
// transfer time per call, so issuing 12,000 expert records held the host ~1.2 s); one
// cudaMemcpyBatchAsync of the same copies returns in about a millisecond, and the copies run at the
// same link rate. Sources are read in stream order, as with cudaMemcpyAsync. Copies within one
// driver batch run in no particular order, so a copy whose destination overlaps an earlier pending
// one first enqueues the pending copies as a batch: the stream-order result (the later copy wins)
// is kept.

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <vector>

namespace ninfer {

class CopyBatch {
public:
    explicit CopyBatch(cudaStream_t stream) noexcept : stream_(stream) {}
    CopyBatch(const CopyBatch&)            = delete;
    CopyBatch& operator=(const CopyBatch&) = delete;

    void add(void* destination, const void* source, std::size_t bytes);
    // Enqueues the pending copies on the stream; nothing when none are pending.
    void flush();

private:
    cudaStream_t stream_;
    std::vector<void*> destinations_;
    std::vector<const void*> sources_;
    std::vector<std::size_t> sizes_;
    std::map<std::uintptr_t, std::uintptr_t> ranges_; // pending destinations: begin -> end
};

} // namespace ninfer
