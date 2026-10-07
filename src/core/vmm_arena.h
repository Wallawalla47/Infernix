#pragma once

// A device arena whose backing grows and shrinks at its top while its base address never moves
// (CUDA virtual memory management): one virtual range is reserved up front, and physical chunks
// are mapped or released at the top. Kernels and captured CUDA graphs may keep addresses below the
// mapped top across resizes. Not thread-safe; the owner orders every resize after the device work
// that reads the released range.

#include <cuda.h>

#include <cstddef>
#include <vector>

namespace infernix {

class VmmArena {
public:
    // Reserves `reserve_bytes` (rounded up to whole chunks) of address space on `device`; maps
    // nothing. `chunk_bytes` must be a multiple of the device's allocation granularity.
    VmmArena(int device, std::size_t reserve_bytes, std::size_t chunk_bytes);
    ~VmmArena();
    VmmArena(const VmmArena&)            = delete;
    VmmArena& operator=(const VmmArena&) = delete;

    // Whether the device supports virtual memory management.
    [[nodiscard]] static bool supported(int device);

    [[nodiscard]] void* base() const noexcept { return reinterpret_cast<void*>(base_); }
    [[nodiscard]] std::size_t chunk_bytes() const noexcept { return chunk_; }
    [[nodiscard]] std::size_t reserved_bytes() const noexcept { return reserved_; }
    [[nodiscard]] std::size_t mapped_bytes() const noexcept { return chunks_.size() * chunk_; }

    // Maps one more chunk at the top; false (nothing mapped) when the device has no memory for it.
    [[nodiscard]] bool map_chunk();
    // Releases the top chunk; the caller has finished every device access to it.
    void unmap_chunk();

private:
    int device_            = 0;
    std::size_t chunk_     = 0;
    std::size_t reserved_  = 0;
    CUdeviceptr base_      = 0;
    std::vector<CUmemGenericAllocationHandle> chunks_;
};

// A reserved virtual range whose chunks are mapped individually, anywhere in it (an elastic pool
// whose planes each grow at their own end: design §19.3.11). Unmapped chunks must never be
// accessed. Not thread-safe; the owner orders every unmap after the device work that reads it.
class VmmRange {
public:
    VmmRange(int device, std::size_t reserve_bytes, std::size_t chunk_bytes);
    ~VmmRange();
    VmmRange(const VmmRange&)            = delete;
    VmmRange& operator=(const VmmRange&) = delete;

    [[nodiscard]] void* base() const noexcept { return reinterpret_cast<void*>(base_); }
    [[nodiscard]] std::size_t chunk_bytes() const noexcept { return chunk_; }
    [[nodiscard]] std::size_t reserved_bytes() const noexcept { return reserved_; }
    [[nodiscard]] std::size_t chunk_count() const noexcept { return handles_.size(); }
    [[nodiscard]] std::size_t mapped_bytes() const noexcept { return mapped_ * chunk_; }
    [[nodiscard]] bool mapped(std::size_t chunk) const noexcept { return handles_[chunk] != 0; }

    // Maps chunk `chunk`; false (nothing mapped) when the device has no memory for it.
    [[nodiscard]] bool map(std::size_t chunk);
    // Releases chunk `chunk`; the caller has finished every device access to it.
    void unmap(std::size_t chunk);

private:
    int device_           = 0;
    std::size_t chunk_    = 0;
    std::size_t reserved_ = 0;
    std::size_t mapped_   = 0;
    CUdeviceptr base_     = 0;
    std::vector<CUmemGenericAllocationHandle> handles_; // 0: unmapped
};

} // namespace infernix
