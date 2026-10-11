#include "core/block_copy.h"

#include "core/device.h"

#include <algorithm>
#include <stdexcept>

namespace infernix {

namespace {

struct BlockCopies {
    std::uint32_t destination[kDeviceBlockCopiesPerLaunch];
    std::uint32_t source[kDeviceBlockCopiesPerLaunch];
};
constexpr int kThreads = 256;
constexpr int kSlices  = 16; // CTAs per block (gridDim.x); the block is blockIdx.y
constexpr int kUnroll  = 4;  // 16-byte loads in flight per thread

// Streaming loads and stores: a moved block is read once and not read again soon.
__global__ void __launch_bounds__(kThreads)
    copy_device_blocks_kernel(unsigned char* __restrict__ base, std::size_t block_bytes,
                              const __grid_constant__ BlockCopies copies) {
    const std::size_t vectors = block_bytes / 16;
    const auto* src = reinterpret_cast<const uint4*>(base + static_cast<std::size_t>(copies.source[blockIdx.y]) * block_bytes);
    auto* dst       = reinterpret_cast<uint4*>(base + static_cast<std::size_t>(copies.destination[blockIdx.y]) * block_bytes);
    const std::size_t step = static_cast<std::size_t>(gridDim.x) * kThreads;
    std::size_t i          = static_cast<std::size_t>(blockIdx.x) * kThreads + threadIdx.x;
    for (; i + (kUnroll - 1) * step < vectors; i += kUnroll * step) {
        uint4 v[kUnroll];
#pragma unroll
        for (int u = 0; u < kUnroll; ++u) { v[u] = __ldcs(src + i + u * step); }
#pragma unroll
        for (int u = 0; u < kUnroll; ++u) { __stcs(dst + i + u * step, v[u]); }
    }
    for (; i < vectors; i += step) { __stcs(dst + i, __ldcs(src + i)); }
}

} // namespace

void copy_device_blocks(void* base, std::size_t block_bytes, const std::uint32_t* destinations,
                        const std::uint32_t* sources, std::uint32_t count, cudaStream_t stream) {
    if (count == 0 || block_bytes == 0) { return; }
    if ((reinterpret_cast<std::uintptr_t>(base) | block_bytes) % 16 != 0) {
        throw std::invalid_argument("copy_device_blocks needs 16-byte aligned blocks");
    }
    for (std::uint32_t first = 0; first < count; first += kDeviceBlockCopiesPerLaunch) {
        const std::uint32_t n = std::min(kDeviceBlockCopiesPerLaunch, count - first);
        BlockCopies copies;
        std::copy_n(destinations + first, n, copies.destination);
        std::copy_n(sources + first, n, copies.source);
        copy_device_blocks_kernel<<<dim3(kSlices, n), kThreads, 0, stream>>>(static_cast<unsigned char*>(base),
                                                                           block_bytes, copies);
        CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace infernix
