#pragma once

// Moves of equal blocks within one device allocation, issued by the host in microseconds. A
// cudaMemcpyAsync per block (or one cudaMemcpyBatchAsync of the blocks) costs the host about 7.5 us
// per copy: moving 1,300 expert frames of 2.64 MiB held it ~10 ms. Here the SMs copy every block of
// a call in one launch per kDeviceBlockCopiesPerLaunch blocks, whose indices travel as kernel
// parameters.

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace infernix {

inline constexpr std::uint32_t kDeviceBlockCopiesPerLaunch = 2048;

// In stream order, block destinations[i] of `base` receives block sources[i], for i < count.
// Destinations are distinct and none is a source; `base` and `block_bytes` are 16-byte aligned.
void copy_device_blocks(void* base, std::size_t block_bytes, const std::uint32_t* destinations,
                        const std::uint32_t* sources, std::uint32_t count, cudaStream_t stream);

} // namespace infernix
