#include "infernix/ops/kernel_images.h"

#include "core/device.h"

#include <atomic>
#include <cuda_runtime.h>

namespace infernix::ops {
namespace {

__global__ void load_kernel_images_kernel() {}

std::atomic<bool> g_loaded{false};

// The resident RAM the Ops module's first launch added on the 2026-11-13 build (kernel_images.h): 4,596 MiB across a
// start's load_kernel_images (the debug line "CUDA kernel images"), on the 27B and Qwen3.8-Flash-Next alike.
constexpr std::uint64_t kImageBytes = 4600ULL << 20;

} // namespace

void load_kernel_images() {
    load_kernel_images_kernel<<<1, 1>>>();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    g_loaded.store(true, std::memory_order_relaxed);
}

std::uint64_t kernel_images_pending_bytes() noexcept {
    return g_loaded.load(std::memory_order_relaxed) ? 0 : kImageBytes;
}

} // namespace infernix::ops
