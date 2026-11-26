#include "infernix/ops/kernel_images.h"

#include "core/device.h"

#include <atomic>
#include <chrono>
#include <cuda_runtime.h>
#include <exception>
#include <thread>
#include <utility>

namespace infernix::ops {
namespace {

__global__ void load_kernel_images_kernel() {}

std::atomic<bool> g_loaded{false};

// The resident RAM the Ops module's first launch adds across a start's load_kernel_images (the debug line "CUDA
// kernel images"), measured 2026-11-15 on the 27B, two starts each: 1,467 MiB for the default build, and 4,596 MiB
// with CUDA line information (INFERNIX_CUDA_LINEINFO, a profiling build), whose image carries 1.5 GiB of embedded
// PTX text and line tables beside the same machine code.
#ifdef INFERNIX_CUDA_LINEINFO
constexpr std::uint64_t kImageBytes = 4600ULL << 20;
#else
constexpr std::uint64_t kImageBytes = 1500ULL << 20;
#endif

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

struct KernelImagesLoad::State {
    std::thread thread;
    std::exception_ptr error;
    double seconds = 0.0;
};

KernelImagesLoad::KernelImagesLoad(int device) {
    if (g_loaded.load(std::memory_order_relaxed)) { return; }
    state_       = new State;
    State* state = state_;
    try {
        state->thread = std::thread([state, device] {
            try {
                const auto begin = std::chrono::steady_clock::now();
                CUDA_CHECK(cudaSetDevice(device)); // the device's primary context, the start's own
                load_kernel_images();
                state->seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
            } catch (...) {
                state->error = std::current_exception();
            }
        });
    } catch (...) {
        delete state_;
        state_ = nullptr; // no thread: the first kernel launch reads the images, as before kernel_images
    }
}

KernelImagesLoad::~KernelImagesLoad() {
    if (state_ == nullptr) { return; }
    if (state_->thread.joinable()) { state_->thread.join(); }
    delete state_;
}

double KernelImagesLoad::finish() {
    if (state_ == nullptr) { return 0.0; }
    if (state_->thread.joinable()) { state_->thread.join(); }
    if (state_->error) { std::rethrow_exception(std::exchange(state_->error, nullptr)); }
    return state_->seconds;
}

} // namespace infernix::ops
