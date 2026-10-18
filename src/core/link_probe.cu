#include "core/link_probe.h"

#include "core/device.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>

namespace infernix {
namespace {

// Keeps one warp of each SM spinning for `cycles` clock cycles, or with `stop` until the word is set
// (bounded by `cycles`): compute load, no host memory traffic.
__global__ void spin_kernel(long long cycles, const volatile unsigned* stop) {
    const long long start = clock64();
    while (clock64() - start < cycles) {
        if (stop != nullptr && *stop != 0U) { return; }
    }
}

// The warm-up's length. Copies alone did not bring the link to its top PCIe generation: probes of a
// warm-up copy and three timed ones measured 3.0 GB/s (once 12.0, the Gen2 and Gen4 x8 rates) on a
// Gen5 x8 link in about one IQ3_S startup in seven; with the SMs busy, 0 of 60 (design §19.3.26).
constexpr float kWarmMilliseconds = 200.0F;

} // namespace

double measure_h2d_bytes_per_second(const void* source, void* destination, std::size_t bytes, int repetitions) {
    if (source == nullptr || destination == nullptr || bytes == 0 || repetitions <= 0) {
        throw std::invalid_argument("link probe needs a source, a destination, bytes and repetitions");
    }
    int device = 0, sms = 0, clock_khz = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    CUDA_CHECK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, device));
    // Cycles of `ms` milliseconds at the device's highest clock (a lower clock only spins longer).
    const auto cycles = [&](float ms) { return static_cast<long long>(static_cast<double>(clock_khz) * ms); };
    cudaStream_t stream = nullptr, busy = nullptr;
    cudaEvent_t start = nullptr, stop = nullptr, warm = nullptr;
    unsigned* done = nullptr;
    double best = 0.0;
    const auto release = [&] {
        if (done != nullptr) { cudaFree(done); }
        if (start != nullptr) { cudaEventDestroy(start); }
        if (stop != nullptr) { cudaEventDestroy(stop); }
        if (warm != nullptr) { cudaEventDestroy(warm); }
        if (busy != nullptr) { cudaStreamDestroy(busy); }
        if (stream != nullptr) { cudaStreamDestroy(stream); }
    };
    try {
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&busy, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventCreateWithFlags(&warm, cudaEventDisableTiming));
        CUDA_CHECK(cudaMalloc(&done, sizeof(unsigned)));
        CUDA_CHECK(cudaMemsetAsync(done, 0, sizeof(unsigned), busy));
        // Warm-up: the SMs busy for kWarmMilliseconds, which brings the GPU (and its link) out of its
        // idle state, while the link copies; then the timed copies run with the SMs still busy, under
        // one spin the copy stream stops once they are done (bounded at 2 s).
        spin_kernel<<<sms, 32, 0, busy>>>(cycles(kWarmMilliseconds), nullptr);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(warm, busy));
        spin_kernel<<<sms, 32, 0, busy>>>(cycles(2000.0F), done);
        CUDA_CHECK(cudaGetLastError());
        do {
            CUDA_CHECK(cudaMemcpyAsync(destination, source, bytes, cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
        } while (cudaEventQuery(warm) == cudaErrorNotReady);
        CUDA_CHECK(cudaEventSynchronize(warm));
        for (int r = 0; r < repetitions; ++r) {
            CUDA_CHECK(cudaEventRecord(start, stream));
            CUDA_CHECK(cudaMemcpyAsync(destination, source, bytes, cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaEventRecord(stop, stream));
            CUDA_CHECK(cudaEventSynchronize(stop));
            float ms = 0.0F;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            if (ms > 0.0F) { best = std::max(best, static_cast<double>(bytes) / (static_cast<double>(ms) * 1e-3)); }
        }
        const unsigned one = 1U;
        CUDA_CHECK(cudaMemcpyAsync(done, &one, sizeof(one), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaStreamSynchronize(busy));
    } catch (...) {
        release();
        throw;
    }
    release();
    return best;
}

} // namespace infernix
