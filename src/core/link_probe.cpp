#include "core/link_probe.h"

#include "core/device.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>

namespace infernix {

double measure_h2d_bytes_per_second(const void* source, void* destination, std::size_t bytes, int repetitions) {
    if (source == nullptr || destination == nullptr || bytes == 0 || repetitions <= 0) {
        throw std::invalid_argument("link probe needs a source, a destination, bytes and repetitions");
    }
    cudaStream_t stream = nullptr;
    cudaEvent_t start = nullptr, stop = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    double best = 0.0;
    try {
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaMemcpyAsync(destination, source, bytes, cudaMemcpyHostToDevice, stream)); // warm-up
        for (int r = 0; r < repetitions; ++r) {
            CUDA_CHECK(cudaEventRecord(start, stream));
            CUDA_CHECK(cudaMemcpyAsync(destination, source, bytes, cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaEventRecord(stop, stream));
            CUDA_CHECK(cudaEventSynchronize(stop));
            float ms = 0.0F;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            if (ms > 0.0F) { best = std::max(best, static_cast<double>(bytes) / (static_cast<double>(ms) * 1e-3)); }
        }
    } catch (...) {
        if (start != nullptr) { cudaEventDestroy(start); }
        if (stop != nullptr) { cudaEventDestroy(stop); }
        cudaStreamDestroy(stream);
        throw;
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
    return best;
}

} // namespace infernix
