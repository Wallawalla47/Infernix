// Fills every SM's shared memory with a pattern (tests/ops/dirty_shared_memory.h).
#include "dirty_shared_memory.h"

#include <stdexcept>
#include <string>

namespace infernix::test {
namespace {

constexpr int kDirtyBytes = 99 * 1024;

__global__ void dirty_shared_memory_kernel(std::uint32_t pattern, int words) {
    extern __shared__ std::uint32_t dirty[];
    for (int i = static_cast<int>(threadIdx.x); i < words; i += static_cast<int>(blockDim.x)) {
        dirty[i] = pattern ^ (static_cast<std::uint32_t>(i) * 2654435761U);
    }
    __syncthreads();
    // A read the compiler cannot drop keeps the stores.
    if (threadIdx.x == 0 && dirty[words - 1] == 0x12345678U) { dirty[0] = 0; }
}

void check(cudaError_t error, const char* what) {
    if (error != cudaSuccess) { throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(error)); }
}

} // namespace

void dirty_shared_memory(std::uint32_t pattern, cudaStream_t stream) {
    static const int sms = [] {
        check(cudaFuncSetAttribute(dirty_shared_memory_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kDirtyBytes),
              "dirty shared memory attribute");
        int count = 0;
        check(cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, 0), "SM count");
        return count;
    }();
    dirty_shared_memory_kernel<<<2 * sms, 256, kDirtyBytes, stream>>>(pattern, kDirtyBytes / 4);
    check(cudaGetLastError(), "dirty shared memory");
}

} // namespace infernix::test
