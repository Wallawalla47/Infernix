// CopyBatch: batched host-to-device copies keep cudaMemcpyAsync's stream-order result. Disjoint
// copies all land; a copy overlapping an earlier pending one (same destination, or a partial
// overlap) wins over it, as a later cudaMemcpyAsync would; the batch also orders after earlier
// work on its stream.
#include "core/copy_batch.h"
#include "core/device.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

namespace {

constexpr int kSkip = 77;

int check(const std::vector<std::uint8_t>& actual, const std::vector<std::uint8_t>& expected, const char* label) {
    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (actual[i] != expected[i]) {
            std::cerr << label << ": byte " << i << " is " << int(actual[i]) << ", expected " << int(expected[i]) << '\n';
            return 1;
        }
    }
    return 0;
}

} // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) { return kSkip; }
    constexpr std::size_t kBlock = 4096, kBlocks = 64;
    std::uint8_t* host   = nullptr;
    std::uint8_t* device = nullptr;
    CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&host), kBlock * kBlocks, cudaHostAllocDefault));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device), kBlock * kBlocks));
    for (std::size_t b = 0; b < kBlocks; ++b) { std::memset(host + b * kBlock, static_cast<int>(b + 1), kBlock); }
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    std::vector<std::uint8_t> expected(kBlock * kBlocks, 0), actual(kBlock * kBlocks);
    int failures = 0;

    // Earlier work on the stream: the batch must land after this memset.
    CUDA_CHECK(cudaMemsetAsync(device, 0, kBlock * kBlocks, stream));
    {
        infernix::CopyBatch batch(stream);
        // Disjoint: device block d <- host block (d * 7) % kBlocks for the first 32 blocks.
        for (std::size_t d = 0; d < 32; ++d) {
            const std::size_t s = (d * 7) % kBlocks;
            batch.add(device + d * kBlock, host + s * kBlock, kBlock);
            std::memset(expected.data() + d * kBlock, static_cast<int>(s + 1), kBlock);
        }
        // The same destination twice: the later copy wins.
        batch.add(device + 40 * kBlock, host + 1 * kBlock, kBlock);
        batch.add(device + 40 * kBlock, host + 2 * kBlock, kBlock);
        std::memset(expected.data() + 40 * kBlock, 3, kBlock);
        // A partial overlap: blocks 44-45 from host block 5, then 45-46 from host block 9 (two blocks).
        batch.add(device + 44 * kBlock, host + 5 * kBlock, 2 * kBlock);
        batch.add(device + 45 * kBlock, host + 9 * kBlock, 2 * kBlock);
        std::memset(expected.data() + 44 * kBlock, 6, kBlock);
        std::memset(expected.data() + 45 * kBlock, 10, kBlock);
        std::memset(expected.data() + 46 * kBlock, 11, kBlock);
        // Ends exactly where a pending copy begins: no overlap, both land.
        batch.add(device + 50 * kBlock, host + 20 * kBlock, kBlock);
        batch.add(device + 49 * kBlock, host + 21 * kBlock, kBlock);
        std::memset(expected.data() + 50 * kBlock, 21, kBlock);
        std::memset(expected.data() + 49 * kBlock, 22, kBlock);
        batch.flush();
        batch.flush(); // nothing pending: no work
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpy(actual.data(), device, kBlock * kBlocks, cudaMemcpyDeviceToHost));
    failures += check(actual, expected, "copy batch");

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(device));
    CUDA_CHECK(cudaFreeHost(host));
    if (failures == 0) { std::cout << "copy batch checks passed\n"; }
    return failures == 0 ? 0 : 1;
}
