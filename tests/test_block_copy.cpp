// copy_device_blocks: every destination block holds its source's bytes and every other block is
// unchanged, against the same moves applied on the host. The moves pass one launch's
// kDeviceBlockCopiesPerLaunch in a permuted order, with a block size that leaves a tail after the
// unrolled vectors, and with large blocks; a misaligned block size throws.
#include "core/block_copy.h"
#include "core/device.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

constexpr int kSkip = 77;

struct Case {
    std::size_t block_bytes;
    std::uint32_t blocks;
    std::uint32_t moves;
};

int check(const Case& c, cudaStream_t stream) {
    int failures            = 0;
    const std::size_t bytes = c.block_bytes * c.blocks;
    std::vector<std::uint8_t> host(bytes);
    for (std::size_t i = 0; i < bytes; ++i) { host[i] = static_cast<std::uint8_t>((i * 2654435761U) >> 13); }
    // Sources are the first `moves` blocks; destinations a permutation of the last `moves` (7919 is
    // prime and divides neither count).
    std::vector<std::uint32_t> from(c.moves), to(c.moves);
    for (std::uint32_t i = 0; i < c.moves; ++i) {
        from[i] = i;
        to[i]   = c.blocks - 1 - static_cast<std::uint32_t>((static_cast<std::uint64_t>(i) * 7919U) % c.moves);
    }
    void* device = nullptr;
    CUDA_CHECK(cudaMalloc(&device, bytes));
    CUDA_CHECK(cudaMemcpy(device, host.data(), bytes, cudaMemcpyHostToDevice));
    infernix::copy_device_blocks(device, c.block_bytes, to.data(), from.data(), c.moves, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<std::uint8_t> expected = host, out(bytes);
    for (std::uint32_t i = 0; i < c.moves; ++i) {
        std::memcpy(expected.data() + to[i] * c.block_bytes, host.data() + from[i] * c.block_bytes, c.block_bytes);
    }
    CUDA_CHECK(cudaMemcpy(out.data(), device, bytes, cudaMemcpyDeviceToHost));
    if (out != expected) {
        ++failures;
        std::cerr << c.moves << " moves of " << c.block_bytes << "-byte blocks do not match the host moves\n";
    }
    try {
        infernix::copy_device_blocks(device, c.block_bytes + 8, to.data(), from.data(), 1, stream);
        ++failures;
        std::cerr << "a block size that is not a 16-byte multiple did not throw\n";
    } catch (const std::invalid_argument&) {}
    CUDA_CHECK(cudaFree(device));
    return failures;
}

} // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) { return kSkip; }
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));
    int failures = 0;
    failures += check({4112, 5000, infernix::kDeviceBlockCopiesPerLaunch + 252}, stream);
    failures += check({1048624, 12, 5}, stream);
    CUDA_CHECK(cudaStreamDestroy(stream));
    if (failures != 0) {
        std::cerr << "block copy test failed\n";
        return 1;
    }
    return 0;
}
