// RegisteredHostBuffer's sliced pinning (candidates SS-02): when the driver refuses one registration of
// the whole block, the block is registered in slices that end at object boundaries, and every object
// reads through the device mapping of its own slice. Needs a CUDA device (77: skipped without one).

#include "core/host_memory.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kMiB = std::size_t{1} << 20;

void require(bool value, const std::string& message) {
    if (!value) { throw std::runtime_error(message); }
}

// The word the device reads at byte `offset` of the block, through device_at().
std::uint32_t device_word(const infernix::RegisteredHostBuffer& block, std::size_t offset) {
    std::uint32_t word = 0;
    if (cudaMemcpy(&word, block.device_at(offset), sizeof(word), cudaMemcpyDefault) != cudaSuccess) {
        throw std::runtime_error("a copy from the block's device mapping failed");
    }
    return word;
}

void mark(infernix::RegisteredHostBuffer& block, std::size_t offset, std::uint32_t value) {
    std::memcpy(block.data() + offset, &value, sizeof(value));
}

void test_whole_block() {
    infernix::RegisteredHostBuffer block(8 * kMiB);
    mark(block, 5 * kMiB, 0x5a5a1234U);
    const std::vector<std::size_t> boundaries{2 * kMiB, 4 * kMiB};
    block.pin(boundaries);
    require(block.registrations() == 1, "an accepted whole block was registered in slices");
    require(device_word(block, 5 * kMiB) == 0x5a5a1234U, "the device does not read the whole block's bytes");
}

void test_slices() {
    infernix::testing::refuse_host_registrations_above(16 * kMiB);
    infernix::RegisteredHostBuffer block(64 * kMiB);
    // Objects at 0, 8, 20, 32 and 48 MiB (and an unaligned offset, which never cuts a slice).
    const std::vector<std::size_t> boundaries{0, 8 * kMiB, 20 * kMiB, 20 * kMiB + 100, 32 * kMiB, 48 * kMiB};
    for (std::size_t i = 0; i < boundaries.size(); ++i) {
        mark(block, boundaries[i], 0xc0de0000U + static_cast<std::uint32_t>(i));
    }
    mark(block, 64 * kMiB - 4, 0xfeedU);
    block.pin(boundaries, 16 * kMiB);
    infernix::testing::refuse_host_registrations_above(0);
    require(block.registrations() == 5, "the refused block was not registered in five slices at its objects");
    for (std::size_t i = 0; i < boundaries.size(); ++i) {
        require(device_word(block, boundaries[i]) == 0xc0de0000U + static_cast<std::uint32_t>(i),
                "an object reads other bytes through its slice's device mapping");
    }
    require(device_word(block, 64 * kMiB - 4) == 0xfeedU, "the last slice's end reads other bytes");
}

void test_refused_slice() {
    infernix::testing::refuse_host_registrations_above(16 * kMiB);
    infernix::RegisteredHostBuffer block(64 * kMiB);
    const std::vector<std::size_t> boundaries{48 * kMiB}; // one 48 MiB object
    bool refused = false;
    try {
        block.pin(boundaries, 16 * kMiB);
    } catch (const std::runtime_error& error) {
        refused = std::string(error.what()).find("a slice failed too") != std::string::npos;
    }
    infernix::testing::refuse_host_registrations_above(0);
    require(refused && block.registrations() == 0 && block.device_at(0) == nullptr,
            "a refused slice did not fail the pin, or left slices registered");
}

} // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::cout << "no CUDA device: skipped\n";
        return 77;
    }
    try {
        test_whole_block();
        test_slices();
        test_refused_slice();
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    std::cout << "host memory tests passed\n";
    return 0;
}
