#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace infernix::test {

// Fills the shared memory of every SM (two CTAs of 99 KiB each) with a pattern derived from
// `pattern`, so a kernel launched next on `stream` finds garbage wherever it reads shared memory it
// did not write.
void dirty_shared_memory(std::uint32_t pattern, cudaStream_t stream);

} // namespace infernix::test
