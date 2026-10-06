#pragma once

// The host-to-device copy rate of pinned memory (the PCIe link as copy engines see it), measured
// once at startup so bandwidth-dependent choices follow the machine instead of constants: a GPU at
// PCIe 5.0 x8 copies about 27.5 GB/s, at x16 about twice that.

#include <cstddef>

namespace ninfer {

// Copies `bytes` from `source` (pinned host memory) to `destination` (device memory) on a private
// stream, once to warm up and then `repetitions` times, and returns the best rate in bytes per
// second. `destination` is overwritten; the caller's streams are not involved.
[[nodiscard]] double measure_h2d_bytes_per_second(const void* source, void* destination, std::size_t bytes,
                                                  int repetitions);

} // namespace ninfer
