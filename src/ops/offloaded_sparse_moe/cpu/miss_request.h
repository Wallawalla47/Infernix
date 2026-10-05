#pragma once

// The mapped-memory request through which the GPU hands a layer call's CPU-served misses to the
// host expert engine (docs/maintainer/qwen3_8-flash-next-design.md §10.3). Shared by the device
// kernels (moe_layer.cu) and the host service (miss_service.h); plain data, no host-only types.

#include <cstdint>

namespace ninfer::ops::offloaded_moe {

inline constexpr int kMaxCpuJobs        = 256; // CPU-served experts per layer call (prefill assist: up to ~2/3 of a layer's misses)
inline constexpr int kMaxCpuColumns     = 8;   // columns per CPU-served expert (the narrow route's n)
inline constexpr int kMaxCpuCallColumns = 256; // columns of a CPU-served call (decode: 8 lanes x width 16; prefill assist: < 256)
inline constexpr int kMaxCpuXColumns    = 256; // x columns one request publishes (compacted: at most the call's columns)
inline constexpr int kMaxLandingSlots   = 32;  // landing frames of one call (MoeExpertSource::landing)

// Written by the device: every field, a system-scope fence, then `sequence`. The host answers by
// writing the same value to the channel's done word after every output is written.
struct alignas(64) MissRequest {
    std::uint32_t sequence;
    std::int32_t layer;
    std::int32_t jobs;
    std::int32_t reserved;
    std::int32_t expert[kMaxCpuJobs];
    std::int32_t ncols[kMaxCpuJobs];
    std::int32_t column[kMaxCpuJobs][kMaxCpuColumns]; // x column of each of the job's columns
};

} // namespace ninfer::ops::offloaded_moe
