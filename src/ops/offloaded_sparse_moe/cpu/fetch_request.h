#pragma once

// The mapped-memory fetch channel through which the GPU asks the host for the records of a layer
// call's SSD-only experts that no CPU job serves (docs/maintainer/qwen3_8-flash-next-design.md
// §19.3.7, the SSD expert tier: prefill, calls past the CPU cap, CPU serving off). Shared by the
// device kernels (moe_layer.cu) and the host side (fetch_channel.h); plain data, no host-only types.
//
// Protocol. The device writes the request (layer, count, experts in job order), a system-scope
// fence, then `sequence` (never 0). The host answers that sequence: it writes `landed = 0` and
// `status = 0`, then `sequence`; then, as records land in job order, record[i] followed by
// `landed = i + 1` (release stores). The device copies record i once it reads landed > i for its
// sequence, and after a pass has copied records [0, b) it stores consumed = sequence << 32 | b, so
// the host may reuse the slots holding them. A nonzero `status` fails the request: the device
// copies nothing more and reports it in the call's error word. A host whose heartbeat word stops
// changing for kHeartbeatTimeoutNs is treated as gone.

#include <cstdint>

namespace infernix::ops::offloaded_moe {

inline constexpr int kMaxFetch = 512; // fetch-served experts per layer call (a layer's experts)

// Device-detected failure codes of a call's error word (layer << 16 | code). Host-reported failures
// carry an errno value, all below 0x8000.
inline constexpr std::uint32_t kErrorUnservedRecord = 0xFF01; // an SSD-only expert with no path to it
inline constexpr std::uint32_t kErrorHostSilent     = 0xFF02; // a heartbeat stopped during a wait

// A wait gives up when the host's heartbeat has not changed for this long (GPU time).
inline constexpr std::uint64_t kHeartbeatTimeoutNs = 1000000000ULL;

struct alignas(64) FetchRequest {
    std::uint32_t sequence;
    std::int32_t layer;
    std::int32_t count;
    std::int32_t reserved;
    std::int32_t expert[kMaxFetch];
};

struct alignas(64) FetchResponse {
    std::uint32_t sequence;
    std::uint32_t landed;
    std::uint32_t status;
    std::uint32_t reserved;
    std::uint64_t record[kMaxFetch]; // host addresses the device can read (mapped pinned memory)
};

} // namespace infernix::ops::offloaded_moe
