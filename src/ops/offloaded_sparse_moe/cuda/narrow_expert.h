#pragma once

// GPU narrow route of offloaded_sparse_moe (docs/maintainer/qwen3_8-flash-next-design.md §8.5,
// §16.2): routed experts with n <= 8 token columns, computed from `nvfp4_expert_rg16_v1` records
// with the canonical W4A4 arithmetic. The output bits equal the CPU engine's
// (cpu/w4a4_expert.h) for every record, scale and input.
//
// This is the first, correctness-first form of K7a/K7b: a quantize kernel, a gate/up kernel with
// one CTA per 16-intermediate unit and dp4a products, and a down kernel. It has no landing
// tickets, PDL, persistent scheduling, shared-memory weight ring or combine yet (§8.5 lists them).

#include "ops/common/canonical_math.h"
#include "ops/offloaded_sparse_moe/cpu/w4a4_expert.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::offloaded_moe {

// One routed expert of one layer call. Device pointers; x and y are [ncols][kHidden] BF16.
struct NarrowJob {
    const std::uint8_t* record; // a cache frame, a staging frame, or mapped host memory
    ExpertScales scales;
    const std::uint16_t* x;
    std::uint16_t* y;
    int ncols; // 1..kMaxColumns
};

// Device workspace bytes for `jobs` jobs: A4 x for gate and up, and A4 h.
std::size_t narrow_workspace_bytes(int jobs);

// Runs `jobs` jobs (an array in device memory) on `stream`. `workspace` holds
// narrow_workspace_bytes(jobs) bytes of device memory.
void launch_narrow_experts(const NarrowJob* jobs, int job_count, void* workspace, cudaStream_t stream);

} // namespace ninfer::ops::offloaded_moe
