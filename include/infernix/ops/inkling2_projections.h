#pragma once

// Inkling 2's few-input projections in one launch (design docs/maintainer/inkling2_mini-design.md §11 K2): several
// BF16 [rows, K] matrices, each bound to one of up to three input vectors, each writing its own [T][rows] output, in
// the arithmetic of ops::linear's BF16 skinny route (src/ops/linear/bf16/bf16_skinny.cuh): for every (row, column) lane
// l folds chunks l, l + 32, ... of 8 weights in order by fma (8 products in k order a chunk), then the xor butterfly
// over the 32 lanes; BF16 or FP32 out. A column's bits therefore equal linear()'s for the same matrix at every width,
// and are independent of the call's other columns and of which matrices share the launch.
//
// Weights are immutable and loaded before the PDL wait; inputs are read after it with ordinary loads. Up to 16 columns
// a pass (a wider call runs passes of 16).

#include <cuda_runtime.h>

#include <cstdint>

namespace infernix::ops::inkling2 {

inline constexpr std::int32_t kMaxProjectionSegments = 8;

struct ProjectionSegment {
    const std::uint16_t* weight = nullptr; // BF16 [rows][K], 16-byte aligned
    std::int32_t rows           = 0;
    std::int32_t input          = 0;       // which of the call's inputs (0 .. 2)
    void* out                   = nullptr; // [T][rows] BF16 or FP32
};

struct ProjectionArgs {
    std::int32_t columns = 0;
    std::int32_t k       = 0;               // a multiple of 8
    const std::uint16_t* x[3] = {};         // BF16 [T][K] each
    bool fp32_out        = false;
    ProjectionSegment segments[kMaxProjectionSegments];
    std::int32_t count   = 0;
};

void projections(const ProjectionArgs& args, cudaStream_t stream);

} // namespace infernix::ops::inkling2
