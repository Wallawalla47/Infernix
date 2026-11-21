#pragma once

// Inkling 2's few-input projections in one launch (design docs/maintainer/inkling2_mini-design.md §11 K2): several
// BF16 [rows, K] matrices, each bound to one of up to three input vectors, each writing its own [T][rows] output, in
// the arithmetic of ops::linear's BF16 skinny route (src/ops/linear/bf16/bf16_skinny.cuh): for every (row, column) lane
// l folds chunks l, l + 32, ... of 8 weights in order by fma (8 products in k order a chunk), then the xor butterfly
// over the 32 lanes; BF16 or FP32 out. A column's bits therefore equal linear()'s for the same matrix at every width,
// and are independent of the call's other columns and of which matrices share the launch.
//
// Weights are immutable and loaded before the PDL wait; inputs are read after it with ordinary loads. Up to 16 columns
// a pass: a wider call with BF16 or Q8 storage and K of 64, 2304, 3072 or 4608 runs its passes in one launch, each lane
// holding its rows' every chunk in registers; any other wider call runs a launch a pass.

#include <cuda_runtime.h>

#include <cstdint>

namespace infernix::ops::inkling2 {

inline constexpr std::int32_t kMaxProjectionSegments = 8;

// A matrix as BF16 rows, or as its bf16_p12 planes (core/weight.h QuantLayout::P12Rows: per chunk of 8 weights 8 code
// bytes and a word of exponent nibbles; per row a word of base and escape start, rows + 1 of them; the parent's escape
// list), which decode to the same BF16 words, so a segment's outputs do not depend on its storage. Every segment of a
// call has the same storage.
//
// Or (LOSSY, an opt-in load switch) as 8-bit weights, Q8_G32_FP16 RowSplit (include/infernix/ops/q8_g32.h): int8 codes
// [rows][q8_stride] and FP16 group scales [rows][q8_stride / 32]; a weight is code * scale, exact in FP32, and the fold
// is the BF16 route's with those weights (so a column's bits are again independent of the call's width and other
// columns), but they are other bits than the BF16 matrix's.
struct ProjectionSegment {
    const std::uint16_t* weight = nullptr; // BF16 [rows][K], 16-byte aligned (null for packed planes)
    std::int32_t rows           = 0;
    std::int32_t input          = 0;       // which of the call's inputs (0 .. 2)
    void* out                   = nullptr; // [T][rows] BF16 or FP32
    const void* codes           = nullptr; // bf16_p12 or Q8: the matrix's first row in each plane
    const void* nibbles         = nullptr;
    const unsigned* words       = nullptr;
    const unsigned* escapes     = nullptr;
    const std::uint16_t* q8_scales = nullptr; // Q8: FP16 bits, the first row's
    std::int32_t q8_stride         = 0;       // Q8: the code bytes a row (K rounded up to 128)
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

// The same segments on tensor cores (BF16 or Q8 storage, one a call; K a multiple of 64; every weight, input and output
// 16-byte aligned and each output row a multiple of 16 bytes). BF16: ops::linear's MMA arithmetic: rows are the A
// operand of mma.m16n8k16, columns the B operand, and each (row, column) accumulates its k16 steps in ascending K from
// zero in FP32 through the MMA's accumulator, K never split. Q8 (lossy storage): per 32-group in ascending order, the
// group's sum by two chained MMAs from zero over its codes (as BF16, exact) and the BF16 inputs, then
// acc = fma(sum, the row's group scale, acc). A column's bits depend only on its row and its own input: not on the call's
// width, its other columns or matrices, nor on the tile shape the launch picks. Other bits than projections()'s fold.
void projections_mma(const ProjectionArgs& args, cudaStream_t stream);

} // namespace infernix::ops::inkling2
