#pragma once

// Inkling 2's LoRA term (`--lora`, docs/lora.md): y = W x + s B (A x), the adapter's part added AFTER the projection's own
// rounding, as PEFT adds it to the base layer's output (result = base(x); result = result + B(A(x)) * s; cast back).
// The base projection kernels are not touched: the Forward runs them, then U = A x through the same fused projections
// (ProjectionArgs: BF16 weights, the call's route), then this Op once for a group of adapted outputs.
//
// Arithmetic (one fixed sequence, so a column's bits depend only on its row, its U and its y):
//   c = a chain of BF16 m16n8k16 MMA k-steps over j in ascending order (rank / 16 steps, from +0) of B[n, j] U[j, t]
//       (the canonical addend of include/infernix/ops/lora.h),
//   y[t * y_stride + n] = bf16(fl32(fl32(y) + fl32(c * scale)))   (FP32 outputs: fl32(y + fl32(c * scale)) unrounded).
// The call's width, its other columns, the launch's tile shape and the target's position in the launch change no bit.
// With B = 0 the output is y's bits (but for a negative zero, which the FP32 add makes +0).

#include "infernix/ops/lora.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace infernix::ops::inkling2 {

inline constexpr std::int32_t kMaxLoraTargets = 8;

struct LoraTarget {
    const std::uint16_t* b = nullptr; // BF16 [rows rounded up to 16][rank], 16-byte aligned, immutable (padded rows zero)
    const std::uint16_t* u = nullptr; // BF16, a column's `rank` values, columns u_stride apart, 4-byte aligned: the shrink's
    void* y                = nullptr; // [T][y_stride] BF16, or FP32 with fp32_out: updated in place, rows [0, rows)
    std::int32_t rows      = 0;
    std::int32_t y_stride  = 0;       // y's elements a column (a gate's forget and message rows share a column)
    float scale            = 0.0F;
    std::int32_t u_stride  = 0;       // U's elements a column (a stack of a group's modules); 0: the rank
};

struct LoraTermArgs {
    std::int32_t columns = 0;
    std::int32_t rank    = 0; // padded: a multiple of 16, at most kLoraMaxRank (every target's)
    bool fp32_out        = false;
    LoraTarget targets[kMaxLoraTargets];
    std::int32_t count = 0;
};

// Adds every target's term to its output in one launch. Reads y, U and B; writes y; no workspace. A consumer of the
// kernel before it (PDL). Throws std::invalid_argument on a malformed target.
void lora_term_add(const LoraTermArgs& args, cudaStream_t stream);

} // namespace infernix::ops::inkling2
