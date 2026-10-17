#pragma once

// CPU route of offloaded_sparse_moe for block-FP8 expert banks (fp8_expert_b128_v1; docs/maintainer/
// qwen3_8-flash-next-design.md §16.2.2, §19.3.25): one routed expert computed in place from its host record with
// the exact arithmetic of ops/fp8_block/fp8_exact.h, so every output has the GPU kernels' bits.
//
// The record holds the gate rows, the up rows ([kIntermediate, kHidden], one E4M3FN byte each), the
// down rows ([kHidden, kIntermediate]), then the BF16 128 x 128 tile scales of gate, up and down,
// each row-major [n / 128][k / 128]. Columns arrive in the ggml route's encoding (CpuLimbSlice:
// X = hi * 2^11 + lo), so the team's encode phases serve both banks.

#include "ops/offloaded_sparse_moe/cpu/ggml_expert.h"

#include <cstdint>

namespace infernix::ops::offloaded_moe {

// Gate and up rows [i_begin, i_end) and SwiGLU: h[col][i] in BF16.
void fp8_gate_up(CpuIsa isa, const std::uint8_t* record, const CpuLimbSlice* const* x, const int* x_emax, int ncols,
                 int i_begin, int i_end, std::uint16_t* const* h);

// Down rows [row_begin, row_end): y[col][row] in BF16.
void fp8_down(CpuIsa isa, const std::uint8_t* record, const CpuLimbSlice* const* h, const int* h_emax, int ncols,
              int row_begin, int row_end, std::uint16_t* const* y);

} // namespace infernix::ops::offloaded_moe
