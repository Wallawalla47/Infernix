#pragma once

// CPU narrow route of offloaded_sparse_moe for experts stored without activation scales under the
// expert-a4-residual-decode fast path (NarrowA16::kA4Residual; docs/maintainer/qwen3_8-flash-next-design.md
// §19.3.23): one routed expert of Qwen3.8-Flash-Next's geometry computed in place from its
// `nvfp4_expert_rg16_v1` host record, with x and h as two NVFP4 terms per 16-element block under one
// FP32 row scale (canon::a4r_block_terms) and the terms' products summed exactly in int64. Every ISA
// variant returns the same bits, and so do the GPU narrow kernels (moe_layer.cu, gate_up_kernel_a4r and
// down_kernel_a4r), so a CPU-served miss gives the GPU's output bit for bit.
//
// A term is an ordinary A4 block (canon::A4Block), so the sums are the W4A4 route's rg16_row_sums over
// kA4ResidualTerms "columns" per column: the weights are read once for every term of up to four
// columns. The record layout, ISA selection and scalars are those of the A16 route (w4a16_expert.h):
// ExpertScales carry no input scales and alpha_* hold the record's multipliers.

#include "ops/common/canonical_math.h"
#include "ops/offloaded_sparse_moe/cpu/w4a4_expert.h"

#include <cstdint>

namespace infernix::ops::offloaded_moe {

// The row scale of a BF16 row of `n` elements (canon::a4r_row_scale of its largest magnitude).
float a4r_row_scale_of(const std::uint16_t* v, int n);

// The terms of blocks [block_begin, block_end) of a row of `blocks` blocks with row scale row_scale,
// term-major: term t of block b at terms[t * blocks + b] (each term of a row is one A4 block run).
void encode_a4r_blocks(const std::uint16_t* v, int blocks, float row_scale, int block_begin, int block_end,
                       canon::A4Block* terms);

// A whole row of `n` (a multiple of 16) elements: its terms (term-major) and, returned, its row scale.
float encode_a4r(const std::uint16_t* v, int n, canon::A4Block* terms);

// Exact two-term int64 row sums S[row][col] for row groups [rg_begin, rg_end) (at most two) of a matrix
// of `blocks` blocks per row: acts[col] holds column col's terms, term-major (kA4ResidualTerms * blocks).
// `out` has (rg_end - rg_begin) * 16 rows of `ncols` values.
void rg16_row_sums_a4r(CpuIsa isa, const std::uint8_t* matrix, int blocks, int rg_begin, int rg_end,
                       const canon::A4Block* const* acts, int ncols, std::int64_t* out,
                       int prefetch_bytes = kDefaultPrefetchBytes);

// Phase A for 16-intermediate units [unit_begin, unit_end): gate/up rows and SwiGLU. x[col] are the
// terms of column col with row scale x_scale[col]; writes h[col][16 * unit + i] in BF16.
void gate_up_units_a4r(CpuIsa isa, const std::uint8_t* record, const ExpertScales& scales,
                       const canon::A4Block* const* x, const float* x_scale, int ncols, int unit_begin, int unit_end,
                       std::uint16_t* const* h, int prefetch_bytes = kDefaultPrefetchBytes);

// Phase B for down row groups [rg_begin, rg_end): h[col] are the terms of column col's h with row
// scale h_scale[col]; writes y[col][row] in BF16 for those rows.
void down_rows_a4r(CpuIsa isa, const std::uint8_t* record, const ExpertScales& scales,
                   const canon::A4Block* const* h, const float* h_scale, int ncols, int rg_begin, int rg_end,
                   std::uint16_t* const* y, int prefetch_bytes = kDefaultPrefetchBytes);

// Whole expert on one thread: y[col][0..2560) from x[col][0..2560), both BF16.
void expert_forward_a4r(CpuIsa isa, const std::uint8_t* record, const ExpertScales& scales, int ncols,
                        const std::uint16_t* const* x, std::uint16_t* const* y,
                        int prefetch_bytes = kDefaultPrefetchBytes);

} // namespace infernix::ops::offloaded_moe
