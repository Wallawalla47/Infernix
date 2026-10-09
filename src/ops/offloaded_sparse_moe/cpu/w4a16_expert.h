#pragma once

// CPU narrow route of offloaded_sparse_moe for experts stored without activation scales: one routed
// expert of Qwen3.8-Flash-Next computed in place from its `nvfp4_expert_rg16_v1` host record with
// the canonical W4A16 arithmetic of docs/maintainer/qwen3_8-flash-next-design.md §16.2.1. Every ISA
// variant returns the same bits, and so does the GPU narrow route.
//
// The record layout, ISA selection and per-expert scalars are those of the W4A4 route
// (w4a4_expert.h). An A16 expert's ExpertScales carry no input scales (zero) and its alpha_* words
// are the record's multipliers m = fl32(1 / weight global scale).

#include "ops/common/canonical_math.h"
#include "ops/offloaded_sparse_moe/cpu/w4a4_expert.h"

#include <cstdint>

namespace infernix::ops::offloaded_moe {

// Encodes a BF16 column of `n` (a multiple of 16) elements, all of whose blocks are encoded with
// one exponent; returns that exponent (canon::a16_column_exponent of the whole column).
int encode_a16(const std::uint16_t* v, int n, canon::A16Block* out);

// Encodes blocks [block_begin, block_end) of a column whose exponent is already known.
void encode_a16_blocks(const std::uint16_t* v, int emax, int block_begin, int block_end, canon::A16Block* out);

// Exact int64 row sums S[row][col] for row groups [rg_begin, rg_end) of a matrix stored as
// row-group-major 144-byte units with `blocks` 16-element blocks per row; `acts[col]` points to
// `blocks` encoded blocks of column col. `out` has (rg_end - rg_begin) * 16 rows of `ncols` values.
void rg16_row_sums_a16(CpuIsa isa, const std::uint8_t* matrix, int blocks, int rg_begin, int rg_end,
                       const canon::A16Block* const* acts, int ncols, std::int64_t* out,
                       int prefetch_bytes = kDefaultPrefetchBytes);

// Phase A for 16-intermediate units [unit_begin, unit_end): gate/up rows and SwiGLU. `x[col]` is
// the encoded x of column col with exponent x_exp[col]; writes h[col][16 * unit + i] in BF16.
void gate_up_units_a16(CpuIsa isa, const std::uint8_t* record, const ExpertScales& scales,
                       const canon::A16Block* const* x, const int* x_exp, int ncols, int unit_begin,
                       int unit_end, std::uint16_t* const* h, int prefetch_bytes = kDefaultPrefetchBytes);

// Phase B for down row groups [rg_begin, rg_end): `h[col]` is the encoded h of column col with
// exponent h_exp[col]; writes y[col][row] in BF16 for those rows.
void down_rows_a16(CpuIsa isa, const std::uint8_t* record, const ExpertScales& scales,
                   const canon::A16Block* const* h, const int* h_exp, int ncols, int rg_begin, int rg_end,
                   std::uint16_t* const* y, int prefetch_bytes = kDefaultPrefetchBytes);

// Whole expert on one thread: y[col][0..2560) from x[col][0..2560), both BF16.
void expert_forward_a16(CpuIsa isa, const std::uint8_t* record, const ExpertScales& scales, int ncols,
                        const std::uint16_t* const* x, std::uint16_t* const* y,
                        int prefetch_bytes = kDefaultPrefetchBytes);

} // namespace infernix::ops::offloaded_moe
