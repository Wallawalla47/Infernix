#pragma once

// CPU route of offloaded_sparse_moe's 8-bit activations for NVFP4 experts stored without activation scales
// (W4A16 banks: the `nvfp4_expert_rg16_v1` records of w4a16_expert.h, at either registered geometry) under
// `--a8-activations` (LOSSY, opt-in; never on the reproducible or lossless route). The arithmetic is
// ../nvfp4_a8.h's: x and h quantized per 32-element slice by ops/ggml/ggml_a8.h's codec, each 16-element block's
// exact int32 P_b, its term fma(fl32(P_b d_s), S^, acc) added into partial s % P (P = 16 for gate and up rows,
// 4 for down rows) in ascending s, the butterfly, then nvfp4_a8::row_output with the record's multiplier. Every
// ISA variant forms the same IEEE binary32 operations in that order (the vector variants keep a row per lane), so
// a CPU-served miss returns the GPU 8-bit expert kernels' bits. The x86 variants need the default MXCSR (no
// flush-to-zero or denormals-are-zero), as the scalar canonical arithmetic does.
//
// The ISA selection, record layout and ExpertScales are w4a4_expert.h's; an A16 expert's alpha_* words are the
// record's multipliers m = fl32(1 / weight global scale).

#include "ops/offloaded_sparse_moe/cpu/w4a4_expert.h"

#include <cstdint>

namespace infernix::ops::offloaded_moe {

// One quantized 32-element slice of a column: its codes as signed bytes (byte j: element j; the vector variants
// take a block's 16 as four int32 words), each 16-element block's correction -12 * (sum of its codes) (the
// variants multiply the unsigned biased weight codes c2 + 12, so starting a block's dot from its correction
// leaves P_b), and the slice's scale d (0 for a slice whose largest magnitude is below 2^-120, NaN for one
// holding Inf or NaN; its codes are then 0).
struct CpuW4A8Slice {
    std::int8_t q[32];
    std::int32_t bias[2];
    float d;
};

// Quantizes slices [slice_begin, slice_end) of the BF16 column v (slice s: v[32 s .. 32 s + 32)) into out[s].
void encode_a8_slices(const std::uint16_t* v, int slice_begin, int slice_end, CpuW4A8Slice* out);

// The p_0 of rows of row groups [rg_begin, rg_end) of a matrix stored as row-group-major 144-byte units with
// `blocks` (even) 16-element blocks per row, against `ncols` quantized columns acts[col] (blocks / 2 slices each),
// with `partials` partials (a power of two, at most 16, at most blocks / 2): out[(row - 16 rg_begin) * ncols + col].
void rg16_rows_a8(CpuIsa isa, const std::uint8_t* matrix, int blocks, int partials, int rg_begin, int rg_end,
                  const CpuW4A8Slice* const* acts, int ncols, float* out, int prefetch_bytes = kDefaultPrefetchBytes);

// Phase A for 16-intermediate units [unit_begin, unit_end) of an expert of `shape`: gate/up rows and SwiGLU.
// `x[col]` is column col quantized (shape.hidden / 32 slices); writes h[col][16 * unit + i] in BF16. Slice t of h
// is units 2t and 2t + 1.
void gate_up_units_a8(CpuIsa isa, const ExpertShape& shape, const std::uint8_t* record, const ExpertScales& scales,
                      const CpuW4A8Slice* const* x, int ncols, int unit_begin, int unit_end, std::uint16_t* const* h,
                      int prefetch_bytes = kDefaultPrefetchBytes);

// Phase B for down row groups [rg_begin, rg_end): `h[col]` is column col's h quantized (shape.intermediate / 32
// slices); writes y[col][row] in BF16 for those rows.
void down_rows_a8(CpuIsa isa, const ExpertShape& shape, const std::uint8_t* record, const ExpertScales& scales,
                  const CpuW4A8Slice* const* h, int ncols, int rg_begin, int rg_end, std::uint16_t* const* y,
                  int prefetch_bytes = kDefaultPrefetchBytes);

// Whole expert on one thread: y[col][0..H) from x[col][0..H), both BF16, H = shape.hidden, ncols <= kMaxColumns.
void expert_forward_a8(CpuIsa isa, const ExpertShape& shape, const std::uint8_t* record, const ExpertScales& scales,
                       int ncols, const std::uint16_t* const* x, std::uint16_t* const* y,
                       int prefetch_bytes = kDefaultPrefetchBytes);

} // namespace infernix::ops::offloaded_moe
