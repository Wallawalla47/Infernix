#pragma once

// CPU route of offloaded_sparse_moe for GGUF (ggml) expert banks (docs/maintainer/
// qwen3_8-flash-next-design.md §16.2.2, §19.3.25): one routed expert computed in place from its host record
// (gate rows, up rows, down rows, each verbatim GGUF blocks) with the exact arithmetic of
// ops/ggml/ggml_exact.h. Every output is the exact int128 sum rounded once, so every ISA variant
// returns the bits of the GPU kernels.
//
// A column's A16 integers X (|X| < 2^21) are stored per 32-element slice as two signed 16-bit
// planes, X = hi * 2^11 + lo with lo in [0, 2047] and hi in [-1024, 1023]. A weight slice's codes
// times its sub-scales fit 16 bits (|code * sub| <= 4064), so 16-bit multiply-adds (vpmaddwd,
// vpdpwssd) give the exact sums sum sub * code * X = 2^11 sum w * hi + sum w * lo.
//
// The expert formats are IQ2_S, IQ3_XXS, IQ3_S and IQ4_XS (gate/up) and IQ4_NL and Q2_0 (down);
// none has a min term.

#include "core/weight.h"
#include "ops/ggml/ggml_exact.h"
#include "ops/offloaded_sparse_moe/cpu/w4a4_expert.h"

#include <cstdint>

namespace infernix::ops::offloaded_moe {

// A ggml bank's record geometry (MoeGgmlBank's formats and row sizes, host side).
struct GgmlExpertFormat {
    QType gate_up                = QType::NVFP4_MUL; // NVFP4_MUL: not a ggml bank
    QType down                   = QType::NVFP4_MUL;
    std::uint32_t gate_row_bytes = 0;
    std::uint32_t down_row_bytes = 0;

    [[nodiscard]] bool valid() const noexcept { return gate_up != QType::NVFP4_MUL; }
};

// One encoded 32-element slice of a column.
struct CpuLimbSlice {
    alignas(32) std::int16_t lo[32]; // X & 2047
    std::int16_t hi[32];             // X >> 11
};

inline constexpr int kGgmlXSlices = kHidden / 32;       // 80
inline constexpr int kGgmlHSlices = kIntermediate / 32; // 20

// Whether this CPU route computes a format pair (the GPU's launch_pass_ggml pairs).
[[nodiscard]] bool ggml_cpu_supported(const GgmlExpertFormat& format) noexcept;

// Encodes BF16 v[0..n) (n a multiple of 32) under its column exponent, which it returns
// (canon::kA16NonFinite for a non-finite column: its X are zero and its outputs NaN). AVX2 when the CPU
// has it; the integers are the same on every CPU.
int ggml_encode_column(const std::uint16_t* v, int n, CpuLimbSlice* out);

// Exact R of rows [row_begin, row_end) of a matrix of `format` rows (row_bytes apart, `slices`
// 32-element slices each) against `ncols` encoded columns: r[(row - row_begin) * ncols + c].
void ggml_row_sums(CpuIsa isa, QType format, const std::uint8_t* matrix, int row_bytes, int slices, int row_begin,
                   int row_end, const CpuLimbSlice* const* columns, int ncols, ggml::I128* r);

// Phase A for intermediates [i_begin, i_end): gate and up rows, SwiGLU; writes h[col][i] in BF16.
void ggml_gate_up(CpuIsa isa, const std::uint8_t* record, const GgmlExpertFormat& format,
                  const CpuLimbSlice* const* x, const int* x_emax, int ncols, int i_begin, int i_end,
                  std::uint16_t* const* h);

// Phase B for down rows [row_begin, row_end): writes y[col][row] in BF16.
void ggml_down(CpuIsa isa, const std::uint8_t* record, const GgmlExpertFormat& format, const CpuLimbSlice* const* h,
               const int* h_emax, int ncols, int row_begin, int row_end, std::uint16_t* const* y);

// Whole expert on one thread: y[col][0..kHidden) from BF16 x[col][0..kHidden), ncols <= kMaxColumns.
void ggml_expert_forward(CpuIsa isa, const std::uint8_t* record, const GgmlExpertFormat& format, int ncols,
                         const std::uint16_t* const* x, std::uint16_t* const* y);

// ------------------------------------------------------------------- the 8-bit activation route
//
// `--a8-activations` (LOSSY, opt-in; design §19.3.31; ops/ggml/ggml_a8.h): x and h quantized per 32-element
// slice to codes in [-127, 127] with a binary32 scale, each slice's dot with the weight codes an exact int32
// isum, and each row's slice terms a8::add_term(acc, isum, d_x, d_w) added in the GPU expert kernels' order
// (cuda/ggml_expert_a8.cuh): slice s to partial s % P from 0.0f in ascending s, then the butterfly
// p_i = p_i + p_(i + o), o = P / 2 .. 1, with P = a8::kGateUpPartials (gate/up rows) or a8::kDownPartials
// (down rows). Every ISA forms the same IEEE binary32 operations in that order (the vector routes keep one
// partial per lane), so a CPU-served miss returns the GPU's bits. The x86 routes need the default MXCSR
// (no flush-to-zero or denormals-are-zero), as the scalar canonical arithmetic does. GGUF pairs only (not
// block-FP8 banks).

// One quantized 32-element slice: its codes (int16, which the 16-bit multiply-adds read directly), scale
// (0 for a slice whose largest magnitude is below 2^-120, NaN for one holding Inf or NaN) and code sum.
struct CpuA8Slice {
    alignas(32) std::int16_t q[32];
    float d;
    std::int32_t sum;
};

// Quantizes BF16 v[0..n) (n a multiple of 32) slice by slice (the GPU's a8_quantize_slice).
void ggml_a8_encode_column(const std::uint16_t* v, int n, CpuA8Slice* out);

// Phase A for intermediates [i_begin, i_end): gate and up rows, SwiGLU; writes h[col][i] in BF16.
void ggml_a8_gate_up(CpuIsa isa, const std::uint8_t* record, const GgmlExpertFormat& format, const CpuA8Slice* const* x,
                     int ncols, int i_begin, int i_end, std::uint16_t* const* h);

// Phase B for down rows [row_begin, row_end) against h quantized by ggml_a8_encode_column: writes
// y[col][row] in BF16.
void ggml_a8_down(CpuIsa isa, const std::uint8_t* record, const GgmlExpertFormat& format, const CpuA8Slice* const* h,
                  int ncols, int row_begin, int row_end, std::uint16_t* const* y);

// Whole expert on one thread on the 8-bit route: y[col][0..kHidden) from BF16 x[col][0..kHidden),
// ncols <= kMaxColumns.
void ggml_a8_expert_forward(CpuIsa isa, const std::uint8_t* record, const GgmlExpertFormat& format, int ncols,
                            const std::uint16_t* const* x, std::uint16_t* const* y);

} // namespace infernix::ops::offloaded_moe
