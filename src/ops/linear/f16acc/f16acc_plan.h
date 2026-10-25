#pragma once

// f16-accumulate-prefill (LinearRoutes::F16Accumulate): the E4M3-weight projections (row-scaled FP8 and
// MXFP8) on FP16 Tensor Cores with FP16 accumulation, which runs at twice the FP32-accumulation rate on
// the RTX 5090 (m16n8k16: about 505 against 254 TFLOP/s).
//
// Operands. Every E4M3 code is an FP16 value (E4M3's 4 significant bits and exponents 2^-9..2^8 lie
// inside FP16's normal range), so the weights enter the MMA exactly; a row's BF16 or FP32 scale (FP8) or
// a 32-value group's UE8M0 power of two (MXFP8) stays outside, in FP32. A BF16 activation column x[:,t]
// enters as x16 = fp16(x * 2^s_t) with the power of two s_t = -floor(log2 max_k |x[k,t]|), so the
// column's maximum lands in [1, 2) and inverse[t] = 2^-s_t (f16acc_encode.cu). A BF16 value has 8
// significant bits and FP16 11, so the encoding is exact for every value whose scaled magnitude is at
// least 2^-17 (within 2^17 of the column maximum); a smaller value lands on FP16's subnormal grid
// (absolute error at most 2^-25 of 2^s_t, so at most 2^-25 of the column maximum), and one more than
// about 2^25 below the maximum flushes to zero.
//
// Accumulation. Each output sums the products of one 32-value K chunk in FP16 (two k16 MMAs, the first
// from a zero accumulator), then adds the chunk's FP16 sum into an FP32 carry (times the group's power
// of two for MXFP8, whose groups are the chunks). Overflow: |code| <= 448 and |x16| < 2, so a chunk's
// partial sums stay below 32 * 448 * 2 = 28,672, under FP16's 65,504 for every input. Rounding: each
// k16 step rounds the chunk's running sum to FP16 (11 significant bits), so a chunk carries a relative
// error of about 2^-11 of its own magnitude; over K / 32 chunks that adds roughly 2^-11 of the output's
// magnitude in RMS, a quarter of the BF16 output rounding (2^-9) the exact route already has.
//
// The epilogue multiplies each carry by inverse[t] (exact) and the row's scale (FP8; MXFP8 has none
// left), then rounds to BF16 as the exact route does. A column's result depends only on that column, so
// the route is column-invariant at every width it serves; it differs from the exact route's bits, so a
// projection takes it only at its registered admission width and above (prefill calls).

#include "core/arena.h"
#include "core/layout.h"
#include "core/tensor.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace infernix::ops::detail {

// The K values one FP16 partial sum spans before its FP32 carry (two k16 MMAs; one MXFP8 group).
inline constexpr int kF16AccumulateChunk = 32;

// A call's encoded activation: x16 [K, T] (column-major like x) and each column's 2^-s_t.
struct F16AccumulateWorkspace {
    __half* x       = nullptr;
    float* inverses = nullptr;
};

inline std::size_t f16acc_checked_bytes(std::int32_t tokens, std::size_t bytes_per_token) {
    if (tokens <= 0) { throw std::invalid_argument("f16 accumulate workspace: T must be positive"); }
    const auto count = static_cast<std::size_t>(tokens);
    if (count > std::numeric_limits<std::size_t>::max() / bytes_per_token) {
        throw std::overflow_error("f16 accumulate workspace size overflow");
    }
    return count * bytes_per_token;
}

template <class Arena>
F16AccumulateWorkspace allocate_f16_accumulate_workspace(Arena& arena, std::int32_t tokens,
                                                         std::int32_t input_rows) {
    if (input_rows <= 0 || (input_rows % kF16AccumulateChunk) != 0) {
        throw std::invalid_argument("f16 accumulate workspace: invalid K");
    }
    const DeviceSpan x = arena.alloc_bytes(
        f16acc_checked_bytes(tokens, static_cast<std::size_t>(input_rows) * sizeof(__half)), 256);
    const DeviceSpan inverses = arena.alloc_bytes(f16acc_checked_bytes(tokens, sizeof(float)), 256);
    return {static_cast<__half*>(x.data), static_cast<float*>(inverses.data)};
}

inline std::size_t f16_accumulate_workspace_capacity_bytes(std::int32_t tokens, std::int32_t input_rows) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_f16_accumulate_workspace(layout, tokens, input_rows);
    return layout.peak_bytes(1);
}

// Writes x16 and the column inverses of every column of x (BF16 [K, T], K in {640, 2560, 6144}), each 16-value
// block in the ldmatrix B route's order (f16acc_encode.cu, f16acc_mma.cuh).
void launch_f16_accumulate_encode(const Tensor& x, F16AccumulateWorkspace workspace, cudaStream_t stream);

// What the FP16 encoding does to one projection input over the calls a scan has seen
// (launch_f16_accumulate_scan accumulates into it on the device; zero-initialize it first). Every
// magnitude is stored as its FP32 bits, which order like the non-negative values.
struct F16AccumulateScanRecord {
    unsigned max_bits             = 0;          // max |x| over every value
    unsigned min_column_max_bits  = 0x7F800000; // the smallest nonzero column maximum
    unsigned max_range_bits       = 0;          // max over columns of column max / column min nonzero |x|
    unsigned max_chunk_bits       = 0;          // max over columns and 32-value chunks of sum |x16| (scaled)
    unsigned long long columns    = 0;
    unsigned long long values     = 0;
    unsigned long long nonzero    = 0;
    unsigned long long inexact    = 0; // nonzero values the FP16 encoding does not represent exactly
    unsigned long long flushed    = 0; // nonzero values it encodes as zero
};

// Scans every column of x (BF16 [K, T], K a multiple of 32) the way launch_f16_accumulate_encode would
// encode it and folds the result into *record (device memory).
void launch_f16_accumulate_scan(const Tensor& x, F16AccumulateScanRecord* record, cudaStream_t stream);

} // namespace infernix::ops::detail
