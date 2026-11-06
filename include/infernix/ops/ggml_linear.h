#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "infernix/ops/linear.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace infernix::ops {

/**
 * How a matrix's GGUF blocks are arranged (docs/maintainer/storage-layouts.md):
 * - Rows (ggml_rows_v1): row r at r * row_bytes, its blocks in order.
 * - Tiles16 (ggml_tiles16_v1; formats of 256-element blocks, ggml_tiles16_supported): rows in tiles
 *   of 16, tile i at 16 i row_bytes and block b of its row r (0-15) at (16 b + r) * block_bytes, the
 *   rows padded to a whole tile with zero blocks: the same bytes, each tile's block b one contiguous
 *   run that the narrow kernels copy in 16-byte units.
 */
enum class GgmlLayout : std::uint8_t { Rows, Tiles16 };

[[nodiscard]] constexpr bool ggml_tiles16_supported(QType format) noexcept {
    return format == QType::GGML_Q4_K || format == QType::GGML_Q5_K || format == QType::GGML_Q6_K ||
           format == QType::GGML_IQ4_XS;
}

/**
 * One matrix of a GGUF (ggml) projection: `rows` rows of `format` blocks (every row `k` elements,
 * arranged as `layout` says) whose outputs are rows [out_offset, out_offset + rows) of output
 * `output` of the projection (its only output for the one-output form).
 */
struct GgmlSegment {
    QType format              = QType::GGML_Q6_K;
    const std::uint8_t* data  = nullptr; // device-visible
    std::int32_t rows         = 0;
    std::int32_t out_offset   = 0;
    std::int32_t output       = 0;
    GgmlLayout layout         = GgmlLayout::Rows;
};

/** Bytes of one `k`-element row of `format` blocks. */
[[nodiscard]] std::size_t ggml_row_bytes(QType format, std::int32_t k);

/**
 * The segments of a projection whose rows [0, N) are split among several outputs: output i takes
 * rows [bounds[i], bounds[i + 1]) (bounds ascending, bounds[0] = 0). Each segment (its out_offset
 * within [0, N)) is cut at the bounds into segments of whole rows, each with its output index and
 * its offset within that output; a Tiles16 segment only at multiples of 16 of its rows.
 */
[[nodiscard]] std::vector<GgmlSegment> ggml_split_segments(std::span<const GgmlSegment> segments, std::int32_t k,
                                                           std::span<const std::int32_t> bounds);

/**
 * Exact projection over GGUF block weights (docs/maintainer/qwen3_8-flash-next-design.md §16.2.2).
 *
 * out[n, t] = round(sum_k w[n, k] * x[perm(k), t]) where w are the stored weights' exact values,
 * each column x[:, t] is first encoded as A16 integers (canon::a16_value: exact for every element
 * within 2^13 of the column's largest, rounded at 2^-21 of it otherwise), the sum is exact
 * (ops/ggml/ggml_exact.h: int128) and the result is rounded once to binary32 (FP32 `out`) and,
 * for a BF16 `out`, once more to BF16. A column's output depends on nothing but its own values:
 * every T, segment split and schedule returns the same bits.
 *
 * `x` is contiguous BF16 [K, T] (K a multiple of every segment's block), `out` contiguous BF16 or
 * FP32 [N, T] with N covering every segment's rows. `input_perm`, when non-null, is a device array
 * of K / perm_block entries: input block j is x's block input_perm[j] (Qwen4Exp's GDN output
 * projection reads its value heads in ggml's tiled order). The workspace holds the encoded columns
 * (ggml_linear_workspace_bytes: 3.125 bytes per element, in batches of at most 512 columns) and,
 * for calls of 32 or more columns, 64 MiB of decoded weight planes.
 *
 * `routes` with LinearRoutes::GgmlBf16Wide (the ggml-bf16-prefill fast path) runs calls of 32 or
 * more columns on BF16 tensor cores instead: x as stored (no A16 encoding), each code exactly as
 * BF16, the FP32 sum of each 16- or 32-element sub-block's products scaled by d * sub (and the
 * K-quants' - dmin * min * sum(x)) and added to an FP32 accumulator in K order, one rounding to the
 * output. Every product is exact and only the FP32 sums round; an output still depends only on its
 * row, column and K, but its bits are not the exact route's. Narrower calls and other flags run the
 * exact route.
 *
 * `routes` with LinearRoutes::GgmlA8 (a8-activations: LOSSY, opt-in, never on the reproducible route;
 * docs/maintainer/qwen3_8-flash-next-design.md §19.3.31) runs calls of 1-31 columns on 8-bit activations
 * instead: each column (input element j at x[perm(j)]) is quantized per 32-element slice by
 * ops/ggml/ggml_a8.h's codec (codes q = rne(v * fl32(127 / m)) in [-127, 127] and d = fl32(m / 127), m
 * the slice's largest magnitude; a slice below 2^-120 is zeros, one holding Inf or NaN makes its column's
 * outputs NaN), each slice's dot with a row's codes and sub-scales is an exact integer, and its terms
 * fl32(fl32(dot * d) * w) (the K-quants also - fl32(fl32(min * sum q) * d) * dmin) add into an FP32
 * accumulator, rounded once to the output (times the format's power-of-two kappa). The order of that sum
 * is fixed by the call's shape: a call's bits are deterministic and a column's do not depend on the other
 * columns' values, but they may change with the call's width, its rows or the device. Calls of 32 or more
 * columns keep the routes above.
 *
 * `routes` with LinearRoutes::GgmlA8Wide (a8-activations at prefill widths: LOSSY, opt-in) runs calls of 32
 * or more columns (K a multiple of 64) on the same 8-bit activations, taking precedence over GgmlBf16Wide:
 * with P_s the exact integer dot of a row's slice-s codes with the column's codes and w_s = d * sub (the
 * slice's block scale times its sub-scale; one per 16 elements for Q6_K and IQ2_S, each half its own term),
 * an output is acc = fma(fl32(P_s * w_s), d_s, acc) over s in K order from +0, the K-quants adding
 * fma(-(dmin * min_s), fl32(d_s * sum q), acc) after each slice's term, then fl32(acc * kappa) rounded once
 * more to a BF16 output. An output depends only on its row, its column and K: its bits are the same at every
 * width of 32 or more and for any other columns or segment split, but they are not the GgmlA8 route's (a
 * different rounding of each term and order of the sum). A non-finite weight scale gives its rows NaN.
 */
void ggml_linear(const Tensor& x, std::span<const GgmlSegment> segments, Tensor& out, const std::int32_t* input_perm,
                 std::int32_t perm_block, WorkspaceArena& workspace, cudaStream_t stream, LinearRoutes routes);

/**
 * ggml_linear into several outputs (one encoding of x): segment s writes rows [s.out_offset,
 * s.out_offset + s.rows) of *outs[s.output]. Every output is contiguous [N_i, T] of one dtype (BF16
 * or FP32), so a fused projection's parts land in their own tensors with no split copy.
 */
void ggml_linear(const Tensor& x, std::span<const GgmlSegment> segments, std::span<Tensor* const> outs,
                 const std::int32_t* input_perm, std::int32_t perm_block, WorkspaceArena& workspace, cudaStream_t stream,
                 LinearRoutes routes);

/**
 * x as ggml_linear's exact route encodes it, written by x's producer (infernix/ops/ggml_encode.h) instead of by
 * ggml_linear: `words` I32 [ggml_encoded_column_words(k), T] holds each column's A16 limbs and slice sums
 * (ops/ggml/ggml_dp4a.cuh) in input order, the projection's input permutation applied, and `emax` I32 [T] the
 * columns' exponents.
 */
struct GgmlEncodedColumns {
    Tensor words;
    Tensor emax;
    std::int32_t k = 0;
};

[[nodiscard]] std::int32_t ggml_encoded_column_words(std::int32_t k);

/** Whether ggml_linear runs a call of `columns` columns of K = k under `routes` from one A16 encoding of x (no wide
 * tiles, no 8-bit route, at most 512 columns): the calls ggml_linear_encoded takes. */
[[nodiscard]] bool ggml_linear_takes_encoded(std::int32_t k, std::int32_t columns, LinearRoutes routes);

/** ggml_linear from encoded columns, for a call ggml_linear_takes_encoded accepts: the same bits as ggml_linear on
 * the x they encode (with its permutation). No workspace. */
void ggml_linear_encoded(const GgmlEncodedColumns& x, std::span<const GgmlSegment> segments,
                         std::span<Tensor* const> outs, cudaStream_t stream);

/** The workspace of a call of `columns` columns of K = k on every route (the GgmlA8 route's encoding of a
 * narrow call, about 1.25 bytes per element, and the GgmlA8Wide route's planes and encoded batches of up to
 * 2,048 columns included). */
[[nodiscard]] std::size_t ggml_linear_workspace_bytes(std::int32_t k, std::int32_t columns);

} // namespace infernix::ops
