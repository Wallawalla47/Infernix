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

[[nodiscard]] std::size_t ggml_linear_workspace_bytes(std::int32_t k, std::int32_t columns);

} // namespace infernix::ops
