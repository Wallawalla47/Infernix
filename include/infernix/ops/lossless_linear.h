#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <span>

namespace infernix::ops {

/**
 * Op: lossless_linear (the --lossless route's projections; docs/maintainer/model-quality.md §6.2)
 *
 * out[out_row + r, t] = round(sum_k w[row_begin + r, k] * x[k, t]) for r in [0, rows): the weights'
 * exact values (the format's decode as a real number), x as stored, the sum exact and rounded once to
 * out's dtype (BF16 or FP32, round to nearest, ties to even). The result is a function of the operands
 * alone: it does not depend on T, the column, the row range, the launch or the build.
 *
 * x is contiguous BF16 [K, T] (K a multiple of 64); out is contiguous [N_out, T] with
 * out_row + rows <= N_out. Weight formats (docs/maintainer/tensor-formats.md), each weight its format's
 * exact value: Q2, Q4, Q5 and Q6 row-split with FP16 scales per 64 and Q8 per 32 (code times scale);
 * nvfp4 (E2M1 code times its 16-element group's E4M3 scale, divided by the matrix's FP32 divisor: the
 * sum is divided exactly); fp8_e4m3fn_row_bf16 (E4M3 code times the row's BF16 scale) and
 * fp8_e4m3fn_row_f32 (times the row's FP32 multiplier); nvfp4_rg16_v1 (a dense nvfp4_mul matrix in the
 * expert record's layout: E2M1 code times its block's E4M3 scale, the sum times the FP32 multiplier);
 * fp8_e4m3fn_g32_e8m0 (E4M3 code times its 32-element group's power of two); the GGUF blocks of
 * ggml_rows_v1 and ggml_tiles16_v1 (Q2_0, Q4_K, Q5_K, Q6_K, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS, IQ4_NL:
 * kappa (d sub code - dmin min_sub)); BF16.
 *
 * Arithmetic. Each 64-element block of a column is put on an integer grid of 29 bits relative to its
 * largest exponent (exact for every element within 21 binades of it; elements 22 to 43 binades below go
 * on a second grid 22 binades lower, multiplied only where a tile holds them); its four byte limbs meet the
 * weights' integer codes (FP8: three 7-bit digit planes) on int8 tensor cores, each scale group's sum
 * is exact in int64, scaled exactly and accumulated in int128 relative to the column's largest block
 * and the row's largest scale. An output whose sum lost any bit (a block spanning more than 43 binades,
 * a term more than 64 binades below the largest possible one, a non-finite input) is recomputed with a
 * wide exact accumulator. BF16 weights take the same route, each 64-element block of a weight row on
 * its own grid. Either way the output is the correctly rounded exact sum.
 *
 * x, out and the workspace must not overlap; the workspace is lossless_linear_workspace_bytes(weight,
 * rows, T) (columns are processed 512 at a time, so it does not grow beyond T = 512; BF16 weights add a
 * scan of the rows).
 */
[[nodiscard]] std::size_t lossless_linear_workspace_bytes(const Weight& weight, std::int32_t rows,
                                                          std::int32_t columns);

/**
 * A quantized activation's scale (the design reference, docs/maintainer/lossless-route.md §3b): x holds
 * the quantized values without it (each exact in BF16: an NVFP4 code times its block scale, an E4M3
 * value), and the scale applies to the exact sum together with the weight's own factor before the one
 * rounding: out = round(sum_k w x / divisor * column_scales[t]). divisor 0 and a null column_scales
 * (FP32 [T], x's columns) are none; with neither, the product is lossless_linear's.
 */
struct LosslessActivationScale {
    float divisor              = 0.0F; // the NVFP4 activation divisor D (positive finite)
    const float* column_scales = nullptr;
};

void lossless_linear(const Tensor& x, const Weight& weight, std::int32_t row_begin, std::int32_t rows, Tensor& out,
                     std::int32_t out_row, WorkspaceArena& workspace, cudaStream_t stream,
                     const LosslessActivationScale& scale = {});

/// Several projections of one input (each exactly lossless_linear's result): rows [row_begin, row_begin +
/// rows) of `weight` into rows [out_row, ...) of `out`. They share the input's encode (one per 512-column
/// chunk), so a layer's projections of one normalized input launch fewer kernels. The weights share K.
struct LosslessProjection {
    const Weight* weight   = nullptr;
    std::int32_t row_begin = 0;
    std::int32_t rows      = 0;
    Tensor* out            = nullptr;
    std::int32_t out_row   = 0;
};
[[nodiscard]] std::size_t lossless_linear_shared_workspace_bytes(std::span<const LosslessProjection> projections,
                                                                 std::int32_t columns);
void lossless_linear_shared(const Tensor& x, std::span<const LosslessProjection> projections, WorkspaceArena& workspace,
                            cudaStream_t stream, const LosslessActivationScale& scale = {});

/**
 * The grouped form (routed experts in a bank): x's columns are whole 64-column tiles, tile i multiplied
 * by the weight rows [e expert_stride + row_offset, ... + rows) for its expert e = tile_experts[i]
 * (device I32 [T / 64]; a negative entry skips the tile and leaves its outputs unwritten). out is
 * contiguous [rows, T]. Every output as lossless_linear's; not for BF16 weights. The workspace is
 * lossless_linear_workspace_bytes(weight, rows, T).
 */
void lossless_linear_grouped(const Tensor& x, const Weight& weight, const Tensor& tile_experts,
                             std::int32_t expert_stride, std::int32_t row_offset, std::int32_t rows, Tensor& out,
                             WorkspaceArena& workspace, cudaStream_t stream);

/** One 64-column tile of lossless_linear_tiles: its expert's record (null: the tile is skipped), the
 * matrix's FP32 multipliers of even and odd rows (NVFP4 records; bits), and the design reference's
 * activation scales of even and odd rows (the expert's FP32 input scales g, applied to the exact sum
 * like LosslessActivationScale; bits, 0: none). */
struct LosslessTile {
    const std::uint8_t* record;
    std::uint32_t alpha[2];
    std::uint32_t act[2] = {0U, 0U};
};

/** A matrix inside routed-expert records: its format (NVFP4_MUL in nvfp4_expert_rg16_v1 units,
 * FP8_E4M3FN_B128 rows with BF16 128 x 128 tile scales, or GGUF rows), where its codes and scales lie
 * in a record, its rows and K, and (GGUF) the bytes of one row. */
struct LosslessBankMatrix {
    QType qtype                 = QType::NVFP4_MUL;
    std::uint64_t codes_offset  = 0;
    std::uint64_t scales_offset = 0;
    std::int32_t rows           = 0;
    std::int32_t k              = 0;
    std::int32_t row_bytes      = 0;
};

/**
 * The tile mode (routed experts in offloaded records): out[:, out_columns[c]] = matrix x[:, in_columns[c]]
 * for c in [0, columns) (columns a multiple of 64), column c's weights from tiles[c / 64]'s record. A
 * null map is the identity; a negative in_columns entry reads a zero column, a negative out_columns
 * entry writes nothing. Every output as lossless_linear's (NVFP4 records: the sum times the row
 * parity's multiplier, exactly; a tile's act scales multiply it too). x is contiguous BF16 [K, *]; out
 * contiguous BF16 or FP32 [rows, *].
 * The workspace is lossless_linear_tiles_workspace_bytes(matrix, columns).
 */
[[nodiscard]] std::size_t lossless_linear_tiles_workspace_bytes(const LosslessBankMatrix& matrix, std::int32_t columns);

void lossless_linear_tiles(const Tensor& x, const std::int32_t* in_columns, std::int32_t columns,
                           const LosslessBankMatrix& matrix, const LosslessTile* tiles, Tensor& out,
                           const std::int32_t* out_columns, WorkspaceArena& workspace, cudaStream_t stream);

/** Whether lossless_linear takes this weight's format. */
[[nodiscard]] bool lossless_linear_supports(const Weight& weight) noexcept;

} // namespace infernix::ops
