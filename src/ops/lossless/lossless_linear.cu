#include "infernix/ops/lossless_linear.h"

#include "ops/lossless/linear_impl.cuh"

// The --lossless route's projections: the API and the non-GGUF formats' instantiations
// (ops/lossless/linear_impl.cuh holds the arithmetic).

namespace infernix::ops {

using namespace lossless_linear_impl;

namespace {

void run(const Call& call, Route route, WorkspaceArena& workspace, cudaStream_t stream) {
    const int k = call.x->ne[0], columns = call.columns;
    const Layout l = layout_of(k, call.rows, columns, route == Route::Bf16);
    auto scope     = workspace.scope();
    auto* base     = static_cast<std::uint8_t*>(workspace.alloc_bytes(l.total).data);
    Call c         = call;
    if (route == Route::Bf16) {
        const int blocks  = k / kBlockK;
        auto* block_exp   = reinterpret_cast<std::int16_t*>(base + l.block_exp);
        auto* tile_lo_w   = base + l.tile_lo_w;
        auto* row_inexact = reinterpret_cast<std::int32_t*>(base + l.row_inexact);
        CUDA_CHECK(cudaMemsetAsync(base + l.tile_lo_w, 0, l.total - l.tile_lo_w, stream));
        const std::size_t warps = static_cast<std::size_t>(c.rows) * blocks;
        const int threads       = 256;
        bf16_weight_scan_kernel<<<static_cast<unsigned>((warps * 32 + threads - 1) / threads), threads, 0, stream>>>(
            reinterpret_cast<const std::uint16_t*>(c.w.codes), k, c.row_begin, c.rows, block_exp, tile_lo_w, row_inexact);
        CUDA_CHECK(cudaGetLastError());
        c.w.block_exp   = block_exp;
        c.w.tile_lo_w   = tile_lo_w;
        c.w.row_inexact = row_inexact;
        c.w.blocks      = blocks;
        c.w.row_base    = c.row_begin;
    }
    for (int first = 0; first < columns; first += kChunkColumns) {
        const int count = std::min(kChunkColumns, columns - first);
        switch (route) {
        case Route::Q2: launch_chunk<RowSplitFormat<2>>(c, first, count, base, l, stream); break;
        case Route::Q4: launch_chunk<RowSplitFormat<4>>(c, first, count, base, l, stream); break;
        case Route::Q5: launch_chunk<RowSplitFormat<5>>(c, first, count, base, l, stream); break;
        case Route::Q6: launch_chunk<RowSplitFormat<6>>(c, first, count, base, l, stream); break;
        case Route::Q8: launch_chunk<RowSplitFormat<8>>(c, first, count, base, l, stream); break;
        case Route::Nvfp4: launch_chunk<Nvfp4Format>(c, first, count, base, l, stream); break;
        case Route::Fp8Row: launch_chunk<Fp8RowFormat>(c, first, count, base, l, stream); break;
        case Route::Bf16: launch_chunk<Bf16GridFormat>(c, first, count, base, l, stream); break;
        case Route::Mxfp8: launch_chunk<Mxfp8Format>(c, first, count, base, l, stream); break;
        case Route::Rg16: launch_chunk<Rg16Format>(c, first, count, base, l, stream); break;
        case Route::Fp8B128: launch_chunk<Fp8B128Format>(c, first, count, base, l, stream); break;
        case Route::Ggml:
            require(launch_ggml_k(c.qtype, c, first, count, base, l, stream) ||
                        launch_ggml_i(c.qtype, c, first, count, base, l, stream),
                    "unsupported GGUF format");
            break;
        default: require(false, "unsupported weight format");
        }
    }
}

} // namespace

bool lossless_linear_supports(const Weight& weight) noexcept { return route_of(weight) != Route::None; }

std::size_t lossless_linear_workspace_bytes(const Weight& weight, std::int32_t rows, std::int32_t columns) {
    require(weight.k > 0 && weight.k % kBlockK == 0 && rows > 0 && columns > 0, "K must be a positive multiple of 64");
    return layout_of(weight.k, rows, columns, route_of(weight) == Route::Bf16).total;
}

void lossless_linear(const Tensor& x, const Weight& weight, std::int32_t row_begin, std::int32_t rows, Tensor& out,
                     std::int32_t out_row, WorkspaceArena& workspace, cudaStream_t stream) {
    const Route route = route_of(weight);
    require(route != Route::None, "unsupported weight format");
    check_x(x, weight);
    const int columns = x.ne[1];
    require(rows > 0 && row_begin >= 0 && row_begin + rows <= weight.n, "the rows must lie within the weight");
    check_out(out, columns, out_row, rows);
    require(weight.qdata != nullptr && (route == Route::Bf16 || route == Route::Ggml || weight.scales != nullptr) &&
                ((route != Route::Q5 && route != Route::Q6) || weight.qhigh != nullptr),
            "the weight's planes are missing");
    if (route == Route::Q2 || route == Route::Q4 || route == Route::Q5 || route == Route::Q6 || route == Route::Q8) {
        require(weight.padded_shape[1] >= x.ne[0], "the row-split weight's padded K must cover K");
    }
    run(Call{&x, view_of(weight, route), row_begin, nullptr, 0, rows, &out, out_row, columns, weight.qtype}, route,
        workspace, stream);
}

void lossless_linear_grouped(const Tensor& x, const Weight& weight, const Tensor& tile_experts, std::int32_t expert_stride,
                             std::int32_t row_offset, std::int32_t rows, Tensor& out, WorkspaceArena& workspace,
                             cudaStream_t stream) {
    const Route route = route_of(weight);
    require(route != Route::None && route != Route::Bf16, "unsupported weight format for the grouped form");
    check_x(x, weight);
    const int columns = x.ne[1];
    require(columns % kTileTokens == 0, "the grouped form's columns are whole 64-column tiles");
    require(tile_experts.dtype == DType::I32 && tile_experts.is_contiguous() &&
                tile_experts.numel() == columns / kTileTokens,
            "tile_experts must be device I32 [columns / 64]");
    require(rows > 0 && expert_stride >= 0 && row_offset >= 0 && row_offset + rows <= weight.n,
            "the rows must lie within the weight");
    check_out(out, columns, 0, rows);
    require(out.ne[0] == rows, "the grouped form's out is [rows, columns]");
    run(Call{&x, view_of(weight, route), row_offset, static_cast<const std::int32_t*>(tile_experts.data),
             expert_stride, rows, &out, 0, columns, weight.qtype},
        route, workspace, stream);
}

namespace {

Route bank_route(QType q) {
    if (q == QType::NVFP4_MUL) { return Route::Rg16; }
    if (q == QType::FP8_E4M3FN_B128) { return Route::Fp8B128; }
    return is_ggml_format(q) ? Route::Ggml : Route::None;
}

} // namespace

std::size_t lossless_linear_tiles_workspace_bytes(const LosslessBankMatrix& matrix, std::int32_t columns) {
    require(matrix.k > 0 && matrix.k % kBlockK == 0 && matrix.rows > 0 && columns > 0,
            "K must be a positive multiple of 64");
    return layout_of(matrix.k, matrix.rows, columns, false).total;
}

void lossless_linear_tiles(const Tensor& x, const std::int32_t* in_columns, std::int32_t columns,
                           const LosslessBankMatrix& matrix, const LosslessTile* tiles, Tensor& out,
                           const std::int32_t* out_columns, WorkspaceArena& workspace, cudaStream_t stream) {
    const Route route = bank_route(matrix.qtype);
    require(route != Route::None, "unsupported expert-record format");
    require(x.dtype == DType::BF16 && x.ne[2] == 1 && x.ne[3] == 1 && x.nb[0] == 2 &&
                x.nb[1] == static_cast<std::int64_t>(x.ne[0]) * 2 && x.ne[0] == matrix.k && matrix.k % kBlockK == 0,
            "x must be contiguous BF16 [K, *] with K the matrix's (a multiple of 64)");
    require(columns > 0 && columns % kTileTokens == 0 && tiles != nullptr && (in_columns != nullptr || x.ne[1] >= columns),
            "the tile mode's columns are whole 64-column tiles, each with its record");
    require((out.dtype == DType::BF16 || out.dtype == DType::FP32) && out.ne[0] == matrix.rows &&
                out.nb[1] == static_cast<std::int64_t>(out.ne[0] * dtype_size(out.dtype)) &&
                (out_columns != nullptr || out.ne[1] >= columns),
            "out must be contiguous BF16 or FP32 [rows, *]");
    require(route != Route::Ggml || matrix.row_bytes > 0, "a GGUF matrix needs its row bytes");
    WeightView v{};
    v.codes     = reinterpret_cast<const std::uint8_t*>(static_cast<std::uintptr_t>(matrix.codes_offset));
    v.scales    = reinterpret_cast<const void*>(static_cast<std::uintptr_t>(matrix.scales_offset));
    v.k         = matrix.k;
    v.row_bytes = matrix.row_bytes;
    v.groups    = matrix.k / 16;
    Call c{&x, v, 0, nullptr, 0, matrix.rows, &out, 0, columns, matrix.qtype};
    c.tiles       = tiles;
    c.in_columns  = in_columns;
    c.out_columns = out_columns;
    run(c, route, workspace, stream);
}

} // namespace infernix::ops
