#include "infernix/ops/lossless_linear.h"

#include "ops/lossless/linear_impl.cuh"

#include <cmath>
#include <cstring>
#include <limits>
#include <span>
#include <vector>

// The --lossless route's projections: the API and the non-GGUF formats' instantiations
// (ops/lossless/linear_impl.cuh holds the arithmetic).

namespace infernix::ops {

using namespace lossless_linear_impl;

namespace {

// A BF16 weight's per-row block data (bf16_weight_scan_kernel) for rows of a call: its bytes, and the scan.
std::size_t bf16_meta_bytes(std::int32_t k, std::int32_t rows) {
    const std::size_t blocks = static_cast<std::size_t>(k) / kBlockK, r = static_cast<std::size_t>(rows);
    return align256(r * blocks * sizeof(std::int16_t)) + align256((r + kTileRows - 1) / kTileRows * blocks) +
           align256(r * sizeof(std::int32_t));
}

std::int32_t scanned_rows(const Call& c) { return c.scan_rows > 0 ? c.scan_rows : c.rows; }

void scan_bf16(Call& c, std::uint8_t* meta, cudaStream_t stream) {
    const int k = c.x->ne[0], blocks = k / kBlockK, rows = scanned_rows(c);
    const std::size_t r = static_cast<std::size_t>(rows);
    auto* block_exp   = reinterpret_cast<std::int16_t*>(meta);
    auto* tile_lo_w   = meta + align256(r * blocks * sizeof(std::int16_t));
    auto* row_inexact = reinterpret_cast<std::int32_t*>(tile_lo_w + align256((r + kTileRows - 1) / kTileRows * blocks));
    CUDA_CHECK(cudaMemsetAsync(tile_lo_w, 0, bf16_meta_bytes(k, rows) - align256(r * blocks * sizeof(std::int16_t)),
                               stream));
    const std::size_t warps = r * blocks;
    const int threads       = 256;
    bf16_weight_scan_kernel<<<static_cast<unsigned>((warps * 32 + threads - 1) / threads), threads, 0, stream>>>(
        reinterpret_cast<const std::uint16_t*>(c.w.codes), k, c.row_begin, rows, block_exp, tile_lo_w, row_inexact);
    CUDA_CHECK(cudaGetLastError());
    c.w.block_exp   = block_exp;
    c.w.tile_lo_w   = tile_lo_w;
    c.w.row_inexact = row_inexact;
    c.w.blocks      = blocks;
    c.w.row_base    = c.row_begin;
}

bool reads_xsum(const Call& c, Route route) { return route == Route::Ggml && ggml_reads_xsum(c.qtype); }

void launch_route(const Call& c, Route route, int first, int count, std::uint8_t* base, const Layout& l,
                  cudaStream_t stream, const ChunkStage* stage) {
    switch (route) {
    case Route::Q2: launch_chunk_stage<RowSplitFormat<2>>(c, first, count, base, l, stream, stage); break;
    case Route::Q4: launch_chunk_stage<RowSplitFormat<4>>(c, first, count, base, l, stream, stage); break;
    case Route::Q5: launch_chunk_stage<RowSplitFormat<5>>(c, first, count, base, l, stream, stage); break;
    case Route::Q6: launch_chunk_stage<RowSplitFormat<6>>(c, first, count, base, l, stream, stage); break;
    case Route::Q8: launch_chunk_stage<RowSplitFormat<8>>(c, first, count, base, l, stream, stage); break;
    case Route::Nvfp4: launch_chunk_stage<Nvfp4Format>(c, first, count, base, l, stream, stage); break;
    case Route::Nvfp4Mul: launch_chunk_stage<Nvfp4MulFormat>(c, first, count, base, l, stream, stage); break;
    case Route::Fp8Row: launch_chunk_stage<Fp8RowFormat>(c, first, count, base, l, stream, stage); break;
    case Route::Fp8RowF32: launch_chunk_stage<Fp8RowF32Format>(c, first, count, base, l, stream, stage); break;
    case Route::Bf16: launch_chunk_stage<Bf16GridFormat>(c, first, count, base, l, stream, stage); break;
    case Route::Mxfp8: launch_chunk_stage<Mxfp8Format>(c, first, count, base, l, stream, stage); break;
    case Route::Rg16: launch_chunk_stage<Rg16Format>(c, first, count, base, l, stream, stage); break;
    case Route::Fp8B128: launch_chunk_stage<Fp8B128Format>(c, first, count, base, l, stream, stage); break;
    case Route::Mxfp8Rg16: launch_chunk_stage<Mxfp8Rg16Format>(c, first, count, base, l, stream, stage); break;
    case Route::Ggml:
        require(launch_ggml_k(c.qtype, c, first, count, base, l, stream, stage) ||
                    launch_ggml_i(c.qtype, c, first, count, base, l, stream, stage),
                "unsupported GGUF format");
        break;
    default: require(false, "unsupported weight format");
    }
}

struct Projection {
    Call call;
    Route route;
};

// A BF16 product narrow enough to split K over CTAs (linear_impl.cuh split_parts).
bool splits(const Call& c, Route route, std::int32_t columns) {
    return route == Route::Bf16 && (c.plane_columns > 0 || columns <= kTileTokens);
}

// The columns the encode holds when they are fewer than the product's (the planes form).
std::int32_t encoded_columns(const Call& c) { return c.plane_columns > 0 ? c.x->ne[1] : 0; }

// The workspace of projections of one input over `columns`: the encode's buffers and one bitmap for the
// widest projection (shared, the projections run in turn), then each BF16 weight's block data.
std::size_t projections_bytes(std::int32_t k, std::span<const Projection> ps, std::int32_t columns) {
    std::int32_t rows = 0;
    std::size_t meta  = 0, split = 0;
    for (const Projection& p : ps) {
        rows = std::max(rows, p.call.rows);
        if (p.route == Route::Bf16) { meta += bf16_meta_bytes(k, scanned_rows(p.call)); }
        // A plan for wider calls also covers the narrow calls it may run (at most 64 columns).
        if (p.route == Route::Bf16) {
            const std::int32_t narrow = p.call.plane_columns > 0 ? columns : std::min(columns, kTileTokens);
            split = std::max(split, split_bytes(p.call.rows, narrow, k));
        }
    }
    return layout_of(k, rows, columns, false, encoded_columns(ps[0].call)).total + meta + split;
}

// Projections of one input (the same x, columns and column maps): each chunk's activation columns are encoded
// once, then every projection's product and fallback run on that encoding.
void run(std::span<Projection> ps, WorkspaceArena& workspace, cudaStream_t stream) {
    const int k = ps[0].call.x->ne[0], columns = ps[0].call.columns;
    std::int32_t rows = 0;
    bool xsum         = false;
    for (const Projection& p : ps) {
        rows = std::max(rows, p.call.rows);
        xsum = xsum || reads_xsum(p.call, p.route);
    }
    const Layout l = layout_of(k, rows, columns, false, encoded_columns(ps[0].call));
    auto scope     = workspace.scope();
    auto* base     = static_cast<std::uint8_t*>(workspace.alloc_bytes(projections_bytes(k, ps, columns)).data);
    std::uint8_t* meta = base + l.total;
    for (Projection& p : ps) {
        if (p.route != Route::Bf16) { continue; }
        scan_bf16(p.call, meta, stream);
        meta += bf16_meta_bytes(k, scanned_rows(p.call));
    }
    // The projections run in turn: their split products share one region, after the scans'.
    for (Projection& p : ps) {
        if (splits(p.call, p.route, columns) && split_parts(p.call.rows, columns, k) > 1) { p.call.split = meta; }
    }
    for (int first = 0; first < columns; first += kChunkColumns) {
        const int count = std::min(kChunkColumns, columns - first);
        for (std::size_t i = 0; i < ps.size(); ++i) {
            const ChunkStage stage{i == 0, xsum};
            launch_route(ps[i].call, ps[i].route, first, count, base, l, stream, &stage);
        }
    }
}

void run(const Call& call, Route route, WorkspaceArena& workspace, cudaStream_t stream) {
    Projection p{call, route};
    run(std::span<Projection>(&p, 1), workspace, stream);
}

// The checked Call of one projection (lossless_linear's contract), with an activation scale.
Call projection_call(const Tensor& x, const Weight& weight, std::int32_t row_begin, std::int32_t rows, Tensor& out,
                     std::int32_t out_row, const LosslessActivationScale& scale, Route& route) {
    route = route_of(weight);
    require(route != Route::None, "unsupported weight format");
    check_x(x, weight);
    const int columns = x.ne[1];
    require(rows > 0 && row_begin >= 0 && row_begin + rows <= weight.n, "the rows must lie within the weight");
    check_out(out, columns, out_row, rows);
    // A dense rg16 matrix keeps its scales inside its units (and its multiplier in the weight).
    require(weight.qdata != nullptr &&
                (route == Route::Bf16 || route == Route::Ggml || route == Route::Rg16 || weight.scales != nullptr) &&
                ((route != Route::Q5 && route != Route::Q6) || weight.qhigh != nullptr),
            "the weight's planes are missing");
    if (route == Route::Q2 || route == Route::Q4 || route == Route::Q5 || route == Route::Q6 || route == Route::Q8) {
        require(weight.padded_shape[1] >= x.ne[0], "the row-split weight's padded K must cover K");
    }
    require(scale.divisor == 0.0F || (std::isfinite(scale.divisor) && scale.divisor > 0.0F),
            "an activation divisor is positive and finite");
    require(scale.multiplier == 0.0F || (std::isfinite(scale.multiplier) && scale.multiplier > 0.0F),
            "an activation multiplier is positive and finite");
    require(scale.divisor == 0.0F || scale.multiplier == 0.0F, "an activation scale is a divisor or a multiplier");
    Call c{&x, view_of(weight, route), row_begin, nullptr, 0, rows, &out, out_row, columns, weight.qtype};
    std::memcpy(&c.w.act_div, &scale.divisor, sizeof(c.w.act_div));
    std::memcpy(&c.w.act_mul[0], &scale.multiplier, sizeof(c.w.act_mul[0]));
    c.w.act_mul[1] = c.w.act_mul[0];
    c.act_column = scale.column_scales;
    return c;
}

} // namespace

bool lossless_linear_supports(const Weight& weight) noexcept { return route_of(weight) != Route::None; }

std::size_t lossless_linear_workspace_bytes(const Weight& weight, std::int32_t rows, std::int32_t columns) {
    require(weight.k > 0 && weight.k % kBlockK == 0 && rows > 0 && columns > 0, "K must be a positive multiple of 64");
    const bool bf16 = route_of(weight) == Route::Bf16;
    return layout_of(weight.k, rows, columns, bf16).total +
           (bf16 ? split_bytes(rows, std::min(columns, kTileTokens), weight.k) : 0);
}

std::size_t lossless_linear_shared_workspace_bytes(std::span<const LosslessProjection> projections, std::int32_t columns) {
    require(!projections.empty() && columns > 0, "at least one projection over positive columns");
    std::vector<Projection> ps;
    const std::int32_t k = projections[0].weight->k;
    for (const LosslessProjection& p : projections) {
        require(p.weight != nullptr && p.weight->k == k && k > 0 && k % kBlockK == 0 && p.rows > 0,
                "the projections share K, a positive multiple of 64");
        Call c{};
        c.rows = p.rows;
        ps.push_back(Projection{c, route_of(*p.weight)});
    }
    return projections_bytes(k, ps, columns);
}

void lossless_linear_shared(const Tensor& x, std::span<const LosslessProjection> projections, WorkspaceArena& workspace,
                            cudaStream_t stream, const LosslessActivationScale& scale) {
    require(!projections.empty(), "at least one projection");
    std::vector<Projection> ps;
    ps.reserve(projections.size());
    for (const LosslessProjection& p : projections) {
        require(p.weight != nullptr && p.out != nullptr, "each projection names its weight and output");
        Route route = Route::None;
        const Call c = projection_call(x, *p.weight, p.row_begin, p.rows, *p.out, p.out_row, scale, route);
        ps.push_back(Projection{c, route});
    }
    run(std::span<Projection>(ps), workspace, stream);
}

void lossless_linear(const Tensor& x, const Weight& weight, std::int32_t row_begin, std::int32_t rows, Tensor& out,
                     std::int32_t out_row, WorkspaceArena& workspace, cudaStream_t stream,
                     const LosslessActivationScale& scale) {
    Route route  = Route::None;
    const Call c = projection_call(x, weight, row_begin, rows, out, out_row, scale, route);
    run(c, route, workspace, stream);
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
    Call call{&x, view_of(weight, route), row_offset, static_cast<const std::int32_t*>(tile_experts.data),
              expert_stride, rows, &out, 0, columns, weight.qtype};
    // The ids are device data: a tile naming an expert whose rows would pass the weight's end is left unwritten (as
    // an unused tile) rather than read out of bounds.
    call.max_expert = expert_stride > 0 ? (weight.n - row_offset - rows) / expert_stride : std::numeric_limits<int>::max();
    run(call, route, workspace, stream);
}

std::size_t lossless_linear_planes_workspace_bytes(const Weight& weight, std::int32_t planes, std::int32_t plane_stride,
                                                   std::int32_t plane_columns, std::int32_t rows) {
    require(route_of(weight) == Route::Bf16 && planes > 0 && plane_columns > 0 && rows > 0 && weight.k % kBlockK == 0,
            "the planes form takes a BF16 weight, positive planes, columns and rows");
    return layout_of(weight.k, rows, planes * kTileTokens, false, planes * plane_columns).total +
           bf16_meta_bytes(weight.k, (planes - 1) * plane_stride + rows) + split_bytes(rows, planes * kTileTokens, weight.k);
}

void lossless_linear_planes(const Tensor& x, const Weight& weight, const Tensor& tile_planes, std::int32_t plane_stride,
                            std::int32_t plane_columns, std::int32_t rows, Tensor& out, WorkspaceArena& workspace,
                            cudaStream_t stream) {
    const Route route = route_of(weight);
    require(route == Route::Bf16, "the planes form takes a BF16 weight");
    check_x(x, weight);
    require(plane_columns > 0 && plane_columns <= kTileTokens && x.ne[1] % plane_columns == 0,
            "the planes form's x holds 1 to 64 columns a plane");
    const int planes = x.ne[1] / plane_columns;
    require(planes * kTileTokens <= kChunkColumns, "the planes form's tiles fit one column chunk");
    require(tile_planes.dtype == DType::I32 && tile_planes.is_contiguous() && tile_planes.numel() == planes,
            "tile_planes must be device I32 [planes]");
    require(plane_stride % kTileRows == 0 && rows > 0 && rows <= plane_stride &&
                static_cast<std::int64_t>(planes - 1) * plane_stride + rows <= weight.n,
            "the planes must lie within the weight, on a 32-row grid");
    check_out(out, x.ne[1], 0, rows);
    Call call{&x, view_of(weight, route), 0, static_cast<const std::int32_t*>(tile_planes.data), plane_stride, rows, &out, 0,
              planes * kTileTokens, weight.qtype};
    call.max_expert    = planes - 1;
    call.scan_rows     = (planes - 1) * plane_stride + rows;
    call.plane_columns = plane_columns;
    run(call, route, workspace, stream);
}

namespace {

Route bank_route(QType q) {
    if (q == QType::NVFP4_MUL) { return Route::Rg16; }
    if (q == QType::FP8_E4M3FN_B128) { return Route::Fp8B128; }
    if (q == QType::FP8_E4M3FN_G32_E8M0) { return Route::Mxfp8Rg16; }
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
                           const std::int32_t* out_columns, WorkspaceArena& workspace, cudaStream_t stream,
                           const LosslessActivationScale& scale, bool sparse) {
    require(scale.multiplier == 0.0F, "the tile mode's activation multipliers are its tiles' act scales");
    require(scale.divisor == 0.0F || (std::isfinite(scale.divisor) && scale.divisor > 0.0F),
            "an activation divisor is positive and finite");
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
    std::memcpy(&v.act_div, &scale.divisor, sizeof(v.act_div));
    Call c{&x, v, 0, nullptr, 0, matrix.rows, &out, 0, columns, matrix.qtype};
    c.tiles       = tiles;
    c.in_columns  = in_columns;
    c.out_columns = out_columns;
    c.act_column  = scale.column_scales;
    c.sparse_columns = sparse;
    run(c, route, workspace, stream);
}

} // namespace infernix::ops

// This unit's copy of the noise twin's seed (ops/lossless/noise.cuh).
INFERNIX_LOSSLESS_NOISE_UNIT(linear)
