// ops::ggml_linear's wide 8-bit activation route (LinearRoutes::GgmlA8Wide, `--a8-activations` at prefill widths:
// LOSSY, opt-in; docs/maintainer/qwen3_8-flash-next-design.md §19.3.31): calls of 32 or more columns over GGUF block
// weights on int8 tensor cores. The arithmetic and the tiles are ops/ggml/ggml_a8_tile.cuh's (compute_plane).
//
// A call runs the exact and BF16 wide routes' structure (ggml_linear.cu): the segments' rows in slabs whose decoded
// planes fit kA8WidePlaneBytes, grouped into passes; per pass every slab's plane is decoded once (plane_kernel:
// PlaneStage images, 5,632 bytes per 64 rows x 64 elements, the scales already in binary32), then for each batch of
// up to kA8WideBatchColumns columns the batch is encoded once (encode_kernel: ggml_a8.h's codec, the narrow route's
// codes and scales, and for the K-quants fl32(d * sum q)) and every slab's tiles run (tile_kernel: 128 rows x 128
// columns a CTA, the column tiles fastest in the grid). An output's bits depend only on its row, its column and K
// (ggml_a8_tile.cuh), so the slabs, passes and batches change no bit.

#include "ops/ggml/ggml_linear_a8.h"

#include "core/device.h"
#include "core/ggml_format.h"
#include "ops/common/canonical_math.h"
#include "ops/ggml/ggml_a8.h"
#include "ops/ggml/ggml_a8_tile.cuh"
#include "ops/ggml/ggml_dp4a.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace infernix::ops {
namespace a8_wide {
namespace {

namespace a8    = ggml::a8;
namespace wt    = ggml::a8::wide;
namespace fast  = ggml::fast;
namespace canon = ops::canon;

constexpr int kEncodeThreads = 256;

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string("ggml_linear (a8 wide): ") + message); }
}

void check_launch(const char* what) {
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) { throw std::runtime_error(std::string("ggml_linear a8 wide ") + what + ": " + cudaGetErrorString(error)); }
}

// Words between encoded columns: a8::column_words rounded up to 16 bytes.
constexpr int column_stride(int k, bool sums) { return static_cast<int>((a8::column_words(k, sums) + 3) / 4 * 4); }

bool min_format(QType q) { return q == QType::GGML_Q4_K || q == QType::GGML_Q5_K; }

// ------------------------------------------------------------------------------------------------- encode

// Four threads per 32-element slice (slice s of column t: items t * K / 32 + s), thread c taking elements
// 8c .. 8c + 7 (input element j at x[perm(j)]; one 16-byte load where `vec`); the slice's maximum and code sum are
// the quad's. Thread c writes code words 2c and 2c + 1, thread 0 the scale d and, with `sums`, fl32(d * sum q):
// the narrow route's codes and scales (ggml_linear_a8.cu's encoder), columns `stride` words apart.
__global__ void __launch_bounds__(kEncodeThreads)
    encode_kernel(const std::uint16_t* __restrict__ x, int k, const std::int32_t* __restrict__ perm, int perm_block,
                  std::int32_t* __restrict__ encoded, int stride, int columns, int sums, int vec) {
    const int slices = k / 32;
    const int i      = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int item = i / 4, c = i % 4;
    const bool live = item < columns * slices;
    const int t = live ? item / slices : 0, s = live ? item % slices : 0;
    const std::uint16_t* v = x + static_cast<std::size_t>(t) * k;
    const auto source      = [&](int j) { return perm == nullptr ? j : perm[j / perm_block] * perm_block + j % perm_block; };
    const int j0           = 32 * s + 8 * c;
    std::uint16_t e[8];
    if (live && vec != 0) {
        const uint4 u       = *reinterpret_cast<const uint4*>(v + source(j0));
        const unsigned w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            e[2 * m]     = static_cast<std::uint16_t>(w[m] & 0xFFFFU);
            e[2 * m + 1] = static_cast<std::uint16_t>(w[m] >> 16);
        }
    } else {
#pragma unroll
        for (int m = 0; m < 8; ++m) { e[m] = live ? v[source(j0 + m)] : std::uint16_t{0}; }
    }
    unsigned key = 0;
#pragma unroll
    for (int m = 0; m < 8; ++m) { key = max(key, static_cast<unsigned>(e[m] & 0x7FFFU)); }
    key = max(key, __shfl_xor_sync(0xFFFFFFFFU, key, 1));
    key = max(key, __shfl_xor_sync(0xFFFFFFFFU, key, 2));
    const a8::SliceScale sc = a8::slice_scale(key);
    int q[8];
    int sum = 0;
#pragma unroll
    for (int m = 0; m < 8; ++m) {
        q[m] = a8::code(e[m], sc.inv);
        sum += q[m];
    }
    sum += __shfl_xor_sync(0xFFFFFFFFU, sum, 1);
    sum += __shfl_xor_sync(0xFFFFFFFFU, sum, 2);
    if (live) {
        std::int32_t* column = encoded + static_cast<std::size_t>(t) * stride;
        *reinterpret_cast<int2*>(column + s * a8::kSliceWords + 2 * c) =
            make_int2(fast::pack4(q[0], q[1], q[2], q[3]), fast::pack4(q[4], q[5], q[6], q[7]));
        if (c == 0) {
            reinterpret_cast<float*>(column + a8::scales_offset(k))[s] = sc.d;
            // |sum| <= 4,064: exact in binary32; a non-finite slice (d NaN, codes 0) gives NaN.
            if (sums != 0) { reinterpret_cast<float*>(column + a8::sums_offset(k))[s] = __fmul_rn(sc.d, static_cast<float>(sum)); }
        }
    }
}

// --------------------------------------------------------------------------------------------- the planes

// One CTA per (plane tile, stage): the stage's image of `rows` rows (the last tile padded with the last row). A
// Tiled (ggml_tiles16_v1) matrix's row pointer is offset so that fast::chunk's block of the stage (a stage lies
// inside one 256-element block) lands on the row's block in its tile.
template <QType F, bool Tiled>
__global__ void __launch_bounds__(wt::kThreads)
    plane_kernel(const std::uint8_t* __restrict__ weights, int rows, int row_bytes, std::uint8_t* __restrict__ plane) {
    const int tile = static_cast<int>(blockIdx.x), st = static_cast<int>(blockIdx.y), stages = static_cast<int>(gridDim.y);
    const auto row_of = [&](int r) -> const std::uint8_t* {
        const int row = min(tile * wt::kPlaneTileRows + r, rows - 1);
        if constexpr (Tiled) {
            constexpr int kBlockBytes = static_cast<int>(ggml_block_bytes(F));
            const int b               = st * wt::kStageSlices * 32 / static_cast<int>(ggml_block_elems(F));
            return weights + static_cast<std::size_t>(row / 16) * 16 * row_bytes + static_cast<std::size_t>(row % 16 + 15 * b) * kBlockBytes;
        } else {
            return weights + static_cast<std::size_t>(row) * row_bytes;
        }
    };
    auto& image = *reinterpret_cast<wt::PlaneStage*>(plane + (static_cast<std::size_t>(tile) * stages + st) * sizeof(wt::PlaneStage));
    wt::decode_plane_stage<F>(image, row_of, st);
}

// ---------------------------------------------------------------------------------------------- the tiles

// CTA (x, y): columns 128 x .. of the batch against rows 128 y .. of the slab (plane tiles 2y and 2y + 1, the
// second clamped to the slab's last; rows past the slab are not stored). out[t * n + out_offset + row].
template <wt::Scaling S, bool OutF32>
__global__ void __launch_bounds__(wt::kThreads, 1)
    tile_kernel(const std::uint8_t* __restrict__ plane, int tiles, int rows, int stages, wt::DenseColumns cols,
                void* __restrict__ out, int n, int out_offset, int kappa_log2) {
    extern __shared__ __align__(16) unsigned char a8_plane_shared[];
    auto& sm       = *reinterpret_cast<wt::PlaneShared*>(a8_plane_shared);
    cols.col0      = static_cast<int>(blockIdx.x) * wt::kColumns;
    const int tile0 = static_cast<int>(blockIdx.y) * 2;
    constexpr std::size_t kStride = sizeof(wt::PlaneStage);
    const std::uint8_t* const images[2] = {plane + static_cast<std::size_t>(min(tile0, tiles - 1)) * stages * kStride,
                                           plane + static_cast<std::size_t>(min(tile0 + 1, tiles - 1)) * stages * kStride};
    wt::Accumulators acc;
    wt::compute_plane<S>(sm, images, stages, cols, acc);
    const int warp = static_cast<int>(threadIdx.x) / 32, lane = static_cast<int>(threadIdx.x) % 32;
#pragma unroll
    for (int mi = 0; mi < wt::kMTiles; ++mi) {
#pragma unroll
        for (int nt = 0; nt < wt::kNTiles; ++nt) {
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int row = static_cast<int>(blockIdx.y) * wt::kPlaneRows + wt::out_row(warp, lane, mi, q);
                const int col = cols.col0 + wt::decode_out_column<wt::kNTiles>(warp, lane, nt, q);
                if (row < rows && col < cols.columns) {
                    const float y        = a8::output_f32(acc.r[mi][nt][q], kappa_log2);
                    const std::size_t at = static_cast<std::size_t>(col) * n + out_offset + row;
                    if constexpr (OutF32) {
                        static_cast<float*>(out)[at] = y;
                    } else {
                        static_cast<std::uint16_t*>(out)[at] = canon::f32_to_bf16_rn(y);
                    }
                }
            }
        }
    }
}

// ----------------------------------------------------------------------------------------------- dispatch

#define INFERNIX_GGML_A8W_FORMATS(X)                                                                                   \
    X(QType::GGML_Q2_0)                                                                                                \
    X(QType::GGML_Q4_K)                                                                                                \
    X(QType::GGML_Q5_K)                                                                                                \
    X(QType::GGML_Q6_K)                                                                                                \
    X(QType::GGML_IQ2_S)                                                                                               \
    X(QType::GGML_IQ3_XXS)                                                                                             \
    X(QType::GGML_IQ3_S)                                                                                               \
    X(QType::GGML_IQ4_XS)                                                                                              \
    X(QType::GGML_IQ4_NL)

wt::Scaling scaling_of(QType q) {
    if (q == QType::GGML_Q6_K || q == QType::GGML_IQ2_S) { return wt::Scaling::kHalf; }
    if (min_format(q)) { return wt::Scaling::kMin; }
    return wt::Scaling::kPlain;
}

// The plane of one slab of format F (ggml_tiles16_v1 only for the 256-element block formats, which the
// contract's validation guarantees).
template <QType F>
void launch_plane_format(const GgmlSegment& s, int k, std::uint8_t* plane, cudaStream_t stream) {
    const dim3 grid((s.rows + wt::kPlaneTileRows - 1) / wt::kPlaneTileRows, k / (32 * wt::kStageSlices));
    const int row_bytes = static_cast<int>(ggml_row_bytes(s.format, k));
    if (s.layout == GgmlLayout::Tiles16) {
        if constexpr (ggml_block_elems(F) == 256) {
            plane_kernel<F, true><<<grid, wt::kThreads, 0, stream>>>(s.data, s.rows, row_bytes, plane);
        } else {
            throw std::logic_error("ggml_linear (a8 wide): a tiled segment's format has no tiled planes");
        }
    } else {
        plane_kernel<F, false><<<grid, wt::kThreads, 0, stream>>>(s.data, s.rows, row_bytes, plane);
    }
}

void launch_plane(const GgmlSegment& s, int k, std::uint8_t* plane, cudaStream_t stream) {
#define INFERNIX_GGML_A8W_PLANE(Q)                                                                                     \
    case Q:                                                                                                            \
        launch_plane_format<Q>(s, k, plane, stream);                                                                   \
        break;
    switch (s.format) {
        INFERNIX_GGML_A8W_FORMATS(INFERNIX_GGML_A8W_PLANE)
    default:
        throw std::invalid_argument("ggml_linear (a8 wide): segment format is not a ggml block format");
    }
#undef INFERNIX_GGML_A8W_PLANE
    check_launch("plane");
}

template <wt::Scaling S, bool OutF32>
void launch_tiles_as(const GgmlSegment& s, int k, const wt::DenseColumns& cols, void* out, int n, const std::uint8_t* plane,
                     cudaStream_t stream) {
    constexpr int kShared = static_cast<int>(sizeof(wt::PlaneShared));
    static const bool configured = [] {
        return cudaFuncSetAttribute(tile_kernel<S, OutF32>, cudaFuncAttributeMaxDynamicSharedMemorySize, kShared) == cudaSuccess;
    }();
    require(configured, "the tile kernel's shared memory exceeds the device's");
    const int tiles = (s.rows + wt::kPlaneTileRows - 1) / wt::kPlaneTileRows;
    require((s.rows + wt::kPlaneRows - 1) / wt::kPlaneRows <= 65535, "a slab of at most 65535 row tiles");
    const dim3 grid((cols.columns + wt::kColumns - 1) / wt::kColumns, (s.rows + wt::kPlaneRows - 1) / wt::kPlaneRows);
    tile_kernel<S, OutF32><<<grid, wt::kThreads, kShared, stream>>>(plane, tiles, s.rows, k / (32 * wt::kStageSlices), cols,
                                                                    out, n, s.out_offset, ggml::kappa_log2(s.format));
    check_launch("tile");
}

template <bool OutF32>
void launch_tiles(const GgmlSegment& s, int k, const wt::DenseColumns& cols, void* out, int n, const std::uint8_t* plane,
                  cudaStream_t stream) {
    switch (scaling_of(s.format)) {
    case wt::Scaling::kHalf: launch_tiles_as<wt::Scaling::kHalf, OutF32>(s, k, cols, out, n, plane, stream); break;
    case wt::Scaling::kMin: launch_tiles_as<wt::Scaling::kMin, OutF32>(s, k, cols, out, n, plane, stream); break;
    default: launch_tiles_as<wt::Scaling::kPlain, OutF32>(s, k, cols, out, n, plane, stream); break;
    }
}

#undef INFERNIX_GGML_A8W_FORMATS

// Bytes of the planes of `rows` rows of K = k.
std::size_t plane_bytes(int rows, int k) {
    const std::size_t tiles = static_cast<std::size_t>((rows + wt::kPlaneTileRows - 1) / wt::kPlaneTileRows);
    return tiles * static_cast<std::size_t>(k / (32 * wt::kStageSlices)) * sizeof(wt::PlaneStage);
}

// The row slabs of a call's segments, each a segment of whole plane tiles whose plane fits kA8WidePlaneBytes, grouped
// into passes whose planes fit together (ggml_linear.cu's plan_slabs at this route's image size).
struct Slab {
    GgmlSegment segment;
    std::size_t offset; // of its plane in the pass's workspace
};

std::vector<std::vector<Slab>> plan_slabs(std::span<const GgmlSegment> segments, int k, std::size_t& most) {
    const std::size_t per_tile = plane_bytes(wt::kPlaneTileRows, k);
    require(per_tile <= kA8WidePlaneBytes, "one plane tile exceeds the planes' workspace");
    const int max_rows = static_cast<int>(std::max<std::size_t>(kA8WidePlaneBytes / per_tile, 1)) * wt::kPlaneTileRows;
    std::vector<std::vector<Slab>> passes(1);
    std::size_t used = 0;
    most             = 0;
    for (const auto& s : segments) {
        const std::size_t row_bytes = ggml_row_bytes(s.format, k);
        for (int r = 0; r < s.rows; r += max_rows) {
            GgmlSegment slab = s;
            slab.rows        = std::min(max_rows, s.rows - r);
            slab.data        = s.data + static_cast<std::size_t>(r) * row_bytes; // whole tiles of 16 (Tiles16)
            slab.out_offset  = s.out_offset + r;
            const std::size_t bytes = (plane_bytes(slab.rows, k) + 255) / 256 * 256;
            if (used + bytes > kA8WidePlaneBytes && !passes.back().empty()) {
                passes.emplace_back();
                used = 0;
            }
            passes.back().push_back(Slab{slab, used});
            used += bytes;
            most = std::max(most, used);
        }
    }
    return passes;
}

} // namespace
} // namespace a8_wide

std::size_t a8_wide_linear_workspace_bytes(std::int32_t k, std::int32_t columns) {
    const std::size_t batch = static_cast<std::size_t>(std::min(columns, kA8WideBatchColumns));
    const std::size_t bytes = static_cast<std::size_t>(a8_wide::column_stride(k, true)) * batch * 4;
    return (bytes + 255) / 256 * 256 + kA8WidePlaneBytes;
}

void ggml_linear_a8_wide(const Tensor& x, std::span<const GgmlSegment> segments, std::span<Tensor* const> outs,
                         const std::int32_t* input_perm, std::int32_t perm_block, WorkspaceArena& workspace,
                         cudaStream_t stream) {
    using namespace a8_wide;
    const int k = x.ne[0], columns = x.ne[1];
    require(columns >= kA8WideMinColumns && k % 64 == 0, "the wide 8-bit route takes calls of 32 or more columns of K a multiple of 64");
    const DType dtype = outs.front()->dtype;
    const bool sums   = std::any_of(segments.begin(), segments.end(), [](const GgmlSegment& s) { return min_format(s.format); });
    const int stride  = column_stride(k, sums);
    std::size_t most  = 0;
    const auto passes = plan_slabs(segments, k, most);
    auto* planes      = static_cast<std::uint8_t*>(workspace.alloc(DType::U8, {static_cast<int>(most)}).data);
    const int batch   = std::min(columns, kA8WideBatchColumns);
    Tensor encoded    = workspace.alloc(DType::I32, {stride, batch});
    auto* codes       = static_cast<std::int32_t*>(encoded.data);
    // The encoder's 16-byte loads wherever the columns allow them.
    const int vec = (input_perm == nullptr || perm_block % 8 == 0) && reinterpret_cast<std::uintptr_t>(x.data) % 16 == 0 ? 1 : 0;
    const std::size_t out_bytes = dtype == DType::FP32 ? 4 : 2;
    for (const auto& pass : passes) {
        for (const auto& slab : pass) { launch_plane(slab.segment, k, planes + slab.offset, stream); }
        for (int first = 0; first < columns; first += batch) {
            const int count = std::min(batch, columns - first);
            const int items = count * (k / 32) * 4;
            encode_kernel<<<(items + kEncodeThreads - 1) / kEncodeThreads, kEncodeThreads, 0, stream>>>(
                static_cast<const std::uint16_t*>(x.data) + static_cast<std::size_t>(first) * k, k, input_perm, perm_block, codes,
                stride, count, sums ? 1 : 0, vec);
            check_launch("encode");
            const ggml::a8::wide::DenseColumns cols{.encoded = codes, .stride = stride, .k = k, .col0 = 0, .columns = count};
            for (const auto& slab : pass) {
                const Tensor& out = *outs[static_cast<std::size_t>(slab.segment.output)];
                void* at          = static_cast<std::byte*>(out.data) + static_cast<std::size_t>(first) * out.ne[0] * out_bytes;
                if (dtype == DType::FP32) {
                    launch_tiles<true>(slab.segment, k, cols, at, out.ne[0], planes + slab.offset, stream);
                } else {
                    launch_tiles<false>(slab.segment, k, cols, at, out.ne[0], planes + slab.offset, stream);
                }
            }
        }
    }
}

} // namespace infernix::ops
