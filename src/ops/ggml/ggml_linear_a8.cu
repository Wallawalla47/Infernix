// ops::ggml_linear's 8-bit activation route (LinearRoutes::GgmlA8, `--a8-activations`: LOSSY, opt-in;
// docs/maintainer/qwen3_8-flash-next-design.md §19.3.31; the kernels' design: ops/ggml/ggml_linear_a8.cuh):
// the column encoder, the dp4a rows kernel and the dispatch of a call's segments to the rows and mma launches
// (ggml_linear_a8_rows.cu, ggml_linear_a8_tiled.cu), which follows the exact route's (ggml_linear.cu): a call
// of exactly two mixed formats in one layout (BF16 outputs, at most A8Group::kMax segments) in one mma launch
// (Rows from 3 columns, Tiles16 at every width), else the segments of each format and layout in launches of up
// to A8Group::kMax.

#include "ops/ggml/ggml_linear_a8.h"
#include "ops/ggml/ggml_linear_a8.cuh"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace infernix::ops {
namespace a8_linear {
namespace {

// ------------------------------------------------------------------------------------------------- encode

// Four threads per 32-element slice of the call (slice s of column t: items t * K / 32 + s), thread c taking
// elements 8c .. 8c + 7 (input element j at x[perm(j)]; one 16-byte load where `vec`: perm_block % 8 == 0 and
// a 16-byte aligned x). The slice's maximum and code sum are the quad's (two shuffles each); thread c writes
// code words 2c, 2c + 1 and thread 0 the scale (and the code sum where `sums`): a8::column_words' layout,
// columns `stride` words apart.
__global__ void __launch_bounds__(kEncodeThreads)
    encode_kernel(const std::uint16_t* __restrict__ x, int k, const std::int32_t* __restrict__ perm, int perm_block,
                  std::int32_t* __restrict__ encoded, int stride, int columns, int sums, int vec) {
    pdl::enter(); // x is the previous kernel's output
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
        const uint4 u = *reinterpret_cast<const uint4*>(v + source(j0));
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
            if (sums != 0) { column[a8::sums_offset(k) + s] = sum; }
        }
    }
}

// --------------------------------------------------------------------------------------------- rows (dp4a)

// The exact int32 dot of a decoded weight slice with one column's slice codes, sub-scales applied
// (ggml_expert_a8.cuh's a8_slice_dot): |isum| < 2^24.
__device__ __forceinline__ int slice_dot(const fast::Slice& w, const int4& x0, const int4& x1) {
    const int p0 = __dp4a(w.q[3], x0.w, __dp4a(w.q[2], x0.z, __dp4a(w.q[1], x0.y, __dp4a(w.q[0], x0.x, 0))));
    const int p1 = __dp4a(w.q[7], x1.w, __dp4a(w.q[6], x1.z, __dp4a(w.q[5], x1.y, __dp4a(w.q[4], x1.x, 0))));
    return w.sub0 * p0 + w.sub1 * p1;
}

// Lanes per row: a power of two in [4, 32] leaving each lane about four or more slices (ggml_linear.cu's
// lanes_for), so a row's order depends on K alone.
int lanes_for(int slices) {
    int lanes = 4;
    while (lanes < 32 && slices / (2 * lanes) >= 4) { lanes *= 2; }
    return lanes;
}

// One output row per `Lanes` lanes, kRowsWarps warps a CTA, every column of a call of at most kRowsMaxColumns
// in one pass (each slice decoded once). Tiled (ggml_tiles16_v1): a row's blocks are 16 apart, so slice s's
// row pointer is offset by s / 8 * 15 blocks (ggml_linear.cu's tiled rows_kernel).
template <QType F, bool OutF32, int Lanes, bool Tiled>
__global__ void __launch_bounds__(kRowsThreads)
    rows_kernel(A8Group g, int slices, const std::int32_t* __restrict__ encoded, int stride, int columns) {
    constexpr int kRowsPerWarp        = 32 / Lanes;
    constexpr std::size_t kBlockBytes = ggml_block_bytes(F);
    const Slot seg = slot_at(g, static_cast<int>(blockIdx.x));
    const int warp = static_cast<int>(threadIdx.x) / 32, lane = static_cast<int>(threadIdx.x) % 32;
    const int part = lane % Lanes;
    const int row  = (seg.block * kRowsWarps + warp) * kRowsPerWarp + lane / Lanes;
    const bool active = row < seg.rows;
    const int r       = active ? row : 0;
    const std::uint8_t* w = Tiled ? seg.weights + static_cast<std::size_t>(r / 16) * 16 * seg.row_bytes + (r % 16) * kBlockBytes
                                  : seg.weights + static_cast<std::size_t>(r) * seg.row_bytes;
    const auto row_of = [&](int s) { // the row pointer fast::slice takes for slice s
        if constexpr (Tiled) {
            static_assert(ggml_block_elems(F) == 256, "tiled rows: 256-element blocks");
            return w + static_cast<std::size_t>(s / 8) * 15 * kBlockBytes;
        } else {
            return w;
        }
    };
    const int k = 32 * slices;
    // A programmatic dependent of the encoder: the lane's first slice is read from the immutable weights while
    // the encoder still runs.
    fast::Slice first{};
    if (active && part < slices) { first = fast::slice<F>(row_of(part), part); }
    pdl::wait_for_dependencies();
    float acc[kRowsMaxColumns];
#pragma unroll
    for (int c = 0; c < kRowsMaxColumns; ++c) { acc[c] = 0.0F; }
    if (active) {
        for (int s = part; s < slices; s += Lanes) {
            const fast::Slice ws = s == part ? first : fast::slice<F>(row_of(s), s);
            const float wd       = weight_scale(ws.d);
            [[maybe_unused]] const float wmin = fast::kMin<F> ? weight_scale(ws.dmin) : 0.0F;
#pragma unroll
            for (int c = 0; c < kRowsMaxColumns; ++c) {
                if (c < columns) { // uniform across the warp
                    const std::int32_t* column = encoded + static_cast<std::size_t>(c) * stride;
                    const int4* codes          = reinterpret_cast<const int4*>(column + s * a8::kSliceWords);
                    const float xd             = __ldg(reinterpret_cast<const float*>(column + a8::scales_offset(k)) + s);
                    acc[c] = a8::add_term(acc[c], slice_dot(ws, __ldg(codes), __ldg(codes + 1)), xd, wd);
                    if constexpr (fast::kMin<F>) {
                        acc[c] = a8::sub_min_term(acc[c], ws.min, __ldg(column + a8::sums_offset(k) + s), xd, wmin);
                    }
                }
            }
        }
    }
    const int kappa = ggml::kappa_log2(F);
#pragma unroll
    for (int c = 0; c < kRowsMaxColumns; ++c) {
        if (c < columns) {
#pragma unroll
            for (int offset = Lanes / 2; offset >= 1; offset >>= 1) {
                acc[c] = canon::add_rn(acc[c], __shfl_xor_sync(0xFFFFFFFFU, acc[c], offset));
            }
            if (active && part == 0) {
                const std::size_t at = static_cast<std::size_t>(c) * seg.n + seg.out_offset + row;
                const float y        = a8::output_f32(acc[c], kappa);
                if constexpr (OutF32) {
                    static_cast<float*>(seg.out)[at] = y;
                } else {
                    static_cast<std::uint16_t*>(seg.out)[at] = canon::f32_to_bf16_rn(y);
                }
            }
        }
    }
}

template <QType F, bool OutF32, bool Tiled>
void launch_rows(A8Group g, const Columns& in, cudaStream_t stream) {
    const int slices  = in.k / 32;
    const int lanes   = lanes_for(slices);
    const int per_cta = kRowsWarps * (32 / lanes);
    g.first[0]        = 0;
    for (int i = 0; i < g.count; ++i) { g.first[i + 1] = g.first[i] + (g.rows[i] + per_cta - 1) / per_cta; }
    const dim3 grid(g.first[g.count]);
#define INFERNIX_GGML_A8_LANES(LN)                                                                                     \
    case LN:                                                                                                           \
        CUDA_CHECK(pdl::launch_consumer({grid, dim3(kRowsThreads), 0, stream}, rows_kernel<F, OutF32, LN, Tiled>, g,   \
                                        slices, in.encoded, in.stride, in.columns));                                   \
        break;
    switch (lanes) {
        INFERNIX_GGML_A8_LANES(4)
        INFERNIX_GGML_A8_LANES(8)
        INFERNIX_GGML_A8_LANES(16)
        INFERNIX_GGML_A8_LANES(32)
    default:
        throw std::logic_error("ggml_linear (a8): lanes");
    }
#undef INFERNIX_GGML_A8_LANES
    check_launch("rows");
}

// A group of one format and layout: the rows kernel where the exact route takes its own (Rows at 1-2
// columns; Tiles16 Q6_K and IQ4_XS at one column, ggml_linear.cu's tiled_rows_route), else the mma route.
template <bool OutF32>
void launch_group(QType format, GgmlLayout layout, const A8Group& g, const Columns& in, cudaStream_t stream) {
    if (layout == GgmlLayout::Tiles16) {
        if (in.columns == 1 && format == QType::GGML_Q6_K) {
            launch_rows<QType::GGML_Q6_K, OutF32, true>(g, in, stream);
        } else if (in.columns == 1 && format == QType::GGML_IQ4_XS) {
            launch_rows<QType::GGML_IQ4_XS, OutF32, true>(g, in, stream);
        } else {
            launch_mma_tiled(format, g, in, OutF32, stream);
        }
        return;
    }
    if (in.columns > kRowsMaxColumns) {
        launch_mma_rows(format, g, in, OutF32, stream);
        return;
    }
#define INFERNIX_GGML_A8_ROWS(Q)                                                                                       \
    case Q:                                                                                                            \
        launch_rows<Q, OutF32, false>(g, in, stream);                                                                  \
        break;
    switch (format) {
        INFERNIX_GGML_A8_FORMATS(INFERNIX_GGML_A8_ROWS)
    default:
        throw std::invalid_argument("ggml_linear (a8): segment format is not a ggml block format");
    }
#undef INFERNIX_GGML_A8_ROWS
}

void add_segment(A8Group& g, const GgmlSegment& s, std::span<Tensor* const> outs, int k, int format) {
    const Tensor& out     = *outs[static_cast<std::size_t>(s.output)];
    g.data[g.count]       = s.data;
    g.out[g.count]        = out.data;
    g.rows[g.count]       = s.rows;
    g.n[g.count]          = out.ne[0];
    g.out_offset[g.count] = s.out_offset;
    g.row_bytes[g.count]  = static_cast<int>(ggml_row_bytes(s.format, k));
    g.format[g.count]     = format;
    ++g.count;
}

} // namespace
} // namespace a8_linear

std::size_t a8_linear_workspace_bytes(std::int32_t k, std::int32_t columns, bool sums) {
    const std::size_t bytes = static_cast<std::size_t>(a8_linear::column_stride(k, sums)) * static_cast<std::size_t>(columns) * 4;
    return (bytes + 255) / 256 * 256;
}

void ggml_linear_a8(const Tensor& x, std::span<const GgmlSegment> segments, std::span<Tensor* const> outs,
                    const std::int32_t* input_perm, std::int32_t perm_block, WorkspaceArena& workspace, cudaStream_t stream) {
    using namespace a8_linear;
    const int k = x.ne[0], columns = x.ne[1];
    require(columns >= 1 && columns <= kA8MaxColumns, "the 8-bit route takes calls of 1-31 columns");
    const DType dtype = outs.front()->dtype;
    const bool sums   = std::any_of(segments.begin(), segments.end(), [](const GgmlSegment& s) {
        return s.format == QType::GGML_Q4_K || s.format == QType::GGML_Q5_K;
    });
    const int stride = column_stride(k, sums);
    Tensor encoded   = workspace.alloc(DType::I32, {stride, columns});
    // The encoder's 16-byte loads wherever the columns allow them.
    const int vec   = (input_perm == nullptr || perm_block % 8 == 0) && reinterpret_cast<std::uintptr_t>(x.data) % 16 == 0 ? 1 : 0;
    const int items = columns * (k / 32) * 4;
    CUDA_CHECK(pdl::launch_consumer({dim3((items + kEncodeThreads - 1) / kEncodeThreads), dim3(kEncodeThreads), 0, stream},
                                    encode_kernel, static_cast<const std::uint16_t*>(x.data), k, input_perm, perm_block,
                                    static_cast<std::int32_t*>(encoded.data), stride, columns, sums ? 1 : 0, vec));
    check_launch("encode");
    const Columns in{.encoded = static_cast<const std::int32_t*>(encoded.data), .stride = stride, .k = k, .columns = columns};
    // A call of exactly two mixed-capable formats in one layout (BF16 outputs, at most A8Group::kMax segments)
    // runs in one launch.
    const GgmlLayout layout0 = segments.front().layout;
    const bool tiled         = layout0 == GgmlLayout::Tiles16;
    if (dtype == DType::BF16 && columns >= (tiled ? 1 : kMmaMinColumns) && segments.size() <= static_cast<std::size_t>(A8Group::kMax)) {
        QType f0 = segments.front().format, f1 = f0;
        bool ok  = true;
        for (const auto& s : segments) {
            ok = ok && mixed_format(s.format) && s.layout == layout0;
            if (s.format != f0 && f1 == f0) {
                f1 = s.format;
            } else if (s.format != f0 && s.format != f1) {
                ok = false;
            }
        }
        if (ok && f1 != f0) {
            if (static_cast<int>(f1) < static_cast<int>(f0)) { std::swap(f0, f1); }
            A8Group g;
            for (const auto& s : segments) { add_segment(g, s, outs, k, s.format == f0 ? 0 : 1); }
            if (tiled ? launch_mma_tiled_mixed(f0, f1, g, in, stream) : launch_mma_rows_mixed(f0, f1, g, in, stream)) { return; }
        }
    }
    // The segments of each format and layout, in order of first appearance, share launches of up to
    // A8Group::kMax.
    std::vector<bool> taken(segments.size(), false);
    for (std::size_t i = 0; i < segments.size(); ++i) {
        if (taken[i]) { continue; }
        const QType format      = segments[i].format;
        const GgmlLayout layout = segments[i].layout;
        A8Group g;
        for (std::size_t j = i; j < segments.size() && g.count < A8Group::kMax; ++j) {
            if (taken[j] || segments[j].format != format || segments[j].layout != layout) { continue; }
            taken[j] = true;
            add_segment(g, segments[j], outs, k, 0);
        }
        if (dtype == DType::FP32) {
            launch_group<true>(format, layout, g, in, stream);
        } else {
            launch_group<false>(format, layout, g, in, stream);
        }
    }
}

} // namespace infernix::ops
