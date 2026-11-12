#include "core/weight.h"
#include "ops/attn_input_proj/q4_q5/q4_q5_attn_input_kernels.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/common/rowsplit_grouped_mma.cuh"
#include "ops/common/rowsplit_grouped_ring.cuh"
#include "ops/linear/common/rowsplit_ring_linear.cuh"
#include "ops/common/token_slices.h"

#include <cstdint>
#include <stdexcept>

namespace infernix::ops::detail {
namespace {

RowSplitGroupedMmaJob make_job(const Weight& weight, std::int32_t row_begin, std::int32_t row_count,
                               Tensor& out) {
    if (row_begin < 0 || row_count <= 0 || row_begin + row_count > weight.n ||
        out.ne[0] != row_count) {
        throw std::invalid_argument("Q4/Q5 attention input grouped MMA row view is invalid");
    }
    const std::int64_t groups = weight.padded_shape[1] / 64;
    const auto* codes         = static_cast<const std::uint8_t*>(weight.qdata) +
                        static_cast<std::int64_t>(row_begin) * groups * 32;
    const auto* high   = weight.qtype == QType::Q5_G64_FP16
                             ? static_cast<const std::uint8_t*>(weight.qhigh) +
                                 static_cast<std::int64_t>(row_begin) * groups * 8
                             : nullptr;
    const auto* scales = static_cast<const std::uint8_t*>(weight.scales) +
                         static_cast<std::int64_t>(row_begin) * groups * 2;
    return RowSplitGroupedMmaJob{
        codes,     high,      scales, static_cast<__nv_bfloat16*>(out.data),
        row_count, out.ne[0], 0,      weight.qtype == QType::Q5_G64_FP16,
    };
}

template <class Schedule, RowSplitGroupedMmaCodec Codec>
void launch_pair(bool full, const Tensor& x, RowSplitGroupedMmaJob first,
                 RowSplitGroupedMmaJob second, cudaStream_t stream) {
    const int tiles = div_up(first.n, Schedule::BM) + div_up(second.n, Schedule::BM);
    const int cols  = x.ne[1];
    const dim3 grid(static_cast<unsigned>(tiles),
                    static_cast<unsigned>(div_up(cols, Schedule::BN)));
    RowSplitGroupedMmaJob empty{};

    if (full) {
        rowsplit_grouped_mma_kernel<Schedule, true, Codec, 2>
            <<<grid, Schedule::THREADS, 0, stream>>>(static_cast<const __nv_bfloat16*>(x.data),
                                                     first, second, empty, empty, x.ne[0], cols,
                                                     x.ne[0]);
    } else {
        rowsplit_grouped_mma_kernel<Schedule, false, Codec, 2>
            <<<grid, Schedule::THREADS, 0, stream>>>(static_cast<const __nv_bfloat16*>(x.data),
                                                     first, second, empty, empty, x.ne[0], cols,
                                                     x.ne[0]);
    }
    CUDA_CHECK(cudaGetLastError());
}

template <class Schedule>
void launch_slice(const Tensor& x, const Weight& query_key_weight, const Weight& gate_value_weight,
                  Tensor& q, Tensor& gate, Tensor& k, Tensor& v, cudaStream_t stream) {
    const bool full = (x.ne[1] % Schedule::BN) == 0;
    launch_pair<Schedule, RowSplitGroupedMmaCodec::Q4>(
        full, x, make_job(query_key_weight, 0, 6144, q), make_job(query_key_weight, 6144, 1024, k),
        stream);
    launch_pair<Schedule, RowSplitGroupedMmaCodec::Q5>(
        full, x, make_job(gate_value_weight, 0, 6144, gate),
        make_job(gate_value_weight, 6144, 1024, v), stream);
}

template <class Schedule>
void launch(const Tensor& x, const Weight& query_key_weight, const Weight& gate_value_weight,
            Tensor& q, Tensor& gate, Tensor& k, Tensor& v, cudaStream_t stream) {
    constexpr std::int32_t kSliceCols = Schedule::BN;
    for_each_token_slice(x.ne[1], kSliceCols, [&](std::int32_t offset, std::int32_t count) {
        const Tensor x_slice = x.slice(1, offset, count);
        Tensor q_slice       = q.slice(1, offset, count);
        Tensor gate_slice    = gate.slice(1, offset, count);
        Tensor k_slice       = k.slice(1, offset, count);
        Tensor v_slice       = v.slice(1, offset, count);
        launch_slice<Schedule>(x_slice, query_key_weight, gate_value_weight, q_slice, gate_slice,
                               k_slice, v_slice, stream);
    });
}

using MmaR32C64S4 = GemmCfg<32, 64, 64, 16, 16, 4, 1, false, true, true>;

template <class S, bool Full>
void mixed_slice(const Tensor& x, const Weight& w0, const Weight& w1, Tensor& q, Tensor& g,
                 Tensor& k, Tensor& v, cudaStream_t stream) {
    const dim3 grid(14336 / S::BM, (x.ne[1] + S::BN - 1) / S::BN);
    rowsplit_grouped_mma_kernel<S, Full, RowSplitGroupedMmaCodec::Mixed, 4>
        <<<grid, S::THREADS, 0, stream>>>(static_cast<const __nv_bfloat16*>(x.data),
                                          make_job(w0, 0, 6144, q), make_job(w0, 6144, 1024, k),
                                          make_job(w1, 0, 6144, g), make_job(w1, 6144, 1024, v),
                                          5120, x.ne[1], 5120);
    CUDA_CHECK(cudaGetLastError());
}

template <class S>
void launch_mixed(const Tensor& x, const Weight& w0, const Weight& w1, Tensor& q, Tensor& g,
                  Tensor& k, Tensor& v, cudaStream_t stream) {
    for_each_token_slice(x.ne[1], S::BN, [&](int begin, int count) {
        const Tensor xs = x.slice(1, begin, count);
        Tensor qs = q.slice(1, begin, count), gs = g.slice(1, begin, count),
               ks = k.slice(1, begin, count), vs = v.slice(1, begin, count);
        if (count % S::BN == 0)
            mixed_slice<S, true>(xs, w0, w1, qs, gs, ks, vs, stream);
        else
            mixed_slice<S, false>(xs, w0, w1, qs, gs, ks, vs, stream);
    });
}
} // namespace

void q4_q5_attn_input_grouped_mma_r32_c64_s4_launch(const Tensor& x, const Weight& query_key_weight,
                                                    const Weight& gate_value_weight, Tensor& q,
                                                    Tensor& gate, Tensor& k, Tensor& v,
                                                    cudaStream_t stream) {
    launch<MmaR32C64S4>(x, query_key_weight, gate_value_weight, q, gate, k, v, stream);
}

void q4_q5_attn_input_mixed_r32_c32_s2_launch(const Tensor& x, const Weight& w0, const Weight& w1,
                                              Tensor& q, Tensor& g, Tensor& k, Tensor& v,
                                              cudaStream_t stream) {
    launch_mixed<GemmCfg<32, 32, 64, 16, 16, 2, 1, false, true, true>>(x, w0, w1, q, g, k, v,
                                                                       stream);
}

void q4_q5_attn_input_mixed_r32_c64_s3_launch(const Tensor& x, const Weight& w0, const Weight& w1,
                                              Tensor& q, Tensor& g, Tensor& k, Tensor& v,
                                              cudaStream_t stream) {
    launch_mixed<GemmCfg<32, 64, 64, 16, 16, 3, 3, false, true, true>>(x, w0, w1, q, g, k, v,
                                                                       stream);
}

void q4_q5_attn_input_mixed_ring_c128_launch(const Tensor& x, const Weight& w0, const Weight& w1,
                                             Tensor& q, Tensor& g, Tensor& k, Tensor& v,
                                             cudaStream_t stream) {
    // The four jobs on the staging ring (ops/common/rowsplit_grouped_ring.cuh). RTX 5090, attn_input_proj_bench
    // (q4q5, CUDA graph, cold L2, A B B A), against the 64-row MMA tile this route ran before: 256 columns -4.2 %,
    // 512 -6.4 %, 896 -6.6 %, 1,024 -5.5 %, 2,048 -6.7 %, 4,096 -6.7 % (2026-11-10).
    using Tile = RowSplitRingTile128;
    using Ring = RowSplitRing<Tile, kRowSplitRingStages, kRowSplitRingProducerWarps>;
    // ZK-11: a remainder past the last whole 128-column tile is cheaper on the narrow routes than a nearly empty ring
    // tile when that tile adds a wave (as the Q4 gate/up ring's, q4_linear_swiglu_gemm_mma.cu, N6-2): at most
    // kExactTailCols columns for one more wave, up to kNarrowTailCols for two. Where the extra tile shares the last
    // wave it costs nothing, and a split there only adds the narrow call (257-280 columns split were 7-22 % slower).
    // RTX 5090, attn_input_proj_bench (q4q5, a16, CUDA graph, cold L2, A B B A, 2026-11-12): 385-408 columns -12.5 ..
    // -12.8 %, 513-536 -9.5 .. -10.0 %, 769-792 -7.5 .. -8.0 %, every other width of 256-792 within 0.3 %. The tail runs
    // on the r32 routes, whose columns have the ring's bits (equal output hashes at all 22 widths); the small_t route's
    // do not (another summation order), so it takes no tail.
    constexpr std::int32_t kNarrowTailCols = 40;
    constexpr std::int32_t kExactTailCols  = 24;
    const std::int32_t tokens = x.ne[1];
    const std::int32_t blocks = tokens / Tile::BN;
    const std::int32_t tail   = tokens - blocks * Tile::BN;
    const std::int64_t rows   = 14336 / Tile::BM;
    const auto extra_waves = [&] {
        return blocks > 0 ? rowsplit_ring_waves(rows * (blocks + 1)) - rowsplit_ring_waves(rows * blocks) : 0;
    };
    const std::int64_t added = tail > 0 ? extra_waves() : 0;
    if (blocks > 0 && tail > 0 && added >= 1 && tail <= (added >= 2 ? kNarrowTailCols : kExactTailCols)) {
        const std::int32_t wide = blocks * Tile::BN;
        Tensor qt = q.slice(1, wide, tail), gt = g.slice(1, wide, tail), kt = k.slice(1, wide, tail),
               vt = v.slice(1, wide, tail);
        const Tensor xt = x.slice(1, wide, tail);
        Tensor qw = q.slice(1, 0, wide), gw = g.slice(1, 0, wide), kw = k.slice(1, 0, wide), vw = v.slice(1, 0, wide);
        q4_q5_attn_input_mixed_ring_c128_launch_tiles(x.slice(1, 0, wide), w0, w1, qw, gw, kw, vw, stream);
        if (tail <= 32) {
            q4_q5_attn_input_mixed_r32_c32_s2_launch(xt, w0, w1, qt, gt, kt, vt, stream);
        } else {
            q4_q5_attn_input_mixed_r32_c64_s3_launch(xt, w0, w1, qt, gt, kt, vt, stream);
        }
        return;
    }
    q4_q5_attn_input_mixed_ring_c128_launch_tiles(x, w0, w1, q, g, k, v, stream);
}

// The ring tiles over every column of x (q4_q5_attn_input_mixed_ring_c128_launch's whole-tile part).
void q4_q5_attn_input_mixed_ring_c128_launch_tiles(const Tensor& x, const Weight& w0, const Weight& w1, Tensor& q,
                                                   Tensor& g, Tensor& k, Tensor& v, cudaStream_t stream) {
    using Tile = RowSplitRingTile128;
    using Ring = RowSplitRing<Tile, kRowSplitRingStages, kRowSplitRingProducerWarps>;
    for_each_token_slice(x.ne[1], Tile::BN, [&](int begin, int count) {
        const Tensor xs = x.slice(1, begin, count);
        Tensor qs = q.slice(1, begin, count), gs = g.slice(1, begin, count), ks = k.slice(1, begin, count),
               vs = v.slice(1, begin, count);
        const dim3 grid(14336 / Tile::BM, (count + Tile::BN - 1) / Tile::BN);
        const auto launch = [&]<bool Full>() {
            constexpr auto kernel =
                rowsplit_grouped_ring_kernel<Tile, kRowSplitRingStages, kRowSplitRingProducerWarps, Full, 4>;
            kernel<<<grid, Ring::kThreads, rowsplit_ring_reserve<kernel, Ring::kSharedBytes>(), stream>>>(
                static_cast<const __nv_bfloat16*>(xs.data), make_job(w0, 0, 6144, qs), make_job(w0, 6144, 1024, ks),
                make_job(w1, 0, 6144, gs), make_job(w1, 6144, 1024, vs), 5120, count, 5120);
            CUDA_CHECK(cudaGetLastError());
        };
        if (count % Tile::BN == 0) {
            launch.template operator()<true>();
        } else {
            launch.template operator()<false>();
        }
    });
}
} // namespace infernix::ops::detail
