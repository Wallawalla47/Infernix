#include "core/weight.h"
#include "ops/linear_swiglu/q4/q4_linear_swiglu_kernels.h"

#include "ops/linear_swiglu/q4/q4_linear_swiglu_gemm_mma.cuh"
#include "ops/linear/common/rowsplit_ring_linear.cuh"
#include "ops/linear_swiglu/q4/q4_linear_swiglu_gemm_ring.cuh"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/common/token_slices.h"
#include "ops/lora/lora_device.cuh"

#include <cstdint>

namespace infernix::ops::detail {
namespace {

// The narrow tail's tile (the plain kernel, q4_linear_swiglu_gemm_mma.cuh).
using GateUpC40Cfg = GemmCfg<64, 40, 64, 64, 8, 2, 1, false, true, true>;
// The ring kernel's tile (q4_linear_swiglu_gemm_ring.cuh): the ring linears' (128 weight rows, so 64 outputs, x
// 128 columns a CTA, sixteen MMA warps, three stages, four producer warps).
using GateUpRingCfg = RowSplitRingTile128;
inline constexpr int kGateUpRingStages        = kRowSplitRingStages;
inline constexpr int kGateUpRingProducerWarps = kRowSplitRingProducerWarps;

// `lora` below: the call's term over its columns (a slice takes it from the slice's first column), or null.
template <class Cfg, bool Full>
void launch_folded(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream, const LoraAddend* lora,
                   std::int32_t first_column) {
    constexpr int PM = Cfg::BM / 2;
    const int t      = x.ne[1];
    const dim3 grid(static_cast<unsigned>(div_up(out.ne[0], PM)),
                    static_cast<unsigned>(div_up(t, Cfg::BN)));
    with_lora_term(lora, first_column, [&](auto term) {
        q4_linear_swiglu_mma_split_half_pair_kernel<Cfg, Full, decltype(term)><<<grid, Cfg::THREADS, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), static_cast<__nv_bfloat16*>(out.data), out.ne[0],
            x.ne[0], t, weight.padded_shape[1], term);
    });
    CUDA_CHECK(cudaGetLastError());
}

template <class Cfg>
void launch_route(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream, const LoraAddend* lora) {
    const bool full = (x.ne[1] % Cfg::BN) == 0;
    for_each_token_slice(x.ne[1], Cfg::BN, [&](std::int32_t offset, std::int32_t count) {
        const Tensor x_slice = x.slice(1, offset, count);
        Tensor out_slice     = out.slice(1, offset, count);
        if (full) {
            launch_folded<Cfg, true>(x_slice, weight, out_slice, stream, lora, offset);
        } else {
            launch_folded<Cfg, false>(x_slice, weight, out_slice, stream, lora, offset);
        }
    });
}

// The ring kernel on Cfg's tile with `Stages` stages and `ProducerWarps` producer warps. A full launch reads no
// bounds: whole column tiles, whole row tiles and no K padding.
template <class Cfg, int Stages, int ProducerWarps>
void launch_ring(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream, const LoraAddend* lora) {
    using Layout    = RowSplitRing<Cfg, Stages, ProducerWarps>;
    const bool full = (x.ne[1] % Cfg::BN) == 0 && (out.ne[0] % (Cfg::BM / 2)) == 0 &&
                      x.ne[0] == weight.padded_shape[1];
    for_each_token_slice(x.ne[1], Cfg::BN, [&](std::int32_t offset, std::int32_t count) {
        const Tensor x_slice = x.slice(1, offset, count);
        Tensor out_slice     = out.slice(1, offset, count);
        const dim3 grid(static_cast<unsigned>(div_up(out.ne[0], Cfg::BM / 2)),
                        static_cast<unsigned>(div_up(count, Cfg::BN)));
        with_lora_term(lora, offset, [&](auto term) {
            const auto launch = [&]<bool Full>() {
                constexpr auto kernel = q4_linear_swiglu_ring_kernel<Cfg, Stages, ProducerWarps, Full, decltype(term)>;
                kernel<<<grid, Layout::kThreads, reserve_dynamic_shared<kernel, Layout::kSharedBytes>(), stream>>>(
                    static_cast<const __nv_bfloat16*>(x_slice.data), static_cast<const std::uint8_t*>(weight.qdata),
                    static_cast<const std::uint8_t*>(weight.scales), static_cast<__nv_bfloat16*>(out_slice.data),
                    out.ne[0], x.ne[0], count, weight.padded_shape[1], term);
                CUDA_CHECK(cudaGetLastError());
            };
            if (full) {
                launch.template operator()<true>();
            } else {
                launch.template operator()<false>();
            }
        });
    });
}

// Whole and partly filled ring tiles only.
void launch_ring_tiles(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream,
                       const LoraAddend* lora) {
    // RTX 5090, q4_linear_swiglu_bench ([34816, 5120], CUDA graph, cold L2, A B B A, the same output hashes at
    // every width), against the 64-row plain tiles this route ran before (2026-11-10): 136 columns -10.7 %, 160
    // -8.4 %, 225 -5.5 %, 256 -5.5 %, 400 -8.1 %, 512 -8.5 %, 768 -12.7 %, 896 -10.1 %, 1,000 -14.3 %, 1,024 -13.2 %,
    // 1,536 -11.6 %, 2,048 -14.1 %, 3,584 -14.2 %, 4,096 -12.9 %, 8,192 -10.4 %, 16,384 -13.1 % (224-227 TFLOP/s at
    // 1,024-2,048 against 194). A 64-row ring tile (two CTAs an SM) measured -8.0 % at 256 columns and less than
    // this one everywhere else; the plain kernel on a 128-row tile, a one-barrier pipelined form of it and a ring
    // whose consumers looked their weights up in a table a row all measured slower than this.
    launch_ring<GateUpRingCfg, kGateUpRingStages, kGateUpRingProducerWarps>(x, weight, out, stream, lora);
}

} // namespace

void q4_linear_swiglu_mma_ring_r64_c128_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                               cudaStream_t stream, const LoraAddend* lora) {
    // The ring tile is 128 columns wide and a ring CTA decodes its rows' weights however few of its columns are
    // live, so a column tile holding a short remainder costs a whole tile: one or two more waves of the launch
    // (272 row tiles a column tile on 170 SMs). A remainder of at most kExactTailCols columns is cheaper on the
    // narrow routes at every width, and one of up to kNarrowTailCols is where its tile would add two waves; any
    // other remainder stays in the ring launch. RTX 5090, q4_linear_swiglu_bench, against the ring alone
    // (2026-11-10): remainders of 8 / 16 / 24 columns after 3 ring tiles -22.5 / -20.1 / -17.6 %, after 5 -16.0
    // / -14.5 / -12.7 %, after 8 -11.2 / -10.2 / -9.0 %, after 16 -6.4 % (8) and -5.2 % (24), after 28 -4.4 /
    // -3.9 / -3.6 %; remainders of 25-40 columns after 3 tiles -11.5 %, after 5 -8.5 % (two waves each), after
    // 2, 4 and 7 tiles +0.2 .. +2.8 % (one wave: they stay in the ring launch). The narrow routes' columns do
    // not have the ring's bits (another summation order).
    constexpr std::int32_t kBlockCols      = GateUpRingCfg::BN;
    constexpr std::int32_t kNarrowTailCols = 40;
    constexpr std::int32_t kExactTailCols  = 24;

    const std::int32_t tokens = x.ne[1];
    const std::int32_t blocks = tokens / kBlockCols;
    const std::int32_t tail   = tokens - blocks * kBlockCols;
    const std::int64_t rows   = div_up(out.ne[0], GateUpRingCfg::BM / 2);
    const bool two_waves      = blocks > 0 && rowsplit_ring_waves(rows * (blocks + 1)) - rowsplit_ring_waves(rows * blocks) >= 2;
    if (blocks == 0 || tail == 0 || tail > (two_waves ? kNarrowTailCols : kExactTailCols)) {
        launch_ring_tiles(x, weight, out, stream, lora);
        return;
    }

    const std::int32_t wide_cols = blocks * kBlockCols;
    const Tensor x_wide = x.slice(1, 0, wide_cols);
    Tensor out_wide     = out.slice(1, 0, wide_cols);
    launch_ring_tiles(x_wide, weight, out_wide, stream, lora);
    const Tensor x_tail = x.slice(1, wide_cols, tail);
    Tensor out_tail     = out.slice(1, wide_cols, tail);
    // The tail's term reads U from its first column.
    LoraAddend tail_lora;
    if (lora != nullptr) { tail_lora = lora_columns_from(*lora, wide_cols); }
    const LoraAddend* tail_term = lora != nullptr ? &tail_lora : nullptr;
    if (tail == 1) {
        // The exact small-T tiled route starts at two columns.
        q4_linear_swiglu_gemv_pair_launch(x_tail, weight, out_tail, stream, tail_term);
    } else if (tail <= kExactTailCols) {
        q4_linear_swiglu_small_t_tiled_launch(x_tail, weight, out_tail, stream, tail_term);
    } else {
        launch_route<GateUpC40Cfg>(x_tail, weight, out_tail, stream, tail_term);
    }
}

} // namespace infernix::ops::detail
