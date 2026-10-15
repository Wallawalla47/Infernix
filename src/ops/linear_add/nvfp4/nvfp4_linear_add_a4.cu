#include "core/weight.h"
#include "ops/linear_add/nvfp4/nvfp4_linear_add_plan.h"

#include "core/device.h"
#include "ops/linear/nvfp4/nvfp4_schedule.cuh"
#include "ops/linear/nvfp4/nvfp4_template_launch.cuh"
#include "ops/linear_add/nvfp4/nvfp4_linear_add_a4_tma_launch.h"
#include "ops/linear/common/epilogue.cuh"

#include <stdexcept>

namespace infernix::ops::detail {
namespace {

// Up to 64 tokens the N=5120 projections are one wave of 80 CTAs, so their time is the weight
// stream per CTA. A third stage keeps more of it in flight: measured on RTX 5090 (cold weights),
// the K=17408 down projection takes 45.5-47.7 us instead of 57.7-60.0 and K=6144 21.1 us instead
// of 27.2, bit-identically.
using M32N64            = Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 3, 2>;
// At 8 tokens the K=17408 down projection streams 512 K per stage over 160 CTAs: 36.6 instead of
// 37.6 us (RTX 5090, cold weights, six interleaved passes), the same K order per output. Other
// widths, K=6144 and 32-row variants with 256 K per stage measured no faster.
using M32N32K512        = Nvfp4A4MmaSchedule<32, 32, 512, 2, 2, 3, 2>;
using M32N128           = Nvfp4A4MmaSchedule<32, 128, 256, 2, 4, 2, 1>;
using M64N128           = Nvfp4A4MmaSchedule<64, 128, 256, 4, 2, 2, 1>;
using M128N128Pipelined = Nvfp4A4MmaSchedule<128, 128, 256, 4, 2, 2, 1>;
using M128N128Resident  = Nvfp4A4MmaSchedule<128, 128, 256, 4, 2, 1, 2>;

// This projection selects its own route, so the layout the quantizer writes below must be derived
// from the same predicate; the two are read together at the call site for that reason. Past 512
// tokens the TMA route is faster than the resident 128x128 tile (RTX 5090, cold weights): 513 tokens
// take 144.7 instead of 187.7 us at K=17408 and 62.0 instead of 70.9 us at K=6144, 640-1000 tokens
// 17-26 % less; at 512 and below the MMA tiles stay faster (114.0 against 133.7 us at K=17408).
// Both routes give the same bits (27B top-32 logprobs over 640-token chunks are byte-identical).
constexpr bool uses_tma(std::int32_t tokens) { return tokens > 512; }

template <class Geometry, class Schedule>
void launch_gemm(const Weight& weight, Tensor& residual, Nvfp4A4Workspace workspace,
                 std::int32_t tokens, cudaStream_t stream) {
    launch_nvfp4_a4_mma<Nvfp4ScheduleInstance<Schedule, Geometry::kInputRows>>(
        nvfp4_a4_operands(weight, workspace, tokens, Nvfp4ScaleLayout::RowMajor),
        LinearBf16Output{static_cast<__nv_bfloat16*>(residual.data), weight.n},
        LinearResidualAddEpilogue{{static_cast<__nv_bfloat16*>(residual.data), weight.n}}, stream,
        {}, pdl::Dependency::Programmatic);
}

template <class Geometry>
void launch_problem(const Weight& weight, Tensor& residual, Nvfp4A4Workspace workspace,
                    std::int32_t tokens, cudaStream_t stream) {
    if (Geometry::kInputRows == 17408 && tokens <= 8) {
        launch_gemm<Geometry, M32N32K512>(weight, residual, workspace, tokens, stream);
    } else if (tokens <= 64) {
        launch_gemm<Geometry, M32N64>(weight, residual, workspace, tokens, stream);
    } else if (tokens <= 128) {
        launch_gemm<Geometry, M32N128>(weight, residual, workspace, tokens, stream);
    } else if (tokens <= 192) {
        launch_gemm<Geometry, M64N128>(weight, residual, workspace, tokens, stream);
    } else if (tokens <= 384) {
        launch_gemm<Geometry, M128N128Resident>(weight, residual, workspace, tokens, stream);
    } else { // at most 512 tokens: larger calls take the TMA route
        launch_gemm<Geometry, M128N128Pipelined>(weight, residual, workspace, tokens, stream);
    }
}

// The A4 residual tiles, each column's activations in `Terms` planes. At K=17408 narrow rounds
// stream 512 K per stage over 160 CTAs, as the A4 route does at up to 8 tokens.
template <int Terms>
using ResidualM16N32K512 = Nvfp4A4MmaSchedule<16, 32, 512, 1, 2, 3, 2, Cache::cg, Cache::cg, Terms>;
template <int Terms>
using ResidualM16N64 = Nvfp4A4MmaSchedule<16, 64, 256, 1, 4, 3, 2, Cache::cg, Cache::cg, Terms>;
template <int Terms>
using ResidualM32N64 = Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 3, 2, Cache::cg, Cache::cg, Terms>;
template <int Terms>
using ResidualM64N64 = Nvfp4A4MmaSchedule<64, 64, 256, 4, 2, 2, 2, Cache::cg, Cache::cg, Terms>;
template <int Terms>
using ResidualM64N128 = Nvfp4A4MmaSchedule<64, 128, 256, 4, 2, 2, 1, Cache::cg, Cache::cg, Terms>;

// The 5,120 outputs give 64-row tiles only 80 CTAs per token tile: 32-token tiles keep 160 CTAs
// to 64 columns (44-46 against 59-63 us at 40-64 for 64-token tiles), 64x64 tiles to 128 (67-71
// against 94-120 us for 64x128 or 128x64), and 64x128 above (98-108 against 128-141 us at 160-256;
// K=17408, RTX 5090).
template <class Geometry, int Terms>
void launch_residual_problem(const Weight& weight, Tensor& residual, Nvfp4A4Workspace workspace,
                             std::int32_t tokens, cudaStream_t stream) {
    if (tokens <= 16) {
        if constexpr (Geometry::kInputRows == 17408) {
            launch_gemm<Geometry, ResidualM16N32K512<Terms>>(weight, residual, workspace, tokens,
                                                             stream);
        } else {
            launch_gemm<Geometry, ResidualM16N64<Terms>>(weight, residual, workspace, tokens,
                                                         stream);
        }
    } else if (tokens <= 64) {
        launch_gemm<Geometry, ResidualM32N64<Terms>>(weight, residual, workspace, tokens, stream);
    } else if (tokens <= 128) {
        launch_gemm<Geometry, ResidualM64N64<Terms>>(weight, residual, workspace, tokens, stream);
    } else {
        launch_gemm<Geometry, ResidualM64N128<Terms>>(weight, residual, workspace, tokens, stream);
    }
}

template <class Geometry>
void launch_residual_terms(const Weight& weight, Tensor& residual, Nvfp4A4Workspace workspace,
                           std::int32_t tokens, cudaStream_t stream) {
    if (workspace.terms != kNvfp4A4ResidualTerms) {
        throw std::invalid_argument("nvfp4 linear_add A4 residual: unexpected activation planes");
    }
    launch_residual_problem<Geometry, kNvfp4A4ResidualTerms>(weight, residual, workspace, tokens,
                                                             stream);
}

} // namespace

void nvfp4_linear_add_a4_residual_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                         Nvfp4A4Workspace workspace, cudaStream_t stream) {
    const std::int32_t tokens = x.ne[1];
    launch_nvfp4_a4_quantize(x, weight, workspace, Nvfp4ScaleLayout::RowMajor, stream);
    switch (resolve_nvfp4_geometry(weight.n, weight.k)) {
    case Nvfp4GeometryId::N5120K6144:
        launch_residual_terms<Nvfp4N5120K6144>(weight, residual, workspace, tokens, stream);
        return;
    case Nvfp4GeometryId::N5120K17408:
        launch_residual_terms<Nvfp4N5120K17408>(weight, residual, workspace, tokens, stream);
        return;
    case Nvfp4GeometryId::N14336K5120:
    case Nvfp4GeometryId::N16384K5120:
    case Nvfp4GeometryId::N34816K5120:
        break;
    }
    throw std::invalid_argument("nvfp4 linear_add: unsupported problem");
}

void nvfp4_linear_add_a4_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                Nvfp4A4Workspace workspace, cudaStream_t stream) {
    const std::int32_t tokens = x.ne[1];
    const auto layout = uses_tma(tokens) ? Nvfp4ScaleLayout::Tiled256 : Nvfp4ScaleLayout::RowMajor;
    launch_nvfp4_a4_quantize(x, weight, workspace, layout, stream);
    if (uses_tma(tokens)) {
        launch_nvfp4_a4_tma_linear_add(nvfp4_a4_operands(weight, workspace, tokens, layout),
                                       static_cast<__nv_bfloat16*>(residual.data), stream);
        return;
    }
    switch (resolve_nvfp4_geometry(weight.n, weight.k)) {
    case Nvfp4GeometryId::N5120K6144:
        launch_problem<Nvfp4N5120K6144>(weight, residual, workspace, tokens, stream);
        return;
    case Nvfp4GeometryId::N5120K17408:
        launch_problem<Nvfp4N5120K17408>(weight, residual, workspace, tokens, stream);
        return;
    case Nvfp4GeometryId::N14336K5120:
    case Nvfp4GeometryId::N16384K5120:
    case Nvfp4GeometryId::N34816K5120:
        break;
    }
    throw std::invalid_argument("nvfp4 linear_add: unsupported problem");
}

} // namespace infernix::ops::detail
