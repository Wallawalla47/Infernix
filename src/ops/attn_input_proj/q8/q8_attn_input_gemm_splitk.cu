#include "ops/linear/q8/q8_geometry.h"
#include "ops/linear/q8/q8_instances.cuh"
#include "core/weight.h"
#include "ops/attn_input_proj/q8/q8_attn_input_kernels.h"

#include "core/device.h"
#include "ops/linear/q8/q8_sliced_k_launch.cuh"
#include "ops/linear/q8/q8_grouped_sliced_k_launch.cuh"

#include <array>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace infernix::ops::detail {
namespace {

constexpr int kTargetRows             = 9216;
constexpr int kCompanionRows          = 6144;
constexpr int kHidden                 = 2048;
constexpr int kRowsPerCta             = 16;
constexpr int kFirstExactCols         = 2;
constexpr int kLastTargetExactCols    = 48;
constexpr int kLastCompanionExactCols = 32;
using TargetOutput                    = LinearBf16SegmentedOutput<4096, 512, 4096, 512>;
using CompanionOutput                 = LinearBf16SegmentedOutput<4096, 1024, 1024>;
using TargetLauncher    = void (*)(const Tensor&, const Weight&, Tensor&, Tensor&, Tensor&, Tensor&,
                                cudaStream_t);
using CompanionLauncher = void (*)(const Tensor&, const Weight&, Tensor&, Tensor&, Tensor&,
                                   cudaStream_t);

template <int ActiveCols, int Rows, class Output>
void launch_output(const Tensor& x, const Weight& weight, Output output, cudaStream_t stream) {
    constexpr int TileCols = ActiveCols <= 8    ? 8
                             : ActiveCols <= 16 ? 16
                             : ActiveCols <= 24 ? 24
                             : ActiveCols <= 32 ? 32
                             : ActiveCols <= 40 ? 40
                                                : 48;
    using Geometry         = Q8LinearGeometry<Rows, kHidden>;
    using Schedule         = Q8SlicedKDefault<TileCols, ActiveCols>;
    launch_q8_a16_sliced_k_mma<
        typename Schedule::template with_problem<Geometry::kInputRows, ActiveCols, true>>(
        q8_linear_operands(x, weight), output, LinearIdentityEpilogue{}, stream);
}

template <int ActiveCols>
void launch_target_active_cols(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                               Tensor& k, Tensor& v, cudaStream_t stream) {
    static_assert((4096 % kRowsPerCta) == 0 && (512 % kRowsPerCta) == 0);
    const TargetOutput output{
        static_cast<__nv_bfloat16*>(q.data), static_cast<__nv_bfloat16*>(k.data),
        static_cast<__nv_bfloat16*>(gate.data), static_cast<__nv_bfloat16*>(v.data)};
    launch_output<ActiveCols, kTargetRows>(x, weight, output, stream);
}

template <int ActiveCols>
void launch_companion_active_cols(const Tensor& x, const Weight& weight, Tensor& q, Tensor& k,
                                  Tensor& v, cudaStream_t stream) {
    static_assert((4096 % kRowsPerCta) == 0 && (1024 % kRowsPerCta) == 0);
    const CompanionOutput output{static_cast<__nv_bfloat16*>(q.data),
                                 static_cast<__nv_bfloat16*>(k.data),
                                 static_cast<__nv_bfloat16*>(v.data)};
    launch_output<ActiveCols, kCompanionRows>(x, weight, output, stream);
}

template <std::size_t... Offsets>
constexpr auto make_target_launchers(std::index_sequence<Offsets...>) {
    return std::array<TargetLauncher, sizeof...(Offsets)>{
        &launch_target_active_cols<kFirstExactCols + static_cast<int>(Offsets)>...};
}

template <std::size_t... Offsets>
constexpr auto make_companion_launchers(std::index_sequence<Offsets...>) {
    return std::array<CompanionLauncher, sizeof...(Offsets)>{
        &launch_companion_active_cols<kFirstExactCols + static_cast<int>(Offsets)>...};
}

constexpr auto kTargetLaunchers =
    make_target_launchers(std::make_index_sequence<kLastTargetExactCols - kFirstExactCols + 1>{});
constexpr auto kCompanionLaunchers = make_companion_launchers(
    std::make_index_sequence<kLastCompanionExactCols - kFirstExactCols + 1>{});

// The column-invariant route's calls wider than the exact instances: the T=48 schedule over every
// 48-column slice (runtime token count, one grid row per slice).
void launch_target_sliced48(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                            Tensor& k, Tensor& v, cudaStream_t stream) {
    using Geometry = Q8LinearGeometry<kTargetRows, kHidden>;
    using Schedule = Q8SlicedKDefault<48, kLastTargetExactCols>;
    const TargetOutput output{
        static_cast<__nv_bfloat16*>(q.data), static_cast<__nv_bfloat16*>(k.data),
        static_cast<__nv_bfloat16*>(gate.data), static_cast<__nv_bfloat16*>(v.data)};
    launch_q8_a16_sliced_k_mma<
        typename Schedule::template with_problem<Geometry::kInputRows, kLastTargetExactCols, false>>(
        q8_linear_operands(x, weight), output, LinearIdentityEpilogue{}, stream);
}

template <int TileCols, int KSplits, int NGroups, int MinBlocks>
void launch_target_medium_cols(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                               Tensor& k, Tensor& v, cudaStream_t stream) {
    static_assert((4096 % kRowsPerCta) == 0 && (512 % kRowsPerCta) == 0);
    const TargetOutput output{
        static_cast<__nv_bfloat16*>(q.data), static_cast<__nv_bfloat16*>(k.data),
        static_cast<__nv_bfloat16*>(gate.data), static_cast<__nv_bfloat16*>(v.data)};
    launch_q8_a16_grouped_sliced_k_mma<Q8A16GroupedSlicedKMmaSchedule<
        TileCols, KSplits, NGroups, 1, MinBlocks, kHidden, Cache::cg, Cache::cg, false>>(
        q8_linear_operands(x, weight), output, LinearIdentityEpilogue{}, stream);
}

template <int TileCols, int KSplits, int NGroups, int MinBlocks>
void launch_companion_medium_cols(const Tensor& x, const Weight& weight, Tensor& q, Tensor& k,
                                  Tensor& v, cudaStream_t stream) {
    static_assert((4096 % kRowsPerCta) == 0 && (1024 % kRowsPerCta) == 0);
    const CompanionOutput output{static_cast<__nv_bfloat16*>(q.data),
                                 static_cast<__nv_bfloat16*>(k.data),
                                 static_cast<__nv_bfloat16*>(v.data)};
    launch_q8_a16_grouped_sliced_k_mma<Q8A16GroupedSlicedKMmaSchedule<
        TileCols, KSplits, NGroups, 1, MinBlocks, kHidden, Cache::cg, Cache::cg, false>>(
        q8_linear_operands(x, weight), output, LinearIdentityEpilogue{}, stream);
}

} // namespace

void q8_attn_input_splitk_mma_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                     Tensor& k, Tensor& v, cudaStream_t stream) {
    if (x.ne[1] < kFirstExactCols || x.ne[1] > 64) {
        throw std::invalid_argument("Q8 attention input split-K MMA requires T=2..64");
    }
    if (x.ne[1] <= kLastTargetExactCols) {
        kTargetLaunchers[x.ne[1] - kFirstExactCols](x, weight, q, gate, k, v, stream);
    } else {
        launch_target_medium_cols<64, 4, 2, 2>(x, weight, q, gate, k, v, stream);
    }
    CUDA_CHECK(cudaGetLastError());
}

// Every width runs the 8 K-warp sliced-K reduction of the exact route's T=2..48 instances: T=1 its
// one-column instance, wider calls its T=48 schedule over 48-column slices. Tile, scale staging and
// token slicing change only staging and predication, so a column's bits do not depend on the call's
// width.
void q8_attn_input_column_invariant_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                           Tensor& gate, Tensor& k, Tensor& v,
                                           cudaStream_t stream) {
    const std::int32_t cols = x.ne[1];
    if (cols < 1) {
        throw std::invalid_argument("Q8 attention input column-invariant route requires T >= 1");
    }
    if (cols == 1) {
        launch_target_active_cols<1>(x, weight, q, gate, k, v, stream);
    } else if (cols <= kLastTargetExactCols) {
        kTargetLaunchers[cols - kFirstExactCols](x, weight, q, gate, k, v, stream);
    } else {
        launch_target_sliced48(x, weight, q, gate, k, v, stream);
    }
    CUDA_CHECK(cudaGetLastError());
}

void q8_attn_input_splitk_mma_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& k,
                                     Tensor& v, cudaStream_t stream) {
    if (x.ne[1] < kFirstExactCols || x.ne[1] > 96) {
        throw std::invalid_argument("Q8 companion attention input split-K MMA requires T=2..96");
    }
    if (x.ne[1] <= kLastCompanionExactCols) {
        kCompanionLaunchers[x.ne[1] - kFirstExactCols](x, weight, q, k, v, stream);
    } else if (x.ne[1] <= 48) {
        launch_companion_medium_cols<48, 4, 2, 3>(x, weight, q, k, v, stream);
    } else if (x.ne[1] <= 64) {
        launch_companion_medium_cols<64, 4, 2, 2>(x, weight, q, k, v, stream);
    } else {
        launch_companion_medium_cols<96, 2, 4, 3>(x, weight, q, k, v, stream);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace infernix::ops::detail
