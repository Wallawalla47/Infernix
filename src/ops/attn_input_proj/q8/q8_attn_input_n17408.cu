// Qwen3.5-122B-A10B's Q8 MTP attention input: the packed [17408, 3072] parent (q 8192, k 512,
// gate 8192, v 512; 32 query and 2 key/value heads of 256 over H 3072). The routes and their bounds
// follow the Qwen3.6-35B-A3B four-output parent ([9216, 2048]) at this K: the GEMV at one column,
// the 8-K-warp sliced-K instances for 2..48 columns, the grouped sliced-K tile to 64, the MMA tiles
// beyond; the column-invariant route runs the sliced-K reduction at every width.
#include "core/weight.h"
#include "ops/attn_input_proj/q8/q8_attn_input_kernels.h"

#include "core/device.h"
#include "ops/linear/q8/q8_geometry.h"
#include "ops/linear/q8/q8_gemv_launch.cuh"
#include "ops/linear/q8/q8_grouped_sliced_k_launch.cuh"
#include "ops/linear/q8/q8_instances.cuh"
#include "ops/linear/q8/q8_mma_launch.cuh"
#include "ops/linear/q8/q8_sliced_k_launch.cuh"

#include <array>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace infernix::ops::detail {
namespace {

constexpr int kRows           = 17408;
constexpr int kHidden         = 3072;
constexpr int kQueryRows      = 8192;
constexpr int kKvRows         = 512;
constexpr int kFirstExactCols = 2;
constexpr int kLastExactCols  = 48;
using Output   = LinearBf16SegmentedOutput<kQueryRows, kKvRows, kQueryRows, kKvRows>;
using Launcher = void (*)(const Tensor&, const Weight&, Tensor&, Tensor&, Tensor&, Tensor&, cudaStream_t);
static_assert(kRows == 2 * kQueryRows + 2 * kKvRows);

Output output_of(Tensor& q, Tensor& gate, Tensor& k, Tensor& v) {
    return Output{static_cast<__nv_bfloat16*>(q.data), static_cast<__nv_bfloat16*>(k.data),
                  static_cast<__nv_bfloat16*>(gate.data), static_cast<__nv_bfloat16*>(v.data)};
}

template <int ActiveCols>
void launch_active_cols(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate, Tensor& k, Tensor& v,
                        cudaStream_t stream) {
    constexpr int TileCols = ActiveCols <= 8    ? 8
                             : ActiveCols <= 16 ? 16
                             : ActiveCols <= 24 ? 24
                             : ActiveCols <= 32 ? 32
                             : ActiveCols <= 40 ? 40
                                                : 48;
    using Geometry = Q8LinearGeometry<kRows, kHidden>;
    using Schedule = Q8SlicedKDefault<TileCols, ActiveCols>;
    launch_q8_a16_sliced_k_mma<typename Schedule::template with_problem<Geometry::kInputRows, ActiveCols, true>>(
        q8_linear_operands(x, weight), output_of(q, gate, k, v), LinearIdentityEpilogue{}, stream);
}

template <std::size_t... Offsets>
constexpr auto make_launchers(std::index_sequence<Offsets...>) {
    return std::array<Launcher, sizeof...(Offsets)>{&launch_active_cols<kFirstExactCols + static_cast<int>(Offsets)>...};
}

constexpr auto kLaunchers = make_launchers(std::make_index_sequence<kLastExactCols - kFirstExactCols + 1>{});

// The column-invariant route's calls wider than the exact instances: the T=48 schedule over every
// 48-column slice (runtime token count, one grid row per slice).
void launch_sliced48(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate, Tensor& k, Tensor& v,
                     cudaStream_t stream) {
    using Geometry = Q8LinearGeometry<kRows, kHidden>;
    using Schedule = Q8SlicedKDefault<48, kLastExactCols>;
    launch_q8_a16_sliced_k_mma<typename Schedule::template with_problem<Geometry::kInputRows, kLastExactCols, false>>(
        q8_linear_operands(x, weight), output_of(q, gate, k, v), LinearIdentityEpilogue{}, stream);
}

template <class Schedule>
void launch_mma(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate, Tensor& k, Tensor& v,
                cudaStream_t stream) {
    static_assert((kQueryRows % Schedule::kBlockRows) == 0 && (kKvRows % Schedule::kBlockRows) == 0);
    launch_q8_a16_mma<Schedule>(q8_linear_operands(x, weight), output_of(q, gate, k, v), LinearIdentityEpilogue{},
                                stream);
}

} // namespace

void q8_attn_input_n17408_decode_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                        Tensor& k, Tensor& v, cudaStream_t stream) {
    constexpr int kRowsPerCta = 8;
    static_assert((kQueryRows % kRowsPerCta) == 0 && (kKvRows % kRowsPerCta) == 0);
    launch_q8_a16_gemv<Q8A16GemvSchedule<kRowsPerCta, 1, 2, kHidden>>(
        q8_linear_operands(x, weight), output_of(q, gate, k, v), LinearIdentityEpilogue{}, stream);
    CUDA_CHECK(cudaGetLastError());
}

void q8_attn_input_n17408_splitk_mma_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                            Tensor& k, Tensor& v, cudaStream_t stream) {
    if (x.ne[1] < kFirstExactCols || x.ne[1] > 64) {
        throw std::invalid_argument("Q8 attention input [17408,3072] split-K MMA requires T=2..64");
    }
    if (x.ne[1] <= kLastExactCols) {
        kLaunchers[x.ne[1] - kFirstExactCols](x, weight, q, gate, k, v, stream);
    } else {
        launch_q8_a16_grouped_sliced_k_mma<
            Q8A16GroupedSlicedKMmaSchedule<64, 4, 2, 1, 2, kHidden, Cache::cg, Cache::cg, false>>(
            q8_linear_operands(x, weight), output_of(q, gate, k, v), LinearIdentityEpilogue{}, stream);
    }
    CUDA_CHECK(cudaGetLastError());
}

void q8_attn_input_n17408_mma_r32_c128_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                              Tensor& k, Tensor& v, cudaStream_t stream) {
    launch_mma<Q8A16MmaSchedule<32, 128, 64, 32, 16, 2, 2>>(x, weight, q, gate, k, v, stream);
    CUDA_CHECK(cudaGetLastError());
}

void q8_attn_input_n17408_mma_r64_c128_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                              Tensor& k, Tensor& v, cudaStream_t stream) {
    launch_mma<Q8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>(x, weight, q, gate, k, v, stream);
    CUDA_CHECK(cudaGetLastError());
}

// Every width runs the 8 K-warp sliced-K reduction of the exact route's T=2..48 instances: T=1 its
// one-column instance, wider calls its T=48 schedule over 48-column slices, so a column's bits do not
// depend on the call's width.
void q8_attn_input_n17408_column_invariant_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                                  Tensor& k, Tensor& v, cudaStream_t stream) {
    const std::int32_t cols = x.ne[1];
    if (cols < 1) { throw std::invalid_argument("Q8 attention input column-invariant route requires T >= 1"); }
    if (cols == 1) {
        launch_active_cols<1>(x, weight, q, gate, k, v, stream);
    } else if (cols <= kLastExactCols) {
        kLaunchers[cols - kFirstExactCols](x, weight, q, gate, k, v, stream);
    } else {
        launch_sliced48(x, weight, q, gate, k, v, stream);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace infernix::ops::detail
