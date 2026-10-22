#include "ops/linear/q8/q8_geometry.h"
#include "core/weight.h"
#include "ops/dynamic_grouped_conv/q8/q8_dynamic_grouped_conv_add_kernels.h"
#include "core/device.h"
#include "ops/linear/q8/q8_schedule.cuh"
#include "ops/linear/q8/q8_launch.h"
#include "ops/linear/common/output.cuh"
#include "ops/linear/q8/q8_sliced_k_launch.cuh"
#include <cuda_bf16.h>
#include <array>
#include <algorithm>
#include <utility>
#include <stdexcept>

namespace infernix::ops::detail {
namespace {
// Hidden width (the projection's output rows) Rows, in G = Rows / 16 groups.
template <int Rows>
__device__ __forceinline__ void finish_value(int row, int col, int width, float current,
                                             float previous, const __nv_bfloat16* base,
                                             const __nv_bfloat16* delta, __nv_bfloat16* residual) {
    constexpr int kGroups = Rows / 16;
    const int index = col * Rows + row, di = col * 2 * kGroups + row / 16;
    float value = fmaf(__bfloat162float(base[2 * Rows + row]) + __bfloat162float(delta[di]),
                       current, __bfloat162float(residual[index]));
    if (col % width != 0)
        value =
            fmaf(__bfloat162float(base[3 * Rows + row]) + __bfloat162float(delta[di + kGroups]),
                 previous, value);
    residual[index] = __float2bfloat16_rn(value);
}

using Launch = Q8Launch;

template <int Rows, int InputRows, int TileColumns>
void tiled_projection(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    constexpr int Warps =
        InputRows == 4096 ? (TileColumns <= 40 ? 8 : 4) : (TileColumns <= 32 ? 8 : 4);
    constexpr Cache Activation =
        InputRows == 4096 && ((TileColumns > 24 && TileColumns <= 40) || TileColumns > 48)
            ? Cache::cg
            : Cache::ca;
    using Geometry    = Q8LinearGeometry<Rows, InputRows>;
    using Schedule    = Q8A16SlicedKMmaSchedule<TileColumns, Warps, 1, Warps == 8 ? 2 : 3,
                                                Q8ScaleAccess::Shared, Activation>;
    LinearBf16Output output{static_cast<__nv_bfloat16*>(out.data), Rows};
    launch_q8_a16_sliced_k_mma<
        typename Schedule::template with_problem<Geometry::kInputRows, TileColumns, false>,
        Q8SlicedKIdentityRows>(q8_linear_operands(x, weight), output, LinearIdentityEpilogue{},
                               stream, {}, pdl::Dependency::Programmatic);
    CUDA_CHECK(cudaGetLastError());
}

// Live columns stay dynamic; only the eight-column MMA accumulator layout is specialized.
template <int Rows, int C, std::size_t... I>
constexpr auto make_launchers(std::index_sequence<I...>) {
    return std::array<Launch, sizeof...(I)>{
        &tiled_projection<Rows, C, 8 * (1 + static_cast<int>(I))>...};
}

// The registered (H, C) pairs: Qwen3.8-27B's drafter (5120; 4096 and 17408),
// Qwen3.8-Flash-Next's (2560; 4096 and 7680) and Qwen3.5-122B-A10B's (3072; 4096 and 9216).
constexpr auto attention_5120 = make_launchers<5120, 4096>(std::make_index_sequence<11>{});
constexpr auto mlp_5120       = make_launchers<5120, 17408>(std::make_index_sequence<11>{});
constexpr auto attention_2560 = make_launchers<2560, 4096>(std::make_index_sequence<11>{});
constexpr auto mlp_2560       = make_launchers<2560, 7680>(std::make_index_sequence<11>{});
constexpr auto attention_3072 = make_launchers<3072, 4096>(std::make_index_sequence<11>{});
constexpr auto mlp_3072       = make_launchers<3072, 9216>(std::make_index_sequence<11>{});

template <int Rows>
__global__ void finish_kernel(const __nv_bfloat16* projected, const __nv_bfloat16* base,
                              const __nv_bfloat16* delta, __nv_bfloat16* residual, int width) {
    pdl::enter();
    const int row = blockIdx.x * blockDim.x + threadIdx.x, col = blockIdx.y;
    if (row >= Rows) return;
    const int index = col * Rows + row;
    finish_value<Rows>(row, col, width, __bfloat162float(projected[index]),
                       col % width ? __bfloat162float(projected[index - Rows]) : 0.0f, base, delta,
                       residual);
}

const std::array<Launch, 11>& tiled_launchers(int rows, int input_rows) {
    if (rows == 5120 && input_rows == 4096) return attention_5120;
    if (rows == 5120 && input_rows == 17408) return mlp_5120;
    if (rows == 2560 && input_rows == 4096) return attention_2560;
    if (rows == 2560 && input_rows == 7680) return mlp_2560;
    if (rows == 3072 && input_rows == 4096) return attention_3072;
    if (rows == 3072 && input_rows == 9216) return mlp_3072;
    throw std::logic_error("linear dynamic grouped conv add: unregistered (H, C)");
}

template <int Rows>
void finish(const Tensor& projected, const Tensor& base, const Tensor& delta, Tensor& residual,
            int width, int tokens, cudaStream_t stream) {
    const dim3 grid((Rows + 255) / 256, tokens);
    CUDA_CHECK(pdl::launch_consumer({grid, dim3(256), 0, stream}, finish_kernel<Rows>,
                                    static_cast<const __nv_bfloat16*>(projected.data),
                                    static_cast<const __nv_bfloat16*>(base.data),
                                    static_cast<const __nv_bfloat16*>(delta.data),
                                    static_cast<__nv_bfloat16*>(residual.data), width));
}

void materialized(Q8DynamicConvAddSchedule schedule, const Tensor& x, const Weight& weight,
                  const Tensor& base, const Tensor& delta, Tensor& residual, Tensor& projected,
                  cudaStream_t stream) {
    const int rows    = residual.ne[0];
    const int tokens  = x.ne[1] * x.ne[2];
    const Tensor flat = x.view({x.ne[0], tokens});
    Tensor result     = projected.view({rows, tokens});
    switch (schedule) {
    case Q8DynamicConvAddSchedule::TiledMma:
        tiled_launchers(rows, x.ne[0])[(tokens - 1) / 8](flat, weight, result, stream);
        break;
    case Q8DynamicConvAddSchedule::MmaK128:
        launch_q8_a16_mma_r64x32_t64_k128_a1(flat, weight, result, stream);
        break;
    }
    if (rows == 5120) {
        finish<5120>(projected, base, delta, residual, x.ne[1], tokens, stream);
    } else if (rows == 3072) {
        finish<3072>(projected, base, delta, residual, x.ne[1], tokens, stream);
    } else {
        finish<2560>(projected, base, delta, residual, x.ne[1], tokens, stream);
    }
}
} // namespace

void q8_dynamic_grouped_conv_add_materialized_launch(Q8DynamicConvAddSchedule schedule,
                                                     const Tensor& x, const Weight& weight,
                                                     const Tensor& base, const Tensor& delta,
                                                     Tensor& residual, Tensor& projected,
                                                     cudaStream_t stream) {
    materialized(schedule, x, weight, base, delta, residual, projected, stream);
}
} // namespace infernix::ops::detail
