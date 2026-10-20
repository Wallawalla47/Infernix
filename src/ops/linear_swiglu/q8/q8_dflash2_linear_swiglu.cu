#include "ops/linear/q8/q8_geometry.h"
#include "core/weight.h"
#include "ops/linear_swiglu/q8/q8_linear_swiglu_kernels.h"

#include "core/device.h"
#include "ops/linear/q8/q8_schedule.cuh"
#include "ops/linear/common/output.cuh"
#include "ops/linear/q8/q8_sliced_k_launch.cuh"
#include "ops/linear_swiglu/q8/q8_linear_swiglu_output.cuh"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace infernix::ops::detail {
namespace {

// The DFlash2 drafters' gate/up parents: [34816, 5120] (Qwen3.8-27B) and [15360, 2560]
// (Qwen3.8-Flash-Next).
constexpr std::int32_t kFirstSmallT = 1;
constexpr std::int32_t kLastSmallT  = 40;
using Launch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

template <int GateUpRows, int Hidden, int Capacity>
void launch_tile(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    using Geometry                       = Q8LinearGeometry<GateUpRows, Hidden>;
    constexpr std::int32_t kIntermediate = Geometry::kOutputRows / 2;
    using Schedule =
        Q8A16SlicedKMmaSchedule<Capacity, Capacity == 24 ? 8 : 4, 1, 2, Q8ScaleAccess::Shared>;
    using RowPolicy = Q8SwiGluPairedRows<kIntermediate>;
    static_assert((Geometry::kInputRows % Schedule::kBlockK) == 0);
    static_assert((kIntermediate % RowPolicy::kOutputRowsPerCta) == 0);

    const LinearBf16Output ignored_output{static_cast<__nv_bfloat16*>(out.data), kIntermediate};
    const Q8SwiGluDirectEpilogue epilogue{};

    launch_q8_a16_sliced_k_mma<
        typename Schedule::template with_problem<Geometry::kInputRows, Capacity, false>, RowPolicy>(
        q8_linear_operands(x, weight), ignored_output, epilogue, stream, RowPolicy{},
        pdl::Dependency::Programmatic);
    CUDA_CHECK(cudaGetLastError());
}

template <int GateUpRows, int Hidden, std::size_t... Offsets>
constexpr auto make_launchers(std::index_sequence<Offsets...>) {
    return std::array<Launch, sizeof...(Offsets)>{
        &launch_tile<GateUpRows, Hidden, 8 * (1 + static_cast<int>(Offsets))>...};
}

constexpr auto k27BLaunchers =
    make_launchers<34816, 5120>(std::make_index_sequence<kLastSmallT / 8>{});
constexpr auto kFlashNextLaunchers =
    make_launchers<15360, 2560>(std::make_index_sequence<kLastSmallT / 8>{});

} // namespace

void q8_dflash2_linear_swiglu_small_t_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                             cudaStream_t stream) {
    if (x.ne[1] < kFirstSmallT || x.ne[1] > kLastSmallT) {
        throw std::invalid_argument("Q8 DFlash2 LinearSwiGLU small-T: unsupported T");
    }
    const std::size_t index = static_cast<std::size_t>((x.ne[1] - 1) / 8);
    if (x.ne[0] == 5120) {
        k27BLaunchers[index](x, weight, out, stream);
    } else if (x.ne[0] == 2560) {
        kFlashNextLaunchers[index](x, weight, out, stream);
    } else {
        throw std::invalid_argument("Q8 DFlash2 LinearSwiGLU small-T: unsupported H");
    }
}

} // namespace infernix::ops::detail
