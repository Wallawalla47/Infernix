#include "ops/linear/fp8/fp8_template_launch.cuh"
#include "core/weight.h"
#include "ops/linear_add/fp8/fp8_linear_add_plan.h"

#include "core/device.h"
#include "ops/linear/fp8/fp8_schedule.cuh"
#include "ops/linear/fp8/fp8_a16_gemv.cuh"
#include "ops/linear/common/epilogue.cuh"
#include "ops/lora/lora_device.cuh"

#include <cuda_bf16.h>

#include <stdexcept>

namespace infernix::ops::detail {
namespace {

template <class Geometry>
void launch(const Tensor& x, const Weight& weight, Tensor& residual, cudaStream_t stream, const LoraAddend* lora) {
    using Schedule = Fp8A16GemvSchedule<8, 2, 8, 4, Fp8CodeCache::Default, 2, 2>;
    auto* output   = static_cast<__nv_bfloat16*>(residual.data);
    const LinearBf16Output destination{output, Geometry::kOutputRows};
    with_lora_epilogue(lora, LinearResidualAddEpilogue{{output, Geometry::kOutputRows}}, [&](auto epilogue) {
        // [5120,6144] also with FP32 row words (fp8_e4m3fn_row_f32: NVIDIA's QAD 27B).
        if constexpr (Geometry::kInputRows == 6144) {
            with_fp8_a16_operands(x, weight, [&](const auto& operands) {
                launch_fp8_a16_gemv<Fp8ScheduleInstance<Schedule, Geometry::kInputRows>>(operands, destination,
                                                                                         epilogue, stream);
            });
        } else {
            launch_fp8_a16_gemv<Fp8ScheduleInstance<Schedule, Geometry::kInputRows>>(
                fp8_a16_operands(x, weight), destination, epilogue, stream);
        }
    });
}

} // namespace

void fp8_linear_add_decode_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                  cudaStream_t stream, const LoraAddend* lora) {
    switch (resolve_fp8_geometry(weight.n, weight.k)) {
    case Fp8GeometryId::N5120K6144:
        launch<Fp8N5120K6144>(x, weight, residual, stream, lora);
        return;
    case Fp8GeometryId::N5120K17408:
        launch<Fp8N5120K17408>(x, weight, residual, stream, lora);
        return;
    case Fp8GeometryId::N14336K5120:
    case Fp8GeometryId::N16384K5120:
    case Fp8GeometryId::N34816K5120:
    case Fp8GeometryId::N248320K5120:
        break;
    }
    throw std::invalid_argument("fp8 linear_add: unsupported problem");
}

} // namespace infernix::ops::detail
