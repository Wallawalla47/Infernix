#include "ops/linear_add/nvfp4/nvfp4_linear_add_a4_tma_launch.h"
#include "ops/linear/nvfp4/nvfp4_a4_tma.cuh"
#include "ops/lora/lora_device.cuh"

namespace infernix::ops::detail {
namespace {
template <int K>
void launch(const Nvfp4A4Operands& p, __nv_bfloat16* y, cudaStream_t stream, const LoraAddend* lora) {
    using S = Nvfp4ScheduleInstance<Nvfp4A4TmaMmaSchedule<256, 3, 1, kNvfp4PromptWeightPromotion>, K>;
    with_lora_epilogue(lora, LinearResidualAddEpilogue{{y, p.rows}}, [&](auto epilogue) {
        launch_nvfp4_a4_tma_mma<S>(p, LinearBf16Output{y, p.rows}, epilogue, stream);
    });
}
} // namespace

void launch_nvfp4_a4_tma_linear_add(const Nvfp4A4Operands& p, __nv_bfloat16* residual,
                                    cudaStream_t stream, const LoraAddend* lora) {
    if (p.k == 6144)
        launch<6144>(p, residual, stream, lora);
    else if (p.k == 17408)
        launch<17408>(p, residual, stream, lora);
    else
        throw std::invalid_argument("NVFP4 TMA linear_add: unsupported K");
}
} // namespace infernix::ops::detail
