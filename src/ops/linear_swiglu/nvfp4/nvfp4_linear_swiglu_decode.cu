#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_epilogue.cuh"
#include "ops/linear/nvfp4/nvfp4_template_launch.cuh"
#include "ops/lora/lora_device.cuh"

namespace infernix::ops::detail {
void nvfp4_linear_swiglu_decode_launch(const Tensor& x, const Weight& w, Tensor& y,
                                       cudaStream_t stream, const LoraAddend* lora) {
    using S = Nvfp4ScheduleInstance<
        Nvfp4A16GemvSchedule<8, 2, 16, 4, Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 2>,
        5120>;
    with_lora_epilogue(lora, Nvfp4SwiGluEpilogue{}, [&](auto epilogue) {
        launch_nvfp4_a16_gemv<S>(nvfp4_a16_operands(x, w),
                                 LinearBf16Output{static_cast<__nv_bfloat16*>(y.data), w.n / 2}, epilogue,
                                 stream, Nvfp4SwiGluRows<1>{});
    });
}
} // namespace infernix::ops::detail
