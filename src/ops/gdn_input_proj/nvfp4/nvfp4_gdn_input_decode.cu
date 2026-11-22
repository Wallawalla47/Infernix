#include "core/weight.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.h"

#include "core/device.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_output.cuh"
#include "ops/linear/nvfp4/nvfp4_schedule.cuh"
#include "ops/linear/nvfp4/nvfp4_template_launch.cuh"
#include "ops/lora/lora_device.cuh"

namespace infernix::ops::detail {

void nvfp4_gdn_input_decode_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                   cudaStream_t stream, const LoraAddend* lora) {
    using Geometry = Nvfp4N16384K5120;
    using Schedule =
        Nvfp4A16GemvSchedule<8, 2, 16, 4, Nvfp4ScaleAccess::StagedRaw, Nvfp4CodeCache::Default, 2>;

    const Nvfp4GdnInputOutput output{static_cast<__nv_bfloat16*>(qkv.data),
                                     static_cast<__nv_bfloat16*>(z.data)};
    with_lora_epilogue(lora, LinearIdentityEpilogue{}, [&](auto epilogue) {
        launch_nvfp4_a16_gemv<Nvfp4ScheduleInstance<Schedule, Geometry::kInputRows>>(nvfp4_a16_operands(x, weight),
                                                                                     output, epilogue, stream);
    });
}

} // namespace infernix::ops::detail
