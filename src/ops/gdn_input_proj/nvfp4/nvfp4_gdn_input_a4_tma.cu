#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_a4_tma_launch.h"
#include "ops/linear/nvfp4/nvfp4_a4_tma.cuh"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_output.cuh"
#include "ops/lora/lora_device.cuh"

namespace infernix::ops::detail {
void launch_nvfp4_a4_tma_gdn(const Nvfp4A4Operands& p, __nv_bfloat16* qkv, __nv_bfloat16* z,
                             cudaStream_t stream, const LoraAddend* lora) {
    with_lora_epilogue(lora, LinearIdentityEpilogue{}, [&](auto epilogue) {
        launch_nvfp4_a4_tma_mma<
            Nvfp4ScheduleInstance<Nvfp4A4TmaMmaSchedule<256, 3, 1, kNvfp4PromptWeightPromotion>, 5120>>(
            p, Nvfp4GdnInputOutput{qkv, z}, epilogue, stream);
    });
}
} // namespace infernix::ops::detail
