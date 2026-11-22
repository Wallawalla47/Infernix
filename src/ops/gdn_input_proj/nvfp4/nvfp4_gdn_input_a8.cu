// The NVFP4 GDN input projection's 8-bit route (LinearPolicy::A8Only): the 8-bit tiles writing the A16 tiles'
// QKV and Z outputs (nvfp4_a8_mma.cuh). The convolution forms take it materialized (nvfp4_gdn_snapshot_plan.cpp).
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_output.cuh"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.h"
// The ring's adapter launchers, declared before the 8-bit tiles' dispatch that calls them (above 64 columns).
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_a8_ring.h"
#include "ops/linear/nvfp4/nvfp4_a8_mma.cuh"
#include "ops/lora/lora_device.cuh"

#include <cuda_bf16.h>

namespace infernix::ops::detail {

void nvfp4_gdn_input_a8_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                               WorkspaceArena& workspace, cudaStream_t stream, const LoraAddend* lora) {
    auto scope   = workspace.scope();
    const auto p = nvfp4_a8_encode(x, weight, workspace, stream);
    with_lora_epilogue(lora, LinearIdentityEpilogue{}, [&](auto epilogue) {
        launch_nvfp4_a8(p,
                        Nvfp4GdnInputOutput{static_cast<__nv_bfloat16*>(qkv.data),
                                            static_cast<__nv_bfloat16*>(z.data)},
                        epilogue, stream);
    });
}

} // namespace infernix::ops::detail
