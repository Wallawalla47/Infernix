// The NVFP4 GDN input projection's 8-bit route (LinearPolicy::A8Only): the 8-bit tiles writing the A16 tiles'
// QKV and Z outputs (nvfp4_a8_mma.cuh). The convolution forms take it materialized (nvfp4_gdn_snapshot_plan.cpp).
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_output.cuh"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.h"
#include "ops/linear/nvfp4/nvfp4_a8_mma.cuh"

#include <cuda_bf16.h>

namespace infernix::ops::detail {

void nvfp4_gdn_input_a8_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                               WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope   = workspace.scope();
    const auto p = nvfp4_a8_encode(x, weight, workspace, stream);
    launch_nvfp4_a8(p, Nvfp4GdnInputOutput{static_cast<__nv_bfloat16*>(qkv.data), static_cast<__nv_bfloat16*>(z.data)},
                    LinearIdentityEpilogue{}, stream);
}

} // namespace infernix::ops::detail
