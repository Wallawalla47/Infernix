// The NVFP4 output and down projections' 8-bit route (LinearPolicy::A8Only): the 8-bit tiles adding into the
// residual with the A16 tiles' epilogue (nvfp4_a8_mma.cuh).
#include "ops/linear/nvfp4/nvfp4_a8_mma.cuh"
#include "ops/linear_add/nvfp4/nvfp4_linear_add_plan.h"
#include "ops/lora/lora_device.cuh"

#include <cuda_bf16.h>

namespace infernix::ops::detail {

void nvfp4_linear_add_a8_launch(const Tensor& x, const Weight& weight, Tensor& residual, WorkspaceArena& workspace,
                                cudaStream_t stream, const LoraAddend* lora) {
    auto scope   = workspace.scope();
    const auto p = nvfp4_a8_encode(x, weight, workspace, stream);
    auto* data   = static_cast<__nv_bfloat16*>(residual.data);
    with_lora_epilogue(lora, LinearResidualAddEpilogue{{data, weight.n}}, [&](auto epilogue) {
        using E = decltype(epilogue);
        launch_nvfp4_a8<LinearBf16Output, E, Nvfp4IdentityRows, Nvfp4IdentityRows, true>(
            p, LinearBf16Output{data, weight.n}, epilogue, stream);
    });
}

} // namespace infernix::ops::detail
