// The NVFP4 gate/up's 8-bit route (LinearPolicy::A8Only): the 8-bit tiles with the A16 tiles' SwiGLU rows and
// epilogue (nvfp4_a8_mma.cuh).
#include "ops/linear/nvfp4/nvfp4_a8_mma.cuh"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_epilogue.cuh"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.h"

#include <cuda_bf16.h>

namespace infernix::ops::detail {

void nvfp4_linear_swiglu_a8_launch(const Tensor& x, const Weight& weight, Tensor& out, WorkspaceArena& workspace,
                                   cudaStream_t stream) {
    auto scope   = workspace.scope();
    const auto p = nvfp4_a8_encode(x, weight, workspace, stream);
    launch_nvfp4_a8(p, LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), weight.n / 2}, Nvfp4SwiGluEpilogue{},
                    stream, Nvfp4SwiGluRows<8>{}, Nvfp4SwiGluMmaRows{});
}

} // namespace infernix::ops::detail
