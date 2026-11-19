// ops::linear's 8-bit route of the registered NVFP4 shapes (LinearPolicy::A8Only): the call's columns encoded by
// ops/ggml/ggml_a8.h's codec, then nvfp4_a8_mma.cuh's tiles with the identity epilogue, FP32 or BF16 out.
#include "ops/linear/nvfp4/nvfp4_a8_mma.cuh"
#include "ops/linear/nvfp4/nvfp4_dispatch.h"
#include "ops/linear/common/output.cuh"

namespace infernix::ops::detail {

void nvfp4_linear_a8_launch(const Tensor& x, const Weight& weight, Tensor& out, WorkspaceArena& workspace,
                            cudaStream_t stream) {
    auto scope        = workspace.scope();
    const auto p      = nvfp4_a8_encode(x, weight, workspace, stream);
    with_linear_output(out, weight.n, [&](auto output) { launch_nvfp4_a8(p, output, LinearIdentityEpilogue{}, stream); });
}

} // namespace infernix::ops::detail
