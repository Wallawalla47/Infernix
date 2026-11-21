// The NVFP4 attention input projection's 8-bit route (LinearPolicy::A8Only): the 8-bit tiles writing the A16
// tiles' Q, K, gate and V outputs (nvfp4_a8_mma.cuh).
#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_output.cuh"
#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.h"
#include "ops/linear/nvfp4/nvfp4_a8_mma.cuh"
#include "ops/lora/lora_device.cuh"

#include <cuda_bf16.h>

namespace infernix::ops::detail {

void nvfp4_attn_input_a8_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate, Tensor& k, Tensor& v,
                                WorkspaceArena& workspace, cudaStream_t stream, const LoraAddend* lora) {
    auto scope   = workspace.scope();
    const auto p = nvfp4_a8_encode(x, weight, workspace, stream);
    with_lora_epilogue(lora, LinearIdentityEpilogue{}, [&](auto epilogue) {
        launch_nvfp4_a8(p,
                        Nvfp4AttentionInputOutput{static_cast<__nv_bfloat16*>(q.data),
                                                  static_cast<__nv_bfloat16*>(k.data),
                                                  static_cast<__nv_bfloat16*>(gate.data),
                                                  static_cast<__nv_bfloat16*>(v.data)},
                        epilogue, stream);
    });
}

} // namespace infernix::ops::detail
