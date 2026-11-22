// The NVFP4 attention input projection's 8-bit ring (nvfp4_a8_ring.cuh) writing the Q, K, gate and V outputs, in the
// non-RDC archive, which keeps its setmaxnreg.
#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_output.cuh"
#include "ops/linear/nvfp4/nvfp4_a8_ring.cuh"
#include "ops/linear/nvfp4/nvfp4_a8_ring.h"
#include "ops/lora/lora_device.cuh"

namespace infernix::ops::detail {

void launch_nvfp4_a8_ring(const Nvfp4A8Operands& p, Nvfp4AttentionInputOutput output, LinearIdentityEpilogue epilogue,
                          Nvfp4IdentityRows rows, cudaStream_t stream) {
    launch_nvfp4_a8_ring_tiles(p, output, epilogue, rows, stream);
}

void launch_nvfp4_a8_ring(const Nvfp4A8Operands& p, Nvfp4AttentionInputOutput output,
                          const LoraEpilogue<LinearIdentityEpilogue>& epilogue, Nvfp4IdentityRows rows,
                          cudaStream_t stream) {
    launch_nvfp4_a8_ring_tiles(p, output, epilogue, rows, stream);
}

void launch_nvfp4_a8_ring(const Nvfp4A8Operands& p, Nvfp4AttentionInputOutput output,
                          const LoraEpilogueWide<LinearIdentityEpilogue>& epilogue, Nvfp4IdentityRows rows,
                          cudaStream_t stream) {
    launch_nvfp4_a8_ring_tiles(p, output, epilogue, rows, stream);
}

} // namespace infernix::ops::detail
