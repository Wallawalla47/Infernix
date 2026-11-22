// The NVFP4 GDN input projection's 8-bit ring (nvfp4_a8_ring.cuh) writing the QKV and Z outputs, in the non-RDC
// archive, which keeps its setmaxnreg; without and with a LoRA term (nvfp4_gdn_input_a8_ring.h).
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_output.cuh"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_a8_ring.h"
#include "ops/linear/nvfp4/nvfp4_a8_ring.cuh"
#include "ops/linear/nvfp4/nvfp4_a8_ring.h"
#include "ops/lora/lora_device.cuh"

namespace infernix::ops::detail {

void launch_nvfp4_a8_ring(const Nvfp4A8Operands& p, Nvfp4GdnInputOutput output, LinearIdentityEpilogue epilogue,
                          Nvfp4IdentityRows rows, cudaStream_t stream) {
    launch_nvfp4_a8_ring_tiles(p, output, epilogue, rows, stream);
}

void launch_nvfp4_a8_ring(const Nvfp4A8Operands& p, Nvfp4GdnInputOutput output,
                          const LoraEpilogue<LinearIdentityEpilogue>& epilogue, Nvfp4IdentityRows rows,
                          cudaStream_t stream) {
    launch_nvfp4_a8_ring_tiles(p, output, epilogue, rows, stream);
}

void launch_nvfp4_a8_ring(const Nvfp4A8Operands& p, Nvfp4GdnInputOutput output,
                          const LoraEpilogueWide<LinearIdentityEpilogue>& epilogue, Nvfp4IdentityRows rows,
                          cudaStream_t stream) {
    launch_nvfp4_a8_ring_tiles(p, output, epilogue, rows, stream);
}

} // namespace infernix::ops::detail
