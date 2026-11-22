// The NVFP4 output and down projections' 8-bit ring (nvfp4_a8_ring.cuh) adding into the residual, in the non-RDC
// archive, which keeps its setmaxnreg.
#include "ops/linear/nvfp4/nvfp4_a8_ring.cuh"
#include "ops/linear/nvfp4/nvfp4_a8_ring.h"
#include "ops/lora/lora_device.cuh"

namespace infernix::ops::detail {

void launch_nvfp4_a8_ring(const Nvfp4A8Operands& p, LinearBf16Output output, LinearResidualAddEpilogue epilogue,
                          Nvfp4IdentityRows rows, cudaStream_t stream) {
    launch_nvfp4_a8_ring_tiles(p, output, epilogue, rows, stream);
}

void launch_nvfp4_a8_ring(const Nvfp4A8Operands& p, LinearBf16Output output,
                          const LoraEpilogue<LinearResidualAddEpilogue>& epilogue, Nvfp4IdentityRows rows,
                          cudaStream_t stream) {
    launch_nvfp4_a8_ring_tiles(p, output, epilogue, rows, stream);
}

void launch_nvfp4_a8_ring(const Nvfp4A8Operands& p, LinearBf16Output output,
                          const LoraEpilogueWide<LinearResidualAddEpilogue>& epilogue, Nvfp4IdentityRows rows,
                          cudaStream_t stream) {
    launch_nvfp4_a8_ring_tiles(p, output, epilogue, rows, stream);
}

} // namespace infernix::ops::detail
