// ops::linear's 8-bit NVFP4 ring (nvfp4_a8_ring.cuh) in the non-RDC archive, which keeps its setmaxnreg.
#include "ops/linear/nvfp4/nvfp4_a8_ring.cuh"
#include "ops/linear/nvfp4/nvfp4_a8_ring.h"

namespace infernix::ops::detail {

void launch_nvfp4_a8_ring(const Nvfp4A8Operands& p, LinearBf16Output output, LinearIdentityEpilogue epilogue,
                          Nvfp4IdentityRows rows, cudaStream_t stream) {
    launch_nvfp4_a8_ring_tiles(p, output, epilogue, rows, stream);
}

void launch_nvfp4_a8_ring(const Nvfp4A8Operands& p, LinearFp32Output output, LinearIdentityEpilogue epilogue,
                          Nvfp4IdentityRows rows, cudaStream_t stream) {
    launch_nvfp4_a8_ring_tiles(p, output, epilogue, rows, stream);
}

} // namespace infernix::ops::detail
