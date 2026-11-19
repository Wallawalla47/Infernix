// The NVFP4 gate/up's 8-bit ring (nvfp4_a8_ring.cuh) with the A16 tiles' SwiGLU rows and epilogue, in the non-RDC
// archive, which keeps its setmaxnreg.
#include "ops/linear/nvfp4/nvfp4_a8_ring.cuh"
#include "ops/linear/nvfp4/nvfp4_a8_ring.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_epilogue.cuh"

namespace infernix::ops::detail {

void launch_nvfp4_a8_ring(const Nvfp4A8Operands& p, LinearBf16Output output, Nvfp4SwiGluEpilogue epilogue,
                          Nvfp4SwiGluMmaRows rows, cudaStream_t stream) {
    launch_nvfp4_a8_ring_tiles(p, output, epilogue, rows, stream);
}

} // namespace infernix::ops::detail
