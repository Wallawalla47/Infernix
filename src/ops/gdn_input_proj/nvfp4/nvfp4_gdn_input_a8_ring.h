#pragma once

// --lora: the 8-bit ring's launchers for the GDN input projection's QKV and Z outputs with a term (narrow and wide
// adapters), defined in nvfp4_gdn_input_a8_ring.cu beside the no-adapter one (declared with the other families' in
// ops/linear/nvfp4/nvfp4_a8_ring.h). launch_nvfp4_a8 (nvfp4_a8_mma.cuh) finds them by its call's arguments; a unit that
// launches the 8-bit tiles with a term includes this header, since without these overloads the adapter epilogue would
// bind the no-adapter launcher's (sliced) epilogue parameter and the term would be lost.

#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_output.cuh"
#include "ops/linear/common/epilogue.cuh"
#include "ops/linear/nvfp4/nvfp4_schedule.cuh"
#include "ops/lora/lora_device.cuh"

#include <cuda_runtime.h>

namespace infernix::ops::detail {

struct Nvfp4A8Operands;

void launch_nvfp4_a8_ring(const Nvfp4A8Operands& p, Nvfp4GdnInputOutput output,
                          const LoraEpilogue<LinearIdentityEpilogue>& epilogue, Nvfp4IdentityRows rows,
                          cudaStream_t stream);
void launch_nvfp4_a8_ring(const Nvfp4A8Operands& p, Nvfp4GdnInputOutput output,
                          const LoraEpilogueWide<LinearIdentityEpilogue>& epilogue, Nvfp4IdentityRows rows,
                          cudaStream_t stream);

} // namespace infernix::ops::detail
