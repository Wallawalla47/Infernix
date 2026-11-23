// --lora: the NVFP4 GDN input projection's A16 MMA tiles (nvfp4_gdn_input_a16.cu, 33 columns and wider) with a narrow
// adapter's term (LoraEpilogue), in a unit of their own (launch_nvfp4_a16_mma_any, ops/lora/nvfp4_a16_mma_lora.cuh);
// nvfp4_gdn_input_a16_mma_lora_wide.cu holds the wide adapters' instances.
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_output.cuh"
#include "ops/lora/nvfp4_a16_mma_lora.cuh"

namespace infernix::ops::detail {
namespace {
using lora_a16::Mma;
} // namespace

template void launch_nvfp4_a16_mma_lora<Mma<64, 64, 64, 32, 16, 2, 2, 5120>, Nvfp4GdnInputOutput,
                                        LoraEpilogue<LinearIdentityEpilogue>, Nvfp4IdentityRows>(
    const Nvfp4A16Operands&, Nvfp4GdnInputOutput, LoraEpilogue<LinearIdentityEpilogue>, cudaStream_t,
    Nvfp4IdentityRows);
template void launch_nvfp4_a16_mma_lora<Mma<64, 96, 128, 64, 16, 1, 2, 5120>, Nvfp4GdnInputOutput,
                                        LoraEpilogue<LinearIdentityEpilogue>, Nvfp4IdentityRows>(
    const Nvfp4A16Operands&, Nvfp4GdnInputOutput, LoraEpilogue<LinearIdentityEpilogue>, cudaStream_t,
    Nvfp4IdentityRows);
template void launch_nvfp4_a16_mma_lora<Mma<64, 128, 64, 64, 16, 2, 2, 5120>, Nvfp4GdnInputOutput,
                                        LoraEpilogue<LinearIdentityEpilogue>, Nvfp4IdentityRows>(
    const Nvfp4A16Operands&, Nvfp4GdnInputOutput, LoraEpilogue<LinearIdentityEpilogue>, cudaStream_t,
    Nvfp4IdentityRows);

} // namespace infernix::ops::detail
