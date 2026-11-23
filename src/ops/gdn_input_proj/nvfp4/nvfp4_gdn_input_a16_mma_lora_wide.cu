// --lora: the NVFP4 GDN input projection's A16 MMA tiles with a wide adapter's term (LoraEpilogueWide), beside the
// narrow ones' unit (nvfp4_gdn_input_a16_mma_lora.cu).
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_output.cuh"
#include "ops/lora/nvfp4_a16_mma_lora.cuh"

namespace infernix::ops::detail {
namespace {
using lora_a16::Mma;
} // namespace

template void launch_nvfp4_a16_mma_lora<Mma<64, 64, 64, 32, 16, 2, 2, 5120>, Nvfp4GdnInputOutput,
                                        LoraEpilogueWide<LinearIdentityEpilogue>, Nvfp4IdentityRows>(
    const Nvfp4A16Operands&, Nvfp4GdnInputOutput, LoraEpilogueWide<LinearIdentityEpilogue>, cudaStream_t,
    Nvfp4IdentityRows);
template void launch_nvfp4_a16_mma_lora<Mma<64, 96, 128, 64, 16, 1, 2, 5120>, Nvfp4GdnInputOutput,
                                        LoraEpilogueWide<LinearIdentityEpilogue>, Nvfp4IdentityRows>(
    const Nvfp4A16Operands&, Nvfp4GdnInputOutput, LoraEpilogueWide<LinearIdentityEpilogue>, cudaStream_t,
    Nvfp4IdentityRows);
template void launch_nvfp4_a16_mma_lora<Mma<64, 128, 64, 64, 16, 2, 2, 5120>, Nvfp4GdnInputOutput,
                                        LoraEpilogueWide<LinearIdentityEpilogue>, Nvfp4IdentityRows>(
    const Nvfp4A16Operands&, Nvfp4GdnInputOutput, LoraEpilogueWide<LinearIdentityEpilogue>, cudaStream_t,
    Nvfp4IdentityRows);

} // namespace infernix::ops::detail
