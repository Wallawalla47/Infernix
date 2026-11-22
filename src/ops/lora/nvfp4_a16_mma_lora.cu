// --lora: the NVFP4 A16 MMA tiles' narrow adapter instances (nvfp4_a16_mma_lora.cuh).
#include "ops/lora/nvfp4_a16_mma_lora.cuh"

namespace infernix::ops::detail {
namespace {
using lora_a16::Mma;
} // namespace

// linear_add (nvfp4_linear_add_a16.cu): the output projection (K 6144) and down projection (K 17408).
template void launch_nvfp4_a16_mma_lora<Mma<64, 64, 128, 32, 16, 2, 1, 6144>, LinearBf16Output,
                                        LoraEpilogue<LinearResidualAddEpilogue>, Nvfp4IdentityRows>(
    const Nvfp4A16Operands&, LinearBf16Output, LoraEpilogue<LinearResidualAddEpilogue>, cudaStream_t,
    Nvfp4IdentityRows);
template void launch_nvfp4_a16_mma_lora<Mma<64, 128, 64, 64, 16, 2, 2, 6144>, LinearBf16Output,
                                        LoraEpilogue<LinearResidualAddEpilogue>, Nvfp4IdentityRows>(
    const Nvfp4A16Operands&, LinearBf16Output, LoraEpilogue<LinearResidualAddEpilogue>, cudaStream_t,
    Nvfp4IdentityRows);
template void launch_nvfp4_a16_mma_lora<Mma<32, 64, 128, 32, 16, 2, 2, 17408>, LinearBf16Output,
                                        LoraEpilogue<LinearResidualAddEpilogue>, Nvfp4IdentityRows>(
    const Nvfp4A16Operands&, LinearBf16Output, LoraEpilogue<LinearResidualAddEpilogue>, cudaStream_t,
    Nvfp4IdentityRows);
template void launch_nvfp4_a16_mma_lora<Mma<64, 128, 64, 64, 16, 2, 2, 17408>, LinearBf16Output,
                                        LoraEpilogue<LinearResidualAddEpilogue>, Nvfp4IdentityRows>(
    const Nvfp4A16Operands&, LinearBf16Output, LoraEpilogue<LinearResidualAddEpilogue>, cudaStream_t,
    Nvfp4IdentityRows);

// attn_input_proj (nvfp4_attn_input_a16.cu): the single parent's q, k, gate and v.
template void launch_nvfp4_a16_mma_lora<Mma<32, 32, 128, 16, 16, 2, 2, 5120>, Nvfp4AttentionInputOutput,
                                        LoraEpilogue<LinearIdentityEpilogue>, Nvfp4IdentityRows>(
    const Nvfp4A16Operands&, Nvfp4AttentionInputOutput, LoraEpilogue<LinearIdentityEpilogue>, cudaStream_t,
    Nvfp4IdentityRows);
template void launch_nvfp4_a16_mma_lora<Mma<32, 64, 128, 16, 16, 1, 3, 5120>, Nvfp4AttentionInputOutput,
                                        LoraEpilogue<LinearIdentityEpilogue>, Nvfp4IdentityRows>(
    const Nvfp4A16Operands&, Nvfp4AttentionInputOutput, LoraEpilogue<LinearIdentityEpilogue>, cudaStream_t,
    Nvfp4IdentityRows);
template void launch_nvfp4_a16_mma_lora<Mma<64, 64, 64, 32, 16, 2, 2, 5120>, Nvfp4AttentionInputOutput,
                                        LoraEpilogue<LinearIdentityEpilogue>, Nvfp4IdentityRows>(
    const Nvfp4A16Operands&, Nvfp4AttentionInputOutput, LoraEpilogue<LinearIdentityEpilogue>, cudaStream_t,
    Nvfp4IdentityRows);
template void launch_nvfp4_a16_mma_lora<Mma<64, 128, 64, 64, 16, 2, 2, 5120>, Nvfp4AttentionInputOutput,
                                        LoraEpilogue<LinearIdentityEpilogue>, Nvfp4IdentityRows>(
    const Nvfp4A16Operands&, Nvfp4AttentionInputOutput, LoraEpilogue<LinearIdentityEpilogue>, cudaStream_t,
    Nvfp4IdentityRows);

// linear_swiglu (nvfp4_linear_swiglu_small_t.cu): the gate/up projection above 32 columns.
template void launch_nvfp4_a16_mma_lora<Mma<32, 64, 128, 16, 16, 1, 3, 5120>, LinearBf16Output,
                                        LoraEpilogue<Nvfp4SwiGluEpilogue>, Nvfp4SwiGluMmaRows>(
    const Nvfp4A16Operands&, LinearBf16Output, LoraEpilogue<Nvfp4SwiGluEpilogue>, cudaStream_t, Nvfp4SwiGluMmaRows);

} // namespace infernix::ops::detail
