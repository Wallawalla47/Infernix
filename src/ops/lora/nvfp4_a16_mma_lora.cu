// --lora: the NVFP4 A16 MMA tiles' adapter instances (launch_nvfp4_a16_mma_any in
// ops/linear/nvfp4/nvfp4_template_launch.cuh). They are compiled here, apart from the translation units that launch
// the same tiles without an adapter, so those units' kernels compile as they did before the adapter existed: beside
// their adapter instances the K=6144 output projection's tile and two 5,120-input tiles compiled to other schedules
// (2026-11-21, SASS against the same sources without the adapter). Each tile a projection launches is instantiated
// here; a missing one fails at link time.

#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_output.cuh"
#include "ops/linear/common/epilogue.cuh"
#include "ops/linear/nvfp4/nvfp4_instances.cuh"
#include "ops/linear/nvfp4/nvfp4_template_launch.cuh"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_epilogue.cuh"
#include "ops/lora/lora_device.cuh"

namespace infernix::ops::detail {

template <class Schedule, class Output, class Epilogue, class Rows>
void launch_nvfp4_a16_mma_lora(const Nvfp4A16Operands& p, Output output, Epilogue epilogue, cudaStream_t stream,
                               Rows rows) {
    static_assert(kLoraEpilogue<Epilogue>, "the adapter instances only");
    launch_nvfp4_a16_mma<Schedule>(p, output, epilogue, stream, rows);
}

namespace {
template <int R, int T, int K, int WR, int WT, int S, int M, int KDim>
using Mma = Nvfp4ScheduleInstance<Nvfp4A16MmaSchedule<R, T, K, WR, WT, S, M>, KDim>;
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
