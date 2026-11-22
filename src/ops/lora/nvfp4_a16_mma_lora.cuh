#pragma once

// --lora: the NVFP4 A16 MMA tiles' adapter instances (launch_nvfp4_a16_mma_any in
// ops/linear/nvfp4/nvfp4_template_launch.cuh), instantiated in units of their own, apart from the translation units
// that launch the same tiles without an adapter, so those units' kernels compile as they did before the adapter
// existed: beside their adapter instances the K=6144 output projection's tile and two 5,120-input tiles compiled to
// other schedules (2026-11-21, SASS against the same sources without the adapter). nvfp4_a16_mma_lora.cu holds the
// narrow adapters' instances (LoraEpilogue), nvfp4_a16_mma_lora_wide.cu the wide ones' (LoraEpilogueWide); each tile a
// projection launches is instantiated in both, and a missing one fails at link time.

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

namespace lora_a16 {
template <int R, int T, int K, int WR, int WT, int S, int M, int KDim>
using Mma = Nvfp4ScheduleInstance<Nvfp4A16MmaSchedule<R, T, K, WR, WT, S, M>, KDim>;
} // namespace lora_a16

} // namespace infernix::ops::detail
