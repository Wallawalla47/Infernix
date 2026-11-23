#pragma once
#include "infernix/ops/lora.h"
#include "ops/linear/nvfp4/nvfp4_operands.h"

namespace infernix::ops::detail {
// `lora`: the call's term (nvfp4_gdn_input_plan.h), or null for the kernel as it was.
void launch_nvfp4_a4_tma_gdn(const Nvfp4A4Operands& p, __nv_bfloat16* qkv, __nv_bfloat16* z,
                             cudaStream_t stream, const LoraAddend* lora = nullptr);
} // namespace infernix::ops::detail
