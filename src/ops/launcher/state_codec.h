#pragma once

#include "core/tensor.h"
#include "infernix/ops/state_codec.h"

#include <cuda_runtime.h>

namespace infernix::ops::detail {

void pack_fp32_state_launch(const Tensor& source, void* packed, PackedStateFormat format,
                            cudaStream_t stream);
void unpack_fp32_state_launch(const void* packed, Tensor& destination, PackedStateFormat format,
                              cudaStream_t stream);

} // namespace infernix::ops::detail
