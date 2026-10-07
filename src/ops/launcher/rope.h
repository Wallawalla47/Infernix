#pragma once

// infernix::ops::detail - private launch prototype for rope. Included by the wrapper
// and defined by the CUDA launcher.

#include "core/device.h"
#include "infernix/ops/rope.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

namespace infernix::ops::detail {

void rope_launch(const Tensor& positions, int rotary_dim, float theta, Tensor& q, Tensor& k,
                 DeviceExecutionView execution);

void rope_single_launch(const Tensor& positions, int rotary_dim, float theta, Tensor& x,
                        DeviceExecutionView execution);
void rope_prepared_launch(const Tensor& positions, const PreparedRope& prepared, Tensor& q,
                          Tensor* k, DeviceExecutionView execution);

} // namespace infernix::ops::detail
