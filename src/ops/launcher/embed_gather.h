#pragma once

// infernix::ops::detail - private launch prototypes for embedding variants.

#include "core/device.h" // DeviceFault
#include "core/weight.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace infernix::ops::detail {

enum class Q8EmbedRoute {
    Auto,
    Grouped,
    Row,
};

// An id outside the table embeds zeros and reports `fault` (ops::embedding).
void embed_gather_dense_launch(const Tensor& ids, const Tensor& table, std::int32_t vocab, Tensor& out,
                               cudaStream_t stream, DeviceFault fault);
void embed_gather_q6_launch(const Tensor& ids, const Weight& table, Tensor& out,
                            cudaStream_t stream, DeviceFault fault);
void embed_gather_q8_launch(const Tensor& ids, const Weight& table, Tensor& out,
                            cudaStream_t stream, DeviceFault fault);
void embed_gather_fp8_launch(const Tensor& ids, const Weight& table, Tensor& out,
                             cudaStream_t stream, DeviceFault fault);
void embed_gather_q8_2048_launch(const Tensor& ids, const Weight& table, Tensor& out,
                                 Q8EmbedRoute route, cudaStream_t stream, DeviceFault fault = {});
const char* q8_embed_route_name(Q8EmbedRoute route);

} // namespace infernix::ops::detail
