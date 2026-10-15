#pragma once

// The q4-w4a8 route of the Q4 LinearSwiGLU (ops::LinearRoutes::Q4W4A8): the activation is quantized
// to int8 with one absmax scale per token, the stored Q4 codes are sign-extended to int8, and an
// integer Tensor Core product per 64-element group is folded with the group's FP16 weight scale.

#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace infernix::ops::detail {

// Caller-owned transient activation of the route: one int8 code per input element, stored like the
// BF16 input ([K,T], K fastest), and one FP32 scale per token.
struct Q4W4A8Workspace {
    std::int8_t* codes = nullptr;
    float* scales      = nullptr;
};

// One allocation recipe for the capacity query (WorkspaceLayoutBuilder) and execution (the arena).
template <class Arena>
Q4W4A8Workspace allocate_q4_w4a8_workspace(Arena& arena, std::int32_t tokens,
                                           std::int32_t input_rows) {
    if (tokens <= 0 || input_rows <= 0) {
        throw std::invalid_argument("q4 w4a8 workspace: T and K must be positive");
    }
    const auto t = static_cast<std::size_t>(tokens);
    const auto k = static_cast<std::size_t>(input_rows);
    if (t > std::numeric_limits<std::size_t>::max() / k) {
        throw std::overflow_error("q4 w4a8 workspace size overflow");
    }
    const DeviceSpan codes  = arena.alloc_bytes(t * k, 256);
    const DeviceSpan scales = arena.alloc_bytes(t * sizeof(float), 256);
    return {static_cast<std::int8_t*>(codes.data), static_cast<float*>(scales.data)};
}

// Gate/up [34816,5120] (RowSplit Q4_G64_FP16), x [5120,T], out [17408,T]; any positive T. Uses
// allocate_q4_w4a8_workspace(ws, T, 5120) inside its own scope of `ws`.
void q4_linear_swiglu_w4a8_launch(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& ws,
                                  cudaStream_t stream);

} // namespace infernix::ops::detail
