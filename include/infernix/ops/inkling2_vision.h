#pragma once

// Inkling 2's vision tower (design docs/maintainer/inkling2_mini-design.md §15): the two elementwise Ops between its
// BF16 GEMMs, LayerNorms, GELU and D64 attention. Each row's result depends on its own inputs alone (no cross-row
// reduction), so the tower is deterministic and request-invariant; both are qualified against FP64 oracles.
//
// Rows are token-major: a row's features are contiguous, rows `row_stride` elements apart.

#include <cuda_runtime.h>

#include <cstdint>

namespace infernix::ops::inkling2 {

// The fused QKV projection's bias and the axial rotary, in place. qkv is BF16 [tokens][3 * H] (q | k | v, each H =
// heads * head_dim, heads contiguous), bias BF16 [3 * H], positions I32 [tokens][2] (row, column; the class token
// (0, 0)). For token t, feature f of q, k or v: z = qkv + bias. q and k then rotate each interleaved pair (2j, 2j + 1)
// of a head, j < head_dim / 2, by angle a_j = p_j * theta^(-(j mod R) / R), R = head_dim / 4, p_j = column for
// j < R and row otherwise:
//   (z_2j, z_2j+1) -> (z_2j cos a_j - z_2j+1 sin a_j, z_2j sin a_j + z_2j+1 cos a_j).
// v is z. Ideal values in FP64 from the represented inputs; stored once in BF16 (private arithmetic FP32).
struct VisionQkvArgs {
    std::uint16_t* qkv              = nullptr;
    const std::uint16_t* bias       = nullptr;
    const std::int32_t* positions   = nullptr;
    std::int32_t tokens             = 0;
    std::int32_t heads              = 0;
    std::int32_t head_dim           = 0; // a multiple of 4
    float theta                     = 10000.0F;
};
void vision_qkv_rotary(const VisionQkvArgs& args, cudaStream_t stream);

// A layer-scaled residual branch: x[t][d] += scale[d] * (y[t][d] + bias[d]) over tokens x width, x and y BF16
// [tokens][width] (y unchanged), scale and bias BF16 [width]. Ideal in FP64; x stored once in BF16.
struct VisionScaledResidualArgs {
    std::uint16_t* x             = nullptr;
    const std::uint16_t* y       = nullptr;
    const std::uint16_t* bias    = nullptr;
    const std::uint16_t* scale   = nullptr;
    std::int32_t tokens          = 0;
    std::int32_t width           = 0; // a multiple of 2
};
void vision_scaled_residual(const VisionScaledResidualArgs& args, cudaStream_t stream);

} // namespace infernix::ops::inkling2
