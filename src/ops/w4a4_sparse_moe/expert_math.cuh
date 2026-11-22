#pragma once

// The arithmetic the narrow and wide routes of w4a4_sparse_moe share (include/infernix/ops/w4a4_sparse_moe.h): the
// softcapped SwiGLU times the routing weight, the row scale of an NVFP4 product, and a column's selection.

#include "infernix/ops/w4a4_sparse_moe.h"
#include "ops/common/canonical_math.h"

#include <cstdint>

namespace infernix::ops::w4a4_moe::detail {

// h = bf16(((softcap(g, gate_cap) * sigmoid(g)) * softcap(u, up_cap)) * gamma), each product rounded.
__device__ __forceinline__ std::uint16_t swiglu(std::uint16_t g, std::uint16_t u, float gamma, float gate_cap,
                                                float up_cap) {
    const float gate = canon::bf16_to_f32(g), up = canon::bf16_to_f32(u);
    const float a    = canon::mul_rn(canon::softcap_c(gate, gate_cap), canon::sigmoid_c(gate));
    return canon::f32_to_bf16_rn(canon::mul_rn(canon::mul_rn(a, canon::softcap_c(up, up_cap)), gamma));
}

// The row scale of an NVFP4 product: (amax_row * amax_w) * rcp, rcp = rcp_rn(1536 * denominator).
__device__ __forceinline__ float nvfp4_rcp(float denominator) {
    return canon::div_rn(1.0F, canon::mul_rn(1536.0F, denominator));
}
__device__ __forceinline__ float nvfp4_row_scale(float amax_row, float amax_w, float rcp) {
    return canon::mul_rn(canon::mul_rn(amax_row, amax_w), rcp);
}

// An expert's selection key sigmoid_c(l) + bias, a NaN key (a NaN logit: the column's x was not finite) ranked below
// every other: the ranks stay a permutation, so every slot is filled (with NaN routing weights, which carry the
// column's non-finite state on), and finite keys keep their order.
__device__ __forceinline__ float selection_key(float sigmoid, float bias) {
    const float key = canon::add_rn(sigmoid, bias);
    return key != key ? canon::f32_from_bits(0xFF800000U) : key; // -inf
}

// One column's selection by one warp from its FP32 logits [experts] (keys and sel are the warp's shared scratch,
// 256 and kTopK floats): keys selection_key(sigmoid_c(l_e), bias_e); the top kTopK by key (ties to the lower index) in
// slots of descending key; gamma_j = (s_j / sum8(s)) * route_scale over the selected sigmoids s.
__device__ __forceinline__ void select_column(const float* logits, const float* bias, int experts, float route_scale,
                                              float* keys, float* sel, int lane, std::int32_t* routed, float* gamma) {
    for (int e = lane; e < experts; e += 32) {
        const float s = canon::sigmoid_c(__ldcg(logits + e));
        keys[e]       = selection_key(s, bias[e]);
    }
    __syncwarp();
    for (int e = lane; e < experts; e += 32) {
        const float key = keys[e];
        int rank        = 0;
        for (int f = 0; f < experts; ++f) {
            const float other = keys[f];
            rank += (other > key || (other == key && f < e)) ? 1 : 0;
        }
        if (rank < kTopK) {
            routed[rank] = e;
            sel[rank]    = canon::sigmoid_c(__ldcg(logits + e));
        }
    }
    __syncwarp();
    if (lane < kTopK) {
        const float sum = canon::add_rn(canon::add_rn(canon::add_rn(sel[0], sel[2]), canon::add_rn(sel[1], sel[3])),
                                        canon::add_rn(canon::add_rn(sel[4], sel[6]), canon::add_rn(sel[5], sel[7])));
        gamma[lane] = canon::mul_rn(canon::div_rn(sel[lane], sum), route_scale);
    }
    __syncwarp();
}

} // namespace infernix::ops::w4a4_moe::detail
