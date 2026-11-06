#pragma once

#include "ops/common/math.cuh"
#include "ops/common/warp.cuh"
#include <math_constants.h>

namespace infernix::ops::detail {


// Grouped scores already include attention_scale; tiled scores fold it into exp2.
__device__ __forceinline__ float causal_exp_difference(float score, float maximum, float scale) {
    return exp2_approx((score - maximum) * scale);
}

__device__ __forceinline__ float causal_exp_scaled(float score, float scaled_maximum, float scale) {
    return exp2_approx(__fmaf_rn(score, scale, -scaled_maximum));
}

// The factor that moves a running sum from the previous maximum's scaled value to the new one: the difference of the
// two rounded scaled maxima the probabilities were taken against (rounded products, never contracted into an FMA), so
// it is exactly 1 while the maximum holds. causal_exp_scaled's FMA form left 2^(the new product's rounding residual),
// which compounded over the key tiles (FlashInfer #6907; candidates UV-01).
__device__ __forceinline__ float causal_rescale(float previous, float scaled_maximum, float scale) {
    return exp2_approx(__fsub_rn(__fmul_rn(previous, scale), scaled_maximum));
}

} // namespace infernix::ops::detail
