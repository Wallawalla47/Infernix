#pragma once

// Two-term NVFP4 queries for the block-scaled FP4 QK of the NVFP4 KV cache (small-T and prompt
// kernels). A rotated D256 Q row, pre-scaled by an FP32 row factor, is encoded as the group-16
// NVFP4 term of the row plus the group-16 NVFP4 term of what the first leaves; the FP4 MMA runs
// once per term against the stored K codes and scales, and the row factor multiplies the FP32
// scores. On rotated rows of the 27B model the two terms represent Q to 0.9 % RMS (a row-scaled
// E4M3 Q: 2.6 %; one NVFP4 term: 9.5 %).

#include "ops/kv_cache/nvfp4_group16_codec.cuh"

#include <cuda_fp4.h>
#include <cuda_fp8.h>

#include <cstdint>

namespace ninfer::ops {

// The NVFP4 Q row factor maps the row's largest magnitude to E2M1 6 at UE4M3 448.
inline constexpr float kCausalNvfp4QTop = 6.0F * 448.0F;

// One group-16 NVFP4 term of the lane's value x, dimension 32 r + lane of a D256 row whose codes
// start at `codes` (128 bytes, 16-byte chunks XOR-swizzled by row & 7) and whose 16 group scales
// start at `scales`. Lanes 0-15 and 16-31 hold groups 2r and 2r+1. Returns the represented value.
// The term only has to approximate Q (the second term absorbs what the first misses), so it uses
// fast reciprocals and no branches, letting the eight dimensions of a lane interleave.
__device__ __forceinline__ float causal_nvfp4_q_term(float x, int lane, int row, int r,
                                                     std::uint8_t* codes, std::uint8_t* scales) {
    constexpr unsigned FullMask = 0xffffffffU;
    float amax                  = fabsf(x);
#pragma unroll
    for (int offset = 8; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(FullMask, amax, offset));
    }
    // An all-zero group keeps the minimum scale; its codes are zero either way.
    const float bounded =
        fminf(kKVCacheNvfp4ScaleMaximum,
              fmaxf(kKVCacheNvfp4ScaleMinimum, amax * (1.0F / kKVCacheNvfp4MaxFinite)));
    const std::uint8_t scale_code = __nv_cvt_float_to_fp8(bounded, __NV_SATFINITE, __NV_E4M3);
    const float scale             = detail::decode_nvfp4_e4m3(scale_code);
    const float quotient          = __fdividef(x, scale);
    const float partner           = __shfl_xor_sync(FullMask, quotient, 1);
    const bool odd                = (lane & 1) != 0;
    const auto byte               = static_cast<std::uint8_t>(__nv_cvt_float2_to_fp4x2(
        odd ? make_float2(partner, quotient) : make_float2(quotient, partner), __NV_E2M1,
        cudaRoundNearest));
    if (!odd) { codes[((r ^ (row & 7)) << 4) | (lane >> 1)] = byte; }
    if ((lane & 15) == 0) { scales[2 * r + (lane >> 4)] = scale_code; }
    const float2 pair = detail::decode_nvfp4_e2m1x2(byte);
    return (odd ? pair.y : pair.x) * scale;
}

} // namespace ninfer::ops
