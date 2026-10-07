#pragma once
#include "ops/common/dflash_rope.cuh"
#include "core/pdl.cuh"
#include "ops/kernel/rope.cuh"
#include "ops/rmsnorm_rope/d128.cuh"
#include "ops/rmsnorm_rope/d256.cuh"
#include <cuda_bf16.h>
#include <cstdint>

namespace infernix::ops {
// A CTA owns eight heads of one token. Pair form uses four Q CTAs and one K CTA;
// the single-K form uses one CTA. Each warp evaluates one complete head.
template <bool Pair, class Coefficients = NativeRopeCoefficients>
__global__ __launch_bounds__(256) void rmsnorm_rope_d128_kernel(
    const std::int32_t* __restrict__ positions, const __nv_bfloat16* __restrict__ q_norm,
    const __nv_bfloat16* __restrict__ k_norm, __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k, Coefficients coefficients = {}) {
    pdl::enter();
    constexpr int kPairs = 64;
    const int token      = blockIdx.x;
    const bool query     = Pair && blockIdx.y < 4;
    const int head =
        (query ? static_cast<int>(blockIdx.y) * 8 : 0) + static_cast<int>(threadIdx.x) / 32;
    const int lane     = threadIdx.x % 32;
    auto* data         = reinterpret_cast<__nv_bfloat162*>(query ? q : k);
    const auto* weight = reinterpret_cast<const __nv_bfloat162*>(query ? q_norm : k_norm);
    __shared__ float cos_cache[kPairs];
    __shared__ float sin_cache[kPairs];
    __shared__ __nv_bfloat162 weight_cache[kPairs];
    if (threadIdx.x < kPairs) {
        const int pair = threadIdx.x;
        coefficients.template sincos<RopeKernelMode::DflashText1D>(
            positions, 0, token, pair, &sin_cache[pair], &cos_cache[pair]);
        weight_cache[pair] = weight[pair];
    }
    __syncthreads();
    const std::int64_t base = (static_cast<std::int64_t>(token) * (query ? 32 : 8) + head) * kPairs;
    const auto out    = detail::rmsnorm_rope_d128_head(data[base + lane], data[base + lane + 32],
                                                       weight_cache[lane], weight_cache[lane + 32],
                                                       cos_cache, sin_cache, lane);
    data[base + lane] = out.first;
    data[base + lane + 32] = out.second;
}

// Text form: D=256 heads, rotary width 64, out of place. One warp owns one head; HeadsPerBlock
// warps share a block. The Q and K heads of one token are laid out as one combined range so a
// single grid covers both tensors and no head group is left half empty.
template <int QHeads, int KHeads, int HeadsPerBlock>
__global__ __launch_bounds__(HeadsPerBlock * 32) void rmsnorm_rope_d256_text_kernel(
    const std::int32_t* __restrict__ positions, const __nv_bfloat162* __restrict__ q_norm,
    const __nv_bfloat162* __restrict__ k_norm, const __nv_bfloat162* __restrict__ q_in,
    const __nv_bfloat162* __restrict__ k_in, __nv_bfloat162* __restrict__ q_out,
    __nv_bfloat162* __restrict__ k_out, std::int32_t tokens) {
    pdl::enter();
    constexpr int kPairs    = 128;
    constexpr int kHalfPair = 16;
    constexpr int kCombined = QHeads + KHeads;
    constexpr int kGroups   = (kCombined + HeadsPerBlock - 1) / HeadsPerBlock;

    const int token = static_cast<int>(blockIdx.x) / kGroups;
    if (token >= tokens) { return; }
    const int group    = static_cast<int>(blockIdx.x) % kGroups;
    const int lane     = static_cast<int>(threadIdx.x) & 31;
    const int warp     = static_cast<int>(threadIdx.x) >> 5;
    const int combined = group * HeadsPerBlock + warp;
    if (combined >= kCombined) { return; }

    const bool query                          = combined < QHeads;
    const int head                            = query ? combined : combined - QHeads;
    const int heads                           = query ? QHeads : KHeads;
    const __nv_bfloat162* __restrict__ input  = query ? q_in : k_in;
    const __nv_bfloat162* __restrict__ weight = query ? q_norm : k_norm;
    __nv_bfloat162* __restrict__ output       = query ? q_out : k_out;

    const std::int64_t base = (static_cast<std::int64_t>(token) * heads + head) * kPairs;
    const auto normalized   = detail::rmsnorm_rope_d256_normalize(input, weight, base, lane);
#pragma unroll
    for (int k = 1; k < 4; ++k) { output[base + lane + k * 32] = normalized.pair[k]; }

    // Lanes 16-31 rotate with the same two coefficient pairs as lanes 0-15. Each half computes one
    // of the two and the partners swap them, so every lane runs one sincos instead of two.
    const bool upper           = lane >= kHalfPair;
    const int coefficient_pair = (lane & (kHalfPair - 1)) * 2 + (upper ? 1 : 0);
    float s = 0.0F, c = 0.0F;
    fixed_sincos<RopeKernelMode::Text1D>(positions, tokens, token, coefficient_pair, &s, &c);
    const float s_partner = __shfl_xor_sync(0xffffffffu, s, kHalfPair);
    const float c_partner = __shfl_xor_sync(0xffffffffu, c, kHalfPair);
    const float s0        = upper ? s_partner : s;
    const float c0        = upper ? c_partner : c;
    const float s1        = upper ? s : s_partner;
    const float c1        = upper ? c : c_partner;
    output[base + lane] =
        detail::rmsnorm_rope_d256_rotate(normalized.pair[0], c0, c1, s0, s1, lane);
}

} // namespace infernix::ops
