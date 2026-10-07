#pragma once

#include "ops/common/warp.cuh"
#include "ops/kv_cache/nvfp4_group16_codec.cuh"

namespace infernix::ops::detail {
// NVFP4-G16 V (K8V4 and NVFP4 KV) for the MXFP8 tiled kernel: packed E2M1 codes and one UE4M3
// scale per 16 dimensions.
struct Nvfp4G16TiledValues {
    using Scale                       = std::uint8_t;
    static constexpr int kCodeBytes   = 128;
    static constexpr int kScaleItems  = 16;
    static constexpr bool kE4m3Values = true;

    // A 64-key FP16 PV partial is bounded by 64 * 6 * max_scale. UE4M3 codes order like their
    // values and 0x70 encodes exactly 128, so a tile whose largest scale is at most 128 stays
    // under 49152; larger scales are (128, 256) or [256, 448], and one or two halvings bring
    // them back to 128. The 64 keys' 16 scale bytes are 32 bytes per lane.
    template <int Keys>
    __device__ __forceinline__ static int tile_shift(const Scale* scales, int lane) {
        static_assert(Keys * kScaleItems == 32 * 32);
        const uint4 s0  = load_vec<uint4>(scales + 32 * lane);
        const uint4 s1  = load_vec<uint4>(scales + 32 * lane + 16);
        unsigned code   = 0;
        const auto fold = [&](unsigned w) {
            code =
                max(code, max(max(w & 0xFFU, (w >> 8) & 0xFFU), max((w >> 16) & 0xFFU, w >> 24)));
        };
        fold(s0.x);
        fold(s0.y);
        fold(s0.z);
        fold(s0.w);
        fold(s1.x);
        fold(s1.y);
        fold(s1.z);
        fold(s1.w);
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            code = max(code, __shfl_xor_sync(0xffffffffU, code, offset));
        }
        return code <= 0x70U ? 0 : (code < 0x78U ? 1 : 2);
    }

    // The 8-bit PV form: the largest scale lies in [2^e, 2^(e+1)) for UE4M3 exponent field e
    // (subnormal scales sit below 2^-6); dividing by 2^(max(e, 1) - 12) puts it in [32, 64), so a
    // decoded value stays within 6 * 64 = 384, inside E4M3.
    template <int Keys>
    __device__ __forceinline__ static int tile_shift_e4m3(const Scale* scales, int lane) {
        static_assert(Keys * kScaleItems == 32 * 32);
        const uint4 s0  = load_vec<uint4>(scales + 32 * lane);
        const uint4 s1  = load_vec<uint4>(scales + 32 * lane + 16);
        unsigned code   = 0;
        const auto fold = [&](unsigned w) {
            code =
                max(code, max(max(w & 0xFFU, (w >> 8) & 0xFFU), max((w >> 16) & 0xFFU, w >> 24)));
        };
        fold(s0.x);
        fold(s0.y);
        fold(s0.z);
        fold(s0.w);
        fold(s1.x);
        fold(s1.y);
        fold(s1.z);
        fold(s1.w);
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            code = max(code, __shfl_xor_sync(0xffffffffU, code, offset));
        }
        return static_cast<int>(max(code >> 3, 1U)) - 12;
    }

    // Eight values as E4M3 bytes: exact FP16 products (as expand) rounded once to E4M3.
    __device__ __forceinline__ static uint2 expand_e4m3(const std::uint8_t* codes, Scale scale,
                                                        __half mul) {
        const int4 h    = expand(codes, scale, mul);
        const auto pair = [](int bits) {
            __half2_raw raw;
            raw.x = static_cast<unsigned short>(static_cast<unsigned>(bits) & 0xFFFFU);
            raw.y = static_cast<unsigned short>(static_cast<unsigned>(bits) >> 16);
            return static_cast<unsigned>(
                __nv_cvt_halfraw2_to_fp8x2(raw, __NV_SATFINITE, __NV_E4M3));
        };
        return make_uint2(pair(h.x) | (pair(h.y) << 16), pair(h.z) | (pair(h.w) << 16));
    }

    // `mul` is the tile's exact power-of-two shift (at least 1/4): every E2M1 code times a
    // legal UE4M3 scale divided by it remains exact in FP16.
    __device__ __forceinline__ static int4 expand(const std::uint8_t* codes, Scale scale,
                                                  __half mul) {
        const std::uint32_t packed = load_vec<std::uint32_t>(codes);
        const std::uint8_t* bytes  = reinterpret_cast<const std::uint8_t*>(&packed);
        __nv_fp8_e4m3 encoded_scale;
        encoded_scale.__x    = scale;
        const __half value   = __hmul(static_cast<__half>(encoded_scale), mul);
        const __half2 scale2 = __halves2half2(value, value);
        unsigned half_bits[4];
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            __nv_fp4x2_e2m1 encoded;
            encoded.__x     = bytes[pair];
            const __half2 v = __hmul2(static_cast<__half2>(encoded), scale2);
            half_bits[pair] = *reinterpret_cast<const unsigned*>(&v);
        }
        return make_int4(static_cast<int>(half_bits[0]), static_cast<int>(half_bits[1]),
                         static_cast<int>(half_bits[2]), static_cast<int>(half_bits[3]));
    }
};
} // namespace infernix::ops::detail
