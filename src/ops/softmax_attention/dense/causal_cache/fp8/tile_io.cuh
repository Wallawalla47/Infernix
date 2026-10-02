#pragma once
#include "ops/softmax_attention/common/causal_tile_io.cuh"

#include "ops/common/math.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/warp.cuh"
#include "ops/kernel/paged_kv_address.cuh"
#include "ops/kv_cache/fp8_e4m3_row_codec.cuh"
#include "ops/softmax_attention/dense/causal_cache/fp8/operands.h"

namespace ninfer::ops::detail {

__device__ __forceinline__ int4 fp8_kv_dequant_f16x8(const std::uint8_t* codes, __half scale) {
    const int2 raw         = load_vec<int2>(codes);
    const std::uint16_t* c = reinterpret_cast<const std::uint16_t*>(&raw);
    unsigned packed[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const __half2 value2 = kv_cache_fp8_dequant_code2_to_half2(c[i], scale);
        packed[i]            = *reinterpret_cast<const unsigned*>(&value2);
    }
    return make_int4(static_cast<int>(packed[0]), static_cast<int>(packed[1]),
                     static_cast<int>(packed[2]), static_cast<int>(packed[3]));
}

struct Fp8KvTiledValues {
    using Scale                       = __half;
    static constexpr int kCodeBytes   = 256;
    static constexpr int kScaleItems  = 1;
    static constexpr bool kE4m3Values = false;

    // A 64-key FP16 PV partial is bounded by 64 * 448 * max_scale. Row scales are nonnegative
    // FP16 values; dividing them by 2^e, e = ilogb(max_scale) when it is at least 2, leaves the
    // largest below 2 and the bound below 57344. The 64 keys' scales are two per lane.
    template <int Keys>
    __device__ __forceinline__ static int tile_shift(const Scale* scales, int lane) {
        static_assert(Keys == 64);
        const __half2 pair  = load_vec<__half2>(scales + 2 * lane);
        const float maximum = warp_max(fmaxf(__low2float(pair), __high2float(pair)), 0xffffffffU);
        return maximum < 2.0F ? 0 : ilogbf(maximum);
    }

    // `mul` is the tile's exact power-of-two shift.
    __device__ __forceinline__ static int4 expand(const std::uint8_t* codes, Scale scale,
                                                  __half mul) {
        return fp8_kv_dequant_f16x8(codes, __hmul(scale, mul));
    }
};

} // namespace ninfer::ops::detail
