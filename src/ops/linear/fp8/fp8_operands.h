#pragma once

#include "core/tensor.h"
#include "core/weight.h"
#include "ops/linear/fp8/fp8_a8_plan.h"
#include <cuda_bf16.h>
#include <stdexcept>
#include <type_traits>

namespace infernix::ops::detail {
// A row-scaled FP8 weight multiplies each row by one stored word: BF16 (fp8_e4m3fn_row_bf16) or
// FP32 (fp8_e4m3fn_row_f32). The word type is a template parameter of every kernel, so a BF16
// instance compiles the BF16 format's code and an FP32 instance reads its words as they are.
template <class Scale = __nv_bfloat16>
struct Fp8A16OperandsT {
    const __nv_bfloat16* x;
    const std::uint8_t* codes;
    const Scale* scales;
    int rows, k, tokens;
};
using Fp8A16Operands = Fp8A16OperandsT<>;

template <class Scale = __nv_bfloat16>
struct Fp8A8OperandsT {
    const std::uint8_t* x;
    const float* x_scales;
    const std::uint8_t* codes;
    const Scale* scales;
    int rows, k, tokens;
    // The A8 residual profile's second plane (Fp8A8Workspace::residual), else null.
    const std::uint8_t* x_residual = nullptr;
};
using Fp8A8Operands = Fp8A8OperandsT<>;

#if defined(__CUDACC__)
// A stored row word as FP32; a BF16 word widens exactly.
__device__ __forceinline__ float fp8_widen(__nv_bfloat16 word) { return __bfloat162float(word); }
__device__ __forceinline__ float fp8_widen(float word) { return word; }
#endif

template <class Scale>
inline void require_fp8_scale_word(const Weight& w) {
    static_assert(std::is_same_v<Scale, __nv_bfloat16> || std::is_same_v<Scale, float>);
    if ((w.qtype == QType::FP8_E4M3FN_ROW_F32) != std::is_same_v<Scale, float>) {
        throw std::invalid_argument("FP8 operands: the instance's scale word does not match the weight");
    }
}

template <class Scale = __nv_bfloat16>
inline Fp8A16OperandsT<Scale> fp8_a16_operands(const Tensor& x, const Weight& w) {
    require_fp8_scale_word<Scale>(w);
    return {static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const Scale*>(w.scales),
            w.n,
            w.k,
            x.ne[1]};
}

template <class Scale = __nv_bfloat16>
inline Fp8A8OperandsT<Scale> fp8_a8_operands(const Weight& w, Fp8A8Workspace x, int tokens) {
    require_fp8_scale_word<Scale>(w);
    return {x.codes,
            x.scales,
            static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const Scale*>(w.scales),
            w.n,
            w.k,
            tokens,
            x.residual};
}

// Private template admission is independent of the public finite shape registry. FP32 words are
// read in pairs (float2), BF16 pairs as one 32-bit word.
template <class Schedule, class Operands>
inline void validate_fp8_operands(const Operands& p) {
    const auto aligned = [](const void* v, int alignment) {
        return v && reinterpret_cast<std::uintptr_t>(v) % alignment == 0;
    };
    constexpr bool kF32 = std::is_same_v<std::remove_cv_t<std::remove_pointer_t<decltype(p.scales)>>, float>;
    if (p.rows <= 0 || p.k <= 0 || p.tokens <= 0 || p.k % 32 || !aligned(p.x, 16) ||
        !aligned(p.codes, 16) || !aligned(p.scales, kF32 ? 8 : 2))
        throw std::invalid_argument("FP8 templates require positive N/K/T and aligned operands");
    if constexpr (requires { p.x_scales; }) {
        if (!aligned(p.x_scales, 4) || !aligned(p.scales, kF32 ? 8 : 4))
            throw std::invalid_argument("FP8 A8 requires scales aligned to four bytes");
    }
    if constexpr (Schedule::kStaticK > 0) {
        if (p.k != Schedule::kStaticK)
            throw std::invalid_argument("FP8 template static K does not match operands");
    }
}
} // namespace infernix::ops::detail
