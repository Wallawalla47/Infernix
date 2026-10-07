#pragma once

// Canonical routed-expert arithmetic shared by the GPU and CPU routes of offloaded_sparse_moe
// (design: docs/maintainer/qwen3_8-flash-next-design.md §16.2).
//
// Every function here is part of a bit-exact contract: the CPU and GPU builds of this header must
// return identical bits for identical inputs. Only IEEE binary32 operations with
// round-to-nearest-even, integer operations, and explicitly fused fmaf are used. Translation units
// that include this header on the host must be compiled with -ffp-contract=off -fno-fast-math.

#include <cmath>
#include <cstdint>
#include <cstring>

#if defined(__CUDACC__)
#    define INFERNIX_CANON_HD __host__ __device__ __forceinline__
#else
#    define INFERNIX_CANON_HD inline
#endif

namespace infernix::ops::canon {

// ---------------------------------------------------------------------------- IEEE helpers

INFERNIX_CANON_HD std::uint32_t f32_bits(float x) {
#if defined(__CUDA_ARCH__)
    return __float_as_uint(x);
#else
    std::uint32_t u;
    std::memcpy(&u, &x, sizeof u);
    return u;
#endif
}

INFERNIX_CANON_HD float f32_from_bits(std::uint32_t u) {
#if defined(__CUDA_ARCH__)
    return __uint_as_float(u);
#else
    float x;
    std::memcpy(&x, &u, sizeof x);
    return x;
#endif
}

INFERNIX_CANON_HD float mul_rn(float a, float b) {
#if defined(__CUDA_ARCH__)
    return __fmul_rn(a, b);
#else
    return a * b;
#endif
}

INFERNIX_CANON_HD float add_rn(float a, float b) {
#if defined(__CUDA_ARCH__)
    return __fadd_rn(a, b);
#else
    return a + b;
#endif
}

INFERNIX_CANON_HD float div_rn(float a, float b) {
#if defined(__CUDA_ARCH__)
    return __fdiv_rn(a, b);
#else
    return a / b;
#endif
}

INFERNIX_CANON_HD float fma_rn(float a, float b, float c) {
#if defined(__CUDA_ARCH__)
    return __fmaf_rn(a, b, c);
#else
    return std::fmaf(a, b, c); // correctly rounded on every supported C++ runtime
#endif
}

// Round-to-nearest-even conversion of an int64 to binary32.
INFERNIX_CANON_HD float i64_to_f32_rn(std::int64_t v) {
#if defined(__CUDA_ARCH__)
    return __ll2float_rn(v);
#else
    return static_cast<float>(v); // the default rounding mode is round-to-nearest-even
#endif
}

// Round-to-nearest-even integer of a binary32 with |x| < 2^22.
INFERNIX_CANON_HD float rint_rn(float x) {
    const float magic = 12582912.0F; // 1.5 * 2^23
    return add_rn(add_rn(x, magic), -magic);
}

INFERNIX_CANON_HD float bf16_to_f32(std::uint16_t h) { return f32_from_bits(std::uint32_t{h} << 16); }

// Round-to-nearest-even binary32 -> bfloat16. Every NaN becomes the canonical quiet NaN 0x7FC0:
// NaN payloads and signs differ between x86 and CUDA arithmetic, so they are not part of the bits.
INFERNIX_CANON_HD std::uint16_t f32_to_bf16_rn(float x) {
    const std::uint32_t u = f32_bits(x);
    if ((u & 0x7FFFFFFFU) > 0x7F800000U) { return 0x7FC0U; }
    const std::uint32_t lsb = (u >> 16) & 1U;
    return static_cast<std::uint16_t>((u + 0x7FFFU + lsb) >> 16);
}

// ---------------------------------------------------------------------------- E2M1 and E4M3FN

// Twice the E2M1 value of a 4-bit code: {0, 1, 2, 3, 4, 6, 8, 12}, negated by bit 3. The
// magnitudes are the bytes of a constant, selected by one byte permute on the device and a shift
// on the host: an indexed array would live in local memory on the device and turn every lookup
// into a stack store and load.
INFERNIX_CANON_HD int e2m1_x2(unsigned code) {
#if defined(__CUDA_ARCH__)
    const int mag = static_cast<int>(__byte_perm(0x03020100U, 0x0C080604U, code & 7U));
#else
    const int mag = static_cast<int>((0x0C08060403020100ULL >> (8U * (code & 7U))) & 0xFFU);
#endif
    return (code & 8U) != 0 ? -mag : mag;
}

#if defined(__CUDACC__)
// e2m1_x2 of four codes at once: the codes are the low nibbles of the bytes of `codes` (the high
// nibbles zero), the result holds the four doubled values as signed bytes in the same order. Two
// byte permutes select the positive and the negated magnitude; the sign bits choose between them.
__device__ __forceinline__ std::uint32_t e2m1_x2_quad(std::uint32_t codes) {
    const std::uint32_t pairs    = codes | (codes >> 4);                                // nibbles 2j, 2j+1 in byte 2j
    const std::uint32_t selector = ((pairs & 0xFFU) | ((pairs >> 8) & 0xFF00U)) & 0x7777U; // magnitude index per byte
    const std::uint32_t positive = __byte_perm(0x03020100U, 0x0C080604U, selector);
    const std::uint32_t negative = __byte_perm(0xFDFEFF00U, 0xF4F8FAFCU, selector);
    const std::uint32_t sign     = ((codes >> 3) & 0x01010101U) * 0xFFU;                // 0xFF where bit 3 is set
    return (positive & ~sign) | (negative & sign);
}
#endif

// E4M3FN value times 2^9 for a non-negative scale word (sign bit ignored): an integer in
// [0, 229376]. Words 0x7F (NaN) are rejected by the format validators and never reach here.
INFERNIX_CANON_HD std::int32_t e4m3_scaled(unsigned word) {
    const unsigned e = (word >> 3) & 15U;
    const unsigned m = word & 7U;
    return e == 0 ? static_cast<std::int32_t>(m) : static_cast<std::int32_t>((8U + m) << (e - 1U));
}

// Exact binary32 value of a non-negative E4M3FN word.
INFERNIX_CANON_HD float e4m3_value(unsigned word) {
    // e4m3_scaled < 2^18 is exact in binary32, and the power-of-two scale is exact.
    return mul_rn(static_cast<float>(e4m3_scaled(word)), 1.0F / 512.0F);
}

// E4M3FN encode of a non-negative binary32 with round-to-nearest-even and saturation to 448
// (the semantics of cvt.rn.satfinite.e4m3x2.f32). NaN encodes to 0x7F.
INFERNIX_CANON_HD std::uint8_t e4m3_rn_satfinite(float x) {
    const std::uint32_t u = f32_bits(x) & 0x7FFFFFFFU;
    if (u > 0x7F800000U) { return 0x7F; }
    if (u >= 0x43E00000U) { return 0x7E; } // >= 448 (and +inf)
    const int exp = static_cast<int>(u >> 23);
    if (exp < 127 - 6) {
        // Subnormal E4M3 range: multiples of 2^-9 up to 7 * 2^-9; 8 * 2^-9 is the smallest normal.
        // Value in units of 2^-9, rounded to nearest even. Exact: scaling by 2^9 is a power of two.
        const float q = mul_rn(f32_from_bits(u), 512.0F);
        const float r = rint_rn(q);
        return static_cast<std::uint8_t>(static_cast<int>(r)); // 8 rounds up into e = 1, m = 0
    }
    // Normal range: keep 3 mantissa bits, round to nearest even on the dropped 20 bits.
    const std::uint32_t mant = u & 0x7FFFFFU;
    std::uint32_t keep       = mant >> 20;
    const std::uint32_t rest = mant & 0xFFFFFU;
    int e                    = exp - 127 + 7;
    if (rest > 0x80000U || (rest == 0x80000U && (keep & 1U) != 0)) {
        ++keep;
        if (keep == 8U) {
            keep = 0;
            ++e;
        }
    }
    if (e > 15 || (e == 15 && keep > 6U)) { return 0x7E; }
    return static_cast<std::uint8_t>((static_cast<unsigned>(e) << 3) | keep);
}

// E2M1 encode with round-to-nearest-even onto {0, 0.5, 1, 1.5, 2, 3, 4, 6} and saturation to 6
// (the semantics of cvt.rn.satfinite.e2m1x2.f32). The sign of x, including -0, sets bit 3.
// Callers never pass NaN: the quantizer divides finite BF16 activations by a positive scale.
INFERNIX_CANON_HD std::uint8_t e2m1_rn_satfinite(float x) {
    const std::uint32_t bits = f32_bits(x);
    const unsigned sign      = (bits >> 31) != 0 ? 8U : 0U;
    const float a            = f32_from_bits(bits & 0x7FFFFFFFU);
    unsigned code;
    if (a <= 0.25F) {
        code = 0;
    } else if (a < 0.75F) {
        code = 1;
    } else if (a <= 1.25F) {
        code = 2;
    } else if (a < 1.75F) {
        code = 3;
    } else if (a <= 2.5F) {
        code = 4;
    } else if (a < 3.5F) {
        code = 5;
    } else if (a <= 5.0F) {
        code = 6;
    } else {
        code = 7;
    }
    return static_cast<std::uint8_t>(sign | code);
}

// ---------------------------------------------------------------------------- A4 quantization

// One quantized 16-element activation block: doubled E2M1 values, the E4M3FN scale word, its
// scaled integer, and the sum of the doubled values (used by unsigned-times-signed dot products).
struct A4Block {
    std::int8_t c2[16];
    std::uint8_t scale_word;
    std::int32_t scale_scaled;
    std::int32_t c2_sum;
};

// ModelOpt's NVFP4 activation rule in exact IEEE binary32 (design §16.2):
//   s = e4m3_rn_satfinite(amax / fl(6 g));  d = fl(e4m3(s) g);  code_j = e2m1_rn_satfinite(v_j / d)
INFERNIX_CANON_HD A4Block quantize_a4_block(const std::uint16_t* v_bf16, float input_scale) {
    A4Block out{};
    float amax = 0.0F;
    for (int j = 0; j < 16; ++j) {
        const float a = f32_from_bits(f32_bits(bf16_to_f32(v_bf16[j])) & 0x7FFFFFFFU);
        amax          = a > amax ? a : amax;
    }
    const std::uint8_t s = e4m3_rn_satfinite(div_rn(amax, mul_rn(6.0F, input_scale)));
    out.scale_word       = s;
    out.scale_scaled     = e4m3_scaled(s);
    if (out.scale_scaled == 0) { return out; }
    const float d = mul_rn(e4m3_value(s), input_scale);
    std::int32_t sum = 0;
    for (int j = 0; j < 16; ++j) {
        const int c = e2m1_x2(e2m1_rn_satfinite(div_rn(bf16_to_f32(v_bf16[j]), d)));
        out.c2[j]   = static_cast<std::int8_t>(c);
        sum += c;
    }
    out.c2_sum = sum;
    return out;
}

// ---------------------------------------------------------------------------- epilogues

// y = bf16_rn((fl32_rn(S) * 2^-20) * alpha); the scaling by 2^-20 is exact.
INFERNIX_CANON_HD std::uint16_t a4_row_output(std::int64_t s, float alpha) {
    return f32_to_bf16_rn(mul_rn(mul_rn(i64_to_f32_rn(s), 1.0F / 1048576.0F), alpha));
}

// The one NaN the transcendental functions return: CUDA arithmetic produces a canonical NaN where
// x86 propagates the input payload, so NaN inputs are mapped explicitly to keep the bits equal.
inline constexpr std::uint32_t kCanonicalNan = 0x7FC00000U;

// exp(x) with Cody-Waite reduction and a fixed degree-7 Taylor polynomial evaluated by explicit fma.
// Overflow returns +inf, deep underflow +0, and NaN the canonical NaN.
INFERNIX_CANON_HD float exp_c(float x) {
    if (x != x) { return f32_from_bits(kCanonicalNan); }
    if (x > 88.72283935546875F) { return f32_from_bits(0x7F800000U); }
    if (x < -103.97208404541015625F) { return 0.0F; }
    const float n  = rint_rn(mul_rn(x, 1.44269502162933349609375F));
    const float r0 = fma_rn(-n, 0.693145751953125F, x);           // ln2 high part (11 bits)
    const float r  = fma_rn(-n, 1.428606765330187045e-06F, r0);   // ln2 low part
    float p        = 1.0F / 5040.0F;
    p              = fma_rn(p, r, 1.0F / 720.0F);
    p              = fma_rn(p, r, 1.0F / 120.0F);
    p              = fma_rn(p, r, 1.0F / 24.0F);
    p              = fma_rn(p, r, 1.0F / 6.0F);
    p              = fma_rn(p, r, 0.5F);
    p              = fma_rn(p, r, 1.0F);
    p              = fma_rn(p, r, 1.0F);
    // Scale by 2^n in two exact power-of-two steps so subnormal results are rounded once.
    const int k  = static_cast<int>(n);
    const int k1 = k / 2;
    const int k2 = k - k1;
    const float s1 = f32_from_bits(static_cast<std::uint32_t>(k1 + 127) << 23);
    const float s2 = f32_from_bits(static_cast<std::uint32_t>(k2 + 127) << 23);
    return mul_rn(mul_rn(p, s1), s2);
}

// SiLU(g) on a binary32, in the form that never overflows exp:
//   g >= 0: g / (1 + exp_c(-g));   g < 0: (g * e) / (1 + e), e = exp_c(g).
// A NaN result (a NaN input, or -inf * 0 at g = -inf) is the canonical NaN.
INFERNIX_CANON_HD float silu_c(float g) {
    float r;
    if (g >= 0.0F) {
        r = div_rn(g, add_rn(1.0F, exp_c(-g)));
    } else {
        const float e = exp_c(g);
        r             = div_rn(mul_rn(g, e), add_rn(1.0F, e));
    }
    return r != r ? f32_from_bits(kCanonicalNan) : r;
}

// SiLU(g) * u on BF16 inputs, rounded to BF16: bf16_rn(silu_c(g) * u).
INFERNIX_CANON_HD std::uint16_t swiglu_bf16(std::uint16_t gate_bf16, std::uint16_t up_bf16) {
    return f32_to_bf16_rn(mul_rn(silu_c(bf16_to_f32(gate_bf16)), bf16_to_f32(up_bf16)));
}

// ---------------------------------------------------------------------------- A4 codes

// The A4 rule of quantize_a4_block split into its two steps and returning the 4-bit E2M1 code
// words that the block-scaled tensor-core (wide) route consumes instead of their doubled values.
// For every block, the scale word equals quantize_a4_block's and e2m1_x2(code[j]) equals its
// c2[j]; a block whose scale encodes to zero has all-zero codes.
INFERNIX_CANON_HD std::uint8_t a4_scale_word(float amax, float input_scale) {
    return e4m3_rn_satfinite(div_rn(amax, mul_rn(6.0F, input_scale)));
}

// The code of one element v of a block with a nonzero scale word s.
INFERNIX_CANON_HD std::uint8_t a4_code(float v, std::uint8_t scale_word, float input_scale) {
    return e2m1_rn_satfinite(div_rn(v, mul_rn(e4m3_value(scale_word), input_scale)));
}

struct A4Codes {
    std::uint8_t code[16]; // E2M1 code words, sign in bit 3
    std::uint8_t scale_word;
};

INFERNIX_CANON_HD A4Codes quantize_a4_codes(const std::uint16_t* v_bf16, float input_scale) {
    A4Codes out{};
    float amax = 0.0F;
    for (int j = 0; j < 16; ++j) {
        const float a = f32_from_bits(f32_bits(bf16_to_f32(v_bf16[j])) & 0x7FFFFFFFU);
        amax          = a > amax ? a : amax;
    }
    out.scale_word = a4_scale_word(amax, input_scale);
    if (e4m3_scaled(out.scale_word) == 0) { return out; }
    for (int j = 0; j < 16; ++j) { out.code[j] = a4_code(bf16_to_f32(v_bf16[j]), out.scale_word, input_scale); }
    return out;
}

} // namespace infernix::ops::canon

#undef INFERNIX_CANON_HD
