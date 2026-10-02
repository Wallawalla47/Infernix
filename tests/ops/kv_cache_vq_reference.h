#pragma once

// Host reference of the vector-quantized KV formats (Vq2, Q4KeyVq2Value): the exact FP32 encoders
// of ops/kv_cache/vq2_codec.cuh and q4_lloyd_codec.cuh (same operations in the same order, so codes
// and scales compare bit for bit), their decoders, the INT8-G64 window rows and the window tags.

#include "ops/kv_cache/q4_lloyd_codec.cuh"
#include "ops/kv_cache/vq2_codec.cuh"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace ninfer::test::vq {

inline constexpr int kDim      = 256;
inline constexpr int kWords    = 32;
inline constexpr int kPatterns = 512;

inline std::uint16_t f32_to_f16(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t sign = (bits >> 16) & 0x8000u;
    const std::uint32_t exp  = (bits >> 23) & 0xffu;
    std::uint32_t mantissa   = bits & 0x007fffffu;
    if (exp == 0xffu) {
        return static_cast<std::uint16_t>(sign | (mantissa == 0 ? 0x7c00u : 0x7e00u));
    }
    const int half_exp = static_cast<int>(exp) - 127 + 15;
    if (half_exp >= 31) return static_cast<std::uint16_t>(sign | 0x7c00u);
    if (half_exp <= 0) {
        if (half_exp < -10) return static_cast<std::uint16_t>(sign);
        mantissa |= 0x00800000u;
        const int shift             = 14 - half_exp;
        std::uint32_t half_mantissa = mantissa >> shift;
        const std::uint32_t halfway = 1u << (shift - 1);
        const std::uint32_t tail    = mantissa & ((1u << shift) - 1u);
        if (tail > halfway || (tail == halfway && (half_mantissa & 1u) != 0u)) ++half_mantissa;
        return static_cast<std::uint16_t>(sign | half_mantissa);
    }
    std::uint32_t half_mantissa = mantissa >> 13;
    const std::uint32_t tail    = mantissa & 0x1fffu;
    std::uint32_t rounded_exp   = static_cast<std::uint32_t>(half_exp);
    if (tail > 0x1000u || (tail == 0x1000u && (half_mantissa & 1u) != 0u)) {
        ++half_mantissa;
        if (half_mantissa == 0x400u) {
            half_mantissa = 0;
            ++rounded_exp;
            if (rounded_exp >= 31) return static_cast<std::uint16_t>(sign | 0x7c00u);
        }
    }
    return static_cast<std::uint16_t>(sign | (rounded_exp << 10) | half_mantissa);
}

inline float f16_to_f32(std::uint16_t bits) {
    const bool negative = (bits & 0x8000u) != 0;
    const int exp       = (bits >> 10) & 0x1f;
    const int mantissa  = bits & 0x03ff;
    float magnitude     = 0.0f;
    if (exp == 0) {
        magnitude = std::ldexp(static_cast<float>(mantissa), -24);
    } else if (exp == 31) {
        magnitude = mantissa == 0 ? std::numeric_limits<float>::infinity()
                                  : std::numeric_limits<float>::quiet_NaN();
    } else {
        magnitude = std::ldexp(1.0f + static_cast<float>(mantissa) / 1024.0f, exp - 15);
    }
    return negative ? -magnitude : magnitude;
}

// normalized_hadamard_d256_inplace with lane l holding dimensions l + 32 r.
inline void hadamard(std::array<float, kDim>& values) {
    for (int block = 0; block < 8; ++block) {
        for (int stride = 1; stride <= 16; stride <<= 1) {
            for (int base = 0; base < 32; base += 2 * stride) {
                for (int offset = 0; offset < stride; ++offset) {
                    const int lo = block * 32 + base + offset;
                    const int hi = lo + stride;
                    const float a = values[lo], b = values[hi];
                    values[lo]    = a + b;
                    values[hi]    = a - b;
                }
            }
        }
    }
    for (int span = 1; span < 8; span <<= 1) {
        for (int base = 0; base < 8; base += 2 * span) {
            for (int offset = 0; offset < span; ++offset) {
                for (int lane = 0; lane < 32; ++lane) {
                    const int lo = lane + 32 * (base + offset);
                    const int hi = lane + 32 * (base + offset + span);
                    const float a = values[lo], b = values[hi];
                    values[lo]    = a + b;
                    values[hi]    = a - b;
                }
            }
        }
    }
    for (float& value : values) value *= 0x1p-4f;
}

// The xor-butterfly warp sum of 32 lane values (identical in every lane).
inline float warp_sum(std::array<float, 32> v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        std::array<float, 32> next{};
        for (int lane = 0; lane < 32; ++lane) next[lane] = v[lane] + v[lane ^ offset];
        v = next;
    }
    return v[0];
}

// Sum of squares as kv_cache_vq_rotate_row: lane l accumulates its 8 values (l + 32 r) by FMA.
inline float sum_square(const std::array<float, kDim>& y) {
    std::array<float, 32> lanes{};
    for (int lane = 0; lane < 32; ++lane) {
        float s = 0.0f;
        for (int r = 0; r < 8; ++r) s = std::fma(y[lane + 32 * r], y[lane + 32 * r], s);
        lanes[lane] = s;
    }
    return warp_sum(lanes);
}

struct EncodedRow {
    std::vector<std::uint8_t> codes; // persistent code bytes
    std::uint16_t scale = 0;         // FP16 bits
    std::array<int, kDim> levels{};  // signed integer levels c (Vq2: 1/16, Q4: 1/32 units)
};

inline float codebook(int p, int j) {
    return static_cast<float>(ninfer::ops::kKVCacheVq2Codebook[8 * p + j]);
}

// kv_cache_vq2_pattern_distance: |P|^2 - 2 * dot + (odd ? 4 * m : 0) in the same FP32 order.
inline float vq2_distance(const float (&A)[8], bool odd, int p) {
    float norm = 0.0f;
    for (int j = 0; j < 8; ++j) norm += codebook(p, j) * codebook(p, j); // exact integers
    volatile float t = A[0] * codebook(p, 0);
    float dot        = t;
    float m          = t;
    for (int j = 1; j < 8; ++j) {
        t   = A[j] * codebook(p, j);
        dot = dot + t;
        m   = std::fmin(m, t);
    }
    const float d = std::fma(-2.0f, dot, norm);
    return odd ? std::fma(4.0f, m, d) : d;
}

inline std::uint16_t finish_scale(float sum_sq, float sigma, const std::array<float, 32>& dots) {
    const float dot = warp_sum(dots);
    return f32_to_f16(sigma > 0.0f && dot > 0.0f ? sum_sq / dot : 0.0f);
}

inline EncodedRow encode_vq2(const std::array<float, kDim>& y) {
    EncodedRow out;
    out.codes.assign(64, 0);
    const float sum_sq   = sum_square(y);
    const float sigma    = std::sqrt(sum_sq * (1.0f / 256.0f));
    const float to_units = sigma > 0.0f ? 16.0f / sigma : 0.0f;
    std::array<float, 32> dots{};
    for (int w = 0; w < kWords; ++w) {
        float A[8];
        unsigned negative = 0;
        for (int j = 0; j < 8; ++j) {
            A[j] = std::fabs(y[8 * w + j]) * to_units;
            negative |= (y[8 * w + j] < 0.0f ? 1u : 0u) << j;
        }
        const bool odd = (std::popcount(negative) & 1) != 0;
        int best_p     = 0;
        float best     = std::numeric_limits<float>::max();
        for (int p = 0; p < kPatterns; ++p) {
            const float d = vq2_distance(A, odd, p);
            if (d < best) {
                best   = d;
                best_p = p;
            }
        }
        std::uint32_t code = 0;
        float dot          = 0.0f;
        if (sigma > 0.0f) {
            unsigned sign = negative;
            if (odd) {
                int flip = 0;
                float m  = A[0] * codebook(best_p, 0);
                for (int j = 1; j < 8; ++j) {
                    const float t = A[j] * codebook(best_p, j);
                    if (t < m) {
                        m    = t;
                        flip = j;
                    }
                }
                sign ^= 1u << flip;
            }
            for (int j = 0; j < 8; ++j) {
                const int v               = ninfer::ops::kKVCacheVq2Codebook[8 * best_p + j];
                out.levels[8 * w + j]     = ((sign >> j) & 1u) ? -v : v;
                dot = std::fma(y[8 * w + j], static_cast<float>(out.levels[8 * w + j]), dot);
            }
            code = static_cast<std::uint32_t>(best_p) | ((sign & 0x7fu) << 9);
        } else {
            for (int j = 0; j < 8; ++j) out.levels[8 * w + j] = ninfer::ops::kKVCacheVq2Codebook[j];
        }
        dots[w]              = dot;
        out.codes[2 * w]     = static_cast<std::uint8_t>(code & 0xffu);
        out.codes[2 * w + 1] = static_cast<std::uint8_t>(code >> 8);
    }
    out.scale = finish_scale(sum_sq, sigma, dots);
    return out;
}

inline int q4_level(int m) {
    static constexpr int levels[8] = {4, 12, 21, 30, 40, 52, 66, 87};
    return levels[m];
}

inline EncodedRow encode_q4(const std::array<float, kDim>& y) {
    EncodedRow out;
    out.codes.assign(128, 0);
    const float sum_sq   = sum_square(y);
    const float sigma    = std::sqrt(sum_sq * (1.0f / 256.0f));
    const float to_units = sigma > 0.0f ? 32.0f / sigma : 0.0f;
    std::array<float, 32> dots{};
    for (int w = 0; w < kWords; ++w) {
        std::uint32_t word = 0;
        float dot          = 0.0f;
        for (int j = 0; j < 8; ++j) {
            const float value = y[8 * w + j];
            const float A     = std::fabs(value) * to_units;
            const int m = (A > 8.0f) + (A > 16.5f) + (A > 25.5f) + (A > 35.0f) + (A > 46.0f) +
                          (A > 59.0f) + (A > 76.5f);
            const bool neg = value < 0.0f;
            word |= static_cast<std::uint32_t>(neg ? 7 - m : 8 + m) << (4 * j);
            const float level = static_cast<float>(q4_level(m));
            dot               = std::fma(value, neg ? -level : level, dot);
        }
        if (sigma == 0.0f) {
            word = 0x77777777u;
            dot  = 0.0f;
        }
        for (int j = 0; j < 8; ++j) {
            const int n           = static_cast<int>((word >> (4 * j)) & 0xfu);
            out.levels[8 * w + j] = n >= 8 ? q4_level(n - 8) : -q4_level(7 - n);
        }
        dots[w] = dot;
        std::memcpy(out.codes.data() + 4 * w, &word, 4);
    }
    out.scale = finish_scale(sum_sq, sigma, dots);
    return out;
}

// INT8-G64 window row of a rotated row (kv_cache_int8_quant_params/code).
struct Int8Row {
    std::array<std::int8_t, kDim> codes{};
    std::array<std::uint16_t, 4> scales{};
};

inline Int8Row encode_int8(const std::array<float, kDim>& y) {
    Int8Row out;
    for (int g = 0; g < 4; ++g) {
        float amax = 0.0f;
        for (int i = 0; i < 64; ++i) amax = std::fmax(amax, std::fabs(y[64 * g + i]));
        const std::uint16_t scale = f32_to_f16(amax > 0.0f ? amax / 127.0f : 0.0f);
        const float represented   = f16_to_f32(scale);
        const float inverse       = represented > 0.0f ? 1.0f / represented : 0.0f;
        out.scales[g]             = scale;
        for (int i = 0; i < 64; ++i) {
            int q = 0;
            if (inverse != 0.0f) {
                q = static_cast<int>(std::nearbyint(y[64 * g + i] * inverse));
                q = std::max(-127, std::min(127, q));
            }
            out.codes[64 * g + i] = static_cast<std::int8_t>(q);
        }
    }
    return out;
}

inline std::uint32_t window_tag(int position, const std::vector<std::uint8_t>& codes,
                                std::uint16_t scale) {
    std::uint32_t hash = (2166136261u ^ static_cast<std::uint32_t>(position)) * 16777619u;
    for (std::size_t i = 0; i + 4 <= codes.size(); i += 4) {
        std::uint32_t word = 0;
        std::memcpy(&word, codes.data() + i, 4);
        hash = (hash ^ word) * 16777619u;
    }
    hash = (hash ^ scale) * 16777619u;
    hash ^= hash >> 15;
    hash *= 0x2c1b3c6du;
    hash ^= hash >> 12;
    return hash | 1u;
}

inline int window_slot(int position) {
    return position < 64 ? position : 64 + (position & 1023);
}

} // namespace ninfer::test::vq
