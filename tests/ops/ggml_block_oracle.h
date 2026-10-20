#pragma once

// Test-owned exact decode of GGUF (ggml) blocks: this file's own transcription of llama.cpp's
// dequantize_row_* formulas, evaluated in binary64. Every decoded value is a multiple of 2^-27 below
// 2^28 in magnitude, so binary64 holds it exactly. Shared by the op suites' oracles.

#include "core/ggml_format.h"
#include "core/weight.h"
#include "ops/ggml/ggml_tables.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace infernix::test::ggml_oracle {

inline float f16(std::uint16_t w) {
    const int sign = (w >> 15) & 1, e = (w >> 10) & 0x1F, m = w & 0x3FF;
    const double v = e == 0 ? std::ldexp(m, -24) : std::ldexp(1024 + m, e - 25);
    return static_cast<float>(sign ? -v : v);
}

inline std::uint16_t u16(const std::uint8_t* p) { return static_cast<std::uint16_t>(p[0] | (p[1] << 8)); }

// One block's exact values (llama.cpp ggml-quants.c dequantize_row_*, evaluated in binary64).
inline void decode_block(QType q, const std::uint8_t* b, double* y) {
    using namespace infernix::ggml;
    switch (q) {
    case QType::GGML_Q2_0: {
        const double d = f16(u16(b));
        for (int j = 0; j < 64; ++j) { y[j] = ((b[2 + j / 4] >> (2 * (j % 4))) & 3) - 1.0; y[j] *= d; }
        break;
    }
    case QType::GGML_IQ4_NL: {
        const double d = f16(u16(b));
        for (int j = 0; j < 16; ++j) {
            y[j]      = d * k_kvalues_iq4nl[b[2 + j] & 0xF];
            y[j + 16] = d * k_kvalues_iq4nl[b[2 + j] >> 4];
        }
        break;
    }
    case QType::GGML_IQ4_XS: {
        const double d = f16(u16(b));
        const int scales_h = u16(b + 2);
        for (int ib = 0; ib < 8; ++ib) {
            const int ls = ((b[4 + ib / 2] >> 4 * (ib % 2)) & 0xF) | (((scales_h >> 2 * ib) & 3) << 4);
            const double dl = d * (ls - 32);
            for (int j = 0; j < 16; ++j) {
                y[32 * ib + j]      = dl * k_kvalues_iq4nl[b[8 + 16 * ib + j] & 0xF];
                y[32 * ib + j + 16] = dl * k_kvalues_iq4nl[b[8 + 16 * ib + j] >> 4];
            }
        }
        break;
    }
    case QType::GGML_Q4_K:
    case QType::GGML_Q5_K: {
        const double d = f16(u16(b)), dmin = f16(u16(b + 2));
        const std::uint8_t* s = b + 4;
        const auto scale_min = [&](int j, int& sc, int& m) {
            if (j < 4) { sc = s[j] & 63; m = s[j + 4] & 63; }
            else { sc = (s[j + 4] & 0xF) | ((s[j - 4] >> 6) << 4); m = (s[j + 4] >> 4) | ((s[j] >> 6) << 4); }
        };
        const bool five = q == QType::GGML_Q5_K;
        const std::uint8_t* qh = b + 16;
        const std::uint8_t* ql = b + (five ? 48 : 16);
        for (int ch = 0; ch < 4; ++ch) {
            int sc1, m1, sc2, m2;
            scale_min(2 * ch, sc1, m1);
            scale_min(2 * ch + 1, sc2, m2);
            for (int l = 0; l < 32; ++l) {
                const int lo = (ql[32 * ch + l] & 0xF) + (five && (qh[l] >> (2 * ch) & 1) ? 16 : 0);
                const int hi = (ql[32 * ch + l] >> 4) + (five && (qh[l] >> (2 * ch + 1) & 1) ? 16 : 0);
                y[64 * ch + l]      = d * sc1 * lo - dmin * m1;
                y[64 * ch + 32 + l] = d * sc2 * hi - dmin * m2;
            }
        }
        break;
    }
    case QType::GGML_Q6_K: {
        const double d = f16(u16(b + 208));
        for (int n = 0; n < 2; ++n) {
            const std::uint8_t* ql = b + 64 * n;
            const std::uint8_t* qh = b + 128 + 32 * n;
            const auto* sc = reinterpret_cast<const std::int8_t*>(b + 192 + 8 * n);
            for (int l = 0; l < 32; ++l) {
                const int is = l / 16;
                y[128 * n + l]      = d * sc[is + 0] * (((ql[l] & 0xF) | ((qh[l] >> 0 & 3) << 4)) - 32);
                y[128 * n + l + 32] = d * sc[is + 2] * (((ql[l + 32] & 0xF) | ((qh[l] >> 2 & 3) << 4)) - 32);
                y[128 * n + l + 64] = d * sc[is + 4] * (((ql[l] >> 4) | ((qh[l] >> 4 & 3) << 4)) - 32);
                y[128 * n + l + 96] = d * sc[is + 6] * (((ql[l + 32] >> 4) | ((qh[l] >> 6 & 3) << 4)) - 32);
            }
        }
        break;
    }
    case QType::GGML_IQ2_S: {
        const double d = f16(u16(b));
        const std::uint8_t *qs = b + 2, *signs = b + 34, *qh = b + 66, *scales = b + 74;
        for (int ib = 0; ib < 8; ++ib) {
            for (int l = 0; l < 4; ++l) {
                const double dl = d * (0.5 + ((scales[ib] >> 4 * (l / 2)) & 0xF)) * 0.25;
                const std::uint64_t g = k_iq2s_grid[qs[4 * ib + l] | ((qh[ib] << (8 - 2 * l)) & 0x300)];
                for (int j = 0; j < 8; ++j) {
                    y[32 * ib + 8 * l + j] = dl * ((g >> 8 * j) & 0xFF) * ((signs[4 * ib + l] & k_kmask_iq2xs[j]) ? -1 : 1);
                }
            }
        }
        break;
    }
    case QType::GGML_IQ3_XXS: {
        const double d = f16(u16(b));
        for (int ib = 0; ib < 8; ++ib) {
            std::uint32_t aux;
            std::memcpy(&aux, b + 66 + 4 * ib, 4);
            const double db = d * (0.5 + (aux >> 28)) * 0.5;
            for (int l = 0; l < 4; ++l) {
                const std::uint8_t signs = k_ksigns_iq2xs[(aux >> 7 * l) & 127];
                const std::uint32_t g1 = k_iq3xxs_grid[b[2 + 8 * ib + 2 * l]], g2 = k_iq3xxs_grid[b[2 + 8 * ib + 2 * l + 1]];
                for (int j = 0; j < 4; ++j) {
                    y[32 * ib + 8 * l + j]     = db * ((g1 >> 8 * j) & 0xFF) * ((signs & k_kmask_iq2xs[j]) ? -1 : 1);
                    y[32 * ib + 8 * l + j + 4] = db * ((g2 >> 8 * j) & 0xFF) * ((signs & k_kmask_iq2xs[j + 4]) ? -1 : 1);
                }
            }
        }
        break;
    }
    case QType::GGML_IQ3_S: {
        const double d = f16(u16(b));
        const std::uint8_t *qs = b + 2, *qh = b + 66, *signs = b + 74, *scales = b + 106;
        for (int sb = 0; sb < 8; ++sb) {
            const double db = d * (1 + 2 * ((scales[sb / 2] >> 4 * (sb % 2)) & 0xF));
            for (int l = 0; l < 4; ++l) {
                const std::uint32_t g1 = k_iq3s_grid[qs[8 * sb + 2 * l] | ((qh[sb] << (8 - 2 * l)) & 256)];
                const std::uint32_t g2 = k_iq3s_grid[qs[8 * sb + 2 * l + 1] | ((qh[sb] << (7 - 2 * l)) & 256)];
                for (int j = 0; j < 4; ++j) {
                    y[32 * sb + 8 * l + j]     = db * ((g1 >> 8 * j) & 0xFF) * ((signs[4 * sb + l] & k_kmask_iq2xs[j]) ? -1 : 1);
                    y[32 * sb + 8 * l + j + 4] = db * ((g2 >> 8 * j) & 0xFF) * ((signs[4 * sb + l] & k_kmask_iq2xs[j + 4]) ? -1 : 1);
                }
            }
        }
        break;
    }
    default:
        throw std::logic_error("decode_block: format");
    }
}

// Each element's subtracted term of one block (Q4_K/Q5_K: dmin * m of its sub-block; 0 elsewhere),
// which error bounds of routes that apply it separately need: y = (y + offset) - offset.
inline void block_offsets(QType q, const std::uint8_t* b, double* offset) {
    const int elems = static_cast<int>(ggml_block_elems(q));
    for (int i = 0; i < elems; ++i) { offset[i] = 0.0; }
    if (q != QType::GGML_Q4_K && q != QType::GGML_Q5_K) { return; }
    const double dmin = f16(u16(b + 2));
    const std::uint8_t* s = b + 4;
    for (int j = 0; j < 8; ++j) {
        const int m = j < 4 ? s[j + 4] & 63 : (s[j + 4] >> 4) | ((s[j] >> 6) << 4);
        for (int l = 0; l < 32; ++l) { offset[32 * j + l] = dmin * m; }
    }
}

} // namespace infernix::test::ggml_oracle
