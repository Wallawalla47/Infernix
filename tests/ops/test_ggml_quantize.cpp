// ggml_quantize_q6_k (include/infernix/ops/ggml_quantize.h) against a host port of ggml's reference quantizer
// (quantize_row_q6_K_ref with make_qx_quants(16, 32, ..., rmse_type 1)), byte for byte, in the Rows and Tiles16
// layouts: the IQ3_S import's GDN control and QSA indexer shapes, single tiles, and blocks that are all zero, hold a
// zero or tiny sub-block, outliers, or equal largest sub-scales. The oracle writes every operation as its own
// statement (one binary32 rounding each, nothing contracted). ggml's dequantizer then reads the blocks back within
// a few per cent of the values.

#include "ops/op_family.h"
#include "core/device.h"
#include "infernix/ops/ggml_quantize.h"

#include "ops/op_tester.h"

#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace infernix;
using namespace infernix::test;

namespace {

constexpr float kEps = 1e-15F;

int rne(float v) { return static_cast<int>(std::nearbyint(v)); }
int clamp_code(int l) { return std::max(-32, std::min(31, l)); }

float make_qx_quants(const float* x, int* L) {
    float max = 0.0F, amax = 0.0F;
    for (int i = 0; i < 16; ++i) {
        const float ax = std::fabs(x[i]);
        if (ax > amax) {
            amax = ax;
            max  = x[i];
        }
    }
    if (amax < kEps) {
        for (int i = 0; i < 16; ++i) { L[i] = 0; }
        return 0.0F;
    }
    const auto sums = [&](float iscale, float& sumlx, float& suml2, int* codes) {
        sumlx = 0.0F;
        suml2 = 0.0F;
        for (int i = 0; i < 16; ++i) {
            const float scaled = iscale * x[i];
            const int l        = clamp_code(rne(scaled));
            if (codes != nullptr) { codes[i] = l + 32; }
            const float w  = x[i] * x[i];
            const float fl = static_cast<float>(l);
            const float a  = w * x[i];
            const float b  = a * fl;
            sumlx          = sumlx + b;
            const float c  = w * fl;
            const float e  = c * fl;
            suml2          = suml2 + e;
        }
    };
    float sumlx = 0.0F, suml2 = 0.0F;
    sums(-32.0F / max, sumlx, suml2, L);
    float scale = suml2 != 0.0F ? sumlx / suml2 : 0.0F;
    float best  = scale * sumlx;
    for (int is = -9; is <= 9; ++is) {
        if (is == 0) { continue; }
        const float step   = 0.1F * static_cast<float>(is);
        const float nmax   = 32.0F + step;
        const float iscale = -nmax / max;
        sums(iscale, sumlx, suml2, nullptr);
        const float lhs = sumlx * sumlx;
        const float rhs = best * suml2;
        if (suml2 > 0.0F && lhs > rhs) {
            sums(iscale, sumlx, suml2, L);
            scale = sumlx / suml2;
            best  = scale * sumlx;
        }
    }
    return scale;
}

// One 256-element block into y[210].
void quantize_block(const float* x, std::uint8_t* y) {
    int L[256];
    float scales[16];
    float max_scale = 0.0F, max_abs = 0.0F;
    for (int ib = 0; ib < 16; ++ib) {
        scales[ib]      = make_qx_quants(x + 16 * ib, L + 16 * ib);
        const float abs = std::fabs(scales[ib]);
        if (abs > max_abs) {
            max_abs   = abs;
            max_scale = scales[ib];
        }
    }
    std::memset(y, 0, 210);
    if (max_abs < kEps) { return; }
    const float iscale  = -128.0F / max_scale;
    const __half d      = __float2half_rn(1.0F / iscale);
    std::uint16_t d_bits = 0;
    std::memcpy(&d_bits, &d, 2);
    std::int8_t sub[16];
    for (int ib = 0; ib < 16; ++ib) {
        const float v = iscale * scales[ib];
        sub[ib]       = static_cast<std::int8_t>(std::min(127, rne(v)));
    }
    const float df = __half2float(d);
    for (int j = 0; j < 16; ++j) {
        const float dl = df * static_cast<float>(sub[j]);
        if (dl == 0.0F) { continue; }
        for (int i = 0; i < 16; ++i) {
            const float q    = x[16 * j + i] / dl;
            L[16 * j + i] = clamp_code(rne(q)) + 32;
        }
    }
    for (int h = 0; h < 2; ++h) {
        const int* q = L + 128 * h;
        for (int l = 0; l < 32; ++l) {
            y[64 * h + l]       = static_cast<std::uint8_t>((q[l] & 0xF) | ((q[l + 64] & 0xF) << 4));
            y[64 * h + 32 + l]  = static_cast<std::uint8_t>((q[l + 32] & 0xF) | ((q[l + 96] & 0xF) << 4));
            y[128 + 32 * h + l] = static_cast<std::uint8_t>((q[l] >> 4) | ((q[l + 32] >> 4) << 2) | ((q[l + 64] >> 4) << 4) |
                                                            ((q[l + 96] >> 4) << 6));
        }
    }
    for (int ib = 0; ib < 16; ++ib) { y[192 + ib] = static_cast<std::uint8_t>(sub[ib]); }
    y[208] = static_cast<std::uint8_t>(d_bits & 0xFF);
    y[209] = static_cast<std::uint8_t>(d_bits >> 8);
}

// ggml's dequantize_row_q6_K of one block.
void dequantize_block(const std::uint8_t* y, float* out) {
    std::uint16_t d_bits = static_cast<std::uint16_t>(y[208] | (y[209] << 8));
    __half d;
    std::memcpy(&d, &d_bits, 2);
    const float df = __half2float(d);
    for (int h = 0; h < 2; ++h) {
        const std::uint8_t* ql = y + 64 * h;
        const std::uint8_t* qh = y + 128 + 32 * h;
        const auto* sc         = reinterpret_cast<const std::int8_t*>(y + 192 + 8 * h);
        for (int l = 0; l < 32; ++l) {
            const int is = l / 16;
            const int q1 = ((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            const int q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            const int q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            const int q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            out[128 * h + l]      = df * sc[is + 0] * q1;
            out[128 * h + l + 32] = df * sc[is + 2] * q2;
            out[128 * h + l + 64] = df * sc[is + 4] * q3;
            out[128 * h + l + 96] = df * sc[is + 6] * q4;
        }
    }
}

// Rows [n, k] of BF16 bits: N(0, sigma) per row, with special blocks mixed in.
std::vector<std::uint16_t> make_rows(int n, int k, std::mt19937& rng) {
    std::vector<std::uint16_t> bits(static_cast<std::size_t>(n) * k);
    std::uniform_real_distribution<float> log_sigma(-9.0F, 1.0F);
    for (int r = 0; r < n; ++r) {
        std::normal_distribution<float> normal(0.0F, std::exp2(log_sigma(rng)));
        for (int i = 0; i < k; ++i) { bits[static_cast<std::size_t>(r) * k + i] = f32_to_bf16(normal(rng)); }
        const int blocks = k / 256;
        for (int b = 0; b < blocks; ++b) {
            std::uint16_t* x = bits.data() + static_cast<std::size_t>(r) * k + 256 * b;
            switch ((r * blocks + b) % 9) {
            case 1: std::fill(x, x + 256, std::uint16_t{0}); break;                 // all zero
            case 2: std::fill(x + 32, x + 48, std::uint16_t{0}); break;             // a zero sub-block
            case 3: x[17] = f32_to_bf16(bf16_to_f32(x[17]) * 300.0F); break;        // an outlier
            case 4:                                                                  // a tiny sub-block
                for (int i = 64; i < 80; ++i) { x[i] = f32_to_bf16(bf16_to_f32(x[i]) * 1e-6F); }
                break;
            case 5:                                                                  // equal sub-blocks
                for (int i = 16; i < 256; ++i) { x[i] = x[i % 16]; }
                break;
            case 6:                                                                  // one non-zero value
                std::fill(x, x + 256, std::uint16_t{0});
                x[200] = f32_to_bf16(-0.75F);
                break;
            default: break;
            }
        }
    }
    return bits;
}

int run_case(int n, int k, std::mt19937& rng) {
    const auto bits    = make_rows(n, k, rng);
    const int per_row  = k / 256;
    const std::size_t bytes = static_cast<std::size_t>(n) * per_row * 210;
    std::vector<std::uint8_t> rows(bytes);
    std::vector<float> x(256);
    for (int r = 0; r < n; ++r) {
        for (int b = 0; b < per_row; ++b) {
            for (int i = 0; i < 256; ++i) { x[i] = bf16_to_f32(bits[static_cast<std::size_t>(r) * k + 256 * b + i]); }
            quantize_block(x.data(), rows.data() + (static_cast<std::size_t>(r) * per_row + b) * 210);
        }
    }
    // The blocks decode (ggml's dequantizer) to their values: within 3 % of the row's RMS, outlier blocks aside.
    int failures = 0;
    for (int r = 0; r < n; ++r) {
        double err = 0.0, ref = 0.0;
        for (int b = 0; b < per_row; ++b) {
            if ((r * per_row + b) % 9 == 3) { continue; }
            float deq[256];
            dequantize_block(rows.data() + (static_cast<std::size_t>(r) * per_row + b) * 210, deq);
            for (int i = 0; i < 256; ++i) {
                const double v = bf16_to_f32(bits[static_cast<std::size_t>(r) * k + 256 * b + i]);
                err += (deq[i] - v) * (deq[i] - v);
                ref += v * v;
            }
        }
        if (ref > 0.0 && std::sqrt(err / ref) > 0.03) {
            std::cerr << "FAIL ggml_quantize_q6_k n=" << n << " k=" << k << " row " << r << ": relative RMS error "
                      << std::sqrt(err / ref) << "\n";
            ++failures;
            break;
        }
    }
    std::vector<std::uint8_t> tiles(bytes);
    for (int r = 0; r < n; ++r) {
        for (int b = 0; b < per_row; ++b) {
            std::memcpy(tiles.data() + ((static_cast<std::size_t>(r / 16) * per_row + b) * 16 + r % 16) * 210,
                        rows.data() + (static_cast<std::size_t>(r) * per_row + b) * 210, 210);
        }
    }
    DeviceBuffer w(bits.size() * 2), out(bytes);
    w.copy_from_host(bits.data(), bits.size() * 2, 0);
    const Tensor tw(w.p, DType::BF16, {k, n});
    for (const auto layout : {ops::GgmlLayout::Rows, ops::GgmlLayout::Tiles16}) {
        const bool tiled = layout == ops::GgmlLayout::Tiles16;
        CUDA_CHECK(cudaMemset(out.p, 0xA5, bytes));
        ops::ggml_quantize_q6_k(tw, layout, static_cast<std::uint8_t*>(out.p), nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<std::uint8_t> got(bytes);
        CUDA_CHECK(cudaMemcpy(got.data(), out.p, bytes, cudaMemcpyDeviceToHost));
        const std::vector<std::uint8_t>& want = tiled ? tiles : rows;
        std::size_t bad = 0, first = bytes;
        for (std::size_t i = 0; i < bytes; ++i) {
            if (got[i] != want[i]) {
                ++bad;
                first = std::min(first, i);
            }
        }
        if (bad != 0) {
            std::cerr << "FAIL ggml_quantize_q6_k n=" << n << " k=" << k << (tiled ? " Tiles16" : " Rows") << ": " << bad
                      << " bytes differ, first at block " << first / 210 << " byte " << first % 210 << "\n";
            ++failures;
        }
    }
    return failures;
}

} // namespace

INFERNIX_OP_TEST(ggml_quantize) {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    std::mt19937 rng(20261101);
    int failures = 0;
    failures += run_case(96, 2560, rng);  // a GDN layer's control projection (a and b)
    failures += run_case(640, 2560, rng); // a QSA layer's indexer (query heads and key)
    failures += run_case(16, 256, rng);
    failures += run_case(48, 768, rng);
    std::cout << (failures == 0 ? "OK" : "FAIL") << " ggml_quantize\n";
    return failures == 0 ? 0 : 1;
}
