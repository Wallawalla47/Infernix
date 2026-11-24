// Inkling 2's decode and verify attention (include/infernix/ops/inkling2_attention.h), sliding and full visits.
//
// One call of 30 columns: lane 0 at positions 1100-1103 (its window starts inside unit 1), lane 1 at 60-81 (crossing
// a unit boundary, history from position 0; two attention groups of 16 and 6), a padding column, lane 2 at 5000
// (sliding: a 17-unit window; full: 79
// units in 27 splits of 3, so the in-split merge runs) and lane 3 at 9000-9001 with its first position 8013 inside
// its lowest unit (keys and conv history below it masked; full visits' tau above 1). Every earlier position of each lane holds random represented
// keys (BF16 K, FP16 V of BF16 values) and the conv rings random inputs. Projections are random BF16 of the
// checkpoint's scales (some query heads scaled so their clip engages; relative tables and sinks of realistic spread).
//
// Oracles:
//   - prologue (no hardware approximation): a host emulation through visit_math.h gives the appended K and V, the
//     final q and r and the rings bit for bit; FP64 stage by stage (k and v within half a BF16 ulp plus the inner conv
//     rounding's ulp, q and r likewise);
//   - attention: FP64 scores over the kernel's stored q, r and represented K/V with the bias table at its BF16
//     boundary (the pinned FP32 fold, itself checked within one BF16 ulp of FP64), FP64 softmax with the sink, the
//     output within half a BF16 ulp of FP64 plus sigmoid(gate) times one ulp of the inner BF16 a plus 2^-16 of the
//     row's largest |v|; ex2.approx makes the bits the kernel's own, so they are pinned by the next check;
//   - decode against verify: the columns run one at a time, in call order, from the same initial state give the call's
//     outputs, K/V pages and rings bit for bit.
//
// The full visit again with its pool in each quantized form (KvCodec int8, fp8, nvfp4, k8v4; the earlier positions'
// keys and values encoded on the host from the same kind of random represented words): the appended codes and side
// records bit for bit against a host encoder of the contract's encoding over the emulation's BF16 K / FP16 V (every
// represented value within half a step of the value it encodes, plus half an ulp of its own rounding), attention
// against FP64 over the decoded represented values at the same criterion, and the same decode = verify and fused =
// prologue bit checks (codes and side records included). The prompt route likewise.
//
// The rotated forms (vq2, k4v2): the codes and FP16 scales bit for bit against a host emulation of the encoder
// (y = 2^-3 H128 x in the kernel's butterfly order, the exhaustive first-nearest pattern search, the lane-ordered
// sums), each row's distortion |S c - y|^2 / |y|^2 within a sanity bound, the stored q' = bf16(2^-4 H128 q) bit for
// bit and within FP64 of the rotation; attention against FP64 in the rotated domain with the output rotated back.

#include "infernix/ops/inkling2_attention.h"
#include "kv_cache_vq_reference.h"
#include "ops/common/canonical_math.h"
#include "ops/ggml/ggml_a8.h"
#include "ops/inkling2/visit_math.h"
#include "ops/op_family.h"
#include "ops/op_tester.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

namespace canon = infernix::ops::canon;
namespace ik    = infernix::ops::inkling2;
using infernix::test::cuda_check;

namespace {

constexpr int D = 128, NH = ik::kHeads, NKV = ik::kKvHeads, G = ik::kGroupHeads, RD = ik::kRelDim;
constexpr int KVW = NKV * D, QW = NH * D, RW = NH * RD, R = ik::kRingPositions, U = ik::kKeyUnit;
constexpr int kLanes = 4, kPoolPages = 256, kTableStride = 160;
int g_failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_failures;
    }
}

template <class T>
T* dalloc(std::size_t n) {
    T* p = nullptr;
    cuda_check(cudaMalloc(&p, std::max<std::size_t>(n, 1) * sizeof(T)), "cudaMalloc");
    cuda_check(cudaMemset(p, 0, std::max<std::size_t>(n, 1) * sizeof(T)), "cudaMemset");
    return p;
}
template <class T>
T* upload(const std::vector<T>& v) {
    T* p = dalloc<T>(v.size());
    cuda_check(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
    return p;
}
template <class T>
void upload_to(T* p, const std::vector<T>& v) {
    cuda_check(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
}
template <class T>
std::vector<T> download(const T* p, std::size_t n) {
    std::vector<T> v(n);
    cuda_check(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost), "download");
    return v;
}

float f(std::uint16_t v) { return canon::bf16_to_f32(v); }
float bf(float x) { return canon::bf16_to_f32(canon::f32_to_bf16_rn(x)); }
std::uint16_t b16(float x) { return canon::f32_to_bf16_rn(x); }
std::uint16_t h16(float x) { return __half_as_ushort(__float2half_rn(x)); }
float hf(std::uint16_t u) { return __half2float(__ushort_as_half(u)); }
double ulp_bf16(double r) {
    const double a = std::fabs(r);
    return a < 0x1p-126 ? 0x1p-133 : std::ldexp(1.0, std::ilogb(a) - 7);
}
double ulp_f16(double r) {
    const double a = std::fabs(r);
    return a < 0x1p-14 ? 0x1p-24 : std::ldexp(1.0, std::ilogb(a) - 10);
}

struct Tally {
    const char* name;
    double worst = 0;
    long bad = 0, differ = 0, n = 0;
    void ratio(double r) {
        worst = std::max(worst, r);
        bad += r > 1.0 ? 1 : 0;
        ++n;
    }
    template <class T>
    void exact(const T* a, const T* b, std::size_t count) {
        for (std::size_t i = 0; i < count; ++i) { differ += std::memcmp(a + i, b + i, sizeof(T)) != 0 ? 1 : 0; }
        n += static_cast<long>(count);
    }
    void report() const {
        std::printf("%-40s %8ld checked, worst %.3f of the criterion, %ld beyond it, %ld differ\n", name, n, worst, bad,
                    differ);
        check(bad == 0 && differ == 0, name);
    }
};

// ---------------------------------------------------------------------------- quantized pools on the host

// The forms of inkling2_attention.h's KvCodec, encoded operation by operation as the contract states (IEEE divisions,
// single rounded products), so the stored bytes match exactly.
enum class Form { Words, Int8, Fp8, Nvfp4, Bf16p, Vq2, Q4 };
Form key_form(ik::KvCodec c) {
    switch (c) {
    case ik::KvCodec::Int8G64: return Form::Int8;
    case ik::KvCodec::Fp8Row:
    case ik::KvCodec::Fp8KeyNvfp4V: return Form::Fp8;
    case ik::KvCodec::Nvfp4G16: return Form::Nvfp4;
    case ik::KvCodec::Bf16Packed: return Form::Bf16p;
    case ik::KvCodec::Vq2: return Form::Vq2;
    case ik::KvCodec::Q4KeyVq2V: return Form::Q4;
    default: return Form::Words;
    }
}
Form value_form(ik::KvCodec c) {
    switch (c) {
    case ik::KvCodec::Int8G64: return Form::Int8;
    case ik::KvCodec::Fp8Row: return Form::Fp8;
    case ik::KvCodec::Nvfp4G16:
    case ik::KvCodec::Fp8KeyNvfp4V: return Form::Nvfp4;
    case ik::KvCodec::Bf16Packed: return Form::Bf16p;
    case ik::KvCodec::Vq2:
    case ik::KvCodec::Q4KeyVq2V: return Form::Vq2;
    default: return Form::Words;
    }
}
bool rotated(Form f) { return f == Form::Vq2 || f == Form::Q4; }
bool rotated(ik::KvCodec c) { return c == ik::KvCodec::Vq2 || c == ik::KvCodec::Q4KeyVq2V; }
int code_bytes(Form f) {
    return f == Form::Int8 || f == Form::Fp8 ? 128
           : f == Form::Nvfp4 || f == Form::Q4 ? 64
           : f == Form::Bf16p                 ? 192
           : f == Form::Vq2                   ? 32
                                              : 0;
}
int side_bytes(Form f) {
    return f == Form::Int8 ? 4 : f == Form::Fp8 || rotated(f) ? 2 : f == Form::Nvfp4 ? 8 : f == Form::Bf16p ? 16 : 0;
}
const char* codec_name(ik::KvCodec c) {
    switch (c) {
    case ik::KvCodec::Int8G64: return "int8";
    case ik::KvCodec::Fp8Row: return "fp8";
    case ik::KvCodec::Nvfp4G16: return "nvfp4";
    case ik::KvCodec::Fp8KeyNvfp4V: return "k8v4";
    case ik::KvCodec::Bf16Packed: return "bf16p";
    case ik::KvCodec::Vq2: return "vq2";
    case ik::KvCodec::Q4KeyVq2V: return "k4v2";
    default: return "bf16";
    }
}

// ---- the rotated forms at 128 (kv_codec.cuh vq2 / q4): a warp's lane l holds dims 4 l .. 4 l + 3.

float rounded_product(float a, float b); // below

// kv_hadamard128 in FP32: H4 over a lane's 4 (pairs (0, 1), (2, 3), then (0, 2), (1, 3)), H32 over the lanes (strides
// 1 .. 16: the lower lane x + y, the upper x - y), then times `scale` (a power of two).
void h128(float* v, float scale) {
    for (int l = 0; l < 32; ++l) {
        float* w = v + 4 * l;
        float a = w[0], b = w[1], c = w[2], d = w[3];
        w[0] = a + b, w[1] = a - b, w[2] = c + d, w[3] = c - d;
        a = w[0], b = w[1], c = w[2], d = w[3];
        w[0] = a + c, w[1] = b + d, w[2] = a - c, w[3] = b - d;
    }
    for (int s = 1; s <= 16; s <<= 1) {
        for (int l = 0; l < 32; ++l) {
            if ((l & s) != 0) { continue; }
            for (int j = 0; j < 4; ++j) {
                const float x = v[4 * l + j], y = v[4 * (l | s) + j];
                v[4 * l + j] = x + y, v[4 * (l | s) + j] = x - y;
            }
        }
    }
    for (int i = 0; i < D; ++i) { v[i] = v[i] * scale; }
}
// The FP64 transform 2^-4 H128 (the inverse of 2^-3 H128 up to rounding): o_i = 2^-4 sum_j (-1)^popc(i & j) x_j, by
// the fast transform's butterflies.
void h128_64(const double* x, double* o) {
    std::copy_n(x, D, o);
    for (int len = 1; len < D; len <<= 1) {
        for (int i = 0; i < D; i += 2 * len) {
            for (int j = i; j < i + len; ++j) {
                const double a = o[j], b = o[j + len];
                o[j] = a + b, o[j + len] = a - b;
            }
        }
    }
    for (int i = 0; i < D; ++i) { o[i] *= 0x1p-4; }
}
// Each row's largest |2^-4 H128 v| (the original domain's represented values of a rotated pool's rows).
std::vector<double> original_row_max(const std::vector<float>& values) {
    std::vector<double> out(values.size() / D);
    double x[D], o[D];
    for (std::size_t row = 0; row < out.size(); ++row) {
        for (int e = 0; e < D; ++e) { x[e] = values[row * D + e]; }
        h128_64(x, o);
        double m = 0;
        for (int e = 0; e < D; ++e) { m = std::max(m, std::fabs(o[e])); }
        out[row] = m;
    }
    return out;
}
// kv_row_dot: a lane's fma fold of its 4 products, then the xor tree over the 32 lanes.
float row_dot(const float* a, const float* b) {
    float lanes[32];
    for (int l = 0; l < 32; ++l) {
        float s = 0.0F;
        for (int j = 0; j < 4; ++j) { s = std::fma(a[4 * l + j], b[4 * l + j], s); }
        lanes[l] = s;
    }
    for (int o = 16; o >= 1; o >>= 1) {
        float next[32];
        for (int l = 0; l < 32; ++l) { next[l] = lanes[l] + lanes[l ^ o]; }
        std::copy_n(next, 32, lanes);
    }
    return lanes[0];
}
// A rotated row's signed integer levels c from its codes.
void rotated_levels(Form form, const std::uint8_t* codes, int* c) {
    for (int e = 0; e < D; ++e) {
        if (form == Form::Vq2) {
            const unsigned code  = codes[2 * (e / 8)] | (unsigned{codes[2 * (e / 8) + 1]} << 8);
            const unsigned seven = (code >> 9) & 0x7FU;
            const unsigned sign  = seven | ((std::popcount(seven) & 1U) << 7);
            const int v          = infernix::ops::kKVCacheVq2Codebook[8 * (code & 0x1FFU) + e % 8];
            c[e]                 = ((sign >> (e % 8)) & 1U) != 0 ? -v : v;
        } else {
            const int n = (codes[e / 2] >> (4 * (e % 2))) & 0xF;
            c[e]        = n >= 8 ? infernix::test::vq::q4_level(n - 8) : -infernix::test::vq::q4_level(7 - n);
        }
    }
}
// The encoder (kv_encode_vq2 / kv_encode_q4) over x, the BF16 storage's values: y = 2^-3 H128 x in FP32, the codes,
// S = fp16_rn(sum y^2 / sum y c) (2 side bytes); y returned (the distortion check's reference).
void rotated_encode(Form form, const float* x, std::uint8_t* codes, std::uint8_t* side, float* y) {
    std::copy_n(x, D, y);
    h128(y, 0x1p-3F);
    const float sum_sq = row_dot(y, y);
    const float sigma  = std::sqrt(sum_sq * (1.0F / 128.0F));
    std::fill(codes, codes + code_bytes(form), std::uint8_t{0});
    float c[D];
    if (form == Form::Vq2) {
        const float to_units = sigma > 0.0F ? 16.0F / sigma : 0.0F;
        for (int w = 0; w < 16; ++w) {
            float A[8];
            unsigned negative = 0;
            for (int j = 0; j < 8; ++j) {
                A[j] = rounded_product(std::fabs(y[8 * w + j]), to_units);
                negative |= (y[8 * w + j] < 0.0F ? 1U : 0U) << j;
            }
            const bool odd = (std::popcount(negative) & 1) != 0;
            int best_p     = 0;
            float best     = std::numeric_limits<float>::infinity();
            for (int p = 0; p < 512; ++p) {
                const float d = infernix::test::vq::vq2_distance(A, odd, p);
                if (d < best) { best = d, best_p = p; }
            }
            unsigned code = 0;
            for (int j = 0; j < 8; ++j) { c[8 * w + j] = 0.0F; }
            if (sigma > 0.0F) {
                unsigned sign = negative;
                if (odd) {
                    int flip = 0;
                    float m  = rounded_product(A[0], infernix::test::vq::codebook(best_p, 0));
                    for (int j = 1; j < 8; ++j) {
                        const float t = rounded_product(A[j], infernix::test::vq::codebook(best_p, j));
                        if (t < m) { m = t, flip = j; }
                    }
                    sign ^= 1U << flip;
                }
                for (int j = 0; j < 8; ++j) {
                    const int v  = infernix::ops::kKVCacheVq2Codebook[8 * best_p + j];
                    c[8 * w + j] = static_cast<float>(((sign >> j) & 1U) != 0 ? -v : v);
                }
                code = static_cast<unsigned>(best_p) | ((sign & 0x7FU) << 9);
            }
            codes[2 * w]     = static_cast<std::uint8_t>(code & 0xFFU);
            codes[2 * w + 1] = static_cast<std::uint8_t>(code >> 8);
        }
    } else {
        const float to_units = sigma > 0.0F ? 32.0F / sigma : 0.0F;
        for (int e = 0; e < D; ++e) {
            const float A  = rounded_product(std::fabs(y[e]), to_units);
            const int m    = (A > 8.0F) + (A > 16.5F) + (A > 25.5F) + (A > 35.0F) + (A > 46.0F) + (A > 59.0F) + (A > 76.5F);
            const bool neg = y[e] < 0.0F;
            int n          = neg ? 7 - m : 8 + m;
            c[e]           = static_cast<float>(neg ? -infernix::test::vq::q4_level(m) : infernix::test::vq::q4_level(m));
            if (sigma == 0.0F) { n = 7, c[e] = -4.0F; }
            codes[e / 2] |= static_cast<std::uint8_t>(n << (4 * (e % 2)));
        }
    }
    const float dot       = row_dot(y, c);
    const std::uint16_t s = h16(sigma > 0.0F && dot > 0.0F ? sum_sq / dot : 0.0F);
    std::memcpy(side, &s, 2);
}

// bf16p at 128 (inkling2_attention.h): the row's BF16 words packed; returns whether it has a packed form.
bool bf16p_encode(const std::uint16_t* bits, std::uint8_t* codes, std::uint8_t* side) {
    unsigned e[D];
    unsigned lowest = 255, highest = 0;
    for (int d = 0; d < D; ++d) {
        e[d] = (bits[d] >> 7) & 0xFFU;
        if (e[d] != 0 && e[d] != 255) { lowest = std::min(lowest, e[d]), highest = std::max(highest, e[d]); }
    }
    unsigned base = 1;
    if (highest >= lowest && highest - lowest < 14) {
        base = std::clamp(lowest, 1U, 240U);
    } else if (highest >= lowest) {
        unsigned best = 0;
        for (unsigned b = 1; b <= 240; ++b) {
            unsigned count = 0;
            for (int d = 0; d < D; ++d) { count += e[d] != 0 && e[d] != 255 && e[d] >= b && e[d] <= b + 13 ? 1 : 0; }
            if (count > best || b == 1) { best = count, base = b; } // the most, ties to the lowest base
        }
    }
    std::fill(codes, codes + 192, std::uint8_t{0});
    std::fill(side, side + 16, std::uint8_t{0});
    unsigned escapes = 0;
    bool zero        = false;
    for (int d = 0; d < D; ++d) {
        const bool normal     = e[d] != 0 && e[d] != 255;
        const unsigned nibble = e[d] == 0 ? 14U : normal && e[d] >= base && e[d] - base < 14 ? e[d] - base : 15U;
        codes[d]              = static_cast<std::uint8_t>(((bits[d] >> 8) & 0x80U) | (bits[d] & 0x7FU));
        codes[128 + d / 2] |= static_cast<std::uint8_t>(nibble << (4 * (d % 2)));
        if (nibble == 15U) {
            if (escapes < 14) { side[2 + escapes] = static_cast<std::uint8_t>(e[d]); }
            ++escapes;
        }
        zero = zero || e[d] == 0;
    }
    side[0] = static_cast<std::uint8_t>(base);
    side[1] = static_cast<std::uint8_t>(std::min(escapes, 14U) | (zero ? 0x80U : 0U));
    return escapes <= 14;
}
// A packed row's 128 BF16 words.
void bf16p_decode(const std::uint8_t* codes, const std::uint8_t* side, std::uint16_t* out) {
    unsigned index = 0;
    for (int d = 0; d < D; ++d) {
        const unsigned n        = (codes[128 + d / 2] >> (4 * (d % 2))) & 0xFU;
        const unsigned exponent = n < 14 ? side[0] + n : n == 14 ? 0U : index < 14 ? side[2 + index] : 0U;
        index += n == 15 ? 1 : 0;
        out[d] = static_cast<std::uint16_t>(((codes[d] & 0x80U) << 8) | ((exponent & 0xFFU) << 7) | (codes[d] & 0x7FU));
    }
}

float rounded_product(float a, float b) {
    volatile float product = a * b; // one rounding, never contracted
    return product;
}
float e4m3_value(std::uint8_t code) {
    const int exponent = (code >> 3) & 0x0F, mantissa = code & 0x07;
    const float magnitude = exponent == 0 ? std::ldexp(static_cast<float>(mantissa), -9)
                                          : std::ldexp(1.0F + static_cast<float>(mantissa) / 8.0F, exponent - 7);
    return (code & 0x80U) != 0 ? -magnitude : magnitude;
}
std::uint8_t e4m3_rne_satfinite(float value) {
    const bool negative   = std::signbit(value);
    const float magnitude = std::fabs(value);
    int selected          = 126;
    if (magnitude < 448.0F) {
        int upper = 0;
        while (upper < 126 && e4m3_value(static_cast<std::uint8_t>(upper)) < magnitude) { ++upper; }
        if (upper == 0) {
            selected = 0;
        } else {
            const int lower = upper - 1;
            const float lo = magnitude - e4m3_value(static_cast<std::uint8_t>(lower));
            const float hi = e4m3_value(static_cast<std::uint8_t>(upper)) - magnitude;
            selected       = lo < hi ? lower : hi < lo ? upper : (lower % 2 == 0 ? lower : upper);
        }
    }
    return static_cast<std::uint8_t>(selected | (negative ? 0x80 : 0));
}
float e2m1_value(std::uint8_t code) {
    constexpr float magnitude[8] = {0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F};
    return (code & 0x08U) != 0 ? -magnitude[code & 7] : magnitude[code & 7];
}
std::uint8_t e2m1_rne_satfinite(float value) {
    const bool negative   = std::signbit(value);
    const float magnitude = std::fabs(value);
    std::uint8_t selected = 7;
    if (magnitude < 6.0F) {
        for (std::uint8_t upper = 1; upper <= 7; ++upper) {
            if (e2m1_value(upper) < magnitude) { continue; }
            const auto lower = static_cast<std::uint8_t>(upper - 1);
            const float lo = magnitude - e2m1_value(lower), hi = e2m1_value(upper) - magnitude;
            selected = lo < hi ? lower : hi < lo ? upper : ((lower & 1U) == 0U ? lower : upper);
            break;
        }
    }
    return static_cast<std::uint8_t>(selected | (negative ? 0x08U : 0U));
}

// Encodes a row of 128 values (x: the BF16 K or FP16 V the BF16 storage would hold; bits: their BF16 source words,
// bf16p's input) into its codes and side record; returns the half-step bound of each value (the codec's own criterion:
// |represented - x| <= bound) and whether the row has a packed form (bf16p).
bool encode_row(Form form, const float* x, const std::uint16_t* bits, std::uint8_t* codes, std::uint8_t* side,
                double* bound, float* rotated_y = nullptr) {
    if (form == Form::Bf16p) {
        std::fill(bound, bound + D, 0.0);
        return bf16p_encode(bits, codes, side);
    }
    if (rotated(form)) { // no per-value bound: the rotated row's distortion is checked instead (rotated_y: its y)
        float y[D];
        rotated_encode(form, x, codes, side, y);
        if (rotated_y != nullptr) { std::copy_n(y, D, rotated_y); }
        std::fill(bound, bound + D, 0.0);
        return true;
    }
    if (form == Form::Int8) {
        for (int g = 0; g < 2; ++g) {
            float m = 0.0F;
            for (int e = 0; e < 64; ++e) { m = std::max(m, std::fabs(x[64 * g + e])); }
            const std::uint16_t s = h16(m > 0.0F ? m / 127.0F : 0.0F);
            std::memcpy(side + 2 * g, &s, 2);
            const float scale   = hf(s);
            const float inverse = scale > 0.0F ? 1.0F / scale : 0.0F;
            for (int e = 0; e < 64; ++e) {
                const float y  = rounded_product(x[64 * g + e], inverse);
                const int code = inverse == 0.0F ? 0 : std::clamp(static_cast<int>(std::nearbyint(y)), -127, 127);
                codes[64 * g + e] = static_cast<std::uint8_t>(static_cast<std::int8_t>(code));
                bound[64 * g + e] = 0.5 * scale * (1.0 + 0x1p-16) + (scale == 0.0F ? std::fabs(x[64 * g + e]) : 0.0);
            }
        }
    } else if (form == Form::Fp8) {
        float m = 0.0F;
        for (int e = 0; e < D; ++e) { m = std::max(m, std::fabs(x[e])); }
        const std::uint16_t s = m == 0.0F ? 0 : h16(std::clamp(m / 448.0F, 0x1p-24F, 65504.0F));
        std::memcpy(side, &s, 2);
        const float scale   = hf(s);
        const float inverse = m == 0.0F ? 0.0F : 1.0F / scale;
        for (int e = 0; e < D; ++e) {
            const float q = rounded_product(x[e], inverse);
            codes[e]      = inverse == 0.0F ? 0 : e4m3_rne_satfinite(q);
            // Half the E4M3 step at |q| (subnormal step 2^-9).
            const double a    = std::fabs(double{q});
            const double step = a < 0x1p-6 ? 0x1p-9 : std::ldexp(1.0, std::ilogb(a) - 3);
            bound[e]          = 0.5 * step * scale * (1.0 + 0x1p-16);
        }
    } else {
        constexpr float kTargets[5] = {6.0F, 4.0F, 4.5F, 5.0F, 5.5F};
        for (int g = 0; g < 8; ++g) {
            const float* v = x + 16 * g;
            float absmax   = 0.0F;
            for (int i = 0; i < 16; ++i) { absmax = std::max(absmax, std::fabs(v[i])); }
            side[g] = 0;
            std::fill(codes + 8 * g, codes + 8 * g + 8, std::uint8_t{0});
            float best = 0.0F;
            if (absmax != 0.0F) {
                for (int t = 0; t < 5; ++t) {
                    const float bounded = std::clamp(rounded_product(absmax, 1.0F / kTargets[t]), 0x1p-9F, 448.0F);
                    const std::uint8_t candidate = e4m3_rne_satfinite(bounded);
                    const float scale            = e4m3_value(candidate);
                    const float inverse          = 1.0F / scale;
                    std::uint8_t nib[16];
                    float error = 0.0F;
                    for (int i = 0; i < 16; ++i) {
                        nib[i]            = e2m1_rne_satfinite(rounded_product(v[i], inverse));
                        const float diff  = v[i] - rounded_product(e2m1_value(nib[i]), scale);
                        error             = error + rounded_product(diff, diff);
                    }
                    if (t == 0 || error < best) {
                        best    = error;
                        side[g] = candidate;
                        for (int p = 0; p < 8; ++p) { codes[8 * g + p] = static_cast<std::uint8_t>(nib[2 * p] | (nib[2 * p + 1] << 4)); }
                    }
                }
            }
            const float scale = e4m3_value(side[g]);
            for (int i = 0; i < 16; ++i) {
                const double a    = std::fabs(double{v[i]}) / (scale == 0.0F ? 1.0 : scale);
                const double step = a < 2.0 ? 0.5 : a < 4.0 ? 1.0 : 2.0;
                bound[16 * g + i] = scale == 0.0F ? std::fabs(v[i]) : 0.5 * step * scale * (1.0 + 0x1p-16);
            }
        }
    }
    return true;
}

// The represented value of element e of a row (K BF16, V FP16).
float decode_value(Form form, const std::uint8_t* codes, const std::uint8_t* side, int e, bool value) {
    float x = 0.0F;
    if (form == Form::Bf16p) {
        std::uint16_t row[D];
        bf16p_decode(codes, side, row);
        return value ? hf(h16(f(row[e]))) : f(row[e]);
    }
    if (rotated(form)) { // the rotated domain's value S c (exact), rounded as a K / V word
        int c[D];
        rotated_levels(form, codes, c);
        std::uint16_t s;
        std::memcpy(&s, side, 2);
        x = static_cast<float>(c[e]) * hf(s);
    } else if (form == Form::Int8) {
        std::uint16_t s;
        std::memcpy(&s, side + 2 * (e / 64), 2);
        x = static_cast<float>(static_cast<std::int8_t>(codes[e])) * hf(s);
    } else if (form == Form::Fp8) {
        std::uint16_t s;
        std::memcpy(&s, side, 2);
        x = e4m3_value(codes[e]) * hf(s);
    } else {
        const std::uint8_t byte = codes[e / 2];
        x = e2m1_value(static_cast<std::uint8_t>(e % 2 == 0 ? byte & 0x0F : byte >> 4)) * e4m3_value(side[e / 16]);
    }
    return value ? hf(h16(x)) : f(b16(x));
}

// A row's codes and side record in a plane (pool: the BF16 layout's words, its bytes reused; row = element / 128).
std::uint8_t* codes_of(std::vector<std::uint16_t>& pool, Form form, std::size_t element) {
    return reinterpret_cast<std::uint8_t*>(pool.data()) + element / D * code_bytes(form);
}
const std::uint8_t* codes_of(const std::vector<std::uint16_t>& pool, Form form, std::size_t element) {
    return reinterpret_cast<const std::uint8_t*>(pool.data()) + element / D * code_bytes(form);
}

// A pool's represented values by element offset (a BF16 pool: its own words, K BF16 and V FP16).
std::vector<float> represented(Form form, const std::vector<std::uint16_t>& pool, const std::vector<std::uint8_t>& side,
                               bool value) {
    std::vector<float> out(pool.size());
    for (std::size_t i = 0; i < pool.size(); ++i) {
        const std::size_t row = i / D;
        if (form == Form::Bf16p) {
            if (i % D == 0) {
                std::uint16_t words[D];
                bf16p_decode(codes_of(pool, form, i), side.data() + row * side_bytes(form), words);
                for (int d = 0; d < D; ++d) { out[i + d] = value ? hf(h16(f(words[d]))) : f(words[d]); }
            }
            continue;
        }
        if (rotated(form)) {
            if (i % D == 0) {
                int c[D];
                rotated_levels(form, codes_of(pool, form, i), c);
                std::uint16_t s;
                std::memcpy(&s, side.data() + row * side_bytes(form), 2);
                for (int d = 0; d < D; ++d) {
                    const float x = static_cast<float>(c[d]) * hf(s);
                    out[i + d]    = value ? hf(h16(x)) : f(b16(x));
                }
            }
            continue;
        }
        out[i] = form == Form::Words ? (value ? hf(pool[i]) : f(pool[i]))
                                     : decode_value(form, codes_of(pool, form, i), side.data() + row * side_bytes(form),
                                                    static_cast<int>(i % D), value);
    }
    return out;
}
// Encodes every row of a BF16 pool (K BF16 words, V FP16 words) into codes and side records in place (a V row's BF16
// source: the BF16 rounding of its FP16 value).
void encode_pool(Form form, std::vector<std::uint16_t>& pool, std::vector<std::uint8_t>& side, bool value) {
    const std::vector<std::uint16_t> words = pool;
    std::fill(pool.begin(), pool.end(), std::uint16_t{0});
    side.assign(words.size() / D * side_bytes(form), 0);
    const std::size_t rows = words.size() / D;
    // Rows are independent: the rotated forms' exhaustive host search (~65K multiply-adds a row) runs over threads.
    const unsigned workers = rotated(form) ? std::max(1U, std::min(16U, std::thread::hardware_concurrency())) : 1U;
    const auto encode_rows = [&](unsigned w) {
        float x[D];
        std::uint16_t bits[D];
        double bound[D];
        for (std::size_t row = w; row < rows; row += workers) {
            const std::size_t at = row * D;
            for (int e = 0; e < D; ++e) {
                x[e]    = value ? hf(words[at + e]) : f(words[at + e]);
                bits[e] = b16(x[e]);
            }
            encode_row(form, x, bits, codes_of(pool, form, at), side.data() + row * side_bytes(form), bound);
        }
    };
    std::vector<std::thread> threads;
    for (unsigned w = 1; w < workers; ++w) { threads.emplace_back(encode_rows, w); }
    encode_rows(0);
    for (std::thread& t : threads) { t.join(); }
}

// ---------------------------------------------------------------------------- the call

struct Call {
    std::vector<int> lane, pos, first = {0, 0, 0, 8013};
    [[nodiscard]] int size() const { return static_cast<int>(lane.size()); }
};
Call the_call() {
    Call c;
    for (int p = 1100; p < 1104; ++p) { c.lane.push_back(0), c.pos.push_back(p); }
    for (int p = 60; p < 82; ++p) { c.lane.push_back(1), c.pos.push_back(p); } // two groups: 16 + 6 columns
    c.lane.push_back(-1), c.pos.push_back(0);
    c.lane.push_back(2), c.pos.push_back(5000);
    for (int p = 9000; p < 9002; ++p) { c.lane.push_back(3), c.pos.push_back(p); }
    return c;
}

// ---------------------------------------------------------------------------- the problem

struct Problem {
    bool full = false;
    ik::KvCodec codec = ik::KvCodec::Bf16; // the full pool's form
    // Projections [T][width].
    std::vector<std::uint16_t> q, r, k, v, sink, gate, ktaps, vtaps;
    // Layer.
    std::vector<std::uint16_t> qscale, rscale, table, kconv, vconv;
    // State: KV (ring [4][8][17][64][128] or pool [8][256 pages][64][128]; quantized: row r's codes at byte r * code
    // bytes of the pool's words, its side record at r * side bytes), its block table, conv rings.
    std::vector<std::uint16_t> kv_k, kv_v, kring, vring;
    std::vector<std::uint8_t> k_side, v_side;
    std::vector<int> block_table;
    [[nodiscard]] Form form(int which) const { return which == 0 ? key_form(codec) : value_form(codec); }
    [[nodiscard]] bool quantized() const { return codec != ik::KvCodec::Bf16; }
};

std::size_t kv_row(const Problem& p, int lane, int head, int pos) {
    if (!p.full) {
        const int page = (pos / U) % ik::kWindowRingPages;
        return ((std::size_t(lane * NKV + head) * ik::kWindowRingPages + page) * U + pos % U) * D;
    }
    const int page = p.block_table[std::size_t(lane) * kTableStride + pos / U];
    return ((std::size_t(page) * NKV + head) * U + pos % U) * D; // page-major pool
}

Problem make_problem(bool full, const Call& call, std::uint32_t seed, ik::KvCodec codec = ik::KvCodec::Bf16) {
    std::mt19937 gen(seed);
    const auto normal = [&](std::size_t n, float sd) {
        std::normal_distribution<float> d(0.0F, sd);
        std::vector<std::uint16_t> out(n);
        for (auto& x : out) { x = b16(d(gen)); }
        return out;
    };
    const int T = call.size();
    Problem p;
    p.full  = full;
    p.q     = normal(std::size_t(T) * QW, 1.0F);
    p.r     = normal(std::size_t(T) * RW, 1.0F);
    p.k     = normal(std::size_t(T) * KVW, 1.0F);
    p.v     = normal(std::size_t(T) * KVW, 1.0F);
    p.sink  = normal(std::size_t(T) * NH, 2.0F);
    p.gate  = normal(std::size_t(T) * QW, 1.5F);
    p.ktaps = normal(std::size_t(T) * NKV * 4, 0.3F);
    p.vtaps = normal(std::size_t(T) * NKV * 4, 0.3F);
    p.qscale = normal(std::size_t(NH) * D, 0.15F);
    for (std::size_t i = 0; i < p.qscale.size(); ++i) {
        const int head = static_cast<int>(i / D);
        p.qscale[i]    = b16(f(p.qscale[i]) + (head % 4 == 0 ? 3.0F : 1.0F)); // heads 0, 4, ... clip
    }
    p.rscale = normal(std::size_t(NH) * RD, 0.5F);
    for (auto& x : p.rscale) { x = b16(f(x) + 2.0F); }
    p.table = normal(std::size_t(RD) * ik::kRelativeExtent, 0.4F);
    p.kconv = normal(std::size_t(4) * KVW, 0.4F);
    p.vconv = normal(std::size_t(4) * KVW, 0.4F);
    p.kring = normal(std::size_t(kLanes) * R * KVW, 1.0F);
    p.vring = normal(std::size_t(kLanes) * R * KVW, 1.0F);
    const std::size_t kv_words = full ? std::size_t(NKV) * kPoolPages * U * D
                                      : std::size_t(kLanes) * NKV * ik::kWindowRingPages * U * D;
    p.kv_k = normal(kv_words, 1.0F);
    p.kv_v = normal(kv_words, 1.0F);
    for (auto& x : p.kv_v) { x = h16(f(x)); } // the represented FP16 V of a BF16 value
    if (full) {
        // Lanes' pages in a scattered order.
        std::vector<int> pages(kPoolPages);
        for (int i = 0; i < kPoolPages; ++i) { pages[i] = i; }
        std::shuffle(pages.begin(), pages.end(), gen);
        p.block_table.assign(std::size_t(kLanes) * kTableStride, -1);
        const int need[kLanes] = {1104 / U + 1, 76 / U + 1, 5000 / U + 1, 9002 / U + 1};
        int next = 0;
        for (int l = 0; l < kLanes; ++l) {
            for (int b = 0; b < need[l]; ++b) { p.block_table[std::size_t(l) * kTableStride + b] = pages[next++]; }
        }
        check(next <= kPoolPages, "the test's pool holds its lanes");
    }
    p.codec = full ? codec : ik::KvCodec::Bf16;
    if (p.codec == ik::KvCodec::Bf16Packed) {
        // Escapes (exponents below a row's 14-binade window; normal rows span fewer binades, so they have none): every
        // fifth pool row holds 1 + row % 6 values far below the rest (K +-2^-40, V +-2^-20: an FP16 subnormal), so the
        // decode reads escape records; column 0's new key row gets 5 (packable) from channels 32-36 of KV head 0 and
        // their conv history at 2^-100.
        const std::size_t rows = p.kv_k.size() / D;
        for (std::size_t row = 0; row < rows; row += 5) {
            for (int j = 0; j <= static_cast<int>(row % 6); ++j) {
                const std::size_t at = row * D + static_cast<std::size_t>((37 * j + 11) % D);
                p.kv_k[at]           = b16((j & 1) != 0 ? -0x1p-40F : 0x1p-40F);
                p.kv_v[at]           = h16((j & 1) != 0 ? -0x1p-20F : 0x1p-20F);
            }
        }
        for (int ch = 32; ch < 37; ++ch) {
            p.k[ch] = b16(0x1p-100F);
            for (int back = 1; back <= 3; ++back) { p.kring[(std::size_t(0) * R + (1100 - back) % R) * KVW + ch] = b16(0x1p-100F); }
        }
    }
    if (p.quantized()) {
        encode_pool(p.form(0), p.kv_k, p.k_side, false);
        encode_pool(p.form(1), p.kv_v, p.v_side, true);
    }
    return p;
}

// The out rows' A8 encoding (--a8-activations: W_o's 8-bit inputs): a column's 3,072 codes then 96 FP32 scales.
constexpr int kOut8Bytes = QW + QW / 8;

struct Device {
    std::uint16_t *q, *r, *k, *v, *sink, *gate, *ktaps, *vtaps, *qscale, *rscale, *table, *kconv, *vconv;
    std::uint16_t *kv_k, *kv_v, *kring, *vring, *out;
    std::uint8_t* out8; // [T][kOut8Bytes]
    std::uint8_t *k_side, *v_side;
    std::uint32_t* unpackable; // [kLanes]
    int *block_table, *lane, *pos, *first, *groups;
    void* ws;
};

// Runs the attention over columns `subset` of the call as one call; the state lives in `state` (kv and rings).
struct Outputs {
    std::vector<std::uint16_t> out, q, r, kv_k, kv_v, kring, vring;
    std::vector<std::uint8_t> out8;
    std::vector<std::uint8_t> k_side, v_side;
    std::vector<std::uint32_t> unpackable;
};

Outputs run(const Problem& p, const Call& call, const std::vector<int>& subset, Device& d, bool fused = false) {
    const int T = static_cast<int>(subset.size());
    std::vector<int> lane, pos, groups;
    int widest = 1;
    for (int i = 0; i < T; ++i) { lane.push_back(call.lane[subset[i]]), pos.push_back(call.pos[subset[i]]); }
    // Each lane's columns in groups of at most 16, as the forward splits them.
    for (int l = 0; l < kLanes; ++l) {
        int begin = -1, count = 0;
        for (int i = 0; i < T; ++i) {
            if (lane[i] == l) {
                if (begin < 0) { begin = i; }
                ++count;
            }
        }
        for (int done = 0; done < count; done += ik::kMaxAttentionLaneColumns) {
            const int n = std::min(ik::kMaxAttentionLaneColumns, count - done);
            groups.insert(groups.end(), {begin + done, n, l});
            widest = std::max(widest, n);
        }
    }
    const auto pick = [&](const std::vector<std::uint16_t>& v, int width) {
        std::vector<std::uint16_t> o(std::size_t(T) * width);
        for (int i = 0; i < T; ++i) { std::copy_n(v.data() + std::size_t(subset[i]) * width, width, o.data() + std::size_t(i) * width); }
        return o;
    };
    upload_to(d.q, pick(p.q, QW));
    upload_to(d.r, pick(p.r, RW));
    upload_to(d.k, pick(p.k, KVW));
    upload_to(d.v, pick(p.v, KVW));
    upload_to(d.sink, pick(p.sink, NH));
    upload_to(d.gate, pick(p.gate, QW));
    upload_to(d.ktaps, pick(p.ktaps, NKV * 4));
    upload_to(d.vtaps, pick(p.vtaps, NKV * 4));
    upload_to(d.lane, lane);
    upload_to(d.pos, pos);
    upload_to(d.groups, groups);
    ik::AttentionArgs a;
    a.columns           = T;
    a.groups            = static_cast<int>(groups.size() / 3);
    a.max_group_columns = widest;
    a.map               = {d.lane, d.pos, d.first, T};
    a.groups_columns    = d.groups;
    a.in               = {d.q, d.r, d.k, d.v, d.sink, d.gate, p.full ? d.ktaps : nullptr, p.full ? d.vtaps : nullptr};
    a.layer            = {p.full, d.qscale, d.rscale, d.table, d.kconv, d.vconv};
    a.key_ring         = {d.kring, KVW, std::int64_t{R} * KVW};
    a.value_ring       = {d.vring, KVW, std::int64_t{R} * KVW};
    a.kv.k            = d.kv_k;
    a.kv.v            = d.kv_v;
    a.kv.lane_stride  = std::int64_t{NKV} * ik::kWindowRingPages * U * D;
    a.kv.table        = p.full ? d.block_table : nullptr;
    a.kv.table_stride = p.full ? kTableStride : 0;
    a.kv.pool_pages   = p.full ? kPoolPages : 0;
    a.kv.codec        = p.codec;
    a.kv.k_side       = p.quantized() ? d.k_side : nullptr;
    a.kv.v_side       = p.quantized() ? d.v_side : nullptr;
    a.kv.unpackable   = p.codec == ik::KvCodec::Bf16Packed ? d.unpackable : nullptr;
    a.ws               = ik::carve_attention_workspace(d.ws);
    a.out              = d.out;
    a.out8             = reinterpret_cast<std::int32_t*>(d.out8);
    a.out8_stride      = kOut8Bytes / 4;
    a.fused            = fused;
    ik::attention(a, nullptr);
    cuda_check(cudaDeviceSynchronize(), "attention");
    Outputs o;
    o.out   = download(d.out, std::size_t(T) * QW);
    o.out8  = download(d.out8, std::size_t(T) * kOut8Bytes);
    o.q     = download(a.ws.q, std::size_t(T) * QW);
    o.r     = download(a.ws.r, std::size_t(T) * RW);
    o.kv_k  = download(d.kv_k, p.kv_k.size());
    o.kv_v  = download(d.kv_v, p.kv_v.size());
    if (p.quantized()) {
        o.k_side = download(d.k_side, p.k_side.size());
        o.v_side = download(d.v_side, p.v_side.size());
    }
    o.unpackable = download(d.unpackable, kLanes);
    o.kring = download(d.kring, p.kring.size());
    o.vring = download(d.vring, p.vring.size());
    return o;
}

void reset_state(const Problem& p, Device& d) {
    upload_to(d.kv_k, p.kv_k);
    upload_to(d.kv_v, p.kv_v);
    if (p.quantized()) {
        upload_to(d.k_side, p.k_side);
        upload_to(d.v_side, p.v_side);
    }
    upload_to(d.unpackable, std::vector<std::uint32_t>(kLanes, 0xFFFFFFFFU));
    upload_to(d.kring, p.kring);
    upload_to(d.vring, p.vring);
}

// The pools (and a quantized pool's side records) of two runs: the same bits.
void same_pools(Tally& tally, const Outputs& a, const Outputs& b) {
    tally.exact(a.kv_k.data(), b.kv_k.data(), b.kv_k.size());
    tally.exact(a.kv_v.data(), b.kv_v.data(), b.kv_v.size());
    tally.exact(a.k_side.data(), b.k_side.data(), b.k_side.size());
    tally.exact(a.v_side.data(), b.v_side.data(), b.v_side.size());
    tally.exact(a.unpackable.data(), b.unpackable.data(), b.unpackable.size());
}

// ---------------------------------------------------------------------------- host prologue emulation

float tau_of(const Problem& p, int pos) { return p.full ? ik::tau(pos) : 1.0F; }

// The conv input of channel ch of column t, k positions back (0, the call's column, or the pre-call ring).
float history(const Call& c, int t, int k, int ch, const std::vector<std::uint16_t>& raw,
              const std::vector<std::uint16_t>& ring) {
    const int lane = c.lane[t];
    if (k == 0) { return f(raw[std::size_t(t) * KVW + ch]); }
    if (lane < 0 || c.pos[t] - k < c.first[lane]) { return 0.0F; }
    if (t - k >= 0 && c.lane[t - k] == lane) { return f(raw[std::size_t(t - k) * KVW + ch]); }
    return f(ring[(std::size_t(lane) * R + (c.pos[t] - k) % R) * KVW + ch]);
}

void prologue_check(const Problem& p, const Call& call, const Outputs& got, const char* name) {
    Tally tally{name}, codec{"  codec: each value within half a step"};
    // Rotated forms: every row's |S c - y|^2 / |y|^2 within 0.15 (vq2: 2 bits a value, the Gaussian rate-distortion
    // floor 0.0625) or 0.02 (q4: 16-level Lloyd-Max, ~0.0095) plus the row's overload term (below); q' within half a
    // BF16 ulp of the FP64 rotation of the emulated query plus 2^-18 of its largest |q| (seven FP32 butterfly roundings).
    Tally distortion{"  codec: rotated rows' |S c - y|^2 / |y|^2 within 0.15 (vq2) / 0.02 (q4) + overload"};
    Tally rotation{"  q' = bf16(2^-4 H128 q) vs FP64"};
    double mean_distortion[2] = {0, 0};
    int distortion_rows[2]    = {0, 0};
    const int T = call.size();
    std::vector<std::uint16_t> kv_k = p.kv_k, kv_v = p.kv_v, kring = p.kring, vring = p.vring;
    std::vector<std::uint8_t> k_side = p.k_side, v_side = p.v_side;
    const bool quantized = p.quantized();
    std::vector<std::uint32_t> flags(kLanes, 0xFFFFFFFFU); // bf16p: each lane's lowest unpackable position
    for (int t = 0; t < T; ++t) {
        const int lane = call.lane[t], pos = call.pos[t];
        if (lane < 0) {
            std::vector<std::uint16_t> zero(QW, 0);
            tally.exact(got.out.data() + std::size_t(t) * QW, zero.data(), QW);
            continue;
        }
        for (int kv = 0; kv < NKV; ++kv) {
            for (int which = 0; which < 2; ++which) {
                const auto& raw  = which == 0 ? p.k : p.v;
                const auto& ring = which == 0 ? p.kring : p.vring;
                const auto& conv = which == 0 ? p.kconv : p.vconv;
                const auto& taps = which == 0 ? p.ktaps : p.vtaps;
                float c[D];
                double c64[D];
                for (int e = 0; e < D; ++e) {
                    const int ch = kv * D + e;
                    float w[4], x[4], tap[4];
                    double acc = 0;
                    for (int lag = 0; lag < 4; ++lag) {
                        w[lag]   = f(conv[std::size_t(lag) * KVW + ch]);
                        x[lag]   = history(call, t, lag, ch, raw, ring);
                        tap[lag] = f(taps[std::size_t(t) * NKV * 4 + kv * 4 + lag]);
                        acc += (double{w[lag]} + (p.full ? double{tap[lag]} : 0.0)) * x[lag];
                    }
                    c[e]   = bf(p.full ? ik::conv_dynamic(w, tap, x) : ik::conv_static(w, x));
                    c64[e] = (acc + x[0]) / (p.full ? 3.0 : std::sqrt(5.0));
                }
                const float sum = ik::head_dot_reference(c, c);
                double s64      = 0;
                for (int e = 0; e < D; ++e) { s64 += double{c[e]} * c[e]; }
                const std::size_t at = kv_row(p, lane, kv, pos);
                // held: the BF16 storage's value (a quantized form's encoder input); its FP64 check is the kernel's on
                // a BF16 pool and the emulation's (bit for bit the kernel's there) on a quantized one.
                float held[D];
                std::uint16_t src[D]; // their BF16 source words (bf16p's input)
                for (int e = 0; e < D; ++e) {
                    if (which == 0) {
                        const float y        = bf(canon::mul_rn(c[e], ik::clip_factor(sum, 22.627416997969522F)));
                        held[e]              = y;
                        src[e]               = b16(y);
                        if (!quantized) { kv_k[at + e] = b16(y); }
                        const double y64     = c[e] * std::min(1.0, 22.627416997969522 / (std::sqrt(s64) + 1e-6));
                        tally.ratio(std::fabs((quantized ? y : f(got.kv_k[at + e])) - y64) /
                                    (0.5 * ulp_bf16(y64) + ulp_bf16(c64[e]) + 1e-30));
                    } else {
                        const float y    = bf(canon::mul_rn(c[e], ik::rms_factor(sum, 128.0F)));
                        held[e]          = hf(h16(y));
                        src[e]           = b16(y);
                        if (!quantized) { kv_v[at + e] = h16(y); }
                        const double y64 = c[e] / std::sqrt(s64 / 128.0 + 1e-6);
                        tally.ratio(std::fabs((quantized ? held[e] : hf(got.kv_v[at + e])) - y64) /
                                    (0.5 * ulp_bf16(y64) + ulp_f16(y64) + ulp_bf16(c64[e]) + 1e-30));
                    }
                }
                if (quantized) {
                    const Form form     = p.form(which);
                    auto& pool          = which == 0 ? kv_k : kv_v;
                    std::uint8_t* side  = (which == 0 ? k_side : v_side).data() + at / D * side_bytes(form);
                    std::uint8_t* codes = codes_of(pool, form, at);
                    double bound[D];
                    float y[D];
                    if (!encode_row(form, held, src, codes, side, bound, y)) { flags[lane] = std::min(flags[lane], static_cast<std::uint32_t>(pos)); }
                    if (rotated(form)) {
                        // The rotated row's distortion |S c - y|^2 / |y|^2 (y = 2^-3 H128 x, S c as represented).
                        double err = 0, norm = 0;
                        for (int e = 0; e < D; ++e) {
                            const double d = double{decode_value(form, codes, side, e, which == 1)} - y[e];
                            err += d * d, norm += double{y[e]} * y[e];
                        }
                        // The bound: the Gaussian-row figure plus the row's own overload term, sum (|y| - top)_+^2 /
                        // |y|^2 with top the form's largest level at the row's sigma (q4 87/32, vq2 127/16 sigma). As
                        // first run (13:28) the bound had no overload term and one q4 key row (column 5, KV head 2:
                        // 0.0237, a coordinate at 4.16 sigma against the top level's 2.72) failed it.
                        const double limit = form == Form::Vq2 ? 0.15 : 0.02;
                        const double sigma = std::sqrt(norm / D);
                        const double top   = (form == Form::Vq2 ? 127.0 / 16.0 : 87.0 / 32.0) * sigma;
                        double over = 0, peak = 0;
                        for (int e = 0; e < D; ++e) {
                            const double a = std::fabs(double{y[e]});
                            over += a > top ? (a - top) * (a - top) : 0.0;
                            peak = std::max(peak, a);
                        }
                        const double bound = norm == 0.0 ? 0.0 : limit + over / norm;
                        distortion.ratio(norm == 0.0 ? err : err / norm / bound);
                        if (norm != 0.0 && err / norm > limit) {
                            std::printf("    row past the Gaussian figure: %s %s, column %d KV head %d: %.4f (bound %.4f "
                                        "with its overload), max |y| / sigma %.2f (reported)\n",
                                        form == Form::Vq2 ? "vq2" : "q4", which == 0 ? "key" : "value", t, kv, err / norm,
                                        bound, peak / sigma);
                        }
                        mean_distortion[form == Form::Vq2 ? 0 : 1] += norm == 0.0 ? 0.0 : err / norm;
                        ++distortion_rows[form == Form::Vq2 ? 0 : 1];
                    } else {
                        for (int e = 0; e < D; ++e) {
                            const float rep = decode_value(form, codes, side, e, which == 1);
                            codec.ratio(std::fabs(double{rep} - held[e]) /
                                        (bound[e] + 0.5 * (which == 0 ? ulp_bf16(rep) : ulp_f16(rep)) + 1e-30));
                        }
                    }
                }
                auto& out_ring = which == 0 ? kring : vring;
                for (int e = 0; e < D; ++e) {
                    out_ring[(std::size_t(lane) * R + pos % R) * KVW + kv * D + e] = raw[std::size_t(t) * KVW + kv * D + e];
                }
            }
        }
        for (int h = 0; h < NH; ++h) {
            float s[D];
            double s64 = 0;
            for (int e = 0; e < D; ++e) {
                s[e] = canon::mul_rn(f(p.q[std::size_t(t) * QW + h * D + e]), f(p.qscale[h * D + e]));
                s64 += double{s[e]} * s[e];
            }
            const float fq = ik::clip_factor(ik::head_dot_reference(s, s), 22.627416997969522F);
            const double f64 = std::min(1.0, 22.627416997969522 / (std::sqrt(s64) + 1e-6));
            std::uint16_t want[D];
            const bool rot = p.full && rotated(p.codec);
            for (int e = 0; e < D; ++e) {
                float y = bf(canon::mul_rn(s[e], fq));
                double y64 = s[e] * f64;
                if (p.full) {
                    const double inner = y64;
                    y                  = bf(canon::mul_rn(y, tau_of(p, pos)));
                    y64                = bf(static_cast<float>(inner)) * (1.0 + 0.1 * std::log(std::max(1.0, (pos + 1) / 8192.0)));
                }
                want[e] = b16(y);
                // A rotated pool's kernel stores q' (checked below): its FP64 stage check is the emulation's (q' is
                // bit for bit the rotation of this q).
                tally.ratio(std::fabs((rot ? y : f(got.q[(std::size_t(t) * NH + h) * D + e])) - y64) /
                            (0.5 * ulp_bf16(y64) + (p.full ? ulp_bf16(y64) : 0.0) + 1e-30));
            }
            if (rot) {
                float qv[D];
                double x64[D], r64[D], qmax = 0;
                for (int e = 0; e < D; ++e) { qv[e] = f(want[e]), x64[e] = qv[e], qmax = std::max(qmax, std::fabs(x64[e])); }
                h128_64(x64, r64);
                h128(qv, 0x1p-4F);
                for (int e = 0; e < D; ++e) {
                    want[e] = b16(qv[e]);
                    rotation.ratio(std::fabs(f(got.q[(std::size_t(t) * NH + h) * D + e]) - r64[e]) /
                                   (0.5 * ulp_bf16(r64[e]) + 0x1p-18 * qmax + 1e-30));
                }
            }
            tally.exact(got.q.data() + (std::size_t(t) * NH + h) * D, want, D);
            float rs[RD];
            float sum = 0;
            double r64 = 0;
            for (int i = 0; i < RD; ++i) {
                rs[i] = canon::mul_rn(f(p.r[std::size_t(t) * RW + h * RD + i]), f(p.rscale[h * RD + i]));
                sum   = canon::fma_rn(rs[i], rs[i], sum);
                r64 += double{rs[i]} * rs[i];
            }
            const float fr = ik::clip_factor(sum, 8.0F);
            std::uint16_t rwant[RD];
            for (int i = 0; i < RD; ++i) {
                float y = bf(canon::mul_rn(rs[i], fr));
                if (p.full) { y = bf(canon::mul_rn(y, tau_of(p, pos))); }
                rwant[i] = b16(y);
                const double y64 = rs[i] * std::min(1.0, 8.0 / (std::sqrt(r64) + 1e-6));
                if (!p.full) { tally.ratio(std::fabs(f(got.r[(std::size_t(t) * NH + h) * RD + i]) - y64) / (0.5 * ulp_bf16(y64) + 1e-30)); }
            }
            tally.exact(got.r.data() + (std::size_t(t) * NH + h) * RD, rwant, RD);
        }
    }
    tally.exact(got.kv_k.data(), kv_k.data(), kv_k.size());
    tally.exact(got.kv_v.data(), kv_v.data(), kv_v.size());
    tally.exact(got.k_side.data(), k_side.data(), k_side.size());
    tally.exact(got.v_side.data(), v_side.data(), v_side.size());
    if (p.codec == ik::KvCodec::Bf16Packed) { tally.exact(got.unpackable.data(), flags.data(), flags.size()); }
    tally.exact(got.kring.data(), kring.data(), kring.size());
    tally.exact(got.vring.data(), vring.data(), vring.size());
    tally.report();
    if (quantized && !(rotated(p.form(0)) && rotated(p.form(1)))) { codec.report(); }
    if (p.full && rotated(p.codec)) {
        distortion.report();
        rotation.report();
        for (int i = 0; i < 2; ++i) {
            if (distortion_rows[i] > 0) {
                std::printf("    mean %s row distortion %.4f over %d rows (reported)\n", i == 0 ? "vq2" : "q4",
                            mean_distortion[i] / distortion_rows[i], distortion_rows[i]);
            }
        }
    }
}

// ---------------------------------------------------------------------------- attention against FP64

void attention_check(const Problem& p, const Call& call, const Outputs& got, const char* name) {
    Tally tally{name}, bias_tally{"  bias table (pinned FP32 fold vs FP64)"};
    const int T = call.size();
    const std::vector<float> keys   = represented(p.form(0), got.kv_k, got.k_side, false);
    const std::vector<float> values = represented(p.form(1), got.kv_v, got.v_side, true);
    // A rotated pool (fixed before the first run): scores over the stored q' and the rotated represented keys, the
    // output 2^-4 H128 of the FP64 rotated output; the FP32 term through that transform's infinity-norm gain of 8
    // (2^-13 of the rows' largest rotated |v|, which also covers the kernel's seven FP32 butterfly roundings).
    const bool rot = p.full && rotated(p.codec);
    for (int t = 0; t < T; ++t) {
        const int lane = call.lane[t], pos = call.pos[t];
        if (lane < 0) { continue; }
        for (int h = 0; h < NH; ++h) {
            const int kv = h / G;
            const std::uint16_t* q = got.q.data() + (std::size_t(t) * NH + h) * D;
            const std::uint16_t* r = got.r.data() + (std::size_t(t) * NH + h) * RD;
            const int lo = std::max(call.first[lane], p.full ? 0 : std::max(0, pos - (ik::kWindow - 1)));
            std::vector<double> score;
            std::vector<const float*> vrows;
            double vmax = 0;
            for (int j = lo; j <= pos; ++j) {
                const std::size_t at = kv_row(p, lane, kv, j);
                double dot = 0;
                for (int e = 0; e < D; ++e) { dot += double{f(q[e])} * keys[at + e]; }
                const int dist = pos - j;
                double bias = 0;
                if (dist < ik::kRelativeExtent) {
                    float acc = 0;
                    double b64 = 0;
                    for (int i = 0; i < RD; ++i) {
                        const float pv = f(p.table[std::size_t(i) * ik::kRelativeExtent + dist]);
                        acc            = canon::fma_rn(f(r[i]), pv, acc);
                        b64 += double{f(r[i])} * pv;
                    }
                    bias = bf(acc);
                    bias_tally.ratio(std::fabs(bias - b64) / (ulp_bf16(b64) + 1e-30));
                }
                score.push_back(dot / std::sqrt(128.0) + bias);
                vrows.push_back(values.data() + at);
                for (int e = 0; e < D; ++e) { vmax = std::max(vmax, std::fabs(double{values[at + e]})); }
            }
            const double sink = f(p.sink[std::size_t(t) * NH + h]);
            double m = sink;
            for (double s : score) { m = std::max(m, s); }
            double den = std::exp(sink - m);
            for (double& s : score) { den += (s = std::exp(s - m)); }
            double accs[D], unrot[D];
            for (int e = 0; e < D; ++e) {
                double acc = 0;
                for (std::size_t j = 0; j < score.size(); ++j) { acc += score[j] * vrows[j][e]; }
                accs[e] = acc;
            }
            if (rot) {
                h128_64(accs, unrot);
                std::copy_n(unrot, D, accs);
            }
            for (int e = 0; e < D; ++e) {
                const double a64   = accs[e] / den;
                const double g     = f(p.gate[std::size_t(t) * QW + h * D + e]);
                const double sg    = 1.0 / (1.0 + std::exp(-g));
                const double out64 = a64 * sg;
                const double y     = f(got.out[std::size_t(t) * QW + h * D + e]);
                tally.ratio(std::fabs(y - out64) /
                            (0.5 * ulp_bf16(out64) + sg * ulp_bf16(a64) + (rot ? 0x1p-13 : 0x1p-16) * vmax));
            }
        }
    }
    bias_tally.report();
    tally.report();
}

void run_case(bool full, std::uint32_t seed, ik::KvCodec codec = ik::KvCodec::Bf16) {
    const Call call  = the_call();
    const int T      = call.size();
    const Problem p  = make_problem(full, call, seed, codec);
    char kind[32];
    std::snprintf(kind, sizeof kind, "%s%s%s", full ? "full" : "sliding", p.quantized() ? " " : "",
                  p.quantized() ? codec_name(p.codec) : "");
    char names[5][96];
    std::snprintf(names[0], sizeof names[0], "%s prologue (emulation, FP64)", kind);
    std::snprintf(names[1], sizeof names[1], "%s attention vs FP64", kind);
    std::snprintf(names[2], sizeof names[2], "  %s: one column at a time = the call", kind);
    std::snprintf(names[3], sizeof names[3], "  %s: the fused prologue = the prologue kernel", kind);
    std::snprintf(names[4], sizeof names[4], "  %s: repeated call", kind);
    Device d{};
    d.q = dalloc<std::uint16_t>(std::size_t(T) * QW), d.r = dalloc<std::uint16_t>(std::size_t(T) * RW);
    d.k = dalloc<std::uint16_t>(std::size_t(T) * KVW), d.v = dalloc<std::uint16_t>(std::size_t(T) * KVW);
    d.sink = dalloc<std::uint16_t>(std::size_t(T) * NH), d.gate = dalloc<std::uint16_t>(std::size_t(T) * QW);
    d.ktaps = dalloc<std::uint16_t>(std::size_t(T) * NKV * 4), d.vtaps = dalloc<std::uint16_t>(std::size_t(T) * NKV * 4);
    d.qscale = upload(p.qscale), d.rscale = upload(p.rscale), d.table = upload(p.table);
    d.kconv = upload(p.kconv), d.vconv = upload(p.vconv);
    d.kv_k = upload(p.kv_k), d.kv_v = upload(p.kv_v), d.kring = upload(p.kring), d.vring = upload(p.vring);
    d.k_side = p.quantized() ? upload(p.k_side) : dalloc<std::uint8_t>(1);
    d.v_side = p.quantized() ? upload(p.v_side) : dalloc<std::uint8_t>(1);
    d.unpackable = upload(std::vector<std::uint32_t>(kLanes, 0xFFFFFFFFU)); // none, as reset_state leaves them
    d.out = dalloc<std::uint16_t>(std::size_t(T) * QW);
    d.out8 = upload(std::vector<std::uint8_t>(std::size_t(T) * kOut8Bytes, 0xFF)); // garbage: every byte the kernel's
    d.block_table = full ? upload(p.block_table) : dalloc<int>(1);
    d.lane = dalloc<int>(T), d.pos = dalloc<int>(T), d.first = upload(call.first), d.groups = dalloc<int>(3 * ik::kMaxAttentionGroups);
    d.ws = dalloc<unsigned char>(ik::attention_workspace_bytes());
    // Garbage in the output, so the padding column's zeros are the kernel's.
    upload_to(d.out, std::vector<std::uint16_t>(std::size_t(T) * QW, 0x7FC0));

    std::vector<int> all(T);
    for (int t = 0; t < T; ++t) { all[t] = t; }
    const Outputs got = run(p, call, all, d);
    prologue_check(p, call, got, names[0]);
    attention_check(p, call, got, names[1]);
    {
        // The out rows' A8 encoding (W_o's 8-bit inputs): the codec of ops/ggml/ggml_a8.h applied to the kernel's own BF16
        // out by its definition, exactly; padding columns all zero (a zero row's encoding).
        namespace a8 = infernix::ggml::a8;
        Tally q8{"  out's A8 rows (exact encoder)"};
        std::vector<std::uint8_t> want(std::size_t(T) * kOut8Bytes, 0);
        for (int t = 0; t < T; ++t) {
            const std::uint16_t* x = got.out.data() + std::size_t(t) * QW;
            std::uint8_t* row      = want.data() + std::size_t(t) * kOut8Bytes;
            for (int s = 0; s < QW / 32; ++s) {
                unsigned key = 0;
                for (int j = 0; j < 32; ++j) { key = std::max(key, static_cast<unsigned>(x[32 * s + j] & 0x7FFFU)); }
                const a8::SliceScale sc = a8::slice_scale(key);
                for (int j = 0; j < 32; ++j) {
                    row[32 * s + j] = static_cast<std::uint8_t>(static_cast<std::int8_t>(a8::code(x[32 * s + j], sc.inv)));
                }
                std::memcpy(row + QW + 4 * s, &sc.d, 4);
            }
        }
        q8.exact(got.out8.data(), want.data(), want.size());
        q8.report();
    }

    // A second identical call over the same initial state: the same bits (no state left in the workspace).
    reset_state(p, d);
    const Outputs again = run(p, call, all, d);
    Tally repeat{names[4]};
    repeat.exact(again.out.data(), got.out.data(), got.out.size());
    repeat.report();

    // Decode against verify.
    reset_state(p, d);
    Tally inv{names[2]};
    Outputs one;
    for (int t = 0; t < T; ++t) {
        one = run(p, call, {t}, d);
        if (call.lane[t] >= 0) { inv.exact(one.out.data(), got.out.data() + std::size_t(t) * QW, QW); }
    }
    same_pools(inv, one, got);
    inv.exact(one.kring.data(), got.kring.data(), got.kring.size());
    inv.exact(one.vring.data(), got.vring.data(), got.vring.size());
    inv.report();

    // The prologue inside the attention launch (fused): the same bits. Over a call without padding whose lanes are one
    // group each (lanes 0, 2 and 3: the forward fuses exactly such calls), and one column at a time.
    std::vector<int> grouped;
    for (int t = 0; t < T; ++t) {
        if (call.lane[t] == 0 || call.lane[t] == 2 || call.lane[t] == 3) { grouped.push_back(t); }
    }
    reset_state(p, d);
    const Outputs plain = run(p, call, grouped, d);
    reset_state(p, d);
    const Outputs fused = run(p, call, grouped, d, true);
    Tally fu{names[3]};
    fu.exact(fused.out.data(), plain.out.data(), plain.out.size());
    fu.exact(fused.q.data(), plain.q.data(), plain.q.size());
    fu.exact(fused.r.data(), plain.r.data(), plain.r.size());
    same_pools(fu, fused, plain);
    fu.exact(fused.kring.data(), plain.kring.data(), plain.kring.size());
    fu.exact(fused.vring.data(), plain.vring.data(), plain.vring.size());
    reset_state(p, d);
    Outputs one_fused;
    for (int t = 0; t < T; ++t) {
        if (call.lane[t] < 0) { continue; }
        one_fused = run(p, call, {t}, d, true);
        fu.exact(one_fused.out.data(), got.out.data() + std::size_t(t) * QW, QW);
    }
    same_pools(fu, one_fused, got);
    fu.exact(one_fused.kring.data(), got.kring.data(), got.kring.size());

    // A NaN column (its q all NaN: every visible score NaN beside the masked keys of its unit): the call completes
    // (a NaN once split a warp before a full-mask shuffle) and every other column keeps its bits.
    if (!p.quantized()) {
        constexpr int kNan = 4; // lane 1, position 60: its unit holds masked keys past the position
        Problem poisoned = p;
        std::fill_n(poisoned.q.begin() + std::size_t(kNan) * QW, QW, std::uint16_t{0x7FC0});
        reset_state(poisoned, d);
        const Outputs nan_out = run(poisoned, call, all, d);
        Tally nan_tally{full ? "  full: a NaN column leaves the others' bits" : "  sliding: a NaN column leaves the others' bits"};
        for (int t = 0; t < T; ++t) {
            if (t == kNan || call.lane[t] < 0) { continue; }
            nan_tally.exact(nan_out.out.data() + std::size_t(t) * QW, got.out.data() + std::size_t(t) * QW, QW);
        }
        nan_tally.report();
    }

    if (p.codec == ik::KvCodec::Bf16Packed) {
        // Lossless: the BF16 storage holding the packed pool's decoded words gives the same outputs and appended words.
        Problem words = p;
        words.codec   = ik::KvCodec::Bf16;
        words.k_side.clear(), words.v_side.clear();
        const std::vector<float> keys = represented(Form::Bf16p, p.kv_k, p.k_side, false);
        const std::vector<float> vals = represented(Form::Bf16p, p.kv_v, p.v_side, true);
        for (std::size_t i = 0; i < keys.size(); ++i) { words.kv_k[i] = b16(keys[i]), words.kv_v[i] = h16(vals[i]); }
        reset_state(words, d);
        const Outputs plain = run(words, call, all, d);
        Tally lossless{"  full bf16p: = the bf16 storage of its words"};
        lossless.exact(plain.out.data(), got.out.data(), got.out.size());
        const std::vector<float> got_keys = represented(Form::Bf16p, got.kv_k, got.k_side, false);
        const std::vector<float> got_vals = represented(Form::Bf16p, got.kv_v, got.v_side, true);
        std::vector<std::uint16_t> packed_k(got_keys.size()), packed_v(got_vals.size());
        for (std::size_t i = 0; i < got_keys.size(); ++i) { packed_k[i] = b16(got_keys[i]), packed_v[i] = h16(got_vals[i]); }
        lossless.exact(plain.kv_k.data(), packed_k.data(), packed_k.size());
        lossless.exact(plain.kv_v.data(), packed_v.data(), packed_v.size());
        lossless.report();

        // A key row with no packed form: lane 0's first column's KV head 0 channels 0-15 and their conv history at
        // 2^-100 (sixteen exponents far below the window: more escapes than the record holds) sets lane 0's flag alone.
        Problem odd = p;
        for (int ch = 0; ch < 16; ++ch) {
            odd.k[ch] = b16(0x1p-100F);
            for (int back = 1; back <= 3; ++back) {
                odd.kring[(std::size_t(0) * R + (1100 - back) % R) * KVW + ch] = b16(0x1p-100F);
            }
        }
        reset_state(odd, d);
        const Outputs flagged = run(odd, call, all, d);
        check(flagged.unpackable.size() == kLanes && flagged.unpackable[0] == 1100U && flagged.unpackable[1] == 0xFFFFFFFFU &&
                  flagged.unpackable[2] == 0xFFFFFFFFU && flagged.unpackable[3] == 0xFFFFFFFFU,
              "bf16p: only lane 0 is flagged");
        prologue_check(odd, call, flagged, "full bf16p: a key row with no packed form (emulation)");
    }
    fu.exact(one_fused.vring.data(), got.vring.data(), got.vring.size());
    fu.report();

    for (void* x : {static_cast<void*>(d.q), static_cast<void*>(d.r), static_cast<void*>(d.k), static_cast<void*>(d.v),
                    static_cast<void*>(d.sink), static_cast<void*>(d.gate), static_cast<void*>(d.ktaps),
                    static_cast<void*>(d.vtaps), static_cast<void*>(d.qscale), static_cast<void*>(d.rscale),
                    static_cast<void*>(d.table), static_cast<void*>(d.kconv), static_cast<void*>(d.vconv),
                    static_cast<void*>(d.kv_k), static_cast<void*>(d.kv_v), static_cast<void*>(d.k_side),
                    static_cast<void*>(d.v_side), static_cast<void*>(d.unpackable), static_cast<void*>(d.kring),
                    static_cast<void*>(d.vring), static_cast<void*>(d.out), static_cast<void*>(d.out8), static_cast<void*>(d.block_table),
                    static_cast<void*>(d.lane), static_cast<void*>(d.pos), static_cast<void*>(d.first),
                    static_cast<void*>(d.groups), d.ws}) {
        cudaFree(x);
    }
}

// ---------------------------------------------------------------------------- the prompt route

// attention_prompt on one lane's prompt call of 150 columns at positions 1000-1149 (three CTA tiles of up to 64 columns;
// sliding windows across 17 units, full rows over all units from the lane's first position 37). Sliding KV is a
// staging ring of staging_ring_pages(150) pages (the forward's for a wide call); the conv inputs before the call come
// from a history snapshot (as the forward's for a wide call). Criteria (fixed before the first run): each output
// within half a BF16 ulp of FP64 (scores over the kernel's stored q, r and represented K/V, the bias at its pinned
// BF16 boundary, softmax with the sink) plus sigmoid(gate) times (one ulp of the inner BF16 a + 2^-11 of the row's
// largest |v|: p is rounded to FP16 for P x V) plus 2^-16 of the row's largest |v|; the call as calls of 37, 64 and 49
// columns from the same initial state gives the same bits; a repeated call the same bits. A quantized full pool
// (`codec`): the same criteria over its decoded represented values, and the chunked calls' codes and side records the
// one call's.
void prompt_case(bool full, std::uint32_t seed, ik::KvCodec codec = ik::KvCodec::Bf16) {
    constexpr int T = 150, P0 = 1000, First = 37;
    const int pages = ik::staging_ring_pages(T);
    std::mt19937 gen(seed);
    const auto normal = [&](std::size_t n, float sd) {
        std::normal_distribution<float> d(0.0F, sd);
        std::vector<std::uint16_t> out(n);
        for (auto& x : out) { x = b16(d(gen)); }
        return out;
    };
    // Projections and layer as make_problem's.
    auto q = normal(std::size_t(T) * QW, 1.0F), r = normal(std::size_t(T) * RW, 1.0F);
    auto k = normal(std::size_t(T) * KVW, 1.0F), v = normal(std::size_t(T) * KVW, 1.0F);
    auto sink = normal(std::size_t(T) * NH, 2.0F), gate = normal(std::size_t(T) * QW, 1.5F);
    auto ktaps = normal(std::size_t(T) * NKV * 4, 0.3F), vtaps = normal(std::size_t(T) * NKV * 4, 0.3F);
    auto qscale = normal(std::size_t(NH) * D, 0.15F);
    for (std::size_t i = 0; i < qscale.size(); ++i) { qscale[i] = b16(f(qscale[i]) + ((i / D) % 4 == 0 ? 3.0F : 1.0F)); }
    auto rscale = normal(std::size_t(NH) * RD, 0.5F);
    for (auto& x : rscale) { x = b16(f(x) + 2.0F); }
    auto table = normal(std::size_t(RD) * ik::kRelativeExtent, 0.4F);
    auto kconv = normal(std::size_t(4) * KVW, 0.4F), vconv = normal(std::size_t(4) * KVW, 0.4F);
    auto kring = normal(std::size_t(R) * KVW, 1.0F), vring = normal(std::size_t(R) * KVW, 1.0F);
    const std::size_t kv_words = full ? std::size_t(NKV) * kPoolPages * U * D : std::size_t(NKV) * pages * U * D;
    auto kv_k = normal(kv_words, 1.0F), kv_v = normal(kv_words, 1.0F);
    for (auto& x : kv_v) { x = h16(f(x)); }
    std::vector<int> block_table;
    if (full) {
        std::vector<int> order(kPoolPages);
        for (int i = 0; i < kPoolPages; ++i) { order[i] = i; }
        std::shuffle(order.begin(), order.end(), gen);
        block_table.assign(kTableStride, -1);
        for (int b = 0; b <= (P0 + T - 1) / U; ++b) { block_table[b] = order[b]; }
    }
    if (!full) { codec = ik::KvCodec::Bf16; }
    const bool quantized = codec != ik::KvCodec::Bf16;
    const Form kform = key_form(codec), vform = value_form(codec);
    std::vector<std::uint8_t> k_side, v_side;
    if (quantized) {
        encode_pool(kform, kv_k, k_side, false);
        encode_pool(vform, kv_v, v_side, true);
    }
    char kind[32];
    std::snprintf(kind, sizeof kind, "%s%s%s", full ? "full" : "sliding", quantized ? " " : "",
                  quantized ? codec_name(codec) : "");
    char names[3][96];
    std::snprintf(names[0], sizeof names[0], "%s prompt attention vs FP64", kind);
    std::snprintf(names[1], sizeof names[1], "  %s: repeated prompt call", kind);
    std::snprintf(names[2], sizeof names[2], "  %s: calls of 37, 64, 49 = one call", kind);
    const auto row_of = [&](int head, int pos) -> std::size_t {
        if (!full) { return ((std::size_t(head) * pages + (pos / U) % pages) * U + pos % U) * D; }
        return ((std::size_t(block_table[pos / U]) * NKV + head) * U + pos % U) * D; // page-major pool
    };
    // Device state.
    std::uint16_t *dq = upload(q), *dr = upload(r), *dk = upload(k), *dv = upload(v), *dsink = upload(sink);
    std::uint16_t *dgate = upload(gate), *dktaps = upload(ktaps), *dvtaps = upload(vtaps);
    std::uint16_t *dqs = upload(qscale), *drs = upload(rscale), *dtable = upload(table), *dkc = upload(kconv),
                  *dvc = upload(vconv);
    std::uint16_t *dkr = upload(kring), *dvr = upload(vring), *dkk = upload(kv_k), *dvv = upload(kv_v);
    std::uint8_t *dks = quantized ? upload(k_side) : dalloc<std::uint8_t>(1),
                 *dvs = quantized ? upload(v_side) : dalloc<std::uint8_t>(1);
    std::uint32_t* dflags = dalloc<std::uint32_t>(1);
    std::uint16_t *dkh = dalloc<std::uint16_t>(std::size_t(ik::kHistorySlots) * KVW),
                  *dvh = dalloc<std::uint16_t>(std::size_t(ik::kHistorySlots) * KVW);
    std::uint16_t* dout = dalloc<std::uint16_t>(std::size_t(T) * QW);
    int* dbt = full ? upload(block_table) : dalloc<int>(1);
    int *dlane = dalloc<int>(T), *dpos = dalloc<int>(T), *dfirst = upload(std::vector<int>{First});
    int* dgroups = dalloc<int>(3 * (T / 16 + 2));
    void* dws = dalloc<unsigned char>(ik::attention_workspace_bytes(T));
    const auto reset = [&] {
        upload_to(dkk, kv_k), upload_to(dvv, kv_v), upload_to(dkr, kring), upload_to(dvr, vring);
        if (quantized) { upload_to(dks, k_side), upload_to(dvs, v_side); }
    };
    struct Got {
        std::vector<std::uint16_t> out, q, r, kk, vv;
        std::vector<std::uint8_t> ks, vs;
    };
    // Columns [c0, c0 + n) of the call as one prompt call.
    const auto run_prompt = [&](int c0, int n) {
        // The history snapshot: ring rows of positions P0 + c0 - 4 .. - 1 at slots pos & 3.
        const auto kr = download(dkr, kring.size()), vr = download(dvr, vring.size());
        std::vector<std::uint16_t> kh(std::size_t(ik::kHistorySlots) * KVW), vh(kh.size());
        for (int back = 1; back <= ik::kHistorySlots; ++back) {
            const int pos = P0 + c0 - back;
            std::copy_n(kr.data() + std::size_t(pos & (R - 1)) * KVW, KVW, kh.data() + std::size_t(pos & 3) * KVW);
            std::copy_n(vr.data() + std::size_t(pos & (R - 1)) * KVW, KVW, vh.data() + std::size_t(pos & 3) * KVW);
        }
        upload_to(dkh, kh), upload_to(dvh, vh);
        const auto slice = [&](const std::vector<std::uint16_t>& src, int width) {
            return std::vector<std::uint16_t>(src.begin() + std::size_t(c0) * width, src.begin() + std::size_t(c0 + n) * width);
        };
        upload_to(dq, slice(q, QW)), upload_to(dr, slice(r, RW)), upload_to(dk, slice(k, KVW)), upload_to(dv, slice(v, KVW));
        upload_to(dsink, slice(sink, NH)), upload_to(dgate, slice(gate, QW));
        upload_to(dktaps, slice(ktaps, NKV * 4)), upload_to(dvtaps, slice(vtaps, NKV * 4));
        std::vector<int> lane(n, 0), pos(n), groups;
        for (int i = 0; i < n; ++i) { pos[i] = P0 + c0 + i; }
        for (int done = 0; done < n; done += ik::kMaxAttentionLaneColumns) {
            groups.insert(groups.end(), {done, std::min(ik::kMaxAttentionLaneColumns, n - done), 0});
        }
        upload_to(dlane, lane), upload_to(dpos, pos), upload_to(dgroups, groups);
        ik::AttentionArgs a;
        a.columns           = n;
        a.groups            = static_cast<int>(groups.size() / 3);
        a.max_group_columns = std::min(n, ik::kMaxAttentionLaneColumns);
        a.map               = {dlane, dpos, dfirst, n};
        a.groups_columns    = dgroups;
        a.host_groups       = groups.data();
        a.in                = {dq, dr, dk, dv, dsink, dgate, full ? dktaps : nullptr, full ? dvtaps : nullptr};
        a.layer             = {full, dqs, drs, dtable, dkc, dvc};
        a.key_ring          = {dkr, KVW, std::int64_t{R} * KVW, dkh, std::int64_t{ik::kHistorySlots} * KVW};
        a.value_ring        = {dvr, KVW, std::int64_t{R} * KVW, dvh, std::int64_t{ik::kHistorySlots} * KVW};
        a.kv.k              = dkk;
        a.kv.v              = dvv;
        a.kv.ring_pages     = pages;
        a.kv.lane_stride    = std::int64_t{NKV} * pages * U * D;
        a.kv.table          = full ? dbt : nullptr;
        a.kv.table_stride   = full ? kTableStride : 0;
        a.kv.pool_pages     = full ? kPoolPages : 0;
        a.kv.codec          = codec;
        a.kv.k_side         = quantized ? dks : nullptr;
        a.kv.v_side         = quantized ? dvs : nullptr;
        a.kv.unpackable     = codec == ik::KvCodec::Bf16Packed ? dflags : nullptr;
        a.ws                = ik::carve_attention_workspace(dws, T);
        a.out               = dout;
        ik::attention_prompt(a, nullptr);
        cuda_check(cudaDeviceSynchronize(), "attention_prompt");
        Got g;
        g.out = download(dout, std::size_t(n) * QW);
        g.q   = download(a.ws.q, std::size_t(n) * QW);
        g.r   = download(a.ws.r, std::size_t(n) * RW);
        g.kk  = download(dkk, kv_k.size());
        g.vv  = download(dvv, kv_v.size());
        if (quantized) { g.ks = download(dks, k_side.size()), g.vs = download(dvs, v_side.size()); }
        return g;
    };
    reset();
    const Got got = run_prompt(0, T);
    const std::vector<float> keys   = represented(kform, got.kk, got.ks, false);
    const std::vector<float> values = represented(vform, got.vv, got.vs, true);
    // A rotated pool (fixed before the first run): as attention_check's, the FP16 p term over the original domain's
    // largest |v| (its error is linear in p: sum dp v'_rot -> sum dp v_orig), the FP32 term 2^-13 of the rotated one.
    const bool rot = full && rotated(codec);
    const std::vector<double> orig_max = rot ? original_row_max(values) : std::vector<double>{};
    // FP64 over the kernel's own q, r and represented K/V.
    Tally tally{names[0]};
    for (int t = 0; t < T; ++t) {
        const int pos = P0 + t;
        for (int h = 0; h < NH; ++h) {
            const int kvh = h / G;
            const std::uint16_t* qv = got.q.data() + (std::size_t(t) * NH + h) * D;
            const std::uint16_t* rv = got.r.data() + (std::size_t(t) * NH + h) * RD;
            const int lo = std::max(First, full ? 0 : std::max(0, pos - (ik::kWindow - 1)));
            std::vector<double> score;
            std::vector<std::size_t> rows;
            double vmax = 0, omax = 0;
            for (int j = lo; j <= pos; ++j) {
                const std::size_t at = row_of(kvh, j);
                double dot = 0;
                for (int e = 0; e < D; ++e) { dot += double{f(qv[e])} * keys[at + e]; }
                const int dist = pos - j;
                double bias    = 0;
                if (dist < ik::kRelativeExtent) {
                    float acc = 0;
                    for (int i = 0; i < RD; ++i) {
                        acc = canon::fma_rn(f(rv[i]), f(table[std::size_t(i) * ik::kRelativeExtent + dist]), acc);
                    }
                    bias = bf(acc);
                }
                score.push_back(dot / std::sqrt(128.0) + bias);
                rows.push_back(at);
                for (int e = 0; e < D; ++e) { vmax = std::max(vmax, std::fabs(double{values[at + e]})); }
                omax = rot ? std::max(omax, orig_max[at / D]) : vmax;
            }
            const double s_sink = f(sink[std::size_t(t) * NH + h]);
            double m = s_sink;
            for (double s : score) { m = std::max(m, s); }
            double den = std::exp(s_sink - m);
            for (double& s : score) { den += (s = std::exp(s - m)); }
            double accs[D], unrot[D];
            for (int e = 0; e < D; ++e) {
                double acc = 0;
                for (std::size_t j = 0; j < score.size(); ++j) { acc += score[j] * values[rows[j] + e]; }
                accs[e] = acc;
            }
            if (rot) {
                h128_64(accs, unrot);
                std::copy_n(unrot, D, accs);
            }
            for (int e = 0; e < D; ++e) {
                const double a64 = accs[e] / den;
                const double sg  = 1.0 / (1.0 + std::exp(-double{f(gate[std::size_t(t) * QW + h * D + e])}));
                const double y   = f(got.out[std::size_t(t) * QW + h * D + e]);
                tally.ratio(std::fabs(y - a64 * sg) / (0.5 * ulp_bf16(a64 * sg) + sg * (ulp_bf16(a64) + 0x1p-11 * omax) +
                                                       (rot ? 0x1p-13 : 0x1p-16) * vmax));
            }
        }
    }
    tally.report();
    reset();
    const Got again = run_prompt(0, T);
    Tally repeat{names[1]};
    repeat.exact(again.out.data(), got.out.data(), got.out.size());
    repeat.report();
    reset();
    Tally chunks{names[2]};
    int c0 = 0;
    Got part;
    for (const int n : {37, 64, 49}) {
        part = run_prompt(c0, n);
        chunks.exact(part.out.data(), got.out.data() + std::size_t(c0) * QW, std::size_t(n) * QW);
        c0 += n;
    }
    if (quantized) {
        chunks.exact(part.kk.data(), got.kk.data(), got.kk.size());
        chunks.exact(part.vv.data(), got.vv.data(), got.vv.size());
        chunks.exact(part.ks.data(), got.ks.data(), got.ks.size());
        chunks.exact(part.vs.data(), got.vs.data(), got.vs.size());
    }
    chunks.report();
    for (void* x : {static_cast<void*>(dq), static_cast<void*>(dr), static_cast<void*>(dk), static_cast<void*>(dv),
                    static_cast<void*>(dsink), static_cast<void*>(dgate), static_cast<void*>(dktaps),
                    static_cast<void*>(dvtaps), static_cast<void*>(dqs), static_cast<void*>(drs), static_cast<void*>(dtable),
                    static_cast<void*>(dkc), static_cast<void*>(dvc), static_cast<void*>(dkr), static_cast<void*>(dvr),
                    static_cast<void*>(dkk), static_cast<void*>(dvv), static_cast<void*>(dks), static_cast<void*>(dvs),
                    static_cast<void*>(dflags), static_cast<void*>(dkh), static_cast<void*>(dvh),
                    static_cast<void*>(dout), static_cast<void*>(dbt), static_cast<void*>(dlane), static_cast<void*>(dpos),
                    static_cast<void*>(dfirst), static_cast<void*>(dgroups), dws}) {
        cudaFree(x);
    }
}

} // namespace

INFERNIX_OP_TEST(inkling2_attention) {
    run_case(false, 7);
    run_case(true, 8);
    prompt_case(false, 17);
    prompt_case(true, 18);
    // The quantized full pools (each form in both planes; k8v4 mixes them).
    std::uint32_t seed = 30;
    for (const ik::KvCodec codec : {ik::KvCodec::Int8G64, ik::KvCodec::Fp8Row, ik::KvCodec::Nvfp4G16,
                                    ik::KvCodec::Fp8KeyNvfp4V, ik::KvCodec::Bf16Packed, ik::KvCodec::Vq2,
                                    ik::KvCodec::Q4KeyVq2V}) {
        run_case(true, seed++, codec);
        prompt_case(true, seed++, codec);
    }
    std::printf(g_failures == 0 ? "PASS\n" : "FAILED (%d)\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
