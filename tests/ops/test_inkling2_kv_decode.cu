// Inkling 2's int8 G64 piece decoder (kv_int8_piece.cuh kv_decode8_int8, the int8 attention instances' decode of pool and
// window-ring rows) against an exact oracle, bit for bit, exhaustively: every code byte (256) under every FP16 scale bit
// pattern (65,536: subnormal, infinite and NaN scales included), for keys (BF16 words) and values (FP16 words).
//
// Oracle (host): the code as a signed integer q and the scale s decoded from their bits; q * s formed in FP64 (exact:
// 8 by 11 significant bits) and rounded to nearest even once to the word's format (BF16 for keys, FP16 for values), by
// bit arithmetic on the FP64 value. A NaN product's key word is 0x7FC0 (canon::f32_to_bf16_rn's), a value's the
// device conversion's NaN (any NaN is accepted there).

#include "ops/inkling2/kv_int8_piece.cuh"
#include "ops/op_family.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace detail = infernix::ops::inkling2::detail;
using infernix::test::cuda_check;

namespace {

// Block b: scale pattern b, 32 threads; thread t decodes piece t of the 256 code bytes (codes 8 t .. 8 t + 7) as keys
// and as values into out[plane][scale][256].
__global__ void decode_kernel(std::uint16_t* out) {
    const unsigned scale = blockIdx.x;
    const int t          = static_cast<int>(threadIdx.x);
    std::uint8_t c[8];
    for (int i = 0; i < 8; ++i) { c[i] = static_cast<std::uint8_t>(8 * t + i); }
    uint2 codes;
    std::memcpy(&codes, c, 8);
    const uint4 k = detail::kv_decode8_int8<false>(codes, static_cast<std::uint16_t>(scale));
    const uint4 v = detail::kv_decode8_int8<true>(codes, static_cast<std::uint16_t>(scale));
    const std::uint32_t kw[4] = {k.x, k.y, k.z, k.w}, vw[4] = {v.x, v.y, v.z, v.w};
    for (int i = 0; i < 8; ++i) {
        out[(0 * 65536ULL + scale) * 256 + 8 * t + i] = static_cast<std::uint16_t>(kw[i / 2] >> (16 * (i % 2)));
        out[(1 * 65536ULL + scale) * 256 + 8 * t + i] = static_cast<std::uint16_t>(vw[i / 2] >> (16 * (i % 2)));
    }
}

double half_value(std::uint16_t h) {
    const int e = (h >> 10) & 0x1F, m = h & 0x3FF;
    const double sign = (h & 0x8000) != 0 ? -1.0 : 1.0;
    if (e == 0x1F) { return m != 0 ? std::nan("") : sign * INFINITY; }
    return sign * (e == 0 ? std::ldexp(m, -24) : std::ldexp(1024 + m, e - 25));
}

// x rounded to nearest even to a format of `mbits` stored mantissa bits, exponent bias `bias`, `ebits` exponent bits;
// returns the format's bits (an infinity past its largest finite value).
std::uint16_t round_to(double x, int mbits, int ebits, int bias) {
    const std::uint16_t sign = std::signbit(x) ? static_cast<std::uint16_t>(1U << (mbits + ebits)) : 0;
    const double a = std::fabs(x);
    const std::uint16_t inf = static_cast<std::uint16_t>(sign | (((1U << ebits) - 1) << mbits));
    if (std::isinf(a)) { return inf; }
    if (a == 0.0) { return sign; }
    int e = std::ilogb(a);                        // a in [2^e, 2^(e+1))
    const int emin = 1 - bias;
    const int q    = std::max(e, emin) - mbits;   // the quantum of a's binade (subnormals: emin's)
    const double scaled = std::ldexp(a, -q);      // exact (a power-of-two scaling)
    double n = std::floor(scaled);
    const double frac = scaled - n;
    if (frac > 0.5 || (frac == 0.5 && std::fmod(n, 2.0) != 0.0)) { n += 1.0; }
    const double r   = std::ldexp(n, q);
    const double max = std::ldexp(2.0 - std::ldexp(1.0, -mbits), (1 << (ebits - 1)) - 1); // the largest finite value
    if (r > max) { return inf; } // (a tie above max rounds to the next power of two: past max)
    if (r < std::ldexp(1.0, emin)) { return static_cast<std::uint16_t>(sign | static_cast<unsigned>(std::ldexp(r, mbits - emin))); }
    e = std::ilogb(r);
    const unsigned mant = static_cast<unsigned>(std::ldexp(r, mbits - e)) - (1U << mbits);
    return static_cast<std::uint16_t>(sign | (static_cast<unsigned>(e + bias) << mbits) | mant);
}

} // namespace

INFERNIX_OP_TEST(inkling2_kv_decode) {
    (void)argc, (void)argv;
    const std::size_t words = 2ULL * 65536 * 256;
    std::uint16_t* d = nullptr;
    cuda_check(cudaMalloc(&d, words * sizeof(std::uint16_t)), "malloc");
    decode_kernel<<<65536, 32>>>(d);
    cuda_check(cudaGetLastError(), "launch");
    std::vector<std::uint16_t> got(words);
    cuda_check(cudaMemcpy(got.data(), d, words * sizeof(std::uint16_t), cudaMemcpyDeviceToHost), "copy");
    cuda_check(cudaFree(d), "free");
    std::size_t bad[2] = {0, 0}, nans = 0, subnormal = 0;
    for (int plane = 0; plane < 2; ++plane) {
        for (unsigned s = 0; s < 65536; ++s) {
            const double scale = half_value(static_cast<std::uint16_t>(s));
            for (int b = 0; b < 256; ++b) {
                const double x = static_cast<double>(static_cast<std::int8_t>(b)) * scale; // exact
                const std::uint16_t g = got[(plane * 65536ULL + s) * 256 + b];
                bool ok;
                if (std::isnan(x)) {
                    ok = plane == 0 ? g == 0x7FC0U : ((g & 0x7C00U) == 0x7C00U && (g & 0x3FFU) != 0);
                    ++nans;
                } else {
                    const std::uint16_t want = plane == 0 ? round_to(x, 7, 8, 127) : round_to(x, 10, 5, 15);
                    ok = g == want;
                    if (x != 0.0 && std::fabs(x) < (plane == 0 ? 0x1p-126 : 0x1p-14)) { ++subnormal; }
                }
                if (!ok && bad[plane]++ == 0) {
                    std::printf("  first %s mismatch: scale 0x%04X code 0x%02X (q * s = %.17g): got 0x%04X\n",
                                plane == 0 ? "key" : "value", s, b, x, g);
                }
            }
        }
    }
    std::printf("int8 piece decoder: 256 codes x 65536 scales: %zu key and %zu value words differ from the exact "
                "rounding (%zu NaN products, %zu subnormal results checked)\n",
                bad[0], bad[1], nans, subnormal);
    std::printf(bad[0] + bad[1] == 0 ? "PASS\n" : "FAILED\n");
    return bad[0] + bad[1] == 0 ? 0 : 1;
}
