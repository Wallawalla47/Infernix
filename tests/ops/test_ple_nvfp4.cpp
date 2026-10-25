// PLE Ops for NVIDIA Qwen3.8-Flash-Next-FP4-Local (docs/maintainer/qwen3_8-flash-next-design.md §12.5):
//
// - ple_embed_nvfp4 decodes NVFP4 n-gram rows (W/2 E2M1 code bytes, then W/16 E4M3FN scales) times the
//   table's FP32 multiplier. The oracle forms the exact product in FP64 (at most 30 significant bits)
//   and rounds it once to BF16, to nearest even; the Op must equal it bit for bit, on every code and
//   scale word, with multipliers whose products land on BF16 ties and beside them.
// - ple_gate with FP32 norm weights (FP4-Local's trained norms) against the FP64 closed formula,
//   under the bound it meets with BF16 norms: |out - ref| <= 2^-8 |ref| + 2^-12 rms(ref).

#include "ops/op_family.h"
#include "infernix/ops/ple.h"
#include "ops/linear/linear_test_common.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <string>
#include <tuple>
#include <vector>

namespace {

namespace t = infernix::test;
using infernix::DType;
using infernix::Tensor;

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

struct Device {
    void* p = nullptr;
    explicit Device(std::size_t bytes) { t::cuda_check(cudaMalloc(&p, bytes), "cudaMalloc"); }
    ~Device() { cudaFree(p); }
    Device(const Device&)            = delete;
    Device& operator=(const Device&) = delete;
};

std::uint32_t next(std::uint32_t& state) { return state = state * 1664525U + 1013904223U; }

double e2m1(unsigned code) {
    static constexpr double kValues[8] = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0};
    return (code & 8U) ? -kValues[code & 7U] : kValues[code & 7U];
}

double e4m3fn(std::uint8_t word) {
    const int exponent = (word >> 3) & 15, mantissa = word & 7;
    const double magnitude = exponent == 0 ? std::ldexp(mantissa, -9) : std::ldexp(8 + mantissa, exponent - 10);
    return (word & 0x80U) ? -magnitude : magnitude;
}

// Round a binary64 value to BF16 once, to nearest even (the values here are normal or zero).
std::uint16_t bf16_rne(double value) {
    if (value == 0.0) { return std::signbit(value) ? 0x8000U : 0U; }
    const double magnitude = std::fabs(value);
    int exponent           = 0;
    (void)std::frexp(magnitude, &exponent); // magnitude in [2^(exponent-1), 2^exponent)
    const double ulp     = std::ldexp(1.0, exponent - 8);
    double quotient      = magnitude / ulp; // in [128, 256), exact
    double integral      = std::floor(quotient);
    const double half    = quotient - integral;
    if (half > 0.5 || (half == 0.5 && std::fmod(integral, 2.0) != 0.0)) { integral += 1.0; }
    const float rounded = static_cast<float>(integral * ulp); // exact: 9 significant bits at most
    std::uint32_t bits;
    std::memcpy(&bits, &rounded, 4);
    return static_cast<std::uint16_t>((bits >> 16) | (value < 0 ? 0x8000U : 0U));
}

void embed_exact() {
    constexpr int kWidth = 160, kHeads = 16, kRowBytes = kWidth / 2 + kWidth / 16, kColumns = 37;
    std::vector<std::uint8_t> rows(static_cast<std::size_t>(kRowBytes) * kHeads * kColumns);
    std::uint32_t state = 2026U;
    for (std::size_t i = 0; i < rows.size(); ++i) {
        const std::size_t within = i % kRowBytes;
        auto byte                = static_cast<std::uint8_t>(next(state) >> 24);
        if (within >= kWidth / 2) { byte &= 0x7FU; } // block scales are nonnegative
        if (within >= kWidth / 2 && byte == 0x7FU) { byte = 0x7EU; }
        rows[i] = byte;
    }
    // Every code and every scale word appears; multipliers: FP4-Local's, a power of two, and 1 + 2^-7
    // (every product then has a 1 just past BF16's last bit, so many land on exact ties) and 1 + 2^-23.
    for (const float multiplier : {3.324236240587197e-05F, 0x1p-14F, 1.0F + 0x1p-7F, 1.0F + 0x1p-23F}) {
        Device rd(rows.size()), md(4), od(static_cast<std::size_t>(kHeads) * kWidth * kColumns * 2);
        t::cuda_check(cudaMemcpy(rd.p, rows.data(), rows.size(), cudaMemcpyHostToDevice), "rows");
        t::cuda_check(cudaMemcpy(md.p, &multiplier, 4, cudaMemcpyHostToDevice), "multiplier");
        const Tensor r(rd.p, DType::U8, {kRowBytes, kHeads, kColumns});
        const Tensor m(md.p, DType::FP32, {1});
        Tensor out(od.p, DType::BF16, {kHeads * kWidth, kColumns});
        infernix::ops::ple_embed_nvfp4(r, m, out, nullptr);
        std::vector<std::uint16_t> got(static_cast<std::size_t>(kHeads) * kWidth * kColumns);
        t::cuda_check(cudaMemcpy(got.data(), od.p, got.size() * 2, cudaMemcpyDeviceToHost), "read");
        std::size_t bad = 0;
        for (int c = 0; c < kColumns; ++c) {
            for (int h = 0; h < kHeads; ++h) {
                const std::uint8_t* row = rows.data() + (static_cast<std::size_t>(c) * kHeads + h) * kRowBytes;
                for (int j = 0; j < kWidth; ++j) {
                    const unsigned code = (row[j / 2] >> (4 * (j % 2))) & 15U;
                    const double exact  = e2m1(code) * e4m3fn(row[kWidth / 2 + j / 16]) * static_cast<double>(multiplier);
                    bad += got[(static_cast<std::size_t>(c) * kHeads + h) * kWidth + j] != bf16_rne(exact);
                }
            }
        }
        check(bad == 0, "ple_embed_nvfp4 equals the exact product rounded once (multiplier " +
                            std::to_string(multiplier) + "): " + std::to_string(bad) + " differ");
    }
}

// The PLE gate's closed formula in FP64 (include/infernix/ops/ple.h).
void gate_oracle(const std::vector<std::uint16_t>& key, const std::vector<std::uint16_t>& value,
                 const std::vector<std::uint16_t>& residual, const std::vector<double>& wk, const std::vector<double>& wq,
                 const std::vector<double>& wc, int S, int H, int T, double eps, std::vector<double>& gated,
                 std::vector<double>& normalized) {
    const int W = S * H;
    gated.assign(static_cast<std::size_t>(W) * T, 0.0);
    normalized.assign(gated.size(), 0.0);
    for (int c = 0; c < T; ++c) {
        for (int s = 0; s < S; ++s) {
            const std::size_t base = static_cast<std::size_t>(c) * W + static_cast<std::size_t>(s) * H;
            double kk = 0, qq = 0;
            for (int d = 0; d < H; ++d) {
                const double k = t::bf16_to_f32(key[base + d]), q = t::bf16_to_f32(residual[base + d]);
                kk += k * k;
                qq += q * q;
            }
            const double ki = 1.0 / std::sqrt(kk / H + eps), qi = 1.0 / std::sqrt(qq / H + eps);
            double dot = 0;
            for (int d = 0; d < H; ++d) {
                const std::size_t ch = static_cast<std::size_t>(s) * H + d;
                dot += t::bf16_to_f32(key[base + d]) * ki * (1.0 + wk[ch]) * t::bf16_to_f32(residual[base + d]) * qi *
                       (1.0 + wq[ch]);
            }
            double g = dot / std::sqrt(static_cast<double>(H));
            g        = std::copysign(std::sqrt(std::max(std::fabs(g), 1e-6)), g);
            const double gate = 1.0 / (1.0 + std::exp(-g));
            double vv         = 0;
            for (int d = 0; d < H; ++d) {
                const double v = gate * t::bf16_to_f32(value[static_cast<std::size_t>(c) * H + d]);
                gated[base + d] = v;
                vv += v * v;
            }
            const double vi = 1.0 / std::sqrt(vv / H + eps);
            for (int d = 0; d < H; ++d) {
                normalized[base + d] = gated[base + d] * vi * (1.0 + wc[static_cast<std::size_t>(s) * H + d]);
            }
        }
    }
}

void gate_fp32_norms() {
    constexpr int S = 4, H = 2560, W = S * H, T = 5;
    constexpr float kEps = 1e-6F;
    std::uint32_t state = 77U;
    const auto uniform  = [&] {
        return static_cast<float>(static_cast<std::int32_t>(next(state) >> 8) - (1 << 23)) / static_cast<float>(1 << 23);
    };
    std::vector<std::uint16_t> key(static_cast<std::size_t>(W) * T), residual(key.size()), value(static_cast<std::size_t>(H) * T);
    for (auto& v : key) { v = t::f32_to_bf16(uniform()); }
    for (auto& v : residual) { v = t::f32_to_bf16(3.0F * uniform()); }
    for (auto& v : value) { v = t::f32_to_bf16(uniform()); }
    for (const bool fp32 : {false, true}) {
        std::vector<double> norms[3];
        std::vector<std::uint16_t> norm16[3];
        std::vector<float> norm32[3];
        Device nd[3] = {Device(W * 4), Device(W * 4), Device(W * 4)};
        for (int n = 0; n < 3; ++n) {
            norms[n].resize(W);
            norm16[n].resize(W);
            norm32[n].resize(W);
            for (int i = 0; i < W; ++i) {
                const float v = 0.25F * uniform() + 0x1p-20F * uniform(); // finer than BF16 for the FP32 case
                norm32[n][i]  = v;
                norm16[n][i]  = t::f32_to_bf16(v);
                norms[n][i]   = fp32 ? v : t::bf16_to_f32(norm16[n][i]);
            }
            t::cuda_check(cudaMemcpy(nd[n].p, fp32 ? static_cast<const void*>(norm32[n].data()) : norm16[n].data(),
                                     static_cast<std::size_t>(W) * (fp32 ? 4 : 2), cudaMemcpyHostToDevice),
                          "norm");
        }
        Device kd(key.size() * 2), vd(value.size() * 2), rd(residual.size() * 2), gd(key.size() * 2), od(key.size() * 2);
        t::cuda_check(cudaMemcpy(kd.p, key.data(), key.size() * 2, cudaMemcpyHostToDevice), "key");
        t::cuda_check(cudaMemcpy(vd.p, value.data(), value.size() * 2, cudaMemcpyHostToDevice), "value");
        t::cuda_check(cudaMemcpy(rd.p, residual.data(), residual.size() * 2, cudaMemcpyHostToDevice), "residual");
        const DType norm_type = fp32 ? DType::FP32 : DType::BF16;
        Tensor gated(gd.p, DType::BF16, {W, T}), normalized(od.p, DType::BF16, {W, T});
        infernix::ops::ple_gate(Tensor(kd.p, DType::BF16, {W, T}), Tensor(vd.p, DType::BF16, {H, T}),
                                Tensor(rd.p, DType::BF16, {W, T}), Tensor(nd[0].p, norm_type, {W}),
                                Tensor(nd[1].p, norm_type, {W}), Tensor(nd[2].p, norm_type, {W}), S, kEps, gated,
                                normalized, nullptr);
        std::vector<std::uint16_t> got_gated(key.size()), got_normalized(key.size());
        t::cuda_check(cudaMemcpy(got_gated.data(), gd.p, key.size() * 2, cudaMemcpyDeviceToHost), "gated");
        t::cuda_check(cudaMemcpy(got_normalized.data(), od.p, key.size() * 2, cudaMemcpyDeviceToHost), "normalized");
        std::vector<double> ref_gated, ref_normalized;
        gate_oracle(key, value, residual, norms[0], norms[1], norms[2], S, H, T, kEps, ref_gated, ref_normalized);
        for (const auto& [got, ref, name] : {std::tuple{&got_gated, &ref_gated, "gated"},
                                             std::tuple{&got_normalized, &ref_normalized, "normalized"}}) {
            double ss = 0;
            for (const double v : *ref) { ss += v * v; }
            const double rms = std::sqrt(ss / static_cast<double>(ref->size()));
            std::size_t bad  = 0;
            for (std::size_t i = 0; i < ref->size(); ++i) {
                const double e = std::fabs(t::bf16_to_f32((*got)[i]) - (*ref)[i]);
                bad += e > std::ldexp(std::fabs((*ref)[i]), -8) + std::ldexp(rms, -12);
            }
            check(bad == 0, std::string("ple_gate ") + (fp32 ? "FP32" : "BF16") + " norms: " + name + " within its bound (" +
                                std::to_string(bad) + " outside)");
        }
    }
}

} // namespace

INFERNIX_OP_TEST(ple_nvfp4) {
    try {
        if (!t::linear::cuda_available()) {
            std::printf("SKIP: no CUDA device\n");
            return 77;
        }
        embed_exact();
        gate_fp32_norms();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        ++failures;
    }
    std::printf(failures == 0 ? "PLE NVFP4 checks passed\n" : "FAIL: %d checks failed\n", failures);
    return failures == 0 ? 0 : 1;
}
