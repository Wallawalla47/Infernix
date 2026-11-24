// The --lossless route's LoRA adapter (include/infernix/ops/lossless_lora.h) against an exact oracle.
//
// lossless_lora_shrink: u = fp32(A x), each output the exact sum of FP32 x BF16 products rounded once (round to
// nearest, ties to even); lossless_lora_add: d = fp32(B u) exactly rounded likewise from the device's u, then
// y = bf16(fp32(y + fp32(d fp32(s)))). The host accumulates each exact sum in a wide integer and rounds it once.
// Every output must equal the oracle bit for bit, at K 1024 / 5120 / 17408 and widths across the 512-column
// chunks, with factors spanning many binades, cancelling columns, all-zero columns and tiny values; the exact split
// refuses a nonzero factor below 2^-110.

#include "ops/op_family.h"
#include "ops/op_tester.h"

#include "core/arena.h"
#include "infernix/ops/lossless_lora.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

using namespace infernix;

std::uint32_t bits_of(float f) {
    std::uint32_t b;
    std::memcpy(&b, &f, 4);
    return b;
}

float float_of(std::uint32_t b) {
    float f;
    std::memcpy(&f, &b, 4);
    return f;
}

float bf16_value(std::uint16_t b) { return float_of(static_cast<std::uint32_t>(b) << 16); }

// A value's integer significand and exponent: v = m 2^e (m < 2^24; zero has m = 0).
void parts(float v, std::int64_t& m, int& e) {
    const std::uint32_t b = bits_of(v);
    const int field       = static_cast<int>((b >> 23) & 0xFF);
    const std::int64_t f  = b & 0x7FFFFF;
    m = field == 0 ? f : (f | 0x800000);
    e = (field == 0 ? 1 : field) - 127 - 23;
    if ((b >> 31) != 0) { m = -m; }
}

// An exact sum of products m 2^e in signed 32-bit words from 2^-kBias (every product of two binary32 values lies
// within the words), its carries deferred.
class ExactSum {
public:
    void add(std::int64_t m, int e) {
        if (m == 0) { return; }
        int at = e + kBias;
        std::int64_t v = m; // |m| < 2^48
        while (v != 0) {
            const int word = at / 32, shift = at % 32;
            const std::int64_t piece = v % (1LL << 16); // 16 bits at a time keep each word's sum in range
            words_[static_cast<std::size_t>(word)] += piece * (1LL << shift);
            v   = (v - piece) / (1LL << 16);
            at += 16;
        }
    }

    // The sum rounded once to binary32.
    float round() const {
        std::vector<std::int64_t> w(words_);
        // Carry-propagate to a signed top word.
        for (std::size_t i = 0; i + 1 < w.size(); ++i) {
            const std::int64_t carry = w[i] >> 32; // floor
            w[i] -= carry * (1LL << 32);
            w[i + 1] += carry;
        }
        bool negative = w.back() < 0;
        if (negative) {
            // Two's complement negation of the multiword value.
            std::int64_t borrow = 0;
            for (auto& x : w) {
                x = -x - borrow;
                borrow = 0;
                if (x < 0) {
                    x += 1LL << 32;
                    borrow = 1;
                }
            }
        }
        int top = -1;
        for (int i = static_cast<int>(w.size()) - 1; i >= 0 && top < 0; --i) {
            for (int b = 40; b >= 0; --b) {
                if ((w[static_cast<std::size_t>(i)] >> b) & 1) {
                    top = 32 * i + b;
                    break;
                }
            }
        }
        if (top < 0) { return 0.0F; } // an exact zero sum is +0
        const auto bit = [&](int p) -> int {
            if (p < 0) { return 0; }
            const std::size_t word = static_cast<std::size_t>(p / 32);
            return word < w.size() ? static_cast<int>((w[word] >> (p % 32)) & 1) : 0;
        };
        const int exponent = top - kBias;
        const int ulp      = std::max(exponent - 23, -149);
        const int low      = ulp + kBias;
        std::uint64_t q    = 0;
        for (int p = top; p >= low; --p) { q = (q << 1) | static_cast<std::uint64_t>(bit(p)); }
        const int half = bit(low - 1);
        bool sticky    = false;
        for (int p = 0; p < low - 1 && !sticky; ++p) { sticky = bit(p) != 0; }
        if (half != 0 && (sticky || (q & 1) != 0)) { ++q; }
        const double magnitude = std::ldexp(static_cast<double>(q), ulp); // exact: q < 2^25
        const float result     = static_cast<float>(magnitude);
        return negative ? -result : result;
    }

private:
    static constexpr int kBias = 700;
    std::vector<std::int64_t> words_ = std::vector<std::int64_t>(48, 0);
};

int compare_words(const std::string& label, const std::uint32_t* got, const std::uint32_t* want, std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) {
        if (got[i] != want[i]) {
            std::cerr << label << ": element " << i << " is " << std::hex << got[i] << ", the oracle " << want[i]
                      << std::dec << '\n';
            return 1;
        }
    }
    return 0;
}

// FP32 values over many binades (with full significands), a few tiny ones and exact zeros.
float wide_value(std::mt19937& rng, float scale) {
    std::uniform_real_distribution<float> unit(-1.0F, 1.0F);
    std::uniform_int_distribution<int> exponent(-12, 2);
    const int pick = static_cast<int>(rng() % 64);
    if (pick == 0) { return 0.0F; }
    if (pick == 1) { return std::ldexp(unit(rng), -90); }
    return std::ldexp(unit(rng), exponent(rng)) * scale;
}

int check_case(std::int32_t rows, std::int32_t k, std::int32_t t, std::uint32_t seed, std::int32_t rank = 32) {
    std::mt19937 rng(seed);
    std::vector<float> a(static_cast<std::size_t>(rows) * k), b(static_cast<std::size_t>(rows) * rank);
    for (auto& v : a) { v = wide_value(rng, 0.03F); }
    for (auto& v : b) { v = wide_value(rng, 0.05F); }
    std::vector<std::uint16_t> x(static_cast<std::size_t>(k) * t);
    for (std::size_t i = 0; i < x.size(); ++i) {
        std::uniform_real_distribution<float> unit(-4.0F, 4.0F);
        x[i] = static_cast<std::uint16_t>(bits_of(unit(rng)) >> 16);
    }
    // Column 0 cancels exactly against column 1's negation for A's first half; column 2 is all zero.
    if (t > 2) {
        for (std::int32_t i = 0; i < k; ++i) {
            x[static_cast<std::size_t>(k) + i] = static_cast<std::uint16_t>(x[i] ^ 0x8000U);
            x[static_cast<std::size_t>(2) * k + i] = 0;
        }
    }
    std::vector<std::uint16_t> a3;
    if (!ops::lossless_lora_split3(a.data(), rows, k, a3)) { throw std::runtime_error("split of A refused"); }

    test::GuardedDeviceBuffer a3_d(a3.size() * 2), x_d(x.size() * 2), u_d(static_cast<std::size_t>(rows) * t * 4),
        b_d(b.size() * 4), y_d(static_cast<std::size_t>(rows) * t * 2);
    a3_d.copy_from_host(a3.data(), a3.size() * 2);
    x_d.copy_from_host(x.data(), x.size() * 2);
    b_d.copy_from_host(b.data(), b.size() * 4);
    std::vector<std::uint16_t> y0(static_cast<std::size_t>(rows) * t);
    for (auto& v : y0) { v = static_cast<std::uint16_t>(bits_of(wide_value(rng, 1.0F)) >> 16); }
    y_d.copy_from_host(y0.data(), y0.size() * 2);

    Tensor xt(x_d.data(), DType::BF16, {k, t});
    Tensor ut(u_d.data(), DType::FP32, {rows, t});
    Tensor yt(y_d.data(), DType::BF16, {rows, t});
    WorkspaceArena workspace(std::max(ops::lossless_lora_shrink_workspace_bytes(rows, k, t),
                                      ops::lossless_lora_add_workspace_bytes(rows, rank, t)));
    ops::lossless_lora_shrink(xt, a3_d.data(), rows, ut, workspace, nullptr);
    // The term over U rows [0, rank) of every row (y_row 0), scale 1.25.
    const float scale = 1.25F;
    ops::lossless_lora_add(yt, 0, static_cast<const float*>(b_d.data()), rows, rank, ut, 0, scale, workspace, nullptr);
    test::cuda_synchronize();

    std::vector<float> u(static_cast<std::size_t>(rows) * t);
    std::vector<std::uint16_t> y(y0.size());
    u_d.copy_to_host(u.data(), u.size() * 4);
    y_d.copy_to_host(y.data(), y.size() * 2);

    int failures = 0;
    const std::string label = "rows=" + std::to_string(rows) + " K=" + std::to_string(k) + " T=" + std::to_string(t) +
                              " r=" + std::to_string(rank);
    std::vector<std::uint32_t> got, want;
    for (std::int32_t c = 0; c < t; ++c) {
        for (std::int32_t r = 0; r < rows; ++r) {
            ExactSum sum;
            for (std::int32_t i = 0; i < k; ++i) {
                std::int64_t ma, mx;
                int ea, ex;
                parts(a[static_cast<std::size_t>(r) * k + i], ma, ea);
                parts(bf16_value(x[static_cast<std::size_t>(c) * k + i]), mx, ex);
                sum.add(ma * mx, ea + ex);
            }
            got.push_back(bits_of(u[static_cast<std::size_t>(c) * rows + r]));
            want.push_back(bits_of(sum.round()));
        }
    }
    failures += compare_words("lossless_lora_shrink " + label, got.data(), want.data(), got.size());
    got.clear();
    want.clear();
    for (std::int32_t c = 0; c < t; ++c) {
        for (std::int32_t r = 0; r < rows; ++r) {
            ExactSum sum;
            for (std::int32_t j = 0; j < rank; ++j) {
                std::int64_t mb, mu;
                int eb, eu;
                parts(b[static_cast<std::size_t>(r) * rank + j], mb, eb);
                parts(u[static_cast<std::size_t>(c) * rows + j], mu, eu);
                sum.add(mb * mu, eb + eu);
            }
            const float d    = sum.round();
            const float term = d * scale; // IEEE binary32 multiply: correctly rounded
            const std::size_t at = static_cast<std::size_t>(c) * rows + r;
            const float sum32 = bf16_value(y0[at]) + term;
            // bf16(fp32 value): round to nearest even on the bit pattern (finite values).
            std::uint32_t bits = bits_of(sum32);
            bits += 0x7FFFU + ((bits >> 16) & 1U);
            got.push_back(y[at]);
            want.push_back(bits >> 16);
        }
    }
    failures += compare_words("lossless_lora_add " + label, got.data(), want.data(), got.size());
    return failures;
}

// The FP32 output's form (Inkling 2's message gates, whose forget and message projections are FP32 modules):
// y = fp32(y + fp32(d fp32(s))), d the exact sum rounded once.
int check_fp32_add(std::int32_t rows, std::int32_t t, std::uint32_t seed, std::int32_t rank = 64) {
    std::mt19937 rng(seed);
    std::vector<float> b(static_cast<std::size_t>(rows) * rank), u(static_cast<std::size_t>(rank) * t),
        y0(static_cast<std::size_t>(rows) * t);
    for (auto& v : b) { v = wide_value(rng, 0.05F); }
    for (auto& v : u) { v = wide_value(rng, 1.0F); }
    for (auto& v : y0) { v = wide_value(rng, 1.0F); }
    test::GuardedDeviceBuffer b_d(b.size() * 4), u_d(u.size() * 4), y_d(y0.size() * 4);
    b_d.copy_from_host(b.data(), b.size() * 4);
    u_d.copy_from_host(u.data(), u.size() * 4);
    y_d.copy_from_host(y0.data(), y0.size() * 4);
    Tensor ut(u_d.data(), DType::FP32, {rank, t});
    Tensor yt(y_d.data(), DType::FP32, {rows, t});
    WorkspaceArena workspace(ops::lossless_lora_add_workspace_bytes(rows, rank, t));
    const float scale = -0.75F;
    ops::lossless_lora_add(yt, 0, static_cast<const float*>(b_d.data()), rows, rank, ut, 0, scale, workspace, nullptr);
    test::cuda_synchronize();
    std::vector<float> y(y0.size());
    y_d.copy_to_host(y.data(), y.size() * 4);
    std::vector<std::uint32_t> got, want;
    for (std::int32_t c = 0; c < t; ++c) {
        for (std::int32_t r = 0; r < rows; ++r) {
            ExactSum sum;
            for (std::int32_t j = 0; j < rank; ++j) {
                std::int64_t mb, mu;
                int eb, eu;
                parts(b[static_cast<std::size_t>(r) * rank + j], mb, eb);
                parts(u[static_cast<std::size_t>(c) * rank + j], mu, eu);
                sum.add(mb * mu, eb + eu);
            }
            const std::size_t at = static_cast<std::size_t>(c) * rows + r;
            got.push_back(bits_of(y[at]));
            want.push_back(bits_of(y0[at] + sum.round() * scale)); // binary32 operations, each correctly rounded
        }
    }
    return compare_words("lossless_lora_add (FP32 y) rows=" + std::to_string(rows) + " T=" + std::to_string(t), got.data(),
                         want.data(), got.size());
}

int check_split_refusal() {
    // (1 + 2^-23) 2^-120: its last bit, 2^-143, is below the smallest BF16 subnormal (2^-133), so no three BF16
    // parts hold it exactly.
    const float tiny[2] = {std::ldexp(1.0F + std::ldexp(1.0F, -23), -120), 1.0F};
    std::vector<std::uint16_t> out;
    if (ops::lossless_lora_split3(tiny, 1, 2, out)) {
        std::cerr << "lossless_lora_split3 accepted a value below 2^-110 with low bits\n";
        return 1;
    }
    // Including a tiny value with few significant bits (1.5 2^-120: one part holds it).
    const float exact[4] = {0.1F, -3.0e-30F, 7.0F, std::ldexp(1.0F, -120) * 1.5F};
    if (!ops::lossless_lora_split3(exact, 1, 4, out)) {
        std::cerr << "lossless_lora_split3 refused exactly splittable values\n";
        return 1;
    }
    for (int c = 0; c < 4; ++c) {
        const float sum = bf16_value(out[static_cast<std::size_t>(c)]) + bf16_value(out[static_cast<std::size_t>(4 + c)]) +
                          bf16_value(out[static_cast<std::size_t>(8 + c)]);
        if (sum != exact[c]) {
            std::cerr << "lossless_lora_split3: the parts of " << exact[c] << " sum to " << sum << '\n';
            return 1;
        }
    }
    return 0;
}

} // namespace

INFERNIX_OP_TEST(lossless_lora) {
    if (test::cuda_unavailable()) { return 77; }
    int failures = 0;
    try {
        failures += check_split_refusal();
        failures += check_case(96, 1024, 3, 1U);
        failures += check_case(96, 5120, 37, 2U);
        failures += check_case(64, 17408, 5, 3U);
        failures += check_case(32, 1024, 600, 4U); // two column chunks
        failures += check_case(256, 5120, 37, 5U, 128); // the largest rank, 128
        failures += check_fp32_add(2304, 3, 6U);        // Inkling 2's gate projections
        failures += check_fp32_add(2304, 600, 7U, 128); // two column chunks, the largest rank
    } catch (const std::exception& error) {
        std::cerr << "lossless lora test: " << error.what() << '\n';
        return 1;
    }
    return failures == 0 ? 0 : 1;
}
