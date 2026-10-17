// The CPU route for GGUF (ggml) and block-FP8 expert banks (src/ops/offloaded_sparse_moe/cpu/ggml_expert.h,
// fp8_expert.h;
// docs/maintainer/qwen3_8-flash-next-design.md §16.2.2, §19.3.25). Host-only.
//
// Records are random ggml blocks (finite binary16 scales) of every expert format pair. Oracle: the
// exact expert of ops/ggml/ggml_exact.h on A16 integers (whose decode, sums and rounding
// infernix_ggml_linear_test qualifies against an independent bignum oracle): x and h encoded under
// their column exponents, gate/up rows rounded to BF16, SwiGLU, the down rows. Every ISA this host
// runs, at every width 1-8, must return its bits, including columns of a wide dynamic range and a
// non-finite column (NaN outputs).

#include "core/ggml_format.h"
#include "ops/common/canonical_math.h"
#include "ops/fp8_block/fp8_exact.h"
#include "ops/ggml/ggml_exact.h"
#include "ops/offloaded_sparse_moe/cpu/expert_team.h"
#include "ops/offloaded_sparse_moe/cpu/ggml_expert.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace moe   = infernix::ops::offloaded_moe;
namespace ggml  = infernix::ggml;
namespace canon = infernix::ops::canon;
using infernix::QType;

namespace {

int g_failures = 0;
std::string g_only_isa; // --isa: only this ISA's kernels (an emulated run)

std::uint32_t row_bytes(QType q, int k) { return k / infernix::ggml_block_elems(q) * infernix::ggml_block_bytes(q); }

void random_rows(QType q, int rows, int k, std::mt19937& rng, std::uint8_t* out) {
    const int blocks = k / static_cast<int>(infernix::ggml_block_elems(q)), size = static_cast<int>(infernix::ggml_block_bytes(q));
    for (int i = 0; i < rows * blocks * size; ++i) { out[i] = static_cast<std::uint8_t>(rng()); }
    for (int i = 0; i < rows * blocks; ++i) {
        std::uint8_t* b = out + static_cast<std::size_t>(i) * size;
        // Small finite scales (exponent field 3..12) keep the outputs in BF16's normal range.
        const auto w = static_cast<std::uint16_t>((rng() & 0x83FF) | ((3 + rng() % 10) << 10));
        b[0] = static_cast<std::uint8_t>(w & 0xFF);
        b[1] = static_cast<std::uint8_t>(w >> 8);
    }
}

ggml::I128 row_dot(QType q, const std::uint8_t* row, const std::int32_t* x, int k) {
    switch (q) {
    case QType::GGML_IQ2_S: return ggml::row_dot<QType::GGML_IQ2_S>(row, x, k);
    case QType::GGML_IQ3_XXS: return ggml::row_dot<QType::GGML_IQ3_XXS>(row, x, k);
    case QType::GGML_IQ3_S: return ggml::row_dot<QType::GGML_IQ3_S>(row, x, k);
    case QType::GGML_IQ4_XS: return ggml::row_dot<QType::GGML_IQ4_XS>(row, x, k);
    case QType::GGML_IQ4_NL: return ggml::row_dot<QType::GGML_IQ4_NL>(row, x, k);
    case QType::GGML_Q2_0: return ggml::row_dot<QType::GGML_Q2_0>(row, x, k);
    default: throw std::logic_error("format");
    }
}

std::vector<std::int32_t> encode(const std::uint16_t* v, int n, int& emax) {
    emax = canon::a16_column_exponent(v, n);
    std::vector<std::int32_t> x(static_cast<std::size_t>(n), 0);
    if (emax != canon::kA16NonFinite) {
        for (int j = 0; j < n; ++j) { x[static_cast<std::size_t>(j)] = canon::a16_value(v[j], emax); }
    }
    return x;
}

// Block-FP8 records (fp8_expert_b128_v1): codes without NaN, then BF16 tile scales 2^-14..2^-7.
constexpr std::size_t kFp8Gate  = static_cast<std::size_t>(moe::kIntermediate) * moe::kHidden;
constexpr int kFp8GateTiles     = (moe::kIntermediate / 128) * (moe::kHidden / 128);

void random_fp8_record(std::mt19937& rng, std::uint8_t* r) {
    for (std::size_t i = 0; i < 3 * kFp8Gate; ++i) {
        auto c = static_cast<std::uint8_t>(rng());
        r[i]   = (c & 0x7F) == 0x7F ? static_cast<std::uint8_t>(c ^ 1) : c;
    }
    auto* s = reinterpret_cast<std::uint16_t*>(r + 3 * kFp8Gate);
    for (int t = 0; t < 3 * kFp8GateTiles; ++t) {
        s[t] = static_cast<std::uint16_t>(((127 - 14 + static_cast<int>(rng() % 8)) << 7) | (rng() & 0x7F));
    }
}

std::uint16_t fp8_row(const std::uint8_t* codes, const std::uint16_t* tiles, const std::int32_t* x, int k, int emax) {
    const int e = infernix::fp8b128::min_exponent(tiles, k / 128);
    return infernix::fp8b128::output_bf16(infernix::fp8b128::row_dot(codes, tiles, x, k, e), e, emax);
}

std::vector<std::uint16_t> oracle(const std::uint8_t* record, const moe::GgmlExpertFormat& f, const std::uint16_t* xv) {
    const int H = moe::kHidden, I = moe::kIntermediate;
    int ex = 0;
    if (f.gate_up == QType::FP8_E4M3FN_B128) { // fp8_exact.h's scalar reference
        const auto x = encode(xv, H, ex);
        const auto* s = reinterpret_cast<const std::uint16_t*>(record + 3 * kFp8Gate);
        std::vector<std::uint16_t> h(static_cast<std::size_t>(I));
        for (int i = 0; i < I; ++i) {
            const std::uint16_t* tiles = s + (i / 128) * (H / 128);
            h[static_cast<std::size_t>(i)] = canon::swiglu_bf16(
                fp8_row(record + static_cast<std::size_t>(i) * H, tiles, x.data(), H, ex),
                fp8_row(record + kFp8Gate + static_cast<std::size_t>(i) * H, tiles + kFp8GateTiles, x.data(), H, ex));
        }
        int eh = 0;
        const auto hx = encode(h.data(), I, eh);
        std::vector<std::uint16_t> y(static_cast<std::size_t>(H));
        for (int r = 0; r < H; ++r) {
            y[static_cast<std::size_t>(r)] = fp8_row(record + 2 * kFp8Gate + static_cast<std::size_t>(r) * I,
                                                     s + 2 * kFp8GateTiles + (r / 128) * (I / 128), hx.data(), I, eh);
        }
        return y;
    }
    const auto x = encode(xv, H, ex);
    std::vector<std::uint16_t> h(static_cast<std::size_t>(I));
    for (int i = 0; i < I; ++i) {
        const auto yg = ggml::output_bf16(row_dot(f.gate_up, record + static_cast<std::size_t>(i) * f.gate_row_bytes, x.data(), H), ex, f.gate_up);
        const auto yu = ggml::output_bf16(row_dot(f.gate_up, record + static_cast<std::size_t>(I + i) * f.gate_row_bytes, x.data(), H), ex, f.gate_up);
        h[static_cast<std::size_t>(i)] = canon::swiglu_bf16(yg, yu);
    }
    int eh = 0;
    const auto hx = encode(h.data(), I, eh);
    std::vector<std::uint16_t> y(static_cast<std::size_t>(H));
    const std::uint8_t* d = record + 2ULL * I * f.gate_row_bytes;
    for (int r = 0; r < H; ++r) {
        y[static_cast<std::size_t>(r)] = ggml::output_bf16(row_dot(f.down, d + static_cast<std::size_t>(r) * f.down_row_bytes, hx.data(), I), eh, f.down);
    }
    return y;
}

std::uint16_t bf16(float f) {
    std::uint32_t u;
    std::memcpy(&u, &f, 4);
    return static_cast<std::uint16_t>((u + 0x7FFF + ((u >> 16) & 1)) >> 16);
}

void run(QType gu, QType down, std::mt19937& rng) {
    const int H = moe::kHidden, I = moe::kIntermediate;
    const bool fp8 = gu == QType::FP8_E4M3FN_B128;
    const moe::GgmlExpertFormat f{.gate_up = gu, .down = down,
                                  .gate_row_bytes = fp8 ? static_cast<std::uint32_t>(H) : row_bytes(gu, H),
                                  .down_row_bytes = fp8 ? static_cast<std::uint32_t>(I) : row_bytes(down, I)};
    if (!moe::ggml_cpu_supported(f)) {
        std::fprintf(stderr, "FAIL: pair (%d, %d) is not supported\n", static_cast<int>(gu), static_cast<int>(down));
        ++g_failures;
        return;
    }
    std::vector<std::uint8_t> record(2ULL * I * f.gate_row_bytes + static_cast<std::size_t>(H) * f.down_row_bytes +
                                     (fp8 ? 3ULL * kFp8GateTiles * 2 : 0));
    if (fp8) {
        random_fp8_record(rng, record.data());
    } else {
        random_rows(gu, 2 * I, H, rng, record.data());
        random_rows(down, H, I, rng, record.data() + 2ULL * I * f.gate_row_bytes);
    }
    // Eight columns: normal values, then a wide dynamic range with zeros, then one with an infinity.
    std::vector<std::uint16_t> x(static_cast<std::size_t>(H) * moe::kMaxColumns);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    std::uniform_real_distribution<float> exponent(-40.0F, 12.0F);
    for (int c = 0; c < moe::kMaxColumns; ++c) {
        for (int j = 0; j < H; ++j) {
            float v = normal(rng);
            if (c == 5) { v = j % 17 == 0 ? 0.0F : std::ldexp(1.0F, static_cast<int>(exponent(rng))) * (j % 2 ? -1.0F : 1.0F); }
            x[static_cast<std::size_t>(c) * H + j] = bf16(v);
        }
    }
    x[static_cast<std::size_t>(7) * H + 123] = 0x7F80; // +inf: column 7's outputs are NaN
    std::vector<std::vector<std::uint16_t>> expected;
    for (int c = 0; c < moe::kMaxColumns; ++c) { expected.push_back(oracle(record.data(), f, x.data() + static_cast<std::size_t>(c) * H)); }
    const moe::CpuIsa isas[] = {moe::CpuIsa::kScalar, moe::CpuIsa::kAvx2, moe::CpuIsa::kAvxVnni, moe::CpuIsa::kAvx512Vnni};
    for (const auto isa : isas) {
        if (!moe::cpu_isa_supported(isa) || (!g_only_isa.empty() && g_only_isa != moe::cpu_isa_name(isa))) { continue; }
        for (const int ncols : {1, 2, 3, 5, 8}) {
            // Columns are taken from the end for the narrow widths, so the special columns are covered.
            const int first = moe::kMaxColumns - ncols;
            std::vector<std::uint16_t> y(static_cast<std::size_t>(H) * ncols, 0xFFFF);
            const std::uint16_t* xp[moe::kMaxColumns];
            std::uint16_t* yp[moe::kMaxColumns];
            for (int c = 0; c < ncols; ++c) {
                xp[c] = x.data() + static_cast<std::size_t>(first + c) * H;
                yp[c] = y.data() + static_cast<std::size_t>(c) * H;
            }
            moe::ggml_expert_forward(isa, record.data(), f, ncols, xp, yp);
            std::size_t bad = 0;
            for (int c = 0; c < ncols; ++c) {
                for (int r = 0; r < H; ++r) {
                    bad += y[static_cast<std::size_t>(c) * H + r] != expected[static_cast<std::size_t>(first + c)][static_cast<std::size_t>(r)];
                }
            }
            if (bad != 0) {
                std::fprintf(stderr, "FAIL: pair (%d, %d), %s, %d columns: %zu of %d outputs differ\n", static_cast<int>(gu),
                             static_cast<int>(down), moe::cpu_isa_name(isa), ncols, bad, H * ncols);
                ++g_failures;
            }
        }
    }
    // One thread, one column, the best ISA: the single-thread rate (information only; not under --isa,
    // which serves emulated runs).
    if (!g_only_isa.empty()) { return; }
    const moe::CpuIsa best = moe::best_cpu_isa();
    const std::uint16_t* xp[1] = {x.data()};
    std::vector<std::uint16_t> y(static_cast<std::size_t>(H));
    std::uint16_t* yp[1] = {y.data()};
    const auto t0 = std::chrono::steady_clock::now();
    constexpr int kRepeats = 20;
    for (int i = 0; i < kRepeats; ++i) { moe::ggml_expert_forward(best, record.data(), f, 1, xp, yp); }
    const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / kRepeats;
    std::printf("pair (%d, %d): %.0f us per expert on one %s thread (%.2f GB/s)\n", static_cast<int>(gu),
                static_cast<int>(down), us, moe::cpu_isa_name(best), static_cast<double>(record.size()) / us / 1e3);
}

// The worker team's GGUF route (CpuExpertTeam::work_ggml) against ggml_expert_forward: three jobs of
// one, two and three columns at 1, 3 and 16 workers.
void check_team_ggml(std::mt19937& rng) {
    const int H = moe::kHidden, I = moe::kIntermediate;
    const QType gu = QType::GGML_IQ2_S, down = QType::GGML_IQ4_NL;
    const moe::GgmlExpertFormat f{.gate_up = gu, .down = down, .gate_row_bytes = row_bytes(gu, H), .down_row_bytes = row_bytes(down, I)};
    const std::size_t record = 2ULL * I * f.gate_row_bytes + static_cast<std::size_t>(H) * f.down_row_bytes;
    constexpr int kJobs = 3, kCols = 3;
    std::vector<std::uint8_t> bank(record * kJobs);
    for (int e = 0; e < kJobs; ++e) {
        random_rows(gu, 2 * I, H, rng, bank.data() + e * record);
        random_rows(down, H, I, rng, bank.data() + e * record + 2ULL * I * f.gate_row_bytes);
    }
    std::vector<std::uint16_t> x(static_cast<std::size_t>(H) * kCols);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    for (auto& v : x) { v = bf16(normal(rng)); }
    const auto at = [&](int j, int c) { return static_cast<std::size_t>(j * kCols + c) * H; };
    std::vector<std::uint16_t> want(static_cast<std::size_t>(H) * kCols * kJobs, 0);
    for (int j = 0; j < kJobs; ++j) {
        const std::uint16_t* xp[kCols];
        std::uint16_t* yp[kCols];
        for (int c = 0; c < kCols; ++c) {
            xp[c] = x.data() + static_cast<std::size_t>(c) * H;
            yp[c] = want.data() + at(j, c);
        }
        moe::ggml_expert_forward(moe::best_cpu_isa(), bank.data() + j * record, f, j + 1, xp, yp);
    }
    for (const int workers : {1, 3, 16}) {
        moe::CpuExpertTeam team({.workers = workers, .max_jobs = kJobs, .activation = moe::ExpertActivation::kA16, .ggml = true});
        std::vector<std::uint16_t> got(want.size(), 0);
        std::vector<moe::CpuExpertJob> jobs(kJobs);
        for (int j = 0; j < kJobs; ++j) {
            auto& job  = jobs[static_cast<std::size_t>(j)];
            job.record = bank.data() + j * record;
            job.ggml   = f;
            job.ncols  = j + 1;
            for (int c = 0; c < job.ncols; ++c) {
                job.x[c] = x.data() + static_cast<std::size_t>(c) * H;
                job.y[c] = got.data() + at(j, c);
            }
        }
        team.run(jobs);
        if (got != want) {
            std::fprintf(stderr, "FAIL: the worker team's GGUF route at %d workers differs from ggml_expert_forward\n", workers);
            ++g_failures;
        }
    }
}

// --rates [fp8] (information only): the worker team's one-column throughput on cold records (256
// records, past the last-level cache) of the GGUF import's most common pair, or of block-FP8 banks,
// for several worker counts.
void rates(bool fp8) {
    const int H = moe::kHidden, I = moe::kIntermediate;
    const QType gu = fp8 ? QType::FP8_E4M3FN_B128 : QType::GGML_IQ3_S;
    const QType down = fp8 ? QType::FP8_E4M3FN_B128 : QType::GGML_IQ4_NL;
    const moe::GgmlExpertFormat f{.gate_up = gu, .down = down,
                                  .gate_row_bytes = fp8 ? static_cast<std::uint32_t>(H) : row_bytes(gu, H),
                                  .down_row_bytes = fp8 ? static_cast<std::uint32_t>(I) : row_bytes(down, I)};
    const std::size_t record = 2ULL * I * f.gate_row_bytes + static_cast<std::size_t>(H) * f.down_row_bytes +
                               (fp8 ? 3ULL * kFp8GateTiles * 2 : 0);
    constexpr int kRecords = 256, kMaxJobs = 16, kRounds = 32;
    std::vector<std::uint8_t> bank(record * kRecords);
    std::mt19937 rng(7);
    for (int e = 0; e < kRecords; ++e) {
        if (fp8) {
            random_fp8_record(rng, bank.data() + e * record);
            continue;
        }
        random_rows(gu, 2 * I, H, rng, bank.data() + e * record);
        random_rows(down, H, I, rng, bank.data() + e * record + 2ULL * I * f.gate_row_bytes);
    }
    std::vector<std::uint16_t> x(static_cast<std::size_t>(H), bf16(0.5F)), y(static_cast<std::size_t>(H) * kMaxJobs);
    // Calls of 16 one-column jobs (the startup rate), and of one and two (a decode call's misses).
    for (const int kJobs : {16, 2, 1}) {
        for (const int workers : {6, 8, 12, 16, 20, 24}) {
            moe::CpuExpertTeam team({.workers = workers, .max_jobs = kMaxJobs, .activation = moe::ExpertActivation::kA16, .ggml = true});
            std::vector<moe::CpuExpertJob> jobs(static_cast<std::size_t>(kJobs));
            const auto fill = [&](int round) {
                for (int k = 0; k < kJobs; ++k) {
                    auto& job  = jobs[static_cast<std::size_t>(k)];
                    job.record = bank.data() + ((round * kJobs + k) * 37 % kRecords) * record;
                    job.ggml   = f;
                    job.ncols  = 1;
                    job.x[0]   = x.data();
                    job.y[0]   = y.data() + static_cast<std::size_t>(k) * H;
                }
            };
            fill(0);
            team.run(jobs);
            const auto t0 = std::chrono::steady_clock::now();
            for (int r = 0; r < kRounds; ++r) {
                fill(r + 1);
                team.run(jobs);
            }
            const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::printf("team of %2d workers, %2d jobs per call: %.0f experts/s (%.1f GB/s), %.1f us per call\n", workers, kJobs,
                        kRounds * kJobs / s, kRounds * kJobs * static_cast<double>(record) / s / 1e9, s * 1e6 / kRounds);
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--rates") {
        rates(argc > 2 && std::string(argv[2]) == "fp8");
        return 0;
    }
    // --isa NAME (cpu_isa_name, e.g. avx512-vnni): only that ISA's kernels against the oracle and the
    // worker team, for a run under an emulator (Intel SDE) on a CPU without it.
    if (argc > 2 && std::string(argv[1]) == "--isa") {
        g_only_isa = argv[2];
        if (!moe::cpu_isa_supported(moe::best_cpu_isa()) || moe::cpu_isa_name(moe::best_cpu_isa()) != g_only_isa) {
            std::fprintf(stderr, "error: --isa %s is not this CPU's best ISA (%s)\n", argv[2], moe::cpu_isa_name(moe::best_cpu_isa()));
            return 1;
        }
    }
    try {
        std::mt19937 rng(20261011);
        const std::pair<QType, QType> pairs[] = {
            {QType::GGML_IQ2_S, QType::GGML_IQ4_NL},   {QType::GGML_IQ2_S, QType::GGML_Q2_0},
            {QType::GGML_IQ3_XXS, QType::GGML_IQ4_NL}, {QType::GGML_IQ3_XXS, QType::GGML_Q2_0},
            {QType::GGML_IQ3_S, QType::GGML_IQ4_NL},   {QType::GGML_IQ3_S, QType::GGML_Q2_0},
            {QType::GGML_IQ4_XS, QType::GGML_IQ4_NL},  {QType::GGML_IQ4_XS, QType::GGML_Q2_0}};
        for (const auto& [gu, down] : pairs) { run(gu, down, rng); }
        run(QType::FP8_E4M3FN_B128, QType::FP8_E4M3FN_B128, rng); // block-FP8 banks (cpu/fp8_expert.h)
        check_team_ggml(rng);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
    if (g_failures != 0) {
        std::fprintf(stderr, "%d ggml CPU expert checks failed\n", g_failures);
        return 1;
    }
    std::printf("ggml CPU experts: every format pair, ISA and width bit-equal to the exact expert\n");
    return 0;
}
