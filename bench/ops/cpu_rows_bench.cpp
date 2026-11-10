// Benchmark of the exact CPU expert kernels that offloaded banks' CPU-served misses and assisted prompt calls run, on
// one thread with Qwen3.8-Flash-Next's gate/up matrices (random codes and scales) and 1-8 columns, every supported AVX
// ISA:
//   a16   the W4A16 row sums (rg16_row_sums_a16, design §16.2.1): FP4-Local's default route
//   w4a4  the W4A4 row sums (rg16_row_sums): Dense8, recipe A, FP4-Local's --expert-a4-activations
//   fp8   the block-FP8 gate/up phase with SwiGLU (fp8_gate_up): the FP8 import, recipe C
// Prints the median time per call over --trials timed runs of --reps calls, the weight bytes per second, and the
// FNV-1a hash of the outputs (equal across ISAs and builds: the sums are exact integers).
//
//   infernix_cpu_rows_bench [--format a16|w4a4|fp8] [--reps 200] [--trials 7] [--columns 1,2,3,4,5,6,7,8]
//
// --records K (a16 or w4a4): whole experts instead (expert_forward_a16 / expert_forward, as a CPU-served miss runs
// one), over K Flash-Next records with converted experts' scale statistics (each row's scales near a row level, a
// few far from it), --threads T at once (each its own records in turn: a working set past the caches reads DRAM, as
// the CPU team does), expanded and as CSF records (csf_record.h) in alternating trials; prints experts per second for
// each, their ratio and both outputs' hashes (equal: CSF changes no bit).
//
//   infernix_cpu_rows_bench --format a16|w4a4 --records 64 --threads 6 [--reps 64] [--trials 7] [--columns 1,8]
//                           [--isa avxvnni]
//
// --csf 0|1 (a16, w4a4, without --records): the gate/up matrix of one record with those scale statistics, expanded (0)
// or as a CSF record (1), in place of random units. --decode-only: the CSF scale decode alone (csf::unit_scales_sse,
// the CPU walks' per-unit decode) over that record's gate/up row groups, in ns a unit; run it pinned to a P-core and an
// E-core to compare them.

#include "ops/offloaded_sparse_moe/cpu/fp8_expert.h"
#include "ops/offloaded_sparse_moe/cpu/w4a16_expert.h"
#include "ops/offloaded_sparse_moe/cpu/w4a4_expert.h"
#include "ops/offloaded_sparse_moe/csf_record.h"
#if defined(__x86_64__) || defined(_M_X64)
#    include "ops/offloaded_sparse_moe/cpu/csf_scales.h"
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace moe   = infernix::ops::offloaded_moe;
namespace canon = infernix::ops::canon;

namespace {

std::uint64_t fnv1a(const void* data, std::size_t bytes) {
    std::uint64_t h = 1469598103934665603ULL;
    const auto* p   = static_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < bytes; ++i) { h = (h ^ p[i]) * 1099511628211ULL; }
    return h;
}

std::uint16_t bf16_bits(float v) {
    std::uint32_t u;
    std::memcpy(&u, &v, 4);
    return static_cast<std::uint16_t>((u + 0x7FFFU + ((u >> 16) & 1U)) >> 16);
}

std::vector<int> parse_list(const std::string& text) {
    std::vector<int> out;
    std::size_t at = 0;
    while (at < text.size()) {
        const std::size_t comma = text.find(',', at);
        out.push_back(std::stoi(text.substr(at, comma - at)));
        if (comma == std::string::npos) { break; }
        at = comma + 1;
    }
    return out;
}

const char* isa_name(moe::CpuIsa isa) {
    switch (isa) {
    case moe::CpuIsa::kAvx512Vnni: return "avx512vnni";
    case moe::CpuIsa::kAvxVnni: return "avxvnni";
    case moe::CpuIsa::kAvx2: return "avx2";
    default: return "scalar";
    }
}

// One format's call at n columns into its output buffer (whose bytes are hashed), and the weight bytes it reads.
struct Kernel {
    std::size_t weight_bytes = 0;
    std::function<void(moe::CpuIsa, int)> call;
    std::function<std::uint64_t()> hash;
};

std::vector<std::uint16_t> random_bf16(std::mt19937_64& rng, int n) {
    std::uniform_real_distribution<float> value(-2.0f, 2.0f);
    std::vector<std::uint16_t> x(static_cast<std::size_t>(n));
    for (auto& v : x) { v = bf16_bits(value(rng)); }
    return x;
}

// Row-group-major 144-byte units with finite E4M3 scale bytes (exponent field < 15), as converted experts hold.
std::vector<std::uint8_t> random_units(std::mt19937_64& rng, int rgs, int blocks) {
    std::vector<std::uint8_t> matrix(static_cast<std::size_t>(rgs) * blocks * moe::kUnitBytes);
    for (auto& b : matrix) { b = static_cast<std::uint8_t>(rng()); }
    for (std::size_t u = 0; u < matrix.size() / moe::kUnitBytes; ++u) {
        for (int r = 0; r < 16; ++r) { matrix[u * moe::kUnitBytes + 128 + r] &= 0x77U; }
    }
    return matrix;
}

// A Flash-Next record (random codes) whose scale bytes follow converted experts': each row a level, each block's
// scale within a few codes of it, ~2 % far from it (E4M3 bytes with exponent field < 15).
std::vector<std::uint8_t> realistic_record(std::mt19937_64& rng, const moe::ExpertShape& shape) {
    std::vector<std::uint8_t> record(shape.record_bytes());
    for (auto& b : record) { b = static_cast<std::uint8_t>(rng()); }
    std::normal_distribution<double> near(0.0, 2.5);
    std::uniform_int_distribution<int> level(0x28, 0x48), far(8, 30), coin(0, 99);
    const auto fill = [&](std::uint8_t* units, int rgs, int blocks) {
        for (int rg = 0; rg < rgs; ++rg) {
            int row_level[16];
            for (int& l : row_level) { l = level(rng); }
            for (int b = 0; b < blocks; ++b) {
                std::uint8_t* s = units + (static_cast<std::size_t>(rg) * blocks + b) * moe::kUnitBytes + 128;
                for (int r = 0; r < 16; ++r) {
                    int v = row_level[r] + static_cast<int>(std::lround(near(rng)));
                    if (coin(rng) < 2) { v += coin(rng) < 50 ? -far(rng) : far(rng); }
                    s[r] = static_cast<std::uint8_t>(std::clamp(v, 1, 0x76));
                }
            }
        }
    };
    fill(record.data(), shape.gate_up_row_groups(), shape.gate_up_blocks());
    fill(record.data() + shape.gate_up_bytes(), shape.down_row_groups(), shape.down_blocks());
    return record;
}

int run_records(const std::string& format, int records, int threads, int reps, int trials, const std::vector<int>& columns,
                const std::string& isa_name_wanted) {
    namespace csf                = moe::csf;
    const moe::ExpertShape shape = moe::kFlashNextShape;
    const csf::Geometry g        = csf::Geometry::of(shape);
    std::mt19937_64 rng(11);
    const std::size_t bytes = shape.record_bytes();
    // Expanded records, and the same records compressed (each its table).
    std::vector<std::vector<std::uint8_t>> expanded, packed;
    std::vector<std::vector<std::uint32_t>> tables;
    std::uint64_t packed_bytes = 0;
    for (int k = 0; k < records; ++k) {
        expanded.push_back(realistic_record(rng, shape));
        packed.push_back(expanded.back());
        tables.emplace_back(static_cast<std::size_t>(g.table_words()));
        packed_bytes += csf::compress_record(shape, packed.back().data(), packed.back().data(), tables.back().data());
    }
    const double share = static_cast<double>(packed_bytes) / (static_cast<double>(bytes) * records);
    std::vector<std::vector<std::uint16_t>> x(moe::kMaxColumns), y(static_cast<std::size_t>(threads) * moe::kMaxColumns);
    for (auto& column : x) { column = random_bf16(rng, shape.hidden); }
    for (auto& column : y) { column.assign(static_cast<std::size_t>(shape.hidden), 0); }
    const moe::ExpertScales scales{1.0f, 1.0f, 1.0f, 1e-3f, 1e-3f, 1e-3f};
    std::printf("# %s experts: %d records of %.2f MB (CSF %.4f of their bytes), %d threads\n", format.c_str(), records,
                static_cast<double>(bytes) / 1e6, share, threads);
    for (const moe::CpuIsa isa : {moe::CpuIsa::kAvx512Vnni, moe::CpuIsa::kAvxVnni, moe::CpuIsa::kAvx2}) {
        if (!moe::cpu_isa_supported(isa) || (!isa_name_wanted.empty() && isa_name_wanted != isa_name(isa))) { continue; }
        for (const int n : columns) {
            // One timed run: every thread `reps` experts, thread t taking records t, t + T, ...; experts per second.
            const auto run = [&](bool compressed) {
                std::atomic<int> ready{0};
                std::atomic<bool> go{false};
                std::vector<std::thread> pool;
                for (int t = 0; t < threads; ++t) {
                    pool.emplace_back([&, t] {
                        const std::uint16_t* xs[moe::kMaxColumns];
                        std::uint16_t* ys[moe::kMaxColumns];
                        for (int c = 0; c < n; ++c) {
                            xs[c] = x[static_cast<std::size_t>(c)].data();
                            ys[c] = y[static_cast<std::size_t>(t) * moe::kMaxColumns + c].data();
                        }
                        ready.fetch_add(1);
                        while (!go.load(std::memory_order_acquire)) {}
                        for (int r = 0; r < reps; ++r) {
                            const auto k = static_cast<std::size_t>((t + r * threads) % records);
                            const moe::Rg16Record record = compressed ? moe::Rg16Record(packed[k].data(), tables[k].data())
                                                                      : moe::Rg16Record(expanded[k].data());
                            if (format == "a16") {
                                moe::expert_forward_a16(isa, shape, record, scales, n, xs, ys);
                            } else {
                                moe::expert_forward(isa, record, scales, n, xs, ys);
                            }
                        }
                    });
                }
                while (ready.load() != threads) {}
                const auto begin = std::chrono::steady_clock::now();
                go.store(true, std::memory_order_release);
                for (auto& thread : pool) { thread.join(); }
                const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
                return static_cast<double>(reps) * threads / seconds;
            };
            const auto hash = [&] {
                std::uint64_t h = 0;
                for (int t = 0; t < threads; ++t) {
                    for (int c = 0; c < n; ++c) {
                        const auto& column = y[static_cast<std::size_t>(t) * moe::kMaxColumns + c];
                        h ^= fnv1a(column.data(), column.size() * 2) + 0x9E3779B97F4A7C15ULL * (t * 8 + c + 1);
                    }
                }
                return h;
            };
            (void)run(false);
            (void)run(true);
            std::vector<double> plain, csf_rate;
            std::uint64_t plain_hash = 0, csf_hash = 0;
            for (int i = 0; i < trials; ++i) { // alternating, so drift falls on both
                plain.push_back(run(false));
                plain_hash = hash();
                csf_rate.push_back(run(true));
                csf_hash = hash();
            }
            std::sort(plain.begin(), plain.end());
            std::sort(csf_rate.begin(), csf_rate.end());
            const double a = plain[plain.size() / 2], b = csf_rate[csf_rate.size() / 2];
            std::printf("%s N=%d | expanded %9.1f experts/s (%6.2f GB/s) | CSF %9.1f experts/s | CSF/expanded %.4f "
                        "(min %.4f max %.4f) | fnv1a %016llx %016llx%s\n",
                        isa_name(isa), n, a, a * static_cast<double>(bytes) / 1e9, b, b / a, csf_rate.front() / plain.back(),
                        csf_rate.back() / plain.front(), static_cast<unsigned long long>(plain_hash),
                        static_cast<unsigned long long>(csf_hash), plain_hash == csf_hash ? "" : " MISMATCH");
            if (plain_hash != csf_hash) { return 1; }
        }
    }
    return 0;
}

} // namespace

// --decode-only: ns a unit of the CPU walks' scale decode over one CSF record's gate/up row groups.
int run_decode_only(int reps, int trials) {
#if defined(__x86_64__) || defined(_M_X64)
    namespace csf                = moe::csf;
    const moe::ExpertShape shape = moe::kFlashNextShape;
    std::mt19937_64 rng(7);
    auto record = realistic_record(rng, shape);
    std::vector<std::uint32_t> table(static_cast<std::size_t>(csf::Geometry::of(shape).table_words()));
    (void)csf::compress_record(shape, record.data(), record.data(), table.data());
    const int rgs = shape.gate_up_row_groups(), blocks = shape.gate_up_blocks();
    std::vector<std::uint8_t> out(static_cast<std::size_t>(blocks) * 16);
    std::uint64_t sink = 0;
    std::vector<double> ns;
    for (int t = 0; t < trials; ++t) {
        const auto begin = std::chrono::steady_clock::now();
        for (int r = 0; r < reps; ++r) {
            for (int rg = 0; rg < rgs; ++rg) {
                const csf::Group g = csf::group(record.data(), table.data(), rg, blocks);
                if (g.bases == nullptr) { continue; }
                const __m128i bases        = _mm_loadu_si128(reinterpret_cast<const __m128i*>(g.bases));
                const std::uint8_t* escape = g.escapes;
                for (int b = 0; b < blocks; ++b) {
                    const csf::UnitScales u = csf::unit_scales_sse(g.units + static_cast<std::size_t>(b) * csf::kBodyBytes, bases, escape);
                    _mm_storeu_si128(reinterpret_cast<__m128i*>(out.data() + 16 * b), u.scales);
                    escape += u.escapes;
                }
                sink += out[static_cast<std::size_t>(rg % (16 * blocks))];
            }
        }
        ns.push_back(std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - begin).count() /
                     (static_cast<double>(reps) * rgs * blocks));
    }
    std::sort(ns.begin(), ns.end());
    std::printf("CSF scale decode: %.3f ns a unit (median of %d trials, min %.3f; sink %llu)\n", ns[ns.size() / 2], trials,
                ns.front(), static_cast<unsigned long long>(sink));
    return 0;
#else
    (void)reps;
    (void)trials;
    std::fprintf(stderr, "--decode-only: x86 only\n");
    return 2;
#endif
}

int main(int argc, char** argv) {
    int reps = 200, trials = 7, records = 0, threads = 1, hot_csf = -1;
    std::string format       = "a16", isa;
    std::vector<int> columns = {1, 2, 3, 4, 5, 6, 7, 8};
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string key = argv[i], value = argv[i + 1];
        if (key == "--reps") { reps = std::stoi(value); }
        else if (key == "--trials") { trials = std::stoi(value); }
        else if (key == "--columns") { columns = parse_list(value); }
        else if (key == "--format") { format = value; }
        else if (key == "--records") { records = std::stoi(value); }
        else if (key == "--threads") { threads = std::stoi(value); }
        else if (key == "--isa") { isa = value; }
        else if (key == "--csf") { hot_csf = std::stoi(value); }
    }
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--decode-only") { return run_decode_only(reps, trials); }
    }
    if (records > 0) {
        if (format != "a16" && format != "w4a4") {
            std::fprintf(stderr, "--records: --format a16 or w4a4\n");
            return 2;
        }
        return run_records(format, records, std::max(threads, 1), reps, trials, columns, isa);
    }
    const moe::ExpertShape shape = moe::kFlashNextShape;
    const int rgs = shape.gate_up_row_groups(), blocks = shape.gate_up_blocks();
    std::mt19937_64 rng(7);
    constexpr int kMaxColumns = 8;

    // Every format's state lives here; the kernel closures refer to it.
    std::vector<std::uint8_t> matrix;
    std::vector<std::vector<canon::A16Block>> a16(kMaxColumns);
    std::vector<std::vector<canon::A4Block>> a4(kMaxColumns);
    std::vector<std::vector<moe::CpuLimbSlice>> limbs(kMaxColumns);
    std::vector<int> emax(kMaxColumns);
    std::vector<std::int64_t> sums;
    std::vector<std::vector<std::uint16_t>> h(kMaxColumns, std::vector<std::uint16_t>(static_cast<std::size_t>(moe::kIntermediate)));
    std::vector<const canon::A16Block*> a16_ptrs;
    std::vector<const canon::A4Block*> a4_ptrs;
    std::vector<const moe::CpuLimbSlice*> limb_ptrs;
    std::vector<std::uint16_t*> h_ptrs;
    for (auto& column : h) { h_ptrs.push_back(column.data()); }

    Kernel kernel;
    // --csf 0 / 1 (a16, w4a4): the gate/up matrix of a record with converted experts' scale statistics, expanded (0) or
    // as a CSF record (1); otherwise random units.
    std::vector<std::uint32_t> table;
    if ((format == "a16" || format == "w4a4") && hot_csf >= 0) {
        matrix = realistic_record(rng, shape);
        if (hot_csf == 1) {
            table.resize(static_cast<std::size_t>(moe::csf::Geometry::of(shape).table_words()));
            (void)moe::csf::compress_record(shape, matrix.data(), matrix.data(), table.data());
        }
    }
    if (format == "a16" || format == "w4a4") {
        if (hot_csf < 0) { matrix = random_units(rng, rgs, blocks); }
        for (int c = 0; c < kMaxColumns; ++c) {
            const auto x = random_bf16(rng, shape.hidden);
            if (format == "a16") {
                a16[c].resize(static_cast<std::size_t>(blocks));
                (void)moe::encode_a16(x.data(), shape.hidden, a16[c].data());
                a16_ptrs.push_back(a16[c].data());
            } else {
                a4[c].resize(static_cast<std::size_t>(blocks));
                moe::quantize_a4(moe::CpuIsa::kScalar, x.data(), shape.hidden, 1.0f, a4[c].data());
                a4_ptrs.push_back(a4[c].data());
            }
        }
        kernel.weight_bytes = matrix.size();
        kernel.call = [&](moe::CpuIsa isa, int n) {
            sums.resize(static_cast<std::size_t>(rgs) * 16 * n);
            if (format == "a16") {
                moe::rg16_row_sums_a16(isa, {matrix.data(), blocks, table.empty() ? nullptr : table.data(), 0}, 0, rgs,
                                       a16_ptrs.data(), n, sums.data());
            } else {
                moe::rg16_row_sums(isa, {matrix.data(), blocks, table.empty() ? nullptr : table.data(), 0}, 0, rgs,
                                   a4_ptrs.data(), n, sums.data());
            }
        };
        kernel.hash = [&] { return fnv1a(sums.data(), sums.size() * sizeof(std::int64_t)); };
    } else if (format == "fp8") {
        // The record: gate, up and down codes (E4M3FN, NaN codes avoided), then the BF16 128 x 128 tile scales.
        const std::size_t matrix_bytes = static_cast<std::size_t>(moe::kIntermediate) * moe::kHidden;
        const std::size_t tiles        = 3 * static_cast<std::size_t>(moe::kIntermediate / 128) * (moe::kHidden / 128);
        matrix.resize(3 * matrix_bytes + 2 * tiles);
        for (std::size_t i = 0; i < 3 * matrix_bytes; ++i) {
            auto b = static_cast<std::uint8_t>(rng());
            if ((b & 0x7FU) == 0x7FU) { b ^= 1U; }
            matrix[i] = b;
        }
        std::uniform_real_distribution<float> scale(0.25f, 4.0f);
        for (std::size_t t = 0; t < tiles; ++t) {
            const std::uint16_t s = bf16_bits(scale(rng));
            std::memcpy(matrix.data() + 3 * matrix_bytes + 2 * t, &s, 2);
        }
        for (int c = 0; c < kMaxColumns; ++c) {
            const auto x = random_bf16(rng, moe::kHidden);
            limbs[c].resize(static_cast<std::size_t>(moe::kHidden / 32));
            emax[c] = moe::ggml_encode_column(x.data(), moe::kHidden, limbs[c].data());
            limb_ptrs.push_back(limbs[c].data());
        }
        kernel.weight_bytes = 2 * matrix_bytes;
        kernel.call = [&](moe::CpuIsa isa, int n) {
            moe::fp8_gate_up(isa, matrix.data(), limb_ptrs.data(), emax.data(), n, 0, moe::kIntermediate, h_ptrs.data());
        };
        kernel.hash = [&] {
            std::uint64_t value = 0;
            for (const auto& column : h) { value ^= fnv1a(column.data(), column.size() * 2) + 0x9E3779B97F4A7C15ULL; }
            return value;
        };
    } else {
        std::fprintf(stderr, "--format: a16, w4a4 or fp8\n");
        return 2;
    }

    std::printf("# %s CPU kernel: %.2f MB of weights a call, one thread\n", format.c_str(),
                static_cast<double>(kernel.weight_bytes) / 1e6);
    for (const moe::CpuIsa isa : {moe::CpuIsa::kAvx512Vnni, moe::CpuIsa::kAvxVnni, moe::CpuIsa::kAvx2}) {
        if (!moe::cpu_isa_supported(isa)) { continue; }
        for (const int n : columns) {
            if (n < 1 || n > kMaxColumns) { throw std::invalid_argument("--columns: 1 to 8"); }
            if (format == "fp8") {
                for (auto& column : h) { std::fill(column.begin(), column.end(), std::uint16_t{0}); }
            }
            for (int w = 0; w < 3; ++w) { kernel.call(isa, n); }
            std::vector<double> us;
            for (int t = 0; t < trials; ++t) {
                const auto begin = std::chrono::steady_clock::now();
                for (int r = 0; r < reps; ++r) { kernel.call(isa, n); }
                us.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - begin).count() /
                             reps);
            }
            std::sort(us.begin(), us.end());
            const double median = us[us.size() / 2];
            std::printf("%s N=%d | per call median %9.2f us (min %9.2f) | %7.2f GB/s | outputs fnv1a %016llx\n",
                        isa_name(isa), n, median, us.front(), static_cast<double>(kernel.weight_bytes) / median / 1e3,
                        static_cast<unsigned long long>(kernel.hash()));
        }
    }
    return 0;
}
