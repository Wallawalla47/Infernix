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

#include "ops/offloaded_sparse_moe/cpu/fp8_expert.h"
#include "ops/offloaded_sparse_moe/cpu/w4a16_expert.h"
#include "ops/offloaded_sparse_moe/cpu/w4a4_expert.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <stdexcept>
#include <string>
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

} // namespace

int main(int argc, char** argv) {
    int reps = 200, trials = 7;
    std::string format       = "a16";
    std::vector<int> columns = {1, 2, 3, 4, 5, 6, 7, 8};
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string key = argv[i], value = argv[i + 1];
        if (key == "--reps") { reps = std::stoi(value); }
        else if (key == "--trials") { trials = std::stoi(value); }
        else if (key == "--columns") { columns = parse_list(value); }
        else if (key == "--format") { format = value; }
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
    if (format == "a16" || format == "w4a4") {
        matrix = random_units(rng, rgs, blocks);
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
                moe::rg16_row_sums_a16(isa, matrix.data(), blocks, 0, rgs, a16_ptrs.data(), n, sums.data());
            } else {
                moe::rg16_row_sums(isa, matrix.data(), blocks, 0, rgs, a4_ptrs.data(), n, sums.data());
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
