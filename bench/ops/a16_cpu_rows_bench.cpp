// Benchmark of the exact W4A16 CPU expert row sums (rg16_row_sums_a16, design §16.2.1) that FP4-Local's CPU-served
// misses and assisted prompt calls run: one thread, Qwen3.8-Flash-Next's gate/up matrix (2 x intermediate rows of
// hidden / 16 blocks, random codes and scales), 1-8 encoded BF16 columns, every supported AVX ISA. Prints the median
// time per call over --trials timed runs of --reps calls, the weight bytes per second, and the FNV-1a hash of the
// sums (equal across ISAs and builds: the sums are exact integers).
//
//   infernix_a16_cpu_rows_bench [--reps 200] [--trials 7] [--columns 1,2,3,4,5,6,7,8]

#include "ops/offloaded_sparse_moe/cpu/w4a16_expert.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace moe   = infernix::ops::offloaded_moe;
namespace canon = infernix::ops::canon;

namespace {

std::uint64_t fnv1a(const void* data, std::size_t bytes) {
    std::uint64_t h        = 1469598103934665603ULL;
    const auto* p          = static_cast<const unsigned char*>(data);
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

} // namespace

int main(int argc, char** argv) {
    int reps = 200, trials = 7;
    std::vector<int> columns = {1, 2, 3, 4, 5, 6, 7, 8};
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string key = argv[i], value = argv[i + 1];
        if (key == "--reps") { reps = std::stoi(value); }
        else if (key == "--trials") { trials = std::stoi(value); }
        else if (key == "--columns") { columns = parse_list(value); }
    }
    const moe::ExpertShape shape = moe::kFlashNextShape;
    const int rgs = shape.gate_up_row_groups(), blocks = shape.gate_up_blocks();
    std::mt19937_64 rng(7);
    std::vector<std::uint8_t> matrix(static_cast<std::size_t>(rgs) * blocks * moe::kUnitBytes);
    for (auto& b : matrix) { b = static_cast<std::uint8_t>(rng()); }
    // Scale bytes: finite E4M3 (exponent field < 15), as converted experts hold.
    for (std::size_t u = 0; u < matrix.size() / moe::kUnitBytes; ++u) {
        for (int r = 0; r < 16; ++r) { matrix[u * moe::kUnitBytes + 128 + r] &= 0x77U; }
    }
    std::uniform_real_distribution<float> value(-2.0f, 2.0f);
    std::vector<std::vector<canon::A16Block>> acts(8, std::vector<canon::A16Block>(static_cast<std::size_t>(blocks)));
    for (auto& column : acts) {
        std::vector<std::uint16_t> x(static_cast<std::size_t>(shape.hidden));
        for (auto& v : x) { v = bf16_bits(value(rng)); }
        (void)moe::encode_a16(x.data(), shape.hidden, column.data());
    }
    std::vector<const canon::A16Block*> pointers;
    for (const auto& column : acts) { pointers.push_back(column.data()); }
    std::printf("# W4A16 CPU row sums: %d row groups x %d blocks (%.2f MB of weights a call), one thread\n", rgs, blocks,
                static_cast<double>(matrix.size()) / 1e6);
    for (const moe::CpuIsa isa : {moe::CpuIsa::kAvx512Vnni, moe::CpuIsa::kAvxVnni, moe::CpuIsa::kAvx2}) {
        if (!moe::cpu_isa_supported(isa)) { continue; }
        for (const int n : columns) {
            std::vector<std::int64_t> out(static_cast<std::size_t>(rgs) * 16 * n);
            const auto call = [&] {
                moe::rg16_row_sums_a16(isa, matrix.data(), blocks, 0, rgs, pointers.data(), n, out.data());
            };
            for (int w = 0; w < 3; ++w) { call(); }
            std::vector<double> us;
            for (int t = 0; t < trials; ++t) {
                const auto begin = std::chrono::steady_clock::now();
                for (int r = 0; r < reps; ++r) { call(); }
                us.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - begin).count() /
                             reps);
            }
            std::sort(us.begin(), us.end());
            const double median = us[us.size() / 2];
            std::printf("%s N=%d | per call median %9.2f us (min %9.2f) | %7.2f GB/s | sums fnv1a %016llx\n",
                        isa_name(isa), n, median, us.front(), static_cast<double>(matrix.size()) / median / 1e3,
                        static_cast<unsigned long long>(fnv1a(out.data(), out.size() * sizeof(std::int64_t))));
        }
    }
    return 0;
}
