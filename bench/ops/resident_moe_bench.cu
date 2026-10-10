// Benchmark of resident_moe_experts, the MTP drafter's device-resident routed experts, at the
// Qwen4Exp drafter's shapes: 512 experts, top-10, H = 2560, I = 640, row-split Q4_G64_FP16 (the
// shipped bank) or Q8_G32_FP16. Each point routes T columns to distinct random experts and times one
// call with the L2 flushed before it (a draft step finds the bank cold: the target model's layers run
// between steps). The bandwidth figure counts the routed experts' codes and scales once per entry.
// The output's FNV-1a hash identifies the result bits, so kernels can be compared for bit identity.
//
//   infernix_resident_moe_bench [--columns 1,2,4] [--codec q4|q8] [--warmup 5] [--repeat 50] [--seed N]

#include "infernix/ops/resident_moe.h"

#include "core/device.h"
#include "infernix_bench_common.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace infernix;

namespace {

constexpr int kExperts = 512;
constexpr int kTopK    = 10;
constexpr int kHidden  = 2560;
constexpr int kInter   = 640;

struct Options {
    std::vector<int> columns{1, 2, 4};
    bool q8               = false;
    int warmup            = 5;
    int repeat            = 50;
    std::uint32_t seed    = 20261010U;
};

std::vector<int> parse_list(const std::string& text) {
    std::vector<int> out;
    std::size_t begin = 0;
    while (begin < text.size()) {
        const std::size_t end = text.find(',', begin);
        out.push_back(std::stoi(text.substr(begin, end == std::string::npos ? std::string::npos : end - begin)));
        if (end == std::string::npos) { break; }
        begin = end + 1;
    }
    return out;
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto value = [&]() -> std::string {
            if (i + 1 >= argc) { throw std::invalid_argument(arg + " needs a value"); }
            return argv[++i];
        };
        if (arg == "--columns") {
            options.columns = parse_list(value());
        } else if (arg == "--codec") {
            const std::string v = value();
            if (v != "q4" && v != "q8") { throw std::invalid_argument("--codec takes q4 or q8"); }
            options.q8 = v == "q8";
        } else if (arg == "--warmup") {
            options.warmup = std::stoi(value());
        } else if (arg == "--repeat") {
            options.repeat = std::stoi(value());
        } else if (arg == "--seed") {
            options.seed = static_cast<std::uint32_t>(std::stoul(value()));
        } else {
            throw std::invalid_argument("unknown option " + arg);
        }
    }
    return options;
}

// A row-split bank [rows, k]: random codes and FP16 group scales in 2^-10 .. 2^-6.
struct Bank {
    DeviceBuffer codes;
    DeviceBuffer scales;
    Weight weight;

    Bank(int rows, int k, bool q8, std::mt19937& rng) {
        const int group           = q8 ? 32 : 64;
        const std::size_t bytes   = static_cast<std::size_t>(rows) * k / (q8 ? 1 : 2);
        const std::size_t ngroups = static_cast<std::size_t>(rows) * (k / group);
        codes                     = DeviceBuffer(bytes);
        CUDA_CHECK(bench::fixture::fill_bytes(codes.p, bytes, rng()));
        std::vector<__half> s(ngroups);
        std::uniform_real_distribution<float> lg(-10.0F, -6.0F);
        for (auto& v : s) { v = __float2half(std::exp2(lg(rng))); }
        scales = DeviceBuffer(ngroups * sizeof(__half));
        CUDA_CHECK(cudaMemcpy(scales.p, s.data(), scales.bytes, cudaMemcpyHostToDevice));
        weight.qtype           = q8 ? QType::Q8_G32_FP16 : QType::Q4_G64_FP16;
        weight.layout          = QuantLayout::RowSplit;
        weight.qdata           = codes.p;
        weight.scales          = scales.p;
        weight.n               = rows;
        weight.k               = k;
        weight.group           = group;
        weight.group_size      = static_cast<std::uint32_t>(group);
        weight.ndim            = 2;
        weight.shape[0]        = weight.padded_shape[0] = rows;
        weight.shape[1]        = weight.padded_shape[1] = k;
        weight.scale_dtype     = DType::FP16;
    }
};

std::uint64_t fnv1a(const std::vector<std::uint8_t>& bytes) {
    std::uint64_t h = 1469598103934665603ULL;
    for (const std::uint8_t b : bytes) { h = (h ^ b) * 1099511628211ULL; }
    return h;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        DeviceContext context;
        std::mt19937 rng(options.seed);
        const Bank gate_up(kExperts * 2 * kInter, kHidden, options.q8, rng);
        const Bank down(kExperts * kHidden, kInter, options.q8, rng);
        const double code_bytes = options.q8 ? 1.0 : 0.5, scale_bytes = 2.0 / (options.q8 ? 32 : 64);
        const double expert_bytes = 3.0 * kHidden * kInter * (code_bytes + scale_bytes);
        bench::L2FlushBuffer flush(256ULL << 20);
        std::printf("# gpu=%s timed=resident_moe_experts (L2 flushed before each call) codec=%s experts=%d top_k=%d "
                    "H=%d I=%d expert=%.2f MB\n",
                    context.props.name, options.q8 ? "q8" : "q4", kExperts, kTopK, kHidden, kInter, expert_bytes / 1e6);
        for (const int columns : options.columns) {
            const int entries = kTopK * columns;
            std::vector<std::int32_t> ids(static_cast<std::size_t>(entries));
            for (int t = 0; t < columns; ++t) {
                std::vector<std::int32_t> order(kExperts);
                std::iota(order.begin(), order.end(), 0);
                std::shuffle(order.begin(), order.end(), rng);
                std::copy_n(order.begin(), kTopK, ids.begin() + static_cast<std::ptrdiff_t>(t) * kTopK);
            }
            DeviceBuffer d_ids(ids.size() * sizeof(std::int32_t));
            CUDA_CHECK(cudaMemcpy(d_ids.p, ids.data(), d_ids.bytes, cudaMemcpyHostToDevice));
            const DeviceBuffer x = bench::make_bf16(static_cast<std::size_t>(kHidden) * columns, options.seed + columns, -2.0F, 2.0F);
            DeviceBuffer out(sizeof(std::uint16_t) * kHidden * entries);
            const std::size_t ws_bytes = ops::resident_moe_workspace_bytes(entries, kInter);
            DeviceBuffer ws(ws_bytes);
            const Tensor tx(x.p, DType::BF16, {kHidden, columns});
            const Tensor tids(d_ids.p, DType::I32, {kTopK, columns});
            Tensor tout(out.p, DType::BF16, {kHidden, entries});
            const auto timing = bench::measure_cold_launch(
                [&](cudaStream_t s) {
                    ops::resident_moe_experts(tx, tids, gate_up.weight, down.weight, kExperts, kInter, ws.p, ws_bytes,
                                              tout, s);
                },
                flush, context.stream, options.warmup, options.repeat);
            std::vector<std::uint8_t> bytes(out.bytes);
            CUDA_CHECK(cudaMemcpy(bytes.data(), out.p, out.bytes, cudaMemcpyDeviceToHost));
            std::printf("T=%d entries=%3d | median %7.1f us (min %7.1f, p95 %7.1f) | %6.0f GB/s | output fnv1a %016llx\n",
                        columns, entries, timing.median_us, timing.min_us, timing.p95_us,
                        expert_bytes * entries / (timing.median_us * 1e-6) / 1e9,
                        static_cast<unsigned long long>(fnv1a(bytes)));
        }
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "infernix_resident_moe_bench: %s\n", error.what());
        return 1;
    }
}
