// Benchmark of the direction cut and inject (--ablate; include/infernix/ops/direction_cut.h) against the plain
// injects it replaces on a cut lane, at Flash-Next's geometry (S = 4 streams of H = 2560, top-10 routing):
// hyper_connection_inject vs hyper_connection_cut_inject (an attention lane) and moe_combine_inject vs
// moe_combine_cut_inject (an MLP lane). Each point times one CUDA graph of --layers back-to-back calls at T columns on
// the same buffers (a decode round's residual and lane outputs sit in L2, as here) and reports the time per call.
//
//   infernix_direction_cut_bench [--columns 1,2,8,32,512,4096] [--top-k 10] [--layers 48] [--warmup 5] [--repeat 50]
//                                [--seed N]

#include "infernix/ops/direction_cut.h"
#include "infernix/ops/hyper_connection.h"
#include "infernix/ops/offloaded_sparse_moe.h"

#include "core/arena.h"
#include "core/device.h"
#include "infernix_bench_common.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

using namespace infernix;

namespace {

constexpr int kStreams = 4;
constexpr int kHidden  = 2560;

struct Options {
    std::vector<int> columns{1, 2, 8, 32, 512, 4096};
    int top_k          = 10; // Flash-Next's routed experts per token
    int layers         = 48;
    int warmup         = 5;
    int repeat         = 50;
    std::uint32_t seed = 20261123U;
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
        } else if (arg == "--top-k") {
            options.top_k = std::stoi(value());
        } else if (arg == "--layers") {
            options.layers = std::stoi(value());
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
    if (options.layers < 1) { throw std::invalid_argument("--layers must be positive"); }
    if (options.top_k < 1 || options.top_k > 16) { throw std::invalid_argument("--top-k must be in [1, 16]"); }
    return options;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        DeviceContext context;
        // A unit direction (every component 1/sqrt(H)).
        std::vector<float> direction(kHidden, 1.0F / std::sqrt(static_cast<float>(kHidden)));
        DeviceBuffer d_dir(direction.size() * sizeof(float));
        bench::upload(d_dir.p, direction.data(), d_dir.bytes);
        const ops::DirectionCut cut{.direction = static_cast<const float*>(d_dir.p), .alpha = 1.2F};
        std::printf("# gpu=%s timed=direction cut (graph of %d calls, same buffers) S=%d H=%d top-%d alpha=%.1f\n",
                    context.props.name, options.layers, kStreams, kHidden, options.top_k, cut.alpha);
        for (const int columns : options.columns) {
            const std::size_t T = static_cast<std::size_t>(columns);
            DeviceBuffer residual = bench::make_bf16(kStreams * kHidden * T, options.seed + columns, -2.0F, 2.0F);
            const DeviceBuffer y       = bench::make_bf16(kHidden * T, options.seed + 1, -1.0F, 1.0F);
            const DeviceBuffer outputs = bench::make_bf16(kHidden * T * options.top_k, options.seed + 2, -1.0F, 1.0F);
            const DeviceBuffer shared  = bench::make_bf16(kHidden * T, options.seed + 3, -1.0F, 1.0F);
            const DeviceBuffer inject  = bench::make_f32(kStreams * T, options.seed + 4, -1.0F, 1.0F);
            const DeviceBuffer weights = bench::make_f32(options.top_k * T, options.seed + 5, 0.0F, 0.25F);
            const DeviceBuffer gate    = bench::make_f32(T, options.seed + 6, 0.0F, 1.0F);
            const DeviceBuffer ids     = bench::make_zeros(options.top_k * T * sizeof(std::int32_t));
            const Tensor ty(y.p, DType::BF16, {kHidden, columns});
            const Tensor to(outputs.p, DType::BF16, {kHidden, columns * options.top_k});
            const Tensor ts(shared.p, DType::BF16, {kHidden, columns});
            const Tensor ti(inject.p, DType::FP32, {kStreams, columns});
            Tensor tr(residual.p, DType::BF16, {kStreams * kHidden, columns});
            const ops::MoeRouting routing{Tensor(ids.p, DType::I32, {options.top_k, columns}),
                                          Tensor(weights.p, DType::FP32, {options.top_k, columns}),
                                          Tensor(gate.p, DType::FP32, {columns}), {}};
            const auto time = [&](const char* name, const std::function<void(cudaStream_t)>& call) {
                bench::TimedGraph graph;
                graph.capture(context.stream, [&](cudaStream_t s) {
                    for (int l = 0; l < options.layers; ++l) { call(s); }
                });
                const auto timing = bench::measure_graph(graph, context.stream, options.warmup, options.repeat);
                const double per  = 1.0 / options.layers;
                std::printf("T=%-5d %-24s per call median %7.2f us (min %7.2f, p95 %7.2f)\n", columns, name,
                            timing.median_us * per, timing.min_us * per, timing.p95_us * per);
            };
            time("inject", [&](cudaStream_t s) { ops::hyper_connection_inject(ty, ti, tr, s); });
            time("cut inject", [&](cudaStream_t s) { ops::hyper_connection_cut_inject(ty, cut, ti, tr, s); });
            time("combine inject", [&](cudaStream_t s) { ops::moe_combine_inject(to, routing, ts, ti, tr, s); });
            time("combine cut inject",
                 [&](cudaStream_t s) { ops::moe_combine_cut_inject(to, routing, ts, cut, ti, tr, s); });
        }
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "infernix_direction_cut_bench: %s\n", error.what());
        return 1;
    }
}
