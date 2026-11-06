// Benchmark of hyper_connection_mix, the block mixer of Qwen4Exp's hyper-connection residual, at
// Flash-Next's geometry: S = 4 streams of H = 2560, rank 320, the down projection with its 4
// injection rows. Weights are row-split Q8_G32_FP16 (Dense8) with K padded to 128, BF16
// contiguous (the bit-exact artifact), or those BF16 values packed at load (bf16_p12: the same output bits as
// --codec bf16 with the same seed). Each point times one CUDA graph of --layers back-to-back calls
// at T columns, each with its own weights (48 layers' weights far exceed the L2, so every call reads
// its weights from DRAM, as decode does; the graph, as decode's, keeps host launch time out), and
// reports the time per call. The bandwidth figure counts one call's weights once. The FNV-1a hash of
// the last call's outputs identifies the result bits. --a8 times hyper_connection_mix_a8_encoded instead (the mix
// also writing x's 8-bit columns for a GGUF projection under --a8-activations; 2-16 columns).
//
//   infernix_hyper_connection_bench [--columns 1,2,5] [--codec q8|bf16|p12] [--layers 48] [--warmup 5] [--repeat 50]
//                                   [--seed N] [--a8]

#include "infernix/ops/hyper_connection.h"

#include "core/arena.h"
#include "core/device.h"
#include "core/weight_view.h"
#include "infernix/ops/bf16_p12.h"
#include "infernix_bench_common.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace infernix;

namespace {

constexpr int kStreams = 4;
constexpr int kHidden  = 2560;
constexpr int kRank    = 320;
constexpr int kWidth   = kStreams * kHidden;

struct Options {
    std::vector<int> columns{1, 2, 5};
    bool bf16          = false;
    bool p12           = false;
    bool a8            = false;
    int layers         = 48;
    int warmup         = 5;
    int repeat         = 50;
    std::uint32_t seed = 20261010U;
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
            if (v != "q8" && v != "bf16" && v != "p12") { throw std::invalid_argument("--codec takes q8, bf16 or p12"); }
            options.bf16 = v == "bf16" || v == "p12";
            options.p12  = v == "p12";
        } else if (arg == "--layers") {
            options.layers = std::stoi(value());
        } else if (arg == "--warmup") {
            options.warmup = std::stoi(value());
        } else if (arg == "--repeat") {
            options.repeat = std::stoi(value());
        } else if (arg == "--seed") {
            options.seed = static_cast<std::uint32_t>(std::stoul(value()));
        } else if (arg == "--a8") {
            options.a8 = true;
        } else {
            throw std::invalid_argument("unknown option " + arg);
        }
    }
    return options;
}

// A [rows, k] weight: Q8 row-split (random codes, FP16 group scales 2^-12 .. 2^-8, K padded to 128)
// or BF16 contiguous (values of the same magnitude), packed (bf16_p12) with p12.
struct BenchWeight {
    DeviceBuffer data;
    DeviceBuffer scales;
    WeightParent parent;
    Weight weight;
    double bytes = 0;

    BenchWeight(int rows, int k, bool bf16, bool p12, std::mt19937& rng) {
        const int padded = (k + 127) / 128 * 128;
        std::uniform_real_distribution<float> lg(-12.0F, -8.0F);
        weight.n = rows;
        weight.k = k;
        weight.ndim = 2;
        weight.shape[0] = weight.padded_shape[0] = rows;
        weight.shape[1] = k;
        if (bf16) {
            weight.padded_shape[1] = k;
            data          = bench::make_bf16(static_cast<std::size_t>(rows) * k, rng(), -0.03F, 0.03F);
            weight.qtype  = QType::BF16;
            weight.layout = QuantLayout::Contiguous;
            weight.qdata  = data.p;
            bytes         = static_cast<double>(data.bytes);
            if (p12) { pack(rows, k); }
            return;
        }
        weight.padded_shape[1] = padded;
        data = DeviceBuffer(static_cast<std::size_t>(rows) * padded);
        CUDA_CHECK(bench::fixture::fill_bytes(data.p, data.bytes, rng()));
        std::vector<__half> s(static_cast<std::size_t>(rows) * (padded / 32));
        for (auto& v : s) { v = __float2half(std::exp2(lg(rng))); }
        scales = DeviceBuffer(s.size() * sizeof(__half));
        bench::upload(scales.p, s.data(), scales.bytes);
        weight.qtype       = QType::Q8_G32_FP16;
        weight.layout      = QuantLayout::RowSplit;
        weight.qdata       = data.p;
        weight.scales      = scales.p;
        weight.group       = 32;
        weight.group_size  = 32;
        weight.scale_dtype = DType::FP16;
        bytes              = static_cast<double>(data.bytes + scales.bytes);
    }

    // The loader's count and pack passes; the BF16 copy is freed, as at load.
    void pack(int rows, int k) {
        DeviceBuffer scratch(ops::bf16_p12_pack_scratch_bytes(rows)), count(4);
        ops::bf16_p12_count(data.p, rows, k, scratch.p, static_cast<std::uint32_t*>(count.p), nullptr);
        std::uint32_t escapes = 0;
        CUDA_CHECK(cudaMemcpy(&escapes, count.p, 4, cudaMemcpyDeviceToHost));
        const WeightGeometry g = bf16_p12_geometry(rows, k, escapes);
        DeviceBuffer packed(g.bytes);
        ops::bf16_p12_pack(data.p, rows, k, escapes, scratch.p, packed.p, nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());
        data            = std::move(packed);
        parent.geometry = g;
        parent.data = parent.device = static_cast<const std::byte*>(data.p);
        const auto n = static_cast<std::uint64_t>(rows), kk = static_cast<std::uint64_t>(k);
        weight       = native_weight(WeightView{{n, kk}, {{&parent, 0, n * kk}}});
        bytes        = static_cast<double>(g.bytes);
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
        if (options.layers < 1) { throw std::invalid_argument("--layers must be positive"); }
        std::vector<std::unique_ptr<BenchWeight>> downs, ups;
        for (int l = 0; l < options.layers; ++l) {
            downs.push_back(std::make_unique<BenchWeight>(kRank + kStreams, kWidth, options.bf16, options.p12, rng));
            ups.push_back(std::make_unique<BenchWeight>(kWidth, kRank, options.bf16, options.p12, rng));
        }
        const BenchWeight& down = *downs.front();
        const BenchWeight& up   = *ups.front();
        const DeviceBuffer norm = bench::make_bf16(kWidth, options.seed + 1, -0.2F, 0.2F);
        int max_columns = 1;
        for (const int c : options.columns) { max_columns = std::max(max_columns, c); }
        WorkspaceArena workspace(std::max<std::size_t>(
            ops::hyper_connection_mix_workspace_capacity_bytes(down.weight, up.weight, ops::LinearPolicy::A16Only,
                                                               kStreams, kRank, max_columns),
            1));
        std::printf("# gpu=%s timed=hyper_connection_mix (graph of %d calls, distinct weights) codec=%s S=%d H=%d "
                    "rank=%d weights=%.2f MB per call\n",
                    context.props.name, options.layers, options.p12 ? "p12" : (options.bf16 ? "bf16" : "q8"), kStreams, kHidden, kRank,
                    (down.bytes + up.bytes) / 1e6);
        for (const int columns : options.columns) {
            const DeviceBuffer residual = bench::make_bf16(static_cast<std::size_t>(kWidth) * columns,
                                                           options.seed + 10 + columns, -2.0F, 2.0F);
            DeviceBuffer x(sizeof(std::uint16_t) * kHidden * columns);
            DeviceBuffer inject(sizeof(float) * kStreams * columns);
            const Tensor tr(residual.p, DType::BF16, {kWidth, columns});
            const Tensor tw(norm.p, DType::BF16, {kWidth});
            Tensor tx(x.p, DType::BF16, {kHidden, columns});
            Tensor ti(inject.p, DType::FP32, {kStreams, columns});
            // --a8: the mix also writes x's 8-bit columns (slice sums on), 800 words a column.
            DeviceBuffer words(options.a8 ? sizeof(std::int32_t) * 800 * columns : 4);
            Tensor tw8(words.p, DType::I32, {800, columns});
            bench::TimedGraph graph;
            graph.capture(context.stream, [&](cudaStream_t s) {
                for (int l = 0; l < options.layers; ++l) {
                    auto scope = workspace.scope();
                    if (options.a8) {
                        if (!ops::hyper_connection_mix_a8_encoded(tr, tw, downs[static_cast<std::size_t>(l)]->weight,
                                                                  ups[static_cast<std::size_t>(l)]->weight, kStreams,
                                                                  kRank, 1e-6F, tx, &ti, tw8, true, workspace, s)) {
                            throw std::invalid_argument("--a8: the fused route does not serve this width");
                        }
                        continue;
                    }
                    ops::hyper_connection_mix(tr, tw, downs[static_cast<std::size_t>(l)]->weight,
                                              ups[static_cast<std::size_t>(l)]->weight, ops::LinearPolicy::A16Only,
                                              kStreams, kRank, 1e-6F, tx, &ti, workspace, s);
                }
            });
            const auto timing = bench::measure_graph(graph, context.stream, options.warmup, options.repeat);
            const double per = 1.0 / options.layers;
            std::vector<std::uint8_t> bytes(x.bytes + inject.bytes);
            bench::read_back(bytes.data(), x.p, x.bytes);
            bench::read_back(bytes.data() + x.bytes, inject.p, inject.bytes);
            std::printf("T=%d | per call median %7.2f us (min %7.2f, p95 %7.2f) | %6.0f GB/s | output fnv1a %016llx\n",
                        columns, timing.median_us * per, timing.min_us * per, timing.p95_us * per,
                        (down.bytes + up.bytes) / (timing.median_us * per * 1e-6) / 1e9,
                        static_cast<unsigned long long>(fnv1a(bytes)));
        }
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "infernix_hyper_connection_bench: %s\n", error.what());
        return 1;
    }
}
