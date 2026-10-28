// Op benchmark of ops::ggml_linear (GGUF block weights) at the Qwen3.8-Flash-Next IQ3_S import's dense projection
// shapes (or --shape ones), timing the public call on each route at prefill widths:
//   exact : LinearRoutes::Exact (the exact A16 tiles over decoded planes from 32 columns);
//   bf16  : LinearRoutes::GgmlBf16Wide (ggml-bf16-prefill: BF16 tiles over the same planes from 32 columns);
//   a8    : LinearRoutes::GgmlA8 | GgmlA8Wide (--a8-activations: the 8-bit narrow route to 31 columns, the int8
//           tiles over their own planes from 32; LOSSY).
// Each point is one ggml_linear call of all of a shape's segments (one output), BF16 outputs, with a workspace of
// exactly ggml_linear_workspace_bytes; the time includes the route's plane decode and column encoding. TOPS counts
// 2 * rows * K * T. --cold flushes the L2 before each call. The output hash shows whether two builds' bits agree.
//
//   infernix_ggml_linear_bench [--tokens 512,2048,4096] [--shapes all|gdn_qkvz,attn_qgkv,...] [--routes exact,bf16,a8]
//                              [--shape NAME=FORMAT:ROWS:K[:tiled][:perm128] ...] [--warmup 3] [--repeat 10] [--cold]
//                              [--seed N]

#include "infernix/ops/ggml_linear.h"

#include "core/device.h"
#include "core/ggml_format.h"
#include "infernix_bench_common.h"

#include <cuda_runtime.h>

#include <algorithm>
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

struct Part {
    QType format;
    int rows;
};

struct Shape {
    std::string name;
    int k = 0;
    std::vector<Part> parts;
    ops::GgmlLayout layout = ops::GgmlLayout::Rows;
    int perm_block         = 0; // > 0: the input permutation in blocks of this many elements
};

// The IQ3_S import's dense projections (Qwen3.8-Flash-Next, hidden 2560; test_ggml_linear_a8.cu's shapes).
std::vector<Shape> builtin_shapes() {
    using L = ops::GgmlLayout;
    return {
        {"gdn_qkvz", 2560, {{QType::GGML_Q6_K, 2048}, {QType::GGML_Q6_K, 2048}, {QType::GGML_Q6_K, 6144}, {QType::GGML_IQ4_XS, 6144}}, L::Tiles16},
        {"attn_qgkv", 2560, {{QType::GGML_IQ4_XS, 6144}, {QType::GGML_IQ4_XS, 6144}, {QType::GGML_Q6_K, 512}, {QType::GGML_Q6_K, 512}}, L::Tiles16},
        {"shexp_gate_up", 2560, {{QType::GGML_Q4_K, 640}, {QType::GGML_Q6_K, 640}}, L::Tiles16},
        {"ssm_out", 6144, {{QType::GGML_Q6_K, 2560}}, L::Tiles16, 128},
        {"attn_output", 6144, {{QType::GGML_IQ4_XS, 2560}}, L::Tiles16},
        {"attn_qkv_q5k", 2560, {{QType::GGML_Q5_K, 10240}}, L::Rows},
        {"attn_gate_q4k", 2560, {{QType::GGML_Q4_K, 6144}}, L::Rows},
        {"iq3s_6144", 2560, {{QType::GGML_IQ3_S, 6144}}, L::Rows},
    };
}

QType parse_format(const std::string& name) {
    const std::pair<const char*, QType> names[] = {{"q2_0", QType::GGML_Q2_0},     {"q4_k", QType::GGML_Q4_K},
                                                   {"q5_k", QType::GGML_Q5_K},     {"q6_k", QType::GGML_Q6_K},
                                                   {"iq2_s", QType::GGML_IQ2_S},   {"iq3_xxs", QType::GGML_IQ3_XXS},
                                                   {"iq3_s", QType::GGML_IQ3_S},   {"iq4_xs", QType::GGML_IQ4_XS},
                                                   {"iq4_nl", QType::GGML_IQ4_NL}};
    for (const auto& [n, q] : names) {
        if (name == n) { return q; }
    }
    throw std::invalid_argument("unknown GGUF format " + name);
}

std::vector<std::string> split(const std::string& text, char sep) {
    std::vector<std::string> out;
    std::size_t begin = 0;
    while (begin <= text.size()) {
        const std::size_t end = text.find(sep, begin);
        out.push_back(text.substr(begin, end == std::string::npos ? std::string::npos : end - begin));
        if (end == std::string::npos) { break; }
        begin = end + 1;
    }
    return out;
}

// NAME=FORMAT:ROWS:K[:tiled][:perm128]
Shape parse_shape(const std::string& text) {
    const std::size_t eq = text.find('=');
    if (eq == std::string::npos) { throw std::invalid_argument("--shape takes NAME=FORMAT:ROWS:K[:tiled][:permN]"); }
    const auto fields = split(text.substr(eq + 1), ':');
    if (fields.size() < 3) { throw std::invalid_argument("--shape takes NAME=FORMAT:ROWS:K[:tiled][:permN]"); }
    Shape s{text.substr(0, eq), std::stoi(fields[2]), {{parse_format(fields[0]), std::stoi(fields[1])}}};
    for (std::size_t i = 3; i < fields.size(); ++i) {
        if (fields[i] == "tiled") {
            s.layout = ops::GgmlLayout::Tiles16;
        } else if (fields[i].rfind("perm", 0) == 0) {
            s.perm_block = std::stoi(fields[i].substr(4));
        } else {
            throw std::invalid_argument("unknown --shape field " + fields[i]);
        }
    }
    return s;
}

struct Route {
    std::string name;
    ops::LinearRoutes routes;
};

struct Options {
    std::vector<int> tokens{512, 2048, 4096};
    std::vector<Shape> shapes;
    std::vector<Route> routes;
    int warmup         = 3;
    int repeat         = 10;
    bool cold          = false;
    std::uint32_t seed = 20261028U;
};

Options parse_options(int argc, char** argv) {
    Options options;
    std::vector<std::string> wanted{"all"}, routes{"exact", "bf16", "a8"};
    std::vector<Shape> custom;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto value = [&]() -> std::string {
            if (i + 1 >= argc) { throw std::invalid_argument(arg + " needs a value"); }
            return argv[++i];
        };
        if (arg == "--tokens") {
            options.tokens.clear();
            for (const auto& t : split(value(), ',')) { options.tokens.push_back(std::stoi(t)); }
        } else if (arg == "--shapes") {
            wanted = split(value(), ',');
        } else if (arg == "--shape") {
            custom.push_back(parse_shape(value()));
        } else if (arg == "--routes") {
            routes = split(value(), ',');
        } else if (arg == "--warmup") {
            options.warmup = std::stoi(value());
        } else if (arg == "--repeat") {
            options.repeat = std::stoi(value());
        } else if (arg == "--cold") {
            options.cold = true;
        } else if (arg == "--seed") {
            options.seed = static_cast<std::uint32_t>(std::stoul(value()));
        } else {
            throw std::invalid_argument("unknown option " + arg);
        }
    }
    for (const Shape& s : builtin_shapes()) {
        if (custom.empty() && (std::find(wanted.begin(), wanted.end(), "all") != wanted.end() ||
                               std::find(wanted.begin(), wanted.end(), s.name) != wanted.end())) {
            options.shapes.push_back(s);
        }
    }
    options.shapes.insert(options.shapes.end(), custom.begin(), custom.end());
    if (options.shapes.empty()) { throw std::invalid_argument("no shape selected"); }
    for (const auto& r : routes) {
        if (r == "exact") {
            options.routes.push_back({r, ops::LinearRoutes::Exact});
        } else if (r == "bf16") {
            options.routes.push_back({r, ops::LinearRoutes::GgmlBf16Wide});
        } else if (r == "a8") {
            options.routes.push_back({r, ops::LinearRoutes::GgmlA8 | ops::LinearRoutes::GgmlA8Wide});
        } else {
            throw std::invalid_argument("--routes takes exact, bf16 and a8");
        }
    }
    return options;
}

// Random block bytes with finite binary16 scales of the import's range (exponent fields 12-13).
std::vector<std::uint8_t> random_blocks(QType q, int rows, int k, std::mt19937& rng) {
    const int blocks = k / static_cast<int>(ggml_block_elems(q)), size = static_cast<int>(ggml_block_bytes(q));
    std::vector<std::uint8_t> data(static_cast<std::size_t>(rows) * blocks * size);
    for (auto& b : data) { b = static_cast<std::uint8_t>(rng()); }
    const auto scale = [&](std::uint8_t* p) {
        const auto w = static_cast<std::uint16_t>((rng() & 0x8000U) | (rng() & 0x3FFU) | ((12 + rng() % 2) << 10));
        p[0]         = static_cast<std::uint8_t>(w & 0xFF);
        p[1]         = static_cast<std::uint8_t>(w >> 8);
    };
    for (int i = 0; i < rows * blocks; ++i) {
        std::uint8_t* b = data.data() + static_cast<std::size_t>(i) * size;
        if (q == QType::GGML_Q6_K) {
            scale(b + 208);
        } else {
            scale(b);
            if (q == QType::GGML_Q4_K || q == QType::GGML_Q5_K) { scale(b + 2); }
        }
    }
    return data;
}

// ggml_tiles16_v1 of `rows` ggml_rows_v1 rows (tile i at 16 i row_bytes, block b of its row r at (16 b + r) * block
// bytes, missing rows zero blocks).
std::vector<std::uint8_t> tiles16(QType q, int rows, int k, const std::vector<std::uint8_t>& data) {
    const std::size_t bb = ggml_block_bytes(q), blocks = static_cast<std::size_t>(k) / ggml_block_elems(q), row_bytes = blocks * bb;
    std::vector<std::uint8_t> out(static_cast<std::size_t>((rows + 15) / 16) * 16 * row_bytes, 0);
    for (int row = 0; row < rows; ++row) {
        for (std::size_t b = 0; b < blocks; ++b) {
            std::memcpy(out.data() + static_cast<std::size_t>(row / 16) * 16 * row_bytes + (16 * b + row % 16) * bb,
                        data.data() + static_cast<std::size_t>(row) * row_bytes + b * bb, bb);
        }
    }
    return out;
}

void run_shape(const Shape& shape, const Options& options, bench::L2FlushBuffer* flush, cudaStream_t stream) {
    std::mt19937 rng(options.seed);
    std::vector<DeviceBuffer> weights;
    std::vector<ops::GgmlSegment> segments;
    int rows = 0;
    for (const Part& part : shape.parts) {
        auto bytes = random_blocks(part.format, part.rows, shape.k, rng);
        if (shape.layout == ops::GgmlLayout::Tiles16) { bytes = tiles16(part.format, part.rows, shape.k, bytes); }
        weights.emplace_back(bytes.size());
        weights.back().copy_from_host(bytes.data(), bytes.size(), 0);
        segments.push_back({part.format, static_cast<const std::uint8_t*>(weights.back().p), part.rows, rows, 0, shape.layout});
        rows += part.rows;
    }
    DeviceBuffer perm;
    if (shape.perm_block > 0) {
        std::vector<std::int32_t> p(static_cast<std::size_t>(shape.k / shape.perm_block));
        for (std::size_t i = 0; i < p.size(); ++i) { p[i] = static_cast<std::int32_t>(i); }
        std::shuffle(p.begin(), p.end(), rng);
        perm = DeviceBuffer(p.size() * sizeof(std::int32_t));
        perm.copy_from_host(p.data(), perm.bytes, 0);
    }
    for (const int tokens : options.tokens) {
        const DeviceBuffer x = bench::make_bf16(static_cast<std::size_t>(shape.k) * tokens, options.seed + 7, -1.0F, 1.0F);
        DeviceBuffer out(sizeof(std::uint16_t) * static_cast<std::size_t>(rows) * tokens);
        WorkspaceArena workspace(ops::ggml_linear_workspace_bytes(shape.k, tokens));
        Tensor tx(x.p, DType::BF16, {shape.k, tokens});
        Tensor ty(out.p, DType::BF16, {rows, tokens});
        double base = 0.0;
        for (const Route& route : options.routes) {
            const auto launch = [&](cudaStream_t s) {
                ops::ggml_linear(tx, segments, ty, shape.perm_block > 0 ? static_cast<const std::int32_t*>(perm.p) : nullptr,
                                 shape.perm_block, workspace, s, route.routes);
            };
            const auto timing = options.cold ? bench::measure_cold_launch(launch, *flush, stream, options.warmup, options.repeat)
                                             : bench::measure_launch(launch, stream, options.warmup, options.repeat);
            if (base == 0.0) { base = timing.median_us; }
            const double tops = 2.0 * rows * shape.k * static_cast<double>(tokens) / (timing.median_us * 1e-6) / 1e12;
            std::vector<std::uint8_t> bytes(out.bytes);
            CUDA_CHECK(cudaMemcpy(bytes.data(), out.p, out.bytes, cudaMemcpyDeviceToHost));
            std::uint64_t hash = 1469598103934665603ULL;
            for (const std::uint8_t b : bytes) { hash = (hash ^ b) * 1099511628211ULL; }
            std::printf("%-14s N=%6d K=%5d T=%5d route=%-5s median %10.1f us (min %10.1f, p95 %10.1f) | %7.1f TOPS | %.3fx the "
                        "first route's time | output fnv1a %016llx\n",
                        shape.name.c_str(), rows, shape.k, tokens, route.name.c_str(), timing.median_us, timing.min_us,
                        timing.p95_us, tops, timing.median_us / base, static_cast<unsigned long long>(hash));
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        DeviceContext context;
        std::printf("# gpu=%s timed=ggml_linear (one call of a shape's segments, plane decode and encoding included%s), BF16 "
                    "outputs\n",
                    context.props.name, options.cold ? ", L2 flushed before each call" : "");
        std::unique_ptr<bench::L2FlushBuffer> flush;
        if (options.cold) { flush = std::make_unique<bench::L2FlushBuffer>(256ULL << 20); }
        for (const Shape& shape : options.shapes) { run_shape(shape, options, flush.get(), context.stream); }
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "infernix_ggml_linear_bench: %s\n", error.what());
        return 1;
    }
}
