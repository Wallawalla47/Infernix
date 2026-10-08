// Tile sweep for the Qwen3.8-Flash-Next BF16 linear problems of the bit-exact (recipe A) artifact: the
// dense projections NVIDIA stores in BF16 (Dense8 re-stores them in Q8) and the two BF16 groups both
// recipes share.
//
// Decode widths (T = 1..8, the column-invariant skinny GEMV of the runtime-shape fallback): each shape's
// time against its DRAM floor (weight bytes at --dram-gbps), so a shape whose skinny route is far from
// the floor is visible. Prefill and verification widths (T >= 9): for each problem and T it times, with
// cold-L2 CUDA graphs:
//   fallback    the runtime-shape GEMM the problems used before V0 (VM1 baseline),
//   registered  the public ops::linear route that production dispatch selects now,
//   candidates  private launchers of the existing BF16 kernel templates (runtime K).
// Every candidate is first screened against the fallback at T = 77 and 1000 (partial token tiles,
// and the 4304 row/K tails); a candidate that fails the screen is never selected.
// It then proposes, per problem, a selector body for src/ops/linear/bf16/shapes/n<N>_k<K>.cu: the
// fastest candidate at each measured T, merged into intervals while the kept candidate stays
// within --tolerance-pct of the fastest. A boundary sits on the last measured T of its interval;
// refine a seam with --t-list. T <= 8 stays on the fallback's skinny GEMV and is not swept.
//
// Usage:
//   infernix_bf16_flash_next_sweep [--shape LABEL] [--t-list 9,16,...] [--warmup N] [--repeat N]
//       [--tolerance-pct P] [--dram-gbps G] [--csv PATH]
#include "core/device.h"
#include "core/weight.h"
#include "direct_bf16_weight.cuh"
#include "infernix/ops/linear.h"
#include "infernix/ops/projection_fp32.h"
#include "ops/projection_fp32/projection_fp32_route.h"
#include "quantized_weight.cuh"
#include "infernix_bench_common.h"
#include "ops/linear/bf16/bf16_dispatch.h"
#include "ops/linear/bf16/bf16_instances.cuh"
#include "ops/linear/bf16/bf16_launch.cuh"
#include "ops/linear/bf16/bf16_shapes.h"
#include "ops/linear/bf16/bf16_skinny.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace infernix;
using namespace infernix::ops::detail;

namespace {

enum class Kind { Tma, TmaTail, Mma, Sliced };

struct Candidate {
    const char* name;
    const char* type;     // C++ schedule type, pasted into the proposed selector
    Kind kind;
    Bf16Launch launch;
    int row_multiple, k_multiple; // 0 = any (tail-capable)
};

#define INFERNIX_TMA(name, ...)                                                                      \
    Candidate {                                                                                    \
        name, #__VA_ARGS__, Kind::Tma, launch_bf16_tma_mma<__VA_ARGS__>, __VA_ARGS__::kBlockRows,  \
            __VA_ARGS__::kBlockK                                                                   \
    }
#define INFERNIX_TAIL(name, ...)                                                                     \
    Candidate { name, #__VA_ARGS__, Kind::TmaTail, launch_bf16_tma_mma<__VA_ARGS__>, 0, 0 }
#define INFERNIX_MMA(name, ...)                                                                      \
    Candidate {                                                                                    \
        name, #__VA_ARGS__, Kind::Mma, launch_bf16_mma<__VA_ARGS__>, __VA_ARGS__::kBlockRows,      \
            __VA_ARGS__::kBlockK                                                                   \
    }
#define INFERNIX_SLICED(name, ...)                                                                   \
    Candidate {                                                                                    \
        name, #__VA_ARGS__, Kind::Sliced, launch_bf16_sliced_k_mma<__VA_ARGS__>,                   \
            __VA_ARGS__::kBlockRows, __VA_ARGS__::kBlockK                                          \
    }

const std::vector<Candidate>& candidates() {
    static const std::vector<Candidate> list{
        INFERNIX_TMA("tma_r64t32k64s3", Bf16A16TmaMmaSchedule<64, 32, 64, 32, 16, 3>),
        INFERNIX_TMA("tma_r64t64k64s3", Bf16A16TmaMmaSchedule<64, 64, 64, 32, 32, 3>),
        INFERNIX_TMA("tma_r64t64k128s2", Bf16A16TmaMmaSchedule<64, 64, 128, 32, 32, 2>),
        INFERNIX_TMA("tma_r64t128k64s2", Bf16A16TmaMmaSchedule<64, 128, 64, 32, 32, 2>),
        INFERNIX_TMA("tma_r64t128k64s2_rows",
                   Bf16A16TmaMmaSchedule<64, 128, 64, 32, 32, 2, 1, Bf16MmaRaster::RowFast>),
        INFERNIX_TMA("tma_r64t128k64s3", Bf16A16TmaMmaSchedule<64, 128, 64, 32, 32, 3>),
        INFERNIX_TMA("tma_r64t128k64s3_rows",
                   Bf16A16TmaMmaSchedule<64, 128, 64, 32, 32, 3, 1, Bf16MmaRaster::RowFast>),
        INFERNIX_TMA("tma_r128t64k64s4", Bf16A16TmaMmaSchedule<128, 64, 64, 32, 32, 4>),
        INFERNIX_TMA("tma_r128t64k64s4_rows",
                   Bf16A16TmaMmaSchedule<128, 64, 64, 32, 32, 4, 1, Bf16MmaRaster::RowFast>),
        INFERNIX_TMA("tma_r128t64k128s2_rows",
                   Bf16A16TmaMmaSchedule<128, 64, 128, 32, 32, 2, 1, Bf16MmaRaster::RowFast>),
        INFERNIX_TMA("tma_r128t128k64s3_w64x32", Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3>),
        INFERNIX_TMA("tma_r128t128k64s3_w64x32_rows",
                   Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3, 1, Bf16MmaRaster::RowFast>),
        INFERNIX_TMA("tma_r128t128k64s3_w32x64_rows",
                   Bf16A16TmaMmaSchedule<128, 128, 64, 32, 64, 3, 1, Bf16MmaRaster::RowFast>),
        INFERNIX_TMA("tma_r128t128k64s3_w64x32_group8",
                   Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3, 1, Bf16MmaRaster::Grouped, 8>),
        INFERNIX_TMA("tma_r64t256k64s2_rows",
                   Bf16A16TmaMmaSchedule<64, 256, 64, 32, 64, 2, 1, Bf16MmaRaster::RowFast>),
        INFERNIX_TAIL("tail_r64t32s3", Bf16A16TmaTailMmaSchedule<64, 32, 32, 16, 3>),
        INFERNIX_TAIL("tail_r64t64s3", Bf16A16TmaTailMmaSchedule<64, 64, 32, 32, 3>),
        INFERNIX_TAIL("tail_r64t128s2", Bf16A16TmaTailMmaSchedule<64, 128, 32, 32, 2>),
        INFERNIX_TAIL("tail_r64t128s2_rows",
                    Bf16A16TmaTailMmaSchedule<64, 128, 32, 32, 2, 1, Bf16MmaRaster::RowFast>),
        INFERNIX_TAIL("tail_r64t128s3_rows",
                    Bf16A16TmaTailMmaSchedule<64, 128, 32, 32, 3, 1, Bf16MmaRaster::RowFast>),
        INFERNIX_TAIL("tail_r128t64s4_rows",
                    Bf16A16TmaTailMmaSchedule<128, 64, 32, 32, 4, 1, Bf16MmaRaster::RowFast>),
        INFERNIX_TAIL("tail_r128t128s3_w64x32_rows",
                    Bf16A16TmaTailMmaSchedule<128, 128, 64, 32, 3, 1, Bf16MmaRaster::RowFast>),
        INFERNIX_TAIL("tail_r128t128s3_w32x64_rows",
                    Bf16A16TmaTailMmaSchedule<128, 128, 32, 64, 3, 1, Bf16MmaRaster::RowFast>),
        INFERNIX_TAIL("tail_r128t128s3_w64x32_group8",
                    Bf16A16TmaTailMmaSchedule<128, 128, 64, 32, 3, 1, Bf16MmaRaster::Grouped, 8>),
        INFERNIX_TAIL("tail_r64t256s2_rows",
                    Bf16A16TmaTailMmaSchedule<64, 256, 32, 64, 2, 1, Bf16MmaRaster::RowFast>),
        INFERNIX_MMA("mma_r32t32k128s3", Bf16A16MmaR32T32K128S3),
        INFERNIX_MMA("mma_r32t32k192s2", Bf16A16MmaR32T32K192S2),
        INFERNIX_MMA("mma_r32t32k256s3", Bf16A16MmaR32T32K256S3),
        INFERNIX_MMA("mma_r64t32k64s3", Bf16A16MmaR64T32K64S3),
        INFERNIX_SLICED("sliced_r32t16w4", Bf16A16SlicedR32T16W4),
    };
    return list;
}

const char* launcher_name(Kind kind) {
    switch (kind) {
    case Kind::Tma:
    case Kind::TmaTail: return "launch_bf16_tma_mma";
    case Kind::Mma: return "launch_bf16_mma";
    case Kind::Sliced: return "launch_bf16_sliced_k_mma";
    }
    return "?";
}

struct Shape {
    const char* label;
    int n, k, max_t;
};

// max_t: a prefill chunk's columns (the deployment's --prefill-chunk 4096, wave-rounded chunks up to
// 8192), or the widest verification for the head.
constexpr Shape kShapes[] = {
    {"qkvg", 13952, 2560, 8192},    // QSA query, gate, key, value and index projections (both recipes)
    {"gdnab", 96, 2560, 8192},      // GDN a/b projections (both recipes)
    {"gdnqkvz", 16384, 2560, 8192}, // GDN query, key, value and z
    {"out6144", 2560, 6144, 8192},  // GDN output and QSA output projections
    {"plekv", 12800, 2560, 8192},   // PLE key and value projections
    {"shgu", 1280, 2560, 8192},     // shared expert gate and up
    {"shdown", 2560, 640, 8192},    // shared expert down
    {"hcdown", 324, 10240, 8192},   // hyper-connection mixer down and inject
    {"hcup", 10240, 320, 8192},     // hyper-connection mixer up
    {"fmdown", 320, 10240, 64},     // final mixer down (the head's columns only)
    {"head", 248320, 2560, 64},     // lm_head (verification columns)
};

const std::vector<int> kDefaultTs{9,   12,  16,  24,   32,   48,   64,   96,   100,  128, 192,
                                  255, 256, 384, 512,  768,  1024, 1536, 2048, 3072, 4096, 8192};

const std::vector<int> kVm1Ts{};

const std::vector<int> kDecodeTs{1, 2, 3, 4, 5, 6, 8};

struct Options {
    std::string shape;
    std::vector<int> ts = kDefaultTs;
    int warmup          = 2;
    int repeat          = 7;
    double tolerance    = 0.02;
    double dram_gbps    = 1792.0; // RTX 5090 GDDR7 peak
    bool decode_only    = false;
    std::string csv;
};

std::vector<int> parse_list(const char* text) {
    std::vector<int> result;
    std::string value(text);
    std::size_t start = 0;
    while (start < value.size()) {
        const std::size_t end = value.find(',', start);
        const std::string item =
            value.substr(start, end == std::string::npos ? std::string::npos : end - start);
        const int t = std::atoi(item.c_str());
        if (t < 9) throw std::invalid_argument("--t-list values must be >= 9");
        result.push_back(t);
        if (end == std::string::npos) break;
        start = end + 1;
    }
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    if (result.empty()) throw std::invalid_argument("--t-list is empty");
    return result;
}

Options parse(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const auto next       = [&]() -> const char* {
            if (i + 1 >= argc) throw std::invalid_argument("missing value for " + arg);
            return argv[++i];
        };
        if (arg == "--shape") {
            options.shape = next();
        } else if (arg == "--t-list") {
            options.ts = parse_list(next());
        } else if (arg == "--warmup") {
            options.warmup = std::atoi(next());
        } else if (arg == "--repeat") {
            options.repeat = std::atoi(next());
        } else if (arg == "--tolerance-pct") {
            options.tolerance = std::atof(next()) / 100.0;
        } else if (arg == "--decode-only") {
            options.decode_only = true;
        } else if (arg == "--dram-gbps") {
            options.dram_gbps = std::atof(next());
        } else if (arg == "--csv") {
            options.csv = next();
        } else {
            throw std::invalid_argument("unknown argument " + arg);
        }
    }
    if (options.warmup < 0 || options.repeat <= 0 || options.tolerance < 0)
        throw std::invalid_argument("invalid warmup/repeat/tolerance");
    return options;
}

float bf16_value(std::uint16_t bits) {
    const std::uint32_t word = static_cast<std::uint32_t>(bits) << 16;
    float value;
    std::memcpy(&value, &word, sizeof(value));
    return value;
}

std::vector<std::uint16_t> download(const DeviceBuffer& buffer, std::size_t elements) {
    std::vector<std::uint16_t> host(elements);
    CUDA_CHECK(cudaMemcpy(host.data(), buffer.p, elements * 2, cudaMemcpyDeviceToHost));
    return host;
}

// Relative L2 error and worst element error (relative to the largest reference magnitude) of a
// candidate against the fallback; any non-finite output fails.
bool screen(const std::vector<std::uint16_t>& got, const std::vector<std::uint16_t>& ref,
            double& relative_l2, double& relative_max) {
    double error = 0, norm = 0, worst = 0, largest = 0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const double a = bf16_value(got[i]), r = bf16_value(ref[i]);
        if (!std::isfinite(a)) {
            relative_l2 = relative_max = std::numeric_limits<double>::infinity();
            return false;
        }
        error += (a - r) * (a - r);
        norm += r * r;
        worst   = std::max(worst, std::fabs(a - r));
        largest = std::max(largest, std::fabs(r));
    }
    relative_l2  = norm > 0 ? std::sqrt(error / norm) : std::sqrt(error);
    relative_max = largest > 0 ? worst / largest : worst;
    return relative_l2 <= 1.0 / 128.0 && relative_max <= 1.0 / 64.0;
}

bench::ColdTiming time_route(const std::function<void(cudaStream_t)>& body,
                             bench::L2FlushBuffer& flush, cudaStream_t stream,
                             const Options& options) {
    bench::TimedGraph graph;
    graph.capture(stream, body);
    return bench::measure_cold_graph(graph, flush, stream, options.warmup, options.repeat);
}

struct Interval {
    int last_t; // inclusive upper bound; INT_MAX for the open last interval
    std::size_t candidate;
};

// us[i][c] is candidate c's median at Ts[i] (NaN when invalid or unmeasured).
std::vector<Interval> select_intervals(const std::vector<int>& ts,
                                       const std::vector<std::vector<double>>& us,
                                       double tolerance) {
    const auto best_at = [&](std::size_t i) {
        std::size_t best = SIZE_MAX;
        for (std::size_t c = 0; c < us[i].size(); ++c) {
            if (std::isnan(us[i][c])) continue;
            if (best == SIZE_MAX || us[i][c] < us[i][best]) best = c;
        }
        if (best == SIZE_MAX)
            throw std::runtime_error("no usable candidate at T=" + std::to_string(ts[i]));
        return best;
    };
    const auto within = [&](std::size_t c, std::size_t first, std::size_t last) {
        for (std::size_t i = first; i <= last; ++i) {
            const std::size_t best = best_at(i);
            if (std::isnan(us[i][c]) || us[i][c] > (1.0 + tolerance) * us[i][best]) return false;
        }
        return true;
    };
    // Greedy forward pass, then merge neighbours whose candidate covers both within tolerance.
    struct Span {
        std::size_t first, last, candidate;
    };
    std::vector<Span> spans;
    for (std::size_t i = 0; i < ts.size(); ++i) {
        if (!spans.empty() && within(spans.back().candidate, i, i)) {
            spans.back().last = i;
        } else {
            spans.push_back({i, i, best_at(i)});
        }
    }
    for (bool merged = true; merged;) {
        merged = false;
        for (std::size_t s = 0; s + 1 < spans.size(); ++s) {
            const Span a = spans[s], b = spans[s + 1];
            for (const std::size_t c : {b.candidate, a.candidate}) {
                if (within(c, a.first, b.last)) {
                    spans[s] = {a.first, b.last, c};
                    spans.erase(spans.begin() + static_cast<std::ptrdiff_t>(s) + 1);
                    merged = true;
                    break;
                }
            }
            if (merged) break;
        }
    }
    std::vector<Interval> result;
    for (std::size_t s = 0; s < spans.size(); ++s) {
        result.push_back(
            {s + 1 == spans.size() ? INT_MAX : ts[spans[s].last], spans[s].candidate});
    }
    return result;
}

void sweep_shape(const Shape& shape, const Options& options, std::FILE* csv,
                 bench::L2FlushBuffer& flush, cudaStream_t stream) {
    std::vector<int> ts;
    for (const int t : options.ts)
        if (t <= shape.max_t) ts.push_back(t);
    for (const int t : kVm1Ts)
        if (t <= shape.max_t && std::find(ts.begin(), ts.end(), t) == ts.end()) ts.push_back(t);
    std::sort(ts.begin(), ts.end());
    const int max_t = ts.back();

    auto weight = bench::make_direct_bf16_weight(shape.n, shape.k);
    DeviceBuffer x  = bench::make_bf16(static_cast<std::size_t>(shape.k) * max_t, 101U);
    DeviceBuffer out(static_cast<std::size_t>(shape.n) * max_t * 2);
    // Screen widths: a partial token tile and the widest swept call (both within the buffers).
    std::vector<int> screens{std::min(77, max_t), std::min(1000, max_t)};
    screens.erase(std::unique(screens.begin(), screens.end()), screens.end());
    DeviceBuffer ref(static_cast<std::size_t>(shape.n) * screens.back() * 2);
    const auto& list = candidates();

    // Decode widths (T <= 8, where Infernix needs a column's bits to be independent of how many
    // columns share the call): the skinny GEMV, the registered routes, and the registered T = 8 route
    // applied at every width. Each is checked for column invariance (column 0's bits at every T
    // against T = 8) and timed against the floor.
    const double weight_bytes = 2.0 * shape.n * shape.k;
    const double floor_us     = weight_bytes / (options.dram_gbps * 1e3);
    struct DecodeRoute {
        const char* name;
        std::function<Bf16Launch(int)> select;
    };
    std::vector<DecodeRoute> decode_routes{
        {"skinny", [](int t) { return select_bf16_general_launch(t); }},
        {"registered", [&](int t) { return select_bf16_launch(shape.n, shape.k, t, ops::LinearPolicy::A16Only); }},
        {"t8_route", [&](int) { return select_bf16_launch(shape.n, shape.k, 8, ops::LinearPolicy::A16Only); }},
    };
    // Skinny GEMV instances <warps, rows per warp, preload>: the same bits as "skinny" by construction
    // (checked below), so a selector may take the fastest per width. A preloading instance must cover
    // every chunk of a row (preload * 256 >= K).
#define INFERNIX_SKINNY(W, R, P)                                                                           \
    if (P == 0 || P * 256 >= shape.k)                                                                      \
        decode_routes.push_back({"skinny_w" #W "r" #R "p" #P, [](int t) { return bf16_skinny_launch<W, R, P>(t); }});
    INFERNIX_SKINNY(8, 2, 0) INFERNIX_SKINNY(4, 2, 0) INFERNIX_SKINNY(8, 1, 0) INFERNIX_SKINNY(4, 1, 0)
    INFERNIX_SKINNY(16, 1, 0) INFERNIX_SKINNY(16, 2, 0) INFERNIX_SKINNY(8, 4, 0) INFERNIX_SKINNY(4, 4, 0)
    INFERNIX_SKINNY(2, 1, 2) INFERNIX_SKINNY(4, 2, 2) INFERNIX_SKINNY(8, 2, 2) INFERNIX_SKINNY(4, 4, 2)
    INFERNIX_SKINNY(4, 1, 3) INFERNIX_SKINNY(8, 1, 3) INFERNIX_SKINNY(8, 2, 3) INFERNIX_SKINNY(4, 4, 3)
    INFERNIX_SKINNY(2, 1, 10) INFERNIX_SKINNY(4, 1, 10) INFERNIX_SKINNY(8, 1, 10) INFERNIX_SKINNY(16, 1, 10)
    INFERNIX_SKINNY(4, 2, 10) INFERNIX_SKINNY(8, 2, 10)
    INFERNIX_SKINNY(4, 1, 24) INFERNIX_SKINNY(8, 1, 24)
#undef INFERNIX_SKINNY
    // The default skinny route's T = 8 output, which every skinny instance must reproduce bit for bit.
    std::vector<std::uint16_t> skinny_reference;
    {
        Tensor x8(x.p, DType::BF16, {shape.k, 8});
        Tensor o8(out.p, DType::BF16, {shape.n, 8});
        select_bf16_general_launch(8)(x8, weight.weight, o8, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        skinny_reference = download(out, static_cast<std::size_t>(shape.n) * 8);
    }
    std::printf("\n## %s [%d,%d] decode widths (floor %.2f us = weight bytes at %.0f GB/s)\n", shape.label,
                shape.n, shape.k, floor_us, options.dram_gbps);
    std::printf("%-12s %-10s", "route", "invariant");
    for (const int t : kDecodeTs) std::printf(" %8s", ("T=" + std::to_string(t)).c_str());
    std::printf("\n");
    DeviceBuffer column(static_cast<std::size_t>(shape.n) * 8 * 2);
    for (const DecodeRoute& route : decode_routes) {
        // Column invariance: column 0 at every T equals column 0 at T = 8.
        Tensor x8(x.p, DType::BF16, {shape.k, 8});
        Tensor o8(column.p, DType::BF16, {shape.n, 8});
        bool invariant = true;
        try {
            route.select(8)(x8, weight.weight, o8, stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            const auto reference = download(column, static_cast<std::size_t>(shape.n));
            if (std::strncmp(route.name, "skinny_", 7) == 0 &&
                download(column, static_cast<std::size_t>(shape.n) * 8) != skinny_reference) {
                invariant = false; // not the skinny family's bits
            }
            for (const int t : kDecodeTs) {
                Tensor xt(x.p, DType::BF16, {shape.k, t});
                Tensor ot(out.p, DType::BF16, {shape.n, t});
                route.select(t)(xt, weight.weight, ot, stream);
                CUDA_CHECK(cudaStreamSynchronize(stream));
                if (download(out, static_cast<std::size_t>(shape.n)) != reference) invariant = false;
            }
        } catch (const std::exception& error) {
            std::printf("%-12s rejected: %s\n", route.name, error.what());
            continue;
        }
        std::printf("%-12s %-10s", route.name, invariant ? "yes" : "NO");
        for (const int t : kDecodeTs) {
            Tensor xt(x.p, DType::BF16, {shape.k, t});
            Tensor ot(out.p, DType::BF16, {shape.n, t});
            const Bf16Launch launch = route.select(t);
            const auto timing = time_route([&](cudaStream_t s) { launch(xt, weight.weight, ot, s); }, flush, stream,
                                           options);
            std::fprintf(csv, "%s,%d,%d,%d,decode_%s,%.3f,%.3f,%.3f,%.2f\n", shape.label, shape.n, shape.k, t,
                         route.name, timing.median_us, timing.min_us, timing.p95_us,
                         2.0 * shape.n * shape.k * t / timing.median_us * 1e-6);
            std::printf(" %8.2f", timing.median_us);
        }
        std::printf("\n");
    }
    std::fflush(stdout);

    if (std::strcmp(shape.label, "head") == 0) {
        // The heads run on ops::projection_fp32 (FP32 logits) in production, not on ops::linear: the BF16
        // head (recipe A) and the q8_g32_fp16 head (Dense8), each on the narrow and the tall mapping, whose
        // outputs must be equal bit for bit.
        using ops::detail::ProjectionRoute;
        DeviceBuffer logits(static_cast<std::size_t>(shape.n) * 16 * 4), other(static_cast<std::size_t>(shape.n) * 16 * 4);
        const Tensor rows(weight.storage.p, DType::BF16, {shape.k, shape.n});
        const Tensor* head[] = {&rows};
        auto q8 = bench::make_row_split_weight(QType::Q8_G32_FP16, shape.n, shape.k, shape.k);
        struct HeadForm {
            const char* name;
            double bytes;
            std::function<void(const Tensor&, Tensor&, ProjectionRoute, cudaStream_t)> run;
        };
        const HeadForm forms[] = {
            {"bf16", weight_bytes,
             [&](const Tensor& xt, Tensor& lt, ProjectionRoute route, cudaStream_t s) {
                 ops::detail::projection_fp32_route(xt, head, lt, route, s);
             }},
            {"q8", static_cast<double>(q8.model_weight_bytes()),
             [&](const Tensor& xt, Tensor& lt, ProjectionRoute route, cudaStream_t s) {
                 ops::detail::projection_fp32_route(xt, q8.weight, lt, route, s);
             }},
        };
        for (const HeadForm& form : forms) {
            const double head_floor = form.bytes / (options.dram_gbps * 1e3);
            std::printf("\n## head [%d,%d] %s on projection_fp32 (floor %.1f us)\n", shape.n, shape.k, form.name,
                        head_floor);
            std::printf("%8s %12s %12s %12s %s\n", "T", "narrow_us", "tall_us", "mma_us", "tall bits");
            for (const int t : {1, 2, 3, 4, 5, 6, 8, 9, 12, 16}) {
                Tensor xt(x.p, DType::BF16, {shape.k, t});
                Tensor lt(logits.p, DType::FP32, {shape.n, t});
                Tensor ot(other.p, DType::FP32, {shape.n, t});
                form.run(xt, lt, ProjectionRoute::Narrow, stream);
                form.run(xt, ot, ProjectionRoute::Tall, stream);
                CUDA_CHECK(cudaStreamSynchronize(stream));
                std::vector<float> a(static_cast<std::size_t>(shape.n) * t), b(a.size());
                CUDA_CHECK(cudaMemcpy(a.data(), logits.p, a.size() * 4, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(b.data(), other.p, b.size() * 4, cudaMemcpyDeviceToHost));
                const bool same = std::memcmp(a.data(), b.data(), a.size() * 4) == 0;
                const auto narrow = time_route([&](cudaStream_t s) { form.run(xt, lt, ProjectionRoute::Narrow, s); },
                                               flush, stream, options);
                const auto tall = time_route([&](cudaStream_t s) { form.run(xt, lt, ProjectionRoute::Tall, s); }, flush,
                                             stream, options);
                std::fprintf(csv, "head_%s,%d,%d,%d,narrow,%.3f,%.3f,%.3f,0\n", form.name, shape.n, shape.k, t,
                             narrow.median_us, narrow.min_us, narrow.p95_us);
                std::fprintf(csv, "head_%s,%d,%d,%d,tall,%.3f,%.3f,%.3f,0\n", form.name, shape.n, shape.k, t,
                             tall.median_us, tall.min_us, tall.p95_us);
                double mma_us = NAN; // the tensor-core mapping serves BF16 heads only
                if (std::strcmp(form.name, "bf16") == 0) {
                    const auto mma = time_route([&](cudaStream_t s) { form.run(xt, lt, ProjectionRoute::Mma, s); },
                                                flush, stream, options);
                    std::fprintf(csv, "head_%s,%d,%d,%d,mma,%.3f,%.3f,%.3f,0\n", form.name, shape.n, shape.k, t,
                                 mma.median_us, mma.min_us, mma.p95_us);
                    mma_us = mma.median_us;
                }
                std::printf("%8d %12.2f %12.2f %12.2f %s\n", t, narrow.median_us, tall.median_us, mma_us,
                            same ? "same" : "DIFFER");
            }
            std::fflush(stdout);
        }
        return; // ops::linear never serves the head
    }

    if (options.decode_only) return;

    // Screen every applicable candidate against the fallback.
    std::vector<bool> usable(list.size(), false);
    for (std::size_t c = 0; c < list.size(); ++c) {
        const Candidate& candidate = list[c];
        if ((candidate.row_multiple && shape.n % candidate.row_multiple) ||
            (candidate.k_multiple && shape.k % candidate.k_multiple)) {
            continue;
        }
        bool pass = true;
        for (const int t : screens) {
            Tensor xt(x.p, DType::BF16, {shape.k, t});
            Tensor rt(ref.p, DType::BF16, {shape.n, t});
            Tensor ot(out.p, DType::BF16, {shape.n, t});
            select_bf16_general_launch(t)(xt, weight.weight, rt, stream);
            CUDA_CHECK(cudaMemsetAsync(out.p, 0xff, static_cast<std::size_t>(shape.n) * t * 2,
                                       stream));
            try {
                candidate.launch(xt, weight.weight, ot, stream);
            } catch (const std::exception& error) {
                std::printf("# %s %s: launch rejected at T=%d: %s\n", shape.label, candidate.name,
                            t, error.what());
                pass = false;
                break;
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));
            const std::size_t elements = static_cast<std::size_t>(shape.n) * t;
            double relative_l2 = 0, relative_max = 0;
            const bool ok = screen(download(out, elements), download(ref, elements), relative_l2,
                                   relative_max);
            if (!ok) {
                std::printf("# %s %s: SCREEN FAIL at T=%d rel_l2=%.3g rel_max=%.3g\n", shape.label,
                            candidate.name, t, relative_l2, relative_max);
                pass = false;
            }
        }
        usable[c] = pass;
    }

    std::vector<std::vector<double>> us(ts.size(), std::vector<double>(list.size(), NAN));
    std::vector<double> fallback_us(ts.size()), registered_us(ts.size());
    for (std::size_t i = 0; i < ts.size(); ++i) {
        const int t = ts[i];
        Tensor xt(x.p, DType::BF16, {shape.k, t});
        Tensor ot(out.p, DType::BF16, {shape.n, t});
        const double flops = 2.0 * shape.n * shape.k * static_cast<double>(t);
        const auto record  = [&](const char* route, const bench::ColdTiming& timing) {
            std::fprintf(csv, "%s,%d,%d,%d,%s,%.3f,%.3f,%.3f,%.2f\n", shape.label, shape.n,
                         shape.k, t, route, timing.median_us, timing.min_us, timing.p95_us,
                         flops / timing.median_us * 1e-6);
        };
        const auto fallback = time_route(
            [&](cudaStream_t s) { select_bf16_general_launch(t)(xt, weight.weight, ot, s); },
            flush, stream, options);
        record("fallback", fallback);
        fallback_us[i]        = fallback.median_us;
        const auto registered = time_route(
            [&](cudaStream_t s) { ops::linear(xt, weight.weight, ot, s); }, flush, stream, options);
        record("registered", registered);
        registered_us[i] = registered.median_us;
        for (std::size_t c = 0; c < list.size(); ++c) {
            if (!usable[c]) continue;
            const auto timing = time_route(
                [&](cudaStream_t s) { list[c].launch(xt, weight.weight, ot, s); }, flush, stream,
                options);
            record(list[c].name, timing);
            us[i][c] = timing.median_us;
        }
        std::fflush(csv);
    }

    // Selection over the swept points only (the VM1 points are reported, not selected from, when
    // the caller narrowed --t-list).
    std::vector<int> selection_ts;
    std::vector<std::vector<double>> selection_us;
    for (std::size_t i = 0; i < ts.size(); ++i) {
        if (std::find(options.ts.begin(), options.ts.end(), ts[i]) == options.ts.end()) continue;
        selection_ts.push_back(ts[i]);
        selection_us.push_back(us[i]);
    }
    const auto intervals = selection_ts.empty()
                               ? std::vector<Interval>{}
                               : select_intervals(selection_ts, selection_us, options.tolerance);

    std::printf("\n## %s [%d,%d]\n", shape.label, shape.n, shape.k);
    std::printf("%8s %12s %12s %12s %10s %10s  %s\n", "T", "fallback_us", "registered_us",
                "best_us", "reg/best", "fb/reg", "best candidate");
    for (std::size_t i = 0; i < ts.size(); ++i) {
        std::size_t best = SIZE_MAX;
        for (std::size_t c = 0; c < list.size(); ++c)
            if (!std::isnan(us[i][c]) && (best == SIZE_MAX || us[i][c] < us[i][best])) best = c;
        if (best == SIZE_MAX) continue;
        std::printf("%8d %12.2f %12.2f %12.2f %10.3f %10.2f  %s\n", ts[i], fallback_us[i],
                    registered_us[i], us[i][best], registered_us[i] / us[i][best],
                    fallback_us[i] / registered_us[i], list[best].name);
    }
    if (intervals.empty()) return;
    std::printf("\nProposed selector (tolerance %.1f%%; seams at the last measured T):\n",
                options.tolerance * 100.0);
    std::printf("Bf16Launch select_bf16_n%d_k%d(std::int32_t tokens) {\n", shape.n, shape.k);
    std::printf("    if (tokens <= 8) return select_bf16_general_launch(tokens);\n");
    for (const Interval& interval : intervals) {
        const Candidate& candidate = list[interval.candidate];
        if (interval.last_t != INT_MAX)
            std::printf("    if (tokens <= %d)\n    ", interval.last_t);
        // The measured instance itself (runtime K): a static-K instance of the same schedule can time
        // differently (TMA tail tiles at [13952, 2560], T = 100-128: 74 us static, 58 us runtime).
        std::printf("    return %s<%s>; // %s\n", launcher_name(candidate.kind), candidate.type, candidate.name);
    }
    std::printf("}\n");
    std::fflush(stdout);
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse(argc, argv);
        std::FILE* csv        = stdout;
        if (!options.csv.empty()) {
            csv = std::fopen(options.csv.c_str(), "w");
            if (csv == nullptr) throw std::runtime_error("cannot open " + options.csv);
        }
        std::fprintf(csv, "shape,N,K,T,route,median_us,min_us,p95_us,tflops\n");
        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        bench::L2FlushBuffer flush(256ULL << 20);
        bool matched = false;
        for (const Shape& shape : kShapes) {
            if (!options.shape.empty() && options.shape != shape.label) continue;
            matched = true;
            sweep_shape(shape, options, csv, flush, stream);
        }
        CUDA_CHECK(cudaStreamDestroy(stream));
        if (csv != stdout) std::fclose(csv);
        if (!matched) throw std::invalid_argument("unknown --shape " + options.shape);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "bf16_flash_next_sweep: %s\n", error.what());
        return 1;
    }
}
