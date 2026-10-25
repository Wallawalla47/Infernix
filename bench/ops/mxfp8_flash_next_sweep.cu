// Schedule sweep for the MXFP8 (fp8_e4m3fn_g32_e8m0, g32_scale_rows16_v1) W8A16 linear problems of
// NVIDIA's Qwen3.8-Flash-Next-FP4-Local (docs/maintainer/qwen3_8-flash-next-design.md §6.1.2): the GDN
// query/key/value/z and output projections, the QSA query/gate/key/value/indexer and output
// projections and the shared experts.
//
// Every candidate is a schedule of the one MXFP8 route (mxfp8_a16_tma_mma.cuh) with one of its two
// producers, whose column bits must not depend on the schedule, the producer or the width (the exact
// route's request invariance). The sweep
// first checks that: every candidate's output at T = 1 and T = 300 must equal the first candidate's
// bit for bit (and pass an FP64 screen on sampled rows); a candidate that differs is reported and
// dropped. It then times the survivors with cold-L2 CUDA graphs at every width of --t-list (decode
// widths included) and proposes a selector: the fastest candidate per measured T, merged into
// intervals within --tolerance-pct.
//
// Usage:
//   infernix_mxfp8_flash_next_sweep [--shape LABEL] [--t-list 1,2,...] [--warmup N] [--repeat N]
//       [--tolerance-pct P] [--dram-gbps G] [--csv PATH]
#include "core/device.h"
#include "core/weight.h"
#include "infernix_bench_common.h"
#include "ops/linear/mxfp8/mxfp8_launch.cuh"
#include "roofline.h"

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
namespace roofline = infernix::bench::roofline;
using namespace infernix::ops::detail;

namespace {

using Mxfp8Launch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

struct Candidate {
    const char* name;
    const char* type; // C++ schedule type, pasted into the proposed selector
    Mxfp8Launch launch;
    int row_tile;     // rows must be a multiple
};

template <int K>
using G = Fp8Geometry<16, K>; // only the static K enters the launch

// Candidates no shape selects (2026-10-20). A cp_ candidate's stages are filled by cp.async (_p2 / _p4:
// spread over two or four producer warps), a tma_ candidate's by TMA tensor loads (Mxfp8Tma).
using T32R16   = Fp8A16RingMmaSchedule<32, 16, 1, 2, 4, 2>;
using T32R32S3 = Fp8A16RingMmaSchedule<32, 32, 1, 4, 3, 2>;

#define INFERNIX_C(name, rows, ...) Candidate{name, #__VA_ARGS__, mxfp8_linear_a16_tma<G<K>, __VA_ARGS__>, rows}

template <int K>
const std::vector<Candidate>& candidates() {
    static const std::vector<Candidate> list{
        INFERNIX_C("cp_t16r16s4b3", 16, Mxfp8TmaT16R16),
        INFERNIX_C("cp_t16r32s3b3", 32, Mxfp8TmaT16R32),
        INFERNIX_C("cp_t16r64s3b2", 64, Mxfp8TmaT16R64),
        INFERNIX_C("cp_t16r128s3", 128, Mxfp8TmaT16R128),
        INFERNIX_C("cp_t32r16s4b2", 16, T32R16),
        INFERNIX_C("cp_t32r32s4b2", 32, Mxfp8TmaT32R32),
        INFERNIX_C("cp_t32r32s3b2", 32, T32R32S3),
        INFERNIX_C("cp_t32r64s6", 64, Fp8A16RingT32R64),
        INFERNIX_C("cp_t32r64s3b2", 64, Fp8A16RingT32R64S3),
        INFERNIX_C("cp_t32r128s4", 128, Fp8A16RingT32R128),
        INFERNIX_C("cp_t64r64s2b2", 64, Mxfp8TmaT64R64),
        INFERNIX_C("cp_t64r128s3", 128, Fp8A16RingT64R128),
        INFERNIX_C("cp_t64r256s2", 256, Mxfp8TmaT64R256),
        INFERNIX_C("cp_t128r128s2", 128, Mxfp8TmaT128R128),
        INFERNIX_C("cp_t16r32s3b3_p2", 32, Mxfp8P2<Mxfp8TmaT16R32>),
        INFERNIX_C("cp_t16r64s3b2_p2", 64, Mxfp8P2<Mxfp8TmaT16R64>),
        INFERNIX_C("cp_t16r128s3_p2", 128, Mxfp8P2<Mxfp8TmaT16R128>),
        INFERNIX_C("cp_t16r128s3_p4", 128, Mxfp8P4<Mxfp8TmaT16R128>),
        INFERNIX_C("cp_t32r32s4b2_p2", 32, Mxfp8P2<Mxfp8TmaT32R32>),
        INFERNIX_C("cp_t32r64s3b2_p2", 64, Mxfp8P2<Fp8A16RingT32R64S3>),
        INFERNIX_C("cp_t64r64s2b2_p2", 64, Mxfp8P2<Mxfp8TmaT64R64>),
        INFERNIX_C("cp_t64r128s3_p2", 128, Mxfp8P2<Fp8A16RingT64R128>),
        INFERNIX_C("cp_t64r128s3_p4", 128, Mxfp8P4<Fp8A16RingT64R128>),
        INFERNIX_C("cp_t64r256s2_p2", 256, Mxfp8P2<Mxfp8TmaT64R256>),
        INFERNIX_C("cp_t64r256s2_p4", 256, Mxfp8P4<Mxfp8TmaT64R256>),
        INFERNIX_C("cp_t128r128s2_p2", 128, Mxfp8P2<Mxfp8TmaT128R128>),
        INFERNIX_C("cp_t128r128s2_p4", 128, Mxfp8P4<Mxfp8TmaT128R128>),
        INFERNIX_C("tma_t16r16s4b3", 16, Mxfp8Tma<Mxfp8TmaT16R16>),
        INFERNIX_C("tma_t16r32s3b3", 32, Mxfp8Tma<Mxfp8TmaT16R32>),
        INFERNIX_C("tma_t16r64s3b2", 64, Mxfp8Tma<Mxfp8TmaT16R64>),
        INFERNIX_C("tma_t16r128s3", 128, Mxfp8Tma<Mxfp8TmaT16R128>),
        INFERNIX_C("tma_t32r16s4b2", 16, Mxfp8Tma<T32R16>),
        INFERNIX_C("tma_t32r32s4b2", 32, Mxfp8Tma<Mxfp8TmaT32R32>),
        INFERNIX_C("tma_t32r32s3b2", 32, Mxfp8Tma<T32R32S3>),
        INFERNIX_C("tma_t32r64s6", 64, Mxfp8Tma<Fp8A16RingT32R64>),
        INFERNIX_C("tma_t32r64s3b2", 64, Mxfp8Tma<Fp8A16RingT32R64S3>),
        INFERNIX_C("tma_t32r128s4", 128, Mxfp8Tma<Fp8A16RingT32R128>),
        INFERNIX_C("tma_t64r64s2b2", 64, Mxfp8Tma<Mxfp8TmaT64R64>),
        INFERNIX_C("tma_t64r128s3", 128, Mxfp8Tma<Fp8A16RingT64R128>),
        INFERNIX_C("tma_t64r256s2", 256, Mxfp8Tma<Mxfp8TmaT64R256>),
        INFERNIX_C("tma_t128r128s2", 128, Mxfp8Tma<Mxfp8TmaT128R128>),
    };
    return list;
}

#undef INFERNIX_C

struct Shape {
    const char* label;
    int n, k;
};

constexpr Shape kShapes[] = {
    {"gdnqkvz", 16384, 2560}, // GDN query, key, value and z
    {"qsa", 13952, 2560},     // QSA query, gate, key, value and indexer
    {"out6144", 2560, 6144},  // GDN output and QSA output projections
    {"shgu", 1280, 2560},     // shared expert gate and up
    {"shdown", 2560, 640},    // shared expert down
};

const std::vector<int> kDefaultTs{1,   2,   3,   4,   5,   6,   7,    8,    9,    12,   16,   24,   32,  48,
                                  64,  96,  128, 192, 256, 384, 512,  768,  1024, 1536, 2048, 3072, 4096, 8192};

struct Options {
    std::string shape;
    std::vector<int> ts = kDefaultTs;
    int warmup          = 2;
    int repeat          = 7;
    double tolerance    = 0.02;
    double dram_gbps    = roofline::kDramSpecGBs;
    std::string csv;
};

std::vector<int> parse_list(const char* text) {
    std::vector<int> result;
    std::string value(text);
    std::size_t start = 0;
    while (start < value.size()) {
        const std::size_t end = value.find(',', start);
        const int t = std::atoi(value.substr(start, end == std::string::npos ? std::string::npos : end - start).c_str());
        if (t < 1) throw std::invalid_argument("--t-list values must be positive");
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

double e4m3_value(std::uint8_t code) {
    const int s = code >> 7, e = (code >> 3) & 15, m = code & 7;
    const double v = e == 0 ? m * std::ldexp(1.0, -9) : (1.0 + m / 8.0) * std::ldexp(1.0, e - 7);
    return s ? -v : v;
}

template <class T>
std::vector<T> download(const void* device, std::size_t elements) {
    std::vector<T> host(elements);
    CUDA_CHECK(cudaMemcpy(host.data(), device, elements * sizeof(T), cudaMemcpyDeviceToHost));
    return host;
}

// A checkpoint-like MXFP8 weight: every finite E4M3FN code and group scales over 2^-7..2^7, with
// activations over 25 binades, so FP32 sums round and a different reduction order shows in the bits.
struct HostWeight {
    std::vector<std::uint8_t> payload;
    std::uint64_t scale_offset = 0;
};

HostWeight checkpoint_like(int n, int k, std::uint32_t seed) {
    std::uint32_t state = 0x9E3779B9U ^ seed;
    const auto next     = [&state] { return state = state * 1664525U + 1013904223U; };
    const int groups    = k / 32;
    HostWeight host;
    host.scale_offset = (static_cast<std::uint64_t>(n) * k + 255) / 256 * 256;
    host.payload.assign(host.scale_offset + static_cast<std::uint64_t>(n) * groups, 0);
    for (std::size_t i = 0; i < static_cast<std::size_t>(n) * k; ++i) {
        std::uint8_t code = static_cast<std::uint8_t>(next() >> 24);
        if ((code & 0x7FU) == 0x7FU) code ^= 1U; // not NaN
        host.payload[i] = code;
    }
    for (std::size_t i = 0; i < static_cast<std::size_t>(n) * groups; ++i) {
        host.payload[host.scale_offset + i] = static_cast<std::uint8_t>(120U + (next() >> 8) % 15U);
    }
    return host;
}

Weight device_weight(const HostWeight& host, void* device, int n, int k) {
    const int groups = k / 32;
    Weight w{};
    w.qtype            = QType::FP8_E4M3FN_G32_E8M0;
    w.layout           = QuantLayout::G32ScaleRows16;
    w.scale_dtype      = DType::U8;
    w.payload          = device;
    w.payload_bytes    = host.payload.size();
    w.high_plane_bytes = 0;
    w.qdata            = device;
    w.qhigh            = nullptr;
    w.scales           = static_cast<std::uint8_t*>(device) + host.scale_offset;
    w.group_size       = 32;
    w.group            = 32;
    w.ndim             = 2;
    w.shape[0] = w.padded_shape[0] = n;
    w.shape[1] = w.padded_shape[1] = k;
    w.shape[2] = w.padded_shape[2] = 1;
    w.shape[3] = w.padded_shape[3] = 1;
    w.scale_ne[0] = 16;
    w.scale_ne[1] = groups;
    w.scale_ne[2] = n / 16;
    w.scale_ne[3] = 1;
    w.scale_nb[0] = 1;
    w.scale_nb[1] = 16;
    w.scale_nb[2] = static_cast<std::int64_t>(groups) * 16;
    w.scale_nb[3] = w.scale_nb[2] * (n / 16);
    w.n           = n;
    w.k           = k;
    return w;
}

// FP64 value of output (row, column) from the represented weights and activations.
double reference(const HostWeight& host, const std::vector<std::uint16_t>& x, int n, int k, int row, int column) {
    const int groups = k / 32;
    double acc       = 0;
    for (int j = 0; j < k; ++j) {
        const std::uint8_t word =
            host.payload[host.scale_offset + static_cast<std::size_t>(row / 16) * groups * 16 +
                         static_cast<std::size_t>(j / 32) * 16 + static_cast<std::size_t>(row % 16)];
        const double w = e4m3_value(host.payload[static_cast<std::size_t>(row) * k + j]) * std::ldexp(1.0, word - 127);
        acc += w * bf16_value(x[static_cast<std::size_t>(column) * k + j]);
    }
    (void)n;
    return acc;
}

bench::ColdTiming time_route(const std::function<void(cudaStream_t)>& body, bench::L2FlushBuffer& flush,
                             cudaStream_t stream, const Options& options) {
    bench::TimedGraph graph;
    graph.capture(stream, body);
    return bench::measure_cold_graph(graph, flush, stream, options.warmup, options.repeat);
}

struct Interval {
    int last_t;
    std::size_t candidate;
};

std::vector<Interval> select_intervals(const std::vector<int>& ts, const std::vector<std::vector<double>>& us,
                                       double tolerance) {
    const auto best_at = [&](std::size_t i) {
        std::size_t best = SIZE_MAX;
        for (std::size_t c = 0; c < us[i].size(); ++c) {
            if (std::isnan(us[i][c])) continue;
            if (best == SIZE_MAX || us[i][c] < us[i][best]) best = c;
        }
        if (best == SIZE_MAX) throw std::runtime_error("no usable candidate at T=" + std::to_string(ts[i]));
        return best;
    };
    const auto within = [&](std::size_t c, std::size_t first, std::size_t last) {
        for (std::size_t i = first; i <= last; ++i) {
            if (std::isnan(us[i][c]) || us[i][c] > (1.0 + tolerance) * us[i][best_at(i)]) return false;
        }
        return true;
    };
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
        for (std::size_t s = 0; s + 1 < spans.size() && !merged; ++s) {
            const Span a = spans[s], b = spans[s + 1];
            for (const std::size_t c : {b.candidate, a.candidate}) {
                if (within(c, a.first, b.last)) {
                    spans[s] = {a.first, b.last, c};
                    spans.erase(spans.begin() + static_cast<std::ptrdiff_t>(s) + 1);
                    merged = true;
                    break;
                }
            }
        }
    }
    std::vector<Interval> result;
    for (std::size_t s = 0; s < spans.size(); ++s) {
        result.push_back({s + 1 == spans.size() ? INT_MAX : ts[spans[s].last], spans[s].candidate});
    }
    return result;
}

void sweep_shape(const Shape& shape, const std::vector<Candidate>& list, const Options& options, std::FILE* csv,
                 bench::L2FlushBuffer& flush, cudaStream_t stream) {
    constexpr int kCheckT = 300;
    const int max_t       = std::max(options.ts.back(), kCheckT);
    const HostWeight host = checkpoint_like(shape.n, shape.k, static_cast<std::uint32_t>(shape.n) * 31U + shape.k);
    DeviceBuffer storage(host.payload.size());
    CUDA_CHECK(cudaMemcpy(storage.p, host.payload.data(), host.payload.size(), cudaMemcpyHostToDevice));
    const Weight weight = device_weight(host, storage.p, shape.n, shape.k);
    std::vector<std::uint16_t> x_host(static_cast<std::size_t>(shape.k) * max_t);
    std::uint32_t state = 0x2545F491U ^ static_cast<std::uint32_t>(shape.k);
    for (auto& bits : x_host) {
        state           = state * 1664525U + 1013904223U;
        const std::uint32_t r = state;
        bits = static_cast<std::uint16_t>(((r >> 31) << 15) | ((107U + (r >> 8) % 25U) << 7) | (r & 0x7FU));
    }
    DeviceBuffer x(x_host.size() * 2);
    CUDA_CHECK(cudaMemcpy(x.p, x_host.data(), x_host.size() * 2, cudaMemcpyHostToDevice));
    DeviceBuffer out(static_cast<std::size_t>(shape.n) * max_t * 2);
    const double bytes    = static_cast<double>(shape.n) * shape.k * (1.0 + 1.0 / 32.0);
    const double floor_us = bytes / (options.dram_gbps * 1e3);

    // ---------------------------------------------------------------- one column order
    const auto run = [&](const Candidate& c, int t) {
        Tensor xt(x.p, DType::BF16, {shape.k, t});
        Tensor ot(out.p, DType::BF16, {shape.n, t});
        c.launch(xt, weight, ot, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        return download<std::uint16_t>(out.p, static_cast<std::size_t>(shape.n) * t);
    };
    std::vector<bool> usable(list.size(), false);
    std::vector<std::uint16_t> one_ref, wide_ref;
    std::printf("\n## %s [%d,%d] (weight-read floor %.2f us at %.0f GB/s)\n", shape.label, shape.n, shape.k,
                floor_us, options.dram_gbps);
    for (std::size_t c = 0; c < list.size(); ++c) {
        if (shape.n % list[c].row_tile) continue;
        try {
            const auto one  = run(list[c], 1);
            const auto wide = run(list[c], kCheckT);
            if (one_ref.empty()) {
                // The reference candidate: an FP64 screen of sampled rows (columns 0 and 299).
                double error = 0, norm = 0;
                for (int i = 0; i < 64; ++i) {
                    const int row = static_cast<int>((static_cast<long long>(i) * 7919) % shape.n);
                    for (const int column : {0, kCheckT - 1}) {
                        const double r = reference(host, x_host, shape.n, shape.k, row, column);
                        const double a = bf16_value(wide[static_cast<std::size_t>(column) * shape.n + row]);
                        error += (a - r) * (a - r);
                        norm += r * r;
                    }
                }
                const double rel = norm > 0 ? std::sqrt(error / norm) : std::sqrt(error);
                std::printf("# reference %s: FP64 screen rel_l2 %.3g%s\n", list[c].name, rel,
                            rel <= 1.0 / 256.0 ? "" : " FAIL");
                if (rel > 1.0 / 256.0) continue;
                one_ref  = one;
                wide_ref = wide;
            }
            const bool one_equal  = one == one_ref;
            const bool wide_equal = wide == wide_ref;
            const bool column0    = std::equal(one.begin(), one.end(), wide.begin());
            usable[c]             = one_equal && wide_equal && column0;
            std::printf("# %-18s %s\n", list[c].name,
                        usable[c] ? "same bits" : "DIFFERENT BITS (dropped: the route must be column-invariant)");
        } catch (const std::exception& error) {
            std::printf("# %-18s rejected: %s\n", list[c].name, error.what());
        }
    }

    // ---------------------------------------------------------------- timing
    std::vector<std::vector<double>> us(options.ts.size(), std::vector<double>(list.size(), NAN));
    for (std::size_t i = 0; i < options.ts.size(); ++i) {
        const int t = options.ts[i];
        Tensor xt(x.p, DType::BF16, {shape.k, t});
        Tensor ot(out.p, DType::BF16, {shape.n, t});
        for (std::size_t c = 0; c < list.size(); ++c) {
            if (!usable[c]) continue;
            const auto timing = time_route([&](cudaStream_t s) { list[c].launch(xt, weight, ot, s); }, flush, stream,
                                           options);
            std::fprintf(csv, "%s,%d,%d,%d,%s,%.3f,%.3f,%.3f\n", shape.label, shape.n, shape.k, t, list[c].name,
                         timing.median_us, timing.min_us, timing.p95_us);
            us[i][c] = timing.median_us;
        }
        std::fflush(csv);
    }
    std::printf("%8s %10s %10s  %s\n", "T", "best_us", "floor_x", "best candidate");
    for (std::size_t i = 0; i < options.ts.size(); ++i) {
        std::size_t best = SIZE_MAX;
        for (std::size_t c = 0; c < list.size(); ++c)
            if (!std::isnan(us[i][c]) && (best == SIZE_MAX || us[i][c] < us[i][best])) best = c;
        if (best == SIZE_MAX) continue;
        std::printf("%8d %10.2f %10.2f  %s\n", options.ts[i], us[i][best], us[i][best] / floor_us, list[best].name);
    }
    std::printf("\nProposed selector (tolerance %.1f%%):\n", options.tolerance * 100.0);
    for (const Interval& interval : select_intervals(options.ts, us, options.tolerance)) {
        const Candidate& candidate = list[interval.candidate];
        if (interval.last_t != INT_MAX) {
            std::printf("    if (tokens <= %d) return mxfp8_linear_a16_tma<Geometry, %s>(x, weight, out, stream); // %s\n",
                        interval.last_t, candidate.type, candidate.name);
        } else {
            std::printf("    mxfp8_linear_a16_tma<Geometry, %s>(x, weight, out, stream); // %s\n", candidate.type,
                        candidate.name);
        }
    }
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
        std::fprintf(csv, "shape,N,K,T,route,median_us,min_us,p95_us\n");
        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        bench::L2FlushBuffer flush(256ULL << 20);
        bool matched = false;
        for (const Shape& shape : kShapes) {
            if (!options.shape.empty() && options.shape != shape.label) continue;
            matched = true;
            const auto& list = shape.k == 2560 ? candidates<2560>() : shape.k == 6144 ? candidates<6144>() : candidates<640>();
            sweep_shape(shape, list, options, csv, flush, stream);
        }
        CUDA_CHECK(cudaStreamDestroy(stream));
        if (csv != stdout) std::fclose(csv);
        if (!matched) throw std::invalid_argument("unknown --shape " + options.shape);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "mxfp8_flash_next_sweep: %s\n", error.what());
        return 1;
    }
}
