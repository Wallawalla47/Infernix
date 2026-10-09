// Tile sweep for the FP8 row-scaled W8A16 linear problems of Qwen3.8-Flash-Next artifacts whose dense
// projections are stored in FP8 (recipe C, docs/maintainer/qwen3_8-flash-next-design.md §6.1): the GDN
// query/key/value/z and output projections, the QSA query/gate/key/value and output projections and the
// shared experts. The routes are A16 (BF16 activations), as the weight-only checkpoint is served.
//
// Decode widths (T = 1..8): every candidate (SIMT of capacity 8, the one-column GEMV, sliced-K and the
// tensor-core tiles), each checked for column invariance (column 0's bits at every T equal its bits at
// T = 8, which speculative verification relies on, §11.3) and grouped by the bits it produces, so a
// selector only mixes candidates of one family.
// Verification and prefill widths (T >= 9): sliced-K, MMA and TMA MMA candidates, each first screened
// against an FP64 host reference on sampled rows at T = 77 and 1000, then timed with cold-L2 CUDA
// graphs; a proposed selector keeps the fastest candidate per measured T, merged into intervals within
// --tolerance-pct.
//
// Usage:
//   infernix_fp8_flash_next_sweep [--shape LABEL] [--t-list 9,16,...] [--warmup N] [--repeat N]
//       [--tolerance-pct P] [--dram-gbps G] [--decode-only] [--csv PATH]
#include "core/device.h"
#include "core/weight.h"
#include "infernix_bench_common.h"
#include "ops/linear/fp8/fp8_a16_tma_mma.cuh"
#include "ops/linear/fp8/fp8_launch.cuh"
#include "quantized_weight.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
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
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

using namespace infernix;
using namespace infernix::ops::detail;

namespace {

using Fp8Launch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

enum class Kind { Gemv, Simt, Sliced, Mma, Tma };

struct Candidate {
    const char* name;
    const char* type; // C++ schedule type, pasted into the proposed selector
    Kind kind;
    Fp8Launch launch;
    int max_t;     // widest supported call
    int row_tile;  // rows must be a multiple (0: any)
};

const char* launcher(Kind kind) {
    switch (kind) {
    case Kind::Gemv: return "fp8_linear_a16_gemv<Geometry, %s>";
    case Kind::Simt: return "fp8_linear_a16_simt<Geometry, 8, %s>";
    case Kind::Sliced: return "fp8_linear_a16_sliced_k<Geometry, %s>";
    case Kind::Mma: return "fp8_linear_a16_mma<Geometry, %s>";
    case Kind::Tma: return "fp8_linear_a16_tma<Geometry, %s>";
    }
    return "?";
}

template <int K>
using G = Fp8Geometry<16, K>; // only the static K enters the launch

using SimtW4R2V16T4 = Fp8A16SimtSchedule<4, 2, 16, 4, 1, Fp8SimtActivationAccess::TokenPacked, Fp8CodeCache::Default, 1,
                                         Fp8SimtBlockOrder::RowsContiguous, 1>;
using SimtW4R2V16T8 = Fp8A16SimtSchedule<4, 2, 16, 8, 1, Fp8SimtActivationAccess::TokenPacked, Fp8CodeCache::Default, 1,
                                         Fp8SimtBlockOrder::RowsContiguous, 1>;
using SimtW8R1V16T8 = Fp8A16SimtSchedule<8, 1, 16, 8, 1, Fp8SimtActivationAccess::TokenPacked, Fp8CodeCache::Default, 1,
                                         Fp8SimtBlockOrder::RowsContiguous, 1>;
using SimtW4R4V16T8 = Fp8A16SimtSchedule<4, 4, 16, 8, 1, Fp8SimtActivationAccess::TokenPacked, Fp8CodeCache::Default, 1,
                                         Fp8SimtBlockOrder::RowsContiguous, 1>;
using SimtW8R2V16T8 = Fp8A16SimtSchedule<8, 2, 16, 8, 1, Fp8SimtActivationAccess::TokenPacked, Fp8CodeCache::Default, 1,
                                         Fp8SimtBlockOrder::RowsContiguous, 1>;
using SimtW4R1V16T8 = Fp8A16SimtSchedule<4, 1, 16, 8, 1, Fp8SimtActivationAccess::TokenPacked, Fp8CodeCache::Default, 1,
                                         Fp8SimtBlockOrder::RowsContiguous, 2>;
using SimtW4R2V32T8 = Fp8A16SimtSchedule<4, 2, 32, 8, 1, Fp8SimtActivationAccess::TokenPacked, Fp8CodeCache::Default, 1,
                                         Fp8SimtBlockOrder::RowsContiguous, 1>;
using SimtW4R2V16T8C2 = Fp8A16SimtSchedule<4, 2, 16, 8, 2, Fp8SimtActivationAccess::TokenPacked, Fp8CodeCache::Default,
                                           1, Fp8SimtBlockOrder::RowsContiguous, 1>;
using SimtW4R2V16T8Sh = Fp8A16SimtSchedule<4, 2, 16, 8, 1, Fp8SimtActivationAccess::SharedPhase, Fp8CodeCache::Default,
                                           1, Fp8SimtBlockOrder::RowsContiguous, 1>;
using SimtW4R2V16T8U2 = Fp8A16SimtSchedule<4, 2, 16, 8, 1, Fp8SimtActivationAccess::TokenPacked, Fp8CodeCache::Default,
                                           2, Fp8SimtBlockOrder::RowsContiguous, 1>;
using SimtW4R2V16T8St = Fp8A16SimtSchedule<4, 2, 16, 8, 1, Fp8SimtActivationAccess::TokenPacked,
                                           Fp8CodeCache::Streaming, 1, Fp8SimtBlockOrder::RowsContiguous, 1>;
// A GEMV and a SIMT tile give one column the same bits only with equal values per lane and
// accumulator chains (the reduction order); the families below pair them.
template <int W, int R, int V, int C, Fp8CodeCache Cache = Fp8CodeCache::Default, int Blocks = 1>
using SimtT8 = Fp8A16SimtSchedule<W, R, V, 8, C, Fp8SimtActivationAccess::TokenPacked, Cache, 1,
                                  Fp8SimtBlockOrder::RowsContiguous, Blocks>;
using GemvW8R2V8C4U2  = Fp8A16GemvSchedule<8, 2, 8, 4, Fp8CodeCache::Default, 2, 2>;
using GemvW4R2V8C4    = Fp8A16GemvSchedule<4, 2, 8, 4, Fp8CodeCache::Default, 1, 2>;
using GemvW8R1V8C4    = Fp8A16GemvSchedule<8, 1, 8, 4, Fp8CodeCache::Default, 1, 2>;
using GemvW8R2V16C2U2 = Fp8A16GemvSchedule<8, 2, 16, 2, Fp8CodeCache::Default, 2, 2>;
using GemvW4R2V16C2   = Fp8A16GemvSchedule<4, 2, 16, 2, Fp8CodeCache::Default, 1, 2>;
using GemvW8R1V16C1   = Fp8A16GemvSchedule<8, 1, 16, 1, Fp8CodeCache::Default, 1, 2>;
using GemvW8R2V16C1U2 = Fp8A16GemvSchedule<8, 2, 16, 1, Fp8CodeCache::Default, 2, 2>;
using GemvW8R1V16C4   = Fp8A16GemvSchedule<8, 1, 16, 4, Fp8CodeCache::Default, 1, 2>;
using GemvW4R4V16C4   = Fp8A16GemvSchedule<4, 4, 16, 4, Fp8CodeCache::Default, 1, 1>;
using GemvW8R1V16C2   = Fp8A16GemvSchedule<8, 1, 16, 2, Fp8CodeCache::Default, 1, 2>;
using MmaR32T64K128   = Fp8A16MmaSchedule<32, 64, 128, 32, 16, 2, 2>;
using MmaR32T64K128W16 = Fp8A16MmaSchedule<32, 64, 128, 16, 16, 1, 3>;
using MmaR64T64K128   = Fp8A16MmaSchedule<64, 64, 128, 32, 16, 2, 2>;
using MmaR64T96K128   = Fp8A16MmaSchedule<64, 96, 128, 64, 16, 1, 2>;
using MmaR64T128K64   = Fp8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>;
using MmaR128T128K64  = Fp8A16MmaSchedule<128, 128, 64, 64, 32, 2, 1>;
using TmaT64R256      = Fp8A16TmaMmaSchedule<64, 256, 2, 8, 2, 1>;
using TmaT128R128     = Fp8A16TmaMmaSchedule<128, 128, 4, 4, 2, 1>;

#define INFERNIX_C(kind, name, max_t, rows, ...) \
    Candidate { name, #__VA_ARGS__, kind, launch_of<kind, __VA_ARGS__, K>(), max_t, rows }

template <Kind kind, class Schedule, int K>
constexpr Fp8Launch launch_of() {
    if constexpr (kind == Kind::Gemv) return fp8_linear_a16_gemv<G<K>, Schedule>;
    if constexpr (kind == Kind::Simt) return fp8_linear_a16_simt<G<K>, 8, Schedule>;
    if constexpr (kind == Kind::Sliced) return fp8_linear_a16_sliced_k<G<K>, Schedule>;
    if constexpr (kind == Kind::Mma) return fp8_linear_a16_mma<G<K>, Schedule>;
    if constexpr (kind == Kind::Tma) return fp8_linear_a16_tma<G<K>, Schedule>;
    return nullptr;
}

template <int K>
const std::vector<Candidate>& candidates() {
    static const std::vector<Candidate> list{
        INFERNIX_C(Kind::Simt, "simt_w4r2v16t4", 8, 0, SimtW4R2V16T4),
        INFERNIX_C(Kind::Simt, "simt_w4r2v16t8", 8, 0, SimtW4R2V16T8),
        INFERNIX_C(Kind::Simt, "simt_w8r1v16t8", 8, 0, SimtW8R1V16T8),
        INFERNIX_C(Kind::Simt, "simt_w4r4v16t8", 8, 0, SimtW4R4V16T8),
        INFERNIX_C(Kind::Simt, "simt_w8r2v16t8", 8, 0, SimtW8R2V16T8),
        INFERNIX_C(Kind::Simt, "simt_w4r1v16t8_b2", 8, 0, SimtW4R1V16T8),
        INFERNIX_C(Kind::Simt, "simt_w4r2v32t8", 8, 0, SimtW4R2V32T8),
        INFERNIX_C(Kind::Simt, "simt_w4r2v16t8c2", 8, 0, SimtW4R2V16T8C2),
        INFERNIX_C(Kind::Simt, "simt_w4r2v16t8_shared", 8, 0, SimtW4R2V16T8Sh),
        INFERNIX_C(Kind::Simt, "simt_w4r2v16t8_u2", 8, 0, SimtW4R2V16T8U2),
        INFERNIX_C(Kind::Simt, "simt_w4r2v16t8_stream", 8, 0, SimtW4R2V16T8St),
        INFERNIX_C(Kind::Simt, "simt_w4r2v8t8c4", 8, 0, SimtT8<4, 2, 8, 4>),
        INFERNIX_C(Kind::Simt, "simt_w8r2v8t8c4", 8, 0, SimtT8<8, 2, 8, 4>),
        INFERNIX_C(Kind::Simt, "simt_w4r4v8t8c4", 8, 0, SimtT8<4, 4, 8, 4>),
        INFERNIX_C(Kind::Simt, "simt_w8r1v8t8c4", 8, 0, SimtT8<8, 1, 8, 4>),
        INFERNIX_C(Kind::Simt, "simt_w4r1v8t8c4_b2", 8, 0, SimtT8<4, 1, 8, 4, Fp8CodeCache::Default, 2>),
        INFERNIX_C(Kind::Simt, "simt_w4r2v8t8c4_stream", 8, 0, SimtT8<4, 2, 8, 4, Fp8CodeCache::Streaming>),
        INFERNIX_C(Kind::Simt, "simt_w4r2v16t8c4", 8, 0, SimtT8<4, 2, 16, 4>),
        INFERNIX_C(Kind::Simt, "simt_w8r1v16t8c4", 8, 0, SimtT8<8, 1, 16, 4>),
        INFERNIX_C(Kind::Simt, "simt_w8r2v16t8c2", 8, 0, SimtT8<8, 2, 16, 2>),
        INFERNIX_C(Kind::Simt, "simt_w8r1v16t8c2", 8, 0, SimtT8<8, 1, 16, 2>),
        INFERNIX_C(Kind::Simt, "simt_w4r1v16t8c2_b2", 8, 0, SimtT8<4, 1, 16, 2, Fp8CodeCache::Default, 2>),
        INFERNIX_C(Kind::Simt, "simt_w4r2v16t8c2_stream", 8, 0, SimtT8<4, 2, 16, 2, Fp8CodeCache::Streaming>),
        INFERNIX_C(Kind::Gemv, "gemv_w8r2v8c4u2", 1, 0, GemvW8R2V8C4U2),
        INFERNIX_C(Kind::Gemv, "gemv_w4r2v8c4", 1, 0, GemvW4R2V8C4),
        INFERNIX_C(Kind::Gemv, "gemv_w8r1v8c4", 1, 0, GemvW8R1V8C4),
        INFERNIX_C(Kind::Gemv, "gemv_w4r4v16c4", 1, 0, GemvW4R4V16C4),
        INFERNIX_C(Kind::Gemv, "gemv_w8r1v16c4", 1, 0, GemvW8R1V16C4),
        INFERNIX_C(Kind::Gemv, "gemv_w8r1v16c2", 1, 0, GemvW8R1V16C2),
        INFERNIX_C(Kind::Gemv, "gemv_w8r2v16c2u2", 1, 0, GemvW8R2V16C2U2),
        INFERNIX_C(Kind::Gemv, "gemv_w4r2v16c2", 1, 0, GemvW4R2V16C2),
        INFERNIX_C(Kind::Gemv, "gemv_w8r1v16c1", 1, 0, GemvW8R1V16C1),
        INFERNIX_C(Kind::Gemv, "gemv_w8r2v16c1u2", 1, 0, GemvW8R2V16C1U2),
        INFERNIX_C(Kind::Sliced, "sliced_t8w4s1", 128, 0, Fp8SlicedInstance<8, 4, 1>),
        INFERNIX_C(Kind::Sliced, "sliced_t16w4s1", 128, 0, Fp8SlicedInstance<16, 4, 1>),
        INFERNIX_C(Kind::Sliced, "sliced_t16w4s2", 128, 0, Fp8SlicedInstance<16, 4, 2>),
        INFERNIX_C(Kind::Sliced, "sliced_t16w8s2", 128, 0, Fp8SlicedInstance<16, 8, 2>),
        INFERNIX_C(Kind::Sliced, "sliced_t32w4s1", 128, 0, Fp8SlicedInstance<32, 4, 1>),
        INFERNIX_C(Kind::Sliced, "sliced_t32w4s2", 128, 0, Fp8SlicedInstance<32, 4, 2>),
        INFERNIX_C(Kind::Sliced, "sliced_t32w8s2", 128, 0, Fp8SlicedInstance<32, 8, 2>),
        INFERNIX_C(Kind::Sliced, "sliced_t64w4s1", 128, 0, Fp8SlicedInstance<64, 4, 1>),
        INFERNIX_C(Kind::Mma, "mma_r32t64k128", INT_MAX, 32, MmaR32T64K128),
        INFERNIX_C(Kind::Mma, "mma_r32t64k128w16", INT_MAX, 32, MmaR32T64K128W16),
        INFERNIX_C(Kind::Mma, "mma_r64t64k128", INT_MAX, 64, MmaR64T64K128),
        INFERNIX_C(Kind::Mma, "mma_r64t96k128", INT_MAX, 64, MmaR64T96K128),
        INFERNIX_C(Kind::Mma, "mma_r64t128k64", INT_MAX, 64, MmaR64T128K64),
        INFERNIX_C(Kind::Mma, "mma_r128t128k64", INT_MAX, 128, MmaR128T128K64),
        INFERNIX_C(Kind::Tma, "tma_t32r64s6", INT_MAX, 64, Fp8A16TmaT32R64),
        INFERNIX_C(Kind::Tma, "tma_t32r64s3b2", INT_MAX, 64, Fp8A16TmaT32R64S3),
        INFERNIX_C(Kind::Tma, "tma_t32r128s4", INT_MAX, 128, Fp8A16TmaT32R128),
        INFERNIX_C(Kind::Tma, "tma_t64r128s3", INT_MAX, 128, Fp8A16TmaT64R128),
        INFERNIX_C(Kind::Tma, "tma_t64r256s2", INT_MAX, 256, TmaT64R256),
        INFERNIX_C(Kind::Tma, "tma_t128r128s2", INT_MAX, 128, TmaT128R128),
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
    {"qkvg", 13312, 2560},    // QSA query, gate, key and value (the BF16 indexer projections run apart)
    {"out6144", 2560, 6144},  // GDN output and QSA output projections
    {"shgu", 1280, 2560},     // shared expert gate and up
    {"shdown", 2560, 640},    // shared expert down
};

const std::vector<int> kDefaultTs{9,   12,  16,  24,  32,   48,   64,   96,   128,  192,
                                  256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 8192};

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
        const int t = std::atoi(value.substr(start, end == std::string::npos ? std::string::npos : end - start).c_str());
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
        } else if (arg == "--dram-gbps") {
            options.dram_gbps = std::atof(next());
        } else if (arg == "--decode-only") {
            options.decode_only = true;
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

template <class T>
std::vector<T> download(const void* device, std::size_t elements) {
    std::vector<T> host(elements);
    CUDA_CHECK(cudaMemcpy(host.data(), device, elements * sizeof(T), cudaMemcpyDeviceToHost));
    return host;
}

double e4m3_value(std::uint8_t code) {
    const int s = code >> 7, e = (code >> 3) & 15, m = code & 7;
    const double v = e == 0 ? m * std::ldexp(1.0, -9) : (1.0 + m / 8.0) * std::ldexp(1.0, e - 7);
    return s ? -v : v;
}

// FP64 reference of `rows` sampled output rows at width t, and the candidate's error against it.
struct Reference {
    std::vector<int> rows;
    std::vector<double> values; // [row sample][t]
};

Reference reference(const std::vector<std::uint8_t>& codes, const std::vector<std::uint16_t>& scales,
                    const std::vector<std::uint16_t>& x, int n, int k, int t) {
    Reference r;
    for (int i = 0; i < 48; ++i) r.rows.push_back(static_cast<int>((static_cast<long long>(i) * 7919) % n));
    r.values.resize(r.rows.size() * static_cast<std::size_t>(t));
    std::vector<double> w(k);
    for (std::size_t i = 0; i < r.rows.size(); ++i) {
        const int row      = r.rows[i];
        const double scale = bf16_value(scales[row]);
        for (int j = 0; j < k; ++j) w[j] = e4m3_value(codes[static_cast<std::size_t>(row) * k + j]) * scale;
        for (int c = 0; c < t; ++c) {
            double acc = 0;
            const std::uint16_t* xc = &x[static_cast<std::size_t>(c) * k];
            for (int j = 0; j < k; ++j) acc += w[j] * bf16_value(xc[j]);
            r.values[i * t + c] = acc;
        }
    }
    return r;
}

bool screen(const Reference& ref, const std::vector<std::uint16_t>& got, int n, int t, double& rel_l2, double& rel_max) {
    double error = 0, norm = 0, worst = 0, largest = 0;
    for (std::size_t i = 0; i < ref.rows.size(); ++i) {
        for (int c = 0; c < t; ++c) {
            const double a = bf16_value(got[static_cast<std::size_t>(c) * n + ref.rows[i]]);
            const double r = ref.values[i * t + c];
            if (!std::isfinite(a)) {
                rel_l2 = rel_max = std::numeric_limits<double>::infinity();
                return false;
            }
            error += (a - r) * (a - r);
            norm += r * r;
            worst   = std::max(worst, std::fabs(a - r));
            largest = std::max(largest, std::fabs(r));
        }
    }
    rel_l2  = norm > 0 ? std::sqrt(error / norm) : std::sqrt(error);
    rel_max = largest > 0 ? worst / largest : worst;
    return rel_l2 <= 1.0 / 256.0 && rel_max <= 1.0 / 128.0;
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
            const std::size_t best = best_at(i);
            if (std::isnan(us[i][c]) || us[i][c] > (1.0 + tolerance) * us[i][best]) return false;
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
        result.push_back({s + 1 == spans.size() ? INT_MAX : ts[spans[s].last], spans[s].candidate});
    }
    return result;
}

std::uint64_t hash_bits(const std::vector<std::uint16_t>& v) {
    std::uint64_t h = 1469598103934665603ULL;
    for (std::uint16_t x : v) h = (h ^ x) * 1099511628211ULL;
    return h;
}

// Checkpoint-like operands: every finite E4M3 code and activations over 25 binades. The FP32 sums
// then round, so routes with different reduction orders give different bits and the invariance
// column and the bit families below can tell them apart (patterned codes with uniform activations
// sum exactly in any order).
void checkpoint_like(const Weight& weight, DeviceBuffer& x, int n, int k, std::size_t x_values) {
    std::uint32_t state = 0x9E3779B9U ^ static_cast<std::uint32_t>(n);
    const auto next     = [&state] { return state = state * 1664525U + 1013904223U; };
    std::vector<std::uint8_t> codes(static_cast<std::size_t>(n) * k);
    for (auto& code : codes) {
        code = static_cast<std::uint8_t>(next() >> 24);
        if ((code & 0x7FU) == 0x7FU) code ^= 1U; // not NaN
    }
    std::vector<std::uint16_t> values(x_values);
    for (auto& bits : values) {
        const std::uint32_t r = next();
        bits = static_cast<std::uint16_t>(((r >> 31) << 15) | ((107U + (r >> 8) % 25U) << 7) | (r & 0x7FU));
    }
    CUDA_CHECK(cudaMemcpy(const_cast<void*>(weight.qdata), codes.data(), codes.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(x.p, values.data(), values.size() * 2, cudaMemcpyHostToDevice));
}

void sweep_shape(const Shape& shape, const std::vector<Candidate>& list, const Options& options, std::FILE* csv,
                 bench::L2FlushBuffer& flush, cudaStream_t stream) {
    const int max_t = std::max(options.ts.back(), 1000);
    auto packed     = bench::make_fp8_weight(shape.n, shape.k, 0x51U + shape.n);
    const Weight& weight = packed.weight;
    DeviceBuffer x = bench::make_bf16(static_cast<std::size_t>(shape.k) * max_t, 101U);
    DeviceBuffer out(static_cast<std::size_t>(shape.n) * max_t * 2);
    checkpoint_like(weight, x, shape.n, shape.k, static_cast<std::size_t>(shape.k) * max_t);
    const auto codes  = download<std::uint8_t>(weight.qdata, static_cast<std::size_t>(shape.n) * shape.k);
    const auto scales = download<std::uint16_t>(weight.scales, static_cast<std::size_t>(shape.n));
    const double floor_us = (static_cast<double>(shape.n) * shape.k + 2.0 * shape.n) / (options.dram_gbps * 1e3);

    // ---------------------------------------------------------------- decode widths
    std::printf("\n## %s [%d,%d] decode widths (floor %.2f us = weight bytes at %.0f GB/s)\n", shape.label, shape.n,
                shape.k, floor_us, options.dram_gbps);
    std::printf("%-24s %-9s %-18s", "route", "invariant", "T=8 bits");
    for (int t = 1; t <= 8; ++t) std::printf(" %7s", ("T=" + std::to_string(t)).c_str());
    std::printf("\n");
    const auto x_all = download<std::uint16_t>(x.p, static_cast<std::size_t>(shape.k) * 8);
    const Reference ref8 = reference(codes, scales, x_all, shape.n, shape.k, 8);
    std::map<std::uint64_t, std::string> col0_families; // column 0 bits at T = 1 by family
    for (const Candidate& c : list) {
        // Tensor-core candidates too: shapes whose K the SIMT tiles cannot cover (the shared expert's
        // down, K = 640) need a decode route among them.
        if (c.row_tile && shape.n % c.row_tile) continue;
        const int widest = c.kind == Kind::Gemv ? 1 : 8;
        std::vector<std::uint16_t> col0;
        bool invariant = true;
        std::uint64_t family = 0;
        try {
            Tensor xw(x.p, DType::BF16, {shape.k, widest});
            Tensor ow(out.p, DType::BF16, {shape.n, widest});
            c.launch(xw, weight, ow, stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            const auto all = download<std::uint16_t>(out.p, static_cast<std::size_t>(shape.n) * widest);
            double l2 = 0, mx = 0;
            if (widest == 8 && !screen(ref8, all, shape.n, 8, l2, mx)) {
                std::printf("%-24s SCREEN FAIL rel_l2=%.3g rel_max=%.3g\n", c.name, l2, mx);
                continue;
            }
            col0.assign(all.begin(), all.begin() + shape.n);
            family = hash_bits(col0);
            for (int t = 1; t < widest; ++t) {
                Tensor xt(x.p, DType::BF16, {shape.k, t});
                Tensor ot(out.p, DType::BF16, {shape.n, t});
                c.launch(xt, weight, ot, stream);
                CUDA_CHECK(cudaStreamSynchronize(stream));
                if (download<std::uint16_t>(out.p, static_cast<std::size_t>(shape.n)) != col0) invariant = false;
            }
        } catch (const std::exception& error) {
            std::printf("%-24s rejected: %s\n", c.name, error.what());
            continue;
        }
        char bits[32];
        std::snprintf(bits, sizeof bits, "%016llx", static_cast<unsigned long long>(family));
        std::printf("%-24s %-9s %-18s", c.name, invariant ? "yes" : "NO", bits);
        for (int t = 1; t <= 8; ++t) {
            if (t > widest) {
                std::printf(" %7s", "-");
                continue;
            }
            Tensor xt(x.p, DType::BF16, {shape.k, t});
            Tensor ot(out.p, DType::BF16, {shape.n, t});
            const auto timing = time_route([&](cudaStream_t s) { c.launch(xt, weight, ot, s); }, flush, stream, options);
            std::fprintf(csv, "%s,%d,%d,%d,%s,%.3f,%.3f,%.3f\n", shape.label, shape.n, shape.k, t, c.name,
                         timing.median_us, timing.min_us, timing.p95_us);
            std::printf(" %7.2f", timing.median_us);
        }
        std::printf("\n");
        std::fflush(stdout);
    }
    if (options.decode_only) return;

    // ---------------------------------------------------------------- verification and prefill widths
    std::vector<int> screens{77, 1000};
    const auto x_screen = download<std::uint16_t>(x.p, static_cast<std::size_t>(shape.k) * 1000);
    std::vector<Reference> refs;
    for (int t : screens) {
        std::vector<std::uint16_t> xs(x_screen.begin(), x_screen.begin() + static_cast<std::size_t>(shape.k) * t);
        refs.push_back(reference(codes, scales, xs, shape.n, shape.k, t));
    }
    std::vector<bool> usable(list.size(), false);
    for (std::size_t c = 0; c < list.size(); ++c) {
        const Candidate& candidate = list[c];
        if (candidate.kind == Kind::Simt || candidate.kind == Kind::Gemv) continue;
        if (candidate.row_tile && shape.n % candidate.row_tile) continue;
        bool pass = true;
        for (std::size_t s = 0; s < screens.size(); ++s) {
            const int t = std::min(screens[s], candidate.max_t);
            if (t != screens[s]) continue;
            Tensor xt(x.p, DType::BF16, {shape.k, t});
            Tensor ot(out.p, DType::BF16, {shape.n, t});
            CUDA_CHECK(cudaMemsetAsync(out.p, 0xff, static_cast<std::size_t>(shape.n) * t * 2, stream));
            try {
                candidate.launch(xt, weight, ot, stream);
                CUDA_CHECK(cudaStreamSynchronize(stream));
            } catch (const std::exception& error) {
                std::printf("# %s %s: launch rejected at T=%d: %s\n", shape.label, candidate.name, t, error.what());
                pass = false;
                break;
            }
            double l2 = 0, mx = 0;
            if (!screen(refs[s], download<std::uint16_t>(out.p, static_cast<std::size_t>(shape.n) * t), shape.n, t, l2, mx)) {
                std::printf("# %s %s: SCREEN FAIL at T=%d rel_l2=%.3g rel_max=%.3g\n", shape.label, candidate.name, t,
                            l2, mx);
                pass = false;
            }
        }
        usable[c] = pass;
    }
    std::vector<std::vector<double>> us(options.ts.size(), std::vector<double>(list.size(), NAN));
    for (std::size_t i = 0; i < options.ts.size(); ++i) {
        const int t = options.ts[i];
        Tensor xt(x.p, DType::BF16, {shape.k, t});
        Tensor ot(out.p, DType::BF16, {shape.n, t});
        for (std::size_t c = 0; c < list.size(); ++c) {
            if (!usable[c] || t > list[c].max_t) continue;
            const auto timing = time_route([&](cudaStream_t s) { list[c].launch(xt, weight, ot, s); }, flush, stream,
                                           options);
            std::fprintf(csv, "%s,%d,%d,%d,%s,%.3f,%.3f,%.3f\n", shape.label, shape.n, shape.k, t, list[c].name,
                         timing.median_us, timing.min_us, timing.p95_us);
            us[i][c] = timing.median_us;
        }
        std::fflush(csv);
    }
    const auto intervals = select_intervals(options.ts, us, options.tolerance);
    std::printf("\n## %s [%d,%d] verification and prefill widths\n", shape.label, shape.n, shape.k);
    std::printf("%8s %10s %10s  %s\n", "T", "best_us", "TFLOPS", "best candidate");
    for (std::size_t i = 0; i < options.ts.size(); ++i) {
        std::size_t best = SIZE_MAX;
        for (std::size_t c = 0; c < list.size(); ++c)
            if (!std::isnan(us[i][c]) && (best == SIZE_MAX || us[i][c] < us[i][best])) best = c;
        if (best == SIZE_MAX) continue;
        std::printf("%8d %10.2f %10.1f  %s\n", options.ts[i], us[i][best],
                    2.0 * shape.n * shape.k * options.ts[i] / us[i][best] * 1e-6, list[best].name);
    }
    std::printf("\nProposed selector for T >= 9 (tolerance %.1f%%):\n", options.tolerance * 100.0);
    for (const Interval& interval : intervals) {
        const Candidate& candidate = list[interval.candidate];
        char call[256];
        std::snprintf(call, sizeof call, launcher(candidate.kind), candidate.type);
        if (interval.last_t != INT_MAX) {
            std::printf("    if (tokens <= %d) return %s(x, weight, out, stream); // %s\n", interval.last_t, call,
                        candidate.name);
        } else {
            std::printf("    %s(x, weight, out, stream); // %s\n", call, candidate.name);
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
        std::fprintf(stderr, "fp8_flash_next_sweep: %s\n", error.what());
        return 1;
    }
}
