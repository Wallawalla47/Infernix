// bf16_p12 (lossless 12-bit BF16) route sweep for the Qwen3.8-Flash-Next BF16 linear problems of recipe A.
//
// For each problem it packs a BF16 matrix (a real tensor from --real DIR, bf16_<N>x<K>.bin as written by
// .local/perf1101/dump_bf16.py, else synthetic values), checks that every weight unpacks to its BF16 bits, and
// then, per T, times with cold-L2 CUDA graphs:
//   bf16        the registered BF16 route (ops::linear on the BF16 matrix): the baseline,
//   p12         the registered bf16_p12 route (ops::linear on the packed matrix),
//   candidates  packed skinny GEMV instances (T <= 16), the register-decode stream MMA (bf16_p12_stream.cuh), the
//               direct MMA (bf16_p12_direct.cuh), the register-decode TMA route (bf16_p12_rd.cuh) and the packed
//               TMA MMA (bf16_p12_tma.cuh) on each schedule below, and, as a reference with no
//               production route, the matrix decoded then the registered BF16 route ("unpack").
// Each candidate is screened against the BF16 route's output (relative L2 error at most 1e-2, '#' when not) and
// checked bit for bit against its twin where it has one ('!' when it differs): a skinny instance against the
// BF16 skinny GEMV over the same columns, a stream schedule against the first stream schedule with the same K
// slices, a TMA schedule against the BF16 TMA kernel on its tile, "unpack" against the registered BF16 route.
// Per problem it proposes the fastest passing candidate at each T and its speedup over the BF16 route.
//
// Usage: infernix_bf16_p12_sweep [--real DIR] [--shape NxK] [--only NAME,...] [--t-list 1,2,...] [--warmup N]
//                                [--repeat N]
#include "core/device.h"
#include "core/weight.h"
#include "core/weight_view.h"
#include "direct_bf16_weight.cuh"
#include "infernix/ops/bf16_p12.h"
#include "infernix/ops/linear.h"
#include "infernix_bench_common.h"
#include "ops/linear/bf16/bf16_launch.cuh"
#include "ops/linear/bf16/bf16_skinny.cuh"
#include "ops/linear/bf16_p12/bf16_p12_tma.cuh"
#include "infernix/ops/projection_fp32.h"
#include "ops/linear/bf16_p12/bf16_p12_direct.cuh"
#include "ops/linear/bf16_p12/bf16_p12_dispatch.h"
#include "ops/linear/bf16_p12/bf16_p12_kernels.cuh"
#include "ops/linear/bf16_p12/bf16_p12_rd.cuh"
#include "ops/linear/bf16_p12/bf16_p12_stream.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace infernix;
using namespace infernix::ops::detail;

namespace {

struct Shape {
    std::int32_t n, k;
    const char* role;
    // A vocabulary head: FP32 logits, against projection_fp32 on the BF16 rows (the head's route).
    bool head = false;
};

constexpr Shape kShapes[] = {
    {16384, 2560, "GDN q/k/v/z projections"}, {13952, 2560, "QSA q/gate/k/v/index"},
    {2560, 6144, "GDN and QSA output projections"}, {12800, 2560, "PLE key/value"}, {96, 2560, "GDN a/b control"},
    {248320, 2560, "LM head (FP32 logits)", true}, {1280, 2560, "shared expert gate/up"},
    {2560, 640, "shared expert down"},
};

constexpr int kTs[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 16, 17, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024,
                       2048, 4096, 8192, 16384};

struct Options {
    std::string real;
    std::string shape;
    std::string only; // comma-separated candidate names; empty: all
    std::vector<int> ts{std::begin(kTs), std::end(kTs)};
    int warmup = 3;
    int repeat = 15;
};

using Launch = std::function<void(const Tensor&, const Weight& packed, const Weight& bf16, Tensor&, cudaStream_t)>;

struct Candidate {
    const char* name;
    int min_t, max_t;
    Launch run;
    // Its twin (same bits expected), or empty: none.
    Launch twin;
    std::int32_t rows_multiple = 1, k_multiple = 8;
};

// Stream schedules <row groups, tokens, row warps, K slices, ahead, stages>.
using StreamT16W8 = P12StreamSchedule<1, 16, 8, 1, 2, 3>;
using StreamT32W8 = P12StreamSchedule<1, 32, 8, 1, 2, 3>;
using StreamT64W8 = P12StreamSchedule<1, 64, 8, 1, 2, 3>;
using StreamT48W8 = P12StreamSchedule<1, 48, 8, 1, 2, 3>;
using StreamT16K4 = P12StreamSchedule<1, 16, 2, 4, 2, 3>;
using StreamT32K4 = P12StreamSchedule<1, 32, 2, 4, 2, 2>;
// Narrower CTAs (64 and 32 rows: 2-4x the CTAs of the 128-row schedules, which leave SMs idle on these shapes),
// with two or three steps of weights in flight.
using StreamT32W4   = P12StreamSchedule<1, 32, 4, 1, 2, 3, 2>;
using StreamT32W4A3 = P12StreamSchedule<1, 32, 4, 1, 3, 3, 2>;
using StreamT48W4   = P12StreamSchedule<1, 48, 4, 1, 2, 3, 2>;
using StreamT64W4   = P12StreamSchedule<1, 64, 4, 1, 2, 3, 2>;
using StreamT64W4A3 = P12StreamSchedule<1, 64, 4, 1, 3, 3, 2>;
using StreamT64W2   = P12StreamSchedule<1, 64, 2, 1, 3, 3, 2>;


// The BF16 TMA tiles the Flash-Next selectors register (bf16/shapes), and their packed twins
// <tile, decode warps, stages>.
using TileR128T128 = Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3>;
using TileR128Rows = Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3, 1, Bf16MmaRaster::RowFast>;
using TileR64T128  = Bf16A16TmaMmaSchedule<64, 128, 64, 32, 32, 2>;
using TileR64T64   = Bf16A16TmaMmaSchedule<64, 64, 64, 32, 32, 3>;
using TileR64T32   = Bf16A16TmaMmaSchedule<64, 32, 64, 32, 16, 3>;

template <class Tile, int DecodeWarps, int Stages, int PackedStages, bool Spin = false>
Candidate tma_candidate(const char* name, int min_t, int max_t) {
    using S = P12TmaSchedule<Tile, DecodeWarps, Stages, PackedStages, Spin>;
    return {name, min_t, max_t,
            [](const Tensor& x, const Weight& p, const Weight&, Tensor& o, cudaStream_t s) {
                launch_bf16_p12_tma<S>(x, p, o, s);
            },
            [](const Tensor& x, const Weight&, const Weight& w, Tensor& o, cudaStream_t s) {
                launch_bf16_tma_mma<Tile>(x, w, o, s);
            },
            Tile::kBlockRows, Tile::kBlockK};
}

template <class S>
Candidate stream_candidate(const char* name, int max_t, Launch twin, int min_t = 1) {
    return {name, min_t, max_t,
            [](const Tensor& x, const Weight& p, const Weight&, Tensor& o, cudaStream_t s) {
                launch_bf16_p12_stream<S>(x, p, o, s);
            },
            std::move(twin), 16, S::kStepK};
}

template <class S>
Launch stream_twin() {
    return [](const Tensor& x, const Weight& p, const Weight&, Tensor& o, cudaStream_t s) {
        launch_bf16_p12_stream<S>(x, p, o, s);
    };
}

// Direct schedules <row groups, tiles, row warps, K slices, ahead>.
using DirectG1T1 = P12DirectSchedule<1, 1, 8, 1, 3>;
using DirectG1T2 = P12DirectSchedule<1, 2, 8, 1, 3>;
using DirectG2T2 = P12DirectSchedule<2, 2, 4, 1, 2>;
using DirectG1T4 = P12DirectSchedule<1, 4, 8, 1, 2>;
using DirectG1T8 = P12DirectSchedule<1, 8, 4, 1, 2>;
using DirectT1K4 = P12DirectSchedule<1, 1, 2, 4, 3>;
using DirectT2K4 = P12DirectSchedule<1, 2, 2, 4, 3>;
using DirectT4K4 = P12DirectSchedule<1, 4, 2, 4, 2>;
using DirectT8K4 = P12DirectSchedule<1, 8, 2, 4, 2>;
using DirectT1K8  = P12DirectSchedule<1, 1, 1, 8, 3>;
using DirectT2K8  = P12DirectSchedule<1, 2, 1, 8, 3>;
using DirectT4K8  = P12DirectSchedule<1, 4, 1, 8, 2>;
using DirectT1K16 = P12DirectSchedule<1, 1, 1, 16, 2>;
using DirectT2K16 = P12DirectSchedule<1, 2, 1, 16, 2>;
using DirectT1K12 = P12DirectSchedule<1, 1, 1, 12, 2>;

template <class S>
Launch direct_launch() {
    return [](const Tensor& x, const Weight& p, const Weight&, Tensor& o, cudaStream_t s) {
        launch_bf16_p12_direct<S>(x, p, o, s);
    };
}

template <class S>
Candidate direct_candidate(const char* name, Launch twin) {
    return {name, 1, S::kMaxTokens, direct_launch<S>(), std::move(twin), 16, 128 * S::kKSlices};
}

// Register-decode TMA schedules <row warps, row groups, tokens, stages, CTAs an SM>; one set of bits for all of them.
using RdW1T64  = P12RdSchedule<1, 1, 64, 4, 2>;
using RdW1T128 = P12RdSchedule<1, 1, 128, 3, 1>;
using RdW4T16  = P12RdSchedule<4, 1, 16, 4, 2>;
using RdW4T32  = P12RdSchedule<4, 1, 32, 4, 2>;
using RdW2T64  = P12RdSchedule<2, 1, 64, 4, 2>;
using RdW4T64  = P12RdSchedule<4, 1, 64, 3, 2>;
using RdW8T64  = P12RdSchedule<8, 1, 64, 3, 1>;
using RdW4R2T64 = P12RdSchedule<4, 2, 64, 3, 1>;
using RdW2T128 = P12RdSchedule<2, 1, 128, 3, 1>;
using RdW4T128 = P12RdSchedule<4, 1, 128, 3, 1>;

template <class S>
Launch rd_launch() {
    return [](const Tensor& x, const Weight& p, const Weight&, Tensor& o, cudaStream_t s) {
        launch_bf16_p12_rd<S>(x, p, o, s);
    };
}

template <class S>
Candidate rd_candidate(const char* name, int min_t, int max_t, Launch twin) {
    return {name, min_t, max_t, rd_launch<S>(), std::move(twin), S::kBlockRows, S::kStepK};
}

template <int Warps, int RowsPerWarp, int Ahead>
Launch skinny_instance() {
    return [](const Tensor& x, const Weight& p, const Weight&, Tensor& o, cudaStream_t s) {
        bf16_p12_skinny_launch<Warps, RowsPerWarp, Ahead>(x.ne[1])(x, p, o, s);
    };
}

// The BF16 skinny GEMV over the same columns, eight at a time: its bits per column depend only on K, so this
// is the packed skinny GEMV's twin at every width.
void bf16_skinny_twin(const Tensor& x, const Weight&, const Weight& w, Tensor& out, cudaStream_t stream) {
    const std::int32_t k = x.ne[0], n = out.ne[0], t = x.ne[1];
    for (std::int32_t first = 0; first < t; first += 8) {
        const std::int32_t width = std::min(8, t - first);
        const std::size_t bytes  = out.dtype == DType::FP32 ? 4 : 2;
        Tensor xs(static_cast<__nv_bfloat16*>(x.data) + static_cast<std::int64_t>(first) * k, DType::BF16, {k, width});
        Tensor os(static_cast<std::byte*>(out.data) + static_cast<std::int64_t>(first) * n * bytes, out.dtype,
                  {n, width});
        bf16_skinny_launch<8, 2, 0>(width)(xs, w, os, stream);
    }
}

std::vector<Candidate> candidates(WorkspaceArena& work, const Weight& decoded) {
    std::vector<Candidate> list;
    list.push_back({"sk_w8r2a1", 1, 16, skinny_instance<8, 2, 1>(), bf16_skinny_twin});
    list.push_back({"sk_w4r1a4", 1, 16, skinny_instance<4, 1, 4>(), bf16_skinny_twin});
    list.push_back({"sk_w2r1a2", 1, 16, skinny_instance<2, 1, 2>(), bf16_skinny_twin});
    const Launch s1 = stream_twin<StreamT16W8>(), s4 = stream_twin<StreamT16K4>();
    list.push_back(stream_candidate<StreamT16W8>("st_t16w8", 256, {}));
    list.push_back(stream_candidate<StreamT32W8>("st_t32w8", 1024, s1));
    list.push_back(stream_candidate<StreamT48W8>("st_t48w8", 4096, s1));
    list.push_back(stream_candidate<StreamT64W8>("st_t64w8", 4096, s1));
    list.push_back(stream_candidate<StreamT16K4>("st_t16k4", 256, {}));
    list.push_back(stream_candidate<StreamT32K4>("st_t32k4", 1024, s4));
    list.push_back(stream_candidate<StreamT32W4>("st_t32w4", 1024, s1));
    list.push_back(stream_candidate<StreamT32W4A3>("st_t32w4a3", 1024, s1));
    list.push_back(stream_candidate<StreamT48W4>("st_t48w4", 4096, s1));
    list.push_back(stream_candidate<StreamT64W4>("st_t64w4", 4096, s1));
    list.push_back(stream_candidate<StreamT64W4A3>("st_t64w4a3", 4096, s1));
    list.push_back(stream_candidate<StreamT64W2>("st_t64w2", 4096, s1));
    // Twins: the widest schedule of each K-slice count (same bits at every width it takes).
    const Launch d1 = direct_launch<DirectG1T8>(), d4 = direct_launch<DirectT8K4>(), d8 = direct_launch<DirectT4K8>(),
                 d16 = direct_launch<DirectT2K16>();
    list.push_back(direct_candidate<DirectG1T1>("dr_g1t1", d1));
    list.push_back(direct_candidate<DirectG1T2>("dr_g1t2", d1));
    list.push_back(direct_candidate<DirectG2T2>("dr_g2t2", d1));
    list.push_back(direct_candidate<DirectG1T4>("dr_g1t4", d1));
    list.push_back(direct_candidate<DirectG1T8>("dr_g1t8", {}));
    list.push_back(direct_candidate<DirectT1K4>("dr_t1k4", d4));
    list.push_back(direct_candidate<DirectT2K4>("dr_t2k4", d4));
    list.push_back(direct_candidate<DirectT4K4>("dr_t4k4", d4));
    list.push_back(direct_candidate<DirectT8K4>("dr_t8k4", {}));
    list.push_back(direct_candidate<DirectT1K8>("dr_t1k8", d8));
    list.push_back(direct_candidate<DirectT2K8>("dr_t2k8", d8));
    list.push_back(direct_candidate<DirectT4K8>("dr_t4k8", {}));
    list.push_back(direct_candidate<DirectT1K12>("dr_t1k12", {}));
    list.push_back(direct_candidate<DirectT1K16>("dr_t1k16", d16));
    list.push_back(direct_candidate<DirectT2K16>("dr_t2k16", {}));
    // The register-decode TMA route; twin: the 16-row schedule (it divides every shape here).
    const Launch rd = rd_launch<RdW1T64>();
    list.push_back(rd_candidate<RdW1T64>("rd_w1t64", 9, 1024, {}));
    list.push_back(rd_candidate<RdW1T128>("rd_w1t128", 33, 4096, rd));
    list.push_back(rd_candidate<RdW4T16>("rd_w4t16", 9, 64, rd));
    list.push_back(rd_candidate<RdW4T32>("rd_w4t32", 9, 256, rd));
    list.push_back(rd_candidate<RdW2T64>("rd_w2t64", 9, 1024, rd));
    list.push_back(rd_candidate<RdW4T64>("rd_w4t64", 9, 1024, rd));
    list.push_back(rd_candidate<RdW8T64>("rd_w8t64", 9, 1024, rd));
    list.push_back(rd_candidate<RdW4R2T64>("rd_w4r2t64", 9, 1024, rd));
    list.push_back(rd_candidate<RdW2T128>("rd_w2t128", 33, 4096, rd));
    list.push_back(rd_candidate<RdW4T128>("rd_w4t128", 33, 4096, rd));
    list.push_back(tma_candidate<TileR128T128, 4, 2, 2>("tm_128x128", 65, 1 << 30));
    list.push_back(tma_candidate<TileR128Rows, 4, 2, 2>("tm_128rows", 65, 1 << 30));
    list.push_back(tma_candidate<TileR64T128, 4, 2, 3>("tm_64x128", 17, 1 << 30));
    list.push_back(tma_candidate<TileR64T64, 4, 3, 3>("tm_64x64", 17, 1 << 30));
    list.push_back(tma_candidate<TileR64T32, 4, 3, 3>("tm_64x32", 17, 1024));
    // Reference (no production route): decode the matrix, then the BF16 route on it.
    list.push_back({"unpack", 17, 1 << 30,
                    [&work, &decoded](const Tensor& x, const Weight& p, const Weight&, Tensor& o, cudaStream_t s) {
                        bf16_p12_unpack(p, const_cast<void*>(decoded.qdata), s);
                        ops::linear(x, decoded, o, ops::LinearPolicy::A16Only, work, s);
                    },
                    [&work](const Tensor& x, const Weight&, const Weight& w, Tensor& o, cudaStream_t s) {
                        ops::linear(x, w, o, ops::LinearPolicy::A16Only, work, s);
                    }});
    return list;
}

double bf16_value(std::uint16_t bits) {
    const std::uint32_t word = static_cast<std::uint32_t>(bits) << 16;
    float value;
    std::memcpy(&value, &word, sizeof(value));
    return value;
}

std::vector<std::uint16_t> download(const void* p, std::size_t elements) {
    std::vector<std::uint16_t> host(elements);
    CUDA_CHECK(cudaMemcpy(host.data(), p, elements * 2, cudaMemcpyDeviceToHost));
    return host;
}

bench::ColdTiming time_route(const std::function<void(cudaStream_t)>& body, bench::L2FlushBuffer& flush,
                             cudaStream_t stream, const Options& options) {
    bench::TimedGraph graph;
    graph.capture(stream, body);
    return bench::measure_cold_graph(graph, flush, stream, options.warmup, options.repeat);
}

// Real recipe A values when a dump exists; otherwise BF16 draws of N(0, 0.02) with one weight in 2,000 a
// hundred times larger (outliers that land outside a row's window).
std::vector<std::uint16_t> weight_values(const Shape& shape, const Options& options) {
    const std::size_t n = static_cast<std::size_t>(shape.n) * shape.k;
    std::vector<std::uint16_t> host(n);
    if (!options.real.empty()) {
        std::ifstream f(options.real + "/bf16_" + std::to_string(shape.n) + "x" + std::to_string(shape.k) + ".bin",
                        std::ios::binary);
        if (f && f.read(reinterpret_cast<char*>(host.data()), static_cast<std::streamsize>(n * 2))) {
            std::printf("  values: real recipe A tensor\n");
            return host;
        }
    }
    std::mt19937 rng(0x5eed);
    std::normal_distribution<float> normal(0.0F, 0.02F);
    std::uniform_int_distribution<int> outlier(0, 1999);
    for (auto& v : host) {
        float x = normal(rng);
        if (outlier(rng) == 0) { x *= 100.0F; }
        const __nv_bfloat16 b = __float2bfloat16(x);
        std::memcpy(&v, &b, 2);
    }
    std::printf("  values: synthetic N(0, 0.02) with outliers\n");
    return host;
}

void sweep(const Shape& shape, const Options& options, bench::L2FlushBuffer& flush, cudaStream_t stream) {
    std::printf("\n## [%d, %d] %s\n", shape.n, shape.k, shape.role);
    const std::size_t elements = static_cast<std::size_t>(shape.n) * shape.k;
    const auto values          = weight_values(shape, options);
    auto bf16                  = bench::make_direct_bf16_weight(shape.n, shape.k);
    // On the bench stream and synchronized: a pageable cudaMemcpy can return before its DMA lands, and the
    // non-blocking stream would not wait for it (the count then read a half-written matrix).
    CUDA_CHECK(cudaMemcpyAsync(bf16.storage.p, values.data(), elements * 2, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Pack (the loader's two passes): wall time of each pass, first in the process (it loads the kernels) and
    // again.
    DeviceBuffer scratch(ops::bf16_p12_pack_scratch_bytes(shape.n)), count_word(4);
    const auto count = [&] {
        ops::bf16_p12_count(bf16.storage.p, shape.n, shape.k, scratch.p, static_cast<std::uint32_t*>(count_word.p),
                            stream);
        std::uint32_t value = 0;
        CUDA_CHECK(cudaMemcpyAsync(&value, count_word.p, 4, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        return std::uint64_t{value};
    };
    const std::uint64_t escapes = count();
    const WeightGeometry g      = bf16_p12_geometry(shape.n, shape.k, escapes);
    DeviceBuffer packed_storage(g.bytes);
    double count_ms[2] = {}, pack_ms[2] = {};
    for (int pass = 0; pass < 2; ++pass) {
        const auto t0 = std::chrono::steady_clock::now();
        if (count() != escapes) { throw std::runtime_error("bf16_p12 count is not repeatable"); }
        const auto t1 = std::chrono::steady_clock::now();
        ops::bf16_p12_pack(bf16.storage.p, shape.n, shape.k, escapes, scratch.p, packed_storage.p, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        const auto t2  = std::chrono::steady_clock::now();
        count_ms[pass] = std::chrono::duration<double, std::milli>(t1 - t0).count();
        pack_ms[pass]  = std::chrono::duration<double, std::milli>(t2 - t1).count();
    }
    Weight packed;
    auto* base     = static_cast<std::byte*>(packed_storage.p);
    packed.payload = base, packed.payload_bytes = g.bytes, packed.high_plane_bytes = g.high_bytes;
    packed.qtype = QType::BF16_P12, packed.layout = QuantLayout::P12Rows, packed.ndim = 2;
    packed.n = packed.shape[0] = packed.padded_shape[0] = shape.n;
    packed.k = packed.shape[1] = packed.padded_shape[1] = shape.k;
    packed.qdata = base, packed.qhigh = base + g.high_offset, packed.scales = base + g.scale_offset;
    packed.escapes = base + g.escape_offset;
    std::printf("  packed %.2f MB -> %.2f MB (%.3f bits a weight), %llu escapes (%.0f per million)\n"
                "  count + pack: %.3f + %.3f ms, again %.3f + %.3f ms\n",
                elements * 2 / 1e6, g.bytes / 1e6, g.bytes * 8.0 / elements, static_cast<unsigned long long>(escapes),
                escapes * 1e6 / elements, count_ms[0], pack_ms[0], count_ms[1], pack_ms[1]);
    auto decoded = bench::make_direct_bf16_weight(shape.n, shape.k);
    {
        bf16_p12_unpack(packed, decoded.storage.p, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        const auto back = download(decoded.storage.p, elements);
        std::size_t wrong = 0;
        for (std::size_t i = 0; i < elements; ++i) { wrong += back[i] != values[i]; }
        std::printf("  unpack: %zu of %zu weights differ\n", wrong, elements);
        if (wrong) { throw std::runtime_error("bf16_p12 unpack is not exact"); }
    }

    std::vector<int> ts;
    for (const int t : options.ts) { ts.push_back(t); }
    const int max_t = *std::max_element(ts.begin(), ts.end());
    DeviceBuffer x = bench::make_bf16(static_cast<std::size_t>(shape.k) * max_t, 101U);
    // A head's logits are FP32 (projection_fp32 on the BF16 rows is its baseline); the projections' BF16.
    const DType dtype           = shape.head ? DType::FP32 : DType::BF16;
    const std::size_t esize     = shape.head ? 4 : 2;
    const std::size_t out_bytes = static_cast<std::size_t>(shape.n) * max_t * esize;
    DeviceBuffer out(out_bytes), twin(out_bytes), reference(out_bytes);
    WorkspaceArena work(64ULL << 20);
    const auto list = candidates(work, decoded.weight);
    const Tensor head_rows(bf16.storage.p, DType::BF16, {shape.k, shape.n});
    const auto bytes_of = [&](const void* p, std::size_t outputs) {
        std::vector<std::uint8_t> host(outputs * esize);
        CUDA_CHECK(cudaMemcpy(host.data(), p, host.size(), cudaMemcpyDeviceToHost));
        return host;
    };
    const auto value_at = [&](const std::vector<std::uint8_t>& v, std::size_t i) -> double {
        if (shape.head) {
            float f;
            std::memcpy(&f, v.data() + 4 * i, 4);
            return f;
        }
        std::uint16_t b;
        std::memcpy(&b, v.data() + 2 * i, 2);
        return bf16_value(b);
    };

    std::printf("  %6s %10s %10s", "T", "bf16_us", "p12_us");
    for (const auto& c : list) { std::printf(" %12s", c.name); }
    std::printf("  best (speedup over bf16)\n");
    for (const int t : ts) {
        const std::size_t outputs = static_cast<std::size_t>(shape.n) * t;
        Tensor xt(x.p, DType::BF16, {shape.k, t});
        Tensor ot(out.p, dtype, {shape.n, t});
        Tensor tt(twin.p, dtype, {shape.n, t});
        Tensor rt(reference.p, dtype, {shape.n, t});
        const auto registered = [&](cudaStream_t s) {
            if (shape.head) {
                const Tensor* rows[] = {&head_rows};
                ops::projection_fp32(xt, rows, rt, s);
            } else {
                ops::linear(xt, bf16.weight, rt, ops::LinearPolicy::A16Only, work, s);
            }
        };
        const double bf16_us = time_route(registered, flush, stream, options).median_us;
        const double p12_us =
            time_route([&](cudaStream_t s) { ops::linear(xt, packed, ot, ops::LinearPolicy::A16Only, work, s); }, flush,
                       stream, options)
                .median_us;
        registered(stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        const auto expected = bytes_of(reference.p, outputs);
        std::printf("  %6d %10.2f %10.2f", t, bf16_us, p12_us);
        double best_us = std::numeric_limits<double>::infinity();
        const char* best = "-";
        for (const auto& c : list) {
            const bool listed = options.only.empty() || ("," + options.only + ",").find("," + std::string(c.name) + ",") !=
                                                            std::string::npos;
            if (!listed || t < c.min_t || t > c.max_t || shape.n % c.rows_multiple != 0 ||
                shape.k % c.k_multiple != 0) {
                std::printf(" %12s", "-");
                continue;
            }
            bool exact = true, close = true;
            try {
                c.run(xt, packed, bf16.weight, ot, stream);
                if (c.twin) { c.twin(xt, packed, bf16.weight, tt, stream); }
                CUDA_CHECK(cudaStreamSynchronize(stream));
                const auto got = bytes_of(out.p, outputs);
                if (c.twin) { exact = got == bytes_of(twin.p, outputs); }
                double err = 0, norm = 0;
                for (std::size_t i = 0; i < outputs; ++i) {
                    const double a = value_at(got, i), b = value_at(expected, i);
                    err += (a - b) * (a - b);
                    norm += b * b;
                }
                close = std::isfinite(err) && err <= 1e-4 * norm;
            } catch (const std::exception& error) {
                std::printf(" %12s", "error");
                std::fprintf(stderr, "%s T=%d: %s\n", c.name, t, error.what());
                continue;
            }
            const double us = time_route([&](cudaStream_t s) { c.run(xt, packed, bf16.weight, ot, s); }, flush, stream,
                                         options)
                                  .median_us;
            std::printf(" %11.2f%s", us, !close ? "#" : exact ? " " : "!");
            if (close && exact && us < best_us) {
                best_us = us;
                best    = c.name;
            }
        }
        std::printf("  %s %.3fx\n", best, bf16_us / best_us);
        std::fflush(stdout);
    }
}

} // namespace

int main(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const auto value      = [&]() -> std::string {
            if (i + 1 >= argc) { throw std::invalid_argument(arg + " needs a value"); }
            return argv[++i];
        };
        if (arg == "--real") {
            options.real = value();
        } else if (arg == "--shape") {
            options.shape = value();
        } else if (arg == "--only") {
            options.only = value();
        } else if (arg == "--warmup") {
            options.warmup = std::stoi(value());
        } else if (arg == "--repeat") {
            options.repeat = std::stoi(value());
        } else if (arg == "--t-list") {
            options.ts.clear();
            std::string list = value();
            for (std::size_t at = 0; at < list.size();) {
                const std::size_t comma = list.find(',', at);
                options.ts.push_back(std::stoi(list.substr(at, comma - at)));
                at = comma == std::string::npos ? list.size() : comma + 1;
            }
        } else {
            std::fprintf(stderr,
                         "usage: %s [--real DIR] [--shape NxK] [--only NAME,...] [--t-list ...] [--warmup N] [--repeat N]\n",
                         argv[0]);
            return 2;
        }
    }
    try {
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        bench::L2FlushBuffer flush(256ULL << 20);
        std::printf("bf16_p12 sweep: cold-L2 graph medians (us); '!' marks a candidate whose bits differ from its twin\n");
        for (const Shape& shape : kShapes) {
            const std::string label = std::to_string(shape.n) + "x" + std::to_string(shape.k);
            if (!options.shape.empty() && options.shape != label) { continue; }
            sweep(shape, options, flush, stream);
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
    return 0;
}
