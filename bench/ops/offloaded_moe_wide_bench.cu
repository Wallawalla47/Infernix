// Prefill-shape benchmark of offloaded_sparse_moe's routed experts (moe_experts) for the
// Qwen3.8-Flash-Next layer (512 experts, top-10, H = 2560, I = 640) or, with --geometry 122b, the
// Qwen3.5-122B-A10B layer (256 experts, top-8, H = 3072, I = 1024; W4A16 only), `nvfp4_expert_rg16_v1`
// records.
//
// Each point routes T columns with a fixed skewed expert popularity (a Zipf-like law, so a few
// experts are wide over several 64-column tiles and the tail stays narrow), then times one
// moe_experts call as the prefill path issues it: 64 staging slots with the double-buffered
// overlap stream. Experts with more than eight columns take the wide route; the rest the narrow
// one (W4A4: the one-column kernels, a pass per column; W4A16: those in calls of at most four
// columns, the multi-column kernels above). --cold flushes the L2 before each
// call, as decode finds a layer's experts (a small T's few experts would otherwise stay L2-resident
// across repeats). Placement:
//   frames : every record in device memory (isolates the expert kernels);
//   staged : every record in the pinned host bank, staged through the slots each call
//            (the cold-layer case, bound by the staging copy).
// The FLOP figure counts the wide entries' gate/up and down products only. --activation a16 gives
// the experts W4A16 arithmetic (no input scales, design §16.2.1); a16x2 and a16x3 run their wide calls
// on the two- and three-term A4 fast paths (design §19.3.23), at either geometry; --narrow a4r (with a
// W4A16 activation, Flash-Next geometry only) runs the experts of at most eight columns on
// expert-a4-residual-decode's two-term narrow kernels.
// Decode shapes: --tokens 1,2,4,8 --cold (one call is a layer's routed experts of one decode or MTP round).
//
//   infernix_offloaded_moe_wide_bench [--tokens 64,256,1024,4096] [--placement frames|staged|both]
//                                   [--activation a4|a16|a16x2|a16x3] [--narrow exact|a4r] [--cold] [--warmup 3]
//                                   [--repeat 10] [--seed N]
//                                   [--columns-per-expert C] [--split-scales] [--geometry flash-next|122b]
//
// --columns-per-expert C routes uniformly instead (every used expert gets exactly C columns while
// 10 T / C <= 512); --split-scales gives every expert distinct gate and up input scales, which keeps
// experts of more than eight columns off the wide route (W4A4 only).

#include "infernix/ops/offloaded_sparse_moe.h"

#include "core/device.h"
#include "infernix_bench_common.h"
#include "ops/common/canonical_math.h"
#include "ops/offloaded_sparse_moe/cpu/w4a4_expert.h"
#include "ops/offloaded_sparse_moe/cuda/wide_expert.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace infernix;
namespace moe = infernix::ops::offloaded_moe;

namespace {

constexpr int kDistinct      = 16; // distinct random records, repeated over the experts
constexpr int kStagingSlots  = 64;

// The layer under test (--geometry).
struct Geometry {
    int experts = 512;
    int top_k   = 10;
    moe::ExpertShape shape = moe::kFlashNextShape;
    [[nodiscard]] std::size_t record_bytes() const { return moe::wide::nvfp4_record_bytes(shape); }
    [[nodiscard]] double expert_flops() const {
        return 2.0 * (2.0 * shape.intermediate * shape.hidden + 1.0 * shape.hidden * shape.intermediate);
    }
};
Geometry g_geometry;

struct Options {
    std::vector<int> tokens{64, 256, 1024, 4096};
    std::string placement = "both";
    int warmup            = 3;
    int repeat            = 10;
    std::uint32_t seed    = 20261004U;
    bool a16              = false;
    moe::WideA16 wide_a16 = moe::WideA16::kExact; // a16x2 / a16x3: W4A16 experts on an A4 residual route
    moe::NarrowA16 narrow_a16 = moe::NarrowA16::kExact; // --narrow a4r: W4A16 narrow jobs on two-term A4
    bool cold             = false;
    int columns_per_expert = 0;     // > 0: uniform routing, every used expert has exactly this many columns
    bool split_scales      = false; // distinct gate and up input scales: no expert takes the wide route
    bool geometry_122b     = false;
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
        if (arg == "--tokens") {
            options.tokens = parse_list(value());
        } else if (arg == "--placement") {
            options.placement = value();
        } else if (arg == "--warmup") {
            options.warmup = std::stoi(value());
        } else if (arg == "--repeat") {
            options.repeat = std::stoi(value());
        } else if (arg == "--activation") {
            const std::string v = value();
            if (v != "a4" && v != "a16" && v != "a16x2" && v != "a16x3") {
                throw std::invalid_argument("--activation takes a4, a16, a16x2 or a16x3");
            }
            options.a16      = v != "a4";
            options.wide_a16 = v == "a16x2"   ? moe::WideA16::kA4Residual
                               : v == "a16x3" ? moe::WideA16::kA4Residual3
                                              : moe::WideA16::kExact;
        } else if (arg == "--narrow") {
            const std::string v = value();
            if (v != "exact" && v != "a4r") { throw std::invalid_argument("--narrow takes exact or a4r"); }
            options.narrow_a16 = v == "a4r" ? moe::NarrowA16::kA4Residual : moe::NarrowA16::kExact;
        } else if (arg == "--cold") {
            options.cold = true;
        } else if (arg == "--seed") {
            options.seed = static_cast<std::uint32_t>(std::stoul(value()));
        } else if (arg == "--columns-per-expert") {
            options.columns_per_expert = std::stoi(value());
            if (options.columns_per_expert < 1) { throw std::invalid_argument("--columns-per-expert must be positive"); }
        } else if (arg == "--split-scales") {
            options.split_scales = true;
        } else if (arg == "--geometry") {
            const std::string v = value();
            if (v != "flash-next" && v != "122b") { throw std::invalid_argument("--geometry takes flash-next or 122b"); }
            options.geometry_122b = v == "122b";
        } else {
            throw std::invalid_argument("unknown option " + arg);
        }
    }
    if (options.placement != "frames" && options.placement != "staged" && options.placement != "both") {
        throw std::invalid_argument("--placement takes frames, staged or both");
    }
    if (options.geometry_122b && !options.a16) {
        throw std::invalid_argument("--geometry 122b registers W4A16 experts only: add --activation a16, a16x2 or a16x3");
    }
    if (options.narrow_a16 == moe::NarrowA16::kA4Residual && (!options.a16 || options.geometry_122b)) {
        throw std::invalid_argument("--narrow a4r takes Flash-Next's W4A16 experts: add --activation a16");
    }
    return options;
}

// Random codes with scale bytes in the checkpoint's usual range (E4M3 words 40-80).
std::vector<std::uint8_t> random_records(std::mt19937& rng) {
    std::vector<std::uint8_t> records(static_cast<std::size_t>(kDistinct) * g_geometry.record_bytes());
    std::uniform_int_distribution<int> byte(0, 255), scale(40, 80);
    for (auto& b : records) { b = static_cast<std::uint8_t>(byte(rng)); }
    for (int r = 0; r < kDistinct; ++r) {
        std::uint8_t* record = records.data() + static_cast<std::size_t>(r) * g_geometry.record_bytes();
        const std::size_t units = static_cast<std::size_t>(g_geometry.record_bytes() / moe::kUnitBytes);
        for (std::size_t u = 0; u < units; ++u) {
            for (int s = 0; s < 16; ++s) { record[u * moe::kUnitBytes + 128 + s] = static_cast<std::uint8_t>(scale(rng)); }
        }
    }
    return records;
}

// Top-10 distinct experts per column under popularity 1 / (rank + 1)^0.7 over a fixed random
// permutation of the experts; with `per_expert` > 0, uniform blocks instead (every used expert gets
// exactly per_expert columns while 10 * columns / per_expert <= 512).
std::vector<float> routing_logits(std::mt19937& rng, int columns, int per_expert) {
    std::vector<int> rank(g_geometry.experts);
    for (int e = 0; e < g_geometry.experts; ++e) { rank[static_cast<std::size_t>(e)] = e; }
    std::shuffle(rank.begin(), rank.end(), rng);
    std::vector<double> weight(g_geometry.experts);
    for (int e = 0; e < g_geometry.experts; ++e) {
        weight[static_cast<std::size_t>(e)] = 1.0 / std::pow(rank[static_cast<std::size_t>(e)] + 1.0, 0.7);
    }
    std::normal_distribution<float> n(0.0F, 1.0F);
    std::vector<float> logits(static_cast<std::size_t>(g_geometry.experts + 1) * columns);
    for (int t = 0; t < columns; ++t) {
        float* column = &logits[static_cast<std::size_t>(t) * (g_geometry.experts + 1)];
        for (int e = 0; e <= g_geometry.experts; ++e) { column[e] = n(rng); }
        if (per_expert > 0) {
            // Uniform: the columns of block t / per_expert all route to that block's ten experts.
            const int block = t / per_expert;
            for (int k = 0; k < g_geometry.top_k; ++k) {
                column[(block * g_geometry.top_k + k) % g_geometry.experts] = 20.0F + static_cast<float>(g_geometry.top_k - k);
            }
            continue;
        }
        std::vector<double> w = weight;
        for (int k = 0; k < g_geometry.top_k; ++k) {
            std::discrete_distribution<int> pick(w.begin(), w.end());
            const int e                    = pick(rng);
            w[static_cast<std::size_t>(e)] = 0.0;
            column[e]                      = 20.0F + static_cast<float>(g_geometry.top_k - k);
        }
    }
    return logits;
}

struct Fixture {
    DeviceBuffer frames_memory;
    DeviceBuffer frames_table;
    DeviceBuffer resident_none;
    DeviceBuffer scales;
    DeviceBuffer staging;
    std::uint8_t* host = nullptr;
    std::uint8_t* host_device = nullptr;
    cudaStream_t overlap = nullptr;
    cudaEvent_t events[5] = {};
    bool a16 = false;

    Fixture(std::uint32_t seed, bool a16, bool split_scales) : a16(a16) {
        std::mt19937 rng(seed);
        const auto records = random_records(rng);
        const std::size_t bank = static_cast<std::size_t>(g_geometry.experts) * g_geometry.record_bytes();
        frames_memory = DeviceBuffer(bank);
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&host), bank, cudaHostAllocMapped));
        CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&host_device), host, 0));
        for (int e = 0; e < g_geometry.experts; ++e) {
            const std::uint8_t* source = records.data() + static_cast<std::size_t>(e % kDistinct) * g_geometry.record_bytes();
            std::memcpy(host + static_cast<std::size_t>(e) * g_geometry.record_bytes(), source, g_geometry.record_bytes());
        }
        CUDA_CHECK(cudaMemcpy(frames_memory.p, host, bank, cudaMemcpyHostToDevice));
        std::vector<std::int32_t> table(g_geometry.experts), none(g_geometry.experts, -1);
        for (int e = 0; e < g_geometry.experts; ++e) { table[static_cast<std::size_t>(e)] = e; }
        frames_table  = DeviceBuffer(sizeof(std::int32_t) * g_geometry.experts);
        resident_none = DeviceBuffer(sizeof(std::int32_t) * g_geometry.experts);
        CUDA_CHECK(cudaMemcpy(frames_table.p, table.data(), frames_table.bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(resident_none.p, none.data(), resident_none.bytes, cudaMemcpyHostToDevice));
        std::vector<moe::ExpertScales> s(g_geometry.experts);
        std::uniform_real_distribution<float> lg(-11.0F, -7.0F);
        for (auto& e : s) {
            e.input_gate = e.input_up = std::exp2(lg(rng));
            if (split_scales) { e.input_up = 1.5F * e.input_gate; }
            e.input_down              = std::exp2(lg(rng));
            e.alpha_gate = e.alpha_up = std::exp2(lg(rng)) * e.input_gate;
            e.alpha_down              = std::exp2(lg(rng)) * e.input_down;
            if (a16) { // W4A16: no input scales, the multipliers alone
                e.input_gate = e.input_up = e.input_down = 0.0F;
                e.alpha_gate = e.alpha_up = std::exp2(lg(rng));
                e.alpha_down              = std::exp2(lg(rng));
            }
        }
        scales = DeviceBuffer(sizeof(moe::ExpertScales) * g_geometry.experts);
        CUDA_CHECK(cudaMemcpy(scales.p, s.data(), scales.bytes, cudaMemcpyHostToDevice));
        staging = DeviceBuffer(static_cast<std::size_t>(kStagingSlots) * g_geometry.record_bytes());
        CUDA_CHECK(cudaStreamCreateWithFlags(&overlap, cudaStreamNonBlocking));
        for (auto& event : events) { CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming)); }
    }

    ~Fixture() {
        for (auto event : events) { cudaEventDestroy(event); }
        if (overlap != nullptr) { cudaStreamDestroy(overlap); }
        if (host != nullptr) { cudaFreeHost(host); }
    }

    Fixture(const Fixture&)            = delete;
    Fixture& operator=(const Fixture&) = delete;

    ops::MoeExpertSource source(bool staged) const {
        ops::MoeExpertSource s{.frame_base    = static_cast<const std::uint8_t*>(frames_memory.p),
                               .frames        = static_cast<const std::int32_t*>(staged ? resident_none.p : frames_table.p),
                               .host_records  = host_device,
                               .record_stride = g_geometry.record_bytes(),
                               .scales        = static_cast<const moe::ExpertScales*>(scales.p),
                               .activation    = a16 ? moe::ExpertActivation::kA16 : moe::ExpertActivation::kA4,
                               .staging_base  = static_cast<std::uint8_t*>(staging.p),
                               .staging_slots = kStagingSlots};
        s.overlap_stream = overlap;
        s.shape          = g_geometry.shape;
        for (int i = 0; i < 5; ++i) { s.overlap_events[i] = events[i]; }
        return s;
    }
};

void run_point(Fixture& fixture, int tokens, const Options& options, bench::L2FlushBuffer* flush, cudaStream_t stream) {
    std::mt19937 rng(options.seed + static_cast<std::uint32_t>(tokens));
    const int entries = g_geometry.top_k * tokens;
    const auto logits = routing_logits(rng, tokens, options.columns_per_expert);
    DeviceBuffer d_logits(logits.size() * sizeof(float));
    CUDA_CHECK(cudaMemcpy(d_logits.p, logits.data(), d_logits.bytes, cudaMemcpyHostToDevice));
    DeviceBuffer ids(sizeof(std::int32_t) * entries), weights(sizeof(float) * entries), gate(sizeof(float) * tokens);
    ops::MoeRouting routing{Tensor(ids.p, DType::I32, {g_geometry.top_k, tokens}), Tensor(weights.p, DType::FP32, {g_geometry.top_k, tokens}),
                            Tensor(gate.p, DType::FP32, {tokens})};
    ops::moe_route(Tensor(d_logits.p, DType::FP32, {g_geometry.experts + 1, tokens}), g_geometry.top_k, routing, stream);
    DeviceBuffer dispatch_memory(ops::moe_dispatch_bytes(g_geometry.experts, entries));
    ops::MoeDispatch dispatch = ops::carve_moe_dispatch(dispatch_memory.p, g_geometry.experts, entries);
    ops::moe_dispatch(routing, g_geometry.experts, dispatch, nullptr, stream);
    std::vector<std::int32_t> offsets(g_geometry.experts + 1);
    CUDA_CHECK(cudaMemcpyAsync(offsets.data(), dispatch.offsets, offsets.size() * sizeof(std::int32_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    int wide_experts = 0, wide_entries = 0, narrow_experts = 0, largest = 0, used = 0;
    for (int e = 0; e < g_geometry.experts; ++e) {
        const int n = offsets[static_cast<std::size_t>(e) + 1] - offsets[static_cast<std::size_t>(e)];
        largest = std::max(largest, n);
        used += n > 0;
        if (n > moe::kMaxColumns) {
            ++wide_experts;
            wide_entries += n;
        } else if (n > 0) {
            ++narrow_experts;
        }
    }
    const DeviceBuffer x = bench::make_bf16(static_cast<std::size_t>(g_geometry.shape.hidden) * tokens, options.seed + 7, -0.15F, 0.15F);
    DeviceBuffer outputs(sizeof(std::uint16_t) * g_geometry.shape.hidden * entries);
    const int max_jobs = std::min(g_geometry.experts, entries);
    DeviceBuffer workspace(ops::moe_experts_workspace_bytes(g_geometry.shape, 
        max_jobs, entries, tokens, options.a16 ? moe::ExpertActivation::kA16 : moe::ExpertActivation::kA4,
        options.wide_a16));
    for (const bool staged : {false, true}) {
        if ((staged && options.placement == "frames") || (!staged && options.placement == "staged")) { continue; }
        ops::MoeExpertSource source = fixture.source(staged);
        source.wide_a16             = options.wide_a16;
        source.narrow_a16           = options.narrow_a16;
        Tensor tx(x.p, DType::BF16, {g_geometry.shape.hidden, tokens});
        Tensor out(outputs.p, DType::BF16, {g_geometry.shape.hidden, entries});
        const auto launch = [&](cudaStream_t s) {
            ops::moe_experts(tx, dispatch, source, g_geometry.top_k, max_jobs, workspace.p, out, s);
        };
        const auto timing = options.cold ? bench::measure_cold_launch(launch, *flush, stream, options.warmup, options.repeat)
                                         : bench::measure_launch(launch, stream, options.warmup, options.repeat);
        const double tflops = g_geometry.expert_flops() * wide_entries / (timing.median_us * 1e-6) / 1e12;
        // FNV-1a of the outputs: equal hashes mean bit-identical results across kernel versions.
        std::vector<std::uint8_t> bytes(outputs.bytes);
        CUDA_CHECK(cudaMemcpy(bytes.data(), outputs.p, outputs.bytes, cudaMemcpyDeviceToHost));
        std::uint64_t hash = 1469598103934665603ULL;
        for (const std::uint8_t b : bytes) { hash = (hash ^ b) * 1099511628211ULL; }
        std::printf("T=%5d placement=%-6s experts used %3d (wide %3d, narrow %3d, largest %4d) wide entries %6d of %6d | "
                    "moe_experts median %9.1f us (min %9.1f, p95 %9.1f) | x48 layers %7.3f s | wide %.1f TFLOP/s | "
                    "output fnv1a %016llx\n",
                    tokens, staged ? "staged" : "frames", used, wide_experts, narrow_experts, largest, wide_entries,
                    entries, timing.median_us, timing.min_us, timing.p95_us, timing.median_us * 48e-6, tflops,
                    static_cast<unsigned long long>(hash));
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        if (options.geometry_122b) { g_geometry = {256, 8, moe::kQwen3_5_122BShape}; }
        DeviceContext context;
        std::printf("# gpu=%s geometry=%s timed=moe_experts (routing and dispatch untimed%s) activation=%s narrow=%s "
                    "staging_slots=%d\n",
                    context.props.name, options.geometry_122b ? "122b" : "flash-next",
                    options.cold ? ", L2 flushed before each call" : "", options.a16 ? "a16" : "a4",
                    options.narrow_a16 == moe::NarrowA16::kA4Residual ? "a4r" : "exact", kStagingSlots);
        Fixture fixture(options.seed, options.a16, options.split_scales);
        std::unique_ptr<bench::L2FlushBuffer> flush;
        if (options.cold) { flush = std::make_unique<bench::L2FlushBuffer>(256ULL << 20); }
        for (const int tokens : options.tokens) { run_point(fixture, tokens, options, flush.get(), context.stream); }
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "infernix_offloaded_moe_wide_bench: %s\n", error.what());
        return 1;
    }
}
