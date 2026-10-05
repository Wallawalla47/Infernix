// Prefill-shape benchmark of offloaded_sparse_moe's routed experts (moe_experts) for the
// Qwen3.8-Flash-Next layer: 512 experts, top-10, H = 2560, I = 640, `nvfp4_expert_rg16_v1` records.
//
// Each point routes T columns with a fixed skewed expert popularity (a Zipf-like law, so a few
// experts are wide over several 64-column tiles and the tail stays narrow), then times one
// moe_experts call as the prefill path issues it: 64 staging slots with the double-buffered
// overlap stream. Experts with more than eight columns take the wide route; the rest the narrow
// one. Placement:
//   frames : every record in device memory (isolates the expert kernels);
//   staged : every record in the pinned host bank, staged through the slots each call
//            (the cold-layer case, bound by the staging copy).
// The FLOP figure counts the wide entries' gate/up and down products only.
//
//   ninfer_offloaded_moe_wide_bench [--tokens 64,256,1024,4096] [--placement frames|staged|both]
//                                   [--warmup 3] [--repeat 10] [--seed N]

#include "ninfer/ops/offloaded_sparse_moe.h"

#include "core/device.h"
#include "ninfer_bench_common.h"
#include "ops/common/canonical_math.h"
#include "ops/offloaded_sparse_moe/cpu/w4a4_expert.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
namespace moe = ninfer::ops::offloaded_moe;

namespace {

constexpr int kExperts       = 512;
constexpr int kTopK          = 10;
constexpr int kDistinct      = 16; // distinct random records, repeated over the 512 experts
constexpr int kStagingSlots  = 64;
constexpr double kExpertFlops = 2.0 * (2.0 * moe::kIntermediate * moe::kHidden + 1.0 * moe::kHidden * moe::kIntermediate);

struct Options {
    std::vector<int> tokens{64, 256, 1024, 4096};
    std::string placement = "both";
    int warmup            = 3;
    int repeat            = 10;
    std::uint32_t seed    = 20261004U;
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
        } else if (arg == "--seed") {
            options.seed = static_cast<std::uint32_t>(std::stoul(value()));
        } else {
            throw std::invalid_argument("unknown option " + arg);
        }
    }
    if (options.placement != "frames" && options.placement != "staged" && options.placement != "both") {
        throw std::invalid_argument("--placement takes frames, staged or both");
    }
    return options;
}

// Random codes with scale bytes in the checkpoint's usual range (E4M3 words 40-80).
std::vector<std::uint8_t> random_records(std::mt19937& rng) {
    std::vector<std::uint8_t> records(static_cast<std::size_t>(kDistinct) * moe::kRecordBytes);
    std::uniform_int_distribution<int> byte(0, 255), scale(40, 80);
    for (auto& b : records) { b = static_cast<std::uint8_t>(byte(rng)); }
    for (int r = 0; r < kDistinct; ++r) {
        std::uint8_t* record = records.data() + static_cast<std::size_t>(r) * moe::kRecordBytes;
        const std::size_t units = static_cast<std::size_t>(moe::kRecordBytes / moe::kUnitBytes);
        for (std::size_t u = 0; u < units; ++u) {
            for (int s = 0; s < 16; ++s) { record[u * moe::kUnitBytes + 128 + s] = static_cast<std::uint8_t>(scale(rng)); }
        }
    }
    return records;
}

// Top-10 distinct experts per column under popularity 1 / (rank + 1)^0.7 over a fixed random
// permutation of the experts.
std::vector<float> routing_logits(std::mt19937& rng, int columns) {
    std::vector<int> rank(kExperts);
    for (int e = 0; e < kExperts; ++e) { rank[static_cast<std::size_t>(e)] = e; }
    std::shuffle(rank.begin(), rank.end(), rng);
    std::vector<double> weight(kExperts);
    for (int e = 0; e < kExperts; ++e) {
        weight[static_cast<std::size_t>(e)] = 1.0 / std::pow(rank[static_cast<std::size_t>(e)] + 1.0, 0.7);
    }
    std::normal_distribution<float> n(0.0F, 1.0F);
    std::vector<float> logits(static_cast<std::size_t>(kExperts + 1) * columns);
    for (int t = 0; t < columns; ++t) {
        float* column = &logits[static_cast<std::size_t>(t) * (kExperts + 1)];
        for (int e = 0; e <= kExperts; ++e) { column[e] = n(rng); }
        std::vector<double> w = weight;
        for (int k = 0; k < kTopK; ++k) {
            std::discrete_distribution<int> pick(w.begin(), w.end());
            const int e                    = pick(rng);
            w[static_cast<std::size_t>(e)] = 0.0;
            column[e]                      = 20.0F + static_cast<float>(kTopK - k);
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

    explicit Fixture(std::uint32_t seed) {
        std::mt19937 rng(seed);
        const auto records = random_records(rng);
        const std::size_t bank = static_cast<std::size_t>(kExperts) * moe::kRecordBytes;
        frames_memory = DeviceBuffer(bank);
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&host), bank, cudaHostAllocMapped));
        CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&host_device), host, 0));
        for (int e = 0; e < kExperts; ++e) {
            const std::uint8_t* source = records.data() + static_cast<std::size_t>(e % kDistinct) * moe::kRecordBytes;
            std::memcpy(host + static_cast<std::size_t>(e) * moe::kRecordBytes, source, moe::kRecordBytes);
        }
        CUDA_CHECK(cudaMemcpy(frames_memory.p, host, bank, cudaMemcpyHostToDevice));
        std::vector<std::int32_t> table(kExperts), none(kExperts, -1);
        for (int e = 0; e < kExperts; ++e) { table[static_cast<std::size_t>(e)] = e; }
        frames_table  = DeviceBuffer(sizeof(std::int32_t) * kExperts);
        resident_none = DeviceBuffer(sizeof(std::int32_t) * kExperts);
        CUDA_CHECK(cudaMemcpy(frames_table.p, table.data(), frames_table.bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(resident_none.p, none.data(), resident_none.bytes, cudaMemcpyHostToDevice));
        std::vector<moe::ExpertScales> s(kExperts);
        std::uniform_real_distribution<float> lg(-11.0F, -7.0F);
        for (auto& e : s) {
            e.input_gate = e.input_up = std::exp2(lg(rng));
            e.input_down              = std::exp2(lg(rng));
            e.alpha_gate = e.alpha_up = std::exp2(lg(rng)) * e.input_gate;
            e.alpha_down              = std::exp2(lg(rng)) * e.input_down;
        }
        scales = DeviceBuffer(sizeof(moe::ExpertScales) * kExperts);
        CUDA_CHECK(cudaMemcpy(scales.p, s.data(), scales.bytes, cudaMemcpyHostToDevice));
        staging = DeviceBuffer(static_cast<std::size_t>(kStagingSlots) * moe::kRecordBytes);
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
                               .record_stride = moe::kRecordBytes,
                               .scales        = static_cast<const moe::ExpertScales*>(scales.p),
                               .staging_base  = static_cast<std::uint8_t*>(staging.p),
                               .staging_slots = kStagingSlots};
        s.overlap_stream = overlap;
        for (int i = 0; i < 5; ++i) { s.overlap_events[i] = events[i]; }
        return s;
    }
};

void run_point(Fixture& fixture, int tokens, const Options& options, cudaStream_t stream) {
    std::mt19937 rng(options.seed + static_cast<std::uint32_t>(tokens));
    const int entries = kTopK * tokens;
    const auto logits = routing_logits(rng, tokens);
    DeviceBuffer d_logits(logits.size() * sizeof(float));
    CUDA_CHECK(cudaMemcpy(d_logits.p, logits.data(), d_logits.bytes, cudaMemcpyHostToDevice));
    DeviceBuffer ids(sizeof(std::int32_t) * entries), weights(sizeof(float) * entries), gate(sizeof(float) * tokens);
    ops::MoeRouting routing{Tensor(ids.p, DType::I32, {kTopK, tokens}), Tensor(weights.p, DType::FP32, {kTopK, tokens}),
                            Tensor(gate.p, DType::FP32, {tokens})};
    ops::moe_route(Tensor(d_logits.p, DType::FP32, {kExperts + 1, tokens}), kTopK, routing, stream);
    DeviceBuffer dispatch_memory(ops::moe_dispatch_bytes(kExperts, entries));
    ops::MoeDispatch dispatch = ops::carve_moe_dispatch(dispatch_memory.p, kExperts, entries);
    ops::moe_dispatch(routing, kExperts, dispatch, nullptr, stream);
    std::vector<std::int32_t> offsets(kExperts + 1);
    CUDA_CHECK(cudaMemcpyAsync(offsets.data(), dispatch.offsets, offsets.size() * sizeof(std::int32_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    int wide_experts = 0, wide_entries = 0, narrow_experts = 0, largest = 0, used = 0;
    for (int e = 0; e < kExperts; ++e) {
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
    const DeviceBuffer x = bench::make_bf16(static_cast<std::size_t>(moe::kHidden) * tokens, options.seed + 7, -0.15F, 0.15F);
    DeviceBuffer outputs(sizeof(std::uint16_t) * moe::kHidden * entries);
    const int max_jobs = std::min(kExperts, entries);
    DeviceBuffer workspace(ops::moe_experts_workspace_bytes(max_jobs, entries));
    for (const bool staged : {false, true}) {
        if ((staged && options.placement == "frames") || (!staged && options.placement == "staged")) { continue; }
        const ops::MoeExpertSource source = fixture.source(staged);
        Tensor tx(x.p, DType::BF16, {moe::kHidden, tokens});
        Tensor out(outputs.p, DType::BF16, {moe::kHidden, entries});
        const auto timing = bench::measure_launch(
            [&](cudaStream_t s) { ops::moe_experts(tx, dispatch, source, kTopK, max_jobs, workspace.p, out, s); },
            stream, options.warmup, options.repeat);
        const double tflops = kExpertFlops * wide_entries / (timing.median_us * 1e-6) / 1e12;
        std::printf("T=%5d placement=%-6s experts used %3d (wide %3d, narrow %3d, largest %4d) wide entries %6d of %6d | "
                    "moe_experts median %9.1f us (min %9.1f, p95 %9.1f) | x48 layers %7.3f s | wide %.1f TFLOP/s\n",
                    tokens, staged ? "staged" : "frames", used, wide_experts, narrow_experts, largest, wide_entries,
                    entries, timing.median_us, timing.min_us, timing.p95_us, timing.median_us * 48e-6, tflops);
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        DeviceContext context;
        std::printf("# gpu=%s timed=moe_experts (routing and dispatch untimed) staging_slots=%d overlap=on\n",
                    context.props.name, kStagingSlots);
        Fixture fixture(options.seed);
        for (const int tokens : options.tokens) { run_point(fixture, tokens, options, context.stream); }
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ninfer_offloaded_moe_wide_bench: %s\n", error.what());
        return 1;
    }
}
