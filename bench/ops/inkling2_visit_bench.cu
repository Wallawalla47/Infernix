// Decode-shape benchmark of Inkling 2's visit-chain kernels (include/infernix/ops/inkling2_visit.h) at T = 1 (or
// --columns): each configuration's kernel 48 times in one CUDA graph on a non-blocking stream (PDL dependents, as the
// forward captures them), median over repeats, microseconds a launch. Random data at realistic scales; the rows are
// L2-resident, as the forward's are.

#include "infernix/ops/inkling2_visit.h"
#include "ops/common/canonical_math.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <string>
#include <vector>

namespace ik    = infernix::ops::inkling2;
namespace moe   = infernix::ops::w4a4_moe;
namespace canon = infernix::ops::canon;

namespace {

constexpr int H = ik::kHidden, kLaunches = 48;

void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(status));
        std::exit(1);
    }
}

std::mt19937 g_rng(7);

template <class T>
T* device(std::size_t n) {
    T* p = nullptr;
    check(cudaMalloc(&p, std::max<std::size_t>(n, 1) * sizeof(T)), "cudaMalloc");
    check(cudaMemset(p, 0, std::max<std::size_t>(n, 1) * sizeof(T)), "cudaMemset");
    return p;
}

std::uint16_t* bf16_rows(std::size_t n, float sd, float mean = 0.0F) {
    std::normal_distribution<float> d(mean, sd);
    std::vector<std::uint16_t> host(n);
    for (auto& x : host) { x = canon::f32_to_bf16_rn(d(g_rng)); }
    auto* p = device<std::uint16_t>(n);
    check(cudaMemcpy(p, host.data(), n * 2, cudaMemcpyHostToDevice), "upload");
    return p;
}

float* f32_rows(std::size_t n, float sd) {
    std::normal_distribution<float> d(0.0F, sd);
    std::vector<float> host(n);
    for (auto& x : host) { x = d(g_rng); }
    auto* p = device<float>(n);
    check(cudaMemcpy(p, host.data(), n * 4, cudaMemcpyHostToDevice), "upload");
    return p;
}

double time_graph(const std::function<void(cudaStream_t)>& launch, cudaStream_t stream, int repeat) {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "capture");
    for (int i = 0; i < kLaunches; ++i) { launch(stream); }
    check(cudaStreamEndCapture(stream, &graph), "capture");
    check(cudaGraphInstantiate(&exec, graph, 0), "instantiate");
    cudaEvent_t start = nullptr, stop = nullptr;
    check(cudaEventCreate(&start), "event");
    check(cudaEventCreate(&stop), "event");
    std::vector<double> us;
    for (int r = 0; r < repeat + 3; ++r) {
        check(cudaEventRecord(start, stream), "record");
        check(cudaGraphLaunch(exec, stream), "launch");
        check(cudaEventRecord(stop, stream), "record");
        check(cudaEventSynchronize(stop), "sync");
        float ms = 0;
        check(cudaEventElapsedTime(&ms, start, stop), "elapsed");
        if (r >= 3) { us.push_back(1000.0 * ms / kLaunches); }
    }
    std::sort(us.begin(), us.end());
    check(cudaGraphExecDestroy(exec), "destroy");
    check(cudaGraphDestroy(graph), "destroy");
    return us[us.size() / 2];
}

} // namespace

int main(int argc, char** argv) {
    int T = 1, repeat = 30;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--columns" && i + 1 < argc) { T = std::atoi(argv[++i]); }
        else if (arg == "--repeat" && i + 1 < argc) { repeat = std::atoi(argv[++i]); }
        else {
            std::fprintf(stderr, "usage: %s [--columns T] [--repeat N]\n", argv[0]);
            return 2;
        }
    }
    cudaStream_t stream = nullptr;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");
    const std::size_t TH = static_cast<std::size_t>(T) * H;
    // One lane at positions 100 .. 100 + T - 1 (history from the ring).
    std::vector<int> lane(T, 0), pos(T), first = {0};
    for (int t = 0; t < T; ++t) { pos[t] = 100 + t; }
    int *d_lane = device<int>(T), *d_pos = device<int>(T), *d_first = device<int>(1);
    check(cudaMemcpy(d_lane, lane.data(), T * 4, cudaMemcpyHostToDevice), "upload");
    check(cudaMemcpy(d_pos, pos.data(), T * 4, cudaMemcpyHostToDevice), "upload");
    check(cudaMemcpy(d_first, first.data(), 4, cudaMemcpyHostToDevice), "upload");
    const ik::ColumnMap map{d_lane, d_pos, d_first, T};
    auto* experts = bf16_rows(TH * moe::kTopK, 0.05F);
    auto* cand    = f32_rows(TH, 1.0F);
    auto* conv    = bf16_rows(4 * H, 0.4F);
    auto* ring    = bf16_rows(std::size_t{ik::kRingPositions} * H, 0.05F);
    auto* input   = bf16_rows(TH, 1.0F);
    std::uint16_t* blocks[4];
    for (auto& b : blocks) { b = bf16_rows(TH, 1.0F); }
    auto* queries  = f32_rows(3 * H, 2.0F);
    auto* mqueries = f32_rows(4 * H, 2.0F);
    std::uint16_t* norms[3];
    for (auto& n : norms) { n = bf16_rows(H, 0.2F, 1.0F); }
    std::uint16_t *msrc0 = bf16_rows(TH, 1.5F), *msrc1 = bf16_rows(TH, 1.5F);
    auto* tail_out = device<std::uint16_t>(TH);
    std::uint16_t* x[3];
    for (auto& p : x) { p = device<std::uint16_t>(TH); }
    auto* mixed   = device<float>(3 * TH);
    auto* message = device<float>(3 * TH);
    auto* gate    = device<std::uint16_t>(3 * TH * 2);
    auto* fg      = f32_rows(3 * TH * 2, 1.2F);
    auto* o       = bf16_rows(TH, 0.5F);
    auto* xm      = device<std::uint16_t>(TH);
    auto* rows    = device<std::uint8_t>(static_cast<std::size_t>(T) * moe::row_bytes(moe::ExpertFormat::Mxfp8, H));
    auto* amax    = device<float>(T);
    std::printf("inkling2 visit kernels, T = %d, %d launches a graph, median of %d (us a launch)\n", T, kLaunches, repeat);

    const auto entry = [&](const char* name, bool tail, bool layer0, int nq, int nblocks, bool messages) {
        ik::EntryArgs a;
        a.columns = T;
        a.map     = map;
        if (tail) {
            a.tail     = {experts, cand, conv, {ring, H, std::int64_t{ik::kRingPositions} * H}};
            a.tail_out = tail_out;
        } else {
            a.input = input;
        }
        if (!layer0) {
            for (int s = 0; s < nblocks; ++s) { a.blocks.rows[s] = blocks[s]; }
            a.blocks.count = nblocks;
            a.queries      = queries;
            a.nq           = nq;
        }
        if (messages) {
            a.message_queries    = mqueries;
            a.message_sources[0] = msrc0;
            a.message_sources[1] = msrc1;
            a.mixed = mixed, a.message = message, a.gate_input = gate;
        } else {
            for (int i = 0; i < 3; ++i) { a.norms[i] = norms[i], a.x[i] = x[i]; }
        }
        std::printf("  %-44s %7.2f\n", name, time_graph([&](cudaStream_t s) { ik::visit_entry(a, s); }, stream, repeat));
    };
    entry("visit_entry layer 0", false, true, 1, 0, false);
    entry("visit_entry sliding, no tail, 2 blocks", false, false, 1, 2, false);
    entry("visit_entry full, no tail, 3 blocks", false, false, 3, 3, false);
    entry("visit_entry sliding, no tail, messages, 2 blocks", false, false, 1, 2, true);
    entry("visit_entry sliding, tail, 2 blocks", true, false, 1, 2, false);
    entry("visit_entry sliding, tail, 4 blocks", true, false, 1, 4, false);
    entry("visit_entry full, tail, 3 blocks", true, false, 3, 3, false);
    entry("visit_entry sliding, tail, messages, 2 blocks", true, false, 1, 2, true);
    entry("visit_entry full, tail, messages, 2 blocks", true, false, 3, 2, true);

    const auto mlp = [&](const char* name, bool messages, moe::ExpertFormat format, int nblocks) {
        ik::MlpEntryArgs a;
        a.columns  = T;
        a.map      = map;
        a.o        = o;
        a.conv     = conv;
        a.ring     = {ring, H, std::int64_t{ik::kRingPositions} * H};
        a.h        = input;
        a.cand     = cand;
        for (int s = 0; s < nblocks; ++s) { a.blocks.rows[s] = blocks[s]; }
        a.blocks.count = nblocks;
        a.query        = queries;
        if (messages) {
            a.message_query      = mqueries + 3 * H;
            a.message_sources[0] = msrc0;
            a.message_sources[1] = msrc1;
            a.mixed = mixed, a.message = message, a.gate_input = gate;
        } else {
            a.norm = norms[0];
            a.x    = xm;
            a.rows = {format, rows, amax};
        }
        std::printf("  %-44s %7.2f\n", name, time_graph([&](cudaStream_t s) { ik::mlp_entry(a, s); }, stream, repeat));
    };
    mlp("mlp_entry NVFP4 rows, 2 blocks", false, moe::ExpertFormat::Nvfp4FourOverSix, 2);
    mlp("mlp_entry MXFP8 rows, 2 blocks", false, moe::ExpertFormat::Mxfp8, 2);
    mlp("mlp_entry messages, 2 blocks", true, moe::ExpertFormat::Nvfp4FourOverSix, 2);

    const auto exit = [&](const char* name, int kinds, bool with_rows) {
        ik::GateExitArgs a;
        a.columns  = T;
        a.kinds    = kinds;
        a.mixed    = mixed;
        a.shared_x = false;
        a.message  = message;
        a.fg       = fg;
        for (int k = 0; k < kinds; ++k) { a.norms[k] = norms[k], a.x[k] = x[k]; }
        if (with_rows) { a.rows = {moe::ExpertFormat::Nvfp4FourOverSix, rows, amax}; }
        std::printf("  %-44s %7.2f\n", name, time_graph([&](cudaStream_t s) { ik::gate_exit(a, s); }, stream, repeat));
    };
    exit("gate_exit 3 kinds", 3, false);
    exit("gate_exit mlp, NVFP4 rows", 1, true);
    exit("gate_exit mlp, no rows", 1, false);
    return 0;
}
