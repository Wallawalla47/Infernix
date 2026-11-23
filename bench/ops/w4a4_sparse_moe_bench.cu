// Decode-shape benchmark of w4a4_sparse_moe (Inkling 2's routed experts, [128, 2304, 1152], top 8): one MoE layer's
// router and experts as a decode or verify round issues them (moe_route, then moe_experts), captured in a CUDA graph and
// replayed with the L2 flushed before each replay (a layer's experts are cold in decode: 47 other visits stream ~4 GB
// between two of its calls). Random weights and router: each call routes T columns to their own top 8.
//
//   infernix_w4a4_sparse_moe_bench [--format nvfp4|mxfp8] [--tokens 1,2,4,8,16,32] [--repeat 20] [--warmup 3]
//                                  [--part all|route|experts|single|prompt] [--layers 8] [--warm]
//                                  (single: moe_single, T = 1; prompt: moe_experts_prompt, the prefill route's experts;
//                                  --warm: no L2 flush between replays, with --layers 1 the weights stay in L2)
//                                  [--core native|exact]   (NVFP4's decode / verify core, Nvfp4Core)
//                                  [--warm-experts K [--warm-down]]   (the expert-prefetch sizing: each replay flushes
//                                  L2, then reads K of the call's routed experts (column 0's first K slots) through L2,
//                                  gate/up bytes only or with --warm-down the whole record; --layers 1 only)
//                                  [--activation checkpoint|a8]   (NVFP4: the checkpoint's A4 rows, or --a8-activations'
//                                  A8 rows (moe_encode_a8 of the columns' x) and h)
//
// Per T: the median and minimum call time, the distinct experts the call reads and their record bytes, and the
// effective bandwidth over those bytes (the floor a weight-streaming kernel approaches).

#include "infernix/ops/w4a4_sparse_moe.h"
#include "ops/common/canonical_math.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <set>
#include <string>
#include <vector>

namespace canon = infernix::ops::canon;
namespace moe   = infernix::ops::w4a4_moe;

namespace {

void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(status));
        std::exit(1);
    }
}

constexpr int E = 128, H = 2304, I = 1152, K = moe::kTopK;

// Reads n16 16-byte words through L2 (ld.global.cg), folding them into *sink so the loads stay.
__global__ void touch_kernel(const uint4* p, std::size_t n16, unsigned* sink) {
    unsigned acc = 0;
    for (std::size_t i = blockIdx.x * std::size_t{blockDim.x} + threadIdx.x; i < n16; i += std::size_t{gridDim.x} * blockDim.x) {
        const uint4 v = __ldcg(p + i);
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x12345678U) { *sink = acc; }
}

std::vector<int> parse_list(const char* text) {
    std::vector<int> out;
    for (const char* p = text; *p;) {
        out.push_back(std::atoi(p));
        while (*p && *p != ',') { ++p; }
        if (*p == ',') { ++p; }
    }
    return out;
}

} // namespace

int main(int argc, char** argv) {
    bool a8 = false;
    std::vector<int> tokens = {1, 2, 4, 8, 16, 32};
    int repeat = 20, warmup = 3;
    std::string part = "all"; // all | route | experts | single (moe_single: T = 1)
    bool warm = false;        // --warm: no L2 flush between replays (with --layers 1 the bank stays in L2)
    int layers = 8;            // layer calls a graph (each its own bank: cold weights, as consecutive visits)
    moe::Nvfp4Core core = moe::Nvfp4Core::Native;
    int warm_experts = -1; // --warm-experts K: K of the routed experts read into L2 after each flush
    bool warm_down   = false;
    moe::ExpertActivation activation = moe::ExpertActivation::Checkpoint;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--format" && i + 1 < argc) { a8 = std::string(argv[++i]) == "mxfp8"; }
        else if (arg == "--tokens" && i + 1 < argc) { tokens = parse_list(argv[++i]); }
        else if (arg == "--repeat" && i + 1 < argc) { repeat = std::atoi(argv[++i]); }
        else if (arg == "--warmup" && i + 1 < argc) { warmup = std::atoi(argv[++i]); }
        else if (arg == "--part" && i + 1 < argc) { part = argv[++i]; }
        else if (arg == "--warm") { warm = true; }
        else if (arg == "--layers" && i + 1 < argc) { layers = std::max(1, std::atoi(argv[++i])); }
        else if (arg == "--core" && i + 1 < argc) {
            core = std::string(argv[++i]) == "exact" ? moe::Nvfp4Core::Exact : moe::Nvfp4Core::Native;
        } else if (arg == "--warm-experts" && i + 1 < argc) {
            warm_experts = std::clamp(std::atoi(argv[++i]), 0, K);
        } else if (arg == "--warm-down") {
            warm_down = true;
        } else if (arg == "--activation" && i + 1 < argc) {
            activation = std::string(argv[++i]) == "a8" ? moe::ExpertActivation::A8 : moe::ExpertActivation::Checkpoint;
        } else {
            std::fprintf(stderr, "usage: %s [--format nvfp4|mxfp8] [--tokens 1,2,4] [--repeat N] [--warmup N]\n", argv[0]);
            return 2;
        }
    }
    if (warm_experts >= 0 && (layers != 1 || warm)) {
        std::fprintf(stderr, "--warm-experts needs --layers 1 and no --warm\n");
        return 2;
    }
    const bool act8 = activation == moe::ExpertActivation::A8;
    if (act8 && a8) {
        std::fprintf(stderr, "--activation a8 is for NVFP4 banks\n");
        return 2;
    }
    const int unit = a8 ? 528 : 144, block = a8 ? 32 : 16;
    const std::uint64_t gate_up = std::uint64_t(2 * I / 16) * (H / block) * unit;
    const std::uint64_t record  = gate_up + std::uint64_t(H / 16) * (I / block) * unit;
    const std::uint64_t stride  = (record + 4095) / 4096 * 4096;
    // Random record bytes: codes of every value; NVFP4 scale words 0x28-0x4E, MXFP8 codes without NaN and UE8M0 110-120.
    std::vector<std::uint8_t> records(E * stride, 0);
    std::uint64_t state = 0x9E3779B97F4A7C15ULL;
    const auto next = [&] {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        return state;
    };
    for (int e = 0; e < E; ++e) {
        for (std::uint64_t u = 0; u < record / unit; ++u) {
            std::uint8_t* p = records.data() + e * stride + u * unit;
            const int codes = a8 ? 512 : 128;
            for (int i = 0; i < codes; i += 8) {
                std::uint64_t r = next();
                std::memcpy(p + i, &r, 8);
            }
            if (a8) {
                for (int i = 0; i < codes; ++i) { p[i] = (p[i] & 0x7F) == 0x7F ? 0x3C : p[i]; }
            }
            for (int i = 0; i < 16; ++i) { p[codes + i] = static_cast<std::uint8_t>(a8 ? 110 + next() % 11 : 0x28 + next() % 39); }
        }
    }
    std::mt19937 rng(5);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    std::vector<float> amax(2 * E);
    for (float& a : amax) { a = 0.05F + 0.28F * static_cast<float>(next() % 1000) / 1000.0F; }
    const float den[2] = {2688.0F, 2688.0F};
    std::vector<std::uint16_t> router(std::size_t(E) * H);
    for (auto& w : router) { w = canon::f32_to_bf16_rn(normal(rng) * 0.02F); }
    std::vector<float> bias(E);
    for (float& b : bias) { b = normal(rng) * 0.01F; }

    std::vector<std::uint8_t*> d_banks(layers, nullptr);
    float *d_amax = nullptr, *d_den = nullptr, *d_bias = nullptr;
    std::uint16_t* d_router = nullptr;
    for (auto& d : d_banks) {
        check(cudaMalloc(&d, records.size()), "cudaMalloc");
        check(cudaMemcpy(d, records.data(), records.size(), cudaMemcpyHostToDevice), "upload");
    }
    check(cudaMalloc(&d_amax, amax.size() * 4), "cudaMalloc");
    check(cudaMemcpy(d_amax, amax.data(), amax.size() * 4, cudaMemcpyHostToDevice), "upload");
    check(cudaMalloc(&d_den, 8), "cudaMalloc");
    check(cudaMemcpy(d_den, den, 8, cudaMemcpyHostToDevice), "upload");
    check(cudaMalloc(&d_router, router.size() * 2), "cudaMalloc");
    check(cudaMemcpy(d_router, router.data(), router.size() * 2, cudaMemcpyHostToDevice), "upload");
    check(cudaMalloc(&d_bias, E * 4), "cudaMalloc");
    check(cudaMemcpy(d_bias, bias.data(), E * 4, cudaMemcpyHostToDevice), "upload");
    void* workspace = nullptr;
    // A workspace for the widest call of the sweep (prefill widths past the decode calls' 128).
    const int capacity = std::max(moe::kMaxColumns, *std::max_element(tokens.begin(), tokens.end()));
    check(cudaMalloc(&workspace, moe::workspace_bytes(E, I, capacity, activation)), "cudaMalloc");
    check(cudaMemset(workspace, 0, moe::workspace_bytes(E, I, capacity, activation)), "cudaMemset");
    const moe::Workspace ws = moe::carve_workspace(workspace, E, I, capacity, activation);
    void* flush = nullptr;
    const std::size_t flush_bytes = std::size_t(256) << 20;
    check(cudaMalloc(&flush, flush_bytes), "cudaMalloc");
    cudaStream_t stream;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");
    moe::Bank bank;
    bank.records       = d_banks[0];
    bank.record_stride = stride;
    bank.format        = a8 ? moe::ExpertFormat::Mxfp8 : moe::ExpertFormat::Nvfp4FourOverSix;
    bank.amax          = d_amax;
    bank.denominators  = d_den;
    bank.experts       = E;
    bank.hidden        = H;
    bank.intermediate  = I;
    std::printf("w4a4_sparse_moe %s%s, record %.2f MB, %s, per layer of %d in one CUDA graph (own banks: cold weights)\n",
                a8 ? "MXFP8" : "NVFP4", act8 ? " A8" : "", record / 1e6, part.c_str(), layers);
    for (const int T : tokens) {
        std::vector<std::uint16_t> x(std::size_t(T) * H);
        std::vector<float> xf(std::size_t(T) * H);
        for (std::size_t i = 0; i < x.size(); ++i) {
            x[i]  = canon::f32_to_bf16_rn(normal(rng));
            xf[i] = canon::bf16_to_f32(x[i]);
        }
        const int row_bytes = moe::row_bytes(bank.format, H);
        std::vector<std::uint8_t> rows(std::size_t(T) * row_bytes);
        std::vector<float> row_amax(T);
        for (int t = 0; t < T; ++t) {
            const float* v = xf.data() + std::size_t(t) * H;
            std::uint8_t* out = rows.data() + std::size_t(t) * row_bytes;
            if (!a8) {
                float m = 0;
                for (int k = 0; k < H; ++k) { m = std::max(m, std::fabs(v[k])); }
                row_amax[t] = m;
                const canon::FourOverSixRow row = canon::four_over_six_row(m);
                for (int b = 0; b < H / 16; ++b) {
                    float blk[16];
                    for (int j = 0; j < 16; ++j) { blk[j] = v[16 * b + j]; }
                    const canon::FourOverSixBlock q = canon::four_over_six_block(blk, row);
                    for (int j = 0; j < 8; ++j) { out[8 * b + j] = q.code[2 * j] | (q.code[2 * j + 1] << 4); }
                    out[H / 2 + b] = q.scale_word;
                }
            } else {
                for (int b = 0; b < H / 32; ++b) {
                    float grp[32];
                    for (int j = 0; j < 32; ++j) { grp[j] = v[32 * b + j]; }
                    const canon::Mxfp8Group q = canon::mxfp8_group(grp);
                    std::memcpy(out + 32 * b, q.code, 32);
                    out[H + b] = q.exponent;
                }
            }
        }
        std::uint16_t *d_x = nullptr, *d_out = nullptr;
        std::uint8_t* d_rows = nullptr;
        float* d_row_amax = nullptr;
        check(cudaMalloc(&d_x, x.size() * 2), "cudaMalloc");
        check(cudaMemcpy(d_x, x.data(), x.size() * 2, cudaMemcpyHostToDevice), "upload");
        check(cudaMalloc(&d_rows, rows.size()), "cudaMalloc");
        check(cudaMemcpy(d_rows, rows.data(), rows.size(), cudaMemcpyHostToDevice), "upload");
        check(cudaMalloc(&d_row_amax, T * 4), "cudaMalloc");
        check(cudaMemcpy(d_row_amax, row_amax.data(), T * 4, cudaMemcpyHostToDevice), "upload");
        check(cudaMalloc(&d_out, std::size_t(T) * K * H * 2), "cudaMalloc");
        cudaGraph_t graph;
        cudaGraphExec_t exec;
        if (part == "experts" || part == "prompt" || warm_experts >= 0) { // the routing the timed experts read, once
            moe::moe_route(d_x, T, d_router, d_bias, E, H, 8.0F, ws, stream);
            check(cudaStreamSynchronize(stream), "route");
        }
        std::vector<std::int32_t> warmed(K);
        if (warm_experts >= 0) { check(cudaMemcpy(warmed.data(), ws.routed, K * 4, cudaMemcpyDeviceToHost), "download"); }
        unsigned* d_sink = nullptr;
        check(cudaMalloc(&d_sink, 4), "cudaMalloc");
        const std::uint64_t warm_bytes = warm_down ? record : gate_up;
        moe::Activations act{d_rows, d_row_amax, T};
        act.core = core;
        std::uint8_t* d_a8 = nullptr;
        if (act8) { // the columns' A8 rows (the producers' codec)
            check(cudaMalloc(&d_a8, std::size_t(T) * moe::kA8XRowBytes), "cudaMalloc");
            moe::moe_encode_a8(d_x, T, d_a8, stream);
            check(cudaStreamSynchronize(stream), "encode");
            act.rows       = d_a8;
            act.amax       = nullptr;
            act.activation = moe::ExpertActivation::A8;
        }
        check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "capture");
        for (int l = 0; l < layers; ++l) {
            moe::Bank layer_bank = bank;
            layer_bank.records   = d_banks[l];
            if (part == "single") { // the router and experts of one column in one launch (T = 1 only)
                moe::moe_single(d_x, act, d_router, d_bias, 8.0F, layer_bank, ws, d_out, stream);
                continue;
            }
            if (part == "prompt") { // the prompt route's experts (moe_experts_prompt) on the dispatched entries
                moe::moe_dispatch(moe::routed_routing(ws), T, E, ws, stream);
                moe::moe_experts_prompt(act, moe::routed_routing(ws), layer_bank, ws, d_out, stream);
                continue;
            }
            if (part != "experts") { moe::moe_route(d_x, T, d_router, d_bias, E, H, 8.0F, ws, stream); }
            if (part == "experts") { moe::moe_dispatch(moe::routed_routing(ws), T, E, ws, stream); }
            if (part != "route") { moe::moe_experts(act, moe::routed_routing(ws), layer_bank, ws, d_out, stream); }
        }
        check(cudaStreamEndCapture(stream, &graph), "capture");
        check(cudaGraphInstantiate(&exec, graph, 0), "instantiate");
        cudaEvent_t start, stop;
        check(cudaEventCreate(&start), "event");
        check(cudaEventCreate(&stop), "event");
        std::vector<float> times;
        for (int r = 0; r < warmup + repeat; ++r) {
            if (!warm) { check(cudaMemsetAsync(flush, r & 0xFF, flush_bytes, stream), "flush"); }
            for (int j = 0; j < std::max(warm_experts, 0); ++j) {
                const auto* p = reinterpret_cast<const uint4*>(d_banks[0] + std::uint64_t(warmed[j]) * stride);
                touch_kernel<<<340, 256, 0, stream>>>(p, warm_bytes / 16, d_sink);
            }
            check(cudaEventRecord(start, stream), "record");
            check(cudaGraphLaunch(exec, stream), "launch");
            check(cudaEventRecord(stop, stream), "record");
            check(cudaEventSynchronize(stop), "sync");
            float ms = 0;
            check(cudaEventElapsedTime(&ms, start, stop), "elapsed");
            if (r >= warmup) { times.push_back(ms * 1000.0F / static_cast<float>(layers)); }
        }
        std::vector<std::int32_t> routed(std::size_t(T) * K);
        check(cudaMemcpy(routed.data(), ws.routed, routed.size() * 4, cudaMemcpyDeviceToHost), "download");
        const std::set<int> distinct(routed.begin(), routed.end());
        std::sort(times.begin(), times.end());
        const double median = times[times.size() / 2], bytes = double(distinct.size()) * record;
        std::printf("T=%3d  median %8.1f us  min %8.1f us  experts %3zu  %7.1f MB  %6.0f GB/s", T, median, times.front(),
                    distinct.size(), bytes / 1e6, bytes / (median * 1e-6) / 1e9);
        if (warm_experts >= 0) { std::printf("  warm experts %d (%s)", warm_experts, warm_down ? "whole records" : "gate/up"); }
        if (part == "prompt") { // the call's multiply-adds: T x top 8 x (gate, up and down: 3 x H x I)
            std::printf("  %6.1f TOPS", 2.0 * T * K * 3.0 * H * I / (median * 1e-6) / 1e12);
        }
        std::printf("\n");
        check(cudaFree(d_sink), "cudaFree");
        check(cudaGraphExecDestroy(exec), "destroy");
        check(cudaGraphDestroy(graph), "destroy");
        for (void* p : {static_cast<void*>(d_x), static_cast<void*>(d_rows), static_cast<void*>(d_row_amax),
                        static_cast<void*>(d_out), static_cast<void*>(d_a8)}) {
            check(cudaFree(p), "cudaFree");
        }
    }
    return 0;
}
