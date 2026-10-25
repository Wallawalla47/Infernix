// Repeat determinism of the two-term FP8 activation routes on Qwen3.8-Flash-Next recipe C's FP8
// problems: fp8-a8-pair-block32 (LinearRoutes::Fp8A8PairBlock32, per-32-value UE8M0 block scales on
// the block-scaled MMA) and fp8-a8-pair (LinearRoutes::Fp8A8Pair, column scales). Every call of a
// case must reproduce the first call's output bits, compared on the device output by output, eager
// and graph-replayed, with the output poisoned (0xFFFF, a BF16 NaN) before each call so that an
// unwritten tile also counts.
//
// A first block-scaled variant (2026-10-17) was dropped after about one 128 x 128 output tile in a
// hundred changed between identical calls on [2560, 6144] at 128-column tiles (T = 513, 1024, 2049),
// before its consumers fenced their shared reads ahead of the TMA refill (op-development.md 4.3).
// The [2560, 6144] block32 cases therefore run thousands of calls; each case prints the FNV-1a hash
// of its reference output so that two builds can be compared bit for bit.
//
//   infernix_fp8_a8_pair_repeat_test [calls per [2560,6144] block32 case (4000)]
//                                    [calls per other case (500)] [label filter]

#include "ops/op_family.h"
#include "core/arena.h"
#include "infernix/ops/linear.h"
#include "ops/linear/linear_test_common.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <random>
#include <string>
#include <vector>

namespace {

using infernix::DType;
using infernix::Tensor;
using infernix::Weight;
using infernix::ops::LinearPolicy;
using infernix::ops::LinearRoutes;
using infernix::test::cuda_check;

// Adds the number of outputs that differ from the reference to *differing, and records the least
// differing index in *first.
__global__ void count_differences(const std::uint16_t* __restrict__ reference,
                                  const std::uint16_t* __restrict__ actual, std::size_t count,
                                  unsigned* differing, unsigned long long* first) {
    unsigned local = 0;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        if (reference[i] != actual[i]) {
            ++local;
            atomicMin(first, static_cast<unsigned long long>(i));
        }
    }
    if (local != 0) atomicAdd(differing, local);
}

// Activations shaped like a transformer's (test_fp8_a8.cpp outlier_activations): column magnitudes
// over nine binades, every 61st channel 2^8 larger, every 37th column zero.
std::vector<std::uint16_t> activations(std::int32_t k, std::int32_t t, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    std::vector<std::uint16_t> bits(static_cast<std::size_t>(k) * t);
    for (std::int32_t token = 0; token < t; ++token) {
        for (std::int32_t channel = 0; channel < k; ++channel) {
            float value = std::ldexp(normal(rng), token % 9 - 4);
            if (channel % 61 == 7) value *= 256.0F;
            if (token % 37 == 5) value = 0.0F;
            bits[static_cast<std::size_t>(token) * k + channel] = infernix::test::f32_to_bf16(value);
        }
    }
    return bits;
}

unsigned long long fnv(const std::vector<std::uint16_t>& v) {
    std::uint64_t h = 1469598103934665603ULL;
    for (const std::uint16_t e : v) { h = (h ^ e) * 1099511628211ULL; }
    return static_cast<unsigned long long>(h);
}

struct Case {
    const char* route_name;
    LinearRoutes routes;
    std::int32_t n, k, t;
    bool heavy; // the [2560, 6144] block32 cases: the long repeat count
};

// The block32 route at the widths that failed (513, 1024, 2049), one beyond (4096) and, on the
// problems it serves below 257 columns, the 64-column tiles; the column-scaled pair at each
// problem's admission and in the 128-column tiles.
std::vector<Case> cases() {
    struct Problem {
        std::int32_t n, k, min_tokens;
    };
    constexpr Problem problems[]{{16384, 2560, 33}, {13312, 2560, 97}, {2560, 6144, 513},
                                 {1280, 2560, 1025}, {2560, 640, 513}};
    std::vector<Case> out;
    for (const Problem& p : problems) {
        const bool heavy = p.n == 2560 && p.k == 6144;
        std::vector<std::int32_t> widths{p.min_tokens, 1024, 2049, 4096};
        if (p.min_tokens <= 256) widths.push_back(200);
        std::sort(widths.begin(), widths.end());
        widths.erase(std::unique(widths.begin(), widths.end()), widths.end());
        for (const std::int32_t t : widths) {
            if (t >= p.min_tokens)
                out.push_back({"fp8-a8-pair-block32", LinearRoutes::Fp8A8PairBlock32, p.n, p.k, t, heavy});
        }
        for (const std::int32_t t : {p.min_tokens, std::max(p.min_tokens, 257) + 44})
            out.push_back({"fp8-a8-pair", LinearRoutes::Fp8A8Pair, p.n, p.k, t, false});
    }
    return out;
}

int run(const Case& c, int calls, std::uint32_t seed) {
    const auto host_weight = infernix::test::linear::make_fp8_weight(c.n, c.k, seed);
    infernix::DeviceBuffer payload(host_weight.payload.size());
    payload.copy_from_host(host_weight.payload.data(), host_weight.payload.size());
    const Weight w = host_weight.device_weight(payload.p);
    const auto bits = activations(c.k, c.t, seed + 1U);
    infernix::DeviceBuffer x(bits.size() * sizeof(std::uint16_t));
    x.copy_from_host(bits.data(), bits.size() * sizeof(std::uint16_t));
    const std::size_t elements = static_cast<std::size_t>(c.n) * c.t;
    infernix::DeviceBuffer reference(elements * sizeof(std::uint16_t));
    infernix::DeviceBuffer y(elements * sizeof(std::uint16_t));
    const std::size_t capacity =
        infernix::ops::linear_workspace_capacity_bytes(w.qtype, w.n, w.k, LinearPolicy::A16Only, c.t, c.t);
    infernix::DeviceArena workspace(std::max<std::size_t>(capacity, 256));
    infernix::DeviceBuffer differing(static_cast<std::size_t>(calls + 1) * sizeof(unsigned));
    infernix::DeviceBuffer first(static_cast<std::size_t>(calls + 1) * sizeof(unsigned long long));
    auto* differing_at = static_cast<unsigned*>(differing.p);
    auto* first_at     = static_cast<unsigned long long*>(first.p);

    cudaStream_t stream = nullptr;
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");
    const Tensor input(x.p, DType::BF16, {w.k, c.t});
    Tensor reference_out(reference.p, DType::BF16, {w.n, c.t});
    Tensor out(y.p, DType::BF16, {w.n, c.t});
    infernix::ops::linear(input, w, reference_out, LinearPolicy::A16Only, workspace, stream, c.routes);
    cuda_check(cudaMemsetAsync(differing.p, 0, differing.bytes, stream), "cudaMemsetAsync");
    cuda_check(cudaMemsetAsync(first.p, 0xFF, first.bytes, stream), "cudaMemsetAsync");
    const int compare_blocks = static_cast<int>(std::min<std::size_t>((elements + 255) / 256, 1184));
    const auto one_call = [&](int slot) {
        cuda_check(cudaMemsetAsync(y.p, 0xFF, y.bytes, stream), "cudaMemsetAsync");
        infernix::ops::linear(input, w, out, LinearPolicy::A16Only, workspace, stream, c.routes);
        count_differences<<<compare_blocks, 256, 0, stream>>>(
            static_cast<const std::uint16_t*>(reference.p), static_cast<const std::uint16_t*>(y.p),
            elements, differing_at + slot, first_at + slot);
        cuda_check(cudaGetLastError(), "count_differences");
    };
    // Eager calls, each into its own counter.
    for (int call = 0; call < calls; ++call) one_call(call);
    cuda_check(cudaStreamSynchronize(stream), "eager calls");
    // Graph replays (captured descriptors), all into the last counter.
    cudaGraph_t graph    = nullptr;
    cudaGraphExec_t exec = nullptr;
    const int replays    = std::max(1, calls / 4);
    cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "cudaStreamBeginCapture");
    one_call(calls);
    cuda_check(cudaStreamEndCapture(stream, &graph), "cudaStreamEndCapture");
    cuda_check(cudaGraphInstantiate(&exec, graph, 0), "cudaGraphInstantiate");
    for (int replay = 0; replay < replays; ++replay) cuda_check(cudaGraphLaunch(exec, stream), "cudaGraphLaunch");
    cuda_check(cudaStreamSynchronize(stream), "graph replays");
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);

    std::vector<std::uint16_t> want(elements);
    reference.copy_to_host(want.data(), want.size() * sizeof(std::uint16_t));
    std::vector<unsigned> counts(static_cast<std::size_t>(calls + 1));
    std::vector<unsigned long long> firsts(static_cast<std::size_t>(calls + 1));
    differing.copy_to_host(counts.data(), counts.size() * sizeof(unsigned));
    first.copy_to_host(firsts.data(), firsts.size() * sizeof(unsigned long long));
    cuda_check(cudaStreamDestroy(stream), "cudaStreamDestroy");

    int bad_calls = 0, reported = 0;
    unsigned long long bad_outputs = 0;
    for (int call = 0; call < calls; ++call) {
        if (counts[static_cast<std::size_t>(call)] == 0) continue;
        ++bad_calls;
        bad_outputs += counts[static_cast<std::size_t>(call)];
        if (reported++ < 4) {
            const unsigned long long i = firsts[static_cast<std::size_t>(call)];
            std::printf("  %s [%d,%d] T=%d call %d: %u outputs differ, first token %llu row %llu "
                        "(128 x 128 tile %llu, %llu)\n",
                        c.route_name, c.n, c.k, c.t, call, counts[static_cast<std::size_t>(call)],
                        i / c.n, i % c.n, i / c.n / 128, i % c.n / 128);
        }
    }
    const unsigned graph_outputs = counts[static_cast<std::size_t>(calls)];
    std::printf("%s [%d,%d] T=%d: reference %016llx; eager %d of %d calls differ (%llu outputs), "
                "graph %u differing outputs over %d replays\n",
                c.route_name, c.n, c.k, c.t, fnv(want), bad_calls, calls, bad_outputs, graph_outputs,
                replays);
    return bad_calls != 0 || graph_outputs != 0 ? 1 : 0;
}

} // namespace

INFERNIX_OP_TEST(fp8_a8_pair_repeat) {
    if (!infernix::test::linear::cuda_available()) {
        std::printf("SKIP: no usable CUDA device\n");
        return 77;
    }
    const int heavy_calls = argc > 1 ? std::max(1, std::atoi(argv[1])) : 4000;
    const int calls       = argc > 2 ? std::max(1, std::atoi(argv[2])) : 500;
    const std::string only = argc > 3 ? argv[3] : "";
    try {
        int failures       = 0;
        std::uint32_t seed = 4049U;
        for (const Case& c : cases()) {
            seed += 2U;
            const std::string label = std::string(c.route_name) + " [" + std::to_string(c.n) + "," +
                                      std::to_string(c.k) + "] T=" + std::to_string(c.t);
            if (!only.empty() && label.find(only) == std::string::npos) continue;
            failures += run(c, c.heavy ? heavy_calls : calls, seed);
        }
        std::printf(failures == 0 ? "OK every pair route call reproduces its first call's bits\n"
                                  : "FAIL: a pair route's output changed between identical calls\n");
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
