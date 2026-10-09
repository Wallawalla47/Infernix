// The wide route of offloaded_sparse_moe (experts with more than eight columns in a call, design
// §8.5, §13, §16.1 row "wide") against its FP64 W4A4 oracle.
//
// Oracles and criteria (fixed before the first run):
//   C0  quantize_a4_codes against quantize_a4_block: the same scale word and e2m1_x2(code) == c2
//       for every element, on random, zero, saturating and subnormal-scale blocks (exact).
//   C1  The gate/up GEMM: each FP32 accumulator of the production kernel (read through a probe
//       epilogue) against the exact block sum S * 2^-20 of the stored weights and the canonical
//       A4(x) with the expert's gate/up input scale, |acc - S 2^-20| <= 2^-16 * sum |terms| 2^-20
//       (40 FP32 accumulation steps of the tensor core with an 8x margin; a layout error moves an
//       accumulator by whole terms).
//   C2  The down GEMM likewise, against the exact sum over A4(h), where h is formed on the host
//       from the C1 accumulators with the canonical boundaries (y = bf16(acc * alpha),
//       h = swiglu_bf16, A4 with the down input scale): checks the SwiGLU and A4 epilogue and the
//       h plane layout as well.
//   C3  The route's outputs equal bf16_rn(acc_down * alpha_down) of the C2 accumulators bit for bit.
//   C4  End to end against the FP64 chain (FP64 products of the exactly decoded weights and the
//       canonical A4 activations, BF16 at the semantic boundaries, FP64 SiLU): every wide entry's
//       relative L2 error <= 2^-4 (an A4 code of h may flip when a y lands next to a BF16 tie; one
//       flip of a block's largest element moves an output by up to ~3 %, a layout error by ~100 %);
//       the mean relative L2 error <= 1.25 x the canonical narrow arithmetic's + 1e-4; and the
//       share of clean entries (relative L2 <= 2^-8, the output rounding alone, no flipped code)
//       >= 0.9 x the narrow arithmetic's. The narrow arithmetic (the CPU engine on the same entry)
//       is the rounding-order control. Elementwise ulp criteria are not used: FP32 accumulation
//       leaves outputs near zero (|y| << sum |terms|) with large relative but tiny absolute errors.
//   C5  Experts with at most eight columns, and experts whose gate and up input scales differ,
//       keep the narrow route: equal to the CPU engine bit for bit.
//   C6  Placement invariance: outputs bit-identical for records in frames, staged through 3 or 64
//       slots, read zero-copy (bulk copies from mapped host memory, last, since only tests use it),
//       in the double-buffered prefill passes, and on a rerun.
#include "core/vmm_arena.h"
#include "infernix/ops/offloaded_sparse_moe.h"
#include "ops/common/canonical_math.h"
#include "ops/host_parallel.h"
#include "ops/offloaded_moe_fixtures.h"
#include "ops/offloaded_sparse_moe/cpu/w4a16_expert.h"
#include "ops/offloaded_sparse_moe/cpu/w4a4_expert.h"
#include "ops/offloaded_sparse_moe/cuda/wide_expert.cuh"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <random>
#include <vector>

namespace canon    = infernix::ops::canon;
namespace moe      = infernix::ops::offloaded_moe;
namespace wide     = infernix::ops::offloaded_moe::wide;
namespace fixtures = infernix::test::offloaded_moe;
using infernix::DType;
using infernix::Tensor;
using infernix::test::cuda_check;

namespace {

int g_failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_failures;
    }
}

template <class T> T* device_copy(const std::vector<T>& v) {
    T* p = nullptr;
    cuda_check(cudaMalloc(&p, v.size() * sizeof(T)), "cudaMalloc");
    cuda_check(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy");
    return p;
}

template <class T> std::vector<T> host_copy(const T* p, std::size_t n) {
    std::vector<T> v(n);
    cuda_check(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost), "cudaMemcpy");
    return v;
}

// ------------------------------------------------------------------------------------- C0

void test_codes_quantizer() {
    std::mt19937 rng(5);
    std::normal_distribution<float> n(0.0F, 1.0F);
    std::uniform_real_distribution<float> lg(-14.0F, 2.0F);
    long mismatches = 0, blocks = 0;
    for (int trial = 0; trial < 200000; ++trial) {
        std::uint16_t v[16];
        const float magnitude = std::exp2(lg(rng));
        for (auto& x : v) { x = fixtures::to_bf16(n(rng) * magnitude); }
        if (trial % 97 == 0) { std::fill(std::begin(v), std::end(v), std::uint16_t{0}); }
        if (trial % 89 == 0) { v[trial % 16] = fixtures::to_bf16(-60000.0F); }
        const float scale = std::exp2(lg(rng)) * 1e-3F;
        const canon::A4Block block = canon::quantize_a4_block(v, scale);
        const canon::A4Codes codes = canon::quantize_a4_codes(v, scale);
        bool same = block.scale_word == codes.scale_word;
        for (int j = 0; j < 16; ++j) { same = same && canon::e2m1_x2(codes.code[j]) == block.c2[j]; }
        mismatches += same ? 0 : 1;
        ++blocks;
    }
    std::printf("C0 quantize_a4_codes vs quantize_a4_block: %ld mismatching blocks of %ld\n", mismatches, blocks);
    check(mismatches == 0, "C0: the A4 code quantizer equals the canonical block quantizer");
}

// ----------------------------------------------------------------------------------- oracles

// Row `row` of a record matrix (§6.2) decoded: doubled E2M1 values and the scaled E4M3 word.
struct DecodedRow {
    std::int8_t c2[2560];
    std::int32_t scale[160];
};

void decode_row(const std::uint8_t* matrix, int blocks, int row, DecodedRow& out) {
    const std::uint8_t* group = matrix + static_cast<std::size_t>(row / 16) * blocks * moe::kUnitBytes;
    const int r = row % 16, i = r % 8, shift = r < 8 ? 0 : 4;
    for (int b = 0; b < blocks; ++b) {
        const std::uint8_t* unit = group + static_cast<std::size_t>(b) * moe::kUnitBytes;
        for (int k = 0; k < 16; ++k) {
            const unsigned byte = unit[32 * (k / 4) + 4 * i + (k % 4)];
            out.c2[16 * b + k]  = static_cast<std::int8_t>(canon::e2m1_x2((byte >> shift) & 15U));
        }
        out.scale[b] = canon::e4m3_scaled(unit[128 + r]);
    }
}

// Exact S and sum |terms| of one row against one quantized vector.
void exact_sum(const DecodedRow& w, int blocks, const canon::A4Block* a, std::int64_t& s, double& magnitude) {
    s         = 0;
    magnitude = 0.0;
    for (int b = 0; b < blocks; ++b) {
        int p = 0;
        for (int k = 0; k < 16; ++k) { p += w.c2[16 * b + k] * a[b].c2[k]; }
        const std::int64_t term = static_cast<std::int64_t>(p) * w.scale[b] * a[b].scale_scaled;
        s += term;
        magnitude += std::fabs(static_cast<double>(term));
    }
}

std::vector<canon::A4Block> quantize(const std::uint16_t* v, int n, float scale) {
    std::vector<canon::A4Block> out(static_cast<std::size_t>(n / 16));
    for (int b = 0; b < n / 16; ++b) { out[static_cast<std::size_t>(b)] = canon::quantize_a4_block(v + 16 * b, scale); }
    return out;
}

double bf16_value(std::uint16_t h) { return static_cast<double>(canon::bf16_to_f32(h)); }

std::uint16_t bf16_of_double(double v) { return canon::f32_to_bf16_rn(static_cast<float>(v)); }

// The FP64 chain of §16.2 with BF16 at the semantic boundaries; the final output unrounded.
std::vector<double> fp64_expert(const fixtures::Expert& e, const std::uint16_t* x) {
    const std::uint8_t* gate_up = e.record.data();
    const std::uint8_t* down    = e.record.data() + moe::kGateUpBytes;
    const auto xg = quantize(x, moe::kHidden, e.scales.input_gate);
    const auto xu = quantize(x, moe::kHidden, e.scales.input_up);
    std::vector<std::uint16_t> h(moe::kIntermediate);
    DecodedRow row{};
    for (int i = 0; i < moe::kIntermediate; ++i) {
        std::int64_t s = 0;
        double magnitude = 0.0;
        decode_row(gate_up, moe::kGateUpBlocks, 2 * i, row);
        exact_sum(row, moe::kGateUpBlocks, xg.data(), s, magnitude);
        const double g = static_cast<double>(s) * 0x1p-20 * e.scales.alpha_gate;
        decode_row(gate_up, moe::kGateUpBlocks, 2 * i + 1, row);
        exact_sum(row, moe::kGateUpBlocks, xu.data(), s, magnitude);
        const double u  = static_cast<double>(s) * 0x1p-20 * e.scales.alpha_up;
        const double gb = bf16_value(bf16_of_double(g)), ub = bf16_value(bf16_of_double(u));
        h[static_cast<std::size_t>(i)] = bf16_of_double(gb / (1.0 + std::exp(-gb)) * ub);
    }
    const auto hq = quantize(h.data(), moe::kIntermediate, e.scales.input_down);
    std::vector<double> y(moe::kHidden);
    for (int r = 0; r < moe::kHidden; ++r) {
        std::int64_t s = 0;
        double magnitude = 0.0;
        decode_row(down, moe::kDownBlocks, r, row);
        exact_sum(row, moe::kDownBlocks, hq.data(), s, magnitude);
        y[static_cast<std::size_t>(r)] = static_cast<double>(s) * 0x1p-20 * e.scales.alpha_down;
    }
    return y;
}

std::vector<std::uint16_t> cpu_column(const fixtures::Expert& e, const std::uint16_t* x) {
    std::vector<std::uint16_t> y(moe::kHidden);
    const std::uint16_t* xp[1] = {x};
    std::uint16_t* yp[1]       = {y.data()};
    moe::expert_forward(moe::best_cpu_isa(), e.record.data(), e.scales, 1, xp, yp);
    return y;
}

struct Error {
    double relative_l2 = 0.0;
    bool clean         = true; // <= 2^-8: BF16 output rounding only
};

Error error_against(const std::uint16_t* y, const std::vector<double>& reference) {
    double diff = 0.0, norm = 0.0;
    for (std::size_t r = 0; r < reference.size(); ++r) {
        const double got = bf16_value(y[r]), want = reference[r];
        diff += (got - want) * (got - want);
        norm += want * want;
    }
    const double relative = norm > 0.0 ? std::sqrt(diff / norm) : std::sqrt(diff);
    return {relative, relative <= 0x1p-8};
}

// ------------------------------------------------------------------------------------ probes

// Writes each valid FP32 accumulator unscaled: probe[(first + column) * rows + row].
struct ProbeEpilogue {
    float* probe;
    int rows;

    __device__ __forceinline__ void operator()(wide::Accumulators& acc, const wide::TileContext& ctx) const {
        using S = wide::Schedule;
#pragma unroll
        for (int m = 0; m < S::kMmaTokens; ++m) {
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int column = ctx.warp_m * S::kWarpTokens + m * 16 + ctx.group + 8 * half;
                if (column >= ctx.columns) { continue; }
                float* out = probe + static_cast<std::size_t>(ctx.first + column) * rows + ctx.row_block * S::kBlockRows;
#pragma unroll
                for (int r = 0; r < S::kMmaRows; ++r) {
                    const int row = wide::tile_row(ctx.warp_n, r, ctx.tig);
                    out[row]      = acc[m][r][2 * half];
                    out[row + 1]  = acc[m][r][2 * half + 1];
                }
            }
        }
    }
};

// ------------------------------------------------------------------------------------- layer

// Routes `columns` columns to `top_k` distinct experts each, drawn with the given popularity, by
// logits that make exactly those the top-k.
std::vector<float> routing_logits(std::mt19937& rng, int experts, int columns, int top_k,
                                  const std::vector<double>& weight) {
    std::normal_distribution<float> n(0.0F, 1.0F);
    std::vector<float> logits(static_cast<std::size_t>(experts + 1) * columns);
    for (int t = 0; t < columns; ++t) {
        float* column = &logits[static_cast<std::size_t>(t) * (experts + 1)];
        for (int e = 0; e <= experts; ++e) { column[e] = n(rng); }
        std::vector<double> w = weight;
        for (int k = 0; k < top_k; ++k) {
            std::discrete_distribution<int> pick(w.begin(), w.end());
            const int e = pick(rng);
            w[static_cast<std::size_t>(e)] = 0.0;
            column[e]                      = 20.0F + static_cast<float>(top_k - k);
        }
    }
    return logits;
}

void test_layer(const char* name, int experts, int columns, int top_k, const std::vector<double>& weight,
                const std::vector<int>& split_experts, std::uint32_t seed) {
    std::printf("== %s: E=%d T=%d k=%d\n", name, experts, columns, top_k);
    std::mt19937 rng(seed);
    const std::size_t stride = moe::kRecordBytes;
    std::vector<fixtures::Expert> bank;
    std::vector<moe::ExpertScales> scales;
    for (int e = 0; e < experts; ++e) {
        const bool split = std::find(split_experts.begin(), split_experts.end(), e) != split_experts.end();
        bank.push_back(fixtures::random_expert(rng, split));
        scales.push_back(bank.back().scales);
    }
    std::uint8_t* host = nullptr;
    cuda_check(cudaHostAlloc(reinterpret_cast<void**>(&host), stride * experts, cudaHostAllocMapped), "cudaHostAlloc");
    for (int e = 0; e < experts; ++e) { std::memcpy(host + stride * e, bank[static_cast<std::size_t>(e)].record.data(), stride); }
    std::uint8_t* host_device = nullptr;
    cuda_check(cudaHostGetDevicePointer(reinterpret_cast<void**>(&host_device), host, 0), "cudaHostGetDevicePointer");
    std::vector<std::int32_t> frames(static_cast<std::size_t>(experts), -1);
    std::vector<std::uint8_t> frame_bytes;
    int resident = 0;
    for (int e = 0; e < experts; e += 3) {
        frames[static_cast<std::size_t>(e)] = resident++;
        frame_bytes.insert(frame_bytes.end(), bank[static_cast<std::size_t>(e)].record.begin(),
                           bank[static_cast<std::size_t>(e)].record.end());
    }
    auto* d_frame_base = device_copy(frame_bytes);
    auto* d_frames     = device_copy(frames);
    auto* d_scales     = device_copy(scales);

    const auto logits = routing_logits(rng, experts, columns, top_k, weight);
    auto* d_logits = device_copy(logits);
    std::int32_t* d_ids  = nullptr;
    float* d_weights     = nullptr;
    float* d_shared_gate = nullptr;
    cuda_check(cudaMalloc(&d_ids, sizeof(std::int32_t) * top_k * columns), "cudaMalloc");
    cuda_check(cudaMalloc(&d_weights, sizeof(float) * top_k * columns), "cudaMalloc");
    cuda_check(cudaMalloc(&d_shared_gate, sizeof(float) * columns), "cudaMalloc");
    infernix::ops::MoeRouting routing{Tensor(d_ids, DType::I32, {top_k, columns}),
                                    Tensor(d_weights, DType::FP32, {top_k, columns}),
                                    Tensor(d_shared_gate, DType::FP32, {columns})};
    infernix::ops::moe_route(Tensor(d_logits, DType::FP32, {experts + 1, columns}), top_k, routing, nullptr);
    void* d_dispatch = nullptr;
    const int entries = top_k * columns;
    cuda_check(cudaMalloc(&d_dispatch, infernix::ops::moe_dispatch_bytes(experts, entries)), "cudaMalloc");
    auto dispatch = infernix::ops::carve_moe_dispatch(d_dispatch, experts, entries);
    infernix::ops::moe_dispatch(routing, experts, dispatch, nullptr, nullptr);
    cuda_check(cudaDeviceSynchronize(), "dispatch");

    const auto x = fixtures::random_activations(rng, columns);
    auto* d_x    = device_copy(x);
    const auto ids     = host_copy(d_ids, static_cast<std::size_t>(entries));
    const auto offsets = host_copy(dispatch.offsets, static_cast<std::size_t>(experts) + 1);
    const auto order   = host_copy(dispatch.entries, static_cast<std::size_t>(entries));
    std::vector<int> counts(static_cast<std::size_t>(experts));
    int wide_experts = 0, wide_entries = 0, max_count = 0;
    for (int e = 0; e < experts; ++e) {
        counts[static_cast<std::size_t>(e)] = offsets[static_cast<std::size_t>(e) + 1] - offsets[static_cast<std::size_t>(e)];
        max_count = std::max(max_count, counts[static_cast<std::size_t>(e)]);
        if (moe::wide_route(counts[static_cast<std::size_t>(e)], scales[static_cast<std::size_t>(e)])) {
            ++wide_experts;
            wide_entries += counts[static_cast<std::size_t>(e)];
        }
    }
    std::printf("   %d wide experts with %d of %d entries; largest expert %d columns\n", wide_experts, wide_entries,
                entries, max_count);
    // entry (t * k + slot) -> packed row p, and whether its expert is wide.
    std::vector<int> packed(static_cast<std::size_t>(entries));
    for (int p = 0; p < entries; ++p) { packed[static_cast<std::size_t>(order[static_cast<std::size_t>(p)])] = p; }
    auto expert_of = [&](int entry) { return ids[static_cast<std::size_t>(entry)]; };
    auto is_wide   = [&](int entry) {
        const int e = expert_of(entry);
        return moe::wide_route(counts[static_cast<std::size_t>(e)], scales[static_cast<std::size_t>(e)]);
    };
    auto x_of = [&](int entry) { return &x[static_cast<std::size_t>(entry / top_k) * moe::kHidden]; };

    const int max_jobs = std::min(experts, entries);
    std::uint8_t* d_staging = nullptr;
    cuda_check(cudaMalloc(&d_staging, stride * 64), "cudaMalloc");
    void* d_workspace = nullptr;
    cuda_check(cudaMalloc(&d_workspace, infernix::ops::moe_experts_workspace_bytes(max_jobs, entries)), "cudaMalloc");
    std::uint16_t* d_out = nullptr;
    const std::size_t out_elements = static_cast<std::size_t>(moe::kHidden) * entries;
    cuda_check(cudaMalloc(&d_out, out_elements * sizeof(std::uint16_t)), "cudaMalloc");
    cudaStream_t side = nullptr;
    cudaEvent_t events[5] = {};
    cuda_check(cudaStreamCreateWithFlags(&side, cudaStreamNonBlocking), "cudaStreamCreate");
    for (auto& event : events) { cuda_check(cudaEventCreateWithFlags(&event, cudaEventDisableTiming), "event"); }

    struct Config {
        const char* name;
        int slots;
        bool overlap;
        bool fork;
    };
    // The first configuration is the reference (C3-C5) and is rerun before the probes, which reuse
    // its single pass and device-resident records; zero-copy runs last.
    const Config configs[] = {{"64 slots", 64, false, false},
                              {"3 slots", 3, false, false},
                              {"8 slots, prefill passes", 8, true, false},
                              {"64 slots, fork stream", 64, false, true},
                              {"zero-copy", 0, false, false},
                              {"64 slots, rerun", 64, false, false}};
    std::vector<std::uint16_t> first_outputs;
    for (const Config& config : configs) {
        cuda_check(cudaMemset(d_out, 0xFF, out_elements * sizeof(std::uint16_t)), "cudaMemset");
        cuda_check(cudaMemset(d_staging, 0, stride * 64), "cudaMemset");
        infernix::ops::MoeExpertSource source{.frame_base    = d_frame_base,
                                            .frames        = d_frames,
                                            .host_records  = host_device,
                                            .record_stride = stride,
                                            .scales        = d_scales,
                                            .staging_base  = config.slots > 0 ? d_staging : nullptr,
                                            .staging_slots = config.slots};
        if (config.overlap) {
            source.overlap_stream = side;
            for (int i = 0; i < 5; ++i) { source.overlap_events[i] = events[i]; }
        }
        if (config.fork) {
            source.fork_stream    = side;
            source.fork_events[0] = events[0];
            source.fork_events[1] = events[1];
        }
        Tensor tx(d_x, DType::BF16, {moe::kHidden, columns});
        Tensor out(d_out, DType::BF16, {moe::kHidden, entries});
        infernix::ops::moe_experts(tx, dispatch, source, top_k, max_jobs, d_workspace, out, nullptr);
        cuda_check(cudaDeviceSynchronize(), "moe_experts");
        const auto got = host_copy(d_out, out_elements);
        if (first_outputs.empty()) {
            first_outputs = got;
        } else {
            long differing = 0;
            for (std::size_t i = 0; i < got.size(); ++i) { differing += got[i] != first_outputs[i]; }
            std::printf("   C6 %-24s: %ld outputs differ from the first run\n", config.name, differing);
            check(differing == 0, "C6: outputs are bit-identical for every placement and pass layout");
        }
    }

    // C7: the Program's frames (and the prefill stream's lent ring) live in a VMM arena mapped in
    // 64 MiB chunks, so a record can straddle two separately mapped chunks. Frame 0 and staging slot
    // 1 are placed across chunk boundaries; outputs must not change.
    if (infernix::VmmArena::supported(0)) {
        constexpr std::size_t kChunk = 64ULL << 20;
        infernix::VmmArena arena(0, 6 * kChunk, kChunk);
        bool mapped = true;
        for (int c = 0; c < 6; ++c) { mapped = mapped && arena.map_chunk(); }
        check(mapped, "C7: the VMM arena maps its chunks");
        auto* base             = static_cast<std::uint8_t*>(arena.base());
        std::uint8_t* frames_v = base + kChunk - stride / 2;
        std::uint8_t* stage_v  = base + 2 * kChunk - stride - stride / 3;
        cuda_check(cudaMemcpy(frames_v, d_frame_base, frame_bytes.size(), cudaMemcpyDeviceToDevice), "copy frames");
        struct Placement {
            const char* name;
            const std::uint8_t* frames;
            std::uint8_t* staging;
            bool overlap;
        };
        for (const Placement& place : {Placement{"VMM frames across chunks", frames_v, d_staging, false},
                                       Placement{"VMM staging across chunks", d_frame_base, stage_v, false},
                                       Placement{"VMM both, prefill passes", frames_v, stage_v, true}}) {
            cuda_check(cudaMemset(d_out, 0xFF, out_elements * sizeof(std::uint16_t)), "cudaMemset");
            infernix::ops::MoeExpertSource source{.frame_base    = place.frames,
                                                .frames        = d_frames,
                                                .host_records  = host_device,
                                                .record_stride = stride,
                                                .scales        = d_scales,
                                                .staging_base  = place.staging,
                                                .staging_slots = place.overlap ? 8 : 64};
            if (place.overlap) {
                source.overlap_stream = side;
                for (int i = 0; i < 5; ++i) { source.overlap_events[i] = events[i]; }
            }
            Tensor tx(d_x, DType::BF16, {moe::kHidden, columns});
            Tensor out(d_out, DType::BF16, {moe::kHidden, entries});
            infernix::ops::moe_experts(tx, dispatch, source, top_k, max_jobs, d_workspace, out, nullptr);
            cuda_check(cudaDeviceSynchronize(), "moe_experts");
            const auto again = host_copy(d_out, out_elements);
            long differing   = 0;
            for (std::size_t i = 0; i < again.size(); ++i) { differing += again[i] != first_outputs[i]; }
            std::printf("   C7 %-26s: %ld outputs differ from the first run\n", place.name, differing);
            check(differing == 0, "C7: outputs do not depend on records straddling VMM chunks");
        }
        // The probes below reuse the 64-slot run's job records and A4(h) plane: run it again before
        // the arena (which those records would otherwise point into) is released.
        infernix::ops::MoeExpertSource source{.frame_base    = d_frame_base,
                                            .frames        = d_frames,
                                            .host_records  = host_device,
                                            .record_stride = stride,
                                            .scales        = d_scales,
                                            .staging_base  = d_staging,
                                            .staging_slots = 64};
        Tensor tx(d_x, DType::BF16, {moe::kHidden, columns});
        Tensor out(d_out, DType::BF16, {moe::kHidden, entries});
        infernix::ops::moe_experts(tx, dispatch, source, top_k, max_jobs, d_workspace, out, nullptr);
        cuda_check(cudaDeviceSynchronize(), "moe_experts");
    } else {
        std::printf("   C7 skipped: no VMM\n");
    }

    const std::vector<std::uint16_t>& got = first_outputs;
    // C5: narrow-route entries equal the CPU engine.
    {
        std::vector<long> mismatches(static_cast<std::size_t>(entries), 0);
        infernix::test::parallel_ranges(entries, infernix::test::host_thread_count(), [&](std::int64_t b, std::int64_t e) {
            for (std::int64_t entry = b; entry < e; ++entry) {
                if (is_wide(static_cast<int>(entry))) { continue; }
                const auto y = cpu_column(bank[static_cast<std::size_t>(expert_of(static_cast<int>(entry)))],
                                          x_of(static_cast<int>(entry)));
                for (int r = 0; r < moe::kHidden; ++r) {
                    mismatches[static_cast<std::size_t>(entry)] +=
                        y[static_cast<std::size_t>(r)] != got[static_cast<std::size_t>(entry) * moe::kHidden + r];
                }
            }
        });
        long total = 0;
        for (long m : mismatches) { total += m; }
        std::printf("   C5 narrow-route entries: %ld outputs differ from the CPU engine\n", total);
        check(total == 0, "C5: narrow-route experts (n <= 8 or split input scales) equal the CPU engine");
    }
    if (wide_entries == 0) {
        std::printf("   no wide entries\n");
    } else {
        // C1-C3: probes of both GEMMs over the 64-slot rerun's single pass: its job records (frames
        // and staging slots) and A4(h) plane are still in place.
        infernix::ops::MoeExpertSource source{.frame_base    = d_frame_base,
                                            .frames        = d_frames,
                                            .host_records  = host_device,
                                            .record_stride = stride,
                                            .scales        = d_scales,
                                            .staging_base  = d_staging,
                                            .staging_slots = 64};
        const auto layout = wide::carve_experts_workspace(d_workspace, max_jobs, entries);
        Tensor tx(d_x, DType::BF16, {moe::kHidden, columns});
        Tensor out(d_out, DType::BF16, {moe::kHidden, entries});
        const wide::Call call = wide::prepare(tx, dispatch, source, top_k, max_jobs, max_jobs, layout, out, nullptr);
        float* d_probe_gu   = nullptr;
        float* d_probe_down = nullptr;
        cuda_check(cudaMalloc(&d_probe_gu, sizeof(float) * entries * 2 * moe::kIntermediate), "cudaMalloc");
        cuda_check(cudaMalloc(&d_probe_down, sizeof(float) * entries * moe::kHidden), "cudaMalloc");
        wide::launch<wide::Matrix::GateUp>(call, 0, ProbeEpilogue{d_probe_gu, 2 * moe::kIntermediate}, nullptr);
        wide::launch<wide::Matrix::Down>(call, 0, ProbeEpilogue{d_probe_down, moe::kHidden}, nullptr);
        cuda_check(cudaDeviceSynchronize(), "probes");
        const auto probe_gu   = host_copy(d_probe_gu, static_cast<std::size_t>(entries) * 2 * moe::kIntermediate);
        const auto probe_down = host_copy(d_probe_down, static_cast<std::size_t>(entries) * moe::kHidden);

        std::vector<long> c1_bad(static_cast<std::size_t>(entries), 0), c2_bad(static_cast<std::size_t>(entries), 0),
            c3_bad(static_cast<std::size_t>(entries), 0);
        std::vector<double> c1_worst(static_cast<std::size_t>(entries), 0.0), c2_worst(static_cast<std::size_t>(entries), 0.0);
        std::vector<Error> wide_error(static_cast<std::size_t>(entries)), control_error(static_cast<std::size_t>(entries));
        infernix::test::parallel_ranges(entries, infernix::test::host_thread_count(), [&](std::int64_t b, std::int64_t e) {
            DecodedRow row{};
            for (std::int64_t entry64 = b; entry64 < e; ++entry64) {
                const int entry = static_cast<int>(entry64);
                if (!is_wide(entry)) { continue; }
                const auto& expert = bank[static_cast<std::size_t>(expert_of(entry))];
                const auto& sc     = expert.scales;
                const std::size_t p = static_cast<std::size_t>(packed[static_cast<std::size_t>(entry)]);
                const std::uint16_t* xe = x_of(entry);
                // C1
                const auto xq = quantize(xe, moe::kHidden, sc.input_gate);
                std::vector<std::uint16_t> h(moe::kIntermediate);
                for (int r = 0; r < 2 * moe::kIntermediate; ++r) {
                    std::int64_t s = 0;
                    double magnitude = 0.0;
                    decode_row(expert.record.data(), moe::kGateUpBlocks, r, row);
                    exact_sum(row, moe::kGateUpBlocks, xq.data(), s, magnitude);
                    const double acc  = probe_gu[p * 2 * moe::kIntermediate + static_cast<std::size_t>(r)];
                    const double diff = std::fabs(acc - static_cast<double>(s) * 0x1p-20);
                    const double tol  = 0x1p-16 * magnitude * 0x1p-20;
                    c1_bad[static_cast<std::size_t>(entry)] += diff > tol;
                    if (magnitude > 0.0) {
                        c1_worst[static_cast<std::size_t>(entry)] =
                            std::max(c1_worst[static_cast<std::size_t>(entry)], diff / (magnitude * 0x1p-20));
                    }
                }
                // h from the C1 accumulators with the canonical boundaries.
                for (int i = 0; i < moe::kIntermediate; ++i) {
                    const float g = probe_gu[p * 2 * moe::kIntermediate + 2 * static_cast<std::size_t>(i)];
                    const float u = probe_gu[p * 2 * moe::kIntermediate + 2 * static_cast<std::size_t>(i) + 1];
                    h[static_cast<std::size_t>(i)] =
                        canon::swiglu_bf16(canon::f32_to_bf16_rn(canon::mul_rn(g, sc.alpha_gate)),
                                           canon::f32_to_bf16_rn(canon::mul_rn(u, sc.alpha_up)));
                }
                const auto hq = quantize(h.data(), moe::kIntermediate, sc.input_down);
                // C2, C3
                for (int r = 0; r < moe::kHidden; ++r) {
                    std::int64_t s = 0;
                    double magnitude = 0.0;
                    decode_row(expert.record.data() + moe::kGateUpBytes, moe::kDownBlocks, r, row);
                    exact_sum(row, moe::kDownBlocks, hq.data(), s, magnitude);
                    const float acc   = probe_down[p * moe::kHidden + static_cast<std::size_t>(r)];
                    const double diff = std::fabs(static_cast<double>(acc) - static_cast<double>(s) * 0x1p-20);
                    c2_bad[static_cast<std::size_t>(entry)] += diff > 0x1p-16 * magnitude * 0x1p-20;
                    if (magnitude > 0.0) {
                        c2_worst[static_cast<std::size_t>(entry)] =
                            std::max(c2_worst[static_cast<std::size_t>(entry)], diff / (magnitude * 0x1p-20));
                    }
                    const std::uint16_t want = canon::f32_to_bf16_rn(canon::mul_rn(acc, sc.alpha_down));
                    c3_bad[static_cast<std::size_t>(entry)] +=
                        want != got[static_cast<std::size_t>(entry) * moe::kHidden + static_cast<std::size_t>(r)];
                }
                // C4
                const auto reference = fp64_expert(expert, xe);
                wide_error[static_cast<std::size_t>(entry)] =
                    error_against(&got[static_cast<std::size_t>(entry) * moe::kHidden], reference);
                const auto control = cpu_column(expert, xe);
                control_error[static_cast<std::size_t>(entry)] = error_against(control.data(), reference);
            }
        });
        long c1 = 0, c2 = 0, c3 = 0, within = 0, control_within = 0, over = 0;
        double worst1 = 0.0, worst2 = 0.0, sum = 0.0, control_sum = 0.0, worst = 0.0, control_worst = 0.0;
        for (int entry = 0; entry < entries; ++entry) {
            if (!is_wide(entry)) { continue; }
            const auto i = static_cast<std::size_t>(entry);
            c1 += c1_bad[i];
            c2 += c2_bad[i];
            c3 += c3_bad[i];
            worst1 = std::max(worst1, c1_worst[i]);
            worst2 = std::max(worst2, c2_worst[i]);
            sum += wide_error[i].relative_l2;
            control_sum += control_error[i].relative_l2;
            worst         = std::max(worst, wide_error[i].relative_l2);
            control_worst = std::max(control_worst, control_error[i].relative_l2);
            within += wide_error[i].clean;
            control_within += control_error[i].clean;
            over += wide_error[i].relative_l2 > 0x1p-4;
        }
        const double mean = sum / wide_entries, control_mean = control_sum / wide_entries;
        std::printf("   C1 gate/up accumulators: %ld of %ld outside the bound; worst |err| / sum|terms| = %.3g\n", c1,
                    static_cast<long>(wide_entries) * 2 * moe::kIntermediate, worst1);
        std::printf("   C2 down accumulators:    %ld of %ld outside the bound; worst |err| / sum|terms| = %.3g\n", c2,
                    static_cast<long>(wide_entries) * moe::kHidden, worst2);
        std::printf("   C3 down epilogue: %ld outputs differ from bf16(acc * alpha)\n", c3);
        std::printf("   C4 relative L2 vs FP64: wide mean %.3e worst %.3e (%ld entries > 2^-4); canonical narrow mean "
                    "%.3e worst %.3e\n",
                    mean, worst, over, control_mean, control_worst);
        std::printf("   C4 clean entries (relative L2 <= 2^-8): wide %ld, canonical narrow %ld of %d\n", within,
                    control_within, wide_entries);
        check(c1 == 0, "C1: gate/up accumulators within the FP32 accumulation bound of the exact sums");
        check(c2 == 0, "C2: down accumulators within the bound of the exact sums over the canonical A4(h)");
        check(c3 == 0, "C3: wide outputs are bf16_rn(acc * alpha_down)");
        check(over == 0, "C4: every wide entry within 2^-4 relative L2 of the FP64 chain");
        check(mean <= 1.25 * control_mean + 1e-4, "C4: mean error within the rounding-order control");
        check(static_cast<double>(within) >= 0.9 * static_cast<double>(control_within),
              "C4: clean entries no fewer than 0.9 x the rounding-order control's");
        cudaFree(d_probe_gu);
        cudaFree(d_probe_down);
    }

    for (auto event : events) { cudaEventDestroy(event); }
    cudaStreamDestroy(side);
    cudaFree(d_out);
    cudaFree(d_workspace);
    cudaFree(d_staging);
    cudaFree(d_x);
    cudaFree(d_dispatch);
    cudaFree(d_shared_gate);
    cudaFree(d_weights);
    cudaFree(d_ids);
    cudaFree(d_logits);
    cudaFree(d_scales);
    cudaFree(d_frames);
    cudaFree(d_frame_base);
    cudaFreeHost(host);
}

// ------------------------------------------------------------------------------ W4A16 layer
//
// The W4A16 wide route (§16.2.1, wide_expert_a16.cuh), criteria fixed before the first run:
//   A1  End to end against the FP64 chain: weights exactly decoded (c2 / 2 x E4M3 scale x m), the BF16
//       x unencoded, BF16 at the semantic boundaries (y_gate, y_up, h, y), FP64 SiLU. Every wide entry's
//       relative L2 error <= 2^-6, and the mean <= 1.25 x the exact narrow arithmetic's (the CPU engine
//       on the same entry, the rounding-order control) + 1e-4.
//   A2  Narrow experts (at most eight columns): equal to the CPU engine bit for bit.
//   A3  Placement invariance: outputs bit-identical for every staging layout, the fork stream,
//       zero-copy records and a rerun.
void test_layer_a16(const char* name, int experts, int columns, int top_k, const std::vector<double>& weight,
                    std::uint32_t seed) {
    std::printf("== W4A16 %s: E=%d T=%d k=%d\n", name, experts, columns, top_k);
    std::mt19937 rng(seed);
    const std::size_t stride = moe::kRecordBytes;
    std::vector<fixtures::Expert> bank;
    std::vector<moe::ExpertScales> scales;
    for (int e = 0; e < experts; ++e) {
        bank.push_back(fixtures::random_a16_expert(rng));
        scales.push_back(bank.back().scales);
    }
    std::uint8_t* host = nullptr;
    cuda_check(cudaHostAlloc(reinterpret_cast<void**>(&host), stride * experts, cudaHostAllocMapped), "cudaHostAlloc");
    for (int e = 0; e < experts; ++e) { std::memcpy(host + stride * e, bank[static_cast<std::size_t>(e)].record.data(), stride); }
    std::uint8_t* host_device = nullptr;
    cuda_check(cudaHostGetDevicePointer(reinterpret_cast<void**>(&host_device), host, 0), "cudaHostGetDevicePointer");
    std::vector<std::int32_t> frames(static_cast<std::size_t>(experts), -1);
    std::vector<std::uint8_t> frame_bytes;
    int resident = 0;
    for (int e = 0; e < experts; e += 3) {
        frames[static_cast<std::size_t>(e)] = resident++;
        frame_bytes.insert(frame_bytes.end(), bank[static_cast<std::size_t>(e)].record.begin(),
                           bank[static_cast<std::size_t>(e)].record.end());
    }
    auto* d_frame_base = device_copy(frame_bytes);
    auto* d_frames     = device_copy(frames);
    auto* d_scales     = device_copy(scales);
    const auto logits  = routing_logits(rng, experts, columns, top_k, weight);
    auto* d_logits     = device_copy(logits);
    std::int32_t* d_ids  = nullptr;
    float* d_weights     = nullptr;
    float* d_shared_gate = nullptr;
    cuda_check(cudaMalloc(&d_ids, sizeof(std::int32_t) * top_k * columns), "cudaMalloc");
    cuda_check(cudaMalloc(&d_weights, sizeof(float) * top_k * columns), "cudaMalloc");
    cuda_check(cudaMalloc(&d_shared_gate, sizeof(float) * columns), "cudaMalloc");
    infernix::ops::MoeRouting routing{Tensor(d_ids, DType::I32, {top_k, columns}),
                                    Tensor(d_weights, DType::FP32, {top_k, columns}),
                                    Tensor(d_shared_gate, DType::FP32, {columns})};
    infernix::ops::moe_route(Tensor(d_logits, DType::FP32, {experts + 1, columns}), top_k, routing, nullptr);
    void* d_dispatch = nullptr;
    const int entries = top_k * columns;
    cuda_check(cudaMalloc(&d_dispatch, infernix::ops::moe_dispatch_bytes(experts, entries)), "cudaMalloc");
    auto dispatch = infernix::ops::carve_moe_dispatch(d_dispatch, experts, entries);
    infernix::ops::moe_dispatch(routing, experts, dispatch, nullptr, nullptr);
    cuda_check(cudaDeviceSynchronize(), "dispatch");
    const auto x   = fixtures::wide_range_activations(rng, columns);
    auto* d_x      = device_copy(x);
    const auto ids = host_copy(d_ids, static_cast<std::size_t>(entries));
    const auto offsets = host_copy(dispatch.offsets, static_cast<std::size_t>(experts) + 1);
    std::vector<int> counts(static_cast<std::size_t>(experts));
    int wide_entries = 0;
    for (int e = 0; e < experts; ++e) {
        counts[static_cast<std::size_t>(e)] = offsets[static_cast<std::size_t>(e) + 1] - offsets[static_cast<std::size_t>(e)];
        if (moe::wide_route(counts[static_cast<std::size_t>(e)], scales[static_cast<std::size_t>(e)])) {
            wide_entries += counts[static_cast<std::size_t>(e)];
        }
    }
    std::printf("   %d of %d entries on the wide route\n", wide_entries, entries);

    const int max_jobs = std::min(experts, entries);
    std::uint8_t* d_staging = nullptr;
    cuda_check(cudaMalloc(&d_staging, stride * 64), "cudaMalloc");
    void* d_workspace = nullptr;
    cuda_check(cudaMalloc(&d_workspace, infernix::ops::moe_experts_workspace_bytes(max_jobs, entries)), "cudaMalloc");
    std::uint16_t* d_out = nullptr;
    const std::size_t out_elements = static_cast<std::size_t>(moe::kHidden) * entries;
    cuda_check(cudaMalloc(&d_out, out_elements * sizeof(std::uint16_t)), "cudaMalloc");
    cudaStream_t side = nullptr;
    cudaEvent_t events[5] = {};
    cuda_check(cudaStreamCreateWithFlags(&side, cudaStreamNonBlocking), "cudaStreamCreate");
    for (auto& event : events) { cuda_check(cudaEventCreateWithFlags(&event, cudaEventDisableTiming), "event"); }
    struct Config {
        const char* name;
        int slots;
        bool overlap;
        bool fork;
    };
    const Config configs[] = {{"64 slots", 64, false, false},   {"3 slots", 3, false, false},
                              {"8 slots, prefill passes", 8, true, false}, {"64 slots, fork stream", 64, false, true},
                              {"zero-copy", 0, false, false},  {"64 slots, rerun", 64, false, false}};
    std::vector<std::uint16_t> first_outputs;
    for (const Config& config : configs) {
        cuda_check(cudaMemset(d_out, 0xFF, out_elements * sizeof(std::uint16_t)), "cudaMemset");
        cuda_check(cudaMemset(d_staging, 0, stride * 64), "cudaMemset");
        infernix::ops::MoeExpertSource source{.frame_base    = d_frame_base,
                                            .frames        = d_frames,
                                            .host_records  = host_device,
                                            .record_stride = stride,
                                            .scales        = d_scales,
                                            .activation    = moe::ExpertActivation::kA16,
                                            .staging_base  = config.slots > 0 ? d_staging : nullptr,
                                            .staging_slots = config.slots};
        if (config.overlap) {
            source.overlap_stream = side;
            for (int i = 0; i < 5; ++i) { source.overlap_events[i] = events[i]; }
        }
        if (config.fork) {
            source.fork_stream    = side;
            source.fork_events[0] = events[0];
            source.fork_events[1] = events[1];
        }
        Tensor tx(d_x, DType::BF16, {moe::kHidden, columns});
        Tensor out(d_out, DType::BF16, {moe::kHidden, entries});
        infernix::ops::moe_experts(tx, dispatch, source, top_k, max_jobs, d_workspace, out, nullptr);
        cuda_check(cudaDeviceSynchronize(), "moe_experts");
        const auto got = host_copy(d_out, out_elements);
        if (first_outputs.empty()) {
            first_outputs = got;
        } else {
            long differing = 0;
            for (std::size_t i = 0; i < got.size(); ++i) { differing += got[i] != first_outputs[i]; }
            std::printf("   A3 %-24s: %ld outputs differ from the first run\n", config.name, differing);
            check(differing == 0, "A3: outputs are bit-identical for every placement and pass layout");
        }
    }

    // A1 / A2: per entry, the FP64 chain and the CPU engine (the exact narrow arithmetic).
    const auto weight_of = [](const std::uint8_t* matrix, int blocks, int row, int k) {
        const int rg = row / 16, r = row % 16, b = k / 16, kk = k % 16;
        const std::uint8_t* unit = matrix + (static_cast<std::size_t>(rg) * blocks + b) * moe::kUnitBytes;
        const std::uint8_t byte  = unit[32 * (kk / 4) + 4 * (r % 8) + kk % 4];
        return canon::e2m1_x2(r < 8 ? (byte & 15U) : (byte >> 4)) * 0.5 * static_cast<double>(canon::e4m3_value(unit[128 + r]));
    };
    const auto bf16 = [](double v) { return static_cast<double>(canon::bf16_to_f32(canon::f32_to_bf16_rn(static_cast<float>(v)))); };
    double wide_error_sum = 0, narrow_error_sum = 0, worst = 0;
    int wide_checked = 0, wide_bad = 0;
    long narrow_mismatch = 0;
    std::vector<std::vector<double>> w_gate_up, w_down; // decoded once per expert, lazily
    for (int entry = 0; entry < entries; ++entry) {
        const int e = ids[static_cast<std::size_t>(entry)];
        const fixtures::Expert& expert = bank[static_cast<std::size_t>(e)];
        const std::uint16_t* xc = &x[static_cast<std::size_t>(entry / top_k) * moe::kHidden];
        std::vector<std::uint16_t> cpu(moe::kHidden);
        {
            const std::uint16_t* xp[1] = {xc};
            std::uint16_t* yp[1]       = {cpu.data()};
            moe::expert_forward_a16(moe::best_cpu_isa(), expert.record.data(), expert.scales, 1, xp, yp);
        }
        const std::uint16_t* got = &first_outputs[static_cast<std::size_t>(entry) * moe::kHidden];
        if (!moe::wide_route(counts[static_cast<std::size_t>(e)], scales[static_cast<std::size_t>(e)])) {
            narrow_mismatch += std::memcmp(got, cpu.data(), moe::kHidden * 2) != 0;
            continue;
        }
        if (wide_checked >= 160) { continue; } // FP64 chains are costly; a sample of wide entries
        const std::uint8_t* gu = expert.record.data();
        const std::uint8_t* dn = expert.record.data() + moe::kGateUpBytes;
        std::vector<double> h(moe::kIntermediate);
        for (int i = 0; i < moe::kIntermediate; ++i) {
            double g = 0, u = 0;
            for (int k = 0; k < moe::kHidden; ++k) {
                const double xv = canon::bf16_to_f32(xc[k]);
                g += weight_of(gu, moe::kGateUpBlocks, 2 * i, k) * xv;
                u += weight_of(gu, moe::kGateUpBlocks, 2 * i + 1, k) * xv;
            }
            const double yg = bf16(g * expert.scales.alpha_gate), yu = bf16(u * expert.scales.alpha_up);
            h[i] = bf16(yg / (1.0 + std::exp(-yg)) * yu);
        }
        double err_wide = 0, err_cpu = 0, norm = 0;
        for (int row = 0; row < moe::kHidden; ++row) {
            double acc = 0;
            for (int k = 0; k < moe::kIntermediate; ++k) { acc += weight_of(dn, moe::kDownBlocks, row, k) * h[static_cast<std::size_t>(k)]; }
            const double ref = bf16(acc * expert.scales.alpha_down);
            const double a = canon::bf16_to_f32(got[row]), c = canon::bf16_to_f32(cpu[static_cast<std::size_t>(row)]);
            err_wide += (a - ref) * (a - ref);
            err_cpu += (c - ref) * (c - ref);
            norm += ref * ref;
        }
        const double rw = std::sqrt(err_wide / std::max(norm, 1e-300)), rc = std::sqrt(err_cpu / std::max(norm, 1e-300));
        wide_error_sum += rw;
        narrow_error_sum += rc;
        worst = std::max(worst, rw);
        wide_bad += rw > std::ldexp(1.0, -6);
        ++wide_checked;
    }
    const double mean_wide = wide_checked ? wide_error_sum / wide_checked : 0;
    const double mean_cpu  = wide_checked ? narrow_error_sum / wide_checked : 0;
    std::printf("   A1 %d wide entries: relative L2 vs FP64 mean %.3g (narrow arithmetic %.3g), worst %.3g, %d over 2^-6\n",
                wide_checked, mean_wide, mean_cpu, worst, wide_bad);
    std::printf("   A2 narrow entries differing from the CPU engine: %ld\n", narrow_mismatch);
    check(wide_bad == 0, "A1: every wide entry within relative L2 2^-6 of the FP64 chain");
    check(mean_wide <= 1.25 * mean_cpu + 1e-4, "A1: the wide route's mean error matches the exact arithmetic's");
    check(narrow_mismatch == 0, "A2: narrow W4A16 experts equal the CPU engine bit for bit");
    for (auto event : events) { cudaEventDestroy(event); }
    cudaStreamDestroy(side);
    cudaFree(d_out);
    cudaFree(d_workspace);
    cudaFree(d_staging);
    cudaFree(d_x);
    cudaFree(d_dispatch);
    cudaFree(d_shared_gate);
    cudaFree(d_weights);
    cudaFree(d_ids);
    cudaFree(d_logits);
    cudaFree(d_scales);
    cudaFree(d_frames);
    cudaFree(d_frame_base);
    cudaFreeHost(host);
}

} // namespace

int main() {
    // Unbuffered, so a failing case's last progress line survives a crash.
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    test_codes_quantizer();
    if (infernix::test::cuda_unavailable()) {
        std::printf("SKIP: no usable CUDA device\n");
        return g_failures == 0 ? 77 : 1;
    }
    try {
        // Popular experts over several 64-column tiles, medium ones in one, rare ones (n <= 8) and
        // two with split input scales on the narrow route.
        std::vector<double> popularity(20, 3.0);
        popularity[0] = popularity[1] = 8.0;
        for (int e = 16; e < 20; ++e) { popularity[static_cast<std::size_t>(e)] = 0.05; }
        test_layer("prefill mix", 20, 300, 4, popularity, {5, 12}, 23);
        // Boundaries: every expert at exactly 9 columns (the narrowest wide expert), and 65 columns
        // (a full tile plus one).
        test_layer("n = 9", 4, 9, 4, std::vector<double>(4, 1.0), {}, 29);
        test_layer("n = 65", 2, 65, 2, std::vector<double>(2, 1.0), {}, 31);
        // A verification-width call: 10 columns at top-10 over 12 experts.
        test_layer("T = 10", 12, 10, 10, std::vector<double>(12, 1.0), {}, 37);
        // W4A16 (§16.2.1): the same prefill mix, the 9- and 65-column boundaries.
        test_layer_a16("prefill mix", 20, 300, 4, popularity, 41);
        test_layer_a16("n = 9", 4, 9, 4, std::vector<double>(4, 1.0), 43);
        test_layer_a16("n = 65", 2, 65, 2, std::vector<double>(2, 1.0), 47);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
    if (g_failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("all offloaded_moe wide-route checks passed\n");
    return 0;
}
