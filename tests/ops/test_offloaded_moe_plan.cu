// The CPU plan of moe_route_dispatch_plan (docs/maintainer/qwen3_8-flash-next-design.md §10.3,
// §19.3.5 S3; moe_layer.cu cpu_plan_body) at Qwen3.5-122B-A10B's routing (256 experts, top-8,
// the shared expert as the last job, BF16 router logits, H 3072) and Qwen3.8-Flash-Next's (512
// experts, top-10, FP32 logits, H 2560), at decode and verification widths (1 to 32 columns).
//
// Each case draws a routing, a residency (frames, streamed records), an optional host table with
// SSD-only experts, tail skip, a column mask and the channel's split parameters (caps, divisors,
// widths, prefill assist). The plan runs twice on the same inputs into buffers prefilled with a
// pattern, so every byte either run writes or leaves is compared.
//
// Oracle: the plan's definition evaluated on the host from the call's own dispatch (the order of
// entries within an expert is unspecified, so each run is checked against the dispatch it produced):
// of the M host-memory misses, the SSD-only misses of at most max_job_columns columns first up to the
// cap, then want = min(cap - n, M - M / d) host-memory ones, the fewest-column ones first and in job
// order within a width; the chosen jobs flagged 1 (tail-skipped ones 2), listed in call->job and the
// request (expert, record, width, columns by compact index); the x columns they read published
// compacted in column order; the request's header and sequence. Every byte of the mapped request,
// the published x columns, cpu_flags and the call's bookkeeping must equal it exactly, in both runs.
// Where both runs produced the same dispatch (always at one column), their bytes are compared directly.

#include "ops/op_family.h"
#include "infernix/ops/offloaded_sparse_moe.h"
#include "ops/offloaded_sparse_moe/cpu/miss_request.h"
#include "ops/offloaded_sparse_moe/cuda/wide_expert.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace moe = infernix::ops::offloaded_moe;
using infernix::DType;
using infernix::Tensor;
using infernix::test::cuda_check;

namespace {

int g_failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++g_failures;
    }
}

template <class T> T* device_copy(const std::vector<T>& v) {
    T* p = nullptr;
    cuda_check(cudaMalloc(&p, std::max<std::size_t>(1, v.size()) * sizeof(T)), "cudaMalloc");
    if (!v.empty()) { cuda_check(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy"); }
    return p;
}

template <class T> std::vector<T> host_copy(const T* p, std::size_t n) {
    std::vector<T> v(n);
    if (n > 0) { cuda_check(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost), "cudaMemcpy"); }
    return v;
}

// moe_layer.cu's CpuCall (wide_expert.h kCpuCallBytes: two words and kMaxCpuJobs job indices).
struct PlanCall {
    std::int32_t pending;
    std::int32_t jobs;
    std::int32_t job[moe::kMaxCpuJobs];
};
static_assert(sizeof(PlanCall) <= moe::wide::kCpuCallBytes, "the call's CPU bookkeeping fits its region");

constexpr std::uint8_t kRequestFill = 0xA5, kXFill = 0x5A, kFlagFill = 0x77, kCallFill = 0x33;

struct Geometry {
    const char* name;
    int experts;
    int top_k;
    bool shared_job;
    infernix::ops::RouterLogits precision;
    moe::ExpertShape shape;
    moe::ExpertActivation activation;
};

// One case's inputs besides the logits (host copies the oracle reads).
struct Case {
    std::vector<std::int32_t> frames;     // [E + 1]
    std::vector<std::int32_t> prefetched; // [E + 1], or empty
    std::vector<std::uint64_t> table;     // [E + 1] host-record addresses (0: SSD-only), or empty
    int tail_skip_rank = 0;
    bool divisor_word      = false; // the channel reads the one-column divisor from a device word
    int one_divisor_device = 0;     // its value (0 or less: pcie_divisor)
    infernix::ops::MoeCpuChannel channel;
    std::vector<std::int32_t> valid; // the column mask's rows, or empty
    int mask_width = 0;
};

// What one run left in the buffers.
struct Run {
    std::vector<std::uint8_t> request, x, flags, call, routing;
    std::vector<std::int32_t> counts, offsets, entries, jobs;
    std::int32_t job_count = 0;
    std::uint32_t sequence_before = 0;
};

// The plan's definition on the host, from the run's own dispatch: the expected request, x, flags and
// call bytes, starting from the fill patterns.
void oracle(const Geometry& g, const Case& k, int columns, int max_jobs, const std::vector<std::uint16_t>& x,
            const Run& run, Run& want) {
    const int E = g.experts, K = g.top_k;
    const int shared_expert = g.shared_job ? E : -1;
    const int shared_first  = g.shared_job ? columns * K : 0x7FFFFFFF;
    const auto& ch          = k.channel;
    want.request.assign(sizeof(moe::MissRequest), kRequestFill);
    want.x.assign(run.x.size(), kXFill);
    std::vector<std::int32_t> flags(static_cast<std::size_t>(max_jobs), 0);
    PlanCall call;
    std::memset(&call, kCallFill, sizeof(call));
    const int jobs = std::min(run.job_count, max_jobs);
    const int max_width = ch.max_job_columns;
    const int cap = ch.wide_from > 0 && columns >= ch.wide_from ? ch.wide_jobs : ch.max_jobs;
    // Each job's candidacy: category 0 (none), w (SSD-only, width w) or 8 + w (host memory).
    std::vector<int> cat(static_cast<std::size_t>(jobs), 0);
    int misses = 0;
    for (int j = 0; j < jobs; ++j) {
        const int e = run.jobs[j];
        const bool resident = e == shared_expert || k.frames[e] >= 0 || (!k.prefetched.empty() && k.prefetched[e] >= 0);
        if (resident) { continue; }
        const int first = run.offsets[e], count = run.offsets[e + 1] - first;
        if (k.tail_skip_rank > 0 && count <= moe::kMaxColumns) {
            bool tail = true;
            for (int c = 0; c < count; ++c) {
                const int entry = run.entries[first + c];
                tail = tail && entry < shared_first && entry % K >= k.tail_skip_rank;
            }
            if (tail) {
                flags[j] = 2;
                continue;
            }
        }
        const bool ssd = !k.table.empty() && k.table[e] == 0;
        misses += ssd ? 0 : 1;
        if (count <= max_width) { cat[j] = (ssd ? 0 : moe::kMaxCpuColumns) + count; }
    }
    std::vector<int> chosen; // jobs in slot order
    for (int w = 1; w <= moe::kMaxCpuColumns; ++w) {
        for (int j = 0; j < jobs; ++j) {
            if (cat[j] == w && static_cast<int>(chosen.size()) < cap) { chosen.push_back(j); }
        }
    }
    const int n       = static_cast<int>(chosen.size());
    const int one     = k.divisor_word && columns == 1 ? k.one_divisor_device : ch.one_column_divisor;
    // The link keeps M / one in a one-column call (one > 0), else (M * pcie_share) >> 8 with a share, else
    // M / pcie_divisor (none for 0).
    const int kept    = columns == 1 && one > 0 ? misses / one
                        : ch.pcie_share > 0     ? (misses * ch.pcie_share) >> 8
                        : ch.pcie_divisor > 0   ? misses / ch.pcie_divisor
                                                : 0;
    const int quota   = std::min(cap - n, misses - kept);
    for (int w = 1; w <= moe::kMaxCpuColumns; ++w) {
        for (int j = 0; j < jobs; ++j) {
            if (cat[j] == moe::kMaxCpuColumns + w && static_cast<int>(chosen.size()) - n < quota) { chosen.push_back(j); }
        }
    }
    call.jobs = static_cast<std::int32_t>(chosen.size());
    if (chosen.empty()) {
        call.pending = 0;
    } else {
        auto* request = reinterpret_cast<moe::MissRequest*>(want.request.data());
        const auto column_of = [&](int entry) { return moe::wide::entry_column(entry, K, shared_first); };
        std::vector<int> used(static_cast<std::size_t>(columns), 0);
        for (const int j : chosen) {
            const int e = run.jobs[j];
            for (int c = run.offsets[e]; c < run.offsets[e + 1]; ++c) { used[column_of(run.entries[c])] = 1; }
        }
        std::vector<int> compact(static_cast<std::size_t>(columns), -1), published;
        for (int t = 0; t < columns; ++t) {
            if (used[t] != 0) {
                compact[t] = static_cast<int>(published.size());
                published.push_back(t);
            }
        }
        for (std::size_t s = 0; s < chosen.size(); ++s) {
            const int j = chosen[s], e = run.jobs[j];
            const int first = run.offsets[e], width = run.offsets[e + 1] - first;
            flags[j]           = 1;
            call.job[s]        = j;
            request->expert[s] = e;
            request->record[s] = k.table.empty() ? 0 : k.table[e];
            request->ncols[s]  = width;
            for (int c = 0; c < width; ++c) { request->column[s][c] = compact[column_of(run.entries[first + c])]; }
        }
        request->layer  = ch.layer;
        request->jobs   = static_cast<std::int32_t>(chosen.size());
        request->tiered = k.table.empty() ? 0 : 1;
        std::uint32_t sequence = run.sequence_before + 1U;
        if (sequence == 0) { sequence = run.sequence_before + 2U; }
        request->sequence = sequence;
        call.pending      = static_cast<std::int32_t>(sequence);
        const std::size_t row = static_cast<std::size_t>(g.shape.hidden) * sizeof(std::uint16_t);
        for (std::size_t c = 0; c < published.size(); ++c) {
            std::memcpy(want.x.data() + c * row, reinterpret_cast<const std::uint8_t*>(x.data()) + published[c] * row, row);
        }
    }
    want.flags.resize(flags.size() * sizeof(std::int32_t));
    std::memcpy(want.flags.data(), flags.data(), want.flags.size());
    want.call.assign(moe::wide::kCpuCallBytes, kCallFill);
    std::memcpy(want.call.data(), &call, sizeof(call));
}

void test_geometry(const Geometry& g, int columns, int cases, std::uint32_t seed) {
    std::mt19937 rng(seed);
    const int E = g.experts, K = g.top_k, H = g.shape.hidden;
    const int routed = K * columns, entries = routed + (g.shared_job ? columns : 0);
    const int dispatch_experts = E + (g.shared_job ? 1 : 0);
    const int max_jobs = std::min(E, routed) + (g.shared_job ? 1 : 0);
    // Device inputs and outputs.
    float* d_logits = nullptr;
    cuda_check(cudaMalloc(&d_logits, sizeof(float) * (E + 1) * columns), "cudaMalloc");
    std::int32_t* d_ids = nullptr;
    float *d_weights = nullptr, *d_gate = nullptr;
    cuda_check(cudaMalloc(&d_ids, sizeof(std::int32_t) * routed), "cudaMalloc");
    cuda_check(cudaMalloc(&d_weights, sizeof(float) * routed), "cudaMalloc");
    cuda_check(cudaMalloc(&d_gate, sizeof(float) * columns), "cudaMalloc");
    void* d_dispatch = nullptr;
    cuda_check(cudaMalloc(&d_dispatch, infernix::ops::moe_dispatch_bytes(dispatch_experts, entries)), "cudaMalloc");
    auto dispatch = infernix::ops::carve_moe_dispatch(d_dispatch, dispatch_experts, entries);
    const std::size_t ws_bytes = infernix::ops::moe_experts_workspace_bytes(g.shape, max_jobs, entries, columns, g.activation);
    void* d_workspace = nullptr;
    cuda_check(cudaMalloc(&d_workspace, ws_bytes), "cudaMalloc");
    const auto layout = moe::wide::carve_experts_workspace(d_workspace, g.shape, max_jobs, entries, columns, g.activation,
                                                           moe::WideA16::kExact);
    std::uint16_t* d_x = nullptr;
    cuda_check(cudaMalloc(&d_x, sizeof(std::uint16_t) * H * columns), "cudaMalloc");
    std::uint8_t* d_dummy = nullptr; // frame base and shared record: addresses only, never read by the plan
    cuda_check(cudaMalloc(&d_dummy, 256), "cudaMalloc");
    std::int32_t *d_frames = nullptr, *d_prefetched = nullptr, *d_divisor = nullptr, *d_valid = nullptr;
    std::uint64_t* d_table = nullptr;
    cuda_check(cudaMalloc(&d_frames, sizeof(std::int32_t) * (E + 1)), "cudaMalloc");
    cuda_check(cudaMalloc(&d_prefetched, sizeof(std::int32_t) * (E + 1)), "cudaMalloc");
    cuda_check(cudaMalloc(&d_table, sizeof(std::uint64_t) * (E + 1)), "cudaMalloc");
    cuda_check(cudaMalloc(&d_divisor, sizeof(std::int32_t)), "cudaMalloc");
    cuda_check(cudaMalloc(&d_valid, sizeof(std::int32_t) * columns), "cudaMalloc");
    // The channel's mapped buffers (as CpuMissService allocates them) and its device counter.
    const auto mapped = [](std::size_t bytes) {
        void* p = nullptr;
        cuda_check(cudaHostAlloc(&p, bytes, cudaHostAllocMapped | cudaHostAllocPortable), "cudaHostAlloc");
        std::memset(p, 0, bytes);
        return p;
    };
    auto* request = static_cast<moe::MissRequest*>(mapped(sizeof(moe::MissRequest)));
    const std::size_t x_bytes = sizeof(std::uint16_t) * H * columns; // at most `columns` columns are published
    auto* cx   = static_cast<std::uint16_t*>(mapped(sizeof(std::uint16_t) * H * moe::kMaxCpuXColumns));
    auto* cy   = static_cast<std::uint16_t*>(mapped(64));
    auto* done = static_cast<std::uint32_t*>(mapped(64));
    std::uint32_t* d_sequence = nullptr;
    cuda_check(cudaMalloc(&d_sequence, sizeof(std::uint32_t)), "cudaMalloc");
    // Start near the wrap, so that the counter's skip of 0 is crossed.
    const std::uint32_t start = 0xFFFFFFFFU - static_cast<std::uint32_t>(cases / 2);
    cuda_check(cudaMemcpy(d_sequence, &start, sizeof(start), cudaMemcpyHostToDevice), "cudaMemcpy");

    std::normal_distribution<float> normal(0.0F, 1.0F);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    const auto pick = [&](int lo, int hi) { return std::uniform_int_distribution<int>(lo, hi)(rng); };
    int planned_cases = 0, published_cases = 0, same_dispatch = 0, multi_chunk = 0, ssd_cases = 0;
    for (int index = 0; index < cases; ++index) {
        // Logits: random, or (every eighth case) each column preferring its own k experts, so that the
        // call has up to T * k distinct jobs (Flash-Next at 32 columns: 320, two ranking chunks).
        std::vector<float> logits(static_cast<std::size_t>(E + 1) * columns);
        for (auto& v : logits) { v = normal(rng); }
        if (index % 8 == 7) {
            for (int t = 0; t < columns; ++t) {
                for (int s = 0; s < K; ++s) {
                    logits[static_cast<std::size_t>(t) * (E + 1) + (t * K + s) % E] += 8.0F + static_cast<float>(s);
                }
            }
        }
        cuda_check(cudaMemcpy(d_logits, logits.data(), logits.size() * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy");
        std::vector<std::uint16_t> x(static_cast<std::size_t>(H) * columns);
        for (auto& v : x) { v = static_cast<std::uint16_t>(rng()); }
        cuda_check(cudaMemcpy(d_x, x.data(), x.size() * sizeof(std::uint16_t), cudaMemcpyHostToDevice), "cudaMemcpy");

        Case k;
        const double resident = std::vector<double>{0.0, 0.3, 0.7}[static_cast<std::size_t>(index % 3)];
        k.frames.assign(static_cast<std::size_t>(E + 1), -1);
        int frame = 0;
        for (int e = 0; e < E; ++e) { k.frames[e] = unit(rng) < resident ? frame++ : -1; }
        if (unit(rng) < 0.25) {
            k.prefetched.assign(static_cast<std::size_t>(E + 1), -1);
            for (int e = 0; e < E; ++e) { k.prefetched[e] = unit(rng) < 0.1 ? pick(0, 15) : -1; }
        }
        if (unit(rng) < 0.4) {
            k.table.assign(static_cast<std::size_t>(E + 1), 0);
            for (int e = 0; e <= E; ++e) { k.table[e] = unit(rng) < 0.3 ? 0 : 0x10000ULL + 4096ULL * e; }
            ++ssd_cases;
        }
        k.tail_skip_rank = unit(rng) < 0.2 ? pick(1, K - 1) : 0;
        auto& ch         = k.channel;
        ch.request       = request;
        ch.x             = cx;
        ch.y             = cy;
        ch.done          = done;
        ch.sequence      = d_sequence;
        ch.layer         = pick(0, 47);
        ch.max_columns   = moe::kMaxCpuCallColumns;
        ch.max_jobs      = unit(rng) < 0.15 ? moe::kMaxCpuJobs : pick(1, 32);
        ch.pcie_divisor  = pick(0, 4);
        ch.pcie_share    = unit(rng) < 0.3 ? pick(1, 255) : 0; // the fractional link share (INFERNIX_TMP_LINK_SHARE)
        ch.one_column_divisor = pick(0, 2);
        if (unit(rng) < 0.3) {
            k.divisor_word       = true;
            k.one_divisor_device = pick(-1, 3);
            cuda_check(cudaMemcpy(d_divisor, &k.one_divisor_device, sizeof(std::int32_t), cudaMemcpyHostToDevice), "cudaMemcpy");
            ch.one_column_divisor_device = d_divisor;
        }
        ch.max_job_columns = pick(1, moe::kMaxCpuColumns);
        if (unit(rng) < 0.2) {
            ch.wide_from = pick(1, columns);
            ch.wide_jobs = pick(1, moe::kMaxCpuJobs);
        }
        if (g.shared_job && columns > 1 && unit(rng) < 0.3) {
            k.mask_width = columns <= 8 ? columns : 8;
            k.valid.resize(static_cast<std::size_t>(columns / k.mask_width));
            for (auto& v : k.valid) { v = pick(1, k.mask_width); }
            cuda_check(cudaMemcpy(d_valid, k.valid.data(), k.valid.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice),
                       "cudaMemcpy");
        }
        cuda_check(cudaMemcpy(d_frames, k.frames.data(), k.frames.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice),
                   "cudaMemcpy");
        if (!k.prefetched.empty()) {
            cuda_check(cudaMemcpy(d_prefetched, k.prefetched.data(), k.prefetched.size() * sizeof(std::int32_t),
                                  cudaMemcpyHostToDevice), "cudaMemcpy");
        }
        if (!k.table.empty()) {
            cuda_check(cudaMemcpy(d_table, k.table.data(), k.table.size() * sizeof(std::uint64_t), cudaMemcpyHostToDevice),
                       "cudaMemcpy");
        }

        Run runs[2];
        for (int arm = 0; arm < 2; ++arm) {
            infernix::ops::MoeExpertSource source;
            source.frame_base     = d_dummy;
            source.frames         = d_frames;
            source.record_stride  = 256;
            source.host_table     = k.table.empty() ? nullptr : reinterpret_cast<const std::uint8_t* const*>(d_table);
            source.prefetched     = k.prefetched.empty() ? nullptr : d_prefetched;
            source.activation     = g.activation;
            source.shape          = g.shape;
            source.shared_expert  = g.shared_job ? E : -1;
            source.shared_record  = g.shared_job ? d_dummy : nullptr;
            source.tail_skip_rank = k.tail_skip_rank;
            source.cpu            = ch;
            std::memset(request, kRequestFill, sizeof(moe::MissRequest));
            std::memset(cx, kXFill, x_bytes);
            cuda_check(cudaMemset(layout.cpu_flags, kFlagFill, sizeof(std::int32_t) * max_jobs), "cudaMemset");
            cuda_check(cudaMemset(layout.cpu_call, kCallFill, moe::wide::kCpuCallBytes), "cudaMemset");
            Run& run = runs[arm];
            cuda_check(cudaMemcpy(&run.sequence_before, d_sequence, sizeof(std::uint32_t), cudaMemcpyDeviceToHost),
                       "cudaMemcpy");
            infernix::ops::MoeRouting routing{Tensor(d_ids, DType::I32, {K, columns}),
                                              Tensor(d_weights, DType::FP32, {K, columns}),
                                              Tensor(d_gate, DType::FP32, {columns})};
            if (!k.valid.empty()) { routing.mask = {d_valid, k.mask_width}; }
            const Tensor t_logits(d_logits, DType::FP32, {E + 1, columns});
            const Tensor tx(d_x, DType::BF16, {H, columns});
            const Tensor out(d_dummy, DType::BF16, {H, entries});
            const bool planned = infernix::ops::moe_route_dispatch_plan(t_logits, K, routing, dispatch, nullptr, tx, source,
                                                                        max_jobs, d_workspace, out, nullptr, g.precision,
                                                                        g.shared_job);
            cuda_check(cudaDeviceSynchronize(), "moe_route_dispatch_plan");
            check(planned, std::string(g.name) + ": decode and verification widths take the fused plan");
            run.request.assign(reinterpret_cast<const std::uint8_t*>(request),
                               reinterpret_cast<const std::uint8_t*>(request) + sizeof(moe::MissRequest));
            run.x.assign(reinterpret_cast<const std::uint8_t*>(cx), reinterpret_cast<const std::uint8_t*>(cx) + x_bytes);
            run.flags = host_copy(reinterpret_cast<const std::uint8_t*>(layout.cpu_flags), sizeof(std::int32_t) * max_jobs);
            run.call  = host_copy(static_cast<const std::uint8_t*>(layout.cpu_call), moe::wide::kCpuCallBytes);
            const auto ids     = host_copy(reinterpret_cast<const std::uint8_t*>(d_ids), sizeof(std::int32_t) * routed);
            const auto weights = host_copy(reinterpret_cast<const std::uint8_t*>(d_weights), sizeof(float) * routed);
            const auto gate    = host_copy(reinterpret_cast<const std::uint8_t*>(d_gate), sizeof(float) * columns);
            run.routing        = ids;
            run.routing.insert(run.routing.end(), weights.begin(), weights.end());
            run.routing.insert(run.routing.end(), gate.begin(), gate.end());
            run.counts    = host_copy(dispatch.counts, static_cast<std::size_t>(dispatch_experts));
            run.offsets   = host_copy(dispatch.offsets, static_cast<std::size_t>(dispatch_experts + 1));
            run.job_count = host_copy(dispatch.job_count, 1)[0];
            run.jobs      = host_copy(dispatch.jobs, static_cast<std::size_t>(run.job_count));
            run.entries   = host_copy(dispatch.entries, static_cast<std::size_t>(run.offsets[dispatch_experts]));

            Run want;
            oracle(g, k, columns, max_jobs, x, run, want);
            const std::string where = std::string(g.name) + " T " + std::to_string(columns) + " case " +
                                      std::to_string(index) + (arm == 1 ? " (run 2)" : " (run 1)");
            check(run.request == want.request, where + ": every byte of the mapped request equals the oracle's");
            check(run.x == want.x, where + ": the published x columns (and the rest of the region) equal the oracle's");
            check(run.flags == want.flags, where + ": cpu_flags equal the oracle's");
            check(run.call == want.call, where + ": the call's bookkeeping equals the oracle's");
            if (arm == 0) {
                const auto* call = reinterpret_cast<const PlanCall*>(want.call.data());
                published_cases += call->jobs > 0 ? 1 : 0;
                multi_chunk += std::min(run.job_count, max_jobs) > 256 ? 1 : 0;
            }
        }
        ++planned_cases;
        const std::string where = std::string(g.name) + " T " + std::to_string(columns) + " case " + std::to_string(index);
        check(runs[0].routing == runs[1].routing && runs[0].counts == runs[1].counts &&
                  runs[0].offsets == runs[1].offsets && runs[0].jobs == runs[1].jobs,
              where + ": both runs route and count identically");
        if (runs[0].entries == runs[1].entries) {
            ++same_dispatch;
            // The requests and calls differ only in the sequence each published (the counter advanced in
            // between); the oracle checked each run's own.
            std::vector<std::uint8_t> ra = runs[0].request, rb = runs[1].request;
            std::memset(ra.data() + offsetof(moe::MissRequest, sequence), 0, sizeof(std::uint32_t));
            std::memset(rb.data() + offsetof(moe::MissRequest, sequence), 0, sizeof(std::uint32_t));
            check(ra == rb && runs[0].x == runs[1].x && runs[0].flags == runs[1].flags,
                  where + ": on the same dispatch both runs write identical bytes");
            std::vector<std::uint8_t> a = runs[0].call, b = runs[1].call;
            std::memset(a.data(), 0, sizeof(std::int32_t));
            std::memset(b.data(), 0, sizeof(std::int32_t));
            check(a == b, where + ": on the same dispatch both runs list the same CPU jobs");
        }
        if (columns == 1) { check(runs[0].entries == runs[1].entries, where + ": a one-column dispatch is unique"); }
    }
    std::printf("%s T=%d: %d cases (%d published a request, %d with a host table, %d of more than 256 jobs), %d with "
                "the same dispatch in both runs\n",
                g.name, columns, planned_cases, published_cases, ssd_cases, multi_chunk, same_dispatch);
    check(published_cases > 0, std::string(g.name) + ": some cases publish a request");

    for (void* p : {static_cast<void*>(d_logits), static_cast<void*>(d_ids), static_cast<void*>(d_weights),
                    static_cast<void*>(d_gate), d_dispatch, d_workspace, static_cast<void*>(d_x),
                    static_cast<void*>(d_dummy), static_cast<void*>(d_frames), static_cast<void*>(d_prefetched),
                    static_cast<void*>(d_table), static_cast<void*>(d_divisor), static_cast<void*>(d_valid),
                    static_cast<void*>(d_sequence)}) {
        cuda_check(cudaFree(p), "cudaFree");
    }
    for (void* p : {static_cast<void*>(request), static_cast<void*>(cx), static_cast<void*>(cy), static_cast<void*>(done)}) {
        cuda_check(cudaFreeHost(p), "cudaFreeHost");
    }
}

} // namespace

INFERNIX_OP_TEST(offloaded_moe_plan) {
    if (infernix::test::cuda_unavailable()) {
        std::printf("SKIP: no usable CUDA device\n");
        return 77;
    }
    try {
        const Geometry geometries[] = {
            {"Qwen3.5-122B-A10B (256 experts, top-8, shared job)", 256, 8, true, infernix::ops::RouterLogits::kBf16,
             moe::kQwen3_5_122BShape, moe::ExpertActivation::kA16},
            {"Qwen3.8-Flash-Next (512 experts, top-10)", 512, 10, false, infernix::ops::RouterLogits::kFp32,
             moe::kFlashNextShape, moe::ExpertActivation::kA4},
        };
        std::uint32_t seed = 1;
        for (const auto& g : geometries) {
            for (const int columns : {1, 2, 3, 4, 8, 16, 24, 32}) {
                test_geometry(g, columns, columns == 1 ? 400 : 120, seed++);
            }
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
    if (g_failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("all offloaded_moe plan checks passed\n");
    return 0;
}
