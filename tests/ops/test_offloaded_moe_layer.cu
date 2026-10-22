// The layer route of offloaded_sparse_moe (moe_route, moe_dispatch, moe_experts) against the CPU
// engine, with expert records split between device frames and the pinned host bank and misses
// read zero-copy, staged through 1, 3 or 64 device slots (one or several staging passes), or
// served by the host expert engine through the CPU miss channel
// (docs/maintainer/qwen3_8-flash-next-design.md §8.6, §10.3, §16.2), and with SSD-only experts
// (the SSD tier, §19.3.7) read through a RecordProvider by CPU jobs or fetched through the fetch
// channel from a host responder, including its failure paths (host status, silent host).
//
// Oracle: the CPU engine's output of each routed (column, expert) pair of the narrow route, bit
// for bit; the narrow route's arithmetic is exact. Every output, the wide route's included (its own
// FP64 qualification is test_offloaded_moe_wide), equals the first configuration's: neither the record's
// location, nor the staging pass a job falls in, nor a CPU-served share may change an output bit.
// moe_dispatch has its own exact oracle (test_dispatch). Layers of W4A16 experts (§16.2.1) run the
// same configurations against expert_forward_a16, which every W4A16 output equals bit for bit (its
// wide route is exact as well).
#include "infernix/ops/hyper_connection.h"
#include "infernix/ops/offloaded_sparse_moe.h"
#include "ops/offloaded_moe_fixtures.h"
#include "ops/offloaded_sparse_moe/cpu/fetch_channel.h"
#include "ops/offloaded_sparse_moe/cpu/miss_service.h"
#include "ops/offloaded_sparse_moe/cpu/w4a16_expert.h"
#include "ops/offloaded_sparse_moe/cpu/w4a4_expert.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

namespace moe      = infernix::ops::offloaded_moe;
namespace fixtures = infernix::test::offloaded_moe;
using infernix::DType;
using infernix::Tensor;
using infernix::test::cuda_check;

namespace {

int g_failures = 0;

// Live hosts get production's bound (kHeartbeatTimeoutNs, 10 s): under a loaded machine a 1 s bound
// mistook OS scheduling stalls for dead hosts. The silent-host cases wait their whole bound, so they
// use a short one.
constexpr std::uint64_t kTestHeartbeatTimeoutNs   = moe::kHeartbeatTimeoutNs;
constexpr std::uint64_t kSilentHeartbeatTimeoutNs = 1000000000ULL;

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

std::vector<std::uint16_t> cpu_column(const fixtures::Expert& e, const std::uint16_t* x, moe::ExpertActivation activation) {
    std::vector<std::uint16_t> y(moe::kHidden);
    const std::uint16_t* xp[1] = {x};
    std::uint16_t* yp[1]       = {y.data()};
    if (activation == moe::ExpertActivation::kA16) {
        moe::expert_forward_a16(moe::best_cpu_isa(), moe::kFlashNextShape, e.record.data(), e.scales, 1, xp, yp);
    } else {
        moe::expert_forward(moe::best_cpu_isa(), e.record.data(), e.scales, 1, xp, yp);
    }
    return y;
}

// Serves SSD-only records to the CPU service from private copies of the bank (the SSD tier's role).
struct CopiedRecords final : moe::RecordProvider {
    const std::uint8_t* bank = nullptr;
    std::size_t stride       = 0;
    std::vector<std::vector<std::uint8_t>> copies;
    std::uint32_t demand(int, int expert) noexcept override {
        const auto* record = bank + static_cast<std::size_t>(expert) * stride;
        copies.emplace_back(record, record + moe::kRecordBytes);
        return static_cast<std::uint32_t>(copies.size() - 1);
    }
    bool landed(std::uint32_t) const noexcept override { return true; }
    const std::uint8_t* wait(std::uint32_t ticket, std::uint32_t& status) noexcept override {
        status = 0;
        return copies[ticket].data();
    }
    void done(std::uint32_t) noexcept override {}
};

// The SSD tier's agent in miniature (design §19.3.7): answers each fetch request with copies of the
// bank's records written into a ring of mapped slots, landing them in swapped pairs (record i + 1
// before record i) and reusing a slot only once the device has consumed its previous record.
// Modes: answer, fail the request at record fail_at (EIO), or stay silent (acknowledge, then
// neither land nor beat).
class Responder {
public:
    enum class Mode { kAnswer, kFail, kSilent };
    static constexpr std::uint32_t kRing = 72; // more than the largest pass (64 jobs) plus one swap

    Responder(const std::uint8_t* bank, std::size_t stride) : bank_(bank), stride_(stride) {
        cuda_check(cudaHostAlloc(reinterpret_cast<void**>(&ring_), stride * kRing, cudaHostAllocMapped), "cudaHostAlloc");
        thread_ = std::thread([this] { run(); });
    }
    ~Responder() {
        stop_.store(true);
        thread_.join();
        cudaFreeHost(ring_);
    }
    [[nodiscard]] infernix::ops::MoeFetchChannel channel(int layer) const { return channel_.channel(layer); }
    // Before a call, with no request open.
    void set(Mode mode, std::uint32_t fail_at = 0) {
        fail_at_.store(fail_at);
        mode_.store(mode);
    }
    // After a call: the records of its request, and how many the device consumed.
    [[nodiscard]] std::uint32_t count() const { return count_.load(); }
    [[nodiscard]] std::uint32_t consumed() const { return consumed_.load(); }
    [[nodiscard]] std::uint64_t requests() const { return requests_.load(); }

private:
    void run() {
        moe::FetchChannel::Request request;
        while (!stop_.load()) {
            const Mode mode = mode_.load();
            if (mode != Mode::kSilent) { channel_.beat(); }
            consumed_.store(channel_.consumed());
            if (!channel_.poll(request)) {
                std::this_thread::yield();
                continue;
            }
            requests_.fetch_add(1);
            const auto count = static_cast<std::uint32_t>(request.experts.size());
            count_.store(count);
            if (mode == Mode::kSilent) { continue; }
            for (std::uint32_t pair = 0; pair < count; pair += 2) {
                bool failed = false;
                for (const std::uint32_t i : {pair + 1, pair}) {
                    if (i >= count) { continue; }
                    if (mode == Mode::kFail && i == fail_at_.load()) {
                        channel_.fail(EIO);
                        failed = true;
                        break;
                    }
                    // Slot i % kRing held record i - kRing: wait until the device has copied it.
                    while (i >= kRing && channel_.consumed() <= i - kRing && !stop_.load()) {
                        channel_.beat();
                        std::this_thread::yield();
                    }
                    std::uint8_t* slot = ring_ + static_cast<std::size_t>(i % kRing) * stride_;
                    std::memcpy(slot, bank_ + static_cast<std::size_t>(request.experts[i]) * stride_, moe::kRecordBytes);
                    channel_.land(i, slot);
                    channel_.beat();
                }
                if (failed) { break; }
            }
        }
    }

    const std::uint8_t* bank_;
    std::size_t stride_;
    std::uint8_t* ring_ = nullptr;
    moe::FetchChannel channel_;
    std::atomic<Mode> mode_{Mode::kAnswer};
    std::atomic<std::uint32_t> fail_at_{0}, count_{0}, consumed_{0};
    std::atomic<std::uint64_t> requests_{0};
    std::atomic<bool> stop_{false};
    std::thread thread_;
};

void test_layer(int experts, int columns, int top_k, std::uint32_t seed,
                moe::ExpertActivation activation = moe::ExpertActivation::kA4) {
    const bool a16 = activation == moe::ExpertActivation::kA16;
    if (a16) { std::printf("--- W4A16 experts\n"); }
    std::mt19937 rng(seed);
    const std::size_t stride = moe::kRecordBytes;
    std::vector<fixtures::Expert> bank;
    std::vector<moe::ExpertScales> scales;
    for (int e = 0; e < experts; ++e) {
        bank.push_back(a16 ? fixtures::random_a16_expert(rng) : fixtures::random_expert(rng, e % 3 == 0));
        scales.push_back(bank.back().scales);
    }
    // Pinned host bank of every record; every third expert also resident in a device frame.
    std::uint8_t* host = nullptr;
    cuda_check(cudaHostAlloc(reinterpret_cast<void**>(&host), stride * experts, cudaHostAllocMapped), "cudaHostAlloc");
    for (int e = 0; e < experts; ++e) { std::memcpy(host + stride * e, bank[e].record.data(), moe::kRecordBytes); }
    std::uint8_t* host_device = nullptr;
    cuda_check(cudaHostGetDevicePointer(reinterpret_cast<void**>(&host_device), host, 0), "cudaHostGetDevicePointer");
    // The same records in shuffled host slots, reached through a host-pointer table (the SSD tier's
    // RAM slots, design §19.3.7 R5): every route must read identical bits through it.
    std::uint8_t* shuffled = nullptr;
    cuda_check(cudaHostAlloc(reinterpret_cast<void**>(&shuffled), stride * experts, cudaHostAllocMapped), "cudaHostAlloc");
    std::vector<int> slot_of(experts);
    for (int e = 0; e < experts; ++e) { slot_of[e] = e; }
    std::shuffle(slot_of.begin(), slot_of.end(), rng);
    std::vector<const std::uint8_t*> table(experts);
    for (int e = 0; e < experts; ++e) {
        std::memcpy(shuffled + stride * slot_of[e], bank[e].record.data(), moe::kRecordBytes);
        std::uint8_t* mapped = nullptr;
        cuda_check(cudaHostGetDevicePointer(reinterpret_cast<void**>(&mapped), shuffled + stride * slot_of[e], 0),
                   "cudaHostGetDevicePointer");
        table[e] = mapped;
    }
    auto* d_table = device_copy(table);
    std::vector<std::int32_t> frames(experts, -1);
    std::vector<std::uint8_t> frame_bytes;
    int resident = 0;
    for (int e = 0; e < experts; e += 3) {
        frames[e] = resident++;
        frame_bytes.insert(frame_bytes.end(), bank[e].record.begin(), bank[e].record.end());
    }
    // Free frames after the resident ones, for landing: slot 1 has none (-1), slot 3 is past the
    // misses when there are few.
    const std::vector<std::int32_t> landing{resident, -1, resident + 1, resident + 2};
    frame_bytes.resize(frame_bytes.size() + 3 * stride, 0);
    auto* d_frame_base = device_copy(frame_bytes);
    auto* d_landing    = device_copy(landing);
    auto* d_landed     = device_copy(std::vector<std::int32_t>(landing.size(), -1));
    auto* d_frames     = device_copy(frames);
    // Streamed records (F2): every second non-resident expert sits in a slot of its own, in reverse
    // order, as a ring the caller filled before the call.
    std::vector<std::int32_t> prefetched(experts, -1);
    std::vector<std::uint8_t> slot_bytes;
    for (int e = experts - 1, slot = 0; e >= 0; --e) {
        if (frames[e] >= 0 || e % 2 != 1) { continue; }
        prefetched[e] = slot++;
        slot_bytes.insert(slot_bytes.end(), bank[e].record.begin(), bank[e].record.end());
    }
    if (slot_bytes.empty()) { slot_bytes.resize(stride, 0); }
    auto* d_prefetched    = device_copy(prefetched);
    auto* d_prefetch_base = device_copy(slot_bytes);
    auto* d_scales     = device_copy(scales);

    // Routing from random FP32 logits.
    std::normal_distribution<float> n(0.0F, 1.0F);
    std::vector<float> logits(static_cast<std::size_t>(experts + 1) * columns);
    for (auto& v : logits) { v = n(rng); }
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
    cuda_check(cudaMalloc(&d_dispatch, infernix::ops::moe_dispatch_bytes(experts, top_k * columns)), "cudaMalloc");
    auto dispatch = infernix::ops::carve_moe_dispatch(d_dispatch, experts, top_k * columns);
    infernix::ops::moe_dispatch(routing, experts, dispatch, nullptr, nullptr);

    const auto x  = a16 ? fixtures::wide_range_activations(rng, columns) : fixtures::random_activations(rng, columns);
    auto* d_x     = device_copy(x);
    std::vector<std::int32_t> ids(static_cast<std::size_t>(top_k) * columns);
    cuda_check(cudaMemcpy(ids.data(), d_ids, ids.size() * sizeof(std::int32_t), cudaMemcpyDeviceToHost), "cudaMemcpy");

    // Expected outputs: column t * k + slot = expert ids[slot, t] applied to x[:, t].
    std::vector<std::uint16_t> expected(static_cast<std::size_t>(moe::kHidden) * top_k * columns);
    for (int t = 0; t < columns; ++t) {
        for (int s = 0; s < top_k; ++s) {
            const auto y = cpu_column(bank[ids[static_cast<std::size_t>(t) * top_k + s]],
                                      &x[static_cast<std::size_t>(t) * moe::kHidden], activation);
            std::memcpy(&expected[(static_cast<std::size_t>(t) * top_k + s) * moe::kHidden], y.data(),
                        moe::kHidden * sizeof(std::uint16_t));
        }
    }

    const int max_jobs = std::min(experts, top_k * columns);
    std::uint8_t* d_staging = nullptr;
    cuda_check(cudaMalloc(&d_staging, stride * 64), "cudaMalloc");
    void* d_workspace = nullptr;
    cuda_check(cudaMalloc(&d_workspace, infernix::ops::moe_experts_workspace_bytes(infernix::ops::offloaded_moe::kFlashNextShape, max_jobs, top_k * columns, columns, activation)),
               "cudaMalloc");
    std::uint16_t* d_out = nullptr;
    cuda_check(cudaMalloc(&d_out, expected.size() * sizeof(std::uint16_t)), "cudaMalloc");
    std::vector<moe::CpuMissService::Layer> layers{{.records = host, .record_stride = stride, .scales = scales.data()}};
    moe::CpuMissService service_two(layers, {.workers = 2, .max_jobs = 2, .max_columns = 64, .cpus = {}, .activation = activation});
    moe::CpuMissService service_eight(layers, {.workers = 4, .max_jobs = 8, .max_columns = 64, .cpus = {}, .activation = activation});
    // The largest cap with every miss offered to the CPU, and a cap whose jobs are limited to two columns.
    moe::CpuMissService service_all(layers, {.workers = 6, .max_jobs = moe::kMaxCpuJobs, .max_columns = 64,
                                             .pcie_divisor = 0, .cpus = {}, .activation = activation});
    moe::CpuMissService service_narrow(layers, {.workers = 6, .max_jobs = 24, .max_columns = 64, .pcie_divisor = 2,
                                                .max_job_columns = 2, .cpus = {}, .activation = activation});
    // A GGUF import's decode split: one-column calls keep every miss on the link (d = 1), wider calls d = 2.
    moe::CpuMissService service_one(layers, {.workers = 6, .max_jobs = 16, .max_columns = 64, .pcie_divisor = 2,
                                             .one_column_divisor = 1, .cpus = {}, .activation = activation});
    // Prefill CPU assist (P7): calls of at least 9 columns take up to 256 CPU-served misses.
    moe::CpuMissService service_assist(layers, {.workers = 6, .max_jobs = 8, .max_columns = 255, .pcie_divisor = 4,
                                                .wide_from = 9, .wide_jobs = moe::kMaxCpuJobs, .cpus = {}, .activation = activation});
    // A prefill call's CPU share (design §19.3.12): calls up to a chunk of 4,096 columns, every narrow
    // miss offered (no PCIe share), up to every expert of the layer.
    moe::CpuMissService service_wide(layers, {.workers = 6, .max_jobs = 8, .max_columns = moe::kMaxCpuCallColumns,
                                              .pcie_divisor = 0, .wide_from = 256, .wide_jobs = moe::kMaxCpuJobs,
                                              .cpus = {}, .activation = activation});
    // SSD tier (design §19.3.7): services that read SSD-only records through a provider.
    CopiedRecords copied;
    copied.bank   = host;
    copied.stride = stride;
    moe::CpuMissService service_ssd_all(layers, {.workers = 6, .max_jobs = moe::kMaxCpuJobs, .max_columns = 64,
                                                 .pcie_divisor = 0, .cpus = {}, .activation = activation, .records = &copied});
    moe::CpuMissService service_ssd8(layers, {.workers = 4, .max_jobs = 8, .max_columns = 64, .cpus = {}, .activation = activation,
                                              .records = &copied});
    // The CPU's share of the misses (design §19.3.5 S3): want = min(cap, M - M / divisor) of the misses
    // with at most max_job_columns columns, so the number served is min(want, eligible misses). The
    // divisor is one_column_divisor (when set) in a one-column call.
    const auto divisor_of = [&](const auto& channel) {
        return columns == 1 && channel.one_column_divisor > 0 ? channel.one_column_divisor : channel.pcie_divisor;
    };
    std::vector<int> expert_columns(experts, 0);
    for (const std::int32_t id : ids) { ++expert_columns[id]; }
    // SSD-only experts have no host record in the table. Narrow set: every third non-resident
    // expert of at most kMaxCpuColumns columns (CPU jobs can take every one). Full set: every third
    // non-resident expert, wide ones too (only the fetch channel can serve those).
    std::vector<std::uint8_t> ssd_narrow(experts, 0), ssd_full(experts, 0);
    std::vector<const std::uint8_t*> narrow_table = table, full_table = table;
    for (int e = 0; e < experts; ++e) {
        if (frames[e] >= 0 || e % 3 != 1) { continue; }
        ssd_full[e]   = 1;
        full_table[e] = nullptr;
        if (expert_columns[e] >= 1 && expert_columns[e] <= moe::kMaxCpuColumns) {
            ssd_narrow[e]   = 1;
            narrow_table[e] = nullptr;
        }
    }
    auto* d_narrow_table = device_copy(narrow_table);
    auto* d_full_table   = device_copy(full_table);
    Responder responder(host, stride);
    // The overlap route (prefill passes staged on a side stream) and its five events.
    cudaStream_t overlap_stream = nullptr;
    cudaEvent_t overlap_events[5] = {};
    cuda_check(cudaStreamCreateWithFlags(&overlap_stream, cudaStreamNonBlocking), "cudaStreamCreate");
    for (auto& event : overlap_events) { cuda_check(cudaEventCreateWithFlags(&event, cudaEventDisableTiming), "event"); }
    std::uint32_t* error_host = nullptr;
    cuda_check(cudaHostAlloc(reinterpret_cast<void**>(&error_host), 64, cudaHostAllocMapped), "cudaHostAlloc");
    std::uint32_t* error_device = nullptr;
    cuda_check(cudaHostGetDevicePointer(reinterpret_cast<void**>(&error_device), error_host, 0), "cudaHostGetDevicePointer");
    // A streamed record counts as resident: the CPU takes only misses the stream left out. With SSD-only
    // experts (`ssd`), the eligible ones are taken first up to the cap; the divisor rule then applies to
    // the host-memory misses.
    const auto expected_cpu_jobs = [&](const moe::CpuMissService& service, bool streamed,
                                       const std::vector<std::uint8_t>* ssd) {
        const auto channel = service.channel(0);
        if (columns > channel.max_columns) { return 0; } // a call this wide takes no CPU-served misses
        int misses = 0, eligible = 0, ssd_misses = 0;
        for (int e = 0; e < experts; ++e) {
            if (expert_columns[e] == 0 || frames[e] >= 0 || (streamed && prefetched[e] >= 0)) { continue; }
            if (ssd != nullptr && (*ssd)[e] != 0) {
                ssd_misses += expert_columns[e] <= channel.max_job_columns ? 1 : 0;
                continue;
            }
            ++misses;
            eligible += expert_columns[e] <= channel.max_job_columns ? 1 : 0;
        }
        const int cap  = channel.wide_from > 0 && columns >= channel.wide_from ? channel.wide_jobs : channel.max_jobs;
        const int take = std::min(cap, ssd_misses);
        const int d    = divisor_of(channel);
        const int want = std::min(cap - take, d > 0 ? misses - misses / d : misses);
        return take + std::min(want, eligible);
    };
    // Narrow SSD-only experts no CPU job takes (no service, a call wider than its columns, or past
    // its cap): without a fetch channel nothing can serve them, so the call must set its error word.
    const auto ssd_unserved = [&](const moe::CpuMissService* service) {
        int ssd_misses = 0;
        for (int e = 0; e < experts; ++e) { ssd_misses += ssd_narrow[e]; }
        if (service == nullptr || columns > service->channel(0).max_columns) { return ssd_misses; }
        const auto channel = service->channel(0);
        const int cap = channel.wide_from > 0 && columns >= channel.wide_from ? channel.wide_jobs : channel.max_jobs;
        return std::max(0, ssd_misses - cap);
    };
    struct Config {
        int slots;
        const moe::CpuMissService* service;
        bool fork = false;
        bool land = false;
        bool streamed = false;
        bool table    = false;
        int ssd       = 0;     // the table leaves SSD-only experts without a host record: 1 narrow set, 2 full set
        Responder::Mode fetch = Responder::Mode::kAnswer;
        bool fetched  = false; // the fetch channel serves SSD-only experts no CPU job takes
        bool overlap  = false; // passes staged on the overlap stream
        int idle      = 0;     // calls after the service idles 60 ms; 2: wake() before each
    };
    // Which misses the CPU takes (design §19.3.5 S3): the fewest-column ones first, in job order within a
    // width, want = min(cap, M - M / divisor) of those with at most max_job_columns columns.
    const auto cpu_selection = [&](const moe::CpuMissService* service, const std::vector<std::int32_t>& jobs,
                                   bool streamed) {
        std::vector<std::uint8_t> chosen(experts, 0);
        if (service == nullptr || columns > service->channel(0).max_columns) { return chosen; }
        const auto channel = service->channel(0);
        const auto miss    = [&](std::int32_t e) { return frames[e] < 0 && !(streamed && prefetched[e] >= 0); };
        int misses = 0;
        for (const std::int32_t e : jobs) { misses += miss(e) ? 1 : 0; }
        const int cap  = channel.wide_from > 0 && columns >= channel.wide_from ? channel.wide_jobs : channel.max_jobs;
        const int d    = divisor_of(channel);
        const int want = std::min(cap, d > 0 ? misses - misses / d : misses);
        int n = 0;
        for (int width = 1; width <= channel.max_job_columns && n < want; ++width) {
            for (const std::int32_t e : jobs) {
                if (n < want && miss(e) && expert_columns[e] == width) {
                    chosen[e] = 1;
                    ++n;
                }
            }
        }
        return chosen;
    };
    // Outputs with the CPU engine's bits: every W4A16 output (its wide route is exact too), and the
    // W4A4 narrow route's (design §8.5): an expert with at most eight columns, or one whose gate and
    // up projections keep separate input scales (the fixture's every third expert).
    std::vector<std::uint8_t> narrow(static_cast<std::size_t>(top_k) * columns);
    for (std::size_t i = 0; i < narrow.size(); ++i) {
        narrow[i] = a16 || expert_columns[ids[i]] <= moe::kMaxCpuColumns || ids[i] % 3 == 0;
    }
    std::vector<std::uint16_t> placed;
    // The fork stream and its events, for the one-pass decode/verification route.
    cudaStream_t fork_stream = nullptr;
    cudaEvent_t fork_events[2] = {};
    cuda_check(cudaStreamCreateWithFlags(&fork_stream, cudaStreamNonBlocking), "cudaStreamCreate");
    for (auto& event : fork_events) { cuda_check(cudaEventCreateWithFlags(&event, cudaEventDisableTiming), "event"); }
    double gpu_idle_extra = 0.0; // what 60 ms of idle adds to a call without the CPU service (the GPU's own wake-up)
    for (const Config config : {Config{0, nullptr}, Config{1, nullptr}, Config{3, nullptr}, Config{64, nullptr},
                                Config{3, &service_two}, Config{64, &service_eight}, Config{0, &service_eight},
                                Config{64, nullptr, true}, Config{64, &service_two, true}, Config{3, nullptr, true},
                                Config{3, &service_all}, Config{64, &service_all, true}, Config{64, &service_narrow},
                                Config{64, &service_narrow, true}, Config{64, &service_one}, Config{64, &service_one, true},
                                Config{64, nullptr, true, true},
                                Config{64, &service_narrow, true, true}, Config{3, nullptr, false, true},
                                Config{64, &service_assist}, Config{3, &service_assist}, Config{0, &service_assist},
                                Config{64, &service_assist, true}, Config{0, nullptr, false, false, true},
                                Config{3, nullptr, false, false, true}, Config{64, nullptr, false, false, true},
                                Config{64, nullptr, true, false, true}, Config{64, &service_wide},
                                Config{0, &service_wide}, Config{64, &service_wide, false, false, true},
                                Config{0, nullptr, false, false, false, true},
                                Config{3, nullptr, false, false, false, true}, Config{64, nullptr, true, false, false, true},
                                Config{64, &service_two, true, false, false, true}, Config{3, &service_all, false, false, false, true},
                                Config{64, &service_assist, false, false, false, true},
                                Config{3, &service_ssd_all, false, false, false, true, true},
                                Config{64, &service_ssd_all, true, false, false, true, true},
                                Config{0, &service_ssd_all, false, false, false, true, true},
                                Config{3, &service_ssd8, false, false, false, true, true},
                                Config{64, &service_ssd8, true, false, false, true, true},
                                Config{3, nullptr, false, false, false, true, true},
                                // Fetch channel: serial passes, overlap, fork, landing, with CPU jobs, failures.
                                Config{.slots = 3, .service = nullptr, .table = true, .ssd = 2, .fetched = true},
                                Config{.slots = 1, .service = nullptr, .table = true, .ssd = 2, .fetched = true},
                                Config{.slots = 16, .service = nullptr, .table = true, .ssd = 2, .fetched = true,
                                       .overlap = true},
                                Config{.slots = 64, .service = nullptr, .fork = true, .table = true, .ssd = 2,
                                       .fetched = true},
                                Config{.slots = 64, .service = nullptr, .fork = true, .land = true, .table = true,
                                       .ssd = 2, .fetched = true},
                                Config{.slots = 3, .service = &service_ssd8, .table = true, .ssd = 2, .fetched = true},
                                Config{.slots = 64, .service = &service_ssd8, .fork = true, .table = true, .ssd = 2,
                                       .fetched = true},
                                Config{.slots = 16, .service = &service_ssd_all, .table = true, .ssd = 2,
                                       .fetched = true, .overlap = true},
                                Config{.slots = 3, .service = nullptr, .table = true, .ssd = 2,
                                       .fetch = Responder::Mode::kFail, .fetched = true},
                                Config{.slots = 3, .service = nullptr, .table = true, .ssd = 2,
                                       .fetch = Responder::Mode::kSilent, .fetched = true},
                                // An idle service answers its first request promptly (Windows: no 15.6 ms
                                // timer-tick sleep): through its short idle wait, and at once after wake().
                                // The first call, without the service, measures the GPU's own idle cost.
                                Config{.slots = 64, .service = nullptr, .idle = 1},
                                Config{.slots = 64, .service = &service_eight, .idle = 1},
                                Config{.slots = 64, .service = &service_eight, .idle = 2}}) {
        const int slots = config.slots;
        // A fresh dispatch: a fetch call reorders its jobs (fetch-served last).
        infernix::ops::moe_dispatch(routing, experts, dispatch, nullptr, nullptr);
        cuda_check(cudaMemset(d_out, 0xFF, expected.size() * sizeof(std::uint16_t)), "cudaMemset");
        cuda_check(cudaMemset(d_staging, 0, stride * 64), "cudaMemset");
        infernix::ops::MoeExpertSource source{.frame_base    = d_frame_base,
                                            .frames        = d_frames,
                                            .host_records  = host_device,
                                            .record_stride = stride,
                                            .scales        = d_scales,
                                            .staging_base  = slots > 0 ? d_staging : nullptr,
                                            .staging_slots = slots,
                                            .cpu = config.service != nullptr ? config.service->channel(0)
                                                                             : infernix::ops::MoeCpuChannel{}};
        source.activation           = activation;
        source.heartbeat_timeout_ns = config.fetch == Responder::Mode::kSilent ? kSilentHeartbeatTimeoutNs : kTestHeartbeatTimeoutNs;
        if (config.service != nullptr) {
            // The CPU wait warms these into L2 while the host works; no output may change.
            source.l2_warm.ptr[0]   = d_frame_base;
            source.l2_warm.bytes[0] = frame_bytes.size();
            source.l2_warm.ptr[1]   = d_x;
            source.l2_warm.bytes[1] = x.size() * sizeof(std::uint16_t);
        }
        if (config.fork) {
            source.fork_stream    = fork_stream;
            source.fork_events[0] = fork_events[0];
            source.fork_events[1] = fork_events[1];
        }
        if (config.land) {
            source.landing       = d_landing;
            source.landed        = d_landed;
            source.landing_slots = static_cast<std::int32_t>(landing.size());
            cuda_check(cudaMemcpy(d_landed, std::vector<std::int32_t>(landing.size(), -1).data(),
                                  landing.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice), "cudaMemcpy");
        }
        if (config.streamed) {
            source.prefetched    = d_prefetched;
            source.prefetch_base = d_prefetch_base;
        }
        if (config.table) {
            source.host_records = nullptr; // only the table may be read
            source.host_table   = config.ssd == 1 ? d_narrow_table : config.ssd == 2 ? d_full_table : d_table;
        }
        if (config.overlap) {
            source.overlap_stream = overlap_stream;
            for (int i = 0; i < 5; ++i) { source.overlap_events[i] = overlap_events[i]; }
        }
        constexpr int kFetchLayer = 5; // names the call in its error word
        if (config.fetched) {
            responder.set(config.fetch, 5);
            source.fetch = responder.channel(kFetchLayer);
        }
        const std::uint64_t requests_before = responder.requests();
        *reinterpret_cast<volatile std::uint32_t*>(error_host) = 0;
        source.error = error_device;
        Tensor tx(d_x, DType::BF16, {moe::kHidden, columns});
        Tensor out(d_out, DType::BF16, {moe::kHidden, top_k * columns});
        if (config.idle > 0) {
            // The time 60 ms of idle adds to a call, against the same call made at once after another. The
            // GPU's own wake-up adds ~0.1 ms without the service. With it, small calls measured +0.3 to +7.3 ms
            // (2026-10-09, median of 5; not yet explained), large ones nothing (the CPU wakes while the GPU
            // works); a timer-tick sleep in the service's idle wait (Windows: 15.6 ms) adds about 8 ms more on
            // average, so the bound is 10 ms beyond the GPU's own, plus 5 % of the call as noise.
            const auto timed = [&] {
                const auto start = std::chrono::steady_clock::now();
                infernix::ops::moe_experts(tx, dispatch, source, top_k, max_jobs, d_workspace, out, nullptr);
                cuda_check(cudaDeviceSynchronize(), "moe_experts");
                return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
            };
            // Other processes on the machine can stall one measurement (CTest runs this entry alone:
            // beside other GPU tests no bound of a few milliseconds holds); a slow idle wait slows every
            // one. The service check passes when any of three measurements meets the bound.
            bool within = false;
            for (int attempt = 0; attempt < 3 && !within; ++attempt) {
                (void)timed();
                const double busy = timed();
                std::array<double, 5> extra{};
                for (double& e : extra) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(60));
                    if (config.idle == 2) { config.service->wake(); }
                    e = timed() - busy;
                }
                std::sort(extra.begin(), extra.end());
                std::printf("E=%d T=%d k=%d idle %s: call %.2f ms busy, +%.2f ms idle (median of 5, max +%.2f)\n",
                            experts, columns, top_k,
                            config.service == nullptr ? "GPU only"
                            : config.idle == 2        ? "service after wake()"
                                                      : "service",
                            busy, extra[2], extra[4]);
                if (config.service == nullptr) {
                    gpu_idle_extra = extra[2];
                    within         = true;
                } else {
                    within = extra[2] - gpu_idle_extra < 10.0 + 0.05 * busy;
                }
            }
            check(within, "an idle CPU miss service adds under 10 ms to a call beyond the GPU's own (median of 5)");
        }
        const std::uint64_t served_before = config.service != nullptr ? config.service->served_experts() : 0;
        // A served call routes again through moe_route_dispatch_plan, which plans the CPU-served misses
        // in its own kernel at decode and verification widths (and otherwise routes as before), so
        // the checks below cover the fused plan.
        bool planned = false;
        if (config.service != nullptr) {
            planned = infernix::ops::moe_route_dispatch_plan(Tensor(d_logits, DType::FP32, {experts + 1, columns}),
                                                             top_k, routing, dispatch, nullptr, tx, source, max_jobs,
                                                             d_workspace, out, nullptr);
        }
        infernix::ops::moe_experts(tx, dispatch, source, top_k, max_jobs, d_workspace, out, nullptr, true, planned);
        cuda_check(cudaDeviceSynchronize(), "moe_experts");
        if (config.service != nullptr) {
            // The service counts after answering, so its count may trail the device by a moment.
            const auto* ssd = config.ssd == 1 ? &ssd_narrow : config.ssd == 2 ? &ssd_full : nullptr;
            const auto want = static_cast<std::uint64_t>(expected_cpu_jobs(*config.service, config.streamed, ssd));
            const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(30);
            while (config.service->served_experts() - served_before < want && std::chrono::steady_clock::now() < until) {
                std::this_thread::yield();
            }
            check(config.service->served_experts() - served_before == want,
                  "the CPU serves min(cap, M - M / divisor) of the eligible misses");
        }
        const std::uint32_t error = *reinterpret_cast<volatile std::uint32_t*>(error_host);
        const int unserved        = config.ssd == 1 && !config.fetched ? ssd_unserved(config.service) : 0;
        if (unserved > 0) {
            // Its outputs are undefined; the word names the layer and the staging code.
            std::printf("E=%d T=%d k=%d staging slots %2d (SSD-only experts), %d unserved: error word %#x\n", experts,
                        columns, top_k, slots, unserved, error);
            check(error == moe::kErrorUnservedRecord, "an SSD-only expert outside the CPU jobs sets the call's error word");
            continue;
        }
        if (config.fetched) {
            // Every fetch-served expert: SSD-only, routed, not taken by a CPU job.
            const bool published = responder.requests() > requests_before;
            const std::uint32_t count = published ? responder.count() : 0;
            if (config.fetch != Responder::Mode::kAnswer) {
                // The wait ends without a trap; the word names the layer and the host's errno or the silence.
                const std::uint32_t want = config.fetch == Responder::Mode::kFail ? EIO : moe::kErrorFetchSilent;
                std::printf("E=%d T=%d k=%d staging slots %2d (fetch, %s): %u records requested, error word %#x\n",
                            experts, columns, top_k, slots, config.fetch == Responder::Mode::kFail ? "failing host" : "silent host",
                            count, error);
                if (count > (config.fetch == Responder::Mode::kFail ? 5U : 0U)) {
                    check(error == (static_cast<std::uint32_t>(kFetchLayer) << 16U | want),
                          "a failed fetch request sets the call's error word (layer << 16 | code)");
                } else {
                    check(error == 0, "a request that fails after its last record (or no request) fails no copy");
                }
                responder.set(Responder::Mode::kAnswer);
                if (error != 0) { continue; }
            } else if (published) {
                // The device consumed every record it was sent (the last pass stores the total).
                const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(30);
                while (responder.consumed() != count && std::chrono::steady_clock::now() < until) { std::this_thread::yield(); }
                std::printf("  fetch: %u records requested, %u consumed\n", count, responder.consumed());
                check(responder.consumed() == count, "the device reports every fetched record consumed");
            }
        }
        check(error == 0, "a call that serves every expert leaves its error word clear");
        std::vector<std::uint16_t> got(expected.size());
        cuda_check(cudaMemcpy(got.data(), d_out, got.size() * sizeof(std::uint16_t), cudaMemcpyDeviceToHost), "cudaMemcpy");
        // The first configuration (every miss zero-copy on the GPU) fixes the wide-route outputs.
        if (placed.empty()) { placed = got; }
        long mismatches = 0, moved = 0;
        for (std::size_t i = 0; i < got.size(); ++i) {
            mismatches += narrow[i / moe::kHidden] && got[i] != expected[i];
            moved += got[i] != placed[i];
        }
        std::printf("E=%d T=%d k=%d staging slots %2d%s%s%s, CPU jobs %d (served %llu): %ld narrow-route outputs "
                    "differ from the CPU engine, %ld outputs from the first placement, of %zu\n",
                    experts, columns, top_k, slots, config.fork ? " (fork)" : "", config.streamed ? " (streamed)" : "",
                    config.fetched ? (config.overlap ? " (host table, SSD-only experts fetched, overlap)"
                                                     : " (host table, SSD-only experts fetched)")
                    : config.ssd != 0 ? " (host table, SSD-only experts)"
                    : config.table    ? " (host table)"
                                      : "",
                    config.service != nullptr ? config.service->channel(0).max_jobs : 0,
                    config.service != nullptr ? static_cast<unsigned long long>(config.service->served_experts()) : 0ULL,
                    mismatches, moved, got.size());
        check(mismatches == 0, "the narrow route equals the CPU engine for every placement and staging pass");
        check(moved == 0, "no placement or staging pass changes an output bit");
        if (config.land) {
            // Oracle: the staged misses in job order (resident and CPU-served jobs excluded); landing slot n
            // receives the n-th of them on the forked route, and nothing on the other routes.
            std::int32_t job_count = 0;
            cuda_check(cudaMemcpy(&job_count, dispatch.job_count, sizeof(job_count), cudaMemcpyDeviceToHost), "cudaMemcpy");
            std::vector<std::int32_t> jobs(static_cast<std::size_t>(job_count));
            cuda_check(cudaMemcpy(jobs.data(), dispatch.jobs, jobs.size() * sizeof(std::int32_t), cudaMemcpyDeviceToHost),
                       "cudaMemcpy");
            const auto cpu = cpu_selection(config.service, jobs, config.streamed);
            std::vector<std::int32_t> staged;
            for (const std::int32_t e : jobs) {
                if (frames[e] < 0 && !(config.streamed && prefetched[e] >= 0) && cpu[e] == 0) { staged.push_back(e); }
            }
            const bool forked_route = config.fork && max_jobs <= std::min(slots, 512) && columns <= 8;
            std::vector<std::int32_t> landed(landing.size());
            cuda_check(cudaMemcpy(landed.data(), d_landed, landed.size() * sizeof(std::int32_t), cudaMemcpyDeviceToHost),
                       "cudaMemcpy");
            bool ok = true;
            for (std::size_t n = 0; n < landing.size(); ++n) {
                const std::int32_t want = forked_route && landing[n] >= 0 && n < staged.size() ? staged[n] : -1;
                ok = ok && landed[n] == want;
                if (want >= 0) {
                    std::vector<std::uint8_t> bytes(moe::kRecordBytes);
                    cuda_check(cudaMemcpy(bytes.data(), d_frame_base + static_cast<std::size_t>(landing[n]) * stride,
                                          bytes.size(), cudaMemcpyDeviceToHost), "cudaMemcpy");
                    ok = ok && std::memcmp(bytes.data(), bank[want].record.data(), bytes.size()) == 0;
                }
            }
            std::printf("  landing: %zu staged misses, landed %d %d %d %d\n", staged.size(), landed[0], landed[1],
                        landed[2], landed[3]);
            check(ok, "landing frames receive the first staged misses in job order (forked route only)");
            // The next configuration starts from free landing frames again.
            cuda_check(cudaMemset(d_frame_base + static_cast<std::size_t>(resident) * stride, 0, 3 * stride), "cudaMemset");
        }
    }
    if (experts == 12) {
        // A CPU channel whose host never answers and whose heartbeat stands still: the wait gives up
        // after heartbeat_timeout_ns and reports it in the error word instead of trapping, and the
        // device stays usable.
        const auto mapped = [](std::size_t bytes) {
            void* p = nullptr;
            cuda_check(cudaHostAlloc(&p, bytes, cudaHostAllocMapped | cudaHostAllocPortable), "cudaHostAlloc");
            std::memset(p, 0, bytes);
            return p;
        };
        auto* request = static_cast<moe::MissRequest*>(mapped(sizeof(moe::MissRequest)));
        auto* cx      = static_cast<std::uint16_t*>(mapped(sizeof(std::uint16_t) * moe::kHidden * moe::kMaxCpuXColumns));
        auto* cy      = static_cast<std::uint16_t*>(mapped(sizeof(std::uint16_t) * moe::kHidden * 8 * moe::kMaxCpuColumns));
        auto* words   = static_cast<std::uint32_t*>(mapped(64));
        std::uint32_t* sequence = nullptr;
        cuda_check(cudaMalloc(&sequence, sizeof(std::uint32_t)), "cudaMalloc");
        cuda_check(cudaMemset(sequence, 0, sizeof(std::uint32_t)), "cudaMemset");
        infernix::ops::MoeExpertSource source{.frame_base    = d_frame_base,
                                            .frames        = d_frames,
                                            .host_records  = host_device,
                                            .record_stride = stride,
                                            .scales        = d_scales,
                                            .staging_base  = d_staging,
                                            .staging_slots = 3};
        source.activation           = activation;
        source.heartbeat_timeout_ns = kSilentHeartbeatTimeoutNs;
        source.cpu = {.request = request, .x = cx, .y = cy, .done = words, .status = words + 1, .heartbeat = words + 2,
                      .sequence = sequence, .layer = 7, .max_jobs = 8, .max_columns = 64};
        *reinterpret_cast<volatile std::uint32_t*>(error_host) = 0;
        source.error = error_device;
        Tensor tx(d_x, DType::BF16, {moe::kHidden, columns});
        Tensor out(d_out, DType::BF16, {moe::kHidden, top_k * columns});
        const auto start = std::chrono::steady_clock::now();
        infernix::ops::moe_experts(tx, dispatch, source, top_k, max_jobs, d_workspace, out, nullptr);
        const cudaError_t status = cudaDeviceSynchronize();
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        const std::uint32_t error = *reinterpret_cast<volatile std::uint32_t*>(error_host);
        std::printf("silent CPU host: %s after %.2f s, error word %#x\n", cudaGetErrorString(status), seconds, error);
        check(status == cudaSuccess && error == (7U << 16U | moe::kErrorHostSilent),
              "a silent CPU host fails the call through its error word, without a trap");
        cuda_check(cudaMemset(d_out, 0, 16), "a device that is still usable");
        cudaFree(sequence);
        cudaFreeHost(words);
        cudaFreeHost(cy);
        cudaFreeHost(cx);
        cudaFreeHost(request);
    }
    for (auto event : fork_events) { cudaEventDestroy(event); }
    cudaStreamDestroy(fork_stream);
    for (auto event : overlap_events) { cudaEventDestroy(event); }
    cudaStreamDestroy(overlap_stream);
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
    cudaFree(d_prefetch_base);
    cudaFree(d_table);
    cudaFree(d_narrow_table);
    cudaFree(d_full_table);
    cudaFreeHost(error_host);
    cudaFreeHost(shuffled);
    cudaFree(d_prefetched);
    cudaFree(d_frames);
    cudaFree(d_landed);
    cudaFree(d_landing);
    cudaFree(d_frame_base);
    cudaFreeHost(host);
}

// moe_dispatch against an exact oracle: counts, offsets, jobs and job_count exact, each expert's
// entries as a set, and the route-log copy. The dispatch arrays start as garbage, and a captured
// call is replayed on new ids, so neither route may depend on memory cleared outside the call.
// Entries <= 1024 take the one-CTA kernel, more the count/scan/scatter kernels.
// Tail skip (MoeExpertSource::tail_skip_rank, --expert-tail-skip): with the CPU service planning the
// call, an expert neither resident nor wider than kMaxColumns columns that every column routes at slot
// >= rank has zero outputs; every other output keeps the CPU engine's bits (the exact narrow route).
// Both plans (moe_experts' own and moe_route_dispatch_plan's fused one), serial and forked.
void test_tail_skip(int experts, int columns, int top_k, int rank, std::uint32_t seed) {
    std::mt19937 rng(seed);
    const std::size_t stride = moe::kRecordBytes;
    std::vector<fixtures::Expert> bank;
    std::vector<moe::ExpertScales> scales;
    for (int e = 0; e < experts; ++e) {
        bank.push_back(fixtures::random_expert(rng, e % 3 == 0));
        scales.push_back(bank.back().scales);
    }
    std::uint8_t* host = nullptr;
    cuda_check(cudaHostAlloc(reinterpret_cast<void**>(&host), stride * experts, cudaHostAllocMapped), "cudaHostAlloc");
    for (int e = 0; e < experts; ++e) { std::memcpy(host + stride * e, bank[e].record.data(), moe::kRecordBytes); }
    std::uint8_t* host_device = nullptr;
    cuda_check(cudaHostGetDevicePointer(reinterpret_cast<void**>(&host_device), host, 0), "cudaHostGetDevicePointer");
    std::vector<std::int32_t> frames(experts, -1);
    std::vector<std::uint8_t> frame_bytes;
    int resident = 0;
    for (int e = 0; e < experts; e += 3) {
        frames[e] = resident++;
        frame_bytes.insert(frame_bytes.end(), bank[e].record.begin(), bank[e].record.end());
    }
    auto* d_frame_base = device_copy(frame_bytes);
    auto* d_frames     = device_copy(frames);
    auto* d_scales     = device_copy(scales);
    std::normal_distribution<float> n(0.0F, 1.0F);
    std::vector<float> logits(static_cast<std::size_t>(experts + 1) * columns);
    for (auto& v : logits) { v = n(rng); }
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
    const Tensor t_logits(d_logits, DType::FP32, {experts + 1, columns});
    infernix::ops::moe_route(t_logits, top_k, routing, nullptr);
    void* d_dispatch = nullptr;
    cuda_check(cudaMalloc(&d_dispatch, infernix::ops::moe_dispatch_bytes(experts, top_k * columns)), "cudaMalloc");
    auto dispatch = infernix::ops::carve_moe_dispatch(d_dispatch, experts, top_k * columns);
    const auto x = fixtures::random_activations(rng, columns);
    auto* d_x    = device_copy(x);
    std::vector<std::int32_t> ids(static_cast<std::size_t>(top_k) * columns);
    cuda_check(cudaMemcpy(ids.data(), d_ids, ids.size() * sizeof(std::int32_t), cudaMemcpyDeviceToHost), "cudaMemcpy");

    // The oracle: an expert is skipped when it misses VRAM, has at most kMaxColumns columns and every
    // column routes it at slot >= rank; its outputs are zero, every other output is the CPU engine's.
    std::vector<int> expert_columns(experts, 0), best_slot(experts, top_k);
    for (std::size_t i = 0; i < ids.size(); ++i) {
        ++expert_columns[ids[i]];
        best_slot[ids[i]] = std::min(best_slot[ids[i]], static_cast<int>(i % top_k));
    }
    const auto skipped = [&](int e) {
        return frames[e] < 0 && expert_columns[e] <= moe::kMaxColumns && best_slot[e] >= rank;
    };
    std::vector<std::uint16_t> expected(static_cast<std::size_t>(moe::kHidden) * top_k * columns, 0);
    int skipped_entries = 0, computed_misses = 0;
    for (int t = 0; t < columns; ++t) {
        for (int s = 0; s < top_k; ++s) {
            const int e = ids[static_cast<std::size_t>(t) * top_k + s];
            if (skipped(e)) {
                ++skipped_entries;
                continue;
            }
            computed_misses += frames[e] < 0 ? 1 : 0;
            const auto y = cpu_column(bank[e], &x[static_cast<std::size_t>(t) * moe::kHidden], moe::ExpertActivation::kA4);
            std::memcpy(&expected[(static_cast<std::size_t>(t) * top_k + s) * moe::kHidden], y.data(),
                        moe::kHidden * sizeof(std::uint16_t));
        }
    }
    check(skipped_entries > 0 && computed_misses > 0, "the tail-skip case skips some misses and computes others");

    const int max_jobs = std::min(experts, top_k * columns);
    std::uint8_t* d_staging = nullptr;
    cuda_check(cudaMalloc(&d_staging, stride * 64), "cudaMalloc");
    void* d_workspace = nullptr;
    cuda_check(cudaMalloc(&d_workspace, infernix::ops::moe_experts_workspace_bytes(infernix::ops::offloaded_moe::kFlashNextShape, max_jobs, top_k * columns, columns,
                                                                                   moe::ExpertActivation::kA4)),
               "cudaMalloc");
    std::uint16_t* d_out = nullptr;
    cuda_check(cudaMalloc(&d_out, expected.size() * sizeof(std::uint16_t)), "cudaMalloc");
    std::vector<moe::CpuMissService::Layer> layers{{.records = host, .record_stride = stride, .scales = scales.data()}};
    // Used at once: a request published before the service thread first runs must still be served.
    moe::CpuMissService service(layers, {.workers = 4, .max_jobs = 8, .max_columns = 64, .cpus = {}});
    cudaStream_t fork_stream = nullptr;
    cudaEvent_t fork_events[2] = {};
    cuda_check(cudaStreamCreateWithFlags(&fork_stream, cudaStreamNonBlocking), "cudaStreamCreate");
    for (auto& event : fork_events) { cuda_check(cudaEventCreateWithFlags(&event, cudaEventDisableTiming), "event"); }
    Tensor tx(d_x, DType::BF16, {moe::kHidden, columns});
    Tensor out(d_out, DType::BF16, {moe::kHidden, top_k * columns});
    for (const bool fork : {false, true}) {
        for (const bool fused : {false, true}) {
            infernix::ops::moe_dispatch(routing, experts, dispatch, nullptr, nullptr);
            cuda_check(cudaMemset(d_out, 0xFF, expected.size() * sizeof(std::uint16_t)), "cudaMemset");
            infernix::ops::MoeExpertSource source{.frame_base    = d_frame_base,
                                                .frames        = d_frames,
                                                .host_records  = host_device,
                                                .record_stride = stride,
                                                .scales        = d_scales,
                                                .staging_base  = d_staging,
                                                .staging_slots = 64,
                                                .cpu           = service.channel(0)};
            source.heartbeat_timeout_ns = kTestHeartbeatTimeoutNs;
            source.tail_skip_rank       = rank;
            if (fork) {
                source.fork_stream    = fork_stream;
                source.fork_events[0] = fork_events[0];
                source.fork_events[1] = fork_events[1];
            }
            const bool planned = fused && infernix::ops::moe_route_dispatch_plan(t_logits, top_k, routing, dispatch, nullptr,
                                                                                 tx, source, max_jobs, d_workspace, out, nullptr);
            infernix::ops::moe_experts(tx, dispatch, source, top_k, max_jobs, d_workspace, out, nullptr, true, planned);
            cuda_check(cudaDeviceSynchronize(), "moe_experts");
            std::vector<std::uint16_t> got(expected.size());
            cuda_check(cudaMemcpy(got.data(), d_out, got.size() * sizeof(std::uint16_t), cudaMemcpyDeviceToHost),
                       "cudaMemcpy");
            // Bits are compared where the CPU engine's are the route's (test_layer's `narrow`): every skipped
            // output, and experts of at most kMaxCpuColumns columns or with separate gate/up scales.
            long mismatches = 0;
            for (std::size_t i = 0; i < got.size(); ++i) {
                const int e = ids[i / moe::kHidden];
                if (skipped(e) || expert_columns[e] <= moe::kMaxCpuColumns || e % 3 == 0) {
                    mismatches += got[i] != expected[i] ? 1 : 0;
                }
            }
            std::printf("E=%d T=%d k=%d tail skip %d (%s, %s plan): %d entries skipped, %d misses computed, %ld "
                        "mismatches\n",
                        experts, columns, top_k, rank, fork ? "forked" : "serial", planned ? "fused" : "own",
                        skipped_entries, computed_misses, mismatches);
            check(mismatches == 0, "tail skip zeroes the skipped experts' outputs and keeps every other output exact");
        }
    }
    cuda_check(cudaStreamDestroy(fork_stream), "cudaStreamDestroy");
    for (auto& event : fork_events) { cuda_check(cudaEventDestroy(event), "event"); }
    for (void* p : {static_cast<void*>(d_frame_base), static_cast<void*>(d_frames), static_cast<void*>(d_scales),
                    static_cast<void*>(d_logits), static_cast<void*>(d_ids), static_cast<void*>(d_weights),
                    static_cast<void*>(d_shared_gate), d_dispatch, static_cast<void*>(d_x),
                    static_cast<void*>(d_staging), d_workspace, static_cast<void*>(d_out)}) {
        cuda_check(cudaFree(p), "cudaFree");
    }
    cuda_check(cudaFreeHost(host), "cudaFreeHost");
}

void test_dispatch(int experts, int entries, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> pick(0, experts - 1);
    const auto random_ids = [&] {
        std::vector<std::int32_t> ids(entries);
        // A few hot experts and many cold ones, as routing produces; some experts stay unused.
        for (auto& id : ids) { id = (rng() % 4 == 0) ? pick(rng) % 8 : pick(rng); }
        return ids;
    };
    std::int32_t* d_ids = nullptr;
    std::int32_t* d_log = nullptr;
    void* d_dispatch    = nullptr;
    const std::size_t bytes = infernix::ops::moe_dispatch_bytes(experts, entries);
    cuda_check(cudaMalloc(&d_ids, sizeof(std::int32_t) * entries), "cudaMalloc");
    cuda_check(cudaMalloc(&d_log, sizeof(std::int32_t) * entries), "cudaMalloc");
    cuda_check(cudaMalloc(&d_dispatch, bytes), "cudaMalloc");
    cuda_check(cudaMemset(d_dispatch, 0x5A, bytes), "cudaMemset");
    cuda_check(cudaMemset(d_log, 0xFF, sizeof(std::int32_t) * entries), "cudaMemset");
    auto dispatch = infernix::ops::carve_moe_dispatch(d_dispatch, experts, entries);
    infernix::ops::MoeRouting routing{Tensor(d_ids, DType::I32, {entries, 1}), Tensor{}, Tensor{}};

    const auto verify = [&](const std::vector<std::int32_t>& ids, const char* phase) {
        std::vector<std::int32_t> counts(experts), offsets(experts + 1), jobs(experts), got_entries(entries), log(entries);
        std::int32_t job_count = -1;
        cuda_check(cudaMemcpy(counts.data(), dispatch.counts, sizeof(std::int32_t) * experts, cudaMemcpyDeviceToHost), "copy");
        cuda_check(cudaMemcpy(offsets.data(), dispatch.offsets, sizeof(std::int32_t) * (experts + 1), cudaMemcpyDeviceToHost),
                   "copy");
        cuda_check(cudaMemcpy(jobs.data(), dispatch.jobs, sizeof(std::int32_t) * experts, cudaMemcpyDeviceToHost), "copy");
        cuda_check(cudaMemcpy(&job_count, dispatch.job_count, sizeof(std::int32_t), cudaMemcpyDeviceToHost), "copy");
        cuda_check(cudaMemcpy(got_entries.data(), dispatch.entries, sizeof(std::int32_t) * entries, cudaMemcpyDeviceToHost),
                   "copy");
        cuda_check(cudaMemcpy(log.data(), d_log, sizeof(std::int32_t) * entries, cudaMemcpyDeviceToHost), "copy");
        std::vector<std::vector<std::int32_t>> by_expert(experts);
        for (int i = 0; i < entries; ++i) { by_expert[ids[i]].push_back(i); }
        // The first mismatching field, so a failure names what went wrong.
        const char* first_bad = log == ids ? nullptr : "route log";
        int used = 0, start = 0, bad_expert = -1;
        for (int e = 0; e < experts && first_bad == nullptr; ++e) {
            const int c = static_cast<int>(by_expert[e].size());
            if (counts[e] != c) {
                first_bad = "counts";
            } else if (offsets[e] != start) {
                first_bad = "offsets";
            } else if (c > 0 && jobs[used++] != e) {
                first_bad = "jobs";
            } else if (c > 0) {
                std::vector<std::int32_t> mine(got_entries.begin() + start, got_entries.begin() + start + c);
                std::sort(mine.begin(), mine.end());
                if (mine != by_expert[e]) { first_bad = "entries"; }
            }
            if (first_bad != nullptr) { bad_expert = e; }
            start += c;
        }
        if (first_bad == nullptr && offsets[experts] != entries) { first_bad = "total offset"; }
        if (first_bad == nullptr && job_count != used) { first_bad = "job count"; }
        const bool ok = first_bad == nullptr;
        if (ok) {
            std::printf("dispatch E=%d entries=%d (%s): exact\n", experts, entries, phase);
        } else {
            std::printf("dispatch E=%d entries=%d (%s): MISMATCH in %s (expert %d)\n", experts, entries, phase,
                        first_bad, bad_expert);
        }
        check(ok, "moe_dispatch equals the exact oracle");
    };

    const auto first = random_ids();
    cuda_check(cudaMemcpy(d_ids, first.data(), sizeof(std::int32_t) * entries, cudaMemcpyHostToDevice), "copy");
    infernix::ops::moe_dispatch(routing, experts, dispatch, d_log, nullptr);
    cuda_check(cudaDeviceSynchronize(), "dispatch");
    verify(first, "eager");

    cudaStream_t stream = nullptr;
    cudaGraph_t graph   = nullptr;
    cudaGraphExec_t exec = nullptr;
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");
    cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "capture");
    infernix::ops::moe_dispatch(routing, experts, dispatch, d_log, stream);
    cuda_check(cudaStreamEndCapture(stream, &graph), "capture");
    cuda_check(cudaGraphInstantiate(&exec, graph, 0), "instantiate");
    for (int replay = 0; replay < 2; ++replay) {
        const auto ids = random_ids();
        // On the replay stream: a pageable cudaMemcpy may return before its DMA lands, and the
        // non-blocking stream does not wait for the legacy stream.
        cuda_check(cudaMemcpyAsync(d_ids, ids.data(), sizeof(std::int32_t) * entries, cudaMemcpyHostToDevice, stream),
                   "copy");
        cuda_check(cudaGraphLaunch(exec, stream), "launch");
        cuda_check(cudaStreamSynchronize(stream), "replay");
        verify(ids, replay == 0 ? "graph replay 1" : "graph replay 2");
    }
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    cudaFree(d_dispatch);
    cudaFree(d_log);
    cudaFree(d_ids);
}

// moe_combine_inject against moe_combine followed by hyper_connection_inject (each qualified against
// its own oracle): the residual streams bit-equal.
void test_combine_inject(int columns, int top_k, int streams, std::uint32_t seed) {
    constexpr int hidden = moe::kHidden;
    std::mt19937 rng(seed);
    std::normal_distribution<float> n(0.0F, 1.0F);
    const auto bf16s = [&](std::size_t count) {
        std::vector<std::uint16_t> v(count);
        for (auto& x : v) { x = infernix::test::f32_to_bf16(n(rng)); }
        return v;
    };
    const auto floats = [&](std::size_t count) {
        std::vector<float> v(count);
        for (auto& x : v) { x = n(rng); }
        return v;
    };
    const auto outputs = bf16s(static_cast<std::size_t>(hidden) * columns * top_k);
    const auto shared  = bf16s(static_cast<std::size_t>(hidden) * columns);
    const auto weights = floats(static_cast<std::size_t>(top_k) * columns);
    const auto gate = floats(columns), inject = floats(static_cast<std::size_t>(streams) * columns);
    const auto residual = bf16s(static_cast<std::size_t>(streams) * hidden * columns);
    auto* d_outputs = device_copy(outputs);
    auto* d_shared  = device_copy(shared);
    auto* d_weights = device_copy(weights);
    auto* d_gate    = device_copy(gate);
    auto* d_inject  = device_copy(inject);
    auto* d_split   = device_copy(residual);
    auto* d_fused   = device_copy(residual);
    std::uint16_t* d_y = nullptr;
    cuda_check(cudaMalloc(&d_y, shared.size() * sizeof(std::uint16_t)), "cudaMalloc");
    const infernix::ops::MoeRouting routing{Tensor{}, Tensor(d_weights, DType::FP32, {top_k, columns}),
                                            Tensor(d_gate, DType::FP32, {columns})};
    const Tensor t_outputs(d_outputs, DType::BF16, {hidden, columns * top_k});
    const Tensor t_shared(d_shared, DType::BF16, {hidden, columns});
    const Tensor t_inject(d_inject, DType::FP32, {streams, columns});
    Tensor t_y(d_y, DType::BF16, {hidden, columns});
    Tensor t_split(d_split, DType::BF16, {streams * hidden, columns});
    Tensor t_fused(d_fused, DType::BF16, {streams * hidden, columns});
    infernix::ops::moe_combine(t_outputs, routing, t_shared, t_y, nullptr);
    infernix::ops::hyper_connection_inject(t_y, t_inject, t_split, nullptr);
    infernix::ops::moe_combine_inject(t_outputs, routing, t_shared, t_inject, t_fused, nullptr);
    cuda_check(cudaDeviceSynchronize(), "combine inject");
    std::vector<std::uint16_t> split(residual.size()), fused(residual.size());
    cuda_check(cudaMemcpy(split.data(), d_split, split.size() * 2, cudaMemcpyDeviceToHost), "copy");
    cuda_check(cudaMemcpy(fused.data(), d_fused, fused.size() * 2, cudaMemcpyDeviceToHost), "copy");
    const bool same = split == fused && split != residual;
    std::printf("combine_inject T=%d k=%d S=%d: %s\n", columns, top_k, streams,
                same ? "bit-equal to combine + inject" : "DIFFERENT");
    check(same, "moe_combine_inject equals moe_combine followed by hyper_connection_inject");
    for (void* p : {static_cast<void*>(d_outputs), static_cast<void*>(d_shared), static_cast<void*>(d_weights),
                    static_cast<void*>(d_gate), static_cast<void*>(d_inject), static_cast<void*>(d_split),
                    static_cast<void*>(d_fused), static_cast<void*>(d_y)}) {
        cudaFree(p);
    }
}

// moe_route_dispatch, captured (so its kernel runs as a programmatic dependent) and replayed on new
// logits, against the oracles: ids exactly the top-k with lower-id ties (the logits repeat values
// so ties occur), weights and shared gate against the FP64 formula, the dispatch exact as in
// test_dispatch, and every output bit equal to moe_route followed by moe_dispatch. Up to 32
// columns take the one-kernel route, 33 the two calls.
void test_route_dispatch(int experts, int columns, int top_k, std::uint32_t seed) {
    std::mt19937 rng(seed);
    const int entries = columns * top_k;
    const std::size_t logit_count = static_cast<std::size_t>(experts + 1) * columns;
    const auto random_logits = [&] {
        std::uniform_int_distribution<int> level(-64, 64);
        std::vector<float> logits(logit_count);
        for (auto& v : logits) { v = static_cast<float>(level(rng)) / 16.0F; } // 129 levels: many ties
        return logits;
    };
    const auto alloc = [](std::size_t bytes) {
        void* p = nullptr;
        cuda_check(cudaMalloc(&p, bytes), "cudaMalloc");
        cuda_check(cudaMemset(p, 0x5A, bytes), "cudaMemset");
        return p;
    };
    const std::size_t dispatch_bytes = infernix::ops::moe_dispatch_bytes(experts, entries);
    auto* d_logits = static_cast<float*>(alloc(sizeof(float) * logit_count));
    struct Outputs {
        infernix::ops::MoeRouting routing;
        infernix::ops::MoeDispatch dispatch;
        void* dispatch_base;
        std::int32_t* log;
    };
    const auto make = [&] {
        Outputs o;
        o.routing = {Tensor(alloc(sizeof(std::int32_t) * entries), DType::I32, {top_k, columns}),
                     Tensor(alloc(sizeof(float) * entries), DType::FP32, {top_k, columns}),
                     Tensor(alloc(sizeof(float) * columns), DType::FP32, {columns})};
        o.dispatch_base = alloc(dispatch_bytes);
        o.dispatch      = infernix::ops::carve_moe_dispatch(o.dispatch_base, experts, entries);
        o.log           = static_cast<std::int32_t*>(alloc(sizeof(std::int32_t) * entries));
        return o;
    };
    Outputs fused = make(), split = make();
    const Tensor logits(d_logits, DType::FP32, {experts + 1, columns});
    const auto host = [](const void* p, std::size_t bytes) {
        std::vector<std::uint8_t> out(bytes);
        cuda_check(cudaMemcpy(out.data(), p, bytes, cudaMemcpyDeviceToHost), "copy");
        return out;
    };
    const auto verify = [&](const std::vector<float>& values, const char* phase) {
        infernix::ops::moe_route(logits, top_k, split.routing, nullptr);
        infernix::ops::moe_dispatch(split.routing, experts, split.dispatch, split.log, nullptr);
        cuda_check(cudaDeviceSynchronize(), "route and dispatch");
        std::vector<std::int32_t> ids(entries);
        std::vector<float> weights(entries), gate(columns);
        cuda_check(cudaMemcpy(ids.data(), fused.routing.ids.data, sizeof(std::int32_t) * entries, cudaMemcpyDeviceToHost),
                   "copy");
        cuda_check(cudaMemcpy(weights.data(), fused.routing.weights.data, sizeof(float) * entries,
                              cudaMemcpyDeviceToHost), "copy");
        cuda_check(cudaMemcpy(gate.data(), fused.routing.shared_gate.data, sizeof(float) * columns,
                              cudaMemcpyDeviceToHost), "copy");
        bool ids_ok = true, values_ok = true;
        for (int t = 0; t < columns; ++t) {
            const float* column = &values[static_cast<std::size_t>(t) * (experts + 1)];
            std::vector<int> order(experts);
            for (int e = 0; e < experts; ++e) { order[e] = e; }
            std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return column[a] > column[b]; });
            double sum = 0.0;
            for (int k = 0; k < top_k; ++k) { sum += std::exp(static_cast<double>(column[order[k]]) - column[order[0]]); }
            for (int k = 0; k < top_k; ++k) {
                const std::size_t i = static_cast<std::size_t>(t) * top_k + k;
                ids_ok = ids_ok && ids[i] == order[k];
                const double want = std::exp(static_cast<double>(column[order[k]]) - column[order[0]]) / sum;
                values_ok = values_ok && std::abs(weights[i] - want) <= 2e-6 * want;
            }
            const double want_gate = 1.0 / (1.0 + std::exp(-static_cast<double>(column[experts])));
            values_ok = values_ok && std::abs(gate[t] - want_gate) <= 2e-6 * want_gate;
        }
        // The dispatch: counts, offsets, jobs and job count exact, each expert's entries as a set.
        const auto words = [&](const std::int32_t* p, int n) {
            std::vector<std::int32_t> out(n);
            cuda_check(cudaMemcpy(out.data(), p, sizeof(std::int32_t) * n, cudaMemcpyDeviceToHost), "copy");
            return out;
        };
        const auto counts = words(fused.dispatch.counts, experts), offsets = words(fused.dispatch.offsets, experts + 1);
        const auto jobs = words(fused.dispatch.jobs, experts), got = words(fused.dispatch.entries, entries);
        const auto log = words(fused.log, entries);
        const int job_count = words(fused.dispatch.job_count, 1)[0];
        bool dispatch_ok = log == ids && offsets[experts] == entries;
        int used = 0;
        for (int e = 0, start = 0; e < experts; ++e) {
            std::vector<std::int32_t> want;
            for (int i = 0; i < entries; ++i) {
                if (ids[i] == e) { want.push_back(i); }
            }
            const int c = static_cast<int>(want.size());
            std::vector<std::int32_t> mine(got.begin() + start, got.begin() + start + c);
            std::sort(mine.begin(), mine.end());
            dispatch_ok = dispatch_ok && counts[e] == c && offsets[e] == start && mine == want &&
                          (c == 0 || jobs[used++] == e);
            start += c;
        }
        dispatch_ok = dispatch_ok && job_count == used;
        // Bit-equal to the two calls (entry order within an expert is unspecified, so as sets above).
        const bool same =
            host(fused.routing.ids.data, sizeof(std::int32_t) * entries) ==
                host(split.routing.ids.data, sizeof(std::int32_t) * entries) &&
            host(fused.routing.weights.data, sizeof(float) * entries) ==
                host(split.routing.weights.data, sizeof(float) * entries) &&
            host(fused.routing.shared_gate.data, sizeof(float) * columns) ==
                host(split.routing.shared_gate.data, sizeof(float) * columns);
        std::printf("route_dispatch E=%d T=%d k=%d (%s): ids %s, weights/gate %s, dispatch %s, %s the two calls\n",
                    experts, columns, top_k, phase, ids_ok ? "exact" : "MISMATCH", values_ok ? "ok" : "MISMATCH",
                    dispatch_ok ? "exact" : "MISMATCH", same ? "equal to" : "DIFFERENT from");
        check(ids_ok && values_ok && dispatch_ok && same, "moe_route_dispatch equals its oracles and the two calls");
    };

    auto values = random_logits();
    cuda_check(cudaMemcpy(d_logits, values.data(), sizeof(float) * logit_count, cudaMemcpyHostToDevice), "copy");
    infernix::ops::moe_route_dispatch(logits, top_k, fused.routing, fused.dispatch, fused.log, nullptr);
    verify(values, "eager");

    cudaStream_t stream  = nullptr;
    cudaGraph_t graph    = nullptr;
    cudaGraphExec_t exec = nullptr;
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");
    cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "capture");
    infernix::ops::moe_route_dispatch(logits, top_k, fused.routing, fused.dispatch, fused.log, stream);
    cuda_check(cudaStreamEndCapture(stream, &graph), "capture");
    cuda_check(cudaGraphInstantiate(&exec, graph, 0), "instantiate");
    values = random_logits();
    cuda_check(cudaMemcpyAsync(d_logits, values.data(), sizeof(float) * logit_count, cudaMemcpyHostToDevice, stream),
               "copy");
    cuda_check(cudaGraphLaunch(exec, stream), "launch");
    cuda_check(cudaStreamSynchronize(stream), "replay");
    verify(values, "graph replay");
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    for (auto* o : {&fused, &split}) {
        cudaFree(o->routing.ids.data);
        cudaFree(o->routing.weights.data);
        cudaFree(o->routing.shared_gate.data);
        cudaFree(o->dispatch_base);
        cudaFree(o->log);
    }
    cudaFree(d_logits);
}

// The CPU miss service refuses a job whose column count lies outside 1..kMaxCpuColumns (it would
// index past the job's x/y): the request is answered with EINVAL and no output, and the service
// keeps serving. Requests are published from the host the way the device writes them.
void test_service_column_bound() {
    std::mt19937 rng(503);
    const fixtures::Expert expert = fixtures::random_expert(rng, false);
    std::uint8_t* host            = nullptr;
    cuda_check(cudaHostAlloc(reinterpret_cast<void**>(&host), moe::kRecordBytes, cudaHostAllocMapped), "cudaHostAlloc");
    std::memcpy(host, expert.record.data(), moe::kRecordBytes);
    const std::vector<moe::ExpertScales> scales{expert.scales};
    std::vector<moe::CpuMissService::Layer> layers{{.records = host, .record_stride = moe::kRecordBytes,
                                                    .scales = scales.data()}};
    {
        moe::CpuMissService service(layers, {.workers = 2, .max_jobs = 2, .max_columns = 64, .cpus = {}});
        const infernix::ops::MoeCpuChannel channel = service.channel(0);
        const auto x = fixtures::random_activations(rng, 1);
        std::memcpy(channel.x, x.data(), sizeof(std::uint16_t) * moe::kHidden);
        auto* y = const_cast<std::uint16_t*>(channel.y);
        const std::size_t y_elements = static_cast<std::size_t>(moe::kHidden) * moe::kMaxCpuColumns;
        std::fill(y, y + y_elements, std::uint16_t{0xABCD});
        const auto publish = [&](std::uint32_t sequence, std::int32_t columns) -> std::uint32_t {
            moe::MissRequest& request = *channel.request;
            request.layer             = 0;
            request.jobs              = 1;
            request.tiered            = 0;
            request.expert[0]         = 0;
            request.ncols[0]          = columns;
            for (int c = 0; c < moe::kMaxCpuColumns; ++c) { request.column[0][c] = 0; }
            std::atomic_thread_fence(std::memory_order_release);
            *reinterpret_cast<volatile std::uint32_t*>(&request.sequence) = sequence;
            const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(30);
            while (*reinterpret_cast<const volatile std::uint32_t*>(channel.done) != sequence &&
                   std::chrono::steady_clock::now() < until) {
                std::this_thread::yield();
            }
            std::atomic_thread_fence(std::memory_order_acquire);
            check(*reinterpret_cast<const volatile std::uint32_t*>(channel.done) == sequence,
                  "the CPU miss service answers every request");
            return *reinterpret_cast<const volatile std::uint32_t*>(channel.status);
        };
        const auto untouched = [&] {
            return std::all_of(y, y + y_elements, [](std::uint16_t v) { return v == 0xABCD; });
        };
        check(publish(1, 0) == EINVAL && untouched(), "a job of no columns is refused (EINVAL) without output");
        check(publish(2, moe::kMaxCpuColumns + 1) == EINVAL && untouched(),
              "a job past kMaxCpuColumns columns is refused (EINVAL) without output");
        check(publish(3, -1) == EINVAL && untouched(), "a negative column count is refused (EINVAL)");
        check(publish(4, 1) == 0, "the service serves a valid job after refusing bad ones");
        const auto want = cpu_column(expert, x.data(), moe::ExpertActivation::kA4);
        check(std::equal(want.begin(), want.end(), y), "the valid job's output equals the CPU engine's");
        check(service.failure().empty(), "refused jobs do not fail the service");
    }
    cudaFreeHost(host);
    std::printf("CPU miss service: column counts outside 1..%d refused\n", moe::kMaxCpuColumns);
}

} // namespace

int main() {
    if (infernix::test::cuda_unavailable()) {
        std::printf("SKIP: no usable CUDA device\n");
        return 77;
    }
    try {
        for (const int entries : {1, 10, 1024, 1025, 40960}) { test_dispatch(512, entries, 100U + entries); }
        test_dispatch(1024, 1024, 31); // every expert slot of the one-CTA kernel
        test_dispatch(3, 1000, 37);    // few experts, long runs per expert
        for (const int columns : {1, 5, 32, 33}) { test_route_dispatch(512, columns, 10, 200U + columns); }
        test_route_dispatch(512, 64, 16, 233); // 1024 entries past the one-kernel column count
        test_route_dispatch(9, 3, 3, 239);     // few experts: lanes past E in every selection
        for (const int columns : {1, 5}) { test_combine_inject(columns, 10, 4, 300U + columns); }
        test_service_column_bound();
        test_layer(12, 1, 10, 7);  // decode: one column, ten experts
        test_layer(24, 8, 10, 11); // the widest forked call: more jobs than one 3-slot pass
        test_layer(9, 5, 3, 13);   // several columns per expert: several passes per job
        test_layer(9, 4, 3, 17);   // MTP verification width
        test_layer(9, 12, 3, 67);  // a call past the fork route, several columns per expert
        test_layer(96, 10, 10, 19); // more misses than the largest cap; publishes a subset of ten columns
        test_layer(400, 64, 10, 23); // more than 256 jobs: the plan ranks in several chunks
        test_layer(512, 512, 10, 29);  // a short prompt's call: ~10 columns per expert, hundreds of narrow jobs
        test_layer(512, 2048, 10, 31); // a wider call: x published from thousands of columns
        test_tail_skip(48, 1, 10, 7, 401);  // decode: one column, Strata's default rank
        test_tail_skip(48, 6, 10, 5, 403);  // MTP verification width: an expert skips only if every column ranks it low
        test_tail_skip(96, 16, 10, 8, 409); // two verify rows: past the fork route, wide experts never skipped
        // W4A16 experts (§16.2.1): decode, verification, several columns per expert, many misses, and
        // a short prompt's call (experts of more than eight columns on the narrow kernels). Calls of
        // four columns take the one-column kernels and of five or more the multi-column ones.
        constexpr auto kA16 = moe::ExpertActivation::kA16;
        test_layer(12, 1, 10, 41, kA16);
        test_layer(24, 8, 10, 43, kA16);
        test_layer(9, 5, 3, 47, kA16);
        test_layer(9, 4, 3, 53, kA16);
        test_layer(9, 12, 3, 71, kA16);
        test_layer(96, 10, 10, 59, kA16);
        test_layer(512, 512, 10, 61, kA16);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
    if (g_failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("all offloaded_moe layer checks passed\n");
    return 0;
}
