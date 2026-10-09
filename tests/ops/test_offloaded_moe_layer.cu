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
// same configurations against expert_forward_a16 (their wide route's BF16 GEMM, too, is qualified
// in test_offloaded_moe_wide).
#include "infernix/ops/offloaded_sparse_moe.h"
#include "ops/offloaded_moe_fixtures.h"
#include "ops/offloaded_sparse_moe/cpu/fetch_channel.h"
#include "ops/offloaded_sparse_moe/cpu/miss_service.h"
#include "ops/offloaded_sparse_moe/cpu/w4a16_expert.h"
#include "ops/offloaded_sparse_moe/cpu/w4a4_expert.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
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
        moe::expert_forward_a16(moe::best_cpu_isa(), e.record.data(), e.scales, 1, xp, yp);
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
    cuda_check(cudaMalloc(&d_workspace, infernix::ops::moe_experts_workspace_bytes(max_jobs, top_k * columns)),
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
    // with at most max_job_columns columns, so the number served is min(want, eligible misses).
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
        const int want = std::min(cap - take, channel.pcie_divisor > 0 ? misses - misses / channel.pcie_divisor : misses);
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
        const int want = std::min(cap, channel.pcie_divisor > 0 ? misses - misses / channel.pcie_divisor : misses);
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
    // Outputs of the narrow route (design §8.5): an expert with at most eight columns, or one whose
    // gate and up projections keep separate input scales (the W4A4 fixture's every third expert).
    std::vector<std::uint8_t> narrow(static_cast<std::size_t>(top_k) * columns);
    for (std::size_t i = 0; i < narrow.size(); ++i) {
        narrow[i] = expert_columns[ids[i]] <= moe::kMaxCpuColumns || (!a16 && ids[i] % 3 == 0);
    }
    std::vector<std::uint16_t> placed;
    // The fork stream and its events, for the one-pass decode/verification route.
    cudaStream_t fork_stream = nullptr;
    cudaEvent_t fork_events[2] = {};
    cuda_check(cudaStreamCreateWithFlags(&fork_stream, cudaStreamNonBlocking), "cudaStreamCreate");
    for (auto& event : fork_events) { cuda_check(cudaEventCreateWithFlags(&event, cudaEventDisableTiming), "event"); }
    for (const Config config : {Config{0, nullptr}, Config{1, nullptr}, Config{3, nullptr}, Config{64, nullptr},
                                Config{3, &service_two}, Config{64, &service_eight}, Config{0, &service_eight},
                                Config{64, nullptr, true}, Config{64, &service_two, true}, Config{3, nullptr, true},
                                Config{3, &service_all}, Config{64, &service_all, true}, Config{64, &service_narrow},
                                Config{64, &service_narrow, true}, Config{64, nullptr, true, true},
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
                                       .fetch = Responder::Mode::kSilent, .fetched = true}}) {
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
        source.activation = activation;
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
        const std::uint64_t served_before = config.service != nullptr ? config.service->served_experts() : 0;
        infernix::ops::moe_experts(tx, dispatch, source, top_k, max_jobs, d_workspace, out, nullptr);
        cuda_check(cudaDeviceSynchronize(), "moe_experts");
        if (config.service != nullptr) {
            // The service counts after answering, so its count may trail the device by a moment.
            const auto* ssd = config.ssd == 1 ? &ssd_narrow : config.ssd == 2 ? &ssd_full : nullptr;
            const auto want = static_cast<std::uint64_t>(expected_cpu_jobs(*config.service, config.streamed, ssd));
            const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(1);
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
                const std::uint32_t want = config.fetch == Responder::Mode::kFail ? EIO : moe::kErrorHostSilent;
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
                const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(1);
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
        // after kHeartbeatTimeoutNs and reports it in the error word instead of trapping, and the
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
        source.activation = activation;
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
        test_layer(12, 1, 10, 7);  // decode: one column, ten experts
        test_layer(24, 8, 10, 11); // verify width: more jobs than one 3-slot pass
        test_layer(9, 5, 3, 13);   // several columns per expert
        test_layer(9, 4, 3, 17);   // MTP verification width: one-column kernels, several passes per job
        test_layer(96, 10, 10, 19); // more misses than the largest cap; publishes a subset of ten columns
        test_layer(400, 64, 10, 23); // more than 256 jobs: the plan ranks in several chunks
        test_layer(512, 512, 10, 29);  // a short prompt's call: ~10 columns per expert, hundreds of narrow jobs
        test_layer(512, 2048, 10, 31); // a wider call: x published from thousands of columns
        // W4A16 experts (§16.2.1): decode, verification, several columns per expert, many misses, and
        // a short prompt's call (experts of more than eight columns on the narrow kernels).
        constexpr auto kA16 = moe::ExpertActivation::kA16;
        test_layer(12, 1, 10, 41, kA16);
        test_layer(24, 8, 10, 43, kA16);
        test_layer(9, 5, 3, 47, kA16);
        test_layer(9, 4, 3, 53, kA16);
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
