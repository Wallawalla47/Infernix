// The CPU expert worker team of offloaded_sparse_moe (docs/maintainer/qwen3_8-flash-next-design.md
// §10.3): any worker count, job mix and ISA gives the single-thread expert_forward bits, and the
// team survives many rounds with workers spinning and parking. Host-only: needs no GPU.

#include "ops/op_family.h"
#include "ops/offloaded_sparse_moe/cpu/expert_team.h"
#include "ops/offloaded_moe_fixtures.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace moe      = infernix::ops::offloaded_moe;
namespace fixtures = infernix::test::offloaded_moe;

namespace {

int g_failures = 0;
bool g_quick   = false; // --quick: fewer workers and rounds, for sanitizer builds

void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_failures;
    }
}

struct Round {
    std::vector<fixtures::Expert> experts;
    std::vector<std::vector<std::uint16_t>> x, y, ref;
    std::vector<int> ncols;
    std::vector<moe::CpuExpertJob> jobs;
};

Round make_round(std::mt19937& rng, const std::vector<fixtures::Expert>& pool, int n_jobs,
                 moe::ExpertActivation activation = moe::ExpertActivation::kA4,
                 const moe::ExpertShape& shape = moe::kFlashNextShape) {
    const bool a16 = activation == moe::ExpertActivation::kA16;
    const auto hidden = static_cast<std::size_t>(shape.hidden);
    Round r;
    for (int j = 0; j < n_jobs; ++j) {
        const int n = 1 + static_cast<int>(rng() % moe::kMaxColumns);
        r.experts.push_back(pool[rng() % pool.size()]);
        r.x.push_back(a16 ? fixtures::wide_range_activations(rng, n, shape.hidden)
                          : fixtures::random_activations(rng, n, shape.hidden));
        r.y.emplace_back(static_cast<std::size_t>(n) * hidden, 0xFFFF);
        r.ref.emplace_back(static_cast<std::size_t>(n) * hidden);
        r.ncols.push_back(n);
    }
    for (int j = 0; j < n_jobs; ++j) {
        moe::CpuExpertJob job;
        job.record = r.experts[j].record.data();
        job.scales = r.experts[j].scales;
        job.ncols  = r.ncols[j];
        const std::uint16_t* xr[moe::kMaxColumns];
        std::uint16_t* yr[moe::kMaxColumns];
        for (int c = 0; c < job.ncols; ++c) {
            job.x[c] = xr[c] = &r.x[j][static_cast<std::size_t>(c) * hidden];
            job.y[c] = &r.y[j][static_cast<std::size_t>(c) * hidden];
            yr[c]    = &r.ref[j][static_cast<std::size_t>(c) * hidden];
        }
        const moe::CpuIsa isa = g_quick ? moe::best_cpu_isa() : moe::CpuIsa::kScalar;
        if (a16) {
            moe::expert_forward_a16(isa, shape, job.record, job.scales, job.ncols, xr, yr);
        } else {
            moe::expert_forward(isa, job.record, job.scales, job.ncols, xr, yr);
        }
        r.jobs.push_back(job);
    }
    return r;
}

bool equal(const Round& r) {
    for (std::size_t j = 0; j < r.y.size(); ++j) {
        if (r.y[j] != r.ref[j]) { return false; }
    }
    return true;
}

void test_worker_counts() {
    std::mt19937 rng(5);
    std::vector<fixtures::Expert> pool;
    for (int i = 0; i < 4; ++i) { pool.push_back(fixtures::random_expert(rng, i == 1)); }
    const std::vector<int> counts = g_quick ? std::vector<int>{1, 3, 4} : std::vector<int>{1, 2, 3, 4, 7, 13, 40};
    for (int workers : counts) {
        moe::CpuExpertTeam team({.workers = workers, .spin_iterations = 256, .cpus = {}});
        for (int n_jobs : {1, 2, 5}) {
            Round r = make_round(rng, pool, n_jobs);
            team.run(r.jobs);
            if (!equal(r)) {
                std::fprintf(stderr, "workers %d jobs %d differ\n", workers, n_jobs);
                check(false, "team output equals expert_forward");
            }
        }
    }
}

// A16 teams (design §16.2.1): any worker count gives expert_forward_a16's bits, at Flash-Next's
// geometry and at Qwen3.5-122B-A10B's (up to I / 16 = 64 workers).
void test_worker_counts_a16(const moe::ExpertShape& shape) {
    std::mt19937 rng(6);
    std::vector<fixtures::Expert> pool;
    for (int i = 0; i < 4; ++i) { pool.push_back(fixtures::random_a16_expert(rng, shape)); }
    const int most = shape.h_blocks();
    const std::vector<int> counts = g_quick ? std::vector<int>{1, 4} : std::vector<int>{1, 2, 3, 5, 13, most};
    for (int workers : counts) {
        moe::CpuExpertTeam team({.workers = workers, .spin_iterations = 256,
                                 .activation = moe::ExpertActivation::kA16, .shape = shape, .cpus = {}});
        for (int n_jobs : {1, 3, 6}) {
            Round r = make_round(rng, pool, n_jobs, moe::ExpertActivation::kA16, shape);
            team.run(r.jobs);
            if (!equal(r)) {
                std::fprintf(stderr, "A16 workers %d jobs %d differ\n", workers, n_jobs);
                check(false, "A16 team output equals expert_forward_a16");
            }
        }
    }
}

void test_many_rounds_with_parking() {
    std::mt19937 rng(9);
    std::vector<fixtures::Expert> pool;
    for (int i = 0; i < 3; ++i) { pool.push_back(fixtures::random_expert(rng, false)); }
    moe::CpuExpertTeam team({.workers = 4, .spin_iterations = 64, .cpus = {}});
    int bad = 0;
    for (int round = 0; round < (g_quick ? 12 : 60); ++round) {
        Round r = make_round(rng, pool, 1 + round % 3);
        if (round % 10 == 0) { std::this_thread::sleep_for(std::chrono::milliseconds(2)); } // workers park
        team.run(r.jobs);
        bad += equal(r) ? 0 : 1;
    }
    check(bad == 0, "every round equals expert_forward, across spinning and parked workers");
}

// --bench: the team's rates on records spread over 256 slots (708 MB, beyond every cache, so each job reads DRAM as a
// decode miss does), W4A4 (shared input scales) and W4A16 jobs, rounds of kBenchJobs one- or eight-column jobs, at one
// and six workers. Experts/s and expert columns/s, as the engine's startup probe reports them.
// One thread: the A4 quantizer of an x column and SwiGLU of an h column, scalar and vector.
void bench_column_functions() {
    std::mt19937 rng(5);
    const auto x = fixtures::random_activations(rng, 1, moe::kHidden);
    const auto u = fixtures::random_activations(rng, 1, moe::kIntermediate);
    std::vector<infernix::ops::canon::A4Block> q(moe::kGateUpBlocks);
    std::vector<std::uint16_t> h(moe::kIntermediate);
    for (const moe::CpuIsa isa : {moe::CpuIsa::kScalar, moe::best_cpu_isa()}) {
        constexpr int kReps = 4000;
        auto start = std::chrono::steady_clock::now();
        for (int r = 0; r < kReps; ++r) { moe::quantize_a4(isa, x.data(), moe::kHidden, 0.01F, q.data()); }
        const double quantize = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        start = std::chrono::steady_clock::now();
        for (int r = 0; r < kReps; ++r) { moe::swiglu_bf16_n(isa, x.data(), u.data(), moe::kIntermediate, h.data()); }
        const double swiglu = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        std::printf("%-12s A4 of a %d-element column %6.2f us; SwiGLU of %d elements %6.2f us\n", moe::cpu_isa_name(isa),
                    moe::kHidden, 1e6 * quantize / kReps, moe::kIntermediate, 1e6 * swiglu / kReps);
    }
    // The BF16 row outputs of one column (gate, up and down rows), scalar on every ISA.
    std::vector<std::int64_t> s(2 * moe::kIntermediate + moe::kHidden);
    for (auto& e : s) { e = static_cast<std::int64_t>(rng() % (1U << 30)) * 1024 - (std::int64_t{1} << 39); }
    std::uint32_t sink = 0;
    constexpr int kReps = 4000;
    const auto start = std::chrono::steady_clock::now();
    for (int r = 0; r < kReps; ++r) {
        for (const std::int64_t e : s) { sink += infernix::ops::canon::a4_row_output(e + r, 1e-7F); }
    }
    const double rows = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::printf("row outputs of a column (%zu) %6.2f us (%u)\n", s.size(), 1e6 * rows / kReps, sink & 1U);
}

// One thread, one column, records in DRAM (64 slots): each phase of a W4A4 and a W4A16 expert.
void bench_phases() {
    namespace canon = infernix::ops::canon;
    constexpr int kRecords = 64, kRounds = 3;
    const std::size_t stride = moe::kRecordBytes;
    const moe::CpuIsa isa = moe::best_cpu_isa();
    std::mt19937 rng(9);
    for (const bool a16 : {false, true}) {
        std::vector<fixtures::Expert> distinct;
        for (int i = 0; i < 4; ++i) { distinct.push_back(a16 ? fixtures::random_a16_expert(rng) : fixtures::random_expert(rng, false)); }
        std::vector<std::uint8_t> bank(stride * kRecords);
        for (int s = 0; s < kRecords; ++s) {
            std::memcpy(bank.data() + stride * s, distinct[static_cast<std::size_t>(s % 4)].record.data(), stride);
        }
        const auto x = a16 ? fixtures::wide_range_activations(rng, 1, moe::kHidden) : fixtures::random_activations(rng, 1, moe::kHidden);
        std::vector<canon::A4Block> xq(moe::kGateUpBlocks), hq(moe::kHBlocks);
        std::vector<canon::A16Block> xb(moe::kGateUpBlocks), hb(moe::kHBlocks);
        std::vector<std::uint16_t> hv(moe::kIntermediate), y(moe::kHidden);
        double t[4] = {};
        for (int r = 0; r < kRounds * kRecords; ++r) {
            const std::uint8_t* record = bank.data() + stride * static_cast<std::size_t>(r % kRecords);
            const moe::ExpertScales& scales = distinct[static_cast<std::size_t>(r % 4)].scales;
            std::uint16_t* yp = y.data();
            const auto t0 = std::chrono::steady_clock::now();
            std::chrono::steady_clock::time_point t1, t2, t3;
            if (a16) {
                const canon::A16Block* xp = xb.data();
                const canon::A16Block* hp = hb.data();
                std::uint16_t* hvp = hv.data();
                const int xe = moe::encode_a16(x.data(), moe::kHidden, xb.data());
                t1 = std::chrono::steady_clock::now();
                moe::gate_up_units_a16(isa, moe::kFlashNextShape, record, scales, &xp, &xe, 1, 0, moe::kHBlocks, &hvp);
                t2 = std::chrono::steady_clock::now();
                const int he = moe::encode_a16(hv.data(), moe::kIntermediate, hb.data());
                t3 = std::chrono::steady_clock::now();
                moe::down_rows_a16(isa, moe::kFlashNextShape, record, scales, &hp, &he, 1, 0, moe::kDownRowGroups, &yp);
            } else {
                const canon::A4Block* xp = xq.data();
                canon::A4Block* hp = hq.data();
                const canon::A4Block* hcp = hq.data();
                moe::quantize_a4(isa, x.data(), moe::kHidden, scales.input_gate, xq.data());
                t1 = std::chrono::steady_clock::now();
                moe::gate_up_units(isa, record, scales, &xp, &xp, 1, 0, moe::kHBlocks, &hp);
                t2 = t3 = std::chrono::steady_clock::now();
                moe::down_rows(isa, record, scales, &hcp, 1, 0, moe::kDownRowGroups, &yp);
            }
            const auto t4 = std::chrono::steady_clock::now();
            if (r < kRecords) { continue; } // the first pass warms the code, not the records
            t[0] += std::chrono::duration<double>(t1 - t0).count();
            t[1] += std::chrono::duration<double>(t2 - t1).count();
            t[2] += std::chrono::duration<double>(t3 - t2).count();
            t[3] += std::chrono::duration<double>(t4 - t3).count();
        }
        const double n = (kRounds - 1) * kRecords / 1e6;
        std::printf("%s one column, one thread (%s): x %6.2f us, gate/up %7.2f us, h %6.2f us, down %7.2f us\n",
                    a16 ? "W4A16" : "W4A4 ", moe::cpu_isa_name(isa), t[0] / n, t[1] / n, t[2] / n, t[3] / n);
    }
}

void bench_team() {
    bench_column_functions();
    bench_phases();
    constexpr int kSlots = 256, kDistinct = 16, kBenchJobs = 8;
    const std::size_t stride = moe::kRecordBytes;
    for (const moe::ExpertActivation activation : {moe::ExpertActivation::kA4, moe::ExpertActivation::kA16}) {
        const bool a16 = activation == moe::ExpertActivation::kA16;
        std::mt19937 rng(77);
        std::vector<fixtures::Expert> distinct;
        for (int i = 0; i < kDistinct; ++i) {
            distinct.push_back(a16 ? fixtures::random_a16_expert(rng) : fixtures::random_expert(rng, false));
        }
        std::vector<std::uint8_t> bank(stride * kSlots);
        for (int s = 0; s < kSlots; ++s) {
            std::memcpy(bank.data() + stride * s, distinct[static_cast<std::size_t>(s % kDistinct)].record.data(), stride);
        }
        const auto x = a16 ? fixtures::wide_range_activations(rng, moe::kMaxColumns, moe::kHidden)
                           : fixtures::random_activations(rng, moe::kMaxColumns, moe::kHidden);
        std::vector<std::uint16_t> y(static_cast<std::size_t>(kBenchJobs) * moe::kMaxColumns * moe::kHidden);
        for (const int workers : {1, 6}) {
            moe::CpuExpertTeam team({.workers = workers, .max_jobs = kBenchJobs, .activation = activation, .cpus = {}});
            for (const int columns : {1, 8}) {
                std::vector<moe::CpuExpertJob> jobs(kBenchJobs);
                int slot = 0;
                const auto fill = [&] {
                    for (int j = 0; j < kBenchJobs; ++j, slot = (slot + 1) % kSlots) {
                        auto& job  = jobs[static_cast<std::size_t>(j)];
                        job.record = bank.data() + stride * slot;
                        job.scales = distinct[static_cast<std::size_t>(slot % kDistinct)].scales;
                        job.ncols  = columns;
                        for (int c = 0; c < columns; ++c) {
                            job.x[c] = x.data() + static_cast<std::size_t>(c) * moe::kHidden;
                            job.y[c] = y.data() + (static_cast<std::size_t>(j) * moe::kMaxColumns + c) * moe::kHidden;
                        }
                    }
                };
                for (int warm = 0; warm < 4; ++warm) {
                    fill();
                    team.run(jobs);
                }
                const int rounds = workers == 1 ? 24 : 160;
                const auto start = std::chrono::steady_clock::now();
                for (int r = 0; r < rounds; ++r) {
                    fill();
                    team.run(jobs);
                }
                const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
                const double experts = static_cast<double>(rounds) * kBenchJobs / seconds;
                std::printf("%s workers %d columns %d: %8.0f experts/s, %8.0f expert columns/s (%.1f us an expert, "
                            "%.2f GB/s of records)\n",
                            a16 ? "W4A16" : "W4A4 ", workers, columns, experts, experts * columns, 1e6 / experts,
                            experts * static_cast<double>(stride) / 1e9);
            }
        }
    }
}

} // namespace

INFERNIX_OP_TEST(offloaded_moe_team) {
    if (argc > 1 && std::string(argv[1]) == "--bench") {
        bench_team();
        return 0;
    }
    g_quick = argc > 1 && std::string(argv[1]) == "--quick";
    test_worker_counts();
    test_worker_counts_a16(moe::kFlashNextShape);
    test_worker_counts_a16(moe::kQwen3_5_122BShape);
    test_many_rounds_with_parking();
    if (g_failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("all offloaded_moe team checks passed\n");
    return 0;
}
