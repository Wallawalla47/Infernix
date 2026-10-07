// The CPU expert worker team of offloaded_sparse_moe (docs/maintainer/qwen3_8-flash-next-design.md
// §10.3): any worker count, job mix and ISA gives the single-thread expert_forward bits, and the
// team survives many rounds with workers spinning and parking. Host-only: needs no GPU.

#include "ops/offloaded_sparse_moe/cpu/expert_team.h"
#include "ops/offloaded_moe_fixtures.h"

#include <chrono>
#include <cstdio>
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

Round make_round(std::mt19937& rng, const std::vector<fixtures::Expert>& pool, int n_jobs) {
    Round r;
    for (int j = 0; j < n_jobs; ++j) {
        const int n = 1 + static_cast<int>(rng() % moe::kMaxColumns);
        r.experts.push_back(pool[rng() % pool.size()]);
        r.x.push_back(fixtures::random_activations(rng, n));
        r.y.emplace_back(static_cast<std::size_t>(n) * moe::kHidden, 0xFFFF);
        r.ref.emplace_back(static_cast<std::size_t>(n) * moe::kHidden);
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
            job.x[c] = xr[c] = &r.x[j][static_cast<std::size_t>(c) * moe::kHidden];
            job.y[c] = &r.y[j][static_cast<std::size_t>(c) * moe::kHidden];
            yr[c]    = &r.ref[j][static_cast<std::size_t>(c) * moe::kHidden];
        }
        moe::expert_forward(g_quick ? moe::best_cpu_isa() : moe::CpuIsa::kScalar, job.record, job.scales, job.ncols, xr, yr);
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

} // namespace

int main(int argc, char** argv) {
    g_quick = argc > 1 && std::string(argv[1]) == "--quick";
    test_worker_counts();
    test_many_rounds_with_parking();
    if (g_failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("all offloaded_moe team checks passed\n");
    return 0;
}
