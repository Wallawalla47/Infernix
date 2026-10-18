// The CPU expert team's rates (docs/maintainer/qwen3_8-flash-next-design.md §19.3.12): measured once at
// startup, for the configured workers, so the decode miss split and the prefill CPU split follow the
// machine's CPU as they follow its link. A temporary team of the service's size computes rounds of jobs
// on real records from the pinned banks, or the SSD tier's RAM copies (across layers, past the
// last-level cache, as misses are) while
// the link copies records into the staging slots, since the GPU staging's DRAM reads compete with the
// CPU's: one-column jobs (memory-bound: experts per second, decode's misses and the split's per-expert
// cost) and kNarrowColumns-column jobs (the narrow route's widest: expert columns per second, the
// split's per-column cost). Three batches of each, the best counts; about 60 ms.
//
// The rates hold for the whole session, so they are taken on a quiet machine: an attempt during
// which other processes used kQuietShare or more of the CPU (another program's build, a virus scan)
// is repeated after a pause, for up to kRetryBudget, and the best rates of the attempts stand. With
// every core loaded by another program the team reads about half its rate (i9-13900K, 6 workers:
// ~10,500 against ~19,500 experts/s on synthetic records), which turned the decode divisor from 3
// to 2 and slowed decode by a fifth for the whole session (design §19.3.28).

#include "models/qwen4_exp/program/program_impl.h"

#include "core/host_load.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

namespace infernix::models::qwen4_exp::detail {
namespace {

constexpr int kRateJobs   = 16;  // jobs per round, as cpu_expert_jobs' default
constexpr int kRounds     = 8;   // timed rounds per batch: 128 records, about 0.35 GiB
constexpr int kRepeats    = 3;   // batches per width, on distinct records
constexpr int kLoadCopies = 150; // records the link copies during a batch (~15 ms at 27.5 GB/s)
constexpr int kNarrowColumns = ops::offloaded_moe::kMaxCpuColumns; // the split's widest CPU job
// Other processes' share of the CPU under which an attempt stands, the pause between attempts and
// the time to keep trying. An idle desktop measured 2-7 %. On an i9-13900K (32 logical CPUs) the
// 6-worker team reads 2 / 6 / 11 / 13 / 35 / 46 % low with 4 / 8 / 12 / 16 / 24 / 32 busy processes
// (12.5-100 %), and the decode divisor (C t switching at 1.35) holds to about 18 % low: below 25 % an
// attempt is at most ~6 % low, clear of both.
constexpr double kQuietShare = 0.25;
constexpr auto kRetryPause   = std::chrono::milliseconds(500);
constexpr auto kRetryBudget  = std::chrono::seconds(20);

} // namespace

void ProgramImpl::measure_cpu_rate(std::uint64_t record_stride) {
    namespace moe      = ops::offloaded_moe;
    const auto& layers = parameters_.layers;
    const auto L       = static_cast<std::uint32_t>(layers.size());
    const auto E       = c_.moe.experts;
    // The records the jobs read: the pinned banks, or the SSD tier's RAM copies once it is filled.
    const auto record_of = [&](std::uint32_t layer, std::uint32_t expert) -> const std::uint8_t* {
        const auto& bank = *layers[layer].moe.bank;
        if (bank.planes.records != nullptr) {
            return reinterpret_cast<const std::uint8_t*>(bank.planes.records) +
                   static_cast<std::uint64_t>(expert) * bank.planes.record_stride;
        }
        return tier_ ? tier_->record(tier_->key(layer, expert)) : nullptr;
    };
    std::vector<std::vector<std::uint32_t>> held(L); // per layer, the experts with a host copy
    bool measured = L > 0;
    int attempts = 0;
    bool quiet   = true;
    double first_share = 0.0, last_share = 0.0; // other processes' CPU share in the first and last attempt
    for (std::uint32_t l = 0; l < L && measured; ++l) {
        for (std::uint32_t e = 0; e < E; ++e) {
            if (record_of(l, e) != nullptr) { held[l].push_back(e); }
        }
        measured = !held[l].empty();
    }
    if (measured) {
        moe::CpuExpertTeam team({.workers    = static_cast<int>(cpu_workers()),
                                 .max_jobs   = kRateJobs,
                                 .activation = layers.front().moe.bank->activation,
                                 .ggml       = ggml_format(*layers.front().moe.bank).valid()});
        std::vector<std::uint16_t> x(static_cast<std::size_t>(moe::kHidden) * kNarrowColumns);
        std::vector<std::uint16_t> y(static_cast<std::size_t>(moe::kHidden) * kRateJobs * kNarrowColumns);
        for (std::size_t i = 0; i < x.size(); ++i) { x[i] = static_cast<std::uint16_t>(0x3C00U + (i * 7919U) % 640U); }
        std::vector<moe::CpuExpertJob> jobs(kRateJobs);
        const auto fill = [&](int round, int columns) {
            for (int k = 0; k < kRateJobs; ++k) {
                const std::uint32_t layer = static_cast<std::uint32_t>(round * kRateJobs + k) % L;
                const auto& bank          = *layers[layer].moe.bank;
                const auto& experts       = held[layer];
                const std::uint32_t expert =
                    experts[(static_cast<std::uint32_t>(round) * 131U + static_cast<std::uint32_t>(k) * 37U) % experts.size()];
                auto& job  = jobs[static_cast<std::size_t>(k)];
                job.record = record_of(layer, expert);
                job.ggml   = ggml_format(bank);
                job.scales = job.ggml.valid() ? moe::ExpertScales{} : bank.scales[expert];
                job.ncols  = columns;
                for (int c = 0; c < columns; ++c) {
                    job.x[c] = x.data() + static_cast<std::size_t>(c) * moe::kHidden;
                    job.y[c] = y.data() + (static_cast<std::size_t>(k) * kNarrowColumns + c) * moe::kHidden;
                }
            }
        };
        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        const std::uint64_t source_stride = layers.front().moe.bank->planes.record_stride; // ≤ the staging pitch
        // Jobs per second at `columns` columns: the best of kRepeats timed batches, as the link probe
        // takes its best (a batch another process slowed would otherwise move the splits between
        // startups). Each width's batches use their own records.
        const auto rate = [&](int columns, int first_round) {
            fill(first_round + kRepeats * kRounds, columns); // warm-up round
            team.run(jobs);
            double best = 0.0;
            for (int repeat = 0; repeat < kRepeats; ++repeat) {
                for (int i = 0; i < kLoadCopies; ++i) {
                    const auto& first = held.front();
                    CUDA_CHECK(cudaMemcpyAsync(static_cast<std::byte*>(staging_.p) + (i % kStagingSlots) * record_stride,
                                               record_of(0, first[static_cast<std::size_t>(i) % first.size()]),
                                               source_stride, cudaMemcpyHostToDevice, stream));
                }
                const auto start = std::chrono::steady_clock::now();
                for (int r = 0; r < kRounds; ++r) {
                    fill(first_round + repeat * kRounds + r, columns);
                    team.run(jobs);
                }
                const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
                best = std::max(best, kRounds * kRateJobs / seconds);
                CUDA_CHECK(cudaStreamSynchronize(stream));
            }
            return best;
        };
        // Attempts until one runs while other processes leave the CPU quiet, or the budget is spent.
        const auto began = std::chrono::steady_clock::now();
        double expert_rate = 0.0, column_rate = 0.0;
        for (;;) {
            const HostCpuTimes before = host_cpu_times();
            const double experts      = rate(1, 0);
            const double columns      = rate(kNarrowColumns, (kRepeats + 1) * kRounds) * kNarrowColumns;
            const double others       = other_process_cpu_share(before, host_cpu_times());
            if (attempts == 0) { first_share = others; }
            last_share = others;
            ++attempts;
            expert_rate = std::max(expert_rate, experts);
            column_rate = std::max(column_rate, columns);
            quiet       = others < kQuietShare;
            if (quiet || std::chrono::steady_clock::now() - began >= kRetryBudget) { break; }
            std::this_thread::sleep_for(kRetryPause);
        }
        cpu_expert_rate_ = expert_rate;
        cpu_column_rate_ = column_rate;
        CUDA_CHECK(cudaStreamDestroy(stream));
    }
    std::string how = measured ? (tier_ ? "measured on the SSD tier's RAM copies" : "measured")
                               : "reference: no expert records in RAM yet";
    char share[160];
    if (measured && attempts > 1) {
        std::snprintf(share, sizeof(share), " after %d attempts: other processes used %.0f %% of the CPU at first, %.0f %% last",
                      attempts, 100.0 * first_share, 100.0 * last_share);
        how += share;
    } else if (measured) {
        std::snprintf(share, sizeof(share), "; other processes used %.0f %% of the CPU", 100.0 * last_share);
        how += share;
    }
    char text[512];
    std::snprintf(text, sizeof(text),
                  "CPU expert team: %u workers, %.0f experts/s and %.0f expert columns/s (%s); decode misses kept on "
                  "the link: 1/%d",
                  cpu_workers(), cpu_expert_rate_, cpu_column_rate_, how.c_str(), pcie_divisor());
    diagnostic(text, DiagnosticLevel::Debug);
    if (measured && !quiet) {
        diagnostic("the CPU expert team's rates were measured while other processes kept the CPU busy (" +
                       std::to_string(static_cast<int>(100.0 * last_share + 0.5)) +
                       " %); the decode miss split follows them for this session and may leave decode slower; "
                       "start the engine when the machine is quieter",
                   DiagnosticLevel::Warning);
    }
}

} // namespace infernix::models::qwen4_exp::detail
