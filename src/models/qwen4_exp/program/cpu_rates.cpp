// The CPU expert team's decode rate (docs/maintainer/qwen3_8-flash-next-design.md §19.3.12): measured
// once at startup, for the configured workers, so the decode miss split follows the machine's CPU as
// it follows its link. A temporary team of the service's size computes rounds of one-column jobs on
// real records from the pinned banks (across layers, past the last-level cache, as decode's misses
// are) while the link copies records into the staging slots, since the GPU staging's DRAM reads
// compete with the CPU's. Three batches, the best counts; about 30 ms.

#include "models/qwen4_exp/program/program_impl.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <vector>

namespace infernix::models::qwen4_exp::detail {
namespace {

constexpr int kRateJobs   = 16;  // jobs per round, as cpu_expert_jobs' default
constexpr int kRounds     = 8;   // timed rounds per batch: 128 records, about 0.35 GiB
constexpr int kRepeats    = 3;   // batches, on distinct records (1 GiB in all)
constexpr int kLoadCopies = 150; // records the link copies during a batch (~15 ms at 27.5 GB/s)

} // namespace

void ProgramImpl::measure_cpu_rate(std::uint64_t record_stride) {
    namespace moe      = ops::offloaded_moe;
    const auto& layers = parameters_.layers;
    const bool banks   = !layers.empty() && layers.front().moe.bank->planes.records != nullptr;
    if (banks) {
        moe::CpuExpertTeam team({.workers    = static_cast<int>(options_.cpu_expert_workers),
                                 .max_jobs   = kRateJobs,
                                 .activation = layers.front().moe.bank->activation});
        const auto L = static_cast<std::uint32_t>(layers.size());
        const auto E = c_.moe.experts;
        std::vector<std::uint16_t> x(moe::kHidden), y(static_cast<std::size_t>(moe::kHidden) * kRateJobs);
        for (std::size_t i = 0; i < x.size(); ++i) { x[i] = static_cast<std::uint16_t>(0x3C00U + (i * 7919U) % 640U); }
        std::vector<moe::CpuExpertJob> jobs(kRateJobs);
        const auto fill = [&](int round) {
            for (int k = 0; k < kRateJobs; ++k) {
                const auto& bank = *layers[static_cast<std::uint32_t>(round * kRateJobs + k) % L].moe.bank;
                const std::uint32_t expert = (static_cast<std::uint32_t>(round) * 131U + static_cast<std::uint32_t>(k) * 37U) % E;
                auto& job  = jobs[static_cast<std::size_t>(k)];
                job.record = reinterpret_cast<const std::uint8_t*>(bank.planes.records) +
                             static_cast<std::uint64_t>(expert) * bank.planes.record_stride;
                job.scales = bank.scales[expert];
                job.ncols  = 1;
                job.x[0]   = x.data();
                job.y[0]   = y.data() + static_cast<std::size_t>(k) * moe::kHidden;
            }
        };
        fill(kRepeats * kRounds); // warm-up round
        team.run(jobs);
        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        const std::byte* source = layers.front().moe.bank->planes.records;
        // The best of kRepeats timed batches, as the link probe takes its best: a batch another
        // process slowed would otherwise move the divisor between startups.
        double best = 0.0;
        for (int repeat = 0; repeat < kRepeats; ++repeat) {
            for (int i = 0; i < kLoadCopies; ++i) {
                CUDA_CHECK(cudaMemcpyAsync(static_cast<std::byte*>(staging_.p) + (i % kStagingSlots) * record_stride,
                                           source + static_cast<std::uint64_t>(i % static_cast<int>(E)) * record_stride,
                                           record_stride, cudaMemcpyHostToDevice, stream));
            }
            const auto start = std::chrono::steady_clock::now();
            for (int r = 0; r < kRounds; ++r) {
                fill(repeat * kRounds + r);
                team.run(jobs);
            }
            const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            best = std::max(best, kRounds * kRateJobs / seconds);
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
        CUDA_CHECK(cudaStreamDestroy(stream));
        cpu_expert_rate_ = best;
    }
    char text[192];
    std::snprintf(text, sizeof(text), "CPU expert team: %u workers, %.0f experts/s (%s); decode misses kept on the link: 1/%d",
                  options_.cpu_expert_workers, cpu_expert_rate_, banks ? "measured" : "reference: the banks stay in the artifact",
                  pcie_divisor());
    diagnostic(text, DiagnosticLevel::Debug);
}

} // namespace infernix::models::qwen4_exp::detail
