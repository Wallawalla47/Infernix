// The CPU expert team's rates (docs/maintainer/qwen3_8-flash-next-design.md §19.3.12): measured once at
// startup, for the configured workers, so the decode miss split and the prefill CPU split follow the
// machine's CPU as they follow its link. A temporary team of the service's size computes rounds of jobs
// on real records from the pinned banks, or the SSD tier's RAM copies (across layers, past the
// last-level cache, as misses are) while
// the link copies records into the staging slots, since the GPU staging's DRAM reads compete with the
// CPU's: one-column jobs (memory-bound: experts per second, decode's misses and the split's per-expert
// cost) and kNarrowColumns-column jobs (the narrow route's widest: expert columns per second, the
// split's per-column cost). Three batches of each, the best counts; about 60 ms.

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
constexpr int kRepeats    = 3;   // batches per width, on distinct records
constexpr int kLoadCopies = 150; // records the link copies during a batch (~15 ms at 27.5 GB/s)
constexpr int kNarrowColumns = ops::offloaded_moe::kMaxCpuColumns; // the split's widest CPU job

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
        cpu_expert_rate_ = rate(1, 0);
        cpu_column_rate_ = rate(kNarrowColumns, (kRepeats + 1) * kRounds) * kNarrowColumns;
        CUDA_CHECK(cudaStreamDestroy(stream));
    }
    char text[256];
    std::snprintf(text, sizeof(text),
                  "CPU expert team: %u workers, %.0f experts/s and %.0f expert columns/s (%s); decode misses kept on "
                  "the link: 1/%d",
                  cpu_workers(), cpu_expert_rate_, cpu_column_rate_,
                  measured ? (tier_ ? "measured on the SSD tier's RAM copies" : "measured")
                           : "reference: no expert records in RAM yet",
                  pcie_divisor());
    diagnostic(text, DiagnosticLevel::Debug);
}

} // namespace infernix::models::qwen4_exp::detail
