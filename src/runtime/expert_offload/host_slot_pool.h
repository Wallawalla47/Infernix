#pragma once

// Page-locked expert slots kept between SSD expert tiers of one process (an EngineHost; design
// docs/maintainer/qwen3_8-flash-next-design.md §19.3.7, "a reused host"). A tier built with a pool takes its
// chunks from it before it page-locks new memory, and gives them back when it ends with the record each slot
// then held: the next tier of the same artifact keeps those records where they are (its startup fill reads only
// the others), and none page-locks memory again. Page-locking costs about 0.11 s/GiB (Qwen3.8-Flash-Next's
// experts: 59 GiB) and the fill reads the artifact at the drive's rate (11 s for 51 GiB).
//
// The pool keeps only idle chunks: while a tier holds a chunk it is the tier's. Tiers may run one after another
// or side by side; each takes idle chunks of exactly the size it pins. What a start did not take is released
// (release_idle) once its tier holds what it took, so memory ends as a fresh start leaves it. It also carries two
// measurements a later start of the same machine takes as they are: the fill's read rate and the CPU expert team's
// settled rates. Thread-safe.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace infernix::runtime::expert_offload {

class HostSlotPool {
public:
    // Unheld slot marker in a chunk's keys.
    static constexpr std::int32_t kNoKey = -1;

    // `release` frees a chunk's memory (the tier's pinned release: cudaFreeHost).
    explicit HostSlotPool(std::function<void(void*)> release);
    ~HostSlotPool();
    HostSlotPool(const HostSlotPool&)            = delete;
    HostSlotPool& operator=(const HostSlotPool&) = delete;

    struct Taken {
        void* memory = nullptr;
        // The record each slot of the chunk holds (kNoKey: none), at the chunk's pitch; empty when the chunk last
        // held records of another identity or pitch (its memory is reused, its bytes are not).
        std::vector<std::int32_t> keys;
    };
    // An idle chunk of exactly `bytes`: one that held records of `identity` at `pitch` first (with its keys),
    // else any (without). Nothing when no idle chunk has that size.
    [[nodiscard]] std::optional<Taken> take(std::size_t bytes, std::string_view identity, std::uint64_t pitch);
    // A tier gives a chunk back: `keys` names the record each of its slots holds (kNoKey: none or partial).
    void keep(void* memory, std::size_t bytes, std::string identity, std::uint64_t pitch, std::vector<std::int32_t> keys);
    // Frees every idle chunk.
    void release_idle() noexcept;
    // Bytes of the idle chunks: page-locked memory a start may count as its own.
    [[nodiscard]] std::uint64_t idle_bytes() const;
    [[nodiscard]] std::size_t idle_chunks() const;
    // The artifact's read rate a startup fill measured (bytes a second) over its records: a later fill that reads
    // fewer (a reused pool's) takes the rate of the largest instead of its own (read_rate: nothing when no fill read
    // more than `records`).
    void note_read_rate(double bytes_per_second, std::size_t records);
    [[nodiscard]] std::optional<double> read_rate(std::size_t records) const;
    // The CPU expert team's settled rates a start measured on a quiet machine, for one configuration `key` (the
    // artifact, the team's workers and CPUs, its arithmetic and the records it reads): a later start of the same
    // configuration takes them instead of reading them again, since settling takes 4-8 s (a reading, a 4 s pause and
    // another, more while they rise; runtime/expert_offload/placement.h settle_cpu_rates). The latest stands.
    struct CpuRates {
        double experts = 0; // experts a second on one-column jobs
        double columns = 0; // expert columns a second on the narrow route's widest jobs
        std::string note;   // how they were measured (the startup report's parenthesis)
    };
    void note_cpu_rates(std::string key, CpuRates rates);
    [[nodiscard]] std::optional<CpuRates> cpu_rates(std::string_view key) const;

private:
    struct Idle {
        void* memory = nullptr;
        std::size_t bytes = 0;
        std::string identity;
        std::uint64_t pitch = 0;
        std::vector<std::int32_t> keys;
    };
    std::function<void(void*)> release_;
    mutable std::mutex mutex_;
    std::vector<Idle> idle_;
    double read_rate_          = 0;
    std::size_t rate_records_  = 0;
    std::vector<std::pair<std::string, CpuRates>> cpu_rates_;
};

} // namespace infernix::runtime::expert_offload
