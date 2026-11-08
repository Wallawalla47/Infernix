#include "runtime/expert_offload/host_slot_pool.h"

#include <stdexcept>
#include <utility>

namespace infernix::runtime::expert_offload {

HostSlotPool::HostSlotPool(std::function<void(void*)> release) : release_(std::move(release)) {
    if (!release_) { throw std::invalid_argument("HostSlotPool: a release function is required"); }
}

HostSlotPool::~HostSlotPool() { release_idle(); }

std::optional<HostSlotPool::Taken> HostSlotPool::take(std::size_t bytes, std::string_view identity, std::uint64_t pitch) {
    std::lock_guard lock(mutex_);
    std::size_t found = idle_.size();
    for (std::size_t i = 0; i < idle_.size(); ++i) {
        if (idle_[i].bytes != bytes) { continue; }
        if (idle_[i].identity == identity && idle_[i].pitch == pitch) {
            found = i;
            break;
        }
        if (found == idle_.size()) { found = i; }
    }
    if (found == idle_.size()) { return std::nullopt; }
    Idle chunk = std::move(idle_[found]);
    idle_.erase(idle_.begin() + static_cast<std::ptrdiff_t>(found));
    Taken taken;
    taken.memory = chunk.memory;
    if (chunk.identity == identity && chunk.pitch == pitch) { taken.keys = std::move(chunk.keys); }
    return taken;
}

void HostSlotPool::keep(void* memory, std::size_t bytes, std::string identity, std::uint64_t pitch,
                        std::vector<std::int32_t> keys) {
    if (memory == nullptr) { return; }
    std::lock_guard lock(mutex_);
    idle_.push_back(Idle{memory, bytes, std::move(identity), pitch, std::move(keys)});
}

void HostSlotPool::release_idle() noexcept {
    std::vector<Idle> idle;
    {
        std::lock_guard lock(mutex_);
        idle.swap(idle_);
    }
    for (const Idle& chunk : idle) {
        try {
            release_(chunk.memory);
        } catch (...) {}
    }
}

std::uint64_t HostSlotPool::idle_bytes() const {
    std::lock_guard lock(mutex_);
    std::uint64_t bytes = 0;
    for (const Idle& chunk : idle_) { bytes += chunk.bytes; }
    return bytes;
}

std::size_t HostSlotPool::idle_chunks() const {
    std::lock_guard lock(mutex_);
    return idle_.size();
}

void HostSlotPool::note_read_rate(double bytes_per_second, std::size_t records) {
    std::lock_guard lock(mutex_);
    if (bytes_per_second > 0 && records > rate_records_) {
        read_rate_    = bytes_per_second;
        rate_records_ = records;
    }
}

std::optional<double> HostSlotPool::read_rate(std::size_t records) const {
    std::lock_guard lock(mutex_);
    if (rate_records_ <= records || read_rate_ <= 0) { return std::nullopt; }
    return read_rate_;
}

void HostSlotPool::note_cpu_rates(std::string key, CpuRates rates) {
    if (!(rates.experts > 0) || !(rates.columns > 0)) { return; }
    std::lock_guard lock(mutex_);
    for (auto& [held, kept] : cpu_rates_) {
        if (held == key) {
            kept = std::move(rates);
            return;
        }
    }
    cpu_rates_.emplace_back(std::move(key), std::move(rates));
}

std::optional<HostSlotPool::CpuRates> HostSlotPool::cpu_rates(std::string_view key) const {
    std::lock_guard lock(mutex_);
    for (const auto& [held, kept] : cpu_rates_) {
        if (held == key) { return kept; }
    }
    return std::nullopt;
}

} // namespace infernix::runtime::expert_offload
