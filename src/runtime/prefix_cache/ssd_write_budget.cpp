#include "runtime/prefix_cache/ssd_write_budget.h"

#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace infernix::runtime::prefix_cache {
namespace {

constexpr const char* kStateTag = "infernix-ssd-write-budget";
constexpr double kHourSeconds   = 3'600.0;

double seconds_between(SsdWriteBudget::Clock::time_point from, SsdWriteBudget::Clock::time_point to) {
    return std::chrono::duration<double>(to - from).count();
}

} // namespace

SsdWriteBudget::SsdWriteBudget(std::uint64_t bytes_per_hour, std::filesystem::path state_file, Clock::time_point now)
    : bytes_per_hour_(bytes_per_hour), level_(static_cast<double>(bytes_per_hour)), last_(now),
      state_file_(std::move(state_file)) {
    if (bytes_per_hour == 0) { throw std::invalid_argument("an SSD write budget needs a positive hourly size"); }
    // A saved level refills from the time it was saved; a missing or unreadable file starts full.
    std::ifstream in(state_file_);
    std::string tag;
    double level           = 0.0;
    long long saved_millis = 0;
    if (in >> tag >> level >> saved_millis && tag == kStateTag && level >= 0.0) {
        level_ = std::min(level, static_cast<double>(bytes_per_hour_));
        last_  = Clock::time_point{std::chrono::milliseconds{saved_millis}};
        refill(now);
    }
}

void SsdWriteBudget::refill(Clock::time_point now) noexcept {
    const double seconds = seconds_between(last_, now);
    if (seconds > 0.0) {
        level_ = std::min(static_cast<double>(bytes_per_hour_), level_ + seconds * static_cast<double>(bytes_per_hour_) / kHourSeconds);
    }
    last_ = std::max(last_, now);
}

double SsdWriteBudget::threshold(double share) const {
    // The value above which the recent candidates hold `share` of their bytes.
    std::vector<Offer> offers(recent_.begin(), recent_.begin() + static_cast<std::ptrdiff_t>(recent_count_));
    std::sort(offers.begin(), offers.end(), [](const Offer& a, const Offer& b) { return a.value > b.value; });
    double total = 0.0;
    for (const Offer& offer : offers) { total += static_cast<double>(offer.bytes); }
    double kept = 0.0;
    for (const Offer& offer : offers) {
        kept += static_cast<double>(offer.bytes);
        if (kept >= share * total) { return offer.value; }
    }
    return offers.empty() ? 0.0 : offers.back().value;
}

bool SsdWriteBudget::admit(double value, std::uint64_t bytes, bool first, Clock::time_point now) {
    refill(now);
    if (first) {
        recent_[recent_next_] = Offer{.value = value, .bytes = bytes};
        recent_next_          = (recent_next_ + 1U) % kRecent;
        recent_count_         = std::min(recent_count_ + 1U, kRecent);
    }
    if (static_cast<double>(bytes) > level_) { return false; }
    const double fill = level_ / static_cast<double>(bytes_per_hour_);
    if (fill >= 0.5 || recent_count_ == 0) { return true; }
    return value >= threshold(2.0 * fill);
}

void SsdWriteBudget::charge(std::uint64_t bytes) noexcept { level_ = std::max(0.0, level_ - static_cast<double>(bytes)); }

std::uint64_t SsdWriteBudget::available_bytes() const noexcept { return static_cast<std::uint64_t>(level_); }

void SsdWriteBudget::save(Clock::time_point now) const noexcept {
    try {
        // The level as of `now` (refill is a pure function of time, so saving it is the same as refilling).
        double level         = level_;
        const double seconds = seconds_between(last_, now);
        if (seconds > 0.0) {
            level = std::min(static_cast<double>(bytes_per_hour_), level + seconds * static_cast<double>(bytes_per_hour_) / kHourSeconds);
        }
        const auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(std::max(now, last_).time_since_epoch()).count();
        std::filesystem::path temporary = state_file_;
        temporary += ".tmp";
        {
            std::ofstream out(temporary, std::ios::trunc);
            out << kStateTag << ' ' << static_cast<long long>(level) << ' ' << static_cast<long long>(millis) << '\n';
            if (!out) { return; }
        }
        std::error_code ignored;
        std::filesystem::rename(temporary, state_file_, ignored);
    } catch (...) {
    }
}

} // namespace infernix::runtime::prefix_cache
