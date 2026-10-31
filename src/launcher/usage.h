#pragma once

// Usage accounting: tokens and requests per local day, configuration and client, from the server's
// own per-request rows (GET /metrics?format=json, matched to the launcher's requests by their
// X-Infernix-Request-Label). Kept in infernix-launcher-usage.json beside the memory file, written
// at most every 30 seconds and when the launcher stops.

#include "launcher/util.h"

#include <cstdint>
#include <filesystem>
#include <map>
#include <mutex>
#include <string>
#include <tuple>

namespace infernix::launcher {

struct UsageSample {
    std::int64_t t = 0; // when it ended (Unix ms)
    std::string config_id;
    std::string model; // the configuration's model id then
    std::string client;
    bool failed = false;
    std::uint64_t prompt_tokens  = 0;
    std::uint64_t cached_tokens  = 0;
    std::uint64_t prefill_tokens = 0;
    std::uint64_t output_tokens  = 0;
    double prefill_s = 0.0;
    double decode_s  = 0.0;
};

// "2026-10-31" (local time).
[[nodiscard]] std::string local_day(std::int64_t ms);

class UsageBook {
public:
    static constexpr const char* kFile = "infernix-launcher-usage.json";

    explicit UsageBook(std::filesystem::path dir);
    ~UsageBook();

    void add(const UsageSample& sample);
    // Rows of days in [from, to] (inclusive "YYYY-MM-DD").
    [[nodiscard]] Json rows(const std::string& from, const std::string& to) const;
    // Writes the file now when anything changed.
    void flush();

private:
    struct Totals {
        std::string model;
        std::uint64_t requests = 0, failed = 0, prompt_tokens = 0, cached_tokens = 0, prefill_tokens = 0,
                      output_tokens = 0;
        double prefill_s = 0.0, decode_s = 0.0;
    };
    using Key = std::tuple<std::string, std::string, std::string>; // day, configuration, client

    void load();
    void save_locked();

    std::filesystem::path file_;
    mutable std::mutex mutex_;
    std::map<Key, Totals> rows_;
    bool dirty_                = false;
    std::int64_t last_save_ms_ = 0;
};

} // namespace infernix::launcher
