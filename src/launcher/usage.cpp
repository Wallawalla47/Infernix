#include "launcher/usage.h"

#include <cstdio>
#include <ctime>

namespace infernix::launcher {

std::string local_day(std::int64_t ms) {
    const std::time_t seconds = static_cast<std::time_t>(ms / 1000);
    std::tm local{};
#ifdef _WIN32
    localtime_s(&local, &seconds);
#else
    localtime_r(&seconds, &local);
#endif
    char day[16];
    std::strftime(day, sizeof(day), "%Y-%m-%d", &local);
    return day;
}

UsageBook::UsageBook(std::filesystem::path dir) : file_(std::move(dir) / kFile) { load(); }

UsageBook::~UsageBook() { flush(); }

void UsageBook::load() {
    const std::optional<std::string> text = read_file(file_);
    if (!text) { return; }
    Json data;
    try {
        data = Json::parse(*text);
    } catch (const Json::exception&) { return; }
    for (const Json& row : data.value("rows", Json::array())) {
        Totals totals;
        totals.model          = json_or<std::string>(row, "model", "");
        totals.requests       = json_or<std::uint64_t>(row, "requests", 0);
        totals.failed         = json_or<std::uint64_t>(row, "failed", 0);
        totals.prompt_tokens  = json_or<std::uint64_t>(row, "prompt_tokens", 0);
        totals.cached_tokens  = json_or<std::uint64_t>(row, "cached_tokens", 0);
        totals.prefill_tokens = json_or<std::uint64_t>(row, "prefill_tokens", 0);
        totals.output_tokens  = json_or<std::uint64_t>(row, "output_tokens", 0);
        totals.prefill_s      = json_or<double>(row, "prefill_s", 0.0);
        totals.decode_s       = json_or<double>(row, "decode_s", 0.0);
        rows_[{json_or<std::string>(row, "day", ""), json_or<std::string>(row, "config_id", ""),
               json_or<std::string>(row, "client", "")}] = totals;
    }
}

void UsageBook::add(const UsageSample& sample) {
    std::lock_guard lock(mutex_);
    Totals& totals = rows_[{local_day(sample.t), sample.config_id, sample.client}];
    totals.model   = sample.model;
    ++totals.requests;
    totals.failed += sample.failed ? 1 : 0;
    totals.prompt_tokens += sample.prompt_tokens;
    totals.cached_tokens += sample.cached_tokens;
    totals.prefill_tokens += sample.prefill_tokens;
    totals.output_tokens += sample.output_tokens;
    totals.prefill_s += sample.prefill_s;
    totals.decode_s += sample.decode_s;
    dirty_ = true;
    if (now_ms() - last_save_ms_ >= 30000) { save_locked(); }
}

Json UsageBook::rows(const std::string& from, const std::string& to) const {
    std::lock_guard lock(mutex_);
    Json out = Json::array();
    for (const auto& [key, totals] : rows_) {
        const auto& [day, config_id, client] = key;
        if (day < from || day > to) { continue; }
        out.push_back({{"day", day},
                       {"config_id", config_id},
                       {"model", totals.model},
                       {"client", client},
                       {"requests", totals.requests},
                       {"failed", totals.failed},
                       {"prompt_tokens", totals.prompt_tokens},
                       {"cached_tokens", totals.cached_tokens},
                       {"prefill_tokens", totals.prefill_tokens},
                       {"output_tokens", totals.output_tokens},
                       {"prefill_s", totals.prefill_s},
                       {"decode_s", totals.decode_s}});
    }
    return out;
}

void UsageBook::flush() {
    std::lock_guard lock(mutex_);
    if (dirty_) { save_locked(); }
}

void UsageBook::save_locked() {
    last_save_ms_ = now_ms();
    dirty_        = false;
    Json rows = Json::array();
    for (const auto& [key, totals] : rows_) {
        const auto& [day, config_id, client] = key;
        rows.push_back({{"day", day},
                        {"config_id", config_id},
                        {"model", totals.model},
                        {"client", client},
                        {"requests", totals.requests},
                        {"failed", totals.failed},
                        {"prompt_tokens", totals.prompt_tokens},
                        {"cached_tokens", totals.cached_tokens},
                        {"prefill_tokens", totals.prefill_tokens},
                        {"output_tokens", totals.output_tokens},
                        {"prefill_s", totals.prefill_s},
                        {"decode_s", totals.decode_s}});
    }
    try {
        write_file_atomic(file_, Json{{"schema", 1}, {"rows", rows}}.dump(1) + "\n");
    } catch (const std::exception& error) { std::fprintf(stderr, "launcher: %s\n", error.what()); }
}

} // namespace infernix::launcher
