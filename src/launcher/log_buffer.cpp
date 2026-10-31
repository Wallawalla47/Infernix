#include "launcher/log_buffer.h"

#include <algorithm>
#include <cctype>
#include <ctime>

namespace infernix::launcher {
namespace {

bool digits(std::string_view text) {
    return !text.empty() && std::all_of(text.begin(), text.end(), [](char c) { return c >= '0' && c <= '9'; });
}

int number(std::string_view text) {
    int value = 0;
    for (const char c : text) { value = value * 10 + (c - '0'); }
    return value;
}

std::string level_name(std::string_view word) {
    const std::string lowered = lower(word);
    if (lowered == "warning") { return "warn"; }
    if (lowered == "crit" || lowered == "critical" || lowered == "fatal") { return "critical"; }
    if (lowered == "err") { return "error"; }
    return lowered;
}

std::string file_stamp() {
    const std::time_t now = std::time(nullptr);
    std::tm local{};
#ifdef _WIN32
    localtime_s(&local, &now);
#else
    localtime_r(&now, &local);
#endif
    char buffer[32];
    std::strftime(buffer, sizeof(buffer), "%Y%m%d-%H%M%S", &local);
    return buffer;
}

} // namespace

Json to_json(const LogRecord& record) {
    return {{"seq", record.seq},       {"instance", record.instance}, {"t", record.t},
            {"level", record.level},   {"source", record.source},     {"text", record.text}};
}

std::optional<ParsedRecord> parse_serve_record(std::string_view line) {
    // 2026-10-30 11:55:04.605  INFO  message
    if (line.size() < 25 || line[4] != '-' || line[7] != '-' || line[10] != ' ' || line[13] != ':' ||
        line[16] != ':' || line[19] != '.') {
        return std::nullopt;
    }
    if (!digits(line.substr(0, 4)) || !digits(line.substr(5, 2)) || !digits(line.substr(8, 2)) ||
        !digits(line.substr(11, 2)) || !digits(line.substr(14, 2)) || !digits(line.substr(17, 2)) ||
        !digits(line.substr(20, 3))) {
        return std::nullopt;
    }
    std::size_t at = 23;
    while (at < line.size() && line[at] == ' ') { ++at; }
    std::size_t end = at;
    while (end < line.size() && std::isalpha(static_cast<unsigned char>(line[end]))) { ++end; }
    if (end == at) { return std::nullopt; }
    const std::string level = level_name(line.substr(at, end - at));
    static constexpr std::string_view kLevels[] = {"trace", "debug", "info", "warn", "error", "critical"};
    if (std::find(std::begin(kLevels), std::end(kLevels), level) == std::end(kLevels)) { return std::nullopt; }
    while (end < line.size() && line[end] == ' ') { ++end; }

    std::tm local{};
    local.tm_year  = number(line.substr(0, 4)) - 1900;
    local.tm_mon   = number(line.substr(5, 2)) - 1;
    local.tm_mday  = number(line.substr(8, 2));
    local.tm_hour  = number(line.substr(11, 2));
    local.tm_min   = number(line.substr(14, 2));
    local.tm_sec   = number(line.substr(17, 2));
    local.tm_isdst = -1;
    const std::time_t seconds = std::mktime(&local);
    ParsedRecord record;
    record.t       = seconds == -1 ? now_ms() : static_cast<std::int64_t>(seconds) * 1000 + number(line.substr(20, 3));
    record.level   = level;
    record.message = std::string(line.substr(end));
    return record;
}

LogBuffer::LogBuffer(std::filesystem::path log_dir) : log_dir_(std::move(log_dir)) {}

void LogBuffer::set_dir(std::filesystem::path log_dir) {
    std::lock_guard lock(mutex_);
    log_dir_ = std::move(log_dir);
}

void LogBuffer::begin_instance(std::uint32_t instance, const std::string& label) {
    std::lock_guard lock(mutex_);
    std::error_code created;
    std::filesystem::create_directories(log_dir_, created);
    partial_.clear();
    partial_instance_ = instance;
    if (file_.is_open()) { file_.close(); }
    std::string safe;
    for (const char c : label) {
        safe.push_back(std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_' || c == '.' ? c : '_');
    }
    if (safe.size() > 80) { safe.resize(80); }
    file_.open(log_dir_ / ("serve-" + file_stamp() + "-" + safe + ".log"), std::ios::binary | std::ios::trunc);
    // Keep the newest 20 server logs.
    std::vector<std::filesystem::directory_entry> logs;
    std::error_code error;
    for (const auto& entry : std::filesystem::directory_iterator(log_dir_, error)) {
        if (entry.path().filename().string().starts_with("serve-")) { logs.push_back(entry); }
    }
    std::sort(logs.begin(), logs.end(), [](const auto& a, const auto& b) {
        return a.path().filename() > b.path().filename();
    });
    for (std::size_t i = 20; i < logs.size(); ++i) { std::filesystem::remove(logs[i].path(), error); }
}

void LogBuffer::end_instance() {
    std::vector<LogRecord> emitted;
    Listener listener;
    {
        std::lock_guard lock(mutex_);
        if (!partial_.empty()) {
            LogRecord record;
            record.instance = partial_instance_;
            record.t        = now_ms();
            record.level    = "raw";
            record.source   = "server";
            record.text     = strip_ansi(partial_);
            partial_.clear();
            push_locked(std::move(record), emitted);
        }
        if (file_.is_open()) { file_.close(); }
        listener = listener_;
    }
    if (listener && !emitted.empty()) { listener(emitted); }
}

void LogBuffer::append_output(std::uint32_t instance, std::string_view chunk) {
    std::vector<LogRecord> emitted;
    Listener listener;
    {
        std::lock_guard lock(mutex_);
        if (file_.is_open()) {
            file_.write(chunk.data(), static_cast<std::streamsize>(chunk.size()));
            file_.flush();
        }
        partial_instance_ = instance;
        partial_.append(chunk);
        std::size_t start = 0;
        for (;;) {
            const std::size_t newline = partial_.find('\n', start);
            if (newline == std::string::npos) { break; }
            const std::string line = strip_ansi(std::string_view(partial_).substr(start, newline - start));
            start                  = newline + 1;
            if (trim(line).empty()) { continue; }
            LogRecord record;
            record.instance = instance;
            record.source   = "server";
            if (const std::optional<ParsedRecord> parsed = parse_serve_record(line)) {
                record.t     = parsed->t;
                record.level = parsed->level;
                record.text  = parsed->message;
            } else {
                record.t     = now_ms();
                record.level = "raw";
                record.text  = line;
                // A CUDA abort or another crash report has no record prefix; show it as an error.
                if (line.find("CUDA_CHECK") != std::string::npos || line.find("failed:") != std::string::npos ||
                    line.find("Assertion") != std::string::npos) {
                    record.level = "error";
                }
            }
            push_locked(std::move(record), emitted);
        }
        partial_.erase(0, start);
        if (partial_.size() > (1 << 20)) { partial_.clear(); }
        listener = listener_;
    }
    if (listener && !emitted.empty()) { listener(emitted); }
}

void LogBuffer::add(std::uint32_t instance, std::string level, std::string source, std::string text) {
    std::vector<LogRecord> emitted;
    Listener listener;
    {
        std::lock_guard lock(mutex_);
        LogRecord record;
        record.instance = instance;
        record.t        = now_ms();
        record.level    = std::move(level);
        record.source   = std::move(source);
        record.text     = std::move(text);
        push_locked(std::move(record), emitted);
        listener = listener_;
    }
    if (listener) { listener(emitted); }
}

void LogBuffer::push_locked(LogRecord record, std::vector<LogRecord>& emitted) {
    record.seq = next_seq_++;
    records_.push_back(record);
    while (records_.size() > kCapacity) { records_.pop_front(); }
    emitted.push_back(std::move(record));
}

Json LogBuffer::page(std::uint64_t after, std::size_t limit) const {
    std::lock_guard lock(mutex_);
    Json lines = Json::array();
    auto it    = std::upper_bound(records_.begin(), records_.end(), after,
                                  [](std::uint64_t seq, const LogRecord& r) { return seq < r.seq; });
    // Without `after`, the newest `limit` records.
    if (after == 0 && static_cast<std::size_t>(records_.end() - it) > limit) { it = records_.end() - static_cast<std::ptrdiff_t>(limit); }
    for (; it != records_.end() && lines.size() < limit; ++it) { lines.push_back(to_json(*it)); }
    return {{"lines", lines},
            {"first_seq", records_.empty() ? 0 : records_.front().seq},
            {"last_seq", records_.empty() ? 0 : records_.back().seq}};
}

std::vector<LogRecord> LogBuffer::recent(std::uint32_t instance, std::size_t count, bool warnings_only) const {
    std::lock_guard lock(mutex_);
    std::vector<LogRecord> out;
    for (auto it = records_.rbegin(); it != records_.rend() && out.size() < count; ++it) {
        if (it->instance != instance || it->source != "server") { continue; }
        if (warnings_only && it->level != "warn" && it->level != "error" && it->level != "critical") { continue; }
        out.push_back(*it);
    }
    std::reverse(out.begin(), out.end());
    return out;
}

void LogBuffer::set_listener(Listener listener) {
    std::lock_guard lock(mutex_);
    listener_ = std::move(listener);
}

} // namespace infernix::launcher
