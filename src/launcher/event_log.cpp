#include "launcher/event_log.h"

#include <algorithm>
#include <cstdio>
#include <ctime>

namespace infernix::launcher {
namespace {

std::string upper(std::string text) {
    for (char& c : text) {
        if (c >= 'a' && c <= 'z') { c = static_cast<char>(c - 'a' + 'A'); }
    }
    return text;
}

// FILE -> FILE.1 -> ... -> FILE.keep; the oldest is deleted.
void rotate(const std::filesystem::path& path, int keep) {
    std::error_code ignored;
    const auto numbered = [&](int index) {
        std::filesystem::path name = path;
        name += "." + std::to_string(index);
        return name;
    };
    std::filesystem::remove(numbered(keep), ignored);
    for (int index = keep - 1; index >= 1; --index) {
        if (std::filesystem::exists(numbered(index), ignored)) {
            std::filesystem::rename(numbered(index), numbered(index + 1), ignored);
        }
    }
    std::filesystem::rename(path, numbered(1), ignored);
}

} // namespace

Json to_json(const EventRecord& record) {
    return {{"seq", record.seq},           {"t", record.t},           {"level", record.level},
            {"category", record.category}, {"message", record.message}, {"fields", record.fields}};
}

std::string format_event_line(const EventRecord& record) {
    const std::time_t seconds = static_cast<std::time_t>(record.t / 1000);
    std::tm local{};
#ifdef _WIN32
    localtime_s(&local, &seconds);
#else
    localtime_r(&seconds, &local);
#endif
    char stamp[32];
    std::strftime(stamp, sizeof(stamp), "%Y-%m-%d %H:%M:%S", &local);
    char head[96];
    std::snprintf(head, sizeof(head), "%s.%03d  %-5s  %-8s  ", stamp, static_cast<int>(record.t % 1000),
                  upper(record.level).c_str(), record.category.c_str());
    std::string line = head + record.message;
    // One record, one line.
    std::replace(line.begin(), line.end(), '\n', ' ');
    std::replace(line.begin(), line.end(), '\r', ' ');
    return line;
}

EventLog::EventLog(std::filesystem::path dir) {
    std::string ignored;
    set_dir(dir, ignored);
}

bool EventLog::set_dir(const std::filesystem::path& dir, std::string& error) {
    std::lock_guard lock(mutex_);
    for (File* file : {&events_, &requests_}) {
        if (file->stream.is_open()) { file->stream.close(); }
        file->bytes  = 0;
        file->failed = false;
    }
    dir_ = dir;
    std::error_code code;
    std::filesystem::create_directories(dir_, code);
    if (code || !std::filesystem::is_directory(dir_, code)) {
        error          = "cannot create " + path_utf8(dir_) + (code ? ": " + code.message() : std::string());
        events_.failed = requests_.failed = true;
        return false;
    }
    // Open the events file now, so an unwritable folder shows at once.
    events_.stream.open(dir_ / kEventsFile, std::ios::binary | std::ios::app);
    if (!events_.stream) {
        error          = "cannot write " + path_utf8(dir_ / kEventsFile);
        events_.failed = true;
        return false;
    }
    events_.bytes = static_cast<std::uint64_t>(std::filesystem::file_size(dir_ / kEventsFile, code));
    return true;
}

std::filesystem::path EventLog::dir() const {
    std::lock_guard lock(mutex_);
    return dir_;
}

std::filesystem::path EventLog::events_file() const {
    std::lock_guard lock(mutex_);
    return dir_ / kEventsFile;
}

std::filesystem::path EventLog::requests_file() const {
    std::lock_guard lock(mutex_);
    return dir_ / kRequestsFile;
}

void EventLog::write_locked(File& file, const std::filesystem::path& path, std::uint64_t limit,
                            const std::string& line) {
    if (file.failed) { return; }
    if (!file.stream.is_open()) {
        file.stream.open(path, std::ios::binary | std::ios::app);
        std::error_code code;
        file.bytes = static_cast<std::uint64_t>(std::filesystem::file_size(path, code));
        if (!file.stream) {
            file.failed = true;
            return;
        }
    }
    if (file.bytes > 0 && file.bytes + line.size() + 1 > limit) {
        file.stream.close();
        rotate(path, kKeep);
        file.stream.open(path, std::ios::binary | std::ios::trunc);
        file.bytes = 0;
        if (!file.stream) {
            file.failed = true;
            return;
        }
    }
    file.stream << line << '\n';
    file.stream.flush();
    file.bytes += line.size() + 1;
}

void EventLog::add(std::string level, std::string category, std::string message, Json fields) {
    EventRecord record;
    Listener listener;
    {
        std::lock_guard lock(mutex_);
        record.seq      = next_seq_++;
        record.t        = now_ms();
        record.level    = std::move(level);
        record.category = std::move(category);
        record.message  = std::move(message);
        record.fields   = std::move(fields);
        write_locked(events_, dir_ / kEventsFile, kEventsLimit, format_event_line(record));
        records_.push_back(record);
        while (records_.size() > kMemory) { records_.pop_front(); }
        listener = listener_;
    }
    if (listener) { listener(record); }
}

void EventLog::add_request_body(const Json& record) {
    const std::string line = record.dump(-1, ' ', false, Json::error_handler_t::replace);
    std::lock_guard lock(mutex_);
    write_locked(requests_, dir_ / kRequestsFile, kRequestsLimit, line);
}

Json EventLog::page(std::uint64_t after, std::size_t limit) const {
    std::lock_guard lock(mutex_);
    Json lines = Json::array();
    auto it    = std::upper_bound(records_.begin(), records_.end(), after,
                                  [](std::uint64_t seq, const EventRecord& r) { return seq < r.seq; });
    if (after == 0 && static_cast<std::size_t>(records_.end() - it) > limit) {
        it = records_.end() - static_cast<std::ptrdiff_t>(limit);
    }
    for (; it != records_.end() && lines.size() < limit; ++it) { lines.push_back(to_json(*it)); }
    return {{"lines", lines},
            {"first_seq", records_.empty() ? 0 : records_.front().seq},
            {"last_seq", records_.empty() ? 0 : records_.back().seq}};
}

void EventLog::set_listener(Listener listener) {
    std::lock_guard lock(mutex_);
    listener_ = std::move(listener);
}

} // namespace infernix::launcher
