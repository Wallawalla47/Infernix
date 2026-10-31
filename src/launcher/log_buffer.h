#pragma once

// The console log shown on the dashboard: infernix-serve's output split into records (its
// `YYYY-MM-DD HH:MM:SS.mmm  LEVEL  message` lines parsed, other lines kept as raw), interleaved with
// the launcher's own records, in a bounded ring with increasing sequence numbers. Each server
// instance's output is also written to `serve-<time>-<model>.log` in the server-log folder.

#include "launcher/util.h"

#include <cstdint>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace infernix::launcher {

struct LogRecord {
    std::uint64_t seq      = 0;
    std::uint32_t instance = 0;
    std::int64_t t         = 0;
    std::string level; // trace debug info warn error critical raw
    std::string source; // server | launcher
    std::string text;
};

[[nodiscard]] Json to_json(const LogRecord& record);

// One serve record parsed (exposed for tests): level and message, with the timestamp as Unix ms
// (local time); nullopt when the line has no record prefix.
struct ParsedRecord {
    std::int64_t t = 0;
    std::string level;
    std::string message;
};
[[nodiscard]] std::optional<ParsedRecord> parse_serve_record(std::string_view line);

class LogBuffer {
public:
    static constexpr std::size_t kCapacity = 20000;
    using Listener = std::function<void(const std::vector<LogRecord>& records)>;

    explicit LogBuffer(std::filesystem::path log_dir);

    // The folder of the next instances' log files.
    void set_dir(std::filesystem::path log_dir);

    // A new server instance: opens its log file and keeps the newest 20 files.
    void begin_instance(std::uint32_t instance, const std::string& label);
    void end_instance();
    // Raw output from the server; complete lines become records.
    void append_output(std::uint32_t instance, std::string_view chunk);
    void add(std::uint32_t instance, std::string level, std::string source, std::string text);

    [[nodiscard]] Json page(std::uint64_t after, std::size_t limit) const;
    // The newest records of `instance` at `level` or worse, newest last.
    [[nodiscard]] std::vector<LogRecord> recent(std::uint32_t instance, std::size_t count,
                                                bool warnings_only) const;
    void set_listener(Listener listener);

private:
    void push_locked(LogRecord record, std::vector<LogRecord>& emitted);

    std::filesystem::path log_dir_;
    mutable std::mutex mutex_;
    std::deque<LogRecord> records_;
    std::uint64_t next_seq_ = 1;
    std::string partial_;
    std::uint32_t partial_instance_ = 0;
    std::ofstream file_;
    Listener listener_;
};

} // namespace infernix::launcher
