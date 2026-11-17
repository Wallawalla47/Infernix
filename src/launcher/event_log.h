#pragma once

// The launcher's own event log: model loads and stops, the watchdog's incidents, downloads,
// settings changes and one line per API request (its metadata: client, route, model, status,
// timings, sizes). Records go to `infernix-launcher.log` in the log folder as text lines in the
// server's format (`YYYY-MM-DD HH:MM:SS.mmm  LEVEL  category  message`), to a bounded in-memory ring
// for the web UI, and to a listener. When asked, full request and response bodies go to
// `infernix-launcher-requests.jsonl` beside it, one JSON object per line. Both files rotate by size
// (`FILE.1` .. `FILE.5`); a folder that cannot be written keeps the records in memory only.

#include "launcher/util.h"

#include <cstdint>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace infernix::launcher {

struct EventRecord {
    std::uint64_t seq = 0;
    std::int64_t t    = 0;
    std::string level;    // debug info warn error
    std::string category; // launcher model watchdog api download settings
    std::string message;
    Json fields; // structured details (an API request's metadata); null when none
};

[[nodiscard]] Json to_json(const EventRecord& record);
// "2026-10-31 07:14:56.431  ERROR  watchdog  message" (local time), without a line end.
[[nodiscard]] std::string format_event_line(const EventRecord& record);

class EventLog {
public:
    static constexpr const char* kEventsFile   = "infernix-launcher.log";
    static constexpr const char* kRequestsFile = "infernix-launcher-requests.jsonl";
    static constexpr std::size_t kMemory       = 5000;
    static constexpr std::uint64_t kEventsLimit   = 16ull << 20;
    static constexpr std::uint64_t kRequestsLimit = 100ull << 20;
    static constexpr int kKeep                    = 5;
    using Listener = std::function<void(const EventRecord& record)>;

    explicit EventLog(std::filesystem::path dir);
    ~EventLog(); // writes what is still queued

    // Moves both files to `dir` (later records only; the queued ones are written where they were
    // logged first); false, with the reason, when it cannot be written (the records then stay in memory).
    bool set_dir(const std::filesystem::path& dir, std::string& error);
    [[nodiscard]] std::filesystem::path dir() const;
    [[nodiscard]] std::filesystem::path events_file() const;
    [[nodiscard]] std::filesystem::path requests_file() const;

    // add and add_request_body keep the record and queue its line; write_files (one writer, the launcher's log
    // thread) writes the queued lines, so a slow or stalled log folder never holds up the thread that logs.
    void add(std::string level, std::string category, std::string message, Json fields = nullptr);
    // One request-content record (written to the requests file only).
    void add_request_body(const Json& record);
    void write_files();

    // Without `after`, the newest `limit` records.
    [[nodiscard]] Json page(std::uint64_t after, std::size_t limit) const;
    void set_listener(Listener listener);

private:
    struct File {
        std::ofstream stream;
        std::uint64_t bytes     = 0;
        std::uint64_t rotate_at = 0; // after a failed rotation, the size of the next try
        bool failed             = false;
    };
    void write_locked(File& file, const std::filesystem::path& path, std::uint64_t limit, const std::string& line);
    void write_pending_locked(); // file_mutex_ held

    // Lock order: file_mutex_, then mutex_.
    std::mutex file_mutex_;  // the files and their state, the folder they are written in
    std::filesystem::path file_dir_;
    File events_;
    File requests_;
    mutable std::mutex mutex_; // the records, the queues and the folder as reported
    std::filesystem::path dir_;
    std::vector<std::string> pending_events_;
    std::vector<std::string> pending_requests_;
    std::deque<EventRecord> records_;
    std::uint64_t next_seq_ = 1;
    Listener listener_;
};

} // namespace infernix::launcher
