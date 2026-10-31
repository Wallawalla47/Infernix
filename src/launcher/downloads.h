#pragma once

// Hugging Face downloads into a folder the user chose: one job per repository, run one at a time
// in order. Each file arrives in fixed-size chunks over several connections into `<file>.part`,
// with the finished chunks recorded beside it, so a pause, a restart of the launcher or a dropped
// connection resumes where it stopped; a finished file is checked against its LFS SHA-256 and
// renamed into place. Jobs persist in the launcher's memory file (through Hooks).

#include "launcher/hf_hub.h"
#include "launcher/util.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace infernix::launcher {

struct DownloadFile {
    std::string path;
    std::uint64_t bytes = 0;
    std::optional<std::string> sha256;
    std::atomic<std::uint64_t> done{0};
    std::string state = "queued"; // queued downloading verifying done failed paused skipped
    std::optional<std::string> error;
};

struct DownloadJob {
    std::string id;
    std::string repo;
    std::string revision;
    std::filesystem::path dest_dir;   // the folder the user chose
    std::filesystem::path target_dir; // dest_dir, or dest_dir/<repo name>
    bool replace = false;             // an update: files already there are fetched again
    std::vector<std::unique_ptr<DownloadFile>> files;
    std::string state = "queued"; // queued running paused verifying done failed cancelled
    std::optional<std::string> error;
    std::int64_t created_at = 0;
    std::optional<std::int64_t> finished_at;
    double speed_bps = 0.0;
};

class Downloads {
public:
    struct Hooks {
        std::function<std::string()> token;
        std::function<int()> connections;
        std::function<bool()> verify;
        // A job finished: its folder should be scanned (and added to the model folders).
        std::function<void(const std::filesystem::path& dest_dir, const std::filesystem::path& target_dir,
                           const std::string& repo)>
            finished;
        // A job failed (the reason).
        std::function<void(const std::string& repo, const std::string& error)> failed;
        // The job list after any change (and once a second while one runs).
        std::function<void(const Json& jobs)> changed;
        // The persisted queue: read once at construction, written after every change.
        std::function<Json()> load_state;
        std::function<void(const Json& jobs)> save_state;
    };

    explicit Downloads(Hooks hooks);
    ~Downloads();

    [[nodiscard]] Json list() const;
    // Queues a job from a DownloadRequest; throws std::invalid_argument / std::runtime_error.
    Json create(const Json& request);
    void pause(const std::string& id);
    void resume(const std::string& id);
    void cancel(const std::string& id);
    void remove(const std::string& id);
    [[nodiscard]] bool active() const;
    void shutdown();

private:
    void run();
    void run_job(DownloadJob& job);
    bool download_file(DownloadJob& job, DownloadFile& file, const std::filesystem::path& final_path);
    void save_locked() const;
    void load();
    [[nodiscard]] Json job_json_locked(const DownloadJob& job) const;
    [[nodiscard]] DownloadJob* find_locked(const std::string& id);
    void notify();

    Hooks hooks_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<std::unique_ptr<DownloadJob>> jobs_;
    std::atomic<bool> interrupt_{false}; // the running job was paused, cancelled or the launcher stops
    std::string running_id_;
    bool stopping_ = false;
    std::thread worker_;
    std::thread reporter_;
};

} // namespace infernix::launcher
