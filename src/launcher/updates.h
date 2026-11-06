#pragma once

// Updates. Models: the Hugging Face repositories downloaded through the launcher are checked for
// a newer revision whose files changed (on demand and every 6 hours); an update downloads the
// changed files into the same folder. Infernix: this build against the latest GitHub release of
// Wallawalla47/Infernix; an update downloads the release's Windows zip (checked against GitHub's
// SHA-256 digest), and installing it hands over to a small script that waits for the launcher to
// exit, copies the new program files over the old ones (the memory file, logs and chats stay) and
// starts the launcher again with the same arguments.

#include "launcher/util.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace infernix::launcher {

// "2026.10.18" from a build id or tag ("windows-2026.10.18-174-g52bcb024-dirty",
// "windows-2026.10.09.2"); nullopt when there is none.
[[nodiscard]] std::optional<std::string> release_version(std::string_view build_or_tag);
// Release versions compared part by part (2026.10.09.2 > 2026.10.09); <0, 0, >0.
[[nodiscard]] int compare_versions(std::string_view a, std::string_view b);
// Whether a release asset is Infernix's Windows zip ("infernix-<version>-windows-x64.zip") with a
// name that is safe as a file name: letters, digits, '.', '_' and '-' only.
[[nodiscard]] bool is_windows_release_asset(std::string_view name);
// GitHub's latest-release answer as the update state shows it: {tag, version, published_at, notes,
// url, asset} with asset {name, bytes, sha256, url} or null when there is no Windows zip. A field
// that is absent or null (a release without notes, an asset GitHub did not hash) reads as empty;
// nullopt when `body` is not a JSON object.
[[nodiscard]] std::optional<Json> parse_release(const std::string& body);

class ModelUpdates {
public:
    struct Hooks {
        std::function<Json()> jobs;            // the download jobs (Downloads::list())
        std::function<std::string()> token;    // Hugging Face token
        std::function<void(const Json&)> changed;
    };

    explicit ModelUpdates(Hooks hooks);
    ~ModelUpdates();

    [[nodiscard]] Json state() const;
    // Starts a check now (returns at once; `changed` reports the result).
    void check();
    // The update for a download job, if any: {repo, target_dir, files[path]}.
    [[nodiscard]] std::optional<Json> update_for(const std::string& job_id) const;
    void shutdown();

private:
    void run();
    void check_now();

    Hooks hooks_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    bool check_requested_ = false;
    bool checking_        = false;
    bool stopping_        = false;
    std::optional<std::int64_t> checked_at_;
    std::optional<std::string> error_;
    Json updates_ = Json::array();
    std::thread worker_;
};

class InfernixUpdate {
public:
    struct Options {
        std::string build_id;                  // this launcher's build
        std::filesystem::path install_dir;     // the launcher's folder
        std::filesystem::path work_dir;        // where the zip and the unpacked release go
        std::vector<std::string> restart_args; // the launcher's command-line arguments
        std::function<void(const Json&)> changed;
    };

    explicit InfernixUpdate(Options options);
    ~InfernixUpdate();

    [[nodiscard]] Json state() const;
    void check();
    // Downloads the latest release's zip (throws std::runtime_error when there is nothing to get).
    void download();
    // Unpacks the downloaded zip and starts the hand-over script; the caller then exits.
    void install();
    void shutdown();

private:
    void run();
    void check_now();
    void download_now();
    [[nodiscard]] Json state_locked() const;
    void publish();

    Options options_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    bool stopping_        = false;
    bool check_requested_ = true;
    bool download_requested_ = false;
    std::string phase_    = "idle"; // idle checking downloading ready installing failed
    std::optional<std::int64_t> checked_at_;
    std::optional<std::string> error_;
    Json latest_ = nullptr;
    std::filesystem::path zip_;
    std::uint64_t done_  = 0;
    std::uint64_t total_ = 0;
    std::atomic<bool> cancel_{false};
    std::thread worker_;
};

} // namespace infernix::launcher
