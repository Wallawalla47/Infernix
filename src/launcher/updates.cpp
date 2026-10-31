#include "launcher/updates.h"

#include "launcher/args.h"
#include "launcher/hf_hub.h"
#include "launcher/process.h"
#include "launcher/sha256.h"

#include <curl/curl.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <map>
#include <memory>
#include <regex>
#include <stdexcept>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace infernix::launcher {
namespace {

constexpr const char* kReleaseApi = "https://api.github.com/repos/Wallawalla47/Infernix/releases/latest";
constexpr auto kModelCheckEvery   = std::chrono::hours(6);
constexpr auto kReleaseCheckEvery = std::chrono::hours(6);

std::vector<std::uint64_t> version_parts(std::string_view version) {
    std::vector<std::uint64_t> parts;
    std::size_t start = 0;
    while (start <= version.size()) {
        const std::size_t dot = version.find('.', start);
        const std::string part(version.substr(start, dot == std::string_view::npos ? std::string_view::npos : dot - start));
        try {
            parts.push_back(std::stoull(part));
        } catch (const std::exception&) { parts.push_back(0); }
        if (dot == std::string_view::npos) { break; }
        start = dot + 1;
    }
    return parts;
}

struct DownloadTarget {
    std::ofstream* out = nullptr;
    std::uint64_t* done = nullptr;
    std::mutex* mutex = nullptr;
    const std::atomic<bool>* cancel = nullptr;
};

size_t write_file(char* data, size_t size, size_t count, void* user) {
    auto* target = static_cast<DownloadTarget*>(user);
    const size_t bytes = size * count;
    target->out->write(data, static_cast<std::streamsize>(bytes));
    if (!*target->out) { return 0; }
    std::lock_guard lock(*target->mutex);
    *target->done += bytes;
    return bytes;
}

int download_progress(void* user, curl_off_t, curl_off_t, curl_off_t, curl_off_t) {
    return static_cast<DownloadTarget*>(user)->cancel->load() ? 1 : 0;
}

bool writable(const std::filesystem::path& dir) {
    const std::filesystem::path probe = dir / ".infernix-update-write-test";
    {
        std::ofstream out(probe, std::ios::binary | std::ios::trunc);
        if (!out) { return false; }
    }
    std::error_code ignored;
    std::filesystem::remove(probe, ignored);
    return true;
}

// The folder of an unpacked release that holds infernix-launcher.
std::optional<std::filesystem::path> release_root(const std::filesystem::path& dir) {
    std::error_code error;
    for (auto it = std::filesystem::recursive_directory_iterator(dir, error);
         it != std::filesystem::recursive_directory_iterator(); it.increment(error)) {
        if (error) { break; }
        if (it.depth() > 3) {
            it.disable_recursion_pending();
            continue;
        }
        const std::string name = lower(path_utf8(it->path().filename()));
        if (name == "infernix-launcher.exe" || name == "infernix-launcher") { return it->path().parent_path(); }
    }
    return std::nullopt;
}

} // namespace

std::optional<std::string> release_version(std::string_view build_or_tag) {
    static const std::regex pattern(R"((\d{4}\.\d{2}\.\d{2}(?:\.\d+)?))");
    std::match_results<std::string_view::const_iterator> match;
    if (!std::regex_search(build_or_tag.begin(), build_or_tag.end(), match, pattern)) { return std::nullopt; }
    return match[1].str();
}

int compare_versions(std::string_view a, std::string_view b) {
    std::vector<std::uint64_t> left = version_parts(a), right = version_parts(b);
    const std::size_t size = std::max(left.size(), right.size());
    left.resize(size, 0);
    right.resize(size, 0);
    for (std::size_t i = 0; i < size; ++i) {
        if (left[i] != right[i]) { return left[i] < right[i] ? -1 : 1; }
    }
    return 0;
}

// --- Models ----------------------------------------------------------------------------------

ModelUpdates::ModelUpdates(Hooks hooks) : hooks_(std::move(hooks)) {
    worker_ = std::thread([this] { run(); });
}

ModelUpdates::~ModelUpdates() { shutdown(); }

void ModelUpdates::shutdown() {
    {
        std::lock_guard lock(mutex_);
        if (stopping_) { return; }
        stopping_ = true;
    }
    cv_.notify_all();
    if (worker_.joinable()) { worker_.join(); }
}

Json ModelUpdates::state() const {
    std::lock_guard lock(mutex_);
    return {{"checked_at", checked_at_ ? Json(*checked_at_) : Json(nullptr)},
            {"checking", checking_},
            {"updates", updates_},
            {"error", error_ ? Json(*error_) : Json(nullptr)}};
}

void ModelUpdates::check() {
    {
        std::lock_guard lock(mutex_);
        check_requested_ = true;
    }
    cv_.notify_all();
}

std::optional<Json> ModelUpdates::update_for(const std::string& job_id) const {
    std::lock_guard lock(mutex_);
    for (const Json& update : updates_) {
        if (update.value("job_id", "") == job_id) { return update; }
    }
    return std::nullopt;
}

void ModelUpdates::run() {
    // A first look two minutes after start, then every six hours (or when asked).
    auto next = std::chrono::steady_clock::now() + std::chrono::minutes(2);
    std::unique_lock lock(mutex_);
    while (!stopping_) {
        cv_.wait_until(lock, next, [this] { return stopping_ || check_requested_; });
        if (stopping_) { break; }
        check_requested_ = false;
        checking_        = true;
        lock.unlock();
        if (hooks_.changed) { hooks_.changed(state()); }
        check_now();
        lock.lock();
        checking_ = false;
        next      = std::chrono::steady_clock::now() + kModelCheckEvery;
        lock.unlock();
        if (hooks_.changed) { hooks_.changed(state()); }
        lock.lock();
    }
}

void ModelUpdates::check_now() {
    const Json jobs = hooks_.jobs ? hooks_.jobs() : Json::array();
    // The newest finished job per repository and folder.
    std::map<std::pair<std::string, std::string>, Json> newest;
    for (const Json& job : jobs) {
        if (job.value("state", "") != "done") { continue; }
        const auto place = std::make_pair(job.value("repo", ""), job.value("target_dir", ""));
        auto it          = newest.find(place);
        if (it == newest.end() || job.value("created_at", std::int64_t{0}) > it->second.value("created_at", std::int64_t{0})) {
            newest[place] = job;
        }
    }
    Json updates = Json::array();
    std::optional<std::string> error;
    const std::string token = hooks_.token ? hooks_.token() : std::string();
    for (const auto& [place, job] : newest) {
        try {
            const HfListing listing = hf_list_files(place.first, token);
            if (listing.revision == job.value("revision", "")) { continue; }
            Json files          = Json::array();
            std::uint64_t bytes = 0;
            for (const Json& file : job.value("files", Json::array())) {
                const std::string path = file.value("path", "");
                const auto it = std::find_if(listing.files.begin(), listing.files.end(),
                                             [&](const HfFile& remote) { return remote.path == path; });
                if (it == listing.files.end()) { continue; }
                const Json known_sha = file.value("sha256", Json(nullptr));
                const bool sha_changed =
                    it->sha256 && known_sha.is_string() && *it->sha256 != known_sha.get<std::string>();
                if (sha_changed || it->size != file.value("bytes", std::uint64_t{0})) {
                    files.push_back({{"path", path}, {"bytes", it->size}, {"change", "changed"}});
                    bytes += it->size;
                }
            }
            if (files.empty()) { continue; }
            updates.push_back({{"job_id", job.value("id", "")},
                               {"repo", place.first},
                               {"target_dir", place.second},
                               {"revision", job.value("revision", "")},
                               {"latest_revision", listing.revision},
                               {"files", files},
                               {"bytes", bytes}});
        } catch (const std::exception& failure) { error = failure.what(); }
    }
    std::lock_guard lock(mutex_);
    updates_    = updates;
    error_      = error;
    checked_at_ = now_ms();
}

// --- Infernix --------------------------------------------------------------------------------

InfernixUpdate::InfernixUpdate(Options options) : options_(std::move(options)) {
    worker_ = std::thread([this] { run(); });
}

InfernixUpdate::~InfernixUpdate() { shutdown(); }

void InfernixUpdate::shutdown() {
    {
        std::lock_guard lock(mutex_);
        if (stopping_) { return; }
        stopping_ = true;
    }
    cancel_ = true;
    cv_.notify_all();
    if (worker_.joinable()) { worker_.join(); }
}

Json InfernixUpdate::state() const {
    std::lock_guard lock(mutex_);
    return state_locked();
}

Json InfernixUpdate::state_locked() const {
    const std::optional<std::string> current = release_version(options_.build_id);
    std::optional<std::string> blocked;
#ifndef _WIN32
    blocked = "Infernix releases are Windows builds; on Linux, update from the source.";
#else
    if (!latest_.is_null() && latest_["asset"].is_null()) {
        blocked = "The latest release has no Windows zip.";
    } else if (!writable(options_.install_dir)) {
        blocked = "The launcher's folder (" + path_utf8(options_.install_dir) + ") cannot be written.";
    }
#endif
    const bool newer = !latest_.is_null() && current &&
                       compare_versions(latest_.value("version", ""), *current) > 0;
    Json progress = nullptr;
    if (phase_ == "downloading" || phase_ == "ready") { progress = {{"done", done_}, {"total", total_}}; }
    return {{"current", options_.build_id},
            {"current_version", current ? Json(*current) : Json(nullptr)},
            {"latest", latest_},
            {"newer", newer},
            {"blocked", blocked ? Json(*blocked) : Json(nullptr)},
            {"state", phase_},
            {"progress", progress},
            {"checked_at", checked_at_ ? Json(*checked_at_) : Json(nullptr)},
            {"error", error_ ? Json(*error_) : Json(nullptr)}};
}

void InfernixUpdate::publish() {
    if (options_.changed) { options_.changed(state()); }
}

void InfernixUpdate::check() {
    {
        std::lock_guard lock(mutex_);
        check_requested_ = true;
    }
    cv_.notify_all();
}

void InfernixUpdate::download() {
    {
        std::lock_guard lock(mutex_);
        if (latest_.is_null() || latest_["asset"].is_null()) { throw std::runtime_error("there is no release to download"); }
        if (phase_ == "downloading" || phase_ == "installing") { return; }
        download_requested_ = true;
        cancel_             = false;
    }
    cv_.notify_all();
}

void InfernixUpdate::run() {
    auto next = std::chrono::steady_clock::now() + std::chrono::seconds(20);
    std::unique_lock lock(mutex_);
    while (!stopping_) {
        cv_.wait_until(lock, next, [this] { return stopping_ || check_requested_ || download_requested_; });
        if (stopping_) { break; }
        if (download_requested_) {
            download_requested_ = false;
            lock.unlock();
            download_now();
            lock.lock();
            continue;
        }
        check_requested_ = false;
        if (phase_ == "downloading" || phase_ == "installing") { continue; }
        lock.unlock();
        check_now();
        lock.lock();
        next = std::chrono::steady_clock::now() + kReleaseCheckEvery;
    }
}

void InfernixUpdate::check_now() {
    {
        std::lock_guard lock(mutex_);
        if (phase_ != "ready") { phase_ = "checking"; }
    }
    publish();
    const HttpResult result = http_get(kReleaseApi, "", 30);
    {
        std::lock_guard lock(mutex_);
        checked_at_ = now_ms();
        if (phase_ == "checking") { phase_ = "idle"; }
        if (!result.error.empty() || result.status != 200) {
            error_ = result.error.empty() ? "GitHub answered HTTP " + std::to_string(result.status)
                                          : "cannot reach GitHub: " + result.error;
        } else {
            try {
                const Json release = Json::parse(result.body);
                Json asset         = nullptr;
                for (const Json& candidate : release.value("assets", Json::array())) {
                    const std::string name = candidate.value("name", "");
                    if (name.starts_with("infernix-") && name.ends_with("-windows-x64.zip")) {
                        std::string digest = candidate.value("digest", "");
                        asset = {{"name", name},
                                 {"bytes", candidate.value("size", std::uint64_t{0})},
                                 {"sha256", digest.starts_with("sha256:") ? Json(digest.substr(7)) : Json(nullptr)},
                                 {"url", candidate.value("browser_download_url", "")}};
                        break;
                    }
                }
                const std::string tag = release.value("tag_name", "");
                latest_ = {{"tag", tag},
                           {"version", release_version(tag).value_or(tag)},
                           {"published_at", release.value("published_at", "")},
                           {"notes", release.value("body", "")},
                           {"url", release.value("html_url", "")},
                           {"asset", asset}};
                error_.reset();
            } catch (const Json::exception& failure) { error_ = std::string("unreadable release: ") + failure.what(); }
        }
    }
    publish();
}

void InfernixUpdate::download_now() {
    std::string url, name;
    std::optional<std::string> sha;
    {
        std::lock_guard lock(mutex_);
        const Json asset = latest_["asset"];
        url              = asset.value("url", "");
        name             = asset.value("name", "update.zip");
        if (asset["sha256"].is_string()) { sha = asset["sha256"].get<std::string>(); }
        phase_ = "downloading";
        done_  = 0;
        total_ = asset.value("bytes", std::uint64_t{0});
        error_.reset();
    }
    publish();
    std::error_code ignored;
    std::filesystem::create_directories(options_.work_dir, ignored);
    const std::filesystem::path part = options_.work_dir / (name + ".part");
    const std::filesystem::path zip  = options_.work_dir / name;
    std::string failure;
    {
        std::ofstream out(part, std::ios::binary | std::ios::trunc);
        std::unique_ptr<CURL, decltype(&curl_easy_cleanup)> curl(curl_easy_init(), &curl_easy_cleanup);
        DownloadTarget target{&out, &done_, &mutex_, &cancel_};
        if (!out || !curl) {
            failure = "cannot write " + path_utf8(part);
        } else {
            curl_easy_setopt(curl.get(), CURLOPT_URL, url.c_str());
            curl_easy_setopt(curl.get(), CURLOPT_FOLLOWLOCATION, 1L);
            curl_easy_setopt(curl.get(), CURLOPT_PROTOCOLS_STR, "https");
            curl_easy_setopt(curl.get(), CURLOPT_REDIR_PROTOCOLS_STR, "https");
            curl_easy_setopt(curl.get(), CURLOPT_USERAGENT, "infernix-launcher");
            curl_easy_setopt(curl.get(), CURLOPT_CONNECTTIMEOUT, 30L);
            curl_easy_setopt(curl.get(), CURLOPT_LOW_SPEED_LIMIT, 16384L);
            curl_easy_setopt(curl.get(), CURLOPT_LOW_SPEED_TIME, 60L);
            curl_easy_setopt(curl.get(), CURLOPT_NOSIGNAL, 1L);
#ifdef _WIN32
            curl_easy_setopt(curl.get(), CURLOPT_SSL_OPTIONS, static_cast<long>(CURLSSLOPT_NATIVE_CA));
#endif
            curl_easy_setopt(curl.get(), CURLOPT_WRITEFUNCTION, write_file);
            curl_easy_setopt(curl.get(), CURLOPT_WRITEDATA, &target);
            curl_easy_setopt(curl.get(), CURLOPT_NOPROGRESS, 0L);
            curl_easy_setopt(curl.get(), CURLOPT_XFERINFOFUNCTION, download_progress);
            curl_easy_setopt(curl.get(), CURLOPT_XFERINFODATA, &target);
            const CURLcode code = curl_easy_perform(curl.get());
            long status         = 0;
            curl_easy_getinfo(curl.get(), CURLINFO_RESPONSE_CODE, &status);
            if (code != CURLE_OK || status != 200) {
                failure = code != CURLE_OK ? std::string(curl_easy_strerror(code)) : "HTTP " + std::to_string(status);
            }
        }
    }
    if (failure.empty() && sha) {
        const std::atomic<bool> never{false};
        const std::optional<std::string> digest = sha256_file(part, nullptr, never);
        if (!digest || *digest != *sha) { failure = "the download does not match the release's SHA-256"; }
    }
    if (failure.empty()) {
        std::filesystem::rename(part, zip, ignored);
        if (ignored) { failure = "cannot rename " + path_utf8(part) + ": " + ignored.message(); }
    } else {
        std::filesystem::remove(part, ignored);
    }
    {
        std::lock_guard lock(mutex_);
        if (failure.empty()) {
            phase_ = "ready";
            zip_   = zip;
        } else {
            phase_ = "failed";
            error_ = "the update could not be downloaded: " + failure;
        }
    }
    publish();
}

void InfernixUpdate::install() {
#ifndef _WIN32
    throw std::runtime_error("Infernix releases are Windows builds; on Linux, update from the source");
#else
    std::filesystem::path zip;
    std::string version;
    {
        std::lock_guard lock(mutex_);
        if (phase_ != "ready" || zip_.empty()) { throw std::runtime_error("download the update first"); }
        zip     = zip_;
        version = latest_.value("version", "update");
        phase_  = "installing";
    }
    publish();
    const auto fail = [&](const std::string& message) {
        {
            std::lock_guard lock(mutex_);
            phase_ = "failed";
            error_ = message;
        }
        publish();
        throw std::runtime_error(message);
    };
    // Unpack with Windows' own tar (bsdtar reads zip).
    const std::filesystem::path unpacked = options_.work_dir / ("infernix-" + version);
    std::error_code ignored;
    std::filesystem::remove_all(unpacked, ignored);
    std::filesystem::create_directories(unpacked, ignored);
    wchar_t system[MAX_PATH];
    const UINT length = GetSystemDirectoryW(system, MAX_PATH);
    const std::filesystem::path tar = std::filesystem::path(std::wstring(system, length)) / L"tar.exe";
    const std::optional<CaptureResult> unpack =
        run_capture(tar, {"-xf", path_utf8(zip), "-C", path_utf8(unpacked)}, std::chrono::minutes(10));
    if (!unpack || unpack->exit_code != 0) {
        fail("the release zip could not be unpacked" + (unpack ? ": " + trim(unpack->output) : std::string()));
    }
    const std::optional<std::filesystem::path> root = release_root(unpacked);
    if (!root) { fail("the release zip holds no infernix-launcher.exe"); }

    // The hand-over: wait for this process to exit, copy the release over the install folder
    // (copy, not mirror: the memory file, logs and chats stay), start the new launcher.
    const auto batch = [](const std::string& text) {
        std::string out;
        for (const char c : text) {
            out.push_back(c);
            if (c == '%') { out.push_back('%'); }
        }
        return out;
    };
    const std::string source = path_utf8(*root);
    const std::string target = path_utf8(options_.install_dir);
    std::string arguments;
    for (const std::string& arg : options_.restart_args) { arguments += " " + windows_quote(arg); }
    const std::string script =
        "@echo off\r\n"
        "rem Infernix update: written by infernix-launcher\r\n"
        "set \"PID=" + std::to_string(GetCurrentProcessId()) + "\"\r\n"
        ":wait\r\n"
        "tasklist /FI \"PID eq %PID%\" /NH 2>nul | find \" %PID% \" >nul\r\n"
        "if not errorlevel 1 (\r\n"
        "  timeout /t 1 /nobreak >nul\r\n"
        "  goto wait\r\n"
        ")\r\n" +
        batch("robocopy \"" + source + "\" \"" + target + "\" /E /R:10 /W:2 /NFL /NDL /NJH /NJS /NP >nul") + "\r\n"
        "if errorlevel 8 (\r\n" +
        batch("  echo The update could not copy its files into \"" + target + "\".") + "\r\n"
        "  pause\r\n"
        "  exit /b 1\r\n"
        ")\r\n" +
        batch("start \"\" /D \"" + target + "\" \"" + target + "\\infernix-launcher.exe\"" + arguments) + "\r\n";
    const std::filesystem::path script_path = options_.work_dir / "apply-update.bat";
    try {
        write_file_atomic(script_path, script);
    } catch (const std::exception& error) { fail(error.what()); }
    if (!spawn_detached(std::filesystem::path(std::wstring(system, length)) / L"cmd.exe",
                        {"/c", path_utf8(script_path)})) {
        fail("cannot start the update script");
    }
#endif
}

} // namespace infernix::launcher
