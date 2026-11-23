#include "launcher/ablation.h"

#include "launcher/sha256.h"

#include <curl/curl.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <memory>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <system_error>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace infernix::launcher {

namespace {

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

// The portable Python Set up downloads when no installed Python 3.11 or newer with pip is found: CPython from
// python-build-standalone (the relocatable builds uv installs; pip included), one release pinned with each archive's
// SHA-256. The archive holds one folder, python/.
struct PortablePython {
    const char* version;
    const char* url;
    const char* sha256;
    std::uint64_t bytes;
};
#if defined(_WIN32) && defined(_M_X64)
constexpr PortablePython kPortable{
    "3.13.18",
    "https://github.com/astral-sh/python-build-standalone/releases/download/20261110/"
    "cpython-3.13.18%2B20261110-x86_64-pc-windows-msvc-install_only.tar.gz",
    "7e93813bd362c6413f30788b248590e2f169df2d17dcac81aafbcc7a6eeb542c", 48466149};
constexpr bool kHasPortable = true;
#elif defined(__linux__) && defined(__x86_64__)
constexpr PortablePython kPortable{
    "3.13.18",
    "https://github.com/astral-sh/python-build-standalone/releases/download/20261110/"
    "cpython-3.13.18%2B20261110-x86_64-unknown-linux-gnu-install_only.tar.gz",
    "50c774f6674740739f791c05a9177ed03e2fe02abebda9bca3512eb7309b37cc", 77155083};
constexpr bool kHasPortable = true;
#else
constexpr PortablePython kPortable{"", "", "", 0};
constexpr bool kHasPortable = false;
#endif

constexpr std::size_t kTaskLogLines = 200;
constexpr std::size_t kJobLogLines  = 300;
// The tool's orderly stop ends its engines one at a time, each saving its expert-cache state (up to a minute); a run
// that has not ended this long after the stop request is ended at once. Shorter while the launcher exits.
constexpr std::chrono::seconds kStopGrace{180};
constexpr std::chrono::seconds kShutdownGrace{60};
constexpr const char* kSetupFile  = "infernix-setup.json";
constexpr const char* kJobFile    = "job.json";
constexpr const char* kPythonFile = "python.json";
// The version, whether pip is there (Set up installs with it: Debian and Ubuntu leave it out of their Python until
// python3-pip is installed), and the interpreter it really is.
constexpr const char* kProbe =
    "import sys, importlib.util; print('INFERNIX-PYTHON', '.'.join(map(str, sys.version_info[:3])), "
    "int(importlib.util.find_spec('pip') is not None), sys.executable, sep='|')";
constexpr std::string_view kProbeTag = "INFERNIX-PYTHON|";

bool active_state(const std::string& state) {
    return state == "waiting_gpu" || state == "running" || state == "stopping";
}

// A sidecar's folder name: one any file system takes.
bool valid_name(const std::string& name) {
    static const std::regex pattern("[A-Za-z0-9][A-Za-z0-9._-]{0,63}");
    return std::regex_match(name, pattern) && name.back() != '.';
}

bool valid_preset(const std::string& preset) {
    static const std::regex pattern("[a-z0-9][a-z0-9-]{0,63}");
    return std::regex_match(preset, pattern);
}

// A child's output as lines: \n and \r end one (pip redraws its progress bar with \r).
class Lines {
public:
    explicit Lines(std::function<void(std::string)> emit) : emit_(std::move(emit)) {}
    void feed(std::string_view chunk) {
        for (const char c : chunk) {
            if (c == '\n' || c == '\r') {
                flush();
            } else {
                partial_.push_back(c);
            }
        }
    }
    void flush() {
        if (!partial_.empty()) { emit_(strip_ansi(partial_)); }
        partial_.clear();
    }

private:
    std::function<void(std::string)> emit_;
    std::string partial_;
};

void push_line(std::deque<std::string>& log, std::string line, std::size_t limit) {
    log.push_back(std::move(line));
    while (log.size() > limit) { log.pop_front(); }
}

// The last non-empty lines of a process's output, for a message.
std::string tail_lines(const std::string& output, std::size_t count) {
    std::vector<std::string> lines;
    std::stringstream in(output);
    std::string line;
    while (std::getline(in, line)) {
        line = trim(strip_ansi(line));
        if (!line.empty()) { lines.push_back(line); }
    }
    std::string out;
    for (std::size_t i = lines.size() > count ? lines.size() - count : 0; i < lines.size(); ++i) {
        if (!out.empty()) { out += '\n'; }
        out += lines[i];
    }
    return out;
}

std::string tail_lines(const std::deque<std::string>& log, std::size_t count) {
    std::string text;
    for (std::size_t i = log.size() > count ? log.size() - count : 0; i < log.size(); ++i) { text += log[i] + '\n'; }
    return tail_lines(text, count);
}

struct Probe {
    fs::path executable;
    std::string version;
    int major = 0;
    int minor = 0;
    bool pip  = false;
};

// Runs `exe` as a Python and reads its version and the interpreter it really is (sys.executable): the Microsoft
// Store's python.exe stub, a py launcher entry whose Python was removed, or a Python 2 answer nothing usable.
std::optional<Probe> probe_python(const fs::path& exe, std::string& why) {
    const std::optional<CaptureResult> result = run_capture(exe, {"-I", "-c", kProbe}, std::chrono::seconds(20));
    if (!result) {
        why = "did not start, or did not answer within 20 s";
        return std::nullopt;
    }
    std::stringstream in(result->output);
    std::string line;
    while (result->exit_code == 0 && std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') { line.pop_back(); }
        if (!line.starts_with(kProbeTag)) { continue; }
        // "version|pip|executable" (the executable last: a POSIX path may hold a bar).
        const std::size_t bar = line.find('|', kProbeTag.size());
        if (bar == std::string::npos) { break; }
        const std::size_t second = line.find('|', bar + 1);
        if (second == std::string::npos) { break; }
        Probe probe;
        probe.version              = line.substr(kProbeTag.size(), bar - kProbeTag.size());
        probe.pip                  = line.substr(bar + 1, second - bar - 1) == "1";
        const std::string location = line.substr(second + 1);
        probe.executable           = location.empty() ? exe : utf8_path(location);
        if (std::sscanf(probe.version.c_str(), "%d.%d", &probe.major, &probe.minor) != 2) { break; }
        return probe;
    }
    const std::string said = tail_lines(result->output, 1);
    why = said.empty() ? "ended with exit code " + std::to_string(result->exit_code) : said.substr(0, 200);
    return std::nullopt;
}

struct Fetch {
    std::ofstream* out = nullptr;
    std::uint64_t done = 0;
    const std::atomic<bool>* cancel = nullptr;
    std::function<void(std::uint64_t)> progress;
};

size_t write_fetched(char* data, size_t size, size_t count, void* user) {
    auto* fetch        = static_cast<Fetch*>(user);
    const size_t bytes = size * count;
    fetch->out->write(data, static_cast<std::streamsize>(bytes));
    if (!*fetch->out) { return 0; }
    fetch->done += bytes;
    if (fetch->progress) { fetch->progress(fetch->done); }
    return bytes;
}

int fetch_progress(void* user, curl_off_t, curl_off_t, curl_off_t, curl_off_t) {
    return static_cast<Fetch*>(user)->cancel->load() ? 1 : 0;
}

// Downloads `url` (HTTPS only, redirects followed, as the launcher's own update does) into `file`: "" when it
// arrived, else why not. `progress` gets the bytes so far.
std::string download(const std::string& url, const fs::path& file, const std::atomic<bool>& cancel,
                     std::function<void(std::uint64_t)> progress) {
    std::ofstream out(file, std::ios::binary | std::ios::trunc);
    std::unique_ptr<CURL, decltype(&curl_easy_cleanup)> curl(curl_easy_init(), &curl_easy_cleanup);
    if (!out || !curl) { return "cannot write " + path_utf8(file); }
    Fetch fetch{&out, 0, &cancel, std::move(progress)};
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
    curl_easy_setopt(curl.get(), CURLOPT_WRITEFUNCTION, write_fetched);
    curl_easy_setopt(curl.get(), CURLOPT_WRITEDATA, &fetch);
    curl_easy_setopt(curl.get(), CURLOPT_NOPROGRESS, 0L);
    curl_easy_setopt(curl.get(), CURLOPT_XFERINFOFUNCTION, fetch_progress);
    curl_easy_setopt(curl.get(), CURLOPT_XFERINFODATA, &fetch);
    const CURLcode code = curl_easy_perform(curl.get());
    long status         = 0;
    curl_easy_getinfo(curl.get(), CURLINFO_RESPONSE_CODE, &status);
    out.close();
    if (cancel.load()) { return "stopped: the launcher is exiting"; }
    if (code != CURLE_OK) { return curl_easy_strerror(code); }
    if (status != 200) { return "HTTP " + std::to_string(status); }
    return "";
}

// The system's tar: Windows' own bsdtar (System32\tar.exe, Windows 10 1803 and later), tar on PATH elsewhere.
fs::path system_tar() {
#ifdef _WIN32
    wchar_t system[MAX_PATH];
    const UINT length = GetSystemDirectoryW(system, MAX_PATH);
    return fs::path(std::wstring(system, length)) / L"tar.exe";
#else
    return find_program("tar").value_or(fs::path("/usr/bin/tar"));
#endif
}

// The Pythons to try, in order: on Windows those the py launcher lists (py -0p, newest first), then python3 and
// python on PATH.
std::vector<fs::path> python_candidates() {
    std::vector<fs::path> found;
#ifdef _WIN32
    if (const std::optional<fs::path> py = find_program("py")) {
        if (const std::optional<CaptureResult> listed = run_capture(*py, {"-0p"}, std::chrono::seconds(15))) {
            std::stringstream in(listed->output);
            std::string line;
            while (std::getline(in, line)) {
                // " -V:3.13 *        C:\Python313\python.exe": the path starts at its drive.
                const std::size_t drive = line.find(":\\");
                if (drive == std::string::npos || drive == 0 ||
                    std::isalpha(static_cast<unsigned char>(line[drive - 1])) == 0) {
                    continue;
                }
                found.push_back(utf8_path(trim(line.substr(drive - 1))));
            }
        }
    }
    for (const char* name : {"python3", "python"}) {
        if (const std::optional<fs::path> program = find_program(name)) { found.push_back(*program); }
    }
#else
    for (const char* name : {"python3", "python3.14", "python3.13", "python3.12", "python3.11", "python"}) {
        if (const std::optional<fs::path> program = find_program(name)) { found.push_back(*program); }
    }
#endif
    std::vector<fs::path> unique;
    for (const fs::path& path : found) {
        const bool seen = std::any_of(unique.begin(), unique.end(), [&](const fs::path& other) {
#ifdef _WIN32
            return iequals(path_utf8(other), path_utf8(path));
#else
            return other == path;
#endif
        });
        if (!seen) { unique.push_back(path); }
    }
    return unique;
}

bool new_enough(int major, int minor) { return major > 3 || (major == 3 && minor >= 11); }

std::string minor_version(int major, int minor) { return std::to_string(major) + "." + std::to_string(minor); }

} // namespace

Ablation::Ablation(fs::path data_dir, std::optional<fs::path> tool_dir, Hooks hooks)
    : data_dir_(std::move(data_dir)), root_(data_dir_ / "ablation"), prompts_(root_ / "prompts"),
      runs_(root_ / "runs"), sidecars_(root_ / "sidecars"), packages_(root_ / "packages"),
      portable_(executable_dir() / "ablate-python"), hooks_(std::move(hooks)) {
    std::error_code error;
    const auto has_tool = [&error](const fs::path& dir) { return fs::is_regular_file(dir / "ablate.py", error); };
    if (tool_dir) {
        tool_looked_in_.push_back(path_utf8(*tool_dir));
        if (has_tool(*tool_dir)) { tool_ = fs::absolute(*tool_dir, error); }
    } else {
        // A release: the `ablate` folder beside the launcher.
        const fs::path beside = executable_dir() / "ablate";
        tool_looked_in_.push_back(path_utf8(beside));
        if (has_tool(beside)) { tool_ = beside; }
        // A build tree: tools/ablate of the checkout it is in, a few folders up.
        fs::path dir = executable_dir();
        for (int up = 0; !tool_ && up < 6; ++up) {
            if (has_tool(dir / "tools" / "ablate")) { tool_ = dir / "tools" / "ablate"; }
            if (!dir.has_parent_path() || dir.parent_path() == dir) { break; }
            dir = dir.parent_path();
        }
        if (!tool_) { tool_looked_in_.push_back("tools/ablate of a checkout above " + path_utf8(executable_dir())); }
    }
    // The last run as the launcher left it: one the launcher ended mid-way is stopped, and can be run again.
    if (const std::optional<std::string> text = read_file(root_ / kJobFile)) {
        try {
            // json_or, not value(): a run the launcher did not finish was saved with "ended": null.
            const Json saved = Json::parse(*text);
            const Json run   = saved.value("run", Json::object());
            Job job;
            job.run.config_id  = json_or<std::string>(run, "config_id", "");
            job.run.model_name = json_or<std::string>(run, "model_name", "");
            job.run.preset     = json_or<std::string>(run, "preset", "");
            job.run.trials     = json_or<int>(run, "trials", 300);
            job.run.name       = json_or<std::string>(run, "name", "");
            job.run_dir        = utf8_path(json_or<std::string>(saved, "run_dir", ""));
            job.state          = json_or<std::string>(saved, "state", "stopped");
            job.message        = json_or<std::string>(saved, "message", "");
            job.started        = json_or<std::int64_t>(saved, "started", 0);
            job.ended          = json_or<std::int64_t>(saved, "ended", 0);
            job.progress       = saved.contains("progress") ? saved["progress"] : Json(nullptr);
            if (saved.contains("exported") && saved["exported"].is_string()) { job.exported = saved["exported"]; }
            if (active_state(job.state)) {
                job.state   = "stopped";
                job.message = "the launcher ended during the run; Run again continues it (finished work is kept)";
                job.ended   = now_ms();
            }
            job_ = std::move(job);
        } catch (const Json::exception&) {}
    }
    // A Linux launcher that ended during a run leaves the tool's engines running (each in a session of its own).
    if (job_ && !job_->run_dir.empty()) { end_leftover_engines(job_->run_dir); }
}

Ablation::~Ablation() { shutdown(); }

std::vector<std::string> Ablation::tool_args(std::vector<std::string> args) const {
    // Unbuffered and UTF-8 whatever the console's code page, so the log keeps up and a prompt or path outside it
    // cannot end the run; no user site-packages, so only the packages the launcher installed are imported.
    std::vector<std::string> out = {"-u", "-X", "utf8", "-s", path_utf8(*tool_ / "ablate.py")};
    out.insert(out.end(), std::make_move_iterator(args.begin()), std::make_move_iterator(args.end()));
    return out;
}

EnvOverlay Ablation::tool_env(const EnvOverlay& extra) const {
    EnvOverlay env = extra;
    env.emplace_back("PYTHONPATH", path_utf8(packages_));
    env.emplace_back("PYTHONDONTWRITEBYTECODE", "1"); // the launcher's folder may not be writable
    env.emplace_back("PYTHONUNBUFFERED", "1");
    env.emplace_back("INFERNIX_ABLATE_TOOL_VERSION", "infernix-launcher " + hooks_.version);
    return env;
}

void Ablation::detect_python() {
    std::lock_guard detect(detect_mutex_);
    std::string chosen;
    if (const std::optional<std::string> text = read_file(root_ / kPythonFile)) {
        try {
            chosen = trim(Json::parse(*text).value("python", ""));
        } catch (const Json::exception&) {}
    }
    Json tried = Json::array();
    std::optional<Python> found;
    // The one chosen on the page; else the portable Python Set up downloaded, then the installed ones.
    std::vector<fs::path> candidates;
    std::error_code error;
    if (!chosen.empty()) {
        candidates.push_back(utf8_path(chosen));
    } else {
        if (fs::exists(portable_python(), error)) { candidates.push_back(portable_python()); }
        for (fs::path& candidate : python_candidates()) { candidates.push_back(std::move(candidate)); }
    }
    for (const fs::path& candidate : candidates) {
        std::string why;
        const std::optional<Probe> probe = probe_python(candidate, why);
        if (probe && new_enough(probe->major, probe->minor) && probe->pip) {
            found = Python{probe->executable, probe->version, probe->major, probe->minor};
            tried.push_back({{"path", path_utf8(candidate)}, {"result", "Python " + probe->version}});
            break;
        }
        tried.push_back({{"path", path_utf8(candidate)},
                         {"result", !probe                                    ? why
                                    : !new_enough(probe->major, probe->minor) ? "Python " + probe->version + ", older than 3.11"
                                                                              : "Python " + probe->version + ", without pip"}});
    }
    std::lock_guard lock(mutex_);
    detected_      = true;
    python_        = found;
    python_chosen_ = chosen;
    python_tried_  = std::move(tried);
    catalogue_     = nullptr;
    catalogue_error_.clear();
}

void Ablation::choose_python(const std::string& executable) {
    const std::string path = trim(executable);
    std::error_code error;
    if (!path.empty() && !fs::exists(utf8_path(path), error)) { throw std::invalid_argument(path + " does not exist"); }
    fs::create_directories(root_, error);
    write_file_atomic(root_ / kPythonFile, Json{{"python", path}}.dump(1));
    detect_python();
    publish();
}

fs::path Ablation::portable_python() const {
#ifdef _WIN32
    return portable_ / "python" / "python.exe";
#else
    return portable_ / "python" / "bin" / "python3";
#endif
}

void Ablation::task_line(std::string line) {
    {
        std::lock_guard lock(mutex_);
        if (task_) { push_line(task_->log, std::move(line), kTaskLogLines); }
    }
    publish();
}

std::string Ablation::install_portable() {
    std::error_code error;
    // Unpacked beside, then renamed into place: a download or unpacking that stops part way leaves nothing behind.
    const fs::path archive = portable_.parent_path() / "ablate-python.tar.gz.part";
    const fs::path staging = portable_.parent_path() / "ablate-python.new";
    fs::remove_all(staging, error);
    fs::create_directories(staging, error);
    if (error) {
        return "cannot write " + path_utf8(portable_.parent_path()) + " (" + error_text(error) +
               "): install Python 3.11 or newer instead, or run the launcher from a folder you can write to";
    }
    const std::uint64_t mib = kPortable.bytes >> 20;
    task_line(std::string("downloading CPython ") + kPortable.version + " (python-build-standalone, " +
              std::to_string(mib) + " MiB) from " + kPortable.url);
    int shown = 0; // tenths reported
    std::string failure = download(kPortable.url, archive, cancel_task_, [&](std::uint64_t done) {
        const int tenths = kPortable.bytes ? static_cast<int>(done * 10 / kPortable.bytes) : 0;
        if (tenths > shown && tenths < 10) {
            shown = tenths;
            task_line("  " + std::to_string(done >> 20) + " of " + std::to_string(mib) + " MiB");
        }
    });
    if (failure.empty()) {
        const std::atomic<bool> never{false};
        const std::optional<std::string> digest = sha256_file(archive, nullptr, never);
        if (!digest || *digest != kPortable.sha256) {
            failure = "the download does not match its pinned SHA-256 (" + digest.value_or("unreadable") + ")";
        }
    }
    if (!failure.empty()) {
        fs::remove(archive, error);
        fs::remove_all(staging, error);
        return "downloading the portable Python failed: " + failure;
    }
    task_line("checked its SHA-256; unpacking into " + path_utf8(portable_));
    const std::optional<CaptureResult> unpacked =
        run_capture(system_tar(), {"-xzf", path_utf8(archive), "-C", path_utf8(staging)}, std::chrono::minutes(10));
    fs::remove(archive, error);
    if (!unpacked || unpacked->exit_code != 0 || !fs::exists(staging / portable_python().lexically_relative(portable_), error)) {
        fs::remove_all(staging, error);
        return "unpacking the portable Python failed" + (unpacked ? ": " + tail_lines(unpacked->output, 3) : std::string());
    }
    fs::remove_all(portable_, error);
    fs::rename(staging, portable_, error);
    if (error) { return "cannot move the portable Python into " + path_utf8(portable_) + ": " + error_text(error); }
    task_line("CPython " + std::string(kPortable.version) + " is in " + path_utf8(portable_));
    return "";
}

bool Ablation::packages_ready_locked() const {
    if (!python_ || !tool_) { return false; }
    const std::optional<std::string> text         = read_file(packages_ / kSetupFile);
    const std::optional<std::string> requirements = read_file(*tool_ / "requirements.txt");
    if (!text || !requirements) { return false; }
    try {
        const Json setup = Json::parse(*text);
        // Compiled packages (numpy) are built for one minor version of Python.
        return setup.value("python_minor", "") == minor_version(python_->major, python_->minor) &&
               setup.value("requirements_sha256", "") == sha256_hex(*requirements);
    } catch (const Json::exception&) { return false; }
}

void Ablation::load_catalogue() {
    std::optional<Python> python;
    {
        std::lock_guard lock(mutex_);
        if (!catalogue_.is_null() || !catalogue_error_.empty() || !python_ || !tool_) { return; }
        python = python_;
    }
    // `datasets list` needs only Python's standard library: the list shows before the packages are installed.
    const std::optional<CaptureResult> result =
        run_capture(python->executable, tool_args({"datasets", "list", "--json"}), std::chrono::seconds(60), tool_env());
    Json catalogue = nullptr;
    std::string error;
    if (!result) {
        error = "the ablation tool did not answer `datasets list`";
    } else if (result->exit_code != 0) {
        error = tail_lines(result->output, 6);
    } else {
        try {
            catalogue = Json::parse(result->output);
            if (!catalogue.is_array()) {
                catalogue = nullptr;
                error     = "the ablation tool's prompt-set list is not a JSON array";
            }
        } catch (const Json::exception&) { error = "the ablation tool's prompt-set list is not JSON"; }
    }
    std::lock_guard lock(mutex_);
    catalogue_       = std::move(catalogue);
    catalogue_error_ = error.empty() && catalogue_.is_null() ? "no prompt sets" : error;
}

Json Ablation::fetched_presets() const {
    Json out = Json::object();
    std::error_code error;
    for (const auto& entry : fs::directory_iterator(prompts_, error)) {
        std::error_code status;
        if (!entry.is_directory(status)) { continue; }
        const std::optional<std::string> text = read_file(entry.path() / "sets.json");
        if (!text || !fs::is_regular_file(entry.path() / "harmful.txt", status) ||
            !fs::is_regular_file(entry.path() / "harmless.txt", status)) {
            continue;
        }
        try {
            const Json sets = Json::parse(*text);
            out[sets.value("preset", path_utf8(entry.path().filename()))] = {
                {"title", sets.value("title", "")},
                {"fetched", sets.value("fetched", "")},
                {"counts", sets.value("counts", Json::object())},
                {"sources", sets.value("sources", Json::object())},
                {"source_text", sets.value("source_text", "")},
                {"dir", path_utf8(entry.path())},
            };
        } catch (const Json::exception&) {}
    }
    return out;
}

Json Ablation::job_json_locked() const {
    if (!job_) { return nullptr; }
    Json log = Json::array();
    for (const std::string& line : job_->log) { log.push_back(line); }
    return {
        {"run",
         {{"config_id", job_->run.config_id},
          {"model_name", job_->run.model_name},
          {"preset", job_->run.preset},
          {"trials", job_->run.trials},
          {"name", job_->run.name}}},
        {"run_dir", path_utf8(job_->run_dir)},
        {"state", job_->state},
        {"active", job_active_},
        {"message", job_->message},
        {"started", job_->started},
        {"ended", job_->ended == 0 ? Json(nullptr) : Json(job_->ended)},
        {"progress", job_->progress},
        {"log", std::move(log)},
        {"exported", job_->exported ? Json(*job_->exported) : Json(nullptr)},
    };
}

Json Ablation::state() {
    bool detected = false;
    {
        std::lock_guard lock(mutex_);
        detected = detected_;
    }
    if (!detected) { detect_python(); }
    load_catalogue();
    Json fetched = fetched_presets();
    std::lock_guard lock(mutex_);
    Json task = nullptr;
    if (task_) {
        Json log = Json::array();
        for (const std::string& line : task_->log) { log.push_back(line); }
        task = {{"kind", task_->kind},       {"preset", task_->preset}, {"state", task_->state},
                {"message", task_->message}, {"log", std::move(log)},   {"started", task_->started},
                {"ended", task_->ended == 0 ? Json(nullptr) : Json(task_->ended)}};
    }
    return {
        {"tool", tool_ ? Json{{"found", true}, {"dir", path_utf8(*tool_)}}
                       : Json{{"found", false}, {"looked_in", tool_looked_in_}}},
        {"python",
         {{"chosen", python_chosen_},
          {"found", python_ ? Json{{"executable", path_utf8(python_->executable)}, {"version", python_->version}}
                            : Json(nullptr)},
          {"tried", python_tried_},
          {"minimum", "3.11"},
          // What Set up downloads when no Python is found (null where no build is pinned for this platform).
          {"portable", kHasPortable ? Json{{"version", kPortable.version},
                                           {"bytes", kPortable.bytes},
                                           {"url", kPortable.url},
                                           {"dir", path_utf8(portable_)},
                                           {"installed", [this] {
                                                std::error_code ignored;
                                                return fs::exists(portable_python(), ignored);
                                            }()}}
                                    : Json(nullptr)}}},
        {"packages", {{"ready", packages_ready_locked()}, {"dir", path_utf8(packages_)}}},
        {"catalogue", catalogue_},
        {"catalogue_error", catalogue_error_.empty() ? Json(nullptr) : Json(catalogue_error_)},
        {"prompts", std::move(fetched)},
        {"prompts_dir", path_utf8(prompts_)},
        {"sidecars_dir", path_utf8(sidecars_)},
        {"task", std::move(task)},
        {"job", job_json_locked()},
        {"limits", {{"min_trials", kMinTrials}, {"max_trials", kMaxTrials}}},
    };
}

void Ablation::publish() {
    if (hooks_.changed) { hooks_.changed(state()); }
}

void Ablation::run_task(Task task, std::function<std::string(fs::path& executable)> prepare,
                        std::vector<std::string> args, EnvOverlay env, std::function<std::string(int code)> finish) {
    if (task_thread_.joinable()) { task_thread_.join(); } // the last task's thread, which has ended
    {
        std::lock_guard lock(mutex_);
        task_ = std::move(task);
    }
    task_thread_ = std::thread([this, prepare = std::move(prepare), args = std::move(args), env = std::move(env),
                                finish = std::move(finish)] {
        auto dirty = std::make_shared<std::atomic<bool>>(true);
        auto lines = std::make_shared<Lines>([this, dirty](std::string line) {
            std::lock_guard lock(mutex_);
            if (task_) { push_line(task_->log, std::move(line), kTaskLogLines); }
            *dirty = true;
        });
        std::string error;
        int code = -1;
        try {
            fs::path executable;
            error = prepare(executable);
            if (!error.empty()) { throw std::runtime_error(error); }
            std::unique_ptr<ChildProcess> child =
                ChildProcess::start(executable, args, root_, [lines](std::string_view chunk) { lines->feed(chunk); }, env);
            {
                std::lock_guard lock(mutex_);
                task_child_ = child.get();
            }
            while (!child->wait_for(std::chrono::milliseconds(500))) {
                if (dirty->exchange(false)) { publish(); }
            }
            child->wait_output(std::chrono::seconds(10));
            code = child->exit_code().value_or(-1);
            {
                std::lock_guard lock(mutex_);
                task_child_ = nullptr;
            }
            child.reset();
            lines->flush();
            error = finish(code);
        } catch (const std::exception& failure) {
            std::lock_guard lock(mutex_);
            task_child_ = nullptr;
            error       = failure.what();
        }
        std::string kind;
        std::string message;
        {
            std::lock_guard lock(mutex_);
            if (task_) {
                task_->state   = error.empty() ? "done" : "failed";
                task_->message = error.empty() ? task_->message : error;
                task_->ended   = now_ms();
                kind           = task_->kind == "setup" ? "installing the ablation tool's packages"
                                                        : "downloading the prompt set " + task_->preset;
                message        = task_->message;
            }
            task_running_ = false;
        }
        if (hooks_.event) { hooks_.event(error.empty() ? "info" : "error", kind + (error.empty() ? ": done" : ": " + error)); }
        if (hooks_.toast) { hooks_.toast(error.empty() ? "success" : "error", error.empty() ? message : kind + " failed: " + error); }
        publish();
    });
}

void Ablation::setup() {
    std::lock_guard control(control_mutex_);
    bool download_python = false;
    std::string python_text;
    {
        std::lock_guard lock(mutex_);
        if (shutting_down_) { throw std::invalid_argument("the launcher is stopping"); }
        if (task_running_) { throw std::invalid_argument("an installation or a download is already running"); }
        if (!tool_) { throw std::invalid_argument("the ablation tool was not found (see where the page looked)"); }
        if (!python_ && (!kHasPortable || !python_chosen_.empty())) {
            throw std::invalid_argument(python_chosen_.empty()
                                            ? "no Python 3.11 or newer with pip was found: install one, or choose it"
                                            : "the Python chosen is not a Python 3.11 or newer with pip: choose another, "
                                              "or let the launcher find one");
        }
        // The packages are swapped in when pip ends: a run importing them meanwhile could fail part way.
        if (job_active_) { throw std::invalid_argument("a run uses the packages: stop it first"); }
        download_python = !python_;
        python_text     = python_ ? "Python " + python_->version : std::string("CPython ") + kPortable.version;
        task_running_   = true;
    }
    cancel_task_ = false;
    // Installed beside, then swapped in: a failed installation leaves the working one.
    const fs::path staging = root_ / "packages.new";
    std::error_code error;
    fs::remove_all(staging, error);
    fs::create_directories(root_, error);
    Task task;
    task.kind    = "setup";
    task.state   = "running";
    task.message = download_python ? "downloading a portable Python, then installing numpy and optuna"
                                   : "installing numpy and optuna for " + python_text;
    task.started = now_ms();
    const fs::path requirements = *tool_ / "requirements.txt";
    // The portable Python first when no Python was found; then pip with the Python found.
    const auto prepare = [this, download_python](fs::path& executable) -> std::string {
        if (download_python) {
            if (std::string failure = install_portable(); !failure.empty()) { return failure; }
            detect_python();
        }
        std::lock_guard lock(mutex_);
        if (!python_) { return "the portable Python does not start (the log of the Pythons tried says why)"; }
        executable = python_->executable;
        if (task_) { task_->message = "installing numpy and optuna for Python " + python_->version; }
        return "";
    };
    run_task(std::move(task), prepare,
             {"-m", "pip", "install", "--disable-pip-version-check", "--no-input", "--no-warn-script-location",
              "--target", path_utf8(staging), "-r", path_utf8(requirements)},
             {{"PYTHONUTF8", "1"}},
             [this, staging, requirements](int code) -> std::string {
                 if (code != 0) { return "pip ended with exit code " + std::to_string(code) + " (the log says why)"; }
                 std::optional<Python> python;
                 {
                     std::lock_guard lock(mutex_);
                     python = python_;
                 }
                 if (!python) { return "no Python"; }
                 std::error_code failure;
                 fs::remove_all(packages_, failure);
                 fs::rename(staging, packages_, failure);
                 if (failure) { return "cannot move the packages into " + path_utf8(packages_) + ": " + error_text(failure); }
                 write_file_atomic(packages_ / kSetupFile,
                                   Json{{"python", path_utf8(python->executable)},
                                        {"python_version", python->version},
                                        {"python_minor", minor_version(python->major, python->minor)},
                                        {"requirements_sha256", sha256_hex(read_file(requirements).value_or(""))},
                                        {"installed", now_ms()}}
                                       .dump(1));
                 std::lock_guard lock(mutex_);
                 if (task_) { task_->message = "numpy and optuna are installed for Python " + python->version; }
                 return "";
             });
    publish();
}

void Ablation::fetch(const std::string& preset) {
    std::lock_guard control(control_mutex_);
    if (!valid_preset(preset)) { throw std::invalid_argument("no prompt set " + preset); }
    std::optional<Python> python;
    std::string title = preset;
    {
        std::lock_guard lock(mutex_);
        if (shutting_down_) { throw std::invalid_argument("the launcher is stopping"); }
        if (task_running_) { throw std::invalid_argument("an installation or a download is already running"); }
        if (!tool_) { throw std::invalid_argument("the ablation tool was not found"); }
        if (!python_) { throw std::invalid_argument("no Python yet: Set up first (it downloads one when none is found)"); }
        if (catalogue_.is_array()) {
            const auto known = std::find_if(catalogue_.begin(), catalogue_.end(),
                                            [&](const Json& entry) { return entry.value("name", "") == preset; });
            if (known == catalogue_.end()) { throw std::invalid_argument("no prompt set " + preset); }
            title = known->value("title", preset);
        }
        python        = python_;
        task_running_ = true;
    }
    std::error_code error;
    fs::create_directories(prompts_, error);
    Task task;
    task.kind    = "fetch";
    task.preset  = preset;
    task.state   = "running";
    task.message = "downloading " + title;
    task.started = now_ms();
    run_task(
        std::move(task),
        [executable = python->executable](fs::path& out) {
            out = executable;
            return std::string();
        },
        tool_args({"datasets", "fetch", preset, "--out", path_utf8(prompts_)}), tool_env(),
        [this, title](int code) -> std::string {
            if (code != 0) { return "the download ended with exit code " + std::to_string(code) + " (see the log)"; }
            std::lock_guard lock(mutex_);
            if (task_) { task_->message = title + " is downloaded"; }
            return "";
        });
    publish();
}

void Ablation::start(const AblationRun& request) {
    std::lock_guard control(control_mutex_);
    std::optional<Python> python;
    {
        std::lock_guard lock(mutex_);
        if (shutting_down_) { throw std::invalid_argument("the launcher is stopping"); }
        if (job_active_) { throw std::invalid_argument("a run is in progress: stop it first"); }
        if (!tool_) { throw std::invalid_argument("the ablation tool was not found"); }
        if (!python_) { throw std::invalid_argument("no Python 3.11 or newer was found"); }
        if (!packages_ready_locked()) { throw std::invalid_argument("install the tool's packages first (Set up)"); }
        if (task_running_ && task_ && task_->kind == "setup") {
            throw std::invalid_argument("the tool's packages are being installed: wait for it");
        }
        python = python_;
    }
    if (!valid_name(request.name)) {
        throw std::invalid_argument("a sidecar name is 1 to 64 letters, digits, '.', '_' or '-', starting with a "
                                    "letter or digit");
    }
    if (request.trials < kMinTrials || request.trials > kMaxTrials) {
        throw std::invalid_argument("trials: " + std::to_string(kMinTrials) + " to " + std::to_string(kMaxTrials));
    }
    if (request.argv.size() < 2) { throw std::invalid_argument("the configuration has no command line"); }
    const fs::path prompts = prompts_ / request.preset;
    std::error_code error;
    if (!valid_preset(request.preset) || !fs::is_regular_file(prompts / "harmful.txt", error) ||
        !fs::is_regular_file(prompts / "harmless.txt", error)) {
        throw std::invalid_argument("download the prompt set " + request.preset + " first");
    }
    if (fs::exists(sidecars_ / utf8_path(request.name), error)) {
        throw std::invalid_argument("a sidecar named " + request.name + " exists in " + path_utf8(sidecars_) +
                                    ": choose another name (or delete that one on the Models page)");
    }
    // One run folder per command line, environment, infernix-serve file, prompt set and sidecar name: the same
    // request continues it, and any change starts a fresh one.
    Json env = Json::array();
    for (const auto& [name, value] : request.env) { env.push_back({name, value}); }
    std::uintmax_t serve_bytes = fs::file_size(utf8_path(request.argv[0]), error);
    if (error) { serve_bytes = 0; }
    const Json identity = {{"argv", request.argv},
                           {"env", env},
                           {"serve", {serve_bytes, modified_ms(utf8_path(request.argv[0]))}},
                           {"preset", request.preset},
                           {"name", request.name}};
    const fs::path run_dir = runs_ / utf8_path(request.name + "-" + sha256_hex(identity.dump()).substr(0, 10));
    if (!fs::is_regular_file(run_dir / "run.json", error)) {
        // init reads the configuration and refuses one it cannot fit, before anything is unloaded.
        fs::remove_all(run_dir, error);
        fs::create_directories(run_dir, error);
        if (error) { throw std::runtime_error("cannot create " + path_utf8(run_dir) + ": " + error_text(error)); }
        write_file_atomic(run_dir / "production-argv.json", Json(request.argv).dump(1));
        const std::optional<CaptureResult> result =
            run_capture(python->executable,
                        tool_args({"init", path_utf8(run_dir), "--production-argv", path_utf8(run_dir / "production-argv.json"),
                                   "--launch-dir", path_utf8(data_dir_)}),
                        std::chrono::minutes(3), tool_env(request.env));
        if (!result || result->exit_code != 0) {
            const std::string said = result ? tail_lines(result->output, 12) : "init did not finish within 3 minutes";
            fs::remove_all(run_dir, error);
            throw std::invalid_argument("the ablation tool cannot fit this configuration:\n" + said);
        }
    }
    if (job_thread_.joinable()) { job_thread_.join(); } // the last run's thread, which has ended
    {
        std::lock_guard lock(mutex_);
        Job job;
        job.run     = request;
        job.run_dir = run_dir;
        job.state   = "waiting_gpu";
        job.message = "unloading the model";
        job.started = now_ms();
        job_        = std::move(job);
        job_active_     = true;
        stop_requested_ = false;
        stop_now_       = false;
        save_job_locked();
    }
    if (hooks_.event) {
        hooks_.event("info", "ablation run of " + request.model_name + " started: prompt set " + request.preset + ", " +
                                 std::to_string(request.trials) + " trials, sidecar " + request.name + " (" +
                                 path_utf8(run_dir) + ")");
    }
    job_thread_ = std::thread([this] { job_main(); });
    publish();
}

void Ablation::stop(bool now) {
    {
        std::lock_guard lock(mutex_);
        if (!job_active_) { throw std::invalid_argument("no run is in progress"); }
        stop_now_       = stop_now_ || now;
        stop_requested_ = true;
    }
    if (hooks_.event) {
        hooks_.event("info", now ? "the ablation run is ended at once from the launcher"
                                 : "stop of the ablation run requested from the launcher");
    }
}

bool Ablation::busy() const {
    std::lock_guard lock(mutex_);
    return job_active_;
}

void Ablation::shutdown() {
    std::lock_guard control(control_mutex_);
    {
        std::lock_guard lock(mutex_);
        shutting_down_ = true;
        if (task_child_ != nullptr) { task_child_->kill(); } // a download or installation: run it again later
        cancel_task_ = true;
    }
    stop_requested_ = true;
    if (job_thread_.joinable()) { job_thread_.join(); }
    if (task_thread_.joinable()) { task_thread_.join(); }
}

void Ablation::save_job_locked() const {
    if (!job_) { return; }
    Json saved = job_json_locked();
    saved.erase("log");
    saved.erase("active");
    std::error_code error;
    fs::create_directories(root_, error);
    try {
        write_file_atomic(root_ / kJobFile, saved.dump(1));
    } catch (const std::exception&) {} // the page still shows it; a restart forgets it
}

void Ablation::end_leftover_engines(const fs::path& run_dir) {
    std::error_code error;
    for (const auto& entry : fs::directory_iterator(run_dir / "engines", error)) {
        if (entry.path().extension() != ".json") { continue; }
        const std::optional<std::string> text = read_file(entry.path());
        if (!text) { continue; }
        try {
            const Json engine        = Json::parse(*text);
            const std::uint32_t pid  = engine.value("pid", std::uint32_t{0});
            const std::string program = path_utf8(utf8_path(engine.value("executable", "")).filename());
            if (pid == 0 || program.empty()) { continue; }
            if (end_leftover_process(pid, program, std::chrono::seconds(30))) {
                std::error_code removed;
                fs::remove(entry.path(), removed);
            } else if (hooks_.event) {
                hooks_.event("warn", "the ablation run's engine (process " + std::to_string(pid) +
                                         ") did not end: end it in the Task Manager before loading a model");
            }
        } catch (const Json::exception&) {}
    }
}

void Ablation::finish_job(const std::string& state, const std::string& message) {
    std::string name;
    std::string model;
    std::optional<std::string> exported;
    {
        std::lock_guard lock(mutex_);
        job_->state   = state;
        job_->message = message;
        job_->ended   = now_ms();
        if (state == "done" && job_->progress.is_object() && job_->progress.value("export", Json(nullptr)).is_string()) {
            job_->exported = job_->progress["export"].get<std::string>();
        }
        name     = job_->run.name;
        model    = job_->run.model_name;
        exported = job_->exported;
        job_active_ = false;
        save_job_locked();
    }
    const std::string level = state == "done" ? "success" : state == "failed" ? "error" : state == "stopped" ? "info" : "warning";
    const std::string what  = state == "done"       ? "the ablation sidecar " + name + " for " + model + " is ready"
                              : state == "no_sidecar" ? "the ablation run of " + model + " made no sidecar: " + message
                              : state == "stopped"    ? "the ablation run of " + model + " stopped; Run again continues it"
                                                      : "the ablation run of " + model + " failed: " + message;
    if (hooks_.event) { hooks_.event(state == "failed" ? "error" : state == "no_sidecar" ? "warn" : "info", what); }
    if (hooks_.toast) { hooks_.toast(level, what); }
    if (state == "done" && hooks_.exported) { hooks_.exported(sidecars_ / utf8_path(name)); }
    publish();
}

void Ablation::job_main() {
    AblationRun run;
    fs::path run_dir;
    std::optional<Python> python;
    {
        std::lock_guard lock(mutex_);
        run     = job_->run;
        run_dir = job_->run_dir;
        python  = python_;
    }
    const auto set = [this](const std::string& state, const std::string& message) {
        {
            std::lock_guard lock(mutex_);
            job_->state   = state;
            job_->message = message;
            save_job_locked();
        }
        publish();
    };
    publish();
    if (!python || !hooks_.take_gpu || !hooks_.take_gpu([this] { return stop_requested_.load(); })) {
        finish_job("stopped", "stopped before the run began");
        return;
    }
    std::error_code error;
    fs::remove(run_dir / "progress.json", error); // a resumed run's last account is not this one's

    const fs::path prompts = prompts_ / run.preset;
    std::vector<std::string> args = {"auto",      path_utf8(run_dir),
                                     "--harmful", path_utf8(prompts / "harmful.txt"),
                                     "--harmless", path_utf8(prompts / "harmless.txt"),
                                     "--trials",  std::to_string(run.trials),
                                     "--out",     path_utf8(sidecars_),
                                     "--name",    run.name};
    const auto present = [](const fs::path& file) {
        std::error_code status;
        return fs::is_regular_file(file, status) && fs::file_size(file, status) > 0;
    };
    if (present(prompts / "test-harmful.txt")) {
        args.insert(args.end(), {"--test-harmful", path_utf8(prompts / "test-harmful.txt")});
        if (present(prompts / "test-benign.txt")) {
            args.insert(args.end(), {"--test-benign", path_utf8(prompts / "test-benign.txt")});
        }
    }
    if (const std::optional<std::string> sets = read_file(prompts / "sets.json")) {
        try {
            const std::string source = Json::parse(*sets).value("source_text", "");
            if (!source.empty()) { args.insert(args.end(), {"--source", source}); }
        } catch (const Json::exception&) {}
    }

    auto log_file = std::make_shared<std::ofstream>(run_dir / "launcher.log", std::ios::app | std::ios::binary);
    auto dirty    = std::make_shared<std::atomic<bool>>(false);
    auto lines    = std::make_shared<Lines>([this, log_file, dirty](std::string line) {
        *log_file << line << '\n';
        log_file->flush();
        {
            std::lock_guard lock(mutex_);
            if (job_) { push_line(job_->log, std::move(line), kJobLogLines); }
        }
        *dirty = true;
    });
    {
        std::lock_guard lock(mutex_);
        job_->log.clear();
    }
    set("running", "starting the ablation tool");
    std::unique_ptr<ChildProcess> child;
    try {
        child = ChildProcess::start(python->executable, tool_args(args), run_dir,
                                    [lines](std::string_view chunk) { lines->feed(chunk); }, tool_env(run.env));
    } catch (const std::exception& failure) {
        hooks_.release_gpu();
        finish_job("failed", failure.what());
        return;
    }
    std::optional<Clock::time_point> asked;
    bool killed = false;
    std::string last_progress;
    const auto read_progress = [&] {
        const std::optional<std::string> text = read_file(run_dir / "progress.json");
        if (!text || *text == last_progress) { return; }
        last_progress = *text;
        try {
            Json progress = Json::parse(*text);
            std::lock_guard lock(mutex_);
            if (job_->state == "running") { job_->message = progress.value("message", job_->message); }
            job_->progress = std::move(progress);
        } catch (const Json::exception&) {} // written whole (a rename), so only a foreign file fails
        *dirty = true;
    };
    while (!child->wait_for(std::chrono::seconds(1))) {
        read_progress();
        if (stop_requested_ && !asked) {
            asked = Clock::now();
            {
                std::lock_guard lock(mutex_);
                job_->state   = "stopping";
                job_->message = "stopping: the tool stops its engines and keeps its finished work";
            }
            *dirty = true;
            if (!child->request_stop()) { child->kill(); }
        }
        bool exiting = false;
        {
            std::lock_guard lock(mutex_);
            exiting = shutting_down_;
        }
        if (!killed && asked && (stop_now_ || Clock::now() - *asked > (exiting ? kShutdownGrace : kStopGrace))) {
            child->kill();
            killed = true;
        }
        if (dirty->exchange(false)) { publish(); }
    }
    const int code = child->exit_code().value_or(-1);
    // The engines it left first: they hold the GPU, and on Linux (each in a session of its own, which ending the
    // tool's process group misses) one may hold the output pipe the tool passed on, whose end the release waits for.
    end_leftover_engines(run_dir);
    child->wait_output(std::chrono::seconds(10));
    // Destroying it closes its job object on Windows: anything else of the tool's still running ends with it.
    child.reset();
    lines->flush();
    read_progress();
    hooks_.release_gpu();

    // The outcome: the exit status, with the tool's own account of it.
    Json progress;
    std::string recent;
    {
        std::lock_guard lock(mutex_);
        progress = job_->progress;
        recent   = tail_lines(job_->log, 4);
    }
    const std::string reported = progress.is_object() ? progress.value("state", "") : "";
    const std::string message  = progress.is_object() ? progress.value("message", "") : "";
    if (code == 0 && (reported == "done" || reported == "no_sidecar")) {
        finish_job(reported, message);
    } else if (asked || code == 130) {
        finish_job("stopped", "stopped; Run again continues it (finished work is kept)");
    } else if (reported == "failed" && !message.empty()) {
        finish_job("failed", message);
    } else {
        finish_job("failed", "the tool ended with exit code " + std::to_string(code) + (recent.empty() ? "" : ":\n" + recent));
    }
}

} // namespace infernix::launcher
