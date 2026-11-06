#include "launcher/process.h"

#include "launcher/args.h"
#include "launcher/util.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <cwchar>
#include <cwctype>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#include <shellapi.h>
#else
#include <cerrno>
#include <csignal>
#include <cstring>
#include <iterator>
#include <fcntl.h>
#include <fstream>
#include <spawn.h>
#include <sstream>
#include <sys/wait.h>
#include <unistd.h>
extern char** environ;
#endif

namespace infernix::launcher {

namespace {

// Set by a child's reader thread once its output pipe reached the end.
struct OutputEnd {
    std::mutex mutex;
    std::condition_variable cv;
    bool done = false;

    void finish() {
        {
            std::lock_guard lock(mutex);
            done = true;
        }
        cv.notify_all();
    }

    bool wait(std::chrono::milliseconds timeout) {
        std::unique_lock lock(mutex);
        return cv.wait_for(lock, timeout, [this] { return done; });
    }
};

} // namespace

#ifdef _WIN32

namespace {

std::wstring wide(std::string_view text) {
    if (text.empty()) { return {}; }
    const int size = MultiByteToWideChar(CP_UTF8, 0, text.data(), static_cast<int>(text.size()), nullptr, 0);
    std::wstring out(static_cast<std::size_t>(size), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, text.data(), static_cast<int>(text.size()), out.data(), size);
    return out;
}

std::string last_error_text() {
    const DWORD code = GetLastError();
    wchar_t* buffer  = nullptr;
    FormatMessageW(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                   nullptr, code, 0, reinterpret_cast<LPWSTR>(&buffer), 0, nullptr);
    std::string text = buffer != nullptr ? path_utf8(std::filesystem::path(buffer)) : std::string{};
    if (buffer != nullptr) { LocalFree(buffer); }
    return trim(text) + " (" + std::to_string(code) + ")";
}

struct Handle {
    HANDLE value = nullptr;
    Handle() = default;
    explicit Handle(HANDLE h) : value(h) {}
    Handle(const Handle&)            = delete;
    Handle& operator=(const Handle&) = delete;
    ~Handle() { reset(); }
    void reset(HANDLE h = nullptr) {
        if (value != nullptr && value != INVALID_HANDLE_VALUE) { CloseHandle(value); }
        value = h;
    }
    HANDLE release() {
        HANDLE h = value;
        value    = nullptr;
        return h;
    }
};

} // namespace

struct ChildProcess::Impl {
    HANDLE process = nullptr;
    HANDLE job     = nullptr;
    HANDLE pipe    = nullptr;
    DWORD pid      = 0;
    std::thread reader;
    std::shared_ptr<OutputEnd> output_end = std::make_shared<OutputEnd>();

    ~Impl() {
        if (process != nullptr && WaitForSingleObject(process, 0) == WAIT_TIMEOUT) {
            TerminateProcess(process, 1);
            WaitForSingleObject(process, 5000);
        }
        if (pipe != nullptr) { CancelIoEx(pipe, nullptr); }
        if (reader.joinable()) { reader.join(); }
        if (pipe != nullptr) { CloseHandle(pipe); }
        if (process != nullptr) { CloseHandle(process); }
        if (job != nullptr) { CloseHandle(job); }
    }
};

// The launcher's environment block with `env` on top (names compared without case, sorted as
// Windows keeps them; the per-drive "=C:" entries kept first). Empty when there is nothing to add.
std::wstring environment_block(const EnvOverlay& env) {
    if (env.empty()) { return {}; }
    std::vector<std::wstring> hidden;
    std::map<std::wstring, std::pair<std::wstring, std::wstring>> named; // folded name -> (name, value)
    const auto fold = [](std::wstring text) {
        for (wchar_t& c : text) { c = static_cast<wchar_t>(towupper(c)); }
        return text;
    };
    if (wchar_t* block = GetEnvironmentStringsW()) {
        for (const wchar_t* entry = block; *entry != L'\0'; entry += wcslen(entry) + 1) {
            const std::wstring text(entry);
            const std::size_t equals = text.find(L'=', 1);
            if (text.starts_with(L"=") || equals == std::wstring::npos) {
                hidden.push_back(text);
                continue;
            }
            named[fold(text.substr(0, equals))] = {text.substr(0, equals), text.substr(equals + 1)};
        }
        FreeEnvironmentStringsW(block);
    }
    for (const auto& [name, value] : env) { named[fold(wide(name))] = {wide(name), wide(value)}; }
    std::wstring out;
    for (const std::wstring& entry : hidden) { out += entry + L'\0'; }
    for (const auto& [folded, entry] : named) { out += entry.first + L'=' + entry.second + L'\0'; }
    out += L'\0';
    return out;
}

std::unique_ptr<ChildProcess> ChildProcess::start(const std::filesystem::path& executable,
                                                  const std::vector<std::string>& args,
                                                  const std::filesystem::path& working_dir,
                                                  OutputHandler on_output, const EnvOverlay& env) {
    SECURITY_ATTRIBUTES inheritable{sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
    HANDLE read_raw = nullptr;
    HANDLE write_raw = nullptr;
    if (!CreatePipe(&read_raw, &write_raw, &inheritable, 1 << 16)) {
        throw std::runtime_error("cannot create a pipe: " + last_error_text());
    }
    Handle read_end(read_raw);
    Handle write_end(write_raw);
    SetHandleInformation(read_end.value, HANDLE_FLAG_INHERIT, 0);
    Handle null_input(CreateFileW(L"NUL", GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, &inheritable,
                                  OPEN_EXISTING, 0, nullptr));

    // Inherit only the pipe and NUL: never the launcher's listening socket or files.
    SIZE_T list_size = 0;
    InitializeProcThreadAttributeList(nullptr, 1, 0, &list_size);
    std::vector<std::byte> list_storage(list_size);
    auto* list = reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(list_storage.data());
    if (!InitializeProcThreadAttributeList(list, 1, 0, &list_size)) {
        throw std::runtime_error("cannot prepare process attributes: " + last_error_text());
    }
    std::array<HANDLE, 2> inherited = {write_end.value, null_input.value};
    UpdateProcThreadAttribute(list, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST, inherited.data(),
                              sizeof(HANDLE) * inherited.size(), nullptr, nullptr);

    STARTUPINFOEXW startup{};
    startup.StartupInfo.cb         = sizeof(startup);
    startup.StartupInfo.dwFlags    = STARTF_USESTDHANDLES;
    startup.StartupInfo.hStdInput  = null_input.value;
    startup.StartupInfo.hStdOutput = write_end.value;
    startup.StartupInfo.hStdError  = write_end.value;
    startup.lpAttributeList        = list;

    std::string command = windows_quote(path_utf8(executable));
    for (const std::string& arg : args) { command += " " + windows_quote(arg); }
    std::wstring command_wide = wide(command);
    const std::wstring directory = working_dir.wstring();

    // The launcher's console is shared, with or without a window (a program may start the launcher on
    // a console that has none), so CTRL_BREAK_EVENT can reach the child's own group. Only a launcher
    // with no console at all gives the child a hidden console of its own.
    DWORD flags = CREATE_NEW_PROCESS_GROUP | CREATE_UNICODE_ENVIRONMENT | EXTENDED_STARTUPINFO_PRESENT |
                  CREATE_SUSPENDED;
    if (GetConsoleCP() == 0) { flags |= CREATE_NO_WINDOW; }
    std::wstring environment = environment_block(env);
    PROCESS_INFORMATION info{};
    const BOOL created = CreateProcessW(executable.wstring().c_str(), command_wide.data(), nullptr, nullptr,
                                        TRUE, flags, environment.empty() ? nullptr : environment.data(),
                                        directory.empty() ? nullptr : directory.c_str(), &startup.StartupInfo, &info);
    DeleteProcThreadAttributeList(list);
    if (!created) {
        throw std::runtime_error("cannot start " + path_utf8(executable) + ": " + last_error_text());
    }
    write_end.reset();
    null_input.reset();

    auto impl     = std::make_unique<Impl>();
    impl->process = info.hProcess;
    impl->pid     = info.dwProcessId;
    // The job ends the child with the launcher, however the launcher ends. A child it cannot hold
    // is not started: it would outlive a crashed launcher, keeping the GPU and its port.
    impl->job = CreateJobObjectW(nullptr, nullptr);
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
    limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    if (impl->job == nullptr ||
        !SetInformationJobObject(impl->job, JobObjectExtendedLimitInformation, &limits, sizeof(limits)) ||
        !AssignProcessToJobObject(impl->job, info.hProcess)) {
        const std::string reason = last_error_text();
        CloseHandle(info.hThread); // impl ends the suspended process
        throw std::runtime_error("cannot tie " + path_utf8(executable) + " to the launcher (a job object): " + reason);
    }
    ResumeThread(info.hThread);
    CloseHandle(info.hThread);
    impl->pipe   = read_end.release();
    HANDLE pipe  = impl->pipe;
    impl->reader = std::thread([pipe, end = impl->output_end, on_output = std::move(on_output)] {
        std::array<char, 1 << 16> buffer{};
        for (;;) {
            DWORD got = 0;
            if (!ReadFile(pipe, buffer.data(), static_cast<DWORD>(buffer.size()), &got, nullptr) || got == 0) {
                break;
            }
            if (on_output) { on_output(std::string_view(buffer.data(), got)); }
        }
        end->finish();
    });
    return std::unique_ptr<ChildProcess>(new ChildProcess(std::move(impl)));
}

std::uint32_t ChildProcess::pid() const noexcept { return impl_->pid; }

std::optional<int> ChildProcess::exit_code() {
    if (WaitForSingleObject(impl_->process, 0) != WAIT_OBJECT_0) { return std::nullopt; }
    DWORD code = 0;
    GetExitCodeProcess(impl_->process, &code);
    return static_cast<int>(code);
}

bool ChildProcess::wait_for(std::chrono::milliseconds timeout) {
    return WaitForSingleObject(impl_->process, static_cast<DWORD>(timeout.count())) == WAIT_OBJECT_0;
}

bool ChildProcess::wait_output(std::chrono::milliseconds timeout) { return impl_->output_end->wait(timeout); }

bool ChildProcess::request_stop() {
    return GenerateConsoleCtrlEvent(CTRL_BREAK_EVENT, impl_->pid) != 0;
}

void ChildProcess::kill() { TerminateProcess(impl_->process, 1); }

std::optional<ProcessUsage> ChildProcess::usage() const {
    PROCESS_MEMORY_COUNTERS_EX memory{};
    if (!GetProcessMemoryInfo(impl_->process, reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&memory),
                              sizeof(memory))) {
        return std::nullopt;
    }
    ProcessUsage usage;
    usage.working_set   = memory.WorkingSetSize;
    usage.private_bytes = memory.PrivateUsage;
    FILETIME created{}, exited{}, kernel{}, user{};
    if (GetProcessTimes(impl_->process, &created, &exited, &kernel, &user)) {
        const auto ticks = [](const FILETIME& t) {
            return (static_cast<std::uint64_t>(t.dwHighDateTime) << 32) | t.dwLowDateTime;
        };
        usage.cpu_seconds = static_cast<double>(ticks(kernel) + ticks(user)) / 1e7;
    }
    return usage;
}

void open_in_browser(const std::string& url) {
    ShellExecuteW(nullptr, L"open", wide(url).c_str(), nullptr, nullptr, SW_SHOWNORMAL);
}

bool spawn_detached(const std::filesystem::path& executable, const std::vector<std::string>& args) {
    std::string command = windows_quote(path_utf8(executable));
    for (const std::string& arg : args) { command += " " + windows_quote(arg); }
    std::wstring command_wide = wide(command);
    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    PROCESS_INFORMATION info{};
    // Out of any job the launcher runs in, so it outlives the launcher; a job that forbids that
    // gets a plain detached process.
    for (const DWORD extra : {static_cast<DWORD>(CREATE_BREAKAWAY_FROM_JOB), DWORD{0}}) {
        std::wstring line = command_wide;
        if (CreateProcessW(executable.wstring().c_str(), line.data(), nullptr, nullptr, FALSE,
                           CREATE_NEW_PROCESS_GROUP | CREATE_NEW_CONSOLE | extra, nullptr, nullptr, &startup, &info)) {
            CloseHandle(info.hThread);
            CloseHandle(info.hProcess);
            return true;
        }
    }
    return false;
}

#else // POSIX

namespace {

// The launcher blocks SIGINT/SIGTERM/SIGHUP (a sigwait thread takes them) and ignores SIGPIPE;
// a child must not inherit either, or SIGTERM could never stop it.
void init_spawn_attributes(posix_spawnattr_t& attributes, bool own_group) {
    posix_spawnattr_init(&attributes);
    sigset_t none;
    sigemptyset(&none);
    posix_spawnattr_setsigmask(&attributes, &none);
    sigset_t defaults;
    sigemptyset(&defaults);
    for (const int signal : {SIGINT, SIGTERM, SIGHUP, SIGPIPE}) { sigaddset(&defaults, signal); }
    posix_spawnattr_setsigdefault(&attributes, &defaults);
    short flags = POSIX_SPAWN_SETSIGMASK | POSIX_SPAWN_SETSIGDEF;
    if (own_group) {
        flags |= POSIX_SPAWN_SETPGROUP;
        posix_spawnattr_setpgroup(&attributes, 0);
    }
    posix_spawnattr_setflags(&attributes, flags);
}

} // namespace

struct ChildProcess::Impl {
    pid_t pid = -1;
    int pipe  = -1;
    std::thread reader;
    std::shared_ptr<OutputEnd> output_end = std::make_shared<OutputEnd>();
    std::mutex mutex;
    std::optional<int> status;

    std::optional<int> poll() {
        std::lock_guard lock(mutex);
        if (status) { return status; }
        int raw = 0;
        if (waitpid(pid, &raw, WNOHANG) == pid) {
            status = WIFEXITED(raw) ? WEXITSTATUS(raw) : 128 + WTERMSIG(raw);
        }
        return status;
    }

    ~Impl() {
        if (!poll()) {
            ::kill(-pid, SIGKILL);
            int raw = 0;
            waitpid(pid, &raw, 0);
        }
        if (reader.joinable()) { reader.join(); }
        if (pipe >= 0) { close(pipe); }
    }
};

// The launcher's environment with `env` on top, as "NAME=value" strings (empty: use `environ`).
std::vector<std::string> environment_strings(const EnvOverlay& env) {
    std::vector<std::string> out;
    if (env.empty()) { return out; }
    for (char** entry = environ; *entry != nullptr; ++entry) {
        const std::string text(*entry);
        const std::string name = text.substr(0, text.find('='));
        const bool replaced    = std::any_of(env.begin(), env.end(), [&](const auto& pair) { return pair.first == name; });
        if (!replaced) { out.push_back(text); }
    }
    for (const auto& [name, value] : env) { out.push_back(name + "=" + value); }
    return out;
}

std::unique_ptr<ChildProcess> ChildProcess::start(const std::filesystem::path& executable,
                                                  const std::vector<std::string>& args,
                                                  const std::filesystem::path& working_dir,
                                                  OutputHandler on_output, const EnvOverlay& env) {
    // Both ends close on exec, so a child started at the same time (a --help probe) cannot hold
    // this pipe open; dup2 into the child's 1 and 2 clears the flag there.
    int fds[2];
    if (::pipe2(fds, O_CLOEXEC) != 0) { throw std::runtime_error("cannot create a pipe"); }
    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_addopen(&actions, 0, "/dev/null", O_RDONLY, 0);
    posix_spawn_file_actions_adddup2(&actions, fds[1], 1);
    posix_spawn_file_actions_adddup2(&actions, fds[1], 2);
    posix_spawn_file_actions_addclose(&actions, fds[1]);
    if (!working_dir.empty()) { posix_spawn_file_actions_addchdir_np(&actions, working_dir.c_str()); }
    posix_spawnattr_t attributes;
    init_spawn_attributes(attributes, true);

    std::vector<std::string> argv_storage;
    argv_storage.push_back(executable.string());
    argv_storage.insert(argv_storage.end(), args.begin(), args.end());
    std::vector<char*> argv;
    for (std::string& arg : argv_storage) { argv.push_back(arg.data()); }
    argv.push_back(nullptr);
    std::vector<std::string> env_storage = environment_strings(env);
    std::vector<char*> envp;
    for (std::string& entry : env_storage) { envp.push_back(entry.data()); }
    envp.push_back(nullptr);
    pid_t pid       = -1;
    const int error = posix_spawn(&pid, executable.c_str(), &actions, &attributes, argv.data(),
                                  env_storage.empty() ? environ : envp.data());
    posix_spawn_file_actions_destroy(&actions);
    posix_spawnattr_destroy(&attributes);
    close(fds[1]);
    if (error != 0) {
        close(fds[0]);
        throw std::runtime_error("cannot start " + executable.string() + ": " + std::strerror(error));
    }
    auto impl  = std::make_unique<Impl>();
    impl->pid  = pid;
    impl->pipe = fds[0];
    const int pipe_fd = fds[0];
    impl->reader = std::thread([pipe_fd, end = impl->output_end, on_output = std::move(on_output)] {
        std::array<char, 1 << 16> buffer{};
        for (;;) {
            const ssize_t got = read(pipe_fd, buffer.data(), buffer.size());
            if (got < 0 && errno == EINTR) { continue; }
            if (got <= 0) { break; }
            if (on_output) { on_output(std::string_view(buffer.data(), static_cast<std::size_t>(got))); }
        }
        end->finish();
    });
    return std::unique_ptr<ChildProcess>(new ChildProcess(std::move(impl)));
}

std::uint32_t ChildProcess::pid() const noexcept { return static_cast<std::uint32_t>(impl_->pid); }

std::optional<int> ChildProcess::exit_code() { return impl_->poll(); }

bool ChildProcess::wait_output(std::chrono::milliseconds timeout) { return impl_->output_end->wait(timeout); }

bool ChildProcess::wait_for(std::chrono::milliseconds timeout) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    for (;;) {
        if (impl_->poll()) { return true; }
        if (std::chrono::steady_clock::now() >= deadline) { return false; }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
}

bool ChildProcess::request_stop() { return ::kill(-impl_->pid, SIGTERM) == 0; }

void ChildProcess::kill() { ::kill(-impl_->pid, SIGKILL); }

std::optional<ProcessUsage> ChildProcess::usage() const {
    std::ifstream status("/proc/" + std::to_string(impl_->pid) + "/status");
    if (!status) { return std::nullopt; }
    ProcessUsage usage;
    std::string line;
    while (std::getline(status, line)) {
        std::istringstream fields(line);
        std::string key;
        std::uint64_t kib = 0;
        fields >> key >> kib;
        if (key == "VmRSS:") { usage.working_set = kib * 1024; }
        if (key == "RssAnon:") { usage.private_bytes = kib * 1024; }
    }
    std::ifstream stat("/proc/" + std::to_string(impl_->pid) + "/stat");
    std::string text((std::istreambuf_iterator<char>(stat)), std::istreambuf_iterator<char>());
    const std::size_t close = text.rfind(')');
    if (close != std::string::npos) {
        std::istringstream fields(text.substr(close + 2));
        std::string field;
        std::uint64_t utime = 0, stime = 0;
        for (int index = 3; fields >> field; ++index) {
            if (index == 14) { utime = std::stoull(field); }
            if (index == 15) {
                stime = std::stoull(field);
                break;
            }
        }
        usage.cpu_seconds = static_cast<double>(utime + stime) / static_cast<double>(sysconf(_SC_CLK_TCK));
    }
    return usage;
}

void open_in_browser(const std::string& url) {
#ifdef __APPLE__
    const char* opener = "open";
#else
    const char* opener = "xdg-open";
#endif
    std::string program = opener;
    std::string target  = url;
    char* argv[]        = {program.data(), target.data(), nullptr};
    pid_t pid           = -1;
    posix_spawnattr_t attributes;
    init_spawn_attributes(attributes, false);
    const int error = posix_spawnp(&pid, opener, nullptr, &attributes, argv, environ);
    posix_spawnattr_destroy(&attributes);
    if (error == 0) {
        std::thread([pid] {
            int status = 0;
            waitpid(pid, &status, 0);
        }).detach();
    }
}

bool spawn_detached(const std::filesystem::path& executable, const std::vector<std::string>& args) {
    std::vector<std::string> storage = {executable.string()};
    storage.insert(storage.end(), args.begin(), args.end());
    std::vector<char*> argv;
    for (std::string& arg : storage) { argv.push_back(arg.data()); }
    argv.push_back(nullptr);
    posix_spawnattr_t attributes;
    init_spawn_attributes(attributes, true);
    pid_t pid       = -1;
    const int error = posix_spawn(&pid, executable.c_str(), nullptr, &attributes, argv.data(), environ);
    posix_spawnattr_destroy(&attributes);
    if (error != 0) { return false; }
    std::thread([pid] {
        int status = 0;
        waitpid(pid, &status, 0);
    }).detach();
    return true;
}

#endif

ChildProcess::ChildProcess(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
ChildProcess::~ChildProcess() = default;

std::optional<CaptureResult> run_capture(const std::filesystem::path& executable,
                                         const std::vector<std::string>& args, std::chrono::milliseconds timeout,
                                         const EnvOverlay& env) {
    auto output = std::make_shared<std::string>();
    auto mutex  = std::make_shared<std::mutex>();
    std::unique_ptr<ChildProcess> child;
    try {
        child = ChildProcess::start(
            executable, args, executable.parent_path(),
            [output, mutex](std::string_view chunk) {
                std::lock_guard lock(*mutex);
                output->append(chunk);
            },
            env);
    } catch (const std::exception&) { return std::nullopt; }
    if (!child->wait_for(timeout)) {
        child->kill();
        child.reset();
        return std::nullopt;
    }
    const int code = child->exit_code().value_or(-1);
    child.reset(); // joins the reader, so the output is complete
    std::lock_guard lock(*mutex);
    return CaptureResult{*output, code};
}

} // namespace infernix::launcher
