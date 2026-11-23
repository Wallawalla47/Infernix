#pragma once

// A child process whose stdout and stderr arrive, merged, through one pipe. Windows: its own
// process group sharing the launcher's console (so CTRL_BREAK_EVENT reaches it alone), only the
// pipe handles inherited, and a job object that ends it if the launcher dies. POSIX: its own
// process group, stopped with SIGTERM to the group; on Linux SIGKILL when the launcher dies
// (PR_SET_PDEATHSIG, armed from a thread that lives as long as the launcher).

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace infernix::launcher {

struct ProcessUsage {
    std::uint64_t working_set   = 0;
    std::uint64_t private_bytes = 0;
    // Total CPU time used so far (user + kernel), for a usage rate between two readings.
    double cpu_seconds = 0.0;
};

// Environment variables a child gets on top of the launcher's own (a later name replaces one set
// earlier; Windows names are case-insensitive).
using EnvOverlay = std::vector<std::pair<std::string, std::string>>;

class ChildProcess {
public:
    using OutputHandler = std::function<void(std::string_view chunk)>;

    // Starts `executable` with `args` (argv[1..]) in `working_dir`, its environment the launcher's
    // with `env` on top. `on_output` runs on the reader thread for every chunk read. Throws
    // std::runtime_error when the process cannot start.
    static std::unique_ptr<ChildProcess> start(const std::filesystem::path& executable,
                                               const std::vector<std::string>& args,
                                               const std::filesystem::path& working_dir,
                                               OutputHandler on_output, const EnvOverlay& env = {});

    ChildProcess(const ChildProcess&)            = delete;
    ChildProcess& operator=(const ChildProcess&) = delete;
    ~ChildProcess();

    [[nodiscard]] std::uint32_t pid() const noexcept;
    // The exit status once the process ended.
    [[nodiscard]] std::optional<int> exit_code();
    // Waits up to `timeout`; true when the process has ended.
    bool wait_for(std::chrono::milliseconds timeout);
    // Waits up to `timeout` for its output to be read to the end (the pipe closes when the process,
    // and anything that inherited its output, has exited); true when it was. The exit is seen
    // before its last lines are read, so a reason for the exit waits for this.
    bool wait_output(std::chrono::milliseconds timeout);
    // Asks for an orderly stop: CTRL_BREAK_EVENT on Windows, SIGTERM elsewhere. False when it
    // could not be delivered.
    bool request_stop();
    // Ends the process at once.
    void kill();
    [[nodiscard]] std::optional<ProcessUsage> usage() const;

private:
    struct Impl;
    explicit ChildProcess(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

// Runs a short-lived command and returns its merged output and exit status (nullopt when it could
// not start or outlived `timeout`, then it is ended).
struct CaptureResult {
    std::string output;
    int exit_code = 0;
};
[[nodiscard]] std::optional<CaptureResult> run_capture(const std::filesystem::path& executable,
                                                       const std::vector<std::string>& args,
                                                       std::chrono::milliseconds timeout, const EnvOverlay& env = {});

// Ends process `pid` while it still runs a program of this file name ("infernix-serve.exe"; case-insensitive on
// Windows): the leftover of a child's own child, in a process group of its own, after the child was ended. True when
// no such process runs any longer (ended within `wait`, gone, or the id belongs to another program now).
bool end_leftover_process(std::uint32_t pid, std::string_view file_name, std::chrono::milliseconds wait);

// A program found on PATH ("python3"; ".exe" is added on Windows when the name has no extension).
[[nodiscard]] std::optional<std::filesystem::path> find_program(std::string_view name);

// Starts a process that outlives the launcher (no pipe, no job object, its own process group).
bool spawn_detached(const std::filesystem::path& executable, const std::vector<std::string>& args);
#ifdef _WIN32
// The same with its command line given whole (cmd_batch_command_line's, which quoting each argument cannot spell).
bool spawn_detached_command_line(const std::filesystem::path& executable, const std::string& command_line);
#endif

// Opens a URL in the user's browser.
void open_in_browser(const std::string& url);

} // namespace infernix::launcher
