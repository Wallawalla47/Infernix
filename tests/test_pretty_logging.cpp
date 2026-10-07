#include "media/decode/decode.h"
#include "product/logging/engine_diagnostics.h"
#include "product/logging/logging.h"
#include "product/logging/startup_log.h"
#include "runtime/engine/diagnostics.h"

#include <spdlog/logger.h>

#ifdef _WIN32
#    include <fcntl.h>
#    include <io.h>
#else
#    include <unistd.h>
#endif

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <regex>
#include <stdexcept>
#include <string>

namespace {

#ifdef _WIN32
constexpr int STDERR_FILENO = 2;
// Text-mode pipe: the stdio layer emits \r\n for the stderr writes, and _read normalizes it
// back to \n so the captured bytes match the POSIX expectation.
inline int port_pipe(int fds[2])   { return _pipe(fds, 0, _O_TEXT); }
inline int port_dup(int fd)        { return _dup(fd); }
inline int port_dup2(int a, int b) { return _dup2(a, b); }
inline int port_close(int fd)      { return _close(fd); }
inline int port_read(int fd, void* buffer, unsigned long count) {
    return _read(fd, buffer, static_cast<unsigned int>(count));
}
#else
inline int port_pipe(int fds[2])   { return ::pipe(fds); }
inline int port_dup(int fd)        { return ::dup(fd); }
inline int port_dup2(int a, int b) { return ::dup2(a, b); }
inline int port_close(int fd)      { return ::close(fd); }
inline int port_read(int fd, void* buffer, std::size_t count) {
    return static_cast<int>(::read(fd, buffer, count));
}
#endif

class StderrCapture {
public:
    StderrCapture() {
        if (port_pipe(pipe_) != 0) { throw std::runtime_error(std::strerror(errno)); }
        saved_ = port_dup(STDERR_FILENO);
        if (saved_ < 0 || port_dup2(pipe_[1], STDERR_FILENO) < 0) {
            throw std::runtime_error(std::strerror(errno));
        }
        port_close(pipe_[1]);
        pipe_[1] = -1;
    }

    ~StderrCapture() {
        if (saved_ >= 0) {
            (void)port_dup2(saved_, STDERR_FILENO);
            port_close(saved_);
        }
        if (pipe_[0] >= 0) { port_close(pipe_[0]); }
    }

    std::string finish() {
        std::fflush(stderr);
        if (port_dup2(saved_, STDERR_FILENO) < 0) {
            throw std::runtime_error(std::strerror(errno));
        }
        port_close(saved_);
        saved_ = -1;

        std::string output;
        std::array<char, 4096> buffer{};
        for (;;) {
            const int count = port_read(pipe_[0], buffer.data(), buffer.size());
            if (count == 0) { break; }
            if (count < 0) {
                if (errno == EINTR) { continue; }
                throw std::runtime_error(std::strerror(errno));
            }
            output.append(buffer.data(), static_cast<std::size_t>(count));
        }
        port_close(pipe_[0]);
        pipe_[0] = -1;
        return output;
    }

private:
    int pipe_[2]{-1, -1};
    int saved_ = -1;
};

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

std::size_t line_count(std::string_view value) {
    return static_cast<std::size_t>(std::count(value.begin(), value.end(), '\n'));
}

} // namespace

int main() {
    int failures = 0;
    std::string service_output;
    {
        StderrCapture capture;
        {
            infernix::product::LoggingRuntime logging(
                {.logger_name  = "infernix-serve",
                 .color        = infernix::product::LogColorMode::Auto,
                 .presentation = infernix::product::LogPresentation::Service});
            logging.logger()->info("throughput | sample");
            logging.flush();
        }
        service_output = capture.finish();
    }
    failures += check(
        std::regex_match(
            service_output,
            std::regex(
                R"(^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{3}  INFO  throughput \| sample\n$)")),
        "service pretty prefix mismatch");
    failures += check(service_output.find("infernix-serve") == std::string::npos,
                      "service pretty output repeated the executable name");
    failures += check(service_output.find("\x1b[") == std::string::npos,
                      "redirected service output contains ANSI escapes");

    // Always mode on a redirected stderr must still deliver the line (plain),
    // not drop it: the colour sink's WriteConsoleA path silently fails on a
    // non-console handle.
    std::string always_redirect_output;
    {
        StderrCapture capture;
        {
            infernix::product::LoggingRuntime logging(
                {.logger_name  = "infernix-serve",
                 .color        = infernix::product::LogColorMode::Always,
                 .presentation = infernix::product::LogPresentation::Service});
            logging.logger()->info("throughput | sample");
            logging.flush();
        }
        always_redirect_output = capture.finish();
    }
    failures += check(
        std::regex_match(
            always_redirect_output,
            std::regex(
                R"(^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{3}  INFO  throughput \| sample\n$)")),
        "Always mode dropped redirected service output");
    failures += check(always_redirect_output.find("\x1b[") == std::string::npos,
                      "redirected Always output contains ANSI escapes");

    std::string startup_output;
    {
        StderrCapture capture;
        {
            infernix::product::LoggingRuntime logging(
                {.logger_name  = "infernix-serve",
                 .color        = infernix::product::LogColorMode::Never,
                 .presentation = infernix::product::LogPresentation::Service});
            infernix::product::StartupLogRenderer startup(logging);
            infernix::StartupObserver observer = startup.observer();
            observer.callback({.phase  = infernix::StartupPhase::EngineStartup,
                               .status = infernix::StartupStatus::Begin});
            observer.callback({.phase  = infernix::StartupPhase::CudaInitialize,
                               .status = infernix::StartupStatus::Begin});
            observer.callback({.phase      = infernix::StartupPhase::CudaInitialize,
                               .status     = infernix::StartupStatus::Complete,
                               .elapsed_ns = 1'000'000'000});
            observer.callback({.phase         = infernix::StartupPhase::WeightsMaterialize,
                               .status        = infernix::StartupStatus::Begin,
                               .progress_unit = infernix::StartupProgressUnit::Bytes,
                               .total         = 16ULL << 30});
            observer.callback({.phase         = infernix::StartupPhase::WeightsMaterialize,
                               .status        = infernix::StartupStatus::Complete,
                               .progress_unit = infernix::StartupProgressUnit::Bytes,
                               .current       = 16ULL << 30,
                               .total         = 16ULL << 30,
                               .elapsed_ns    = 2'000'000'000});
            observer.callback({.phase         = infernix::StartupPhase::PrefixCacheLoad,
                               .status        = infernix::StartupStatus::Begin,
                               .progress_unit = infernix::StartupProgressUnit::Bytes,
                               .total         = 24ULL << 30});
            observer.callback({.phase         = infernix::StartupPhase::PrefixCacheLoad,
                               .status        = infernix::StartupStatus::Complete,
                               .progress_unit = infernix::StartupProgressUnit::Bytes,
                               .current       = 24ULL << 30,
                               .total         = 24ULL << 30,
                               .elapsed_ns    = 4'000'000'000});
            observer.callback({.phase      = infernix::StartupPhase::EngineStartup,
                               .status     = infernix::StartupStatus::Complete,
                               .elapsed_ns = 3'000'000'000});
            startup.engine_ready({.model_name           = "qwen3.6-27b",
                                  .cuda_sync_mode       = "blocking",
                                  .weight_formats       = {"q4_g64_fp16", "q8_g32_fp16"},
                                  .host_to_device_bytes = 16ULL << 30});
            logging.flush();
        }
        startup_output = capture.finish();
    }
    failures += check(
        line_count(startup_output) == 6 &&
            startup_output.find("starting engine") != std::string::npos &&
            startup_output.find("loading weights | 16.0 GiB") != std::string::npos &&
            startup_output.find("weights ready | 16.0 GiB | 2.0s | 8.00 GiB/s") !=
                std::string::npos &&
            startup_output.find("loading prefix cache | 24.0 GiB") != std::string::npos &&
            startup_output.find("prefix cache read | 24.0 GiB | 4.0s | 6.00 GiB/s") !=
                std::string::npos &&
            startup_output.find("engine ready | qwen3.6-27b | total 3.0s | weights 16.0 GiB") !=
                std::string::npos &&
            startup_output.find("CUDA sync blocking") != std::string::npos &&
            startup_output.find("CUDA initialized") == std::string::npos,
        "normal startup pretty output is noisy or incomplete");

    // A Host tier smaller than the saved file warns with what was kept and what the file needs.
    std::string partial_output;
    {
        StderrCapture capture;
        {
            infernix::product::LoggingRuntime logging(
                {.logger_name  = "infernix-serve",
                 .color        = infernix::product::LogColorMode::Never,
                 .presentation = infernix::product::LogPresentation::Service});
            infernix::product::StartupLogRenderer startup(logging);
            infernix::LoadSummary load;
            load.model_name     = "qwen3.6-27b";
            load.cuda_sync_mode = "blocking";
            load.prefix_cache   = {.attempted           = true,
                                   .restored            = true,
                                   .blocks              = 9000,
                                   .snapshots           = 60,
                                   .bytes               = 30ULL << 30,
                                   .seconds             = 7.0,
                                   .saved_blocks        = 15807,
                                   .saved_snapshots     = 98,
                                   .required_host_bytes = 50ULL << 30,
                                   .host_bytes          = 31ULL << 30};
            startup.engine_ready(load);
            logging.flush();
        }
        partial_output = capture.finish();
    }
    failures += check(
        partial_output.find("  WARN  prefix cache partly restored | 60 of 98 snapshots | 9,000 of "
                            "15,807 blocks | 30.0 GiB | 7.0s | the file needs 50.0 GiB of Host "
                            "tier; --host-context-mib gives 31.0 GiB") != std::string::npos &&
            partial_output.find("prefix cache restored") == std::string::npos,
        "a partial prefix cache restore is not reported as a warning");

    // A Host tier that keeps no snapshot restores nothing resumable: the warning says so rather
    // than claiming the most valuable snapshots were kept.
    std::string none_kept_output;
    {
        StderrCapture capture;
        {
            infernix::product::LoggingRuntime logging(
                {.logger_name  = "infernix-serve",
                 .color        = infernix::product::LogColorMode::Never,
                 .presentation = infernix::product::LogPresentation::Service});
            infernix::product::StartupLogRenderer startup(logging);
            infernix::LoadSummary load;
            load.model_name     = "qwen3.6-27b";
            load.cuda_sync_mode = "blocking";
            load.prefix_cache   = {.attempted           = true,
                                   .restored            = true,
                                   .blocks              = 0,
                                   .snapshots           = 0,
                                   .bytes               = 0,
                                   .seconds             = 0.1,
                                   .saved_blocks        = 15807,
                                   .saved_snapshots     = 98,
                                   .required_host_bytes = 50ULL << 30,
                                   .host_bytes          = 1ULL << 30};
            startup.engine_ready(load);
            logging.flush();
        }
        none_kept_output = capture.finish();
    }
    failures +=
        check(none_kept_output.find(
                  "  WARN  prefix cache not restored | none of the file's 98 snapshots "
                  "fits: it needs 50.0 GiB of Host tier and --host-context-mib gives "
                  "1.00 GiB | the save at shutdown replaces the file") != std::string::npos &&
                  none_kept_output.find("partly restored") == std::string::npos &&
                  none_kept_output.find("most valuable") == std::string::npos,
              "a restore that kept no snapshot claims its most valuable snapshots were kept");

    std::string tool_output;
    {
        StderrCapture capture;
        {
            infernix::product::LoggingRuntime logging(
                {.logger_name  = "infernix",
                 .color        = infernix::product::LogColorMode::Never,
                 .presentation = infernix::product::LogPresentation::Tool});
            logging.logger()->info("engine ready");
            logging.logger()->error("failed");
            logging.flush();
        }
        tool_output = capture.finish();
    }
    failures +=
        check(tool_output == "engine ready\nerror: failed\n", "tool pretty prefix mismatch");

    // Engine diagnostics are ordinary records at the level the Engine assigned: a lease extension
    // is debug, shown only when the verbosity is raised, and a truncating lease is a warning.
    for (const infernix::product::LogLevel level :
         {infernix::product::LogLevel::Info, infernix::product::LogLevel::Debug}) {
        std::string output;
        {
            StderrCapture capture;
            {
                infernix::product::LoggingRuntime logging(
                    {.logger_name  = "infernix",
                     .level        = level,
                     .color        = infernix::product::LogColorMode::Never,
                     .presentation = infernix::product::LogPresentation::Tool});
                const infernix::DiagnosticObserver observer =
                    infernix::product::engine_diagnostic_observer(logging.logger());
                infernix::runtime::publish_diagnostic(observer, infernix::DiagnosticLevel::Debug,
                                                    "Device KV lease of lane %u extended", 0U);
                infernix::runtime::publish_diagnostic(observer, infernix::DiagnosticLevel::Warning,
                                                    "Device KV lease of lane %u cannot grow", 1U);
                logging.flush();
            }
            output = capture.finish();
        }
        const bool debug_shown =
            output.find("engine | Device KV lease of lane 0 extended") != std::string::npos;
        failures += check(debug_shown == (level == infernix::product::LogLevel::Debug),
                          "a debug Engine diagnostic must follow the log level");
        failures += check(output.find("engine | Device KV lease of lane 1 cannot grow") !=
                              std::string::npos,
                          "a warning Engine diagnostic must show at the default level");
    }

    // FFmpeg's media-decoding lines are records of the runtime's logger rather than direct stderr
    // writes that would land inside a terminal footer: an FFmpeg error is a warning and its
    // warnings are debug detail. The runtime restores FFmpeg's previous route when it ends.
    namespace decode = infernix::media::decode;
    for (const infernix::product::LogLevel level :
         {infernix::product::LogLevel::Info, infernix::product::LogLevel::Debug}) {
        std::string output;
        bool routed = false;
        {
            StderrCapture capture;
            {
                infernix::product::LoggingRuntime logging(
                    {.logger_name  = "infernix",
                     .level        = level,
                     .color        = infernix::product::LogColorMode::Never,
                     .presentation = infernix::product::LogPresentation::Tool});
                const decode::LibraryLogHandler route = decode::set_library_log_handler({});
                (void)decode::set_library_log_handler(route);
                routed = static_cast<bool>(route);
                if (routed) {
                    route({.severity = decode::LibraryLogSeverity::Error,
                           .source   = "mjpeg",
                           .message  = "overread 8\x1b[2J"});
                    route({.severity = decode::LibraryLogSeverity::Warning,
                           .source   = "swscaler",
                           .message  = "deprecated pixel format used"});
                }
                logging.flush();
            }
            output = capture.finish();
        }
        failures += check(routed, "the logging runtime did not route FFmpeg's log");
        failures +=
            check(output.find("warning: media | mjpeg: overread 8?[2J\n") != std::string::npos,
                  "an FFmpeg error must be a sanitized warning record");
        const bool debug_shown =
            output.find("debug: media | swscaler: deprecated pixel format used\n") !=
            std::string::npos;
        failures += check(debug_shown == (level == infernix::product::LogLevel::Debug),
                          "an FFmpeg warning must be a debug record");
    }
    failures += check(!decode::set_library_log_handler({}),
                      "the logging runtime left its FFmpeg route installed after it ended");
    return failures == 0 ? 0 : 1;
}
