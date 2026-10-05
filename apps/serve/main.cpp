#include "ninfer_build_id.h"

#include "product/logging/logging.h"
#include "product/logging/engine_diagnostics.h"
#include "product/logging/startup_log.h"
#include "serve/operational_log.h"
#include "serve/generation_service.h"
#include "serve/http_server.h"
#include "serve/serve_options.h"
#include "serve/stop_control.h"

#include <spdlog/logger.h>

#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <utility>

#ifdef _WIN32
#    include <windows.h>
#else
#    include <pthread.h>
#endif

namespace {

using ninfer::serve::StopControl;
using ninfer::serve::StopEvent;

// Routes console and signal events to the StopControl and withdraws an unconfirmed Ctrl+C once
// its window passes. Events arrive on a thread of their own (the Windows console-control thread or
// the POSIX signal thread), never inside a signal handler, so handling may lock and log.
class StopEvents {
public:
    explicit StopEvents(StopControl& control) : control_(control) {
        timer_ = std::thread([this] { run_timer(); });
        std::lock_guard lock(target_mutex());
        target() = this;
    }

    ~StopEvents() {
        {
            std::lock_guard lock(target_mutex());
            target() = nullptr;
        }
        {
            std::lock_guard lock(mutex_);
            done_ = true;
        }
        wake_.notify_all();
        timer_.join();
        control_.finish();
    }

    StopEvents(const StopEvents&)            = delete;
    StopEvents& operator=(const StopEvents&) = delete;

    // Returns false when nothing handles `event`: the caller then applies its default action.
    static bool dispatch(StopEvent event) {
        std::lock_guard lock(target_mutex());
        StopEvents* const events = target();
        if (events == nullptr) { return false; }
        const bool handled = events->control_.handle(event, StopControl::Clock::now());
        {
            std::lock_guard timer_lock(events->mutex_);
            events->changed_ = true;
        }
        events->wake_.notify_all();
        return handled;
    }

private:
    // Never destroyed: an event may still arrive while the process exits.
    static std::mutex& target_mutex() {
        static auto* const mutex = new std::mutex();
        return *mutex;
    }

    static StopEvents*& target() {
        static StopEvents* events = nullptr;
        return events;
    }

    void run_timer() {
        std::unique_lock lock(mutex_);
        while (!done_) {
            changed_ = false;
            lock.unlock();
            const std::optional<StopControl::Clock::time_point> deadline =
                control_.expire(StopControl::Clock::now());
            lock.lock();
            const auto woken = [this] { return done_ || changed_; };
            if (deadline) {
                // expire() withdraws only after the deadline, so wake just past it.
                wake_.wait_until(lock, *deadline + std::chrono::milliseconds(1), woken);
            } else {
                wake_.wait(lock, woken);
            }
        }
    }

    StopControl& control_;
    std::mutex mutex_;
    std::condition_variable wake_;
    bool done_    = false;
    bool changed_ = false;
    std::thread timer_;
};

class ServingScope {
public:
    ServingScope(StopControl& control, ninfer::serve::HttpServer& server) : control_(control) {
        control_.serve([&server] { server.stop(); });
    }

    ~ServingScope() { control_.end_serving(); }

    ServingScope(const ServingScope&)            = delete;
    ServingScope& operator=(const ServingScope&) = delete;

private:
    StopControl& control_;
};

#ifdef _WIN32
// A Ctrl+C with console text selected copies it and never reaches the process. Closing the console
// window terminates the process as soon as this handler returns, so the close blocks here while
// the stop runs; Windows ends the process anyway about 5 seconds after the close, and an
// unfinished prefix-cache save leaves the previous file in place.
BOOL WINAPI handle_console_event(DWORD event) {
    switch (event) {
    case CTRL_C_EVENT:
        return StopEvents::dispatch(StopEvent::Interrupt) ? TRUE : FALSE;
    case CTRL_BREAK_EVENT:
        return StopEvents::dispatch(StopEvent::Terminate) ? TRUE : FALSE;
    case CTRL_CLOSE_EVENT:
        if (!StopEvents::dispatch(StopEvent::Terminate)) { return FALSE; }
        Sleep(INFINITE);
        return TRUE;
    default:
        return FALSE;
    }
}

void install_stop_handlers() { SetConsoleCtrlHandler(handle_console_event, TRUE); }
#else
// SIGINT and SIGTERM are blocked before any other thread starts, so every thread inherits the
// mask and only this thread receives them, outside signal-handler context.
void install_stop_handlers() {
    sigset_t signals;
    sigemptyset(&signals);
    sigaddset(&signals, SIGINT);
    sigaddset(&signals, SIGTERM);
    pthread_sigmask(SIG_BLOCK, &signals, nullptr);
    std::thread([signals] {
        for (;;) {
            int signal = 0;
            if (sigwait(&signals, &signal) != 0) { continue; }
            const StopEvent event = signal == SIGINT ? StopEvent::Interrupt : StopEvent::Terminate;
            if (StopEvents::dispatch(event)) { continue; }
            // Nothing is served yet or any more: the default action ends the process.
            std::signal(signal, SIG_DFL);
            sigset_t only;
            sigemptyset(&only);
            sigaddset(&only, signal);
            pthread_sigmask(SIG_UNBLOCK, &only, nullptr);
            std::raise(signal);
        }
    }).detach();
}
#endif

// Identity of this exact binary for the persisted prefix cache: the build id alone repeats for
// every uncommitted build, so the executable's size and modification time are included and any
// rebuild invalidates a saved cache whose bytes it may compute differently.
std::string binary_identity(const char* argv0) {
    std::string out;
#ifdef NINFER_BUILD_ID
    out = NINFER_BUILD_ID;
#endif
    std::error_code error;
#ifdef _WIN32
    // argv[0] is whatever the shell typed, which for a PATH launch is not a path to this file.
    (void)argv0;
    std::wstring module(MAX_PATH, L'\0');
    DWORD length = 0;
    while ((length = GetModuleFileNameW(nullptr, module.data(),
                                        static_cast<DWORD>(module.size()))) == module.size()) {
        module.resize(module.size() * 2);
    }
    module.resize(length);
    const std::filesystem::path self(module);
#else
    std::filesystem::path self = std::filesystem::read_symlink("/proc/self/exe", error);
    if (error) { self = std::filesystem::absolute(argv0, error); }
#endif
    const auto size = std::filesystem::file_size(self, error);
    if (!error) { out += ";size=" + std::to_string(size); }
    const auto time = std::filesystem::last_write_time(self, error);
    if (!error) { out += ";mtime=" + std::to_string(time.time_since_epoch().count()); }
    return out;
}

} // namespace

int main(int argc, char** argv) {
    ninfer::serve::ServeOptions options;
    try {
        options = ninfer::serve::parse_serve_options(argc, argv);
    } catch (const std::invalid_argument& exception) {
        std::cerr << "ninfer-serve: " << exception.what() << '\n';
        std::cerr << ninfer::serve::serve_usage_text(argv[0]);
        return 1;
    } catch (const std::exception& exception) {
        std::cerr << "ninfer-serve: " << exception.what() << '\n';
        return 1;
    }
    if (options.help_requested) {
        std::cout << ninfer::serve::serve_usage_text(argv[0]);
        return 0;
    }
    if (!options.context_cache.hybrid.persistent_file.empty()) {
        options.context_cache.hybrid.persistent_identity = binary_identity(argv[0]);
    }
    install_stop_handlers();
    // Token/level colouring is opt-in (--log-colours on); the default keeps the log plain.
    ninfer::serve::set_operational_log_colours(options.log_colours);

    ninfer::product::LoggingOptions logging_options;
    logging_options.logger_name  = "ninfer-serve";
    logging_options.level        = options.log_level;
    logging_options.presentation = ninfer::product::LogPresentation::Service;
    logging_options.color        = options.log_colours ? ninfer::product::LogColorMode::Always
                                                       : ninfer::product::LogColorMode::Never;
    ninfer::product::LoggingRuntime logging(logging_options);
    const std::shared_ptr<spdlog::logger> logger = logging.logger();
    ninfer::product::StartupLogRenderer startup_log(logging);
    ninfer::serve::OperationalLog operational_log(logger);
    // Stop prompts use the transient bottom line beneath the statistics panel; output without one
    // (redirected, or not a terminal) logs them instead.
    const std::shared_ptr<ninfer::product::TerminalProgress> console_line =
        logging.terminal_progress();
    // Shares its state with the Engine's copy of the options, and outlives the Engine.
    const ninfer::PrefixCacheSaveControl save_control =
        options.context_cache.hybrid.persistent_save;
    StopControl stop_control(
        !options.context_cache.hybrid.persistent_file.empty(),
        {.show =
             [&](const ninfer::serve::StopConsoleLine& line) {
                 if (console_line->enabled()) {
                     if (line.text.empty()) {
                         console_line->clear();
                     } else {
                         console_line->update(line.text);
                     }
                 } else if (line.prompt) {
                     operational_log.write({.severity = ninfer::serve::OperationalSeverity::Warning,
                                            .message  = line.text});
                 }
             },
         .record =
             [&](const ninfer::serve::OperationalRecord& record) { operational_log.write(record); },
         // The writer checks between slabs of a few MiB, so it lets go within milliseconds unless
         // the disk stalls.
         .abandon_save = [save_control] { return save_control.abandon(std::chrono::seconds(2)); },
         .exit_now =
             [&] {
                 logging.flush();
                 std::_Exit(130);
             }});
    const StopEvents stop_events(stop_control);
    bool serving = false;

    try {
        ninfer::serve::HttpServer server(options, logger, logging.terminal_panel());
        if (!server.bind()) {
            operational_log.bind_failure(options.host, options.port);
            return 1;
        }

#ifdef NINFER_BUILD_ID
        logger->info("build {}", NINFER_BUILD_ID);
#endif
        ninfer::serve::GenerationService service(
            options, startup_log.observer(), ninfer::product::engine_diagnostic_observer(logger));
        startup_log.engine_ready(service.load_summary());
        operational_log.engine_capacity(service);

        using Clock                            = std::chrono::steady_clock;
        const Clock::time_point warmup_started = Clock::now();
        operational_log.warmup_started();
        try {
            service.warmup();
        } catch (const std::exception& exception) {
            const double seconds =
                std::chrono::duration<double>(Clock::now() - warmup_started).count();
            operational_log.warmup_failure(seconds, exception.what());
            return 1;
        }
        operational_log.warmup_complete(
            std::chrono::duration<double>(Clock::now() - warmup_started).count());
        server.attach(service);
        const ServingScope serving_scope(stop_control, server);

        serving = true;
        operational_log.server_ready(options.host, options.port, server.public_model_id(),
                                     !options.api_key.empty());

        const bool ok = server.listen();
        if (!ok) {
            operational_log.listen_failure(options.host, options.port);
            return 1;
        }
        operational_log.server_stopped();
        return 0;
    } catch (const std::exception& exception) {
        operational_log.server_failure(serving, exception.what());
        return 1;
    }
}
