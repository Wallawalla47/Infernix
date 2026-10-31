// infernix-launcher: the web UI and the model-switching OpenAI/Anthropic endpoint in front of
// infernix-serve (docs/launcher.md).

#include "infernix_build_id.h"
#include "launcher/launcher.h"
#include "launcher/store.h"
#include "launcher/util.h"

#include <curl/curl.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <thread>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <csignal>
#include <pthread.h>
#endif

namespace {

using infernix::launcher::Launcher;
using infernix::launcher::LauncherOptions;

std::atomic<Launcher*> g_launcher{nullptr};

void print_usage(const char* program) {
    std::printf(
        "usage: %s [options]\n\n"
        "Starts the Infernix Launcher: a web UI to download, configure and run models, and one\n"
        "OpenAI- and Anthropic-compatible endpoint that loads whichever model a request names.\n\n"
        "  --host H          listen address (default: settings, 127.0.0.1)\n"
        "  --port N          listen port (default: settings, 1234)\n"
        "  --data-dir DIR    the folder of infernix-launcher.json (settings, model\n"
        "                    configurations, downloads) and, unless Settings name another,\n"
        "                    the logs (default: the launcher's own folder, else\n"
        "                    %s when that is read-only)\n"
        "  --serve FILE      the infernix-serve executable (default: beside the launcher)\n"
        "  --no-browser      do not open the web UI in a browser\n"
        "  --ui-dir DIR      serve the web UI from DIR instead of the built-in copy (development)\n"
        "  --version         print the build and exit\n"
        "  --help, -h        show this help and exit\n",
        program, infernix::launcher::path_utf8(infernix::launcher::user_data_dir()).c_str());
}

#ifdef _WIN32
BOOL WINAPI on_console_event(DWORD event) {
    switch (event) {
    case CTRL_C_EVENT:
    case CTRL_BREAK_EVENT:
    case CTRL_CLOSE_EVENT:
    case CTRL_LOGOFF_EVENT:
    case CTRL_SHUTDOWN_EVENT:
        if (Launcher* launcher = g_launcher.load()) {
            std::printf("\nstopping: the loaded model is stopped in order (it saves its caches)...\n");
            std::fflush(stdout);
            launcher->stop();
        }
        return TRUE;
    default: return FALSE;
    }
}
#endif

} // namespace

int main(int argc, char** argv) {
    LauncherOptions options;
    options.version = INFERNIX_BUILD_ID;
    for (int i = 1; i < argc; ++i) { options.restart_args.emplace_back(argv[i]); }
    std::optional<std::filesystem::path> data_dir;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const auto value      = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "infernix-launcher: %s needs a value\n", arg.c_str());
                std::exit(1);
            }
            return argv[++i];
        };
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        }
        if (arg == "--version") {
            std::printf("infernix-launcher %s\n", INFERNIX_BUILD_ID);
            return 0;
        }
        if (arg == "--host") {
            options.host = value();
        } else if (arg == "--port") {
            const std::string text = value();
            try {
                const int port = std::stoi(text);
                if (port < 1 || port > 65535) { throw std::out_of_range("port"); }
                options.port = port;
            } catch (const std::exception&) {
                std::fprintf(stderr, "infernix-launcher: --port takes 1-65535, not %s\n", text.c_str());
                return 1;
            }
        } else if (arg == "--data-dir") {
            // Absolute: the server runs in the data folder, so a relative path would mean another place.
            data_dir = std::filesystem::absolute(infernix::launcher::utf8_path(value()));
        } else if (arg == "--serve") {
            options.serve_executable = std::filesystem::absolute(infernix::launcher::utf8_path(value()));
        } else if (arg == "--ui-dir") {
            options.ui_dir = std::filesystem::absolute(infernix::launcher::utf8_path(value()));
        } else if (arg == "--no-browser") {
            options.open_browser = false;
        } else {
            std::fprintf(stderr, "infernix-launcher: unknown option %s (see --help)\n", arg.c_str());
            return 1;
        }
    }

    bool data_fallback = false;
    if (data_dir) {
        options.data_dir = *data_dir;
    } else {
        const infernix::launcher::DataDirChoice choice = infernix::launcher::choose_data_dir();
        options.data_dir                                = choice.dir;
        data_fallback                                   = choice.fallback;
    }

#ifdef _WIN32
    // The managed server shares this console so an orderly stop (CTRL_BREAK) can reach it; started
    // without one, the launcher makes a hidden one.
    if (GetConsoleWindow() == nullptr && AllocConsole()) { ShowWindow(GetConsoleWindow(), SW_HIDE); }
    SetConsoleOutputCP(CP_UTF8);
    // Inherited by the server: a crash ends it at once instead of waiting on Windows' error
    // dialog, so the watchdog sees the crash.
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX);
#else
    sigset_t signals;
    sigemptyset(&signals);
    sigaddset(&signals, SIGINT);
    sigaddset(&signals, SIGTERM);
    sigaddset(&signals, SIGHUP);
    pthread_sigmask(SIG_BLOCK, &signals, nullptr);
    std::signal(SIGPIPE, SIG_IGN);
#endif
    curl_global_init(CURL_GLOBAL_DEFAULT);

    int status = 0;
    {
        std::unique_ptr<Launcher> launcher;
        try {
            launcher = std::make_unique<Launcher>(options);
        } catch (const std::exception& error) {
            std::fprintf(stderr, "infernix-launcher: %s\n", error.what());
            curl_global_cleanup();
            return 1;
        }
        std::string error;
        if (!launcher->bind(error)) {
            std::fprintf(stderr, "infernix-launcher: %s\n", error.c_str());
            launcher->stop();
            curl_global_cleanup();
            return 1;
        }
        g_launcher = launcher.get();
#ifdef _WIN32
        SetConsoleCtrlHandler(on_console_event, TRUE);
#else
        std::thread([&launcher, signals] {
            int received = 0;
            sigwait(&signals, &received);
            std::printf("\nstopping: the loaded model is stopped in order (it saves its caches)...\n");
            std::fflush(stdout);
            launcher->stop();
        }).detach();
#endif
        const std::string url = launcher->ui_url();
        std::printf("Infernix Launcher %s\n", INFERNIX_BUILD_ID);
        std::printf("  Web UI     %s\n", url.c_str());
        std::printf("  OpenAI     %s/v1\n", url.c_str());
        std::printf("  Anthropic  %s\n", url.c_str());
        std::printf("  Memory     %s\n",
                    infernix::launcher::path_utf8(options.data_dir / infernix::launcher::Store::kMemoryFile).c_str());
        if (data_fallback) {
            std::printf("             (the launcher's own folder is read-only, so its memory is kept here)\n");
        }
        std::printf("  Event log  %s\n", launcher->event_log_file().c_str());
        std::printf("Press Ctrl+C to stop.\n\n");
        std::fflush(stdout);
        launcher->serve();
        launcher->stop();
        g_launcher = nullptr;
    }
    curl_global_cleanup();
    return status;
}
