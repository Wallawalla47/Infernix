#include "serve/stop_control.h"

#include "infernix/types.h"

#include <atomic>
#include <chrono>
#include <iostream>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {

using namespace infernix::serve;
using Clock   = StopControl::Clock;
using Abandon = infernix::PrefixCacheSaveControl::Abandon;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

bool contains(std::string_view text, std::string_view needle) {
    return text.find(needle) != std::string_view::npos;
}

// Records what the control asked the console, the Engine's save and the process to do.
struct Observed {
    std::vector<StopConsoleLine> lines;
    std::vector<OperationalRecord> records;
    int stops              = 0;
    int abandons           = 0;
    int exits              = 0;
    Abandon abandon_result = Abandon::Unsaved;

    StopControlActions actions() {
        return {.show   = [this](const StopConsoleLine& line) { lines.push_back(line); },
                .record = [this](const OperationalRecord& record) { records.push_back(record); },
                .abandon_save =
                    [this] {
                        ++abandons;
                        return abandon_result;
                    },
                .exit_now = [this] { ++exits; }};
    }

    [[nodiscard]] std::string line() const { return lines.empty() ? "" : lines.back().text; }
};

const Clock::time_point t0        = Clock::time_point{} + std::chrono::hours(1);
constexpr Clock::duration kWindow = StopControl::kConfirmWindow;
constexpr Clock::duration kTick   = std::chrono::milliseconds(1);

} // namespace

int main() {
    int failures = 0;

    {
        // Before serving, events keep their default action (a Ctrl+C during loading ends it).
        Observed seen;
        StopControl control(true, seen.actions());
        failures += check(!control.handle(StopEvent::Interrupt, t0) &&
                              !control.handle(StopEvent::Terminate, t0) && seen.lines.empty(),
                          "an inactive control must leave events to their default action");
    }

    {
        // With a prefix cache file: one Ctrl+C only asks; a second inside the window stops.
        Observed seen;
        StopControl control(true, seen.actions());
        control.serve([&] { ++seen.stops; });
        failures += check(control.handle(StopEvent::Interrupt, t0) && seen.stops == 0,
                          "a single Ctrl+C must not stop the server");
        failures += check(seen.lines.size() == 1 && seen.lines.back().prompt &&
                              seen.line() == "Press Ctrl+C again within 5 s to save the prefix "
                                             "cache and close",
                          "the first Ctrl+C must prompt to save the prefix cache and close");
        failures += check(control.expire(t0 + kWindow) == t0 + kWindow && seen.lines.size() == 1,
                          "the prompt must stay until its window has passed");
        failures += check(!control.expire(t0 + kWindow + kTick) && seen.line().empty(),
                          "an unanswered prompt must be withdrawn after its window");

        // After the window a Ctrl+C asks again; the second press on the window's edge still counts.
        const Clock::time_point t1 = t0 + kWindow + kTick;
        control.handle(StopEvent::Interrupt, t1);
        failures += check(seen.stops == 0 && seen.lines.back().prompt,
                          "a Ctrl+C after the window must prompt again, not stop");
        control.handle(StopEvent::Interrupt, t1 + kWindow);
        failures += check(seen.stops == 1, "a confirming Ctrl+C must stop the server");
        failures += check(
            !seen.records.empty() && seen.records.back().severity == OperationalSeverity::Info &&
                contains(seen.records.back().message, "Ctrl+C: stopping") &&
                contains(seen.records.back().message, "requests are cancelled") &&
                contains(seen.records.back().message,
                         "Press Ctrl+C again to exit without saving the prefix cache"),
            "the stop must be logged as a Ctrl+C stop that cancels requests");
        failures += check(!seen.lines.back().prompt &&
                              seen.line() == "Closing: saving the prefix cache | Press Ctrl+C "
                                             "again to exit without saving",
                          "the bottom line must show the save in progress while stopping");

        // While stopping, one more Ctrl+C abandons the save and exits at once.
        control.handle(StopEvent::Interrupt, t1 + kWindow + kTick);
        failures += check(seen.abandons == 1 && seen.exits == 1 && seen.stops == 1,
                          "one Ctrl+C while stopping must abandon the save and exit");
        failures += check(
            seen.records.back().severity == OperationalSeverity::Warning &&
                contains(seen.records.back().message, "exiting without saving the prefix cache") &&
                contains(seen.records.back().message, "previous file is kept"),
            "exiting during the save must say the previous file is kept");
    }

    {
        // The exit reports what became of the save.
        const auto exit_record = [](Abandon result) {
            Observed seen;
            seen.abandon_result = result;
            StopControl control(true, seen.actions());
            control.serve([&] { ++seen.stops; });
            control.handle(StopEvent::Terminate, t0);
            control.handle(StopEvent::Interrupt, t0 + kTick);
            return seen.records.back().message;
        };
        failures += check(contains(exit_record(Abandon::Saved), "already saved"),
                          "an exit after a finished save must say the cache was saved");
        failures += check(contains(exit_record(Abandon::StillWriting), "may remain"),
                          "an exit while the save still writes must say its file may remain");
    }

    {
        // Ctrl+Break, console close and SIGTERM stop at once and never exit early.
        Observed seen;
        StopControl control(true, seen.actions());
        control.serve([&] { ++seen.stops; });
        control.handle(StopEvent::Interrupt, t0);
        failures += check(control.handle(StopEvent::Terminate, t0 + kTick) && seen.stops == 1 &&
                              !contains(seen.records.back().message, "Ctrl+C:"),
                          "a terminate event must stop at once, even with a Ctrl+C pending");
        control.handle(StopEvent::Terminate, t0 + 2 * kTick);
        failures += check(seen.stops == 1 && seen.exits == 0,
                          "a repeated terminate event must neither stop again nor exit");
        // A Ctrl+C during that stop exits at once too.
        control.handle(StopEvent::Interrupt, t0 + 3 * kTick);
        failures += check(seen.exits == 1 && seen.abandons == 1,
                          "a Ctrl+C during a terminate stop must abandon the save and exit");
    }

    {
        // Without a prefix cache file the texts do not mention one and nothing is abandoned.
        Observed seen;
        StopControl control(false, seen.actions());
        control.serve([&] { ++seen.stops; });
        control.handle(StopEvent::Interrupt, t0);
        failures += check(seen.line() == "Press Ctrl+C again within 5 s to close",
                          "without a cache file the prompt must offer to close");
        control.handle(StopEvent::Interrupt, t0 + kTick);
        failures += check(seen.stops == 1 && !contains(seen.records.back().message, "prefix") &&
                              seen.line() == "Closing | Press Ctrl+C again to exit at once",
                          "without a cache file the stop must not mention the prefix cache");
        control.handle(StopEvent::Interrupt, t0 + 2 * kTick);
        failures += check(seen.exits == 1 && seen.abandons == 0 &&
                              !contains(seen.records.back().message, "prefix"),
                          "without a cache file one more Ctrl+C must exit without a save");
    }

    {
        // After listen() returns, no event reaches the server; finish() restores default actions.
        Observed seen;
        StopControl control(true, seen.actions());
        control.serve([&] { ++seen.stops; });
        control.end_serving();
        control.handle(StopEvent::Interrupt, t0);
        failures += check(seen.stops == 0 && seen.exits == 1,
                          "after listen() returns, one Ctrl+C must exit, not stop");
        control.finish();
        failures +=
            check(!control.handle(StopEvent::Interrupt, t0 + 2 * kTick) && seen.line().empty(),
                  "a finished control must clear the line and decline events");
    }

    {
        // The handshake the exit relies on: abandon() returns once a save in progress has noticed,
        // removed its file and ended, and a save it precedes never begins.
        const infernix::PrefixCacheSaveControl control;
        failures += check(control.begin(), "a save must begin while nothing abandoned it");
        std::atomic<bool> removed{false};
        std::thread writer([&] {
            while (!control.abandoned()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            removed = true;
            control.end(false);
        });
        const Abandon result = control.abandon(std::chrono::seconds(10));
        writer.join();
        failures += check(result == Abandon::Unsaved && removed,
                          "abandon() must wait until the save has let go of its file");
        failures += check(!control.begin(), "a save must not begin once abandoned");

        const infernix::PrefixCacheSaveControl stuck;
        (void)stuck.begin();
        failures += check(stuck.abandon(std::chrono::milliseconds(10)) == Abandon::StillWriting,
                          "abandon() must give up on a save that does not stop");
        stuck.end(false);

        const infernix::PrefixCacheSaveControl saved;
        (void)saved.begin();
        saved.end(true);
        failures += check(saved.abandon(std::chrono::milliseconds(0)) == Abandon::Saved,
                          "abandon() after a completed save must report it saved");

        const infernix::PrefixCacheSaveControl idle;
        failures +=
            check(idle.abandon(std::chrono::milliseconds(0)) == Abandon::Unsaved && !idle.begin(),
                  "abandon() before any save must keep one from starting");

        // Copies share one state, as the product's copy and the Engine's options do.
        const infernix::PrefixCacheSaveControl original;
        const infernix::PrefixCacheSaveControl copy =
            original; // NOLINT(performance-unnecessary-copy-initialization)
        (void)copy.abandon(std::chrono::milliseconds(0));
        failures += check(original.abandoned(), "copies of a save control must share its state");
    }

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
