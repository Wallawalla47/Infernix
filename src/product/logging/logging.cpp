#include "product/logging/logging.h"

#include "core/cuda_diagnostics.h"
#include "media/decode/decode.h"
#include "product/log_colour/log_colour.h"
#include "product/logging/engine_diagnostics.h"
#include "product/logging/pretty_format.h"

#include <spdlog/details/log_msg_buffer.h>
#include <spdlog/formatter.h>
#include <spdlog/logger.h>
#include <spdlog/pattern_formatter.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#ifndef _WIN32
#    include <sys/ioctl.h>
#    include <unistd.h>
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <deque>
#include <iterator>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace infernix::product {
namespace {

spdlog::level::level_enum to_spdlog_level(LogLevel level) {
    switch (level) {
    case LogLevel::Trace:
        return spdlog::level::trace;
    case LogLevel::Debug:
        return spdlog::level::debug;
    case LogLevel::Info:
        return spdlog::level::info;
    case LogLevel::Warning:
        return spdlog::level::warn;
    case LogLevel::Error:
        return spdlog::level::err;
    case LogLevel::Critical:
        return spdlog::level::critical;
    case LogLevel::Off:
        return spdlog::level::off;
    }
    throw std::invalid_argument("LoggingOptions level is invalid");
}

std::string_view service_level_name(spdlog::level::level_enum level) noexcept {
    switch (level) {
    case spdlog::level::trace:
        return "TRACE";
    case spdlog::level::debug:
        return "DEBUG";
    case spdlog::level::info:
        return "INFO ";
    case spdlog::level::warn:
        return "WARN ";
    case spdlog::level::err:
        return "ERROR";
    case spdlog::level::critical:
        return "FATAL";
    case spdlog::level::off:
        return "OFF  ";
    case spdlog::level::n_levels:
        break;
    }
    return "UNKWN";
}

std::string_view tool_level_name(spdlog::level::level_enum level) noexcept {
    switch (level) {
    case spdlog::level::trace:
        return "trace: ";
    case spdlog::level::debug:
        return "debug: ";
    case spdlog::level::info:
        return {};
    case spdlog::level::warn:
        return "warning: ";
    case spdlog::level::err:
        return "error: ";
    case spdlog::level::critical:
        return "fatal: ";
    case spdlog::level::off:
    case spdlog::level::n_levels:
        break;
    }
    return {};
}

class PrettyLogFormatter final : public spdlog::formatter {
public:
    explicit PrettyLogFormatter(LogPresentation presentation) : presentation_(presentation) {}

    void format(const spdlog::details::log_msg& message,
                spdlog::memory_buf_t& destination) override {
        if (presentation_ == LogPresentation::Service) {
            const auto since_epoch = message.time.time_since_epoch();
            const auto whole_seconds =
                std::chrono::duration_cast<std::chrono::seconds>(since_epoch);
            const auto milliseconds =
                std::chrono::duration_cast<std::chrono::milliseconds>(since_epoch - whole_seconds)
                    .count();
            const std::time_t wall_seconds = std::chrono::system_clock::to_time_t(
                std::chrono::system_clock::time_point(whole_seconds));
            std::tm local{};
#ifdef _WIN32
            localtime_s(&local, &wall_seconds);
#else
            localtime_r(&wall_seconds, &local);
#endif
            fmt::format_to(std::back_inserter(destination),
                           "{:04}-{:02}-{:02} {:02}:{:02}:{:02}.{:03}  ", local.tm_year + 1900,
                           local.tm_mon + 1, local.tm_mday, local.tm_hour, local.tm_min,
                           local.tm_sec, milliseconds);
            message.color_range_start    = destination.size();
            const std::string_view level = service_level_name(message.level);
            destination.append(level.data(), level.data() + level.size());
            message.color_range_end = destination.size();
            destination.push_back(' ');
        } else {
            const std::string_view level = tool_level_name(message.level);
            message.color_range_start    = destination.size();
            destination.append(level.data(), level.data() + level.size());
            message.color_range_end = destination.size();
        }
        destination.append(message.payload.data(), message.payload.data() + message.payload.size());
        destination.push_back('\n');
    }

    [[nodiscard]] std::unique_ptr<spdlog::formatter> clone() const override {
        return std::make_unique<PrettyLogFormatter>(presentation_);
    }

private:
    LogPresentation presentation_;
};

spdlog::color_mode to_spdlog_color_mode(LogColorMode mode) {
    switch (mode) {
    case LogColorMode::Auto:
        return spdlog::color_mode::automatic;
    case LogColorMode::Always:
        return spdlog::color_mode::always;
    case LogColorMode::Never:
        return spdlog::color_mode::never;
    }
    throw std::invalid_argument("LoggingOptions color mode is invalid");
}

// FFmpeg's lines concern client-supplied media and name no request, and a decode that fails also
// fails its request, which reports that in its own record. FFmpeg's errors are therefore warnings,
// and its warnings and notices, such as swscaler's "deprecated pixel format" notice on every
// full-range (JPEG-coded) video, are debug detail.
media::decode::LibraryLogHandler media_log_handler(std::shared_ptr<spdlog::logger> logger) {
    return [logger = std::move(logger)](const media::decode::LibraryLogLine& line) {
        const spdlog::level::level_enum level =
            line.severity == media::decode::LibraryLogSeverity::Error ? spdlog::level::warn
                                                                      : spdlog::level::debug;
        if (!logger->should_log(level)) { return; }
        const std::string message = format_pretty_text(line.message);
        if (line.source.empty()) {
            logger->log(level, "media | {}", message);
        } else {
            logger->log(level, "media | {}: {}", format_pretty_text(line.source), message);
        }
    };
}

void report_logging_error(const std::string& message) noexcept {
    static std::atomic_flag reported = ATOMIC_FLAG_INIT;
    if (reported.test_and_set(std::memory_order_relaxed)) { return; }
    std::fprintf(stderr, "infernix logging failure: %s\n", message.c_str());
    std::fflush(stderr);
}

struct TerminalSize {
    std::size_t columns = 0;
    std::size_t rows    = 0;
};

TerminalSize stderr_terminal_size() noexcept {
#ifdef _WIN32
    CONSOLE_SCREEN_BUFFER_INFO info{};
    if (::GetConsoleScreenBufferInfo(::GetStdHandle(STD_ERROR_HANDLE), &info) == 0) { return {}; }
    return {
        .columns = static_cast<std::size_t>(info.srWindow.Right - info.srWindow.Left + 1),
        .rows    = static_cast<std::size_t>(info.srWindow.Bottom - info.srWindow.Top + 1),
    };
#else
    winsize size{};
    if (::ioctl(STDERR_FILENO, TIOCGWINSZ, &size) != 0) { return {}; }
    return {.columns = size.ws_col, .rows = size.ws_row};
#endif
}

// SGR sequences matching spdlog's console level colours.
std::string_view level_colour(spdlog::level::level_enum level) noexcept {
    switch (level) {
    case spdlog::level::trace:
        return "\x1b[37m";
    case spdlog::level::debug:
        return "\x1b[36m";
    case spdlog::level::info:
        return "\x1b[32m";
    case spdlog::level::warn:
        return "\x1b[33m\x1b[1m";
    case spdlog::level::err:
        return "\x1b[31m\x1b[1m";
    case spdlog::level::critical:
        return "\x1b[1m\x1b[41m";
    case spdlog::level::off:
    case spdlog::level::n_levels:
        break;
    }
    return {};
}


// The transient footer beneath the scrolling records: an optional pinned panel followed by an
// optional progress line.
struct ConsoleFooter {
    std::string progress_line;
    std::vector<std::string> panel_lines;
};

// Draws the records and the footer on stderr. Every update erases the footer, writes its records
// where the footer began, and redraws the footer below them, so the footer stays at the bottom of
// the console while the scrollback keeps every record.
//
// With VT cursor control, each erase-records-redraw is one write bracketed as a synchronized update,
// so a terminal never paints a frame with the footer missing or half drawn. Without it the footer
// can only be the single progress line, erased with spaces around spdlog's own console writes.
// Not thread-safe: its owner serializes every call.
class ConsoleRenderer {
public:
    // The Windows color sink writes messages that carry a colour range through
    // WriteConsoleA, which silently fails on a redirected handle; with
    // color_mode::always every such message would vanish from file/pipe logs.
    // The formatter colours every service line's level, so any non-console
    // stderr must fall back to plain (uncoloured) writes.
    explicit ConsoleRenderer(spdlog::color_mode color)
        : sink_(log_colour::stderr_is_console() ? color : spdlog::color_mode::never),
          interactive_(log_colour::stderr_is_console()),
          cursor_control_(interactive_ && log_colour::enable_stderr_vt_processing()),
          colour_(interactive_ && color != spdlog::color_mode::never),
          formatter_(std::make_unique<spdlog::pattern_formatter>()) {}

    [[nodiscard]] bool interactive() const noexcept { return interactive_; }

    [[nodiscard]] bool cursor_control() const noexcept { return cursor_control_; }

    void set_pattern(const std::string& pattern) {
        formatter_ = std::make_unique<spdlog::pattern_formatter>(pattern);
        sink_.set_pattern(pattern);
    }

    void set_formatter(std::unique_ptr<spdlog::formatter> formatter) {
        formatter_ = formatter->clone();
        sink_.set_formatter(std::move(formatter));
    }

    // One console update: erases the drawn footer, runs `body` (its record() and released_panel()
    // calls write in order), then draws `footer`.
    template <typename Body> void update(const ConsoleFooter& footer, Body&& body) {
        begin();
        try {
            body();
        } catch (...) {
            end(footer);
            throw;
        }
        end(footer);
    }

    void record(const spdlog::details::log_msg& message) {
        if (!cursor_control_) {
            sink_.log(message);
            return;
        }
        spdlog::memory_buf_t formatted;
        formatter_->format(message, formatted);
        const std::string_view text(formatted.data(), formatted.size());
        const std::string_view colour = colour_ ? level_colour(message.level) : "";
        if (!colour.empty() && message.color_range_end > message.color_range_start) {
            pending_.append(text.substr(0, message.color_range_start));
            pending_.append(colour);
            pending_.append(text.substr(message.color_range_start,
                                        message.color_range_end - message.color_range_start));
            pending_.append("\x1b[m");
            pending_.append(text.substr(message.color_range_end));
        } else {
            pending_.append(text);
        }
    }

    // A panel left on screen as ordinary output, so the final statistics survive exit.
    void released_panel(const std::vector<std::string>& lines) {
        if (!cursor_control_) { return; }
        const std::size_t columns = stderr_terminal_size().columns;
        for (const std::string& line : lines) {
            pending_ += columns > 1 ? fit_terminal_line(line, columns - 1) : line;
            pending_.push_back('\n');
        }
    }

    void flush() {
        sink_.flush();
        if (interactive_) { std::fflush(stderr); }
    }

private:
    void begin() {
        if (cursor_control_) {
            pending_ = "\x1b[?2026h";
            append_erase(pending_);
            return;
        }
        erase_legacy();
    }

    void end(const ConsoleFooter& footer) {
        if (cursor_control_) {
            append_draw(pending_, footer);
            pending_ += "\x1b[?2026l";
            std::fwrite(pending_.data(), 1, pending_.size(), stderr);
            std::fflush(stderr);
            pending_.clear();
            return;
        }
        draw_legacy(footer);
    }

    void append_erase(std::string& out) {
        if (rendered_widths_.empty()) { return; }
        // A resize may have rewrapped the drawn lines; count the rows they occupy now.
        const std::size_t columns = stderr_terminal_size().columns;
        std::size_t rows          = 0;
        for (const std::size_t width : rendered_widths_) {
            rows += columns == 0 || width <= columns ? 1 : (width + columns - 1) / columns;
        }
        out += '\r';
        if (rows > 1) { out += "\x1b[" + std::to_string(rows - 1) + 'A'; }
        out += "\x1b[J";
        rendered_widths_.clear();
    }

    void append_draw(std::string& out, const ConsoleFooter& footer) {
        const TerminalSize size = stderr_terminal_size();
        // Keep one column free so no footer line reaches the auto-wrap margin, and keep at least
        // one scrolling row above the footer; a console too short for the panel shows only the
        // progress line.
        const std::size_t columns = size.columns > 1 ? size.columns - 1 : 0;
        const std::size_t footer_rows =
            footer.panel_lines.size() + (footer.progress_line.empty() ? 0 : 1);
        const bool show_panel  = size.rows == 0 || footer_rows < size.rows;
        const auto append_line = [&](std::string_view line) {
            if (!rendered_widths_.empty()) { out.push_back('\n'); }
            const std::string fitted =
                columns == 0 ? std::string(line) : fit_terminal_line(line, columns);
            out += fitted;
            rendered_widths_.push_back(terminal_display_width(fitted));
        };
        if (show_panel) {
            for (const std::string& line : footer.panel_lines) { append_line(line); }
        }
        if (!footer.progress_line.empty()) { append_line(footer.progress_line); }
    }

    void erase_legacy() {
        if (rendered_widths_.empty()) { return; }
        std::fputc('\r', stderr);
        for (std::size_t index = 0; index < rendered_widths_.front(); ++index) {
            std::fputc(' ', stderr);
        }
        std::fputc('\r', stderr);
        rendered_widths_.clear();
    }

    void draw_legacy(const ConsoleFooter& footer) {
        if (!interactive_) { return; }
        if (!footer.progress_line.empty()) {
            std::fwrite(footer.progress_line.data(), 1, footer.progress_line.size(), stderr);
            rendered_widths_.push_back(footer.progress_line.size());
        }
        std::fflush(stderr);
    }

    spdlog::sinks::stderr_color_sink_st sink_;
    bool interactive_    = false;
    bool cursor_control_ = false;
    bool colour_         = false;
    std::unique_ptr<spdlog::formatter> formatter_;
    // The update being assembled (cursor control only).
    std::string pending_;
    // Visible width of each footer line currently on screen, top to bottom.
    std::vector<std::size_t> rendered_widths_;
};

// Console-queue level classes, dropped lowest first: trace and debug, info, warnings and errors.
constexpr std::size_t kConsoleLevelClasses = 3;
constexpr std::size_t kConsoleInfoClass    = 1;

std::size_t console_level_class(spdlog::level::level_enum level) noexcept {
    if (level <= spdlog::level::debug) { return 0; }
    return level == spdlog::level::info ? kConsoleInfoClass : 2;
}

// How long an error record waits for its write, flush() for the queue, and the sink's destruction
// for the drained writer to end.
constexpr std::chrono::milliseconds kConsoleRecordWait{500};
constexpr std::chrono::milliseconds kConsoleDrainWait{2000};
constexpr std::chrono::milliseconds kConsoleExitWait{100};

struct QueuedConsoleEntry {
    std::uint64_t sequence = 0;
    std::size_t bytes      = 0;
    spdlog::details::log_msg_buffer message; // a record, unless `panel`
    std::vector<std::string> released_panel;
    bool panel = false;
};

// What the stderr sink and its console writer share. The writer holds a reference of its own, so a
// writer still blocked in a stderr write when the sink is destroyed is detached, not joined, and
// ends with the process (or once stderr takes output) without touching freed memory.
struct ConsoleState {
    ConsoleState(spdlog::color_mode color, std::size_t queue_bytes)
        : renderer(color), queue_bytes(queue_bytes) {}

    [[nodiscard]] bool pending() const noexcept {
        return waiting_count != 0 || dropped != 0 || footer_version != footer_taken;
    }

    // Lock order: `mutex`, then `render_mutex`; the writer never holds both.
    std::mutex render_mutex;
    ConsoleRenderer renderer; // under render_mutex

    std::mutex mutex; // guards everything below
    // The writer waits on `work`; waits for a written record, a drained queue or the writer's
    // exit use `progress`.
    std::condition_variable work;
    std::condition_variable progress;
    ConsoleFooter footer;
    std::uint64_t footer_version = 0; // footer changes
    std::uint64_t footer_taken   = 0; // the change the writer last drew
    // Oldest first within each level class; the writer merges them by sequence.
    std::array<std::deque<QueuedConsoleEntry>, kConsoleLevelClasses> waiting;
    std::size_t waiting_count   = 0;
    std::size_t waiting_bytes   = 0;
    std::size_t queue_bytes     = 0;
    std::uint64_t next_sequence = 1;
    std::uint64_t written       = 0; // every entry up to this sequence is written or dropped
    std::uint64_t dropped       = 0; // records dropped since the last notice
    bool writing                = false;
    // A wait for the writer timed out and the writer has finished no write since: an error
    // record then queues without waiting.
    bool stalled  = false;
    bool stopping = false;
    bool exited   = false;
};

// One console update: the notice of records dropped since the last one, then `batch` in order.
void write_console_batch(ConsoleState& state, const std::vector<QueuedConsoleEntry>& batch,
                         std::uint64_t dropped, const ConsoleFooter& footer) noexcept {
    try {
        const std::lock_guard render(state.render_mutex);
        state.renderer.update(footer, [&] {
            if (dropped != 0) {
                const std::string notice = "console | " + format_pretty_count(dropped) +
                                           (dropped == 1 ? " line" : " lines") +
                                           " not shown: stderr was not taking output";
                state.renderer.record(spdlog::details::log_msg(
                    spdlog::string_view_t(), spdlog::level::warn,
                    spdlog::string_view_t(notice.data(), notice.size())));
            }
            for (const QueuedConsoleEntry& entry : batch) {
                if (entry.panel) {
                    state.renderer.released_panel(entry.released_panel);
                } else {
                    state.renderer.record(entry.message);
                }
            }
        });
        state.renderer.flush();
    } catch (const std::exception& failure) {
        report_logging_error(failure.what());
    } catch (...) {
        report_logging_error("console write failed");
    }
}

// The console writer: takes every waiting entry, oldest first, and writes them as one update with
// the footer as it is then.
void run_console_writer(ConsoleState& state) {
    std::vector<QueuedConsoleEntry> batch;
    ConsoleFooter footer;
    std::unique_lock lock(state.mutex);
    for (;;) {
        state.work.wait(lock, [&] { return state.stopping || state.pending(); });
        if (!state.pending()) { break; }
        for (;;) {
            std::deque<QueuedConsoleEntry>* oldest = nullptr;
            for (std::deque<QueuedConsoleEntry>& entries : state.waiting) {
                if (!entries.empty() &&
                    (oldest == nullptr || entries.front().sequence < oldest->front().sequence)) {
                    oldest = &entries;
                }
            }
            if (oldest == nullptr) { break; }
            batch.push_back(std::move(oldest->front()));
            oldest->pop_front();
        }
        state.waiting_count           = 0;
        state.waiting_bytes           = 0;
        const std::uint64_t dropped   = std::exchange(state.dropped, 0);
        const std::uint64_t through   = state.next_sequence - 1;
        if (state.footer_taken != state.footer_version) {
            footer             = state.footer;
            state.footer_taken = state.footer_version;
        }
        state.writing = true;
        lock.unlock();
        write_console_batch(state, batch, dropped, footer);
        batch.clear();
        lock.lock();
        state.writing = false;
        state.stalled = false;
        state.written = through;
        state.progress.notify_all();
    }
}

// The operational stderr sink. With a console queue, one writer thread owns every stderr write
// (LoggingOptions::console_queue_bytes); without one, each record and footer change is written on
// the calling thread under one mutex.
class ProgressAwareStderrSink final : public spdlog::sinks::sink {
public:
    ProgressAwareStderrSink(spdlog::color_mode color, std::size_t queue_bytes)
        : state_(std::make_shared<ConsoleState>(color, queue_bytes)),
          interactive_(state_->renderer.interactive()),
          cursor_control_(state_->renderer.cursor_control()),
          queued_(queue_bytes != 0) {
        if (queued_) {
            writer_ = std::thread([state = state_] {
                try {
                    run_console_writer(*state);
                } catch (const std::exception& failure) {
                    report_logging_error(failure.what());
                } catch (...) {
                    report_logging_error("console writer failed");
                }
                const std::lock_guard lock(state->mutex);
                state->exited = true;
                state->progress.notify_all();
            });
        }
    }

    ~ProgressAwareStderrSink() override {
        clear();
        if (!writer_.joinable()) { return; }
        (void)drain(kConsoleDrainWait);
        bool exited = false;
        try {
            std::unique_lock lock(state_->mutex);
            state_->stopping = true;
            state_->work.notify_one();
            exited = state_->progress.wait_for(lock, kConsoleExitWait,
                                               [this] { return state_->exited; });
        } catch (...) {}
        if (exited) {
            writer_.join();
        } else {
            writer_.detach();
        }
    }

    ProgressAwareStderrSink(const ProgressAwareStderrSink&)            = delete;
    ProgressAwareStderrSink& operator=(const ProgressAwareStderrSink&) = delete;

    void log(const spdlog::details::log_msg& message) override {
        if (!queued_) {
            const std::lock_guard lock(state_->mutex);
            const std::lock_guard render(state_->render_mutex);
            state_->renderer.update(state_->footer, [&] { state_->renderer.record(message); });
            return;
        }
        QueuedConsoleEntry entry{.message = spdlog::details::log_msg_buffer(message)};
        entry.bytes = sizeof(QueuedConsoleEntry) + message.payload.size();
        std::unique_lock lock(state_->mutex);
        const std::uint64_t sequence =
            push_locked(std::move(entry), console_level_class(message.level));
        // The record just before an abort (a failed CUDA_CHECK) must reach a healthy console.
        if (message.level < spdlog::level::err || state_->stalled) { return; }
        if (!state_->progress.wait_for(lock, kConsoleRecordWait, [&] {
                return state_->written >= sequence || state_->stalled;
            })) {
            state_->stalled = true;
        }
    }

    // The console writer writes through after every update; only the direct console flushes here.
    void flush() override {
        if (queued_) { return; }
        const std::lock_guard lock(state_->mutex);
        const std::lock_guard render(state_->render_mutex);
        state_->renderer.flush();
    }

    void set_pattern(const std::string& pattern) override {
        const std::lock_guard render(state_->render_mutex);
        state_->renderer.set_pattern(pattern);
    }

    void set_formatter(std::unique_ptr<spdlog::formatter> formatter) override {
        const std::lock_guard render(state_->render_mutex);
        state_->renderer.set_formatter(std::move(formatter));
    }

    [[nodiscard]] bool interactive() const noexcept { return interactive_; }

    [[nodiscard]] bool cursor_control() const noexcept { return cursor_control_; }

    void update(std::string line) {
        if (!interactive_) { return; }
        change_footer([&](ConsoleFooter& footer) { footer.progress_line = std::move(line); });
    }

    void update_panel(std::vector<std::string> lines) {
        if (!cursor_control_) { return; }
        change_footer([&](ConsoleFooter& footer) { footer.panel_lines = std::move(lines); });
    }

    void clear() noexcept {
        if (!interactive_) { return; }
        try {
            change_footer([](ConsoleFooter& footer) { footer.progress_line.clear(); });
        } catch (...) {}
    }

    // Leaves the last panel on screen as ordinary output, so the final statistics survive exit.
    void release_panel() noexcept {
        if (!cursor_control_) { return; }
        try {
            const std::lock_guard lock(state_->mutex);
            std::vector<std::string> lines = std::exchange(state_->footer.panel_lines, {});
            if (!queued_) {
                const std::lock_guard render(state_->render_mutex);
                state_->renderer.update(state_->footer,
                                        [&] { state_->renderer.released_panel(lines); });
                return;
            }
            ++state_->footer_version;
            QueuedConsoleEntry entry{.panel = true};
            entry.bytes = sizeof(QueuedConsoleEntry);
            for (const std::string& line : lines) { entry.bytes += line.size(); }
            entry.released_panel = std::move(lines);
            (void)push_locked(std::move(entry), kConsoleInfoClass);
        } catch (...) {}
    }

    void clear_panel() noexcept {
        if (!cursor_control_) { return; }
        try {
            change_footer([](ConsoleFooter& footer) { footer.panel_lines.clear(); });
        } catch (...) {}
    }

    // Waits up to `bound` for every queued record and footer change to be written.
    bool drain(std::chrono::milliseconds bound) noexcept {
        if (!queued_) { return true; }
        try {
            std::unique_lock lock(state_->mutex);
            if (state_->exited) { return !state_->pending(); }
            const bool drained = state_->progress.wait_for(
                lock, bound, [this] { return !state_->pending() && !state_->writing; });
            if (!drained) { state_->stalled = true; }
            return drained;
        } catch (...) {
            return false;
        }
    }

private:
    template <typename Change> void change_footer(Change&& change) {
        const std::lock_guard lock(state_->mutex);
        change(state_->footer);
        if (queued_) {
            ++state_->footer_version;
            state_->work.notify_one();
            return;
        }
        const std::lock_guard render(state_->render_mutex);
        state_->renderer.update(state_->footer, [] {});
    }

    // Queues `entry` in `level_class`. A full queue drops the oldest waiting entry of the lowest
    // class present while the entry does not fit, or the entry itself when its class is lower than
    // every waiting one. Returns the entry's sequence.
    std::uint64_t push_locked(QueuedConsoleEntry entry, std::size_t level_class) {
        ConsoleState& state   = *state_;
        entry.sequence        = state.next_sequence++;
        const std::uint64_t sequence = entry.sequence;
        state.work.notify_one();
        while (state.waiting_count != 0 && state.waiting_bytes + entry.bytes > state.queue_bytes) {
            std::size_t lowest = 0;
            while (state.waiting[lowest].empty()) { ++lowest; }
            ++state.dropped;
            if (level_class < lowest) { return sequence; }
            state.waiting_bytes -= state.waiting[lowest].front().bytes;
            --state.waiting_count;
            state.waiting[lowest].pop_front();
        }
        state.waiting_bytes += entry.bytes;
        ++state.waiting_count;
        state.waiting[level_class].push_back(std::move(entry));
        return sequence;
    }

    std::shared_ptr<ConsoleState> state_;
    bool interactive_    = false;
    bool cursor_control_ = false;
    bool queued_         = false;
    std::thread writer_;
};

} // namespace

LogLevel parse_log_level(std::string_view value) {
    if (value == "trace") { return LogLevel::Trace; }
    if (value == "debug") { return LogLevel::Debug; }
    if (value == "info") { return LogLevel::Info; }
    if (value == "warning" || value == "warn") { return LogLevel::Warning; }
    if (value == "error") { return LogLevel::Error; }
    if (value == "critical" || value == "fatal") { return LogLevel::Critical; }
    if (value == "off") { return LogLevel::Off; }
    throw std::invalid_argument("invalid log level: " + std::string(value));
}

namespace {

// Length of the ANSI CSI sequence starting at `index`, or 0 when none starts there.
std::size_t csi_length(std::string_view line, std::size_t index) noexcept {
    if (line[index] != '\x1b' || index + 1 >= line.size() || line[index + 1] != '[') { return 0; }
    std::size_t end = index + 2;
    while (end < line.size() && !(line[end] >= 0x40 && line[end] <= 0x7e)) { ++end; }
    return std::min(end + 1, line.size()) - index;
}

std::size_t utf8_length(std::string_view line, std::size_t index) noexcept {
    const auto byte    = static_cast<unsigned char>(line[index]);
    std::size_t length = byte >= 0xf0 ? 4 : byte >= 0xe0 ? 3 : byte >= 0xc0 ? 2 : 1;
    return std::min(length, line.size() - index);
}

} // namespace

std::size_t terminal_display_width(std::string_view line) noexcept {
    std::size_t width = 0;
    for (std::size_t index = 0; index < line.size();) {
        if (const std::size_t escape = csi_length(line, index); escape != 0) {
            index += escape;
            continue;
        }
        index += utf8_length(line, index);
        ++width;
    }
    return width;
}

std::string fit_terminal_line(std::string_view line, std::size_t columns) {
    std::string out;
    out.reserve(line.size());
    std::size_t width = 0;
    bool styled       = false;
    for (std::size_t index = 0; index < line.size();) {
        if (const std::size_t escape = csi_length(line, index); escape != 0) {
            out.append(line.substr(index, escape));
            styled = true;
            index += escape;
            continue;
        }
        if (width == columns) {
            if (styled) { out += "\x1b[0m"; }
            return out;
        }
        const std::size_t length = utf8_length(line, index);
        out.append(line.substr(index, length));
        index += length;
        ++width;
    }
    return out;
}

struct TerminalProgress::Impl {
    std::shared_ptr<ProgressAwareStderrSink> sink;
    bool info_enabled = true;
};

TerminalProgress::TerminalProgress(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

TerminalProgress::~TerminalProgress() { clear(); }

bool TerminalProgress::enabled() const noexcept {
    return impl_->info_enabled && impl_->sink->interactive();
}

void TerminalProgress::update(std::string line) {
    if (impl_->info_enabled) { impl_->sink->update(std::move(line)); }
}

void TerminalProgress::clear() noexcept { impl_->sink->clear(); }

struct TerminalPanel::Impl {
    std::shared_ptr<ProgressAwareStderrSink> sink;
    bool info_enabled = true;
};

TerminalPanel::TerminalPanel(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

TerminalPanel::~TerminalPanel() { impl_->sink->release_panel(); }

bool TerminalPanel::enabled() const noexcept {
    return impl_->info_enabled && impl_->sink->cursor_control();
}

void TerminalPanel::update(std::vector<std::string> lines) {
    if (enabled()) { impl_->sink->update_panel(std::move(lines)); }
}

void TerminalPanel::clear() noexcept { impl_->sink->clear_panel(); }

struct LoggingRuntime::Impl {
    explicit Impl(LoggingOptions options) {
        const spdlog::level::level_enum level = to_spdlog_level(options.level);
        auto sink = std::make_shared<ProgressAwareStderrSink>(to_spdlog_color_mode(options.color),
                                                              options.console_queue_bytes);
        console   = sink;
        progress  = std::shared_ptr<TerminalProgress>(new TerminalProgress(
            std::make_unique<TerminalProgress::Impl>(sink, level <= spdlog::level::info)));
        panel     = std::shared_ptr<TerminalPanel>(new TerminalPanel(
            std::make_unique<TerminalPanel::Impl>(sink, level <= spdlog::level::info)));
        logger    = std::make_shared<spdlog::logger>(std::move(options.logger_name), sink);
        logger->set_formatter(std::make_unique<PrettyLogFormatter>(options.presentation));
        logger->set_level(level);
        logger->flush_on(spdlog::level::warn);
        // An optional second sink duplicates every record to a file, beside (never instead of) the
        // console. Its own plain pattern drops the colour and cursor control the logger adds around
        // a record, so the level and timestamp land bare; a record that colours its own message text
        // (the serve panels with --log-colours, see OperationalLog::write) keeps those escapes, since
        // the pattern wraps the payload rather than rewriting it. Every record is written through as
        // it is logged: a failed CUDA_CHECK aborts without flushing stdio buffers, and the records
        // just before it are the ones a bug report needs (stderr is unbuffered, so flushing costs
        // the console nothing). The file comes first, so a console queue that is full or an error
        // record's wait for the console never delays it; it is written on the thread that logs.
        // A file that cannot be opened leaves the console logging, with a warning.
        if (!options.log_file.empty()) {
            try {
                auto file_sink =
                    std::make_shared<spdlog::sinks::basic_file_sink_mt>(options.log_file, false);
                file_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] %v");
                logger->sinks().insert(logger->sinks().begin(), std::move(file_sink));
                logger->flush_on(spdlog::level::trace);
            } catch (const std::exception& failure) {
                logger->warn("log file | {} (logging to the console only)", failure.what());
            }
        }
        logger->set_error_handler([sink](const std::string& message) noexcept {
            sink->clear();
            report_logging_error(message);
        });
        // A library writing to stderr directly would land inside the terminal footer, and the
        // next redraw would leave footer rows behind in the scrollback.
        previous_media_log = media::decode::set_library_log_handler(media_log_handler(logger));
        // A CUDA_CHECK failure aborts without flushing stdio, so its record must reach the file
        // before that; the reporter writes the bare stderr line first, then publishes here.
        previous_cuda_diagnostics =
            set_cuda_diagnostic_observer(engine_diagnostic_observer(logger));
    }

    ~Impl() {
        set_cuda_diagnostic_observer(std::move(previous_cuda_diagnostics));
        media::decode::set_library_log_handler(std::move(previous_media_log));
    }

    Impl(const Impl&)            = delete;
    Impl& operator=(const Impl&) = delete;

    std::shared_ptr<ProgressAwareStderrSink> console;
    std::shared_ptr<spdlog::logger> logger;
    std::shared_ptr<TerminalProgress> progress;
    std::shared_ptr<TerminalPanel> panel;
    media::decode::LibraryLogHandler previous_media_log;
    DiagnosticObserver previous_cuda_diagnostics;
};

LoggingRuntime::LoggingRuntime(LoggingOptions options)
    : impl_(std::make_unique<Impl>(std::move(options))) {}

LoggingRuntime::~LoggingRuntime() { flush(); }

std::shared_ptr<spdlog::logger> LoggingRuntime::logger() const noexcept { return impl_->logger; }

std::shared_ptr<TerminalProgress> LoggingRuntime::terminal_progress() const noexcept {
    return impl_->progress;
}

std::shared_ptr<TerminalPanel> LoggingRuntime::terminal_panel() const noexcept {
    return impl_->panel;
}

void LoggingRuntime::flush() noexcept {
    try {
        impl_->logger->flush();
        (void)impl_->console->drain(kConsoleDrainWait);
    } catch (const std::exception& exception) {
        impl_->progress->clear();
        report_logging_error(exception.what());
    } catch (...) {
        impl_->progress->clear();
        report_logging_error("unknown flush error");
    }
}

} // namespace infernix::product
