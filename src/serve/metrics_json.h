#pragma once

// The JSON representation of GET /metrics?format=json. The key names are a consumer contract:
// monitor.py's frozen Strata mapping and the launcher read them directly (see docs/serving.md).
// Every string in the types below is engine-authored (protocol, finish and phase are server
// constants) except a request's label, the client's own X-Infernix-Request-Label (printable ASCII,
// at most 128 bytes; request_label); no request text is copied into them.

#include "core/host_memory.h"
#include "serve/console_stats.h"
#include "serve/generation_service.h"
#include "serve/request_events.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace infernix::serve {

inline constexpr const char* kMetricsJsonSchema   = "infernix.telemetry/1";
inline constexpr std::size_t kMetricsRequestLimit = 128;

[[nodiscard]] inline double unix_time_seconds() {
    return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch())
        .count();
}

struct MetricsJsonRequest {
    std::uint64_t id                      = 0;
    double time_unix                      = 0.0;
    std::string label; // X-Infernix-Request-Label; empty: none (rendered null)
    std::string protocol;
    std::string finish;
    std::int64_t prompt_tokens            = 0;
    std::uint64_t cached_prompt_tokens    = 0;
    std::uint64_t computed_prefill_tokens = 0;
    std::int64_t output_tokens            = 0;
    double ttft_seconds                   = 0.0;
    double prefill_seconds                = 0.0;
    double decode_seconds                 = 0.0;
    double total_seconds                  = 0.0;
    std::uint64_t speculative_draft_tokens    = 0;
    std::uint64_t speculative_accepted_tokens = 0;
    std::uint64_t ngram_draft_tokens          = 0;
    std::uint64_t ngram_accepted_tokens       = 0;
    std::uint64_t expert_lookups              = 0;
    std::uint64_t expert_hits                 = 0;
};

struct MetricsJsonCounts {
    std::uint64_t completed = 0;
    std::uint64_t cancelled = 0;
    std::uint64_t failed    = 0;
    std::uint64_t rejected  = 0;
};

// Where a request in flight stands: queued until the Engine admits it, then prefill until its first
// output token, then decode.
enum class ActiveRequestPhase : std::uint8_t { Queued, Prefill, Decode };

[[nodiscard]] const char* active_request_phase_name(ActiveRequestPhase phase) noexcept;

// One request in flight, as live.active lists it. The token counts are its current generation's (a
// request continuing after a server-side tool shows its latest generation).
struct MetricsJsonActiveRequest {
    std::uint64_t id = 0;
    std::string label; // X-Infernix-Request-Label; empty: none (rendered null)
    std::string protocol;
    ActiveRequestPhase phase       = ActiveRequestPhase::Queued;
    std::uint32_t prompt_tokens    = 0;
    std::uint32_t cached_tokens    = 0;
    // Prompt tokens done, the cached ones included; prompt_tokens once the prompt is done.
    std::uint32_t processed_tokens = 0;
    std::uint32_t output_tokens    = 0;
    double elapsed_ms              = 0.0;
};

// The generation requests in flight: from their submission to the Engine (after prompt preparation)
// until their terminal record, fed by the Engine's progress observations
// (infernix::GenerationObservationOptions::progress). Thread-safe; a poll copies at most
// --max-concurrency + --max-pending-requests rows.
class LiveRequests {
public:
    using Clock = std::chrono::steady_clock;

    // `started`: when the server took the request (its preparation's start); a default time point
    // stands for now.
    void begin(std::uint64_t id, std::string label, std::string protocol, std::uint32_t prompt_tokens,
               Clock::time_point started);
    // The request's newest position; ignored once it has ended.
    void progress(std::uint64_t id, const infernix::GenerationProgressObservation& observation);
    void end(std::uint64_t id) noexcept;
    // Oldest first (by server request id).
    [[nodiscard]] std::vector<MetricsJsonActiveRequest> rows(Clock::time_point now) const;

private:
    struct Entry {
        std::string label;
        std::string protocol;
        Clock::time_point started;
        bool admitted = false;
        infernix::GenerationProgressObservation position;
    };

    mutable std::mutex mutex_;
    std::map<std::uint64_t, Entry> entries_;
};

struct MetricsJsonSnapshot {
    std::string instance_id;
    std::string model;
    std::string architecture;
    std::string weights;
    std::string kv;
    std::string speculative_backend;
    double time_unix       = 0.0;
    double start_time_unix = 0.0;
    std::uint32_t max_context     = 0;
    std::uint32_t max_concurrency = 0;
    infernix::RuntimeStats stats{};
    // `memory`: the parts are stats.memory; the device's and the host's totals and this process's use are these
    // readings (GenerationBackend::device_memory, host_memory_snapshot), taken with the snapshot.
    infernix::DeviceMemoryReading device_memory{};
    HostMemorySnapshot host_memory{};
    MetricsJsonCounts counts{};
    double decode_tokens_per_second  = 0.0;
    double prefill_tokens_per_second = 0.0;
    bool rates_valid                 = false;
    // The requests in flight (LiveRequests::rows), rendered as live.active.
    std::vector<MetricsJsonActiveRequest> active;
    std::vector<MetricsJsonRequest> requests;
    // The console statistics panel's aggregates (session and last ten), rendered as
    // `session_stats`; absent when not supplied.
    std::optional<ConsoleStatsSnapshot> session_stats;
};

enum class MetricsFormat : std::uint8_t { Prometheus, Json, Invalid };

// Pure: query/header in, representation out. An unknown explicit format is Invalid (the route
// answers 400); an absent format keeps Prometheus unless the client asks for JSON.
[[nodiscard]] MetricsFormat select_metrics_format(std::string_view format_param,
                                                  std::string_view accept_header) noexcept;

[[nodiscard]] std::string render_metrics_json(const MetricsJsonSnapshot& snapshot);

[[nodiscard]] MetricsJsonRequest make_metrics_json_request(std::uint64_t id,
                                                           const RequestLogContext& context,
                                                           const GenerationOutcome& outcome,
                                                           double time_unix);
[[nodiscard]] MetricsJsonRequest make_metrics_json_failure(std::uint64_t id,
                                                           const RequestLogContext& context,
                                                           bool cancelled, double time_unix);

// Bounded terminal-request window plus the last measured interval rates. Not durable by design:
// --request-log-jsonl is the durable record.
class RequestHistory {
public:
    void done(const RequestLogContext& context, const GenerationOutcome& outcome);
    void failure(const RequestLogContext& context, bool cancelled);
    void set_rates(double decode_tokens_per_second, double prefill_tokens_per_second);
    [[nodiscard]] std::vector<MetricsJsonRequest> rows() const;
    [[nodiscard]] bool rates(double& decode, double& prefill) const;

private:
    mutable std::mutex mutex_;
    std::deque<MetricsJsonRequest> rows_;
    double decode_rate_  = 0.0;
    double prefill_rate_ = 0.0;
    bool rates_valid_    = false;
};

} // namespace infernix::serve
