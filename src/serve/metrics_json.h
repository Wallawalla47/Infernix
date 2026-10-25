#pragma once

// The JSON representation of GET /metrics?format=json. The key names are a consumer contract:
// monitor.py's frozen Strata mapping reads them directly (see docs/serving.md). Every string in
// the types below is engine-authored (protocol and finish are server constants); no request text
// is copied into them.

#include "serve/generation_service.h"
#include "serve/request_events.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
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
    MetricsJsonCounts counts{};
    double decode_tokens_per_second  = 0.0;
    double prefill_tokens_per_second = 0.0;
    bool rates_valid                 = false;
    std::vector<MetricsJsonRequest> requests;
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
