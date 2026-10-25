#include "serve/metrics_json.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>

namespace infernix::serve {
namespace {

using Json = nlohmann::json;

Json number_or_null(bool valid, double value) {
    return valid && std::isfinite(value) ? Json(value) : Json(nullptr);
}

double milliseconds(double seconds) noexcept { return seconds * 1000.0; }

const char* state_name(const infernix::RuntimeStats& stats) noexcept {
    if (stats.prefilling_requests != 0) { return "reading"; }
    if (stats.running_requests != 0) { return "generating"; }
    return "idle";
}

} // namespace

MetricsFormat select_metrics_format(std::string_view format_param,
                                    std::string_view accept_header) noexcept {
    if (format_param == "json") { return MetricsFormat::Json; }
    if (format_param == "prometheus") { return MetricsFormat::Prometheus; }
    if (!format_param.empty()) { return MetricsFormat::Invalid; }
    return accept_header.find("application/json") != std::string_view::npos
               ? MetricsFormat::Json
               : MetricsFormat::Prometheus;
}

MetricsJsonRequest make_metrics_json_request(std::uint64_t id, const RequestLogContext& context,
                                             const GenerationOutcome& outcome,
                                             double time_unix) {
    const auto prompt_tokens = static_cast<std::uint64_t>(std::max(outcome.prompt_tokens, 0));
    MetricsJsonRequest row;
    row.id                      = id;
    row.time_unix               = time_unix;
    row.protocol                = context.protocol;
    row.finish                  = finish_reason_name(outcome.finish_reason);
    row.prompt_tokens           = outcome.prompt_tokens;
    row.cached_prompt_tokens    = std::min<std::uint64_t>(
        outcome.reused_prompt_tokens, prompt_tokens);
    row.computed_prefill_tokens = outcome.computed_prefill_tokens;
    row.output_tokens           = outcome.completion_tokens;
    row.ttft_seconds            = outcome.timings.first_token_seconds;
    row.prefill_seconds         = outcome.timings.prefill_seconds;
    row.decode_seconds          = outcome.timings.decode_share_seconds;
    row.total_seconds           = outcome.timings.total_seconds;
    row.speculative_draft_tokens    = outcome.speculative.drafted_tokens;
    row.speculative_accepted_tokens = outcome.speculative.accepted_tokens;
    row.ngram_draft_tokens          = outcome.speculative.ngram_drafted_tokens;
    row.ngram_accepted_tokens       = outcome.speculative.ngram_accepted_tokens;
    if (outcome.expert_cache) {
        row.expert_lookups = outcome.expert_cache->routed;
        row.expert_hits    = outcome.expert_cache->hits;
    }
    return row;
}

MetricsJsonRequest make_metrics_json_failure(std::uint64_t id, const RequestLogContext& context,
                                             bool cancelled, double time_unix) {
    MetricsJsonRequest row;
    row.id        = id;
    row.time_unix = time_unix;
    row.protocol  = context.protocol;
    row.finish    = cancelled ? "cancelled" : "failed";
    return row;
}

std::string render_metrics_json(const MetricsJsonSnapshot& s) {
    Json requests = Json::array();
    for (const MetricsJsonRequest& r : s.requests) {
        const double prefill_tps =
            r.prefill_seconds > 0.0
                ? static_cast<double>(r.computed_prefill_tokens) / r.prefill_seconds
                : 0.0;
        const double decode_tps =
            r.decode_seconds > 0.0
                ? static_cast<double>(r.output_tokens > 0 ? r.output_tokens - 1 : 0) /
                      r.decode_seconds
                : 0.0;
        const double hit_rate =
            r.prompt_tokens > 0
                ? static_cast<double>(r.cached_prompt_tokens) /
                      static_cast<double>(r.prompt_tokens)
                : 0.0;
        requests.push_back(Json{
            {"id", r.id},
            {"time", r.time_unix},
            {"protocol", r.protocol},
            {"finish", r.finish},
            {"duration_s", r.total_seconds},
            {"prompt_tokens", r.prompt_tokens},
            {"reused", r.cached_prompt_tokens},
            {"output_tokens", r.output_tokens},
            {"ttft_ms", milliseconds(r.ttft_seconds)},
            {"prompt_ms", milliseconds(r.prefill_seconds)},
            {"decode_ms", milliseconds(r.decode_seconds)},
            {"prefill_tok_s", r.prefill_seconds > 0.0 ? Json(prefill_tps) : Json(nullptr)},
            {"decode_tok_s", r.decode_seconds > 0.0 ? Json(decode_tps) : Json(nullptr)},
            {"hit_rate", r.prompt_tokens > 0 ? Json(hit_rate) : Json(nullptr)},
            {"drafts_offered", r.speculative_draft_tokens},
            {"drafts_accepted", r.speculative_accepted_tokens},
            {"ngram_drafts", r.ngram_draft_tokens},
            {"ngram_accepted", r.ngram_accepted_tokens},
            {"expert_lookups", r.expert_lookups},
            {"expert_hits", r.expert_hits},
        });
    }

    const std::uint64_t terminal =
        s.counts.completed + s.counts.cancelled + s.counts.failed + s.counts.rejected;

    const Json root{
        {"schema", kMetricsJsonSchema},
        {"time", s.time_unix},
        {"engine",
         Json{{"name", "infernix"},
              {"model", s.model},
              {"architecture", s.architecture},
              {"weights", s.weights},
              {"kv", s.kv},
              {"speculative_backend", s.speculative_backend},
              {"max_context", s.max_context},
              {"max_concurrency", s.max_concurrency}}},
        {"live",
         Json{{"state", state_name(s.stats)},
              {"running", s.stats.running_requests},
              {"waiting", s.stats.waiting_requests},
              {"prefilling", s.stats.prefilling_requests},
              {"decode_ready", s.stats.decode_ready_requests},
              {"paused", s.stats.paused_requests},
              {"replaying", s.stats.replaying_requests},
              {"materializing", s.stats.materializing_requests},
              {"prompt_tokens", nullptr},
              {"tok_s", number_or_null(s.rates_valid, s.decode_tokens_per_second)},
              {"tok_s_mean", number_or_null(s.rates_valid, s.decode_tokens_per_second)},
              {"prefill_tok_s_mean", number_or_null(s.rates_valid, s.prefill_tokens_per_second)},
              {"start_time", s.start_time_unix},
              {"instance_id", s.instance_id}}},
        {"totals",
         Json{{"requests", terminal},
              {"completed", s.counts.completed},
              {"cancelled", s.counts.cancelled},
              {"failed", s.counts.failed},
              {"rejected", s.counts.rejected},
              {"prompt_tokens", s.stats.prompt_tokens},
              {"reused", s.stats.reused_prompt_tokens},
              {"computed_prefill_tokens", s.stats.computed_prefill_tokens},
              {"output_tokens", s.stats.generated_tokens},
              {"decode_tokens", s.stats.committed_decode_tokens},
              {"drafts_offered", s.stats.speculative_draft_tokens},
              {"drafts_accepted", s.stats.speculative_accepted_tokens},
              {"preemptions", s.stats.preemptions},
              {"replayed_tokens", s.stats.replayed_tokens},
              {"prompt_ms", nullptr},
              {"decode_ms", nullptr}}},
        {"requests", std::move(requests)},
    };
    return root.dump() + '\n';
}

void RequestHistory::done(const RequestLogContext& context, const GenerationOutcome& outcome) {
    std::lock_guard lock(mutex_);
    rows_.push_back(make_metrics_json_request(context.id, context, outcome, unix_time_seconds()));
    while (rows_.size() > kMetricsRequestLimit) { rows_.pop_front(); }
}

void RequestHistory::failure(const RequestLogContext& context, bool cancelled) {
    std::lock_guard lock(mutex_);
    rows_.push_back(make_metrics_json_failure(context.id, context, cancelled, unix_time_seconds()));
    while (rows_.size() > kMetricsRequestLimit) { rows_.pop_front(); }
}

void RequestHistory::set_rates(double decode_tokens_per_second,
                               double prefill_tokens_per_second) {
    if (!(decode_tokens_per_second >= 0.0) || !(prefill_tokens_per_second >= 0.0)) { return; }
    std::lock_guard lock(mutex_);
    decode_rate_  = decode_tokens_per_second;
    prefill_rate_ = prefill_tokens_per_second;
    rates_valid_  = true;
}

std::vector<MetricsJsonRequest> RequestHistory::rows() const {
    std::lock_guard lock(mutex_);
    return {rows_.begin(), rows_.end()};
}

bool RequestHistory::rates(double& decode, double& prefill) const {
    std::lock_guard lock(mutex_);
    decode  = decode_rate_;
    prefill = prefill_rate_;
    return rates_valid_;
}

} // namespace infernix::serve
