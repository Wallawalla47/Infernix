#include "serve/metrics_json.h"

#include "product/speculative_options.h"

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

Json ratio(std::uint64_t part, std::uint64_t whole) {
    return whole == 0 ? Json(nullptr)
                      : Json(static_cast<double>(part) / static_cast<double>(whole));
}

Json rate(std::uint64_t tokens, double seconds) {
    return seconds > 0.0 ? Json(static_cast<double>(tokens) / seconds) : Json(nullptr);
}

// One panel row as numbers: the same sums and the same ratios the console prints.
// One `memory` part: listed when it holds memory (or, with `count`, experts).
void add_part(Json& parts, const char* key, std::uint64_t bytes, std::optional<std::uint64_t> used = std::nullopt,
              std::optional<std::uint64_t> count = std::nullopt) {
    if (bytes == 0 && count.value_or(0) == 0) { return; }
    Json part{{"key", key}, {"bytes", bytes}};
    if (used) { part["used_bytes"] = std::min(*used, bytes); }
    if (count) { part["count"] = *count; }
    parts.push_back(std::move(part));
}

Json bytes_or_null(std::uint64_t bytes) { return bytes != 0 ? Json(bytes) : Json(nullptr); }

// `memory` (docs/serving.md): what the server holds on the device, in RAM and on the SSD, by part, with the totals the
// parts are shown against.
Json memory_json(const MetricsJsonSnapshot& s) {
    const infernix::MemoryUsage& m = s.stats.memory;
    Json device_parts              = Json::array();
    add_part(device_parts, "weights", m.device.weights);
    add_part(device_parts, "drafter", m.device.drafter);
    add_part(device_parts, "experts", m.device.experts, std::nullopt,
             m.device.experts != 0 ? std::optional<std::uint64_t>(m.device.expert_frames) : std::nullopt);
    add_part(device_parts, "kv_cache", m.device.kv, m.device.kv_used);
    add_part(device_parts, "state", m.device.state);
    add_part(device_parts, "prefix_cache", m.device.prefix);
    add_part(device_parts, "workspace", m.device.workspace);
    add_part(device_parts, "cuda_graphs", m.device.cuda_graphs);
    add_part(device_parts, "staging", m.device.staging);
    Json host_parts = Json::array();
    add_part(host_parts, "experts", m.host.experts, std::nullopt,
             m.host.expert_records != 0 ? std::optional<std::uint64_t>(m.host.expert_records) : std::nullopt);
    add_part(host_parts, "prefix_cache", m.host.prefix, m.host.prefix_used);
    add_part(host_parts, "weights", m.host.weights);
    add_part(host_parts, "ngram", m.host.ngram);
    add_part(host_parts, "staging", m.host.staging);
    const infernix::DeviceMemoryReading& d = s.device_memory;
    const HostMemorySnapshot& h            = s.host_memory;
    Json out{
        {"device",
         Json{{"total_bytes", bytes_or_null(d.total_bytes)},
              {"process_bytes", d.process_bytes ? Json(*d.process_bytes) : Json(nullptr)},
              {"free_bytes", d.total_bytes != 0 ? Json(d.free_bytes) : Json(nullptr)},
              {"headroom_bytes", m.vram_headroom},
              {"parts", std::move(device_parts)}}},
        {"host",
         Json{{"total_bytes", bytes_or_null(h.total_physical)},
              {"available_bytes", h.total_physical != 0 ? Json(h.available_physical) : Json(nullptr)},
              {"process_bytes", bytes_or_null(h.process_resident)},
              {"parts", std::move(host_parts)}}},
        {"ssd", nullptr},
    };
    if (m.ssd.prefix != 0 || m.ssd.expert_records != 0) {
        Json ssd_parts = Json::array();
        add_part(ssd_parts, "prefix_cache", m.ssd.prefix, m.ssd.prefix_used);
        add_part(ssd_parts, "experts", m.ssd.experts, std::nullopt, m.ssd.expert_records);
        out["ssd"] = Json{{"parts", std::move(ssd_parts)}};
    }
    return out;
}

Json stats_row(const ConsoleStatsTotals& totals) {
    const ConsoleRequestSample& s = totals.sum;
    return Json{
        {"requests", totals.requests},
        {"ttft_s", totals.requests == 0
                       ? Json(nullptr)
                       : Json(s.ttft_seconds / static_cast<double>(totals.requests))},
        {"prompt_tokens", s.prompt_tokens},
        {"cached_tokens", s.cache_hit_tokens},
        {"cache_hit_rate", ratio(s.cache_hit_tokens, s.prompt_tokens)},
        {"prefill_tokens", s.computed_prefill_tokens},
        {"prefill_s", s.prefill_seconds},
        {"prefill_tok_s", rate(s.computed_prefill_tokens, s.prefill_seconds)},
        {"decode_tokens", s.decode_tokens},
        {"decode_s", s.decode_seconds},
        {"decode_tok_s", rate(s.decode_tokens, s.decode_seconds)},
        {"batch", s.decode_seconds > 0.0 ? Json(s.decode_round_seconds / s.decode_seconds)
                                         : Json(nullptr)},
        {"drafted", s.model_drafted_tokens},
        {"accepted", s.model_accepted_tokens},
        {"acceptance", ratio(s.model_accepted_tokens, s.model_drafted_tokens)},
        {"accepted_per_round",
         s.model_rounds == 0 ? Json(nullptr)
                             : Json(static_cast<double>(s.model_accepted_tokens) /
                                    static_cast<double>(s.model_rounds))},
        {"ngram_drafted", s.ngram_drafted_tokens},
        {"ngram_accepted", s.ngram_accepted_tokens},
        {"ngram_acceptance", ratio(s.ngram_accepted_tokens, s.ngram_drafted_tokens)},
        {"ngram_rounds", s.ngram_rounds},
        {"archive_drafted", s.archive_drafted_tokens},
        {"archive_accepted", s.archive_accepted_tokens},
        {"archive_acceptance", ratio(s.archive_accepted_tokens, s.archive_drafted_tokens)},
        {"expert_lookups", s.routed_experts},
        {"expert_hits", s.vram_expert_hits},
        {"expert_hit_rate", ratio(s.vram_expert_hits, s.routed_experts)},
    };
}

Json session_stats_json(const ConsoleStatsSnapshot& s) {
    return Json{
        {"recent_window", s.recent_window},
        {"drafter", s.speculative_backend == SpeculativeBackend::None
                        ? Json(nullptr)
                        : Json(product::speculative_backend_name(s.speculative_backend))},
        {"ngram_enabled", s.ngram_enabled},
        {"completed", s.completed},
        {"failed", s.failed},
        {"cancelled", s.cancelled},
        {"rejected", s.rejected},
        {"session", stats_row(s.session)},
        {"recent", stats_row(s.recent)},
    };
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

const char* active_request_phase_name(ActiveRequestPhase phase) noexcept {
    switch (phase) {
    case ActiveRequestPhase::Queued:
        return "queued";
    case ActiveRequestPhase::Prefill:
        return "prefill";
    case ActiveRequestPhase::Decode:
        return "decode";
    }
    return "queued";
}

MetricsJsonRequest make_metrics_json_request(std::uint64_t id, const RequestLogContext& context,
                                             const GenerationOutcome& outcome,
                                             double time_unix) {
    const auto prompt_tokens = static_cast<std::uint64_t>(std::max(outcome.prompt_tokens, 0));
    MetricsJsonRequest row;
    row.id                      = id;
    row.time_unix               = time_unix;
    row.label                   = context.label;
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
    row.label     = context.label;
    row.protocol  = context.protocol;
    row.finish    = cancelled ? "cancelled" : "failed";
    return row;
}

std::string render_metrics_json(const MetricsJsonSnapshot& s) {
    Json active = Json::array();
    for (const MetricsJsonActiveRequest& a : s.active) {
        active.push_back(Json{
            {"id", a.id},
            {"label", a.label.empty() ? Json(nullptr) : Json(a.label)},
            {"protocol", a.protocol},
            {"phase", active_request_phase_name(a.phase)},
            {"prompt_tokens", a.prompt_tokens},
            {"cached_tokens", a.cached_tokens},
            {"processed_tokens", a.processed_tokens},
            {"output_tokens", a.output_tokens},
            {"elapsed_ms", a.elapsed_ms},
        });
    }
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
            {"label", r.label.empty() ? Json(nullptr) : Json(r.label)},
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

    Json root{
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
              {"progress_owner", s.stats.progress_owner_request_id == 0
                                     ? Json(nullptr)
                                     : Json(s.stats.progress_owner_request_id)},
              {"no_progress_s", s.stats.no_progress_seconds},
              {"no_progress_recoveries", s.stats.no_progress_recoveries},
              {"device_memory_recoveries", s.stats.device_memory_recoveries},
              {"prompt_tokens", nullptr},
              {"tok_s", number_or_null(s.rates_valid, s.decode_tokens_per_second)},
              {"tok_s_mean", number_or_null(s.rates_valid, s.decode_tokens_per_second)},
              {"prefill_tok_s_mean", number_or_null(s.rates_valid, s.prefill_tokens_per_second)},
              {"start_time", s.start_time_unix},
              {"instance_id", s.instance_id},
              {"active", std::move(active)}}},
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
        {"memory", memory_json(s)},
    };
    if (s.session_stats) { root["session_stats"] = session_stats_json(*s.session_stats); }
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

void LiveRequests::begin(std::uint64_t id, std::string label, std::string protocol,
                         std::uint32_t prompt_tokens, Clock::time_point started) {
    Entry entry;
    entry.label                  = std::move(label);
    entry.protocol               = std::move(protocol);
    entry.started                = started == Clock::time_point{} ? Clock::now() : started;
    entry.position.prompt_tokens = prompt_tokens;
    std::lock_guard lock(mutex_);
    entries_.insert_or_assign(id, std::move(entry));
}

void LiveRequests::progress(std::uint64_t id,
                            const infernix::GenerationProgressObservation& observation) {
    std::lock_guard lock(mutex_);
    const auto found = entries_.find(id);
    if (found == entries_.end()) { return; }
    found->second.admitted = true;
    found->second.position = observation;
}

void LiveRequests::end(std::uint64_t id) noexcept {
    try {
        std::lock_guard lock(mutex_);
        entries_.erase(id);
    } catch (...) {}
}

std::vector<MetricsJsonActiveRequest> LiveRequests::rows(Clock::time_point now) const {
    std::vector<MetricsJsonActiveRequest> out;
    std::lock_guard lock(mutex_);
    out.reserve(entries_.size());
    for (const auto& [id, entry] : entries_) {
        const infernix::GenerationProgressObservation& p = entry.position;
        MetricsJsonActiveRequest row;
        row.id            = id;
        row.label         = entry.label;
        row.protocol      = entry.protocol;
        row.phase         = !entry.admitted              ? ActiveRequestPhase::Queued
                            : p.generated_tokens != 0 ? ActiveRequestPhase::Decode
                                                      : ActiveRequestPhase::Prefill;
        row.prompt_tokens = p.prompt_tokens;
        row.cached_tokens = entry.admitted ? std::min(p.reused_prompt_tokens, p.prompt_tokens) : 0U;
        // A generating request's prompt is done whatever its last prefill observation said.
        row.processed_tokens = row.phase == ActiveRequestPhase::Decode
                                   ? p.prompt_tokens
                                   : std::min(p.processed_prompt_tokens, p.prompt_tokens);
        row.output_tokens    = p.generated_tokens;
        row.elapsed_ms       = std::max(
            0.0, std::chrono::duration<double, std::milli>(now - entry.started).count());
        out.push_back(std::move(row));
    }
    return out;
}

} // namespace infernix::serve
