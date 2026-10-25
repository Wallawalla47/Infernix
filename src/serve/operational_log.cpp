#include "serve/operational_log.h"

#include "product/log_colour/log_colour.h"
#include "product/logging/pretty_format.h"
#include "product/speculative_options.h"

#include <spdlog/logger.h>

#include <algorithm>
#include <atomic>
#include <iomanip>
#include <sstream>
#include <string>
#include <utility>

namespace infernix::serve {
namespace {

std::atomic<bool>& colours_enabled() {
    static std::atomic<bool> enabled{false};
    return enabled;
}

const char* phase_name(RequestFailurePhase phase) noexcept {
    switch (phase) {
    case RequestFailurePhase::Prepare:
        return "prepare";
    case RequestFailurePhase::Generation:
        return "generation";
    case RequestFailurePhase::ResponseRender:
        return "response rendering";
    case RequestFailurePhase::ResponseStore:
        return "response storage";
    case RequestFailurePhase::Transport:
        return "transport";
    case RequestFailurePhase::Http:
        return "http";
    }
    return "unknown";
}

const char* classification_name(RequestFailureClass classification) noexcept {
    switch (classification) {
    case RequestFailureClass::ClientInput:
        return "client input";
    case RequestFailureClass::ClientDisconnected:
        return "client disconnected";
    case RequestFailureClass::Overload:
        return "overload";
    case RequestFailureClass::Timeout:
        return "timeout";
    case RequestFailureClass::Unavailable:
        return "unavailable";
    case RequestFailureClass::Upstream:
        return "upstream error";
    case RequestFailureClass::Internal:
        return "internal error";
    }
    return "unknown";
}

OperationalSeverity failure_severity(RequestFailureClass classification) noexcept {
    switch (classification) {
    case RequestFailureClass::ClientInput:
    case RequestFailureClass::ClientDisconnected:
        return OperationalSeverity::Info;
    case RequestFailureClass::Overload:
    case RequestFailureClass::Timeout:
    case RequestFailureClass::Unavailable:
    case RequestFailureClass::Upstream:
        return OperationalSeverity::Warning;
    case RequestFailureClass::Internal:
        return OperationalSeverity::Error;
    }
    return OperationalSeverity::Error;
}

const char* finish_reason_text(infernix::FinishReason reason) noexcept {
    switch (reason) {
    case infernix::FinishReason::None:
        return "none";
    case infernix::FinishReason::OutputLimit:
        return "output limit";
    case infernix::FinishReason::ContextCapacity:
        return "context capacity";
    case infernix::FinishReason::StopToken:
        return "stop token";
    case infernix::FinishReason::StopString:
        return "stop string";
    case infernix::FinishReason::Cancelled:
        return "cancelled";
    }
    return "unknown";
}

const char* prefix_reuse_path_name(infernix::PrefixReusePath path) noexcept {
    switch (path) {
    case infernix::PrefixReusePath::Root:
        return "root";
    case infernix::PrefixReusePath::HybridEndpoint:
        return "hybrid_endpoint";
    case infernix::PrefixReusePath::HybridSnapshot:
        return "hybrid_snapshot";
    }
    return "unknown";
}

std::string_view requested_effort_name(const RequestLogContext& context) noexcept {
    if (context.requested_reasoning_effort)
        return requested_reasoning_effort_name(*context.requested_reasoning_effort);
    return "template default";
}

const char* protocol_name(std::string_view protocol) noexcept {
    if (protocol == "openai_chat_completions") { return "openai-chat"; }
    if (protocol == "openai_responses") { return "openai-responses"; }
    if (protocol == "anthropic_messages") { return "anthropic"; }
    if (protocol == "openai_responses_input_tokens") { return "openai-input-tokens"; }
    if (protocol == "anthropic_count_tokens") { return "anthropic-count-tokens"; }
    if (protocol == "decide") { return "decide"; }
    return "http";
}

const char* kv_capacity_mode_name(infernix::KvCapacityMode mode) noexcept {
    return mode == infernix::KvCapacityMode::Automatic ? "auto" : "explicit";
}

void append_clause(std::ostringstream& out, std::string_view clause) { out << " | " << clause; }

void append_counted_clause(std::ostringstream& out, std::string_view label, std::uint64_t count) {
    out << " | " << label << ' ' << product::format_pretty_count(count);
}

std::string pretty_code(std::string_view code) {
    std::string result;
    result.reserve(code.size());
    for (const char ch : code) { result.push_back(ch == '_' ? ' ' : ch); }
    return result;
}

template <class T>
T monotonic_delta(T previous, T current) noexcept {
    return current >= previous ? current - previous : T{};
}

std::uint64_t host_active_ns(const ThroughputReport& report) noexcept {
    const infernix::RuntimeHostWorkStats& previous = report.previous.host_work;
    const infernix::RuntimeHostWorkStats& current  = report.current.host_work;
    return monotonic_delta(previous.engine_boundary_ns, current.engine_boundary_ns) +
           monotonic_delta(previous.program_submit_ns, current.program_submit_ns) +
           monotonic_delta(previous.program_post_ns, current.program_post_ns) +
           monotonic_delta(previous.engine_commit_output_ns, current.engine_commit_output_ns) +
           monotonic_delta(previous.engine_maintenance_ns, current.engine_maintenance_ns);
}

void append_failure_fields(std::ostringstream& out, const RequestFailure& failure) {
    if (failure.http_status != 0) { out << " | HTTP " << failure.http_status; }
    if (!failure.error_code.empty()) {
        append_clause(out, pretty_code(failure.error_code));
    } else {
        append_clause(out, classification_name(failure.classification));
    }
}

} // namespace

void set_operational_log_colours(bool enabled) { colours_enabled().store(enabled); }

bool operational_log_colours_enabled() { return colours_enabled().load(); }

OperationalRecord render_request_start(const RequestLogContext& context) {
    std::ostringstream out;
    out << "req#" << context.id << " started | " << protocol_name(context.protocol) << ' '
        << (context.stream ? "stream" : "non-stream") << " | "
        << product::format_pretty_count(context.message_count)
        << (context.message_count == 1 ? " message" : " messages") << " | max output "
        << product::format_pretty_count(
               static_cast<std::uint64_t>(std::max(context.requested_output_tokens, 0)));
    out << " | thinking ";
    if (context.enable_thinking) {
        out << requested_effort_name(context);
        if (context.thinking_budget) {
            out << ", budget " << product::format_pretty_count(*context.thinking_budget);
        }
    } else {
        out << "off";
    }
    if (context.media_item_count != 0) {
        append_counted_clause(out, "media", context.media_item_count);
        if (context.preparation.seconds > 0.0) {
            out << ", prepared " << product::format_pretty_duration(context.preparation.seconds);
        }
    }
    if (context.tool_count != 0) { append_counted_clause(out, "tools", context.tool_count); }
    if (context.preserve_thinking == true) { append_clause(out, "preserve thinking"); }
    return {.severity = OperationalSeverity::Info, .message = out.str()};
}

OperationalRecord render_request_rejected(const RequestRejectionLogContext& context) {
    const RequestFailure failure =
        make_request_failure(RequestFailurePhase::Prepare, context.error);
    std::ostringstream out;
    const char* status = "failed";
    if (failure.classification == RequestFailureClass::ClientDisconnected) {
        status = "cancelled";
    } else if (failure.classification == RequestFailureClass::ClientInput ||
               failure.classification == RequestFailureClass::Overload) {
        status = "rejected";
    }
    out << "req#" << context.id << ' ' << status << " during prepare | "
        << protocol_name(context.protocol) << ' ' << (context.stream ? "stream" : "non-stream");
    append_failure_fields(out, failure);
    append_counted_clause(out, "messages", context.message_count);
    if (context.media_item_count != 0) {
        append_counted_clause(out, "media", context.media_item_count);
    }
    if (context.tool_count != 0) { append_counted_clause(out, "tools", context.tool_count); }
    return {.severity = failure_severity(failure.classification), .message = out.str()};
}

OperationalRecord render_request_done(const RequestLogContext& context,
                                      const GenerationOutcome& outcome) {
    const double computed_prefill_tokens = static_cast<double>(outcome.computed_prefill_tokens);
    const double decode_tokens =
        outcome.completion_tokens > 0 ? static_cast<double>(outcome.completion_tokens - 1) : 0.0;
    std::ostringstream out;
    out << "req#" << context.id << " done | " << protocol_name(context.protocol) << " | ";
    if (outcome.tool_calls.empty()) {
        out << finish_reason_text(outcome.finish_reason);
    } else {
        out << "tool calls " << product::format_pretty_count(outcome.tool_calls.size());
    }
    out << " | prompt "
        << product::format_pretty_count(
               static_cast<std::uint64_t>(std::max(outcome.prompt_tokens, 0)))
        << " | output "
        << product::format_pretty_count(
               static_cast<std::uint64_t>(std::max(outcome.completion_tokens, 0)));
    const double cache_ratio = outcome.prompt_tokens > 0
                                   ? static_cast<double>(outcome.reused_prompt_tokens) /
                                         static_cast<double>(outcome.prompt_tokens)
                                   : 0.0;
    out << " | cache " << product::format_pretty_count(outcome.reused_prompt_tokens) << " ("
        << product::format_pretty_percent(cache_ratio);
    if (outcome.prefix_reuse_path != infernix::PrefixReusePath::Root) {
        out << ", " << prefix_reuse_path_name(outcome.prefix_reuse_path);
    }
    out << ") | TTFT " << product::format_pretty_duration(outcome.timings.first_token_seconds)
        << " | total " << product::format_pretty_duration(outcome.timings.total_seconds);
    if (outcome.engine_timing.queue_wait_seconds >= 0.01) {
        out << " | queue "
            << product::format_pretty_duration(outcome.engine_timing.queue_wait_seconds);
    }
    if (outcome.timings.prefill_seconds > 0.0) {
        out << " | prefill "
            << product::format_pretty_rate(
                   computed_prefill_tokens / outcome.timings.prefill_seconds, "tok");
    }
    if (outcome.timings.decode_seconds > 0.0) {
        out << " | decode "
            << product::format_pretty_rate(decode_tokens / outcome.timings.decode_seconds, "tok");
    }
    if (outcome.speculative.drafted_tokens != 0) {
        const double acceptance = static_cast<double>(outcome.speculative.accepted_tokens) /
                                  static_cast<double>(outcome.speculative.drafted_tokens);
        out << " | "
            << (outcome.speculative.ngram_rounds != 0
                    ? "mixed speculation"
                    : product::speculative_backend_name(outcome.speculative.backend))
            << " accepted " << product::format_pretty_count(outcome.speculative.accepted_tokens)
            << '/' << product::format_pretty_count(outcome.speculative.drafted_tokens) << " ("
            << product::format_pretty_percent(acceptance) << ')';
    }
    if (outcome.speculative.ngram_rounds != 0) {
        out << " | ngram "
            << product::format_pretty_count(outcome.speculative.ngram_accepted_tokens) << '/'
            << product::format_pretty_count(outcome.speculative.ngram_drafted_tokens)
            << " accepted, " << product::format_pretty_count(outcome.speculative.ngram_rounds)
            << " rounds";
    }
    if (outcome.expert_cache && outcome.expert_cache->routed != 0) {
        out << " | expert cache "
            << product::format_pretty_percent(static_cast<double>(outcome.expert_cache->hits) /
                                              static_cast<double>(outcome.expert_cache->routed))
            << " hit";
        if (outcome.expert_cache->decode_routed != 0) {
            out << ", decode "
                << product::format_pretty_percent(
                       static_cast<double>(outcome.expert_cache->decode_hits) /
                       static_cast<double>(outcome.expert_cache->decode_routed));
        }
    }
    if (outcome.speculative.tree_rounds != 0) {
        out << " | trees " << product::format_pretty_count(outcome.speculative.tree_rounds)
            << " rounds, "
            << product::format_pretty_count(outcome.speculative.tree_side_accepted_tokens)
            << " side-branch drafts";
    }
    if (outcome.ngram_archive.enabled) {
        out << " | archive " << (outcome.ngram_archive.bound ? "bound" : "unbound");
        if (outcome.ngram_archive.bound) {
            out << " "
                << product::format_pretty_count(outcome.speculative.ngram_archive_accepted_tokens)
                << '/'
                << product::format_pretty_count(outcome.speculative.ngram_archive_drafted_tokens)
                << " accepted, gen " << outcome.ngram_archive.generation << ", "
                << product::format_pretty_count(outcome.ngram_archive.sources) << " sources";
        }
    }
    if (outcome.thinking.configured_budget) {
        out << " | thinking "
            << product::format_pretty_count(outcome.thinking.model_thinking_tokens) << '/'
            << product::format_pretty_count(*outcome.thinking.configured_budget);
        if (outcome.thinking.injected_tokens != 0) {
            out << ", control " << product::format_pretty_count(outcome.thinking.injected_tokens);
        }
    }
    if (outcome.thinking.loop_detected) {
        out << " | reasoning loop after "
            << product::format_pretty_count(outcome.thinking.loop_thinking_tokens) << " thinking tokens ("
            << static_cast<int>(outcome.thinking.loop_coverage * 100.0F + 0.5F) << "% repeated), "
            << (outcome.thinking.applied ? "concluded" : "stopped");
    }
    if (outcome.timings.vision_offload_window_seconds > 0.0) {
        out << " vision_offload_ms=" << outcome.timings.vision_offload_window_seconds * 1000.0
            << " vision_offload_evicted_mib="
            << outcome.timings.vision_offload_evicted_bytes / (1024 * 1024)
            << " vision_offload_evict_ms=" << outcome.timings.vision_offload_evict_seconds * 1000.0
            << " vision_offload_restore_ms="
            << outcome.timings.vision_offload_restore_seconds * 1000.0
            << " vision_offload_staged_mib="
            << outcome.timings.vision_offload_staged_bytes / (1024 * 1024);
    }
    return {.severity = OperationalSeverity::Info, .message = out.str()};
}

std::optional<OperationalRecord> render_tool_call_fallback(const RequestLogContext& context,
                                                           const GenerationOutcome& outcome) {
    const infernix::ToolCallParseFallbackReason reason = outcome.tool_call_parse.fallback_reason;
    if (outcome.tool_call_parse.recovered_from_reasoning) {
        return OperationalRecord{
            .severity = OperationalSeverity::Info,
            .message  = "req#" + std::to_string(context.id) +
                        " tool calls recovered from unclosed thinking | calls=" +
                        std::to_string(outcome.tool_call_parse.structured_call_count),
        };
    }
    if (!outcome.tool_call_parse.marker_seen ||
        reason == infernix::ToolCallParseFallbackReason::None) {
        return std::nullopt;
    }
    // Tolerant recovery retained structured calls after discarding a trailing suffix, or after
    // keeping a call whose closing tags were cut off at the region end. That is a successful
    // parse, not a fallback-to-text failure; note it as informational transparency. When the
    // tail truncation kept no calls, the region was returned as text and falls through to the
    // warning record below.
    if (reason == infernix::ToolCallParseFallbackReason::TruncatedTail &&
        outcome.tool_call_parse.structured_call_count > 0) {
        return OperationalRecord{
            .severity = OperationalSeverity::Info,
            .message  = "req#" + std::to_string(context.id) +
                        " tolerated tool-call suffix discarded | " +
                        pretty_code(infernix::tool_call_parse_fallback_reason_name(reason)),
        };
    }
    // The reason names the verdict; a bounded, single-line snippet of the returned markup shows
    // what earned it. Text before the first marker never appears.
    constexpr std::size_t kMarkupSnippetBytes = 240;
    std::string snippet;
    if (const std::size_t marker = outcome.text.find("<tool_call>");
        marker != std::string::npos) {
        snippet = outcome.text.substr(marker, kMarkupSnippetBytes);
        if (outcome.text.size() - marker > kMarkupSnippetBytes) { snippet += "..."; }
        for (char& byte : snippet) {
            if (byte == '\n' || byte == '\r' || byte == '\t') { byte = ' '; }
        }
    }

    return OperationalRecord{
        .severity = OperationalSeverity::Warning,
        .message  = "req#" + std::to_string(context.id) + " tool markup returned as text | " +
                   pretty_code(infernix::tool_call_parse_fallback_reason_name(reason)) +
                   (snippet.empty() ? std::string{} : " | " + snippet),
    };
}

OperationalRecord render_request_failure(const RequestLogContext& context,
                                         const RequestFailure& failure) {
    std::ostringstream out;
    out << "req#" << context.id
        << (failure.classification == RequestFailureClass::ClientDisconnected ? " cancelled during "
                                                                              : " failed during ")
        << phase_name(failure.phase) << " | " << protocol_name(context.protocol);
    append_failure_fields(out, failure);
    return {.severity = failure_severity(failure.classification), .message = out.str()};
}

OperationalRecord render_response_failure(std::uint64_t request_id, const RequestFailure& failure) {
    std::ostringstream out;
    out << "req#" << request_id << " response failed during " << phase_name(failure.phase);
    append_failure_fields(out, failure);
    return {.severity = failure_severity(failure.classification), .message = out.str()};
}

OperationalRecord render_throughput(const ThroughputReport& report) {
    const double prefill_rate =
        report.interval_seconds > 0.0
            ? static_cast<double>(report.computed_prefill_tokens) / report.interval_seconds
            : 0.0;
    const double decode_rate =
        report.interval_seconds > 0.0
            ? static_cast<double>(report.committed_decode_tokens) / report.interval_seconds
            : 0.0;
    std::ostringstream out;
    out << "throughput | " << product::format_pretty_duration(report.interval_seconds);
    if (report.computed_prefill_tokens != 0) {
        out << " | prefill " << product::format_pretty_rate(prefill_rate, "tok") << " ("
            << product::format_pretty_count(report.computed_prefill_tokens) << " tok)";
    }
    if (report.committed_decode_tokens != 0) {
        out << " | decode " << product::format_pretty_rate(decode_rate, "tok") << " ("
            << product::format_pretty_count(report.committed_decode_tokens) << " tok)";
    }
    out << " | running " << report.current.running_requests;
    if (report.current.prefilling_requests != 0 || report.current.decode_ready_requests != 0) {
        out << " (";
        bool separator = false;
        if (report.current.prefilling_requests != 0) {
            out << "prefill " << report.current.prefilling_requests;
            separator = true;
        }
        if (report.current.decode_ready_requests != 0) {
            if (separator) { out << ", "; }
            out << "decode-ready " << report.current.decode_ready_requests;
        }
        out << ')';
    }
    if (report.current.waiting_requests != 0) {
        out << " | waiting " << report.current.waiting_requests;
    }
    if (report.current.paused_requests != 0) {
        out << " | paused " << report.current.paused_requests;
    }
    if (report.current.replaying_requests != 0) {
        out << " | replaying " << report.current.replaying_requests;
    }
    if (report.current.materializing_requests != 0) {
        out << " | materializing " << report.current.materializing_requests;
    }
    if (report.current.terminal_pending_requests != 0) {
        out << " | terminal-pending " << report.current.terminal_pending_requests;
    }
    if (report.decode_rounds != 0) {
        out << " | batch " << std::fixed << std::setprecision(2)
            << static_cast<double>(report.decode_row_rounds) /
                   static_cast<double>(report.decode_rounds);
    }
    const double host_seconds = static_cast<double>(host_active_ns(report)) * 1.0e-9;
    const double host_ratio =
        report.interval_seconds > 0.0 ? host_seconds / report.interval_seconds : 0.0;
    out << " | host " << product::format_pretty_percent(host_ratio) << " ("
        << product::format_pretty_duration(host_seconds) << ')';
    // The device at the interval's end and its energy over it; the clock reasons other than idle.
    if (const auto& gpu = report.gpu_current; gpu.valid) {
        const GpuInterval interval = gpu_interval(report);
        out << " | gpu " << gpu.sm_clock_mhz << '/' << gpu.memory_clock_mhz << " MHz " << std::fixed
            << std::setprecision(0) << gpu.power_watts << " W " << gpu.temperature_c << " C";
        if (interval.average_power_watts) { out << " avg " << *interval.average_power_watts << " W"; }
        if (interval.joules_per_decode_token) {
            out << ' ' << std::setprecision(2) << *interval.joules_per_decode_token << " J/tok";
        }
        if (const std::uint64_t limiting = gpu.clock_event_reasons & ~std::uint64_t{1}; limiting != 0) {
            out << " limited " << gpu_clock_event_reason_names(limiting);
        }
        if (interval.power_violation_fraction && *interval.power_violation_fraction > 0.0) {
            out << " power-capped " << product::format_pretty_percent(*interval.power_violation_fraction);
        }
        if (interval.thermal_violation_fraction && *interval.thermal_violation_fraction > 0.0) {
            out << " thermal " << product::format_pretty_percent(*interval.thermal_violation_fraction);
        }
    }
    return {.severity = OperationalSeverity::Info, .message = out.str()};
}

std::optional<OperationalRecord> render_output_limit_note(std::uint32_t max_context,
                                                          int default_max_tokens) {
    if (default_max_tokens <= 0 ||
        max_context < static_cast<std::uint64_t>(default_max_tokens) * kOutputLimitNoteRatio) {
        return std::nullopt;
    }
    std::ostringstream out;
    out << "output limit | a request that sets none stops after "
        << product::format_pretty_count(static_cast<std::uint64_t>(default_max_tokens))
        << " tokens of its " << product::format_pretty_count(max_context)
        << " context (--default-max-tokens)";
    return OperationalRecord{.severity = OperationalSeverity::Info, .message = out.str()};
}

OperationalLog::OperationalLog(std::shared_ptr<spdlog::logger> logger)
    : logger_(std::move(logger)) {}

void OperationalLog::write(OperationalRecord record) const {
    // Colour the message's key=value tokens when the serve was started with
    // --log-colours on; the [timestamp] [level] prefix from the sink pattern
    // stays plain.
    const std::string message =
        infernix::product::log_colour::colourise_stats_line(record.message,
                                                          operational_log_colours_enabled());
    switch (record.severity) {
    case OperationalSeverity::Info:
        logger_->info("{}", message);
        return;
    case OperationalSeverity::Warning:
        logger_->warn("{}", message);
        return;
    case OperationalSeverity::Error:
        logger_->error("{}", message);
        return;
    }
}

void OperationalLog::request_start(const RequestLogContext& context) const {
    write(render_request_start(context));
}

void OperationalLog::request_rejected(const RequestRejectionLogContext& context) const {
    write(render_request_rejected(context));
}

void OperationalLog::request_done(const RequestLogContext& context,
                                  const GenerationOutcome& outcome) const {
    write(render_request_done(context, outcome));
    if (std::optional<OperationalRecord> fallback = render_tool_call_fallback(context, outcome)) {
        write(std::move(*fallback));
    }
}

void OperationalLog::request_failure(const RequestLogContext& context,
                                     const RequestFailure& failure) const {
    write(render_request_failure(context, failure));
}

void OperationalLog::response_failure(std::uint64_t request_id,
                                      const RequestFailure& failure) const {
    write(render_response_failure(request_id, failure));
}

void OperationalLog::throughput(const ThroughputReport& report) const {
    write(render_throughput(report));
}

void OperationalLog::http_failure(std::string_view endpoint, const RequestFailure& failure,
                                  std::string_view request_id) const {
    std::ostringstream out;
    out << "HTTP request failed during " << phase_name(failure.phase) << " | "
        << protocol_name(endpoint);
    if (!request_id.empty()) { out << " | request " << product::format_pretty_text(request_id); }
    append_failure_fields(out, failure);
    write({.severity = failure_severity(failure.classification), .message = out.str()});
}

void OperationalLog::engine_capacity(const GenerationService& service) const {
    const infernix::MemorySummary memory            = service.memory_summary();
    const infernix::EngineOptions& engine           = service.engine_options();
    const infernix::ContextCacheOptions& cache      = engine.context_cache;
    const infernix::ContextCostSummary context_cost = service.load_summary().context_cost;

    logger_->info("capacity | {} tokens of context across all requests ({} KV)",
                  product::format_pretty_count(memory.kv_capacity),
                  infernix::kv_cache_cli_name(memory.kv_cache));
    logger_->debug("capacity | KV {} tokens, {}, {} | pages {}/{} | runtime {} | free {}",
                   product::format_pretty_count(memory.kv_capacity),
                   infernix::kv_cache_cli_name(memory.kv_cache),
                   kv_capacity_mode_name(memory.kv_capacity_mode),
                   product::format_pretty_count(memory.kv_capacity_page_groups),
                   product::format_pretty_count(memory.kv_capacity_max_page_groups),
                   product::format_pretty_bytes(memory.runtime_reservation_bytes),
                   product::format_pretty_bytes(memory.available_after_startup_bytes));
    if (const auto note =
            render_output_limit_note(engine.max_context, service.options().default_max_tokens)) {
        logger_->info("{}", note->message);
    }

    if (cache.enabled) {
        logger_->info("prefix cache | on | {} in RAM", product::format_pretty_bytes(*cache.host_capacity_bytes));
    } else {
        logger_->info("prefix cache | off");
    }
    logger_->debug("context | history {} | {} active + {} snapshot device states | host {}",
                   cache.enabled ? "on" : "off", engine.max_concurrency,
                   *cache.hybrid.device_snapshot_slots,
                   product::format_pretty_bytes(*cache.host_capacity_bytes));

    if (product::draft_tree_enabled(engine.speculative)) {
        logger_->info("draft trees | {} | up to {} paths",
                      product::draft_tree_nodes_text(engine.speculative),
                      engine.speculative.draft_tree_paths);
    }

    if (service.options().enable_vision) {
        const infernix::MediaCacheSummary media = service.media_cache_summary();
        logger_->info("media | {} preprocess workers | cache {} | live {}",
                      media.preprocess_threads, product::format_pretty_bytes(media.capacity_bytes),
                      product::format_pretty_bytes(media.live_capacity_bytes));
    }

    logger_->debug("memory ledger | after weights {} | after startup {} | headroom {} | slack {} | "
                   "CUDA graphs {} allowed, {} used",
                   product::format_pretty_bytes(memory.available_after_weights_bytes),
                   product::format_pretty_bytes(memory.available_after_startup_bytes),
                   product::format_pretty_bytes(memory.vram_headroom_bytes),
                   product::format_pretty_bytes(memory.planned_slack_bytes),
                   product::format_pretty_bytes(memory.cuda_graph_allowance_bytes),
                   product::format_pretty_bytes(memory.cuda_graph_measured_bytes));
    if (memory.cuda_graph_measured_bytes > memory.cuda_graph_allowance_bytes) {
        // The KV pool was sized against the allowance, so the excess came out of the slack left
        // beside it; with no headroom the next allocation can fail.
        logger_->warn("CUDA graphs used {} but the KV sizing allowed {}; leave more room with "
                      "--vram-headroom-mib or a smaller --kv-capacity",
                      product::format_pretty_bytes(memory.cuda_graph_measured_bytes),
                      product::format_pretty_bytes(memory.cuda_graph_allowance_bytes));
    }
    logger_->debug("context cost | transfer {} | prefill {} | hardware {} | prefill signature {}",
                   infernix::context_cost_preset_source_name(context_cost.transfer_source),
                   infernix::context_cost_preset_source_name(context_cost.prefill_source),
                   product::format_pretty_text(context_cost.hardware_class),
                   product::format_pretty_text(context_cost.prefill_signature));
}

void OperationalLog::warmup_started() const { logger_->debug("warming up"); }

void OperationalLog::warmup_complete(double seconds) const {
    if (seconds >= 0.25) {
        logger_->info("warmup complete | {}", product::format_pretty_duration(seconds));
    } else {
        logger_->debug("warmup complete | {}", product::format_pretty_duration(seconds));
    }
}

void OperationalLog::warmup_failure(double seconds, std::string_view detail) const {
    logger_->critical("warmup failed | {} | {}", product::format_pretty_duration(seconds),
                      product::format_pretty_text(detail));
}

void OperationalLog::bind_failure(std::string_view host, int port) const {
    logger_->error("cannot bind {}:{}", product::format_pretty_text(host), port);
}

void OperationalLog::listen_failure(std::string_view host, int port) const {
    logger_->error("server listen failed | {}:{}", product::format_pretty_text(host), port);
}

void OperationalLog::server_ready(std::string_view host, int port, std::string_view model_id,
                                  bool auth_enabled) const {
    logger_->info("listening on http://{}:{} | model {} | auth {}",
                  product::format_pretty_text(host), port, product::format_pretty_text(model_id),
                  auth_enabled ? "bearer" : "disabled");
    if (!auth_enabled && !is_loopback_host(host)) {
        logger_->warn("{} is reachable beyond this machine without --api-key: anyone who can reach port {} "
                      "can use the model",
                      product::format_pretty_text(host), port);
    }
}

void OperationalLog::server_stopped() const { logger_->info("server stopped"); }

void OperationalLog::server_failure(bool serving, std::string_view detail) const {
    logger_->critical("server failed during {} | {}", serving ? "serving" : "startup",
                      product::format_pretty_text(detail));
}

void OperationalLog::engine_failure_exit(int status) const {
    logger_->critical("inference engine failed; server stopped, exit status {} (restart to serve again)",
                      status);
}

} // namespace infernix::serve
