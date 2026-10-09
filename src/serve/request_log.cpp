#include "serve/request_log.h"
#include "product/constraint_observation.h"
#include "product/logging/pretty_format.h"
#include "product/speculative_options.h"

#include <spdlog/logger.h>

#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

namespace infernix::serve {
namespace {

using Json = nlohmann::json;

// The log is a text-mode stream: each '\n' is written as "\r\n" on Windows.
#ifdef _WIN32
constexpr std::uint64_t kLineTerminatorBytes = 2;
#else
constexpr std::uint64_t kLineTerminatorBytes = 1;
#endif

template <class T>
T monotonic_delta(T previous, T current) noexcept {
    return current >= previous ? current - previous : T{};
}

std::uint64_t unix_time_ms() {
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(now).count());
}

std::string new_server_instance_id() {
    const auto now    = std::chrono::system_clock::now().time_since_epoch();
    const auto micros = std::chrono::duration_cast<std::chrono::microseconds>(now).count();
#ifdef _WIN32
    const auto process_id = ::_getpid();
#else
    const auto process_id = ::getpid();
#endif
    return "serve-" + std::to_string(static_cast<long long>(process_id)) + '-' +
           std::to_string(micros);
}

std::filesystem::path normalized_absolute_path(const std::string& value) {
    std::error_code error;
    std::filesystem::path path = std::filesystem::weakly_canonical(value, error);
    if (!error) { return path; }
    error.clear();
    path = std::filesystem::absolute(value, error);
    return error ? std::filesystem::path(value).lexically_normal() : path.lexically_normal();
}

std::string cuda_version_string(int version) {
    if (version <= 0) { return {}; }
    return std::to_string(version / 1000) + '.' + std::to_string((version % 1000) / 10);
}

std::string cuda_uuid_string(const cudaUUID_t& uuid) {
    std::ostringstream out;
    out << "GPU-" << std::hex << std::setfill('0');
    for (int i = 0; i < 16; ++i) {
        if (i == 4 || i == 6 || i == 8 || i == 10) { out << '-'; }
        out << std::setw(2) << static_cast<unsigned int>(static_cast<unsigned char>(uuid.bytes[i]));
    }
    return out.str();
}

const char* finish_reason_name(infernix::FinishReason reason) {
    switch (reason) {
    case infernix::FinishReason::None:
        return "none";
    case infernix::FinishReason::OutputLimit:
        return "output_limit";
    case infernix::FinishReason::ContextCapacity:
        return "context_capacity";
    case infernix::FinishReason::StopToken:
        return "stop_token";
    case infernix::FinishReason::StopString:
        return "stop_string";
    case infernix::FinishReason::Cancelled:
        return "cancelled";
    }
    return "unknown";
}

Json tool_call_parse_json(const infernix::ToolCallParseDiagnostics& diagnostics) {
    return Json{{"marker_seen", diagnostics.marker_seen},
                {"structured_call_count", diagnostics.structured_call_count},
                {"empty_arguments_omitted", diagnostics.empty_arguments_omitted},
                {"schema_mismatch_arguments", diagnostics.schema_mismatch_arguments},
                {"duplicate_parameters_repaired", diagnostics.duplicate_parameters_repaired},
                {"fallback_reason",
                 infernix::tool_call_parse_fallback_reason_name(diagnostics.fallback_reason)},
                {"tolerant_recovered", diagnostics.tolerant_recovered},
                {"recovered_from_reasoning", diagnostics.recovered_from_reasoning}};
}

std::string tool_choice_name(const ToolChoice& choice) {
    switch (choice.mode) {
    case ToolChoiceMode::Auto:
        return "auto";
    case ToolChoiceMode::None:
        return "none";
    }
    return "unknown";
}

Json requested_reasoning_effort_json(const std::optional<RequestedReasoningEffort>& requested) {
    return requested ? Json(std::string(requested_reasoning_effort_name(*requested)))
                     : Json(nullptr);
}

const char* kv_cache_name(infernix::KvCacheStorage storage) {
    switch (storage) {
    case infernix::KvCacheStorage::BFloat16:
        return "bf16";
    case infernix::KvCacheStorage::Int8Group64:
        return "int8-group64";
    case infernix::KvCacheStorage::Fp8E4M3Row256:
        return "fp8-e4m3-row256";
    case infernix::KvCacheStorage::Nvfp4Group16:
        return "nvfp4";
    case infernix::KvCacheStorage::Fp8KeyNvfp4Value:
        return "k8v4";
    case infernix::KvCacheStorage::Vq2:
        return "vq2";
    case infernix::KvCacheStorage::Q4KeyVq2Value:
        return "k4v2";
    }
    return "unknown";
}

const char* kv_capacity_mode_name(infernix::KvCapacityMode mode) {
    return mode == infernix::KvCapacityMode::Automatic ? "auto" : "explicit";
}

const char* proposal_head_name(infernix::ProposalHead proposal) {
    return proposal == infernix::ProposalHead::Optimized ? "optimized" : "full";
}

const char* prefix_reuse_path_name(infernix::PrefixReusePath path) {
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

Json event_base(const std::string& server_instance_id, std::uint64_t timestamp, const char* event) {
    return Json{{"artifact_type", kRequestLogArtifactType},
                {"schema_version", kRequestLogSchemaVersion},
                {"event", event},
                {"timestamp_unix_ms", timestamp},
                {"server_instance_id", server_instance_id}};
}

const char* scheduling_transition_name(infernix::GenerationSchedulingTransition transition) {
    using Transition = infernix::GenerationSchedulingTransition;
    switch (transition) {
    case Transition::PauseStarted:
        return "pause_started";
    case Transition::Paused:
        return "paused";
    case Transition::RestoreStarted:
        return "restore_started";
    case Transition::Restored:
        return "restored";
    case Transition::ReplayComplete:
        return "replay_complete";
    case Transition::RecoveryComplete:
        return "recovery_complete";
    case Transition::Terminal:
        return "terminal";
    }
    throw std::logic_error("invalid scheduling transition");
}

Json recovery_route_json(infernix::GenerationRecoveryRoute route) {
    switch (route) {
    case infernix::GenerationRecoveryRoute::None:
        return nullptr;
    case infernix::GenerationRecoveryRoute::Replay:
        return "replay";
    }
    throw std::logic_error("invalid recovery route");
}

Json sampler_json(const infernix::ResolvedSamplingParameters& sampling) {
    return Json{{"temperature", sampling.temperature},
                {"top_p", sampling.top_p},
                {"top_k", sampling.top_k},
                {"min_p", sampling.min_p},
                {"presence_penalty", sampling.presence_penalty},
                {"frequency_penalty", sampling.frequency_penalty},
                {"seed", sampling.seed}};
}

Json preset_json(const infernix::SamplingPreset& preset) {
    return Json{{"temperature", preset.temperature},
                {"top_p", preset.top_p},
                {"top_k", preset.top_k},
                {"min_p", preset.min_p},
                {"presence_penalty", preset.presence_penalty},
                {"frequency_penalty", preset.frequency_penalty}};
}

Json overrides_json(const infernix::SamplingOverrides& overrides) {
    Json result{{"temperature", nullptr},
                {"top_p", nullptr},
                {"top_k", nullptr},
                {"min_p", nullptr},
                {"presence_penalty", nullptr},
                {"frequency_penalty", nullptr},
                {"seed", nullptr}};
    if (overrides.temperature) { result["temperature"] = *overrides.temperature; }
    if (overrides.top_p) { result["top_p"] = *overrides.top_p; }
    if (overrides.top_k) { result["top_k"] = *overrides.top_k; }
    if (overrides.min_p) { result["min_p"] = *overrides.min_p; }
    if (overrides.presence_penalty) { result["presence_penalty"] = *overrides.presence_penalty; }
    if (overrides.frequency_penalty) { result["frequency_penalty"] = *overrides.frequency_penalty; }
    if (overrides.seed) { result["seed"] = *overrides.seed; }
    return result;
}

Json request_json(const RequestLogContext& context) {
    Json thinking_budget = nullptr;
    if (context.thinking_budget) { thinking_budget = *context.thinking_budget; }
    return Json{{"request_id", context.id},
                {"http_request_id", context.http_request_id},
                {"response_id", context.response_id},
                {"protocol", context.protocol},
                {"model", context.model},
                {"stream", context.stream},
                {"message_count", context.message_count},
                {"media_item_count", context.media_item_count},
                {"requested_output_tokens", context.requested_output_tokens},
                {"requested_output_tokens_source",
                 context.requested_output_tokens_client_set ? "client" : "server_default"},
                {"tool_count", context.tool_count},
                {"tool_choice", tool_choice_name(context.tool_choice)},
                {"has_tool_history", context.has_tool_history},
                {"enable_thinking", context.enable_thinking},
                {"thinking_budget", std::move(thinking_budget)},
                {"requested_reasoning_effort",
                 requested_reasoning_effort_json(context.requested_reasoning_effort)},
                {"preserve_thinking",
                 context.preserve_thinking ? Json(*context.preserve_thinking) : Json(nullptr)},
                {"preserve_thinking_semantic_change", context.preserve_thinking_semantic_change},
                {"sampling", sampler_json(context.sampling)}};
}

Json preparation_json(const RequestLogContext& context) {
    const PromptPreparationStats& stats = context.preparation;
    return Json{{"total", stats.seconds},
                {"acquisition", context.acquisition_seconds},
                {"media_preprocess", stats.media_preprocess_seconds},
                {"media_preprocess_work", stats.media_preprocess_work_seconds},
                {"tokenize", stats.tokenize_seconds},
                {"media_items", stats.media_items},
                {"media_bytes", stats.media_bytes},
                {"raw_patches", stats.raw_patches},
                {"vision_tokens", stats.vision_tokens},
                {"patch_bytes", stats.patch_bytes},
                {"cache_hits", stats.media_cache_hits},
                {"cache_misses", stats.media_cache_misses},
                {"singleflight_waits", stats.media_singleflight_waits},
                {"built_patch_bytes", stats.built_patch_bytes},
                {"reused_patch_bytes", stats.reused_patch_bytes}};
}

Json rejected_request_json(const RequestRejectionLogContext& context) {
    return Json{{"request_id", context.id},
                {"http_request_id", context.http_request_id},
                {"response_id", context.response_id},
                {"protocol", context.protocol},
                {"model", context.model},
                {"stream", context.stream},
                {"message_count", context.message_count},
                {"media_item_count", context.media_item_count},
                {"requested_output_tokens", context.requested_output_tokens},
                {"requested_output_tokens_source",
                 context.requested_output_tokens_client_set ? "client" : "server_default"},
                {"tool_count", context.tool_count},
                {"tool_choice", tool_choice_name(context.tool_choice)},
                {"has_tool_history", context.has_tool_history},
                {"requested_reasoning_effort",
                 requested_reasoning_effort_json(context.requested_reasoning_effort)}};
}

Json error_json(const ApiError& error) {
    Json code  = error.code.empty() ? Json(nullptr) : Json(error.code);
    Json param = error.param.empty() ? Json(nullptr) : Json(error.param);
    return Json{{"status", error.status},
                {"type", error.type},
                {"code", std::move(code)},
                {"param", std::move(param)},
                {"message", error.message}};
}

Json arena_json(const infernix::ArenaMemorySummary& arena) {
    return Json{{"capacity_bytes", arena.capacity_bytes},
                {"used_bytes", arena.used_bytes},
                {"peak_used_bytes", arena.peak_used_bytes}};
}

Json vision_workspace_json(const std::optional<infernix::VisionWorkspaceMemorySummary>& vision) {
    if (!vision) { return nullptr; }
    return Json{{"aggregate_prompt_tokens", vision->aggregate_prompt_tokens},
                {"max_item_tokens", vision->max_item_tokens},
                {"general_capacity_bytes", vision->general_capacity_bytes},
                {"encode_peak_bytes", vision->encode_peak_bytes},
                {"handoff_offset_bytes", vision->handoff_offset_bytes},
                {"handoff_capacity_bytes", vision->handoff_capacity_bytes},
                {"handoff_active_bytes", vision->handoff_active_bytes},
                {"handoff_peak_bytes", vision->handoff_peak_bytes}};
}

Json speculative_json(const GenerationMetrics& metrics) {
    return Json{{"backend", product::speculative_backend_name(metrics.speculative_backend)},
                {"draft_window", metrics.speculative_draft_window},
                {"rounds", metrics.speculative_rounds},
                {"drafted_tokens", metrics.speculative_draft_tokens},
                {"accepted_tokens", metrics.speculative_accepted_tokens},
                {"fallback_steps", metrics.speculative_fallback_steps},
                {"accepted_per_position", metrics.speculative_accepted_per_position},
                {"ngram_rounds", metrics.ngram_rounds},
                {"ngram_drafted_tokens", metrics.ngram_drafted_tokens},
                {"ngram_accepted_tokens", metrics.ngram_accepted_tokens},
                {"ngram_archive_rounds", metrics.ngram_archive_rounds},
                {"ngram_archive_drafted_tokens", metrics.ngram_archive_drafted_tokens},
                {"ngram_archive_accepted_tokens", metrics.ngram_archive_accepted_tokens},
                {"tree_rounds", metrics.tree_rounds},
                {"tree_side_rounds", metrics.tree_side_rounds},
                {"tree_side_accepted_tokens", metrics.tree_side_accepted_tokens},
                {"ngram_archive",
                 {{"enabled", metrics.ngram_archive.enabled},
                  {"bound", metrics.ngram_archive.bound},
                  {"published", metrics.ngram_archive.published},
                  {"generation", metrics.ngram_archive.generation},
                  {"sources", metrics.ngram_archive.sources},
                  {"session_bytes", metrics.ngram_archive.session_bytes},
                  {"total_bytes", metrics.ngram_archive.total_bytes},
                  {"sampling_seed", metrics.ngram_archive.sampling_seed
                                        ? Json(*metrics.ngram_archive.sampling_seed)
                                        : Json(nullptr)}}}};
}

Json scheduling_json(const infernix::GenerationSchedulingStats& stats) {
    return Json{{"preemptions", stats.preemptions},
                {"snapshot_restores", stats.snapshot_restores},
                {"replay_restores", stats.replay_restores},
                {"replayed_tokens", stats.replayed_tokens},
                {"paused_ns", stats.paused_ns}};
}

Json admission_json(const infernix::GenerationAdmissionStats& stats) {
    const char* reason = nullptr;
    switch (stats.fallback_reason) {
    case infernix::AdmissionFallbackReason::None:
        reason = "none";
        break;
    case infernix::AdmissionFallbackReason::SourceInvalid:
        reason = "source_invalid";
        break;
    case infernix::AdmissionFallbackReason::CapacityLimit:
        reason = "capacity_limit";
        break;
    case infernix::AdmissionFallbackReason::IsolatedCapacity:
        reason = "isolated_capacity";
        break;
    }
    if (!reason) { throw std::logic_error("invalid admission fallback reason"); }
    return Json{{"preferred_reused_tokens", stats.preferred_reused_tokens},
                {"source_wait_seconds", stats.source_wait_seconds},
                {"fallback_reason", reason}};
}

double nanoseconds_to_seconds(std::uint64_t value) noexcept {
    return static_cast<double>(value) * 1.0e-9;
}

double nanoseconds_to_microseconds(std::uint64_t value) noexcept {
    return static_cast<double>(value) * 1.0e-3;
}

double request_host_exposed_seconds(const infernix::GenerationEngineTiming& timing) noexcept {
    return timing.engine_boundary_exposed_seconds + timing.program_submit_exposed_seconds +
           timing.program_post_exposed_seconds + timing.engine_commit_output_exposed_seconds +
           timing.engine_maintenance_exposed_seconds;
}

Json request_engine_timing_json(const infernix::GenerationEngineTiming& timing) {
    return Json{
        {"queue_wait_seconds", timing.queue_wait_seconds},
        {"host_exposed_seconds",
         Json{{"engine_boundary", timing.engine_boundary_exposed_seconds},
              {"program_submit", timing.program_submit_exposed_seconds},
              {"program_post", timing.program_post_exposed_seconds},
              {"engine_commit_output", timing.engine_commit_output_exposed_seconds},
              {"engine_maintenance", timing.engine_maintenance_exposed_seconds},
              {"total", request_host_exposed_seconds(timing)}}},
        {"device_wait_exposed_seconds", timing.device_wait_exposed_seconds},
        {"constraint_draft_wait_exposed_seconds", timing.constraint_draft_wait_exposed_seconds},
        {"decode", Json{{"host_exposed_seconds", timing.decode_host_exposed_seconds},
                        {"device_wait_exposed_seconds", timing.decode_device_wait_exposed_seconds},
                        {"rounds", timing.decode_rounds}}},
        {"units", Json{{"prefill", timing.prefill_units}, {"control", timing.control_units}}},
    };
}

Json work_timing_json(const infernix::GenerationWorkTiming& timing) {
    return Json{{"submit_seconds", timing.submit_seconds},
                {"wait_seconds", timing.wait_seconds},
                {"post_seconds", timing.post_seconds},
                {"gpu_seconds", timing.gpu_seconds}};
}

Json first_output_timing_json(const infernix::GenerationFirstOutputTiming& timing) {
    return Json{{"elapsed_seconds", timing.elapsed_seconds},
                {"initial_binding_seconds", timing.initial_binding_seconds},
                {"engine", request_engine_timing_json(timing.engine)},
                {"prefill", work_timing_json(timing.prefill)},
                {"replay", work_timing_json(timing.replay)},
                {"scheduling", scheduling_json(timing.scheduling)},
                {"computed_prefill_tokens", timing.computed_prefill_tokens}};
}

infernix::RuntimeHostWorkStats host_work_delta(const infernix::RuntimeHostWorkStats& previous,
                                             const infernix::RuntimeHostWorkStats& current) {
    return infernix::RuntimeHostWorkStats{
        .engine_boundary_ns =
            monotonic_delta(previous.engine_boundary_ns, current.engine_boundary_ns),
        .program_submit_ns = monotonic_delta(previous.program_submit_ns, current.program_submit_ns),
        .program_post_ns   = monotonic_delta(previous.program_post_ns, current.program_post_ns),
        .engine_commit_output_ns =
            monotonic_delta(previous.engine_commit_output_ns, current.engine_commit_output_ns),
        .engine_maintenance_ns =
            monotonic_delta(previous.engine_maintenance_ns, current.engine_maintenance_ns),
        .device_wait_ns = monotonic_delta(previous.device_wait_ns, current.device_wait_ns),
        .constraint_draft_wait_ns =
            monotonic_delta(previous.constraint_draft_wait_ns, current.constraint_draft_wait_ns),
        .decode_host_ns = monotonic_delta(previous.decode_host_ns, current.decode_host_ns),
        .decode_device_wait_ns =
            monotonic_delta(previous.decode_device_wait_ns, current.decode_device_wait_ns),
        .prefill_host_ns = monotonic_delta(previous.prefill_host_ns, current.prefill_host_ns),
        .prefill_device_wait_ns =
            monotonic_delta(previous.prefill_device_wait_ns, current.prefill_device_wait_ns),
        .control_host_ns = monotonic_delta(previous.control_host_ns, current.control_host_ns),
        .control_device_wait_ns =
            monotonic_delta(previous.control_device_wait_ns, current.control_device_wait_ns),
        .prefill_units = monotonic_delta(previous.prefill_units, current.prefill_units),
        .control_units = monotonic_delta(previous.control_units, current.control_units),
        .stats_publication_ns =
            monotonic_delta(previous.stats_publication_ns, current.stats_publication_ns),
        .stats_publication_invocations = monotonic_delta(previous.stats_publication_invocations,
                                                         current.stats_publication_invocations),
    };
}

std::uint64_t host_active_ns(const infernix::RuntimeHostWorkStats& timing) noexcept {
    return timing.engine_boundary_ns + timing.program_submit_ns + timing.program_post_ns +
           timing.engine_commit_output_ns + timing.engine_maintenance_ns;
}

Json microseconds_per(std::uint64_t nanoseconds, std::uint64_t count) {
    if (count == 0) { return nullptr; }
    return nanoseconds_to_microseconds(nanoseconds) / static_cast<double>(count);
}

} // namespace

std::string format_server_start_json(
    const std::string& server_instance_id, std::uint64_t timestamp, const ServeOptions& options,
    const infernix::EngineOptions& engine_options,
    const infernix::ModelSamplingDefaults& sampling_defaults, const std::string& public_model_id,
    const infernix::LoadSummary& load, const infernix::MemorySummary& memory,
    const ServerLogEnvironment& environment, std::optional<std::uint64_t> artifact_size_bytes) {
    Json record = event_base(server_instance_id, timestamp, "server_start");

    Json artifact_size = nullptr;
    if (artifact_size_bytes.has_value()) { artifact_size = *artifact_size_bytes; }
    Json default_thinking_budget = nullptr;
    if (options.default_thinking_budget) {
        default_thinking_budget = *options.default_thinking_budget;
    }

    record["server"] =
        Json{{"host", options.host},
             {"port", options.port},
             {"public_model_id", public_model_id},
             {"api_key_configured", !options.api_key.empty()},
             {"cors_enabled", options.enable_cors},
             {"max_request_bytes", options.max_request_bytes},
             {"media_cache_bytes", options.media_cache_bytes},
             {"media_live_bytes", options.media_live_bytes},
             {"media_preprocess_threads", options.media_preprocess_threads},
             {"request_log_jsonl", options.request_log_jsonl},
             {"default_output_tokens", options.default_max_tokens},
             {"default_thinking",
              options.enable_thinking ? Json(*options.enable_thinking) : Json(nullptr)},
             {"default_thinking_budget", std::move(default_thinking_budget)},
             {"reasoning_loop", reasoning_loop_name(options.reasoning_loop)},
             {"default_preserve_thinking",
              options.preserve_thinking ? Json(*options.preserve_thinking) : Json(nullptr)}};
    record["artifact"]                             = Json{{"path", options.artifact_path},
                                                          {"size_bytes", std::move(artifact_size)},
                                                          {"architecture", load.architecture},
                                                          {"name", load.model_name},
                                                          {"formats", load.weight_formats},
                                                          {"prefill_signature", load.prefill_signature},
                                                          {"bytes_read", load.artifact_bytes_read},
                                                          {"host_to_device_bytes", load.host_to_device_bytes},
                                                          {"peak_staging_bytes", load.peak_staging_bytes},
                                                          {"device_object_count", load.device_object_count},
                                                          {"host_object_count", load.host_object_count},
                                                          {"load_seconds", load.load_seconds},
                                                          {"upload_seconds", load.upload_seconds}};
    const infernix::ContextCacheOptions& cache       = engine_options.context_cache;
    const infernix::ContextCostSummary& context_cost = load.context_cost;
    const std::uint64_t total_device_state_slots =
        static_cast<std::uint64_t>(engine_options.max_concurrency) +
        cache.hybrid.device_snapshot_slots.value();
    record["engine"] =
        Json{{"device", engine_options.device},
             {"max_context", engine_options.max_context},
             {"kv_capacity_mode", kv_capacity_mode_name(memory.kv_capacity_mode)},
             {"kv_capacity", memory.kv_capacity},
             {"kv_capacity_page_groups", memory.kv_capacity_page_groups},
             {"kv_capacity_max_page_groups", memory.kv_capacity_max_page_groups},
             {"max_concurrency", engine_options.max_concurrency},
             {"max_pending_requests", engine_options.max_pending_requests},
             {"pending_timeout_ms", engine_options.pending_timeout_ms},
             {"prefill_chunk", engine_options.prefill_chunk},
             {"original_int8_prefill_kernel", engine_options.original_int8_prefill_kernel},
             {"prefill_8bit_pv", prefill_pv8_name(engine_options.prefill_8bit_pv)},
             {"prefill_split_workspace_mib", engine_options.prefill_split_workspace_mib},
             {"original_nvfp4_prefill_kernel", engine_options.original_nvfp4_prefill_kernel},
             {"log_stats_interval_ms", options.log_stats_interval_ms},
             {"kv_cache", kv_cache_name(engine_options.kv_cache)},
             {"vision", engine_options.enable_vision},
             {"cuda_graph", engine_options.use_cuda_graph},
             {"prefix_reuse", options.allow_prefix_reuse},
             {"speculative_backend",
              product::speculative_backend_name(engine_options.speculative.backend)},
             {"speculative_draft_window", engine_options.speculative.draft_tokens},
             {"draft_tree_nodes", engine_options.speculative.draft_tree_nodes},
             {"draft_tree_auto", engine_options.speculative.draft_tree_auto},
             {"draft_tree_paths", engine_options.speculative.draft_tree_paths},
             {"ngram_draft_window", engine_options.speculative.ngram_draft_tokens},
             {"ngram_min_match", engine_options.speculative.ngram_min_match},
             {"ngram_archive_bytes", engine_options.speculative.ngram_archive_bytes},
             {"ngram_session_bytes", engine_options.speculative.ngram_session_bytes},
             {"ngram_native_sessions", options.ngram_native_sessions},
             {"proposal_head", proposal_head_name(engine_options.speculative.proposal_head)},
             {"context_cost", Json{{"transfer_source", infernix::context_cost_preset_source_name(
                                                           context_cost.transfer_source)},
                                   {"prefill_source", infernix::context_cost_preset_source_name(
                                                          context_cost.prefill_source)},
                                   {"hardware_class", context_cost.hardware_class},
                                   {"prefill_signature", context_cost.prefill_signature},
                                   {"preset_path", context_cost.preset_path.string()}}},
             {"context_cache", Json{{"enabled", cache.enabled},
                                    {"device_snapshot_slots",
                                     cache.hybrid.device_snapshot_slots.value()},
                                    {"total_device_state_slots", total_device_state_slots},
                                    {"host_capacity_bytes", cache.host_capacity_bytes.value()}}}};
    record["sampling_defaults"] =
        Json{{"thinking", preset_json(sampling_defaults.thinking)},
             {"non_thinking", preset_json(sampling_defaults.non_thinking)},
             {"server_overrides", overrides_json(options.sampling_overrides)},
             {"omitted_seed", "random"},
             {"greedy", options.greedy}};
    record["memory"] =
        Json{{"weights", arena_json(memory.weights)},
             {"sequence", arena_json(memory.sequence)},
             {"workspace", arena_json(memory.workspace)},
             {"vision_workspace", vision_workspace_json(memory.vision_workspace)},
             {"minimum_runtime_reservation_bytes", memory.minimum_runtime_reservation_bytes},
             {"kv_capacity_increment_bytes", memory.kv_capacity_increment_bytes},
             {"runtime_reservation_bytes", memory.runtime_reservation_bytes},
             {"available_after_weights_bytes", memory.available_after_weights_bytes},
             {"available_after_startup_bytes", memory.available_after_startup_bytes},
             {"vram_headroom_bytes", memory.vram_headroom_bytes},
             {"planned_slack_bytes", memory.planned_slack_bytes},
             {"cuda_graph_allowance_bytes", memory.cuda_graph_allowance_bytes},
             {"cuda_graph_measured_bytes", memory.cuda_graph_measured_bytes},
             {"kv_payload_bytes", memory.kv_payload_bytes}};
    record["environment"] =
        Json{{"device", environment.device},
             {"gpu_name", environment.gpu_name},
             {"gpu_uuid", environment.gpu_uuid},
             {"total_device_memory_bytes", environment.total_device_memory_bytes},
             {"compute_capability_major", environment.compute_capability_major},
             {"compute_capability_minor", environment.compute_capability_minor},
             {"cuda_compile_version", environment.cuda_compile_version},
             {"cuda_runtime_version", environment.cuda_runtime_version},
             {"cuda_driver_version", environment.cuda_driver_version}};
    record["argv"] = options.startup_argv;
    return record.dump();
}

std::string
format_request_scheduling_json(const std::string& server_instance_id,
                               std::uint64_t timestamp_unix_ms, std::uint64_t request_id,
                               const std::string& http_request_id,
                               const infernix::GenerationSchedulingObservation& observation) {
    Json record       = event_base(server_instance_id, timestamp_unix_ms, "request_scheduling");
    record["request"] = Json{{"request_id", request_id}, {"http_request_id", http_request_id}};
    record["engine_request_id"] = observation.engine_request_id;
    record["transition"]        = scheduling_transition_name(observation.transition);
    record["route"]             = recovery_route_json(observation.route);
    record["steady_ns"]         = observation.steady_ns;
    record["elapsed_ns"]        = observation.elapsed_ns;
    record["preemption_index"]  = observation.preemption_index;
    record["progress"]          = Json{
                 {"global_prefill_tokens", observation.global_prefill_tokens},
                 {"global_decode_tokens", observation.global_decode_tokens},
                 {"global_replayed_tokens", observation.global_replayed_tokens},
                 {"request_prefill_tokens", observation.request_prefill_tokens},
                 {"request_decode_tokens", observation.request_decode_tokens},
                 {"request_replayed_tokens", observation.request_replayed_tokens},
    };
    return record.dump();
}

std::string format_request_start_json(const std::string& server_instance_id,
                                      std::uint64_t timestamp, const RequestLogContext& context) {
    Json record                   = event_base(server_instance_id, timestamp, "request_start");
    record["request"]             = request_json(context);
    record["preparation_seconds"] = preparation_json(context);
    return record.dump();
}

std::string format_request_rejected_json(const std::string& server_instance_id,
                                         std::uint64_t timestamp,
                                         const RequestRejectionLogContext& context) {
    Json record       = event_base(server_instance_id, timestamp, "request_rejected");
    record["phase"]   = "prepare";
    record["request"] = rejected_request_json(context);
    record["error"]   = error_json(context.error);
    return record.dump();
}

std::string format_request_done_json(const std::string& server_instance_id, std::uint64_t timestamp,
                                     const RequestLogContext& context,
                                     const GenerationOutcome& outcome) {
    Json record       = event_base(server_instance_id, timestamp, "request_done");
    record["request"] = request_json(context);
    record["result"] =
        Json{{"finish_reason", finish_reason_name(outcome.finish_reason)},
             {"prompt_tokens", outcome.prompt_tokens},
             {"completion_tokens", outcome.completion_tokens},
             {"generated_token_ids", outcome.generated_token_ids},
             {"computed_prefill_tokens", outcome.metrics.computed_prefill_tokens},
             {"prefix_cache_hit_tokens", outcome.metrics.prefix_cache_hit_tokens},
             {"prefix_reuse_path", prefix_reuse_path_name(outcome.metrics.prefix_reuse_path)},
             {"thinking_budget", outcome.thinking.configured_budget
                                     ? Json(*outcome.thinking.configured_budget)
                                     : Json(nullptr)},
             {"model_thinking_tokens", outcome.thinking.model_thinking_tokens},
             {"thinking_control_tokens", outcome.thinking.injected_tokens},
             {"thinking_control_applied", outcome.thinking.applied},
             {"reasoning_loop_detected", outcome.thinking.loop_detected},
             {"reasoning_loop_thinking_tokens", outcome.thinking.loop_thinking_tokens},
             {"reasoning_loop_coverage", outcome.thinking.loop_coverage},
             {"tool_call_count", outcome.tool_calls.size()},
             {"tool_call_parse", tool_call_parse_json(outcome.tool_call_parse)}};
    record["timings_seconds"] = Json{
        {"prepare", outcome.metrics.prepare_seconds}, {"ttft", outcome.metrics.ttft_seconds},
        {"vision", outcome.metrics.vision_seconds},   {"prefill", outcome.metrics.prefill_seconds},
        {"decode", outcome.metrics.decode_seconds},   {"total", outcome.metrics.total_seconds}};
    record["engine_timing"] = request_engine_timing_json(outcome.metrics.engine_timing);
    record["constraint"]    = product::constraint_observation_json(outcome.constraint);
    record["first_output_timing"] =
        outcome.metrics.first_output_timing
            ? first_output_timing_json(*outcome.metrics.first_output_timing)
            : Json(nullptr);
    record["speculative"] = speculative_json(outcome.metrics);
    record["generation"]  = Json{{"engine_request_id", outcome.metrics.engine_request_id},
                                 {"scheduling", scheduling_json(outcome.metrics.scheduling)},
                                 {"admission", admission_json(outcome.metrics.admission)}};
    return record.dump();
}

std::string format_request_error_json(const std::string& server_instance_id,
                                      std::uint64_t timestamp, const RequestLogContext& context,
                                      const std::string& message) {
    Json record       = event_base(server_instance_id, timestamp, "request_error");
    record["request"] = request_json(context);
    record["error"]   = Json{{"message", message}};
    return record.dump();
}

std::string format_throughput_json(const std::string& server_instance_id, std::uint64_t timestamp,
                                   const ThroughputReport& report) {
    Json record                          = event_base(server_instance_id, timestamp, "throughput");
    const infernix::RuntimeStats& previous = report.previous;
    const infernix::RuntimeStats& current  = report.current;
    const double prefill_rate =
        report.interval_seconds > 0.0
            ? static_cast<double>(report.computed_prefill_tokens) / report.interval_seconds
            : 0.0;
    const double decode_rate =
        report.interval_seconds > 0.0
            ? static_cast<double>(report.committed_decode_tokens) / report.interval_seconds
            : 0.0;
    Json average_batch = nullptr;
    if (report.decode_rounds != 0) {
        average_batch = static_cast<double>(report.decode_row_rounds) /
                        static_cast<double>(report.decode_rounds);
    }
    const infernix::RuntimeHostWorkStats host =
        host_work_delta(previous.host_work, current.host_work);
    const std::uint64_t active_host = host_active_ns(host);
    record["interval_seconds"]      = report.interval_seconds;
    record["final_interval"]        = report.final_interval;
    record["tokens"]                = Json{{"computed_prefill", report.computed_prefill_tokens},
                                           {"committed_decode", report.committed_decode_tokens}};
    record["throughput_tokens_per_second"] =
        Json{{"prefill", prefill_rate}, {"decode", decode_rate}};
    record["scheduler"]  = Json{{"running", current.running_requests},
                                {"prefilling", current.prefilling_requests},
                                {"decode_ready", current.decode_ready_requests},
                                {"waiting", current.waiting_requests},
                                {"paused", current.paused_requests},
                                {"replaying", current.replaying_requests},
                                {"materializing", current.materializing_requests},
                                {"terminal_pending", current.terminal_pending_requests}};
    record["scheduling"] = Json{
        {"preemptions", monotonic_delta(previous.preemptions, current.preemptions)},
        {"snapshot_restores",
         monotonic_delta(previous.snapshot_restores, current.snapshot_restores)},
        {"replay_restores", monotonic_delta(previous.replay_restores, current.replay_restores)},
        {"replayed_tokens", monotonic_delta(previous.replayed_tokens, current.replayed_tokens)}};
    record["decode_batch"] = Json{{"rounds", report.decode_rounds},
                                  {"row_rounds", report.decode_row_rounds},
                                  {"average_size", std::move(average_batch)}};
    record["host_work"]    = Json{
           {"elapsed_seconds",
            Json{{"engine_boundary", nanoseconds_to_seconds(host.engine_boundary_ns)},
                 {"program_submit", nanoseconds_to_seconds(host.program_submit_ns)},
                 {"program_post", nanoseconds_to_seconds(host.program_post_ns)},
                 {"engine_commit_output", nanoseconds_to_seconds(host.engine_commit_output_ns)},
                 {"engine_maintenance", nanoseconds_to_seconds(host.engine_maintenance_ns)},
                 {"total", nanoseconds_to_seconds(active_host)}}},
           {"device_wait_seconds", nanoseconds_to_seconds(host.device_wait_ns)},
           {"constraint_draft_wait_seconds", nanoseconds_to_seconds(host.constraint_draft_wait_ns)},
           {"work_class_seconds",
            Json{{"decode_host", nanoseconds_to_seconds(host.decode_host_ns)},
                 {"decode_device_wait", nanoseconds_to_seconds(host.decode_device_wait_ns)},
                 {"prefill_host", nanoseconds_to_seconds(host.prefill_host_ns)},
                 {"prefill_device_wait", nanoseconds_to_seconds(host.prefill_device_wait_ns)},
                 {"control_host", nanoseconds_to_seconds(host.control_host_ns)},
                 {"control_device_wait", nanoseconds_to_seconds(host.control_device_wait_ns)}}},
           {"detail_subset_seconds",
            Json{{"stats_publication", nanoseconds_to_seconds(host.stats_publication_ns)}}},
           {"detail_invocations", Json{{"stats_publication", host.stats_publication_invocations}}},
           {"units", Json{{"prefill", host.prefill_units}, {"control", host.control_units}}},
           {"decode_host_microseconds_per_round",
            microseconds_per(host.decode_host_ns, report.decode_rounds)},
           {"decode_host_microseconds_per_row_round",
            microseconds_per(host.decode_host_ns, report.decode_row_rounds)},
           {"decode_device_wait_microseconds_per_round",
            microseconds_per(host.decode_device_wait_ns, report.decode_rounds)},
           {"detail_microseconds_per_invocation",
            Json{{"stats_publication",
                  microseconds_per(host.stats_publication_ns, host.stats_publication_invocations)}}},
    };
    record["context_cache"] = Json{
        {"selections",
         Json{{"root", monotonic_delta(previous.root_selections, current.root_selections)},
              {"prefix", monotonic_delta(previous.prefix_selections, current.prefix_selections)},
              {"reused_prompt_tokens",
               monotonic_delta(previous.reused_prompt_tokens, current.reused_prompt_tokens)}}},
        {"last_selection", Json{{"frontier_tokens", current.last_selected_frontier_tokens}}},
        {"state_operations",
         Json{{"forks", monotonic_delta(previous.state_forks, current.state_forks)},
              {"materialization_forks", monotonic_delta(previous.materialization_state_forks,
                                                        current.materialization_state_forks)},
              {"restores", monotonic_delta(previous.state_restores, current.state_restores)}}},
        {"partial_tail_cow_pages",
         monotonic_delta(previous.partial_tail_cow_pages, current.partial_tail_cow_pages)},
        {"occupancy", Json{{"device_state_slots", current.device_state_occupied_slots},
                           {"device_main_kv_pages", current.device_main_kv_occupied_pages},
                           {"device_backend_kv_pages", current.device_backend_kv_occupied_pages}}}};
    // Hybrid prefix cache (the serving default): absolute occupancy, per-interval events.
    if (current.hybrid_snapshots != 0 || current.hybrid_tree_blocks != 0 ||
        current.hybrid_blocks_inserted != 0) {
        const auto delta = [&](std::uint64_t RuntimeStats::* field) {
            return monotonic_delta(previous.*field, current.*field);
        };
        record["context_cache"]["hybrid"] =
            Json{{"device_blocks", current.hybrid_cached_blocks},
                 {"evictable_blocks", current.hybrid_evictable_blocks},
                 {"tree_blocks", current.hybrid_tree_blocks},
                 {"snapshots", current.hybrid_snapshots},
                 {"host_capacity_bytes", current.hybrid_host_capacity_bytes},
                 {"host_used_bytes", current.hybrid_host_used_bytes},
                 {"snapshot_hits", delta(&RuntimeStats::hybrid_snapshot_hits)},
                 {"reused_tokens", delta(&RuntimeStats::hybrid_reused_tokens)},
                 {"blocks_inserted", delta(&RuntimeStats::hybrid_blocks_inserted)},
                 {"blocks_reattached", delta(&RuntimeStats::hybrid_blocks_reattached)},
                 {"blocks_duplicate", delta(&RuntimeStats::hybrid_blocks_duplicate)},
                 {"taps_created", delta(&RuntimeStats::hybrid_taps_created)},
                 {"taps_skipped", delta(&RuntimeStats::hybrid_taps_skipped)},
                 {"endpoints_created", delta(&RuntimeStats::hybrid_endpoints_created)},
                 {"host_image_writes", delta(&RuntimeStats::hybrid_host_image_writes)},
                 {"host_block_writes", delta(&RuntimeStats::hybrid_host_block_writes)},
                 {"host_image_restores", delta(&RuntimeStats::hybrid_host_image_restores)},
                 {"host_block_restores", delta(&RuntimeStats::hybrid_host_block_restores)},
                 {"prefetched_blocks", delta(&RuntimeStats::hybrid_prefetched_blocks)},
                 {"host_write_bytes", delta(&RuntimeStats::hybrid_host_write_bytes)},
                 {"host_restore_bytes", delta(&RuntimeStats::hybrid_host_restore_bytes)},
                 {"evicted_blocks", delta(&RuntimeStats::hybrid_evicted_blocks)},
                 {"host_snapshot_evictions", delta(&RuntimeStats::hybrid_host_snapshot_evictions)},
                 {"host_dead_reclaims", delta(&RuntimeStats::hybrid_host_dead_reclaims)},
                 {"unbacked_node_losses", delta(&RuntimeStats::hybrid_unbacked_node_losses)},
                 {"held_snapshots", current.hybrid_held_snapshots},
                 {"held_device_evictions", delta(&RuntimeStats::hybrid_held_device_evictions)},
                 {"held_snapshot_losses", delta(&RuntimeStats::hybrid_held_snapshot_losses)},
                 {"held_host_refusals", delta(&RuntimeStats::hybrid_held_host_refusals)}};
    }
    return record.dump();
}

ServerLogEnvironment query_server_log_environment(int device) {
    ServerLogEnvironment environment;
    environment.device               = device;
    environment.cuda_compile_version = cuda_version_string(CUDART_VERSION);

    int runtime_version = 0;
    if (cudaRuntimeGetVersion(&runtime_version) == cudaSuccess) {
        environment.cuda_runtime_version = cuda_version_string(runtime_version);
    }
    int driver_version = 0;
    if (cudaDriverGetVersion(&driver_version) == cudaSuccess) {
        environment.cuda_driver_version = cuda_version_string(driver_version);
    }
    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, device) == cudaSuccess) {
        environment.gpu_name                  = properties.name;
        environment.gpu_uuid                  = cuda_uuid_string(properties.uuid);
        environment.total_device_memory_bytes = properties.totalGlobalMem;
        environment.compute_capability_major  = properties.major;
        environment.compute_capability_minor  = properties.minor;
    }
    return environment;
}

JsonlRequestLog::JsonlRequestLog(const std::string& path,
                                 const std::string& protected_artifact_path,
                                 std::shared_ptr<spdlog::logger> logger,
                                 RequestLogRotation rotation)
    : path_(path), logger_(std::move(logger)), rotation_(rotation) {
    if (path_.empty()) { return; }
    if (rotation_.keep > kMaximumRequestLogKeep) {
        throw std::invalid_argument("request JSONL log keeps too many rotated files");
    }
    if (!protected_artifact_path.empty()) {
        const std::filesystem::path artifact = normalized_absolute_path(protected_artifact_path);
        if (normalized_absolute_path(path_) == artifact) {
            throw std::invalid_argument("request JSONL log must not overwrite the model artifact");
        }
        for (std::uint32_t index = 1; rotation_.max_bytes != 0 && index <= rotation_.keep;
             ++index) {
            if (normalized_absolute_path(rotated_path(index)) == artifact) {
                throw std::invalid_argument(
                    "a rotated request JSONL log must not overwrite the model artifact");
            }
        }
    }
    server_instance_id_ = new_server_instance_id();
    output_.open(path_, std::ios::out | std::ios::app);
    if (!output_) {
        throw std::runtime_error("failed to open request JSONL log for append: " + path_);
    }
    enabled_ = true;
    std::error_code error;
    const std::uintmax_t existing = std::filesystem::file_size(path_, error);
    written_bytes_                = error ? 0 : static_cast<std::uint64_t>(existing);
    // An existing file already at the limit is rotated before this server writes to it.
    if (rotation_.max_bytes != 0 && written_bytes_ >= rotation_.max_bytes) {
        const std::string warning = rotate_locked();
        if (!warning.empty() && logger_ != nullptr) { logger_->warn("{}", warning); }
        if (failed_) {
            throw std::runtime_error("failed to reopen request JSONL log after rotation: " + path_);
        }
    }
}

void JsonlRequestLog::write_server_start(const ServeOptions& options,
                                         const infernix::EngineOptions& engine_options,
                                         const infernix::ModelSamplingDefaults& sampling_defaults,
                                         const std::string& public_model_id,
                                         const infernix::LoadSummary& load,
                                         const infernix::MemorySummary& memory) {
    if (!enabled()) { return; }
    std::error_code error;
    const std::uintmax_t size = std::filesystem::file_size(options.artifact_path, error);
    const std::optional<std::uint64_t> artifact_size =
        error ? std::nullopt : std::optional<std::uint64_t>(size);
    std::string record = format_server_start_json(
        server_instance_id_, unix_time_ms(), options, engine_options, sampling_defaults,
        public_model_id, load, memory, query_server_log_environment(options.device), artifact_size);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        server_start_record_ = record;
    }
    append(std::move(record));
}

void JsonlRequestLog::write_request_start(const RequestLogContext& context) {
    if (!enabled()) { return; }
    append(format_request_start_json(server_instance_id_, unix_time_ms(), context));
}

void JsonlRequestLog::write_request_scheduling(
    std::uint64_t request_id, const std::string& http_request_id,
    const infernix::GenerationSchedulingObservation& observation) {
    if (!enabled()) { return; }
    append(format_request_scheduling_json(server_instance_id_, unix_time_ms(), request_id,
                                          http_request_id, observation));
}

void JsonlRequestLog::write_request_rejected(const RequestRejectionLogContext& context) {
    if (!enabled()) { return; }
    append(format_request_rejected_json(server_instance_id_, unix_time_ms(), context));
}

void JsonlRequestLog::write_request_done(const RequestLogContext& context,
                                         const GenerationOutcome& outcome) {
    if (!enabled()) { return; }
    append(format_request_done_json(server_instance_id_, unix_time_ms(), context, outcome));
}

void JsonlRequestLog::write_request_error(const RequestLogContext& context,
                                          const std::string& message) {
    if (!enabled()) { return; }
    append(format_request_error_json(server_instance_id_, unix_time_ms(), context, message));
}

void JsonlRequestLog::write_throughput(const ThroughputReport& report) {
    if (!enabled()) { return; }
    append(format_throughput_json(server_instance_id_, unix_time_ms(), report));
}

void JsonlRequestLog::append(std::string record) {
    bool report_failure = false;
    std::string rotation_warning;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (failed_) { return; }
        output_ << record << '\n';
        output_.flush();
        if (!output_) {
            failed_        = true;
            report_failure = true;
        } else {
            written_bytes_ += record.size() + kLineTerminatorBytes;
            if (rotation_.max_bytes != 0 && written_bytes_ >= rotation_.max_bytes) {
                rotation_warning = rotate_locked();
                report_failure   = failed_;
            }
        }
    }
    if (!rotation_warning.empty() && logger_ != nullptr) { logger_->warn("{}", rotation_warning); }
    if (report_failure && logger_ != nullptr) {
        logger_->error("request log disabled | write failed | {}",
                       product::format_pretty_text(path_));
    }
}

std::string JsonlRequestLog::rotated_path(std::uint32_t index) const {
    return path_ + '.' + std::to_string(index);
}

std::string JsonlRequestLog::rotate_locked() {
    output_.close();
    std::error_code error;
    if (rotation_.keep == 0) {
        std::filesystem::remove(path_, error);
    } else {
        // Shift PATH.(keep-1) .. PATH.1 up by one, overwriting the oldest, then retire PATH.
        for (std::uint32_t index = rotation_.keep - 1; index >= 1; --index) {
            std::error_code shift_error;
            if (!std::filesystem::exists(rotated_path(index), shift_error)) { continue; }
            std::filesystem::remove(rotated_path(index + 1), shift_error);
            std::filesystem::rename(rotated_path(index), rotated_path(index + 1), shift_error);
        }
        std::error_code remove_error;
        std::filesystem::remove(rotated_path(1), remove_error);
        std::filesystem::rename(path_, rotated_path(1), error);
    }
    std::string warning;
    if (error) {
        // Typically another process holds the file open without delete sharing (Windows). Keep
        // appending to it and try again once another max_bytes have been written.
        warning = "request log rotation failed | " + product::format_pretty_text(path_) + " | " +
                  error.message() + " | appending to the current file";
    }
    output_.clear();
    output_.open(path_, std::ios::out | std::ios::app);
    written_bytes_ = 0;
    if (!output_) {
        failed_ = true;
        return warning;
    }
    if (!error && !server_start_record_.empty()) {
        output_ << server_start_record_ << '\n';
        output_.flush();
        if (!output_) {
            failed_ = true;
            return warning;
        }
        written_bytes_ = server_start_record_.size() + kLineTerminatorBytes;
    }
    return warning;
}

} // namespace infernix::serve
