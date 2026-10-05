#include "runtime/engine/model_instance.h"
#include "artifact/reader.h"
#include "artifact/formats.h"
#include "core/startup.h"
#include "models/qwen3_5/load.h"
#include "models/qwen3_5/measurement.h"

#include <algorithm>
#include <system_error>
#include <string>
#include <filesystem>
#include <chrono>
#include <map>
#include <set>
#include <stdexcept>
#include <utility>

namespace ninfer::runtime {
namespace {
using Clock = std::chrono::steady_clock;
} // namespace

void validate_engine_options(const EngineOptions& options, models::Architecture architecture) {
    if (options.artifact_path.empty()) {
        throw std::invalid_argument("Engine artifact_path must not be empty");
    }
    if (options.artifact_path.extension() != ".ninfer") {
        throw std::invalid_argument("NInfer accepts only .ninfer artifacts");
    }
    if (options.max_context == 0) {
        throw std::invalid_argument("Engine max_context must be nonzero");
    }
    switch (options.kv_capacity.mode) {
    case KvCapacityMode::Explicit:
        if (options.kv_capacity.explicit_tokens == 0) {
            throw std::invalid_argument("Engine explicit kv_capacity must be nonzero");
        }
        // Qwen3.5's headroom is what automatic KV capacity leaves; an explicit capacity has none
        // (Qwen3.8-Flash-Next's is what its expert cache leaves, with any KV capacity).
        if (architecture == models::Architecture::Qwen3_5 && options.vram_headroom_bytes.has_value()) {
            throw std::invalid_argument(
                "Qwen3.5 applies --vram-headroom-mib only with --kv-capacity auto");
        }
        break;
    case KvCapacityMode::Automatic:
        if (options.kv_capacity.explicit_tokens != 0) {
            throw std::invalid_argument(
                "Engine automatic kv_capacity must not carry explicit tokens");
        }
        break;
    default:
        throw std::invalid_argument("Engine kv_capacity mode is invalid");
    }
    if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("Engine max_concurrency must be in [1,8]");
    }
    if (options.max_pending_requests == 0 || options.pending_timeout_ms == 0) {
        throw std::invalid_argument("Engine pending request capacity and timeout must be nonzero");
    }
    if (options.enable_vision && options.media_live_bytes == 0) {
        throw std::invalid_argument(
            "Engine media_live_bytes must be nonzero when Vision is enabled");
    }
    if (options.media_preprocess_threads > 64) {
        throw std::invalid_argument("Engine media_preprocess_threads must be in [0,64]");
    }
    // The prompt-kernel choice exists only for INT8 KV; accepting it elsewhere would select a
    // kernel that never runs.
    if (options.original_int8_prefill_kernel && options.kv_cache != KvCacheStorage::Int8Group64) {
        throw std::invalid_argument(
            "the original INT8 prefill kernel requires the INT8 KV cache (--kv-dtype int8)");
    }
    if (options.prefill_split_workspace_mib > kMaximumPrefillSplitWorkspaceMiB) {
        throw std::invalid_argument("Engine prefill_split_workspace_mib must be in [0,16384]");
    }
    if (options.original_nvfp4_prefill_kernel && options.kv_cache != KvCacheStorage::Nvfp4Group16) {
        throw std::invalid_argument(
            "the original NVFP4 prefill kernel requires the NVFP4 KV cache (--kv-dtype nvfp4)");
    }
}

namespace {

// The hybrid index ranks admission sources and values snapshots with the same calibrated
// prefill and Host-to-Device coefficients as the original cache's ResourceManager.
// Uncalibrated (zero) terms keep the index's generic defaults.
prefix_cache::CacheCostModel hybrid_cache_cost(const ContextMachineCostModel& model) {
    constexpr double kSecondsPerNs = 1.0e-9;
    constexpr double kQ32          = 4294967296.0;
    prefix_cache::CacheCostModel cost;
    if (model.prefill.chunk_ns != 0) {
        cost.chunk_seconds = static_cast<double>(model.prefill.chunk_ns) * kSecondsPerNs;
    }
    if (model.prefill.token_ns_q32 != 0) {
        cost.token_seconds = static_cast<double>(model.prefill.token_ns_q32) / kQ32 * kSecondsPerNs;
    }
    if (model.prefill.attention_pair_ns_q32 != 0) {
        cost.attention_pair_seconds =
            static_cast<double>(model.prefill.attention_pair_ns_q32) / kQ32 * kSecondsPerNs;
    }
    const ContextTransferCost& h2d =
        model.transfer[static_cast<std::size_t>(ContextTransferDirection::HostToDevice)];
    if (h2d.ns_per_byte_q32 != 0) {
        cost.h2d_bytes_per_second =
            1.0 / (static_cast<double>(h2d.ns_per_byte_q32) / kQ32 * kSecondsPerNs);
    }
    if (h2d.batch_ns != 0) {
        cost.transfer_batch_seconds = static_cast<double>(h2d.batch_ns) * kSecondsPerNs;
    }
    return cost;
}

} // namespace

// The exact artifact, its execution signature, the KV and speculative formats, RoPE scaling and the
// product binary's build identity. Any difference makes the saved state meaningless, so the file is
// ignored.
std::string hybrid_cache_fingerprint(const EngineOptions& options, const std::string& signature) {
    std::error_code error;
    const auto size = std::filesystem::file_size(options.artifact_path, error);
    const auto time = std::filesystem::last_write_time(options.artifact_path, error);
    std::string out = "artifact=" + std::filesystem::absolute(options.artifact_path).string();
    out += ";size=" + std::to_string(error ? 0U : size);
    out += ";mtime=" + std::to_string(error ? 0 : time.time_since_epoch().count());
    out += ";signature=" + signature;
    out += ";kv=" + std::to_string(static_cast<int>(options.kv_cache));
    out += ";speculative=" + std::to_string(static_cast<int>(options.speculative.backend));
    out += ";yarn=" + std::to_string(options.rope_yarn_factor);
    out += ";build=" + options.context_cache.hybrid.persistent_identity;
    return out;
}

namespace {

std::size_t current_free_device_bytes() {
    std::size_t free_bytes  = 0;
    std::size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    return free_bytes;
}

} // namespace

EngineOptions normalize_engine_options(EngineOptions options, models::Architecture architecture) {
    switch (options.purpose) {
    case EnginePurpose::Generation:
        break;
    case EnginePurpose::CausalScoring:
        options.max_concurrency      = 1;
        options.max_pending_requests = 1;
        options.kv_capacity         = KvCapacityPolicy::explicit_capacity(options.max_context);
        options.speculative          = {};
        options.enable_vision        = false;
        options.use_cuda_graph       = false;
        options.context_cache        = ContextCacheOptions{
                   .enabled = false, .device_state_slots = 0, .host_capacity_bytes = 0};
        break;
    default:
        throw std::invalid_argument("Engine purpose is invalid");
    }
    if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("Engine max_concurrency must be in [1,8]");
    }

    ContextCacheOptions& cache = options.context_cache;
    if (options.speculative.ngram_archive_bytes != 0 &&
        options.speculative.ngram_draft_tokens == 0) {
        throw std::invalid_argument(
            "the cross-request ngram archive requires ngram drafting; enable --ngram-draft-tokens");
    }
    if (options.speculative.ngram_archive_bytes != 0 &&
        (options.speculative.ngram_session_bytes < (1ULL << 20) ||
         options.speculative.ngram_session_bytes > options.speculative.ngram_archive_bytes)) {
        throw std::invalid_argument("ngram session capacity must be between 1 MiB and the total "
                                    "archive capacity");
    }
    const std::uint32_t concurrency = options.max_concurrency;
    const bool qwen4_exp            = architecture == models::Architecture::Qwen4Exp;
    if (qwen4_exp && cache.enabled && cache.mode == ContextCacheMode::Original) {
        throw std::invalid_argument(
            "Qwen3.8-Flash-Next has no original prefix cache: use the hybrid prefix cache or "
            "disable the context cache");
    }
    if (cache.enabled && cache.mode == ContextCacheMode::Hybrid) {
        if (cache.device_state_slots) {
            throw std::invalid_argument(
                "the hybrid prefix cache sizes its Device state slots with device_snapshot_slots");
        }
        // One pinned Host slab pool serves blocks and snapshots alike; its size is the only
        // capacity a deployment has to choose (docs/maintainer/hybrid-prefix-cache-spec.md §5.4).
        cache.host_capacity_bytes = cache.host_capacity_bytes.value_or(
            qwen4_exp ? kDefaultQwen4ExpHybridHostCacheBytes : kDefaultHybridHostCacheBytes);
        const bool host_tier             = *cache.host_capacity_bytes != 0;
        HybridPrefixCacheOptions& hybrid = cache.hybrid;
        if (qwen4_exp) {
            // Qwen3.8-Flash-Next's snapshots are born in the Host tier (design §19.3.1): it needs
            // the tier and has no Device snapshot slots.
            if (!host_tier) {
                throw std::invalid_argument(
                    "Qwen3.8-Flash-Next's prefix cache keeps its snapshots in the Host tier: "
                    "--host-context-mib must be positive");
            }
            if (hybrid.device_snapshot_slots.value_or(0U) != 0) {
                throw std::invalid_argument(
                    "Qwen3.8-Flash-Next's prefix cache has no Device snapshot slots");
            }
            hybrid.device_snapshot_slots = 0U;
        }
        // One resident snapshot per request lane keeps every live conversation's latest
        // endpoint restorable without PCIe traffic; one more slot stages taps and endpoints while
        // their Host copies are written. Without a Host tier these slots are the only snapshot
        // storage, so one more is kept for shared prefixes.
        if (!qwen4_exp) {
            hybrid.device_snapshot_slots =
                hybrid.device_snapshot_slots.value_or(concurrency + (host_tier ? 1U : 2U));
        }
        // Taps without a Host tier would evict other conversations' resident snapshots.
        hybrid.max_new_taps = hybrid.max_new_taps.value_or(host_tier ? 8U : 2U);
        // Ladder taps are realized on prefill chunk boundaries, so the ladder never refines below
        // the chunk: coarser ladders only waste snapshots on taps that share one boundary.
        const std::uint32_t chunk = std::max<std::uint32_t>(options.prefill_chunk, 64U);
        hybrid.tap_ladder_tokens =
            hybrid.tap_ladder_tokens.value_or(std::max<std::uint32_t>(4096U, 2U * chunk));
        hybrid.tap_min_gap_tokens =
            hybrid.tap_min_gap_tokens.value_or(std::max<std::uint32_t>(1024U, chunk));
        if ((!qwen4_exp && *hybrid.device_snapshot_slots == 0) || *hybrid.device_snapshot_slots > 64) {
            throw std::invalid_argument("hybrid device snapshot slots must be in [1,64]");
        }
        if (*hybrid.max_new_taps > 64) {
            throw std::invalid_argument("hybrid taps per request must be at most 64");
        }
        if (*hybrid.tap_ladder_tokens < 64 || *hybrid.tap_min_gap_tokens < 64) {
            throw std::invalid_argument(
                "hybrid tap ladder and minimum gap must be at least 64 tokens");
        }
        // Every lane holds one active sequence; retained context lives in the Program's prefix
        // index, whose snapshots take the extra Device StateImage slots.
        cache.device_state_slots = hybrid.device_snapshot_slots;
    } else {
        if (cache.mode == ContextCacheMode::Hybrid) { cache.mode = ContextCacheMode::Original; }
        cache.device_state_slots = cache.device_state_slots.value_or(concurrency);
    }
    const std::uint64_t total_device_state_slots =
        static_cast<std::uint64_t>(concurrency) + *cache.device_state_slots;
    if (total_device_state_slots > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("context cache Device state capacity exceeds uint32");
    }
    return options;
}

ModelInstance::ModelInstance(std::unique_ptr<models::qwen3_5::Model> source,
                             const EngineOptions& options)
    : model(std::move(source)), parameters(*model),
      frontend(models::qwen3_5::make_frontend(
          model->resources(),
          {.chat_template_path       = options.chat_template_path,
           .architecture             = model->config().text.architecture,
           .vision_enabled           = options.enable_vision,
           .max_context              = options.max_context,
           .media_cache_bytes        = options.media_cache_bytes,
           .media_live_bytes         = options.media_live_bytes,
           .media_preprocess_threads = options.media_preprocess_threads,
           .vision_max_merged_tokens = options.vision_max_merged_tokens,
           .ngram_sources_enabled    = options.speculative.ngram_draft_tokens != 0,
           .ngram_archive_enabled    = options.speculative.ngram_archive_bytes != 0})),
      capacity(options.max_context) {}

ModelInstance::~ModelInstance() = default;

ConstructedModel construct_model(EngineOptions& options, DeviceContext& device) {
    validate_engine_options(options, models::Architecture::Qwen3_5);
    const auto start = Clock::now();
    StartupPhaseScope inspect(options.startup_observer, StartupPhase::ArtifactInspect);
    artifact::Reader reader(options.artifact_path);
    inspect.complete();
    StartupPhaseScope binding(options.startup_observer, StartupPhase::TargetPlan);
    auto plan = models::qwen3_5::plan_load(reader, models::load_options(options, models::Architecture::Qwen3_5));
    plan.set_host_reserve(options.ram_headroom_bytes);
    binding.complete();
    auto model =
        models::qwen3_5::materialize_model(std::move(plan), device, &options.startup_observer);
    device.synchronize();
    StartupPhaseScope frontend(options.startup_observer, StartupPhase::FrontendInitialize);
    auto instance = std::make_unique<ModelInstance>(std::move(model), options);
    frontend.complete();
    StartupPhaseScope planning(options.startup_observer, StartupPhase::TargetFinalize);
    const auto signature = models::qwen3_5::prefill_signature(*instance->model);
    auto context_cost    = resolve_context_machine_cost(
        {.hardware_class =
                context_cost_hardware_class(device.props.name, device.props.major, device.props.minor),
            .prefill_signature = signature},
        options.context_cost.preset_path);
    auto planner    = models::qwen3_5::make_sequence_planner(instance->parameters, device, options);
    auto resolution = resolve_kv_capacity(
        options.kv_capacity, planner.capacity_curve(), current_free_device_bytes(),
        options.vram_headroom_bytes.value_or(kDefaultKvCapacityHeadroomBytes));
    auto sequence   = std::move(planner).finalize(resolution.main_page_groups);
    if (sequence.device_reservation_bytes() != resolution.runtime_reservation_bytes ||
        sequence.kv_capacity() != resolution.resolved_tokens) {
        throw std::logic_error("resolved KV capacity does not match the finalized Program plan");
    }
    options.context_cache.host_capacity_bytes = sequence.host_capacity_bytes();
    // Automatic tree widths on a target that cannot verify trees resolve to chain verification.
    options.speculative.draft_tree_auto = sequence.draft_tree_auto();
    instance->kv_capacity_resolution    = resolution;
    planning.complete();
    StartupPhaseScope program(options.startup_observer, StartupPhase::ProgramInitialize);
    instance->program = models::qwen3_5::create_program(instance->parameters, std::move(sequence),
                                                        device, options.startup_observer);
    LoadSummary::PrefixCacheRestore restore;
    if (options.context_cache.enabled && options.context_cache.mode == ContextCacheMode::Hybrid) {
        instance->program->set_hybrid_cost(hybrid_cache_cost(context_cost.model));
        // A request waiting for a sibling's snapshot stays in the FIFO, so the predicted wait
        // is kept well inside its queue timeout.
        instance->program->set_hybrid_coalesce_wait_limit(
            static_cast<double>(options.pending_timeout_ms) / 1000.0 / 2.0);
        const std::filesystem::path& file = options.context_cache.hybrid.persistent_file;
        if (!file.empty()) {
            const models::qwen3_5::HybridCachePersistence loaded =
                instance->program->attach_hybrid_cache_file(
                    file, hybrid_cache_fingerprint(options, signature), options.startup_observer);
            restore = LoadSummary::PrefixCacheRestore{
                .attempted           = true,
                .restored            = loaded.ok,
                .message             = loaded.message,
                .blocks              = loaded.blocks,
                .snapshots           = loaded.snapshots,
                .bytes               = loaded.bytes,
                .seconds             = loaded.seconds,
                .saved_blocks        = loaded.saved_blocks,
                .saved_snapshots     = loaded.saved_snapshots,
                .required_host_bytes = loaded.required_host_bytes,
                .host_bytes          = loaded.host_bytes,
            };
        }
    }
    device.synchronize();
    program.complete();
    instance->kv_capacity_resolution.available_after_startup_bytes = current_free_device_bytes();
    const auto& stats = instance->model->storage_stats();
    LoadSummary summary;
    summary.architecture = models::architecture_name(instance->model->config().text.architecture);
    summary.model_name   = instance->model->info().name;
    summary.prefill_signature = signature;
    std::set<std::string> formats;
    for (const auto& weight : instance->model->weight_data()) {
        for (const auto& part : weight.view.parts) {
            formats.emplace(artifact::format_name(part.parent->geometry.format));
        }
    }
    summary.weight_formats.assign(formats.begin(), formats.end());
    summary.load_seconds         = std::chrono::duration<double>(Clock::now() - start).count();
    summary.upload_seconds       = stats.upload_seconds;
    summary.artifact_bytes_read  = stats.read_bytes;
    summary.host_to_device_bytes = stats.h2d_bytes;
    summary.peak_staging_bytes   = stats.peak_staging_bytes;
    summary.device_object_count  = stats.device_object_count;
    summary.host_object_count    = stats.host_object_count;
    summary.context_cost         = std::move(context_cost.summary);
    summary.prefix_cache         = std::move(restore);

    // Model metadata for /v1/models. The architecture owns the dimension facts; the Engine adds the
    // registered identity and the artifact-measured element and payload totals. The effective
    // context ceiling is a property of this instance, so it is not part of the model facts.
    ModelMetadata metadata;
    metadata.model_id       = instance->model->info().name;
    metadata.vocab_size     = instance->model->config().text.vocab_size;
    metadata.embedding_size = instance->model->config().text.hidden_size;
    metadata.native_context = instance->model->config().text.max_position_embeddings;
    // Total logical elements and encoded payload bytes over the artifact's distinct weight parents.
    // Several logical weights (tied embeddings, packed projections) can share one encoded parent,
    // so count each parent once. ftype is the format covering the most weights (a single
    // llama.cpp-style name); the full format set stays in LoadSummary::weight_formats.
    std::set<const WeightParent*> counted;
    std::map<std::string, std::uint64_t> elements_by_format;
    for (const auto& weight : instance->model->weight_data()) {
        for (const auto& part : weight.view.parts) {
            if (part.parent == nullptr || !counted.insert(part.parent).second) { continue; }
            metadata.parameters += part.parent->geometry.elements;
            metadata.weight_bytes += part.parent->geometry.bytes;
            elements_by_format[std::string(artifact::format_name(part.parent->geometry.format))] +=
                part.parent->geometry.elements;
        }
    }
    std::uint64_t dominant_elements = 0;
    for (const auto& [name, elements] : elements_by_format) {
        if (elements > dominant_elements) {
            dominant_elements   = elements;
            metadata.weights_id = name;
        }
    }
    return {std::move(instance), std::move(summary), std::move(metadata),
            std::move(context_cost.model)};
}

} // namespace ninfer::runtime
