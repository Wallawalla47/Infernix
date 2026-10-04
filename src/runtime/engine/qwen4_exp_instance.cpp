#include "runtime/engine/qwen4_exp_instance.h"

#include "artifact/reader.h"
#include "core/device.h"
#include "core/host_memory.h"
#include "core/startup.h"
#include "models/load_options.h"
#include "models/qwen4_exp/load.h"
#include "models/qwen4_exp/memory_plan.h"
#include "models/qwen4_exp/program/ngram_volume.h"
#include "models/qwen4_exp/program/route_trace.h"
#include "models/registry.h"

#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::runtime {
namespace {

// Planning allowance for the Program's pinned and mapped buffers (I/O staging, sampling, the CPU
// expert service's mapped records); the measured total is well below it.
constexpr std::uint64_t kProgramPinnedAllowance = 256ULL << 20;
// The materializer's upload slots and direct-read bounce buffer during the load.
constexpr std::uint64_t kLoadStagingBytes = 4ULL * (64ULL << 20) + (8ULL << 20);

void report(const EngineOptions& options, DiagnosticLevel level, const std::string& text) {
    if (options.diagnostic_observer.callback) {
        try {
            options.diagnostic_observer.callback(Diagnostic{.level = level, .message = text});
        } catch (...) {}
    } else {
        std::fprintf(stderr, "[engine] %s\n", text.c_str());
    }
}

void validate(const EngineOptions& options) {
    if (options.purpose != EnginePurpose::Generation) {
        throw std::invalid_argument("Qwen3.8-Flash-Next serves generation only (causal scoring is not wired yet)");
    }
    if (options.enable_vision) { throw std::invalid_argument("Qwen3.8-Flash-Next vision is not supported yet"); }
    if ((options.speculative.backend != SpeculativeBackend::None &&
         options.speculative.backend != SpeculativeBackend::Mtp) ||
        options.speculative.ngram_archive_bytes != 0) {
        throw std::invalid_argument("Qwen3.8-Flash-Next speculates with its MTP drafter and n-gram copy proposals");
    }
    if (options.kv_cache != KvCacheStorage::BFloat16 && options.kv_cache != KvCacheStorage::Int8Group64) {
        throw std::invalid_argument("Qwen3.8-Flash-Next supports --kv-dtype bf16 or int8");
    }
    if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("Engine max_concurrency must be in [1,8]");
    }
}

} // namespace

bool artifact_is_qwen4_exp(const std::filesystem::path& artifact) {
    const artifact::Reader reader(artifact);
    return models::qwen4_exp::is_qwen4_exp(reader);
}

Qwen4ExpInstance::Qwen4ExpInstance(std::unique_ptr<models::qwen4_exp::Model> source, const EngineOptions& options)
    : model(std::move(source)), parameters(*model),
      frontend(models::qwen3_5::make_frontend(model->resources(),
                                              {.chat_template_path      = options.chat_template_path,
                                               .architecture            = models::Architecture::Qwen4Exp,
                                               .vision_enabled          = false,
                                               .max_context             = options.max_context,
                                               .media_cache_bytes       = options.media_cache_bytes,
                                               .media_live_bytes        = options.media_live_bytes,
                                               .media_preprocess_threads = options.media_preprocess_threads,
                                               .thinking_budget_message = options.thinking_budget_message})),
      capacity(options.max_context) {}

Qwen4ExpInstance::~Qwen4ExpInstance() = default;

ConstructedQwen4Exp construct_qwen4_exp(const EngineOptions& options, DeviceContext& device) {
    validate(options);
    const auto start = std::chrono::steady_clock::now();
    StartupPhaseScope inspect(options.startup_observer, StartupPhase::ArtifactInspect);
    const artifact::Reader reader(options.artifact_path);
    if (!models::qwen4_exp::is_qwen4_exp(reader)) {
        throw std::invalid_argument("artifact is not Qwen4ExpForCausalLM");
    }
    inspect.complete();
    auto plan = models::qwen4_exp::plan_load(reader, models::load_options(options));
    // The RAM ledger (design §19.3.7): the experts take what the reserve and every other planned
    // allocation leave, before anything is pinned or read.
    models::qwen4_exp::HostMemoryDemand demand;
    demand.reserve      = options.ram_headroom_bytes;
    demand.expert_banks = plan.pinned_expert_bytes();
    demand.other_pinned = plan.pinned_other_bytes();
    demand.later_pinned = kProgramPinnedAllowance;
    demand.pageable     = models::qwen4_exp::NgramVolume::cache_bytes(plan.config().text.ple.table);
    demand.load_staging = kLoadStagingBytes;
    const auto ledger   = models::qwen4_exp::plan_host_memory(host_memory_snapshot(), demand);
    report(options, DiagnosticLevel::Info, ledger.describe());
    if (ledger.placement != models::qwen4_exp::ExpertPlacement::Full) {
        throw std::runtime_error(
            "Qwen3.8-Flash-Next keeps every expert in Host RAM and this machine has too little free now (" +
            ledger.describe() + "). Free Host memory, or lower --ram-headroom-mib (now " +
            std::to_string(options.ram_headroom_bytes >> 20) + " MiB)");
    }
    // A small reserve trims other programs' working sets; keep NInfer's own pageable memory
    // resident so its heap and driver pages never hard-fault from the page file.
    if (!reserve_process_working_set(demand.process_growth)) {
        report(options, DiagnosticLevel::Warning,
               "could not reserve a minimum working set; under memory pressure the engine's pageable memory may "
               "be paged out");
    }
    plan.set_host_reserve(options.ram_headroom_bytes, demand.later_pinned);
    auto model = models::qwen4_exp::materialize_model(std::move(plan), device, &options.startup_observer);
    device.synchronize();

    StartupPhaseScope frontend(options.startup_observer, StartupPhase::FrontendInitialize);
    auto instance = std::make_unique<Qwen4ExpInstance>(std::move(model), options);
    frontend.complete();

    StartupPhaseScope program(options.startup_observer, StartupPhase::ProgramInitialize);
    models::qwen4_exp::ProgramOptions program_options;
    program_options.max_context     = options.max_context;
    program_options.max_concurrency = options.max_concurrency;
    program_options.prefill_chunk   = options.prefill_chunk;
    program_options.kv_cache        = options.kv_cache;
    program_options.diagnostics     = options.diagnostic_observer;
    program_options.kv_capacity_tokens =
        options.kv_capacity.mode == KvCapacityMode::Explicit ? options.kv_capacity.explicit_tokens : 0U;
    program_options.ngram_draft_tokens = options.speculative.ngram_draft_tokens;
    program_options.ngram_min_match    = options.speculative.ngram_min_match;
    program_options.mtp_draft_tokens   = options.speculative.backend == SpeculativeBackend::Mtp
                                             ? options.speculative.draft_tokens
                                             : 0U;
    program_options.ngram_volume = options.ngram_volume_path.empty()
                                       ? models::qwen4_exp::default_ngram_volume(options.artifact_path)
                                       : options.ngram_volume_path;
    program_options.route_trace = models::qwen4_exp::testing::route_trace();
    instance->program =
        std::make_unique<models::qwen4_exp::Program>(instance->parameters, device, std::move(program_options));
    const MemorySummary memory = instance->program->memory_summary();
    instance->kv_capacity_resolution.mode                     = options.kv_capacity.mode;
    instance->kv_capacity_resolution.main_page_groups         = memory.kv_capacity_page_groups;
    instance->kv_capacity_resolution.maximum_main_page_groups = memory.kv_capacity_max_page_groups;
    instance->kv_capacity_resolution.resolved_tokens          = memory.kv_capacity;
    program.complete();

    const auto& stats = instance->model->storage_stats();
    const auto& text  = instance->model->config().text;
    LoadSummary summary;
    summary.architecture         = std::string(models::architecture_name(models::Architecture::Qwen4Exp));
    summary.model_name           = instance->model->info().name;
    summary.load_seconds         = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    summary.upload_seconds       = stats.upload_seconds;
    summary.artifact_bytes_read  = stats.read_bytes;
    summary.host_to_device_bytes = stats.h2d_bytes;
    summary.peak_staging_bytes   = stats.peak_staging_bytes;
    summary.device_object_count  = stats.device_object_count;
    summary.host_object_count    = stats.host_object_count;
    summary.weight_formats       = {"bf16", "fp32", "nvfp4_mul"};

    ModelMetadata metadata;
    metadata.model_id       = instance->model->info().name;
    metadata.vocab_size     = text.vocab_size;
    metadata.embedding_size = text.hidden_size;
    metadata.native_context = text.max_position_embeddings;
    metadata.weights_id     = "nvfp4_mul";
    metadata.weight_bytes   = stats.device_capacity_bytes + stats.pinned_bytes;

    EngineOptions resolved = options;
    return {std::move(instance), std::move(summary), std::move(metadata), std::move(resolved)};
}

} // namespace ninfer::runtime
