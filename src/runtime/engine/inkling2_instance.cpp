#include "runtime/engine/inkling2_instance.h"

#include "core/device.h"
#include "models/inkling2/execution/forward.h"
#include "models/inkling2/load.h"
#include "runtime/engine/diagnostics.h"

#include <chrono>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace infernix::runtime {
namespace {

void refuse(bool condition, const char* what) {
    if (condition) { throw std::invalid_argument(std::string("Inkling 2: ") + what); }
}

void report(const EngineOptions& options, const std::string& text) {
    publish_diagnostic(options.diagnostic_observer, DiagnosticLevel::Info, "%s", text.c_str());
}

models::inkling2::ProgramOptions program_options(const EngineOptions& options) {
    return {.max_context     = options.max_context,
            .max_concurrency = options.max_concurrency,
            .causal_scoring  = options.purpose == EnginePurpose::CausalScoring,
            .prefill_chunk   = options.prefill_chunk,
            .use_cuda_graph  = options.use_cuda_graph,
            .diagnostics     = options.diagnostic_observer};
}

std::string read_text(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) { throw std::invalid_argument("cannot read the chat template " + path.string()); }
    std::ostringstream text;
    text << file.rdbuf();
    return text.str();
}

} // namespace

bool artifact_is_inkling2(const artifact::Reader& reader) { return models::inkling2::is_inkling2(reader); }

EngineOptions normalize_inkling2_options(EngineOptions options) {
    refuse(options.lossless || options.reference, "the lossless and reference routes are a later milestone");
    refuse(options.expert_a4_activations || options.expert_a16_activations || options.a8_activations,
           "the expert and 8-bit activation switches belong to other architectures");
    refuse(options.kv_cache != KvCacheStorage::BFloat16, "the KV cache stores BF16 keys and FP16 values (INT8 KV is "
                                                         "a later milestone)");
    refuse(options.logits_dtype != LogitsDtype::FP32, "the logits are FP32");
    refuse(options.enable_vision, "the vision tower is a later milestone");
    refuse(!options.ngram_volume_path.empty() || options.speculative.ngram_draft_tokens != 0,
           "no n-gram drafting");
    refuse(options.max_context_auto, "--max-context auto is not resolved yet: give a context limit");
    refuse(options.max_concurrency == 0 || options.max_concurrency > models::inkling2::kMaximumConcurrency,
           "max_concurrency must be 1-8");
    // 512: the fastest chunk measured (2026-11-21, infernix_bench pp2048: 1,303.7 at 128, 1,505.5 at 512, 1,398.1 at
    // 2,048 tok/s); every chunk gives the same bits.
    if (options.prefill_chunk == 0) { options.prefill_chunk = models::inkling2::kDefaultPrefillChunk; }
    refuse(options.prefill_chunk > static_cast<std::uint32_t>(models::inkling2::kMaxPrefillChunk),
           "prefill chunks are at most 8191 columns");
    if (options.context_cache.enabled) {
        options.context_cache.enabled = false;
        report(options, "Inkling 2: the prefix cache is not available yet; it is off");
    }
    if (options.speculative.backend != SpeculativeBackend::None || options.speculative.backend_auto) {
        if (!options.speculative.backend_auto) {
            throw std::invalid_argument("Inkling 2: speculative decoding (DSpark) is a later milestone");
        }
        options.speculative.backend      = SpeculativeBackend::None;
        options.speculative.backend_auto = false;
        report(options, "Inkling 2: --spec auto decodes without a drafter (DSpark is a later milestone)");
    }
    options.speculative.ngram_archive_bytes = 0;
    options.speculative.ngram_session_bytes = 0;
    return options;
}

Inkling2Instance::Inkling2Instance(std::unique_ptr<models::inkling2::Model> source, const EngineOptions& options)
    : model(std::move(source)), parameters(*model),
      frontend(model->resources(),
               {.max_context = options.max_context, .thinking_budget_message = options.thinking_budget_message}),
      capacity(options.max_context) {}

Inkling2Instance::~Inkling2Instance() = default;

ConstructedInkling2 construct_inkling2(const EngineOptions& options, DeviceContext& device, EnginePlan* planned) {
    const auto start = std::chrono::steady_clock::now();
    const artifact::Reader reader(options.artifact_path);
    if (!models::inkling2::is_inkling2(reader)) { throw std::invalid_argument("artifact is not Inkling2LoopForCausalLM"); }
    models::LoadOptions load = models::load_options(options, models::Architecture::Inkling2Loop);
    auto plan                = models::inkling2::plan_load(reader, load);
    const models::inkling2::ProgramOptions program = program_options(options);
    const models::inkling2::ProgramDevicePlan device_plan = models::inkling2::Program::plan_device(program, plan.config());
    if (planned != nullptr) {
        planned->architecture                   = std::string(models::architecture_name(models::Architecture::Inkling2Loop));
        planned->vram.weights_bytes             = plan.device_bytes();
        planned->vram.kv_bytes                  = device_plan.kv;
        planned->vram.kv_tokens                 = options.max_context * options.max_concurrency;
        planned->vram.minimum_reservation_bytes = device_plan.fixed_bytes();
        planned->vram.workspace_bytes           = device_plan.workspace;
        return {};
    }
    std::string template_override;
    auto model = models::inkling2::materialize_model(std::move(plan), device, &options.startup_observer);
    const auto& packed = model->packed();
    if (packed.source_bytes != 0) {
        constexpr double kMiB = 1024.0 * 1024.0;
        publish_diagnostic(options.diagnostic_observer, DiagnosticLevel::Debug,
                           "packed BF16 weights: %zu matrices, %.0f MiB -> %.0f MiB (bf16_p12, %llu escapes) in %.0f ms",
                           packed.parents.size(), packed.source_bytes / kMiB, packed.storage.bytes / kMiB,
                           static_cast<unsigned long long>(packed.escapes), packed.seconds * 1000.0);
    }
    EngineOptions resolved = options;
    auto instance          = std::make_unique<Inkling2Instance>(std::move(model), resolved);
    if (!options.chat_template_path.empty()) {
        template_override = read_text(options.chat_template_path);
        models::inkling2::FrontendResources resources = instance->model->resources();
        resources.chat_template_jinja                 = template_override;
        instance->frontend = models::inkling2::Frontend(resources, {.max_context = options.max_context,
                                                                    .thinking_budget_message =
                                                                        options.thinking_budget_message});
    }
    instance->program = std::make_unique<models::inkling2::Program>(instance->parameters, device, program);

    const auto& stats = instance->model->storage_stats();
    const auto& text  = instance->model->config().text;
    LoadSummary summary;
    summary.architecture         = std::string(models::architecture_name(models::Architecture::Inkling2Loop));
    summary.model_name           = instance->model->info().name;
    summary.load_seconds         = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    summary.upload_seconds       = stats.upload_seconds;
    summary.artifact_bytes_read  = stats.read_bytes;
    summary.host_to_device_bytes = stats.h2d_bytes;
    summary.peak_staging_bytes   = stats.peak_staging_bytes;
    summary.device_object_count  = stats.device_object_count;
    summary.host_object_count    = stats.host_object_count;
    summary.weight_formats       = {"bf16", "fp32", "nvfp4_mul", "fp8_e4m3fn_g32_e8m0"};
    if (instance->model->packed().source_bytes != 0) { summary.weight_formats.push_back("bf16_p12"); }
    summary.fast_paths_disabled  = !options.fast_paths;
    summary.request_invariant    = true; // every route is column-invariant and the decode rows independent

    ModelMetadata metadata;
    metadata.model_id       = instance->model->info().name;
    metadata.vocab_size     = text.vocab_size;
    metadata.embedding_size = text.hidden_size;
    metadata.native_context = text.max_position_embeddings;
    metadata.weights_id     = "nvfp4_mul";
    metadata.weight_bytes   = instance->model->device_weight_bytes();
    return {std::move(instance), std::move(summary), std::move(metadata), std::move(resolved)};
}

} // namespace infernix::runtime
