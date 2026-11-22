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

// --vision-max-merged: image tokens (cells) an image may take; zero is the compiled default.
std::uint32_t vision_max_cells(const EngineOptions& options) {
    return options.vision_max_merged_tokens == 0 ? 32768U : options.vision_max_merged_tokens;
}

models::inkling2::FrontendOptions frontend_options(const models::inkling2::Model& model, const EngineOptions& options) {
    return {.max_context             = options.max_context,
            .thinking_budget_message = options.thinking_budget_message,
            .vision                  = model.config().vision,
            .vision_max_cells        = vision_max_cells(options)};
}

models::inkling2::ProgramOptions program_options(const EngineOptions& options) {
    return {.max_context     = options.max_context,
            .max_concurrency = options.max_concurrency,
            .causal_scoring  = options.purpose == EnginePurpose::CausalScoring,
            .prefill_chunk   = options.prefill_chunk,
            .use_cuda_graph  = options.use_cuda_graph,
            .dspark            = options.speculative.backend == SpeculativeBackend::DSpark &&
                      options.purpose == EnginePurpose::Generation,
            .draft_tokens      = options.speculative.draft_tokens == 0 ? 7U : options.speculative.draft_tokens,
            .draft_tokens_auto = options.speculative.draft_tokens_auto,
            .expert_a16        = options.expert_a16_activations,
            .vision            = options.enable_vision,
            .vision_max_cells  = options.enable_vision ? vision_max_cells(options) : 0U,
            .kv_codec          = models::inkling2::kv_codec(options.kv_cache).value_or(ops::inkling2::KvCodec::Bf16),
            .diagnostics       = options.diagnostic_observer};
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
    refuse(options.expert_a4_activations, "--expert-a4-activations belongs to Qwen3.8-Flash-Next's FP4-Local artifact");
    refuse(options.a8_activations, "--a8-activations is not built for Inkling 2 yet");
    if (options.expert_a16_activations) {
        report(options, "Inkling 2: --a16-activations: the routed experts take 16-bit activations instead of the "
                        "checkpoint's four-over-six NVFP4 and MXFP8 ones (changes the output; the default follows the "
                        "trained arithmetic and is recommended)");
    }
    if (options.q8_weights && !options.fast_paths) {
        throw std::invalid_argument("--q8-weights is approximate; the reproducible route (--reproducible) refuses it");
    }
    if (options.q8_weights) {
        report(options, "Inkling 2: --q8-weights: the attention input and output projections and the head run as 8-bit "
                        "weights (approximate: the output differs from the artifact's BF16 weights)");
    }
    refuse(!models::inkling2::kv_codec(options.kv_cache).has_value(),
           "--kv-dtype: bf16 (BF16 keys, FP16 values), bf16p, int8, fp8, nvfp4, k8v4, vq2 or k4v2");
    if (options.kv_cache == KvCacheStorage::Vq2 || options.kv_cache == KvCacheStorage::Q4KeyVq2Value) {
        report(options, std::string("Inkling 2: --kv-dtype ") + kv_cache_cli_name(options.kv_cache) +
                            " stores the full visits' keys and values as Hadamard-rotated vector codes (2-4 bits a "
                            "value, no exact recent window; the sliding windows stay BF16 keys, FP16 values; the output "
                            "differs from bf16's)");
    } else if (options.kv_cache == KvCacheStorage::Bf16Packed) {
        report(options, "Inkling 2: --kv-dtype bf16p packs the full visits' BF16 keys and values losslessly to 12 bits "
                        "(the output is bf16's; a request with a row that has no packed form fails)");
    } else if (options.kv_cache != KvCacheStorage::BFloat16) {
        report(options, std::string("Inkling 2: --kv-dtype ") + kv_cache_cli_name(options.kv_cache) +
                            " stores the full visits' keys and values quantized (the sliding windows stay BF16 keys, "
                            "FP16 values; the output differs from bf16's)");
    }
    refuse(options.logits_dtype != LogitsDtype::FP32, "the logits are FP32");
    refuse(!options.enable_vision && options.vision_offload == VisionOffload::On, "--vision-offload on needs --vision");
    if (options.enable_vision && options.vision_offload == VisionOffload::On) {
        report(options, "Inkling 2: --vision-offload on: the vision tower's weights (~0.6 GiB) stay in pinned host memory "
                        "and stream through ~0.1 GiB of device slots per image (the same rows; each image's encode "
                        "waits on the PCIe link)");
    }
    if (options.enable_vision && (options.vision_max_merged_tokens > 32768U ||
                                  (options.vision_max_merged_tokens != 0 && options.vision_max_merged_tokens < 64U))) {
        throw std::invalid_argument("--vision-max-merged must be 64-32768 image tokens");
    }
    refuse(!options.ngram_volume_path.empty() || options.speculative.ngram_draft_tokens != 0,
           "no n-gram drafting");
    refuse(options.max_context_auto, "--max-context auto is not resolved yet: give a context limit");
    refuse(options.max_concurrency == 0 || options.max_concurrency > models::inkling2::kMaximumConcurrency,
           "max_concurrency must be 1-8");
    // 4,096: the fastest chunk measured on the prompt route (2026-11-21, infernix_bench C = 1, pp2048 / pp10000 tok/s:
    // 9,514 / 8,514 at 512, 13,827 / 12,851 at 1,024, 16,400 / 14,926 at 2,048, 16,387 / 16,312 at 4,096); every
    // chunk gives the same bits.
    if (options.prefill_chunk == 0) { options.prefill_chunk = models::inkling2::kDefaultPrefillChunk; }
    refuse(options.prefill_chunk > static_cast<std::uint32_t>(models::inkling2::kMaxPrefillChunk),
           "prefill chunks are at most 8191 columns");
    if (options.context_cache.enabled) {
        options.context_cache.enabled = false;
        report(options, "Inkling 2: the prefix cache is not available yet; it is off");
    }
    SpeculativeOptions& spec = options.speculative;
    if (spec.backend_auto) {
        spec.backend      = SpeculativeBackend::None;
        spec.backend_auto = false;
        report(options, "Inkling 2: --spec auto decodes without a drafter (--spec dspark drafts with the bundled one)");
    }
    if (spec.backend == SpeculativeBackend::DSpark) {
        refuse(options.purpose != EnginePurpose::Generation, "DSpark drafts for generation only");
        refuse(spec.proposal_head != ProposalHead::Full, "--lm-head-draft is not available with DSpark yet");
        if (spec.draft_tokens_auto) {
            if (spec.draft_tokens == 0 || spec.draft_tokens > 7) { spec.draft_tokens = 7; } // the block less its anchor
        } else {
            refuse(spec.draft_tokens == 0 || spec.draft_tokens > 7, "--spec dspark drafts 1-7 tokens a round");
        }
    } else if (spec.backend != SpeculativeBackend::None) {
        throw std::invalid_argument("Inkling 2's drafter is DSpark (--spec dspark)");
    }
    options.speculative.ngram_archive_bytes = 0;
    options.speculative.ngram_session_bytes = 0;
    return options;
}

Inkling2Instance::Inkling2Instance(std::unique_ptr<models::inkling2::Model> source, const EngineOptions& options)
    : model(std::move(source)), parameters(*model), frontend(model->resources(), frontend_options(*model, options)),
      capacity(options.max_context) {}

Inkling2Instance::~Inkling2Instance() = default;

ConstructedInkling2 construct_inkling2(const EngineOptions& options, DeviceContext& device, EnginePlan* planned) {
    const auto start = std::chrono::steady_clock::now();
    const artifact::Reader reader(options.artifact_path);
    if (!models::inkling2::is_inkling2(reader)) { throw std::invalid_argument("artifact is not Inkling2LoopForCausalLM"); }
    models::LoadOptions load = models::load_options(options, models::Architecture::Inkling2Loop);
    auto plan                = models::inkling2::plan_load(reader, load);
    models::inkling2::ProgramOptions program = program_options(options);
    program.vision_host = plan.vision_host() ? &*plan.vision_host() : nullptr; // the plan's until materialized
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
    program.vision_host = model->vision_host() ? &*model->vision_host() : nullptr;
    const auto& packed = model->packed();
    if (packed.source_bytes != 0) {
        constexpr double kMiB = 1024.0 * 1024.0;
        publish_diagnostic(options.diagnostic_observer, DiagnosticLevel::Debug,
                           "packed BF16 weights: %zu matrices, %.0f MiB -> %.0f MiB (bf16_p12, %llu escapes) in %.0f ms",
                           packed.parents.size(), packed.source_bytes / kMiB, packed.storage.bytes / kMiB,
                           static_cast<unsigned long long>(packed.escapes), packed.seconds * 1000.0);
    }
    if (const auto& q8 = model->q8(); q8.source_bytes != 0) {
        constexpr double kMiB = 1024.0 * 1024.0;
        publish_diagnostic(options.diagnostic_observer, DiagnosticLevel::Debug,
                           "8-bit weights: %zu matrices, %.0f MiB -> %.0f MiB (q8_g32_fp16) in %.0f ms",
                           q8.parents.size(), q8.source_bytes / kMiB, q8.storage.bytes / kMiB, q8.seconds * 1000.0);
    }
    EngineOptions resolved = options;
    auto instance          = std::make_unique<Inkling2Instance>(std::move(model), resolved);
    if (!options.chat_template_path.empty()) {
        template_override = read_text(options.chat_template_path);
        models::inkling2::FrontendResources resources = instance->model->resources();
        resources.chat_template_jinja                 = template_override;
        instance->frontend = models::inkling2::Frontend(resources, frontend_options(*instance->model, options));
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
    if (instance->model->q8().source_bytes != 0) { summary.weight_formats.push_back("q8_g32_fp16"); }
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
