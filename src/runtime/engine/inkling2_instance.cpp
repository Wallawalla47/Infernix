#include "runtime/engine/inkling2_instance.h"

#include "core/device.h"
#include "core/host_memory.h"
#include "core/vram_budget.h"
#include "models/inkling2/execution/forward.h"
#include "models/inkling2/load.h"
#include "runtime/engine/diagnostics.h"
#include "runtime/engine/engine_plan.h"
#include "runtime/engine/expert_memory.h"
#include "runtime/engine/model_instance.h"
#include "runtime/expert_offload/memory_plan.h"
#include "runtime/prefix_cache/persist_file.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <optional>
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
            .lossless        = options.lossless,
            .reference       = options.reference,
            .prefill_chunk   = options.prefill_chunk,
            .use_cuda_graph  = options.use_cuda_graph,
            .dspark            = options.speculative.backend == SpeculativeBackend::DSpark &&
                      options.purpose == EnginePurpose::Generation,
            .draft_tokens      = options.speculative.draft_tokens == 0 ? 7U : options.speculative.draft_tokens,
            .draft_tokens_auto = options.speculative.draft_tokens_auto,
            .expert_a16        = options.expert_a16_activations,
            .a8                = options.a8_activations,
            .vision            = options.enable_vision,
            .vision_max_cells  = options.enable_vision ? vision_max_cells(options) : 0U,
            .kv_codec          = models::inkling2::kv_codec(options.kv_cache).value_or(ops::inkling2::KvCodec::Bf16),
            .prefix_cache      = options.context_cache.enabled && options.purpose == EnginePurpose::Generation,
            .prefix_host_bytes = options.context_cache.host_capacity_bytes.value_or(0U),
            .prefix_max_new_taps   = options.context_cache.hybrid.max_new_taps.value_or(8U),
            .prefix_ladder_tokens  = options.context_cache.hybrid.tap_ladder_tokens.value_or(4096U),
            .prefix_min_gap_tokens = options.context_cache.hybrid.tap_min_gap_tokens.value_or(1024U),
            .prefix_save           = options.context_cache.hybrid.persistent_save,
            .prefix_ssd_bytes      = options.context_cache.ssd_capacity_bytes,
            .prefix_ssd_directory  = options.context_cache.ssd_directory,
            .prefix_ssd_write_bytes_per_hour =
                options.context_cache.ssd_write_bytes_per_hour == ContextCacheOptions::kSsdWritesUnlimited
                    ? 0U
                    : (options.context_cache.ssd_write_bytes_per_hour == 0 ? ContextCacheOptions::kDefaultSsdWriteBytesPerHour
                                                                           : options.context_cache.ssd_write_bytes_per_hour),
            // With a prefix cache file the SSD tier's file is kept beside it across restarts.
            .prefix_ssd_kept_file = options.context_cache.ssd_capacity_bytes != 0 &&
                                            !options.context_cache.hybrid.persistent_file.empty()
                                        ? runtime::prefix_cache::kept_ssd_file(options.context_cache.ssd_directory,
                                                                               options.context_cache.hybrid.persistent_file)
                                        : std::filesystem::path{},
            .diagnostics       = options.diagnostic_observer};
}

// The prefix cache's Device pool: with --kv-capacity auto every page the free VRAM holds beside the Program's other
// buffers and the display headroom (the lanes' contexts first, the rest cache-only), else the explicit capacity; never
// less than every lane's context.
std::uint32_t cache_pool_pages(const EngineOptions& options, models::inkling2::ProgramOptions program,
                               const models::inkling2::Config& config, std::uint64_t free_bytes, std::uint64_t headroom) {
    const std::uint64_t lanes = static_cast<std::uint64_t>(models::inkling2::pages_per_lane(options.max_context)) *
                                options.max_concurrency;
    if (options.kv_capacity.mode != KvCapacityMode::Automatic) {
        const std::uint64_t asked =
            (std::uint64_t{options.kv_capacity.explicit_tokens} + ops::inkling2::kKeyUnit - 1) / ops::inkling2::kKeyUnit;
        return static_cast<std::uint32_t>(std::min<std::uint64_t>(std::max(asked, lanes), 0x7FFFFFF0U));
    }
    program.kv_pages                                = 0;
    const models::inkling2::ProgramDevicePlan plan = models::inkling2::Program::plan_device(program, config);
    const std::uint64_t used                       = plan.fixed_bytes() + headroom;
    const std::uint64_t extra = free_bytes > used && plan.page_bytes != 0 ? (free_bytes - used) / plan.page_bytes : 0U;
    return static_cast<std::uint32_t>(std::min<std::uint64_t>(lanes + extra, 0x7FFFFFF0U));
}

// --host-context-mib auto: the Host tier takes the RAM the plan (its pinned weights) leaves beyond --ram-free-mib, as
// the other models' tiers do (expert_memory.h).
std::uint64_t cache_host_bytes(const EngineOptions& options, std::uint64_t pinned_bytes, const VramSnapshot& vram) {
    if (options.context_cache.host_capacity_bytes) { return *options.context_cache.host_capacity_bytes; }
    const expert_offload::HostMemoryDemand demand =
        expert_ram_demand(options, {.expert_banks = 0, .other_pinned = pinned_bytes}, 0, vram);
    return auto_host_context_bytes(expert_offload::plan_host_memory(host_memory_snapshot(), demand));
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

bool artifact_has_inkling2_drafter(const artifact::Reader& reader) {
    return reader.directory().components.contains("dspark");
}

EngineOptions normalize_inkling2_options(EngineOptions options, bool bundled_drafter) {
    refuse(!options.lora_path.empty(), "--lora applies to the dense Qwen3.5-family models, not to Inkling 2");
    refuse(!options.ablate_sidecar_path.empty(), "--ablate is not available on Inkling 2 yet");
    refuse(options.reference && !options.lossless, "--reference implies --lossless");
    refuse(options.reference && options.kv_cache != KvCacheStorage::BFloat16,
           "the design reference is built for --kv-dtype bf16 only (V stored as FP16); the coded formats are not yet");
    if (options.lossless) {
        // The lossless route (docs/maintainer/lossless-route.md §2b, §3a): a scored window in one pass, or generation
        // in the cache form (eager calls, a lane at a time), text only.
        refuse(options.enable_vision, "--lossless takes text (no --vision)");
        refuse(options.expert_a16_activations, "--lossless runs the experts' declared arithmetic (no --a16-activations)");
        refuse(options.q8_weights, "--lossless takes the artifact's exact weights (no --q8-weights)");
        refuse(options.kv_cache != KvCacheStorage::BFloat16,
               "--lossless keeps K and V exactly (--kv-dtype bf16; V stored as BF16 on this route)");
        refuse(options.fast_paths, "--lossless has no fast paths");
    }
    refuse(options.expert_a4_activations, "--expert-a4-activations belongs to Qwen3.8-Flash-Next's FP4-Local artifact");
    if (options.a8_activations) {
        // The 8-bit route (Forward's a8): the --q8-weights matrices, whose attention input and output projections take
        // 8-bit activations in prompt calls.
        refuse(!options.fast_paths, "--a8-activations is approximate; the reproducible route (--reproducible) refuses it");
        refuse(options.expert_a16_activations,
               "--a16-activations runs with 16-bit activations: it excludes --a8-activations");
        options.q8_weights = true;
        report(options, "Inkling 2: --a8-activations: 8-bit activations (int8 codes with an FP32 scale per 32 values, as "
                        "llama.cpp's Q8_1) in every call: the attention input and output projections and the head (on "
                        "8-bit weights, --q8-weights) and the NVFP4 layers' routed experts (instead of the checkpoint's "
                        "4-bit activations); the MXFP8 experts keep their 8-bit arithmetic, the message gates and the "
                        "router 16-bit activations, as Basalt runs them (approximate: changes the output)");
    }
    if (options.expert_a16_activations) {
        report(options, "Inkling 2: --a16-activations: the routed experts take 16-bit activations instead of the "
                        "checkpoint's four-over-six NVFP4 and MXFP8 ones (changes the output; the default follows the "
                        "trained arithmetic and is recommended)");
    }
    if (options.q8_weights && !options.fast_paths) {
        throw std::invalid_argument("--q8-weights is approximate; the reproducible route (--reproducible) refuses it");
    }
    if (options.q8_weights && !options.a8_activations) {
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
    refuse(options.rope_yarn_factor != 1.0F, "--rope-yarn does not apply: the model has no RoPE (its relative attention "
                                             "and log-scaled scores reach 1,048,576 positions natively)");
    if (options.embedding_host) {
        report(options, "Inkling 2: --embedding-host: the token embedding (0.86 GiB) stays in pinned host memory, its "
                        "rows read over the link");
    }
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
    refuse(!options.ngram_volume_path.empty() || options.speculative.ngram_draft_tokens != 0 ||
               options.speculative.ngram_draft_tokens_auto,
           "no n-gram drafting");
    refuse(options.speculative.draft_tree_auto ||
               std::any_of(options.speculative.draft_tree_nodes.begin(), options.speculative.draft_tree_nodes.end(),
                           [](std::uint32_t nodes) { return nodes != 0; }),
           "DSpark verifies its drafts as a chain (no --draft-tree-nodes)");
    refuse(options.max_context_auto && options.purpose != EnginePurpose::Generation,
           "--max-context auto sizes a generation server: causal scoring takes N");
    refuse(options.max_concurrency == 0 || options.max_concurrency > models::inkling2::kMaximumConcurrency,
           "max_concurrency must be 1-8");
    // 4,096: the fastest chunk measured on the prompt route (2026-11-21, infernix_bench C = 1, pp2048 / pp10000 tok/s:
    // 9,514 / 8,514 at 512, 13,827 / 12,851 at 1,024, 16,400 / 14,926 at 2,048, 16,387 / 16,312 at 4,096); every
    // chunk gives the same bits.
    if (options.prefill_chunk == 0) { options.prefill_chunk = models::inkling2::kDefaultPrefillChunk; }
    refuse(options.prefill_chunk > static_cast<std::uint32_t>(models::inkling2::kMaxPrefillChunk),
           "prefill chunks are at most 8191 columns");
    // The prefix cache (models/inkling2/program/prefix_cache.h): generation only, its snapshots born in the Host tier.
    ContextCacheOptions& cache = options.context_cache;
    if (options.purpose != EnginePurpose::Generation) { cache.enabled = false; }
    if (cache.enabled) {
        HybridPrefixCacheOptions& hybrid = cache.hybrid;
        refuse(cache.host_capacity_bytes && *cache.host_capacity_bytes == 0,
               "the prefix cache keeps its snapshots in the Host tier: --host-context-mib must be positive");
        refuse(hybrid.device_snapshot_slots.value_or(0U) != 0, "the prefix cache has no Device snapshot slots");
        refuse(hybrid.host_state_precision != HybridHostStatePrecision::Fp32,
               "the prefix cache keeps snapshot state exact (--cache-host-state fp32)");
        refuse(hybrid.tool_call_hold.value_or(std::chrono::milliseconds{0}).count() != 0,
               "the prefix cache has no tool-call holds");
        refuse(hybrid.next_turn_prefill_quiet.value_or(std::chrono::milliseconds{0}).count() != 0,
               "no idle next-turn prefill yet");
        hybrid.device_snapshot_slots   = 0U;
        hybrid.tool_call_hold          = std::chrono::milliseconds{0};
        hybrid.next_turn_prefill_quiet = std::chrono::milliseconds{0};
        hybrid.max_new_taps            = hybrid.max_new_taps.value_or(8U);
        hybrid.tap_ladder_tokens = hybrid.tap_ladder_tokens.value_or(std::max<std::uint32_t>(4096U, 2U * options.prefill_chunk));
        hybrid.tap_min_gap_tokens = hybrid.tap_min_gap_tokens.value_or(std::max<std::uint32_t>(1024U, options.prefill_chunk));
    } else {
        cache.host_capacity_bytes = 0U;
    }
    SpeculativeOptions& spec = options.speculative;
    if (spec.backend_auto) {
        // As the Qwen families' --spec auto: the choice follows the drafter's --draft-tokens auto calibration.
        if (!spec.draft_tokens_auto) { throw std::invalid_argument("--spec auto chooses the drafter: it takes --draft-tokens auto"); }
        spec.backend_auto = false;
        if (options.purpose != EnginePurpose::Generation) {
            spec.backend = SpeculativeBackend::None;
        } else if (options.lossless || options.reference) {
            spec.backend = SpeculativeBackend::None;
            report(options, "Inkling 2: --spec auto: no speculation on the lossless and reference routes");
        } else if (bundled_drafter) {
            spec.backend = SpeculativeBackend::DSpark;
            report(options, "Inkling 2: --spec auto drafts with the bundled DSpark drafter (--draft-tokens auto)");
        } else {
            spec.backend = SpeculativeBackend::None;
            report(options, "Inkling 2: --spec auto: the artifact bundles no drafter (convert with --components text,dspark)");
        }
    }
    if (spec.backend == SpeculativeBackend::DSpark) {
        refuse(options.purpose != EnginePurpose::Generation, "DSpark drafts for generation only");
        refuse(options.lossless, "--lossless has no speculation (no --spec dspark)");
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

namespace {

// The largest context every lane's KV holds beside the weights in `free_bytes` less the display headroom (each of the
// --max-concurrency lanes reserves its whole context, so no request is ever paused), up to the model's positions, on
// the 64-position page grid; empty when none fits. `vision_host`: the tower's pinned groups, as the Program plans them.
std::optional<std::uint32_t> largest_context(const EngineOptions& options, const models::inkling2::Config& config,
                                             const models::inkling2::VisionHostWeights* vision_host,
                                             std::uint64_t free_bytes, std::uint64_t headroom) {
    const auto fits = [&](std::uint32_t context) {
        EngineOptions trial                      = options;
        trial.max_context                        = context;
        models::inkling2::ProgramOptions program = program_options(trial);
        program.vision_host                      = vision_host;
        return models::inkling2::Program::plan_device(program, config).fixed_bytes() + headroom <= free_bytes;
    };
    const auto found = largest_fitting_context(config.text.max_position_embeddings, options.max_context, fits);
    if (!found) { return std::nullopt; }
    const std::uint32_t unit = static_cast<std::uint32_t>(ops::inkling2::kKeyUnit);
    return *found >= unit ? *found / unit * unit : *found;
}

} // namespace

ConstructedInkling2 construct_inkling2(const EngineOptions& requested, DeviceContext& device, EnginePlan* planned) {
    const auto start = std::chrono::steady_clock::now();
    const artifact::Reader reader(requested.artifact_path);
    if (!models::inkling2::is_inkling2(reader)) { throw std::invalid_argument("artifact is not Inkling2LoopForCausalLM"); }
    EngineOptions options    = requested; // --max-context auto resolves below
    models::LoadOptions load = models::load_options(options, models::Architecture::Inkling2Loop);
    auto plan                = models::inkling2::plan_load(reader, load);
    const VramSnapshot vram  = open_vram_budget_source(device.device, options.vram_past_budget)->query();
    const std::uint64_t headroom =
        expert_offload::display_headroom(vram.display, false, options.vram_headroom_bytes);
    if (planned != nullptr) {
        // Engine::plan: the free memory a start reads after its weights, so the largest context is the start's.
        const std::uint64_t free_after = planned_free_after_weights(free_device_bytes(vram), plan.device_bytes());
        const auto* vision_host        = plan.vision_host() ? &*plan.vision_host() : nullptr;
        planned->max_context           = largest_context(options, plan.config(), vision_host, free_after, headroom);
        if (options.max_context_auto) {
            if (!planned->max_context) {
                plan_refuse(*planned, "--max-context auto: no context fits beside the weights in the free VRAM");
            } else {
                options.max_context = *planned->max_context;
            }
        }
        models::inkling2::ProgramOptions program = program_options(options);
        program.vision_host                      = vision_host;
        if (program.prefix_cache) { program.kv_pages = cache_pool_pages(options, program, plan.config(), free_after, headroom); }
        const models::inkling2::ProgramDevicePlan device_plan = models::inkling2::Program::plan_device(program, plan.config());
        planned->architecture                   = std::string(models::architecture_name(models::Architecture::Inkling2Loop));
        planned->vram.weights_bytes             = plan.device_bytes();
        planned->vram.kv_bytes                  = device_plan.kv;
        planned->vram.kv_tokens                 = options.max_context * options.max_concurrency;
        planned->vram.minimum_reservation_bytes = device_plan.fixed_bytes();
        planned->vram.workspace_bytes           = device_plan.workspace;
        return {};
    }
    std::string template_override;
    const std::uint64_t pinned_bytes = plan.materialization().pinned_capacity_bytes;
    auto model = models::inkling2::materialize_model(std::move(plan), device, &options.startup_observer);
    const auto* vision_host = model->vision_host() ? &*model->vision_host() : nullptr;
    if (options.max_context_auto) {
        const VramSnapshot after = open_vram_budget_source(device.device, options.vram_past_budget)->query();
        const auto context       = largest_context(options, model->config(), vision_host, free_device_bytes(after), headroom);
        if (!context) { throw std::runtime_error("--max-context auto: no context fits beside the weights in the free VRAM"); }
        options.max_context = *context;
        publish_diagnostic(options.diagnostic_observer, DiagnosticLevel::Info,
                           "--max-context auto: %u tokens, the largest context each of the %u lanes holds here (every "
                           "lane reserves its whole context: no request is paused)",
                           options.max_context, options.max_concurrency);
    }
    models::inkling2::ProgramOptions program = program_options(options);
    program.vision_host                      = vision_host;
    if (program.prefix_cache) {
        const VramSnapshot now                     = open_vram_budget_source(device.device, options.vram_past_budget)->query();
        options.context_cache.host_capacity_bytes  = cache_host_bytes(options, pinned_bytes, now);
        program.prefix_host_bytes                  = *options.context_cache.host_capacity_bytes;
        program.kv_pages = cache_pool_pages(options, program, model->config(), free_device_bytes(now), headroom);
        const models::inkling2::ProgramDevicePlan sized = models::inkling2::Program::plan_device(program, model->config());
        constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
        const std::uint64_t lane_pages =
            static_cast<std::uint64_t>(models::inkling2::pages_per_lane(options.max_context)) * options.max_concurrency;
        publish_diagnostic(options.diagnostic_observer, DiagnosticLevel::Info,
                           "Inkling 2: prefix cache: %u KV pages (%.1f GiB: the lanes' contexts and %llu pages of cached "
                           "blocks), Host tier %.1f GiB",
                           sized.kv_pages - 1U, static_cast<double>(sized.kv) / kGiB,
                           static_cast<unsigned long long>(sized.kv_pages - 1U > lane_pages ? sized.kv_pages - 1U - lane_pages : 0U),
                           static_cast<double>(program.prefix_host_bytes) / kGiB);
    }
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
    // --prefix-cache-file: the Host tier restored from a file written for this artifact and these options (the KV
    // format, the drafter, the 8-bit weights and 16-bit expert activations change what the cache holds).
    LoadSummary::PrefixCacheRestore restore;
    if (program.prefix_cache && !options.context_cache.hybrid.persistent_file.empty()) {
        std::string signature = "inkling2";
        signature += ";q8=" + std::to_string(options.q8_weights ? 1 : 0);
        signature += ";a16=" + std::to_string(options.expert_a16_activations ? 1 : 0);
        signature += ";vision=" + std::to_string(options.enable_vision ? 1 : 0);
        const runtime::prefix_cache::PersistResult loaded = instance->program->attach_prefix_cache_file(
            options.context_cache.hybrid.persistent_file, hybrid_cache_fingerprint(options, signature, device.props),
            options.startup_observer);
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
    summary.prefix_cache         = std::move(restore);

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
