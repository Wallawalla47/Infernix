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

// An option another family applies and nothing on Inkling 2's path reads: refused by its flag, so a start never runs
// without what it asked for.
void refuse_option(bool given, const char* flag, const char* why) {
    if (given) { throw std::invalid_argument(std::string("Inkling 2: ") + flag + " " + why); }
}

constexpr const char* kResidentExperts = "applies to models with offloaded experts; Inkling 2 keeps its experts in VRAM";
constexpr const char* kQwenPromptKernel =
    "selects a Qwen3.5 prompt-attention kernel; Inkling 2 has its own prompt route for every KV format";
constexpr const char* kQwenMedia =
    "sizes the Qwen models' media pipeline; Inkling 2's frontend decodes each image without it";
constexpr const char* kNoNgrams = "configures n-gram drafting, which Inkling 2 does not have (its drafter is DSpark)";

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
            .vision_max_cells        = vision_max_cells(options),
            .grammar_cache_bytes     = options.grammar_cache_bytes};
}

models::inkling2::ProgramOptions program_options(const EngineOptions& options, std::uint32_t lora_rank) {
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
            .lora_rank         = lora_rank,
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
    // As every model (validate_engine_options, which this family's start does not run): the lossless route computes
    // the model's own forward correctly rounded, and a cut is not part of it.
    refuse(!options.ablate_sidecar_path.empty() && options.lossless, "--ablate does not combine with --lossless (docs/ablate.md)");
    // As every model (validate_engine_options): the design reference and the noise twin are the lossless route's
    // causal scoring.
    refuse(options.reference && options.purpose != EnginePurpose::CausalScoring,
           "the design reference (--reference) is the lossless route's causal scoring");
    refuse(options.lossless_noise_seed != 0 && (!options.lossless || options.purpose != EnginePurpose::CausalScoring),
           "the noise twin (lossless_noise_seed) is the lossless route's causal scoring");
    // Inkling 2's own lossless kernels (ops/lossless/lossless_inkling2.cu) round to nearest only: a seed would perturb
    // the shared linear kernels alone, a twin of no defined noise level.
    refuse(options.lossless_noise_seed != 0,
           "the noise twin (lossless_noise_seed, --noise-seed) is built for the Qwen families' lossless kernels, not "
           "Inkling 2's");
    refuse(!(options.lossless_noise_scale > 0.0F && options.lossless_noise_scale <= 1.0F),
           "lossless_noise_scale must be in (0, 1]");
    // The other families' options, which nothing on Inkling 2's path reads (docs/inkling2-mini.md, "Not supported").
    // Options that take a default are refused when they differ from it: a default given explicitly asks for nothing.
    refuse_option(options.original_int8_prefill_kernel, "--use-original-int8-prefill-kernel", kQwenPromptKernel);
    refuse_option(options.original_nvfp4_prefill_kernel, "--use-original-nvfp4-prefill-kernel", kQwenPromptKernel);
    refuse_option(options.prefill_split_workspace_mib != kDefaultPrefillSplitWorkspaceMiB, "--prefill-split-workspace-mib",
                  "sizes the Qwen3.5 prompt-attention split; Inkling 2's prompt attention takes no split workspace");
    refuse_option(!options.fast_path_candidates.empty(), "--fast-path",
                  "selects a registered fast path; Inkling 2 registers none (its approximate routes are --q8-weights and "
                  "--a8-activations)");
    refuse_option(!options.fast_path_exclusions.empty(), "--no-fast-path",
                  "leaves out a registered fast path; Inkling 2 registers none");
    refuse_option(options.expert_tail_skip != 0, "--expert-tail-skip", kResidentExperts);
    refuse_option(options.expert_ram_bytes.has_value(), "--expert-ram-mib", kResidentExperts);
    refuse_option(options.expert_ram_layout.has_value(), "--expert-ram-layout", kResidentExperts);
    refuse_option(options.expert_csf, "--expert-csf", kResidentExperts);
    refuse_option(options.expert_pin_bytes.has_value(), "--expert-pin-mib", kResidentExperts);
    // --expert-state FILE cannot be told from infernix-serve's and the CLI's default path here: they refuse it
    // (serve::refuse_unsupported_options, the CLI's start).
    refuse_option(options.expert_state_off, "--expert-state", kResidentExperts);
    refuse_option(!options.expert_stripe_path.empty(), "--expert-stripe", kResidentExperts);
    refuse_option(!options.route_trace_path.empty(), "--route-trace", kResidentExperts);
    refuse_option(!options.context_cost.preset_path.empty(), "--context-cost-presets",
                  "calibrates the Qwen prefix caches' cost model; Inkling 2's prefix cache has none");
    refuse_option(!options.context_cache.hybrid.queue_holds, "--no-queue-holds",
                  "turns off the queue holds of the Qwen prefix caches; Inkling 2's prefix cache builds none");
    refuse_option(options.media_cache_bytes != kDefaultMediaCacheBytes, "--media-cache-mib", kQwenMedia);
    refuse_option(options.media_live_bytes != kDefaultMediaLiveBytes, "--media-live-mib", kQwenMedia);
    refuse_option(options.media_preprocess_threads != 0, "--media-preprocess-threads", kQwenMedia);
    refuse_option(!options.enable_vision && options.vision_max_merged_tokens != 0 &&
                      options.vision_max_merged_tokens != 32768U,
                  "--vision-max-merged", "bounds an image's tokens: it needs --vision");
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
    const SpeculativeOptions defaults;
    const SpeculativeOptions& asked = options.speculative;
    refuse_option(!options.ngram_volume_path.empty(), "--ngram-volume", kNoNgrams);
    refuse_option(options.ngram_residency != NgramResidency::Auto, "--ngram-residency", kNoNgrams);
    refuse_option(asked.ngram_draft_tokens != 0 || asked.ngram_draft_tokens_auto, "--ngram-draft-tokens", kNoNgrams);
    refuse_option(asked.ngram_min_match != defaults.ngram_min_match, "--ngram-min-match", kNoNgrams);
    refuse_option(asked.ngram_tree_nodes != 0, "--ngram-tree-nodes", kNoNgrams);
    refuse_option(asked.ngram_batch_draft_tokens != 0, "ngram_batch_draft_tokens", kNoNgrams);
    refuse_option(asked.ngram_archive_bytes != 0, "--ngram-archive-mib", kNoNgrams);
    // The resolved options carry 0 (below), and a start from them is the same start.
    refuse_option(asked.ngram_session_bytes != defaults.ngram_session_bytes && asked.ngram_session_bytes != 0,
                  "--ngram-session-mib", kNoNgrams);
    refuse_option(!asked.draft_model.empty(), "--draft-model",
                  "names Qwen3.8-Flash-Next's DFlash2 companion; Inkling 2's drafter is the artifact's DSpark component");
    refuse_option(asked.draft_min_p.has_value(), "--spec-min-p",
                  "tunes MTP and DFlash2 rounds; DSpark drafts --draft-tokens");
    refuse(asked.draft_tree_auto ||
               std::any_of(asked.draft_tree_nodes.begin(), asked.draft_tree_nodes.end(),
                           [](std::uint32_t nodes) { return nodes != 0; }) ||
               asked.draft_tree_paths != defaults.draft_tree_paths,
           "DSpark verifies its drafts as a chain (no --draft-tree-nodes or --draft-tree-paths)");
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
    // What it does not have is refused on every purpose.
    ContextCacheOptions& cache = options.context_cache;
    {
        const HybridPrefixCacheOptions& hybrid = cache.hybrid;
        refuse(hybrid.device_snapshot_slots.value_or(0U) != 0, "the prefix cache has no Device snapshot slots");
        refuse(hybrid.host_state_precision != HybridHostStatePrecision::Fp32,
               "the prefix cache keeps snapshot state exact (--cache-host-state fp32)");
        refuse(hybrid.tool_call_hold.value_or(std::chrono::milliseconds{0}).count() != 0,
               "the prefix cache has no tool-call holds");
        refuse(hybrid.next_turn_prefill_quiet.value_or(std::chrono::milliseconds{0}).count() != 0,
               "no idle next-turn prefill yet");
    }
    if (options.purpose == EnginePurpose::Generation) {
        // The VRAM and RAM figures size --max-context auto, --kv-capacity auto (the prefix cache's Device pool) and
        // the automatic Host tier (--host-context-mib auto); a start that sizes none of them would leave them unread.
        const bool sizes_vram = options.max_context_auto ||
                                (cache.enabled && options.kv_capacity.mode == KvCapacityMode::Automatic);
        refuse_option(options.vram_headroom_bytes.has_value() && !sizes_vram, "--vram-headroom-mib",
                      "sizes --max-context auto and the prefix cache's --kv-capacity auto; this start sizes neither");
        refuse_option(options.vram_past_budget && !sizes_vram, "--vram-past-budget",
                      "widens the VRAM --max-context auto and the prefix cache's --kv-capacity auto size from; this "
                      "start sizes neither");
        refuse_option(options.ram_free_bytes != kDefaultRamFreeBytes &&
                          !(cache.enabled && !cache.host_capacity_bytes.has_value()),
                      "--ram-free-mib", "sizes the prefix cache's automatic Host tier (--host-context-mib auto); this "
                      "start has none");
        if (cache.enabled && options.kv_capacity.mode == KvCapacityMode::Explicit) {
            // Each lane reserves its whole context in the pool, so a smaller pool would be raised to it.
            refuse_option(options.max_context_auto, "--kv-capacity N",
                          "with --max-context auto: each lane reserves its whole context, which --max-context auto "
                          "sizes from the free VRAM (give --max-context N, or --kv-capacity auto)");
            refuse_option(std::uint64_t{options.kv_capacity.explicit_tokens} <
                              std::uint64_t{options.max_context} * options.max_concurrency,
                          "--kv-capacity N",
                          "must hold every lane's whole context (--max-context x --max-concurrency): each lane "
                          "reserves it in the prefix cache's pool");
        }
    } else {
        // Causal scoring runs without the prefix cache, the drafter or the automatic sizings: as the other models
        // (normalize_engine_options), a scoring start with a generation configuration says what it leaves unused.
        std::string unused;
        const auto note = [&unused](const char* what) { unused += (unused.empty() ? "" : ", ") + std::string(what); };
        if (!cache.hybrid.persistent_file.empty()) { note("--prefix-cache-file"); }
        if (cache.host_capacity_bytes.value_or(0U) != 0) { note("--host-context-mib"); }
        if (cache.ssd_capacity_bytes != 0) { note("--ssd-context-gib"); }
        if (options.speculative.proposal_head == ProposalHead::Optimized) { note("--lm-head-draft"); }
        if (options.vram_headroom_bytes) { note("--vram-headroom-mib"); }
        if (options.vram_past_budget) { note("--vram-past-budget"); }
        if (options.ram_free_bytes != kDefaultRamFreeBytes) { note("--ram-free-mib"); }
        if (!unused.empty()) {
            report(options, "Inkling 2: causal scoring runs without the prefix cache, the drafter and the automatic "
                            "memory sizing: " + unused + " unused");
        }
        cache.enabled = false;
    }
    if (cache.enabled) {
        HybridPrefixCacheOptions& hybrid = cache.hybrid;
        refuse(cache.host_capacity_bytes && *cache.host_capacity_bytes == 0,
               "the prefix cache keeps its snapshots in the Host tier: --host-context-mib must be positive");
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
    refuse_option(options.purpose == EnginePurpose::Generation && spec.proposal_head == ProposalHead::Optimized &&
                      spec.backend != SpeculativeBackend::DSpark,
                  "--lm-head-draft", "shortlists DSpark's drafts: this start runs no drafter (--spec auto finds none "
                  "bundled, or the route has no speculation)");
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
                                             std::uint32_t lora_rank, std::uint64_t free_bytes, std::uint64_t headroom) {
    const auto fits = [&](std::uint32_t context) {
        EngineOptions trial                      = options;
        trial.max_context                        = context;
        models::inkling2::ProgramOptions program = program_options(trial, lora_rank);
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
    // --lora's factors and --ablate's direction, allocated with the Parameters after the free memory is read below.
    const std::uint32_t lora_rank = plan.lora() != nullptr ? plan.lora()->padded_rank : 0;
    const std::uint64_t adapter_bytes = (plan.lora() != nullptr ? plan.lora()->device_bytes(options.lossless) : 0) +
                                        (plan.ablate() != nullptr ? plan.ablate()->device_bytes() : 0);
    if (planned != nullptr) {
        // Engine::plan: the free memory a start reads after its weights, so the largest context is the start's.
        const std::uint64_t free_after =
            planned_free_after_weights(free_device_bytes(vram), plan.device_bytes(), adapter_bytes);
        const auto* vision_host        = plan.vision_host() ? &*plan.vision_host() : nullptr;
        planned->max_context           = largest_context(options, plan.config(), vision_host, lora_rank, free_after, headroom);
        if (options.max_context_auto) {
            if (!planned->max_context) {
                plan_refuse(*planned, "--max-context auto: no context fits beside the weights in the free VRAM");
            } else {
                options.max_context = *planned->max_context;
            }
        }
        models::inkling2::ProgramOptions program = program_options(options, lora_rank);
        program.vision_host                      = vision_host;
        if (program.prefix_cache) { program.kv_pages = cache_pool_pages(options, program, plan.config(), free_after, headroom); }
        const models::inkling2::ProgramDevicePlan device_plan = models::inkling2::Program::plan_device(program, plan.config());
        planned->architecture                   = std::string(models::architecture_name(models::Architecture::Inkling2Loop));
        planned->vram.weights_bytes             = plan.device_bytes();
        if (adapter_bytes != 0) { planned->vram.adapter_bytes = adapter_bytes; }
        planned->vram.kv_bytes                  = device_plan.kv;
        planned->vram.kv_tokens                 = options.max_context * options.max_concurrency;
        planned->vram.minimum_reservation_bytes = device_plan.fixed_bytes();
        planned->vram.workspace_bytes           = device_plan.workspace;
        planned->vram.planned_bytes =
            planned->vram.weights_bytes + planned_beside_weights_bytes(planned->vram) + device_plan.fixed_bytes();
        planned->vram.headroom_bytes = headroom;
        // The readings the plan was made against, and whether the start fits them (an explicit --max-context,
        // --kv-capacity or --max-concurrency that does not fit is refused here, as its start would fail).
        const HostMemorySnapshot host = host_memory_snapshot();
        plan_readings(*planned, device.props.name, vram, host, options);
        const std::uint64_t free_now = free_device_bytes(vram);
        const std::uint64_t used     = planned->vram.planned_bytes + headroom;
        planned->vram.left_bytes     = free_now > used ? free_now - used : 0;
        if (used > free_now) {
            plan_refuse(*planned, "the weights, KV cache and workspaces need " + std::to_string(used >> 20U) +
                                      " MiB with the display headroom, and " + std::to_string(free_now >> 20U) +
                                      " MiB of VRAM is free");
        }
        // RAM: the weights the load pins (the vision tower offloaded with --vision-offload on) and the Host tier.
        const std::uint64_t pinned = plan.materialization().pinned_capacity_bytes;
        if (pinned != 0) { planned->ram.pinned_bytes = pinned; }
        std::uint64_t host_tier = 0;
        if (program.prefix_cache) {
            host_tier                       = cache_host_bytes(options, pinned, vram);
            planned->ram.host_context_bytes = host_tier;
        }
        planned->ram.planned_bytes      = pinned + host_tier;
        const std::uint64_t spare       = host.available_physical > options.ram_free_bytes
                                              ? host.available_physical - options.ram_free_bytes
                                              : 0U;
        planned->ram.left_bytes         = spare > planned->ram.planned_bytes ? spare - planned->ram.planned_bytes : 0U;
        if (planned->ram.planned_bytes > spare) {
            plan_refuse(*planned, "the pinned weights and the Host tier need " +
                                      std::to_string(planned->ram.planned_bytes >> 20U) + " MiB of RAM, and " +
                                      std::to_string(spare >> 20U) + " MiB is available beyond --ram-free-mib");
        }
        return {};
    }
    std::string template_override;
    const std::uint64_t pinned_bytes = plan.materialization().pinned_capacity_bytes;
    auto model = models::inkling2::materialize_model(std::move(plan), device, &options.startup_observer);
    const auto* vision_host = model->vision_host() ? &*model->vision_host() : nullptr;
    if (options.max_context_auto) {
        const VramSnapshot after = open_vram_budget_source(device.device, options.vram_past_budget)->query();
        const std::uint64_t free_after = free_device_bytes(after) > adapter_bytes ? free_device_bytes(after) - adapter_bytes : 0;
        const auto context = largest_context(options, model->config(), vision_host, lora_rank, free_after, headroom);
        if (!context) { throw std::runtime_error("--max-context auto: no context fits beside the weights in the free VRAM"); }
        options.max_context = *context;
        publish_diagnostic(options.diagnostic_observer, DiagnosticLevel::Info,
                           "--max-context auto: %u tokens, the largest context each of the %u lanes holds here (every "
                           "lane reserves its whole context: no request is paused)",
                           options.max_context, options.max_concurrency);
    }
    models::inkling2::ProgramOptions program = program_options(options, lora_rank);
    program.vision_host                      = vision_host;
    if (program.prefix_cache) {
        const VramSnapshot now                     = open_vram_budget_source(device.device, options.vram_past_budget)->query();
        options.context_cache.host_capacity_bytes  = cache_host_bytes(options, pinned_bytes, now);
        program.prefix_host_bytes                  = *options.context_cache.host_capacity_bytes;
        const std::uint64_t free_now = free_device_bytes(now) > adapter_bytes ? free_device_bytes(now) - adapter_bytes : 0;
        program.kv_pages = cache_pool_pages(options, program, model->config(), free_now, headroom);
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
    // --lora / --ablate: what the start applied (the factors and the direction were allocated with the Parameters).
    if (const auto* lora = instance->model->lora()) {
        publish_diagnostic(options.diagnostic_observer, DiagnosticLevel::Info, "LoRA adapter %s: %s; %llu MiB of VRAM",
                           lora->adapter->weights_path().parent_path().string().c_str(), lora->describe().c_str(),
                           static_cast<unsigned long long>((instance->parameters.lora->factors.bytes + (1U << 20) - 1) >> 20));
    }
    if (const auto* ablate = instance->model->ablate()) {
        publish_diagnostic(options.diagnostic_observer, DiagnosticLevel::Info, "ablation sidecar %s: %s",
                           ablate->path.string().c_str(), ablate->describe().c_str());
        if (const std::string warning = ablate->artifact_mismatch(instance->model->info().artifact_id); !warning.empty()) {
            publish_diagnostic(options.diagnostic_observer, DiagnosticLevel::Warning, "%s", warning.c_str());
        }
    }
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
    if (const auto* lora = instance->model->lora()) {
        summary.lora_adapter = lora->adapter->weights_path().parent_path().string() + ": " + lora->describe();
    }
    if (const auto* ablate = instance->model->ablate()) {
        summary.ablate_sidecar = ablate->path.string() + ": " + ablate->describe();
    }

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
