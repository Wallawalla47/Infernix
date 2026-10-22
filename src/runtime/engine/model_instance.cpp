#include "runtime/engine/model_instance.h"
#include "artifact/reader.h"
#include "artifact/formats.h"
#include "core/decode_graph.h"
#include "core/host_memory.h"
#include "core/startup.h"
#include "core/vram_budget.h"
#include "models/qwen3_5/load.h"
#include "models/qwen3_5/measurement.h"
#include "models/qwen3_5/program/offload.h"
#include "models/qwen3_5/program/planning/fast_paths.h"
#include "runtime/engine/diagnostics.h"
#include "runtime/engine/expert_memory.h"
#include "runtime/expert_offload/route_trace.h"
#include "runtime/expert_offload/vram_monitor.h"

#include <algorithm>
#include <system_error>
#include <string>
#include <filesystem>
#include <chrono>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <utility>

namespace infernix::runtime {
namespace {
using Clock = std::chrono::steady_clock;

// The extra VRAM headroom of the one startup retry after the CUDA Graphs found no device memory
// (NInfer 193675826's calibration retry, adapted): their allowance is measured at 2.2-4.1 MiB per
// executable, and 1 GiB covers the largest graph set (~274 MiB) several times.
constexpr std::size_t kGraphRetryHeadroomBytes = std::size_t{1} << 30;
} // namespace

void validate_engine_options(const EngineOptions& options, models::Architecture architecture,
                             bool offloaded_experts) {
    if (options.artifact_path.empty()) {
        throw std::invalid_argument("Engine artifact_path must not be empty");
    }
    // .infernix and .ninfer name the same v3 container (Infernix began as a fork of NInfer).
    if (const auto extension = options.artifact_path.extension();
        extension != ".infernix" && extension != ".ninfer") {
        throw std::invalid_argument("Infernix accepts only .infernix and .ninfer artifacts");
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
        // (with offloaded experts, as on Qwen3.8-Flash-Next, it is what the expert cache leaves,
        // with any KV capacity).
        if (architecture == models::Architecture::Qwen3_5 && !offloaded_experts &&
            options.vram_headroom_bytes.has_value()) {
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
    if (const auto interval = options.context_cache.hybrid.persistent_save_interval; interval.count() != 0) {
        if (interval.count() < 0) { throw std::invalid_argument("prefix cache save interval must not be negative"); }
        if (options.context_cache.hybrid.persistent_file.empty()) {
            throw std::invalid_argument("a periodic prefix cache save needs a prefix cache file");
        }
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
    // The lossless route is what the fast paths are measured against; the design reference is the lossless
    // route with the configuration's declared steps (FP4-Local's optional W4A4 experts among them).
    if (options.lossless && (options.fast_paths || !options.fast_path_candidates.empty() ||
                             (options.expert_a4_activations && !options.reference))) {
        throw std::invalid_argument("the lossless route runs no fast path: disable fast paths");
    }
    if (options.reference && (!options.lossless || options.purpose != EnginePurpose::CausalScoring)) {
        throw std::invalid_argument("the design reference (--reference) is the lossless route's causal scoring");
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

// The exact artifact, its execution signature, the KV and speculative formats, RoPE scaling, the
// prefill chunk and the product binary's build identity. Any difference makes the saved state meaningless, so the file is
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
    // Qwen3.5's exact-route prefill bits follow the chunk grid, so a state prefilled at another
    // chunk would not be the one this engine computes.
    out += ";chunk=" + std::to_string(options.prefill_chunk);
    out += ";host_state=" +
           std::to_string(static_cast<int>(options.context_cache.hybrid.host_state_precision));
    // Fast paths change the numbers cached KV and state hold; the lossless route keeps its own (V as
    // BF16, its state layout).
    out += ";fast=" + std::to_string(options.fast_paths ? 1 : 0);
    out += ";lossless=" + std::to_string(options.lossless ? 1 : 0);
    for (const FastPath path : options.fast_path_candidates) { out += std::string(",") + fast_path_name(path); }
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

bool artifact_offloads_experts(const std::filesystem::path& artifact) {
    const artifact::Reader reader(artifact);
    return reader.directory().bindings.contains("text/layers/0/moe/experts");
}

EngineOptions normalize_engine_options(EngineOptions options, models::Architecture architecture,
                                       bool offloaded_experts) {
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
        options.context_cache        = ContextCacheOptions{.enabled = false};
        break;
    default:
        throw std::invalid_argument("Engine purpose is invalid");
    }
    if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("Engine max_concurrency must be in [1,8]");
    }

    ContextCacheOptions& cache = options.context_cache;
    if (options.speculative.ngram_archive_bytes != 0 &&
        options.speculative.ngram_draft_tokens == 0 && !options.speculative.ngram_draft_tokens_auto) {
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
    // Offloaded routed experts (Qwen3.8-Flash-Next, Qwen3.5-122B-A10B) share the offload defaults.
    const bool offloaded            = qwen4_exp || offloaded_experts;
    // --draft-tokens auto where no draft length beats plain decode (the configuration's measurement,
    // models::qwen3_5::auto_drafts_nothing): the engine decodes without a drafter, which it then
    // never binds, before any setting follows the speculative backend.
    // Forced adaptive policies (--fast-path adaptive-speculation with adaptive-dflash or adaptive-mtp) keep
    // the drafter: they choose per round, down to no draft (design §12.2's A/B).
    const auto forced = [&](FastPath path) {
        return std::find(options.fast_path_candidates.begin(), options.fast_path_candidates.end(), path) !=
               options.fast_path_candidates.end();
    };
    const bool forced_adaptive = forced(FastPath::AdaptiveSpeculation) &&
                                 (forced(FastPath::AdaptiveDFlash) || forced(FastPath::AdaptiveMtp));
    if (!qwen4_exp && options.speculative.draft_tokens_auto && options.speculative.backend != SpeculativeBackend::None &&
        models::qwen3_5::auto_drafts_nothing(offloaded_experts) && !forced_adaptive) {
        publish_diagnostic(options.diagnostic_observer, DiagnosticLevel::Info,
                           "--draft-tokens auto: plain decode is faster than any draft length here (each verified "
                           "draft brings its own offloaded-expert misses), so the engine decodes without a "
                           "drafter");
        options.speculative.backend           = SpeculativeBackend::None;
        options.speculative.draft_tokens      = 0;
        options.speculative.draft_tokens_auto = false;
        options.speculative.proposal_head     = ProposalHead::Full;
    }
    // The automatic copy window is measured on the Qwen3.5 family's drafters only.
    if (options.speculative.ngram_draft_tokens_auto && qwen4_exp) {
        throw std::invalid_argument("--ngram-draft-tokens auto applies to Qwen3.8-27B / Qwen3.6-35B-A3B; "
                                    "give Qwen3.8-Flash-Next a window (docs/qwen3_8-flash-next.md)");
    }
    // The model's default prefill chunk. Qwen3.8-Flash-Next's exact route split at 1,024, 4,096
    // and 12,288 tokens gave identical bits (36 perplexity windows of 14,336 tokens; its calls
    // split on the 16-token GDN grid and its routes are column-invariant), and wide chunks prefill
    // faster with no decode cost: Dense8, INT8 KV, RTX 5090,
    // pp32768 9,639 -> 11,211 tok/s from 4,096 to 16,384 (pp65536 9,887 -> 11,268; 26 fewer expert
    // frames of ~8,560). The Qwen3.5 family keeps 1,024: its prefill bits follow the chunk grid,
    // and idle-wide-prefill already widens its idle steps (16,384 measured 3.5 % slower at
    // pp16384 and pp32768 on the 27B). A Qwen3.5 artifact with offloaded experts starts from
    // Flash-Next's 16,384: small chunks are link-bound there (122B design §10).
    if (options.prefill_chunk == 0) { options.prefill_chunk = offloaded ? 16384U : 1024U; }
    if (options.expert_tail_skip != 0 && !offloaded) {
        throw std::invalid_argument("--expert-tail-skip applies to artifacts with offloaded experts only");
    }
    // The lossless route's offloaded experts read each call's job counts on the host, so its rounds run
    // eagerly (as Qwen3.8-Flash-Next's lossless rounds do).
    if (options.lossless && offloaded_experts && !qwen4_exp) { options.use_cuda_graph = false; }
    // --vision-offload auto with offloaded experts: the tower stays in pinned RAM and the expert cache
    // lends each encode window's device memory (122B design D8), as on Qwen3.8-Flash-Next.
    if (offloaded_experts && options.vision_offload == VisionOffload::Auto) { options.vision_offload = VisionOffload::On; }
    if (options.expert_a4_activations && !qwen4_exp) {
        throw std::invalid_argument("--expert-a4-activations applies to nvidia/Qwen3.8-Flash-Next-FP4-Local only");
    }
    // An approximation whose output depends on the cache state: never on the exact route.
    if (options.expert_tail_skip != 0 && !options.fast_paths) {
        throw std::invalid_argument("--expert-tail-skip is approximate; the reproducible route (--reproducible) refuses it");
    }
    // An automatic draft length chooses up to the model's most drafts unless given a ceiling: MTP on
    // Qwen3.8-Flash-Next verifies 8 columns column-invariantly, the Qwen3.5 planner captures 5. The
    // block drafters propose their whole block: Qwen3.8-Flash-Next's DFlash2 7; the Qwen3.5 planner
    // resolves its DFlash and DFlash2 blocks per configuration (resolve_auto_draft_length).
    const bool block_drafter = options.speculative.backend == SpeculativeBackend::DFlash ||
                               options.speculative.backend == SpeculativeBackend::DFlash2;
    if (options.speculative.draft_tokens_auto) {
        if (options.speculative.backend != SpeculativeBackend::Mtp && !block_drafter) {
            throw std::invalid_argument("--draft-tokens auto requires --spec mtp, dflash or dflash2");
        }
        if (options.speculative.draft_tokens == 0 && (qwen4_exp || !block_drafter)) {
            options.speculative.draft_tokens = qwen4_exp ? 7U : 5U;
        }
    } else {
        for (const FastPath policy : {FastPath::AdaptiveMtp, FastPath::AdaptiveDFlash}) {
            if (std::find(options.fast_path_candidates.begin(), options.fast_path_candidates.end(), policy) !=
                options.fast_path_candidates.end()) {
                throw std::invalid_argument(std::string("--fast-path ") + fast_path_name(policy) +
                                            " is the policy of --draft-tokens auto; a fixed --draft-tokens N "
                                            "drafts N");
            }
        }
    }
    // The draft-confidence gate exists only on Qwen3.8-Flash-Next's one-row MTP rounds; another
    // model refuses it rather than ignoring a tuning value.
    if (const auto& p = options.speculative.draft_min_p) {
        if (!(*p >= 0.0F && *p <= 1.0F)) {
            throw std::invalid_argument("speculative draft_min_p (--spec-min-p) must be in [0,1]");
        }
        if (!qwen4_exp || (options.speculative.backend != SpeculativeBackend::Mtp &&
                           options.speculative.backend != SpeculativeBackend::DFlash2)) {
            throw std::invalid_argument(
                "--spec-min-p applies to Qwen3.8-Flash-Next with --spec mtp or dflash2 only");
        }
    }
    if (cache.enabled) {
        // One pinned Host slab pool serves blocks and snapshots alike; its size is the only
        // capacity a deployment has to choose (docs/maintainer/hybrid-prefix-cache-spec.md §5.4).
        cache.host_capacity_bytes = cache.host_capacity_bytes.value_or(kDefaultHybridHostCacheBytes);
        const bool host_tier             = *cache.host_capacity_bytes != 0;
        HybridPrefixCacheOptions& hybrid = cache.hybrid;
        // Whether a turn resumes from lossy Host state depends on what eviction left, so the exact
        // route's output would depend on load (model-quality 6.1).
        if (!options.fast_paths && hybrid.host_state_precision != HybridHostStatePrecision::Fp32) {
            throw std::invalid_argument(
                "--cache-host-state fp24 and bf16 are lossy; --reproducible and --lossless require fp32");
        }
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
            if (hybrid.host_state_precision != HybridHostStatePrecision::Fp32) {
                throw std::invalid_argument(
                    "Qwen3.8-Flash-Next's prefix cache keeps snapshot state exact on the Host tier");
            }
            // Its endpoints publish asynchronously, after the lane that captured them may be
            // reused; holding them needs a capture-to-snapshot handover it does not have.
            if (hybrid.tool_call_hold.value_or(std::chrono::milliseconds{0}).count() != 0) {
                throw std::invalid_argument(
                    "Qwen3.8-Flash-Next's prefix cache has no tool-call holds");
            }
        }
        hybrid.tool_call_hold =
            hybrid.tool_call_hold.value_or(qwen4_exp ? std::chrono::milliseconds{0} : kDefaultToolCallHold);
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
        // Qwen3.8-Flash-Next splits its prefill calls at taps instead (free-cache-resume), so its
        // ladder keeps the 4,096-token chunk's spacing whatever wider chunk it prefills with.
        const std::uint32_t chunk = std::max<std::uint32_t>(
            qwen4_exp ? std::min<std::uint32_t>(options.prefill_chunk, 4096U) : options.prefill_chunk,
            64U);
        hybrid.tap_ladder_tokens =
            hybrid.tap_ladder_tokens.value_or(std::max<std::uint32_t>(4096U, 2U * chunk));
        hybrid.tap_min_gap_tokens =
            hybrid.tap_min_gap_tokens.value_or(std::max<std::uint32_t>(1024U, chunk));
        if ((!qwen4_exp && *hybrid.device_snapshot_slots == 0) || *hybrid.device_snapshot_slots > 64) {
            throw std::invalid_argument("hybrid device snapshot slots must be in [1,64]");
        }
        if (hybrid.tool_call_hold->count() < 0) {
            throw std::invalid_argument("hybrid tool-call hold must not be negative");
        }
        if (*hybrid.max_new_taps > 64) {
            throw std::invalid_argument("hybrid taps per request must be at most 64");
        }
        if (*hybrid.tap_ladder_tokens < 64 || *hybrid.tap_min_gap_tokens < 64) {
            throw std::invalid_argument(
                "hybrid tap ladder and minimum gap must be at least 64 tokens");
        }
    } else {
        // Without a prefix cache every request starts from the root and nothing is retained:
        // the lanes' StateImages are the only Device state and there is no Host tier.
        if (cache.host_capacity_bytes.value_or(0U) != 0 ||
            cache.hybrid.device_snapshot_slots.value_or(0U) != 0) {
            throw std::invalid_argument(
                "prefix cache capacities require the context cache to be enabled");
        }
        if (cache.hybrid.tool_call_hold.value_or(std::chrono::milliseconds{0}).count() != 0 ||
            cache.hybrid.host_state_precision != HybridHostStatePrecision::Fp32) {
            throw std::invalid_argument("prefix cache policies require the context cache to be enabled");
        }
        cache.host_capacity_bytes          = 0U;
        cache.hybrid.device_snapshot_slots = 0U;
        cache.hybrid.tool_call_hold        = std::chrono::milliseconds{0};
    }
    return options;
}

ModelInstance::ModelInstance(std::unique_ptr<models::qwen3_5::Model> source,
                             const EngineOptions& options)
    : model(std::move(source)), parameters(*model),
      frontend(models::qwen3_5::make_frontend(
          model->resources(),
          {.chat_template_path       = options.chat_template_path,
           .grammar_cache_bytes      = options.grammar_cache_bytes,
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
    bool offloaded_experts = false;
    try {
        offloaded_experts = artifact_offloads_experts(options.artifact_path);
    } catch (const std::exception&) {} // an unreadable artifact is reported below, after the option checks
    validate_engine_options(options, models::Architecture::Qwen3_5, offloaded_experts);
    const auto start = Clock::now();
    StartupPhaseScope inspect(options.startup_observer, StartupPhase::ArtifactInspect);
    artifact::Reader reader(options.artifact_path);
    inspect.complete();
    StartupPhaseScope binding(options.startup_observer, StartupPhase::TargetPlan);
    auto plan = models::qwen3_5::plan_load(reader, models::load_options(options, models::Architecture::Qwen3_5));
    // Offloaded experts (Qwen3.5-122B-A10B design §11.2): the RAM ledger places the routed banks in
    // pinned RAM or in the SSD tier before anything is pinned or read.
    std::optional<ExpertRamPlan> expert_ram;
    if (options.expert_tail_skip != 0) {
        // normalize_engine_options admitted it for an artifact with offloaded experts.
        const auto& moe = std::get<models::qwen3_5::MoeConfig>(plan.config().text.ffn);
        if (options.expert_tail_skip >= moe.num_experts_per_tok) {
            throw std::invalid_argument("--expert-tail-skip takes a top-" + std::to_string(moe.num_experts_per_tok) +
                                        " rank 1-" + std::to_string(moe.num_experts_per_tok - 1) + " (0: off)");
        }
        if (options.purpose != EnginePurpose::Generation || options.lossless) {
            throw std::invalid_argument("--expert-tail-skip applies to generation (its decode rounds) only, not "
                                        "on the lossless route");
        }
    }
    if (plan.offloaded_experts()) {
        const auto vram   = open_vram_budget_source(device.device, options.vram_past_budget);
        auto demand = expert_ram_demand(
            options, {.expert_banks = plan.pinned_expert_bytes(), .other_pinned = plan.pinned_other_bytes()},
            options.context_cache.enabled ? options.context_cache.host_capacity_bytes.value_or(0U) : 0U,
            vram->query());
        // The frontend's caches grow to their caps after planning: the grammar cache, the n-gram
        // archive, and with vision the decoded-media cache and its live items.
        demand.pageable = options.grammar_cache_bytes + options.speculative.ngram_archive_bytes +
                          (options.enable_vision ? options.media_cache_bytes + options.media_live_bytes : 0U);
        expert_ram = plan_expert_ram(options, demand, host_memory_snapshot());
        if (expert_ram->tier) {
            auto streamed           = models::load_options(options, models::Architecture::Qwen3_5);
            streamed.stream_experts = true;
            plan                    = models::qwen3_5::plan_load(reader, streamed);
        }
        plan.set_host_reserve(options.ram_headroom_bytes, expert_ram->demand.later_pinned);
    } else {
        plan.set_host_reserve(options.ram_headroom_bytes);
    }
    // The Program's offloaded experts: the engine's options from these options and the ledger's tier.
    std::optional<models::qwen3_5::detail::ProgramOffloadOptions> offload_options;
    if (expert_ram) {
        offload_options.emplace();
        auto& o              = offload_options->engine;
        o.vram_headroom      = options.vram_headroom_bytes;
        o.vram_past_budget   = options.vram_past_budget;
        o.expert_tail_skip   = options.expert_tail_skip;
        o.route_trace        = options.route_trace_path.empty() ? runtime::expert_offload::testing::route_trace()
                                                                : options.route_trace_path;
        o.vram_grow_delay_seconds = runtime::expert_offload::testing::vram_grow_delay();
        // The CPU expert team keeps its default 6 workers: on the 122B, 4 measured 6 % slower and 8, 10
        // and 12 no faster (design §0.2), as on Flash-Next.
        // The lossless route computes every expert on the GPU: no CPU split. Its wide passes read the
        // prefill stream's ring (each missing expert copied once per pass, under the earlier layers).
        if (options.lossless) { o.prefill_cpu_split = false; }
        if (!options.expert_state_path.empty()) {
            // The state belongs to the artifact: its experts and their records, not the build or KV format.
            std::error_code error;
            const auto size         = std::filesystem::file_size(options.artifact_path, error);
            const auto time         = std::filesystem::last_write_time(options.artifact_path, error);
            o.expert_state          = options.expert_state_path;
            o.expert_state_identity = "artifact=" + std::filesystem::absolute(options.artifact_path).string() +
                                      ";size=" + std::to_string(error ? 0U : size) +
                                      ";mtime=" + std::to_string(error ? 0 : time.time_since_epoch().count());
        }
        if (expert_ram->tier) {
            o.expert_ram_bytes      = expert_ram->expert_ram_bytes;
            o.expert_unpinned_bytes = expert_ram->expert_unpinned_bytes;
            o.later_pinned_bytes    = expert_ram->later_pinned_bytes;
            o.expert_stripe         = options.expert_stripe_path;
        } else if (!options.expert_stripe_path.empty()) {
            publish_diagnostic(options.diagnostic_observer, DiagnosticLevel::Info,
                               "--expert-stripe is not used: every expert fits in RAM, so none is read from disk");
        }
        if (options.expert_tail_skip != 0) {
            publish_diagnostic(options.diagnostic_observer, DiagnosticLevel::Warning,
                               "--expert-tail-skip %u: decode and verification rounds skip a routed expert missing "
                               "from VRAM that every token routes at rank %u or lower; output changes (lower "
                               "quality) for faster decode",
                               options.expert_tail_skip, options.expert_tail_skip);
        }
        offload_options->diagnostic = [observer = options.diagnostic_observer](const std::string& text,
                                                                                DiagnosticLevel level) {
            report_diagnostic(observer, level, text);
        };
    }
    binding.complete();
    // Offloaded experts: the dense weights must fit beside the headroom, and land in device memory
    // (WDDM can place an allocation in shared system memory silently); the expert frames take what the
    // Program leaves (ExpertOffload::size_frames warns when few fit).
    std::unique_ptr<VramBudgetSource> spill_source;
    std::optional<SpillGuard> spill;
    std::uint64_t dense_bytes = 0;
    if (expert_ram) {
        spill_source              = open_vram_budget_source(device.device, options.vram_past_budget);
        dense_bytes               = plan.materialization().device_capacity_bytes;
        const VramSnapshot before = spill_source->query();
        const std::uint64_t headroom =
            runtime::expert_offload::display_headroom(before.display, false, options.vram_headroom_bytes);
        if (dense_bytes + headroom > before.device_free) {
            throw std::runtime_error("the model's dense weights (" + std::to_string(dense_bytes >> 20) +
                                     " MiB) do not fit in the free device memory (" +
                                     std::to_string(before.device_free >> 20) + " MiB) beside the " +
                                     std::to_string(headroom >> 20) + " MiB VRAM headroom; free VRAM");
        }
        spill.emplace(*spill_source);
        spill->begin();
    }
    auto model =
        models::qwen3_5::materialize_model(std::move(plan), device, &options.startup_observer);
    device.synchronize();
    if (spill) {
        if (const std::uint64_t spilled = spill->end(dense_bytes); spilled != 0) {
            throw std::runtime_error(std::to_string(spilled >> 20) +
                                     " MiB of the dense weights were placed in shared system memory by the driver's "
                                     "sysmem fallback: free VRAM and start again");
        }
    }
    StartupPhaseScope frontend(options.startup_observer, StartupPhase::FrontendInitialize);
    // --ngram-draft-tokens auto: the configuration's copy window, before the frontend decides
    // whether prompts build copy sources.
    if (options.speculative.ngram_draft_tokens_auto) {
        if (const std::uint32_t window = models::qwen3_5::resolve_auto_ngram(*model, options); window != 0) {
            publish_diagnostic(options.diagnostic_observer, DiagnosticLevel::Info,
                               "--ngram-draft-tokens auto: copy proposals of up to %u drafts, minimum match %u",
                               window, options.speculative.ngram_min_match);
        } else {
            publish_diagnostic(options.diagnostic_observer, DiagnosticLevel::Info,
                               "--ngram-draft-tokens auto: no copy window was measured faster for this "
                               "artifact, drafter and route, so copy proposals are off");
            if (options.speculative.ngram_archive_bytes != 0) {
                throw std::invalid_argument("--ngram-archive-mib needs copy proposals, which "
                                            "--ngram-draft-tokens auto leaves off here");
            }
        }
    }
    auto instance = std::make_unique<ModelInstance>(std::move(model), options);
    frontend.complete();
    std::optional<StartupPhaseScope> planning;
    planning.emplace(options.startup_observer, StartupPhase::TargetFinalize);
    // --draft-tokens auto where its policy does not run: the configuration's fastest fixed length.
    if (const std::uint32_t fixed = models::qwen3_5::resolve_auto_draft_length(instance->parameters, options)) {
        publish_diagnostic(options.diagnostic_observer, DiagnosticLevel::Info,
                           "--draft-tokens auto: %s is not selected for this artifact (or fast paths are "
                           "disabled), so every round drafts %u, its fastest fixed length",
                           options.speculative.backend == SpeculativeBackend::Mtp ? "adaptive-mtp" : "adaptive-dflash",
                           fixed);
    }
    const auto signature = models::qwen3_5::prefill_signature(*instance->model);
    auto context_cost    = resolve_context_machine_cost(
        {.hardware_class =
                context_cost_hardware_class(device.props.name, device.props.major, device.props.minor),
            .prefill_signature = signature},
        options.context_cost.preset_path);
    const std::size_t base_headroom = options.vram_headroom_bytes.value_or(kDefaultKvCapacityHeadroomBytes);
    std::size_t retry_headroom      = 0;
    // Offloaded experts take the VRAM the Program leaves (their frames are sized after its fixed
    // allocations), so an automatic KV capacity is every lane's full context, as on
    // Qwen3.8-Flash-Next, rather than the free VRAM.
    KvCapacityPolicy kv_policy = options.kv_capacity;
    std::uint32_t planned_prefill_chunk = 0, planned_idle_width = 0;
    std::optional<StartupPhaseScope> program;
    for (int attempt = 0;; ++attempt) {
        if (!planning) { planning.emplace(options.startup_observer, StartupPhase::TargetFinalize); }
        auto planner    = models::qwen3_5::make_sequence_planner(instance->parameters, device, options);
        if (expert_ram && options.kv_capacity.mode == KvCapacityMode::Automatic) {
            // Every lane's full context in whole pages (a lane's last page holds its context's tail).
            const std::uint64_t page   = planner.capacity_curve().main_page_tokens;
            const std::uint64_t tokens = static_cast<std::uint64_t>(options.max_concurrency) *
                                         ((static_cast<std::uint64_t>(options.max_context) + page - 1U) / page) * page;
            if (tokens > std::numeric_limits<std::uint32_t>::max()) {
                throw std::invalid_argument("--max-context x --max-concurrency exceeds the KV capacity range");
            }
            kv_policy = KvCapacityPolicy::explicit_capacity(static_cast<std::uint32_t>(tokens));
        }
        const std::size_t free_bytes = current_free_device_bytes();
        const std::size_t headroom   = base_headroom + retry_headroom;
        auto resolution = resolve_kv_capacity(kv_policy, planner.capacity_curve(), free_bytes, headroom);
        // idle-wide-prefill: plan the widest idle width (4096, 2048, then 1024 tokens, as the chunk is
        // rounded) whose larger prefill buffers and workspace keep the KV capacity: an explicit one
        // whole, an automatic one down to every lane's full context. None fitting keeps chunk-wide steps.
        {
            const std::uint64_t page_tokens = planner.capacity_curve().main_page_tokens;
            const std::uint64_t lane_page_groups =
                static_cast<std::uint64_t>(options.max_concurrency) *
                ((static_cast<std::uint64_t>(options.max_context) + page_tokens - 1U) / page_tokens);
            for (const std::uint32_t width :
                 models::qwen3_5::idle_prefill_widths(instance->parameters, options)) {
                auto wide = models::qwen3_5::make_sequence_planner(instance->parameters, device,
                                                                   options, width);
                if (const auto fitted =
                        resolve_kv_capacity_beside(kv_policy, wide.capacity_curve(),
                                                   free_bytes, headroom, resolution, lane_page_groups)) {
                    planner    = std::move(wide);
                    resolution = *fitted;
                    break;
                }
            }
        }
        auto sequence   = std::move(planner).finalize(resolution.main_page_groups);
        if (sequence.device_reservation_bytes() != resolution.runtime_reservation_bytes ||
            sequence.kv_capacity() != resolution.resolved_tokens) {
            throw std::logic_error("resolved KV capacity does not match the finalized Program plan");
        }
        options.context_cache.host_capacity_bytes = sequence.host_capacity_bytes();
        // Automatic tree widths on a target that cannot verify trees resolve to chain verification.
        options.speculative.draft_tree_auto = sequence.draft_tree_auto();
        instance->kv_capacity_resolution    = resolution;
        instance->kv_capacity_resolution.mode = options.kv_capacity.mode; // auto stays reported as auto
        planned_prefill_chunk = sequence.prefill_chunk();
        planned_idle_width    = sequence.idle_prefill_width();
        planning->complete();
        planning.reset();
        program.emplace(options.startup_observer, StartupPhase::ProgramInitialize);
        try {
            instance->program = models::qwen3_5::create_program(instance->parameters, std::move(sequence),
                                                                device, options.startup_observer,
                                                                offload_options ? &*offload_options : nullptr);
        } catch (const GraphMemoryExhausted& error) {
            // The CUDA Graphs needed more device memory than their allowance left (the driver's
            // per-executable cost varies): an automatic KV capacity retries once with
            // kGraphRetryHeadroomBytes more headroom, so fewer KV pages; an explicit one cannot shrink.
            if (attempt > 0 || kv_policy.mode != KvCapacityMode::Automatic) { throw; }
            retry_headroom = kGraphRetryHeadroomBytes;
            program.reset();
            publish_diagnostic(options.diagnostic_observer, DiagnosticLevel::Warning,
                               "%s at startup; retrying with %zu MiB more VRAM headroom (a smaller automatic KV "
                               "capacity)",
                               error.what(), kGraphRetryHeadroomBytes >> 20);
            continue;
        }
        break;
    }
    LoadSummary::PrefixCacheRestore restore;
    if (options.context_cache.enabled) {
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
    program->complete();
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
    {
        const models::qwen3_5::FastPathSelection fast =
            models::qwen3_5::select_fast_paths(*instance->model, options);
        summary.fast_path_configuration = models::qwen3_5::describe(fast.configuration);
        for (const FastPath path : fast.paths) { summary.fast_paths.emplace_back(fast_path_name(path)); }
        summary.fast_paths_disabled = !options.fast_paths;
        // A tail skip depends on what the expert cache holds.
        summary.request_invariant = fast.request_invariant() && options.expert_tail_skip == 0;
        // Decode and verification run the column-invariant parameters unless batched-decode runs.
        if (fast.invariant(FastPath::BatchedDecode)) {
            for (const auto& entry : instance->parameters.column_by_column) {
                summary.column_by_column_projections.push_back(
                    entry.role + " " + std::string(artifact::format_name(entry.qtype)) + " [" +
                    std::to_string(entry.rows) + "," + std::to_string(entry.cols) + "] x" +
                    std::to_string(entry.count));
            }
        }
        summary.prefill_chunk       = planned_prefill_chunk;
        if (fast.selected(FastPath::IdleWidePrefill)) { summary.idle_prefill_width = planned_idle_width; }
    }
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
    return {std::move(instance), std::move(summary), std::move(metadata)};
}

} // namespace infernix::runtime
