#include "runtime/engine/qwen4_exp_instance.h"

#include "artifact/reader.h"
#include "core/device.h"
#include "core/host_memory.h"
#include "core/startup.h"
#include "core/vram_budget.h"
#include "models/load_options.h"
#include "models/qwen4_exp/load.h"
#include "models/qwen4_exp/memory_plan.h"
#include "models/qwen4_exp/program/fast_paths.h"
#include "models/qwen4_exp/program/ngram_volume.h"
#include "models/qwen4_exp/program/route_trace.h"
#include "models/qwen4_exp/program/vram_monitor.h"
#include "models/registry.h"
#include "runtime/engine/model_instance.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>

namespace infernix::runtime {
namespace {

// The pinned Host tier of Qwen3.8-Flash-Next's prefix cache when --host-context-mib is not given.
// Planning allowance for the Program's pinned and mapped buffers (I/O staging, sampling, the CPU
// expert service's mapped records); the measured total is well below it.
constexpr std::uint64_t kProgramPinnedAllowance = 256ULL << 20;
// The materializer's upload slots and direct-read bounce buffer during the load.
constexpr std::uint64_t kLoadStagingBytes = 4ULL * (64ULL << 20) + (8ULL << 20);
// Below this many expert frames most routed experts miss the VRAM cache and decode slows down
// markedly (design §19.2).
constexpr std::uint32_t kFewFrames = 2048;

unsigned long long mib(std::uint64_t bytes) { return static_cast<unsigned long long>(bytes >> 20); }

// The fixed device allocations, largest first in the order a user can act on them.
std::string device_contributors(std::uint64_t dense, const models::qwen4_exp::ProgramDevicePlan& plan) {
    char kv[64];
    if (plan.kv_max > plan.kv) {
        std::snprintf(kv, sizeof(kv), "%llu (grows to %llu from the expert cache)", mib(plan.kv), mib(plan.kv_max));
    } else {
        std::snprintf(kv, sizeof(kv), "%llu", mib(plan.kv));
    }
    char text[320];
    std::snprintf(text, sizeof(text),
                  "dense weights %llu MiB, KV %s, workspace %llu, expert staging %llu, state and io %llu, expert "
                  "tables %llu",
                  mib(dense), kv, mib(plan.workspace), mib(plan.staging), mib(plan.state + plan.io),
                  mib(plan.residency));
    return text;
}

void report(const EngineOptions& options, DiagnosticLevel level, const std::string& text) {
    if (options.diagnostic_observer.callback) {
        try {
            options.diagnostic_observer.callback(Diagnostic{.level = level, .message = text});
        } catch (...) {}
    } else if (level != DiagnosticLevel::Debug) {
        std::fprintf(stderr, "[engine] %s\n", text.c_str());
    }
}

void validate(const EngineOptions& options) {
    if (options.purpose != EnginePurpose::Generation && options.purpose != EnginePurpose::CausalScoring) {
        throw std::invalid_argument("Qwen3.8-Flash-Next serves generation and causal scoring");
    }
    if (options.speculative.backend == SpeculativeBackend::DFlash || options.speculative.ngram_archive_bytes != 0) {
        throw std::invalid_argument("Qwen3.8-Flash-Next speculates with its MTP drafter, a DFlash2 drafter "
                                    "(--spec dflash2 --draft-model FILE) and n-gram copy proposals");
    }
    // Verification widths up to 8 keep its dense routes column-invariant (design 11.3); its rounds
    // verify chains.
    if (options.speculative.backend == SpeculativeBackend::DFlash2 &&
        (options.speculative.draft_tokens == 0 || options.speculative.draft_tokens > 7)) {
        throw std::invalid_argument("Qwen3.8-Flash-Next DFlash2 drafts 1-7 tokens per round (--draft-tokens)");
    }
    if (options.speculative.draft_tree_auto ||
        std::any_of(options.speculative.draft_tree_nodes.begin(), options.speculative.draft_tree_nodes.end(),
                    [](std::uint32_t nodes) { return nodes != 0; })) {
        throw std::invalid_argument("Qwen3.8-Flash-Next verifies chains: --draft-tree-nodes is not supported");
    }
    // 16-bit activations run every linear A16 and every expert bank W4A16 (the exact W4A16
    // arithmetic of design §16.2.1): the quality reference, so causal scoring only, on the exact
    // route.
    if (options.a16_activations && options.purpose != EnginePurpose::CausalScoring) {
        throw std::invalid_argument(
            "Qwen3.8-Flash-Next runs 16-bit activations (the quality reference) for causal scoring only");
    }
    if (options.a16_activations && options.fast_paths) {
        throw std::invalid_argument(
            "16-bit activations (the quality reference) run the exact route: disable fast paths");
    }
    if (options.a16_activations && !options.fast_path_candidates.empty()) {
        throw std::invalid_argument("16-bit activations (the quality reference) run no fast path");
    }
    if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("Engine max_concurrency must be in [1,8]");
    }
    if (options.expert_tail_skip > 9) {
        throw std::invalid_argument("--expert-tail-skip takes a top-10 rank 1-9 (0: off)");
    }
    if (options.expert_tail_skip != 0 && options.purpose != EnginePurpose::Generation) {
        throw std::invalid_argument("--expert-tail-skip applies to generation (its decode rounds) only");
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
                                               .vision_enabled          = options.enable_vision,
                                               .max_context             = options.max_context,
                                               .media_cache_bytes       = options.media_cache_bytes,
                                               .media_live_bytes        = options.media_live_bytes,
                                               .media_preprocess_threads = options.media_preprocess_threads,
                                               .vision_max_merged_tokens = options.vision_max_merged_tokens,
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
    auto plan = models::qwen4_exp::plan_load(reader, models::load_options(options, models::Architecture::Qwen4Exp));
    // The RAM ledger (design §19.3.7): the experts take what the reserve and every other planned
    // allocation leave, before anything is pinned or read.
    models::qwen4_exp::HostMemoryDemand demand;
    demand.reserve      = options.ram_headroom_bytes;
    demand.expert_banks = plan.pinned_expert_bytes();
    demand.other_pinned = plan.pinned_other_bytes();
    demand.later_pinned = kProgramPinnedAllowance;
    // The prefix cache pins its whole Host tier at startup, beside the experts.
    if (options.context_cache.enabled) {
        demand.prefix_cache = options.context_cache.host_capacity_bytes.value();
    }
    // The n-gram table its volume holds: the artifact's, or another volume's (any format) in its place.
    const std::filesystem::path volume = options.ngram_volume_path.empty()
                                             ? models::qwen4_exp::default_ngram_volume(options.artifact_path)
                                             : options.ngram_volume_path;
    const models::qwen4_exp::NgramTableConfig table =
        models::qwen4_exp::resolve_ngram_volume(volume, plan.config().text.ple);
    if (const auto& own = plan.config().text.ple.table;
        table.format != own.format || table.volume_id != own.volume_id) {
        const auto name = [](models::qwen4_exp::NgramRowFormat format) {
            return format == models::qwen4_exp::NgramRowFormat::Bf16    ? std::string("BF16")
                   : format == models::qwen4_exp::NgramRowFormat::Iq4Nl ? std::string("IQ4_NL")
                                                                        : std::string("FP8");
        };
        report(options, DiagnosticLevel::Info,
               "n-gram table: " + name(table.format) + " rows from " + volume.string() + " in place of the artifact's " +
                   name(own.format) + " table");
    }
    demand.pageable     = models::qwen4_exp::NgramVolume::cache_bytes(table);
    demand.load_staging = kLoadStagingBytes;
    demand.expert_cap   = options.expert_ram_bytes;
    // Windows page-locks memory for the GPU only up to the adapter's shared system memory; the
    // experts past that stay in RAM unpinned (the SSD tier's unpinned records).
    auto vram             = open_vram_budget_source(device.device, options.vram_past_budget);
    demand.pin_room       = models::qwen4_exp::pin_room(vram->query());
    demand.expert_pin_cap = options.expert_pin_bytes;
    // The n-gram table in RAM (design §12.5): pinned at Program start, so it is a later pin. Auto keeps it
    // only while every expert stays pinned in RAM with it; Ram keeps it whatever the experts' placement.
    const auto snapshot = host_memory_snapshot();
    bool ngram_in_ram   = false;
    if (options.ngram_residency != NgramResidency::Volume) {
        auto with_table         = demand;
        with_table.later_pinned += models::qwen4_exp::NgramVolume::resident_bytes(table);
        with_table.pageable     = 0;
        const auto trial        = models::qwen4_exp::plan_host_memory(snapshot, with_table);
        if (options.ngram_residency == NgramResidency::Ram ||
            (trial.placement == models::qwen4_exp::ExpertPlacement::Full && !trial.cap_too_large)) {
            demand       = with_table;
            ngram_in_ram = true;
        }
    }
    const auto ledger   = models::qwen4_exp::plan_host_memory(snapshot, demand);
    report(options, DiagnosticLevel::Info,
           std::string("n-gram table: ") +
               (ngram_in_ram ? "in RAM (" + std::to_string(models::qwen4_exp::NgramVolume::resident_bytes(table) >> 20) +
                                   " MiB pinned)"
                             : options.ngram_residency == NgramResidency::Volume
                                   ? "read from its volume"
                                   : "read from its volume (the RAM ledger keeps the experts resident first)"));
    report(options, DiagnosticLevel::Info, ledger.summary());
    report(options, DiagnosticLevel::Debug, ledger.describe());
    if (ledger.cap_too_large) {
        throw std::runtime_error("--expert-ram-mib " + std::to_string(*options.expert_ram_bytes >> 20) +
                                 " asks for more Host RAM than is free or pinnable now (" + ledger.describe() +
                                 "); lower it, use auto, or free Host memory");
    }
    // The SSD tier (design §19.3.7): the banks stay in the artifact; the Program pins RAM slots for
    // the experts the ledger leaves room for (less their lock overhead) and reads the rest on use.
    const bool tier = ledger.placement == models::qwen4_exp::ExpertPlacement::Tier;
    if (tier) {
        auto streamed           = models::load_options(options, models::Architecture::Qwen4Exp);
        streamed.stream_experts = true;
        plan                    = models::qwen4_exp::plan_load(reader, streamed);
    }
    // A small reserve trims other programs' working sets; keep Infernix's own pageable memory
    // resident so its heap and driver pages never hard-fault from the page file.
    if (const auto refused = reserve_process_working_set(demand.process_growth); !refused.empty()) {
        report(options, DiagnosticLevel::Warning,
               "Windows would not guarantee Infernix " + std::to_string(demand.process_growth >> 20) +
                   " MiB of resident working memory (" + refused +
                   "); everything still works, but if RAM runs short Windows may move some of it to the "
                   "page file and responses slow down");
    }
    plan.set_host_reserve(options.ram_headroom_bytes, demand.later_pinned);

    models::qwen4_exp::ProgramOptions program_options;
    program_options.max_context     = options.max_context;
    program_options.max_concurrency = options.max_concurrency;
    program_options.causal_scoring  = options.purpose == EnginePurpose::CausalScoring;
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
    program_options.dflash2_draft_tokens = options.speculative.backend == SpeculativeBackend::DFlash2
                                               ? options.speculative.draft_tokens
                                               : 0U;
    // A fixed draft length verifies every draft unless --spec-min-p asks for the cut.
    program_options.auto_draft_length = options.speculative.draft_tokens_auto;
    // MTP's automatic length cuts at 0.5 by default; DFlash2's calibrated length needs no cut.
    program_options.draft_min_p = options.speculative.draft_min_p.value_or(
        options.speculative.draft_tokens_auto && options.speculative.backend == SpeculativeBackend::Mtp ? 0.5F : 0.0F);
    if (!options.expert_state_path.empty()) {
        // The state belongs to the artifact: its experts and their records, not the build or KV format.
        std::error_code error;
        const auto size = std::filesystem::file_size(options.artifact_path, error);
        const auto time = std::filesystem::last_write_time(options.artifact_path, error);
        program_options.expert_state          = options.expert_state_path;
        program_options.expert_state_identity = "artifact=" + std::filesystem::absolute(options.artifact_path).string() +
                                                ";size=" + std::to_string(error ? 0U : size) +
                                                ";mtime=" + std::to_string(error ? 0 : time.time_since_epoch().count());
    }
    program_options.ngram_volume = volume;
    program_options.ngram_table  = table;
    program_options.ngram_in_ram = ngram_in_ram;
    if (!options.expert_stripe_path.empty()) {
        if (tier) {
            program_options.expert_stripe = options.expert_stripe_path;
        } else {
            report(options, DiagnosticLevel::Info,
                   "--expert-stripe is not used: every expert fits in RAM, so none is read from disk");
        }
    }
    // The hybrid prefix cache (design §19.3.1). Host-born snapshots live in the pinned slab pool;
    // the cost model ranks resume sources by the measured prefill coefficients.
    const ContextCacheOptions& cache = options.context_cache;
    if (cache.enabled) {
        const std::uint32_t chunk         = options.prefill_chunk;
        program_options.prefix_cache      = true;
        // normalize_engine_options resolved every value for this architecture.
        program_options.prefix_host_bytes          = cache.host_capacity_bytes.value();
        program_options.prefix_taps.max_new_taps   = cache.hybrid.max_new_taps.value();
        program_options.prefix_taps.ladder_tokens  = cache.hybrid.tap_ladder_tokens.value();
        program_options.prefix_taps.min_gap_tokens = cache.hybrid.tap_min_gap_tokens.value();
        // Fitted 2026-10-05 (RTX 5090 at PCIe 5.0 x8, INT8 KV; design §19.4): 1.67 s per layer-walk
        // span, 0.16 s per further call, 22 us per token and 1.06 ns per attention pair, within 5 %
        // from 4K to 128K. The span is one pass of the experts over the link: the Program rescales
        // it, and the restore rate, to the link it measures at startup.
        program_options.prefix_cost.chunk_seconds          = 0.1645;
        program_options.prefix_cost.chunk_tokens           = chunk;
        program_options.prefix_cost.token_seconds          = 2.23e-5;
        program_options.prefix_cost.attention_pair_seconds = 1.06e-9;
        program_options.prefix_cost.call_route_fraction    = 10.0 / 512.0;
        program_options.prefix_span_seconds                = 1.67;
        program_options.prefix_cost.h2d_bytes_per_second   = 27.5e9; // the reference link, until measured
        program_options.prefix_cost.transfer_batch_seconds = 20.0e-6;
        program_options.prefix_save                        = cache.hybrid.persistent_save;
        // A request waiting for a sibling's snapshot stays in the FIFO, so the predicted wait is
        // kept well inside its queue timeout.
        program_options.prefix_coalesce_wait_seconds = static_cast<double>(options.pending_timeout_ms) / 1000.0 / 2.0;
    }
    program_options.route_trace   = options.route_trace_path.empty()
                                        ? models::qwen4_exp::testing::route_trace()
                                        : options.route_trace_path;
    program_options.vram_grow_delay_seconds = models::qwen4_exp::testing::vram_grow_delay();
    program_options.vram_headroom = options.vram_headroom_bytes;
    program_options.vram_past_budget = options.vram_past_budget;
    program_options.expert_tail_skip = options.expert_tail_skip;
    if (options.expert_tail_skip != 0) {
        report(options, DiagnosticLevel::Warning,
               "--expert-tail-skip " + std::to_string(options.expert_tail_skip) +
                   ": decode and verification rounds skip a routed expert missing from VRAM that every token "
                   "routes at top-10 rank " + std::to_string(options.expert_tail_skip) +
                   " or lower; output changes (lower quality) for faster decode");
    }
    if (tier) {
        program_options.expert_ram_bytes =
            static_cast<std::uint64_t>(static_cast<double>(ledger.expert_pinned) / (1.0 + demand.lock_overhead));
        program_options.expert_unpinned_bytes = ledger.expert_pageable;
        program_options.later_pinned_bytes    = demand.prefix_cache + demand.later_pinned + models::qwen4_exp::kPinRoomMarginBytes;
    }

    // The VRAM check (design §19.3.7), before the weights are read: the dense weights, the
    // Program's fixed allocations, the reserve for graph executables and the display headroom
    // must leave room for the expert cache. Fast paths are selected once the weights are bound, so
    // the check counts sampled-mtp-drafts' draft graphs whenever it could be selected.
    program_options.sampled_mtp_drafts =
        options.speculative.backend == SpeculativeBackend::Mtp &&
        (options.fast_paths || std::find(options.fast_path_candidates.begin(), options.fast_path_candidates.end(),
                                         FastPath::SampledMtpDrafts) != options.fast_path_candidates.end());
    const auto device_plan =
        models::qwen4_exp::Program::plan_device(program_options, plan.config(), plan.public_token_count());
    const std::uint64_t dense = plan.device_bytes();
    const VramSnapshot before = vram->query();
    models::qwen4_exp::VramDemand vram_demand;
    vram_demand.headroom    = options.vram_headroom_bytes;
    vram_demand.graphs      = device_plan.graph_bound;
    vram_demand.fixed       = dense + device_plan.fixed_bytes();
    vram_demand.frame_bytes = device_plan.expert_frame_bytes; // an average frame of the per-size pools
    vram_demand.max_frames  = device_plan.max_frames;
    const auto check        = models::qwen4_exp::size_expert_frames(before, vram_demand);
    if (check.frame_bytes < 0) {
        throw std::runtime_error(
            "Qwen3.8-Flash-Next does not fit in device memory: " + std::to_string(mib(before.device_free)) +
            " MiB free; " + device_contributors(dense, device_plan) + "; reserve " + std::to_string(mib(check.reserve)) +
            "; headroom " + std::to_string(mib(check.headroom)) + " (" + check.describe() +
            "). Lower --max-context, --max-concurrency or --prefill-chunk, use --kv-dtype int8, lower "
            "--vram-headroom-mib, or free VRAM");
    }
    if (check.frames < kFewFrames) {
        report(options, DiagnosticLevel::Warning,
               "decode will be slow: only " + std::to_string(check.frames) + " expert frames fit in VRAM (" +
                   std::to_string(kFewFrames) + " or more recommended); lower --max-context or --max-concurrency, "
                   "or free VRAM");
    }

    // The dense arena must land in device memory (WDDM can place it in system memory silently).
    SpillGuard guard(*vram);
    guard.begin();
    auto model = models::qwen4_exp::materialize_model(std::move(plan), device, &options.startup_observer);
    device.synchronize();
    if (const std::uint64_t spilled = guard.end(dense); spilled != 0) {
        throw std::runtime_error(std::to_string(mib(spilled)) +
                                 " MiB of the dense weights were placed in shared system memory by the driver's "
                                 "sysmem fallback: free VRAM and start again");
    }
    const std::uint64_t after_weights = vram->query().device_free;

    StartupPhaseScope frontend(options.startup_observer, StartupPhase::FrontendInitialize);
    auto instance = std::make_unique<Qwen4ExpInstance>(std::move(model), options);
    frontend.complete();

    // The fast paths qualified for this configuration (docs/maintainer/model-quality.md section 6.7).
    const models::qwen4_exp::FastPathSelection fast = models::qwen4_exp::select_fast_paths(instance->parameters, options);
    program_options.fast_paths        = fast.forward();
    program_options.pair_speculation  = fast.selected(FastPath::AdaptiveSpeculation);
    program_options.free_cache_resume = fast.selected(FastPath::FreeCacheResume);
    program_options.preemption        = fast.selected(FastPath::Preemption);
    program_options.sampled_mtp_drafts = fast.selected(FastPath::SampledMtpDrafts);

    StartupPhaseScope program(options.startup_observer, StartupPhase::ProgramInitialize);
    instance->program =
        std::make_unique<models::qwen4_exp::Program>(instance->parameters, device, std::move(program_options));
    const auto& sizing = instance->program->vram_sizing();
    const std::uint64_t frames_bytes =
        static_cast<std::uint64_t>(sizing.frames) * device_plan.expert_frame_bytes;
    const std::uint64_t after_startup = vram->query().device_free;
    {
        const char* display = before.display == DisplayState::Headless   ? "no display"
                              : before.display == DisplayState::Attached ? "display attached"
                                                                         : "display unknown";
        // --vram-past-budget: the allowance already counted as free.
        const std::string past = before.budget_allowance == 0
                                     ? std::string()
                                     : " (" + std::to_string(mib(before.budget_allowance)) + " MiB past the OS budget)";
        char line[704];
        std::snprintf(line, sizeof(line),
                      "VRAM ledger: %llu MiB card, %s, %llu in use before loading%s; %s; reserve %llu (%u graphs); "
                      "headroom %llu (%s); expert frames %u (%.2f GiB); %llu free after startup",
                      mib(before.device_total), display, mib(before.device_total - before.device_free), past.c_str(),
                      device_contributors(dense, device_plan).c_str(), mib(sizing.reserve), device_plan.graph_bound,
                      mib(sizing.headroom), options.vram_headroom_bytes ? "set" : "auto", sizing.frames,
                      static_cast<double>(frames_bytes) / static_cast<double>(1ULL << 30), mib(after_startup));
        report(options, DiagnosticLevel::Debug, line);
        const double gib = static_cast<double>(1ULL << 30);
        char summary[320];
        if (device_plan.kv_max > device_plan.kv) {
            std::snprintf(summary, sizeof(summary),
                          "VRAM: model %.2f GiB, KV cache %.2f GiB (grows to %.2f GiB as contexts lengthen, taking room "
                          "from the expert cache), expert cache %u experts (%.1f GiB)",
                          static_cast<double>(dense) / gib, static_cast<double>(device_plan.kv) / gib,
                          static_cast<double>(device_plan.kv_max) / gib, sizing.frames,
                          static_cast<double>(frames_bytes) / gib);
        } else {
            std::snprintf(summary, sizeof(summary), "VRAM: model %.2f GiB, KV cache %.2f GiB, expert cache %u experts (%.1f GiB)",
                          static_cast<double>(dense) / gib, static_cast<double>(device_plan.kv) / gib, sizing.frames,
                          static_cast<double>(frames_bytes) / gib);
        }
        report(options, DiagnosticLevel::Info, summary);
    }
    const MemorySummary memory = instance->program->memory_summary();
    auto& resolution                         = instance->kv_capacity_resolution;
    resolution.mode                          = options.kv_capacity.mode;
    resolution.main_page_groups              = memory.kv_capacity_page_groups;
    resolution.maximum_main_page_groups      = memory.kv_capacity_max_page_groups;
    resolution.resolved_tokens               = memory.kv_capacity;
    resolution.runtime_reservation_bytes     = device_plan.fixed_bytes() + frames_bytes;
    resolution.available_after_weights_bytes = after_weights;
    resolution.available_after_startup_bytes = after_startup;
    resolution.automatic_headroom_bytes      = sizing.headroom;
    resolution.planned_slack_bytes =
        after_weights > resolution.runtime_reservation_bytes ? after_weights - resolution.runtime_reservation_bytes : 0;
    LoadSummary::PrefixCacheRestore restore;
    if (const std::filesystem::path& file = options.context_cache.hybrid.persistent_file;
        options.context_cache.enabled && !file.empty()) {
        // Saved states depend on the n-gram rows: another volume's differ from the artifact's table.
        std::string signature = "qwen4_exp;ngram=" + std::to_string(static_cast<int>(table.format)) + ":";
        for (const std::uint8_t byte : table.volume_id) {
            constexpr char kHex[] = "0123456789abcdef";
            signature += kHex[byte >> 4U];
            signature += kHex[byte & 15U];
        }
        if (table.fp8_scale) { signature += ";scale=" + std::to_string(*table.fp8_scale); }
        // A DFlash2 drafter's ring rides in every snapshot: it belongs to that companion artifact.
        if (const auto& draft = instance->model->info().draft_artifact_id) {
            signature += ";draft=";
            for (const std::byte byte : *draft) {
                constexpr char kHex[] = "0123456789abcdef";
                signature += kHex[std::to_integer<unsigned>(byte) >> 4U];
                signature += kHex[std::to_integer<unsigned>(byte) & 15U];
            }
        }
        const models::qwen4_exp::PrefixCachePersistence loaded = instance->program->attach_prefix_cache_file(
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
    summary.prefix_cache         = std::move(restore);
    summary.fast_path_configuration = models::qwen4_exp::describe(fast.configuration);
    for (const FastPath path : fast.paths) { summary.fast_paths.emplace_back(fast_path_name(path)); }
    summary.fast_paths_disabled        = !options.fast_paths;
    summary.request_invariance_capable = true;
    // Tail skip makes the output depend on which experts the cache holds.
    summary.request_invariant          = fast.request_invariant() && options.expert_tail_skip == 0;

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

} // namespace infernix::runtime
