#include "runtime/engine/model_instance.h"
#include "artifact/ablate_sidecar.h"
#include "artifact/reader.h"
#include "artifact/formats.h"
#include "artifact/materializer.h"
#include "core/decode_graph.h"
#include "core/evictable_weight_pool.h"
#include "core/host_memory.h"
#include "core/startup.h"
#include "core/temporary_switches.h"
#include "core/vmm_arena.h"
#include "core/vram_budget.h"
#include "models/qwen3_5/load.h"
#include "models/qwen3_5/measurement.h"
#include "models/qwen3_5/program/offload.h"
#include "models/qwen3_5/program/planning/fast_paths.h"
#include "models/qwen3_5/program/program.h"
#include "runtime/engine/diagnostics.h"
#include "runtime/engine/engine_plan.h"
#include "runtime/engine/expert_memory.h"
#include "infernix/ops/kernel_images.h"
#include "runtime/expert_offload/drafter_pool.h"
#include "runtime/expert_offload/expert_residency.h"
#include "runtime/expert_offload/memory_plan.h"
#include "runtime/expert_offload/route_trace.h"
#include "runtime/expert_offload/vram_monitor.h"
#include "runtime/prefix_cache/persist_file.h"

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
#include <vector>

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
    if (!options.lora_path.empty()) {
        if (architecture == models::Architecture::Qwen4Exp) {
            throw std::invalid_argument("LoRA adapter " + options.lora_path.string() +
                                        " is refused: Qwen3.8-Flash-Next is a mixture-of-experts model, and Infernix "
                                        "applies LoRA adapters to the dense Qwen3.5-family models (the 27B) and "
                                        "Inkling 2 only");
        }
        if (offloaded_experts) {
            throw std::invalid_argument("LoRA adapter " + options.lora_path.string() +
                                        " is refused: this artifact is a mixture-of-experts model (its experts are "
                                        "offloaded), and Infernix applies LoRA adapters to the dense Qwen3.5-family "
                                        "models (the 27B) and Inkling 2 only");
        }
    }
    // Qwen3.8-Flash-Next's Program reads no use_cuda_graph: its rounds replay CUDA graphs on every route but the lossless
    // one, which runs them eagerly already; a scoring Program runs no rounds. Refused rather than silently ignored.
    if (architecture == models::Architecture::Qwen4Exp && !options.use_cuda_graph &&
        options.purpose == EnginePurpose::Generation && !options.lossless) {
        throw std::invalid_argument("--no-cuda-graph: Qwen3.8-Flash-Next runs its decode rounds as CUDA graphs (its "
                                    "--lossless route runs them eagerly)");
    }
    // The lossless route computes the model's own forward correctly rounded; a cut is not part of it. Refused here,
    // before anything loads.
    if (!options.ablate_sidecar_path.empty() && options.lossless) {
        throw std::invalid_argument("--ablate does not combine with --lossless (docs/ablate.md)");
    }
    // An automatic context takes all the VRAM the KV pool may have: offloaded experts size their cache from what
    // the fixed allocations leave, so it would starve the expert cache.
    if (options.max_context_auto) {
        if (architecture != models::Architecture::Qwen3_5 || offloaded_experts) {
            throw std::invalid_argument(
                "--max-context auto sizes one request to the whole KV cache, which offloaded experts share the VRAM "
                "with (their expert cache would shrink to nothing): give --max-context N (infernix-serve --plan "
                "reports the largest that fits)");
        }
        if (options.purpose != EnginePurpose::Generation) {
            throw std::invalid_argument("--max-context auto sizes a generation server: causal scoring takes N");
        }
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
        // Added to a steady_clock time at every save (int64 nanoseconds): past a year it would overflow there.
        if (interval.count() < 0 || interval > std::chrono::hours(24 * 366)) {
            throw std::invalid_argument("prefix cache save interval must be in [0, one year]");
        }
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
                             (options.expert_a4_activations && !options.reference) || options.a8_activations)) {
        throw std::invalid_argument("the lossless route runs no fast path: disable fast paths");
    }
    if (!options.fast_path_exclusions.empty() && (!options.fast_paths || options.lossless)) {
        throw std::invalid_argument("--no-fast-path leaves a path out of the default route's selection: the "
                                    "reproducible and lossless routes select none");
    }
    if (options.reference && (!options.lossless || options.purpose != EnginePurpose::CausalScoring)) {
        throw std::invalid_argument("the design reference (--reference) is the lossless route's causal scoring");
    }
    if (options.lossless_noise_seed != 0 && (!options.lossless || options.purpose != EnginePurpose::CausalScoring)) {
        throw std::invalid_argument("the noise twin (lossless_noise_seed) is the lossless route's causal scoring");
    }
    if (!(options.lossless_noise_scale > 0.0F && options.lossless_noise_scale <= 1.0F)) {
        throw std::invalid_argument("lossless_noise_scale must be in (0, 1]");
    }
}

namespace {

// NVIDIA's dense NVFP4 Qwen3.5 exports (the 27B's W4A4 and FP8 layers): no MoE layer, at least one NVFP4 text
// projection permitted 4-bit activations, and every text projection permitted 4- or 8-bit ones (the head aside)
// bound to NVFP4 or row-scaled FP8 parents. `--a8-activations` then binds those Uses A8Only
// (LoadOptions::a8_dense), 8-bit activations at every width; a MoE artifact keeps the flag's expert routes.
bool dense_a8_artifact(const artifact::Directory& directory) {
    for (const auto& [name, binding] : directory.bindings) {
        if (name.starts_with("text/") && name.find("/moe/") != std::string::npos) { return false; }
    }
    bool w4a4 = false;
    for (const auto& [key, use] : directory.uses) {
        if (!use.parameter.starts_with("text/") || use.parameter == "text/output_head" || !use.activation_policy ||
            *use.activation_policy == artifact::ActivationPolicy::A16Only) {
            continue;
        }
        const auto binding = directory.bindings.find(use.parameter);
        if (binding == directory.bindings.end() || binding->second.parts.empty()) { return false; }
        for (const auto& part : binding->second.parts) {
            const auto* tensor = std::get_if<artifact::TensorObject>(&directory.objects.at(part.object.index));
            const bool nvfp4 = tensor != nullptr && (tensor->format == "nvfp4" || tensor->format == "nvfp4_mul");
            if (!nvfp4 && (tensor == nullptr || (tensor->format != "fp8_e4m3fn_row_bf16" &&
                                                 tensor->format != "fp8_e4m3fn_row_f32"))) {
                return false;
            }
            w4a4 = w4a4 || (nvfp4 && *use.activation_policy == artifact::ActivationPolicy::AllowA4);
        }
    }
    return w4a4;
}

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
// prefill chunk, the options that change the numbers cached state holds (fast paths, routes, tail skip) and the product
// binary's build identity. Any difference makes the saved state meaningless, so the file is ignored.
std::string hybrid_cache_fingerprint(const EngineOptions& options, const std::string& signature,
                                     const cudaDeviceProp& device) {
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
    // The 8-bit activation routes change the KV and state their calls write (decode, short and prefill calls).
    if (options.a8_activations) { out += ";a8"; }
    const auto stamp = [](const std::filesystem::path& file) {
        std::error_code e;
        const auto bytes = std::filesystem::file_size(file, e);
        const auto when  = std::filesystem::last_write_time(file, e);
        return std::filesystem::absolute(file).string() + ":" + std::to_string(e ? 0U : bytes) + ":" +
               std::to_string(e ? 0 : when.time_since_epoch().count());
    };
    // A LoRA adapter changes every adapted projection: its weights file and configuration name it as the artifact's
    // path, size and time do.
    if (!options.lora_path.empty()) {
        const std::filesystem::path weights = std::filesystem::is_directory(options.lora_path, error)
                                                  ? options.lora_path / "adapter_model.safetensors"
                                                  : options.lora_path;
        const std::filesystem::path config = weights.parent_path() / "adapter_config.json";
        out += ";lora=" + stamp(weights) + "|" + stamp(config);
    }
    // An ablation sidecar changes the residual writes of every layer it cuts: the sidecar and its direction file name
    // it the same way.
    if (!options.ablate_sidecar_path.empty()) {
        out += ";ablate=" + stamp(options.ablate_sidecar_path) + "|" +
               stamp(artifact::AblateSidecar::direction_path(options.ablate_sidecar_path));
    }
    // Tail skip leaves experts out of decode and verification rounds, so the KV and state those rounds wrote
    // (endpoints, continued turns) differ from what a run without it computes, and the reverse.
    if (options.expert_tail_skip != 0) { out += ";tail_skip=" + std::to_string(options.expert_tail_skip); }
    for (const FastPath path : options.fast_path_candidates) { out += std::string(",") + fast_path_name(path); }
    for (const FastPath path : options.fast_path_exclusions) { out += std::string(",-") + fast_path_name(path); }
    out += ";build=" + options.context_cache.hybrid.persistent_identity;
    // The reproducible route's bits hold for one GPU: split-K and wave-filling shares follow the SM count.
    out += ";device=" + std::string(device.name) + ";cc=" + std::to_string(device.major) + "." +
           std::to_string(device.minor) + ";sm=" + std::to_string(device.multiProcessorCount);
    return out;
}

namespace {

std::size_t current_free_device_bytes(int device, bool past_budget) {
    return free_device_bytes(open_vram_budget_source(device, past_budget)->query());
}

// What stepped growth maps on `device` now, up to `max_bytes`: 64 MiB VMM chunks as the expert frames grow, each
// mapped only while growth_room covers it and kept only while the spill guard finds it resident (SpillGuard::end),
// all released after. Under
// WDDM the OS budget and free memory need not see another process's VRAM, so a resident model's automatic KV sized
// from them was one allocation that made Windows evict the other process: the 27B NVFP4 at --max-context auto beside
// a 3 GiB holder resolved 370,304 tokens as on an empty card and the holder's VRAM fell from 3,504 to 416 MiB (RTX
// 5090, 2026-11-17, issue #68), while the frames' stepped growth stopped where Windows lowered the budget and the
// holder kept its memory.
struct GrowthProbe {
    std::uint64_t bytes = 0;     // what it mapped (max_bytes when nothing stopped it)
    bool stopped        = false; // growth_room, a refused chunk or a spill stopped it short of max_bytes
    double milliseconds = 0.0;
    std::string spill_readings; // a spill stopped it: the readings that found it (SpillGuard::readings)
};

GrowthProbe probe_growth(int device, VramBudgetSource& source, const expert_offload::GrowthLimit& limit,
                         std::uint64_t max_bytes) {
    GrowthProbe out;
    const auto start        = std::chrono::steady_clock::now();
    const std::size_t chunk = expert_offload::ExpertResidency::kChunkBytes;
    const auto chunk_signed = static_cast<std::int64_t>(chunk);
    VmmArena arena(device, static_cast<std::size_t>(max_bytes), chunk);
    SpillGuard guard(source);
    while (arena.mapped_bytes() + chunk <= max_bytes) {
        guard.begin();
        if (expert_offload::growth_room(guard.before(), limit) < chunk_signed || !arena.map_chunk()) {
            out.stopped = true;
            break;
        }
        CUDA_CHECK(cudaMemset(static_cast<std::byte*>(arena.base()) + arena.mapped_bytes() - chunk, 0, chunk));
        CUDA_CHECK(cudaDeviceSynchronize());
        if (guard.end(chunk) != 0) {
            out.spill_readings = guard.readings(chunk);
            arena.unmap_chunk();
            out.stopped = true;
            break;
        }
    }
    out.bytes = out.stopped ? arena.mapped_bytes() : max_bytes;
    while (arena.mapped_bytes() != 0) { arena.unmap_chunk(); }
    out.milliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    return out;
}

// Offloaded experts: an estimate of the expert frames beside `sequence`. A start sizes them from a live
// reading once the Program's allocations exist (ExpertOffload::size_frames); here they take what the
// Program's reservation less its CUDA Graph allowance (with the elastic text KV region whole, as when
// every lane's context is full, and the engine's staging and tables), the MTP drafter's expert pool and
// `not_uploaded` bytes the snapshot still counts as free (Engine::plan's weights) leave of `snapshot`,
// less the pools' chunk slack the start keeps off them.
struct OffloadedFrames {
    std::uint32_t frames     = 0;
    std::uint64_t frame_bytes = 0, headroom = 0;
    std::uint64_t beside     = 0; // the drafter's expert pool
    std::vector<std::uint64_t> strides;
    std::uint32_t experts = 0;
};

OffloadedFrames estimate_offloaded_frames(const models::qwen3_5::Model& model, const models::qwen3_5::SequencePlan& sequence,
                                          std::uint64_t reservation, std::uint64_t not_uploaded, const VramSnapshot& snapshot,
                                          const expert_offload::OffloadOptions& offload, DeviceContext& device) {
    OffloadedFrames out;
    const auto& moe = std::get<models::qwen3_5::MoeConfig>(model.config().text.ffn);
    for (const auto& bank : model.expert_banks()) { out.strides.push_back(bank.planes.record_stride); }
    out.experts = moe.num_experts;
    const expert_offload::ExpertLayout layout{static_cast<std::uint32_t>(out.strides.size()), moe.num_experts,
                                              moe.num_experts_per_tok};
    // Beside the reservation (which holds the engine's staging and tables, SequencePlan's offload_fixed_bytes):
    // the MTP drafter's expert pool, allocated before the engine sizes its frames (ProgramOffload::start).
    if (const std::uint32_t slots = model.config().mtp_expert_slots; slots != 0) {
        out.beside = expert_offload::DrafterExpertPool::device_bytes(model.config().mtp_expert_bytes,
                                                                     static_cast<std::int32_t>(slots),
                                                                     static_cast<std::int32_t>(moe.num_experts));
    }
    expert_offload::VramDemand demand;
    demand.headroom = offload.vram_headroom;
    demand.elastic  = offload.vram_monitor && VmmArena::supported(device.device);
    // The engine's graph bound (ProgramImpl's constructor).
    demand.graphs = static_cast<std::uint32_t>(
        (sequence.graph_allowance_bytes() + expert_offload::kGraphExecutableBytes - 1) /
        expert_offload::kGraphExecutableBytes);
    demand.frame_bytes = expert_offload::ExpertResidency::average_frame_bytes(out.strides);
    demand.max_frames  = expert_offload::ExpertResidency::max_frames(layout);
    demand.fixed       = not_uploaded + (reservation - sequence.graph_allowance_bytes()) + out.beside;
    const expert_offload::VramSizing sizing = expert_offload::size_expert_frames(snapshot, demand);
    const std::uint32_t slack = expert_offload::pool_slack_frames(
        expert_offload::frame_pools(out.strides), expert_offload::ExpertResidency::kChunkBytes, demand.frame_bytes);
    out.frames      = sizing.frames > slack ? sizing.frames - slack : 0U;
    out.frame_bytes = demand.frame_bytes;
    out.headroom    = sizing.headroom;
    return out;
}

// Engine::plan, Qwen3.5 family: the finalized sequence plan's device memory beside the weights (whose
// bytes plan.vram already holds), the Host tier it settled, and with offloaded experts an estimate of the
// expert frames (estimate_offloaded_frames).
void plan_qwen3_5_program(EnginePlan& plan, const EngineOptions& options, const ModelInstance& instance,
                          const models::qwen3_5::SequencePlan& sequence, const KvCapacityResolution& resolution,
                          std::size_t free_after_weights, const expert_offload::OffloadOptions* offload,
                          DeviceContext& device) {
    EnginePlanVram& vram       = plan.vram;
    vram.kv_bytes              = sequence.kv_payload_bytes();
    vram.kv_tokens             = resolution.resolved_tokens;
    vram.workspace_bytes       = sequence.workspace_capacity_bytes();
    vram.graph_allowance_bytes = sequence.graph_allowance_bytes();
    const std::uint64_t reservation = resolution.runtime_reservation_bytes;
    vram.planned_bytes              = vram.weights_bytes + planned_beside_weights_bytes(vram) + reservation;
    std::uint64_t used              = reservation + vram.headroom_bytes;
    if (options.context_cache.enabled) {
        plan.ram.host_context_bytes = sequence.host_capacity_bytes();
        if (!plan.ram.tier) {
            plan_ram_totals(plan, plan.ram.pinned_bytes.value_or(0) + sequence.host_capacity_bytes());
        }
    }
    if (offload != nullptr) {
        // Not yet on the device: the weights and what a start allocates beside them before its frames. On the 122B
        // that is an --ablate sidecar's direction (12 KiB shared, 576 KiB per layer: under one frame); it takes no
        // --lora, and its DFlash2 head (hidden size 3072) has no draft screen.
        const OffloadedFrames estimate =
            estimate_offloaded_frames(*instance.model, sequence, reservation,
                                      vram.weights_bytes + planned_beside_weights_bytes(vram),
                                      open_vram_budget_source(device.device, options.vram_past_budget)->query(),
                                      *offload, device);
        vram.expert_frames      = estimate.frames;
        vram.expert_cache_bytes = static_cast<std::uint64_t>(estimate.frames) * estimate.frame_bytes;
        vram.headroom_bytes     = estimate.headroom;
        vram.planned_bytes += estimate.beside + *vram.expert_cache_bytes;
        used = reservation + estimate.beside + *vram.expert_cache_bytes + estimate.headroom;
    }
    vram.left_bytes = free_after_weights > used ? free_after_weights - used : 0;
}

// Offloaded experts (candidates RF-04 (c)): whether the expert cache, every lane's context full, lends
// `sequence`'s widest prefill pass its workspace (ProgramOffload::wide_workspace) with a frame to spare,
// with the frames estimated as Engine::plan estimates them (`now`); on the reproducible route also at a
// nominal launch (expert_offload::nominal_snapshot), so its chunk depends only on the GPU, the options and
// the artifact unless this launch's own cache is the smaller. A pass the cache cannot lend allocates its
// workspace per prompt, which fails on a full card after admission.
struct PrefillLend {
    bool now = true, nominal = true;
    [[nodiscard]] bool lent() const noexcept { return now && nominal; }
};

PrefillLend offloaded_prefill_lent(const models::qwen3_5::Model& model, const models::qwen3_5::SequencePlan& sequence,
                                   std::uint64_t reservation, std::uint64_t not_uploaded, const VramSnapshot& snapshot,
                                   std::optional<std::uint64_t> in_use_before_load, bool reproducible,
                                   const expert_offload::OffloadOptions& offload, DeviceContext& device) {
    PrefillLend out;
    const std::uint64_t pass = sequence.offload_wide_pass_bytes();
    if (pass == 0 || !offload.expert_cache) { return out; }
    const auto lends = [&](const VramSnapshot& at) {
        const OffloadedFrames estimate = estimate_offloaded_frames(model, sequence, reservation, not_uploaded, at, offload, device);
        return expert_offload::lend_capacity(estimate.strides, estimate.experts, estimate.frames).lends(pass);
    };
    out.now = lends(snapshot);
    if (reproducible) { out.nominal = lends(expert_offload::nominal_snapshot(snapshot, in_use_before_load)); }
    return out;
}

// Engine::plan, Qwen3.5 family: the largest --max-context whose smallest reservation resolves beside the
// weights as the start resolves it (the base sequence plan, without idle-wide-prefill, with the offloaded
// experts' engine options when `offload` is set), the other options as given. An explicit KV capacity
// equal to --max-context (infernix-serve's default without the prefix cache) follows the context; any
// other stays fixed. Bounded by the model's positions (with YaRN).
void plan_qwen3_5_context(EnginePlan& plan, const EngineOptions& options,
                          const models::qwen3_5::execution::Parameters& parameters, DeviceContext& device,
                          const models::qwen3_5::detail::ProgramOffloadOptions* offload,
                          std::size_t free_after_weights, std::size_t headroom) {
    const double positions = static_cast<double>(parameters.model.config().text.max_position_embeddings) *
                             std::max(1.0, static_cast<double>(options.rope_yarn_factor));
    const auto upper       = static_cast<std::uint32_t>(
        std::min(positions, static_cast<double>(std::numeric_limits<std::int32_t>::max())));
    // An automatic context's starting point is no capacity to follow.
    const bool kv_follows = !options.max_context_auto && options.kv_capacity.mode == KvCapacityMode::Explicit &&
                            options.kv_capacity.explicit_tokens == options.max_context;
    plan.max_context = largest_fitting_context(upper, options.max_context, [&](std::uint32_t context) {
        EngineOptions trial = options;
        trial.max_context   = context;
        if (kv_follows) { trial.kv_capacity = KvCapacityPolicy::explicit_capacity(context); }
        auto planner            = models::qwen3_5::make_sequence_planner(parameters, device, trial, offload);
        KvCapacityPolicy policy = trial.kv_capacity;
        if (offload != nullptr && policy.mode == KvCapacityMode::Automatic) {
            const std::uint64_t page   = planner.capacity_curve().main_page_tokens;
            const std::uint64_t tokens = static_cast<std::uint64_t>(trial.max_concurrency) *
                                         ((static_cast<std::uint64_t>(context) + page - 1U) / page) * page;
            if (tokens > std::numeric_limits<std::uint32_t>::max()) { return false; }
            policy = KvCapacityPolicy::explicit_capacity(static_cast<std::uint32_t>(tokens));
        }
        (void)resolve_kv_capacity(policy, planner.capacity_curve(), free_after_weights, headroom);
        return true;
    });
}

} // namespace

bool artifact_offloads_experts(const artifact::Reader& reader) {
    return reader.directory().bindings.contains("text/layers/0/moe/experts");
}

models::qwen3_5::BundledDrafters artifact_drafters(const artifact::Reader& reader) {
    const auto& components = reader.directory().components;
    return {.mtp     = components.contains("mtp"),
            .dflash  = components.contains("dflash"),
            .dflash2 = components.contains("dflash2")};
}

void resolve_auto_speculative_backend(EngineOptions& options, models::Architecture architecture,
                                      bool offloaded_experts, const models::qwen3_5::BundledDrafters& drafters,
                                      const std::vector<std::filesystem::path>& dflash2_companions) {
    SpeculativeOptions& speculative = options.speculative;
    if (!speculative.backend_auto || options.purpose != EnginePurpose::Generation) { return; }
    // The choice follows each drafter's --draft-tokens auto measurement; a fixed length or a cut tuned
    // for one drafter would not mean the same for another.
    if (!speculative.draft_tokens_auto) {
        throw std::invalid_argument("--spec auto chooses the drafter: it takes --draft-tokens auto");
    }
    if (speculative.draft_min_p) {
        throw std::invalid_argument("--spec-min-p tunes one drafter's cut: --spec auto takes none");
    }
    std::string why;
    if (options.lossless) {
        // No speculation is lossless (docs/maintainer/lossless-route.md).
        speculative.backend = SpeculativeBackend::None;
        speculative.draft_model.clear();
        why = "the lossless route runs no speculation";
    } else if (architecture == models::Architecture::Qwen4Exp) {
        // Inco AI's DFlash2 drafter (a companion artifact) decoded 8-26 % faster than the MTP layer on every
        // conversion measured (README "Qwen3.8-Flash-Next: DFlash2 against MTP", 2026-10-20): the --draft-model
        // drafter, else one companion found alone beside the artifact, else MTP where the artifact bundles it.
        if (!speculative.draft_model.empty()) {
            speculative.backend = SpeculativeBackend::DFlash2;
            why                 = "the --draft-model drafter, Qwen3.8-Flash-Next's fastest measured";
        } else if (dflash2_companions.size() == 1) {
            speculative.backend     = SpeculativeBackend::DFlash2;
            speculative.draft_model = dflash2_companions.front();
            why = "the DFlash2 drafter beside the artifact, " + speculative.draft_model.filename().string() +
                  ", Qwen3.8-Flash-Next's fastest measured";
        } else {
            speculative.backend = drafters.mtp ? SpeculativeBackend::Mtp : SpeculativeBackend::None;
            if (dflash2_companions.empty()) {
                why = "the DFlash2 drafter, about 10 % faster on Qwen3.8-Flash-Next, needs --draft-model FILE or its "
                      "companion artifact in the model's folder";
            } else {
                why = "several DFlash2 drafters lie beside the artifact (";
                for (std::size_t i = 0; i < dflash2_companions.size(); ++i) {
                    why += (i > 0 ? ", " : "") + dflash2_companions[i].filename().string();
                }
                why += "): name one with --draft-model";
            }
        }
    } else {
        speculative.backend = models::qwen3_5::auto_speculative_backend(offloaded_experts, drafters);
        why                 = "the fastest measured of the drafters the artifact bundles";
    }
    speculative.backend_auto = false;
    const char* chosen       = "mtp";
    switch (speculative.backend) {
    case SpeculativeBackend::Mtp: break;
    case SpeculativeBackend::DFlash: chosen = "dflash"; break;
    case SpeculativeBackend::DFlash2: chosen = "dflash2"; break;
    case SpeculativeBackend::DSpark: chosen = "dspark"; break;
    case SpeculativeBackend::DSpine: chosen = "dspine"; break; // never chosen: its artifact is named
    case SpeculativeBackend::None:
        // Nothing to draft with: plain decode, and every drafter setting goes with the drafter (copy
        // proposals stay where the model verifies them alone).
        speculative.draft_tokens            = 0;
        speculative.draft_tokens_auto       = false;
        speculative.proposal_head           = ProposalHead::Full;
        speculative.draft_tree_auto         = false;
        speculative.draft_tree_nodes        = {};
        speculative.ngram_draft_tokens_auto = false;
        chosen                              = "plain decode";
        if (options.lossless) { break; }
        if (architecture == models::Architecture::Qwen4Exp) {
            why = "the artifact bundles no MTP drafter, and " + why;
        } else {
            why = (drafters.mtp || drafters.dflash || drafters.dflash2) ? "none of the artifact's drafters beats it here"
                                                                         : "the artifact bundles no drafter";
        }
        break;
    }
    publish_diagnostic(options.diagnostic_observer, DiagnosticLevel::Info, "--spec auto: %s (%s)", chosen, why.c_str());
}

EngineOptions normalize_engine_options(EngineOptions options, models::Architecture architecture,
                                       bool offloaded_experts) {
    switch (options.purpose) {
    case EnginePurpose::Generation:
        break;
    case EnginePurpose::CausalScoring: {
        // Scoring runs one teacher-forced call at a time, without speculation, vision, CUDA Graphs or the prefix
        // cache. A server started for scoring with a generation configuration (the launcher's defaults name lanes
        // and a drafter) says which of its settings that leaves unused rather than refusing them.
        std::string ignored;
        const auto ignore = [&ignored](const std::string& what) { ignored += (ignored.empty() ? "" : ", ") + what; };
        if (options.max_concurrency > 1) { ignore("--max-concurrency " + std::to_string(options.max_concurrency)); }
        const SpeculativeOptions& spec = options.speculative;
        if (spec.backend != SpeculativeBackend::None || spec.backend_auto || !spec.draft_model.empty() ||
            spec.ngram_draft_tokens != 0 || spec.ngram_draft_tokens_auto) {
            ignore("speculative decoding (--spec, --draft-model, --ngram-draft-tokens)");
        }
        if (options.enable_vision) { ignore("vision"); }
        if (!options.context_cache.hybrid.persistent_file.empty()) { ignore("--prefix-cache-file"); }
        if (!ignored.empty()) {
            publish_diagnostic(options.diagnostic_observer, DiagnosticLevel::Info,
                               "causal scoring runs one request at a time without speculation, vision or the prefix "
                               "cache: %s unused",
                               ignored.c_str());
        }
        options.max_concurrency      = 1;
        options.decision_lanes       = 0;
        options.max_pending_requests = 1;
        options.kv_capacity         = KvCapacityPolicy::explicit_capacity(options.max_context);
        options.speculative          = {};
        options.enable_vision        = false;
        options.use_cuda_graph       = false;
        options.context_cache        = ContextCacheOptions{.enabled = false};
        break;
    }
    default:
        throw std::invalid_argument("Engine purpose is invalid");
    }
    if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("Engine max_concurrency must be in [1,8]");
    }
    if (options.decision_lanes >= options.max_concurrency) {
        throw std::invalid_argument(
            "Engine decision_lanes must leave at least one general lane (less than max_concurrency)");
    }
    // The Engine resolves --spec auto from the artifact's drafters first (resolve_auto_speculative_backend);
    // an unresolved one means it could not read them.
    if (options.speculative.backend_auto) {
        throw std::invalid_argument("--spec auto chooses from the artifact's drafters, which could not be read");
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
        models::qwen3_5::auto_drafts_nothing(offloaded_experts, options.speculative.backend) && !forced_adaptive) {
        publish_diagnostic(options.diagnostic_observer, DiagnosticLevel::Info,
                           "--draft-tokens auto: plain decode is faster than any of this drafter's lengths here "
                           "(each verified draft brings its own offloaded-expert misses), so the engine decodes "
                           "without a drafter");
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
    if (options.expert_a16_activations) {
        if (options.expert_a4_activations || options.a8_activations) {
            throw std::invalid_argument("--a16-activations runs with 16-bit activations: it excludes "
                                        "--expert-a4-activations and --a8-activations");
        }
        if (options.lossless) {
            throw std::invalid_argument("the lossless route sets its own activations: it refuses --a16-activations");
        }
    }
    // An approximation whose output depends on the cache state: never on the exact route.
    if (options.expert_tail_skip != 0 && !options.fast_paths) {
        throw std::invalid_argument("--expert-tail-skip is approximate; the reproducible route (--reproducible) refuses it");
    }
    // 8-bit activations change the arithmetic of the decode, verification and prefill calls they cover (each
    // family's A8Routes); the engine refuses an artifact without such a route.
    if (options.a8_activations && !options.fast_paths) {
        throw std::invalid_argument("--a8-activations is approximate; the reproducible route (--reproducible) refuses it");
    }
    // Inkling 2's options are normalized apart (inkling2_instance.cpp): every artifact reaching here refuses it.
    if (options.q8_weights) { throw std::invalid_argument("--q8-weights applies to Inkling 2 artifacts only"); }
    if (options.kv_window_cache) {
        throw std::invalid_argument("--kv-window-dtype applies to Inkling 2 artifacts only (their sliding windows)");
    }
    // An automatic draft length chooses up to the model's most drafts unless given a ceiling: MTP on
    // Qwen3.8-Flash-Next verifies 8 columns column-invariantly, the Qwen3.5 planner captures 5. The
    // block drafters propose their whole block: Qwen3.8-Flash-Next's DFlash2 7; the Qwen3.5 planner
    // resolves its DFlash and DFlash2 blocks per configuration (resolve_auto_draft_length).
    const bool block_drafter = options.speculative.backend == SpeculativeBackend::DFlash ||
                               options.speculative.backend == SpeculativeBackend::DFlash2 ||
                               options.speculative.backend == SpeculativeBackend::DSpine;
    // A DSpine drafter runs beside a Qwen3.5-family model (docs/maintainer/dspine.md).
    if (options.speculative.backend == SpeculativeBackend::DSpine && qwen4_exp) {
        throw std::invalid_argument("--spec dspine applies to Qwen3.5-family artifacts only");
    }
    if (options.speculative.draft_tokens_auto) {
        if (options.speculative.backend != SpeculativeBackend::Mtp && !block_drafter) {
            throw std::invalid_argument("--draft-tokens auto requires --spec mtp, dflash, dflash2 or auto");
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
    // The draft-confidence cut exists on Qwen3.8-Flash-Next's one-row MTP and DFlash2 rounds and on
    // Qwen3.5's one-row MTP rounds; another combination refuses it rather than ignoring a tuning value.
    if (const auto& p = options.speculative.draft_min_p) {
        if (!(*p >= 0.0F && *p <= 1.0F)) {
            throw std::invalid_argument("speculative draft_min_p (--spec-min-p) must be in [0,1]");
        }
        const bool mtp = options.speculative.backend == SpeculativeBackend::Mtp;
        if (qwen4_exp ? !mtp && options.speculative.backend != SpeculativeBackend::DFlash2 : !mtp) {
            throw std::invalid_argument("--spec-min-p applies to --spec mtp (and dflash2 on Qwen3.8-Flash-Next) only");
        }
        // Qwen3.8-Flash-Next MTP with --draft-tokens auto cuts its drafts by their cumulative acceptance instead.
        if (qwen4_exp && mtp && options.speculative.draft_tokens_auto) {
            throw std::invalid_argument("--spec-min-p applies to a fixed --draft-tokens N with Qwen3.8-Flash-Next "
                                        "MTP; --draft-tokens auto cuts by the drafts' cumulative acceptance");
        }
    }
    if (cache.enabled) {
        // One pinned Host slab pool serves blocks and snapshots alike; its size is the only
        // capacity a deployment has to choose (docs/maintainer/hybrid-prefix-cache-spec.md §5.4).
        // Unset (--host-context-mib auto, the default): sized once the model's RAM is planned
        // (construct_model, construct_qwen4_exp; ContextCacheOptions::host_capacity_bytes); an automatic
        // size is always a Host tier (at least 1 GiB).
        const bool host_tier = !cache.host_capacity_bytes || *cache.host_capacity_bytes != 0;
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
        // Idle-time prefill of finished turns (hybrid-prefix-cache-spec §7.9). Qwen3.8-Flash-Next: off by default
        // (its default route already resumes 94 % of agentic turns at the reply's end), and refused on the
        // reproducible route, where a turn resumed from the pass's grid snapshot inside a cold run's call differed
        // from an uncached run (2026-10-27, infernix_qwen4_exp_prefix_cache_real_test --next-turn).
        if (qwen4_exp && !options.fast_paths && hybrid.next_turn_prefill_quiet.value_or(std::chrono::milliseconds{0}).count() != 0) {
            throw std::invalid_argument(
                "Qwen3.8-Flash-Next's idle next-turn prefill is not exact: --reproducible and --lossless refuse it");
        }
        hybrid.next_turn_prefill_quiet = hybrid.next_turn_prefill_quiet.value_or(
            qwen4_exp ? std::chrono::milliseconds{0} : kDefaultNextTurnPrefillQuiet);
        if (hybrid.next_turn_prefill_quiet->count() < 0 ||
            *hybrid.next_turn_prefill_quiet > std::chrono::milliseconds{3'600'000}) {
            throw std::invalid_argument("idle prefill quiet period must be in [0,3600000] ms");
        }
        // One resident snapshot per request lane keeps every live conversation's latest
        // endpoint restorable without PCIe traffic; one more slot stages taps and endpoints while
        // their Host copies are written. Without a Host tier these slots are the only snapshot
        // storage, so one more is kept for shared prefixes. Decision lanes hold no conversation
        // (a decision publishes no endpoint), so they add no slot.
        if (!qwen4_exp) {
            // With offloaded experts and a Host tier one slot (candidates §22 VM-03, 2026-10-27): each further slot
            // is 146.5 MiB of the 122B's frames, and the VRAM defaults measured its one-request decode +1.0-1.7 %
            // with one; its conversations' endpoints restore from the Host tier instead. Resident artifacts take one
            // too since 2026-11-15 (a slot per lane before): the slots' VRAM is KV capacity, and on the 27B at four
            // lanes one slot against five gave 14,272 more tokens and, where two agent conversations did not fit
            // the pool together, continuing turns of 1.9 s mean and 9.2 s worst against 2.5-2.6 s and 16.5 s
            // (hybrid spec §16.13); an image restores from the Host tier in a few milliseconds, pipelined behind
            // the first pass. Without a Host tier the slots are the only snapshot storage: one per general lane
            // and two (decision lanes hold no conversation).
            hybrid.device_snapshot_slots = hybrid.device_snapshot_slots.value_or(
                host_tier ? 1U : concurrency - options.decision_lanes + 2U);
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
        // The SSD tier sits below the Host tier and moves Host slabs (hybrid-prefix-cache-spec §5.7).
        if (cache.ssd_capacity_bytes != 0 || !cache.ssd_directory.empty()) {
            if (cache.ssd_capacity_bytes == 0 || cache.ssd_directory.empty()) {
                throw std::invalid_argument("--ssd-context-gib and --ssd-context-dir go together");
            }
            if (!host_tier) {
                throw std::invalid_argument("the SSD tier needs a Host tier: --host-context-mib must be positive");
            }
            std::error_code ec;
            if (!std::filesystem::is_directory(cache.ssd_directory, ec)) {
                throw std::invalid_argument("--ssd-context-dir " + cache.ssd_directory.string() +
                                            " is not an existing directory");
            }
            cache.ssd_directory = std::filesystem::absolute(cache.ssd_directory, ec);
            if (cache.ssd_write_bytes_per_hour == 0) {
                cache.ssd_write_bytes_per_hour = ContextCacheOptions::kDefaultSsdWriteBytesPerHour;
            }
        }
    } else {
        if (cache.ssd_capacity_bytes != 0 || !cache.ssd_directory.empty()) {
            throw std::invalid_argument("the SSD tier requires the prefix cache (--no-prefix-reuse refuses it)");
        }
        // Without a prefix cache every request starts from the root and nothing is retained:
        // the lanes' StateImages are the only Device state and there is no Host tier.
        if (cache.host_capacity_bytes.value_or(0U) != 0 ||
            cache.hybrid.device_snapshot_slots.value_or(0U) != 0) {
            throw std::invalid_argument(
                "prefix cache capacities require the context cache to be enabled");
        }
        if (cache.hybrid.tool_call_hold.value_or(std::chrono::milliseconds{0}).count() != 0 ||
            cache.hybrid.next_turn_prefill_quiet.value_or(std::chrono::milliseconds{0}).count() != 0 ||
            cache.hybrid.host_state_precision != HybridHostStatePrecision::Fp32) {
            throw std::invalid_argument("prefix cache policies require the context cache to be enabled");
        }
        cache.host_capacity_bytes          = 0U;
        cache.hybrid.device_snapshot_slots = 0U;
        cache.hybrid.tool_call_hold        = std::chrono::milliseconds{0};
        cache.hybrid.next_turn_prefill_quiet    = std::chrono::milliseconds{0};
    }
    return options;
}

namespace {

models::qwen3_5::Frontend instance_frontend(const models::qwen3_5::Model& model, const EngineOptions& options) {
    return models::qwen3_5::make_frontend(
        model.resources(),
        {.chat_template_path       = options.chat_template_path,
         .grammar_cache_bytes      = options.grammar_cache_bytes,
         .architecture             = model.config().text.architecture,
         .vision_enabled           = options.enable_vision,
         .max_context              = options.max_context,
         .media_cache_bytes        = options.media_cache_bytes,
         .media_live_bytes         = options.media_live_bytes,
         .media_preprocess_threads = options.media_preprocess_threads,
         .vision_max_merged_tokens = options.vision_max_merged_tokens,
         .ngram_sources_enabled    = options.speculative.ngram_draft_tokens != 0,
         .ngram_archive_enabled    = options.speculative.ngram_archive_bytes != 0});
}

} // namespace

ModelInstance::ModelInstance(std::unique_ptr<models::qwen3_5::Model> source,
                             const EngineOptions& options)
    : model(std::move(source)), parameters(*model), frontend(instance_frontend(*model, options)),
      capacity(options.max_context) {}

void ModelInstance::resize_context(const EngineOptions& options) {
    frontend = instance_frontend(*model, options);
    capacity = options.max_context;
}

ModelInstance::~ModelInstance() = default;

ConstructedModel construct_model(EngineOptions& options, DeviceContext& device, EnginePlan* planned) {
    // The artifact is opened once; an unreadable one is reported after the option checks.
    const bool plan_only = planned != nullptr;
    const auto start = Clock::now();
    StartupPhaseScope inspect(options.startup_observer, StartupPhase::ArtifactInspect);
    std::optional<artifact::Reader> opened;
    try {
        opened.emplace(options.artifact_path);
    } catch (const std::exception&) {}
    validate_engine_options(options, models::Architecture::Qwen3_5, opened && artifact_offloads_experts(*opened));
    if (!opened) opened.emplace(options.artifact_path);
    artifact::Reader& reader = *opened;
    inspect.complete();
    StartupPhaseScope binding(options.startup_observer, StartupPhase::TargetPlan);
    models::LoadOptions load = models::load_options(options, models::Architecture::Qwen3_5);
    const bool dense_a8      = options.a8_activations && options.fast_paths && !options.lossless &&
                          dense_a8_artifact(reader.directory());
    load.a8_dense = dense_a8;
    auto plan     = models::qwen3_5::plan_load(reader, load);
    // --a16-activations on an artifact whose text projections permit 4- or 8-bit activations (NVIDIA's NVFP4 and FP8
    // 27B layers): they run 16-bit instead. The default follows the checkpoint's specification, so the launch says so.
    if (options.expert_a16_activations) {
        const auto& uses   = reader.directory().uses;
        const bool reduced = std::any_of(uses.begin(), uses.end(), [](const auto& entry) {
            const artifact::Use& use = entry.second;
            return use.parameter.starts_with("text/") && use.activation_policy &&
                   *use.activation_policy != artifact::ActivationPolicy::A16Only;
        });
        if (reduced) {
            report_engine(options, DiagnosticLevel::Warning,
                          "--a16-activations: this artifact's text projections are specified with 4- or 8-bit "
                          "activations at their calibrated scales (NVIDIA's NVFP4 and FP8 layers, as NVIDIA serves "
                          "them); they now run with 16-bit activations everywhere, which changes the output and "
                          "reads prompts more slowly. The default follows the checkpoint's specification and is the "
                          "recommended configuration");
        }
    }
    // Offloaded experts: the VRAM in use before anything is uploaded, for the reproducible route's nominal
    // prefill-chunk fit (offloaded_prefill_lent), as Qwen3.8-Flash-Next takes it.
    std::optional<std::uint64_t> in_use_before_load;
    if (plan.offloaded_experts()) {
        const VramSnapshot before = open_vram_budget_source(device.device, options.vram_past_budget)->query();
        in_use_before_load        = before.device_total - std::min(before.device_total, before.device_free);
    }
    // Engine::plan: the readings a start plans against, before anything is planned, and the weights' arena
    // (which a refusal below still reports).
    if (plan_only) {
        planned->architecture = models::architecture_name(plan.config().text.architecture);
        plan_readings(*planned, device.props.name, open_vram_budget_source(device.device, options.vram_past_budget)->query(),
                      host_memory_snapshot(), options);
        planned->vram.weights_bytes = plan.materialization().device_capacity_bytes;
        planned->vram.planned_bytes = planned->vram.weights_bytes;
    }
    // Offloaded experts (Qwen3.5-122B-A10B design §11.2): the RAM ledger places the routed banks in
    // pinned RAM or in the SSD tier before anything is pinned or read.
    std::optional<ExpertRamPlan> expert_ram;
    bool exclusive_experts = false;
    bool ladder            = false; // the overlay vision window's evictable weight ladder
    std::uint64_t tail_mirror = 0;  // its pinned mirror
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
    // --host-context-mib auto (the size unset): the prefix cache's Host tier from the RAM the model's plan leaves
    // (expert_memory.h), resolved here so the RAM plan, Engine::plan and the Program see one size.
    const bool prefix_auto = options.context_cache.enabled && !options.context_cache.host_capacity_bytes;
    // With --ram-resize auto an automatic tier may grow later, into RAM other programs free.
    options.context_cache.host_growth_capacity_bytes.reset();
    const auto set_growth = [&](const HostMemorySnapshot& snapshot, const expert_offload::HostMemoryDemand& demand) {
        if (!options.ram_resize) { return; }
        const std::uint64_t startup = *options.context_cache.host_capacity_bytes;
        const std::uint64_t growth  = auto_host_context_ceiling_bytes(snapshot, demand, startup);
        if (growth > startup) { options.context_cache.host_growth_capacity_bytes = growth; }
    };
    if (plan.offloaded_experts()) {
        const auto vram   = open_vram_budget_source(device.device, options.vram_past_budget);
        auto demand = expert_ram_demand(
            options, {.expert_banks = plan.pinned_expert_bytes(), .other_pinned = plan.pinned_other_bytes()},
            options.context_cache.enabled ? options.context_cache.host_capacity_bytes.value_or(0U) : 0U,
            vram->query());
        demand.pageable = frontend_cache_bytes(options);
        // --expert-ram-layout exclusive (design §11.3; the 122B's default is full): RAM keeps only the experts the
        // frames do not, and shadows for what VRAM gives back without a promotion. The Program's engine sizes it once
        // its frames are known (ExpertOffload::plan_exclusive), so the banks load as the SSD tier's with the ledger's
        // share as its budget; an explicit --expert-ram-mib keeps the ledger's placement.
        exclusive_experts = options.expert_ram_layout.value_or(ExpertRamLayout::Full) == ExpertRamLayout::Exclusive &&
                            !options.expert_ram_bytes;
        const HostMemorySnapshot snapshot = host_memory_snapshot();
        if (prefix_auto) {
            // After the experts, as Qwen3.8-Flash-Next: the ledger places them first.
            options.context_cache.host_capacity_bytes =
                auto_host_context_bytes(expert_offload::plan_host_memory(snapshot, demand));
            set_growth(snapshot, demand);
            demand.prefix_cache = *options.context_cache.host_capacity_bytes;
        }
        if (plan_only) {
            // A start refuses an --expert-ram-mib the ledger cannot give (plan_expert_ram).
            const auto ledger = expert_offload::plan_host_memory(snapshot, demand);
            if (std::string refusal = expert_ram_refusal(options, ledger); !refusal.empty()) {
                plan_refuse(*planned, std::move(refusal));
                planned->ram.experts_bytes = ledger.expert_ram;
                plan_ram_totals(*planned, ledger.planned + ledger.expert_ram);
                return {};
            }
        }
        expert_ram = plan_expert_ram(options, demand, snapshot, exclusive_experts);
        if (!plan_only) { reserve_expert_working_set(options, expert_ram->demand); }
        if (expert_ram->tier) {
            auto streamed           = models::load_options(options, models::Architecture::Qwen3_5);
            streamed.stream_experts = true;
            streamed.a8_dense       = dense_a8;
            plan                    = models::qwen3_5::plan_load(reader, streamed);
        }
        plan.set_host_reserve(options.ram_free_bytes, expert_ram->demand.later_pinned);
    } else {
        // --vision-offload on: the weights' pool (the evictable ladder, load.cpp) maps the arena in whole chunks and,
        // once loaded, mirrors its evictable tail (the embedding, the heads, MTP) in pinned RAM, a pin after the
        // block's (EvictableWeightPool::capture_tail_mirror).
        const artifact::MaterializationPlan& placement = plan.materialization();
        ladder = placement.evictable_tail_bytes != 0;
        if (ladder) {
            tail_mirror = EvictableWeightPool::mirror_bytes(placement.device_capacity_bytes, placement.evictable_tail_bytes);
        }
        plan.set_host_reserve(options.ram_free_bytes, tail_mirror);
        if (prefix_auto) {
            // The weights sit in VRAM: the tier takes the RAM beside the model's pinned weights, the Program's later
            // pins and the frontend's caches, down to --ram-free-mib, within what the process may still page-lock.
            const auto vram = open_vram_budget_source(device.device, options.vram_past_budget);
            auto demand =
                expert_ram_demand(options, {.expert_banks = 0, .other_pinned = placement.pinned_capacity_bytes + tail_mirror},
                                  0U, vram->query());
            demand.full_margin = 0; // no experts to keep in RAM
            demand.pageable    = frontend_cache_bytes(options);
            const HostMemorySnapshot snapshot = host_memory_snapshot();
            options.context_cache.host_capacity_bytes =
                auto_host_context_bytes(expert_offload::plan_host_memory(snapshot, demand));
            set_growth(snapshot, demand);
            // The tier page-locks RAM down to the reserve, so the process's own pageable memory (its heap, the
            // driver's pages, the kernel images) is what Windows would trim first under pressure: the growth the
            // plan counted stays resident, as beside offloaded experts.
            if (!plan_only) { reserve_expert_working_set(options, demand); }
        }
    }
    // Engine::plan: the pinned block's check (materialize's), the RAM report, and the weights; then the
    // plan-only load reads the offloaded experts' scales from the artifact (streamed), as the SSD tier does.
    std::uint64_t planned_weights = 0, planned_beside_weights = 0;
    std::string pinned_refusal; // a start meets it in the load, after the dense weights' check
    if (plan_only) {
        const artifact::MaterializationPlan& placement = plan.materialization();
        // A DSpine drafter artifact's weights sit beside the model's, in their own allocation.
        planned_weights = placement.device_capacity_bytes + plan.draft_device_bytes();
        // The ladder's chunk alignment beyond the arena (an overlay window borrows evicted chunks: no staging).
        if (ladder) { planned_beside_weights = EvictableWeightPool::mapped_bytes(planned_weights) - planned_weights; }
        planned->vram.weights_bytes = planned_weights;
        EnginePlanRam& ram          = planned->ram;
        if (const std::uint64_t pinned = placement.pinned_capacity_bytes + tail_mirror; pinned != 0) {
            ram.pinned_bytes = pinned;
        }
        const std::uint64_t host_tier =
            options.context_cache.enabled ? options.context_cache.host_capacity_bytes.value_or(0U) : 0U;
        if (options.context_cache.enabled) { ram.host_context_bytes = host_tier; }
        if (expert_ram) {
            ram.experts_bytes = expert_ram->ledger.expert_ram;
            // Full: the ledger keeps every expert in RAM (the exclusive layout's tier only shares it with VRAM).
            ram.tier = expert_ram->ledger.placement == expert_offload::ExpertPlacement::Full
                           ? EnginePlanExpertTier::Full
                           : EnginePlanExpertTier::Ssd;
            ram.expert_ram_layout = exclusive_experts ? ExpertRamLayout::Exclusive : ExpertRamLayout::Full;
            plan_ram_totals(*planned, expert_ram->ledger.planned + expert_ram->ledger.expert_ram);
        } else {
            plan_ram_totals(*planned, placement.pinned_capacity_bytes + tail_mirror + host_tier);
        }
        pinned_refusal = artifact::pinned_shortfall(placement, available_host_memory_bytes());
        if (plan.offloaded_experts() && !(expert_ram && expert_ram->tier)) {
            auto streamed           = models::load_options(options, models::Architecture::Qwen3_5);
            streamed.stream_experts = true;
            streamed.a8_dense       = dense_a8;
            plan                    = models::qwen3_5::plan_load(reader, streamed);
        }
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
        // --expert-state off: no file and no calibrated seed (the cache starts empty).
        o.expert_seed = !options.expert_state_off;
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
            o.ram_exclusive         = exclusive_experts;
            o.expert_csf            = options.expert_csf;
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
        dense_bytes               = plan_only ? planned_weights
                                              : plan.materialization().device_capacity_bytes + plan.draft_device_bytes();
        const VramSnapshot before = spill_source->query();
        const std::uint64_t headroom =
            runtime::expert_offload::display_headroom(before.display, false, options.vram_headroom_bytes);
        if (dense_bytes + headroom > before.device_free) {
            std::string refusal = "the model's dense weights (" + std::to_string(dense_bytes >> 20) +
                                  " MiB) do not fit in the free device memory (" +
                                  std::to_string(before.device_free >> 20) + " MiB) beside the " +
                                  std::to_string(headroom >> 20) + " MiB VRAM headroom; free VRAM";
            if (!plan_only) { throw std::runtime_error(refusal); }
            plan_refuse(*planned, std::move(refusal));
            planned->vram.headroom_bytes = headroom;
            planned->vram.planned_bytes  = dense_bytes;
            return {};
        }
    }
    // Every start checks that its weights landed in device memory: a resident model's arena is as exposed to
    // the sysmem fallback as an offloaded one's dense weights, and without the check its spill surfaced only as
    // a KV sizing failure that did not name the weights. The load page-locks the pinned block and the ladder's
    // tail mirror meanwhile (its staging slots are released again), which the guard takes off the shared system
    // memory use it reads.
    std::uint64_t pinned_in_load = 0;
    if (!plan_only) {
        if (!spill_source) { spill_source = open_vram_budget_source(device.device, options.vram_past_budget); }
        dense_bytes    = plan.materialization().device_capacity_bytes + plan.draft_device_bytes();
        pinned_in_load = plan.materialization().pinned_capacity_bytes + tail_mirror;
        spill.emplace(*spill_source);
        spill->begin();
    }
    if (plan_only && !pinned_refusal.empty()) { plan_refuse(*planned, std::move(pinned_refusal)); }
    // The kernel images load beside the weights: the RAM plan above counted them as the process's growth.
    std::optional<ops::KernelImagesLoad> kernel_images;
    if (!plan_only) { kernel_images.emplace(device.device); }
    auto model = plan_only ? models::qwen3_5::plan_model(std::move(plan))
                          : models::qwen3_5::materialize_model(std::move(plan), device, &options.startup_observer);
    device.synchronize();
    if (kernel_images) { report_kernel_images(options, kernel_images->finish()); }
    if (spill) {
        // The upload's staging slots count as page-locked too: they are released before this reading, but a
        // release the OS has not yet taken off the process's use must not read as a spill.
        pinned_in_load += model->storage_stats().peak_staging_bytes;
        if (const std::uint64_t spilled = spill->end(dense_bytes, pinned_in_load); spilled != 0) {
            throw std::runtime_error(std::to_string(spilled >> 20) + " MiB of the model's " +
                                     (expert_ram ? "dense weights" : "weights") + " (" +
                                     std::to_string(dense_bytes >> 20) +
                                     " MiB) were placed in shared system memory by the driver's sysmem fallback: free "
                                     "VRAM and start again (" + spill->readings(dense_bytes) + ")");
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
    // The native Parameters allocate the DFlash2 full head's screen (models::qwen3_5::execution::Parameters::
    // draft_screen, 357.6 MB for the 27B's head) outside the weights' arena, after the check above: the same guard
    // reads its step, so a screen the driver placed in shared system memory fails the start by name instead of
    // slowing every proposal round.
    if (spill) { spill->begin(); }
    auto instance = std::make_unique<ModelInstance>(std::move(model), options);
    // The adapter's VRAM as allocated (or, for Engine::plan, as it would be): the lossless route's layout is larger.
    if (const auto* lora = instance->model->lora()) {
        publish_diagnostic(options.diagnostic_observer, DiagnosticLevel::Info, "LoRA adapter %s: %s; %llu MiB of VRAM",
                           lora->adapter->weights_path().parent_path().string().c_str(), lora->describe().c_str(),
                           static_cast<unsigned long long>((instance->parameters.lora_bytes() + (1U << 20) - 1) >> 20));
    }
    // --ablate: the cut applied, and a warning when the sidecar was fitted on another artifact.
    if (const auto* ablate = instance->model->ablate()) {
        publish_diagnostic(options.diagnostic_observer, DiagnosticLevel::Info, "ablation sidecar %s: %s",
                           ablate->path.string().c_str(), ablate->describe().c_str());
        if (const std::string warning = ablate->artifact_mismatch(instance->model->info().artifact_id); !warning.empty()) {
            publish_diagnostic(options.diagnostic_observer, DiagnosticLevel::Warning, "%s", warning.c_str());
        }
    }
    if (spill) {
        const std::uint64_t screen = instance->parameters.draft_screen.bytes;
        const std::uint64_t lora   = instance->parameters.allocated_adapter_bytes();
        const std::uint64_t beside = screen + lora;
        if (const std::uint64_t spilled = beside != 0 ? spill->end(beside, 0) : 0; spilled != 0) {
            // --ablate's direction counts beside a LoRA adapter's factors (Parameters::allocated_adapter_bytes).
            const char* adapter = instance->parameters.lora_factors.bytes != 0 ? "the LoRA adapter's factors"
                                                                               : "the ablation sidecar's direction";
            const std::string named = screen != 0 && lora != 0 ? std::string("the DFlash2 draft screen and ") + adapter
                                      : screen != 0            ? std::string("the DFlash2 draft screen")
                                                               : std::string(adapter);
            const char* what = named.c_str();
            throw std::runtime_error(std::to_string(spilled >> 20) + " MiB of " + what + " (" +
                                     std::to_string(beside >> 20) +
                                     " MiB) were placed in shared system memory by the driver's sysmem fallback: free "
                                     "VRAM and start again (" + spill->readings(beside) + ")");
        }
    }
    // Engine::plan: the device memory beside the weights' arena the Parameters plan (planned_bytes counts it).
    if (planned != nullptr) {
        if (const std::uint64_t screen = instance->parameters.draft_screen_bytes(); screen != 0) {
            planned->vram.draft_screen_bytes = screen;
        }
        if (const std::uint64_t adapter = instance->parameters.adapter_bytes(); adapter != 0) {
            planned->vram.adapter_bytes = adapter;
        }
    }
    frontend.complete();
    // --a8-activations (LOSSY, opt-in; design §19.3.31): the 8-bit activation routes of this artifact
    // (models::qwen3_5::A8Routes, which the Program's ExecutionSchedule takes too), in decode and verification calls
    // and in prefill calls, or a refusal where it has none. Every other call keeps its route.
    if (options.a8_activations) {
        const models::qwen3_5::A8Routes a8 =
            models::qwen3_5::a8_activation_routes(models::qwen3_5::fast_path_configuration(*instance->model, options));
        if (!a8.any() && !dense_a8) {
            throw std::invalid_argument(
                "--a8-activations: no 8-bit activation route applies to this artifact (it covers offloaded experts in "
                "decode and prefill, a groupwise Q4 MLP gate/up in prefill, and every NVFP4 and FP8 text projection "
                "of a dense NVFP4 27B artifact whose NVFP4 projections permit 4-bit activations)");
        }
        if (dense_a8) {
            publish_diagnostic(options.diagnostic_observer, DiagnosticLevel::Info,
                               "--a8-activations (approximate): every NVFP4 and FP8 text projection at every width "
                               "(NVFP4: int8 codes per 32 values with an FP32 scale; FP8: one E4M3 term); the head "
                               "and the drafters keep their routes");
        }
        // The CPU-served misses compute the GPU's 8-bit route (offloaded experts: the offload options exist).
        if (a8.experts) { offload_options.value().engine.narrow_a16 = ops::offloaded_moe::NarrowA16::kA8; }
        if (a8.any()) {
            publish_diagnostic(options.diagnostic_observer, DiagnosticLevel::Info,
                               "--a8-activations (approximate): %s; every other call keeps its route",
                               a8.describe().c_str());
        }
    }
    std::optional<StartupPhaseScope> planning;
    planning.emplace(options.startup_observer, StartupPhase::TargetFinalize);
    // --draft-tokens auto where its policy does not run: the configuration's fastest fixed length.
    const FastPath length_policy =
        options.speculative.backend == SpeculativeBackend::Mtp ? FastPath::AdaptiveMtp : FastPath::AdaptiveDFlash;
    const bool policy_excluded = std::find(options.fast_path_exclusions.begin(), options.fast_path_exclusions.end(),
                                           length_policy) != options.fast_path_exclusions.end();
    if (const std::uint32_t fixed = models::qwen3_5::resolve_auto_draft_length(instance->parameters, options)) {
        publish_diagnostic(options.diagnostic_observer, DiagnosticLevel::Info,
                           "--draft-tokens auto: each round drafts %u tokens, the fastest fixed length here (%s, "
                           "which adapts the length, %s)",
                           fixed, fast_path_name(length_policy),
                           !options.fast_paths ? "is off with the fast paths"
                           : policy_excluded   ? "is off (--no-fast-path)"
                                               : "is not qualified for this artifact");
    }
    // TEMPORARY (MTP drafting package A/B): what the switches resolved to (their values are in the
    // startup log's temporary switches line).
    if (options.speculative.backend == SpeculativeBackend::Mtp) {
        publish_diagnostic(options.diagnostic_observer, DiagnosticLevel::Debug, "mtp drafting | %s",
                           models::qwen3_5::mtp_draft_switch_report(instance->parameters, options).c_str());
    }
    const auto signature = models::qwen3_5::prefill_signature(*instance->model);
    auto context_cost    = resolve_context_machine_cost(
        {.hardware_class =
                context_cost_hardware_class(device.props.name, device.props.major, device.props.minor),
            .prefill_signature = signature},
        options.context_cost.preset_path);
    std::size_t retry_headroom = 0;
    // Offloaded experts take the VRAM the Program leaves (their frames are sized after its fixed
    // allocations), so an automatic KV capacity is every lane's full context, as on
    // Qwen3.8-Flash-Next, rather than the free VRAM.
    KvCapacityPolicy kv_policy = options.kv_capacity;
    std::uint32_t planned_prefill_chunk = 0, planned_idle_width = 0;
    std::optional<StartupPhaseScope> program;
    for (int attempt = 0;; ++attempt) {
        if (!planning) { planning.emplace(options.startup_observer, StartupPhase::TargetFinalize); }
        const models::qwen3_5::detail::ProgramOffloadOptions* offload = offload_options ? &*offload_options : nullptr;
        // Engine::plan reads the free memory where a start does, less the weights it did not upload and the
        // parameters' derived buffers it did not allocate (the DFlash2 drafter's head screen).
        const VramSnapshot vram_now  = open_vram_budget_source(device.device, options.vram_past_budget)->query();
        const std::size_t free_now   = free_device_bytes(vram_now);
        // What the capacity resolves in: with a resident model's automatic capacity the display headroom comes
        // off free device memory and the margin off the OS budget (kv_sizing_bytes); otherwise free_now.
        const std::size_t sizing_headroom =
            static_cast<std::size_t>(runtime::expert_offload::display_headroom(vram_now.display, false,
                                                                              options.vram_headroom_bytes)) +
            retry_headroom;
        const std::size_t sizing_now = kv_sizing_bytes(
            vram_now, !expert_ram && kv_policy.mode == KvCapacityMode::Automatic, sizing_headroom, retry_headroom);
        std::size_t free_bytes =
            plan_only ? planned_free_after_weights(sizing_now, planned_weights,
                                                   planned_beside_weights + instance->parameters.draft_screen_bytes() +
                                                       instance->parameters.dspine_head_write_bytes() +
                                                       instance->parameters.adapter_bytes())
                      : sizing_now;
        // Other processes' VRAM by NVML, which counts every process (the OS budget does not): its device-wide use
        // less this process's; 0 without NVML. On the RTX 5090 it read ~200 MiB on a quiet card (the desktop
        // window manager's) and ~3,700 MiB beside a 3 GiB holder (2026-11-17).
        const std::uint64_t others_bytes =
            vram_now.physical_used > vram_now.local_usage ? vram_now.physical_used - vram_now.local_usage : 0U;
        if (vram_now.has_budget) {
            publish_diagnostic(options.diagnostic_observer, DiagnosticLevel::Debug,
                               "KV sizing: %zu MiB free (cudaMemGetInfo %zu MiB; OS budget %llu MiB, %llu used; NVML: "
                               "every process uses %llu MiB, other processes %llu)",
                               free_now >> 20, vram_now.device_free >> 20,
                               static_cast<unsigned long long>(vram_now.local_budget >> 20),
                               static_cast<unsigned long long>(vram_now.local_usage >> 20),
                               static_cast<unsigned long long>(vram_now.physical_used >> 20),
                               static_cast<unsigned long long>(others_bytes >> 20));
        }
        // An automatic KV capacity leaves the offloaded models' display headroom (256 MiB with no display on
        // the GPU, 1 GiB with one or when unknown). Measured headless on the 27B and 35B at 4 x 65,536 (RTX
        // 5090, 2026-11-03): the earlier fixed 1 GiB stayed free, and long prefills, decode at four lanes
        // and a constrained tool call took 2 MiB more after startup.
        const std::size_t headroom = sizing_headroom;
        // A resident model's automatic capacity under an OS budget (Windows), while other processes hold more VRAM
        // than the headroom leaves them, takes only what stepped growth maps at startup (probe_growth), so the
        // memory they hold stays theirs. The capacity is fixed from then on: it never shrinks after startup.
        //  - The probe costs ~270 ms (11.8 GiB in 64 MiB steps), so it runs only past the headroom by
        //    kOthersSlackBytes, which covers NVML's and the budget's differing accounts: a quiet card or the
        //    desktop alone pay nothing and size as before.
        //  - An explicit --max-context keeps one request's whole context: below it the probe gives way and Windows
        //    pages the other program out, as before. --max-context auto takes what fits.
        //  - Engine::plan cannot allocate, so it keeps the budget's figure: the two differ while the probe acts.
        constexpr std::uint64_t kOthersSlackBytes = 256ULL << 20;
        if (!plan_only && !expert_ram && kv_policy.mode == KvCapacityMode::Automatic && vram_now.has_budget &&
            vram_now.physical_used != 0 && others_bytes > headroom + kOthersSlackBytes && free_bytes > headroom &&
            VmmArena::supported(device.device)) {
            const auto source       = open_vram_budget_source(device.device, options.vram_past_budget);
            const GrowthProbe probe = probe_growth(device.device, *source, expert_offload::GrowthLimit{headroom, 0},
                                                   free_bytes - headroom);
            std::size_t probed = static_cast<std::size_t>(probe.bytes) + headroom;
            if (probe.stopped && !options.max_context_auto) {
                const auto curve =
                    models::qwen3_5::make_sequence_planner(instance->parameters, device, options, offload).capacity_curve();
                const std::uint64_t pages = std::clamp<std::uint64_t>(
                    (static_cast<std::uint64_t>(options.max_context) + curve.main_page_tokens - 1U) / curve.main_page_tokens,
                    curve.minimum_main_page_groups, curve.maximum_main_page_groups);
                probed = std::max(probed, curve.reservation_bytes(static_cast<std::uint32_t>(pages)) + headroom);
            }
            if (probe.stopped && probed < free_bytes) {
                publish_diagnostic(options.diagnostic_observer, DiagnosticLevel::Info,
                                   "other programs hold %llu MiB of VRAM the OS budget does not show: the automatic KV "
                                   "capacity takes %zu MiB instead of %zu, so they keep it",
                                   static_cast<unsigned long long>(others_bytes >> 20), (probed - headroom) >> 20,
                                   (free_bytes - headroom) >> 20);
                free_bytes = probed;
            }
            publish_diagnostic(options.diagnostic_observer, DiagnosticLevel::Debug,
                               "KV room probe: %zu MiB mapped in 64 MiB steps in %.0f ms%s",
                               static_cast<std::size_t>(probe.bytes >> 20), probe.milliseconds,
                               probe.stopped ? ", stopped short" : "");
            if (!probe.spill_readings.empty()) {
                publish_diagnostic(options.diagnostic_observer, DiagnosticLevel::Debug,
                                   "KV room probe: a chunk read as placed in system memory stopped it: %s",
                                   probe.spill_readings.c_str());
            }
        }
        // The sequence plan of `trial` (these options at another prefill chunk, for the fit below) with its
        // KV capacity resolved; none when Engine::plan finds no capacity that fits.
        const auto plan_sequence = [&](const EngineOptions& trial)
            -> std::optional<std::pair<models::qwen3_5::SequencePlan, KvCapacityResolution>> {
            auto planner = models::qwen3_5::make_sequence_planner(instance->parameters, device, trial, offload);
            if (expert_ram && trial.kv_capacity.mode == KvCapacityMode::Automatic) {
                // Every lane's full context in whole pages (a lane's last page holds its context's tail).
                const std::uint64_t page   = planner.capacity_curve().main_page_tokens;
                const std::uint64_t tokens = static_cast<std::uint64_t>(trial.max_concurrency) *
                                             ((static_cast<std::uint64_t>(trial.max_context) + page - 1U) / page) * page;
                if (tokens > std::numeric_limits<std::uint32_t>::max()) {
                    throw std::invalid_argument("--max-context x --max-concurrency exceeds the KV capacity range");
                }
                kv_policy = KvCapacityPolicy::explicit_capacity(static_cast<std::uint32_t>(tokens));
            }
            KvCapacityResolution resolution;
            if (plan_only) {
                const auto fitted = plan_capacity(*planned, kv_policy, planner.capacity_curve(), free_bytes, headroom);
                if (!fitted) { return std::nullopt; }
                resolution = *fitted;
            } else {
                resolution = resolve_kv_capacity(kv_policy, planner.capacity_curve(), free_bytes, headroom);
            }
            // idle-wide-prefill: plan the widest idle width (4096, 2048, then 1024 tokens, as the chunk is
            // rounded) whose larger prefill buffers and workspace keep the KV capacity: an explicit one
            // whole, an automatic one down to every lane's full context. None fitting keeps chunk-wide steps.
            {
                const std::uint64_t page_tokens = planner.capacity_curve().main_page_tokens;
                std::uint64_t lane_page_groups =
                    static_cast<std::uint64_t>(trial.max_concurrency) *
                    ((static_cast<std::uint64_t>(trial.max_context) + page_tokens - 1U) / page_tokens);
                // INFERNIX_TMP_IDLE_WIDE_KV_PERCENT: the idle width may also cost that share of the pages.
                if (const std::int64_t percent =
                        temporary_switches::integer("INFERNIX_TMP_IDLE_WIDE_KV_PERCENT", 0, 0, 20);
                    percent != 0) {
                    lane_page_groups = std::min<std::uint64_t>(
                        lane_page_groups,
                        resolution.main_page_groups * static_cast<std::uint64_t>(100 - percent) / 100U);
                }
                for (const std::uint32_t width : models::qwen3_5::idle_prefill_widths(instance->parameters, trial)) {
                    auto wide = models::qwen3_5::make_sequence_planner(instance->parameters, device, trial, width, offload);
                    if (const auto fitted = resolve_kv_capacity_beside(kv_policy, wide.capacity_curve(), free_bytes,
                                                                       headroom, resolution, lane_page_groups)) {
                        planner    = std::move(wide);
                        resolution = *fitted;
                        break;
                    }
                }
            }
            auto sequence = std::move(planner).finalize(resolution.main_page_groups);
            if (sequence.device_reservation_bytes() != resolution.runtime_reservation_bytes ||
                sequence.kv_capacity() != resolution.resolved_tokens) {
                throw std::logic_error("resolved KV capacity does not match the finalized Program plan");
            }
            return std::pair{std::move(sequence), resolution};
        };
        // --max-context auto: the largest context whose smallest reservation resolves here (Engine::plan's
        // answer), so one request may fill the KV pool; every attempt resolves it again, as a retry's larger
        // headroom leaves fewer pages. The frontend and the Engine's capacity take it.
        if (options.max_context_auto) {
            EnginePlan probe;
            plan_qwen3_5_context(probe, options, instance->parameters, device, offload, free_bytes, headroom);
            if (!probe.max_context) {
                throw std::runtime_error("--max-context auto: no context fits beside the weights in the free VRAM");
            }
            if (*probe.max_context != options.max_context) {
                options.max_context = *probe.max_context;
                instance->resize_context(options);
            }
            // The model's positions bound it (plan_qwen3_5_context): its native context, more only with YaRN.
            const double positions =
                static_cast<double>(instance->parameters.model.config().text.max_position_embeddings) *
                std::max(1.0, static_cast<double>(options.rope_yarn_factor));
            if (static_cast<double>(options.max_context) >= positions) {
                publish_diagnostic(options.diagnostic_observer, DiagnosticLevel::Info,
                                   options.rope_yarn_factor > 1.0F
                                       ? "--max-context auto: %u tokens, the positions --rope-yarn-factor allows"
                                       : "--max-context auto: %u tokens, the model's native context (more needs "
                                         "--rope-yarn-factor)",
                                   options.max_context);
            } else {
                publish_diagnostic(options.diagnostic_observer, DiagnosticLevel::Info,
                                   "--max-context auto: %u tokens, the largest context the KV cache holds here",
                                   options.max_context);
            }
        }
        auto planned_sequence = plan_sequence(options);
        if (!planned_sequence) {
            plan_qwen3_5_context(*planned, options, instance->parameters, device, offload, free_bytes, headroom);
            return {};
        }
        if (offload != nullptr) {
            // The widest prefill chunk at most --prefill-chunk (on its 128-token grid) whose widest pass the
            // expert cache lends with every lane's context full (offloaded_prefill_lent): the workspace grows
            // with the chunk and the frames shrink, so the widest that lends is the fastest. Its buffers are
            // planned at that chunk, as Engine::plan reports them; a start sizes its frames from a live reading,
            // so a launch smaller than the estimate still lends the workspace per prompt when it must
            // (ProgramOffload::wide_workspace). As Qwen3.8-Flash-Next's start fits its chunk (design §19.3.7),
            // though the 122B's chunk is its prefill grid (taps, resumes, replay steps), so it is fitted once,
            // for full contexts, not re-fitted per call.
            const VramSnapshot snapshot = open_vram_budget_source(device.device, options.vram_past_budget)->query();
            const std::uint64_t not_uploaded = plan_only ? planned->vram.weights_bytes : 0U;
            const bool reproducible          = !options.fast_paths;
            const auto lent = [&](const std::pair<models::qwen3_5::SequencePlan, KvCapacityResolution>& p) {
                return offloaded_prefill_lent(*instance->model, p.first, p.second.runtime_reservation_bytes, not_uploaded,
                                              snapshot, in_use_before_load, reproducible, offload->engine, device);
            };
            if (const PrefillLend requested = lent(*planned_sequence); !requested.lent()) {
                const std::uint32_t asked = options.prefill_chunk;
                const std::uint64_t pass  = planned_sequence->first.offload_wide_pass_bytes();
                const auto lent_at = [&](std::uint32_t chunk) {
                    EngineOptions trial  = options;
                    trial.prefill_chunk  = chunk;
                    const auto candidate = plan_sequence(trial);
                    return candidate ? lent(*candidate) : PrefillLend{false, false};
                };
                constexpr std::uint32_t kGrid = models::qwen3_5::kPrefillChunkAlignment;
                std::uint32_t low = 1, high = asked / kGrid - 1U, fitted = 0;
                while (low <= high) {
                    const std::uint32_t mid = low + (high - low) / 2U;
                    if (lent_at(mid * kGrid).lent()) {
                        fitted = mid * kGrid;
                        low    = mid + 1U;
                    } else {
                        high = mid - 1U;
                    }
                }
                if (fitted != 0) {
                    // What bound it: this launch's cache, or the nominal one, one grid step above.
                    const PrefillLend above = fitted + kGrid == asked ? requested : lent_at(fitted + kGrid);
                    options.prefill_chunk = fitted;
                    planned_sequence      = plan_sequence(options);
                    if (!planned_sequence) { throw std::logic_error("the fitted prefill chunk lost its KV capacity"); }
                    publish_diagnostic(
                        options.diagnostic_observer, DiagnosticLevel::Warning,
                        "--prefill-chunk %u needs %llu MiB of workspace for its widest pass, more than the expert cache "
                        "can lend once every lane's context is full; prefilling at %u tokens (--prefill-chunk %u "
                        "plans that from the start)",
                        asked, static_cast<unsigned long long>(pass >> 20), fitted, fitted);
                    if (reproducible && !above.now && above.nominal) {
                        publish_diagnostic(
                            options.diagnostic_observer, DiagnosticLevel::Warning,
                            "the expert cache is smaller than the one the reproducible route fits its prefill chunk to "
                            "(%llu MiB of VRAM were in use before loading, against a nominal %llu): this launch prefills "
                            "at %u tokens, so its output can differ from a launch that fits a wider chunk (free VRAM, or "
                            "set --prefill-chunk %u on every launch)",
                            static_cast<unsigned long long>(in_use_before_load.value_or(0) >> 20),
                            static_cast<unsigned long long>(expert_offload::kNominalInUseBytes >> 20), fitted, fitted);
                    }
                }
            }
        }
        auto& [sequence, resolution] = *planned_sequence;
        if (plan_only) {
            plan_qwen3_5_program(*planned, options, *instance, sequence, resolution, free_bytes,
                                 offload_options ? &offload_options->engine : nullptr, device);
            plan_qwen3_5_context(*planned, options, instance->parameters, device, offload, free_bytes, headroom);
            return {};
        }
        options.context_cache.host_capacity_bytes = sequence.host_capacity_bytes();
        if (sequence.host_capacity_bytes() == 0) { options.context_cache.host_growth_capacity_bytes.reset(); }
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
    if (options.ram_resize) {
        instance->program->start_ram_resize(
            options.ram_free_bytes, prefix_auto,
            [observer = options.diagnostic_observer](const std::string& text, DiagnosticLevel level) {
                publish_diagnostic(observer, level, "%s", text.c_str());
            });
    }
    if (options.decision_lanes != 0) {
        // Decision lanes are ordinary lanes (a StateImage, buffers and Graph rows each), so their
        // Device memory comes out of the KV budget: an automatic pool is that much smaller, an
        // explicit one needs that much more free memory. Say how much, from the same planner.
        EngineOptions general   = options;
        general.max_concurrency = options.max_concurrency - options.decision_lanes;
        general.decision_lanes  = 0;
        const SequenceCapacityCurve with =
            models::qwen3_5::make_sequence_planner(instance->parameters, device, options)
                .capacity_curve();
        const SequenceCapacityCurve without =
            models::qwen3_5::make_sequence_planner(instance->parameters, device, general)
                .capacity_curve();
        const std::size_t lane_bytes =
            with.minimum_device_reservation_bytes > without.minimum_device_reservation_bytes
                ? with.minimum_device_reservation_bytes - without.minimum_device_reservation_bytes
                : 0U;
        const std::uint64_t lane_tokens =
            with.bytes_per_additional_main_page_group == 0
                ? 0U
                : static_cast<std::uint64_t>(lane_bytes) / with.bytes_per_additional_main_page_group *
                      with.main_page_tokens;
        publish_diagnostic(
            options.diagnostic_observer, DiagnosticLevel::Info,
            "decision lanes: %u of %u lanes take only decisions; they take %.0f MiB of Device memory "
            "(about %llu KV tokens), %s; KV capacity is %u tokens",
            options.decision_lanes, options.max_concurrency,
            static_cast<double>(lane_bytes) / static_cast<double>(1U << 20U),
            static_cast<unsigned long long>(lane_tokens),
            options.kv_capacity.mode == KvCapacityMode::Automatic
                ? "taken from the automatic KV pool"
                : "beside the explicit --kv-capacity",
            instance->kv_capacity_resolution.resolved_tokens);
    }
    LoadSummary::PrefixCacheRestore restore;
    if (options.context_cache.enabled) {
        instance->program->set_hybrid_cost(hybrid_cache_cost(context_cost.model));
        // A request waiting for a sibling's snapshot stays in the FIFO, so the predicted wait
        // is kept well inside its queue timeout.
        instance->program->set_hybrid_coalesce_wait_limit(
            static_cast<double>(options.pending_timeout_ms) / 1000.0 / 2.0);
        const std::filesystem::path& file = options.context_cache.hybrid.persistent_file;
        // A prefix cache file no longer used leaves its kept SSD file behind, which nothing removes: the start names
        // each (another running server's is among them, so none is deleted).
        if (!file.empty() && options.context_cache.ssd_capacity_bytes != 0) {
            for (const runtime::prefix_cache::KeptSsdFile& other : runtime::prefix_cache::other_kept_ssd_files(
                     options.context_cache.ssd_directory,
                     runtime::prefix_cache::kept_ssd_file(options.context_cache.ssd_directory, file))) {
                publish_diagnostic(options.diagnostic_observer, DiagnosticLevel::Info,
                                   "SSD prefix cache: %s (%llu MiB) is not used by this server",
                                   other.path.string().c_str(), static_cast<unsigned long long>(other.bytes >> 20));
            }
        }
        if (!file.empty()) {
            // A DSpine drafter's ring rides in every snapshot: it belongs to that drafter artifact.
            std::string cache_signature = signature;
            if (const auto& draft = instance->model->info().draft_artifact_id) {
                cache_signature += ";draft=";
                for (const std::byte byte : *draft) {
                    constexpr char kHex[] = "0123456789abcdef";
                    cache_signature += kHex[std::to_integer<unsigned>(byte) >> 4U];
                    cache_signature += kHex[std::to_integer<unsigned>(byte) & 15U];
                }
            }
            const runtime::prefix_cache::PersistResult loaded =
                instance->program->attach_hybrid_cache_file(
                    file, hybrid_cache_fingerprint(options, cache_signature, device.props),
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
                .ssd_blocks          = loaded.ssd_blocks,
                .ssd_snapshots       = loaded.ssd_snapshots,
                .ssd_message         = loaded.ssd_message,
            };
        }
    }
    device.synchronize();
    program->complete();
    instance->kv_capacity_resolution.available_after_startup_bytes =
        current_free_device_bytes(device.device, options.vram_past_budget);
    const auto& stats = instance->model->storage_stats();
    LoadSummary summary;
    summary.architecture = models::architecture_name(instance->model->config().text.architecture);
    summary.model_name   = instance->model->info().name;
    if (const auto* lora = instance->model->lora()) {
        summary.lora_adapter = lora->adapter->weights_path().parent_path().string() + ": " + lora->describe();
    }
    if (const auto* ablate = instance->model->ablate()) {
        summary.ablate_sidecar = ablate->path.string() + ": " + ablate->describe();
    }
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
        for (const FastPath path : kFastPaths) {
            if (models::qwen3_5::fast_path_applies(path, fast.configuration, options)) {
                summary.fast_paths_applicable.emplace_back(fast_path_name(path));
            }
        }
        // A tail skip depends on what the expert cache holds; the 8-bit routes' arithmetic on the call's width.
        summary.request_invariant =
            fast.request_invariant() && options.expert_tail_skip == 0 && !options.a8_activations;
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
