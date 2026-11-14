#include "runtime/engine/expert_memory.h"

#include "core/host_memory.h"
#include "infernix/ops/kernel_images.h"

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace infernix::runtime {
namespace {

unsigned long long mib(std::uint64_t bytes) { return static_cast<unsigned long long>(bytes >> 20); }

} // namespace

void report_diagnostic(const DiagnosticObserver& observer, DiagnosticLevel level, const std::string& text) {
    if (observer.callback) {
        try {
            observer.callback(Diagnostic{.level = level, .message = text});
        } catch (...) {}
    } else if (level != DiagnosticLevel::Debug) {
        std::fprintf(stderr, "[engine] %s\n", text.c_str());
    }
}

void report_engine(const EngineOptions& options, DiagnosticLevel level, const std::string& text) {
    report_diagnostic(options.diagnostic_observer, level, text);
}

expert_offload::HostMemoryDemand expert_ram_demand(const EngineOptions& options, const ExpertPinnedTerms& pinned,
                                                   std::uint64_t prefix_cache_bytes, const VramSnapshot& vram) {
    expert_offload::HostMemoryDemand demand;
    demand.reserve      = options.ram_free_bytes;
    demand.expert_banks = pinned.expert_banks;
    demand.other_pinned = pinned.other_pinned;
    demand.later_pinned = kProgramPinnedAllowance;
    // The prefix cache pins its whole Host tier at startup, beside the experts.
    demand.prefix_cache = prefix_cache_bytes;
    demand.load_staging = kLoadStagingBytes;
    // A start has read the kernel images into RAM before planning (load_kernel_images); Engine::plan has not, and
    // counts them as the process's growth.
    demand.process_growth = expert_offload::kLedgerGiB + ops::kernel_images_pending_bytes();
    demand.expert_cap   = options.expert_ram_bytes;
    // Windows page-locks memory for the GPU only up to the adapter's shared system memory; the
    // experts past that stay in RAM unpinned (the SSD tier's unpinned records).
    demand.pin_room       = expert_offload::pin_room(vram);
    demand.expert_pin_cap = options.expert_pin_bytes;
    return demand;
}

std::uint64_t ledger_left_bytes(const expert_offload::HostMemoryLedger& ledger) noexcept {
    const std::uint64_t used = ledger.demand.reserve + ledger.planned + ledger.expert_ram;
    return ledger.snapshot.available_physical > used ? ledger.snapshot.available_physical - used : 0;
}

std::uint64_t frontend_cache_bytes(const EngineOptions& options) noexcept {
    return options.grammar_cache_bytes + options.speculative.ngram_archive_bytes +
           (options.enable_vision ? options.media_cache_bytes + options.media_live_bytes : 0U);
}

std::optional<std::uint64_t> ledger_pin_left_bytes(const expert_offload::HostMemoryLedger& ledger) noexcept {
    const expert_offload::HostMemoryDemand& demand = ledger.demand;
    if (!demand.pin_room) { return std::nullopt; }
    const std::uint64_t others = demand.other_pinned + demand.later_pinned + demand.prefix_cache;
    const std::uint64_t pinned =
        others + static_cast<std::uint64_t>(static_cast<double>(others) * demand.lock_overhead) + 1 + ledger.expert_pinned;
    return *demand.pin_room > pinned ? *demand.pin_room - pinned : 0;
}

std::uint64_t auto_host_context_bytes(const expert_offload::HostMemoryLedger& ledger) noexcept {
    constexpr std::uint64_t kStep = 256ULL << 20;
    // Physical memory, and the commit limit (a small page file), each beyond the reserve and the plan.
    const std::uint64_t used        = ledger.demand.reserve + ledger.planned + ledger.expert_ram;
    const std::uint64_t commit_left = ledger.snapshot.available_commit > used ? ledger.snapshot.available_commit - used : 0;
    std::uint64_t left              = std::min(ledger_left_bytes(ledger), commit_left);
    if (const auto pin_left = ledger_pin_left_bytes(ledger)) { left = std::min(left, *pin_left); }
    // The tier is pinned: its lock overhead comes out of the same room.
    const auto bytes = static_cast<std::uint64_t>(static_cast<double>(left) / (1.0 + ledger.demand.lock_overhead));
    return std::max(kMinAutoHostContextBytes, bytes / kStep * kStep);
}

std::uint64_t auto_host_context_ceiling_bytes(HostMemorySnapshot snapshot, const expert_offload::HostMemoryDemand& demand,
                                              std::uint64_t startup) noexcept {
    // What the rest of the system holds beyond this process: other programs, and the OS's own share, which never comes
    // free (the growth stops at the RAM actually available; this only bounds it).
    const std::uint64_t held   = snapshot.available_physical + snapshot.process_resident;
    const std::uint64_t others = snapshot.total_physical > held ? snapshot.total_physical - held : 0;
    snapshot.available_physical += others;
    snapshot.available_commit += others;
    return std::max(startup, auto_host_context_bytes(expert_offload::plan_host_memory(snapshot, demand)));
}

ExpertRamPlan plan_expert_ram(const EngineOptions& options, const expert_offload::HostMemoryDemand& demand,
                              const HostMemorySnapshot& snapshot, bool always_tier) {
    ExpertRamPlan out;
    out.demand = demand;
    out.ledger = expert_offload::plan_host_memory(snapshot, demand);
    report_engine(options, DiagnosticLevel::Info, out.ledger.summary());
    report_engine(options, DiagnosticLevel::Debug, out.ledger.describe());
    if (std::string refusal = expert_ram_refusal(options, out.ledger); !refusal.empty()) {
        throw std::runtime_error(refusal);
    }
    // The SSD tier: the banks stay in the artifact; the Program pins RAM slots for the experts the
    // ledger leaves room for (less their lock overhead) and reads the rest on use.
    out.tier = out.ledger.placement == expert_offload::ExpertPlacement::Tier || always_tier;
    if (out.tier) {
        out.expert_ram_bytes =
            static_cast<std::uint64_t>(static_cast<double>(out.ledger.expert_pinned) / (1.0 + demand.lock_overhead));
        out.expert_unpinned_bytes = out.ledger.expert_pageable;
        out.later_pinned_bytes    = demand.prefix_cache + demand.later_pinned + expert_offload::kPinRoomMarginBytes;
    }
    return out;
}

std::string expert_ram_refusal(const EngineOptions& options, const expert_offload::HostMemoryLedger& ledger) {
    if (!ledger.cap_too_large) { return {}; }
    return "--expert-ram-mib " + std::to_string(options.expert_ram_bytes.value_or(0) >> 20) +
           " asks for more Host RAM than is free or pinnable now (" + ledger.describe() +
           "); lower it, use auto, or free Host memory";
}

void reserve_expert_working_set(const EngineOptions& options, const expert_offload::HostMemoryDemand& demand) {
    // Keep Infernix's own pageable memory resident so its heap and driver pages never hard-fault from
    // the page file.
    if (const auto refused = reserve_process_working_set(demand.process_growth); !refused.empty()) {
        report_engine(options, DiagnosticLevel::Warning,
                      "Windows would not guarantee Infernix " + std::to_string(demand.process_growth >> 20) +
                          " MiB of resident working memory (" + refused +
                          "); everything still works, but if RAM runs short Windows may move some of it to the "
                          "page file and responses slow down");
    }
}

ExpertVramCheck size_expert_vram(const EngineOptions& options, const std::string& model, const VramSnapshot& before,
                                 std::uint64_t fixed, std::uint32_t graphs, std::uint64_t frame_bytes,
                                 std::uint32_t max_frames, const std::string& contributors) {
    expert_offload::VramDemand demand;
    demand.headroom    = options.vram_headroom_bytes;
    demand.graphs      = graphs;
    demand.fixed       = fixed;
    demand.frame_bytes = frame_bytes; // an average frame of the per-size pools
    demand.max_frames  = max_frames;
    ExpertVramCheck out;
    out.sizing = expert_offload::size_expert_frames(before, demand);
    const auto& check = out.sizing;
    if (check.frame_bytes < 0) {
        out.refusal =
            model + " does not fit in device memory: " + std::to_string(mib(before.device_free)) + " MiB free; " +
            contributors + "; reserve " + std::to_string(mib(check.reserve)) + "; headroom " +
            std::to_string(mib(check.headroom)) + " (" + check.describe() +
            "). Lower --max-context, --max-concurrency or --prefill-chunk, use --kv-dtype int8, lower "
            "--vram-headroom-mib, or free VRAM";
    }
    return out;
}

expert_offload::VramSizing check_expert_vram(const EngineOptions& options, const std::string& model,
                                             const VramSnapshot& before, std::uint64_t fixed, std::uint32_t graphs,
                                             std::uint64_t frame_bytes, std::uint32_t max_frames,
                                             const std::string& contributors) {
    ExpertVramCheck sized = size_expert_vram(options, model, before, fixed, graphs, frame_bytes, max_frames, contributors);
    if (!sized.refusal.empty()) { throw std::runtime_error(sized.refusal); }
    const auto& check = sized.sizing;
    if (check.frames < kFewExpertFrames) {
        report_engine(options, DiagnosticLevel::Warning,
                      "decode will be slow: only " + std::to_string(check.frames) + " expert frames fit in VRAM (" +
                          std::to_string(kFewExpertFrames) +
                          " or more recommended); lower --max-context or --max-concurrency, or free VRAM");
    }
    return check;
}

} // namespace infernix::runtime
