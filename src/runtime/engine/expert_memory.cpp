#include "runtime/engine/expert_memory.h"

#include "core/host_memory.h"

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
    demand.reserve      = options.ram_headroom_bytes;
    demand.expert_banks = pinned.expert_banks;
    demand.other_pinned = pinned.other_pinned;
    demand.later_pinned = kProgramPinnedAllowance;
    // The prefix cache pins its whole Host tier at startup, beside the experts.
    demand.prefix_cache = prefix_cache_bytes;
    demand.load_staging = kLoadStagingBytes;
    demand.expert_cap   = options.expert_ram_bytes;
    // Windows page-locks memory for the GPU only up to the adapter's shared system memory; the
    // experts past that stay in RAM unpinned (the SSD tier's unpinned records).
    demand.pin_room       = expert_offload::pin_room(vram);
    demand.expert_pin_cap = options.expert_pin_bytes;
    return demand;
}

ExpertRamPlan plan_expert_ram(const EngineOptions& options, const expert_offload::HostMemoryDemand& demand,
                              const HostMemorySnapshot& snapshot) {
    ExpertRamPlan out;
    out.demand = demand;
    out.ledger = expert_offload::plan_host_memory(snapshot, demand);
    report_engine(options, DiagnosticLevel::Info, out.ledger.summary());
    report_engine(options, DiagnosticLevel::Debug, out.ledger.describe());
    if (out.ledger.cap_too_large) {
        throw std::runtime_error("--expert-ram-mib " + std::to_string(*options.expert_ram_bytes >> 20) +
                                 " asks for more Host RAM than is free or pinnable now (" + out.ledger.describe() +
                                 "); lower it, use auto, or free Host memory");
    }
    // The SSD tier: the banks stay in the artifact; the Program pins RAM slots for the experts the
    // ledger leaves room for (less their lock overhead) and reads the rest on use.
    out.tier = out.ledger.placement == expert_offload::ExpertPlacement::Tier;
    if (out.tier) {
        out.expert_ram_bytes =
            static_cast<std::uint64_t>(static_cast<double>(out.ledger.expert_pinned) / (1.0 + demand.lock_overhead));
        out.expert_unpinned_bytes = out.ledger.expert_pageable;
        out.later_pinned_bytes    = demand.prefix_cache + demand.later_pinned + expert_offload::kPinRoomMarginBytes;
    }
    // A small reserve trims other programs' working sets; keep Infernix's own pageable memory
    // resident so its heap and driver pages never hard-fault from the page file.
    if (const auto refused = reserve_process_working_set(demand.process_growth); !refused.empty()) {
        report_engine(options, DiagnosticLevel::Warning,
                      "Windows would not guarantee Infernix " + std::to_string(demand.process_growth >> 20) +
                          " MiB of resident working memory (" + refused +
                          "); everything still works, but if RAM runs short Windows may move some of it to the "
                          "page file and responses slow down");
    }
    return out;
}

expert_offload::VramSizing check_expert_vram(const EngineOptions& options, const std::string& model,
                                             const VramSnapshot& before, std::uint64_t fixed, std::uint32_t graphs,
                                             std::uint64_t frame_bytes, std::uint32_t max_frames,
                                             const std::string& contributors) {
    expert_offload::VramDemand demand;
    demand.headroom    = options.vram_headroom_bytes;
    demand.graphs      = graphs;
    demand.fixed       = fixed;
    demand.frame_bytes = frame_bytes; // an average frame of the per-size pools
    demand.max_frames  = max_frames;
    const auto check   = expert_offload::size_expert_frames(before, demand);
    if (check.frame_bytes < 0) {
        throw std::runtime_error(
            model + " does not fit in device memory: " + std::to_string(mib(before.device_free)) + " MiB free; " +
            contributors + "; reserve " + std::to_string(mib(check.reserve)) + "; headroom " +
            std::to_string(mib(check.headroom)) + " (" + check.describe() +
            "). Lower --max-context, --max-concurrency or --prefill-chunk, use --kv-dtype int8, lower "
            "--vram-headroom-mib, or free VRAM");
    }
    if (check.frames < kFewExpertFrames) {
        report_engine(options, DiagnosticLevel::Warning,
                      "decode will be slow: only " + std::to_string(check.frames) + " expert frames fit in VRAM (" +
                          std::to_string(kFewExpertFrames) +
                          " or more recommended); lower --max-context or --max-concurrency, or free VRAM");
    }
    return check;
}

} // namespace infernix::runtime
