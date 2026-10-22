#pragma once

// Qwen3.8-Flash-Next instantiates the shared expert-offload component (src/runtime/expert_offload/;
// docs/maintainer/qwen3_5-122b-offload-design.md §7): its residency, host and SSD tiers, expert
// store, memory plan, VRAM monitor, route trace and prefill expert stream.

#include "models/qwen4_exp/config.h"
#include "runtime/expert_offload/expert_bank.h"
#include "runtime/expert_offload/expert_cache.h"
#include "runtime/expert_offload/expert_residency.h"
#include "runtime/expert_offload/expert_state.h"
#include "runtime/expert_offload/expert_offload.h"
#include "runtime/expert_offload/expert_store.h"
#include "runtime/expert_offload/expert_stream.h"
#include "runtime/expert_offload/host_expert_tier.h"
#include "runtime/expert_offload/host_tier.h"
#include "runtime/expert_offload/memory_plan.h"
#include "runtime/expert_offload/route_trace.h"
#include "runtime/expert_offload/vram_monitor.h"

namespace infernix::models::qwen4_exp {

namespace expert_cache = runtime::expert_offload::expert_cache;
using runtime::expert_offload::CpuSplitPolicy;
using runtime::expert_offload::display_headroom;
using runtime::expert_offload::ExpertBank;
using runtime::expert_offload::ExpertLayout;
using runtime::expert_offload::ExpertPlacement;
using runtime::expert_offload::ExpertResidency;
using runtime::expert_offload::ExpertStore;
using runtime::expert_offload::ExpertStream;
using runtime::expert_offload::ExpertStripe;
using runtime::expert_offload::ggml_format;
using runtime::expert_offload::graph_bound;
using runtime::expert_offload::HostExpertTier;
using runtime::expert_offload::HostMemoryDemand;
using runtime::expert_offload::HostMemoryLedger;
using runtime::expert_offload::internal_reserve;
using runtime::expert_offload::kGraphExecutableBytes;
using runtime::expert_offload::kLedgerGiB;
using runtime::expert_offload::kLedgerMiB;
using runtime::expert_offload::kPinRoomMarginBytes;
using runtime::expert_offload::kVramBudgetMarginBytes;
using runtime::expert_offload::pin_room;
using runtime::expert_offload::plan_host_memory;
using runtime::expert_offload::read_expert_stripe;
using runtime::expert_offload::RouteTrace;
using runtime::expert_offload::route_trace_index;
using runtime::expert_offload::RouteTraceGraph;
using runtime::expert_offload::RouteTraceKind;
using runtime::expert_offload::RouteTraceRound;
using runtime::expert_offload::RouteTraceSetup;
using runtime::expert_offload::size_expert_frames;
using runtime::expert_offload::VramControl;
using runtime::expert_offload::VramDemand;
using runtime::expert_offload::VramMonitor;
using runtime::expert_offload::VramSizing;
using runtime::expert_offload::with_stripe;

// The routed experts the residency caches: every layer's.
[[nodiscard]] inline ExpertLayout expert_layout(const TextConfig& config) noexcept {
    return {config.num_hidden_layers, config.moe.experts, config.moe.top_k};
}

} // namespace infernix::models::qwen4_exp
