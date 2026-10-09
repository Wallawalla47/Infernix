#include "models/qwen3_5/program/program_impl.h"
#include "models/qwen3_5/program/context_work.h"
#include "models/qwen3_5/execution/vision_overlay.h"
#include "core/device.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>

namespace infernix::models::qwen3_5::detail {

void ProgramImpl::preencode_overlay_vision(execution::VisionPrefillSession& session,
                                           const PreparedPromptData& prompt,
                                           const VisionPrefillPlan& plan, std::uint32_t base) {
    if (!parameters.model.overlay_vision() || !plan.control) { return; }
    // Overlay: the full tower is absent from VRAM. Items wholly inside the reused prefix are
    // never encoded again; the rest are encoded through the evictable window now, once, and the
    // session copies their embeddings in as prefill reaches them.
    // Indices are absolute prepared-item indices, as use.prepared_item_index is: the plan's
    // items start at control->prepared_item_begin inside the prepared prompt, and the session
    // expects prepared_item_begin + items.size() entries.
    const std::uint32_t item_end = plan.control->prepared_item_begin +
                                   static_cast<std::uint32_t>(plan.control->items.size());
    std::uint32_t first_needed = item_end;
    for (const auto& use : plan.uses) {
        if (use.end > base && use.prepared_item_index < first_needed) {
            first_needed = use.prepared_item_index;
        }
    }
    execution::VisionOverlayWindowStats window_stats;
    std::vector<execution::PinnedVisionResult> preencoded;
    if (first_needed < item_end) {
        preencoded = execution::encode_items_overlay(device, parameters, prompt, plan,
                                                     first_needed, &window_stats);
    } else {
        preencoded.resize(item_end);
    }
    session.set_preencoded(std::move(preencoded), window_stats);
}

void ProgramImpl::initialize_prefill(std::uint32_t lane, std::uint32_t base) {
    auto& request    = requests[lane];
    const auto& plan = *request.base;
    auto& staged     = request.prefill.emplace();
    staged.prompt    = *plan.prompt;
    staged.base = staged.cursor = base;
    staged.prompt_tokens        = plan.summary.prompt_tokens;
    staged.prepare_mtp          = speculative_backend == SpeculativeBackend::Mtp;
    staged.initial_mtp_extent   = initial_mtp_extent(plan);
    // The binding names its prefix-cache path when it reuses a prefix.
    staged.reuse                = PrefixReusePath::Root;
    staged.mtp_bridge           = !staged.prepare_mtp || !base   ? MtpBridgeMode::None
                                  : base == staged.prompt_tokens ? MtpBridgeMode::AfterExactHit
                                                                 : MtpBridgeMode::BeforeSuffix;
    if (plan.vision_control_plan) {
        auto& vision        = staged.vision_plan.emplace();
        vision.control_plan = plan.vision_control_plan;
        vision.control      = std::make_shared<VisionControl>(
            build_vision_control(staged.prompt, *vision.control_plan, 0));
        for (std::uint32_t i = 0; i < vision.control_plan->items.size(); ++i) {
            const auto& item = vision.control_plan->items[i];
            if (item.token_end <= base) { continue; }
            const auto begin =
                staged.prepare_mtp && item.token_begin ? item.token_begin - 1U : item.token_begin;
            vision.uses.push_back({begin, item.token_end, i, i});
            vision.max_merged_count = std::max(vision.max_merged_count, item.merged_count);
        }
        if (!vision.uses.empty()) {
            staged.vision = std::make_unique<execution::VisionPrefillSession>(
                device, parameters,
                DeviceSpan{workspace_storage.base(), workspace_storage.capacity()},
                *workspace_plan.vision, staged.prompt, vision, vision_handoff,
                vision_handoff_peak_bytes);
            preencode_overlay_vision(*staged.vision, staged.prompt, vision, base);
        }
    }
}

// Stages a binding's request state and unit demands before its capacity check. A failed check
// resets requests[lane].
void ProgramImpl::plan_binding_units(ContextTransaction& tx, const RequestBasePlan& base,
                                     ResumeState* resume, ExecutionUnitKind resume_kind,
                                     std::uint32_t resume_tokens) {
    const auto lane   = tx.lane;
    auto& state       = sequences[lane];
    auto& request     = requests[lane];
    state.lane        = lane;
    request.base      = base.impl_;
    request.lifecycle = Lifecycle::Prefilling;
    initialize_prefill(lane, tx.reuse_frontier);
    UnitDemand first;
    if (!resume) {
        first = prefill_unit(base.summary().prompt_tokens, tx.reuse_frontier,
                             initial_mtp_extent(*base.impl_));
    } else {
        first = {.kind          = ExecutionUnitKind::Replay,
                 .main_frontier = std::min(tx.reuse_frontier + prefill_chunk, resume->frontier())};
        first.backend_frontier = backend_kv_cache() ? first.main_frontier : 0;
    }
    tx.first_unit         = first;
    tx.reservation_demand = first;
    if (resume) {
        const auto& resumed_sequence = resume->impl_->sequence;
        auto coverage =
            next_unit(resumed_sequence, resume->impl_->control, resume_kind, resume_tokens);
        coverage.main_frontier = std::max(coverage.main_frontier, resume->frontier());
        if (backend_kv_cache()) {
            coverage.backend_frontier = std::max(coverage.backend_frontier, resume->frontier());
        }
        tx.recovery           = RecoveryPermit{.coverage      = coverage,
                                               .frontier      = resume->frontier(),
                                               .ledger_tokens = resumed_sequence.ledger.size()};
        tx.reservation_demand = coverage;
    }
}

BindingReservation ProgramImpl::start_binding(const RequestBasePlan& base, runtime::LaneId lane_id,
                                              const SourceCandidate& candidate, ResumeState* resume,
                                              ExecutionUnitKind resume_kind,
                                              std::uint32_t resume_tokens) {
    const auto lane = lane_id.value;
    if (context_transaction_ || lane >= max_concurrency ||
        requests[lane].lifecycle != Lifecycle::Empty || !base.impl_) {
        throw std::logic_error("binding requires a free lane and context transaction slot");
    }
    return start_hybrid_binding(base, lane, candidate, resume, resume_kind, resume_tokens);
}

void ProgramImpl::install_binding(ContextTransaction& tx) {
    auto& state        = sequences[tx.lane];
    auto& request      = requests[tx.lane];
    const auto binding = state.state;
    const auto kv      = state.kv;
    if (tx.resume) {
        auto& saved                   = *tx.resume->impl_;
        request                       = std::move(saved.control);
        state                         = std::move(saved.sequence);
        state.kv                      = kv;
        state.state                   = binding;
        state.lane                    = tx.lane;
        state.mtp_draft_count         = 0;
        request.resume_lifecycle      = request.lifecycle;
        request.lifecycle             = Lifecycle::Replaying;
        request.replay_target         = saved.frontier;
        request.replay_cursor         = tx.reuse_frontier;
        state.text_kv_valid           = tx.reuse_frontier;
        state.mtp_kv_valid            = tx.backend_frontier;
        state.dflash_context_frontier = tx.reuse_frontier;
        // Reconstruct references into the moved durable prefill object.
        if (request.prefill && request.prefill->vision_plan &&
            !request.prefill->vision_plan->uses.empty()) {
            request.prefill->vision = std::make_unique<execution::VisionPrefillSession>(
                device, parameters,
                DeviceSpan{workspace_storage.base(), workspace_storage.capacity()},
                *workspace_plan.vision, request.prefill->prompt, *request.prefill->vision_plan,
                vision_handoff, vision_handoff_peak_bytes);
            preencode_overlay_vision(*request.prefill->vision, request.prefill->prompt,
                                     *request.prefill->vision_plan, request.prefill->cursor);
        }
        install_resume_sampling(state, request);
    } else {
        state.ledger = tx.base->prompt->token_ids;
        state.ledger_splits.assign(*tx.base->prompt);
        state.rope_delta              = tx.base->prompt->rope_delta;
        state.text_kv_valid           = tx.reuse_frontier;
        state.mtp_kv_valid            = tx.backend_frontier;
        state.dflash_context_frontier = tx.reuse_frontier;
        state.tail_hidden_valid       = false;
        request.lifecycle             = Lifecycle::Prefilling;
        install_sampling(state, request, tx.base->sampling);
        // A paused request keeps its proposer in the saved control and resumes with it.
        if (ngram_draft_window != 0) { take_ngram_index(request, *tx.base); }
    }
    refresh_state_views(state);
    request.permit   = tx.first_unit;
    request.recovery = tx.recovery;
}

} // namespace infernix::models::qwen3_5::detail
