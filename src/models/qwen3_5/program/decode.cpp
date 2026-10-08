#include "models/qwen3_5/program/program_impl.h"
#include "models/qwen3_5/program/context_work.h"
#include "models/qwen3_5/program/execution_context.h"
#include "models/qwen3_5/program/graph_execution.h"
#include "models/qwen3_5/program/planning/graph_profiles.h"
#include "core/nvtx.h"
#include "core/device.h"
#include "infernix/ops/prepare_ragged_prefix.h"
#include "infernix/ops/sampling.h"
#include "infernix/ops/scatter.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace infernix::models::qwen3_5::execution {
namespace {

auto ordinary_batch_body(OrdinaryBatchContext& state, std::int32_t batch_size,
                         ops::CausalAttentionExecutionEnvelope envelope) {
    return [&state, batch_size, envelope] {
        if (batch_size <= 0 || batch_size > static_cast<std::int32_t>(kMaximumConcurrency)) {
            throw std::logic_error("ordinary decode batch state is incomplete");
        }

        qwen3_5::OrdinaryDecodeState& ordinary = state.frame;
        CUDA_CHECK(cudaMemcpyAsync(ordinary.ingress.data, &state.host_ingress,
                                   sizeof(qwen3_5::OrdinaryDecodeIngress), cudaMemcpyHostToDevice,
                                   state.execution.device.stream));

        TextContext card(state.execution.device, state.execution.parameters, state.execution.work,
                         {}, state.execution.state_images, state.execution.io,
                         state.execution.prefill_hidden, state.execution.prefill_chunk, 0, {},
                         &state.text_cache);

        Tensor tokens             = ordinary.tokens.slice(0, 0, batch_size);
        Tensor cache_positions    = ordinary.cache_positions.slice(0, 0, batch_size);
        Tensor rope_positions     = ordinary.rope_positions.slice(0, 0, batch_size);
        Tensor kv_rows            = ordinary.text_kv_table_rows.slice(0, 0, batch_size);
        Tensor state_sources      = ordinary.state_source_slots.slice(0, 0, batch_size);
        Tensor state_destinations = ordinary.state_destination_slots.slice(0, 0, batch_size);
        Tensor hidden             = ordinary.hidden.slice(1, 0, batch_size);
        Tensor logits             = ordinary.logits.slice(1, 0, batch_size);
        Tensor sampled            = ordinary.sampled_tokens.slice(0, 0, batch_size);

        card.ordinary_decode_batch(tokens, cache_positions, rope_positions, kv_rows, state_sources,
                                   state_destinations, envelope, hidden, logits);
        ops::scatter(hidden, state_destinations, state.continuation_hidden_store,
                     state.execution.device.stream);
        qwen3_5::constrain_round_logits(
            state.execution.io, logits, nullptr,
            dimension(state.execution.parameters.model.resources().public_token_count),
            state.execution.device.stream);
        ops::sample(logits, sampled,
                    dimension(state.execution.parameters.model.resources().public_token_count),
                    ordinary.sampling, cache_positions, ops::kSamplePurposeDecode,
                    state.execution.work, state.execution.device.stream);
        CUDA_CHECK(cudaMemcpyAsync(&state.host_egress, ordinary.egress.data,
                                   sizeof(qwen3_5::OrdinaryDecodeEgress), cudaMemcpyDeviceToHost,
                                   state.execution.device.stream));
    };
}

} // namespace

void capture_ordinary_decode_batch(OrdinaryBatchContext& state, std::int32_t batch_size,
                                   ops::CausalAttentionExecutionEnvelope envelope,
                                   DecodeGraphDefinition& definition) {
    auto body = ordinary_batch_body(state, batch_size, envelope);
    capture_graph(state, definition, body);
}

void ordinary_decode_batch(OrdinaryBatchContext& state, std::int32_t batch_size,
                           ops::CausalAttentionExecutionEnvelope envelope,
                           DecodeGraphExecutable* executable) {
    auto body = ordinary_batch_body(state, batch_size, envelope);
    run_prepared(state, executable, body);
}

} // namespace infernix::models::qwen3_5::execution

namespace infernix::models::qwen3_5::detail {

namespace {} // namespace

void ProgramImpl::install_sampling(SequenceState& sequence, RequestControl& request,
                                   const ops::SamplingConfig& config) {
    Tensor counts = token_counts.slice(1, static_cast<std::int32_t>(sequence.lane), 1)
                        .view({dimension(parameters.model.resources().public_token_count)});
    request.sampling_host     = config;
    request.speculative_stats = SpeculativeStats{
        .backend               = speculative_backend,
        .enabled               = speculative_backend != SpeculativeBackend::None,
        .draft_window          = draft_window,
        .accepted_per_position = std::vector<std::uint64_t>(draft_window, 0),
    };
    const bool penalties = request.sampling_host.presence_penalty != 0.0F ||
                           request.sampling_host.frequency_penalty != 0.0F;
    if (penalties) { CUDA_CHECK(cudaMemsetAsync(counts.data, 0, counts.bytes(), device.stream)); }
    request.sampling_host.token_counts =
        penalties ? static_cast<std::int32_t*>(counts.data) : nullptr;
    Tensor config_lane = sampling_config.slice(1, static_cast<std::int32_t>(sequence.lane), 1);
    CUDA_CHECK(cudaMemcpyAsync(config_lane.data, &request.sampling_host,
                               sizeof(request.sampling_host), cudaMemcpyHostToDevice,
                               device.stream));
}

void ProgramImpl::copy_tail(SequenceState& sequence, const Tensor& source) {
    if (source.dtype != DType::BF16 ||
        source.ne[0] != dimension(parameters.model.config().text.hidden_size) ||
        source.ne[1] != 1) {
        throw std::logic_error("target tail hidden has an invalid shape");
    }
    CUDA_CHECK(cudaMemcpyAsync(sequence.tail_hidden.data, source.data, sequence.tail_hidden.bytes(),
                               cudaMemcpyDeviceToDevice, device.stream));
    sequence.tail_hidden_valid = true;
}

void ProgramImpl::copy_round_token() {
    CUDA_CHECK(cudaMemcpyAsync(host_tokens, io.token.data, sizeof(TokenId), cudaMemcpyDeviceToHost,
                               device.stream));
}

void ProgramImpl::mark_workspace_usage(std::size_t phase_bytes) noexcept {
    workspace_logical_peak_bytes = std::max(workspace_logical_peak_bytes, phase_bytes);
}

const GdnReplayRecords* ProgramImpl::round_replay_records(std::uint32_t verify_drafts) const {
    if (!replay_records) { return nullptr; }
    if (verify_drafts == draft_window) { return &*replay_records; }
    for (const NarrowReplayView& view : narrow_replay_views) {
        if (view.records.spec.width == static_cast<std::int32_t>(verify_drafts + 1U)) {
            return &view.records;
        }
    }
    throw std::logic_error("speculative round width has no ReplaySSM record view");
}

const ops::GdnReplayFoldPlan& ProgramImpl::round_replay_fold(std::uint32_t verify_drafts) const {
    if (!replay_fold) { throw std::logic_error("speculative round has no ReplaySSM fold"); }
    if (verify_drafts == draft_window) { return *replay_fold; }
    for (const NarrowReplayView& view : narrow_replay_views) {
        if (view.records.spec.width == static_cast<std::int32_t>(verify_drafts + 1U)) {
            return view.fold;
        }
    }
    throw std::logic_error("speculative round width has no ReplaySSM fold");
}

SpeculativeRoundFamily& ProgramImpl::round_family(SpeculativeRoundKind kind, std::uint32_t drafts) {
    SpeculativeRoundFamily* best = nullptr;
    for (SpeculativeRoundFamily& family : round_families) {
        if (family.shape.kind != kind || family.shape.verify_drafts < drafts) { continue; }
        if (best == nullptr || family.shape.verify_drafts < best->shape.verify_drafts) {
            best = &family;
        }
    }
    if (best == nullptr) {
        throw std::logic_error("speculative round has no family of the requested width");
    }
    return *best;
}

void ProgramImpl::enqueue_dflash_context_append(std::span<const std::uint32_t> lanes,
                                                std::span<const std::uint32_t> starts,
                                                std::span<const std::uint32_t> counts) {
    if (!is_masked_draft_backend(speculative_backend) || !dflash || !io.dflash_decode ||
        lanes.empty() || lanes.size() > max_concurrency || starts.size() != lanes.size() ||
        counts.size() != lanes.size()) {
        throw std::logic_error("DFlash context append has invalid membership");
    }

    std::uint32_t minimum_count = draft_window + 1U;
    std::uint32_t maximum_count = 0;
    *dflash_host_ingress        = {};
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        const std::uint32_t lane = lanes[row];
        if (lane >= max_concurrency || counts[row] == 0 || counts[row] > draft_window + 1U ||
            std::find(lanes.begin(), lanes.begin() + static_cast<std::ptrdiff_t>(row), lane) !=
                lanes.begin() + static_cast<std::ptrdiff_t>(row)) {
            throw std::logic_error("DFlash context append contains an invalid row");
        }
        SequenceState& sequence   = active_sequence(lane);
        const std::uint32_t start = starts[row];
        const std::uint64_t end64 = static_cast<std::uint64_t>(start) + counts[row];
        const std::uint32_t end   = static_cast<std::uint32_t>(end64);
        if (!sequence.kv || text_kv_addresses->bound_row(sequence.kv->text) < 0 ||
            (backend_kv_cache() && (!sequence.kv->backend ||
                                    backend_kv_addresses->bound_row(*sequence.kv->backend) < 0)) ||
            end64 > capacity) {
            throw std::logic_error("DFlash context append is outside retained target storage");
        }
        dflash_host_ingress->context_frontiers[row] =
            checked_i32(start, "DFlash append context frontier");
        dflash_host_ingress->execution_frontiers[row] =
            checked_i32(end, "DFlash append target frontier");
        dflash_host_ingress->dflash_kv_table_rows[row] =
            sequence.kv->backend ? backend_kv_addresses->bound_row(*sequence.kv->backend) : 0;
        dflash_host_ingress->active_lanes[row]            = static_cast<std::int32_t>(lane);
        const StateImageSelectors selectors               = state_selectors(sequence);
        dflash_host_ingress->state_source_slots[row]      = selectors.source;
        dflash_host_ingress->state_destination_slots[row] = selectors.destination;
        // Context append writes only the draft caches. Target execution owns Main KV
        // coverage, including any uncommitted suffix that survives until the round is settled.
        // DFlash2 has only fixed cyclic state; DFlash also grows its Full backend KV here.
        if (sequence.kv->backend) {
            backend_kv_addresses->ensure_mapped_to_tokens(*sequence.kv->backend, end,
                                                          device.stream);
        }
        minimum_count = std::min(minimum_count, counts[row]);
        maximum_count = std::max(maximum_count, counts[row]);
    }

    qwen3_5::DFlashDecodeState& frame = *io.dflash_decode;
    CUDA_CHECK(cudaMemcpyAsync(frame.ingress.data, dflash_host_ingress,
                               sizeof(qwen3_5::DFlashDecodeIngress), cudaMemcpyHostToDevice,
                               device.stream));
    const auto batch                = static_cast<std::int32_t>(lanes.size());
    Tensor active_lane_tensor       = frame.active_lanes.slice(0, 0, batch);
    Tensor state_destination_tensor = frame.state_destination_slots.slice(0, 0, batch);
    Tensor device_starts            = frame.context_frontiers.slice(0, 0, batch);
    Tensor device_ends              = frame.execution_frontiers.slice(0, 0, batch);
    Tensor table_rows               = frame.dflash_kv_table_rows.slice(0, 0, batch);
    Tensor positions                = frame.append_positions.slice(1, 0, batch);
    Tensor device_counts            = frame.append_counts.slice(0, 0, batch);

    work.reset();
    Tensor features =
        work.alloc(DType::BF16, {dimension(parameters.draft->feature_projection.weight.k),
                                 static_cast<std::int32_t>(draft_window + 1U), batch});
    ops::prepare_ragged_prefix(dflash->pending_features, active_lane_tensor, device_starts,
                               device_ends, features, positions, device_counts, device.stream);

    execution::DFlashAppendContext state{{device, parameters, work, *state_images,
                                          replay_records ? &*replay_records : nullptr, io,
                                          prefill_hidden, prefill_chunk, proposal_head,
                                          prompt_attention},
                                         *dflash};
    mark_workspace_usage(workspace_plan.dflash_context);
    execution::dflash_append_context(state, features, positions, device_counts,
                                     state_destination_tensor, table_rows,
                                     {minimum_count, maximum_count});
}

void ProgramImpl::validate_licensed_tokens(std::span<const TokenId> tokens) const {
    for (const TokenId token : tokens) {
        if (token < 0 || token >= dimension(parameters.model.resources().public_token_count)) {
            throw std::runtime_error("target returned a token outside the public token domain");
        }
    }
}

runtime::BatchedGeneratedRound ProgramImpl::decode_ordinary_batch(
    std::span<const std::uint32_t> lanes, std::span<const runtime::RoundBudget> budgets,
    runtime::ExecutionTiming* failed_timing, runtime::TokenMaskProvider* masks) {
    nvtx::ScopedRange round_range(nvtx::Name::DecodeOrdinaryRound, nvtx::Category::Decode,
                                  static_cast<std::uint64_t>(lanes.size()));
    runtime::ExecutionTimingRecorder timing(runtime::ExecutionTimingPhase::Submit, failed_timing);
    if (speculative_backend != SpeculativeBackend::None) {
        throw std::logic_error("ordinary batch execution requires the ordinary backend");
    }
    if (lanes.empty() || lanes.size() > max_concurrency || budgets.size() != lanes.size()) {
        throw std::invalid_argument("ordinary batch membership is invalid");
    }

    std::uint32_t maximum_frontier = 0;
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        const std::uint32_t lane = lanes[row];
        if (lane >= max_concurrency ||
            std::find(lanes.begin(), lanes.begin() + static_cast<std::ptrdiff_t>(row), lane) !=
                lanes.begin() + static_cast<std::ptrdiff_t>(row)) {
            throw std::invalid_argument("ordinary batch contains an invalid or duplicate lane");
        }
        const SequenceState& sequence = active_sequence(lane);
        const RequestControl& request = requests[lane];
        if (request.lifecycle != Lifecycle::Active ||
            budgets[row].generated_tokens_remaining == 0 || !sequence.kv ||
            text_kv_addresses->bound_row(sequence.kv->text) < 0 ||
            sequence.execution_frontier >= capacity ||
            sequence.ledger_frontier != sequence.execution_frontier + 1 ||
            sequence.ledger.size() != sequence.ledger_frontier ||
            sequence.prefix_identity.size() != sequence.ledger_frontier ||
            sequence.prefix_digests.size() != sequence.ledger_frontier) {
            throw std::logic_error("ordinary batch row is not decode-ready");
        }
        maximum_frontier = std::max(maximum_frontier, sequence.execution_frontier);
    }

    const auto start = Clock::now();
    try {
        std::optional<nvtx::ScopedRange> submit_range;
        submit_range.emplace(nvtx::Name::DecodeOrdinarySubmit, nvtx::Category::Decode,
                             static_cast<std::uint64_t>(lanes.size()));
        DecodeGraphExecutable* executable = nullptr;
        ops::CausalAttentionExecutionEnvelope envelope{maximum_frontier + 1, maximum_frontier + 1};
        if (use_cuda_graph) {
            DecodeGraphProfile& profile =
                ordinary_graphs.select(static_cast<std::uint32_t>(lanes.size()), maximum_frontier);
            executable = &ordinary_graphs.install(profile);
            envelope   = {profile.min_execution_frontier + 1, profile.max_execution_frontier + 1};
        }

        for (std::size_t row = 0; row < lanes.size(); ++row) {
            SequenceState& sequence            = active_sequence(lanes[row]);
            const RequestControl& request      = requests[lanes[row]];
            const std::uint32_t frontier       = sequence.execution_frontier;
            ordinary_host_ingress->tokens[row] = sequence.ledger.back();
            ordinary_host_ingress->cache_positions[row] =
                checked_i32(frontier, "ordinary batch position");
            ordinary_host_ingress->rope_positions[row] =
                checked_i32(frontier, "ordinary batch RoPE position") + sequence.rope_delta;
            ordinary_host_ingress->text_kv_table_rows[row] =
                text_kv_addresses->bound_row(sequence.kv->text);
            const StateImageSelectors selectors                 = state_selectors(sequence);
            ordinary_host_ingress->state_source_slots[row]      = selectors.source;
            ordinary_host_ingress->state_destination_slots[row] = selectors.destination;
            ordinary_host_ingress->sampling[row]                = request.sampling_host;
            ordinary_host_ingress->sampling[row].mask           = fill_grammar_mask(masks, row, {});
            ensure_sequence_kv_mapped(sequence, frontier + 1, 0);
        }

        execution::OrdinaryBatchContext schedule_state{
            {device, parameters, work, *state_images,
             replay_records ? &*replay_records : nullptr, io, prefill_hidden, prefill_chunk,
             proposal_head, prompt_attention},
            decoder->text_kv,
            *io.ordinary,
            *ordinary_host_ingress,
            *ordinary_host_egress,
            state_images->continuation_hidden_store()};

        std::array<ConstraintRow, kMaximumConcurrency> constraint_rows{};
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            constraint_rows[row] = constraint_row(lanes[row]);
        }
        stage_constraint_columns(
            std::span<const ConstraintRow>(constraint_rows.data(), lanes.size()), 1);

        mark_workspace_usage(workspace_plan.ordinary_round);
        execution::ordinary_decode_batch(schedule_state, static_cast<std::int32_t>(lanes.size()),
                                         envelope, executable);
        submit_range.reset();
        timing.begin_wait();
        {
            nvtx::ScopedRange wait_range(nvtx::Name::DecodeOrdinaryWait, nvtx::Category::Control,
                                         static_cast<std::uint64_t>(lanes.size()));
            device.synchronize();
        }
        timing.end_wait();

        const double seconds = std::chrono::duration<double>(Clock::now() - start).count();
        const double share   = seconds / static_cast<double>(lanes.size());
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            SequenceState& sequence    = active_sequence(lanes[row]);
            RequestControl& request    = requests[lanes[row]];
            const std::uint32_t base_E = sequence.execution_frontier;
            const std::uint32_t base_S = sequence.ledger_frontier;
            const TokenId token        = ordinary_host_egress->sampled_tokens[row];
            validate_licensed_tokens(std::span<const TokenId>(&token, 1));
            if (const ConstraintRow& constrained = constraint_rows[row];
                constrained.plan != nullptr) {
                request.constraint_round =
                    read_constraint_draws(*constrained.plan, constrained.first_step, row,
                                          std::span<const TokenId>(&token, 1));
            }
            sequence.text_kv_valid = base_E + 1;
            commit_sequence_kv(sequence, sequence.text_kv_valid, 0);
            sequence.tail_hidden_valid = true;
            sequence.ledger.push_back(token);
            sequence.prefix_identity.append_generated(1, sequence.rope_delta);
            sequence.prefix_digests.append_generated(std::span<const TokenId>(&token, 1),
                                                     sequence.rope_delta);
            request.pending   = PendingCandidate{.kind          = PendingKind::Ordinary,
                                                 .base_E        = base_E,
                                                 .base_S        = base_S,
                                                 .prompt_tokens = 0,
                                                 .produced      = 1};
            request.lifecycle = Lifecycle::Pending;
            request.timings.decode_seconds += seconds;
            request.timings.decode_share_seconds += share;
        }
        return runtime::BatchedGeneratedRound{
            .tokens =
                std::span<const TokenId>(ordinary_host_egress->sampled_tokens.data(), lanes.size()),
            .timing = timing.finish(),
        };
    } catch (...) {
        timing.begin_wait();
        try {
            nvtx::ScopedRange wait_range(nvtx::Name::DecodeOrdinaryWait, nvtx::Category::Control,
                                         static_cast<std::uint64_t>(lanes.size()));
            device.synchronize();
        } catch (...) {}
        timing.end_wait();
        clear_execution_failure_lanes(lanes);
        throw;
    }
}

std::vector<NgramProposer::Match>
ProgramImpl::propose_ngram(std::span<const std::uint32_t> lanes,
                           std::span<const runtime::RoundBudget> budgets) {
    std::vector<NgramProposer::Match> matches(lanes.size());
    if (ngram_draft_window == 0) { return matches; }
    if (budgets.size() != lanes.size()) {
        throw std::invalid_argument("ngram budget count does not match lane count");
    }
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        if (lanes[row] >= max_concurrency) {
            throw std::invalid_argument("ngram lane is out of range");
        }
        matches[row] = propose_ngram_one(lanes[row], budgets[row]);
    }
    return matches;
}

void ProgramImpl::take_ngram_index(RequestControl& request, const RequestBasePlanImpl& base) {
    if (!base.ngram_index || !base.ngram_index->index) {
        throw std::logic_error("request plan carries no ngram index while ngram drafting is on");
    }
    request.ngram          = std::move(base.ngram_index->index);
    request.ngram_indexed  = base.prompt->token_ids.size();
    request.ngram_snapshot = base.prompt->ngram_snapshot;
}

NgramProposer::Match ProgramImpl::propose_ngram_one(std::uint32_t lane,
                                                    const runtime::RoundBudget& budget) {
    auto& request        = requests[lane];
    const auto& sequence = active_sequence(lane);
    const auto& ledger   = sequence.ledger;
    const auto room =
        sequence.execution_frontier < capacity ? capacity - sequence.execution_frontier - 1U : 0U;
    // The output budget does not size the copy: the round's kind and width follow from the copy
    // alone, and the round verifies only the drafts the budget admits. A budget-limited final
    // round therefore has the arithmetic of an unlimited one.
    const auto maximum = std::min(ngram_draft_window, room);
    // The neural path still validates decode readiness. No draft fits an anchor-only tail.
    if (maximum == 0 || budget.generated_tokens_remaining <= 1) { return {}; }
    if (!request.ngram || request.ngram_indexed > ledger.size()) {
        throw std::logic_error("ngram committed history is not initialized");
    }
    while (request.ngram_indexed < ledger.size()) {
        request.ngram->append(ledger[request.ngram_indexed++]);
    }
    const auto history =
        std::span<const TokenId>(ledger).last(std::min(kNgramMatchHistoryTokens, ledger.size()));
    NgramProposer::Match match;
    if (request.ngram_snapshot && request.ngram_copy_source != 0 &&
        ledger.size() >= request.ngram_copy_ledger) {
        const auto offset = request.ngram_copy_offset + ledger.size() - request.ngram_copy_ledger;
        if (offset <= std::numeric_limits<std::uint32_t>::max()) {
            match = request.ngram_snapshot->continue_copy(request.ngram_copy_source,
                                                          static_cast<std::uint32_t>(offset),
                                                          history, maximum + 2U, ngram_min_match);
        }
    }
    if (match.tokens.size() < maximum + 2U) {
        match = request.ngram->propose(history, maximum + 2U, ngram_min_match);
        // Equal candidates keep the live source; a maximal match cannot be improved.
        if (request.ngram_snapshot &&
            !NgramProposer::maximal(match, history.size(), maximum + 2U)) {
            auto retained = request.ngram_snapshot->propose(history, maximum + 2U, ngram_min_match);
            if (retained.matched > match.matched ||
                (retained.matched == match.matched &&
                 retained.tokens.size() > match.tokens.size())) {
                match = std::move(retained);
            }
        }
    }
    match = NgramProposer::finish_round(std::move(match), maximum, neural_draft_window);
    // A cursor is only a candidate. Next round rechecks it against committed ledger
    // tokens, including correction/bonus/control tokens, before skipping lookup.
    request.ngram_copy_source = !match.tokens.empty() && match.archived ? match.source : 0;
    request.ngram_copy_offset = match.offset;
    request.ngram_copy_ledger = ledger.size();
    return match;
}

runtime::BatchedGeneratedRound ProgramImpl::decode_mtp_batch(
    std::span<const std::uint32_t> lanes, std::span<const runtime::RoundBudget> budgets,
    runtime::ExecutionTiming* failed_timing, runtime::TokenMaskProvider* masks) {
    nvtx::ScopedRange round_range(nvtx::Name::DecodeMtpRound, nvtx::Category::Mtp,
                                  static_cast<std::uint64_t>(lanes.size()));
    runtime::ExecutionTimingRecorder timing(runtime::ExecutionTimingPhase::Submit, failed_timing);
    if (speculative_backend != SpeculativeBackend::Mtp || !io.mtp_decode ||
        decoder->mtp_cache() == nullptr) {
        throw std::logic_error("MTP batch execution requires the MTP backend");
    }
    if (lanes.empty() || lanes.size() > max_concurrency || budgets.size() != lanes.size()) {
        throw std::invalid_argument("MTP batch membership is invalid");
    }

    const auto started   = Clock::now();
    const auto matches   = propose_ngram(lanes, budgets);
    const bool any_ngram = std::any_of(matches.begin(), matches.end(),
                                       [](const auto& match) { return !match.tokens.empty(); });
    std::uint32_t longest_copy = 0;
    for (const auto& match : matches) {
        longest_copy = std::max(longest_copy, static_cast<std::uint32_t>(match.tokens.size()));
    }
    // As for DFlash, a round with at least one copy proposal verifies at the narrowest n-gram
    // window that holds its longest copy and an all-neural round at the MTP window, each on the
    // frame viewed at that width. Every round proposes the next round's drafts at the configured
    // MTP depth.
    SpeculativeRoundFamily& family =
        any_ngram ? round_family(SpeculativeRoundKind::Ngram, longest_copy)
                  : round_family(SpeculativeRoundKind::Neural, neural_draft_window);
    const std::uint32_t verify_drafts = family.shape.verify_drafts;
    const std::uint32_t mtp_ar_depth  = neural_draft_window;
    const std::uint32_t width         = verify_drafts + 1;
    std::uint32_t maximum_frontier    = 0;
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        const std::uint32_t lane = lanes[row];
        if (lane >= max_concurrency ||
            std::find(lanes.begin(), lanes.begin() + static_cast<std::ptrdiff_t>(row), lane) !=
                lanes.begin() + static_cast<std::ptrdiff_t>(row)) {
            throw std::invalid_argument("MTP batch contains an invalid or duplicate lane");
        }
        const SequenceState& sequence = active_sequence(lane);
        const RequestControl& request = requests[lane];
        if (request.lifecycle != Lifecycle::Active ||
            budgets[row].generated_tokens_remaining == 0 || !sequence.kv || !sequence.kv->backend ||
            text_kv_addresses->bound_row(sequence.kv->text) < 0 ||
            backend_kv_addresses->bound_row(*sequence.kv->backend) < 0 ||
            sequence.execution_frontier >= capacity ||
            sequence.mtp_kv_valid != sequence.execution_frontier ||
            sequence.ledger_frontier != sequence.execution_frontier + 1 ||
            sequence.ledger.size() != sequence.ledger_frontier ||
            sequence.prefix_identity.size() != sequence.ledger_frontier ||
            sequence.prefix_digests.size() != sequence.ledger_frontier ||
            sequence.mtp_draft_count > mtp_ar_depth) {
            throw std::logic_error("MTP batch row is not decode-ready");
        }
        maximum_frontier = std::max(maximum_frontier, sequence.execution_frontier);
    }

    try {
        std::optional<nvtx::ScopedRange> submit_range;
        submit_range.emplace(nvtx::Name::DecodeMtpSubmit, nvtx::Category::Mtp,
                             static_cast<std::uint64_t>(lanes.size()));
        DecodeGraphExecutable* forward                   = nullptr;
        DecodeGraphExecutable* finish                    = nullptr;
        execution::MtpCausalAttentionEnvelopes envelopes = mtp_causal_attention_envelopes(
            maximum_frontier, verify_drafts, capacity, mtp_ar_depth);
        if (use_cuda_graph) {
            const auto batch = static_cast<std::uint32_t>(lanes.size());
            auto& profile    = family.forward.select(batch, maximum_frontier);
            forward          = &family.forward.install(profile);
            finish = &family.finish.install(family.finish.select(batch, maximum_frontier));
            envelopes = mtp_causal_attention_envelopes(profile.max_execution_frontier, verify_drafts,
                                                       capacity, mtp_ar_depth);
        }

        for (std::size_t row = 0; row < lanes.size(); ++row) {
            SequenceState& sequence           = active_sequence(lanes[row]);
            const RequestControl& request     = requests[lanes[row]];
            const std::uint32_t frontier      = sequence.execution_frontier;
            const std::uint32_t max_by_budget = budgets[row].generated_tokens_remaining > 1
                                                    ? budgets[row].generated_tokens_remaining - 1
                                                    : 0;
            const bool row_ngram              = !matches[row].tokens.empty();
            const std::uint32_t extent        = std::min(
                {row_ngram ? static_cast<std::uint32_t>(matches[row].tokens.size())
                           : sequence.mtp_draft_count,
                 verify_drafts, max_by_budget, capacity - sequence.execution_frontier - 1});
            mtp_host_ingress->anchors[row]        = sequence.ledger.back();
            mtp_host_ingress->base_frontiers[row] = checked_i32(frontier, "MTP batch frontier");
            mtp_host_ingress->remaining_budgets[row] =
                checked_i32(budgets[row].generated_tokens_remaining, "MTP batch remaining budget");
            mtp_host_ingress->current_extents[row]      = static_cast<std::int32_t>(extent);
            mtp_host_ingress->target_valid_columns[row] = static_cast<std::int32_t>(extent + 1);
            for (std::uint32_t j = 0; j < verify_drafts; ++j) {
                mtp_host_ingress->current_drafts[row * verify_drafts + j] =
                    j < extent ? (row_ngram ? matches[row].tokens[j] : sequence.mtp_drafts[j])
                               : sequence.ledger.back();
            }
            for (std::uint32_t j = 0; j < width; ++j) {
                const std::uint32_t position = frontier + std::min(j, extent);
                mtp_host_ingress->target_rope_positions[row * width + j] =
                    checked_i32(position, "MTP batch RoPE position") + sequence.rope_delta;
            }
            mtp_host_ingress->text_kv_table_rows[row] =
                text_kv_addresses->bound_row(sequence.kv->text);
            mtp_host_ingress->mtp_kv_table_rows[row] =
                backend_kv_addresses->bound_row(*sequence.kv->backend);
            const StateImageSelectors selectors            = state_selectors(sequence);
            mtp_host_ingress->state_source_slots[row]      = selectors.source;
            mtp_host_ingress->state_destination_slots[row] = selectors.destination;
            mtp_host_ingress->rope_deltas[row]             = sequence.rope_delta;
            mtp_host_ingress->sampling[row]                = request.sampling_host;
            mtp_host_ingress->sampling[row].mask           = bind_grammar_mask(masks, row);
            ensure_sequence_kv_mapped(sequence, frontier + extent + 1,
                                      std::min(capacity, frontier + extent + mtp_ar_depth));
        }

        std::array<ConstraintRow, kMaximumConcurrency> constraint_rows{};
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            constraint_rows[row] = constraint_row(lanes[row]);
        }
        stage_constraint_columns(
            std::span<const ConstraintRow>(constraint_rows.data(), lanes.size()), width);

        execution::MtpBatchContext schedule_state{
            {device, parameters, work, *state_images, round_replay_records(verify_drafts),
             io, prefill_hidden, prefill_chunk, proposal_head, prompt_attention},
            decoder->text_kv,
            *decoder->mtp_cache(),
            *io.mtp_decode,
            *mtp_host_ingress,
            *mtp_host_egress,
            state_images->continuation_hidden_store()};
        schedule_state.neural_proposal_drafts = mtp_ar_depth;
        mark_workspace_usage(workspace_plan.mtp_round);
        const auto batch = static_cast<std::int32_t>(lanes.size());
        execution::mtp_decode_batch(schedule_state, batch, verify_drafts, envelopes, forward,
                                    execution::SpeculativePhase::Forward);
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            // The verified drafts are this round's ingress: MTP's own or an ngram copy.
            const auto extent = static_cast<std::size_t>(mtp_host_ingress->current_extents[row]);
            (void)fill_grammar_mask(
                masks, row, {&mtp_host_ingress->current_drafts[row * verify_drafts], extent});
        }
        execution::mtp_decode_batch(schedule_state, batch, verify_drafts, envelopes, finish,
                                    execution::SpeculativePhase::Finish);
        submit_range.reset();
        timing.begin_wait();
        {
            nvtx::ScopedRange wait_range(nvtx::Name::DecodeMtpWait, nvtx::Category::Control,
                                         static_cast<std::uint64_t>(lanes.size()));
            device.synchronize();
        }
        timing.end_wait();

        const double seconds = std::chrono::duration<double>(Clock::now() - started).count();
        const double share   = seconds / static_cast<double>(lanes.size());
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            SequenceState& sequence       = active_sequence(lanes[row]);
            RequestControl& request       = requests[lanes[row]];
            const std::uint32_t base_E    = sequence.execution_frontier;
            const std::uint32_t base_S    = sequence.ledger_frontier;
            const std::int32_t count_i    = mtp_host_egress->licensed_counts[row];
            const std::int32_t accepted_i = mtp_host_egress->accepted_drafts[row];
            const std::int32_t next_i     = mtp_host_egress->next_extents[row];
            if (count_i <= 0 || count_i > static_cast<std::int32_t>(width) || accepted_i < 0 ||
                accepted_i + 1 != count_i || next_i < 0 ||
                next_i > static_cast<std::int32_t>(mtp_ar_depth) ||
                static_cast<std::uint32_t>(count_i) > budgets[row].generated_tokens_remaining ||
                static_cast<std::uint64_t>(base_E) + static_cast<std::uint32_t>(count_i) >
                    capacity) {
                throw std::runtime_error("MTP batch returned invalid row metadata");
            }
            const std::span<const TokenId> row_tokens(mtp_host_egress->licensed_tokens.data() +
                                                          row * width,
                                                      static_cast<std::size_t>(count_i));
            validate_licensed_tokens(row_tokens);
            if (const ConstraintRow& constrained = constraint_rows[row];
                constrained.plan != nullptr) {
                request.constraint_round = read_constraint_draws(
                    *constrained.plan, constrained.first_step, row * width, row_tokens);
            }
            const std::uint32_t pcur =
                static_cast<std::uint32_t>(mtp_host_ingress->current_extents[row]);
            if (pcur == 0) {
                request.speculative_stats.fallback_steps += 1;
            } else {
                request.speculative_stats.rounds += 1;
                request.speculative_stats.drafted_tokens += pcur;
                request.speculative_stats.accepted_tokens += static_cast<std::uint32_t>(accepted_i);
                for (std::int32_t i = 0; i < accepted_i; ++i) {
                    request.speculative_stats.accepted_per_position[static_cast<std::size_t>(i)] +=
                        1;
                }
            }
            if (!matches[row].tokens.empty()) {
                request.speculative_stats.ngram_rounds += 1;
                request.speculative_stats.ngram_drafted_tokens += pcur;
                request.speculative_stats.ngram_accepted_tokens += accepted_i;
                if (matches[row].archived) {
                    request.speculative_stats.ngram_archive_rounds += 1;
                    request.speculative_stats.ngram_archive_drafted_tokens += pcur;
                    request.speculative_stats.ngram_archive_accepted_tokens += accepted_i;
                }
            }
            request.pending = PendingCandidate{
                .kind          = PendingKind::Speculative,
                .base_E        = base_E,
                .base_S        = base_S,
                .prompt_tokens = 0,
                .produced      = static_cast<std::uint32_t>(count_i),
                .verify_drafts = verify_drafts,
            };
            request.lifecycle = Lifecycle::Pending;
            request.timings.decode_seconds += seconds;
            request.timings.decode_share_seconds += share;
        }
        return runtime::BatchedGeneratedRound{
            .tokens     = std::span<const TokenId>(mtp_host_egress->licensed_tokens.data(),
                                                   lanes.size() * width),
            .row_counts = std::span<const std::int32_t>(mtp_host_egress->licensed_counts.data(),
                                                        lanes.size()),
            .row_stride = width,
            .timing     = timing.finish(),
        };
    } catch (...) {
        timing.begin_wait();
        try {
            nvtx::ScopedRange wait_range(nvtx::Name::DecodeMtpWait, nvtx::Category::Control,
                                         static_cast<std::uint64_t>(lanes.size()));
            device.synchronize();
        } catch (...) {}
        timing.end_wait();
        clear_execution_failure_lanes(lanes);
        throw;
    }
}

runtime::BatchedGeneratedRound ProgramImpl::decode_dflash_batch(
    std::span<const std::uint32_t> lanes, std::span<const runtime::RoundBudget> budgets,
    runtime::ExecutionTiming* failed_timing, runtime::TokenMaskProvider* masks) {
    nvtx::ScopedRange round_range(nvtx::Name::DecodeDFlashRound, nvtx::Category::DFlash,
                                  static_cast<std::uint64_t>(lanes.size()));
    runtime::ExecutionTimingRecorder timing(runtime::ExecutionTimingPhase::Submit, failed_timing);
    if (!is_masked_draft_backend(speculative_backend) || !io.dflash_decode || !dflash) {
        throw std::logic_error("DFlash batch execution requires the DFlash backend");
    }
    if (lanes.empty() || lanes.size() > max_concurrency || budgets.size() != lanes.size()) {
        throw std::invalid_argument("DFlash batch membership is invalid");
    }

    const auto started         = Clock::now();
    const auto matches         = propose_ngram(lanes, budgets);
    bool any_ngram             = false;
    std::uint32_t longest_copy = 0;
    for (const auto& row_match : matches) {
        any_ngram    = any_ngram || !row_match.tokens.empty();
        longest_copy = std::max(longest_copy, static_cast<std::uint32_t>(row_match.tokens.size()));
    }
    std::uint32_t maximum_frontier = 0;
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        const std::uint32_t lane = lanes[row];
        if (lane >= max_concurrency ||
            std::find(lanes.begin(), lanes.begin() + static_cast<std::ptrdiff_t>(row), lane) !=
                lanes.begin() + static_cast<std::ptrdiff_t>(row)) {
            throw std::invalid_argument("DFlash batch contains an invalid or duplicate lane");
        }
        const SequenceState& sequence = active_sequence(lane);
        const RequestControl& request = requests[lane];
        if (request.lifecycle != Lifecycle::Active ||
            budgets[row].generated_tokens_remaining == 0 || !sequence.kv ||
            text_kv_addresses->bound_row(sequence.kv->text) < 0 ||
            (backend_kv_cache() && (!sequence.kv->backend ||
                                    backend_kv_addresses->bound_row(*sequence.kv->backend) < 0)) ||
            sequence.execution_frontier >= capacity ||
            sequence.text_kv_valid != sequence.execution_frontier ||
            sequence.dflash_context_frontier > sequence.execution_frontier ||
            sequence.execution_frontier - sequence.dflash_context_frontier > draft_window + 1U ||
            sequence.ledger_frontier != sequence.execution_frontier + 1 ||
            sequence.ledger.size() != sequence.ledger_frontier ||
            sequence.prefix_identity.size() != sequence.ledger_frontier ||
            sequence.prefix_digests.size() != sequence.ledger_frontier) {
            throw std::logic_error("DFlash batch row is not decode-ready");
        }
        maximum_frontier = std::max(maximum_frontier, sequence.execution_frontier);
    }
    // Every round verifies at its family's own window, on the frame viewed at that width for any
    // batch size. A round with at least one copy proposal replays the narrowest ngram family that
    // holds its longest copy; an all-neural round replays the tree family its batch size selects,
    // or else the neural family (both run the drafter). In a batch>1 ngram round the drafter also
    // runs and the per-row copy payload overlays it on the device, so a row without a copy keeps
    // its neural proposal (extent neural_draft_window) instead of decoding one token. A tree round
    // verifies a draft tree for every row whose whole main chain fits (extent K and room for every
    // column); other rows verify their chain prefix. Automatic mode lets the controller pick the
    // width of every all-neural round from its batch size and longest context. Grammar masks
    // follow one proposal chain, so a round with a constrained row verifies the chain.
    bool constrained = false;
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        grammar_dead_positions[row] = 0;
        constrained |= masks && masks->constrained(row);
    }
    const auto batch_rows    = static_cast<std::uint32_t>(lanes.size());
    std::uint32_t tree_nodes = 0;
    if (!any_ngram && !constrained && tree_controller) {
        const std::uint32_t columns = tree_controller->choose(batch_rows, maximum_frontier);
        tree_nodes                  = columns == neural_draft_window + 1U ? 0U : columns;
    } else if (!any_ngram && !constrained) {
        tree_nodes = tree_widths.fixed[batch_rows - 1U];
    }
    const bool tree_round = tree_nodes != 0U;
    SpeculativeRoundFamily& family =
        any_ngram    ? round_family(SpeculativeRoundKind::Ngram, longest_copy)
        : tree_round ? round_family(SpeculativeRoundKind::Tree, tree_nodes - 1U)
                     : round_family(SpeculativeRoundKind::Neural, neural_draft_window);
    const std::uint32_t verify_drafts = family.shape.verify_drafts;
    const bool drafter_runs           = !any_ngram || lanes.size() > 1;
    const bool dflash2_backend        = speculative_backend == SpeculativeBackend::DFlash2;
    qwen3_5::DFlashDecodeState& frame = *io.dflash_decode;
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        dflash_host_ingress->copy_rows[row] = any_ngram && !matches[row].tokens.empty() ? 1 : 0;
    }
    for (std::size_t row = 0; any_ngram && row < lanes.size(); ++row) {
        const std::uint32_t row_extent = static_cast<std::uint32_t>(matches[row].tokens.size());
        for (std::uint32_t step = 0; step < verify_drafts; ++step) {
            const auto token = step < row_extent ? matches[row].tokens[step] : 0;
            dflash_host_ingress->ngram_tokens[row * verify_drafts + step] = token;
            for (std::uint32_t slot = 0; slot < ops::kSparseSpeculativeCandidates; ++slot) {
                const auto cand = row * verify_drafts * ops::kSparseSpeculativeCandidates +
                                  step * ops::kSparseSpeculativeCandidates + slot;
                dflash_host_ingress->ngram_candidates[cand] =
                    (token + static_cast<TokenId>(slot)) %
                    parameters.model.resources().public_token_count;
                dflash_host_ingress->ngram_q[cand] = slot == 0 ? 1.0F : 0.0F;
            }
        }
    }
    const std::uint32_t width = verify_drafts + 1U;
    const auto row_extent     = [&](std::size_t row, std::uint32_t frontier) {
        const std::uint32_t max_by_budget = budgets[row].generated_tokens_remaining > 1
                                                ? budgets[row].generated_tokens_remaining - 1U
                                                : 0U;
        const bool row_copy               = any_ngram && !matches[row].tokens.empty();
        const std::uint32_t extent =
            std::min({row_copy ? static_cast<std::uint32_t>(matches[row].tokens.size())
                               : (drafter_runs ? neural_draft_window : 0U),
                      max_by_budget, capacity - frontier - 1U});
        // A constrained row verifies its chain: the constraint names one step per chain column.
        if (tree_round && extent == neural_draft_window &&
            capacity - frontier - 1U >= verify_drafts && constraint_plan(lanes[row]) == nullptr) {
            return verify_drafts;
        }
        return extent;
    };
    // Verification attention partitions each row by its full window, so the envelope covers the
    // window rather than the drafts a row actually carries.
    const auto target_envelope_for = [&](std::uint32_t max_execution_frontier) {
        return ops::CausalAttentionExecutionEnvelope{
            1, static_cast<std::uint32_t>(std::min<std::uint64_t>(
                   capacity, static_cast<std::uint64_t>(max_execution_frontier) + width))};
    };

    try {
        std::optional<nvtx::ScopedRange> submit_range;
        submit_range.emplace(nvtx::Name::DecodeDFlashSubmit, nvtx::Category::DFlash,
                             static_cast<std::uint64_t>(lanes.size()));
        DecodeGraphExecutable* forward       = nullptr;
        DecodeGraphExecutable* finish        = nullptr;
        execution::DFlashEnvelopes envelopes = dflash_envelopes(0, maximum_frontier);
        auto target_envelope                 = target_envelope_for(maximum_frontier);
        if (use_cuda_graph) {
            const auto batch = static_cast<std::uint32_t>(lanes.size());
            auto& profile    = family.forward.select(batch, maximum_frontier);
            forward          = &family.forward.install(profile);
            finish = &family.finish.install(family.finish.select(batch, maximum_frontier));
            envelopes =
                dflash_envelopes(profile.min_execution_frontier, profile.max_execution_frontier);
            target_envelope = target_envelope_for(profile.max_execution_frontier);
        }

        for (std::size_t row = 0; row < lanes.size(); ++row) {
            SequenceState& sequence           = active_sequence(lanes[row]);
            const RequestControl& request     = requests[lanes[row]];
            const std::uint32_t frontier      = sequence.execution_frontier;
            const std::uint32_t extent        = row_extent(row, frontier);
            const bool tree_row               = tree_round && extent == verify_drafts;
            dflash_host_ingress->anchors[row] = sequence.ledger.back();
            dflash_host_ingress->execution_frontiers[row] =
                checked_i32(frontier, "DFlash batch frontier");
            dflash_host_ingress->context_frontiers[row] =
                checked_i32(sequence.dflash_context_frontier, "DFlash context frontier");
            // DFlash2 drafts its whole block; DFlash drafts the columns the row verifies.
            dflash_host_ingress->proposal_valid_columns[row] = static_cast<std::int32_t>(
                (dflash2_backend ? neural_draft_window : std::min(extent, neural_draft_window)) +
                1U);
            dflash_host_ingress->proposal_extents[row]     = static_cast<std::int32_t>(extent);
            dflash_host_ingress->target_valid_columns[row] = static_cast<std::int32_t>(extent + 1U);
            for (std::uint32_t column = 0; column < width; ++column) {
                // A tree row's columns take the anchor's position plus their depth, which the
                // device adds once the round's tree is built.
                const std::uint32_t position =
                    frontier + (tree_row ? 0U : std::min(column, extent));
                dflash_host_ingress->target_rope_positions[row * width + column] =
                    checked_i32(position, "DFlash target RoPE position") + sequence.rope_delta;
            }
            dflash_host_ingress->text_kv_table_rows[row] =
                text_kv_addresses->bound_row(sequence.kv->text);
            dflash_host_ingress->dflash_kv_table_rows[row] =
                sequence.kv->backend ? backend_kv_addresses->bound_row(*sequence.kv->backend) : 0;
            dflash_host_ingress->active_lanes[row]       = static_cast<std::int32_t>(sequence.lane);
            const StateImageSelectors selectors          = state_selectors(sequence);
            dflash_host_ingress->state_source_slots[row] = selectors.source;
            dflash_host_ingress->state_destination_slots[row] = selectors.destination;
            dflash_host_ingress->sampling[row]                = request.sampling_host;
            dflash_host_ingress->sampling[row].mask           = bind_grammar_mask(masks, row);
            ensure_sequence_kv_mapped(sequence, frontier + extent + 1U,
                                      backend_kv_cache() ? frontier : 0U);
        }

        std::array<ConstraintRow, kMaximumConcurrency> constraint_rows{};
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            constraint_rows[row] = constraint_row(lanes[row]);
        }
        stage_constraint_columns(
            std::span<const ConstraintRow>(constraint_rows.data(), lanes.size()), width);

        execution::DFlashBatchContext schedule_state{
            {device, parameters, work, *state_images,
             round_replay_records(verify_drafts), io, prefill_hidden, prefill_chunk,
             proposal_head, prompt_attention},
            decoder->text_kv,
            *dflash,
            frame,
            *dflash_host_ingress,
            *dflash_host_egress,
            state_images->continuation_hidden_store(),
            dflash_draft_handoff->tokens(),
            dflash_draft_handoff->ready};

        schedule_state.ngram                  = any_ngram;
        schedule_state.neural_proposal_drafts = neural_draft_window;
        schedule_state.tree                   = tree_round;
        schedule_state.tree_paths             = draft_tree_paths;
        mark_workspace_usage(workspace_plan.dflash_round);
        const auto batch = static_cast<std::int32_t>(lanes.size());
        execution::dflash_decode_batch(schedule_state, batch, verify_drafts, envelopes,
                                       target_envelope, forward,
                                       execution::SpeculativePhase::Forward);
        if (constrained) {
            // The forward graph signals draft readiness before running the target model. The
            // handoff carries the verified drafts of every row, copy or neural, at the round's
            // width.
            timing.begin_constraint_wait();
            dflash_draft_handoff->ready.synchronize();
            timing.end_constraint_wait();
            timing.resume_submit();
            for (std::size_t row = 0; row < lanes.size(); ++row) {
                const auto extent =
                    static_cast<std::size_t>(dflash_host_ingress->proposal_extents[row]);
                (void)fill_grammar_mask(
                    masks, row, schedule_state.host_drafts.subspan(row * verify_drafts, extent));
            }
        }
        execution::dflash_decode_batch(schedule_state, batch, verify_drafts, envelopes,
                                       target_envelope, finish,
                                       execution::SpeculativePhase::Finish);
        submit_range.reset();
        timing.begin_wait();
        {
            nvtx::ScopedRange wait_range(nvtx::Name::DecodeDFlashWait, nvtx::Category::Control,
                                         static_cast<std::uint64_t>(lanes.size()));
            device.synchronize();
        }
        timing.end_wait();

        const double seconds = std::chrono::duration<double>(Clock::now() - started).count();
        const double share   = seconds / static_cast<double>(lanes.size());
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            SequenceState& sequence       = active_sequence(lanes[row]);
            RequestControl& request       = requests[lanes[row]];
            const std::uint32_t base_E    = sequence.execution_frontier;
            const std::uint32_t base_S    = sequence.ledger_frontier;
            const std::int32_t count_i    = dflash_host_egress->licensed_counts[row];
            const std::int32_t accepted_i = dflash_host_egress->accepted_drafts[row];
            const std::uint32_t extent =
                static_cast<std::uint32_t>(dflash_host_ingress->proposal_extents[row]);
            if (count_i <= 0 || count_i > static_cast<std::int32_t>(width) || accepted_i < 0 ||
                accepted_i + 1 != count_i || accepted_i > static_cast<std::int32_t>(extent) ||
                static_cast<std::uint32_t>(count_i) > budgets[row].generated_tokens_remaining ||
                static_cast<std::uint64_t>(base_E) + static_cast<std::uint32_t>(count_i) >
                    capacity) {
                throw std::runtime_error("DFlash batch returned invalid row metadata");
            }
            const std::span<const TokenId> row_tokens(dflash_host_egress->licensed_tokens.data() +
                                                          row * width,
                                                      static_cast<std::size_t>(count_i));
            validate_licensed_tokens(row_tokens);
            if (const ConstraintRow& constrained = constraint_rows[row];
                constrained.plan != nullptr) {
                // Chain row: licensed token i was chosen at verification column i.
                request.constraint_round = read_constraint_draws(
                    *constrained.plan, constrained.first_step, row * width, row_tokens);
            }
            if (extent == 0) {
                request.speculative_stats.fallback_steps += 1;
            } else {
                request.speculative_stats.rounds += 1;
                request.speculative_stats.drafted_tokens += extent;
                request.speculative_stats.accepted_tokens += static_cast<std::uint32_t>(accepted_i);
                for (std::int32_t i = 0; i < accepted_i; ++i) {
                    request.speculative_stats.accepted_per_position[static_cast<std::size_t>(i)] +=
                        1;
                }
            }
            if (any_ngram && !matches[row].tokens.empty()) {
                request.speculative_stats.ngram_rounds += 1;
                request.speculative_stats.ngram_drafted_tokens += extent;
                request.speculative_stats.ngram_accepted_tokens += accepted_i;
                if (matches[row].archived) {
                    request.speculative_stats.ngram_archive_rounds += 1;
                    request.speculative_stats.ngram_archive_drafted_tokens += extent;
                    request.speculative_stats.ngram_archive_accepted_tokens += accepted_i;
                }
            }
            if (tree_round && extent == verify_drafts) {
                if (tree_controller) {
                    tree_controller->observe_tree_row(
                        std::span<const std::int32_t>(dflash_host_egress->accepted_path.data() +
                                                          row * width,
                                                      static_cast<std::size_t>(accepted_i) + 1U),
                        width);
                }
                request.speculative_stats.tree_rounds += 1;
                const std::int32_t branch = dflash_host_egress->accepted_branch[row];
                if (branch > 0) {
                    if (branch > accepted_i) {
                        throw std::runtime_error("DFlash batch returned invalid row metadata");
                    }
                    request.speculative_stats.tree_side_rounds += 1;
                    request.speculative_stats.tree_side_accepted_tokens +=
                        static_cast<std::uint32_t>(accepted_i - branch + 1);
                }
            }
            sequence.dflash_context_frontier = base_E;
            request.pending                  = PendingCandidate{
                                 .kind          = PendingKind::Speculative,
                                 .base_E        = base_E,
                                 .base_S        = base_S,
                                 .prompt_tokens = 0,
                                 .produced      = static_cast<std::uint32_t>(count_i),
                                 .verify_drafts = verify_drafts,
            };
            request.lifecycle = Lifecycle::Pending;
            request.timings.decode_seconds += seconds;
            request.timings.decode_share_seconds += share;
        }
        if (tree_controller && !any_ngram) {
            tree_controller->observe_round(batch_rows, maximum_frontier, width, seconds);
        }
        return runtime::BatchedGeneratedRound{
            .tokens     = std::span<const TokenId>(dflash_host_egress->licensed_tokens.data(),
                                                   lanes.size() * width),
            .row_counts = std::span<const std::int32_t>(dflash_host_egress->licensed_counts.data(),
                                                        lanes.size()),
            .row_stride = width,
            .timing     = timing.finish(),
        };
    } catch (...) {
        timing.begin_wait();
        try {
            nvtx::ScopedRange wait_range(nvtx::Name::DecodeDFlashWait, nvtx::Category::Control,
                                         static_cast<std::uint64_t>(lanes.size()));
            device.synchronize();
        } catch (...) {}
        timing.end_wait();
        clear_execution_failure_lanes(lanes);
        throw;
    }
}

runtime::BatchedGeneratedRound ProgramImpl::decode_raw(
    std::span<const std::uint32_t> lanes, std::span<const runtime::RoundBudget> budgets,
    runtime::ExecutionTiming* failed_timing, runtime::TokenMaskProvider* masks) {
    if (speculative_backend == SpeculativeBackend::None) {
        return decode_ordinary_batch(lanes, budgets, failed_timing, masks);
    }
    if (speculative_backend == SpeculativeBackend::Mtp) {
        return decode_mtp_batch(lanes, budgets, failed_timing, masks);
    }
    return decode_dflash_batch(lanes, budgets, failed_timing, masks);
}

runtime::ExecutionTiming ProgramImpl::resolve_non_speculative_pending(
    SequenceState& sequence, RequestControl& request, std::uint32_t accepted_tokens, bool terminal,
    std::optional<std::uint32_t> prefix_execution_split_after,
    runtime::ExecutionTiming* failed_timing) {
    runtime::ExecutionTimingRecorder timing(runtime::ExecutionTimingPhase::Post, failed_timing);
    if (request.lifecycle != Lifecycle::Pending) {
        throw std::logic_error("pending resolution requires a pending generated round");
    }
    if ((request.pending.kind != PendingKind::Begin &&
         request.pending.kind != PendingKind::Ordinary) ||
        request.pending.produced != 1 || accepted_tokens != 1) {
        throw std::logic_error("non-speculative pending round must commit its single token");
    }

    const std::uint32_t base_ledger_frontier = request.pending.kind == PendingKind::Begin
                                                   ? request.pending.prompt_tokens
                                                   : request.pending.base_S;
    commit_generated_prefix_identity(
        sequence, base_ledger_frontier,
        std::span<const TokenId>(sequence.ledger).subspan(base_ledger_frontier, accepted_tokens),
        prefix_execution_split_after);

    switch (request.pending.kind) {
    case PendingKind::Begin:
        sequence.execution_frontier = request.pending.prompt_tokens;
        sequence.ledger_frontier    = request.pending.prompt_tokens + 1;
        break;
    case PendingKind::Ordinary:
        sequence.execution_frontier = request.pending.base_E + request.pending.produced;
        sequence.ledger_frontier    = request.pending.base_S + request.pending.produced;
        break;
    case PendingKind::Speculative:
    case PendingKind::None:
        throw std::logic_error("non-speculative pending round has an invalid kind");
    }
    if (sequence.ledger_frontier != sequence.execution_frontier + 1 ||
        sequence.ledger.size() != sequence.ledger_frontier ||
        sequence.prefix_identity.size() != sequence.ledger_frontier ||
        sequence.prefix_digests.size() != sequence.ledger_frontier) {
        throw std::logic_error("resolved round did not establish a valid frontier");
    }
    // Begin publishes a sampled token but does not execute it through the target. An exact-hit
    // Fork therefore still names an immutable read source and an unwritten destination here; the
    // first state-mutating decode commit closes it. A suffix prefill already closed its Fork at
    // the committed prefill frontier.
    if (request.pending.kind == PendingKind::Begin && terminal && sequence.state.fork_pending) {
        const StateImageSelectors selectors = state_selectors(sequence);
        timing.resume_submit();
        state_images->copy_slot(selectors.source, selectors.destination, device.stream);
        timing.begin_wait();
        device.synchronize();
        timing.end_wait();
        settle_state_fork(sequence);
    } else if (request.pending.kind == PendingKind::Ordinary) {
        settle_state_fork(sequence);
    }
    trim_sequence_kv(sequence, sequence.text_kv_valid, backend_kv_valid(sequence));
    if (terminal) { sequence.mtp_draft_count = 0; }
    request.lifecycle = terminal ? Lifecycle::Finishable : Lifecycle::Active;
    request.pending   = {};
    return timing.finish();
}


} // namespace infernix::models::qwen3_5::detail
